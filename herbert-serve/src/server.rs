//! llama-server's HTTP API, one slot, in front of whichever engine [`Loader`] gives: `POST
//! /v1/chat/completions` (streamed as SSE or not, tools, `chat_template_kwargs.enable_thinking`,
//! `timings_per_token`, `stream_options.include_usage`), `GET /props`, `GET /v1/models`, `GET
//! /health`, `POST /apply-template`, `POST /tokenize`, `POST /models/load`, `POST /models/unload`.
//! The engine is left to the binary.
//!
//! The port opens first; the model loads after (MODEL_DIR), or on `POST /models/load {"model":
//! DIR}`, and leaves on `POST /models/unload` (the device's memory given back, the server still
//! up). While it loads, `/health` answers 503 (`"status": "loading"`) and `/props` gives
//! `model_state` and `loading` (stage, progress from 0 to 1, seconds), from
//! `herbert_chat::progress`, which the engine's load moves.

use crate::backend::Backend;
use crate::prefix::{fnv64, PrefixCache};
use crate::worker::{error_body, Job, Out, Worker, WorkerOptions};
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use herbert_chat::{progress, Dialect, Family, Template};
use serde_json::{json, Value};
use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

/// The options every engine's server takes.
pub struct ServeOptions {
    /// `--host H`: the address listened on (default 127.0.0.1, this machine only; `--host 0.0.0.0`
    /// for every interface).
    pub host: String,
    /// `--port P`: default 8000.
    pub port: u16,
    /// `--ctx N`: the positions of the context, the `n_ctx` of /props (default 32768).
    pub ctx: usize,
    /// `--alias NAME`: the model name the API reports (default: the directory's name).
    pub alias: Option<String>,
    /// `--log-dir DIR`: per request, the request, the rendered prompt, the raw output with its
    /// special tokens, the response.
    pub log_dir: Option<PathBuf>,
    /// `--prefix-cache DIR`, `--prefix-cache-files N` (default 8).
    pub prefix_dir: Option<PathBuf>,
    pub prefix_files: usize,
    /// `--keep-warm SECS` (default 0: never).
    pub keep_warm: Option<std::time::Duration>,
    /// `--side-slot 1`: one conversation kept aside (default 0: none). Any N above 0 is one: the
    /// engine keeps a single side state.
    pub side_slot: usize,
    /// `--presence-penalty P`: the presence penalty of a request that sends none (default 0).
    pub presence_penalty: f32,
}

/// The usage of the options of [`ServeOptions`], for an engine's usage line.
pub const USAGE: &str = "[--host H] [--port P] [--ctx N] [--alias NAME] [--log-dir DIR] [--prefix-cache DIR] [--prefix-cache-files N] [--keep-warm SECS] [--side-slot 0|1] [--presence-penalty P]";

/// A model on the device, ready to serve: its worker (run on the thread that loaded it), its
/// template, tokenizer and `/props`.
pub struct Loaded {
    pub template: Arc<Template>,
    pub tokenizer: Arc<Tokenizer>,
    pub family: Family,
    pub props: Value,
    pub model: String,
    pub run: Box<dyn FnOnce(std::sync::mpsc::Receiver<Job>)>,
}

/// The bytes of a request body at most: a few images of [`crate::image::MAX_BYTES`] in base64.
pub const BODY_LIMIT: usize = 128 << 20;

/// What a binary gives the server: its engine.
pub trait Loader: Send + Sync + 'static {
    /// `/props`'s `build_info`: the crate, its version, the device's API.
    fn build_info(&self) -> String;
    /// Loads the checkpoint of `dir` with [`loaded`]; `alias` names the model when given.
    fn load(&self, dir: &Path, alias: Option<String>, o: &ServeOptions) -> Result<Loaded, String>;
}

/// The template of a model that has none: rendering it is an error, which a chat request gets as a
/// 400.
pub const NO_CHAT_TEMPLATE: &str = "{{ raise_exception('this model has no chat template: it serves /v1/embeddings or /v1/rerank') }}";

/// The model of `dir` on `engine`, ready to serve: the tokenizer, the template and the family's
/// tokens read from `dir`, `/props` written, the prefix cache opened when `--prefix-cache` was
/// given and the engine can write its K/V (`engine_desc`: what else the K/V depends on, such as a
/// subgroup forced), the worker built.
pub fn loaded<B: Backend>(
    engine: B,
    dir: &Path,
    alias: Option<String>,
    o: &ServeOptions,
    build_info: String,
    fingerprint: &'static str,
    engine_desc: String,
) -> Result<Loaded, String> {
    let family = Family::detect(dir);
    let model = alias.unwrap_or_else(|| {
        dir.file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| family.name().into())
    });
    let tokenizer = Arc::new(
        Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| format!("tokenizer.json: {e}"))?,
    );
    // A model without a chat template (an embedding model, a cross-encoder reranker) still loads:
    // its template refuses every chat request, and `/v1/embeddings` or `/v1/rerank` is what it
    // serves.
    let template = Arc::new(match Template::load(dir, family) {
        Ok(t) => t,
        Err(e) => {
            eprintln!(
                "[serve] {}: {e}: no chat route; /v1/embeddings and /v1/rerank only",
                dir.display()
            );
            Template::from_source(NO_CHAT_TEMPLATE.into(), family).map_err(|e| e.to_string())?
        }
    });
    let dialect = Dialect::new(family, &tokenizer, &engine.eos()).map_err(|e| e.to_string())?;
    let n_ctx = engine.capacity();
    let wants = |given: bool, what: &str, can: bool| {
        if given && !can {
            eprintln!("[serve] {what}: not with this engine and this model, ignored");
        }
        given && can
    };
    let prefix = match (
        wants(
            o.prefix_dir.is_some(),
            "--prefix-cache",
            engine.kv_fingerprint().is_some(),
        ),
        &o.prefix_dir,
        engine.kv_fingerprint(),
    ) {
        (true, Some(d), Some(desc)) => {
            // What the K/V of a prefix depends on besides the engine's description: the template,
            // the tokenizer, the weights (names, sizes, times), what the engine adds.
            let mut files: Vec<String> = std::fs::read_dir(dir)
                .map(|rd| {
                    rd.flatten()
                        .filter(|e| e.file_name().to_string_lossy().ends_with(".safetensors"))
                        .map(|e| {
                            let m = e.metadata().ok();
                            let t = m
                                .as_ref()
                                .and_then(|m| m.modified().ok())
                                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                                .map_or(0, |d| d.as_secs());
                            format!(
                                "{} {} {t}",
                                e.file_name().to_string_lossy(),
                                m.map_or(0, |m| m.len())
                            )
                        })
                        .collect()
                })
                .unwrap_or_default();
            files.sort();
            let fnv = |b: &[u8]| format!("{:016x}", fnv64(b));
            let extra = format!(
                "template {}; tokenizer {}; weights {files:?}; {engine_desc}",
                fnv(template.source.as_bytes()),
                std::fs::read(dir.join("tokenizer.json"))
                    .map(|b| fnv(&b))
                    .unwrap_or_default(),
            );
            match PrefixCache::new(
                d,
                o.prefix_files,
                format!("prefix cache format 1; {desc}; {extra}"),
            ) {
                Ok(pc) => Some(pc),
                Err(e) => {
                    eprintln!("[prefix-cache] {}: {e}: no prefix cache", d.display());
                    None
                }
            }
        }
        _ => None,
    };
    let (bos, eos) = family.bos_eos();
    let props = json!({
        "build_info": build_info,
        "model_path": dir.display().to_string(),
        "model_alias": model,
        "model_state": "ready",
        "total_slots": 1,
        "default_generation_settings": { "n_ctx": n_ctx, "params": { "temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05, "presence_penalty": o.presence_penalty, "frequency_penalty": 0.0 } },
        "chat_template": template.source,
        "bos_token": bos,
        "eos_token": eos,
        "modalities": { "vision": engine.vision().is_some(), "audio": engine.audio().is_some() },
        "embedding_dim": engine.embedding_dim(),
        // What `/v1/embeddings` takes beside text (EmbeddingGemma 2: images and recordings).
        "embedding_media": { "image": engine.embed_media_kinds().0, "audio": engine.embed_media_kinds().1 },
    });
    let wo = WorkerOptions {
        log_dir: o.log_dir.clone(),
        keep_warm: o.keep_warm,
        prefix,
        presence_default: o.presence_penalty,
        fingerprint,
    };
    let mut worker = Worker::new(
        engine,
        tokenizer.clone(),
        template.clone(),
        dialect,
        model.clone(),
        wo,
    );
    if let crate::qwen3::Head::Reranker(r) = crate::qwen3::head(dir)? {
        worker.reranker = Some(r);
    }
    Ok(Loaded {
        template,
        tokenizer,
        family,
        props,
        model,
        run: Box::new(move |rx| worker.run(rx)),
    })
}

/// The command line: `[MODEL_DIR]`, then the options of [`ServeOptions`] and the engine's own,
/// which `engine_flag` takes (`Ok(false)`: not one of its own). On a bad line, the error and
/// `usage` on stderr, exit status 1.
pub fn run<L: Loader>(
    usage: &str,
    engine_flag: impl FnMut(&str, &mut dyn FnMut() -> Result<String, String>) -> Result<bool, String>,
    make: impl FnOnce(&ServeOptions) -> Result<L, String>,
) {
    run_with(std::env::args().skip(1).collect(), usage, engine_flag, make)
}

/// [`run`] on a command line given (without the program's name): the engine may have rewritten it
/// first (a model downloaded, a service's arguments).
pub fn run_with<L: Loader>(
    args: Vec<String>,
    usage: &str,
    engine_flag: impl FnMut(&str, &mut dyn FnMut() -> Result<String, String>) -> Result<bool, String>,
    make: impl FnOnce(&ServeOptions) -> Result<L, String>,
) {
    if let Err(e) = start(&args, engine_flag, make) {
        eprintln!("error: {e}\nusage: {usage}");
        std::process::exit(1);
    }
}

fn start<L: Loader>(
    args: &[String],
    mut engine_flag: impl FnMut(
        &str,
        &mut dyn FnMut() -> Result<String, String>,
    ) -> Result<bool, String>,
    make: impl FnOnce(&ServeOptions) -> Result<L, String>,
) -> Result<(), String> {
    let mut it = args.iter().peekable();
    // The model is optional: without it the server starts empty and waits for `POST /models/load`.
    let model_dir = it.next_if(|a| !a.starts_with("--")).map(PathBuf::from);
    let mut o = ServeOptions {
        host: "127.0.0.1".into(),
        port: 8000,
        ctx: 32768,
        alias: None,
        log_dir: None,
        prefix_dir: None,
        prefix_files: 8,
        keep_warm: None,
        side_slot: 0,
        presence_penalty: 0.0,
    };
    while let Some(flag) = it.next() {
        let mut value = || it.next().cloned().ok_or(format!("{flag} needs a value"));
        let num =
            |v: String| -> Result<u64, String> { v.parse().map_err(|e| format!("{flag}: {e}")) };
        match flag.as_str() {
            "--host" => o.host = value()?,
            "--port" => {
                o.port = u16::try_from(num(value()?)?).map_err(|_| format!("{flag}: 0 to 65535"))?
            }
            "--ctx" => o.ctx = num(value()?)? as usize,
            "--alias" => o.alias = Some(value()?),
            "--log-dir" => o.log_dir = Some(PathBuf::from(value()?)),
            "--prefix-cache" => o.prefix_dir = Some(PathBuf::from(value()?)),
            "--prefix-cache-files" => o.prefix_files = num(value()?)? as usize,
            "--side-slot" => o.side_slot = num(value()?)? as usize,
            "--keep-warm" => {
                o.keep_warm =
                    Some(std::time::Duration::from_secs(num(value()?)?)).filter(|d| !d.is_zero())
            }
            "--presence-penalty" => {
                o.presence_penalty = value()?.parse().map_err(|e| format!("{flag}: {e}"))?
            }
            other => {
                if !engine_flag(other, &mut value)? {
                    return Err(format!("unknown option {other}"));
                }
            }
        }
    }
    if let Some(d) = &o.log_dir {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let loader = make(&o)?;
    let app = Arc::new(App {
        next: AtomicU64::new(1),
        model: Mutex::new(Model::Empty),
        thread: Mutex::new(None),
        loader: Box::new(loader),
        o,
    });
    let router = Router::new()
        .route("/health", get(health))
        .route("/props", get(props_route))
        .route("/models/load", post(load_route))
        .route("/models/unload", post(unload_route))
        .route("/v1/models", get(models))
        .route("/models", get(models))
        .route("/apply-template", post(apply_template))
        .route("/tokenize", post(tokenize))
        .route("/v1/chat/completions", post(chat))
        .route("/chat/completions", post(chat))
        // `prompt` instead of `messages`: the raw text completed (the answer has the shape of a
        // chat completion).
        .route("/completion", post(chat))
        .route("/v1/completions", post(chat))
        // OpenAI's embeddings and llama-server's rerank (Jina's shape): one answer each, the
        // engine's state used then reset.
        .route("/v1/embeddings", post(embeddings))
        .route("/embeddings", post(embeddings))
        .route("/v1/rerank", post(rerank))
        .route("/rerank", post(rerank))
        // A request carries its images as base64 (`image.rs`): axum's 2 MiB would refuse a photo.
        .layer(axum::extract::DefaultBodyLimit::max(BODY_LIMIT))
        .with_state(app.clone());
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    rt.block_on(async move {
        // The port first: /props and /health answer while the model loads.
        let (host, port) = (app.o.host.clone(), app.o.port);
        let listener = tokio::net::TcpListener::bind((host.as_str(), port))
            .await
            .map_err(|e| format!("bind {host}:{port}: {e}"))?;
        eprintln!("[serve] listening on http://{host}:{port}");
        if let Some(dir) = model_dir {
            let alias = app.o.alias.clone();
            start_load(&app, dir, alias)?;
        }
        axum::serve(listener, router)
            .await
            .map_err(|e| e.to_string())
    })
}

/// A model being served: its worker's queue (taken to unload) and what the routes read.
struct Ready {
    jobs: Mutex<Option<std::sync::mpsc::Sender<Job>>>,
    template: Arc<Template>,
    tokenizer: Arc<Tokenizer>,
    family: Family,
    props: Value,
    model: String,
}

enum Model {
    Empty,
    Loading { dir: PathBuf },
    Ready(Arc<Ready>),
    Failed { dir: PathBuf, error: String },
}

struct App {
    next: AtomicU64,
    model: Mutex<Model>,
    /// The engine's thread: joined on unload, once the device memory is given back.
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    loader: Box<dyn Loader>,
    o: ServeOptions,
}

impl App {
    fn ready(&self) -> Option<Arc<Ready>> {
        match &*self.model.lock().unwrap() {
            Model::Ready(l) => Some(l.clone()),
            _ => None,
        }
    }

    /// Why a request cannot run: no model, a model loading, a load that failed.
    fn not_ready(&self) -> Response {
        let message = match &*self.model.lock().unwrap() {
            Model::Loading { .. } => "Loading model".to_string(),
            Model::Failed { error, .. } => format!("the model failed to load: {error}"),
            _ => "no model loaded".to_string(),
        };
        error(503, error_body(503, "unavailable_error", &message))
    }
}

/// Loads `dir` on a thread of its own, which then runs the model's worker. The model before it, if
/// any, is unloaded first.
fn start_load(app: &Arc<App>, dir: PathBuf, alias: Option<String>) -> Result<(), String> {
    {
        let mut model = app.model.lock().unwrap();
        if let Model::Loading { dir: other } = &*model {
            return Err(format!("{} is loading", other.display()));
        }
        progress::begin();
        let previous = std::mem::replace(&mut *model, Model::Loading { dir: dir.clone() });
        // The thread taken under the model's lock, with the model it serves.
        let thread = app.thread.lock().unwrap().take();
        drop(model);
        if let Model::Ready(ready) = previous {
            unload(&ready, thread);
        }
    }
    let shared = app.clone();
    let app = app.clone();
    let dir_failed = dir.clone();
    // The model's lock held until the thread is stored: the thread cannot mark its model ready, nor
    // an unload take that model, before its handle is where the unload looks for it.
    let mut model = shared.model.lock().unwrap();
    let spawned = std::thread::Builder::new()
        .name("engine".into())
        .spawn(move || {
            let (tx, rx) = std::sync::mpsc::channel::<Job>();
            // A panic of the loader or of the worker leaves the model failed, with its message,
            // instead of loading or ready for good.
            let loaded = catch(|| app.loader.load(&dir, alias, &app.o)).and_then(|r| r);
            match loaded {
                Ok(l) => {
                    let ready = Arc::new(Ready {
                        jobs: Mutex::new(Some(tx)),
                        template: l.template,
                        tokenizer: l.tokenizer,
                        family: l.family,
                        props: l.props,
                        model: l.model,
                    });
                    *app.model.lock().unwrap() = Model::Ready(ready.clone());
                    let run = l.run;
                    match catch(move || run(rx)) {
                        Ok(()) => eprintln!("[serve] {}: unloaded", dir.display()),
                        Err(error) => {
                            eprintln!("[serve] {}: {error}", dir.display());
                            let mut model = app.model.lock().unwrap();
                            // Only the model this thread served: an unload may have replaced it.
                            if matches!(&*model, Model::Ready(r) if Arc::ptr_eq(r, &ready)) {
                                *model = Model::Failed { dir, error };
                            }
                        }
                    }
                }
                Err(error) => {
                    eprintln!("[serve] {}: {error}", dir.display());
                    *app.model.lock().unwrap() = Model::Failed { dir, error };
                }
            }
        })
        .map_err(|e| format!("the engine's thread: {e}"));
    match spawned {
        Ok(thread) => *shared.thread.lock().unwrap() = Some(thread),
        Err(error) => {
            *model = Model::Failed {
                dir: dir_failed,
                error: error.clone(),
            };
            return Err(error);
        }
    }
    drop(model);
    Ok(())
}

/// `f`, a panic turned into its message.
fn catch<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|p| {
        let what = p
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| p.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "no message".into());
        format!("the engine panicked: {what}")
    })
}

/// The worker's queue closed: it ends after its current request, and the engine, dropped with it,
/// frees the memory. `thread` is the one that served `ready`, taken under the model's lock with it:
/// a load started since has a thread of its own, which this does not wait for.
fn unload(ready: &Ready, thread: Option<std::thread::JoinHandle<()>>) {
    ready.jobs.lock().unwrap().take();
    if let Some(thread) = thread {
        let _ = thread.join();
    }
}

/// The load in progress: its stage, the fraction done and the seconds spent.
fn loading(dir: &Path) -> Value {
    let p = progress::snapshot();
    json!({ "model": dir.display().to_string(), "stage": p.stage, "progress": p.fraction, "seconds": p.seconds })
}

/// `ok` once the model answers; 503 before, as llama-server while it loads, with how far the load
/// is.
async fn health(State(app): State<Arc<App>>) -> Response {
    let body = match &*app.model.lock().unwrap() {
        Model::Ready(_) => return Json(json!({ "status": "ok" })).into_response(),
        Model::Loading { dir } => json!({ "status": "loading", "loading": loading(dir) }),
        Model::Failed { dir, error } => {
            json!({ "status": "error", "model": dir.display().to_string(), "error": error })
        }
        Model::Empty => json!({ "status": "no_model" }),
    };
    (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response()
}

/// llama-server's `/props` once the model is ready; before, `model_state` (`empty`, `loading`,
/// `error`) and, while it loads, `loading` (stage, progress from 0 to 1, seconds).
async fn props_route(State(app): State<Arc<App>>) -> Json<Value> {
    let build_info = app.loader.build_info();
    Json(match &*app.model.lock().unwrap() {
        Model::Ready(l) => l.props.clone(),
        Model::Loading { dir } => {
            json!({ "build_info": build_info, "model_state": "loading", "loading": loading(dir) })
        }
        Model::Failed { dir, error } => {
            json!({ "build_info": build_info, "model_state": "error", "model_path": dir.display().to_string(), "error": error })
        }
        Model::Empty => json!({ "build_info": build_info, "model_state": "empty" }),
    })
}

/// `{"model": "<directory>", "alias": "<name>"}`: load it, the current model unloaded first.
/// Answers at once (202); `/props` follows the load.
async fn load_route(State(app): State<Arc<App>>, Json(body): Json<Value>) -> Response {
    let Some(dir) = body["model"].as_str().filter(|d| !d.is_empty()) else {
        return error(
            400,
            error_body(
                400,
                "invalid_request_error",
                "\"model\": the model's directory is required",
            ),
        );
    };
    // `--alias` names the model of the command line only: another model has its own name.
    let alias = body["alias"].as_str().map(str::to_owned);
    let (app2, dir) = (app.clone(), PathBuf::from(dir));
    // Unloading the previous model joins its thread: off the async workers.
    match tokio::task::spawn_blocking(move || start_load(&app2, dir, alias)).await {
        Ok(Ok(())) => (StatusCode::ACCEPTED, props_route(State(app)).await).into_response(),
        Ok(Err(e)) => error(409, error_body(409, "invalid_request_error", &e)),
        Err(e) => error(500, error_body(500, "server_error", &e.to_string())),
    }
}

/// The model leaves the device (and the host's memory): the server stays, empty, until the next
/// `/models/load`.
async fn unload_route(State(app): State<Arc<App>>) -> Response {
    let previous = {
        let mut model = app.model.lock().unwrap();
        // A load cannot be cut short: it goes on.
        if matches!(&*model, Model::Loading { .. }) {
            return error(
                409,
                error_body(409, "invalid_request_error", "a model is loading"),
            );
        }
        let previous = std::mem::replace(&mut *model, Model::Empty);
        (previous, app.thread.lock().unwrap().take())
    };
    if let (Model::Ready(ready), thread) = previous {
        let _ = tokio::task::spawn_blocking(move || unload(&ready, thread)).await;
    }
    Json(json!({ "model_state": "empty" })).into_response()
}

async fn models(State(app): State<Arc<App>>) -> Json<Value> {
    let data: Vec<Value> = app
        .ready()
        .map(|l| json!({ "id": l.model, "object": "model", "owned_by": "herbert" }))
        .into_iter()
        .collect();
    Json(json!({ "object": "list", "data": data }))
}

fn error(code: u16, body: Value) -> Response {
    (
        StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        Json(body),
    )
        .into_response()
}

async fn apply_template(State(app): State<Arc<App>>, Json(body): Json<Value>) -> Response {
    let Some(ready) = app.ready() else {
        return app.not_ready();
    };
    // The family's default, as a chat request renders it.
    let thinking = body
        .pointer("/chat_template_kwargs/enable_thinking")
        .and_then(Value::as_bool)
        .unwrap_or(ready.family.thinks_by_default());
    match ready
        .template
        .render(&body["messages"], body.get("tools"), thinking, true)
    {
        Ok(prompt) => Json(json!({ "prompt": prompt })).into_response(),
        Err(e) => error(
            400,
            error_body(400, "invalid_request_error", &e.to_string()),
        ),
    }
}

async fn tokenize(State(app): State<Arc<App>>, Json(body): Json<Value>) -> Response {
    let Some(ready) = app.ready() else {
        return app.not_ready();
    };
    let content = body["content"].as_str().unwrap_or("");
    let special = body["add_special"].as_bool().unwrap_or(false);
    match ready.tokenizer.encode(content, special) {
        Ok(enc) => Json(json!({ "tokens": enc.get_ids() })).into_response(),
        Err(e) => error(
            400,
            error_body(400, "invalid_request_error", &e.to_string()),
        ),
    }
}

async fn embeddings(State(app): State<Arc<App>>, Json(body): Json<Value>) -> Response {
    routed(app, body, "embeddings").await
}

async fn rerank(State(app): State<Arc<App>>, Json(body): Json<Value>) -> Response {
    routed(app, body, "rerank").await
}

/// The body marked with its route for the worker; a body that is not a JSON object is a 400.
async fn routed(app: Arc<App>, mut body: Value, route: &str) -> Response {
    let Some(fields) = body.as_object_mut() else {
        return error(
            400,
            error_body(
                400,
                "invalid_request_error",
                "the body must be a JSON object",
            ),
        );
    };
    fields.insert("__route".into(), json!(route));
    one_answer(app, body).await
}

/// A request answered by one message (no stream), through the worker like a chat.
async fn one_answer(app: Arc<App>, body: Value) -> Response {
    let Some(ready) = app.ready() else {
        return app.not_ready();
    };
    let id = app.next.fetch_add(1, Ordering::Relaxed);
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Out>(4);
    let sent = ready
        .jobs
        .lock()
        .unwrap()
        .as_ref()
        .map(|j| j.send(Job { id, body, tx }).is_ok())
        .unwrap_or(false);
    if !sent {
        return app.not_ready();
    }
    match rx.recv().await {
        Some(Out::Message(v)) => Json(v).into_response(),
        Some(Out::Error(code, body)) => error(code, body),
        _ => error(
            500,
            error_body(500, "server_error", "no answer from the engine"),
        ),
    }
}

async fn chat(State(app): State<Arc<App>>, Json(body): Json<Value>) -> Response {
    let stream = body["stream"].as_bool().unwrap_or(false);
    let Some(ready) = app.ready() else {
        return app.not_ready();
    };
    let id = app.next.fetch_add(1, Ordering::Relaxed);
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Out>(256);
    let sent = ready
        .jobs
        .lock()
        .unwrap()
        .as_ref()
        .map(|j| j.send(Job { id, body, tx }).is_ok())
        .unwrap_or(false);
    if !sent {
        return app.not_ready();
    }
    if !stream {
        return match rx.recv().await {
            Some(Out::Message(v)) => Json(v).into_response(),
            Some(Out::Error(code, body)) => error(code, body),
            _ => error(
                500,
                error_body(500, "server_error", "no answer from the engine"),
            ),
        };
    }
    // A request refused before the first chunk is an HTTP error, as llama-server does.
    let first = match rx.recv().await {
        Some(Out::Error(code, body)) => return error(code, body),
        Some(o) => o,
        None => {
            return error(
                500,
                error_body(500, "server_error", "no answer from the engine"),
            )
        }
    };
    let events = tokio_stream::once(first)
        .chain(ReceiverStream::new(rx))
        .map(|o| {
            Ok::<Event, Infallible>(match o {
                Out::Chunk(v) | Out::Message(v) => Event::default().data(v.to_string()),
                Out::Done => Event::default().data("[DONE]"),
                Out::Error(_, body) => Event::default().data(body.to_string()),
            })
        });
    let mut r = Sse::new(events).into_response();
    r.headers_mut()
        .insert(header::CACHE_CONTROL, "no-cache".parse().unwrap());
    r
}
