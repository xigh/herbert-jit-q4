//! ```text
//! herbert-jit-llama-serve [MODEL_DIR | org/name[@revision]] [options]
//! herbert-jit-llama-serve service install|uninstall|start|stop|status [--name NAME] [MODEL [options]]
//! ```
//!
//! MODEL_DIR: a model's directory, or a repository of Hugging Face already in its cache
//! (`src/hub.rs`, downloaded with `hf download`). `service`: the server as a systemd user unit, a
//! LaunchAgent or a Windows service (`src/service.rs`). It listens on 127.0.0.1:8000 by default
//! (`--host 0.0.0.0` for every interface).
//!
//! Gemma 4 (E2B, E4B, 12B, 31B, 26B-A4B) and Qwen3.5 (dense) on the CPU, every kernel emitted at
//! run time, behind llama-server's HTTP API, one slot: herbert-serve's server (the routes, the load
//! and unload, the queue, the prefix reused, the answer streamed with its tool calls and thinking),
//! on this crate's engine (`serve/llama.rs`). Built with `--features serve`.
//!
//! The port opens first; the model loads after (MODEL_DIR), or on `POST /models/load {"model":
//! DIR}`, and leaves on `POST /models/unload` (the memory given back, the server still up).
//!
//! Gemma 4 keeps the prefix a request shares with the previous one in its K/V cache, and with
//! `--side-slot 1` one more conversation aside (two snapshots of the whole context in
//! memory: a request that shares a longer prefix with it swaps the two); `--prefix-cache DIR`
//! writes the K/V of a request's stable prefix (system messages, tools) to DIR and reads it back,
//! this run or the next (the files are tied to the executable, the machine's instructions, the code
//! of the generated programs, the tuning, the model, the capacity); `--keep-warm SECS` runs one
//! token, a block of two rows and a read of every weight page after SECS without a request, so that
//! the system neither compresses nor evicts the idle server's memory. Its end tokens are those of
//! `config.json` with the family's.
//!
//! Qwen3.5 (`qwen3_5`, `qwen3_5_moe`) starts a request from the live state when the prompt extends
//! it, else from the longest snapshot of the DeltaNet states its prompt extends (`--snapshots`,
//! taken before the last messages of a prompt and at the end of each answer), each lane keeping a
//! conversation's K/V (`--lanes`), else from an empty conversation. LiquidAI's LFM2 and LFM2.5
//! (`lfm2`) and their mixtures of experts (`lfm2_moe`: LFM2-8B-A1B, LFM2-24B-A2B) the same way: the
//! state of their short convolutions does not go back either. LFM3 (`lfm3`: LFM3-1.7B) as well, its
//! sliding layers keeping their whole K/V.
//!
//! The prompt is fed by pieces of one prefill pass: a client that goes away stops it between two of
//! them.
//!
//! Embedding and reranking models, each an encoder in one generated program, `/v1/embeddings` or
//! `/v1/rerank` and no chat route: EmbeddingGemma 2 (`embedding_gemma2`: text and images,
//! `engine/embed_gemma.rs`), XLM-RoBERTa (`xlm-roberta`: BAAI/bge-m3, BAAI/bge-reranker-v2-m3,
//! `engine/xlmr.rs`); their projections in Q4 by default (`Q4_0`'s grid), bf16 under `--tune
//! emb.weights=16`; the inputs of a request in blocks of `--ctx` rows. A Qwen3 embedding model
//! or reranker takes the same Q4 default. `HERBERT_ENCODER_TIMES=1` prints the times of each block,
//! `HERBERT_ENCODER_PROFILE=1` (x86-64) the encoder's time by operation.
//!
//! Gemma 4 E2B, E4B and 12B read images (x86-64, AArch64): the vision tower (12B: its encoder-free
//! embedder) in emitted kernels, mounted at the first image, its soft tokens fed as one block
//! (`engine/vision.rs`). `HERBERT_VISION_TIMES=1` prints the times of an image,
//! `HERBERT_JIT_FEATS=FILE` writes its soft tokens in f32, `HERBERT_VISION_PROFILE=1` (x86-64) the
//! tower's time by operation; `HERBERT_VISION_ATTN_ROWS=N` sets the queries per block of the
//! tower's attention (8; 0: row by row).
//!
//! They hear recordings (`input_audio`): the audio tower in emitted kernels, mounted at the first
//! recording, its soft tokens fed as causal positions by blocks of 280 (`engine/audio.rs`).
//! `HERBERT_AUDIO_TIMES=1` prints the times of a recording, `HERBERT_JIT_FEATS=FILE` writes its
//! soft tokens, `HERBERT_AUDIO_PROFILE=1` (x86-64) the tower's time by operation.
//!
//! ```text
//!   --threads N          workers (default: the performance cores; `HERBERT_THREADS`)
//!   --int8               int8 weights and activations for the projections
//!   --q4                 4-bit weights (groups of 32, f32 scales, signed max / -8: the grid of
//!                        Q4_0) with int8 activations
//!   --tune emb.weights=16|4  an embedding or reranking model (EmbeddingGemma 2, bge-m3,
//!                        bge-reranker-v2-m3, Qwen3 Embedding and Reranker): its projections in
//!                        bf16 (`16`) or in Q4 (`4`, the default; `--int8` without `emb.weights`
//!                        keeps int8); a chat model is not concerned
//!   --tune emb.q4grid=8|7  an encoder's Q4 grid: `Q4_0`'s `signed max / -8` (8, the default) or
//!                        `absmax / 7` (7)
//!   --q4-grid7           with --q4: scale = absmax / 7 (codes in ±7) instead of the grid of Q4_0
//!   --q4-grid8           with --q4: the grid of Q4_0, the default (accepted for older scripts)
//!   --q4-head-int8       with --q4: the language head stays int8
//!   --q4-sse42           with --q4: the SSE4.2-only micro-kernel (x86-64, a comparison: the engine
//!                        itself needs AVX2 and FMA)
//!   --amx                AArch64: the bf16 projections of a block on Apple's AMX (the bits of the
//!                        vector kernel)
//!   --prefill-workers N  participants of the prefill (default: the whole pool)
//!   --decode-workers N   participants of the decode (default: the whole pool)
//!   --pool-qos C         macOS: the QoS class of the pool's threads and of participant 0
//!                        (interactive, initiated, default, utility, background; none: inherited,
//!                        the default)
//!   --prefill-executor spmd-spin|spmd-hybrid  the barriers of the prefill: spin, or spin then wait
//!                        (HERBERT_HYBRID_WAIT: yield, the default on macOS; condvar elsewhere;
//!                        ulock); default: spmd-hybrid on Apple Silicon, spmd-spin elsewhere
//!   --decode-executor spmd-spin|spmd-hybrid  the same for the decode
//!   --hybrid-spin N      pauses a hybrid participant spins before it waits (default 256 on Apple
//!                        Silicon, 512 elsewhere)
//!   --task-grain N       AArch64: a node's tasks taken by ticket in chunks of tasks /
//!                        (participants · N); 0 = static
//!   --task-guided        AArch64: the chunk recomputed from the remaining tasks at every grab
//!                        (--no-task-guided)
//!   --attn-kv-task, --no-attn-kv-task  x86-64: the prefill attention by tasks of one KV head,
//!                        its query heads in turn on one participant, or one task per query head
//!                        (default)
//!   --moe-row-prefill    a mixture-of-experts model prefills one row at a time (x86-64 groups the
//!                        rows by expert)
//!   --prefill-pass N     rows of one prefill pass (default 256; 1024 on the 26B-A4B: more rows per
//!                        expert)
//!   --deltanet-state bf16|f32  Qwen: the recurrent state rounded to bf16 each time it is stored
//!                        (default, as the K/V cache), or kept in f32
//!   --saved-states 0|1   the former name of `--side-slot`: 1 keeps one more conversation aside
//!   --snapshots N        Qwen: DeltaNet snapshots kept for the prefix cache (default 12; the conv
//!                        and recurrent states of every DeltaNet layer each, allocated on first
//!                        use)
//!   --lanes N            Qwen: K/V caches of `--ctx` positions each (default 2): two conversations
//!                        alternating on the server (a client's side requests between its turns)
//!                        each keep their prefix
//! ```
//!
//! and herbert-serve's options (`--host`, `--port`, `--ctx`, `--alias`, `--log-dir`,
//! `--prefix-cache`, `--prefix-cache-files`, `--keep-warm`, `--side-slot`, `--presence-penalty`).
//!
//! On stderr per request: the prompt, the prefix reused, the speeds, the end.

use herbert_chat::Family;
use herbert_jit_q4::engine::{Engine, Measure, Tuning};
use herbert_jit_q4::serve::llama::Cpu;
use herbert_serve::{Loaded, Loader, ServeOptions};
use std::path::Path;

/// The engine's options the command line sets (`--ctx` is herbert-serve's).
struct Jit {
    threads: usize,
    tuning: Tuning,
    /// `--saved-states 1`: a side state even without `--side-slot`.
    saved_states: bool,
    /// Qwen: `--snapshots`, `--lanes`.
    snapshots: usize,
    lanes: usize,
    /// `--tune emb.weights=16|4`: the projections of an embedding or reranking model in 16 bits or
    /// in Q4; `None`, the default: Q4, unless `--q4` or `--int8` says otherwise.
    emb_weights: Option<EmbWeights>,
    /// `--tune emb.q4grid=7`: an encoder's Q4 on the grid `absmax / 7` (the chat's `--q4-grid7`)
    /// instead of `Q4_0`'s `signed max / -8` (`emb.q4grid=8`, the default, as the chat's
    /// `--q4`).
    emb_grid7: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum EmbWeights {
    Q4,
    Bf16,
}

impl Jit {
    /// The tuning of an embedding or reranking model: Q4 by default (`--tune emb.weights=4`), bf16
    /// under `--tune emb.weights=16`; without `emb.weights`, `--q4` and `--int8` as given, Q4
    /// when neither is and the CPU has the kernel.
    fn encoder_tuning(&self) -> Tuning {
        let mut t = self.tuning.clone();
        let n = &mut t.kernels.neon;
        match self.emb_weights {
            Some(EmbWeights::Q4) => (n.w_int8, n.w_q4) = (true, true),
            Some(EmbWeights::Bf16) => (n.w_int8, n.w_q4) = (false, false),
            // The default Q4 only where the CPU has the kernel (no RISC-V, no AArch64 without
            // dotprod): bf16 elsewhere, as an explicit `--q4` would be refused there.
            None if !n.w_int8 && herbert_jit_q4::jit::int8::available() => {
                (n.w_int8, n.w_q4) = (true, true)
            }
            None => {}
        }
        // Q4: `Q4_0`'s grid unless `emb.q4grid=7` (Ryzen 9 9900X, bge-m3: cosine to f32 0.939 →
        // 0.951, bge-reranker-v2-m3: mean logit gap 0.59 → 0.44, EmbeddingGemma 2 0.979 →
        // 0.983, Qwen3-Embedding 0.929 → 0.928; the same speed).
        if n.w_q4 {
            n.q4_grid8 = !self.emb_grid7;
        }
        // AArch64: the 16-bit projections of a block on Apple's AMX where the machine has it, the
        // bits of the vector kernel (`--amx`; `HERBERT_ENCODER_AMX=0` leaves them on the
        // vector kernel). bge-m3 in 16 bits on the M3 Ultra: 6.9 → 12.0 extracts/s, the
        // same cosines.
        if cfg!(target_arch = "aarch64")
            && std::env::var("HERBERT_ENCODER_AMX").map_or(true, |v| v != "0")
        {
            t.kernels.gemm_amx = true;
        }
        t
    }
}

/// `eos_token_id` of `config.json` (at its root or in `text_config`), one id or a list.
fn config_eos(dir: &Path) -> Vec<u32> {
    let v: serde_json::Value = std::fs::read(dir.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    match v
        .get("eos_token_id")
        .or_else(|| v.pointer("/text_config/eos_token_id"))
    {
        Some(serde_json::Value::Array(a)) => a
            .iter()
            .filter_map(serde_json::Value::as_u64)
            .map(|n| n as u32)
            .collect(),
        Some(n) => n.as_u64().map(|n| vec![n as u32]).unwrap_or_default(),
        None => Vec::new(),
    }
}

impl Loader for Jit {
    fn build_info(&self) -> String {
        format!(
            "herbert-jit {} (CPU, emitted kernels)",
            env!("CARGO_PKG_VERSION")
        )
    }

    fn load(
        &self,
        model_dir: &Path,
        alias: Option<String>,
        so: &ServeOptions,
    ) -> Result<Loaded, String> {
        let dir = herbert_jit_q4::model::loader::resolve_model_dir(model_dir)
            .map_err(|e| e.to_string())?;
        // EmbeddingGemma 2 (`embedding_gemma2`): its encoder, `/v1/embeddings` only (text, images);
        // `--ctx` caps one input (8 192 positions at most).
        if herbert_jit_q4::engine::embed_gemma::detect(&dir) {
            let engine = herbert_jit_q4::engine::embed_gemma::EmbedGemma::load(
                &dir,
                self.threads,
                so.ctx,
                &self.encoder_tuning(),
            )
            .map_err(|e| e.to_string())?;
            eprintln!(
                "[engine] {}: EmbeddingGemma 2 loaded in {:.1} s, {} positions per input, vectors of {}",
                herbert_jit_q4::core::isa::summary(),
                engine.load_seconds,
                engine.capacity(),
                engine.embedding_dim()
            );
            let desc = format!(
                "threads {}, {}",
                self.threads,
                if engine.quantized {
                    "quantized"
                } else {
                    "bf16"
                }
            );
            return herbert_serve::loaded(
                engine,
                &dir,
                alias,
                so,
                self.build_info(),
                "herbert-jit",
                desc,
            );
        }
        // XLM-RoBERTa (bge-m3, bge-reranker-v2-m3): an encoder, `/v1/embeddings` or `/v1/rerank`,
        // no chat; `--ctx` caps the tokens of one input (8 192 at most, the model's
        // positions).
        if herbert_jit_q4::engine::xlmr::is_xlmr(&dir) {
            let engine = herbert_jit_q4::engine::xlmr::XlmrEngine::load(
                &dir,
                self.threads,
                so.ctx,
                &self.encoder_tuning(),
            )
            .map_err(|e| e.to_string())?;
            eprintln!(
                "[engine] {}: XLM-RoBERTa ({:?}) loaded in {:.1} s, {} tokens per input",
                herbert_jit_q4::core::isa::summary(),
                engine.c.kind,
                engine.load_seconds,
                engine.capacity()
            );
            let desc = format!(
                "threads {}, {:?}, {}",
                self.threads,
                engine.c.kind,
                if engine.quantized {
                    "quantized"
                } else {
                    "bf16"
                }
            );
            return herbert_serve::loaded(
                engine,
                &dir,
                alias,
                so,
                self.build_info(),
                "herbert-jit",
                desc,
            );
        }
        let family = Family::detect(&dir);
        // A Qwen3 embedding model or reranker: Q4 by default as the other encoders; a chat model
        // keeps the tuning.
        let head = herbert_serve::qwen3::head(&dir)?;
        let tuning = if matches!(head, herbert_serve::qwen3::Head::Lm) {
            self.tuning.clone()
        } else {
            // An input in one prefill pass (`--ctx` rows at most), the norms, residual additions
            // and the chain around the attention on the whole team, as Qwen2.5
            // (`Engine::load`); a pass given on the command line is kept.
            // `HERBERT_ENCODER_PASS=N` and `HERBERT_ENCODER_TEAMS=0|1` (measurement).
            let mut t = self.encoder_tuning();
            let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok());
            if t.kernels.prefill_pass == herbert_jit_q4::jit::schedule::PREFILL_PASS_DEFAULT {
                t.kernels.prefill_pass = env("HERBERT_ENCODER_PASS")
                    .unwrap_or(so.ctx.clamp(1, herbert_jit_q4::jit::DEFAULT_MAX_BATCH));
            }
            let teams = env("HERBERT_ENCODER_TEAMS").unwrap_or(1) != 0;
            (t.kernels.elem_team, t.kernels.attn_chain_team) = (teams, teams);
            eprintln!(
                "[tuning] Qwen3 encoder: prefill pass {}, teams {teams}",
                t.kernels.prefill_pass
            );
            t
        };
        let engine = Engine::load(&dir, self.threads, so.ctx, &tuning, &Measure::none())
            .map_err(|e| e.to_string())?;
        eprintln!(
            "[engine] {}: loaded in {:.1} s, mounted in {:.1} s, n_ctx {}",
            herbert_jit_q4::core::isa::summary(),
            engine.load_seconds,
            engine.mount_seconds,
            engine.reserved()
        );
        let gemma = family == Family::Gemma;
        let mut cpu = Cpu::new(
            engine,
            gemma,
            if gemma { config_eos(&dir) } else { Vec::new() },
        );
        cpu.head = head;
        eprintln!("[engine] {}", cpu.head.describe());
        if so.side_slot > 0 || self.saved_states {
            if gemma {
                let bytes = cpu.side_enable();
                eprintln!(
                    "[engine] side state: the whole context, {:.2} GiB of memory",
                    bytes as f64 / (1u64 << 30) as f64
                );
            } else {
                eprintln!("[serve] --side-slot: Gemma 4 only, ignored for Qwen3.5 (its lanes and snapshots keep the prefixes)");
            }
        }
        if !gemma {
            let (one, lanes) = cpu.snapshots_and_lanes(self.snapshots, self.lanes);
            eprintln!(
                "[engine] {} lanes ({:.2} GiB for the K/V copies), {} snapshots of {:.1} MiB (allocated on first use)",
                self.lanes.max(1),
                lanes as f64 / (1u64 << 30) as f64,
                self.snapshots,
                one as f64 / (1u64 << 20) as f64
            );
        }
        let desc = format!("threads {}", self.threads);
        herbert_serve::loaded(cpu, &dir, alias, so, self.build_info(), "herbert-jit", desc)
    }
}

fn main() {
    if let Some(code) = herbert_jit_q4::core::isa::refuse_if_below_floor() {
        std::process::exit(code);
    }
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("service") => {
            if let Err(e) = herbert_jit_q4::service::command(&args[1..]) {
                eprintln!("error: {e}\nusage: {}", herbert_jit_q4::service::USAGE);
                std::process::exit(1);
            }
            return;
        }
        #[cfg(windows)]
        Some("service-run") => {
            if let Err(e) = herbert_jit_q4::service::run_service(&args[1..], serve) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
            return;
        }
        // A model of Hugging Face (`org/name[@revision]`, not a directory that exists): its
        // snapshot in the Hub's cache, where `hf download` put it (`hub`), or an error before the
        // port opens. `POST /models/load` takes directories only. Its repository is the name the
        // API reports, unless --alias says another.
        Some(model) if !model.starts_with("--") => match herbert_jit_q4::hub::model_dir(model) {
            Ok(dir) => {
                if let Some(alias) = herbert_jit_q4::hub::alias_for(model)
                    .filter(|_| !args.iter().any(|a| a == "--alias"))
                {
                    args.extend(["--alias".to_string(), alias]);
                }
                args[0] = dir.to_string_lossy().into_owned();
            }
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        },
        _ => {}
    }
    serve(args);
}

fn serve(args: Vec<String>) {
    let usage = format!(
        "herbert-jit-llama-serve [MODEL_DIR | org/name[@revision]] [--threads N] [--int8 | --q4 [--q4-grid7] [--q4-head-int8] [--q4-sse42]] [--prefill-workers N] [--decode-workers N] [--pool-qos C] [--prefill-executor E] [--decode-executor E] [--hybrid-spin N] [--task-grain N] [--task-guided] [--attn-kv-task] [--moe-row-prefill] [--amx] [--prefill-pass N] [--deltanet-state bf16|f32] [--saved-states 0|1] [--snapshots N] [--lanes N] [--tune emb.weights=16|4,emb.q4grid=8|7] {}",
        herbert_serve::USAGE
    );
    let threads = std::env::var("HERBERT_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(herbert_jit_q4::core::cpu_detection::get_performance_cpu_count);
    let engine = std::cell::RefCell::new(Some(Jit {
        threads,
        tuning: Tuning::default(),
        saved_states: false,
        snapshots: 12,
        lanes: 2,
        emb_weights: None,
        emb_grid7: false,
    }));
    herbert_serve::run_with(
        args,
        &usage,
        |flag, value| {
            let mut o = engine.borrow_mut();
            let o = o.as_mut().unwrap();
            let num = |v: String| v.parse::<usize>().map_err(|e| format!("{flag}: {e}"));
            match flag {
                "--threads" => o.threads = num(value()?)?,
                "--int8" => o.tuning.kernels.neon.w_int8 = true,
                "--q4" => (o.tuning.kernels.neon.w_int8, o.tuning.kernels.neon.w_q4) = (true, true),
                "--q4-sse42" => {
                    (
                        o.tuning.kernels.neon.w_int8,
                        o.tuning.kernels.neon.w_q4,
                        o.tuning.kernels.neon.w_sse42,
                    ) = (true, true, true)
                }
                "--q4-grid8" => o.tuning.kernels.neon.q4_grid8 = true,
                "--q4-grid7" => o.tuning.kernels.neon.q4_grid8 = false,
                "--q4-head-int8" => o.tuning.kernels.neon.q4_head_int8 = true,
                "--moe-row-prefill" => o.tuning.moe_row_prefill = true,
                "--amx" => o.tuning.kernels.gemm_amx = true,
                "--prefill-pass" => o.tuning.kernels.prefill_pass = num(value()?)?,
                "--attn-kv-task" => o.tuning.kernels.attn_kv_task = true,
                "--no-attn-kv-task" => o.tuning.kernels.attn_kv_task = false,
                "--prefill-workers" => o.tuning.prefill_workers = Some(num(value()?)?),
                "--decode-workers" => o.tuning.decode_workers = Some(num(value()?)?),
                "--pool-qos" => {
                    let v = value()?;
                    o.tuning.pool.qos = if v == "none" {
                        None
                    } else {
                        Some(herbert_jit_q4::pool::settings::Qos::parse(&v).ok_or("--pool-qos: interactive, initiated, default, utility, background or none")?)
                    }
                }
                "--prefill-executor" => {
                    o.tuning.prefill_executor =
                        value()?.parse().map_err(|e| format!("{flag}: {e}"))?
                }
                "--decode-executor" => {
                    o.tuning.decode_executor =
                        value()?.parse().map_err(|e| format!("{flag}: {e}"))?
                }
                "--hybrid-spin" => {
                    o.tuning.hybrid_spin =
                        u32::try_from(num(value()?)?).map_err(|e| format!("{flag}: {e}"))?
                }
                "--task-grain" => o.tuning.kernels.task_grain = num(value()?)?,
                "--task-guided" => o.tuning.kernels.task_guided = true,
                "--no-task-guided" => o.tuning.kernels.task_guided = false,
                "--deltanet-state" => {
                    o.tuning.kernels.deltanet_state_f32 = match value()?.as_str() {
                        "bf16" => false,
                        "f32" => true,
                        _ => return Err("--deltanet-state: bf16 or f32".into()),
                    }
                }
                "--saved-states" => match num(value()?)? {
                    0 => o.saved_states = false,
                    1 => o.saved_states = true,
                    _ => return Err("--saved-states: 0 or 1".into()),
                },
                "--snapshots" => o.snapshots = num(value()?)?,
                "--lanes" => o.lanes = num(value()?)?.max(1),
                "--tune" => {
                    for kv in value()?.split(',') {
                        match kv.trim() {
                            "emb.weights=16" | "emb.weights=bf16" => o.emb_weights = Some(EmbWeights::Bf16),
                            "emb.weights=4" | "emb.weights=q4" => o.emb_weights = Some(EmbWeights::Q4),
                            "emb.q4grid=7" => o.emb_grid7 = true,
                            "emb.q4grid=8" => o.emb_grid7 = false,
                            other => return Err(format!("--tune {other}: this engine takes emb.weights=16|4, emb.q4grid=8|7")),
                        }
                    }
                }
                _ => return Ok(false),
            }
            Ok(true)
        },
        |_| Ok(engine.take().unwrap()),
    );
}
