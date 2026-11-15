//! The thread that owns the engine and answers the chat requests one at a time (one slot): the
//! template over the request, the prefix it shares with what the engine holds reused, the rest fed
//! by the engine's prefill blocks, then sampling up to an end of turn, a tool call or the budget,
//! the answer split as it streams (`herbert_chat::Parser`) and sent as OpenAI chunks or one
//! message.
//!
//! What a request reuses depends on the engine's state:
//!
//! - one that cuts back to any position (K/V caches, [`Backend::can_truncate`]): the longest prefix
//!   shared with the previous request, cut to [`Backend::block_align`]. With a side state
//!   (`--side-slot`), a request that shares a longer prefix with the state kept aside swaps the
//!   two, and one that would discard most of the live state saves it first. With `--prefix-cache
//!   DIR` (`prefix.rs`), the K/V of the request's stable prefix (the system messages and the tools)
//!   is read from DIR instead of computed when a file holds it (the newest one is loaded at start),
//!   else computed apart and written there;
//! - one that cannot (DeltaNet's): the live state when the prompt extends the tokens it holds (a
//!   client sending the conversation back as it was generated), else the longest snapshot the
//!   prompt extends whose lane still holds its positions, else an empty conversation in the lane
//!   used least recently. An engine that keeps snapshots ([`Backend::snapshot_slots`]) takes one
//!   before each of the last [`BOUNDARIES`] messages of a prompt (cut down to
//!   [`Backend::block_align`]) and one at the end of each answer: a client re-rendering the
//!   conversation (the template drops the previous turns' thinking, rewrites a tool call) shares
//!   the prompt up to the start of the last assistant message, and the snapshot taken there serves.
//!   Without snapshots, the live state or an empty conversation.
//!
//! An image (an `image_url` part, `image.rs`) takes the place of its `<|image|>` placeholder as the
//! processor writes it: `<|image>`, one position per soft token, `<image|>`. The tokens the server
//! keeps for those positions are not the placeholder's but keys made from the image's pixels
//! (`media_key`, above every token of a vocabulary): what is reused from one request to the next
//! is the same image at the same place, never another one; and a prefix reused is never cut
//! inside an image, which goes through the tower whole. Past an image, the cut to the query tiles
//! counts from the image's end (`reuse_point`): the text after it is fed from there.
//!
//! A recording (an `input_audio` part, `audio.rs`) takes the place of its `<|audio|>` placeholder
//! the same way: `<|audio>`, one position per soft token (keys made from its samples), `<audio|>`;
//! its soft tokens are causal, fed whole by the engine's tower ([`Backend::prefill_audio`]).
//!
//! A client that goes away (the channel of its answer closed) stops the request between two prefill
//! blocks or two tokens: what was fed stays, the next request finds it.
//!
//! Logs, on stderr per request: the prompt, the prefix reused, the speeds and the end; with
//! `--log-dir`, the request, the prompt, the raw output (special tokens included) and the response
//! of each request in files of their own.

use crate::audio::{self, Recording};
use crate::backend::{Audio, Backend};
use crate::image::{self, Image};
use crate::prefix::PrefixCache;
use herbert_chat::sampler::{Sampler, Sampling, PENALTY_LAST_N};
use herbert_chat::{Dialect, Piece, Template};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
use tokio::sync::mpsc::Sender;

/// What the worker sends back for one request.
pub enum Out {
    /// One SSE chunk (`data: …`).
    Chunk(Value),
    /// The whole answer of a request that does not stream.
    Message(Value),
    /// The end of a stream (`data: [DONE]`).
    Done,
    /// An HTTP status and the error body (llama-server's shape).
    Error(u16, Value),
}

pub struct Job {
    pub id: u64,
    pub body: Value,
    pub tx: Sender<Out>,
}

/// Snapshots taken in a prompt: before its last `BOUNDARIES` messages.
pub const BOUNDARIES: usize = 3;

/// A snapshot of the engine: its slot, the lane whose K/V cache it goes with, the tokens it stands
/// for, when it last served.
struct Snap {
    slot: usize,
    lane: usize,
    tokens: Vec<u32>,
    used: u64,
}

/// Which medium of the request a span of soft tokens is: the image or the recording of that index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Image(usize),
    Audio(usize),
}

/// The soft tokens of one medium in a prompt: where they start, how many, which image or recording
/// of the request.
struct Span {
    start: usize,
    len: usize,
    kind: Kind,
    /// The RoPE positions the medium takes: `len`, but `max(rows, columns)` of a Qwen image's
    /// merged grid (its soft tokens turn by `(t, h, w)`, `get_rope_index`); the text after it
    /// is shifted back by the difference ([`Backend::set_rope_shift`]).
    rope: usize,
}

/// The RoPE offset of the text at `p`: the positions the media ending at or before `p` saved.
fn rope_shift_at(p: usize, spans: &[Span]) -> usize {
    spans
        .iter()
        .filter(|s| s.start + s.len <= p)
        .map(|s| s.len - s.rope)
        .sum()
}

/// The token the server keeps for the soft token `i` of an image or a recording whose pixels or
/// samples hash to `digest`: above every token of a vocabulary (bit 31 set), different from one
/// medium to the next.
fn media_key(digest: u64, i: usize) -> u32 {
    let mut z = digest ^ (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    0x8000_0000 | (((z ^ (z >> 31)) as u32) & 0x7fff_ffff)
}

/// `p`, or the start of the image (or recording) it falls inside: a prefix reused ends before it or
/// after it.
fn outside_images(p: usize, spans: &[Span]) -> usize {
    spans
        .iter()
        .find(|s| s.start < p && p < s.start + s.len)
        .map_or(p, |s| s.start)
}

/// Where a prefix of `p` tokens is reused from: cut to a multiple of `align` counted from the end
/// of the last image before it (the text after an image is fed from there the first time, so its
/// blocks start on the same tiles), never inside an image. Counted from 0, the cut fell inside the
/// image whenever it ended past a multiple of `align` (with `align` 32, for example): the whole
/// image went through the tower again.
fn reuse_point(p: usize, align: usize, spans: &[Span]) -> usize {
    let seg = spans
        .iter()
        .map(|s| s.start + s.len)
        .filter(|&end| end <= p)
        .max()
        .unwrap_or(0);
    outside_images(p - (p - seg) % align, spans)
}

/// Each `<|image|>` placeholder of `ids` replaced by its image's group (`<|image>`, a key per soft
/// token, `<image|>`), each `<|audio|>` placeholder by its recording's (`<|audio>`, a key per soft
/// token, `<audio|>`), the groups' tokens (`image_tokens`, `audio_tokens`) the tokenizer's: the
/// tokens and where the soft tokens of each medium are. Qwen's template writes the image's opening
/// and closing tokens itself (`<|vision_start|><|image_pad|><|vision_end|>`): its `<|image_pad|>`
/// is replaced by the keys alone (`image_tokens` without the two). `image_count`: the soft tokens
/// of an image and the RoPE positions they take.
#[allow(clippy::type_complexity)]
fn expand_media(
    ids: Vec<u32>,
    image_tokens: Option<(Option<u32>, u32, Option<u32>)>,
    images: &[Image],
    image_count: &dyn Fn(&Image) -> Result<(usize, usize), String>,
    audio_tokens: Option<(u32, u32, u32)>,
    recordings: &[Recording],
    a: Option<Audio>,
) -> Result<(Vec<u32>, Vec<Span>), String> {
    let img = if images.is_empty() {
        None
    } else {
        Some(image_tokens.ok_or("the tokenizer has no <|image|> token")?)
    };
    let aud = if recordings.is_empty() {
        None
    } else {
        Some(audio_tokens.ok_or("the tokenizer has no <|audio|> token")?)
    };
    let count = |p: Option<u32>| p.map_or(0, |p| ids.iter().filter(|&&x| x == p).count());
    if count(img.map(|t| t.1)) != images.len() {
        return Err(format!(
            "{} images for {} <|image|> placeholders in the prompt",
            images.len(),
            count(img.map(|t| t.1))
        ));
    }
    if count(aud.map(|t| t.1)) != recordings.len() {
        return Err(format!(
            "{} recordings for {} <|audio|> placeholders in the prompt",
            recordings.len(),
            count(aud.map(|t| t.1))
        ));
    }
    let (mut out, mut spans) = (Vec::with_capacity(ids.len()), Vec::new());
    let (mut ni, mut na) = (0, 0);
    for t in ids {
        let (open, close, soft, rope, digest, kind) = match (img, aud) {
            (Some((boi, p, eoi)), _) if t == p => {
                let im = &images[ni];
                let (soft, rope) = image_count(im).map_err(|e| format!("image {ni}: {e}"))?;
                ni += 1;
                (boi, eoi, soft, rope, im.digest, Kind::Image(ni - 1))
            }
            (_, Some((boa, p, eoa))) if t == p => {
                let r = &recordings[na];
                let soft = a.ok_or("no audio tower")?.soft_tokens(r.wave.len());
                na += 1;
                (
                    Some(boa),
                    Some(eoa),
                    soft,
                    soft,
                    r.digest,
                    Kind::Audio(na - 1),
                )
            }
            _ => {
                out.push(t);
                continue;
            }
        };
        out.extend(open);
        spans.push(Span {
            start: out.len(),
            len: soft,
            kind,
            rope,
        });
        out.extend((0..soft).map(|i| media_key(digest, i)));
        out.extend(close);
    }
    Ok((out, spans))
}

/// A lane of the engine: the tokens whose K/V its cache holds, position by position, and when it
/// last served.
#[derive(Default)]
struct LaneState {
    kv: Vec<u32>,
    used: u64,
}

/// What a worker is given besides its engine.
pub struct WorkerOptions {
    pub log_dir: Option<PathBuf>,
    /// `--keep-warm SECS`: after this long without a request, [`Backend::keep_warm`], so that the
    /// system does not evict the buffers of a server that idles (Windows empties the working set
    /// of a server idle for ten minutes).
    pub keep_warm: Option<std::time::Duration>,
    /// `--prefix-cache DIR`: the K/V of the stable prefixes on disk.
    pub prefix: Option<PrefixCache>,
    /// `--presence-penalty`: the presence penalty of a request that sends none (llama-server's
    /// flag).
    pub presence_default: f32,
    /// The `system_fingerprint` of the answers: the engine's crate.
    pub fingerprint: &'static str,
}

pub struct Worker<B: Backend> {
    pub engine: B,
    pub tokenizer: Arc<Tokenizer>,
    pub template: Arc<Template>,
    pub dialect: Dialect,
    pub model: String,
    o: WorkerOptions,
    /// The tokens whose state the engine holds, in order.
    history: Vec<u32>,
    /// The tokens of the state the engine keeps aside (`--side-slot`), when it keeps one: a
    /// request that shares a longer prefix with it than with the live state swaps the two; a
    /// request that would discard most of the live state saves it first. An agent's short side
    /// requests and its conversation thus each keep their prefix.
    aside: Vec<u32>,
    /// A state that cannot be cut back: its snapshots, the engine's lanes, a clock for the least
    /// recently used.
    snaps: Vec<Snap>,
    lanes: Vec<LaneState>,
    clock: u64,
    /// The token that opens a message (Qwen's `<|im_start|>`): where the snapshots of a prompt are
    /// taken.
    turn_start: Option<u32>,
    /// Gemma 4's `<|image>`, `<|image|>` and `<image|>`, when the tokenizer has them.
    image_tokens: Option<(Option<u32>, u32, Option<u32>)>,
    /// Gemma 4's `<|audio>`, `<|audio|>` and `<audio|>`, when the tokenizer has them.
    audio_tokens: Option<(u32, u32, u32)>,
    /// Qwen3-Reranker: the pair in its prompt ([`crate::qwen3::Reranker::prompt`]) instead of the
    /// tokenizer's.
    pub reranker: Option<crate::qwen3::Reranker>,
}

/// Fewer shared tokens than this are not worth a copy of the state.
const SIDE_MARGIN: usize = 64;

/// The fewest tokens of a stable prefix worth a file.
const PREFIX_MIN: usize = 64;

/// llama-server's `ignore_eos`: the logits with every end-of-generation token at −∞, and their
/// greatest token, so that the sampler picks the next best token instead of stopping.
fn without_stops(logits: &[u16], stops: &[u32]) -> (u32, Vec<u16>) {
    let mut l = logits.to_vec();
    let ninf = herbert_chat::half16::from_f32(f32::NEG_INFINITY);
    for &s in stops {
        if let Some(v) = l.get_mut(s as usize) {
            *v = ninf;
        }
    }
    let best = l
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| {
            let f = herbert_chat::half16::to_f32(v);
            if f > b.1 {
                (i, f)
            } else {
                b
            }
        });
    (best.0 as u32, l)
}

fn unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The tokens a reranker reads for (`query`, `doc`): by default the tokenizer's pair (`<s> query
/// </s></s> doc </s>`, the cross-encoders'); a reranker that reads a prompt (Qwen3-Reranker) gets
/// its own form here.
fn rerank_tokens(
    tok: &Tokenizer,
    reranker: Option<&crate::qwen3::Reranker>,
    query: &str,
    doc: &str,
) -> Result<Vec<u32>, String> {
    if let Some(r) = reranker {
        return r.prompt(tok, query, doc);
    }
    let e = tok
        .encode((query, doc), true)
        .map_err(|e| format!("tokenizer: {e}"))?;
    Ok(e.get_ids().to_vec())
}

/// `f(0..n)` in order, spread over the host's threads when there are several inputs: the tokenizer
/// of a request's inputs (each `encode` on one thread, the inputs side by side).
fn map_threads<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let threads = std::thread::available_parallelism()
        .map_or(1, |t| t.get())
        .min(n.div_ceil(4));
    if threads <= 1 {
        return (0..n).map(f).collect();
    }
    let per = n.div_ceil(threads);
    let f = &f;
    std::thread::scope(|s| {
        let parts: Vec<_> = (0..n)
            .step_by(per)
            .map(|a| s.spawn(move || (a..(a + per).min(n)).map(f).collect::<Vec<T>>()))
            .collect();
        parts
            .into_iter()
            .flat_map(|p| p.join().expect("a tokenizer thread"))
            .collect()
    })
}

pub fn error_body(code: u16, kind: &str, message: &str) -> Value {
    json!({ "error": { "code": code, "message": message, "type": kind } })
}

impl<B: Backend> Worker<B> {
    pub fn new(
        engine: B,
        tokenizer: Arc<Tokenizer>,
        template: Arc<Template>,
        dialect: Dialect,
        model: String,
        o: WorkerOptions,
    ) -> Worker<B> {
        let lanes = (0..engine.lanes()).map(|_| LaneState::default()).collect();
        let turn_start = tokenizer.token_to_id("<|im_start|>");
        let image_tokens = match (
            tokenizer.token_to_id("<|image>"),
            tokenizer.token_to_id("<|image|>"),
            tokenizer.token_to_id("<image|>"),
        ) {
            (Some(boi), Some(img), Some(eoi)) => Some((Some(boi), img, Some(eoi))),
            // LFM3-VL: `<|vision_start|><|image|><|vision_end|>` written by the template, the
            // placeholder alone replaced.
            (None, Some(img), None) if tokenizer.token_to_id("<|vision_start|>").is_some() => {
                Some((None, img, None))
            }
            // Qwen: the template writes `<|vision_start|>` and `<|vision_end|>` around the
            // placeholder.
            _ => tokenizer
                .token_to_id("<|image_pad|>")
                .map(|pad| (None, pad, None)),
        };
        let audio_tokens = match (
            tokenizer.token_to_id("<|audio>"),
            tokenizer.token_to_id("<|audio|>"),
            tokenizer.token_to_id("<audio|>"),
        ) {
            (Some(boa), Some(aud), Some(eoa)) => Some((boa, aud, eoa)),
            _ => None,
        };
        Worker {
            engine,
            tokenizer,
            template,
            dialect,
            model,
            o,
            history: Vec::new(),
            aside: Vec::new(),
            snaps: Vec::new(),
            lanes,
            clock: 0,
            turn_start,
            image_tokens,
            audio_tokens,
            reranker: None,
        }
    }

    pub fn run(mut self, rx: std::sync::mpsc::Receiver<Job>) {
        if let Some(pc) = &self.o.prefix {
            let t = Instant::now();
            match pc.newest() {
                Some((tokens, kv, path)) => match self.engine.kv_load(tokens.len(), &kv) {
                    Ok(()) => {
                        eprintln!(
                            "[prefix-cache] {} tokens ({:.1} MiB) loaded from {} at start in {:.0} ms",
                            tokens.len(),
                            kv.len() as f64 / (1u64 << 20) as f64,
                            path.display(),
                            t.elapsed().as_secs_f64() * 1e3
                        );
                        self.history = tokens;
                    }
                    Err(e) => {
                        eprintln!("[prefix-cache] {}: {e}: not loaded", path.display());
                        let _ = self.engine.reset();
                    }
                },
                None => eprintln!(
                    "[prefix-cache] {}: no file of this executable, device, model and tuning yet",
                    pc.dir().display()
                ),
            }
        }
        eprintln!(
            "[serve] worker ready · {} · capacity {}",
            self.dialect.describe(),
            self.engine.capacity()
        );
        if self.engine.snapshot_slots() > 0 {
            eprintln!(
                "[serve] {} lanes · {} snapshots of {:.1} MiB",
                self.engine.lanes(),
                self.engine.snapshot_slots(),
                self.engine.snapshot_bytes() as f64 / (1u64 << 20) as f64
            );
        }
        let mut warm_steps = 0u64;
        // A step the engine skips (an encoder without one, a full context) is said once, not at
        // every period.
        let mut skip_said = false;
        loop {
            let job = match self.o.keep_warm {
                Some(d) => match rx.recv_timeout(d) {
                    Ok(job) => job,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        warm_steps += 1;
                        let (pos, t) = (self.engine.pos(), Instant::now());
                        match self.engine.keep_warm() {
                            Ok(Some((step, rows, block))) => eprintln!(
                                "[keep-warm] step {warm_steps} at position {pos}: {:.1} ms (one token {:.1} ms, a block of {} rows {:.1} ms)",
                                t.elapsed().as_secs_f64() * 1e3,
                                step * 1e3,
                                rows,
                                block * 1e3
                            ),
                            Ok(None) if !skip_said => {
                                skip_said = true;
                                eprintln!("[keep-warm] step {warm_steps} at position {pos}: skipped (no step for this model, or the context is full), said once")
                            }
                            Ok(None) => {}
                            Err(e) => eprintln!("[keep-warm] step {warm_steps} at position {pos} failed after {:.1} ms: {e}", t.elapsed().as_secs_f64() * 1e3),
                        }
                        continue;
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                },
                None => match rx.recv() {
                    Ok(job) => job,
                    Err(_) => break,
                },
            };
            let (id, tx) = (job.id, job.tx.clone());
            if let Err((code, body)) = self.answer(job) {
                eprintln!("[req {id}] error {code}: {body}");
                let _ = tx.blocking_send(Out::Error(code, body));
                if code >= 500 {
                    // A failed call may leave the state other than the tokens say (the CPU engine
                    // refuses to go on before a reset): the next request starts
                    // from an empty conversation.
                    if let Err(e) = self.engine.reset() {
                        eprintln!("[req {id}] reset after the error: {e}");
                    }
                    self.history.clear();
                }
            }
        }
    }

    fn log_file(&self, id: u64, name: &str, data: &[u8]) {
        if let Some(d) = &self.o.log_dir {
            let _ = std::fs::write(d.join(format!("req-{id:05}.{name}")), data);
        }
    }

    /// The tokens of the request's stable prefix: the template over its leading system messages and
    /// its tools alone, when they begin the request's tokens (at least [`PREFIX_MIN`] of them,
    /// the last prompt token not among them), cut to a multiple of [`Backend::block_align`].
    fn stable_prefix(
        &self,
        messages: &Value,
        tools: Option<&Value>,
        thinking: bool,
        ids: &[u32],
    ) -> Option<usize> {
        let system: Vec<Value> = messages
            .as_array()?
            .iter()
            .take_while(|m| matches!(m["role"].as_str(), Some("system" | "developer")))
            .cloned()
            .collect();
        let text = self
            .template
            .render(&Value::Array(system), tools, thinking, false)
            .ok()?;
        let p = self
            .tokenizer
            .encode(text.as_str(), false)
            .ok()?
            .get_ids()
            .to_vec();
        let n = p.len() - p.len() % self.engine.block_align();
        (n >= PREFIX_MIN && p.len() < ids.len() && ids.starts_with(&p)).then_some(n)
    }

    /// `ids[from..to]` fed: the text by the engine's prefill blocks, each image or recording whole
    /// through its tower. The greedy token of the last token when it is text (0 after a
    /// medium), `None` when `stop` said so (the state then holds what was fed).
    #[allow(clippy::too_many_arguments)]
    fn feed(
        engine: &mut B,
        ids: &[u32],
        from: usize,
        to: usize,
        spans: &[Span],
        images: &[Image],
        recordings: &[Recording],
        stop: &mut dyn FnMut() -> bool,
    ) -> Result<Option<u32>, String> {
        let (mut at, mut last) = (from, Some(0));
        for s in spans
            .iter()
            .filter(|s| s.start + s.len > from && s.start < to)
        {
            if s.start < from || s.start + s.len > to {
                return Err(format!(
                    "the medium at positions {}..{} is cut by {from}..{to}",
                    s.start,
                    s.start + s.len
                ));
            }
            if at < s.start {
                engine.set_rope_shift(rope_shift_at(at, spans))?;
                last = engine.prefill_until(&ids[at..s.start], stop)?;
                if last.is_none() {
                    return Ok(None);
                }
            }
            if stop() {
                return Ok(None);
            }
            engine.set_rope_shift(rope_shift_at(s.start, spans))?;
            match s.kind {
                Kind::Image(i) => {
                    let im = &images[i];
                    engine.prefill_image(&im.rgb, im.h, im.w, s.len)?;
                }
                Kind::Audio(i) => engine.prefill_audio(&recordings[i].wave, s.len)?,
            }
            (at, last) = (s.start + s.len, Some(0));
        }
        // The text after the last medium, and the tokens generated after it, turn by its offset.
        engine.set_rope_shift(rope_shift_at(at, spans))?;
        if at < to {
            last = engine.prefill_until(&ids[at..to], stop)?;
        }
        Ok(last)
    }

    /// The engine's state given to an embedding or a rerank, then an empty conversation: what the
    /// cache held is gone.
    fn after_one_shot(&mut self) {
        if let Err(e) = self.engine.reset() {
            eprintln!("[serve] reset after an embedding or a rerank: {e}");
        }
        self.history.clear();
    }

    /// OpenAI's `/v1/embeddings`: `input` a string or an array of strings, each embedded apart
    /// ([`Backend::embed`]), cut to `dimensions` when asked (Matryoshka models: the first
    /// components) and normalized to length 1. The prefixes a model wants (EmbeddingGemma's
    /// `task: search result | query: `, Qwen3-Embedding's instruction) are the client's.
    ///
    /// A model that takes media ([`Backend::embed_media_kinds`], EmbeddingGemma 2) takes an input
    /// as an array of the chats' content parts too (see [`embedding_inputs`]): its text parts
    /// joined, each image or recording at the place of its part (a `<|image|>` or `<|audio|>`
    /// placeholder), unless the text writes the placeholders itself (they are then taken in
    /// order), as the model's chat template does.
    fn embeddings(&mut self, job: Job) -> std::result::Result<(), (u16, Value)> {
        let (id, body, tx) = (job.id, job.body, job.tx);
        let bad = |m: String| (400u16, error_body(400, "invalid_request_error", &m));
        let internal = |m: String| (500u16, error_body(500, "server_error", &m));
        let Some(dim) = self.engine.embedding_dim() else {
            return Err(bad(format!(
                "{}: this model gives no embedding",
                self.model
            )));
        };
        let inputs =
            embedding_inputs(&body["input"], self.engine.embed_media_kinds()).map_err(bad)?;
        let keep = match body["dimensions"].as_u64() {
            None => dim,
            Some(d) if d as usize >= 1 && d as usize <= dim => d as usize,
            Some(d) => return Err(bad(format!("dimensions {d}: between 1 and {dim}"))),
        };
        let t = Instant::now();
        let (mut data, mut tokens) = (Vec::with_capacity(inputs.len()), 0usize);
        let unit = |mut v: Vec<f32>| {
            v.truncate(keep);
            let norm = v
                .iter()
                .map(|x| (*x as f64) * (*x as f64))
                .sum::<f64>()
                .sqrt();
            if norm > 0.0 {
                v.iter_mut().for_each(|x| *x = (*x as f64 / norm) as f32);
            }
            v
        };
        let text_only = inputs.iter().all(|x| x.media.is_empty());
        let (mut t_tok, mut t_eng) = (0.0, 0.0);
        if text_only {
            // Text only: every input tokenized, then all given to the engine at once
            // (`Backend::embed_batch`).
            let mut all = Vec::with_capacity(inputs.len());
            let tok = &self.tokenizer;
            let encoded = map_threads(inputs.len(), |i| {
                tok.encode(inputs[i].text.as_str(), true)
                    .map(|e| e.get_ids().to_vec())
            });
            for (i, ids) in encoded.into_iter().enumerate() {
                let ids = ids.map_err(|e| internal(format!("tokenizer: {e}")))?;
                if ids.is_empty() || ids.len() > self.engine.capacity() {
                    return Err(bad(format!(
                        "input {i}: {} tokens, the context holds {}",
                        ids.len(),
                        self.engine.capacity()
                    )));
                }
                tokens += ids.len();
                all.push(ids);
            }
            let refs: Vec<&[u32]> = all.iter().map(Vec::as_slice).collect();
            t_tok = t.elapsed().as_secs_f64();
            let r = self.engine.embed_batch(&refs);
            t_eng = t.elapsed().as_secs_f64() - t_tok;
            self.after_one_shot();
            for (i, v) in r.map_err(internal)?.into_iter().enumerate() {
                data.push(
                    json!({ "object": "embedding", "index": i, "embedding": short_f32s(&unit(v)) }),
                );
            }
        }
        for (i, input) in inputs.iter().enumerate().filter(|_| !text_only) {
            let ids = self
                .tokenizer
                .encode(input.text.as_str(), true)
                .map_err(|e| internal(format!("tokenizer: {e}")))?
                .get_ids()
                .to_vec();
            if ids.is_empty() || ids.len() > self.engine.capacity() {
                self.after_one_shot();
                return Err(bad(format!(
                    "input {i}: {} tokens, the context holds {}",
                    ids.len(),
                    self.engine.capacity()
                )));
            }
            tokens += ids.len();
            let media: Vec<crate::backend::Medium> = input
                .media
                .iter()
                .map(|m| match m {
                    InputMedium::Image(im) => crate::backend::Medium::Image(im),
                    InputMedium::Audio(a) => crate::backend::Medium::Audio(a),
                })
                .collect();
            let r = self.engine.embed_media(&ids, &media);
            self.after_one_shot();
            let v = r.map_err(|e| {
                if input.media.is_empty() {
                    internal(e)
                } else {
                    bad(format!("input {i}: {e}"))
                }
            })?;
            data.push(
                json!({ "object": "embedding", "index": i, "embedding": short_f32s(&unit(v)) }),
            );
        }
        let n_media: usize = inputs.iter().map(|x| x.media.len()).sum();
        let times = if text_only {
            format!(" (tokenizer {:.3} s, engine {:.3} s)", t_tok, t_eng)
        } else {
            String::new()
        };
        eprintln!("[req {id}] embeddings: {} inputs ({n_media} media), {tokens} tokens, {keep} of {dim} dimensions, {:.3} s{times}", inputs.len(), t.elapsed().as_secs_f64());
        let answer = json!({ "object": "list", "data": data, "model": self.model, "usage": { "prompt_tokens": tokens, "total_tokens": tokens } });
        let _ = tx.blocking_send(Out::Message(answer));
        Ok(())
    }

    /// llama-server's `/v1/rerank` (Jina's shape): `query`, `documents` (strings), `top_n`; the
    /// documents by relevance ([`Backend::score`]), highest first.
    fn rerank(&mut self, job: Job) -> std::result::Result<(), (u16, Value)> {
        let (id, body, tx) = (job.id, job.body, job.tx);
        let bad = |m: String| (400u16, error_body(400, "invalid_request_error", &m));
        let internal = |m: String| (500u16, error_body(500, "server_error", &m));
        let query = body["query"]
            .as_str()
            .ok_or_else(|| bad("query is required".into()))?
            .to_string();
        let docs: Vec<String> = match body["documents"].as_array() {
            Some(a) if !a.is_empty() && a.iter().all(Value::is_string) => a
                .iter()
                .map(|v| v.as_str().unwrap_or_default().to_string())
                .collect(),
            _ => return Err(bad("documents must be a non-empty array of strings".into())),
        };
        let top_n = body["top_n"]
            .as_u64()
            .map_or(docs.len(), |n| (n as usize).min(docs.len()));
        let t = Instant::now();
        let (mut all, mut tokens) = (Vec::with_capacity(docs.len()), 0usize);
        let (tok, reranker) = (&*self.tokenizer, self.reranker.as_ref());
        let encoded = map_threads(docs.len(), |i| {
            rerank_tokens(tok, reranker, &query, &docs[i])
        });
        for (i, ids) in encoded.into_iter().enumerate() {
            let ids = ids.map_err(internal)?;
            if ids.len() > self.engine.capacity() {
                return Err(bad(format!(
                    "document {i}: {} tokens with the query, the context holds {}",
                    ids.len(),
                    self.engine.capacity()
                )));
            }
            tokens += ids.len();
            all.push(ids);
        }
        // Every pair given to the engine at once (`Backend::score_batch`: one by one unless the
        // engine batches them).
        let refs: Vec<&[u32]> = all.iter().map(Vec::as_slice).collect();
        let t_tok = t.elapsed().as_secs_f64();
        let r = self.engine.score_batch(&refs);
        let t_eng = t.elapsed().as_secs_f64() - t_tok;
        self.after_one_shot();
        let mut scored: Vec<(usize, f32)> = r.map_err(internal)?.into_iter().enumerate().collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        let results: Vec<Value> = scored
            .iter()
            .take(top_n)
            .map(|(i, s)| json!({ "index": i, "relevance_score": s }))
            .collect();
        eprintln!("[req {id}] rerank: {} documents, {tokens} tokens, {:.3} s (tokenizer {t_tok:.3} s, engine {t_eng:.3} s)", docs.len(), t.elapsed().as_secs_f64());
        let answer = json!({ "model": self.model, "object": "list", "results": results, "usage": { "prompt_tokens": tokens, "total_tokens": tokens } });
        let _ = tx.blocking_send(Out::Message(answer));
        Ok(())
    }

    /// The current lane holds the tokens of the state, position by position: what was fed there
    /// since the state's beginning (positions past it written over, or not to be trusted).
    fn sync_lane(&mut self) {
        let l = self.engine.lane();
        self.lanes[l].kv.clone_from(&self.history);
    }

    /// Keep the live state as a snapshot of `history` (an identical one is only touched), the least
    /// recently used one given up when every slot is taken.
    fn snapshot(&mut self) -> Result<(), String> {
        let max = self.engine.snapshot_slots();
        if max == 0 {
            return Ok(());
        }
        self.sync_lane();
        self.clock += 1;
        let lane = self.engine.lane();
        // Snapshots whose lane has since been written over their positions serve no more.
        let lanes = &self.lanes;
        self.snaps
            .retain(|s| lanes[s.lane].kv.starts_with(&s.tokens));
        if let Some(s) = self
            .snaps
            .iter_mut()
            .find(|s| s.lane == lane && s.tokens == self.history)
        {
            s.used = self.clock;
            return Ok(());
        }
        let i = if self.snaps.len() < max {
            let used: Vec<usize> = self.snaps.iter().map(|s| s.slot).collect();
            let slot = (0..max).find(|i| !used.contains(i)).unwrap_or(0);
            self.snaps.push(Snap {
                slot,
                lane,
                tokens: Vec::new(),
                used: 0,
            });
            self.snaps.len() - 1
        } else {
            self.snaps
                .iter()
                .enumerate()
                .min_by_key(|(_, s)| s.used)
                .map(|(i, _)| i)
                .unwrap_or(0)
        };
        self.engine.snapshot(self.snaps[i].slot)?;
        self.snaps[i].lane = lane;
        self.snaps[i].tokens = self.history.clone();
        self.snaps[i].used = self.clock;
        Ok(())
    }

    /// A state that cannot be cut back: the longest one `ids` extends (at least one token left to
    /// feed), the live one, a snapshot whose lane still holds its positions, or an empty
    /// conversation in the lane used least recently (the current one when the request asks for
    /// no cache). How the prefix was found.
    fn rewind(&mut self, ids: &[u32], cache: bool) -> Result<String, String> {
        self.sync_lane();
        self.clock += 1;
        let cur = self.engine.lane();
        let fits = |t: &[u32]| t.len() < ids.len() && ids.starts_with(t);
        let live = cache && fits(&self.history);
        let best = if cache {
            self.snaps
                .iter()
                .enumerate()
                .filter(|(_, s)| fits(&s.tokens) && self.lanes[s.lane].kv.starts_with(&s.tokens))
                .max_by_key(|(_, s)| (s.tokens.len(), s.lane == cur))
                .map(|(i, s)| (i, s.tokens.len()))
        } else {
            None
        };
        let how = match best {
            Some((i, n)) if n > 0 && (!live || n > self.history.len()) => {
                self.snaps[i].used = self.clock;
                let (slot, lane) = (self.snaps[i].slot, self.snaps[i].lane);
                self.engine.set_lane(lane)?;
                let pos = self.engine.restore(slot)?;
                if pos != n {
                    return Err(format!(
                        "snapshot {slot} stands for {pos} positions, not {n}"
                    ));
                }
                self.history = self.snaps[i].tokens.clone();
                format!("snapshot, lane {lane}")
            }
            _ if live && !self.history.is_empty() => format!("live, lane {cur}"),
            _ => {
                let lane = if cache {
                    (0..self.lanes.len())
                        .min_by_key(|&l| self.lanes[l].used)
                        .unwrap_or(0)
                } else {
                    cur
                };
                self.engine.set_lane(lane)?;
                self.engine.reset()?;
                self.history.clear();
                format!(
                    "{}, lane {lane}",
                    if cache { "empty" } else { "cache_prompt false" }
                )
            }
        };
        let lane = self.engine.lane();
        self.lanes[lane].used = self.clock;
        Ok(how)
    }

    fn answer(&mut self, job: Job) -> std::result::Result<(), (u16, Value)> {
        match job.body["__route"].as_str() {
            Some("embeddings") => return self.embeddings(job),
            Some("rerank") => return self.rerank(job),
            _ => {}
        }
        let (id, body, tx) = (job.id, job.body, job.tx);
        let bad = |m: String| (400u16, error_body(400, "invalid_request_error", &m));
        let internal = |m: String| (500u16, error_body(500, "server_error", &m));
        self.log_file(
            id,
            "request.json",
            &serde_json::to_vec_pretty(&body).unwrap_or_default(),
        );
        // llama-server's `/completion`: `prompt` is the raw text, fed as it is (no template, no
        // special token added).
        let raw = body.get("prompt").and_then(Value::as_str);
        let no_messages = json!([]);
        let messages = match raw {
            Some(_) => &no_messages,
            None => body
                .get("messages")
                .filter(|m| m.is_array())
                .ok_or_else(|| bad("messages is required".into()))?,
        };
        // `n_probs`: the first candidates of each step, written to the request's log (`--log-dir`),
        // `probs.json`, and nowhere else: without a log they are not computed.
        let n_probs = if self.o.log_dir.is_some() {
            body["n_probs"].as_u64().unwrap_or(0) as usize
        } else {
            0
        };
        let tools = body
            .get("tools")
            .filter(|t| t.as_array().is_some_and(|a| !a.is_empty()));
        let stream = body["stream"].as_bool().unwrap_or(false);
        let thinking = body
            .pointer("/chat_template_kwargs/enable_thinking")
            .and_then(Value::as_bool)
            .unwrap_or(self.dialect.family.thinks_by_default());
        let per_token = body["timings_per_token"].as_bool().unwrap_or(false);
        let include_usage = body
            .pointer("/stream_options/include_usage")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let sampling = Sampling {
            temperature: body["temperature"].as_f64().unwrap_or(0.8) as f32,
            top_k: body["top_k"].as_u64().unwrap_or(40) as usize,
            top_p: body["top_p"].as_f64().unwrap_or(0.95) as f32,
            min_p: body["min_p"].as_f64().unwrap_or(0.05) as f32,
            seed: body["seed"].as_u64().unwrap_or_else(unix),
        };
        let presence = body["presence_penalty"]
            .as_f64()
            .map(|v| v as f32)
            .unwrap_or(self.o.presence_default);
        let frequency = body["frequency_penalty"].as_f64().unwrap_or(0.0) as f32;
        let max_tokens = body["max_tokens"]
            .as_u64()
            .or_else(|| body["max_completion_tokens"].as_u64())
            .or_else(|| body["n_predict"].as_u64())
            .unwrap_or(0) as usize;
        // llama-server's `cache_prompt` (default true): false feeds the whole prompt from an empty
        // conversation.
        let cache = body["cache_prompt"].as_bool().unwrap_or(true);
        // llama-server's `ignore_eos` (default false): the end-of-generation tokens are never
        // picked, the answer runs to `max_tokens` (or the context). Greedy without
        // penalties, the logits are copied only when their argmax is one of them.
        let ignore_eos = body["ignore_eos"].as_bool().unwrap_or(false);
        let mask_always =
            ignore_eos && (sampling.temperature > 0.0 || presence != 0.0 || frequency != 0.0);

        let prompt = match raw {
            Some(p) => p.to_string(),
            None => self
                .template
                .render(messages, tools, thinking, true)
                .map_err(|e| bad(e.to_string()))?,
        };
        self.log_file(id, "prompt.txt", prompt.as_bytes());
        let ids: Vec<u32> = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| internal(format!("tokenizer: {e}")))?
            .get_ids()
            .to_vec();
        let mut images = image::collect(messages).map_err(bad)?;
        for (i, im) in images.iter_mut().enumerate() {
            im.soft = self
                .engine
                .image_soft_tokens(im.h, im.w)
                .transpose()
                .map_err(|e| bad(format!("image {i}: {e}")))?;
        }
        let recordings = audio::collect(messages, self.engine.audio())
            .map_err(|e| bad(format!("{}: {e}", self.model)))?;
        let (ids, spans) = if images.is_empty() && recordings.is_empty() {
            (ids, Vec::new())
        } else {
            let v = self.engine.vision();
            if !images.is_empty() && v.is_none() {
                return Err(bad(format!("{}: this model reads no image", self.model)));
            }
            let engine = &self.engine;
            let image_count = |im: &Image| -> Result<(usize, usize), String> {
                if let Some(n) = im.soft {
                    return Ok((n, n));
                }
                match engine.image_tokens(im.h, im.w) {
                    Some(r) => r,
                    None => {
                        image::soft_tokens(im.h, im.w, v.ok_or("no image tower")?).map(|n| (n, n))
                    }
                }
            };
            expand_media(
                ids,
                self.image_tokens,
                &images,
                &image_count,
                self.audio_tokens,
                &recordings,
                self.engine.audio(),
            )
            .map_err(bad)?
        };
        if ids.is_empty() {
            return Err(bad("the prompt is empty".into()));
        }
        let cap = self.engine.capacity();
        if ids.len() >= cap {
            return Err((
                400,
                json!({ "error": { "code": 400, "type": "exceed_context_size_error", "message": format!("the request exceeds the available context size ({} tokens, n_ctx {cap})", ids.len()), "n_prompt_tokens": ids.len(), "n_ctx": cap } }),
            ));
        }
        let gone = || tx.is_closed();
        let t_prompt = Instant::now();
        // What the request found in the cache, and what it computed for the prefix cache
        // (`computed`, `from_disk`).
        let (mut from_disk, mut computed) = (0usize, 0usize);
        let (how, mut common): (String, usize);
        if self.engine.can_truncate() {
            how = "prefix".into();
            // The prefix shared with what the K/V cache holds; at least the last prompt token is
            // fed again, for its logits.
            let shared = |h: &[u32]| {
                if cache {
                    h.iter()
                        .zip(&ids)
                        .take_while(|(a, b)| a == b)
                        .count()
                        .min(ids.len() - 1)
                } else {
                    0
                }
            };
            common = shared(&self.history);
            // `--side-slot`: the state kept aside when it shares more with the request, the live
            // one kept aside when the request would discard most of it.
            let slots = self.engine.side_slots();
            if slots > 0 {
                let valid = self.engine.side_valid();
                let aside = if valid { shared(&self.aside) } else { 0 };
                let t = Instant::now();
                if valid && aside > common + SIDE_MARGIN {
                    let r = self.engine.side_swap();
                    let bytes = r.map_err(|e| {
                        self.history.clear();
                        self.aside.clear();
                        internal(e)
                    })?;
                    std::mem::swap(&mut self.aside, &mut self.history);
                    common = aside;
                    eprintln!(
                        "[req {id}] side: swap, {aside} tokens of the state kept aside reused, the live one of {} tokens kept aside, {:.1} MiB in {:.0} ms",
                        self.aside.len(),
                        bytes as f64 / (1u64 << 20) as f64,
                        t.elapsed().as_secs_f64() * 1e3
                    );
                } else if self.history.len() > common + SIDE_MARGIN
                    && 2 * common < self.history.len()
                    && (!valid || self.history.len() >= self.aside.len())
                    && (self.history.len() <= slots || ids.len() < slots)
                {
                    // Another conversation: the live state would lose most of what it holds, saved
                    // first (unless what is aside is longer, or both the live
                    // state and the request run past the head, which would drop it at once).
                    let bytes = self.engine.side_save().map_err(internal)?;
                    self.aside.clone_from(&self.history);
                    eprintln!("[req {id}] side: save, the live state of {} tokens kept aside, {:.1} MiB in {:.0} ms", self.aside.len(), bytes as f64 / (1u64 << 20) as f64, t.elapsed().as_secs_f64() * 1e3);
                }
            }
            // The prompt's blocks start on the attention's query tiles: the same bytes whatever was
            // reused.
            common = reuse_point(common, self.engine.block_align(), &spans);
            common = self.engine.truncate(common).map_err(internal)?;
            self.history.truncate(common);
            // `--prefix-cache`: the stable prefix read from disk, else computed apart and written
            // there; a one-shot request (no answer nor tool result in it: an agent's
            // side request) writes there what it shares with the cache past its
            // stable prefix (the day's calendar); the longest file that begins the request read
            // when the cache holds less of it.
            let stable = self
                .o
                .prefix
                .as_ref()
                .and_then(|_| self.stable_prefix(messages, tools, thinking, &ids));
            if let (Some(pc), Some(p)) = (&self.o.prefix, stable) {
                if common >= p {
                    pc.used(&ids[..p]);
                }
                let one_shot = messages.as_array().is_some_and(|m| {
                    m.iter()
                        .all(|m| !matches!(m["role"].as_str(), Some("assistant" | "tool")))
                });
                if one_shot && common >= p + PREFIX_MIN {
                    if pc.has(&ids[..common]) {
                        pc.used(&ids[..common]);
                    } else {
                        let t = Instant::now();
                        match self.engine.kv_save(common) {
                            Ok(kv) => {
                                eprintln!("[prefix-cache] [req {id}] {common} tokens shared with the cache past the {p} of the stable prefix, K/V read back in {:.0} ms ({:.1} MiB), written by a thread", t.elapsed().as_secs_f64() * 1e3, kv.len() as f64 / (1u64 << 20) as f64);
                                drop(pc.store(ids[..common].to_vec(), kv));
                            }
                            Err(e) => eprintln!(
                                "[prefix-cache] [req {id}] the K/V could not be read back: {e}"
                            ),
                        }
                    }
                }
            }
            let best =
                self.o.prefix.as_ref().and_then(|pc| {
                    pc.best(&ids, common.max(stable.unwrap_or(0).saturating_sub(1)))
                });
            if let Some(p) = best.or(stable.filter(|&p| common < p)) {
                let pc = self.o.prefix.as_ref().unwrap();
                let t = Instant::now();
                if let Some(kv) = pc.load(&ids[..p]) {
                    match self.engine.kv_load(p, &kv) {
                        Ok(()) => {
                            eprintln!(
                                "[prefix-cache] [req {id}] {p} tokens of the prefix ({:.1} MiB) read from {} in {:.0} ms",
                                kv.len() as f64 / (1u64 << 20) as f64,
                                pc.dir().display(),
                                t.elapsed().as_secs_f64() * 1e3
                            );
                            self.history = ids[..p].to_vec();
                            from_disk = p - common;
                            common = p;
                        }
                        Err(e) => {
                            eprintln!("[prefix-cache] [req {id}] {e}: the prefix is computed");
                            self.engine.truncate(0).map_err(internal)?;
                            self.history.clear();
                            common = 0;
                        }
                    }
                }
                if common < p {
                    let fed = Self::feed(
                        &mut self.engine,
                        &ids,
                        common,
                        p,
                        &spans,
                        &images,
                        &recordings,
                        &mut || gone(),
                    )
                    .map_err(internal)?;
                    let pos = self.engine.pos();
                    self.history.extend_from_slice(&ids[common..pos]);
                    if fed.is_none() {
                        eprintln!("[req {id}] the client went away during the prompt: {} of {} tokens fed in {:.2} s, cache now {pos} tokens", pos - common, ids.len() - common, t_prompt.elapsed().as_secs_f64());
                        return Ok(());
                    }
                    computed = p - common;
                    let t = Instant::now();
                    match self.engine.kv_save(p) {
                        Ok(kv) => {
                            eprintln!("[prefix-cache] [req {id}] {p} tokens of the prefix computed in {:.2} s, K/V read back in {:.0} ms ({:.1} MiB), written by a thread", t_prompt.elapsed().as_secs_f64(), t.elapsed().as_secs_f64() * 1e3, kv.len() as f64 / (1u64 << 20) as f64);
                            drop(pc.store(ids[..p].to_vec(), kv));
                        }
                        Err(e) => eprintln!(
                            "[prefix-cache] [req {id}] the K/V could not be read back: {e}"
                        ),
                    }
                    common = p;
                }
            }
        } else {
            how = self.rewind(&ids, cache).map_err(internal)?;
            common = self.history.len();
        }
        eprintln!(
            "[req {id}] {} messages, {} tools, thinking {thinking}, stream {stream}, temperature {} top_k {} top_p {} min_p {} presence {presence} frequency {frequency} seed {}, max_tokens {max_tokens} · prompt {} tokens, {common} reused from the cache ({how}), {} to feed",
            messages.as_array().map_or(0, Vec::len),
            tools.and_then(Value::as_array).map_or(0, Vec::len),
            sampling.temperature,
            sampling.top_k,
            sampling.top_p,
            sampling.min_p,
            sampling.seed,
            ids.len(),
            ids.len() - common
        );
        if !images.is_empty() {
            let fed: Vec<String> = spans
                .iter()
                .filter(|s| s.start >= common)
                .filter_map(|s| {
                    if let Kind::Image(i) = s.kind {
                        Some(format!("{}×{} → {}", images[i].w, images[i].h, s.len))
                    } else {
                        None
                    }
                })
                .collect();
            eprintln!(
                "[req {id}] {} images, {} of them to feed (pixels → soft tokens): {}",
                images.len(),
                fed.len(),
                fed.join(", ")
            );
        }
        if !recordings.is_empty() {
            let fed: Vec<String> = spans
                .iter()
                .filter(|s| s.start >= common)
                .filter_map(|s| {
                    if let Kind::Audio(i) = s.kind {
                        Some(format!(
                            "{:.2} s at {} Hz → {}",
                            recordings[i].seconds, recordings[i].rate, s.len
                        ))
                    } else {
                        None
                    }
                })
                .collect();
            eprintln!(
                "[req {id}] {} recordings, {} of them to feed (duration → soft tokens): {}",
                recordings.len(),
                fed.len(),
                fed.join(", ")
            );
        }
        if from_disk > 0 || computed > 0 {
            eprintln!("[req {id}] of the {common} tokens of the cache: {from_disk} read from the disk, {computed} computed for it");
        }

        let mut start = common;
        // A state that cannot be cut back, with snapshots: the prompt by segments, a snapshot
        // before each of its last messages.
        if !self.engine.can_truncate() && self.engine.snapshot_slots() > 0 {
            let mut cuts: Vec<usize> = match self.turn_start {
                Some(t) => (common + 1..ids.len()).filter(|&p| ids[p] == t).collect(),
                None => Vec::new(),
            };
            cuts = cuts.split_off(cuts.len().saturating_sub(BOUNDARIES));
            // Each cut down to a multiple of the engine's alignment: the prompt fed by segments
            // then gives the bytes of one call (a block that starts elsewhere changes
            // them, as this engine's SME2 kernels do on an Apple M5).
            let align = self.engine.block_align();
            cuts = cuts
                .into_iter()
                .map(|c| reuse_point(c, align, &spans))
                .filter(|&c| c > start)
                .collect();
            cuts.dedup();
            for cut in cuts {
                let fed = Self::feed(
                    &mut self.engine,
                    &ids,
                    start,
                    cut,
                    &spans,
                    &images,
                    &recordings,
                    &mut || gone(),
                )
                .map_err(internal)?;
                let pos = self.engine.pos();
                self.history.extend_from_slice(&ids[start..pos]);
                if fed.is_none() {
                    eprintln!("[req {id}] the client went away during the prompt: {} of {} tokens fed in {:.2} s, cache now {pos} tokens", pos - common, ids.len() - common, t_prompt.elapsed().as_secs_f64());
                    return Ok(());
                }
                start = cut;
                self.snapshot().map_err(internal)?;
            }
        }
        let Some(argmax) = Self::feed(
            &mut self.engine,
            &ids,
            start,
            ids.len(),
            &spans,
            &images,
            &recordings,
            &mut || gone(),
        )
        .map_err(internal)?
        else {
            let pos = self.engine.pos();
            self.history.extend_from_slice(&ids[start..pos]);
            eprintln!("[req {id}] the client went away during the prompt: {} of {} tokens fed in {:.2} s, cache now {pos} tokens", pos - start, ids.len() - start, t_prompt.elapsed().as_secs_f64());
            return Ok(());
        };
        self.history.extend_from_slice(&ids[start..]);
        // What the request computed (the prefix computed for the cache included); `common`: what it
        // did not.
        common -= computed;
        let prompt_s = t_prompt.elapsed().as_secs_f64();
        let prompt_n = ids.len() - common;

        let created = unix();
        let cid = format!("chatcmpl-herbert-{id}");
        let model = self.model.clone();
        let fingerprint = self.o.fingerprint;
        let chunk = |delta: Value, finish: Option<&str>, timings: Option<Value>| -> Value {
            let mut c = json!({ "id": cid, "object": "chat.completion.chunk", "created": created, "model": model, "system_fingerprint": fingerprint, "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }] });
            if let Some(t) = timings {
                c["timings"] = t;
            }
            c
        };
        let timings = |predicted_n: usize, predicted_s: f64| {
            json!({
                "cache_n": common, "prompt_n": prompt_n, "prompt_ms": prompt_s * 1e3, "prompt_per_token_ms": prompt_s * 1e3 / prompt_n.max(1) as f64, "prompt_per_second": prompt_n as f64 / prompt_s.max(1e-9),
                "predicted_n": predicted_n, "predicted_ms": predicted_s * 1e3, "predicted_per_token_ms": predicted_s * 1e3 / predicted_n.max(1) as f64, "predicted_per_second": predicted_n as f64 / predicted_s.max(1e-9),
            })
        };
        if stream
            && tx
                .blocking_send(Out::Chunk(chunk(
                    json!({ "role": "assistant", "content": null }),
                    None,
                    None,
                )))
                .is_err()
        {
            return Ok(());
        }

        let mut sampler = Sampler::new(sampling);
        let mut parser = self.dialect.parser(&self.tokenizer, tools, &ids);
        let (mut content, mut reasoning, mut calls) =
            (String::new(), String::new(), Vec::<Value>::new());
        let mut produced: Vec<u32> = Vec::new();
        let mut finish = "stop";
        let mut cancelled = false;
        let room = if max_tokens == 0 {
            cap - ids.len()
        } else {
            max_tokens.min(cap - ids.len())
        };
        let t_gen = Instant::now();
        let tok = self.tokenizer.clone();
        let stops = self.dialect.stops.clone();
        // One piece of the answer: kept for the final message, streamed when asked.
        let emit = |pieces: Vec<Piece>,
                    n: usize,
                    t: f64,
                    content: &mut String,
                    reasoning: &mut String,
                    calls: &mut Vec<Value>|
         -> bool {
            for p in pieces {
                let delta = match p {
                    Piece::Content(s) => {
                        content.push_str(&s);
                        json!({ "content": s })
                    }
                    Piece::Reasoning(s) => {
                        reasoning.push_str(&s);
                        json!({ "reasoning_content": s })
                    }
                    Piece::ToolCall { name, arguments } => {
                        let call = json!({ "id": format!("call_{id}_{}", calls.len()), "type": "function", "function": { "name": name, "arguments": arguments } });
                        let mut d = call.clone();
                        d["index"] = json!(calls.len());
                        calls.push(call);
                        json!({ "tool_calls": [d] })
                    }
                };
                if stream
                    && tx
                        .blocking_send(Out::Chunk(chunk(
                            delta,
                            None,
                            per_token.then(|| timings(n, t)),
                        )))
                        .is_err()
                {
                    return false;
                }
            }
            true
        };
        let mut probs: Vec<Value> = Vec::new();
        if n_probs > 0 {
            probs.push(top_candidates(self.engine.logits(), n_probs));
        }
        let first = if ignore_eos && (mask_always || stops.contains(&argmax)) {
            let (a, l) = without_stops(self.engine.logits(), &stops);
            sampler.pick(a, &l, &[], presence, frequency)
        } else {
            sampler.pick(argmax, self.engine.logits(), &[], presence, frequency)
        };
        produced.push(first);
        if !stops.contains(&first) {
            let pieces = parser.push(&tok, first);
            if !emit(
                pieces,
                1,
                t_gen.elapsed().as_secs_f64(),
                &mut content,
                &mut reasoning,
                &mut calls,
            ) {
                cancelled = true;
            }
            if !cancelled && room > 1 {
                let fed = self
                    .engine
                    .generate(first, room - 1, &mut |argmax, logits| {
                        if tx.is_closed() {
                            cancelled = true;
                            return None;
                        }
                        if n_probs > 0 {
                            probs.push(top_candidates(logits, n_probs));
                        }
                        let recent = &produced[produced.len().saturating_sub(PENALTY_LAST_N)..];
                        let t = if ignore_eos && (mask_always || stops.contains(&argmax)) {
                            let (a, l) = without_stops(logits, &stops);
                            sampler.pick(a, &l, recent, presence, frequency)
                        } else {
                            sampler.pick(argmax, logits, recent, presence, frequency)
                        };
                        produced.push(t);
                        if stops.contains(&t) {
                            return None;
                        }
                        let pieces = parser.push(&tok, t);
                        if !emit(
                            pieces,
                            produced.len(),
                            t_gen.elapsed().as_secs_f64(),
                            &mut content,
                            &mut reasoning,
                            &mut calls,
                        ) {
                            cancelled = true;
                            return None;
                        }
                        Some(t)
                    })
                    .map_err(internal)?;
                self.history.extend_from_slice(&produced[..fed]);
                if !cancelled && !stops.contains(produced.last().unwrap_or(&0)) {
                    finish = "length";
                }
            } else if room <= 1 {
                finish = "length";
            }
        }
        let tail = parser.finish(&tok);
        emit(
            tail,
            produced.len(),
            t_gen.elapsed().as_secs_f64(),
            &mut content,
            &mut reasoning,
            &mut calls,
        );
        if !calls.is_empty() && finish == "stop" {
            finish = "tool_calls";
        }
        if cancelled {
            finish = "cancelled";
        }
        let gen_s = t_gen.elapsed().as_secs_f64();
        // Tokens counted as llama-server counts them: the end-of-turn token is one of them.
        let predicted_n = produced.len();
        let usage = json!({ "prompt_tokens": ids.len(), "completion_tokens": predicted_n, "total_tokens": ids.len() + predicted_n, "prompt_tokens_details": { "cached_tokens": common } });
        let t_all = timings(predicted_n, gen_s);
        self.log_file(
            id,
            "raw.txt",
            self.tokenizer
                .decode(&produced, false)
                .unwrap_or_default()
                .as_bytes(),
        );
        if n_probs > 0 {
            let steps: Vec<Value> = probs
                .into_iter()
                .zip(&produced)
                .map(|(mut s, &t)| {
                    s["chosen"] = json!(t);
                    s["chosen_text"] =
                        json!(self.tokenizer.decode(&[t], false).unwrap_or_default());
                    if let Some(list) = s["top"].as_array_mut() {
                        for c in list {
                            let id = c["id"].as_u64().unwrap_or(0) as u32;
                            c["text"] =
                                json!(self.tokenizer.decode(&[id], false).unwrap_or_default());
                        }
                    }
                    s
                })
                .collect();
            self.log_file(
                id,
                "probs.json",
                &serde_json::to_vec_pretty(
                    &json!({ "prompt_ids": ids, "generated_ids": produced, "steps": steps }),
                )
                .unwrap_or_default(),
            );
        }
        let answer = if stream {
            let _ = tx.blocking_send(Out::Chunk(chunk(
                json!({}),
                Some(finish),
                Some(t_all.clone()),
            )));
            if include_usage {
                let _ = tx.blocking_send(Out::Chunk(json!({ "id": cid, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [], "usage": usage, "timings": t_all })));
            }
            let _ = tx.blocking_send(Out::Done);
            json!({ "streamed": true, "content": content, "reasoning_content": reasoning, "tool_calls": calls, "finish_reason": finish, "usage": usage, "timings": t_all })
        } else {
            let mut msg = json!({ "role": "assistant", "content": content });
            if !reasoning.is_empty() {
                msg["reasoning_content"] = json!(reasoning);
            }
            if !calls.is_empty() {
                msg["tool_calls"] = json!(calls);
            }
            let full = json!({ "id": cid, "object": "chat.completion", "created": created, "model": model, "system_fingerprint": fingerprint, "choices": [{ "index": 0, "message": msg, "finish_reason": finish }], "usage": usage, "timings": t_all });
            let _ = tx.blocking_send(Out::Message(full.clone()));
            full
        };
        self.log_file(
            id,
            "response.json",
            &serde_json::to_vec_pretty(&answer).unwrap_or_default(),
        );
        eprintln!(
            "[req {id}] {finish}: prompt {prompt_n} tokens in {prompt_s:.2} s ({:.1} tok/s), {predicted_n} generated in {gen_s:.2} s ({:.2} tok/s), {} tool calls, content {} chars, reasoning {} chars · cache now {} tokens",
            prompt_n as f64 / prompt_s.max(1e-9),
            predicted_n as f64 / gen_s.max(1e-9),
            calls.len(),
            content.len(),
            reasoning.len(),
            self.history.len()
        );
        if !self.engine.can_truncate() {
            if let Err(e) = self.snapshot() {
                eprintln!("[req {id}] snapshot after the answer: {e}");
            }
        }
        Ok(())
    }
}

/// A medium of an embedding input, decoded.
enum InputMedium {
    Image(Image),
    /// A WAV file's bytes.
    Audio(Vec<u8>),
}

/// One input of `/v1/embeddings`: its text (the placeholders of its media in it) and its media in
/// order.
struct EmbeddingInput {
    text: String,
    media: Vec<InputMedium>,
}

/// The inputs of `/v1/embeddings`. `input` is a string, an array of strings (OpenAI's), or, for a
/// model that takes media (`kinds`: images, recordings), one input written as an array of content
/// parts, or an array of inputs each a string or such an array. The parts are the chats': `{"type":
/// "text", "text": …}`, `{"type": "image_url", "image_url": {"url": "data:image/…;base64,…"}}` (or
/// `"image"`), `{"type": "input_audio", "input_audio": {"data": base64, "format": "wav"}}`.
fn embedding_inputs(input: &Value, kinds: (bool, bool)) -> Result<Vec<EmbeddingInput>, String> {
    let parts_input = |parts: &[Value], at: usize| -> Result<EmbeddingInput, String> {
        let texts: String = parts
            .iter()
            .filter(|p| p["type"] == "text")
            .filter_map(|p| p["text"].as_str())
            .collect();
        // The template's rule: placeholders written in the text are the media's places; otherwise
        // each part gives its own.
        let manual = ["<|image|>", "<|audio|>", "<|video|>"]
            .iter()
            .any(|m| texts.contains(m));
        let (mut text, mut media) = (String::new(), Vec::new());
        for (j, p) in parts.iter().enumerate() {
            match p["type"].as_str() {
                Some("text") => text.push_str(
                    p["text"]
                        .as_str()
                        .ok_or(format!("input {at}, part {j}: text without a string"))?,
                ),
                Some("image_url" | "image") => {
                    if !kinds.0 {
                        return Err(format!("input {at}, part {j}: this model embeds no image"));
                    }
                    let url = p["image_url"]["url"]
                        .as_str()
                        .or_else(|| p["image_url"].as_str())
                        .or_else(|| p["url"].as_str())
                        .or_else(|| p["image"].as_str());
                    let url =
                        url.ok_or(format!("input {at}, part {j}: no URL in the image part"))?;
                    media.push(InputMedium::Image(
                        image::decode(url).map_err(|e| format!("input {at}, part {j}: {e}"))?,
                    ));
                    if !manual {
                        text.push_str("<|image|>");
                    }
                }
                Some("input_audio") => {
                    if !kinds.1 {
                        return Err(format!("input {at}, part {j}: this model embeds no audio"));
                    }
                    let a = &p["input_audio"];
                    let format = a["format"].as_str().unwrap_or("wav");
                    if format != "wav" {
                        return Err(format!(
                            "input {at}, part {j}: audio format {format:?}, only wav is read"
                        ));
                    }
                    let data = a["data"]
                        .as_str()
                        .ok_or(format!("input {at}, part {j}: input_audio without data"))?;
                    if data.len() / 4 * 3 > image::MAX_BYTES {
                        return Err(format!(
                            "input {at}, part {j}: more than {} MiB",
                            image::MAX_BYTES >> 20
                        ));
                    }
                    media.push(InputMedium::Audio(
                        base64::decode(data.trim())
                            .map_err(|e| format!("input {at}, part {j}: base64: {e}"))?,
                    ));
                    if !manual {
                        text.push_str("<|audio|>");
                    }
                }
                other => return Err(format!("input {at}, part {j}: a part of type {other:?}")),
            }
        }
        Ok(EmbeddingInput { text, media })
    };
    let one = |v: &Value, at: usize| -> Result<EmbeddingInput, String> {
        match v {
            Value::String(s) => Ok(EmbeddingInput {
                text: s.clone(),
                media: Vec::new(),
            }),
            Value::Array(parts) if !parts.is_empty() && parts.iter().all(Value::is_object) => {
                parts_input(parts, at)
            }
            _ => Err(format!(
                "input {at}: a string or a non-empty array of content parts"
            )),
        }
    };
    match input {
        Value::String(_) => Ok(vec![one(input, 0)?]),
        // An array of parts is one input; an array of strings or of arrays of parts, one input each.
        Value::Array(a) if !a.is_empty() && a.iter().all(Value::is_object) => Ok(vec![parts_input(a, 0)?]),
        Value::Array(a) if !a.is_empty() => a.iter().enumerate().map(|(i, v)| one(v, i)).collect(),
        _ => Err("input must be a string, an array of strings, or (a model that takes media) arrays of content parts".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Vision;

    #[test]
    fn a_reused_prefix_ends_after_an_image_on_its_tiles() {
        // `<bos><|turn>user\n<|image>` then 264 soft tokens (5..269), `<image|>` and the question:
        // 283 tokens.
        let spans = [Span {
            start: 5,
            len: 264,
            kind: Kind::Image(0),
            rope: 264,
        }];
        assert_eq!(reuse_point(282, 32, &spans), 269);
        assert_eq!(reuse_point(282, 16, &spans), 269);
        assert_eq!(reuse_point(310, 32, &spans), 301);
        // Inside the image (a state cut off in the middle of it): before it.
        assert_eq!(reuse_point(200, 32, &spans), 5);
        // No image: the tiles counted from 0, as before.
        assert_eq!(reuse_point(282, 32, &[]), 256);
    }

    /// An image and a recording in one prompt: each placeholder replaced by its group, the keys of
    /// its soft tokens its own (another recording, other keys), a prefix reused never cut
    /// inside the recording; a count that does not match the placeholders, or a recording for a
    /// model without the tokens, refused.
    #[test]
    fn a_recording_takes_its_placeholder_as_an_image_does() {
        let (it, at) = ((Some(10), 11, Some(12)), (20, 21, 22));
        let im = Image {
            rgb: vec![0; 768 * 768 * 3],
            h: 768,
            w: 768,
            digest: 7,
            soft: None,
        };
        let rec = |digest, samples| Recording {
            wave: vec![0.0; samples],
            rate: 16_000,
            seconds: samples as f64 / 16_000.0,
            digest,
        };
        let v = Vision {
            patch: 16,
            pool: 3,
            max_soft: 280,
        };
        let count = |im: &Image| image::soft_tokens(im.h, im.w, v).map(|n| (n, n));
        let none = |_: &Image| -> Result<(usize, usize), String> { Err("no image tower".into()) };
        // `<bos> <|image|> x <|audio|> y`: 1 s of audio, 25 soft tokens.
        let ids = vec![2, 11, 5, 21, 6];
        let (out, spans) = expand_media(
            ids.clone(),
            Some(it),
            std::slice::from_ref(&im),
            &count,
            Some(at),
            &[rec(9, 16_000)],
            Some(Audio::GEMMA4),
        )
        .unwrap();
        assert_eq!(out.len(), 5 - 2 + 2 + 256 + 2 + 25);
        assert_eq!(
            (spans[0].start, spans[0].len, spans[0].kind),
            (2, 256, Kind::Image(0))
        );
        assert_eq!(
            (spans[1].start, spans[1].len, spans[1].kind),
            (2 + 256 + 3, 25, Kind::Audio(0))
        );
        assert_eq!(
            (
                out[1],
                out[2 + 256],
                out[2 + 256 + 2],
                out[spans[1].start - 1],
                out[spans[1].start + 25]
            ),
            (10, 12, 20, 20, 22)
        );
        assert!(out[spans[1].start..spans[1].start + 25]
            .iter()
            .all(|&k| k & 0x8000_0000 != 0));
        let (other, _) = expand_media(
            ids.clone(),
            Some(it),
            std::slice::from_ref(&im),
            &count,
            Some(at),
            &[rec(8, 16_000)],
            Some(Audio::GEMMA4),
        )
        .unwrap();
        assert_eq!(out[..spans[1].start], other[..spans[1].start]);
        assert_ne!(
            out[spans[1].start..spans[1].start + 25],
            other[spans[1].start..spans[1].start + 25]
        );
        assert_eq!(reuse_point(spans[1].start + 10, 1, &spans), spans[1].start);
        // A recording alone, two placeholders for one, a model without the audio tokens.
        let (alone, s) = expand_media(
            vec![2, 21, 5],
            Some(it),
            &[],
            &none,
            Some(at),
            &[rec(9, 48_000)],
            Some(Audio::GEMMA4),
        )
        .unwrap();
        assert_eq!((alone.len(), s.len(), s[0].len), (3 - 1 + 2 + 75, 1, 75));
        assert!(expand_media(
            vec![2, 21, 21],
            Some(it),
            &[],
            &none,
            Some(at),
            &[rec(9, 16_000)],
            Some(Audio::GEMMA4)
        )
        .is_err());
        assert!(expand_media(
            vec![2, 21],
            Some(it),
            &[],
            &none,
            None,
            &[rec(9, 16_000)],
            Some(Audio::GEMMA4)
        )
        .is_err());
    }

    /// A chat request's `input_audio` part reaches the template as its `<|audio|>` and the worker
    /// as its recording.
    #[test]
    fn a_chat_recording_is_collected_in_order() {
        let tone: Vec<i16> = (0..8_000).map(|i| (i % 50) as i16 * 100).collect();
        let part = json!({"type": "input_audio", "input_audio": {"data": base64::encode(crate::audio::tests::wav16(&tone, 1, 16_000)), "format": "wav"}});
        let messages = json!([{"role": "user", "content": [{"type": "text", "text": "Transcris :"}, part.clone()]}, {"role": "assistant", "content": "…"}, {"role": "user", "content": [part]}]);
        let got = audio::collect(&messages, Some(Audio::GEMMA4)).unwrap();
        assert_eq!(
            (got.len(), got[0].wave.len(), got[0].digest == got[1].digest),
            (2, 8_000, true)
        );
        assert_eq!(Audio::GEMMA4.soft_tokens(got[0].wave.len()), 13);
    }

    /// `/v1/embeddings`' inputs: strings, an array of parts for one input, an array of inputs; the
    /// parts' placeholders written in order unless the text has its own; media refused by a
    /// model that takes none.
    #[test]
    fn embedding_inputs_are_read() {
        let mut png = Vec::new();
        ::image::RgbImage::from_fn(2, 2, |_, _| ::image::Rgb([1, 2, 3]))
            .write_to(
                &mut std::io::Cursor::new(&mut png),
                ::image::ImageFormat::Png,
            )
            .unwrap();
        let url = format!("data:image/png;base64,{}", base64::encode(&png));
        let img = json!({"type": "image_url", "image_url": {"url": url}});
        let wav = json!({"type": "input_audio", "input_audio": {"data": base64::encode(b"RIFF....WAVE"), "format": "wav"}});
        let text = |t: &str| json!({"type": "text", "text": t});
        let one = embedding_inputs(&json!("a"), (false, false)).unwrap();
        assert_eq!(
            (one.len(), one[0].text.as_str(), one[0].media.len()),
            (1, "a", 0)
        );
        assert_eq!(
            embedding_inputs(&json!(["a", "b"]), (false, false))
                .unwrap()
                .len(),
            2
        );
        let parts = embedding_inputs(
            &json!([text("x "), img.clone(), text(" y"), wav.clone()]),
            (true, true),
        )
        .unwrap();
        assert_eq!(
            (parts.len(), parts[0].text.as_str(), parts[0].media.len()),
            (1, "x <|image|> y<|audio|>", 2)
        );
        let manual = embedding_inputs(
            &json!([
                [text("<|audio|> puis <|image|>"), img.clone(), wav.clone()],
                "z"
            ]),
            (true, true),
        )
        .unwrap();
        assert_eq!(
            (
                manual.len(),
                manual[0].text.as_str(),
                manual[1].text.as_str()
            ),
            (2, "<|audio|> puis <|image|>", "z")
        );
        assert!(matches!(manual[0].media[0], InputMedium::Image(_)));
        assert!(embedding_inputs(&json!([img.clone()]), (false, true)).is_err());
        assert!(embedding_inputs(&json!([wav]), (true, false)).is_err());
        assert!(embedding_inputs(&json!([]), (true, true)).is_err());
        assert!(embedding_inputs(&json!(3), (true, true)).is_err());
    }
}

/// The `n` largest logits of a step (in the engine's 16-bit format, widened to f32) with their
/// probability after the softmax over the whole vocabulary (computed in f64), largest first; ties
/// keep the lower id first.
fn top_candidates(logits: &[u16], n: usize) -> Value {
    let f: Vec<f32> = logits
        .iter()
        .map(|&h| herbert_chat::half16::to_f32(h))
        .collect();
    let max = f.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = f.iter().map(|&v| ((v as f64) - max).exp()).sum();
    let mut order: Vec<usize> = (0..f.len()).collect();
    order.sort_by(|&a, &b| {
        f[b].partial_cmp(&f[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let top: Vec<Value> = order
        .iter()
        .take(n)
        .map(|&i| json!({ "id": i, "logit": f[i], "prob": ((f[i] as f64) - max).exp() / sum }))
        .collect();
    json!({ "top": top })
}

/// A vector of f32 as JSON numbers written with the fewest digits that give back the same f32 (as
/// `f32`'s `Display`: 9 significant digits at most), not those of its f64 widening (17): the same
/// values for a client that reads them as f32, half the bytes of an embedding's answer (1.39 → 0.7
/// MB for 64 vectors of 1024).
fn short_f32s(v: &[f32]) -> Value {
    Value::Array(
        v.iter()
            .map(|x| {
                x.to_string()
                    .parse::<f64>()
                    .ok()
                    .and_then(serde_json::Number::from_f64)
                    .map_or(Value::Null, Value::Number)
            })
            .collect(),
    )
}

#[cfg(test)]
mod short_f32s_tests {
    /// Every value read back as f32 is the one written, in at most as many bytes as its f64
    /// widening takes.
    #[test]
    fn short_f32s_give_back_the_same_f32() {
        let v: Vec<f32> = (0..20000)
            .map(|i| ((i as f32 * 0.7731).sin() * 0.05) / (1.0 + (i % 7) as f32))
            .chain([0.0, -0.0, 1.0, 1e-30, -3.4e38])
            .collect();
        let s = super::short_f32s(&v).to_string();
        let back: Vec<f64> = serde_json::from_str(&s).unwrap();
        assert!(v.iter().zip(&back).all(|(a, b)| *a == *b as f32));
        assert!(s.len() < serde_json::json!(v).to_string().len());
    }
}
