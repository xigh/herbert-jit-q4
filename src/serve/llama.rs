//! herbert-jit-llama-serve's engine behind herbert-serve (llama-server's HTTP API). One engine
//! for both families:
//!
//! - Gemma 4: the K/V cut back to any position, with one more conversation kept aside
//!   (`--side-slot`: two snapshots of the whole context, the state aside and the spare of a swap),
//!   the K/V of a prefix written to disk and read back (`--prefix-cache`), the end tokens of
//!   `config.json`;
//! - Qwen3.5: the DeltaNet's state is not cut back, it is kept in snapshots (`--snapshots`, the
//!   conv and recurrent states alone), and the K/V caches in lanes (`--lanes`: the live cache
//!   exchanged with a lane's copy, the generated programs holding the live cache's addresses): a
//!   request starts from the longest of them its prompt extends.
//!
//! Both: the prompt fed by pieces of one prefill pass, a client gone asked between two of them; an
//! idle server kept warm (`--keep-warm`).

use crate::engine::Engine;
use crate::jit::model::Instantaneous;
use crate::jit::{KvLane, Recurrence};
use herbert_serve::backend::Result;
use herbert_serve::backend::Vision;
use herbert_serve::qwen3::{self, Head};
use herbert_serve::Audio;
use herbert_serve::Backend;
use std::time::Instant;

/// The conversation kept aside: its snapshot, the spare buffer of a swap, its position, whether it
/// holds one.
struct Side {
    state: Instantaneous,
    spare: Instantaneous,
    pos: usize,
    valid: bool,
}

/// The K/V caches of a state that cannot be cut back: the live one belongs to lane `current`,
/// `stores[i]` holds lane `holder[i]`'s; `held[l]`: the positions lane `l` held when it was left
/// (the server trusts no more of it).
struct Lanes {
    stores: Vec<KvLane>,
    holder: Vec<usize>,
    held: Vec<usize>,
    current: usize,
}

/// Rows of [`Cpu::keep_warm`]'s block: few, since each attends to every position
/// before it.
const KEEP_WARM_ROWS: usize = 2;
/// The token [`Cpu::keep_warm`] feeds (`<bos>`); its K/V is overwritten by the next token fed.
const KEEP_WARM_TOKEN: u32 = 2;

pub struct Cpu {
    pub engine: Engine,
    /// Gemma 4: the state is a K/V cache only, cut back to any position.
    truncates: bool,
    /// The end tokens of `config.json` (`eos_token_id`).
    eos: Vec<u32>,
    side: Option<Side>,
    /// Qwen3.5: the snapshots the server may take, each allocated on first use.
    snapshots: Vec<Option<Recurrence>>,
    snapshot_bytes: usize,
    lanes: Lanes,
    /// Qwen3.5: the DeltaNet states [`Cpu::keep_warm`] puts back, allocated on first use.
    warm: Option<Recurrence>,
    /// A Qwen3 embedding model or reranker read from the last token's hidden state
    /// (herbert-serve's `qwen3.rs`), else the logits.
    pub head: Head,
}

impl Cpu {
    pub fn new(engine: Engine, truncates: bool, eos: Vec<u32>) -> Cpu {
        let lanes = Lanes {
            stores: Vec::new(),
            holder: Vec::new(),
            held: vec![0],
            current: 0,
        };
        Cpu {
            engine,
            truncates,
            eos,
            side: None,
            snapshots: Vec::new(),
            snapshot_bytes: 0,
            lanes,
            warm: None,
            head: Head::Lm,
        }
    }

    /// The hidden state of the last of `ids` after the final norm, fed from an empty conversation.
    fn hidden_of(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        self.engine.reset();
        self.engine.prefill(ids).map_err(|e| e.to_string())?;
        Ok(qwen3::widen(&self.engine.final_hidden()))
    }

    /// Room for one more conversation kept aside (`--side-slot`, Gemma only): two snapshots sized
    /// for the whole reservation, allocated here; their bytes.
    pub fn side_enable(&mut self) -> usize {
        let (state, spare) = (
            self.engine.snapshot_reserve(),
            self.engine.snapshot_reserve(),
        );
        let bytes = state.byte_size() + spare.byte_size();
        self.side = Some(Side {
            state,
            spare,
            pos: 0,
            valid: false,
        });
        bytes
    }

    /// The state [`Backend::keep_warm`] found at position `p` put back: Gemma's K/V cut back,
    /// Qwen3.5's DeltaNet states restored from [`Cpu::warm`].
    fn put_back(&mut self, p: usize) -> Result<()> {
        let e = |e: crate::core::error::HerbertError| e.to_string();
        if self.truncates {
            return self.engine.truncate(p).map_err(e);
        }
        let r = self
            .warm
            .as_ref()
            .ok_or("keep-warm: no DeltaNet states taken")?;
        match self.engine.recurrence_restore(r).map_err(e)? {
            q if q == p => Ok(()),
            q => Err(format!(
                "keep-warm: the DeltaNet states put back at {q}, not {p}"
            )),
        }
    }

    /// Qwen3.5: `snapshots` DeltaNet snapshots (allocated on first use: the bytes of one) and
    /// `lanes` K/V caches (the live one and `lanes - 1` copies, allocated here: their bytes).
    pub fn snapshots_and_lanes(&mut self, snapshots: usize, lanes: usize) -> (usize, usize) {
        self.snapshot_bytes = self.engine.recurrence_reserve().byte_size();
        self.snapshots = (0..snapshots).map(|_| None).collect();
        let lanes = lanes.max(1);
        self.lanes.stores = (1..lanes).map(|_| self.engine.lane_reserve()).collect();
        self.lanes.holder = (1..lanes).collect();
        self.lanes.held = vec![0; lanes];
        (
            self.snapshot_bytes,
            self.lanes.stores.iter().map(KvLane::byte_size).sum(),
        )
    }
}

impl Backend for Cpu {
    fn capacity(&self) -> usize {
        self.engine.reserved()
    }
    fn pos(&self) -> usize {
        self.engine.pos()
    }
    fn reset(&mut self) -> Result<()> {
        self.engine.reset();
        Ok(())
    }
    fn prefill_until(
        &mut self,
        ids: &[u32],
        stop: &mut dyn FnMut() -> bool,
    ) -> Result<Option<u32>> {
        self.engine
            .prefill_until(ids, stop)
            .map_err(|e| e.to_string())
    }
    fn generate(
        &mut self,
        first: u32,
        max_steps: usize,
        f: &mut dyn FnMut(u32, &[u16]) -> Option<u32>,
    ) -> Result<usize> {
        self.engine
            .generate(first, max_steps, &mut |argmax, logits| {
                Ok(f(argmax, logits))
            })
            .map_err(|e| e.to_string())
    }
    fn logits(&self) -> &[u16] {
        self.engine.logits()
    }
    fn embed(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        match self.head {
            Head::Embedding { .. } => self.hidden_of(ids),
            _ => Err("this model gives no embedding".into()),
        }
    }
    fn embedding_dim(&self) -> Option<usize> {
        self.head.embedding_dim()
    }
    fn score(&mut self, ids: &[u32]) -> Result<f32> {
        if !matches!(self.head, Head::Reranker(_)) {
            return Err("this model is not a reranker".into());
        }
        let h = self.hidden_of(ids)?;
        let Head::Reranker(r) = &self.head else {
            unreachable!()
        };
        Ok(r.score(&h))
    }
    fn eos(&self) -> Vec<u32> {
        self.eos.clone()
    }
    fn vision(&self) -> Option<Vision> {
        self.engine.vision().map(|(patch, pool, max_soft)| Vision {
            patch,
            pool,
            max_soft,
        })
    }
    fn image_soft_tokens(&self, h: usize, w: usize) -> Option<std::result::Result<usize, String>> {
        self.engine.image_soft_tokens(h, w)
    }
    // An image: resized and cut on the CPU (Pillow's bicubic), the tower in
    // emitted kernels, its soft tokens fed as one block.
    fn prefill_image(&mut self, rgb: &[u8], h: usize, w: usize, soft: usize) -> Result<()> {
        self.engine
            .prefill_image(rgb, h, w, soft)
            .map_err(|e| e.to_string())
    }
    // Qwen3.5 / Qwen3.6: `smart_resize` and the multimodal RoPE (`Engine::image_tokens`,
    // `Engine::set_rope_shift`).
    fn image_tokens(&self, h: usize, w: usize) -> Option<Result<(usize, usize)>> {
        self.engine
            .image_tokens(h, w)
            .map(|r| r.map_err(|e| e.to_string()))
    }
    fn set_rope_shift(&mut self, shift: usize) -> Result<()> {
        self.engine.set_rope_shift(shift).map_err(|e| e.to_string())
    }
    fn audio(&self) -> Option<Audio> {
        // E2B, E4B: the conformer's framing; 12B: one soft token per chunk of raw samples.
        self.engine.audio().map(|(_, _, chunk)| match chunk {
            Some(n) => Audio {
                framing: herbert_serve::AudioFraming::Chunks(n),
                ..Audio::GEMMA4
            },
            None => Audio::GEMMA4,
        })
    }
    // A recording: the log-mel frames on the CPU from the 16 kHz waveform,
    // the tower in emitted kernels, its soft tokens fed as causal positions by blocks of 280
    // (`Engine::prefill_audio`).
    fn prefill_audio(&mut self, wave: &[f32], soft: usize) -> Result<()> {
        self.engine
            .prefill_audio(wave, soft)
            .map_err(|e| e.to_string())
    }
    fn can_truncate(&self) -> bool {
        self.truncates
    }
    fn truncate(&mut self, pos: usize) -> Result<usize> {
        self.engine
            .truncate(pos)
            .map(|()| pos)
            .map_err(|e| e.to_string())
    }
    // 1 by default on every target, both families (the same bits wherever a call starts, measured
    // on a Ryzen 9 9900X, an M3 Ultra and an M5: the block attention's tiles and the DeltaNet
    // chunks sit on absolute positions, the SME2 residue rows on SME2's bfdot); 16 on an SME2
    // machine under `--gemm-tail-mac bfmlal` (`Runner::block_align`). herbert-serve rounds
    // its snapshot cuts down to it.
    fn block_align(&self) -> usize {
        self.engine.block_align()
    }
    // One token and a block of two rows at the current position, put back after (Gemma's K/V cut
    // back; Qwen3.5's DeltaNet states restored, the K/V past the position overwritten by the
    // next token fed), then one word per page of every matrix read (the experts a token and two
    // rows do not reach): the decode's and the prefill's programs and buffers,
    // the conversation's K/V and the weights in use, so that the system neither compresses nor
    // evicts them.
    fn keep_warm(&mut self) -> Result<Option<(f64, usize, f64)>> {
        let p = self.engine.pos();
        if p + KEEP_WARM_ROWS >= self.engine.reserved() {
            return Ok(None);
        }
        let e = |e: crate::core::error::HerbertError| e.to_string();
        if !self.truncates {
            let r = self
                .warm
                .get_or_insert_with(|| self.engine.recurrence_reserve());
            self.engine.recurrence_take(r).map_err(e)?;
        }
        let t = Instant::now();
        let r = self
            .engine
            .generate(KEEP_WARM_TOKEN, 1, &mut |_, _| Ok(None));
        self.put_back(p)?;
        r.map_err(e)?;
        let step = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let r = self.engine.prefill(&[KEEP_WARM_TOKEN; KEEP_WARM_ROWS]);
        self.put_back(p)?;
        r.map_err(e)?;
        let block = t.elapsed().as_secs_f64();
        self.engine.touch_weights();
        Ok(Some((step, KEEP_WARM_ROWS, block)))
    }
    fn side_slots(&self) -> usize {
        if self.side.is_some() {
            self.engine.reserved()
        } else {
            0
        }
    }
    fn side_valid(&self) -> bool {
        self.side.as_ref().is_some_and(|s| s.valid)
    }
    fn side_save(&mut self) -> Result<usize> {
        let side = self.side.as_mut().ok_or("no side state (--side-slot)")?;
        self.engine
            .snapshot_into(&mut side.state)
            .map_err(|e| e.to_string())?;
        (side.pos, side.valid) = (self.engine.pos(), true);
        Ok(side.state.byte_size())
    }
    // The live state into the spare, the one aside restored, the two exchanged.
    fn side_swap(&mut self) -> Result<usize> {
        let side = self.side.as_mut().ok_or("no side state (--side-slot)")?;
        if !side.valid {
            return Err("no state kept aside".into());
        }
        let live = self.engine.pos();
        self.engine
            .snapshot_into(&mut side.spare)
            .map_err(|e| e.to_string())?;
        self.engine
            .restore(&side.state, side.pos)
            .map_err(|e| e.to_string())?;
        std::mem::swap(&mut side.state, &mut side.spare);
        side.pos = live;
        Ok(side.state.byte_size() + side.spare.byte_size())
    }
    fn kv_fingerprint(&self) -> Option<String> {
        self.engine.kv_fingerprint()
    }
    fn kv_save(&mut self, n: usize) -> Result<Vec<u8>> {
        self.engine.kv_save(n).map_err(|e| e.to_string())
    }
    fn kv_load(&mut self, n: usize, bytes: &[u8]) -> Result<()> {
        self.engine.kv_load(n, bytes).map_err(|e| e.to_string())
    }
    fn snapshot_slots(&self) -> usize {
        self.snapshots.len()
    }
    fn snapshot_bytes(&self) -> usize {
        self.snapshot_bytes
    }
    fn snapshot(&mut self, slot: usize) -> Result<()> {
        let n = self.snapshots.len();
        let s = self
            .snapshots
            .get_mut(slot)
            .ok_or_else(|| format!("snapshot {slot} of {n}"))?;
        let r = s.get_or_insert_with(|| self.engine.recurrence_reserve());
        self.engine.recurrence_take(r).map_err(|e| e.to_string())
    }
    fn restore(&mut self, slot: usize) -> Result<usize> {
        let r = self
            .snapshots
            .get(slot)
            .and_then(Option::as_ref)
            .ok_or_else(|| format!("snapshot {slot} was never taken"))?;
        self.engine.recurrence_restore(r).map_err(|e| e.to_string())
    }
    fn lanes(&self) -> usize {
        self.lanes.held.len()
    }
    fn lane(&self) -> usize {
        self.lanes.current
    }
    // The live K/V and the copy of lane `l` exchanged: the positions either holds, the R16 copies
    // of the incoming ones rebuilt (AArch64).
    fn set_lane(&mut self, l: usize) -> Result<()> {
        let lanes = &mut self.lanes;
        if l >= lanes.held.len() {
            return Err(format!("lane {l} of {}", lanes.held.len()));
        }
        if l == lanes.current {
            return Ok(());
        }
        let i = lanes
            .holder
            .iter()
            .position(|&h| h == l)
            .ok_or_else(|| format!("lane {l} has no copy"))?;
        let (live, held) = (self.engine.pos(), lanes.held[l]);
        let t = Instant::now();
        let bytes = self
            .engine
            .lane_swap(&mut lanes.stores[i], live, held)
            .map_err(|e| e.to_string())?;
        eprintln!(
            "[lanes] lane {} ({live} positions) → lane {l} ({held} positions): {:.1} MiB exchanged in {:.1} ms",
            lanes.current,
            bytes as f64 / (1u64 << 20) as f64,
            t.elapsed().as_secs_f64() * 1e3
        );
        lanes.held[lanes.current] = live;
        lanes.holder[i] = lanes.current;
        lanes.current = l;
        Ok(())
    }
}

/// What an encoder answers to the chat routes: it generates nothing.
const NO_CHAT: &str =
    "an embedding or reranking model generates nothing: /v1/embeddings or /v1/rerank";

/// EmbeddingGemma 2's encoder: `/v1/embeddings` of text and images; no chat (no head, no K/V kept
/// between inputs).
impl Backend for crate::engine::embed_gemma::EmbedGemma {
    fn capacity(&self) -> usize {
        crate::engine::embed_gemma::EmbedGemma::capacity(self)
    }
    fn pos(&self) -> usize {
        0
    }
    // No state between two inputs: every one is encoded whole.
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
    fn prefill_until(
        &mut self,
        _ids: &[u32],
        _stop: &mut dyn FnMut() -> bool,
    ) -> Result<Option<u32>> {
        Err(NO_CHAT.into())
    }
    fn generate(
        &mut self,
        _first: u32,
        _max_steps: usize,
        _f: &mut dyn FnMut(u32, &[u16]) -> Option<u32>,
    ) -> Result<usize> {
        Err(NO_CHAT.into())
    }
    fn logits(&self) -> &[u16] {
        &[]
    }
    fn eos(&self) -> Vec<u32> {
        self.c.eos.clone()
    }
    fn embedding_dim(&self) -> Option<usize> {
        Some(crate::engine::embed_gemma::EmbedGemma::embedding_dim(self))
    }
    fn embed(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        crate::engine::embed_gemma::EmbedGemma::embed(self, ids, &[]).map_err(|e| e.to_string())
    }
    fn embed_media(&mut self, ids: &[u32], media: &[herbert_serve::Medium]) -> Result<Vec<f32>> {
        use crate::engine::embed_gemma::Medium as M;
        // The recordings decoded first (WAV, channels averaged).
        let waves: Vec<Option<(Vec<f32>, usize)>> = media
            .iter()
            .map(|m| match m {
                herbert_serve::Medium::Audio(bytes) => {
                    crate::engine::audio::read_wav_bytes(bytes, "input_audio")
                        .map(Some)
                        .map_err(|e| e.to_string())
                }
                herbert_serve::Medium::Image(_) => Ok(None),
            })
            .collect::<Result<_>>()?;
        let ms: Vec<M> = media
            .iter()
            .zip(&waves)
            .map(|(m, w)| match (m, w) {
                (herbert_serve::Medium::Image(im), _) => M::Image {
                    rgb: &im.rgb,
                    h: im.h,
                    w: im.w,
                },
                (_, Some((wave, rate))) => M::Audio { wave, rate: *rate },
                (herbert_serve::Medium::Audio(_), None) => unreachable!("decoded above"),
            })
            .collect();
        crate::engine::embed_gemma::EmbedGemma::embed(self, ids, &ms).map_err(|e| e.to_string())
    }
    fn embed_media_kinds(&self) -> (bool, bool) {
        self.media_kinds()
    }
    // The text inputs of a request end to end in blocks of `--ctx` rows (each row attends to its
    // own input's), the same bits as one at a time; an input past the window alone.
    fn embed_batch(&mut self, inputs: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        crate::engine::embed_gemma::EmbedGemma::embed_batch(self, inputs).map_err(|e| e.to_string())
    }
}

/// XLM-RoBERTa's encoder: `/v1/embeddings` (bge-m3) or `/v1/rerank` (the reranker); no chat.
impl Backend for crate::engine::xlmr::XlmrEngine {
    fn capacity(&self) -> usize {
        crate::engine::xlmr::XlmrEngine::capacity(self)
    }
    fn pos(&self) -> usize {
        0
    }
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
    fn prefill_until(
        &mut self,
        _ids: &[u32],
        _stop: &mut dyn FnMut() -> bool,
    ) -> Result<Option<u32>> {
        Err(NO_CHAT.into())
    }
    fn generate(
        &mut self,
        _first: u32,
        _max_steps: usize,
        _f: &mut dyn FnMut(u32, &[u16]) -> Option<u32>,
    ) -> Result<usize> {
        Err(NO_CHAT.into())
    }
    fn logits(&self) -> &[u16] {
        &[]
    }
    fn embed(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        if self.c.kind != crate::engine::xlmr::Kind::Embedding {
            return Err("a reranker gives no embedding".into());
        }
        self.encode_cls(ids).map_err(|e| e.to_string())
    }
    fn embedding_dim(&self) -> Option<usize> {
        crate::engine::xlmr::XlmrEngine::embedding_dim(self)
    }
    fn score(&mut self, ids: &[u32]) -> Result<f32> {
        crate::engine::xlmr::XlmrEngine::score(self, ids).map_err(|e| e.to_string())
    }
    // The inputs of a request end to end in blocks of `--ctx` rows (each row attends to its own
    // input's), the same bits as one at a time.
    fn embed_batch(&mut self, inputs: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        if self.c.kind != crate::engine::xlmr::Kind::Embedding {
            return Err("a reranker gives no embedding".into());
        }
        self.encode_cls_batch(inputs).map_err(|e| e.to_string())
    }
    fn score_batch(&mut self, inputs: &[&[u32]]) -> Result<Vec<f32>> {
        crate::engine::xlmr::XlmrEngine::score_batch(self, inputs).map_err(|e| e.to_string())
    }
}
