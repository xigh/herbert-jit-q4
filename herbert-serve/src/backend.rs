//! What an engine gives the server: a state it feeds tokens to and generates from, on its device.
//! Everything else (the template, the tokens, the parser of the answer, the sampler, the HTTP API,
//! the reuse of a prefix) is the server's.
//!
//! The engine stays on the thread that loaded it (the device's handles are not `Send`): the server
//! loads it and runs its worker on a thread of its own.
//!
//! The required methods are what every engine has; the provided ones say "not this engine": a state
//! that cannot be cut back (DeltaNet's) nor kept in snapshots, one lane, no side state, no K/V to
//! write to disk, no keep-warm, a device that is never lost, no image tower, no audio tower.

/// An engine's error, as the server sends it back.
pub type Result<T> = std::result::Result<T, String>;

/// An image tower (Gemma 4's): what the server needs to count the soft tokens of an image before it
/// feeds the prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vision {
    /// The side of a patch, in pixels.
    pub patch: usize,
    /// The side of the pooling window, in patches.
    pub pool: usize,
    /// The soft tokens of one image at most (the processor's `max_soft_tokens`).
    pub max_soft: usize,
}

/// An audio tower or embedder (Gemma 4's): what the server needs to cut a recording and count its
/// soft tokens before it feeds the prompt. The server reads the WAV, mixes it down to mono and
/// resamples it to [`Audio::rate`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Audio {
    /// The sampling rate of the waveform the engine takes (16 kHz).
    pub rate: usize,
    /// The samples of one recording at most (the feature extractor's `max_length`, 30 s): the rest
    /// is cut.
    pub max_samples: usize,
    /// The soft tokens of one recording at most (the processor's `audio_seq_length`).
    pub max_soft: usize,
    /// How the samples make soft tokens.
    pub framing: AudioFraming,
}

/// How a recording's samples make soft tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioFraming {
    /// The conformer of E2B and E4B (`gemma4_audio`): log-mel frames of `frame` samples every
    /// `hop`, the recording left-padded by half a frame, then two convolutions of stride 2
    /// (the processor's `_compute_audio_num_tokens`).
    Mel { frame: usize, hop: usize },
    /// 12B's encoder-free embedder (`gemma4_unified_audio`): one soft token per chunk of this many
    /// raw samples, the last one zero-padded.
    Chunks(usize),
}

impl Audio {
    /// Gemma 4's conformer (E2B, E4B): 16 kHz, 30 s, 750 soft tokens, 20 ms frames every 10 ms.
    pub const GEMMA4: Audio = Audio {
        rate: 16_000,
        max_samples: 480_000,
        max_soft: 750,
        framing: AudioFraming::Mel {
            frame: 320,
            hop: 160,
        },
    };

    /// The soft tokens of a recording of `samples` samples at [`Audio::rate`], once cut to
    /// [`Audio::max_samples`].
    pub fn soft_tokens(&self, samples: usize) -> usize {
        let samples = samples.min(self.max_samples);
        let n = match self.framing {
            AudioFraming::Mel { frame, hop } => {
                let padded = samples + frame / 2;
                if padded < frame + 1 {
                    return 0;
                }
                let frames = (padded - (frame + 1)) / hop + 1;
                frames.div_ceil(2).div_ceil(2)
            }
            AudioFraming::Chunks(n) => samples.div_ceil(n),
        };
        n.min(self.max_soft)
    }
}

/// A medium of an embedding input (`/v1/embeddings`), in the order of its placeholder.
pub enum Medium<'a> {
    /// An image decoded to RGB.
    Image(&'a crate::image::Image),
    /// A recording: the bytes of its WAV file, as sent.
    Audio(&'a [u8]),
}

pub trait Backend: 'static {
    /// The positions of the context (`n_ctx`).
    fn capacity(&self) -> usize;
    /// The tokens the state holds.
    fn pos(&self) -> usize;
    /// The state of an empty conversation.
    fn reset(&mut self) -> Result<()>;
    /// Feeds `ids` at the current position, `stop` asked between two blocks (a client gone): the
    /// greedy token of the last one, `None` when `stop` said so (the state then holds what was
    /// fed, [`Backend::pos`]).
    fn prefill_until(&mut self, ids: &[u32], stop: &mut dyn FnMut() -> bool)
        -> Result<Option<u32>>;
    /// From `first` (fed first), up to `max_steps` tokens: `f` gets the greedy token and the logits
    /// of each step and returns the token to feed, or `None` to end. Returns how many tokens
    /// were fed.
    fn generate(
        &mut self,
        first: u32,
        max_steps: usize,
        f: &mut dyn FnMut(u32, &[u16]) -> Option<u32>,
    ) -> Result<usize>;
    /// The logits of the last token fed, in the engine's 16-bit format (`herbert_chat::half16`).
    fn logits(&self) -> &[u16];

    /// The end-of-turn tokens of the checkpoint's configuration (Gemma's `eos_token_id`); the
    /// family adds its own.
    fn eos(&self) -> Vec<u32> {
        Vec::new()
    }
    /// An embedding model's vector of `ids` (the tokens of one input, the tokenizer's special
    /// tokens included): pooled as the model says (the last token, the mean, the first) and
    /// projected when it has a projection, not normalized (the server does it, after cutting it
    /// to the `dimensions` asked). The state is the engine's to use: the server resets it
    /// after.
    fn embed(&mut self, _ids: &[u32]) -> Result<Vec<f32>> {
        Err("this model gives no embedding".into())
    }
    /// The size of [`Backend::embed`]'s vectors, `None` when the model gives none.
    fn embedding_dim(&self) -> Option<usize> {
        None
    }
    /// [`Backend::embed`] of an input that holds media (EmbeddingGemma 2): each `<|image|>` or
    /// `<|audio|>` placeholder of `ids` (one token each, as the tokenizer writes it) is the
    /// next medium of its kind in `media`, which the engine expands as its processor does.
    /// Without media, [`Backend::embed`].
    fn embed_media(&mut self, ids: &[u32], media: &[Medium]) -> Result<Vec<f32>> {
        if media.is_empty() {
            self.embed(ids)
        } else {
            Err("this model embeds text only".into())
        }
    }
    /// The media [`Backend::embed_media`] takes: `(images, recordings)`.
    fn embed_media_kinds(&self) -> (bool, bool) {
        (false, false)
    }
    /// A reranker's relevance of the pair whose tokens are `ids` (as [`crate::worker`]'s
    /// `rerank_input` writes them): a score where higher is more relevant (a logit, or log-odds
    /// of `yes` against `no`). The server resets the state after.
    fn score(&mut self, _ids: &[u32]) -> Result<f32> {
        Err("this model is not a reranker".into())
    }
    /// [`Backend::embed`] of several inputs of text, in order. By default one by one, the state
    /// reset after each; an engine that runs several inputs at once (XLM-RoBERTa's encoder)
    /// overrides it. The server resets the state after.
    fn embed_batch(&mut self, inputs: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(inputs.len());
        for ids in inputs {
            let r = self.embed(ids);
            self.reset()?;
            out.push(r?);
        }
        Ok(out)
    }
    /// [`Backend::score`] of several pairs, in order. By default one by one, the state reset after
    /// each; an engine that runs several inputs at once overrides it. The server resets the
    /// state after.
    fn score_batch(&mut self, inputs: &[&[u32]]) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(inputs.len());
        for ids in inputs {
            let r = self.score(ids);
            self.reset()?;
            out.push(r?);
        }
        Ok(out)
    }
    /// The image tower, `None` without one.
    fn vision(&self) -> Option<Vision> {
        None
    }
    /// The soft tokens of an `h × w` image when the engine counts them its own way (LFM3-VL's
    /// `smart_resize`), `None` for Gemma 4's rule ([`crate::image::soft_tokens`] over
    /// [`Backend::vision`]).
    fn image_soft_tokens(
        &self,
        _h: usize,
        _w: usize,
    ) -> Option<std::result::Result<usize, String>> {
        None
    }
    /// The image (`rgb`: `h` rows of `w` pixels, 3 bytes each) through the tower, its `soft` tokens
    /// fed at the current position (the `<|image|>` placeholders of its group, as many as
    /// [`crate::image::soft_tokens`] counts).
    fn prefill_image(&mut self, _rgb: &[u8], _h: usize, _w: usize, _soft: usize) -> Result<()> {
        Err("this engine has no image tower".into())
    }
    /// The soft tokens of an `h × w` image and the RoPE positions they take in the language model,
    /// when the engine counts them itself (Qwen's `smart_resize`: `None` says the server's
    /// [`crate::image::soft_tokens`] counts, one position per soft token).
    fn image_tokens(&self, _h: usize, _w: usize) -> Option<Result<(usize, usize)>> {
        None
    }
    /// The RoPE offset of the text fed from the current position on: its K/V position minus its
    /// RoPE position (the positions the images before it saved, Qwen's multimodal RoPE). The
    /// server sets it before each piece it feeds; an engine whose images take one position per
    /// soft token ignores it.
    fn set_rope_shift(&mut self, _shift: usize) -> Result<()> {
        Ok(())
    }
    /// The audio tower, `None` without one.
    fn audio(&self) -> Option<Audio> {
        None
    }
    /// The recording (`wave`: mono, at [`Audio::rate`], cut to [`Audio::max_samples`]) through the
    /// tower, its `soft` tokens fed at the current position as causal tokens (the `<|audio|>`
    /// placeholders of its group, as many as [`Audio::soft_tokens`] counts).
    fn prefill_audio(&mut self, _wave: &[f32], _soft: usize) -> Result<()> {
        Err("this engine has no audio tower".into())
    }
    /// Whether [`Backend::truncate`] cuts the state back to any position (K/V caches). A state that
    /// cannot (DeltaNet's) is reused only by a prompt that extends it, else started again.
    fn can_truncate(&self) -> bool {
        false
    }
    /// The state cut back to its first `pos` tokens: the position it then holds, `pos` or less (a
    /// sliding ring that lost the window before `pos` sends it back to 0).
    fn truncate(&mut self, _pos: usize) -> Result<usize> {
        Err("this engine cannot cut its state back".into())
    }
    /// The positions a prefill block starts on (the attention's query tiles): what a request reuses
    /// is cut to a multiple, and so are the snapshots taken in a prompt, so that its blocks
    /// give the same bytes whatever was reused.
    fn block_align(&self) -> usize {
        1
    }
    /// One token and one small block through the engine at the current position, the conversation's
    /// cache kept, so that the system does not evict the buffers of a server that idles: `(one
    /// token's seconds, the block's rows, its seconds)`, or `None` when there is no room.
    fn keep_warm(&mut self) -> Result<Option<(f64, usize, f64)>> {
        Ok(None)
    }

    /// The positions of the side state (`--side-slot`), 0 without one.
    fn side_slots(&self) -> usize {
        0
    }
    /// Whether the side state holds a conversation.
    fn side_valid(&self) -> bool {
        false
    }
    /// The live state copied aside: the bytes copied.
    fn side_save(&mut self) -> Result<usize> {
        Err("no side state".into())
    }
    /// The live state and the one aside swapped: the bytes copied.
    fn side_swap(&mut self) -> Result<usize> {
        Err("no side state".into())
    }

    /// What the K/V of a prefix depends on in the engine (the executable, the device, its driver,
    /// the model, the tuning, the capacity), `None` when its K/V cannot be written to disk.
    fn kv_fingerprint(&self) -> Option<String> {
        None
    }
    /// The K/V of positions `0..n`.
    fn kv_save(&mut self, _n: usize) -> Result<Vec<u8>> {
        Err("no K/V to save".into())
    }
    /// The K/V of positions `0..n` read back, the state then at `n`.
    fn kv_load(&mut self, _n: usize, _bytes: &[u8]) -> Result<()> {
        Err("no K/V to load".into())
    }

    /// The snapshots of a state that cannot be cut back (DeltaNet's) the engine keeps room for, 0
    /// without: a request whose prompt extends the tokens of one starts from it instead of an
    /// empty conversation.
    fn snapshot_slots(&self) -> usize {
        0
    }
    /// The bytes of one snapshot.
    fn snapshot_bytes(&self) -> usize {
        0
    }
    /// The state that cannot be cut back copied into snapshot `slot`; the K/V caches stay where
    /// they are, in the current lane (positions are kept by not being written over).
    fn snapshot(&mut self, _slot: usize) -> Result<()> {
        Err("no snapshots".into())
    }
    /// Snapshot `slot` back into the state, with the current lane's K/V caches as they are: the
    /// position it stands for.
    fn restore(&mut self, _slot: usize) -> Result<usize> {
        Err("no snapshots".into())
    }
    /// The K/V caches the engine holds, each a conversation's ("lanes"): two conversations that
    /// alternate (an agent's side requests and its conversation) each keep their positions,
    /// their snapshots with them.
    fn lanes(&self) -> usize {
        1
    }
    /// The lane the next tokens are fed in.
    fn lane(&self) -> usize {
        0
    }
    /// The next tokens fed in lane `l`; the state that cannot be cut back is left as it is
    /// (restored or reset after).
    fn set_lane(&mut self, l: usize) -> Result<()> {
        if l == 0 {
            Ok(())
        } else {
            Err(format!("lane {l} of 1"))
        }
    }
}
