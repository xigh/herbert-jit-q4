//! EmbeddingGemma 2 (`google/embeddinggemma-2`, `model_type` `embedding_gemma2`) on the CPU: the
//! encoder of `modeling_embedding_gemma2` in one generated program (`jit/schedule/encoder.rs`), one
//! input at a time, all its rows in one block; the images through Gemma 4's vision tower
//! (`engine/vision.rs`, the checkpoint's own weights, bf16).
//!
//! The input embeddings `x₀`: a token's row of `embed_tokens` times `√hidden` (in f32, rounded to
//! bf16), or a soft token of an image as the tower's embedder made it, between `<|image>` and
//! `<image|>` as the processor expands a `<|image|>` placeholder. The program runs the layers and
//! the final norm of every row; then the host takes the mean of the rows in f64
//! (sentence-transformers' `1_Pooling`: `mean`, the special tokens included) and projects it with
//! `embedding_projection` (512 → 768, in the model: projecting each row then averaging gives the
//! same vector). The server normalizes.
//!
//! The projections are the text's GEMMs: bf16 by default, 4-bit under `--q4` (int8 under `--int8`);
//! the vision tower stays bf16 (its weights are not on a 4-bit grid). Positions: `--ctx` caps one
//! input, at most 8 192 rows. A sliding layer sees `|q − k| ≤ 512`: an input of more than 513 rows
//! runs a second program, mounted at the first one, whose sliding layers bound their keys on both
//! sides (`AttentionSpec::ahead`).

use crate::core::avec::AVec;
use crate::core::error::{HerbertError, Result};
use crate::engine::tensors::{from_bf16, to_bf16, Tensors};
use crate::engine::tuning::Tuning;
use crate::jit::execution::build::Mounted;
use crate::jit::model::apart::Apart;
use crate::jit::schedule::encoder::{
    lower_gemma_encoder, GemmaEncoderBuffers, GemmaEncoderLayer, GemmaEncoderShape,
    GemmaEncoderWeights,
};
use crate::jit::Standalone;
use crate::model::PackedBf16R16Weight;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The positions of one input at most: the model's context.
pub const EMBED_CONTEXT: usize = 8192;

/// The fields of `config.json` the encoder reads (`text_config`, and the media tokens at the root).
#[derive(Clone, Debug)]
pub struct EmbedGemmaConfig {
    pub hidden: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub per_layer: usize,
    pub embedding_dim: usize,
    pub vocab: usize,
    pub window: usize,
    pub eps: f32,
    /// Per layer: head size, K/V heads, sliding.
    pub layers: Vec<(usize, usize, bool)>,
    pub sliding_theta: f64,
    pub full_theta: f64,
    pub image_token: Option<u32>,
    pub boi: u32,
    pub eoi: u32,
    pub audio_token: Option<u32>,
    pub boa: u32,
    pub eoa: u32,
    pub eos: Vec<u32>,
}

/// Whether `dir` holds an EmbeddingGemma 2 checkpoint.
pub fn detect(dir: &Path) -> bool {
    let v: serde_json::Value = std::fs::read(dir.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    v["model_type"].as_str() == Some("embedding_gemma2")
}

impl EmbedGemmaConfig {
    pub fn from_dir(dir: &Path) -> Result<EmbedGemmaConfig> {
        let root: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let t = &root["text_config"];
        let n = |v: &serde_json::Value, key: &str| {
            v[key]
                .as_u64()
                .map(|x| x as usize)
                .ok_or_else(|| HerbertError::Config(format!("config.json: no {key}")))
        };
        let (hd, kvh, n_layers) = (
            n(t, "head_dim")?,
            n(t, "num_key_value_heads")?,
            n(t, "num_hidden_layers")?,
        );
        let types: Vec<&str> = t["layer_types"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
            .unwrap_or_default();
        if types.len() != n_layers {
            return Err(HerbertError::Config(
                "config.json: layer_types does not list every layer".into(),
            ));
        }
        let mut layers = Vec::with_capacity(n_layers);
        for (l, ty) in types.iter().enumerate() {
            let over = &t["per_layer_config"][format!("{l:02}")];
            let lhd = over["head_dim"].as_u64().map_or(hd, |x| x as usize);
            let lkv = over["num_key_value_heads"]
                .as_u64()
                .map_or(kvh, |x| x as usize);
            layers.push((lhd, lkv, *ty == "sliding_attention"));
        }
        let theta = |ty: &str| {
            t["rope_parameters"][ty]["rope_theta"]
                .as_f64()
                .ok_or_else(|| HerbertError::Config(format!("config.json: no rope_theta for {ty}")))
        };
        let id = |key: &str| root[key].as_u64().map(|x| x as u32);
        let eos = match &t["eos_token_id"] {
            serde_json::Value::Array(a) => a
                .iter()
                .filter_map(serde_json::Value::as_u64)
                .map(|x| x as u32)
                .collect(),
            v => v.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        };
        if let Some(a) = t["hidden_activation"]
            .as_str()
            .filter(|a| *a != "gelu_pytorch_tanh")
        {
            return Err(HerbertError::Config(format!(
                "hidden_activation {a:?}: the encoder computes gelu_pytorch_tanh"
            )));
        }
        Ok(EmbedGemmaConfig {
            hidden: n(t, "hidden_size")?,
            heads: n(t, "num_attention_heads")?,
            intermediate: n(t, "intermediate_size")?,
            per_layer: n(t, "hidden_size_per_layer_input")?,
            embedding_dim: n(t, "embedding_dim")?,
            vocab: n(t, "vocab_size")?,
            window: n(t, "sliding_window")?,
            eps: t["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            layers,
            sliding_theta: theta("sliding_attention")?,
            full_theta: theta("full_attention")?,
            image_token: id("image_token_id"),
            boi: id("boi_token_id").unwrap_or(255999),
            eoi: id("eoi_token_id").unwrap_or(258882),
            audio_token: id("audio_token_id"),
            boa: id("boa_token_id").unwrap_or(256000),
            eoa: id("eoa_token_index")
                .or(id("eoa_token_id"))
                .unwrap_or(258883),
            eos,
        })
    }
}

/// One medium of an input, in the order of its placeholder.
pub enum Medium<'a> {
    /// `h` rows of `w` RGB pixels.
    Image { rgb: &'a [u8], h: usize, w: usize },
    /// A mono waveform and its rate.
    Audio { wave: &'a [f32], rate: usize },
}

struct OwnedLayer {
    norms: [AVec<u16>; 7],
    linears: Vec<PackedBf16R16Weight>,
    scalar: f32,
}

pub struct EmbedGemma {
    pub c: EmbedGemmaConfig,
    shape: GemmaEncoderShape,
    // The programs hold the addresses of everything below them: declared first, dropped first.
    short: Mounted,
    long: Option<Mounted>,
    tower: Option<crate::engine::vision::Tower>,
    audio_tower: Option<crate::engine::audio::AudioTower>,
    buffers: Box<GemmaEncoderBuffers>,
    layers: Vec<OwnedLayer>,
    ple_model: PackedBf16R16Weight,
    ple_norm: AVec<u16>,
    final_norm: AVec<u16>,
    /// `embedding_projection`, `[embedding_dim][hidden]`, f32.
    proj: Vec<f32>,
    /// The RoPE tables of positions `0..rows` per head size, copied into the buffers' per row
    /// (each row's place in its own input).
    rope: Vec<(usize, Vec<u16>, Vec<u16>)>,
    /// The inputs of one block at most (`HERBERT_ENCODER_BATCH`, 256 by default; 1: one input at a
    /// time).
    pub batch: usize,
    t: Tensors,
    exec: Standalone,
    workers: usize,
    dir: PathBuf,
    /// `rows` words the frames point at.
    tokens: Vec<u32>,
    /// `HERBERT_ENCODER_TIMES=1`: the times of each input on stderr.
    times: bool,
    /// `HERBERT_ENCODER_PROFILE=1`: the encoder's time by operation on stderr.
    profile: bool,
    pub load_seconds: f64,
    /// `--q4` (or `--int8`) took the projections.
    pub quantized: bool,
}

/// The `cos`, `sin` tables of `theta` for a head of `hd`, positions `0..rows`, `[row][hd / 2]` in
/// bf16: the angle `p · θ^(-2i/hd)` in f32 (the inverse frequency in f64, then f32, as
/// transformers), its cosine and sine through f64.
fn rope_tables(theta: f64, hd: usize, rows: usize, cos: &mut [u16], sin: &mut [u16]) {
    let inv: Vec<f32> = (0..hd / 2)
        .map(|i| (1.0 / theta.powf((2 * i) as f64 / hd as f64)) as f32)
        .collect();
    for p in 0..rows {
        for (i, &f) in inv.iter().enumerate() {
            let a = p as f32 * f;
            cos[p * hd / 2 + i] = to_bf16((a as f64).cos() as f32);
            sin[p * hd / 2 + i] = to_bf16((a as f64).sin() as f32);
        }
    }
}

/// The encoder's program over `buffers` (`windowed`: the sliding layers bound their keys).
#[allow(clippy::too_many_arguments)]
fn mount(
    shape: &GemmaEncoderShape,
    layers: &[OwnedLayer],
    ple_model: &PackedBf16R16Weight,
    ple_norm: &[u16],
    final_norm: &[u16],
    buffers: &mut GemmaEncoderBuffers,
    workers: usize,
    spin: Option<u32>,
    windowed: bool,
    trace: bool,
) -> Result<Mounted> {
    let weights = GemmaEncoderWeights {
        ple_model: &ple_model.data_r16,
        ple_norm,
        final_norm,
        layers: layers
            .iter()
            .map(|l| GemmaEncoderLayer {
                norms: [
                    &l.norms[0],
                    &l.norms[1],
                    &l.norms[2],
                    &l.norms[3],
                    &l.norms[4],
                ],
                q_norm: &l.norms[5],
                k_norm: &l.norms[6],
                linears: std::array::from_fn(|i| &l.linears[i].data_r16[..]),
                scalar: l.scalar,
            })
            .collect(),
    };
    lower_gemma_encoder(shape, &weights, buffers, workers, spin, windowed, trace)
        .map_err(|e| HerbertError::Backend(format!("embed: {e}")))
}

/// A token's row of `embed_tokens` times `scale`, rounded to bf16, onto `x0`.
fn token_row(
    x0: &mut Vec<u16>,
    t: &Tensors,
    vocab: usize,
    h: usize,
    scale: f32,
    id: u32,
) -> Result<()> {
    if id as usize >= vocab {
        return Err(HerbertError::Backend(format!(
            "token {id} past the vocabulary of {vocab}"
        )));
    }
    x0.extend(
        t.f32_range("embed_tokens.weight", id as usize * h, h)?
            .into_iter()
            .map(|v| to_bf16(v * scale)),
    );
    Ok(())
}

impl EmbedGemma {
    /// Read, pack and mount the encoder of `dir` for inputs of `capacity` rows at most, on
    /// `threads` workers, with the tuning's knobs (`--q4`, `--int8`).
    pub fn load(
        dir: &Path,
        threads: usize,
        capacity: usize,
        tuning: &Tuning,
    ) -> Result<EmbedGemma> {
        if !crate::jit::engine_supported() || cfg!(target_arch = "riscv64") {
            return Err(HerbertError::Backend(
                "EmbeddingGemma 2: the encoder needs x86-64 (AVX2 or AVX-512) or AArch64".into(),
            ));
        }
        crate::pool::power::opt_out_of_power_throttling();
        tuning.install()?;
        if threads > 0 {
            crate::pool::thread_pool::configure_global_pool(threads)
                .map_err(|e| HerbertError::Backend(format!("thread pool: {e}")))?;
        }
        let t0 = Instant::now();
        let dir = crate::model::loader::resolve_model_dir(dir)?;
        let c = EmbedGemmaConfig::from_dir(&dir)?;
        if c.per_layer != c.hidden || c.hidden % 32 != 0 || c.intermediate % 32 != 0 {
            return Err(HerbertError::Config(
                "an EmbeddingGemma 2 geometry this engine does not serve".into(),
            ));
        }
        let t = Tensors::open(&dir, &["language_model.", "model.language_model."])?;
        let (h, nh, inter, pl) = (c.hidden, c.heads, c.intermediate, c.per_layer);
        let n_layers = c.layers.len();
        let pack = |name: &str, n: usize, k: usize| -> Result<PackedBf16R16Weight> {
            PackedBf16R16Weight::pack_from_row_major_r16_only(&t.bf16(name, n * k)?, n, k)
        };
        let vec16 = |name: &str, len: usize| -> Result<AVec<u16>> {
            Ok(AVec::from_vec(t.bf16(name, len)?))
        };
        let mut layers = Vec::with_capacity(n_layers);
        for (l, &(hd, kvh, _)) in c.layers.iter().enumerate() {
            let p = format!("layers.{l}.");
            let (qd, kvd) = (nh * hd, kvh * hd);
            let norms = [
                vec16(&format!("{p}input_layernorm.weight"), h)?,
                vec16(&format!("{p}post_attention_layernorm.weight"), h)?,
                vec16(&format!("{p}pre_feedforward_layernorm.weight"), h)?,
                vec16(&format!("{p}post_feedforward_layernorm.weight"), h)?,
                vec16(&format!("{p}ple_block.post_per_layer_input_norm.weight"), h)?,
                vec16(&format!("{p}self_attn.q_norm.weight"), hd)?,
                vec16(&format!("{p}self_attn.k_norm.weight"), hd)?,
            ];
            let mut linears = Vec::with_capacity(9);
            for (name, n, k) in [
                ("self_attn.q_proj", qd, h),
                ("self_attn.k_proj", kvd, h),
                ("self_attn.v_proj", kvd, h),
                ("self_attn.o_proj", h, qd),
                ("mlp.gate_proj", inter, h),
                ("mlp.up_proj", inter, h),
                ("mlp.down_proj", h, inter),
                ("ple_block.per_layer_input_gate", pl, h),
                ("ple_block.per_layer_projection", h, pl),
            ] {
                linears.push(pack(&format!("{p}{name}.weight"), n, k)?);
            }
            let scalar = t.f32(&format!("{p}layer_scalar"), 1)?[0];
            layers.push(OwnedLayer {
                norms,
                linears,
                scalar,
            });
        }
        let ple_model = pack("ple.per_layer_model_projection.weight", n_layers * pl, h)?;
        let ple_norm = vec16("ple.per_layer_projection_norm.weight", pl)?;
        let final_norm = vec16("norm.weight", h)?;
        let proj = t.f32("embedding_projection.weight", c.embedding_dim * h)?;
        let quantized = crate::jit::int8::gemm(h, h);
        let rows = capacity.clamp(1, EMBED_CONTEXT);
        let shape = GemmaEncoderShape {
            hidden: h,
            heads: nh,
            intermediate: inter,
            per_layer: pl,
            layers: c.layers.clone(),
            window: c.window,
            eps: c.eps,
            rows,
        };
        let workers = crate::pool::thread_pool::global_pool().num_workers();
        let mut buffers = Box::new(GemmaEncoderBuffers::new(&shape, workers));
        let rope: Vec<(usize, Vec<u16>, Vec<u16>)> = buffers
            .tables
            .iter()
            .map(|(hd, _, _)| {
                let theta = if c.layers.iter().any(|l| l.0 == *hd && l.2) {
                    c.sliding_theta
                } else {
                    c.full_theta
                };
                let (mut cos, mut sin) = (vec![0u16; rows * hd / 2], vec![0u16; rows * hd / 2]);
                rope_tables(theta, *hd, rows, &mut cos, &mut sin);
                (*hd, cos, sin)
            })
            .collect();
        let exec = Standalone::new(tuning.executor_options());
        // `HERBERT_ENCODER_PROFILE=1` (x86-64): the programs stamp their nodes, their times summed
        // by operation per input.
        let profile =
            cfg!(target_arch = "x86_64") && std::env::var_os("HERBERT_ENCODER_PROFILE").is_some();
        let t_read = t0.elapsed().as_secs_f64();
        let short = mount(
            &shape,
            &layers,
            &ple_model,
            &ple_norm,
            &final_norm,
            &mut buffers,
            workers,
            exec.apart_spin(),
            false,
            profile,
        )?;
        let load_seconds = t0.elapsed().as_secs_f64();
        eprintln!(
            "[embed] EmbeddingGemma 2: {} layers, hidden {h}, {} projections; read and packed in {t_read:.2} s, mounted in {:.2} s; {:.0} MiB of buffers for {rows} rows; {} bytes of code",
            n_layers,
            if quantized { if crate::jit::int8::q4_for(h) { "Q4" } else { "int8" } } else { "bf16" },
            load_seconds - t_read,
            buffers.bytes() as f64 / (1u64 << 20) as f64,
            short.program.code().len()
        );
        Ok(EmbedGemma {
            short,
            long: None,
            tower: None,
            audio_tower: None,
            c,
            shape,
            buffers,
            layers,
            ple_model,
            ple_norm,
            final_norm,
            proj,
            rope,
            batch: std::env::var("HERBERT_ENCODER_BATCH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(256),
            t,
            exec,
            workers,
            dir,
            tokens: vec![0; rows],
            times: std::env::var_os("HERBERT_ENCODER_TIMES").is_some(),
            profile,
            load_seconds,
            quantized,
        })
    }

    fn mount_long(&mut self) -> Result<Mounted> {
        let spin = self.exec.apart_spin();
        mount(
            &self.shape,
            &self.layers,
            &self.ple_model,
            &self.ple_norm,
            &self.final_norm,
            &mut self.buffers,
            self.workers,
            spin,
            true,
            self.profile,
        )
    }

    /// The rows of one input at most.
    pub fn capacity(&self) -> usize {
        self.shape.rows
    }

    pub fn embedding_dim(&self) -> usize {
        self.c.embedding_dim
    }

    /// The media this checkpoint takes: `(images, recordings)`.
    pub fn media_kinds(&self) -> (bool, bool) {
        let audio = self.c.audio_token.is_some()
            && crate::engine::audio::AudioConfig::of(&self.dir).is_some()
            && !cfg!(target_arch = "riscv64");
        (
            self.c.image_token.is_some()
                && crate::engine::vision::VisionConfig::of(&self.dir).is_some_and(|c| !c.unified),
            audio,
        )
    }

    /// The soft tokens of one image, `[soft][hidden]` bf16, through the vision tower (mounted at
    /// the first image).
    fn image_rows(&mut self, rgb: &[u8], h: usize, w: usize) -> Result<Vec<u16>> {
        if self.tower.is_none() {
            let cfg = crate::engine::vision::VisionConfig::of(&self.dir).ok_or_else(|| {
                HerbertError::Backend(
                    "this checkpoint has no vision tower this engine serves".into(),
                )
            })?;
            let t0 = Instant::now();
            self.tower = Some(crate::engine::vision::Tower::load(
                &self.dir,
                "",
                cfg,
                self.c.hidden,
                &self.exec,
                self.workers,
            )?);
            eprintln!(
                "[embed] vision tower ready in {:.2} s",
                t0.elapsed().as_secs_f64()
            );
        }
        let tower = self.tower.as_mut().expect("the tower");
        let (gw, gh, soft, prep, tower_s) = tower.encode(&mut self.exec, rgb, h, w)?;
        if self.times {
            eprintln!("[embed] image {w}×{h} → {gw}×{gh} patches → {soft} soft tokens: preparation {:.1} ms, tower {:.1} ms", prep * 1e3, tower_s * 1e3);
        }
        Ok(tower.table()[..soft * self.c.hidden].to_vec())
    }

    /// The soft tokens of one recording (mono, `rate` Hz), `[soft][hidden]` bf16, through the audio
    /// tower (mounted at the first recording). The recording goes to 16 kHz by the resampler
    /// closest found to sentence-transformers' reading (librosa's soxr:
    /// `resample_kaiser_best`); past 30 s it is refused.
    fn audio_rows(&mut self, wave: &[f32], rate: usize) -> Result<Vec<u16>> {
        if self.audio_tower.is_none() {
            let cfg = crate::engine::audio::AudioConfig::of(&self.dir).ok_or_else(|| {
                HerbertError::Backend(
                    "this checkpoint has no audio tower this engine serves".into(),
                )
            })?;
            let t0 = Instant::now();
            self.audio_tower = Some(crate::engine::audio::AudioTower::load(
                &self.dir,
                "",
                cfg,
                self.c.hidden,
                &mut self.exec,
                self.workers,
            )?);
            eprintln!(
                "[embed] audio tower ready in {:.2} s",
                t0.elapsed().as_secs_f64()
            );
        }
        use crate::engine::audio::{MAX_SAMPLES, SAMPLE_RATE};
        // The length checked before the resampling, which would otherwise run over the whole
        // recording only to refuse it.
        if (wave.len() as u64 * SAMPLE_RATE as u64).div_ceil(rate as u64) > MAX_SAMPLES as u64 {
            return Err(HerbertError::Backend(format!(
                "{:.1} s of audio: this engine's audio tower takes {} s at most",
                wave.len() as f64 / rate as f64,
                MAX_SAMPLES / SAMPLE_RATE
            )));
        }
        let t0 = Instant::now();
        let wave = crate::engine::audio::resample_kaiser_best(wave, rate, SAMPLE_RATE)
            .map_err(HerbertError::Backend)?;
        let t_rs = t0.elapsed().as_secs_f64();
        let tower = self.audio_tower.as_mut().expect("the tower");
        let (frames, soft, prep, tower_s) = tower.encode(&mut self.exec, &wave)?;
        if self.times {
            eprintln!(
                "[embed] audio of {:.2} s at {rate} Hz → {frames} frames → {soft} soft tokens: resampling {:.1} ms, preparation {:.1} ms, tower {:.1} ms",
                wave.len() as f64 / crate::engine::audio::SAMPLE_RATE as f64,
                t_rs * 1e3,
                prep * 1e3,
                tower_s * 1e3
            );
        }
        Ok(tower.table()[..soft * self.c.hidden].to_vec())
    }

    /// The input embeddings of one input (`ids`, its media), `[rows][hidden]`, and the seconds of
    /// its media.
    fn input_rows(&mut self, ids: &[u32], media: &[Medium]) -> Result<(Vec<u16>, f64)> {
        let h = self.c.hidden;
        let n_img = ids
            .iter()
            .filter(|&&t| Some(t) == self.c.image_token)
            .count();
        let n_aud = ids
            .iter()
            .filter(|&&t| Some(t) == self.c.audio_token)
            .count();
        let images: Vec<(&[u8], usize, usize)> = media
            .iter()
            .filter_map(|m| {
                if let Medium::Image { rgb, h, w } = m {
                    Some((*rgb, *h, *w))
                } else {
                    None
                }
            })
            .collect();
        let audios: Vec<(&[f32], usize)> = media
            .iter()
            .filter_map(|m| {
                if let Medium::Audio { wave, rate } = m {
                    Some((*wave, *rate))
                } else {
                    None
                }
            })
            .collect();
        if n_img != images.len() || n_aud != audios.len() {
            return Err(HerbertError::Backend(format!("{n_img} <|image|> and {n_aud} <|audio|> placeholders for {} images and {} recordings", images.len(), audios.len())));
        }
        // The input embeddings, row by row.
        let scale = (h as f32).sqrt();
        let mut x0: Vec<u16> = Vec::with_capacity(ids.len() * h);
        let mut images = images.into_iter();
        let mut audios = audios.into_iter();
        let (vocab, boi, eoi, boa, eoa) =
            (self.c.vocab, self.c.boi, self.c.eoi, self.c.boa, self.c.eoa);
        let mut t_media = 0.0;
        for &id in ids {
            if Some(id) == self.c.image_token {
                let (rgb, ih, iw) = images.next().expect("counted");
                let tm = Instant::now();
                let rows = self.image_rows(rgb, ih, iw)?;
                t_media += tm.elapsed().as_secs_f64();
                token_row(&mut x0, &self.t, vocab, h, scale, boi)?;
                x0.extend_from_slice(&rows);
                token_row(&mut x0, &self.t, vocab, h, scale, eoi)?;
            } else if Some(id) == self.c.audio_token {
                let (wave, rate) = audios.next().expect("counted");
                let tm = Instant::now();
                let rows = self.audio_rows(wave, rate)?;
                t_media += tm.elapsed().as_secs_f64();
                token_row(&mut x0, &self.t, vocab, h, scale, boa)?;
                x0.extend_from_slice(&rows);
                token_row(&mut x0, &self.t, vocab, h, scale, eoa)?;
            } else {
                token_row(&mut x0, &self.t, vocab, h, scale, id)?;
            }
            if x0.len() > self.shape.rows * h {
                return Err(HerbertError::Backend(format!(
                    "more than {} positions",
                    self.shape.rows
                )));
            }
        }
        if x0.is_empty() {
            return Err(HerbertError::Backend("an empty input".into()));
        }
        Ok((x0, t_media))
    }

    /// The block of `lens` inputs end to end, their rows already in `input`: each row's RoPE angles
    /// those of its place in its input, its keys its input's; one input past the window runs
    /// the windowed program alone. Its seconds.
    fn run_block(&mut self, lens: &[usize]) -> Result<f64> {
        let align = crate::engine::xlmr::encoder_align();
        let rows: usize = lens.iter().fold(0, |r, &n| r.next_multiple_of(align) + n);
        let wide = lens.iter().any(|&n| n > self.c.window + 1);
        if wide && lens.len() > 1 {
            return Err(HerbertError::Backend(
                "inputs past the window run one at a time".into(),
            ));
        }
        let windowed = wide;
        let mut r = 0usize;
        for &n in lens {
            // The rows up to the input's aligned start (`encoder_align`): in the previous input's
            // segment.
            let to = r.next_multiple_of(align);
            for i in r..to {
                self.buffers.segments[2 * i] = self.buffers.segments[2 * r - 2];
                self.buffers.segments[2 * i + 1] = self.buffers.segments[2 * r - 1];
            }
            r = to;
            for ((hd, cos, sin), (_, mc, ms)) in self.buffers.tables.iter_mut().zip(&self.rope) {
                let half = *hd / 2;
                cos[r * half..(r + n) * half].copy_from_slice(&mc[..n * half]);
                sin[r * half..(r + n) * half].copy_from_slice(&ms[..n * half]);
            }
            for i in r..r + n {
                self.buffers.segments[2 * i] = r as u32;
                self.buffers.segments[2 * i + 1] = (r + n) as u32;
            }
            r += n;
        }
        let t_run = Instant::now();
        if windowed {
            if self.long.is_none() {
                let tm = Instant::now();
                self.long = Some(self.mount_long()?);
                eprintln!(
                    "[embed] the program of the inputs past {} rows mounted in {:.2} s",
                    self.c.window + 1,
                    tm.elapsed().as_secs_f64()
                );
            }
            let program = self.long.as_mut().expect("mounted");
            if self.profile {
                self.exec.reserve_apart_trace(program)?;
            }
            self.exec.run_apart(program, &self.tokens[..rows])?;
            if self.profile {
                crate::engine::vision::print_profile_as(
                    "embed",
                    &self.exec,
                    program,
                    t_run.elapsed().as_secs_f64(),
                );
            }
        } else {
            if self.profile {
                self.exec.reserve_apart_trace(&self.short)?;
            }
            self.exec.run_apart(&mut self.short, &self.tokens[..rows])?;
            if self.profile {
                crate::engine::vision::print_profile_as(
                    "embed",
                    &self.exec,
                    &self.short,
                    t_run.elapsed().as_secs_f64(),
                );
            }
        }
        Ok(t_run.elapsed().as_secs_f64())
    }

    /// The vector of the input at rows `start..start + rows` of the last block: the mean of its
    /// normed rows in f64, then the projection.
    fn pooled(&self, start: usize, rows: usize) -> Result<Vec<f32>> {
        let h = self.c.hidden;
        let mut mean = vec![0f64; h];
        let mut bad = 0usize;
        for row in self.buffers.out[start * h..(start + rows) * h].chunks_exact(h) {
            for (m, &v) in mean.iter_mut().zip(row) {
                let x = from_bf16(v);
                bad += usize::from(!x.is_finite());
                *m += x as f64;
            }
        }
        if bad > 0 {
            return Err(HerbertError::Backend(format!(
                "{bad} values of the last layer are not finite"
            )));
        }
        mean.iter_mut().for_each(|v| *v /= rows as f64);
        Ok(self
            .proj
            .chunks_exact(h)
            .map(|w| w.iter().zip(&mean).map(|(a, b)| *a as f64 * b).sum::<f64>() as f32)
            .collect())
    }

    /// The vector of one input, pooled (the mean of its rows) and projected, not normalized: `ids`
    /// its tokens (`<bos>` … `<eos>` as the tokenizer writes them), each `<|image|>`
    /// placeholder taken by the next image of `media` and expanded as the processor does
    /// (`<|image>`, its soft tokens, `<image|>`).
    pub fn embed(&mut self, ids: &[u32], media: &[Medium]) -> Result<Vec<f32>> {
        let t0 = Instant::now();
        let h = self.c.hidden;
        let (x0, t_media) = self.input_rows(ids, media)?;
        let rows = x0.len() / h;
        self.buffers.input[..rows * h].copy_from_slice(&x0);
        let t_run = self.run_block(&[rows])?;
        let out = self.pooled(0, rows)?;
        if self.times {
            eprintln!(
                "[embed] {} tokens, {rows} rows: media {:.1} ms, encoder {:.1} ms, total {:.1} ms",
                ids.len(),
                t_media * 1e3,
                t_run * 1e3,
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok(out)
    }

    /// [`EmbedGemma::embed`] of several text inputs, in order: those within the window end to end
    /// in blocks of at most [`EmbedGemma::capacity`] rows (each row attending to its own
    /// input's, the same bits as one at a time), a longer one alone.
    pub fn embed_batch(&mut self, inputs: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        let h = self.c.hidden;
        let (cap, most, wide) = (self.shape.rows, self.batch.max(1), self.c.window + 1);
        let mut out = Vec::with_capacity(inputs.len());
        let mut i = 0;
        while i < inputs.len() {
            let t0 = Instant::now();
            let mut lens: Vec<usize> = Vec::new();
            let mut rows = 0usize;
            let mut j = i;
            let align = crate::engine::xlmr::encoder_align();
            while j < inputs.len() && lens.len() < most {
                let (x0, _) = self.input_rows(inputs[j], &[])?;
                let n = x0.len() / h;
                let alone = n > wide;
                // Inputs start on multiples of `align` rows (`encoder_align`), zeros before.
                let at = rows.next_multiple_of(align);
                if !lens.is_empty() && (alone || at + n > cap) {
                    break;
                }
                self.buffers.input[rows * h..at * h].fill(0);
                self.buffers.input[at * h..(at + n) * h].copy_from_slice(&x0);
                lens.push(n);
                rows = at + n;
                j += 1;
                if alone {
                    break;
                }
            }
            let t_run = self.run_block(&lens)?;
            let mut r = 0usize;
            for &n in &lens {
                r = r.next_multiple_of(align);
                out.push(self.pooled(r, n)?);
                r += n;
            }
            if self.times {
                eprintln!(
                    "[embed] {} inputs, {rows} rows: encoder {:.1} ms, total {:.1} ms",
                    lens.len(),
                    t_run * 1e3,
                    t0.elapsed().as_secs_f64() * 1e3
                );
            }
            i = j;
        }
        Ok(out)
    }
}
