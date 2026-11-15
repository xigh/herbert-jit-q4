//! The Gemma 4 audio tower (E2B, E4B) of the engine: its configuration, its
//! weights read from the checkpoint and packed at the first recording, the
//! recording as the processor prepares it, and the four generated programs
//! of `jit/schedule/audio.rs` that turn it into soft tokens.
//!
//! The preparation, on the host: mono,
//! 16 kHz, at most 30 s, left-padded by half a frame, 20 ms frames every
//! 10 ms (periodic Hann window, 512-point FFT in f64), the magnitude through
//! 128 HTK mel filters, `ln(mel + 0.001)`, then bf16. Only the real frames
//! exist: the extractor pads the samples and masks the padding, which leaves
//! the same frames and, through the tower, the same soft tokens. The host then
//! lays out the 3 × 3 patches of the first subsampling layer and the rows the
//! second one gathers (the kernels do the rest).

use crate::core::avec::AVec;
use crate::core::error::{HerbertError, Result};
use crate::engine::packs::to_bf16;
use crate::jit::model::apart::Apart;
use crate::jit::schedule::audio::{
    lower_audio, AudioBuffers, AudioLayerWeights, AudioPrograms, AudioShape, AudioWeights, CONV0_K,
};
use crate::jit::schedule::vision::TowerLinear;
use crate::model::PackedBf16R16Weight;
use std::path::Path;
use std::time::Instant;

/// The processor's sampling rate.
pub const SAMPLE_RATE: usize = 16_000;
/// The extractor's `max_length`: 30 s.
pub const MAX_SAMPLES: usize = 480_000;
const FRAME: usize = 320;
const HOP: usize = 160;
const FFT: usize = 512;
/// Mel bins of a frame.
pub const MELS: usize = 128;
/// Mel frames of the longest recording.
pub const MAX_FRAMES: usize = (MAX_SAMPLES + FRAME / 2 - (FRAME + 1)) / HOP + 1;

/// The tower of `config.json` (`audio_config`), and the media tokens at the root.
#[derive(Clone, Debug)]
pub struct AudioConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub output_proj_dims: usize,
    pub context_left: usize,
    pub context_right: usize,
    pub conv_kernel_size: usize,
    pub conv_channels: [usize; 2],
    pub logit_cap: f32,
    pub residual_weight: f32,
    pub gradient_clipping: f32,
    pub rms_norm_eps: f32,
    pub clipped_linears: bool,
    /// `<|audio|>`, `<|audio>`, `<audio|>`.
    pub audio_token_id: u32,
    pub boa_token_id: u32,
    pub eoa_token_id: u32,
}

impl AudioConfig {
    /// The tower of the checkpoint in `dir`, when it has one this engine serves: no right context,
    /// the attention's window within the sixteen lanes of its kernel, heads of a multiple of
    /// 32, the subsampling of 128 mel bins, and no clamp to `±gradient_clipping` below 1e10.
    pub fn of(dir: &Path) -> Option<AudioConfig> {
        let root: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).ok()?).ok()?;
        let a = root.get("audio_config").filter(|v| v.is_object())?;
        let u = |key: &str| {
            a.get(key)
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
        };
        let f = |key: &str| {
            a.get(key)
                .and_then(serde_json::Value::as_f64)
                .map(|n| n as f32)
        };
        // EmbeddingGemma 2 names the last one `eoa_token_index`.
        let id = |key: &str| {
            root.get(key)
                .or_else(|| root.get(key.replace("_id", "_index").as_str()))
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as u32)
        };
        let ch = a
            .get("subsampling_conv_channels")
            .and_then(serde_json::Value::as_array)?;
        let ch = |i: usize| {
            ch.get(i)
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
        };
        let c = AudioConfig {
            hidden_size: u("hidden_size")?,
            num_layers: u("num_hidden_layers")?,
            num_heads: u("num_attention_heads")?,
            output_proj_dims: u("output_proj_dims")?,
            context_left: u("attention_context_left")?,
            context_right: u("attention_context_right")?,
            conv_kernel_size: u("conv_kernel_size")?,
            conv_channels: [ch(0)?, ch(1)?],
            logit_cap: f("attention_logit_cap")?,
            residual_weight: f("residual_weight")?,
            gradient_clipping: f("gradient_clipping")?,
            rms_norm_eps: a
                .get("rms_norm_eps")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1e-6) as f32,
            clipped_linears: a
                .get("use_clipped_linears")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            audio_token_id: id("audio_token_id")?,
            boa_token_id: id("boa_token_id")?,
            eoa_token_id: id("eoa_token_id")?,
        };
        let (h, heads) = (c.hidden_size, c.num_heads);
        let served = c.context_right == 0
            && (3..=17).contains(&c.context_left)
            && (1..=16).contains(&c.conv_kernel_size)
            && heads > 0
            && h.is_multiple_of(heads)
            && (h / heads).is_multiple_of(32)
            && h.is_multiple_of(32)
            && c.output_proj_dims.is_multiple_of(32)
            && c.conv_channels[0].is_multiple_of(32)
            && c.conv_channels[1].is_multiple_of(32)
            && c.conv_channels[0] / 4 == MELS / 4
            && c.gradient_clipping >= 1e10
            && c.logit_cap > 0.0;
        served.then_some(c)
    }
}

/// Soft tokens of `samples` 16 kHz samples (the processor's `_compute_audio_num_tokens`, the
/// recording cut at 30 s).
pub fn soft_tokens(samples: usize) -> usize {
    frames(samples).map_or(0, |f| f.div_ceil(2).div_ceil(2))
}

/// Mel frames of `samples` 16 kHz samples (cut at 30 s), `None` under 20 ms.
pub fn frames(samples: usize) -> Option<usize> {
    let padded = samples.min(MAX_SAMPLES) + FRAME / 2;
    (padded > FRAME).then(|| (padded - (FRAME + 1)) / HOP + 1)
}

fn bad(msg: impl Into<String>) -> HerbertError {
    HerbertError::Backend(msg.into())
}

/// Samples of a RIFF/WAVE file (PCM 8/16/24/32 or float 32), channels averaged, and its rate.
pub fn read_wav(path: &Path) -> Result<(Vec<f32>, usize)> {
    read_wav_bytes(&std::fs::read(path)?, &path.display().to_string())
}

/// [`read_wav`] of a file's bytes, `name` in the errors.
pub fn read_wav_bytes(b: &[u8], name: &str) -> Result<(Vec<f32>, usize)> {
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err(bad(format!(
            "{name}: not a RIFF/WAVE file (convert it: ffmpeg -i in -ar 16000 -ac 1 out.wav)"
        )));
    }
    let (mut at, mut fmt, mut data) = (12usize, None, None);
    while at + 8 <= b.len() {
        let id = &b[at..at + 4];
        let len = u32::from_le_bytes([b[at + 4], b[at + 5], b[at + 6], b[at + 7]]) as usize;
        let body = at + 8;
        let end = (body + len).min(b.len());
        if id == b"fmt " && end - body >= 16 {
            let u16at = |o: usize| u16::from_le_bytes([b[body + o], b[body + o + 1]]);
            let mut tag = u16at(0);
            if tag == 0xFFFE && end - body >= 26 {
                tag = u16at(24);
            }
            let rate =
                u32::from_le_bytes([b[body + 4], b[body + 5], b[body + 6], b[body + 7]]) as usize;
            fmt = Some((tag, u16at(2) as usize, rate, u16at(14) as usize));
        } else if id == b"data" {
            data = Some(&b[body..end]);
        }
        at = body + len + (len & 1);
    }
    let (tag, channels, rate, bits) = fmt.ok_or_else(|| bad(format!("{name}: no fmt chunk")))?;
    let data = data.ok_or_else(|| bad(format!("{name}: no data chunk")))?;
    if channels == 0 || rate == 0 {
        return Err(bad(format!("{name}: {channels} channels at {rate} Hz")));
    }
    // The format is checked before the samples are cut: a width of 0 bytes (1 to 7 bits) would
    // cut nothing.
    let one: fn(&[u8]) -> f32 = match (tag, bits) {
        (1, 8) => |s| (s[0] as f32 - 128.0) / 128.0,
        (1, 16) => |s| i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0,
        (1, 24) => |s| (i32::from_le_bytes([0, s[0], s[1], s[2]]) >> 8) as f32 / 8_388_608.0,
        (1, 32) => |s| i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f32 / 2_147_483_648.0,
        (3, 32) => |s| f32::from_le_bytes([s[0], s[1], s[2], s[3]]),
        _ => {
            return Err(bad(format!(
                "{name}: format {tag} at {bits} bits is not read (PCM 8/16/24/32 bits or float 32)"
            )))
        }
    };
    let width = bits / 8;
    let frame = width * channels;
    let out = data
        .chunks_exact(frame)
        .map(|f| {
            (0..channels)
                .map(|c| one(&f[c * width..(c + 1) * width]))
                .sum::<f32>()
                / channels as f32
        })
        .collect();
    Ok((out, rate))
}

/// Windowed-sinc resampling (Hann window, 16 zero crossings, cut-off at the lower Nyquist
/// frequency).
pub fn resample(x: &[f32], from: usize, to: usize) -> Vec<f32> {
    if from == to {
        return x.to_vec();
    }
    let cutoff = (to as f64 / from as f64).min(1.0);
    let half = (16.0 / cutoff).ceil() as isize;
    let n_out = (x.len() as u64 * to as u64 / from as u64) as usize;
    let mut out = Vec::with_capacity(n_out);
    for n in 0..n_out {
        let t = n as f64 * from as f64 / to as f64;
        let c = t.floor() as isize;
        let mut acc = 0.0f64;
        for i in (c - half + 1)..=(c + half) {
            if i < 0 || i as usize >= x.len() {
                continue;
            }
            let d = i as f64 - t;
            let w = 0.5 + 0.5 * (std::f64::consts::PI * d / half as f64).cos();
            let a = std::f64::consts::PI * d * cutoff;
            let s = if a.abs() < 1e-12 { 1.0 } else { a.sin() / a };
            acc += x[i as usize] as f64 * s * cutoff * w;
        }
        out.push(acc as f32);
    }
    out
}

pub use herbert_chat::media::resample_kaiser_best;

/// In place radix-2 FFT of `re + i·im`.
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (wr, wi) = ((ang * k as f64).cos(), (ang * k as f64).sin());
                let (a, b) = (start + k, start + k + len / 2);
                let (tr, ti) = (re[b] * wr - im[b] * wi, re[b] * wi + im[b] * wr);
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len <<= 1;
    }
}

/// `[bin][mel]` HTK triangular filters, unnormalized (transformers' `mel_filter_bank`).
fn mel_filters() -> Vec<f64> {
    let bins = FFT / 2 + 1;
    let hz_to_mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
    let mel_to_hz = |m: f64| 700.0 * (10f64.powf(m / 2595.0) - 1.0);
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(8000.0));
    let edges: Vec<f64> = (0..MELS + 2)
        .map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (MELS + 1) as f64))
        .collect();
    let mut fb = vec![0.0; bins * MELS];
    for b in 0..bins {
        let f = 8000.0 * b as f64 / (bins - 1) as f64;
        for m in 0..MELS {
            let down = (f - edges[m]) / (edges[m + 1] - edges[m]);
            let up = (edges[m + 2] - f) / (edges[m + 2] - edges[m + 1]);
            fb[b * MELS + m] = down.min(up).max(0.0);
        }
    }
    fb
}

/// The log-mel features of a 16 kHz mono waveform (cut at 30 s), `[frames][128]` bf16.
pub fn log_mel(wave: &[f32]) -> Result<Vec<u16>> {
    let wave = &wave[..wave.len().min(MAX_SAMPLES)];
    let frames =
        frames(wave.len()).ok_or_else(|| bad("the recording is too short: under 20 ms"))?;
    let pad = FRAME / 2;
    let sample = |i: usize| -> f64 {
        if i < pad || i - pad >= wave.len() {
            0.0
        } else {
            wave[i - pad] as f64
        }
    };
    let window: Vec<f64> = (0..FRAME)
        .map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / FRAME as f64).cos())
        .collect();
    let fb = mel_filters();
    let bins = FFT / 2 + 1;
    let mut features = vec![0u16; frames * MELS];
    // The frames over the host's threads, each its own FFT buffers.
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .clamp(1, 16);
    let per = frames.div_ceil(threads).max(1);
    std::thread::scope(|sc| {
        for (c, part) in features.chunks_mut(per * MELS).enumerate() {
            let (window, fb, sample) = (&window, &fb, &sample);
            sc.spawn(move || {
                let (mut re, mut im) = (vec![0.0; FFT], vec![0.0; FFT]);
                let mut mag = vec![0.0; bins];
                for (i, row) in part.chunks_mut(MELS).enumerate() {
                    let t = c * per + i;
                    re.iter_mut().for_each(|v| *v = 0.0);
                    im.iter_mut().for_each(|v| *v = 0.0);
                    for n in 0..FRAME {
                        re[n] = sample(t * HOP + n) * window[n];
                    }
                    fft(&mut re, &mut im);
                    for b in 0..bins {
                        mag[b] = (re[b] * re[b] + im[b] * im[b]).sqrt();
                    }
                    for (m, out) in row.iter_mut().enumerate() {
                        let mut s = 0.0;
                        for b in 0..bins {
                            s += mag[b] * fb[b * MELS + m];
                        }
                        *out = to_bf16((s + 0.001).ln() as f32);
                    }
                }
            });
        }
    });
    Ok(features)
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// A linear's packed weight and bounds, owned.
struct OwnedLinear {
    w: PackedBf16R16Weight,
    n: usize,
    k: usize,
    clip: Option<[f32; 4]>,
}

impl OwnedLinear {
    fn view(&self) -> TowerLinear<'_> {
        TowerLinear {
            weights: &self.w.data_r16,
            n: self.n,
            k: self.k,
            clip: self.clip,
        }
    }
}

struct OwnedLayer {
    /// `ff1.pre, ff1.post (× the half step when folded), ff2.pre, ff2.post, norm_pre_attn,
    /// norm_post_attn, norm_out, conv_pre, conv_norm, softplus, conv_dw (transposed)`.
    vectors: Vec<AVec<u16>>,
    /// `ff1.l1, ff1.l2, ff2.l1, ff2.l2, q, k, v, post, conv_start, conv_end`.
    linears: Vec<OwnedLinear>,
    rel_k: PackedBf16R16Weight,
}

/// The tower mounted: its weights, its buffers, its programs.
pub struct AudioTower {
    pub cfg: AudioConfig,
    shape: AudioShape,
    // The programs hold the addresses of everything below: declared first, dropped first.
    programs: AudioPrograms,
    buffers: Box<AudioBuffers>,
    #[allow(dead_code)]
    owned: (Vec<OwnedLayer>, Vec<PackedBf16R16Weight>, Vec<AVec<u16>>),
    /// Zero words the programs' frames point at.
    tokens: Vec<u32>,
    /// `HERBERT_AUDIO_PROFILE`: the layers' program traced.
    profile: bool,
}

impl AudioTower {
    /// Read, pack and mount the tower of `dir` for soft tokens of `out_hidden` values, `workers`
    /// participants, and make its relative keys; the tensors' names start with `prefix`
    /// (`model.` in Gemma 4, nothing in EmbeddingGemma 2).
    pub(crate) fn load(
        dir: &Path,
        prefix: &str,
        cfg: AudioConfig,
        out_hidden: usize,
        runner: &mut dyn Apart,
        workers: usize,
    ) -> Result<AudioTower> {
        let t0 = Instant::now();
        let mut t = crate::engine::vision::read_tensors_under(
            dir,
            prefix,
            &["audio_tower.", "embed_audio."],
        )?;
        let h = cfg.hidden_size;
        let [c0, c1] = cfg.conv_channels;
        let (od, taps, window) = (
            cfg.output_proj_dims,
            cfg.conv_kernel_size,
            cfg.context_left - 1,
        );
        let hd = h / cfg.num_heads;
        let mut take = |name: &str, len: usize| -> Result<Vec<u16>> {
            let v = t
                .remove(name)
                .ok_or_else(|| HerbertError::ModelLoad(format!("audio: {name} missing")))?;
            if v.len() != len {
                return Err(HerbertError::ModelLoad(format!(
                    "audio: {name} has {} values, {len} expected",
                    v.len()
                )));
            }
            Ok(v)
        };
        let p = "audio_tower.";
        // The first subsampling layer: `[c0][1][3][3]`, nine taps then zeros to `CONV0_K`.
        let w0 = take(
            &format!("{p}subsample_conv_projection.layer0.conv.weight"),
            c0 * 9,
        )?;
        let mut w0p = vec![0u16; c0 * CONV0_K];
        for o in 0..c0 {
            w0p[o * CONV0_K..o * CONV0_K + 9].copy_from_slice(&w0[o * 9..o * 9 + 9]);
        }
        // The second: `[c1][c0][3][3]` → `[c1][tap][c0]`.
        let w1 = take(
            &format!("{p}subsample_conv_projection.layer1.conv.weight"),
            c1 * c0 * 9,
        )?;
        let mut w1p = vec![0u16; c1 * 9 * c0];
        for o in 0..c1 {
            for i in 0..c0 {
                for tap in 0..9 {
                    w1p[(o * 9 + tap) * c0 + i] = w1[(o * c0 + i) * 9 + tap];
                }
            }
        }
        // A LayerNorm without a bias: `[g | 0]`.
        let ln = |g: Vec<u16>| -> AVec<u16> {
            let n = g.len();
            let mut v = g;
            v.resize(2 * n, 0);
            AVec::from_vec(v)
        };
        let norm0 = ln(take(
            &format!("{p}subsample_conv_projection.layer0.norm.weight"),
            c0,
        )?);
        let norm1 = ln(take(
            &format!("{p}subsample_conv_projection.layer1.norm.weight"),
            c1,
        )?);
        let proj_in = (MELS / 4) * c1;
        let packed = vec![
            PackedBf16R16Weight::pack_bf16(&w0p, c0, CONV0_K)?,
            PackedBf16R16Weight::pack_bf16(&w1p, c1, 9 * c0)?,
            PackedBf16R16Weight::pack_bf16(
                &take(
                    &format!("{p}subsample_conv_projection.input_proj_linear.weight"),
                    h * proj_in,
                )?,
                h,
                proj_in,
            )?,
            PackedBf16R16Weight::pack_bf16(
                &take(&format!("{p}output_proj.weight"), od * h)?,
                od,
                h,
            )?,
            PackedBf16R16Weight::pack_bf16(
                &take("embed_audio.embedding_projection.weight", out_hidden * od)?,
                out_hidden,
                od,
            )?,
        ];
        let out_bias = AVec::from_vec(take(&format!("{p}output_proj.bias"), od)?);
        // The sinusoids of the relative positions `window … 0`, `[sin… | cos…]`, in f32 then bf16.
        let half = h / 2;
        let incr = (10000f64.ln() / (half.max(2) - 1) as f64) as f32;
        let inv: Vec<f32> = (0..half).map(|i| (-(i as f32) * incr).exp()).collect();
        let mut pos = Vec::with_capacity((window + 1) * h);
        for p_id in (0..=window).rev() {
            let a: Vec<f32> = inv.iter().map(|f| p_id as f32 * f).collect();
            pos.extend(a.iter().map(|a| to_bf16(a.sin())));
            pos.extend(a.iter().map(|a| to_bf16(a.cos())));
        }
        let positions = AVec::from_vec(pos);
        // The half step folded into the post norms when it is a power of two (the same bits).
        let fold = cfg.residual_weight > 0.0 && cfg.residual_weight.log2().fract() == 0.0;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for l in 0..cfg.num_layers {
            let p = format!("{p}layers.{l}.");
            let mut vec = |name: &str, len: usize| -> Result<AVec<u16>> {
                Ok(AVec::from_vec(take(&format!("{p}{name}"), len)?))
            };
            let mut vectors = Vec::with_capacity(11);
            for f in ["feed_forward1", "feed_forward2"] {
                vectors.push(vec(&format!("{f}.pre_layer_norm.weight"), h)?);
                let mut post = vec(&format!("{f}.post_layer_norm.weight"), h)?;
                if fold {
                    post.iter_mut()
                        .for_each(|w| *w = to_bf16(bf16_to_f32(*w) * cfg.residual_weight));
                }
                vectors.push(post);
            }
            for name in [
                "norm_pre_attn",
                "norm_post_attn",
                "norm_out",
                "lconv1d.pre_layer_norm",
                "lconv1d.conv_norm",
            ] {
                vectors.push(vec(&format!("{name}.weight"), h)?);
            }
            let scale = take(&format!("{p}self_attn.per_dim_scale"), hd)?;
            vectors.push(AVec::from_vec(
                scale
                    .iter()
                    .map(|&b| {
                        let x = bf16_to_f32(b);
                        to_bf16(if x > 20.0 { x } else { x.exp().ln_1p() })
                    })
                    .collect(),
            ));
            let dw = take(&format!("{p}lconv1d.depthwise_conv1d.weight"), h * taps)?;
            let mut dwt = vec![0u16; taps * h];
            for c in 0..h {
                for r in 0..taps {
                    dwt[r * h + c] = dw[c * taps + r];
                }
            }
            vectors.push(AVec::from_vec(dwt));
            let mut linears = Vec::with_capacity(10);
            for (name, n, k) in [
                ("feed_forward1.ffw_layer_1", 4 * h, h),
                ("feed_forward1.ffw_layer_2", h, 4 * h),
                ("feed_forward2.ffw_layer_1", 4 * h, h),
                ("feed_forward2.ffw_layer_2", h, 4 * h),
                ("self_attn.q_proj", h, h),
                ("self_attn.k_proj", h, h),
                ("self_attn.v_proj", h, h),
                ("self_attn.post", h, h),
                ("lconv1d.linear_start", 2 * h, h),
                ("lconv1d.linear_end", h, h),
            ] {
                let w = PackedBf16R16Weight::pack_bf16(
                    &take(&format!("{p}{name}.linear.weight"), n * k)?,
                    n,
                    k,
                )?;
                let clip = if cfg.clipped_linears {
                    let mut four = [0f32; 4];
                    for (i, s) in ["input_min", "input_max", "output_min", "output_max"]
                        .iter()
                        .enumerate()
                    {
                        four[i] = bf16_to_f32(take(&format!("{p}{name}.{s}"), 1)?[0]);
                    }
                    Some(four)
                } else {
                    None
                };
                linears.push(OwnedLinear { w, n, k, clip });
            }
            let rel_k = PackedBf16R16Weight::pack_bf16(
                &take(&format!("{p}self_attn.relative_k_proj.weight"), h * h)?,
                h,
                h,
            )?;
            layers.push(OwnedLayer {
                vectors,
                linears,
                rel_k,
            });
        }
        let t_read = t0.elapsed().as_secs_f64();
        let shape = AudioShape {
            hidden: h,
            heads: cfg.num_heads,
            window,
            taps,
            c0,
            c1,
            mels: MELS,
            out_dims: od,
            out_hidden,
            eps: cfg.rms_norm_eps,
            cap: cfg.logit_cap,
            residual: cfg.residual_weight,
            max_frames: MAX_FRAMES,
        };
        let mut buffers = Box::new(AudioBuffers::new(&shape, cfg.num_layers));
        let weights = AudioWeights {
            conv0: &packed[0].data_r16,
            norm0: &norm0,
            conv1: &packed[1].data_r16,
            norm1: &norm1,
            input_proj: &packed[2].data_r16,
            out_proj: &packed[3].data_r16,
            out_bias: &out_bias,
            embed_proj: &packed[4].data_r16,
            positions: &positions,
            layers: layers
                .iter()
                .map(|l| {
                    let (v, lin) = (&l.vectors, &l.linears);
                    AudioLayerWeights {
                        ff: [
                            (&v[0][..], &v[1][..], lin[0].view(), lin[1].view()),
                            (&v[2][..], &v[3][..], lin[2].view(), lin[3].view()),
                        ],
                        norm_pre_attn: &v[4],
                        norm_post_attn: &v[5],
                        norm_out: &v[6],
                        attn: [lin[4].view(), lin[5].view(), lin[6].view(), lin[7].view()],
                        rel_k: &l.rel_k.data_r16,
                        softplus: &v[9],
                        conv_pre: &v[7],
                        conv_start: lin[8].view(),
                        conv_dw: &v[10],
                        conv_norm: &v[8],
                        conv_end: lin[9].view(),
                    }
                })
                .collect(),
        };
        // `HERBERT_AUDIO_PROFILE=1` (x86-64): the layers' program stamps its nodes, their times
        // summed by operation.
        let profile =
            cfg!(target_arch = "x86_64") && std::env::var_os("HERBERT_AUDIO_PROFILE").is_some();
        let mut programs = lower_audio(
            &shape,
            &weights,
            &mut buffers,
            workers,
            runner.apart_spin(),
            profile,
        )
        .map_err(|e| HerbertError::Backend(format!("audio: {e}")))?;
        drop(weights);
        let tokens = vec![0u32; shape.max_rows0()];
        runner.run_apart(&mut programs.relative, &tokens[..window + 1])?;
        eprintln!(
            "[audio] tower read and packed in {t_read:.2} s, mounted in {:.2} s ({:.0} MiB of buffers, {} bytes of code)",
            t0.elapsed().as_secs_f64() - t_read,
            buffers.bytes() as f64 / (1u64 << 20) as f64,
            programs.layers.program.code().len() + programs.conv0.program.code().len() + programs.conv1.program.code().len()
        );
        eprintln!("[audio] attention: {}", programs.attention);
        Ok(AudioTower {
            cfg,
            shape,
            programs,
            buffers,
            owned: (layers, packed, vec![norm0, norm1, out_bias, positions]),
            tokens,
            profile,
        })
    }

    /// The soft tokens, `[max_tokens][out_hidden]`: the table of the audio regime.
    pub fn table(&self) -> &[u16] {
        &self.buffers.out
    }

    /// Soft tokens of the longest recording.
    pub fn max_tokens(&self) -> usize {
        self.shape.max_tokens()
    }

    /// The 16 kHz mono `wave` (cut at 30 s) prepared and through the tower: its soft tokens in
    /// [`Self::table`]. `(frames, soft tokens, seconds of the preparation, seconds of the
    /// tower)`.
    pub(crate) fn encode(
        &mut self,
        runner: &mut dyn Apart,
        wave: &[f32],
    ) -> Result<(usize, usize, f64, f64)> {
        let t0 = Instant::now();
        let feats = log_mel(wave)?;
        let frames = feats.len() / MELS;
        let (half, quarter) = (MELS / 2, MELS / 4);
        let ta = frames.div_ceil(2);
        let t = ta.div_ceil(2);
        let b = &mut *self.buffers;
        // The 3 × 3 patches of the first layer, stride 2, one zero of padding.
        for tp in 0..ta {
            for fp in 0..half {
                let row = &mut b.patches0[(tp * half + fp) * CONV0_K..][..9];
                for kt in 0..3 {
                    for kf in 0..3 {
                        let (tm, fm) =
                            ((2 * tp + kt).wrapping_sub(1), (2 * fp + kf).wrapping_sub(1));
                        row[kt * 3 + kf] = if tm < frames && fm < MELS {
                            feats[tm * MELS + fm]
                        } else {
                            0
                        };
                    }
                }
            }
        }
        // The rows of the first layer's output the second one reads, tap by tap.
        let (rows0, rows1) = (self.shape.max_rows0(), self.shape.max_rows1());
        for kt in 0..3 {
            for kf in 0..3 {
                let index = &mut b.index1[(kt * 3 + kf) * rows1..][..t * quarter];
                for tp in 0..t {
                    for fp in 0..quarter {
                        let (tm, fm) =
                            ((2 * tp + kt).wrapping_sub(1), (2 * fp + kf).wrapping_sub(1));
                        index[tp * quarter + fp] = if tm < ta && fm < half {
                            (tm * half + fm) as u32
                        } else {
                            rows0 as u32
                        };
                    }
                }
            }
        }
        let t1 = Instant::now();
        let AudioTower {
            programs,
            tokens,
            profile,
            ..
        } = self;
        runner.run_apart(&mut programs.conv0, &tokens[..ta * half])?;
        runner.run_apart(&mut programs.conv1, &tokens[..t * quarter])?;
        if *profile {
            runner.reserve_apart_trace(&programs.layers)?;
        }
        let tl = Instant::now();
        runner.run_apart(&mut programs.layers, &tokens[..t])?;
        if *profile {
            crate::engine::vision::print_profile_as(
                "audio",
                runner,
                &programs.layers,
                tl.elapsed().as_secs_f64(),
            );
        }
        Ok((
            frames,
            t,
            (t1 - t0).as_secs_f64(),
            t1.elapsed().as_secs_f64(),
        ))
    }
}

/// 12B's encoder-free audio (`audio_config` `gemma4_unified_audio`): one soft token per chunk of
/// `samples` raw 16 kHz samples.
#[derive(Clone, Debug)]
pub struct ChunkConfig {
    pub samples: usize,
    pub rms_norm_eps: f32,
}

impl ChunkConfig {
    /// The embedder of the checkpoint in `dir`, when it has one this engine serves (chunks of a
    /// multiple of 32).
    pub fn of(dir: &Path) -> Option<ChunkConfig> {
        let root: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).ok()?).ok()?;
        let a = root.get("audio_config").filter(|a| {
            a.get("model_type").and_then(serde_json::Value::as_str) == Some("gemma4_unified_audio")
        })?;
        let samples = a
            .get("audio_embed_dim")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(640) as usize;
        let rms_norm_eps = a
            .get("rms_norm_eps")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1e-6) as f32;
        (samples > 0 && samples.is_multiple_of(32)).then_some(ChunkConfig {
            samples,
            rms_norm_eps,
        })
    }

    /// Soft tokens of `samples` samples (cut at 30 s): one per chunk, the last one partial.
    pub fn soft_tokens(&self, samples: usize) -> usize {
        samples.min(MAX_SAMPLES).div_ceil(self.samples)
    }
}

/// 12B's embedder mounted: `embed_audio`'s projection, its buffers, its program.
pub struct ChunkEmbedder {
    pub cfg: ChunkConfig,
    // The program holds the addresses of everything below: declared first, dropped first.
    program: crate::jit::execution::build::Mounted,
    buffers: Box<crate::jit::schedule::audio::ChunkBuffers>,
    #[allow(dead_code)]
    proj: PackedBf16R16Weight,
    tokens: Vec<u32>,
}

impl ChunkEmbedder {
    /// Read and pack `{prefix}embed_audio.embedding_projection` and mount the program for soft
    /// tokens of `out_hidden`.
    pub(crate) fn load(
        dir: &Path,
        prefix: &str,
        cfg: ChunkConfig,
        out_hidden: usize,
        runner: &dyn Apart,
        workers: usize,
    ) -> Result<ChunkEmbedder> {
        let t0 = Instant::now();
        let mut t = crate::engine::vision::read_tensors_under(dir, prefix, &["embed_audio."])?;
        let name = "embed_audio.embedding_projection.weight";
        let w = t
            .remove(name)
            .ok_or_else(|| HerbertError::ModelLoad(format!("audio: {name} missing")))?;
        if w.len() != out_hidden * cfg.samples {
            return Err(HerbertError::ModelLoad(format!(
                "audio: {name} has {} values, {} expected",
                w.len(),
                out_hidden * cfg.samples
            )));
        }
        let proj = PackedBf16R16Weight::pack_bf16(&w, out_hidden, cfg.samples)?;
        let rows = MAX_SAMPLES.div_ceil(cfg.samples);
        let mut buffers = Box::new(crate::jit::schedule::audio::ChunkBuffers::new(
            cfg.samples,
            out_hidden,
            rows,
        ));
        let program = crate::jit::schedule::audio::lower_chunks(
            cfg.samples,
            out_hidden,
            cfg.rms_norm_eps,
            &proj.data_r16,
            &mut buffers,
            workers,
            runner.apart_spin(),
        )
        .map_err(|e| HerbertError::Backend(format!("audio: {e}")))?;
        eprintln!(
            "[audio] encoder-free embedder: chunks of {} samples, mounted in {:.2} s",
            cfg.samples,
            t0.elapsed().as_secs_f64()
        );
        Ok(ChunkEmbedder {
            cfg,
            program,
            buffers,
            proj,
            tokens: vec![0u32; rows],
        })
    }

    /// The soft tokens: the table of the audio regime.
    pub fn table(&self) -> &[u16] {
        &self.buffers.out
    }

    /// Soft tokens of the longest recording.
    pub fn max_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// The chunks of `wave` (cut at 30 s, rounded to bf16 as transformers casts them, the last one
    /// zero-padded) through the norm and the projection: `(soft tokens, seconds)`.
    pub(crate) fn encode(&mut self, runner: &mut dyn Apart, wave: &[f32]) -> Result<(usize, f64)> {
        let t0 = Instant::now();
        let wave = &wave[..wave.len().min(MAX_SAMPLES)];
        let n = self.cfg.soft_tokens(wave.len());
        if n == 0 {
            return Err(bad("an empty recording"));
        }
        let d = self.cfg.samples;
        for (j, v) in self.buffers.chunks[..n * d].iter_mut().enumerate() {
            *v = to_bf16(wave.get(j).copied().unwrap_or(0.0));
        }
        runner.run_apart(&mut self.program, &self.tokens[..n])?;
        Ok((n, t0.elapsed().as_secs_f64()))
    }
}

/// What turns a recording into soft tokens: the conformer of E2B and E4B, or 12B's encoder-free
/// embedder. One per engine: the size difference between the variants does not matter.
#[allow(clippy::large_enum_variant)]
pub enum Hearing {
    Tower(AudioTower),
    Chunks(ChunkEmbedder),
}

impl Hearing {
    /// The soft tokens of the last recording, `[max_tokens][out_hidden]`.
    pub fn table(&self) -> &[u16] {
        match self {
            Hearing::Tower(t) => t.table(),
            Hearing::Chunks(c) => c.table(),
        }
    }
    /// Soft tokens of the longest recording.
    pub fn max_tokens(&self) -> usize {
        match self {
            Hearing::Tower(t) => t.max_tokens(),
            Hearing::Chunks(c) => c.max_tokens(),
        }
    }
    /// `wave` through it: `(mel frames or 0, soft tokens, seconds of the preparation, seconds of
    /// the tower)`.
    pub(crate) fn encode(
        &mut self,
        runner: &mut dyn Apart,
        wave: &[f32],
    ) -> Result<(usize, usize, f64, f64)> {
        match self {
            Hearing::Tower(t) => t.encode(runner, wave),
            Hearing::Chunks(c) => c.encode(runner, wave).map(|(n, s)| (0, n, 0.0, s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The processor's counts: 30 s make 2999 frames and 750 soft tokens; under 20 ms, none.
    #[test]
    fn the_counts_are_the_processors() {
        assert_eq!(MAX_FRAMES, 2999);
        assert_eq!(soft_tokens(MAX_SAMPLES), 750);
        assert_eq!(soft_tokens(MAX_SAMPLES * 2), 750);
        assert_eq!(frames(160), None);
        assert_eq!(soft_tokens(16_000), 25);
    }

    /// A WAV of 4 bits per sample is refused by its format (a width of 0 bytes would panic when
    /// cut), and a rate that does not reduce against 16 kHz by the resampler.
    #[test]
    fn malformed_recordings_are_errors() {
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF\0\0\0\0WAVEfmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        for v in [1u16, 1] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(&16_000u32.to_le_bytes());
        b.extend_from_slice(&8_000u32.to_le_bytes());
        for v in [1u16, 4] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(b"data");
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(&[0x12, 0x34, 0x56, 0x78]);
        let e = read_wav_bytes(&b, "t").unwrap_err().to_string();
        assert!(e.contains("4 bits"), "{e}");
        assert!(resample_kaiser_best(&[0.0; 64], 44_101, SAMPLE_RATE).is_err());
        assert_eq!(
            resample_kaiser_best(&[0.0; 48], 48_000, SAMPLE_RATE)
                .unwrap()
                .len(),
            16
        );
    }
}
