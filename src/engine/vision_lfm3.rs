//! The LFM3-VL vision tower (LFM3-VL-2B, `lfm3_vl_vision`) of the engine: its
//! configuration, its weights and its projector's read from the checkpoint,
//! the image as `Lfm3VlImageProcessor` prepares it, and the two generated
//! programs of `jit/schedule/vision_lfm3.rs`.
//!
//! The preparation is the processor's: `smart_resize` (both sides a multiple
//! of `patch · merge`, Python's `round`, the pixels between the processor's
//! `shortest_edge` and `longest_edge`), torchvision's antialiased bicubic on
//! the 8-bit image (`tvF.resize`, torch's byte path of
//! `_upsample_bicubic2d_aa`: i16 weights, the width first, the pass between
//! kept in bytes; equal to the bit), `(v − 127.5) / 127.5` in f32 then bf16,
//! each patch `(channel, frame, y, x)` with its frame twice
//! (`temporal_patch_size`), the patches in merge-block order. The RoPE angles
//! are `pos · θ^(−2i/32)` in f32, the row's for the first 16, the column's
//! for the next 16, their cosines and sines rounded to bf16 (transformers
//! keeps them in f32).
//!
//! The tower is lowered for the patches of the largest image seen so far
//! (rounded up to [`ROWS_STEP`]), again when a larger one comes; the soft
//! tokens land in a table that does not move ([`Lfm3Tower::table`], room for
//! the largest image the processor makes), which the text model's media
//! regime reads by blocks.

use crate::core::avec::AVec;
use crate::core::error::{HerbertError, Result};
use crate::engine::packs::to_bf16;
use crate::jit::model::apart::Apart;
use crate::jit::schedule::vision_lfm3::{
    lower_lfm3_tower, Lfm3LayerWeights, Lfm3Operator, Lfm3TowerBuffers, Lfm3TowerPrograms,
    Lfm3TowerShape, Lfm3TowerWeights,
};
use crate::model::PackedBf16R16Weight;
use std::path::Path;
use std::time::Instant;

/// The tower's rows grow by this many patches.
pub const ROWS_STEP: usize = 4096;

/// The tower and the processor of `config.json` and `processor_config.json`.
#[derive(Clone, Debug)]
pub struct Lfm3VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    pub temporal: usize,
    pub merge: usize,
    pub conv_kernel: usize,
    pub rope_theta: f64,
    pub norm_eps: f32,
    /// `"conv"` (false) or `"full_attention"` (true), per layer.
    pub attention: Vec<bool>,
    pub proj_hidden: usize,
    pub proj_eps: f32,
    /// The processor's `shortest_edge` and `longest_edge`: the pixels of a resized image, at least
    /// and at most.
    pub min_pixels: usize,
    pub max_pixels: usize,
}

impl Lfm3VisionConfig {
    /// The tower of `dir` when it is an LFM3-VL checkpoint this engine serves.
    pub fn of(dir: &Path) -> Option<Lfm3VisionConfig> {
        let root: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).ok()?).ok()?;
        if root.get("model_type").and_then(|t| t.as_str()) != Some("lfm3_vl") {
            return None;
        }
        let v = root.get("vision_config")?;
        if v.get("model_type").and_then(|t| t.as_str()) != Some("lfm3_vl_vision") {
            return None;
        }
        let u = |o: &serde_json::Value, key: &str| {
            o.get(key)
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
        };
        let hidden = u(v, "hidden_size")?;
        // Zero heads or a patch of zero pixels would divide by zero below: no tower.
        let heads = u(v, "num_attention_heads").filter(|&n| n > 0)?;
        let attention: Vec<bool> = v
            .get("layer_types")?
            .as_array()?
            .iter()
            .map(|t| t.as_str() == Some("full_attention"))
            .collect();
        let size = std::fs::read(dir.join("processor_config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .and_then(|p| {
                p.get("image_processor")
                    .and_then(|i| i.get("size"))
                    .cloned()
            });
        let edge =
            |key: &str, default: usize| size.as_ref().and_then(|s| u(s, key)).unwrap_or(default);
        let c = Lfm3VisionConfig {
            hidden_size: hidden,
            intermediate_size: u(v, "intermediate_size")?,
            num_heads: heads,
            num_kv_heads: u(v, "num_key_value_heads")?,
            head_dim: hidden / heads,
            patch_size: u(v, "patch_size").filter(|&n| n > 0)?,
            temporal: u(v, "temporal_patch_size")?,
            merge: u(v, "spatial_merge_size")?,
            conv_kernel: u(v, "conv_kernel_size")?,
            rope_theta: v
                .get("rope_parameters")
                .and_then(|r| r.get("rope_theta"))
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1e6),
            norm_eps: v
                .get("norm_eps")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1e-5) as f32,
            attention,
            proj_hidden: u(&root, "projector_hidden_size")?,
            proj_eps: root
                .get("projector_layer_norm_eps")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1e-6) as f32,
            min_pixels: edge("shortest_edge", 65_536),
            max_pixels: edge("longest_edge", 16_777_216),
        };
        let served = c.merge == 2
            && c.conv_kernel == 3
            && c.head_dim.is_multiple_of(32)
            && c.num_kv_heads > 0
            && c.num_heads.is_multiple_of(c.num_kv_heads)
            && c.hidden_size.is_multiple_of(32)
            && c.intermediate_size.is_multiple_of(32)
            && c.proj_hidden.is_multiple_of(32)
            && u(&root, "downsample_factor") == Some(c.merge)
            && root
                .get("projector_bias")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(true)
            && root.get("projector_hidden_act").and_then(|a| a.as_str()) == Some("gelu")
            && c.attention.len() == u(v, "num_hidden_layers")?;
        served.then_some(c)
    }

    fn factor(&self) -> usize {
        self.patch_size * self.merge
    }

    /// `smart_resize`: the size the processor resizes an `h × w` image to.
    pub fn target_size(&self, h: usize, w: usize) -> std::result::Result<(usize, usize), String> {
        smart_resize(h, w, self.factor(), self.min_pixels, self.max_pixels)
    }

    /// The language tokens of an `h × w` image.
    pub fn soft_tokens(&self, h: usize, w: usize) -> std::result::Result<usize, String> {
        let (th, tw) = self.target_size(h, w)?;
        Ok((th / self.factor()) * (tw / self.factor()))
    }

    /// Language tokens of one image at most.
    pub fn max_soft(&self) -> usize {
        self.max_pixels / (self.factor() * self.factor())
    }
}

/// `smart_resize` of `image_processing_lfm3_vl.py`, Python's `round` (half to even).
pub fn smart_resize(
    h: usize,
    w: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> std::result::Result<(usize, usize), String> {
    if h == 0 || w == 0 {
        return Err("an empty image".into());
    }
    if h.max(w) as f64 / h.min(w) as f64 > 200.0 {
        return Err(format!(
            "absolute aspect ratio must be smaller than 200, got {}",
            h.max(w) as f64 / h.min(w) as f64
        ));
    }
    let (hf, wf, f) = (h as f64, w as f64, factor as f64);
    let mut hb = (hf / f).round_ties_even() as usize * factor;
    let mut wb = (wf / f).round_ties_even() as usize * factor;
    if hb * wb > max_pixels {
        let beta = ((hf * wf) / max_pixels as f64).sqrt();
        hb = factor.max((hf / beta / f).floor() as usize * factor);
        wb = factor.max((wf / beta / f).floor() as usize * factor);
    } else if hb * wb < min_pixels {
        let beta = (min_pixels as f64 / (hf * wf)).sqrt();
        hb = (hf * beta / f).ceil() as usize * factor;
        wb = (wf * beta / f).ceil() as usize * factor;
    }
    Ok((hb, wb))
}

/// torch's `aa_filter` for the bicubic (`a = −0.5`, `cubic_convolution1` and `2`), in f64.
fn cubic(x: f64) -> f64 {
    let a = -0.5f64;
    let x = x.abs();
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
    } else {
        0.0
    }
}

/// The taps of one axis as torch's uint8 path computes them
/// (`_compute_index_ranges_int16_weights`): the weights of `_compute_indices_min_size_weights_aa`
/// in f64, then rescaled to i16 with one precision for the whole axis (the largest such that the
/// largest weight times `2^(p+1)` stays under `2^15`, at most 22), rounded half away from zero. For
/// each output: its first input and the weights; and the precision.
fn taps(input: usize, output: usize) -> (Vec<(usize, Vec<i32>)>, u32) {
    let scale = input as f64 / output as f64;
    let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
    let max_size = support.ceil() as i64 * 2 + 1;
    let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let mut wt_max = 0f64;
    let rows: Vec<(usize, Vec<f64>)> = (0..output)
        .map(|i| {
            let center = scale * (i as f64 + 0.5);
            let xmin = ((center - support + 0.5) as i64).max(0);
            let xsize =
                (((center + support + 0.5) as i64).min(input as i64) - xmin).clamp(0, max_size);
            let mut w: Vec<f64> = (0..xsize)
                .map(|j| cubic(((j + xmin) as f64 - center + 0.5) * invscale))
                .collect();
            let mut total = 0f64;
            for &x in &w {
                total += x;
            }
            if total != 0.0 {
                for x in &mut w {
                    *x /= total;
                    wt_max = wt_max.max(*x);
                }
            }
            (xmin as usize, w)
        })
        .collect();
    let mut p = 0u32;
    while p < 22 && ((0.5 + wt_max * (1i64 << (p + 1)) as f64) as i32) < (1 << 15) {
        p += 1;
    }
    let one = (1i64 << p) as f64;
    let rows = rows
        .into_iter()
        .map(|(x0, w)| {
            let q = w
                .iter()
                .map(|&x| {
                    let v = x * one;
                    (if v < 0.0 {
                        (-0.5 + v) as i32
                    } else {
                        (0.5 + v) as i32
                    }) as i16 as i32
                })
                .collect();
            (x0, q)
        })
        .collect();
    (rows, p)
}

/// One output byte: `2^(p−1) + Σ t·w` in i32, shifted right by `p`, clamped to a byte.
#[inline]
fn tap_sum(p: u32, ws: &[i32], at: impl Fn(usize) -> u8) -> u8 {
    let mut acc = 1i32 << (p - 1);
    for (j, &wt) in ws.iter().enumerate() {
        acc += at(j) as i32 * wt;
    }
    (acc >> p).clamp(0, 255) as u8
}

/// torchvision's antialiased bicubic of an 8-bit RGB image, as torch runs it on bytes
/// (`upsample_separable_Nd_kernel_impl`, and the same values from its AVX2 kernel): the width then
/// the height, each pass only if that side changes, in integers on i16 weights (`taps`), the pass
/// between them kept in bytes. Equal to the bit to torchvision 0.29 / torch 2.15 (AVX-512 and
/// AVX2). Each pass by bands of rows on `threads` threads.
pub fn resize_bicubic_aa(
    src: &[u8],
    h: usize,
    w: usize,
    th: usize,
    tw: usize,
    threads: usize,
) -> Vec<u8> {
    let threads = threads.max(1);
    let mid = if tw == w {
        src.to_vec()
    } else {
        let (hx, p) = taps(w, tw);
        let mut mid = vec![0u8; h * tw * 3];
        let band = h.div_ceil(threads).max(1);
        std::thread::scope(|scope| {
            for (b, rows) in mid.chunks_mut(band * tw * 3).enumerate() {
                let hx = &hx;
                scope.spawn(move || {
                    for (r, line) in rows.chunks_exact_mut(tw * 3).enumerate() {
                        let row = &src[(b * band + r) * w * 3..][..w * 3];
                        for (x, (x0, ws)) in hx.iter().enumerate() {
                            for c in 0..3 {
                                line[x * 3 + c] = tap_sum(p, ws, |j| row[(x0 + j) * 3 + c]);
                            }
                        }
                    }
                });
            }
        });
        mid
    };
    if th == h {
        return mid;
    }
    let (vy, p) = taps(h, th);
    let mut out = vec![0u8; th * tw * 3];
    let band = th.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (b, rows) in out.chunks_mut(band * tw * 3).enumerate() {
            let (vy, mid) = (&vy, &mid);
            scope.spawn(move || {
                for (r, line) in rows.chunks_exact_mut(tw * 3).enumerate() {
                    let (y0, ws) = &vy[b * band + r];
                    for (i, o) in line.iter_mut().enumerate() {
                        *o = tap_sum(p, ws, |j| mid[(y0 + j) * tw * 3 + i]);
                    }
                }
            });
        }
    });
    out
}

struct OwnedOperator {
    attention: bool,
    /// Attention: q, k, v, out; convolution: in_proj, out.
    linears: Vec<PackedBf16R16Weight>,
    /// Attention: q_norm, k_norm; convolution: the kernel transposed `[9][hidden]`.
    small: Vec<AVec<u16>>,
}

struct OwnedLayer {
    operator_norm: AVec<u16>,
    ffn_norm: AVec<u16>,
    op: OwnedOperator,
    mlp: [PackedBf16R16Weight; 3],
}

/// The tower mounted.
pub struct Lfm3Tower {
    pub cfg: Lfm3VisionConfig,
    shape: Lfm3TowerShape,
    // The programs hold the addresses of everything below: declared first, dropped first.
    programs: Option<Lfm3TowerPrograms>,
    buffers: Option<Box<Lfm3TowerBuffers>>,
    patchifier: PackedBf16R16Weight,
    embedding_norm: AVec<u16>,
    layers: Vec<OwnedLayer>,
    ln: AVec<u16>,
    linear_1: PackedBf16R16Weight,
    bias_1: AVec<u16>,
    linear_2: PackedBf16R16Weight,
    bias_2: AVec<u16>,
    /// `[max_soft][out_hidden]`: the soft tokens, the table the media regime reads.
    out: AVec<u16>,
    tokens: Vec<u32>,
    workers: usize,
    spin: Option<u32>,
    profile: bool,
}

impl Lfm3Tower {
    /// Read and pack the tower and the projector of `dir` (tensors under `prefix`, `model.`) for
    /// soft tokens of `out_hidden` values; lowered at the first image.
    pub(crate) fn load(
        dir: &Path,
        prefix: &str,
        cfg: Lfm3VisionConfig,
        out_hidden: usize,
        runner: &dyn Apart,
        workers: usize,
    ) -> Result<Lfm3Tower> {
        let t0 = Instant::now();
        let mut t = crate::engine::vision::read_tensors_under(
            dir,
            prefix,
            &["vision_tower.", "multi_modal_projector."],
        )?;
        let (h, hd, inter) = (cfg.hidden_size, cfg.head_dim, cfg.intermediate_size);
        let (qd, kvd) = (cfg.num_heads * hd, cfg.num_kv_heads * hd);
        let pix = 3 * cfg.temporal * cfg.patch_size * cfg.patch_size;
        let m2 = cfg.merge * cfg.merge;
        let take = |t: &mut std::collections::HashMap<String, Vec<u16>>,
                    name: &str,
                    len: usize|
         -> Result<Vec<u16>> {
            let v = t
                .remove(name)
                .ok_or_else(|| HerbertError::ModelLoad(format!("vision: {name} missing")))?;
            if v.len() != len {
                return Err(HerbertError::ModelLoad(format!(
                    "vision: {name} has {} values, {len} expected",
                    v.len()
                )));
            }
            Ok(v)
        };
        let pack = |t: &mut std::collections::HashMap<String, Vec<u16>>,
                    name: &str,
                    n: usize,
                    k: usize|
         -> Result<PackedBf16R16Weight> {
            PackedBf16R16Weight::pack_bf16(&take(t, name, n * k)?, n, k)
        };
        let t = &mut t;
        let patchifier = pack(t, "vision_tower.patchifier.weight", h, pix)?;
        let mut layers = Vec::with_capacity(cfg.attention.len());
        for (l, &attention) in cfg.attention.iter().enumerate() {
            let p = format!("vision_tower.layers.{l}.");
            let op = if attention {
                let linears = vec![
                    pack(t, &format!("{p}self_attn.q_proj.weight"), qd, h)?,
                    pack(t, &format!("{p}self_attn.k_proj.weight"), kvd, h)?,
                    pack(t, &format!("{p}self_attn.v_proj.weight"), kvd, h)?,
                    pack(t, &format!("{p}self_attn.out_proj.weight"), h, qd)?,
                ];
                let small = vec![
                    AVec::from_vec(take(t, &format!("{p}self_attn.q_layernorm.weight"), hd)?),
                    AVec::from_vec(take(t, &format!("{p}self_attn.k_layernorm.weight"), hd)?),
                ];
                OwnedOperator {
                    attention,
                    linears,
                    small,
                }
            } else {
                let linears = vec![
                    pack(t, &format!("{p}conv.in_proj.weight"), 3 * h, h)?,
                    pack(t, &format!("{p}conv.out_proj.weight"), h, h)?,
                ];
                // `[hidden][1][3][3]` → `[9][hidden]`.
                let k = take(t, &format!("{p}conv.spatial_conv.weight"), h * 9)?;
                let mut kt = vec![0u16; 9 * h];
                for c in 0..h {
                    for tap in 0..9 {
                        kt[tap * h + c] = k[c * 9 + tap];
                    }
                }
                OwnedOperator {
                    attention,
                    linears,
                    small: vec![AVec::from_vec(kt)],
                }
            };
            layers.push(OwnedLayer {
                operator_norm: AVec::from_vec(take(t, &format!("{p}operator_norm.weight"), h)?),
                ffn_norm: AVec::from_vec(take(t, &format!("{p}ffn_norm.weight"), h)?),
                op,
                mlp: [
                    pack(t, &format!("{p}feed_forward.w1.weight"), inter, h)?,
                    pack(t, &format!("{p}feed_forward.w3.weight"), inter, h)?,
                    pack(t, &format!("{p}feed_forward.w2.weight"), h, inter)?,
                ],
            });
        }
        let embedding_norm = AVec::from_vec(take(t, "vision_tower.embedding_norm.weight", h)?);
        let mut ln = take(t, "multi_modal_projector.layer_norm.weight", h)?;
        ln.extend(take(t, "multi_modal_projector.layer_norm.bias", h)?);
        let ph = cfg.proj_hidden;
        let linear_1 = PackedBf16R16Weight::pack_bf16(
            &take(t, "multi_modal_projector.linear_1.weight", ph * m2 * h)?,
            ph,
            m2 * h,
        )?;
        let bias_1 = AVec::from_vec(take(t, "multi_modal_projector.linear_1.bias", ph)?);
        let linear_2 = PackedBf16R16Weight::pack_bf16(
            &take(t, "multi_modal_projector.linear_2.weight", out_hidden * ph)?,
            out_hidden,
            ph,
        )?;
        let bias_2 = AVec::from_vec(take(t, "multi_modal_projector.linear_2.bias", out_hidden)?);
        let shape = Lfm3TowerShape {
            hidden: h,
            heads: cfg.num_heads,
            kv_heads: cfg.num_kv_heads,
            head_dim: hd,
            intermediate: inter,
            pixels: pix,
            eps: cfg.norm_eps,
            eps_ln: cfg.proj_eps,
            merge2: m2,
            proj_hidden: ph,
            out_hidden,
            max_rows: 0,
        };
        let out = AVec::zeroed(cfg.max_soft() * out_hidden);
        let profile =
            cfg!(target_arch = "x86_64") && std::env::var_os("HERBERT_VISION_PROFILE").is_some();
        eprintln!(
            "[vision] LFM3 tower and projector read and packed in {:.2} s",
            t0.elapsed().as_secs_f64()
        );
        Ok(Lfm3Tower {
            cfg,
            shape,
            programs: None,
            buffers: None,
            patchifier,
            embedding_norm,
            layers,
            ln: AVec::from_vec(ln),
            linear_1,
            bias_1,
            linear_2,
            bias_2,
            out,
            tokens: Vec::new(),
            workers,
            spin: runner.apart_spin(),
            profile,
        })
    }

    /// The soft tokens, `[max_soft][out_hidden]`.
    pub fn table(&self) -> &[u16] {
        &self.out
    }

    /// Rows of the table.
    pub fn max_tokens(&self) -> usize {
        self.cfg.max_soft()
    }

    /// The programs lowered for `rows` patches at least.
    fn ensure(&mut self, rows: usize) -> Result<()> {
        if self.programs.is_some() && self.shape.max_rows >= rows {
            return Ok(());
        }
        let t0 = Instant::now();
        self.programs = None;
        self.buffers = None;
        let max_rows = rows
            .next_multiple_of(ROWS_STEP)
            .min((self.cfg.max_soft() * self.shape.merge2).next_multiple_of(self.shape.merge2))
            .max(rows);
        let shape = Lfm3TowerShape {
            max_rows,
            ..self.shape
        };
        let mut buffers = Box::new(Lfm3TowerBuffers::new(&shape, self.workers));
        let weights = Lfm3TowerWeights {
            patchifier: &self.patchifier.data_r16,
            embedding_norm: &self.embedding_norm,
            layers: self
                .layers
                .iter()
                .map(|l| Lfm3LayerWeights {
                    operator_norm: &l.operator_norm,
                    ffn_norm: &l.ffn_norm,
                    op: if l.op.attention {
                        Lfm3Operator::Attention {
                            q: &l.op.linears[0].data_r16,
                            k: &l.op.linears[1].data_r16,
                            v: &l.op.linears[2].data_r16,
                            out: &l.op.linears[3].data_r16,
                            q_norm: &l.op.small[0],
                            k_norm: &l.op.small[1],
                        }
                    } else {
                        Lfm3Operator::Conv {
                            in_proj: &l.op.linears[0].data_r16,
                            kernel: &l.op.small[0],
                            out: &l.op.linears[1].data_r16,
                        }
                    },
                    w1: &l.mlp[0].data_r16,
                    w3: &l.mlp[1].data_r16,
                    w2: &l.mlp[2].data_r16,
                })
                .collect(),
            ln: &self.ln,
            linear_1: &self.linear_1.data_r16,
            bias_1: &self.bias_1,
            linear_2: &self.linear_2.data_r16,
            bias_2: &self.bias_2,
        };
        let programs = lower_lfm3_tower(
            &shape,
            &weights,
            &mut buffers,
            &mut self.out,
            self.workers,
            self.spin,
            self.profile,
        )
        .map_err(|e| HerbertError::Backend(format!("vision: {e}")))?;
        drop(weights);
        eprintln!(
            "[vision] LFM3 tower lowered for {max_rows} patches in {:.2} s ({:.0} MiB of buffers, {} + {} bytes of code)",
            t0.elapsed().as_secs_f64(),
            buffers.bytes() as f64 / (1u64 << 20) as f64,
            programs.layers.program.code().len(),
            programs.project.program.code().len()
        );
        self.shape = shape;
        self.tokens = vec![0u32; max_rows];
        self.buffers = Some(buffers);
        self.programs = Some(programs);
        Ok(())
    }

    /// The image (`rgb`: `h` rows of `w` pixels) prepared and through the tower and the projector:
    /// its soft tokens in [`Self::table`]. `(grid_w, grid_h, soft tokens, seconds of the
    /// preparation, seconds of the tower)`.
    pub(crate) fn encode(
        &mut self,
        runner: &mut dyn Apart,
        rgb: &[u8],
        h: usize,
        w: usize,
    ) -> Result<(usize, usize, usize, f64, f64)> {
        let t0 = Instant::now();
        if rgb.len() != h * w * 3 {
            return Err(HerbertError::Backend(format!(
                "an image of {w} × {h} pixels in {} bytes",
                rgb.len()
            )));
        }
        let (th, tw) = self.cfg.target_size(h, w).map_err(HerbertError::Backend)?;
        let (patch, merge) = (self.cfg.patch_size, self.cfg.merge);
        let (gw, gh) = (tw / patch, th / patch);
        let rows = gw * gh;
        let soft = rows / (merge * merge);
        if soft > self.cfg.max_soft() {
            return Err(HerbertError::Backend(format!(
                "an image of {gw} × {gh} patches does not fit the tower"
            )));
        }
        self.ensure(rows)?;
        let resized = if th == h && tw == w {
            rgb.to_vec()
        } else {
            resize_bicubic_aa(rgb, h, w, th, tw, self.workers)
        };
        self.prepare(&resized, gw, gh);
        // `HERBERT_VISION_PATCHES=FILE`: the patches as the tower reads them, in f32 (the
        // processor's `pixel_values`).
        if let Some(path) = std::env::var_os("HERBERT_VISION_PATCHES") {
            let b = self.buffers.as_ref().expect("lowered");
            let f: Vec<u8> = b.patches[..rows * self.shape.pixels]
                .iter()
                .flat_map(|&v| f32::from_bits(u32::from(v) << 16).to_le_bytes())
                .collect();
            std::fs::write(path, f)?;
        }
        let t1 = Instant::now();
        let Lfm3Tower {
            programs,
            tokens,
            profile,
            ..
        } = self;
        let programs = programs.as_mut().expect("lowered");
        if *profile {
            runner.reserve_apart_trace(&programs.layers)?;
        }
        let t = Instant::now();
        runner.run_apart(&mut programs.layers, &tokens[..rows])?;
        if *profile {
            crate::engine::vision::print_profile_as(
                "vision",
                runner,
                &programs.layers,
                t.elapsed().as_secs_f64(),
            );
        }
        runner.run_apart(&mut programs.project, &tokens[..soft])?;
        Ok((
            gw,
            gh,
            soft,
            (t1 - t0).as_secs_f64(),
            t1.elapsed().as_secs_f64(),
        ))
    }

    /// The patches, their RoPE tables and their neighbours into the buffers.
    fn prepare(&mut self, img: &[u8], gw: usize, gh: usize) {
        let (patch, merge, temporal) = (self.cfg.patch_size, self.cfg.merge, self.cfg.temporal);
        let tw = gw * patch;
        let lut: Vec<u16> = (0..256u32)
            .map(|v| to_bf16((v as f32 - 127.5) / 127.5))
            .collect();
        let hd = self.cfg.head_dim;
        let (half, quarter) = (hd / 2, hd / 4);
        // `1 / θ^(arange(0, hd/2, 2) / (hd/2))`, in f32 as torch.
        let inv: Vec<f32> = (0..quarter)
            .map(|i| 1.0 / (self.cfg.rope_theta as f32).powf((2 * i) as f32 / half as f32))
            .collect();
        let max_rows = self.shape.max_rows;
        let pix = self.shape.pixels;
        let b = self.buffers.as_mut().expect("lowered");
        let (bw, m2) = (gw / merge, merge * merge);
        let row_of = |py: usize, px: usize| {
            ((py / merge) * bw + px / merge) * m2 + (py % merge) * merge + px % merge
        };
        let plane = patch * patch;
        let mut row = 0;
        for by in 0..gh / merge {
            for bx in 0..bw {
                for dy in 0..merge {
                    for dx in 0..merge {
                        let (py, px) = (by * merge + dy, bx * merge + dx);
                        let dst = &mut b.patches[row * pix..(row + 1) * pix];
                        for y in 0..patch {
                            let at = ((py * patch + y) * tw + px * patch) * 3;
                            for x in 0..patch {
                                for c in 0..3 {
                                    let v = lut[img[at + x * 3 + c] as usize];
                                    for f in 0..temporal {
                                        dst[(c * temporal + f) * plane + y * patch + x] = v;
                                    }
                                }
                            }
                        }
                        for (axis, pos) in [py, px].into_iter().enumerate() {
                            for (i, &f) in inv.iter().enumerate() {
                                let a = pos as f32 * f;
                                let at = row * half + axis * quarter + i;
                                b.cos[at] = to_bf16(a.cos());
                                b.sin[at] = to_bf16(a.sin());
                            }
                        }
                        for tap in 0..9 {
                            let (ny, nx) =
                                ((py + tap / 3) as isize - 1, (px + tap % 3) as isize - 1);
                            let inside =
                                ny >= 0 && nx >= 0 && (ny as usize) < gh && (nx as usize) < gw;
                            b.neighbours[tap * max_rows + row] = if inside {
                                row_of(ny as usize, nx as usize) as u32
                            } else {
                                max_rows as u32
                            };
                        }
                        row += 1;
                    }
                }
            }
        }
    }
}
