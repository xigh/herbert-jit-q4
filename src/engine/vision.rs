//! The Gemma 4 vision tower (E2B, E4B, 26B-A4B) of the engine: its configuration, its
//! weights read from the checkpoint and packed at the first image, the image
//! as the processor prepares it, and the two generated programs of
//! `jit/schedule/vision.rs` that turn it into soft tokens.
//!
//! The preparation, on the host: Pillow's
//! bicubic resize in fixed point, both sides a multiple of `patch · pool`, at
//! most `max_soft · pool²` patches, the values `2·(p − ½)` in bf16. Here the
//! patches are laid out cell by cell (the `pool × pool` patches of one soft
//! token together), with their grid positions and the cosines and sines of
//! the 2-D RoPE (`[x freqs | y freqs]` per row, computed in f32 then rounded).

use crate::core::avec::AVec;
use crate::core::error::{HerbertError, Result};
use crate::engine::packs::to_bf16;
use crate::jit::execution::build::Mounted;
use crate::jit::model::apart::Apart;
use crate::jit::schedule::vision::{
    lower_tower, lower_unified, TowerBuffers, TowerLayerWeights, TowerLinear, TowerPrograms,
    TowerShape, TowerWeights, UnifiedBuffers, UnifiedShape, UnifiedWeights,
};
use crate::model::PackedBf16R16Weight;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// Soft tokens of one image at most (the processor's `default_output_length`).
pub const IMAGE_ROWS: usize = 280;

/// The tower of `config.json` (`vision_config`).
#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub position_embedding_size: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f32,
    pub clipped_linears: bool,
    pub standardize: bool,
    /// 12B's encoder-free embedder (`gemma4_unified_vision`): no tower, every `patch · pool`-wide
    /// patch through LayerNorms, a dense with bias and the factorized positions
    /// (`mm_posemb_size` per axis), then the embedder; `hidden_size` is `mm_embed_dim`, the
    /// tower's geometry (layers, heads, MLP) 0.
    pub unified: bool,
    pub mm_posemb_size: usize,
}

impl VisionConfig {
    /// The vision of the checkpoint in `dir`, when it has one this engine serves: the separate
    /// tower of E2B, E4B and 26B-A4B (as many K/V heads as heads; 26B-A4B's `standardize`d) or
    /// the encoder-free embedder of the 12B.
    pub fn of(dir: &Path) -> Option<VisionConfig> {
        let root: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).ok()?).ok()?;
        let v = root.get("vision_config").filter(|v| v.is_object())?;
        let u = |key: &str| {
            v.get(key)
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
        };
        let model_type = v.get("model_type").and_then(|t| t.as_str()).unwrap_or("");
        if model_type.starts_with("gemma4_unified") {
            if model_type != "gemma4_unified_vision" {
                return None;
            }
            let dim = u("mm_embed_dim")?;
            let c = VisionConfig {
                hidden_size: dim,
                num_layers: 0,
                num_heads: 0,
                num_kv_heads: 0,
                head_dim: 0,
                intermediate_size: 0,
                patch_size: u("patch_size")?,
                pooling_kernel_size: u("pooling_kernel_size")?,
                position_embedding_size: 0,
                rope_theta: 0.0,
                rms_norm_eps: v
                    .get("rms_norm_eps")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(1e-6) as f32,
                clipped_linears: false,
                standardize: false,
                unified: true,
                mm_posemb_size: u("mm_posemb_size")?,
            };
            let side = c.patch_size * c.pooling_kernel_size;
            let served = u("output_proj_dims") == Some(dim)
                && dim.is_multiple_of(32)
                && (3 * side * side).is_multiple_of(32)
                && c.pooling_kernel_size > 0
                && c.mm_posemb_size > 0;
            return served.then_some(c);
        }
        let c = VisionConfig {
            hidden_size: u("hidden_size")?,
            num_layers: u("num_hidden_layers")?,
            num_heads: u("num_attention_heads")?,
            num_kv_heads: u("num_key_value_heads")?,
            head_dim: u("head_dim")?,
            intermediate_size: u("intermediate_size")?,
            patch_size: u("patch_size")?,
            pooling_kernel_size: u("pooling_kernel_size")?,
            position_embedding_size: u("position_embedding_size")?,
            rope_theta: v
                .get("rope_parameters")
                .and_then(|r| r.get("rope_theta"))
                .and_then(serde_json::Value::as_f64)?,
            rms_norm_eps: v
                .get("rms_norm_eps")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1e-6) as f32,
            clipped_linears: v
                .get("use_clipped_linears")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            standardize: v
                .get("standardize")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            unified: false,
            mm_posemb_size: 0,
        };
        // Heads of any multiple of 8 (the attention's and the per-head norms' last partial block;
        // 72 in 26B-A4B), the RoPE's quarter whole; an MLP of any multiple of 16 (the
        // GEMM's output blocks; 4 304 in 26B-A4B). Whether this machine's kernels serve
        // them is the mounting's to say (AVX-512 for a partial block).
        let served = c.num_kv_heads == c.num_heads
            && c.head_dim.is_multiple_of(8)
            && c.hidden_size.is_multiple_of(32)
            && c.intermediate_size.is_multiple_of(16)
            && (3 * c.patch_size * c.patch_size).is_multiple_of(32)
            && c.pooling_kernel_size > 0
            && c.pooling_kernel_size * c.pooling_kernel_size <= u8::MAX as usize;
        served.then_some(c)
    }
}

/// The vision tensors of the checkpoint (`{prefix}vision_tower.`, `{prefix}embed_vision.`, 12B's
/// `{prefix}vision_embedder.`), bf16, under their names without the prefix.
fn read_tensors(dir: &Path, prefix: &str) -> Result<HashMap<String, Vec<u16>>> {
    read_tensors_under(
        dir,
        prefix,
        &["vision_tower.", "embed_vision.", "vision_embedder."],
    )
}

/// The tensors of the checkpoint whose names start with `prefix` then one of `roots`, bf16, under
/// their names without the prefix (a tower's: the vision tower's, the audio tower's).
pub(crate) fn read_tensors_under(
    dir: &Path,
    prefix: &str,
    roots: &[&str],
) -> Result<HashMap<String, Vec<u16>>> {
    let roots: Vec<String> = roots.iter().map(|r| format!("{prefix}{r}")).collect();
    let keep = |name: &str| roots.iter().any(|r| name.starts_with(r.as_str()));
    let index = dir.join("model.safetensors.index.json");
    let shards: Vec<String> = if index.exists() {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&index)?)?;
        let map = v
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| HerbertError::ModelLoad("index without weight_map".into()))?;
        let mut s: Vec<String> = map
            .iter()
            .filter(|(t, _)| keep(t))
            .filter_map(|(_, f)| f.as_str().map(str::to_string))
            .collect();
        s.sort();
        s.dedup();
        s
    } else {
        vec!["model.safetensors".to_string()]
    };
    let mut out = HashMap::new();
    for shard in shards {
        let file = std::fs::File::open(dir.join(&shard))
            .map_err(|e| HerbertError::ModelLoad(format!("cannot open shard {shard}: {e}")))?;
        // SAFETY: the file is read-only and not modified while mapped.
        let mapped = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| HerbertError::ModelLoad(format!("cannot map shard {shard}: {e}")))?;
        let tensors = safetensors::SafeTensors::deserialize(&mapped[..])
            .map_err(|e| HerbertError::ModelLoad(format!("cannot parse shard {shard}: {e}")))?;
        for name in tensors.names() {
            if keep(name) {
                let view = tensors
                    .tensor(name)
                    .map_err(|e| HerbertError::ModelLoad(format!("{name}: {e}")))?;
                out.insert(
                    name[prefix.len()..].to_string(),
                    crate::model::loader::load_bf16_from_view(&view, name)?,
                );
            }
        }
    }
    Ok(out)
}

/// A linear's weight and bounds, owned.
struct OwnedLinear {
    w: PackedBf16R16Weight,
    clip: Option<[f32; 4]>,
}

struct OwnedLayer {
    norms: [AVec<u16>; 6],
    linears: Vec<OwnedLinear>,
}

/// The tower mounted: its weights, its buffers, its two programs.
pub struct Tower {
    pub cfg: VisionConfig,
    shape: TowerShape,
    // The programs hold the addresses of everything below: declared first, dropped first.
    programs: TowerPrograms,
    buffers: Box<TowerBuffers>,
    #[allow(dead_code)]
    input_proj: PackedBf16R16Weight,
    #[allow(dead_code)]
    position_table: AVec<u16>,
    #[allow(dead_code)]
    embed_proj: PackedBf16R16Weight,
    /// `std_bias` then `std_scale` (`standardize`).
    #[allow(dead_code)]
    standardize: Option<AVec<u16>>,
    #[allow(dead_code)]
    layers: Vec<OwnedLayer>,
    /// `rows` zero words the programs' frames point at.
    tokens: Vec<u32>,
    /// `HERBERT_VISION_PROFILE`: the layers' program traced.
    profile: bool,
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

impl Tower {
    /// Read, pack and mount the tower of `dir` for soft tokens of `out_hidden` values, `workers`
    /// participants; the tensors' names start with `prefix` (`model.` in Gemma 4, nothing in
    /// EmbeddingGemma 2).
    pub(crate) fn load(
        dir: &Path,
        prefix: &str,
        cfg: VisionConfig,
        out_hidden: usize,
        runner: &dyn Apart,
        workers: usize,
    ) -> Result<Tower> {
        if cfg.unified {
            return Err(HerbertError::Backend(
                "an encoder-free vision embedder has no tower".into(),
            ));
        }
        let t0 = Instant::now();
        let mut t = read_tensors(dir, prefix)?;
        let (h, hd, inter) = (cfg.hidden_size, cfg.head_dim, cfg.intermediate_size);
        let qd = cfg.num_heads * hd;
        let pix = 3 * cfg.patch_size * cfg.patch_size;
        let mut take = |name: &str, len: usize| -> Result<Vec<u16>> {
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
        let input_proj = PackedBf16R16Weight::pack_bf16(
            &take("vision_tower.patch_embedder.input_proj.weight", h * pix)?,
            h,
            pix,
        )?;
        let position_table = AVec::from_vec(take(
            "vision_tower.patch_embedder.position_embedding_table",
            2 * cfg.position_embedding_size * h,
        )?);
        let embed_proj = PackedBf16R16Weight::pack_bf16(
            &take("embed_vision.embedding_projection.weight", out_hidden * h)?,
            out_hidden,
            h,
        )?;
        let standardize = if cfg.standardize {
            let mut v = take("vision_tower.std_bias", h)?;
            v.extend(take("vision_tower.std_scale", h)?);
            Some(AVec::from_vec(v))
        } else {
            None
        };
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for l in 0..cfg.num_layers {
            let p = format!("vision_tower.encoder.layers.{l}.");
            let mut norm = |name: &str, len: usize| -> Result<AVec<u16>> {
                Ok(AVec::from_vec(take(&format!("{p}{name}.weight"), len)?))
            };
            let norms = [
                norm("input_layernorm", h)?,
                norm("post_attention_layernorm", h)?,
                norm("pre_feedforward_layernorm", h)?,
                norm("post_feedforward_layernorm", h)?,
                norm("self_attn.q_norm", hd)?,
                norm("self_attn.k_norm", hd)?,
            ];
            let mut linears = Vec::with_capacity(7);
            for (name, n, k) in [
                ("self_attn.q_proj", qd, h),
                ("self_attn.k_proj", qd, h),
                ("self_attn.v_proj", qd, h),
                ("self_attn.o_proj", h, qd),
                ("mlp.gate_proj", inter, h),
                ("mlp.up_proj", inter, h),
                ("mlp.down_proj", h, inter),
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
                linears.push(OwnedLinear { w, clip });
            }
            layers.push(OwnedLayer { norms, linears });
        }
        let t_read = t0.elapsed().as_secs_f64();
        let shape = TowerShape {
            hidden: h,
            heads: cfg.num_heads,
            head_dim: hd,
            intermediate: inter,
            pixels: pix,
            positions: cfg.position_embedding_size,
            eps: cfg.rms_norm_eps,
            pool: cfg.pooling_kernel_size,
            max_soft: IMAGE_ROWS,
            out_hidden,
            standardize: cfg.standardize,
        };
        let mut buffers = Box::new(TowerBuffers::new(&shape, workers));
        let weights = TowerWeights {
            input_proj: &input_proj.data_r16,
            position_table: &position_table,
            embed_proj: &embed_proj.data_r16,
            standardize: standardize.as_deref(),
            layers: layers
                .iter()
                .map(|l| {
                    let lin = |i: usize, n: usize, k: usize| TowerLinear {
                        weights: &l.linears[i].w.data_r16,
                        n,
                        k,
                        clip: l.linears[i].clip,
                    };
                    TowerLayerWeights {
                        input_norm: &l.norms[0],
                        post_attn_norm: &l.norms[1],
                        pre_ff_norm: &l.norms[2],
                        post_ff_norm: &l.norms[3],
                        q_norm: &l.norms[4],
                        k_norm: &l.norms[5],
                        linears: [
                            lin(0, qd, h),
                            lin(1, qd, h),
                            lin(2, qd, h),
                            lin(3, h, qd),
                            lin(4, inter, h),
                            lin(5, inter, h),
                            lin(6, h, inter),
                        ],
                    }
                })
                .collect(),
        };
        // `HERBERT_VISION_PROFILE=1` (x86-64): the layers' program stamps its nodes, their times
        // summed by operation.
        let profile =
            cfg!(target_arch = "x86_64") && std::env::var_os("HERBERT_VISION_PROFILE").is_some();
        let programs = lower_tower(
            &shape,
            &weights,
            &mut buffers,
            workers,
            runner.apart_spin(),
            profile,
        )
        .map_err(|e| HerbertError::Backend(format!("vision: {e}")))?;
        drop(weights);
        let tokens = vec![0u32; shape.max_rows()];
        eprintln!(
            "[vision] tower read and packed in {t_read:.2} s, mounted in {:.2} s ({:.0} MiB of buffers, {} + {} bytes of code)",
            t0.elapsed().as_secs_f64() - t_read,
            buffers.bytes() as f64 / (1u64 << 20) as f64,
            programs.layers.program.code().len(),
            programs.embed.program.code().len()
        );
        Ok(Tower {
            cfg,
            shape,
            programs,
            buffers,
            input_proj,
            position_table,
            embed_proj,
            standardize,
            layers,
            tokens,
            profile,
        })
    }

    /// The soft tokens, `[max_soft][out_hidden]`: the table of the image regime.
    pub fn table(&self) -> &[u16] {
        &self.buffers.out
    }

    /// The image (`rgb`: `h` rows of `w` pixels) prepared and through the tower: its soft tokens in
    /// [`Self::table`]. `(grid_w, grid_h, soft tokens, seconds of the preparation, seconds of
    /// the tower)`.
    pub(crate) fn encode(
        &mut self,
        runner: &mut dyn Apart,
        rgb: &[u8],
        h: usize,
        w: usize,
    ) -> Result<(usize, usize, usize, f64, f64)> {
        let t0 = Instant::now();
        let (gw, gh) = self.prepare(rgb, h, w)?;
        let t1 = Instant::now();
        let k = self.shape.pool;
        let (rows, soft) = (gw * gh, gw * gh / (k * k));
        let Tower {
            programs,
            tokens,
            profile,
            ..
        } = self;
        if *profile {
            runner.reserve_apart_trace(&programs.layers)?;
        }
        let t = Instant::now();
        run(runner, &mut programs.layers, &tokens[..rows])?;
        if *profile {
            print_profile_as(
                "vision",
                runner,
                &programs.layers,
                t.elapsed().as_secs_f64(),
            );
        }
        run(runner, &mut programs.embed, &tokens[..soft])?;
        Ok((
            gw,
            gh,
            soft,
            (t1 - t0).as_secs_f64(),
            t1.elapsed().as_secs_f64(),
        ))
    }

    /// The patches, their positions and their RoPE tables into the buffers; the grid's `(width,
    /// height)` in patches.
    fn prepare(&mut self, rgb: &[u8], h: usize, w: usize) -> Result<(usize, usize)> {
        let (patch, k) = (self.cfg.patch_size, self.cfg.pooling_kernel_size);
        if rgb.len() != h * w * 3 {
            return Err(HerbertError::Backend(format!(
                "an image of {w} × {h} pixels in {} bytes",
                rgb.len()
            )));
        }
        let (th, tw) = target_size(h, w, patch, IMAGE_ROWS * k * k, k)?;
        let resized = if th == h && tw == w {
            rgb.to_vec()
        } else {
            resize(rgb, h, w, th, tw)
        };
        let (gw, gh) = (tw / patch, th / patch);
        if gw * gh > self.shape.max_rows() || !gw.is_multiple_of(k) || !gh.is_multiple_of(k) {
            return Err(HerbertError::Backend(format!(
                "an image of {gw} × {gh} patches does not fit the tower"
            )));
        }
        // `2·(v/255 − ½)` of every 8-bit value: rescaled in f64 then f32, `2·(x − ½)` in f32, then
        // bf16.
        let lut: Vec<u16> = (0..256u32)
            .map(|v| to_bf16(2.0 * ((v as f64 * 0.00392156862745098) as f32 - 0.5)))
            .collect();
        let hd = self.cfg.head_dim;
        let (half, quarter) = (hd / 2, hd / 4);
        let inv: Vec<f32> = (0..quarter)
            .map(|i| (1.0 / self.cfg.rope_theta.powf(2.0 * i as f64 / half as f64)) as f32)
            .collect();
        let pix = 3 * patch * patch;
        let b = &mut *self.buffers;
        let mut row = 0;
        for cy in 0..gh / k {
            for cx in 0..gw / k {
                for dy in 0..k {
                    for dx in 0..k {
                        let (px, py) = (cx * k + dx, cy * k + dy);
                        let dst = &mut b.patches[row * pix..(row + 1) * pix];
                        for y in 0..patch {
                            let at = ((py * patch + y) * tw + px * patch) * 3;
                            for (d, &v) in dst[y * patch * 3..(y + 1) * patch * 3]
                                .iter_mut()
                                .zip(&resized[at..at + patch * 3])
                            {
                                *d = lut[v as usize];
                            }
                        }
                        b.pos_x[row] = px as u32;
                        b.pos_y[row] = py as u32;
                        for (axis, p) in [px, py].into_iter().enumerate() {
                            for (i, &f) in inv.iter().enumerate() {
                                let a = p as u32 as f32 * f;
                                let at = row * half + axis * quarter + i;
                                b.cos[at] = to_bf16(a.cos());
                                b.sin[at] = to_bf16(a.sin());
                            }
                        }
                        row += 1;
                    }
                }
            }
        }
        Ok((gw, gh))
    }
}

/// 12B's encoder-free embedder mounted (`Gemma4UnifiedVisionEmbedder`): its weights, its buffers,
/// its program (`lower_unified` in `jit/schedule/vision.rs`).
pub struct Unified {
    pub cfg: VisionConfig,
    shape: UnifiedShape,
    // The program holds the addresses of everything below: declared first, dropped first.
    program: Mounted,
    buffers: Box<UnifiedBuffers>,
    #[allow(dead_code)]
    weights: [AVec<u16>; 5],
    #[allow(dead_code)]
    dense: PackedBf16R16Weight,
    #[allow(dead_code)]
    proj: PackedBf16R16Weight,
    tokens: Vec<u32>,
}

impl Unified {
    /// Read, pack and mount the embedder of `dir` for soft tokens of `out_hidden` values, `workers`
    /// participants.
    pub(crate) fn load(
        dir: &Path,
        prefix: &str,
        cfg: VisionConfig,
        out_hidden: usize,
        runner: &dyn Apart,
        workers: usize,
    ) -> Result<Unified> {
        let t0 = Instant::now();
        let mut t = read_tensors(dir, prefix)?;
        let d = cfg.hidden_size;
        let side = cfg.patch_size * cfg.pooling_kernel_size;
        let pix = 3 * side * side;
        let shape = UnifiedShape {
            hidden: d,
            pixels: pix,
            positions: cfg.mm_posemb_size,
            // `nn.LayerNorm`'s default.
            eps_ln: 1e-5,
            eps_rms: cfg.rms_norm_eps,
            max_soft: IMAGE_ROWS,
            out_hidden,
        };
        let mut take = |name: &str, len: usize| -> Result<Vec<u16>> {
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
        // A LayerNorm as `ElemOp::LayerNorm` reads it: `g` then `b`.
        let mut ln = |name: &str, n: usize| -> Result<AVec<u16>> {
            let mut g = take(&format!("vision_embedder.{name}.weight"), n)?;
            g.extend(take(&format!("vision_embedder.{name}.bias"), n)?);
            Ok(AVec::from_vec(g))
        };
        let (ln1, ln2, ln3) = (
            ln("patch_ln1", pix)?,
            ln("patch_ln2", d)?,
            ln("pos_norm", d)?,
        );
        // The dense with its bias as the input column `pix` (the buffers hold a 1 there), zeros
        // after.
        let kd = shape.dense_k();
        let w = take("vision_embedder.patch_dense.weight", d * pix)?;
        let bias = take("vision_embedder.patch_dense.bias", d)?;
        let mut wb = vec![0u16; d * kd];
        for n in 0..d {
            wb[n * kd..n * kd + pix].copy_from_slice(&w[n * pix..(n + 1) * pix]);
            wb[n * kd + pix] = bias[n];
        }
        drop(w);
        let dense = PackedBf16R16Weight::pack_bf16(&wb, d, kd)?;
        drop(wb);
        // `pos_embedding` `[positions][2][d]` split into its `x` and `y` tables, `[positions][d]`
        // each.
        let pos = take("vision_embedder.pos_embedding", cfg.mm_posemb_size * 2 * d)?;
        let (mut px, mut py) = (
            Vec::with_capacity(cfg.mm_posemb_size * d),
            Vec::with_capacity(cfg.mm_posemb_size * d),
        );
        for row in pos.chunks_exact(2 * d) {
            px.extend_from_slice(&row[..d]);
            py.extend_from_slice(&row[d..]);
        }
        let proj = PackedBf16R16Weight::pack_bf16(
            &take("embed_vision.embedding_projection.weight", out_hidden * d)?,
            out_hidden,
            d,
        )?;
        let weights = [ln1, ln2, ln3, AVec::from_vec(px), AVec::from_vec(py)];
        let t_read = t0.elapsed().as_secs_f64();
        let mut buffers = Box::new(UnifiedBuffers::new(&shape));
        let w = UnifiedWeights {
            ln1: &weights[0],
            dense: &dense.data_r16,
            ln2: &weights[1],
            pos_x: &weights[3],
            pos_y: &weights[4],
            ln3: &weights[2],
            proj: &proj.data_r16,
        };
        let program = lower_unified(&shape, &w, &mut buffers, workers, runner.apart_spin())
            .map_err(|e| HerbertError::Backend(format!("vision: {e}")))?;
        eprintln!(
            "[vision] encoder-free embedder: patches of {side}×{side}, width {d}; read and packed in {t_read:.2} s, mounted in {:.2} s ({:.0} MiB of buffers, {} bytes of code)",
            t0.elapsed().as_secs_f64() - t_read,
            buffers.bytes() as f64 / (1u64 << 20) as f64,
            program.program.code().len()
        );
        Ok(Unified {
            cfg,
            shape,
            program,
            buffers,
            weights,
            dense,
            proj,
            tokens: vec![0u32; IMAGE_ROWS],
        })
    }

    /// The soft tokens, `[max_soft][out_hidden]`: the table of the image regime.
    pub fn table(&self) -> &[u16] {
        &self.buffers.out
    }

    /// The image prepared (`Gemma4UnifiedImageProcessor`: the same resize, the `pool × pool` groups
    /// of `patch`-wide patches merged, which is the image cut in `patch · pool`-wide patches in
    /// raster order, each its pixels row by row, channels interleaved, `p/255` not centered)
    /// and through the embedder. As [`Tower::encode`].
    pub(crate) fn encode(
        &mut self,
        runner: &mut dyn Apart,
        rgb: &[u8],
        h: usize,
        w: usize,
    ) -> Result<(usize, usize, usize, f64, f64)> {
        let t0 = Instant::now();
        let (patch, k) = (self.cfg.patch_size, self.cfg.pooling_kernel_size);
        if rgb.len() != h * w * 3 {
            return Err(HerbertError::Backend(format!(
                "an image of {w} × {h} pixels in {} bytes",
                rgb.len()
            )));
        }
        let (th, tw) = target_size(h, w, patch, IMAGE_ROWS * k * k, k)?;
        let resized = if th == h && tw == w {
            rgb.to_vec()
        } else {
            resize(rgb, h, w, th, tw)
        };
        let side = patch * k;
        let (gw, gh) = (tw / side, th / side);
        let soft = gw * gh;
        if soft > IMAGE_ROWS || gw > self.shape.positions || gh > self.shape.positions {
            return Err(HerbertError::Backend(format!(
                "an image of {gw} × {gh} patches does not fit the embedder"
            )));
        }
        // `v/255` of every 8-bit value, rescaled in f64 then f32, then bf16 (`do_normalize` off).
        let lut: Vec<u16> = (0..256u32)
            .map(|v| to_bf16((v as f64 * 0.00392156862745098) as f32))
            .collect();
        let pix = self.shape.pixels;
        let b = &mut *self.buffers;
        for py in 0..gh {
            for px in 0..gw {
                let row = py * gw + px;
                let dst = &mut b.patches[row * pix..(row + 1) * pix];
                for y in 0..side {
                    let at = ((py * side + y) * tw + px * side) * 3;
                    for (d, &v) in dst[y * side * 3..(y + 1) * side * 3]
                        .iter_mut()
                        .zip(&resized[at..at + side * 3])
                    {
                        *d = lut[v as usize];
                    }
                }
                b.pos_x[row] = px as u32;
                b.pos_y[row] = py as u32;
            }
        }
        let t1 = Instant::now();
        run(runner, &mut self.program, &self.tokens[..soft])?;
        Ok((
            gw * k,
            gh * k,
            soft,
            (t1 - t0).as_secs_f64(),
            t1.elapsed().as_secs_f64(),
        ))
    }
}

/// The vision of a checkpoint mounted: the tower (E2B, E4B) or the encoder-free embedder (12B).
pub enum Encoder {
    Tower(Box<Tower>),
    Unified(Box<Unified>),
}

impl Encoder {
    /// Read, pack and mount what `cfg` describes.
    pub(crate) fn load(
        dir: &Path,
        prefix: &str,
        cfg: VisionConfig,
        out_hidden: usize,
        runner: &dyn Apart,
        workers: usize,
    ) -> Result<Encoder> {
        Ok(if cfg.unified {
            Encoder::Unified(Box::new(Unified::load(
                dir, prefix, cfg, out_hidden, runner, workers,
            )?))
        } else {
            Encoder::Tower(Box::new(Tower::load(
                dir, prefix, cfg, out_hidden, runner, workers,
            )?))
        })
    }

    /// The soft tokens' table.
    pub fn table(&self) -> &[u16] {
        match self {
            Encoder::Tower(t) => t.table(),
            Encoder::Unified(u) => u.table(),
        }
    }

    /// The image through it: `(grid_w, grid_h` in patches of `patch_size`, soft tokens, seconds of
    /// the preparation, seconds of the encoder`)`.
    pub(crate) fn encode(
        &mut self,
        runner: &mut dyn Apart,
        rgb: &[u8],
        h: usize,
        w: usize,
    ) -> Result<(usize, usize, usize, f64, f64)> {
        match self {
            Encoder::Tower(t) => t.encode(runner, rgb, h, w),
            Encoder::Unified(u) => u.encode(runner, rgb, h, w),
        }
    }
}

/// The times of the layers' program by operation: a node lasts from the last departure from the
/// previous barrier to the last departure from its own; the stamps scaled to the call's wall time.
/// The times of a traced `program` run (`seconds` of wall time), summed by operation, on stderr
/// after `[tag]`.
pub(crate) fn print_profile_as(tag: &str, runner: &dyn Apart, program: &Mounted, seconds: f64) {
    let Some(trace) = runner.apart_sync_trace() else {
        return;
    };
    let (p, per) = (trace.participants, trace.nodes_per_block);
    let labels = program.program.plan().labels();
    let leave = |node: usize| {
        trace.records[node * p..(node + 1) * p]
            .iter()
            .map(|r| r.barrier_end)
            .max()
            .unwrap_or(0)
    };
    let first = trace.records[..p]
        .iter()
        .map(|r| r.work_start)
        .min()
        .unwrap_or(0);
    let last = leave(per.min(labels.len()) - 1);
    let tick = seconds / (last.saturating_sub(first)).max(1) as f64;
    let mut by: Vec<(&str, f64, usize)> = Vec::new();
    let mut prev = first;
    for (node, label) in labels.iter().enumerate().take(per) {
        let end = leave(node);
        let d = end.saturating_sub(prev) as f64 * tick;
        prev = end;
        match by.iter_mut().find(|(l, _, _)| *l == label.operation) {
            Some(e) => {
                e.1 += d;
                e.2 += 1;
            }
            None => by.push((label.operation, d, 1)),
        }
    }
    by.sort_by(|a, b| b.1.total_cmp(&a.1));
    let line: Vec<String> = by
        .iter()
        .map(|(l, d, n)| format!("{l} {:.1} ms ({n})", d * 1e3))
        .collect();
    eprintln!(
        "[{tag}] profile ({:.1} ms): {}",
        seconds * 1e3,
        line.join(", ")
    );
}

fn run(runner: &mut dyn Apart, program: &mut Mounted, tokens: &[u32]) -> Result<()> {
    runner.run_apart(program, tokens)
}

/// The size the processor resizes to (`get_aspect_ratio_preserving_size`, `herbert_chat::media`).
pub fn target_size(
    h: usize,
    w: usize,
    patch: usize,
    max_patches: usize,
    pool: usize,
) -> Result<(usize, usize)> {
    herbert_chat::media::target_size(h, w, patch, max_patches, pool).map_err(HerbertError::Backend)
}

fn bicubic(x: f64) -> f64 {
    let a = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * a
    } else {
        0.0
    }
}

/// Pillow's `precompute_coeffs` for the bicubic filter, in 8-bit fixed point:
/// per output pixel, `(first input index, coefficients)`.
fn coefficients(input: usize, output: usize) -> Vec<(usize, Vec<i64>)> {
    let scale = input as f64 / output as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let ss = 1.0 / filterscale;
    (0..output)
        .map(|xx| {
            let center = (xx as f64 + 0.5) * scale;
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax =
                (((center + support + 0.5) as i64).min(input as i64) as usize).saturating_sub(xmin);
            let mut k: Vec<f64> = (0..xmax)
                .map(|x| bicubic((x as f64 + xmin as f64 - center + 0.5) * ss))
                .collect();
            let ww: f64 = k.iter().sum();
            if ww != 0.0 {
                k.iter_mut().for_each(|v| *v /= ww);
            }
            let fixed = k
                .iter()
                .map(|&v| {
                    if v < 0.0 {
                        (-0.5 + v * (1 << 22) as f64) as i64
                    } else {
                        (0.5 + v * (1 << 22) as f64) as i64
                    }
                })
                .collect();
            (xmin, fixed)
        })
        .collect()
}

fn clip8(v: i64) -> u8 {
    (v >> 22).clamp(0, 255) as u8
}

/// RGB `[h][w][3]` resized to `(th, tw)` the way Pillow does: horizontal pass, then vertical.
pub(crate) fn resize(src: &[u8], h: usize, w: usize, th: usize, tw: usize) -> Vec<u8> {
    let mut cur = src.to_vec();
    let mut cw = w;
    if tw != w {
        let coeffs = coefficients(w, tw);
        let mut out = vec![0u8; h * tw * 3];
        for y in 0..h {
            for (x, (xmin, k)) in coeffs.iter().enumerate() {
                for c in 0..3 {
                    let mut ss = 1i64 << 21;
                    for (i, &kv) in k.iter().enumerate() {
                        ss += cur[(y * cw + xmin + i) * 3 + c] as i64 * kv;
                    }
                    out[(y * tw + x) * 3 + c] = clip8(ss);
                }
            }
        }
        cur = out;
        cw = tw;
    }
    if th != h {
        let coeffs = coefficients(h, th);
        let mut out = vec![0u8; th * cw * 3];
        for (y, (ymin, k)) in coeffs.iter().enumerate() {
            for x in 0..cw {
                for c in 0..3 {
                    let mut ss = 1i64 << 21;
                    for (i, &kv) in k.iter().enumerate() {
                        ss += cur[((ymin + i) * cw + x) * 3 + c] as i64 * kv;
                    }
                    out[(y * cw + x) * 3 + c] = clip8(ss);
                }
            }
        }
        cur = out;
    }
    cur
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A grid of whole cells within the budget.
    #[test]
    fn the_grid_is_whole_cells_within_the_budget() {
        for (h, w) in [(1664, 928), (282, 1601), (768, 768), (60, 2000)] {
            let (th, tw) = target_size(h, w, 16, IMAGE_ROWS * 9, 3).unwrap();
            let (gh, gw) = (th / 16, tw / 16);
            assert!(
                gw.is_multiple_of(3) && gh.is_multiple_of(3),
                "{h}×{w}: {gh}×{gw}"
            );
            assert!(gw * gh / 9 <= IMAGE_ROWS);
        }
    }
}
