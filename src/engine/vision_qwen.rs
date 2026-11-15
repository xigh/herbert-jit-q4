//! The vision of Qwen3.5 / Qwen3.6 (`qwen3_5`, `qwen3_5_moe`) and of Qwen3-VL (`qwen3_vl_moe`): its
//! configuration, the processor's resize (`smart_resize`), the tower's weights read and packed at
//! the first image, the image laid out as the processor writes it, and the two generated programs
//! of `jit/schedule/vision_qwen.rs` that turn it into soft tokens; then the multimodal RoPE rows of
//! the soft tokens in the language model.
//!
//! The preparation, as transformers 5.22.1's `Qwen2VLImageProcessorFast` and
//! `Qwen3_5MoeVisionModel`:
//!
//! - the size: `smart_resize` with factor `patch · merge` (32) between the `shortest_edge` and
//!   `longest_edge` pixels of `preprocessor_config.json`; the resize is torchvision's on the 8-bit
//!   image (`tvF.resize`, bicubic with antialias: ATen's separable fixed-point kernel on `u8`,
//!   [`resize_torch`]), which differs from Pillow's by the precision of its weights (16 bits rather
//!   than 22) and then by one level on a few values in a thousand;
//! - the values `(v − 127.5) / 127.5` in f32 then bf16, each patch `(c, t, y, x)` with its frame
//!   twice (`temporal_patch_size` 2), the patches in merge-block order;
//! - the learned positions (`pos_embed`, a `48 × 48` grid) resampled to the image's grid by the
//!   bilinear interpolation with aligned corners of `get_vision_interpolation_indices_and_weights`:
//!   per axis `src = i · 47 / (n − 1)` in f32, the taps `⌊src⌋` and `⌊src⌋ + 1` (clamped) with
//!   weights `1 − |src − ⌊src⌋ − o|`, the four products `e · wₕ · w_w` in f32 summed in order, one
//!   rounding;
//! - the RoPE of the tower: axial, the frequencies `θ^(−2i / (head_dim / 2))` of a quarter head,
//!   row `h` angles then `w` angles, f32, rounded to the bf16 of the kernel's tables.
//!
//! Qwen3-VL (`Qwen3VLMoeVisionModel`) is the same tower, processor and multimodal RoPE
//! (`get_rope_index`, `mrope_section` interleaved), with its deepstack: after the layers of
//! `deepstack_visual_indexes`, a merger of its own each (`deepstack_merger_list`, a LayerNorm over
//! the `merge²` patches of a soft token at once) makes the vectors the language model adds to the
//! image's rows after its first layers ([`QwenTower::deepstack`]).

use crate::core::avec::AVec;
use crate::core::error::{HerbertError, Result};
use crate::engine::packs::to_bf16;
use crate::jit::model::apart::Apart;
use crate::jit::schedule::vision_qwen::{
    lower_qwen_tower, padded, QwenDeepstackWeights, QwenLayerWeights, QwenTowerBuffers,
    QwenTowerPrograms, QwenTowerShape, QwenTowerWeights,
};
use crate::model::PackedBf16R16Weight;
use std::path::Path;
use std::time::Instant;

/// The tower of `config.json` (`vision_config`) and what the processor and the language model's
/// RoPE need.
#[derive(Clone, Debug)]
pub struct QwenVisionConfig {
    pub hidden_size: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge: usize,
    pub out_hidden_size: usize,
    /// `num_position_embeddings`: a square grid of learned positions.
    pub num_position_embeddings: usize,
    pub rope_theta: f64,
    /// `preprocessor_config.json`: `size.shortest_edge`, `size.longest_edge` (pixels, `min_pixels`
    /// and `max_pixels`).
    pub min_pixels: usize,
    pub max_pixels: usize,
    /// The language model's multimodal RoPE (`text_config.rope_parameters`): `mrope_section`,
    /// `mrope_interleaved`.
    pub mrope_section: [usize; 3],
    pub mrope_interleaved: bool,
    /// Qwen3-VL: `deepstack_visual_indexes`, the layers after which a deepstack merger reads the
    /// hidden state (8, 16, 24); empty for Qwen3.5 / Qwen3.6.
    pub deepstack: Vec<usize>,
}

impl QwenVisionConfig {
    /// The vision of the checkpoint in `dir`, when it has one this engine serves: the tower of
    /// Qwen3.5 / Qwen3.6, or of Qwen3-VL with its deepstack.
    pub fn of(dir: &Path) -> Option<QwenVisionConfig> {
        let root: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).ok()?).ok()?;
        let v = root.get("vision_config").filter(|v| v.is_object())?;
        let model_type = v.get("model_type").and_then(|t| t.as_str()).unwrap_or("");
        if !matches!(model_type, "qwen3_5" | "qwen3_5_moe" | "qwen3_vl_moe") {
            return None;
        }
        let deepstack: Vec<usize> = match v.get("deepstack_visual_indexes") {
            None | Some(serde_json::Value::Null) => Vec::new(),
            Some(d) => d
                .as_array()?
                .iter()
                .map(|i| i.as_u64().map(|n| n as usize))
                .collect::<Option<Vec<_>>>()?,
        };
        // The language model must add as many as the tower makes (`Config::qwen_vl`,
        // `deepstack_layers`).
        if !deepstack.is_empty() && model_type != "qwen3_vl_moe" {
            return None;
        }
        let u = |key: &str| {
            v.get(key)
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
        };
        let rope = root
            .get("text_config")
            .and_then(|t| t.get("rope_parameters").or_else(|| t.get("rope_scaling")))?;
        let section: Vec<usize> = rope
            .get("mrope_section")?
            .as_array()?
            .iter()
            .filter_map(serde_json::Value::as_u64)
            .map(|n| n as usize)
            .collect();
        let pre: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("preprocessor_config.json")).ok()?)
                .ok()?;
        let size = pre.get("size")?;
        let c = QwenVisionConfig {
            hidden_size: u("hidden_size")?,
            // `HERBERT_VISION_DEPTH=N` (diagnostic): the first `N` layers only, then the merger, to
            // follow the gap to a reference truncated the same way layer by layer.
            depth: u("depth")?.min(
                std::env::var("HERBERT_VISION_DEPTH")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(usize::MAX),
            ),
            num_heads: u("num_heads")?,
            intermediate_size: u("intermediate_size")?,
            patch_size: u("patch_size")?,
            temporal_patch_size: u("temporal_patch_size")?,
            merge: u("spatial_merge_size")?,
            out_hidden_size: u("out_hidden_size")?,
            num_position_embeddings: u("num_position_embeddings")?,
            rope_theta: v
                .get("rope_parameters")
                .and_then(|r| r.get("rope_theta"))
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(10_000.0),
            min_pixels: size
                .get("shortest_edge")
                .and_then(serde_json::Value::as_u64)? as usize,
            max_pixels: size
                .get("longest_edge")
                .and_then(serde_json::Value::as_u64)? as usize,
            mrope_section: <[usize; 3]>::try_from(section.as_slice()).ok()?,
            mrope_interleaved: rope
                .get("mrope_interleaved")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            deepstack,
        };
        let side = (c.num_position_embeddings as f64).sqrt() as usize;
        let served = c.num_heads > 0
            && c.hidden_size.is_multiple_of(c.num_heads)
            && c.head_dim().is_multiple_of(8)
            && c.hidden_size.is_multiple_of(32)
            && c.intermediate_size.is_multiple_of(16)
            && c.merge > 0
            && side * side == c.num_position_embeddings
            && c.min_pixels <= c.max_pixels
            && c.deepstack.windows(2).all(|d| d[0] < d[1])
            && c.deepstack
                .last()
                .is_none_or(|&d| d < u("depth").unwrap_or(0));
        served.then_some(c)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// The size the processor resizes an `h × w` image to (`smart_resize`, factor `patch · merge`).
    pub fn smart_resize(&self, h: usize, w: usize) -> Result<(usize, usize)> {
        let factor = (self.patch_size * self.merge) as f64;
        let (hf, wf) = (h as f64, w as f64);
        if h == 0 || w == 0 || hf.max(wf) / hf.min(wf) > 200.0 {
            return Err(HerbertError::Backend(format!(
                "an image of {w} × {h} pixels: aspect ratio over 200"
            )));
        }
        // Python's `round`: half to even.
        let round = |x: f64| x.round_ties_even();
        let mut hb = round(hf / factor) * factor;
        let mut wb = round(wf / factor) * factor;
        if hb * wb > self.max_pixels as f64 {
            let beta = ((hf * wf) / self.max_pixels as f64).sqrt();
            hb = factor.max((hf / beta / factor).floor() * factor);
            wb = factor.max((wf / beta / factor).floor() * factor);
        } else if hb * wb < self.min_pixels as f64 {
            let beta = (self.min_pixels as f64 / (hf * wf)).sqrt();
            hb = (hf * beta / factor).ceil() * factor;
            wb = (wf * beta / factor).ceil() * factor;
        }
        Ok((hb as usize, wb as usize))
    }

    /// The soft tokens of an `h × w` image and the RoPE positions they take in the language model
    /// (`max(grid_h, grid_w) / merge`, `get_rope_index`).
    pub fn tokens(&self, h: usize, w: usize) -> Result<(usize, usize)> {
        let (th, tw) = self.smart_resize(h, w)?;
        let side = self.patch_size * self.merge;
        let (mh, mw) = (th / side, tw / side);
        Ok((mh * mw, mh.max(mw)))
    }
}

/// The tower mounted: its weights, its buffers and programs for `shape.max_rows` patches (lowered
/// again for a larger image).
pub struct QwenTower {
    pub cfg: QwenVisionConfig,
    // The programs hold the addresses of the buffers and the weights: dropped first.
    programs: Option<QwenTowerPrograms>,
    buffers: Option<Box<QwenTowerBuffers>>,
    shape: QwenTowerShape,
    patch: PackedBf16R16Weight,
    pos_embed: Vec<u16>,
    norms: Vec<[AVec<u16>; 2]>,
    linears: Vec<[PackedBf16R16Weight; 6]>,
    merger_norm: AVec<u16>,
    merger_fc: [PackedBf16R16Weight; 2],
    /// Qwen3-VL: per deepstack merger, its layer, its LayerNorm (`g` then `b`, `merged` each) and
    /// `[fc1, fc2]`.
    deepstack: Vec<(usize, AVec<u16>, [PackedBf16R16Weight; 2])>,
    workers: usize,
    spin: Option<u32>,
    profile: bool,
    tokens: Vec<u32>,
}

/// `w` (`n × k`) with `bias` in the column `k` of [`padded`]`(k)`, zeros after, packed.
fn pack_with_bias(w: &[u16], bias: &[u16], n: usize, k: usize) -> Result<PackedBf16R16Weight> {
    let kp = padded(k);
    let mut wb = vec![0u16; n * kp];
    for r in 0..n {
        wb[r * kp..r * kp + k].copy_from_slice(&w[r * k..(r + 1) * k]);
        wb[r * kp + k] = bias[r];
    }
    PackedBf16R16Weight::pack_bf16(&wb, n, kp)
}

impl QwenTower {
    /// Read and pack the tower of `dir` (`model.visual.`), `workers` participants; mounted at the
    /// first image.
    pub(crate) fn load(
        dir: &Path,
        cfg: QwenVisionConfig,
        runner: &dyn Apart,
        workers: usize,
    ) -> Result<QwenTower> {
        let t0 = Instant::now();
        let mut t = crate::engine::vision::read_tensors_under(dir, "model.", &["visual."])?;
        let (h, inter, m) = (
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.hidden_size * cfg.merge * cfg.merge,
        );
        let pix = 3 * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
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
        let patch = pack_with_bias(
            &take("visual.patch_embed.proj.weight", h * pix)?,
            &take("visual.patch_embed.proj.bias", h)?,
            h,
            pix,
        )?;
        let pos_embed = take("visual.pos_embed.weight", cfg.num_position_embeddings * h)?;
        let mut norms = Vec::with_capacity(cfg.depth);
        let mut linears = Vec::with_capacity(cfg.depth);
        let ln = |take: &mut dyn FnMut(&str, usize) -> Result<Vec<u16>>,
                  name: &str|
         -> Result<AVec<u16>> {
            let mut g = take(&format!("{name}.weight"), h)?;
            g.extend(take(&format!("{name}.bias"), h)?);
            Ok(AVec::from_vec(g))
        };
        for l in 0..cfg.depth {
            let p = format!("visual.blocks.{l}.");
            norms.push([
                ln(&mut take, &format!("{p}norm1"))?,
                ln(&mut take, &format!("{p}norm2"))?,
            ]);
            let qkv = take(&format!("{p}attn.qkv.weight"), 3 * h * h)?;
            let qkv_b = take(&format!("{p}attn.qkv.bias"), 3 * h)?;
            let part = |i: usize| {
                pack_with_bias(
                    &qkv[i * h * h..(i + 1) * h * h],
                    &qkv_b[i * h..(i + 1) * h],
                    h,
                    h,
                )
            };
            linears.push([
                part(0)?,
                part(1)?,
                part(2)?,
                pack_with_bias(
                    &take(&format!("{p}attn.proj.weight"), h * h)?,
                    &take(&format!("{p}attn.proj.bias"), h)?,
                    h,
                    h,
                )?,
                pack_with_bias(
                    &take(&format!("{p}mlp.linear_fc1.weight"), inter * h)?,
                    &take(&format!("{p}mlp.linear_fc1.bias"), inter)?,
                    inter,
                    h,
                )?,
                pack_with_bias(
                    &take(&format!("{p}mlp.linear_fc2.weight"), h * inter)?,
                    &take(&format!("{p}mlp.linear_fc2.bias"), h)?,
                    h,
                    inter,
                )?,
            ]);
        }
        let merger_norm = ln(&mut take, "visual.merger.norm")?;
        let merger_fcs = |take: &mut dyn FnMut(&str, usize) -> Result<Vec<u16>>,
                          p: &str|
         -> Result<[PackedBf16R16Weight; 2]> {
            Ok([
                pack_with_bias(
                    &take(&format!("{p}.linear_fc1.weight"), m * m)?,
                    &take(&format!("{p}.linear_fc1.bias"), m)?,
                    m,
                    m,
                )?,
                pack_with_bias(
                    &take(&format!("{p}.linear_fc2.weight"), cfg.out_hidden_size * m)?,
                    &take(&format!("{p}.linear_fc2.bias"), cfg.out_hidden_size)?,
                    cfg.out_hidden_size,
                    m,
                )?,
            ])
        };
        let merger_fc = merger_fcs(&mut take, "visual.merger")?;
        // Qwen3-VL: the deepstack mergers whose layer the tower runs (`HERBERT_VISION_DEPTH` may
        // cut it short).
        let mut deepstack = Vec::new();
        for (i, &after) in cfg.deepstack.iter().enumerate() {
            let p = format!("visual.deepstack_merger_list.{i}");
            let mut g = take(&format!("{p}.norm.weight"), m)?;
            g.extend(take(&format!("{p}.norm.bias"), m)?);
            let fcs = merger_fcs(&mut take, &p)?;
            if after < cfg.depth {
                deepstack.push((after, AVec::from_vec(g), fcs));
            }
        }
        let shape = QwenTowerShape {
            hidden: h,
            heads: cfg.num_heads,
            head_dim: cfg.head_dim(),
            intermediate: inter,
            pixels: pix,
            merge: cfg.merge,
            out_hidden: cfg.out_hidden_size,
            max_rows: 0,
            deepstack: deepstack.len(),
        };
        let profile =
            cfg!(target_arch = "x86_64") && std::env::var_os("HERBERT_VISION_PROFILE").is_some();
        eprintln!("[vision] Qwen tower read and packed in {:.2} s ({} layers, width {h}, heads of {}, deepstack after {:?})", t0.elapsed().as_secs_f64(), cfg.depth, cfg.head_dim(), cfg.deepstack);
        Ok(QwenTower {
            cfg,
            programs: None,
            buffers: None,
            shape,
            patch,
            pos_embed,
            norms,
            linears,
            merger_norm,
            merger_fc,
            deepstack,
            workers,
            spin: runner.apart_spin(),
            profile,
            tokens: Vec::new(),
        })
    }

    /// The soft tokens, `[max_soft][out_hidden]`: the table of the media regime (empty before the
    /// first image).
    pub fn table(&self) -> &[u16] {
        self.buffers.as_ref().map_or(&[][..], |b| &b.out[..])
    }

    /// Qwen3-VL: the deepstack vectors, `[deepstack][max_soft][out_hidden]` (empty without
    /// deepstack or before the first image); vectors `i` of soft token `r` at `(i · max_soft +
    /// r) · out_hidden`.
    pub fn deepstack(&self) -> (&[u16], usize) {
        (
            self.buffers.as_ref().map_or(&[][..], |b| &b.deep[..]),
            self.shape.max_soft(),
        )
    }

    /// The deepstack vectors, writable (`HERBERT_VISION_DEEPSTACK`: a reference's in their place).
    pub fn deepstack_mut(&mut self) -> &mut [u16] {
        self.buffers
            .as_mut()
            .map_or(&mut [][..], |b| &mut b.deep[..])
    }

    /// The soft tokens' table, writable (`HERBERT_VISION_SOFT`: the merger's output of a reference
    /// in its place).
    pub fn table_mut(&mut self) -> &mut [u16] {
        self.buffers
            .as_mut()
            .map_or(&mut [][..], |b| &mut b.out[..])
    }

    /// The programs and buffers for `rows` patches at least; `true` when they were lowered again
    /// (the table moved).
    fn reserve(&mut self, rows: usize) -> Result<bool> {
        if self.programs.is_some() && self.shape.max_rows >= rows {
            return Ok(false);
        }
        let t0 = Instant::now();
        self.programs = None;
        self.buffers = None;
        // Room by steps of 1 024 patches (256 soft tokens), at least 4 096.
        let shape = QwenTowerShape {
            max_rows: rows.max(4096).next_multiple_of(1024),
            ..self.shape
        };
        let mut buffers = Box::new(QwenTowerBuffers::new(&shape, self.workers));
        let weights = QwenTowerWeights {
            patch: &self.patch.data_r16,
            layers: self
                .norms
                .iter()
                .zip(&self.linears)
                .map(|(n, l)| QwenLayerWeights {
                    norm1: &n[0],
                    norm2: &n[1],
                    linears: [
                        &l[0].data_r16,
                        &l[1].data_r16,
                        &l[2].data_r16,
                        &l[3].data_r16,
                        &l[4].data_r16,
                        &l[5].data_r16,
                    ],
                })
                .collect(),
            merger_norm: &self.merger_norm,
            merger_fc1: &self.merger_fc[0].data_r16,
            merger_fc2: &self.merger_fc[1].data_r16,
            deepstack: self
                .deepstack
                .iter()
                .map(|(after, norm, fc)| QwenDeepstackWeights {
                    after: *after,
                    norm,
                    fc1: &fc[0].data_r16,
                    fc2: &fc[1].data_r16,
                })
                .collect(),
        };
        let programs = lower_qwen_tower(
            &shape,
            &weights,
            &mut buffers,
            self.workers,
            self.spin,
            self.profile,
        )
        .map_err(|e| HerbertError::Backend(format!("vision: {e}")))?;
        drop(weights);
        eprintln!(
            "[vision] Qwen tower mounted for {} patches in {:.2} s ({:.0} MiB of buffers, {} + {} bytes of code)",
            shape.max_rows,
            t0.elapsed().as_secs_f64(),
            buffers.bytes() as f64 / (1u64 << 20) as f64,
            programs.layers.iter().map(|p| p.program.code().len()).sum::<usize>(),
            programs.merger.program.code().len() + programs.deepstack.iter().map(|p| p.program.code().len()).sum::<usize>()
        );
        self.shape = shape;
        self.buffers = Some(buffers);
        self.programs = Some(programs);
        self.tokens = vec![0u32; shape.max_rows];
        Ok(true)
    }

    /// The image (`rgb`: `h` rows of `w` pixels) prepared and through the tower: its soft tokens in
    /// [`Self::table`]. `(merged grid height, merged grid width, whether the table moved,
    /// seconds of the preparation, of the tower)`.
    pub(crate) fn encode(
        &mut self,
        runner: &mut dyn Apart,
        rgb: &[u8],
        h: usize,
        w: usize,
    ) -> Result<(usize, usize, bool, f64, f64)> {
        let t0 = Instant::now();
        if rgb.len() != h * w * 3 {
            return Err(HerbertError::Backend(format!(
                "an image of {w} × {h} pixels in {} bytes",
                rgb.len()
            )));
        }
        let (th, tw) = self.cfg.smart_resize(h, w)?;
        let (patch, k) = (self.cfg.patch_size, self.cfg.merge);
        let (gh, gw) = (th / patch, tw / patch);
        let rows = gh * gw;
        let moved = self.reserve(rows)?;
        let resized = if th == h && tw == w {
            rgb.to_vec()
        } else {
            resize_torch(rgb, h, w, th, tw)
        };
        self.prepare(&resized, gh, gw)?;
        let t1 = Instant::now();
        let soft = rows / (k * k);
        let QwenTower {
            programs,
            tokens,
            profile,
            ..
        } = self;
        let programs = programs.as_mut().expect("reserved");
        // The segments of the layers, each deepstack merger after its own.
        for (i, segment) in programs.layers.iter_mut().enumerate() {
            if *profile {
                runner.reserve_apart_trace(segment)?;
            }
            let t = Instant::now();
            runner.run_apart(segment, &tokens[..rows])?;
            if *profile {
                crate::engine::vision::print_profile_as(
                    "vision",
                    runner,
                    segment,
                    t.elapsed().as_secs_f64(),
                );
            }
            if let Some(merger) = programs.deepstack.get_mut(i) {
                runner.run_apart(merger, &tokens[..soft])?;
            }
        }
        runner.run_apart(&mut programs.merger, &tokens[..soft])?;
        Ok((
            gh / k,
            gw / k,
            moved,
            (t1 - t0).as_secs_f64(),
            t1.elapsed().as_secs_f64(),
        ))
    }

    /// The patches, their learned positions and their RoPE tables into the buffers (`gh × gw`
    /// patches of `resized`). `HERBERT_VISION_PIXELS=FILE`: the patches read from `FILE`
    /// instead (f32, `[rows][3 · temporal · patch²]`, the `pixel_values` of transformers'
    /// processor), to compare the tower alone.
    fn prepare(&mut self, resized: &[u8], gh: usize, gw: usize) -> Result<()> {
        let cfg = &self.cfg;
        let (patch, k, h, tp) = (
            cfg.patch_size,
            cfg.merge,
            cfg.hidden_size,
            cfg.temporal_patch_size,
        );
        let tw = gw * patch;
        let pix = 3 * tp * patch * patch;
        let pp = padded(pix);
        let b = self.buffers.as_mut().expect("reserved");
        let lut: Vec<u16> = (0..256u32)
            .map(|v| to_bf16((v as f32 - 127.5) / 127.5))
            .collect();
        let given = std::env::var_os("HERBERT_VISION_PIXELS")
            .map(std::fs::read)
            .transpose()?;
        if let Some(bytes) = &given {
            if bytes.len() != gh * gw * pix * 4 {
                return Err(HerbertError::Backend(format!(
                    "HERBERT_VISION_PIXELS: {} bytes, {} expected",
                    bytes.len(),
                    gh * gw * pix * 4
                )));
            }
        }
        // The interpolation of the learned grid, per axis: `(taps, weights)` of each index.
        let side = (cfg.num_position_embeddings as f64).sqrt() as usize;
        let axis = |n: usize| -> Vec<([usize; 2], [f32; 2])> {
            (0..n)
                .map(|i| {
                    let src = (i as f32 * (side - 1) as f32) / (n.saturating_sub(1).max(1)) as f32;
                    let floor = src.floor();
                    let mut taps = [0usize; 2];
                    let mut ws = [0f32; 2];
                    for o in 0..2 {
                        taps[o] = ((floor as i64 + o as i64).clamp(0, side as i64 - 1)) as usize;
                        let distance = (src - floor - o as f32).abs();
                        ws[o] = (1.0 - distance).max(0.0);
                    }
                    (taps, ws)
                })
                .collect()
        };
        let (ah, aw) = (axis(gh), axis(gw));
        let hd = cfg.head_dim();
        let (half, quarter) = (hd / 2, hd / 4);
        let inv: Vec<f32> = (0..quarter)
            .map(|i| 1.0 / (cfg.rope_theta as f32).powf((2 * i) as f32 / half as f32))
            .collect();
        let e = |idx: usize, c: usize| f32::from_bits(u32::from(self.pos_embed[idx * h + c]) << 16);
        let mut row = 0;
        for by in 0..gh / k {
            for bx in 0..gw / k {
                for dy in 0..k {
                    for dx in 0..k {
                        let (py, px) = (by * k + dy, bx * k + dx);
                        let dst = &mut b.patches[row * pp..row * pp + pix];
                        match &given {
                            Some(bytes) => {
                                for (j, d) in dst.iter_mut().enumerate() {
                                    let at = (row * pix + j) * 4;
                                    *d = to_bf16(f32::from_le_bytes(
                                        bytes[at..at + 4].try_into().unwrap(),
                                    ));
                                }
                            }
                            None => {
                                // `(c, t, y, x)`: the frame repeated over `t`.
                                for c in 0..3 {
                                    for t in 0..tp {
                                        for y in 0..patch {
                                            let at = (c * tp + t) * patch * patch + y * patch;
                                            let src = ((py * patch + y) * tw + px * patch) * 3 + c;
                                            for x in 0..patch {
                                                dst[at + x] = lut[resized[src + x * 3] as usize];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // The learned position: four taps, `e · (wₕ · w_w)` in f32 summed in order,
                        // one rounding.
                        let ((th, wh), (tw_, ww)) = (ah[py], aw[px]);
                        let taps = [
                            (th[0], tw_[0], wh[0] * ww[0]),
                            (th[0], tw_[1], wh[0] * ww[1]),
                            (th[1], tw_[0], wh[1] * ww[0]),
                            (th[1], tw_[1], wh[1] * ww[1]),
                        ];
                        let pos = &mut b.pos[row * h..(row + 1) * h];
                        for (c, out) in pos.iter_mut().enumerate() {
                            let mut s = 0f32;
                            for &(a, bb, wt) in &taps {
                                s += e(a * side + bb, c) * wt;
                            }
                            *out = to_bf16(s);
                        }
                        for (axis, p) in [py, px].into_iter().enumerate() {
                            for (i, &f) in inv.iter().enumerate() {
                                let a = p as f32 * f;
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
        if let Some(path) = std::env::var_os("HERBERT_VISION_PATCHES") {
            let f: Vec<u8> = (0..row)
                .flat_map(|r| b.patches[r * pp..r * pp + pix].to_vec())
                .flat_map(|v| f32::from_bits(u32::from(v) << 16).to_le_bytes())
                .collect();
            std::fs::write(path, f)?;
        }
        Ok(())
    }
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

/// ATen's antialiased bicubic weights of one axis on 8-bit images
/// (`_compute_index_ranges_int16_weights`): per output pixel `(first input index, weights)`, the
/// weights of `f64` normalized then scaled to `i16` by the largest precision at which the largest
/// weight stays under 2¹⁵; and that precision.
fn torch_weights(input: usize, output: usize) -> (Vec<(usize, Vec<i64>)>, u32) {
    let scale = input as f64 / output as f64;
    let (support, invscale) = if scale >= 1.0 {
        (2.0 * scale, 1.0 / scale)
    } else {
        (2.0, 1.0)
    };
    let mut all: Vec<(usize, Vec<f64>)> = Vec::with_capacity(output);
    for i in 0..output {
        let center = scale * (i as f64 + 0.5);
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xsize = ((center + support + 0.5) as i64).min(input as i64) as usize - xmin;
        let mut w: Vec<f64> = (0..xsize)
            .map(|j| bicubic((j as f64 + xmin as f64 - center + 0.5) * invscale))
            .collect();
        let total: f64 = w.iter().sum();
        if total != 0.0 {
            w.iter_mut().for_each(|v| *v /= total);
        }
        all.push((xmin, w));
    }
    let w_max = all
        .iter()
        .flat_map(|(_, w)| w.iter().copied())
        .fold(0.0f64, f64::max);
    let mut precision = 0u32;
    while precision < 15 {
        let next = (0.5 + w_max * (1u32 << (precision + 1)) as f64) as i64;
        if next >= 1 << 15 {
            break;
        }
        precision += 1;
    }
    let fixed = all
        .into_iter()
        .map(|(xmin, w)| {
            let k = w
                .iter()
                .map(|&v| {
                    let v = v * (1u32 << precision) as f64;
                    (if v < 0.0 { v - 0.5 } else { v + 0.5 }) as i16 as i64
                })
                .collect();
            (xmin, k)
        })
        .collect();
    (fixed, precision)
}

/// RGB `[h][w][3]` resized to `(th, tw)` as torchvision does on a `uint8` tensor (bicubic,
/// antialias): the horizontal pass, then the vertical one, each `Σ w·v + 2^(p−1)` shifted by `p`
/// and clamped to 0..255.
pub fn resize_torch(src: &[u8], h: usize, w: usize, th: usize, tw: usize) -> Vec<u8> {
    let pass = |cur: &[u8], rows: usize, cols: usize, horizontal: bool, out_n: usize| -> Vec<u8> {
        let (weights, p) = torch_weights(if horizontal { cols } else { rows }, out_n);
        let (orow, ocol) = if horizontal {
            (rows, out_n)
        } else {
            (out_n, cols)
        };
        let mut out = vec![0u8; orow * ocol * 3];
        for y in 0..orow {
            for x in 0..ocol {
                let (xmin, k) = &weights[if horizontal { x } else { y }];
                for c in 0..3 {
                    let mut ss = 1i64 << (p.max(1) - 1);
                    for (i, &kv) in k.iter().enumerate() {
                        let at = if horizontal {
                            (y * cols + xmin + i) * 3 + c
                        } else {
                            ((xmin + i) * cols + x) * 3 + c
                        };
                        ss += cur[at] as i64 * kv;
                    }
                    out[(y * ocol + x) * 3 + c] = (ss >> p).clamp(0, 255) as u8;
                }
            }
        }
        out
    };
    let mut cur = src.to_vec();
    let mut cw = w;
    if tw != w {
        cur = pass(&cur, h, w, true, tw);
        cw = tw;
    }
    if th != h {
        cur = pass(&cur, h, cw, false, th);
    }
    cur
}

/// The RoPE rows `(cos, sin)` of the language model for the positions `(t, h, w)`, frequency `i`
/// turned by its axis (`mrope_section`, interleaved or not): `angle = position · inv_freq[i]` in
/// f32, its cos and sin, bf16 (as the tables of the text:
/// [`crate::model::loader::compute_rope_cache`]).
pub fn mrope_bf16(
    cfg: &QwenVisionConfig,
    inv_freq: &[f32],
    position: [usize; 3],
    cos: &mut [u16],
    sin: &mut [u16],
) {
    let s = cfg.mrope_section;
    for (i, &f) in inv_freq.iter().enumerate() {
        let axis = if cfg.mrope_interleaved {
            if i % 3 == 1 && i < 3 * s[1] {
                1
            } else if i % 3 == 2 && i < 3 * s[2] {
                2
            } else {
                0
            }
        } else if i < s[0] {
            0
        } else if i < s[0] + s[1] {
            1
        } else {
            2
        };
        let a = position[axis] as f32 * f;
        cos[i] = to_bf16(a.cos());
        sin[i] = to_bf16(a.sin());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> QwenVisionConfig {
        QwenVisionConfig {
            hidden_size: 1152,
            depth: 27,
            num_heads: 16,
            intermediate_size: 4304,
            patch_size: 16,
            temporal_patch_size: 2,
            merge: 2,
            out_hidden_size: 2048,
            num_position_embeddings: 2304,
            rope_theta: 10_000.0,
            min_pixels: 65_536,
            max_pixels: 16_777_216,
            mrope_section: [11, 11, 10],
            mrope_interleaved: true,
            deepstack: Vec::new(),
        }
    }

    /// `smart_resize` of transformers 5.22.1 (factor 32) on the two images of the bench and on the
    /// edges.
    #[test]
    fn smart_resize_matches_the_processor() {
        let c = cfg();
        assert_eq!(c.smart_resize(488, 640).unwrap(), (480, 640));
        assert_eq!(c.smart_resize(2499, 2939).unwrap(), (2496, 2944));
        assert_eq!(c.tokens(488, 640).unwrap(), (300, 20));
        // Under `min_pixels`: scaled up.
        assert_eq!(c.smart_resize(100, 100).unwrap(), (256, 256));
        // Half to even: 2512 / 32 = 78.5 → 78, 2960 / 32 = 92.5 → 92.
        assert_eq!(c.smart_resize(2512, 2960).unwrap(), (2496, 2944));
    }
}
