//! The vision tower of Qwen3.5 / Qwen3.6 (`Qwen3_5VisionModel`, `Qwen3_5MoeVisionModel`) and of
//! Qwen3-VL (`Qwen3VLMoeVisionModel`, with its deepstack), lowered into generated programs: the
//! patches of one image through the tower's layers, then the merger that turns each `merge × merge`
//! group of patches into one soft token. Qwen3-VL's deepstack cuts the layers into segments: after
//! layer `deepstack_visual_indexes[i]` (8, 16, 24), a merger of its own
//! (`deepstack_merger_list[i]`, `use_postshuffle_norm`: one LayerNorm over the `merge²` patches of
//! a soft token read as one row) turns the hidden state of that layer into the deepstack vectors
//! `i` (`[soft][out_hidden]`), which the language model adds to the image's rows after its layer
//! `i`. Every computation is an emitted kernel of this crate (the bf16 GEMM, `ElemOp::LayerNorm`,
//! `ElemOp::Rope`, the K/V append and the attention of the text, `ElemOp::GeluTanh` and
//! `ElemOp::GeluErf`, `ElemOp::ResidualAdd`); the host lays out the patches, their learned
//! positions (interpolated) and their RoPE tables (`engine/vision_qwen.rs`).
//!
//! The patches arrive in **merge-block order**, as the processor writes them: the `merge × merge`
//! patches of one soft token are consecutive rows (raster inside the block), the blocks in raster
//! order. The attention has no mask (every patch of the image sees every other, `whole_block`), the
//! merger reads `merge²` consecutive rows as one.
//!
//! A layer, as transformers 5.22.1's `Qwen3_5MoeVisionBlock`:
//! `h += proj(attn(rope(q(LN₁(h))), rope(k(LN₁(h))), v(LN₁(h))))`, then `h +=
//! fc2(geluTanh(fc1(LN₂(h))))`; every linear has a bias, the LayerNorms a weight and a bias (ε
//! 1e-6), the attention scales by `head_dim^-½`. The RoPE is one `rotate_half` over the whole head,
//! its table row `[h · f₀ … h · f_{n−1} | w · f₀ … w · f_{n−1}]` (axial, the frequencies of a
//! quarter head). The merger: `LN(h)` per patch, `merge²` rows read as one, `fc2(gelu(fc1(·)))`
//! with the exact (erf) GELU.
//!
//! **The biases ride the GEMM.** Every input of a linear is a buffer whose rows hold, past the `k`
//! values, a column of ones (then zeros to a multiple of 32 values, [`padded`]); the weight holds
//! the bias in that column. The f32 accumulator of the GEMM then sums `Σ x·w + 1·b` and rounds
//! once, as the bf16 `nn.Linear` (and the `Conv3d` of the patch embedding) does. The kernels that
//! write those inputs write `k` values per row and leave the column alone.

use super::lowering::{binding, Buffer};
use super::vision::{elem, gemm_labeled, rows_node, tower_sdpa};
use crate::core::avec::AVec;
use crate::jit::arch::{KernelSet, Native};
use crate::jit::emit_elem::ElemOp;
use crate::jit::execution::{
    self as ex,
    abi::TaskGrid,
    build::{Builder, Mounted},
    ExecutionError,
};
use crate::jit::ir::SdpaShape;
use crate::jit::GemmShape;

fn err(e: impl std::fmt::Display) -> ExecutionError {
    ExecutionError(e.to_string())
}

/// The width of a linear's input with its column of ones: `k + 1` rounded up to 32 values (rows
/// 64-byte aligned).
pub fn padded(k: usize) -> usize {
    (k + 1).next_multiple_of(32)
}

/// The geometry of the tower, for `max_rows` patches.
#[derive(Clone, Copy, Debug)]
pub struct QwenTowerShape {
    /// The tower's width.
    pub hidden: usize,
    /// Its attention heads.
    pub heads: usize,
    /// Elements per head.
    pub head_dim: usize,
    /// The MLP's width.
    pub intermediate: usize,
    /// Values of one patch, `3 · temporal · patch²` (1 536).
    pub pixels: usize,
    /// The side of a merge block, in patches.
    pub merge: usize,
    /// The language model's width.
    pub out_hidden: usize,
    /// Patches of one image at most: the rows of the layers' program (a multiple of `merge²`).
    pub max_rows: usize,
    /// Qwen3-VL: the deepstack mergers (3); 0 for Qwen3.5 / Qwen3.6.
    pub deepstack: usize,
}

impl QwenTowerShape {
    /// Soft tokens of one image at most.
    pub fn max_soft(&self) -> usize {
        self.max_rows / (self.merge * self.merge)
    }
    /// The merger's width, `merge² · hidden`.
    pub fn merged(&self) -> usize {
        self.merge * self.merge * self.hidden
    }
}

/// One layer: the LayerNorms (`g` then `b`) and the linears, each packed with its bias in the
/// column `k` of a [`padded`] input (`q`, `k`, `v`, `proj`, `fc1`, `fc2`).
pub struct QwenLayerWeights<'a> {
    /// `norm1`, before the attention.
    pub norm1: &'a [u16],
    /// `norm2`, before the MLP.
    pub norm2: &'a [u16],
    /// `q`, `k`, `v`, `proj`, `fc1`, `fc2`.
    pub linears: [&'a [u16]; 6],
}

/// The tower's weights.
pub struct QwenTowerWeights<'a> {
    /// The patch embedding (`Conv3d` as a linear), `hidden × padded(pixels)`, packed, its bias in
    /// column `pixels`.
    pub patch: &'a [u16],
    /// The blocks.
    pub layers: Vec<QwenLayerWeights<'a>>,
    /// The merger's LayerNorm, `g` then `b` (`hidden` each).
    pub merger_norm: &'a [u16],
    /// `fc1`, `merged × padded(merged)`, and `fc2`, `out_hidden × padded(merged)`, packed with
    /// their biases.
    pub merger_fc1: &'a [u16],
    /// See `merger_fc1`.
    pub merger_fc2: &'a [u16],
    /// Qwen3-VL: the deepstack mergers, in the order of their layers.
    pub deepstack: Vec<QwenDeepstackWeights<'a>>,
}

/// One deepstack merger of Qwen3-VL (`Qwen3VLMoeVisionPatchMerger` with `use_postshuffle_norm`).
pub struct QwenDeepstackWeights<'a> {
    /// The tower's layer after which it reads the hidden state (`deepstack_visual_indexes[i]`).
    pub after: usize,
    /// Its LayerNorm over the `merged` values of a soft token, `g` then `b`.
    pub norm: &'a [u16],
    /// `fc1`, `merged × padded(merged)`, and `fc2`, `out_hidden × padded(merged)`, packed with
    /// their biases.
    pub fc1: &'a [u16],
    /// See `fc1`.
    pub fc2: &'a [u16],
}

/// The tower's buffers for `max_rows` patches. The host writes `patches` (the column of ones is in
/// place), `pos`, `cos` and `sin` before a call; `out` holds the soft tokens after it.
pub struct QwenTowerBuffers {
    /// `[rows][padded(pixels)]`, the patches in merge-block order, each `(c, t, y, x)`,
    /// normalized, bf16.
    pub patches: AVec<u16>,
    /// `[rows][hidden]`: the learned positions of each patch, interpolated, bf16.
    pub pos: AVec<u16>,
    /// `[rows][head_dim / 2]`: the cosines of the `h` angles, then those of the `w` angles.
    pub cos: AVec<u16>,
    /// The sines, the same way.
    pub sin: AVec<u16>,
    /// The soft tokens, `[max_soft][out_hidden]`.
    pub out: AVec<u16>,
    /// Qwen3-VL: the deepstack vectors, `[deepstack][max_soft][out_hidden]`.
    pub deep: AVec<u16>,
    h: AVec<u16>,
    xn: AVec<u16>,
    q: AVec<u16>,
    k: AVec<u16>,
    v: AVec<u16>,
    attn: AVec<u16>,
    o: AVec<u16>,
    f1: AVec<u16>,
    g: AVec<u16>,
    keys: AVec<u16>,
    values: AVec<u16>,
    keys_r16: AVec<u16>,
    values_r16: AVec<u16>,
    scores: AVec<f32>,
    packed: AVec<u16>,
    mln: AVec<u16>,
    m1: AVec<u16>,
    mg: AVec<u16>,
    ones: AVec<u16>,
    scores_stride: usize,
}

/// `rows` rows of `width` zeros with a 1 at column `k` (the bias column).
fn with_ones(rows: usize, width: usize, k: usize) -> AVec<u16> {
    let mut v = AVec::zeroed(rows * width);
    for r in 0..rows {
        v[r * width + k] = 0x3f80;
    }
    v
}

impl QwenTowerBuffers {
    /// Everything for `s` and `workers` participants, the columns of ones in place.
    pub fn new(s: &QwenTowerShape, workers: usize) -> Self {
        let r = s.max_rows;
        let soft = s.max_soft();
        let (h, qd, inter, m) = (s.hidden, s.heads * s.head_dim, s.intermediate, s.merged());
        let packed = [
            (s.pixels, h, r),
            (h, qd, r),
            (qd, h, r),
            (h, inter, r),
            (inter, h, r),
            (m, m, soft),
            (m, s.out_hidden, soft),
        ]
        .iter()
        .map(|&(k, n, rows)| {
            let k = padded(k);
            let plan = crate::jit::arch::gemm_plan_bf16(GemmShape { k, n }, rows);
            if plan.opts.matrix {
                crate::jit::emit::packed_bytes(plan.opts, rows, k).div_ceil(2)
            } else {
                0
            }
        })
        .max()
        .unwrap_or(0);
        let scores_stride = crate::jit::sdpa::full_scratch_f32(r, s.head_dim);
        let (cap, r16) = crate::jit::sdpa::full_kv_capacity(r);
        Self {
            patches: with_ones(r, padded(s.pixels), s.pixels),
            pos: AVec::zeroed(r * h),
            cos: AVec::zeroed(r * s.head_dim / 2),
            sin: AVec::zeroed(r * s.head_dim / 2),
            out: AVec::zeroed(soft * s.out_hidden),
            deep: AVec::zeroed(s.deepstack * soft * s.out_hidden),
            h: AVec::zeroed(r * h),
            xn: with_ones(r, padded(h), h),
            q: AVec::zeroed(r * qd),
            k: AVec::zeroed(r * qd),
            v: AVec::zeroed(r * qd),
            attn: with_ones(r, padded(qd), qd),
            o: AVec::zeroed(r * h),
            f1: AVec::zeroed(r * inter),
            g: with_ones(r, padded(inter), inter),
            keys: AVec::zeroed(cap * qd),
            values: AVec::zeroed(cap * qd),
            keys_r16: AVec::zeroed(if r16 { cap * qd } else { 0 }),
            values_r16: AVec::zeroed(if r16 { cap * qd } else { 0 }),
            scores: AVec::zeroed(scores_stride * workers.max(1)),
            packed: AVec::zeroed(packed.max(1)),
            mln: with_ones(soft, padded(m), m),
            m1: AVec::zeroed(soft * m),
            mg: with_ones(soft, padded(m), m),
            ones: AVec::from_vec(vec![0x3f80; inter.max(m)]),
            scores_stride,
        }
    }

    /// Bytes reserved.
    pub fn bytes(&self) -> usize {
        let u16s = [
            &self.patches,
            &self.pos,
            &self.cos,
            &self.sin,
            &self.out,
            &self.deep,
            &self.h,
            &self.xn,
            &self.q,
            &self.k,
            &self.v,
            &self.attn,
            &self.o,
            &self.f1,
            &self.g,
            &self.keys,
            &self.values,
            &self.keys_r16,
            &self.values_r16,
            &self.packed,
            &self.mln,
            &self.m1,
            &self.mg,
            &self.ones,
        ];
        u16s.iter().map(|b| b.len() * 2).sum::<usize>() + self.scores.len() * 4
    }
}

/// The programs: the layers over the patches, in segments cut after each deepstack layer (one
/// segment without deepstack), the deepstack mergers (run after the segment `i`), the merger over
/// the soft tokens.
pub struct QwenTowerPrograms {
    pub(crate) layers: Vec<Mounted>,
    pub(crate) deepstack: Vec<Mounted>,
    pub(crate) merger: Mounted,
}

/// A merger over the soft tokens from the tower's hidden state `hb` into `out`: the LayerNorm `ln`
/// of each patch (`per_patch`, the final merger) or of the `merge²` patches of a soft token as one
/// row (`use_postshuffle_norm`, the deepstack's), then `fc2(geluErf(fc1(·)))`.
#[allow(clippy::too_many_arguments)]
fn lower_merger(
    s: &QwenTowerShape,
    b: &mut QwenTowerBuffers,
    hb: *mut u16,
    norm: &[u16],
    per_patch: bool,
    fc1: &[u16],
    fc2: &[u16],
    out: *mut u16,
    workers: usize,
    spin: Option<u32>,
) -> std::result::Result<Mounted, ExecutionError> {
    let (h, m, mm) = (s.hidden, s.merged(), s.merge * s.merge);
    let eps = 1e-6f32;
    let mut builder = Builder::new(s.max_soft())?;
    let p = |v: &mut AVec<u16>| v.as_mut_ptr();
    let (mln, m1, mg) = (p(&mut b.mln), p(&mut b.m1), p(&mut b.mg));
    let packed = Buffer {
        ptr: b.packed.as_mut_ptr(),
        len: b.packed.len(),
    };
    let mp2 = padded(m) * 2;
    if per_patch {
        let ln_h = elem(ElemOp::LayerNorm, h)?;
        // The LayerNorm of each patch, patch `j` of a block into columns `j·hidden..` of its soft
        // token's row.
        for j in 0..mm {
            rows_node(
                &mut builder,
                "vision-merger-norm",
                &ln_h,
                &[
                    (unsafe { hb.add(j * h) } as u64, m * 2),
                    (norm.as_ptr() as u64, 0),
                    (unsafe { mln.add(j * h) } as u64, mp2),
                ],
                &[eps],
                workers,
            )?;
        }
    } else {
        let ln_m = elem(ElemOp::LayerNorm, m)?;
        rows_node(
            &mut builder,
            "vision-deepstack-norm",
            &ln_m,
            &[
                (hb as u64, m * 2),
                (norm.as_ptr() as u64, 0),
                (mln as u64, mp2),
            ],
            &[eps],
            workers,
        )?;
    }
    let gelu_erf = elem(ElemOp::GeluErf, m)?;
    linear(
        &mut builder,
        "vision-merger-fc1",
        mln,
        fc1,
        m1,
        m,
        m,
        packed,
        workers,
    )?;
    rows_node(
        &mut builder,
        "vision-merger-gelu",
        &gelu_erf,
        &[
            (m1 as u64, m * 2),
            (b.ones.as_ptr() as u64, 0),
            (mg as u64, mp2),
        ],
        &[],
        workers,
    )?;
    linear(
        &mut builder,
        "vision-merger-fc2",
        mg,
        fc2,
        out,
        m,
        s.out_hidden,
        packed,
        workers,
    )?;
    builder.finish_with_instrumentation(1, spin, false, false, false, false)
}

/// A linear from `x` (rows of [`padded`]`(k)`, the ones in place) into `y` (rows of `n`).
#[allow(clippy::too_many_arguments)]
fn linear(
    builder: &mut Builder,
    label: &'static str,
    x: *mut u16,
    w: &[u16],
    y: *mut u16,
    k: usize,
    n: usize,
    packed: Buffer,
    workers: usize,
) -> std::result::Result<(), ExecutionError> {
    gemm_labeled(builder, label, x, w, y, padded(k), n, packed, workers)
}

/// The tower's two programs over `b`, for `workers` participants. The weights and the buffers must
/// outlive them.
pub(crate) fn lower_qwen_tower(
    s: &QwenTowerShape,
    w: &QwenTowerWeights<'_>,
    b: &mut QwenTowerBuffers,
    workers: usize,
    spin: Option<u32>,
    trace: bool,
) -> std::result::Result<QwenTowerPrograms, ExecutionError> {
    let (h, hd, nh, inter) = (s.hidden, s.head_dim, s.heads, s.intermediate);
    let qd = nh * hd;
    let m = s.merged();
    let rows = s.max_rows;
    let mm = s.merge * s.merge;
    let deep_ok = w.deepstack.len() == s.deepstack
        && w.deepstack.windows(2).all(|d| d[0].after < d[1].after)
        && w.deepstack
            .iter()
            .all(|d| d.after < w.layers.len() && d.norm.len() >= 2 * m)
        && b.deep.len() >= s.deepstack * s.max_soft() * s.out_hidden;
    if !deep_ok
        || hd % 4 != 0
        || mm == 0
        || !rows.is_multiple_of(mm)
        || w.merger_norm.len() < 2 * h
        || w.layers
            .iter()
            .any(|l| l.norm1.len() < 2 * h || l.norm2.len() < 2 * h)
    {
        return Err(err(
            "vision: a Qwen tower geometry this engine does not serve",
        ));
    }
    let (cap, r16) = crate::jit::sdpa::full_kv_capacity(rows);
    let ln_h = elem(ElemOp::LayerNorm, h)?;
    let rope = elem(ElemOp::Rope, hd)?;
    let add_h = elem(ElemOp::ResidualAdd, h)?;
    let gelu = elem(ElemOp::GeluTanh, inter)?;
    let sdpa = tower_sdpa(hd)?;
    // `HERBERT_VISION_ATTN_ROWS=N`: queries per block of the attention (0: the row kernel alone).
    let block_rows = std::env::var("HERBERT_VISION_ATTN_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(crate::jit::emit_sdpa::SDPA_PREFILL_MAX_ROWS);
    let block = crate::jit::sdpa::prefill_kernel_full(
        SdpaShape {
            head_dim: hd,
            sink: false,
        },
        block_rows,
        qd,
        padded(qd),
        cap,
        sdpa.spec().qk_group,
        0,
    )
    .map_err(err)?;
    let chunk = block.as_ref().map_or(1, |k| k.rows());
    let r16 = r16 && block.is_some();
    if b.scores_stride < block.as_ref().map_or(0, |k| k.scratch_len())
        || b.keys.len() < cap * qd
        || (r16 && b.keys_r16.len() < cap * qd)
    {
        return Err(err(
            "vision: the attention's scratch is larger than its slot",
        ));
    }
    if std::env::var_os("HERBERT_VISION_PLAN").is_some_and(|v| v != "0") {
        match block.as_deref() {
            Some(p) => eprintln!("[vision plan] Qwen tower, head_dim {hd}, {rows} patches: block of {} rows, scratch {} f32", p.rows(), p.scratch_len()),
            None => eprintln!("[vision plan] Qwen tower, head_dim {hd}, {rows} patches: row kernel"),
        }
    }
    let p = |v: &mut AVec<u16>| v.as_mut_ptr();
    let (hb, xn, q, k, v, attn, o, f1, g) = (
        p(&mut b.h),
        p(&mut b.xn),
        p(&mut b.q),
        p(&mut b.k),
        p(&mut b.v),
        p(&mut b.attn),
        p(&mut b.o),
        p(&mut b.f1),
        p(&mut b.g),
    );
    let packed = Buffer {
        ptr: b.packed.as_mut_ptr(),
        len: b.packed.len(),
    };
    let (h2, qd2, hp2, qp2, ip2) = (
        h * 2,
        qd * 2,
        padded(h) * 2,
        padded(qd) * 2,
        padded(inter) * 2,
    );
    let eps = 1e-6f32;
    let half = hd / 2;

    // ── The patches through the layers ──────────────────────────────────────
    let mut segments = Vec::with_capacity(w.deepstack.len() + 1);
    let mut deepstack = Vec::with_capacity(w.deepstack.len());
    let mut builder = Builder::new(rows)?;
    linear(
        &mut builder,
        "vision-patch-embed",
        b.patches.as_mut_ptr(),
        w.patch,
        hb,
        s.pixels,
        h,
        packed,
        workers,
    )?;
    rows_node(
        &mut builder,
        "vision-position-add",
        &add_h,
        &[(hb as u64, h2), (b.pos.as_ptr() as u64, h2)],
        &[],
        workers,
    )?;
    let kv_spec = ex::abi::KvSpec {
        heads: nh,
        head_dim: hd,
        stride: cap * hd,
        capacity: cap,
        max_batch: rows,
        r16,
    };
    for (l, lw) in w.layers.iter().enumerate() {
        builder.begin_layer(l);
        rows_node(
            &mut builder,
            "vision-norm",
            &ln_h,
            &[
                (hb as u64, h2),
                (lw.norm1.as_ptr() as u64, 0),
                (xn as u64, hp2),
            ],
            &[eps],
            workers,
        )?;
        for (i, dst) in [(0usize, q), (1, k), (2, v)] {
            linear(
                &mut builder,
                "vision-qkv",
                xn,
                lw.linears[i],
                dst,
                h,
                qd,
                packed,
                workers,
            )?;
        }
        // Q and K: one `rotate_half` over the whole head, the table row of the patch.
        for data in [q, k] {
            builder.label("vision-rope");
            builder.rows(
                Native::norm_rope(None, &rope, nh, hd)?,
                binding(
                    &[
                        (data as u64, qd2),
                        (b.ones.as_ptr() as u64, 0),
                        (b.cos.as_ptr() as u64, half * 2),
                        (b.sin.as_ptr() as u64, half * 2),
                    ],
                    &[eps],
                ),
                workers,
                workers,
            )?;
        }
        let kv_call = builder.keep(ex::abi::KvCall {
            keys: b.keys.as_mut_ptr(),
            values: b.values.as_mut_ptr(),
            input_keys: k,
            input_values: v,
            rows: &raw const builder.frame.rows,
            past: &raw const builder.frame.position,
            keys_r16: if r16 {
                b.keys_r16.as_mut_ptr()
            } else {
                std::ptr::null_mut()
            },
            values_r16: if r16 {
                b.values_r16.as_mut_ptr()
            } else {
                std::ptr::null_mut()
            },
        });
        let mut append = binding(&[(kv_call as u64, 0), (0, 1)], &[]);
        append.tasks = nh;
        builder.label("vision-kv-append");
        builder.grid(
            Native::kv_append(kv_spec)?,
            append,
            TaskGrid {
                rows_per_task: rows,
                tasks_per_tile: nh,
                last_block_only: false,
            },
            workers,
            workers,
        )?;
        let call = builder.keep(ex::abi::AttentionCall {
            q,
            keys: b.keys.as_ptr(),
            values: b.values.as_ptr(),
            output: attn,
            rows: &raw const builder.frame.rows,
            past: &raw const builder.frame.position,
            kv_stride: cap * hd,
            scale: 1.0 / (hd as f32).sqrt(),
            sink: 0.0,
            keys_r16: if r16 {
                b.keys_r16.as_ptr()
            } else {
                std::ptr::null()
            },
            values_r16: if r16 {
                b.values_r16.as_ptr()
            } else {
                std::ptr::null()
            },
            segments: std::ptr::null(),
        });
        let mut args = binding(
            &[(call as u64, 0), (0, 1), (b.scores.as_mut_ptr() as u64, 0)],
            &[],
        );
        args.participant_strides[2] = (b.scores_stride * 4) as u64;
        if b.scores.len() < b.scores_stride * workers {
            return Err(err("vision: attention score slots too short"));
        }
        builder.label("vision-attention");
        builder.grid(
            Native::attention_tile_selector(
                &sdpa,
                block.as_deref(),
                ex::abi::AttentionSpec {
                    heads: nh,
                    gqa: 1,
                    head_dim: hd,
                    q_row: qd,
                    q_head: hd,
                    out_row: padded(qd),
                    grouped: false,
                    kv_task: false,
                    window: 0,
                    whole_block: true,
                    ahead: 0,
                    segments: false,
                },
            )?,
            args,
            TaskGrid {
                rows_per_task: chunk,
                tasks_per_tile: nh,
                last_block_only: false,
            },
            workers,
            workers,
        )?;
        // The tasks dealt cyclically, as in Gemma's tower: every worker on the same head at once,
        // its keys and values shared in L3. `HERBERT_VISION_ATTN_CYCLIC=0`: the contiguous
        // ranges. The same bits either way.
        if std::env::var("HERBERT_VISION_ATTN_CYCLIC").map_or(true, |v| v != "0") {
            builder.cyclic_last();
        }
        let _ = qp2;
        linear(
            &mut builder,
            "vision-proj",
            attn,
            lw.linears[3],
            o,
            qd,
            h,
            packed,
            workers,
        )?;
        rows_node(
            &mut builder,
            "vision-residual",
            &add_h,
            &[(hb as u64, h2), (o as u64, h2)],
            &[],
            workers,
        )?;
        rows_node(
            &mut builder,
            "vision-norm",
            &ln_h,
            &[
                (hb as u64, h2),
                (lw.norm2.as_ptr() as u64, 0),
                (xn as u64, hp2),
            ],
            &[eps],
            workers,
        )?;
        linear(
            &mut builder,
            "vision-fc1",
            xn,
            lw.linears[4],
            f1,
            h,
            inter,
            packed,
            workers,
        )?;
        // `geluTanh(x) · 1`: the activation alone, rounded once.
        rows_node(
            &mut builder,
            "vision-gelu",
            &gelu,
            &[
                (f1 as u64, inter * 2),
                (b.ones.as_ptr() as u64, 0),
                (g as u64, ip2),
            ],
            &[],
            workers,
        )?;
        linear(
            &mut builder,
            "vision-fc2",
            g,
            lw.linears[5],
            o,
            inter,
            h,
            packed,
            workers,
        )?;
        rows_node(
            &mut builder,
            "vision-residual",
            &add_h,
            &[(hb as u64, h2), (o as u64, h2)],
            &[],
            workers,
        )?;
        builder.layer_end(l);
        // Qwen3-VL: the segment ends after a deepstack layer; its merger reads the hidden state
        // before the next segment changes it.
        if let Some(d) = w.deepstack.get(deepstack.len()).filter(|d| d.after == l) {
            let next = Builder::new(rows)?;
            segments.push(
                std::mem::replace(&mut builder, next)
                    .finish_with_instrumentation(1, spin, trace, false, false, false)?,
            );
            let i = deepstack.len();
            let out = unsafe { b.deep.as_mut_ptr().add(i * s.max_soft() * s.out_hidden) };
            deepstack.push(lower_merger(
                s, b, hb, d.norm, false, d.fc1, d.fc2, out, workers, spin,
            )?);
        }
    }
    if w.deepstack
        .last()
        .is_none_or(|d| d.after + 1 < w.layers.len())
    {
        segments.push(builder.finish_with_instrumentation(1, spin, trace, false, false, false)?);
    }

    // ── The merger ──────────────────────────────────────────────────────────
    let out = b.out.as_mut_ptr();
    let merger = lower_merger(
        s,
        b,
        hb,
        w.merger_norm,
        true,
        w.merger_fc1,
        w.merger_fc2,
        out,
        workers,
        spin,
    )?;
    Ok(QwenTowerPrograms {
        layers: segments,
        deepstack,
        merger,
    })
}
