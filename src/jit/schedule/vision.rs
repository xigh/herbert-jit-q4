//! The Gemma 4 vision tower (E2B, E4B) lowered into two generated programs:
//! the patches of one image through the tower's layers, then their pooling
//! and the projection into the language model's space. Every computation is
//! an emitted kernel of this crate — the bf16 GEMM, the row norms, the head
//! norms and the RoPE, the K/V append and the row attention of the text, and
//! two elementwise kernels of their own (`ElemOp::Clamp`, the bounds of the
//! clipped linears; `ElemOp::PoolRows`, the pooling); the host only lays out
//! the patches, their positions and their RoPE tables (`engine/vision.rs`).
//!
//! The patches arrive **cell by cell**: the `k × k` patches of one soft token
//! are consecutive rows (raster order inside the cell), the cells in raster
//! order. The tower sees every patch through per-row operations and an
//! attention without mask (every patch sees every other, `whole_block`), so
//! the order of its rows changes nothing but the order of the attention's
//! sums; the pooling is then the mean of `k²` consecutive rows.
//!
//! A layer, as the reference:
//! `h += norm(o(attn(rope(norm(q(norm(h)))), rope(norm(k(…))), norm(v(…)))))`,
//! then `h += norm(down(gelu(gate(norm(h))) · up(norm(h))))`; a clipped linear
//! clamps its input (into a copy) and its output (in place). The positions
//! are `h += bf16(table[0][x] + table[1][y])` after the patch embedding; the
//! RoPE is 2-D: the first half of a head rotates by `x`, the second by `y`,
//! each as `rotate_half`. The attention scales by one (Q and K normed).
//!
//! 12B has no tower but an encoder-free embedder (`lower_unified`): one
//! program over the soft tokens, LayerNorms, a dense with bias, factorized
//! positions, then the same embedder.

use super::lowering::{binding, packed_matrix, Buffer, Pack};
use crate::jit::arch::{KernelSet, Native};
use crate::jit::compile::CompiledGemm;
use crate::jit::elem::CompiledElem;
use crate::jit::emit_elem::{ElemOp, ElemSpec};
use crate::jit::execution::{
    self as ex,
    abi::TaskGrid,
    build::{Builder, Mounted},
    ExecutionError,
};
use crate::jit::ir::SdpaShape;
use crate::jit::GemmShape;
use std::sync::Arc;

fn err(e: impl std::fmt::Display) -> ExecutionError {
    ExecutionError(e.to_string())
}

/// The geometry of a tower and of the soft tokens it makes.
#[derive(Clone, Copy, Debug)]
pub struct TowerShape {
    /// The tower's width.
    pub hidden: usize,
    /// Its attention heads (as many K/V heads).
    pub heads: usize,
    /// Elements per head.
    pub head_dim: usize,
    /// The MLP's width.
    pub intermediate: usize,
    /// Values of one patch, `3 · patch²`.
    pub pixels: usize,
    /// Positions per axis of the position table.
    pub positions: usize,
    /// The RMS norms' epsilon.
    pub eps: f32,
    /// The pooling window's side, in patches.
    pub pool: usize,
    /// Soft tokens of one image at most.
    pub max_soft: usize,
    /// The language model's width (the soft tokens' width).
    pub out_hidden: usize,
    /// The pooled soft tokens standardized (`standardize`, 26B-A4B): `(x · √hidden − std_bias) ·
    /// std_scale` in f32.
    pub standardize: bool,
}

impl TowerShape {
    /// Patches of one image at most: the rows of the tower's program.
    pub fn max_rows(&self) -> usize {
        self.max_soft * self.pool * self.pool
    }
    fn q_dim(&self) -> usize {
        self.heads * self.head_dim
    }
}

/// A linear of the tower: its R16-packed bf16 weight (`n` rows of `k`), and for a clipped one its
/// `[input_min, input_max, output_min, output_max]`.
#[derive(Clone, Copy)]
pub struct TowerLinear<'a> {
    /// The packed weight ([`crate::model::PackedBf16R16Weight::pack_bf16`]).
    pub weights: &'a [u16],
    /// Output features.
    pub n: usize,
    /// Input features.
    pub k: usize,
    /// The bounds of a clipped linear.
    pub clip: Option<[f32; 4]>,
}

/// One layer of the tower.
pub struct TowerLayerWeights<'a> {
    /// The norm before the attention.
    pub input_norm: &'a [u16],
    /// The norm of the attention's output.
    pub post_attn_norm: &'a [u16],
    /// The norm before the MLP.
    pub pre_ff_norm: &'a [u16],
    /// The norm of the MLP's output.
    pub post_ff_norm: &'a [u16],
    /// The per-head norm of the queries, `head_dim`.
    pub q_norm: &'a [u16],
    /// The per-head norm of the keys.
    pub k_norm: &'a [u16],
    /// `q, k, v, o, gate, up, down`.
    pub linears: [TowerLinear<'a>; 7],
}

/// The tower's weights.
pub struct TowerWeights<'a> {
    /// The patch embedding, `hidden × pixels`, packed.
    pub input_proj: &'a [u16],
    /// The position table, `[2][positions][hidden]` row-major.
    pub position_table: &'a [u16],
    /// The projection into the language model, `out_hidden × hidden`, packed.
    pub embed_proj: &'a [u16],
    /// With `standardize`: `std_bias` then `std_scale`, `2 · hidden`.
    pub standardize: Option<&'a [u16]>,
    /// The layers.
    pub layers: Vec<TowerLayerWeights<'a>>,
}

/// The tower's buffers, allocated once for the largest image. The host writes `patches`, `pos_x`,
/// `pos_y`, `cos` and `sin` before a call; `out` holds the soft tokens after it.
pub struct TowerBuffers {
    /// `[rows][pixels]`, the patches cell by cell, `2·(p − ½)` in bf16.
    pub patches: crate::core::avec::AVec<u16>,
    /// The column and the row of each patch in the grid, u32.
    pub pos_x: crate::core::avec::AVec<u32>,
    /// The row of each patch.
    pub pos_y: crate::core::avec::AVec<u32>,
    /// `[rows][head_dim / 2]`: the cosines of the `x` angles, then those of the `y` angles.
    pub cos: crate::core::avec::AVec<u16>,
    /// The sines, the same way.
    pub sin: crate::core::avec::AVec<u16>,
    /// The soft tokens, `[max_soft][out_hidden]`: the table the image regime reads.
    pub out: crate::core::avec::AVec<u16>,
    h: crate::core::avec::AVec<u16>,
    xn: crate::core::avec::AVec<u16>,
    xc: crate::core::avec::AVec<u16>,
    q: crate::core::avec::AVec<u16>,
    k: crate::core::avec::AVec<u16>,
    v: crate::core::avec::AVec<u16>,
    attn: crate::core::avec::AVec<u16>,
    o: crate::core::avec::AVec<u16>,
    gate: crate::core::avec::AVec<u16>,
    up: crate::core::avec::AVec<u16>,
    keys: crate::core::avec::AVec<u16>,
    values: crate::core::avec::AVec<u16>,
    /// The R16 copies of `keys` and `values` (the AArch64 block attention), empty without it.
    keys_r16: crate::core::avec::AVec<u16>,
    values_r16: crate::core::avec::AVec<u16>,
    scores: crate::core::avec::AVec<f32>,
    packed: crate::core::avec::AVec<u16>,
    pooled: crate::core::avec::AVec<u16>,
    pooled_n: crate::core::avec::AVec<u16>,
    ones: crate::core::avec::AVec<u16>,
    scores_stride: usize,
}

impl TowerBuffers {
    /// Everything, zeroed, for `shape` and `workers` participants.
    pub fn new(shape: &TowerShape, workers: usize) -> Self {
        use crate::core::avec::AVec;
        let r = shape.max_rows();
        let (h, qd, inter) = (shape.hidden, shape.q_dim(), shape.intermediate);
        let widest = h.max(qd).max(inter).max(shape.pixels);
        // The packed activations of the matrix engines (SME, AMX), for the widest input at the most
        // rows.
        let packed = [shape.pixels, h, qd, inter]
            .iter()
            .map(|&k| {
                let plan = crate::jit::arch::gemm_plan_bf16(GemmShape { k, n: h }, r);
                if plan.opts.matrix {
                    crate::jit::emit::packed_bytes(plan.opts, r, k).div_ceil(2)
                } else {
                    0
                }
            })
            .max()
            .unwrap_or(0);
        // The row kernel's scores, or a block's (`SDPA_PREFILL_MAX_ROWS` rows of them).
        let scores_stride = crate::jit::sdpa::full_scratch_f32(r, shape.head_dim);
        let (cap, r16) = crate::jit::sdpa::full_kv_capacity(r);
        Self {
            patches: AVec::zeroed(r * shape.pixels),
            pos_x: AVec::zeroed(r),
            pos_y: AVec::zeroed(r),
            cos: AVec::zeroed(r * shape.head_dim / 2),
            sin: AVec::zeroed(r * shape.head_dim / 2),
            out: AVec::zeroed(shape.max_soft * shape.out_hidden),
            h: AVec::zeroed(r * h),
            xn: AVec::zeroed(r * h),
            xc: AVec::zeroed(r * widest),
            q: AVec::zeroed(r * qd),
            k: AVec::zeroed(r * qd),
            v: AVec::zeroed(r * qd),
            attn: AVec::zeroed(r * qd),
            o: AVec::zeroed(r * h),
            gate: AVec::zeroed(r * inter),
            up: AVec::zeroed(r * inter),
            keys: AVec::zeroed(cap * qd),
            values: AVec::zeroed(cap * qd),
            keys_r16: AVec::zeroed(if r16 { cap * qd } else { 0 }),
            values_r16: AVec::zeroed(if r16 { cap * qd } else { 0 }),
            scores: AVec::zeroed(scores_stride * workers.max(1)),
            packed: AVec::zeroed(packed),
            pooled: AVec::zeroed(shape.max_soft * h),
            pooled_n: AVec::zeroed(shape.max_soft * h),
            ones: AVec::from_vec(vec![0x3f80; h.max(shape.head_dim)]),
            scores_stride,
        }
    }

    /// Bytes reserved.
    pub fn bytes(&self) -> usize {
        let u16s = [
            &self.patches,
            &self.cos,
            &self.sin,
            &self.out,
            &self.h,
            &self.xn,
            &self.xc,
            &self.q,
            &self.k,
            &self.v,
            &self.attn,
            &self.o,
            &self.gate,
            &self.up,
            &self.keys,
            &self.values,
            &self.packed,
            &self.pooled,
            &self.pooled_n,
            &self.ones,
        ];
        u16s.iter().map(|b| b.len() * 2).sum::<usize>()
            + (self.pos_x.len() + self.pos_y.len() + self.scores.len()) * 4
    }
}

/// The two programs of the tower: the patches through the layers (`rows` = the patches), then the
/// pooling and the projection (`rows` = the soft tokens).
pub struct TowerPrograms {
    /// The patch embedding, the positions, the layers.
    pub(crate) layers: Mounted,
    /// The pooling, the norm without weight, the projection into `TowerBuffers::out`.
    pub(crate) embed: Mounted,
}

/// The kernels the tower compiles beyond the GEMMs.
struct Kernels {
    norm_h: Arc<CompiledElem>,
    norm_head: Arc<CompiledElem>,
    rope: Arc<CompiledElem>,
    add_h: Arc<CompiledElem>,
    gelu: Arc<CompiledElem>,
    clamp: Vec<(usize, Arc<CompiledElem>)>,
    pool: Arc<CompiledElem>,
    /// With `standardize`.
    standardize: Option<Arc<CompiledElem>>,
    sdpa: Arc<crate::jit::sdpa::CompiledSdpa>,
    /// The block attention without the causal mask (x86-64 with AVX-512 or AVX2, AArch64), `None`:
    /// the row kernel alone.
    block: Option<Arc<crate::jit::sdpa::CompiledSdpaPrefill>>,
}

impl Kernels {
    fn clamp(&self, dim: usize) -> std::result::Result<&CompiledElem, ExecutionError> {
        self.clamp
            .iter()
            .find(|(d, _)| *d == dim)
            .map(|(_, k)| k.as_ref())
            .ok_or_else(|| err(format!("no clamp of {dim}")))
    }
}

/// The row attention of the tower (the rows past the last whole block): the text's
/// (`sdpa::head_kernel`); on x86-64 with AVX-512, its QK by groups of 16 positions, or the form and
/// the group `HERBERT_VISION_SDPA_FORM` (`fused`, `chained`, `split`) and `HERBERT_VISION_SDPA_QK`
/// (0, 8, 16) ask. Row by row on a Ryzen 9 9900X (E2B, 2 376 patches, 16 layers): chained 205 ms
/// with groups of 16, 212 ms with 8, 275 ms without, fused 360 ms.
pub(super) fn tower_sdpa(
    hd: usize,
) -> std::result::Result<Arc<crate::jit::sdpa::CompiledSdpa>, ExecutionError> {
    let shape = SdpaShape {
        head_dim: hd,
        sink: false,
    };
    let base = crate::jit::sdpa::head_kernel(shape)
        .map_err(err)?
        .ok_or_else(|| err(format!("vision: no attention of head_dim {hd}")))?;
    let form = std::env::var("HERBERT_VISION_SDPA_FORM")
        .ok()
        .map(|f| match f.as_str() {
            "fused" => crate::jit::ir::SdpaForm::Fused,
            "split" => crate::jit::ir::SdpaForm::Split,
            _ => crate::jit::ir::SdpaForm::Chained,
        });
    let qk = std::env::var("HERBERT_VISION_SDPA_QK")
        .ok()
        .and_then(|q| q.parse::<u8>().ok());
    if !cfg!(target_arch = "x86_64") || !crate::jit::gemm::is_supported() {
        return Ok(base);
    }
    let form = form.unwrap_or(base.spec().form);
    // QK by groups of 16 positions: the same scores as 8, the tower's attention 3 % faster (Ryzen 9
    // 9900X).
    let qk_group = qk.unwrap_or(if form == crate::jit::ir::SdpaForm::Fused {
        0
    } else {
        16
    });
    let spec = crate::jit::emit_sdpa::SdpaKernelSpec {
        form,
        qk_group,
        ..base.spec()
    };
    if crate::jit::options::debug_plan() {
        eprintln!("[vision] attention {form:?}, QK group {qk_group}");
    }
    crate::jit::sdpa::get_or_compile(spec)
        .map_err(err)?
        .ok_or_else(|| {
            err(format!(
                "vision: the attention {form:?} qk {qk_group} is refused"
            ))
        })
}

pub(super) fn elem(
    op: ElemOp,
    dim: usize,
) -> std::result::Result<Arc<CompiledElem>, ExecutionError> {
    crate::jit::elem::get_or_compile(ElemSpec { op, dim })
        .map_err(err)?
        .ok_or_else(|| err(format!("vision: {} dim={dim} not schedulable", op.name())))
}

/// A row-wise elementwise node on the whole team: one task per row.
pub(super) fn rows_node(
    builder: &mut Builder,
    label: &'static str,
    kernel: &CompiledElem,
    args: &[(u64, usize)],
    floats: &[f32],
    workers: usize,
) -> std::result::Result<(), ExecutionError> {
    builder.label(label);
    builder.rows(
        Native::elementary(kernel)?,
        binding(args, floats),
        workers,
        workers,
    )
}

/// `y = x · Wᵀ` over the block's rows, bf16 in and out, on the bf16 GEMM whatever `--int8` says;
/// the activations packed first when the target runs it on its matrix engine.
fn gemm(
    builder: &mut Builder,
    x: *mut u16,
    weights: &[u16],
    y: *mut u16,
    k: usize,
    n: usize,
    packed: Buffer,
    workers: usize,
) -> std::result::Result<(), ExecutionError> {
    gemm_labeled(
        builder,
        "vision-projection",
        x,
        weights,
        y,
        k,
        n,
        packed,
        workers,
    )
}

/// [`gemm`] under the node label `label`.
#[allow(clippy::too_many_arguments)]
pub(super) fn gemm_labeled(
    builder: &mut Builder,
    label: &'static str,
    x: *mut u16,
    weights: &[u16],
    y: *mut u16,
    k: usize,
    n: usize,
    packed: Buffer,
    workers: usize,
) -> std::result::Result<(), ExecutionError> {
    let shape = GemmShape { k, n };
    shape.check().map_err(err)?;
    if weights.len()
        < n.div_ceil(crate::jit::ir::ROWS_INTERLEAVED)
            * crate::jit::ir::ROWS_INTERLEAVED
            * k.next_multiple_of(2)
    {
        return Err(err(format!(
            "vision: a packed weight of {} values for {n} × {k}",
            weights.len()
        )));
    }
    let plan = crate::jit::arch::gemm_plan_bf16(shape, builder.batch);
    let activations: *const u8 = if plan.opts.matrix {
        packed_matrix(
            builder,
            Buffer {
                ptr: x,
                len: builder.batch * k,
            },
            k,
            plan,
            Pack::Into(packed),
            workers,
            true,
        )?
    } else {
        x.cast()
    };
    let kernel: Arc<CompiledGemm> = crate::jit::gemm::get_or_compile(
        shape,
        plan.tile,
        plan.opts,
        true,
        crate::jit::emit::AFormat::Bf16,
    )
    .map_err(err)?
    .ok_or_else(|| err(format!("vision: the GEMM {k} → {n} is refused")))?;
    let call = builder.keep(ex::abi::Bf16Call {
        activations,
        weights: weights.as_ptr(),
        output: y.cast(),
        rows: 0,
        row_count: &raw const builder.frame.rows,
    });
    builder.label(label);
    builder.grid(
        Native::bf16_tile_selector(&kernel)?,
        binding(&[(call as u64, 0), (0, 1)], &[]),
        TaskGrid {
            rows_per_task: kernel.tile().mr,
            tasks_per_tile: kernel.n_tile_count(),
            last_block_only: false,
        },
        workers,
        workers,
    )
}

/// A linear of the tower from `x` into `y`: a clipped one clamps its input into `xc` first (unless
/// `xc` already holds `x` clamped to the same bounds: `prev`, the bounds of the linear before on
/// the same input) and its output in place.
#[allow(clippy::too_many_arguments)]
fn linear(
    builder: &mut Builder,
    kernels: &Kernels,
    lin: &TowerLinear<'_>,
    prev: Option<&TowerLinear<'_>>,
    x: *mut u16,
    y: *mut u16,
    xc: *mut u16,
    packed: Buffer,
    workers: usize,
) -> std::result::Result<(), ExecutionError> {
    let (k, n) = (lin.k, lin.n);
    let bounds = |l: &TowerLinear<'_>| l.clip.map(|c| [c[0].to_bits(), c[1].to_bits()]);
    let input = match lin.clip {
        Some(_) if prev.is_some_and(|p| bounds(p) == bounds(lin)) => xc,
        Some([lo, hi, _, _]) => {
            rows_node(
                builder,
                "vision-clamp-input",
                kernels.clamp(k)?,
                &[(x as u64, k * 2), (xc as u64, k * 2)],
                &[lo, hi],
                workers,
            )?;
            xc
        }
        None => x,
    };
    gemm(builder, input, lin.weights, y, k, n, packed, workers)?;
    if let Some([_, _, lo, hi]) = lin.clip {
        rows_node(
            builder,
            "vision-clamp-output",
            kernels.clamp(n)?,
            &[(y as u64, n * 2), (y as u64, n * 2)],
            &[lo, hi],
            workers,
        )?;
    }
    Ok(())
}

/// The tower's two programs over `b`, for `workers` participants (`spin`: the hybrid executor's
/// spin, as the prefill's; `trace`: the layers' program stamps every node, `rdtscp`). The weights
/// and the buffers must outlive them.
pub(crate) fn lower_tower(
    shape: &TowerShape,
    w: &TowerWeights<'_>,
    b: &mut TowerBuffers,
    workers: usize,
    spin: Option<u32>,
    trace: bool,
) -> std::result::Result<TowerPrograms, ExecutionError> {
    let s = *shape;
    let (h, hd, nh, inter, qd) = (s.hidden, s.head_dim, s.heads, s.intermediate, s.q_dim());
    let rows = s.max_rows();
    let (cap, r16) = crate::jit::sdpa::full_kv_capacity(rows);
    if hd % 4 != 0
        || s.pool == 0
        || s.pool * s.pool > u8::MAX as usize
        || w.position_table.len() < 2 * s.positions * h
    {
        return Err(err("vision: a tower geometry this engine does not serve"));
    }
    let (half, quarter) = (hd / 2, hd / 4);
    // The bounds' kernels only when a linear is clipped (E2B, E4B; not 26B-A4B).
    let mut clamp = Vec::new();
    if w.layers
        .iter()
        .any(|l| l.linears.iter().any(|lin| lin.clip.is_some()))
    {
        for d in [h, qd, inter] {
            if !clamp.iter().any(|(e, _)| *e == d) {
                clamp.push((d, elem(ElemOp::Clamp, d)?));
            }
        }
    }
    if s.standardize != w.standardize.is_some_and(|t| t.len() >= 2 * h) {
        return Err(err("vision: standardize without its two vectors"));
    }
    let kernels = Kernels {
        norm_h: elem(ElemOp::RmsNorm, h)?,
        norm_head: elem(ElemOp::RmsNorm, hd)?,
        rope: elem(ElemOp::Rope, half)?,
        add_h: elem(ElemOp::ResidualAdd, h)?,
        gelu: elem(ElemOp::GeluTanh, inter)?,
        clamp,
        pool: elem(ElemOp::PoolRows((s.pool * s.pool) as u8), h)?,
        standardize: if s.standardize {
            Some(elem(ElemOp::Standardize, h)?)
        } else {
            None
        },
        sdpa: tower_sdpa(hd)?,
        block: None,
    };
    // `HERBERT_VISION_ATTN_ROWS=N`: queries per block of the attention (0: the row kernel alone).
    let block_rows = std::env::var("HERBERT_VISION_ATTN_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(crate::jit::emit_sdpa::SDPA_PREFILL_MAX_ROWS);
    let kernels = Kernels {
        block: crate::jit::sdpa::prefill_kernel_full(
            SdpaShape {
                head_dim: hd,
                sink: false,
            },
            block_rows,
            qd,
            qd,
            cap,
            kernels.sdpa.spec().qk_group,
            0,
        )
        .map_err(err)?,
        ..kernels
    };
    let chunk = kernels.block.as_ref().map_or(1, |k| k.rows());
    // The R16 copies when the block kernel reads them.
    let r16 = r16 && kernels.block.is_some();
    if b.scores_stride < kernels.block.as_ref().map_or(0, |k| k.scratch_len())
        || b.keys.len() < cap * qd
        || (r16 && b.keys_r16.len() < cap * qd)
    {
        return Err(err(
            "vision: the attention's scratch is larger than its slot",
        ));
    }
    let p = |v: &mut crate::core::avec::AVec<u16>| v.as_mut_ptr();
    let (hb, xn, xc, q, k, v, attn, o, gate, up) = (
        p(&mut b.h),
        p(&mut b.xn),
        p(&mut b.xc),
        p(&mut b.q),
        p(&mut b.k),
        p(&mut b.v),
        p(&mut b.attn),
        p(&mut b.o),
        p(&mut b.gate),
        p(&mut b.up),
    );
    let packed = Buffer {
        ptr: b.packed.as_mut_ptr(),
        len: b.packed.len(),
    };
    let (h2, qd2, hd2) = ((h * 2), (qd * 2), (hd * 2));

    // ── The patches through the layers ──────────────────────────────────────
    let mut builder = Builder::new(rows)?;
    gemm(
        &mut builder,
        b.patches.as_mut_ptr(),
        w.input_proj,
        hb,
        s.pixels,
        h,
        packed,
        workers,
    )?;
    // The positions: `t = table[0][x]`, `u = table[1][y]`, `t += u`, `h += t` (the sum rounded
    // before the addition).
    let table = w.position_table.as_ptr();
    for (index, at, dst) in [
        (b.pos_x.as_ptr(), 0, xn),
        (b.pos_y.as_ptr(), s.positions * h, o),
    ] {
        builder.label("vision-position-lookup");
        builder.rows(
            Native::embedding()?,
            binding(
                &[
                    (index as u64, 4),
                    (unsafe { table.add(at) } as u64, 0),
                    (dst as u64, h2),
                    (h as u64, 0),
                ],
                &[],
            ),
            workers,
            workers,
        )?;
    }
    rows_node(
        &mut builder,
        "vision-position-sum",
        &kernels.add_h,
        &[(xn as u64, h2), (o as u64, h2)],
        &[],
        workers,
    )?;
    rows_node(
        &mut builder,
        "vision-position-add",
        &kernels.add_h,
        &[(hb as u64, h2), (xn as u64, h2)],
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
    // `HERBERT_VISION_PLAN=1`: each layer's attention kernel on stderr when the tower is mounted
    // (same bits on both paths on x86-64: this line says which one runs).
    let plan = std::env::var_os("HERBERT_VISION_PLAN").is_some_and(|v| v != "0");
    let attn_cyclic = std::env::var_os("HERBERT_VISION_ATTN_CYCLIC").is_none_or(|v| v != "0");
    for (l, lw) in w.layers.iter().enumerate() {
        if plan {
            match kernels.block.as_deref() {
                Some(p) => eprintln!(
                    "[vision plan] layer {l} head_dim {hd}, {rows} patches: block of {} rows{}, scratch {} f32, {} bytes",
                    p.rows(),
                    if p.spec().full { ", unmasked" } else { "" },
                    p.scratch_len(),
                    p.code_bytes()
                ),
                None => eprintln!("[vision plan] layer {l} head_dim {hd}, {rows} patches: row kernel"),
            }
        }
        builder.begin_layer(l);
        rows_node(
            &mut builder,
            "vision-norm",
            &kernels.norm_h,
            &[
                (hb as u64, h2),
                (lw.input_norm.as_ptr() as u64, 0),
                (xn as u64, h2),
            ],
            &[s.eps],
            workers,
        )?;
        for (i, dst) in [(0usize, q), (1, k), (2, v)] {
            let prev = i.checked_sub(1).map(|j| &lw.linears[j]);
            linear(
                &mut builder,
                &kernels,
                &lw.linears[i],
                prev,
                xn,
                dst,
                xc,
                packed,
                workers,
            )?;
        }
        // Q and K: normed per head, the first half of every head rotated by `x`, the second by `y`.
        for (data, weight) in [(q, lw.q_norm), (k, lw.k_norm)] {
            if weight.len() < hd {
                return Err(err("vision: a head norm weight too short"));
            }
            builder.label("vision-head-norm-rope-x");
            builder.rows(
                Native::norm_rope(Some(&kernels.norm_head), &kernels.rope, nh, hd)?,
                binding(
                    &[
                        (data as u64, qd2),
                        (weight.as_ptr() as u64, 0),
                        (b.cos.as_ptr() as u64, half * 2),
                        (b.sin.as_ptr() as u64, half * 2),
                    ],
                    &[s.eps],
                ),
                workers,
                workers,
            )?;
            builder.label("vision-rope-y");
            builder.rows(
                Native::norm_rope(None, &kernels.rope, nh, hd)?,
                binding(
                    &[
                        (unsafe { data.add(half) } as u64, qd2),
                        (weight.as_ptr() as u64, 0),
                        (unsafe { b.cos.as_ptr().add(quarter) } as u64, half * 2),
                        (unsafe { b.sin.as_ptr().add(quarter) } as u64, half * 2),
                    ],
                    &[s.eps],
                ),
                workers,
                workers,
            )?;
        }
        // V: normed per head without a weight.
        builder.label("vision-value-norm");
        builder.grid(
            Native::elementary(&kernels.norm_head)?,
            binding(
                &[
                    (v as u64, hd2),
                    (b.ones.as_ptr() as u64, 0),
                    (v as u64, hd2),
                ],
                &[s.eps],
            ),
            TaskGrid {
                rows_per_task: 1,
                tasks_per_tile: nh,
                last_block_only: false,
            },
            workers,
            workers,
        )?;
        // K and V head-major, as a cache of the block's rows.
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
        // Every patch sees every other.
        let call = builder.keep(ex::abi::AttentionCall {
            q,
            keys: b.keys.as_ptr(),
            values: b.values.as_ptr(),
            output: attn,
            rows: &raw const builder.frame.rows,
            past: &raw const builder.frame.position,
            kv_stride: cap * hd,
            scale: 1.0,
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
                &kernels.sdpa,
                kernels.block.as_deref(),
                ex::abi::AttentionSpec {
                    heads: nh,
                    gqa: 1,
                    head_dim: hd,
                    q_row: qd,
                    q_head: hd,
                    out_row: qd,
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
        // The tasks dealt cyclically: every worker on the same head at once, its keys and values (8
        // MB at 28 704 patches of head_dim 72) shared in L3 instead of one head per worker.
        // `HERBERT_VISION_ATTN_CYCLIC=0`: the contiguous ranges. The same bits: a task's
        // output depends on nothing but its head and its rows.
        if attn_cyclic {
            builder.cyclic_last();
        }
        linear(
            &mut builder,
            &kernels,
            &lw.linears[3],
            None,
            attn,
            o,
            xc,
            packed,
            workers,
        )?;
        rows_node(
            &mut builder,
            "vision-norm",
            &kernels.norm_h,
            &[
                (o as u64, h2),
                (lw.post_attn_norm.as_ptr() as u64, 0),
                (xn as u64, h2),
            ],
            &[s.eps],
            workers,
        )?;
        rows_node(
            &mut builder,
            "vision-residual",
            &kernels.add_h,
            &[(hb as u64, h2), (xn as u64, h2)],
            &[],
            workers,
        )?;
        // The MLP.
        rows_node(
            &mut builder,
            "vision-norm",
            &kernels.norm_h,
            &[
                (hb as u64, h2),
                (lw.pre_ff_norm.as_ptr() as u64, 0),
                (xn as u64, h2),
            ],
            &[s.eps],
            workers,
        )?;
        linear(
            &mut builder,
            &kernels,
            &lw.linears[4],
            None,
            xn,
            gate,
            xc,
            packed,
            workers,
        )?;
        linear(
            &mut builder,
            &kernels,
            &lw.linears[5],
            Some(&lw.linears[4]),
            xn,
            up,
            xc,
            packed,
            workers,
        )?;
        rows_node(
            &mut builder,
            "vision-gelu",
            &kernels.gelu,
            &[
                (gate as u64, inter * 2),
                (up as u64, inter * 2),
                (gate as u64, inter * 2),
            ],
            &[],
            workers,
        )?;
        linear(
            &mut builder,
            &kernels,
            &lw.linears[6],
            None,
            gate,
            o,
            xc,
            packed,
            workers,
        )?;
        rows_node(
            &mut builder,
            "vision-norm",
            &kernels.norm_h,
            &[
                (o as u64, h2),
                (lw.post_ff_norm.as_ptr() as u64, 0),
                (xn as u64, h2),
            ],
            &[s.eps],
            workers,
        )?;
        rows_node(
            &mut builder,
            "vision-residual",
            &kernels.add_h,
            &[(hb as u64, h2), (xn as u64, h2)],
            &[],
            workers,
        )?;
        builder.layer_end(l);
    }
    let layers = builder.finish_with_instrumentation(1, spin, trace, false, false, false)?;

    // ── The pooling and the projection ──────────────────────────────────────
    let k2 = s.pool * s.pool;
    let mut builder = Builder::new(s.max_soft)?;
    // `√hidden` as the reference's 16-bit scalar. With `standardize` (transformers ≥ 5.20's
    // `Gemma4VisionModel`): the mean alone in bf16 (times one, exact), then `(x · √hidden − b)
    // · c` in f32 with `√hidden` the f32 of the double, one rounding.
    let scale = f32::from_bits(u32::from(crate::engine::packs::to_bf16((h as f32).sqrt())) << 16);
    let pool_scale = if s.standardize { 1.0 } else { scale };
    rows_node(
        &mut builder,
        "vision-pool",
        &kernels.pool,
        &[(hb as u64, k2 * h2), (b.pooled.as_mut_ptr() as u64, h2)],
        &[pool_scale, 1.0 / k2 as f32],
        workers,
    )?;
    if let (Some(kernel), Some(std)) = (&kernels.standardize, w.standardize) {
        let pooled = b.pooled.as_mut_ptr() as u64;
        rows_node(
            &mut builder,
            "vision-standardize",
            kernel,
            &[(pooled, h2), (std.as_ptr() as u64, 0), (pooled, h2)],
            &[(h as f64).sqrt() as f32],
            workers,
        )?;
    }
    rows_node(
        &mut builder,
        "vision-embed-norm",
        &kernels.norm_h,
        &[
            (b.pooled.as_ptr() as u64, h2),
            (b.ones.as_ptr() as u64, 0),
            (b.pooled_n.as_mut_ptr() as u64, h2),
        ],
        &[s.eps],
        workers,
    )?;
    let packed_soft = Buffer {
        ptr: b.packed.as_mut_ptr(),
        len: b.packed.len(),
    };
    gemm(
        &mut builder,
        b.pooled_n.as_mut_ptr(),
        w.embed_proj,
        b.out.as_mut_ptr(),
        h,
        s.out_hidden,
        packed_soft,
        workers,
    )?;
    let embed = builder.finish_with_instrumentation(1, spin, false, false, false, false)?;
    Ok(TowerPrograms { layers, embed })
}

// ── 12B's encoder-free embedder ──────────────────────────────────────────────

/// The geometry of 12B's encoder-free embedder (`gemma4_unified_vision`): one soft token per merged
/// patch.
#[derive(Clone, Copy, Debug)]
pub struct UnifiedShape {
    /// The embedder's width (`mm_embed_dim`).
    pub hidden: usize,
    /// Values of one merged patch, `3 · (patch · pool)²`.
    pub pixels: usize,
    /// Positions per axis of the factorized table (`mm_posemb_size`).
    pub positions: usize,
    /// The LayerNorms' epsilon (`nn.LayerNorm`'s default).
    pub eps_ln: f32,
    /// The epsilon of the embedder's norm without weight.
    pub eps_rms: f32,
    /// Soft tokens of one image at most: the rows of the program.
    pub max_soft: usize,
    /// The language model's width.
    pub out_hidden: usize,
}

impl UnifiedShape {
    /// The dense's input width: the patch, then a column of ones that carries the bias through the
    /// GEMM (its f32 accumulator: `Σ x·w + 1·b`, rounded once, as the bf16 `nn.Linear`), padded
    /// to keep the rows 64-byte aligned.
    pub fn dense_k(&self) -> usize {
        self.pixels + 32
    }
}

/// The embedder's weights.
pub struct UnifiedWeights<'a> {
    /// `patch_ln1`, `g` then `b` (`pixels` each).
    pub ln1: &'a [u16],
    /// `patch_dense` with its bias as the column `pixels`, `hidden × dense_k`, packed.
    pub dense: &'a [u16],
    /// `patch_ln2`, `g` then `b`.
    pub ln2: &'a [u16],
    /// The `x` half of `pos_embedding`, `[positions][hidden]`.
    pub pos_x: &'a [u16],
    /// Its `y` half.
    pub pos_y: &'a [u16],
    /// `pos_norm`, `g` then `b`.
    pub ln3: &'a [u16],
    /// `embed_vision.embedding_projection`, `out_hidden × hidden`, packed.
    pub proj: &'a [u16],
}

/// The embedder's buffers. The host writes `patches`, `pos_x` and `pos_y` before a call; `out`
/// holds the soft tokens after it.
pub struct UnifiedBuffers {
    /// `[max_soft][pixels]`, the merged patches in raster order, `p/255` in bf16.
    pub patches: crate::core::avec::AVec<u16>,
    /// The column of each patch in the grid.
    pub pos_x: crate::core::avec::AVec<u32>,
    /// Its row.
    pub pos_y: crate::core::avec::AVec<u32>,
    /// The soft tokens, `[max_soft][out_hidden]`: the table the image regime reads.
    pub out: crate::core::avec::AVec<u16>,
    /// `[max_soft][dense_k]`: `LN₁` of the patches, then a 1 and zeros (written once).
    xn: crate::core::avec::AVec<u16>,
    a: crate::core::avec::AVec<u16>,
    b: crate::core::avec::AVec<u16>,
    c: crate::core::avec::AVec<u16>,
    e: crate::core::avec::AVec<u16>,
    ones: crate::core::avec::AVec<u16>,
    packed: crate::core::avec::AVec<u16>,
}

impl UnifiedBuffers {
    /// Everything for `shape`, the column of ones in place.
    pub fn new(shape: &UnifiedShape) -> Self {
        use crate::core::avec::AVec;
        let (r, d, kd) = (shape.max_soft, shape.hidden, shape.dense_k());
        let packed = [(kd, d), (d, shape.out_hidden)]
            .iter()
            .map(|&(k, n)| {
                let plan = crate::jit::arch::gemm_plan_bf16(GemmShape { k, n }, r);
                if plan.opts.matrix {
                    crate::jit::emit::packed_bytes(plan.opts, r, k).div_ceil(2)
                } else {
                    0
                }
            })
            .max()
            .unwrap_or(0);
        let mut xn = vec![0u16; r * kd];
        for row in 0..r {
            xn[row * kd + shape.pixels] = 0x3f80;
        }
        Self {
            patches: AVec::zeroed(r * shape.pixels),
            pos_x: AVec::zeroed(r),
            pos_y: AVec::zeroed(r),
            out: AVec::zeroed(r * shape.out_hidden),
            xn: AVec::from_vec(xn),
            a: AVec::zeroed(r * d),
            b: AVec::zeroed(r * d),
            c: AVec::zeroed(r * d),
            e: AVec::zeroed(r * d),
            ones: AVec::from_vec(vec![0x3f80; d]),
            packed: AVec::zeroed(packed.max(1)),
        }
    }

    /// Bytes reserved.
    pub fn bytes(&self) -> usize {
        let u16s = [
            &self.patches,
            &self.out,
            &self.xn,
            &self.a,
            &self.b,
            &self.c,
            &self.e,
            &self.ones,
            &self.packed,
        ];
        u16s.iter().map(|b| b.len() * 2).sum::<usize>() + (self.pos_x.len() + self.pos_y.len()) * 4
    }
}

/// The embedder's program over `b` (`rows` = the soft tokens), as `Gemma4UnifiedVisionEmbedder`: `t
/// = LN₁(p)`, `t = dense(t) + bias` (one rounding), `t = LN₂(t)`, `t += bf16(table[x][0] +
/// table[y][1])`, `t = LN₃(t)`, then the embedder: `t / rms(t)` without weight and the projection;
/// each step rounded to bf16 as the bf16 module does. Every node is an emitted kernel of this crate
/// (`ElemOp::LayerNorm`, the embedding lookup, `ElemOp::ResidualAdd`, `ElemOp::RmsNorm`, the bf16
/// GEMM).
pub(crate) fn lower_unified(
    shape: &UnifiedShape,
    w: &UnifiedWeights<'_>,
    b: &mut UnifiedBuffers,
    workers: usize,
    spin: Option<u32>,
) -> std::result::Result<Mounted, ExecutionError> {
    let s = *shape;
    let (d, pix, kd) = (s.hidden, s.pixels, s.dense_k());
    if w.ln1.len() < 2 * pix
        || [w.ln2, w.ln3].iter().any(|n| n.len() < 2 * d)
        || w.pos_x.len() < s.positions * d
        || w.pos_y.len() < s.positions * d
    {
        return Err(err(
            "vision: an encoder-free embedder this engine does not serve",
        ));
    }
    let ln_pix = elem(ElemOp::LayerNorm, pix)?;
    let ln_d = elem(ElemOp::LayerNorm, d)?;
    let add_d = elem(ElemOp::ResidualAdd, d)?;
    let rms_d = elem(ElemOp::RmsNorm, d)?;
    let p = |v: &mut crate::core::avec::AVec<u16>| v.as_mut_ptr();
    let (xn, ba, bb, bc, be) = (
        p(&mut b.xn),
        p(&mut b.a),
        p(&mut b.b),
        p(&mut b.c),
        p(&mut b.e),
    );
    let packed = Buffer {
        ptr: b.packed.as_mut_ptr(),
        len: b.packed.len(),
    };
    let d2 = d * 2;
    let mut builder = Builder::new(s.max_soft)?;
    rows_node(
        &mut builder,
        "vision-ln1",
        &ln_pix,
        &[
            (b.patches.as_ptr() as u64, pix * 2),
            (w.ln1.as_ptr() as u64, 0),
            (xn as u64, kd * 2),
        ],
        &[s.eps_ln],
        workers,
    )?;
    gemm(&mut builder, xn, w.dense, ba, kd, d, packed, workers)?;
    rows_node(
        &mut builder,
        "vision-ln2",
        &ln_d,
        &[(ba as u64, d2), (w.ln2.as_ptr() as u64, 0), (bb as u64, d2)],
        &[s.eps_ln],
        workers,
    )?;
    // The positions: `c = table_x[x]`, `e = table_y[y]`, `c += e`, `b += c` (the sum rounded before
    // the addition).
    for (index, table, dst) in [
        (b.pos_x.as_ptr(), w.pos_x, bc),
        (b.pos_y.as_ptr(), w.pos_y, be),
    ] {
        builder.label("vision-position-lookup");
        builder.rows(
            Native::embedding()?,
            binding(
                &[
                    (index as u64, 4),
                    (table.as_ptr() as u64, 0),
                    (dst as u64, d2),
                    (d as u64, 0),
                ],
                &[],
            ),
            workers,
            workers,
        )?;
    }
    rows_node(
        &mut builder,
        "vision-position-sum",
        &add_d,
        &[(bc as u64, d2), (be as u64, d2)],
        &[],
        workers,
    )?;
    rows_node(
        &mut builder,
        "vision-position-add",
        &add_d,
        &[(bb as u64, d2), (bc as u64, d2)],
        &[],
        workers,
    )?;
    rows_node(
        &mut builder,
        "vision-ln3",
        &ln_d,
        &[(bb as u64, d2), (w.ln3.as_ptr() as u64, 0), (ba as u64, d2)],
        &[s.eps_ln],
        workers,
    )?;
    rows_node(
        &mut builder,
        "vision-embed-norm",
        &rms_d,
        &[
            (ba as u64, d2),
            (b.ones.as_ptr() as u64, 0),
            (bb as u64, d2),
        ],
        &[s.eps_rms],
        workers,
    )?;
    gemm(
        &mut builder,
        bb,
        w.proj,
        b.out.as_mut_ptr(),
        d,
        s.out_hidden,
        packed,
        workers,
    )?;
    builder.finish_with_instrumentation(1, spin, false, false, false, false)
}
