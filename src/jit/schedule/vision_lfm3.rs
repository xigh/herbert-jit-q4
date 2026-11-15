//! The LFM3-VL vision tower (`lfm3_vl_vision`, LFM3-VL-2B) and its projector,
//! lowered into two generated programs: the patches through the tower's
//! layers, then the projection of each block of `merge²` patches into the
//! language model's space. Every computation is an emitted kernel of this
//! crate: the bf16 GEMM, the row norms, the head norms and the RoPE, the K/V
//! append, the unmasked block attention of the Gemma tower (here with two
//! query heads per K/V head), `ElemOp::SwiGlu`, `ElemOp::LayerNorm`,
//! `ElemOp::GeluErf`, the embedding lookup and `ElemOp::DepthwiseRows`; the
//! host lays out the patches, their RoPE tables and the neighbours of each
//! patch (`engine/vision_lfm3.rs`).
//!
//! The patches arrive in **merge-block order**, as the processor writes them:
//! the `2 × 2` patches of one language token consecutive (raster order inside
//! the block), the blocks in raster order. A layer, as `Lfm3VlVisionLayer`:
//! `h += op(norm(h))`, then `h += w2(silu(w1(norm(h))) · w3(norm(h)))`, `op`
//! being one of
//!
//! - the attention: `q, k = rope(norm(q(x))), rope(norm(k(x)))`, `v = v(x)`, 16 query heads over 8
//!   K/V heads, every patch sees every other, scale `head_dim^-½`; the RoPE is axial over the whole
//!   head (`rotate_half`, the first 16 angles by the patch's row, the next 16 by its column);
//! - the gated depthwise convolution: `B, C, x = in_proj(x)`, `g = bf16(B·x)`, `c = bf16(Σ₉ w·g)`
//!   over the 3 × 3 neighbours of the patch in the grid (zero outside), `out_proj(bf16(C·c))`. The
//!   products are `ElemOp::DepthwiseRows` of one tap (`bf16(x·w)`, `w` the other operand's row);
//!   the neighbours are gathered by nine embedding lookups (the host's table of their rows, a zero
//!   row past the patches for the border) into nine consecutive rows per patch, then
//!   `ElemOp::DepthwiseRows(9)` with the kernel transposed.
//!
//! The patch embedding (no bias) is followed by an RMS norm, the last layer by
//! nothing; the projector is a LayerNorm per patch, then four consecutive rows
//! read as one row of `4 · hidden`, `linear_1` + bias, the exact GeLU,
//! `linear_2` + bias (each bias added after the GEMM's rounding).

use super::lowering::{binding, Buffer};
use super::vision::{elem, gemm_labeled, rows_node};
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

/// The geometry of the tower and of its projector.
#[derive(Clone, Copy, Debug)]
pub struct Lfm3TowerShape {
    /// The tower's width.
    pub hidden: usize,
    /// Query heads.
    pub heads: usize,
    /// K/V heads.
    pub kv_heads: usize,
    /// Elements per head.
    pub head_dim: usize,
    /// The MLP's width.
    pub intermediate: usize,
    /// Values of one patch, `3 · temporal · patch²`.
    pub pixels: usize,
    /// The RMS norms' epsilon.
    pub eps: f32,
    /// The projector's LayerNorm epsilon.
    pub eps_ln: f32,
    /// Patches merged into one language token (`merge²`).
    pub merge2: usize,
    /// The projector's width.
    pub proj_hidden: usize,
    /// The language model's width.
    pub out_hidden: usize,
    /// Patches of one image at most: the rows of the tower's program.
    pub max_rows: usize,
}

impl Lfm3TowerShape {
    fn q_dim(&self) -> usize {
        self.heads * self.head_dim
    }
    fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }
    /// Language tokens of one image at most.
    pub fn max_soft(&self) -> usize {
        self.max_rows / self.merge2
    }
}

/// One layer: its operator (attention or convolution), then the MLP.
pub enum Lfm3Operator<'a> {
    /// The attention.
    Attention {
        /// `q_proj`, R16-packed.
        q: &'a [u16],
        /// `k_proj`.
        k: &'a [u16],
        /// `v_proj`.
        v: &'a [u16],
        /// `out_proj`.
        out: &'a [u16],
        /// The query heads' norm.
        q_norm: &'a [u16],
        /// The key heads' norm.
        k_norm: &'a [u16],
    },
    /// The gated depthwise convolution.
    Conv {
        /// `in_proj` (`3 · hidden` rows: B, C, x), R16-packed.
        in_proj: &'a [u16],
        /// The kernel transposed, `[9][hidden]`.
        kernel: &'a [u16],
        /// `out_proj`.
        out: &'a [u16],
    },
}

/// One layer of the tower.
pub struct Lfm3LayerWeights<'a> {
    /// The norm before the operator.
    pub operator_norm: &'a [u16],
    /// The norm before the MLP.
    pub ffn_norm: &'a [u16],
    /// The attention or the convolution.
    pub op: Lfm3Operator<'a>,
    /// `w1` (gate), `w3` (up), `w2` (down), R16-packed.
    pub w1: &'a [u16],
    /// `w3` (up).
    pub w3: &'a [u16],
    /// `w2` (down).
    pub w2: &'a [u16],
}

/// The tower's and the projector's weights.
pub struct Lfm3TowerWeights<'a> {
    /// `hidden × pixels`, packed.
    pub patchifier: &'a [u16],
    /// The norm after it.
    pub embedding_norm: &'a [u16],
    /// The layers.
    pub layers: Vec<Lfm3LayerWeights<'a>>,
    /// The projector's LayerNorm, `g` then `b`.
    pub ln: &'a [u16],
    /// `proj_hidden × merge2·hidden`, packed, and its bias.
    pub linear_1: &'a [u16],
    /// Its bias.
    pub bias_1: &'a [u16],
    /// `out_hidden × proj_hidden`, packed, and its bias.
    pub linear_2: &'a [u16],
    /// Its bias.
    pub bias_2: &'a [u16],
}

/// The tower's buffers for `max_rows` patches. The host writes `patches`, `cos`, `sin` and
/// `neighbours` before a call.
pub struct Lfm3TowerBuffers {
    /// `[rows][pixels]`, merge-block order, bf16.
    pub patches: AVec<u16>,
    /// `[rows][head_dim / 2]`: the cosines of the row angles, then those of the column angles.
    pub cos: AVec<u16>,
    /// The sines, the same way.
    pub sin: AVec<u16>,
    /// `[9][rows]`: for tap `t = 3·ky + kx`, the row of each patch's neighbour `(y + ky − 1, x +
    /// kx − 1)`, or `max_rows` (a zero row) outside the grid.
    pub neighbours: AVec<u32>,
    h: AVec<u16>,
    xn: AVec<u16>,
    bcx: AVec<u16>,
    gated: AVec<u16>,
    gathered: AVec<u16>,
    conv: AVec<u16>,
    q: AVec<u16>,
    k: AVec<u16>,
    v: AVec<u16>,
    attn: AVec<u16>,
    o: AVec<u16>,
    gate: AVec<u16>,
    up: AVec<u16>,
    keys: AVec<u16>,
    values: AVec<u16>,
    keys_r16: AVec<u16>,
    values_r16: AVec<u16>,
    scores: AVec<f32>,
    packed: AVec<u16>,
    p1: AVec<u16>,
    ones: AVec<u16>,
    scores_stride: usize,
}

impl Lfm3TowerBuffers {
    /// Everything, zeroed, for `s` and `workers` participants.
    pub fn new(s: &Lfm3TowerShape, workers: usize) -> Self {
        let r = s.max_rows;
        let (h, qd, kvd, inter) = (s.hidden, s.q_dim(), s.kv_dim(), s.intermediate);
        let soft = s.max_soft();
        let packed = [
            (s.pixels, h, r),
            (h, 3 * h, r),
            (qd, h, r),
            (inter, h, r),
            (s.merge2 * h, s.proj_hidden, soft),
            (s.proj_hidden, s.out_hidden, soft),
        ]
        .iter()
        .map(|&(k, n, rows)| {
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
            patches: AVec::zeroed(r * s.pixels),
            cos: AVec::zeroed(r * s.head_dim / 2),
            sin: AVec::zeroed(r * s.head_dim / 2),
            neighbours: AVec::zeroed(9 * r),
            h: AVec::zeroed(r * h),
            xn: AVec::zeroed(r * h),
            bcx: AVec::zeroed(r * 3 * h),
            // One zero row past the patches: the neighbours outside the grid.
            gated: AVec::zeroed((r + 1) * h),
            gathered: AVec::zeroed(r * 9 * h),
            conv: AVec::zeroed(r * h),
            q: AVec::zeroed(r * qd),
            k: AVec::zeroed(r * kvd),
            v: AVec::zeroed(r * kvd),
            attn: AVec::zeroed(r * qd),
            o: AVec::zeroed(r * h),
            gate: AVec::zeroed(r * inter),
            up: AVec::zeroed(r * inter),
            keys: AVec::zeroed(cap * kvd),
            values: AVec::zeroed(cap * kvd),
            keys_r16: AVec::zeroed(if r16 { cap * kvd } else { 0 }),
            values_r16: AVec::zeroed(if r16 { cap * kvd } else { 0 }),
            scores: AVec::zeroed(scores_stride * workers.max(1)),
            packed: AVec::zeroed(packed),
            p1: AVec::zeroed(soft * s.proj_hidden),
            ones: AVec::from_vec(vec![0x3f80; s.proj_hidden]),
            scores_stride,
        }
    }

    /// Bytes reserved.
    pub fn bytes(&self) -> usize {
        let u16s = [
            &self.patches,
            &self.cos,
            &self.sin,
            &self.h,
            &self.xn,
            &self.bcx,
            &self.gated,
            &self.gathered,
            &self.conv,
            &self.q,
            &self.k,
            &self.v,
            &self.attn,
            &self.o,
            &self.gate,
            &self.up,
            &self.keys,
            &self.values,
            &self.keys_r16,
            &self.values_r16,
            &self.packed,
            &self.p1,
            &self.ones,
        ];
        u16s.iter().map(|b| b.len() * 2).sum::<usize>()
            + (self.neighbours.len() + self.scores.len()) * 4
    }
}

/// The two programs: the patches through the layers (`rows` = the patches), then the projector
/// (`rows` = the tokens).
pub struct Lfm3TowerPrograms {
    pub(crate) layers: Mounted,
    pub(crate) project: Mounted,
}

/// The tower's two programs over `b`, the projector writing `[tokens][out_hidden]` into `out`, for
/// `workers` participants. The weights, the buffers and `out` must outlive them.
pub(crate) fn lower_lfm3_tower(
    s: &Lfm3TowerShape,
    w: &Lfm3TowerWeights<'_>,
    b: &mut Lfm3TowerBuffers,
    out: &mut [u16],
    workers: usize,
    spin: Option<u32>,
    trace: bool,
) -> std::result::Result<Lfm3TowerPrograms, ExecutionError> {
    let s = *s;
    let (h, hd, nh, nkv, inter, qd, kvd) = (
        s.hidden,
        s.head_dim,
        s.heads,
        s.kv_heads,
        s.intermediate,
        s.q_dim(),
        s.kv_dim(),
    );
    let rows = s.max_rows;
    if hd % 32 != 0
        || nkv == 0
        || nh % nkv != 0
        || s.merge2 == 0
        || !rows.is_multiple_of(s.merge2)
        || out.len() < s.max_soft() * s.out_hidden
    {
        return Err(err(
            "vision: an LFM3 tower geometry this engine does not serve",
        ));
    }
    let half = hd / 2;
    let (cap, r16) = crate::jit::sdpa::full_kv_capacity(rows);
    let norm_h = elem(ElemOp::RmsNorm, h)?;
    let norm_head = elem(ElemOp::RmsNorm, hd)?;
    let rope = elem(ElemOp::Rope, hd)?;
    let add_h = elem(ElemOp::ResidualAdd, h)?;
    let mul_h = elem(ElemOp::DepthwiseRows(1), h)?;
    let dw9 = elem(ElemOp::DepthwiseRows(9), h)?;
    let swiglu = elem(ElemOp::SwiGlu, inter)?;
    let ln = elem(ElemOp::LayerNorm, h)?;
    let gelu = elem(ElemOp::GeluErf, s.proj_hidden)?;
    let add_p = elem(ElemOp::ResidualAdd, s.proj_hidden)?;
    let add_out = elem(ElemOp::ResidualAdd, s.out_hidden)?;
    let shape = SdpaShape {
        head_dim: hd,
        sink: false,
    };
    let sdpa = crate::jit::sdpa::head_kernel(shape)
        .map_err(err)?
        .ok_or_else(|| err(format!("vision: no attention of head_dim {hd}")))?;
    // `HERBERT_VISION_ATTN_ROWS=N`: queries per block of the attention (0: the row kernel alone).
    let block_rows = std::env::var("HERBERT_VISION_ATTN_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(crate::jit::emit_sdpa::SDPA_PREFILL_MAX_ROWS);
    let block = crate::jit::sdpa::prefill_kernel_full(
        shape,
        block_rows,
        qd,
        qd,
        cap,
        sdpa.spec().qk_group,
        0,
    )
    .map_err(err)?;
    let chunk = block.as_ref().map_or(1, |k| k.rows());
    let r16 = r16 && block.is_some();
    if std::env::var_os("HERBERT_VISION_PLAN").is_some_and(|v| v != "0") {
        match block.as_deref() {
            Some(p) => eprintln!(
                "[vision plan] LFM3 tower, head_dim {hd}, {nh} heads over {nkv}, {rows} patches at most: block of {} rows{}, scratch {} f32",
                p.rows(),
                if p.spec().full { ", unmasked" } else { "" },
                p.scratch_len()
            ),
            None => eprintln!("[vision plan] LFM3 tower, head_dim {hd}: row kernel"),
        }
    }
    if b.scores_stride < block.as_ref().map_or(0, |k| k.scratch_len())
        || b.keys.len() < cap * kvd
        || (r16 && b.keys_r16.len() < cap * kvd)
    {
        return Err(err(
            "vision: the attention's scratch is larger than its slot",
        ));
    }
    if b.scores.len() < b.scores_stride * workers {
        return Err(err("vision: attention score slots too short"));
    }
    let p = |v: &mut AVec<u16>| v.as_mut_ptr();
    let (hb, xn, bcx, gated, gathered, conv, q, k, v, attn, o, gate, up) = (
        p(&mut b.h),
        p(&mut b.xn),
        p(&mut b.bcx),
        p(&mut b.gated),
        p(&mut b.gathered),
        p(&mut b.conv),
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
    let h2 = h * 2;
    let gemm = |builder: &mut Builder, x: *mut u16, wt: &[u16], y: *mut u16, k: usize, n: usize| {
        gemm_labeled(
            builder,
            "vision-projection",
            x,
            wt,
            y,
            k,
            n,
            packed,
            workers,
        )
    };

    // ── The patches through the layers ──────────────────────────────────────
    let mut builder = Builder::new(rows)?;
    gemm(
        &mut builder,
        b.patches.as_mut_ptr(),
        w.patchifier,
        xn,
        s.pixels,
        h,
    )?;
    rows_node(
        &mut builder,
        "vision-norm",
        &norm_h,
        &[
            (xn as u64, h2),
            (w.embedding_norm.as_ptr() as u64, 0),
            (hb as u64, h2),
        ],
        &[s.eps],
        workers,
    )?;
    let kv_spec = ex::abi::KvSpec {
        heads: nkv,
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
            &norm_h,
            &[
                (hb as u64, h2),
                (lw.operator_norm.as_ptr() as u64, 0),
                (xn as u64, h2),
            ],
            &[s.eps],
            workers,
        )?;
        match &lw.op {
            Lfm3Operator::Attention {
                q: wq,
                k: wk,
                v: wv,
                out,
                q_norm,
                k_norm,
            } => {
                gemm(&mut builder, xn, wq, q, h, qd)?;
                gemm(&mut builder, xn, wk, k, h, kvd)?;
                gemm(&mut builder, xn, wv, v, h, kvd)?;
                for (data, weight, heads, width) in [(q, *q_norm, nh, qd), (k, *k_norm, nkv, kvd)] {
                    if weight.len() < hd {
                        return Err(err("vision: a head norm weight too short"));
                    }
                    builder.label("vision-head-norm-rope");
                    builder.rows(
                        Native::norm_rope(Some(&norm_head), &rope, heads, hd)?,
                        binding(
                            &[
                                (data as u64, width * 2),
                                (weight.as_ptr() as u64, 0),
                                (b.cos.as_ptr() as u64, half * 2),
                                (b.sin.as_ptr() as u64, half * 2),
                            ],
                            &[s.eps],
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
                append.tasks = nkv;
                builder.label("vision-kv-append");
                builder.grid(
                    Native::kv_append(kv_spec)?,
                    append,
                    TaskGrid {
                        rows_per_task: rows,
                        tasks_per_tile: nkv,
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
                builder.label("vision-attention");
                builder.grid(
                    Native::attention_tile_selector(
                        &sdpa,
                        block.as_deref(),
                        ex::abi::AttentionSpec {
                            heads: nh,
                            gqa: nh / nkv,
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
                gemm(&mut builder, attn, out, o, qd, h)?;
            }
            Lfm3Operator::Conv {
                in_proj,
                kernel,
                out,
            } => {
                if kernel.len() < 9 * h {
                    return Err(err("vision: a convolution kernel too short"));
                }
                gemm(&mut builder, xn, in_proj, bcx, h, 3 * h)?;
                let bcx3 = 3 * h2;
                // g = bf16(B·x).
                rows_node(
                    &mut builder,
                    "vision-conv-gate",
                    &mul_h,
                    &[
                        (unsafe { bcx.add(2 * h) } as u64, bcx3),
                        (bcx as u64, bcx3),
                        (gated as u64, h2),
                    ],
                    &[],
                    workers,
                )?;
                // The nine neighbours of each patch, consecutive.
                for t in 0..9 {
                    builder.label("vision-conv-gather");
                    builder.rows(
                        Native::embedding()?,
                        binding(
                            &[
                                (unsafe { b.neighbours.as_ptr().add(t * rows) } as u64, 4),
                                (gated as u64, 0),
                                (unsafe { gathered.add(t * h) } as u64, 9 * h2),
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
                    "vision-conv",
                    &dw9,
                    &[
                        (gathered as u64, 9 * h2),
                        (kernel.as_ptr() as u64, 0),
                        (conv as u64, h2),
                    ],
                    &[],
                    workers,
                )?;
                // bf16(C·c).
                rows_node(
                    &mut builder,
                    "vision-conv-out-gate",
                    &mul_h,
                    &[
                        (conv as u64, h2),
                        (unsafe { bcx.add(h) } as u64, bcx3),
                        (xn as u64, h2),
                    ],
                    &[],
                    workers,
                )?;
                gemm(&mut builder, xn, out, o, h, h)?;
            }
        }
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
            &norm_h,
            &[
                (hb as u64, h2),
                (lw.ffn_norm.as_ptr() as u64, 0),
                (xn as u64, h2),
            ],
            &[s.eps],
            workers,
        )?;
        gemm(&mut builder, xn, lw.w1, gate, h, inter)?;
        gemm(&mut builder, xn, lw.w3, up, h, inter)?;
        rows_node(
            &mut builder,
            "vision-swiglu",
            &swiglu,
            &[
                (gate as u64, inter * 2),
                (up as u64, inter * 2),
                (gate as u64, inter * 2),
            ],
            &[],
            workers,
        )?;
        gemm(&mut builder, gate, lw.w2, o, inter, h)?;
        rows_node(
            &mut builder,
            "vision-residual",
            &add_h,
            &[(hb as u64, h2), (o as u64, h2)],
            &[],
            workers,
        )?;
        builder.layer_end(l);
    }
    if w.ln.len() < 2 * h {
        return Err(err("vision: the projector's LayerNorm too short"));
    }
    rows_node(
        &mut builder,
        "vision-projector-norm",
        &ln,
        &[(hb as u64, h2), (w.ln.as_ptr() as u64, 0), (xn as u64, h2)],
        &[s.eps_ln],
        workers,
    )?;
    let layers = builder.finish_with_instrumentation(1, spin, trace, false, false, false)?;

    // ── The projector: `merge²` consecutive rows as one ─────────────────────
    let mut builder = Builder::new(s.max_soft())?;
    let (pw, ow) = (s.proj_hidden, s.out_hidden);
    let p1 = b.p1.as_mut_ptr();
    gemm(&mut builder, xn, w.linear_1, p1, s.merge2 * h, pw)?;
    rows_node(
        &mut builder,
        "vision-projector-bias",
        &add_p,
        &[(p1 as u64, pw * 2), (w.bias_1.as_ptr() as u64, 0)],
        &[],
        workers,
    )?;
    rows_node(
        &mut builder,
        "vision-projector-gelu",
        &gelu,
        &[
            (p1 as u64, pw * 2),
            (b.ones.as_ptr() as u64, 0),
            (p1 as u64, pw * 2),
        ],
        &[],
        workers,
    )?;
    gemm(&mut builder, p1, w.linear_2, out.as_mut_ptr(), pw, ow)?;
    rows_node(
        &mut builder,
        "vision-projector-bias",
        &add_out,
        &[
            (out.as_mut_ptr() as u64, ow * 2),
            (w.bias_2.as_ptr() as u64, 0),
        ],
        &[],
        workers,
    )?;
    let project = builder.finish_with_instrumentation(1, spin, false, false, false, false)?;
    Ok(Lfm3TowerPrograms { layers, project })
}
