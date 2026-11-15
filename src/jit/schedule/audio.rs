//! The Gemma 4 audio tower (E2B, E4B) lowered into four generated programs of
//! emitted kernels:
//!
//! 1. `relative` (once, at mount): the sinusoids of the relative positions `window … 0` through
//!    every layer's `relative_k_proj`, next to the layer's `softplus(per_dim_scale)` (the table the
//!    attention reads);
//! 2. `conv0` (one row per output cell of the first subsampling layer): its 3 × 3 patches (laid out
//!    by the host from the log-mel frames, nine taps and zeros to sixteen) through a GEMM into 128
//!    channels, then the LayerNorm of the channels and the ReLU (`Clamp` to `[0, max]`);
//! 3. `conv1` (one row per output cell of the second): the nine taps of each cell gathered from
//!    `conv0`'s output by the embedding kernel (row indices from the host, a zero row for the
//!    padding), a GEMM into 32 channels, LayerNorm and ReLU — the `[t][32 · 32]` the projection
//!    reads;
//! 4. `layers` (one row per soft token): the projection into the tower's width, the Conformer
//!    layers, the output projection and its bias, the embedder's norm without weight and its
//!    projection into the language model's width ([`AudioBuffers::out`], the table the audio regime
//!    reads).
//!
//! A Conformer layer, as the reference (`Gemma4AudioLayer`):
//! `h += ½·norm(l2(silu(l1(norm(h)))))`; `h += norm(post(attn(q, k, v)))` on
//! `norm(h)`; `h += end(silu(norm(dw(glu(start(norm(h)))))))`; the second
//! half-step feed-forward; `h = norm(h)`. The attention is
//! [`ElemOp::LocalAttention`] — its block kernel by default, tiles of
//! [`ATTN_ROWS`] rows × one head per task (`HERBERT_AUDIO_ATTN_ROWS`: 8, 16,
//! 24 or 32; `0` the row kernel, one row and every head per task, the same
//! bits) — the causal depthwise convolution
//! [`ElemOp::DepthwiseRows`] over rows of a buffer that keeps `taps − 1` zero
//! rows before the first frame (the keys and the values keep `window − 1`).
//! The half step's ½ is folded into the post norm's weight when it is a power
//! of two (the same bits), else a `Scale` node. The clamps to
//! `±gradient_clipping` are left out at the checkpoints' 1e10 (no bf16 value
//! of the tower reaches it); a smaller bound is refused.

use super::lowering::{binding, Buffer};
use super::vision::{elem, gemm_labeled, rows_node, TowerLinear};
use crate::core::avec::AVec;
use crate::jit::arch::{KernelSet, Native};
use crate::jit::elem::CompiledElem;
use crate::jit::emit_elem::{ElemOp, LocalAttentionCall};
use crate::jit::execution::abi::TaskGrid;
use crate::jit::execution::{
    build::{Builder, Mounted},
    ExecutionError,
};
use crate::jit::GemmShape;
use std::sync::Arc;

fn err(e: impl std::fmt::Display) -> ExecutionError {
    ExecutionError(e.to_string())
}

/// Values of one patch of the first subsampling layer: nine taps, zeros to sixteen.
pub const CONV0_K: usize = 16;

/// The rows of a tile of the attention's block kernel, by default.
pub const ATTN_ROWS: usize = 16;

/// The rows the block kernel reads at most past the last one: q, k, v and the attention's output
/// are allocated to a multiple of the widest tile.
const ATTN_PAD: usize = 32;

/// The rows of a tile of the attention (`HERBERT_AUDIO_ATTN_ROWS`, [`ATTN_ROWS`] by default), `0`
/// for the row kernel.
pub fn attention_rows() -> std::result::Result<usize, ExecutionError> {
    match std::env::var("HERBERT_AUDIO_ATTN_ROWS") {
        Err(_) => Ok(ATTN_ROWS),
        Ok(v) => match v.trim().parse::<usize>() {
            Ok(r) if r == 0 || (r % 8 == 0 && r <= ATTN_PAD) => Ok(r),
            _ => Err(err(format!(
                "HERBERT_AUDIO_ATTN_ROWS={v}: 0 (the row kernel), 8, 16, 24 or 32"
            ))),
        },
    }
}

/// The geometry of a tower.
#[derive(Clone, Copy, Debug)]
pub struct AudioShape {
    /// The tower's width.
    pub hidden: usize,
    /// Its attention heads.
    pub heads: usize,
    /// The keys a row sees, itself included (`attention_context_left − 1`).
    pub window: usize,
    /// Taps of the light convolution.
    pub taps: usize,
    /// The channels of the two subsampling layers.
    pub c0: usize,
    /// The second one's.
    pub c1: usize,
    /// Mel bins of a frame.
    pub mels: usize,
    /// The width of the output projection (`output_proj_dims`).
    pub out_dims: usize,
    /// The language model's width.
    pub out_hidden: usize,
    /// The norms' epsilon.
    pub eps: f32,
    /// The attention's soft cap.
    pub cap: f32,
    /// The half step of the feed-forwards (`residual_weight`).
    pub residual: f32,
    /// Mel frames of the longest recording.
    pub max_frames: usize,
}

impl AudioShape {
    /// Frames after the first subsampling layer, at most.
    pub fn max_t0(&self) -> usize {
        self.max_frames.div_ceil(2)
    }
    /// Soft tokens of the longest recording.
    pub fn max_tokens(&self) -> usize {
        self.max_t0().div_ceil(2)
    }
    /// Rows of `conv0`: one per output cell, `[t0][mels / 2]`.
    pub fn max_rows0(&self) -> usize {
        self.max_t0() * (self.mels / 2)
    }
    /// Rows of `conv1`: `[t][mels / 4]`.
    pub fn max_rows1(&self) -> usize {
        self.max_tokens() * (self.mels / 4)
    }
    /// The input of the second layer's GEMM: nine taps of `c0` channels.
    pub fn k1(&self) -> usize {
        9 * self.c0
    }
    /// The projection's input: `[mels / 4][c1]` per frame.
    pub fn proj_in(&self) -> usize {
        (self.mels / 4) * self.c1
    }
    fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }
}

/// One layer of the tower. The linears are R16-packed bf16
/// ([`crate::model::PackedBf16R16Weight::pack_bf16`]).
pub struct AudioLayerWeights<'a> {
    /// The two feed-forwards: their pre norm, post norm, `ffw_layer_1`, `ffw_layer_2`.
    pub ff: [(&'a [u16], &'a [u16], TowerLinear<'a>, TowerLinear<'a>); 2],
    /// The norm before the attention.
    pub norm_pre_attn: &'a [u16],
    /// The norm of the attention's output.
    pub norm_post_attn: &'a [u16],
    /// The norm at the end of the layer.
    pub norm_out: &'a [u16],
    /// `q, k, v, post`.
    pub attn: [TowerLinear<'a>; 4],
    /// `relative_k_proj`, packed.
    pub rel_k: &'a [u16],
    /// `softplus(per_dim_scale)`, bf16, `head_dim`.
    pub softplus: &'a [u16],
    /// The norm before the light convolution.
    pub conv_pre: &'a [u16],
    /// `linear_start`, into `2·hidden`.
    pub conv_start: TowerLinear<'a>,
    /// The depthwise kernel transposed, `[taps][hidden]`.
    pub conv_dw: &'a [u16],
    /// The norm after the depthwise convolution.
    pub conv_norm: &'a [u16],
    /// `linear_end`.
    pub conv_end: TowerLinear<'a>,
}

/// The tower's weights.
pub struct AudioWeights<'a> {
    /// The first subsampling layer, `c0 × CONV0_K` packed, and its LayerNorm `[g | 0]`.
    pub conv0: &'a [u16],
    /// The first layer's LayerNorm, `[g | 0]`.
    pub norm0: &'a [u16],
    /// The second, `c1 × 9·c0` packed (taps major, the channels inside), and its LayerNorm.
    pub conv1: &'a [u16],
    /// The second layer's LayerNorm.
    pub norm1: &'a [u16],
    /// `input_proj_linear`, `hidden × proj_in`, packed.
    pub input_proj: &'a [u16],
    /// `output_proj`, `out_dims × hidden`, packed, and its bias.
    pub out_proj: &'a [u16],
    /// `output_proj`'s bias.
    pub out_bias: &'a [u16],
    /// `embed_audio.embedding_projection`, `out_hidden × out_dims`, packed.
    pub embed_proj: &'a [u16],
    /// The sinusoids of the relative positions `window … 0`, `[window + 1][hidden]`.
    pub positions: &'a [u16],
    /// The Conformer layers.
    pub layers: Vec<AudioLayerWeights<'a>>,
}

/// The tower's buffers, allocated once for the longest recording. The host writes `patches0` and
/// `index1` before a call; `out` holds the soft tokens after it.
pub struct AudioBuffers {
    /// `[rows0][CONV0_K]`, the patches of the first layer.
    pub patches0: AVec<u16>,
    /// `[9][rows1]`: per tap, the row of `conv0`'s output each cell of the second layer reads
    /// (`max_rows0`: the zero row).
    pub index1: AVec<u32>,
    /// The soft tokens, `[max_tokens][out_hidden]`: the table of the audio regime.
    pub out: AVec<u16>,
    /// `[max_rows0 + 1][c0]`, its last row zero.
    c0out: AVec<u16>,
    im2col: AVec<u16>,
    c1out: AVec<u16>,
    h: AVec<u16>,
    xn: AVec<u16>,
    xc: AVec<u16>,
    wide: AVec<u16>,
    tmp: AVec<u16>,
    q: AVec<u16>,
    /// Keys and values, `window − 1` zero rows first.
    k: AVec<u16>,
    v: AVec<u16>,
    attn: AVec<u16>,
    /// The GLU's output, `taps − 1` zero rows first.
    glu: AVec<u16>,
    /// Per layer, `softplus | relative keys [window + 1][hidden]`.
    rel: Vec<AVec<u16>>,
    proj: AVec<u16>,
    normed: AVec<u16>,
    ones: AVec<u16>,
    packed: AVec<u16>,
}

impl AudioBuffers {
    /// Everything, zeroed, for `shape` and `layers` layers.
    pub fn new(shape: &AudioShape, layers: usize) -> Self {
        let s = shape;
        let (t, h) = (s.max_tokens(), s.hidden);
        // The packed activations of the matrix engines (SME, AMX), for the widest input of each
        // program.
        let packed = [
            (s.max_rows0(), CONV0_K, s.c0),
            (s.max_rows1(), s.k1(), s.c1),
            (t, s.proj_in(), h),
            (t, h, 4 * h),
            (t, 4 * h, h),
            (t, s.out_dims, s.out_hidden),
            (s.window + 1, h, h),
        ]
        .iter()
        .map(|&(rows, k, n)| {
            let plan = crate::jit::arch::gemm_plan_bf16(GemmShape { k, n }, rows);
            if plan.opts.matrix {
                crate::jit::emit::packed_bytes(plan.opts, rows, k).div_ceil(2)
            } else {
                0
            }
        })
        .max()
        .unwrap_or(0);
        let (pw, pt) = (s.window - 1, s.taps - 1);
        // The attention's tiles read up to the end of the last one.
        let ta = t.next_multiple_of(ATTN_PAD);
        Self {
            patches0: AVec::zeroed(s.max_rows0() * CONV0_K),
            index1: AVec::zeroed(9 * s.max_rows1()),
            out: AVec::zeroed(t * s.out_hidden),
            c0out: AVec::zeroed((s.max_rows0() + 1) * s.c0),
            im2col: AVec::zeroed(s.max_rows1() * s.k1()),
            c1out: AVec::zeroed(s.max_rows1() * s.c1),
            h: AVec::zeroed(t * h),
            xn: AVec::zeroed(t * h),
            xc: AVec::zeroed(t * 4 * h),
            wide: AVec::zeroed(t * 4 * h),
            tmp: AVec::zeroed(t * h),
            q: AVec::zeroed(ta * h),
            k: AVec::zeroed((pw + ta) * h),
            v: AVec::zeroed((pw + ta) * h),
            attn: AVec::zeroed(ta * h),
            glu: AVec::zeroed((pt + t) * h),
            rel: (0..layers)
                .map(|_| AVec::zeroed(s.head_dim() + (s.window + 1) * h))
                .collect(),
            proj: AVec::zeroed(t * s.out_dims),
            normed: AVec::zeroed(t * s.out_dims),
            ones: AVec::from_vec(vec![0x3f80; (4 * h).max(s.out_dims)]),
            packed: AVec::zeroed(packed),
        }
    }

    /// Bytes reserved.
    pub fn bytes(&self) -> usize {
        let u16s = [
            &self.patches0,
            &self.out,
            &self.c0out,
            &self.im2col,
            &self.c1out,
            &self.h,
            &self.xn,
            &self.xc,
            &self.wide,
            &self.tmp,
            &self.q,
            &self.k,
            &self.v,
            &self.attn,
            &self.glu,
            &self.proj,
            &self.normed,
            &self.ones,
            &self.packed,
        ];
        u16s.iter().map(|b| b.len() * 2).sum::<usize>()
            + self.rel.iter().map(|r| r.len() * 2).sum::<usize>()
            + self.index1.len() * 4
    }
}

/// The four programs of the tower.
pub struct AudioPrograms {
    /// The relative keys of every layer (`rows` = `window + 1`), run once.
    pub(crate) relative: Mounted,
    /// The first subsampling layer (`rows` = its output cells).
    pub(crate) conv0: Mounted,
    /// The second (`rows` = its output cells).
    pub(crate) conv1: Mounted,
    /// The projection, the layers, the output and the embedder (`rows` = the soft tokens).
    pub(crate) layers: Mounted,
    /// The attention's kernel, for the mount log.
    pub(crate) attention: String,
}

struct Kernels {
    norm_h: Arc<CompiledElem>,
    norm_out: Arc<CompiledElem>,
    add_h: Arc<CompiledElem>,
    add_out: Arc<CompiledElem>,
    silu_wide: Arc<CompiledElem>,
    silu_h: Arc<CompiledElem>,
    glu: Arc<CompiledElem>,
    dw: Arc<CompiledElem>,
    attn: Arc<CompiledElem>,
    scale_h: Option<Arc<CompiledElem>>,
    clamp: Vec<(usize, Arc<CompiledElem>)>,
}

impl Kernels {
    fn clamp(&self, dim: usize) -> std::result::Result<&CompiledElem, ExecutionError> {
        self.clamp
            .iter()
            .find(|(d, _)| *d == dim)
            .map(|(_, k)| k.as_ref())
            .ok_or_else(|| err(format!("audio: no clamp of {dim}")))
    }
}

/// A linear of the tower from `x` into `y`, as the vision tower's: a clipped one clamps its input
/// into `xc` (unless `prev`, the linear before on the same input, left it there with the same
/// bounds) and its output in place.
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
                "audio-clamp-input",
                kernels.clamp(k)?,
                &[(x as u64, k * 2), (xc as u64, k * 2)],
                &[lo, hi],
                workers,
            )?;
            xc
        }
        None => x,
    };
    gemm_labeled(
        builder,
        "audio-projection",
        input,
        lin.weights,
        y,
        k,
        n,
        packed,
        workers,
    )?;
    if let Some([_, _, lo, hi]) = lin.clip {
        rows_node(
            builder,
            "audio-clamp-output",
            kernels.clamp(n)?,
            &[(y as u64, n * 2), (y as u64, n * 2)],
            &[lo, hi],
            workers,
        )?;
    }
    Ok(())
}

/// The tower's four programs over `b`, for `workers` participants (`spin`: the prefill's; `trace`:
/// the layers' program stamps its nodes). The weights and the buffers must outlive them.
pub(crate) fn lower_audio(
    shape: &AudioShape,
    w: &AudioWeights<'_>,
    b: &mut AudioBuffers,
    workers: usize,
    spin: Option<u32>,
    trace: bool,
) -> std::result::Result<AudioPrograms, ExecutionError> {
    let s = *shape;
    let (h, hd, od) = (s.hidden, s.head_dim(), s.out_dims);
    if s.heads == 0
        || h % s.heads != 0
        || hd % 32 != 0
        || s.window < 2
        || s.window > 16
        || s.taps < 1
        || s.taps > 16
        || !s.mels.is_multiple_of(4)
    {
        return Err(err("audio: a tower geometry this engine does not serve"));
    }
    if w.layers.len() != b.rel.len() || w.positions.len() < (s.window + 1) * h {
        return Err(err("audio: weights and buffers of different towers"));
    }
    let fold = s.residual > 0.0 && s.residual.log2().fract() == 0.0;
    let attn_rows = attention_rows()?;
    let mut clamp = Vec::new();
    for d in [h, 2 * h, 4 * h, s.c0, s.c1] {
        if !clamp.iter().any(|(e, _)| *e == d) {
            clamp.push((d, elem(ElemOp::Clamp, d)?));
        }
    }
    let kernels = Kernels {
        norm_h: elem(ElemOp::RmsNorm, h)?,
        norm_out: elem(ElemOp::RmsNorm, od)?,
        add_h: elem(ElemOp::ResidualAdd, h)?,
        add_out: elem(ElemOp::ResidualAdd, od)?,
        silu_wide: elem(ElemOp::SwiGlu, 4 * h)?,
        silu_h: elem(ElemOp::SwiGlu, h)?,
        glu: elem(ElemOp::SigmoidMul, h)?,
        dw: elem(ElemOp::DepthwiseRows(s.taps as u8), h)?,
        attn: elem(
            ElemOp::LocalAttention {
                heads: s.heads as u8,
                window: s.window as u8,
                rows: attn_rows as u8,
            },
            hd,
        )?,
        scale_h: if fold {
            None
        } else {
            Some(elem(ElemOp::Scale, h)?)
        },
        clamp,
    };
    let p = |v: &mut AVec<u16>| v.as_mut_ptr();
    let packed = Buffer {
        ptr: b.packed.as_mut_ptr(),
        len: b.packed.len(),
    };
    let (h2, od2) = (h * 2, od * 2);

    // ── The relative keys ───────────────────────────────────────────────────
    let mut builder = Builder::new(s.window + 1)?;
    for (l, lw) in w.layers.iter().enumerate() {
        let rel = b.rel[l].as_mut_ptr();
        // SAFETY: the buffer holds `hd + (window + 1)·hidden` values.
        gemm_labeled(
            &mut builder,
            "audio-relative-keys",
            w.positions.as_ptr().cast_mut(),
            lw.rel_k,
            unsafe { rel.add(hd) },
            h,
            h,
            packed,
            workers,
        )?;
    }
    let relative = builder.finish_with_instrumentation(1, spin, false, false, false, false)?;
    for (l, lw) in w.layers.iter().enumerate() {
        if lw.softplus.len() != hd {
            return Err(err("audio: a per_dim_scale of another width"));
        }
        b.rel[l][..hd].copy_from_slice(lw.softplus);
    }

    // ── The first subsampling layer ─────────────────────────────────────────
    let c0out = p(&mut b.c0out);
    let mut builder = Builder::new(s.max_rows0())?;
    gemm_labeled(
        &mut builder,
        "audio-conv0",
        p(&mut b.patches0),
        w.conv0,
        c0out,
        CONV0_K,
        s.c0,
        packed,
        workers,
    )?;
    rows_node(
        &mut builder,
        "audio-conv0-norm",
        elem(ElemOp::LayerNorm, s.c0)?.as_ref(),
        &[
            (c0out as u64, s.c0 * 2),
            (w.norm0.as_ptr() as u64, 0),
            (c0out as u64, s.c0 * 2),
        ],
        &[s.eps],
        workers,
    )?;
    rows_node(
        &mut builder,
        "audio-conv0-relu",
        kernels.clamp(s.c0)?,
        &[(c0out as u64, s.c0 * 2), (c0out as u64, s.c0 * 2)],
        &[0.0, f32::MAX],
        workers,
    )?;
    let conv0 = builder.finish_with_instrumentation(1, spin, false, false, false, false)?;

    // ── The second ──────────────────────────────────────────────────────────
    let (rows1, k1) = (s.max_rows1(), s.k1());
    let (im2col, c1out) = (p(&mut b.im2col), p(&mut b.c1out));
    let mut builder = Builder::new(rows1)?;
    for tap in 0..9 {
        let index = unsafe { b.index1.as_ptr().add(tap * rows1) };
        builder.label("audio-conv1-gather");
        builder.rows(
            Native::embedding()?,
            binding(
                &[
                    (index as u64, 4),
                    (c0out as u64, 0),
                    (unsafe { im2col.add(tap * s.c0) } as u64, k1 * 2),
                    (s.c0 as u64, 0),
                ],
                &[],
            ),
            workers,
            workers,
        )?;
    }
    gemm_labeled(
        &mut builder,
        "audio-conv1",
        im2col,
        w.conv1,
        c1out,
        k1,
        s.c1,
        packed,
        workers,
    )?;
    rows_node(
        &mut builder,
        "audio-conv1-norm",
        elem(ElemOp::LayerNorm, s.c1)?.as_ref(),
        &[
            (c1out as u64, s.c1 * 2),
            (w.norm1.as_ptr() as u64, 0),
            (c1out as u64, s.c1 * 2),
        ],
        &[s.eps],
        workers,
    )?;
    rows_node(
        &mut builder,
        "audio-conv1-relu",
        kernels.clamp(s.c1)?,
        &[(c1out as u64, s.c1 * 2), (c1out as u64, s.c1 * 2)],
        &[0.0, f32::MAX],
        workers,
    )?;
    let conv1 = builder.finish_with_instrumentation(1, spin, false, false, false, false)?;

    // ── The layers ──────────────────────────────────────────────────────────
    let (hb, xn, xc, wide, tmp, q, attn) = (
        p(&mut b.h),
        p(&mut b.xn),
        p(&mut b.xc),
        p(&mut b.wide),
        p(&mut b.tmp),
        p(&mut b.q),
        p(&mut b.attn),
    );
    // SAFETY: the keys, the values and the GLU's output keep their leading zero rows.
    let (k, v, glu) = unsafe {
        (
            b.k.as_mut_ptr().add((s.window - 1) * h),
            b.v.as_mut_ptr().add((s.window - 1) * h),
            b.glu.as_mut_ptr().add((s.taps - 1) * h),
        )
    };
    let glu_first = b.glu.as_mut_ptr();
    let ones = b.ones.as_ptr();
    let mut builder = Builder::new(s.max_tokens())?;
    gemm_labeled(
        &mut builder,
        "audio-input-projection",
        c1out,
        w.input_proj,
        hb,
        s.proj_in(),
        h,
        packed,
        workers,
    )?;
    let norm = |builder: &mut Builder,
                x: *mut u16,
                weight: &[u16],
                y: *mut u16|
     -> std::result::Result<(), ExecutionError> {
        rows_node(
            builder,
            "audio-norm",
            &kernels.norm_h,
            &[(x as u64, h2), (weight.as_ptr() as u64, 0), (y as u64, h2)],
            &[s.eps],
            workers,
        )
    };
    let add = |builder: &mut Builder,
               a: *mut u16,
               x: *mut u16|
     -> std::result::Result<(), ExecutionError> {
        rows_node(
            builder,
            "audio-residual",
            &kernels.add_h,
            &[(a as u64, h2), (x as u64, h2)],
            &[],
            workers,
        )
    };
    let q_scale = ((hd as f64).powf(-0.5) / std::f64::consts::LN_2) as f32;
    let k_scale = ((1.0 + std::f64::consts::E).ln() / std::f64::consts::LN_2) as f32;
    for (l, lw) in w.layers.iter().enumerate() {
        builder.begin_layer(l);
        for f in 0..2 {
            let (pre, post, l1, l2) = &lw.ff[f];
            norm(&mut builder, hb, pre, xn)?;
            linear(
                &mut builder,
                &kernels,
                l1,
                None,
                xn,
                wide,
                xc,
                packed,
                workers,
            )?;
            rows_node(
                &mut builder,
                "audio-silu",
                &kernels.silu_wide,
                &[(wide as u64, 8 * h), (ones as u64, 0), (wide as u64, 8 * h)],
                &[],
                workers,
            )?;
            linear(
                &mut builder,
                &kernels,
                l2,
                None,
                wide,
                tmp,
                xc,
                packed,
                workers,
            )?;
            norm(&mut builder, tmp, post, xn)?;
            if let Some(scale) = &kernels.scale_h {
                rows_node(
                    &mut builder,
                    "audio-half-step",
                    scale,
                    &[(xn as u64, h2), (xn as u64, h2)],
                    &[s.residual],
                    workers,
                )?;
            }
            add(&mut builder, hb, xn)?;
            if f == 1 {
                break;
            }
            // The attention.
            norm(&mut builder, hb, lw.norm_pre_attn, xn)?;
            let [lq, lk, lv, lpost] = &lw.attn;
            linear(&mut builder, &kernels, lq, None, xn, q, xc, packed, workers)?;
            linear(
                &mut builder,
                &kernels,
                lk,
                Some(lq),
                xn,
                k,
                xc,
                packed,
                workers,
            )?;
            linear(
                &mut builder,
                &kernels,
                lv,
                Some(lk),
                xn,
                v,
                xc,
                packed,
                workers,
            )?;
            if attn_rows == 0 {
                rows_node(
                    &mut builder,
                    "audio-attention",
                    &kernels.attn,
                    &[
                        (q as u64, h2),
                        (k as u64, h2),
                        (v as u64, h2),
                        (b.rel[l].as_ptr() as u64, 0),
                        (attn as u64, h2),
                        (0, 1),
                    ],
                    &[q_scale, k_scale, s.cap],
                    workers,
                )?;
            } else {
                // One task per (tile, head): `task = tile · heads + head`.
                let call = builder.keep(LocalAttentionCall {
                    q,
                    k,
                    v,
                    rel: b.rel[l].as_ptr(),
                    y: attn,
                    rows: &raw const builder.frame.rows,
                });
                builder.label("audio-attention-block");
                builder.grid(
                    Native::elementary(&kernels.attn)?,
                    binding(&[(call as u64, 0), (0, 1)], &[q_scale, k_scale, s.cap]),
                    TaskGrid {
                        rows_per_task: attn_rows,
                        tasks_per_tile: s.heads,
                        last_block_only: false,
                    },
                    workers,
                    workers,
                )?;
            }
            linear(
                &mut builder,
                &kernels,
                lpost,
                None,
                attn,
                tmp,
                xc,
                packed,
                workers,
            )?;
            norm(&mut builder, tmp, lw.norm_post_attn, xn)?;
            add(&mut builder, hb, xn)?;
            // The light convolution.
            norm(&mut builder, hb, lw.conv_pre, xn)?;
            linear(
                &mut builder,
                &kernels,
                &lw.conv_start,
                None,
                xn,
                wide,
                xc,
                packed,
                workers,
            )?;
            // SAFETY: `wide` holds `[rows][2·hidden]` here.
            rows_node(
                &mut builder,
                "audio-glu",
                &kernels.glu,
                &[
                    (wide as u64, 2 * h2),
                    (unsafe { wide.add(h) } as u64, 2 * h2),
                    (glu as u64, h2),
                ],
                &[],
                workers,
            )?;
            rows_node(
                &mut builder,
                "audio-depthwise",
                &kernels.dw,
                &[
                    (glu_first as u64, h2),
                    (lw.conv_dw.as_ptr() as u64, 0),
                    (tmp as u64, h2),
                ],
                &[],
                workers,
            )?;
            norm(&mut builder, tmp, lw.conv_norm, xn)?;
            rows_node(
                &mut builder,
                "audio-silu",
                &kernels.silu_h,
                &[(xn as u64, h2), (ones as u64, 0), (xn as u64, h2)],
                &[],
                workers,
            )?;
            linear(
                &mut builder,
                &kernels,
                &lw.conv_end,
                None,
                xn,
                tmp,
                xc,
                packed,
                workers,
            )?;
            add(&mut builder, hb, tmp)?;
        }
        norm(&mut builder, hb, lw.norm_out, hb)?;
        builder.layer_end(l);
    }
    let (proj, normed) = (p(&mut b.proj), p(&mut b.normed));
    gemm_labeled(
        &mut builder,
        "audio-output-projection",
        hb,
        w.out_proj,
        proj,
        h,
        od,
        packed,
        workers,
    )?;
    rows_node(
        &mut builder,
        "audio-output-bias",
        &kernels.add_out,
        &[(proj as u64, od2), (w.out_bias.as_ptr() as u64, 0)],
        &[],
        workers,
    )?;
    rows_node(
        &mut builder,
        "audio-embed-norm",
        &kernels.norm_out,
        &[(proj as u64, od2), (ones as u64, 0), (normed as u64, od2)],
        &[s.eps],
        workers,
    )?;
    gemm_labeled(
        &mut builder,
        "audio-embed-projection",
        normed,
        w.embed_proj,
        p(&mut b.out),
        od,
        s.out_hidden,
        packed,
        workers,
    )?;
    let layers = builder.finish_with_instrumentation(1, spin, trace, false, false, false)?;
    let isa = match crate::jit::elem::select_isa() {
        // The row kernel is the same VEX ymm blob on both x86 arms; the block kernel has a zmm arm.
        Some(crate::jit::elem::ElemIsa::Zmm) if attn_rows == 0 => {
            "x86-64 AVX-512, the VEX ymm kernel"
        }
        Some(crate::jit::elem::ElemIsa::Zmm) => "x86-64 AVX-512, zmm",
        Some(crate::jit::elem::ElemIsa::Ymm) => "x86-64 AVX2",
        Some(crate::jit::elem::ElemIsa::Neon) => "AArch64 NEON",
        _ => "?",
    };
    let attention = if attn_rows == 0 {
        format!(
            "row kernel (local_attention, one row × {} heads per task, {isa})",
            s.heads
        )
    } else {
        format!("block kernel (local_attention_block, tiles of {attn_rows} rows × one head per task, {isa})")
    };
    Ok(AudioPrograms {
        relative,
        conv0,
        conv1,
        layers,
        attention,
    })
}

/// The buffers of 12B's encoder-free audio embedder, for `rows` soft tokens: the host writes
/// `chunks`; `out` holds the soft tokens after a call.
pub struct ChunkBuffers {
    /// `[rows][samples]`, the raw samples rounded to bf16, the last chunk zero-padded.
    pub chunks: AVec<u16>,
    /// The soft tokens, `[rows][out_hidden]`: the table of the audio regime.
    pub out: AVec<u16>,
    normed: AVec<u16>,
    ones: AVec<u16>,
    packed: AVec<u16>,
}

impl ChunkBuffers {
    /// Everything, zeroed.
    pub fn new(samples: usize, out_hidden: usize, rows: usize) -> Self {
        let plan = crate::jit::arch::gemm_plan_bf16(
            GemmShape {
                k: samples,
                n: out_hidden,
            },
            rows,
        );
        let packed = if plan.opts.matrix {
            crate::jit::emit::packed_bytes(plan.opts, rows, samples).div_ceil(2)
        } else {
            0
        };
        Self {
            chunks: AVec::zeroed(rows * samples),
            out: AVec::zeroed(rows * out_hidden),
            normed: AVec::zeroed(rows * samples),
            ones: AVec::from_vec(vec![0x3f80; samples]),
            packed: AVec::zeroed(packed),
        }
    }
}

/// 12B's encoder-free audio embedder (`gemma4_unified_audio`) lowered: per soft token, its chunk of
/// `samples` raw samples through the norm without weight and `embed_audio`'s projection (`proj`,
/// packed) into `b.out`.
pub(crate) fn lower_chunks(
    samples: usize,
    out_hidden: usize,
    eps: f32,
    proj: &[u16],
    b: &mut ChunkBuffers,
    workers: usize,
    spin: Option<u32>,
) -> std::result::Result<Mounted, ExecutionError> {
    let rows = b.chunks.len() / samples;
    let d2 = samples * 2;
    let mut builder = Builder::new(rows)?;
    let (chunks, normed) = (b.chunks.as_mut_ptr(), b.normed.as_mut_ptr());
    rows_node(
        &mut builder,
        "audio-chunk-norm",
        elem(ElemOp::RmsNorm, samples)?.as_ref(),
        &[
            (chunks as u64, d2),
            (b.ones.as_ptr() as u64, 0),
            (normed as u64, d2),
        ],
        &[eps],
        workers,
    )?;
    let packed = Buffer {
        ptr: b.packed.as_mut_ptr(),
        len: b.packed.len(),
    };
    gemm_labeled(
        &mut builder,
        "audio-embed-projection",
        normed,
        proj,
        b.out.as_mut_ptr(),
        samples,
        out_hidden,
        packed,
        workers,
    )?;
    builder.finish_with_instrumentation(1, spin, false, false, false, false)
}
