//! The encoders of the embedding and reranking models, each lowered into one
//! generated program that takes a whole input as one block of rows: every row
//! sees every other (bidirectional attention), so the input is never cut into
//! passes and there is no cache to keep between two inputs.
//!
//! - EmbeddingGemma 2 (`modeling_embedding_gemma2`): the Gemma 4 layer of `lowering.rs` (norms, q/k
//!   norms and RoPE, value norm, attention scaled by one, GeLU MLP, the per-layer input branch, the
//!   layer scalar) on its geometry — 4 query heads over 2 K/V heads of 256 on the sliding layers,
//!   over 1 of 512 on the full ones — with attention that sees the keys on both sides: every key on
//!   a full layer, `|q − k| ≤ window` on a sliding one (`AttentionSpec::ahead`). The per-layer
//!   inputs are the projection of the input embeddings alone, scaled by `hidden^-½` and normed per
//!   layer ("projection-only PLE"). The program ends on the final norm of every row; the host takes
//!   their mean and projects it (`embedding_projection`).
//!
//! Every computation is an emitted kernel of this crate; the projections are
//! the text's GEMMs (`projections`: bf16, or int8 / Q4 under `--int8`,
//! `--q4`).

use super::lowering::{
    binary, binding, norm, projections, projections_f32, Buffer, F32Buffer, Pack, Projection,
    ProjectionF32,
};
use crate::core::avec::AVec;
use crate::jit::arch::{KernelSet, Native};
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

fn elem(op: ElemOp, dim: usize) -> std::result::Result<Arc<CompiledElem>, ExecutionError> {
    crate::jit::elem::get_or_compile(ElemSpec { op, dim })
        .map_err(err)?
        .ok_or_else(|| err(format!("encoder: {} dim={dim} not schedulable", op.name())))
}

fn buffer(v: &mut AVec<u16>) -> Buffer {
    Buffer {
        ptr: v.as_mut_ptr(),
        len: v.len(),
    }
}

/// A row-wise elementwise node on the whole team: one task per row.
fn rows_node(
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

/// The attention kernels of one head size: the row kernel (on x86-64 with AVX-512, its QK by groups
/// of 16 when the emitter takes it), and the block kernel without the causal mask (x86-64 with
/// AVX-512 or AVX2, and AArch64, `None` elsewhere); with `window`, the block of `|q − k| < window`
/// too (`block_window`, a sliding layer of a long input).
struct HeadKernels {
    head_dim: usize,
    norm: Arc<CompiledElem>,
    rope: Arc<CompiledElem>,
    sdpa: Arc<crate::jit::sdpa::CompiledSdpa>,
    block: Option<Arc<crate::jit::sdpa::CompiledSdpaPrefill>>,
    block_window: Option<Arc<crate::jit::sdpa::CompiledSdpaPrefill>>,
}

/// The cache of an encoder's attention: its positions per KV head (`rows`, rounded up to sixteen on
/// AArch64), the R16 copies of the AArch64 block kernel when it has them.
fn kv_layout(rows: usize) -> (usize, bool) {
    crate::jit::sdpa::full_kv_capacity(rows)
}

fn head_kernels(
    hd: usize,
    q_row: usize,
    rows: usize,
    window: usize,
) -> std::result::Result<HeadKernels, ExecutionError> {
    let shape = SdpaShape {
        head_dim: hd,
        sink: false,
    };
    let base = crate::jit::sdpa::head_kernel(shape)
        .map_err(err)?
        .ok_or_else(|| err(format!("encoder: no attention of head_dim {hd}")))?;
    let sdpa = if cfg!(target_arch = "x86_64")
        && crate::jit::gemm::is_supported()
        && base.spec().form != crate::jit::ir::SdpaForm::Fused
    {
        let spec = crate::jit::emit_sdpa::SdpaKernelSpec {
            qk_group: 16,
            ..base.spec()
        };
        crate::jit::sdpa::get_or_compile(spec)
            .map_err(err)?
            .unwrap_or(base)
    } else {
        base
    };
    // `HERBERT_ENCODER_ATTN_ROWS=N`: queries per block of the attention (0: the row kernel alone).
    let block_rows = std::env::var("HERBERT_ENCODER_ATTN_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(crate::jit::emit_sdpa::SDPA_PREFILL_MAX_ROWS);
    let capacity = kv_layout(rows).0;
    let full = |window| {
        crate::jit::sdpa::prefill_kernel_full(
            shape,
            block_rows,
            q_row,
            q_row,
            capacity,
            sdpa.spec().qk_group,
            window,
        )
        .map_err(err)
    };
    let block = full(0)?;
    let block_window = if window > 0 { full(window)? } else { None };
    Ok(HeadKernels {
        head_dim: hd,
        norm: elem(ElemOp::RmsNorm, hd)?,
        rope: elem(ElemOp::Rope, hd)?,
        sdpa,
        block,
        block_window,
    })
}

/// Bytes of the activations a projection of `k` inputs reads packed or quantized, at `rows` rows.
fn packed_len(rows: usize, ks: &[usize], ns: &[usize]) -> usize {
    let mut most = 0;
    for &k in ks {
        for &n in ns {
            let plan = crate::jit::arch::gemm_plan(GemmShape { k, n }, rows);
            if plan.opts.matrix {
                most = most.max(crate::jit::emit::packed_bytes(plan.opts, rows, k));
            }
            if crate::jit::int8::gemm(n, k) {
                most = most.max(crate::jit::int8::a_bytes(rows, k));
            }
        }
    }
    most.div_ceil(2)
}

// ── EmbeddingGemma 2 ─────────────────────────────────────────────────────────

/// The geometry of an EmbeddingGemma 2 encoder.
#[derive(Clone, Debug)]
pub struct GemmaEncoderShape {
    /// The width of the rows.
    pub hidden: usize,
    /// Query heads.
    pub heads: usize,
    /// The MLP's width.
    pub intermediate: usize,
    /// The width of one layer's per-layer input.
    pub per_layer: usize,
    /// Per layer: its head size, its K/V heads, whether it slides.
    pub layers: Vec<(usize, usize, bool)>,
    /// A sliding layer's keys: `|q − k| ≤ window`.
    pub window: usize,
    /// The RMS norms' epsilon.
    pub eps: f32,
    /// The rows of one input at most.
    pub rows: usize,
}

impl GemmaEncoderShape {
    fn q_width(&self) -> usize {
        self.layers
            .iter()
            .map(|l| self.heads * l.0)
            .max()
            .unwrap_or(0)
    }
    fn kv_width(&self) -> usize {
        self.layers.iter().map(|l| l.0 * l.1).max().unwrap_or(0)
    }
    /// The head sizes, each once: the sliding layers' and the full layers'.
    fn head_dims(&self) -> Vec<(usize, bool)> {
        let mut v: Vec<(usize, bool)> = Vec::new();
        for &(hd, _, sliding) in &self.layers {
            if !v.iter().any(|&(d, _)| d == hd) {
                v.push((hd, sliding));
            }
        }
        v
    }
}

/// One layer's weights: norms in bf16, projections packed as the text's GEMMs read them.
pub struct GemmaEncoderLayer<'a> {
    /// `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`,
    /// `post_feedforward_layernorm`, `ple_block.post_per_layer_input_norm`.
    pub norms: [&'a [u16]; 5],
    /// The per-head norm of the queries.
    pub q_norm: &'a [u16],
    /// The per-head norm of the keys.
    pub k_norm: &'a [u16],
    /// `q, k, v, o, gate, up, down, ple gate, ple projection`.
    pub linears: [&'a [u16]; 9],
    /// `layer_scalar`.
    pub scalar: f32,
}

/// The encoder's weights.
pub struct GemmaEncoderWeights<'a> {
    /// `ple.per_layer_model_projection`, `layers · per_layer × hidden`, packed.
    pub ple_model: &'a [u16],
    /// `ple.per_layer_projection_norm`.
    pub ple_norm: &'a [u16],
    /// The final norm.
    pub final_norm: &'a [u16],
    /// The layers.
    pub layers: Vec<GemmaEncoderLayer<'a>>,
}

/// The encoder's buffers, for its largest input. The host writes `input` (the input embeddings) and
/// the RoPE tables (`cos`, `sin`: one pair per head size, `[rows][hd / 2]`, position `r` at row
/// `r`) before a call; `out` holds the rows after the final norm.
pub struct GemmaEncoderBuffers {
    /// `[rows][hidden]`: the input embeddings (`x₀`).
    pub input: AVec<u16>,
    /// `[rows][hidden]`: the last layer's rows after the final norm.
    pub out: AVec<u16>,
    /// The RoPE tables of each head size, in the order of `GemmaEncoderShape::head_dims`.
    pub tables: Vec<(usize, AVec<u16>, AVec<u16>)>,
    /// `[rows][2]`: the first and the end row of each row's input, read by the program without the
    /// window (several inputs end to end in one block, `AttentionSpec::segments`).
    pub segments: AVec<u32>,
    h: AVec<u16>,
    xn: AVec<u16>,
    q: AVec<u16>,
    k: AVec<u16>,
    v: AVec<u16>,
    attn: AVec<u16>,
    o: AVec<u16>,
    gate: AVec<u16>,
    up: AVec<u16>,
    fold: AVec<u16>,
    gate_pl: AVec<u16>,
    keys: AVec<u16>,
    values: AVec<u16>,
    /// The R16 copies of `keys` and `values` (the AArch64 block attention), empty without it.
    keys_r16: AVec<u16>,
    values_r16: AVec<u16>,
    scores: AVec<f32>,
    packed: AVec<u16>,
    ones: AVec<u16>,
    scores_stride: usize,
}

impl GemmaEncoderBuffers {
    /// Everything, zeroed, for `s` and `workers` participants.
    pub fn new(s: &GemmaEncoderShape, workers: usize) -> Self {
        let r = s.rows;
        let (h, qw, kw, inter, pl) = (
            s.hidden,
            s.q_width(),
            s.kv_width(),
            s.intermediate,
            s.per_layer,
        );
        let lpl = s.layers.len() * pl;
        let packed = packed_len(r, &[h, qw, inter, pl], &[h, qw, kw, inter, pl, lpl]);
        let scores_stride =
            crate::jit::sdpa::full_scratch_f32(r, s.layers.iter().map(|l| l.0).max().unwrap_or(0));
        let (cap, r16) = kv_layout(r);
        let tables = s
            .head_dims()
            .iter()
            .map(|&(hd, _)| (hd, AVec::zeroed(r * hd / 2), AVec::zeroed(r * hd / 2)))
            .collect();
        Self {
            input: AVec::zeroed(r * h),
            out: AVec::zeroed(r * h),
            tables,
            segments: AVec::zeroed(2 * r),
            h: AVec::zeroed(r * h),
            xn: AVec::zeroed(r * h),
            q: AVec::zeroed(r * qw),
            k: AVec::zeroed(r * kw),
            v: AVec::zeroed(r * kw),
            attn: AVec::zeroed(r * qw),
            o: AVec::zeroed(r * h),
            gate: AVec::zeroed(r * inter),
            up: AVec::zeroed(r * inter),
            fold: AVec::zeroed(r * lpl),
            gate_pl: AVec::zeroed(r * pl),
            keys: AVec::zeroed(cap * kw),
            values: AVec::zeroed(cap * kw),
            keys_r16: AVec::zeroed(if r16 { cap * kw } else { 0 }),
            values_r16: AVec::zeroed(if r16 { cap * kw } else { 0 }),
            scores: AVec::zeroed(scores_stride * workers.max(1)),
            packed: AVec::zeroed(packed.max(1)),
            ones: AVec::from_vec(vec![0x3f80; h.max(qw)]),
            scores_stride,
        }
    }

    /// Bytes reserved.
    pub fn bytes(&self) -> usize {
        let u16s = [
            &self.input,
            &self.out,
            &self.h,
            &self.xn,
            &self.q,
            &self.k,
            &self.v,
            &self.attn,
            &self.o,
            &self.gate,
            &self.up,
            &self.fold,
            &self.gate_pl,
            &self.keys,
            &self.values,
            &self.keys_r16,
            &self.values_r16,
            &self.packed,
            &self.ones,
        ];
        u16s.iter().map(|b| b.len() * 2).sum::<usize>()
            + self
                .tables
                .iter()
                .map(|t| (t.1.len() + t.2.len()) * 2)
                .sum::<usize>()
            + self.scores.len() * 4
            + self.segments.len() * 4
    }
}

/// The encoder's program over `b` for `workers` participants. `windowed`: the sliding layers see
/// `|q − k| ≤ window` (an input longer than `window + 1` rows); without, every row sees every key
/// of its own input on every layer (one or several inputs end to end, `b.segments`, none longer
/// than the window: it hides nothing), and the block attention serves every layer where the target
/// has it.
pub(crate) fn lower_gemma_encoder(
    s: &GemmaEncoderShape,
    w: &GemmaEncoderWeights<'_>,
    b: &mut GemmaEncoderBuffers,
    workers: usize,
    spin: Option<u32>,
    windowed: bool,
    trace: bool,
) -> std::result::Result<Mounted, ExecutionError> {
    let rows = s.rows;
    let (h, nh, inter, pl) = (s.hidden, s.heads, s.intermediate, s.per_layer);
    let n_layers = s.layers.len();
    let lpl = n_layers * pl;
    if w.layers.len() != n_layers
        || s.layers
            .iter()
            .any(|&(hd, kvh, _)| hd % 4 != 0 || kvh == 0 || nh % kvh != 0)
        || w.ple_norm.len() < pl
        || w.final_norm.len() < h
    {
        return Err(err("encoder: a geometry this engine does not serve"));
    }
    let heads: Vec<HeadKernels> = s
        .head_dims()
        .iter()
        .map(|&(hd, sliding)| {
            head_kernels(
                hd,
                nh * hd,
                rows,
                if windowed && sliding { s.window + 1 } else { 0 },
            )
        })
        .collect::<std::result::Result<_, _>>()?;
    let (cap, r16) = kv_layout(rows);
    // `HERBERT_ENCODER_PLAN=1`: each layer's attention kernel on stderr when the program is
    // mounted.
    let plan = std::env::var_os("HERBERT_ENCODER_PLAN").is_some_and(|v| v != "0");
    if plan {
        eprintln!(
            "[encoder plan] {rows} rows, {}",
            if windowed {
                "windowed"
            } else {
                "every key of each input"
            }
        );
    }
    for k in &heads {
        if b.scores_stride
            < k.block
                .iter()
                .chain(&k.block_window)
                .map(|p| p.scratch_len())
                .max()
                .unwrap_or(0)
            || b.keys.len() < cap * s.kv_width()
            || (r16 && b.keys_r16.len() < cap * s.kv_width())
        {
            return Err(err(
                "encoder: the attention's scratch is larger than its slot",
            ));
        }
    }
    let norm_h = elem(ElemOp::RmsNorm, h)?;
    let norm_pl = elem(ElemOp::RmsNorm, pl)?;
    let add_h = elem(ElemOp::ResidualAdd, h)?;
    let scale_h = elem(ElemOp::Scale, h)?;
    let scale_lpl = elem(ElemOp::Scale, lpl)?;
    let gelu = elem(ElemOp::GeluTanh, inter)?;
    let gelu_pl = elem(ElemOp::GeluTanh, pl)?;
    let packed = buffer(&mut b.packed);
    let (x0, x, xn, q, k, v, attn, o, gate, up, fold, gpl, out) = (
        buffer(&mut b.input),
        buffer(&mut b.h),
        buffer(&mut b.xn),
        buffer(&mut b.q),
        buffer(&mut b.k),
        buffer(&mut b.v),
        buffer(&mut b.attn),
        buffer(&mut b.o),
        buffer(&mut b.gate),
        buffer(&mut b.up),
        buffer(&mut b.fold),
        buffer(&mut b.gate_pl),
        buffer(&mut b.out),
    );
    let team = (workers, workers);
    let h2 = h * 2;
    let mut program = Builder::new(rows)?;
    let builder = &mut program;
    // The input: `h = x₀`; the per-layer inputs `norm(hidden^-½ · W · x₀)`, one norm per layer.
    rows_node(
        builder,
        "encoder-input",
        &scale_h,
        &[(x0.ptr as u64, h2), (x.ptr as u64, h2)],
        &[1.0],
        workers,
    )?;
    projections(
        builder,
        x0,
        &[Projection {
            weights: w.ple_model,
            output: fold,
            n: lpl,
        }],
        h,
        workers,
        true,
        Pack::Into(packed),
    )?;
    rows_node(
        builder,
        "per-layer-projection-scale",
        &scale_lpl,
        &[(fold.ptr as u64, lpl * 2), (fold.ptr as u64, lpl * 2)],
        &[(h as f32).powf(-0.5)],
        workers,
    )?;
    builder.label("per-layer-projection-norm");
    builder.grid(
        Native::elementary(&norm_pl)?,
        binding(
            &[
                (fold.ptr as u64, pl * 2),
                (w.ple_norm.as_ptr() as u64, 0),
                (fold.ptr as u64, pl * 2),
            ],
            &[s.eps],
        ),
        TaskGrid {
            rows_per_task: 1,
            tasks_per_tile: n_layers,
            last_block_only: false,
        },
        workers,
        workers,
    )?;
    for (l, lw) in w.layers.iter().enumerate() {
        let (hd, kvh, sliding) = s.layers[l];
        let hk = heads
            .iter()
            .find(|k| k.head_dim == hd)
            .expect("a kernel per head size");
        let (cos, sin) = b
            .tables
            .iter()
            .find(|t| t.0 == hd)
            .map(|t| (t.1.as_ptr(), t.2.as_ptr()))
            .expect("a table per head size");
        let (qd, kvd, half) = (nh * hd, kvh * hd, hd / 2);
        if lw.q_norm.len() < hd || lw.k_norm.len() < hd {
            return Err(err("encoder: a head norm weight too short"));
        }
        builder.begin_layer(l);
        norm(builder, &norm_h, x, lw.norms[0], xn, h, s.eps, team)?;
        projections(
            builder,
            xn,
            &[
                Projection {
                    weights: lw.linears[0],
                    output: q,
                    n: qd,
                },
                Projection {
                    weights: lw.linears[1],
                    output: k,
                    n: kvd,
                },
                Projection {
                    weights: lw.linears[2],
                    output: v,
                    n: kvd,
                },
            ],
            h,
            workers,
            true,
            Pack::Into(packed),
        )?;
        for (data, weight, nheads, row) in [(q, lw.q_norm, nh, qd), (k, lw.k_norm, kvh, kvd)] {
            builder.label("head-norm-rope");
            builder.rows(
                Native::norm_rope(Some(&hk.norm), &hk.rope, nheads, hd)?,
                binding(
                    &[
                        (data.ptr as u64, row * 2),
                        (weight.as_ptr() as u64, 0),
                        (cos as u64, half * 2),
                        (sin as u64, half * 2),
                    ],
                    &[s.eps],
                ),
                workers,
                workers,
            )?;
        }
        builder.label("value-head-norm");
        builder.grid(
            Native::elementary(&hk.norm)?,
            binding(
                &[
                    (v.ptr as u64, hd * 2),
                    (b.ones.as_ptr() as u64, 0),
                    (v.ptr as u64, hd * 2),
                ],
                &[s.eps],
            ),
            TaskGrid {
                rows_per_task: 1,
                tasks_per_tile: kvh,
                last_block_only: false,
            },
            workers,
            workers,
        )?;
        // Every key on a full layer; `|q − k| ≤ window` on a sliding one when the input is long
        // enough for it to hide something (the block kernel of that window, x86-64 with
        // AVX-512 or AVX2, and AArch64).
        let window = windowed && sliding;
        let block = if window {
            hk.block_window.as_deref()
        } else {
            hk.block.as_deref()
        };
        if plan {
            match block {
                Some(p) => eprintln!(
                    "[encoder plan] layer {l} head_dim {hd} {}: block of {} rows, window {}{}",
                    if sliding { "sliding" } else { "full" },
                    p.rows(),
                    p.spec().window,
                    if p.spec().full { ", unmasked" } else { "" }
                ),
                None => eprintln!(
                    "[encoder plan] layer {l} head_dim {hd} {}: row kernel, window {}",
                    if sliding { "sliding" } else { "full" },
                    if window { s.window + 1 } else { 0 }
                ),
            }
        }
        // The R16 copies when the layer's block kernel reads them.
        let r16 = r16 && block.is_some();
        // K and V head-major, as a cache of the input's rows.
        let kv_call = builder.keep(ex::abi::KvCall {
            keys: b.keys.as_mut_ptr(),
            values: b.values.as_mut_ptr(),
            input_keys: k.ptr,
            input_values: v.ptr,
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
        append.tasks = kvh;
        builder.label("kv-append");
        builder.grid(
            Native::kv_append(ex::abi::KvSpec {
                heads: kvh,
                head_dim: hd,
                stride: cap * hd,
                capacity: cap,
                max_batch: rows,
                r16,
            })?,
            append,
            TaskGrid {
                rows_per_task: rows,
                tasks_per_tile: kvh,
                last_block_only: false,
            },
            workers,
            workers,
        )?;
        let chunk = block.map_or(1, |p| p.rows());
        let call = builder.keep(ex::abi::AttentionCall {
            q: q.ptr,
            keys: b.keys.as_ptr(),
            values: b.values.as_ptr(),
            output: attn.ptr,
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
            segments: if windowed {
                std::ptr::null()
            } else {
                b.segments.as_ptr()
            },
        });
        let mut args = binding(
            &[(call as u64, 0), (0, 1), (b.scores.as_mut_ptr() as u64, 0)],
            &[],
        );
        args.participant_strides[2] = (b.scores_stride * 4) as u64;
        if b.scores.len() < b.scores_stride * workers {
            return Err(err("encoder: attention score slots too short"));
        }
        builder.label("attention");
        builder.grid(
            Native::attention_tile_selector(
                &hk.sdpa,
                block,
                ex::abi::AttentionSpec {
                    heads: nh,
                    gqa: nh / kvh,
                    head_dim: hd,
                    q_row: qd,
                    q_head: hd,
                    out_row: qd,
                    grouped: false,
                    kv_task: false,
                    window: if window { s.window + 1 } else { 0 },
                    whole_block: true,
                    ahead: if window { s.window } else { 0 },
                    segments: !windowed,
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
        projections(
            builder,
            attn,
            &[Projection {
                weights: lw.linears[3],
                output: o,
                n: h,
            }],
            qd,
            workers,
            true,
            Pack::Into(packed),
        )?;
        norm(builder, &norm_h, o, lw.norms[1], xn, h, s.eps, team)?;
        binary(builder, &add_h, x, xn, h, team)?;
        // The MLP.
        norm(builder, &norm_h, x, lw.norms[2], xn, h, s.eps, team)?;
        projections(
            builder,
            xn,
            &[
                Projection {
                    weights: lw.linears[4],
                    output: gate,
                    n: inter,
                },
                Projection {
                    weights: lw.linears[5],
                    output: up,
                    n: inter,
                },
            ],
            h,
            workers,
            true,
            Pack::Into(packed),
        )?;
        rows_node(
            builder,
            "mlp-gelu",
            &gelu,
            &[
                (gate.ptr as u64, inter * 2),
                (up.ptr as u64, inter * 2),
                (gate.ptr as u64, inter * 2),
            ],
            &[],
            workers,
        )?;
        projections(
            builder,
            gate,
            &[Projection {
                weights: lw.linears[6],
                output: o,
                n: h,
            }],
            inter,
            workers,
            true,
            Pack::Into(packed),
        )?;
        norm(builder, &norm_h, o, lw.norms[3], xn, h, s.eps, team)?;
        binary(builder, &add_h, x, xn, h, team)?;
        // The per-layer input branch: `x += norm(proj(gelu(gate(x)) · input_l))`, then the layer
        // scalar.
        projections(
            builder,
            x,
            &[Projection {
                weights: lw.linears[7],
                output: gpl,
                n: pl,
            }],
            h,
            workers,
            true,
            Pack::Into(packed),
        )?;
        let input_l = unsafe { fold.ptr.add(l * pl) } as u64;
        rows_node(
            builder,
            "layer-branch-gelu",
            &gelu_pl,
            &[
                (gpl.ptr as u64, pl * 2),
                (input_l, lpl * 2),
                (gpl.ptr as u64, pl * 2),
            ],
            &[],
            workers,
        )?;
        projections(
            builder,
            gpl,
            &[Projection {
                weights: lw.linears[8],
                output: o,
                n: h,
            }],
            pl,
            workers,
            true,
            Pack::Into(packed),
        )?;
        norm(builder, &norm_h, o, lw.norms[4], xn, h, s.eps, team)?;
        binary(builder, &add_h, x, xn, h, team)?;
        rows_node(
            builder,
            "layer-scalar",
            &scale_h,
            &[(x.ptr as u64, h2), (x.ptr as u64, h2)],
            &[lw.scalar],
            workers,
        )?;
        builder.layer_end(l);
    }
    norm(builder, &norm_h, x, w.final_norm, out, h, s.eps, team)?;
    program.finish_with_instrumentation(1, spin, trace, false, false, false)
}

// ── XLM-RoBERTa ──────────────────────────────────────────────────────────────

/// The geometry of an XLM-RoBERTa encoder (bge-m3, bge-reranker-v2-m3).
#[derive(Clone, Copy, Debug)]
pub struct XlmrShape {
    /// The width of the rows.
    pub hidden: usize,
    /// Attention heads (as many K/V heads).
    pub heads: usize,
    /// The MLP's width.
    pub intermediate: usize,
    /// Layers.
    pub layers: usize,
    /// The LayerNorms' epsilon.
    pub eps: f32,
    /// The rows of one input at most.
    pub rows: usize,
}

/// One layer's weights: the matrices packed as the text's GEMMs read them, the biases bf16, the
/// LayerNorms bf16 as [`ElemOp::LayerNorm`] reads them (`g` then `b`).
pub struct XlmrLayer<'a> {
    /// `query`, `key`, `value`, the attention's `output.dense`, `intermediate.dense`,
    /// `output.dense`.
    pub linears: [&'a [u16]; 6],
    /// Their biases.
    pub biases: [&'a [u16]; 6],
    /// `attention.output.LayerNorm`, `output.LayerNorm`.
    pub norms: [&'a [u16]; 2],
}

/// The encoder's weights.
pub struct XlmrWeights<'a> {
    /// Row 0 of `token_type_embeddings`.
    pub token_type: &'a [u16],
    /// `embeddings.LayerNorm`.
    pub norm: &'a [u16],
    /// The layers.
    pub layers: Vec<XlmrLayer<'a>>,
}

/// The encoder's buffers, for its largest block. The host writes the rows of the word table into
/// `input`, the position rows into `positions` and the inputs' bounds into `segments` before a call
/// (one input or several end to end); `h` holds the last layer's rows after it (an input's first
/// row: its CLS).
pub struct XlmrBuffers {
    /// `[rows][hidden]`: the word embeddings of the inputs.
    pub input: AVec<u16>,
    /// `[rows][hidden]`: the position embedding of each row (its place in its own input).
    pub positions: AVec<u16>,
    /// `[rows][2]`: the first and the end row of each row's input (`AttentionSpec::segments`).
    pub segments: AVec<u32>,
    /// `[rows][hidden]`: the residual, the last layer's rows after the call.
    pub h: AVec<u16>,
    q: AVec<u16>,
    k: AVec<u16>,
    v: AVec<u16>,
    attn: AVec<u16>,
    o: AVec<u16>,
    up: AVec<u16>,
    keys: AVec<u16>,
    values: AVec<u16>,
    /// The R16 copies of `keys` and `values` (the AArch64 block attention), empty without it.
    keys_r16: AVec<u16>,
    values_r16: AVec<u16>,
    scores: AVec<f32>,
    packed: AVec<u16>,
    ones: AVec<u16>,
    scores_stride: usize,
}

impl XlmrBuffers {
    /// Everything, zeroed, for `s` and `workers` participants.
    pub fn new(s: &XlmrShape, workers: usize) -> Self {
        let (r, h, inter) = (s.rows, s.hidden, s.intermediate);
        let packed = packed_len(r, &[h, inter], &[h, inter]);
        let scores_stride = crate::jit::sdpa::full_scratch_f32(r, h / s.heads.max(1));
        let (cap, r16) = kv_layout(r);
        Self {
            input: AVec::zeroed(r * h),
            positions: AVec::zeroed(r * h),
            segments: AVec::zeroed(2 * r),
            h: AVec::zeroed(r * h),
            q: AVec::zeroed(r * h),
            k: AVec::zeroed(r * h),
            v: AVec::zeroed(r * h),
            attn: AVec::zeroed(r * h),
            o: AVec::zeroed(r * h),
            up: AVec::zeroed(r * inter),
            keys: AVec::zeroed(cap * h),
            values: AVec::zeroed(cap * h),
            keys_r16: AVec::zeroed(if r16 { cap * h } else { 0 }),
            values_r16: AVec::zeroed(if r16 { cap * h } else { 0 }),
            scores: AVec::zeroed(scores_stride * workers.max(1)),
            packed: AVec::zeroed(packed.max(1)),
            ones: AVec::from_vec(vec![0x3f80; inter]),
            scores_stride,
        }
    }

    /// Bytes reserved.
    pub fn bytes(&self) -> usize {
        let u16s = [
            &self.input,
            &self.positions,
            &self.h,
            &self.q,
            &self.k,
            &self.v,
            &self.attn,
            &self.o,
            &self.up,
            &self.keys,
            &self.values,
            &self.keys_r16,
            &self.values_r16,
            &self.packed,
            &self.ones,
        ];
        u16s.iter().map(|b| b.len() * 2).sum::<usize>()
            + self.scores.len() * 4
            + self.segments.len() * 4
    }
}

/// The XLM-RoBERTa encoder's program over `b` for `workers` participants (a block of one input or
/// of several end to end, each row attending to its own input's: `b.segments`), post-norm as
/// `modeling_xlm_roberta`: `h = LN(word + position + type)`, then per layer `h = LN(o(attn(q, k,
/// v)) + b_o + h)`, `h = LN(down(gelu(up(h) + b_up)) + b_down + h)`; the attention bidirectional
/// over the whole input, scaled by `1/√hd`, the GeLU exact ([`ElemOp::GeluErf`]).
pub(crate) fn lower_xlmr(
    s: &XlmrShape,
    w: &XlmrWeights<'_>,
    b: &mut XlmrBuffers,
    workers: usize,
    spin: Option<u32>,
    trace: bool,
) -> std::result::Result<Mounted, ExecutionError> {
    let (rows, h, nh, inter) = (s.rows, s.hidden, s.heads, s.intermediate);
    let hd = h / nh;
    if h % nh != 0
        || hd % 4 != 0
        || w.layers.len() != s.layers
        || b.positions.len() < rows * h
        || b.segments.len() < 2 * rows
        || w.token_type.len() < h
        || w.norm.len() < 2 * h
    {
        return Err(err("xlmr: a geometry this engine does not serve"));
    }
    let hk = head_kernels(hd, h, rows, 0)?;
    let (cap, r16) = kv_layout(rows);
    let r16 = r16 && hk.block.is_some();
    if b.scores_stride < hk.block.as_ref().map_or(0, |p| p.scratch_len())
        || b.scores.len() < b.scores_stride * workers
        || b.keys.len() < cap * h
        || (r16 && b.keys_r16.len() < cap * h)
    {
        return Err(err("xlmr: the attention's scratch is larger than its slot"));
    }
    let ln = elem(ElemOp::LayerNorm, h)?;
    let add_h = elem(ElemOp::ResidualAdd, h)?;
    let add_inter = elem(ElemOp::ResidualAdd, inter)?;
    let gelu = elem(ElemOp::GeluErf, inter)?;
    let packed = buffer(&mut b.packed);
    let (x0, x, q, k, v, attn, o, up) = (
        buffer(&mut b.input),
        buffer(&mut b.h),
        buffer(&mut b.q),
        buffer(&mut b.k),
        buffer(&mut b.v),
        buffer(&mut b.attn),
        buffer(&mut b.o),
        buffer(&mut b.up),
    );
    let h2 = h * 2;
    let mut program = Builder::new(rows)?;
    let builder = &mut program;
    // `x₀ += position; x₀ += type; h = LN(x₀)`.
    rows_node(
        builder,
        "embedding-position",
        &add_h,
        &[(x0.ptr as u64, h2), (b.positions.as_ptr() as u64, h2)],
        &[],
        workers,
    )?;
    rows_node(
        builder,
        "embedding-type",
        &add_h,
        &[(x0.ptr as u64, h2), (w.token_type.as_ptr() as u64, 0)],
        &[],
        workers,
    )?;
    rows_node(
        builder,
        "embedding-norm",
        &ln,
        &[
            (x0.ptr as u64, h2),
            (w.norm.as_ptr() as u64, 0),
            (x.ptr as u64, h2),
        ],
        &[s.eps],
        workers,
    )?;
    for (l, lw) in w.layers.iter().enumerate() {
        builder.begin_layer(l);
        projections(
            builder,
            x,
            &[
                Projection {
                    weights: lw.linears[0],
                    output: q,
                    n: h,
                },
                Projection {
                    weights: lw.linears[1],
                    output: k,
                    n: h,
                },
                Projection {
                    weights: lw.linears[2],
                    output: v,
                    n: h,
                },
            ],
            h,
            workers,
            true,
            Pack::Into(packed),
        )?;
        for (data, bias) in [(q, lw.biases[0]), (k, lw.biases[1]), (v, lw.biases[2])] {
            rows_node(
                builder,
                "bias",
                &add_h,
                &[(data.ptr as u64, h2), (bias.as_ptr() as u64, 0)],
                &[],
                workers,
            )?;
        }
        let kv_call = builder.keep(ex::abi::KvCall {
            keys: b.keys.as_mut_ptr(),
            values: b.values.as_mut_ptr(),
            input_keys: k.ptr,
            input_values: v.ptr,
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
        builder.label("kv-append");
        builder.grid(
            Native::kv_append(ex::abi::KvSpec {
                heads: nh,
                head_dim: hd,
                stride: cap * hd,
                capacity: cap,
                max_batch: rows,
                r16,
            })?,
            append,
            TaskGrid {
                rows_per_task: rows,
                tasks_per_tile: nh,
                last_block_only: false,
            },
            workers,
            workers,
        )?;
        let block = hk.block.as_deref();
        // `HERBERT_ENCODER_PLAN=1`, as for EmbeddingGemma 2: the layer's attention kernel when the
        // program is mounted.
        if std::env::var_os("HERBERT_ENCODER_PLAN").is_some_and(|v| v != "0") {
            match block {
                Some(p) => eprintln!(
                    "[encoder plan] xlmr {rows} rows, layer {l} head_dim {hd}: block of {} rows{}",
                    p.rows(),
                    if p.spec().full { ", unmasked" } else { "" }
                ),
                None => eprintln!(
                    "[encoder plan] xlmr {rows} rows, layer {l} head_dim {hd}: row kernel"
                ),
            }
        }
        let call = builder.keep(ex::abi::AttentionCall {
            q: q.ptr,
            keys: b.keys.as_ptr(),
            values: b.values.as_ptr(),
            output: attn.ptr,
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
            segments: b.segments.as_ptr(),
        });
        let mut args = binding(
            &[(call as u64, 0), (0, 1), (b.scores.as_mut_ptr() as u64, 0)],
            &[],
        );
        args.participant_strides[2] = (b.scores_stride * 4) as u64;
        builder.label("attention");
        builder.grid(
            Native::attention_tile_selector(
                &hk.sdpa,
                block,
                ex::abi::AttentionSpec {
                    heads: nh,
                    gqa: 1,
                    head_dim: hd,
                    q_row: h,
                    q_head: hd,
                    out_row: h,
                    grouped: false,
                    kv_task: false,
                    window: 0,
                    whole_block: true,
                    ahead: 0,
                    segments: true,
                },
            )?,
            args,
            TaskGrid {
                rows_per_task: block.map_or(1, |p| p.rows()),
                tasks_per_tile: nh,
                last_block_only: false,
            },
            workers,
            workers,
        )?;
        projections(
            builder,
            attn,
            &[Projection {
                weights: lw.linears[3],
                output: o,
                n: h,
            }],
            h,
            workers,
            true,
            Pack::Into(packed),
        )?;
        rows_node(
            builder,
            "bias",
            &add_h,
            &[(o.ptr as u64, h2), (lw.biases[3].as_ptr() as u64, 0)],
            &[],
            workers,
        )?;
        rows_node(
            builder,
            "residual",
            &add_h,
            &[(o.ptr as u64, h2), (x.ptr as u64, h2)],
            &[],
            workers,
        )?;
        rows_node(
            builder,
            "layer-norm",
            &ln,
            &[
                (o.ptr as u64, h2),
                (lw.norms[0].as_ptr() as u64, 0),
                (x.ptr as u64, h2),
            ],
            &[s.eps],
            workers,
        )?;
        // The MLP.
        projections(
            builder,
            x,
            &[Projection {
                weights: lw.linears[4],
                output: up,
                n: inter,
            }],
            h,
            workers,
            true,
            Pack::Into(packed),
        )?;
        rows_node(
            builder,
            "bias",
            &add_inter,
            &[
                (up.ptr as u64, inter * 2),
                (lw.biases[4].as_ptr() as u64, 0),
            ],
            &[],
            workers,
        )?;
        rows_node(
            builder,
            "mlp-gelu",
            &gelu,
            &[
                (up.ptr as u64, inter * 2),
                (b.ones.as_ptr() as u64, 0),
                (up.ptr as u64, inter * 2),
            ],
            &[],
            workers,
        )?;
        projections(
            builder,
            up,
            &[Projection {
                weights: lw.linears[5],
                output: o,
                n: h,
            }],
            inter,
            workers,
            true,
            Pack::Into(packed),
        )?;
        rows_node(
            builder,
            "bias",
            &add_h,
            &[(o.ptr as u64, h2), (lw.biases[5].as_ptr() as u64, 0)],
            &[],
            workers,
        )?;
        rows_node(
            builder,
            "residual",
            &add_h,
            &[(o.ptr as u64, h2), (x.ptr as u64, h2)],
            &[],
            workers,
        )?;
        rows_node(
            builder,
            "layer-norm",
            &ln,
            &[
                (o.ptr as u64, h2),
                (lw.norms[1].as_ptr() as u64, 0),
                (x.ptr as u64, h2),
            ],
            &[s.eps],
            workers,
        )?;
        builder.layer_end(l);
    }
    program.finish_with_instrumentation(1, spin, trace, false, false, false)
}

/// The reranker's head (`RobertaClassificationHead`) over the CLS row `cls` (`hidden` values), one
/// row: `logits = out_proj(tanh(dense(cls) + b_dense))` in f32 (`out_proj` packed with its one row
/// padded to 16, its bias added by the caller). `t` holds `hidden` values of scratch, `packed` the
/// GEMMs' activations.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lower_xlmr_head(
    hidden: usize,
    cls: *mut u16,
    dense: &[u16],
    dense_bias: &[u16],
    out_proj: &[u16],
    t: &mut AVec<u16>,
    logits: &mut AVec<f32>,
    packed: &mut AVec<u16>,
    workers: usize,
    spin: Option<u32>,
) -> std::result::Result<Mounted, ExecutionError> {
    if t.len() < hidden || logits.len() < 16 || dense_bias.len() < hidden {
        return Err(err("xlmr head: buffers too short"));
    }
    let add_h = elem(ElemOp::ResidualAdd, hidden)?;
    let tanh = elem(ElemOp::Softcap, hidden)?;
    let mut program = Builder::new(1)?;
    let builder = &mut program;
    let x = Buffer {
        ptr: cls,
        len: hidden,
    };
    let tb = buffer(t);
    let pk = buffer(packed);
    projections(
        builder,
        x,
        &[Projection {
            weights: dense,
            output: tb,
            n: hidden,
        }],
        hidden,
        workers,
        true,
        Pack::Into(pk),
    )?;
    rows_node(
        builder,
        "head-bias",
        &add_h,
        &[(tb.ptr as u64, hidden * 2), (dense_bias.as_ptr() as u64, 0)],
        &[],
        workers,
    )?;
    rows_node(
        builder,
        "head-tanh",
        &tanh,
        &[(tb.ptr as u64, hidden * 2), (tb.ptr as u64, hidden * 2)],
        &[1.0],
        workers,
    )?;
    projections_f32(
        builder,
        tb,
        &[ProjectionF32 {
            weights: out_proj,
            output: F32Buffer {
                ptr: logits.as_mut_ptr(),
                len: logits.len(),
            },
            n: 16,
        }],
        hidden,
        workers,
        true,
        Pack::Into(pk),
    )?;
    program.finish_with_instrumentation(1, spin, false, false, false, false)
}
