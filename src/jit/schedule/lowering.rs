//! The lowering: one layer of the model becomes a sequence of nodes of the
//! generated program, with the buffers, kernels and team sizes the mounted
//! scheduler holds.
//!
//! An attention layer: input norm, the Q/K/V projections, the head norms and
//! RoPE, the KV append, the attention, the output gate, `o_proj`, the
//! residual; then the MLP: post-attention norm, gate/up projections, SwiGLU,
//! `down_proj`, the residual. A GatedDeltaNet layer replaces the attention
//! part by its projections, the conv, the L2 norms and gates, the recurrence,
//! the gated norm and `out_proj`. Every node names its kernel, its task grid
//! and how many participants take it.

use super::*;
use crate::jit::arch::{KernelSet, Native};
use crate::jit::execution::{
    self as ex, abi::TaskGrid, build::Builder, Argument, ExecutionError, TaskBinding,
};

fn err(e: impl std::fmt::Display) -> ExecutionError {
    ExecutionError(e.to_string())
}
#[derive(Clone, Copy)]
pub(super) struct Buffer {
    pub(super) ptr: *mut u16,
    pub(super) len: usize,
}
impl Buffer {
    fn of(v: &mut [u16]) -> Self {
        Self {
            ptr: v.as_mut_ptr(),
            len: v.len(),
        }
    }
    fn check(self, rows: usize, width: usize) -> std::result::Result<(), ExecutionError> {
        if rows.checked_mul(width).is_none_or(|n| n > self.len) {
            return Err(ExecutionError("mounted layer buffer is too short".into()));
        }
        Ok(())
    }
}
#[derive(Clone, Copy)]
pub(super) struct Projection<'a> {
    pub(super) weights: &'a [u16],
    pub(super) output: Buffer,
    pub(super) n: usize,
}

#[derive(Clone, Copy)]
pub(super) struct F32Buffer {
    pub(super) ptr: *mut f32,
    pub(super) len: usize,
}
impl F32Buffer {
    fn of(v: &mut [f32]) -> Self {
        Self {
            ptr: v.as_mut_ptr(),
            len: v.len(),
        }
    }
    fn check(self, rows: usize, width: usize) -> std::result::Result<(), ExecutionError> {
        if rows.checked_mul(width).is_none_or(|n| n > self.len) {
            return Err(ExecutionError(
                "mounted f32 layer buffer is too short".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(super) struct ProjectionF32<'a> {
    pub(super) weights: &'a [u16],
    pub(super) output: F32Buffer,
    pub(super) n: usize,
}

/// Where an attention layer's cache and RoPE tables live, resolved at mount.
pub(crate) struct LayerMount {
    pub keys: *mut u16,
    pub values: *mut u16,
    /// The R16 copies of the block attention (`KvCall::keys_r16`), or null.
    pub keys_r16: *mut u16,
    pub values_r16: *mut u16,
    /// bf16 elements reserved per KV head.
    pub stride: usize,
    /// Positions reserved per KV head.
    pub capacity: usize,
    pub cos: *const u16,
    pub sin: *const u16,
    /// The layer appends its keys and values to the cache (false: it reads another layer's, Gemma
    /// 4).
    pub write: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct DeltaMount {
    pub conv: *mut f32,
    pub recurrent: *mut f32,
}

pub(super) fn binding(args: &[(u64, usize)], floats: &[f32]) -> TaskBinding {
    let mut b = TaskBinding::default();
    for (a, &(base, stride)) in b.arguments.iter_mut().zip(args) {
        *a = Argument {
            base,
            stride: stride as u64,
        };
    }
    for (dst, src) in b.floats.iter_mut().zip(floats) {
        *dst = src.to_bits();
    }
    b
}
pub(super) fn norm(
    builder: &mut Builder,
    kernel: &CompiledElem,
    x: Buffer,
    w: &[u16],
    y: Buffer,
    dim: usize,
    eps: f32,
    team: (usize, usize),
) -> std::result::Result<(), ExecutionError> {
    x.check(builder.batch, dim)?;
    y.check(builder.batch, dim)?;
    if w.len() < dim {
        return Err(ExecutionError("mounted RMSNorm weight is too short".into()));
    }
    builder.label("rms-norm");
    builder.rows(
        Native::elementary(kernel)?,
        binding(
            &[
                (x.ptr as u64, dim * 2),
                (w.as_ptr() as u64, 0),
                (y.ptr as u64, dim * 2),
            ],
            &[eps],
        ),
        team.0,
        team.1,
    )
}
pub(super) fn binary(
    builder: &mut Builder,
    kernel: &CompiledElem,
    x: Buffer,
    y: Buffer,
    dim: usize,
    team: (usize, usize),
) -> std::result::Result<(), ExecutionError> {
    x.check(builder.batch, dim)?;
    y.check(builder.batch, dim)?;
    builder.label("elementwise-binary");
    builder.rows(
        Native::elementary(kernel)?,
        binding(
            &[
                (x.ptr as u64, dim * 2),
                (y.ptr as u64, dim * 2),
                (x.ptr as u64, dim * 2),
            ],
            &[],
        ),
        team.0,
        team.1,
    )
}
/// The team of the row-wise RMS norms and residual additions: the whole team
/// under `Knobs::elem_team`, one task per row, participant zero otherwise.
fn elem_team(workers: usize) -> (usize, usize) {
    if super::spmd_elem_team() {
        (
            workers,
            if crate::jit::options::decode().pool_m1 {
                workers
            } else {
                1
            },
        )
    } else {
        (1, 1)
    }
}
/// The team of the chain around the attention (`head-norm-rope`, `kv-append`,
/// `attention-output-gate`): the whole team under `Knobs::attn_chain_team`,
/// participant zero otherwise.
fn attn_chain_team(workers: usize) -> (usize, usize) {
    if super::spmd_attn_chain_team() {
        (
            workers,
            if crate::jit::options::decode().pool_m1 {
                workers
            } else {
                1
            },
        )
    } else {
        (1, 1)
    }
}

/// `HERBERT_ATTN_PLAN=1`: one line on stderr per attention layer and regime, the kernel its block
/// of rows takes — the block kernel (ISA, rows, window, scratch) or the row kernel. Same bits on
/// both paths on x86-64, so the bytes of an answer cannot say which one ran; this line does.
fn attn_plan(layer: usize, batch: usize, window: usize, prefill: Option<&CompiledSdpaPrefill>) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("HERBERT_ATTN_PLAN").is_ok_and(|v| v != "0")) {
        return;
    }
    let isa = if cfg!(target_arch = "aarch64") {
        "neon"
    } else if crate::jit::gemm::is_supported() {
        "avx512"
    } else {
        "avx2"
    };
    match prefill {
        Some(p) => eprintln!(
            "[attn-plan] layer {layer} batch {batch} window {window}: block kernel {isa}, rows {}, window {}, scratch {} f32, {} bytes",
            p.rows(),
            p.spec().window,
            p.scratch_len(),
            p.code_bytes()
        ),
        None => eprintln!("[attn-plan] layer {layer} batch {batch} window {window}: row kernel"),
    }
}

fn offset(
    builder: &mut Builder,
    source: *const usize,
    factor: usize,
    subtract: usize,
) -> std::result::Result<*const u64, ExecutionError> {
    let out = builder.keep(0u64);
    builder.label("dynamic-offset");
    builder.once(
        Native::offset(factor, subtract)?,
        binding(&[(source as u64, 0), (out as u64, 0)], &[]),
    )?;
    Ok(out)
}
/// How a group of projections reads its input: row-major as it is, packed
/// into the matrix engine's buffer first, or from that buffer already packed
/// by the previous group on the same input.
#[derive(Clone, Copy)]
pub(super) enum Pack {
    Into(Buffer),
    Already(Buffer),
}

/// The activations the projections of `x` read: `x` itself, or the packed
/// buffer after a packing node when the target's plan runs this regime on its
/// matrix engine.
fn packed_input(
    builder: &mut Builder,
    x: Buffer,
    k: usize,
    first_n: usize,
    pack: Pack,
    workers: usize,
    par_m1: bool,
) -> std::result::Result<*const u8, ExecutionError> {
    let plan = crate::jit::arch::gemm_plan(GemmShape { k, n: first_n }, builder.batch);
    if !plan.opts.matrix {
        return Ok(x.ptr.cast());
    }
    packed_matrix(builder, x, k, plan, pack, workers, par_m1)
}

/// The quantized activations the int8 projections of `x` read (`jit/int8.rs`): the buffer after a
/// quantizing node, emitted for the first group of projections on `x` (`Pack::Into`) and again for
/// a later one (`Pack::Already`) when no earlier group on `x` had an int8 projection to quantize
/// for.
fn quantized_input(
    builder: &mut Builder,
    x: Buffer,
    k: usize,
    pack: Pack,
    workers: usize,
    par_m1: bool,
) -> std::result::Result<*const u8, ExecutionError> {
    let (packed, emit) = match pack {
        Pack::Into(p) => (p, true),
        Pack::Already(p) => (p, !builder.int8_ready),
    };
    if crate::jit::int8::a_bytes(builder.batch, k) > packed.len * 2 {
        return Err(ExecutionError(
            "mounted int8 activation buffer is too short".into(),
        ));
    }
    if emit {
        let call = builder.keep(ex::abi::PackCall {
            source: x.ptr,
            dest: packed.ptr,
            rows: &raw const builder.frame.rows,
        });
        builder.label("quantize-activations");
        builder.grid(
            Native::int8_quantize(k)?,
            binding(&[(call as u64, 0), (0, 1)], &[]),
            TaskGrid {
                rows_per_task: 1,
                tasks_per_tile: 1,
                last_block_only: false,
            },
            workers,
            if par_m1 { workers } else { 1 },
        )?;
    }
    builder.int8_ready = true;
    Ok(packed.ptr.cast())
}

pub(super) fn packed_matrix(
    builder: &mut Builder,
    x: Buffer,
    k: usize,
    plan: crate::jit::emit::GemmPlan,
    pack: Pack,
    workers: usize,
    par_m1: bool,
) -> std::result::Result<*const u8, ExecutionError> {
    let (packed, emit) = match pack {
        Pack::Into(p) => (p, true),
        Pack::Already(p) => (p, false),
    };
    if crate::jit::emit::packed_bytes(plan.opts, builder.batch, k) > packed.len * 2 {
        return Err(ExecutionError("mounted packing buffer is too short".into()));
    }
    if emit {
        let call = builder.keep(ex::abi::PackCall {
            source: x.ptr,
            dest: packed.ptr,
            rows: &raw const builder.frame.rows,
        });
        // SME: sixteen rows by pairs of `K`; AMX: thirty-two rows by single `K` (f32 or f16 under
        // `--f16`).
        let (kernel, rows_per_task) = if plan.opts.amx {
            (
                Native::bf16_pack_amx(k, plan.opts.neon)?,
                crate::jit::emit::AMX_ROWS,
            )
        } else {
            (Native::bf16_pack(k)?, crate::jit::emit::PACK_ROWS)
        };
        builder.label("pack-activations");
        builder.grid(
            kernel,
            binding(&[(call as u64, 0), (0, 1)], &[]),
            TaskGrid {
                rows_per_task,
                tasks_per_tile: 1,
                last_block_only: false,
            },
            workers,
            if par_m1 { workers } else { 1 },
        )?;
    }
    Ok(packed.ptr.cast())
}

pub(super) fn projections(
    builder: &mut Builder,
    x: Buffer,
    outputs: &[Projection<'_>],
    k: usize,
    workers: usize,
    par_m1: bool,
    pack: Pack,
) -> std::result::Result<(), ExecutionError> {
    x.check(builder.batch, k)?;
    // A group that mixes int8 projections and bf16 ones (LFM3's gate per head, 16 rows, beside its
    // int8 `q`, `k`, `v`) runs as two groups: the int8 activations and the matrix engine's packed
    // ones share one buffer, and a fused node one tile height. The int8 group first, then the bf16
    // one packs the buffer again.
    let int8 = |o: &&Projection<'_>| crate::jit::int8::gemm(o.n, k);
    if outputs.iter().any(|o| int8(&o)) && !outputs.iter().all(|o| int8(&o)) {
        let (quantized, plain): (Vec<Projection<'_>>, Vec<Projection<'_>>) =
            outputs.iter().partition(int8);
        let buffer = match pack {
            Pack::Into(p) | Pack::Already(p) => p,
        };
        projections(builder, x, &quantized, k, workers, par_m1, pack)?;
        return projections(builder, x, &plain, k, workers, par_m1, Pack::Into(buffer));
    }
    let activations = packed_input(builder, x, k, outputs[0].n, pack, workers, par_m1)?;
    let quantized = if outputs.iter().any(|o| crate::jit::int8::gemm(o.n, k)) {
        Some(quantized_input(builder, x, k, pack, workers, par_m1)?)
    } else {
        if matches!(pack, Pack::Into(_)) {
            builder.int8_ready = false;
        }
        None
    };
    // AArch64 only: the x86 target has no fused projection node.
    let fuse = cfg!(target_arch = "aarch64")
        && crate::jit::knobs::knobs().fuse_projections
        && outputs.len() > 1;
    let mut parts: Vec<(
        std::sync::Arc<crate::jit::compile::CompiledGemm>,
        ex::abi::Bf16Call,
    )> = Vec::new();
    for output in outputs {
        output.output.check(builder.batch, output.n)?;
        let shape = GemmShape { k, n: output.n };
        shape.check().map_err(err)?;
        if output.weights.len() < crate::jit::int8::weight_len(shape.n, shape.k) {
            return Err(ExecutionError(
                "mounted R16 weight geometry mismatch".into(),
            ));
        }
        // The bf16 GEMM of this shape: compiled once per shape, cached.
        let plan = crate::jit::arch::gemm_plan(shape, builder.batch);
        let kernel = crate::jit::gemm::get_or_compile(
            shape,
            plan.tile,
            plan.opts,
            true,
            crate::jit::emit::AFormat::Bf16,
        )
        .map_err(err)?
        .ok_or_else(|| ExecutionError("mounted bf16 kernel refused".into()))?;
        let call = ex::abi::Bf16Call {
            activations: if crate::jit::int8::gemm(output.n, k) {
                quantized.unwrap_or(activations)
            } else {
                activations
            },
            weights: output.weights.as_ptr(),
            output: output.output.ptr.cast(),
            rows: 0,
            row_count: &raw const builder.frame.rows,
        };
        if fuse {
            parts.push((kernel, call));
            continue;
        }
        let call = builder.keep(call);
        builder.label("bf16-projection");
        builder.grid(
            Native::bf16_tile_selector(&kernel)?,
            binding(&[(call as u64, 0), (0, 1)], &[]),
            TaskGrid {
                rows_per_task: kernel.tile().mr,
                tasks_per_tile: kernel.n_tile_count(),
                last_block_only: false,
            },
            workers,
            if par_m1 { workers } else { 1 },
        )?;
    }
    if fuse {
        fused_projections(builder, "bf16-projection", &parts, workers, par_m1)?;
    }
    Ok(())
}

/// One node for the `parts` projections of an input: their tasks end to end,
/// the multi selector of the target dispatching each to its own selector.
fn fused_projections(
    builder: &mut Builder,
    label: &'static str,
    parts: &[(
        std::sync::Arc<crate::jit::compile::CompiledGemm>,
        ex::abi::Bf16Call,
    )],
    workers: usize,
    par_m1: bool,
) -> std::result::Result<(), ExecutionError> {
    if parts.len() > 4 {
        return Err(ExecutionError(
            "a fused projection node takes at most four parts".into(),
        ));
    }
    let mut multi = ex::abi::Bf16MultiCall {
        calls: [parts[0].1; 4],
        count: parts.len(),
    };
    for (i, (_, call)) in parts.iter().enumerate() {
        multi.calls[i] = *call;
    }
    let call = builder.keep(multi);
    let kernels: Vec<&crate::jit::compile::CompiledGemm> =
        parts.iter().map(|(k, _)| k.as_ref()).collect();
    let tasks: usize = kernels.iter().map(|k| k.n_tile_count()).sum();
    builder.label(label);
    builder.grid(
        Native::bf16_multi_tile_selector(&kernels)?,
        binding(&[(call as u64, 0), (0, 1)], &[]),
        TaskGrid {
            rows_per_task: kernels[0].tile().mr,
            tasks_per_tile: tasks,
            last_block_only: false,
        },
        workers,
        if par_m1 { workers } else { 1 },
    )
}

pub(super) fn projections_f32(
    builder: &mut Builder,
    x: Buffer,
    outputs: &[ProjectionF32<'_>],
    k: usize,
    workers: usize,
    par_m1: bool,
    pack: Pack,
) -> std::result::Result<(), ExecutionError> {
    x.check(builder.batch, k)?;
    let activations = packed_input(builder, x, k, outputs[0].n, pack, workers, par_m1)?;
    let quantized = if outputs.iter().any(|o| crate::jit::int8::gemm(o.n, k)) {
        Some(quantized_input(builder, x, k, pack, workers, par_m1)?)
    } else {
        if matches!(pack, Pack::Into(_)) {
            builder.int8_ready = false;
        }
        None
    };
    // AArch64 only: the x86 target has no fused projection node.
    let fuse = cfg!(target_arch = "aarch64")
        && crate::jit::knobs::knobs().fuse_projections
        && outputs.len() > 1;
    let mut parts: Vec<(
        std::sync::Arc<crate::jit::compile::CompiledGemm>,
        ex::abi::Bf16Call,
    )> = Vec::new();
    for output in outputs {
        output.output.check(builder.batch, output.n)?;
        let shape = GemmShape { k, n: output.n };
        shape.check().map_err(err)?;
        if output.weights.len() < crate::jit::int8::weight_len(shape.n, shape.k) {
            return Err(ExecutionError(
                "mounted R16 f32 weight geometry mismatch".into(),
            ));
        }
        // Same GEMM, f32 output: the GatedDeltaNet mixer keeps its projections in f32.
        let plan = crate::jit::arch::gemm_plan(shape, builder.batch);
        let kernel = crate::jit::gemm::get_or_compile(
            shape,
            plan.tile,
            plan.opts,
            false,
            crate::jit::emit::AFormat::Bf16,
        )
        .map_err(err)?
        .ok_or_else(|| ExecutionError("mounted bf16-to-f32 kernel refused".into()))?;
        let call = ex::abi::Bf16Call {
            activations: if crate::jit::int8::gemm(output.n, k) {
                quantized.unwrap_or(activations)
            } else {
                activations
            },
            weights: output.weights.as_ptr(),
            output: output.output.ptr.cast(),
            rows: 0,
            row_count: &raw const builder.frame.rows,
        };
        if fuse {
            parts.push((kernel, call));
            continue;
        }
        let call = builder.keep(call);
        builder.label("bf16-projection-f32");
        builder.grid(
            Native::bf16_tile_selector(&kernel)?,
            binding(&[(call as u64, 0), (0, 1)], &[]),
            TaskGrid {
                rows_per_task: kernel.tile().mr,
                tasks_per_tile: kernel.n_tile_count(),
                last_block_only: false,
            },
            workers,
            if par_m1 { workers } else { 1 },
        )?;
    }
    if fuse {
        fused_projections(builder, "bf16-projection-f32", &parts, workers, par_m1)?;
    }
    Ok(())
}

impl Scheduler {
    /// All pointers refer to the Runner's reserved buffers and borrowed weights.
    /// This method only constructs code/tables; it never executes a partial layer.
    pub(crate) fn lower_attention_layer(
        &mut self,
        builder: &mut Builder,
        w: &LayerWeights<'_>,
        mount: LayerMount,
        layer: usize,
        workers: usize,
    ) -> std::result::Result<(), ExecutionError> {
        if builder.batch > self.max_batch || !matches!(w.mixer, Mixer::Attention) {
            return Err(ExecutionError(
                "layer is not yet supported by attention lowering".into(),
            ));
        }
        let s = self.shape;
        let sl = s.for_layer(
            w.head_dim,
            w.rotary_ndims,
            w.n_kv_heads.unwrap_or(s.n_kv_heads),
        );
        let (hidden, hd, nh, nkv, inter) = (
            s.hidden,
            sl.head_dim,
            s.n_heads,
            sl.n_kv_heads,
            w.intermediate,
        );
        if hd > s.head_dim
            || nkv > s.n_kv_heads
            || inter > s.intermediate
            || sl.rotary_ndims > hd
            || nkv == 0
            || nh % nkv != 0
        {
            return Err(ExecutionError(
                "mounted layer exceeds scheduler geometry".into(),
            ));
        }
        let hk = self
            .heads
            .iter()
            .find(|k| k.head_dim == hd)
            .ok_or_else(|| ExecutionError("missing mounted head kernels".into()))?;
        let rope = self
            .ropes
            .iter()
            .find(|(rn, _)| *rn == w.rotary_ndims)
            .map(|(_, k)| k)
            .ok_or_else(|| ExecutionError("missing mounted RoPE kernel".into()))?;
        let (qd, kvd, qrs, half) = (sl.q_dim(), sl.kv_dim(), sl.q_row_stride(), sl.rope_half());
        builder.begin_layer(layer);
        let b = &mut self.buffers;
        let packed = Buffer::of(&mut b.packed);
        let x = Buffer::of(&mut b.x);
        let xn = Buffer::of(&mut b.x_norm);
        let q = Buffer::of(&mut b.q);
        let k = Buffer::of(&mut b.k);
        let v = Buffer::of(&mut b.v);
        let attn = Buffer::of(&mut b.attn);
        let out = Buffer::of(&mut b.proj_out);
        let kv_spec = ex::abi::KvSpec {
            heads: nkv,
            head_dim: hd,
            stride: mount.stride,
            capacity: mount.capacity,
            max_batch: builder.batch,
            r16: !mount.keys_r16.is_null(),
        };
        let kv_call = builder.keep(ex::abi::KvCall {
            keys: mount.keys,
            values: mount.values,
            input_keys: k.ptr,
            input_values: v.ptr,
            rows: &raw const builder.frame.rows,
            past: &raw const builder.frame.position,
            keys_r16: mount.keys_r16,
            values_r16: mount.values_r16,
        });
        norm(
            builder,
            &self.norm_hidden,
            x,
            w.input_norm,
            xn,
            hidden,
            s.eps,
            elem_team(workers),
        )?;
        let par_m1 = crate::jit::options::decode().pool_m1;
        // LFM3: the gate per head (`self_attn.gate_proj`, `heads` logits a row, bf16) into the
        // MLP's `gate` buffer, free until the MLP; a sliding layer's window.
        let lfm3 = w.lfm3.as_ref();
        let head_gate = lfm3
            .filter(|x| !x.head_gate.is_empty())
            .map(|x| x.head_gate);
        let window = lfm3.map_or(0, |x| x.window);
        let gate_logits = Buffer::of(&mut b.gate);
        let mut outputs = vec![
            Projection {
                weights: w.wq,
                output: q,
                n: sl.q_gemm_dim(),
            },
            Projection {
                weights: w.wk,
                output: k,
                n: kvd,
            },
            Projection {
                weights: w.wv,
                output: v,
                n: kvd,
            },
        ];
        if let Some(g) = head_gate {
            if !nh.is_multiple_of(16) {
                return Err(ExecutionError(
                    "the gate per head wants a multiple of 16 heads (R16 GEMM)".into(),
                ));
            }
            outputs.push(Projection {
                weights: g,
                output: gate_logits,
                n: nh,
            });
        }
        projections(
            builder,
            xn,
            &outputs,
            hidden,
            workers,
            par_m1,
            Pack::Into(packed),
        )?;
        // Qwen2.5: the biases of the three projections, before the RoPE and the append.
        if let Some((bias_q, bias_kv)) = &hk.bias {
            for (data, bias, kernel, width, row_stride) in [
                (q, w.q_bias, bias_q, qd, qrs),
                (k, w.k_bias, bias_kv, kvd, kvd),
                (v, w.v_bias, bias_kv, kvd, kvd),
            ] {
                if bias.len() < width {
                    return Err(ExecutionError("mounted projection bias too short".into()));
                }
                data.check(builder.batch, row_stride)?;
                builder.label("qkv-bias");
                builder.rows(
                    Native::elementary(kernel)?,
                    binding(
                        &[(data.ptr as u64, row_stride * 2), (bias.as_ptr() as u64, 0)],
                        &[],
                    ),
                    attn_chain_team(workers).0,
                    attn_chain_team(workers).1,
                )?;
            }
        }
        let pos = &raw const builder.frame.position;
        let rope_offset = offset(builder, pos, half * 2, 0)?;
        for (data, weights, heads, stride, row_stride) in
            [(q, w.q_norm, nh, hd, qrs), (k, w.k_norm, nkv, hd, kvd)].into_iter()
        {
            // No per-head norm (Qwen2.5): the RoPE alone.
            let head_norm = if weights.is_empty() {
                None
            } else {
                Some(&*hk.norm_head)
            };
            if head_norm.is_some() && weights.len() < hd {
                return Err(ExecutionError("mounted head norm weight too short".into()));
            }
            data.check(builder.batch, row_stride)?;
            let mut binding = binding(
                &[
                    (data.ptr as u64, row_stride * 2),
                    (weights.as_ptr() as u64, 0),
                    (mount.cos as u64, half * 2),
                    (mount.sin as u64, half * 2),
                ],
                &[s.eps],
            );
            binding.argument_bases[2] = rope_offset;
            binding.argument_bases[3] = rope_offset;
            builder.label(if head_norm.is_some() {
                "head-norm-rope"
            } else {
                "head-rope"
            });
            builder.rows(
                Native::norm_rope(head_norm, rope, heads, stride)?,
                binding,
                attn_chain_team(workers).0,
                attn_chain_team(workers).1,
            )?;
        }
        // The append is serial; the remaining team waits at the plan barrier.
        let mut append = binding(&[(kv_call as u64, 0), (0, 1)], &[]);
        append.tasks = nkv;
        builder.label("kv-append");
        builder.grid(
            Native::kv_append(kv_spec)?,
            append,
            TaskGrid {
                rows_per_task: builder.batch,
                tasks_per_tile: nkv,
                last_block_only: false,
            },
            attn_chain_team(workers).0,
            attn_chain_team(workers).1,
        )?;
        // A sliding layer (LFM3) takes the block kernel of its window, compiled here from the
        // causal one (cached by spec); its scratch is no longer than the causal one's for a
        // window shorter than the cache.
        let windowed =
            match hk.sdpa_prefill.as_deref() {
                Some(p) if window > 0 => crate::jit::sdpa::get_or_compile_prefill(
                    crate::jit::emit_sdpa::SdpaPrefillSpec { window, ..p.spec() },
                )
                .map_err(|e| ExecutionError(format!("windowed block attention: {e}")))?,
                _ => None,
            };
        if windowed
            .as_ref()
            .is_some_and(|p| p.scratch_len() > self.scores_stride)
        {
            return Err(ExecutionError(
                "windowed block attention: scratch longer than the score slots".into(),
            ));
        }
        let prefill = if window > 0 {
            windowed.as_deref()
        } else {
            hk.sdpa_prefill.as_deref()
        };
        attn_plan(layer, builder.batch, window, prefill);
        let chunk = prefill.map_or(1, CompiledSdpaPrefill::rows);
        // A block of rows (the prefill) with the block kernel: one task per KV head and chunk, its
        // query heads in turn.
        let kv_task = cfg!(target_arch = "x86_64")
            && crate::jit::knobs::knobs().attn_kv_task
            && builder.batch > 1
            && prefill.is_some()
            && sl.gqa() > 1;
        if b.scores.len() / self.scores_stride < workers {
            return Err(ExecutionError(
                "mounted attention score slots too short".into(),
            ));
        }
        let call = builder.keep(ex::abi::AttentionCall {
            q: q.ptr,
            keys: mount.keys,
            values: mount.values,
            output: attn.ptr,
            rows: &raw const builder.frame.rows,
            past: &raw const builder.frame.position,
            kv_stride: mount.stride,
            scale: sl.scale(),
            sink: 0.0,
            keys_r16: mount.keys_r16,
            values_r16: mount.values_r16,
            segments: std::ptr::null(),
        });
        let mut args = binding(
            &[(call as u64, 0), (0, 1), (b.scores.as_mut_ptr() as u64, 0)],
            &[],
        );
        args.participant_strides[2] = (self.scores_stride * 4) as u64;
        builder.label("attention");
        builder.grid(
            Native::attention_tile_selector(
                &hk.sdpa,
                prefill,
                ex::abi::AttentionSpec {
                    heads: nh,
                    gqa: sl.gqa(),
                    head_dim: hd,
                    q_row: qrs,
                    q_head: hd,
                    out_row: qd,
                    grouped: super::pass::sdpa_item_order_grouped(),
                    kv_task,
                    window,
                    whole_block: false,
                    ahead: 0,
                    segments: false,
                },
            )?,
            args,
            TaskGrid {
                rows_per_task: chunk,
                tasks_per_tile: if kv_task { nkv } else { nh },
                last_block_only: false,
            },
            workers,
            if par_m1 { workers } else { 1 },
        )?;
        // LFM3: each head's output times `bf16(σ(g_h))`, one task per head over the block's rows.
        if head_gate.is_some() {
            attn.check(builder.batch, qd)?;
            gate_logits.check(builder.batch, nh)?;
            let call = builder.keep(ex::abi::HeadGateCall {
                output: attn.ptr,
                gate: gate_logits.ptr,
                rows: &raw const builder.frame.rows,
            });
            builder.label("head-gate");
            builder.grid(
                Native::head_gate(nh, hd)?,
                binding(&[(call as u64, 0), (0, 1)], &[]),
                TaskGrid {
                    rows_per_task: builder.batch,
                    tasks_per_tile: nh,
                    last_block_only: false,
                },
                workers,
                if par_m1 { workers } else { 1 },
            )?;
        }
        // The gate is compact in `q`, right after `Q` on the same row: the
        // whole-row kernel, the second operand at the row stride of `q`
        // (`2·q_dim`) while `attn` steps by `q_dim`.
        if s.output_gate {
            let kernel = hk
                .gate_row
                .as_ref()
                .ok_or_else(|| ExecutionError("missing output gate kernel".into()))?;
            attn.check(builder.batch, qd)?;
            q.check(builder.batch, qrs)?;
            builder.label("attention-output-gate");
            builder.rows(
                Native::elementary(kernel)?,
                binding(
                    &[
                        (attn.ptr as u64, qd * 2),
                        (unsafe { q.ptr.add(qd) } as u64, qrs * 2),
                        (attn.ptr as u64, qd * 2),
                    ],
                    &[],
                ),
                attn_chain_team(workers).0,
                attn_chain_team(workers).1,
            )?;
        }
        projections(
            builder,
            attn,
            &[Projection {
                weights: w.wo,
                output: out,
                n: hidden,
            }],
            qd,
            workers,
            par_m1,
            Pack::Into(packed),
        )?;
        mixer_residual(
            builder,
            &self.norm_hidden,
            &self.residual,
            x,
            out,
            xn,
            w,
            hidden,
            s.eps,
            elem_team(workers),
        )?;
        self.lower_qwen_mlp(builder, w, workers)?;
        builder.layer_end(layer);
        Ok(())
    }
}

/// `x += out`, or with LFM3's sandwich norm `x += post_mixer_norm(out)` (the norm into `xn`, free
/// once the mixer has read it).
#[allow(clippy::too_many_arguments)]
fn mixer_residual(
    builder: &mut Builder,
    norm_kernel: &CompiledElem,
    residual: &CompiledElem,
    x: Buffer,
    out: Buffer,
    xn: Buffer,
    w: &LayerWeights<'_>,
    hidden: usize,
    eps: f32,
    team: (usize, usize),
) -> std::result::Result<(), ExecutionError> {
    match w.lfm3.as_ref().filter(|l| !l.post_mixer_norm.is_empty()) {
        Some(l) => {
            norm(
                builder,
                norm_kernel,
                out,
                l.post_mixer_norm,
                xn,
                hidden,
                eps,
                team,
            )?;
            binary(builder, residual, x, xn, hidden, team)
        }
        None => binary(builder, residual, x, out, hidden, team),
    }
}

impl Scheduler {
    /// Lower the complete Qwen GatedDeltaNet mixer and the shared dense MLP.
    /// Persistent state pointers belong to the Runner and remain stable for
    /// the lifetime of the mounted call.
    pub(crate) fn lower_delta_layer(
        &mut self,
        builder: &mut Builder,
        w: &LayerWeights<'_>,
        mount: DeltaMount,
        layer: usize,
        workers: usize,
    ) -> std::result::Result<(), ExecutionError> {
        let Mixer::GatedDeltaNet(dw) = &w.mixer else {
            return Err(ExecutionError("layer is not a DeltaNet mixer".into()));
        };
        let s = self.shape;
        if builder.batch > self.max_batch {
            return Err(ExecutionError(
                "generated DeltaNet requires the supported Qwen standard dense layer".into(),
            ));
        }
        if dw.short_conv {
            return self.lower_short_conv_layer(builder, w, mount, layer, workers);
        }
        let (hidden, inter) = (s.hidden, w.intermediate);
        let (nk, nv, hkd, hvd, ck) = (
            dw.num_k_heads,
            dw.num_v_heads,
            dw.head_k_dim,
            dw.head_v_dim,
            dw.conv_kernel,
        );
        let shape = ex::abi::Shape {
            nk,
            nv,
            hkd,
            hvd,
            ck,
        };
        shape.check()?;
        let (value_dim, conv_dim) = (shape.value(), shape.conv());
        if dw.mamba {
            return self.lower_mamba_layer(builder, w, mount, layer, workers, shape);
        }
        if inter > s.intermediate
            || dw.conv1d_weight.len() < conv_dim * ck
            || dw.a_log.len() < nv
            || dw.dt_bias.len() < nv
            || dw.norm_weight.len() < hvd
            || mount.conv.is_null()
            || mount.recurrent.is_null()
        {
            return Err(ExecutionError(
                "generated DeltaNet weights, state, or layer geometry mismatch".into(),
            ));
        }
        // Build every generated blob before exposing any state to execution.
        // The engine ISA floor includes FMA on both AVX2 and AVX-512.
        let conv_kernel = Native::delta_convolution(shape, self.conv.as_deref(), true)?;
        let recurrence_kernel = Native::delta_recurrence(shape, self.recurrence.as_deref())?;
        let norm_kernel = Native::delta_normed_gate(hvd, self.gated_norm.as_deref(), true)?;
        let tables = builder.keep(ex::math_data::TABLES);
        let par_m1 = crate::jit::options::decode().pool_m1;
        let b = &mut self.buffers;
        let packed = Buffer::of(&mut b.packed);
        let x = Buffer::of(&mut b.x);
        let xn = Buffer::of(&mut b.x_norm);
        let out = Buffer::of(&mut b.proj_out);
        let qkv = F32Buffer::of(&mut b.dn_qkv);
        let z = F32Buffer::of(&mut b.dn_z);
        let ab = F32Buffer::of(&mut b.dn_ab);
        let delta_out = F32Buffer::of(&mut b.dn_out);
        qkv.check(builder.batch, conv_dim.max(nv))?;
        z.check(builder.batch, value_dim)?;
        ab.check(builder.batch, 2 * nv)?;
        delta_out.check(builder.batch, value_dim.max(nv))?;
        if b.dn_beta.len() < builder.batch * nv
            || b.dn_decay.len() < builder.batch * nv
            || b.dn_out_bf16.len() < builder.batch * value_dim
        {
            return Err(ExecutionError(
                "mounted DeltaNet reserve is too short".into(),
            ));
        }

        builder.begin_layer(layer);
        norm(
            builder,
            &self.norm_hidden,
            x,
            w.input_norm,
            xn,
            hidden,
            s.eps,
            elem_team(workers),
        )?;
        if cfg!(target_arch = "aarch64")
            && crate::jit::knobs::knobs().fuse_projections
            && builder.batch == 1
        {
            // One row: a and b land straight in the a|b row (the kernel's row
            // stride is never used), and the four projections form one node.
            let ab_b = F32Buffer {
                ptr: unsafe { ab.ptr.add(nv) },
                len: ab.len - nv,
            };
            projections_f32(
                builder,
                xn,
                &[
                    ProjectionF32 {
                        weights: dw.in_proj_a,
                        output: ab,
                        n: nv,
                    },
                    ProjectionF32 {
                        weights: dw.in_proj_b,
                        output: ab_b,
                        n: nv,
                    },
                    ProjectionF32 {
                        weights: dw.in_proj_qkv,
                        output: qkv,
                        n: conv_dim,
                    },
                    ProjectionF32 {
                        weights: dw.in_proj_z,
                        output: z,
                        n: value_dim,
                    },
                ],
                hidden,
                workers,
                par_m1,
                Pack::Into(packed),
            )?;
        } else {
            // A and B use scratch that is dead before the recurrence. Their
            // generated interleave establishes the a|b row layout.
            projections_f32(
                builder,
                xn,
                &[
                    ProjectionF32 {
                        weights: dw.in_proj_a,
                        output: qkv,
                        n: nv,
                    },
                    ProjectionF32 {
                        weights: dw.in_proj_b,
                        output: delta_out,
                        n: nv,
                    },
                ],
                hidden,
                workers,
                par_m1,
                Pack::Into(packed),
            )?;
            let interleave = builder.keep(ex::abi::InterleaveCall {
                a: qkv.ptr,
                b: delta_out.ptr,
                output: ab.ptr,
            });
            builder.label("delta-interleave-a-b");
            builder.rows(
                Native::delta_interleave(nv)?,
                binding(&[(interleave as u64, 0), (0, 1)], &[]),
                1,
                1,
            )?;
            projections_f32(
                builder,
                xn,
                &[
                    ProjectionF32 {
                        weights: dw.in_proj_qkv,
                        output: qkv,
                        n: conv_dim,
                    },
                    ProjectionF32 {
                        weights: dw.in_proj_z,
                        output: z,
                        n: value_dim,
                    },
                ],
                hidden,
                workers,
                par_m1,
                Pack::Already(packed),
            )?;
        }

        let conv_call = builder.keep(ex::abi::ConvCall {
            data: qkv.ptr,
            weights: dw.conv1d_weight.as_ptr(),
            state: mount.conv,
            rows: &raw const builder.frame.rows,
            tables,
            bias: std::ptr::null(),
        });
        let conv_workers = if crate::jit::conv::single_thread() {
            1
        } else {
            workers
        };
        builder.label("delta-causal-convolution");
        builder.grid(
            conv_kernel,
            binding(&[(conv_call as u64, 0), (0, 1)], &[]),
            TaskGrid {
                rows_per_task: builder.batch,
                tasks_per_tile: conv_dim.div_ceil(16),
                last_block_only: false,
            },
            conv_workers,
            if par_m1 { conv_workers } else { 1 },
        )?;
        // The L2 of the row's q/k heads and its gates as one node, one task
        // per (row, head), the L2 on the task of head zero: one barrier fewer
        // than two nodes. Rows and heads are independent: the contiguous task
        // ranges give each participant whole rows at `m ≥ 4`, and at `m = 1`
        // the `nv` gates of the single row still spread over the team (one
        // task per row put them all on participant zero, +0.018 ms per decode
        // step).
        let l2_gates_call = builder.keep(ex::abi::L2GatesCall {
            l2: ex::abi::L2Call {
                qkv: qkv.ptr,
                eps: super::DN_L2_EPS,
            },
            gates: ex::abi::GateCall {
                ab: ab.ptr,
                a_log: dw.a_log.as_ptr(),
                bias: dw.dt_bias.as_ptr(),
                beta: b.dn_beta.as_mut_ptr(),
                decay: b.dn_decay.as_mut_ptr(),
                tables,
            },
        });
        let l2_workers = workers;
        builder.label("delta-l2-gates");
        builder.grid(
            Native::delta_l2_gates(shape, true)?,
            binding(&[(l2_gates_call as u64, 0), (0, 1)], &[]),
            TaskGrid {
                rows_per_task: 1,
                tasks_per_tile: nv,
                last_block_only: false,
            },
            l2_workers,
            if par_m1 { l2_workers } else { 1 },
        )?;
        let recurrence_call = builder.keep(ex::abi::RecurrenceCall {
            state: mount.recurrent,
            output: delta_out.ptr,
            qkv: qkv.ptr,
            beta: b.dn_beta.as_ptr(),
            decay: b.dn_decay.as_ptr(),
            rows: &raw const builder.frame.rows,
            position: &raw const builder.frame.position,
            prefill: usize::from(builder.batch > 1),
            d: std::ptr::null(),
        });
        let recurrence_workers = if crate::jit::recurrence::single_thread() {
            1
        } else {
            workers
        };
        builder.label("delta-causal-recurrence");
        builder.grid(
            recurrence_kernel,
            binding(&[(recurrence_call as u64, 0), (0, 1)], &[]),
            TaskGrid {
                rows_per_task: builder.batch,
                tasks_per_tile: nv * crate::jit::recurrence::vd_split(hvd),
                last_block_only: false,
            },
            recurrence_workers,
            if par_m1 { recurrence_workers } else { 1 },
        )?;
        let norm_call = builder.keep(ex::abi::NormCall {
            output: delta_out.ptr,
            gate: z.ptr,
            weights: dw.norm_weight.as_ptr(),
            output_bf16: b.dn_out_bf16.as_mut_ptr(),
            tables,
            eps: s.eps,
        });
        let norm_workers = if crate::jit::gated_norm::single_thread() {
            1
        } else {
            workers
        };
        builder.label("delta-gated-norm");
        builder.grid(
            norm_kernel,
            binding(&[(norm_call as u64, 0), (0, 1)], &[]),
            TaskGrid {
                rows_per_task: 1,
                tasks_per_tile: nv,
                last_block_only: false,
            },
            norm_workers,
            if par_m1 { norm_workers } else { 1 },
        )?;
        projections(
            builder,
            Buffer::of(&mut b.dn_out_bf16),
            &[Projection {
                weights: dw.out_proj,
                output: out,
                n: hidden,
            }],
            value_dim,
            workers,
            par_m1,
            Pack::Into(packed),
        )?;

        // Qwen's standard residual and dense MLP are shared with attention.
        binary(builder, &self.residual, x, out, hidden, elem_team(workers))?;
        self.lower_qwen_mlp(builder, w, workers)?;
        builder.layer_end(layer);
        Ok(())
    }
}

impl Scheduler {
    /// Granite 4's Mamba2 mixer in the DeltaNet's buffers and state (`DeltaWeights::mamba`), then
    /// the layer's MLP (the shared MLP, and the experts when it has them): `xn = norm(x)`;
    /// `dt`, `[C | B | x]` and the gate `z` in f32; the causal conv with its biases, SiLU; `dt
    /// = softplus(dt + dt_bias)` and its decay per (row, head); the recurrence per head with
    /// the skip `D`; `w · rms(y · silu(z))` per row, to bf16; `out_proj`; the residual.
    fn lower_mamba_layer(
        &mut self,
        builder: &mut Builder,
        w: &LayerWeights<'_>,
        mount: DeltaMount,
        layer: usize,
        workers: usize,
        shape: ex::abi::Shape,
    ) -> std::result::Result<(), ExecutionError> {
        let Mixer::GatedDeltaNet(dw) = &w.mixer else {
            return Err(ExecutionError("layer is not a Mamba2 mixer".into()));
        };
        let s = self.shape;
        let (hidden, nv, hvd) = (s.hidden, shape.nv, shape.hvd);
        let (value_dim, conv_dim) = (shape.value(), shape.conv());
        if w.intermediate > s.intermediate
            || dw.conv1d_weight.len() < conv_dim * shape.ck
            || dw.conv1d_bias.len() < conv_dim
            || dw.a_log.len() < nv
            || dw.dt_bias.len() < nv
            || dw.d.len() < nv
            || dw.norm_weight.len() < value_dim
            || mount.conv.is_null()
            || mount.recurrent.is_null()
        {
            return Err(ExecutionError(
                "generated Mamba2 weights, state, or layer geometry mismatch".into(),
            ));
        }
        let spec = ex_recurrence_spec(hvd);
        let vector = crate::jit::recurrence::get_or_compile(spec).map_err(err)?;
        let conv_kernel = Native::mamba_convolution(shape, true)?;
        let gates_kernel = Native::mamba_gates(nv, true)?;
        let recurrence_kernel = Native::mamba_recurrence(shape, vector.as_deref())?;
        let norm_kernel = Native::mamba_normed_gate(value_dim, true)?;
        let tables = builder.keep(ex::math_data::TABLES);
        let par_m1 = crate::jit::options::decode().pool_m1;
        let b = &mut self.buffers;
        let packed = Buffer::of(&mut b.packed);
        let x = Buffer::of(&mut b.x);
        let xn = Buffer::of(&mut b.x_norm);
        let out = Buffer::of(&mut b.proj_out);
        let qkv = F32Buffer::of(&mut b.dn_qkv);
        let z = F32Buffer::of(&mut b.dn_z);
        let ab = F32Buffer::of(&mut b.dn_ab);
        let delta_out = F32Buffer::of(&mut b.dn_out);
        qkv.check(builder.batch, conv_dim)?;
        z.check(builder.batch, value_dim)?;
        ab.check(builder.batch, nv)?;
        delta_out.check(builder.batch, value_dim)?;
        if b.dn_beta.len() < builder.batch * nv
            || b.dn_decay.len() < builder.batch * nv
            || b.dn_out_bf16.len() < builder.batch * value_dim
        {
            return Err(ExecutionError("mounted Mamba2 reserve is too short".into()));
        }

        builder.begin_layer(layer);
        norm(
            builder,
            &self.norm_hidden,
            x,
            w.input_norm,
            xn,
            hidden,
            s.eps,
            elem_team(workers),
        )?;
        projections_f32(
            builder,
            xn,
            &[
                ProjectionF32 {
                    weights: dw.in_proj_a,
                    output: ab,
                    n: nv,
                },
                ProjectionF32 {
                    weights: dw.in_proj_qkv,
                    output: qkv,
                    n: conv_dim,
                },
                ProjectionF32 {
                    weights: dw.in_proj_z,
                    output: z,
                    n: value_dim,
                },
            ],
            hidden,
            workers,
            par_m1,
            Pack::Into(packed),
        )?;
        let conv_call = builder.keep(ex::abi::ConvCall {
            data: qkv.ptr,
            weights: dw.conv1d_weight.as_ptr(),
            state: mount.conv,
            rows: &raw const builder.frame.rows,
            tables,
            bias: dw.conv1d_bias.as_ptr(),
        });
        builder.label("mamba-causal-convolution");
        builder.grid(
            conv_kernel,
            binding(&[(conv_call as u64, 0), (0, 1)], &[]),
            TaskGrid {
                rows_per_task: builder.batch,
                tasks_per_tile: conv_dim.div_ceil(16),
                last_block_only: false,
            },
            workers,
            if par_m1 { workers } else { 1 },
        )?;
        let gates_call = builder.keep(ex::abi::GateCall {
            ab: ab.ptr,
            a_log: dw.a_log.as_ptr(),
            bias: dw.dt_bias.as_ptr(),
            beta: b.dn_beta.as_mut_ptr(),
            decay: b.dn_decay.as_mut_ptr(),
            tables,
        });
        builder.label("mamba-gates");
        builder.grid(
            gates_kernel,
            binding(&[(gates_call as u64, 0), (0, 1)], &[]),
            TaskGrid {
                rows_per_task: 1,
                tasks_per_tile: nv,
                last_block_only: false,
            },
            workers,
            if par_m1 { workers } else { 1 },
        )?;
        let recurrence_call = builder.keep(ex::abi::RecurrenceCall {
            state: mount.recurrent,
            output: delta_out.ptr,
            qkv: qkv.ptr,
            beta: b.dn_beta.as_ptr(),
            decay: b.dn_decay.as_ptr(),
            rows: &raw const builder.frame.rows,
            position: &raw const builder.frame.position,
            prefill: usize::from(builder.batch > 1),
            d: dw.d.as_ptr(),
        });
        builder.label("mamba-causal-recurrence");
        builder.grid(
            recurrence_kernel,
            binding(&[(recurrence_call as u64, 0), (0, 1)], &[]),
            TaskGrid {
                rows_per_task: builder.batch,
                tasks_per_tile: nv,
                last_block_only: false,
            },
            workers,
            if par_m1 { workers } else { 1 },
        )?;
        let norm_call = builder.keep(ex::abi::NormCall {
            output: delta_out.ptr,
            gate: z.ptr,
            weights: dw.norm_weight.as_ptr(),
            output_bf16: b.dn_out_bf16.as_mut_ptr(),
            tables,
            eps: s.eps,
        });
        builder.label("mamba-gated-norm");
        builder.grid(
            norm_kernel,
            binding(&[(norm_call as u64, 0), (0, 1)], &[]),
            TaskGrid {
                rows_per_task: 1,
                tasks_per_tile: 1,
                last_block_only: false,
            },
            workers,
            if par_m1 { workers } else { 1 },
        )?;
        projections(
            builder,
            Buffer::of(&mut b.dn_out_bf16),
            &[Projection {
                weights: dw.out_proj,
                output: out,
                n: hidden,
            }],
            value_dim,
            workers,
            par_m1,
            Pack::Into(packed),
        )?;
        binary(builder, &self.residual, x, out, hidden, elem_team(workers))?;
        self.lower_qwen_mlp(builder, w, workers)?;
        builder.layer_end(layer);
        Ok(())
    }
}

impl Scheduler {
    /// LiquidAI's gated short convolution (`lfm2`, `lfm2_moe`) in the DeltaNet's buffers and conv
    /// state (`DeltaWeights::short_conv`), then the layer's MLP or experts: `xn = norm(x)`; `B`
    /// (into `dn_z`), `C` (into `dn_out`) and `x` (into `dn_qkv`), f32; the conv over the
    /// block's rows, sixteen channels a task, writing `bf16(C · conv(bf16(B · x)))` into
    /// `dn_out_bf16`; `out_proj`; the residual.
    fn lower_short_conv_layer(
        &mut self,
        builder: &mut Builder,
        w: &LayerWeights<'_>,
        mount: DeltaMount,
        layer: usize,
        workers: usize,
    ) -> std::result::Result<(), ExecutionError> {
        let Mixer::GatedDeltaNet(dw) = &w.mixer else {
            return Err(ExecutionError("layer is not a short convolution".into()));
        };
        let s = self.shape;
        let (hidden, ck) = (s.hidden, dw.conv_kernel);
        if w.intermediate > s.intermediate
            || dw.head_v_dim != hidden
            || dw.conv1d_weight.len() < hidden * ck
            || mount.conv.is_null()
        {
            return Err(ExecutionError(
                "generated short convolution weights, state, or layer geometry mismatch".into(),
            ));
        }
        let conv_kernel = Native::short_conv(hidden, ck)?;
        let par_m1 = crate::jit::options::decode().pool_m1;
        let b = &mut self.buffers;
        let packed = Buffer::of(&mut b.packed);
        let x = Buffer::of(&mut b.x);
        let xn = Buffer::of(&mut b.x_norm);
        let out = Buffer::of(&mut b.proj_out);
        let xs = F32Buffer::of(&mut b.dn_qkv);
        let bs = F32Buffer::of(&mut b.dn_z);
        let cs = F32Buffer::of(&mut b.dn_out);
        xs.check(builder.batch, hidden)?;
        bs.check(builder.batch, hidden)?;
        cs.check(builder.batch, hidden)?;
        if b.dn_out_bf16.len() < builder.batch * hidden {
            return Err(ExecutionError(
                "mounted short convolution reserve is too short".into(),
            ));
        }

        builder.begin_layer(layer);
        norm(
            builder,
            &self.norm_hidden,
            x,
            w.input_norm,
            xn,
            hidden,
            s.eps,
            elem_team(workers),
        )?;
        projections_f32(
            builder,
            xn,
            &[
                ProjectionF32 {
                    weights: dw.in_proj_z,
                    output: bs,
                    n: hidden,
                },
                ProjectionF32 {
                    weights: dw.in_proj_c,
                    output: cs,
                    n: hidden,
                },
                ProjectionF32 {
                    weights: dw.in_proj_qkv,
                    output: xs,
                    n: hidden,
                },
            ],
            hidden,
            workers,
            par_m1,
            Pack::Into(packed),
        )?;
        let call = builder.keep(ex::abi::ShortConvCall {
            b: bs.ptr,
            c: cs.ptr,
            x: xs.ptr,
            weights: dw.conv1d_weight.as_ptr(),
            state: mount.conv,
            output: b.dn_out_bf16.as_mut_ptr(),
            rows: &raw const builder.frame.rows,
        });
        builder.label("short-convolution");
        builder.grid(
            conv_kernel,
            binding(&[(call as u64, 0), (0, 1)], &[]),
            TaskGrid {
                rows_per_task: builder.batch,
                tasks_per_tile: hidden / 16,
                last_block_only: false,
            },
            workers,
            if par_m1 { workers } else { 1 },
        )?;
        projections(
            builder,
            Buffer::of(&mut b.dn_out_bf16),
            &[Projection {
                weights: dw.out_proj,
                output: out,
                n: hidden,
            }],
            hidden,
            workers,
            par_m1,
            Pack::Into(packed),
        )?;
        mixer_residual(
            builder,
            &self.norm_hidden,
            &self.residual,
            x,
            out,
            xn,
            w,
            hidden,
            s.eps,
            elem_team(workers),
        )?;
        self.lower_qwen_mlp(builder, w, workers)?;
        builder.layer_end(layer);
        Ok(())
    }
}

/// The vector Mamba2 recurrence the knobs ask for (the state's format and the FMA of the
/// DeltaNet's).
fn ex_recurrence_spec(hvd: usize) -> crate::jit::emit_recurrence::RecurrenceSpec {
    crate::jit::emit_recurrence::RecurrenceSpec {
        mamba: true,
        ..crate::jit::recurrence::spec_of_knobs(hvd)
    }
}

impl Scheduler {
    /// Qwen3-VL's deepstack (`Qwen3VLMoeTextModel._deepstack_process`), after layer `layer`: `x =
    /// bf16(x + d)` on every row of the block, `d` the layer's slice of
    /// [`Scheduler::deepstack_mut`]. Lowered in an image block only.
    pub(crate) fn lower_deepstack(
        &mut self,
        builder: &mut Builder,
        layer: usize,
        workers: usize,
    ) -> std::result::Result<(), ExecutionError> {
        let hidden = self.shape.hidden;
        let rows = self.max_batch * hidden;
        if layer >= self.shape.deepstack_layers || self.deepstack.len() < (layer + 1) * rows {
            return Err(ExecutionError(
                "deepstack layer beyond the reserved vectors".into(),
            ));
        }
        let x = Buffer::of(&mut self.buffers.x);
        let d = Buffer::of(&mut self.deepstack[layer * rows..(layer + 1) * rows]);
        binary(builder, &self.residual, x, d, hidden, elem_team(workers))
    }

    /// The MLP half of a Qwen layer, after the mixer's residual: `xn = norm(x)`, the dense SwiGLU
    /// MLP (on a sparse `qwen3_5_moe` layer, the shared expert) into `proj_out`, then either `x
    /// += proj_out` or the expert block (`Qwen3_5MoeSparseMoeBlock`): the router (bf16, sixteen
    /// experts per GEMM block) on `xn`, the route ([`ex::abi::RouteRule::Qwen`]), the `top_k`
    /// experts' `[gate | up]`, SwiGLU and `down`, and `qwen_moe_out`, which adds the experts,
    /// the shared expert gated by `σ(shared_gate · xn)` and the residual. One row (the
    /// decode, or a row-by-row prefill): the router writes the weight pointers of the `top_k` slot
    /// nodes. A block of rows (x86-64 and AArch64, int8 / 4-bit experts): the rows grouped by
    /// expert, one GEMM per expert over its rows.
    fn lower_qwen_mlp(
        &mut self,
        builder: &mut Builder,
        w: &LayerWeights<'_>,
        workers: usize,
    ) -> std::result::Result<(), ExecutionError> {
        let s = self.shape;
        let (hidden, inter) = (s.hidden, w.intermediate);
        if inter > s.intermediate {
            return Err(ExecutionError(
                "mounted layer exceeds scheduler geometry".into(),
            ));
        }
        let find_act = |dim: usize| {
            self.swiglu
                .iter()
                .find(|(a, d, _)| *a == w.activation && *d == dim)
                .map(|(_, _, k)| Arc::clone(k))
                .ok_or_else(|| ExecutionError("missing mounted MLP activation".into()))
        };
        // No dense MLP (Granite 3): only the experts.
        let dense = inter != 0;
        let activation = if dense { Some(find_act(inter)?) } else { None };
        let moe = match (&w.qwen_moe, s.qwen_moe) {
            (Some(m), Some(qs)) => Some((m, qs, find_act(qs.moe.inter)?)),
            (None, _) => None,
            (Some(_), None) => {
                return Err(ExecutionError(
                    "a layer with experts in a model shape without them".into(),
                ))
            }
        };
        let par_m1 = crate::jit::options::decode().pool_m1;
        let team = elem_team(workers);
        let team_all = (workers, if par_m1 { workers } else { 1 });
        let b = &mut self.buffers;
        let packed = Buffer::of(&mut b.packed);
        let x = Buffer::of(&mut b.x);
        let xn = Buffer::of(&mut b.x_norm);
        let out = Buffer::of(&mut b.proj_out);
        let gate = Buffer::of(&mut b.gate);
        let up = Buffer::of(&mut b.up);
        norm(
            builder,
            &self.norm_hidden,
            x,
            w.post_attn_norm,
            xn,
            hidden,
            s.eps,
            team,
        )?;
        if let Some(activation) = &activation {
            projections(
                builder,
                xn,
                &[
                    Projection {
                        weights: w.w_gate,
                        output: gate,
                        n: inter,
                    },
                    Projection {
                        weights: w.w_up,
                        output: up,
                        n: inter,
                    },
                ],
                hidden,
                workers,
                par_m1,
                Pack::Into(packed),
            )?;
            binary(builder, activation, gate, up, inter, (workers, 1))?;
            projections(
                builder,
                gate,
                &[Projection {
                    weights: w.w_down,
                    output: out,
                    n: hidden,
                }],
                inter,
                workers,
                par_m1,
                Pack::Into(packed),
            )?;
        }
        let Some((m, qs, act_moe)) = moe else {
            // LFM3's sandwich norm: `x += post_mlp_norm(out)`, the norm into `xn` (read by `gate`
            // and `up` already).
            if let Some(l) = w.lfm3.as_ref().filter(|l| !l.post_mlp_norm.is_empty()) {
                norm(
                    builder,
                    &self.norm_hidden,
                    out,
                    l.post_mlp_norm,
                    xn,
                    hidden,
                    s.eps,
                    team,
                )?;
                return binary(builder, &self.residual, x, xn, hidden, team);
            }
            return binary(builder, &self.residual, x, out, hidden, team);
        };
        if w.lfm3.as_ref().is_some_and(|l| !l.post_mlp_norm.is_empty()) {
            return Err(ExecutionError(
                "a sandwich norm after experts is not served".into(),
            ));
        }
        if !dense && self.zeros.len() < builder.batch * hidden {
            return Err(ExecutionError("mounted zero rows too short".into()));
        }
        let MoeShape {
            experts: e_n,
            top_k,
            inter: mi,
        } = qs.moe;
        let rule = match qs.lfm_scale {
            Some(scale) => ex::abi::RouteRule::Lfm {
                norm: qs.norm_topk,
                scale_bits: scale.to_bits(),
            },
            None => ex::abi::RouteRule::Qwen { norm: qs.norm_topk },
        };
        // LiquidAI's route reads its bias per expert where Gemma's reads its scale.
        if qs.lfm_scale.is_some() && m.expert_bias.len() != e_n {
            return Err(ExecutionError(
                "mounted lfm2_moe expert bias of unexpected length".into(),
            ));
        }
        let per_expert: *const f32 = if qs.lfm_scale.is_some() {
            m.expert_bias.as_ptr()
        } else {
            std::ptr::null()
        };
        let scores = Buffer::of(&mut b.moe_scores);
        let gate_up_rows = Buffer::of(&mut b.moe_gate_up);
        let act_rows = Buffer::of(&mut b.moe_act);
        let down_rows = Buffer::of(&mut b.moe_down);
        let mixture = b.moe_weights.as_mut_ptr();
        if m.router.len() * 16 != e_n
            || m.gate_up.len() != e_n
            || m.down.len() != e_n
            || top_k > ex::abi::MAX_TOP_K
            || m.shared_gate.len() < hidden
        {
            return Err(ExecutionError(
                "mounted mixture-of-experts geometry mismatch".into(),
            ));
        }
        let rows = builder.batch;
        scores.check(rows, e_n)?;
        gate_up_rows.check(rows * top_k, 2 * mi)?;
        act_rows.check(rows * top_k, mi)?;
        down_rows.check(rows * top_k, hidden)?;
        if b.moe_weights.len() < rows * top_k {
            return Err(ExecutionError(
                "mounted mixture-of-experts weights buffer is too short".into(),
            ));
        }
        // The router: sixteen experts per GEMM block, block `j` written as `[rows][16]` at `j ·
        // rows · 16`.
        for (group, blocks) in m.router.chunks(4).enumerate() {
            let outputs: Vec<Projection<'_>> = blocks
                .iter()
                .enumerate()
                .map(|(i, &weights)| {
                    let at = (group * 4 + i) * rows * 16;
                    Projection {
                        weights,
                        output: Buffer {
                            ptr: unsafe { scores.ptr.add(at) },
                            len: rows * 16,
                        },
                        n: 16,
                    }
                })
                .collect();
            projections(
                builder,
                xn,
                &outputs,
                hidden,
                workers,
                par_m1,
                Pack::Into(packed),
            )?;
        }
        let gu_shape = GemmShape {
            k: hidden,
            n: 2 * mi,
        };
        let dn_shape = GemmShape { k: mi, n: hidden };
        gu_shape.check().map_err(err)?;
        dn_shape.check().map_err(err)?;
        let compile = |shape: GemmShape| -> std::result::Result<_, ExecutionError> {
            let plan = crate::jit::arch::gemm_plan(shape, rows);
            crate::jit::gemm::get_or_compile(
                shape,
                plan.tile,
                plan.opts,
                true,
                crate::jit::emit::AFormat::Bf16,
            )
            .map_err(err)?
            .ok_or_else(|| ExecutionError("mounted expert kernel refused".into()))
        };
        let gu_kernel = compile(gu_shape)?;
        let dn_kernel = compile(dn_shape)?;

        let pos: *const u32 = if rows > 1 {
            // A block of rows (the prefill): the rows grouped by expert, one GEMM per expert over
            // its rows.
            if !crate::jit::int8::gemm(gu_shape.n, gu_shape.k)
                || !crate::jit::int8::gemm(dn_shape.n, dn_shape.k)
            {
                return Err(ExecutionError("the mixture-of-experts block of a block of rows needs the int8 or 4-bit experts (--int8, --q4)".into()));
            }
            if b.moe_pos.len() < rows * top_k
                || b.moe_ids.len() < rows * top_k
                || b.moe_gathered.len() * 2 < crate::jit::int8::a_bytes(rows * top_k, hidden)
                || b.moe_act_q.len() * 2 < crate::jit::int8::a_bytes(rows * top_k, mi)
            {
                return Err(ExecutionError(
                    "mounted mixture-of-experts block buffers are too short".into(),
                ));
            }
            // The expert input (`xn`) quantized once for every row; the router gathers its rows by
            // expert.
            let q_in = quantized_input(builder, xn, hidden, Pack::Into(packed), workers, par_m1)?;
            let zeros = vec![0u64; e_n];
            let row_counts = builder.keep_slice(&zeros);
            let cursor = builder.keep_slice(&zeros);
            let gu_grids = builder.keep_slice(&zeros);
            let dn_grids = builder.keep_slice(&zeros);
            let total = builder.keep(0usize);
            let gathered = b.moe_gathered.as_mut_ptr().cast::<u8>();
            let act_q = b.moe_act_q.as_mut_ptr().cast::<u8>();
            let call_of = |builder: &mut Builder, weights: &[u16], e: usize| {
                builder.keep(ex::abi::Bf16Call {
                    activations: std::ptr::null(),
                    weights: weights.as_ptr(),
                    output: std::ptr::null_mut(),
                    rows: 0,
                    row_count: unsafe { row_counts.add(e) }.cast::<usize>(),
                })
            };
            let gu_calls: Vec<*mut ex::abi::Bf16Call> = (0..e_n)
                .map(|e| call_of(builder, m.gate_up[e], e))
                .collect();
            let dn_calls: Vec<*mut ex::abi::Bf16Call> =
                (0..e_n).map(|e| call_of(builder, m.down[e], e)).collect();
            let gu_table =
                builder.keep_slice(&gu_calls.iter().map(|&c| c as u64).collect::<Vec<_>>());
            let dn_table =
                builder.keep_slice(&dn_calls.iter().map(|&c| c as u64).collect::<Vec<_>>());
            let route_call = builder.keep(ex::abi::RouteRowsCall {
                scores: scores.ptr,
                per_expert_scale: per_expert,
                frame: &raw const *builder.frame,
                weights_out: mixture,
                pos_out: b.moe_pos.as_mut_ptr(),
                ids: b.moe_ids.as_mut_ptr(),
                row_counts,
                cursor,
                source: q_in,
                gathered,
                total: total.cast::<u64>(),
                gate_up_calls: gu_table,
                down_calls: dn_table,
                gate_up_grids: gu_grids,
                down_grids: dn_grids,
                gate_up_out: gate_up_rows.ptr.cast(),
                down_in: act_q,
                down_out: down_rows.ptr.cast(),
            });
            let spec = ex::abi::RouteRowsSpec {
                experts: e_n,
                top_k,
                max_rows: rows,
                a_row: crate::jit::int8::a_row_bytes(hidden),
                gate_up_out_row: 2 * mi * 2,
                down_in_row: crate::jit::int8::a_row_bytes(mi),
                down_out_row: hidden * 2,
                gate_up_tiles: (gu_kernel.tile().mr, gu_kernel.n_tile_count()),
                down_tiles: (dn_kernel.tile().mr, dn_kernel.n_tile_count()),
                rule,
            };
            builder.label("moe-route-rows");
            builder.once(
                Native::moe_route_rows(spec)?,
                binding(&[(route_call as u64, 0)], &[]),
            )?;
            let grid_index = |builder: &Builder| {
                builder
                    .last_grid()
                    .ok_or_else(|| ExecutionError("an expert GEMM node without a task grid".into()))
            };
            for (e, &call) in gu_calls.iter().enumerate() {
                builder.label("moe-gate-up");
                builder.grid(
                    Native::bf16_tile_selector(&gu_kernel)?,
                    binding(&[(call as u64, 0), (0, 1)], &[]),
                    TaskGrid {
                        rows_per_task: gu_kernel.tile().mr,
                        tasks_per_tile: gu_kernel.n_tile_count(),
                        last_block_only: false,
                    },
                    team_all.0,
                    team_all.1,
                )?;
                unsafe { *gu_grids.add(e) = grid_index(builder)? as u64 };
            }
            // The activation of every gathered row: `silu(gate) · up`.
            builder.label("moe-activation");
            builder.grid(
                Native::elementary(&act_moe)?,
                binding(
                    &[
                        (gate_up_rows.ptr as u64, 2 * mi * 2),
                        (unsafe { gate_up_rows.ptr.add(mi) } as u64, 2 * mi * 2),
                        (act_rows.ptr as u64, mi * 2),
                    ],
                    &[],
                ),
                TaskGrid {
                    rows_per_task: 1,
                    tasks_per_tile: top_k,
                    last_block_only: false,
                },
                team_all.0,
                team_all.1,
            )?;
            let call = builder.keep(ex::abi::PackCall {
                source: act_rows.ptr,
                dest: act_q.cast(),
                rows: total,
            });
            builder.label("moe-quantize");
            builder.grid(
                Native::int8_quantize(mi)?,
                binding(&[(call as u64, 0), (0, 1)], &[]),
                TaskGrid {
                    rows_per_task: 1,
                    tasks_per_tile: top_k,
                    last_block_only: false,
                },
                team_all.0,
                team_all.1,
            )?;
            for (e, &call) in dn_calls.iter().enumerate() {
                builder.label("moe-down");
                builder.grid(
                    Native::bf16_tile_selector(&dn_kernel)?,
                    binding(&[(call as u64, 0), (0, 1)], &[]),
                    TaskGrid {
                        rows_per_task: dn_kernel.tile().mr,
                        tasks_per_tile: dn_kernel.n_tile_count(),
                        last_block_only: false,
                    },
                    team_all.0,
                    team_all.1,
                )?;
                unsafe { *dn_grids.add(e) = grid_index(builder)? as u64 };
            }
            b.moe_pos.as_ptr()
        } else {
            // One row: the experts' weights are chosen at run time, the router writes the weight
            // pointer of each slot node's call struct before the slots run. The structs
            // come first, so the router knows them.
            let gu_table = builder.keep_slice(
                &m.gate_up
                    .iter()
                    .map(|w| w.as_ptr() as u64)
                    .collect::<Vec<_>>(),
            );
            let dn_table =
                builder.keep_slice(&m.down.iter().map(|w| w.as_ptr() as u64).collect::<Vec<_>>());
            let gu_quantized = crate::jit::int8::gemm(gu_shape.n, hidden);
            let dn_quantized = crate::jit::int8::gemm(dn_shape.n, mi);
            if dn_quantized && crate::jit::int8::a_bytes(top_k, mi) > packed.len * 2 {
                return Err(ExecutionError(
                    "mounted int8 activation buffer is too short for the experts".into(),
                ));
            }
            let dn_row_bytes = if dn_quantized {
                crate::jit::int8::a_row_bytes(mi)
            } else {
                2 * mi
            };
            let dn_base: *const u8 = if dn_quantized {
                packed.ptr.cast()
            } else {
                act_rows.ptr.cast()
            };
            let rows_ptr: *const usize = &raw const builder.frame.rows;
            let mut gu_calls = Vec::with_capacity(top_k);
            let mut dn_calls = Vec::with_capacity(top_k);
            for k in 0..top_k {
                gu_calls.push(builder.keep(ex::abi::Bf16Call {
                    activations: xn.ptr.cast(),
                    weights: m.gate_up[0].as_ptr(),
                    output: unsafe { gate_up_rows.ptr.add(k * 2 * mi) }.cast(),
                    rows: 0,
                    row_count: rows_ptr,
                }));
                dn_calls.push(builder.keep(ex::abi::Bf16Call {
                    activations: unsafe { dn_base.add(k * dn_row_bytes) },
                    weights: m.down[0].as_ptr(),
                    output: unsafe { down_rows.ptr.add(k * hidden) }.cast(),
                    rows: 0,
                    row_count: rows_ptr,
                }));
            }
            let weights_field = std::mem::offset_of!(ex::abi::Bf16Call, weights);
            let mut route_call = ex::abi::RouteCall {
                scores: scores.ptr,
                per_expert_scale: per_expert,
                weights_out: mixture,
                gate_up_table: gu_table,
                down_table: dn_table,
                gate_up_slots: [std::ptr::null_mut(); ex::abi::MAX_TOP_K],
                down_slots: [std::ptr::null_mut(); ex::abi::MAX_TOP_K],
            };
            for k in 0..top_k {
                route_call.gate_up_slots[k] =
                    unsafe { (gu_calls[k] as *mut u8).add(weights_field) }.cast();
                route_call.down_slots[k] =
                    unsafe { (dn_calls[k] as *mut u8).add(weights_field) }.cast();
            }
            let route_call = builder.keep(route_call);
            builder.label("moe-route");
            builder.once(
                Native::moe_route_rule(e_n, top_k, rule)?,
                binding(&[(route_call as u64, 0)], &[]),
            )?;
            let gu_in = if gu_quantized {
                // The quantized expert input is read by the calls above: the buffer is the same for
                // every slot.
                quantized_input(builder, xn, hidden, Pack::Into(packed), workers, par_m1)?
            } else {
                xn.ptr.cast()
            };
            for &call in &gu_calls {
                unsafe { (*call).activations = gu_in };
            }
            for &call in &gu_calls {
                builder.label("moe-gate-up");
                builder.grid(
                    Native::bf16_tile_selector(&gu_kernel)?,
                    binding(&[(call as u64, 0), (0, 1)], &[]),
                    TaskGrid {
                        rows_per_task: gu_kernel.tile().mr,
                        tasks_per_tile: gu_kernel.n_tile_count(),
                        last_block_only: false,
                    },
                    team_all.0,
                    team_all.1,
                )?;
            }
            // The activation of every chosen expert: `silu(gate) · up`, one task per slot.
            builder.label("moe-activation");
            builder.grid(
                Native::elementary(&act_moe)?,
                binding(
                    &[
                        (gate_up_rows.ptr as u64, 2 * mi * 2),
                        (unsafe { gate_up_rows.ptr.add(mi) } as u64, 2 * mi * 2),
                        (act_rows.ptr as u64, mi * 2),
                    ],
                    &[],
                ),
                TaskGrid {
                    rows_per_task: builder.batch,
                    tasks_per_tile: top_k,
                    last_block_only: false,
                },
                team_all.0,
                team_all.1,
            )?;
            if dn_quantized {
                let count = builder.keep(top_k);
                let call = builder.keep(ex::abi::PackCall {
                    source: act_rows.ptr,
                    dest: packed.ptr,
                    rows: count,
                });
                builder.label("moe-quantize");
                builder.grid(
                    Native::int8_quantize(mi)?,
                    binding(&[(call as u64, 0), (0, 1)], &[]),
                    TaskGrid {
                        rows_per_task: builder.batch,
                        tasks_per_tile: top_k,
                        last_block_only: false,
                    },
                    team_all.0,
                    team_all.1,
                )?;
            }
            for &call in &dn_calls {
                builder.label("moe-down");
                builder.grid(
                    Native::bf16_tile_selector(&dn_kernel)?,
                    binding(&[(call as u64, 0), (0, 1)], &[]),
                    TaskGrid {
                        rows_per_task: dn_kernel.tile().mr,
                        tasks_per_tile: dn_kernel.n_tile_count(),
                        last_block_only: false,
                    },
                    team_all.0,
                    team_all.1,
                )?;
            }
            // The slots in order: the route wrote them in increasing expert order.
            let words: Vec<u64> = (0..top_k as u64)
                .step_by(2)
                .map(|i| i | ((i + 1) << 32))
                .collect();
            builder.keep_slice(&words).cast::<u32>()
        };
        let call = builder.keep(ex::abi::QwenOutCall {
            rows: down_rows.ptr,
            pos,
            weights: mixture,
            xn: xn.ptr,
            gate: m.shared_gate.as_ptr(),
            // Without a shared expert, rows of zeros under a gate of zeros: `bf16(σ(0) · 0) = 0` is
            // added.
            shared: if dense { out.ptr } else { self.zeros.as_ptr() },
            x: x.ptr,
        });
        builder.label("moe-qwen-out");
        builder.rows(
            Native::moe_qwen_out(hidden, top_k)?,
            binding(&[(call as u64, 0), (0, 1)], &[]),
            team_all.0,
            team_all.1,
        )?;
        builder.int8_ready = false;
        Ok(())
    }
}

impl Scheduler {
    /// The token embedding, once per generated block: one row of the table
    /// copied per token. Token ids are validated against the vocabulary before
    /// the plan is launched.
    pub(crate) fn lower_input(
        &mut self,
        builder: &mut Builder,
        embed: &[u16],
    ) -> std::result::Result<(), ExecutionError> {
        let s = self.shape;
        let hidden = s.hidden;
        if embed.is_empty() || !embed.len().is_multiple_of(hidden) || builder.batch > self.max_batch
        {
            return Err(ExecutionError("mounted embedding geometry mismatch".into()));
        }
        let b = &mut self.buffers;
        let mut args = binding(
            &[
                (0, 4),
                (embed.as_ptr() as u64, 0),
                (b.x.as_mut_ptr() as u64, hidden * 2),
                (hidden as u64, 0),
            ],
            &[],
        );
        args.argument_bases[0] = (&raw const builder.frame.token_cursor).cast();
        builder.label("token-embedding");
        builder.rows(Native::embedding()?, args, 1, 1)?;
        Ok(())
    }
}

impl Scheduler {
    /// Final-row norm, language projection, configured logit transforms and
    /// deterministic argmax. These nodes execute only on the last token block.
    pub(crate) fn lower_output(
        &mut self,
        builder: &mut Builder,
        norm_weight: &[u16],
        head: &[u16],
        logits: &mut [u16],
        true_vocab: usize,
        argmax: *mut u32,
        workers: usize,
    ) -> std::result::Result<(), ExecutionError> {
        let hidden = self.shape.hidden;
        let vocab = logits.len();
        if norm_weight.len() < hidden || true_vocab == 0 || true_vocab > vocab {
            return Err(ExecutionError(
                "invalid mounted language head geometry".into(),
            ));
        }
        builder.final_only(true);
        let rows = &raw const builder.frame.rows;
        let last = offset(builder, rows, hidden * 2, 1)?;
        let b = &mut self.buffers;
        let mut args = binding(
            &[
                (b.x.as_ptr() as u64, 0),
                (norm_weight.as_ptr() as u64, 0),
                (b.x_norm.as_mut_ptr() as u64, 0),
            ],
            &[self.shape.eps],
        );
        args.argument_bases[0] = last;
        builder.label("final-norm");
        builder.once(Native::elementary(&self.norm_hidden)?, args)?;
        {
            let shape = GemmShape {
                k: hidden,
                n: vocab,
            };
            shape.check().map_err(err)?;
            if head.len() < crate::jit::int8::weight_len(shape.n, shape.k) {
                return Err(ExecutionError("mounted R16 language head mismatch".into()));
            }
            // One row: the vector plan of the target.
            let plan = crate::jit::arch::gemm_plan(shape, 1);
            let kernel = crate::jit::gemm::get_or_compile(
                shape,
                plan.tile,
                plan.opts,
                true,
                crate::jit::emit::AFormat::Bf16,
            )
            .map_err(err)?
            .ok_or_else(|| ExecutionError("mounted language kernel refused".into()))?;
            // `--int8`: the one row of the head is quantized first (into the packing buffer).
            let head_activations: *const u8 = if crate::jit::int8::gemm(vocab, hidden) {
                let packed = Buffer::of(&mut b.packed);
                if crate::jit::int8::a_bytes(1, hidden) > packed.len * 2 {
                    return Err(ExecutionError(
                        "mounted int8 activation buffer is too short".into(),
                    ));
                }
                let one = builder.keep(1usize);
                let quantize = builder.keep(ex::abi::PackCall {
                    source: b.x_norm.as_ptr(),
                    dest: packed.ptr,
                    rows: one,
                });
                builder.label("quantize-head-activations");
                builder.grid(
                    Native::int8_quantize(hidden)?,
                    binding(&[(quantize as u64, 0), (0, 1)], &[]),
                    TaskGrid {
                        rows_per_task: builder.batch,
                        tasks_per_tile: 1,
                        last_block_only: true,
                    },
                    workers,
                    workers,
                )?;
                packed.ptr.cast()
            } else {
                b.x_norm.as_ptr().cast()
            };
            let call = builder.keep(ex::abi::Bf16Call {
                activations: head_activations,
                weights: head.as_ptr(),
                output: logits.as_mut_ptr().cast(),
                rows: 1,
                row_count: std::ptr::null(),
            });
            builder.label("language-head-bf16");
            builder.grid(
                Native::bf16_tile_selector(&kernel)?,
                binding(&[(call as u64, 0), (0, 1)], &[]),
                TaskGrid {
                    rows_per_task: builder.batch,
                    tasks_per_tile: kernel.total_items(1),
                    last_block_only: true,
                },
                workers,
                workers,
            )?;
        }
        // Gemma 4: the logits are soft-capped, `cap · tanh(logit / cap)`, in slices over the team.
        if let Some(cap) = self.shape.gemma.map(|g| g.softcap).filter(|&c| c > 0.0) {
            let len = logits.len();
            let chunk = if len.is_multiple_of(4096) { 4096 } else { len };
            let kernel = self.elem(ElemOp::Softcap, chunk)?;
            builder.label("logits-softcap");
            builder.grid(
                Native::elementary(&kernel)?,
                binding(
                    &[
                        (logits.as_ptr() as u64, chunk * 2),
                        (logits.as_ptr() as u64, chunk * 2),
                    ],
                    &[cap],
                ),
                TaskGrid {
                    rows_per_task: builder.batch,
                    tasks_per_tile: len / chunk,
                    last_block_only: true,
                },
                workers,
                workers,
            )?;
        }
        let slices = crate::jit::knobs::knobs().argmax_tasks.min(64);
        if slices > 0 {
            let slots = builder.keep([0u64; 64]) as u64;
            builder.label("argmax-slices");
            builder.grid(
                Native::argmax_slices()?,
                binding(
                    &[
                        (logits.as_ptr() as u64, 0),
                        (slots, 0),
                        (true_vocab as u64, 0),
                        (0, 1),
                        (slices as u64, 0),
                    ],
                    &[],
                ),
                TaskGrid {
                    rows_per_task: builder.batch,
                    tasks_per_tile: slices,
                    last_block_only: true,
                },
                workers,
                workers,
            )?;
            builder.label("argmax");
            builder.once(
                Native::argmax_merge()?,
                binding(&[(slots, 0), (argmax as u64, 0), (slices as u64, 0)], &[]),
            )?;
        } else {
            builder.label("argmax");
            builder.once(
                Native::argmax_bf16()?,
                binding(
                    &[
                        (logits.as_ptr() as u64, 0),
                        (argmax as u64, 0),
                        (true_vocab as u64, 0),
                    ],
                    &[],
                ),
            )?;
        }
        builder.final_only(false);
        Ok(())
    }
}

impl Scheduler {
    /// The elementwise kernel `op` over `dim` values, compiled now (mount time).
    fn elem(
        &self,
        op: ElemOp,
        dim: usize,
    ) -> std::result::Result<Arc<CompiledElem>, ExecutionError> {
        elem_kernel(ElemSpec { op, dim })
            .map_err(err)?
            .ok_or_else(|| ExecutionError(format!("{} dim={dim} not schedulable", op.name())))
    }

    /// One Gemma 4 layer. The attention is `x += post_norm(o_proj(attention(norm(x))))`, the MLP
    /// `x += post_norm(down(gelu(gate(pre_norm(x))) · up(pre_norm(x))))`, then the per-layer input
    /// branch `x += norm(proj(gelu(gate(x)) · input_l))` and `x *= layer_scalar`. The attention
    /// scales by one, the queries and keys being normed per head; the values are normed without
    /// a weight; a sliding layer sees the last `window` keys; a layer that reads another
    /// layer's cache (`mount.write == false`) computes only the queries. `image`: the rows are
    /// the soft tokens of an image, which see one another whole on a sliding layer
    /// (the full layers stay causal).
    pub(crate) fn lower_gemma_layer(
        &mut self,
        builder: &mut Builder,
        w: &LayerWeights<'_>,
        mount: LayerMount,
        layer: usize,
        workers: usize,
        image: bool,
    ) -> std::result::Result<(), ExecutionError> {
        let (Some(g), Some(gs)) = (w.gemma.as_ref(), self.shape.gemma) else {
            return Err(ExecutionError(
                "a Gemma layer without its Gemma weights".into(),
            ));
        };
        if builder.batch > self.max_batch || !matches!(w.mixer, Mixer::Attention) {
            return Err(ExecutionError(
                "layer is not supported by the Gemma lowering".into(),
            ));
        }
        let s = self.shape;
        let sl = s.for_layer(
            w.head_dim,
            w.rotary_ndims,
            w.n_kv_heads.unwrap_or(s.n_kv_heads),
        );
        let (hidden, hd, nh, nkv, inter) = (
            s.hidden,
            sl.head_dim,
            s.n_heads,
            sl.n_kv_heads,
            w.intermediate,
        );
        if hd > s.head_dim
            || nkv > s.n_kv_heads
            || inter > s.intermediate
            || sl.rotary_ndims > hd
            || nkv == 0
            || nh % nkv != 0
        {
            return Err(ExecutionError(
                "mounted layer exceeds scheduler geometry".into(),
            ));
        }
        let hk = self
            .heads
            .iter()
            .find(|k| k.head_dim == hd)
            .ok_or_else(|| ExecutionError("missing mounted head kernels".into()))?;
        let rope = self
            .ropes
            .iter()
            .find(|(rn, _)| *rn == w.rotary_ndims)
            .map(|(_, k)| k)
            .ok_or_else(|| ExecutionError("missing mounted RoPE kernel".into()))?;
        let activation = self
            .swiglu
            .iter()
            .find(|(a, d, _)| *a == w.activation && *d == inter)
            .map(|(_, _, k)| k)
            .ok_or_else(|| ExecutionError("missing mounted MLP activation".into()))?;
        let gelu_pl = self
            .swiglu
            .iter()
            .find(|(a, d, _)| *a == Activation::GeluTanh && *d == gs.pl_dim)
            .map(|(_, _, k)| k);
        let gk = self
            .gemma
            .as_ref()
            .ok_or_else(|| ExecutionError("missing the Gemma kernels".into()))?;
        let (qd, kvd, qrs, half) = (sl.q_dim(), sl.kv_dim(), sl.q_row_stride(), sl.rope_half());
        let (pl, lpl) = (gs.pl_dim, gs.n_layers * gs.pl_dim);
        let copy = if g.k_eq_v && mount.write {
            Some(self.elem(ElemOp::Scale, kvd)?)
        } else {
            None
        };
        builder.begin_layer(layer);
        let b = &mut self.buffers;
        let packed = Buffer::of(&mut b.packed);
        let x = Buffer::of(&mut b.x);
        let xn = Buffer::of(&mut b.x_norm);
        let q = Buffer::of(&mut b.q);
        let k = Buffer::of(&mut b.k);
        let v = Buffer::of(&mut b.v);
        let attn = Buffer::of(&mut b.attn);
        let out = Buffer::of(&mut b.proj_out);
        let gate = Buffer::of(&mut b.gate);
        let up = Buffer::of(&mut b.up);
        let fold = Buffer::of(&mut b.ple_fold);
        let gate_pl = Buffer::of(&mut b.ple_gate);
        fold.check(builder.batch, lpl)?;
        gate_pl.check(builder.batch, pl)?;
        let team = elem_team(workers);
        let par_m1 = crate::jit::options::decode().pool_m1;
        let kv_spec = ex::abi::KvSpec {
            heads: nkv,
            head_dim: hd,
            stride: mount.stride,
            capacity: mount.capacity,
            max_batch: builder.batch,
            r16: !mount.keys_r16.is_null(),
        };
        let kv_call = builder.keep(ex::abi::KvCall {
            keys: mount.keys,
            values: mount.values,
            input_keys: k.ptr,
            input_values: v.ptr,
            rows: &raw const builder.frame.rows,
            past: &raw const builder.frame.position,
            keys_r16: mount.keys_r16,
            values_r16: mount.values_r16,
        });
        norm(
            builder,
            &self.norm_hidden,
            x,
            w.input_norm,
            xn,
            hidden,
            s.eps,
            team,
        )?;
        let own = mount.write;
        let mut outputs = vec![Projection {
            weights: w.wq,
            output: q,
            n: sl.q_gemm_dim(),
        }];
        if own {
            outputs.push(Projection {
                weights: w.wk,
                output: k,
                n: kvd,
            });
            if !g.k_eq_v {
                outputs.push(Projection {
                    weights: w.wv,
                    output: v,
                    n: kvd,
                });
            }
        }
        projections(
            builder,
            xn,
            &outputs,
            hidden,
            workers,
            par_m1,
            Pack::Into(packed),
        )?;
        if let Some(copy) = &copy {
            // The values are the keys as the projection wrote them, before the key norm and the
            // RoPE.
            builder.label("kv-copy-k-to-v");
            builder.rows(
                Native::elementary(copy)?,
                binding(
                    &[(k.ptr as u64, kvd * 2), (v.ptr as u64, kvd * 2)],
                    &[1.0f32],
                ),
                attn_chain_team(workers).0,
                attn_chain_team(workers).1,
            )?;
        }
        let pos = &raw const builder.frame.position;
        let rope_offset = offset(builder, pos, half * 2, 0)?;
        for (data, weights, heads, stride, row_stride) in
            [(q, w.q_norm, nh, hd, qrs), (k, w.k_norm, nkv, hd, kvd)]
                .into_iter()
                .take(if own { 2 } else { 1 })
        {
            if weights.len() < hd {
                return Err(ExecutionError("mounted head norm weight too short".into()));
            }
            data.check(builder.batch, row_stride)?;
            let mut binding = binding(
                &[
                    (data.ptr as u64, row_stride * 2),
                    (weights.as_ptr() as u64, 0),
                    (mount.cos as u64, half * 2),
                    (mount.sin as u64, half * 2),
                ],
                &[s.eps],
            );
            binding.argument_bases[2] = rope_offset;
            binding.argument_bases[3] = rope_offset;
            builder.label("head-norm-rope");
            builder.rows(
                Native::norm_rope(Some(&hk.norm_head), rope, heads, stride)?,
                binding,
                attn_chain_team(workers).0,
                attn_chain_team(workers).1,
            )?;
        }
        if own {
            // The values are normed per head without a weight (ones).
            v.check(builder.batch, kvd)?;
            builder.label("value-head-norm");
            builder.grid(
                Native::elementary(&hk.norm_head)?,
                binding(
                    &[
                        (v.ptr as u64, hd * 2),
                        (self.ones.as_ptr() as u64, 0),
                        (v.ptr as u64, hd * 2),
                    ],
                    &[s.eps],
                ),
                TaskGrid {
                    rows_per_task: 1,
                    tasks_per_tile: nkv,
                    last_block_only: false,
                },
                attn_chain_team(workers).0,
                attn_chain_team(workers).1,
            )?;
            let mut append = binding(&[(kv_call as u64, 0), (0, 1)], &[]);
            append.tasks = nkv;
            builder.label("kv-append");
            builder.grid(
                Native::kv_append(kv_spec)?,
                append,
                TaskGrid {
                    rows_per_task: builder.batch,
                    tasks_per_tile: nkv,
                    last_block_only: false,
                },
                attn_chain_team(workers).0,
                attn_chain_team(workers).1,
            )?;
        }
        // The block kernel of the layer's head (`HeadKernels::sdpa_prefill`, causal), as on the
        // other models: a full layer (head 512 on Gemma 4) takes it as it is, a sliding
        // layer the one of its window, compiled here from the causal one (cached by spec;
        // AVX-512, AVX2 and NEON all have it). The rows of an image on a sliding layer see one
        // another whole inside a window (`whole_block` with a window): no block kernel has
        // that mask, they stay row by row; on a full layer they are causal and take the
        // block. A layer that reads another layer's cache reads it as its owner wrote it, the
        // block's rows included (the owner comes first), R16 copies too on AArch64.
        let whole = image && g.window > 0;
        let prefill: Option<Arc<CompiledSdpaPrefill>> = match hk.sdpa_prefill.as_ref() {
            Some(_) if whole => None,
            Some(p) if g.window > 0 => {
                crate::jit::sdpa::get_or_compile_prefill(crate::jit::emit_sdpa::SdpaPrefillSpec {
                    window: g.window,
                    ..p.spec()
                })
                .map_err(|e| ExecutionError(format!("windowed block attention: {e}")))?
            }
            p => p.cloned(),
        };
        if prefill
            .as_ref()
            .is_some_and(|p| p.scratch_len() > self.scores_stride)
        {
            return Err(ExecutionError(
                "windowed block attention: scratch longer than the score slots".into(),
            ));
        }
        attn_plan(layer, builder.batch, g.window, prefill.as_deref());
        let chunk = prefill.as_deref().map_or(1, CompiledSdpaPrefill::rows);
        let kv_task = cfg!(target_arch = "x86_64")
            && crate::jit::knobs::knobs().attn_kv_task
            && builder.batch > 1
            && prefill.is_some()
            && sl.gqa() > 1;
        if b.scores.len() / self.scores_stride < workers {
            return Err(ExecutionError(
                "mounted attention score slots too short".into(),
            ));
        }
        let call = builder.keep(ex::abi::AttentionCall {
            q: q.ptr,
            keys: mount.keys,
            values: mount.values,
            output: attn.ptr,
            rows: &raw const builder.frame.rows,
            past: &raw const builder.frame.position,
            kv_stride: mount.stride,
            scale: sl.scale(),
            sink: 0.0,
            keys_r16: mount.keys_r16,
            values_r16: mount.values_r16,
            segments: std::ptr::null(),
        });
        let mut args = binding(
            &[(call as u64, 0), (0, 1), (b.scores.as_mut_ptr() as u64, 0)],
            &[],
        );
        args.participant_strides[2] = (self.scores_stride * 4) as u64;
        builder.label("attention");
        builder.grid(
            Native::attention_tile_selector(
                &hk.sdpa,
                prefill.as_deref(),
                ex::abi::AttentionSpec {
                    heads: nh,
                    gqa: sl.gqa(),
                    head_dim: hd,
                    q_row: qrs,
                    q_head: hd,
                    out_row: qd,
                    grouped: super::pass::sdpa_item_order_grouped(),
                    kv_task,
                    window: g.window,
                    whole_block: whole,
                    ahead: 0,
                    segments: false,
                },
            )?,
            args,
            TaskGrid {
                rows_per_task: chunk,
                tasks_per_tile: if kv_task { nkv } else { nh },
                last_block_only: false,
            },
            workers,
            if par_m1 { workers } else { 1 },
        )?;
        projections(
            builder,
            attn,
            &[Projection {
                weights: w.wo,
                output: out,
                n: hidden,
            }],
            qd,
            workers,
            par_m1,
            Pack::Into(packed),
        )?;
        norm(
            builder,
            &self.norm_hidden,
            out,
            g.post_attn_out_norm,
            xn,
            hidden,
            s.eps,
            team,
        )?;
        binary(builder, &self.residual, x, xn, hidden, team)?;
        // The MLP.
        norm(
            builder,
            &self.norm_hidden,
            x,
            w.post_attn_norm,
            xn,
            hidden,
            s.eps,
            team,
        )?;
        projections(
            builder,
            xn,
            &[
                Projection {
                    weights: w.w_gate,
                    output: gate,
                    n: inter,
                },
                Projection {
                    weights: w.w_up,
                    output: up,
                    n: inter,
                },
            ],
            hidden,
            workers,
            par_m1,
            Pack::Into(packed),
        )?;
        binary(builder, activation, gate, up, inter, (workers, 1))?;
        projections(
            builder,
            gate,
            &[Projection {
                weights: w.w_down,
                output: out,
                n: hidden,
            }],
            inter,
            workers,
            par_m1,
            Pack::Into(packed),
        )?;
        if let (Some(m), Some(ms)) = (&g.moe, gs.moe) {
            // The experts run in parallel to the dense MLP: `h = dense + experts`, normed, added to
            // the residual.
            let act_moe = self
                .swiglu
                .iter()
                .find(|(a, d, _)| *a == Activation::GeluTanh && *d == ms.inter)
                .map(|(_, _, k)| k)
                .ok_or_else(|| ExecutionError("missing the expert activation".into()))?;
            let dense = Buffer::of(&mut b.moe_dense);
            let router_in = Buffer::of(&mut b.moe_router_in);
            let scores = Buffer::of(&mut b.moe_scores);
            let expert_in = Buffer::of(&mut b.moe_expert_in);
            let gate_up_rows = Buffer::of(&mut b.moe_gate_up);
            let act_rows = Buffer::of(&mut b.moe_act);
            let down_rows = Buffer::of(&mut b.moe_down);
            let sum = Buffer::of(&mut b.moe_sum);
            let mixture = b.moe_weights.as_mut_ptr();
            if m.router.len() * 16 != ms.experts
                || m.gate_up.len() != ms.experts
                || m.down.len() != ms.experts
                || ms.top_k > ex::abi::MAX_TOP_K
            {
                return Err(ExecutionError(
                    "mounted mixture-of-experts geometry mismatch".into(),
                ));
            }
            if builder.batch > 1 {
                // A block of rows (the prefill): the rows grouped by expert, one GEMM per expert
                // over its rows.
                let rows = builder.batch;
                let (e_n, top_k, inter) = (ms.experts, ms.top_k, ms.inter);
                for (buf, width) in [
                    (dense, hidden),
                    (router_in, hidden),
                    (expert_in, hidden),
                    (sum, hidden),
                ] {
                    buf.check(rows, width)?;
                }
                scores.check(rows, e_n)?;
                gate_up_rows.check(rows * top_k, 2 * inter)?;
                act_rows.check(rows * top_k, inter)?;
                down_rows.check(rows * top_k, hidden)?;
                if b.moe_weights.len() < rows * top_k
                    || b.moe_pos.len() < rows * top_k
                    || b.moe_ids.len() < rows * top_k
                    || b.moe_gathered.len() * 2 < crate::jit::int8::a_bytes(rows * top_k, hidden)
                    || b.moe_act_q.len() * 2 < crate::jit::int8::a_bytes(rows * top_k, inter)
                {
                    return Err(ExecutionError(
                        "mounted mixture-of-experts block buffers are too short".into(),
                    ));
                }
                norm(
                    builder,
                    &self.norm_hidden,
                    out,
                    m.post_ff_norm_1,
                    dense,
                    hidden,
                    s.eps,
                    team,
                )?;
                norm(
                    builder,
                    &self.norm_hidden,
                    x,
                    m.router_scale,
                    router_in,
                    hidden,
                    s.eps,
                    team,
                )?;
                // The router: sixteen experts per GEMM block, block `j` written as `[rows][16]` at
                // `j · rows · 16`.
                for (group, blocks) in m.router.chunks(4).enumerate() {
                    let outputs: Vec<Projection<'_>> = blocks
                        .iter()
                        .enumerate()
                        .map(|(i, &weights)| {
                            let at = (group * 4 + i) * rows * 16;
                            Projection {
                                weights,
                                output: Buffer {
                                    ptr: unsafe { scores.ptr.add(at) },
                                    len: rows * 16,
                                },
                                n: 16,
                            }
                        })
                        .collect();
                    projections(
                        builder,
                        router_in,
                        &outputs,
                        hidden,
                        workers,
                        par_m1,
                        Pack::Into(packed),
                    )?;
                }
                norm(
                    builder,
                    &self.norm_hidden,
                    x,
                    m.pre_ff_norm_2,
                    expert_in,
                    hidden,
                    s.eps,
                    team,
                )?;
                let gu_shape = GemmShape {
                    k: hidden,
                    n: 2 * inter,
                };
                let dn_shape = GemmShape {
                    k: inter,
                    n: hidden,
                };
                gu_shape.check().map_err(err)?;
                dn_shape.check().map_err(err)?;
                if !crate::jit::int8::gemm(gu_shape.n, gu_shape.k)
                    || !crate::jit::int8::gemm(dn_shape.n, dn_shape.k)
                {
                    return Err(ExecutionError("the mixture-of-experts block of a block of rows needs the int8 or 4-bit experts (--int8, --q4)".into()));
                }
                let compile = |shape: GemmShape| -> std::result::Result<_, ExecutionError> {
                    let plan = crate::jit::arch::gemm_plan(shape, rows);
                    crate::jit::gemm::get_or_compile(
                        shape,
                        plan.tile,
                        plan.opts,
                        true,
                        crate::jit::emit::AFormat::Bf16,
                    )
                    .map_err(err)?
                    .ok_or_else(|| ExecutionError("mounted expert kernel refused".into()))
                };
                let gu_kernel = compile(gu_shape)?;
                let dn_kernel = compile(dn_shape)?;
                // The expert input quantized once for every row; the router gathers its rows by
                // expert.
                let q_in = quantized_input(
                    builder,
                    expert_in,
                    hidden,
                    Pack::Into(packed),
                    workers,
                    par_m1,
                )?;
                let zeros = vec![0u64; e_n];
                let row_counts = builder.keep_slice(&zeros);
                let cursor = builder.keep_slice(&zeros);
                let gu_grids = builder.keep_slice(&zeros);
                let dn_grids = builder.keep_slice(&zeros);
                let total = builder.keep(0usize);
                let gathered = b.moe_gathered.as_mut_ptr().cast::<u8>();
                let act_q = b.moe_act_q.as_mut_ptr().cast::<u8>();
                let call_of = |builder: &mut Builder, weights: &[u16], e: usize| {
                    builder.keep(ex::abi::Bf16Call {
                        activations: std::ptr::null(),
                        weights: weights.as_ptr(),
                        output: std::ptr::null_mut(),
                        rows: 0,
                        row_count: unsafe { row_counts.add(e) }.cast::<usize>(),
                    })
                };
                let gu_calls: Vec<*mut ex::abi::Bf16Call> = (0..e_n)
                    .map(|e| call_of(builder, m.gate_up[e], e))
                    .collect();
                let dn_calls: Vec<*mut ex::abi::Bf16Call> =
                    (0..e_n).map(|e| call_of(builder, m.down[e], e)).collect();
                let gu_table =
                    builder.keep_slice(&gu_calls.iter().map(|&c| c as u64).collect::<Vec<_>>());
                let dn_table =
                    builder.keep_slice(&dn_calls.iter().map(|&c| c as u64).collect::<Vec<_>>());
                let route_call = builder.keep(ex::abi::RouteRowsCall {
                    scores: scores.ptr,
                    per_expert_scale: m.per_expert_scale.as_ptr(),
                    frame: &raw const *builder.frame,
                    weights_out: mixture,
                    pos_out: b.moe_pos.as_mut_ptr(),
                    ids: b.moe_ids.as_mut_ptr(),
                    row_counts,
                    cursor,
                    source: q_in,
                    gathered,
                    total: total.cast::<u64>(),
                    gate_up_calls: gu_table,
                    down_calls: dn_table,
                    gate_up_grids: gu_grids,
                    down_grids: dn_grids,
                    gate_up_out: gate_up_rows.ptr.cast(),
                    down_in: act_q,
                    down_out: down_rows.ptr.cast(),
                });
                let spec = ex::abi::RouteRowsSpec {
                    experts: e_n,
                    top_k,
                    max_rows: rows,
                    a_row: crate::jit::int8::a_row_bytes(hidden),
                    gate_up_out_row: 2 * inter * 2,
                    down_in_row: crate::jit::int8::a_row_bytes(inter),
                    down_out_row: hidden * 2,
                    gate_up_tiles: (gu_kernel.tile().mr, gu_kernel.n_tile_count()),
                    down_tiles: (dn_kernel.tile().mr, dn_kernel.n_tile_count()),
                    rule: ex::abi::RouteRule::Gemma,
                };
                builder.label("moe-route-rows");
                builder.once(
                    Native::moe_route_rows(spec)?,
                    binding(&[(route_call as u64, 0)], &[]),
                )?;
                let team_all = (workers, if par_m1 { workers } else { 1 });
                let grid_index = |builder: &Builder| {
                    builder.last_grid().ok_or_else(|| {
                        ExecutionError("an expert GEMM node without a task grid".into())
                    })
                };
                for (e, &call) in gu_calls.iter().enumerate() {
                    builder.label("moe-gate-up");
                    builder.grid(
                        Native::bf16_tile_selector(&gu_kernel)?,
                        binding(&[(call as u64, 0), (0, 1)], &[]),
                        TaskGrid {
                            rows_per_task: gu_kernel.tile().mr,
                            tasks_per_tile: gu_kernel.n_tile_count(),
                            last_block_only: false,
                        },
                        team_all.0,
                        team_all.1,
                    )?;
                    unsafe { *gu_grids.add(e) = grid_index(builder)? as u64 };
                }
                // The activation of every gathered row: `gelu(gate) · up`.
                builder.label("moe-activation");
                builder.grid(
                    Native::elementary(act_moe)?,
                    binding(
                        &[
                            (gate_up_rows.ptr as u64, 2 * inter * 2),
                            (unsafe { gate_up_rows.ptr.add(inter) } as u64, 2 * inter * 2),
                            (act_rows.ptr as u64, inter * 2),
                        ],
                        &[],
                    ),
                    TaskGrid {
                        rows_per_task: 1,
                        tasks_per_tile: top_k,
                        last_block_only: false,
                    },
                    team_all.0,
                    team_all.1,
                )?;
                let call = builder.keep(ex::abi::PackCall {
                    source: act_rows.ptr,
                    dest: act_q.cast(),
                    rows: total,
                });
                builder.label("moe-quantize");
                builder.grid(
                    Native::int8_quantize(inter)?,
                    binding(&[(call as u64, 0), (0, 1)], &[]),
                    TaskGrid {
                        rows_per_task: 1,
                        tasks_per_tile: top_k,
                        last_block_only: false,
                    },
                    team_all.0,
                    team_all.1,
                )?;
                for (e, &call) in dn_calls.iter().enumerate() {
                    builder.label("moe-down");
                    builder.grid(
                        Native::bf16_tile_selector(&dn_kernel)?,
                        binding(&[(call as u64, 0), (0, 1)], &[]),
                        TaskGrid {
                            rows_per_task: dn_kernel.tile().mr,
                            tasks_per_tile: dn_kernel.n_tile_count(),
                            last_block_only: false,
                        },
                        team_all.0,
                        team_all.1,
                    )?;
                    unsafe { *dn_grids.add(e) = grid_index(builder)? as u64 };
                }
                let combine = builder.keep(ex::abi::CombineRowsCall {
                    rows: down_rows.ptr,
                    pos: b.moe_pos.as_ptr(),
                    weights: mixture,
                    out: sum.ptr,
                });
                builder.label("moe-combine-rows");
                builder.rows(
                    Native::moe_combine_rows(hidden, top_k)?,
                    binding(&[(combine as u64, 0), (0, 1)], &[]),
                    team_all.0,
                    team_all.1,
                )?;
                builder.int8_ready = false;
            } else {
                norm(
                    builder,
                    &self.norm_hidden,
                    out,
                    m.post_ff_norm_1,
                    dense,
                    hidden,
                    s.eps,
                    team,
                )?;
                // The router: the residual normed with `router.scale · hidden^-½`, sixteen experts
                // per GEMM block.
                norm(
                    builder,
                    &self.norm_hidden,
                    x,
                    m.router_scale,
                    router_in,
                    hidden,
                    s.eps,
                    team,
                )?;
                for (group, blocks) in m.router.chunks(4).enumerate() {
                    let outputs: Vec<Projection<'_>> = blocks
                        .iter()
                        .enumerate()
                        .map(|(i, &weights)| {
                            let at = (group * 4 + i) * 16;
                            Projection {
                                weights,
                                output: Buffer {
                                    ptr: unsafe { scores.ptr.add(at) },
                                    len: 16,
                                },
                                n: 16,
                            }
                        })
                        .collect();
                    projections(
                        builder,
                        router_in,
                        &outputs,
                        hidden,
                        workers,
                        par_m1,
                        Pack::Into(packed),
                    )?;
                }
                // The experts' weights are chosen at run time: the call struct of each slot node
                // holds a weight pointer that the router node writes before they
                // run. The structs come first, so the router knows them.
                let gu_table = builder.keep_slice(
                    &m.gate_up
                        .iter()
                        .map(|w| w.as_ptr() as u64)
                        .collect::<Vec<_>>(),
                );
                let dn_table = builder
                    .keep_slice(&m.down.iter().map(|w| w.as_ptr() as u64).collect::<Vec<_>>());
                norm(
                    builder,
                    &self.norm_hidden,
                    x,
                    m.pre_ff_norm_2,
                    expert_in,
                    hidden,
                    s.eps,
                    team,
                )?;
                let gu_shape = GemmShape {
                    k: hidden,
                    n: 2 * ms.inter,
                };
                let dn_shape = GemmShape {
                    k: ms.inter,
                    n: hidden,
                };
                gu_shape.check().map_err(err)?;
                dn_shape.check().map_err(err)?;
                let compile = |shape: GemmShape| -> std::result::Result<_, ExecutionError> {
                    let plan = crate::jit::arch::gemm_plan(shape, 1);
                    crate::jit::gemm::get_or_compile(
                        shape,
                        plan.tile,
                        plan.opts,
                        true,
                        crate::jit::emit::AFormat::Bf16,
                    )
                    .map_err(err)?
                    .ok_or_else(|| ExecutionError("mounted expert kernel refused".into()))
                };
                let gu_kernel = compile(gu_shape)?;
                let dn_kernel = compile(dn_shape)?;
                let gu_quantized = crate::jit::int8::gemm(gu_shape.n, hidden);
                let dn_quantized = crate::jit::int8::gemm(dn_shape.n, ms.inter);
                if dn_quantized && crate::jit::int8::a_bytes(ms.top_k, ms.inter) > packed.len * 2 {
                    return Err(ExecutionError(
                        "mounted int8 activation buffer is too short for the experts".into(),
                    ));
                }
                let dn_row_bytes = if dn_quantized {
                    crate::jit::int8::a_row_bytes(ms.inter)
                } else {
                    2 * ms.inter
                };
                let dn_base: *const u8 = if dn_quantized {
                    packed.ptr.cast()
                } else {
                    act_rows.ptr.cast()
                };
                let rows_ptr: *const usize = &raw const builder.frame.rows;
                let mut gu_calls = Vec::with_capacity(ms.top_k);
                let mut dn_calls = Vec::with_capacity(ms.top_k);
                for k in 0..ms.top_k {
                    gu_calls.push(builder.keep(ex::abi::Bf16Call {
                        activations: expert_in.ptr.cast(),
                        weights: m.gate_up[0].as_ptr(),
                        output: unsafe { gate_up_rows.ptr.add(k * 2 * ms.inter) }.cast(),
                        rows: 0,
                        row_count: rows_ptr,
                    }));
                    dn_calls.push(builder.keep(ex::abi::Bf16Call {
                        activations: unsafe { dn_base.add(k * dn_row_bytes) },
                        weights: m.down[0].as_ptr(),
                        output: unsafe { down_rows.ptr.add(k * hidden) }.cast(),
                        rows: 0,
                        row_count: rows_ptr,
                    }));
                }
                let weights_field = std::mem::offset_of!(ex::abi::Bf16Call, weights);
                let mut route_call = ex::abi::RouteCall {
                    scores: scores.ptr,
                    per_expert_scale: m.per_expert_scale.as_ptr(),
                    weights_out: mixture,
                    gate_up_table: gu_table,
                    down_table: dn_table,
                    gate_up_slots: [std::ptr::null_mut(); ex::abi::MAX_TOP_K],
                    down_slots: [std::ptr::null_mut(); ex::abi::MAX_TOP_K],
                };
                for k in 0..ms.top_k {
                    route_call.gate_up_slots[k] =
                        unsafe { (gu_calls[k] as *mut u8).add(weights_field) }.cast();
                    route_call.down_slots[k] =
                        unsafe { (dn_calls[k] as *mut u8).add(weights_field) }.cast();
                }
                let route_call = builder.keep(route_call);
                builder.label("moe-route");
                builder.once(
                    Native::moe_route(ms.experts, ms.top_k)?,
                    binding(&[(route_call as u64, 0)], &[]),
                )?;
                let gu_in = if gu_quantized {
                    // The quantized expert input is read by the calls above: the buffer is the same
                    // for every slot.
                    quantized_input(
                        builder,
                        expert_in,
                        hidden,
                        Pack::Into(packed),
                        workers,
                        par_m1,
                    )?
                } else {
                    expert_in.ptr.cast()
                };
                for &call in &gu_calls {
                    unsafe { (*call).activations = gu_in };
                }
                let team_all = (workers, if par_m1 { workers } else { 1 });
                for &call in &gu_calls {
                    builder.label("moe-gate-up");
                    builder.grid(
                        Native::bf16_tile_selector(&gu_kernel)?,
                        binding(&[(call as u64, 0), (0, 1)], &[]),
                        TaskGrid {
                            rows_per_task: gu_kernel.tile().mr,
                            tasks_per_tile: gu_kernel.n_tile_count(),
                            last_block_only: false,
                        },
                        team_all.0,
                        team_all.1,
                    )?;
                }
                // The activation of every chosen expert: `gelu(gate) · up`, one task per slot.
                builder.label("moe-activation");
                builder.grid(
                    Native::elementary(act_moe)?,
                    binding(
                        &[
                            (gate_up_rows.ptr as u64, 2 * ms.inter * 2),
                            (
                                unsafe { gate_up_rows.ptr.add(ms.inter) } as u64,
                                2 * ms.inter * 2,
                            ),
                            (act_rows.ptr as u64, ms.inter * 2),
                        ],
                        &[],
                    ),
                    TaskGrid {
                        rows_per_task: builder.batch,
                        tasks_per_tile: ms.top_k,
                        last_block_only: false,
                    },
                    team_all.0,
                    team_all.1,
                )?;
                if dn_quantized {
                    let count = builder.keep(ms.top_k);
                    let call = builder.keep(ex::abi::PackCall {
                        source: act_rows.ptr,
                        dest: packed.ptr,
                        rows: count,
                    });
                    builder.label("moe-quantize");
                    builder.grid(
                        Native::int8_quantize(ms.inter)?,
                        binding(&[(call as u64, 0), (0, 1)], &[]),
                        TaskGrid {
                            rows_per_task: builder.batch,
                            tasks_per_tile: ms.top_k,
                            last_block_only: false,
                        },
                        team_all.0,
                        team_all.1,
                    )?;
                }
                for &call in &dn_calls {
                    builder.label("moe-down");
                    builder.grid(
                        Native::bf16_tile_selector(&dn_kernel)?,
                        binding(&[(call as u64, 0), (0, 1)], &[]),
                        TaskGrid {
                            rows_per_task: dn_kernel.tile().mr,
                            tasks_per_tile: dn_kernel.n_tile_count(),
                            last_block_only: false,
                        },
                        team_all.0,
                        team_all.1,
                    )?;
                }
                let combine = builder.keep(ex::abi::CombineCall {
                    rows: down_rows.ptr,
                    weights: mixture,
                    out: sum.ptr,
                });
                builder.label("moe-combine");
                builder.once(
                    Native::moe_combine(hidden, ms.top_k)?,
                    binding(&[(combine as u64, 0)], &[]),
                )?;
            }
            norm(
                builder,
                &self.norm_hidden,
                sum,
                m.post_ff_norm_2,
                xn,
                hidden,
                s.eps,
                team,
            )?;
            binary(builder, &self.residual, dense, xn, hidden, team)?;
            norm(
                builder,
                &self.norm_hidden,
                dense,
                g.post_ff_norm,
                xn,
                hidden,
                s.eps,
                team,
            )?;
        } else {
            norm(
                builder,
                &self.norm_hidden,
                out,
                g.post_ff_norm,
                xn,
                hidden,
                s.eps,
                team,
            )?;
        }
        binary(builder, &self.residual, x, xn, hidden, team)?;
        // The per-layer input branch: `x += norm(proj(gelu(gate(x)) · input_l))`, then the layer
        // scalar.
        if let Some(p) = &g.ple {
            let gelu_pl = gelu_pl
                .ok_or_else(|| ExecutionError("missing the gelu of the per-layer input".into()))?;
            projections(
                builder,
                x,
                &[Projection {
                    weights: p.gate,
                    output: gate_pl,
                    n: pl,
                }],
                hidden,
                workers,
                par_m1,
                Pack::Into(packed),
            )?;
            let input_l = unsafe { fold.ptr.add(layer * pl) } as u64;
            builder.label("layer-branch-gelu");
            builder.rows(
                Native::elementary(gelu_pl)?,
                binding(
                    &[
                        (gate_pl.ptr as u64, pl * 2),
                        (input_l, lpl * 2),
                        (gate_pl.ptr as u64, pl * 2),
                    ],
                    &[],
                ),
                team.0,
                team.1,
            )?;
            projections(
                builder,
                gate_pl,
                &[Projection {
                    weights: p.proj,
                    output: out,
                    n: hidden,
                }],
                pl,
                workers,
                par_m1,
                Pack::Into(packed),
            )?;
            norm(
                builder,
                &self.norm_hidden,
                out,
                p.norm,
                xn,
                hidden,
                s.eps,
                team,
            )?;
            binary(builder, &self.residual, x, xn, hidden, team)?;
        }
        builder.label("layer-scalar");
        builder.rows(
            Native::elementary(&gk.scale_hidden)?,
            binding(
                &[(x.ptr as u64, hidden * 2), (x.ptr as u64, hidden * 2)],
                &[g.layer_scalar],
            ),
            team.0,
            team.1,
        )?;
        builder.layer_end(layer);
        Ok(())
    }

    /// The Gemma 4 input: the embedding of the token times `√hidden`, and the per-layer inputs of
    /// every layer: `((norm(proj(x)) · hidden^-½) + lookup(token) · √pl) · 2^-½`, kept for the
    /// layers in `ple_fold`.
    ///
    /// `image`: the rows are the soft tokens of an image (the vision tower's output, copied by
    /// `lower_input` from its table): their embedding is not scaled, and their per-layer lookup
    /// is row 0 of the table (the padding token), as the reference gives the image tokens.
    pub(crate) fn lower_gemma_input(
        &mut self,
        builder: &mut Builder,
        weights: &crate::jit::GemmaModelWeights<'_>,
        workers: usize,
        image: bool,
    ) -> std::result::Result<(), ExecutionError> {
        let (Some(gs), Some(gk)) = (self.shape.gemma, self.gemma.as_ref()) else {
            return Err(ExecutionError(
                "a Gemma input without the Gemma shape".into(),
            ));
        };
        let s = self.shape;
        let (hidden, pl, lpl) = (s.hidden, gs.pl_dim, gs.n_layers * gs.pl_dim);
        let team = elem_team(workers);
        let par_m1 = crate::jit::options::decode().pool_m1;
        let b = &mut self.buffers;
        let packed = Buffer::of(&mut b.packed);
        let x = Buffer::of(&mut b.x);
        if !image {
            builder.label("embedding-scale");
            builder.rows(
                Native::elementary(&gk.scale_hidden)?,
                binding(
                    &[(x.ptr as u64, hidden * 2), (x.ptr as u64, hidden * 2)],
                    &[(hidden as f32).sqrt()],
                ),
                team.0,
                team.1,
            )?;
        }
        let (Some(ple), Some(pk)) = (weights.ple.as_ref(), gk.ple.as_ref()) else {
            return Ok(());
        };
        if ple.table.is_empty() || !ple.table.len().is_multiple_of(lpl) || ple.proj_norm.len() < pl
        {
            return Err(ExecutionError(
                "mounted per-layer embedding geometry mismatch".into(),
            ));
        }
        let fold = Buffer::of(&mut b.ple_fold);
        let tmp = Buffer::of(&mut b.ple_tmp);
        fold.check(builder.batch, lpl)?;
        tmp.check(builder.batch, lpl)?;
        // The lookup of the token in the per-layer table, times √pl_dim.
        let mut args = binding(
            &[
                (0, 4),
                (ple.table.as_ptr() as u64, 0),
                (tmp.ptr as u64, lpl * 2),
                (lpl as u64, 0),
            ],
            &[],
        );
        if image {
            // Every row reads token 0.
            let pad = builder.keep(0u64);
            args.arguments[0] = Argument {
                base: pad as u64,
                stride: 0,
            };
        } else {
            args.argument_bases[0] = (&raw const builder.frame.token_cursor).cast();
        }
        builder.label("per-layer-lookup");
        builder.rows(Native::embedding()?, args, 1, 1)?;
        builder.label("per-layer-lookup-scale");
        builder.rows(
            Native::elementary(&pk.scale_ple)?,
            binding(
                &[(tmp.ptr as u64, lpl * 2), (tmp.ptr as u64, lpl * 2)],
                &[(pl as f32).sqrt()],
            ),
            team.0,
            team.1,
        )?;
        // The projection of the embedding, scaled, normed per layer.
        projections(
            builder,
            x,
            &[Projection {
                weights: ple.model_proj,
                output: fold,
                n: lpl,
            }],
            hidden,
            workers,
            par_m1,
            Pack::Into(packed),
        )?;
        builder.label("per-layer-projection-scale");
        builder.rows(
            Native::elementary(&pk.scale_ple)?,
            binding(
                &[(fold.ptr as u64, lpl * 2), (fold.ptr as u64, lpl * 2)],
                &[(hidden as f32).powf(-0.5)],
            ),
            team.0,
            team.1,
        )?;
        builder.label("per-layer-projection-norm");
        builder.grid(
            Native::elementary(&pk.norm_pl)?,
            binding(
                &[
                    (fold.ptr as u64, pl * 2),
                    (ple.proj_norm.as_ptr() as u64, 0),
                    (fold.ptr as u64, pl * 2),
                ],
                &[s.eps],
            ),
            TaskGrid {
                rows_per_task: 1,
                tasks_per_tile: gs.n_layers,
                last_block_only: false,
            },
            team.0,
            team.1,
        )?;
        builder.label("per-layer-combine");
        builder.rows(
            Native::elementary(&pk.add_ple)?,
            binding(
                &[(fold.ptr as u64, lpl * 2), (tmp.ptr as u64, lpl * 2)],
                &[],
            ),
            team.0,
            team.1,
        )?;
        builder.label("per-layer-combine-scale");
        builder.rows(
            Native::elementary(&pk.scale_ple)?,
            binding(
                &[(fold.ptr as u64, lpl * 2), (fold.ptr as u64, lpl * 2)],
                &[0.5f32.sqrt()],
            ),
            team.0,
            team.1,
        )?;
        Ok(())
    }
}
