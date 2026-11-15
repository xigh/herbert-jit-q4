//! The x86-64 target, AVX-512 and AVX2 arms: the GEMM micro-kernel, the
//! elementwise, SDPA and GatedDeltaNet emitters, their register tables, and
//! the generated nodes of the execution plan.

pub mod emit;
pub mod emit_conv;
pub mod emit_elem;
pub mod emit_elem_avx2;
pub mod emit_gated_norm;
pub mod emit_gemm_avx2;
pub mod emit_local_attn;
pub mod emit_recurrence;
pub mod emit_sdpa;
pub mod emit_sdpa_avx2;
pub mod execution;
pub mod gemm_int8;
pub mod gemm_int8_sse42;
#[cfg(test)]
mod int8_tests;
#[cfg(test)]
mod moe_tests;
pub mod regs;
#[cfg(test)]
mod sdpa_avx2_tests;
#[cfg(test)]
mod sdpa_bench;
#[cfg(test)]
mod sdpa_tests;

use crate::core::error::Result;
use crate::jit::arch::{Assembled, KernelSet, Target};
use crate::jit::emit::MicroKernelSpec;
use crate::jit::emit_conv::ConvSpec;
use crate::jit::emit_elem::ElemSpec;
use crate::jit::emit_gated_norm::GatedNormSpec;
use crate::jit::emit_recurrence::RecurrenceSpec;
use crate::jit::emit_sdpa::{Pass, SdpaKernelSpec, SdpaPrefillSpec};
use crate::jit::execution::abi::{AttentionSpec, KvSpec, Shape, TaskGrid};
use crate::jit::execution::{ExecutionError, ExecutionPlan, GeneratedKernel};
use crate::jit::ir::{Decline, TileShape, ROWS_INTERLEAVED};
use execution::{attention, audit, bf16, delta, entry, frame, kernels, kv, transforms};

/// `sub rsp, bytes` a page at a time, each page touched on the way down: Windows commits the stack
/// through a single guard page, which a frame of more than 4 KiB written from its bottom would
/// skip.
pub(crate) fn stack_reserve(
    a: &mut iced_x86::code_asm::CodeAssembler,
    bytes: i32,
) -> std::result::Result<(), iced_x86::IcedError> {
    use iced_x86::code_asm::{dword_ptr, rsp};
    let mut left = bytes;
    while left > 4096 {
        a.sub(rsp, 4096)?;
        a.or(dword_ptr(rsp), 0)?;
        left -= 4096;
    }
    a.sub(rsp, left)?;
    Ok(())
}

/// Vector registers consumed by a `mr × nr` tile of the AVX-512 kernel:
/// `mr · nb` accumulators, `nb` registers of `B`, 1 of `A` broadcast.
///
/// The allocation is computed from the tile, never a hard-coded register list.
pub const fn zmm_needed(mr: usize, nb: usize) -> usize {
    mr * nb + nb + 1
}

/// Does the tile fit in the 32 zmm, with `nr` a multiple of 16?
pub fn tile_fits(tile: TileShape) -> std::result::Result<(), Decline> {
    if tile.mr == 0 || tile.nr == 0 || !tile.nr.is_multiple_of(ROWS_INTERLEAVED) {
        return Err(Decline::TileTooWide);
    }
    if zmm_needed(tile.mr, tile.n_blocks()) > regs::N_REGS {
        return Err(Decline::TileTooWide);
    }
    Ok(())
}

/// The largest legal width ≤ `max_cols` for a height `mr`, in
/// multiples of 16. `None` if even 16 columns don't fit.
pub fn widest_tile(mr: usize, max_cols: usize) -> Option<usize> {
    let mut nb = max_cols / ROWS_INTERLEAVED;
    while nb > 0 {
        if zmm_needed(mr, nb) <= regs::N_REGS {
            return Some(nb * ROWS_INTERLEAVED);
        }
        nb -= 1;
    }
    None
}

/// Vector registers of the recurrence kernel outside the accumulators:
/// `beta`, `decay`, `scale`, `kval`, `qval`, `rv`, `t`.
const RECURRENCE_FIXED_REGS: usize = 7;

/// The x86-64 target.
pub(crate) struct X86;

impl Target for X86 {
    const R16_ROWS: usize = 16;
    fn gemm_tile(shape: crate::jit::ir::GemmShape) -> crate::jit::ir::TileShape {
        crate::jit::x86_production_tile(shape)
    }
    /// `--int8` / `--q4`: the shapes that take them run `gemm_int8.rs` (AVX-512 VNNI, AVX-VNNI, or
    /// AVX2 without VNNI).
    fn gemm_plan(shape: crate::jit::ir::GemmShape, rows: usize) -> crate::jit::emit::GemmPlan {
        let mut neon = crate::jit::knobs::knobs().neon;
        if crate::jit::int8::gemm(shape.n, shape.k) {
            neon.w_q4 = crate::jit::int8::q4_for(shape.n);
            let max_rows = if neon.w_sse42 {
                gemm_int8_sse42::MAX_ROWS
            } else {
                gemm_int8::max_rows()
            };
            let tile = TileShape {
                mr: if rows == 1 { 1 } else { max_rows },
                nr: 16,
            };
            return crate::jit::emit::GemmPlan {
                tile,
                opts: crate::jit::emit::KernelOptions {
                    neon,
                    ..Default::default()
                },
            };
        }
        Self::gemm_plan_bf16(shape, rows)
    }
    fn gemm_plan_bf16(
        shape: crate::jit::ir::GemmShape,
        _rows: usize,
    ) -> crate::jit::emit::GemmPlan {
        let mut neon = crate::jit::knobs::knobs().neon;
        neon.w_int8 = false;
        neon.w_q4 = false;
        crate::jit::emit::GemmPlan {
            tile: Self::gemm_tile(shape),
            opts: crate::jit::emit::KernelOptions {
                neon,
                ..Default::default()
            },
        }
    }
    fn tile_fits(tile: TileShape) -> std::result::Result<(), Decline> {
        tile_fits(tile)
    }
    fn widest_tile(mr: usize, max_cols: usize) -> Option<usize> {
        widest_tile(mr, max_cols)
    }
    fn check_gemm(spec: &MicroKernelSpec) -> std::result::Result<(), Decline> {
        if spec.k_dynamic {
            // `K` is an immediate of every x86 kernel.
            return Err(Decline::OddK);
        }
        emit::check_avx512(spec)
    }
    /// `7 + 2·(hvd/16) ≤ 32`: every accumulator in a zmm.
    fn check_recurrence(spec: &RecurrenceSpec) -> std::result::Result<(), Decline> {
        if spec.store_once
            || spec.pairs
            || spec.q_hoist
            || spec.pass1_lanes != 32
            || spec.width != 0
        {
            // AArch64 forms: the x86 recurrence has the one form.
            return Err(Decline::OutOfRegisters);
        }
        if RECURRENCE_FIXED_REGS + 2 * spec.blocks16() > regs::N_REGS {
            return Err(Decline::OutOfRegisters);
        }
        Ok(())
    }
    fn emit_gemm(spec: &MicroKernelSpec) -> Result<Vec<u8>> {
        emit::emit(spec)
    }
    /// The arm the process selected: zmm with AVX-512, ymm otherwise.
    fn emit_elem(spec: &ElemSpec) -> Result<Vec<u8>> {
        if let crate::jit::emit_elem::ElemOp::LocalAttention { .. } = spec.op {
            return emit_local_attn::emit(spec);
        }
        match crate::jit::elem::select_isa() {
            Some(crate::jit::elem::ElemIsa::Zmm) => emit_elem::emit_elem(spec),
            _ => emit_elem_avx2::emit_elem_avx2(spec),
        }
    }
    fn emit_sdpa(spec: &SdpaKernelSpec, pass: Option<Pass>) -> Result<Vec<u8>> {
        emit_sdpa::emit_sdpa(spec, pass)
    }
    /// The block kernel of the arm the process selected, as `emit_sdpa`'s
    /// driver does for the decode kernel.
    fn emit_sdpa_prefill(spec: &SdpaPrefillSpec) -> Result<Vec<u8>> {
        if !crate::jit::gemm::is_supported() {
            return emit_sdpa_avx2::emit_sdpa_prefill_avx2(spec);
        }
        emit_sdpa::emit_sdpa_prefill(spec)
    }
    fn emit_conv(spec: &ConvSpec) -> Result<Vec<u8>> {
        emit_conv::emit_conv(spec)
    }
    fn emit_recurrence(spec: &RecurrenceSpec) -> Result<Vec<u8>> {
        emit_recurrence::emit_recurrence(spec)
    }
    fn emit_gated_norm(spec: &GatedNormSpec) -> Result<Vec<u8>> {
        emit_gated_norm::emit_gated_norm(spec)
    }
    fn audit_kernel(bytes: &[u8]) -> std::result::Result<(), ExecutionError> {
        audit::kernel(bytes)
    }
    fn ret_kernel() -> std::result::Result<Vec<u8>, ExecutionError> {
        audit::ret()
    }
    fn assemble(
        plan: &ExecutionPlan,
        hybrid_spin: Option<u32>,
        trace_sync: bool,
        trace_pmu: bool,
        shared_barrier: bool,
        fuse_serial: bool,
    ) -> std::result::Result<Assembled, ExecutionError> {
        entry::assemble(
            plan,
            hybrid_spin,
            trace_sync,
            trace_pmu,
            shared_barrier,
            fuse_serial,
        )
    }
    fn audit_calls(
        code: &[u8],
        runtime_calls: &[(usize, usize)],
    ) -> std::result::Result<(), ExecutionError> {
        entry::audit_calls(code, runtime_calls)
    }
}

type Kernel = std::result::Result<GeneratedKernel, ExecutionError>;

impl KernelSet for X86 {
    fn begin_block(max_batch: usize, grids: &[TaskGrid]) -> Kernel {
        frame::begin_block(max_batch, grids)
    }
    fn end_block() -> Kernel {
        frame::end_block()
    }
    fn embedding() -> Kernel {
        kernels::embedding()
    }
    fn argmax_slices() -> Kernel {
        Err(ExecutionError(
            "the sliced argmax is an AArch64 and RISC-V node".into(),
        ))
    }
    fn argmax_merge() -> Kernel {
        Err(ExecutionError(
            "the sliced argmax is an AArch64 and RISC-V node".into(),
        ))
    }
    fn argmax_bf16() -> Kernel {
        kernels::argmax_bf16()
    }
    fn elementary(kernel: &crate::jit::elem::CompiledElem) -> Kernel {
        transforms::elementary(kernel)
    }
    fn offset(factor: usize, subtract: usize) -> Kernel {
        transforms::offset(factor, subtract)
    }
    fn norm_rope(
        norm: Option<&crate::jit::elem::CompiledElem>,
        rope: &crate::jit::elem::CompiledElem,
        heads: usize,
        stride: usize,
    ) -> Kernel {
        transforms::norm_rope(norm, rope, heads, stride)
    }
    fn bf16_multi_tile_selector(_parts: &[&crate::jit::compile::CompiledGemm]) -> Kernel {
        Err(ExecutionError(
            "fused projection nodes are AArch64 nodes".into(),
        ))
    }
    fn bf16_tile_selector(gemm: &crate::jit::compile::CompiledGemm) -> Kernel {
        bf16::tile_selector(gemm)
    }
    fn int8_quantize(k: usize) -> Kernel {
        execution::int8::quantize(k)
    }
    fn moe_route(experts: usize, top_k: usize) -> Kernel {
        execution::moe::route(experts, top_k)
    }
    fn moe_combine(dim: usize, top_k: usize) -> Kernel {
        execution::moe::combine(dim, top_k)
    }
    fn moe_route_rows(spec: crate::jit::execution::abi::RouteRowsSpec) -> Kernel {
        execution::moe::route_rows(spec)
    }
    fn moe_combine_rows(dim: usize, top_k: usize) -> Kernel {
        execution::moe::combine_rows(dim, top_k)
    }
    fn moe_route_rule(
        experts: usize,
        top_k: usize,
        rule: crate::jit::execution::abi::RouteRule,
    ) -> Kernel {
        execution::moe::route_with(experts, top_k, rule)
    }
    fn moe_qwen_out(dim: usize, top_k: usize) -> Kernel {
        execution::moe::qwen_out(dim, top_k)
    }
    fn attention_tile_selector(
        sdpa: &crate::jit::sdpa::CompiledSdpa,
        prefill: Option<&crate::jit::sdpa::CompiledSdpaPrefill>,
        spec: AttentionSpec,
    ) -> Kernel {
        attention::tile_selector(sdpa, prefill, spec)
    }
    fn kv_append(spec: KvSpec) -> Kernel {
        kv::append(spec)
    }
    fn delta_convolution(
        shape: Shape,
        kernel: Option<&crate::jit::conv::CompiledConv>,
        fma: bool,
    ) -> Kernel {
        delta::convolution(shape, kernel, fma)
    }
    fn delta_recurrence(
        shape: Shape,
        kernel: Option<&crate::jit::recurrence::CompiledRecurrence>,
    ) -> Kernel {
        delta::recurrence(shape, kernel)
    }
    fn delta_normed_gate(
        dim: usize,
        kernel: Option<&crate::jit::gated_norm::CompiledGatedNorm>,
        fma: bool,
    ) -> Kernel {
        delta::normed_gate(dim, kernel, fma)
    }
    fn delta_interleave(nv: usize) -> Kernel {
        delta::interleave(nv)
    }
    fn delta_l2_gates(shape: Shape, fma: bool) -> Kernel {
        delta::l2_gates(shape, fma)
    }
    fn mamba_convolution(shape: Shape, fma: bool) -> Kernel {
        delta::convolution_mamba(shape, fma)
    }
    fn mamba_gates(nv: usize, fma: bool) -> Kernel {
        delta::gates_of(nv, fma, true)
    }
    fn mamba_recurrence(
        shape: Shape,
        kernel: Option<&crate::jit::recurrence::CompiledRecurrence>,
    ) -> Kernel {
        delta::recurrence_mamba(shape, kernel)
    }
    fn mamba_normed_gate(dim: usize, fma: bool) -> Kernel {
        delta::normed_gate_mamba(dim, fma)
    }
    fn short_conv(width: usize, taps: usize) -> Kernel {
        execution::short_conv::short_conv(width, taps)
    }
    fn head_gate(heads: usize, head_dim: usize) -> Kernel {
        execution::short_conv::head_gate(heads, head_dim)
    }
}
