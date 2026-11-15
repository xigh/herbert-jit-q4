//! The kernels of the mixture-of-experts blocks of Gemma 4 (the 26B-A4B), of `qwen3_5_moe`
//! (Qwen3.6-35B-A3B) and of LiquidAI's LFM2 MoE, x86-64 (AVX2 + FMA): the router (one row, or a
//! block of rows grouped by expert), the weighted sum of the chosen experts (Gemma), and the end of
//! Qwen's sparse block with its residual (`qwen_out`). Same contracts as
//! `arch/aarch64/execution/moe.rs`. The route of a row follows its `RouteRule`.
use super::{ExecutionError, GeneratedKernel};
use crate::jit::execution::abi::Bf16Call;
pub use crate::jit::execution::abi::{
    CallFrame, CombineCall, CombineRowsCall, QwenOutCall, RouteCall, RouteRowsCall, RouteRowsSpec,
    RouteRule, MAX_TOP_K,
};
use iced_x86::code_asm::*;
use std::mem::offset_of;

fn err(e: impl std::fmt::Display) -> ExecutionError {
    ExecutionError(e.to_string())
}

/// `exp(x)` for `x <= 0` in `xmm0`, result in `xmm0`; constants in `xmm8..xmm15` (-87, log2 e,
/// c5..c1, 1); clobbers `xmm1..xmm4`, `eax`, `edx`.
pub(crate) fn emit_exp(a: &mut CodeAssembler) -> Result<(), iced_x86::IcedError> {
    a.vmaxss(xmm0, xmm0, xmm8)?;
    a.vmulss(xmm1, xmm0, xmm9)?;
    a.vroundss(xmm2, xmm2, xmm1, 9)?;
    a.vsubss(xmm3, xmm1, xmm2)?;
    a.vmovaps(xmm4, xmm10)?;
    for c in [xmm11, xmm12, xmm13, xmm14, xmm15] {
        a.vfmadd213ss(xmm4, xmm3, c)?;
    }
    a.vcvttss2si(eax, xmm2)?;
    a.shl(eax, 23)?;
    a.vmovd(edx, xmm4)?;
    a.add(eax, edx)?;
    a.vmovd(xmm0, eax)?;
    Ok(())
}

/// `xmm0` rounded to bf16 (nearest even), kept as f32; clobbers `eax`, `edx`.
pub(crate) fn emit_round_bf16(a: &mut CodeAssembler) -> Result<(), iced_x86::IcedError> {
    a.vmovd(eax, xmm0)?;
    a.mov(edx, eax)?;
    a.shr(edx, 16)?;
    a.and(edx, 1)?;
    a.add(eax, 0x7fff)?;
    a.add(eax, edx)?;
    a.and(eax, 0xffff_0000u32 as i32)?;
    a.vmovd(xmm0, eax)?;
    Ok(())
}

/// `v` rounded to bf16 (nearest even) in its eight f32 lanes, with `ymm12` = 1, `ymm13` = 0x7fff
/// and `ymm15` = 0xffff0000 in every lane; `t` is clobbered.
fn emit_round_bf16_ymm(
    a: &mut CodeAssembler,
    v: AsmRegisterYmm,
    t: AsmRegisterYmm,
) -> Result<(), iced_x86::IcedError> {
    a.vpsrld(t, v, 16u32)?;
    a.vpand(t, t, ymm12)?;
    a.vpaddd(v, v, t)?;
    a.vpaddd(v, v, ymm13)?;
    a.vpand(v, v, ymm15)?;
    Ok(())
}

/// The weights of [`RouteRule::Qwen`], once the selection has left the `top_k` best scores (best
/// first) at `val_off`, their experts at `idx_off` and `-inf` in their places among the f32 scores
/// at `rsp`: `e_k = exp(s_k − s_0)` (`s_0` the largest), `w_k = e_k / Σ_k e_k` (`norm`) or `e_k /
/// Σ_experts e` (the chosen ones count in the sum by their `e_k`, the `-inf` left in their places
/// by `exp(-87)`), each rounded to bf16; then the `top_k` pairs sorted by expert, increasing.
/// Exponential constants in `xmm8..xmm15`; clobbers `xmm0..xmm7`, `r8`, `r9`, `eax`, `edx`.
fn emit_qwen_weights(
    a: &mut CodeAssembler,
    experts: usize,
    top_k: usize,
    norm: bool,
    idx_off: usize,
    val_off: usize,
) -> Result<(), iced_x86::IcedError> {
    a.vmovss(xmm6, dword_ptr(rsp + val_off as i32))?;
    a.vxorps(xmm7, xmm7, xmm7)?;
    for k in 0..top_k {
        a.vmovss(xmm0, dword_ptr(rsp + (val_off + 4 * k) as i32))?;
        a.vsubss(xmm0, xmm0, xmm6)?;
        emit_exp(a)?;
        a.vmovss(dword_ptr(rsp + (val_off + 4 * k) as i32), xmm0)?;
        a.vaddss(xmm7, xmm7, xmm0)?;
    }
    if !norm {
        a.vxorps(xmm5, xmm5, xmm5)?;
        a.xor(r9d, r9d)?;
        let mut all = a.create_label();
        a.set_label(&mut all)?;
        a.vmovss(xmm0, dword_ptr(rsp + r9 * 4))?;
        a.vsubss(xmm0, xmm0, xmm6)?;
        emit_exp(a)?;
        a.vaddss(xmm5, xmm5, xmm0)?;
        a.inc(r9)?;
        a.cmp(r9, experts as i32)?;
        a.jb(all)?;
        a.vaddss(xmm7, xmm7, xmm5)?;
    }
    for k in 0..top_k {
        a.vmovss(xmm0, dword_ptr(rsp + (val_off + 4 * k) as i32))?;
        a.vdivss(xmm0, xmm0, xmm7)?;
        emit_round_bf16(a)?;
        a.vmovss(dword_ptr(rsp + (val_off + 4 * k) as i32), xmm0)?;
    }
    // Sorted by expert: a bubble network over the `top_k` pairs, unrolled.
    for i in 0..top_k.saturating_sub(1) {
        for j in 0..top_k - 1 - i {
            let mut keep = a.create_label();
            let (ij, ij1) = ((idx_off + 8 * j) as i32, (idx_off + 8 * (j + 1)) as i32);
            let (vj, vj1) = ((val_off + 4 * j) as i32, (val_off + 4 * (j + 1)) as i32);
            a.mov(r8, qword_ptr(rsp + ij))?;
            a.mov(r9, qword_ptr(rsp + ij1))?;
            a.cmp(r8, r9)?;
            a.jbe(keep)?;
            a.mov(qword_ptr(rsp + ij), r9)?;
            a.mov(qword_ptr(rsp + ij1), r8)?;
            a.mov(eax, dword_ptr(rsp + vj))?;
            a.mov(edx, dword_ptr(rsp + vj1))?;
            a.mov(dword_ptr(rsp + vj), edx)?;
            a.mov(dword_ptr(rsp + vj1), eax)?;
            a.set_label(&mut keep)?;
        }
    }
    Ok(())
}

/// [`RouteRule::Lfm`], before the selection: the `experts` f32 scores at `rsp` become `s =
/// bf16(σ(score))`, kept at `rsp + sig_off`, and in their places `s + bias[e]` (the f32 biases at
/// `rsi`), which the selection ranks. `σ(x)` is `1 / (1 + e)` for `x ≥ 0` and `e / (1 + e)` below,
/// `e = exp(−|x|)`. Exponential constants in `xmm8..xmm15`; clobbers `xmm0..xmm6`, `r9`, `eax`,
/// `edx`.
fn emit_lfm_scores(
    a: &mut CodeAssembler,
    experts: usize,
    sig_off: usize,
) -> Result<(), iced_x86::IcedError> {
    let mut each = a.create_label();
    a.xor(r9d, r9d)?;
    a.set_label(&mut each)?;
    let mut negative = a.create_label();
    let mut done = a.create_label();
    a.vmovss(xmm5, dword_ptr(rsp + r9 * 4))?;
    a.vmovd(eax, xmm5)?;
    a.or(eax, 0x8000_0000u32 as i32)?;
    a.vmovd(xmm0, eax)?;
    emit_exp(a)?;
    a.vaddss(xmm6, xmm0, xmm15)?;
    a.vmovd(eax, xmm5)?;
    a.test(eax, eax)?;
    a.js(negative)?;
    a.vdivss(xmm0, xmm15, xmm6)?;
    a.jmp(done)?;
    a.set_label(&mut negative)?;
    a.vdivss(xmm0, xmm0, xmm6)?;
    a.set_label(&mut done)?;
    emit_round_bf16(a)?;
    a.vmovss(dword_ptr(rsp + r9 * 4 + sig_off as i32), xmm0)?;
    a.vaddss(xmm0, xmm0, dword_ptr(rsi + r9 * 4))?;
    a.vmovss(dword_ptr(rsp + r9 * 4), xmm0)?;
    a.inc(r9)?;
    a.cmp(r9, experts as i32)?;
    a.jb(each)?;
    Ok(())
}

/// The weights of [`RouteRule::Lfm`], once the selection has left the `top_k` best choices (best
/// first) at `val_off` and their experts at `idx_off`: `w_k = s_k` (the rounded sigmoid at `rsp +
/// sig_off`), divided by `bf16(bf16(Σ_k s_k) + 1e-6)` when `norm` and rounded, times `scale` and
/// rounded unless it is one; then the `top_k` pairs sorted by expert, increasing. Clobbers `xmm0`,
/// `xmm6`, `xmm7`, `r8`, `r9`, `eax`, `edx`.
fn emit_lfm_weights(
    a: &mut CodeAssembler,
    top_k: usize,
    norm: bool,
    scale: f32,
    idx_off: usize,
    val_off: usize,
    sig_off: usize,
) -> Result<(), iced_x86::IcedError> {
    a.vxorps(xmm7, xmm7, xmm7)?;
    for k in 0..top_k {
        a.mov(r8, qword_ptr(rsp + (idx_off + 8 * k) as i32))?;
        a.vmovss(xmm0, dword_ptr(rsp + r8 * 4 + sig_off as i32))?;
        a.vmovss(dword_ptr(rsp + (val_off + 4 * k) as i32), xmm0)?;
        a.vaddss(xmm7, xmm7, xmm0)?;
    }
    if norm {
        a.vmovaps(xmm0, xmm7)?;
        emit_round_bf16(a)?;
        a.mov(eax, 1e-6f32.to_bits())?;
        a.vmovd(xmm6, eax)?;
        a.vaddss(xmm0, xmm0, xmm6)?;
        emit_round_bf16(a)?;
        a.vmovaps(xmm7, xmm0)?;
    }
    for k in 0..top_k {
        a.vmovss(xmm0, dword_ptr(rsp + (val_off + 4 * k) as i32))?;
        if norm {
            a.vdivss(xmm0, xmm0, xmm7)?;
            emit_round_bf16(a)?;
        }
        if scale != 1.0 {
            a.mov(eax, scale.to_bits())?;
            a.vmovd(xmm6, eax)?;
            a.vmulss(xmm0, xmm0, xmm6)?;
            emit_round_bf16(a)?;
        }
        a.vmovss(dword_ptr(rsp + (val_off + 4 * k) as i32), xmm0)?;
    }
    emit_sort_by_expert(a, top_k, idx_off, val_off)
}

/// The `top_k` (expert, weight) pairs at `idx_off` and `val_off` sorted by expert, increasing: a
/// bubble network, unrolled. Clobbers `r8`, `r9`, `eax`, `edx`.
fn emit_sort_by_expert(
    a: &mut CodeAssembler,
    top_k: usize,
    idx_off: usize,
    val_off: usize,
) -> Result<(), iced_x86::IcedError> {
    for i in 0..top_k.saturating_sub(1) {
        for j in 0..top_k - 1 - i {
            let mut keep = a.create_label();
            let (ij, ij1) = ((idx_off + 8 * j) as i32, (idx_off + 8 * (j + 1)) as i32);
            let (vj, vj1) = ((val_off + 4 * j) as i32, (val_off + 4 * (j + 1)) as i32);
            a.mov(r8, qword_ptr(rsp + ij))?;
            a.mov(r9, qword_ptr(rsp + ij1))?;
            a.cmp(r8, r9)?;
            a.jbe(keep)?;
            a.mov(qword_ptr(rsp + ij), r9)?;
            a.mov(qword_ptr(rsp + ij1), r8)?;
            a.mov(eax, dword_ptr(rsp + vj))?;
            a.mov(edx, dword_ptr(rsp + vj1))?;
            a.mov(dword_ptr(rsp + vj), edx)?;
            a.mov(dword_ptr(rsp + vj1), eax)?;
            a.set_label(&mut keep)?;
        }
    }
    Ok(())
}

/// Where [`RouteRule::Lfm`] keeps its sigmoid scores on the stack (after the selection's values),
/// and the frame it then needs; the other rules keep their frame.
fn lfm_frame(
    rule: RouteRule,
    experts: usize,
    val_off: usize,
    frame_bytes: usize,
) -> (usize, usize) {
    match rule {
        RouteRule::Lfm { .. } => {
            let sig_off = (val_off + 4 * MAX_TOP_K).next_multiple_of(32);
            (sig_off, (sig_off + experts * 4 + 8).next_multiple_of(16))
        }
        _ => (0, frame_bytes),
    }
}

/// Emit `(call: *const RouteCall)` for `experts` experts and `top_k` of them, Gemma's rule.
pub fn route(experts: usize, top_k: usize) -> Result<GeneratedKernel, ExecutionError> {
    route_with(experts, top_k, RouteRule::Gemma)
}

/// Emit `(call: *const RouteCall)` for `experts` experts and `top_k` of them, the weights by
/// `rule`.
pub fn route_with(
    experts: usize,
    top_k: usize,
    rule: RouteRule,
) -> Result<GeneratedKernel, ExecutionError> {
    if experts == 0
        || !experts.is_multiple_of(8)
        || top_k == 0
        || top_k > MAX_TOP_K
        || top_k > experts
    {
        return Err(ExecutionError(format!(
            "no router for {experts} experts, top {top_k}"
        )));
    }
    let scores_bytes = (experts * 4).next_multiple_of(32);
    let idx_off = scores_bytes;
    let val_off = idx_off + 8 * MAX_TOP_K;
    let frame_bytes = (val_off + 4 * MAX_TOP_K + 8).next_multiple_of(16);
    let (sig_off, frame_bytes) = lfm_frame(rule, experts, val_off, frame_bytes);
    let build = || -> Result<Vec<u8>, iced_x86::IcedError> {
        let mut a = CodeAssembler::new(64)?;
        a.set_prefer_vex(true);
        crate::jit::arch::x86::stack_reserve(&mut a, frame_bytes as i32)?;
        // The scores widened to f32 on the stack.
        a.mov(rsi, qword_ptr(rdi + offset_of!(RouteCall, scores) as i32))?;
        a.mov(rcx, (experts / 8) as u64)?;
        a.xor(r10d, r10d)?;
        let mut widen = a.create_label();
        a.set_label(&mut widen)?;
        a.vpmovzxwd(ymm0, xmmword_ptr(rsi))?;
        a.vpslld(ymm0, ymm0, 16u32)?;
        a.vmovups(ymmword_ptr(rsp + r10), ymm0)?;
        a.add(rsi, 16)?;
        a.add(r10, 32)?;
        a.dec(rcx)?;
        a.jne(widen)?;
        // The constants of the exponential.
        for (reg, value) in [
            (xmm8, -87.0f32),
            (xmm9, std::f32::consts::LOG2_E),
            (xmm10, 0.001_333_355_8),
            (xmm11, 0.009_618_129),
            (xmm12, 0.055_504_11),
            (xmm13, 0.240_226_5),
            (xmm14, std::f32::consts::LN_2),
            (xmm15, 1.0),
        ] {
            a.mov(eax, value.to_bits())?;
            a.vmovd(reg, eax)?;
        }
        if let RouteRule::Lfm { .. } = rule {
            a.mov(
                rsi,
                qword_ptr(rdi + offset_of!(RouteCall, per_expert_scale) as i32),
            )?;
            emit_lfm_scores(&mut a, experts, sig_off)?;
        }
        // Selection: top_k passes of "first maximum", the winner replaced by -inf.
        for k in 0..top_k {
            let mut scan = a.create_label();
            let mut keep = a.create_label();
            a.mov(eax, 0xff80_0000u32)?;
            a.vmovd(xmm0, eax)?;
            a.xor(r8d, r8d)?; // best index
            a.xor(r9d, r9d)?; // i
            a.set_label(&mut scan)?;
            a.vmovss(xmm1, dword_ptr(rsp + r9 * 4))?;
            a.vucomiss(xmm1, xmm0)?;
            a.jbe(keep)?; // not greater (or unordered)
            a.vmovaps(xmm0, xmm1)?;
            a.mov(r8, r9)?;
            a.set_label(&mut keep)?;
            a.inc(r9)?;
            a.cmp(r9, experts as i32)?;
            a.jb(scan)?;
            a.mov(qword_ptr(rsp + (idx_off + 8 * k) as i32), r8)?;
            a.vmovss(dword_ptr(rsp + (val_off + 4 * k) as i32), xmm0)?;
            a.mov(dword_ptr(rsp + r8 * 4), 0xff80_0000u32)?;
        }
        if rule != RouteRule::Gemma {
            match rule {
                RouteRule::Qwen { norm } => {
                    emit_qwen_weights(&mut a, experts, top_k, norm, idx_off, val_off)?
                }
                RouteRule::Lfm { norm, scale_bits } => emit_lfm_weights(
                    &mut a,
                    top_k,
                    norm,
                    f32::from_bits(scale_bits),
                    idx_off,
                    val_off,
                    sig_off,
                )?,
                RouteRule::Gemma => unreachable!(),
            }
            a.mov(
                rdx,
                qword_ptr(rdi + offset_of!(RouteCall, weights_out) as i32),
            )?;
            a.mov(
                r10,
                qword_ptr(rdi + offset_of!(RouteCall, gate_up_table) as i32),
            )?;
            a.mov(
                r11,
                qword_ptr(rdi + offset_of!(RouteCall, down_table) as i32),
            )?;
            for k in 0..top_k {
                a.mov(r8, qword_ptr(rsp + (idx_off + 8 * k) as i32))?;
                a.vmovss(xmm0, dword_ptr(rsp + (val_off + 4 * k) as i32))?;
                a.vmovss(dword_ptr(rdx + (4 * k) as i32), xmm0)?;
                a.mov(rax, qword_ptr(r10 + r8 * 8))?;
                a.mov(
                    rcx,
                    qword_ptr(rdi + (offset_of!(RouteCall, gate_up_slots) + 8 * k) as i32),
                )?;
                a.mov(qword_ptr(rcx), rax)?;
                a.mov(rax, qword_ptr(r11 + r8 * 8))?;
                a.mov(
                    rcx,
                    qword_ptr(rdi + (offset_of!(RouteCall, down_slots) + 8 * k) as i32),
                )?;
                a.mov(qword_ptr(rcx), rax)?;
            }
            a.add(rsp, frame_bytes as i32)?;
            a.vzeroupper()?;
            a.ret()?;
            return a.assemble(0);
        }
        // The softmax of the chosen scores (the first is the largest); the sum first.
        a.vmovss(xmm6, dword_ptr(rsp + val_off as i32))?;
        a.vxorps(xmm7, xmm7, xmm7)?;
        for k in 0..top_k {
            a.vmovss(xmm0, dword_ptr(rsp + (val_off + 4 * k) as i32))?;
            a.vsubss(xmm0, xmm0, xmm6)?;
            emit_exp(&mut a)?;
            a.vmovss(dword_ptr(rsp + (val_off + 4 * k) as i32), xmm0)?;
            a.vaddss(xmm7, xmm7, xmm0)?;
        }
        a.mov(
            rsi,
            qword_ptr(rdi + offset_of!(RouteCall, per_expert_scale) as i32),
        )?;
        a.mov(
            rdx,
            qword_ptr(rdi + offset_of!(RouteCall, weights_out) as i32),
        )?;
        a.mov(
            r10,
            qword_ptr(rdi + offset_of!(RouteCall, gate_up_table) as i32),
        )?;
        a.mov(
            r11,
            qword_ptr(rdi + offset_of!(RouteCall, down_table) as i32),
        )?;
        for k in 0..top_k {
            a.mov(r8, qword_ptr(rsp + (idx_off + 8 * k) as i32))?;
            a.vmovss(xmm0, dword_ptr(rsp + (val_off + 4 * k) as i32))?;
            a.vdivss(xmm0, xmm0, xmm7)?;
            a.vmulss(xmm0, xmm0, dword_ptr(rsi + r8 * 4))?;
            a.vmovss(dword_ptr(rdx + (4 * k) as i32), xmm0)?;
            a.mov(rax, qword_ptr(r10 + r8 * 8))?;
            a.mov(
                rcx,
                qword_ptr(rdi + (offset_of!(RouteCall, gate_up_slots) + 8 * k) as i32),
            )?;
            a.mov(qword_ptr(rcx), rax)?;
            a.mov(rax, qword_ptr(r11 + r8 * 8))?;
            a.mov(
                rcx,
                qword_ptr(rdi + (offset_of!(RouteCall, down_slots) + 8 * k) as i32),
            )?;
            a.mov(qword_ptr(rcx), rax)?;
        }
        a.add(rsp, frame_bytes as i32)?;
        a.vzeroupper()?;
        a.ret()?;
        a.assemble(0)
    };
    unsafe { GeneratedKernel::new(build().map_err(err)?) }
}

/// Emit `(call: *const CombineCall)`: `out[d] = Σ_{k<top_k} weights[k] · rows[k][d]` over `dim`
/// values (a multiple of 8), rounded to nearest even.
pub fn combine(dim: usize, top_k: usize) -> Result<GeneratedKernel, ExecutionError> {
    if dim == 0 || !dim.is_multiple_of(8) || top_k == 0 || top_k > MAX_TOP_K {
        return Err(ExecutionError(format!(
            "no combine of {top_k} rows of {dim}"
        )));
    }
    let build = || -> Result<Vec<u8>, iced_x86::IcedError> {
        let mut a = CodeAssembler::new(64)?;
        a.set_prefer_vex(true);
        a.mov(rsi, qword_ptr(rdi + offset_of!(CombineCall, rows) as i32))?;
        a.mov(
            rdx,
            qword_ptr(rdi + offset_of!(CombineCall, weights) as i32),
        )?;
        a.mov(r8, qword_ptr(rdi + offset_of!(CombineCall, out) as i32))?;
        // The constants of the rounding to bf16: 1 in ymm12, 0x7fff in ymm13.
        a.mov(eax, 1u32)?;
        a.vmovd(xmm12, eax)?;
        a.vpbroadcastd(ymm12, xmm12)?;
        a.mov(eax, 0x7fffu32)?;
        a.vmovd(xmm13, eax)?;
        a.vpbroadcastd(ymm13, xmm13)?;
        // The weights broadcast in ymm4..ymm11.
        let wregs = [ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11];
        for k in 0..top_k {
            a.vbroadcastss(wregs[k], dword_ptr(rdx + (4 * k) as i32))?;
        }
        a.xor(rcx, rcx)?; // byte offset within a row
        let mut top = a.create_label();
        a.set_label(&mut top)?;
        a.vxorps(ymm0, ymm0, ymm0)?;
        for k in 0..top_k {
            a.vpmovzxwd(ymm1, xmmword_ptr(rsi + rcx + (k * dim * 2) as i32))?;
            a.vpslld(ymm1, ymm1, 16u32)?;
            a.vfmadd231ps(ymm0, ymm1, wregs[k])?;
        }
        // Round to bf16 (nearest even) and store 8 values.
        a.vpsrld(ymm1, ymm0, 16u32)?;
        a.vpand(ymm1, ymm1, ymm12)?;
        a.vpaddd(ymm1, ymm1, ymm0)?;
        a.vpaddd(ymm1, ymm1, ymm13)?;
        a.vpsrld(ymm1, ymm1, 16u32)?;
        a.vextracti128(xmm2, ymm1, 1)?;
        a.vpackusdw(xmm1, xmm1, xmm2)?;
        a.vmovdqu(xmmword_ptr(r8 + rcx), xmm1)?;
        a.add(rcx, 16)?;
        a.cmp(rcx, (dim * 2) as i32)?;
        a.jb(top)?;
        a.vzeroupper()?;
        a.ret()?;
        a.assemble(0)
    };
    unsafe { GeneratedKernel::new(build().map_err(err)?) }
}

/// Emit `(call: *const RouteRowsCall)` (`abi::RouteRowsCall`): the router of the `frame.rows` rows
/// of a block, the selection, softmax and scale of each row as in [`route`], then the rows grouped
/// by expert. One task.
pub fn route_rows(spec: RouteRowsSpec) -> Result<GeneratedKernel, ExecutionError> {
    let RouteRowsSpec {
        experts,
        top_k,
        max_rows,
        a_row,
        gate_up_out_row,
        down_in_row,
        down_out_row,
        gate_up_tiles,
        down_tiles,
        rule,
    } = spec;
    let fits = |v: usize| i32::try_from(v).is_ok();
    if experts == 0
        || !experts.is_multiple_of(16)
        || top_k == 0
        || top_k > MAX_TOP_K
        || top_k > experts
        || max_rows == 0
        || ![
            a_row,
            gate_up_out_row,
            down_in_row,
            down_out_row,
            experts / 16 * max_rows * 32,
            gate_up_tiles.1,
            down_tiles.1,
        ]
        .into_iter()
        .all(fits)
        || gate_up_tiles.0 == 0
        || down_tiles.0 == 0
    {
        return Err(ExecutionError(format!("no router of a block for {spec:?}")));
    }
    let scores_bytes = (experts * 4).next_multiple_of(32);
    let idx_off = scores_bytes;
    let val_off = idx_off + 8 * MAX_TOP_K;
    let frame_bytes = (val_off + 4 * MAX_TOP_K + 8).next_multiple_of(16);
    let (sig_off, frame_bytes) = lfm_frame(rule, experts, val_off, frame_bytes);
    let field = |f: usize| f as i32;
    let build = || -> Result<Vec<u8>, iced_x86::IcedError> {
        let mut a = CodeAssembler::new(64)?;
        a.set_prefer_vex(true);
        for reg in [rbx, rbp, r12, r13, r14, r15] {
            a.push(reg)?;
        }
        crate::jit::arch::x86::stack_reserve(&mut a, frame_bytes as i32)?;
        a.mov(rbx, rdi)?;
        // r15 = the rows of the block.
        a.mov(
            rax,
            qword_ptr(rbx + field(offset_of!(RouteRowsCall, frame))),
        )?;
        a.mov(r15, qword_ptr(rax + field(offset_of!(CallFrame, rows))))?;
        // No expert has a row yet.
        a.mov(
            rax,
            qword_ptr(rbx + field(offset_of!(RouteRowsCall, row_counts))),
        )?;
        a.xor(ecx, ecx)?;
        let mut zero = a.create_label();
        a.set_label(&mut zero)?;
        a.mov(qword_ptr(rax + rcx * 8), 0)?;
        a.inc(rcx)?;
        a.cmp(rcx, experts as i32)?;
        a.jb(zero)?;
        // The constants of the exponential.
        for (reg, value) in [
            (xmm8, -87.0f32),
            (xmm9, std::f32::consts::LOG2_E),
            (xmm10, 0.001_333_355_8),
            (xmm11, 0.009_618_129),
            (xmm12, 0.055_504_11),
            (xmm13, 0.240_226_5),
            (xmm14, std::f32::consts::LN_2),
            (xmm15, 1.0),
        ] {
            a.mov(eax, value.to_bits())?;
            a.vmovd(reg, eax)?;
        }
        // ── Every row: its scores widened, the selection, the softmax, its experts counted. ──
        let mut rows_done = a.create_label();
        a.xor(r14d, r14d)?;
        a.test(r15, r15)?;
        a.je(rows_done)?;
        let mut row_top = a.create_label();
        a.set_label(&mut row_top)?;
        a.mov(
            rsi,
            qword_ptr(rbx + field(offset_of!(RouteRowsCall, scores))),
        )?;
        a.mov(rax, r14)?;
        a.shl(rax, 5)?;
        a.add(rsi, rax)?;
        for b in 0..experts / 16 {
            for half in 0..2 {
                a.vpmovzxwd(
                    ymm0,
                    xmmword_ptr(rsi + (b * max_rows * 32 + half * 16) as i32),
                )?;
                a.vpslld(ymm0, ymm0, 16u32)?;
                a.vmovups(ymmword_ptr(rsp + (b * 64 + half * 32) as i32), ymm0)?;
            }
        }
        if let RouteRule::Lfm { .. } = rule {
            a.mov(
                rsi,
                qword_ptr(rbx + field(offset_of!(RouteRowsCall, per_expert_scale))),
            )?;
            emit_lfm_scores(&mut a, experts, sig_off)?;
        }
        for k in 0..top_k {
            let mut scan = a.create_label();
            let mut keep = a.create_label();
            a.mov(eax, 0xff80_0000u32)?;
            a.vmovd(xmm0, eax)?;
            a.xor(r8d, r8d)?;
            a.xor(r9d, r9d)?;
            a.set_label(&mut scan)?;
            a.vmovss(xmm1, dword_ptr(rsp + r9 * 4))?;
            a.vucomiss(xmm1, xmm0)?;
            a.jbe(keep)?;
            a.vmovaps(xmm0, xmm1)?;
            a.mov(r8, r9)?;
            a.set_label(&mut keep)?;
            a.inc(r9)?;
            a.cmp(r9, experts as i32)?;
            a.jb(scan)?;
            a.mov(qword_ptr(rsp + (idx_off + 8 * k) as i32), r8)?;
            a.vmovss(dword_ptr(rsp + (val_off + 4 * k) as i32), xmm0)?;
            a.mov(dword_ptr(rsp + r8 * 4), 0xff80_0000u32)?;
        }
        match rule {
            RouteRule::Gemma => {
                a.vmovss(xmm6, dword_ptr(rsp + val_off as i32))?;
                a.vxorps(xmm7, xmm7, xmm7)?;
                for k in 0..top_k {
                    a.vmovss(xmm0, dword_ptr(rsp + (val_off + 4 * k) as i32))?;
                    a.vsubss(xmm0, xmm0, xmm6)?;
                    emit_exp(&mut a)?;
                    a.vmovss(dword_ptr(rsp + (val_off + 4 * k) as i32), xmm0)?;
                    a.vaddss(xmm7, xmm7, xmm0)?;
                }
            }
            RouteRule::Qwen { norm } => {
                emit_qwen_weights(&mut a, experts, top_k, norm, idx_off, val_off)?
            }
            RouteRule::Lfm { norm, scale_bits } => emit_lfm_weights(
                &mut a,
                top_k,
                norm,
                f32::from_bits(scale_bits),
                idx_off,
                val_off,
                sig_off,
            )?,
        }
        if rule == RouteRule::Gemma {
            a.mov(
                rsi,
                qword_ptr(rbx + field(offset_of!(RouteRowsCall, per_expert_scale))),
            )?;
        }
        a.imul_3(rax, r14, (4 * top_k) as i32)?;
        a.mov(
            rdx,
            qword_ptr(rbx + field(offset_of!(RouteRowsCall, weights_out))),
        )?;
        a.add(rdx, rax)?;
        a.mov(r10, qword_ptr(rbx + field(offset_of!(RouteRowsCall, ids))))?;
        a.add(r10, rax)?;
        a.mov(
            r11,
            qword_ptr(rbx + field(offset_of!(RouteRowsCall, row_counts))),
        )?;
        for k in 0..top_k {
            a.mov(r8, qword_ptr(rsp + (idx_off + 8 * k) as i32))?;
            a.vmovss(xmm0, dword_ptr(rsp + (val_off + 4 * k) as i32))?;
            if rule == RouteRule::Gemma {
                a.vdivss(xmm0, xmm0, xmm7)?;
                a.vmulss(xmm0, xmm0, dword_ptr(rsi + r8 * 4))?;
            }
            a.vmovss(dword_ptr(rdx + (4 * k) as i32), xmm0)?;
            a.mov(dword_ptr(r10 + (4 * k) as i32), r8d)?;
            a.inc(qword_ptr(r11 + r8 * 8))?;
        }
        a.inc(r14)?;
        a.cmp(r14, r15)?;
        a.jb(row_top)?;
        a.set_label(&mut rows_done)?;
        // ── Every expert: its first row, its calls moved there, the task counts of its two grids.
        // ──
        a.mov(
            rax,
            qword_ptr(rbx + field(offset_of!(RouteRowsCall, frame))),
        )?;
        a.mov(r11, qword_ptr(rax + field(offset_of!(CallFrame, counts))))?;
        a.xor(r12d, r12d)?; // e
        a.xor(r13d, r13d)?; // the first row of e
        let mut expert_top = a.create_label();
        a.set_label(&mut expert_top)?;
        a.mov(
            rax,
            qword_ptr(rbx + field(offset_of!(RouteRowsCall, row_counts))),
        )?;
        a.mov(rbp, qword_ptr(rax + r12 * 8))?;
        a.mov(
            rax,
            qword_ptr(rbx + field(offset_of!(RouteRowsCall, cursor))),
        )?;
        a.mov(qword_ptr(rax + r12 * 8), r13)?;
        // The gate/up call: activations at the first row of the gathered input, output at its first
        // row.
        a.mov(
            rax,
            qword_ptr(rbx + field(offset_of!(RouteRowsCall, gate_up_calls))),
        )?;
        a.mov(r8, qword_ptr(rax + r12 * 8))?;
        a.imul_3(r9, r13, a_row as i32)?;
        a.add(
            r9,
            qword_ptr(rbx + field(offset_of!(RouteRowsCall, gathered))),
        )?;
        a.mov(qword_ptr(r8 + field(offset_of!(Bf16Call, activations))), r9)?;
        a.imul_3(r9, r13, gate_up_out_row as i32)?;
        a.add(
            r9,
            qword_ptr(rbx + field(offset_of!(RouteRowsCall, gate_up_out))),
        )?;
        a.mov(qword_ptr(r8 + field(offset_of!(Bf16Call, output))), r9)?;
        // The down call, the same way.
        a.mov(
            rax,
            qword_ptr(rbx + field(offset_of!(RouteRowsCall, down_calls))),
        )?;
        a.mov(r8, qword_ptr(rax + r12 * 8))?;
        a.imul_3(r9, r13, down_in_row as i32)?;
        a.add(
            r9,
            qword_ptr(rbx + field(offset_of!(RouteRowsCall, down_in))),
        )?;
        a.mov(qword_ptr(r8 + field(offset_of!(Bf16Call, activations))), r9)?;
        a.imul_3(r9, r13, down_out_row as i32)?;
        a.add(
            r9,
            qword_ptr(rbx + field(offset_of!(RouteRowsCall, down_out))),
        )?;
        a.mov(qword_ptr(r8 + field(offset_of!(Bf16Call, output))), r9)?;
        // ⌈rows / mr⌉ · column tiles for each grid.
        for (tiles, grids) in [
            (gate_up_tiles, offset_of!(RouteRowsCall, gate_up_grids)),
            (down_tiles, offset_of!(RouteRowsCall, down_grids)),
        ] {
            a.lea(rax, qword_ptr(rbp + (tiles.0 as i32 - 1)))?;
            a.xor(edx, edx)?;
            a.mov(rcx, tiles.0 as u64)?;
            a.div(rcx)?;
            a.imul_3(rax, rax, tiles.1 as i32)?;
            a.mov(rcx, qword_ptr(rbx + field(grids)))?;
            a.mov(rcx, qword_ptr(rcx + r12 * 8))?;
            a.mov(qword_ptr(r11 + rcx * 8), rax)?;
        }
        a.add(r13, rbp)?;
        a.inc(r12)?;
        a.cmp(r12, experts as i32)?;
        a.jb(expert_top)?;
        // ── Every (row, k): its place among the rows of its expert, the row copied there. ──
        let mut copies_done = a.create_label();
        a.xor(r14d, r14d)?;
        a.test(r15, r15)?;
        a.je(copies_done)?;
        a.mov(
            rbp,
            qword_ptr(rbx + field(offset_of!(RouteRowsCall, cursor))),
        )?;
        let mut copy_top = a.create_label();
        a.set_label(&mut copy_top)?;
        a.imul_3(rax, r14, (4 * top_k) as i32)?;
        a.mov(r10, qword_ptr(rbx + field(offset_of!(RouteRowsCall, ids))))?;
        a.add(r10, rax)?;
        a.mov(
            r11,
            qword_ptr(rbx + field(offset_of!(RouteRowsCall, pos_out))),
        )?;
        a.add(r11, rax)?;
        a.imul_3(r13, r14, a_row as i32)?;
        a.add(
            r13,
            qword_ptr(rbx + field(offset_of!(RouteRowsCall, source))),
        )?;
        for k in 0..top_k {
            a.mov(eax, dword_ptr(r10 + (4 * k) as i32))?;
            a.mov(r8, qword_ptr(rbp + rax * 8))?;
            a.lea(r9, qword_ptr(r8 + 1))?;
            a.mov(qword_ptr(rbp + rax * 8), r9)?;
            a.mov(dword_ptr(r11 + (4 * k) as i32), r8d)?;
            a.imul_3(rdi, r8, a_row as i32)?;
            a.add(
                rdi,
                qword_ptr(rbx + field(offset_of!(RouteRowsCall, gathered))),
            )?;
            a.mov(rsi, r13)?;
            a.mov(rcx, a_row as u64)?;
            a.rep().movsb()?;
        }
        a.inc(r14)?;
        a.cmp(r14, r15)?;
        a.jb(copy_top)?;
        a.set_label(&mut copies_done)?;
        // The rows of the activation and of its quantization.
        a.imul_3(rax, r15, top_k as i32)?;
        a.mov(
            rcx,
            qword_ptr(rbx + field(offset_of!(RouteRowsCall, total))),
        )?;
        a.mov(qword_ptr(rcx), rax)?;
        a.add(rsp, frame_bytes as i32)?;
        for reg in [r15, r14, r13, r12, rbp, rbx] {
            a.pop(reg)?;
        }
        a.vzeroupper()?;
        a.ret()?;
        a.assemble(0)
    };
    unsafe { GeneratedKernel::new(build().map_err(err)?) }
}

/// Emit `(call: *const CombineRowsCall, row: usize)`:
/// `out[row][d] = Σ_{k<top_k} weights[row·top_k + k] · rows[pos[row·top_k + k]][d]` over `dim`
/// values (a multiple of 8), in the order and with the rounding of `combine`.
pub fn combine_rows(dim: usize, top_k: usize) -> Result<GeneratedKernel, ExecutionError> {
    if dim == 0
        || !dim.is_multiple_of(8)
        || top_k == 0
        || top_k > MAX_TOP_K
        || i32::try_from(dim * 2 * top_k).is_err()
    {
        return Err(ExecutionError(format!(
            "no combine of {top_k} rows of {dim} for a block"
        )));
    }
    let build = || -> Result<Vec<u8>, iced_x86::IcedError> {
        let mut a = CodeAssembler::new(64)?;
        a.set_prefer_vex(true);
        a.sub(rsp, (8 * MAX_TOP_K) as i32)?;
        a.imul_3(r9, rsi, (4 * top_k) as i32)?;
        a.mov(
            rdx,
            qword_ptr(rdi + offset_of!(CombineRowsCall, weights) as i32),
        )?;
        a.add(rdx, r9)?;
        a.mov(
            r10,
            qword_ptr(rdi + offset_of!(CombineRowsCall, pos) as i32),
        )?;
        a.add(r10, r9)?;
        a.imul_3(r8, rsi, (dim * 2) as i32)?;
        a.add(r8, qword_ptr(rdi + offset_of!(CombineRowsCall, out) as i32))?;
        a.mov(
            r11,
            qword_ptr(rdi + offset_of!(CombineRowsCall, rows) as i32),
        )?;
        // The address of each of the `top_k` rows, on the stack.
        for k in 0..top_k {
            a.mov(eax, dword_ptr(r10 + (4 * k) as i32))?;
            a.imul_3(rax, rax, (dim * 2) as i32)?;
            a.add(rax, r11)?;
            a.mov(qword_ptr(rsp + (8 * k) as i32), rax)?;
        }
        a.mov(eax, 1u32)?;
        a.vmovd(xmm12, eax)?;
        a.vpbroadcastd(ymm12, xmm12)?;
        a.mov(eax, 0x7fffu32)?;
        a.vmovd(xmm13, eax)?;
        a.vpbroadcastd(ymm13, xmm13)?;
        let wregs = [ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11];
        for k in 0..top_k {
            a.vbroadcastss(wregs[k], dword_ptr(rdx + (4 * k) as i32))?;
        }
        a.xor(rcx, rcx)?;
        let mut top = a.create_label();
        a.set_label(&mut top)?;
        a.vxorps(ymm0, ymm0, ymm0)?;
        for k in 0..top_k {
            a.mov(rax, qword_ptr(rsp + (8 * k) as i32))?;
            a.vpmovzxwd(ymm1, xmmword_ptr(rax + rcx))?;
            a.vpslld(ymm1, ymm1, 16u32)?;
            a.vfmadd231ps(ymm0, ymm1, wregs[k])?;
        }
        a.vpsrld(ymm1, ymm0, 16u32)?;
        a.vpand(ymm1, ymm1, ymm12)?;
        a.vpaddd(ymm1, ymm1, ymm0)?;
        a.vpaddd(ymm1, ymm1, ymm13)?;
        a.vpsrld(ymm1, ymm1, 16u32)?;
        a.vextracti128(xmm2, ymm1, 1)?;
        a.vpackusdw(xmm1, xmm1, xmm2)?;
        a.vmovdqu(xmmword_ptr(r8 + rcx), xmm1)?;
        a.add(rcx, 16)?;
        a.cmp(rcx, (dim * 2) as i32)?;
        a.jb(top)?;
        a.add(rsp, (8 * MAX_TOP_K) as i32)?;
        a.vzeroupper()?;
        a.ret()?;
        a.assemble(0)
    };
    unsafe { GeneratedKernel::new(build().map_err(err)?) }
}

/// Emit `(call: *const QwenOutCall, row: usize)` (`abi::QwenOutCall`): the end of a sparse block of
/// `qwen3_5_moe` and the residual for one row, `dim` values (a multiple of 8), `top_k` experts. The
/// logit of the shared expert's gate is a dot product in f32 (eight lanes, then the halves),
/// rounded to bf16; its sigmoid `1 / (1 + e^-l)` (or `e^l / (1 + e^l)` below zero) rounded to bf16;
/// then per eight values every product and every sum rounded to bf16 in the order of the module
/// (`index_add_` into a bf16 zero vector, slot by slot).
pub fn qwen_out(dim: usize, top_k: usize) -> Result<GeneratedKernel, ExecutionError> {
    if dim == 0
        || !dim.is_multiple_of(8)
        || top_k == 0
        || top_k > MAX_TOP_K
        || i32::try_from(dim * 2 * top_k).is_err()
    {
        return Err(ExecutionError(format!(
            "no Qwen expert output of {top_k} rows of {dim}"
        )));
    }
    let build = || -> Result<Vec<u8>, iced_x86::IcedError> {
        let mut a = CodeAssembler::new(64)?;
        a.set_prefer_vex(true);
        a.sub(rsp, (8 * MAX_TOP_K) as i32)?;
        a.imul_3(r9, rsi, (4 * top_k) as i32)?;
        a.mov(r10, qword_ptr(rdi + offset_of!(QwenOutCall, pos) as i32))?;
        a.add(r10, r9)?;
        a.imul_3(r8, rsi, (dim * 2) as i32)?;
        a.mov(r11, qword_ptr(rdi + offset_of!(QwenOutCall, rows) as i32))?;
        // The address of each of the `top_k` expert rows, on the stack.
        for k in 0..top_k {
            a.mov(eax, dword_ptr(r10 + (4 * k) as i32))?;
            a.imul_3(rax, rax, (dim * 2) as i32)?;
            a.add(rax, r11)?;
            a.mov(qword_ptr(rsp + (8 * k) as i32), rax)?;
        }
        // The logit `shared_gate · xn[row]`.
        a.mov(r9, qword_ptr(rdi + offset_of!(QwenOutCall, xn) as i32))?;
        a.add(r9, r8)?;
        a.mov(r10, qword_ptr(rdi + offset_of!(QwenOutCall, gate) as i32))?;
        a.vxorps(ymm0, ymm0, ymm0)?;
        a.xor(ecx, ecx)?;
        let mut dot = a.create_label();
        a.set_label(&mut dot)?;
        a.vpmovzxwd(ymm1, xmmword_ptr(r9 + rcx))?;
        a.vpslld(ymm1, ymm1, 16u32)?;
        a.vpmovzxwd(ymm2, xmmword_ptr(r10 + rcx))?;
        a.vpslld(ymm2, ymm2, 16u32)?;
        a.vfmadd231ps(ymm0, ymm1, ymm2)?;
        a.add(rcx, 16)?;
        a.cmp(rcx, (dim * 2) as i32)?;
        a.jb(dot)?;
        a.vextractf128(xmm1, ymm0, 1)?;
        a.vaddps(xmm0, xmm0, xmm1)?;
        a.vmovhlps(xmm1, xmm0, xmm0)?;
        a.vaddps(xmm0, xmm0, xmm1)?;
        a.vmovshdup(xmm1, xmm0)?;
        a.vaddss(xmm0, xmm0, xmm1)?;
        emit_round_bf16(&mut a)?;
        // Its sigmoid, from `e = exp(-|l|)`.
        a.vmovaps(xmm5, xmm0)?;
        for (reg, value) in [
            (xmm8, -87.0f32),
            (xmm9, std::f32::consts::LOG2_E),
            (xmm10, 0.001_333_355_8),
            (xmm11, 0.009_618_129),
            (xmm12, 0.055_504_11),
            (xmm13, 0.240_226_5),
            (xmm14, std::f32::consts::LN_2),
            (xmm15, 1.0),
        ] {
            a.mov(eax, value.to_bits())?;
            a.vmovd(reg, eax)?;
        }
        a.vxorps(xmm1, xmm1, xmm1)?;
        a.vsubss(xmm1, xmm1, xmm0)?;
        a.vminss(xmm0, xmm0, xmm1)?;
        emit_exp(&mut a)?;
        a.vaddss(xmm6, xmm0, xmm15)?;
        a.vxorps(xmm1, xmm1, xmm1)?;
        let mut negative = a.create_label();
        a.vucomiss(xmm5, xmm1)?;
        a.jb(negative)?;
        a.vmovaps(xmm0, xmm15)?;
        a.set_label(&mut negative)?;
        a.vdivss(xmm0, xmm0, xmm6)?;
        emit_round_bf16(&mut a)?;
        a.vbroadcastss(ymm14, xmm0)?;
        // The rounding constants, the weights broadcast in ymm4..ymm11.
        for (reg, low, value) in [
            (ymm12, xmm12, 1u32),
            (ymm13, xmm13, 0x7fff),
            (ymm15, xmm15, 0xffff_0000),
        ] {
            a.mov(eax, value)?;
            a.vmovd(low, eax)?;
            a.vpbroadcastd(reg, low)?;
        }
        // The weights of the row (`edx` was the exponential's scratch).
        a.imul_3(r9, rsi, (4 * top_k) as i32)?;
        a.mov(
            rdx,
            qword_ptr(rdi + offset_of!(QwenOutCall, weights) as i32),
        )?;
        a.add(rdx, r9)?;
        let wregs = [ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11];
        for k in 0..top_k {
            a.vbroadcastss(wregs[k], dword_ptr(rdx + (4 * k) as i32))?;
        }
        a.mov(r9, qword_ptr(rdi + offset_of!(QwenOutCall, shared) as i32))?;
        a.add(r9, r8)?;
        a.mov(r10, qword_ptr(rdi + offset_of!(QwenOutCall, x) as i32))?;
        a.add(r10, r8)?;
        a.xor(ecx, ecx)?;
        let mut top = a.create_label();
        a.set_label(&mut top)?;
        for k in 0..top_k {
            a.mov(rax, qword_ptr(rsp + (8 * k) as i32))?;
            a.vpmovzxwd(ymm1, xmmword_ptr(rax + rcx))?;
            a.vpslld(ymm1, ymm1, 16u32)?;
            a.vmulps(ymm1, ymm1, wregs[k])?;
            emit_round_bf16_ymm(&mut a, ymm1, ymm2)?;
            if k == 0 {
                a.vmovaps(ymm0, ymm1)?;
            } else {
                a.vaddps(ymm0, ymm0, ymm1)?;
                emit_round_bf16_ymm(&mut a, ymm0, ymm2)?;
            }
        }
        // `+ bf16(g · shared)`, rounded; then the residual.
        a.vpmovzxwd(ymm1, xmmword_ptr(r9 + rcx))?;
        a.vpslld(ymm1, ymm1, 16u32)?;
        a.vmulps(ymm1, ymm1, ymm14)?;
        emit_round_bf16_ymm(&mut a, ymm1, ymm2)?;
        a.vaddps(ymm0, ymm0, ymm1)?;
        emit_round_bf16_ymm(&mut a, ymm0, ymm2)?;
        a.vpmovzxwd(ymm1, xmmword_ptr(r10 + rcx))?;
        a.vpslld(ymm1, ymm1, 16u32)?;
        a.vaddps(ymm0, ymm0, ymm1)?;
        a.vpsrld(ymm1, ymm0, 16u32)?;
        a.vpand(ymm1, ymm1, ymm12)?;
        a.vpaddd(ymm1, ymm1, ymm0)?;
        a.vpaddd(ymm1, ymm1, ymm13)?;
        a.vpsrld(ymm1, ymm1, 16u32)?;
        a.vextracti128(xmm2, ymm1, 1)?;
        a.vpackusdw(xmm1, xmm1, xmm2)?;
        a.vmovdqu(xmmword_ptr(r10 + rcx), xmm1)?;
        a.add(rcx, 16)?;
        a.cmp(rcx, (dim * 2) as i32)?;
        a.jb(top)?;
        a.add(rsp, (8 * MAX_TOP_K) as i32)?;
        a.vzeroupper()?;
        a.ret()?;
        a.assemble(0)
    };
    unsafe { GeneratedKernel::new(build().map_err(err)?) }
}
