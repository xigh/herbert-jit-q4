//! LiquidAI's gated short convolution (`lfm2`, `lfm2_moe`), x86-64 (AVX2 + FMA, `ymm`): one task
//! per block of sixteen channels, every row of the block in order, as `Lfm2ShortConv` computes it
//! in bf16. Same contract and order of operations as `arch/aarch64/execution/short_conv.rs`.
use super::{ExecutionError, GeneratedKernel};
pub(crate) use crate::jit::execution::abi::ShortConvCall;
use iced_x86::code_asm::*;
use std::mem::offset_of;

fn err(e: impl std::fmt::Display) -> ExecutionError {
    ExecutionError(e.to_string())
}

/// `v` rounded to bf16 (nearest even) in its eight f32 lanes, kept as f32: `ymm12` = 1, `ymm13` =
/// 0x7fff, `ymm15` = 0xffff0000 in every lane; `ymm14` is clobbered.
fn round(a: &mut CodeAssembler, v: AsmRegisterYmm) -> Result<(), iced_x86::IcedError> {
    a.vpsrld(ymm14, v, 16u32)?;
    a.vpand(ymm14, ymm14, ymm12)?;
    a.vpaddd(v, v, ymm14)?;
    a.vpaddd(v, v, ymm13)?;
    a.vpand(v, v, ymm15)
}

/// `(call: *const ShortConvCall, channel block)`: sixteen channels of `width`, the `rows` rows of
/// the block in order. For each row and channel: `B`, `C`, `x` rounded to bf16 (the module's bf16
/// `in_proj`), `bx = bf16(B · x)`; the channel's state (its last `taps` values of `bx`, oldest
/// first) shifts by one and takes `bx`; `conv = bf16(w_0·s_0 + w_1·s_1 + …)`, the product of the
/// oldest tap then one fused multiply-add per tap, oldest first; `y = bf16(C · conv)` stored as
/// bf16. Two halves of eight channels, each in `ymm` lanes.
pub(crate) fn short_conv(width: usize, taps: usize) -> Result<GeneratedKernel, ExecutionError> {
    if width == 0
        || !width.is_multiple_of(16)
        || width * 4 > i32::MAX as usize / 2
        || !(2..=8).contains(&taps)
    {
        return Err(ExecutionError(format!(
            "no short convolution of {width} channels and {taps} taps"
        )));
    }
    let regs = [ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7];
    let field = |f: usize| f as i32;
    let build = || -> Result<Vec<u8>, iced_x86::IcedError> {
        let mut a = CodeAssembler::new(64)?;
        a.set_prefer_vex(true);
        let mut exit = a.create_label();
        let mut row = a.create_label();
        // rax = rows, r8 = B, r9 = C, r10 = x, r11 = output, rdx = state, rcx = weights of the
        // block.
        a.mov(rax, qword_ptr(rdi + field(offset_of!(ShortConvCall, rows))))?;
        a.mov(rax, qword_ptr(rax))?;
        a.test(rax, rax)?;
        a.je(exit)?;
        a.imul_3(r8, rsi, 64)?;
        a.mov(r9, r8)?;
        a.mov(r10, r8)?;
        a.add(r8, qword_ptr(rdi + field(offset_of!(ShortConvCall, b))))?;
        a.add(r9, qword_ptr(rdi + field(offset_of!(ShortConvCall, c))))?;
        a.add(r10, qword_ptr(rdi + field(offset_of!(ShortConvCall, x))))?;
        a.imul_3(r11, rsi, 32)?;
        a.add(
            r11,
            qword_ptr(rdi + field(offset_of!(ShortConvCall, output))),
        )?;
        a.imul_3(rdx, rsi, (64 * taps) as i32)?;
        a.add(
            rdx,
            qword_ptr(rdi + field(offset_of!(ShortConvCall, state))),
        )?;
        a.imul_3(rcx, rsi, (32 * taps) as i32)?;
        a.add(
            rcx,
            qword_ptr(rdi + field(offset_of!(ShortConvCall, weights))),
        )?;
        for (x, y, value) in [
            (xmm12, ymm12, 1u32),
            (xmm13, ymm13, 0x7fff),
            (xmm15, ymm15, 0xffff_0000),
        ] {
            a.mov(esi, value)?;
            a.vmovd(x, esi)?;
            a.vpbroadcastd(y, x)?;
        }
        a.set_label(&mut row)?;
        for h in 0..2i32 {
            let (f, w) = (32 * h, 16 * h);
            // bx = bf16(bf16(B) · bf16(x)).
            a.vmovups(ymm8, ymmword_ptr(r8 + f))?;
            round(&mut a, ymm8)?;
            a.vmovups(ymm9, ymmword_ptr(r10 + f))?;
            round(&mut a, ymm9)?;
            a.vmulps(ymm8, ymm8, ymm9)?;
            round(&mut a, ymm8)?;
            // The state shifts by one tap, `bx` last.
            for j in 0..taps - 1 {
                a.vmovups(regs[j], ymmword_ptr(rdx + (64 * (j + 1)) as i32 + f))?;
                a.vmovups(ymmword_ptr(rdx + (64 * j) as i32 + f), regs[j])?;
            }
            a.vmovups(ymmword_ptr(rdx + (64 * (taps - 1)) as i32 + f), ymm8)?;
            // conv = bf16(Σ_j w_j · s_j), oldest first.
            for j in 0..taps {
                a.vpmovzxwd(ymm11, xmmword_ptr(rcx + (32 * j) as i32 + w))?;
                a.vpslld(ymm11, ymm11, 16u32)?;
                let s = if j + 1 < taps { regs[j] } else { ymm8 };
                if j == 0 {
                    a.vmulps(ymm10, s, ymm11)?;
                } else {
                    a.vfmadd231ps(ymm10, s, ymm11)?;
                }
            }
            round(&mut a, ymm10)?;
            // y = bf16(bf16(C) · conv), eight bf16.
            a.vmovups(ymm9, ymmword_ptr(r9 + f))?;
            round(&mut a, ymm9)?;
            a.vmulps(ymm9, ymm9, ymm10)?;
            round(&mut a, ymm9)?;
            a.vpsrld(ymm9, ymm9, 16u32)?;
            a.vextracti128(xmm11, ymm9, 1)?;
            a.vpackusdw(xmm9, xmm9, xmm11)?;
            a.vmovdqu(xmmword_ptr(r11 + w), xmm9)?;
        }
        a.add(r8, (width * 4) as i32)?;
        a.add(r9, (width * 4) as i32)?;
        a.add(r10, (width * 4) as i32)?;
        a.add(r11, (width * 2) as i32)?;
        a.dec(rax)?;
        a.jnz(row)?;
        a.set_label(&mut exit)?;
        a.vzeroupper()?;
        a.ret()?;
        a.assemble(0)
    };
    unsafe { GeneratedKernel::new(build().map_err(err)?) }
}

/// `(call: *const HeadGateCall, head)`: LFM3's gate on one head's output, the `rows` rows of the
/// block in order, as `Lfm3Attention` computes it in bf16: `s = bf16(σ(g))`, `g` the head's bf16
/// logit (`1 / (1 + e)` for `g ≥ 0`, `e / (1 + e)` below, `e = exp(−|g|)`, the exponential of the
/// routers), then `o = bf16(o · s)` over the head's `head_dim` values, in place. Same contract and
/// order of operations as `arch/aarch64/execution/short_conv.rs`.
pub(crate) fn head_gate(heads: usize, head_dim: usize) -> Result<GeneratedKernel, ExecutionError> {
    use super::moe::{emit_exp, emit_round_bf16};
    use crate::jit::execution::abi::HeadGateCall;
    if heads == 0
        || head_dim == 0
        || !head_dim.is_multiple_of(8)
        || heads * head_dim * 2 > i32::MAX as usize / 2
    {
        return Err(ExecutionError(format!(
            "no head gate of {heads} heads of {head_dim}"
        )));
    }
    let field = |f: usize| f as i32;
    let build = || -> Result<Vec<u8>, iced_x86::IcedError> {
        let mut a = CodeAssembler::new(64)?;
        a.set_prefer_vex(true);
        let mut exit = a.create_label();
        let mut row = a.create_label();
        // rcx = rows, r8 = the head's output in the row, r9 = its logit.
        a.mov(rcx, qword_ptr(rdi + field(offset_of!(HeadGateCall, rows))))?;
        a.mov(rcx, qword_ptr(rcx))?;
        a.test(rcx, rcx)?;
        a.je(exit)?;
        a.imul_3(r8, rsi, (head_dim * 2) as i32)?;
        a.add(r8, qword_ptr(rdi + field(offset_of!(HeadGateCall, output))))?;
        a.lea(r9, qword_ptr(rsi + rsi))?;
        a.add(r9, qword_ptr(rdi + field(offset_of!(HeadGateCall, gate))))?;
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
        a.set_label(&mut row)?;
        {
            let mut negative = a.create_label();
            let mut done = a.create_label();
            // g, then −|g|, e = exp(−|g|), σ(g).
            a.movzx(eax, word_ptr(r9))?;
            a.shl(eax, 16)?;
            a.vmovd(xmm5, eax)?;
            a.or(eax, 0x8000_0000u32 as i32)?;
            a.vmovd(xmm0, eax)?;
            emit_exp(&mut a)?;
            a.vaddss(xmm6, xmm0, xmm15)?;
            a.vmovd(eax, xmm5)?;
            a.test(eax, eax)?;
            a.js(negative)?;
            a.vdivss(xmm0, xmm15, xmm6)?;
            a.jmp(done)?;
            a.set_label(&mut negative)?;
            a.vdivss(xmm0, xmm0, xmm6)?;
            a.set_label(&mut done)?;
            emit_round_bf16(&mut a)?;
        }
        a.vbroadcastss(ymm5, xmm0)?;
        // The rounding constants, which the exponential clobbers: 1, 0x7fff, 0xffff0000.
        for (x, y, value) in [
            (xmm1, ymm1, 1u32),
            (xmm2, ymm2, 0x7fff),
            (xmm3, ymm3, 0xffff_0000),
        ] {
            a.mov(eax, value)?;
            a.vmovd(x, eax)?;
            a.vpbroadcastd(y, x)?;
        }
        // o = bf16(o · s), eight at a time.
        for i in 0..(head_dim / 8) as i32 {
            a.vpmovzxwd(ymm6, xmmword_ptr(r8 + 16 * i))?;
            a.vpslld(ymm6, ymm6, 16u32)?;
            a.vmulps(ymm6, ymm6, ymm5)?;
            a.vpsrld(ymm7, ymm6, 16u32)?;
            a.vpand(ymm7, ymm7, ymm1)?;
            a.vpaddd(ymm6, ymm6, ymm7)?;
            a.vpaddd(ymm6, ymm6, ymm2)?;
            a.vpand(ymm6, ymm6, ymm3)?;
            a.vpsrld(ymm6, ymm6, 16u32)?;
            a.vextracti128(xmm7, ymm6, 1)?;
            a.vpackusdw(xmm6, xmm6, xmm7)?;
            a.vmovdqu(xmmword_ptr(r8 + 16 * i), xmm6)?;
        }
        a.add(r8, (heads * head_dim * 2) as i32)?;
        a.add(r9, (heads * 2) as i32)?;
        a.dec(rcx)?;
        a.jnz(row)?;
        a.set_label(&mut exit)?;
        a.vzeroupper()?;
        a.ret()?;
        a.assemble(0)
    };
    unsafe { GeneratedKernel::new(build().map_err(err)?) }
}
