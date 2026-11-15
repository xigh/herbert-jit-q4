//! LiquidAI's gated short convolution (`lfm2`, `lfm2_moe`), AArch64 (NEON): one task per block of
//! sixteen channels, every row of the block in order, as `Lfm2ShortConv` computes it in bf16. Same
//! contract and order of operations as `arch/x86/execution/short_conv.rs`.
use super::{ExecutionError, GeneratedKernel};
use crate::jit::arch::aarch64::elem::{narrow_quad, widen};
use crate::jit::arch::aarch64::util::{exec_err, mul_imm};
pub(crate) use crate::jit::execution::abi::ShortConvCall;
use a64::*;
use std::mem::offset_of;

/// The four f32 of `v(r)` rounded to bf16 (nearest even), kept as f32; `v(t)` and `v(u)` are
/// clobbered.
fn round(a: &mut Asm, r: u8, t: u8, u: u8) {
    narrow_quad(a, t, r, u);
    widen(a, r, t, false);
}

/// `(call: *const ShortConvCall, channel block)`: sixteen channels of `width`, the `rows` rows of
/// the block in order. For each row and channel: `B`, `C`, `x` rounded to bf16 (the module's bf16
/// `in_proj`), `bx = bf16(B · x)`; the channel's state (its last `taps` values of `bx`, oldest
/// first) shifts by one and takes `bx`; `conv = bf16(w_0·s_0 + w_1·s_1 + …)`, the product of the
/// oldest tap then one fused multiply-add per tap, oldest first; `y = bf16(C · conv)` stored as
/// bf16. Four quads of four channels.
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
    let mut a = Asm::new();
    let row = a.label();
    let exit = a.label();
    let field = |f: usize| mem(X0, f as i64);
    // x2 rows, x3 B, x4 C, x5 x, x6 output, x7 state, x8 weights of the block; x9, x10 scratch.
    a.ldr(X2, field(offset_of!(ShortConvCall, rows)));
    a.ldr(X2, at(X2));
    a.cbz(X2, exit);
    mul_imm(&mut a, X9, X1, 64, X10);
    a.ldr(X3, field(offset_of!(ShortConvCall, b)));
    a.add(X3, X3, X9);
    a.ldr(X4, field(offset_of!(ShortConvCall, c)));
    a.add(X4, X4, X9);
    a.ldr(X5, field(offset_of!(ShortConvCall, x)));
    a.add(X5, X5, X9);
    mul_imm(&mut a, X9, X1, 32, X10);
    a.ldr(X6, field(offset_of!(ShortConvCall, output)));
    a.add(X6, X6, X9);
    mul_imm(&mut a, X9, X1, (64 * taps) as u64, X10);
    a.ldr(X7, field(offset_of!(ShortConvCall, state)));
    a.add(X7, X7, X9);
    mul_imm(&mut a, X9, X1, (32 * taps) as u64, X10);
    a.ldr(X8, field(offset_of!(ShortConvCall, weights)));
    a.add(X8, X8, X9);
    a.mov_imm(X9, (width * 4) as u64);
    a.mov_imm(X10, (width * 2) as u64);
    a.bind(row);
    for i in 0..4i64 {
        let (f, w) = (16 * i, 8 * i);
        // bx = bf16(bf16(B) · bf16(x)).
        a.ldr(q(16), mem(X3, f));
        round(&mut a, 16, 20, 21);
        a.ldr(q(17), mem(X5, f));
        round(&mut a, 17, 20, 21);
        a.fmul(v(16).s4(), v(16).s4(), v(17).s4());
        round(&mut a, 16, 20, 21);
        // The state shifts by one tap, `bx` last.
        for j in 0..taps as i64 - 1 {
            a.ldr(q(j as u8), mem(X7, 64 * (j + 1) + f));
            a.str(q(j as u8), mem(X7, 64 * j + f));
        }
        a.str(q(16), mem(X7, 64 * (taps as i64 - 1) + f));
        // conv = bf16(Σ_j w_j · s_j), oldest first.
        for j in 0..taps {
            a.ldr(d(19), mem(X8, (32 * j) as i64 + w));
            widen(&mut a, 19, 19, false);
            let s = if j + 1 < taps { j as u8 } else { 16 };
            if j == 0 {
                a.fmul(v(18).s4(), v(s).s4(), v(19).s4());
            } else {
                a.fmla(v(18).s4(), v(s).s4(), v(19).s4());
            }
        }
        round(&mut a, 18, 20, 21);
        // y = bf16(bf16(C) · conv), four bf16.
        a.ldr(q(17), mem(X4, f));
        round(&mut a, 17, 20, 21);
        a.fmul(v(17).s4(), v(17).s4(), v(18).s4());
        narrow_quad(&mut a, 19, 17, 20);
        a.str(d(19), mem(X6, w));
    }
    a.add(X3, X3, X9);
    a.add(X4, X4, X9);
    a.add(X5, X5, X9);
    a.add(X6, X6, X10);
    a.subs(X2, X2, 1);
    a.bne(row);
    a.bind(exit);
    a.ret();
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// `(call: *const HeadGateCall, head)`: LFM3's gate on one head's output, the `rows` rows of the
/// block in order, as `Lfm3Attention` computes it in bf16: `s = bf16(σ(g))`, `g` the head's bf16
/// logit (`1 / (1 + e)` for `g ≥ 0`, `e / (1 + e)` below, `e = exp(−|g|)`, the exponential of the
/// routers), then `o = bf16(o · s)` over the head's `head_dim` values, in place. Same contract and
/// order of operations as `arch/x86/execution/short_conv.rs`.
pub(crate) fn head_gate(heads: usize, head_dim: usize) -> Result<GeneratedKernel, ExecutionError> {
    use super::moe::{emit_exp, emit_exp_constants, round_bf16_s};
    use crate::jit::execution::abi::HeadGateCall;
    if heads == 0
        || head_dim == 0
        || !head_dim.is_multiple_of(4)
        || head_dim * 2 > 4096
        || heads * head_dim * 2 > i32::MAX as usize / 2
    {
        return Err(ExecutionError(format!(
            "no head gate of {heads} heads of {head_dim}"
        )));
    }
    let mut a = Asm::new();
    let row = a.label();
    let exit = a.label();
    let negative = a.label();
    let done = a.label();
    let field = |f: usize| mem(X0, f as i64);
    // x2 rows, x3 the head's output in the row, x4 its logit; x9, x10 scratch; x11, x12 the row
    // strides.
    a.ldr(X2, field(offset_of!(HeadGateCall, rows)));
    a.ldr(X2, at(X2));
    a.cbz(X2, exit);
    mul_imm(&mut a, X9, X1, (head_dim * 2) as u64, X10);
    a.ldr(X3, field(offset_of!(HeadGateCall, output)));
    a.add(X3, X3, X9);
    a.add(X9, X1, X1);
    a.ldr(X4, field(offset_of!(HeadGateCall, gate)));
    a.add(X4, X4, X9);
    emit_exp_constants(&mut a);
    a.mov_imm(X11, (heads * head_dim * 2) as u64);
    a.mov_imm(X12, (heads * 2) as u64);
    a.bind(row);
    // g, then −|g|, e = exp(−|g|), σ(g).
    a.ldrh(W9, at(X4));
    a.lsl(W9, W9, 16);
    a.fmov_from_gp(s(6), W9);
    a.orr(W9, W9, 0x8000_0000u64);
    a.fmov_from_gp(s(0), W9);
    emit_exp(&mut a);
    a.fadd(s(7), s(0), s(23));
    a.fmov_to_gp(W9, s(6));
    a.tbnz(W9, 31, negative);
    a.fdiv(s(0), s(23), s(7));
    a.b(done);
    a.bind(negative);
    a.fdiv(s(0), s(0), s(7));
    a.bind(done);
    round_bf16_s(&mut a, 0, 1);
    a.dup(v(24).s4(), v(0).s(0));
    // o = bf16(o · s), four at a time.
    for i in 0..(head_dim / 4) as i64 {
        a.ldr(d(25), mem(X3, 8 * i));
        widen(&mut a, 25, 25, false);
        a.fmul(v(25).s4(), v(25).s4(), v(24).s4());
        narrow_quad(&mut a, 26, 25, 27);
        a.str(d(26), mem(X3, 8 * i));
    }
    a.add(X3, X3, X11);
    a.add(X4, X4, X12);
    a.subs(X2, X2, 1);
    a.bne(row);
    a.bind(exit);
    a.ret();
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}
