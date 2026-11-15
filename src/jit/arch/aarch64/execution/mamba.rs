//! Mamba2's nodes on AArch64 (Granite 4), in the DeltaNet's buffers and state: the step and its
//! decay, the causal conv with its biases, the recurrence without the delta rule, the gated norm
//! over the whole row. The order of the operations is the x86 kernels'
//! (`arch/x86/execution/delta.rs`, `delta_scalar.rs`): the same products and sums.

use super::math;
use super::{ExecutionError, GeneratedKernel};
use crate::jit::arch::aarch64::elem::{narrow_quad, widen};
use crate::jit::arch::aarch64::util::{add_imm, exec_err, madd_imm, mul_imm, Frame};
pub(crate) use crate::jit::execution::abi::{ConvCall, GateCall, NormCall, RecurrenceCall, Shape};
use a64::*;
use std::mem::offset_of;

/// `(call: *const GateCall, row·nv + head)`: `dt = softplus(a + dt_bias)` (`a` itself above
/// twenty), stored as `beta`, and `decay = exp(−exp(a_log)·dt)`. The row of `ab` holds the `nv`
/// steps alone.
pub(crate) fn gates(nv: usize, fma: bool) -> Result<GeneratedKernel, ExecutionError> {
    if nv == 0 || nv > i32::MAX as usize / 4 {
        return Err(ExecutionError("invalid Mamba2 gate heads".into()));
    }
    let mut a = Asm::new();
    let exp_label = a.label();
    let log_label = a.label();
    let dt = a.label();
    let frame = Frame::enter(&mut a, true, &[X19, X20, X21, X22], false, 16);
    let field = |f: usize| mem(X19, f as i64);
    a.mov(X19, X0);
    a.mov(X20, X1);
    a.mov_imm(X9, nv as u64);
    a.udiv(X10, X20, X9);
    a.msub(X21, X10, X9, X20);
    a.ldr(X9, field(offset_of!(GateCall, ab)));
    a.ldr(s(0), idx_scaled(X9, X20));
    a.ldr(X9, field(offset_of!(GateCall, bias)));
    a.ldrh(W10, idx_scaled(X9, X21));
    a.lsl(W10, W10, 16);
    a.fmov_from_gp(s(1), W10);
    a.fadd(s(0), s(0), s(1));
    a.mov_imm(W10, u64::from(20.0f32.to_bits()));
    a.fmov_from_gp(s(1), W10);
    a.fcmp(s(0), Some(s(1)));
    a.bgt(dt);
    a.ldr(X0, field(offset_of!(GateCall, tables)));
    a.bl(exp_label);
    a.fmov_imm(s(1), 1.0);
    a.fadd(s(0), s(0), s(1));
    a.bl(log_label);
    a.bind(dt);
    a.str(s(0), at(SP));
    a.ldr(X9, field(offset_of!(GateCall, beta)));
    a.str(s(0), idx_scaled(X9, X20));
    a.ldr(X9, field(offset_of!(GateCall, a_log)));
    a.ldr(s(0), idx_scaled(X9, X21));
    a.ldr(X0, field(offset_of!(GateCall, tables)));
    a.bl(exp_label);
    a.fneg(s(0), s(0));
    a.ldr(s(1), at(SP));
    a.fmul(s(0), s(0), s(1));
    a.bl(exp_label);
    a.ldr(X9, field(offset_of!(GateCall, decay)));
    a.str(s(0), idx_scaled(X9, X20));
    frame.leave(&mut a);
    a.ret();
    a.bind(exp_label);
    math::exp_body(&mut a, fma);
    a.bind(log_label);
    math::log_body(&mut a, fma);
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// `(call: *const ConvCall, channel block)`: sixteen channels, every row of the block in order. Per
/// channel the state shifts by one and takes the row's value, `Σ_j w[j]·state[j]` left to right,
/// plus the bias, then `x / (1 + exp(−x))`.
pub(crate) fn convolution(shape: Shape, fma: bool) -> Result<GeneratedKernel, ExecutionError> {
    shape.check()?;
    if !shape.conv().is_multiple_of(16) {
        return Err(ExecutionError(format!(
            "the AArch64 Mamba2 conv serves whole blocks of 16 channels, not {}",
            shape.conv()
        )));
    }
    let ck = shape.ck;
    let mut a = Asm::new();
    let exp_label = a.label();
    let row = a.label();
    let channel = a.label();
    let exit = a.label();
    let field = |f: usize| mem(X19, f as i64);
    // x19 call, x20 rows left, x21 the row's 16 values, x22 state, x23 weights, x24 biases, x25
    // channels left.
    let frame = Frame::enter(
        &mut a,
        true,
        &[X19, X20, X21, X22, X23, X24, X25, X26],
        false,
        0,
    );
    a.mov(X19, X0);
    a.mov(X26, X1);
    a.ldr(X20, field(offset_of!(ConvCall, rows)));
    a.ldr(X20, at(X20));
    a.ldr(X21, field(offset_of!(ConvCall, data)));
    madd_imm(&mut a, X21, X26, 64, X21, X9);
    a.cbz(X20, exit);
    a.bind(row);
    a.ldr(X22, field(offset_of!(ConvCall, state)));
    madd_imm(&mut a, X22, X26, (64 * ck) as u64, X22, X9);
    a.ldr(X23, field(offset_of!(ConvCall, weights)));
    madd_imm(&mut a, X23, X26, (32 * ck) as u64, X23, X9);
    a.ldr(X24, field(offset_of!(ConvCall, bias)));
    madd_imm(&mut a, X24, X26, 64, X24, X9);
    a.mov_imm(X25, 16);
    a.bind(channel);
    // The state of the channel shifts by one, the row's value last.
    for j in 0..ck - 1 {
        a.ldr(s(16), mem(X22, 4 * (j as i64 + 1)));
        a.str(s(16), mem(X22, 4 * j as i64));
    }
    a.ldr(s(16), at(X21));
    a.str(s(16), mem(X22, 4 * (ck as i64 - 1)));
    a.movi(v(16).d2(), 0);
    for j in 0..ck {
        a.ldrh(W10, mem(X23, 2 * j as i64));
        a.lsl(W10, W10, 16);
        a.fmov_from_gp(s(17), W10);
        a.ldr(s(18), mem(X22, 4 * j as i64));
        a.fmul(s(17), s(17), s(18));
        a.fadd(s(16), s(16), s(17));
    }
    a.ldr(s(17), at(X24));
    a.fadd(s(16), s(16), s(17));
    a.fneg(s(0), s(16));
    a.ldr(X0, field(offset_of!(ConvCall, tables)));
    a.bl(exp_label);
    a.fmov_imm(s(1), 1.0);
    a.fadd(s(0), s(0), s(1));
    a.fdiv(s(16), s(16), s(0));
    a.str(s(16), post(X21, 4));
    add_imm(&mut a, X22, X22, (4 * ck) as i64, X10);
    add_imm(&mut a, X23, X23, (2 * ck) as i64, X10);
    add_imm(&mut a, X24, X24, 4, X10);
    a.subs(X25, X25, 1);
    a.bne(channel);
    // The next row: back over the block's 16 values, then one row forward.
    add_imm(&mut a, X21, X21, (shape.conv() * 4) as i64 - 64, X10);
    a.subs(X20, X20, 1);
    a.bne(row);
    a.bind(exit);
    frame.leave(&mut a);
    a.ret();
    a.bind(exp_label);
    math::exp_body(&mut a, fma);
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// `(call: *const RecurrenceCall, head)`: every row of the block in order. Four lanes of the value
/// head at a time: for each `kd`, `dB = k[kd]·dt`, `s ← s·decay + dB·x`, `out += s·q[kd]`; then
/// `out += x·D[head]`. The state bf16 (`state16`) or f32, rounded after `out` has read it, as on
/// x86.
pub(crate) fn recurrence(
    shape: Shape,
    fma: bool,
    state16: bool,
) -> Result<GeneratedKernel, ExecutionError> {
    shape.check()?;
    if !shape.hvd.is_multiple_of(4) {
        return Err(ExecutionError(
            "the AArch64 Mamba2 recurrence serves value heads of a multiple of 4".into(),
        ));
    }
    let elem = if state16 { 2 } else { 4 };
    let line = (shape.hvd * elem) as i64;
    let mut a = Asm::new();
    let row = a.label();
    let lanes = a.label();
    let over_k = a.label();
    let exit = a.label();
    let field = |f: usize| mem(X19, f as i64);
    // x19 call, x20 head, x21 row, x22 rows, x23 the head's state, x24 lane group,
    // x9 state cursor, x10 k cursor, x11 q cursor, x12 kd left, x13 q of the row, x14 x of the row,
    // x15 out of the row.
    let frame = Frame::enter(&mut a, false, &[X19, X20, X21, X22, X23, X24], false, 0);
    a.mov(X19, X0);
    a.mov(X20, X1);
    a.ldr(X22, field(offset_of!(RecurrenceCall, rows)));
    a.ldr(X22, at(X22));
    a.cbz(X22, exit);
    a.ldr(X23, field(offset_of!(RecurrenceCall, state)));
    madd_imm(
        &mut a,
        X23,
        X20,
        (shape.hkd * shape.hvd * elem) as u64,
        X23,
        X9,
    );
    a.ldr(X9, field(offset_of!(RecurrenceCall, d)));
    a.ldr(s(30), idx_scaled(X9, X20));
    a.dup(v(30).s4(), v(30).s(0));
    a.mov_imm(X21, 0);
    a.bind(row);
    // beta (dt) and decay of (row, head).
    mul_imm(&mut a, X10, X21, shape.nv as u64, X9);
    a.add(X10, X10, X20);
    a.ldr(X9, field(offset_of!(RecurrenceCall, beta)));
    a.ldr(s(29), idx_scaled(X9, X10));
    a.dup(v(29).s4(), v(29).s(0));
    a.ldr(X9, field(offset_of!(RecurrenceCall, decay)));
    a.ldr(s(28), idx_scaled(X9, X10));
    a.dup(v(28).s4(), v(28).s(0));
    // q (C) of the head's group, k (B) after it, x of the head.
    a.ldr(X13, field(offset_of!(RecurrenceCall, qkv)));
    madd_imm(&mut a, X13, X21, (shape.conv() * 4) as u64, X13, X9);
    add_imm(&mut a, X14, X13, (shape.key() * 8) as i64, X9);
    madd_imm(&mut a, X14, X20, (shape.hvd * 4) as u64, X14, X9);
    a.mov_imm(X9, (shape.nv / shape.nk) as u64);
    a.udiv(X10, X20, X9);
    madd_imm(&mut a, X13, X10, (shape.hkd * 4) as u64, X13, X9);
    a.ldr(X15, field(offset_of!(RecurrenceCall, output)));
    madd_imm(&mut a, X15, X21, (shape.value() * 4) as u64, X15, X9);
    madd_imm(&mut a, X15, X20, (shape.hvd * 4) as u64, X15, X9);
    a.mov_imm(X24, 0);
    a.bind(lanes);
    // v20 = x, v21 = out of the four lanes.
    a.ldr(q(20), idx_scaled(X14, X24));
    a.movi(v(21).d2(), 0);
    add_imm(&mut a, X10, X13, (shape.key() * 4) as i64, X9);
    a.mov(X11, X13);
    if state16 {
        a.add(X9, X23, X24.lsl(3));
    } else {
        a.add(X9, X23, X24.lsl(4));
    }
    a.mov_imm(X12, shape.hkd as u64);
    a.bind(over_k);
    a.ld1r(v(16).s4(), post(X10, 4));
    a.fmul(v(16).s4(), v(16).s4(), v(29).s4());
    a.ld1r(v(17).s4(), post(X11, 4));
    if state16 {
        a.ldr(d(18), at(X9));
        widen(&mut a, 18, 18, false);
    } else {
        a.ldr(q(18), at(X9));
    }
    a.fmul(v(18).s4(), v(18).s4(), v(28).s4());
    if fma {
        a.fmla(v(18).s4(), v(16).s4(), v(20).s4());
        a.fmla(v(21).s4(), v(18).s4(), v(17).s4());
    } else {
        a.fmul(v(19).s4(), v(16).s4(), v(20).s4());
        a.fadd(v(18).s4(), v(18).s4(), v(19).s4());
        a.fmul(v(19).s4(), v(18).s4(), v(17).s4());
        a.fadd(v(21).s4(), v(21).s4(), v(19).s4());
    }
    if state16 {
        narrow_quad(&mut a, 19, 18, 22);
        a.str(d(19), at(X9));
    } else {
        a.str(q(18), at(X9));
    }
    add_imm(&mut a, X9, X9, line, X16);
    a.subs(X12, X12, 1);
    a.bne(over_k);
    if fma {
        a.fmla(v(21).s4(), v(20).s4(), v(30).s4());
    } else {
        a.fmul(v(19).s4(), v(20).s4(), v(30).s4());
        a.fadd(v(21).s4(), v(21).s4(), v(19).s4());
    }
    a.str(q(21), idx_scaled(X15, X24));
    a.add(X24, X24, 1);
    a.mov_imm(X16, (shape.hvd / 4) as u64);
    a.cmp(X24, X16);
    a.bne(lanes);
    a.add(X21, X21, 1);
    a.cmp(X21, X22);
    a.bne(row);
    a.bind(exit);
    frame.leave(&mut a);
    a.ret();
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// `(call: *const NormCall, row)`: `g = y · x/(1 + exp(−z))` over `y`, `Σ g²` left to right, `r =
/// 1/sqrt(Σ/dim + eps)`, then `w · (g · r)` rounded to bf16 to nearest even.
pub(crate) fn normed_gate(dim: usize, fma: bool) -> Result<GeneratedKernel, ExecutionError> {
    if dim == 0 || dim > i32::MAX as usize / 4 {
        return Err(ExecutionError("invalid Mamba2 gated norm width".into()));
    }
    let mut a = Asm::new();
    let exp_label = a.label();
    let gate = a.label();
    let apply = a.label();
    let field = |f: usize| mem(X19, f as i64);
    // x19 call, x20 y, x21 z, x22 w, x23 out, x24 index; [sp] eps, [sp+4] Σ, [sp+8] z.
    let frame = Frame::enter(&mut a, true, &[X19, X20, X21, X22, X23, X24], false, 16);
    a.mov(X19, X0);
    mul_imm(&mut a, X9, X1, (dim * 4) as u64, X10);
    mul_imm(&mut a, X11, X1, (dim * 2) as u64, X10);
    a.ldr(X20, field(offset_of!(NormCall, output)));
    a.add(X20, X20, X9);
    a.ldr(X21, field(offset_of!(NormCall, gate)));
    a.add(X21, X21, X9);
    a.ldr(X22, field(offset_of!(NormCall, weights)));
    a.ldr(X23, field(offset_of!(NormCall, output_bf16)));
    a.add(X23, X23, X11);
    a.ldr(s(0), field(offset_of!(NormCall, eps)));
    a.str(s(0), at(SP));
    a.str(WZR, mem(SP, 4));
    a.mov_imm(X24, 0);
    a.bind(gate);
    a.ldr(s(0), idx_scaled(X21, X24));
    a.str(s(0), mem(SP, 8));
    a.fneg(s(0), s(0));
    a.ldr(X0, field(offset_of!(NormCall, tables)));
    a.bl(exp_label);
    a.fmov_imm(s(1), 1.0);
    a.fadd(s(0), s(0), s(1));
    a.ldr(s(1), mem(SP, 8));
    a.fdiv(s(1), s(1), s(0));
    a.ldr(s(2), idx_scaled(X20, X24));
    a.fmul(s(2), s(2), s(1));
    a.str(s(2), idx_scaled(X20, X24));
    a.fmul(s(2), s(2), s(2));
    a.ldr(s(3), mem(SP, 4));
    a.fadd(s(3), s(3), s(2));
    a.str(s(3), mem(SP, 4));
    a.add(X24, X24, 1);
    a.mov_imm(X16, dim as u64);
    a.cmp(X24, X16);
    a.bne(gate);
    a.ldr(s(1), mem(SP, 4));
    a.mov_imm(W10, u64::from((dim as f32).to_bits()));
    a.fmov_from_gp(s(2), W10);
    a.fdiv(s(1), s(1), s(2));
    a.ldr(s(2), at(SP));
    a.fadd(s(1), s(1), s(2));
    a.fsqrt(s(1), s(1));
    a.fmov_imm(s(3), 1.0);
    a.fdiv(s(3), s(3), s(1));
    a.mov_imm(X24, 0);
    a.bind(apply);
    a.ldr(s(0), idx_scaled(X20, X24));
    a.fmul(s(0), s(0), s(3));
    a.ldrh(W10, idx_scaled(X22, X24));
    a.lsl(W10, W10, 16);
    a.fmov_from_gp(s(2), W10);
    a.fmul(s(2), s(2), s(0));
    // Round to nearest even: (bits + 0x7fff + bit16) >> 16.
    a.fmov_to_gp(W10, s(2));
    a.lsr(W11, W10, 16);
    a.and(W11, W11, 1u64);
    a.add(W10, W10, W11);
    a.mov_imm(W11, 0x7fff);
    a.add(W10, W10, W11);
    a.lsr(W10, W10, 16);
    a.strh(W10, idx_scaled(X23, X24));
    a.add(X24, X24, 1);
    a.mov_imm(X16, dim as u64);
    a.cmp(X24, X16);
    a.bne(apply);
    frame.leave(&mut a);
    a.ret();
    a.bind(exp_label);
    math::exp_body(&mut a, fma);
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}
