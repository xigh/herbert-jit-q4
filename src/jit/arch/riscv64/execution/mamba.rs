//! Mamba2's nodes on RISC-V (Granite 4), in the DeltaNet's buffers and state: the step and its
//! decay, the causal conv with its biases, the recurrence without the delta rule (RVV, the value
//! head cut into slices of one register group), the gated norm over the whole row. The order of the
//! operations is the x86 and AArch64 kernels'.

use super::math;
use super::{ExecutionError, GeneratedKernel};
use crate::jit::arch::riscv64::util::{add_imm, exec_err, Frame};
pub(crate) use crate::jit::execution::abi::{ConvCall, GateCall, NormCall, RecurrenceCall, Shape};
use rv::*;
use std::mem::offset_of;

const E16M4: Vtype = Vtype::new(Sew::E16, Lmul::M4);
const E32M8: Vtype = Vtype::new(Sew::E32, Lmul::M8);

/// `rd = base + rs · imm` (through `tmp`).
fn madd_imm(a: &mut Asm, rd: X, rs: X, imm: u64, base: X, tmp: X) {
    a.li(tmp, imm as i64);
    a.mul(tmp, rs, tmp);
    a.add(rd, base, tmp);
}

/// `f = 1.0` (through `t0`).
fn one(a: &mut Asm, f: F) {
    a.li(T0, i64::from(1.0f32.to_bits()));
    a.fmv_w_x(f, T0);
}

/// `(call: *const GateCall, row·nv + head)`: `dt = softplus(a + dt_bias)` (`a + dt_bias` itself
/// above twenty), stored as `beta`, and `decay = exp(−exp(a_log)·dt)`. The row of `ab` holds the
/// `nv` steps alone.
pub(crate) fn gates(nv: usize, fma: bool) -> Result<GeneratedKernel, ExecutionError> {
    if nv == 0 || nv > i32::MAX as usize / 4 {
        return Err(ExecutionError("invalid Mamba2 gate heads".into()));
    }
    let mut a = Asm::new();
    let exp_label = a.label();
    let log_label = a.label();
    let dt = a.label();
    let f = |field: usize| field as i64;
    // s1 call, s2 task, s3 head; 0(sp) holds dt.
    let frame = Frame::enter(&mut a, true, &[S1, S2, S3], &[], 16);
    a.mv(S1, A0);
    a.mv(S2, A1);
    a.li(T0, nv as i64);
    a.remu(S3, S2, T0);
    a.ld(T0, S1, f(offset_of!(GateCall, ab)));
    a.sh2add(T0, S2, T0);
    a.flw(FA0, T0, 0);
    a.ld(T0, S1, f(offset_of!(GateCall, bias)));
    a.sh1add(T0, S3, T0);
    a.lhu(T1, T0, 0);
    a.slli(T1, T1, 16);
    a.fmv_w_x(FA1, T1);
    a.fadd_s(FA0, FA0, FA1);
    a.li(T0, i64::from(20.0f32.to_bits()));
    a.fmv_w_x(FA1, T0);
    a.flt_s(T0, FA1, FA0);
    a.bnez(T0, dt);
    a.ld(A0, S1, f(offset_of!(GateCall, tables)));
    a.call(exp_label);
    one(&mut a, FA1);
    a.fadd_s(FA0, FA0, FA1);
    a.call(log_label);
    a.bind(dt);
    a.fsw(FA0, SP, 0);
    a.ld(T0, S1, f(offset_of!(GateCall, beta)));
    a.sh2add(T0, S2, T0);
    a.fsw(FA0, T0, 0);
    a.ld(T0, S1, f(offset_of!(GateCall, a_log)));
    a.sh2add(T0, S3, T0);
    a.flw(FA0, T0, 0);
    a.ld(A0, S1, f(offset_of!(GateCall, tables)));
    a.call(exp_label);
    a.fneg_s(FA0, FA0);
    a.flw(FA1, SP, 0);
    a.fmul_s(FA0, FA0, FA1);
    a.call(exp_label);
    a.ld(T0, S1, f(offset_of!(GateCall, decay)));
    a.sh2add(T0, S2, T0);
    a.fsw(FA0, T0, 0);
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
            "the RISC-V Mamba2 conv serves whole blocks of 16 channels, not {}",
            shape.conv()
        )));
    }
    let ck = shape.ck;
    let mut a = Asm::new();
    let exp_label = a.label();
    let exit = a.label();
    let f = |field: usize| field as i64;
    // s1 call, s2 rows left, s3 the row's 16 values, s4 state, s5 weights, s6 biases, s7 channels
    // left, s8 block; fs0 the sum across the call of exp.
    let frame = Frame::enter(&mut a, true, &[S1, S2, S3, S4, S5, S6, S7, S8], &[FS0], 0);
    a.mv(S1, A0);
    a.mv(S8, A1);
    a.ld(S2, S1, f(offset_of!(ConvCall, rows)));
    a.ld(S2, S2, 0);
    a.ld(S3, S1, f(offset_of!(ConvCall, data)));
    madd_imm(&mut a, S3, S8, 64, S3, T0);
    a.beqz(S2, exit);
    let row = a.here();
    a.ld(S4, S1, f(offset_of!(ConvCall, state)));
    madd_imm(&mut a, S4, S8, (64 * ck) as u64, S4, T0);
    a.ld(S5, S1, f(offset_of!(ConvCall, weights)));
    madd_imm(&mut a, S5, S8, (32 * ck) as u64, S5, T0);
    a.ld(S6, S1, f(offset_of!(ConvCall, bias)));
    madd_imm(&mut a, S6, S8, 64, S6, T0);
    a.li(S7, 16);
    let channel = a.here();
    for j in 0..ck - 1 {
        a.flw(FT0, S4, 4 * (j as i64 + 1));
        a.fsw(FT0, S4, 4 * j as i64);
    }
    a.flw(FT0, S3, 0);
    a.fsw(FT0, S4, 4 * (ck as i64 - 1));
    a.fmv_w_x(FS0, ZERO);
    for j in 0..ck {
        a.lhu(T1, S5, 2 * j as i64);
        a.slli(T1, T1, 16);
        a.fmv_w_x(FT1, T1);
        a.flw(FT2, S4, 4 * j as i64);
        a.fmul_s(FT1, FT1, FT2);
        a.fadd_s(FS0, FS0, FT1);
    }
    a.flw(FT1, S6, 0);
    a.fadd_s(FS0, FS0, FT1);
    a.fneg_s(FA0, FS0);
    a.ld(A0, S1, f(offset_of!(ConvCall, tables)));
    a.call(exp_label);
    one(&mut a, FA1);
    a.fadd_s(FA0, FA0, FA1);
    a.fdiv_s(FS0, FS0, FA0);
    a.fsw(FS0, S3, 0);
    a.addi(S3, S3, 4);
    add_imm(&mut a, S4, S4, (4 * ck) as i64, T0);
    add_imm(&mut a, S5, S5, (2 * ck) as i64, T0);
    a.addi(S6, S6, 4);
    a.addi(S7, S7, -1);
    a.bnez(S7, channel);
    // The next row: back over the block's 16 values, then one row forward.
    add_imm(&mut a, S3, S3, (shape.conv() * 4) as i64 - 64, T0);
    a.addi(S2, S2, -1);
    a.bnez(S2, row);
    a.bind(exit);
    frame.leave(&mut a);
    a.ret();
    a.bind(exp_label);
    math::exp_body(&mut a, fma);
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// `(call: *const RecurrenceCall, head)`: every row of the block in order, the value head by slices
/// of at most `2·vlenb` lanes (one `e32, m8` group, eight registers): for each `kd`, `dB =
/// k[kd]·dt`, `s ← s·decay + dB·x`, `out += s·q[kd]`; then `out += x·D[head]`. The state bf16
/// (`state16`) or f32, rounded after `out` has read it.
///
/// Groups: `v0` the product, `v8` x, `v16` the state row, `v24` out.
pub(crate) fn recurrence(
    shape: Shape,
    fma: bool,
    state16: bool,
) -> Result<GeneratedKernel, ExecutionError> {
    shape.check()?;
    let elem = if state16 { 2 } else { 4 };
    let line = (shape.hvd * elem) as i64;
    let slice = (2 * crate::core::isa::vlenb()).min(shape.hvd);
    let (prod, xv, st, out) = (V0, V8, V16, V24);
    let mut a = Asm::new();
    let exit = a.label();
    let f = |field: usize| field as i64;
    // s1 call, s2 head, s3 rows, s4 row, s5 the head's state, s6 q of the row, s7 x of the row, s8
    // out of the row; fs0 dt, fs1 decay, fs2 D.
    let frame = Frame::enter(
        &mut a,
        false,
        &[S1, S2, S3, S4, S5, S6, S7, S8],
        &[FS0, FS1, FS2],
        0,
    );
    a.mv(S1, A0);
    a.mv(S2, A1);
    a.ld(S3, S1, f(offset_of!(RecurrenceCall, rows)));
    a.ld(S3, S3, 0);
    a.beqz(S3, exit);
    a.ld(S5, S1, f(offset_of!(RecurrenceCall, state)));
    madd_imm(
        &mut a,
        S5,
        S2,
        (shape.hkd * shape.hvd * elem) as u64,
        S5,
        T0,
    );
    a.ld(T0, S1, f(offset_of!(RecurrenceCall, d)));
    a.sh2add(T0, S2, T0);
    a.flw(FS2, T0, 0);
    a.li(S4, 0);
    let row = a.here();
    a.li(T0, shape.nv as i64);
    a.mul(T1, S4, T0);
    a.add(T1, T1, S2);
    a.ld(T0, S1, f(offset_of!(RecurrenceCall, beta)));
    a.sh2add(T0, T1, T0);
    a.flw(FS0, T0, 0);
    a.ld(T0, S1, f(offset_of!(RecurrenceCall, decay)));
    a.sh2add(T0, T1, T0);
    a.flw(FS1, T0, 0);
    a.ld(S6, S1, f(offset_of!(RecurrenceCall, qkv)));
    madd_imm(&mut a, S6, S4, (shape.conv() * 4) as u64, S6, T0);
    add_imm(&mut a, S7, S6, (shape.key() * 8) as i64, T0);
    madd_imm(&mut a, S7, S2, (shape.hvd * 4) as u64, S7, T0);
    a.li(T0, (shape.nv / shape.nk) as i64);
    a.divu(T1, S2, T0);
    madd_imm(&mut a, S6, T1, (shape.hkd * 4) as u64, S6, T0);
    a.ld(S8, S1, f(offset_of!(RecurrenceCall, output)));
    madd_imm(&mut a, S8, S4, (shape.value() * 4) as u64, S8, T0);
    madd_imm(&mut a, S8, S2, (shape.hvd * 4) as u64, S8, T0);
    let mut so = 0;
    while so < shape.hvd {
        let width = slice.min(shape.hvd - so);
        a.li(T6, width as i64);
        a.vsetvli(ZERO, T6, E32M8);
        add_imm(&mut a, T0, S7, (so * 4) as i64, T1);
        a.vle(Sew::E32, xv, T0);
        a.vmv_v_i(out, 0);
        // t3 state row, t4 k (B), t5 q (C), a6 kd left.
        add_imm(&mut a, T3, S5, (so * elem) as i64, T1);
        add_imm(&mut a, T4, S6, (shape.key() * 4) as i64, T1);
        a.mv(T5, S6);
        a.li(A6, shape.hkd as i64);
        let over_k = a.here();
        a.flw(FT0, T4, 0);
        a.fmul_s(FT0, FT0, FS0);
        a.flw(FT1, T5, 0);
        if state16 {
            a.vsetvli(ZERO, ZERO, E16M4);
            a.vle(Sew::E16, prod, T3);
            a.vfwcvtbf16_f_f_v(st, prod);
            a.vsetvli(ZERO, ZERO, E32M8);
        } else {
            a.vle(Sew::E32, st, T3);
        }
        a.vfmul_vf(st, st, FS1);
        if fma {
            a.vfmacc_vf(st, FT0, xv);
            a.vfmacc_vf(out, FT1, st);
        } else {
            a.vfmul_vf(prod, xv, FT0);
            a.vfadd_vv(st, st, prod);
            a.vfmul_vf(prod, st, FT1);
            a.vfadd_vv(out, out, prod);
        }
        if state16 {
            a.vsetvli(ZERO, ZERO, E16M4);
            a.vfncvtbf16_f_f_w(prod, st);
            a.vse(Sew::E16, prod, T3);
            a.vsetvli(ZERO, ZERO, E32M8);
        } else {
            a.vse(Sew::E32, st, T3);
        }
        a.addi(T4, T4, 4);
        a.addi(T5, T5, 4);
        add_imm(&mut a, T3, T3, line, T1);
        a.addi(A6, A6, -1);
        a.bnez(A6, over_k);
        if fma {
            a.vfmacc_vf(out, FS2, xv);
        } else {
            a.vfmul_vf(prod, xv, FS2);
            a.vfadd_vv(out, out, prod);
        }
        add_imm(&mut a, T0, S8, (so * 4) as i64, T1);
        a.vse(Sew::E32, out, T0);
        so += width;
    }
    a.addi(S4, S4, 1);
    a.bltu(S4, S3, row);
    a.bind(exit);
    frame.leave(&mut a);
    a.ret();
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// `(call: *const NormCall, row)`: `g = y · z/(1 + exp(−z))` over `y`, `Σ g²` left to right, `r =
/// 1/sqrt(Σ/dim + eps)`, then `w · (g · r)` rounded to bf16 to nearest even.
pub(crate) fn normed_gate(dim: usize, fma: bool) -> Result<GeneratedKernel, ExecutionError> {
    if dim == 0 || dim > i32::MAX as usize / 4 {
        return Err(ExecutionError("invalid Mamba2 gated norm width".into()));
    }
    let mut a = Asm::new();
    let exp_label = a.label();
    let f = |field: usize| field as i64;
    // s1 call, s2 y, s3 z, s4 w, s5 out, s6 index; fs0 Σ, fs1 z, fs2 eps then r.
    let frame = Frame::enter(&mut a, true, &[S1, S2, S3, S4, S5, S6], &[FS0, FS1, FS2], 0);
    a.mv(S1, A0);
    a.li(T0, (dim * 4) as i64);
    a.mul(T1, A1, T0);
    a.li(T0, (dim * 2) as i64);
    a.mul(T2, A1, T0);
    a.ld(S2, S1, f(offset_of!(NormCall, output)));
    a.add(S2, S2, T1);
    a.ld(S3, S1, f(offset_of!(NormCall, gate)));
    a.add(S3, S3, T1);
    a.ld(S4, S1, f(offset_of!(NormCall, weights)));
    a.ld(S5, S1, f(offset_of!(NormCall, output_bf16)));
    a.add(S5, S5, T2);
    a.flw(FS2, S1, f(offset_of!(NormCall, eps)));
    a.fmv_w_x(FS0, ZERO);
    a.li(S6, 0);
    let gate = a.here();
    a.sh2add(T0, S6, S3);
    a.flw(FS1, T0, 0);
    a.fneg_s(FA0, FS1);
    a.ld(A0, S1, f(offset_of!(NormCall, tables)));
    a.call(exp_label);
    one(&mut a, FA1);
    a.fadd_s(FA0, FA0, FA1);
    a.fdiv_s(FS1, FS1, FA0);
    a.sh2add(T0, S6, S2);
    a.flw(FT0, T0, 0);
    a.fmul_s(FT0, FT0, FS1);
    a.fsw(FT0, T0, 0);
    a.fmul_s(FT0, FT0, FT0);
    a.fadd_s(FS0, FS0, FT0);
    a.addi(S6, S6, 1);
    a.li(T0, dim as i64);
    a.bltu(S6, T0, gate);
    a.li(T0, i64::from((dim as f32).to_bits()));
    a.fmv_w_x(FT0, T0);
    a.fdiv_s(FS0, FS0, FT0);
    a.fadd_s(FS0, FS0, FS2);
    a.fsqrt_s(FS0, FS0);
    one(&mut a, FS2);
    a.fdiv_s(FS2, FS2, FS0);
    a.li(S6, 0);
    let apply = a.here();
    a.sh2add(T0, S6, S2);
    a.flw(FT0, T0, 0);
    a.fmul_s(FT0, FT0, FS2);
    a.sh1add(T0, S6, S4);
    a.lhu(T1, T0, 0);
    a.slli(T1, T1, 16);
    a.fmv_w_x(FT1, T1);
    a.fmul_s(FT1, FT1, FT0);
    // Round to nearest even: (bits + 0x7fff + bit16) >> 16.
    a.fmv_x_w(T1, FT1);
    a.srliw(T2, T1, 16);
    a.andi(T2, T2, 1);
    a.addw(T1, T1, T2);
    a.li(T2, 0x7fff);
    a.addw(T1, T1, T2);
    a.srliw(T1, T1, 16);
    a.sh1add(T0, S6, S5);
    a.sh(T1, T0, 0);
    a.addi(S6, S6, 1);
    a.li(T0, dim as i64);
    a.bltu(S6, T0, apply);
    frame.leave(&mut a);
    a.ret();
    a.bind(exp_label);
    math::exp_body(&mut a, fma);
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}
