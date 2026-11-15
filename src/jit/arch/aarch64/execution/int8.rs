//! The node that quantizes the activations of the projections to int8 (`jit/int8.rs`).
use super::{ExecutionError, GeneratedKernel};
use crate::jit::arch::aarch64::util::{add_imm, exec_err, madd_imm};
use crate::jit::execution::abi::PackCall;
use crate::jit::int8::{a_row_bytes, GROUP};
use a64::*;
use std::mem::offset_of;

/// Emit `(call: *const PackCall, row: usize)`: row `row` of `A` (row-major, `k` bf16) quantized
/// into row `row` of the destination, `k` int8 then `k/32` f32 scales. Per group of 32 values: the
/// largest magnitude (`and` with `0x7fff`, `umax` of the halves, widened), `scale = max / 127`
/// stored, every value times `127 / max`, rounded to nearest even (`fcvtns`) and narrowed (`xtn`:
/// the values are within ±127). A group of zeros (or of values below 1e-20) is quantized with `max
/// = 1e-20`: its int8 values are zero. Rows past the current count are not touched.
pub fn quantize(k: usize) -> Result<GeneratedKernel, ExecutionError> {
    if k == 0 || !k.is_multiple_of(GROUP) {
        return Err(ExecutionError(format!(
            "no int8 quantization of rows of {k}: groups of {GROUP}"
        )));
    }
    let mut a = Asm::new();
    let exit = a.label();
    let field = |f: usize| mem(X0, f as i64);
    a.ldr(X8, field(offset_of!(PackCall, rows)));
    a.ldr(X8, at(X8));
    a.cmp(X1, X8);
    a.bhs(exit);
    a.ldr(X2, field(offset_of!(PackCall, source)));
    madd_imm(&mut a, X2, X1, (2 * k) as u64, X2, X11);
    a.ldr(X3, field(offset_of!(PackCall, dest)));
    madd_imm(&mut a, X3, X1, a_row_bytes(k) as u64, X3, X11);
    add_imm(&mut a, X4, X3, k as i64, X16);
    // Constants: v31 = 0x7fff halves, s30 = 1/127, s29 = 1e-20, s28 = 127.
    a.mov_imm(W9, 0x7fff);
    a.dup_gp(v(31).h8(), W9);
    a.mov_imm(W9, (1.0f32 / 127.0).to_bits() as u64);
    a.fmov_from_gp(s(30), W9);
    a.mov_imm(W9, 1.0e-20f32.to_bits() as u64);
    a.fmov_from_gp(s(29), W9);
    a.mov_imm(W9, 127.0f32.to_bits() as u64);
    a.fmov_from_gp(s(28), W9);
    let top = a.label();
    a.mov_imm(X15, (k / GROUP) as u64);
    a.bind(top);
    a.ldp(q(0), q(1), post(X2, 32));
    a.ldp(q(2), q(3), post(X2, 32));
    for r in 0..4u8 {
        a.and_v(v(4 + r).b16(), v(r).b16(), v(31).b16());
    }
    a.umax_v(v(4).h8(), v(4).h8(), v(5).h8());
    a.umax_v(v(6).h8(), v(6).h8(), v(7).h8());
    a.umax_v(v(4).h8(), v(4).h8(), v(6).h8());
    a.shll(v(5).s4(), v(4).h4());
    a.shll(v(6).s4(), v(4).h8());
    a.fmax(v(5).s4(), v(5).s4(), v(6).s4());
    a.fmaxv(s(5), v(5).s4());
    a.fmax(s(5), s(5), s(29));
    a.fmul(s(6), s(5), s(30));
    a.str(s(6), post(X4, 4));
    a.fdiv(s(7), s(28), s(5));
    a.dup(v(7).s4(), v(7).s(0));
    for r in 0..4u8 {
        a.shll(v(16 + 2 * r).s4(), v(r).h4());
        a.shll(v(17 + 2 * r).s4(), v(r).h8());
    }
    for r in 0..8u8 {
        a.fmul(v(16 + r).s4(), v(16 + r).s4(), v(7).s4());
        a.fcvtns(v(16 + r).s4(), v(16 + r).s4());
    }
    for r in 0..4u8 {
        a.xtn(v(24 + r).h4(), v(16 + 2 * r).s4());
        a.xtn(v(24 + r).h8(), v(17 + 2 * r).s4());
    }
    a.xtn(v(4).b8(), v(24).h8());
    a.xtn(v(4).b16(), v(25).h8());
    a.xtn(v(5).b8(), v(26).h8());
    a.xtn(v(5).b16(), v(27).h8());
    a.stp(q(4), q(5), post(X3, 32));
    a.subs(X15, X15, 1);
    a.bne(top);
    a.bind(exit);
    a.ret();
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}
