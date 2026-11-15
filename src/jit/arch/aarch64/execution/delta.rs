//! The generated GatedDeltaNet nodes on AArch64: L2 norms and gates, the
//! interleave of the gate projections, and the traversals of the conv, the
//! recurrence and the gated norm over rows and heads.
//!
//! The L2 norm keeps the reference order bit for bit (a left
//! to right sum of separate products and additions, a square root and a
//! division, then one product per element). Its vector body puts four heads
//! in the four lanes of a register — a 4×4 transposition brings element `d`
//! of the four heads into one register — so each lane runs the scalar chain
//! of its head, unchanged.

use super::kernels::copy_constant;
use super::math;
use super::{ExecutionError, GeneratedKernel};
use crate::jit::arch::aarch64::util::{add_imm, exec_err, madd_imm, mul_imm, Frame};
pub(crate) use crate::jit::execution::abi::{
    ConvCall, GateCall, InterleaveCall, L2Call, L2GatesCall, NormCall, RecurrenceCall, Shape,
};
use a64::*;
use std::mem::offset_of;

/// `heads` contiguous heads of `dim` f32 at `x0`, eps in `s0`: `x ← x · 1/√(Σx² + eps)`.
/// Uses `x9..x12`, `x16`, `x17`, `v1..v7`, `v16..v27`, and advances `x0`.
fn emit_l2(a: &mut Asm, dim: usize, heads: usize) {
    let head_bytes = dim * 4;
    // Four heads per vector (the transposed reduction, the exact scalar order), the rest one at a
    // time.
    let vector_heads = if dim.is_multiple_of(4) {
        heads / 4 * 4
    } else {
        0
    };
    if vector_heads > 0 {
        // v16 eps, v17 one.
        a.dup(v(16).s4(), v(0).s(0));
        a.fmov_imm(v(17).s4(), 1.0);
        let group = a.label();
        let sum = a.label();
        a.mov(X9, X0);
        a.mov_imm(X10, (vector_heads / 4) as u64);
        a.bind(group);
        // x11 = heads 1..3 of the group, by stride.
        a.movi(v(1).d2(), 0);
        a.mov(X11, X9);
        a.mov_imm(X12, (dim / 4) as u64);
        a.bind(sum);
        a.ldr(q(20), at(X11));
        add_imm(a, X16, X11, head_bytes as i64, X17);
        a.ldr(q(21), at(X16));
        add_imm(a, X16, X16, head_bytes as i64, X17);
        a.ldr(q(22), at(X16));
        add_imm(a, X16, X16, head_bytes as i64, X17);
        a.ldr(q(23), at(X16));
        a.trn1(v(24).s4(), v(20).s4(), v(21).s4());
        a.trn2(v(25).s4(), v(20).s4(), v(21).s4());
        a.trn1(v(26).s4(), v(22).s4(), v(23).s4());
        a.trn2(v(27).s4(), v(22).s4(), v(23).s4());
        a.trn1(v(20).d2(), v(24).d2(), v(26).d2());
        a.trn1(v(21).d2(), v(25).d2(), v(27).d2());
        a.trn2(v(22).d2(), v(24).d2(), v(26).d2());
        a.trn2(v(23).d2(), v(25).d2(), v(27).d2());
        for e in 20..24 {
            a.fmul(v(e).s4(), v(e).s4(), v(e).s4());
            a.fadd(v(1).s4(), v(1).s4(), v(e).s4());
        }
        a.add(X11, X11, 16);
        a.subs(X12, X12, 1);
        a.bne(sum);
        a.fadd(v(1).s4(), v(1).s4(), v(16).s4());
        a.fsqrt(v(1).s4(), v(1).s4());
        a.fdiv(v(1).s4(), v(17).s4(), v(1).s4());
        for h in 0..4u8 {
            a.mov(X11, X9);
            if h > 0 {
                add_imm(a, X11, X9, (usize::from(h) * head_bytes) as i64, X17);
            }
            a.mov_imm(X12, (dim / 4) as u64);
            let apply_h = a.label();
            a.bind(apply_h);
            a.ldr(q(20), at(X11));
            a.fmul(v(20).s4(), v(20).s4(), v(1).s(h));
            a.str(q(20), post(X11, 16));
            a.subs(X12, X12, 1);
            a.bne(apply_h);
        }
        add_imm(a, X9, X9, (4 * head_bytes) as i64, X17);
        a.subs(X10, X10, 1);
        a.bne(group);
        add_imm(a, X0, X0, (vector_heads * head_bytes) as i64, X17);
    }
    for _ in vector_heads..heads {
        // One head, the scalar chain.
        let sum = a.label();
        let apply = a.label();
        a.fmov_from_gp(s(1), WZR);
        a.mov(X11, X0);
        a.mov_imm(X12, dim as u64);
        a.bind(sum);
        a.ldr(s(2), post(X11, 4));
        a.fmul(s(2), s(2), s(2));
        a.fadd(s(1), s(1), s(2));
        a.subs(X12, X12, 1);
        a.bne(sum);
        a.fadd(s(1), s(1), s(0));
        a.fsqrt(s(1), s(1));
        a.fmov_imm(s(2), 1.0);
        a.fdiv(s(2), s(2), s(1));
        a.mov(X11, X0);
        a.mov_imm(X12, dim as u64);
        a.bind(apply);
        a.ldr(s(3), at(X11));
        a.fmul(s(3), s(3), s(2));
        a.str(s(3), post(X11, 4));
        a.subs(X12, X12, 1);
        a.bne(apply);
        add_imm(a, X0, X0, head_bytes as i64, X17);
    }
}

/// The gate of one `(row, head)` task: `beta = 1/(1 + exp(−b))`, `decay =
/// exp(−exp(a_log)·dt)` with `dt = x` above twenty, `log(1 + exp(x))` below,
/// `x = a + bias`. `(call: *const GateCall, task)` in `x19`, `x20`; the head
/// in `x21`, the `a` index in `x22`; `[sp]` holds `dt`.
fn emit_gate_body(a: &mut Asm, nv: usize, exp_label: Label, log_label: Label) {
    let field = |f: usize| mem(X19, f as i64);
    let dt = a.label();
    a.mov_imm(X9, nv as u64);
    a.udiv(X10, X20, X9);
    a.msub(X21, X10, X9, X20);
    a.lsl(X22, X20, 1);
    a.sub(X22, X22, X21);
    a.ldr(X9, field(offset_of!(GateCall, ab)));
    add_imm(a, X9, X9, (nv * 4) as i64, X10);
    a.ldr(s(0), idx_scaled(X9, X22));
    a.fneg(s(0), s(0));
    a.ldr(X0, field(offset_of!(GateCall, tables)));
    a.bl(exp_label);
    a.fmov_imm(s(1), 1.0);
    a.fadd(s(0), s(0), s(1));
    a.fdiv(s(1), s(1), s(0));
    a.ldr(X9, field(offset_of!(GateCall, beta)));
    a.str(s(1), idx_scaled(X9, X20));
    a.ldr(X9, field(offset_of!(GateCall, ab)));
    a.ldr(s(0), idx_scaled(X9, X22));
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
}

/// `(context, row)` converts two `[rows, nv]` projection outputs into the
/// fixed `[rows, a|b]` gate layout before their scratch areas are reused.
pub(crate) fn interleave(nv: usize) -> Result<GeneratedKernel, ExecutionError> {
    if nv == 0 || nv > i32::MAX as usize / 8 {
        return Err(ExecutionError("invalid DeltaNet interleave width".into()));
    }
    let mut a = Asm::new();
    let field = |f: usize| mem(X0, f as i64);
    mul_imm(&mut a, X9, X1, (nv * 4) as u64, X10);
    a.ldr(X2, field(offset_of!(InterleaveCall, a)));
    a.add(X2, X2, X9);
    a.ldr(X3, field(offset_of!(InterleaveCall, b)));
    a.add(X3, X3, X9);
    mul_imm(&mut a, X9, X1, (nv * 8) as u64, X10);
    a.ldr(X4, field(offset_of!(InterleaveCall, output)));
    a.add(X4, X4, X9);
    // a and b land side by side: two contiguous runs of nv floats.
    copy_constant(&mut a, X4, 0, X2, 0, nv * 4);
    copy_constant(&mut a, X4, nv * 4, X3, 0, nv * 4);
    a.ret();
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// `(context, row·nv + head)`: the gate `head` of the row, and, on the task of
/// head zero, the L2 of the row's `2·nk` contiguous Q then K heads first — one
/// node instead of two. Rows and heads are independent, so the task order is
/// free; the order inside a head and inside a gate is the reference's.
pub(crate) fn l2_gates(shape: Shape, fma: bool) -> Result<GeneratedKernel, ExecutionError> {
    shape.check()?;
    if shape.key() != shape.nk * shape.hkd {
        return Err(ExecutionError(
            "the L2 expects contiguous Q and K heads".into(),
        ));
    }
    let mut a = Asm::new();
    let exp_label = a.label();
    let log_label = a.label();
    let gate = a.label();
    let frame = Frame::enter(&mut a, true, &[X19, X20, X21, X22], false, 16);
    let l2 = offset_of!(L2GatesCall, l2);
    a.mov(X19, X0);
    a.mov(X20, X1);
    a.mov_imm(X9, shape.nv as u64);
    a.udiv(X10, X20, X9);
    a.msub(X11, X10, X9, X20);
    a.cbnz(X11, gate);
    a.ldr(X0, mem(X19, (l2 + offset_of!(L2Call, qkv)) as i64));
    madd_imm(&mut a, X0, X10, (shape.conv() * 4) as u64, X0, X9);
    a.ldr(s(0), mem(X19, (l2 + offset_of!(L2Call, eps)) as i64));
    emit_l2(&mut a, shape.hkd, 2 * shape.nk);
    a.bind(gate);
    add_imm(&mut a, X19, X19, offset_of!(L2GatesCall, gates) as i64, X9);
    emit_gate_body(&mut a, shape.nv, exp_label, log_label);
    frame.leave(&mut a);
    a.ret();
    a.bind(exp_label);
    math::exp_body(&mut a, fma);
    a.bind(log_label);
    math::log_body(&mut a, fma);
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// `(context, V-head)` traverses all positions causally and invokes the
/// recurrence kernel. Separate heads own disjoint state/output slices.
pub(crate) fn recurrence(
    shape: Shape,
    kernel: Option<&crate::jit::recurrence::CompiledRecurrence>,
) -> Result<GeneratedKernel, ExecutionError> {
    recurrence_with(shape, kernel, crate::jit::knobs::knobs().deltanet_chunked)
}

/// [`recurrence`] with the choice of the chunks given (`Knobs::deltanet_chunked`).
///
/// By chunks, the chunks sit on multiples of sixteen absolute positions, so
/// that the bits do not depend on where a call starts (the block attention's
/// rule). A prefill pass that starts inside a chunk finishes it on the
/// head's [`Carry`] when the carry stands for the pass's first position (the
/// state then still at the chunk's first position), on the sequential kernel
/// otherwise (after a decode step); a pass that ends inside a chunk runs it
/// on the carry, zero rows after its own, for the output rows only, and
/// leaves the state at the chunk's first position. A decode step after such
/// a pass closes the chunk on the state first (the carry's rows, the zero
/// rows adding nothing), then runs the sequential kernel.
///
/// [`Carry`]: crate::jit::arch::aarch64::deltanet_chunk::Carry
pub(crate) fn recurrence_with(
    shape: Shape,
    kernel: Option<&crate::jit::recurrence::CompiledRecurrence>,
    chunked: bool,
) -> Result<GeneratedKernel, ExecutionError> {
    use crate::jit::arch::aarch64::deltanet_chunk::{self as dc, Carry, ChunkSpec, CHUNK};
    shape.check()?;
    let wanted = crate::jit::recurrence::spec_of_knobs(shape.hvd);
    // `q_hoist` scales `q` into 2 KiB of the kernel's stack: 512 values at most.
    if wanted.q_hoist && shape.hkd > 512 {
        return Err(ExecutionError(format!(
            "--deltanet-q-hoist: key heads of {} values, 512 at most",
            shape.hkd
        )));
    }
    // Bytes of one value of the state (bf16 or f32); the output and `v` stay f32.
    let elem = if wanted.state16 { 2 } else { 4 };
    let split = crate::jit::recurrence::vd_split(shape.hvd);
    let width = if split > 1 {
        shape.hvd / split
    } else {
        shape.hvd
    };
    let bytes = match kernel.filter(|k| k.spec() == wanted) {
        Some(k) => k.lowering_bytes(),
        None => crate::jit::emit_recurrence::emit_recurrence(&wanted).map_err(exec_err)?,
    };
    // Chunks of sixteen positions on the matrix engine (`Knobs::deltanet_chunked`),
    // when the head is whole and the machine has one.
    let chunk = ChunkSpec {
        hkd: shape.hkd,
        hvd: shape.hvd,
        row_bytes: shape.conv() * 4,
        out_bytes: shape.value() * 4,
        gate_bytes: shape.nv * 4,
        state: true,
        state16: wanted.state16,
    };
    let carry = Carry::of(shape.hkd, shape.hvd);
    let chunk_bytes = if chunked && split == 1 && chunk.check().is_ok() {
        Some((
            dc::emit(&chunk).map_err(exec_err)?,
            dc::emit(&ChunkSpec {
                state16: wanted.state16,
                ..carry.spec(shape.hkd, shape.hvd, true)
            })
            .map_err(exec_err)?,
            dc::emit(&ChunkSpec {
                state16: wanted.state16,
                ..carry.spec(shape.hkd, shape.hvd, false)
            })
            .map_err(exec_err)?,
        ))
    } else {
        None
    };
    let mut a = Asm::new();
    let entry = a.label();
    let chunk_entry = a.label();
    let carry_entry = a.label();
    let carry_out_entry = a.label();
    let again = a.label();
    let exit = a.label();
    let field = |f: usize| mem(X19, f as i64);
    // The output rows of a chunk run on the carry: sixteen rows of the head.
    let out_scratch = if chunk_bytes.is_some() {
        (CHUNK * shape.hvd * 4).next_multiple_of(4096)
    } else {
        0
    };
    // x19 call, x20 head, x21 q/k head offset, x22 state, x23 rows, x24 row.
    // With a split: task = head · split + slice; x25 = the slice's byte offset
    // into the value columns of the state, the output and v. By chunks: x26
    // the head's carry, x27 the pass's first position, x28 its rows in the
    // chunk it starts inside.
    let saved: &[R] = if chunk_bytes.is_some() {
        &[X19, X20, X21, X22, X23, X24, X25, X26, X27, X28]
    } else {
        &[X19, X20, X21, X22, X23, X24, X25, X26]
    };
    let frame = Frame::enter(&mut a, true, saved, false, out_scratch);
    a.mov(X19, X0);
    if split > 1 {
        a.mov_imm(X9, split as u64);
        a.udiv(X20, X1, X9);
        a.msub(X25, X20, X9, X1);
        mul_imm(&mut a, X25, X25, (width * 4) as u64, X9);
    } else {
        a.mov(X20, X1);
        a.mov_imm(X25, 0);
    }
    a.mov_imm(X9, (shape.nv / shape.nk) as u64);
    a.udiv(X21, X20, X9);
    mul_imm(&mut a, X21, X21, (shape.hkd * 4) as u64, X9);
    a.ldr(X22, field(offset_of!(RecurrenceCall, state)));
    madd_imm(
        &mut a,
        X22,
        X20,
        (shape.hkd * shape.hvd * elem) as u64,
        X22,
        X9,
    );
    if elem == 2 {
        a.lsr(X9, X25, 1);
        a.add(X22, X22, X9);
    } else {
        a.add(X22, X22, X25);
    }
    a.ldr(X23, field(offset_of!(RecurrenceCall, rows)));
    a.ldr(X23, at(X23));
    a.mov_imm(X24, 0);
    a.cbz(X23, exit);
    if chunk_bytes.is_some() {
        let (prefill, chunks, rest, fallback, first_whole, marked) = (
            a.label(),
            a.label(),
            a.label(),
            a.label(),
            a.label(),
            a.label(),
        );
        // x26 = the head's carry, after the layer's `nv` states.
        a.ldr(X26, field(offset_of!(RecurrenceCall, state)));
        add_imm(
            &mut a,
            X26,
            X26,
            (shape.nv * shape.hkd * shape.hvd * elem) as i64,
            X9,
        );
        madd_imm(&mut a, X26, X20, carry.bytes as u64, X26, X9);
        a.ldr(X27, field(offset_of!(RecurrenceCall, position)));
        a.ldr(X27, at(X27));
        a.ldr(X9, field(offset_of!(RecurrenceCall, prefill)));
        a.cbnz(X9, prefill);
        // Decode: a chunk left open on the carry is closed on the state first.
        let decode_rows = a.label();
        a.and(X9, X27, 15u64);
        a.cbz(X9, decode_rows);
        add_imm(&mut a, X10, X26, carry.mark as i64, X11);
        a.ldr(X10, at(X10));
        a.cmp(X10, X27);
        a.bne(decode_rows);
        call_carry(&mut a, &carry, carry_entry);
        a.mov_imm(X9, u64::MAX);
        add_imm(&mut a, X10, X26, carry.mark as i64, X11);
        a.str(X9, at(X10));
        a.bind(decode_rows);
        sequential_row(&mut a, &shape, entry);
        a.add(X24, X24, 1);
        a.cmp(X24, X23);
        a.blo(decode_rows);
        a.b(exit);

        a.bind(prefill);
        // x9 = j, the pass's place in its first chunk; x28 = min(16 − j, rows).
        a.and(X9, X27, 15u64);
        a.cbz(X9, chunks);
        a.mov_imm(X10, CHUNK as u64);
        a.sub(X28, X10, X9);
        a.cmp(X28, X23);
        a.csel(X28, X23, X28, Cond::Hi);
        add_imm(&mut a, X10, X26, carry.mark as i64, X11);
        a.ldr(X10, at(X10));
        a.cmp(X10, X27);
        a.bne(fallback);
        // The carry stands for this position: the pass's first rows join it.
        a.mov(X11, XZR);
        a.mov(X12, X9);
        a.mov(X13, X28);
        carry_in(&mut a, &shape, &carry);
        a.and(X9, X27, 15u64);
        a.add(X12, X9, X28);
        a.cmp(X12, CHUNK as u64);
        a.bhs(first_whole);
        // The pass ends inside the chunk: zero rows after it, the output rows only.
        carry_zero(&mut a, &carry);
        call_carry(&mut a, &carry, carry_out_entry);
        a.and(X12, X27, 15u64);
        a.mov(X11, XZR);
        a.mov(X13, X28);
        carry_out(&mut a, &shape);
        a.b(marked);
        a.bind(first_whole);
        call_carry(&mut a, &carry, carry_entry);
        a.and(X12, X27, 15u64);
        a.mov(X11, XZR);
        a.mov(X13, X28);
        carry_out(&mut a, &shape);
        a.mov(X24, X28);
        a.b(chunks);
        // No carry for this position (a decode step came between): the
        // sequential kernel up to the chunk's end, the carry left as it is
        // when the pass ends before it.
        a.bind(fallback);
        let fallback_rows = a.label();
        a.bind(fallback_rows);
        sequential_row(&mut a, &shape, entry);
        a.add(X24, X24, 1);
        a.cmp(X24, X28);
        a.blo(fallback_rows);
        a.and(X9, X27, 15u64);
        a.add(X12, X9, X28);
        a.cmp(X12, CHUNK as u64);
        a.blo(exit);
        // Whole chunks while row + 16 ≤ rows.
        a.bind(chunks);
        a.add(X9, X24, 16);
        a.cmp(X9, X23);
        a.bhi(rest);
        a.mov(X0, X22);
        a.ldr(X1, field(offset_of!(RecurrenceCall, output)));
        madd_imm(&mut a, X1, X24, (shape.value() * 4) as u64, X1, X9);
        madd_imm(&mut a, X1, X20, (shape.hvd * 4) as u64, X1, X9);
        a.ldr(X10, field(offset_of!(RecurrenceCall, qkv)));
        madd_imm(&mut a, X10, X24, (shape.conv() * 4) as u64, X10, X9);
        a.add(X2, X10, X21);
        add_imm(&mut a, X3, X2, (shape.key() * 4) as i64, X9);
        add_imm(&mut a, X4, X10, (shape.key() * 8) as i64, X9);
        madd_imm(&mut a, X4, X20, (shape.hvd * 4) as u64, X4, X9);
        mul_imm(&mut a, X10, X24, shape.nv as u64, X9);
        a.add(X10, X10, X20);
        a.ldr(X5, field(offset_of!(RecurrenceCall, beta)));
        a.add(X5, X5, X10.lsl(2));
        a.ldr(X6, field(offset_of!(RecurrenceCall, decay)));
        a.add(X6, X6, X10.lsl(2));
        a.bl(chunk_entry);
        a.add(X24, X24, 16);
        a.b(chunks);
        // The pass ends inside a chunk: its rows on the carry, zero rows after
        // them, the output rows only; the state stays at the chunk's start.
        a.bind(rest);
        a.cmp(X24, X23);
        a.bhs(marked);
        a.mov(X11, X24);
        a.mov(X12, XZR);
        a.sub(X13, X23, X24);
        carry_in(&mut a, &shape, &carry);
        a.sub(X12, X23, X24);
        carry_zero(&mut a, &carry);
        call_carry(&mut a, &carry, carry_out_entry);
        a.mov(X11, X24);
        a.mov(X12, XZR);
        a.sub(X13, X23, X24);
        carry_out(&mut a, &shape);
        // The carry stands for the pass's end.
        a.bind(marked);
        a.add(X9, X27, X23);
        add_imm(&mut a, X10, X26, carry.mark as i64, X11);
        a.str(X9, at(X10));
        a.b(exit);
    }
    a.bind(again);
    sequential_row(&mut a, &shape, entry);
    a.add(X24, X24, 1);
    a.cmp(X24, X23);
    a.blo(again);
    a.bind(exit);
    frame.leave(&mut a);
    a.ret();
    a.align(16);
    a.bind(entry);
    a.bytes(&bytes);
    if let Some((direct, on_carry, on_carry_out)) = &chunk_bytes {
        for (label, blob) in [
            (chunk_entry, direct),
            (carry_entry, on_carry),
            (carry_out_entry, on_carry_out),
        ] {
            a.align(16);
            a.bind(label);
            a.bytes(blob);
        }
    }
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// Row `x24` on the sequential kernel at `entry`.
fn sequential_row(a: &mut Asm, shape: &Shape, entry: Label) {
    let field = |f: usize| mem(X19, f as i64);
    a.mov(X0, X22);
    a.ldr(X1, field(offset_of!(RecurrenceCall, output)));
    madd_imm(a, X1, X24, (shape.value() * 4) as u64, X1, X9);
    madd_imm(a, X1, X20, (shape.hvd * 4) as u64, X1, X9);
    a.add(X1, X1, X25);
    a.ldr(X10, field(offset_of!(RecurrenceCall, qkv)));
    madd_imm(a, X10, X24, (shape.conv() * 4) as u64, X10, X9);
    a.add(X2, X10, X21);
    add_imm(a, X3, X2, (shape.key() * 4) as i64, X9);
    add_imm(a, X4, X10, (shape.key() * 8) as i64, X9);
    madd_imm(a, X4, X20, (shape.hvd * 4) as u64, X4, X9);
    a.add(X4, X4, X25);
    a.mov_imm(X5, shape.hkd as u64);
    mul_imm(a, X10, X24, shape.nv as u64, X9);
    a.add(X10, X10, X20);
    a.ldr(X9, field(offset_of!(RecurrenceCall, beta)));
    a.ldr(s(0), idx_scaled(X9, X10));
    a.ldr(X9, field(offset_of!(RecurrenceCall, decay)));
    a.ldr(s(1), idx_scaled(X9, X10));
    a.mov_imm(W9, u64::from((1.0 / (shape.hkd as f32).sqrt()).to_bits()));
    a.fmov_from_gp(s(2), W9);
    a.bl(entry);
}

/// The chunk kernel at `label` on the head's carry (`x26`), the state `x22`,
/// the output rows into the scratch at `sp`.
fn call_carry(a: &mut Asm, carry: &crate::jit::arch::aarch64::deltanet_chunk::Carry, label: Label) {
    a.mov(X0, X22);
    a.add(X1, SP, 0);
    a.mov(X2, X26);
    add_imm(a, X3, X26, carry.k as i64, X9);
    add_imm(a, X4, X26, carry.v as i64, X9);
    add_imm(a, X5, X26, carry.beta as i64, X9);
    add_imm(a, X6, X26, carry.decay as i64, X9);
    a.bl(label);
}

/// `x13` rows of the pass from row `x11` into the carry (`x26`) from its row
/// `x12`: `q`, `k`, `v` of the head, the gate and the decay. Uses `x7..x17`,
/// `v16`, `v17`.
fn carry_in(a: &mut Asm, shape: &Shape, carry: &crate::jit::arch::aarch64::deltanet_chunk::Carry) {
    let top = a.label();
    a.bind(top);
    // x7 = the pass's `qkv` row, x8 = the carry's row.
    a.ldr(X7, mem(X19, offset_of!(RecurrenceCall, qkv) as i64));
    madd_imm(a, X7, X11, (shape.conv() * 4) as u64, X7, X14);
    madd_imm(a, X8, X12, carry.row_bytes as u64, X26, X14);
    a.add(X9, X7, X21);
    copy_constant(a, X8, 0, X9, 0, shape.hkd * 4);
    copy_constant(a, X8, carry.k, X9, shape.key() * 4, shape.hkd * 4);
    madd_imm(a, X9, X20, (shape.hvd * 4) as u64, X7, X14);
    copy_constant(a, X8, carry.v, X9, shape.key() * 8, shape.hvd * 4);
    mul_imm(a, X14, X11, shape.nv as u64, X9);
    a.add(X14, X14, X20);
    a.add(X10, X26, X12.lsl(2));
    a.ldr(X9, mem(X19, offset_of!(RecurrenceCall, beta) as i64));
    a.ldr(s(16), idx_scaled(X9, X14));
    add_imm(a, X9, X10, carry.beta as i64, X9);
    a.str(s(16), at(X9));
    a.ldr(X9, mem(X19, offset_of!(RecurrenceCall, decay) as i64));
    a.ldr(s(16), idx_scaled(X9, X14));
    add_imm(a, X9, X10, carry.decay as i64, X9);
    a.str(s(16), at(X9));
    a.add(X11, X11, 1);
    a.add(X12, X12, 1);
    a.subs(X13, X13, 1);
    a.bne(top);
}

/// The carry's rows from `x12` to the chunk's end zeroed: `q`, `k`, `v` and
/// the gate zero, the decay one. Uses `x9`, `x10`, `x12`, `v16`, `v17`.
fn carry_zero(a: &mut Asm, carry: &crate::jit::arch::aarch64::deltanet_chunk::Carry) {
    use crate::jit::arch::aarch64::deltanet_chunk::CHUNK;
    let (top, done) = (a.label(), a.label());
    a.movi(v(16).d2(), 0);
    a.fmov_imm(s(17), 1.0);
    a.bind(top);
    a.cmp(X12, CHUNK as u64);
    a.bhs(done);
    madd_imm(a, X10, X12, carry.row_bytes as u64, X26, X9);
    debug_assert_eq!(carry.row_bytes % 32, 0);
    for _ in 0..carry.row_bytes / 32 {
        a.stp(q(16), q(16), post(X10, 32));
    }
    a.add(X10, X26, X12.lsl(2));
    add_imm(a, X9, X10, carry.beta as i64, X9);
    a.str(s(16), at(X9));
    add_imm(a, X9, X10, carry.decay as i64, X9);
    a.str(s(17), at(X9));
    a.add(X12, X12, 1);
    a.b(top);
    a.bind(done);
}

/// `x13` output rows of the scratch (`sp`) from its row `x12` into the
/// output rows of the pass from row `x11`. Uses `x9..x17`, `v16`, `v17`.
fn carry_out(a: &mut Asm, shape: &Shape) {
    let top = a.label();
    a.bind(top);
    a.ldr(X9, mem(X19, offset_of!(RecurrenceCall, output) as i64));
    madd_imm(a, X9, X11, (shape.value() * 4) as u64, X9, X14);
    madd_imm(a, X9, X20, (shape.hvd * 4) as u64, X9, X14);
    a.add(X10, SP, 0);
    madd_imm(a, X10, X12, (shape.hvd * 4) as u64, X10, X14);
    copy_constant(a, X9, 0, X10, 0, shape.hvd * 4);
    a.add(X11, X11, 1);
    a.add(X12, X12, 1);
    a.subs(X13, X13, 1);
    a.bne(top);
}

/// `(context, channel-block)` traverses positions in order, 16 channels per
/// independent task, through the conv kernel.
pub(crate) fn convolution(
    shape: Shape,
    kernel: Option<&crate::jit::conv::CompiledConv>,
    _fma: bool,
) -> Result<GeneratedKernel, ExecutionError> {
    shape.check()?;
    if !shape.conv().is_multiple_of(16) {
        return Err(ExecutionError(format!(
            "the AArch64 conv serves whole blocks of 16 channels, not {}",
            shape.conv()
        )));
    }
    let bytes = match kernel.filter(|k| k.k() == shape.ck) {
        Some(k) => k.lowering_bytes(),
        None => crate::jit::emit_conv::emit_conv(&crate::jit::emit_conv::ConvSpec { k: shape.ck })
            .map_err(exec_err)?,
    };
    let mut a = Asm::new();
    let entry = a.label();
    let again = a.label();
    let exit = a.label();
    let field = |f: usize| mem(X19, f as i64);
    // x19 call, x20 data cursor, x21 weights, x22 state, x23 rows left.
    let frame = Frame::enter(&mut a, true, &[X19, X20, X21, X22, X23], false, 0);
    a.mov(X19, X0);
    a.ldr(X23, field(offset_of!(ConvCall, rows)));
    a.ldr(X23, at(X23));
    a.ldr(X20, field(offset_of!(ConvCall, data)));
    madd_imm(&mut a, X20, X1, 64, X20, X9);
    a.ldr(X22, field(offset_of!(ConvCall, state)));
    madd_imm(&mut a, X22, X1, (64 * shape.ck) as u64, X22, X9);
    a.ldr(X21, field(offset_of!(ConvCall, weights)));
    madd_imm(&mut a, X21, X1, (32 * shape.ck) as u64, X21, X9);
    a.cbz(X23, exit);
    a.bind(again);
    a.mov(X0, X20);
    a.mov(X1, X21);
    a.mov(X2, X22);
    a.mov_imm(X3, 1);
    a.bl(entry);
    add_imm(&mut a, X20, X20, (shape.conv() * 4) as i64, X9);
    a.subs(X23, X23, 1);
    a.bne(again);
    a.bind(exit);
    frame.leave(&mut a);
    a.ret();
    a.align(16);
    a.bind(entry);
    a.bytes(&bytes);
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// `(context, row)` applies the gated norm kernel to one V-head row and writes
/// the bf16 projection input.
pub(crate) fn normed_gate(
    dim: usize,
    kernel: Option<&crate::jit::gated_norm::CompiledGatedNorm>,
    _fma: bool,
) -> Result<GeneratedKernel, ExecutionError> {
    if dim == 0 || dim > i32::MAX as usize / 4 {
        return Err(ExecutionError("invalid DeltaNet gated norm width".into()));
    }
    let bytes = match kernel.filter(|k| k.dim() == dim) {
        Some(k) => k.lowering_bytes(),
        None => crate::jit::emit_gated_norm::emit_gated_norm(
            &crate::jit::emit_gated_norm::GatedNormSpec { dim },
        )
        .map_err(exec_err)?,
    };
    let mut a = Asm::new();
    let entry = a.label();
    let field = |f: usize| mem(X8, f as i64);
    let frame = Frame::enter(&mut a, true, &[], false, 0);
    a.mov(X8, X0);
    mul_imm(&mut a, X9, X1, (dim * 4) as u64, X10);
    mul_imm(&mut a, X11, X1, (dim * 2) as u64, X10);
    a.ldr(X0, field(offset_of!(NormCall, output)));
    a.add(X0, X0, X9);
    a.ldr(X1, field(offset_of!(NormCall, gate)));
    a.add(X1, X1, X9);
    a.ldr(X2, field(offset_of!(NormCall, weights)));
    a.ldr(X3, field(offset_of!(NormCall, output_bf16)));
    a.add(X3, X3, X11);
    a.mov_imm(X4, 1);
    a.ldr(s(0), field(offset_of!(NormCall, eps)));
    a.bl(entry);
    frame.leave(&mut a);
    a.ret();
    a.align(16);
    a.bind(entry);
    a.bytes(&bytes);
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}
