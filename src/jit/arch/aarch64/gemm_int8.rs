//! The GEMM micro-kernel for **int8 weights and activations**, NEON with FEAT_DotProd (`sdot`).
//!
//! The layouts are those of `jit/int8.rs`: `W` in R16 blocks of groups of 32 values of `k` (512
//! bytes of int8, then the sixteen f32 scales of the rows), `A` quantized per row (`K` int8, then
//! `K/32` f32 scales) by the node that precedes the projections. The ABI is that of `gemm.rs`: x0 =
//! row 0 of `A`, x1 = the block of `W`, x2 = `C`, x3.. the other rows; x6/x7 point at the scales of
//! rows 0/1 of `A`, x15 counts the groups.
//!
//! Per group of 32 values of `k` and per row of the tile: eight steps of four `k`; per step the
//! four quads of `W` (four rows each) meet the four bytes of `A` with one `sdot` by element per
//! quad into int32 accumulators; at the end of the group the accumulators are converted (`scvtf`),
//! multiplied by `scale_w · scale_a` and added into the f32 accumulators, which the epilogue of
//! `gemm.rs` stores.
//!
//! ## Registers
//!
//! int32 accumulators `v0..v3` (row 0), `v4..v7` (row 1); f32 accumulators `v8..v11`, `v12..v15`;
//! `A` of the group `v16`/`v17`, `v18`/`v19`; `W` two sets of four, `v20..v23` and `v24..v27`;
//! `v28..v31` temporaries.

use crate::core::error::{HerbertError, Result};
use crate::jit::arch::aarch64::gemm_f16::{row, store_tile};
use crate::jit::arch::aarch64::util::{add_imm, backend_err, Frame};
use crate::jit::emit::{AFormat, MicroKernelSpec};
use crate::jit::int8::{a_row_bytes, GROUP};
use crate::jit::ir::{Decline, ROWS_INTERLEAVED};
use a64::*;

/// Rows of a tile: the registers hold two.
pub const MAX_ROWS: usize = 2;

/// What this kernel emits: quantized `A`, sixteen columns, one or two rows, `K` a multiple of 32.
pub fn check(spec: &MicroKernelSpec) -> std::result::Result<(), Decline> {
    let most = if spec.opts.neon.w_q4 {
        super::gemm_q4::max_rows()
    } else {
        MAX_ROWS
    };
    if spec.nr != ROWS_INTERLEAVED || spec.mr == 0 || spec.mr > most {
        return Err(Decline::TileTooWide);
    }
    let k = 2 * spec.k_pairs - usize::from(spec.k_odd);
    if spec.a_fmt != AFormat::Bf16
        || spec.opts.neon.a_packed
        || spec.opts.matrix
        || spec.k_dynamic
        || spec.k_odd
    {
        return Err(Decline::OutOfRegisters);
    }
    if k == 0 || !k.is_multiple_of(GROUP) || spec.lda_bytes != a_row_bytes(k) {
        return Err(Decline::OddK);
    }
    Ok(())
}

/// Generates the kernel of `spec`.
pub fn emit(spec: &MicroKernelSpec) -> Result<Vec<u8>> {
    check(spec).map_err(|d| HerbertError::Backend(format!("backend_jit int8: {d}")))?;
    let mr = spec.mr;
    let k = 2 * spec.k_pairs;
    let groups = k / GROUP;
    let acc = |i: usize, c: usize| (4 * i + c) as u8;
    let facc = |i: usize, c: usize| (8 + 4 * i + c) as u8;
    let areg = |i: usize, h: usize| (16 + 2 * i + h) as u8;
    let wset = |s: usize| 20 + 4 * (s % 2) as u8;
    let mut a = Asm::new();
    let frame = Frame::enter(&mut a, false, &[], true, 0);
    for i in 1..mr {
        add_imm(&mut a, row(i), row(i - 1), spec.lda_bytes as i64, X16);
    }
    // The scales of `A` follow the `K` bytes of the row.
    for i in 0..mr {
        add_imm(&mut a, x((6 + i) as u8), row(i), k as i64, X16);
    }
    for i in 0..mr {
        for c in 0..4 {
            a.movi(v(facc(i, c)).d2(), 0);
        }
    }
    let top = a.label();
    a.mov_imm(X15, groups as u64);
    a.bind(top);
    for i in 0..mr {
        a.ldp(q(areg(i, 0)), q(areg(i, 1)), post(row(i), 32));
        for c in 0..4 {
            a.movi(v(acc(i, c)).d2(), 0);
        }
    }
    for s in 0..8usize {
        let w = wset(s);
        a.ldp(q(w), q(w + 1), post(X1, 32));
        a.ldp(q(w + 2), q(w + 3), post(X1, 32));
        for i in 0..mr {
            for c in 0..4u8 {
                a.sdot(
                    v(acc(i, c as usize)).s4(),
                    v(w + c).b16(),
                    v(areg(i, s / 4)).s((s % 4) as u8),
                );
            }
        }
    }
    // The sixteen f32 scales of the rows of `W`, then per row: · scale of `A`, int32 to f32,
    // accumulate.
    let w = wset(0);
    a.ldp(q(w), q(w + 1), post(X1, 32));
    a.ldp(q(w + 2), q(w + 3), post(X1, 32));
    for i in 0..mr {
        let sa = areg(i, 0);
        a.ldr(s(sa), post(x((6 + i) as u8), 4));
        let t = 24 + 4 * i as u8;
        for c in 0..4u8 {
            a.fmul(v(t + c).s4(), v(w + c).s4(), v(sa).s(0));
        }
        for c in 0..4 {
            a.scvtf(v(acc(i, c)).s4(), v(acc(i, c)).s4());
        }
        for c in 0..4u8 {
            a.fmla(
                v(facc(i, c as usize)).s4(),
                v(acc(i, c as usize)).s4(),
                v(t + c).s4(),
            );
        }
    }
    a.subs(X15, X15, 1);
    a.bne(top);
    store_tile(
        &mut a,
        spec,
        &|i| [facc(i, 0), facc(i, 1), facc(i, 2), facc(i, 3)],
        20,
    );
    frame.leave(&mut a);
    a.ret();
    a.finish()
        .map(|c| c.bytes)
        .map_err(backend_err(format_args!("gemm int8 {}x16 (k={k})", mr)))
}
