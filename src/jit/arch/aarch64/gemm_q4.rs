//! The GEMM micro-kernel for **4-bit weights and int8 activations**, NEON with FEAT_DotProd
//! (`sdot`).
//!
//! As `gemm_int8.rs` with the weight layout of the Q4 form of `jit/int8.rs`: per group of 32 values
//! of `k` and per block of sixteen rows, four steps of eight `k`. A step is four quads of sixteen
//! bytes; in a byte the low nibble is the code of `k = 8t+j`, the high nibble that of `k = 8t+4+j`.
//! `shl 4` of a quad leaves `16·low` as signed bytes, `and 0xf0` leaves `16·high`: two `sdot` by
//! element per quad against the four bytes of `A` of those `k` (the stored scale already holds the
//! `1/16`). Everything else is the int8 kernel.
//!
//! ## Registers
//!
//! int32 accumulators `v0..v3`, `v4..v7`; f32 accumulators `v8..v11`, `v12..v15`; `A` of the group
//! `v16`/`v17`, `v18`/`v19`; `W` `v20..v23`; the shifted and masked copies `v24..v27`; the `0xf0`
//! mask `v28`.
//!
//! ## Up to four rows ([`ROWS4`], the default)
//!
//! The tile of `mr ≤ 4` rows and sixteen columns in two passes of eight columns over the whole of
//! `K` (the rows of `A` read twice, from the caches): the unpacked weights of a step serve four
//! rows instead of two. Per pass, int32 accumulators `v0..v7` (row `i`: `v2i`, `v2i+1`), f32
//! accumulators `v8..v15`, `A` of the group `v16..v23`, the two quads of `W` `v24`/`v25`, their
//! shifted or masked copies `v26`/`v27`, the mask `v28`. The same integer sums and the same f32
//! operations as the kernel of two rows, in the same order: the same bits. `HERBERT_Q4_ROWS=2`
//! keeps the kernel of two rows (the control).

use crate::core::error::{HerbertError, Result};
use crate::jit::arch::aarch64::gemm_f16::{row, store_tile};
use crate::jit::arch::aarch64::util::{add_imm, backend_err, Frame};
use crate::jit::emit::MicroKernelSpec;
use crate::jit::int8::GROUP;
use crate::jit::ir::Decline;
use a64::*;

pub use super::gemm_int8::{check, MAX_ROWS};

/// Rows of a tile of the four-row kernel.
pub const ROWS4: usize = 4;

/// Rows of a Q4 tile: four, or two under `HERBERT_Q4_ROWS=2`.
pub fn max_rows() -> usize {
    match std::env::var("HERBERT_Q4_ROWS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        Some(2) => MAX_ROWS,
        _ => ROWS4,
    }
}

/// Generates the kernel of `spec`.
pub fn emit(spec: &MicroKernelSpec) -> Result<Vec<u8>> {
    check(spec).map_err(|d: Decline| HerbertError::Backend(format!("backend_jit q4: {d}")))?;
    if max_rows() == ROWS4 {
        return emit4(spec);
    }
    emit2(spec)
}

/// The kernel of two rows (module documentation).
pub(crate) fn emit2(spec: &MicroKernelSpec) -> Result<Vec<u8>> {
    let mr = spec.mr;
    let k = 2 * spec.k_pairs;
    let groups = k / GROUP;
    let acc = |i: usize, c: usize| (4 * i + c) as u8;
    let facc = |i: usize, c: usize| (8 + 4 * i + c) as u8;
    let areg = |i: usize, h: usize| (16 + 2 * i + h) as u8;
    let mut a = Asm::new();
    let frame = Frame::enter(&mut a, false, &[], true, 0);
    for i in 1..mr {
        add_imm(&mut a, row(i), row(i - 1), spec.lda_bytes as i64, X16);
    }
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
    // The epilogue of a group uses v28.. for the second row: the mask is set again every group.
    a.movi(v(28).b16(), 0xf0);
    for i in 0..mr {
        a.ldp(q(areg(i, 0)), q(areg(i, 1)), post(row(i), 32));
        for c in 0..4 {
            a.movi(v(acc(i, c)).d2(), 0);
        }
    }
    for t in 0..4usize {
        a.ldp(q(20), q(21), post(X1, 32));
        a.ldp(q(22), q(23), post(X1, 32));
        // Low nibbles: the step `2t` of `A`; high nibbles: the step `2t+1`.
        for (half, step) in [(0usize, 2 * t), (1, 2 * t + 1)] {
            for c in 0..4u8 {
                if half == 0 {
                    a.shl(v(24 + c).b16(), v(20 + c).b16(), 4);
                } else {
                    a.and_v(v(24 + c).b16(), v(20 + c).b16(), v(28).b16());
                }
            }
            for i in 0..mr {
                for c in 0..4u8 {
                    a.sdot(
                        v(acc(i, c as usize)).s4(),
                        v(24 + c).b16(),
                        v(areg(i, step / 4)).s((step % 4) as u8),
                    );
                }
            }
        }
    }
    // The sixteen f32 scales of the rows of `W`, then per row: · scale of `A`, int32 to f32,
    // accumulate.
    a.ldp(q(20), q(21), post(X1, 32));
    a.ldp(q(22), q(23), post(X1, 32));
    for i in 0..mr {
        let sa = areg(i, 0);
        a.ldr(s(sa), post(x((6 + i) as u8), 4));
        let t = 24 + 4 * i as u8;
        for c in 0..4u8 {
            a.fmul(v(t + c).s4(), v(20 + c).s4(), v(sa).s(0));
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
        .map_err(backend_err(format_args!("gemm q4 {}x16 (k={k})", mr)))
}

/// The kernel of up to four rows, in two passes of eight columns (module documentation).
pub(crate) fn emit4(spec: &MicroKernelSpec) -> Result<Vec<u8>> {
    let mr = spec.mr;
    if mr == 0 || mr > ROWS4 {
        return Err(HerbertError::Backend(format!("backend_jit q4: {mr} rows")));
    }
    let k = 2 * spec.k_pairs;
    let groups = k / GROUP;
    let acc = |i: usize, c: usize| (2 * i + c) as u8;
    let facc = |i: usize, c: usize| (8 + 2 * i + c) as u8;
    let areg = |i: usize, h: usize| (16 + 2 * i + h) as u8;
    let (w0, c0, mask) = (24u8, 26u8, 28u8);
    let masked = spec.n_tail < crate::jit::ir::ROWS_INTERLEAVED;
    let elem = if spec.out_bf16 { 2 } else { 4 };
    let mut a = Asm::new();
    let frame = Frame::enter(&mut a, false, &[], true, 0);
    for i in 1..mr {
        add_imm(&mut a, row(i), row(i - 1), spec.lda_bytes as i64, X16);
    }
    for i in 0..mr {
        a.mov(x((10 + i) as u8), row(i));
    }
    a.mov(X14, X1);
    for h in 0..2usize {
        if h == 1 {
            for i in 0..mr {
                a.mov(row(i), x((10 + i) as u8));
            }
            a.mov(X1, X14);
        }
        for i in 0..mr {
            add_imm(&mut a, x((6 + i) as u8), row(i), k as i64, X16);
            for c in 0..2 {
                a.movi(v(facc(i, c)).d2(), 0);
            }
        }
        let top = a.label();
        a.mov_imm(X15, groups as u64);
        a.bind(top);
        a.movi(v(mask).b16(), 0xf0);
        for i in 0..mr {
            a.ldp(q(areg(i, 0)), q(areg(i, 1)), post(row(i), 32));
            for c in 0..2 {
                a.movi(v(acc(i, c)).d2(), 0);
            }
        }
        for t in 0..4usize {
            a.ldp(q(w0), q(w0 + 1), mem(X1, (t * 64 + 32 * h) as i64));
            for (half, step) in [(0usize, 2 * t), (1, 2 * t + 1)] {
                for c in 0..2u8 {
                    if half == 0 {
                        a.shl(v(c0 + c).b16(), v(w0 + c).b16(), 4);
                    } else {
                        a.and_v(v(c0 + c).b16(), v(w0 + c).b16(), v(mask).b16());
                    }
                }
                for i in 0..mr {
                    for c in 0..2u8 {
                        a.sdot(
                            v(acc(i, c as usize)).s4(),
                            v(c0 + c).b16(),
                            v(areg(i, step / 4)).s((step % 4) as u8),
                        );
                    }
                }
            }
        }
        // The eight f32 scales of these columns, then per row: · scale of `A`, int32 to f32,
        // accumulate.
        a.ldp(q(w0), q(w0 + 1), mem(X1, (256 + 32 * h) as i64));
        add_imm(&mut a, X1, X1, crate::jit::int8::Q4_GROUP_BYTES as i64, X16);
        for i in 0..mr {
            let sa = areg(i, 0);
            a.ldr(s(sa), post(x((6 + i) as u8), 4));
            for c in 0..2u8 {
                a.fmul(v(c0 + c).s4(), v(w0 + c).s4(), v(sa).s(0));
            }
            for c in 0..2 {
                a.scvtf(v(acc(i, c)).s4(), v(acc(i, c)).s4());
            }
            for c in 0..2u8 {
                a.fmla(
                    v(facc(i, c as usize)).s4(),
                    v(acc(i, c as usize)).s4(),
                    v(c0 + c).s4(),
                );
            }
        }
        a.subs(X15, X15, 1);
        a.bne(top);
        // The eight columns of this pass.
        let n = if masked {
            spec.n_tail.saturating_sub(8 * h).min(8)
        } else {
            8
        };
        if n == 0 {
            continue;
        }
        for i in 0..mr {
            if i == 0 {
                a.mov(X16, X2);
            } else {
                add_imm(&mut a, X16, X2, (i * spec.ldc_bytes) as i64, X17);
            }
            let quads = [facc(i, 0), facc(i, 1)];
            let offset = (8 * h * elem) as i64;
            if spec.out_bf16 {
                if n == 8 {
                    crate::jit::arch::aarch64::elem::narrow(&mut a, c0, quads[0], quads[1]);
                    a.str(q(c0), mem(X16, offset));
                } else {
                    crate::jit::arch::aarch64::gemm::store_partial_bf16(
                        &mut a, &quads, c0, X16, offset, n,
                    );
                }
            } else if n == 8 {
                a.stp(q(quads[0]), q(quads[1]), mem(X16, offset));
            } else {
                crate::jit::arch::aarch64::gemm::store_partial_f32(&mut a, &quads, X16, offset, n);
            }
        }
    }
    frame.leave(&mut a);
    a.ret();
    a.finish()
        .map(|c| c.bytes)
        .map_err(backend_err(format_args!(
            "gemm q4 {}x16 (k={k}, four rows)",
            mr
        )))
}
