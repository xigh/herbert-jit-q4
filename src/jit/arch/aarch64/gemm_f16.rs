//! The bf16 GEMM micro-kernels for **f16 weights**, NEON without BF16 (a Cortex-A76).
//!
//! The weights are packed as in `model/weight.rs` with `NeonOptions::w_f16`: the blocks of sixteen
//! rows keep their size, `[K/2 pairs][64 bytes]`, but each 64-byte pair holds f16 values *k-major*:
//! the sixteen rows of k, then the sixteen rows of k+1. So a block is read as a plain stream of
//! 32-byte slices, one per k, each slice two `q` registers of sixteen columns. `A` is bf16 `[M, K]`
//! row-major, `C` as in `gemm.rs`.
//!
//! Two kernels share the layout and the ABI of `gemm.rs` (x0 = row 0 of `A`, x1 = the block of `W`,
//! x2 = `C`; x3..x5 the other rows, x15 the counter, x16/x17 scratch):
//!
//! * **widen** (any height, any K): per k, the slice of `W` is widened by `fcvtl`/`fcvtl2` to four
//!   f32 quads and multiplied by the f32 of `A[i][k]` (the bf16 widened by `shll`) with four `fmla`
//!   by element. Every weight is the f16 of a bf16 and every product is exact in f32, so the sum in
//!   order of k, one fused multiply-add each, has **the bits of the bf16 `bfmlal` kernel**. The
//!   decode (one row) runs this one: it is bound by the memory, not by the instructions.
//! * **fma** (heights 2 to 4, sixteen columns, `K` a multiple of 8, `NeonOptions::f16_flush > 0`):
//!   `A` is converted per group of eight k to f16 (`shll`, `fcvtn`: exact for a bf16 in the f16
//!   range), and per k two `fmla .8h` by element — sixteen columns — add the products into **f16
//!   accumulators**, twice the lanes of an f32 `fmla`. Every `f16_flush` steps of k those
//!   accumulators are widened (`fcvtl`) and added into the f32 ones, and cleared: the accumulation
//!   error stays that of a few terms. The prefill runs this one.
//!
//! ## Registers
//!
//! widen: accumulators `4·mr`, `A` as two f32 quads per row, two of `W`, four of f32 weights: at
//! most 30. fma: `A` f16 `v0..v3` (a half element of `fmla` by element must live in `v0..v15`), `W`
//! `v4`/`v5`, temporaries `v6`/`v7`, the f16 accumulators `v8..v15`, the f32 ones `v16..v31`.

use crate::core::error::{HerbertError, Result};
use crate::jit::arch::aarch64::elem::{narrow, widen};
use crate::jit::arch::aarch64::gemm::{store_partial_bf16, store_partial_f32};
use crate::jit::arch::aarch64::util::{add_imm, backend_err, Frame};
use crate::jit::emit::{AFormat, MicroKernelSpec};
use crate::jit::ir::{Decline, ROWS_INTERLEAVED};
use a64::*;

/// The registers of the allocation order of `gemm.rs`: caller-saved first, then `v8..v15`.
const ORDER: [u8; 32] = [
    0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 8, 9,
    10, 11, 12, 13, 14, 15,
];

fn k_of(spec: &MicroKernelSpec) -> usize {
    2 * spec.k_pairs - usize::from(spec.k_odd)
}

/// The f16 products into f16 accumulators: from height 2, `K` a multiple of 8, and a flush period
/// asked for.
fn use_fma(spec: &MicroKernelSpec) -> bool {
    let k = k_of(spec);
    spec.opts.neon.f16_flush > 0 && spec.mr >= 2 && k > 0 && k.is_multiple_of(8)
}

/// What these kernels emit: bf16 `A` row-major, sixteen columns, up to four rows.
pub fn check(spec: &MicroKernelSpec) -> std::result::Result<(), Decline> {
    if spec.nr != ROWS_INTERLEAVED || spec.mr == 0 || spec.mr > 4 {
        return Err(Decline::TileTooWide);
    }
    if spec.a_fmt != AFormat::Bf16 || spec.opts.neon.a_packed || spec.opts.matrix || spec.k_dynamic
    {
        return Err(Decline::OutOfRegisters);
    }
    Ok(())
}

/// The rows of `A`: row 0 in x0, the others in x3..x5.
pub(super) fn row(i: usize) -> R {
    if i == 0 {
        X0
    } else {
        x((2 + i) as u8)
    }
}

/// Store the `mr × 16` tile held in `quads(i)` (four f32 quads of row `i`): the epilogue of
/// `gemm.rs`.
pub(super) fn store_tile(
    a: &mut Asm,
    spec: &MicroKernelSpec,
    quads: &dyn Fn(usize) -> [u8; 4],
    t: u8,
) {
    let masked = spec.n_tail < ROWS_INTERLEAVED;
    for i in 0..spec.mr {
        let c = X16;
        if i == 0 {
            a.mov(c, X2);
        } else {
            add_imm(a, c, X2, (i * spec.ldc_bytes) as i64, X17);
        }
        let q4 = quads(i);
        if spec.out_bf16 {
            if !masked {
                for half in 0..2 {
                    narrow(a, t, q4[2 * half], q4[2 * half + 1]);
                    a.str(q(t), mem(c, 16 * half as i64));
                }
            } else {
                store_partial_bf16(a, &q4, t, c, 0, spec.n_tail);
            }
        } else if !masked {
            a.stp(q(q4[0]), q(q4[1]), mem(c, 0));
            a.stp(q(q4[2]), q(q4[3]), mem(c, 32));
        } else {
            store_partial_f32(a, &q4, c, 0, spec.n_tail);
        }
    }
}

/// Generates the kernel of `spec`.
pub fn emit(spec: &MicroKernelSpec) -> Result<Vec<u8>> {
    check(spec).map_err(|d| HerbertError::Backend(format!("backend_jit f16: {d}")))?;
    if use_fma(spec) {
        emit_fma(spec)
    } else {
        emit_widen(spec)
    }
}

/// Widening kernel: exact f32 accumulation in the order of k.
fn emit_widen(spec: &MicroKernelSpec) -> Result<Vec<u8>> {
    let mr = spec.mr;
    let k = k_of(spec);
    // Allocation: accumulators, `A` (low and high quad per row), `W` (two), f32 weights (four).
    let mut cursor = 0usize;
    let mut take = |n: usize| -> Vec<u8> {
        let r = ORDER[cursor..cursor + n].to_vec();
        cursor += n;
        r
    };
    let acc: Vec<[u8; 4]> = (0..mr).map(|_| take(4).try_into().unwrap()).collect();
    let af: Vec<[u8; 2]> = (0..mr).map(|_| take(2).try_into().unwrap()).collect();
    let w = take(2);
    let cv = take(4);
    let save = cursor > 24;
    let mut a = Asm::new();
    let frame = Frame::enter(&mut a, false, &[], save, 0);
    for i in 1..mr {
        add_imm(&mut a, row(i), row(i - 1), spec.lda_bytes as i64, X16);
    }
    for q4 in &acc {
        for &r in q4 {
            a.movi(v(r).d2(), 0);
        }
    }
    // One k: the slice of sixteen f16 weights widened, times `A[i][k]` (lane `lane` of
    // `af[i][half]`).
    let step = |a: &mut Asm, half: usize, lane: u8| {
        a.ldp(q(w[0]), q(w[1]), post(X1, 32));
        a.fcvtl(v(cv[0]).s4(), v(w[0]).h4());
        a.fcvtl(v(cv[1]).s4(), v(w[0]).h8());
        a.fcvtl(v(cv[2]).s4(), v(w[1]).h4());
        a.fcvtl(v(cv[3]).s4(), v(w[1]).h8());
        for i in 0..mr {
            for qq in 0..4 {
                a.fmla(v(acc[i][qq]).s4(), v(cv[qq]).s4(), v(af[i][half]).s(lane));
            }
        }
    };
    let groups = k / 8;
    if groups > 0 {
        let top = a.label();
        a.mov_imm(X15, groups as u64);
        a.bind(top);
        for i in 0..mr {
            // Eight bf16 of the row: the high four widened first, then the low four in place.
            a.ldr(q(af[i][0]), post(row(i), 16));
            widen(&mut a, af[i][1], af[i][0], true);
            widen(&mut a, af[i][0], af[i][0], false);
        }
        for kk in 0..8usize {
            step(&mut a, kk / 4, (kk % 4) as u8);
        }
        a.subs(X15, X15, 1);
        a.bne(top);
    }
    for _ in 0..k % 8 {
        for i in 0..mr {
            a.ldr(h(af[i][0]), post(row(i), 2));
            widen(&mut a, af[i][0], af[i][0], false);
        }
        step(&mut a, 0, 0);
    }
    store_tile(&mut a, spec, &|i| acc[i], w[0]);
    frame.leave(&mut a);
    a.ret();
    a.finish()
        .map(|c| c.bytes)
        .map_err(backend_err(format_args!(
            "gemm f16 widen {}x16 (k={k})",
            mr
        )))
}

/// f16 kernel: products into f16 accumulators, flushed into the f32 ones every `f16_flush` steps of
/// k.
fn emit_fma(spec: &MicroKernelSpec) -> Result<Vec<u8>> {
    let mr = spec.mr;
    let k = k_of(spec);
    let flush = usize::from(spec.opts.neon.f16_flush).max(8) / 8 * 8;
    let groups = k / 8;
    let per_flush = (flush / 8).min(groups).max(1);
    let acch = |i: usize, c: usize| 8 + 2 * i as u8 + c as u8;
    let accf = |i: usize, qq: usize| 16 + 4 * i as u8 + qq as u8;
    let mut a = Asm::new();
    let frame = Frame::enter(&mut a, false, &[], true, 0);
    for i in 1..mr {
        add_imm(&mut a, row(i), row(i - 1), spec.lda_bytes as i64, X16);
    }
    for i in 0..mr {
        for qq in 0..4 {
            a.movi(v(accf(i, qq)).d2(), 0);
        }
        for c in 0..2 {
            a.movi(v(acch(i, c)).d2(), 0);
        }
    }
    // `n` groups of eight k: `A` to f16 per row, then per k two `fmla .8h` by element per row.
    let emit_groups = |a: &mut Asm, n: usize| {
        for _ in 0..n {
            for i in 0..mr {
                a.ldr(q(i as u8), post(row(i), 16));
                widen(a, 6, i as u8, false);
                widen(a, 7, i as u8, true);
                a.fcvtn(v(i as u8).h4(), v(6).s4());
                a.fcvtn(v(i as u8).h8(), v(7).s4());
            }
            for kk in 0..8u8 {
                a.ldp(q(4), q(5), post(X1, 32));
                for i in 0..mr {
                    a.fmla_h(v(acch(i, 0)).h8(), v(4).h8(), v(i as u8).h(kk));
                    a.fmla_h(v(acch(i, 1)).h8(), v(5).h8(), v(i as u8).h(kk));
                }
            }
        }
    };
    let flush_acc = |a: &mut Asm| {
        for i in 0..mr {
            for c in 0..2 {
                let hacc = v(acch(i, c));
                a.fcvtl(v(6).s4(), hacc.h4());
                a.fcvtl(v(7).s4(), hacc.h8());
                a.fadd(v(accf(i, 2 * c)).s4(), v(accf(i, 2 * c)).s4(), v(6).s4());
                a.fadd(
                    v(accf(i, 2 * c + 1)).s4(),
                    v(accf(i, 2 * c + 1)).s4(),
                    v(7).s4(),
                );
                a.movi(hacc.d2(), 0);
            }
        }
    };
    let blocks = groups / per_flush;
    if blocks > 0 {
        let top = a.label();
        a.mov_imm(X15, blocks as u64);
        a.bind(top);
        emit_groups(&mut a, per_flush);
        flush_acc(&mut a);
        a.subs(X15, X15, 1);
        a.bne(top);
    }
    let rest = groups % per_flush;
    if rest > 0 {
        emit_groups(&mut a, rest);
        flush_acc(&mut a);
    }
    store_tile(
        &mut a,
        spec,
        &|i| [accf(i, 0), accf(i, 1), accf(i, 2), accf(i, 3)],
        4,
    );
    frame.leave(&mut a);
    a.ret();
    a.finish()
        .map(|c| c.bytes)
        .map_err(backend_err(format_args!(
            "gemm f16 fma {}x16 (k={k}, flush {flush})",
            mr
        )))
}
