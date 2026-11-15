//! The bf16 GEMM micro-kernel on the matrix engine: SME2 `bfmopa` into ZA.
//!
//! `bfmopa za<t>.s, p/m, p/m, zn.h, zm.h` adds to every f32 element `(i, j)`
//! of a 16×16 tile (streaming vector length 64 bytes) the two products of a
//! pair of `K`: `zn.h[2i]·zm.h[2j] + zn.h[2i+1]·zm.h[2j+1]`. A pair of `K`
//! for sixteen columns of `W` is 64 contiguous bytes of the R16 layout: one
//! `ld1h` is the `zm` of one tile. The same pair for sixteen rows of `A` is
//! one `ld1h` of the packed activations (`jit/emit.rs`, [`packed_a_bytes`]):
//! the packing node of the plan has put `A` in that layout once for all the
//! projections that read it.
//!
//! A call computes up to 16 rows × 64 columns: the four 32-bit tiles of ZA
//! accumulate four blocks of sixteen columns, each pair of `K` costs one load
//! of `A`, one load per block and one outer product per block. The four tiles
//! are independent, which is what the matrix engine needs to overlap its
//! products. The epilogue reads each row of each tile — `st1w` straight from
//! ZA for an f32 output; for a bf16 output `mova` to a vector, `bfcvt` (round
//! to nearest even) and the truncating `st1h`. Only the `height` valid rows and
//! the `n_tail` valid columns of the last block are stored.
//!
//! `smstart` and `smstop` clear the vector registers: the prologue saves the
//! low halves of `v8..v15`, which belong to the caller.
//!
//! [`packed_a_bytes`]: crate::jit::emit::packed_a_bytes

use crate::core::error::{HerbertError, Result};
use crate::jit::arch::aarch64::util::{add_imm, backend_err, Frame};
use crate::jit::emit::{b_block_bytes, AFormat, MicroKernelSpec, PACK_ROWS};
use crate::jit::ir::{Decline, ROWS_INTERLEAVED};
use a64::*;

/// The widest tile: sixteen rows, the four ZA tiles of sixteen columns.
pub const SME_MAX_COLS: usize = 64;

/// Is the matrix engine usable here: SME2 with the bf16 outer product.
pub fn available() -> bool {
    let f = crate::core::isa::feature;
    f("sme2")
        && f("sme_b16f32")
        && crate::core::isa::detected()
            .as_ref()
            .is_ok_and(|d| d.sme_max_svl_bytes == 64)
}

/// The shapes this kernel serves.
pub fn check(spec: &MicroKernelSpec) -> std::result::Result<(), Decline> {
    if !available() {
        return Err(Decline::TileTooWide);
    }
    if spec.mr == 0
        || spec.mr > PACK_ROWS
        || spec.nr == 0
        || spec.nr > SME_MAX_COLS
        || !spec.nr.is_multiple_of(ROWS_INTERLEAVED)
    {
        return Err(Decline::TileTooWide);
    }
    // A packed block holds whole pairs; the loop takes them four at a time.
    if spec.k_odd || spec.a_fmt != AFormat::Bf16 || !spec.k_pairs.is_multiple_of(4) {
        return Err(Decline::OddK);
    }
    Ok(())
}

/// The residue rows of the matrix path on SME2's `bfdot` into ZA
/// (`NeonMac::SmeDot`): `spec.mr` rows (at most eight) of a packed block of
/// `A`, the `spec.nr` columns of a tile (`n_tail` valid in the last block).
/// Each pair of `K` costs one load per block of `W`, one `ld1rw` of the
/// row's pair, and per row one `bfdot` over the four blocks (`vgx4`): row `t`
/// of block `j` accumulates in the ZA vector `t + 16·j` (slice `t/4 + 4·j`
/// of tile `t mod 4`), the pairs in the order of the tile, by the
/// arithmetic of `bfmopa` — the bits of the tile, without the sixteen-row
/// outer product of a padded tile. The epilogue is the tile's.
pub fn emit_rows(spec: &MicroKernelSpec) -> Result<Vec<u8>> {
    check(spec).map_err(|d| HerbertError::Backend(format!("backend_jit sme rows: {d}")))?;
    if spec.mr > 8 {
        return Err(HerbertError::Backend(format!(
            "backend_jit sme rows: {} rows, at most 8",
            spec.mr
        )));
    }
    let nb = spec.nr / ROWS_INTERLEAVED;
    let bstride = b_block_bytes(spec.k_pairs) as i64;
    let block = |j: usize| if j == 0 { X1 } else { x((2 + j) as u8) };
    let mut a = Asm::new();
    let frame = Frame::enter(&mut a, false, &[], true, 0);
    for j in 1..nb {
        add_imm(&mut a, block(j), block(j - 1), bstride, X16);
    }
    a.smstart();
    a.zero_za();
    a.ptrue(p(0), Et::H);
    a.ptrue(p(2), Et::S);
    // The blocks of pair `u` in z8+4u.. (a group of four, zero beyond `nb`), the rows' pairs in
    // z0..z7.
    if nb < 4 {
        a.mov_imm(X17, 0);
        for u in 0..4u8 {
            for j in nb..4 {
                a.dup_z(z(8 + 4 * u + j as u8), Et::S, W17);
            }
        }
    }
    a.mov_imm(W8, 0);
    let top = a.label();
    a.mov_imm(X15, (spec.k_pairs / 4) as u64);
    a.bind(top);
    for u in 0..4u8 {
        for j in 0..nb {
            a.ld1_z(
                z(8 + 4 * u + j as u8),
                Et::H,
                p(0),
                mem(block(j), i64::from(u)),
            );
        }
    }
    for u in 0..4u8 {
        for t in 0..spec.mr {
            a.ld1rw_z(z(t as u8), p(0), X0, u32::from(u) * 64 + 4 * t as u32);
        }
        for t in 0..spec.mr {
            a.bfdot_za_vgx4(W8, t as u8, z(8 + 4 * u), z(t as u8));
        }
    }
    a.addvl(X0, X0, 4);
    for j in 0..nb {
        a.addvl(block(j), block(j), 4);
    }
    a.subs(X15, X15, 1);
    a.bne(top);

    // ── Epilogue: row by row, block by block. ────────────────────────────
    let masked = spec.n_tail < ROWS_INTERLEAVED;
    if masked {
        a.mov_imm(X17, spec.n_tail as u64);
        a.whilelo(p(1), Et::S, XZR, X17);
    }
    a.mov(X16, X2);
    for t in 0..spec.mr {
        for j in 0..nb {
            let pred = if masked && j == nb - 1 { p(1) } else { p(2) };
            a.mov_imm(W12, (4 * j) as u64);
            let slice = ZaS {
                tile: (t % 4) as u8,
                dir: Slice::H,
                index: W12,
                offset: (t / 4) as u8,
            };
            a.mov_imm(X17, (j * 16) as u64);
            if spec.out_bf16 {
                a.mova_from_za(z(0), p(2), slice);
                a.bfcvt_z(z(0), p(2), z(0));
                a.st1h_s(z(0), pred, idx_scaled(X16, X17));
            } else {
                a.st1w_za(slice, pred, X16, X17);
            }
        }
        add_imm(&mut a, X16, X16, spec.ldc_bytes as i64, X17);
    }
    a.smstop();
    frame.leave(&mut a);
    a.ret();
    a.finish()
        .map(|c| c.bytes)
        .map_err(backend_err(format_args!(
            "sme rows {}x{} (k_pairs={})",
            spec.mr, spec.nr, spec.k_pairs
        )))
}

/// Generates the kernel described by `spec`, `A` packed.
pub fn emit(spec: &MicroKernelSpec) -> Result<Vec<u8>> {
    check(spec).map_err(|d| HerbertError::Backend(format!("backend_jit sme gemm: {d}")))?;
    let nb = spec.nr / ROWS_INTERLEAVED;
    let bstride = b_block_bytes(spec.k_pairs) as i64;
    let block = |j: usize| if j == 0 { X1 } else { x((2 + j) as u8) };
    let mut a = Asm::new();
    let frame = Frame::enter(&mut a, false, &[], true, 0);
    for j in 1..nb {
        add_imm(&mut a, block(j), block(j - 1), bstride, X16);
    }
    a.smstart();
    a.zero_za();
    a.ptrue(p(0), Et::H);
    a.ptrue(p(2), Et::S);
    // Four pairs per turn: `A` in z0..z3, the blocks in z4.., one vector each.
    // Under `k_dynamic` the caller has put the groups in `x15`.
    let top = a.label();
    if !spec.k_dynamic {
        a.mov_imm(X15, (spec.k_pairs / 4) as u64);
    }
    a.bind(top);
    for u in 0..4i64 {
        a.ld1_z(z(u as u8), Et::H, p(0), mem(X0, u));
    }
    for j in 0..nb {
        for u in 0..4i64 {
            a.ld1_z(
                z(4 + (4 * j + u as usize) as u8),
                Et::H,
                p(0),
                mem(block(j), u),
            );
        }
    }
    for u in 0..4u8 {
        for j in 0..nb {
            a.bfmopa(j as u8, p(0), p(0), z(u), z(4 + 4 * j as u8 + u));
        }
    }
    a.addvl(X0, X0, 4);
    for j in 0..nb {
        a.addvl(block(j), block(j), 4);
    }
    a.subs(X15, X15, 1);
    a.bne(top);

    // ── Epilogue: row by row, tile by tile. ──────────────────────────────
    // x16 = the C row, x17 = the column index of the block; w12 + i = the row.
    let masked = spec.n_tail < ROWS_INTERLEAVED;
    if masked {
        a.mov_imm(X17, spec.n_tail as u64);
        a.whilelo(p(1), Et::S, XZR, X17);
    }
    a.mov(X16, X2);
    for group in 0..spec.mr.div_ceil(4) {
        a.mov_imm(W12, (4 * group) as u64);
        for off in 0..4u8 {
            let row = 4 * group + usize::from(off);
            if row >= spec.mr {
                break;
            }
            for j in 0..nb {
                let pred = if masked && j == nb - 1 { p(1) } else { p(2) };
                let slice = ZaS {
                    tile: j as u8,
                    dir: Slice::H,
                    index: W12,
                    offset: off,
                };
                a.mov_imm(X17, (j * 16) as u64);
                if spec.out_bf16 {
                    a.mova_from_za(z(0), p(2), slice);
                    a.bfcvt_z(z(0), p(2), z(0));
                    a.st1h_s(z(0), pred, idx_scaled(X16, X17));
                } else {
                    a.st1w_za(slice, pred, X16, X17);
                }
            }
            add_imm(&mut a, X16, X16, spec.ldc_bytes as i64, X17);
        }
    }
    a.smstop();
    frame.leave(&mut a);
    a.ret();
    a.finish()
        .map(|c| c.bytes)
        .map_err(backend_err(format_args!(
            "sme gemm {}x{} (k_pairs={})",
            spec.mr, spec.nr, spec.k_pairs
        )))
}

/// The residue kernel (`emit_rows`) against the tile (`emit`) on the same packed rows: the bits of
/// the tile for every row count up to six, every width, a masked last group, f32 and bf16 outputs.
/// Skipped without SME2.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::emit::{KernelOptions, NeonMac, NeonOptions};

    fn rng(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }

    fn bf16(x: f32) -> u16 {
        half::bf16::from_f32(x).to_bits()
    }

    #[test]
    fn rows_have_the_bits_of_the_tile() {
        if !available() {
            eprintln!("no SME2: skipped");
            return;
        }
        let (k, n) = (264usize, 64usize);
        let k_pairs = k / 2;
        let mut seed = 7u64;
        // Values over several decades so that the order and the rounding of the sums show.
        let value = |seed: &mut u64| bf16(rng(seed) * 10f32.powf(3.0 * rng(seed).abs() - 1.5));
        let a: Vec<u16> = (0..16 * k).map(|_| value(&mut seed)).collect();
        let w: Vec<u16> = (0..n * k).map(|_| value(&mut seed)).collect();
        // A packed `[pairs][16 rows][2]`, W in R16 `[block][pairs][16 columns][2]`.
        let mut ap = vec![0u16; 16 * k];
        for p in 0..k_pairs {
            for r in 0..16 {
                ap[(p * 16 + r) * 2] = a[r * k + 2 * p];
                ap[(p * 16 + r) * 2 + 1] = a[r * k + 2 * p + 1];
            }
        }
        let mut wp = vec![0u16; n * k];
        for b in 0..n / 16 {
            for p in 0..k_pairs {
                for c in 0..16 {
                    wp[((b * k_pairs + p) * 16 + c) * 2] = w[(b * 16 + c) * k + 2 * p];
                    wp[((b * k_pairs + p) * 16 + c) * 2 + 1] = w[(b * 16 + c) * k + 2 * p + 1];
                }
            }
        }
        for out_bf16 in [false, true] {
            for nr in [16usize, 32, 48, 64] {
                for n_tail in [16usize, 5] {
                    for mr in 1..=6 {
                        let spec = |opts: KernelOptions| MicroKernelSpec {
                            mr,
                            nr,
                            k_pairs,
                            k_odd: false,
                            lda_bytes: 2 * k,
                            a_fmt: AFormat::Bf16,
                            ldc_bytes: n * if out_bf16 { 2 } else { 4 },
                            n_tail,
                            out_bf16,
                            opts,
                            k_dynamic: false,
                        };
                        let tile = emit(&spec(KernelOptions {
                            matrix: true,
                            ..Default::default()
                        }))
                        .unwrap();
                        let neon = NeonOptions {
                            a_packed: true,
                            mac: NeonMac::SmeDot,
                            ..Default::default()
                        };
                        let rows = emit_rows(&spec(KernelOptions {
                            neon,
                            ..Default::default()
                        }))
                        .unwrap();
                        let run = |code: &[u8]| -> Vec<u8> {
                            let exec =
                                crate::jit::arch::map_exec(code.len(), |m| m.copy_from_slice(code))
                                    .unwrap();
                            let mut c = vec![0xa5u8; 6 * n * 4];
                            let f: extern "C" fn(*const u16, *const u16, *mut u8) =
                                unsafe { std::mem::transmute(exec.as_ptr()) };
                            f(ap.as_ptr(), wp.as_ptr(), c.as_mut_ptr());
                            c
                        };
                        assert_eq!(
                            run(&tile),
                            run(&rows),
                            "mr {mr} nr {nr} n_tail {n_tail} bf16 {out_bf16}"
                        );
                    }
                }
            }
        }
    }
}
