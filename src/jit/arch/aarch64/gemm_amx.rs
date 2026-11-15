//! The GEMM micro-kernels on Apple's AMX: `matfp` bf16 → f32 or f16 → f32,
//! or `fma32` f16 × f32 → f32, into Z.
//!
//! `matfp` with data type 1 adds to Z the outer product of the 32 bf16 of Y
//! and the 32 bf16 of X, in f32, each lane one fused multiply-add rounded to
//! nearest even, denormals kept (`crates/a64/src/amx.rs`): the arithmetic of
//! one `bfmlalb` or `bfmlalt` lane. Lane `i` of X times lane `j` of Y lands
//! in row `2j + (i & 1)` of Z, column `i >> 1`. Data type 3 is the same
//! outer product of f16 operands.
//!
//! **bf16** (data type 1, the M2 and later: the M1's unit has no bf16). X
//! takes a pair of `K` of one R16 block of `W` — 64 contiguous bytes,
//! sixteen columns `n` × the two elements of the pair, lane `2n + t` — and Y
//! one `K` of thirty-two rows of `A`, packed by the plan's packing node
//! (`[K][32]`, `jit/emit.rs`, [`packed_a_amx_bytes`]). The even element of
//! the pair lands, for row `m` and column `n`, in row `2m` of Z, column `n`.
//! The odd element is brought to the same place by reading X two bytes
//! further (lane `2n` then holds element `2n + 1`) against the next `K` of
//! `A`. So each element of `C` is one accumulator of Z that receives its `K`
//! products one after the other, the even element of a pair before the odd,
//! the pairs in order — the order and the arithmetic of the vector kernel
//! under `bfmlal` (`gemm.rs`), hence its bits. The odd rows of Z collect
//! cross products and are never read: half of each outer product is spent,
//! the price of keeping both the R16 layout and that order.
//!
//! **f16** (`--f16`, every Apple M). The block of `W` is the k-major one of
//! `NeonOptions::w_f16` (`model/weight.rs`): in the 64 bytes of a pair, the
//! sixteen f16 of the even `K`, then the sixteen of the odd. Three forms of
//! the activations, `NeonOptions::amx_a` ([`AmxF16A`], `--amx-a`):
//!
//! * `F32` (the default): `A` widened to f32 by the packing node (exact), and `fma32` with X read
//!   as f16 (see `body_fma32`): one fused multiply-add per `k` in order, the f16 weight widened
//!   exactly, whatever the activation — the bits of the `widen` kernel of `gemm_f16.rs` (`--f16
//!   --f16-flush 0`) for every input, subnormals and values beyond the f16 range included, save the
//!   payload of a NaN: the unit returns the default NaN, `widen` the NaN of its input (both NaN).
//! * `Mixed`: per block of thirty-two rows, the packing node says in a header word whether every
//!   activation came back unchanged from f16; if so the `matfp` f16 path below, otherwise the `F32`
//!   one. The same bits.
//! * `F16`: the `matfp` path for every block, with every activation rounded to f16: the bits of the
//!   `widen` kernel while every activation is exact in f16 (from 2^-17 to 65504 in magnitude);
//!   outside, those of `widen` fed with the activations rounded to f16.
//!
//! The `matfp` f16 path (data type 3): X at the pair puts column `n` of the
//! even `K` in lane `n`; X 32 bytes further, the odd `K`. Y is one `K` of
//! thirty-two rows of `A` converted to f16 by the packing node (exact for a
//! bf16 in the f16 range, as the `fma` kernel of `gemm_f16.rs` converts it).
//! Lane `n < 16` lands in row `2m + (n & 1)`, column `n >> 1`: the
//! interleaved pair of rows `2m`, `2m + 1` holds row `m` of `C`, which one
//! `stzi` stores in order. Lanes 16 to 31 (the other `K`, or the next X
//! register) land in columns 8 to 15, never read: the same half spent. A
//! product of two f16 is exact in f32, so each accumulator receives
//! `A[m][k] · W[n][k]` rounded once per addition, in the order of `k`.
//!
//! A call computes 32 rows × 16 columns. `set` hands a zeroed Z, as the vector
//! kernel's accumulators start at +0. `matfp`: four pairs per turn, four `ldx`
//! (one pair of `W` each), eight `ldy` (one `K` of `A` each), eight `matfp`.
//! The epilogue stores the `height` useful rows of Z to the stack (`stz`, or
//! `stzi` for f16), and the core converts them (`bfcvtn`, nearest even, as
//! the vector kernel) or copies them, storing the `n_tail` valid columns only.
//!
//! Measured (`F32`, 2026-11-05): a
//! first form that widened `W` on the core (`fcvtl`) into the stack for
//! `ldx` ran at less than half the speed of `matfp` f16; the same kernel
//! without the conversion (wrong values, for the time only) ran faster than
//! `matfp` f16: the cost was the round trip of the core's stores to the
//! unit's loads, not the `fma32`. Hence X read as f16 by the unit.
//!
//! [`packed_a_amx_bytes`]: crate::jit::emit::packed_a_amx_bytes

use crate::core::error::{HerbertError, Result};
use crate::jit::arch::aarch64::elem::has_bf16;
use crate::jit::arch::aarch64::gemm::{store_partial_bf16, store_partial_f32};
use crate::jit::arch::aarch64::util::{add_imm, backend_err, Frame};
use crate::jit::emit::{AFormat, AmxF16A, MicroKernelSpec, AMX_ROWS};
use crate::jit::ir::{Decline, TileShape, ROWS_INTERLEAVED};
use a64::*;

/// The tile: thirty-two rows of `A`, one R16 block of `W`.
pub const AMX_TILE: TileShape = TileShape {
    mr: AMX_ROWS,
    nr: ROWS_INTERLEAVED,
};

/// `matfp`, data type 1 (bits 42–45): bf16 × bf16 → f32 (M2 and later).
const TYPE_BF16_F32: u64 = 1 << 42;

/// `matfp`, data type 3: f16 × f16 → f32 (M1 and later).
const TYPE_F16_F32: u64 = 3 << 42;

/// Is the AMX there: every Apple M so far (M1 to M5; checked on the M1, the
/// M3 Ultra and the M5). The knob `Knobs::gemm_amx` asks for it.
pub fn available() -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
}

/// Does the unit take the weights of this process: f16 (`--f16`) on any
/// Apple M; bf16 where the core has FEAT_BF16, the M2 and later — the M1's
/// unit reads data type 1 as f16 → f16 (corsix/amx, `matfp.md`).
pub fn serves(w_f16: bool) -> bool {
    available() && (w_f16 || has_bf16())
}

/// The shapes this kernel serves.
pub fn check(spec: &MicroKernelSpec) -> std::result::Result<(), Decline> {
    if !serves(spec.opts.neon.w_f16) {
        return Err(Decline::TileTooWide);
    }
    if spec.mr == 0 || spec.mr > AMX_ROWS || spec.nr != ROWS_INTERLEAVED {
        return Err(Decline::TileTooWide);
    }
    // The packed block holds whole `K`s of bf16; the loop takes four pairs a turn.
    if spec.k_odd
        || spec.k_dynamic
        || spec.a_fmt != AFormat::Bf16
        || !spec.k_pairs.is_multiple_of(4)
    {
        return Err(Decline::OddK);
    }
    Ok(())
}

/// `matfp` operand: data type `ty`, X at byte `x_off`, Y at byte `y_off`.
const fn matfp(ty: u64, x_off: u64, y_off: u64) -> u64 {
    ty | x_off << 10 | y_off
}

/// `fma32` operand, matrix mode, X read as f16 (bit 61: lane `i` is the f16
/// at byte `x_off + 4i`, widened exactly): X at byte `x_off` (modulo the 512
/// bytes of X), Y at byte `y_off`, the accumulator `z` (0 to 3, rows `4j + z`
/// of Z) at bits 20–25; lanes 0–7 only (`high` false: enable mode 2, the
/// first eight) or 8–15 only (mode 3, the last eight), bits 41–47.
const fn fma32_x16(x_off: u64, y_off: u64, z: u64, high: bool) -> u64 {
    let enable = if high { 3 << 5 | 8 } else { 2 << 5 | 8 };
    1 << 61 | enable << 41 | z << 20 | (x_off % 512) << 10 | y_off
}

/// The registers of the main loops. x3..x6: the four X operands of a turn,
/// each with its X register in bits 56–58; x7..x14: the eight Y operands,
/// each with its Y register; x15 the turns; x19..x26 the eight `matfp` or
/// `fma32` operands.
fn wp(p: usize) -> R {
    x((3 + p) as u8)
}
fn ak(u: usize) -> R {
    x((7 + u) as u8)
}
fn op(i: usize) -> R {
    x((19 + i) as u8)
}

/// The rows of Z stored for the core, 64 bytes each, aligned on 64.
const SCRATCH: usize = AMX_ROWS * 64 + 64;

/// Generates the kernel described by `spec`, `A` packed `[K][32]`.
pub fn emit(spec: &MicroKernelSpec) -> Result<Vec<u8>> {
    check(spec).map_err(|d| HerbertError::Backend(format!("backend_jit amx gemm: {d}")))?;
    let f16 = spec.opts.neon.w_f16;
    let saved = [X19, X20, X21, X22, X23, X24, X25, X26];
    let mut a = Asm::new();
    let frame = Frame::enter(&mut a, false, &saved, false, SCRATCH);
    // x16 = the scratch, aligned.
    a.add(X16, SP, 63);
    a.and(X16, X16, !63u64);
    match (f16, spec.opts.neon.amx_a) {
        (false, _) => body_matfp(&mut a, spec, false),
        (true, AmxF16A::F16) => body_matfp(&mut a, spec, true),
        (true, AmxF16A::F32) => body_fma32(&mut a, spec),
        (true, AmxF16A::Mixed) => {
            // The header word of the block: 0 f16, 1 f32; the data 64 bytes on.
            let wide = a.label();
            let done = a.label();
            a.ldr(w(9), at(X0));
            a.add(X0, X0, 64);
            a.cbnz(w(9), wide);
            body_matfp(&mut a, spec, true);
            a.b(done);
            a.bind(wide);
            body_fma32(&mut a, spec);
            a.bind(done);
        }
    }
    frame.leave(&mut a);
    a.ret();
    a.finish()
        .map(|c| c.bytes)
        .map_err(backend_err(format_args!(
            "amx gemm {}x{} (k_pairs={}, f16 {f16})",
            spec.mr, spec.nr, spec.k_pairs
        )))
}

/// `matfp` bf16 (or f16 with `f16`), `A` packed in 64 bytes per `K`: the
/// loop, then the rows of Z to the stack and to `C`.
fn body_matfp(a: &mut Asm, spec: &MicroKernelSpec, f16: bool) {
    // bf16: the odd element of a pair two bytes on, rows 2m of Z; f16: the odd
    // `K` 32 bytes on, the interleaved pair of rows 2m, 2m + 1.
    let (ty, odd, store) = if f16 {
        (TYPE_F16_F32, 32, AmxOp::Stzi)
    } else {
        (TYPE_BF16_F32, 2, AmxOp::Stz)
    };
    for p in 0..4u64 {
        a.mov_imm(X17, p * 64 + (p << 56));
        a.add(wp(p as usize), X1, X17);
    }
    for u in 0..8u64 {
        a.mov_imm(X17, u * 64 + (u << 56));
        a.add(ak(u as usize), X0, X17);
    }
    for p in 0..4u64 {
        // Even element: X at the pair, Y at K = 2p; odd: X `odd` bytes on, Y at K = 2p + 1.
        a.mov_imm(op(2 * p as usize), matfp(ty, p * 64, 2 * p * 64));
        a.mov_imm(
            op(2 * p as usize + 1),
            matfp(ty, p * 64 + odd, (2 * p + 1) * 64),
        );
    }
    a.amx_set();
    a.mov_imm(X15, (spec.k_pairs / 4) as u64);
    let top = a.here();
    for p in 0..4 {
        a.amx(AmxOp::Ldx, wp(p));
    }
    for u in 0..8 {
        a.amx(AmxOp::Ldy, ak(u));
    }
    for i in 0..8 {
        a.amx(AmxOp::Matfp, op(i));
    }
    for p in 0..4 {
        a.add(wp(p), wp(p), 256);
    }
    for u in 0..8 {
        a.add(ak(u), ak(u), 512);
    }
    a.subs(X15, X15, 1);
    a.bne(top);
    // Row m of C: row 2m of Z, or the pair 2m, 2m + 1 by its high five bits
    // and its left half.
    epilogue(a, spec, store, &|m| 2 * m, false);
}

/// `fma32` (matrix mode, f32 × f32 → f32, sixteen by sixteen) with X read
/// as f16, `A` packed in f32, 128 bytes per `K`: rows 0–15 then 16–31. X
/// takes a pair of `K` of `W` as the f16 kernel does; each lane of an `fma32`
/// reads the f16 four bytes from the previous one, so one operation takes the
/// even columns of a `K` into lanes 0–7 (X at the `K`, the first eight lanes
/// enabled) and another the odd columns into lanes 8–15 (X 30 bytes before
/// it, the last eight enabled), each against the `K` of `A` in Y. Z lane `i`
/// is column `2i` (`i < 8`) or `2(i − 8) + 1`. Rows 0–15 accumulate in rows
/// `4j` of Z, 16–31 in rows `4j + 1`. Each element of `C` is one accumulator
/// receiving `A[m][k] · W[n][k]` fused, the f16 widened exactly, in the order
/// of `k`: the `widen` kernel. Per turn, two pairs: two `ldx`, eight `ldy`
/// (four `K` of `A`, two halves each), sixteen `fma32`.
fn body_fma32(a: &mut Asm, spec: &MicroKernelSpec) {
    // x3/x4: the two pairs of W of a turn (X 0 and 1); x7..x14: the four Ks
    // of A, two halves each (Y 0..7); x19..x26: the operands of the first
    // pair (X 0, Y 0..3), x0 the step to those of the second (X 1, Y 4..7):
    // 64 more on the X offset, whose carry out of its nine bits lands in
    // ignored bit 19, and 256 more on the Y offset.
    for p in 0..2u64 {
        a.mov_imm(X17, p * 64 + (p << 56));
        a.add(wp(p as usize), X1, X17);
    }
    for u in 0..8u64 {
        a.mov_imm(X17, u * 64 + (u << 56));
        a.add(ak(u as usize), X0, X17);
    }
    let mut i = 0;
    for kk in 0..2u64 {
        for g in 0..2u64 {
            for high in [false, true] {
                let x_off = 32 * kk + if high { 512 - 30 } else { 0 };
                a.mov_imm(op(i), fma32_x16(x_off, (2 * kk + g) * 64, g, high));
                i += 1;
            }
        }
    }
    a.mov_imm(X0, (64 << 10) + 256);
    a.amx_set();
    a.mov_imm(X15, (spec.k_pairs / 2) as u64);
    let top = a.here();
    for p in 0..2 {
        a.amx(AmxOp::Ldx, wp(p));
    }
    for u in 0..8 {
        a.amx(AmxOp::Ldy, ak(u));
    }
    for i in 0..8 {
        a.amx(AmxOp::Fma32, op(i));
    }
    for i in 0..8 {
        a.add(X17, op(i), X0);
        a.amx(AmxOp::Fma32, X17);
    }
    for p in 0..2 {
        a.add(wp(p), wp(p), 128);
    }
    for u in 0..8 {
        a.add(ak(u), ak(u), 512);
    }
    a.subs(X15, X15, 1);
    a.bne(top);
    // Row m of C: row 4m of Z (m < 16), row 4(m − 16) + 1 otherwise; its
    // columns even then odd.
    epilogue(a, spec, AmxOp::Stz, &|m| 4 * (m % 16) + m / 16, true);
}

/// The `mr` rows of C: row `m` from the row (or pair) `zrow(m)` of Z, stored
/// to the scratch by `store`, then converted (`bfcvtn`, nearest even, as the
/// vector kernel) or copied by the core, the `n_tail` valid columns only.
/// With `split`, the row holds the even columns then the odd ones: `zip1` /
/// `zip2` put them back in order.
fn epilogue(
    a: &mut Asm,
    spec: &MicroKernelSpec,
    store: AmxOp,
    zrow: &dyn Fn(u64) -> u64,
    split: bool,
) {
    // x9 = the stz (stzi) operand; x15 walks the scratch, x10 the row of C
    // (x17 is the scratch register of the partial stores).
    for m in 0..spec.mr as u64 {
        a.mov_imm(X9, m * 64 + (zrow(m) << 56));
        a.add(X9, X16, X9);
        a.amx(store, X9);
    }
    a.amx_clr();
    a.mov(X15, X16);
    a.mov(X10, X2);
    for m in 0..spec.mr {
        a.ldp(q(0), q(1), mem(X15, 0));
        a.ldp(q(2), q(3), mem(X15, 32));
        let quads = if split {
            a.zip1(v(16).s4(), v(0).s4(), v(2).s4());
            a.zip2(v(17).s4(), v(0).s4(), v(2).s4());
            a.zip1(v(18).s4(), v(1).s4(), v(3).s4());
            a.zip2(v(19).s4(), v(1).s4(), v(3).s4());
            [16, 17, 18, 19]
        } else {
            [0, 1, 2, 3]
        };
        if spec.out_bf16 {
            store_partial_bf16(a, &quads, 4, X10, 0, spec.n_tail);
        } else {
            store_partial_f32(a, &quads, X10, 0, spec.n_tail);
        }
        if m + 1 < spec.mr {
            a.add(X15, X15, 64);
            add_imm(a, X10, X10, spec.ldc_bytes as i64, X11);
        }
    }
}
