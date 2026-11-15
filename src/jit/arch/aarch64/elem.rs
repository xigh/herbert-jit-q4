//! The elementwise kernels of a layer, NEON: one blob per `(op, dim)`.
//!
//! bf16 in memory, f32 in registers: a `q` load brings eight bf16, `shll` and
//! `shll2` widen them exactly (a bf16 is the high half of its f32), `bfcvtn`
//! and `bfcvtn2` narrow eight f32 back, rounded to nearest even. The
//! arithmetic between is IEEE single precision, fused where the x86 kernel
//! fuses. `dim` is an immediate (a multiple of 32, `ElemSpec::check`): the
//! body is a loop over blocks of 32 elements, no remainder.
//!
//! The exponential is the polynomial of the x86 kernels (`exp_constants.rs`
//! values, same Horner chain, same clamps), so an activation computes the
//! same function on both targets; only the rounding of the few operations
//! the ISAs order differently can move a last bit.

use crate::core::error::{HerbertError, Result};
use crate::jit::arch::aarch64::util::{backend_err, splat, Frame};
use crate::jit::emit_elem::{ElemOp, ElemSpec};
use a64::*;

pub(crate) use crate::jit::exp_constants::{
    EXP_C0, EXP_C1, EXP_C2, EXP_C3, EXP_C4, EXP_LOG2E, EXP_NEG87, EXP_ONE_BITS,
};
/// Upper bound of the argument where it can be positive (see the x86 `EXP_MAX88`).
pub(crate) const EXP_MAX88: f32 = 88.0;

/// The registers holding the polynomial constants.
#[derive(Clone, Copy)]
pub(crate) struct ExpRegs {
    pub neg87: u8,
    pub log2e: u8,
    pub c4: u8,
    pub c3: u8,
    pub c2: u8,
    pub c1: u8,
    pub c0: u8,
    /// `0x3F80_0000`: the exponent bias of the reconstruction, and 1.0.
    pub one: u8,
}

impl ExpRegs {
    /// The constants in eight consecutive registers from `first`.
    pub(crate) fn at(first: u8) -> Self {
        ExpRegs {
            neg87: first,
            log2e: first + 1,
            c4: first + 2,
            c3: first + 3,
            c2: first + 4,
            c1: first + 5,
            c0: first + 6,
            one: first + 7,
        }
    }
    /// Materialise them, through `tmp`.
    pub(crate) fn load(&self, a: &mut Asm, tmp: R) {
        splat(a, self.neg87, EXP_NEG87.to_bits(), tmp);
        splat(a, self.log2e, EXP_LOG2E.to_bits(), tmp);
        splat(a, self.c4, EXP_C4.to_bits(), tmp);
        splat(a, self.c3, EXP_C3.to_bits(), tmp);
        splat(a, self.c2, EXP_C2.to_bits(), tmp);
        splat(a, self.c1, EXP_C1.to_bits(), tmp);
        splat(a, self.c0, EXP_C0.to_bits(), tmp);
        splat(a, self.one, EXP_ONE_BITS, tmp);
    }
}

/// `io = exp(io)` on four lanes, for `io ≤ 88` (the caller bounds a positive
/// argument). The chain of the x86 `exp_inline`: clamp at −87.34, `2^k·p(f)`
/// with `k` rounded to nearest even and a degree-4 Horner in fused
/// multiply-adds. Four temporaries.
pub(crate) fn vexp(a: &mut Asm, k: &ExpRegs, io: u8, t: [u8; 4]) {
    let [t1, t2, t3, t4] = t;
    let f = |r: u8| v(r).s4();
    a.fmax(f(io), f(io), f(k.neg87));
    a.fmul(f(t1), f(io), f(k.log2e));
    a.frintn(f(t2), f(t1));
    a.fsub(f(t1), f(t1), f(t2));
    // t3 = c4·f + c3, t4 = t3·f + c2, t3 = t4·f + c1, t4 = t3·f + c0.
    a.mov_v(v(t3).b16(), v(k.c3).b16());
    a.fmla(f(t3), f(k.c4), f(t1));
    a.mov_v(v(t4).b16(), v(k.c2).b16());
    a.fmla(f(t4), f(t3), f(t1));
    a.mov_v(v(t3).b16(), v(k.c1).b16());
    a.fmla(f(t3), f(t4), f(t1));
    a.mov_v(v(t4).b16(), v(k.c0).b16());
    a.fmla(f(t4), f(t3), f(t1));
    a.fcvtzs(f(t2), f(t2));
    a.shl(f(t2), f(t2), 23);
    a.add_v(f(t2), f(t2), f(k.one));
    a.fmul(f(io), f(t4), f(t2));
}

/// Widen the low (`high = false`) or high four bf16 of `src` into f32 `dst`.
pub(crate) fn widen(a: &mut Asm, dst: u8, src: u8, high: bool) {
    if high {
        a.shll(v(dst).s4(), v(src).h8());
    } else {
        a.shll(v(dst).s4(), v(src).h4());
    }
}

/// Does this CPU have the BF16 extension (`bfdot`, `bfmlal`, `bfcvtn`)? Without
/// it (a Cortex-A76, say) the kernels widen bf16 to f32 by integer shifts and
/// use `fmla`, and round f32 to bf16 by [`rne_quad`]: the same bits as the BF16
/// path where that path is IEEE (`bfmlal`, `bfcvtn`).
pub(crate) fn has_bf16() -> bool {
    crate::core::isa::feature("bf16")
}

/// Four f32 of `src` rounded to nearest even into four bf16: the low half of
/// `dst`, or its high half when `upper` (the `2` form, which keeps the low
/// half). `tmp` is clobbered; it must differ from `src` and may be `dst`.
///
/// `x` rounds to `(x + 0x7fff + bit16(x)) >> 16`. With `nb = 1 − bit16(x)`,
/// that is `(x − nb + 0x8000) >> 16`, which `rshrn` computes (its sum does not
/// wrap, so `x = 0` gives `0`): five instructions, no constant register.
/// Finite values, zeros, infinities and subnormals come out as `bfcvtn` gives
/// them; a NaN is not made quiet.
pub(crate) fn rne_quad(a: &mut Asm, dst: u8, src: u8, tmp: u8, upper: bool) {
    a.shl(v(tmp).s4(), v(src).s4(), 15);
    a.not_v(v(tmp).b16(), v(tmp).b16());
    a.ushr(v(tmp).s4(), v(tmp).s4(), 31);
    a.sub_v(v(tmp).s4(), v(src).s4(), v(tmp).s4());
    if upper {
        a.rshrn(v(dst).h8(), v(tmp).s4(), 16);
    } else {
        a.rshrn(v(dst).h4(), v(tmp).s4(), 16);
    }
}

/// Four f32 of `src` into four bf16 in the low half of `dst` (the other half
/// zeroed). Without BF16, `tmp` is clobbered; `src` is kept.
pub(crate) fn narrow_quad(a: &mut Asm, dst: u8, src: u8, tmp: u8) {
    if has_bf16() {
        a.bfcvtn(v(dst).h4(), v(src).s4());
    } else {
        rne_quad(a, dst, src, tmp, false);
    }
}

/// Narrow two f32 quads into eight bf16 in `dst`, rounded to nearest even.
/// Without BF16, `lo` is clobbered (and `dst` serves as the scratch of the
/// low quad): every caller is done with `lo` and `hi`.
pub(crate) fn narrow(a: &mut Asm, dst: u8, lo: u8, hi: u8) {
    if has_bf16() {
        a.bfcvtn(v(dst).h4(), v(lo).s4());
        a.bfcvtn(v(dst).h8(), v(hi).s4());
    } else {
        rne_quad(a, dst, lo, dst, false);
        rne_quad(a, dst, hi, lo, true);
    }
}

/// A loop over `blocks` blocks of 32 bf16 whose body is `body(a)`: the
/// pointers advance inside the body (post-indexed loads and stores).
fn blocks32(a: &mut Asm, blocks: usize, counter: R, mut body: impl FnMut(&mut Asm)) {
    let top = a.label();
    a.mov_imm(counter, blocks as u64);
    a.bind(top);
    body(a);
    a.subs(counter, counter, 1);
    a.bne(top);
}

/// Generates the kernel of `spec`.
pub fn emit(spec: &ElemSpec) -> Result<Vec<u8>> {
    spec.check_whole()
        .map_err(|d| HerbertError::Backend(format!("backend_jit elem: {d}")))?;
    let nb = spec.blocks32();
    let mut a = Asm::new();
    match spec.op {
        // ── y = x · rsqrt(Σx²/d + ε) · w   (· (1+w) in the Gemma variant) ──
        // x0 = x, x1 = w, x2 = y, s0 = eps.
        ElemOp::RmsNorm | ElemOp::RmsNormGemma => {
            let plus_one = spec.op == ElemOp::RmsNormGemma;
            a.fmov(s(31), s(0));
            // Pass 1: Σx² in four chains of fused products of the widened x.
            for c in 0..4 {
                a.movi(v(c).d2(), 0);
            }
            a.mov(X9, X0);
            blocks32(&mut a, nb, X10, |a| {
                a.ldp(q(16), q(17), post(X9, 32));
                a.ldp(q(18), q(19), post(X9, 32));
                for (i, src) in [16u8, 17, 18, 19].into_iter().enumerate() {
                    widen(a, 20, src, false);
                    widen(a, 21, src, true);
                    let c = (2 * i as u8) % 4;
                    a.fmla(v(c).s4(), v(20).s4(), v(20).s4());
                    a.fmla(v(c + 1).s4(), v(21).s4(), v(21).s4());
                }
            });
            a.fadd(v(0).s4(), v(0).s4(), v(1).s4());
            a.fadd(v(2).s4(), v(2).s4(), v(3).s4());
            a.fadd(v(0).s4(), v(0).s4(), v(2).s4());
            a.faddp(v(0).s4(), v(0).s4(), Some(v(0).s4()));
            a.faddp(s(0), v(0).s2(), None);
            // inv = 1 / sqrt(Σx² · (1/dim) + eps), in scalar.
            a.mov_imm(W9, u64::from((1.0f32 / spec.dim as f32).to_bits()));
            a.fmov_from_gp(s(1), W9);
            a.fmul(s(0), s(0), s(1));
            a.fadd(s(0), s(0), s(31));
            a.fsqrt(s(0), s(0));
            a.fmov_imm(s(1), 1.0);
            a.fdiv(s(0), s(1), s(0));
            a.dup(v(30).s4(), v(0).s(0));
            if plus_one {
                a.fmov_imm(v(29).s4(), 1.0);
            }
            // Pass 2: y = (x · inv) · w, eight at a time.
            blocks32(&mut a, nb, X10, |a| {
                a.ldp(q(16), q(17), post(X0, 32));
                a.ldp(q(18), q(19), post(X0, 32));
                a.ldp(q(20), q(21), post(X1, 32));
                a.ldp(q(22), q(23), post(X1, 32));
                for (xb, wb, out) in [(16u8, 20u8, 24u8), (17, 21, 25), (18, 22, 26), (19, 23, 27)]
                {
                    for (half, dst) in [(false, 2u8), (true, 3u8)] {
                        widen(a, dst, xb, half);
                        widen(a, dst + 2, wb, half);
                        if plus_one {
                            a.fadd(v(dst + 2).s4(), v(dst + 2).s4(), v(29).s4());
                        }
                        a.fmul(v(dst).s4(), v(dst).s4(), v(30).s4());
                        a.fmul(v(dst).s4(), v(dst).s4(), v(dst + 2).s4());
                    }
                    narrow(a, out, 2, 3);
                }
                a.stp(q(24), q(25), post(X2, 32));
                a.stp(q(26), q(27), post(X2, 32));
            });
        }

        // ── y = gate · σ(gate) · up, and its two variants ────────────────
        // x0 = gate, x1 = up, x2 = y. SwiGLU: σ(g); geluTanh: σ(g·(K + K·C·g²));
        // SigmoidMul: y = a · σ(b), σ of the SECOND operand, no final product.
        ElemOp::SwiGlu | ElemOp::GeluTanh | ElemOp::SigmoidMul => {
            let cubic = spec.op == ElemOp::GeluTanh;
            let gate_out = spec.op == ElemOp::SigmoidMul;
            let k = ExpRegs::at(16);
            k.load(&mut a, X9);
            splat(&mut a, 24, EXP_MAX88.to_bits(), X9);
            if cubic {
                const K: f32 = 1.595_769;
                const C: f32 = 0.044_715;
                splat(&mut a, 25, K.to_bits(), X9);
                splat(&mut a, 26, (K * C).to_bits(), X9);
            }
            blocks32(&mut a, nb, X10, |a| {
                for _ in 0..4 {
                    a.ldr(q(0), post(X0, 16));
                    a.ldr(q(1), post(X1, 16));
                    for (half, out) in [(false, 28u8), (true, 29u8)] {
                        widen(a, 2, 0, half);
                        widen(a, 3, 1, half);
                        // v4 ← the sigmoid argument, negated.
                        if gate_out {
                            a.fneg(v(4).s4(), v(3).s4());
                        } else if cubic {
                            a.fmul(v(4).s4(), v(2).s4(), v(2).s4());
                            a.fmul(v(4).s4(), v(4).s4(), v(26).s4());
                            a.fadd(v(4).s4(), v(4).s4(), v(25).s4());
                            a.fmul(v(4).s4(), v(4).s4(), v(2).s4());
                            a.fneg(v(4).s4(), v(4).s4());
                        } else {
                            a.fneg(v(4).s4(), v(2).s4());
                        }
                        a.fmin(v(4).s4(), v(4).s4(), v(24).s4());
                        vexp(a, &k, 4, [5, 6, 7, 27]);
                        a.fadd(v(4).s4(), v(4).s4(), v(k.one).s4());
                        a.fdiv(v(out).s4(), v(2).s4(), v(4).s4());
                        if !gate_out {
                            a.fmul(v(out).s4(), v(out).s4(), v(3).s4());
                        }
                    }
                    narrow(a, 30, 28, 29);
                    a.str(q(30), post(X2, 16));
                }
            });
        }

        // ── RoPE, rotate_half convention ────────────────────────────────
        // x0 = x (read and written), x1 = cos, x2 = sin; pairs (i, i + dim/2).
        ElemOp::Rope => {
            let half = spec.dim / 2;
            a.add(X3, X0, (half * 2) as u64);
            for _ in (0..half).step_by(8) {
                a.ldr(q(0), at(X0));
                a.ldr(q(1), at(X3));
                a.ldr(q(2), post(X1, 16));
                a.ldr(q(3), post(X2, 16));
                for (hi, (ra, rb)) in [(false, (24u8, 25u8)), (true, (26, 27))] {
                    widen(&mut a, 4, 0, hi);
                    widen(&mut a, 5, 1, hi);
                    widen(&mut a, 6, 2, hi);
                    widen(&mut a, 7, 3, hi);
                    // a' = a·c − b·s ; b' = a·s + b·c
                    a.fmul(v(ra).s4(), v(4).s4(), v(6).s4());
                    a.fmls(v(ra).s4(), v(5).s4(), v(7).s4());
                    a.fmul(v(rb).s4(), v(4).s4(), v(7).s4());
                    a.fmla(v(rb).s4(), v(5).s4(), v(6).s4());
                }
                narrow(&mut a, 28, 24, 26);
                narrow(&mut a, 29, 25, 27);
                a.str(q(28), post(X0, 16));
                a.str(q(29), post(X3, 16));
            }
        }

        // ── a += b ──────────────────────────────────────────────────────
        ElemOp::ResidualAdd => {
            blocks32(&mut a, nb, X10, |a| {
                a.ldp(q(16), q(17), mem(X0, 0));
                a.ldp(q(18), q(19), mem(X0, 32));
                a.ldp(q(20), q(21), post(X1, 32));
                a.ldp(q(22), q(23), post(X1, 32));
                for (ab, bb, out) in [(16u8, 20u8, 24u8), (17, 21, 25), (18, 22, 26), (19, 23, 27)]
                {
                    for (half, dst) in [(false, 2u8), (true, 3u8)] {
                        widen(a, dst, ab, half);
                        widen(a, dst + 2, bb, half);
                        a.fadd(v(dst).s4(), v(dst).s4(), v(dst + 2).s4());
                    }
                    narrow(a, out, 2, 3);
                }
                a.stp(q(24), q(25), post(X0, 32));
                a.stp(q(26), q(27), post(X0, 32));
            });
        }

        // ── y = x · s ───────────────────────────────────────────────────
        ElemOp::Scale => {
            a.dup(v(31).s4(), v(0).s(0));
            blocks32(&mut a, nb, X10, |a| {
                a.ldp(q(16), q(17), post(X0, 32));
                a.ldp(q(18), q(19), post(X0, 32));
                for (xb, out) in [(16u8, 24u8), (17, 25), (18, 26), (19, 27)] {
                    for (half, dst) in [(false, 2u8), (true, 3u8)] {
                        widen(a, dst, xb, half);
                        a.fmul(v(dst).s4(), v(dst).s4(), v(31).s4());
                    }
                    narrow(a, out, 2, 3);
                }
                a.stp(q(24), q(25), post(X1, 32));
                a.stp(q(26), q(27), post(X1, 32));
            });
        }

        // ── y = cap · tanh(x / cap) ─────────────────────────────────────
        // `tanh(u) = (1 − e^{−2u})/(1 + e^{−2u})` on `|u|`, the sign restored.
        ElemOp::Softcap => {
            a.dup(v(31).s4(), v(0).s(0));
            let k = ExpRegs::at(16);
            k.load(&mut a, X9);
            splat(&mut a, 24, (-2.0f32).to_bits(), X9);
            a.fdiv(v(25).s4(), v(24).s4(), v(31).s4());
            a.movi(v(26).s4(), 0x8000_0000);
            blocks32(&mut a, nb, X10, |a| {
                for _ in 0..4 {
                    a.ldr(q(0), post(X0, 16));
                    for (half, out) in [(false, 28u8), (true, 29u8)] {
                        widen(a, 1, 0, half);
                        a.and_v(v(2).b16(), v(1).b16(), v(26).b16());
                        a.fabs(v(1).s4(), v(1).s4());
                        a.fmul(v(1).s4(), v(1).s4(), v(25).s4());
                        vexp(a, &k, 1, [3, 4, 5, 6]);
                        a.fsub(v(3).s4(), v(k.one).s4(), v(1).s4());
                        a.fadd(v(4).s4(), v(k.one).s4(), v(1).s4());
                        a.fdiv(v(1).s4(), v(3).s4(), v(4).s4());
                        a.fmul(v(1).s4(), v(1).s4(), v(31).s4());
                        a.orr_v(v(out).b16(), v(1).b16(), v(2).b16());
                    }
                    narrow(a, 30, 28, 29);
                    a.str(q(30), post(X1, 16));
                }
            });
        }

        // ── y = bf16(Σᵣ x[r] · w[r]) ─────────────────────────────────────
        // x0 = x (`taps` rows of `dim`), x1 = w (`[taps][dim]`), x2 = y. The
        // sums of a block of 32 in v20..v27, its rows and kernel rows read at
        // their offsets, x0 and x1 moved by one block after them.
        ElemOp::DepthwiseRows(taps) => {
            let row = spec.dim * 2;
            blocks32(&mut a, nb, X10, |a| {
                for acc in 20u8..28 {
                    a.movi(v(acc).s4(), 0);
                }
                for r in 0..usize::from(taps) {
                    for qd in 0..4u8 {
                        a.ldr(q(16 + qd), mem(X0, (r * row + usize::from(qd) * 16) as i64));
                        a.ldr(q(4 + qd), mem(X1, (r * row + usize::from(qd) * 16) as i64));
                    }
                    for qd in 0..4u8 {
                        for (h, high) in [(0u8, false), (1, true)] {
                            widen(a, 2, 16 + qd, high);
                            widen(a, 3, 4 + qd, high);
                            a.fmla(v(20 + 2 * qd + h).s4(), v(2).s4(), v(3).s4());
                        }
                    }
                }
                a.add(X0, X0, 64);
                a.add(X1, X1, 64);
                for qd in 0..4u8 {
                    narrow(a, 16 + qd, 20 + 2 * qd, 21 + 2 * qd);
                }
                a.stp(q(16), q(17), post(X2, 32));
                a.stp(q(18), q(19), post(X2, 32));
            });
        }
        // Its own emitter (`local_attn.rs`).
        ElemOp::LocalAttention { .. } => return crate::jit::arch::aarch64::local_attn::emit(spec),
        // Not ported yet: the vision tower of 26B-A4B (`Standardize`) runs on AVX-512.
        ElemOp::Standardize => {
            return Err(HerbertError::Backend(format!(
                "aarch64: elem {} is not emitted",
                spec.op.name()
            )))
        }
        // ── y = (x − μ) · rsqrt(σ² + ε) · g + b ─────────────────────────
        // x0 = x, x1 = w (`g` then `b`), x2 = y, s0 = eps. Three passes over the
        // row: Σx, Σ(x − μ)², the output; μ in v30, 1/σ in v29, eps in s31.
        ElemOp::LayerNorm => {
            // v8..v15 hold `b` and the widened values: their low halves (d8..d15) are the caller's,
            // saved here.
            let frame = Frame::enter(&mut a, false, &[], true, 0);
            a.fmov(s(31), s(0));
            let sum = |a: &mut Asm, centred: bool| {
                for c in 0..4 {
                    a.movi(v(c).d2(), 0);
                }
                a.mov(X9, X0);
                blocks32(a, nb, X10, |a| {
                    a.ldp(q(16), q(17), post(X9, 32));
                    a.ldp(q(18), q(19), post(X9, 32));
                    for (i, src) in [16u8, 17, 18, 19].into_iter().enumerate() {
                        widen(a, 20, src, false);
                        widen(a, 21, src, true);
                        let c = (2 * i as u8) % 4;
                        if centred {
                            a.fsub(v(20).s4(), v(20).s4(), v(30).s4());
                            a.fsub(v(21).s4(), v(21).s4(), v(30).s4());
                            a.fmla(v(c).s4(), v(20).s4(), v(20).s4());
                            a.fmla(v(c + 1).s4(), v(21).s4(), v(21).s4());
                        } else {
                            a.fadd(v(c).s4(), v(c).s4(), v(20).s4());
                            a.fadd(v(c + 1).s4(), v(c + 1).s4(), v(21).s4());
                        }
                    }
                });
                a.fadd(v(0).s4(), v(0).s4(), v(1).s4());
                a.fadd(v(2).s4(), v(2).s4(), v(3).s4());
                a.fadd(v(0).s4(), v(0).s4(), v(2).s4());
                a.faddp(v(0).s4(), v(0).s4(), Some(v(0).s4()));
                a.faddp(s(0), v(0).s2(), None);
                a.mov_imm(W9, u64::from((1.0f32 / spec.dim as f32).to_bits()));
                a.fmov_from_gp(s(1), W9);
                a.fmul(s(0), s(0), s(1));
            };
            sum(&mut a, false);
            a.dup(v(30).s4(), v(0).s(0));
            sum(&mut a, true);
            a.fadd(s(0), s(0), s(31));
            a.fsqrt(s(0), s(0));
            a.fmov_imm(s(1), 1.0);
            a.fdiv(s(0), s(1), s(0));
            a.dup(v(29).s4(), v(0).s(0));
            // Pass 3: y = (x − μ) · inv · g + b; `b` from x3 (the offset through x9 past an `add`'s
            // 12-bit immediate, for rows of 2 048 values and more).
            if spec.dim * 2 < 4096 {
                a.add(X3, X1, (spec.dim * 2) as u64);
            } else {
                a.mov_imm(X9, (spec.dim * 2) as u64);
                a.add(X3, X1, X9);
            }
            blocks32(&mut a, nb, X10, |a| {
                a.ldp(q(16), q(17), post(X0, 32));
                a.ldp(q(18), q(19), post(X0, 32));
                a.ldp(q(20), q(21), post(X1, 32));
                a.ldp(q(22), q(23), post(X1, 32));
                a.ldp(q(8), q(9), post(X3, 32));
                a.ldp(q(10), q(11), post(X3, 32));
                for (xb, gb, bb, out) in [
                    (16u8, 20u8, 8u8, 24u8),
                    (17, 21, 9, 25),
                    (18, 22, 10, 26),
                    (19, 23, 11, 27),
                ] {
                    for (half, x, g, b) in [(false, 2u8, 12u8, 14u8), (true, 3, 13, 15)] {
                        widen(a, x, xb, half);
                        widen(a, g, gb, half);
                        widen(a, b, bb, half);
                        a.fsub(v(x).s4(), v(x).s4(), v(30).s4());
                        a.fmul(v(x).s4(), v(x).s4(), v(29).s4());
                        a.fmla(v(b).s4(), v(x).s4(), v(g).s4());
                    }
                    narrow(a, out, 14, 15);
                }
                a.stp(q(24), q(25), post(X2, 32));
                a.stp(q(26), q(27), post(X2, 32));
            });
            frame.leave(&mut a);
        }

        // ── y = gelu(gate) · up, the exact GeLU ─────────────────────────
        // x0 = gate, x1 = up, x2 = y. The `erfcc` of the x86 kernels: the
        // exponential's constants in v16..v23, `P2..P9` in v24..v31, `[P0, P1,
        // ½, 1/√2]` in the lanes of v15; `½·E` or `1 − ½·E` chosen by `bsl`.
        ElemOp::GeluErf => {
            const P: [f32; 10] = [
                -1.265_512_2,
                1.000_023_7,
                0.374_091_96,
                0.096_784_18,
                -0.186_288_06,
                0.278_868_07,
                -1.135_204,
                1.488_515_9,
                -0.822_152_23,
                0.170_872_77,
            ];
            // Every vector register: d8..d15 are the caller's, saved here.
            let frame = Frame::enter(&mut a, false, &[], true, 0);
            let k = ExpRegs::at(16);
            k.load(&mut a, X9);
            for (i, c) in P.iter().enumerate().skip(2) {
                splat(&mut a, 22 + i as u8, c.to_bits(), X9);
            }
            for (lane, c) in [P[0], P[1], 0.5, std::f32::consts::FRAC_1_SQRT_2]
                .into_iter()
                .enumerate()
            {
                a.mov_imm(W9, u64::from(c.to_bits()));
                a.ins_gp(v(15).s(lane as u8), W9);
            }
            blocks32(&mut a, nb, X10, |a| {
                for _ in 0..4 {
                    a.ldr(q(0), post(X0, 16));
                    a.ldr(q(1), post(X1, 16));
                    for (half, res) in [(false, 12u8), (true, 13u8)] {
                        let f = |r: u8| v(r).s4();
                        widen(a, 2, 0, half);
                        // a = |g|/√2 (v4), t = 1/(1 + ½·a) (v5).
                        a.fabs(f(4), f(2));
                        a.dup(f(7), v(15).s(3));
                        a.fmul(f(4), f(4), f(7));
                        a.dup(f(7), v(15).s(2));
                        a.mov_v(v(5).b16(), v(k.one).b16());
                        a.fmla(f(5), f(4), f(7));
                        a.fdiv(f(5), f(k.one), f(5));
                        // P(t) by Horner, alternating between v6 and v7, from P9; then − a².
                        a.mov_v(v(6).b16(), v(31).b16());
                        let mut cur = 6u8;
                        for i in (0..9).rev() {
                            let next = if cur == 6 { 7 } else { 6 };
                            if i >= 2 {
                                a.mov_v(v(next).b16(), v(22 + i as u8).b16());
                            } else {
                                a.dup(f(next), v(15).s(i as u8));
                            }
                            a.fmla(f(next), f(cur), f(5));
                            cur = next;
                        }
                        a.fmls(f(cur), f(4), f(4));
                        vexp(a, &k, cur, [8, 9, 10, 11]);
                        // E = t·exp(…); Φ = ½·E where g < 0 (`½ − ½·(1 − E)` would lose it to
                        // cancellation), 1 − ½·E elsewhere.
                        a.fmul(f(cur), f(cur), f(5));
                        let other = if cur == 6 { 7 } else { 6 };
                        a.dup(f(4), v(15).s(2));
                        a.fmul(f(cur), f(cur), f(4));
                        a.fsub(f(other), f(k.one), f(cur));
                        a.fcmge_zero(f(14), f(2));
                        a.bsl(v(14).b16(), v(other).b16(), v(cur).b16());
                        a.fmul(f(res), f(14), f(2));
                        widen(a, 3, 1, half);
                        a.fmul(f(res), f(res), f(3));
                    }
                    narrow(a, 0, 12, 13);
                    a.str(q(0), post(X2, 16));
                }
            });
            frame.leave(&mut a);
        }

        // ── y = min(max(x, lo), hi) ─────────────────────────────────────
        // x0 = x, x1 = y, s0 = lo, s1 = hi.
        ElemOp::Clamp => {
            a.dup(v(30).s4(), v(0).s(0));
            a.dup(v(31).s4(), v(1).s(0));
            blocks32(&mut a, nb, X10, |a| {
                a.ldp(q(16), q(17), post(X0, 32));
                a.ldp(q(18), q(19), post(X0, 32));
                for (xb, out) in [(16u8, 24u8), (17, 25), (18, 26), (19, 27)] {
                    for (half, dst) in [(false, 2u8), (true, 3u8)] {
                        widen(a, dst, xb, half);
                        a.fmax(v(dst).s4(), v(dst).s4(), v(30).s4());
                        a.fmin(v(dst).s4(), v(dst).s4(), v(31).s4());
                    }
                    narrow(a, out, 2, 3);
                }
                a.stp(q(24), q(25), post(X1, 32));
                a.stp(q(26), q(27), post(X1, 32));
            });
        }

        // ── y = bf16(Σᵣ x[r] · inv) · s ─────────────────────────────────
        // x0 = x (`rows` rows of `dim`), x1 = y, s0 = s, s1 = inv. The sums of a
        // block of 32 in v20..v27, its rows read at their offsets, x0 moved by
        // one block after them.
        ElemOp::PoolRows(rows) => {
            a.dup(v(30).s4(), v(0).s(0));
            a.dup(v(31).s4(), v(1).s(0));
            let row = spec.dim * 2;
            blocks32(&mut a, nb, X10, |a| {
                for acc in 20u8..28 {
                    a.movi(v(acc).s4(), 0);
                }
                for r in 0..usize::from(rows) {
                    for qd in 0..4u8 {
                        a.ldr(q(16 + qd), mem(X0, (r * row + usize::from(qd) * 16) as i64));
                    }
                    for qd in 0..4u8 {
                        for (h, high) in [(0u8, false), (1, true)] {
                            widen(a, 2, 16 + qd, high);
                            a.fmla(v(20 + 2 * qd + h).s4(), v(2).s4(), v(31).s4());
                        }
                    }
                }
                a.add(X0, X0, 64);
                for qd in 0..4u8 {
                    for h in 0..2u8 {
                        // The mean rounded to bf16, widened back, times `s`.
                        let acc = 20 + 2 * qd + h;
                        narrow_quad(a, 4, acc, 5);
                        widen(a, acc, 4, false);
                        a.fmul(v(acc).s4(), v(acc).s4(), v(30).s4());
                    }
                    narrow(a, 16 + qd, 20 + 2 * qd, 21 + 2 * qd);
                }
                a.stp(q(16), q(17), post(X1, 32));
                a.stp(q(18), q(19), post(X1, 32));
            });
        }
    }
    a.ret();
    a.finish()
        .map(|c| c.bytes)
        .map_err(backend_err(format_args!(
            "elem {} dim={}",
            spec.op.name(),
            spec.dim
        )))
}
