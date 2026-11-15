//! The kernels of the mixture-of-experts blocks of Gemma 4 (the 26B-A4B), of `qwen3_5_moe`
//! (Qwen3.6-35B-A3B) and of LiquidAI's LFM2 MoE, AArch64.
//!
//! * `route`: the router scores (bf16) of one row → the `top_k` best experts (largest score first,
//!   the smaller index on a tie), the softmax of those scores times `per_expert_scale`, and the
//!   pointers of the weights of those experts written into the call structs of the nodes that
//!   multiply by them. The experts are known only here, at run time: the nodes of the expert GEMMs
//!   read the weight pointer from their call struct when they run, after this node.
//! * `combine`: `out = Σ_k w_k · row_k`, the `top_k` rows of expert outputs (bf16) summed with
//!   their weights (f32), rounded to bf16 once.
//! * `route_with(.., RouteRule::Qwen)`: Qwen's router, the weights of `RouteRule::Qwen` (the slots
//!   in increasing expert order), the same pointers written.
//! * `route_rows`: the router of a block of rows (`abi::RouteRowsCall`), each row as `route_with`
//!   does it, then the rows grouped by expert, one GEMM per expert over its rows (the prefill of
//!   `qwen3_5_moe`).
//! * `combine_rows`: `combine` for one row of a block whose expert outputs were grouped by expert
//!   (`abi::CombineRowsCall`), the end of Gemma's grouped prefill.
//! * `qwen_out`: the end of Qwen's sparse block and the residual of one row (`abi::QwenOutCall`),
//!   every step rounded to bf16 as the x86-64 kernel of the same name does.
use super::{ExecutionError, GeneratedKernel};
use crate::jit::arch::aarch64::elem::{has_bf16, narrow};
use crate::jit::arch::aarch64::util::Frame;
use crate::jit::arch::aarch64::util::{add_imm, exec_err};
use crate::jit::execution::abi::Bf16Call;
pub use crate::jit::execution::abi::{
    CallFrame, CombineCall, CombineRowsCall, QwenOutCall, RouteCall, RouteRowsCall, RouteRowsSpec,
    RouteRule, MAX_TOP_K,
};
use a64::*;
use std::mem::offset_of;

/// `exp(x)` for `x <= 0` in `s0`, result in `s0`; clobbers `s1..s4`, `w14`, `w15` and uses
/// `s16..s23` as constants.
pub(crate) fn emit_exp(a: &mut Asm) {
    a.fmax(s(0), s(0), s(16)); // clamp at -87
    a.fmul(s(1), s(0), s(17)); // t = x · log2(e)
    a.frintm(s(2), s(1)); // n = floor(t)
    a.fsub(s(3), s(1), s(2)); // f in [0, 1)
    a.fmov(s(4), s(18)); // c5
    a.fmadd(s(4), s(4), s(3), s(19)); // c5·f + c4
    a.fmadd(s(4), s(4), s(3), s(20)); // · f + c3
    a.fmadd(s(4), s(4), s(3), s(21)); // · f + c2
    a.fmadd(s(4), s(4), s(3), s(22)); // · f + c1
    a.fmadd(s(4), s(4), s(3), s(23)); // · f + 1
    a.fcvtzs_gp(W14, s(2));
    a.lsl(W14, W14, 23);
    a.fmov_to_gp(W15, s(4));
    a.add(W15, W15, W14);
    a.fmov_from_gp(s(0), W15);
}

/// `s(r)` rounded to bf16 (nearest even), kept as f32 (the other lanes of `v(r)` are clobbered);
/// clobbers `v(t)`, and `w14`, `w15` without the BF16 extension.
pub(crate) fn round_bf16_s(a: &mut Asm, r: u8, t: u8) {
    if has_bf16() {
        a.bfcvt(h(t), s(r));
        a.shll(v(r).s4(), v(t).h4());
    } else {
        a.fmov_to_gp(W14, s(r));
        a.lsr(W15, W14, 16);
        a.and(W15, W15, 1u64);
        a.add(W14, W14, W15);
        a.mov_imm(W15, 0x7fff);
        a.add(W14, W14, W15);
        a.lsr(W14, W14, 16);
        a.lsl(W14, W14, 16);
        a.fmov_from_gp(s(r), W14);
    }
}

/// The four f32 of `v(r)` rounded to bf16 (nearest even), kept as f32; clobbers `v(t)`. Without the
/// BF16 extension, `v25` holds 1 and `v26` holds 0x7fff in every lane.
fn round_bf16_quad(a: &mut Asm, r: u8, t: u8) {
    if has_bf16() {
        a.bfcvtn(v(t).h4(), v(r).s4());
        a.shll(v(r).s4(), v(t).h4());
    } else {
        a.ushr(v(t).s4(), v(r).s4(), 16);
        a.and_v(v(t).b16(), v(t).b16(), v(25).b16());
        a.add_v(v(r).s4(), v(r).s4(), v(t).s4());
        a.add_v(v(r).s4(), v(r).s4(), v(26).s4());
        a.ushr(v(r).s4(), v(r).s4(), 16);
        a.shl(v(r).s4(), v(r).s4(), 16);
    }
}

/// The weights of [`RouteRule::Qwen`], once the selection has left the `top_k` best scores (best
/// first) at `val_off`, their experts at `idx_off` and `-inf` in their places among the f32 scores
/// at `sp`: `e_k = exp(s_k − s_0)` (`s_0` the largest), `w_k = e_k / Σ_k e_k` (`norm`) or `e_k /
/// Σ_experts e` (the chosen ones count in the sum by their `e_k`, the `-inf` left in their places
/// by `exp(-87)`), each rounded to bf16; then the `top_k` pairs sorted by expert, increasing. The
/// order of every operation is that of the x86-64 kernel. Exponential constants in `s16..s23`;
/// clobbers `v0..v4`, `s24..s26`, `x9..x15`.
fn emit_qwen_weights(
    a: &mut Asm,
    experts: usize,
    top_k: usize,
    norm: bool,
    idx_off: usize,
    val_off: usize,
) {
    a.ldr(s(24), mem(SP, val_off as i64)); // max
    a.mov_imm(W9, 0);
    a.fmov_from_gp(s(25), W9); // sum = 0
    for k in 0..top_k {
        a.ldr(s(0), mem(SP, (val_off + 4 * k) as i64));
        a.fsub(s(0), s(0), s(24));
        emit_exp(a);
        a.str(s(0), mem(SP, (val_off + 4 * k) as i64));
        a.fadd(s(25), s(25), s(0));
    }
    if !norm {
        a.mov_imm(W9, 0);
        a.fmov_from_gp(s(26), W9);
        a.mov(X10, SP);
        a.mov_imm(X13, experts as u64);
        let all = a.label();
        a.bind(all);
        a.ldr(s(0), post(X10, 4));
        a.fsub(s(0), s(0), s(24));
        emit_exp(a);
        a.fadd(s(26), s(26), s(0));
        a.subs(X13, X13, 1);
        a.bne(all);
        a.fadd(s(25), s(25), s(26));
    }
    for k in 0..top_k {
        a.ldr(s(0), mem(SP, (val_off + 4 * k) as i64));
        a.fdiv(s(0), s(0), s(25));
        round_bf16_s(a, 0, 1);
        a.str(s(0), mem(SP, (val_off + 4 * k) as i64));
    }
    // Sorted by expert: a bubble network over the `top_k` pairs, unrolled.
    for i in 0..top_k.saturating_sub(1) {
        for j in 0..top_k - 1 - i {
            let keep = a.label();
            let (ij, ij1) = ((idx_off + 8 * j) as i64, (idx_off + 8 * (j + 1)) as i64);
            let (vj, vj1) = ((val_off + 4 * j) as i64, (val_off + 4 * (j + 1)) as i64);
            a.ldr(X10, mem(SP, ij));
            a.ldr(X11, mem(SP, ij1));
            a.cmp(X10, X11);
            a.b_cond(Cond::Ls, keep);
            a.str(X11, mem(SP, ij));
            a.str(X10, mem(SP, ij1));
            a.ldr(W12, mem(SP, vj));
            a.ldr(W13, mem(SP, vj1));
            a.str(W13, mem(SP, vj));
            a.str(W12, mem(SP, vj1));
            a.bind(keep);
        }
    }
}

/// [`RouteRule::Lfm`], before the selection: the `experts` f32 scores at `sp` become `s =
/// bf16(σ(score))`, kept at `sp + sig_off`, and in their places `s + bias[e]` (the f32 biases at
/// `x9`), which the selection ranks. `σ(x)` is `1 / (1 + e)` for `x ≥ 0` and `e / (1 + e)` below,
/// `e = exp(−|x|)`; the order of the x86-64 kernel. Exponential constants in `s16..s23`; clobbers
/// `v0..v7`, `x9..x13`, `w14`, `w15`.
fn emit_lfm_scores(a: &mut Asm, experts: usize, sig_off: usize) {
    a.mov(X10, SP);
    add_imm(a, X11, SP, sig_off as i64, X13);
    a.mov(X12, X9);
    a.mov_imm(X13, experts as u64);
    let each = a.label();
    a.bind(each);
    let negative = a.label();
    let done = a.label();
    a.ldr(s(6), at(X10));
    a.fmov_to_gp(W9, s(6));
    a.orr(W9, W9, 0x8000_0000u64);
    a.fmov_from_gp(s(0), W9);
    emit_exp(a);
    a.fadd(s(7), s(0), s(23));
    a.fmov_to_gp(W9, s(6));
    a.tbnz(W9, 31, negative);
    a.fdiv(s(0), s(23), s(7));
    a.b(done);
    a.bind(negative);
    a.fdiv(s(0), s(0), s(7));
    a.bind(done);
    round_bf16_s(a, 0, 1);
    a.str(s(0), post(X11, 4));
    a.ldr(s(1), post(X12, 4));
    a.fadd(s(0), s(0), s(1));
    a.str(s(0), post(X10, 4));
    a.subs(X13, X13, 1);
    a.bne(each);
}

/// The weights of [`RouteRule::Lfm`], once the selection has left the `top_k` best choices (best
/// first) at `val_off` and their experts at `idx_off`: `w_k = s_k` (the rounded sigmoid at `sp +
/// sig_off`), divided by `bf16(bf16(Σ_k s_k) + 1e-6)` when `norm` and rounded, times `scale` and
/// rounded unless it is one; then the `top_k` pairs sorted by expert, increasing. The order of the
/// x86-64 kernel. Clobbers `v0`, `v1`, `v6`, `v7`, `x9..x15`.
fn emit_lfm_weights(
    a: &mut Asm,
    top_k: usize,
    norm: bool,
    scale: f32,
    idx_off: usize,
    val_off: usize,
    sig_off: usize,
) {
    a.mov_imm(W9, 0);
    a.fmov_from_gp(s(7), W9);
    add_imm(a, X11, SP, sig_off as i64, X10);
    for k in 0..top_k {
        a.ldr(X12, mem(SP, (idx_off + 8 * k) as i64));
        a.ldr(s(0), idx_scaled(X11, X12));
        a.str(s(0), mem(SP, (val_off + 4 * k) as i64));
        a.fadd(s(7), s(7), s(0));
    }
    if norm {
        a.fmov(s(0), s(7));
        round_bf16_s(a, 0, 1);
        a.mov_imm(W9, u64::from(1e-6f32.to_bits()));
        a.fmov_from_gp(s(6), W9);
        a.fadd(s(0), s(0), s(6));
        round_bf16_s(a, 0, 1);
        a.fmov(s(7), s(0));
    }
    for k in 0..top_k {
        a.ldr(s(0), mem(SP, (val_off + 4 * k) as i64));
        if norm {
            a.fdiv(s(0), s(0), s(7));
            round_bf16_s(a, 0, 1);
        }
        if scale != 1.0 {
            a.mov_imm(W9, u64::from(scale.to_bits()));
            a.fmov_from_gp(s(6), W9);
            a.fmul(s(0), s(0), s(6));
            round_bf16_s(a, 0, 1);
        }
        a.str(s(0), mem(SP, (val_off + 4 * k) as i64));
    }
    emit_sort_by_expert(a, top_k, idx_off, val_off);
}

/// The `top_k` (expert, weight) pairs at `idx_off` and `val_off` sorted by expert, increasing: a
/// bubble network, unrolled. Clobbers `x10..x13`.
fn emit_sort_by_expert(a: &mut Asm, top_k: usize, idx_off: usize, val_off: usize) {
    for i in 0..top_k.saturating_sub(1) {
        for j in 0..top_k - 1 - i {
            let keep = a.label();
            let (ij, ij1) = ((idx_off + 8 * j) as i64, (idx_off + 8 * (j + 1)) as i64);
            let (vj, vj1) = ((val_off + 4 * j) as i64, (val_off + 4 * (j + 1)) as i64);
            a.ldr(X10, mem(SP, ij));
            a.ldr(X11, mem(SP, ij1));
            a.cmp(X10, X11);
            a.b_cond(Cond::Ls, keep);
            a.str(X11, mem(SP, ij));
            a.str(X10, mem(SP, ij1));
            a.ldr(W12, mem(SP, vj));
            a.ldr(W13, mem(SP, vj1));
            a.str(W13, mem(SP, vj));
            a.str(W12, mem(SP, vj1));
            a.bind(keep);
        }
    }
}

/// Where [`RouteRule::Lfm`] keeps its sigmoid scores on the stack (after the selection's values),
/// and the frame it then needs; the other rules keep their frame.
fn lfm_frame(
    rule: RouteRule,
    experts: usize,
    val_off: usize,
    frame_bytes: usize,
) -> (usize, usize) {
    match rule {
        RouteRule::Lfm { .. } => {
            let sig_off = (val_off + 4 * MAX_TOP_K).next_multiple_of(16);
            (sig_off, (sig_off + experts * 4).next_multiple_of(16))
        }
        _ => (0, frame_bytes),
    }
}

/// The constants of the exponential: s16 = -87, s17 = log2 e, s18..s22 = c5..c1, s23 = 1; clobbers
/// `w9`.
pub(crate) fn emit_exp_constants(a: &mut Asm) {
    for (reg, value) in [
        (16u8, -87.0f32),
        (17, std::f32::consts::LOG2_E),
        (18, 0.001_333_355_8),
        (19, 0.009_618_129),
        (20, 0.055_504_11),
        (21, 0.240_226_5),
        (22, std::f32::consts::LN_2),
        (23, 1.0),
    ] {
        a.mov_imm(W9, u64::from(value.to_bits()));
        a.fmov_from_gp(s(reg), W9);
    }
}

/// Selection among the `experts` f32 scores at `sp`: `top_k` passes of "first maximum" (the smaller
/// index on a tie), the winner replaced by -inf; the indices (8 bytes each) at `sp + idx_off`, the
/// scores at `sp + val_off`. Clobbers `w8`, `x10`, `x12..x14`, `s0`, `s1`.
fn emit_select(a: &mut Asm, experts: usize, top_k: usize, idx_off: usize, val_off: usize) {
    a.mov_imm(W8, 0xff80_0000);
    for k in 0..top_k {
        let scan = a.label();
        let keep = a.label();
        a.fmov_from_gp(s(0), W8); // best = -inf
        a.mov_imm(X12, 0); // best index
        a.mov_imm(X13, 0); // i
        a.mov(X10, SP);
        a.bind(scan);
        a.ldr(s(1), post(X10, 4));
        a.fcmp(s(1), Some(s(0)));
        a.b_cond(Cond::Le, keep);
        a.b_cond(Cond::Vs, keep);
        a.fmov(s(0), s(1));
        a.mov(X12, X13);
        a.bind(keep);
        a.add(X13, X13, 1);
        a.cmp(X13, experts as u64);
        a.blo(scan);
        a.str(X12, mem(SP, (idx_off + 8 * k) as i64));
        a.str(s(0), mem(SP, (val_off + 4 * k) as i64));
        a.mov(X14, SP);
        a.add(X14, X14, X12.lsl(2));
        a.str(W8, at(X14));
    }
}

/// Emit `(call: *const RouteCall)` for `experts` experts and `top_k` of them, Gemma's rule.
pub fn route(experts: usize, top_k: usize) -> Result<GeneratedKernel, ExecutionError> {
    route_with(experts, top_k, RouteRule::Gemma)
}

/// Emit `(call: *const RouteCall)` for `experts` experts and `top_k` of them, the weights by
/// `rule`.
pub fn route_with(
    experts: usize,
    top_k: usize,
    rule: RouteRule,
) -> Result<GeneratedKernel, ExecutionError> {
    if experts == 0
        || !experts.is_multiple_of(8)
        || top_k == 0
        || top_k > MAX_TOP_K
        || top_k > experts
    {
        return Err(ExecutionError(format!(
            "no router for {experts} experts, top {top_k}"
        )));
    }
    // Stack: the scores as f32 (experts · 4 bytes), then the chosen indices (8 bytes each) and
    // values (4 bytes each).
    let scores_bytes = (experts * 4).next_multiple_of(16);
    let idx_off = scores_bytes;
    let val_off = idx_off + 8 * MAX_TOP_K;
    let frame_bytes = (val_off + 4 * MAX_TOP_K).next_multiple_of(16);
    let (sig_off, frame_bytes) = lfm_frame(rule, experts, val_off, frame_bytes);
    let mut a = Asm::new();
    a.sub(SP, SP, frame_bytes as i64);
    let field = |f: usize| mem(X0, f as i64);
    // The scores widened to f32 on the stack.
    a.ldr(X9, field(offset_of!(RouteCall, scores)));
    a.mov(X10, SP);
    a.mov_imm(X11, (experts / 8) as u64);
    let widen_loop = a.label();
    a.bind(widen_loop);
    a.ldr(q(0), post(X9, 16));
    a.shll(v(1).s4(), v(0).h4());
    a.shll(v(2).s4(), v(0).h8());
    a.stp(q(1), q(2), post(X10, 32));
    a.subs(X11, X11, 1);
    a.bne(widen_loop);
    emit_exp_constants(&mut a);
    if let RouteRule::Lfm { .. } = rule {
        a.ldr(X9, field(offset_of!(RouteCall, per_expert_scale)));
        emit_lfm_scores(&mut a, experts, sig_off);
    }
    emit_select(&mut a, experts, top_k, idx_off, val_off);
    if rule != RouteRule::Gemma {
        match rule {
            RouteRule::Qwen { norm } => {
                emit_qwen_weights(&mut a, experts, top_k, norm, idx_off, val_off)
            }
            RouteRule::Lfm { norm, scale_bits } => emit_lfm_weights(
                &mut a,
                top_k,
                norm,
                f32::from_bits(scale_bits),
                idx_off,
                val_off,
                sig_off,
            ),
            RouteRule::Gemma => unreachable!(),
        }
        a.ldr(X10, field(offset_of!(RouteCall, weights_out)));
        a.ldr(X11, field(offset_of!(RouteCall, gate_up_table)));
        a.ldr(X15, field(offset_of!(RouteCall, down_table)));
        for k in 0..top_k {
            a.ldr(X12, mem(SP, (idx_off + 8 * k) as i64));
            a.ldr(s(0), mem(SP, (val_off + 4 * k) as i64));
            a.str(s(0), mem(X10, (4 * k) as i64));
            a.ldr(X13, idx_scaled8(X11, X12));
            a.ldr(X14, field(offset_of!(RouteCall, gate_up_slots) + 8 * k));
            a.str(X13, at(X14));
            a.ldr(X13, idx_scaled8(X15, X12));
            a.ldr(X14, field(offset_of!(RouteCall, down_slots) + 8 * k));
            a.str(X13, at(X14));
        }
        a.add(SP, SP, frame_bytes as i64);
        a.ret();
        return unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) };
    }
    // The softmax of the chosen scores (the first is the largest), scaled per expert; the sum
    // first.
    a.ldr(s(24), mem(SP, val_off as i64)); // max
    a.mov_imm(W9, 0);
    a.fmov_from_gp(s(25), W9); // sum = 0
    for k in 0..top_k {
        a.ldr(s(0), mem(SP, (val_off + 4 * k) as i64));
        a.fsub(s(0), s(0), s(24));
        emit_exp(&mut a);
        a.str(s(0), mem(SP, (val_off + 4 * k) as i64));
        a.fadd(s(25), s(25), s(0));
    }
    a.ldr(X9, field(offset_of!(RouteCall, per_expert_scale)));
    a.ldr(X10, field(offset_of!(RouteCall, weights_out)));
    a.ldr(X11, field(offset_of!(RouteCall, gate_up_table)));
    a.ldr(X15, field(offset_of!(RouteCall, down_table)));
    for k in 0..top_k {
        a.ldr(X12, mem(SP, (idx_off + 8 * k) as i64));
        a.ldr(s(0), mem(SP, (val_off + 4 * k) as i64));
        a.fdiv(s(0), s(0), s(25));
        a.add(X14, X9, X12.lsl(2));
        a.ldr(s(1), at(X14));
        a.fmul(s(0), s(0), s(1));
        a.str(s(0), mem(X10, (4 * k) as i64));
        // The weights of expert idx for slot k.
        a.ldr(X13, idx_scaled8(X11, X12));
        a.ldr(X14, field(offset_of!(RouteCall, gate_up_slots) + 8 * k));
        a.str(X13, at(X14));
        a.ldr(X13, idx_scaled8(X15, X12));
        a.ldr(X14, field(offset_of!(RouteCall, down_slots) + 8 * k));
        a.str(X13, at(X14));
    }
    a.add(SP, SP, frame_bytes as i64);
    a.ret();
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// `[base, index, lsl 3]`.
fn idx_scaled8(base: R, index: R) -> Mem {
    idx_scaled(base, index)
}

/// Emit `(call: *const CombineCall)`: `out[d] = Σ_{k<top_k} weights[k] · rows[k][d]` over `dim`
/// values (a multiple of 8), rounded to nearest even.
pub fn combine(dim: usize, top_k: usize) -> Result<GeneratedKernel, ExecutionError> {
    if dim == 0 || !dim.is_multiple_of(8) || top_k == 0 || top_k > MAX_TOP_K {
        return Err(ExecutionError(format!(
            "no combine of {top_k} rows of {dim}"
        )));
    }
    let mut a = Asm::new();
    let field = |f: usize| mem(X0, f as i64);
    // The row pointers x2..x9 (one per slot), the weights in v28/v29, the output in x10.
    a.ldr(X1, field(offset_of!(CombineCall, rows)));
    for k in 0..top_k {
        add_row(&mut a, x((2 + k) as u8), X1, k * dim * 2);
    }
    a.ldr(X11, field(offset_of!(CombineCall, weights)));
    a.ldp(q(28), q(29), at(X11));
    a.ldr(X10, field(offset_of!(CombineCall, out)));
    let top = a.label();
    a.mov_imm(X12, (dim / 8) as u64);
    a.bind(top);
    a.movi(v(0).d2(), 0);
    a.movi(v(1).d2(), 0);
    for k in 0..top_k {
        a.ldr(q(2), post(x((2 + k) as u8), 16));
        a.shll(v(3).s4(), v(2).h4());
        a.shll(v(4).s4(), v(2).h8());
        let w = if k < 4 { v(28) } else { v(29) };
        a.fmla(v(0).s4(), v(3).s4(), w.s((k % 4) as u8));
        a.fmla(v(1).s4(), v(4).s4(), w.s((k % 4) as u8));
    }
    narrow(&mut a, 5, 0, 1);
    a.str(q(5), post(X10, 16));
    a.subs(X12, X12, 1);
    a.bne(top);
    a.ret();
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// Emit `(call: *const CombineRowsCall, row: usize)` (`abi::CombineRowsCall`): `out[row][d] =
/// Σ_{k<top_k} weights[row·top_k + k] · rows[pos[row·top_k + k]][d]` over `dim` values (a multiple
/// of 8), in the order and with the rounding of [`combine`] (and of the x86-64 kernel of the same
/// name): `acc = 0`, then one fused multiply-add per slot in slot order, rounded to bf16 (nearest
/// even) once. Each weight is read alone (no read past the row's `top_k` weights).
pub fn combine_rows(dim: usize, top_k: usize) -> Result<GeneratedKernel, ExecutionError> {
    if dim == 0 || !dim.is_multiple_of(8) || top_k == 0 || top_k > MAX_TOP_K {
        return Err(ExecutionError(format!(
            "no combine of {top_k} rows of {dim} for a block"
        )));
    }
    let mut a = Asm::new();
    let field = |f: usize| mem(X0, f as i64);
    // x2 = this row's weights, x3 = its places, x4 = its output, x5 = the gathered rows, x14 = the
    // bytes of a row.
    a.mov_imm(X15, (4 * top_k) as u64);
    a.ldr(X2, field(offset_of!(CombineRowsCall, weights)));
    a.madd(X2, X1, X15, X2);
    a.ldr(X3, field(offset_of!(CombineRowsCall, pos)));
    a.madd(X3, X1, X15, X3);
    a.mov_imm(X14, (dim * 2) as u64);
    a.ldr(X4, field(offset_of!(CombineRowsCall, out)));
    a.madd(X4, X1, X14, X4);
    a.ldr(X5, field(offset_of!(CombineRowsCall, rows)));
    // The row of each slot in x6..x13, its weight in s16..s23.
    for k in 0..top_k {
        let p = x((6 + k) as u8);
        a.ldr(w((6 + k) as u8), mem(X3, (4 * k) as i64));
        a.madd(p, p, X14, X5);
        a.ldr(s((16 + k) as u8), mem(X2, (4 * k) as i64));
    }
    let top = a.label();
    a.mov_imm(X15, (dim / 8) as u64);
    a.bind(top);
    a.movi(v(0).d2(), 0);
    a.movi(v(1).d2(), 0);
    for k in 0..top_k {
        let wk = v((16 + k) as u8);
        a.ldr(q(2), post(x((6 + k) as u8), 16));
        a.shll(v(3).s4(), v(2).h4());
        a.shll(v(4).s4(), v(2).h8());
        a.fmla(v(0).s4(), v(3).s4(), wk.s(0));
        a.fmla(v(1).s4(), v(4).s4(), wk.s(0));
    }
    narrow(&mut a, 5, 0, 1);
    a.str(q(5), post(X4, 16));
    a.subs(X15, X15, 1);
    a.bne(top);
    a.ret();
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

fn add_row(a: &mut Asm, dst: R, base: R, bytes: usize) {
    if bytes < 4096 {
        a.add(dst, base, bytes as i64);
    } else {
        a.mov_imm(X13, bytes as u64);
        a.add(dst, base, X13);
    }
}

/// Emit `(call: *const QwenOutCall, row: usize)` (`abi::QwenOutCall`): the end of a sparse block of
/// `qwen3_5_moe` and the residual for one row, `dim` values (a multiple of 8), `top_k` experts. The
/// logit of the shared expert's gate is a dot product in f32 (eight lanes as two quads, then the
/// halves, as the x86-64 kernel reduces its eight lanes), rounded to bf16; its sigmoid `1 / (1 +
/// e^-l)` (or `e^l / (1 + e^l)` below zero) rounded to bf16; then per eight values every product
/// and every sum rounded to bf16 in the order of the module (`index_add_` into a bf16 zero vector,
/// slot by slot), `+ bf16(g · shared)`, `+ x`.
pub fn qwen_out(dim: usize, top_k: usize) -> Result<GeneratedKernel, ExecutionError> {
    if dim == 0 || !dim.is_multiple_of(8) || top_k == 0 || top_k > MAX_TOP_K {
        return Err(ExecutionError(format!(
            "no Qwen expert output of {top_k} rows of {dim}"
        )));
    }
    let mut a = Asm::new();
    let field = |f: usize| mem(X0, f as i64);
    // x12 = the bytes of a row, x13 = the offset of row `x1`.
    a.mov_imm(X12, (dim * 2) as u64);
    a.mul(X13, X1, X12);
    // The logit `shared_gate · xn[row]`: lanes 0..3 in v0, 4..7 in v1.
    a.ldr(X9, field(offset_of!(QwenOutCall, xn)));
    a.add(X9, X9, X13);
    a.ldr(X10, field(offset_of!(QwenOutCall, gate)));
    a.movi(v(0).d2(), 0);
    a.movi(v(1).d2(), 0);
    a.mov_imm(X11, (dim / 8) as u64);
    let dot = a.label();
    a.bind(dot);
    a.ldr(q(2), post(X9, 16));
    a.ldr(q(3), post(X10, 16));
    a.shll(v(4).s4(), v(2).h4());
    a.shll(v(5).s4(), v(2).h8());
    a.shll(v(6).s4(), v(3).h4());
    a.shll(v(7).s4(), v(3).h8());
    a.fmla(v(0).s4(), v(4).s4(), v(6).s4());
    a.fmla(v(1).s4(), v(5).s4(), v(7).s4());
    a.subs(X11, X11, 1);
    a.bne(dot);
    // (l0 + l4, ..), then (· + the lane two above), then the two lanes: the x86-64 order.
    a.fadd(v(0).s4(), v(0).s4(), v(1).s4());
    a.ext(v(1).b16(), v(0).b16(), v(0).b16(), 8);
    a.fadd(v(0).s4(), v(0).s4(), v(1).s4());
    a.faddp(s(0), v(0).s2(), None);
    round_bf16_s(&mut a, 0, 1);
    // Its sigmoid, from `e = exp(-|l|)`.
    a.fmov(s(5), s(0));
    emit_exp_constants(&mut a);
    a.fneg(s(1), s(0));
    a.fmin(s(0), s(0), s(1));
    emit_exp(&mut a);
    a.fadd(s(6), s(0), s(23));
    let negative = a.label();
    a.fcmp(s(5), None);
    a.b_cond(Cond::Lt, negative);
    a.fmov(s(0), s(23));
    a.bind(negative);
    a.fdiv(s(0), s(0), s(6));
    round_bf16_s(&mut a, 0, 1);
    a.fmov(s(24), s(0)); // g
    if !has_bf16() {
        a.mov_imm(W9, 1);
        a.dup_gp(v(25).s4(), W9);
        a.mov_imm(W9, 0x7fff);
        a.dup_gp(v(26).s4(), W9);
    }
    // The expert rows of the row: x2..x9, `rows + pos[row·top_k + k] · dim · 2`.
    a.mov_imm(X11, (4 * top_k) as u64);
    a.mul(X11, X1, X11);
    a.ldr(X10, field(offset_of!(QwenOutCall, pos)));
    a.add(X10, X10, X11);
    a.ldr(X16, field(offset_of!(QwenOutCall, rows)));
    for k in 0..top_k {
        a.ldr(W17, mem(X10, (4 * k) as i64));
        a.madd(x((2 + k) as u8), X17, X12, X16);
    }
    // The weights of the row in s16..s(16 + top_k).
    a.ldr(X10, field(offset_of!(QwenOutCall, weights)));
    a.add(X10, X10, X11);
    for k in 0..top_k {
        a.ldr(s((16 + k) as u8), mem(X10, (4 * k) as i64));
    }
    a.ldr(X14, field(offset_of!(QwenOutCall, shared)));
    a.add(X14, X14, X13);
    a.ldr(X15, field(offset_of!(QwenOutCall, x)));
    a.add(X15, X15, X13);
    a.mov_imm(X11, (dim / 8) as u64);
    let top = a.label();
    a.bind(top);
    // v3 (values 0..3) and v4 (4..7): Σ_k bf16(w_k · y_k), each sum rounded.
    for k in 0..top_k {
        let (lo, hi) = if k == 0 { (3, 4) } else { (1, 2) };
        a.ldr(q(0), post(x((2 + k) as u8), 16));
        a.shll(v(lo).s4(), v(0).h4());
        a.shll(v(hi).s4(), v(0).h8());
        a.fmul(v(lo).s4(), v(lo).s4(), v((16 + k) as u8).s(0));
        a.fmul(v(hi).s4(), v(hi).s4(), v((16 + k) as u8).s(0));
        round_bf16_quad(&mut a, lo, 6);
        round_bf16_quad(&mut a, hi, 7);
        if k > 0 {
            a.fadd(v(3).s4(), v(3).s4(), v(1).s4());
            a.fadd(v(4).s4(), v(4).s4(), v(2).s4());
            round_bf16_quad(&mut a, 3, 6);
            round_bf16_quad(&mut a, 4, 7);
        }
    }
    // `+ bf16(g · shared)`, rounded; then the residual.
    a.ldr(q(0), post(X14, 16));
    a.shll(v(1).s4(), v(0).h4());
    a.shll(v(2).s4(), v(0).h8());
    a.fmul(v(1).s4(), v(1).s4(), v(24).s(0));
    a.fmul(v(2).s4(), v(2).s4(), v(24).s(0));
    round_bf16_quad(&mut a, 1, 6);
    round_bf16_quad(&mut a, 2, 7);
    a.fadd(v(3).s4(), v(3).s4(), v(1).s4());
    a.fadd(v(4).s4(), v(4).s4(), v(2).s4());
    round_bf16_quad(&mut a, 3, 6);
    round_bf16_quad(&mut a, 4, 7);
    a.ldr(q(0), at(X15));
    a.shll(v(1).s4(), v(0).h4());
    a.shll(v(2).s4(), v(0).h8());
    a.fadd(v(3).s4(), v(3).s4(), v(1).s4());
    a.fadd(v(4).s4(), v(4).s4(), v(2).s4());
    narrow(&mut a, 5, 3, 4);
    a.str(q(5), post(X15, 16));
    a.subs(X11, X11, 1);
    a.bne(top);
    a.ret();
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// `dst = base + index · stride` with `stride` in `x17`; `dst` may be `base`.
fn scaled_add(a: &mut Asm, dst: R, index: R, base: R, stride: usize) {
    a.mov_imm(X17, stride as u64);
    a.madd(dst, index, X17, base);
}

/// Copy `bytes` bytes from `[src]` to `[dst]`, both advanced; clobbers `q0`, `x16`, `w17`.
fn copy_bytes(a: &mut Asm, dst: R, src: R, bytes: usize) {
    if bytes >= 16 {
        let top = a.label();
        a.mov_imm(X16, (bytes / 16) as u64);
        a.bind(top);
        a.ldr(q(0), post(src, 16));
        a.str(q(0), post(dst, 16));
        a.subs(X16, X16, 1);
        a.bne(top);
    }
    let mut rest = bytes % 16;
    for (step, reg) in [(8usize, X17), (4, W17)] {
        if rest >= step {
            a.ldr(reg, post(src, step as i64));
            a.str(reg, post(dst, step as i64));
            rest -= step;
        }
    }
    if rest >= 2 {
        a.ldrh(W17, post(src, 2));
        a.strh(W17, post(dst, 2));
        rest -= 2;
    }
    if rest == 1 {
        a.ldrb(W17, post(src, 1));
        a.strb(W17, post(dst, 1));
    }
}

/// Emit `(call: *const RouteRowsCall)` (`abi::RouteRowsCall`): the router of the `frame.rows` rows
/// of a block, the selection and the weights of each row as in [`route_with`] (bit for bit), then
/// the rows grouped by expert. One task. The same contract and order as the x86-64 kernel of the
/// same name.
pub fn route_rows(spec: RouteRowsSpec) -> Result<GeneratedKernel, ExecutionError> {
    let RouteRowsSpec {
        experts,
        top_k,
        max_rows,
        a_row,
        gate_up_out_row,
        down_in_row,
        down_out_row,
        gate_up_tiles,
        down_tiles,
        rule,
    } = spec;
    if experts == 0
        || !experts.is_multiple_of(16)
        || top_k == 0
        || top_k > MAX_TOP_K
        || top_k > experts
        || max_rows == 0
        || a_row == 0
        || gate_up_tiles.0 == 0
        || down_tiles.0 == 0
    {
        return Err(ExecutionError(format!("no router of a block for {spec:?}")));
    }
    let scores_bytes = (experts * 4).next_multiple_of(16);
    let idx_off = scores_bytes;
    let val_off = idx_off + 8 * MAX_TOP_K;
    let frame_bytes = (val_off + 4 * MAX_TOP_K).next_multiple_of(16);
    let (sig_off, frame_bytes) = lfm_frame(rule, experts, val_off, frame_bytes);
    let mut a = Asm::new();
    // x19 = the call, x20 = the rows of the block, x21 = a row, x22 = an expert, x23 = its first
    // row, x24 = its rows.
    let frame = Frame::enter(
        &mut a,
        false,
        &[X19, X20, X21, X22, X23, X24],
        false,
        frame_bytes,
    );
    let field = |f: usize| mem(X19, f as i64);
    a.mov(X19, X0);
    a.ldr(X9, field(offset_of!(RouteRowsCall, frame)));
    a.ldr(X20, mem(X9, offset_of!(CallFrame, rows) as i64));
    // No expert has a row yet.
    a.ldr(X9, field(offset_of!(RouteRowsCall, row_counts)));
    a.mov_imm(X10, experts as u64);
    let zero = a.label();
    a.bind(zero);
    a.str(XZR, post(X9, 8));
    a.subs(X10, X10, 1);
    a.bne(zero);
    emit_exp_constants(&mut a);
    // ── Every row: its scores widened, the selection, the weights, its experts counted. ──
    let rows_done = a.label();
    a.mov_imm(X21, 0);
    a.cbz(X20, rows_done);
    let row_top = a.label();
    a.bind(row_top);
    // Row r of block b at `scores + (b · max_rows + r) · 32`, sixteen scores widened to `sp + 64
    // b`.
    a.ldr(X9, field(offset_of!(RouteRowsCall, scores)));
    a.add(X9, X9, X21.lsl(5));
    a.mov(X10, SP);
    a.mov_imm(X11, (max_rows * 32) as u64);
    a.mov_imm(X12, (experts / 16) as u64);
    let widen = a.label();
    a.bind(widen);
    a.ldp(q(0), q(1), at(X9));
    a.add(X9, X9, X11);
    a.shll(v(2).s4(), v(0).h4());
    a.shll(v(3).s4(), v(0).h8());
    a.shll(v(4).s4(), v(1).h4());
    a.shll(v(5).s4(), v(1).h8());
    a.stp(q(2), q(3), post(X10, 32));
    a.stp(q(4), q(5), post(X10, 32));
    a.subs(X12, X12, 1);
    a.bne(widen);
    if let RouteRule::Lfm { .. } = rule {
        a.ldr(X9, field(offset_of!(RouteRowsCall, per_expert_scale)));
        emit_lfm_scores(&mut a, experts, sig_off);
    }
    emit_select(&mut a, experts, top_k, idx_off, val_off);
    match rule {
        RouteRule::Gemma => {
            a.ldr(s(24), mem(SP, val_off as i64)); // max
            a.mov_imm(W9, 0);
            a.fmov_from_gp(s(25), W9); // sum = 0
            for k in 0..top_k {
                a.ldr(s(0), mem(SP, (val_off + 4 * k) as i64));
                a.fsub(s(0), s(0), s(24));
                emit_exp(&mut a);
                a.str(s(0), mem(SP, (val_off + 4 * k) as i64));
                a.fadd(s(25), s(25), s(0));
            }
            a.ldr(X9, field(offset_of!(RouteRowsCall, per_expert_scale)));
        }
        RouteRule::Qwen { norm } => {
            emit_qwen_weights(&mut a, experts, top_k, norm, idx_off, val_off)
        }
        RouteRule::Lfm { norm, scale_bits } => emit_lfm_weights(
            &mut a,
            top_k,
            norm,
            f32::from_bits(scale_bits),
            idx_off,
            val_off,
            sig_off,
        ),
    }
    // x10 = the weights of the row, x11 = its experts, x15 = the rows of every expert.
    a.mov_imm(X12, (4 * top_k) as u64);
    a.mul(X12, X21, X12);
    a.ldr(X10, field(offset_of!(RouteRowsCall, weights_out)));
    a.add(X10, X10, X12);
    a.ldr(X11, field(offset_of!(RouteRowsCall, ids)));
    a.add(X11, X11, X12);
    a.ldr(X15, field(offset_of!(RouteRowsCall, row_counts)));
    for k in 0..top_k {
        a.ldr(X12, mem(SP, (idx_off + 8 * k) as i64));
        a.ldr(s(0), mem(SP, (val_off + 4 * k) as i64));
        if rule == RouteRule::Gemma {
            a.fdiv(s(0), s(0), s(25));
            a.add(X14, X9, X12.lsl(2));
            a.ldr(s(1), at(X14));
            a.fmul(s(0), s(0), s(1));
        }
        a.str(s(0), mem(X10, (4 * k) as i64));
        a.str(W12, mem(X11, (4 * k) as i64));
        a.ldr(X13, idx_scaled8(X15, X12));
        a.add(X13, X13, 1);
        a.str(X13, idx_scaled8(X15, X12));
    }
    a.add(X21, X21, 1);
    a.cmp(X21, X20);
    a.blo(row_top);
    a.bind(rows_done);
    // ── Every expert: its first row, its calls moved there, the task counts of its two grids. ──
    a.mov_imm(X22, 0);
    a.mov_imm(X23, 0);
    let expert_top = a.label();
    a.bind(expert_top);
    a.ldr(X9, field(offset_of!(RouteRowsCall, row_counts)));
    a.ldr(X24, idx_scaled8(X9, X22));
    a.ldr(X9, field(offset_of!(RouteRowsCall, cursor)));
    a.str(X23, idx_scaled8(X9, X22));
    // The gate/up call: activations at the first row of the gathered input, output at its first
    // row; the down call the same way.
    for (calls, input, in_row, output, out_row) in [
        (
            offset_of!(RouteRowsCall, gate_up_calls),
            offset_of!(RouteRowsCall, gathered),
            a_row,
            offset_of!(RouteRowsCall, gate_up_out),
            gate_up_out_row,
        ),
        (
            offset_of!(RouteRowsCall, down_calls),
            offset_of!(RouteRowsCall, down_in),
            down_in_row,
            offset_of!(RouteRowsCall, down_out),
            down_out_row,
        ),
    ] {
        a.ldr(X9, field(calls));
        a.ldr(X10, idx_scaled8(X9, X22));
        a.ldr(X11, field(input));
        scaled_add(&mut a, X11, X23, X11, in_row);
        a.str(X11, mem(X10, offset_of!(Bf16Call, activations) as i64));
        a.ldr(X11, field(output));
        scaled_add(&mut a, X11, X23, X11, out_row);
        a.str(X11, mem(X10, offset_of!(Bf16Call, output) as i64));
    }
    // ⌈rows / mr⌉ · column tiles for each grid.
    a.ldr(X9, field(offset_of!(RouteRowsCall, frame)));
    a.ldr(X15, mem(X9, offset_of!(CallFrame, counts) as i64));
    for (tiles, grids) in [
        (gate_up_tiles, offset_of!(RouteRowsCall, gate_up_grids)),
        (down_tiles, offset_of!(RouteRowsCall, down_grids)),
    ] {
        a.mov_imm(X11, (tiles.0 - 1) as u64);
        a.add(X9, X24, X11);
        a.mov_imm(X11, tiles.0 as u64);
        a.udiv(X9, X9, X11);
        a.mov_imm(X11, tiles.1 as u64);
        a.mul(X9, X9, X11);
        a.ldr(X12, field(grids));
        a.ldr(X12, idx_scaled8(X12, X22));
        a.str(X9, idx_scaled8(X15, X12));
    }
    a.add(X23, X23, X24);
    a.add(X22, X22, 1);
    a.cmp(X22, experts as u64);
    a.blo(expert_top);
    // ── Every (row, k): its place among the rows of its expert, the row copied there. ──
    let copies_done = a.label();
    a.mov_imm(X21, 0);
    a.cbz(X20, copies_done);
    a.ldr(X22, field(offset_of!(RouteRowsCall, cursor)));
    let copy_top = a.label();
    a.bind(copy_top);
    a.mov_imm(X12, (4 * top_k) as u64);
    a.mul(X12, X21, X12);
    a.ldr(X10, field(offset_of!(RouteRowsCall, ids)));
    a.add(X10, X10, X12);
    a.ldr(X11, field(offset_of!(RouteRowsCall, pos_out)));
    a.add(X11, X11, X12);
    a.ldr(X23, field(offset_of!(RouteRowsCall, source)));
    scaled_add(&mut a, X23, X21, X23, a_row);
    for k in 0..top_k {
        a.ldr(W12, mem(X10, (4 * k) as i64));
        a.ldr(X13, idx_scaled8(X22, X12));
        a.add(X14, X13, 1);
        a.str(X14, idx_scaled8(X22, X12));
        a.str(W13, mem(X11, (4 * k) as i64));
        a.ldr(X14, field(offset_of!(RouteRowsCall, gathered)));
        scaled_add(&mut a, X14, X13, X14, a_row);
        a.mov(X15, X23);
        copy_bytes(&mut a, X14, X15, a_row);
    }
    a.add(X21, X21, 1);
    a.cmp(X21, X20);
    a.blo(copy_top);
    a.bind(copies_done);
    // The rows of the activation and of its quantization.
    a.mov_imm(X9, top_k as u64);
    a.mul(X9, X20, X9);
    a.ldr(X10, field(offset_of!(RouteRowsCall, total)));
    a.str(X9, at(X10));
    frame.leave(&mut a);
    a.ret();
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}
