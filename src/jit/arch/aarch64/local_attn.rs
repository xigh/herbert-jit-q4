//! [`ElemOp::LocalAttention`] on AArch64 (NEON): the chunked local attention
//! of the Gemma 4 audio tower, one query row and every head per call — the
//! computation of the x86 kernel (`x86/emit_local_attn.rs`), four lanes wide.
//!
//! Per head: `q' = q · q_scale · sp` widened into a stack row of f32; the
//! `window` logits `(q'·k_{t−a}) · k_scale + q'·rel[window − a]` into sixteen
//! stack lanes; `cap · tanh(l / cap)` by the exponential of the elementary
//! kernels; the lanes `a ≥ min(t + 1, window)` masked; the softmax; then
//! `Σ_a p_a · v_{t−a}` narrowed to bf16. The rows before row 0 of `k` and `v`
//! are read (zeros) and weigh zero.
//!
//! The block kernel (`rows > 0`, `emit_block`): one head of a tile of rows
//! per call, the same values bit for bit, as the x86 one: `q'` of the tile on
//! the stack; the logits by register tiles of four rows × four keys (sixteen
//! chains, each the row kernel's: the low then the high four lanes of a chunk
//! into one accumulator, chunks in order, the same pairwise sum), each key
//! chunk widened once for the four rows, the relative keys likewise; the
//! softmax of each row by the row kernel's code; then `Σ_a p_a · v_{t−a}` by
//! groups of eight rows, each value chunk widened once for the rows that see
//! it, `a` rising per row.

use crate::core::error::{HerbertError, Result};
use crate::jit::arch::aarch64::elem::{narrow, vexp, widen, ExpRegs};
use crate::jit::arch::aarch64::util::{backend_err, splat, Frame};
use crate::jit::emit_elem::{ElemOp, ElemSpec};
use a64::*;

/// `-1e30`, the logit of a masked lane.
const MASKED: u32 = 0xF149_F2CA;

/// Generates the kernel of `spec` (an [`ElemOp::LocalAttention`]).
pub fn emit(spec: &ElemSpec) -> Result<Vec<u8>> {
    spec.check()
        .map_err(|d| HerbertError::Backend(format!("backend_jit local attention: {d}")))?;
    let ElemOp::LocalAttention {
        heads,
        window,
        rows,
    } = spec.op
    else {
        return Err(HerbertError::Backend(
            "local_attn: not a local attention".into(),
        ));
    };
    let (dim, heads, window) = (spec.dim, usize::from(heads), usize::from(window));
    if rows > 0 {
        return emit_block(dim, heads, window, usize::from(rows)).map_err(backend_err(
            format_args!("elem {} dim={}", spec.op.name(), spec.dim),
        ));
    }
    let db = (dim * 2) as i64;
    let hb = (heads * dim * 2) as u64;
    let chunks = dim / 8;
    // The frame: q' (f32), then the sixteen logits.
    let ol = (dim * 4) as i64;
    let mut a = Asm::new();
    let frame = Frame::enter(&mut a, false, &[], false, dim * 4 + 64);

    // x0 = q, x1 = k, x2 = v, x3 = rel (`sp`, the per-dimension scale, then the relative keys
    // at x3 + dim·2), x4 = y, x5 = t; s0 = q_scale, s1 = k_scale, s2 = cap.
    let k = ExpRegs::at(16);
    k.load(&mut a, X9);
    a.dup(v(24).s4(), v(0).s(0));
    a.fmov(s(25), s(1));
    a.dup(v(26).s4(), v(2).s(0));
    splat(&mut a, 27, (-2.0f32).to_bits(), X9);
    splat(&mut a, 28, 0x8000_0000, X9);
    // x9 = min(t + 1, window), x10 = heads left, x11 = the relative keys of the head, x12 = a row's
    // bytes.
    a.add(X9, X5, 1);
    a.mov_imm(X13, window as u64);
    a.cmp(X9, X13);
    a.csel(X9, X13, X9, Cond::Hi);
    a.mov_imm(X10, heads as u64);
    a.add(X11, X3, db);
    a.mov_imm(X12, hb);

    let head = a.label();
    a.bind(head);
    // q' = q · q_scale · sp, f32.
    for ch in 0..chunks {
        let at = (ch * 16) as i64;
        a.ldr(q(0), mem(X0, at));
        a.ldr(q(1), mem(X3, at));
        widen(&mut a, 2, 0, false);
        widen(&mut a, 3, 0, true);
        widen(&mut a, 4, 1, false);
        widen(&mut a, 5, 1, true);
        a.fmul(v(2).s4(), v(2).s4(), v(24).s4());
        a.fmul(v(2).s4(), v(2).s4(), v(4).s4());
        a.fmul(v(3).s4(), v(3).s4(), v(24).s4());
        a.fmul(v(3).s4(), v(3).s4(), v(5).s4());
        a.stp(q(2), q(3), mem(SP, (ch * 32) as i64));
    }
    // The logits, the lanes past the window zero.
    a.movi(v(0).d2(), 0);
    a.stp(q(0), q(0), mem(SP, ol));
    a.stp(q(0), q(0), mem(SP, ol + 32));
    a.mov(X13, X1);
    a.mov_imm(X14, window as u64 * hb);
    a.add(X14, X11, X14);
    for key in 0..window {
        a.movi(v(4).d2(), 0);
        a.movi(v(5).d2(), 0);
        for ch in 0..chunks {
            let at = (ch * 16) as i64;
            a.ldp(q(2), q(3), mem(SP, (ch * 32) as i64));
            a.ldr(q(0), mem(X13, at));
            widen(&mut a, 6, 0, false);
            widen(&mut a, 7, 0, true);
            a.fmla(v(4).s4(), v(2).s4(), v(6).s4());
            a.fmla(v(4).s4(), v(3).s4(), v(7).s4());
            a.ldr(q(1), mem(X14, at));
            widen(&mut a, 6, 1, false);
            widen(&mut a, 7, 1, true);
            a.fmla(v(5).s4(), v(2).s4(), v(6).s4());
            a.fmla(v(5).s4(), v(3).s4(), v(7).s4());
        }
        for r in [4u8, 5] {
            a.faddp(v(r).s4(), v(r).s4(), Some(v(r).s4()));
            a.faddp(s(r), v(r).s2(), None);
        }
        a.fmul(s(4), s(4), s(25));
        a.fadd(s(4), s(4), s(5));
        a.str(s(4), mem(SP, ol + (key * 4) as i64));
        a.sub(X13, X13, X12);
        a.sub(X14, X14, X12);
    }
    softmax16(&mut a, &k, SP, ol);
    // y = Σ_a p_a · v_{t−a}, narrowed to bf16.
    for ch in 0..chunks {
        a.movi(v(4).d2(), 0);
        a.movi(v(5).d2(), 0);
        a.add(X13, X2, (ch * 16) as i64);
        for key in 0..window {
            a.ldr(s(0), mem(SP, ol + (key * 4) as i64));
            a.ldr(q(1), at(X13));
            widen(&mut a, 2, 1, false);
            widen(&mut a, 3, 1, true);
            a.fmla(v(4).s4(), v(2).s4(), v(0).s(0));
            a.fmla(v(5).s4(), v(3).s4(), v(0).s(0));
            a.sub(X13, X13, X12);
        }
        narrow(&mut a, 1, 4, 5);
        a.str(q(1), mem(X4, (ch * 16) as i64));
    }
    for r in [X0, X1, X2, X4, X11] {
        a.add(r, r, db);
    }
    a.subs(X10, X10, 1);
    a.bne(head);

    frame.leave(&mut a);
    a.ret();
    a.finish()
        .map(|c| c.bytes)
        .map_err(backend_err(format_args!(
            "elem {} dim={}",
            spec.op.name(),
            spec.dim
        )))
}

/// The lanes `x9..16` of the sixteen logits at `base + at` set to `bits` (`x13`, `w15`, `x16`
/// clobbered).
fn mask(a: &mut Asm, base: R, at: i64, bits: u32) {
    let (top, done) = (a.label(), a.label());
    a.mov_imm(W15, u64::from(bits));
    a.mov(X13, X9);
    a.add(X16, base, at);
    a.bind(top);
    a.cmp(X13, 16);
    a.bhs(done);
    a.str(W15, idx_scaled(X16, X13));
    a.add(X13, X13, 1);
    a.b(top);
    a.bind(done);
}

/// The sixteen logits at `base + at` (lanes `x9..16` dropped) into the probabilities: `cap · tanh(l
/// / cap)` (`sign(u) · (1 − e^{−2|u|}) / (1 + e^{−2|u|})`, `cap` in `v26`, `−2` in `v27`, the sign
/// bit in `v28`), then the softmax. `v0..v6`, `x13`, `w15`, `x16` clobbered.
fn softmax16(a: &mut Asm, k: &ExpRegs, base: R, at: i64) {
    for i in 0..4i64 {
        a.ldr(q(0), mem(base, at + i * 16));
        a.fdiv(v(0).s4(), v(0).s4(), v(26).s4());
        a.fabs(v(1).s4(), v(0).s4());
        a.fmul(v(1).s4(), v(1).s4(), v(27).s4());
        vexp(a, k, 1, [2, 3, 4, 5]);
        a.fsub(v(2).s4(), v(k.one).s4(), v(1).s4());
        a.fadd(v(3).s4(), v(k.one).s4(), v(1).s4());
        a.fdiv(v(2).s4(), v(2).s4(), v(3).s4());
        a.and_v(v(0).b16(), v(0).b16(), v(28).b16());
        a.orr_v(v(2).b16(), v(2).b16(), v(0).b16());
        a.fmul(v(2).s4(), v(2).s4(), v(26).s4());
        a.str(q(2), mem(base, at + i * 16));
    }
    // The softmax over the valid lanes.
    mask(a, base, at, MASKED);
    a.ldp(q(0), q(1), mem(base, at));
    a.ldp(q(2), q(3), mem(base, at + 32));
    a.fmax(v(0).s4(), v(0).s4(), v(1).s4());
    a.fmax(v(2).s4(), v(2).s4(), v(3).s4());
    a.fmax(v(0).s4(), v(0).s4(), v(2).s4());
    a.fmaxv(s(0), v(0).s4());
    a.dup(v(6).s4(), v(0).s(0));
    for i in 0..4i64 {
        a.ldr(q(0), mem(base, at + i * 16));
        a.fsub(v(0).s4(), v(0).s4(), v(6).s4());
        vexp(a, k, 0, [1, 2, 3, 4]);
        a.str(q(0), mem(base, at + i * 16));
    }
    mask(a, base, at, 0);
    a.ldp(q(0), q(1), mem(base, at));
    a.ldp(q(2), q(3), mem(base, at + 32));
    a.fadd(v(0).s4(), v(0).s4(), v(1).s4());
    a.fadd(v(2).s4(), v(2).s4(), v(3).s4());
    a.fadd(v(0).s4(), v(0).s4(), v(2).s4());
    a.faddp(v(0).s4(), v(0).s4(), Some(v(0).s4()));
    a.faddp(s(0), v(0).s2(), None);
    a.fmov_imm(s(1), 1.0);
    a.fdiv(s(0), s(1), s(0));
    a.dup(v(6).s4(), v(0).s(0));
    for i in 0..4i64 {
        a.ldr(q(0), mem(base, at + i * 16));
        a.fmul(v(0).s4(), v(0).s4(), v(6).s4());
        a.str(q(0), mem(base, at + i * 16));
    }
}

/// `rd = rn + imm`, through `x17` when `imm` is wider than an immediate.
fn add_wide(a: &mut Asm, rd: R, rn: R, imm: i64) {
    if (0..4096).contains(&imm) {
        a.add(rd, rn, imm);
    } else {
        a.mov_imm(X17, imm as u64);
        a.add(rd, rn, X17);
    }
}

/// One register tile of the logits: four rows of a group against four keys (`Keys(j0)`: the keys
/// `j0..` of the group, row `r` sees key `j` at `a = r + window − 1 − j`) or four relative keys
/// (`Rel(a0)`: `a0..`).
#[derive(Clone, Copy)]
enum Tile {
    Keys(usize),
    Rel(usize),
}

/// The block kernel: one head of a tile of `rows` rows.
fn emit_block(
    dim: usize,
    heads: usize,
    window: usize,
    rows: usize,
) -> std::result::Result<Vec<u8>, a64::Error> {
    const G: usize = 4; // rows of a register tile of the logits
    const KG: usize = 4; // its keys
    const PG: usize = 8; // rows of a group of the output
    let db = (dim * 2) as i64;
    let hb = (heads * dim * 2) as i64;
    let chunks = dim / 8;
    let qrow = (dim * 4) as i64;
    let rows_i = rows as i64;
    // The frame: the three scalars, the pointers, q' of the tile (f32), then per row the sixteen
    // lanes of the dots with the keys (then the logits, then the probabilities), then those
    // with the relative keys.
    let os = 0i64;
    let og = 16i64;
    let (g_q, g_k, g_rel, g_v, g_y, g_t0, g_n, g_sp) = (
        og,
        og + 8,
        og + 16,
        og + 24,
        og + 32,
        og + 40,
        og + 48,
        og + 56,
    );
    let oq = og + 64;
    let oa = oq + rows_i * qrow;
    let ob = oa + rows_i * 64;
    let locals = (ob + rows_i * 64 + 15) & !15;
    let mut a = Asm::new();
    let frame = Frame::enter(&mut a, false, &[X19, X20, X21, X22, X23, X24], true, 0);
    // The locals, wider than an immediate: their 4 KiB pages, then the rest.
    let (pages, rest) = (locals & !0xfff, locals & 0xfff);
    for part in [pages, rest] {
        if part > 0 {
            a.sub(SP, SP, part);
        }
    }
    let exit = a.label();

    // x0 = call, x1 = task; s0 = q_scale, s1 = k_scale, s2 = cap.
    a.str(s(0), mem(SP, os));
    a.str(s(1), mem(SP, os + 4));
    a.str(s(2), mem(SP, os + 8));
    // x10 = tile, x11 = head; x12 = t0, x13 = n = min(rows, *call.rows − t0).
    a.mov_imm(X9, heads as u64);
    a.udiv(X10, X1, X9);
    a.msub(X11, X10, X9, X1);
    a.ldr(X13, mem(X0, 40));
    a.ldr(X13, mem(X13, 0));
    a.mov_imm(X9, rows as u64);
    a.mul(X12, X10, X9);
    a.subs(X13, X13, X12);
    a.ble(exit);
    a.cmp(X13, X9);
    a.csel(X13, X9, X13, Cond::Hi);
    a.str(X13, mem(SP, g_n));
    a.str(X12, mem(SP, g_t0));
    // x14 = the bytes of head x11, x15 = those of row t0 and the head.
    a.mov_imm(X9, db as u64);
    a.mul(X14, X11, X9);
    a.mov_imm(X9, hb as u64);
    a.madd(X15, X12, X9, X14);
    for (field, at, back) in [
        (0i64, g_q, 0i64),
        (8, g_k, (window as i64 - 1) * hb),
        (16, g_v, (window as i64 - 1) * hb),
        (32, g_y, 0),
    ] {
        a.ldr(X9, mem(X0, field));
        a.add(X9, X9, X15);
        if back != 0 {
            a.mov_imm(X10, back as u64);
            a.sub(X9, X9, X10);
        }
        a.str(X9, mem(SP, at));
    }
    // sp, and the relative key of position 1 (a = window − 1) of the head.
    a.ldr(X9, mem(X0, 24));
    a.str(X9, mem(SP, g_sp));
    a.add(X9, X9, X14);
    add_wide(&mut a, X9, X9, db + hb);
    a.str(X9, mem(SP, g_rel));
    // The dots zero, the lanes past the window kept so.
    a.movi(v(0).d2(), 0);
    add_wide(&mut a, X9, SP, oa);
    for i in 0..(rows_i * 8) {
        a.str(q(0), mem(X9, i * 16));
    }

    // q' = q · q_scale · sp, f32, for the rows of the register tiles (x19 = ⌈n / 4⌉·4).
    a.ldr(s(24), mem(SP, os));
    a.dup(v(24).s4(), v(24).s(0));
    a.ldr(X19, mem(SP, g_n));
    a.add(X19, X19, (G - 1) as i64);
    a.lsr(X19, X19, 2);
    a.lsl(X19, X19, 2);
    a.ldr(X20, mem(SP, g_q));
    a.ldr(X21, mem(SP, g_sp));
    add_wide(&mut a, X22, SP, oq);
    let qrows = a.label();
    a.bind(qrows);
    for ch in 0..chunks {
        let at = (ch * 16) as i64;
        a.ldr(q(0), mem(X20, at));
        a.ldr(q(1), mem(X21, at));
        widen(&mut a, 2, 0, false);
        widen(&mut a, 3, 0, true);
        widen(&mut a, 4, 1, false);
        widen(&mut a, 5, 1, true);
        a.fmul(v(2).s4(), v(2).s4(), v(24).s4());
        a.fmul(v(2).s4(), v(2).s4(), v(4).s4());
        a.fmul(v(3).s4(), v(3).s4(), v(24).s4());
        a.fmul(v(3).s4(), v(3).s4(), v(5).s4());
        a.str(q(2), mem(X22, (ch * 32) as i64));
        a.str(q(3), mem(X22, (ch * 32 + 16) as i64));
    }
    add_wide(&mut a, X20, X20, hb);
    add_wide(&mut a, X22, X22, qrow);
    a.subs(X19, X19, 1);
    a.bne(qrows);

    // The dots, by groups of four rows: x20 = q' of the group, x21 = its dots, x22 = its key 0, x23
    // = the relative key of a = window − 1, x19 = groups left; per chunk x10 = the key pointer,
    // x11 = the q' one, x16 = chunks left.
    a.ldr(X19, mem(SP, g_n));
    a.add(X19, X19, (G - 1) as i64);
    a.lsr(X19, X19, 2);
    add_wide(&mut a, X20, SP, oq);
    add_wide(&mut a, X21, SP, oa);
    a.ldr(X22, mem(SP, g_k));
    a.ldr(X23, mem(SP, g_rel));
    let tiles: Vec<Tile> = (0..(G + window - 1))
        .step_by(KG)
        .map(Tile::Keys)
        .chain((0..window).step_by(KG).map(Tile::Rel))
        .collect();
    let groups = a.label();
    a.bind(groups);
    for tile in tiles {
        // Per register (r, kk): the key's offset from the base and the lane it lands in (at x21 +
        // …), if row r sees it.
        let lane = |r: usize, kk: usize| -> Option<(i64, i64)> {
            match tile {
                Tile::Keys(j0) => {
                    let j = j0 + kk;
                    let at = (r + window - 1).checked_sub(j).filter(|&at| at < window)?;
                    Some((j as i64 * hb, (r * 64 + at * 4) as i64))
                }
                Tile::Rel(a0) => {
                    let at = a0 + kk;
                    Some((
                        (window - 1 - at) as i64 * hb,
                        (rows * 64 + r * 64 + at * 4) as i64,
                    ))
                }
            }
        };
        let (n_k, base) = match tile {
            Tile::Keys(j0) => (KG.min(G + window - 1 - j0), X22),
            Tile::Rel(a0) => (KG.min(window - a0), X23),
        };
        let pairs: Vec<(usize, usize, i64)> = (0..G)
            .flat_map(|r| (0..n_k).map(move |kk| (r, kk)))
            .filter_map(|(r, kk)| lane(r, kk).map(|(_, l)| (r, kk, l)))
            .collect();
        for &(r, kk, _) in &pairs {
            a.movi(v((r * KG + kk) as u8).d2(), 0);
        }
        a.mov(X10, base);
        a.mov(X11, X20);
        a.mov_imm(X16, chunks as u64);
        let lanes = a.label();
        a.bind(lanes);
        for kk in 0..n_k {
            if let Some((off, _)) = (0..G).find_map(|r| lane(r, kk)) {
                let raw = (24 + kk) as u8;
                a.ldr(q(raw), mem(X10, off));
                widen(&mut a, (16 + 2 * kk) as u8, raw, false);
                widen(&mut a, (17 + 2 * kk) as u8, raw, true);
            }
        }
        for r in 0..G {
            if !pairs.iter().any(|&(pr, _, _)| pr == r) {
                continue;
            }
            a.ldr(q(29), mem(X11, r as i64 * qrow));
            a.ldr(q(30), mem(X11, r as i64 * qrow + 16));
            for &(_, kk, _) in pairs.iter().filter(|&&(pr, _, _)| pr == r) {
                let acc = (r * KG + kk) as u8;
                a.fmla(v(acc).s4(), v(29).s4(), v((16 + 2 * kk) as u8).s4());
                a.fmla(v(acc).s4(), v(30).s4(), v((17 + 2 * kk) as u8).s4());
            }
        }
        a.add(X10, X10, 16);
        a.add(X11, X11, 32);
        a.subs(X16, X16, 1);
        a.bne(lanes);
        for &(r, kk, l) in &pairs {
            let acc = (r * KG + kk) as u8;
            a.faddp(v(acc).s4(), v(acc).s4(), Some(v(acc).s4()));
            a.faddp(s(acc), v(acc).s2(), None);
            a.str(s(acc), mem(X21, l));
        }
    }
    add_wide(&mut a, X20, X20, G as i64 * qrow);
    a.add(X21, X21, G as i64 * 64);
    add_wide(&mut a, X22, X22, G as i64 * hb);
    a.subs(X19, X19, 1);
    a.bne(groups);

    // Per row: the logits `(q'·k) · k_scale + q'·rel`, then the row kernel's softmax; x20 = the
    // row's lanes, x21 = t, x9 = min(t + 1, window) the valid lanes.
    let k = ExpRegs::at(16);
    k.load(&mut a, X9);
    a.ldr(s(25), mem(SP, os + 4));
    a.dup(v(25).s4(), v(25).s(0));
    a.ldr(s(26), mem(SP, os + 8));
    a.dup(v(26).s4(), v(26).s(0));
    splat(&mut a, 27, (-2.0f32).to_bits(), X9);
    splat(&mut a, 28, 0x8000_0000, X9);
    a.ldr(X19, mem(SP, g_n));
    a.ldr(X21, mem(SP, g_t0));
    add_wide(&mut a, X20, SP, oa);
    a.mov_imm(X22, window as u64);
    let softmax = a.label();
    a.bind(softmax);
    a.add(X9, X21, 1);
    a.cmp(X9, X22);
    a.csel(X9, X22, X9, Cond::Hi);
    for i in 0..4i64 {
        a.ldr(q(0), mem(X20, i * 16));
        a.ldr(q(1), mem(X20, rows_i * 64 + i * 16));
        a.fmul(v(0).s4(), v(0).s4(), v(25).s4());
        a.fadd(v(0).s4(), v(0).s4(), v(1).s4());
        a.str(q(0), mem(X20, i * 16));
    }
    softmax16(&mut a, &k, X20, 0);
    a.add(X20, X20, 64);
    a.add(X21, X21, 1);
    a.subs(X19, X19, 1);
    a.bne(softmax);

    // y = Σ_a p_a · v_{t−a}, by groups of eight rows: x20 = the group's value 0, x21 = its
    // probabilities, x22 = its output, x23 = its rows left to store, x19 = groups left; per
    // chunk x10 = the value pointer, x11 = the output one, x16 = chunks left. Value j of the
    // group serves row r at a = r + window − 1 − j: j falling, a rises.
    a.ldr(X19, mem(SP, g_n));
    a.mov(X23, X19);
    a.add(X19, X19, (PG - 1) as i64);
    a.lsr(X19, X19, 3);
    a.ldr(X20, mem(SP, g_v));
    add_wide(&mut a, X21, SP, oa);
    a.ldr(X22, mem(SP, g_y));
    let out_groups = a.label();
    a.bind(out_groups);
    a.mov(X10, X20);
    a.mov(X11, X22);
    a.mov_imm(X16, chunks as u64);
    let lanes = a.label();
    a.bind(lanes);
    for r in 0..2 * PG {
        a.movi(v(r as u8).d2(), 0);
    }
    let mut p = 0usize;
    for j in (0..PG + window - 1).rev() {
        a.ldr(q(18), mem(X10, j as i64 * hb));
        widen(&mut a, 16, 18, false);
        widen(&mut a, 17, 18, true);
        for r in 0..PG {
            let Some(at) = (r + window - 1).checked_sub(j).filter(|&at| at < window) else {
                continue;
            };
            let pr = (19 + p % 4) as u8;
            p += 1;
            a.ldr(s(pr), mem(X21, (r * 64 + at * 4) as i64));
            a.fmla(v((2 * r) as u8).s4(), v(16).s4(), v(pr).s(0));
            a.fmla(v((2 * r + 1) as u8).s4(), v(17).s4(), v(pr).s(0));
        }
    }
    let stored = a.label();
    for r in 0..PG {
        if r > 0 {
            a.cmp(X23, r as i64);
            a.bls(stored);
        }
        narrow(&mut a, 24, (2 * r) as u8, (2 * r + 1) as u8);
        a.str(q(24), mem(X11, r as i64 * hb));
    }
    a.bind(stored);
    a.add(X10, X10, 16);
    a.add(X11, X11, 16);
    a.subs(X16, X16, 1);
    a.bne(lanes);
    add_wide(&mut a, X20, X20, PG as i64 * hb);
    a.add(X21, X21, PG as i64 * 64);
    add_wide(&mut a, X22, X22, PG as i64 * hb);
    a.sub(X23, X23, PG as i64);
    a.subs(X19, X19, 1);
    a.bne(out_groups);

    a.bind(exit);
    for part in [rest, pages] {
        if part > 0 {
            a.add(SP, SP, part);
        }
    }
    frame.leave(&mut a);
    a.ret();
    a.finish().map(|c| c.bytes)
}
