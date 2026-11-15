//! [`ElemOp::LocalAttention`] on x86-64: the chunked local attention of the
//! Gemma 4 audio tower, one query row and every head per call, in VEX `ymm`
//! (AVX2 + FMA), the same blob on both arms: the work of a row is twelve keys
//! of eight heads, small beside the tower's GEMMs.
//!
//! Per head: `q' = q · q_scale · sp` widened into a stack row of f32; the
//! `window` logits `(q'·k_{t−a}) · k_scale + q'·rel[window − a]` (two chains
//! of fused products, summed horizontally) into sixteen stack lanes; the
//! softcap `cap · tanh(l / cap)` on them, `tanh` by the exponential of the
//! elementary kernels (`tanh(u) = sign(u) · (1 − e^{−2|u|}) / (1 + e^{−2|u|})`);
//! the lanes `a ≥ min(t + 1, window)` masked; the softmax; then
//! `Σ_a p_a · v_{t−a}` rounded to bf16 by hand (round to nearest even, as the
//! AVX2 arm of the elementary kernels does). The keys and values of the rows
//! before row 0 are read (zeros the tower keeps there) and weigh zero.
//!
//! The block kernel (`rows > 0`, `assemble_block`): one head of a tile of
//! rows per call, the same values bit for bit. `q'` of the tile on the stack;
//! the logits by register tiles of four rows × three keys (twelve chains, each
//! the row kernel's: eight lanes, chunks in order, the same horizontal sum),
//! each key chunk widened once for the four rows, the relative keys the same
//! way (shared by every row: a small GEMM); the softmax of each row by the row
//! kernel's code; then `Σ_a p_a · v_{t−a}` by groups of eight rows, each value
//! chunk widened once for the rows that see it, summed per row in the row
//! kernel's order (`a` rising).

use crate::core::error::{HerbertError, Result};
use crate::jit::emit_elem::{ElemOp, ElemSpec};
use crate::jit::exp_constants::{
    EXP_C0, EXP_C1, EXP_C2, EXP_C3, EXP_C4, EXP_LOG2E, EXP_NEG87, EXP_ONE_BITS,
};
use iced_x86::code_asm::*;
use iced_x86::IcedError;

const YMM: [AsmRegisterYmm; 16] = [
    ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14,
    ymm15,
];
const ZMM: [AsmRegisterZmm; 32] = [
    zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7, zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14,
    zmm15, zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23, zmm24, zmm25, zmm26, zmm27,
    zmm28, zmm29, zmm30, zmm31,
];
const XMM: [AsmRegisterXmm; 16] = [
    xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14,
    xmm15,
];

/// The constants, one broadcast `ymm` slot each, in the stack frame.
mod slot {
    pub const NEG87: i32 = 0;
    pub const LOG2E: i32 = 1;
    pub const C4: i32 = 2;
    pub const C3: i32 = 3;
    pub const C2: i32 = 4;
    pub const C1: i32 = 5;
    pub const C0: i32 = 6;
    pub const ONE_BITS: i32 = 7;
    pub const ABS: i32 = 8;
    pub const SGN: i32 = 9;
    pub const NEG2: i32 = 10;
    pub const ONE_I32: i32 = 11;
    pub const BIAS: i32 = 12;
    pub const N: i32 = 13;
}

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
            "emit_local_attn: not a local attention".into(),
        ));
    };
    let (heads, window) = (usize::from(heads), usize::from(window));
    if rows > 0 {
        let zmm = crate::jit::elem::select_isa() == Some(crate::jit::elem::ElemIsa::Zmm);
        return assemble_block(spec.dim, heads, window, usize::from(rows), zmm).map_err(|e| {
            HerbertError::Backend(format!(
                "backend_jit: emission local_attention_block dim={}: {e}",
                spec.dim
            ))
        });
    }
    assemble(spec.dim, heads, window).map_err(|e| {
        HerbertError::Backend(format!(
            "backend_jit: emission local_attention dim={}: {e}",
            spec.dim
        ))
    })
}

/// A broadcast constant slot of the frame (constants at `rsp`).
fn c(s: i32) -> AsmMemoryOperand {
    ymmword_ptr(rsp + s * 32)
}

/// The constants into their slots (`rax`, `ymm0` clobbered).
fn constants(a: &mut CodeAssembler) -> std::result::Result<(), IcedError> {
    for (s, bits) in [
        (slot::NEG87, EXP_NEG87.to_bits()),
        (slot::LOG2E, EXP_LOG2E.to_bits()),
        (slot::C4, EXP_C4.to_bits()),
        (slot::C3, EXP_C3.to_bits()),
        (slot::C2, EXP_C2.to_bits()),
        (slot::C1, EXP_C1.to_bits()),
        (slot::C0, EXP_C0.to_bits()),
        (slot::ONE_BITS, EXP_ONE_BITS),
        (slot::ABS, 0x7FFF_FFFF),
        (slot::SGN, 0x8000_0000),
        (slot::NEG2, (-2.0f32).to_bits()),
        (slot::ONE_I32, 1),
        (slot::BIAS, 0x7FFF),
    ] {
        a.mov(eax, bits)?;
        a.vmovd(xmm0, eax)?;
        a.vbroadcastss(ymm0, xmm0)?;
        a.vmovups(c(s), ymm0)?;
    }
    Ok(())
}

/// `io = exp(io)` for `io ≤ 88`, the chain of the elementary kernels; t1..t3 clobbered.
fn exp(
    a: &mut CodeAssembler,
    io: usize,
    t1: usize,
    t2: usize,
    t3: usize,
) -> std::result::Result<(), IcedError> {
    a.vmaxps(YMM[io], YMM[io], c(slot::NEG87))?;
    a.vmulps(YMM[t1], YMM[io], c(slot::LOG2E))?;
    a.vroundps(YMM[t2], YMM[t1], 0)?;
    a.vsubps(YMM[t1], YMM[t1], YMM[t2])?;
    a.vmovups(YMM[t3], c(slot::C4))?;
    a.vfmadd213ps(YMM[t3], YMM[t1], c(slot::C3))?;
    a.vfmadd213ps(YMM[t3], YMM[t1], c(slot::C2))?;
    a.vfmadd213ps(YMM[t3], YMM[t1], c(slot::C1))?;
    a.vfmadd213ps(YMM[t3], YMM[t1], c(slot::C0))?;
    a.vcvtps2dq(YMM[t2], YMM[t2])?;
    a.vpslld(YMM[t2], YMM[t2], 23)?;
    a.vpaddd(YMM[t2], YMM[t2], c(slot::ONE_BITS))?;
    a.vmulps(YMM[io], YMM[t3], YMM[t2])?;
    Ok(())
}

/// The f32 sum of the eight lanes of `r` into its low lane; `t` clobbered.
fn hsum(a: &mut CodeAssembler, r: usize, t: usize) -> std::result::Result<(), IcedError> {
    a.vextractf128(XMM[t], YMM[r], 1)?;
    a.vaddps(XMM[r], XMM[r], XMM[t])?;
    a.vhaddps(XMM[r], XMM[r], XMM[r])?;
    a.vhaddps(XMM[r], XMM[r], XMM[r])?;
    Ok(())
}

/// The lanes `r10..16` of the sixteen logits at `base + at` set to `bits` (`rax` clobbered).
fn mask(
    a: &mut CodeAssembler,
    base: AsmRegister64,
    at: i32,
    bits: u32,
) -> std::result::Result<(), IcedError> {
    let mut top = a.create_label();
    let mut done = a.create_label();
    a.mov(rax, r10)?;
    a.set_label(&mut top)?;
    a.cmp(rax, 16)?;
    a.jae(done)?;
    a.mov(dword_ptr(base + rax * 4 + at), bits)?;
    a.inc(rax)?;
    a.jmp(top)?;
    a.set_label(&mut done)?;
    Ok(())
}

/// The sixteen logits at `base + at` (lanes `r10..16` dropped) into the probabilities: `cap ·
/// tanh(l / cap)` (`cap` at `cap_at`, a frame slot), the softmax. `rax`, `ymm0..7`, `ymm12`,
/// `ymm13` clobbered.
fn softmax16(
    a: &mut CodeAssembler,
    base: AsmRegister64,
    at: i32,
    cap_at: AsmMemoryOperand,
) -> std::result::Result<(), IcedError> {
    // l = cap · tanh(l / cap).
    a.vbroadcastss(ymm13, cap_at)?;
    for half in 0..2 {
        let at = at + half * 32;
        a.vmovups(ymm0, ymmword_ptr(base + at))?;
        a.vdivps(ymm0, ymm0, ymm13)?;
        a.vandps(ymm1, ymm0, c(slot::ABS))?;
        a.vmulps(ymm1, ymm1, c(slot::NEG2))?;
        exp(a, 1, 5, 6, 7)?;
        a.vmovups(ymm4, c(slot::ONE_BITS))?;
        a.vsubps(ymm2, ymm4, ymm1)?;
        a.vaddps(ymm3, ymm4, ymm1)?;
        a.vdivps(ymm2, ymm2, ymm3)?;
        a.vandps(ymm0, ymm0, c(slot::SGN))?;
        a.vorps(ymm2, ymm2, ymm0)?;
        a.vmulps(ymm2, ymm2, ymm13)?;
        a.vmovups(ymmword_ptr(base + at), ymm2)?;
    }
    // The softmax over the valid lanes.
    mask(a, base, at, MASKED)?;
    a.vmovups(ymm0, ymmword_ptr(base + at))?;
    a.vmaxps(ymm0, ymm0, ymmword_ptr(base + (at + 32)))?;
    a.vextractf128(xmm1, ymm0, 1)?;
    a.vmaxps(xmm0, xmm0, xmm1)?;
    a.vpermilps(xmm1, xmm0, 0x4E)?;
    a.vmaxps(xmm0, xmm0, xmm1)?;
    a.vpermilps(xmm1, xmm0, 0xB1)?;
    a.vmaxps(xmm0, xmm0, xmm1)?;
    a.vbroadcastss(ymm12, xmm0)?;
    for half in 0..2 {
        let at = at + half * 32;
        a.vmovups(ymm0, ymmword_ptr(base + at))?;
        a.vsubps(ymm0, ymm0, ymm12)?;
        exp(a, 0, 5, 6, 7)?;
        a.vmovups(ymmword_ptr(base + at), ymm0)?;
    }
    mask(a, base, at, 0)?;
    a.vmovups(ymm0, ymmword_ptr(base + at))?;
    a.vaddps(ymm0, ymm0, ymmword_ptr(base + (at + 32)))?;
    hsum(a, 0, 1)?;
    a.mov(eax, 1.0f32.to_bits())?;
    a.vmovd(xmm1, eax)?;
    a.vdivss(xmm0, xmm1, xmm0)?;
    a.vbroadcastss(ymm12, xmm0)?;
    for half in 0..2 {
        let at = at + half * 32;
        a.vmulps(ymm0, ymm12, ymmword_ptr(base + at))?;
        a.vmovups(ymmword_ptr(base + at), ymm0)?;
    }
    Ok(())
}

/// The f32 lanes of `src` rounded to bf16 (to nearest even, by hand) and stored at `dst` (sixteen
/// bytes); `t1`, `t2` clobbered.
fn round_store(
    a: &mut CodeAssembler,
    src: usize,
    t1: usize,
    t2: usize,
    dst: AsmMemoryOperand,
) -> std::result::Result<(), IcedError> {
    a.vpsrld(YMM[t1], YMM[src], 16)?;
    a.vpand(YMM[t1], YMM[t1], c(slot::ONE_I32))?;
    a.vpaddd(YMM[t1], YMM[t1], c(slot::BIAS))?;
    a.vpaddd(YMM[t1], YMM[t1], YMM[src])?;
    a.vpsrld(YMM[t1], YMM[t1], 16)?;
    a.vextracti128(XMM[t2], YMM[t1], 1)?;
    a.vpackusdw(XMM[t1], XMM[t1], XMM[t2])?;
    a.vmovdqu(dst, XMM[t1])?;
    Ok(())
}

/// The block kernel of `spec` on one arm (`zmm`: AVX-512, else AVX2), whatever the process
/// selected: the tests run both arms on an AVX-512 machine.
pub fn emit_block_arm(spec: &ElemSpec, zmm: bool) -> Result<Vec<u8>> {
    spec.check()
        .map_err(|d| HerbertError::Backend(format!("backend_jit local attention: {d}")))?;
    let ElemOp::LocalAttention {
        heads,
        window,
        rows,
    } = spec.op
    else {
        return Err(HerbertError::Backend(
            "emit_local_attn: not a local attention".into(),
        ));
    };
    if rows == 0 {
        return Err(HerbertError::Backend(
            "emit_local_attn: not a block kernel".into(),
        ));
    }
    assemble_block(
        spec.dim,
        usize::from(heads),
        usize::from(window),
        usize::from(rows),
        zmm,
    )
    .map_err(|e| {
        HerbertError::Backend(format!(
            "backend_jit: emission local_attention_block dim={}: {e}",
            spec.dim
        ))
    })
}

fn assemble(dim: usize, heads: usize, window: usize) -> std::result::Result<Vec<u8>, IcedError> {
    let mut a = CodeAssembler::new(64)?;
    a.set_prefer_vex(true);
    let db = (dim * 2) as i32;
    let hb = (heads * dim * 2) as i32;
    let chunks = (dim / 8) as i32;
    // The frame: the constants, q' (f32), the sixteen logits, the three scalars.
    let oq = slot::N * 32;
    let ol = oq + (dim * 4) as i32;
    let os = ol + 64;
    let frame = (os + 16 + 31) & !31;

    // rdi = q, rsi = k, rdx = v, rcx = rel, r8 = y, r9 = t; xmm0 = q_scale, xmm1 = k_scale, xmm2 =
    // cap.
    a.push(rbx)?;
    super::stack_reserve(&mut a, frame)?;
    a.vmovss(dword_ptr(rsp + os), xmm0)?;
    a.vmovss(dword_ptr(rsp + (os + 4)), xmm1)?;
    a.vmovss(dword_ptr(rsp + (os + 8)), xmm2)?;
    constants(&mut a)?;
    // r10 = min(t + 1, window): the valid lanes.
    a.lea(r10, ptr(r9 + 1))?;
    a.mov(eax, window as u32)?;
    a.cmp(r10, rax)?;
    a.cmova(r10, rax)?;
    // rbx = the relative keys of the head (rel + dim), r11 = heads left.
    a.lea(rbx, ptr(rcx + db))?;
    a.mov(r11, heads as u64)?;

    let mut head = a.create_label();
    a.set_label(&mut head)?;
    // q' = q · q_scale · sp, f32.
    a.vbroadcastss(ymm15, dword_ptr(rsp + os))?;
    for blk in 0..chunks {
        a.vpmovzxwd(ymm0, xmmword_ptr(rdi + blk * 16))?;
        a.vpslld(ymm0, ymm0, 16)?;
        a.vpmovzxwd(ymm1, xmmword_ptr(rcx + blk * 16))?;
        a.vpslld(ymm1, ymm1, 16)?;
        a.vmulps(ymm0, ymm0, ymm15)?;
        a.vmulps(ymm0, ymm0, ymm1)?;
        a.vmovups(ymmword_ptr(rsp + (oq + blk * 32)), ymm0)?;
    }
    // The logits, the lanes past the window zero.
    a.vxorps(ymm0, ymm0, ymm0)?;
    a.vmovups(ymmword_ptr(rsp + ol), ymm0)?;
    a.vmovups(ymmword_ptr(rsp + (ol + 32)), ymm0)?;
    a.vmovss(xmm14, dword_ptr(rsp + (os + 4)))?;
    for k in 0..window as i32 {
        a.vxorps(ymm2, ymm2, ymm2)?;
        a.vxorps(ymm3, ymm3, ymm3)?;
        let (koff, roff) = (-k * hb, (window as i32 - k) * hb);
        for blk in 0..chunks {
            a.vmovups(ymm0, ymmword_ptr(rsp + (oq + blk * 32)))?;
            a.vpmovzxwd(ymm1, xmmword_ptr(rsi + (koff + blk * 16)))?;
            a.vpslld(ymm1, ymm1, 16)?;
            a.vfmadd231ps(ymm2, ymm0, ymm1)?;
            a.vpmovzxwd(ymm1, xmmword_ptr(rbx + (roff + blk * 16)))?;
            a.vpslld(ymm1, ymm1, 16)?;
            a.vfmadd231ps(ymm3, ymm0, ymm1)?;
        }
        hsum(&mut a, 2, 4)?;
        hsum(&mut a, 3, 4)?;
        a.vmulss(xmm2, xmm2, xmm14)?;
        a.vaddss(xmm2, xmm2, xmm3)?;
        a.vmovss(dword_ptr(rsp + (ol + k * 4)), xmm2)?;
    }
    softmax16(&mut a, rsp, ol, dword_ptr(rsp + (os + 8)))?;
    // y = Σ_a p_a · v_{t−a}, rounded to bf16.
    for blk in 0..chunks {
        a.vxorps(ymm2, ymm2, ymm2)?;
        for k in 0..window as i32 {
            a.vbroadcastss(ymm0, dword_ptr(rsp + (ol + k * 4)))?;
            a.vpmovzxwd(ymm1, xmmword_ptr(rdx + (-k * hb + blk * 16)))?;
            a.vpslld(ymm1, ymm1, 16)?;
            a.vfmadd231ps(ymm2, ymm0, ymm1)?;
        }
        round_store(&mut a, 2, 3, 4, xmmword_ptr(r8 + blk * 16))?;
    }
    for r in [rdi, rsi, rdx, r8, rbx] {
        a.add(r, db)?;
    }
    a.dec(r11)?;
    a.jnz(head)?;

    a.add(rsp, frame)?;
    a.pop(rbx)?;
    a.vzeroupper()?;
    a.ret()?;
    a.assemble(0)
}

/// One register tile of the logits: four rows of a group against three keys (`Keys(j0)`: the keys
/// `j0..` of the group, row `r` sees key `j` at `a = r + window − 1 − j`) or three relative keys
/// (`Rel(a0)`: `a0..`).
#[derive(Clone, Copy)]
enum Tile {
    Keys(usize),
    Rel(usize),
}

/// The block kernel: one head of a tile of `rows` rows. `zmm` (AVX-512): the same bits with twice
/// the lanes per instruction — the logits by register tiles of eight rows × six keys, two rows in
/// the two halves of a `zmm` (each half one row's eight-lane chain; `q'` of the tile stored by
/// pairs of rows, chunk by chunk), the output sixteen lanes at a time (lanes are independent
/// there).
#[allow(non_snake_case)]
fn assemble_block(
    dim: usize,
    heads: usize,
    window: usize,
    rows: usize,
    zmm: bool,
) -> std::result::Result<Vec<u8>, IcedError> {
    let (G, KG) = if zmm {
        (8usize, 6usize)
    } else {
        (4usize, 3usize)
    }; // a register tile of the logits: rows, keys
    const PG: usize = 8; // rows of a group of the output
    let mut a = CodeAssembler::new(64)?;
    a.set_prefer_vex(true);
    let (w, r_) = (window as i32, rows as i32);
    let db = (dim * 2) as i32;
    let hb = (heads * dim * 2) as i32;
    let chunks = (dim / 8) as i32;
    let qrow = (dim * 4) as i32;
    // The frame: the constants, the three scalars, the pointers, q' of the tile (f32), then per row
    // the sixteen lanes of the dots with the keys (then the logits, then the probabilities),
    // then those with the relative keys.
    let os = slot::N * 32;
    let og = os + 32;
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
    let oa = oq + r_ * qrow;
    let ob = oa + r_ * 64;
    let frame = (ob + r_ * 64 + 31) & !31;
    let saved = [rbx, rbp, r12, r13, r14, r15];
    let mut exit = a.create_label();

    // rdi = call, rsi = task; xmm0 = q_scale, xmm1 = k_scale, xmm2 = cap.
    for r in saved {
        a.push(r)?;
    }
    super::stack_reserve(&mut a, frame)?;
    a.vmovss(dword_ptr(rsp + os), xmm0)?;
    a.vmovss(dword_ptr(rsp + (os + 4)), xmm1)?;
    a.vmovss(dword_ptr(rsp + (os + 8)), xmm2)?;
    constants(&mut a)?;
    // rax = tile, rdx = head; r9 = t0, r8 = n = min(rows, *call.rows − t0).
    a.mov(rax, rsi)?;
    a.xor(edx, edx)?;
    a.mov(ecx, heads as u32)?;
    a.div(rcx)?;
    a.mov(r8, qword_ptr(rdi + 40))?;
    a.mov(r8, qword_ptr(r8))?;
    a.imul_3(r9, rax, r_)?;
    a.sub(r8, r9)?;
    a.jle(exit)?;
    a.mov(ecx, rows as u32)?;
    a.cmp(r8, rcx)?;
    a.cmova(r8, rcx)?;
    a.mov(qword_ptr(rsp + g_n), r8)?;
    a.mov(qword_ptr(rsp + g_t0), r9)?;
    // r11 = the bytes of row t0, head `rdx`.
    a.imul_3(rdx, rdx, db)?;
    a.imul_3(r11, r9, hb)?;
    a.add(r11, rdx)?;
    for (field, at, back) in [
        (0, g_q, 0),
        (8, g_k, (w - 1) * hb),
        (16, g_v, (w - 1) * hb),
        (32, g_y, 0),
    ] {
        a.mov(rax, qword_ptr(rdi + field))?;
        a.add(rax, r11)?;
        if back != 0 {
            a.sub(rax, back)?;
        }
        a.mov(qword_ptr(rsp + at), rax)?;
    }
    // sp, and the relative key of position 1 (a = window − 1) of the head.
    a.mov(rax, qword_ptr(rdi + 24))?;
    a.mov(qword_ptr(rsp + g_sp), rax)?;
    a.lea(rax, ptr(rax + rdx + (db + hb)))?;
    a.mov(qword_ptr(rsp + g_rel), rax)?;
    // The dots zero, the lanes past the window kept so.
    a.vxorps(ymm0, ymm0, ymm0)?;
    for i in 0..(r_ * 4) {
        a.vmovups(ymmword_ptr(rsp + (oa + i * 32)), ymm0)?;
    }

    // q' = q · q_scale · sp, f32, for the rows of the register tiles (r12 = ⌈n / G⌉·G; with zmm, by
    // pairs of rows).
    a.mov(r12, qword_ptr(rsp + g_n))?;
    a.add(r12, (G - 1) as i32)?;
    a.and(r12, -(G as i32))?;
    if zmm {
        a.shr(r12, 1)?;
    }
    a.mov(rsi, qword_ptr(rsp + g_q))?;
    a.mov(rcx, qword_ptr(rsp + g_sp))?;
    a.lea(rbx, ptr(rsp + oq))?;
    a.vbroadcastss(ymm15, dword_ptr(rsp + os))?;
    let mut qrows = a.create_label();
    a.set_label(&mut qrows)?;
    // Row `h` of a pair, chunk `blk`: at `blk · 64 + h · 32` of the pair (zmm), else at `blk · 32`
    // of the row.
    let rows_per_step = if zmm { 2 } else { 1 };
    for h in 0..rows_per_step {
        for blk in 0..chunks {
            a.vpmovzxwd(ymm0, xmmword_ptr(rsi + (h * hb + blk * 16)))?;
            a.vpslld(ymm0, ymm0, 16)?;
            a.vpmovzxwd(ymm1, xmmword_ptr(rcx + blk * 16))?;
            a.vpslld(ymm1, ymm1, 16)?;
            a.vmulps(ymm0, ymm0, ymm15)?;
            a.vmulps(ymm0, ymm0, ymm1)?;
            let at = if zmm { blk * 64 + h * 32 } else { blk * 32 };
            a.vmovups(ymmword_ptr(rbx + at), ymm0)?;
        }
    }
    a.add(rsi, rows_per_step * hb)?;
    a.add(rbx, rows_per_step * qrow)?;
    a.sub(r12, 1)?;
    a.jnz(qrows)?;

    // The dots, by groups of four rows: r14 = q' of the group, r15 = its dots, r12 = its key 0, r13
    // = the relative key of a = window − 1, rbx = groups left.
    a.mov(rbx, qword_ptr(rsp + g_n))?;
    a.add(rbx, (G - 1) as i32)?;
    a.shr(rbx, G.trailing_zeros())?;
    a.lea(r14, ptr(rsp + oq))?;
    a.lea(r15, ptr(rsp + oa))?;
    a.mov(r12, qword_ptr(rsp + g_k))?;
    a.mov(r13, qword_ptr(rsp + g_rel))?;
    let tiles: Vec<Tile> = (0..(G + window - 1))
        .step_by(KG)
        .map(Tile::Keys)
        .chain((0..window).step_by(KG).map(Tile::Rel))
        .collect();
    let mut groups = a.create_label();
    a.set_label(&mut groups)?;
    for tile in tiles {
        // Per register (r, kk): the key's offset from the base and the lane it lands in (at r15 +
        // …), if row r sees it.
        let lane = |r: usize, kk: usize| -> Option<(i32, i32)> {
            match tile {
                Tile::Keys(j0) => {
                    let j = j0 + kk;
                    let at = (r + window - 1).checked_sub(j).filter(|&at| at < window)?;
                    Some((j as i32 * hb, (r * 64 + at * 4) as i32))
                }
                Tile::Rel(a0) => {
                    let at = a0 + kk;
                    Some((
                        (window - 1 - at) as i32 * hb,
                        (rows * 64 + r * 64 + at * 4) as i32,
                    ))
                }
            }
        };
        let (n_k, base) = match tile {
            Tile::Keys(j0) => (KG.min(G + window - 1 - j0), r12),
            Tile::Rel(a0) => (KG.min(window - a0), r13),
        };
        if zmm {
            dots_zmm(&mut a, &lane, n_k, base, qrow, db)?;
            continue;
        }
        let pairs: Vec<(usize, usize, i32)> = (0..G)
            .flat_map(|r| (0..n_k).map(move |kk| (r, kk)))
            .filter_map(|(r, kk)| lane(r, kk).map(|(_, l)| (r, kk, l)))
            .collect();
        for &(r, kk, _) in &pairs {
            let acc = r * KG + kk;
            a.vxorps(YMM[acc], YMM[acc], YMM[acc])?;
        }
        a.xor(eax, eax)?;
        let mut lanes = a.create_label();
        a.set_label(&mut lanes)?;
        for kk in 0..n_k {
            if let Some((off, _)) = (0..G).find_map(|r| lane(r, kk)) {
                a.vpmovzxwd(YMM[12 + kk], xmmword_ptr(base + rax + off))?;
                a.vpslld(YMM[12 + kk], YMM[12 + kk], 16)?;
            }
        }
        for &(r, kk, _) in &pairs {
            a.vfmadd231ps(
                YMM[r * KG + kk],
                YMM[12 + kk],
                ymmword_ptr(r14 + rax * 2 + r as i32 * qrow),
            )?;
        }
        a.add(eax, 16)?;
        a.cmp(eax, db)?;
        a.jb(lanes)?;
        for &(r, kk, l) in &pairs {
            let acc = r * KG + kk;
            hsum(&mut a, acc, 15)?;
            a.vmovss(dword_ptr(r15 + l), XMM[acc])?;
        }
    }
    a.add(r14, G as i32 * qrow)?;
    a.add(r15, G as i32 * 64)?;
    a.add(r12, G as i32 * hb)?;
    a.sub(rbx, 1)?;
    a.jnz(groups)?;

    // Per row: the logits `(q'·k) · k_scale + q'·rel`, then the row kernel's softmax; r13 = the
    // row's lanes, r14 = t.
    a.mov(rbx, qword_ptr(rsp + g_n))?;
    a.mov(r14, qword_ptr(rsp + g_t0))?;
    a.lea(r13, ptr(rsp + oa))?;
    let mut softmax = a.create_label();
    a.set_label(&mut softmax)?;
    a.lea(r10, ptr(r14 + 1))?;
    a.mov(eax, window as u32)?;
    a.cmp(r10, rax)?;
    a.cmova(r10, rax)?;
    a.vbroadcastss(ymm14, dword_ptr(rsp + (os + 4)))?;
    for half in 0..2 {
        a.vmovups(ymm0, ymmword_ptr(r13 + half * 32))?;
        a.vmulps(ymm0, ymm0, ymm14)?;
        a.vaddps(ymm0, ymm0, ymmword_ptr(r13 + (r_ * 64 + half * 32)))?;
        a.vmovups(ymmword_ptr(r13 + half * 32), ymm0)?;
    }
    softmax16(&mut a, r13, 0, dword_ptr(rsp + (os + 8)))?;
    a.add(r13, 64)?;
    a.add(r14, 1)?;
    a.sub(rbx, 1)?;
    a.jnz(softmax)?;

    // y = Σ_a p_a · v_{t−a}, by groups of eight rows: r12 = the group's value 0, r13 = its
    // probabilities, r14 = its output, r15 = its rows left to store, rbx = groups left. Value j
    // of the group serves row r at a = r + window − 1 − j: j falling, a rises for every row,
    // the row kernel's order.
    a.mov(rbx, qword_ptr(rsp + g_n))?;
    a.mov(r15, rbx)?;
    a.add(rbx, (PG - 1) as i32)?;
    a.shr(rbx, 3)?;
    a.mov(r12, qword_ptr(rsp + g_v))?;
    a.lea(r13, ptr(rsp + oa))?;
    a.mov(r14, qword_ptr(rsp + g_y))?;
    let mut out_groups = a.create_label();
    a.set_label(&mut out_groups)?;
    a.xor(eax, eax)?;
    let mut lanes = a.create_label();
    a.set_label(&mut lanes)?;
    for r in 0..PG {
        a.vxorps(YMM[r], YMM[r], YMM[r])?;
    }
    let mut p = 0usize;
    for j in (0..PG + window - 1).rev() {
        if zmm {
            a.vpmovzxwd(zmm8, ymmword_ptr(r12 + rax + j as i32 * hb))?;
            a.vpslld(zmm8, zmm8, 16)?;
        } else {
            a.vpmovzxwd(ymm8, xmmword_ptr(r12 + rax + j as i32 * hb))?;
            a.vpslld(ymm8, ymm8, 16)?;
        }
        for r in 0..PG {
            let Some(at) = (r + window - 1).checked_sub(j).filter(|&at| at < window) else {
                continue;
            };
            let pr = 9 + (p & 1);
            p += 1;
            if zmm {
                a.vbroadcastss(ZMM[pr], dword_ptr(r13 + (r * 64 + at * 4) as i32))?;
                a.vfmadd231ps(ZMM[r], ZMM[pr], zmm8)?;
            } else {
                a.vbroadcastss(YMM[pr], dword_ptr(r13 + (r * 64 + at * 4) as i32))?;
                a.vfmadd231ps(YMM[r], YMM[pr], ymm8)?;
            }
        }
    }
    let mut stored = a.create_label();
    for r in 0..PG {
        if r > 0 {
            a.cmp(r15, r as i32)?;
            a.jbe(stored)?;
        }
        if zmm {
            // The row kernel's rounding, lane by lane, on sixteen lanes; `vpmovdw` packs (every
            // value below 2¹⁶).
            a.vpsrld(zmm11, ZMM[r], 16)?;
            a.vpandd(zmm11, zmm11, dword_bcst(rsp + slot::ONE_I32 * 32))?;
            a.vpaddd(zmm11, zmm11, dword_bcst(rsp + slot::BIAS * 32))?;
            a.vpaddd(zmm11, zmm11, ZMM[r])?;
            a.vpsrld(zmm11, zmm11, 16)?;
            a.vpmovdw(ymmword_ptr(r14 + rax + r as i32 * hb), zmm11)?;
        } else {
            round_store(&mut a, r, 11, 12, xmmword_ptr(r14 + rax + r as i32 * hb))?;
        }
    }
    a.set_label(&mut stored)?;
    a.add(eax, if zmm { 32 } else { 16 })?;
    a.cmp(eax, db)?;
    a.jb(lanes)?;
    a.add(r12, PG as i32 * hb)?;
    a.add(r13, PG as i32 * 64)?;
    a.add(r14, PG as i32 * hb)?;
    a.sub(r15, PG as i32)?;
    a.sub(rbx, 1)?;
    a.jnz(out_groups)?;

    a.set_label(&mut exit)?;
    a.add(rsp, frame)?;
    for r in saved.iter().rev() {
        a.pop(*r)?;
    }
    a.vzeroupper()?;
    a.ret()?;
    a.assemble(0)
}

/// One register tile of the logits on the zmm arm: eight rows (four pairs, a pair's two rows in the
/// two halves of a register) against `n_k ≤ 6` keys at `base + key offset`; `lane(r, kk)` as in
/// [`assemble_block`]. Registers: the keys `zmm0..6` (each chunk of eight bf16 widened into both
/// halves), the accumulators `zmm8..32`, `ymm6` the broadcast chunk; `q'` of the group at `r14`
/// (pair `p`, chunk `c` at `p · 2·qrow + c · 64`), the dots at `r15`.
fn dots_zmm(
    a: &mut CodeAssembler,
    lane: &dyn Fn(usize, usize) -> Option<(i32, i32)>,
    n_k: usize,
    base: AsmRegister64,
    qrow: i32,
    db: i32,
) -> std::result::Result<(), IcedError> {
    const KG: usize = 6;
    let acc = |p: usize, kk: usize| 8 + p * KG + kk;
    // (pair, key): accumulated when either row of the pair sees the key.
    let regs: Vec<(usize, usize)> = (0..4)
        .flat_map(|p| (0..n_k).map(move |kk| (p, kk)))
        .filter(|&(p, kk)| lane(2 * p, kk).is_some() || lane(2 * p + 1, kk).is_some())
        .collect();
    for &(p, kk) in &regs {
        let r = acc(p, kk);
        a.vpxord(ZMM[r], ZMM[r], ZMM[r])?;
    }
    a.xor(eax, eax)?;
    let mut lanes = a.create_label();
    a.set_label(&mut lanes)?;
    for kk in 0..n_k {
        if let Some((off, _)) = (0..8).find_map(|r| lane(r, kk)) {
            a.vbroadcasti128(ymm6, xmmword_ptr(base + rax + off))?;
            a.vpmovzxwd(ZMM[kk], ymm6)?;
            a.vpslld(ZMM[kk], ZMM[kk], 16)?;
        }
    }
    for &(p, kk) in &regs {
        a.vfmadd231ps(
            ZMM[acc(p, kk)],
            ZMM[kk],
            zmmword_ptr(r14 + rax * 4 + p as i32 * 2 * qrow),
        )?;
    }
    a.add(eax, 16)?;
    a.cmp(eax, db)?;
    a.jb(lanes)?;
    // Each half summed as the row kernel sums its chain, in ymm1 (low row) and ymm2 (high row).
    for &(p, kk) in &regs {
        let r = acc(p, kk);
        let (lo, hi) = (lane(2 * p, kk), lane(2 * p + 1, kk));
        if let Some((_, l)) = lo {
            a.vextractf64x4(ymm1, ZMM[r], 0)?;
            hsum(a, 1, 3)?;
            a.vmovss(dword_ptr(r15 + l), xmm1)?;
        }
        if let Some((_, l)) = hi {
            a.vextractf64x4(ymm2, ZMM[r], 1)?;
            hsum(a, 2, 3)?;
            a.vmovss(dword_ptr(r15 + l), xmm2)?;
        }
    }
    Ok(())
}
