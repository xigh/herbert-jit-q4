//! The GEMM micro-kernel for **int8 or 4-bit weights and int8 activations**, x86-64 restricted to
//! **SSE4.2** (no AVX at all): the SSE emulation of the AVX-VNNI kernel of `gemm_int8.rs`. The
//! engine itself requires AVX2 + FMA (`engine_isa`), so this kernel is never selected
//! automatically: it is reached only through `--q4-sse42` / `--int8-sse42`, as a comparison kernel.
//! The weight and activation layouts are **identical** to `gemm_int8.rs` (`jit/int8.rs`, x86
//! offset-binary): nothing changes in the packer, only how the dot product is computed.
//!
//! ## `vpdpbusd` emulated in SSE
//!
//! AVX-VNNI `vpdpbusd acc, u, s` does, per 32-bit lane, `acc += Σ_{4 bytes} u·s` (u unsigned, s
//! signed), with the accumulation in 32 bits. There is no SSE instruction that multiplies
//! unsigned×signed bytes straight into 32-bit lanes, so it is built in three steps over sixteen
//! bytes (= four columns of four `k`):
//! * `pmaddubsw t, a` (SSSE3): per pair of adjacent bytes `u0·a0 + u1·a1`, **signed-saturated to 16
//!   bits**;
//! * `pmaddwd t, ones` (SSE2, `ones` = sixteen-bit 1): sums the two 16-bit halves of each column
//!   into a 32-bit lane;
//! * `paddd iacc, t` accumulates.
//!
//! **Saturation caveat.** `pmaddubsw` saturates the intermediate `u0·a0 + u1·a1` to a signed 16-bit
//! value. With int8 weights the unsigned code reaches 255 and the product `255·127` already fills
//! most of the range, so the sum of two can exceed 32767 and **saturate** — the int8 SSE path is
//! therefore *approximate* (documented, not a bug). With 4-bit weights the unsigned code is at most
//! 15 (`q + 8`) and `|a| ≤ 127`, `2·15·127 = 3810 ≪ 32767`, so the Q4 path never saturates and
//! matches the scalar reference exactly (up to the f32 scaling). Q4 is the intended use.
//!
//! The epilogue is the one of `gemm_int8.rs` with SSE register widths: remove `offset·Σx` (`pslld`
//! by 7 int8 / 3 Q4 of the int32 code-sum the quantizer stored, then `psubd`), `cvtdq2ps`, `·
//! scale_w · scale_a`, and accumulate into the f32 accumulators with `mulps` + `addps` (no FMA on
//! an SSE4.2 target — not the bits of the `vfmadd231ps` of the VNNI kernel, but within the engine's
//! f32 tolerance). bf16 output is rounded to nearest even exactly as `gemm_int8.rs` /
//! `emit_gemm_avx2`, packed with `packusdw`.
//!
//! ## Registers (one row, sixteen columns = four `xmm` of int32 lanes)
//!
//! f32 accumulators `xmm0..xmm3`, int32 accumulators `xmm4..xmm7`; `ones` `xmm8`; int8: the
//! broadcast of `A` `xmm9`, a weight temporary `xmm10`. Q4: the nibble mask `xmm9`, the two `A`
//! broadcasts (low/high nibble) `xmm10`/`xmm11`, the weights `xmm12`/`xmm13`. The epilogue reuses
//! the free registers above the accumulators. GPRs: `rdi` the row of `A`, `r10` its metadata, `rsi`
//! the block of `W`, `rdx` `C`, `rcx` the groups.

use crate::core::error::{HerbertError, Result};
use crate::jit::emit::MicroKernelSpec;
use crate::jit::int8::{a_row_bytes, GROUP};
use crate::jit::ir::{Decline, ROWS_INTERLEAVED};
use iced_x86::code_asm::*;
use iced_x86::IcedError;

/// Rows of a tile: one. Two rows would need sixteen `xmm` for the accumulators alone.
pub const MAX_ROWS: usize = 1;

const XMM: [AsmRegisterXmm; 16] = [
    xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14,
    xmm15,
];

/// What this kernel emits: quantized `A`, sixteen full columns, exactly one row, `K` a multiple of
/// 32.
pub fn check(spec: &MicroKernelSpec) -> std::result::Result<(), Decline> {
    if spec.nr != ROWS_INTERLEAVED || spec.mr != MAX_ROWS {
        return Err(Decline::TileTooWide);
    }
    if spec.n_tail != ROWS_INTERLEAVED {
        return Err(Decline::PartialRowBlock);
    }
    let k = 2 * spec.k_pairs;
    if spec.k_odd
        || spec.k_dynamic
        || spec.opts.matrix
        || spec.a_fmt != crate::jit::emit::AFormat::Bf16
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
    check(spec).map_err(|d| HerbertError::Backend(format!("backend_jit int8 x86 sse4.2: {d}")))?;
    assemble(spec).map_err(|e| {
        HerbertError::Backend(format!(
            "backend_jit: int8 x86 sse4.2 emission 1x16 (K={}): {e}",
            2 * spec.k_pairs
        ))
    })
}

fn assemble(spec: &MicroKernelSpec) -> std::result::Result<Vec<u8>, IcedError> {
    let k = 2 * spec.k_pairs;
    let groups = k / GROUP;
    let q4 = spec.opts.neon.w_q4;
    let facc = |c: usize| XMM[c]; // xmm0..3
    let iacc = |c: usize| XMM[4 + c]; // xmm4..7
    let ones = XMM[8];
    let mut asm = CodeAssembler::new(64)?;
    asm.set_prefer_vex(false); // legacy SSE encodings, no VEX

    asm.lea(r10, qword_ptr(rdi + k as i32))?; // the metadata of A: after its k int8 values
    for c in 0..4 {
        asm.pxor(facc(c), facc(c))?;
    }
    // ones = sixteen-bit 1 in every word.
    asm.mov(eax, 0x0001_0001u32)?;
    asm.movd(ones, eax)?;
    asm.pshufd(ones, ones, 0u32)?;
    let mask = XMM[9];
    if q4 {
        asm.mov(eax, 0x0f0f_0f0fu32)?;
        asm.movd(mask, eax)?;
        asm.pshufd(mask, mask, 0u32)?;
    }

    let mut top = asm.create_label();
    asm.mov(ecx, groups as u32)?;
    asm.set_label(&mut top)?;
    for c in 0..4 {
        asm.pxor(iacc(c), iacc(c))?;
    }

    if q4 {
        let (tlo, thi, wlo, whi) = (XMM[10], XMM[11], XMM[12], XMM[13]);
        for tt in 0..4usize {
            // Low nibbles carry k = 8·tt .. 8·tt+3, high nibbles k = 8·tt+4 .. 8·tt+7.
            asm.movd(tlo, dword_ptr(rdi + (4 * (2 * tt)) as i32))?;
            asm.pshufd(tlo, tlo, 0u32)?;
            asm.movd(thi, dword_ptr(rdi + (4 * (2 * tt) + 4) as i32))?;
            asm.pshufd(thi, thi, 0u32)?;
            for c in 0..4usize {
                asm.movdqu(wlo, xmmword_ptr(rsi + (tt * 64 + c * 16) as i32))?;
                asm.movdqa(whi, wlo)?;
                asm.psrlw(whi, 4u32)?;
                asm.pand(whi, mask)?;
                asm.pand(wlo, mask)?;
                asm.pmaddubsw(wlo, tlo)?;
                asm.pmaddwd(wlo, ones)?;
                asm.paddd(iacc(c), wlo)?;
                asm.pmaddubsw(whi, thi)?;
                asm.pmaddwd(whi, ones)?;
                asm.paddd(iacc(c), whi)?;
            }
        }
        asm.add(rsi, 256)?;
    } else {
        let (t, w) = (XMM[9], XMM[10]); // int8 has no mask: xmm9 is the A broadcast
        for s in 0..8usize {
            asm.movd(t, dword_ptr(rdi + (4 * s) as i32))?;
            asm.pshufd(t, t, 0u32)?;
            for c in 0..4usize {
                asm.movdqu(w, xmmword_ptr(rsi + (s * 64 + c * 16) as i32))?;
                asm.pmaddubsw(w, t)?;
                asm.pmaddwd(w, ones)?;
                asm.paddd(iacc(c), w)?;
            }
        }
        asm.add(rsi, 512)?;
    }

    // Epilogue: offset·Σx removed, int32 → f32, · scale_w · scale_a, accumulate. (ones / mask must
    // survive.)
    let (toff, ua, vw) = (XMM[10], XMM[11], XMM[12]);
    asm.movd(toff, dword_ptr(r10 + 4))?; // int32 code-sum of the group
    asm.pshufd(toff, toff, 0u32)?;
    asm.pslld(toff, if q4 { 3u32 } else { 7u32 })?; // offset·sum (offset = 8 or 128)
    asm.movd(ua, dword_ptr(r10))?; // f32 scale of A (bit pattern)
    asm.pshufd(ua, ua, 0u32)?;
    for c in 0..4usize {
        asm.psubd(iacc(c), toff)?;
        asm.cvtdq2ps(iacc(c), iacc(c))?;
        asm.movups(vw, xmmword_ptr(rsi + (c * 16) as i32))?; // four f32 scales of W
        asm.mulps(vw, ua)?;
        asm.mulps(iacc(c), vw)?;
        asm.addps(facc(c), iacc(c))?;
    }
    asm.add(rsi, 64)?;
    asm.add(r10, 8)?;
    asm.add(rdi, 32)?;
    asm.dec(ecx)?;
    asm.jne(top)?;

    // One row: only the column offset is used, never `ldc_bytes`.
    if spec.out_bf16 {
        let (c1, c7, ta) = (XMM[9], XMM[10], XMM[11]);
        asm.mov(eax, 1u32)?;
        asm.movd(c1, eax)?;
        asm.pshufd(c1, c1, 0u32)?;
        asm.mov(eax, 0x7fffu32)?;
        asm.movd(c7, eax)?;
        asm.pshufd(c7, c7, 0u32)?;
        for c in 0..4usize {
            asm.movdqa(ta, facc(c))?;
            asm.psrld(ta, 16u32)?;
            asm.pand(ta, c1)?;
            asm.paddd(ta, facc(c))?;
            asm.paddd(ta, c7)?;
            asm.psrld(ta, 16u32)?;
            asm.packusdw(ta, ta)?; // four f32 → four bf16 in the low 64 bits
            asm.movq(qword_ptr(rdx + (c * 8) as i32), ta)?;
        }
    } else {
        for c in 0..4usize {
            asm.movups(xmmword_ptr(rdx + (c * 16) as i32), facc(c))?;
        }
    }
    asm.ret()?;
    asm.assemble(0)
}
