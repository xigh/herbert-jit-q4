//! The GEMM micro-kernel for **int8 or 4-bit weights and int8 activations**, x86-64 with AVX-VNNI
//! (`vpdpbusd` on `ymm`, VEX-encoded: no AVX-512 in it), or AVX2 alone (`vpdpbusd` emulated,
//! below), or AVX-512 VNNI (`zmm`, below). The layouts are those of `jit/int8.rs`.
//!
//! `vpdpbusd acc, u, s` multiplies groups of four unsigned bytes of `u` with four signed bytes of
//! `s` and adds them into 32-bit lanes. The weights are the unsigned operand, offset to unsigned at
//! packing (int8: `q + 128`; Q4: `q + 8`, two codes per byte), the activations the signed one,
//! broadcast four bytes at a time. What the offset adds, `offset · Σ x` over the group, is removed
//! with the int32 sum of the codes of the group that the quantizing node stored after the f32 scale
//! (`offset = 128` or `8`: a shift by 7 or 3).
//!
//! Per group of 32 values of `k` and per row of the tile (sixteen columns = two `ymm` of int32
//! lanes): eight steps of four `k` (Q4: four steps of eight, the low nibbles then the high ones,
//! `vpand 0x0f` and `vpsrlw 4` + `vpand`), then `acc − offset·Σx`, `vcvtdq2ps`, `· scale_w ·
//! scale_a` and `vfmadd231ps` into the f32 accumulators.
//!
//! ## Registers
//!
//! f32 accumulators `ymm0..ymm3`, int32 accumulators `ymm4..ymm7`; `W` `ymm8`/`ymm9`; the broadcast
//! of `A` `ymm10`; Q4: the nibble mask `ymm11`, the high nibbles `ymm12`/`ymm13`. In the epilogue
//! of a group: the scales of `W` in `ymm8`/`ymm9`, temporaries `ymm10`, `ymm12`, `ymm13`.
//! GPRs: `rdi` row 0 of `A`, `r8` row 1, `r10`/`r11` their metadata, `rsi` the block of `W`, `rdx`
//! `C`, `rcx` the groups.
//!
//! ## AVX2 without VNNI (Zen 2, Zen 3, Intel Haswell to Comet Lake): the same kernel, `vpdpbusd` emulated
//!
//! The same rows, columns, registers and epilogue, each `vpdpbusd acc, u, a` replaced by AVX2
//! (selected when the machine has no AVX-VNNI: [`YmmDot`]). `vpmaddubsw p, u, a` multiplies the
//! unsigned bytes of `u` by the signed bytes of `a` and adds adjacent pairs into 16-bit lanes,
//! **saturating** to `i16`; `vpmaddwd p, p, ones` adds the two halves of a column into its 32-bit
//! lane.
//!
//! * **Q4** ([`YmmDot::Avx2Words`], the default): the codes are `≤ 15` and the activations `|a| ≤
//!   127` (`execution/int8.rs`), so a pair is at most `2·15·127 = 3 810` and the eight steps of a
//!   group at most `16·15·127 = 30 480 < 32 767`: each step is `vpmaddubsw` + `vpaddw` into
//!   **16-bit** accumulators, and one `vpmaddwd(ones)` per group and accumulator widens them before
//!   the epilogue. Nothing saturates, nothing wraps: a group's int32 sums are `vpdpbusd`'s, the
//!   epilogue the same instructions, the output the same bits (test `q4_ymm_avx2_bits`).
//!   `HERBERT_Q4_AVX2_I32=1` takes the three-instruction step below instead (the control: the same
//!   bits).
//! * **int8** ([`YmmDot::Avx2`]): `vpmaddubsw` + `vpmaddwd(ones)` + `vpaddd` per step. A code `q +
//!   128` reaches 255 and `2·255·127` exceeds 32 767: the pair saturates, as in the SSE4.2 kernel —
//!   **approximate**, kept for `--int8` without VNNI, not the bits of `vpdpbusd`.
//!
//! Registers: those of the VNNI kernel, plus `ones` (sixteen-bit 1) in `ymm14` and the product in
//! `ymm15`.
//!
//! ## AVX-512 (`avx512vnni` and `avx512bw`): up to [`ZMM_ROWS`] rows
//!
//! The same computation on `zmm`: the sixteen columns of a step are one 64-byte load (lane `c` =
//! column `c`), the weights of a group unpacked once (eight `zmm`) for every row of the tile, `A`
//! broadcast from memory by `vpdpbusd`'s `{1to16}`. Per group and row: eight `vpdpbusd`, then the
//! offset removed, `vcvtdq2ps`, `scale_w · scale_a` (`vmulps` with the broadcast scale) and
//! `vfmadd231ps`: the integer sums and the f32 operations of the `ymm` kernel in the same order,
//! the same bits. Registers: f32 accumulators `zmm0..zmm9`, int32 accumulators `zmm10..zmm19`, the
//! weights `zmm20..zmm27`, the nibble mask `zmm28`, the activation offset `zmm29`, the weight
//! scales `zmm30`, the combined scale `zmm31` (the bf16 epilogue reuses `zmm29..zmm31`). GPRs:
//! `rdi` row 0 of `A` (row `i` at `i · lda`), `r10` its metadata, `rsi` `W`, `rdx` `C`, `rcx` the
//! groups.

use crate::core::error::{HerbertError, Result};
use crate::jit::emit::MicroKernelSpec;
use crate::jit::int8::{a_row_bytes, GROUP};
use crate::jit::ir::{Decline, ROWS_INTERLEAVED};
use iced_x86::code_asm::*;
use iced_x86::IcedError;

/// Rows of a tile of the `ymm` kernel.
pub const MAX_ROWS: usize = 2;
/// Rows of a tile of the `zmm` kernel at most (`2·rows + 12` registers).
pub const ZMM_ROWS: usize = 10;

/// Rows of a tile of the `zmm` kernel: 8 (`HERBERT_Q4_ROWS=N`, 1 to [`ZMM_ROWS`]: measurement).
pub fn zmm_rows() -> usize {
    std::env::var("HERBERT_Q4_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|r| (1..=ZMM_ROWS).contains(r))
        .unwrap_or(8)
}

/// Does this machine run the `zmm` kernel (`HERBERT_Q4_YMM=1`: the `ymm` one, the control)?
pub fn zmm() -> bool {
    let f = crate::core::isa::feature;
    f("avx512f") && f("avx512bw") && f("avx512vnni") && std::env::var_os("HERBERT_Q4_YMM").is_none()
}

/// How the `ymm` kernel forms its four-byte dot products (module documentation).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum YmmDot {
    /// `vpdpbusd` (AVX-VNNI).
    Vnni,
    /// AVX2: `vpmaddubsw`, `vpmaddwd(ones)`, `vpaddd` per step (int8: saturates; Q4: exact).
    Avx2,
    /// AVX2, Q4 only: `vpmaddubsw`, `vpaddw` per step into 16-bit lanes, one `vpmaddwd(ones)` per
    /// group (exact).
    Avx2Words,
}

/// The `ymm` kernel's dot product on this machine: `vpdpbusd` with AVX-VNNI, else AVX2
/// (`HERBERT_Q4_AVX2_I32=1`: Q4 with the three-instruction step, the control).
pub fn ymm_dot(q4: bool) -> YmmDot {
    if crate::core::isa::feature("avxvnni") {
        YmmDot::Vnni
    } else if q4 && std::env::var_os("HERBERT_Q4_AVX2_I32").is_none() {
        YmmDot::Avx2Words
    } else {
        YmmDot::Avx2
    }
}

/// Can this machine run this module's kernels: AVX-512 VNNI, AVX-VNNI, or AVX2 + FMA (the engine's
/// floor)?
pub fn available() -> bool {
    let f = crate::core::isa::feature;
    zmm() || f("avxvnni") || (f("avx2") && f("fma"))
}

/// The kernel this machine runs, for the startup log.
pub fn name(q4: bool) -> &'static str {
    if zmm() {
        "zmm AVX-512 VNNI"
    } else {
        match ymm_dot(q4) {
            YmmDot::Vnni => "ymm AVX-VNNI",
            YmmDot::Avx2 => "ymm AVX2 (vpmaddubsw + vpmaddwd per step)",
            YmmDot::Avx2Words => "ymm AVX2 (vpmaddubsw, 16-bit sums per group)",
        }
    }
}

/// Rows of a tile on this machine.
pub fn max_rows() -> usize {
    if zmm() {
        zmm_rows()
    } else {
        MAX_ROWS
    }
}

const YMM: [AsmRegisterYmm; 16] = [
    ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14,
    ymm15,
];
const XMM: [AsmRegisterXmm; 16] = [
    xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14,
    xmm15,
];

/// What this kernel emits: quantized `A`, sixteen full columns, one or two rows, `K` a multiple of
/// 32.
pub fn check(spec: &MicroKernelSpec) -> std::result::Result<(), Decline> {
    if spec.nr != ROWS_INTERLEAVED || spec.mr == 0 || spec.mr > max_rows() {
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
    check(spec).map_err(|d| HerbertError::Backend(format!("backend_jit int8 x86: {d}")))?;
    assemble(spec).map_err(|e| {
        HerbertError::Backend(format!(
            "backend_jit: int8 x86 emission {}x16 (K={}): {e}",
            spec.mr,
            2 * spec.k_pairs
        ))
    })
}

/// The kernel of `spec` on `zmm` or on `ymm` (`mr` within that kernel's rows), whatever this
/// machine selects.
#[cfg(test)]
pub(crate) fn emit_with(spec: &MicroKernelSpec, zmm: bool) -> Result<Vec<u8>> {
    let r = if zmm {
        assemble_zmm(spec)
    } else {
        assemble_ymm(spec, ymm_dot(spec.opts.neon.w_q4))
    };
    r.map_err(|e| {
        HerbertError::Backend(format!(
            "backend_jit: int8 x86 emission {}x16 (K={}): {e}",
            spec.mr,
            2 * spec.k_pairs
        ))
    })
}

/// The `ymm` kernel of `spec` with the dot product `dot`, whatever this machine selects.
#[cfg(test)]
pub(crate) fn emit_ymm(spec: &MicroKernelSpec, dot: YmmDot) -> Result<Vec<u8>> {
    assemble_ymm(spec, dot).map_err(|e| {
        HerbertError::Backend(format!(
            "backend_jit: int8 x86 emission {}x16 (K={}): {e}",
            spec.mr,
            2 * spec.k_pairs
        ))
    })
}

fn assemble(spec: &MicroKernelSpec) -> std::result::Result<Vec<u8>, IcedError> {
    if zmm() {
        assemble_zmm(spec)
    } else {
        assemble_ymm(spec, ymm_dot(spec.opts.neon.w_q4))
    }
}

fn assemble_ymm(spec: &MicroKernelSpec, dot: YmmDot) -> std::result::Result<Vec<u8>, IcedError> {
    let mr = spec.mr;
    let k = 2 * spec.k_pairs;
    let groups = k / GROUP;
    let q4 = spec.opts.neon.w_q4;
    // 16-bit sums only where they cannot saturate (module documentation): Q4.
    let dot = if dot == YmmDot::Avx2Words && !q4 {
        YmmDot::Avx2
    } else {
        dot
    };
    let (ones, p) = (YMM[14], YMM[15]);
    let ldc = spec.ldc_bytes;
    let facc = |i: usize, j: usize| YMM[2 * i + j];
    let iacc = |i: usize, j: usize| YMM[4 + 2 * i + j];
    let (w0, w1, t) = (YMM[8], YMM[9], YMM[10]);
    let (mask, h0, h1) = (YMM[11], YMM[12], YMM[13]);
    let rows = [rdi, r8];
    let metas = [r10, r11];
    let mut asm = CodeAssembler::new(64)?;
    asm.set_prefer_vex(true);
    if mr > 1 {
        asm.lea(r8, qword_ptr(rdi + spec.lda_bytes as i32))?;
    }
    for i in 0..mr {
        asm.lea(metas[i], qword_ptr(rows[i] + k as i32))?;
        for j in 0..2 {
            asm.vxorps(facc(i, j), facc(i, j), facc(i, j))?;
        }
    }
    if q4 {
        asm.mov(eax, 0x0f0f_0f0fu32)?;
        asm.vmovd(XMM[11], eax)?;
        asm.vpbroadcastd(mask, XMM[11])?;
    }
    if dot != YmmDot::Vnni {
        asm.mov(eax, 0x0001_0001u32)?;
        asm.vmovd(XMM[14], eax)?;
        asm.vpbroadcastd(ones, XMM[14])?;
    }
    let mut top = asm.create_label();
    asm.mov(ecx, groups as u32)?;
    asm.set_label(&mut top)?;
    for i in 0..mr {
        for j in 0..2 {
            asm.vpxor(iacc(i, j), iacc(i, j), iacc(i, j))?;
        }
    }
    // One step: `u0`/`u1` the weights of rows 0..7 and 8..15, `step` the index of the four bytes of
    // `A`.
    let step = |asm: &mut CodeAssembler,
                u0: AsmRegisterYmm,
                u1: AsmRegisterYmm,
                step: usize|
     -> std::result::Result<(), IcedError> {
        for i in 0..mr {
            asm.vpbroadcastd(t, dword_ptr(rows[i] + (4 * step) as i32))?;
            for (j, u) in [(0, u0), (1, u1)] {
                match dot {
                    YmmDot::Vnni => asm.vpdpbusd(iacc(i, j), u, t)?,
                    YmmDot::Avx2 => {
                        asm.vpmaddubsw(p, u, t)?;
                        asm.vpmaddwd(p, p, ones)?;
                        asm.vpaddd(iacc(i, j), iacc(i, j), p)?;
                    }
                    YmmDot::Avx2Words => {
                        asm.vpmaddubsw(p, u, t)?;
                        asm.vpaddw(iacc(i, j), iacc(i, j), p)?;
                    }
                }
            }
        }
        Ok(())
    };
    if q4 {
        for tt in 0..4usize {
            asm.vmovdqu(w0, ymmword_ptr(rsi + (tt * 64) as i32))?;
            asm.vmovdqu(w1, ymmword_ptr(rsi + (tt * 64 + 32) as i32))?;
            asm.vpsrlw(h0, w0, 4u32)?;
            asm.vpsrlw(h1, w1, 4u32)?;
            asm.vpand(h0, h0, mask)?;
            asm.vpand(h1, h1, mask)?;
            asm.vpand(w0, w0, mask)?;
            asm.vpand(w1, w1, mask)?;
            step(&mut asm, w0, w1, 2 * tt)?;
            step(&mut asm, h0, h1, 2 * tt + 1)?;
        }
        asm.add(rsi, 256)?;
    } else {
        for s in 0..8usize {
            asm.vmovdqu(w0, ymmword_ptr(rsi + (s * 64) as i32))?;
            asm.vmovdqu(w1, ymmword_ptr(rsi + (s * 64 + 32) as i32))?;
            step(&mut asm, w0, w1, s)?;
        }
        asm.add(rsi, 512)?;
    }
    // The scales of `W`, then per row: remove the offset, int32 to f32, · scale of `A`, accumulate.
    asm.vmovups(w0, ymmword_ptr(rsi))?;
    asm.vmovups(w1, ymmword_ptr(rsi + 32))?;
    asm.add(rsi, 64)?;
    let (u, v) = (YMM[12], YMM[13]);
    for i in 0..mr {
        if dot == YmmDot::Avx2Words {
            for j in 0..2 {
                asm.vpmaddwd(iacc(i, j), iacc(i, j), ones)?;
            }
        }
        asm.vpbroadcastd(t, dword_ptr(metas[i] + 4))?;
        asm.vpslld(t, t, if q4 { 3u32 } else { 7u32 })?;
        for j in 0..2 {
            asm.vpsubd(iacc(i, j), iacc(i, j), t)?;
            asm.vcvtdq2ps(iacc(i, j), iacc(i, j))?;
        }
        asm.vbroadcastss(u, dword_ptr(metas[i]))?;
        asm.vmulps(v, w0, u)?;
        asm.vfmadd231ps(facc(i, 0), iacc(i, 0), v)?;
        asm.vmulps(v, w1, u)?;
        asm.vfmadd231ps(facc(i, 1), iacc(i, 1), v)?;
        asm.add(metas[i], 8)?;
        asm.add(rows[i], 32)?;
    }
    asm.dec(ecx)?;
    asm.jne(top)?;
    // Store: f32, or bf16 rounded to nearest even as `emit_gemm_avx2`.
    if spec.out_bf16 {
        let (ta, tb, tc) = (YMM[8], YMM[9], YMM[10]);
        asm.mov(eax, 1u32)?;
        asm.vmovd(XMM[9], eax)?;
        asm.vpbroadcastd(tb, XMM[9])?;
        asm.mov(eax, 0x7fffu32)?;
        asm.vmovd(XMM[10], eax)?;
        asm.vpbroadcastd(tc, XMM[10])?;
        for i in 0..mr {
            for j in 0..2 {
                let a = facc(i, j);
                asm.vpsrld(ta, a, 16u32)?;
                asm.vpand(ta, ta, tb)?;
                asm.vpaddd(ta, ta, a)?;
                asm.vpaddd(ta, ta, tc)?;
                asm.vpsrld(ta, ta, 16u32)?;
                asm.vextracti128(XMM[12], ta, 1)?;
                asm.vpackusdw(XMM[12], XMM[8], XMM[12])?;
                asm.vmovdqu(xmmword_ptr(rdx + (i * ldc + j * 16) as i32), XMM[12])?;
            }
        }
    } else {
        for i in 0..mr {
            for j in 0..2 {
                asm.vmovups(ymmword_ptr(rdx + (i * ldc + j * 32) as i32), facc(i, j))?;
            }
        }
    }
    asm.vzeroupper()?;
    asm.ret()?;
    asm.assemble(0)
}

const ZMM: [AsmRegisterZmm; 32] = [
    zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7, zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14,
    zmm15, zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23, zmm24, zmm25, zmm26, zmm27,
    zmm28, zmm29, zmm30, zmm31,
];

/// The `zmm` kernel (module documentation): `mr` rows up to [`ZMM_ROWS`], sixteen columns.
fn assemble_zmm(spec: &MicroKernelSpec) -> std::result::Result<Vec<u8>, IcedError> {
    let mr = spec.mr;
    let k = 2 * spec.k_pairs;
    let groups = k / GROUP;
    let q4 = spec.opts.neon.w_q4;
    let (ldc, lda) = (spec.ldc_bytes, spec.lda_bytes);
    let facc = |i: usize| ZMM[i];
    let iacc = |i: usize| ZMM[10 + i];
    let w = |s: usize| ZMM[20 + s];
    let (mask, t, ws, v) = (ZMM[28], ZMM[29], ZMM[30], ZMM[31]);
    let row = |i: usize, off: usize| (i * lda + off) as i32;
    let mut asm = CodeAssembler::new(64)?;
    asm.lea(r10, qword_ptr(rdi + k as i32))?;
    for i in 0..mr {
        asm.vpxord(facc(i), facc(i), facc(i))?;
    }
    if q4 {
        asm.mov(eax, 0x0f0f_0f0fu32)?;
        asm.vpbroadcastd(mask, eax)?;
    }
    let mut top = asm.create_label();
    asm.mov(ecx, groups as u32)?;
    asm.set_label(&mut top)?;
    for i in 0..mr {
        asm.vpxord(iacc(i), iacc(i), iacc(i))?;
    }
    if q4 {
        for tt in 0..4usize {
            asm.vmovdqu64(w(2 * tt), zmmword_ptr(rsi + (tt * 64) as i32))?;
            asm.vpsrlw(w(2 * tt + 1), w(2 * tt), 4u32)?;
            asm.vpandd(w(2 * tt), w(2 * tt), mask)?;
            asm.vpandd(w(2 * tt + 1), w(2 * tt + 1), mask)?;
        }
    } else {
        for st in 0..8usize {
            asm.vmovdqu64(w(st), zmmword_ptr(rsi + (st * 64) as i32))?;
        }
    }
    for st in 0..8usize {
        for i in 0..mr {
            asm.vpdpbusd(iacc(i), w(st), dword_bcst(rdi + row(i, 4 * st)))?;
        }
    }
    let wbytes = if q4 { 256 } else { 512 };
    asm.vmovups(ws, zmmword_ptr(rsi + wbytes))?;
    asm.add(rsi, wbytes + 64)?;
    for i in 0..mr {
        asm.vpbroadcastd(t, dword_ptr(r10 + row(i, 4)))?;
        asm.vpslld(t, t, if q4 { 3u32 } else { 7u32 })?;
        asm.vpsubd(iacc(i), iacc(i), t)?;
        asm.vcvtdq2ps(iacc(i), iacc(i))?;
        asm.vmulps(v, ws, dword_bcst(r10 + row(i, 0)))?;
        asm.vfmadd231ps(facc(i), iacc(i), v)?;
    }
    asm.add(r10, 8)?;
    asm.add(rdi, 32)?;
    asm.dec(ecx)?;
    asm.jne(top)?;
    if spec.out_bf16 {
        let (ta, one, round) = (ZMM[29], ZMM[30], ZMM[31]);
        asm.mov(eax, 1u32)?;
        asm.vpbroadcastd(one, eax)?;
        asm.mov(eax, 0x7fffu32)?;
        asm.vpbroadcastd(round, eax)?;
        for i in 0..mr {
            let a = facc(i);
            asm.vpsrld(ta, a, 16u32)?;
            asm.vpandd(ta, ta, one)?;
            asm.vpaddd(ta, ta, a)?;
            asm.vpaddd(ta, ta, round)?;
            asm.vpsrld(ta, ta, 16u32)?;
            asm.vpmovdw(ymmword_ptr(rdx + (i * ldc) as i32), ta)?;
        }
    } else {
        for i in 0..mr {
            asm.vmovups(zmmword_ptr(rdx + (i * ldc) as i32), facc(i))?;
        }
    }
    asm.vzeroupper()?;
    asm.ret()?;
    asm.assemble(0)
}
