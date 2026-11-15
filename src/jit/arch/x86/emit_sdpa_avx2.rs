//! The decode attention (SDPA), AVX2 arm: one head, the three passes QK,
//! softmax and SV chained in a single blob, with the ABI and semantics of
//! `emit_sdpa` in the chained form. What changes is one fact: sixteen
//! registers instead of thirty-two, eight lanes instead of sixteen.
//!
//! What the port simplifies:
//!
//! 1. **No QK by groups**, so no `vcompressps` and no opmask: the QK is done position by position —
//!    a dot product, a horizontal reduction, a `vmovss`. Slower, and the only instructions of the
//!    AVX-512 kernel without an AVX2 equivalent are exactly the ones this avoids.
//! 2. **The SV accumulator lives on the stack.** At `head_dim = 256` the output is 32 vectors of 8
//!    f32, impossible in registers; everything spills, uniformly, and the round trip stays in L1.
//! 3. **One form, one exp.** The fused form already saturates thirty-two registers and the split
//!    form is not used by the decode driver: both decline by name here.
//!
//! What the port does not change: the summation order of the dot product
//! (four chains), the exact sequence of the softmax (maximum, `exp`, sum,
//! normalisation — a division without sink, a multiplication by the inverse
//! with one), the `-inf` padding that makes the sweeps mask-free, and the
//! coefficients of the exp polynomial (`exp_constants.rs`). Bit-for-bit
//! equality with the AVX-512 kernel is not reachable: the horizontal
//! reductions run over eight lanes instead of sixteen, so the additions
//! happen in another order; the two agree within a tolerance.

use crate::core::error::{HerbertError, Result};
use crate::jit::emit_sdpa::{SdpaKernelSpec, SdpaPrefillSpec, SDPA_PREFILL_MAX_ROWS};
use crate::jit::ir::{Decline, ExpStrategy, SdpaForm};

/// Can SDPA AVX2 run here? `avx2` + `fma`, the ISA floor.
pub fn is_supported() -> bool {
    crate::jit::engine_isa() != crate::jit::IsaEngine::BelowFloor
}

/// Is this shape served by the AVX2 kernel?
///
/// Refusals are **named**: an out-of-scope form must be seen, not
/// guessed from a false output.
pub fn check_avx2(spec: &SdpaKernelSpec) -> std::result::Result<(), Decline> {
    if spec.form != SdpaForm::Chained {
        return Err(Decline::OutOfScopeAvx2);
    }
    if spec.exp != ExpStrategy::Inline {
        return Err(Decline::OutOfScopeAvx2);
    }
    // Eight f32 per register: `head_dim` must divide evenly. All
    // served heads (64, 128, 256, 512) do.
    if spec.shape.head_dim == 0 || !spec.shape.head_dim.is_multiple_of(16) {
        return Err(Decline::PartialRowBlock);
    }
    Ok(())
}

/// Distance between two rows of scores in the block kernel's scratch, in
/// f32: the slot of `kv_capacity` positions rounded to sixteen, as
/// [`SdpaPrefillSpec::scratch_len`] sizes it on x86. A constant, so the rows
/// sit at immediate offsets. With a window, the block reads at most
/// `min(kv_capacity, window − 1 + rows)` keys, and a row's `-inf` padding,
/// counted from its own first key, reaches up to seven columns past its last
/// one: that length rounded to sixteen, plus sixteen. Without the causal mask
/// (`full`), the `n ≤ kv_capacity` keys every row sees; with the bidirectional
/// window, `N ≤ min(kv_capacity, 2·window − 2 + rows)` keys, plus sixteen. One
/// formula, [`SdpaPrefillSpec::scratch_len`]'s, divided by the rows.
pub fn block_scores_row(spec: &SdpaPrefillSpec) -> usize {
    spec.scratch_len() / spec.rows.max(1)
}

/// Is this block served by the AVX2 block kernel? The causal block, with or
/// without a sliding window, and the block without the causal mask (`full`),
/// with or without the bidirectional window; one exp, no QK group nor
/// pairing nor imposed dimension blocks: those are the AVX-512 kernel's axes,
/// and the decode kernel this block reproduces has none of them.
pub fn check_block_avx2(spec: &SdpaPrefillSpec) -> std::result::Result<(), Decline> {
    let hd = spec.shape.head_dim;
    if hd == 0 || !hd.is_multiple_of(16) {
        return Err(Decline::PartialRowBlock);
    }
    if spec.rows == 0 {
        return Err(Decline::Empty);
    }
    if spec.rows > SDPA_PREFILL_MAX_ROWS || spec.row_stride < hd || spec.out_row_stride < hd {
        return Err(Decline::PrefillRows);
    }
    if spec.exp != ExpStrategy::Inline
        || spec.qk_group != 0
        || spec.qk_pair > 1
        || spec.dim_blocks != 0
    {
        return Err(Decline::OutOfScopeAvx2);
    }
    // The window's bounds are 32-bit immediates and displacements (`2·window`
    // with the bidirectional one).
    if spec.window > i32::MAX as usize / 4 {
        return Err(Decline::PrefillRows);
    }
    // Every offset of the block is an immediate: rows of scores, of `q`, of
    // `out`, and the tail rows of `K` and `V`.
    let far = [
        spec.rows * block_scores_row(spec) * 4,
        spec.rows * spec.row_stride * 2,
        spec.rows * spec.out_row_stride * 2,
        spec.rows * hd * 2,
    ];
    if far.iter().any(|&b| b > i32::MAX as usize / 2) {
        return Err(Decline::PrefillRows);
    }
    Ok(())
}

pub use x86::{emit_sdpa_avx2, emit_sdpa_prefill_avx2};

mod x86 {
    use super::*;
    use iced_x86::code_asm::*;
    use iced_x86::IcedError;

    /// **Sixteen entries.** `ymm16..ymm31` exist only under AVX-512.
    const YMM: [AsmRegisterYmm; 16] = [
        ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12, ymm13,
        ymm14, ymm15,
    ];
    const XMM: [AsmRegisterXmm; 16] = [
        xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13,
        xmm14, xmm15,
    ];

    use crate::jit::exp_constants::{
        EXP_C0, EXP_C1, EXP_C2, EXP_C3, EXP_C4, EXP_LOG2E, EXP_NEG87, EXP_ONE_BITS,
    };

    /// Constant table slots, in 32-byte units.
    mod constant {
        pub const NEG87: i32 = 0;
        pub const LOG2E: i32 = 1;
        pub const C4: i32 = 2;
        pub const C3: i32 = 3;
        pub const C2: i32 = 4;
        pub const C1: i32 = 5;
        pub const C0: i32 = 6;
        pub const ONE_BITS: i32 = 7;
        /// `0xFFFF0000` — the upper half of each bf16 pair.
        pub const HI: i32 = 8;
        /// The integer `1`, for the least significant bit of the bf16 rounding.
        pub const ONE_I32: i32 = 9;
        /// `0x7FFF`, the half-ULP of bf16 rounding.
        pub const BIAS: i32 = 10;
        pub const N: i32 = 11;
    }

    /// Offset of the scale in the frame.
    const OFF_SCALE: i32 = 0;
    /// Four pointers resident: `v`, `out`, `scores`, `n`.
    const OFF_HOME: i32 = 64;
    /// Base of the constant table.
    const OFF_CST: i32 = 128;
    /// Base of the accumulator `SV`.
    const OFF_SV: i32 = OFF_CST + constant::N * 32;

    fn c(slot: i32) -> AsmMemoryOperand {
        ymmword_ptr(rsp + OFF_CST + slot * 32)
    }

    /// Frame size: the table, the home slots, and `nv` vectors of
    /// 8 f32 for the accumulator `SV`. Rounded to 64.
    fn frame_bytes(nv: usize) -> i32 {
        let raw = OFF_SV + (nv as i32) * 32;
        (raw + 63) & !63
    }

    fn store_const(asm: &mut CodeAssembler, slot: i32, bits: u32, t: usize) -> Result2 {
        asm.mov(eax, bits)?;
        asm.mov(dword_ptr(rsp + OFF_SCALE + 32), eax)?;
        asm.vpbroadcastd(YMM[t], dword_ptr(rsp + OFF_SCALE + 32))?;
        asm.vmovups(c(slot), YMM[t])?;
        Ok(())
    }

    type Result2 = std::result::Result<(), IcedError>;

    fn build_table(asm: &mut CodeAssembler, t: usize) -> Result2 {
        store_const(asm, constant::NEG87, EXP_NEG87.to_bits(), t)?;
        store_const(asm, constant::LOG2E, EXP_LOG2E.to_bits(), t)?;
        store_const(asm, constant::C4, EXP_C4.to_bits(), t)?;
        store_const(asm, constant::C3, EXP_C3.to_bits(), t)?;
        store_const(asm, constant::C2, EXP_C2.to_bits(), t)?;
        store_const(asm, constant::C1, EXP_C1.to_bits(), t)?;
        store_const(asm, constant::C0, EXP_C0.to_bits(), t)?;
        store_const(asm, constant::ONE_BITS, EXP_ONE_BITS, t)?;
        store_const(asm, constant::HI, 0xFFFF_0000u32, t)?;
        store_const(asm, constant::ONE_I32, 1u32, t)?;
        store_const(asm, constant::BIAS, 0x7FFFu32, t)?;
        Ok(())
    }

    /// `io = exp(io)` — the sequence of the AVX-512 `exp_inline`
    /// (`emit_sdpa/x86/base.rs`), constants in memory.
    ///
    /// Domain `io ≤ 88`. The softmax is only called with arguments `≤ 0`
    /// (the maximum has been subtracted), so the upper bound is not set here —
    /// exactly as in the AVX-512 kernel.
    fn exp_inline(asm: &mut CodeAssembler, io: usize, t1: usize, t2: usize, t3: usize) -> Result2 {
        asm.vmaxps(YMM[io], YMM[io], c(constant::NEG87))?;
        asm.vmulps(YMM[t1], YMM[io], c(constant::LOG2E))?;
        asm.vroundps(YMM[t2], YMM[t1], 0u32)?;
        asm.vsubps(YMM[t1], YMM[t1], YMM[t2])?;
        asm.vmovups(YMM[t3], c(constant::C4))?;
        asm.vfmadd213ps(YMM[t3], YMM[t1], c(constant::C3))?;
        asm.vfmadd213ps(YMM[t3], YMM[t1], c(constant::C2))?;
        asm.vfmadd213ps(YMM[t3], YMM[t1], c(constant::C1))?;
        asm.vfmadd213ps(YMM[t3], YMM[t1], c(constant::C0))?;
        asm.vcvtps2dq(YMM[t2], YMM[t2])?;
        asm.vpslld(YMM[t2], YMM[t2], 23u32)?;
        asm.vpaddd(YMM[t2], YMM[t2], c(constant::ONE_BITS))?;
        asm.vmulps(YMM[io], YMM[t3], YMM[t2])?;
        Ok(())
    }

    /// Horizontal sum of a ymm → scalar in `xmm[dst]`.
    fn hsum(asm: &mut CodeAssembler, dst: usize, t: usize) -> Result2 {
        asm.vextractf128(XMM[t], YMM[dst], 1)?;
        asm.vaddps(XMM[dst], XMM[dst], XMM[t])?;
        asm.vhaddps(XMM[dst], XMM[dst], XMM[dst])?;
        asm.vhaddps(XMM[dst], XMM[dst], XMM[dst])?;
        Ok(())
    }

    /// Maximum horizontal of a ymm → scalar in `xmm[dst]`.
    fn hmax(asm: &mut CodeAssembler, dst: usize, t: usize) -> Result2 {
        asm.vextractf128(XMM[t], YMM[dst], 1)?;
        asm.vmaxps(XMM[dst], XMM[dst], XMM[t])?;
        // Two permutations suffice for four lanes: (1,0,3,2) then
        // (2,3,0,1). `vhaddps` is not suitable — it adds.
        asm.vshufps(XMM[t], XMM[dst], XMM[dst], 0b10_11_00_01u32)?;
        asm.vmaxps(XMM[dst], XMM[dst], XMM[t])?;
        asm.vshufps(XMM[t], XMM[dst], XMM[dst], 0b01_00_11_10u32)?;
        asm.vmaxps(XMM[dst], XMM[dst], XMM[t])?;
        Ok(())
    }

    /// Widens 8 bf16 (16 bytes) to 8 f32.
    fn widen(asm: &mut CodeAssembler, dst: usize, at: AsmMemoryOperand) -> Result2 {
        asm.vpmovzxwd(YMM[dst], at)?;
        asm.vpslld(YMM[dst], YMM[dst], 16u32)?;
        Ok(())
    }

    /// `acc += a·b` on bf16 pairs — emulating `vdpbf16ps`, the same
    /// as in [`crate::jit::emit_gemm_avx2`].
    fn dp_bf16(
        asm: &mut CodeAssembler,
        acc: usize,
        a: usize,
        b: AsmMemoryOperand,
        ta: usize,
        tb: usize,
    ) -> Result2 {
        asm.vpslld(YMM[ta], YMM[a], 16u32)?;
        asm.vmovups(YMM[tb], b)?;
        asm.vpslld(YMM[tb], YMM[tb], 16u32)?;
        asm.vfmadd231ps(YMM[acc], YMM[ta], YMM[tb])?;
        asm.vpand(YMM[ta], YMM[a], c(constant::HI))?;
        asm.vmovups(YMM[tb], b)?;
        asm.vpand(YMM[tb], YMM[tb], c(constant::HI))?;
        asm.vfmadd231ps(YMM[acc], YMM[ta], YMM[tb])?;
        Ok(())
    }

    /// 8 f32 → 8 bf16 (16 bytes) in `xmm[dst]`, rounded to even.
    fn cvt_bf16(asm: &mut CodeAssembler, dst: usize, src: usize, t: usize) -> Result2 {
        asm.vpsrld(YMM[t], YMM[src], 16u32)?;
        asm.vpand(YMM[t], YMM[t], c(constant::ONE_I32))?;
        asm.vpaddd(YMM[t], YMM[t], c(constant::BIAS))?;
        asm.vpaddd(YMM[t], YMM[t], YMM[src])?;
        asm.vpsrld(YMM[t], YMM[t], 16u32)?;
        asm.vextracti128(XMM[dst], YMM[t], 1)?;
        asm.vpackusdw(XMM[dst], XMM[t], XMM[dst])?;
        Ok(())
    }

    /// The softmax of one row of scores, in place: `rdi` = the row (kept),
    /// `r11` = its blocks of eight (kept); `rsi`, `rcx`, `eax` and `ymm0..7`
    /// clobbered. One function for the decode kernel and the block kernel:
    /// the same instructions, hence the same bits.
    fn softmax(asm: &mut CodeAssembler, sink_on: bool) -> Result2 {
        const MAXV: usize = 0;
        const SUMV: usize = 1;
        const WORK: usize = 2;
        const T1: usize = 3;
        const T2: usize = 4;
        const T3: usize = 5;
        const RED: usize = 6;
        const TMP: usize = 7;
        // 1. maximum
        asm.mov(eax, f32::NEG_INFINITY.to_bits())?;
        asm.mov(dword_ptr(rsp + OFF_SCALE + 32), eax)?;
        asm.vpbroadcastd(YMM[MAXV], dword_ptr(rsp + OFF_SCALE + 32))?;
        asm.mov(rsi, rdi)?;
        asm.mov(rcx, r11)?;
        let mut max_top = asm.create_label();
        asm.set_label(&mut max_top)?;
        asm.vmovups(YMM[WORK], ymmword_ptr(rsi))?;
        asm.vmaxps(YMM[MAXV], YMM[MAXV], YMM[WORK])?;
        asm.add(rsi, 32)?;
        asm.dec(rcx)?;
        asm.jnz(max_top)?;
        asm.vmovups(YMM[RED], YMM[MAXV])?;
        hmax(asm, RED, TMP)?;
        if sink_on {
            asm.vmovss(XMM[TMP], dword_ptr(rsp + OFF_SCALE + 4))?;
            asm.vmaxss(XMM[RED], XMM[RED], XMM[TMP])?;
        }
        asm.vbroadcastss(YMM[MAXV], XMM[RED])?;

        // 2. exp and sum
        asm.vxorps(YMM[SUMV], YMM[SUMV], YMM[SUMV])?;
        asm.mov(rsi, rdi)?;
        asm.mov(rcx, r11)?;
        let mut exp_top = asm.create_label();
        asm.set_label(&mut exp_top)?;
        asm.vmovups(YMM[WORK], ymmword_ptr(rsi))?;
        asm.vsubps(YMM[WORK], YMM[WORK], YMM[MAXV])?;
        exp_inline(asm, WORK, T1, T2, T3)?;
        asm.vaddps(YMM[SUMV], YMM[SUMV], YMM[WORK])?;
        asm.vmovups(ymmword_ptr(rsi), YMM[WORK])?;
        asm.add(rsi, 32)?;
        asm.dec(rcx)?;
        asm.jnz(exp_top)?;
        // `RED` still holds the scalar maximum: the sum goes into `TMP`,
        // which no longer has a role here.
        asm.vmovups(YMM[TMP], YMM[SUMV])?;
        hsum(asm, TMP, T1)?;
        if sink_on {
            // The sink term enters the reference sum.
            asm.vmovss(XMM[T1], dword_ptr(rsp + OFF_SCALE + 4))?;
            asm.vsubss(XMM[T1], XMM[T1], XMM[RED])?;
            asm.vbroadcastss(YMM[WORK], XMM[T1])?;
            exp_inline(asm, WORK, T1, T2, T3)?;
            asm.vaddss(XMM[TMP], XMM[TMP], XMM[WORK])?;
        }

        // 3. normalization
        if sink_on {
            asm.mov(eax, 1.0f32.to_bits())?;
            asm.mov(dword_ptr(rsp + OFF_SCALE + 32), eax)?;
            asm.vmovss(XMM[T1], dword_ptr(rsp + OFF_SCALE + 32))?;
            asm.vdivss(XMM[TMP], XMM[T1], XMM[TMP])?;
        }
        asm.vbroadcastss(YMM[SUMV], XMM[TMP])?;
        asm.mov(rsi, rdi)?;
        asm.mov(rcx, r11)?;
        let mut norm_top = asm.create_label();
        asm.set_label(&mut norm_top)?;
        asm.vmovups(YMM[WORK], ymmword_ptr(rsi))?;
        if sink_on {
            asm.vmulps(YMM[WORK], YMM[WORK], YMM[SUMV])?;
        } else {
            asm.vdivps(YMM[WORK], YMM[WORK], YMM[SUMV])?;
        }
        asm.vmovups(ymmword_ptr(rsi), YMM[WORK])?;
        asm.add(rsi, 32)?;
        asm.dec(rcx)?;
        asm.jnz(norm_top)?;
        Ok(())
    }

    /// Generates the SDPA decode **kernel** described by `spec`, **in AVX2**.
    pub fn emit_sdpa_avx2(spec: &SdpaKernelSpec) -> Result<Vec<u8>> {
        check_avx2(spec)
            .map_err(|d| HerbertError::Backend(format!("backend_jit sdpa avx2: {d}")))?;
        assemble(spec).map_err(|e| {
            HerbertError::Backend(format!(
                "backend_jit: sdpa avx2 emission head_dim={}: {e}",
                spec.shape.head_dim
            ))
        })
    }

    fn assemble(spec: &SdpaKernelSpec) -> std::result::Result<Vec<u8>, IcedError> {
        // Vectors of **8** f32, and blocks of **16** bf16 (32 bytes).
        let nv = spec.shape.head_dim / 8;
        let nb16 = spec.shape.head_dim / 16;
        let row = spec.shape.kv_row_bytes() as i32;
        let sink_on = spec.shape.sink;

        let mut asm = CodeAssembler::new(64)?;
        // Without it, `iced` might choose an EVEX encoding.
        asm.set_prefer_vex(true);

        // ── prologue, ABI Chained ─────────────────────────────────────────
        // rdi=q, rsi=k, rdx=v, rcx=scores, r8=out, r9=n, xmm0=scale, xmm1=sink
        asm.push(rbp)?;
        asm.mov(rbp, rsp)?;
        asm.and(rsp, -64i32)?;
        crate::jit::arch::x86::stack_reserve(&mut asm, frame_bytes(nv))?;

        // The scale and the sink first: building the table clobbers
        // xmm0. The sink goes on the stack too — one fewer register to pin
        // across the three passes.
        asm.vmovss(dword_ptr(rsp + OFF_SCALE), XMM[0])?;
        if sink_on {
            asm.vmovss(dword_ptr(rsp + OFF_SCALE + 4), XMM[1])?;
        }
        asm.mov(qword_ptr(rsp + OFF_HOME), rdx)?; // v
        asm.mov(qword_ptr(rsp + OFF_HOME + 8), r8)?; // out
        asm.mov(qword_ptr(rsp + OFF_HOME + 16), rcx)?; // scores
        asm.mov(qword_ptr(rsp + OFF_HOME + 24), r9)?; // n
        build_table(&mut asm, 15)?;

        // ── pass 1: QK, position by position ───────────────────────────
        //
        // rdi=q (preserved), rsi=k (advanced), rdx=scores (advanced), rcx=n.
        // Four accumulation chains, like the AVX-512 kernel: it is
        // the summation order, it does not change lightly.
        {
            const A: [usize; 4] = [0, 1, 2, 3];
            const KV: usize = 4;
            const TA: usize = 5;
            const TB: usize = 6;
            let n_acc = 4.min(nb16.max(1));
            asm.mov(rdx, rcx)?; // scores
            asm.mov(rcx, r9)?; // n
            let mut top = asm.create_label();
            let mut after = asm.create_label();
            let mut pad_top = asm.create_label();
            let mut no_pad = asm.create_label();
            asm.test(rcx, rcx)?;
            asm.jz(after)?;
            asm.set_label(&mut top)?;
            for &a in &A[..n_acc] {
                asm.vxorps(YMM[a], YMM[a], YMM[a])?;
            }
            for b in 0..nb16 {
                asm.vmovdqu(YMM[KV], ymmword_ptr(rsi + (b * 32) as i32))?;
                dp_bf16(
                    &mut asm,
                    A[b % n_acc],
                    KV,
                    ymmword_ptr(rdi + (b * 32) as i32),
                    TA,
                    TB,
                )?;
            }
            // Tree: (0+1) + (2+3) when all four chains exist — this is
            // the order of the AVX-512 kernel.
            if n_acc == 4 {
                asm.vaddps(YMM[A[0]], YMM[A[0]], YMM[A[1]])?;
                asm.vaddps(YMM[A[2]], YMM[A[2]], YMM[A[3]])?;
                asm.vaddps(YMM[A[0]], YMM[A[0]], YMM[A[2]])?;
            } else {
                for &a in &A[1..n_acc] {
                    asm.vaddps(YMM[A[0]], YMM[A[0]], YMM[a])?;
                }
            }
            hsum(&mut asm, A[0], TA)?;
            asm.vmovss(XMM[TA], dword_ptr(rsp + OFF_SCALE))?;
            asm.vmulss(XMM[A[0]], XMM[A[0]], XMM[TA])?;
            asm.vmovss(dword_ptr(rdx), XMM[A[0]])?;
            asm.add(rsi, row)?;
            asm.add(rdx, 4)?;
            asm.dec(rcx)?;
            asm.jnz(top)?;

            // `-inf` padding up to a multiple of **8** — this is what makes
            // the three softmax passes maskless.
            asm.set_label(&mut after)?;
            asm.mov(rax, qword_ptr(rsp + OFF_HOME + 24))?;
            asm.and(rax, 7)?;
            asm.jz(no_pad)?;
            asm.mov(r11, 8u64)?;
            asm.sub(r11, rax)?;
            asm.set_label(&mut pad_top)?;
            asm.mov(dword_ptr(rdx), 0xFF80_0000u32)?;
            asm.add(rdx, 4)?;
            asm.dec(r11)?;
            asm.jnz(pad_top)?;
            asm.set_label(&mut no_pad)?;
        }

        // ── pass 2: softmax, three sweeps over ceil(n/8) blocks ────────
        {
            asm.mov(rdi, qword_ptr(rsp + OFF_HOME + 16))?; // scores
            asm.mov(r11, qword_ptr(rsp + OFF_HOME + 24))?; // n
            asm.add(r11, 7i32)?;
            asm.shr(r11, 3u32)?; // nb = ceil(n/8)

            softmax(&mut asm, sink_on)?;
        }

        // ── pass 3: SV, accumulator ENTIRELY on stack ────────────────
        {
            const SCORE: usize = 0;
            const VLIG: usize = 1;
            const ACC: usize = 2;
            const TMP: usize = 3;
            asm.mov(rdi, qword_ptr(rsp + OFF_HOME + 16))?; // scores
            asm.mov(rsi, qword_ptr(rsp + OFF_HOME))?; // v
            asm.mov(rdx, qword_ptr(rsp + OFF_HOME + 8))?; // out
            asm.mov(rcx, qword_ptr(rsp + OFF_HOME + 24))?; // n

            // Zeroing the accumulator.
            asm.vxorps(YMM[ACC], YMM[ACC], YMM[ACC])?;
            for v in 0..nv {
                asm.vmovups(ymmword_ptr(rsp + OFF_SV + (v as i32) * 32), YMM[ACC])?;
            }
            let mut top = asm.create_label();
            let mut done = asm.create_label();
            asm.test(rcx, rcx)?;
            asm.jz(done)?;
            asm.set_label(&mut top)?;
            asm.vbroadcastss(YMM[SCORE], dword_ptr(rdi))?;
            for v in 0..nv {
                // An output YMM covers 8 dimensions, i.e., **16 bytes** of
                // `V` in bf16.
                widen(&mut asm, VLIG, xmmword_ptr(rsi + (v * 16) as i32))?;
                asm.vmovups(YMM[ACC], ymmword_ptr(rsp + OFF_SV + (v as i32) * 32))?;
                asm.vfmadd231ps(YMM[ACC], YMM[VLIG], YMM[SCORE])?;
                asm.vmovups(ymmword_ptr(rsp + OFF_SV + (v as i32) * 32), YMM[ACC])?;
            }
            asm.add(rdi, 4)?;
            asm.add(rsi, row)?;
            asm.dec(rcx)?;
            asm.jnz(top)?;
            asm.set_label(&mut done)?;

            // Epilogue: the accumulation was in f32 in the scratch; what
            // goes to memory is bf16 — the engine's invariant.
            for v in 0..nv {
                asm.vmovups(YMM[ACC], ymmword_ptr(rsp + OFF_SV + (v as i32) * 32))?;
                cvt_bf16(&mut asm, VLIG, ACC, TMP)?;
                asm.vmovdqu(xmmword_ptr(rdx + (v * 16) as i32), XMM[VLIG])?;
            }
        }

        // ── epilogue ──────────────────────────────────────────────────────
        asm.mov(rsp, rbp)?;
        asm.pop(rbp)?;
        asm.vzeroupper()?;
        asm.ret()?;
        asm.assemble(0)
    }

    // ══ The block kernel: `rows` consecutive queries of one head ══════════
    //
    // ABI of `SdpaPrefillFn`: rdi=q, rsi=k, rdx=v, rcx=scores, r8=out,
    // r9=past, xmm0=scale, xmm1=sink (and, with `full`, r9=n, or the
    // bidirectional window's `past` with rax=end, below). Row `i` sees
    // `past + i + 1` keys (the last `window` of them with a window); its
    // scores sit at `scores + i · block_scores_row` (the slot of
    // `scratch_len`).
    //
    // Each row goes through exactly the decode kernel's sequence: the same
    // four QK chains over the same blocks, the same tree and horizontal sum,
    // the scale by `vmulss`, the `-inf` padding to eight, `softmax` itself,
    // and the SV fused multiply-adds position by position from the first key.
    // What changes is residence: a block of `K` is loaded and split once for
    // up to three rows (twelve accumulators), a pair of `V` vectors widened
    // once for up to six rows (twelve accumulators), the queries split once
    // per block, and the SV accumulators stay in registers instead of being
    // reloaded and stored at every position. So the block gives the bits of
    // the decode kernel called row by row.
    //
    // The causal mask is in the geometry: the keys `[0, past]` are seen by
    // every row (one loop), the key `past + t` (`t ≥ 1`) by the rows `≥ t`
    // (emitted unrolled; a row that does not see a key is not touched).
    //
    // A sliding window (`SdpaPrefillSpec::window`, LFM3's and Gemma 4's
    // sliding layers): the call gives the absolute `past` and the cache from
    // position 0. The prologue moves `K` and `V` past the `s0 = max(0, past +
    // 1 − window)` keys no row of the block sees, and `past` becomes `past' =
    // past − s0 ≤ window − 1`. Row `i` then sees `[start_i, end_i)`, `end_i =
    // past' + i + 1`, `start_i = max(0, end_i − window) ≤ i`. QK and SV keep
    // their loops over `[0, end_i)`: at most seven keys more for a row, against
    // the window's. The softmax runs from `start_i` exactly, so its eight lanes
    // and its padding are those of the decode kernel called on the row's
    // window; the columns `[0, start_i)` then become `+0`, and SV's fused
    // multiply-adds of them leave its accumulators (`+0` until the first key
    // of the window) as they are. So the windowed block also gives the bits
    // of the decode kernel row by row.
    //
    // [`Mask`] holds the per-row bounds; pass 2 and the zeroing are their only
    // consumers, with the prologue's shift and the unrolled tail's
    // [`Mask::sees_tail`].
    //
    // Without the causal mask (`full`, the vision tower's patches, one input of
    // an encoder): the sixth argument is the count `n` of keys, every row sees
    // `[0, n)`, QK and SV run their loops over the `n` keys for every row and
    // there is no tail — the decode kernel called on `n` keys, row by row.
    //
    // With the bidirectional window too (`full && window`, EmbeddingGemma 2's
    // sliding layers, `|q − k| < window`): the sixth argument is the tile's
    // `past` as for the causal window, and `rax` the end of the keys (the
    // calling convention of the AVX-512 block). The prologue moves `K` and `V`
    // past `s0` as above, `end' = end − s0`; row `i` sees `[d_i, e_i)`, `d_i =
    // max(0, past' + i + 1 − window)`, `e_i = min(past' + i + window, end')`.
    // QK and SV run over `[0, N)` for every row, `N = e_{rows−1}` (at most
    // `rows − 1` keys more than a row's `2·window − 1`); the softmax runs on
    // `[d_i, e_i)` exactly, then `[0, d_i)` and `[e_i, N)` become `+0` — after
    // it, since the exp of the `-inf` padding past `e_i` is not zero (the
    // argument is clamped at −87). The bits are the decode kernel's on the
    // row's window.

    /// Home of the six arguments.
    const H_Q: i32 = OFF_HOME;
    const H_K: i32 = OFF_HOME + 8;
    const H_V: i32 = OFF_HOME + 16;
    const H_S: i32 = OFF_HOME + 24;
    const H_O: i32 = OFF_HOME + 32;
    const H_P: i32 = OFF_HOME + 40;
    /// The bidirectional window: the end of the keys (`rax` at entry), `end'`
    /// after the prologue.
    const H_E: i32 = OFF_HOME + 48;
    /// Without the causal mask: the keys every row's QK and SV loops run over
    /// (`n`, or `N` with the window).
    const H_N: i32 = OFF_HOME + 56;
    /// The queries split once: per row and per block of 16 bf16, the even
    /// halves widened (`q << 16`) then the odd ones (`q & HI`), 64 bytes.
    const OFF_QS: i32 = OFF_SV;

    /// The rows split into the fewest balanced groups of at most `max`.
    fn groups(rows: usize, max: usize) -> Vec<(usize, usize)> {
        let n = rows.div_ceil(max.max(1));
        let (base, extra) = (rows / n, rows % n);
        let mut at = 0;
        (0..n)
            .map(|g| {
                let len = base + usize::from(g < extra);
                at += len;
                (at - len, at)
            })
            .collect()
    }

    /// Generates the prefill **block kernel** of `spec`, in AVX2.
    pub fn emit_sdpa_prefill_avx2(spec: &SdpaPrefillSpec) -> Result<Vec<u8>> {
        check_block_avx2(spec)
            .map_err(|d| HerbertError::Backend(format!("backend_jit sdpa prefill avx2: {d}")))?;
        assemble_block(spec).map_err(|e| {
            HerbertError::Backend(format!(
                "backend_jit: sdpa prefill avx2 emission head_dim={} rows={}: {e}",
                spec.shape.head_dim, spec.rows
            ))
        })
    }

    struct Block {
        nb16: usize,
        n_acc: usize,
        row: i32,
        srow: i32,
    }

    impl Block {
        fn qs(&self, r: usize, b: usize, hi: usize) -> AsmMemoryOperand {
            ymmword_ptr(rsp + OFF_QS + (((r * self.nb16 + b) * 2 + hi) * 32) as i32)
        }

        /// The scores of one key for the rows `act` of the group starting at
        /// `g0`: `K` at `rsi + k_off`, scores at `rdx + s_off`.
        fn qk(
            &self,
            asm: &mut CodeAssembler,
            act: &[usize],
            g0: usize,
            k_off: i32,
            s_off: i32,
        ) -> Result2 {
            const KH: usize = 14;
            const KL: usize = 15;
            let n = self.n_acc;
            for &r in act {
                for c in 0..n {
                    let a = (r - g0) * n + c;
                    asm.vxorps(YMM[a], YMM[a], YMM[a])?;
                }
            }
            for b in 0..self.nb16 {
                asm.vmovdqu(YMM[KH], ymmword_ptr(rsi + k_off + (b * 32) as i32))?;
                asm.vpslld(YMM[KL], YMM[KH], 16u32)?;
                asm.vpand(YMM[KH], YMM[KH], c(constant::HI))?;
                for &r in act {
                    let a = (r - g0) * n + b % n;
                    asm.vfmadd231ps(YMM[a], YMM[KL], self.qs(r, b, 0))?;
                    asm.vfmadd231ps(YMM[a], YMM[KH], self.qs(r, b, 1))?;
                }
            }
            for &r in act {
                let a0 = (r - g0) * n;
                if n == 4 {
                    asm.vaddps(YMM[a0], YMM[a0], YMM[a0 + 1])?;
                    asm.vaddps(YMM[a0 + 2], YMM[a0 + 2], YMM[a0 + 3])?;
                    asm.vaddps(YMM[a0], YMM[a0], YMM[a0 + 2])?;
                } else {
                    for c in 1..n {
                        asm.vaddps(YMM[a0], YMM[a0], YMM[a0 + c])?;
                    }
                }
                hsum(asm, a0, KL)?;
                asm.vmulss(XMM[a0], XMM[a0], dword_ptr(rsp + OFF_SCALE))?;
                asm.vmovss(dword_ptr(rdx + r as i32 * self.srow + s_off), XMM[a0])?;
            }
            Ok(())
        }

        /// One key of `V` (at `rsi + v_off`, dimensions `[8·d0, 8·(d0 + dn))`)
        /// weighted by the scores of the rows `act` (at `rdi + s_off`).
        #[allow(clippy::too_many_arguments)]
        fn sv(
            &self,
            asm: &mut CodeAssembler,
            act: &[usize],
            g0: usize,
            d0: usize,
            dn: usize,
            v_off: i32,
            s_off: i32,
        ) -> Result2 {
            const S: usize = 15;
            let vb = 15 - dn;
            for d in 0..dn {
                widen(
                    asm,
                    vb + d,
                    xmmword_ptr(rsi + v_off + ((d0 + d) * 16) as i32),
                )?;
            }
            for &r in act {
                asm.vbroadcastss(YMM[S], dword_ptr(rdi + r as i32 * self.srow + s_off))?;
                for d in 0..dn {
                    let a = (r - g0) * dn + d;
                    asm.vfmadd231ps(YMM[a], YMM[vb + d], YMM[S])?;
                }
            }
            Ok(())
        }
    }

    /// Output vectors (8 dimensions each) an SV loop accumulates together.
    const SV_D: usize = 2;

    /// The keys each row of the block sees, `[start_i, end_i)` in the cache
    /// as the prologue left it (`past'`): causal, `end_i = past' + i + 1`;
    /// with a window, `start_i = max(0, end_i − window)`, else 0.
    struct Mask {
        window: usize,
        /// Without the causal mask (`SdpaPrefillSpec::full`).
        full: bool,
        rows: usize,
    }

    impl Mask {
        fn bidir(&self) -> bool {
            self.full && self.window > 0
        }

        /// Does row `r` see the key `past' + t` (`t ≥ 1`) of the unrolled
        /// tail? The causal mask: the rows from `t` on. Without it there is no
        /// tail: the common loop covers every key.
        fn sees_tail(&self, r: usize, t: usize) -> bool {
            !self.full && r >= t
        }

        /// `dst` = the keys the common QK and SV loops run over, for every row:
        /// `past' + 1` causal, `n` or `N` without the mask.
        fn common(&self, asm: &mut CodeAssembler, dst: AsmRegister64) -> Result2 {
            if self.full {
                asm.mov(dst, qword_ptr(rsp + H_N))?;
            } else {
                asm.mov(dst, qword_ptr(rsp + H_P))?;
                asm.inc(dst)?;
            }
            Ok(())
        }

        /// Without the causal mask, after the shift: `H_N` = `n` (`r9`), or
        /// `N = min(past' + rows − 1 + window, end')`. `rax` clobbered.
        fn keys(&self, asm: &mut CodeAssembler) -> Result2 {
            if !self.full {
                return Ok(());
            }
            if self.bidir() {
                asm.lea(rax, qword_ptr(r9 + (self.rows - 1 + self.window) as i32))?;
                asm.cmp(rax, qword_ptr(rsp + H_E))?;
                asm.cmova(rax, qword_ptr(rsp + H_E))?;
                asm.mov(qword_ptr(rsp + H_N), rax)?;
            } else {
                asm.mov(qword_ptr(rsp + H_N), r9)?;
            }
            Ok(())
        }

        /// The prologue: `K` and `V` (homes `H_K`, `H_V`) move past the keys
        /// no row sees, `past` (in `r9` and `H_P`) becomes `past'`. `rax`,
        /// `r10` clobbered.
        fn shift(&self, asm: &mut CodeAssembler, row: i32) -> Result2 {
            if self.window == 0 {
                return Ok(());
            }
            // s0 = max(0, past − (window − 1))
            asm.mov(rax, r9)?;
            asm.sub(rax, (self.window - 1) as i32)?;
            asm.xor(r10d, r10d)?;
            asm.test(rax, rax)?;
            asm.cmovs(rax, r10)?;
            asm.sub(r9, rax)?;
            asm.mov(qword_ptr(rsp + H_P), r9)?;
            if self.bidir() {
                asm.sub(qword_ptr(rsp + H_E), rax)?;
            }
            asm.imul_3(rax, rax, row)?;
            asm.add(qword_ptr(rsp + H_K), rax)?;
            asm.add(qword_ptr(rsp + H_V), rax)?;
            Ok(())
        }

        /// `dst = end_i` of the row whose causal end `past' + i + 1` is in
        /// `pos`: `pos` causal, `n` without the mask, `min(pos + window − 1,
        /// end')` with the bidirectional window.
        fn row_end(
            &self,
            asm: &mut CodeAssembler,
            dst: AsmRegister64,
            pos: AsmRegister64,
        ) -> Result2 {
            if self.bidir() {
                asm.lea(dst, qword_ptr(pos + (self.window - 1) as i32))?;
                asm.cmp(dst, qword_ptr(rsp + H_E))?;
                asm.cmova(dst, qword_ptr(rsp + H_E))?;
            } else if self.full {
                asm.mov(dst, qword_ptr(rsp + H_N))?;
            } else {
                asm.mov(dst, pos)?;
            }
            Ok(())
        }

        /// `dst = start_i` of the row whose causal end is in `pos`; `zero`
        /// clobbered. Not emitted without a window (`start_i = 0`).
        fn row_start(
            &self,
            asm: &mut CodeAssembler,
            dst: AsmRegister64,
            pos: AsmRegister64,
            zero: AsmRegister64,
        ) -> Result2 {
            asm.mov(dst, pos)?;
            asm.xor(zero, zero)?;
            asm.sub(dst, self.window as i32)?;
            asm.cmovs(dst, zero)?;
            Ok(())
        }
    }

    fn assemble_block(spec: &SdpaPrefillSpec) -> std::result::Result<Vec<u8>, IcedError> {
        let hd = spec.shape.head_dim;
        let nv = hd / 8;
        let nb16 = hd / 16;
        let rows = spec.rows;
        let blk = Block {
            nb16,
            n_acc: 4.min(nb16.max(1)),
            row: spec.shape.kv_row_bytes() as i32,
            srow: (block_scores_row(spec) * 4) as i32,
        };
        let qrow = (spec.row_stride * 2) as i32;
        let orow = (spec.out_row_stride * 2) as i32;
        let frame = (OFF_QS + (rows * nb16 * 64) as i32 + 63) & !63;

        let mut asm = CodeAssembler::new(64)?;
        asm.set_prefer_vex(true);
        asm.push(rbp)?;
        asm.mov(rbp, rsp)?;
        asm.and(rsp, -64i32)?;
        crate::jit::arch::x86::stack_reserve(&mut asm, frame)?;
        asm.vmovss(dword_ptr(rsp + OFF_SCALE), XMM[0])?;
        if spec.shape.sink {
            asm.vmovss(dword_ptr(rsp + OFF_SCALE + 4), XMM[1])?;
        }
        asm.mov(qword_ptr(rsp + H_Q), rdi)?;
        asm.mov(qword_ptr(rsp + H_K), rsi)?;
        asm.mov(qword_ptr(rsp + H_V), rdx)?;
        asm.mov(qword_ptr(rsp + H_S), rcx)?;
        asm.mov(qword_ptr(rsp + H_O), r8)?;
        asm.mov(qword_ptr(rsp + H_P), r9)?;
        let mask = Mask {
            window: spec.window,
            full: spec.full,
            rows,
        };
        if mask.bidir() {
            asm.mov(qword_ptr(rsp + H_E), rax)?;
        }
        mask.shift(&mut asm, blk.row)?;
        mask.keys(&mut asm)?;
        build_table(&mut asm, 15)?;

        // ── the queries, split once ───────────────────────────────────────
        for r in 0..rows {
            for b in 0..nb16 {
                asm.vmovdqu(YMM[0], ymmword_ptr(rdi + r as i32 * qrow + (b * 32) as i32))?;
                asm.vpslld(YMM[1], YMM[0], 16u32)?;
                asm.vpand(YMM[0], YMM[0], c(constant::HI))?;
                asm.vmovups(blk.qs(r, b, 0), YMM[1])?;
                asm.vmovups(blk.qs(r, b, 1), YMM[0])?;
            }
        }

        // ── pass 1: QK, a block of `K` for up to three rows ──────────────
        for (g0, g1) in groups(rows, 14 / blk.n_acc) {
            let all: Vec<usize> = (g0..g1).collect();
            asm.mov(rsi, qword_ptr(rsp + H_K))?;
            asm.mov(rdx, qword_ptr(rsp + H_S))?;
            mask.common(&mut asm, rcx)?;
            let mut top = asm.create_label();
            asm.set_label(&mut top)?;
            blk.qk(&mut asm, &all, g0, 0, 0)?;
            asm.add(rsi, blk.row)?;
            asm.add(rdx, 4)?;
            asm.dec(rcx)?;
            asm.jnz(top)?;
            for t in 1..rows {
                let act: Vec<usize> = (g0..g1).filter(|&r| mask.sees_tail(r, t)).collect();
                if !act.is_empty() {
                    blk.qk(
                        &mut asm,
                        &act,
                        g0,
                        (t as i32 - 1) * blk.row,
                        (t as i32 - 1) * 4,
                    )?;
                }
            }
        }

        // ── pass 2: per row, the `-inf` padding then the softmax ─────────
        //
        // rdi = the row's scores (column 0), r10 = its causal end `past' + i +
        // 1`, r8 = rows left; per row, r9 = `start_i`, rdx = `end_i`. The
        // softmax keeps rdi, r8, r9, r10 (it clobbers rsi, rcx, eax).
        {
            let windowed = mask.window > 0;
            asm.mov(rdi, qword_ptr(rsp + H_S))?;
            asm.mov(r10, qword_ptr(rsp + H_P))?;
            asm.inc(r10)?; // the keys of row 0
            asm.mov(r8, rows as u64)?;
            let mut row_top = asm.create_label();
            let mut pad_top = asm.create_label();
            let mut no_pad = asm.create_label();
            asm.set_label(&mut row_top)?;
            mask.row_end(&mut asm, rdx, r10)?;
            asm.mov(rax, rdx)?;
            if windowed {
                mask.row_start(&mut asm, r9, r10, rcx)?;
                asm.sub(rax, r9)?;
            }
            // rax = the row's keys, r11 = their blocks of eight.
            asm.lea(r11, qword_ptr(rax + 7))?;
            asm.shr(r11, 3u32)?;
            asm.lea(rdx, ptr(rdi + rdx * 4))?;
            asm.and(rax, 7)?;
            asm.jz(no_pad)?;
            asm.mov(rcx, 8u64)?;
            asm.sub(rcx, rax)?;
            asm.set_label(&mut pad_top)?;
            asm.mov(dword_ptr(rdx), 0xFF80_0000u32)?;
            asm.add(rdx, 4)?;
            asm.dec(rcx)?;
            asm.jnz(pad_top)?;
            asm.set_label(&mut no_pad)?;
            if windowed {
                asm.lea(rdi, ptr(rdi + r9 * 4))?;
            }
            softmax(&mut asm, spec.shape.sink)?;
            if mask.bidir() {
                // The columns `[e_i, N)` become `+0` (rdi is at column `d_i`,
                // r9 = `d_i`), over the exp of the padding among them.
                let (mut tail_top, mut tail_done) = (asm.create_label(), asm.create_label());
                mask.row_end(&mut asm, rdx, r10)?;
                asm.mov(rcx, qword_ptr(rsp + H_N))?;
                asm.sub(rcx, rdx)?;
                asm.jz(tail_done)?;
                asm.mov(rax, rdx)?;
                asm.sub(rax, r9)?;
                asm.lea(rsi, ptr(rdi + rax * 4))?;
                asm.set_label(&mut tail_top)?;
                asm.mov(dword_ptr(rsi), 0u32)?;
                asm.add(rsi, 4)?;
                asm.dec(rcx)?;
                asm.jnz(tail_top)?;
                asm.set_label(&mut tail_done)?;
            }
            if windowed {
                // The columns `[0, start_i)` become `+0`, walking rdi back to
                // the row's column 0.
                let (mut zero_top, mut zero_done) = (asm.create_label(), asm.create_label());
                asm.test(r9, r9)?;
                asm.jz(zero_done)?;
                asm.set_label(&mut zero_top)?;
                asm.sub(rdi, 4)?;
                asm.mov(dword_ptr(rdi), 0u32)?;
                asm.dec(r9)?;
                asm.jnz(zero_top)?;
                asm.set_label(&mut zero_done)?;
            }
            asm.add(rdi, blk.srow)?;
            asm.inc(r10)?;
            asm.dec(r8)?;
            asm.jnz(row_top)?;
        }

        // ── pass 3: SV, two vectors of `V` for up to six rows ────────────
        let dn = SV_D;
        for (g0, g1) in groups(rows, 15 / dn - 1) {
            let all: Vec<usize> = (g0..g1).collect();
            for d0 in (0..nv).step_by(dn) {
                asm.mov(rsi, qword_ptr(rsp + H_V))?;
                asm.mov(rdi, qword_ptr(rsp + H_S))?;
                mask.common(&mut asm, rcx)?;
                for a in 0..(g1 - g0) * dn {
                    asm.vxorps(YMM[a], YMM[a], YMM[a])?;
                }
                let mut top = asm.create_label();
                asm.set_label(&mut top)?;
                blk.sv(&mut asm, &all, g0, d0, dn, 0, 0)?;
                asm.add(rdi, 4)?;
                asm.add(rsi, blk.row)?;
                asm.dec(rcx)?;
                asm.jnz(top)?;
                for t in 1..rows {
                    let act: Vec<usize> = (g0..g1).filter(|&r| mask.sees_tail(r, t)).collect();
                    if !act.is_empty() {
                        blk.sv(
                            &mut asm,
                            &act,
                            g0,
                            d0,
                            dn,
                            (t as i32 - 1) * blk.row,
                            (t as i32 - 1) * 4,
                        )?;
                    }
                }
                asm.mov(rdx, qword_ptr(rsp + H_O))?;
                for r in g0..g1 {
                    for d in 0..dn {
                        let a = (r - g0) * dn + d;
                        cvt_bf16(&mut asm, 15 - dn, a, 15)?;
                        asm.vmovdqu(
                            xmmword_ptr(rdx + r as i32 * orow + ((d0 + d) * 16) as i32),
                            XMM[15 - dn],
                        )?;
                    }
                }
            }
        }

        asm.mov(rsp, rbp)?;
        asm.pop(rbp)?;
        asm.vzeroupper()?;
        asm.ret()?;
        asm.assemble(0)
    }
}
