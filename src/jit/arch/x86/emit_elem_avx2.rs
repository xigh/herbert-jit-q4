//! The elementwise kernels, AVX2 arm: the same operations as `emit_elem.rs`
//! for a machine without AVX-512, sixteen `ymm` registers of eight lanes.
//!
//! Three deviations from the AVX-512 kernels, and what they cost:
//!
//! 1. **Sixteen registers, not thirty-two.** The SwiGLU keeps twelve constants in registers under
//!    AVX-512; here they go into a stack table read as memory operands. Cost: a `sub rsp`/`add rsp`
//!    and the table construction at entry, outside any loop.
//! 2. **`ymm16..ymm31` do not exist.** Using them would make `iced` encode EVEX, that is AVX-512,
//!    without any warning until the `SIGILL`. This module allocates through `RegFile::low` only and
//!    sets `set_prefer_vex(true)`; the check that catches a slip is a disassembly with zero EVEX
//!    encodings.
//! 3. **No `vcvtneps2bf16`.** The rounding to bf16 is done by hand, round to nearest even in five
//!    integer operations:
//!
//!    ```text
//!    t = (bits >> 16) & 1        ; the least significant bit of the target bf16
//!    t = bits + 0x7FFF + t       ; half an ulp, biased toward even
//!    bf16 = t >> 16
//!    ```
//!
//!    This is what the silicon does for finite values. On `NaN` the hardware
//!    always propagates a `NaN`; this sequence does not: depending on the
//!    payload the result is a `NaN`, an infinity (a payload confined to the low
//!    16 bits rounds away) or a zero (the rounding carries out of the exponent
//!    field).
//!
//! The two arms agree to one bf16 ulp, not to the bit: the sum of squares of
//! an RMSNorm is partitioned over eight lanes instead of sixteen and reduced
//! by a different tree (the divergences are counted).

use crate::core::error::{HerbertError, Result};
use crate::jit::emit_elem::{loop_unroll, ElemOp, ElemSpec, LoopArm};
use crate::jit::ir::Decline;

pub use x86::emit_elem_avx2;

mod x86 {
    use super::*;
    use crate::jit::regs::{RegFile, Vlow};
    use iced_x86::code_asm::*;
    use iced_x86::IcedError;

    /// **Sixteen entries, not thirty-two.** `ymm16..ymm31` exist only
    /// under AVX-512; listing them here would suffice to make them accidentally
    /// reachable, and `iced` would encode them as EVEX. The bound is in the type.
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
    const EXP_MAX88: f32 = 88.0;
    /// `K = 2·√(2/π)` of geluTanh.
    const GELU_K: f32 = 1.595_769;
    /// `C` of geluTanh.
    const GELU_C: f32 = 0.044_715;
    /// The coefficients of `erfcc` (Numerical Recipes), lowest first: the exact GeLU of the AVX-512
    /// arm.
    const ERFC_P: [f32; 10] = [
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

    /// The entries of the constants table, in **slot numbers** of
    /// 32 bytes. Order does not matter; **stability** does: these
    /// numbers are the only displacements written in the emitted code.
    mod constant {
        pub const NEG87: i32 = 0;
        pub const LOG2E: i32 = 1;
        pub const C4: i32 = 2;
        pub const C3: i32 = 3;
        pub const C2: i32 = 4;
        pub const C1: i32 = 5;
        pub const C0: i32 = 6;
        /// `0x3F80_0000` — seen as an **integer**, this is what the
        /// reconstruction of `2^k` adds to the exponent field.
        pub const ONE_BITS: i32 = 7;
        /// `1.0f` — seen as **float**, for `1 + e` and `1 + w`.
        pub const FONE: i32 = 8;
        pub const MAX88: i32 = 9;
        pub const GK: i32 = 10;
        pub const GKC: i32 = 11;
        /// The integer `1`, to extract the least significant bit of the targeted bf16.
        pub const ONE_I32: i32 = 12;
        /// `0x7FFF`, the half-ULP of bf16 rounding.
        pub const BIAS: i32 = 13;
        /// `0x7FFF_FFFF` — the absolute value mask of the softcap.
        pub const ABS: i32 = 14;
        /// `0x8000_0000` — the sign mask of the softcap.
        pub const SGN: i32 = 15;
        /// `−2.0`, numerator of `−2/cap`.
        pub const NEG2: i32 = 16;
        /// `½` and `1/√2` of the exact GeLU.
        pub const HALF: i32 = 17;
        pub const ISQ: i32 = 18;
        /// The ten coefficients of the exact GeLU's `erfc`, `P0..P9`.
        pub const P0: i32 = 19;
        /// Number of reserved slots.
        pub const N: i32 = 29;
    }

    /// Table size, rounded up to 64 bytes.
    const TABLE: i32 = (constant::N * 32 + 63) & !63;

    /// Address of a constant slot.
    fn c(slot: i32) -> AsmMemoryOperand {
        ymmword_ptr(rsp + slot * 32)
    }

    /// The two ways an emission can fail — same pattern
    /// as `emit_elem` and `emit_sdpa`.
    #[derive(Debug)]
    enum EmitErr {
        Iced(IcedError),
        Alloc(Decline),
    }
    impl From<IcedError> for EmitErr {
        fn from(e: IcedError) -> Self {
            EmitErr::Iced(e)
        }
    }
    impl From<Decline> for EmitErr {
        fn from(d: Decline) -> Self {
            EmitErr::Alloc(d)
        }
    }
    impl std::fmt::Display for EmitErr {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                EmitErr::Iced(e) => write!(f, "{e}"),
                EmitErr::Alloc(d) => write!(f, "register allocation: {d}"),
            }
        }
    }

    /// Writes a broadcast constant in its lane. Called **at the input**,
    /// outside any loop. `t` is any scratchpad.
    fn store_const(
        asm: &mut CodeAssembler,
        slot: i32,
        bits: u32,
        t: Vlow,
    ) -> std::result::Result<(), IcedError> {
        asm.mov(eax, bits)?;
        asm.vmovd(XMM[t.x()], eax)?;
        asm.vpbroadcastd(YMM[t.idx()], XMM[t.x()])?;
        asm.vmovups(c(slot), YMM[t.idx()])?;
        Ok(())
    }

    /// Builds the full table. `t` is returned at the end.
    fn build_table(asm: &mut CodeAssembler, t: Vlow) -> std::result::Result<(), IcedError> {
        store_const(asm, constant::NEG87, EXP_NEG87.to_bits(), t)?;
        store_const(asm, constant::LOG2E, EXP_LOG2E.to_bits(), t)?;
        store_const(asm, constant::C4, EXP_C4.to_bits(), t)?;
        store_const(asm, constant::C3, EXP_C3.to_bits(), t)?;
        store_const(asm, constant::C2, EXP_C2.to_bits(), t)?;
        store_const(asm, constant::C1, EXP_C1.to_bits(), t)?;
        store_const(asm, constant::C0, EXP_C0.to_bits(), t)?;
        store_const(asm, constant::ONE_BITS, EXP_ONE_BITS, t)?;
        store_const(asm, constant::FONE, 1.0f32.to_bits(), t)?;
        store_const(asm, constant::MAX88, EXP_MAX88.to_bits(), t)?;
        store_const(asm, constant::GK, GELU_K.to_bits(), t)?;
        store_const(asm, constant::GKC, (GELU_K * GELU_C).to_bits(), t)?;
        store_const(asm, constant::ONE_I32, 1u32, t)?;
        store_const(asm, constant::BIAS, 0x7FFFu32, t)?;
        store_const(asm, constant::ABS, 0x7FFF_FFFFu32, t)?;
        store_const(asm, constant::SGN, 0x8000_0000u32, t)?;
        store_const(asm, constant::NEG2, (-2.0f32).to_bits(), t)?;
        store_const(asm, constant::HALF, 0.5f32.to_bits(), t)?;
        store_const(
            asm,
            constant::ISQ,
            std::f32::consts::FRAC_1_SQRT_2.to_bits(),
            t,
        )?;
        for (i, c) in ERFC_P.iter().enumerate() {
            store_const(asm, constant::P0 + i as i32, c.to_bits(), t)?;
        }
        Ok(())
    }

    /// `io = exp(io)` — the sequence of the AVX-512 `exp_inline` (`emit_elem.rs`), constants in
    /// memory instead of registers.
    ///
    /// **Domain: `io ≤ 88`**, as in `emit_elem::exp_inline`. The lower bound is
    /// applied here; the upper bound belongs to the caller whose domain
    /// contains positives.
    fn exp_inline(
        asm: &mut CodeAssembler,
        io: Vlow,
        t1: Vlow,
        t2: Vlow,
        t3: Vlow,
    ) -> std::result::Result<(), IcedError> {
        let (io, t1, t2, t3) = (io.idx(), t1.idx(), t2.idx(), t3.idx());
        asm.vmaxps(YMM[io], YMM[io], c(constant::NEG87))?;
        asm.vmulps(YMM[t1], YMM[io], c(constant::LOG2E))?;
        // `vrndscaleps` is AVX-512; `vroundps` with immediate 0 performs the same
        // rounding (to nearest, exceptions suppressed on the `vrndscale` side).
        asm.vroundps(YMM[t2], YMM[t1], 0u32)?;
        asm.vsubps(YMM[t1], YMM[t1], YMM[t2])?;
        // **`vmovups`, never `vmovaps`**: `vmovaps` requires 32-byte
        // alignment, and the table lives under `rsp`, which is `≡ 8 mod 16` at
        // the entry of a SysV function — not even aligned to 16. A `vmovaps`
        // there causes a #GP, hence a SIGSEGV. The other memory operands in
        // this module are in VEX, which imposes **no** alignment.
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

    /// Widens the lower 8 bf16 (`half=0`) or upper 8 bf16 (`half=1`) of `src` to f32.
    ///
    /// One ymm holds **16** bf16 — half as many as a zmm, hence twice as many
    /// blocks and the same "two halves" structure.
    fn widen_half(
        asm: &mut CodeAssembler,
        dst: Vlow,
        src: Vlow,
        half: usize,
    ) -> std::result::Result<(), IcedError> {
        let (d, s) = (dst.idx(), src.idx());
        if half == 0 {
            asm.vpmovzxwd(YMM[d], XMM[src.x()])?;
        } else {
            asm.vextracti128(XMM[dst.x()], YMM[s], 1)?;
            asm.vpmovzxwd(YMM[d], XMM[dst.x()])?;
        }
        asm.vpslld(YMM[d], YMM[d], 16u32)?;
        Ok(())
    }

    /// `dst.x` receives the 8 f32 from `src` **rounded to bf16** (128 bits).
    ///
    /// The rounding is that of `vcvtneps2bf16` on finite values: nearest,
    /// even in case of a tie. See the module header for the assumed gap on
    /// `NaN`s.
    ///
    /// `t` is a scratch register; `dst` and `t` must be distinct from `src`.
    fn cvt_bf16(
        asm: &mut CodeAssembler,
        dst: Vlow,
        src: Vlow,
        t: Vlow,
    ) -> std::result::Result<(), IcedError> {
        let (d, s, tt) = (dst.idx(), src.idx(), t.idx());
        asm.vpsrld(YMM[tt], YMM[s], 16u32)?;
        asm.vpand(YMM[tt], YMM[tt], c(constant::ONE_I32))?;
        asm.vpaddd(YMM[tt], YMM[tt], c(constant::BIAS))?;
        asm.vpaddd(YMM[tt], YMM[tt], YMM[s])?;
        asm.vpsrld(YMM[tt], YMM[tt], 16u32)?;
        // All eight dwords fit in 16 bits: `vpackusdw` saturates to
        // 0xFFFF at the top, so it **never saturates** here. The lane order
        // is the one we want: `dst = [low(0..3), high(0..3)]`.
        asm.vextracti128(XMM[d], YMM[tt], 1)?;
        asm.vpackusdw(XMM[d], XMM[tt], XMM[d])?;
        Ok(())
    }

    /// Horizontal sum of a ymm f32 → scalar in `xmm[dst]`.
    fn hsum(asm: &mut CodeAssembler, dst: Vlow, t: Vlow) -> std::result::Result<(), IcedError> {
        asm.vextractf128(XMM[t.x()], YMM[dst.idx()], 1)?;
        asm.vaddps(XMM[dst.x()], XMM[dst.x()], XMM[t.x()])?;
        asm.vhaddps(XMM[dst.x()], XMM[dst.x()], XMM[dst.x()])?;
        asm.vhaddps(XMM[dst.x()], XMM[dst.x()], XMM[dst.x()])?;
        Ok(())
    }

    /// Emits the kernel described by `spec`, **in AVX2**. Same ABI, same
    /// semantics, and the same result **to one bf16 ulp** as
    /// `emit_elem::emit_elem` — not the same bits.
    ///
    /// The three gaps are listed and measured in the documentation of
    /// [`crate::jit::elem::select_isa`].
    /// The widest is the denormals: the software rounding of this arm keeps
    /// them, `vcvtneps2bf16` flushes them to `±0`.
    pub fn emit_elem_avx2(spec: &ElemSpec) -> Result<Vec<u8>> {
        emit_with(spec, loop_unroll(spec.op, spec.dim, LoopArm::Ymm))
    }

    fn emit_with(spec: &ElemSpec, unroll: Option<usize>) -> Result<Vec<u8>> {
        spec.check_whole()
            .map_err(|d| HerbertError::Backend(format!("backend_jit elem avx2: {d}")))?;
        assemble(spec, unroll).map_err(|e| {
            HerbertError::Backend(format!(
                "backend_jit: avx2 emission {} dim={}: {e}",
                spec.op.name(),
                spec.dim
            ))
        })
    }

    fn assemble(spec: &ElemSpec, unroll: Option<usize>) -> std::result::Result<Vec<u8>, EmitErr> {
        let mut asm = CodeAssembler::new(64)?;
        // Without this, `iced` may choose the EVEX encoding for certain shapes,
        // that is AVX-512 in a kernel that must carry none.
        asm.set_prefer_vex(true);

        // A ymm holds 16 bf16: twice as many blocks as in zmm.
        let nb = spec.dim / 16;
        let mut f = RegFile::new();

        // **First and foremost**: `xmm0` carries `eps` / `s` / `cap` depending on the ABI, and
        // building the table will overwrite it. Saving it in an
        // **integer** register puts it out of reach of everything this kernel does —
        // this is the workaround in `emit_elem`, for the same reason.
        let lit_xmm0 = matches!(
            spec.op,
            ElemOp::RmsNorm
                | ElemOp::RmsNormGemma
                | ElemOp::Scale
                | ElemOp::Softcap
                | ElemOp::Clamp
                | ElemOp::PoolRows(_)
                | ElemOp::LayerNorm
        );
        if lit_xmm0 {
            asm.vmovd(r10d, xmm0)?;
        }
        // The second scalar of `Clamp` and `PoolRows` (`hi`, `inv`), out of reach the same way.
        if matches!(spec.op, ElemOp::Clamp | ElemOp::PoolRows(_)) {
            asm.vmovd(r11d, xmm1)?;
        }
        asm.sub(rsp, TABLE)?;
        {
            let t = f.low("table:construction scratch")?;
            build_table(&mut asm, t)?;
            f.free_low(t);
        }

        match spec.op {
            // ── y = x · rsqrt(Σx²/d + ε) · w   (· (1+w) in the Gemma variant) ─
            //
            // rdi = x, rsi = w, rdx = y, xmm0 = eps (saved in r10d).
            ElemOp::RmsNorm | ElemOp::RmsNormGemma => {
                let plus_one = spec.op == ElemOp::RmsNormGemma;
                // Pass 1: Σx². No `vdpbf16ps` in AVX2: we widen the
                // two halves then accumulate. Four chains to avoid
                // paying latency in series, like the AVX-512 arm.
                let n_acc = 4.min(nb);
                let acc = f.low_run(n_acc, "rms:Σx² chains")?;
                let ld = f.low("rms:loaded block / scalar temporary")?;
                let w0 = f.low("rms:widened half")?;
                for a in &acc {
                    asm.vxorps(YMM[a.idx()], YMM[a.idx()], YMM[a.idx()])?;
                }
                for cb in 0..nb {
                    asm.vmovdqu(YMM[ld.idx()], ymmword_ptr(rdi + (cb * 32) as i32))?;
                    for half in 0..2 {
                        widen_half(&mut asm, w0, ld, half)?;
                        asm.vfmadd231ps(YMM[acc[cb % n_acc].idx()], YMM[w0.idx()], YMM[w0.idx()])?;
                    }
                }
                let a0 = acc[0];
                for a in &acc[1..] {
                    asm.vaddps(YMM[a0.idx()], YMM[a0.idx()], YMM[a.idx()])?;
                }
                hsum(&mut asm, a0, ld)?;
                // rsqrt(Σx²/dim + eps), in f32 scalar — same sequence as
                // the AVX-512 arm, so same last bits.
                asm.mov(eax, (1.0f32 / spec.dim as f32).to_bits())?;
                asm.vmovd(XMM[ld.x()], eax)?;
                asm.vmulss(XMM[a0.x()], XMM[a0.x()], XMM[ld.x()])?;
                asm.vmovd(XMM[ld.x()], r10d)?; // eps, returned from its full register
                asm.vaddss(XMM[a0.x()], XMM[a0.x()], XMM[ld.x()])?;
                asm.vsqrtss(XMM[a0.x()], XMM[a0.x()], XMM[a0.x()])?;
                asm.mov(eax, 1.0f32.to_bits())?;
                asm.vmovd(XMM[ld.x()], eax)?;
                asm.vdivss(XMM[a0.x()], XMM[ld.x()], XMM[a0.x()])?;
                let inv = f.low("rms:1/rms broadcast")?;
                asm.vbroadcastss(YMM[inv.idx()], XMM[a0.x()])?;

                for a in acc {
                    f.free_low(a);
                }
                f.free_low(ld);
                f.free_low(w0);

                // Pass 2: y = x · inv_rms · w.
                let xb = f.low("rms:block of x")?;
                let wb = f.low("rms:block of w")?;
                let xw = f.low("rms:x extended")?;
                let ww = f.low("rms:w expanded")?;
                let tmp = f.low("rms:rounding scratch")?;
                let out = f.low_run(2, "rms:output halves")?;
                for cb in 0..nb {
                    asm.vmovdqu(YMM[xb.idx()], ymmword_ptr(rdi + (cb * 32) as i32))?;
                    asm.vmovdqu(YMM[wb.idx()], ymmword_ptr(rsi + (cb * 32) as i32))?;
                    for half in 0..2 {
                        widen_half(&mut asm, xw, xb, half)?;
                        widen_half(&mut asm, ww, wb, half)?;
                        if plus_one {
                            asm.vaddps(YMM[ww.idx()], YMM[ww.idx()], c(constant::FONE))?;
                        }
                        asm.vmulps(YMM[xw.idx()], YMM[xw.idx()], YMM[inv.idx()])?;
                        asm.vmulps(YMM[xw.idx()], YMM[xw.idx()], YMM[ww.idx()])?;
                        cvt_bf16(&mut asm, out[half], xw, tmp)?;
                    }
                    // The two halves reconstruct a 16 bf16 ymm.
                    asm.vinserti128(YMM[out[0].idx()], YMM[out[0].idx()], XMM[out[1].x()], 1)?;
                    asm.vmovdqu(ymmword_ptr(rdx + (cb * 32) as i32), YMM[out[0].idx()])?;
                }
            }

            // ── y = gate · σ(gate) · up (and its two variants) ───────────
            //
            // rdi = gate, rsi = up, rdx = y.
            ElemOp::SwiGlu | ElemOp::GeluTanh | ElemOp::SigmoidMul => {
                let cubic = spec.op == ElemOp::GeluTanh;
                let gate_out = spec.op == ElemOp::SigmoidMul;
                let gb = f.low("swiglu:gate block")?;
                let ub = f.low("swiglu:up block")?;
                let g = f.low("swiglu:gate widened")?;
                let u = f.low("swiglu:up expanded")?;
                let e = f.low("swiglu:exp(-g) then result")?;
                let z = f.low("swiglu:zero")?;
                let tmp = f.low("swiglu:rounding scratch")?;
                let t = f.low_run(3, "swiglu:polynomial temporaries")?;
                let out = f.low_run(2, "swiglu:output halves")?;
                // One block of 16: the same instructions whether the kernel
                // is unrolled (bases rdi/rsi/rdx, immediate offsets) or looped
                // (cursors rcx/r8/r9, offsets within the turn). The constant
                // table lives at `rsp` and is untouched by the cursors.
                let block = |asm: &mut CodeAssembler,
                             pg: AsmRegister64,
                             pu: AsmRegister64,
                             py: AsmRegister64,
                             o: i32|
                 -> std::result::Result<(), IcedError> {
                    asm.vmovdqu(YMM[gb.idx()], ymmword_ptr(pg + o))?;
                    asm.vmovdqu(YMM[ub.idx()], ymmword_ptr(pu + o))?;
                    for half in 0..2 {
                        widen_half(asm, g, gb, half)?;
                        widen_half(asm, u, ub, half)?;
                        asm.vxorps(YMM[z.idx()], YMM[z.idx()], YMM[z.idx()])?;
                        if gate_out {
                            asm.vsubps(YMM[e.idx()], YMM[z.idx()], YMM[u.idx()])?;
                        } else if cubic {
                            // t = g·(K·C·g² + K), then negate.
                            asm.vmulps(YMM[e.idx()], YMM[g.idx()], YMM[g.idx()])?;
                            asm.vmulps(YMM[e.idx()], YMM[e.idx()], c(constant::GKC))?;
                            asm.vaddps(YMM[e.idx()], YMM[e.idx()], c(constant::GK))?;
                            asm.vmulps(YMM[e.idx()], YMM[e.idx()], YMM[g.idx()])?;
                            asm.vsubps(YMM[e.idx()], YMM[z.idx()], YMM[e.idx()])?;
                        } else {
                            asm.vsubps(YMM[e.idx()], YMM[z.idx()], YMM[g.idx()])?;
                        }
                        // Upper bound: the argument is positive as soon as `g` is
                        // negative, and this is the only call in this case.
                        asm.vminps(YMM[e.idx()], YMM[e.idx()], c(constant::MAX88))?;
                        exp_inline(asm, e, t[0], t[1], t[2])?;
                        asm.vaddps(YMM[e.idx()], YMM[e.idx()], c(constant::FONE))?;
                        asm.vdivps(YMM[e.idx()], YMM[g.idx()], YMM[e.idx()])?;
                        if !gate_out {
                            asm.vmulps(YMM[e.idx()], YMM[e.idx()], YMM[u.idx()])?;
                        }
                        cvt_bf16(asm, out[half], e, tmp)?;
                    }
                    asm.vinserti128(YMM[out[0].idx()], YMM[out[0].idx()], XMM[out[1].x()], 1)?;
                    asm.vmovdqu(ymmword_ptr(py + o), YMM[out[0].idx()])?;
                    Ok(())
                };
                match unroll {
                    None => {
                        for cb in 0..nb {
                            block(&mut asm, rdi, rsi, rdx, (cb * 32) as i32)?;
                        }
                    }
                    Some(unroll) => {
                        asm.mov(rcx, rdi)?;
                        asm.mov(r8, rsi)?;
                        asm.mov(r9, rdx)?;
                        asm.mov(r11d, (nb / unroll) as u32)?;
                        let mut top = asm.create_label();
                        asm.set_label(&mut top)?;
                        for j in 0..unroll {
                            block(&mut asm, rcx, r8, r9, (j * 32) as i32)?;
                        }
                        let step = (unroll * 32) as i32;
                        asm.add(rcx, step)?;
                        asm.add(r8, step)?;
                        asm.add(r9, step)?;
                        asm.dec(r11d)?;
                        asm.jnz(top)?;
                    }
                }
            }

            // ── a += b ────────────────────────────────────────────────────
            //
            // rdi = a (read and written), rsi = b.
            ElemOp::ResidualAdd => {
                let ab = f.low("add:block of a")?;
                let bb = f.low("add:b block")?;
                let aw = f.low("add: a widened")?;
                let bw = f.low("add:b widened")?;
                let tmp = f.low("add:rounding scratch")?;
                let out = f.low_run(2, "add:output halves")?;
                for cb in 0..nb {
                    asm.vmovdqu(YMM[ab.idx()], ymmword_ptr(rdi + (cb * 32) as i32))?;
                    asm.vmovdqu(YMM[bb.idx()], ymmword_ptr(rsi + (cb * 32) as i32))?;
                    for half in 0..2 {
                        widen_half(&mut asm, aw, ab, half)?;
                        widen_half(&mut asm, bw, bb, half)?;
                        asm.vaddps(YMM[aw.idx()], YMM[aw.idx()], YMM[bw.idx()])?;
                        cvt_bf16(&mut asm, out[half], aw, tmp)?;
                    }
                    asm.vinserti128(YMM[out[0].idx()], YMM[out[0].idx()], XMM[out[1].x()], 1)?;
                    asm.vmovdqu(ymmword_ptr(rdi + (cb * 32) as i32), YMM[out[0].idx()])?;
                }
            }

            // ── y = x · s ─────────────────────────────────────────────────
            //
            // rdi = x, rsi = y, xmm0 = s (saved in r10d).
            ElemOp::Scale => {
                let sc = f.low("scale: broadcast")?;
                let xb = f.low("scale:block of x")?;
                let xw = f.low("scale:x expanded")?;
                let tmp = f.low("scale:rounding scratch")?;
                let out = f.low_run(2, "scale:output halves")?;
                // `s` comes from the full register: the table overwrote it in xmm0.
                asm.vmovd(XMM[sc.x()], r10d)?;
                asm.vbroadcastss(YMM[sc.idx()], XMM[sc.x()])?;
                for cb in 0..nb {
                    asm.vmovdqu(YMM[xb.idx()], ymmword_ptr(rdi + (cb * 32) as i32))?;
                    for half in 0..2 {
                        widen_half(&mut asm, xw, xb, half)?;
                        asm.vmulps(YMM[xw.idx()], YMM[xw.idx()], YMM[sc.idx()])?;
                        cvt_bf16(&mut asm, out[half], xw, tmp)?;
                    }
                    asm.vinserti128(YMM[out[0].idx()], YMM[out[0].idx()], XMM[out[1].x()], 1)?;
                    asm.vmovdqu(ymmword_ptr(rsi + (cb * 32) as i32), YMM[out[0].idx()])?;
                }
            }

            // ── RoPE, rotate_half convention ──────────────────────────────
            //
            // rdi = x (read and written), rsi = cos, rdx = sin.
            // For j < dim/2:  x[j]   ← x[j]·c − x[j+h]·s
            //                   x[j+h] ← x[j]·s + x[j+h]·c
            //
            // The scheduler calls it directly on `q` and `k`.
            ElemOp::Rope => {
                let half_dim = spec.dim / 2;
                let hoff = (half_dim * 2) as i32;
                let a = f.low("rope:x[j]")?;
                let b = f.low("rope:x[j+h]")?;
                let co = f.low("rope:cos")?;
                let sn = f.low("rope:sin")?;
                let ra = f.low("rope:low output")?;
                let rb = f.low("rope:high output")?;
                let t = f.low("rope:rounding scratch")?;
                let x = f.low("rope:conversion")?;
                // **Eight** values per pass, versus sixteen in zmm: `half_dim` is
                // a multiple of 16 (`dim` is a multiple of 32), and therefore a multiple of 8.
                for j in (0..half_dim).step_by(8) {
                    let o = (j * 2) as i32;
                    for (reg, base, off) in
                        [(a, rdi, o), (b, rdi, o + hoff), (co, rsi, o), (sn, rdx, o)]
                    {
                        asm.vpmovzxwd(YMM[reg.idx()], xmmword_ptr(base + off))?;
                        asm.vpslld(YMM[reg.idx()], YMM[reg.idx()], 16u32)?;
                    }
                    // a' = a·c − b·s
                    asm.vmulps(YMM[ra.idx()], YMM[a.idx()], YMM[co.idx()])?;
                    asm.vfnmadd231ps(YMM[ra.idx()], YMM[b.idx()], YMM[sn.idx()])?;
                    // b' = a·s + b·c
                    asm.vmulps(YMM[rb.idx()], YMM[a.idx()], YMM[sn.idx()])?;
                    asm.vfmadd231ps(YMM[rb.idx()], YMM[b.idx()], YMM[co.idx()])?;
                    cvt_bf16(&mut asm, x, ra, t)?;
                    asm.vmovdqu(xmmword_ptr(rdi + o), XMM[x.x()])?;
                    cvt_bf16(&mut asm, x, rb, t)?;
                    asm.vmovdqu(xmmword_ptr(rdi + (o + hoff)), XMM[x.x()])?;
                }
            }

            // ── y = cap · tanh(x / cap)  (softcapping of logits, Gemma) ───
            //
            // rdi = x, rsi = y, xmm0 = cap (saved in r10d).
            //
            // `tanh(u) = (1 − e^{−2u})/(1 + e^{−2u})` evaluated on `|u|`: this
            // bounds the argument of `exp` to `≤ 0` — the safe domain — and makes
            // saturation exact. The sign is reinjected by a `vpor`.
            ElemOp::Softcap => {
                let capv = f.low("softcap: broadcast cap")?;
                let inv = f.low("softcap:−2/cap")?;
                let xb = f.low("softcap:block of x")?;
                let a = f.low("softcap:argument")?;
                let sg = f.low("softcap:sign")?;
                let t1 = f.low("softcap:temp 1")?;
                let t2 = f.low("softcap:temp 2")?;
                let t3 = f.low("softcap:temp 3")?;
                let tmp = f.low("softcap:rounding scratch")?;
                let out = f.low_run(2, "softcap:output halves")?;
                // `cap` comes from the full register: the table has clobbered xmm0.
                asm.vmovd(XMM[capv.x()], r10d)?;
                asm.vbroadcastss(YMM[capv.idx()], XMM[capv.x()])?;
                // `−2/cap`, **one** division in the kernel setup, never in the
                // loop. No `vrcpps`: its 12 bits would skew the
                // slope near 0, where `tanh(u) ≈ u`.
                asm.vmovups(YMM[inv.idx()], c(constant::NEG2))?;
                asm.vdivps(YMM[inv.idx()], YMM[inv.idx()], YMM[capv.idx()])?;
                for cb in 0..nb {
                    asm.vmovdqu(YMM[xb.idx()], ymmword_ptr(rdi + (cb * 32) as i32))?;
                    for half in 0..2 {
                        widen_half(&mut asm, a, xb, half)?;
                        asm.vpand(YMM[sg.idx()], YMM[a.idx()], c(constant::SGN))?;
                        asm.vpand(YMM[a.idx()], YMM[a.idx()], c(constant::ABS))?;
                        // a ← −2·|x|/cap ≤ 0
                        asm.vmulps(YMM[a.idx()], YMM[a.idx()], YMM[inv.idx()])?;
                        exp_inline(&mut asm, a, t1, t2, t3)?;
                        asm.vmovups(YMM[t3.idx()], c(constant::FONE))?;
                        asm.vsubps(YMM[t1.idx()], YMM[t3.idx()], YMM[a.idx()])?;
                        asm.vaddps(YMM[t2.idx()], YMM[t3.idx()], YMM[a.idx()])?;
                        asm.vdivps(YMM[a.idx()], YMM[t1.idx()], YMM[t2.idx()])?;
                        asm.vmulps(YMM[a.idx()], YMM[a.idx()], YMM[capv.idx()])?;
                        asm.vpor(YMM[a.idx()], YMM[a.idx()], YMM[sg.idx()])?;
                        cvt_bf16(&mut asm, out[half], a, tmp)?;
                    }
                    asm.vinserti128(YMM[out[0].idx()], YMM[out[0].idx()], XMM[out[1].x()], 1)?;
                    asm.vmovdqu(ymmword_ptr(rsi + (cb * 32) as i32), YMM[out[0].idx()])?;
                }
            }

            // ── y = bf16(Σᵣ x[r] · w[r]) ──────────────────────────────────
            //
            // rdi = x (`taps` rows of `dim`), rsi = w (`[taps][dim]`), rdx = y.
            ElemOp::DepthwiseRows(taps) => {
                let xb = f.low("dw:block of a row")?;
                let wb = f.low("dw:block of the kernel")?;
                let xw = f.low("dw:row widened")?;
                let ww = f.low("dw:kernel widened")?;
                let tmp = f.low("dw:rounding scratch")?;
                let acc = f.low_run(2, "dw:sums of the halves")?;
                let out = f.low_run(2, "dw:output halves")?;
                let row = (spec.dim * 2) as i32;
                for cb in 0..nb {
                    for &a in &acc {
                        asm.vxorps(YMM[a.idx()], YMM[a.idx()], YMM[a.idx()])?;
                    }
                    for r in 0..taps as i32 {
                        asm.vmovdqu(
                            YMM[xb.idx()],
                            ymmword_ptr(rdi + (r * row + (cb * 32) as i32)),
                        )?;
                        asm.vmovdqu(
                            YMM[wb.idx()],
                            ymmword_ptr(rsi + (r * row + (cb * 32) as i32)),
                        )?;
                        for (half, &a) in acc.iter().enumerate() {
                            widen_half(&mut asm, xw, xb, half)?;
                            widen_half(&mut asm, ww, wb, half)?;
                            asm.vfmadd231ps(YMM[a.idx()], YMM[xw.idx()], YMM[ww.idx()])?;
                        }
                    }
                    for (half, &a) in acc.iter().enumerate() {
                        cvt_bf16(&mut asm, out[half], a, tmp)?;
                    }
                    asm.vinserti128(YMM[out[0].idx()], YMM[out[0].idx()], XMM[out[1].x()], 1)?;
                    asm.vmovdqu(ymmword_ptr(rdx + (cb * 32) as i32), YMM[out[0].idx()])?;
                }
            }
            // Its own emitter (`emit_local_attn.rs`), the same on both x86 arms.
            ElemOp::LocalAttention { .. } => {
                return Err(EmitErr::Alloc(Decline::Empty));
            }
            // Not ported to this arm yet: the vision tower of 26B-A4B (`Standardize`) runs on
            // AVX-512.
            ElemOp::Standardize => {
                return Err(EmitErr::Alloc(Decline::Empty));
            }
            // ── y = (x − μ) · rsqrt(σ² + ε) · g + b ───────────────────────
            //
            // rdi = x, rsi = w (`g` then `b`), rdx = y, xmm0 = eps (saved in r10d). The three
            // passes of the AVX-512 arm.
            ElemOp::LayerNorm => {
                let n_acc = 4.min(2 * nb);
                let acc = f.low_run(n_acc, "ln:chains of the sums")?;
                let ld = f.low("ln:block read / temporary scalar")?;
                let xw = f.low("ln:x widened")?;
                let mu = f.low("ln:μ broadcast")?;
                let inv = f.low("ln:1/σ broadcast")?;
                let scalar =
                    |asm: &mut CodeAssembler, mul: f32| -> std::result::Result<(), IcedError> {
                        let a0 = acc[0];
                        for a in &acc[1..] {
                            asm.vaddps(YMM[a0.idx()], YMM[a0.idx()], YMM[a.idx()])?;
                        }
                        hsum(asm, a0, ld)?;
                        asm.mov(eax, mul.to_bits())?;
                        asm.vmovd(XMM[ld.x()], eax)?;
                        asm.vmulss(XMM[a0.x()], XMM[a0.x()], XMM[ld.x()])?;
                        Ok(())
                    };
                for a in &acc {
                    asm.vxorps(YMM[a.idx()], YMM[a.idx()], YMM[a.idx()])?;
                }
                for cb in 0..nb {
                    asm.vmovdqu(YMM[ld.idx()], ymmword_ptr(rdi + (cb * 32) as i32))?;
                    for half in 0..2 {
                        widen_half(&mut asm, xw, ld, half)?;
                        let a = acc[(2 * cb + half) % n_acc];
                        asm.vaddps(YMM[a.idx()], YMM[a.idx()], YMM[xw.idx()])?;
                    }
                }
                scalar(&mut asm, 1.0 / spec.dim as f32)?;
                asm.vbroadcastss(YMM[mu.idx()], XMM[acc[0].x()])?;
                for a in &acc {
                    asm.vxorps(YMM[a.idx()], YMM[a.idx()], YMM[a.idx()])?;
                }
                for cb in 0..nb {
                    asm.vmovdqu(YMM[ld.idx()], ymmword_ptr(rdi + (cb * 32) as i32))?;
                    for half in 0..2 {
                        widen_half(&mut asm, xw, ld, half)?;
                        asm.vsubps(YMM[xw.idx()], YMM[xw.idx()], YMM[mu.idx()])?;
                        let a = acc[(2 * cb + half) % n_acc];
                        asm.vfmadd231ps(YMM[a.idx()], YMM[xw.idx()], YMM[xw.idx()])?;
                    }
                }
                scalar(&mut asm, 1.0 / spec.dim as f32)?;
                let a0 = acc[0];
                asm.vmovd(XMM[ld.x()], r10d)?;
                asm.vaddss(XMM[a0.x()], XMM[a0.x()], XMM[ld.x()])?;
                asm.vsqrtss(XMM[a0.x()], XMM[a0.x()], XMM[a0.x()])?;
                asm.mov(eax, 1.0f32.to_bits())?;
                asm.vmovd(XMM[ld.x()], eax)?;
                asm.vdivss(XMM[a0.x()], XMM[ld.x()], XMM[a0.x()])?;
                asm.vbroadcastss(YMM[inv.idx()], XMM[a0.x()])?;
                for a in acc {
                    f.free_low(a);
                }
                f.free_low(ld);
                let xb = f.low("ln:block of x")?;
                let gb = f.low("ln:block of g")?;
                let bb = f.low("ln:block of b")?;
                let gw = f.low("ln:g widened")?;
                let bw = f.low("ln:b widened")?;
                let tmp = f.low("ln:rounding scratch")?;
                let out = f.low_run(2, "ln:output halves")?;
                let boff = (spec.dim * 2) as i32;
                for cb in 0..nb {
                    let o = (cb * 32) as i32;
                    asm.vmovdqu(YMM[xb.idx()], ymmword_ptr(rdi + o))?;
                    asm.vmovdqu(YMM[gb.idx()], ymmword_ptr(rsi + o))?;
                    asm.vmovdqu(YMM[bb.idx()], ymmword_ptr(rsi + (boff + o)))?;
                    for half in 0..2 {
                        widen_half(&mut asm, xw, xb, half)?;
                        widen_half(&mut asm, gw, gb, half)?;
                        widen_half(&mut asm, bw, bb, half)?;
                        asm.vsubps(YMM[xw.idx()], YMM[xw.idx()], YMM[mu.idx()])?;
                        asm.vmulps(YMM[xw.idx()], YMM[xw.idx()], YMM[inv.idx()])?;
                        asm.vfmadd213ps(YMM[xw.idx()], YMM[gw.idx()], YMM[bw.idx()])?;
                        cvt_bf16(&mut asm, out[half], xw, tmp)?;
                    }
                    asm.vinserti128(YMM[out[0].idx()], YMM[out[0].idx()], XMM[out[1].x()], 1)?;
                    asm.vmovdqu(ymmword_ptr(rdx + o), YMM[out[0].idx()])?;
                }
            }

            // ── y = gelu(gate) · up, the exact GeLU ───────────────────────
            //
            // rdi = gate, rsi = up, rdx = y. The `erfcc` of the AVX-512 arm, its constants in the
            // table.
            ElemOp::GeluErf => {
                let gb = f.low("gelu:gate block")?;
                let ub = f.low("gelu:up block")?;
                let g = f.low("gelu:gate widened")?;
                let u = f.low("gelu:up widened")?;
                let a = f.low("gelu:a = |g|/√2")?;
                let t = f.low("gelu:t")?;
                let p = f.low("gelu:P(t), then the result")?;
                let sg = f.low("gelu:1 − ½·E")?;
                let tmp = f.low("gelu:rounding scratch")?;
                let te = f.low_run(3, "gelu:exp temporaries")?;
                let out = f.low_run(2, "gelu:output halves")?;
                let block = |asm: &mut CodeAssembler,
                             pg: AsmRegister64,
                             pu: AsmRegister64,
                             py: AsmRegister64,
                             o: i32|
                 -> std::result::Result<(), IcedError> {
                    asm.vmovdqu(YMM[gb.idx()], ymmword_ptr(pg + o))?;
                    asm.vmovdqu(YMM[ub.idx()], ymmword_ptr(pu + o))?;
                    for half in 0..2 {
                        widen_half(asm, g, gb, half)?;
                        widen_half(asm, u, ub, half)?;
                        asm.vandps(YMM[a.idx()], YMM[g.idx()], c(constant::ABS))?;
                        asm.vmulps(YMM[a.idx()], YMM[a.idx()], c(constant::ISQ))?;
                        // t = 1 / (½·a + 1), fused as the AVX-512 arm.
                        asm.vmovaps(YMM[t.idx()], YMM[a.idx()])?;
                        asm.vmovups(YMM[p.idx()], c(constant::HALF))?;
                        asm.vfmadd213ps(YMM[t.idx()], YMM[p.idx()], c(constant::FONE))?;
                        asm.vmovups(YMM[p.idx()], c(constant::FONE))?;
                        asm.vdivps(YMM[t.idx()], YMM[p.idx()], YMM[t.idx()])?;
                        asm.vmovups(YMM[p.idx()], c(constant::P0 + 9))?;
                        for i in (0..9).rev() {
                            asm.vfmadd213ps(YMM[p.idx()], YMM[t.idx()], c(constant::P0 + i))?;
                        }
                        asm.vfnmadd231ps(YMM[p.idx()], YMM[a.idx()], YMM[a.idx()])?;
                        exp_inline(asm, p, te[0], te[1], te[2])?;
                        // Φ = ½·E where g < 0, 1 − ½·E elsewhere: `vblendvps` reads the sign bit of
                        // g.
                        asm.vmulps(YMM[p.idx()], YMM[p.idx()], YMM[t.idx()])?;
                        asm.vmulps(YMM[p.idx()], YMM[p.idx()], c(constant::HALF))?;
                        asm.vmovups(YMM[sg.idx()], c(constant::FONE))?;
                        asm.vsubps(YMM[sg.idx()], YMM[sg.idx()], YMM[p.idx()])?;
                        asm.vblendvps(YMM[p.idx()], YMM[sg.idx()], YMM[p.idx()], YMM[g.idx()])?;
                        asm.vmulps(YMM[p.idx()], YMM[p.idx()], YMM[g.idx()])?;
                        asm.vmulps(YMM[p.idx()], YMM[p.idx()], YMM[u.idx()])?;
                        cvt_bf16(asm, out[half], p, tmp)?;
                    }
                    asm.vinserti128(YMM[out[0].idx()], YMM[out[0].idx()], XMM[out[1].x()], 1)?;
                    asm.vmovdqu(ymmword_ptr(py + o), YMM[out[0].idx()])?;
                    Ok(())
                };
                match unroll {
                    None => {
                        for cb in 0..nb {
                            block(&mut asm, rdi, rsi, rdx, (cb * 32) as i32)?;
                        }
                    }
                    Some(unroll) => {
                        asm.mov(rcx, rdi)?;
                        asm.mov(r8, rsi)?;
                        asm.mov(r9, rdx)?;
                        asm.mov(r11d, (nb / unroll) as u32)?;
                        let mut top = asm.create_label();
                        asm.set_label(&mut top)?;
                        for j in 0..unroll {
                            block(&mut asm, rcx, r8, r9, (j * 32) as i32)?;
                        }
                        let step = (unroll * 32) as i32;
                        asm.add(rcx, step)?;
                        asm.add(r8, step)?;
                        asm.add(r9, step)?;
                        asm.dec(r11d)?;
                        asm.jnz(top)?;
                    }
                }
            }

            // ── y = min(max(x, lo), hi) ───────────────────────────────────
            //
            // rdi = x, rsi = y, xmm0 = lo (saved in r10d), xmm1 = hi (saved in r11d).
            ElemOp::Clamp => {
                let lo = f.low("clamp:lo")?;
                let hi = f.low("clamp:hi")?;
                let xb = f.low("clamp:block of x")?;
                let xw = f.low("clamp:x widened")?;
                let tmp = f.low("clamp:rounding scratch")?;
                let out = f.low_run(2, "clamp:output halves")?;
                asm.vmovd(XMM[lo.x()], r10d)?;
                asm.vbroadcastss(YMM[lo.idx()], XMM[lo.x()])?;
                asm.vmovd(XMM[hi.x()], r11d)?;
                asm.vbroadcastss(YMM[hi.idx()], XMM[hi.x()])?;
                for cb in 0..nb {
                    asm.vmovdqu(YMM[xb.idx()], ymmword_ptr(rdi + (cb * 32) as i32))?;
                    for half in 0..2 {
                        widen_half(&mut asm, xw, xb, half)?;
                        asm.vmaxps(YMM[xw.idx()], YMM[xw.idx()], YMM[lo.idx()])?;
                        asm.vminps(YMM[xw.idx()], YMM[xw.idx()], YMM[hi.idx()])?;
                        cvt_bf16(&mut asm, out[half], xw, tmp)?;
                    }
                    asm.vinserti128(YMM[out[0].idx()], YMM[out[0].idx()], XMM[out[1].x()], 1)?;
                    asm.vmovdqu(ymmword_ptr(rsi + (cb * 32) as i32), YMM[out[0].idx()])?;
                }
            }

            // ── y = bf16(Σᵣ x[r] · inv) · s ───────────────────────────────
            //
            // rdi = x (`rows` rows of `dim`), rsi = y, xmm0 = s (saved in r10d), xmm1 = inv (saved
            // in r11d).
            ElemOp::PoolRows(rows) => {
                let sc = f.low("pool:s")?;
                let iv = f.low("pool:inv")?;
                let xb = f.low("pool:block of a row")?;
                let xw = f.low("pool:row widened")?;
                let tmp = f.low("pool:rounding scratch")?;
                let acc = f.low_run(2, "pool:sums of the halves")?;
                let out = f.low_run(2, "pool:output halves")?;
                asm.vmovd(XMM[sc.x()], r10d)?;
                asm.vbroadcastss(YMM[sc.idx()], XMM[sc.x()])?;
                asm.vmovd(XMM[iv.x()], r11d)?;
                asm.vbroadcastss(YMM[iv.idx()], XMM[iv.x()])?;
                let row = (spec.dim * 2) as i32;
                for cb in 0..nb {
                    for &a in &acc {
                        asm.vxorps(YMM[a.idx()], YMM[a.idx()], YMM[a.idx()])?;
                    }
                    for r in 0..rows as i32 {
                        asm.vmovdqu(
                            YMM[xb.idx()],
                            ymmword_ptr(rdi + (r * row + (cb * 32) as i32)),
                        )?;
                        for (half, &a) in acc.iter().enumerate() {
                            widen_half(&mut asm, xw, xb, half)?;
                            asm.vfmadd231ps(YMM[a.idx()], YMM[xw.idx()], YMM[iv.idx()])?;
                        }
                    }
                    for (half, &a) in acc.iter().enumerate() {
                        // The mean rounded to bf16, widened back, times `s`, rounded again.
                        cvt_bf16(&mut asm, out[half], a, tmp)?;
                        asm.vpmovzxwd(YMM[a.idx()], XMM[out[half].x()])?;
                        asm.vpslld(YMM[a.idx()], YMM[a.idx()], 16u32)?;
                        asm.vmulps(YMM[a.idx()], YMM[a.idx()], YMM[sc.idx()])?;
                        cvt_bf16(&mut asm, out[half], a, tmp)?;
                    }
                    asm.vinserti128(YMM[out[0].idx()], YMM[out[0].idx()], XMM[out[1].x()], 1)?;
                    asm.vmovdqu(ymmword_ptr(rsi + (cb * 32) as i32), YMM[out[0].idx()])?;
                }
            }
        }

        asm.add(rsp, TABLE)?;
        asm.vzeroupper()?;
        asm.ret()?;
        Ok(asm.assemble(0)?)
    }
}
