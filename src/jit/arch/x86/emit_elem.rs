//! The elementwise kernels of a layer, AVX-512: one blob per `(op, dim)`.
//!
//! | kernel | site in the layer | formula |
//! |---|---|---|
//! | [`ElemOp::RmsNorm`], [`ElemOp::RmsNormGemma`] | the layer norms, the final norm | `y = x · rsqrt(Σx²/d + ε) · w` (Gemma: `· (1 + w)`) |
//! | [`ElemOp::SwiGlu`] | the MLP activation | `y = gate · σ(gate) · up` |
//! | [`ElemOp::SigmoidMul`] | the attention output gate | `y = x · σ(gate)` |
//! | [`ElemOp::ResidualAdd`] | the two residuals | `a += b` |
//! | [`ElemOp::Rope`] | the rotation of Q and K | `(x, x') ← (x·c − x'·s, x·s + x'·c)` |
//! | [`ElemOp::GeluTanh`], [`ElemOp::GeluErf`] | the MLP activation (Gemma 4, XLM-RoBERTa) | `y = gelu(gate) · up` |
//! | [`ElemOp::Scale`] | the scales of the Gemma 4 input branch | `y = x · s` |
//! | [`ElemOp::Softcap`] | the logit softcapping | `y = cap · tanh(x / cap)` |
//! | [`ElemOp::Clamp`] | the clipped linears of the vision tower | `y = min(max(x, lo), hi)` |
//! | [`ElemOp::PoolRows`] | the average pooling of the vision tower | `y = (Σᵣ x[r] · inv) · s` |
//! | [`ElemOp::Standardize`] | the standardization of the pooled vision tokens | `y = (x · r − b) · c` |
//! | [`ElemOp::LayerNorm`] | the norms of the encoders | `y = (x − μ) · rsqrt(σ² + ε) · g + b` |
//! | [`ElemOp::DepthwiseRows`] | the depthwise convolution of the audio tower | `y = Σᵣ x[r] · w[r]` |
//!
//! [`ElemOp::LocalAttention`] has its own emitter (`emit_local_attn.rs`).
//!
//! Everything is bf16 in memory and accumulates in f32 in registers. Two
//! consequences: the sum of squares of an RMSNorm is a dot product of `x`
//! with itself, both operands bf16, one `vdpbf16ps` per 32 elements; and no
//! kernel converts at its input or output — the producer writes what the
//! consumer reads. `dim` is an immediate: one shape, one kernel, no remainder
//! check at run time. Above `LOOP_MIN_DIM` the body loops over blocks
//! instead of unrolling, so a 9216-wide SwiGLU stays in L1i.

pub(crate) use crate::jit::emit_elem::{loop_unroll, ElemOp, ElemSpec, LoopArm};

pub use x86::emit_elem;

/// The DeltaNet emitters (`emit_conv.rs`, `emit_gated_norm.rs`) use the same
/// exponential (`ExpConsts`, `exp_inline`, `EXP_MAX88`): one `exp` polynomial
/// in the engine, not two.
pub(crate) mod x86 {
    use super::{loop_unroll, ElemOp, ElemSpec, LoopArm};
    use crate::core::error::{HerbertError, Result};
    use crate::jit::ir::Decline;
    use crate::jit::regs::{RegFile, Vlow, V};
    use iced_x86::code_asm::*;
    use iced_x86::IcedError;

    const ZMM: [AsmRegisterZmm; 32] = [
        zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7, zmm8, zmm9, zmm10, zmm11, zmm12, zmm13,
        zmm14, zmm15, zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23, zmm24, zmm25, zmm26,
        zmm27, zmm28, zmm29, zmm30, zmm31,
    ];
    const YMM: [AsmRegisterYmm; 32] = [
        ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12, ymm13,
        ymm14, ymm15, ymm16, ymm17, ymm18, ymm19, ymm20, ymm21, ymm22, ymm23, ymm24, ymm25, ymm26,
        ymm27, ymm28, ymm29, ymm30, ymm31,
    ];
    const XMM: [AsmRegisterXmm; 32] = [
        xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13,
        xmm14, xmm15, xmm16, xmm17, xmm18, xmm19, xmm20, xmm21, xmm22, xmm23, xmm24, xmm25, xmm26,
        xmm27, xmm28, xmm29, xmm30, xmm31,
    ];

    use crate::jit::exp_constants::{
        EXP_C0, EXP_C1, EXP_C2, EXP_C3, EXP_C4, EXP_LOG2E, EXP_NEG87, EXP_ONE_BITS,
    };
    /// Upper **bound** of the domain of `exp_inline`, for SwiGLU only.
    ///
    /// `exp(88)` equals 1.65e38 and still fits in f32; beyond that, the
    /// reconstruction `2^k` overflows the exponent field and returns an
    /// **arbitrary** number instead of `+inf`. SwiGLU is the only caller whose
    /// argument can be positive (it computes `exp(−g)`), and Qwen3 produces
    /// very negative `g` values in its last layers: without this bound, a
    /// `g < −88` caused a huge `silu(g)` instead of ~0.
    pub(crate) const EXP_MAX88: f32 = 88.0;

    /// The eight polynomial registers, **allocated** and no longer assigned to the
    /// main. High zone for consistency with `emit_sdpa`: this kernel calls
    /// nothing, so the `vzeroupper` constraint does not apply here, but a role
    /// that lives from start to finish has nothing to do in the zone contested
    /// by horizontal reductions.
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct ExpConsts {
        neg87: V,
        log2e: V,
        c4: V,
        c3: V,
        c2: V,
        c1: V,
        c0: V,
        /// `0x3F80_0000` — the exponent bias of the reconstruction, **and**
        /// `1,0f` for those who want a diffuse 1 (the Softcap, the DeltaNet conv).
        pub(crate) one: V,
    }

    impl ExpConsts {
        pub(crate) fn alloc(f: &mut RegFile) -> std::result::Result<Self, Decline> {
            Ok(Self {
                neg87: f.high("exp:−87.34")?,
                log2e: f.high("exp:log2(e)")?,
                c4: f.high("exp:c4")?,
                c3: f.high("exp:c3")?,
                c2: f.high("exp:c2")?,
                c1: f.high("exp:c1")?,
                c0: f.high("exp:c0")?,
                one: f.high("exp:1.0")?,
            })
        }
    }

    pub(crate) fn bcast_imm(
        asm: &mut CodeAssembler,
        dst: V,
        bits: u32,
    ) -> std::result::Result<(), IcedError> {
        asm.mov(eax, bits)?;
        asm.vpbroadcastd(ZMM[dst.idx()], eax)?;
        Ok(())
    }

    pub(crate) fn load_exp_constants(
        asm: &mut CodeAssembler,
        k: &ExpConsts,
    ) -> std::result::Result<(), IcedError> {
        bcast_imm(asm, k.neg87, EXP_NEG87.to_bits())?;
        bcast_imm(asm, k.log2e, EXP_LOG2E.to_bits())?;
        bcast_imm(asm, k.c4, EXP_C4.to_bits())?;
        bcast_imm(asm, k.c3, EXP_C3.to_bits())?;
        bcast_imm(asm, k.c2, EXP_C2.to_bits())?;
        bcast_imm(asm, k.c1, EXP_C1.to_bits())?;
        bcast_imm(asm, k.c0, EXP_C0.to_bits())?;
        bcast_imm(asm, k.one, EXP_ONE_BITS)?;
        Ok(())
    }

    /// `io = exp(io)`, the engine's exponential polynomial (`ExpConsts`).
    ///
    /// **Domain: `io ≤ 88`.** The lower bound is applied here (`vmaxps`
    /// at −87.34) because it holds for all **callers**. The upper bound
    /// does not: **softmax** never calls with a positive argument, and
    /// imposing a `vminps` per **block** would be costly in its hot **loop**.
    /// **Thus, it is up to the caller whose domain contains positives**
    /// to bound its **input** — see [`ElemOp::SwiGlu`] and [`EXP_MAX88`].
    pub(crate) fn exp_inline(
        asm: &mut CodeAssembler,
        k: &ExpConsts,
        io: V,
        t1: V,
        t2: V,
        t3: V,
    ) -> std::result::Result<(), IcedError> {
        let (io, t1, t2, t3) = (io.idx(), t1.idx(), t2.idx(), t3.idx());
        asm.vmaxps(ZMM[io], ZMM[io], ZMM[k.neg87.idx()])?;
        asm.vmulps(ZMM[t1], ZMM[io], ZMM[k.log2e.idx()])?;
        asm.vrndscaleps(ZMM[t2], ZMM[t1], 0)?;
        asm.vsubps(ZMM[t1], ZMM[t1], ZMM[t2])?;
        asm.vmovaps(ZMM[t3], ZMM[k.c4.idx()])?;
        asm.vfmadd213ps(ZMM[t3], ZMM[t1], ZMM[k.c3.idx()])?;
        asm.vfmadd213ps(ZMM[t3], ZMM[t1], ZMM[k.c2.idx()])?;
        asm.vfmadd213ps(ZMM[t3], ZMM[t1], ZMM[k.c1.idx()])?;
        asm.vfmadd213ps(ZMM[t3], ZMM[t1], ZMM[k.c0.idx()])?;
        asm.vcvtps2dq(ZMM[t2], ZMM[t2])?;
        asm.vpslld(ZMM[t2], ZMM[t2], 23u32)?;
        asm.vpaddd(ZMM[t2], ZMM[t2], ZMM[k.one.idx()])?;
        asm.vmulps(ZMM[io], ZMM[t3], ZMM[t2])?;
        Ok(())
    }

    /// Extends the 16 lower (`half=0`) or upper (`half=1`) bf16 of `src` to f32
    /// into `dst`. A zmm holds 32 bf16; processing them requires both
    /// halves.
    /// All its shapes are EVEX (`vpmovzxwd`, `vextracti64x4`): the area is
    /// free on both sides.
    fn widen_half(
        asm: &mut CodeAssembler,
        dst: V,
        src: V,
        half: usize,
    ) -> std::result::Result<(), IcedError> {
        let (d, s) = (dst.idx(), src.idx());
        if half == 0 {
            asm.vpmovzxwd(ZMM[d], YMM[src.y_evex()])?;
        } else {
            asm.vextracti64x4(YMM[dst.y_evex()], ZMM[s], 1)?;
            asm.vpmovzxwd(ZMM[d], YMM[dst.y_evex()])?;
        }
        asm.vpslld(ZMM[d], ZMM[d], 16u32)?;
        Ok(())
    }

    /// Horizontal sum of a zmm f32 → scalar in `xmm[dst]`.
    ///
    /// [`Vlow`] : `vextractf128`, `vhaddps` and `vaddps xmm` are emitted in VEX,
    /// which stops at `xmm15`.
    pub(crate) fn hsum(
        asm: &mut CodeAssembler,
        dst: Vlow,
        t: Vlow,
    ) -> std::result::Result<(), IcedError> {
        let (dst, t) = (dst.x(), t.x());
        asm.vextractf64x4(YMM[t], ZMM[dst], 1)?;
        asm.vaddps(YMM[dst], YMM[dst], YMM[t])?;
        asm.vextractf128(XMM[t], YMM[dst], 1)?;
        asm.vaddps(XMM[dst], XMM[dst], XMM[t])?;
        asm.vhaddps(XMM[dst], XMM[dst], XMM[dst])?;
        asm.vhaddps(XMM[dst], XMM[dst], XMM[dst])?;
        Ok(())
    }

    /// `k2` = the low `n` lanes (`n < 32`): the partial last block of a row whose width is not a
    /// multiple of 32 ([`ElemSpec::tails`]), read by zero-masked loads, written by masked
    /// stores; the lanes past the row are never touched, the arithmetic on them (zeros) is
    /// thrown away. Clobbers `eax`.
    fn set_tail_mask(asm: &mut CodeAssembler, n: usize) -> std::result::Result<(), IcedError> {
        debug_assert!(n > 0 && n < 32);
        asm.mov(eax, ((1u64 << n) - 1) as u32)?;
        asm.kmovd(k2, eax)
    }

    /// `dst ← [at]`, the `k2` lanes only (the others zero) when `tail`.
    fn load_block(
        asm: &mut CodeAssembler,
        dst: V,
        at: AsmMemoryOperand,
        tail: bool,
    ) -> std::result::Result<(), IcedError> {
        if tail {
            asm.vmovdqu16(ZMM[dst.idx()].k2().z(), at)
        } else {
            asm.vmovdqu16(ZMM[dst.idx()], at)
        }
    }

    /// `[at] ← src`, the `k2` lanes only when `tail`.
    fn store_block(
        asm: &mut CodeAssembler,
        at: AsmMemoryOperand,
        src: V,
        tail: bool,
    ) -> std::result::Result<(), IcedError> {
        if tail {
            asm.vmovdqu16(at.k2(), ZMM[src.idx()])
        } else {
            asm.vmovdqu16(at, ZMM[src.idx()])
        }
    }

    /// Generates the kernel described by `spec`. Position-independent code, with no
    /// rodata or relocation: constants are materialized in registers.
    pub fn emit_elem(spec: &ElemSpec) -> Result<Vec<u8>> {
        emit_with(spec, loop_unroll(spec.op, spec.dim, LoopArm::Zmm))
    }

    fn emit_with(spec: &ElemSpec, unroll: Option<usize>) -> Result<Vec<u8>> {
        spec.check()
            .map_err(|d| HerbertError::Backend(format!("backend_jit elem: {d}")))?;
        assemble(spec, unroll).map_err(|e| {
            HerbertError::Backend(format!(
                "backend_jit: emission {} dim={}: {e}",
                spec.op.name(),
                spec.dim
            ))
        })
    }

    /// The two ways an emission can fail — see `emit_sdpa`.
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

    fn assemble(spec: &ElemSpec, unroll: Option<usize>) -> std::result::Result<Vec<u8>, EmitErr> {
        let mut asm = CodeAssembler::new(64)?;
        let nb = spec.blocks32();
        // The bf16 of a partial last block (0: none), and the blocks with it.
        let tail = if spec.tails() { spec.dim % 32 } else { 0 };
        let nbt = nb + usize::from(tail > 0);
        // An elementary kernel, a role table. Each arm is independent: none
        // assumes what another left behind.
        let mut f = RegFile::new();
        match spec.op {
            // ── y = x · rsqrt(Σx²/d + ε) · w   (· (1+w) in the Gemma variant) ─
            //
            // rdi = x, rsi = w, rdx = y, xmm0 = eps.
            ElemOp::RmsNorm | ElemOp::RmsNormGemma => {
                // The only difference between the two: `w` or `1 + w`. The addition is
                // performed on `w` **widened to f32**, never on bf16 in
                // memory — see the note for `ElemOp::RmsNormGemma`.
                let plus_one = spec.op == ElemOp::RmsNormGemma;
                // Pass 1: Σx². The two operands of the dot product are
                // `x` itself, in bf16: a `vdpbf16ps` per block of 32.
                // Four chains to avoid paying latency in series. Low zone:
                // chain 0 is reduced horizontally (VEX) then
                // processed in scalar.
                let n_acc = 4.min(nbt);
                let acc = f.low_run(n_acc, "rms:chains of Σx²")?;
                let ld = f.low("rms:loaded block / scalar temporary")?;
                // `eps` arrives in xmm0, which the horizontal sum will overwrite.
                // Saving it in a zmm was a bug: zeroing the
                // accumulators would erase it. An **integer** register is out
                // of reach of everything this kernel does — and that's why
                // it doesn't appear in the vector roles table.
                asm.vmovd(r10d, XMM[0])?;
                if tail > 0 {
                    set_tail_mask(&mut asm, tail)?;
                }
                for a in &acc {
                    asm.vxorps(ZMM[a.idx()], ZMM[a.idx()], ZMM[a.idx()])?;
                }
                // The partial last block read with zeros past the row: they add nothing to Σx².
                for c in 0..nbt {
                    load_block(
                        &mut asm,
                        ld.into(),
                        zmmword_ptr(rdi + (c * 64) as i32),
                        c == nb,
                    )?;
                    asm.vdpbf16ps(ZMM[acc[c % n_acc].idx()], ZMM[ld.idx()], ZMM[ld.idx()])?;
                }
                let a0 = acc[0];
                for a in &acc[1..] {
                    asm.vaddps(ZMM[a0.idx()], ZMM[a0.idx()], ZMM[a.idx()])?;
                }
                hsum(&mut asm, a0, ld)?;
                // rsqrt(Σx²/dim + eps), en f32 scalar.
                asm.mov(eax, (1.0f32 / spec.dim as f32).to_bits())?;
                asm.vmovd(XMM[ld.x()], eax)?;
                asm.vmulss(XMM[a0.x()], XMM[a0.x()], XMM[ld.x()])?;
                asm.vmovd(XMM[ld.x()], r10d)?; // eps, returned from its integer register
                asm.vaddss(XMM[a0.x()], XMM[a0.x()], XMM[ld.x()])?;
                asm.vsqrtss(XMM[a0.x()], XMM[a0.x()], XMM[a0.x()])?;
                asm.mov(eax, 1.0f32.to_bits())?;
                asm.vmovd(XMM[ld.x()], eax)?;
                asm.vdivss(XMM[a0.x()], XMM[ld.x()], XMM[a0.x()])?;
                let inv = f.any("rms:1/rms broadcast")?;
                asm.vbroadcastss(ZMM[inv.idx()], XMM[a0.x()])?;

                // The chains and the temporary from pass 1 are dead:
                // saying so frees their space for pass 2, which needs it.
                for a in acc {
                    f.free_low(a);
                }
                f.free_low(ld);

                // Pass 2: y = x · inv_rms · w, block of 32 by block of 32.
                let xb = f.any("rms:block of x")?;
                let wb = f.any("rms:block of w")?;
                let xw = f.any("rms:x expanded")?;
                let ww = f.any("rms:w expanded")?;
                // The 1.0 of the Gemma variant, broadcast once for the whole
                // kernel — allocated only if used, like the
                // geluTanh constants.
                let one = if plus_one {
                    let r = f.any("rms_gemma:1.0")?;
                    bcast_imm(&mut asm, r, 1.0f32.to_bits())?;
                    Some(r)
                } else {
                    None
                };
                let out = f.run(2, "rms: output halves")?;
                for c in 0..nbt {
                    load_block(&mut asm, xb, zmmword_ptr(rdi + (c * 64) as i32), c == nb)?;
                    load_block(&mut asm, wb, zmmword_ptr(rsi + (c * 64) as i32), c == nb)?;
                    for half in 0..2 {
                        widen_half(&mut asm, xw, xb, half)?;
                        widen_half(&mut asm, ww, wb, half)?;
                        if let Some(one) = one {
                            asm.vaddps(ZMM[ww.idx()], ZMM[ww.idx()], ZMM[one.idx()])?;
                        }
                        asm.vmulps(ZMM[xw.idx()], ZMM[xw.idx()], ZMM[inv.idx()])?;
                        asm.vmulps(ZMM[xw.idx()], ZMM[xw.idx()], ZMM[ww.idx()])?;
                        asm.vcvtneps2bf16(YMM[out[half].y_evex()], ZMM[xw.idx()])?;
                    }
                    // The two halves reconstruct a 32 bf16 zmm.
                    asm.vinserti64x4(
                        ZMM[out[0].idx()],
                        ZMM[out[0].idx()],
                        YMM[out[1].y_evex()],
                        1,
                    )?;
                    store_block(
                        &mut asm,
                        zmmword_ptr(rdx + (c * 64) as i32),
                        out[0],
                        c == nb,
                    )?;
                }
            }

            // ── y = gate · σ(gate) · up ───────────────────────────────────
            //
            // rdi = gate, rsi = up, rdx = y. σ(t) = 1/(1+e^{-t}).
            //
            // Three variants of the **same** arm, which differ only by two
            // things: where the sigmoid argument comes from, and whether a final
            // `vmulps` remains.
            //
            // | op | σ argument | vdivps numerator | final ·up |
            // |---|---|---|---|
            // | `SwiGlu`     | `g`                | `g` | yes |
            // | `GeluTanh`   | `g·(K + K·C·g²)`   | `g` | yes |
            // | `SigmoidMul` | `u` (second operand) | `g` | **no** |
            //
            // The last line is the identity
            // `g / (1 + exp(−u)) = g · σ(u)`. The `vdivps` already does all the
            // work, it just needs to not remultiply.
            ElemOp::SwiGlu | ElemOp::GeluTanh | ElemOp::SigmoidMul => {
                // The only difference between the two: the argument passed to the
                // sigmoid. `g` for SwiGLU, `g·(K + K·C·g²)` for geluTanh.
                let cubic = spec.op == ElemOp::GeluTanh;
                // `y = a · σ(b)` : σ applies to the second operand, and the
                // product by `up` does not occur — see the table above.
                let gate_out = spec.op == ElemOp::SigmoidMul;
                let k = ExpConsts::alloc(&mut f)?;
                let one = f.high("swiglu:1.0")?;
                let max88 = f.high("swiglu: upper bound of the domain of the exp")?;
                // GeluTanh constants, allocated only if used:
                // K = 2√(2/π) and K·C.
                let (kk, kc) = if cubic {
                    (
                        Some(f.high("gelutanh:K = 2√(2/π)")?),
                        Some(f.high("gelutanh:K·C")?),
                    )
                } else {
                    (None, None)
                };
                let gb = f.any("swiglu:gate block")?;
                let ub = f.any("swiglu:up block")?;
                let g = f.any("swiglu:gate expanded")?;
                let u = f.any("swiglu:up extended")?;
                let e = f.any("swiglu:exp(-g) then result")?;
                let t = f.run(3, "swiglu:polynomial temporaries")?;
                let out = f.run(2, "swiglu:output halves")?;
                load_exp_constants(&mut asm, &k)?;
                bcast_imm(&mut asm, one, 1.0f32.to_bits())?;
                bcast_imm(&mut asm, max88, EXP_MAX88.to_bits())?;
                if let (Some(kk), Some(kc)) = (kk, kc) {
                    const K: f32 = 1.595_769; // 2·√(2/π)
                    const C: f32 = 0.044_715;
                    bcast_imm(&mut asm, kk, K.to_bits())?;
                    bcast_imm(&mut asm, kc, (K * C).to_bits())?;
                }
                // One block of 32: the same instructions whether the kernel
                // is unrolled (bases rdi/rsi/rdx, immediate offsets) or looped
                // (cursors rcx/r8/r9, offsets within the turn). Keeping a
                // single body is what makes the two forms bit-identical.
                // `last`: the partial last block, its loads and its store masked by `k2`.
                let block = |asm: &mut CodeAssembler,
                             pg: AsmRegister64,
                             pu: AsmRegister64,
                             py: AsmRegister64,
                             o: i32,
                             last: bool|
                 -> std::result::Result<(), IcedError> {
                    load_block(asm, gb, zmmword_ptr(pg + o), last)?;
                    load_block(asm, ub, zmmword_ptr(pu + o), last)?;
                    for half in 0..2 {
                        widen_half(asm, g, gb, half)?;
                        widen_half(asm, u, ub, half)?;
                        // `e` receives the sigmoid argument, negative:
                        // SwiGLU     → −g
                        // geluTanh   → −t, t = g·(K + K·C·g²)
                        // SigmoidMul → −u  (the SECOND operand)
                        if gate_out {
                            asm.vxorps(ZMM[e.idx()], ZMM[e.idx()], ZMM[e.idx()])?;
                            asm.vsubps(ZMM[e.idx()], ZMM[e.idx()], ZMM[u.idx()])?;
                        } else if let (Some(kk), Some(kc)) = (kk, kc) {
                            // t = g·(K·C·g² + K), then we negate. `t[0]` serves as
                            // zero: it is free until `exp_inline`.
                            asm.vmulps(ZMM[e.idx()], ZMM[g.idx()], ZMM[g.idx()])?;
                            asm.vmulps(ZMM[e.idx()], ZMM[e.idx()], ZMM[kc.idx()])?;
                            asm.vaddps(ZMM[e.idx()], ZMM[e.idx()], ZMM[kk.idx()])?;
                            asm.vmulps(ZMM[e.idx()], ZMM[e.idx()], ZMM[g.idx()])?;
                            asm.vxorps(ZMM[t[0].idx()], ZMM[t[0].idx()], ZMM[t[0].idx()])?;
                            asm.vsubps(ZMM[e.idx()], ZMM[t[0].idx()], ZMM[e.idx()])?;
                        } else {
                            asm.vxorps(ZMM[e.idx()], ZMM[e.idx()], ZMM[e.idx()])?;
                            asm.vsubps(ZMM[e.idx()], ZMM[e.idx()], ZMM[g.idx()])?;
                        }
                        // **Upper bounded**: the argument is positive as soon as
                        // `g` is negative, and this is the only call to
                        // `exp_inline` in this case (see `EXP_MAX88`).
                        asm.vminps(ZMM[e.idx()], ZMM[e.idx()], ZMM[max88.idx()])?;
                        exp_inline(asm, &k, e, t[0], t[1], t[2])?;
                        // silu(g) = g / (1 + e) — and for the output gate,
                        // `g / (1 + exp(−u))` **is already** `g · σ(u)` : the
                        // following `vmulps` is unnecessary.
                        asm.vaddps(ZMM[e.idx()], ZMM[e.idx()], ZMM[one.idx()])?;
                        asm.vdivps(ZMM[e.idx()], ZMM[g.idx()], ZMM[e.idx()])?;
                        if !gate_out {
                            asm.vmulps(ZMM[e.idx()], ZMM[e.idx()], ZMM[u.idx()])?;
                        }
                        asm.vcvtneps2bf16(YMM[out[half].y_evex()], ZMM[e.idx()])?;
                    }
                    asm.vinserti64x4(
                        ZMM[out[0].idx()],
                        ZMM[out[0].idx()],
                        YMM[out[1].y_evex()],
                        1,
                    )?;
                    store_block(asm, zmmword_ptr(py + o), out[0], last)?;
                    Ok(())
                };
                if tail > 0 {
                    set_tail_mask(&mut asm, tail)?;
                }
                match unroll {
                    None => {
                        for c in 0..nbt {
                            block(&mut asm, rdi, rsi, rdx, (c * 64) as i32, c == nb)?;
                        }
                    }
                    Some(unroll) => {
                        // `rcx`/`r8`/`r9` walk gate/up/y, `r11d` counts the
                        // turns. Nothing else in this arm touches them.
                        asm.mov(rcx, rdi)?;
                        asm.mov(r8, rsi)?;
                        asm.mov(r9, rdx)?;
                        asm.mov(r11d, (nb / unroll) as u32)?;
                        let mut top = asm.create_label();
                        asm.set_label(&mut top)?;
                        for j in 0..unroll {
                            block(&mut asm, rcx, r8, r9, (j * 64) as i32, false)?;
                        }
                        let step = (unroll * 64) as i32;
                        asm.add(rcx, step)?;
                        asm.add(r8, step)?;
                        asm.add(r9, step)?;
                        asm.dec(r11d)?;
                        asm.jnz(top)?;
                        // The cursors past the whole blocks: the partial one.
                        if tail > 0 {
                            block(&mut asm, rcx, r8, r9, 0, true)?;
                        }
                    }
                }
            }

            // ── RoPE, convention rotate_half ──────────────────────────────
            //
            // rdi = x (read and written), rsi = cos, rdx = sin.
            // For i < dim/2:  x[i]      ← x[i]·c − x[i+h]·s
            //                   x[i+h]    ← x[i]·s + x[i+h]·c
            // The two halves are `h·2` bytes apart.
            ElemOp::Rope => {
                let half = spec.dim / 2;
                let hoff = (half * 2) as i32;
                // Both outputs are written by a `vmovups ymm` emitted in
                // **VEX**, which stops at `ymm15`: low zone. The EVEX form
                // exists and would go up to `ymm31` (cf. `regs.rs`), for 2
                // more bytes per store — useless here. Only constraint
                // of this kernel, and it is now in the type.
                let outs = f.low_run(2, "rope:output halves (vmovups VEX)")?;
                let a = f.any("rope:x[j]")?;
                let b = f.any("rope:x[j+h]")?;
                let co = f.any("rope:cos")?;
                let sn = f.any("rope:sin")?;
                let (ra, rb) = (outs[0], outs[1]);
                // 16 values per pass: the tables have `half` entries. When `dim` is a multiple of
                // 32, `half` is a multiple of 16; otherwise (heads of 72 rotated
                // per axis: 36, so 18 pairs) the last pass holds the `half % 16`
                // pairs left, read and written under `k2` (their lanes in a dword or a word mask
                // alike), and touches nothing past them.
                let rest = half % 16;
                if rest > 0 {
                    set_tail_mask(&mut asm, rest)?;
                }
                for j in (0..half).step_by(16) {
                    let o = (j * 2) as i32;
                    let last = half - j < 16;
                    for (reg, base, off) in
                        [(a, rdi, o), (b, rdi, o + hoff), (co, rsi, o), (sn, rdx, o)]
                    {
                        if last {
                            asm.vpmovzxwd(ZMM[reg.idx()].k2().z(), ymmword_ptr(base + off))?;
                        } else {
                            asm.vpmovzxwd(ZMM[reg.idx()], ymmword_ptr(base + off))?;
                        }
                        asm.vpslld(ZMM[reg.idx()], ZMM[reg.idx()], 16u32)?;
                    }
                    // a' = a·c − b·s
                    asm.vmulps(ZMM[ra.idx()], ZMM[a.idx()], ZMM[co.idx()])?;
                    asm.vfnmadd231ps(ZMM[ra.idx()], ZMM[b.idx()], ZMM[sn.idx()])?;
                    // b' = a·s + b·c
                    asm.vmulps(ZMM[rb.idx()], ZMM[a.idx()], ZMM[sn.idx()])?;
                    asm.vfmadd231ps(ZMM[rb.idx()], ZMM[b.idx()], ZMM[co.idx()])?;
                    asm.vcvtneps2bf16(YMM[ra.y()], ZMM[ra.idx()])?;
                    asm.vcvtneps2bf16(YMM[rb.y()], ZMM[rb.idx()])?;
                    if last {
                        asm.vmovdqu16(ymmword_ptr(rdi + o).k2(), YMM[ra.y()])?;
                        asm.vmovdqu16(ymmword_ptr(rdi + (o + hoff)).k2(), YMM[rb.y()])?;
                    } else {
                        asm.vmovups(ymmword_ptr(rdi + o), YMM[ra.y()])?;
                        asm.vmovups(ymmword_ptr(rdi + (o + hoff)), YMM[rb.y()])?;
                    }
                }
            }

            // ── a += b ────────────────────────────────────────────────────
            //
            // rdi = a (read and written), rsi = b.
            ElemOp::ResidualAdd => {
                let ab = f.any("add:block of a")?;
                let bb = f.any("add:block")?;
                let aw = f.any("add: extended")?;
                let bw = f.any("add:b extended")?;
                let out = f.run(2, "add:output halves")?;
                for c in 0..nb {
                    asm.vmovdqu16(ZMM[ab.idx()], zmmword_ptr(rdi + (c * 64) as i32))?;
                    asm.vmovdqu16(ZMM[bb.idx()], zmmword_ptr(rsi + (c * 64) as i32))?;
                    for half in 0..2 {
                        widen_half(&mut asm, aw, ab, half)?;
                        widen_half(&mut asm, bw, bb, half)?;
                        asm.vaddps(ZMM[aw.idx()], ZMM[aw.idx()], ZMM[bw.idx()])?;
                        asm.vcvtneps2bf16(YMM[out[half].y_evex()], ZMM[aw.idx()])?;
                    }
                    asm.vinserti64x4(
                        ZMM[out[0].idx()],
                        ZMM[out[0].idx()],
                        YMM[out[1].y_evex()],
                        1,
                    )?;
                    asm.vmovdqu16(zmmword_ptr(rdi + (c * 64) as i32), ZMM[out[0].idx()])?;
                }
            }

            // ── y = x · s ─────────────────────────────────────────────────
            //
            // rdi = x, rsi = y, xmm0 = s.
            ElemOp::Scale => {
                let sc = f.any("scale:s broadcasted")?;
                let xb = f.any("scale:block of x")?;
                let xw = f.any("scale:x expanded")?;
                let out = f.run(2, "scale:output halves")?;
                // **First emitted instruction**, before any vector write:
                // `f.run` serves the low zone, so output halves may land on
                // `zmm0`, whose view is `xmm0`. The same trap as the `eps` of
                // RMSNorm, which it fixes by saving into a full register; here
                // the emission order suffices — this kernel has no horizontal
                // reduction.
                asm.vbroadcastss(ZMM[sc.idx()], XMM[0])?;
                for c in 0..nb {
                    asm.vmovdqu16(ZMM[xb.idx()], zmmword_ptr(rdi + (c * 64) as i32))?;
                    for half in 0..2 {
                        widen_half(&mut asm, xw, xb, half)?;
                        asm.vmulps(ZMM[xw.idx()], ZMM[xw.idx()], ZMM[sc.idx()])?;
                        asm.vcvtneps2bf16(YMM[out[half].y_evex()], ZMM[xw.idx()])?;
                    }
                    asm.vinserti64x4(
                        ZMM[out[0].idx()],
                        ZMM[out[0].idx()],
                        YMM[out[1].y_evex()],
                        1,
                    )?;
                    asm.vmovdqu16(zmmword_ptr(rsi + (c * 64) as i32), ZMM[out[0].idx()])?;
                }
            }
            // ── y = cap · tanh(x / cap)  (softcapping of logits, Gemma) ────
            //
            // rdi = x, rsi = y, xmm0 = cap. Same ABI as `Scale`.
            //
            // `tanh(u) = (1 − e^{−2u})/(1 + e^{−2u})` evaluated on `|u|`: this
            // bounds the argument of `exp` to `≤ 0` (the safe domain, see
            // `EXP_MAX88`) and makes saturation exact — `|x| ≫ cap` gives
            // `e ≈ 0`, so `±cap`. The sign is reinjected by `vpord`.
            ElemOp::Softcap => {
                let capv = f.high("softcap: broadcast cap")?;
                let neg2 = f.high("softcap:−2")?;
                let inv = f.high("softcap:−2/cap")?;
                let absm = f.high("softcap:mask |·|")?;
                let sgnm = f.high("softcap:sign mask")?;
                let k = ExpConsts::alloc(&mut f)?;
                let xb = f.any("softcap: x block")?;
                let a = f.any("softcap:argument")?;
                let sg = f.any("softcap:sign")?;
                let t1 = f.any("softcap:temp exp 1")?;
                let t2 = f.any("softcap:temp exp 2")?;
                let t3 = f.any("softcap:temp exp 3")?;
                let out = f.run(2, "softcap: output halves")?;
                // **First emitted instruction**: `f.run` serves the lower zone,
                // so half of the output may fall on `zmm0`,
                // of which `xmm0` — which holds `cap` — is the view. Same trap as
                // the `Scale` arm, same defense: read `xmm0` before anything.
                asm.vbroadcastss(ZMM[capv.idx()], XMM[0])?;
                bcast_imm(&mut asm, neg2, (-2.0f32).to_bits())?;
                // `−2/cap`, a division at kernel setup, never in the
                // loop. No `vrcp14ps`: its 14 bits would skew the
                // slope near 0, where `tanh(u) ≈ u`.
                asm.vdivps(ZMM[inv.idx()], ZMM[neg2.idx()], ZMM[capv.idx()])?;
                bcast_imm(&mut asm, absm, 0x7FFF_FFFF)?;
                bcast_imm(&mut asm, sgnm, 0x8000_0000)?;
                load_exp_constants(&mut asm, &k)?;
                for c in 0..nb {
                    asm.vmovdqu16(ZMM[xb.idx()], zmmword_ptr(rdi + (c * 64) as i32))?;
                    for half in 0..2 {
                        widen_half(&mut asm, a, xb, half)?;
                        asm.vpandd(ZMM[sg.idx()], ZMM[a.idx()], ZMM[sgnm.idx()])?;
                        asm.vpandd(ZMM[a.idx()], ZMM[a.idx()], ZMM[absm.idx()])?;
                        // a ← −2·|x|/cap ≤ 0
                        asm.vmulps(ZMM[a.idx()], ZMM[a.idx()], ZMM[inv.idx()])?;
                        exp_inline(&mut asm, &k, a, t1, t2, t3)?;
                        // `k.one` holds 0x3F80_0000 : this is 1.0f, the same
                        // pattern that `exp` adds as an integer to the bias.
                        asm.vsubps(ZMM[t1.idx()], ZMM[k.one.idx()], ZMM[a.idx()])?;
                        asm.vaddps(ZMM[t2.idx()], ZMM[k.one.idx()], ZMM[a.idx()])?;
                        asm.vdivps(ZMM[a.idx()], ZMM[t1.idx()], ZMM[t2.idx()])?;
                        asm.vmulps(ZMM[a.idx()], ZMM[a.idx()], ZMM[capv.idx()])?;
                        asm.vpord(ZMM[a.idx()], ZMM[a.idx()], ZMM[sg.idx()])?;
                        asm.vcvtneps2bf16(YMM[out[half].y_evex()], ZMM[a.idx()])?;
                    }
                    asm.vinserti64x4(
                        ZMM[out[0].idx()],
                        ZMM[out[0].idx()],
                        YMM[out[1].y_evex()],
                        1,
                    )?;
                    asm.vmovdqu16(zmmword_ptr(rsi + (c * 64) as i32), ZMM[out[0].idx()])?;
                }
            }
            // ── y = bf16(Σᵣ x[r] · w[r]) ──────────────────────────────────
            //
            // rdi = x (`taps` rows of `dim`), rsi = w (`[taps][dim]`), rdx = y.
            ElemOp::DepthwiseRows(taps) => {
                let xb = f.any("dw:block of a row")?;
                let wb = f.any("dw:block of the kernel")?;
                let xw = f.any("dw:row widened")?;
                let ww = f.any("dw:kernel widened")?;
                let acc = f.run(2, "dw:sums of the halves")?;
                let out = f.run(2, "dw:output halves")?;
                let row = (spec.dim * 2) as i32;
                for c in 0..nb {
                    for &a in &acc {
                        asm.vpxord(ZMM[a.idx()], ZMM[a.idx()], ZMM[a.idx()])?;
                    }
                    for r in 0..taps as i32 {
                        asm.vmovdqu16(
                            ZMM[xb.idx()],
                            zmmword_ptr(rdi + (r * row + (c * 64) as i32)),
                        )?;
                        asm.vmovdqu16(
                            ZMM[wb.idx()],
                            zmmword_ptr(rsi + (r * row + (c * 64) as i32)),
                        )?;
                        for (half, &a) in acc.iter().enumerate() {
                            widen_half(&mut asm, xw, xb, half)?;
                            widen_half(&mut asm, ww, wb, half)?;
                            asm.vfmadd231ps(ZMM[a.idx()], ZMM[xw.idx()], ZMM[ww.idx()])?;
                        }
                    }
                    for (half, &a) in acc.iter().enumerate() {
                        asm.vcvtneps2bf16(YMM[out[half].y_evex()], ZMM[a.idx()])?;
                    }
                    asm.vinserti64x4(
                        ZMM[out[0].idx()],
                        ZMM[out[0].idx()],
                        YMM[out[1].y_evex()],
                        1,
                    )?;
                    asm.vmovdqu16(zmmword_ptr(rdx + (c * 64) as i32), ZMM[out[0].idx()])?;
                }
            }
            // Its own emitter (`emit_local_attn.rs`), the same on both x86 arms.
            ElemOp::LocalAttention { .. } => {
                return Err(EmitErr::Alloc(Decline::Empty));
            }
            // ── y = (x − μ) · rsqrt(σ² + ε) · g + b ───────────────────────
            //
            // rdi = x, rsi = w (`g` then `b`, `dim` each), rdx = y, xmm0 = eps.
            // Three passes over the row: Σx, then Σ(x − μ)² (the variance about
            // the mean, as torch's LayerNorm), then the output.
            ElemOp::LayerNorm => {
                asm.vmovd(r10d, XMM[0])?;
                let n_acc = 4.min(2 * nb);
                let acc = f.low_run(n_acc, "ln:chains of the sums")?;
                let ld = f.low("ln:block read / temporary scalar")?;
                let xw = f.any("ln:x widened")?;
                let mu = f.any("ln:μ broadcast")?;
                let inv = f.any("ln:1/σ broadcast")?;
                let scalar =
                    |asm: &mut CodeAssembler, mul: f32| -> std::result::Result<(), IcedError> {
                        let a0 = acc[0];
                        for a in &acc[1..] {
                            asm.vaddps(ZMM[a0.idx()], ZMM[a0.idx()], ZMM[a.idx()])?;
                        }
                        hsum(asm, a0, ld)?;
                        asm.mov(eax, mul.to_bits())?;
                        asm.vmovd(XMM[ld.x()], eax)?;
                        asm.vmulss(XMM[a0.x()], XMM[a0.x()], XMM[ld.x()])?;
                        Ok(())
                    };
                // Pass 1: μ = Σx / dim.
                for a in &acc {
                    asm.vxorps(ZMM[a.idx()], ZMM[a.idx()], ZMM[a.idx()])?;
                }
                for c in 0..nb {
                    asm.vmovdqu16(ZMM[ld.idx()], zmmword_ptr(rdi + (c * 64) as i32))?;
                    for half in 0..2 {
                        widen_half(&mut asm, xw, ld.into(), half)?;
                        let a = acc[(2 * c + half) % n_acc];
                        asm.vaddps(ZMM[a.idx()], ZMM[a.idx()], ZMM[xw.idx()])?;
                    }
                }
                scalar(&mut asm, 1.0 / spec.dim as f32)?;
                asm.vbroadcastss(ZMM[mu.idx()], XMM[acc[0].x()])?;
                // Pass 2: σ² = Σ(x − μ)² / dim, then 1/√(σ² + ε).
                for a in &acc {
                    asm.vxorps(ZMM[a.idx()], ZMM[a.idx()], ZMM[a.idx()])?;
                }
                for c in 0..nb {
                    asm.vmovdqu16(ZMM[ld.idx()], zmmword_ptr(rdi + (c * 64) as i32))?;
                    for half in 0..2 {
                        widen_half(&mut asm, xw, ld.into(), half)?;
                        asm.vsubps(ZMM[xw.idx()], ZMM[xw.idx()], ZMM[mu.idx()])?;
                        let a = acc[(2 * c + half) % n_acc];
                        asm.vfmadd231ps(ZMM[a.idx()], ZMM[xw.idx()], ZMM[xw.idx()])?;
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
                asm.vbroadcastss(ZMM[inv.idx()], XMM[a0.x()])?;
                for a in acc {
                    f.free_low(a);
                }
                f.free_low(ld);
                // Pass 3: y = (x − μ) · inv · g + b.
                let xb = f.any("ln:block of x")?;
                let gb = f.any("ln:block of g")?;
                let bb = f.any("ln:block of b")?;
                let gw = f.any("ln:g widened")?;
                let bw = f.any("ln:b widened")?;
                let out = f.run(2, "ln:output halves")?;
                let boff = (spec.dim * 2) as i32;
                for c in 0..nb {
                    let o = (c * 64) as i32;
                    asm.vmovdqu16(ZMM[xb.idx()], zmmword_ptr(rdi + o))?;
                    asm.vmovdqu16(ZMM[gb.idx()], zmmword_ptr(rsi + o))?;
                    asm.vmovdqu16(ZMM[bb.idx()], zmmword_ptr(rsi + (boff + o)))?;
                    for half in 0..2 {
                        widen_half(&mut asm, xw, xb, half)?;
                        widen_half(&mut asm, gw, gb, half)?;
                        widen_half(&mut asm, bw, bb, half)?;
                        asm.vsubps(ZMM[xw.idx()], ZMM[xw.idx()], ZMM[mu.idx()])?;
                        asm.vmulps(ZMM[xw.idx()], ZMM[xw.idx()], ZMM[inv.idx()])?;
                        asm.vfmadd213ps(ZMM[xw.idx()], ZMM[gw.idx()], ZMM[bw.idx()])?;
                        asm.vcvtneps2bf16(YMM[out[half].y_evex()], ZMM[xw.idx()])?;
                    }
                    asm.vinserti64x4(
                        ZMM[out[0].idx()],
                        ZMM[out[0].idx()],
                        YMM[out[1].y_evex()],
                        1,
                    )?;
                    asm.vmovdqu16(zmmword_ptr(rdx + o), ZMM[out[0].idx()])?;
                }
            }

            // ── y = gelu(gate) · up, the exact GeLU ───────────────────────
            //
            // rdi = gate, rsi = up, rdx = y. `gelu(g) = g · Φ(g)`, `Φ(g) =
            // ½·erfc(−g/√2)`, the complementary error function of Numerical
            // Recipes (`erfcc`, fractional error below 1.2e-7 everywhere):
            // `a = |g|/√2`, `t = 1/(1 + a/2)`, `E = t·exp(−a² + P(t))` with `P`
            // of degree 9; `Φ = ½·E` where `g < 0`, `1 − ½·E` elsewhere. The exponential's
            // argument is never positive (`P(t) ≤ P(1) ≈ 0`). The ten
            // coefficients of `P` live in a table under `rsp`, read as broadcast
            // operands.
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
                let k = ExpConsts::alloc(&mut f)?;
                let one = f.high("gelu:1.0")?;
                let half_c = f.high("gelu:½")?;
                let isq = f.high("gelu:1/√2")?;
                let absm = f.high("gelu:mask |·|")?;
                let gb = f.any("gelu:gate block")?;
                let ub = f.any("gelu:up block")?;
                let g = f.any("gelu:gate widened")?;
                let u = f.any("gelu:up widened")?;
                let a = f.any("gelu:a = |g|/√2")?;
                let t = f.any("gelu:t")?;
                let p = f.any("gelu:P(t), then the result")?;
                let sg = f.any("gelu:1 − ½·E")?;
                let te = f.run(3, "gelu:exp temporaries")?;
                let out = f.run(2, "gelu:output halves")?;
                load_exp_constants(&mut asm, &k)?;
                bcast_imm(&mut asm, one, 1.0f32.to_bits())?;
                bcast_imm(&mut asm, half_c, 0.5f32.to_bits())?;
                bcast_imm(&mut asm, isq, std::f32::consts::FRAC_1_SQRT_2.to_bits())?;
                bcast_imm(&mut asm, absm, 0x7FFF_FFFF)?;
                asm.sub(rsp, 64)?;
                for (i, c) in P.iter().enumerate() {
                    asm.mov(dword_ptr(rsp + (4 * i) as i32), c.to_bits())?;
                }
                let block = |asm: &mut CodeAssembler,
                             pg: AsmRegister64,
                             pu: AsmRegister64,
                             py: AsmRegister64,
                             o: i32|
                 -> std::result::Result<(), IcedError> {
                    asm.vmovdqu16(ZMM[gb.idx()], zmmword_ptr(pg + o))?;
                    asm.vmovdqu16(ZMM[ub.idx()], zmmword_ptr(pu + o))?;
                    for half in 0..2 {
                        widen_half(asm, g, gb, half)?;
                        widen_half(asm, u, ub, half)?;
                        asm.vpandd(ZMM[a.idx()], ZMM[g.idx()], ZMM[absm.idx()])?;
                        asm.vmulps(ZMM[a.idx()], ZMM[a.idx()], ZMM[isq.idx()])?;
                        // t = 1 / (a·½ + 1)
                        asm.vmovaps(ZMM[t.idx()], ZMM[a.idx()])?;
                        asm.vfmadd213ps(ZMM[t.idx()], ZMM[half_c.idx()], ZMM[one.idx()])?;
                        asm.vdivps(ZMM[t.idx()], ZMM[one.idx()], ZMM[t.idx()])?;
                        // P(t) by Horner from the highest coefficient, then − a².
                        asm.vbroadcastss(ZMM[p.idx()], dword_ptr(rsp + 36))?;
                        for i in (0..9).rev() {
                            asm.vfmadd213ps(ZMM[p.idx()], ZMM[t.idx()], dword_bcst(rsp + (4 * i)))?;
                        }
                        asm.vfnmadd231ps(ZMM[p.idx()], ZMM[a.idx()], ZMM[a.idx()])?;
                        exp_inline(asm, &k, p, te[0], te[1], te[2])?;
                        // E = t·exp(…) = erfc(a); Φ = ½·E where g < 0 (taken as it is: `½ − ½·(1 −
                        // E)` would lose it to cancellation), 1 − ½·E
                        // elsewhere, chosen by the sign bit of g.
                        asm.vmulps(ZMM[p.idx()], ZMM[p.idx()], ZMM[t.idx()])?;
                        asm.vmulps(ZMM[p.idx()], ZMM[p.idx()], ZMM[half_c.idx()])?;
                        asm.vsubps(ZMM[sg.idx()], ZMM[one.idx()], ZMM[p.idx()])?;
                        asm.vpmovd2m(k1, ZMM[g.idx()])?;
                        asm.vblendmps(ZMM[p.idx()].k1(), ZMM[sg.idx()], ZMM[p.idx()])?;
                        asm.vmulps(ZMM[p.idx()], ZMM[p.idx()], ZMM[g.idx()])?;
                        asm.vmulps(ZMM[p.idx()], ZMM[p.idx()], ZMM[u.idx()])?;
                        asm.vcvtneps2bf16(YMM[out[half].y_evex()], ZMM[p.idx()])?;
                    }
                    asm.vinserti64x4(
                        ZMM[out[0].idx()],
                        ZMM[out[0].idx()],
                        YMM[out[1].y_evex()],
                        1,
                    )?;
                    asm.vmovdqu16(zmmword_ptr(py + o), ZMM[out[0].idx()])?;
                    Ok(())
                };
                match unroll {
                    None => {
                        for c in 0..nb {
                            block(&mut asm, rdi, rsi, rdx, (c * 64) as i32)?;
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
                            block(&mut asm, rcx, r8, r9, (j * 64) as i32)?;
                        }
                        let step = (unroll * 64) as i32;
                        asm.add(rcx, step)?;
                        asm.add(r8, step)?;
                        asm.add(r9, step)?;
                        asm.dec(r11d)?;
                        asm.jnz(top)?;
                    }
                }
                asm.add(rsp, 64)?;
            }

            // ── y = bf16((x · r − b) · c) ─────────────────────────────────
            //
            // rdi = x, rsi = w (`b` then `c`), rdx = y, xmm0 = r. Three f32 operations, no fusion,
            // one rounding.
            ElemOp::Standardize => {
                let r = f.high("std:r")?;
                asm.vbroadcastss(ZMM[r.idx()], XMM[0])?;
                if tail > 0 {
                    set_tail_mask(&mut asm, tail)?;
                }
                let xb = f.any("std:block of x")?;
                let bb = f.any("std:block of b")?;
                let cb = f.any("std:block of c")?;
                let xw = f.any("std:x widened")?;
                let tw = f.any("std:b or c widened")?;
                let out = f.run(2, "std:output halves")?;
                let coff = (spec.dim * 2) as i32;
                for c in 0..nbt {
                    let o = (c * 64) as i32;
                    load_block(&mut asm, xb, zmmword_ptr(rdi + o), c == nb)?;
                    load_block(&mut asm, bb, zmmword_ptr(rsi + o), c == nb)?;
                    load_block(&mut asm, cb, zmmword_ptr(rsi + (coff + o)), c == nb)?;
                    for half in 0..2 {
                        widen_half(&mut asm, xw, xb, half)?;
                        asm.vmulps(ZMM[xw.idx()], ZMM[xw.idx()], ZMM[r.idx()])?;
                        widen_half(&mut asm, tw, bb, half)?;
                        asm.vsubps(ZMM[xw.idx()], ZMM[xw.idx()], ZMM[tw.idx()])?;
                        widen_half(&mut asm, tw, cb, half)?;
                        asm.vmulps(ZMM[xw.idx()], ZMM[xw.idx()], ZMM[tw.idx()])?;
                        asm.vcvtneps2bf16(YMM[out[half].y_evex()], ZMM[xw.idx()])?;
                    }
                    asm.vinserti64x4(
                        ZMM[out[0].idx()],
                        ZMM[out[0].idx()],
                        YMM[out[1].y_evex()],
                        1,
                    )?;
                    store_block(&mut asm, zmmword_ptr(rdx + o), out[0], c == nb)?;
                }
            }

            // ── y = min(max(x, lo), hi) ───────────────────────────────────
            //
            // rdi = x, rsi = y, xmm0 = lo, xmm1 = hi.
            ElemOp::Clamp => {
                // The bounds in the high zone, broadcast before any vector write: `xmm0` and `xmm1`
                // are the views of `zmm0` and `zmm1`, which the low zone serves.
                let lo = f.high("clamp:lo")?;
                let hi = f.high("clamp:hi")?;
                asm.vbroadcastss(ZMM[lo.idx()], XMM[0])?;
                asm.vbroadcastss(ZMM[hi.idx()], XMM[1])?;
                let xb = f.any("clamp:block of x")?;
                let xw = f.any("clamp:x widened")?;
                let out = f.run(2, "clamp:output halves")?;
                for c in 0..nb {
                    asm.vmovdqu16(ZMM[xb.idx()], zmmword_ptr(rdi + (c * 64) as i32))?;
                    for half in 0..2 {
                        widen_half(&mut asm, xw, xb, half)?;
                        asm.vmaxps(ZMM[xw.idx()], ZMM[xw.idx()], ZMM[lo.idx()])?;
                        asm.vminps(ZMM[xw.idx()], ZMM[xw.idx()], ZMM[hi.idx()])?;
                        asm.vcvtneps2bf16(YMM[out[half].y_evex()], ZMM[xw.idx()])?;
                    }
                    asm.vinserti64x4(
                        ZMM[out[0].idx()],
                        ZMM[out[0].idx()],
                        YMM[out[1].y_evex()],
                        1,
                    )?;
                    asm.vmovdqu16(zmmword_ptr(rsi + (c * 64) as i32), ZMM[out[0].idx()])?;
                }
            }

            // ── y = bf16(Σᵣ x[r] · inv) · s ───────────────────────────────
            //
            // rdi = x (`rows` rows of `dim`), rsi = y, xmm0 = s, xmm1 = inv.
            ElemOp::PoolRows(rows) => {
                let sc = f.high("pool:s")?;
                let iv = f.high("pool:inv")?;
                asm.vbroadcastss(ZMM[sc.idx()], XMM[0])?;
                asm.vbroadcastss(ZMM[iv.idx()], XMM[1])?;
                let xb = f.any("pool:block of a row")?;
                let xw = f.any("pool:row widened")?;
                let acc = f.run(2, "pool:sums of the halves")?;
                let out = f.run(2, "pool:output halves")?;
                let row = (spec.dim * 2) as i32;
                for c in 0..nb {
                    for &a in &acc {
                        asm.vpxord(ZMM[a.idx()], ZMM[a.idx()], ZMM[a.idx()])?;
                    }
                    for r in 0..rows as i32 {
                        asm.vmovdqu16(
                            ZMM[xb.idx()],
                            zmmword_ptr(rdi + (r * row + (c * 64) as i32)),
                        )?;
                        for (half, &a) in acc.iter().enumerate() {
                            widen_half(&mut asm, xw, xb, half)?;
                            asm.vfmadd231ps(ZMM[a.idx()], ZMM[xw.idx()], ZMM[iv.idx()])?;
                        }
                    }
                    for (half, &a) in acc.iter().enumerate() {
                        // The mean rounded to bf16, widened back, times `s`, rounded again.
                        asm.vcvtneps2bf16(YMM[out[half].y_evex()], ZMM[a.idx()])?;
                        asm.vpmovzxwd(ZMM[a.idx()], YMM[out[half].y_evex()])?;
                        asm.vpslld(ZMM[a.idx()], ZMM[a.idx()], 16u32)?;
                        asm.vmulps(ZMM[a.idx()], ZMM[a.idx()], ZMM[sc.idx()])?;
                        asm.vcvtneps2bf16(YMM[out[half].y_evex()], ZMM[a.idx()])?;
                    }
                    asm.vinserti64x4(
                        ZMM[out[0].idx()],
                        ZMM[out[0].idx()],
                        YMM[out[1].y_evex()],
                        1,
                    )?;
                    asm.vmovdqu16(zmmword_ptr(rsi + (c * 64) as i32), ZMM[out[0].idx()])?;
                }
            }
        }
        asm.vzeroupper()?;
        asm.ret()?;
        Ok(asm.assemble(0)?)
    }
}
