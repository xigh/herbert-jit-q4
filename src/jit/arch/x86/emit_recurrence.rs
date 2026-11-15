//! The GatedDeltaNet recurrence, AVX-512: one V head, one position.
//!
//! `s` : `[hkd · hvd]` f32, persistent, `s[kd·hvd + vd]`. The order is the
//! specification:
//!
//! ```text
//! 1. s ← decay·s ;  kv[vd] = Σ_kd s[kd·hvd+vd] · k[kd]
//!    δ[vd] = (v[vd] − kv[vd]) · beta
//! 2. s[kd·hvd+vd] += k[kd]·δ[vd]
//!    out[vd] = Σ_kd s[kd·hvd+vd] · q[kd] · scale
//! ```
//!
//! ## The ABI
//!
//! `fn(s: *mut f32, out: *mut f32, q: *const f32, k: *const f32, v: *const f32,
//! hkd: usize, beta: f32, decay: f32, scale: f32)` — sysv64: `rdi, rsi, rdx,
//! rcx, r8, r9` for the integers, `xmm0, xmm1, xmm2` for the floats.
//!
//! `hkd` is a parameter, `hvd` an immediate: `hvd` fixes the number of zmm
//! per accumulator (`nb = hvd/16`), hence the register allocation; `hkd` is
//! only the bound of the two loops.
//!
//! ## The registers
//!
//! `beta`, `decay`, `scale` broadcast (high zone, out of reach of the
//! accumulators), `kval`, `qval`, `rv` (the state row), `t` (the product),
//! then `nb` accumulators `kv` — reused for `δ` after pass 1 — and `nb`
//! accumulators `out`. `7 + 2·nb ≤ 32` ⇒ `hvd ≤ 192`; at `hvd = 128`, 23
//! registers and no spill.
//!
//! ## Two forms: without FMA (the scalar bits) and with
//!
//! Without FMA: `vmulps` then `vaddps`, like the scalar reference
//! (Rust does not contract); each `out[vd]` sums over increasing `kd` —
//! same operations, same order, same roundings, the kernel returns the bits
//! of the scalar. With FMA
//! (`Knobs::deltanet_fma`): `vfmadd231ps` for the three dot products, 4
//! instructions per block and per `kd` instead of 5 then 6; it returns the
//! bits of a scalar written with `f32::mul_add` at the same places, and
//! differs from the other form at the last bit. Measured at zero gain: the
//! FMA only removes additions that were not on the critical resource.
//!
//! With `RecurrenceSpec::state16` the state is kept in bf16: widened as it is
//! loaded, not stored by pass 1 (pass 2 recomputes the decay, the same
//! product), rounded to nearest even as pass 2 stores it (`vcvtneps2bf16`, as
//! the activations), after `out` has read it.

pub(crate) use crate::jit::emit_recurrence::RecurrenceSpec;
pub use x86::emit_recurrence;

mod x86 {
    use super::RecurrenceSpec;
    use crate::core::error::{HerbertError, Result};
    use crate::jit::ir::Decline;
    use crate::jit::regs::RegFile;
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

    /// Emits the kernel of `spec`. The shape must have passed
    /// [`RecurrenceSpec::check`] — here an allocation refusal is a failure.
    pub fn emit_recurrence(spec: &RecurrenceSpec) -> Result<Vec<u8>> {
        assemble(spec).map_err(|e| {
            HerbertError::Backend(format!(
                "backend_jit: recurrence emission hvd={} fma={} state16={} mamba={}: {e}",
                spec.hvd, spec.fma, spec.state16, spec.mamba
            ))
        })
    }

    /// The two ways an emission can fail — see `emit_elem`.
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

    // rdi = s, rsi = out, rdx = q, rcx = k, r8 = v, r9 = hkd ;
    // xmm0 = beta, xmm1 = decay, xmm2 = scale.
    //
    // Working integer registers, all volatile in sysv64: r10 the counter
    // of `kd`, r11 the cursor in `s`, rax the cursor in `k`; in pass 2,
    // `rdx` (q) advances in place — we no longer read it.
    fn assemble(spec: &RecurrenceSpec) -> std::result::Result<Vec<u8>, EmitErr> {
        let mut asm = CodeAssembler::new(64)?;
        let nb = spec.blocks16();
        let mut f = RegFile::new();
        let beta = f.high("rec:beta broadcast")?;
        let decay = f.high("rec:decay broadcast")?;
        let scale = f.high("rec:scale broadcast")?;
        let kval = f.any("rec:k[kd] broadcast")?;
        let qval = f.any("rec:q[kd]·scale broadcast")?;
        let rv = f.any("rec:state row")?;
        let t = f.any("rec:product temporary")?;
        let kv = f.run(nb, "rec:kv accumulator, then δ")?;
        let out = f.run(nb, "rec:accumulator out")?;

        // **First emitted instructions.** `f.run` serves the low area,
        // so the accumulators fall on `zmm0..`, where `xmm0-2` — which
        // carry `beta`, `decay`, `scale` — are the views. Broadcast them to
        // the high area before any vector write, like `Scale` in
        // `emit_elem`.
        asm.vbroadcastss(ZMM[beta.idx()], XMM[0])?;
        asm.vbroadcastss(ZMM[decay.idx()], XMM[1])?;
        asm.vbroadcastss(ZMM[scale.idx()], XMM[2])?;

        // Bytes of one value of the state: bf16 (`state16`) or f32.
        let elem = if spec.state16 { 2 } else { 4 };
        let line = (spec.hvd * elem) as i32;
        // A row block of sixteen values into `rv`, as f32 (a bf16 is the high half of its f32).
        let load = |asm: &mut CodeAssembler, off: i32| -> std::result::Result<(), IcedError> {
            if spec.state16 {
                asm.vpmovzxwd(ZMM[rv.idx()], ymmword_ptr(r11 + off))?;
                asm.vpslld(ZMM[rv.idx()], ZMM[rv.idx()], 16)
            } else {
                asm.vmovups(ZMM[rv.idx()], zmmword_ptr(r11 + off))
            }
        };

        if spec.mamba {
            // ── Mamba2, one pass ───────────────────────────────────────────
            //   s[kd][j] ← s[kd][j]·decay + (k[kd]·beta)·v[j] ;  out[j] += s[kd][j]·q[kd]
            // then out[j] += v[j]·scale (the skip D). `kv` holds v.
            for (j, a) in kv.iter().enumerate() {
                asm.vmovups(ZMM[a.idx()], zmmword_ptr(r8 + (j * 64) as i32))?;
            }
            for a in &out {
                asm.vxorps(ZMM[a.idx()], ZMM[a.idx()], ZMM[a.idx()])?;
            }
            asm.mov(r10, r9)?;
            asm.mov(r11, rdi)?;
            asm.mov(rax, rcx)?;
            let mut pass = asm.create_label();
            asm.set_label(&mut pass)?;
            asm.vbroadcastss(ZMM[kval.idx()], dword_ptr(rax))?;
            // dB = dt·B once per `kd`, as the reference forms it before multiplying by x.
            asm.vmulps(ZMM[kval.idx()], ZMM[kval.idx()], ZMM[beta.idx()])?;
            asm.vbroadcastss(ZMM[qval.idx()], dword_ptr(rdx))?;
            for j in 0..nb {
                let off = (j * 16 * elem) as i32;
                load(&mut asm, off)?;
                asm.vmulps(ZMM[rv.idx()], ZMM[rv.idx()], ZMM[decay.idx()])?;
                if spec.fma {
                    asm.vfmadd231ps(ZMM[rv.idx()], ZMM[kval.idx()], ZMM[kv[j].idx()])?;
                    asm.vfmadd231ps(ZMM[out[j].idx()], ZMM[rv.idx()], ZMM[qval.idx()])?;
                } else {
                    asm.vmulps(ZMM[t.idx()], ZMM[kval.idx()], ZMM[kv[j].idx()])?;
                    asm.vaddps(ZMM[rv.idx()], ZMM[rv.idx()], ZMM[t.idx()])?;
                    asm.vmulps(ZMM[t.idx()], ZMM[rv.idx()], ZMM[qval.idx()])?;
                    asm.vaddps(ZMM[out[j].idx()], ZMM[out[j].idx()], ZMM[t.idx()])?;
                }
                if spec.state16 {
                    asm.vcvtneps2bf16(YMM[t.idx()], ZMM[rv.idx()])?;
                    asm.vmovdqu16(ymmword_ptr(r11 + off), YMM[t.idx()])?;
                } else {
                    asm.vmovups(zmmword_ptr(r11 + off), ZMM[rv.idx()])?;
                }
            }
            asm.add(r11, line)?;
            asm.add(rax, 4)?;
            asm.add(rdx, 4)?;
            asm.dec(r10)?;
            asm.jnz(pass)?;
            for (j, a) in out.iter().enumerate() {
                if spec.fma {
                    asm.vfmadd231ps(ZMM[a.idx()], ZMM[kv[j].idx()], ZMM[scale.idx()])?;
                } else {
                    asm.vmulps(ZMM[t.idx()], ZMM[kv[j].idx()], ZMM[scale.idx()])?;
                    asm.vaddps(ZMM[a.idx()], ZMM[a.idx()], ZMM[t.idx()])?;
                }
                asm.vmovups(zmmword_ptr(rsi + (j * 64) as i32), ZMM[a.idx()])?;
            }
            asm.vzeroupper()?;
            asm.ret()?;
            return Ok(asm.assemble(0)?);
        }

        // ── Pass 1: decay + read ───────────────────────────────────────────
        //   kv[j] = Σ_kd (s[kd][j] · decay) · k[kd],  s[kd][j] ← s[kd][j]·decay
        for a in &kv {
            asm.vxorps(ZMM[a.idx()], ZMM[a.idx()], ZMM[a.idx()])?;
        }
        asm.mov(r10, r9)?;
        asm.mov(r11, rdi)?;
        asm.mov(rax, rcx)?;
        let mut pass1 = asm.create_label();
        asm.set_label(&mut pass1)?;
        asm.vbroadcastss(ZMM[kval.idx()], dword_ptr(rax))?;
        for (j, a) in kv.iter().enumerate() {
            let off = (j * 16 * elem) as i32;
            load(&mut asm, off)?;
            asm.vmulps(ZMM[rv.idx()], ZMM[rv.idx()], ZMM[decay.idx()])?;
            if !spec.state16 {
                // In bf16 nothing is stored here: pass 2 recomputes the decay.
                asm.vmovups(zmmword_ptr(r11 + off), ZMM[rv.idx()])?;
            }
            if spec.fma {
                // kv[j] ← rv·kval + kv[j], a single rounding.
                asm.vfmadd231ps(ZMM[a.idx()], ZMM[rv.idx()], ZMM[kval.idx()])?;
            } else {
                // `rv·kval` then `+=`: two instructions, same rounding
                // as `out[vd] += row[vd] * k_val` — no FMA (header).
                asm.vmulps(ZMM[t.idx()], ZMM[rv.idx()], ZMM[kval.idx()])?;
                asm.vaddps(ZMM[a.idx()], ZMM[a.idx()], ZMM[t.idx()])?;
            }
        }
        asm.add(r11, line)?;
        asm.add(rax, 4)?;
        asm.dec(r10)?;
        asm.jnz(pass1)?;

        // ── δ = (v − kv)·beta, in place in the kv registers ─────────────
        // No state access: the hinge between the two passes.
        for (j, a) in kv.iter().enumerate() {
            let off = (j * 64) as i32;
            asm.vmovups(ZMM[t.idx()], zmmword_ptr(r8 + off))?;
            asm.vsubps(ZMM[t.idx()], ZMM[t.idx()], ZMM[a.idx()])?;
            asm.vmulps(ZMM[a.idx()], ZMM[t.idx()], ZMM[beta.idx()])?;
        }

        // ── Pass 2: write + query ─────────────────────────────────────────
        //   s[kd][j] += k[kd]·δ[j] ;  out[j] += s[kd][j] · (q[kd]·scale)
        for a in &out {
            asm.vxorps(ZMM[a.idx()], ZMM[a.idx()], ZMM[a.idx()])?;
        }
        asm.mov(r10, r9)?;
        asm.mov(r11, rdi)?;
        asm.mov(rax, rcx)?;
        let mut pass2 = asm.create_label();
        asm.set_label(&mut pass2)?;
        asm.vbroadcastss(ZMM[kval.idx()], dword_ptr(rax))?;
        asm.vbroadcastss(ZMM[qval.idx()], dword_ptr(rdx))?;
        // `q[kd]·scale` once per `kd`, like the scalar
        // (`acc += *cell * (q[kd] * scale)`) — the parentheses matter.
        asm.vmulps(ZMM[qval.idx()], ZMM[qval.idx()], ZMM[scale.idx()])?;
        for j in 0..nb {
            let off = (j * 16 * elem) as i32;
            load(&mut asm, off)?;
            if spec.state16 {
                // The decayed state again: the same product as in pass 1.
                asm.vmulps(ZMM[rv.idx()], ZMM[rv.idx()], ZMM[decay.idx()])?;
            }
            if spec.fma {
                // rv ← kval·δ[j] + rv ; then out[j] ← rv·qval + out[j].
                asm.vfmadd231ps(ZMM[rv.idx()], ZMM[kval.idx()], ZMM[kv[j].idx()])?;
                asm.vfmadd231ps(ZMM[out[j].idx()], ZMM[rv.idx()], ZMM[qval.idx()])?;
            } else {
                asm.vmulps(ZMM[t.idx()], ZMM[kval.idx()], ZMM[kv[j].idx()])?;
                asm.vaddps(ZMM[rv.idx()], ZMM[rv.idx()], ZMM[t.idx()])?;
                asm.vmulps(ZMM[t.idx()], ZMM[rv.idx()], ZMM[qval.idx()])?;
                asm.vaddps(ZMM[out[j].idx()], ZMM[out[j].idx()], ZMM[t.idx()])?;
            }
            // Read by `out` above unrounded, then stored: in bf16 by the
            // activations' `vcvtneps2bf16`, to nearest even.
            if spec.state16 {
                asm.vcvtneps2bf16(YMM[t.idx()], ZMM[rv.idx()])?;
                asm.vmovdqu16(ymmword_ptr(r11 + off), YMM[t.idx()])?;
            } else {
                asm.vmovups(zmmword_ptr(r11 + off), ZMM[rv.idx()])?;
            }
        }
        asm.add(r11, line)?;
        asm.add(rax, 4)?;
        asm.add(rdx, 4)?;
        asm.dec(r10)?;
        asm.jnz(pass2)?;

        // ── out ← registers ───────────────────────────────────────────────
        for (j, a) in out.iter().enumerate() {
            asm.vmovups(zmmword_ptr(rsi + (j * 64) as i32), ZMM[a.idx()])?;
        }
        asm.vzeroupper()?;
        asm.ret()?;
        Ok(asm.assemble(0)?)
    }
}
