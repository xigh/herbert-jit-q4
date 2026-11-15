//! SDPA, AVX-512: the registers, the constants, the exponential and the
//! horizontal reductions every pass uses.

#[allow(unused_imports)]
use super::*;

pub(super) const ZMM: [AsmRegisterZmm; 32] = [
    zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7, zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14,
    zmm15, zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23, zmm24, zmm25, zmm26, zmm27,
    zmm28, zmm29, zmm30, zmm31,
];
pub(super) const YMM: [AsmRegisterYmm; 32] = [
    ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14,
    ymm15, ymm16, ymm17, ymm18, ymm19, ymm20, ymm21, ymm22, ymm23, ymm24, ymm25, ymm26, ymm27,
    ymm28, ymm29, ymm30, ymm31,
];
pub(super) const XMM: [AsmRegisterXmm; 32] = [
    xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14,
    xmm15, xmm16, xmm17, xmm18, xmm19, xmm20, xmm21, xmm22, xmm23, xmm24, xmm25, xmm26, xmm27,
    xmm28, xmm29, xmm30, xmm31,
];

// The eight constants of the `exp` polynomial, in the order that
// [`exp_inline`] consumes them: one copy for the whole crate.
pub(super) use crate::jit::exp_constants::{
    EXP_C0, EXP_C1, EXP_C2, EXP_C3, EXP_C4, EXP_LOG2E, EXP_NEG87, EXP_ONE_BITS,
};

/// Name of the sink role for attention.
pub(super) const SINK_ROLE: &str = "softmax:sink (scalar)";

/// The eight polynomial registers, **allocated** in the high zone.
///
/// The high zone (`zmm16-31`) leaves the low zone, the only one VEX
/// instructions can address, to the roles that need a `ymm`/`xmm` view. It is
/// requested at [`RegFile::high`], which refuses rather than overflowing
/// silently.
#[derive(Clone, Copy, Debug)]
pub(super) struct ExpConsts {
    pub(super) neg87: V,
    pub(super) log2e: V,
    pub(super) c4: V,
    pub(super) c3: V,
    pub(super) c2: V,
    pub(super) c1: V,
    pub(super) c0: V,
    pub(super) one: V,
}

impl ExpConsts {
    pub(super) fn alloc(f: &mut RegFile) -> std::result::Result<Self, Decline> {
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

/// Aligned scratch bytes reserved by the prologue, per zone.
///
/// Two distinct zones: [`EXP_BUF`] (the 16 scores of the fused form's
/// current block) and [`HOME`] (the anchored arguments of the chained shape).
/// They must not overlap: both are alive at the same time.
pub(super) const SCRATCH_BYTES: i32 = 64;
/// The 16 f32 scores of the fused form's current block (64 bytes, 64-byte
/// aligned), written by its QK and re-read by its softmax and SV.
pub(super) const EXP_BUF: i32 = 0;
/// Home of arguments (chained shape; scale of the fused).
pub(super) const HOME: i32 = SCRATCH_BYTES;
/// Scale hosted by QK at 16 decode positions, in the [`HOME`] area
/// and **beyond** the three pointers of the chained shape
/// (`+8, +16, +24`): the two coexist in the same blob.
pub(super) const QK_SCALE: i32 = HOME + 32;
/// Size of the frame **outside** the SV pass f32 scratch buffer: the two zones.
pub(super) const FRAME_BYTES: i32 = HOME + SCRATCH_BYTES;
/// Scratch f32 of the output accumulators that the SV pass does not hold
/// in registers — **a third zone**, disjoint from the other two.
///
/// It lives on the **stack**, not in a caller buffer: it is
/// zero allocation by construction, it is thread-local with
/// nothing to synchronize, and it is the hottest memory in the process.
/// 64 bytes per overflowed segment, so `320` at `head_dim` 512 and `2048` if
/// [`SdpaKernelSpec::sv_spill_all`].
pub(super) const SV_SPILL: i32 = FRAME_BYTES;

/// Frame to reserve for `spec`: the three fixed zones, plus the SV pass
/// f32 scratch. Remainder is a multiple of 64, so the prologue alignment
/// holds.
pub(super) fn frame_bytes(spec: &SdpaKernelSpec, pass: Option<Pass>) -> i32 {
    let sv_here = matches!(
        (spec.form, pass),
        (SdpaForm::Chained, _) | (SdpaForm::Split, Some(Pass::Sv))
    );
    if sv_here {
        FRAME_BYTES + 64 * spec.sv_spill() as i32
    } else {
        FRAME_BYTES
    }
}

// The generated code never calls out: the exponential is emitted in the blob
// ([`exp_inline`]).

/// Generates the blob described by `spec`. `pass` is only meaningful for
/// [`SdpaForm::Split`], where it selects which of the three passes to emit;
/// the `Chained` and `Fused` forms produce a single blob.
///
/// The code is **position-independent**: no constants in rodata, no
/// relocations, no external symbols — only short jumps.
pub fn emit_sdpa(spec: &SdpaKernelSpec, pass: Option<Pass>) -> Result<Vec<u8>> {
    spec.check()
        .map_err(|d| HerbertError::Backend(format!("backend_jit sdpa: {d}")))?;
    if spec.form == SdpaForm::Split && pass.is_none() {
        return Err(HerbertError::Backend(
            "backend_jit sdpa: split shape, specify the pass".to_string(),
        ));
    }
    assemble(spec, pass).map_err(|e| {
        HerbertError::Backend(format!(
            "backend_jit: sdpa emission head_dim={} shape={} exp={} pass={pass:?}: {e}",
            spec.shape.head_dim,
            spec.form.name(),
            spec.exp.name(),
        ))
    })
}

/// Generates the prefill blob described by `spec`: a block of `spec.rows`
/// requests against the same K/V, cf. `assemble_prefill`.
pub fn emit_sdpa_prefill(spec: &SdpaPrefillSpec) -> Result<Vec<u8>> {
    spec.check()
        .map_err(|d| HerbertError::Backend(format!("backend_jit sdpa prefill: {d}")))?;
    assemble_prefill(spec).map_err(|e| {
        HerbertError::Backend(format!(
            "backend_jit: sdpa prefill emission head_dim={} rows={} exp={}: {e}",
            spec.shape.head_dim,
            spec.rows,
            spec.exp.name(),
        ))
    })
}

/// Loads an f32 constant into an entire zmm, without rodata or relocation.
pub(super) fn bcast_imm(
    asm: &mut CodeAssembler,
    dst: V,
    bits: u32,
) -> std::result::Result<(), IcedError> {
    asm.mov(eax, bits)?;
    asm.vpbroadcastd(ZMM[dst.idx()], eax)?;
    Ok(())
}

/// Materializes the eight polynomial constants (`Inline` strategy).
pub(super) fn load_exp_constants(
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

/// `io = exp(io)`, `t1/t2/t3` destroyed.
///
/// The input is clamped below at −87.34 and scaled by `log2(e)`; the nearest
/// integer `n` is split off, the remainder goes through a degree-4 polynomial
/// and `n` is added to the exponent. The constants are registers
/// ([`ExpConsts`]), not memory operands: the blob stays position-independent.
/// `emit_elem.rs` emits the same sequence with the same constants.
pub(super) fn exp_inline(
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

/// The exp strategy **and what it requires**, bundled together so the call
/// site doesn't have to remember it. The inline exponential clobbers only the registers it is
/// given.
pub(super) struct Exp {
    pub(super) strategy: ExpStrategy,
    pub(super) consts: Option<ExpConsts>,
}

/// `io = exp(io)` according to the chosen strategy.
///
/// `t1/t2/t3` are the polynomial's temporaries ([`exp_inline`]): nothing outside the given
/// registers is clobbered.
pub(super) fn emit_exp(
    asm: &mut CodeAssembler,
    e: &Exp,
    io: V,
    t1: V,
    t2: V,
    t3: V,
) -> std::result::Result<(), IcedError> {
    // The blob re-emits the polynomial, always: no symbol to resolve.
    match e.strategy {
        ExpStrategy::Inline => {
            let k = e
                .consts
                .expect("ExpStrategy::Inline without allocated constants");
            exp_inline(asm, &k, io, t1, t2, t3)
        }
    }
}

/// Horizontal sum of a zmm: scalar result in `xmm[dst]` (lane 0).
///
/// [`Vlow`] and not [`V`]: `vextractf128`, `vhaddps` and `vaddps xmm` are
/// emitted in **VEX**, which stops at `xmm15`. The domain is here, in the
/// signature — not in a comment above the caller.
pub(super) fn hsum(
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

/// Maximum horizontal of a zmm: scalar result in `xmm[dst]` (lane 0).
/// Same VEX constraint as [`hsum`].
pub(super) fn hmax(
    asm: &mut CodeAssembler,
    dst: Vlow,
    t: Vlow,
) -> std::result::Result<(), IcedError> {
    let (dst, t) = (dst.x(), t.x());
    asm.vextractf64x4(YMM[t], ZMM[dst], 1)?;
    asm.vmaxps(YMM[dst], YMM[dst], YMM[t])?;
    asm.vextractf128(XMM[t], YMM[dst], 1)?;
    asm.vmaxps(XMM[dst], XMM[dst], XMM[t])?;
    asm.vshufps(XMM[t], XMM[dst], XMM[dst], 0b01_00_11_10)?;
    asm.vmaxps(XMM[dst], XMM[dst], XMM[t])?;
    asm.vshufps(XMM[t], XMM[dst], XMM[dst], 0b10_11_00_01)?;
    asm.vmaxps(XMM[dst], XMM[dst], XMM[t])?;
    Ok(())
}

/// Load 16 bf16 from `[base + off]` and extend them to f32 in `ZMM[dst]`.
/// This is the conversion of the proven path (`vpmovzxwd` + `vpslld 16`), not a
/// `vdpbf16ps` contraction: the scalar reference multiplies correctly in f32.
pub(super) fn load_bf16_f32(
    asm: &mut CodeAssembler,
    dst: V,
    base: AsmRegister64,
    off: i32,
) -> std::result::Result<(), IcedError> {
    let dst = dst.idx();
    asm.vpmovzxwd(ZMM[dst], ymmword_ptr(base + off))?;
    asm.vpslld(ZMM[dst], ZMM[dst], 16u32)?;
    Ok(())
}

// ─────────────── partial last block: `head_dim` a multiple of 8 ───────────────
//
// At `head_dim` 72 a row is 2 whole blocks of 32 bf16 and 8 bf16, or 4 whole
// vectors of 16 f32 and 8 lanes. The whole blocks are emitted exactly as before;
// only the last, partial one goes through three mask registers, set once per
// blob by [`set_tail_masks`] and touched by nothing else (`k1` is the QK group's
// `0x3333`). No masked access reads beyond the row: the row that follows, or the
// end of the buffer, may hold anything (`0 × Inf = NaN`), and a masked-off
// element raises no fault.

/// `vdpbf16ps` of the partial block: its f32 lanes, one per pair of bf16.
pub(super) const K_TAIL_DOT: AsmRegisterK = k2;
/// Load of the partial block into a register: its bf16 words, the rest zeroed.
pub(super) const K_TAIL_LOAD: AsmRegisterK = k3;
/// Partial vector of the SV pass: its f32 lanes on the load (`vpmovzxwd`), its
/// bf16 words on the store.
pub(super) const K_TAIL_SV: AsmRegisterK = k4;

/// Is block `c` of 32 bf16 the partial last one?
pub(super) fn tail32_at(shape: SdpaShape, c: usize) -> bool {
    shape.tail32() != 0 && c + 1 == shape.vecs().div_ceil(2)
}

/// Is vector `c` of 16 f32 the partial last one?
pub(super) fn tail16_at(shape: SdpaShape, c: usize) -> bool {
    shape.tail16() != 0 && c + 1 == shape.vecs()
}

/// Sets the masks of the partial block, through `eax`. Emits nothing when
/// `head_dim` is a multiple of 32: the blob of those shapes does not move a byte.
pub(super) fn set_tail_masks(
    asm: &mut CodeAssembler,
    shape: SdpaShape,
) -> std::result::Result<(), IcedError> {
    let t32 = shape.tail32() as u32;
    if t32 != 0 {
        asm.mov(eax, (1u32 << (t32 / 2)) - 1)?;
        asm.kmovd(K_TAIL_DOT, eax)?;
        asm.mov(eax, (1u32 << t32) - 1)?;
        asm.kmovd(K_TAIL_LOAD, eax)?;
    }
    let t16 = shape.tail16() as u32;
    if t16 != 0 {
        asm.mov(eax, (1u32 << t16) - 1)?;
        asm.kmovd(K_TAIL_SV, eax)?;
    }
    Ok(())
}

/// `ZMM[dst]` = the 32 bf16 of block `c` at `at`, or only the partial block's
/// words, zero beyond.
pub(super) fn load_bf16_block(
    asm: &mut CodeAssembler,
    shape: SdpaShape,
    c: usize,
    dst: usize,
    at: AsmMemoryOperand,
) -> std::result::Result<(), IcedError> {
    if tail32_at(shape, c) {
        asm.vmovdqu16(ZMM[dst].k3().z(), at)
    } else {
        asm.vmovdqu16(ZMM[dst], at)
    }
}

/// `acc += a · b` over the 32 bf16 of block `c`, or over the partial block's
/// lanes only: the others keep `acc` and never read `b`.
pub(super) fn dot_bf16_block<B>(
    asm: &mut CodeAssembler,
    shape: SdpaShape,
    c: usize,
    acc: usize,
    a: usize,
    b: B,
) -> std::result::Result<(), IcedError>
where
    CodeAssembler:
        iced_x86::code_asm::asm_traits::CodeAsmVdpbf16ps<AsmRegisterZmm, AsmRegisterZmm, B>,
{
    if tail32_at(shape, c) {
        asm.vdpbf16ps(ZMM[acc].k2(), ZMM[a], b)
    } else {
        asm.vdpbf16ps(ZMM[acc], ZMM[a], b)
    }
}

/// [`load_bf16_f32`] for vector `c` of a row: the partial last one loads its
/// lanes only and zeroes the rest.
pub(super) fn load_bf16_f32_at(
    asm: &mut CodeAssembler,
    shape: SdpaShape,
    c: usize,
    dst: V,
    base: AsmRegister64,
    off: i32,
) -> std::result::Result<(), IcedError> {
    if !tail16_at(shape, c) {
        return load_bf16_f32(asm, dst, base, off);
    }
    let dst = dst.idx();
    asm.vpmovzxwd(ZMM[dst].k4().z(), ymmword_ptr(base + off))?;
    asm.vpslld(ZMM[dst], ZMM[dst], 16u32)?;
    Ok(())
}

/// Store of the bf16 conversion `ymm[t]` of vector `c`: the partial last one
/// writes its words only, never the next head's.
pub(super) fn store_bf16_at(
    asm: &mut CodeAssembler,
    shape: SdpaShape,
    c: usize,
    at: AsmMemoryOperand,
    t: Vlow,
) -> std::result::Result<(), IcedError> {
    if tail16_at(shape, c) {
        asm.vmovdqu16(at.k4(), YMM[t.y()])
    } else {
        asm.vmovups(at, YMM[t.y()])
    }
}
