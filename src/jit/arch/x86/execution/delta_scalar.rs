//! Exact scalar fallbacks emitted at mount, including non-vector geometries.
use super::{delta::Shape, math, ExecutionError, GeneratedKernel};
use iced_x86::code_asm::*;
fn err(e: impl std::fmt::Display) -> ExecutionError {
    ExecutionError(e.to_string())
}

/// Same six-integer/three-float ABI as the mounted recurrence kernel. Unless
/// `Knobs::deltanet_state_f32`, the state is bf16: a value is loaded as the
/// high half of its f32 (`pinsrw`), the decay does not store it (the update
/// recomputes `s · decay`, the same product), and the update stores it
/// rounded to nearest even after the output has read it: `(x + 0x7fff +
/// bit16(x))`, the AVX2 activations' rounding, its high half (`pextrw`).
pub(crate) fn recurrence(shape: Shape) -> Result<GeneratedKernel, ExecutionError> {
    shape.check()?;
    let round = !crate::jit::knobs::knobs().deltanet_state_f32;
    let elem = if round { 2 } else { 4 };
    let line = (shape.hvd * elem) as i32;
    // A value of the state at `m` into lane 0 of `x`, as f32.
    let load = |a: &mut CodeAssembler,
                x: AsmRegisterXmm,
                m: AsmMemoryOperand|
     -> Result<(), iced_x86::IcedError> {
        if round {
            a.pxor(x, x)?;
            a.pinsrw(x, word_ptr(m), 1)
        } else {
            a.movd(x, dword_ptr(m))
        }
    };
    let mut a = CodeAssembler::new(64).map_err(err)?;
    let emit = |a: &mut CodeAssembler| -> Result<(), iced_x86::IcedError> {
        if round {
            // xmm8 = 1, xmm9 = 0x7fff (lane 0).
            a.mov(eax, 1)?;
            a.movd(xmm8, eax)?;
            a.mov(eax, 0x7fff)?;
            a.movd(xmm9, eax)?;
        }
        let mut zero = a.create_label();
        let mut decay_k = a.create_label();
        let mut decay_v = a.create_label();
        let mut delta = a.create_label();
        let mut update_v = a.create_label();
        let mut update_k = a.create_label();
        a.xor(r11d, r11d)?;
        a.set_label(&mut zero)?;
        a.mov(dword_ptr(rsi + r11 * 4), 0)?;
        a.inc(r11)?;
        a.cmp(r11, shape.hvd as i32)?;
        a.jb(zero)?;
        a.xor(r10d, r10d)?;
        a.mov(rax, rdi)?;
        a.set_label(&mut decay_k)?;
        a.movd(xmm3, dword_ptr(rcx + r10 * 4))?;
        a.xor(r11d, r11d)?;
        a.set_label(&mut decay_v)?;
        if round {
            load(a, xmm4, rax + r11 * 2)?;
            a.mulss(xmm4, xmm1)?;
        } else {
            a.movd(xmm4, dword_ptr(rax + r11 * 4))?;
            a.mulss(xmm4, xmm1)?;
            a.movd(dword_ptr(rax + r11 * 4), xmm4)?;
        }
        a.mulss(xmm4, xmm3)?;
        a.movd(xmm5, dword_ptr(rsi + r11 * 4))?;
        a.addss(xmm5, xmm4)?;
        a.movd(dword_ptr(rsi + r11 * 4), xmm5)?;
        a.inc(r11)?;
        a.cmp(r11, shape.hvd as i32)?;
        a.jb(decay_v)?;
        a.add(rax, line)?;
        a.inc(r10)?;
        a.cmp(r10, r9)?;
        a.jb(decay_k)?;
        a.xor(r11d, r11d)?;
        a.set_label(&mut delta)?;
        a.movd(xmm3, dword_ptr(r8 + r11 * 4))?;
        a.subss(xmm3, dword_ptr(rsi + r11 * 4))?;
        a.mulss(xmm3, xmm0)?;
        a.movd(dword_ptr(rsi + r11 * 4), xmm3)?;
        a.inc(r11)?;
        a.cmp(r11, shape.hvd as i32)?;
        a.jb(delta)?;
        a.xor(r11d, r11d)?;
        a.set_label(&mut update_v)?;
        a.movd(xmm3, dword_ptr(rsi + r11 * 4))?;
        a.xorps(xmm4, xmm4)?;
        if round {
            a.lea(rax, ptr(rdi + r11 * 2))?;
        } else {
            a.lea(rax, ptr(rdi + r11 * 4))?;
        }
        a.xor(r10d, r10d)?;
        a.set_label(&mut update_k)?;
        a.movd(xmm5, dword_ptr(rcx + r10 * 4))?;
        a.mulss(xmm5, xmm3)?;
        load(a, xmm6, rax + 0)?;
        if round {
            a.mulss(xmm6, xmm1)?;
        }
        a.addss(xmm6, xmm5)?;
        if round {
            a.movaps(xmm7, xmm6)?;
            a.psrld(xmm7, 16)?;
            a.pand(xmm7, xmm8)?;
            a.paddd(xmm7, xmm6)?;
            a.paddd(xmm7, xmm9)?;
            a.pextrw(word_ptr(rax), xmm7, 1)?;
        } else {
            a.movd(dword_ptr(rax), xmm6)?;
        }
        a.movd(xmm5, dword_ptr(rdx + r10 * 4))?;
        a.mulss(xmm5, xmm2)?;
        a.mulss(xmm6, xmm5)?;
        a.addss(xmm4, xmm6)?;
        a.add(rax, line)?;
        a.inc(r10)?;
        a.cmp(r10, r9)?;
        a.jb(update_k)?;
        a.movd(dword_ptr(rsi + r11 * 4), xmm4)?;
        a.inc(r11)?;
        a.cmp(r11, shape.hvd as i32)?;
        a.jb(update_v)?;
        a.ret()
    };
    emit(&mut a).map_err(err)?;
    unsafe { GeneratedKernel::new(a.assemble(0).map_err(err)?) }
}

/// `(data, weights, state, channels, tables)` for one causal position.
pub(crate) fn convolution(shape: Shape, fma: bool) -> Result<GeneratedKernel, ExecutionError> {
    convolution_with(shape, fma, false)
}

/// [`convolution`], and with `bias` the sixth argument (`r9`) points at the channels' biases, added
/// to the sum before the SiLU (Mamba2).
pub(crate) fn convolution_with(
    shape: Shape,
    fma: bool,
    bias: bool,
) -> Result<GeneratedKernel, ExecutionError> {
    shape.check()?;
    let exp = math::exp(fma)?;
    let mut a = CodeAssembler::new(64).map_err(err)?;
    let mut exp_entry = a.create_label();
    let emit = |a: &mut CodeAssembler| -> Result<(), iced_x86::IcedError> {
        let mut channel = a.create_label();
        let mut shift = a.create_label();
        let mut sum = a.create_label();
        let mut exit = a.create_label();
        for r in [rbx, rbp, r12, r13, r14] {
            a.push(r)?;
        }
        a.sub(rsp, 16)?;
        a.mov(rbx, rdi)?;
        a.mov(rbp, rsi)?;
        a.mov(r12, rdx)?;
        a.mov(r13, rcx)?;
        a.mov(r14, r8)?;
        if bias {
            a.mov(qword_ptr(rsp + 8), r9)?;
        }
        a.test(r13, r13)?;
        a.je(exit)?;
        a.set_label(&mut channel)?;
        a.xor(ecx, ecx)?;
        a.set_label(&mut shift)?;
        a.mov(eax, dword_ptr(r12 + rcx * 4 + 4))?;
        a.mov(dword_ptr(r12 + rcx * 4), eax)?;
        a.inc(rcx)?;
        a.cmp(rcx, (shape.ck - 1) as i32)?;
        a.jb(shift)?;
        a.mov(eax, dword_ptr(rbx))?;
        a.mov(dword_ptr(r12 + (shape.ck * 4 - 4) as i32), eax)?;
        a.xor(ecx, ecx)?;
        a.xorps(xmm0, xmm0)?;
        a.set_label(&mut sum)?;
        a.movzx(eax, word_ptr(rbp + rcx * 2))?;
        a.shl(eax, 16)?;
        a.movd(xmm1, eax)?;
        a.mulss(xmm1, dword_ptr(r12 + rcx * 4))?;
        a.addss(xmm0, xmm1)?;
        a.inc(rcx)?;
        a.cmp(rcx, shape.ck as i32)?;
        a.jb(sum)?;
        if bias {
            a.mov(rax, qword_ptr(rsp + 8))?;
            a.addss(xmm0, dword_ptr(rax))?;
            a.add(qword_ptr(rsp + 8), 4)?;
        }
        a.movd(dword_ptr(rsp), xmm0)?;
        a.movd(eax, xmm0)?;
        a.xor(eax, 0x80000000u32)?;
        a.movd(xmm0, eax)?;
        a.mov(rdi, r14)?;
        a.call(exp_entry)?;
        a.mov(eax, 1f32.to_bits())?;
        a.movd(xmm1, eax)?;
        a.addss(xmm0, xmm1)?;
        a.movd(xmm1, dword_ptr(rsp))?;
        a.divss(xmm1, xmm0)?;
        a.movd(dword_ptr(rbx), xmm1)?;
        a.add(rbx, 4)?;
        a.add(rbp, (shape.ck * 2) as i32)?;
        a.add(r12, (shape.ck * 4) as i32)?;
        a.dec(r13)?;
        a.jnz(channel)?;
        a.set_label(&mut exit)?;
        a.add(rsp, 16)?;
        for r in [r14, r13, r12, rbp, rbx] {
            a.pop(r)?;
        }
        a.ret()
    };
    emit(&mut a).map_err(err)?;
    a.set_label(&mut exp_entry).map_err(err)?;
    a.db(&exp.bytes).map_err(err)?;
    unsafe { GeneratedKernel::new(a.assemble(0).map_err(err)?) }
}

/// `(out, z, weight, bf16_out, 1, tables, eps)` for one V-head row.
/// Keeps the literal left-to-right RMS sum, then SiLU, then RNE narrowing.
pub(crate) fn gated_norm(dim: usize, fma: bool) -> Result<GeneratedKernel, ExecutionError> {
    if dim == 0 || dim > i32::MAX as usize / 4 {
        return Err(ExecutionError("invalid DeltaNet gated norm width".into()));
    }
    let exp = math::exp(fma)?;
    let mut a = CodeAssembler::new(64).map_err(err)?;
    let mut exp_entry = a.create_label();
    let emit = |a: &mut CodeAssembler| -> Result<(), iced_x86::IcedError> {
        let mut sum = a.create_label();
        let mut apply = a.create_label();
        for r in [rbx, rbp, r12, r13, r14, r15] {
            a.push(r)?;
        }
        a.sub(rsp, 24)?;
        a.mov(rbx, rdi)?;
        a.mov(rbp, rsi)?;
        a.mov(r12, rdx)?;
        a.mov(r13, rcx)?;
        a.mov(r14, r9)?;
        a.xor(eax, eax)?;
        a.xorps(xmm1, xmm1)?;
        a.set_label(&mut sum)?;
        a.movd(xmm2, dword_ptr(rbx + rax * 4))?;
        a.mulss(xmm2, xmm2)?;
        a.addss(xmm1, xmm2)?;
        a.inc(rax)?;
        a.cmp(rax, dim as i32)?;
        a.jb(sum)?;
        a.mov(eax, (dim as f32).to_bits())?;
        a.movd(xmm2, eax)?;
        a.divss(xmm1, xmm2)?;
        a.addss(xmm1, xmm0)?;
        a.sqrtss(xmm1, xmm1)?;
        a.mov(eax, 1f32.to_bits())?;
        a.movd(xmm0, eax)?;
        a.divss(xmm0, xmm1)?;
        a.movd(dword_ptr(rsp), xmm0)?;
        a.xor(r15d, r15d)?;
        a.set_label(&mut apply)?;
        a.mov(eax, dword_ptr(rbp + r15 * 4))?;
        a.mov(dword_ptr(rsp + 4), eax)?;
        a.xor(eax, 0x80000000u32)?;
        a.movd(xmm0, eax)?;
        a.mov(rdi, r14)?;
        a.call(exp_entry)?;
        a.mov(eax, 1f32.to_bits())?;
        a.movd(xmm1, eax)?;
        a.addss(xmm0, xmm1)?;
        a.movd(xmm1, dword_ptr(rsp + 4))?;
        a.divss(xmm1, xmm0)?;
        a.movd(xmm0, dword_ptr(rbx + r15 * 4))?;
        a.mulss(xmm0, dword_ptr(rsp))?;
        a.movzx(eax, word_ptr(r12 + r15 * 2))?;
        a.shl(eax, 16)?;
        a.movd(xmm2, eax)?;
        a.mulss(xmm0, xmm2)?;
        a.mulss(xmm0, xmm1)?;
        a.movd(dword_ptr(rbx + r15 * 4), xmm0)?;
        // Same wrapping integer RNE formula as core::tensor::f32_to_bf16,
        // including its existing treatment of non-finite payloads.
        a.movd(eax, xmm0)?;
        a.mov(ecx, eax)?;
        a.shr(ecx, 16)?;
        a.and(ecx, 1)?;
        a.add(eax, 0x7fff)?;
        a.add(eax, ecx)?;
        a.shr(eax, 16)?;
        a.mov(word_ptr(r13 + r15 * 2), ax)?;
        a.inc(r15)?;
        a.cmp(r15, dim as i32)?;
        a.jb(apply)?;
        a.add(rsp, 24)?;
        for r in [r15, r14, r13, r12, rbp, rbx] {
            a.pop(r)?;
        }
        a.ret()
    };
    emit(&mut a).map_err(err)?;
    a.set_label(&mut exp_entry).map_err(err)?;
    a.db(&exp.bytes).map_err(err)?;
    unsafe { GeneratedKernel::new(a.assemble(0).map_err(err)?) }
}

/// Mamba2's recurrence, the ABI of [`recurrence`] (`beta` the step `dt`, `scale` the skip `D`): for
/// each `kd`, `dB = k[kd]·beta`, then for each `vd` `s ← s·decay + dB·v[vd]` and `out[vd] +=
/// s·q[kd]`; then `out[vd] += v[vd]·D`. The state bf16 unless `Knobs::deltanet_state_f32`, rounded
/// as [`recurrence`] rounds it, after `out` has read it.
pub(crate) fn recurrence_mamba(shape: Shape) -> Result<GeneratedKernel, ExecutionError> {
    shape.check()?;
    let round = !crate::jit::knobs::knobs().deltanet_state_f32;
    let elem = if round { 2 } else { 4 };
    let line = (shape.hvd * elem) as i32;
    let mut a = CodeAssembler::new(64).map_err(err)?;
    let emit = |a: &mut CodeAssembler| -> Result<(), iced_x86::IcedError> {
        if round {
            a.mov(eax, 1)?;
            a.movd(xmm8, eax)?;
            a.mov(eax, 0x7fff)?;
            a.movd(xmm9, eax)?;
        }
        let mut zero = a.create_label();
        let mut over_k = a.create_label();
        let mut over_v = a.create_label();
        let mut skip = a.create_label();
        a.xor(r11d, r11d)?;
        a.set_label(&mut zero)?;
        a.mov(dword_ptr(rsi + r11 * 4), 0)?;
        a.inc(r11)?;
        a.cmp(r11, shape.hvd as i32)?;
        a.jb(zero)?;
        a.xor(r10d, r10d)?;
        a.mov(rax, rdi)?;
        a.set_label(&mut over_k)?;
        a.movd(xmm3, dword_ptr(rcx + r10 * 4))?;
        a.mulss(xmm3, xmm0)?;
        a.movd(xmm10, dword_ptr(rdx + r10 * 4))?;
        a.xor(r11d, r11d)?;
        a.set_label(&mut over_v)?;
        if round {
            a.pxor(xmm6, xmm6)?;
            a.pinsrw(xmm6, word_ptr(rax + r11 * 2), 1)?;
        } else {
            a.movd(xmm6, dword_ptr(rax + r11 * 4))?;
        }
        a.mulss(xmm6, xmm1)?;
        a.movd(xmm5, dword_ptr(r8 + r11 * 4))?;
        a.mulss(xmm5, xmm3)?;
        a.addss(xmm6, xmm5)?;
        if round {
            a.movaps(xmm7, xmm6)?;
            a.psrld(xmm7, 16)?;
            a.pand(xmm7, xmm8)?;
            a.paddd(xmm7, xmm6)?;
            a.paddd(xmm7, xmm9)?;
            a.pextrw(word_ptr(rax + r11 * 2), xmm7, 1)?;
        } else {
            a.movd(dword_ptr(rax + r11 * 4), xmm6)?;
        }
        a.mulss(xmm6, xmm10)?;
        a.movd(xmm5, dword_ptr(rsi + r11 * 4))?;
        a.addss(xmm5, xmm6)?;
        a.movd(dword_ptr(rsi + r11 * 4), xmm5)?;
        a.inc(r11)?;
        a.cmp(r11, shape.hvd as i32)?;
        a.jb(over_v)?;
        a.add(rax, line)?;
        a.inc(r10)?;
        a.cmp(r10, r9)?;
        a.jb(over_k)?;
        a.xor(r11d, r11d)?;
        a.set_label(&mut skip)?;
        a.movd(xmm5, dword_ptr(r8 + r11 * 4))?;
        a.mulss(xmm5, xmm2)?;
        a.addss(xmm5, dword_ptr(rsi + r11 * 4))?;
        a.movd(dword_ptr(rsi + r11 * 4), xmm5)?;
        a.inc(r11)?;
        a.cmp(r11, shape.hvd as i32)?;
        a.jb(skip)?;
        a.ret()
    };
    emit(&mut a).map_err(err)?;
    unsafe { GeneratedKernel::new(a.assemble(0).map_err(err)?) }
}

/// Mamba2's gated norm over a whole row (`GraniteMoeHybridRMSNormGated`), the ABI of
/// [`gated_norm`]: `g = y · silu(z)` written over `y`, summed squared left to right, `r = 1 /
/// sqrt(Σ/dim + eps)`, then `w · (g · r)` rounded to bf16 to nearest even.
pub(crate) fn mamba_gated_norm(dim: usize, fma: bool) -> Result<GeneratedKernel, ExecutionError> {
    if dim == 0 || dim > i32::MAX as usize / 4 {
        return Err(ExecutionError("invalid Mamba2 gated norm width".into()));
    }
    let exp = math::exp(fma)?;
    let mut a = CodeAssembler::new(64).map_err(err)?;
    let mut exp_entry = a.create_label();
    let emit = |a: &mut CodeAssembler| -> Result<(), iced_x86::IcedError> {
        let mut gate = a.create_label();
        let mut apply = a.create_label();
        for r in [rbx, rbp, r12, r13, r14, r15] {
            a.push(r)?;
        }
        a.sub(rsp, 24)?;
        a.mov(rbx, rdi)?;
        a.mov(rbp, rsi)?;
        a.mov(r12, rdx)?;
        a.mov(r13, rcx)?;
        a.mov(r14, r9)?;
        a.movd(dword_ptr(rsp + 8), xmm0)?;
        a.mov(dword_ptr(rsp + 12), 0)?;
        a.xor(r15d, r15d)?;
        // g = y · z / (1 + exp(−z)), the sum of g² beside it.
        a.set_label(&mut gate)?;
        a.mov(eax, dword_ptr(rbp + r15 * 4))?;
        a.mov(dword_ptr(rsp + 4), eax)?;
        a.xor(eax, 0x80000000u32)?;
        a.movd(xmm0, eax)?;
        a.mov(rdi, r14)?;
        a.call(exp_entry)?;
        a.mov(eax, 1f32.to_bits())?;
        a.movd(xmm1, eax)?;
        a.addss(xmm0, xmm1)?;
        a.movd(xmm1, dword_ptr(rsp + 4))?;
        a.divss(xmm1, xmm0)?;
        a.movd(xmm0, dword_ptr(rbx + r15 * 4))?;
        a.mulss(xmm0, xmm1)?;
        a.movd(dword_ptr(rbx + r15 * 4), xmm0)?;
        a.mulss(xmm0, xmm0)?;
        a.addss(xmm0, dword_ptr(rsp + 12))?;
        a.movd(dword_ptr(rsp + 12), xmm0)?;
        a.inc(r15)?;
        a.cmp(r15, dim as i32)?;
        a.jb(gate)?;
        a.movd(xmm1, dword_ptr(rsp + 12))?;
        a.mov(eax, (dim as f32).to_bits())?;
        a.movd(xmm2, eax)?;
        a.divss(xmm1, xmm2)?;
        a.addss(xmm1, dword_ptr(rsp + 8))?;
        a.sqrtss(xmm1, xmm1)?;
        a.mov(eax, 1f32.to_bits())?;
        a.movd(xmm3, eax)?;
        a.divss(xmm3, xmm1)?;
        a.xor(r15d, r15d)?;
        a.set_label(&mut apply)?;
        a.movd(xmm0, dword_ptr(rbx + r15 * 4))?;
        a.mulss(xmm0, xmm3)?;
        a.movzx(eax, word_ptr(r12 + r15 * 2))?;
        a.shl(eax, 16)?;
        a.movd(xmm2, eax)?;
        a.mulss(xmm2, xmm0)?;
        a.movd(eax, xmm2)?;
        a.mov(ecx, eax)?;
        a.shr(ecx, 16)?;
        a.and(ecx, 1)?;
        a.add(eax, 0x7fff)?;
        a.add(eax, ecx)?;
        a.shr(eax, 16)?;
        a.mov(word_ptr(r13 + r15 * 2), ax)?;
        a.inc(r15)?;
        a.cmp(r15, dim as i32)?;
        a.jb(apply)?;
        a.add(rsp, 24)?;
        for r in [r15, r14, r13, r12, rbp, rbx] {
            a.pop(r)?;
        }
        a.ret()
    };
    emit(&mut a).map_err(err)?;
    a.set_label(&mut exp_entry).map_err(err)?;
    a.db(&exp.bytes).map_err(err)?;
    unsafe { GeneratedKernel::new(a.assemble(0).map_err(err)?) }
}
