//! The node that quantizes the activations of the projections to int8 (`jit/int8.rs`), x86-64
//! (AVX2).
use super::{ExecutionError, GeneratedKernel};
use crate::jit::execution::abi::PackCall;
use crate::jit::int8::{a_row_bytes, GROUP};
use iced_x86::code_asm::*;

fn err(e: impl std::fmt::Display) -> ExecutionError {
    ExecutionError(e.to_string())
}

/// Emit `(call: *const PackCall, row: usize)`: row `row` of `A` (row-major, `k` bf16) quantized
/// into row `row` of the destination: `k` int8, then per group of 32 the f32 scale (`absmax / 127`)
/// and the int32 sum of the 32 codes (`jit/int8.rs`). The values are multiplied by `127 / max`,
/// rounded to nearest even (`vcvtps2dq`) and packed with saturation (they are within ±127). A group
/// of zeros (or of values below 1e-20) is quantized with `max = 1e-20`. Rows past the current count
/// are not touched.
pub fn quantize(k: usize) -> Result<GeneratedKernel, ExecutionError> {
    if k == 0 || !k.is_multiple_of(GROUP) {
        return Err(ExecutionError(format!(
            "no int8 quantization of rows of {k}: groups of {GROUP}"
        )));
    }
    let build = || -> Result<Vec<u8>, iced_x86::IcedError> {
        let mut a = CodeAssembler::new(64)?;
        a.set_prefer_vex(true);
        let mut exit = a.create_label();
        a.mov(
            rax,
            qword_ptr(rdi + std::mem::offset_of!(PackCall, rows) as i32),
        )?;
        a.mov(rax, qword_ptr(rax))?;
        a.cmp(rsi, rax)?;
        a.jae(exit)?;
        a.mov(
            r8,
            qword_ptr(rdi + std::mem::offset_of!(PackCall, source) as i32),
        )?;
        a.imul_3(r9, rsi, (2 * k) as i32)?;
        a.add(r8, r9)?;
        a.mov(
            r10,
            qword_ptr(rdi + std::mem::offset_of!(PackCall, dest) as i32),
        )?;
        a.imul_3(r9, rsi, a_row_bytes(k) as i32)?;
        a.add(r10, r9)?;
        a.lea(r11, qword_ptr(r10 + k as i32))?;
        // Constants: ymm15 = |x| mask, xmm14 = 1e-20, xmm13 = 1/127, xmm12 = 127, ymm11 = vpermd
        // order 0,4,1,5,2,6,3,7.
        a.vpcmpeqd(ymm15, ymm15, ymm15)?;
        a.vpsrld(ymm15, ymm15, 1u32)?;
        a.mov(eax, 1.0e-20f32.to_bits())?;
        a.vmovd(xmm14, eax)?;
        a.mov(eax, (1.0f32 / 127.0).to_bits())?;
        a.vmovd(xmm13, eax)?;
        a.mov(eax, 127.0f32.to_bits())?;
        a.vmovd(xmm12, eax)?;
        a.sub(rsp, 32)?;
        for (i, v) in [0u32, 4, 1, 5, 2, 6, 3, 7].into_iter().enumerate() {
            a.mov(dword_ptr(rsp + 4 * i as i32), v)?;
        }
        a.vmovdqu(ymm11, ymmword_ptr(rsp))?;
        a.add(rsp, 32)?;
        let mut top = a.create_label();
        a.mov(ecx, (k / GROUP) as u32)?;
        a.set_label(&mut top)?;
        for (i, y) in [ymm0, ymm1, ymm2, ymm3].into_iter().enumerate() {
            a.vpmovzxwd(y, xmmword_ptr(r8 + 16 * i as i32))?;
            a.vpslld(y, y, 16u32)?;
        }
        for (i, y) in [ymm4, ymm5, ymm6, ymm7].into_iter().enumerate() {
            a.vandps(y, [ymm0, ymm1, ymm2, ymm3][i], ymm15)?;
        }
        a.vmaxps(ymm4, ymm4, ymm5)?;
        a.vmaxps(ymm6, ymm6, ymm7)?;
        a.vmaxps(ymm4, ymm4, ymm6)?;
        a.vextractf128(xmm5, ymm4, 1)?;
        a.vmaxps(xmm4, xmm4, xmm5)?;
        a.vmovhlps(xmm5, xmm5, xmm4)?;
        a.vmaxps(xmm4, xmm4, xmm5)?;
        a.vmovshdup(xmm5, xmm4)?;
        a.vmaxss(xmm4, xmm4, xmm5)?;
        a.vmaxss(xmm4, xmm4, xmm14)?;
        a.vmulss(xmm6, xmm4, xmm13)?;
        a.vmovss(dword_ptr(r11), xmm6)?;
        a.vdivss(xmm7, xmm12, xmm4)?;
        a.vbroadcastss(ymm7, xmm7)?;
        for y in [ymm0, ymm1, ymm2, ymm3] {
            a.vmulps(y, y, ymm7)?;
            a.vcvtps2dq(y, y)?;
        }
        // The sum of the 32 codes.
        a.vpaddd(ymm4, ymm0, ymm1)?;
        a.vpaddd(ymm5, ymm2, ymm3)?;
        a.vpaddd(ymm4, ymm4, ymm5)?;
        a.vextracti128(xmm5, ymm4, 1)?;
        a.vpaddd(xmm4, xmm4, xmm5)?;
        a.vpshufd(xmm5, xmm4, 0x4e)?;
        a.vpaddd(xmm4, xmm4, xmm5)?;
        a.vpshufd(xmm5, xmm4, 0xb1)?;
        a.vpaddd(xmm4, xmm4, xmm5)?;
        a.vmovd(dword_ptr(r11 + 4), xmm4)?;
        a.vpackssdw(ymm0, ymm0, ymm1)?;
        a.vpackssdw(ymm2, ymm2, ymm3)?;
        a.vpacksswb(ymm0, ymm0, ymm2)?;
        a.vpermd(ymm0, ymm11, ymm0)?;
        a.vmovdqu(ymmword_ptr(r10), ymm0)?;
        a.add(r10, 32)?;
        a.add(r11, 8)?;
        a.add(r8, 64)?;
        a.dec(ecx)?;
        a.jne(top)?;
        a.set_label(&mut exit)?;
        a.vzeroupper()?;
        a.ret()?;
        a.assemble(0)
    };
    unsafe { GeneratedKernel::new(build().map_err(err)?) }
}
