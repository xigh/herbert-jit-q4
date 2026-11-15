//! The engine's 16-bit format, bf16: the logits the sampler reads are in it. The conversions: exact
//! widening, rounding to nearest even.

/// bf16 → f32, exact.
#[inline(always)]
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// f32 → bf16, round to nearest even: the one rounding of the workspace (the engine re-exports it,
/// `core::tensor::f32_to_bf16`).
#[inline(always)]
pub fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    let round_bit = (bits >> 15) & 1;
    let sticky = bits & 0x7FFF;
    let mut result = bits >> 16;
    if round_bit == 1 && (sticky != 0 || lsb == 1) {
        result += 1;
    }
    result as u16
}

/// Exact widening.
#[inline(always)]
pub fn to_f32(h: u16) -> f32 {
    bf16_to_f32(h)
}

/// Round to nearest even.
#[inline(always)]
pub fn from_f32(v: f32) -> u16 {
    f32_to_bf16(v)
}
