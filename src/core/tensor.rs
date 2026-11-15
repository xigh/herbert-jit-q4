//! The bf16 element type and its two conversions: widening is exact, narrowing
//! rounds to nearest even like the `vcvtneps2bf16` epilogue of the kernels.

pub type BF16 = u16;

/// The two conversions, shared with the sampler and the server (`herbert_chat::half16`): one
/// rounding for the whole workspace.
pub use herbert_chat::half16::{bf16_to_f32, f32_to_bf16};
