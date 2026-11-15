//! The AArch64 kernels of the execution plan: frame preparation, data
//! movement and output, the projection and attention traversals, the KV
//! append, the DeltaNet stages and the emitted scalar math.

#[allow(unused_imports)]
pub(crate) use crate::jit::execution::*;

/// Lowering of SDPA head/chunk traversal and split-pass orchestration.
pub mod attention;
pub(crate) mod audit;
/// Lowering of bf16 R16 projection tile traversal.
pub mod bf16;
pub(crate) mod delta;
pub(crate) mod entry;
/// Generated preparation and completion of a token block.
pub mod frame;
/// The int8 quantization of the activations.
pub mod int8;
/// Generated memory movement, lookup and output kernels.
pub mod kernels;
/// Lowering of the KV append.
pub mod kv;
pub(crate) mod mamba;
pub(crate) mod math;
/// The mixture-of-experts kernels (Gemma 4 26B-A4B, Qwen MoE, LFM2 MoE).
pub mod moe;
pub(crate) mod short_conv;
pub(crate) mod transforms;
