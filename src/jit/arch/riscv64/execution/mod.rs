//! The RISC-V kernels of the execution plan: frame preparation, data
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
/// Generated memory movement, lookup and output kernels.
pub mod kernels;
/// Lowering of the KV append.
pub mod kv;
pub(crate) mod mamba;
pub(crate) mod math;
pub(crate) mod transforms;
