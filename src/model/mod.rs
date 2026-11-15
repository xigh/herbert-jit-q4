//! The checkpoint in memory: the model structures, the packed weight layout
//! the emitted GEMM reads, and the loaders that fill them (the weights live in the huge-page
//! buffer of `core/hugepages.rs`).

pub mod gemma;
pub mod loader;
pub mod model;
pub mod weight;

pub use model::{Attention, DecoderLayer, GatedDeltaNet, LayerBlock, Mlp, Model, QwenMoe};
pub use weight::PackedBf16R16Weight;
