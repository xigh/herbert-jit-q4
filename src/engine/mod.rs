//! The model mounted into the generated engine: the bf16 loader with its
//! census, the geometry checks, the DeltaNet packs, the weight borrowing, the
//! `Engine` the chat drives, the trace writer and the settings `main` builds.

pub mod audio;
pub mod embed_gemma;
pub mod engine;
pub mod geometry;
pub mod loader;
pub mod packs;
pub mod prefill_profile;
pub mod tensors;
pub mod trace;
pub mod tuning;
pub mod vision;
pub mod vision_lfm3;
pub mod vision_qwen;
pub mod weights_of;
pub mod xlmr;

pub use engine::Engine;
pub use tuning::{Measure, Tuning};
