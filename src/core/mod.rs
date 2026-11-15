//! What every other module needs and nothing else: the model's `config.json`,
//! the error type, bf16 conversions, the ISA and memory checks, the CPU count
//! and the sampler.
//!
//! The crate builds for x86_64 Linux and Windows, aarch64 macOS and Linux, and riscv64 Linux
//! (`lib.rs`, `build.rs`): the engine emits AVX2/AVX-512, NEON or RVV kernels and enters them
//! through the `sysv64`, AAPCS64 or LP64D calling convention. The machine that runs the binary
//! is checked again at start (`isa::refuse_if_below_floor`).

pub mod avec;
pub mod config;
pub mod cpu_detection;
pub mod error;
pub mod hugepages;
pub mod isa;
pub mod memory;
pub mod sampler;
pub mod tensor;

pub use config::Config;
pub use error::HerbertError;
