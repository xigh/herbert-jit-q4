//! herbert-jit-q4 — LLM inference on the CPU (Gemma 4, Qwen3.5/3.6, Qwen3-VL, LFM2/2.5/3, their
//! vision towers and the encoders): the weights quantized to 4 bits (`Q4_0`) at load, or kept in
//! bf16, every kernel emitted at run time. The CLI is `src/main.rs`, the llama-server compatible
//! server `src/bin/herbert-jit-llama-serve.rs` (feature `serve`).
//!
//! The modules follow the order in which `main` uses them:
//!
//! | module   | what it does |
//! |----------|--------------|
//! | `core`   | the model's `config.json`, bf16 conversions, the ISA check, the memory check, the sampler |
//! | `pool`   | the worker threads the generated program runs on, pinned to physical cores |
//! | `model`  | the checkpoint read from safetensors into packed weights (R16 layout: bf16, or int8 / Q4 groups) |
//! | `jit`    | the emitters (GEMM, norms, attention, GatedDeltaNet), the generated program and its scheduler |
//! | `engine` | the model mounted into the two generated regimes (prefill, decode) and driven step by step |
//! | `chat`   | the chat templates, tokens, streaming detokenizer, tools, and the session |
//!
//! Numerical contract: weights bf16, activations bf16, accumulators f32, KV
//! cache bf16. With `--int8` or `--q4` (and by default for the server's embedding and reranking
//! models), the projection weights are quantized at load into groups of 32 values with one f32
//! scale each (int8, or 4 bits by the `Q4_0` rule), and the activations are quantized to int8
//! before those projections (`jit/int8.rs`); a matrix of a shape the int8 GEMM does not take, and
//! the vision and audio towers, stay bf16.
//!
//! Three targets emit the kernels: x86-64 (AVX2/AVX-512, `iced-x86`), AArch64
//! (NEON with the BF16 extension, the `a64` crate of this workspace) and
//! RISC-V (RVV 1.0 with the bf16 extensions, the `rv` crate).

// Index loops over emitted registers, wide kernel signatures and the
// `engine::engine` / `model::model` files read better than what these lints ask for.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::module_inception
)]

#[cfg(not(any(
    all(
        target_arch = "x86_64",
        any(target_os = "linux", target_os = "windows")
    ),
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "riscv64", target_os = "linux")
)))]
compile_error!(
    "herbert-jit-q4 builds for x86_64 Linux and Windows, aarch64 macOS and Linux, and riscv64 Linux: the \
     engine emits its own AVX2/AVX-512, NEON/BF16 or RVV kernels and calls them through \
     `sysv64`, AAPCS64 or LP64D."
);

pub mod chat;
pub mod core;
pub mod engine;
/// A model of Hugging Face found in its cache (`org/name` as the model of the CLI and the server).
pub mod hub;
pub mod jit;
pub mod model;
pub mod pool;
#[cfg(feature = "serve")]
pub mod serve;
/// The server as a service of the system: systemd (user), launchd (LaunchAgent), Windows.
#[cfg(feature = "serve")]
pub mod service;
