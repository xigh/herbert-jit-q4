//! herbert-serve — llama-server's HTTP API (the OpenAI chat completions with tools and thinking,
//! `/props`, `/health`, `/models/load`) in front of any herbert engine, for clients written against
//! llama-server.
//!
//! | module    | what it does |
//! |-----------|--------------|
//! | `backend` | what an engine gives: a state to feed and generate from, on its device |
//! | `image`   | the images of a request: decoded, told apart, their soft tokens counted |
//! | `audio`   | the recordings of a request: WAV read, resampled to the engine's rate, told apart |
//! | `worker`  | the requests answered one at a time: the prefix reused, the side state, the answer streamed |
//! | `prefix`  | the K/V of the stable prefixes on disk (`--prefix-cache`) |
//! | `qwen3`   | Qwen3-Embedding and Qwen3-Reranker: told apart from a chat model, the reranker's prompt and score |
//! | `server`  | the routes, the load and unload, the command line every engine's server shares |
//!
//! An engine's binary implements [`Backend`] on its engine and [`Loader`] to load one, then calls
//! [`run`] (here, `src/bin/herbert-jit-llama-serve.rs` with `src/serve/llama.rs`).

pub mod audio;
pub mod backend;
pub mod image;
pub mod prefix;
pub mod qwen3;
pub mod server;
pub mod worker;

pub use backend::{Audio, AudioFraming, Backend, Medium, Vision};
pub use server::{loaded, run, run_with, Loaded, Loader, ServeOptions, USAGE};
