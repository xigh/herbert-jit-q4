//! herbert-chat — what an engine needs to know about a model's chat, whatever runs the model:
//!
//! | module     | what it does |
//! |------------|--------------|
//! | `family`   | Gemma 4, Qwen3.5/Qwen3, Qwen2.5, Granite, Ministral 3, LFM2, LFM3, SmolLM3: the tokens that end a turn, the parser of the answer, the thinking default |
//! | `template` | the checkpoint's own chat template (jinja), as llama-server renders it |
//! | `gemma`    | Gemma 4's answer split into content, thinking and tool calls |
//! | `qwen`     | the same for the other families (Qwen3.5's markers, or each family's own) |
//! | `detok`    | the streaming detokenization |
//! | `sampler`  | temperature, top-k, top-p, min-p, the penalties |
//! | `half16`   | the engine's 16-bit format (bf16, or f16 with `f16`), the logits' |
//! | `progress` | how far a load has gone, for a server that answers while its model loads |

pub mod detok;
pub mod error;
pub mod family;
pub mod gemma;
pub mod half16;
pub mod media;
pub mod progress;
pub mod qwen;
pub mod sampler;
pub mod template;

/// The model files the tests that need one read: under `$HERBERT_MODELS` (default `~/models`), the
/// first of `candidates` (folders relative to it) that holds `file`. Those tests are `#[ignore]`d
/// (`cargo test -- --ignored` runs them) and fail, naming what they looked for, when none is
/// there: a test that returns early would pass without testing anything.
#[cfg(test)]
pub(crate) fn test_model_file(candidates: &[&str], file: &str) -> std::path::PathBuf {
    let root = std::env::var_os("HERBERT_MODELS").map_or_else(
        || std::path::Path::new(&std::env::var_os("HOME").unwrap_or_default()).join("models"),
        std::path::PathBuf::from,
    );
    candidates
        .iter()
        .map(|m| root.join(m).join(file))
        .find(|p| p.exists())
        .unwrap_or_else(|| {
            panic!(
                "none of {candidates:?} holds {file} under {} (set HERBERT_MODELS)",
                root.display()
            )
        })
}

pub use error::{Error, Result};
pub use family::{Dialect, Family, Parser};
pub use template::Template;

/// A piece of the answer, as it streams.
#[derive(Debug, Clone, PartialEq)]
pub enum Piece {
    Content(String),
    Reasoning(String),
    ToolCall { name: String, arguments: String },
}
