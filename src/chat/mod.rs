//! The chat layer of the CLI: the tokenizer and its special tokens, the Qwen template (and the
//! Gemma 4 chat, `gemma.rs`),
//! the streaming detokenizer, the tools, and the session that ties them to
//! the engine.

pub mod detok;
pub mod gemma;
pub mod session;
pub mod template;
pub mod tokens;
pub mod tools;

pub use session::{Block, Chat, TurnStats};
pub use tokens::Tokens;
pub use tools::{current_date_time, Tool};
