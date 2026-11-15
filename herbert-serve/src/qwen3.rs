//! Qwen3-Embedding and Qwen3-Reranker: Qwen3 checkpoints (`model_type` `qwen3`, `Qwen3ForCausalLM`)
//! that an engine runs as it runs a chat model, read differently at the end. What tells them apart
//! is their sentence-transformers packaging (`modules.json`), else the name of their directory.
//!
//! - Qwen3-Embedding (`1_Pooling`, `pooling_mode_lasttoken`): the hidden state of the last token
//!   after the final norm, the tokenizer appending `<|endoftext|>` (its `TemplateProcessing`), 1024
//!   values for the 0.6B. The server normalizes it. A query is sent with its instruction, which is
//!   the client's (`Instruct: Given a web search query, retrieve relevant passages that answer the
//!   query\nQuery:` and the query right after, the model card's `prompts.query`); a document is
//!   sent as it is.
//! - Qwen3-Reranker (`1_LogitScore`, or `CrossEncoder` in `config_sentence_transformers.json`, or
//!   `reranker` in the directory's name): the pair in the model's prompt ([`Reranker::prompt`]: its
//!   system message "Judge whether the Document meets the requirements…", `<Instruct>`, `<Query>`,
//!   `<Document>`, the assistant's turn opened with an empty thinking block), and the score
//!   logit(`yes`) − logit(`no`) of the last token ([`Reranker::score`]): the log-odds whose sigmoid
//!   is the model card's probability. It is computed from the same hidden state with the head's two
//!   rows, read in 16 bits from the checkpoint (not from the engine's head, Q4_0 on some), so it
//!   does not depend on whether the engine keeps the logits.

use serde_json::Value;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use tokenizers::Tokenizer;

/// How the server reads a Qwen3 checkpoint.
pub enum Head {
    /// A chat model: the logits.
    Lm,
    /// An embedding model: the last token's hidden state, `dim` values.
    Embedding { dim: usize },
    /// A reranker: the pair in its prompt, `yes` against `no`.
    Reranker(Reranker),
}

/// Qwen3-Reranker's prompt and score.
pub struct Reranker {
    /// The tokens `yes` and `no` (`1_LogitScore`'s `true_token_id`, `false_token_id`).
    pub yes: u32,
    pub no: u32,
    /// The `<Instruct>` of a request (the model card's default, `prompts.query`).
    pub instruction: String,
    /// The head's row of `yes` minus its row of `no`, in f32.
    w: Vec<f32>,
}

/// The system message and the start of the user's turn, before the pair.
const PREFIX: &str = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";
/// The end of the user's turn and the assistant's, its thinking empty: the token after it is `yes`
/// or `no`.
const SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
/// The model card's instruction when the packaging gives none.
const INSTRUCTION: &str =
    "Given a web search query, retrieve relevant passages that answer the query";

fn json(path: &Path) -> Option<Value> {
    serde_json::from_slice(&std::fs::read(path).ok()?).ok()
}

/// The head of the checkpoint in `dir`: [`Head::Lm`] for anything but a Qwen3 embedding model or
/// reranker.
pub fn head(dir: &Path) -> Result<Head, String> {
    let config = json(&dir.join("config.json")).ok_or("config.json: unreadable")?;
    let text = config.get("text_config").unwrap_or(&config);
    if text["model_type"].as_str() != Some("qwen3") {
        return Ok(Head::Lm);
    }
    let hidden = text["hidden_size"]
        .as_u64()
        .ok_or("config.json: no hidden_size")? as usize;
    let modules: Vec<(String, String)> = json(&dir.join("modules.json"))
        .and_then(|m| m.as_array().cloned())
        .unwrap_or_default()
        .iter()
        .map(|m| {
            (
                m["type"].as_str().unwrap_or_default().to_string(),
                m["path"].as_str().unwrap_or_default().to_string(),
            )
        })
        .collect();
    let name = dir
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    let st = json(&dir.join("config_sentence_transformers.json"));
    // A cross-encoder: the packaging's `yes` and `no`, else the tokenizer's.
    let logit = modules.iter().find(|(t, _)| t.ends_with("LogitScore"));
    let cross = st.as_ref().and_then(|s| s["model_type"].as_str()) == Some("CrossEncoder");
    if logit.is_some() || cross || (modules.is_empty() && name.contains("reranker")) {
        let ids = logit.and_then(|(_, p)| json(&dir.join(p).join("config.json")));
        let tok = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| format!("tokenizer.json: {e}"))?;
        let id = |key: &str, word: &str| -> Result<u32, String> {
            match ids.as_ref().and_then(|c| c[key].as_u64()) {
                Some(i) => Ok(i as u32),
                None => tok
                    .token_to_id(word)
                    .ok_or(format!("tokenizer.json: no token {word:?}")),
            }
        };
        let (yes, no) = (id("true_token_id", "yes")?, id("false_token_id", "no")?);
        let instruction = st
            .as_ref()
            .and_then(|s| s["prompts"]["query"].as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or(INSTRUCTION)
            .to_string();
        let tied = text["tie_word_embeddings"]
            .as_bool()
            .or_else(|| config["tie_word_embeddings"].as_bool())
            .unwrap_or(false);
        let (ry, rn) = (
            head_row(dir, yes as usize, hidden, tied)?,
            head_row(dir, no as usize, hidden, tied)?,
        );
        let w = ry.iter().zip(&rn).map(|(a, b)| a - b).collect();
        return Ok(Head::Reranker(Reranker {
            yes,
            no,
            instruction,
            w,
        }));
    }
    // A pooling module: the last token's, nothing after it but the normalization (no `Dense`).
    if let Some((_, p)) = modules.iter().find(|(t, _)| t.ends_with("Pooling")) {
        let pc =
            json(&dir.join(p).join("config.json")).ok_or(format!("{p}/config.json: unreadable"))?;
        if pc["pooling_mode_lasttoken"].as_bool() != Some(true) {
            return Err(format!("{p}/config.json: a Qwen3 embedding model pooled otherwise than on its last token is not served"));
        }
        if let Some((t, _)) = modules.iter().find(|(t, _)| {
            !(t.ends_with("Transformer") || t.ends_with("Pooling") || t.ends_with("Normalize"))
        }) {
            return Err(format!("modules.json: {t} after the pooling is not served"));
        }
        return Ok(Head::Embedding {
            dim: pc["word_embedding_dimension"]
                .as_u64()
                .map_or(hidden, |d| d as usize),
        });
    }
    if modules.is_empty() && name.contains("embedding") {
        return Ok(Head::Embedding { dim: hidden });
    }
    Ok(Head::Lm)
}

impl Head {
    /// The size of the vectors of an embedding model.
    pub fn embedding_dim(&self) -> Option<usize> {
        match self {
            Head::Embedding { dim } => Some(*dim),
            _ => None,
        }
    }

    /// What the logs say of it.
    pub fn describe(&self) -> String {
        match self {
            Head::Lm => "language model".into(),
            Head::Embedding { dim } => {
                format!("Qwen3 embedding model: the last token's hidden state, {dim} values")
            }
            Head::Reranker(r) => format!(
                "Qwen3 reranker: logit(yes {}) − logit(no {}) of the last token",
                r.yes, r.no
            ),
        }
    }
}

impl Reranker {
    /// The tokens of (`query`, `doc`) in the model's prompt, its three parts encoded apart without
    /// special tokens, as the model card does.
    pub fn prompt(&self, tok: &Tokenizer, query: &str, doc: &str) -> Result<Vec<u32>, String> {
        let pair = format!(
            "<Instruct>: {}\n<Query>: {query}\n<Document>: {doc}",
            self.instruction
        );
        let mut ids = Vec::new();
        for part in [PREFIX, pair.as_str(), SUFFIX] {
            ids.extend_from_slice(
                tok.encode(part, false)
                    .map_err(|e| format!("tokenizer: {e}"))?
                    .get_ids(),
            );
        }
        Ok(ids)
    }

    /// logit(`yes`) − logit(`no`) from the last token's hidden state after the final norm.
    pub fn score(&self, hidden: &[f32]) -> f32 {
        self.w
            .iter()
            .zip(hidden)
            .map(|(w, h)| (*w as f64) * (*h as f64))
            .sum::<f64>() as f32
    }
}

/// The engine's 16-bit values (`herbert_chat::half16`) of a hidden state, in f32.
pub fn widen(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| herbert_chat::half16::to_f32(b))
        .collect()
}

/// Row `row` of the language head (`lm_head.weight`, or the embeddings when they are `tied`),
/// `hidden` values, read from the checkpoint's safetensors.
fn head_row(dir: &Path, row: usize, hidden: usize, tied: bool) -> Result<Vec<f32>, String> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    let embed = &[
        "model.embed_tokens.weight",
        "embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
    ][..];
    let order: &[&[&str]] = if tied {
        &[embed]
    } else {
        &[&["lm_head.weight"], embed]
    };
    for names in order {
        for f in &files {
            let mut file = std::fs::File::open(f).map_err(|e| format!("{}: {e}", f.display()))?;
            let mut n = [0u8; 8];
            file.read_exact(&mut n)
                .map_err(|e| format!("{}: {e}", f.display()))?;
            let n = u64::from_le_bytes(n);
            let mut header = vec![0u8; n as usize];
            file.read_exact(&mut header)
                .map_err(|e| format!("{}: {e}", f.display()))?;
            let header: Value =
                serde_json::from_slice(&header).map_err(|e| format!("{}: {e}", f.display()))?;
            let Some(t) = names.iter().find_map(|k| header.get(*k)) else {
                continue;
            };
            let size = match t["dtype"].as_str() {
                Some("BF16" | "F16") => 2,
                Some("F32") => 4,
                d => return Err(format!("{}: the head in {d:?}", f.display())),
            };
            let shape: Vec<u64> = t["shape"]
                .as_array()
                .map(|a| a.iter().filter_map(Value::as_u64).collect())
                .unwrap_or_default();
            if shape.len() != 2 || shape[1] as usize != hidden || row as u64 >= shape[0] {
                return Err(format!(
                    "{}: the head's shape {shape:?}, row {row} of {hidden}",
                    f.display()
                ));
            }
            let start = t["data_offsets"][0]
                .as_u64()
                .ok_or("safetensors: no data_offsets")?;
            file.seek(SeekFrom::Start(
                8 + n + start + (row * hidden * size) as u64,
            ))
            .map_err(|e| e.to_string())?;
            let mut b = vec![0u8; hidden * size];
            file.read_exact(&mut b).map_err(|e| e.to_string())?;
            return Ok(match t["dtype"].as_str() {
                Some("BF16") => b
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                    .collect(),
                Some("F16") => b
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect(),
                _ => b
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            });
        }
    }
    Err(format!(
        "{}: no language head nor embeddings in the safetensors",
        dir.display()
    ))
}

/// IEEE half to f32, exactly.
fn f16_to_f32(h: u16) -> f32 {
    let (s, e, m) = (
        (h >> 15) as u32,
        ((h >> 10) & 0x1f) as u32,
        (h & 0x3ff) as u32,
    );
    let mag = match e {
        0 => m as f32 * (1.0 / 16_777_216.0),
        31 => f32::from_bits(0x7f80_0000 | (m << 13)),
        _ => f32::from_bits(((e + 112) << 23) | (m << 13)),
    };
    if s == 1 {
        -mag
    } else {
        mag
    }
}
