//! What tells one family of models from another once its tokens come out: the tokens that end its
//! turn, the markers its output is split on, whether it thinks unless told not to, its `bos` and
//! `eos` as `/props` gives them.

use crate::error::{Error, Result};
use crate::{gemma, qwen, Piece};
use serde_json::Value;
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Gemma 4: `<|channel>thought`, `<|tool_call>call:NAME{…}`, `<turn|>`.
    Gemma,
    /// Qwen3.5 (Qwen3.8-27B with it): `<think>`, `<tool_call>` in XML, `<|im_end|>`.
    Qwen,
    /// Qwen2.5 (`qwen2`), Qwen3-VL (`qwen3_vl_moe`) and Qwen3-2507 (`qwen3_moe`): `<tool_call>` in
    /// JSON, `<|im_end|>`; Qwen2.5's template adds its own system message when the request has
    /// none. `<think>…</think>` when a Thinking model thinks (its template opens the block,
    /// whatever `enable_thinking` says); an Instruct model never opens it.
    Qwen2,
    /// Granite 3 and 4 (`granitemoe`, `granite`, `granitemoehybrid`):
    /// `<|start_of_role|>…<|end_of_role|>`, `<|end_of_text|>`, no thinking block; a
    /// tool call (`<|tool_call|>` and a JSON list) is left in the content.
    Granite,
    /// Ministral 3 (`mistral3`): `[INST]…[/INST]`, `</s>`, `[THINK]…[/THINK]` when it thinks (the
    /// Reasoning models, told so by their default system message), `[TOOL_CALLS]NAME[ARGS]{…}`
    /// for a call, closed by the next call or the end.
    Mistral,
    /// LFM2 and LFM2.5 (`lfm2`, `lfm2_moe`; Liquid): `<|im_start|>…<|im_end|>` after
    /// `<|startoftext|>`, `<think>…</think>` when a Thinking model thinks, a call as a Python
    /// list (`<|tool_call_start|>[f(a='x'), g()]<|tool_call_end|>`).
    Lfm2,
    /// LFM3 (`lfm3`): the same turns, thinks unless told not to (`<think>`, or `<think></think>`
    /// off), a call in Qwen3.5's XML form between `<tool_call>` and `</tool_call>`.
    Lfm3,
    /// SmolLM3 (`smollm3`): `<|im_start|>…<|im_end|>`, thinks unless told not to (`/think`,
    /// `/no_think`, or `enable_thinking`), `<think>…</think>`, a call in JSON between
    /// `<tool_call>` and `</tool_call>`.
    Smol,
}

impl Family {
    /// From the checkpoint's `config.json`: a `model_type` that starts with `qwen3_5` is Qwen3.5,
    /// `qwen2` is Qwen2.5, `granitemoe` or `granite` is Granite 3, `mistral3` or `ministral3`
    /// is Ministral 3, `lfm3` or `lfm3_vl` is LFM3, another `lfm…` is LFM2, anything else is
    /// taken for Gemma 4.
    pub fn detect(model_dir: &Path) -> Family {
        let config: Value = std::fs::read(model_dir.join("config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or(Value::Null);
        match config["model_type"].as_str() {
            // Qwen3 (Chocolatine 2 among its fine-tunes) speaks as Qwen3.5: the same markers, its
            // calls in JSON.
            Some(t) if t.starts_with("qwen3_5") || t == "qwen3" => Family::Qwen,
            Some("smollm3") => Family::Smol,
            // Qwen3-VL (`qwen3_vl_moe`) and Qwen3-2507 (`qwen3_moe`), Instruct and Thinking alike,
            // speak as Qwen2.5: their calls in JSON between `<tool_call>` and
            // `</tool_call>`; a Thinking model's block as LFM2's.
            Some("qwen2" | "qwen3_vl_moe" | "qwen3_moe") => Family::Qwen2,
            Some("granitemoe" | "granite" | "granitemoehybrid") => Family::Granite,
            Some("mistral3" | "ministral3") => Family::Mistral,
            // LFM3-VL (`lfm3_vl`) speaks as LFM3.
            Some("lfm3" | "lfm3_vl") => Family::Lfm3,
            Some(t) if t.starts_with("lfm") => Family::Lfm2,
            _ => Family::Gemma,
        }
    }

    /// The family's name, the model's when the request gives none.
    pub fn name(self) -> &'static str {
        match self {
            Family::Gemma => "gemma-4",
            Family::Qwen => "qwen3.5",
            Family::Qwen2 => "qwen2.5",
            Family::Granite => "granite-3",
            Family::Mistral => "ministral-3",
            Family::Lfm2 => "lfm2",
            Family::Lfm3 => "lfm3",
            Family::Smol => "smollm3",
        }
    }

    /// Whether a request that does not set `enable_thinking` thinks: Qwen3.5's template thinks
    /// unless told not to, as llama-server renders it; Gemma's does not.
    pub fn thinks_by_default(self) -> bool {
        matches!(self, Family::Qwen | Family::Lfm3 | Family::Smol)
    }

    /// `bos_token` and `eos_token` as `/props` gives them.
    pub fn bos_eos(self) -> (&'static str, &'static str) {
        match self {
            Family::Gemma => ("<bos>", "<eos>"),
            Family::Qwen | Family::Qwen2 => ("", "<|im_end|>"),
            Family::Granite => ("<|end_of_text|>", "<|end_of_text|>"),
            Family::Mistral => ("<s>", "</s>"),
            Family::Lfm2 | Family::Lfm3 => ("<|startoftext|>", "<|im_end|>"),
            Family::Smol => ("<|begin_of_text|>", "<|im_end|>"),
        }
    }
}

/// A family's tokens, read from the model's tokenizer.
pub struct Dialect {
    pub family: Family,
    /// The tokens that end the answer.
    pub stops: Vec<u32>,
    markers: Option<qwen::Markers>,
}

impl Dialect {
    /// `eos`: the end tokens the engine read in the checkpoint's configuration (Gemma's); the
    /// family's own are added.
    pub fn new(family: Family, tokenizer: &Tokenizer, eos: &[u32]) -> Result<Dialect> {
        let id = |s: &str| {
            tokenizer
                .token_to_id(s)
                .ok_or_else(|| Error(format!("the tokenizer has no `{s}` token")))
        };
        Ok(match family {
            Family::Gemma => {
                let mut stops = eos.to_vec();
                for t in ["<turn|>", "<eos>"] {
                    if let Some(id) = tokenizer.token_to_id(t) {
                        if !stops.contains(&id) {
                            stops.push(id);
                        }
                    }
                }
                // A tool response is the client's to write: the turn ends there.
                stops.push(gemma::TOOL_RESPONSE_OPEN);
                Dialect {
                    family,
                    stops,
                    markers: None,
                }
            }
            Family::Qwen => {
                let markers = qwen::Markers {
                    think_open: id("<think>")?,
                    think_close: id("</think>")?,
                    call_open: id("<tool_call>")?,
                    call_close: id("</tool_call>")?,
                };
                Dialect {
                    family,
                    stops: vec![id("<|im_end|>")?, id("<|endoftext|>")?],
                    markers: Some(markers),
                }
            }
            Family::Qwen2 => {
                // The thinking markers when the tokenizer has them (Qwen3-VL, Qwen3-2507; not
                // Qwen2.5, whose markers are then ids no tokenizer gives, so the
                // parser never enters the block), as LFM2: the parser starts in the
                // block when the prompt ends inside one (a Thinking model's generation prompt
                // `<think>\n`); an Instruct model's prompt has none and its answer
                // stays content.
                let or_none = |s: &str| tokenizer.token_to_id(s).unwrap_or(u32::MAX);
                let markers = qwen::Markers {
                    think_open: or_none("<think>"),
                    think_close: or_none("</think>"),
                    call_open: id("<tool_call>")?,
                    call_close: id("</tool_call>")?,
                };
                Dialect {
                    family,
                    stops: vec![id("<|im_end|>")?, id("<|endoftext|>")?],
                    markers: Some(markers),
                }
            }
            Family::Granite => {
                // The parser of Qwen with markers no tokenizer gives: everything is content.
                let markers = qwen::Markers {
                    think_open: u32::MAX,
                    think_close: u32::MAX,
                    call_open: u32::MAX,
                    call_close: u32::MAX,
                };
                Dialect {
                    family,
                    stops: vec![id("<|end_of_text|>")?],
                    markers: Some(markers),
                }
            }
            Family::Mistral => {
                // A call has no closing marker: `call_close` is an id no tokenizer gives, the next
                // `[TOOL_CALLS]` or the end closes it.
                let or_none = |s: &str| tokenizer.token_to_id(s).unwrap_or(u32::MAX);
                let markers = qwen::Markers {
                    think_open: or_none("[THINK]"),
                    think_close: or_none("[/THINK]"),
                    call_open: or_none("[TOOL_CALLS]"),
                    call_close: u32::MAX,
                };
                Dialect {
                    family,
                    stops: vec![id("</s>")?],
                    markers: Some(markers),
                }
            }
            Family::Smol => {
                let markers = qwen::Markers {
                    think_open: id("<think>")?,
                    think_close: id("</think>")?,
                    call_open: id("<tool_call>")?,
                    call_close: id("</tool_call>")?,
                };
                Dialect {
                    family,
                    stops: vec![id("<|im_end|>")?, id("<|end_of_text|>")?],
                    markers: Some(markers),
                }
            }
            Family::Lfm2 | Family::Lfm3 => {
                let or_none = |s: &str| tokenizer.token_to_id(s).unwrap_or(u32::MAX);
                let (call_open, call_close) = if family == Family::Lfm3 {
                    (id("<tool_call>")?, id("</tool_call>")?)
                } else {
                    (or_none("<|tool_call_start|>"), or_none("<|tool_call_end|>"))
                };
                let markers = qwen::Markers {
                    think_open: or_none("<think>"),
                    think_close: or_none("</think>"),
                    call_open,
                    call_close,
                };
                let mut stops = vec![id("<|im_end|>")?];
                if let Some(e) = tokenizer.token_to_id("<|endoftext|>") {
                    stops.push(e);
                }
                Dialect {
                    family,
                    stops,
                    markers: Some(markers),
                }
            }
        })
    }

    /// For the logs.
    pub fn describe(&self) -> String {
        match &self.markers {
            Some(m) => format!("stops {:?} · markers {m:?}", self.stops),
            None => format!("stops {:?}", self.stops),
        }
    }

    /// The parser of the answer that follows `prompt`; `tools` (the request's) gives Qwen's
    /// parameters their types.
    pub fn parser(&self, tokenizer: &Tokenizer, tools: Option<&Value>, prompt: &[u32]) -> Parser {
        match self.markers {
            Some(m) => Parser::Qwen(qwen::OutputParser::after_prompt(m, tools, prompt)),
            None => Parser::Gemma(gemma::OutputParser::after_prompt(tokenizer, prompt)),
        }
    }
}

/// The answer split as it streams: content, thinking, tool calls.
pub enum Parser {
    Gemma(gemma::OutputParser),
    Qwen(qwen::OutputParser),
}

impl Parser {
    /// One generated token; what it adds to the answer.
    pub fn push(&mut self, tokenizer: &Tokenizer, id: u32) -> Vec<Piece> {
        match self {
            Parser::Gemma(p) => p.push(tokenizer, id),
            Parser::Qwen(p) => p.push(tokenizer, id),
        }
    }

    /// The end of the answer: what is still held.
    pub fn finish(&mut self, tokenizer: &Tokenizer) -> Vec<Piece> {
        match self {
            Parser::Gemma(p) => p.finish(tokenizer),
            Parser::Qwen(p) => p.finish(tokenizer),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Needs Qwen3-VL's tokenizer ([`crate::test_model_file`]). Thinking's generation prompt opens
    /// `<think>`: the thought goes to the reasoning, the rest to the content. Instruct's prompt
    /// opens nothing: everything is content, as before.
    #[test]
    #[ignore = "needs Qwen3-VL's tokenizer.json (HERBERT_MODELS)"]
    fn qwen3_vl_thinking_splits_instruct_stays_content() {
        let path = crate::test_model_file(
            &[
                "qwen/Qwen3-VL-30B-A3B-Thinking",
                "qwen/Qwen3-VL-30B-A3B-Instruct",
                "Qwen3-VL-30B-A3B-Thinking",
                "Qwen3-VL-30B-A3B-Instruct",
            ],
            "tokenizer.json",
        );
        let tok = Tokenizer::from_file(&path).unwrap();
        let d = Dialect::new(Family::Qwen2, &tok, &[]).unwrap();
        let ids = |text: &str| tok.encode(text, false).unwrap().get_ids().to_vec();
        let run = |prompt: &str, answer: &str| {
            let mut p = d.parser(&tok, None, &ids(prompt));
            let mut pieces = Vec::new();
            for t in ids(answer) {
                pieces.extend(p.push(&tok, t));
            }
            pieces.extend(p.finish(&tok));
            let pick = |reasoning: bool| -> String {
                pieces
                    .iter()
                    .filter_map(|x| match x {
                        Piece::Reasoning(s) if reasoning => Some(s.as_str()),
                        Piece::Content(s) if !reasoning => Some(s.as_str()),
                        _ => None,
                    })
                    .collect()
            };
            (pick(true), pick(false))
        };
        let (r, c) = run(
            "<|im_start|>user\nCiel ?<|im_end|>\n<|im_start|>assistant\n<think>\n",
            "La diffusion de Rayleigh.\n</think>\n\nLe ciel est bleu.",
        );
        assert_eq!(
            (r.as_str(), c.as_str()),
            ("La diffusion de Rayleigh.", "Le ciel est bleu.")
        );
        let (r, c) = run(
            "<|im_start|>user\nCiel ?<|im_end|>\n<|im_start|>assistant\n",
            "Le ciel est bleu.",
        );
        assert_eq!((r.as_str(), c.as_str()), ("", "Le ciel est bleu."));
    }
}
