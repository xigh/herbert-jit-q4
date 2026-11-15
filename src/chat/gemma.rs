//! The chat of a Gemma 4 model: `<bos><|turn>user\n…<turn|>\n<|turn>model\n`, the answer read until
//! a turn end or the end of text. Single-pass and greedy, the figures printed as the Qwen chat
//! prints them.

use crate::core::error::{HerbertError, Result};
use crate::engine::Engine;
use std::io::Write;
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

/// The end tokens: `<eos>`, `<turn|>` and one more end marker, as ids fixed here, not read from
/// the checkpoint's `generation_config.json`.
const STOPS: [u32; 3] = [1, 106, 50];

/// One turn per `asks` entry, on the same conversation; `audio`, a WAV recording, goes before the
/// first turn's text (`<|audio>`, its soft tokens, `<audio|>`, as the processor expands it).
pub fn run(
    engine: &mut Engine,
    model_dir: &Path,
    asks: &[String],
    audio: Option<&Path>,
    system: Option<&str>,
    think: bool,
    max_tokens: usize,
) -> Result<()> {
    let path = crate::model::loader::resolve_model_dir(model_dir)?.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&path)
        .map_err(|e| HerbertError::ModelLoad(format!("{}: {e}", path.display())))?;
    let encode = |text: &str| -> Result<Vec<u32>> {
        Ok(tokenizer
            .encode(text, false)
            .map_err(|e| HerbertError::Backend(format!("tokenizer: {e}")))?
            .get_ids()
            .to_vec())
    };
    eprintln!(
        "[tokens] Gemma 4 turns, stops {STOPS:?}, vocab {}",
        tokenizer.get_vocab_size(true)
    );
    let mut out = std::io::stdout();
    for (turn, ask) in asks.iter().enumerate() {
        let mut text = String::new();
        if turn == 0 {
            text.push_str("<bos>");
            if think || system.is_some() {
                text.push_str("<|turn>system\n");
                if think {
                    text.push_str("<|think|>\n");
                }
                if let Some(s) = system {
                    text.push_str(s);
                }
                text.push_str("<turn|>\n");
            }
        } else {
            // The turn end the previous answer stopped on was not fed to the state.
            text.push_str("<turn|>\n");
        }
        text.push_str("<|turn>user\n");
        let started = Instant::now();
        let mut prompt = 0;
        if let (0, Some(path)) = (turn, audio) {
            // The text up to `<|audio>`, the recording's soft tokens, then the rest from
            // `<audio|>`.
            let (wave, rate) = crate::engine::audio::read_wav(path)?;
            let wave =
                crate::engine::audio::resample(&wave, rate, crate::engine::audio::SAMPLE_RATE);
            let soft = match engine.audio() {
                Some((_, _, Some(chunk))) => wave
                    .len()
                    .min(crate::engine::audio::MAX_SAMPLES)
                    .div_ceil(chunk),
                _ => crate::engine::audio::soft_tokens(wave.len()),
            };
            text.push_str("<|audio>");
            let ids = encode(&text)?;
            engine.prefill(&ids)?;
            engine.prefill_audio(&wave, soft)?;
            prompt += ids.len() + soft;
            eprintln!(
                "[audio] {}: {:.2} s at {rate} Hz, {soft} soft tokens",
                path.display(),
                wave.len() as f64 / crate::engine::audio::SAMPLE_RATE as f64
            );
            text = String::from("<audio|>");
        }
        text.push_str(&format!("{ask}<turn|>\n<|turn>model\n"));
        let ids = encode(&text)?;
        println!("\n>>> {ask}");
        let first = engine.prefill(&ids)?;
        prompt += ids.len();
        let prefill_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let mut produced: Vec<u32> = Vec::with_capacity(max_tokens);
        if !STOPS.contains(&first) {
            produced.push(first);
            let room = max_tokens.saturating_sub(1);
            let blocks = engine.generate(first, room, &mut |argmax, _logits| {
                if STOPS.contains(&argmax) {
                    return Ok(None);
                }
                produced.push(argmax);
                Ok(Some(argmax))
            })?;
            produced.truncate(blocks.max(1));
        }
        let decode_seconds = started.elapsed().as_secs_f64();
        let answer = tokenizer.decode(&produced, true).unwrap_or_default();
        println!("\n[answer]\n{answer}");
        let _ = out.flush();
        eprintln!(
            "[ids] {}",
            produced
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        );
        eprintln!("[decoded] {:?}", answer);
        eprintln!(
            "[turn {}] {} prompt tokens in {:.3} s ({:.1} tok/s) · {} generated in {:.2} s ({:.2} tok/s) · 0 tool calls · rewrite 0.000 s · position {}",
            turn + 1,
            prompt,
            prefill_seconds,
            prompt as f64 / prefill_seconds.max(1e-9),
            produced.len(),
            decode_seconds,
            produced.len() as f64 / decode_seconds.max(1e-9),
            engine.pos()
        );
    }
    Ok(())
}
