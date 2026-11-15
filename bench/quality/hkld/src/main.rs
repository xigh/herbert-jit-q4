//! hkld: the measurement tool of `bench/quality/`. It runs herbert's engine through its own load,
//! prefill and decode (no line of the engine changed) on tokens given in advance, and writes what
//! the other engines of the comparison write: the log-probabilities of every scored position in
//! the exact format of `llama-perplexity --kl-divergence-base`, generated ids, or raw logits.
//!
//! ```text
//!   hkld kld    MODEL CORPUS.ids  OUT.kld [flags] [--chunks N]
//!   hkld greedy MODEL PROMPTS.txt OUT.txt [flags] [--n 256]
//!   hkld last   MODEL PROMPTS.txt OUT.bin [flags]
//!
//!   flags:
//!   --threads T          workers (default 8)
//!   --q4                 4-bit weights, the grid of Q4_0 (signed max / -8), int8 activations
//!   --q4-grid7           with --q4: the older grid, absmax / 7
//!   --q4-grid8           with --q4: the grid of Q4_0, the default (accepted for older scripts)
//!   --int8               int8 weights and activations
//!   --prefill-executor spmd-spin|spmd-hybrid  the barriers of the prefill (default: the engine's)
//!   (none of --q4 / --int8: the weights stay bf16; every other option of the engine is its
//!   default)
//! ```
//!
//! MODEL is a model folder (`config.json`, tokenizer, safetensors) or, as for `herbert-jit-chat`
//! and the server, `org/name[@revision]` already in the Hugging Face cache (`hf download`).
//!
//! # `hkld kld`: the log-probabilities, as llama-perplexity writes them
//!
//! CORPUS.ids is written by `scripts/ids.py`: `"_ids____"`, then four i32 (`n_ctx`, `n_chunk`,
//! `add_bos`, `bos`), then `n_ctx · n_chunk` i32 tokens. The text is cut into chunks of `n_ctx`
//! tokens and each chunk is read from an empty context, as llama-perplexity does: the first token
//! of a chunk is replaced by `bos` when `add_bos`, the first half of the chunk is context only, and
//! the positions `first = n_ctx/2 .. n_ctx-2` are scored (row `j` predicts token `j+1`). Only the
//! first N chunks are read under `--chunks N`.
//!
//! Per chunk: a prefill of tokens `0..=first` gives the row of position `first`; then the decode
//! (`Engine::generate`) is fed the true next token of the text at each step, not the token the
//! model chose, and gives the other rows. So the scored rows go through the decode kernels, one
//! position at a time, as an answer is written; llama-perplexity scores the same positions in one
//! batch. The logits are the engine's (bf16), widened to f32.
//!
//! OUT.kld: `"_logits_"`, three i32 (`n_ctx`, the vocabulary size, `n_chunk`), the tokens, then per
//! scored row llama.cpp's compressed log-softmax (`compress`, the arithmetic of `log_softmax` in
//! `perplexity.cpp`): the f32 scale and the f32 minimum log-probability as four u16, then one u16
//! per token of the vocabulary, the log-probability between the floor and the maximum (a logit more
//! than 16 below the maximum is stored as the floor, 0), padded to an even count. `OUT.kld.nll`
//! beside it: the exact negative log-likelihood of each scored row, in f64 (the compressed file
//! loses precision; `scripts/cmp.py` takes this one when it exists). On stderr, the perplexity
//! after each chunk.
//!
//! # `hkld greedy`: the ids herbert writes
//!
//! PROMPTS.txt: one prompt per line, its token ids separated by spaces (no template is applied).
//! Per prompt, from an empty context: a prefill, then the greedy decode (the argmax at each step)
//! to `--n` tokens in all, the end-of-sequence tokens ignored. OUT.txt: one line of generated ids
//! per prompt. Two runs, two revisions or two machines compare with `diff`.
//!
//! # `hkld last`: the logits of the last position
//!
//! Per prompt (PROMPTS.txt as above), from an empty context: a prefill, then the logits of its last
//! position. OUT.bin: per prompt, the vocabulary size (u32) then the logits (f32). On stderr: the
//! argmax, the number of non-finite logits and the five best tokens with their logits. A
//! diagnostic: the prefill alone, against another engine's logits at the same position.
use herbert_jit_q4::engine::{Engine, Measure, Tuning};
use std::io::{BufRead, Read, Write};

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn tuning_from(args: &[String]) -> Tuning {
    let flag = |k: &str| args.iter().any(|a| a == k);
    let mut t = Tuning::default();
    if flag("--int8") {
        t.kernels.neon.w_int8 = true;
    }
    if flag("--q4") {
        (t.kernels.neon.w_int8, t.kernels.neon.w_q4) = (true, true);
    }
    if flag("--q4-grid8") {
        t.kernels.neon.q4_grid8 = true;
    }
    if flag("--q4-grid7") {
        t.kernels.neon.q4_grid8 = false;
    }
    if let Some(v) = val(args, "--prefill-executor") {
        t.prefill_executor = v
            .parse()
            .unwrap_or_else(|e| panic!("--prefill-executor: {e}"));
    }
    t
}

fn val(args: &[String], k: &str) -> Option<String> {
    args.iter()
        .position(|a| a == k)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// llama.cpp's `log_softmax` (perplexity.cpp): same arithmetic, same compression.
fn compress(logits: &[f32], out: &mut Vec<u16>) {
    let n = logits.len();
    let mut max_logit = logits[0];
    let mut min_logit = logits[0];
    for &x in &logits[1..] {
        max_logit = max_logit.max(x);
        min_logit = min_logit.min(x);
    }
    min_logit = min_logit.max(max_logit - 16.0);
    let mut sum_exp = 0f64;
    for &x in logits {
        sum_exp += (x - max_logit).exp() as f64;
    }
    let log_sum_exp = sum_exp.ln() as f32;
    let min_log_prob = min_logit - max_logit - log_sum_exp;
    let scale = (max_logit - min_logit) / 65535.0f32;
    let s = scale.to_bits();
    let m = min_log_prob.to_bits();
    out.extend_from_slice(&[
        (s & 0xffff) as u16,
        (s >> 16) as u16,
        (m & 0xffff) as u16,
        (m >> 16) as u16,
    ]);
    if scale != 0.0 {
        let inv = 1.0 / scale;
        for &x in logits {
            out.push(if x > min_logit {
                (inv * (x - min_logit)).round_ties_even() as u16
            } else {
                0
            });
        }
    } else {
        out.extend(std::iter::repeat_n(0u16, n));
    }
    if n % 2 == 1 {
        out.push(0);
    }
}

/// `hkld kld` (the header): llama-perplexity's KL base file and the exact nll, row by row.
fn kld(args: &[String]) {
    let model = &model_dir(&args[0]);
    let mut b = Vec::new();
    std::fs::File::open(&args[1])
        .unwrap()
        .read_to_end(&mut b)
        .unwrap();
    assert_eq!(&b[0..8], b"_ids____");
    let i32_at = |o: usize| i32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let (n_ctx, mut n_chunk, add_bos, bos) = (
        i32_at(8) as usize,
        i32_at(12) as usize,
        i32_at(16) != 0,
        i32_at(20) as u32,
    );
    let ids: Vec<u32> = (0..n_ctx * n_chunk)
        .map(|i| i32_at(24 + 4 * i) as u32)
        .collect();
    if let Some(c) = val(args, "--chunks") {
        n_chunk = n_chunk.min(c.parse().unwrap());
    }
    let threads: usize = val(args, "--threads").map_or(8, |v| v.parse().unwrap());
    let tuning = tuning_from(args);
    let mut engine =
        Engine::load(model, threads, n_ctx + 16, &tuning, &Measure::none()).expect("engine");
    let vocab = engine.config().vocab_size;
    let first = n_ctx / 2;
    let rows = n_ctx - 1 - first;
    let mut f = std::io::BufWriter::new(std::fs::File::create(&args[2]).unwrap());
    f.write_all(b"_logits_").unwrap();
    for v in [n_ctx as i32, vocab as i32, n_chunk as i32] {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
    for &t in &ids[..n_ctx * n_chunk] {
        f.write_all(&(t as i32).to_le_bytes()).unwrap();
    }
    let mut fnll =
        std::io::BufWriter::new(std::fs::File::create(format!("{}.nll", args[2])).unwrap());
    let t0 = std::time::Instant::now();
    let (mut sum, mut cnt) = (0f64, 0usize);
    let mut nlls: Vec<f64> = Vec::new();
    for c in 0..n_chunk {
        let mut tok: Vec<u32> = ids[c * n_ctx..(c + 1) * n_ctx].to_vec();
        let truth = tok.clone();
        if add_bos {
            tok[0] = bos;
        }
        engine.reset();
        engine.prefill(&tok[..=first]).expect("prefill");
        let mut buf: Vec<u16> = Vec::with_capacity(rows * (vocab + 6));
        let mut lp = |logits: &[f32], pos: usize, buf: &mut Vec<u16>| {
            let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            let lse = logits
                .iter()
                .map(|&x| (x as f64 - m).exp())
                .sum::<f64>()
                .ln()
                + m;
            let nll = lse - logits[truth[pos + 1] as usize] as f64;
            assert!(nll.is_finite(), "non-finite nll at chunk {c} pos {pos}");
            sum += nll;
            cnt += 1;
            nlls.push(nll);
            compress(logits, buf);
        };
        let row0: Vec<f32> = engine.logits().iter().map(|&x| bf16_to_f32(x)).collect();
        lp(&row0, first, &mut buf);
        let steps = rows - 1;
        let mut k = 0usize;
        let n = engine
            .generate(tok[first + 1], steps, &mut |_argmax, logits| {
                let row: Vec<f32> = logits[..vocab].iter().map(|&x| bf16_to_f32(x)).collect();
                lp(&row, first + 1 + k, &mut buf);
                k += 1;
                Ok(if k < steps {
                    Some(tok[first + 1 + k])
                } else {
                    None
                })
            })
            .expect("generate");
        assert_eq!(k, steps, "generate ran {n} blocks, {k} rows");
        for v in &buf {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        for v in nlls.drain(..) {
            fnll.write_all(&v.to_le_bytes()).unwrap();
        }
        eprintln!(
            "[kld] chunk {}/{} rows {} ppl {:.4} ({:.1} s)",
            c + 1,
            n_chunk,
            cnt,
            (sum / cnt as f64).exp(),
            t0.elapsed().as_secs_f64()
        );
    }
    f.flush().unwrap();
    fnll.flush().unwrap();
    eprintln!(
        "[kld] done: {n_chunk} chunks, n_ctx {n_ctx}, {cnt} rows, mean nll {:.6}, ppl {:.4}",
        sum / cnt as f64,
        (sum / cnt as f64).exp()
    );
}

/// `hkld greedy` (the header): `--n` greedy ids per prompt, end-of-sequence ignored.
fn greedy(args: &[String]) {
    let model = &model_dir(&args[0]);
    let prompts: Vec<Vec<u32>> = std::io::BufReader::new(std::fs::File::open(&args[1]).unwrap())
        .lines()
        .map(|l| {
            l.unwrap()
                .split_whitespace()
                .map(|x| x.parse().unwrap())
                .collect()
        })
        .filter(|v: &Vec<u32>| !v.is_empty())
        .collect();
    let threads: usize = val(args, "--threads").map_or(8, |v| v.parse().unwrap());
    let n: usize = val(args, "--n").map_or(256, |v| v.parse().unwrap());
    let tuning = tuning_from(args);
    let maxlen = prompts.iter().map(|p| p.len()).max().unwrap();
    let mut engine = Engine::load(
        model,
        threads,
        (maxlen + n + 16).next_multiple_of(16),
        &tuning,
        &Measure::none(),
    )
    .expect("engine");
    let mut f = std::fs::File::create(&args[2]).unwrap();
    for (i, p) in prompts.iter().enumerate() {
        engine.reset();
        let t0 = engine.prefill(p).expect("prefill");
        let mut out = vec![t0];
        if n > 1 {
            engine
                .generate(t0, n - 1, &mut |argmax, _l| {
                    out.push(argmax);
                    Ok(if out.len() < n { Some(argmax) } else { None })
                })
                .expect("generate");
        }
        writeln!(
            f,
            "{}",
            out.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        )
        .unwrap();
        eprintln!(
            "[greedy] prompt {}/{}: {} prompt tokens, {} generated",
            i + 1,
            prompts.len(),
            p.len(),
            out.len()
        );
    }
}

/// `hkld last MODEL PROMPTS.txt OUT.bin [flags]`: per prompt, prefill then the last row's logits
/// (f32, a u32 vocab size before each row); on stderr: argmax, top 5, non-finite values.
fn last(args: &[String]) {
    let model = &model_dir(&args[0]);
    let prompts: Vec<Vec<u32>> = std::io::BufReader::new(std::fs::File::open(&args[1]).unwrap())
        .lines()
        .map(|l| {
            l.unwrap()
                .split_whitespace()
                .map(|x| x.parse().unwrap())
                .collect()
        })
        .filter(|v: &Vec<u32>| !v.is_empty())
        .collect();
    let threads: usize = val(args, "--threads").map_or(8, |v| v.parse().unwrap());
    let tuning = tuning_from(args);
    let maxlen = prompts.iter().map(|p| p.len()).max().unwrap();
    let mut engine = Engine::load(
        model,
        threads,
        (maxlen + 16).next_multiple_of(16),
        &tuning,
        &Measure::none(),
    )
    .expect("engine");
    let vocab = engine.config().vocab_size;
    let mut f = std::io::BufWriter::new(std::fs::File::create(&args[2]).unwrap());
    for p in &prompts {
        engine.reset();
        let am = engine.prefill(p).expect("prefill");
        let row: Vec<f32> = engine.logits().iter().map(|&x| bf16_to_f32(x)).collect();
        f.write_all(&(vocab as u32).to_le_bytes()).unwrap();
        for v in &row {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        let bad = row.iter().filter(|x| !x.is_finite()).count();
        let mut idx: Vec<usize> = (0..row.len()).collect();
        idx.sort_by(|&a, &b| {
            row[b]
                .partial_cmp(&row[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top: Vec<String> = idx[..5]
            .iter()
            .map(|&i| format!("{}:{:.3}", i, row[i]))
            .collect();
        eprintln!(
            "[last] {} tokens: argmax {} non-finite {} top5 {}",
            p.len(),
            am,
            bad,
            top.join(" ")
        );
    }
}

/// MODEL as the two binaries read it: a folder, or `org/name[@rev]` in the Hugging Face cache.
fn model_dir(arg: &str) -> std::path::PathBuf {
    herbert_jit_q4::hub::model_dir(arg).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2)
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("kld") => kld(&args[1..]),
        Some("greedy") => greedy(&args[1..]),
        Some("last") => last(&args[1..]),
        _ => eprintln!(
            "usage: hkld kld MODEL CORPUS.ids OUT.kld [flags] [--chunks N] | hkld greedy MODEL \
             PROMPTS.txt OUT.txt [flags] [--n 256] | hkld last MODEL PROMPTS.txt OUT.bin [flags]\n\
             flags: --threads T, --q4 [--q4-grid7], --int8, --prefill-executor E (the header of \
             bench/quality/hkld/src/main.rs)"
        ),
    }
}
