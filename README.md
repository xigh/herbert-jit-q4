# herbert-jit-q4

LLM inference on the CPU. The weights are quantized to 4 bits (`Q4_0`) when the model loads, from the original Hugging Face safetensors; every compute kernel is emitted at run time (x86-64 with AVX-512 or AVX2, AArch64 with NEON, RISC-V with RVV). The server speaks llama-server's HTTP API, which includes OpenAI's `/v1/chat/completions`.

This repository holds everything needed to build it and to reproduce the published numbers:

- the engine (the crate at the root: `src/`, `crates/a64` and `crates/rv` for the AArch64 and RISC-V emitters);
- `herbert-chat/`: chat templates, answer parsing (content, thinking, tool calls), sampler, detokenization;
- `herbert-serve/`: the HTTP server;
- `bench/`: the benchmark scripts, the raw results and the predictions written before measuring (`bench/README.md`).

One Cargo workspace, one `Cargo.lock`. It builds alone after a clone, with Rust stable and the system's linker.

## Quick start

```sh
git clone https://github.com/xigh/herbert-jit-q4 && cd herbert-jit-q4
cargo build --release --locked --features serve
```

The binaries are in `target/release/`: `herbert-jit-llama-serve` (the server), `herbert-jit-chat` (a command-line chat) and `herbert-jit-stream` (x86-64 Linux: the memory bandwidth, STREAM's kernels emitted). None downloads anything (no TLS in the build), and no C code is compiled. Rust still links through the system's toolchain, which rustup does not install: on a bare Linux, `cargo build` stops on ``linker `cc` not found`` until it is there (`build-essential` on Debian and Ubuntu, `gcc` elsewhere); on macOS the command line tools (`xcode-select --install`); on Windows, Visual Studio's Build Tools ("Desktop development with C++").

Download a model with Hugging Face's own tool, `hf` (the `huggingface_hub` package), into its standard cache (`~/.cache/huggingface/hub`, or `$HF_HOME/hub`, or `$HF_HUB_CACHE`). Install it in a Python environment of its own, either with Python's `venv`:

```sh
python3 -m venv ~/.venvs/hf
~/.venvs/hf/bin/pip install -U huggingface_hub
export PATH="$HOME/.venvs/hf/bin:$PATH"    # Windows: %USERPROFILE%\.venvs\hf\Scripts
```

or with [uv](https://docs.astral.sh/uv/), which also fetches a recent Python when the machine's is too old (`huggingface_hub` 2.x needs Python 3.10 or later):

```sh
uv tool install huggingface_hub    # puts `hf` in ~/.local/bin
```

Then download only the files the engine reads:

```sh
hf download Qwen/Qwen3.5-0.8B --include "*.json" --include "*.safetensors" --include "*.jinja" \
    --include "*.model" --include "merges.txt" --include "vocab.txt"
```

One pattern per `--include`: in `--include "*.json" "*.safetensors"`, `hf` takes the second pattern for a file name and then ignores the filter, so `config.json` and the tokenizer are not downloaded. The patterns leave out what the engine does not read (GGUF, `.bin`, ONNX weights) when the repository has it. Gated models (Gemma) need a token first: `hf auth login`. The binaries then take `org/name` (or `org/name@revision`) and find it in the cache, or a local model folder.

To free disk space, clean the cache with the same tool, one model at a time, never by deleting the cache folder: it is shared with every other program that uses Hugging Face (transformers, other engines), so its other models are not herbert's to remove.

```sh
hf cache ls --sort size                     # what is there, biggest first
hf cache ls --revisions                     # each downloaded version of each model, with its hash
hf cache rm model/Qwen/Qwen3.5-0.8B --dry-run   # what would go, without removing anything
hf cache rm model/Qwen/Qwen3.5-0.8B         # the whole model, after a confirmation
hf cache rm <revision hash>                 # one version only, the others stay
hf cache prune --dry-run                    # older versions no branch points to any more
hf cache prune
```

A service installed with `herbert-jit-llama-serve service install` keeps the absolute path of the snapshot it was given: remove that model, or that revision, only after `service uninstall`, or the service will no longer start.

Ask a question from the command line:

```sh
target/release/herbert-jit-chat Qwen/Qwen3.5-0.8B --q4 --no-think --max-tokens 64 --ask "What is the capital of France?"
```

Run the server (it listens on `127.0.0.1:8000` by default; `--host 0.0.0.0` opens it to the network):

```sh
target/release/herbert-jit-llama-serve Qwen/Qwen3.5-0.8B --q4
```

and ask it, from another terminal:

```sh
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false}}'
```

Routes: `/v1/chat/completions` (streamed or not, tools, thinking), `/completion` and `/v1/completions`, `/v1/embeddings`, `/v1/rerank`, `/v1/models`, `/tokenize`, `/apply-template`, `/props`, `/health`, `/models/load`, `/models/unload`.

Useful options: `--q4` (4-bit weights; without it the weights stay bf16), `--threads N` (default: the physical cores, or the performance cores), `--ctx N` (context, default 32768), `--port N`, `--alias NAME`. The header of `src/bin/herbert-jit-llama-serve.rs` lists them all.

## Run it as a service

The server installs itself as a service of the system, started with the session (Linux, macOS) or the machine (Windows), restarted when it fails. `install` takes the server's own command line; a Hugging Face model must already be in the cache (`hf download` above), and the service gets the absolute path of its snapshot and of the binary.

```sh
target/release/herbert-jit-llama-serve service install Qwen/Qwen3.5-0.8B --q4
target/release/herbert-jit-llama-serve service status
target/release/herbert-jit-llama-serve service stop
target/release/herbert-jit-llama-serve service start
target/release/herbert-jit-llama-serve service uninstall
```

`--name NAME` (right after the action) names the service, `herbert-jit-q4` by default; one service per name.

- **Linux**: a systemd user unit, `~/.config/systemd/user/herbert-jit-q4.service`; its output goes to the journal (`journalctl --user -u herbert-jit-q4`). A user unit stops at the last logout unless `loginctl enable-linger $USER`. A unit to copy by hand is in `packaging/systemd/`.
- **macOS**: a LaunchAgent, `~/Library/LaunchAgents/herbert-jit-q4.plist`; its output in `~/Library/Logs/herbert-jit-q4.log`. A plist to copy by hand is in `packaging/launchd/`.
- **Windows**: a service of the service manager (automatic start, LocalSystem), from an elevated prompt; its output in `%ProgramData%\herbert-jit-q4\herbert-jit-q4.log`.

## Supported models

From the safetensors of Hugging Face (`config.json`, tokenizer, weights), in bf16 or with `--q4`:

- **Gemma 4**: E2B, E4B, 12B, 26B-A4B (mixture of experts), including the QAT `-unquantized` checkpoints; images (vision tower) and, for E2B and E4B, audio (`input_audio`).
- **Qwen3.5 and Qwen3.6** (`qwen3_5`, `qwen3_5_moe`: 0.8B to 9B, 35B-A3B), with their vision tower; **Qwen3-VL** mixture of experts (`qwen3_vl_moe`); Qwen3, Qwen3 MoE and Qwen2.5 text models.
- **LiquidAI LFM2 and LFM2.5** (`lfm2`, `lfm2_moe`: LFM2-8B-A1B, LFM2-24B-A2B), **LFM3** (`lfm3`, LFM3-1.7B) and LFM3-VL (images).
- **Encoders**, on `/v1/embeddings` and `/v1/rerank`: EmbeddingGemma 2 (text and images), BAAI/bge-m3, BAAI/bge-reranker-v2-m3 (XLM-RoBERTa), Qwen3 Embedding and Reranker.

A mixture of experts prefills its rows grouped by expert (each expert one GEMM over its own rows; `--moe-row-prefill` goes one row at a time). Gemma 4 26B-A4B on the Ryzen 9 9900X, 12 threads, `--q4`: 436 prompt tokens in 1.14 s instead of 20.9 s row by row, the same 54 greedy tokens (a lab measurement whose logs are not in `bench/`).

## Quantization

`--q4` quantizes the projections at load in groups of 32 weights with an f32 scale, activations in int8. The default grid is `Q4_0`'s (scale = signed max / −8, codes in [−8, 7], ggml's rule): at equal weights it gives llama.cpp's KL divergence (`bench/`). `--q4 --q4-grid7` keeps the older grid (scale = absmax / 7, codes in ±7), which has a higher KL; `--q4-grid8` is accepted and changes nothing.

## Reproduce one line of the benchmark

The release compares herbert with llama.cpp `5676dfbe0` (v1.0.0 is `b1b770d90d`, 7 commits later without code changes) and ik_llama.cpp `7aa7e014` on Qwen3.6-35B-A3B in pure `Q4_0` (`bench/q4_0-pure/` makes the GGUF). One line: the Ryzen 9 9900X, 12 threads, a 217-token prompt, published at 377 tok/s prefill and 21.3 tok/s decode. Start the server on the model (`Qwen/Qwen3.6-35B-A3B` once `hf download` has put it in the cache, about 70 GB, or its local folder):

```sh
target/release/herbert-jit-llama-serve Qwen/Qwen3.6-35B-A3B --q4 --threads 12 --ctx 8192
```

then, once it answers `/health`:

```sh
WORDS=78 SUFFIX='' python3 bench/throughput/throughput.py http://127.0.0.1:8000 herbert-g8 res.json
```

It prints three repetitions with `prompt_per_second` (prefill) and `predicted_per_second` (decode). `bench/README.md` gives the whole protocol (six arms, one server at a time, nothing else running), the quality measurements (KL against transformers bf16) and the raw results.

## Environment variables

None is needed. `HERBERT_THREADS=N` sets the default worker count (`--threads` wins). The others serve measurement and comparison with a reference; a value of `0` or an empty one turns a switch off.

- Which kernel runs, on stderr: `HERBERT_JIT_PLAN=1` (every kernel compiled, and why a shape was declined), `HERBERT_ATTN_PLAN=1` (the attention of each prefill block), `HERBERT_VISION_PLAN=1` and `HERBERT_ENCODER_PLAN=1` (the attention of each layer of a vision tower or an encoder, at mount).
- Times: `HERBERT_PREFILL_PROFILE=1` (the prefill program's nodes, summed by operation), `HERBERT_VISION_PROFILE=1`, `HERBERT_AUDIO_PROFILE=1`, `HERBERT_ENCODER_PROFILE=1` (the same for the towers and encoders, x86-64), `HERBERT_VISION_TIMES`, `HERBERT_AUDIO_TIMES`, `HERBERT_ENCODER_TIMES` (one line per input); `HERBERT_PERF_MAP=1` writes `/tmp/perf-PID.map` so that `perf report` names the generated code.
- The control arm of a comparison: `HERBERT_NO_AVX512=1` (the AVX2 kernels on an AVX-512 CPU), `HERBERT_NO_BF16=1` (the widening kernels on an AArch64 CPU with BF16), `HERBERT_Q4_YMM=1` (the `ymm` Q4 kernel on an AVX-512 CPU), `HERBERT_Q4_AVX2_I32=1` (the AVX2 Q4 kernel's three-instruction step), `HERBERT_Q4_ROWS=N` (rows of a Q4 tile), `HERBERT_SDPA_SV_TILE=N`, `HERBERT_VISION_SDPA_FORM=fused|split|chained`, `HERBERT_VISION_SDPA_QK=N`, `HERBERT_VISION_ATTN_ROWS=N`, `HERBERT_AUDIO_ATTN_ROWS=N`, `HERBERT_ENCODER_ATTN_ROWS=N` (0: the row kernel), `HERBERT_VISION_ATTN_CYCLIC=0`, `HERBERT_ENCODER_AMX=0`, `HERBERT_ENCODER_ALIGN=N`, `HERBERT_ENCODER_BATCH=N`, `HERBERT_ENCODER_PASS=N`, `HERBERT_ENCODER_TEAMS=0`, `HERBERT_HYBRID_WAIT=condvar|yield|ulock` (how an idle worker waits; `ulock` is a private macOS call).
- Against a reference, layer by layer: `HERBERT_VISION_PATCHES=FILE`, `HERBERT_VISION_PIXELS=FILE` (the tower's input read from a file), `HERBERT_VISION_SOFT=FILE`, `HERBERT_VISION_DEEPSTACK=FILE` (the tower's output read from a file, to test the language model alone), `HERBERT_VISION_DEPTH=N` (the first `N` layers of the tower), `HERBERT_JIT_FEATS=FILE`, `HERBERT_JIT_DEEPSTACK=FILE` (the tower's output written to a file, f32).
- XLM-R encoders (bge-m3): `HERBERT_XLMR_Q4_IMP=0|1|FILE` (the importance of each input column, which weights the choice of their Q4 scales: none, `MODEL/herbert/xlmr-q4-importance.safetensors`, or another file), `HERBERT_XLMR_Q4_IMP_STATS=1`.
- Tests: the speed measurements among the tests are `#[ignore]`d and run by hand (`q4_rows_speed`, `window_attn_speed`, `local_attention_timing`, `sdpa_bench`, `bench_sse42_vs_vnni_q4`, `q4_zmm_speed`, `q4_ymm_dot_speed`: `cargo test --release --lib NAME -- --ignored --nocapture`, the header of each says what it measures); `HERBERT_MODELS=DIR`, where the `#[ignore]`d tests of `herbert-chat` find the tokenizers and templates they need (`cargo test -p herbert-chat -- --ignored`).

## Known limits

- Accuracy (KL against transformers bf16) is measured on AVX-512 only. On AVX2, only the perplexity of Qwen3.6 (same as AVX-512 within 0.2 %, older grid); no KL on NEON yet.
- At equal format (`Q4_0`) the KL of herbert matches llama.cpp's; against Q4_K_M, which keeps some tensors in 6 or 8 bits, it is 1.4 to 3.8 times larger (LFM, Qwen).
- The vision of Gemma 4 26B-A4B and of Qwen3.6 is refused on AVX2 without AVX-512 ("N is not a multiple of 16", measured on Zen 3); not measured on AArch64. On AVX2, the large image of LFM3-VL is slow.
- The vision tower of Gemma 4 26B-A4B against transformers: median cosine 0.998, but isolated tokens fall to 0.22–0.40; the cause is not established.
- Images are decoded by the `image` crate (JPEG, PNG, WebP); its JPEG decoding differs slightly from the reference's.
- On AVX2, Qwen3.6 can end its answer at the first token on a long prompt (the end-of-sequence token wins by a small margin); `ignore_eos` is honored.
- Gemma 4 E2B prefills slower than llama.cpp; on macOS the decode drops at long context.
- Gemma 4 31B: its configuration is read and its template checked, but it has not been run on this engine.
- On Qwen3.6 in `Q4_0`, ik_llama.cpp decodes 2.7 to 8.7 % faster, and prefills long prompts faster: on the Ryzen 9 9900X at 6 125 tokens, +17 % without `-rtr` and +28 % with it; herbert is ahead on short prompts (`bench/results/2026-11-14-replays-grid8/`, medians of three).
- Sampling seeds can draw the same sample twice.
- Gemma 4 26B-A4B in bf16 is at a KL of 0.10 from transformers bf16, against a bf16/f32 floor of 0.08; herbert against f32 is not measured.
- A "thought" marker after a tool call was seen once on another engine that shares `herbert-chat`; not checked on this one.
- macOS: a LaunchAgent cannot read a model on an external volume (it stays stuck at load, most likely the privacy permission of launchd on removable volumes); keep the model on the internal disk (`HF_HOME`).
- The Windows service was tested on Windows 11 x86-64 only.

## License

MIT (`LICENSE`). `third_party/arm-optimized-routines` (the algorithm, constants and tables of the vector `exp` and `log` the three emitters write, `src/jit/execution/math_data.rs`) keeps its own license, MIT OR Apache-2.0 WITH LLVM-exception, used under MIT: see its `LICENSE` and `SOURCE.json`.
