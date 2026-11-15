# bench/

Everything behind the numbers published with this release: the scripts, the raw results, and the predictions written before measuring. Same tag as the code.

## Machines

The raw results name the hardware, not the machine:

- `epyc-4344p`: AMD EPYC 4344P (Zen 4, 8 cores, AVX-512), Linux, 8 threads.
- `ryzen-9-9900x`: AMD Ryzen 9 9900X (Zen 5, 12 cores, AVX-512), Linux, 12 threads.
- `ryzen-7-5700g`: AMD Ryzen 7 5700G (Zen 3, 8 cores, AVX2 without VNNI), Linux.

**The raw data are raw.** Copying them here changed only the names of the machines (in file names and in the text, replaced by the hardware above), the paths (home directories and model folders written `~/…`, scripts renamed as below), and, in the French notes and comments, private mentions: names of people, of tools and of the lab's private notes (their paths and commit ids; the times stay, and lab folders not published here are given by the date of the measurement). No value was touched. The predictions (`predictions/`) went through the same rules, so the sha256 taken when they were written no longer matches the published files: `predictions/ORIGINAL-SHA256.txt` gives both, and each file has the sha256 of its published version. Nothing else in them changed. The short commit ids that stay in the notes and traces (`68c3a96`, `75e75fc`, `d24603d`, `b68ce37`) name the engine code as it was measured, in the lab's history: they do not resolve in this repository's.

Script names: the copies inside `results/` (prefixed `_`) are the scripts as they ran; `_throughput.py` and `_run_q4_0_pure.sh` (in `results/2026-11-14-q4_0-pure/throughput-ryzen-9-9900x/`) are the published `throughput/throughput.py` and `throughput/run_q4_0_pure.sh` before their paths were made into parameters. Those copies are traces: they point to the folders of the lab, not to this layout, and do not run here as they are, nor do the three traces that kept their names (`tableau.py`, `comparaison.py`, `exemples.py`). The same holds for every script inside the folders of 14 November added below (`_render.py`, `_srv_gen.py`, `_ref_gen.py`, `_chaine_*.sh`, `_q4_bf16_*.sh`, `_analyse.py`, `_executer.sh`…): they are kept as they ran, as traces, with machine names and paths replaced like the rest; they expect the lab's folders, servers and ports, and are not meant to run from this tree.

## Which grid each label is

The default of `--q4` changed during these measurements. Since this release, `--q4` is the grid of `Q4_0` (scale = signed max / −8, ggml's rule), and `--q4 --q4-grid7` the older grid (absmax / 7, codes in ±7).

- `results/2026-11-13-quality/` and `results/2026-11-14-q4_0-pure/` (measured on an earlier revision): `herbert-q4` = **grid 7**, `herbert-q4g8` = **grid 8** (`--q4 --q4-grid8` then).
- `results/2026-11-14-replays-grid8/` (measured on the release candidate, whose engine code is this tag's): `herbert-g8` = `--q4` (grid 8), `herbert-g7` = `--q4 --q4-grid7`.

## Versions compared

- llama.cpp `5676dfbe0`. Its tag v1.0.0 is `b1b770d90d`, 7 commits later: version numbers, CI, removed examples; `git diff 5676dfbe0 b1b770d90d -- src ggml/src common include tools/server` touches only `tools/server/README.md`. The logs print `5676dfbe0`.
- ik_llama.cpp `7aa7e014`.
- References: transformers in bf16, versions 5.19 to 5.22.2 depending on the model; the version of each reference is in the lab logs, not in this tree.

Both were built with `GGML_NATIVE=ON` and `GGML_BLAS=OFF` (their `CMakeCache.txt`), e.g. `cmake -B build -DGGML_NATIVE=ON -DGGML_BLAS=OFF && cmake --build build -j`.

## Throughput (`throughput/`)

`throughput.py URL LABEL OUT.json` sends to a llama-server compatible server a fixed French and English text repeated to `WORDS` words (the sizes, e.g. `"78 380 4240"`: 217, 621 and 6 125 tokens with Qwen3.6), followed by `SUFFIX`, three times per size, 128 generated tokens with `ignore_eos`, and records the server's own `timings` (prompt and generation tokens per second). A short warm-up request comes first. `run_q4_0_pure.sh` runs the six arms one server at a time (see its header); it was written for this release from the scripts that ran and has not been run as such. Of this file's commands, only the `throughput.py` line of the top-level README and `hkld greedy` were run from the release archive.

One line, against a running server, is in the top-level README. Published:

| machine | tokens | herbert `--q4` prefill / decode (tok/s) | source |
|---|---|---|---|
| Ryzen 9 9900X, 12 threads | 217 | 377 / 21.3 | `results/2026-11-14-q4_0-pure/throughput-ryzen-9-9900x/`, replayed in `results/2026-11-14-replays-grid8/B/` (375.9 / 21.29) |
| Ryzen 9 9900X, 12 threads | 621 | 369 / 21.1 | idem |
| Ryzen 9 9900X, 12 threads | 6 125 | 387 / 18.9 | idem |
| EPYC 4344P, 8 threads | 627 | 336 / 25.9 | `results/2026-11-14-q4_0-pure/throughput-epyc-4344p/`, replayed in `.../replays-grid8/A/` |
| EPYC 4344P, 8 threads | 6 131 | 306 / 23.8 | idem |

The other arms (llama.cpp, ik_llama.cpp with and without `-rtr`, Q4_K_M) are in the same folders, measured in the same window; the medians are those of the `res-*.json` (three repetitions).

## Quality (`quality/`)

KL divergence, perplexity and top-1 agreement against transformers bf16, with `llama-perplexity`'s convention (n_ctx 256, second half of each chunk scored, BOS first when the model has one): wikitext-2 raw test, 48 chunks (6 096 positions), and *Les Misérables* vol. I (Project Gutenberg #17489, public domain), 24 chunks (3 048 positions).

Every engine is fed the same tokens and writes the same file: for each scored position, the whole distribution of the next token, in the format of `llama-perplexity --kl-divergence-base` (llama.cpp's compressed log-softmax, one u16 per token of the vocabulary). llama.cpp and ik_llama.cpp write it themselves; `hkld` writes it for herbert and `ref_tf.py` for transformers; one comparator reads them all. Each file of `quality/` describes itself in its header; the chain, in order:

1. `quality/scripts/ids.py`: the corpus tokenized with the model's Hugging Face tokenizer and cut into chunks (`.ids`). `tokcheck.py` checks that llama.cpp's tokenizer gives the same ids, since `llama-perplexity` tokenizes the corpus itself.
2. `quality/scripts/ref_tf.py`: the reference, transformers bf16, in the same format, with the exact log-likelihood of each position beside it (`.kld.nll`, f64).
3. `quality/scripts/sketch.py`: the reference reduced to the 4 096 most probable tokens of each position (`.npz`): the full file is about 3 GB for Qwen3.6 (vocabulary 248 320).
4. `quality/scripts/arms.sh`: the arms of one model, one after the other (herbert through `hkld kld`, `llama-perplexity` of llama.cpp and of ik_llama.cpp, with or without `-rtr`), each compared with the reference by `cmp.py` (full reference) or `cmp2.py` (sketch: the KL of the distribution coarsened to those 4 096 tokens plus the rest, a lower bound of the full KL; the gap was measured as zero on the test).
5. `quality/scripts/table.py`: the table of the campaign (`TABLE.md`) from the `out/*.json` of each machine.

`quality/hkld/` is herbert's side: a member of this workspace (`cargo build --release -p hkld`) that runs the engine through its own load, prefill and decode, the engine unchanged. Per chunk it prefills the first half, then feeds the decode the true next token of the text at each step: the scored positions go through the decode kernels, one at a time, where `llama-perplexity` scores them in one batch. `hkld greedy` writes the greedy ids of given prompts (to compare two revisions or two machines with `diff`), `hkld last` the logits of the last position of a prompt (a diagnostic of the prefill).

```sh
cargo build --release -p hkld
cd WORK    # holds corpus/ (the two texts) and ref/
python3 $REPO/bench/quality/scripts/ids.py MODEL_DIR corpus/wiki.test.raw 256 48 0 248044 ref/qwen36-wiki.ids
python3 $REPO/bench/quality/scripts/ref_tf.py MODEL_DIR ref/qwen36-wiki.ids ref/qwen36-wiki.kld --threads 12
python3 $REPO/bench/quality/scripts/sketch.py ref/qwen36-wiki.kld ref/qwen36-wiki.npz --k 4096   # optional
LLAMA=~/llama.cpp/build/bin $REPO/bench/quality/scripts/arms.sh qwen36 MODEL_DIR wiki 48 \
    herbert:herbert-g8:--q4 llama:llama-q40:qwen3.6-35b-a3b-q4_0.gguf
python3 $REPO/bench/quality/scripts/table.py DIR_OF_MACHINES > TABLE.md
```

`0 248044` are Qwen3.6's `add_bos` and BOS id (no BOS); Gemma 4 takes `1 2`. The `.ids` of the campaign are published (`results/2026-11-13-quality/*/ref/`): with them, step 1 can be skipped and every engine gets the very tokens measured. The reference `.kld` files and their sketches are not published (their size); their `.kld.nll` are.

- `quality/corpus/fr-miserables-t1.txt` (sha256 `e639fee1002b3dac97ec84624c052aa13f942d7f779a06051cdf5ca26684c8c7`). wikitext-2: `wget https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip`, file `wiki.test.raw`.
- Results: `results/2026-11-13-quality/TABLE.md` and the `out/*.json` of each machine (`ref/*.ids` are the tokens fed to every engine), `results/2026-11-14-q4_0-pure/quality-*` for Qwen3.6 on the pure Q4_0 GGUF. At equal format (Q4_0), herbert's KL matches llama.cpp's: Qwen3.6, wikitext / French, herbert grid 8 0.0720 / 0.0759, llama.cpp Q4_0 0.0723 / 0.0767, ik_llama.cpp 0.0736 / 0.0767, herbert grid 7 0.0894 / 0.0984, llama.cpp Q4_K_M 0.0191 / 0.0198.
- `results/2026-11-14-divergence-examples/`: what the divergence changes at given positions (text, reference token, each arm's token).
- Two rows of `TABLE.md`, Gemma 4 E4B under OpenVINO (`ov-int4g32`) and under vLLM (`vllm-bf16`), on the EPYC 4344P, come from three scripts outside `arms.sh`: `quality/scripts/ov_kld.py` (an OpenVINO version of `ref_tf.py`, compared by `cmp.py` or `cmp2.py` like any arm), and `vllm_nll.py` with `cmp_nll.py` (vLLM returns only the log-probability of the true token and its best candidate: Δnll and top-1, no KL in those rows).

## Pure Q4_0 GGUF (`q4_0-pure/`)

`make_q4_0_pure.sh` makes the GGUF llama.cpp and ik_llama.cpp were measured on (`llama-quantize --pure`, no imatrix); `sha256-qwen36-q40pur.txt` is its sha256. `gguf_types_q40.py` lists the type of each tensor (`results/.../types-qwen36.json`: Q4_0 everywhere but the norms, the router, `shared_expert_gate`, conv1d, `ssm_a`, `ssm_dt`), `tensor_sha.py`, `layout.py` and `rebuild_q40.py` compare and realign two converts. Differences with herbert `--q4`: herbert keeps `token_embd` in bf16 and `in_proj_a/b` in f32, uses f32 scales (f16 in Q4_0) and int8 activations per 32 with an f32 scale (Q8_0 in llama.cpp).

## Whole answers, side by side (`results/2026-11-14-answers-side-by-side/`)

What quantization changes in whole answers, at equal format and against Q4_K_M. Qwen3.6-35B-A3B, fifteen short questions in French (`questions.json`), written with `PREDICTIONS.md` before any generation. Five arms on the Ryzen 9 9900X (12 threads), one after the other: transformers 5.22.1 bf16 (the reference), herbert `--q4` (grid 8, engine code of this tag), llama.cpp `5676dfbe0` and ik_llama.cpp `7aa7e014` on the pure Q4_0 GGUF, llama.cpp on Q4_K_M. Chat template with `enable_thinking=False`, greedy decoding, 200 tokens at most. Every arm got the same input ids (checked: `prompt_ok`, `tokenize_matches` in `logs/gen-*.log`).

- `prompts.jsonl` (rendered text and ids); `out/<arm>.jsonl` (generated ids and text; `ik-q40pur.raw.jsonl` holds ik's streamed pieces, from which `_ik_ids.py` rebuilds the ids); `out/ref-tf-bf16-marges.jsonl` (the reference's margin between its first two tokens at each step).
- `analyse.json` (`_analyse.py`: first divergence from the reference, pairs of arms), `jugements.json` (the judgments and the texts of the comparison page), `_table.py` → `_table.out`.
- `logs/`: servers, generations, herbert's requests with their `probs.json`, the reference.
- `NOTES-fr.md`: the lab notes (French): protocol, table, the three examples of the page, predictions and their outcome.

To redo: render the prompts (`_render.py`), run the reference (`_ref_gen.py`), start each server and send the ids (`_srv_gen.py`), then `_analyse.py`. The commands of each step are in `_chaine_ryzen-9-9900x.sh` and `_chaine_ik.sh`.

## A long document, then a question (`results/2026-11-14-long-context/`)

Same arms, machine and scripts, with a whole English text in the prompt and a question on it, asked once in English and once in French. Two short stories by H. G. Wells from *The Stolen Bacillus and Other Incidents* (Project Gutenberg eBook #12750): "The Temptation of Harringay" (2 689 tokens) and "A Moth — Genus Novo" (5 249 tokens), ten prompts each, greedy, 150 tokens at most, context 8 192.

- `source/pg12750.txt`: the eBook as downloaded on 14 November 2026 from `https://www.gutenberg.org/cache/epub/12750/pg12750.txt`, byte for byte, with the Project Gutenberg header and license (sha256 `0f0cb578…75ff3b`); `source/harringay.txt` and `source/moth.txt`: the two stories extracted by `_extract.py` (line ends and paragraphs only). `questions.sha256` covers the three texts, `questions.json`, `_questions.py` and `_extract.py`.
- `questions.json` (`_questions.py`), `ATTENDUES.md` (expected answers) and `PREDICTIONS.md`, each with its sha256, written before any generation.
- `ryzen-9-9900x/`: the working folder of the run: scripts, `prompts.jsonl`, `out/` (ids and text of each arm), `logs/`, `analyse-sans-ref.json` (pairs only, before the reference finished). At the top: `analyse.json` (with the reference), `comptes.json` (`_comptes.py`: times and counts), `jugements.json` (right or wrong, by hand).
- `NOTES-fr.md`: the lab notes (French), with the table, the judgments and the outcome of the predictions. The machine was shared during the runs: its times are not a benchmark.

## Code, side by side (`results/2026-11-14-code-side-by-side/`)

Same arms, ten small tasks in French (seven in Python, three in Rust: write a function, fix a supplied bug, complete a function, tell what a snippet prints), greedy, 500 tokens at most. Scored by execution, not by a judge: the first code block of each answer is run against hidden tests (`tests/`, never shown to the model).

- Preregistration: `taches.json` (tasks, from `_taches.py`), `tests/`, `attendus.json` (expected output of the two "predict" tasks), `_solutions/` (control solutions, which pass every test), `PREDICTIONS.md`, all in `SHA256SUMS`; the generation scripts and scoring rules in `SHA256SUMS-2`. Both sums are recomputed on the published bytes. `_attente.sh` (a wait before taking the machine's lock) was rewritten twice after it was hashed; `predictions/ORIGINAL-SHA256.txt` says so.
- `prompts.jsonl`, `out/<arm>.jsonl`, `analyse.json`, `reponses/<arm>.json` (the scored texts), `exec/<arm>/<task>/` (extracted code, tests, launcher) and `exec/<arm>/<task>.log` (full container output), `resultats.json`.
- Sandbox: every (arm, task) ran on the Ryzen 7 5700G, idle, in a throwaway podman 4.9.3 container: images `python:3.12-slim` and `rust:1.82-slim`, `--network none`, no home mounted (only the task folder, read-only, on `/task`), `--read-only` with tmpfs `/tmp` (64 MB) and `/work` (768 MB), `--memory 1g --memory-swap 1g`, `--pids-limit 128`, `--cpus 2`, `--user 65534:65534`, `--cap-drop ALL`, `--security-opt no-new-privileges`. Python: `py_compile`, import within 10 s, each hidden test within 10 s; Rust: a crate without dependencies, `cargo test --offline --no-run`, then each test alone within 10 s. Probe: `logs/sonde-isolation.log` (uid 65534, empty `/home`, network unreachable, read-only file system, a 2 GB allocation killed). Harness checks: `logs/controle-harnais*`.
- **The files under `exec/`, `out/` and `reponses/` contain code written by the models. Do not run them outside such a sandbox.**
- `NOTES-fr.md`: the lab notes (French), with the results table and the outcome of the predictions.

## Q4 against bf16, same code (`results/2026-11-14-q4-vs-bf16-*/`)

Speed and memory of herbert `--q4` against herbert in bf16, same binary, same bf16 safetensors (the Q4 is made at load), same window, one server at a time, order bf16, Q4, Q4, bf16. `/completion`, `cache_prompt:false`, `ignore_eos`, 128 tokens, 3 repetitions per server, medians of the server's `timings`. Memory from `/proc/PID/status` (`VmHWM` after load and at the end of the requests, samples every 0.5 s in `*/rss-*.log`); load time from process start to the first `/health` answer, files in cache. Bytes read per token: `_octets.py`, from the safetensors headers.

- `2026-11-14-q4-vs-bf16-epyc-4344p/`: EPYC 4344P, 8 threads; Qwen3.5-2B and Gemma 4 E4B (and Qwen3.6-35B-A3B in Q4 alone: its 71.9 GB of bf16 weights do not fit in this machine's 64 GB). `run-epyc-4344p.log` (whole chain), `*/server-*.log`, `*/res-*.json`, `*/rss-*.log`, `octets.txt`, `tableau.md` (per server), `q4-contre-bf16.json` (read by the comparison page), `PREDICTIONS.md`; `NOTES-fr.md` holds the table and the reading.
- `2026-11-14-q4-vs-bf16-ryzen-9-9900x/`: the same protocol on the Ryzen 9 9900X, 12 threads: Qwen3.5-4B, Qwen3.5-2B, Gemma 4 E4B, and Qwen3.6-35B-A3B in both formats (its bf16 fits in this machine's memory: 71.2 GB resident). `PREDICTIONS.md` and its sha256, the scripts, `run-ryzen-9-9900x.log` (and `run-ryzen-9-9900x-essai1-arrete.log`, a first attempt stopped before any arm; `run-ryzen-9-9900x-suite35b.log`, the last two 35B servers), `tableau.md`, `resume.md` and the raw files per model; `NOTES-fr.md` holds the table, the ratios per pair of servers and the reading.
  - **Another job shared the machine.** All servers but one ran while an unrelated FPGA job (simulation, synthesis, place and route) used about one core; `fpga-intervalles.txt` gives its intervals (job names removed) and the tables give the mean FPGA cores during each server. Where bf16 and Q4 of a model did not run under the same load, `NOTES-fr.md` gives the ratios per pair of servers measured under equal load.
  - **The bf16 prefill of Qwen3.6-35B-A3B is not comparable** (8.6 tokens/s against 349 in Q4): in bf16, herbert runs the prompt of this MoE model through the decode program, one token at a time, while the Q4 path has its own prefill program. Its decode (7.20 against 20.77 tokens/s) and memory are comparable; its prefill ratio says nothing about the 4-bit format.

To redo: `_q4_bf16_epyc-4344p.sh` / `_q4_bf16_ryzen-9-9900x.sh` (as they ran) start each server with and without `--q4` and call `throughput.py`; `_tableau.py` makes the table, `_octets.py` the bytes per token.

## The comparison page

`results/2026-11-14-q4_0-pure/comparaison-q4_0-2026-11-14.html` (French), generated by `comparaison.py` from the folders above: the pure Q4_0 table, what the divergence changes word by word, Q4 against bf16 (EPYC 4344P and Ryzen 9 9900X), whole answers side by side, a long document then a question, code.

## Predictions (`predictions/`)

Written before the measurements they predict, in French: `PREDICTIONS-quality-2026-11-13.md`, `PREDICTIONS-q40pur.md`, and, for the campaigns of 14 November, `PREDICTIONS-answers-side-by-side.md`, `PREDICTIONS-long-context.md`, `PREDICTIONS-code-side-by-side.md`, `PREDICTIONS-q4-vs-bf16-epyc-4344p.md` and `PREDICTIONS-q4-vs-bf16-ryzen-9-9900x.md` (copies of the `PREDICTIONS.md` of each results folder, same bytes), each with the sha256 of its published version (`shasum -a 256 -c PREDICTIONS-q40pur.sha256`) and, in `ORIGINAL-SHA256.txt`, the sha256 of the original before the machine names were replaced. `ORIGINAL-SHA256.txt` also lists every other file hashed before a campaign (questions, expected answers, tasks, tests, control solutions, scripts) with its published and original sha256; most are identical. In each results folder, the `*.sha256` and `SHA256SUMS*` files are recomputed on the published bytes: `shasum -a 256 -c` works there. Their outcome: `results/2026-11-14-q4_0-pure/sort-predictions.json` and the comparison page `comparaison-q4_0-2026-11-14.html`.

## Licenses of third-party data

The code of this repository is under its own license (`LICENSE` at the top). The data below are not: they keep their own terms.

- **WikiText-2** (Stephen Merity, Caiming Xiong, James Bradbury, Richard Socher, *Pointer Sentinel Mixture Models*, 2016; built from Wikipedia articles, written by their contributors). The corpus itself is not in this tree: `quality/` gives its download link. Derived from it are the token files `results/2026-11-13-quality/*/ref/*-wiki.ids` (the test set, tokenized); the examples of `results/2026-11-14-divergence-examples/` and of the comparison page quote *Les Misérables* only. The dataset card (https://huggingface.co/datasets/Salesforce/wikitext) tags it `cc-by-sa-3.0` and `gfdl` while its text says CC BY-SA 4.0; Wikipedia text is under CC BY-SA. Those derived files are therefore distributed under **CC BY-SA 3.0** (https://creativecommons.org/licenses/by-sa/3.0/), with the attribution above, not under the license of the code; anything made from them must keep the attribution and the same license.
- ***Les Misérables***, volume I (Victor Hugo, 1862; public domain), from Project Gutenberg eBook #17489 (https://www.gutenberg.org/ebooks/17489): `quality/corpus/fr-miserables-t1.txt` is the text alone, without the Project Gutenberg header, license or name, so Project Gutenberg's license does not attach to it (section 1.C of https://www.gutenberg.org/policy/license.html); it is cited here as the source.
- ***The Stolen Bacillus and Other Incidents*** (H. G. Wells, 1895; public domain), Project Gutenberg eBook #12750 (https://www.gutenberg.org/ebooks/12750). `results/2026-11-14-long-context/source/pg12750.txt` is the eBook byte for byte, **with its Project Gutenberg header and full license**, as section 1.C of that license allows when the work is shared without charge in the same format with the license attached. Whoever redistributes it must keep it whole (or remove every reference to Project Gutenberg, section 1.C) and must not charge for it without the agreement of section 1.E.8 or 1.E.9. The two extracted stories (`source/harringay.txt`, `source/moth.txt`) and the prompts built from them (`questions.json`, `prompts.jsonl`) carry no Project Gutenberg name or license; the source is cited here.
- **Model outputs.** Generated text is published from Qwen3.6-35B-A3B (answers, long context, code), and generated tokens or throughput runs from Qwen3.5-2B, Qwen3.5-4B and Gemma 4 (E4B, 26B). Qwen3.6 and Qwen3.5 are under the Apache License 2.0 (model cards read: https://huggingface.co/Qwen/Qwen3.6-35B-A3B, https://huggingface.co/Qwen/Qwen3.5-2B, https://huggingface.co/Qwen/Qwen3.5-4B). Gemma 4 is also under the Apache License 2.0 (model cards https://huggingface.co/google/gemma-4-E4B-it and https://huggingface.co/google/gemma-4-26B-A4B-it, https://ai.google.dev/gemma/docs/gemma_4_license, text at https://ai.google.dev/gemma/apache_2; the Gemma Terms of Use at https://ai.google.dev/gemma/terms apply to earlier Gemma models and send Gemma 4 to the Apache license). The Apache License 2.0 sets no condition on outputs. For the LFM models (LFM Open License v1.0, read at https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/main/LICENSE, which does not mention outputs either), this tree holds no generated text: only KL divergences, log-likelihoods and the corpora tokenized by their tokenizers. The models themselves are not redistributed.

## Not here

Multiple-choice and tool-calling benchmarks were run but not scored: no number from them is published. No KL on AVX2 or NEON yet.
