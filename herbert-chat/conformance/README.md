# Template conformance bench

`herbert-chat`'s rendering (minijinja) against Python's jinja2 as `transformers` runs it in `apply_chat_template` (`render_jinja_template`): every template of the catalog on every conversation of `cases.json`, thinking on then off, compared byte for byte. A template refusal (`raise_exception`) counts as agreement when both sides refuse.

- `templates/`: the distinct templates of the catalog (`chat_template.jinja` of the Hugging Face repositories, or the `chat_template` of their `tokenizer_config.json`): Gemma 4 and Qwen3.5 (six templates), Qwen2.5, Granite 3.1 (experts), 4.1 and 4.0 H, Ministral 3 Instruct and Reasoning, LFM2 (8B-A1B), LFM2.5 (1.2B Instruct, 24B-A2B), LFM3 (1.7B), Chocolatine 2 (4B, a Qwen template of its own) and SmolLM3 (3B): eighteen. The models that share a template are listed in `cases.json`.
- `cases.json`: the conversations (system, empty system, several turns, reasoning kept in the history or as `<think>` in the content, Unicode, content in parts, tools with nested schemas, call then result, arguments as a JSON string, three calls in one message, final answer without a generation prompt, expected refusals).
- `reference.py`: the reference render (Python with `transformers` and `jinja2`).
- `examples/conformance.rs`: `herbert-chat`'s render.
- `compare.py`: the comparison, per template, and for each difference the byte where it starts with the text around it.
- `run.sh PYTHON [OUT_DIR]`: the three in a row, outputs in `OUT_DIR` (default `out/`).

Tool-call arguments given as a JSON string are parsed into an object before the template, on both sides: `herbert-chat` does it for every request, as llama-server does.

Rerun it for every updated template, every minijinja version, every added case.

## Results

- 2026-11-04, Ryzen 7 5700G (Linux), `transformers` 5.20.1, `jinja2` 3.1.6: 228 of 228 renders identical (6 templates × 19 conversations × thinking on or off), including the Qwen templates' 8 refusals of a conversation with no user message.
- 2026-11-08, Ryzen 9 9900X (Linux), `transformers` 5.14.1, `jinja2` 3.1.6, under `LC_ALL=C.UTF-8` (Granite writes today's date in English): 608 of 608 identical (16 templates × 19 conversations × thinking on or off). Two `herbert-chat` fixes got there: a Python-style `tojson` for Granite (the tools of Granite 4.0 H and 4.1) and its `indent` argument (Granite 3.1 writes `tools | tojson(indent=4)`). The LFM templates carry transformers' `{% generation %}` tag, which `herbert-chat` strips before compiling them (it writes nothing).
- 2026-11-15, Apple M5 (macOS), `transformers` 5.16.1, `jinja2` 3.1.6, under `LC_ALL=C.UTF-8`: 684 of 684 identical (18 templates × 19 conversations × thinking on or off), Chocolatine 2 and SmolLM3 included.
