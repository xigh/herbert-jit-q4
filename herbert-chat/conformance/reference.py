#!/usr/bin/env python3
"""The reference render: each template of templates/ on each conversation of cases.json, thinking on then off, by
transformers (render_jinja_template, Python's jinja2, as apply_chat_template does).

usage: python3 -I reference.py CASES_JSON TEMPLATES_DIR OUT_JSON

Tool-call arguments given as a JSON string are parsed into an object first, as herbert-chat (and llama-server) do before
the template: both renders start from the same messages.
"""
import copy, json, sys
from pathlib import Path

import jinja2
import transformers
from transformers.utils.chat_template_utils import render_jinja_template

BOS_EOS = {"gemma": ("<bos>", "<eos>"), "qwen": ("", "<|im_end|>"), "qwen2": ("", "<|im_end|>"), "granite": ("<|end_of_text|>", "<|end_of_text|>"), "mistral": ("<s>", "</s>"), "lfm2": ("<|startoftext|>", "<|im_end|>"), "lfm3": ("<|startoftext|>", "<|im_end|>"), "smol": ("<|begin_of_text|>", "<|im_end|>")}


def normalise(messages):
    messages = copy.deepcopy(messages)
    for m in messages:
        for c in m.get("tool_calls") or []:
            a = c.get("function", {}).get("arguments")
            if isinstance(a, str):
                try:
                    c["function"]["arguments"] = json.loads(a)
                except ValueError:
                    pass
    return messages


def main():
    cases_path, tdir, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    spec = json.load(open(cases_path, encoding="utf-8"))
    results = {"_versions": {"transformers": transformers.__version__, "jinja2": jinja2.__version__}}
    for name, t in spec["templates"].items():
        source = (tdir / f"{name}.jinja").read_text(encoding="utf-8")
        bos, eos = BOS_EOS[t["family"]]
        for case in spec["cases"]:
            tools = spec["tools"][case["tools"]] if case.get("tools") else None
            for thinking in spec["thinking"]:
                key = f"{name}|{case['id']}|{'think' if thinking else 'nothink'}"
                try:
                    r = render_jinja_template(
                        conversations=[normalise(case["messages"])],
                        tools=tools,
                        chat_template=source,
                        add_generation_prompt=case.get("generation_prompt", True),
                        bos_token=bos,
                        eos_token=eos,
                        enable_thinking=thinking,
                    )
                    rendered = r[0] if isinstance(r, tuple) else r
                    if isinstance(rendered, list):
                        rendered = rendered[0]
                    results[key] = {"ok": rendered}
                except Exception as e:  # the template's own raise_exception, or a jinja error
                    results[key] = {"error": f"{type(e).__name__}: {e}"}
    json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{out}: {len(results) - 1} renders")


main()
