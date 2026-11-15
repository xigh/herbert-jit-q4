"""Invites rendues par le gabarit du modèle (réflexion coupée), en texte et en ids.
_render.py MODEL_DIR questions.json OUT.jsonl"""
import sys, json
from transformers import AutoTokenizer
M, Q, OUT = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(M)
with open(OUT, "w") as f:
    for q in json.load(open(Q)):
        msgs = [{"role": "user", "content": q["texte"]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, add_special_tokens=False).input_ids
        ids2 = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=True)["input_ids"]
        assert list(ids) == list(ids2), q["id"]
        f.write(json.dumps({"id": q["id"], "text": text, "ids": list(ids)}, ensure_ascii=False) + "\n")
        print(q["id"], len(ids), file=sys.stderr)
print(repr(text[-80:]), file=sys.stderr)
