#!/usr/bin/env python3
"""sha256 of each tensor of a GGUF (name, type, shape, data sha256) and of its metadata. Usage: tensor_sha.py GGUF LLAMA_DIR"""
import sys, os, hashlib
sys.path.insert(0, os.path.join(sys.argv[2], "gguf-py")); import gguf
r = gguf.GGUFReader(sys.argv[1])
for f in r.fields.values():
    if f.name.startswith("tokenizer.ggml.") and f.name not in ("tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.add_bos_token"):
        v = hashlib.sha256(b"".join(bytes(f.parts[i]) for i in f.data)).hexdigest()[:16]
    else:
        try: v = f.contents()
        except Exception: v = "?"
        if isinstance(v, (list, str)) and len(str(v)) > 80: v = hashlib.sha256(str(v).encode()).hexdigest()[:16]
    print(f"kv\t{f.name}\t{v}")
for t in r.tensors:
    print(f"t\t{t.name}\t{t.tensor_type.name}\t{[int(x) for x in t.shape]}\t{hashlib.sha256(t.data.tobytes()).hexdigest()}")
