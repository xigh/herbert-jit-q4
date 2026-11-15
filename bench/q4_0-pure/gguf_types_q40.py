#!/usr/bin/env python3
"""Lists the tensor types of a GGUF (gguf-py of llama.cpp 5676dfbe0) next to the format herbert --q4 uses for the same tensor.
Usage: LLAMA_CPP=path/to/llama.cpp gguf_types_q40.py GGUF OUT.json [SHA256]   (also prints every tensor)"""
import os, re, sys, json, collections
sys.path.insert(0, os.path.join(os.environ["LLAMA_CPP"], "gguf-py"))
import gguf

path, out = sys.argv[1], sys.argv[2]
sha = sys.argv[3] if len(sys.argv) > 3 else "?"
r = gguf.GGUFReader(path)
fam = collections.OrderedDict()
count = collections.Counter()
for t in r.tensors:
    ty = t.tensor_type.name
    count[ty] += 1
    print(f"{t.name}\t{ty}\t{[int(x) for x in t.shape]}")
    f = re.sub(r"^blk\.\d+\.", "blk.*.", t.name)
    fam.setdefault(f, collections.Counter())[ty] += 1

# herbert --q4 (src/jit/int8.rs, src/model/loader.rs): every projection of 64 rows or more with k a multiple of 32 in Q4
# (groups of 32, f32 scale; the Q4_0 rule by default, signed max / -8, absmax / 7 with --q4-grid7); the router (blocks of 16 experts, < 64 rows)
# and shared_expert_gate in bf16; in_proj_a/b and conv1d in f32; the embedding table in bf16; the head in Q4; norms in f32.
Q4 = "Q4, groups of 32, f32 scale (grid8: Q4_0 rule)"
HERB = [
    (r"token_embd\.weight", "bf16 (table read, not quantized)"),
    (r"^output\.weight", Q4 + " — head"),
    (r"ffn_gate_inp\.weight", "bf16 (router, blocks of 16 experts)"),
    (r"ffn_gate_inp_shexp\.weight", "bf16 (shared_expert_gate)"),
    (r"ssm_alpha\.weight|ssm_beta\.weight|ssm_ba\.weight", "f32 (in_proj_a / in_proj_b)"),
    (r"ssm_conv1d\.weight", "f32 (conv1d)"),
    (r"ssm_a$|ssm_dt\.bias|ssm_norm\.weight|norm\.weight$|_norm\.weight", "f32 (small vector or norm)"),
    (r"(attn_q|attn_k|attn_v|attn_output|attn_qkv|attn_gate|ssm_out|ffn_(gate|up|down)_(exps|shexp))\.weight", Q4),
]
fams = []
for f, c in fam.items():
    g = ", ".join(f"{k}×{v}" for k, v in c.items())
    h = next((d for p, d in HERB if re.search(p, f)), "?")
    gt = set(c)
    if h.startswith("Q4"):
        same = gt == {"Q4_0"}
        if same:
            h += "; Q4_0: f16 scale"
    elif h.startswith("bf16"):
        same = gt <= {"BF16"} or gt <= {"F32"}  # F32 converted from bf16: same values
        if gt <= {"F32"}:
            h += "; F32 in the GGUF = exact bf16 values"
    elif h.startswith("f32"):
        same = gt <= {"F32"}
    else:
        same = False
    fams.append({"family": f, "n": sum(c.values()), "gguf": g, "herbert": h, "same": same})
json.dump({"file": os.path.basename(path), "bytes": os.path.getsize(path), "sha256": sha, "n": len(r.tensors),
           "count": dict(count), "families": fams}, open(out, "w"), indent=1, ensure_ascii=False)
for x in fams:
    print("#", x["family"], x["gguf"], "|", x["herbert"], "|", "same" if x["same"] else "DIFFERENT")
