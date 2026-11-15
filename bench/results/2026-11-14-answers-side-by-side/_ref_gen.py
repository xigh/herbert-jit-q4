"""Référence : transformers bf16, glouton, 200 jetons au plus, arrêt sur les fins de séquence du modèle.
_ref_gen.py MODEL_DIR prompts.jsonl OUT.jsonl [--threads T]"""
import sys, json, time, torch, transformers
M, P, OUT = sys.argv[1:4]
torch.set_num_threads(int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 12)
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(M)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
print(f"loaded {type(model).__name__} in {time.time()-t0:.0f} s, transformers {transformers.__version__}, torch {torch.__version__}", file=sys.stderr, flush=True)
eos = [248046, 248044]
done = set()
try:
    done = {json.loads(l)["id"] for l in open(OUT)}
except FileNotFoundError:
    pass
with open(OUT, "a") as f:
    for line in open(P):
        p = json.loads(line)
        if p["id"] in done:
            continue
        x = torch.tensor([p["ids"]])
        t0 = time.time()
        with torch.inference_mode():
            y = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=200, do_sample=False, temperature=None, top_k=None, top_p=None, eos_token_id=eos, pad_token_id=248044)
        gen = y[0, x.shape[1]:].tolist()
        f.write(json.dumps({"id": p["id"], "prompt_ids": p["ids"], "ids": gen, "text": tok.decode(gen, skip_special_tokens=False)}, ensure_ascii=False) + "\n"); f.flush()
        print(f"[ref] {p['id']} {len(p['ids'])} -> {len(gen)} tokens in {time.time()-t0:.0f} s", file=sys.stderr, flush=True)
