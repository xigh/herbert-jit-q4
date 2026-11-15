"""Contrôle de la référence : second passage glouton transformers bf16 avec les logits de chaque pas (output_scores) ;
vérifie que les ids sont les mêmes qu'au premier passage et garde, à chaque pas, les deux premiers jetons et leurs probabilités.
_ref_marges.py MODEL_DIR prompts.jsonl ref-tf-bf16.jsonl OUT.jsonl [--threads T]"""
import sys, json, time, torch
M, P, R1, OUT = sys.argv[1:5]
torch.set_num_threads(int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 12)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
first = {json.loads(l)["id"]: json.loads(l)["ids"] for l in open(R1)}
with open(OUT, "w") as f:
    for line in open(P):
        p = json.loads(line); x = torch.tensor([p["ids"]]); t0 = time.time()
        with torch.inference_mode():
            o = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=200, do_sample=False, temperature=None, top_k=None, top_p=None,
                               eos_token_id=[248046, 248044], pad_token_id=248044, output_scores=True, output_logits=True, return_dict_in_generate=True)
        gen = o.sequences[0, x.shape[1]:].tolist()
        steps = []
        for lg in o.logits:
            pr = torch.softmax(lg[0].float(), -1); v, i = pr.topk(2)
            steps.append([[int(i[0]), float(v[0])], [int(i[1]), float(v[1])]])
        f.write(json.dumps({"id": p["id"], "ids": gen, "identique_au_premier_passage": gen == first[p["id"]], "top2": steps}) + "\n"); f.flush()
        print(f"[marges] {p['id']} {len(gen)} tokens, identique au premier passage : {gen == first[p['id']]} ({time.time()-t0:.0f} s)", file=sys.stderr, flush=True)
