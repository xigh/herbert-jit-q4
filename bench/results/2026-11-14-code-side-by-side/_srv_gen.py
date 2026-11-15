"""Glouton par un serveur : llama.cpp / ik (/completion, invite en ids) ou herbert (/completion, invite en texte rendu,
ids lus dans le probs.json de --log-dir). 500 jetons au plus, fin de séquence respectée. Vérifie les ids d'entrée.
_srv_gen.py KIND(llama|ik|herbert) URL prompts.jsonl OUT.jsonl [LOGDIR]"""
import sys, json, glob, os, urllib.request
kind, url, P, OUT = sys.argv[1:5]; LOGDIR = sys.argv[5] if len(sys.argv) > 5 else None
def post(path, body):
    r = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))
with open(OUT, "w") as f:
    for line in open(P):
        p = json.loads(line)
        rec = {"id": p["id"]}
        if kind in ("llama", "ik"):
            tk = post("/tokenize", {"content": p["text"], "add_special": False, "parse_special": True}).get("tokens")
            rec["tokenize_matches"] = (tk == p["ids"])
            body = dict(prompt=p["ids"], n_predict=500, temperature=0, top_k=1, cache_prompt=False, n_probs=1, return_tokens=True)
            r = post("/completion", body)
            cp = r.get("completion_probabilities", [])
            if kind == "ik":  # ik ne rend pas les ids : les morceaux de chaque pas, convertis par _ik_ids.py
                rec["pieces"] = [c["content"] for c in cp]
                toks = []
            else:
                toks = r.get("tokens") or [c["id"] for c in cp]
            rec.update(prompt_n=r.get("tokens_evaluated"), prompt_ids=p["ids"], ids=toks, text=r.get("content"), stop_type=r.get("stop_type"), stopping_word=r.get("stopping_word"))
            rec["prompt_ok"] = (r.get("tokens_evaluated") == len(p["ids"]))
        else:
            tk = post("/tokenize", {"content": p["text"]}).get("tokens")
            rec["tokenize_matches"] = (tk == p["ids"])
            before = set(glob.glob(LOGDIR + "/*.probs.json"))
            body = dict(prompt=p["text"], n_predict=500, temperature=0, top_k=1, presence_penalty=0, frequency_penalty=0, cache_prompt=False, n_probs=1)
            r = post("/completion", body)
            new = sorted(set(glob.glob(LOGDIR + "/*.probs.json")) - before)
            assert len(new) == 1, new
            pr = json.load(open(new[0]))
            rec.update(prompt_ids=pr["prompt_ids"], ids=pr["generated_ids"], text=json.dumps(r, ensure_ascii=False)[:20000], probs_file=os.path.basename(new[0]))
            rec["prompt_ok"] = (pr["prompt_ids"] == p["ids"])
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        print(f"[{kind}] {p['id']} prompt_ok={rec['prompt_ok']} tokenize_matches={rec['tokenize_matches']} -> {len(rec['ids'])} tokens", file=sys.stderr, flush=True)
