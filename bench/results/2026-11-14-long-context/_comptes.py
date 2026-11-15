"""Comptes de la campagne : sorties de la référence par bras et par longueur, premier jeton divergent (médiane),
identités entre bras, temps rendus par les serveurs et par la référence. _comptes.py (dans le dossier ; lit ryzen-9-9900x/)."""
import json, statistics as st, re
A = json.load(open("ryzen-9-9900x/analyse.json"))
ARMS = ["herbert-q40", "llama-q40pur", "ik-q40pur", "llama-q4km"]
C = {"sorties": {}, "paires": {}, "temps": {}}
for a in ARMS:
    for L in ("court", "long"):
        rows = [r for r in A if r["longueur"] == L]
        div = [r["bras"][a]["div"] for r in rows if r["bras"][a]["div"] is not None]
        C["sorties"][f"{a}|{L}"] = {"quitte": len(div), "sur": len(rows), "mediane_div": st.median(div) if div else None, "divs": div}
for p in ["herbert-q40|llama-q40pur", "llama-q40pur|ik-q40pur", "herbert-q40|ik-q40pur", "llama-q40pur|llama-q4km"]:
    for L in ("court", "long"):
        rows = [r for r in A if r["longueur"] == L]
        C["paires"][f"{p}|{L}"] = sum(r["paires"][p]["div"] is None for r in rows)
def med(xs): return round(st.median(xs), 2)
for a, f in (("llama-q40pur", "llama-q40pur"), ("llama-q4km", "llama-q4km"), ("ik-q40pur", "ik-q40pur.raw")):
    R = [json.loads(l) for l in open(f"ryzen-9-9900x/out/{f}.jsonl")]
    for L, pre in (("court", "c"), ("long", "l")):
        T = [r["timings"] for r in R if r["id"].startswith(pre)]
        C["temps"][f"{a}|{L}"] = {"prefill": med([t["prompt_per_second"] for t in T]), "decode": med([t["predicted_per_second"] for t in T])}
pat = re.compile(r"prompt (\d+) tokens in [\d.]+ s \(([\d.]+) tok/s\), (\d+) generated in [\d.]+ s \(([\d.]+) tok/s\)")
H = [m.groups() for m in pat.finditer(open("ryzen-9-9900x/logs/srv-herbert-q40.log").read())]
for L, cond in (("court", lambda n: n < 4000), ("long", lambda n: n >= 4000)):
    X = [h for h in H if cond(int(h[0]))]
    C["temps"][f"herbert-q40|{L}"] = {"prefill": med([float(h[1]) for h in X]), "decode": med([float(h[3]) for h in X])}
R = [json.loads(l) for l in open("ryzen-9-9900x/out/ref-tf-bf16.jsonl")]
for L, pre in (("court", "c"), ("long", "l")):
    X = [r for r in R if r["id"].startswith(pre)]
    C["temps"][f"ref|{L}"] = {"prefill_s": [round(r["prefill_s"], 1) for r in X], "prefill_tok_s": [round(len(r["prompt_ids"]) / r["prefill_s"]) for r in X],
                              "decode_s_par_jeton": [round(r["decode_s_par_jeton"], 3) for r in X], "total_s": round(sum(r["total_s"] for r in X))}
json.dump(C, open("comptes.json", "w"), ensure_ascii=False, indent=1)
for k, v in C["sorties"].items(): print(k, v["quitte"], "/", v["sur"], "médiane", v["mediane_div"], v["divs"])
for k, v in C["paires"].items(): print("identiques", k, v)
for k, v in C["temps"].items(): print(k, v)
