"""Dépouillement : point de divergence de chaque bras contre la référence, paires entre bras, textes décodés par le même
tokenizer HF. _analyse.py MODEL_DIR OUT_DIR questions.json analyse.json
Adapté de 2026-11-14-reponses-cote-a-cote : 150 jetons, ids d'entrée vérifiés contre prompts.jsonl, marche sans référence (paires seulement), paire llama Q4_0 / Q4_K_M, timings gardés."""
import sys, json, os
from tokenizers import Tokenizer
M, OD, Q, OUT = sys.argv[1:5]
tok = Tokenizer.from_file(os.path.join(M, "tokenizer.json"))
EOS = {248046, 248044}
ARMS = ["herbert-q40", "llama-q40pur", "ik-q40pur", "llama-q4km"]
def load(n):
    p = os.path.join(OD, n + ".jsonl")
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)} if os.path.exists(p) else {}
def norm(ids):
    for i, t in enumerate(ids):
        if t in EOS:
            return ids[:i], "eos"
    return ids, ("coupe" if len(ids) >= 150 else "?")
def dec(ids):
    return tok.decode(ids, skip_special_tokens=False)
def first_diff(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))
def cmp(a, fa, b, fb):
    d = first_diff(a, b)
    if d is None:
        return {"div": None, "etat": "identique" if fa == fb == "eos" else "identique jusqu'à la coupe" if fa == fb else "identique"}
    if d == min(len(a), len(b)):
        if fa == "coupe" or fb == "coupe":
            return {"div": None, "etat": "identique jusqu'à la coupe"}
        return {"div": d, "etat": "fin différente"}
    return {"div": d, "etat": "dévie"}
ref = load("ref-tf-bf16"); A = {a: load(a) for a in ARMS}
P = {json.loads(l)["id"]: json.loads(l)["ids"] for l in open(os.path.join(OD, "..", "prompts.jsonl"))}
MG = load("ref-tf-bf16-marges")  # second passage de la référence, deux premiers jetons à chaque pas
res = []
for q in json.load(open(Q)):
    i = q["id"]
    has_ref = i in ref   # sans référence (pas encore calculée) : seulement les paires entre bras
    r, fr_ = norm(ref[i]["ids"]) if has_ref else ([], None)
    row = {"ref_reproduite": MG[i]["identique_au_premier_passage"] if i in MG else None, "id": i, "longueur": q["longueur"], "genre": q["genre"], "langue": q["langue"], "question": q["question"],
           "ref": {"ids": r, "fin": fr_, "n": len(r), "texte": dec(r)} if has_ref else None, "bras": {}, "paires": {}}
    N = {}
    for a in ARMS:
        if i not in A[a]:
            continue
        x, fx = norm(A[a][i]["ids"]); N[a] = (x, fx)
        assert A[a][i]["prompt_ids"] == P[i], (a, i)
        assert A[a][i]["prompt_ok"], (a, i)
        if not has_ref:
            row["bras"][a] = {"div": None, "etat": "pas de référence", "ids": x, "fin": fx, "n": len(x), "texte": dec(x), "timings": A[a][i].get("timings")}
            continue
        assert ref[i]["prompt_ids"] == P[i], i
        c = cmp(r, fr_, x, fx)
        d = c["div"]
        pre = dec(x[:d]) if d is not None else dec(x)
        full = dec(x)
        suf = full[len(pre):] if full.startswith(pre) else dec(x[d:]) if d is not None else ""
        rpre = dec(r[:d]) if d is not None else ""
        rsuf = row["ref"]["texte"][len(rpre):] if d is not None and row["ref"]["texte"].startswith(rpre) else (dec(r[d:]) if d is not None else "")
        c.update(ids=x, fin=fx, n=len(x), texte=full, prefixe=pre, suite=suf, suite_ref=rsuf,
                 jeton=None if d is None or d >= len(x) else dec([x[d]]), jeton_ref=None if d is None or d >= len(r) else dec([r[d]]),
                 entree_identique=True, tokenize_ok=A[a][i].get("tokenize_matches"), timings=A[a][i].get("timings"))
        if i in MG and d is not None and d < len(MG[i]["top2"]) and MG[i]["identique_au_premier_passage"]:
            (i1, p1), (i2, p2) = MG[i]["top2"][d]
            c["marge_ref"] = {"p1": p1, "t1": dec([i1]), "p2": p2, "t2": dec([i2]), "jeton_bras_est_2e": d < len(x) and x[d] == i2}
        row["bras"][a] = c
    for a, b in (("herbert-q40", "llama-q40pur"), ("llama-q40pur", "llama-q4km"), ("llama-q40pur", "ik-q40pur"), ("herbert-q40", "ik-q40pur")):
        if a in N and b in N:
            row["paires"][a + "|" + b] = cmp(N[a][0], N[a][1], N[b][0], N[b][1])
    res.append(row)
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
# Affichage compact
for row in res:
    print(f"\n### {row['id']} ({row['longueur']}, {row['genre']}) : {row['question']}")
    if row["ref"]:
        print(f"RÉF ({row['ref']['n']} j., fin {row['ref']['fin']}):", row["ref"]["texte"].replace("\n", "⏎"))
    for a, c in row["bras"].items():
        if c["etat"] == "pas de référence":
            print(f"  {a:13s} ({c['n']} j., fin {c['fin']}): {c['texte']!r}")
        elif c["div"] is None:
            print(f"  {a:13s} {c['etat']}")
        else:
            print(f"  {a:13s} {c['etat']} au jeton {c['div']} : réf {c['jeton_ref']!r} / bras {c['jeton']!r} (fin {c['fin']}, {c['n']} j.)")
            print(f"      bras→ {c['suite'][:400]!r}")
    for p, c in row["paires"].items():
        print(f"  paire {p}: {c['etat']}" + ("" if c["div"] is None else f" au jeton {c['div']}"))
