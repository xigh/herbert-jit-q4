"""Tableau markdown des quinze questions (analyse.json + jugements.json) et comptes. _table.py DOSSIER"""
import json, os, sys, re
D = sys.argv[1]
A = json.load(open(os.path.join(D, "analyse.json"))); J = json.load(open(os.path.join(D, "jugements.json")))
B = [("herbert-q40", "herbert Q4_0"), ("llama-q40pur", "llama Q4_0"), ("ik-q40pur", "ik Q4_0"), ("llama-q4km", "llama Q4_K_M")]
st = lambda c: "=" if c["div"] is None else f"≠ {c['div'] + 1}"
print("| q | question | jetons réf. | " + " | ".join(n for _, n in B) + " | herbert / llama Q4_0 | fond (jugement) |")
print("|---|---|---|" + "---|" * len(B) + "---|---|")
for r in A:
    p = r["paires"]["herbert-q40|llama-q40pur"]
    print(f"| {r['id']} | {J['questions'][r['id']]['court']} | {r['ref']['n']}{'' if r['ref']['fin'] == 'eos' else ' (coupée)'} | "
          + " | ".join(st(r["bras"][a]) for a, _ in B) + f" | {st(p)} | {re.sub('<[^>]+>', '', J['questions'][r['id']]['fond'])} |")
print()
for a, n in B:
    dv = [r for r in A if r["bras"][a]["div"] is not None]
    late = sum(r["bras"][a]["div"] + 1 > 10 for r in dv)
    print(f"- {n} : dévie sur {len(dv)}/15, identique sur {15 - len(dv)} ; premier jeton différent après le 10e dans {late} des {len(dv)}.")
same_end = sum(r["paires"]["herbert-q40|llama-q40pur"]["div"] is None for r in A)
both = [r for r in A if r["bras"]["herbert-q40"]["div"] is not None and r["bras"]["llama-q40pur"]["div"] is not None]
same_pt = sum(r["bras"]["herbert-q40"]["div"] == r["bras"]["llama-q40pur"]["div"] and r["bras"]["herbert-q40"]["jeton"] == r["bras"]["llama-q40pur"]["jeton"] for r in both)
ik_same = sum(r["paires"]["llama-q40pur|ik-q40pur"]["div"] is None for r in A)
cat1 = [r["id"] for r in A if r["bras"]["herbert-q40"]["div"] is not None and r["bras"]["llama-q40pur"]["div"] is not None and r["bras"]["llama-q4km"]["div"] is None]
cat2 = [r["id"] for r in A if all(c["div"] is None for c in r["bras"].values())]
print(f"- herbert et llama Q4_0 identiques entre eux jusqu'au bout (ou la coupe) : {same_end}/15 ; quand les deux dévient ({len(both)}), même jeton au même rang : {same_pt}.")
print(f"- llama Q4_0 et ik Q4_0 (même fichier) identiques jusqu'au bout : {ik_same}/15.")
ens = [i for i in cat1 for r in A if r["id"] == i and r["bras"]["herbert-q40"]["div"] == r["bras"]["llama-q40pur"]["div"] and r["bras"]["herbert-q40"]["jeton"] == r["bras"]["llama-q40pur"]["jeton"]]
print(f"- Les deux Q4_0 dévient, Q4_K_M identique à la référence : {len(cat1)} ({', '.join(cat1)}) ; dont ensemble (même rang, même jeton) : {len(ens)} ({', '.join(ens)}).")
print(f"- Les cinq bras identiques : {len(cat2)} ({', '.join(cat2)}).")
print(f"- Référence rejouée : identique au premier passage sur {sum(bool(r.get('ref_reproduite')) for r in A)}/15.")
