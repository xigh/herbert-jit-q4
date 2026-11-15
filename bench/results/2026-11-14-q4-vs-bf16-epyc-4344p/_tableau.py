#!/usr/bin/env python3
"""Dépouille run-epyc-4344p.log et les res-*.json : médianes de 3 par serveur et par taille, chargement, mémoire.
Usage : _tableau.py DOSSIER > tableau.md ; écrit aussi DOSSIER/q4-contre-bf16.json (lu par _comparaison.py)."""
import glob, json, os, re, statistics as st, sys

D = sys.argv[1]
log = open(os.path.join(D, "run-epyc-4344p.log")).read()
kb = lambda s, k: int(re.search(k + r": (\d+) kB", s).group(1)) * 1024
srv = {}
for m in re.finditer(r"^LOAD (\S+) ([\d.]+) s", log, re.M): srv.setdefault(m.group(1), {})["load_s"] = float(m.group(2))
for m in re.finditer(r"^REST (\S+) (.*)$", log, re.M):
    s = srv.setdefault(m.group(1), {})
    if "rss_rest" not in s: s.update(rss_rest=kb(m.group(2), "VmRSS"), anon_rest=kb(m.group(2), "RssAnon"), file_rest=kb(m.group(2), "RssFile"), hwm_load=kb(m.group(2), "VmHWM"))
for m in re.finditer(r"^END (\S+) (.*)$", log, re.M): srv.setdefault(m.group(1), {}).update(hwm_req=kb(m.group(2), "VmHWM"), rss_end=kb(m.group(2), "VmRSS"))
for f in glob.glob(os.path.join(D, "*", "res-*.json")):
    r = json.load(open(f)); lab = r["label"]; s = srv.setdefault(lab, {}); s["sizes"] = {}
    for n in sorted({x["prompt_n"] for x in r["text"]}):
        xs = [x for x in r["text"] if x["prompt_n"] == n]
        s["sizes"][n] = dict(pp=st.median(x["prompt_per_second"] for x in xs), tg=st.median(x["predicted_per_second"] for x in xs),
                             pn=sorted({x["predicted_n"] for x in xs}), reps=len(xs),
                             tg_all=[round(x["predicted_per_second"], 2) for x in xs], pp_all=[round(x["prompt_per_second"], 1) for x in xs])
json.dump(srv, open(os.path.join(D, "q4-contre-bf16.json"), "w"), indent=1, sort_keys=True)
G = 1e9
print("| serveur | jetons | prefill (tok/s) | decode (tok/s) | predicted_n | répétitions decode | chargement (s) | RSS repos (Go) | dont fichier (Mo) | pic de chargement VmHWM (Go) | pic pendant les requêtes (Go) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for lab in sorted(srv):
    s = srv[lab]
    for n, z in sorted(s.get("sizes", {}).items()):
        print(f"| {lab} | {n} | {z['pp']:.1f} | {z['tg']:.2f} | {z['pn']} | {z['tg_all']} | {s.get('load_s', 0):.1f} | {s.get('rss_rest', 0)/G:.2f} | {s.get('file_rest', 0)/1e6:.0f} | {s.get('hwm_load', 0)/G:.2f} | {s.get('hwm_req', 0)/G:.2f} |")
