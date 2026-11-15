#!/usr/bin/env python3
"""Tableau du README depuis q4-contre-bf16.json : moyenne des deux serveurs par bras (médianes de 3 chacun), rapports Q4/bf16."""
import json, os, sys
D = sys.argv[1]; J = json.load(open(os.path.join(D, "q4-contre-bf16.json"))); G = 1e9
f = lambda x, n: f"{x:,.{n}f}".replace(",", " ").replace(".", ",")
MODS = [("qwen35-4b", "Qwen3.5-4B", (627, 6131), 8.412, 2.630, 9.320), ("qwen35-2b", "Qwen3.5-2B", (627, 6131), 3.764, 1.179, 4.548),
        ("gemmae4b", "Gemma 4 E4B", (626, 6184), 9.289, 2.904, 15.882), ("qwen36-35b", "Qwen3.6-35B-A3B", (627, 6131), 5.893, 1.843, 71.904)]
med = lambda fm, m, n, k: sum(J[f"{fm}-{m}-r{r}"]["sizes"][str(n)][k] for r in (1, 2)) / 2
mem = lambda fm, m, k: sum(J[f"{fm}-{m}-r{r}"][k] for r in (1, 2)) / 2
print("| modèle | format | jetons | prefill (tok/s) | decode (tok/s) | octets relus par jeton | Go/s lus | RSS au repos (Go) | dont fichier | pic de chargement (Go) | pic pendant les requêtes (Go) | chargement (s) | FPGA (cœurs moyens r1 / r2) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for m, nom, sz, ob, oq, disk in MODS:
    for n in sz:
        for fm, o in (("bf16", ob), ("q4", oq)):
            pp, tg = med(fm, m, n, "pp"), med(fm, m, n, "tg")
            c = [nom, "bf16" if fm == "bf16" else "Q4", f(n, 0), f(pp, 1 if pp < 100 else 0), f(tg, 2)]
            if n == sz[0]:
                ld = [J[f"{fm}-{m}-r{r}"]["load_s"] for r in (1, 2)]
                c += [f(o, 3) + " Go", f(tg * o, 1), f(mem(fm, m, "rss_rest") / G, 2), f(mem(fm, m, "file_rest") / G, 2), f(mem(fm, m, "hwm_load") / G, 2),
                      f(mem(fm, m, "hwm_req") / G, 2), " / ".join(f(x, 1) for x in ld), " / ".join(f(J[f"{fm}-{m}-r{r}"]["fpga_coeurs"], 2) for r in (1, 2))]
            else:
                c += [""] * 8
            print("| " + " | ".join(c) + " |")
print("\n| modèle | decode ~630 | decode ~6 150 | prefill ~630 | prefill ~6 150 | octets par jeton | RSS au repos | pic des requêtes | pic de chargement | chargement (plus court des deux serveurs) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for m, nom, sz, ob, oq, disk in MODS:
    a, b = sz; x = lambda k, n: med("q4", m, n, k) / med("bf16", m, n, k); y = lambda k: mem("q4", m, k) / mem("bf16", m, k)
    print(f"| {nom} | **× {f(x('tg', a), 2)}** | × {f(x('tg', b), 2)} | × {f(x('pp', a), 2)} | × {f(x('pp', b), 2)} | {f(oq / ob, 3)} | **{f(y('rss_rest'), 2)}** | {f(y('hwm_req'), 2)} | {f(y('hwm_load'), 2)} | × {f(min(J[f'q4-{m}-r{r}']['load_s'] for r in (1, 2)) / min(J[f'bf16-{m}-r{r}']['load_s'] for r in (1, 2)), 1)} |")
