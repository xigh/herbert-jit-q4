#!/usr/bin/env python3
"""Tableau ancienne / nouvelle valeur des lignes de l'article (rejeux du 14/11, grille 8 par défaut).
Usage : _tableau.py DEPOT LAB_DATA. Médianes de 3 recalculées depuis les res-*.json (nouvelles et anciennes) ;
« notée » = valeur du cahier des charges (liste des lignes à rejouer du 14 novembre 2026, non publiée), écart calculé contre elle si elle existe, sinon contre la médiane d'origine."""
import json, os, statistics as st, sys
dep, raw = sys.argv[1], sys.argv[2]
Q = raw + "/2026-11-14-q40pur/"; B13 = raw + "/2026-11-13-epyc-4344p-banc-cpu/"
def med(path, words):
    if not os.path.exists(path): return None
    xs = [x for x in json.load(open(path))["text"] if x["words"] == words]
    if not xs: return None
    return (st.median(x["prompt_per_second"] for x in xs), st.median(x["predicted_per_second"] for x in xs), min(x["predicted_n"] for x in xs), xs[0]["prompt_n"], len(xs))
# (bloc, bras du rejeu, nombre de mots, valeur notée (pp, tg) ou None, ancien json, ce que mesurait l'ancien bras)
L = []
def a(b, arm, w, note, old, oldarm): L.append((b, arm, w, note, old, oldarm))
for w, tag, tagik in ((380, "A", "A2"), (4240, "B", "B")):
    n = {380: {"g8": (336, 25.9), "l": (172, 23.8), "ik": (313, 26.7), "rtr": (359, 26.7), "km": (158, 16.6)},
         4240: {"g8": (306, 23.8), "l": (156, 20.2), "ik": (304, 24.7), "rtr": (350, 24.8), "km": None}}[w]
    a("A", "herbert-g8-qwen36", w, n["g8"], Q + f"debit/res-herbert-q4g8-qwen36-{tag}.json", "herbert b68ce37 --q4 --q4-grid8")
    a("A", "llama-q40pur-qwen36", w, n["l"], Q + f"debit/res-llama-q40pur-qwen36-{tag}.json", "llama.cpp Q4_0 pur")
    a("A", "ik-q40pur-qwen36", w, n["ik"], Q + f"debit/res-ik-q40pur-qwen36-{tagik}.json", "ik Q4_0 pur")
    a("A", "ikrtr-q40pur-qwen36", w, n["rtr"], Q + f"debit/res-ikrtr-q40pur-qwen36-{tagik}.json", "ik -rtr Q4_0 pur")
    a("A", "llama-q4km-qwen36", w, n["km"], Q + f"debit/res-llama-q4km-qwen36-{tag}.json", "llama.cpp Q4_K_M")
    a("A", "herbert-g7-qwen36", w, None, Q + f"debit/res-herbert-q4-qwen36-{tag}.json", "herbert b68ce37 --q4 (grille 7)")
for w, tag in ((78, "reste"), (380, "deux"), (4240, "reste")):
    n = {78: {"g8": (377, 21.3), "l": (239, 19.0), "ik": (326, 22.2), "rtr": (353, 22.2), "km": (203, 13.3)},
         380: {"g8": (369, 21.1), "l": (266, 18.7), "ik": (405, 22.1), "rtr": (439, 22.0), "km": (227, 13.4)},
         4240: {"g8": (387, 18.9), "l": (250, 17.2), "ik": (457, 20.4), "rtr": (505, 20.6), "km": (208, 12.4)}}[w]
    d = Q + "debit-ryzen-9-9900x/"
    a("B", "herbert-g8-qwen36", w, n["g8"], d + f"res-herbert-q4g8-qwen36-{tag}.json", "herbert d24603d --q4 --q4-grid8")
    a("B", "llama-q40pur-qwen36", w, n["l"], d + f"res-llama-q40pur-qwen36-{tag}.json", "llama.cpp Q4_0 pur")
    a("B", "ik-q40pur-qwen36", w, n["ik"], d + f"res-ik-q40pur-qwen36-{tag}.json", "ik Q4_0 pur")
    a("B", "ikrtr-q40pur-qwen36", w, n["rtr"], d + f"res-ikrtr-q40pur-qwen36-{tag}.json", "ik -rtr Q4_0 pur")
    a("B", "llama-q4km-qwen36", w, n["km"], d + f"res-llama-q4km-qwen36-{tag}.json", "llama.cpp Q4_K_M")
    a("B", "herbert-g7-qwen36", w, None, d + f"res-herbert-q4-qwen36-{tag}.json", "herbert d24603d --q4 (grille 7)")
a("C", "herbert-g8-gemmae4b", 380, (399, 17.8), B13 + "grid8/res-herbert-grid8-gemmae4b.json", "herbert b68ce37 --q4 --q4-grid8")
a("C", "llama-gemmae4b", 380, (279, 16.4), B13 + "grid8/res-llama-gemmae4b.json", "llama.cpp")
a("D", "herbert-g8-gemma26", 380, (350, 20.2), B13 + "grid8/res-herbert-grid8-gemma26.json", "herbert b68ce37 --q4 --q4-grid8")
a("D", "llama-gemma26", 380, (140, 18.7), B13 + "grid8/res-llama-gemma26.json", "llama.cpp")
a("D", "ikrtr-gemma26", 380, (360, 19.7), dep + "/ref-epyc-4344p-ik-1113/res-ik-gemma26-rtr.json", "ik -rtr (face à herbert grille 7)")
a("D", "ik-gemma26", 380, None, dep + "/ref-epyc-4344p-ik-1113/res-ik-gemma26.json", "ik")
a("E", "herbert-g8-gemmae2b", 380, None, None, "(pas de valeur d'origine en grille 8)")
a("E", "herbert-g7-gemmae2b", 380, (802, 35.6), B13 + "gemmae/res-herbert-gemmae2b.json", "herbert b68ce37 --q4 (grille 7)")
a("E", "llama-gemmae2b", 380, (949, 31.3), B13 + "gemmae/res-llama-gemmae2b.json", "llama.cpp (--mmproj)")
a("E", "ik-gemmae2b", 380, None, None, "(pas de valeur d'origine sur EPYC 4344P)")
a("E", "ikrtr-gemmae2b", 380, None, None, "(pas de valeur d'origine sur EPYC 4344P)")
f = lambda v, d=0: "—" if v is None else (f"{v:.{d}f}".replace(".", ","))
def ec(new, old): return None if (new is None or old is None) else 100 * (new - old) / old
rows = []
for b, arm, w, note, old, oldarm in L:
    nw = med(f"{dep}/{b}/res-{arm}.json", w); od = med(old, w) if old else None
    ref = note if note else ((round(od[0]), round(od[1], 1)) if od else None)
    e_pp = ec(nw and nw[0], ref and ref[0]); e_tg = ec(nw and nw[1], ref and ref[1])
    flag = any(e is not None and abs(e) > 2 for e in (e_pp, e_tg))
    rows.append((flag, b, arm, w, nw, od, note, ref, e_pp, e_tg, oldarm))
print("| | bloc | bras du rejeu (68c3a96) | jetons | notée (cahier) | médiane d'origine (json) | nouvelle | écart prefill | écart decode | n prédit | ancien bras |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for flag, b, arm, w, nw, od, note, ref, e_pp, e_tg, oldarm in rows:
    s = lambda e: "—" if e is None else (("**" if abs(e) > 2 else "") + f"{e:+.1f} %".replace(".", ",") + ("**" if abs(e) > 2 else ""))
    print(f"| {'> 2 %' if flag else ''} | {b} | {arm} | {nw[3] if nw else (od[3] if od else '?')} | {f(note[0]) + ' / ' + f(note[1],1) if note else '—'} | "
          f"{f(od[0],1) + ' / ' + f(od[1],2) if od else '—'} | {f(nw[0],1) + ' / ' + f(nw[1],2) if nw else 'MANQUE'} | {s(e_pp)} | {s(e_tg)} | {nw[2] if nw else '—'} | {oldarm} |")
