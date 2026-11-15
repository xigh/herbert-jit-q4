#!/usr/bin/env python3
"""(copie du 13/11, SUFFIX ajouté le 14/11) Banc d'un serveur compatible llama-server : prefill, decode, image. Usage : throughput.py URL LABEL OUT.json IMG..."""
import base64, json, os, sys, time, urllib.error, urllib.request

url, label, out = sys.argv[1], sys.argv[2], sys.argv[3]
imgs = sys.argv[4:]
REPS = 3

PARA = ("Le port de Marseille s'éveille avant l'aube : les pêcheurs remontent leurs filets, les cafés lèvent leurs rideaux, "
        "et les premiers ferries pour la Corse font gronder leurs moteurs. Plus haut, Notre-Dame-de-la-Garde veille sur la ville. "
        "In the meantime, engineers at the harbour authority review the tide tables, the weather reports and the cargo manifests, "
        "deciding which ships may dock first and which must wait offshore until the wind drops. ")


def post(path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        data = json.loads(r.read())
    return data, time.perf_counter() - t


def text_prompt(n_words_target):
    s, i = "", 0
    while len(s.split()) < n_words_target:
        s += f"[{i}] " + PARA
        i += 1
    # 14 novembre : SUFFIX (variable d'environnement) ajouté après la consigne. herbert-serve n'a pas ignore_eos :
    # sans suite, Qwen3.6 en Q4 grille 8 rend la fin de séquence au premier jeton ; une liste commencée l'évite.
    return s + "\n\nRésume ce texte en détail, paragraphe par paragraphe :\n" + os.environ.get("SUFFIX", "")


res = {"label": label, "text": [], "image": []}
post("/completion", {"prompt": "Bonjour", "n_predict": 4, "temperature": 0, "cache_prompt": False})  # chauffe

import os
WORDS = [int(w) for w in os.environ.get("WORDS", "380 1500").split()]
for words in WORDS:
    p = text_prompt(words)
    for r in range(REPS):
        d, wall = post("/completion", {"prompt": p, "n_predict": 128, "ignore_eos": True, "temperature": 0, "cache_prompt": False})
        t = d.get("timings", {})
        row = {"words": words, "rep": r, "wall_s": wall, **{k: t.get(k) for k in ("prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_second")}}
        res["text"].append(row)
        print(label, "text", json.dumps(row), flush=True)

for img in imgs:
    b64 = base64.b64encode(open(img, "rb").read()).decode()
    for r in range(REPS + 1):  # la première ne compte pas (chauffe de la tour)
        msg = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": ("data:image/png;base64," if img.endswith(".png") else "data:image/jpeg;base64,") + b64}}, {"type": "text", "text": "Décris cette image."}]}]
        try:
            d, wall = post("/v1/chat/completions", {"messages": msg, "max_tokens": 1, "temperature": 0, "cache_prompt": False})
        except urllib.error.HTTPError as e:
            row = {"image": img.split("/")[-1], "rep": r, "error": f"{e.code} {e.read().decode()[:200]}"}
            res["image"].append(row); print(label, "image", json.dumps(row), flush=True)
            break
        t = d.get("timings", {})
        row = {"image": img.split("/")[-1], "rep": r, "warmup": r == 0, "wall_s": wall, **{k: t.get(k) for k in ("prompt_n", "prompt_ms", "prompt_per_second")}}
        res["image"].append(row)
        print(label, "image", json.dumps(row), flush=True)

json.dump(res, open(out, "w"), indent=1)
print("BANC DONE", label, flush=True)
