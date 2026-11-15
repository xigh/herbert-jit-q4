#!/usr/bin/env python3
"""Octets de poids relus par jeton de decode, depuis les en-têtes safetensors (comme #001 : partie langage, tour de vision,
audio et MTP exclues, tête liée comptée une fois en entier, tables de recherche (embed_tokens, embed_tokens_per_layer) exclues
sauf la tête liée). bf16 : 2 octets par poids des matrices. Q4 d'herbert (jit/int8.rs) : par bloc de 16 lignes et groupe de
32 k, 320 octets (256 de codes + 16 échelles f32), soit 0,625 octet par poids, lignes arrondies à 16. Le reste (normes, conv1d,
A_log, dt_bias, scalaires) au format du fichier dans les deux bras. Usage : _octets.py DOSSIER [DOSSIER...]"""
import json, struct, sys, glob, os, re

DT = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "I64": 8, "I32": 4, "U8": 1}
SKIP = re.compile(r"(^|\.)(visual|vision_tower|audio_tower|embed_vision|embed_audio|vision|audio)\.|^mtp\.")
LOOKUP = re.compile(r"embed_tokens(_per_layer)?\.weight$")

def headers(d):
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with open(f, "rb") as h:
            n = struct.unpack("<Q", h.read(8))[0]
            for k, v in json.loads(h.read(n)).items():
                if k != "__metadata__":
                    yield k, v["dtype"], v["shape"]

def numel(s):
    p = 1
    for x in s: p *= x
    return p

for d in sys.argv[1:]:
    cfg = json.load(open(os.path.join(d, "config.json")))
    tied = cfg.get("tie_word_embeddings", cfg.get("text_config", {}).get("tie_word_embeddings", False))
    tot = {}; lang = 0; head_seen = False
    bf = q4 = other = lookup = 0; nmat = 0; experts = 0
    for k, dt, s in headers(d):
        b = numel(s) * DT[dt]
        tot[dt] = tot.get(dt, 0) + b
        if SKIP.search(k): continue
        lang += b
        if LOOKUP.search(k):
            lookup += b
            if tied and k.endswith("embed_tokens.weight"):  # la tête liée relit toute la table
                bf += numel(s) * 2; q4 += -(-s[0] // 16) * 16 * s[1] // 32 * 20; nmat += 1
            continue
        if "experts" in k and len(s) == 3:
            experts += b; continue  # MoE : relus selon le routage, comptés à part
        if len(s) == 2 and min(s) >= 32 and s[1] % 32 == 0:
            bf += numel(s) * 2; q4 += -(-s[0] // 16) * 16 * s[1] // 32 * 20; nmat += 1
        else:
            other += b
    G = 1e9
    print(f"{os.path.basename(d.rstrip('/'))} : tête liée {tied} ; fichiers {sum(tot.values())/G:.3f} Go "
          f"({', '.join(f'{k} {v/G:.3f}' for k, v in sorted(tot.items()))}) ; partie langage {lang/G:.3f} Go dont tables de recherche {lookup/G:.3f}"
          + (f", experts {experts/G:.3f}" if experts else ""))
    print(f"  matrices relues par jeton : {nmat}, bf16 {bf/G:.3f} Go ({bf/2**30:.2f} GiB), Q4 {q4/G:.3f} Go ({q4/2**30:.2f} GiB) ; reste {other/G:.4f} Go")
    print(f"  octets relus par jeton (hors experts) : bf16 {(bf+other)/G:.3f} Go, Q4 {(q4+other)/G:.3f} Go, rapport {(q4+other)/(bf+other):.3f}")
