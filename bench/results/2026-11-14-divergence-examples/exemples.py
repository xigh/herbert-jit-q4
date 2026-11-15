"""Exemples de divergence lisibles, tirés des mesures de justesse Q4 (justesse-1113, Ryzen 9 9900X).

_exemples.py OUT.json [TOP.npz-prefix]

1. Comptes (wiki et fr), depuis les .npz des passages d'origine (out/NOM-K-BRAS.npz : same = accord top-1, kl)
   et l'esquisse de la référence (ref/NOM-K.npz : top_lp, pour l'écart p1 − p2 de la référence).
2. Reproduction : les bras fr refaits avec KEEP=1 (étiquette BRAS-re) comparés aux passages d'origine (.json et .npz).
3. Lecture des logits fr (ref/NOM-fr.kld et out/NOM-fr-BRAS-re.kld) : top-10 de la référence et de chaque bras à
   chaque position, probabilité que chaque bras donne au premier choix de la référence. Gardés dans PREFIX-NOM-fr.npz.
4. Choix d'un exemple par catégorie (fr), le plus « typique » : celui dont la statistique de la catégorie est la
   plus proche de sa médiane sur la catégorie, parmi les positions lisibles. Les 5 suivants sont écrits aussi.

Catégories (définitions exactes, appliquées aux deux corpus) :
 C1 quasi-égalité : écart p1 − p2 de la référence < 0,10 et tous les bras Q4_0 ont un top-1 différent de la référence.
    Sur fr seulement (logits disponibles) : C1s = C1 et tous les bras Q4_0 choisissent le même jeton.
    Statistique de typicité : l'écart p1 − p2.
 C2 grille 7 seule : herbert --q4 (grille 7) change le top-1, tous les autres bras gardent celui de la référence.
    Comptes aussi : grille 7 change et grille 8 garde (quels que soient les autres), et l'inverse.
    Statistique : KL de herbert --q4.
 C3 (Qwen) Q4_K_M seul juste : llama Q4_K_M garde le top-1 de la référence, tous les bras Q4_0 le changent.
    Compte aussi l'inverse (Q4_K_M change, tous les Q4_0 gardent). Statistique : KL moyenne des bras Q4_0.
 C4 KL haute, choix gardé : tous les bras gardent le top-1 de la référence, et la KL de chaque bras Q4_0 est
    au-dessus de son propre 90e centile sur le corpus. Statistique : KL moyenne des bras Q4_0.
Recouvrement : C1 et C3 exigent tous deux que tous les Q4_0 basculent. Pour Qwen, l'exemple C1 est choisi parmi
les C1s où Q4_K_M bascule aussi (vivier), l'exemple C3 parmi les C3 dont l'écart de la référence est ≥ 0,10 (hors C1).
Les comptes donnent la catégorie entière, le vivier et C1 ∩ C3.
Ex aequo : positions où les deux meilleures log-probabilités de la référence sont égales (quantification 16 bits du
fichier) ; le premier choix y est un départage d'argmax. Comptées (clés __dont_ex_aequo) et exclues du vivier.
Lisible : contexte (200 derniers caractères) sans chiffre ni mot de 4 majuscules ou plus (titres), au moins 70 %
de lettres hors espaces ; premier choix de la référence et jetons montrés sans caractère de remplacement (UTF-8
coupé) ; premier choix de la référence avec au moins une lettre ; aucun jeton montré vide une fois décodé ; chaque jeton
montré (deux premiers de la référence, premier choix de chaque bras) est de la ponctuation ou un mot entier qui
apparaît dans le corpus français (pas un morceau de mot comme « accomp » ; une lettre seule : a, à, y).
Le contexte est le texte détokenisé du morceau jusqu'à la position, coupé aux 28 derniers mots, fin exacte gardée.
"""
import sys, os, json, struct, re
import numpy as np
from tokenizers import Tokenizer

J = os.path.expanduser("~/justesse-1113")
MODELS = {
    "qwen36": dict(hf="~/models/Qwen3.6-35B-A3B", nom="Qwen3.6-35B-A3B",
                   arms=["herbert-q4", "herbert-q4g8", "llama-q40pur", "ik-q40pur", "llama-q4km"],
                   q40=["herbert-q4", "herbert-q4g8", "llama-q40pur", "ik-q40pur"], km="llama-q4km"),
    "e4b": dict(hf=os.path.expanduser("~/models/gemma-4-E4B-it-qat-q4_0-unquantized"), nom="Gemma 4 E4B",
                arms=["herbert-q4", "herbert-q4g8", "llama-q40", "ik-q40"],
                q40=["herbert-q4", "herbert-q4g8", "llama-q40", "ik-q40"], km=None),
}
G7, G8 = "herbert-q4", "herbert-q4g8"
C1_MARGIN = 0.10
C4_PCT = 90
TOPK = 10
out_json = sys.argv[1]
prefix = sys.argv[2] if len(sys.argv) > 2 else out_json.replace(".json", "")


def masks(cfg, same, kl, marg, tie):
    arms, q40 = cfg["arms"], cfg["q40"]
    allflip = np.all([~same[a] for a in q40], 0)
    m = {}
    m["C1"] = (marg < C1_MARGIN) & allflip
    m["C2"] = ~same[G7] & np.all([same[a] for a in arms if a != G7], 0)
    m["C2_g7flip_g8keep"] = ~same[G7] & same[G8]
    m["C2_inverse_g8flip_g7keep"] = same[G7] & ~same[G8]
    if cfg["km"]:
        m["C3"] = same[cfg["km"]] & allflip
        m["C3_inverse_km_flip_q40_keep"] = ~same[cfg["km"]] & np.all([same[a] for a in q40], 0)
    m["C4"] = np.all([same[a] for a in arms], 0) & np.all([kl[a] >= np.percentile(kl[a], C4_PCT) for a in q40], 0)
    m["all_q40_flip"] = allflip
    m["ref_margin_lt_0.10"] = marg < C1_MARGIN
    m["all_keep"] = np.all([same[a] for a in arms], 0)
    # ex aequo exacts de la référence (deux log-probabilités égales après la quantification 16 bits du fichier) :
    # le premier choix n'y est qu'un départage d'argmax, une « bascule » peut n'en être pas une.
    base = [k for k in m]
    m["ex_aequo_ref"] = tie
    for k in base:
        if k != "ex_aequo_ref": m[k + "__dont_ex_aequo"] = m[k] & tie
    return m


def open_kld(p):
    f = open(p, "rb")
    assert f.read(8) == b"_logits_"
    n_ctx, V, C = struct.unpack("<3i", f.read(12))
    toks = np.frombuffer(f.read(4 * n_ctx * C), dtype="<i4")
    return f, n_ctx, V, C, toks


def read_logits(name, arms):
    """top-10 de la référence et des bras, p(bras, top-1 réf), argmax de chaque bras, à chaque position fr."""
    files = {"ref": f"{J}/ref/{name}-fr.kld"}
    for a in arms: files[a] = f"{J}/out/{name}-fr-{a}-re.kld"
    H = {k: open_kld(p) for k, p in files.items()}
    _, n_ctx, V, C, tr = H["ref"]
    for k, h in H.items():
        assert (h[1], h[2], h[3]) == (n_ctx, V, C), k
        assert np.array_equal(h[4], tr), "TOKENS DIFFER " + k
    first = n_ctx // 2; R = n_ctx - 1 - first; nv = 2 * ((V + 1) // 2) + 4
    res = {k: dict(top_ids=[], top_lp=[], argmax=[]) for k in H}
    p_reftop = {a: [] for a in arms}
    for c in range(C):
        L = {}
        for k, h in H.items():
            rows = np.frombuffer(h[0].read(2 * R * nv), dtype="<u2").reshape(R, nv)
            hdr = rows[:, :4].copy().view("<f4")
            L[k] = rows[:, 4:4 + V].astype(np.float32) * hdr[:, 0:1] + hdr[:, 1:2]
        rt = L["ref"].argmax(1)
        for k in H:
            res[k]["argmax"].append(L[k].argmax(1))
            idx = np.argpartition(-L[k], TOPK, axis=1)[:, :TOPK]
            v = np.take_along_axis(L[k], idx, 1); o = np.argsort(-v, 1)
            res[k]["top_ids"].append(np.take_along_axis(idx, o, 1)); res[k]["top_lp"].append(np.take_along_axis(v, o, 1))
            if k != "ref": p_reftop[k].append(L[k][np.arange(R), rt])
    out = {}
    for k in H:
        out[f"{k}:top_ids"] = np.concatenate(res[k]["top_ids"]).astype(np.int32)
        out[f"{k}:top_lp"] = np.concatenate(res[k]["top_lp"]).astype(np.float32)
        out[f"{k}:argmax"] = np.concatenate(res[k]["argmax"]).astype(np.int32)
    for a in arms: out[f"{a}:lp_reftop"] = np.concatenate(p_reftop[a]).astype(np.float32)
    out["tokens"] = tr.astype(np.int32); out["n_ctx"] = n_ctx
    return out


CORPUS_MOTS = None


def mot_entier(t):
    """Jeton montré lisible : ponctuation seule, ou un mot entier du texte (précédé ou non d'une espace), pas un
    morceau de mot. « Mot entier » : la forme (minuscules) apparaît comme mot dans le corpus français ; une lettre
    seule n'est admise que pour a, à, y."""
    global CORPUS_MOTS
    if CORPUS_MOTS is None:
        txt = open(f"{J}/corpus/fr-miserables-t1.txt", encoding="utf-8").read().lower()
        CORPUS_MOTS = {w for w in re.findall(r"[^\W\d_]+", txt) if len(w) > 1 or w in ("a", "à", "y")}
    w = t.strip()
    if not w: return False
    if not re.search(r"[^\W\d_]", w): return True
    return w.lower() in CORPUS_MOTS


def main():
    report = dict(criteres=__doc__, seuils=dict(C1_marge=C1_MARGIN, C4_centile=C4_PCT), modeles={})
    for name, cfg in MODELS.items():
        arms, q40 = cfg["arms"], cfg["q40"]
        R = report["modeles"][name] = dict(nom=cfg["nom"], bras=arms, comptes={}, reproduction={}, exemples=[])
        orig = {}
        for K in ("fr", "wiki"):
            ref = np.load(f"{J}/ref/{name}-{K}.npz")
            p = np.exp(ref["top_lp"][:, :2].astype(np.float64)); marg = p[:, 0] - p[:, 1]
            tie = ref["top_lp"][:, 0] == ref["top_lp"][:, 1]
            Z = {a: np.load(f"{J}/out/{name}-{K}-{a}.npz") for a in arms}
            same = {a: Z[a]["same"] for a in arms}; kl = {a: Z[a]["kl"] for a in arms}
            m = masks(cfg, same, kl, marg, tie)
            R["comptes"][K] = dict(positions=int(len(marg)), **{k: int(v.sum()) for k, v in m.items()},
                                  kl_p90={a: float(np.percentile(kl[a], C4_PCT)) for a in q40},
                                  top1={a: float(same[a].mean()) for a in arms})
            if cfg["km"]:
                R["comptes"][K]["C1_inter_C3"] = int((m["C1"] & m["C3"]).sum())
                R["comptes"][K]["C3_hors_C1_marge_ge_0.10"] = int((m["C3"] & (marg >= C1_MARGIN)).sum())
                R["comptes"][K]["C1_et_km_bascule"] = int((m["C1"] & ~same[cfg["km"]]).sum())
            if K == "fr": orig = dict(same=same, kl=kl, marg=marg, m=m, tie=tie)
            print(name, K, json.dumps(R["comptes"][K]), flush=True)
        # reproduction
        for a in arms:
            o = json.load(open(f"{J}/out/{name}-fr-{a}.json")); n = json.load(open(f"{J}/out/{name}-fr-{a}-re.json"))
            zo = np.load(f"{J}/out/{name}-fr-{a}.npz"); zn = np.load(f"{J}/out/{name}-fr-{a}-re.npz")
            keys = ["ppl_test", "dnll", "kl_mean", "kl_median", "kl_p99", "kl_max", "top1_n"]
            R["reproduction"][a] = dict(
                ancien={k: o[k] for k in keys}, refait={k: n[k] for k in keys},
                same_identique=bool(np.array_equal(zo["same"], zn["same"])),
                kl_ecart_max=float(np.abs(zo["kl"] - zn["kl"]).max()),
                nll_test_ecart_max=float(np.abs(zo["nll_test"] - zn["nll_test"]).max()))
            print("repro", name, a, json.dumps(R["reproduction"][a]), flush=True)
        # logits fr
        top_path = f"{prefix}-{name}-fr.npz"
        if os.path.exists(top_path):
            T = dict(np.load(top_path))
        else:
            T = read_logits(name, arms); np.savez_compressed(top_path, **T)
        tok = Tokenizer.from_file(cfg["hf"] + "/tokenizer.json")
        n_ctx = int(T["n_ctx"]); first = n_ctx // 2; Rr = n_ctx - 1 - first; toks = T["tokens"]
        dec = lambda ids: tok.decode([int(i) for i in ids], skip_special_tokens=True)
        reftop = T["ref:argmax"]  # argmax exact (comme _cmp.py) ; le top-10 peut ordonner autrement des ex aequo
        am = {a: T[f"{a}:argmax"] for a in arms}
        # contrôle : accord top-1 relu dans les logits refaits = same d'origine ?
        R["controle_same_depuis_logits"] = {a: bool(np.array_equal(am[a] == reftop, orig["same"][a])) for a in arms}
        # C1s : tous les Q4_0 sur le même jeton
        allsame = np.all([am[a] == am[q40[0]] for a in q40], 0)
        c1s = orig["m"]["C1"] & allsame
        R["comptes"]["fr"]["C1s_meme_jeton"] = int(c1s.sum())
        # second choix de la référence : le meilleur jeton différent de l'argmax (ex aequo compris)
        ti = T["ref:top_ids"]; nz = ti != reftop[:, None]
        sec_col = nz.argmax(1); ref2 = ti[np.arange(len(ti)), sec_col]
        R["comptes"]["fr"]["C1s_jeton_commun_est_le_2e_de_la_ref"] = int((c1s & (am[q40[0]] == ref2)).sum())
        R["comptes"]["fr"]["C1s__dont_ex_aequo"] = int((c1s & orig["tie"]).sum())
        print(name, "C1s", R["comptes"]["fr"]["C1s_meme_jeton"], flush=True)

        def ctx_and_target(i):
            c, j = divmod(i, Rr)
            ctx_ids = toks[c * n_ctx: c * n_ctx + first + 1 + j]
            txt = dec(ctx_ids)
            return txt, int(toks[c * n_ctx + first + 1 + j]), c, j

        def lisible(i):
            txt, _, _, _ = ctx_and_target(i)
            tail = txt[-200:]
            if re.search(r"\d", tail) or re.search(r"\b[A-ZÀ-Ý]{4,}\b", tail): return False
            ns = re.sub(r"\s", "", tail)
            if not ns or sum(ch.isalpha() for ch in ns) / len(ns) < 0.7: return False
            if orig["tie"][i]: return False
            shown = [reftop[i], ref2[i]] + [am[a][i] for a in arms]
            if any("�" in dec([t]) for t in shown): return False
            if not re.search(r"[^\W\d_]", dec([reftop[i]])): return False
            return all(mot_entier(dec([t])) for t in shown)

        meankl = np.mean([orig["kl"][a] for a in q40], 0)
        # C1 et C3 se recouvrent (les deux exigent que tous les Q4_0 basculent) : pour que les exemples racontent
        # deux choses, l'exemple C1 de Qwen est pris là où Q4_K_M bascule aussi, et l'exemple C3 hors quasi-égalité.
        c1_pick, c3_pick = c1s, orig["m"].get("C3")
        if cfg["km"]:
            c1_pick = c1s & ~orig["same"][cfg["km"]]
            c3_pick = orig["m"]["C3"] & (orig["marg"] >= C1_MARGIN)
            R["comptes"]["fr"]["C1s_et_km_bascule"] = int(c1_pick.sum())
        cats = [("C1", c1_pick, orig["marg"], "écart p1 − p2 de la référence"),
                ("C2", orig["m"]["C2"], orig["kl"][G7], "KL de herbert --q4"),
                ("C3", c3_pick, meankl, "KL moyenne des bras Q4_0"),
                ("C4", orig["m"]["C4"], meankl, "KL moyenne des bras Q4_0")]
        chosen = set()
        full = {"C1": orig["m"]["C1"], "C2": orig["m"]["C2"], "C3": orig["m"].get("C3"), "C4": orig["m"]["C4"]}
        for cat, mask, stat, statname in cats:
            if mask is None: continue
            idx = np.flatnonzero(mask)
            if len(idx) == 0: continue
            med = float(np.median(stat[idx]))
            order = idx[np.argsort(np.abs(stat[idx] - med), kind="stable")]
            cand = [int(i) for i in order if lisible(int(i)) and int(i) not in chosen]
            n_lisibles = len(cand)
            picks = cand[:6]
            if not picks: continue
            chosen.add(picks[0])
            for rank, i in enumerate(picks):
                txt, tgt, c, j = ctx_and_target(i)
                starts = [mm.start() for mm in re.finditer(r"\S+", txt)]
                ex = dict(categorie=cat, rang=rank, retenu=(rank == 0), position=i, morceau=c, ligne=j,
                          contexte=("… " + txt[starts[-28]:]) if len(starts) > 28 else txt,
                          texte_suivant=dec([tgt]), texte_suivant_id=tgt,
                          stat=statname, stat_valeur=float(stat[i]), stat_mediane_categorie=med,
                          stat_rang_dans_categorie=float((stat[idx] <= stat[i]).mean()),
                          n_categorie_fr=int(full[cat].sum()), n_vivier_fr=int(len(idx)), n_lisibles=n_lisibles,
                          ref=dict(top=[dict(id=int(t), texte=dec([t]), p=float(np.exp(lp)))
                                        for t, lp in sorted(zip(T["ref:top_ids"][i, :6], T["ref:top_lp"][i, :6]),
                                                            key=lambda x: (x[0] != reftop[i], -x[1]))[:5]]),
                          bras={})
                for a in arms:
                    ex["bras"][a] = dict(choix=dec([am[a][i]]), choix_id=int(am[a][i]), p_choix=float(np.exp(T[f"{a}:top_lp"][i, 0])),
                                         p_choix_ref=float(np.exp(T[f"{a}:lp_reftop"][i])), kl=float(orig["kl"][a][i]),
                                         top=[dict(texte=dec([t]), p=float(np.exp(lp))) for t, lp in zip(T[f"{a}:top_ids"][i, :5], T[f"{a}:top_lp"][i, :5])])
                R["exemples"].append(ex)
                print(f"{name} {cat} rang {rank} pos {i} stat {stat[i]:.4f} (méd {med:.4f}) | {ex['contexte'][-120:]!r} -> réf "
                      f"{ex['ref']['top'][0]['texte']!r} {ex['ref']['top'][0]['p']:.3f} / {ex['ref']['top'][1]['texte']!r} {ex['ref']['top'][1]['p']:.3f} | "
                      + " ".join(f"{a}={ex['bras'][a]['choix']!r}:{ex['bras'][a]['p_choix']:.3f}/{ex['bras'][a]['p_choix_ref']:.3f}" for a in arms), flush=True)
    json.dump(report, open(out_json, "w"), ensure_ascii=False, indent=1)


main()
