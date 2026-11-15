#!/usr/bin/env python3
"""Génère la page locale de comparaison herbert / llama.cpp / ik_llama.cpp en Q4_0 pur (Qwen3.6-35B-A3B).
Usage : _comparaison.py DOSSIER_LABO SORTIE.html
Lit, dans DOSSIER_LABO : types-qwen36.json (types des tenseurs du GGUF), qualite/*.json (bras du jour, Ryzen 9 9900X),
qualite-nuit/*.json (bras de la nuit, Ryzen 9 9900X), debit/res-*.json (banc EPYC 4344P du jour), sort-predictions.json,
PREDICTIONS-q40pur.md ; et le banc d'hier dans ../2026-11-13-epyc-4344p-banc-cpu/."""
import glob, html, json, os, statistics as st, sys, datetime

D, OUT = sys.argv[1], sys.argv[2]
HIER = os.path.join(D, "..", "2026-11-13-epyc-4344p-banc-cpu")
E = html.escape


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def fr(x, n=3):
    return f"{x:.{n}f}".replace(".", ",")


# ---------- qualité ----------
ARMS = [  # (étiquette fichier, nom affiché, source)
    ("herbert-q4g8", "herbert <code>--q4 --q4-grid8</code> (règle Q4_0)", "qualite-nuit"),
    ("herbert-q4", "herbert <code>--q4</code> (grille absmax/7, pour mémoire)", "qualite-nuit"),
    ("llama-q40pur", "llama.cpp 5676dfbe0, Q4_0 pur", "qualite"),
    ("ik-q40pur", "ik_llama.cpp 7aa7e014, Q4_0 pur", "qualite"),
    ("ikrtr-q40pur", "ik_llama.cpp 7aa7e014 <code>-rtr</code>, Q4_0 pur", "qualite"),
    ("llama-q4km", "llama.cpp, Q4_K_M (rappel)", "qualite-nuit"),
    ("ik-q4km", "ik_llama.cpp, Q4_K_M (rappel)", "qualite-nuit"),
    ("herbert-bf16", "herbert bf16 (test nul)", "qualite-nuit"),
]


def qual_rows(corpus):
    rows = []
    for lab, name, src in ARMS:
        j = load(os.path.join(D, src, f"qwen36-{corpus}-{lab}.json"))
        if j is None:
            rows.append(f"<tr class=miss><td>{name}</td><td colspan=6>pas encore mesuré</td></tr>")
            continue
        when = "nuit du 13 au 14" if src == "qualite-nuit" else "14 novembre"
        rows.append(
            f"<tr><td>{name}</td><td class=n>{fr(j['ppl_test'])}</td><td class=n>{'+' if j['dnll'] >= 0 else '−'}{fr(abs(j['dnll']), 4)} ± {fr(j['dnll_se'], 4)}</td>"
            f"<td class=n><b>{fr(j['kl_mean'], 4)}</b></td><td class=n>{fr(j['kl_p99'], 3)}</td><td class=n>{fr(100 * j['top1'], 1)} %</td><td class=src>Ryzen 9 9900X, {when}</td></tr>")
    ref = None
    for lab, _, src in ARMS:
        j = load(os.path.join(D, src, f"qwen36-{corpus}-{lab}.json"))
        if j:
            ref = j['ppl_ref']; break
    return ref, "\n".join(rows)


# ---------- débit ----------
def medians(path):
    js = [load(p) for p in sorted(glob.glob(path.replace(".json", "-*.json"))) if "smoke" not in p] + [load(path)]
    js = [j for j in js if j]
    if not js:
        return None
    W = {}
    for t in (t for j in js for t in j["text"]):
        if t.get("prompt_per_second") is None:
            continue
        W.setdefault(t["words"], []).append(t)
    return {w: (st.median(x["prompt_n"] for x in v), st.median(x["prompt_per_second"] for x in v), st.median(x["predicted_per_second"] for x in v), len(v), min(x["predicted_n"] for x in v)) for w, v in W.items()}


DEBIT = [
    ("herbert-q4g8", "herbert <code>--q4 --q4-grid8</code>"),
    ("herbert-q4", "herbert <code>--q4</code>"),
    ("llama-q40pur", "llama.cpp Q4_0 pur"),
    ("ik-q40pur", "ik_llama.cpp Q4_0 pur"),
    ("ikrtr-q40pur", "ik_llama.cpp <code>-rtr</code> Q4_0 pur"),
    ("llama-q4km", "llama.cpp Q4_K_M (rejoué dans la fenêtre)"),
]
COMPARABLE = {"herbert-q4g8": True, "llama-q40pur": True, "ik-q40pur": True, "ikrtr-q40pur": True,
              "herbert-q4": False, "llama-q4km": False}


def _kl(lab, src, c):
    j = load(os.path.join(D, src, f"qwen36-{c}-{lab}.json"))
    return j["kl_mean"] if j else None


def _ecart():
    out = {}
    for lab in ("herbert-q4", "llama-q4km"):
        parts = []
        for c in ("wiki", "fr"):
            a, b = _kl(lab, "qualite-nuit", c), _kl("llama-q40pur", "qualite", c)
            if a and b:
                parts.append(f"KL {fr(a, 4)} contre {fr(b, 4)} pour le Q4_0, {'+' if a > b else '−'}{fr(abs(a / b - 1) * 100, 0)} % ({c})")
        out[lab] = " ; ".join(parts)
    return out


ECART = _ecart()


def debit_table(sub="debit"):
    data = {lab: medians(os.path.join(D, sub, f"res-{lab}-qwen36.json")) for lab, _ in DEBIT}
    words = sorted({w for v in data.values() if v for w in v})
    if not words:
        return "<p class=miss>Aucun débit mesuré aujourd'hui pour l'instant.</p>"
    base = data.get("herbert-q4g8") or data.get("herbert-q4")
    h = ["<table><tr><th>moteur</th>" + "".join(f"<th>{w} mots</th>" for w in words) + "</tr>"]
    for what, k in (("prefill (tok/s)", 1), ("decode (tok/s)", 2)):
        h.append(f"<tr class=sec><td colspan={len(words) + 1}>{what}</td></tr>")
        for lab, name in DEBIT:
            v = data.get(lab)
            cells = []
            for w in words:
                if v and w in v:
                    r = ""
                    if base and w in base and lab not in ("herbert-q4g8", "herbert-q4"):
                        r = f" <span class=r>herbert ×{fr(base[w][k] / v[w][k], 2)}</span>"
                    bad = " <span class=nc>" + f"{v[w][4]} jetons" + "</span>" if (k == 2 and v[w][4] < 128) else ""
                    bad += f" <span class=r>n={v[w][3]}</span>" if v[w][3] != 3 else ""
                    cells.append(f"<td class=n>{fr(v[w][k], 1)}{r}{bad}</td>")
                else:
                    cells.append("<td class=miss>—</td>")
            flag = "" if COMPARABLE[lab] else f" <span class=nc>non comparable à qualité égale ({ECART.get(lab, '')})</span>"
            h.append(f"<tr><td>{name}{flag}</td>{''.join(cells)}</tr>")
    h.append("<tr class=sec><td colspan=%d>jetons de prompt (médiane)</td></tr>" % (len(words) + 1))
    v = next((x for x in data.values() if x), {})
    h.append("<tr><td>prompt_n</td>" + "".join(f"<td class=n>{int(v[w][0]) if w in v else '—'}</td>" for w in words) + "</tr>")
    h.append("</table>")
    return "\n".join(h)


def hier_table():
    rows = []
    for f, name in (("final/res-herbert-qwen36.json", "herbert --q4"), ("final/res-llama-qwen36.json", "llama.cpp Q4_K_M"),
                    ("tailles/res-herbert-qwen36.json", "herbert --q4"), ("tailles/res-llama-qwen36.json", "llama.cpp Q4_K_M")):
        j = load(os.path.join(HIER, f)); m = None
        if j:
            W = {}
            for t in j["text"]: W.setdefault(t["words"], []).append(t)
            m = {w: (st.median(x["prompt_n"] for x in v), st.median(x["prompt_per_second"] for x in v), st.median(x["predicted_per_second"] for x in v)) for w, v in W.items()}
        if m:
            rows.append(f"<tr><td>{name}</td><td>{f.split('/')[0]}</td><td>" + " ; ".join(
                f"{int(v[0])} j. : {fr(v[1], 0)} / {fr(v[2], 1)}" for w, v in sorted(m.items())) + "</td></tr>")
    return "<table><tr><th>moteur</th><th>série</th><th>jetons : prefill / decode (tok/s)</th></tr>" + "".join(rows) + "</table>"


# ---------- types ----------
def types_block():
    t = load(os.path.join(D, "types-qwen36.json"))
    if not t:
        return "<p class=miss>Types des tenseurs : pas encore relevés.</p>"
    h = [f"<p>Fichier <code>{E(t['file'])}</code>, {fr(t['bytes'] / 1e9, 2)} Go, sha256 <code>{t['sha256'][:16]}…</code>, "
         f"{t['n']} tenseurs. Comptes par type : {', '.join(f'{k} {v}' for k, v in sorted(t['count'].items()))}.</p>"]
    h.append("<table><tr><th>famille de tenseurs</th><th>type dans le GGUF</th><th>herbert <code>--q4</code></th><th>même format ?</th></tr>")
    for r in t["families"]:
        cls = "" if r["same"] else " class=diff"
        h.append(f"<tr{cls}><td><code>{E(r['family'])}</code> ({r['n']})</td><td>{E(r['gguf'])}</td><td>{E(r['herbert'])}</td><td>{'oui' if r['same'] else 'non'}</td></tr>")
    h.append("</table>")
    return "\n".join(h)


def pred_block():
    s = load(os.path.join(D, "sort-predictions.json")) or {}
    txt = open(os.path.join(D, "PREDICTIONS-q40pur.md")).read()
    items = [l for l in txt.splitlines() if l[:2].rstrip(".").isdigit() or (l[:1].isdigit() and l[1:3] == ". ")]
    h = ["<table><tr><th>prédiction (06:35, avant mesure)</th><th>sort</th></tr>"]
    for l in items:
        k = l.split(".")[0]
        v = s.get(k, "en attente")
        cls = {"juste": "ok", "fausse": "ko"}.get(v.split(" ")[0], "")
        h.append(f"<tr><td>{E(l)}</td><td class='{cls}'>{E(v)}</td></tr>")
    h.append("</table>")
    return "\n".join(h)


def status():
    t = load(os.path.join(D, "types-qwen36.json"))
    q = sorted(os.path.basename(p) for p in glob.glob(os.path.join(D, "qualite", "qwen36-*.json")))
    db = sorted(os.path.basename(p) for p in glob.glob(os.path.join(D, "debit", "res-*.json")))
    extra = open(os.path.join(D, "etat.txt")).read().strip() if os.path.exists(os.path.join(D, "etat.txt")) else ""
    fait, manque = [], []
    (fait if t else manque).append("GGUF Q4_0 pur produit et types relevés")
    (fait if len(q) >= 6 else manque).append(f"qualité llama / ik / ik-rtr sur le Q4_0 pur ({len(q)}/6 mesures)")
    for sub, nom in (("debit", "EPYC 4344P, 8 fils, invite à suffixe"), ("debit-ryzen-9-9900x", "Ryzen 9 9900X, 12 fils, invite d'hier, herbert ignore_eos")):
        labs = {os.path.basename(p).split("-qwen36")[0][4:] for p in glob.glob(os.path.join(D, sub, "res-*-qwen36-*.json")) if "smoke" not in p}
        tags = {os.path.basename(p).rsplit("-", 1)[1][:-5] for p in glob.glob(os.path.join(D, sub, "res-*-qwen36-*.json")) if "smoke" not in p}
        full = len(labs) >= 6 and len(tags) >= 2
        (fait if full else manque).append(f"débit {nom} ({len(labs)}/6 bras, séries {', '.join(sorted(tags)) or '—'})")
    return fait, manque, extra


# ---------- exemples de divergence (../2026-11-14-exemples-divergence/exemples.json, _exemples.py) ----------
ARM_NOM = {"herbert-q4": "herbert <code>--q4</code> (grille 7)", "herbert-q4g8": "herbert <code>--q4 --q4-grid8</code> (grille 8)",
           "llama-q40pur": "llama.cpp Q4_0 pur", "ik-q40pur": "ik_llama.cpp Q4_0 pur", "llama-q4km": "llama.cpp Q4_K_M",
           "llama-q40": "llama.cpp Q4_0 (QAT de Google)", "ik-q40": "ik_llama.cpp Q4_0 (QAT de Google)"}
CAT_TITRE = {"C1": "Quasi-égalité de la référence : tous les Q4_0 basculent",
             "C2": "La grille 7 d'herbert bascule, pas la grille 8",
             "C3": "Q4_K_M reste juste, tous les Q4_0 tombent",
             "C4": "Le choix ne change pas, la distribution si"}


def tok(s):
    s = s.replace(" ", "␣").replace("\n", "↵")
    return f"<span class=tok>{E(s)}</span>"


def pc(p):
    return f"{100 * p:.1f}".replace(".", ",") + " %" if p >= 0.001 else f"{100 * p:.2f}".replace(".", ",") + " %"


def entier(n):
    return f"{n:,}".replace(",", " ")


def un_sur(n, tot):
    return f"1 position sur {round(tot / n)}" if n else "aucune position"


def exemples_block():
    X = load(os.path.join(D, "..", "2026-11-14-exemples-divergence", "exemples.json"))
    if not X:
        return ""
    out = ["""<h2>Ce que la divergence change, mot à mot</h2>
<p>Les tableaux de qualité donnent des moyennes. Voici, sur le texte français (<i>Les Misérables</i>, t. I, 3 048 positions), quelques positions précises : le texte qui précède (tel que le modèle le voit, détokenisé), ce que la référence 16 bits (transformers bf16) met en premier et en second, puis, pour chaque bras, le jeton qu'il met en premier, sa probabilité, et la probabilité qu'il donne au premier choix de la référence. Les jetons sont encadrés ; <span class=tok>␣</span> est l'espace qui commence le jeton, <span class=tok>↵</span> un saut de ligne. « Bascule » veut dire : le premier choix du bras n'est pas celui de la référence. <b>Ex aequo</b> : à certaines positions, les deux premiers jetons de la référence ont exactement la même log-probabilité dans le fichier (stockée sur 16 bits) ; le « premier choix » n'y est qu'un départage de l'argmax, et une bascule à ces positions n'en est pas une. Elles comptent aussi dans l'accord top-1 de la campagne ; elles sont comptées à part ci-dessous et exclues des exemples.</p>
<p><b>Comment les positions ont été choisies.</b> Quatre catégories, définies sur l'accord du premier choix (déjà mesuré : <code>same</code> des <code>.npz</code> de la campagne) et sur l'écart p1 − p2 de la référence :</p>
<ul>
<li><b>C1, quasi-égalité</b> : écart p1 − p2 de la référence inférieur à 0,10, et tous les bras Q4_0 basculent ; pour l'exemple, en plus, tous les Q4_0 choisissent le même jeton (et, pour Qwen, Q4_K_M bascule aussi, pour ne pas recouvrir C3).</li>
<li><b>C2, grille 7 seule</b> : herbert <code>--q4</code> (grille 7) bascule, tous les autres bras gardent le premier choix de la référence.</li>
<li><b>C3, Q4_K_M seul juste</b> (Qwen) : llama.cpp Q4_K_M garde le premier choix de la référence, tous les bras Q4_0 basculent ; pour l'exemple, écart de la référence d'au moins 0,10 (hors C1).</li>
<li><b>C4, choix gardé malgré une KL haute</b> : tous les bras gardent le premier choix de la référence, et la KL de chaque bras Q4_0 dépasse son propre 90e centile sur le corpus.</li>
</ul>
<p>Dans chaque catégorie, l'exemple est la position <b>la plus typique</b> : celle dont la statistique de la catégorie (C1 : l'écart p1 − p2 ; C2 : la KL de la grille 7 ; C3 et C4 : la KL moyenne des Q4_0) est la plus proche de la médiane de la catégorie, parmi les positions dont le contexte se lit (pas de chiffre ni de titre en majuscules dans les 200 derniers caractères, au moins 70 % de lettres, jetons montrés des mots entiers du texte ou de la ponctuation, pas des morceaux de mot, pas d'ex aequo exact de la référence). Ce filtre de lisibilité réduit beaucoup le vivier : chaque exemple indique combien de positions lisibles il restait, et le rang centile de sa statistique dans le vivier (50 = au milieu). Les cinq suivants dans cet ordre sont dans <code>exemples.json</code>.</p>
<p>Les jetons et probabilités viennent des bras <b>rejoués</b> le 14 novembre avec les logits gardés (mêmes binaires, même commande) ; """]
    rep = [(m, a, r) for m, R in X["modeles"].items() for a, r in R["reproduction"].items()]
    ident = all(r["same_identique"] and r["kl_ecart_max"] == 0 and r["nll_test_ecart_max"] == 0 for _, _, r in rep)
    out.append("la reproduction est <b>identique au bit</b> aux passages d'origine pour tous les bras (accord top-1, KL et nll à chaque position).</p>" if ident else
               "la reproduction <b>n'est pas identique au bit</b> pour : " + ", ".join(f"{m} {a} (KL écart max {r['kl_ecart_max']:.2e})" for m, a, r in rep if not (r["same_identique"] and r["kl_ecart_max"] == 0)) + ".</p>")
    for m, R in X["modeles"].items():
        cf, cw = R["comptes"]["fr"], R["comptes"]["wiki"]
        nf, nw = cf["positions"], cw["positions"]
        km = "C3" in cf
        out.append(f"<h3>{E(R['nom'])}</h3>")
        sa = lambda c, k: c[k] - c[k + "__dont_ex_aequo"]
        rows = [("Ex aequo exacts de la référence (deux premiers égaux au bit du fichier)", cf["ex_aequo_ref"], cw["ex_aequo_ref"]),
                ("C1 quasi-égalité, tous les Q4_0 basculent", cf["C1"], cw["C1"]),
                ("&nbsp;&nbsp;hors ex aequo", sa(cf, "C1"), sa(cw, "C1")),
                ("&nbsp;&nbsp;dont tous les Q4_0 sur le même jeton (fr)", cf["C1s_meme_jeton"], None),
                ("Tous les Q4_0 basculent (quel que soit l'écart)", cf["all_q40_flip"], cw["all_q40_flip"]),
                ("C2 grille 7 seule bascule (tous les autres gardent)", cf["C2"], cw["C2"]),
                ("&nbsp;&nbsp;hors ex aequo", sa(cf, "C2"), sa(cw, "C2")),
                ("&nbsp;&nbsp;grille 7 bascule, grille 8 garde (les autres quels qu'ils soient)", cf["C2_g7flip_g8keep"], cw["C2_g7flip_g8keep"]),
                ("&nbsp;&nbsp;inverse : grille 8 bascule, grille 7 garde", cf["C2_inverse_g8flip_g7keep"], cw["C2_inverse_g8flip_g7keep"])]
        if km:
            rows += [("C3 Q4_K_M garde, tous les Q4_0 basculent", cf["C3"], cw["C3"]),
                     ("&nbsp;&nbsp;hors ex aequo", sa(cf, "C3"), sa(cw, "C3")),
                     ("&nbsp;&nbsp;dont quasi-égalités (C1 ∩ C3)", cf["C1_inter_C3"], cw["C1_inter_C3"]),
                     ("&nbsp;&nbsp;inverse : Q4_K_M bascule, tous les Q4_0 gardent", cf["C3_inverse_km_flip_q40_keep"], cw["C3_inverse_km_flip_q40_keep"])]
        rows += [("C4 tous gardent, KL de chaque Q4_0 au-dessus de son p90", cf["C4"], cw["C4"]),
                 ("Tous les bras gardent le premier choix", cf["all_keep"], cw["all_keep"])]
        t = "".join(f"<tr><td>{a}</td><td class=n>{entier(b)}</td><td class=n>{pc(b / nf)}</td><td class=n>{'' if c is None else entier(c)}</td><td class=n>{'' if c is None else pc(c / nw)}</td></tr>" for a, b, c in rows)
        out.append(f"<p>Comptes des catégories (positions) :</p><table><tr><th>catégorie</th><th>fr</th><th>part fr</th><th>wiki</th><th>part wiki</th></tr>{t}</table>")
        q40 = [a for a in R["bras"] if "q4km" not in a]
        for ex in [e for e in R["exemples"] if e["retenu"]]:
            c = ex["categorie"]
            r1, r2 = ex["ref"]["top"][0], ex["ref"]["top"][1]
            B = ex["bras"]
            out.append(f"<h4>{E(CAT_TITRE[c])}</h4>")
            out.append(f"<blockquote>{E(ex['contexte'])}<span class=tok>?</span></blockquote>")
            out.append(f"<p class=sub>Position {ex['position']} (morceau {ex['morceau']}). Le texte de Hugo continue par {tok(ex['texte_suivant'])}.</p>")
            t = (f"<tr class=sec><td>référence 16 bits</td><td>{tok(r1['texte'])}</td><td class=n>{pc(r1['p'])}</td><td class=n>{pc(r1['p'])}</td><td class=n>—</td>"
                 f"<td>2e : {tok(r2['texte'])} {pc(r2['p'])}</td></tr>")
            for a, b in B.items():
                bas = b["choix_id"] != r1["id"]
                t += (f"<tr{' class=diff' if bas else ''}><td>{ARM_NOM.get(a, E(a))}</td><td>{tok(b['choix'])}</td><td class=n>{pc(b['p_choix'])}</td>"
                      f"<td class=n>{pc(b['p_choix_ref'])}</td><td class=n>{fr(b['kl'])}</td><td>{'bascule' if bas else 'garde'}</td></tr>")
            out.append(f"<table><tr><th>bras</th><th>premier choix</th><th>sa probabilité</th><th>p du choix de la réf.</th><th>KL</th><th></th></tr>{t}</table>")
            stat = (f"{ex['stat']} {fr(ex['stat_valeur'])}, médiane du vivier {fr(ex['stat_mediane_categorie'])}, "
                    f"rang centile {round(100 * ex['stat_rang_dans_categorie'])}")
            if c == "C1":
                commun = B[q40[0]]["choix"]
                km_aussi = km and B["llama-q4km"]["choix_id"] == B[q40[0]]["choix_id"]
                second = B[q40[0]]["choix_id"] == r2["id"]
                txt = (f"La référence hésite entre {tok(r1['texte'])} ({pc(r1['p'])}) et {tok(r2['texte'])} ({pc(r2['p'])}). "
                       f"Tous les Q4_0 passent à {tok(commun)}{', son second choix' if second else ''}" + (", et Q4_K_M aussi" if km_aussi else "") +
                       ". Les deux suites sont presque aussi probables pour la référence : une petite perturbation suffit à les départager. "
                       f"{un_sur(cf['C1'], nf)} est une quasi-égalité où tous les Q4_0 basculent sur le français ({entier(cf['C1'])} sur {entier(nf)}, dont {entier(cf['C1__dont_ex_aequo'])} ex aequo exacts de la référence ; {entier(cw['C1'])} sur {entier(nw)} sur wikitext, dont {entier(cw['C1__dont_ex_aequo'])}) ; "
                       f"elles font {pc(cf['C1'] / max(cf['all_q40_flip'], 1))} des positions où tous les Q4_0 basculent ensemble ({entier(cf['all_q40_flip'])}). "
                       f"Dans {entier(cf['C1s_meme_jeton'])} des {entier(cf['C1'])}, tous les Q4_0 tombent sur le même jeton, "
                       f"le second choix de la référence dans {entier(cf['C1s_jeton_commun_est_le_2e_de_la_ref'])}"
                       + (f" ; et Q4_K_M bascule aussi dans {entier(ex['n_vivier_fr'])} de ces {entier(cf['C1s_meme_jeton'])} (le vivier de cet exemple, {un_sur(ex['n_vivier_fr'], nf)})." if km else "."))
            elif c == "C2":
                txt = (f"Seule la grille 7 d'herbert (<code>absmax/7</code>) quitte {tok(r1['texte'])} pour {tok(B['herbert-q4']['choix'])} ; "
                       f"la grille 8 (la règle de Q4_0) et les moteurs GGUF le gardent. {un_sur(cf['C2'], nf)} est de ce genre sur le français "
                       f"({entier(cf['C2'])} ; {entier(cw['C2'])} sur wikitext). Ce n'est pas un avantage net de la grille 8 position par position : "
                       f"grille 7 bascule et grille 8 garde {entier(cf['C2_g7flip_g8keep'])} fois, mais l'inverse arrive {entier(cf['C2_inverse_g8flip_g7keep'])} fois "
                       f"(wikitext : {entier(cw['C2_g7flip_g8keep'])} contre {entier(cw['C2_inverse_g8flip_g7keep'])}).")
            elif c == "C3":
                txt = (f"Q4_K_M garde {tok(r1['texte'])} ; les quatre Q4_0, herbert comme llama.cpp et ik, passent à "
                       + ", ".join(sorted({tok(B[a]['choix']) for a in q40})) +
                       f". {un_sur(cf['C3'], nf)} est de ce genre sur le français ({entier(cf['C3'])} ; {entier(cw['C3'])} sur wikitext), "
                       f"dont {entier(cf['C1_inter_C3'])} sont aussi des quasi-égalités (l'exemple est pris parmi les {entier(ex['n_vivier_fr'])} autres) ; l'inverse (Q4_K_M seul bascule) arrive {entier(cf['C3_inverse_km_flip_q40_keep'])} fois "
                       f"({entier(cw['C3_inverse_km_flip_q40_keep'])} sur wikitext). Ici, à format égal, les trois moteurs Q4_0 basculent ensemble.")
            else:
                ps = [B[a]["p_choix_ref"] for a in q40]
                txt = (f"Tous les bras gardent {tok(r1['texte'])}, mais la référence lui donne {pc(r1['p'])} et les Q4_0 entre {pc(min(ps))} et {pc(max(ps))} : "
                       f"la distribution bouge (KL entre {fr(min(B[a]['kl'] for a in q40))} et {fr(max(B[a]['kl'] for a in q40))}) sans changer le choix glouton. "
                       f"{un_sur(cf['C4'], nf)} est de ce genre sur le français ({entier(cf['C4'])} ; {entier(cw['C4'])} sur wikitext) : le seuil (le p90 de chaque bras Q4_0 à la fois) est strict. "
                       f"Au total, tous les bras gardent le premier choix à {entier(cf['all_keep'])} positions sur {entier(nf)}.")
            out.append(f"<p>{txt} <span class=r>({stat} ; {entier(ex['n_lisibles'])} positions lisibles dans le vivier de {entier(ex['n_vivier_fr'])}.)</span></p>")
    out.append("<p class=src>Données : <code>2026-11-14-exemples-divergence/</code> (<code>_exemples.py</code>, <code>exemples.json</code>, top-10 de chaque bras par position).</p>")
    return "\n".join(out)


def q4bf16_block():
    """Section « Q4 contre bf16 : la vitesse et la mémoire » : dépôts 2026-11-14-q4-contre-bf16 (EPYC 4344P, 8 fils) et
    2026-11-14-q4-contre-bf16-ryzen-9-9900x (Ryzen 9 9900X, 12 fils), même code 68c3a96 ; la machine en colonne."""
    G = 1e9
    MACH = [("EPYC 4344P", "2026-11-14-q4-contre-bf16", [("qwen35-2b", "Qwen3.5-2B", (627, 6131), 3.764, 1.179, 4.548),
                                                    ("gemmae4b", "Gemma 4 E4B", (626, 6184), 9.289, 2.904, 15.882)]),
            ("Ryzen 9 9900X", "2026-11-14-q4-contre-bf16-ryzen-9-9900x", [("qwen35-4b", "Qwen3.5-4B", (627, 6131), 8.412, 2.630, 9.320),
                                                           ("qwen35-2b", "Qwen3.5-2B", (627, 6131), 3.764, 1.179, 4.548),
                                                           ("gemmae4b", "Gemma 4 E4B", (626, 6184), 9.289, 2.904, 15.882),
                                                           ("qwen36-35b", "Qwen3.6-35B-A3B", (627, 6131), 5.893, 1.843, 71.904)])]
    data = {}
    for mach, dep, _ in MACH:
        try:
            data[mach] = json.load(open(os.path.join(D, "..", dep, "q4-contre-bf16.json")))
        except OSError:
            pass
    if "EPYC 4344P" not in data:
        return "<h2>Q4 contre bf16 : la vitesse et la mémoire</h2><p class=miss>Données absentes.</p>"
    rows, rat = [], []
    for mach, dep, mods in MACH:
        if mach not in data:
            continue
        J = data[mach]
        med = lambda fmt, mod, n, k: sum(J[f"{fmt}-{mod}-r{r}"]["sizes"][str(n)][k] for r in (1, 2)) / 2
        mem = lambda fmt, mod, k: sum(J[f"{fmt}-{mod}-r{r}"][k] for r in (1, 2)) / 2
        # chargement : moyenne des deux serveurs sur EPYC 4344P (comme son README) ; le plus court des deux sur Ryzen 9 9900X (premiers chargements
        # d'E4B et du 35B en partie relus sur le disque)
        load = (lambda fmt, mod: min(J[f"{fmt}-{mod}-r{r}"]["load_s"] for r in (1, 2))) if mach == "Ryzen 9 9900X" else (lambda fmt, mod: mem(fmt, mod, "load_s"))
        for mod, nom, sizes, ob, oq, disk in mods:
            for n in sizes:
                for fmt, oct_ in (("bf16", ob), ("q4", oq)):
                    pp, tg = med(fmt, mod, n, "pp"), med(fmt, mod, n, "tg")
                    cells = [mach, E(nom), "bf16" if fmt == "bf16" else "Q4", fr_int(n), fr(pp, 1 if pp < 100 else 0), fr(tg, 2)]
                    if n == sizes[0]:
                        cells += [fr(oct_, 3) + " Go", fr(tg * oct_, 1), fr(mem(fmt, mod, "rss_rest") / G, 2), fr(mem(fmt, mod, "file_rest") / G, 2),
                                  fr(mem(fmt, mod, "hwm_req") / G, 2), fr(mem(fmt, mod, "hwm_load") / G, 2), fr(load(fmt, mod), 1), fr(disk, 3) + " Go"]
                    else:
                        cells += [""] * 8
                    rows.append("<tr>" + "".join(f"<td{' class=n' if i >= 3 else ''}>{c}</td>" for i, c in enumerate(cells)) + "</tr>")
            a, b = sizes
            x = lambda k, n: med("q4", mod, n, k) / med("bf16", mod, n, k)
            y = lambda k: mem("q4", mod, k) / mem("bf16", mod, k)
            rat.append(f"<tr><td>{mach}</td><td>{E(nom)}</td><td class=n><b>× {fr(x('tg', a), 2)}</b></td><td class=n>× {fr(x('tg', b), 2)}</td><td class=n>× {fr(x('pp', a), 2)}</td><td class=n>× {fr(x('pp', b), 2)}</td>"
                       f"<td class=n>{fr(oq / ob, 3)}</td><td class=n><b>{fr(y('rss_rest'), 2)}</b></td><td class=n>{fr(y('hwm_req'), 2)}</td><td class=n>× {fr(load('q4', mod) / load('bf16', mod), 1)}</td></tr>")
    q36 = data["EPYC 4344P"].get("q4-qwen36-35b", {})
    ryzen = "Ryzen 9 9900X" in data
    return f"""<h2>Q4 contre bf16 : la vitesse et la mémoire</h2>
<p>Même code (archive 68c3a96, <code>herbert-jit-llama-serve --ctx 8192</code>, bras Q4 = <code>--q4</code>, bras bf16 = sans), mêmes fichiers safetensors bf16, bras alternés bf16, Q4, Q4, bf16 dans une même fenêtre par machine : <b>EPYC 4344P</b> (EPYC 4344P, Zen 4, 8 fils, sonde 54 Go/s) et <b>Ryzen 9 9900X</b> (Ryzen 9 9900X, Zen 5, 12 fils, sonde 44 Go/s à 12 fils). 128 jetons produits, 3 répétitions par serveur, médianes, moyenne des deux serveurs (chargement sur Ryzen 9 9900X : le plus court des deux serveurs). Raccord à la campagne C1 de #001 : Qwen3.5-2B bf16 13,71 tok/s alors sur EPYC 4344P, 13,88 ici ; Qwen3.5-4B (celui de l'article) n'est que sur Ryzen 9 9900X, bf16 5,09 contre 5,12 dans C1, et le 2B 11,43 contre 11,51. Octets relus par jeton : depuis les en-têtes safetensors, méthode de #001 ; en Q4, 0,625 octet par poids (codes 4 bits et une échelle f32 par 32) ; pour le 35B-A3B, les 8 experts routés sur 256 sont ajoutés. RSS : <code>/proc/PID/status</code> au repos après chargement ; « dont fichier » = <code>RssFile</code>, des pages du fichier projeté. Go = 10⁹ octets.</p>
<table><tr><th>machine</th><th>modèle</th><th>format</th><th>jetons</th><th>prefill</th><th>decode</th><th>octets par jeton</th><th>Go/s lus</th><th>RSS repos (Go)</th><th>dont fichier</th><th>pic requêtes</th><th>pic chargement</th><th>chargement (s)</th><th>disque</th></tr>{''.join(rows)}</table>
<p>Rapports Q4 / bf16 :</p>
<table><tr><th>machine</th><th>modèle</th><th>decode ~630</th><th>decode ~6 150</th><th>prefill ~630</th><th>prefill ~6 150</th><th>octets par jeton</th><th>RSS repos</th><th>pic requêtes</th><th>chargement</th></tr>{''.join(rat)}</table>
<p>Sur les deux machines, les deux formats décodent à la bande passante (≈ 51–52 Go/s lus sur EPYC 4344P, ≈ 41–43 sur Ryzen 9 9900X) : le Q4 relit 3,2 fois moins d'octets et décode environ 3,1 fois plus vite, au même rapport à 1 ou 2 % près d'une machine à l'autre. Le gain en mémoire est moindre qu'en octets : RSS divisée par 2,1 (4B) et 1,9 (2B) ; celle d'E4B seulement par 1,3 à 1,4, car 9,3 Go restent des pages du fichier dans les deux bras (ses tables de recherche, <code>embed_tokens_per_layer</code> surtout, lues en place ; mémoire anonyme seule sur EPYC 4344P : 12,38 → 6,35 Go). Les modèles denses en Q4 se chargent 1,9 à 2,7 fois plus lentement (quantification au montage) ; le Q4 ne réduit pas la taille sur disque (mêmes safetensors).</p>
<p><b>Qwen3.6-35B-A3B</b> : en bf16, <b>71,90 Go</b> de safetensors (69,32 Go de partie langage). Sur EPYC 4344P (67,0 Go de mémoire), il ne tient pas ; en Q4, RSS au repos <b>{fr(q36.get('rss_rest', 0) / G, 2)} Go</b>, pic pendant les requêtes {fr(q36.get('hwm_req', 0) / G, 2)} Go (pic de chargement {fr(q36.get('hwm_load', 0) / G, 2)} Go), et le débit de la table des rejeux : 334,8 / 25,93 tok/s à 627 jetons, 305,8 / 23,85 à 6 131. {'Sur Ryzen 9 9900X, il tient dans les deux formats (lignes ci-dessus) : bf16 71,2 Go résidents, decode 7,20 tok/s (42 Go/s lus pour 5,9 Go par jeton), Q4 24,9 Go et 20,77 tok/s, soit × 2,89 en decode et 0,35 en mémoire. Le prefill bf16 ne fait que 8,6 tok/s : le journal du serveur donne au prefill bf16 de ce MoE le même programme que le decode (jeton par jeton), alors que le Q4 a son propre programme de prefill (349 tok/s).' if ryzen else ''}</p>
<p>{'⚠️ Sur Ryzen 9 9900X, un travail FPGA en cours sur la même machine occupait environ un cœur pendant presque toutes les mesures, pas également pour tous les serveurs : le second serveur Q4 du 2B a tourné sans lui (+2,9 % en decode, +8,9 % en prefill contre le premier), et le bf16 du 4B en a eu plus que son Q4 (1,3 et 1,9 cœur contre 0,8 et 1,0). Rapports par paire de serveurs dans le README du dépôt : à conditions égales (2B r1, E4B), decode × 3,02 à × 3,13, prefill × 1,41 à × 1,46. Les premiers chargements d’E4B et du 35B ont relu une partie des fichiers sur le disque, d’où le plus court des deux serveurs.' if ryzen else ''}</p>
<p class=src>Données : <code>2026-11-14-q4-contre-bf16/</code> (EPYC 4344P : <code>README.md</code>, <code>PREDICTIONS.md</code> sha256 141d70a3…, <code>run-epyc-4344p.log</code>, <code>_octets.py</code>) et <code>2026-11-14-q4-contre-bf16-ryzen-9-9900x/</code> (Ryzen 9 9900X : <code>README.md</code>, <code>PREDICTIONS.md</code> sha256 4919b1c2…, <code>run-ryzen-9-9900x.log</code>, <code>fpga-intervalles.txt</code>).</p>"""

def fr_int(n):
    return f"{n:,}".replace(",", " ")


# ---------- réponses entières côte à côte (../2026-11-14-reponses-cote-a-cote/analyse.json, jugements.json) ----------
REP_BRAS = [("herbert-q40", "herbert Q4_0 (grille 8)"), ("llama-q40pur", "llama.cpp Q4_0 pur"), ("ik-q40pur", "ik_llama.cpp Q4_0 pur"),
            ("llama-q4km", "llama.cpp Q4_K_M")]


def _rep_txt(c):
    """Texte d'un bras : partie commune avec la référence en clair, le reste surligné, le premier jeton différent souligné."""
    if c["div"] is None:
        return E(c["texte"])
    s = c["suite"]
    j = c.get("jeton") or ""
    head = s[:len(j)] if j and s.startswith(j) else ""
    return E(c["prefixe"]) + "<span class=dv>" + (f"<b class=dv1>{E(head)}</b>" if head else "") + E(s[len(head):]) + "</span>"


def _rep_etat(c):
    if c is None:
        return "—"
    if c["div"] is None:
        return "identique" if c["etat"] == "identique" else "identique jusqu'à la coupe"
    m = c.get("marge_ref") or {}
    return f"jeton {c['div'] + 1}" + (" (fin)" if c["etat"] == "fin différente" else "") + (" (ex aequo)" if m.get("ex_aequo") else "")


def reponses_block():
    R = load(os.path.join(D, "..", "2026-11-14-reponses-cote-a-cote", "analyse.json"))
    J = load(os.path.join(D, "..", "2026-11-14-reponses-cote-a-cote", "jugements.json"))
    if not R or not J:
        return ""
    bras = [(a, n) for a, n in REP_BRAS if any(a in r["bras"] for r in R)]
    out = [f"""<h2>Des réponses entières, côte à côte</h2>
<p>{J['intro']}</p>"""]
    byid = {r["id"]: r for r in R}
    for ex in J["exemples"]:
        r = byid[ex["id"]]
        out.append(f"<h4>{E(ex['titre'])}</h4>")
        out.append(f"<blockquote>{E(r['question']).replace(chr(10), '<br>')}</blockquote>")
        cols = [f"<div class=rep><div class=rh>référence 16 bits (transformers bf16)</div>{E(r['ref']['texte'])}</div>"]
        for a, n in bras:
            c = r["bras"].get(a)
            if c:
                cols.append(f"<div class=rep><div class=rh>{n} · {_rep_etat(c)}</div>{_rep_txt(c)}</div>")
        out.append("<div class=cols>" + "".join(cols) + "</div>")
        out.append(f"<p>{ex['commentaire']}</p>")
    t = ""
    for r in R:
        jg = J["questions"].get(r["id"], {})
        p = r["paires"].get("herbert-q40|llama-q40pur")
        cells = "".join(f"<td class=n>{_rep_etat(r['bras'].get(a))}</td>" for a, _ in bras)
        t += (f"<tr><td>{r['id']}</td><td>{E(r['sorte'])}</td><td>{E(jg.get('court', ''))}</td><td class=n>{r['ref']['n']}{'' if r['ref']['fin'] == 'eos' else ' (coupée)'}</td>{cells}"
              f"<td class=n>{_rep_etat(p)}</td><td>{jg.get('fond', '')}</td></tr>")
    head = "".join(f"<th>{n}</th>" for _, n in bras)
    out.append(f"<p>{J['tableau_intro']}</p>")
    out.append(f"<table><tr><th></th><th>sorte</th><th>question</th><th>jetons de la réf.</th>{head}<th>herbert contre llama.cpp Q4_0</th><th>fond (jugement)</th></tr>{t}</table>")
    out.append(f"<p>{J['comptes']}</p>")
    out.append("<p class=src>Données : <code>2026-11-14-reponses-cote-a-cote/</code> (questions, <code>PREDICTIONS.md</code> et son sha256, ids et textes de chaque bras dans <code>out/</code>, <code>_analyse.py</code>, <code>analyse.json</code>, <code>jugements.json</code>).</p>")
    return "\n".join(out)


# ---------- du code, noté par exécution (../2026-11-14-code-cote-a-cote/_section_code.py) ----------
def code_block():
    import importlib.util
    f = os.path.join(D, "..", "2026-11-14-code-cote-a-cote", "_section_code.py")
    if not os.path.exists(f):
        return ""
    spec = importlib.util.spec_from_file_location("_section_code", f); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.code_block()


# ---------- un document long, puis une question (../2026-11-14-long-contexte/analyse.json, jugements.json) ----------
LONG_BRAS = [("herbert-q40", "herbert Q4_0 (grille 8)"), ("llama-q40pur", "llama.cpp Q4_0 pur"), ("ik-q40pur", "ik_llama.cpp Q4_0 pur"),
             ("llama-q4km", "llama.cpp Q4_K_M")]
LONG_NOTE = {"J": "<span class=ok>juste</span>", "P": "partiel", "F": "<span class=ko>faux</span>"}


def long_block():
    LD = os.path.join(D, "..", "2026-11-14-long-contexte")
    R = load(os.path.join(LD, "analyse.json"))
    J = load(os.path.join(LD, "jugements.json"))
    if not R or not J:
        return ""
    out = [f"""<h2>Un document long, puis une question</h2>
<p>{J['intro']}</p>"""]
    byid = {r["id"]: r for r in R}
    for ex in J["exemples"]:
        r = byid[ex["id"]]
        out.append(f"<h4>{E(ex['titre'])}</h4>")
        out.append(f"<blockquote>{E(r['question'])}</blockquote>")
        cols = [f"<div class=rep><div class=rh>référence 16 bits (transformers bf16)</div>{E(r['ref']['texte'])}</div>"]
        for a, n in LONG_BRAS:
            c = r["bras"].get(a)
            if c:
                cols.append(f"<div class=rep><div class=rh>{n} · {_rep_etat(c)}</div>{_rep_txt(c)}</div>")
        out.append("<div class=cols>" + "".join(cols) + "</div>")
        out.append(f"<p>{ex['commentaire']}</p>")
    t = ""
    for r in R:
        jg = J["notes"][r["id"]]
        cells = f"<td class=n>{LONG_NOTE[jg['ref']]}</td>" + "".join(
            f"<td class=n>{_rep_etat(r['bras'].get(a))} · {LONG_NOTE[jg[a]]}</td>" for a, _ in LONG_BRAS)
        p = r["paires"].get("herbert-q40|llama-q40pur")
        t += (f"<tr><td>{r['id']}</td><td>{E(r['longueur'])}</td><td>{E(r['genre'])}</td><td>{E(r['langue'])}</td>{cells}"
              f"<td class=n>{_rep_etat(p)}</td><td>{jg.get('remarque', '')}</td></tr>")
    head = "<th>réf. bf16</th>" + "".join(f"<th>{n}</th>" for _, n in LONG_BRAS)
    out.append(f"<p>{J['tableau_intro']}</p>")
    out.append(f"<table><tr><th></th><th>texte</th><th>question</th><th>langue</th>{head}<th>herbert contre llama.cpp Q4_0</th><th>remarque (jugement)</th></tr>{t}</table>")
    out.append(J["comptes_html"])
    out.append(J["temps_html"])
    out.append("<p class=src>Données : <code>2026-11-14-long-contexte/</code> (textes et source Gutenberg, <code>questions.json</code>, <code>ATTENDUES.md</code> et <code>PREDICTIONS.md</code> avec leurs sha256, ids et textes de chaque bras dans <code>ryzen-9-9900x/out/</code>, <code>_analyse.py</code>, <code>analyse.json</code>, <code>jugements.json</code>, <code>README.md</code>).</p>")
    return "\n".join(out)

fait, manque, extra = status()
ref_w, rows_w = qual_rows("wiki")
ref_f, rows_f = qual_rows("fr")
now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

page = f"""<!doctype html>
<html lang=fr><head><meta charset=utf-8><meta name=viewport content="width=device-width, initial-scale=1">
<title>Q4_0 pur, Qwen3.6</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1d1f;--mut:#6b6b70;--line:#e2dfd8;--acc:#2f5d8a;--ok:#2c7a3f;--ko:#a3342a;--diff:#fff3df;--miss:#9a9aa0}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#18181a;--fg:#e9e7e2;--mut:#a0a0a8;--line:#333338;--acc:#8fb4dc;--ok:#7cc48d;--ko:#e88a80;--diff:#3a2f1e;--miss:#77777f}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding:24px 16px;max-width:1100px;margin:auto}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:18px;margin-top:32px;border-bottom:1px solid var(--line);padding-bottom:4px}} h3{{font-size:15px;margin-top:22px}}
.sub{{color:var(--mut)}} table{{border-collapse:collapse;width:100%;margin:8px 0;font-size:13.5px;display:block;overflow-x:auto}}
th,td{{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top}} th{{color:var(--mut);font-weight:600}}
td.n{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}} .miss,tr.miss td{{color:var(--miss)}} tr.diff td{{background:var(--diff)}}
tr.sec td{{font-weight:600;color:var(--acc);padding-top:10px}} .r{{color:var(--mut);font-size:12px}} .nc{{color:var(--ko);font-size:12px}}
.ok{{color:var(--ok)}} .ko{{color:var(--ko)}} .src{{color:var(--mut);font-size:12px}} code{{font-size:12.5px}}
.box{{border:1px solid var(--line);border-radius:6px;padding:10px 14px;margin:12px 0}}
.tok{{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;background:var(--diff);border:1px solid var(--line);border-radius:3px;padding:0 3px;white-space:pre}}
blockquote{{margin:8px 0;padding:6px 14px;border-left:3px solid var(--acc);font-family:Georgia,serif}} h4{{font-size:14.5px;margin:24px 0 4px}}
.cols{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:8px;margin:8px 0}} .rep{{border:1px solid var(--line);border-radius:6px;padding:8px 10px;font-size:13px;white-space:pre-wrap;overflow-wrap:anywhere}}
.rh{{color:var(--mut);font-size:12px;font-weight:600;margin-bottom:4px;white-space:normal}} .dv{{background:var(--diff)}} .dv1{{font-weight:600;text-decoration:underline;text-decoration-color:var(--ko)}}
</style></head><body>
<h1>herbert contre llama.cpp et ik_llama.cpp à format égal : Q4_0 pur</h1>
<div class=sub>Page générée le {now} (heure de Paris) par <code>_comparaison.py</code> ; données brutes : <code>2026-11-14-q40pur/</code>.</div>
<div class=box><b>Résumé.</b> Qwen3.6-35B-A3B : {E(extra) if extra else 'mesures en cours.'}<br>
<b>Fait :</b> {'; '.join(fait) or '—'}.<br><b>Manque :</b> {'; '.join(manque) or 'rien'}.</div>

<h2>Qwen3.6-35B-A3B</h2>
<h3>Le fichier et ses types de tenseurs</h3>
{types_block()}

<h3>Qualité contre la référence 16 bits (transformers bf16)</h3>
<p>Toutes ces mesures viennent de <b>Ryzen 9 9900X</b> (Zen 5, AVX-512), pas de epyc-4344p. Méthode : convention de <code>llama-perplexity</code>, n_ctx 256, seconde moitié notée. KL(réf ‖ moteur) moyenne par position, exacte (référence complète). Δnll = nll du moteur moins nll de la référence, ± erreur type par jeton. La KL est la mesure de fidélité ; le Δnll peut passer sous zéro sans que le modèle soit plus fidèle.</p>
<p><b>wikitext-2</b>, 48 morceaux, 6 096 positions, ppl de la référence {fr(ref_w) if ref_w else '—'}</p>
<table><tr><th>bras</th><th>ppl</th><th>Δnll ± se</th><th>KL moy.</th><th>KL p99</th><th>top-1</th><th>source</th></tr>{rows_w}</table>
<p><b>français</b> (<i>Les Misérables</i>, t. I), 24 morceaux, 3 048 positions, ppl de la référence {fr(ref_f) if ref_f else '—'}</p>
<table><tr><th>bras</th><th>ppl</th><th>Δnll ± se</th><th>KL moy.</th><th>KL p99</th><th>top-1</th><th>source</th></tr>{rows_f}</table>

<h3>Débit sur EPYC 4344P (même fenêtre)</h3>
<p>Débit mesuré sur <b>EPYC 4344P</b>, qualité sur <b>Ryzen 9 9900X</b>. EPYC 4344P : AMD EPYC 4344P (Zen 4, 8 cœurs, AVX-512), 62 Go. Un serveur à la fois sous <code>~/bench-epyc-4344p.lock</code>. API <code>/completion</code>, texte brut, <code>cache_prompt:false</code>, <code>ignore_eos</code>, 128 jetons, température 0, 3 répétitions, médianes des <code>timings</code> du serveur, 8 fils. Changement par rapport à hier : l'invite finit par « \\n1. Le port de Marseille » (variable SUFFIX) pour tous les moteurs, car herbert-serve n'a pas <code>ignore_eos</code> et, sans cette suite, Qwen3.6 en grille 8 rend la fin de séquence au premier jeton (premier essai gardé dans <code>debit-essai1-eos/</code>) ; d'où ~6 jetons de prompt de plus qu'hier. Ratios « herbert × » : herbert <code>--q4 --q4-grid8</code> divisé par le moteur de la ligne. Les lignes marquées « non comparable à qualité égale » n'ont pas la même qualité que le Q4_0 (voir le tableau de qualité).</p>
{debit_table()}
<h3>Débit sur Ryzen 9 9900X (Zen 5, 12 fils)</h3>
<p>Ryzen 9 9900X : AMD Zen 5, AVX-512, 24 fils matériels ; même script adapté (<code>run_q4_0_pure.sh</code>, dans <code>debit-ryzen-9-9900x/</code>), 12 fils comme le banc Ryzen 9 9900X d'hier, même fichier Q4_0 pur (sha256 11d3f9d5…), mêmes six bras sous <code>~/bench-ryzen-9-9900x.lock</code>. <b>L'invite n'a pas de suffixe</b> (celle d'hier) : herbert y tourne avec <code>ignore_eos</code> (branche <code>serve-ignore-eos</code>, d24603d, sur b68ce37 : les jetons de fin passent à −∞ avant le tirage, comme dans llama-server). Les débits de Ryzen 9 9900X et de EPYC 4344P ne se comparent donc pas entre eux (machine, fils, invite).</p>
{debit_table("debit-ryzen-9-9900x")}
<p class=sub>Rappel seulement, banc d'hier (13 novembre, EPYC 4344P, autre fenêtre et invite sans la suite « 1. Le port de Marseille » ; hors comparaison), médianes :</p>
{hier_table()}

<h3>Conditions exactes</h3>
<ul>
<li>GGUF : <code>llama-quantize --pure</code> de llama.cpp 5676dfbe0 (construit sur Ryzen 9 9900X le 14 novembre), sans imatrix, depuis <code>~/models/Qwen3.6-35B-A3B-BF16.gguf</code>. Ce bf16 est identique au bit près au safetensors de Ryzen 9 9900X sur les tenseurs comparés (<code>poids-qwen36-ryzen-9-9900x.log</code>). Les mêmes tenseurs ont le même sha256 dans le safetensors de EPYC 4344P, celui que sert herbert (<code>poids-qwen36-epyc-4344p.log</code>).</li>
<li>Fichier sur EPYC 4344P : fabriqué sur place (convert_hf_to_gguf 5676dfbe0 en bf16, llama-quantize --pure, 55 s), puis remis sur l'en-tête et la disposition du fichier de Ryzen 9 9900X (le convert du jour ajoute la couche MTP <code>blk.40</code>, 20 tenseurs, inutilisée ; les 733 tenseurs communs sont identiques au bit, <code>tensor-sha-*.txt</code>) : sha256 du fichier entier identique à celui de Ryzen 9 9900X (<code>11d3f9d5…</code>). C'est donc le même fichier pour la qualité (Ryzen 9 9900X) et le débit (EPYC 4344P).</li>
<li>Débit Ryzen 9 9900X : script <code>debit-ryzen-9-9900x/run_q4_0_pure.sh</code> (vérifié : même fichier <code>~/q40pur/Qwen3.6-35B-A3B-Q4_0-pure.gguf</code>, <code>T=12</code>, <code>SUFFIX=''</code>, les deux bras herbert sur le binaire <code>~/herbert-ignore-eos</code> = branche serve-ignore-eos d24603d, ajoutée sur b68ce37), llama.cpp 5676dfbe0, ik_llama.cpp 7aa7e014, sous <code>~/bench-ryzen-9-9900x.lock</code>.</li>
<li>Qualité : Ryzen 9 9900X, références transformers 5.22 bf16 de la nuit, <code>_arms.sh</code> (12 fils), <code>_cmp.py</code>. herbert : <code>hkld</code> (logits de production).</li>
<li>Débit : EPYC 4344P ; herbert <code>herbert-jit-llama-serve --threads 8 --ctx 8192</code>, binaire du 13 novembre 08:36 UTC (branche locale banc-epyc-4344p-1113, b68ce37 d'après le banc d'hier ; <code>~/herbert-banc</code> n'est pas un dépôt git sur EPYC 4344P, le commit n'y est pas vérifiable) ; llama.cpp 5676dfbe0 et ik_llama.cpp 7aa7e014 : <code>llama-server -t 8 -c 8192 -np 1</code>.</li>
</ul>

<h3>Prédictions écrites avant mesure, et leur sort</h3>
{pred_block()}

<h3>Réserves</h3>
<ul>
<li>La qualité vient de Ryzen 9 9900X (AVX-512 Zen 5), le débit de EPYC 4344P (Zen 4). Les voies AVX-512 des deux machines sont les mêmes pour herbert, mais la qualité n'a pas été remesurée sur epyc-4344p.</li>
<li>Activations : llama.cpp multiplie Q4_0 par des activations Q8_0 (blocs de 32, échelle f16) ; ik_llama.cpp utilise ses propres formats 8 bits d'activations (non vérifiés ici, a fortiori avec <code>-rtr</code>) ; herbert quantifie en int8 par groupes de 32 avec échelle f32. Échelles des poids : f16 dans Q4_0, f32 chez herbert (même règle max signé / −8 en grille 8).</li>
<li>Les bras herbert de qualité datent de la nuit (même binaire <code>hkld</code>, même référence) et n'ont pas été rejoués.</li>
<li>Formats encore différents entre herbert et le Q4_0 pur : voir le tableau des types (lignes surlignées).</li>
</ul>
{exemples_block()}
{q4bf16_block()}
{reponses_block()}
{long_block()}
{code_block()}
</body></html>
"""
open(OUT, "w").write(page)
print("écrit", OUT)
