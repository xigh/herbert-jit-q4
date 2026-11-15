"""Section « Du code » de la page de comparaison : lue par _comparaison.py (2026-11-14-q40pur) via code_block().
Données : resultats.json, analyse.json, jugements.json (textes : intro, exemples, comptes) de ce dossier."""
import os, json, html
E = html.escape
H = os.path.dirname(os.path.abspath(__file__))
BRAS = [("herbert-q40", "herbert Q4_0 (grille 8)"), ("llama-q40pur", "llama.cpp Q4_0 pur"), ("ik-q40pur", "ik_llama.cpp Q4_0 pur"),
        ("llama-q4km", "llama.cpp Q4_K_M")]


def _load(n):
    p = os.path.join(H, n)
    return json.load(open(p)) if os.path.exists(p) else None


def _cell(c, ref=False):
    if c is None:
        return "—"
    s = f"<span class={'ok' if c['reussit'] else 'ko'}>{c['passes']}/{c['total']}</span>"
    if c["bloc"] != "fermé":
        s += f" (bloc {c['bloc']})"
    if c.get("compile") is False:
        s += " (ne compile pas)"
    if not ref:
        s += " · " + ("identique" if c.get("div") is None else f"jeton {c['div'] + 1}")
    return s


def _txt(c):
    if c["div"] is None:
        return E(c["texte"])
    s = c["suite"]; j = c.get("jeton") or ""
    head = s[:len(j)] if j and s.startswith(j) else ""
    return E(c["prefixe"]) + "<span class=dv>" + (f"<b class=dv1>{E(head)}</b>" if head else "") + E(s[len(head):]) + "</span>"


def code_block():
    R = _load("resultats.json"); A = _load("analyse.json"); J = _load("jugements.json")
    if not R or not A or not J:
        return ""
    T = {t["id"]: t for t in json.load(open(os.path.join(H, "taches.json")))}
    byid = {r["id"]: r for r in A}
    out = [f"<h2>Du code</h2>\n<p>{J['intro']}</p>"]
    head = "".join(f"<th>{n}</th>" for _, n in BRAS)
    t = ""
    for i, tc in T.items():
        r = R[i]; p = r["paires"].get("herbert-q40|llama-q40pur", {})
        t += (f"<tr><td>{i}</td><td>{tc['langue']}</td><td>{E(tc['sorte'])}</td><td>{E(J['court'].get(i, ''))}</td>"
              f"<td class=n>{_cell(r['ref-tf-bf16'], True)}</td>" + "".join(f"<td class=n>{_cell(r.get(a))}</td>" for a, _ in BRAS) +
              f"<td class=n>{'identique' if p.get('div') is None else 'jeton ' + str(p['div'] + 1)}</td></tr>")
    out.append(f"<p>{J['tableau_intro']}</p>")
    out.append(f"<table><tr><th></th><th>langue</th><th>sorte</th><th>tâche</th><th>référence 16 bits</th>{head}<th>herbert contre llama.cpp Q4_0</th></tr>{t}</table>")
    out.append(f"<p>{J['comptes']}</p>")
    for ex in J.get("exemples", []):
        r = byid[ex["id"]]
        out.append(f"<h4>{E(ex['titre'])}</h4>")
        out.append(f"<blockquote>{E(r['question']).replace(chr(10), '<br>')}</blockquote>")
        cols = [f"<div class=rep><div class=rh>référence 16 bits (transformers bf16) · {_cell(R[ex['id']]['ref-tf-bf16'], True)}</div>{E(r['ref']['texte'])}</div>"]
        for a, n in BRAS:
            if a in ex.get("bras", [a for a, _ in BRAS]) and a in r["bras"]:
                cols.append(f"<div class=rep><div class=rh>{n} · {_cell(R[ex['id']].get(a))}</div>{_txt(r['bras'][a])}</div>")
        out.append("<div class=cols>" + "".join(cols) + "</div>")
        out.append(f"<p>{ex['commentaire']}</p>")
    out.append("<p class=src>Données : <code>2026-11-14-code-cote-a-cote/</code> (tâches, tests cachés, attendus, <code>PREDICTIONS.md</code> et <code>SHA256SUMS</code>, ids et textes de chaque bras dans <code>out/</code>, résultats d'exécution dans <code>resultats.json</code> et <code>exec/</code>).</p>")
    return "\n".join(out)
