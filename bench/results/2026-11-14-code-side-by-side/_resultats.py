"""Notation par exécution. Deux temps :
  _resultats.py preparer analyse.json EXEC_DIR   : textes des cinq bras (ids décodés par le même tokenizer, fins de séquence
                                                    retirées, pris dans analyse.json) → reponses/<bras>.json → _preparer.py
  _resultats.py relever  analyse.json EXEC_DIR resultats.json : lit EXEC_DIR/<bras>/<tâche>.log et extraction.json.
Règles, fixées avant toute génération :
  - extraction : le premier bloc ``` … ``` de la réponse ; « non fermé » si le bloc s'ouvre sans se fermer (le code jusqu'à la
    fin est quand même exécuté, et compté à part) ; « absent » sans bloc ;
  - « prédire » : contenu du premier bloc, strip(), égal à attendus.json ; pas d'exécution ;
  - Python : « compile » = py_compile ; « charge » = import sans erreur en moins de 10 s ; tests cachés, 10 s chacun ;
  - Rust : « compile » = cargo test --offline --no-run ; chaque test seul, 10 s ;
  - « réussit » = bloc fermé et tous les tests passés (ou sortie exacte)."""
import sys, json, os, re, subprocess
H = os.path.dirname(os.path.abspath(__file__))
BRAS = ["ref-tf-bf16", "herbert-q40", "llama-q40pur", "ik-q40pur", "llama-q4km"]
T = {t["id"]: t for t in json.load(open(os.path.join(H, "taches.json")))}
ATT = json.load(open(os.path.join(H, "attendus.json")))
mode, AN, ED = sys.argv[1:4]
A = {r["id"]: r for r in json.load(open(AN))}
def texte(r, b):
    return r["ref"]["texte"] if b == "ref-tf-bf16" else r["bras"][b]["texte"]
if mode == "preparer":
    os.makedirs(os.path.join(H, "reponses"), exist_ok=True)
    for b in BRAS:
        R = {i: texte(r, b) for i, r in A.items() if b == "ref-tf-bf16" or b in r["bras"]}
        p = os.path.join(H, "reponses", b + ".json")
        json.dump(R, open(p, "w"), ensure_ascii=False, indent=1)
        subprocess.run([sys.executable, os.path.join(H, "_preparer.py"), b, p, ED], check=True)
    sys.exit(0)
OUT = sys.argv[4]
def ntests(i):
    s = open(os.path.join(H, "tests", i + (".py" if T[i]["langue"] == "python" else ".rs"))).read()
    return len(re.findall(r"^def test_", s, re.M)) if T[i]["langue"] == "python" else s.count("#[test]")
res = {}
for b in BRAS:
    X = json.load(open(os.path.join(ED, b, "extraction.json")))
    for i, t in T.items():
        x = X.get(i, {"bloc": "absent", "code": None}); c = {"bloc": x["bloc"], "code": x["code"]}
        if t["sorte"] == "prédire":
            c["compile"] = None
            c["sortie"] = None if x["code"] is None else x["code"].strip()
            c["passes"], c["total"] = int(c["sortie"] == ATT[i]), 1
        elif x["code"] is None:
            c.update(compile=None, passes=0, total=ntests(i))
        else:
            log = open(os.path.join(ED, b, i + ".log")).read(); c["log"] = log[-4000:]
            if t["langue"] == "python":
                m = re.search(r"@@RESULTAT@@(.*)", log)
                R = json.loads(m.group(1)) if m else {"compile": False, "charge": False, "tests": {}, "erreur": "pas de résultat (tué ?)"}
                c.update(compile=R["compile"], charge=R["charge"], erreur=R["erreur"], tests=R["tests"],
                         passes=sum(v == "ok" for v in R["tests"].values()), total=ntests(i))
            else:
                c["compile"] = "@@COMPILE@@ok" in log
                tt = dict(re.findall(r"@@TEST@@(\S+?)@@(\S+)", log))
                c.update(tests=tt, passes=sum(v == "ok" for v in tt.values()), total=ntests(i))
        c["reussit"] = c["bloc"] == "fermé" and c["passes"] == c["total"]
        a = A.get(i)
        if a is not None and b != "ref-tf-bf16" and b in a["bras"]:
            c["div"] = a["bras"][b]["div"]; c["etat"] = a["bras"][b]["etat"]; c["n"] = a["bras"][b]["n"]; c["fin"] = a["bras"][b]["fin"]
        elif a is not None:
            c["n"] = a["ref"]["n"]; c["fin"] = a["ref"]["fin"]
        res.setdefault(i, {})[b] = c
for i, a in A.items():
    res[i]["paires"] = a["paires"]
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
# Tableau compact
for i, t in T.items():
    r = res[i]
    cells = []
    for b in BRAS:
        c = r[b]
        s = ("✓" if c["reussit"] else "✗") + f" {c['passes']}/{c['total']}"
        if c["bloc"] != "fermé": s += f" [{c['bloc']}]"
        if c.get("compile") is False: s += " [ne compile pas]"
        if b != "ref-tf-bf16": s += " =" if c.get("div") is None else f" ≠{c['div']}"
        cells.append(s)
    hl = r["paires"].get("herbert-q40|llama-q40pur", {})
    print(f"{i} {t['langue']:6s} {t['sorte']:9s} | " + " | ".join(cells) + f" | H/L0 {'=' if hl.get('div') is None else '≠' + str(hl.get('div'))}")
