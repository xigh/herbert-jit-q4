"""Extrait le premier bloc de code de chaque réponse et prépare un dossier d'exécution par (bras, tâche).
_preparer.py BRAS REPONSES.json EXEC_DIR   (REPONSES.json : {id_tâche: texte de la réponse})
Écrit EXEC_DIR/BRAS/<id>/ (Python : solution.py, tests.py, _lanceur.py ; Rust : crate/) et EXEC_DIR/BRAS/extraction.json."""
import sys, json, os, re, shutil
bras, rep, ED = sys.argv[1:4]
H = os.path.dirname(os.path.abspath(__file__))
T = {t["id"]: t for t in json.load(open(os.path.join(H, "taches.json")))}
R = json.load(open(rep)); X = {}
def extrait(txt):
    m = re.search(r"```[^\n]*\n(.*?)```", txt, re.S)
    if m:
        return "fermé", m.group(1)
    m = re.search(r"```[^\n]*\n(.*)", txt, re.S)
    if m:
        return "non fermé", m.group(1)
    return "absent", None
for i, txt in R.items():
    t = T[i]; etat, code = extrait(txt); X[i] = {"bloc": etat, "code": code}
    if code is None or t["sorte"] == "prédire":
        continue
    d = os.path.join(ED, bras, i); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    if t["langue"] == "python":
        open(os.path.join(d, "solution.py"), "w").write(code)
        shutil.copy(os.path.join(H, "tests", i + ".py"), os.path.join(d, "tests.py"))
        shutil.copy(os.path.join(H, "_lanceur.py"), d)
    else:
        os.makedirs(os.path.join(d, "crate", "src"))
        open(os.path.join(d, "crate", "Cargo.toml"), "w").write('[package]\nname = "sol"\nversion = "0.1.0"\nedition = "2021"\n\n[lib]\npath = "src/lib.rs"\n\n[dependencies]\n')
        tests = open(os.path.join(H, "tests", i + ".rs")).read()
        open(os.path.join(d, "crate", "src", "lib.rs"), "w").write(code + "\n\n#[cfg(test)]\nmod tests_caches {\n    use super::*;\n" + tests + "}\n")
        shutil.copy(os.path.join(H, "_lanceur_rs.sh"), d)
os.makedirs(os.path.join(ED, bras), exist_ok=True)
json.dump(X, open(os.path.join(ED, bras, "extraction.json"), "w"), ensure_ascii=False, indent=1)
print(bras, {i: x["bloc"] for i, x in X.items()})
