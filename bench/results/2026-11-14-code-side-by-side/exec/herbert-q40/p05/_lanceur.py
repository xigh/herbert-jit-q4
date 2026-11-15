"""Lanceur dans le conteneur : compile solution.py, l'importe, passe chaque test_* de tests.py, 10 s chacun.
Rend un JSON sur la dernière ligne de la sortie. python -I /task/_lanceur.py"""
import sys, json, signal, py_compile, importlib.util, traceback
R = {"compile": False, "charge": False, "tests": {}, "erreur": None}
class Temps(Exception): pass
def alarme(*_): raise Temps("plus de 10 s")
signal.signal(signal.SIGALRM, alarme)
def fin():
    sys.stdout.flush(); print("\n@@RESULTAT@@" + json.dumps(R, ensure_ascii=False)); sys.exit(0)
try:
    py_compile.compile("/task/solution.py", cfile="/tmp/sol.pyc", doraise=True); R["compile"] = True
except BaseException as e:
    R["erreur"] = "compile: " + repr(e)[:500]; fin()
def charge(nom, chemin):
    spec = importlib.util.spec_from_file_location(nom, chemin); m = importlib.util.module_from_spec(spec)
    sys.modules[nom] = m; spec.loader.exec_module(m); return m
try:
    signal.alarm(10); sol = charge("solution", "/task/solution.py"); signal.alarm(0); R["charge"] = True
except BaseException as e:
    signal.alarm(0); R["erreur"] = "chargement: " + repr(e)[:500]; fin()
T = charge("tests_caches", "/task/tests.py")
for nom in [n for n in vars(T) if n.startswith("test_")]:
    try:
        signal.alarm(10); getattr(T, nom)(sol); signal.alarm(0); R["tests"][nom] = "ok"
    except BaseException as e:
        signal.alarm(0); R["tests"][nom] = "échec: " + repr(e)[:300]
fin()
