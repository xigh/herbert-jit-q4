"""Écrit taches.json (les consignes données au modèle). Les tests cachés sont dans tests/, les attendus dans attendus.json.
_taches.py  (dans le dossier de la campagne)"""
import json

FIN_PY = "Réponds par le code seul, dans un seul bloc ```python, sans explication."
FIN_RS = "Réponds par le code seul, dans un seul bloc ```rust, sans explication ni fonction main."
FIN_OUT = "Réponds par la sortie exacte seule, dans un seul bloc de code, sans explication."

T = [
 dict(id="p01", langue="python", sorte="écrire", bord="liste vide, intervalles qui se touchent", fonction="fusionner_intervalles",
      texte="Écris en Python une fonction `fusionner_intervalles(intervalles: list[tuple[int, int]]) -> list[tuple[int, int]]`. "
            "Chaque intervalle (a, b) est fermé, avec a <= b. La fonction rend la liste des intervalles fusionnés, triée par début : "
            "deux intervalles qui se chevauchent ou se touchent (comme (1, 3) et (3, 5)) n'en font qu'un. "
            "La liste d'entrée peut être vide ou dans le désordre, et ne doit pas être modifiée. " + FIN_PY),
 dict(id="p02", langue="python", sorte="écrire", bord="Unicode (accents, cédille)", fonction="slug",
      texte="Écris en Python une fonction `slug(titre: str) -> str` qui fabrique un identifiant d'URL : tout en minuscules, "
            "lettres accentuées ramenées à leur lettre de base (é → e, ç → c, ï → i), chaque suite de caractères hors de a-z et 0-9 "
            "remplacée par un seul tiret, et aucun tiret au début ni à la fin. Par exemple, `slug(\"Les Misérables, tome 1 !\")` "
            "rend `\"les-miserables-tome-1\"`. N'utilise que la bibliothèque standard. " + FIN_PY),
 dict(id="p03", langue="python", sorte="corriger", bord="", fonction="recherche_dichotomique",
      texte="Cette fonction Python doit rendre l'indice de `x` dans la liste triée `liste`, ou -1 si `x` n'y est pas. "
            "Elle a un bogue. Corrige-la en gardant son nom et sa signature.\n\n```python\n"
            "def recherche_dichotomique(liste: list[int], x: int) -> int:\n"
            "    lo, hi = 0, len(liste) - 1\n"
            "    while lo < hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if liste[mid] == x:\n"
            "            return mid\n"
            "        elif liste[mid] < x:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return -1\n```\n\n" + FIN_PY),
 dict(id="p04", langue="python", sorte="corriger", bord="liste vide", fonction="mediane",
      texte="Cette fonction Python doit rendre la médiane d'une liste de nombres (la moyenne des deux valeurs du milieu si la "
            "longueur est paire), sans modifier la liste, et lever `ValueError` si la liste est vide. Elle a un bogue. "
            "Corrige-la en gardant son nom et sa signature.\n\n```python\n"
            "def mediane(nombres: list[float]) -> float:\n"
            "    if not nombres:\n"
            "        raise ValueError(\"liste vide\")\n"
            "    s = sorted(nombres)\n"
            "    n = len(s)\n"
            "    if n % 2 == 1:\n"
            "        return s[n // 2]\n"
            "    return (s[n // 2] + s[n // 2 + 1]) / 2\n```\n\n" + FIN_PY),
 dict(id="p05", langue="python", sorte="compléter", bord="", fonction="en_romain",
      texte="Complète cette fonction Python, qui écrit un entier de 1 à 3999 en chiffres romains (4 → \"IV\", 1994 → \"MCMXCIV\"). "
            "Garde son nom, sa signature et sa table.\n\n```python\n"
            "def en_romain(n: int) -> str:\n"
            "    table = [(1000, \"M\"), (900, \"CM\"), (500, \"D\"), (400, \"CD\"), (100, \"C\"), (90, \"XC\"),\n"
            "             (50, \"L\"), (40, \"XL\"), (10, \"X\"), (9, \"IX\"), (5, \"V\"), (4, \"IV\"), (1, \"I\")]\n"
            "    resultat = \"\"\n"
            "    # à compléter\n"
            "    return resultat\n```\n\n" + FIN_PY),
 dict(id="p06", langue="python", sorte="prédire", bord="argument par défaut mutable", fonction=None,
      texte="Qu'affiche ce programme Python ?\n\n```python\n"
            "def f(xs, acc=[]):\n"
            "    acc.append(len(xs))\n"
            "    return acc\n\n"
            "f(\"ab\")\n"
            "f([1, 2, 3])\n"
            "print(f(\"\"))\n```\n\n" + FIN_OUT),
 dict(id="p07", langue="python", sorte="prédire", bord="arrondi bancaire, division entière des négatifs, flottants", fonction=None,
      texte="Qu'affiche ce programme Python 3 ?\n\n```python\n"
            "print(round(2.5), round(3.5), -7 // 2, -7 % 3, 0.1 + 0.2 == 0.3)\n```\n\n" + FIN_OUT),
 dict(id="r01", langue="rust", sorte="écrire", bord="tranche vide, dépassement", fonction="moyenne_entiere",
      texte="Écris en Rust une fonction `pub fn moyenne_entiere(v: &[i32]) -> Option<i32>` qui rend la moyenne des valeurs, "
            "arrondie vers zéro (comme la division entière de Rust), ou `None` si la tranche est vide. "
            "Elle ne doit jamais déborder ni paniquer, même pour des valeurs proches de `i32::MAX` ou `i32::MIN`. "
            "Bibliothèque standard seulement. " + FIN_RS),
 dict(id="r02", langue="rust", sorte="corriger", bord="Unicode (caractères sur plusieurs octets)", fonction="inverser",
      texte="Cette fonction Rust doit rendre la chaîne avec ses caractères (`char`) dans l'ordre inverse. Elle a un bogue. "
            "Corrige-la en gardant son nom et sa signature.\n\n```rust\n"
            "pub fn inverser(s: &str) -> String {\n"
            "    let octets: Vec<u8> = s.bytes().rev().collect();\n"
            "    String::from_utf8(octets).unwrap()\n"
            "}\n```\n\n" + FIN_RS),
 dict(id="r03", langue="rust", sorte="compléter", bord="égalités départagées, minuscules accentuées", fonction="compter_mots",
      texte="Complète cette fonction Rust. Elle découpe le texte sur les blancs (`split_whitespace`), met chaque mot en minuscules "
            "(`to_lowercase`) et rend les couples (mot, nombre d'occurrences), triés par nombre décroissant, puis par ordre "
            "alphabétique du mot (ordre des `String`) en cas d'égalité. Garde son nom et sa signature.\n\n```rust\n"
            "use std::collections::HashMap;\n\n"
            "pub fn compter_mots(texte: &str) -> Vec<(String, usize)> {\n"
            "    let mut compte: HashMap<String, usize> = HashMap::new();\n"
            "    // à compléter\n"
            "}\n```\n\n" + FIN_RS),
]
json.dump(T, open("taches.json", "w"), ensure_ascii=False, indent=1)
print(len(T), "tâches")
