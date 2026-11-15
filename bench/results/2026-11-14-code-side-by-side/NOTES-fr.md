# Du code côte à côte : Qwen3.6-35B-A3B, référence bf16, herbert Q4_0, llama.cpp Q4_0 pur et Q4_K_M, ik Q4_0 pur, noté par exécution (14 novembre 2026)

But : « un petit test avec du code pour voir ». C'est le seul cas où « juste ou faux » ne dépend pas d'un juge : on exécute. La campagne s'ajoute à « réponses entières côte à côte » (`../2026-11-14-answers-side-by-side/`), avec les mêmes bras et les mêmes scripts. Dix tâches, consignes en français : sept en Python, trois en Rust (`taches.json`, écrit par `_taches.py`). Tests cachés dans `tests/`, jamais montrés au modèle ; réponses attendues des deux tâches « prédire » dans `attendus.json`.

Antériorité : les tâches, les tests, les attendus, mes solutions de contrôle (`_solutions/`) et `PREDICTIONS.md` sont hachés dans `SHA256SUMS` et commités à 19:59:29, heure de Paris. `PREDICTIONS.md` a été copié sur Ryzen 9 9900X à 19:59:18 (même sha256, 4842d81a…). Son en-tête dit 20:05 : c'était une estimation écrite avant le hachage, le fichier n'a pas été retouché. Les scripts de génération et les règles de notation (`_resultats.py` : extraction, « prédire », « réussit ») sont hachés dans `SHA256SUMS-2` et commités à 20:05:55. La génération a démarré à 20:24:18. Une seule modification après coup : `_attente.sh` (le guet avant la prise du verrou) a été réécrit deux fois, la seconde à 20:10 : la référence f32 qu'il fallait laisser passer d'abord avait été annulée. Le script ne touche ni la génération ni la notation ; les trois versions ont leur journal dans `logs-attente-{1,2,3}.out`.

Les tests ont été validés avant toute génération dans le même bac à sable. Mes solutions passent tout. Le code fourni par les consignes échoue : p03 dichotomie bogue, p04 médiane bogue, p05 squelette, r02 inversion par octets (2 tests sur 6), r03 squelette (ne compile pas).

## Protocole

- Invites : gabarit de conversation du modèle, `enable_thinking=False` (`_render.py` → `prompts.jsonl`). Glouton, **500 jetons au plus**, arrêt à la fin de séquence. La consigne impose le nom et la signature, et finit par « Réponds par le code seul, dans un seul bloc ```python (ou ```rust), sans explication ». Pour « prédire », elle demande « la sortie exacte seule, dans un seul bloc de code ».
- Mêmes entrées : `prompt_ok=True` et `tokenize_matches=True` pour les cinq bras et les dix tâches (`logs/gen-*.log`). ik ne rend pas les ids : on les reconstruit depuis les morceaux (`_ik_ids.py`), sans aucun morceau illisible, et les ids redonnent le texte sur les dix tâches (`logs/ik-ids.log`).
- Bras, tous sur Ryzen 9 9900X (Ryzen 9 9900X, Zen 5, AVX-512, 12 fils), sous `~/bench-ryzen-9-9900x.lock` après la fin du serveur du port 8620 (`_attente.sh`, `_chaine_ryzen-9-9900x.sh`, ports 32481 à 32484, de 20:24 à 20:36) :
  - référence transformers 5.22.1 bf16 ;
  - herbert-jit-llama-serve de `git archive 68c3a96` (`~/publication-jit-q4-rejeu`), `--q4`, grille 8 par défaut ;
  - llama.cpp 5676dfbe0 sur `Qwen3.6-35B-A3B-Q4_0-pure.gguf` ;
  - ik_llama.cpp 7aa7e014 sur le même fichier ;
  - llama.cpp 5676dfbe0 sur `Qwen3.6-35B-A3B-Q4_K_M.gguf`.
- Dépouillement : `_analyse.py` (copie de celui des réponses entières, coupe à 500) → `analyse.json`. Les textes de tous les bras sont décodés par le même tokenizer, fins de séquence retirées.
- Notation (`_resultats.py`, `_preparer.py`, `_executer.sh`, `_lanceur.py`, `_lanceur_rs.sh`) :
  - extraction du premier bloc ``` de la réponse ;
  - « prédire » : contenu du bloc, `strip()`, égal à l'attendu ;
  - Python : `py_compile`, import en moins de 10 s, chaque test caché avec 10 s ;
  - Rust : crate jetable sans dépendance, `cargo test --offline --no-run`, puis chaque test seul avec 10 s ;
  - « réussit » : bloc fermé et tous les tests passés.
- **Isolation** : sur **Ryzen 7 5700G**, au repos (charge 0,00, rien de mesuré), pas sur ryzen-9-9900x. Chaque (bras, tâche) passe dans un conteneur podman 4.9.3 jetable :
  - images `python:3.12-slim` et `rust:1.82-slim` ;
  - `--network none` ;
  - aucun montage du home, seul le dossier de la tâche monté en lecture seule sur `/task` ;
  - `--read-only`, avec des tmpfs `/tmp` (64 Mo) et `/work` (768 Mo, pour la compilation Rust) ;
  - `--memory 1g --memory-swap 1g`, `--pids-limit 128`, `--cpus 2` ;
  - `--user 65534:65534`, `--cap-drop ALL`, `--security-opt no-new-privileges`.

  Sonde (`logs/sonde-isolation.log`) : uid 65534, `/home` vide, réseau injoignable (Errno 101), écriture refusée (système en lecture seule), une allocation de 2 Go tuée (code 137). Aucun conteneur ne reste après les passes (`podman ps`). Aucun code généré n'a tourné sur le Mac ni hors du conteneur.

## Résultats

Tests cachés passés sur le total. Le rang est celui du premier jeton où le bras quitte la référence, compté à partir de 1 ; « = » : réponse identique au jeton près. Tous les blocs sont extraits et fermés, et tout compile.

| tâche | langue | sorte | piège | réf. (jetons) | herbert Q4_0 | llama Q4_0 | ik Q4_0 | llama Q4_K_M | herbert / llama Q4_0 |
|---|---|---|---|---|---|---|---|---|---|
| p01 fusionner des intervalles | Python | écrire | liste vide, intervalles qui se touchent | 8/8 (180) | 8/8 ≠ 45 | 8/8 ≠ 45 | 8/8 ≠ 45 | 8/8 = | = |
| p02 slug d'URL | Python | écrire | accents, cédille | 8/8 (185) | 8/8 ≠ 11 | 8/8 ≠ 25 | 8/8 ≠ 25 | 8/8 ≠ 70 | ≠ 11 |
| p03 dichotomie | Python | corriger | | 8/8 (107) | 8/8 = | 8/8 = | 8/8 = | 8/8 = | = |
| p04 médiane paire | Python | corriger | liste vide | 7/7 (89) | 7/7 = | 7/7 = | 7/7 = | 7/7 = | = |
| p05 chiffres romains | Python | compléter | | 6/6 (150) | 6/6 ≠ 122 | 6/6 = | 6/6 = | 6/6 = | ≠ 122 |
| p06 défaut mutable | Python | prédire | | 1/1 (14) | 1/1 = | 1/1 = | 1/1 = | 1/1 = | = |
| p07 `round`, `//`, `%`, `0.1+0.2` | Python | prédire | | 1/1 (13) | 1/1 = | 1/1 = | 1/1 = | 1/1 = | = |
| r01 moyenne entière | Rust | écrire | tranche vide, dépassement | 8/8 (154) | 8/8 ≠ 77 | 8/8 ≠ 77 | 8/8 ≠ 85 | 8/8 ≠ 99 | ≠ 87 |
| r02 inversion par octets | Rust | corriger | Unicode sur plusieurs octets | 6/6 (28) | 6/6 = | 6/6 = | 6/6 = | 6/6 = | = |
| r03 compte de mots | Rust | compléter | égalités, accents | 5/5 (153) | 5/5 = | 5/5 ≠ 62 | 5/5 ≠ 62 | 5/5 = | ≠ 62 |

Ce que changent les écarts : rien au comportement.

- p01 : noms de variables et commentaires en français.
- p02 : NFD et filtre de la catégorie `Mn` (herbert, llama et ik Q4_0) au lieu de NFKD et `unicodedata.combining` (référence, Q4_K_M) ; une ligne vide de plus pour herbert ; des commentaires numérotés pour llama et ik.
- p05 : `valeur, symbole` au lieu de `value, symbol` (herbert).
- r01 :
  - herbert : des commentaires en plus ;
  - llama Q4_0 : une version plus courte ;
  - ik : un test de bornes en plus, qui ne peut jamais servir ;
  - Q4_K_M : un autre commentaire.
- r03 : `mot_minuscule` au lieu de `mot_min` (llama et ik Q4_0).

## Comptes

| sorte | tâches | réussies, référence | réussies, chaque bras 4 bits |
|---|---|---|---|
| écrire | 3 (p01, p02, r01) | 3 | 3 |
| corriger | 3 (p03, p04, r02) | 3 | 3 |
| compléter | 2 (p05, r03) | 2 | 2 |
| prédire | 2 (p06, p07) | 2 | 2 |

- 50 cases sur 50 réussies, 56 tests cachés et 2 sorties exactes par bras, tous passés.
- Une version 4 bits casse un code que la référence réussit : **0 cas**. L'inverse : 0 cas (la référence réussit tout).
- Quittent la référence au jeton près : herbert Q4_0 sur 4 tâches sur 10, llama.cpp Q4_0 sur 4, ik Q4_0 sur 4, Q4_K_M sur 2. Les cinq bras sont identiques sur 5 tâches (p03, p04, p06, p07, r02).
- herbert et llama.cpp Q4_0 identiques jusqu'au bout : 6 sur 10. Quand les deux quittent la référence (p01, p02, r01), ils le font au même rang avec le même jeton sur 2 des 3 cas (p01, r01).
- llama.cpp et ik Q4_0 (même fichier) identiques jusqu'au bout : 7 sur 10 ; ils se séparent sur p01, p02 et r01.
- Aucune réponse n'approche les 500 jetons : la plus longue fait 261 jetons.

**Dix tâches ne font pas une statistique.** Elles sont courtes, et les fonctions, connues (dichotomie, médiane, chiffres romains) ; aucune ne demande plus de vingt lignes. Le résultat dit seulement ceci : sur ces dix tâches, la quantification change la forme du code et pas son comportement. Il ne dit rien des programmes longs, des bibliothèques peu connues ni d'un taux d'erreur.

## Prédictions, et leur sort

- Par tâche, « les cinq réussissent » sur p01 à p06, r02 et r03 : juste. p07, « au moins un bras quantifié se trompe » : faux, les cinq ont juste. r01, donnée comme seconde candidate à casser : rien n'a cassé.
- 1, la référence réussit 10 sur 10 : juste.
- 2, chaque bras 4 bits réussit au moins 8 tâches, avec 0 à 2 cas de casse et 0 ou 1 cas inverse : juste (10 sur 10, 0 et 0).
- 3, les Q4_0 quittent la référence sur 7 à 10 tâches, le Q4_K_M sur 4 à 7 : **faux**, 4, 4 et 4, puis 2. Le code est plus stable que je ne le pensais : la moitié des tâches sont identiques partout, au jeton près. La seconde moitié de la prédiction est juste : les écarts sont des noms, des commentaires et des variantes équivalentes.
- 4, herbert et llama.cpp Q4_0 identiques sur 4 à 7 tâches, même point d'écart dans la majorité : juste (6 ; 2 sur 3).
- 5, ik s'écarte de llama.cpp Q4_0 sur au moins 3 tâches : juste, de justesse (3).
- 6, tous les blocs fermés, tout compile, au plus une erreur Rust : juste (0).
- 7, les tâches « prédire » sont tranchées de la même façon par herbert et llama.cpp Q4_0 : juste (réponses identiques au jeton près).

## Réserves

- Dix tâches, un seul tirage glouton par bras : c'est un échantillon, pas une loi.
- L'image `rust:1.82-slim` date de 2024. Elle n'a gêné aucune réponse : tout compile.
- p01 compare des tuples : une réponse qui rendrait des listes de listes échouerait sans être fausse sur le fond. Le cas ne se présente pas.
- Le décompte « quitte la référence » compte un saut de ligne ou un nom de variable comme un écart. Il mesure la stabilité du texte, pas celle du code.

## Fichiers

- Préenregistrement : `taches.json`, `tests/`, `attendus.json`, `_solutions/`, `PREDICTIONS.md`, `SHA256SUMS`, `SHA256SUMS-2`.
- Invites : `prompts.jsonl` (texte rendu et ids).
- Sorties : `out/ref-tf-bf16.jsonl`, `out/herbert-q40.jsonl`, `out/llama-q40pur.jsonl`, `out/ik-q40pur.jsonl` (et `ik-q40pur.raw.jsonl`, les morceaux), `out/llama-q4km.jsonl` (ids générés, ids d'entrée).
- Dépouillement : `analyse.json` (textes, points d'écart, paires), `reponses/<bras>.json` (textes notés).
- Exécution : `exec/<bras>/<tâche>/` (code extrait, tests, lanceur) et `exec/<bras>/<tâche>.log` (sortie complète du conteneur), `exec/<bras>/extraction.json`. Résultats dans `resultats.json`.
- Section de la page : `jugements.json`, `_section_code.py`.
- Scripts : `_taches.py`, `_render.py`, `_srv_gen.py`, `_ref_gen.py`, `_ik_ids.py`, `_chaine_ryzen-9-9900x.sh`, `_attente.sh`, `_analyse.py`, `_preparer.py`, `_executer.sh`, `_lanceur.py`, `_lanceur_rs.sh`, `_resultats.py`.
- Logs : `logs/` (serveurs, générations, requêtes d'herbert avec `probs.json`, référence, ik, analyse, préparation, exécution sur Ryzen 7 5700G, relevé, contrôles du harnais, sonde d'isolation), `logs-attente-{1,2,3}.out`.
- Page : la section « Du code » est ajoutée à `2026-11-14-q40pur/_comparaison.py` (`code_block`, qui charge `_section_code.py`). La page est régénérée sans autre changement : diff de 107 lignes ajoutées, aucune retirée. Elle est copiée dans `~/work/comparaison-q4_0-2026-11-14.html`.
