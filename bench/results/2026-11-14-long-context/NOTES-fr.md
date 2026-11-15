# Un document long, puis une question : Qwen3.6-35B-A3B, référence bf16 contre herbert Q4_0, llama.cpp Q4_0 pur et Q4_K_M, ik Q4_0 pur

Demande de l'auteur : « un truc un peu long, genre une lecture d'un fichier, d'une partie de livre, H. G. Wells ou Edgar Poe en anglais, puis une question dessus », puis « question en français ou en anglais ». Suite de [2026-11-14-reponses-cote-a-cote](../2026-11-14-answers-side-by-side/NOTES-fr.md) : mêmes bras, mêmes binaires, mêmes modèles, même machine (Ryzen 9 9900X), mêmes scripts adaptés (150 jetons, contexte 8 192, ports neufs).

## Le texte

Project Gutenberg n° 12750, H. G. Wells, *The Stolen Bacillus and Other Incidents* (1895), https://www.gutenberg.org/cache/epub/12750/pg12750.txt, téléchargé le 14 novembre 2026 par `curl` : `source/pg12750.txt`, sha256 `0f0cb5785ced1cdc782cea3b504f58b46cad71eac58963c1da6ed3e47075ff3b`.

Deux nouvelles entières, extraites par `_extract.py` (fins de ligne CRLF ôtées, chaque paragraphe sur une ligne, rien d'autre retouché ; les `_italiques_` et appels de notes `[A]` restent) :

- texte court : « The Temptation of Harringay », `source/harringay.txt`, 1 920 mots, **2 689 jetons**, sha256 `df317cf9…3bd9b` ;
- texte long : « A Moth — Genus Novo », `source/moth.txt`, 3 994 mots, **5 249 jetons**, sha256 `1907d6a3…d155d17`.

Pourquoi : deux récits mineurs du recueil, peu cités, pour mesurer la lecture et non la mémoire du modèle (*The Stolen Bacillus* lui-même, plus connu, est écarté). Deux textes distincts plutôt qu'un texte coupé, pour que chaque question porte sur un récit complet. Écartés : « The Triumphs of a Taxidermist » (1 467 mots, la bonne longueur courte) pour une injure raciale au cinquième paragraphe, « A Deal in Ostriches » (1 832 mots) pour ses clichés antisémites et racistes. Les longueurs ne tombent pas sur la demande (environ 2 000 et 6 000) : 2 689 et 5 249 jetons, un rapport d'environ 2 seulement ; avec la consigne et le gabarit, les invites font 2 731 à 2 759 et 5 291 à 5 320 jetons.

## Questions, réponses attendues, prédictions

`_questions.py` construit `questions.json` : cinq questions par texte (un détail du début, un du milieu, un de la fin, une citation exacte, un résumé en deux phrases), chacune posée **en anglais et en français** sur le même texte anglais (consigne autour du texte dans la langue de la question, sans demander de langue de réponse) : 20 invites. `_render.py` (inchangé) les rend par le gabarit du modèle, réflexion coupée : `ryzen-9-9900x/prompts.jsonl`.

Écrits et hachés avant toute génération, en lecture seule ensuite :

- `ATTENDUES.md` (sha256 `9fa433f5…8b742`, `ATTENDUES.sha256`) : réponses attendues dans les deux langues et règles de jugement (contenu juste / partiel / faux ; langue notée à part ; tolérances des citations ; faits exigés des résumés) ;
- `PREDICTIONS.md` (sha256 `b111a16b…0c5`, `PREDICTIONS.sha256`) ;
- `questions.sha256` : questions, sources et scripts de construction.

La précision « français ou anglais » est arrivée avant le hachage : il n'y a qu'un fichier de réponses attendues, bilingue. Les fichiers hachés ont été écrits et hachés à 18 h 33 (heure de Paris) ; la chaîne des bras Q4 a démarré à 18 h 40 (`logs-chaine-q4.out`, 17:40:24 UTC), la référence à 19 h 12 ; ils n'ont été commités qu'avec les résultats. `shasum -a 256 -c` sur les trois `.sha256` : tous OK au moment du commit.

## Bras et conditions

Tous sur **Ryzen 9 9900X** (AMD Ryzen 9 9900X, Zen 5, AVX-512, 12 fils), sous `flock ~/bench-ryzen-9-9900x.lock`, l'un après l'autre :

| bras | binaire / modèle | port |
|---|---|---|
| référence | transformers 5.22.1 bf16, torch 2.9.1+cpu, `_ref_gen.py` (venv `tf522-lfm3`, celle de la campagne courte), `~/models/Qwen3.6-35B-A3B` | — |
| herbert Q4_0 (grille 8) | `~/publication-jit-q4-rejeu/…/herbert-jit-llama-serve` (68c3a96 hérité de la campagne courte, non vérifiable ici : le dossier n'est pas un dépôt git et le log du serveur ne donne pas de version ; même chemin que la campagne courte, sha256 `1483ab97…`), `--q4 --threads 12 --ctx 8192` | 32481 |
| llama.cpp Q4_0 pur | `llama.cpp-5676dfbe0/build/bin/llama-server` (sha256 `a7b2adbe…`), `~/q40pur/Qwen3.6-35B-A3B-Q4_0-pure.gguf`, `-t 12 -c 8192 -np 1` | 32482 |
| llama.cpp Q4_K_M | même serveur, `~/models/gguf/Qwen3.6-35B-A3B-Q4_K_M.gguf` | 32483 |
| ik_llama.cpp Q4_0 pur | `ik_llama.cpp/build/bin/llama-server` (7aa7e014, sha256 `172c366b…`), même fichier Q4_0 pur | 32484 |

Glouton, 150 jetons au plus, arrêt à la fin de séquence. **Entrées identiques au jeton** : les 80 requêtes des serveurs ont reçu exactement les ids de `prompts.jsonl` (`prompt_ok`, `tokenize_matches`), la référence les mêmes (vérifié par `_analyse.py`). **Aucune réutilisation de préfixe** : herbert « 0 reused from the cache » sur ses 20 requêtes, llama.cpp et ik `timings.prompt_n` plein. ik ne rend pas les ids générés : `_ik_ids.py` (inchangé) les retrouve depuis les morceaux, aucun illisible, le texte redonné à l'identique. La référence a un seul ajout à `_ref_gen.py` : un `LogitsProcessor` qui note l'heure de chaque pas sans toucher aux logits, pour séparer prefill et decode.

Ryzen 9 9900X était partagé pendant les passages (le serveur Qwen3.6 du port 8620, les compilations et tests de l'emballage, charge 7 à 12) : les temps ci-dessous ne sont pas un banc.

Horaires (heure de Paris) : bras Q4 de 18 h 40 à 19 h 06 ; référence de 19 h 12 à 19 h 41.

## Résultats

Tableau par invite (rang du premier jeton différent de la référence, compté à partir de 0 comme dans `analyse.json` ; « = » : identique jusqu'au bout ; puis mon jugement du contenu, J juste, P partiel, F faux) :

| invite | réf. | herbert | llama.cpp Q4_0 | ik Q4_0 | Q4_K_M | herbert = llama Q4_0 |
|---|---|---|---|---|---|---|
| c1-en début | J | 0 · J | 0 · J | 0 · J | 0 · J | oui |
| c1-fr | J | 12 · J | 12 · J | 12 · J | = · J | oui |
| c2-en milieu | J | 5 · J | = · J | = · J | = · J | non (5) |
| c2-fr | J | 0 · J | 0 · J | 0 · J | 16 · J | non (89, fin) |
| c3-en fin | J | 6 · J | 6 · J | 6 · J | 6 · J | non (34) |
| c3-fr | J | 28 · J | 28 · J | 28 · J | 84 · J | oui |
| c4-en citation | J | = · J | = · J | = · J | = · J | oui |
| c4-fr | J | 0 · J | 0 · J | 0 · J | = · J | oui |
| c5-en résumé | J | 8 · **P** | 37 · **P** | 8 · **P** | 0 · J | non (8) |
| c5-fr | J | 50 · J | 10 · J | 10 · J | 15 · J | non (10) |
| l1-en début | J | = · J | 12 · J | = · J | 12 · J | non (12) |
| l1-fr | J | 8 · J | 8 · J | 8 · J | = · J | non (34) |
| l2-en milieu | J | 28 · J | 28 · J | 28 · J | 28 · J | oui |
| l2-fr | J | 35 · J | 23 · J | 33 · J | = · J | non (23) |
| l3-en fin | J | 0 · J | 0 · J | 0 · J | 0 · J | oui |
| l3-fr | J | 26 · J | 26 · J | 26 · J | 22 · J | non (36) |
| l4-en citation | J | = · J | 0 · **P** | 0 · **P** | = · J | non (0) |
| l4-fr | **P** | = · **P** | = · **P** | = · **P** | = · **P** | oui |
| l5-en résumé | J | 0 · J | 0 · J | 0 · J | 1 · J | non (8) |
| l5-fr | J | 3 · J | 3 · J | 3 · J | 27 · J | non (50) |

Textes complets et ids : `ryzen-9-9900x/out/*.jsonl` ; points de divergence, jetons de part et d'autre, textes décodés : `analyse.json` et `ryzen-9-9900x/logs/analyse.log` ; jugements et remarques par invite : `jugements.json` ; comptes : `comptes.json` (`_comptes.py`).

### Juste ou faux (mon jugement)

Le jugement est le mien, à la main, sur les règles hachées d'`ATTENDUES.md`. Deux cas y sont tranchés hors de leur lettre, la règle (« tout mot ajouté rend la citation fausse ») n'ayant pas prévu ces formes : la phrase citée suivie de son incise de dialogue (« …," said the picture. », « …," said Hapley, awkwardly. ») est jugée **juste** ; un passage de cinq phrases qui contient la phrase demandée (l4-fr partout, l4-en de llama.cpp et ik) est jugé **partiel**. À la lettre de la règle, les deux seraient faux, pour tous les bras, référence comprise.

| | juste | partiel | faux |
|---|---|---|---|
| référence bf16 | 19 | 1 | 0 |
| herbert Q4_0 | 18 | 2 | 0 |
| llama.cpp Q4_0 pur | 17 | 3 | 0 |
| ik_llama.cpp Q4_0 pur | 17 | 3 | 0 |
| llama.cpp Q4_K_M | 19 | 1 | 0 |

**À la lettre des règles hachées** (incise ou passage = citation fausse), les quatre citations c4-en, c4-fr, l4-en, l4-fr sont fausses pour tous les bras : référence 16 / 0 / 4, herbert 15 / 1 / 4, llama.cpp Q4_0 15 / 1 / 4, ik 15 / 1 / 4, Q4_K_M 16 / 0 / 4 ; la différence d'herbert et de Q4_K_M avec llama.cpp et ik sur l4-en disparaît alors. « Aucune réponse fausse » ne vaut que sous ma lecture.

Autre décision de jugement : « vermilion (or bright red) » (c2) est jugé juste, alors qu'`ATTENDUES.md` dit que le « bright red » passé ensuite en travers du tableau est un autre geste ; compté comme ajout faux, c2 serait partiel pour la référence (en, fr), herbert (en, fr), llama.cpp et ik (en, fr) et Q4_K_M (en).

Par longueur (juste sur 10, court / long) : référence 10 / 9 ; herbert 9 / 9 ; llama.cpp Q4_0 9 / 8 ; ik 9 / 8 ; Q4_K_M 10 / 9.

Tous les détails ponctuels (dix heures, vermillon, 24 par 20 vert pâle, 1891, troisième jour, chambre capitonnée et fantôme de Pawkins) sont justes pour les cinq bras dans les deux langues. Les écarts : le résumé court en anglais (c5-en), où les trois Q4_0 parlent d'une critique de son inspiration et oublient le marché du diable que la référence et Q4_K_M donnent ; la citation l4-en, où llama.cpp et ik Q4_0 (même fichier) partent dès le premier jeton sur le passage entier, alors que herbert et Q4_K_M rendent la référence au jeton près ; la citation l4-fr, passage entier pour les cinq bras, référence comprise.

**Langue** : les réponses aux 10 questions françaises sont en français pour les cinq bras, sauf les deux citations (c4-fr, l4-fr), rendues en anglais seul, sans traduction, partout ; je compte ces deux réponses comme « pas en français ».

### L'écart grandit-il avec le contexte ?

Pas de croissance visible, avec une réserve qui limite la portée : les deux longueurs sont deux récits et deux jeux de questions différents, 10 invites chacun, pour un rapport de longueur de 2 seulement (2 700 contre 5 300 jetons) ; ce n'est pas une mesure de l'effet de la longueur seule. Invites où le bras quitte la référence, court / long, sur 10 : herbert 9 / 7, llama.cpp Q4_0 8 / 9, ik 8 / 8, Q4_K_M 6 / 6 ; médianes du rang du premier jeton différent (depuis 0) : herbert 6 / 8, llama.cpp 8 / 8, ik 7 / 5,5, Q4_K_M 10,5 / 17. Les écarts tombent dans les premiers mots de la réponse et touchent la tournure (« Based on the text » contre « According to the text », « studio » contre « atelier », « a publié » contre « publie »), pas le fait demandé ; plusieurs sont partagés par les quatre bras au même jeton (c1-en, c3-en, l2-en, l3-en). Le fond ne change qu'à deux endroits, un par longueur.

### herbert et llama.cpp en Q4_0

Identiques jusqu'au bout sur 8 invites sur 20 (5 courtes, 3 longues) ; sur 6 des 12 autres, ils quittent la référence au même rang avec le même jeton, puis se séparent plus loin. llama.cpp et ik, sur le même fichier, sont identiques sur 12 (7 et 5) ; herbert et ik sur 8. Deux fois herbert suit la référence jusqu'au bout là où llama.cpp la quitte (l1-en, sans effet sur le fond, ik y suit aussi la référence ; l4-en, la citation, où llama.cpp et ik la quittent et où cela change le jugement), une fois l'inverse (c2-en, sans effet sur le fond).

### Temps

Serveurs (médianes rendues par les serveurs, jetons/s), prefill et decode, court puis long :

| bras | prefill court | prefill long | decode court | decode long |
|---|---|---|---|---|
| herbert Q4_0 | 406 | 399 | 15,84 | 15,21 |
| llama.cpp Q4_0 pur | 254 | 240 | 17,17 | 16,46 |
| ik_llama.cpp Q4_0 pur | 491 | 475 | 21,10 | 20,50 |
| llama.cpp Q4_K_M | 226 | 210 | 12,87 | 12,51 |

**Référence transformers bf16, prefill mesuré** (premier point au-delà de 257 jetons ; heure du premier appel du `LogitsProcessor` moins l'heure d'entrée dans `generate`) :

- texte court, 2 731 à 2 759 jetons : 204,3 s pour la première invite (la première du processus ; cause non mesurée) puis 26,5 ; 15,2 ; 16,1 ; 14,6 ; 15,8 ; 27,4 ; 37,4 ; 38,7 ; 33,7 s, soit **15 à 39 s, 70 à 188 jetons/s**, médiane 27,0 s (premier passage à froid compris) ;
- texte long, 5 291 à 5 320 jetons : 77,0 ; 72,6 ; 48,0 ; 82,5 ; 63,1 ; 74,9 ; 63,4 ; 82,1 ; 64,6 ; 77,3 s, soit **48 à 83 s, 64 à 110 jetons/s**, médiane 73,8 s ;
- decode : 0,25 à 0,77 s par jeton sur le court, 0,52 à 0,91 s sur le long ; 10 min 24 s pour les 10 invites courtes, 18 min 34 s pour les 10 longues, 29 min en tout.

La variation d'une invite à l'autre (×2 à ×2,5 sur une même longueur) n'a pas été expliquée ; Ryzen 9 9900X était chargé par d'autres travaux pendant le passage. Mon estimation avant lancement (texte long : 10 min à 1 h 05 ; les 20 invites : 15 min à 1 h 40) tenait ; la réalité est en bas de la fourchette.

## Prédictions : ce qui était juste ou faux

`PREDICTIONS.md`, onze prédictions :

1. Référence juste sur au moins 18/20, erreurs éventuelles sur une citation française ou un résumé : **juste** (19/20, partiel sur l4-fr).
2. Chaque Q4 juste sur au moins 16/20, détails ponctuels justes partout, erreurs sur citations ou résumés : **juste** (17 à 19).
3. Aucun Q4 au-dessus de la référence : **juste** (Q4_K_M à égalité, 19).
4. Q4_K_M au plus autant d'erreurs que les Q4_0 : **juste** (1 contre 2 et 3).
5. Toutes les réponses françaises en français : **faux** (les deux citations françaises en anglais seul, cinq bras) ; la phrase anglaise donnée dans au moins la moitié des citations françaises : juste.
6. Chaque Q4 quitte la référence sur au moins la moitié des invites : **juste** (12 à 17 sur 20) ; plus rarement sur les réponses courtes que sur les résumés : **faux** (herbert quitte la référence sur 11 des 12 questions de détail, souvent au premier mot).
7. Q4_K_M la quitte moins souvent que les trois Q4_0 : **juste** (12 contre 16, 17, 16).
8. L'écart grandit avec le contexte (60 %) : **faux** (9/7, 8/9, 8/8, 6/6 ; médianes sans tendance commune).
9. herbert et llama.cpp Q4_0 diffèrent sur au moins 4 invites, et sont identiques entre eux plus souvent que chacun avec la référence : **juste** (12 différences ; 8 identités entre eux contre 4 et 3 avec la référence).
10. ik plus proche de llama.cpp que de herbert : **juste** (12 contre 8).
11. Prefill des Q4 entre 80 et 250 jetons/s sur 5 300 jetons : **faux** pour herbert (399) et ik (475), juste pour llama.cpp (240) et Q4_K_M (210) ; decode plus lent de 5 à 25 % sur le long : **faux** pour les quatre (−4,0 % herbert, −4,1 % llama.cpp, −2,8 % ik, −2,8 % Q4_K_M ; tous sous 5 %).

## Fichiers

- `source/` : le texte Gutenberg et les deux nouvelles extraites ;
- `_extract.py`, `_questions.py`, `questions.json`, `questions.sha256` ;
- `ATTENDUES.md` / `.sha256`, `PREDICTIONS.md` / `.sha256` ;
- `ryzen-9-9900x/` : copie du dossier de travail `~/long-1114` de Ryzen 9 9900X : `_chaine_q4.sh`, `_chaine_ref.sh`, `_srv_gen.py`, `_ref_gen.py`, `_render.py`, `_ik_ids.py`, `_analyse.py`, `prompts.jsonl`, `out/` (ids et textes de chaque bras), `logs/` (serveurs, générations, requêtes d'herbert avec leurs probabilités, référence, analyses), `logs-chaine-q4.out`, `logs-chaine-ref.out`, `analyse-sans-ref.json` (paires seulement, avant la référence), `llama.log` (écrit par ik dans le dossier courant) ;
- copies à la racine des scripts envoyés sur Ryzen 9 9900X, plus `_comptes.py`, `analyse.json` (avec la référence), `comptes.json`, `jugements.json` ;
- section « Un document long, puis une question » de la page de comparaison Q4_0 : `../2026-11-14-q4_0-pure/comparaison.py`, fonction `long_block`, page régénérée (seule la section s'ajoute) et copiée dans `~/work/comparaison-q4_0-2026-11-14.html`. Le tableau de la page compte les rangs à partir de 1.
