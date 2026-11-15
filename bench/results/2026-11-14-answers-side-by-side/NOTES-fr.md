# Réponses entières côte à côte : Qwen3.6-35B-A3B, référence bf16, herbert Q4_0, llama.cpp Q4_0 pur et Q4_K_M, ik Q4_0 pur (14 novembre 2026)

But : montrer ce que la quantification change sur des réponses entières, à format égal (Q4_0 contre Q4_0) et contre le Q4_K_M, pour l'article « comparer ce qui est comparable ». Quinze questions courtes en français, écrites avec les prédictions avant toute génération (`questions.json`, `PREDICTIONS.md`, sha256 dans `*.sha256`). Preuve d'antériorité : `PREDICTIONS.md` est copié sur Ryzen 9 9900X à 18:04:51 heure de Paris (`stat`, même sha256 2087847e…) ; la chaîne démarre à 18:05:00 (`logs-chaine.out`) ; le commit date de 18:06:42, pendant le chargement du premier serveur (prêt à 18:06:44), avant toute réponse. L'en-tête du fichier dit 18:10 : une estimation écrite avant le commit.

## Protocole

- Invites : gabarit de conversation du modèle (`chat_template` de `~/models/Qwen3.6-35B-A3B`, transformers 5.22.1), `enable_thinking=False`, `add_generation_prompt=True` (`_render.py` → `prompts.jsonl`, texte rendu et ids ; les ids de `apply_chat_template(tokenize=True)` et du tokenizer sur le texte rendu sont identiques).
- Décodage glouton, 200 jetons au plus, arrêt à la fin de séquence (248046 `<|im_end|>`, 248044). Les fins de séquence sont retirées avant comparaison ; « coupée » = 200 jetons sans fin.
- Mêmes entrées : llama.cpp et ik reçoivent les ids (`/completion`, `prompt` en liste d'ids, `tokens_evaluated` vérifié) ; herbert reçoit le texte rendu (`/completion` prend du texte) et ses `prompt_ids` (lus dans le `probs.json` de `--log-dir`) sont comparés aux ids de référence ; en plus, `/tokenize` de chaque serveur sur le texte rendu redonne les mêmes ids. Résultat : **ids d'entrée identiques pour les cinq bras et les quinze questions** (`prompt_ok` et `tokenize_matches` dans `logs/gen-*.log` ; `_analyse.py` le vérifie encore).
- Ids générés gardés pour chaque bras (`out/*.jsonl`). ik_llama.cpp ne rend pas les ids : chaque pas rend le texte de son jeton (`completion_probabilities[].content`), ramené à son id par le vocabulaire byte-level du tokenizer (`_ik_ids.py`, aucun morceau illisible, les ids redonnent le texte de la réponse sur les quinze). Contrôle de la méthode : appliquée aux morceaux des réponses de llama.cpp Q4_0, elle redonne leurs ids, 15/15.
- Référence rejouée une seconde fois avec les logits de chaque pas (`_ref_marges.py`, `out/ref-tf-bf16-marges.jsonl`) : mêmes ids qu'au premier passage, 15/15 ; on y lit les deux premiers jetons de la référence au rang de chaque écart. 4 pas sur 931 sont des ex aequo exacts (deux jetons à la même probabilité) ; deux d'entre eux sont des points d'écart (q02 pour le Q4_K_M, q11 pour les trois Q4_0) : la référence y prend un jeton, le bras l'autre.

## Bras (tous sur Ryzen 9 9900X : AMD Ryzen 9 9900X, Zen 5, AVX-512, 12 fils, sous `~/bench-ryzen-9-9900x.lock`)

| bras | moteur | fichier | commande |
|---|---|---|---|
| référence | transformers 5.22.1, torch 2.9.1+cpu, bf16 (`~/venvs/tf522-lfm3`) | `~/models/Qwen3.6-35B-A3B` (safetensors) | `_ref_gen.py`, `generate(do_sample=False, max_new_tokens=200)` |
| herbert-q40 | herbert-jit-llama-serve, `git archive 68c3a96` (`~/publication-jit-q4-rejeu`, tar identique au bit à `git archive 68c3a96` du worktree, sha256 612f9c40…) | mêmes safetensors | `--q4 --threads 12 --ctx 4096 --log-dir …` (grille 8 par défaut dans 68c3a96) |
| llama-q40pur | llama.cpp 5676dfbe0 | `~/q40pur/Qwen3.6-35B-A3B-Q4_0-pure.gguf` | `llama-server -t 12 -c 4096 -np 1` |
| ik-q40pur | ik_llama.cpp 7aa7e014 | le même fichier | `llama-server -t 12 -c 4096 -np 1` |
| llama-q4km | llama.cpp 5676dfbe0 | `~/models/gguf/Qwen3.6-35B-A3B-Q4_K_M.gguf` | `llama-server -t 12 -c 4096 -np 1` |

Un autre serveur (port 8620, herbert, 12 fils) occupait la machine en même temps : le glouton ne dépend pas de la charge, seule la vitesse en souffre. Requêtes : `temperature 0, top_k 1, cache_prompt false` (herbert : en plus `presence_penalty 0, frequency_penalty 0`).

## Résultats

Rang du premier jeton différent de la référence (« = » : identique jusqu'à la fin de séquence, ou jusqu'à la coupe des 200 jetons pour q05). Le jugement « fond » ou « formulation » est le mien.

| q | question | jetons réf. | herbert Q4_0 | llama Q4_0 | ik Q4_0 | llama Q4_K_M | herbert / llama Q4_0 | fond (jugement) |
|---|---|---|---|---|---|---|---|---|
| q01 | capitale de l'Australie | 10 | = | = | = | = | = | identique partout |
| q02 | prise de la Bastille | 37 | = | = | = | ≠ 20 | = | Q4_K_M : formulation (« célébrée » pour « commémorée ») ; à un ex aequo exact de la référence (deux jetons à 41,3 %) |
| q03 | tungstène | 35 | = | = | = | = | = | identique partout |
| q04 | 17 × 24 | 3 | = | = | = | = | = | identique partout (408) |
| q05 | train, 189 km à 84 km/h | 200 (coupée) | ≠ 56 | ≠ 56 | ≠ 56 | ≠ 56 | = | formulation : même calcul (2,25 h = 2 h 15), tous coupés à 200 jetons avant l'heure d'arrivée, référence comprise |
| q06 | −25 % puis +25 % | 84 | ≠ 3 | ≠ 3 | ≠ 3 | ≠ 15 | = | formulation : 75 € partout |
| q07 | ciel bleu, trois phrases | 89 | ≠ 6 | ≠ 6 | ≠ 6 | ≠ 26 | ≠ 7 | formulation ; herbert écrit « molécles » (faute) et nomme la diffusion de Rayleigh |
| q08 | inflation, trois phrases | 78 | ≠ 30 | ≠ 30 | ≠ 5 | = | = | fond pour herbert et llama Q4_0 (l'indice des prix devient une formule circulaire) ; ik : formulation ; Q4_K_M identique |
| q09 | cinq fleuves | 15 | ≠ 14 | ≠ 14 | ≠ 14 | ≠ 15 | = | fond, en mieux : la référence cite deux fois la Loire ; Rhin (Q4_0), Somme (Q4_K_M) |
| q10 | sommeil, trois puces | 84 | ≠ 2 | ≠ 2 | ≠ 2 | ≠ 63 | ≠ 9 | formulation : les trois mêmes conseils ; puces « *␣␣␣ » pour les Q4_0 |
| q11 | résumé Myriel, deux lignes | 52 | ≠ 13 | ≠ 13 | ≠ 13 | ≠ 18 | ≠ 22 | fond, en mieux : la référence attribue la respectabilité à la servante, les quatre bras quantifiés à Baptistine ; les Q4_0 partent d'un ex aequo exact de la référence (deux jetons à 39,85 %) |
| q12 | résumé de l'auberge, trois lignes | 106 | ≠ 5 | ≠ 5 | ≠ 5 | ≠ 5 | ≠ 26 | fond, en mieux : la référence place Labarre à Grenoble, les quatre bras quantifiés à Digne (Croix-de-Colbas) |
| q13 | JSON, plus grande ville du Canada | 34 | ≠ 8 | ≠ 8 | ≠ 1 | ≠ 8 | ≠ 29 | fond, en pire : Montréal au lieu de Toronto, pour les quatre bras quantifiés |
| q14 | JSON extrait de l'incipit | 68 | ≠ 31 | ≠ 1 | ≠ 31 | = | ≠ 1 | formulation : « évêque de Digne » (herbert, ik) ; llama Q4_0 sans clôture ```json |
| q15 | traduction | 22 | ≠ 5 | ≠ 5 | ≠ 5 | = | ≠ 19 | formulation : « watched » pour « looked » (Q4_0) |

- herbert Q4_0 : dévie sur 11/15, identique sur 4 ; premier jeton différent après le 10e dans 5 des 11.
- llama Q4_0 : dévie sur 11/15, identique sur 4 ; premier jeton différent après le 10e dans 4 des 11.
- ik Q4_0 : dévie sur 11/15, identique sur 4 ; premier jeton différent après le 10e dans 4 des 11.
- llama Q4_K_M : dévie sur 9/15, identique sur 6 ; premier jeton différent après le 10e dans 7 des 9.
- herbert et llama Q4_0 identiques entre eux jusqu'au bout (ou la coupe) : 8/15 ; quand les deux dévient (11), même jeton au même rang : 10.
- llama Q4_0 et ik Q4_0 (même fichier) identiques jusqu'au bout : 7/15.
- Les deux Q4_0 dévient, Q4_K_M identique à la référence : 3 (q08, q14, q15) ; dont ensemble (même rang, même jeton) : 2 (q08, q15).
- Les cinq bras identiques : 3 (q01, q03, q04).
- Référence rejouée : identique au premier passage sur 15/15.

Ex aequo exacts de la référence au point d'écart : q02 (Q4_K_M, « célébr » et « comm » à 41,3 % chacun) et q11 (les trois Q4_0, « le » et « M » à 39,85 %). Hors ex aequo, le jeton du bras au point d'écart est l'un des deux premiers de la référence dans 32 cas sur 38 ; les 6 autres sont q05 et q06 pour les trois Q4_0.

## Trois exemples (section « Des réponses entières, côte à côte » de la page de comparaison)

1. Les deux Q4_0 dévient ensemble, le Q4_K_M suit la référence : **q08** (inflation). herbert et llama.cpp Q4_0 rendent la même réponse, jeton pour jeton, qui quitte la référence au 30e jeton (« des consommateurs », second choix de la référence à 16,5 % contre 74,0 %) et remplace l'indice des prix par une formule circulaire ; le Q4_K_M rend la référence au jeton près ; ik s'écarte dès le 5e jeton. Seul autre cas de cette sorte : q15 (« watched » pour « looked »).
2. Tout le monde dit la même chose : **q03** (tungstène, W, 74), cinq bras identiques. Aussi q01 et q04.
3. Le fond change : **q13**. Toronto devient Montréal au 8e jeton pour herbert, llama.cpp Q4_0 et aussi le Q4_K_M (la référence : 54,7 % contre 42,6 %), et ik aussi. Dans l'autre sens, sur q09, q11 et q12, c'est la référence qui se trompe (Loire citée deux fois ; respectabilité attribuée à la servante, à partir d'un ex aequo ; Labarre placé à Grenoble) et les quatre bras quantifiés non.

## Prédictions, et leur sort

- Par question, « = » ou « ≠ » pour herbert, llama Q4_0 et Q4_K_M : 31 cases justes sur 45. Faux surtout sur les réponses courtes que je croyais sûres : q09, q13, q15 et q14 dévient. Fond : j'avais prédit « inchangé » partout, il change sur 5 questions (10/15 justes).
- 1 (Q4_0 dévient sur 7 à 9, Q4_K_M sur 4 à 6, réponses de moins de 40 jetons identiques) : faux, 11, 9, et q09, q13, q15 dévient.
- 2 (herbert et llama Q4_0 se séparent sur les réponses longues, même point d'écart sur 1 à 3) : faux. Ils sont identiques jusqu'au bout sur 8 questions, et partent au même rang avec le même jeton sur 10 des 11 écarts.
- 3 (le cas « Q4_0 ensemble, Q4_K_M avec la référence » existe sur 1 à 3) : juste, 2 (q08, q15).
- 4 (fond changé sur 0 à 2, Q4_0 seulement) : faux. Il change sur 5 questions, dont 4 aussi pour le Q4_K_M ; 3 sont des corrections d'une référence fautive.
- 5 (écart après le 10e jeton dans au moins 2 cas sur 3) : faux pour les Q4_0 (5, 4, 4 sur 11), juste pour le Q4_K_M (7 sur 9).
- 6 (ik s'écarte de llama.cpp Q4_0 sur au moins 3 questions) : juste, 8.

## Réserves

- Quinze questions : un échantillon, pas une loi. Les comptes servent à ne pas présenter une exception comme la règle, pas à mesurer un taux.
- Le jugement de fond est le mien. q08 (« fond appauvri ») est discutable ; q09, q11 et q12 comptent comme des changements de fond, mais en mieux. Les différences de présentation n'y sont pas comptées : clôture ```json en plus (q13, ik) ou en moins (q14, llama.cpp Q4_0), saut de ligne entre les deux phrases du résumé q11. Ma prédiction 4 rangeait pourtant « consigne de format non tenue » dans le fond ; en les comptant, q14 s'ajouterait (6 questions).
- Gemma 4 E4B : pas tenté, faute de budget. Il n'y a pas de Q4_K_M de Gemma sur les machines, et le GGUF de Google est un Q4_0 QAT, pas un Q4_0 pur fait depuis les mêmes poids bf16 ; une référence transformers d'E4B existe déjà (`justesse-1113`, exemples de divergence).
- Les écarts herbert / llama.cpp en Q4_0 ne viennent pas des poids (identiques au bit en grille 8) mais des calculs (activations, ordre des sommes, embeddings bf16 chez herbert).

## Fichiers

- `questions.json`, `PREDICTIONS.md` et leurs sha256 ; `prompts.jsonl` (textes rendus et ids).
- `out/ref-tf-bf16.jsonl`, `out/herbert-q40.jsonl`, `out/llama-q40pur.jsonl`, `out/ik-q40pur.jsonl` (+ `ik-q40pur.raw.jsonl`, les morceaux), `out/llama-q4km.jsonl`, `out/ref-tf-bf16-marges.jsonl`.
- `analyse.json` (`_analyse.py`), `jugements.json` (mes jugements et textes de la page), `_table.py` → `_table.out`.
- Scripts : `_render.py`, `_ref_gen.py`, `_srv_gen.py`, `_ik_ids.py`, `_ref_marges.py`, `_chaine_ryzen-9-9900x.sh`, `_chaine_ik.sh`.
- Logs : `logs-chaine.out`, `logs-chaine-ik.out`, `logs-render.log`, `logs/` (serveurs, générations, requêtes d'herbert avec `probs.json`, référence, marges, analyse ; `logs/gen-ik-q40pur-passage1.log` : premier passage d'ik, arrêté sur l'absence d'ids).
- Page : section ajoutée à `2026-11-14-q40pur/_comparaison.py` (`reponses_block`), page régénérée dans `2026-11-14-q40pur/comparaison-q4_0-2026-11-14.html` et copiée dans `~/work/`.
