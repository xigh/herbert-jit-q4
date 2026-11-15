# Prédictions avant mesure — réponses entières côte à côte, Qwen3.6-35B-A3B (14 novembre 2026, 18:10 heure de Paris)

Écrites avant toute génération : aucun bras lancé, aucune invite rendue. Jamais retouchées après. Questions dans `questions.json` (15 questions, en français). Ce que je sais en les écrivant : sur la perplexité (wiki, fr), le Q4_0 pur de llama.cpp et herbert grille 8 ont une KL de 0,072 à 0,077 contre la référence transformers bf16, le Q4_K_M de 0,019 ; je ne connais aucune génération gloutonne de Qwen3.6 en Q4 sur ces questions.

Protocole : glouton, gabarit du modèle sans réflexion, 200 jetons au plus, arrêt à la fin de séquence. Bras : référence transformers bf16 ; herbert 68c3a96 `--q4` (grille 8) ; llama.cpp 5676dfbe0 Q4_0 pur ; llama.cpp Q4_K_M ; ik_llama.cpp 7aa7e014 Q4_0 pur si le temps le permet. « Dévie » = au moins un jeton différent de la référence avant la fin.

## Par question

Notation : H = herbert Q4_0, L0 = llama.cpp Q4_0 pur, LK = llama.cpp Q4_K_M. « = » : identique à la référence ; « ≠ » : dévie.

- q01 (capitale de l'Australie, une phrase) : H =, L0 =, LK =. Fond inchangé.
- q02 (Bastille, une phrase) : H =, L0 =, LK =. Fond inchangé.
- q03 (tungstène) : H =, L0 =, LK =. Fond inchangé.
- q04 (17 × 24) : H =, L0 =, LK =. Fond inchangé (408 partout).
- q05 (train, 189 km à 84 km/h) : H ≠, L0 ≠, LK ≠ (plus tard que les Q4_0). Fond inchangé (16 h 25 partout).
- q06 (−25 % puis +25 %) : H ≠, L0 ≠, LK =. Fond inchangé (75 €).
- q07 (ciel bleu, trois phrases) : H ≠, L0 ≠, LK ≠. Fond inchangé.
- q08 (inflation, trois phrases) : H ≠, L0 ≠, LK ≠. Fond inchangé.
- q09 (cinq fleuves, un par ligne) : H =, L0 =, LK =. Fond inchangé.
- q10 (sommeil, trois puces) : H ≠, L0 ≠, LK =. Fond inchangé (mêmes sortes de conseils, formulés autrement).
- q11 (résumé Myriel, deux lignes) : H ≠, L0 ≠, LK ≠. Fond inchangé.
- q12 (résumé auberge, trois lignes) : H ≠, L0 ≠, LK ≠. Fond inchangé.
- q13 (JSON, plus grande ville du Canada) : H =, L0 =, LK =. Fond inchangé (Toronto).
- q14 (JSON extrait de l'incipit) : H =, L0 =, LK =. Fond inchangé.
- q15 (traduction) : H =, L0 =, LK =. Fond inchangé.

## Comptes et paires

1. Les deux Q4_0 dévient de la référence sur 7 à 9 questions sur 15 ; le Q4_K_M sur 4 à 6. Les réponses courtes (moins de 40 jetons) restent identiques à la référence pour tous les bras.
2. herbert et llama.cpp Q4_0 pur (mêmes poids au bit, calculs différents) ne sont identiques entre eux, jusqu'au bout, que sur les questions où les deux suivent la référence ; sur les réponses longues ils se séparent l'un de l'autre aussi. Ils quittent la référence au même jeton, avec le même jeton, sur 1 à 3 questions seulement.
3. Le cas « les deux Q4_0 dévient ensemble, le Q4_K_M suit la référence » existe, sur 1 à 3 questions.
4. Le fond change (fait, nombre, conseil ou élément du résumé différent, consigne de format non tenue) sur 0 à 2 questions, et seulement pour un Q4_0.
5. Quand un bras dévie, c'est plus tard que le 10e jeton de la réponse dans la plupart des cas (au moins 2 sur 3).
6. ik_llama.cpp Q4_0 pur (s'il est mesuré) dévie de llama.cpp Q4_0 pur sur au moins 3 questions : même fichier, pas les mêmes noyaux.
