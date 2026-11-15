# Prédictions avant mesure — justesse du Q4 d'herbert (13 novembre 2026, 20:55 UTC)

Écrites avant toute mesure, y compris avant le test nul. Grandeurs : Δnll = nll moyen du moteur moins nll de la référence bf16 (transformers), mêmes jetons, en nats ; KL = KL(réf ‖ moteur) moyenne par position.

Ce que je sais en les écrivant : le Q4 d'herbert lancé comme au banc (`--q4` seul) est W4A8, groupes de 32, échelle f32, grille symétrique `absmax/7` (codes ±7, 15 niveaux), sur toutes les projections y compris la tête ; Q4_K_M garde des sous-blocs de 32 asymétriques (échelle et minimum sur 6 bits), met `attn_v`, la moitié des `ffn_down` et la tête en Q6_K. En octobre sur Qwen3.5-2B : Q4 `absmax/7` +0,132 ± 0,018, grille `max/-8` et tête int8 +0,094.

Sens général : le Q4 d'herbert (banc) est plus loin de la référence que Q4_K_M sur tous les modèles, d'un facteur 2 à 4 sur le Δnll et sur la KL. ik et llama.cpp au même Q4_K_M : écart entre eux sous 0,005 nats, dans le bruit.

| modèle | herbert Q4 (banc) Δnll | Q4_K_M Δnll | Q4_0 (QAT/QAD) Δnll | KL herbert / Q4_K_M | top-1 herbert / Q4_K_M |
|---|---|---|---|---|---|
| Gemma 4 26B-A4B (QAT) | +0,04 à +0,08 | pas de GGUF | +0,01 | 0,05 / — (q4_0 : 0,015) | 88 % / — (q4_0 : 94 %) |
| Gemma 4 E4B (QAT) | +0,06 à +0,12 | pas de GGUF | +0,02 | 0,07 / — (q4_0 : 0,02) | 86 % / — (q4_0 : 93 %) |
| Qwen3.6-35B-A3B | +0,04 à +0,08 | +0,02 | — | 0,05 / 0,02 | 88 % / 93 % |
| Qwen3-VL-30B-A3B Instruct | +0,04 à +0,08 | +0,02 | — | 0,05 / 0,02 | 88 % / 93 % |
| LFM3-1.7B | +0,10 à +0,20 | +0,05 | +0,03 (QAD) | 0,12 / 0,05 | 82 % / 90 % |
| LFM2.5-8B-A1B | +0,06 à +0,12 | +0,03 | — | 0,08 / 0,03 | 86 % / 92 % |

Gemma QAT, bras « poids égaux » (`--q4 --q4-grid8`, même grille que Q4_0) : Δnll proche du Q4_0 de llama.cpp (écart sous 0,01), nettement meilleur que le bras banc, parce que la grille `absmax/7` défait l'alignement appris par le QAT.

Accord glouton à poids égaux (Gemma E2B, E4B, 26B, 256 jetons, ~20 prompts) : llama.cpp contre ik, longueur médiane d'accord exact 60 à 150 jetons ; herbert (grille 8) contre llama.cpp, 20 à 80 jetons (activations int8 par groupes de 32 chez herbert, q8_0 chez llama.cpp : arrondis différents) ; part de jetons identiques position par position 40 à 70 %. herbert au banc (grille 7) : accord plus court, 10 à 40 jetons.

Qwen3.6 en AVX2 (Ryzen 7 5700G) : je parie à 60 % sur un vrai défaut de la voie AVX2 (somme 16 bits saturée de `vpmaddubsw`, ou chemin DeltaNet/MoE propre à ymm), visible comme une KL AVX2 contre référence nettement au-dessus de la KL AVX-512 contre référence (facteur 2 ou plus) ; à 40 % sur de l'arrondi seul (KL des deux voies semblables, arrêt à 1 jeton dû à autre chose que les logits).
