# Prédictions avant mesure — Qwen3.6-35B-A3B en Q4_0 pur (14 novembre 2026, 06:35 heure de Paris)

Écrites avant toute mesure sur le Q4_0 pur : la quantification `llama-quantize --pure` tourne sur Ryzen 9 9900X, aucun bras n'est lancé, aucun débit mesuré. Jamais retouchées après. Ce que je sais en les écrivant : les mesures de cette nuit sur Ryzen 9 9900X (KL wiki/fr : herbert `--q4` 0,089/0,098, `--q4 --q4-grid8` 0,072/0,076, llama Q4_K_M 0,019/0,020, top-1 86–88 % contre 93–94 %) ; le banc EPYC 4344P d'hier (herbert 334 tok/s en prefill et 26,0 en decode à 621 jetons, llama.cpp Q4_K_M 156 et 16,6) ; le code d'herbert : `--q4` met en Q4 (groupes de 32, échelle f32, activations int8 par 32) toute projection de 64 lignes ou plus, tête comprise ; le routeur (blocs de 16 experts), `shared_expert_gate`, `in_proj_a/b`, conv1d et normes restent en 16 ou 32 bits ; la table d'embeddings reste bf16.

## Justesse (Ryzen 9 9900X, références transformers complètes, wiki 48 et fr 24)

1. llama.cpp Q4_0 pur : KL 0,060 à 0,075 (wiki et fr), top-1 87 à 89 %, Δnll +0,03 à +0,06. Donc 3 à 4 fois la KL du Q4_K_M.
2. herbert `--q4 --q4-grid8` contre llama Q4_0 pur : même grille, mêmes groupes ; je parie à 70 % que les deux KL sont à moins de 10 % l'une de l'autre (sur wiki comme sur fr). Si l'écart dépasse 10 %, je parie qu'herbert est le plus bas (embeddings bf16 au lieu de Q4_0, échelles f32 au lieu de f16) plutôt que le plus haut.
3. ik_llama.cpp sur le même fichier : KL à moins de 3 % de llama.cpp. `-rtr` (réempaquetage Q4_0_R8) : KL à moins de 5 % d'ik sans `-rtr`.
4. herbert `--q4` (grille 7) reste au-dessus du Q4_0 pur sur la KL, de 15 à 35 %.

## Débit (EPYC 4344P, 8 fils, `/completion`, 128 jetons, médianes de 3)

5. llama.cpp Q4_0 pur contre Q4_K_M (même fenêtre) : decode plus rapide de 10 à 25 % (fichier plus petit, partie non experte en Q4 au lieu de Q8_0), soit 18 à 21 tok/s à 621 jetons ; prefill plus rapide de 10 à 40 % (170 à 220 tok/s). Le ×1,56 du decode d'herbert devient ×1,25 à ×1,45.
6. ik_llama.cpp Q4_0 pur sans `-rtr` : prefill 200 à 300 tok/s, decode 18 à 22 tok/s ; avec `-rtr` : prefill 280 à 400 tok/s (peut rejoindre herbert), decode égal à ±5 % de sans `-rtr`.
7. herbert rejoué dans la même fenêtre : à ±3 % d'hier (334 / 26,0 à 621 jetons).
8. Aux grandes tailles (4 240 mots, ~6 100 jetons), l'écart de prefill herbert / llama.cpp grandit (attention), comme hier (×2,5 contre ×2,1).
