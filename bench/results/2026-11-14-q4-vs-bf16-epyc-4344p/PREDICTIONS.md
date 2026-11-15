# Prédictions — Q4 contre bf16 sur EPYC 4344P (écrites le 14/11/2026 vers 18 h 15, heure de Paris, avant toute mesure de ce dépôt)

Déjà vu avant d'écrire : la table des rejeux du même jour (2026-11-14-rejeux-grille8) donne le Q4 grille 8 de 68c3a96 sur EPYC 4344P, Gemma 4 E4B 626 jetons 398 / 17,54 tok/s, Qwen3.6-35B-A3B 627 jetons 335 / 25,9. Le journal du Q4 d'E4B dit « read as bf16 : 344 matrices packed R16 (8.65 GiB) » sur 14,79 GiB projetés. Le bf16 de Qwen3.5-4B (#001, C1 EPYC 4344P) : 282 / 6,15, 8,412 Go relus par jeton. Qwen3.5-4B n'est pas sur EPYC 4344P : je mesure Qwen3.5-2B à sa place (C1 EPYC 4344P : 678 / 13,71, 3,764 Go).

Format Q4 d'herbert : 32 valeurs = 16 octets de codes + une échelle f32 de 4 octets, soit 0,625 octet par poids contre 2 en bf16 (rapport 0,3125), la tête comprise (pas de --q4-head-int8).

## Rapports attendus Q4 / bf16 (invite d'environ 630 jetons, 128 produits, 8 fils)

| modèle | decode Q4/bf16 | prefill Q4/bf16 | mémoire résidente au repos Q4/bf16 | pic (VmHWM) Q4/bf16 | octets relus par jeton Q4/bf16 |
|---|---|---|---|---|---|
| Qwen3.5-2B | 2,5 (bf16 ≈ 13,3, Q4 ≈ 33) | 1,3 | 0,45 | 0,5 | 0,33 |
| Gemma 4 E4B | 3,0 (bf16 ≈ 5,8, Q4 17,5 déjà vu) | 1,3 | 0,45 | 0,5 | 0,32 |

À environ 6 000 jetons : le rapport du decode baisse (l'attention et le cache pèsent davantage, et ne changent pas avec le format des poids) : 2,2 pour le 2B, 2,6 pour E4B.

Chargement (lancement → /health prêt, fichiers chauds) : Q4 plus long que bf16 d'environ 1,3 fois (la quantification se fait au montage).

Taille sur disque : identique pour les deux bras (herbert lit les mêmes safetensors bf16 et quantifie au chargement).

Risque que je vois : si les pages du fichier projeté restent résidentes (RssFile), la mémoire résidente du Q4 ne descend pas autant ; dans ce cas le rapport au repos serait près de 0,7 au lieu de 0,45.

## Qwen3.6-35B-A3B

Taille bf16 depuis les en-têtes safetensors : environ 70 Go (au-dessus des 62 Go de EPYC 4344P). Q4 : mémoire résidente au repos environ 22 Go, pic environ 23 Go.
