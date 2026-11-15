# Prédictions — Q4 contre bf16 sur Ryzen 9 9900X (écrites le 14/11/2026 vers 18 h 50, heure de Paris, avant toute mesure de ce dépôt)

Déjà vu avant d'écrire : le dépôt de EPYC 4344P du même jour (`2026-11-14-q4-contre-bf16/`, 68c3a96, 8 fils : 2B × 3,11 en decode, E4B × 3,15, prefill × 1,5, RSS × 0,52 et × 0,72, chargement Q4 × 2) ; C1 Ryzen 9 9900X (mesure du 8 octobre 2026, bf16, 12 fils, pp148 / tg64, binaire publié de #001) : 4B 461 / 5,12, 2B 1 103 / 11,51, 43,1 et 43,3 Go/s lus, sonde 44,1 Go/s à 12 fils ; rejeux B de Ryzen 9 9900X (Qwen3.6-35B-A3B Q4 grille 8, 68c3a96, sans suffixe) : 621 jetons 370,6 / 21,12, 6 125 jetons 389,4 / 18,87. Octets par jeton depuis les en-têtes (`_octets.py`, lancé avant d'écrire) : 4B bf16 8,412 / Q4 2,630 Go ; 2B 3,764 / 1,179 ; E4B 9,289 / 2,904 ; 35B hors experts 3,880 / 1,214, plus les experts routés (8 sur 256 : 64,425 × 8 / 256 = 2,013 Go en bf16, 0,629 en Q4), soit environ 5,89 / 1,84 Go. Mémoire de Ryzen 9 9900X au moment d'écrire : /dev/shm tient 64 Go (d'autres travaux), le serveur du port 8620 24 Go, MemAvailable ≈ 80 Go pendant que le calcul de référence tourne.

## Rapports attendus Q4 / bf16 (invite d'environ 630 jetons, 128 produits, 12 fils)

| modèle | decode bf16 | decode Q4 | decode Q4/bf16 | prefill Q4/bf16 | RSS au repos Q4/bf16 | pic des requêtes Q4/bf16 | chargement Q4/bf16 |
|---|---|---|---|---|---|---|---|
| Qwen3.5-4B | 5,1 (≈ #001) | 15,5 | × 3,0 | × 1,4 | 0,48 | 0,48 | × 2,0 |
| Qwen3.5-2B | 11,4 | 34 | × 3,0 | × 1,4 | 0,52 | 0,52 | × 2,0 |
| Gemma 4 E4B | 4,6 | 14 | × 3,0 | × 1,4 | 0,72 | 0,72 | × 2,0 |

Prefill bf16 du 4B à 630 jetons : environ 470 tok/s (C1 : 461 à 148 jetons). Les deux formats lisent autour de 40 à 44 Go/s en decode (Ryzen 9 9900X lit moins que EPYC 4344P : 44 contre 54 Go/s de sonde), le Q4 un peu en dessous du bf16 (39 à 42 Go/s).

À environ 6 150 jetons : decode × 2,9 pour les trois, prefill × 1,35.

Mémoire en Go : mêmes valeurs absolues que EPYC 4344P pour le 2B et E4B (même code, mêmes fichiers, même `--ctx 8192`) à ± 3 % ; 4B bf16 au repos ≈ 11,5, Q4 ≈ 5,5. Chargement : plus court que sur EPYC 4344P (12 fils, Zen 5), rapport Q4/bf16 ≈ × 2.

## Qwen3.6-35B-A3B (si la mémoire le permet)

Il tient : bf16 au repos ≈ 74 Go résidents (anonymes), pic de chargement ≈ 80 Go ; Q4 ≈ 25 Go au repos (EPYC 4344P : 24,67). Decode bf16 ≈ 6,8 tok/s à 630 jetons (5,89 Go par jeton à 40 Go/s), Q4 ≈ 21 (avec le suffixe), rapport × 3,1 ; prefill bf16 ≈ 250 tok/s, rapport × 1,5. L'invite de 6 150 jetons en bf16 prend moins d'une minute par répétition : les deux tailles sont faites.

Risque que je vois : le serveur du port 8620 n'est pas tout à fait au repos (28 % de CPU moyen relevé à 18:40) ; s'il sert une requête pendant un bras, le decode de ce bras baisse de quelques pour cent.
