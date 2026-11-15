# Q4 contre bf16 : la vitesse et la mémoire, même code, même fenêtre (EPYC 4344P, 14 novembre 2026)

Question : herbert fait tourner les modèles en Q4 ; qu'est-ce que ça donne contre le bf16 du **même code**, mesuré dans la **même fenêtre** ? L'article #001 donnait le bf16 de Qwen3.5-4B sur EPYC 4344P (282 / 6,15 tok/s, 8,412 Go de poids relus par jeton, campagne C1, mesure du 8 octobre 2026), sans Q4.

**Qwen3.5-4B n'est pas sur EPYC 4344P** (aucun safetensors de 4B sous `~/models/` ni ailleurs, recherche `find /` des safetensors de plus de 1 Go). Il n'a pas été copié (consigne : la liaison est lente). À sa place, **Qwen3.5-2B**, même famille, mesuré dans la même campagne C1 (EPYC 4344P, bf16 : 678 / 13,71, 3,764 Go par jeton). Le raccord à C1 est approché : C1 était pp148 / tg64 avec le binaire publié de #001 hors serveur ; ici, le serveur de 68c3a96 et 627 / 6 131 jetons.

## Conditions

- Machine : EPYC 4344P, AMD EPYC 4344P (Zen 4, 8 cœurs, AVX-512 BF16, pas d'AVX-VNNI : Q4 sur le micro-noyau zmm AVX-512 VNNI), 2 × 32 Go DDR5-5200 (63 907 Mio vus par Linux), 8 fils. Machine libre au départ (charge 0,00, aucun serveur herbert ni llama), toute la chaîne sous `flock ~/bench-epyc-4344p.lock`, un serveur à la fois, port neuf par serveur (33101 à 33109).
- Code : `epyc-4344p:~/publication-jit-q4/` (`ARCHIVE-COMMIT.txt` = 68c3a96, l'archive des rejeux), binaire `herbert-jit-llama-serve` sha256 `37e5e1b5…847b`, `--threads 8 --ctx 8192` ; bras Q4 : `--q4` (grille 8, défaut de 68c3a96) ; bras bf16 : sans `--q4`. Les deux bras lisent **les mêmes fichiers safetensors bf16** ; le Q4 est fait au montage.
- Banc : Qwen3.5-2B, `~/q40pur/throughput.py` avec le suffixe « \n1. Le port de Marseille » (protocole du bloc A des rejeux, 380 et 4 240 mots, 627 et 6 131 jetons) ; Gemma 4 E4B (safetensors QAT `unquantized`), `~/banc-1113/throughput.py` (bloc C : 78 mots d'abord, première invite froide, puis 380 et 4 240, 626 et 6 184 jetons). `/completion`, `cache_prompt:false`, `ignore_eos`, 128 jetons, température 0, 3 répétitions par serveur, médianes des `timings` du serveur. **`predicted_n` = 128 partout.**
- Alternance : par modèle, fichiers relus à chaud (`cat > /dev/null`, journalisé), puis bf16, Q4, Q4, bf16 (ABBA). Fenêtre 18:09–18:31, heure de Paris.
- Mémoire : `/proc/PID/status` 2 s après `/health` (repos), avec `VmHWM` à ce moment = pic du chargement ; puis `echo 5 > clear_refs` remet `VmHWM` à zéro ; `VmHWM` à la fin du banc = pic pendant les requêtes (dont la longue). Échantillons toutes les 0,5 s dans `*/rss-*.log`. Go = 10⁹ octets.
- Chargement : du lancement du processus à la première réponse de `/health` (sondage toutes les 0,1 s), fichiers en cache.
- Octets relus par jeton : `_octets.py`, depuis les en-têtes safetensors, méthode de #001 (partie langage, tour de vision, audio et MTP exclues, tête liée comptée une fois en entier, tables de recherche exclues). Il retrouve exactement 1,505 Go (0.8B) et 3,764 Go (2B) de C1. Q4 d'herbert (`jit/int8.rs`) : 32 poids = 16 octets de codes + une échelle f32, 0,625 octet par poids (tête comprise), lignes arrondies à 16 ; le reste (normes, conv1d, A_log…) au format du fichier. Mêmes listes de matrices que le journal du chargeur (151 pour le 2B, 3,50 GiB ; 344 pour E4B, 8,65 GiB) ; le moteur dit « weights touched » 1,10 / 3,50 GiB (2B) et 2,66 / 8,50 GiB (E4B). Sortie : `octets.txt`.

## Tableau (médianes ; deux serveurs par bras, valeurs r1 / r2 dans `tableau.md`)

| modèle | format | jetons | prefill (tok/s) | decode (tok/s) | octets relus par jeton | Go/s lus (decode × octets) | RSS au repos (Go) | dont fichier | pic de chargement (Go) | pic pendant les requêtes (Go) | chargement (s) | disque |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-2B | bf16 | 627 | 720 | 13,88 | 3,764 Go | 52,2 | 6,44 | 0,01 | 8,86 | 6,44 | 3,9 | 4,548 Go |
| Qwen3.5-2B | Q4 | 627 | 1 097 | 43,21 | 1,179 Go | 50,9 | 3,38 | 0,01 | 6,46 | 3,38 | 7,8 | les mêmes 4,548 Go |
| Qwen3.5-2B | bf16 | 6 131 | 669 | 13,58 | | | | | | | | |
| Qwen3.5-2B | Q4 | 6 131 | 954 | 40,42 | | | | | | | | |
| Gemma 4 E4B | bf16 | 626 | 265 | 5,65 | 9,289 Go | 52,4 | 21,68 | 9,30 | 22,07 | 21,69 | 6,1 | 15,882 Go |
| Gemma 4 E4B | Q4 | 626 | 400 | 17,81 | 2,904 Go | 51,7 | 15,60 | 9,30 | 15,98 | 15,61 | 11,7 | les mêmes 15,882 Go |
| Gemma 4 E4B | bf16 | 6 184 | 243 | 5,53 | | | | | | | | |
| Gemma 4 E4B | Q4 | 6 184 | 347 | 16,71 | | | | | | | | |

Rapports Q4 / bf16 :

| modèle | decode ~630 | decode ~6 150 | prefill ~630 | prefill ~6 150 | octets par jeton | RSS au repos | pic des requêtes | pic de chargement | chargement |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-2B | **× 3,11** | × 2,98 | × 1,52 | × 1,43 | 0,313 | **0,52** | 0,52 | 0,73 | × 2,0 |
| Gemma 4 E4B | **× 3,15** | × 3,02 | × 1,51 | × 1,43 | 0,313 | **0,72** (mémoire anonyme seule : 0,51) | 0,72 | 0,72 | × 1,9 |

Le chemin bf16 est à jour : Qwen3.5-2B bf16 décode à 13,88 tok/s contre 13,71 dans C1 (#001), à 52,2 Go/s lus (C1 : 51,6 ; sonde de EPYC 4344P 54,0). Gemma E4B bf16 lit 52,4 Go/s. Les deux formats sont à la bande passante (Q4 : 50,9 et 51,7 Go/s) : le decode suit les octets relus, le Q4 en relit 3,2 fois moins et décode 3,1 fois plus vite. Le Q4 d'E4B retrouve les rejeux (400 / 17,81 contre 398 / 17,54).

## Qwen3.6-35B-A3B

- **bf16 : 71,90 Go** de safetensors (en-têtes, tout bf16), dont 69,32 Go de partie langage (64,43 Go d'experts). EPYC 4344P a 63 907 Mio = 67,0 Go de mémoire en tout : **le bf16 ne tient pas**, même sans vision ni MTP.
- **Q4 (même code, mesuré ici, même banc, 6 131 jetons)** : RSS au repos **24,67 Go** (anonyme, fichier 9 Mo), pic pendant les requêtes **24,67 Go**, pic de chargement 27,36 Go, chargement 55,7 s (fichiers non relus avant : 67 Gio ne tiennent pas dans le cache).
- Débit en Q4, celui de la table des rejeux (non remesuré pour l'article) : 627 jetons **334,8 / 25,93** ; 6 131 jetons 305,8 / 23,85 tok/s. Le serveur de ce banc a donné 304,8 / 23,82 à 6 131 jetons, cohérent.
- Pour situer : le GGUF Q4_0 pur du même modèle fait 19,58 Go sur disque.

## Lecture (sans verdict)

- Mémoire : sur Qwen3.5-2B, RSS divisée par 1,9 (6,44 → 3,38 Go). Sur Gemma E4B, 21,68 → 15,60 Go seulement (× 0,72) : 9,30 Go restent des pages du fichier projeté dans les deux bras (`RssFile`), les tables de recherche d'E4B (`embed_tokens` et surtout `embed_tokens_per_layer`, 6,98 Go dans les en-têtes) lues en place ; la mémoire anonyme, elle, passe de 12,38 à 6,35 Go (× 0,51). Ces pages de fichier sont du cache partagé, récupérable par le noyau, mais elles comptent dans `VmRSS`.
- Le Q4 garde de la mémoire au-delà des poids Q4 : 3,38 Go résidents pour 1,18 Go de matrices Q4 sur le 2B (la table d'embeddings bf16 de 1,02 Go pour la recherche, le cache K/V réservé pour 8 192 positions, les états, les programmes). Non détaillé ici.
- Chargement : le Q4 prend deux fois plus de temps (quantification au montage : 7,8 s contre 3,9 sur le 2B, 11,7 contre 6,1 sur E4B), et son pic de chargement dépasse sa mémoire au repos (les matrices passent par le bf16 avant le Q4 : 6,46 Go de pic pour 3,38 au repos sur le 2B ; 27,4 pour 24,7 sur le 35B).
- Disque : herbert lit les mêmes safetensors bf16 dans les deux bras ; le Q4 ne réduit pas la taille sur disque (contrairement à un GGUF Q4_0 : 5,15 Go pour E4B contre 15,88).
- Contexte long : le gain du decode baisse peu (× 3,11 → × 2,98 ; × 3,15 → × 3,02), le prefill passe de × 1,5 à × 1,43.

## Prédictions

`PREDICTIONS.md` (sha256 `141d70a3…7a13`, `PREDICTIONS.md.sha256`), écrit et commité (18:08 heure de Paris ; le fichier dit « vers 18 h 15 », il a été écrit à 18:08) avant le lancement du banc (18:09). Sort :

| prédiction | mesure | |
|---|---|---|
| octets par jeton Q4/bf16 0,33 (2B), 0,32 (E4B) | 0,313 et 0,313 | juste |
| decode 2B × 2,5 (bf16 13,3, Q4 33) | × 3,11 (13,88, 43,21) | faux : sous-estimé, le Q4 du 2B reste à la bande passante |
| decode E4B × 3,0 (bf16 5,8) | × 3,15 (5,65) | proche |
| decode long : × 2,2 (2B), × 2,6 (E4B) | × 2,98, × 3,02 | faux : le rapport baisse beaucoup moins |
| prefill × 1,3 | × 1,52 et × 1,51 | faux : sous-estimé |
| RSS au repos × 0,45 ; pic × 0,5 | 2B 0,52 / 0,52 ; E4B 0,72 / 0,72 | faux pour les deux ; pour E4B le risque écrit (« près de 0,7 » si des pages du fichier restent) est ce qui arrive |
| chargement Q4 × 1,3 | × 2,0 et × 1,9 | faux |
| disque identique | identique | juste |
| 35B bf16 ≈ 70 Go | 71,90 Go | juste |
| 35B Q4 repos ≈ 22 Go, pic ≈ 23 | 24,67 et 24,67 (chargement 27,36) | proche, sous-estimé |

## Fichiers

`run-epyc-4344p.log` (chaîne complète), `*/server-*.log`, `*/res-*.json`, `*/rss-*.log`, `octets.txt`, `tableau.md` (par serveur), `q4-contre-bf16.json` (lu par la page de comparaison) ; outillage `_q4_bf16_epyc-4344p.sh`, `_octets.py`, `_tableau.py`.
