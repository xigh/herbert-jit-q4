# Q4 contre bf16 : la vitesse et la mémoire, même code, même fenêtre (Ryzen 9 9900X, 14 novembre 2026)

Même mesure que sur EPYC 4344P (`2026-11-14-q4-contre-bf16/`), mêmes scripts adaptés, sur Ryzen 9 9900X (Ryzen 9 9900X, Zen 5, AVX-512, 12 cœurs, 186 Gio), à 12 fils. Sur Ryzen 9 9900X, Qwen3.5-4B (celui de l'article #001) est là ; Qwen3.6-35B-A3B tient en bf16 : c'est le seul endroit où il est mesuré dans les deux formats.

⚠️ **Toutes les mesures, sauf une (Q4 du 2B, deuxième serveur), ont tourné avec environ un cœur occupé par un autre travail** : les simulations, synthèses et placements-routages FPGA en cours sur la même machine (`~/another-project`, `run_recorded.py`, nextpnr, yosys), commencés à 20:38:45, 27 s après le début du premier bras, et encore actifs à la fin. Ils n'ont pas été arrêtés (consigne). Les cœurs FPGA moyens pendant chaque banc sont dans la dernière colonne des tableaux, reconstitués depuis les `result.json` de ces travaux (`fpga-intervalles.txt`, `_fpga_intervalles.sh`). La charge n'a pas été la même pour tous les serveurs. Sur le **2B**, `q4-qwen35-2b-r2` a tourné sans FPGA (0 cœur), `bf16-r2` avec 0,78 cœur, les deux serveurs r1 avec 1,00 : `q4-r2` décode à 35,63 tok/s contre 34,64 pour `q4-r1` (+2,9 %), et son prefill fait 2 155 contre 1 978 (+8,9 %). Sur le **4B**, le bf16 a eu plus de FPGA que le Q4 (1,29 et 1,89 cœur contre 0,82 et 1,00, avec par moments deux travaux en parallèle), et `bf16-r2` décode à 5,04 contre 5,13 au r1. E4B a eu 1,00 cœur pour les quatre serveurs ; le 35B, 0,91 et 0,95 en bf16, 1,00 en Q4. Les moyennes des tableaux mêlent donc des paires mesurées dans des conditions différentes. Voici les rapports par paire (serveurs de même rang, bf16 rN contre Q4 rN, mesurés à quelques minutes d'écart) :

| paire | cœurs FPGA bf16 / Q4 | decode ~630 | decode ~6 150 | prefill ~630 | prefill ~6 150 |
|---|---|---|---|---|---|
| Qwen3.5-4B r1 | 1,29 / 0,82 | × 3,08 | × 2,80 | × 1,43 | × 1,30 |
| Qwen3.5-4B r2 | 1,89 / 1,00 | × 3,17 | × 2,82 | × 1,45 | × 1,32 |
| Qwen3.5-2B r1 | 1,00 / 1,00 | × 3,02 | × 2,83 | × 1,41 | × 1,29 |
| Qwen3.5-2B r2 | 0,78 / **0** | × 3,12 | × 2,90 | **× 1,60** | × 1,41 |
| Gemma 4 E4B r1 | 1,00 / 1,00 | × 3,06 | × 2,88 | × 1,46 | × 1,36 |
| Gemma 4 E4B r2 | 1,00 / 1,00 | × 3,13 | × 2,94 | × 1,42 | × 1,33 |
| Qwen3.6-35B-A3B r1 | 0,91 / 1,00 | × 2,91 | × 2,73 | × 41,4 | × 44,8 |
| Qwen3.6-35B-A3B r2 | 0,95 / 1,00 | × 2,87 | × 2,70 | × 40,1 | × 44,6 |

À conditions égales (2B r1, E4B r1 et r2), le decode fait × 3,02 à × 3,13 et le prefill × 1,41 à × 1,46. Le × 1,50 du prefill du 2B dans le tableau moyen vient du serveur Q4 sans FPGA.

Serveur des appels d'outils (port 8620) : à 19:41, il tournait à 914 % de CPU (panel en cours sur la même machine) ; un premier lancement a été arrêté avant tout bras (`run-ryzen-9-9900x-essai1-arrete.log`, rien de mesuré). La chaîne a été relancée avec une garde de repos. Le 8620 était arrêté pendant toutes les mesures (« absent » à chaque bras, `P8620-*` dans les logs). Il n'occupait donc ni mémoire ni CPU.

## Conditions

- **Code** : `ryzen-9-9900x:~/publication-jit-q4-rejeu/` (`ARCHIVE-COMMIT.txt` = 68c3a96), le `git archive 68c3a96` des rejeux (`2026-11-14-rejeux-grille8/`, build `--release --locked --features serve`). Binaire `herbert-jit-llama-serve` sha256 `1483ab97…fa2f` (compilé sur Ryzen 9 9900X ; celui de EPYC 4344P, compilé sur EPYC 4344P depuis la même archive, a un autre hachage). Pas 75e75fc (renomme les binaires). Options : `--threads 12 --ctx 8192`, bras Q4 `--q4` (grille 8, défaut de 68c3a96), bras bf16 sans `--q4`, mêmes safetensors bf16 pour les deux bras.
- **Modèles** : `~/models/Qwen3.5-4B`, `~/models/Qwen3.5-2B`, `~/models/gemma-4-E4B-it-qat-q4_0-unquantized`, `~/models/Qwen3.6-35B-A3B` (mêmes tailles et mêmes en-têtes que ceux de EPYC 4344P : `_octets.py` redonne 3,764 et 9,289 Go).
- **Banc** : un seul client, `~/banc-q40/throughput.py` (md5 `367da85c…`, le même que `epyc-4344p:~/q40pur/throughput.py`). Sans `SUFFIX`, il envoie exactement les invites de `~/banc-1113/throughput.py` (md5 `1059e64f…`), absent de ryzen-9-9900x. Les Qwen ont le suffixe « \n1. Le port de Marseille », WORDS 380 et 4240, soit 627 et 6 131 jetons. Gemma est sans suffixe, WORDS 78, 380 et 4240, soit 218 jetons (première invite, froide, non comptée), 626 et 6 184. `/completion`, `cache_prompt:false`, `ignore_eos`, 128 jetons, température 0, 3 répétitions par serveur, médianes des `timings`. **`predicted_n` = 128 partout.**
- **Alternance** : par modèle, fichiers relus (`cat > /dev/null`), puis bf16, Q4, Q4, bf16. Un serveur à la fois, port neuf (34101 à 34134), serveurs lancés sous `choom -n 1000`. Un `flock ~/bench-ryzen-9-9900x.lock` par modèle (`_chaine_ryzen-9-9900x.sh`), pour laisser passer entre deux modèles l'emballage et les compilations qui attendaient le verrou. Garde avant chaque bras : d'abord serveur 8620 sous 5 % d'un cœur et charge sous 3 ; à partir de 22:33, sur consigne, port 8620 sans processus et aucun processus autre que le travail FPGA au-dessus de 5 %. Le segment du 35B a été arrêté pendant une attente, entre `q4-r1` et `q4-r2`, et repris avec cette garde (`run-ryzen-9-9900x-suite35b.log`). `uptime`, `free`, `MemAvailable` et les 5 premiers processus sont relevés à chaque bras.
- **Fenêtre** (heure de Paris) : 20:37:40 – 23:18:14. 4B 20:37–20:55, 2B 20:57–21:07, E4B 21:09–21:32, 35B 21:35–22:32 et 22:32–23:18.
- **Mémoire, chargement, octets** : comme sur EPYC 4344P (`/proc/PID/status` 2 s après `/health` ; `clear_refs 5` ; `VmHWM` en fin de banc ; échantillons toutes les 0,5 s dans `*/rss-*.log` ; `_octets.py` ; Go = 10⁹ octets). Pour le 35B, on ajoute aux octets hors experts de `_octets.py` les experts routés (8 sur 256, `octets.txt`) : 5,893 Go en bf16, 1,843 Go en Q4.

## Tableau (moyenne des deux serveurs par bras, chacun médiane de 3 ; par serveur dans `tableau.md`)

| modèle | format | jetons | prefill (tok/s) | decode (tok/s) | octets relus par jeton | Go/s lus | RSS au repos (Go) | dont fichier | pic de chargement (Go) | pic pendant les requêtes (Go) | chargement (s) | FPGA (cœurs moyens r1 / r2) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B | bf16 | 627 | 554 | 5,09 | 8,412 Go | 42,8 | 11,43 | 0,01 | 12,98 | 11,45 | 4,0 / 4,0 | 1,29 / 1,89 |
| Qwen3.5-4B | Q4 | 627 | 799 | 15,89 | 2,630 Go | 41,8 | 5,45 | 0,01 | 9,24 | 5,48 | 10,1 / 9,8 | 0,82 / 1,00 |
| Qwen3.5-4B | bf16 | 6 131 | 553 | 4,90 |  |  |  |  |  |  |  |  |
| Qwen3.5-4B | Q4 | 6 131 | 724 | 13,77 |  |  |  |  |  |  |  |  |
| Qwen3.5-2B | bf16 | 627 | 1 376 | 11,43 | 3,764 Go | 43,0 | 6,82 | 0,01 | 9,19 | 6,82 | 2,6 / 2,5 | 1,00 / 0,78 |
| Qwen3.5-2B | Q4 | 627 | 2 066 | 35,13 | 1,179 Go | 41,4 | 3,67 | 0,01 | 6,75 | 3,67 | 6,9 / 6,9 | 1,00 / 0,00 |
| Qwen3.5-2B | bf16 | 6 131 | 1 439 | 11,07 |  |  |  |  |  |  |  |  |
| Qwen3.5-2B | Q4 | 6 131 | 1 942 | 31,72 |  |  |  |  |  |  |  |  |
| Gemma 4 E4B | bf16 | 626 | 509 | 4,64 | 9,289 Go | 43,1 | 20,97 | 8,56 | 21,41 | 20,98 | 14,8 / 4,6 | 1,00 / 1,00 |
| Gemma 4 E4B | Q4 | 626 | 734 | 14,38 | 2,904 Go | 41,8 | 16,05 | 9,30 | 16,42 | 16,07 | 15,4 / 10,0 | 1,00 / 1,00 |
| Gemma 4 E4B | bf16 | 6 184 | 504 | 4,49 |  |  |  |  |  |  |  |  |
| Gemma 4 E4B | Q4 | 6 184 | 678 | 13,07 |  |  |  |  |  |  |  |  |
| Qwen3.6-35B-A3B | bf16 | 627 | 8,6 | 7,20 | 5,893 Go | 42,4 | 71,20 | 0,01 | 73,40 | 71,23 | 234,8 / 45,5 | 0,91 / 0,95 |
| Qwen3.6-35B-A3B | Q4 | 627 | 349 | 20,77 | 1,843 Go | 38,3 | 24,92 | 0,01 | 27,99 | 24,92 | 172,9 / 32,8 | 1,00 / 1,00 |
| Qwen3.6-35B-A3B | bf16 | 6 131 | 8,3 | 6,87 |  |  |  |  |  |  |  |  |
| Qwen3.6-35B-A3B | Q4 | 6 131 | 371 | 18,64 |  |  |  |  |  |  |  |  |

| modèle | decode ~630 | decode ~6 150 | prefill ~630 | prefill ~6 150 | octets par jeton | RSS au repos | pic des requêtes | pic de chargement | chargement (plus court des deux serveurs) |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B | **× 3,12** | × 2,81 | × 1,44 | × 1,31 | 0,313 | **0,48** | 0,48 | 0,71 | × 2,5 |
| Qwen3.5-2B | **× 3,07** | × 2,87 | × 1,50 | × 1,35 | 0,313 | **0,54** | 0,54 | 0,73 | × 2,7 |
| Gemma 4 E4B | **× 3,10** | × 2,91 | × 1,44 | × 1,35 | 0,313 | **0,77** | 0,77 | 0,77 | × 2,2 |
| Qwen3.6-35B-A3B | **× 2,89** | × 2,71 | × 40,74 | × 44,72 | 0,313 | **0,35** | 0,35 | 0,38 | × 0,7 |

La dernière colonne du premier tableau donne les cœurs occupés par le travail FPGA pendant le banc (r1 / r2). Le chargement est donné par serveur, r1 / r2 : le premier chargement d'E4B en bf16 (14,8 s) et les premiers chargements du 35B (234,8 s en bf16, 172,9 s en Q4) ont relu une partie des fichiers sur le disque, malgré le `cat` préalable. 72 Go de safetensors, plus 64 Go de `/dev/shm` pris par d'autres travaux, et le cache ne garde pas tout. La colonne « chargement » des rapports prend donc, par bras, le plus court des deux serveurs. Le 4B et le 2B ont des chargements réguliers. Même effet sur le RSS au repos d'E4B en bf16 : `RssFile` vaut 7,83 Go au r1 contre 9,30 au r2. Avec le seul r2, le rapport au repos d'E4B est de 0,74 (16,12 / 21,80).

## Raccord à #001 (C1 Ryzen 9 9900X, `2026-10-08-c1-campagne-ryzen-9-9900x/`)

C1 mesurait le binaire publié de #001 hors serveur, à pp148 / tg64 et 12 fils. Ici, c'est le serveur de 68c3a96, à 627 jetons et 128 produits, avec environ un cœur pris par le FPGA.

| | C1 (#001) | ici, bf16 | ici, Q4 |
|---|---|---|---|
| Qwen3.5-4B decode | 5,12 | 5,09 (−0,6 %) | 15,89 (× 3,12) |
| Qwen3.5-4B prefill | 461 (148 jetons) | 554 (627 jetons) | 799 |
| Qwen3.5-2B decode | 11,51 | 11,43 (−0,7 %) | 35,13 (× 3,07) |
| Qwen3.5-2B prefill | 1 103 (148 jetons) | 1 376 (627 jetons) | 2 066 |

Le decode bf16 retrouve #001 à moins de 1 % : 42,8 et 43,0 Go/s lus, contre 43,1 et 43,3 dans C1, et une sonde de 44,1 Go/s à 12 fils. Le prefill n'est pas comparable : longueur d'invite et code différents. En Q4, le 4B de #001 décode 3,1 fois plus vite.

## Les deux machines (rapports Q4 / bf16, invite d'environ 630 jetons)

| modèle | machine | decode | decode ~6 150 | prefill | Go/s lus bf16 / Q4 | RSS au repos | chargement |
|---|---|---|---|---|---|---|---|
| Qwen3.5-4B | Ryzen 9 9900X | × 3,12 | × 2,81 | × 1,44 | 42,8 / 41,8 | 0,48 | × 2,5 |
| Qwen3.5-2B | EPYC 4344P | × 3,11 | × 2,98 | × 1,52 | 52,2 / 50,9 | 0,52 | × 2,0 |
| Qwen3.5-2B | Ryzen 9 9900X | × 3,07 | × 2,87 | × 1,50 | 43,0 / 41,4 | 0,54 | × 2,7 |
| Gemma 4 E4B | EPYC 4344P | × 3,15 | × 3,02 | × 1,51 | 52,4 / 51,7 | 0,72 | × 1,9 |
| Gemma 4 E4B | Ryzen 9 9900X | × 3,10 | × 2,91 | × 1,44 | 43,1 / 41,8 | 0,77 (r2 seul : 0,74) | × 2,2 |
| Qwen3.6-35B-A3B | Ryzen 9 9900X | × 2,89 | × 2,71 | × 40,7 | 42,4 / 38,3 | 0,35 | × 0,7 |

## Qwen3.6-35B-A3B dans les deux formats

- **Mémoire.** Le bf16 tient : 71,20 Go résidents au repos, tout en mémoire anonyme (`RssFile` 0,01 Go), pic de chargement 73,40 Go, pic des requêtes 71,23 Go. `MemAvailable` valait 121 Go avant le bras. En Q4 : 24,92 Go au repos (EPYC 4344P : 24,67), pic de chargement 27,99 Go. Rapport 0,35.
- **Decode.** bf16 7,20 tok/s à 627 jetons, soit 42,4 Go/s pour 5,89 Go relus par jeton : à la bande passante, comme les modèles denses. Q4 20,77 tok/s (rejeux B, sans suffixe, à 621 jetons : 21,12), soit 38,3 Go/s. Rapport × 2,89 ; × 2,71 à 6 131 jetons.
- **Prefill.** En bf16, **8,6 tok/s** à 627 jetons et 8,3 à 6 131, à peine plus que le decode. En Q4 : 349 et 371. Le journal du serveur bf16 donne un programme de prefill de la même taille que celui du decode (`prefill program Some((11919824, …))`, `decode program Some((11919824, …))`) ; en Q4, le prefill a son propre programme (198 226 545 contre 7 329 495). Le prefill bf16 de ce MoE passe donc par le programme du decode, jeton par jeton. Une répétition à 6 131 jetons prend environ 12 min 20 s : sous la limite de 30 min, les deux tailles ont été faites. Le bras bf16 dure environ 43 min par serveur.
- **Chargement.** 45,5 s en bf16 et 32,8 s en Q4 au deuxième serveur, avec les fichiers en cache. Au premier serveur, 234,8 s et 172,9 s, avec une partie des fichiers relue sur le disque. Contrairement aux modèles denses, le Q4 du 35B se charge plus vite que le bf16.

## Lecture (sans verdict)

- Les deux formats décodent à la bande passante de Ryzen 9 9900X : 41 à 43 Go/s lus, contre 44,1 pour la sonde à 12 fils. Le Q4 relit 3,2 fois moins d'octets et décode 3,1 fois plus vite, comme sur EPYC 4344P, où la bande passante est plus haute (52 Go/s). Le rapport Q4/bf16 du decode est le même sur les deux machines à 1 ou 2 % près ; d'une paire à l'autre, sur Ryzen 9 9900X, il varie de 2 à 3 %.
- Prefill : × 1,44 à × 1,50 à 630 jetons en moyenne, mais × 1,41 à × 1,46 pour les paires mesurées à conditions égales (le × 1,50 du 2B vient du serveur Q4 sans FPGA) ; sur EPYC 4344P, × 1,51 et × 1,52. Le 35B est à part (voir plus haut).
- Mémoire : le 4B passe de 11,43 à 5,45 Go (0,48), le 2B de 6,82 à 3,67 (0,54). Pour E4B, comme sur EPYC 4344P, 9,30 Go de pages du fichier (ses tables de recherche) restent dans les deux bras. Les valeurs absolues de Ryzen 9 9900X sont un peu au-dessus de celles de EPYC 4344P pour le 2B (6,82 contre 6,44 en bf16, 3,67 contre 3,38 en Q4), non expliqué ici. Le compte de fils diffère (12 contre 8).
- Contexte long : le gain du decode baisse un peu plus que sur EPYC 4344P (× 3,12 → × 2,81 pour le 4B, × 3,07 → × 2,87 pour le 2B).

## Prédictions

`PREDICTIONS.md` (sha256 `4919b1c2…90c0`, `PREDICTIONS.md.sha256`), commité (18:43, heure de Paris ; le fichier dit « vers 18 h 50 ») avant le premier bras (20:38). Sort :

| prédiction | mesure | |
|---|---|---|
| decode bf16 4B 5,1 ; 2B 11,4 ; E4B 4,6 | 5,09 ; 11,43 ; 4,64 | juste |
| decode Q4 4B 15,5 ; 2B 34 ; E4B 14 | 15,89 ; 35,13 ; 14,38 | proche (un peu sous-estimé) |
| decode Q4/bf16 × 3,0 | × 3,12 ; × 3,07 ; × 3,10 | proche |
| prefill × 1,4 | × 1,44 ; × 1,50 ; × 1,44 (paires à conditions égales : × 1,41 à × 1,46) | juste |
| prefill bf16 4B ≈ 470 | 554 | faux : sous-estimé |
| Go/s : 40 à 44, Q4 un peu en dessous (39 à 42) | bf16 42,8 à 43,1, Q4 41,4 à 41,8 | juste |
| ~6 150 jetons : decode × 2,9, prefill × 1,35 | × 2,81 / 2,87 / 2,91 ; × 1,31 / 1,35 / 1,35 | juste |
| RSS au repos 0,48 / 0,52 / 0,72 ; pic idem | 0,48 / 0,54 / 0,77 (0,74 au r2) | juste pour le 4B, proche pour les autres |
| mémoire absolue = EPYC 4344P à ± 3 % ; 4B bf16 ≈ 11,5, Q4 ≈ 5,5 | 2B +6 % et +9 % ; E4B −3 % et +3 % ; 4B 11,43 et 5,45 | faux pour le 2B, juste pour le 4B |
| chargement plus court que EPYC 4344P, Q4/bf16 ≈ × 2 | plus court ; × 2,5 / 2,7 / 2,2 | à moitié : le rapport est sous-estimé |
| 35B tient ; bf16 ≈ 74 Go au repos, pic ≈ 80 ; Q4 ≈ 25 | tient ; 71,20, pic 73,40 ; 24,92 | proche |
| 35B decode bf16 ≈ 6,8, Q4 ≈ 21, × 3,1 | 7,20 ; 20,77 ; × 2,89 | proche pour les valeurs, rapport faux |
| 35B prefill bf16 ≈ 250, × 1,5 ; 6 150 jetons en moins d'une minute par répétition | 8,6 ; × 40,7 ; 12 min 20 s | **faux** : le prefill bf16 du MoE passe par le programme du decode |
| risque : le serveur 8620 actif pendant un bras | arrêté pendant toutes les mesures ; la perturbation venue d'ailleurs (FPGA, un cœur) | faux sur la source |

## Fichiers

`run-ryzen-9-9900x.log` (chaîne, du 4B au `q4-r1` du 35B), `run-ryzen-9-9900x-suite35b.log` (fin du 35B), `run-ryzen-9-9900x-essai1-arrete.log` (premier lancement, arrêté avant tout bras), `*/server-*.log`, `*/res-*.json`, `*/rss-*.log`, `octets.txt`, `fpga-intervalles.txt`, `tableau.md` (par serveur), `resume.md`, `q4-contre-bf16.json` (lu par la page de comparaison). Outillage : `_q4_bf16_ryzen-9-9900x.sh`, `_chaine_ryzen-9-9900x.sh`, `_octets.py` (copie de EPYC 4344P), `_tableau.py` (adapté : deux logs, ticks du 8620, FPGA), `_resume.py`, `_fpga_intervalles.sh`.
