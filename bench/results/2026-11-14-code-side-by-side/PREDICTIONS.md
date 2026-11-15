# Prédictions avant mesure : du code côte à côte, Qwen3.6-35B-A3B (14 novembre 2026, 20:05 heure de Paris)

Écrites avant toute génération : aucun bras lancé, aucune invite rendue par le gabarit. Jamais retouchées après. Tâches dans `taches.json` (10 tâches, consignes en français : 7 en Python, 3 en Rust), tests cachés dans `tests/`, réponses attendues des deux tâches « prédire » dans `attendus.json`, sha256 dans `SHA256SUMS`. Les tests ont été validés avant ce fichier sur mes propres solutions (`_solutions/`, tout passe) et sur le code fourni par les consignes « corriger » et « compléter » (il échoue), dans le bac à sable de Ryzen 7 5700G ; aucune réponse du modèle n'existait alors.

Ce que je sais en les écrivant : la campagne « réponses entières côte à côte » du même jour (15 questions en prose, mêmes bras) : les trois Q4_0 dévient de la référence sur 11/15, le Q4_K_M sur 9/15 ; herbert et llama.cpp Q4_0 identiques jusqu'au bout sur 8/15 et au même point d'écart sur 10 des 11 écarts ; le fond change sur 5 questions dont 3 en mieux.

Protocole : glouton, gabarit du modèle sans réflexion, 500 jetons au plus, arrêt à la fin de séquence. Bras : référence transformers bf16 (R) ; herbert 68c3a96 `--q4` grille 8 (H) ; llama.cpp 5676dfbe0 Q4_0 pur (L0) ; ik_llama.cpp 7aa7e014 Q4_0 pur (IK) ; llama.cpp Q4_K_M (LK). « Réussit » = bloc extrait et tous les tests cachés passés (ou sortie exacte pour « prédire »).

## Par tâche

- p01 (fusionner des intervalles) : les cinq réussissent.
- p02 (slug, accents) : les cinq réussissent (NFKD puis ASCII).
- p03 (corriger une dichotomie, `lo < hi`) : les cinq réussissent.
- p04 (corriger une médiane paire) : les cinq réussissent.
- p05 (compléter les chiffres romains) : les cinq réussissent.
- p06 (prédire, défaut mutable, `[2, 3, 0]`) : les cinq réussissent.
- p07 (prédire, `round`, `//` et `%` des négatifs, `0.1 + 0.2`) : la plus fragile. R réussit ; au moins un bras quantifié se trompe (arrondi « 3 » pour `round(2.5)` ou « -3 » pour `-7 // 2`).
- r01 (moyenne entière sans débordement) : R, H, L0, LK réussissent ; c'est la seconde candidate à casser (somme en i32, ou arrondi vers le bas au lieu de vers zéro).
- r02 (corriger l'inversion par octets) : les cinq réussissent.
- r03 (compléter le compte de mots) : les cinq réussissent.

## Comptes

1. La référence réussit 10 tâches sur 10 (au pire 9, en perdant p07 ou r01).
2. Chaque bras quantifié réussit au moins 8 tâches sur 10 ; une version 4 bits casse un code que la référence réussit sur 0 à 2 tâches au total (tous bras confondus : au plus 3 cases) ; l'inverse (un bras réussit là où la référence échoue) : 0 ou 1.
3. Jetons : les trois Q4_0 quittent la référence sur 7 à 10 tâches, le Q4_K_M sur 4 à 7. Le code est plus contraint que la prose, mais les réponses sont plus longues ; la plupart des écarts sont des noms de variables, des commentaires ou un ordre d'instructions, sans effet sur les tests.
4. herbert et llama.cpp Q4_0 identiques entre eux jusqu'au bout sur 4 à 7 tâches ; quand les deux dévient, même point d'écart avec le même jeton dans la majorité des cas.
5. ik Q4_0 s'écarte de llama.cpp Q4_0 (même fichier) sur au moins 3 tâches.
6. Tous les blocs sont extraits et fermés (aucune réponse ne touche 500 jetons) ; tout le Python compile ; au plus une erreur de compilation Rust sur les 15 réponses Rust.
7. Les deux tâches « prédire » sont réussies ou ratées de la même façon par herbert et llama.cpp Q4_0.
