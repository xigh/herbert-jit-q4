# Prédictions

Écrites avant toute génération : aucune invite rendue, aucun bras lancé, aucune sortie de la campagne des 15 questions courtes lue au-delà des temps de la référence (`logs/ref-tf-bf16.log`) et des longueurs de réponse de `logs-chaine.out`. Hachées dans `PREDICTIONS.sha256`, jamais retouchées ensuite. Ce que je sais en les écrivant : sur la perplexité (wiki, fr), Q4_0 pur de llama.cpp et herbert grille 8 ont une KL de 0,072 à 0,077 contre la référence bf16, Q4_K_M de 0,019 ; dans la campagne courte, herbert et llama.cpp Q4_0 rendent des réponses de longueurs différentes sur au moins 4 questions sur 15 (q07, q10, q11, q15), donc ne sont pas identiques au jeton.

Bras : référence transformers bf16 ; herbert 68c3a96 `--q4` (grille par défaut) ; llama.cpp 5676dfbe0 Q4_0 pur ; llama.cpp Q4_K_M ; ik_llama.cpp 7aa7e014 Q4_0 pur. 20 invites : 2 textes × 5 questions × 2 langues. « Quitte la référence » = premier jeton généré différent de celui de la référence.

## Justesse

1. La référence bf16 est juste sur au moins 18 des 20 réponses (contenu). Ses erreurs probables, s'il y en a : la citation c4 ou l4 en français (traduite seulement), ou un résumé partiel.
2. Chaque bras Q4 est juste sur au moins 16 des 20. Les détails ponctuels (c1, c2, c3, l1, l2, l3) sont justes partout, dans les deux langues, pour les cinq bras. Si une erreur tombe, elle tombe sur une citation (c4, l4) ou un résumé.
3. Aucun bras Q4 ne fait mieux que la référence en nombre de réponses justes.
4. Q4_K_M a au plus autant d'erreurs que les bras Q4_0 (herbert, llama.cpp, ik).
5. Langue : toutes les réponses aux questions françaises sont en français, pour les cinq bras ; les citations françaises donnent la phrase anglaise (avec ou sans traduction) dans au moins la moitié des cas.

## Divergence contre la référence

6. Chaque bras Q4 quitte la référence sur au moins la moitié des 20 invites ; sur les réponses courtes (un nombre, une couleur), il la quitte plus rarement que sur les résumés.
7. Q4_K_M quitte la référence moins souvent que les trois bras Q4_0.
8. L'écart grandit avec le contexte : pour chaque bras Q4, le nombre d'invites où il quitte la référence est plus grand sur le texte long (5 249 jetons) que sur le court (2 689), et le premier jeton divergent arrive plus tôt en médiane. Je mets cette prédiction à 60 % : le rapport de longueur n'est que de 2, et les réponses courtes laissent peu de place à la divergence.

## herbert contre llama.cpp Q4_0

9. herbert et llama.cpp Q4_0 ne sont pas identiques au jeton sur les 20 : ils diffèrent sur au moins 4 invites. Ils sont identiques entre eux plus souvent que chacun ne l'est avec la référence.
10. ik_llama.cpp Q4_0 est plus proche de llama.cpp Q4_0 que de herbert (plus d'invites identiques au jeton).

## Temps

11. Prefill des serveurs Q4 sur Ryzen 9 9900X, 12 fils : entre 80 et 250 jetons/s sur 5 300 jetons ; decode plus lent sur le texte long que sur le court, de 5 à 25 %.
