# Réponses attendues et règles de jugement

Écrites avant toute génération (aucune invite rendue, aucun bras lancé), hachées dans `ATTENDUES.sha256`, jamais retouchées ensuite. Questions dans `questions.json` (construit par `_questions.py`) : 10 questions, chacune posée en anglais (`-en`) et en français (`-fr`) sur le même texte anglais. Le jugement est le mien, fait à la main contre ce fichier, et signalé comme tel dans le README.

## Règles générales

Chaque réponse reçoit deux notes séparées.

1. **Contenu** : juste, partiel ou faux contre la réponse attendue ci-dessous. Une réponse coupée à 150 jetons est jugée sur ce qu'elle contient ; si l'élément demandé n'y est pas encore, elle est fausse et marquée « coupée ». Un ajout faux (un détail inventé à côté de la bonne réponse) rend la réponse partielle.
2. **Langue** : la réponse est-elle dans la langue de la question (oui / non / mêlée). Une citation anglaise dans une réponse française ne compte pas comme mélange.

Citations (c4, l4) : « mot pour mot » tolère les guillemets (droits, typographiques, chevrons ou absents), la ponctuation finale (point, virgule, point d'exclamation) et la casse de la première lettre ; tout autre mot changé, ajouté ou retiré rend la citation fausse. Pour la question française, la phrase anglaise exacte est attendue (une traduction en plus est acceptée) ; une traduction française seule, sans la phrase anglaise, est jugée **partielle**.

## Texte court : « The Temptation of Harringay » (2 689 jetons de texte)

- **c1 (début)** : vers dix heures du matin (« about ten o'clock »). Juste : dix heures. Le matin n'est pas exigé (le texte ne dit que « about ten o'clock »).
- **c2 (milieu)** : du vermillon (« vermilion »). « Rouge vif » seul est partiel ; le rouge (« bright red ») qu'il passe ensuite en travers du tableau est un autre geste.
- **c3 (fin)** : une petite toile de 24 sur 20, émaillée vert pâle (« a small canvas (24 by 20) enamelled a pale green »). Juste : les deux éléments ; un seul des deux : partiel. L'unité n'est pas dans le texte : une unité ajoutée (pouces) est tolérée. « Hedge Sparrow's Egg Tint » donné comme nom de la teinte, en plus du vert pâle, est accepté ; à la place du vert pâle, partiel.
- **c4 (citation)** : `"That was rather hasty of you,"` — la phrase est « That was rather hasty of you ». Toute autre réplique du tableau (« I don't », « It's yourself »…) est fausse.
- **c5 (résumé)** : juste si les deux phrases contiennent : (a) le peintre Harringay travaille au portrait d'un joueur d'orgue de Barbarie italien qui, en le retouchant, prend vie sous les traits d'un diable ; (b) le tableau parle et lui propose des chefs-d'œuvre en échange de son âme (ou un marché) ; (c) Harringay refuse et le recouvre de peinture / d'émail jusqu'à ce qu'il se taise. Un bonus non exigé : on n'a que sa parole, il n'a jamais produit de chef-d'œuvre. Manque un des trois : partiel. Un fait inventé (il accepte le marché, il devient célèbre…) : faux. Plus de deux phrases : noté à part, sans changer le contenu.

## Texte long : « A Moth — Genus Novo » (5 249 jetons de texte)

- **l1 (début)** : 1891. Les années 1863 et 1864 des notes de bas de page sont fausses.
- **l2 (milieu)** : le troisième jour (« on the third day of the diatoms »).
- **l3 (fin)** : dans une chambre capitonnée (« a padded room », un asile est accepté : « The asylum doctor ») ; il dit que le papillon est le fantôme de Pawkins (« the ghost of Pawkins »), donc un spécimen unique qui vaut la peine d'être capturé (ce dernier point non exigé). Juste : les deux ; un seul : partiel. « Une hallucination » à la place du fantôme de Pawkins est faux pour la seconde moitié (c'est l'avis du médecin, pas le sien).
- **l4 (citation)** : `"The eye of faith is no better than the eye of science,"` — la phrase est « The eye of faith is no better than the eye of science ». « You don't see a moth on the edge of the table there? » (dite avant) ou « Genus novo, by heavens! » sont fausses.
- **l5 (résumé)** : juste si les deux phrases contiennent : (a) l'entomologiste Hapley mène une querelle acharnée contre Pawkins, qui meurt après sa dernière attaque ; (b) Hapley, au repos à la campagne, est hanté par un papillon inconnu que personne d'autre ne voit (associé à Pawkins) ; (c) il finit fou, dans une chambre capitonnée / un asile. Manque un des trois : partiel. Un fait inventé (il capture le papillon, Pawkins est vivant…) : faux.
