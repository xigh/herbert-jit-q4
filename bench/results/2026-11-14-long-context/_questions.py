"""Construit questions.json : chaque question posée sur le texte entier, une fois en anglais, une fois en français.
Le texte (anglais) est le même dans les deux langues ; seule la consigne autour et la question changent.
_questions.py  (dans le dossier de la campagne, lit source/*.txt)"""
import json
TXT = {"court": open("source/harringay.txt").read(), "long": open("source/moth.txt").read()}
WRAP = {
    "en": "Read the following text carefully, then answer the question that follows it.\n\n<<<\n{t}>>>\n\nQuestion: {q}",
    "fr": "Lis attentivement le texte suivant (en anglais), puis réponds à la question qui le suit.\n\n<<<\n{t}>>>\n\nQuestion : {q}",
}
Q = [
 ("c1", "court", "début",    "At about what time did Harringay go into his studio at the start of the story?",
                             "À quelle heure environ Harringay entre-t-il dans son atelier au début de l'histoire ?"),
 ("c2", "court", "milieu",   "While retouching the eye, what colour did Harringay load his brush with by mistake, instead of brown?",
                             "En retouchant l'œil, de quelle couleur Harringay a-t-il chargé son pinceau par erreur, au lieu du brun ?"),
 ("c3", "court", "fin",      "What are the dimensions of the canvas Harringay shows as evidence, and what colour is it?",
                             "Quelles sont les dimensions de la toile qu'Harringay montre comme preuve, et de quelle couleur est-elle ?"),
 ("c4", "court", "citation", "Quote, word for word, the first sentence the picture speaks aloud.",
                             "Cite, mot pour mot, la première phrase que le tableau prononce à voix haute."),
 ("c5", "court", "résumé",   "Summarise the story in two sentences.",
                             "Résume l'histoire en deux phrases."),
 ("l1", "long",  "début",    "In what year did Pawkins publish his work on the mesoblast of the Death's Head Moth?",
                             "En quelle année Pawkins publie-t-il son travail sur le mésoblaste du sphinx tête-de-mort (Death's Head Moth) ?"),
 ("l2", "long",  "milieu",   "On which day of his study of diatoms did Hapley first notice the strange moth?",
                             "Quel jour de son étude des diatomées Hapley remarque-t-il pour la première fois l'étrange papillon ?"),
 ("l3", "long",  "fin",      "At the end of the story, where is Hapley spending the rest of his days, and what does he say the moth is?",
                             "À la fin de l'histoire, où Hapley passe-t-il le reste de ses jours, et que dit-il que le papillon est ?"),
 ("l4", "long",  "citation", "Quote, word for word, the sentence Hapley says to the Vicar right after the Vicar answers \"Certainly not.\"",
                             "Cite, mot pour mot, la phrase que Hapley dit au vicaire juste après que celui-ci a répondu « Certainly not »."),
 ("l5", "long",  "résumé",   "Summarise the story in two sentences.",
                             "Résume l'histoire en deux phrases."),
]
out = []
for qid, L, kind, en, fr in Q:
    for lang, q in (("en", en), ("fr", fr)):
        out.append({"id": f"{qid}-{lang}", "longueur": L, "genre": kind, "langue": lang, "question": q,
                    "texte": WRAP[lang].format(t=TXT[L], q=q)})
json.dump(out, open("questions.json", "w"), ensure_ascii=False, indent=1)
print(len(out), "questions")
