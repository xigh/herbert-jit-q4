"""Extrait deux nouvelles de Project Gutenberg n° 12750 (H. G. Wells, The Stolen Bacillus and Other Incidents, 1895) :
fins de ligne CRLF ôtées, chaque paragraphe remis sur une ligne (les retours de ligne internes deviennent une espace),
paragraphes séparés par une ligne vide, rien d'autre retouché (soulignés _italiques_ et appels de notes [A] gardés).
_extract.py  (dans le dossier de la campagne : lit source/pg12750.txt, écrit source/harringay.txt et source/moth.txt)"""
import re
L = open("source/pg12750.txt", encoding="utf-8").read().replace("\r", "").split("\n")
def story(title, nxt):
    a = L.index(title); b = L.index(nxt)          # lignes de titre seules (la table des matières les préfixe d'un chiffre romain)
    body = "\n".join(L[a:b]).strip()
    return "\n\n".join(re.sub(r"\s*\n\s*", " ", p.strip()) for p in re.split(r"\n\s*\n", body) if p.strip()) + "\n"
open("source/harringay.txt", "w").write(story("THE TEMPTATION OF HARRINGAY", "THE FLYING MAN"))
open("source/moth.txt", "w").write(story("A MOTH--GENUS NOVO", "THE TREASURE IN THE FOREST"))
