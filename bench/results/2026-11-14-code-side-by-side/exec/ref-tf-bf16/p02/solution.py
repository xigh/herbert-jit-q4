import unicodedata
import re

def slug(titre: str) -> str:
    # Normaliser la chaîne pour décomposer les caractères accentués
    nfkd_form = unicodedata.normalize('NFKD', titre)
    # Ne garder que les caractères de base (enlever les diacritiques)
    ascii_form = ''.join([c for c in nfkd_form if not unicodedata.combining(c)])
    # Convertir en minuscules
    ascii_form = ascii_form.lower()
    # Remplacer toute séquence de caractères non alphanumériques par un tiret
    slug = re.sub(r'[^a-z0-9]+', '-', ascii_form)
    # Enlever les tirets au début et à la fin
    slug = slug.strip('-')
    return slug
