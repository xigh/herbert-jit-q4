import unicodedata
import re


def slug(titre: str) -> str:
    # Normaliser la chaîne pour décomposer les caractères accentués
    normalized = unicodedata.normalize('NFD', titre.lower())
    
    # Ne garder que les caractères de base (enlever les diacritiques)
    ascii_str = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    
    # Remplacer toute séquence de caractères non alphanumériques par un tiret
    slugified = re.sub(r'[^a-z0-9]+', '-', ascii_str)
    
    # Enlever les tirets au début et à la fin
    slugified = slugified.strip('-')
    
    return slugified
