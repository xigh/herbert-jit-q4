import unicodedata
import re

def slug(titre: str) -> str:
    # 1. Mettre en minuscules
    titre = titre.lower()
    
    # 2. Décomposer les caractères accentués (NFD) et ne garder que la base
    # unicodedata.normalize('NFD', titre) décompose 'é' en 'e' + accent aigu
    # On filtre ensuite pour ne garder que les caractères de base (sans les diacritiques)
    slug_text = unicodedata.normalize('NFD', titre)
    slug_text = ''.join(c for c in slug_text if unicodedata.category(c) != 'Mn')
    
    # 3. Remplacer toute suite de caractères non alphanumériques par un tiret
    # [^a-z0-9] correspond à tout ce qui n'est pas une lettre minuscule ou un chiffre
    slug_text = re.sub(r'[^a-z0-9]+', '-', slug_text)
    
    # 4. Enlever les tirets au début et à la fin
    slug_text = slug_text.strip('-')
    
    return slug_text
