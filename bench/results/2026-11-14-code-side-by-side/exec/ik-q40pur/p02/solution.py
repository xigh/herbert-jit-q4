import unicodedata
import re

def slug(titre: str) -> str:
    # 1. Mettre en minuscules
    titre = titre.lower()
    
    # 2. Décomposer les caractères accentués (NFD) pour séparer les accents
    # ex: 'é' devient 'e' + '\u0301'
    titre = unicodedata.normalize('NFD', titre)
    
    # 3. Supprimer les caractères diacritiques (les accents)
    # On garde uniquement les caractères qui ne sont pas des diacritiques
    titre = ''.join(c for c in titre if not unicodedata.combining(c))
    
    # 4. Remplacer toute séquence de caractères non alphanumériques par un tiret
    # [^a-z0-9] correspond à tout ce qui n'est pas une lettre minuscule ou un chiffre
    titre = re.sub(r'[^a-z0-9]+', '-', titre)
    
    # 5. Enlever les tirets au début et à la fin
    titre = titre.strip('-')
    
    return titre
