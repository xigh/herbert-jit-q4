def fusionner_intervalles(intervalles: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervalles:
        return []
    
    # Trier les intervalles par leur début
    triees = sorted(intervalles, key=lambda x: x[0])
    
    resultat = [triees[0]]
    
    for debut, fin in triees[1:]:
        dernier_debut, dernier_fin = resultat[-1]
        
        # Si l'intervalle courant se chevauche ou touche le dernier ajouté
        if debut <= dernier_fin:
            # Fusionner en prenant le maximum des fins
            resultat[-1] = (dernier_debut, max(dernier_fin, fin))
        else:
            resultat.append((debut, fin))
            
    return resultat
