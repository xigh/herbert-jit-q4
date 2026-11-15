def fusionner_intervalles(intervalles: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervalles:
        return []
    
    # Trier par début d'intervalle
    sorted_intervals = sorted(intervalles, key=lambda x: x[0])
    
    merged = [sorted_intervals[0]]
    
    for current in sorted_intervals[1:]:
        last = merged[-1]
        # Si l'intervalle courant chevauche ou touche le dernier intervalle fusionné
        if current[0] <= last[1]:
            # Fusionner : étendre la borne supérieure si nécessaire
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    
    return merged
