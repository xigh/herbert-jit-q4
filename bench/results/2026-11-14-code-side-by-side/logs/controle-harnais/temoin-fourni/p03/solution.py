def recherche_dichotomique(liste: list[int], x: int) -> int:
    lo, hi = 0, len(liste) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if liste[mid] == x:
            return mid
        elif liste[mid] < x:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
