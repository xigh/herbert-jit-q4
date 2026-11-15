def mediane(nombres: list[float]) -> float:
    if not nombres:
        raise ValueError("liste vide")
    s = sorted(nombres)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2] + s[n // 2 + 1]) / 2
