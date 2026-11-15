def fusionner_intervalles(intervalles):
    r = []
    for a, b in sorted(intervalles):
        if r and a <= r[-1][1]:
            r[-1] = (r[-1][0], max(r[-1][1], b))
        else:
            r.append((a, b))
    return r
