def test_vide(m): assert m.fusionner_intervalles([]) == []
def test_un(m): assert m.fusionner_intervalles([(2, 2)]) == [(2, 2)]
def test_desordre(m): assert m.fusionner_intervalles([(8, 10), (1, 3), (2, 6), (15, 18)]) == [(1, 6), (8, 10), (15, 18)]
def test_touchent(m): assert m.fusionner_intervalles([(3, 5), (1, 3)]) == [(1, 5)]
def test_inclus(m): assert m.fusionner_intervalles([(1, 10), (2, 3), (4, 5)]) == [(1, 10)]
def test_disjoints(m): assert m.fusionner_intervalles([(5, 6), (1, 2)]) == [(1, 2), (5, 6)]
def test_negatifs(m): assert m.fusionner_intervalles([(-5, -1), (-2, 0), (2, 4)]) == [(-5, 0), (2, 4)]
def test_non_modifiee(m):
    x = [(4, 5), (1, 2)]
    m.fusionner_intervalles(x)
    assert x == [(4, 5), (1, 2)]
