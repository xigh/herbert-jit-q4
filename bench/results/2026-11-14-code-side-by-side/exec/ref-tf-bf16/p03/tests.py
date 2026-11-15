def test_milieu(m): assert m.recherche_dichotomique([1, 3, 5, 7, 9], 5) == 2
def test_premier(m): assert m.recherche_dichotomique([1, 3, 5, 7, 9], 1) == 0
def test_dernier(m): assert m.recherche_dichotomique([1, 3, 5, 7, 9], 9) == 4
def test_un_seul(m): assert m.recherche_dichotomique([4], 4) == 0
def test_vide(m): assert m.recherche_dichotomique([], 4) == -1
def test_absent(m): assert m.recherche_dichotomique([1, 3, 5, 7, 9], 4) == -1
def test_absent_bords(m): assert m.recherche_dichotomique([1, 3, 5], 0) == -1 and m.recherche_dichotomique([1, 3, 5], 6) == -1
def test_tous(m):
    l = list(range(0, 200, 3))
    assert all(m.recherche_dichotomique(l, v) == i for i, v in enumerate(l))
    assert all(m.recherche_dichotomique(l, v + 1) == -1 for v in l)
