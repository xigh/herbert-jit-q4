def test_impair(m): assert m.mediane([3, 1, 2]) == 2
def test_pair(m): assert m.mediane([4, 1, 3, 2]) == 2.5
def test_deux(m): assert m.mediane([1, 2]) == 1.5
def test_un(m): assert m.mediane([7]) == 7
def test_vide(m):
    try:
        m.mediane([])
    except ValueError:
        return
    raise AssertionError("pas de ValueError")
def test_non_modifiee(m):
    x = [5, 1, 4, 2]
    m.mediane(x)
    assert x == [5, 1, 4, 2]
def test_flottants(m): assert m.mediane([0.5, -1.5, 2.0, 10.0]) == 1.25
