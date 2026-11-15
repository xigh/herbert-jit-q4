def test_4(m): assert m.en_romain(4) == "IV"
def test_1994(m): assert m.en_romain(1994) == "MCMXCIV"
def test_1(m): assert m.en_romain(1) == "I"
def test_3999(m): assert m.en_romain(3999) == "MMMCMXCIX"
def test_ix_xl(m): assert m.en_romain(49) == "XLIX" and m.en_romain(9) == "IX"
def test_tous(m):
    def ref(n):
        t = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
        r = ""
        for v, s in t:
            while n >= v:
                r += s; n -= v
        return r
    assert all(m.en_romain(n) == ref(n) for n in range(1, 4000))
