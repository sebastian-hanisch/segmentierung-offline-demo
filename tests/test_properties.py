"""Eigenschaften der Verfahren (je 20 bis 50 Zufallsinstanzen, feste Zufalls-Seeds): Invarianzen, Monotonie, Grenzfälle. Die Instanzen sind Normalreihen aus dem SplitMix64-Strom."""

import numpy as np
import pytest

import sgo_segmentation as S
from sgo_normal import Stream


def _series(seed, R=30, n=60, jump=2.0):
    X = Stream(seed).normal((R, n))
    X[:, n // 2:] += jump
    return X


def test_dp_zielwert_ist_kleiner_oder_gleich_dem_jeder_anderen_zerlegung():
    X = _series(1, R=30, n=80)
    for r in range(X.shape[0]):
        for m in (1, 3, 5):
            b = 6.0
            dp = S.dp_batch(X[r], [b], m).F[0, 0]
            for other in (S.pelt(X[r], b, m).breaks, S.pelt(X[r], b, m, delayed=True).breaks, S.binseg(X[r], b, m)[0], [], [40]):
                if not all(c >= m and 80 - c >= m for c in other) or any(y - x < m for x, y in zip(other, other[1:])):
                    continue
                assert S.objective(X[r], other, b) >= dp - 1e-9


@pytest.mark.parametrize("shift", [1e3, 1e6, 1e8, 1e10])
def test_verschiebung_der_reihe_aendert_die_brueche_nicht_auch_bei_grossem_mittel(shift):
    """Befund J der Planung: ohne Zentrieren verliert die Summenformel ab etwa 1e6 bis 1e7 Stellen; die Demo zentriert, die Brüche bleiben gleich."""
    X = _series(2, R=40, n=100, jump=1.0)
    base = S.dp_batch(X, [2 * np.log(100)], 5)
    shifted = S.dp_batch(X + shift, [2 * np.log(100)], 5)
    for r in range(40):
        assert shifted.breaks(r, 0) == base.breaks(r, 0)
    assert [S.pelt(X[r] + shift, 9.0, 5).breaks for r in range(10)] == [S.pelt(X[r], 9.0, 5).breaks for r in range(10)]
    assert [S.binseg(X[r] + shift, 9.0, 5)[0] for r in range(10)] == [S.binseg(X[r], 9.0, 5)[0] for r in range(10)]


def test_skalierung_der_reihe_mit_a_und_der_strafe_mit_a_quadrat_gibt_gleiche_brueche():
    X = _series(3, R=30, n=60)
    for a in (0.1, 3.0, 25.0):
        r1 = S.dp_batch(X, [5.0], 3)
        r2 = S.dp_batch(a * X, [5.0 * a * a], 3)
        for r in range(30):
            assert r1.breaks(r, 0) == r2.breaks(r, 0)


def test_spiegelung_der_reihe_spiegelt_die_zielwerte_und_die_brueche():
    X = _series(4, R=30, n=60)
    f = S.dp_batch(X, [6.0], 4)
    g = S.dp_batch(X[:, ::-1], [6.0], 4)
    for r in range(30):
        assert f.F[r, 0] == pytest.approx(g.F[r, 0], abs=1e-9)
        a, b = f.breaks(r, 0), [60 - s for s in g.breaks(r, 0)][::-1]
        if a != b:                                       # nur bei Gleichstand: dann gleicher Zielwert
            assert S.objective(X[r], a, 6.0) == pytest.approx(S.objective(X[r], b, 6.0), abs=1e-9)


def test_zahl_der_brueche_nimmt_mit_der_strafe_nicht_zu():
    X = _series(5, R=40, n=100, jump=1.0)
    betas = np.array([0.5, 1, 2, 4, 8, 16, 32, 64.0])
    cnt = S.dp_batch(X, betas, 3).counts()
    assert np.all(np.diff(cnt, axis=1) <= 0)           # bei exakten Gleichständen (Wahrscheinlichkeit 0 für reelle Werte) nicht streng, hier nie zunehmend


def test_zielwert_nimmt_mit_der_mindestlaenge_nicht_ab():
    X = _series(6, R=30, n=60)
    prev = None
    for m in (1, 2, 3, 5, 10):
        f = S.dp_batch(X, [5.0], m).F[:, 0]
        if prev is not None:
            assert np.all(f >= prev - 1e-9)
        prev = f


def test_beta_null_und_mindestlaenge_1_gibt_n_minus_1_brueche():
    X = _series(7, R=10, n=30)
    assert np.all(S.dp_batch(X, [0.0], 1).counts() == 29)
    assert S.dp_batch(X[0], [0.0], 1).F[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_sehr_grosse_strafe_gibt_keinen_bruch_und_konstante_reihe_auch_keinen():
    X = _series(8, R=10, n=40)
    assert np.all(S.dp_batch(X, [1e9], 1).counts() == 0)
    assert S.dp_batch(np.full(25, 7.5), [0.0001], 1).counts()[0, 0] == 0
    assert S.pelt(np.full(25, 7.5), 1.0, 1).breaks == [] and S.binseg(np.full(25, 7.5), 1.0, 1)[0] == []


def test_gewinnformel_und_nu_konvention():
    x = np.array([1.0, 3.0, 2.0, 8.0, 9.0])
    lo, hi, tau = 0, 5, 3
    ml, mr = x[:tau].mean(), x[tau:].mean()
    assert S.gain(x, lo, hi, tau) == pytest.approx(tau * (hi - tau) / hi * (ml - mr) ** 2, rel=1e-12)
    r = S.dp_batch(np.array([0, 0, 0, 5, 5, 5, 5, 5], float), [1.0], 1)
    assert r.breaks() == [3]                              # „3 Beobachtungen davor“: x[0..2] vor, x[3..] nach der Änderung


def test_zentrieren_aendert_die_zerlegung_nicht_und_die_reihe_des_aufrufers_bleibt_unveraendert():
    x = np.array([5.0, 5.2, 4.9, 9.0, 9.1, 8.8])
    kopie = x.copy()
    S.dp_batch(x, [2.0], 1), S.pelt(x, 2.0, 1), S.binseg(x, 2.0, 1), S.objective(x, [3], 2.0)
    assert np.array_equal(x, kopie)
    assert S.center(x).mean() == pytest.approx(0.0, abs=1e-12)


def test_gleichstand_der_dp_waehlt_den_kleinsten_startpunkt():
    """Vier gleiche Werte, Mindestlänge 2, β = 0: „kein Bruch“ und „Bruch bei 2“ haben beide den Zielwert 0; bei F(4) gewinnt der Startpunkt s = 0 vor s = 2."""
    x = np.zeros(4)
    e = S.enumerate_best(x, 0.0, 2)
    assert e.count == 2 and sorted(e.optimal) == [[], [2]]
    r = S.dp_batch(x, [0.0], 2)
    assert r.F[0, 0] == pytest.approx(0.0, abs=1e-12) and r.breaks() == []
