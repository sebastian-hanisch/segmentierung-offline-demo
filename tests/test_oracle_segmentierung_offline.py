"""Orakel: die Verfahren der Demo (Gleitkomma, vektorisiert, zentriert) gegen einen anderen Rechenweg: Aufzählung aller Zerlegungen und DP/PELT/binäre Segmentierung
mit `fractions.Fraction` auf ganzzahligen Reihen (`exact_ref`, kein Code der Demo). Instanzen: feste Zufalls-Seeds (Python `random`, nur zur Erzeugung der Eingaben).

Gleichstände: bei ganzzahligen Reihen kommen exakte Gleichstände vor; die Gleitkomma-Rechnung kann dann eine andere der gleichwertigen Zerlegungen wählen. Verglichen wird darum der
**Zielwert**, und die Bruchmenge nur, wo die optimale Zerlegung eindeutig ist (Aufzählung)."""

import random
from fractions import Fraction as Fr

import numpy as np
import pytest

import exact_ref as E
import sgo_segmentation as S


def _instances(seed, count, n_hi=11, ml_choices=(1, 1, 2, 3), jumps=2):
    rnd = random.Random(seed)
    out = []
    while len(out) < count:
        m = rnd.choice(ml_choices)
        n = rnd.randint(max(3, m), n_hi)
        x = [rnd.randint(-4, 4) for _ in range(n)]
        for _ in range(rnd.randint(0, jumps)):
            p = rnd.randint(1, n - 1)
            h = rnd.randint(-5, 5)
            x = [v + (h if i >= p else 0) for i, v in enumerate(x)]
        out.append((x, Fr(rnd.choice([1, 2, 3, 5, 8, 13]), rnd.choice([1, 2])), m))
    return out


def test_exakte_dp_gleich_aufzaehlung_300_instanzen():
    """Das Orakel selbst: Fraction-DP gegen Fraction-Aufzählung (300 Instanzen; im Scratch-Lauf der Planung 3 000 mit 0 Abweichungen)."""
    for x, be, m in _instances(1601, 300):
        a, b = E.dp_exact(x, be, m), E.brute_exact(x, be, m)
        assert a[0] == b[0] and a[1] in b[1]


def test_float_dp_der_demo_gleich_exakter_dp_zielwert_und_bruchmenge():
    n_unique = 0
    for x, be, m in _instances(1602, 300, n_hi=12):
        ex = E.dp_exact(x, be, m)
        r = S.dp_batch(np.array(x, float), [float(be)], m)
        assert abs(r.F[0, 0] - float(ex[0])) < 1e-9
        optimal = E.brute_exact(x, be, m)[1]
        if len(optimal) == 1:               # eindeutiges Optimum: auch die Bruchmenge ist festgelegt
            n_unique += 1
            assert r.breaks() == optimal[0]
        else:                               # Gleichstand: die gewählte Zerlegung muss eine der optimalen sein
            assert r.breaks() in optimal
    assert n_unique > 100                    # die Prüfung der Bruchmenge ist nicht leer


def test_aufzaehlung_der_demo_gleich_exakter_aufzaehlung():
    for x, be, m in _instances(1603, 150, n_hi=12):
        e = S.enumerate_best(x, float(be), m)
        b = E.brute_exact(x, be, m)
        assert abs(e.best - float(b[0])) < 1e-9
        assert e.count == len(E.compositions(len(x), m))
        assert sorted(e.optimal) == sorted(b[1])


def test_pelt_gleich_dp_bei_mindestlaenge_1_exakt_und_in_der_demo():
    for x, be, _ in _instances(1604, 300, n_hi=12, ml_choices=(1,)):
        dp = E.dp_exact(x, be, 1)[0]
        assert E.pelt_exact(x, be, 1)[0] == dp
        assert abs(S.pelt(x, float(be), 1).objective - float(dp)) < 1e-9


def test_verzoegertes_streichen_gleich_dp_in_allen_instanzen():
    for x, be, m in _instances(1605, 300, n_hi=14, ml_choices=(1, 2, 3, 5)):
        if len(x) < m:
            continue
        dp = E.dp_exact(x, be, m)[0]
        assert E.pelt_exact(x, be, m, delayed=True)[0] == dp
        assert abs(S.pelt(x, float(be), m, delayed=True).objective - float(dp)) < 1e-9


def test_pelt_weicht_bei_mindestlaenge_2_manchmal_ab_nie_nach_unten():
    """Erwarteter Befund (nicht ein Fehler): PELT mit K = 0 hat bei Mindestlänge >= 2 manchmal einen größeren Zielwert; nie einen kleineren als das Optimum."""
    worse = 0
    for x, be, m in _instances(1606, 600, n_hi=16, ml_choices=(2,)):
        dp = E.dp_exact(x, be, m)[0]
        pv = S.pelt(x, float(be), m).objective
        assert pv >= float(dp) - 1e-9
        worse += pv > float(dp) + 1e-9
    assert worse > 0


def test_binaere_segmentierung_nie_besser_als_dp_und_gleich_exakter_fassung():
    worse = same_set = unique = 0
    for x, be, m in _instances(1607, 400, n_hi=14, ml_choices=(1, 2), jumps=3):
        dp = E.dp_exact(x, be, m)
        bs = S.binseg(x, float(be), m)[0]
        v = S.objective(x, bs, float(be))
        assert v >= float(dp[0]) - 1e-9                           # nie besser
        worse += v > float(dp[0]) + 1e-9
        ex_bs = E.binseg_exact(x, be, m)
        if E.objective_exact(x, ex_bs, be) != E.objective_exact(x, bs, be):   # Gleitkomma-Gleichstand der Teilung kann eine andere Stelle wählen
            continue
        unique += 1
        same_set += bs == ex_bs
    assert worse > 0 and unique > 300 and same_set > 0.95 * unique


def test_zielwert_ist_die_summe_der_kosten_plus_beta_mal_zahl_der_brueche():
    for x, be, m in _instances(1608, 100):
        r = S.dp_batch(np.array(x, float), [float(be)], m)
        assert abs(S.objective(x, r.breaks(), float(be)) - float(E.objective_exact(x, r.breaks(), be))) < 1e-9


def test_float_dp_gegen_exakt_auf_zufallsreihen_mit_nichtganzen_werten():
    """Nicht ganzzahlige Eingaben (Hundertstel, als Brüche exakt): Zielwert gleich, Abweichung kleiner als 1e-9."""
    rnd = random.Random(1609)
    for _ in range(100):
        n = rnd.randint(4, 12)
        m = rnd.choice([1, 2, 3])
        if n < m:
            continue
        xs = [Fr(rnd.randint(-300, 300), 100) for _ in range(n)]
        be = Fr(rnd.randint(50, 800), 100)
        ex = E.dp_exact(xs, be, m)[0]
        r = S.dp_batch(np.array([float(v) for v in xs]), [float(be)], m)
        assert abs(r.F[0, 0] - float(ex)) < 1e-9


@pytest.mark.parametrize("m", [1, 2, 5])
def test_dp_batch_stapel_gleich_einzelrechnung(m):
    rng = np.random.default_rng(5)
    X = rng.standard_normal((6, 40)) + np.where(np.arange(40) >= 20, 1.5, 0.0)
    betas = [1.0, 3.0, 8.0]
    stapel = S.dp_batch(X, betas, m)
    for r in range(6):
        for p, b in enumerate(betas):
            einzel = S.dp_batch(X[r], [b], m)
            assert stapel.breaks(r, p) == einzel.breaks() and stapel.F[r, p] == pytest.approx(einzel.F[0, 0], abs=1e-9)
