"""Orakel für die Nullreihe: exakte Wahrscheinlichkeiten (n = 2 geschlossene Formel, n = 3 Winkelintegral) gegen einen zweiten Rechenweg (scipy: chi2, quad; zweidimensionales
Integral) und gegen Simulation mit der DP der Demo und dem SplitMix64-Strom (Band 4 Standardfehler)."""

import math

import pytest

pytest.importorskip("scipy")
from scipy.stats import chi2  # noqa: E402

import exact_ref as E  # noqa: E402
import sgo_exact as EX  # noqa: E402
import sgo_segmentation as S  # noqa: E402
from sgo_normal import Stream  # noqa: E402


@pytest.mark.parametrize("c", [0.5, 1.0, 2.0, 3.0])
def test_n2_exakt_gleich_chi2_und_tabellenwerte(c):
    be = c * math.log(2)
    assert abs(EX.p_break_n2(be) - float(chi2.sf(be, 1))) < 1e-14
    soll = {0.5: 0.55606, 1.0: 0.40510, 2.0: 0.23903, 3.0: 0.14929}[c]
    assert abs(EX.p_break_n2(be) - soll) < 5e-6


@pytest.mark.parametrize("c", [0.5, 1.0, 2.0, 3.0])
def test_n3_winkelintegral_gleich_zweidimensionalem_integral(c):
    be = c * math.log(3)
    a = EX.p_nobreak_n3(be)
    assert abs(a - E.p_nobreak_n3_grid(be)) < 1e-9
    assert abs(a - E.p_nobreak_n3_polar(be)) < 1e-9
    soll = {0.5: 0.304008, 1.0: 0.511109, 2.0: 0.752975, 3.0: 0.871957}[c]
    assert abs(a - soll) < 5e-7


@pytest.mark.parametrize("c", [1.0, 3.0])
def test_n2_simulation_mit_der_dp_der_demo_im_band(c):
    be = c * math.log(2)
    R = 40_000
    res = S.dp_batch(Stream(2).normal((R, 2)), [be], 1)
    p = float((res.counts()[:, 0] > 0).mean())
    ex = EX.p_break_n2(be)
    assert abs(p - ex) < 4 * math.sqrt(ex * (1 - ex) / R)


@pytest.mark.parametrize("c", [1.0, 2.0])
def test_n3_simulation_mit_der_dp_der_demo_im_band(c):
    be = c * math.log(3)
    R = 60_000
    res = S.dp_batch(Stream(3).normal((R, 3)), [be], 1)
    p = float((res.counts()[:, 0] == 0).mean())
    ex = EX.p_nobreak_n3(be)
    assert abs(p - ex) < 4 * math.sqrt(ex * (1 - ex) / R)


def test_n3_aufzaehlung_in_numpy_gleich_exakt_im_band():
    for c in (0.5, 2.0):
        be = c * math.log(3)
        mc, se = E.mc_nobreak_n3(be, 200_000, 11)
        assert abs(mc - EX.p_nobreak_n3(be)) < 4 * se


def test_gewinn_einer_festen_stelle_ist_chi2_mit_einem_freiheitsgrad():
    """Gewinn der Teilung bei s = 100 einer Nullreihe n = 200: Anteil über β gleich 2·Φ̄(√β) (feste Stelle, kein Maximum über Stellen)."""
    R = 2000
    X = Stream(8).normal((R, 200))
    be = 3.0
    g = [S.gain(X[i], 0, 200, 100) for i in range(R)]
    p = sum(v > be for v in g) / R
    ex = EX.p_fixed_site(be)
    assert abs(p - ex) < 4 * math.sqrt(ex * (1 - ex) / R)


def test_stellen_mal_einzelwert_ist_keine_schranke():
    assert EX.sites_times_single(200, 1.0, 5) > 1.0        # über 100 %
    assert EX.sites_times_single(200, 2.0, 5) < 1.0
