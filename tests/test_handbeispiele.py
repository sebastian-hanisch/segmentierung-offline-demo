"""Handbeispiele H 1 bis H 8 des Plans als exakte Tests. Die Erwartungswerte stehen als Rechnung von Hand (Brüche, Ganzzahlen), nicht als Abschrift des Codes; geprüft werden die
exakte Referenz (`exact_ref`, `fractions`) und die Demo-Verfahren (`sgo_segmentation`, Gleitkomma, vektorisiert) gegeneinander und gegen die Handrechnung."""

import itertools
import math
from fractions import Fraction as Fr

import numpy as np
import pytest

import exact_ref as E
import sgo_exact as EX
import sgo_online as ON
import sgo_segmentation as S


# ------------------------------------------------------------------------------------------------------------------------------ H 1: ein Bruch, DP-Tabelle
def test_h1_kosten_gewinn_und_tabelle(sprung):
    sums = E.prefix_sums(sprung)
    assert E.seg_cost(sums, 0, 6) == 6                      # 12 - 36/6
    assert E.seg_cost(sums, 0, 5) == Fr(24, 5)              # 8 - 16/5
    assert E.gain_exact(sprung, 0, 6, 3) == Fr(3 * 3, 6) * (2 - 0) ** 2 == 6   # τ(n−τ)/n · (Mittel links − Mittel rechts)²
    assert S.gain(sprung, 0, 6, 3) == pytest.approx(6.0, abs=1e-12)
    f, brk, tab = E.dp_exact(sprung, Fr(2), 1)
    assert tab == [-2, 0, 0, 0, 2, 2, 2] and f == 2 and brk == [3]
    r = S.dp_batch(np.array(sprung, float), [2.0], 1)
    assert np.allclose(r.table[0, 0], [-2, 0, 0, 0, 2, 2, 2], atol=1e-12) and r.breaks() == [3] and r.F[0, 0] == pytest.approx(2.0)


def test_h1_nur_eine_der_32_zerlegungen_erreicht_den_optimalen_wert(sprung):
    best, arg = E.brute_exact(sprung, Fr(2), 1)
    assert best == 2 and arg == [[3]]
    assert S.count_decompositions(6, 1) == 32 == len(E.compositions(6, 1))
    e = S.enumerate_best(sprung, 2.0, 1)
    assert e.count == 32 and e.optimal == [[3]] and e.best == pytest.approx(2.0)
    assert E.objective_exact(sprung, [], Fr(2)) == 6 and E.objective_exact(sprung, [3], Fr(2)) == 2
    assert S.objective(sprung, [], 2.0) == pytest.approx(6.0) and S.objective(sprung, [3], 2.0) == pytest.approx(2.0)


# ------------------------------------------------------------------------------------------------------------------------------ H 2: Zahl der Zerlegungen
def test_h2_zerlegungszahlen():
    for n in range(1, 11):
        assert S.count_decompositions(n, 1) == 2 ** (n - 1) == len(E.compositions(n, 1))
    fib = {1: 0, 2: 1, 3: 1}
    for n in range(4, 15):
        fib[n] = fib[n - 1] + fib[n - 2]
    for n in range(2, 15):
        assert S.count_decompositions(n, 2) == fib[n] == len(E.compositions(n, 2))
    assert S.count_decompositions(6, 2) == 5          # 6, 2+4, 4+2, 3+3, 2+2+2
    assert S.count_decompositions(1, 2) == 0


# ------------------------------------------------------------------------------------------------------------------------------ H 4: Puls, binäre Segmentierung blind
def test_h4_puls_kosten_und_einzelgewinn(puls):
    sums = E.prefix_sums(puls)
    assert E.seg_cost(sums, 0, 9) == 8                  # 12 - 36/9
    assert E.gain_exact(puls, 0, 9, 3) == 2 and E.gain_exact(puls, 0, 9, 6) == 2
    assert max(E.gain_exact(puls, 0, 9, t) for t in range(1, 9)) == 2
    assert E.objective_exact(puls, [3, 6], 0) == 0       # beide Brüche zusammen entfernen alle Kosten
    assert S.gain(puls, 0, 9, 3) == pytest.approx(2.0) and S.gain(puls, 0, 9, 6) == pytest.approx(2.0)


@pytest.mark.parametrize("beta", [2, Fr(5, 2), 3, Fr(7, 2)])
def test_h4_fenster_2_bis_unter_4_binaer_blind_dp_findet_beide(puls, beta):
    assert E.binseg_exact(puls, beta, 1) == []
    v, br, _ = E.dp_exact(puls, beta, 1)
    assert br == [3, 6] and v == 2 * beta
    assert E.brute_exact(puls, beta, 1)[1] == [[3, 6]]
    bs, steps = S.binseg(puls, float(beta), 1)
    assert bs == [] and max(s[3] for s in steps) == pytest.approx(2.0)
    r = S.dp_batch(np.array(puls, float), [float(beta)], 1)
    assert r.breaks() == [3, 6] and r.F[0, 0] == pytest.approx(2 * float(beta))


def test_h4_ausserhalb_des_fensters(puls):
    assert E.binseg_exact(puls, Fr(3, 2), 1) == [3, 6]          # kleine Strafe: auch binär findet beide
    assert S.binseg(puls, 1.5, 1)[0] == [3, 6]
    assert E.dp_exact(puls, Fr(4), 1)[0] == 8                     # Gleichstand 8 = 2·4
    assert E.dp_exact(puls, Fr(5), 1)[1] == [] and S.dp_batch(np.array(puls, float), [5.0], 1).breaks() == []


# ------------------------------------------------------------------------------------------------------------------------------ H 5: PELT-Gegenbeispiel
def test_h5_gegenbeispiel_von_hand():
    x = [2, 1, 0, 0, 2]
    assert len(E.compositions(5, 2)) == 3                          # 5 = 5 = 2+3 = 3+2
    assert E.objective_exact(x, [], 1) == 4
    assert E.objective_exact(x, [2], 1) == Fr(25, 6)               # 1/2 + 8/3 + 1
    assert E.objective_exact(x, [3], 1) == 5                       # 2 + 2 + 1
    assert E.dp_exact(x, 1, 2)[:2] == (4, [])
    pv, pb, log = E.pelt_exact(x, 1, 2, trace=True)
    assert pv == Fr(25, 6) and pb == [2]
    assert log[2] == [0] and log[3] == [0] and log[4] == [2]       # bei t = 4 fällt der Start 0 (7/4 >= 3/2)
    dv, db = E.pelt_exact(x, 1, 2, delayed=True)
    assert dv == 4 and db == []


def test_h5_gegenbeispiel_in_der_demo():
    x = np.array([2, 1, 0, 0, 2], float)
    p = S.pelt(x, 1.0, 2, trace=True)
    d = S.pelt(x, 1.0, 2, delayed=True, trace=True)
    r = S.dp_batch(x, [1.0], 2)
    assert r.F[0, 0] == pytest.approx(4.0) and r.breaks() == []
    assert p.objective == pytest.approx(25 / 6) and p.breaks == [2] and p.log[4] == [2] and p.log[3] == [0]
    assert d.objective == pytest.approx(4.0) and d.breaks == []
    assert S.enumerate_best(x, 1.0, 2).count == 3


def test_h5_alle_1024_reihen_n5_40_und_14_abweichungen_exakt():
    cnt = {1: 0, 2: 0}
    cnt_del = cnt1 = 0
    for row in itertools.product(range(4), repeat=5):
        for be in (1, 2):
            a = E.dp_exact(list(row), be, 2)[0]
            cnt[be] += a != E.pelt_exact(list(row), be, 2)[0]
            cnt_del += a != E.pelt_exact(list(row), be, 2, delayed=True)[0]
            cnt1 += E.dp_exact(list(row), be, 1)[0] != E.pelt_exact(list(row), be, 1)[0]
    assert (cnt[1], cnt[2]) == (40, 14) and cnt_del == 0 and cnt1 == 0


def test_h5_alle_1024_reihen_n5_in_der_demo_gleitkomma():
    """Dasselbe mit den Verfahren der Demo (Gleitkomma): 40 und 14 von 1024, verzögert 0; die Zählung hat keinen Gleichstand-Spielraum (nur der Zielwert zählt)."""
    cnt = {1: 0, 2: 0}
    cnt_del = 0
    for row in itertools.product(range(4), repeat=5):
        x = np.array(row, float)
        for be in (1, 2):
            dp = S.dp_batch(x, [float(be)], 2).F[0, 0]
            cnt[be] += abs(S.pelt(x, float(be), 2).objective - dp) > 1e-9
            cnt_del += abs(S.pelt(x, float(be), 2, delayed=True).objective - dp) > 1e-9
    assert (cnt[1], cnt[2]) == (40, 14) and cnt_del == 0


def test_h5_beweisbausteine_kostenregel_und_folgerung():
    """(i) C(s,t) + C(t,T) <= C(s,T); (ii) ist s bei t markiert (F(s) + C(s,t) >= F(t)), so gilt F(s) + C(s,T) >= F(t) + C(t,T) für jedes T > t."""
    import random
    rnd = random.Random(11)
    n_tri = n_mark = 0
    for _ in range(60):
        xx = [rnd.randint(-4, 4) for _ in range(9)]
        sm = E.prefix_sums(xx)
        be = Fr(rnd.choice([1, 2, 3]))
        Fv = E.dp_exact(xx, be, 1)[2]
        for s_ in range(9):
            for t_ in range(s_ + 1, 9):
                for T_ in range(t_ + 1, 10):
                    n_tri += 1
                    assert E.seg_cost(sm, s_, t_) + E.seg_cost(sm, t_, T_) <= E.seg_cost(sm, s_, T_)
                    if Fv[s_] + E.seg_cost(sm, s_, t_) >= Fv[t_]:
                        n_mark += 1
                        assert Fv[s_] + E.seg_cost(sm, s_, T_) >= Fv[t_] + E.seg_cost(sm, t_, T_)
    assert n_tri > 3000 and n_mark > 300


# ------------------------------------------------------------------------------------------------------------------------------ H 6: Nullreihe
def test_h6_n2_formel_integral_und_dp():
    from scipy import integrate
    from scipy.stats import norm
    for be in (0.5, 1.0, 2.0, 2 * math.log(2)):
        val, _ = integrate.quad(lambda d: norm.pdf(d), math.sqrt(be), np.inf)
        assert abs(EX.p_break_n2(be) - 2 * val) < 1e-9
        assert abs(EX.p_break_n2(be) - 2 * norm.sf(math.sqrt(be))) < 1e-14
    assert abs(EX.p_break_n2(2 * math.log(2)) - 0.23903) < 5e-6


def test_h6_n3_zwei_wege():
    for be in (1.0, 2.0, 3.0, 2 * math.log(3)):
        a, b = EX.p_nobreak_n3(be), E.p_nobreak_n3_grid(be)
        assert abs(a - b) < 1e-7, be
        assert abs(a - E.p_nobreak_n3_polar(be)) < 1e-9
    assert EX.p_nobreak_n3(1e-6) < 1e-5 and EX.p_nobreak_n3(60.0) > 1 - 1e-12
    assert abs(EX.p_nobreak_n3(2 * math.log(3)) - 0.752975) < 5e-7


def test_h6_feste_stelle_und_stellen_mal_einzelwert():
    assert abs(EX.p_fixed_site(math.log(200)) - 0.0213) / 0.0213 < 0.05
    assert abs(EX.p_fixed_site(2 * math.log(200)) - 0.001133) / 0.001133 < 0.05
    assert EX.n_sites(200, 5) == 191
    assert abs(EX.sites_times_single(200, 1.0, 5) - 4.077) < 5e-3 and abs(EX.sites_times_single(200, 2.0, 5) - 0.216) < 5e-3


# ------------------------------------------------------------------------------------------------------------------------------ H 7: online gegen offline
def test_h7_cusum_folge_und_alarm_von_hand():
    xo = [Fr(0), Fr(0), Fr(1), Fr(1), Fr(1), Fr(1)]
    cs, alarm = E.cusum_up_trace(xo, Fr(1, 2), Fr(3, 2))
    assert cs == [0, 0, Fr(1, 2), 1, Fr(3, 2), 2] and alarm == 5          # Alarm bei C >= h
    assert next((t for t, c in enumerate(cs, 1) if c > Fr(3, 2)), None) == 6   # mit C > h erst bei t = 6
    assert max(t for t, c in enumerate(cs, 1) if c == 0) == 2 and alarm - 2 == 3
    C, T = ON.cusum_paths(np.array([[0, 0, 1, 1, 1, 1]], float), 0.5, 1.5)
    assert list(C[0]) == [0, 0, 0, 0.5, 1.0, 1.5, 2.0] and T[0] == 5 and ON.last_zero_before(C[0], 5) == 2


def test_h7_offline_auf_derselben_reihe():
    x = [0, 0, 1, 1, 1, 1]
    v, brk, _ = E.dp_exact(x, Fr(1), 1)
    assert brk == [2] and v == 1 and E.gain_exact(x, 0, 6, 2) == Fr(4, 3)       # Gewinn 4/3 > β = 1
    assert E.dp_exact(x, Fr(3, 2), 1)[1] == []
    assert S.dp_batch(np.array(x, float), [1.0], 1).breaks() == [2] and S.dp_batch(np.array(x, float), [1.5], 1).breaks() == []


# ------------------------------------------------------------------------------------------------------------------------------ H 8: Toleranz und F1
def test_h8_toleranz_und_f1():
    assert E.f1_tolerance([98, 150], [100], 5) == (Fr(1, 2), 1, Fr(2, 3))
    assert S.f1_tolerance([98, 150], [100], 5) == pytest.approx((0.5, 1.0, 2 / 3))
    assert S.f1_tolerance([103], [100], 3) == (0.0, 0.0, 0.0)          # Abstand 3 ist nicht näher als 3
    assert S.f1_tolerance([103], [100], 4) == (1.0, 1.0, 1.0)
    p, r, f = S.f1_tolerance([], [100], 5)
    assert p is None and r == 0.0 and f == 0.0                          # leere Vorhersage: F1 = 0 (Festlegung)
    assert S.f1_tolerance([], [], 5) == (None, None, None)              # ohne wahren Bruch nicht definiert
    assert S.f1_tolerance([75], [50, 100], 30)[:2] == (1.0, 1.0) and S.f1_tolerance([75], [50, 100], 20)[1] == 0.0
    for K in (1, 2, 3):
        assert S.f1_tolerance([0], [0] + [10 * (i + 1) for i in range(K)], 3)[2] == pytest.approx(2 / (K + 2))


def test_zerlegungszahl_von_hand_n6_minlen2():
    assert sorted(E.compositions(6, 2)) == sorted([[], [2], [4], [3], [2, 4]])
