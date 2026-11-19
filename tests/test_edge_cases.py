"""Randfälle (DEMO-PLAYBOOK §12): leere und zu kurze Reihen, n = 1, n = 2, Mindestlänge über n, NaN und Unendlich, Dezimalkomma, Aufzählungsgrenze n = 14, Gleichstände,
identische Werte, Regler am Maximum, R = 1."""

import math
import time

import numpy as np
import pytest

import sgo_constants as C
import sgo_scenario as SC
import sgo_segmentation as S


# ------------------------------------------------------------------------------------------------------------------------------ Eingabe der eigenen Reihe
def test_leere_eingabe_gibt_meldung():
    for text in ("", "   ", "\n\n", None):
        with pytest.raises(SC.SeriesError, match="leer"):
            SC.parse_series(text)


def test_dezimalkomma_semikolon_und_zeilenumbruch():
    assert SC.parse_series("0,5 1,25;2\n3") == [0.5, 1.25, 2.0, 3.0]
    assert SC.parse_series("  1   2  ") == [1.0, 2.0]


def test_nicht_numerisch_nan_und_unendlich_geben_meldung_statt_zahl():
    for text, frag in (("1 2 abc", "keine Zahl"), ("1 nan 3", "endlich"), ("1 inf", "endlich"), ("1 -inf", "endlich"), ("0,0,0,2", "keine Zahl")):
        with pytest.raises(SC.SeriesError, match=frag):
            SC.parse_series(text)


def test_zu_lange_reihe_gibt_meldung():
    with pytest.raises(SC.SeriesError, match="höchstens 400"):
        SC.parse_series(" ".join(["1"] * 401))
    assert len(SC.parse_series(" ".join(["1"] * 400))) == 400


# ------------------------------------------------------------------------------------------------------------------------------ Größen
def test_n1_und_n2():
    r1 = SC.run_exp1([3.0], None, 0.0, 1, C.M_OPT)
    assert r1.breaks == [] and r1.no_break_possible and r1.enumeration.count == 1
    r2 = SC.run_exp1([0.0, 5.0], None, 1.0, 1, C.M_OPT)                  # n = 2: Gewinn 12,5 > β = 1
    assert r2.breaks == [1] and not r2.no_break_possible
    r2b = SC.run_exp1([0.0, 1.0], None, 1.0, 1, C.M_OPT)                 # Gewinn 0,5 <= 1: kein Bruch
    assert r2b.breaks == []
    assert S.dp_batch(np.array([0.0, 5.0]), [1.0], 1).breaks() == [1]


def test_mindestlaenge_ueber_n_gibt_meldung_und_zu_kurz_fuer_einen_bruch_ist_kein_fehler():
    with pytest.raises(SC.SeriesError, match="Mindestlänge"):
        SC.run_exp1([1.0, 2.0, 3.0], None, 1.0, 4, C.M_OPT)
    with pytest.raises(ValueError, match="Mindestlänge"):
        S.dp_batch(np.array([1.0, 2.0]), [1.0], 3)
    r = SC.run_exp1([1.0, 2.0, 3.0, 4.0, 9.0], None, 1.0, 3, C.M_OPT)    # n = 5 < 2·3: kein Bruch möglich
    assert r.no_break_possible and r.breaks == []
    assert r.pelt.breaks == [] and r.bin_breaks == []


def test_aufzaehlungsgrenze_n14_und_ab_n15_keine_aufzaehlung():
    x14 = [0.0] * 7 + [4.0] * 7
    r = SC.run_exp1(x14, None, 2.0, 1, C.M_OPT)
    assert r.enumeration is not None and r.enumeration.count == 2 ** 13 and r.breaks == [7]
    r15 = SC.run_exp1([0.0] * 7 + [4.0] * 8, None, 2.0, 1, C.M_OPT)
    assert r15.enumeration is None
    with pytest.raises(ValueError, match="Aufzählung nur bis"):
        S.enumerate_best(np.zeros(15), 1.0, 1)


def test_identische_werte_geben_keinen_bruch():
    r = SC.run_exp1([4.0] * 20, None, 0.5, 1, C.M_OPT)
    assert r.breaks == [] and r.objective == pytest.approx(0.0, abs=1e-12) and r.pelt.breaks == [] and r.bin_breaks == []


def test_gleichstand_in_der_aufzaehlung_wird_gemeldet_und_nicht_als_fehler():
    r = SC.run_exp1([0.0, 0.0, 0.0, 0.0], None, 0.0, 2, C.M_OPT)       # [] und [2] haben beide den Zielwert 0
    assert r.enumeration.count == 2 and len(r.enumeration.optimal) == 2 and r.breaks == []


def test_grosse_verschiebung_in_der_app_rechnung():
    x = [1e9 + v for v in (0, 0, 0, 2, 2, 2)]
    assert SC.run_exp1(x, None, 2.0, 1, C.M_OPT).breaks == [3]


def test_strafe_bei_n1_ist_null_und_nie_negativ():
    assert C.beta_of(2.0, 1) == 0.0 and C.beta_of(2.0, 2) == pytest.approx(2 * math.log(2)) and C.beta_of(0.5, 400) > 0


# ------------------------------------------------------------------------------------------------------------------------------ Obergrenzen und Läufe
def test_exp3_obergrenze_n_mal_R():
    with pytest.raises(SC.LiveLimitError, match="Obergrenze"):
        SC.run_exp3(800, 2.0, 100, 1)                                  # 80 000 > 40 000
    r = SC.run_exp3(800, 2.0, 50, 1)                                   # 40 000 genau: erlaubt (inklusive)
    assert r.R == 50 and r.minlen == 5


def test_R_gleich_1_in_allen_experimenten():
    assert len(SC.run_exp2("ein", 100, 2.0, 5, 1, C.SIGMA_BEKANNT, 1).p_right) == len(C.C_GRID)
    assert SC.run_exp3(50, 1.0, 1, 1).R == 1
    assert SC.run_exp4("ein", 5, 2.0, 1, 1).R == 1
    assert SC.run_exp5(1.0, 100, 1, 1, 4.0965).R == 1


def test_exp2_und_exp4_und_exp5_regler_ueber_dem_maximum_geben_meldung():
    with pytest.raises(SC.LiveLimitError):
        SC.run_exp2("ein", 200, 1.0, 5, C.R2_MAX + 1, C.SIGMA_BEKANNT, 1)
    with pytest.raises(SC.LiveLimitError):
        SC.run_exp4("ein", 5, 2.0, C.R4_MAX + 1, 1)
    with pytest.raises(SC.LiveLimitError):
        SC.run_exp5(1.0, 100, C.R5_MAX + 1, 1, 4.0965)


def test_worst_case_am_regler_maximum_mit_zeitgrenze():
    """Zeitgrenze statt Zeitmessung (Plattform): n = 400 mit allen vier Verfahren, R = 200 bei n = 200 mit sieben Strafen, Experiment 4 mit R = 100."""
    t0 = time.perf_counter()
    X, true = SC.draw_series("zickzack", 400, 1.0, 1, 5)
    SC.run_exp1(list(X[0]), true, C.beta_of(2.0, 400), 1, C.M_OPT)
    SC.run_exp2("zickzack", 200, 1.0, 1, C.R2_MAX, C.SIGMA_ROBUST, 5)
    SC.run_exp4("puls", 1, 1.0, C.R4_MAX, 5)
    SC.run_exp5(1.0, 100, C.R5_MAX, 5, 4.0965)
    assert time.perf_counter() - t0 < 60


# ------------------------------------------------------------------------------------------------------------------------------ Szenarien
@pytest.mark.parametrize("scen,n,soll", [("ein", 200, [100]), ("treppe", 200, [70, 140]), ("puls", 200, [80, 120]), ("zickzack", 200, [50, 100, 150]), ("null", 200, [])])
def test_wahre_brueche_bei_n200_wie_in_der_vormessung(scen, n, soll):
    assert SC.true_breaks(scen, n) == soll


def test_mittelwertprofile():
    n = 200
    assert list(np.unique(SC.mean_profile("treppe", n, 1.0))) == [0.0, 1.0, 2.0]
    assert SC.mean_profile("puls", n, 2.0)[79] == 0 and SC.mean_profile("puls", n, 2.0)[80] == 2.0 and SC.mean_profile("puls", n, 2.0)[119] == 2.0 and SC.mean_profile("puls", n, 2.0)[120] == 0
    z = SC.mean_profile("zickzack", n, 1.0)
    assert [z[49], z[50], z[99], z[100], z[149], z[150]] == [0, 1, 1, 0, 0, 1]
    assert not SC.mean_profile("null", n, 5.0).any() and SC.mean_profile("ein", n, 3.0)[99] == 0 and SC.mean_profile("ein", n, 3.0)[100] == 3.0


def test_erste_zeilen_bleiben_gleich_wenn_R_waechst():
    a, _ = SC.draw_series("ein", 50, 1.0, 5, 9)
    b, _ = SC.draw_series("ein", 50, 1.0, 10, 9)
    assert np.array_equal(a, b[:5])


def test_gleicher_zufalls_seed_gleiche_ausgabe_anderer_seed_andere():
    a = SC.run_exp2("ein", 100, 2.0, 5, 50, C.SIGMA_BEKANNT, 3)
    assert a.p_right == SC.run_exp2("ein", 100, 2.0, 5, 50, C.SIGMA_BEKANNT, 3).p_right
    assert a.p_right != SC.run_exp2("ein", 100, 2.0, 5, 50, C.SIGMA_BEKANNT, 4).p_right


def test_nullreihe_hat_kein_f1_und_keine_lagefehler():
    r = SC.run_exp2("null", 100, 0.0, 5, 50, C.SIGMA_BEKANNT, 1)
    assert r.true == [] and all(v is None for v in r.f1.values()) and all(v is None for v in r.f1_zero.values())
    assert r.p_right[C.C_GRID.index(10.0)] == 1.0


def test_exp5_nu_ausserhalb_gibt_fehler():
    with pytest.raises(ValueError):
        SC.run_exp5(1.0, 0, 10, 1, 4.0965)
    with pytest.raises(ValueError):
        SC.run_exp5(1.0, 200, 10, 1, 4.0965)


def test_pelt_bei_beta_null_streicht_den_besten_kandidaten_nicht_regressionstest():
    """Fund des Randfall-Tests beim Bau: bei β = 0 strich die Regel F(s) + C(s, t) >= F(t) den besten Kandidaten selbst; bei Mindestlänge 2 war die Kandidatenliste leer
    (argmin über eine leere Folge). Jetzt gilt bei β <= 0 die strenge Ungleichung."""
    x = np.zeros(4)
    for delayed in (False, True):
        r = S.pelt(x, 0.0, 2, delayed=delayed)
        assert r.objective == pytest.approx(0.0, abs=1e-12)
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    for m in (1, 2, 3):
        dp = S.dp_batch(x, [0.0], m).F[0, 0]
        assert S.pelt(x, 0.0, m).objective >= dp - 1e-9 and S.pelt(x, 0.0, m, delayed=True).objective == pytest.approx(dp, abs=1e-9)
