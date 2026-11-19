"""Je Schnellstart die erwartete Verdikt-Zeile (kein Schnellstart ohne Wirkung meldet eine Wirkung), über die App-Rechenfunktionen mit den Einstellungen des Schnellstarts."""

import pytest

import sgo_constants as C
import sgo_evaluation as EV
import sgo_reference as REF
import sgo_scenario as SC


def _exp1(s: EV.Settings):
    if s.szen1 == C.SZ_EIGEN:
        x, true = SC.parse_series(s.reihe), None
    else:
        X, true = SC.draw_series(C.SCEN_KEY[s.szen1], s.n1, s.delta1, 1, s.seed)
        x = list(X[0])
    beta = C.beta_of(s.c1, len(x)) if s.modus == C.MODUS_C else s.beta1
    r = SC.run_exp1(x, true, beta, s.minlen1, s.methode)
    return r, EV.verdict_exp1(r.n, r.true, r.breaks, r.method, r.beta, r.minlen, r.objectives, r.obj_none, r.dp_breaks, r.bin_steps, r.no_break_possible)


def _exp2(s: EV.Settings):
    key = C.SCEN_KEY[s.szen2]
    d = 0.0 if key == "null" else s.delta2
    r = SC.run_exp2(key, s.n2, d, s.minlen2, s.r2, s.sigma2, s.seed)
    return r, EV.verdict_exp2(key, d, r.n, r.minlen, r.R, list(C.C_GRID), r.p_right, r.p_any, r.mean_count, r.window, s.sigma2 == C.SIGMA_ROBUST)


def test_puls_von_hand_verdikt_ist_der_anker():
    s = EV.Settings(**C.PRESETS["Puls von Hand: binäre Segmentierung blind (β = 3)"])
    r, (kind, text) = _exp1(s)
    assert r.breaks == [] and r.dp_breaks == [3, 6] and kind == "warning"
    assert "findet nichts" in text and "Zielwert 6,00 statt 8,00" in text and "Einzelgewinn einer Teilung ist 2,00" in text


def test_pelt_gegenbeispiel_verdikt_ist_der_anker():
    s = EV.Settings(**C.PRESETS["PELT-Gegenbeispiel (n = 5, Mindestlänge 2)"])
    r, (kind, text) = _exp1(s)
    assert kind == "warning" and r.breaks == [2] and r.dp_breaks == [] and "4,17" in text and "4,00" in text and "Mindestlänge 2" in text
    assert r.enumeration.count == 3


def test_ein_bruch_klar_findet_den_bruch_und_meldet_erfolg():
    r, (kind, text) = _exp1(EV.Settings(**C.PRESETS["Ein Bruch, klar (δ = 3)"]))
    assert kind == "success" and len(r.breaks) == 1 and abs(r.breaks[0] - 100) <= 5 and "Zahl richtig" in text


def test_nullreihe_preset_meldet_nichts_ueber_wahre_brueche_und_kein_erfolg_ohne_befund():
    s = EV.Settings(**C.PRESETS["Nullreihe: Fehlbrüche (c = 1)"])
    r, (kind, text) = _exp1(s)
    assert r.true == []
    assert (kind == "warning" and "Fehlbr" in text) or (kind == "success" and not r.breaks)      # je nach Zufalls-Seed; nie ein Erfolg bei gefundenem Bruch
    r2, (kind2, text2) = _exp2(s)
    assert kind2 == "warning" and "Nullreihe" in text2 and "Fehlbruch" in text2
    i1 = C.C_GRID.index(1.0)
    assert r2.p_any[i1] > 0.3                                            # c = 1: viele Läufe mit Fehlbruch (gefroren 60 %)


def test_zickzack_preset_meldet_kein_fenster_und_der_standard_gleich():
    for name in ("Zickzack δ = 1: kein c passt", "Ein Bruch, klar (δ = 3)"):
        s = EV.Settings(**C.PRESETS[name])
        r, (kind, text) = _exp2(s)
        assert kind == "warning" and "kein c in der Tabelle erreicht 90 %" in text, name
        assert r.window == []


def test_ein_bruch_in_experiment_2_hat_ein_fenster_bei_delta_2():
    r = SC.run_exp2("ein", 200, 2.0, 5, 200, C.SIGMA_BEKANNT, 35)
    kind, text = EV.verdict_exp2("ein", 2.0, 200, 5, 200, list(C.C_GRID), r.p_right, r.p_any, r.mean_count, r.window)
    assert kind == "success" and 2.0 in r.window


def test_online_preset_nennt_exakte_zahlen_und_die_simulation_bleibt_im_band():
    s = EV.Settings(**C.PRESETS["Online gegen offline (δ = 1)"])
    r = SC.run_exp5(s.delta5, s.nu5, s.r5, s.seed, REF.h_cusum())
    kind, text = EV.verdict_exp5(r.delta, r.nu, r.R, r.p_early_sim, r.p_early_exact, r.delay_sim, r.delay_exact, r.off_one, r.off_lage_mean, r.online_lage_mean, r.online_n, r.same_n)
    assert kind == "info" and "23,02 %" in text and "7,9058" in text and r.R == 200
    assert abs(r.p_early_sim - r.p_early_exact) < 4 * (r.p_early_exact * (1 - r.p_early_exact) / r.R) ** 0.5


def test_streuung_unbekannt_preset_ist_robust_und_nicht_besser_als_bekannt_im_mittel_der_gefrorenen_zeile():
    s = EV.Settings(**C.PRESETS["Streuung unbekannt: Zickzack δ = 1"])
    assert s.sigma2 == C.SIGMA_ROBUST
    _, (_, text) = _exp2(s)
    assert "robust geschätzt" in text
    i2 = C.C_GRID.index(2.0)
    bek = REF.fenster_pooled("zickzack", 1.0, "bekannt")["p_right"]["mean"][i2]
    rob = REF.fenster_pooled("zickzack", 1.0, "robust")["p_right"]["mean"][i2]
    assert rob < bek                                                       # gefroren: die robuste Schätzung kostet (R = 3000 je Modus)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jeder_schnellstart_rechnet_alle_experimente_ohne_fehler(name):
    s = EV.Settings(**C.PRESETS[name])
    _exp1(s)
    _exp2(s)
    assert SC.run_exp3(s.n3, s.c3, s.r3, s.seed).R == s.r3
    assert SC.run_exp4(C.SCEN_KEY[s.szen4], s.minlen4, s.c4, s.r4, s.seed).R == s.r4
    assert SC.run_exp5(s.delta5, s.nu5, s.r5, s.seed, REF.h_cusum()).R == s.r5
