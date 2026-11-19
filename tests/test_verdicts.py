"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier. Ein Verdikt ist eine reine Funktion (Art, Text)."""

import sgo_constants as C
import sgo_evaluation as ev

OBJ_PULS = {C.M_OPT: 6.0, C.M_PELT: 6.0, C.M_PELT_V: 6.0, C.M_BIN: 8.0}


# --------------------------------------------------------------------------------------------------------------------------------- Experiment 1
def test_exp1_binaer_blind_nennt_einzelgewinn_beta_und_zielwerte():
    kind, text = ev.verdict_exp1(9, None, [], C.M_BIN, 3.0, 1, OBJ_PULS, 8.0, [3, 6], [(0, 9, 3, 2.0, False)], False)
    assert kind == "warning"
    assert "3 und 6" in text and "Zielwert 6,00 statt 8,00" in text and "findet nichts" in text and "Einzelgewinn einer Teilung ist 2,00" in text and "β = 3,00" in text


def test_exp1_binaer_findet_nichts_und_dp_auch_nichts_ist_kein_blindflug():
    kind, text = ev.verdict_exp1(9, None, [], C.M_BIN, 5.0, 1, {C.M_OPT: 8.0, C.M_PELT: 8.0, C.M_PELT_V: 8.0, C.M_BIN: 8.0}, 8.0, [], [(0, 9, 3, 2.0, False)], False)
    assert "findet nichts" not in text and kind == "info"          # eigene Reihe, ohne wahre Brüche


def test_exp1_pelt_abweichung_nennt_beide_zielwerte_und_die_mindestlaenge():
    obj = {C.M_OPT: 4.0, C.M_PELT: 25 / 6, C.M_PELT_V: 4.0, C.M_BIN: 4.0}
    kind, text = ev.verdict_exp1(5, None, [2], C.M_PELT, 1.0, 2, obj, 4.0, [], [], False)
    assert kind == "warning" and "4,17" in text and "mit einem Bruch (bei 2)" in text and "4,00" in text and "ohne Bruch" in text and "Mindestlänge 2" in text and "K = 0" in text
    kind2, _ = ev.verdict_exp1(5, None, [], C.M_PELT_V, 1.0, 2, obj, 4.0, [], [], False)
    assert kind2 == "info"                                         # verzögert gleich dem Optimum: keine Abweichung gemeldet


def test_exp1_kein_bruch_moeglich_bei_zu_kurzer_reihe():
    kind, text = ev.verdict_exp1(5, None, [], C.M_OPT, 1.0, 3, {C.M_OPT: 7.0}, 7.0, [], [], True)
    assert kind == "warning" and "2 · 3 = 6" in text and "kein Bruch möglich" in text and "7,00" in text


def test_exp1_szenario_richtige_zahl_und_lage():
    obj = {C.M_OPT: 250.0, C.M_PELT: 250.0, C.M_PELT_V: 250.0, C.M_BIN: 250.0}
    kind, text = ev.verdict_exp1(200, [100], [102], C.M_OPT, 10.0, 5, obj, 300.0, [102], [], False)
    assert kind == "success" and "Lagefehler 2" in text and "bei 102" in text
    kind, _ = ev.verdict_exp1(200, [100], [110], C.M_OPT, 10.0, 5, obj, 300.0, [110], [], False)          # Fehler 10 > 5
    assert kind == "info"
    kind, text = ev.verdict_exp1(200, [100], [60, 100], C.M_OPT, 10.0, 5, obj, 300.0, [60, 100], [], False)
    assert kind == "warning" and "Zahl stimmt nicht" in text and "gefunden 2" in text


def test_exp1_nullreihe_fehlbruch_und_richtig():
    obj = {C.M_OPT: 150.0, C.M_PELT: 150.0, C.M_PELT_V: 150.0, C.M_BIN: 150.0}
    kind, text = ev.verdict_exp1(200, [], [37], C.M_OPT, 5.0, 5, obj, 160.0, [37], [], False)
    assert kind == "warning" and "keinen wahren Bruch" in text and "Fehlbruch" in text
    kind, text = ev.verdict_exp1(200, [], [10, 30], C.M_OPT, 5.0, 5, obj, 160.0, [10, 30], [], False)
    assert "alle 2 gefundenen Brüche sind Fehlbrüche" in text
    kind, text = ev.verdict_exp1(200, [], [], C.M_OPT, 10.0, 5, obj, 150.0, [], [], False)
    assert kind == "success" and "ein einzelner Lauf" in text


# --------------------------------------------------------------------------------------------------------------------------------- Experiment 2
def test_exp2_nullreihe_nennt_fehlbruchanteile_bei_c1_und_c2():
    pany = [1.0, 0.6, 0.04, 0.0, 0.0, 0.0, 0.0]
    kind, text = ev.verdict_exp2("null", 0.0, 200, 5, 100, list(C.C_GRID), [0, 0.4, 0.96, 1, 1, 1, 1], pany, [9.0, 1.5, 0.05, 0, 0, 0, 0], [2.0, 3.0, 4.0, 6.0, 10.0])
    assert kind == "warning" and "60,0 %" in text and "4,0 %" in text and "1,50" in text and "R = 100" in text and "Streuung 1 bekannt" in text


def test_exp2_fenster_und_kein_fenster():
    pr = [0.0, 0.35, 0.95, 0.99, 0.99, 0.95, 0.4]
    kind, text = ev.verdict_exp2("ein", 1.0, 200, 5, 1000, list(C.C_GRID), pr, [1] * 7, [1.0] * 7, [2.0, 3.0, 4.0, 6.0])
    assert kind == "success" and "c = 2 bis 6" in text and "mindestens 90 %" in text
    kind, text = ev.verdict_exp2("ein", 1.0, 200, 5, 1000, list(C.C_GRID), pr, [1] * 7, [1.0] * 7, [3.0])
    assert "nur c = 3" in text
    zz = [0.0, 0.36, 0.86, 0.55, 0.2, 0.0, 0.0]
    kind, text = ev.verdict_exp2("zickzack", 1.0, 200, 5, 1000, list(C.C_GRID), zz, [1] * 7, [4.0] * 7, [])
    assert kind == "warning" and "kein c in der Tabelle erreicht 90 %" in text and "c = 2 mit 86,0 %" in text


def test_exp2_robuste_streuung_wird_genannt():
    _, text = ev.verdict_exp2("ein", 2.0, 200, 5, 100, list(C.C_GRID), [0.0, 0.4, 0.95, 0.99, 1, 1, 1], [1] * 7, [1.0] * 7, [2.0, 3.0, 4.0, 6.0, 10.0], True)
    assert "robust geschätzt" in text and "Streuung 1 bekannt" not in text


# --------------------------------------------------------------------------------------------------------------------------------- Experiment 3
def test_exp3_exakt_im_band_und_ausserhalb():
    kind, text = ev.verdict_exp3(3, 2.0, 2.1972, 1, 5000, 0.2448, 0.0061, 0.2470)
    assert kind == "success" and "24,700 %" in text and "innerhalb" in text and "Winkelintegral" in text
    kind, text = ev.verdict_exp3(2, 2.0, 1.3863, 1, 5000, 0.30, 0.0061, 0.239)
    assert kind == "warning" and "außerhalb" in text and "geschlossene Formel" in text
    kind, text = ev.verdict_exp3(200, 2.0, 10.6, 5, 200, 0.04, 0.0139, None)
    assert kind == "info" and "keine geschlossene Form" in text


# --------------------------------------------------------------------------------------------------------------------------------- Experiment 4
def test_exp4_zaehlungen_und_verhaeltnisse_im_text():
    kind, text = ev.verdict_exp4("puls", 2.0, 5, 100, 200, 0, 0, 82, 82, 0, 1000, 380, 407)
    assert kind == "warning" and "keinem der 100 Läufe" in text and "82 von 100" in text and "in 82 mit größerem Zielwert" in text and "0,380" in text and "0,407" in text
    kind, text = ev.verdict_exp4("ein", 2.0, 2, 100, 200, 3, 0, 0, 0, 0, 1000, 400, 400)
    assert "3 von 100" in text and "0 von 100" in text
    kind, text = ev.verdict_exp4("ein", 2.0, 1, 100, 200, 0, 0, 0, 0, 0, 1000, 360, 360)
    assert kind == "success"
    kind, text = ev.verdict_exp4("ein", 2.0, 1, 100, 200, 0, 0, 5, 4, 1, 1000, 360, 360)
    assert kind == "error" and "besser" in text


# --------------------------------------------------------------------------------------------------------------------------------- Experiment 5
def test_exp5_nennt_beide_seiten_mit_ihrem_mass():
    kind, text = ev.verdict_exp5(1.0, 100, 200, 0.18, 0.2302, 7.65, 7.9058, 0.965, 3.91, 2.96, 157, 157)
    assert kind == "info"
    for frag in ("23,02 %", "18,0 %", "7,9058", "7,65 im Mittel", "96,5 %", "3,91", "2,96", "157 Läufe mit Alarm nach ν", "verschiedene Maße", "einseitig"):
        assert frag in text, frag
    _, text = ev.verdict_exp5(2.0, 100, 50, 0.2, 0.2302, None, 3.109, 0.9, None, None, 0, 0)
    assert "keine Läufe mit Alarm nach ν" in text and "ohne Lageschätzung" in text and "nie genau ein Bruch" in text


# --------------------------------------------------------------------------------------------------------------------------------- Format
def test_zahlenformat_und_listen():
    assert ev.de(3.14159, 2) == "3,14" and ev.de_s(-0.5, 1) == "-0,5" and ev.de_s(0.0, 1) == "+0,0" and ev.pct(0.123, 1) == "12,3 %"
    assert ev.lst([]) == "keiner" and ev.lst([3]) == "3" and ev.lst([3, 6]) == "3 und 6" and ev.lst([50, 100, 150]) == "50, 100 und 150"
    assert ev.zahl_brueche(0) == "keinen Bruch" and ev.zahl_brueche(1) == "einen Bruch" and ev.zahl_brueche(3) == "3 Brüche"
    assert ev.se_share(0.0, 100) == 0.005 and abs(ev.se_share(0.5, 100) - 0.05) < 1e-12
    assert ev.c_text(2.0) == "2" and ev.c_text(0.5) == "0,5"
