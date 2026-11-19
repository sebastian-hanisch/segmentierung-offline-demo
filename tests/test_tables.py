"""Gefrorene Tabellen (`sgo_*.json`): Aufbau, Zufalls-Seed und R je Zelle, Beziehungen, Abgleich mit den Vormessungen der Planung im Band, exakte Teile gleich der Neurechnung,
kleine Neurechnung der Simulationsteile (`--check` der drei Werkzeuge). Bänder statt Gleichheit: Simulation, nie bitgleich über Plattformen."""

import importlib.util
import math
import sys
from pathlib import Path


import sgo_constants as C
import sgo_exact as EX
import sgo_online as ON
import sgo_reference as REF

ROOT = Path(__file__).resolve().parent.parent


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------------------------------------ Fenster
def test_fenster_aufbau_jede_zelle_nennt_seed_und_R_und_drei_zufalls_seeds():
    f = REF.fenster()
    assert f["meta"]["R"] == 1000 and f["meta"]["n"] == 200 and f["meta"]["minlen"] == 5 and f["meta"]["c_grid"] == list(C.C_GRID)
    assert set(f["cells"]) == {"null|0.0"} | {f"{s}|{d:.1f}" for s in ("ein", "treppe", "puls", "zickzack") for d in (1.0, 2.0, 3.0)}
    for key, cell in f["cells"].items():
        assert len(cell["per_seed"]) == 3 and len({ps["seed"] for ps in cell["per_seed"]}) == 3
        for ps in cell["per_seed"]:
            assert ps["R"] == 1000
            for mode in ("bekannt", "robust"):
                for k in ("p_right", "p_any", "mean_count", "p_loc", "loc_error"):
                    assert len(ps[mode][k]) == len(C.C_GRID), (key, k)
                assert all(0.0 <= v <= 1.0 for v in ps[mode]["p_right"] + ps[mode]["p_any"])
        assert cell["wahre_brueche"] == __import__("sgo_scenario").true_breaks(cell["scen"], 200)
    assert f["meta"]["stand"] and f["meta"]["quelle"] == "tools/sgo_make_fenster.py"


def test_fenster_beziehungen():
    """Richtige Zahl und Fehlbruch-Anteil hängen zusammen: bei der Nullreihe ist 'richtig' (0 Brüche) gleich 1 minus 'mindestens ein Bruch'; kleiner c gibt mehr Brüche."""
    for ps in REF.fenster()["cells"]["null|0.0"]["per_seed"]:
        for a, b in zip(ps["bekannt"]["p_right"], ps["bekannt"]["p_any"]):
            assert abs(a - (1 - b)) < 1e-12
    for cell in REF.fenster()["cells"].values():
        for ps in cell["per_seed"]:
            mc = ps["bekannt"]["mean_count"]
            assert all(x >= y - 0.02 for x, y in zip(mc, mc[1:]))      # mittlere Bruchzahl fällt mit c (Rauschen von 0,02)


def test_fenster_stimmt_mit_den_messungen_der_planung_im_band():
    """Plan 07 Befund E (Zufalls-Seeds der Planung, numpy-Zufall): Nullreihe c = 1/2/3 richtige Zahl 40,2/95,8/99,9 und 37,6/95,3/99,9; ein Bruch δ = 1 34,9/94,8/99,6 usw. Hier das Band ±4 Prozentpunkte."""
    i = {c: C.C_GRID.index(c) for c in (1.0, 2.0, 3.0)}
    soll = {("null", 0.0): (40.2, 95.8, 99.9), ("ein", 1.0): (34.9, 94.8, 99.6), ("treppe", 1.0): (37.7, 93.0, 90.9), ("puls", 1.0): (35.3, 90.1, 67.9), ("zickzack", 1.0): (35.7, 89.4, 57.1)}
    for (scen, d), vals in soll.items():
        p = REF.fenster_pooled(scen, d)
        for c, v in zip((1.0, 2.0, 3.0), vals):
            assert abs(100 * p["p_right"]["mean"][i[c]] - v) < 4.0, (scen, d, c, 100 * p["p_right"]["mean"][i[c]], v)


def test_fenster_aussagen_der_demo_gelten_in_den_gefrorenen_zahlen():
    z = REF.fenster_pooled("zickzack", 1.0)
    assert z["window"] == [] and max(z["p_right"]["hi"]) < 0.9            # kein c mit 90 %, auch nicht in der Spanne der drei Zufalls-Seeds
    n = REF.fenster_pooled("null", 0.0)
    assert 0.55 < n["p_any"]["lo"][1] and n["p_any"]["hi"][1] < 0.65 and n["p_any"]["hi"][2] < 0.06
    e = REF.fenster_pooled("ein", 1.0)
    assert e["window"] == [2.0, 3.0, 4.0, 6.0]
    # die untere Kante ist scharf: mittlere Zahl der Brüche in der Nullreihe bei c = 0,5 / 1 / 2 fällt stark
    assert n["mean_count"]["mean"][0] > 8 and 1.3 < n["mean_count"]["mean"][1] < 1.8 and n["mean_count"]["mean"][2] < 0.08


def test_mindestlaenge_und_pulsbreite_tabellen():
    f = REF.fenster()
    ml = f["mindestlaenge"]
    assert list(ml) == ["1", "2", "5", "10", "20"]
    assert all(ml[a][0] >= ml[b][0] - 0.02 for a, b in (("1", "2"), ("2", "5"), ("5", "10"), ("10", "20")))   # Fehlbrüche bei c = 1 fallen mit der Mindestlänge
    pw = f["puls_breite"]
    assert pw["2"]["p_right"] < 0.2 and pw["4"]["p_right"] > 0.8 and pw["5"]["p_right"] > 0.9       # kurzer Puls fällt weg


def test_fenster_check_werkzeug_gibt_keine_abweichung():
    mod = _tool("sgo_make_fenster")
    assert mod.check(ROOT / "sgo_fenster.json", ROOT / "sgo_null_n.json") == 0


# ------------------------------------------------------------------------------------------------------------------------------ Nullreihe
def test_null_n_exakte_teile_gleich_neurechnung_und_simulation_im_band():
    z = REF.null_n()
    for c, row in z["n2"].items():
        assert abs(row["exakt"] - EX.p_break_n2(row["beta"])) < 1e-12
        for key, runs in (("sim_formel", row["runs_formel"]), ("sim_dp", row["runs_dp"])):
            assert abs(row[key] - row["exakt"]) < 4 * math.sqrt(row["exakt"] * (1 - row["exakt"]) / runs), (c, key)
    for c, row in z["n3"].items():
        assert abs(row["exakt_kein_bruch"] - EX.p_nobreak_n3(row["beta"])) < 1e-12
        for key, runs in (("sim_aufzaehlung", row["runs_aufzaehlung"]), ("sim_dp", row["runs_dp"])):
            p = row["exakt_kein_bruch"]
            assert abs(row[key] - p) < 4 * math.sqrt(p * (1 - p) / runs), (c, key)


def test_null_n_reihe_trend_und_feste_stelle():
    r = REF.null_n()["n_reihe"]
    ns = [50, 100, 200, 400, 800]
    assert list(r) == [str(n) for n in ns]
    c1 = [r[str(n)]["1.0"]["p_fehlbruch"] for n in ns]
    c2 = [r[str(n)]["2.0"]["p_fehlbruch"] for n in ns]
    assert c1[-1] > c1[0] + 0.15 and c2[-1] < c2[0] - 0.03                # steigt bei c = 1, fällt bei c = 2 (Zusammenhang über die ganze Spanne, nicht Schritt für Schritt: Rauschen 1,6 Punkte)
    for n in ns:
        row = r[str(n)]
        assert row["n_stellen"] == EX.n_sites(n, 5)
        assert abs(row["1.0"]["feste_stelle"] - EX.p_fixed_site(math.log(n))) < 1e-12
    assert abs(r["200"]["1.0"]["stellen_mal_einzel"] - 4.077) < 5e-3 and r["200"]["1.0"]["stellen_mal_einzel"] > 1.0


# ------------------------------------------------------------------------------------------------------------------------------ Online
def test_online_exakte_teile_gleich_neurechnung():
    ex = REF.online()["exakt"]
    h = ON.solve_h()
    assert abs(ex["h"] - h) < 1e-9 and abs(REF.h_cusum() - h) < 1e-9
    assert abs(ex["arl0_integral"] - ON.ARL0_TARGET) < 1e-6
    assert abs(ex["arl0_kette_400"] - 370.3887) < 5e-4 and abs(ex["arl0_kette_800"] - 370.3959) < 5e-4
    assert abs(ex["arl1"]["1.0"]["integral"] - 8.5751) < 5e-4 and abs(ex["arl1"]["2.0"]["integral"] - 3.4068) < 5e-4
    assert abs(ex["verzoegerung"]["1.0"]["mittel"] - 7.9058) < 5e-4 and abs(ex["verzoegerung"]["2.0"]["mittel"] - 3.1090) < 5e-4
    assert abs(ex["verzoegerung"]["1.0"]["p_alarm_vor_nu"] - 0.2302) < 5e-5


def test_online_simulation_im_band_der_exakten_werte():
    z = REF.online()
    for d, s in z["simulation"].items():
        ex = z["exakt"]["verzoegerung"][d]
        assert s["R"] == 4000 and s["nu"] == 100
        assert abs(s["p_alarm_vor_nu"] - ex["p_alarm_vor_nu"]) < 4 * math.sqrt(ex["p_alarm_vor_nu"] * (1 - ex["p_alarm_vor_nu"]) / s["R"])
        assert abs(s["verzoegerung_mittel"] - ex["mittel"]) < 4 * s["verzoegerung_se"]
        assert s["gleiche_laeufe_n"] == s["online_n"] and s["gleiche_laeufe_n"] < s["R"]      # gleiche Läufe = Läufe ohne früheren Alarm


def test_online_phase1_tabelle():
    p = REF.online()["phase1"]
    assert list(p) == ["0.0", "0.5", "1.0", "1.5", "2.0", "3.0"] and all(v["N"] == 100 and v["R"] == 2000 for v in p.values())
    assert 0.03 < p["0.0"]["mitte"] < 0.08 and 0.03 < p["0.0"]["rand"] < 0.08       # δ = 0: nur Fehlbrüche (Plan 4,8 und 5,3 %)
    assert p["1.0"]["mitte"] > 0.97 and 0.55 < p["1.0"]["rand"] < 0.72 and p["3.0"]["rand"] > 0.99
    assert all(p[a]["mitte"] <= p[b]["mitte"] + 0.01 for a, b in zip(list(p), list(p)[1:]))


def test_online_check_werkzeug_gibt_keine_abweichung():
    mod = _tool("sgo_make_online")
    assert mod.check(ROOT / "sgo_online.json") == 0


# ------------------------------------------------------------------------------------------------------------------------------ Verfahren
def test_verfahren_exakte_zaehlungen_n5():
    v = REF.verfahren()["n5_exakt"]
    assert v["1"] == dict(reihen=1024, pelt_abw_beta1=0, pelt_abw_beta2=0, verzoegert_abw=0, instanzen=2048)
    assert v["2"] == dict(reihen=1024, pelt_abw_beta1=40, pelt_abw_beta2=14, verzoegert_abw=0, instanzen=2048)


def test_verfahren_zufallsreihen_und_binaer_exakt():
    z = REF.verfahren()
    assert z["zufall_exakt"]["1"]["pelt_abw"] == 0 and all(z["zufall_exakt"][m]["verzoegert_abw"] == 0 for m in z["zufall_exakt"])
    assert all(z["zufall_exakt"][m]["pelt_abw"] > 0 for m in ("2", "3", "5")) and all(z["zufall_exakt"][m]["instanzen"] == 1500 for m in z["zufall_exakt"])
    b = z["binseg_exakt"]
    assert b["instanzen"] == 3000 and b["zielwert_kleiner"] == 0 and 0 < b["zielwert_groesser"] <= b["andere_bruchmenge"]


def test_verfahren_n200():
    z = REF.verfahren()["pelt_n200"]
    assert z["1"]["pelt_abw"] == 0 and z["1"]["instanzen"] == 1040 and z["5"]["instanzen"] == 8320 and z["2"]["instanzen"] == 2080
    assert all(z[m]["verzoegert_abw"] == 0 for m in z)
    for m in z:
        assert z[m]["verhaeltnis_verzoegert"] >= z[m]["verhaeltnis_pelt"] - 1e-12 and 0.3 < z[m]["verhaeltnis_pelt"] < 0.45
    b = REF.verfahren()["binseg_n200"]
    assert b["gesamt"]["besser"] == 0 and b["gesamt"]["laeufe_mal_strafen"] == 15600 and 0.35 < b["gesamt"]["anteil_andere"] < 0.43
    assert len(b["cells"]) == 13 and all(v[c]["besser"] == 0 for v in b["cells"].values() for c in ("1.0", "2.0", "4.0"))
    anteile = [v["2.0"]["andere"] for v in b["cells"].values()]
    assert min(anteile) < 0.02 and max(anteile) > 0.7        # Spanne je Zelle weit


def test_verfahren_check_werkzeug_gibt_keine_abweichung():
    mod = _tool("sgo_make_verfahren")
    assert mod.check(ROOT / "sgo_verfahren.json") == 0


def test_alle_tabellen_tragen_stand_und_quelle():
    for name, fn in (("fenster", REF.fenster), ("null_n", REF.null_n), ("online", REF.online), ("verfahren", REF.verfahren)):
        meta = fn()["meta"]
        assert meta["stand"] and meta["quelle"].startswith("tools/sgo_make_"), name
