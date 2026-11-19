"""Jede Zahl der README-Tabelle „Befunde“ (und der Tabelle „Befunde und Korrekturen“) wird aus den gefrorenen Tabellen, den exakten Rechnungen oder einer Neurechnung erzeugt und
**wörtlich in der README gesucht**. Die Zahl-zu-Test-Tabelle der README nennt für jede Zahl die Funktion hier. Neu gerechnet wird nur, wo der Test schnell ist; Simulationszahlen stammen
aus den gefrorenen Tabellen (Bänder sind in `test_tables.py` und `test_oracle_*.py` geprüft)."""

import math
import re
from pathlib import Path


import sgo_constants as C
import sgo_exact as EX
import sgo_online as ON
import sgo_reference as REF
import sgo_segmentation as S
from sgo_evaluation import de

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
CG = list(C.C_GRID)


def has(*parts):
    for p in parts:
        assert p in README, f"nicht in der README: {p!r}"


def j(vals, nd=1):
    """Werte (Anteile 0 bis 1) als „a / b / c“ in Prozent mit Dezimalkomma, ohne Prozentzeichen."""
    return " / ".join(de(100 * v, nd) for v in vals)


def at(p, key, cs):
    return [p[key]["mean"][CG.index(c)] for c in cs]


def test_fenster_nullreihe():
    p = REF.fenster_pooled("null", 0.0)
    i1, i2, i3 = CG.index(1.0), CG.index(2.0), CG.index(3.0)
    has(f"{de(100 * p['p_any']['mean'][i1])} / {de(100 * p['p_any']['mean'][i2])} / {de(100 * p['p_any']['mean'][i3])} %")
    has(f"{de(100 * p['p_any']['mean'][i1])} % (Spanne {de(100 * p['p_any']['lo'][i1])} bis {de(100 * p['p_any']['hi'][i1])}) / {de(100 * p['p_any']['mean'][i2])} % (2,7 bis 4,0)")
    assert de(100 * p["p_any"]["lo"][i2]) == "2,7" and de(100 * p["p_any"]["hi"][i2]) == "4,0"
    has("9,01 / 1,54 / 0,04")
    assert [de(v, 2) for v in p["mean_count"]["mean"][:3]] == ["9,01", "1,54", "0,04"]
    assert 55 <= 100 * p["p_any"]["lo"][i1] and 100 * p["p_any"]["hi"][i1] <= 65


def test_nullreihe_gegen_n():
    r = REF.null_n()["n_reihe"]
    ns = ("50", "100", "200", "400", "800")
    has(" / ".join(de(100 * r[n]["1.0"]["p_fehlbruch"]) for n in ns) + " % (steigt)")
    has(" / ".join(de(100 * r[n]["2.0"]["p_fehlbruch"]) for n in ns) + " % (fällt)")
    has(" / ".join(de(100 * r[n]["1.0"]["p_fehlbruch"]) for n in ns) + " %", " / ".join(de(100 * r[n]["2.0"]["p_fehlbruch"]) for n in ns) + " %")
    c1 = [r[n]["1.0"]["p_fehlbruch"] for n in ns]
    c2 = [r[n]["2.0"]["p_fehlbruch"] for n in ns]
    assert c1[-1] > c1[0] and c2[-1] < c2[0]
    has(f"{de(100 * EX.p_fixed_site(math.log(200)), 2)} % / {de(100 * EX.p_fixed_site(2 * math.log(200)), 3)} %")
    has(f"191 Stellen: {de(100 * EX.sites_times_single(200, 1.0, 5), 0)} % / {de(100 * EX.sites_times_single(200, 2.0, 5), 1)} %")


def test_fenster_ein_bruch():
    p = REF.fenster_pooled("ein", 1.0)
    has(j(at(p, "p_right", (1.0, 2.0, 3.0, 4.0, 6.0, 10.0))) + " %; Fenster (≥ 90 %): c = 2 bis 6")
    assert p["window"] == [2.0, 3.0, 4.0, 6.0]


def test_fenster_treppe_puls_zickzack():
    t = REF.fenster_pooled("treppe", 1.0)
    has(j(at(t, "p_right", (2.0, 3.0, 4.0))) + " %; Fenster c = 2 bis 3")
    assert t["window"] == [2.0, 3.0]
    p = REF.fenster_pooled("puls", 1.0)
    i2 = CG.index(2.0)
    has(f"{de(100 * p['p_right']['mean'][i2])} % (Spanne der drei Zufalls-Seeds {de(100 * p['p_right']['lo'][i2])} bis {de(100 * p['p_right']['hi'][i2])}: **an der Schwelle**) / "
        f"{j(at(p, 'p_right', (3.0, 4.0)))} %")
    assert p["p_right"]["lo"][i2] < 0.9 < p["p_right"]["hi"][i2]        # an der Schwelle: ein Zufalls-Seed darunter, die anderen darüber
    seeds_unter = sum(ps["bekannt"]["p_right"][i2] < 0.9 for ps in REF.fenster_cell("puls", 1.0)["per_seed"])
    assert seeds_unter == 1
    has("ein Zufalls-Seed darunter, zwei darüber")
    z = REF.fenster_pooled("zickzack", 1.0)
    has(f"{de(100 * z['p_right']['mean'][i2])} % ({de(100 * z['p_right']['lo'][i2])} bis {de(100 * z['p_right']['hi'][i2])}) / {j(at(z, 'p_right', (3.0, 4.0, 6.0)))} %; **kein c mit 90 %**")
    assert z["window"] == [] and max(z["p_right"]["hi"]) < 0.9
    assert de(100 * z["p_right"]["mean"][CG.index(4.0)]) == "20,1"
    has("20,1 %", "91,8 %")
    assert de(100 * t["p_right"]["mean"][CG.index(3.0)]) == "91,8"


def test_fenster_delta_2_3():
    erwartet = {("ein", 2.0): None, ("ein", 3.0): None, ("treppe", 2.0): None, ("treppe", 3.0): None, ("puls", 3.0): None, ("zickzack", 3.0): None}
    for (scen, d) in erwartet:
        assert REF.fenster_pooled(scen, d)["window"] == [2.0, 3.0, 4.0, 6.0, 10.0], (scen, d)
    for scen in ("puls", "zickzack"):
        assert REF.fenster_pooled(scen, 2.0)["window"] == [2.0, 3.0, 4.0, 6.0], scen
    has("ein Bruch, Treppe δ = 2, 3 und Puls, Zickzack δ = 3: c = 2 bis 10; Puls und Zickzack δ = 2: c = 2 bis 6")
    # zwölf von dreizehn Zellen mit Fenster (Puls δ = 1 auf der Schwelle: der Mittelwert der drei Zufalls-Seeds liegt über 90 %)
    zellen = [("null", 0.0)] + [(s, d) for s in ("ein", "treppe", "puls", "zickzack") for d in (1.0, 2.0, 3.0)]
    assert sum(bool(REF.fenster_pooled(s, d)["window"]) for s, d in zellen) == 12
    has("in **elf bis zwölf von dreizehn** Zellen")
    zz3 = REF.fenster_pooled("zickzack", 3.0)["p_right"]["mean"]
    assert min(zz3[CG.index(3.0):]) >= 0.997 - 1e-9
    has("bei δ = 3 in diesem Raster mindestens 99,7 % für c = 3 bis 10")


def test_lagefehler():
    vals = [REF.fenster_pooled("ein", d)["loc_error"][CG.index(2.0)] for d in (1.0, 2.0, 3.0)]
    has(" / ".join(de(v, 2) for v in vals))
    assert [de(v, 2) for v in vals] == ["3,27", "0,59", "0,15"]


def test_toleranz_f1():
    e = REF.fenster_pooled("ein", 1.0)["f1"]
    has(" / ".join(de(e[str(m)], 3) for m in (1, 2, 5, 10, 20)))
    z = REF.fenster_pooled("zickzack", 1.0)["f1"]
    has(" / ".join(de(z[str(m)], 3) for m in (1, 2, 5, 10, 20)) + " (zulässig nur M < 25")
    cell = REF.fenster_cell("zickzack", 1.0)
    genau = [ps["bekannt"]["p_genau_wahre_menge"] for ps in cell["per_seed"]]
    mc = [ps["bekannt"]["mean_count"][CG.index(2.0)] for ps in cell["per_seed"]]
    has(f"{de(100 * sum(genau) / 3)} % ({de(100 * min(genau))} bis {de(100 * max(genau))}); {de(sum(mc) / 3, 2)} statt 3")
    # dasselbe Ergebnis liegt je nach M zwischen 0,25 und 0,97 (Zählung über die Toleranzen M = 1 … 20)
    assert min(e.values()) < 0.3 and e["20"] > 0.96


def test_streuung_unbekannt():
    out = []
    for scen, d in (("zickzack", 1.0), ("puls", 1.0), ("ein", 1.0)):
        b = REF.fenster_pooled(scen, d, "bekannt")["p_right"]["mean"][CG.index(2.0)]
        r = REF.fenster_pooled(scen, d, "robust")["p_right"]["mean"][CG.index(2.0)]
        out.append((r, b))
    has(" / ".join(f"{de(100 * r)} gegen {de(100 * b)}" for r, b in out) + " %")
    has("81,2 gegen 86,0 %")
    zellen = [("null", 0.0)] + [(s, d) for s in ("ein", "treppe", "puls", "zickzack") for d in (1.0, 2.0, 3.0)]
    diffs = [100 * (REF.fenster_pooled(s, d, "bekannt")["p_right"]["mean"][CG.index(2.0)] - REF.fenster_pooled(s, d, "robust")["p_right"]["mean"][CG.index(2.0)]) for s, d in zellen]
    assert de(min(diffs)) == "0,8" and de(max(diffs)) == "4,7" and all(v > 0 for v in diffs)
    has("0,8 bis 4,7 Prozentpunkte")


def test_mindestlaenge():
    ml = REF.fenster()["mindestlaenge"]
    s = " · ".join(f"{de(100 * ml[m][0])} / {de(100 * ml[m][1])} / {de(100 * ml[m][2])}" for m in ("1", "2", "5", "10", "20"))
    has(s)
    pw = REF.fenster()["puls_breite"]
    has(" / ".join(de(100 * pw[w]["p_right"]) for w in ("2", "4", "5", "8", "20")) + " %")


def test_pelt_n200():
    z = REF.verfahren()["pelt_n200"]
    has(f"{z['1']['pelt_abw']} von {z['1']['instanzen']:d} / {z['5']['pelt_abw']} von {z['5']['instanzen']:d} / {z['2']['pelt_abw']} von {z['2']['instanzen']:d}")
    assert all(z[m]["verzoegert_abw"] == 0 for m in z)
    has(f"PELT {de(z['1']['verhaeltnis_pelt'], 3)} / {de(z['5']['verhaeltnis_pelt'], 3)}; verzögert (5) {de(z['5']['verhaeltnis_verzoegert'], 3)}")
    has("0 von 1040 Instanzen (n = 200)")


def test_pelt_zufallsreihen_exakt():
    z = REF.verfahren()["zufall_exakt"]
    has(" / ".join(str(z[m]["pelt_abw"]) for m in ("1", "2", "3", "5")) + "; verzögert 0 in allen")
    assert all(z[m]["verzoegert_abw"] == 0 and z[m]["instanzen"] == 1500 for m in z)
    has("0 von 1500 (n ≤ 16)")


def test_binaere_segmentierung():
    b = REF.verfahren()["binseg_exakt"]
    has(f"andere Bruchmenge in {b['andere_bruchmenge']}, Zielwert größer in {b['zielwert_groesser']}, kleiner in {b['zielwert_kleiner']}")
    g = REF.verfahren()["binseg_n200"]
    has(f"andere Bruchmenge in {g['gesamt']['andere']} von 15 600 ({de(100 * g['gesamt']['anteil_andere'])} %); je c {' / '.join(de(100 * g['gesamt']['je_c'][c]) for c in ('1.0', '2.0', '4.0'))} %")
    anteile = [v["2.0"]["andere"] for v in g["cells"].values()]
    has(f"je Zelle bei c = 2 von {de(100 * min(anteile))} bis {de(100 * max(anteile))} %")
    ein = [g["cells"][f"ein|{d}"]["2.0"]["andere"] for d in ("1.0", "2.0", "3.0")]
    has(f"{' / '.join(de(100 * v) for v in ein)} % für δ = 1 / 2 / 3")
    assert g["gesamt"]["besser"] == 0 and g["gesamt"]["laeufe_mal_strafen"] == 15600


def test_online_exakt():
    ex = REF.online()["exakt"]
    has(f"h = {de(ex['h'], 5)}; {de(ex['arl0_kette_400'], 4)} / {de(ex['arl0_kette_800'], 4)}")
    has("8,5751 / 3,4068 auf beiden Wegen")
    assert (de(ex["arl1"]["1.0"]["integral"], 4), de(ex["arl1"]["2.0"]["integral"], 4)) == ("8,5751", "3,4068")
    v = ex["verzoegerung"]
    has(f"{de(100 * v['1.0']['p_alarm_vor_nu'], 2)} %; {de(v['1.0']['mittel'], 4)} / {de(v['2.0']['mittel'], 4)} Beobachtungen")
    h = ON.solve_h()
    assert abs(ex["h"] - h) < 1e-9


def test_online_simulation():
    s1, s2 = REF.online()["simulation"]["1.0"], REF.online()["simulation"]["2.0"]
    has(f"{de(100 * s1['offline_genau_ein_bruch'])} / {de(100 * s2['offline_genau_ein_bruch'])} %; {de(s1['offline_lage_betrag'], 2)} / {de(s2['offline_lage_betrag'], 2)}")
    has(f"({s1['gleiche_laeufe_n']} von {s1['R']}, kein Alarm vor ν), δ = 1: Lagefehler online / offline | {de(s1['gleiche_online_lage_betrag'], 2)} / {de(s1['gleiche_offline_lage_betrag'], 2)}")
    has(f"({s2['gleiche_laeufe_n']} von {s2['R']}, kein Alarm vor ν), δ = 2: Lagefehler online / offline | {de(s2['gleiche_online_lage_betrag'], 2)} / {de(s2['gleiche_offline_lage_betrag'], 2)}")
    assert s2['gleiche_offline_lage_betrag'] < s2['gleiche_online_lage_betrag']       # bei δ = 2 ist der Rückblick bei der Lage genauer
    has(f"{de(100 * s1['p_alarm_vor_nu'])} % / {de(s1['verzoegerung_mittel'], 3)} ± {de(s1['verzoegerung_se'], 3)}")


def test_phase1():
    assert de(1 / 2, 1) == "0,5" and de(math.sqrt(1 + 1.0 ** 2 / 4), 3) == "1,118"
    has("0,5 und 1,118 (= √(1 + δ²/4))")
    p = REF.online()["phase1"]
    has(" / ".join(de(100 * p[d]["mitte"]) for d in ("0.0", "0.5", "1.0")) + " %")
    has(f"{' / '.join(de(100 * p[d]['mitte'], 1) if p[d]['mitte'] < 1 else '100' for d in ('0.0', '0.5', '1.0', '1.5', '2.0'))} %; am Rand (bei 10): "
        f"{' / '.join(de(100 * p[d]['rand'], 1) for d in ('0.0', '0.5', '1.0', '1.5', '2.0'))} %")


def test_ziel_und_zerlegungen():
    assert abs(ON.ARL0_TARGET - 1 / (2 * 0.001349898031630095)) < 1e-6
    has("370,398 = 1/(2 Φ(−3))")
    assert S.count_decompositions(6, 2) == 5
    has("exakt; 5")


def test_online_tabelle_in_der_korrekturen_stimmt():
    s = REF.online()["simulation"]["1.0"]
    has(f"{de(100 * s['p_alarm_vor_nu'])} % / {de(s['verzoegerung_mittel'], 3)} ± {de(s['verzoegerung_se'], 3)} / {de(s['online_lage_betrag'], 2)} / {de(s['offline_lage_betrag'], 2)}")
    has(f"({s['gleiche_laeufe_n']} von 4000")


# ------------------------------------------------------------------------------------------------------------------------------ Texte
def test_testzahl_und_wortliste():
    verboten = ("Saatwert", "Schwanz", "trägt", "tragen", "TODO", "geplant", "noch nicht gebaut")
    for w in verboten:
        assert w not in README, w
    assert "Zufalls-Seed" in README


def test_jede_zahl_der_befundtabelle_nennt_eine_testfunktion_die_es_gibt():
    tabelle = README.split("## Befunde (gemessen, keine Behauptungen)")[1].split("**Gelesen als Zählung:**")[0]
    namen = set(re.findall(r"`(test_[a-z0-9_]+\.py)(?:::([a-z0-9_*]+))?`", tabelle))
    assert len(namen) > 20
    root = Path(__file__).resolve().parent
    for datei, funktion in namen:
        text = (root / datei).read_text(encoding="utf-8")
        if funktion and not funktion.endswith("*"):
            assert f"def {funktion}(" in text, (datei, funktion)
        elif funktion:
            assert f"def {funktion[:-1]}" in text, (datei, funktion)


def test_korrekturen_tabelle_hier_gemessen_spalte():
    """Die Spalte „hier gemessen“ der Tabelle „Befunde und Korrekturen“ (die Plan-Spalte ist aus PLAN-07 übernommen, nicht neu gemessen)."""
    v = REF.verfahren()
    pw = REF.fenster()["puls_breite"]
    has(" / ".join(de(100 * pw[w]["p_right"]) for w in ("2", "4", "5")) + " %")
    z = v["pelt_n200"]["5"]
    assert z["pelt_abw"] == 2 and z["instanzen"] == 8320
    has("| 5 von 8 320 | 2 von 8 320 |")
    zf = v["zufall_exakt"]
    has(" / ".join(str(zf[m]["pelt_abw"]) for m in ("2", "3", "5")))
    b = v["binseg_exakt"]
    has(f"{b['andere_bruchmenge']} (größer in {b['zielwert_groesser']}, kleiner in {b['zielwert_kleiner']})")
    g = v["binseg_n200"]["gesamt"]
    has(f"{de(100 * g['anteil_andere'])} % ({' / '.join(de(100 * g['je_c'][c]) for c in ('1.0', '2.0', '4.0'))} %)")
    p = v["pelt_n200"]
    has(f"{de(p['1']['verhaeltnis_pelt'], 3)} / {de(p['5']['verhaeltnis_pelt'], 3)}, {de(p['5']['verhaeltnis_verzoegert'], 3)}")
    f = REF.fenster_pooled("null", 0.0)
    has(f"{de(100 * f['p_any']['mean'][CG.index(1.0)])} % (Spanne 59,6 bis 59,9) / {de(100 * f['p_any']['mean'][CG.index(2.0)])} % (2,7 bis 4,0)")
    ml = REF.fenster()["mindestlaenge"]
    has(" / ".join(de(100 * ml[m][0]) for m in ("1", "2", "5", "10", "20")) + " %")
    on = REF.online()
    has("3093 von 4000")
    assert on["simulation"]["1.0"]["gleiche_laeufe_n"] == 3093


def test_zaehlung_verzoegertes_streichen():
    v = REF.verfahren()
    exakt = v["n5_exakt"]["2"]["instanzen"] + sum(v["zufall_exakt"][m]["instanzen"] for m in ("2", "3", "5"))
    gleit = v["pelt_n200"]["5"]["instanzen"] + v["pelt_n200"]["2"]["instanzen"]
    assert (exakt, gleit) == (6548, 10400)
    assert v["n5_exakt"]["2"]["verzoegert_abw"] == 0 and all(v["zufall_exakt"][m]["verzoegert_abw"] == 0 for m in ("2", "3", "5")) and v["pelt_n200"]["5"]["verzoegert_abw"] == 0 and v["pelt_n200"]["2"]["verzoegert_abw"] == 0
    has("0 von 6548 exakten Instanzen mit Mindestlänge ≥ 2 (2048 mit n = 5, dazu 4500 zufällige Reihen n ≤ 16 mit Mindestlänge 2, 3 und 5) und 0 von 10 400 Gleitkomma-Instanzen")
