"""Abbildungen: alle Achsen fest (fixedrange), Spuren und Randfälle (leere Listen, keine wahren Brüche, kein Alarm)."""

import numpy as np

import sgo_constants as C
import sgo_reference as REF
import sgo_visualization as V


def _fest(fig):
    for ax in fig.select_xaxes():
        assert ax.fixedrange is True
    for ax in fig.select_yaxes():
        assert ax.fixedrange is True


def test_reihe_mit_und_ohne_wahre_brueche_und_gefundene():
    x = list(np.arange(10.0))
    segs = [(0, 5, 2.0), (5, 10, 7.0)]
    fig = V.build_series(x, [5], [4], segs)
    _fest(fig)
    namen = [t.name for t in fig.data]
    assert "wahrer Bruch" in namen and "gefundener Bruch" in namen and namen.count("Mittel je Strecke") == 2 or len(segs) == 2
    fig = V.build_series(x, None, [], [(0, 10, 4.5)])
    _fest(fig)
    assert "wahrer Bruch" not in [t.name for t in fig.data] and "gefundener Bruch" not in [t.name for t in fig.data]
    assert list(fig.layout.xaxis.range) == [0.5, 10.5]


def test_fenster_mit_und_ohne_gefrorene_kurve():
    p = REF.fenster_pooled("ein", 1.0)
    fig = V.build_window(C.C_GRID, p["p_right"], [0.0, 0.4, 0.9, 1, 1, 1, 0.5], 100)
    _fest(fig)
    assert len(fig.data) == 3                                  # Spanne, Mittel, live
    fig = V.build_window(C.C_GRID, None, [0.0, 0.4, 0.9, 1, 1, 1, 0.5], 100)
    assert len(fig.data) == 1 and fig.layout.xaxis.type == "log"


def test_f1_und_null_n_und_verzoegerung():
    f1 = REF.fenster_pooled("ein", 1.0)["f1"]
    fig = V.build_f1(C.TOL_M, f1, {m: 0.5 for m in C.TOL_M}, 25)
    _fest(fig)
    assert len(fig.data) == 2
    _fest(V.build_f1(C.TOL_M, None, {m: 0.5 for m in C.TOL_M}, None))
    _fest(V.build_null_n(REF.null_n()["n_reihe"]))
    sf = REF.online()["exakt"]["verzoegerung"]["1.0"]["sf"]
    fig = V.build_delay(sf, [3, 5, 5, 9])
    _fest(fig)
    assert len(fig.data) == 2
    assert len(V.build_delay(sf, []).data) == 1                  # ohne simulierte Läufe nur die exakte Linie


def test_online_ohne_alarm():
    x = list(np.zeros(20))
    C0 = list(np.zeros(21))
    fig = V.build_online(x, C0, None, 10, 4.1, [])
    _fest(fig)
    assert not any((t.name or "").startswith("Alarm bei") for t in fig.data)
    fig = V.build_online(x, C0, 15, 10, 4.1, [11])
    assert any((t.name or "").startswith("Alarm bei t = 15") for t in fig.data)
