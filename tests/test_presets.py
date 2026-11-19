"""Schnellstarts: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink (Begrenzen und Einrasten)."""

import pytest

import sgo_constants as C
import sgo_evaluation as ev
import sgo_presets as P
from sgo_scenario import SeriesError, parse_series


def test_jeder_schnellstart_hat_einen_hilfetext_und_nur_bekannte_schluessel():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert C.PRESET_HELP[name]
        assert set(preset) <= set(P.PRESET_KEYS), name


def test_werte_liegen_in_den_reglergrenzen_und_optionen():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[P.PRESET_KEYS[key]]
            if spec.options is not None:
                assert value in spec.options, (name, key)
            elif spec.caster is str:
                assert len(value) <= C.OWN_TEXT_MAX
            else:
                assert spec.lo <= value <= spec.hi, (name, key)
                if spec.step:
                    k = (value - spec.lo) / spec.step
                    assert abs(k - round(k)) < 1e-9, (name, key, "nicht auf dem Raster des Reglers")


def test_standard_preset_gleich_standardeinstellungen():
    assert ev.Settings(**C.PRESETS["Ein Bruch, klar (δ = 3)"]) == ev.Settings()
    assert next(iter(C.PRESETS)) == "Ein Bruch, klar (δ = 3)"


def test_setting_specs_decken_jeden_preset_schluessel_ab_und_die_standardwerte_stimmen():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    s = ev.Settings()
    for key, state_key in P.PRESET_KEYS.items():
        assert getattr(s, key) == P.SETTING_SPECS[state_key].default, key


def test_keine_zwei_schnellstarts_haben_dieselben_reglerwerte():
    """Sonst wäre einer nie hervorgehoben (aktives_preset gibt den ersten Treffer)."""
    werte = [tuple(sorted(ev.Settings(**p).__dict__.items())) for p in C.PRESETS.values()]
    assert len(set(werte)) == len(werte)


def test_eigene_reihen_der_schnellstarts_sind_gueltig():
    for preset in C.PRESETS.values():
        if "reihe" in preset:
            assert len(parse_series(preset["reihe"])) >= 2
    assert parse_series(C.PRESETS["Puls von Hand: binäre Segmentierung blind (β = 3)"]["reihe"]) == [0, 0, 0, 2, 2, 2, 0, 0, 0]
    assert parse_series(C.PRESETS["PELT-Gegenbeispiel (n = 5, Mindestlänge 2)"]["reihe"]) == [2, 1, 0, 0, 2]


def test_einrasten_der_permalink_werte():
    snap = P._snap
    assert snap(99.0, P.SETTING_SPECS["delta1_slider"]) == C.DELTA1_MAX
    assert snap(2.3, P.SETTING_SPECS["delta1_slider"]) == 2.5 and snap(-4.0, P.SETTING_SPECS["delta1_slider"]) == C.DELTA1_MIN
    assert snap(204, P.SETTING_SPECS["n1_slider"]) == 200 and snap(206, P.SETTING_SPECS["n1_slider"]) == 210 and snap(9999, P.SETTING_SPECS["n1_slider"]) == C.N1_MAX
    assert snap(1.4, P.SETTING_SPECS["delta5_select"]) == 1.0 and snap(1.6, P.SETTING_SPECS["delta5_select"]) == 2.0
    assert snap("unsinn", P.SETTING_SPECS["szen1_select"]) is None and snap(C.SZ_PULS, P.SETTING_SPECS["szen1_select"]) == C.SZ_PULS
    assert snap("1 2 3", P.SETTING_SPECS["reihe_text"]) == "1 2 3" and len(snap("x" * 9000, P.SETTING_SPECS["reihe_text"])) == C.OWN_TEXT_MAX


def test_jeder_regler_der_experimente_hat_eine_obere_und_untere_grenze_die_zusammenpassen():
    for key, spec in P.SETTING_SPECS.items():
        if spec.options is not None:
            assert spec.default in spec.options, key
        elif spec.caster is not str:
            assert spec.lo <= spec.default <= spec.hi, key
    with pytest.raises(SeriesError):
        parse_series("a")


def test_voreinstellung_der_live_laeufe_haelt_die_obergrenzen_ein():
    assert C.DEFAULT_N3 * C.DEFAULT_R3 <= C.LIVE_LIMIT_NULL
    assert C.DEFAULT_R2 <= C.R2_MAX and C.DEFAULT_R4 <= C.R4_MAX and C.DEFAULT_R5 <= C.R5_MAX
    for name, preset in C.PRESETS.items():
        s = ev.Settings(**preset)
        assert s.n3 * s.r3 <= C.LIVE_LIMIT_NULL, name        # kein Schnellstart löst die Obergrenze-Meldung aus
