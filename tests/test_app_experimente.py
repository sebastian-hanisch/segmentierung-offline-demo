"""AppTest der Experimente: Verdikte im Standard, jeder Schnellstart, jede Option jeder Auswahl, gesperrte Regler, Fehlermeldungen, Obergrenzen."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import sgo_constants as C
import sgo_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texte(at) -> str:
    return " ".join([w.value for w in at.success] + [w.value for w in at.info] + [w.value for w in at.warning] + [w.value for w in at.error])


def _expander(at, titel):
    return next(e for e in at.expander if e.label == titel)


def test_seite_hat_fuenf_experimente_kernfragen_grenzen_und_titel():
    at = _fresh_app()
    assert at.title
    heads = " ".join(h.value for h in at.subheader)
    for k in ("Experiment 1", "Experiment 2", "Experiment 3", "Experiment 4", "Experiment 5", "Kernfragen", "Wo die Annahmen enden"):
        assert k in heads, k
    assert _expander(at, "📐 Mathematische Formulierung") is not None


def test_standard_zeigt_alle_verdikte_und_die_diagramme():
    at = _fresh_app()
    text = _texte(at)
    assert "Zahl richtig, Lagefehler" in text                       # Experiment 1
    assert "kein c in der Tabelle erreicht 90 %" in text            # Experiment 2 (Zickzack δ = 1)
    assert "Exakt: **24,70" in text                                  # Experiment 3 (n = 3, c = 2)
    assert "Die binäre Segmentierung liefert" in text               # Experiment 4
    assert "23,02 %" in text and "7,9058" in text                    # Experiment 5 (exakt)
    assert len(at.get("plotly_chart")) == 6                         # Reihe, Fenster, F1, Nullreihe gegen n, Online, Verzögerung


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_laeuft_und_setzt_alle_regler(name):
    at = _fresh_app()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    for key, state_key in P.PRESET_KEYS.items():
        soll = C.PRESETS[name].get(key, P.SETTING_SPECS[state_key].default)
        ist = at.session_state[state_key]
        assert abs(ist - soll) < 1e-9 if isinstance(soll, float) else ist == soll, (name, state_key)


def test_zufalls_button_aendert_den_seed():
    at = _fresh_app()
    alt = at.session_state["seed_input"]
    at.sidebar.button[-1].click().run(timeout=TIMEOUT)
    assert not at.exception and at.session_state["seed_input"] != alt


def test_every_option_of_every_selection_does_not_raise():
    """Jede Option jeder Auswahl (selectbox, select_slider, radio) einzeln."""
    gefunden = 0
    erwartet = 0
    for state_key, spec in P.SETTING_SPECS.items():
        if spec.options is None:
            continue
        erwartet += len(spec.options)
        for option in spec.options:
            at = _fresh_app()
            w = None
            for art in ("select_slider", "selectbox", "radio"):
                try:
                    w = getattr(at.sidebar, art)(key=state_key)
                    break
                except KeyError:
                    continue
            assert w is not None, state_key
            w.set_value(option).run(timeout=TIMEOUT)
            gefunden += 1
            assert not at.exception, (state_key, option, [str(e) for e in at.exception])
            assert at.session_state[state_key] == option
    assert gefunden == erwartet and erwartet == 6 + 2 + 4 + 5 + 3 + 4 + 5 + 2 + 9 + 4 + 7 + 5 + 4 + 3 + 2


def test_gesperrte_regler_sind_im_standard_und_bei_den_bedingungen_gesperrt():
    at = _fresh_app()
    assert at.sidebar.text_area(key="reihe_text").proto.disabled                 # Szenario statt eigener Reihe
    assert at.sidebar.slider(key="beta1_slider").proto.disabled                  # Strafe als c · ln n
    assert not at.sidebar.slider(key="c1_slider").proto.disabled and not at.sidebar.slider(key="delta1_slider").proto.disabled
    at.sidebar.selectbox(key="szen1_select").set_value(C.SZ_EIGEN).run(timeout=TIMEOUT)
    assert not at.sidebar.text_area(key="reihe_text").proto.disabled
    assert at.sidebar.slider(key="delta1_slider").proto.disabled and at.sidebar.slider(key="n1_slider").proto.disabled
    at.sidebar.selectbox(key="szen1_select").set_value(C.SZ_NULL).run(timeout=TIMEOUT)
    assert at.sidebar.slider(key="delta1_slider").proto.disabled and not at.sidebar.slider(key="n1_slider").proto.disabled
    at.sidebar.radio(key="modus_radio").set_value(C.MODUS_BETA).run(timeout=TIMEOUT)
    assert at.sidebar.slider(key="c1_slider").proto.disabled and not at.sidebar.slider(key="beta1_slider").proto.disabled


def test_eigene_reihe_ungueltig_zeigt_fehlermeldung_und_die_uebrigen_experimente_laufen():
    at = _fresh_app()
    at.sidebar.selectbox(key="szen1_select").set_value(C.SZ_EIGEN).run(timeout=TIMEOUT)
    at.sidebar.text_area(key="reihe_text").set_value("1 2 abc").run(timeout=TIMEOUT)
    assert not at.exception
    assert any("keine Zahl" in e.value for e in at.error)
    assert "23,02 %" in _texte(at)                                   # Experiment 5 läuft weiter
    at.sidebar.text_area(key="reihe_text").set_value("").run(timeout=TIMEOUT)
    assert any("leer" in e.value for e in at.error)


def test_obergrenze_der_nullreihe_zeigt_meldung_statt_rechnung():
    at = _fresh_app()
    at.sidebar.select_slider(key="n3_select").set_value(800).run(timeout=TIMEOUT)      # Standard R = 5000: 4 Mio. > 40 000
    assert not at.exception, [str(e) for e in at.exception]
    assert any("Live-Lauf nicht gerechnet" in w.value for w in at.warning)


def test_gefrorene_kurve_nur_bei_n200_und_mindestlaenge_5():
    at = _fresh_app()
    at.sidebar.select_slider(key="n2_select").set_value(100).run(timeout=TIMEOUT)
    assert not at.exception and any("gilt nur für n = 200" in i.value for i in at.info)


def test_puls_von_hand_preset_zeigt_den_anker_im_verdikt():
    at = _fresh_app()
    at.sidebar.button(key="preset_Puls von Hand: binäre Segmentierung blind (β = 3)").click().run(timeout=TIMEOUT)
    assert not at.exception
    text = _texte(at)
    assert "3 und 6" in text and "findet nichts" in text and "Zielwert 6,00 statt 8,00" in text
    assert any("Zweitentscheider" in s.value for s in at.success)


def test_pelt_gegenbeispiel_preset_zeigt_die_abweichung_und_die_aufzaehlung():
    at = _fresh_app()
    at.sidebar.button(key="preset_PELT-Gegenbeispiel (n = 5, Mindestlänge 2)").click().run(timeout=TIMEOUT)
    assert not at.exception
    assert "4,17" in _texte(at) and "4,00" in _texte(at)
    assert any("Zweitentscheider" in s.value for s in at.success)


def test_robuste_streuung_laeuft():
    at = _fresh_app()
    at.sidebar.radio(key="sigma2_radio").set_value(C.SIGMA_ROBUST).run(timeout=TIMEOUT)
    assert not at.exception and "robust geschätzt" in _texte(at)
