"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass

import streamlit as st

import sgo_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None      # Auswahl (select_slider, radio): nur diese Werte sind gültig
    step: float | None = None         # Raster eines Zahlenreglers (Permalink rastet ein)


SETTING_SPECS = {
    "szen1_select": SettingSpec("szen1", str, C.SZ_EIN, options=C.SCEN_ALL),
    "reihe_text": SettingSpec("reihe", str, C.DEFAULT_OWN),
    "delta1_slider": SettingSpec("d1", float, C.DEFAULT_DELTA1, C.DELTA1_MIN, C.DELTA1_MAX, step=C.DELTA1_STEP),
    "n1_slider": SettingSpec("n1", int, C.DEFAULT_N1, C.N1_MIN, C.N1_MAX, step=C.N1_STEP),
    "modus_radio": SettingSpec("modus", str, C.DEFAULT_MODUS, options=C.MODI),
    "c1_slider": SettingSpec("c1", float, C.DEFAULT_C1, C.C1_MIN, C.C1_MAX, step=C.C1_STEP),
    "beta1_slider": SettingSpec("b1", float, C.DEFAULT_BETA1, C.BETA1_MIN, C.BETA1_MAX, step=C.BETA1_STEP),
    "minlen1_slider": SettingSpec("m1", int, C.DEFAULT_MINLEN1, C.MINLEN_MIN, C.MINLEN_MAX, step=1),
    "methode_radio": SettingSpec("meth", str, C.DEFAULT_METHODE, options=C.METHODEN),
    "szen2_select": SettingSpec("szen2", str, C.DEFAULT_SZEN2, options=C.SCEN_FIX),
    "delta2_select": SettingSpec("d2", float, C.DEFAULT_DELTA2, C.DELTA2_OPTIONS[0], C.DELTA2_OPTIONS[-1], options=C.DELTA2_OPTIONS),
    "n2_select": SettingSpec("n2", int, C.DEFAULT_N2, C.N2_OPTIONS[0], C.N2_OPTIONS[-1], options=C.N2_OPTIONS),
    "minlen2_select": SettingSpec("m2", int, C.DEFAULT_MINLEN2, C.MINLEN2_OPTIONS[0], C.MINLEN2_OPTIONS[-1], options=C.MINLEN2_OPTIONS),
    "r2_slider": SettingSpec("r2", int, C.DEFAULT_R2, C.R2_MIN, C.R2_MAX, step=C.R2_STEP),
    "sigma2_radio": SettingSpec("streu", str, C.DEFAULT_SIGMA, options=C.SIGMAS),
    "n3_select": SettingSpec("n3", int, C.DEFAULT_N3, C.N3_OPTIONS[0], C.N3_OPTIONS[-1], options=C.N3_OPTIONS),
    "c3_select": SettingSpec("c3", float, C.DEFAULT_C3, C.C3_OPTIONS[0], C.C3_OPTIONS[-1], options=C.C3_OPTIONS),
    "r3_select": SettingSpec("r3", int, C.DEFAULT_R3, C.R3_OPTIONS[0], C.R3_OPTIONS[-1], options=C.R3_OPTIONS),
    "szen4_select": SettingSpec("szen4", str, C.DEFAULT_SZEN4, options=C.SCEN_FIX),
    "minlen4_select": SettingSpec("m4", int, C.DEFAULT_MINLEN4, C.MINLEN4_OPTIONS[0], C.MINLEN4_OPTIONS[-1], options=C.MINLEN4_OPTIONS),
    "c4_select": SettingSpec("c4", float, C.DEFAULT_C4, C.C4_OPTIONS[0], C.C4_OPTIONS[-1], options=C.C4_OPTIONS),
    "r4_slider": SettingSpec("r4", int, C.DEFAULT_R4, C.R4_MIN, C.R4_MAX, step=C.R4_STEP),
    "delta5_select": SettingSpec("d5", float, C.DEFAULT_DELTA5, C.DELTA5_OPTIONS[0], C.DELTA5_OPTIONS[-1], options=C.DELTA5_OPTIONS),
    "nu5_slider": SettingSpec("nu", int, C.DEFAULT_NU5, C.NU5_MIN, C.NU5_MAX, step=C.NU5_STEP),
    "r5_slider": SettingSpec("r5", int, C.DEFAULT_R5, C.R5_MIN, C.R5_MAX, step=C.R5_STEP),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"szen1": "szen1_select", "reihe": "reihe_text", "delta1": "delta1_slider", "n1": "n1_slider", "modus": "modus_radio", "c1": "c1_slider", "beta1": "beta1_slider",
               "minlen1": "minlen1_slider", "methode": "methode_radio", "szen2": "szen2_select", "delta2": "delta2_select", "n2": "n2_select", "minlen2": "minlen2_select",
               "r2": "r2_slider", "sigma2": "sigma2_radio", "n3": "n3_select", "c3": "c3_select", "r3": "r3_select", "szen4": "szen4_select", "minlen4": "minlen4_select",
               "c4": "c4_select", "r4": "r4_slider", "delta5": "delta5_select", "nu5": "nu5_slider", "r5": "r5_slider", "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _snap(value, spec: SettingSpec):
    """Permalink-Wert auf die Reglergrenzen begrenzen und auf das Raster bzw. die Auswahl einrasten; None, wenn nicht möglich."""
    if spec.options is not None:
        if spec.caster is str:
            return value if value in spec.options else None
        return min(spec.options, key=lambda o: abs(o - value))
    if spec.caster is str:                       # freies Textfeld (eigene Reihe): auf die Feldlänge begrenzen
        return value[: C.OWN_TEXT_MAX]
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    if spec.step is not None and spec.lo is not None:
        value = spec.lo + round((value - spec.lo) / spec.step) * spec.step
        value = min(spec.hi, value)
        if spec.caster is float:
            value = round(value, 4)
    return spec.caster(value)


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt und eingerastet."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            value = _snap(value, spec)
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Ein Preset setzt ALLE Regler: was es nicht nennt, kehrt zum Standardwert zurück (sonst bliebe ein früheres Preset halb erhalten)."""
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name].get(key, SETTING_SPECS[state_key].default)


def _gleich(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) < 1e-9
    return a == b


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            _gleich(st.session_state.get(state_key, spec.default), preset.get(preset_key_of.get(state_key), spec.default))
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
