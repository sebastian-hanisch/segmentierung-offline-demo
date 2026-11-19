"""Brüche im Rückblick finden: die beste Zerlegung ist exakt, die Zahl der Brüche ist eine Wahl - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 7 der Reihe „Änderungserkennung“: eine fertige Reihe wird im Rückblick in Strecken mit gleichem Mittel zerlegt (Mittelwertsprünge, bekannte Streuung 1). Die optimale
Zerlegung ist exakt berechenbar; die Zahl der Brüche folgt aus der Strafe je Bruch. Gemessen werden Bruchzahl, Bruchlage, Zielwert und F1 mit Toleranz (keine ARL).

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

import sgo_constants as C
import sgo_evaluation as EV
import sgo_exact as EX
import sgo_reference as REF
import sgo_scenario as SC
import sgo_segmentation as S
import sgo_texts as TX
import sgo_visualization as V
from sgo_layout import dunkles_design, kopf_css
from sgo_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)

st.set_page_config(page_title="Brüche im Rückblick finden – Sebastian Hanisch", layout="wide")

de, pct = EV.de, EV.pct


@st.cache_data(show_spinner=False)
def _exp1(x, true, beta, minlen, method):
    return SC.run_exp1(list(x), list(true) if true is not None else None, beta, minlen, method)


@st.cache_data(show_spinner=False)
def _exp2(scen_key, n, delta, minlen, R, sigma_mode, seed):
    return SC.run_exp2(scen_key, n, delta, minlen, R, sigma_mode, seed)


@st.cache_data(show_spinner=False)
def _exp3(n, c, R, seed):
    return SC.run_exp3(n, c, R, seed)


@st.cache_data(show_spinner=False)
def _exp4(scen_key, minlen, c, R, seed):
    return SC.run_exp4(scen_key, minlen, c, R, seed)


@st.cache_data(show_spinner=False)
def _exp5(delta, nu, R, seed):
    return SC.run_exp5(delta, nu, R, seed, REF.h_cusum())


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in sgo_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 Brüche im Rückblick finden: die beste Zerlegung ist exakt, die Zahl der Brüche ist eine Wahl")
st.markdown(TX.INTRO)

with st.expander("So funktioniert die Suche nach Brüchen", expanded=True):
    st.markdown(TX.SO_FUNKTIONIERT)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Experiment 1: eine Reihe")
    szen1 = st.selectbox("Reihe", C.SCEN_ALL, key="szen1_select", help="Szenario mit wahren Brüchen (Lage als Anteil von n) oder eine eigene Reihe aus dem Feld darunter.")
    eigen = szen1 == C.SZ_EIGEN
    st.text_area("Eigene Reihe (nur bei „Eigene Reihe“)", key="reihe_text", height=70, disabled=not eigen, max_chars=C.OWN_TEXT_MAX,
                 help="Zahlen getrennt durch Leerzeichen, Semikolon oder Zeilenumbruch; das Komma gilt als Dezimalkomma. Höchstens 400 Werte.")
    delta1 = st.slider("Sprunghöhe δ", *bounds("delta1_slider"), step=C.DELTA1_STEP, key="delta1_slider", format="%.1f", disabled=eigen or szen1 == C.SZ_NULL,
                       help="Höhe der Mittelwertsprünge in Einheiten der Streuung 1. Bei der Nullreihe und bei eigener Reihe ohne Wirkung (Regler gesperrt).")
    n1 = st.slider("Länge n", *bounds("n1_slider"), step=C.N1_STEP, key="n1_slider", disabled=eigen, help="Zahl der Beobachtungen; die wahren Brüche liegen als Anteile von n (bei n = 200 wie in der Vormessung).")
    modus = st.radio("Strafe angeben als", C.MODI, key="modus_radio", help="β = c · ln n (c = 2 heißt „BIC-Wahl“, eine Setzung dieses Berichts) oder absolut.")
    c1 = st.slider("Strafkonstante c", *bounds("c1_slider"), step=C.C1_STEP, key="c1_slider", format="%.1f", disabled=modus != C.MODUS_C, help="β = c · ln n.")
    beta1 = st.slider("Strafe β absolut", *bounds("beta1_slider"), step=C.BETA1_STEP, key="beta1_slider", format="%.1f", disabled=modus != C.MODUS_BETA,
                      help="Zielwert = Summe der Kosten + β · Zahl der Brüche.")
    minlen1 = st.slider("Mindestlänge je Strecke", *bounds("minlen1_slider"), step=1, key="minlen1_slider", help="Jede Strecke hat mindestens so viele Beobachtungen.")
    methode = st.radio("Verfahren", C.METHODEN, key="methode_radio", help="Optimal = Dynamische Programmierung ohne Abkürzung; PELT mit K = 0; PELT verzögert (eigener Vorschlag); binäre Segmentierung.")

    st.header("⚙️ Experiment 2: die Strafe ist ein Fenster")
    szen2 = st.selectbox("Szenario", C.SCEN_FIX, key="szen2_select", help="Wahre Brüche wie in Experiment 1.")
    delta2 = st.select_slider("Sprunghöhe δ", options=C.DELTA2_OPTIONS, key="delta2_select", disabled=szen2 == C.SZ_NULL, help="Gemessen (gefroren) ist δ = 1, 2, 3. Bei der Nullreihe ohne Wirkung (gesperrt).")
    n2 = st.select_slider("Länge n", options=C.N2_OPTIONS, key="n2_select", help="Die gefrorene Kurve gilt für n = 200 und Mindestlänge 5.")
    minlen2 = st.select_slider("Mindestlänge", options=C.MINLEN2_OPTIONS, key="minlen2_select", help="Zweite Stellschraube neben der Strafe.")
    r2 = st.slider("Live-Läufe R", *bounds("r2_slider"), step=C.R2_STEP, key="r2_slider", help="Zahl der simulierten Reihen, die live gerechnet werden.")
    sigma2 = st.radio("Streuung", C.SIGMAS, key="sigma2_radio", help="Bekannt: 1 wird vorausgesetzt. Robust geschätzt: Median der Beträge der Differenzen durch (√2 · 0,6745), die Reihe wird durch den Schätzwert geteilt.")

    st.header("⚙️ Experiment 3: die Nullreihe")
    n3 = st.select_slider("Länge n", options=C.N3_OPTIONS, key="n3_select", help="n = 2 und 3: exakte Rechnung daneben. Mindestlänge 1 bis n = 10, sonst 5.")
    c3 = st.select_slider("Strafkonstante c", options=C.C3_OPTIONS, key="c3_select", help="β = c · ln n.")
    r3 = st.select_slider("Live-Läufe R", options=C.R3_OPTIONS, key="r3_select", help="Obergrenze n · R ≤ 40 000 Zufallszahlen; größere Läufe nur aus der gefrorenen Tabelle.")

    st.header("⚙️ Experiment 4: schnell gegen optimal")
    szen4 = st.selectbox("Szenario", C.SCEN_FIX, key="szen4_select", help="Reihen mit n = 200 und Sprunghöhe 1 (die Nullreihe: δ = 0).")
    minlen4 = st.select_slider("Mindestlänge", options=C.MINLEN4_OPTIONS, key="minlen4_select", help="Bei Mindestlänge 1 ist PELT beweisbar exakt.")
    c4 = st.select_slider("Strafkonstante c", options=C.C4_OPTIONS, key="c4_select", help="β = c · ln 200.")
    r4 = st.slider("Live-Läufe R", *bounds("r4_slider"), step=C.R4_STEP, key="r4_slider", help="Zahl der Reihen, die mit allen vier Verfahren gerechnet werden.")

    st.header("⚙️ Experiment 5: dieselbe Reihe, zwei Fragen")
    delta5 = st.select_slider("Sprunghöhe δ", options=C.DELTA5_OPTIONS, key="delta5_select", help="Mittelwertsprung nach oben bei ν; n = 200.")
    nu5 = st.slider("Lage ν (Beobachtungen vor dem Sprung)", *bounds("nu5_slider"), step=C.NU5_STEP, key="nu5_slider", help="Ein Bruch bei ν heißt: ν Beobachtungen liegen davor.")
    r5 = st.slider("Live-Läufe R", *bounds("r5_slider"), step=C.R5_STEP, key="r5_slider", help="Zahl der Reihen, die online und offline ausgewertet werden.")

    st.header("⚙️ Zufall")
    seed = st.number_input("Zufalls-Seed der Live-Läufe", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Live-Läufe", width="stretch", on_click=randomize_seed)

_values = {"szen1_select": szen1, "reihe_text": st.session_state["reihe_text"], "delta1_slider": float(delta1), "n1_slider": int(n1), "modus_radio": modus, "c1_slider": float(c1),
           "beta1_slider": float(beta1), "minlen1_slider": int(minlen1), "methode_radio": methode, "szen2_select": szen2, "delta2_select": float(delta2), "n2_select": int(n2),
           "minlen2_select": int(minlen2), "r2_slider": int(r2), "sigma2_radio": sigma2, "n3_select": int(n3), "c3_select": float(c3), "r3_select": int(r3), "szen4_select": szen4,
           "minlen4_select": int(minlen4), "c4_select": float(c4), "r4_slider": int(r4), "delta5_select": float(delta5), "nu5_slider": int(nu5), "r5_slider": int(r5), "seed_input": int(seed)}
if len(_values["reihe_text"]) > 300:
    del _values["reihe_text"]   # lange eigene Reihen stehen nicht in der Adresszeile
sync_query_params(_values)
S1 = EV.Settings(szen1, st.session_state["reihe_text"], float(delta1), int(n1), modus, float(c1), float(beta1), int(minlen1), methode, szen2, float(delta2), int(n2), int(minlen2),
                 int(r2), sigma2, int(n3), float(c3), int(r3), szen4, int(minlen4), float(c4), int(r4), float(delta5), int(nu5), int(r5), int(seed))

# =====================================================================================================================================================
# Experiment 1
# =====================================================================================================================================================
st.markdown("---")
st.subheader("🔬 Experiment 1: eine Reihe im Rückblick")
st.markdown(TX.exp1_frage())
res1 = None
try:
    if S1.szen1 == C.SZ_EIGEN:
        x1, true1 = SC.parse_series(S1.reihe), None
    else:
        key1 = C.SCEN_KEY[S1.szen1]
        X1, true1 = SC.draw_series(key1, S1.n1, S1.delta1, 1, S1.seed)
        x1 = [float(v) for v in X1[0]]
    n_eff = len(x1)
    beta_eff = C.beta_of(S1.c1, n_eff) if S1.modus == C.MODUS_C else S1.beta1
    res1 = _exp1(tuple(x1), tuple(true1) if true1 is not None else None, beta_eff, S1.minlen1, S1.methode)
except SC.SeriesError as e:
    st.error(str(e))
if res1 is not None:
    r = res1
    errs = S.location_errors(r.breaks, r.true) if r.true else []
    c1_, c2_, c3_, c4_ = st.columns(4)
    c1_.metric("Gefundene Brüche", str(len(r.breaks)), delta=(f"wahr: {len(r.true)}" if r.true is not None else "eigene Reihe"), delta_color="off")
    c2_.metric(f"Zielwert ({r.method})", de(r.objective, 2), delta=f"{EV.de_s(r.objective - r.dp_objective, 2)} gegen die optimale Zerlegung", delta_color="inverse")
    c3_.metric("Strafe β / Mindestlänge", f"{de(r.beta, 2)} / {r.minlen}", delta=(f"c = {de(S1.c1, 1)}" if S1.modus == C.MODUS_C else "absolut"), delta_color="off")
    c4_.metric("Lagefehler (Beobachtungen)", ", ".join(de(e, 0) for e in errs) if errs else "–", delta=("bei richtiger Zahl" if errs else "nur bei richtiger Zahl und Szenario"), delta_color="off")
    st.plotly_chart(V.build_series(r.x, r.true, r.breaks, r.segments), width="stretch", key="series_chart")
    kind, text = EV.verdict_exp1(r.n, r.true, r.breaks, r.method, r.beta, r.minlen, r.objectives, r.obj_none, r.dp_breaks, r.bin_steps, r.no_break_possible)
    getattr(st, kind)(text)
    ev_dp = S.dp_evaluations(r.n, r.minlen)
    rows = [{"Verfahren": C.M_OPT, "Zielwert": de(r.objectives[C.M_OPT], 2), "Brüche": EV.lst(r.dp_breaks), "Kostenauswertungen": str(ev_dp)},
            {"Verfahren": C.M_PELT, "Zielwert": de(r.objectives[C.M_PELT], 2), "Brüche": EV.lst(r.pelt.breaks), "Kostenauswertungen": str(r.pelt.evaluations)},
            {"Verfahren": C.M_PELT_V, "Zielwert": de(r.objectives[C.M_PELT_V], 2), "Brüche": EV.lst(r.pelt_delayed.breaks), "Kostenauswertungen": str(r.pelt_delayed.evaluations)},
            {"Verfahren": C.M_BIN, "Zielwert": de(r.objectives[C.M_BIN], 2), "Brüche": EV.lst(r.bin_breaks), "Kostenauswertungen": "–"}]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Zielwert ohne Bruch: {de(r.obj_none, 2)}. Kostenauswertungen = zulässige Strecken, deren Kosten das Verfahren ausrechnet (die binäre Segmentierung zählt hier nicht). "
               f"Zerlegungen insgesamt (Mindestlänge {r.minlen}): {S.count_decompositions(r.n, r.minlen):,}".replace(",", ".") + f". Live gerechnet in {de(r.seconds, 3)} s (Rechenzeit dieser Maschine, kein Befund).")
    if r.enumeration is not None:
        en = r.enumeration
        st.success(f"Zweitentscheider: die Aufzählung aller {en.count:,} Zerlegungen".replace(",", ".") + f" gibt als kleinsten Zielwert {de(en.best, 4)}, die optimale Zerlegung {de(r.dp_objective, 4)}: gleich; "
                   + (f"genau eine Zerlegung erreicht diesen Wert (Brüche: {EV.lst(en.optimal[0])})." if len(en.optimal) == 1 else f"{len(en.optimal)} Zerlegungen erreichen diesen Wert (Gleichstand), zum Beispiel mit Brüchen {'; '.join(EV.lst(o) for o in en.optimal[:4])}{' …' if len(en.optimal) > 4 else ''}."))
    elif r.n > S.ENUM_LIMIT:
        st.caption(f"Die Aufzählung aller Zerlegungen als Zweitentscheider gibt es bis n = {S.ENUM_LIMIT} (2¹³ = 8 192 Zerlegungen bei Mindestlänge 1); diese Reihe hat n = {r.n}.")
    with st.expander("Rechenschritte: F-Tabelle der optimalen Zerlegung und Kandidaten von PELT"):
        if r.n <= 40:
            st.markdown("**F(t) = kleinster Zielwert der ersten t Beobachtungen** (F(0) = −β, F(t) bleibt unendlich, solange t < Mindestlänge):")
            st.dataframe(pd.DataFrame({"t": list(range(0, r.n + 1)), "F(t)": [("∞" if not np.isfinite(v) else de(v, 3)) for v in r.f_table]}).set_index("t").T, width="stretch")
            ts = sorted(r.pelt.log)
            st.markdown("**Kandidatenliste (Startpunkte s) nach dem Beschneiden bei t:**")
            st.dataframe(pd.DataFrame({"t": ts, "PELT": [EV.lst(r.pelt.log[t]) if r.pelt.log[t] else "–" for t in ts], "PELT verzögert": [EV.lst(r.pelt_delayed.log[t]) if r.pelt_delayed.log[t] else "–" for t in ts]}),
                         hide_index=True, width="stretch")
        else:
            st.caption("Die F-Tabelle und die Kandidatenlisten stehen bis n = 40 zur Verfügung.")
st.markdown(TX.exp1_negativ())

# =====================================================================================================================================================
# Experiment 2
# =====================================================================================================================================================
st.markdown("---")
st.subheader("🔬 Experiment 2: die Strafe ist ein Fenster")
st.markdown(TX.exp2_frage())
res2 = None
key2 = C.SCEN_KEY[S1.szen2]
delta2_eff = 0.0 if key2 == "null" else S1.delta2
try:
    with st.spinner(f"Rechne {S1.r2} Läufe (n = {S1.n2}) für {len(C.C_GRID)} Strafen …"):
        res2 = _exp2(key2, S1.n2, delta2_eff, S1.minlen2, S1.r2, S1.sigma2, S1.seed)
except SC.LiveLimitError as e:
    st.warning(f"Live-Lauf nicht gerechnet: {e}.")
mode2 = "robust" if S1.sigma2 == C.SIGMA_ROBUST else "bekannt"
frozen2 = REF.fenster_pooled(key2, delta2_eff, mode2) if (S1.n2 == 200 and S1.minlen2 == 5) else None
if res2 is not None:
    tab_w, tab_f1 = st.tabs(["Fenster der Strafe", "Toleranz (F1 gegen M)"])
    with tab_w:
        m1, m2, m3, m4 = st.columns(4)
        i2 = C.C_GRID.index(2.0)
        m1.metric("Richtige Bruchzahl bei c = 2", pct(res2.p_right[i2]), delta=(f"gefroren: {pct(frozen2['p_right']['mean'][i2])}" if frozen2 else "gefroren nur für n = 200, Mindestlänge 5"), delta_color="off")
        m2.metric("Fenster (≥ 90 %)", (f"c = {EV.c_text(min(res2.window))} bis {EV.c_text(max(res2.window))}" if res2.window else "keines"), delta=f"in der Tabelle mit c = {C.C_GRID[0]:g} bis {C.C_GRID[-1]:g}", delta_color="off")
        m3.metric("Mittlere Bruchzahl bei c = 2", de(res2.mean_count[i2], 2), delta=f"wahr: {len(res2.true)}", delta_color="off")
        m4.metric("Live-Lauf", f"{de(res2.seconds, 2)} s", delta=f"{res2.R} Läufe, n = {res2.n}", delta_color="off")
        st.plotly_chart(V.build_window(C.C_GRID, frozen2["p_right"] if frozen2 else None, res2.p_right, res2.R), width="stretch", key="window_chart")
        kind, text = EV.verdict_exp2(key2, delta2_eff, res2.n, res2.minlen, res2.R, list(C.C_GRID), res2.p_right, res2.p_any, res2.mean_count, res2.window, S1.sigma2 == C.SIGMA_ROBUST)
        getattr(st, kind)(text)
        rows = []
        for j, c in enumerate(C.C_GRID):
            row = {"c": c, "β": de(C.beta_of(c, res2.n), 2), "live: richtige Zahl": pct(res2.p_right[j]), "live: mit Fehlbruch": pct(res2.p_any[j]), "live: mittlere Zahl": de(res2.mean_count[j], 2),
                   "live: Lagefehler": (de(res2.loc_error[j], 2) if res2.loc_error[j] is not None else "–")}
            if frozen2:
                row["gefroren: richtige Zahl (Spanne)"] = f"{pct(frozen2['p_right']['mean'][j])} ({de(100 * frozen2['p_right']['lo'][j], 1)} bis {de(100 * frozen2['p_right']['hi'][j], 1)})"
                row["gefroren: mit Fehlbruch"] = pct(frozen2["p_any"]["mean"][j])
                row["gefroren: mittlere Zahl"] = de(frozen2["mean_count"]["mean"][j], 2)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if frozen2:
            st.caption(f"Gefroren: drei Zufalls-Seeds mit je R = {frozen2['R']} Läufen, n = 200, Mindestlänge 5, Streuung {'robust geschätzt' if mode2 == 'robust' else 'bekannt'}; Spanne = kleinster bis größter Wert der drei Zufalls-Seeds. "
                       f"Herkunft: `sgo_fenster.json` aus `tools/sgo_make_fenster.py` (SplitMix64, Stand {REF.fenster()['meta']['stand']}). Zähleinheit Läufe.")
        else:
            st.info("Die gefrorene Tabelle gilt nur für n = 200, Mindestlänge 5 und die gemessenen Sprunghöhen δ = 1, 2, 3: hier rechnet nur der Live-Lauf.")
        if S1.sigma2 == C.SIGMA_ROBUST and res2.sigma_hat_mean is not None:
            st.caption(f"Robuste Schätzung: mittlerer Schätzwert der Streuung in den Live-Läufen {de(res2.sigma_hat_mean, 3)} (wahr: 1).")
    with tab_f1:
        gaps = [b - a for a, b in zip(res2.true[:-1], res2.true[1:])]
        half = min(gaps) / 2 if gaps else None
        if not res2.true:
            st.info("Ohne wahren Bruch ist F1 nicht definiert (Nullreihe): Brüche, die gefunden werden, sind Fehlbrüche.")
        else:
            ms = list(C.TOL_M)
            frozen_f1 = frozen2["f1"] if (frozen2 and "f1" in frozen2) else None
            st.plotly_chart(V.build_f1(ms, frozen_f1, res2.f1, half), width="stretch", key="f1_chart")
            rows = []
            for m in ms:
                ok = (half is None) or m < half
                rows.append({"M": m, "live: mittleres F1": de(res2.f1[m], 3), "live: Läufe mit F1 = 0": pct(res2.f1_zero[m]), "zulässig": "ja" if ok else "nein (M ≥ halber Abstand)"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("Ein Bruch gilt als gefunden, wenn ein anderer **näher als M** Beobachtungen liegt; F1 = 0, wenn nichts getroffen oder nichts gefunden wird. Dasselbe Ergebnis liegt je nach M zwischen "
                       f"{de(min(res2.f1.values()), 2)} und {de(max(res2.f1.values()), 2)} (c = 2, {res2.R} Läufe). Truong u. a.: M soll kleiner als der kleinste Abstand wahrer Brüche sein; "
                       + (f"hier verschärft auf den halben Abstand ({de(half, 0)}), eigene Verschärfung, keine Quellenaussage: Werte darüber sagen nichts." if half else "ein einzelner wahrer Bruch hat keinen Abstand zu einem anderen."))
st.markdown(TX.exp2_mindestlaenge_text())
st.warning(TX.exp2_falle())

# =====================================================================================================================================================
# Experiment 3
# =====================================================================================================================================================
st.markdown("---")
st.subheader("🔬 Experiment 3: die Nullreihe exakt")
st.markdown(TX.exp3_frage())
res3 = None
try:
    with st.spinner(f"Rechne {S1.r3} Nullreihen mit n = {S1.n3} …"):
        res3 = _exp3(S1.n3, S1.c3, S1.r3, S1.seed)
except SC.LiveLimitError as e:
    st.warning(f"Live-Lauf nicht gerechnet: {e}. Die gefrorene Tabelle (R = 1000) steht unten.")
if res3 is not None:
    c1_, c2_, c3_ = st.columns(3)
    c1_.metric("Läufe mit Fehlbruch (Simulation)", pct(res3.p_any, 2), delta=f"± {de(100 * res3.se, 2)} Standardfehler, R = {res3.R}", delta_color="off")
    c2_.metric("Exakt", pct(res3.exact_p_any, 3) if res3.exact_p_any is not None else "keine geschlossene Form", delta=("n = 2: χ²₁; n = 3: Winkelintegral" if res3.exact_p_any is not None else "n ≥ 4: nicht hergeleitet"),
               delta_color="off")
    c3_.metric("Feste Stelle: 2·Φ̄(√β)", pct(EX.p_fixed_site(res3.beta), 3), delta=f"β = {de(res3.beta, 3)}", delta_color="off")
    kind, text = EV.verdict_exp3(res3.n, res3.c, res3.beta, res3.minlen, res3.R, res3.p_any, res3.se, res3.exact_p_any)
    getattr(st, kind)(text)
    st.caption(f"Live gerechnet in {de(res3.seconds, 2)} s (Rechenzeit dieser Maschine, kein Befund); mittlere Zahl gefundener Brüche {de(res3.mean_count, 3)}. Zähleinheit Läufe.")
nz = REF.null_n()
rows = []
for c, row in nz["n2"].items():
    rows.append({"n": 2, "c": c, "exakt": pct(row["exakt"], 3), "Simulation (Formel, 2 Mio. Läufe)": pct(row["sim_formel"], 3), "Simulation (DP, 200 000 Läufe)": pct(row["sim_dp"], 3)})
for c, row in nz["n3"].items():
    rows.append({"n": 3, "c": c, "exakt": pct(1 - row["exakt_kein_bruch"], 3), "Simulation (Formel, 2 Mio. Läufe)": f"Aufzählung: {pct(1 - row['sim_aufzaehlung'], 3)}", "Simulation (DP, 200 000 Läufe)": f"DP (1 Mio.): {pct(1 - row['sim_dp'], 3)}"})
st.markdown("**Gefroren: n = 2 und n = 3, exakt neben Simulation** (Anteil Läufe mit mindestens einem Fehlbruch; Mindestlänge 1; Zufalls-Seeds in `sgo_null_n.json`):")
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.plotly_chart(V.build_null_n(nz["n_reihe"]), width="stretch", key="null_n_chart")
rows = []
for n_, row in nz["n_reihe"].items():
    rows.append({"n": int(n_), "Stellen (Mindestlänge 5)": row["n_stellen"], "c = 1: mit Fehlbruch": pct(row["1.0"]["p_fehlbruch"]), "c = 1: feste Stelle": pct(row["1.0"]["feste_stelle"], 2),
                 "c = 1: Stellen × Einzelwert (keine Schranke)": pct(row["1.0"]["stellen_mal_einzel"], 0), "c = 2: mit Fehlbruch": pct(row["2.0"]["p_fehlbruch"]),
                 "c = 2: Stellen × Einzelwert (keine Schranke)": pct(row["2.0"]["stellen_mal_einzel"], 1), "c = 3: mit Fehlbruch": pct(row["3.0"]["p_fehlbruch"])})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption(f"Gefroren: R = 1000 Läufe je Zeile, Mindestlänge 5, Standardfehler bis 1,6 Prozentpunkte; Stand {nz['meta']['stand']}. Zähleinheit Läufe.")
st.markdown(TX.exp3_negativ())

# =====================================================================================================================================================
# Experiment 4
# =====================================================================================================================================================
st.markdown("---")
st.subheader("🔬 Experiment 4: schnell gegen optimal")
st.markdown(TX.exp4_frage())
key4 = C.SCEN_KEY[S1.szen4]
delta4 = 0.0 if key4 == "null" else 1.0
res4 = None
try:
    with st.spinner(f"Rechne {S1.r4} Reihen mit n = {C.N4} und allen vier Verfahren …"):
        res4 = _exp4(key4, S1.minlen4, S1.c4, S1.r4, S1.seed)
except SC.LiveLimitError as e:
    st.warning(f"Live-Lauf nicht gerechnet: {e}.")
if res4 is not None:
    c1_, c2_, c3_, c4_ = st.columns(4)
    c1_.metric("PELT: größerer Zielwert in", f"{res4.pelt_diff} von {res4.R} Läufen", delta=("Mindestlänge 1: beweisbar exakt" if res4.minlen == 1 else f"Mindestlänge {res4.minlen}"), delta_color="off")
    c2_.metric("Verzögert: Abweichung in", f"{res4.delayed_diff} von {res4.R}", delta="Zielwert oder Bruchmenge", delta_color="off")
    c3_.metric("Binär: andere Bruchmenge in", f"{res4.bin_other} von {res4.R}", delta=f"davon schlechter: {res4.bin_worse}; besser: {res4.bin_better}", delta_color="off")
    c4_.metric("Kostenauswertungen PELT / DP", de(res4.ev_pelt / res4.ev_dp, 3), delta=f"verzögert: {de(res4.ev_delayed / res4.ev_dp, 3)}", delta_color="off")
    kind, text = EV.verdict_exp4(key4, res4.c, res4.minlen, res4.R, res4.n, res4.pelt_diff, res4.delayed_diff, res4.bin_other, res4.bin_worse, res4.bin_better, res4.ev_dp, res4.ev_pelt, res4.ev_delayed)
    getattr(st, kind)(text)
    st.caption(f"Live gerechnet in {de(res4.seconds, 2)} s (Rechenzeit dieser Maschine, kein Befund). Sprunghöhe δ = {de(delta4, 0)}, Zufalls-Seed {S1.seed}; Zähleinheit Läufe und Kostenauswertungen.")
st.markdown("**Das Beispiel n = 5 schrittweise:** x = (2, 1, 0, 0, 2), Mindestlänge 2, β = 1. Es gibt nur drei Zerlegungen (ohne Bruch, Bruch bei 2, Bruch bei 3) mit den Zielwerten 4, 25/6 und 5.")
ex_x = np.array([2.0, 1.0, 0.0, 0.0, 2.0])
ex_dp = S.dp_batch(ex_x, [1.0], 2)
ex_p = S.pelt(ex_x, 1.0, 2, trace=True)
ex_d = S.pelt(ex_x, 1.0, 2, delayed=True, trace=True)
st.dataframe(pd.DataFrame({"t": sorted(ex_p.log), "F(t) (PELT)": [de(ex_p.f_table[t], 4) for t in sorted(ex_p.log)], "Kandidaten PELT": [EV.lst(ex_p.log[t]) for t in sorted(ex_p.log)],
                           "Kandidaten verzögert": [EV.lst(ex_d.log[t]) for t in sorted(ex_d.log)]}), hide_index=True, width="stretch")
st.caption(f"Die optimale Zerlegung findet {EV.lst(ex_dp.breaks()) if ex_dp.breaks() else 'keinen Bruch'} mit Zielwert {de(ex_dp.F[0, 0], 4)}; PELT endet bei {de(ex_p.objective, 4)} mit Bruch bei {EV.lst(ex_p.breaks)}, "
           f"das verzögerte Streichen bei {de(ex_d.objective, 4)} ohne Bruch. Bei t = 4 streicht die Regel den Start 0 (F(0) + C(0, 4) = 7/4 ≥ F(4) = 3/2); der Ersatzweg über 4 hat aber die letzte Strecke (4, 5] "
           "mit der Länge 1 < 2 und ist nicht zulässig. Auch im Schnellstart „PELT-Gegenbeispiel“ in Experiment 1.")
vf = REF.verfahren()
rows = []
for ml, rec in vf["n5_exakt"].items():
    rows.append({"Mindestlänge": ml, "Instanzen (n = 5, alle 1024 Reihen, β = 1 und 2)": rec["instanzen"], "PELT weicht ab": rec["pelt_abw_beta1"] + rec["pelt_abw_beta2"], "verzögert weicht ab": rec["verzoegert_abw"]})
st.markdown("**Gefroren, exakt (Brüche, `fractions`):** alle ganzzahligen Reihen mit n = 5 und Werten 0 bis 3:")
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
rows = []
for ml, rec in vf["pelt_n200"].items():
    rows.append({"Mindestlänge": ml, "Instanzen (n = 200, 13 Zellen × 40 Läufe × c = 1, 2 je Zufalls-Seed)": rec["instanzen"], "PELT weicht ab": rec["pelt_abw"], "verzögert weicht ab": rec["verzoegert_abw"],
                 "Kostenauswertungen PELT / DP": de(rec["verhaeltnis_pelt"], 3), "verzögert / DP": de(rec["verhaeltnis_verzoegert"], 3), "Zufalls-Seeds": len(rec["seeds"])})
st.markdown("**Gefroren, Simulation:** n = 200, Szenario-Zellen der Vormessung:")
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
rows = []
for k, v in vf["binseg_n200"]["cells"].items():
    sc, d = k.split("|")
    rows.append({"Zelle": f"{EV.scen_text(sc)}" + (f" δ = {d.replace('.', ',')}" if sc != "null" else ""), "binär ≠ optimal, c = 1": pct(v["1.0"]["andere"]), "c = 2": pct(v["2.0"]["andere"]), "c = 4": pct(v["4.0"]["andere"])})
st.markdown("**Gefroren: binäre Segmentierung gegen die optimale Zerlegung** (Anteil Läufe mit anderer Bruchmenge, n = 200, Mindestlänge 5, 400 Läufe je Zelle und Strafe):")
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption(f"Alle Zellen: {vf['binseg_n200']['gesamt']['andere']:,} von {vf['binseg_n200']['gesamt']['laeufe_mal_strafen']:,} Läufen × Strafen verschieden ({pct(vf['binseg_n200']['gesamt']['anteil_andere'])}), "
           f"Zielwert nie besser (besser in {vf['binseg_n200']['gesamt']['besser']}); das Mittel hängt an der gesetzten Mischung (13 Zellen × c = 1, 2, 4 gleich gewichtet). "
           f"Exakt, ganzzahlige Reihen n = 4 bis 14: {vf['binseg_exakt']['andere_bruchmenge']} von {vf['binseg_exakt']['instanzen']} verschieden, Zielwert nie kleiner (kleiner in {vf['binseg_exakt']['zielwert_kleiner']}).".replace(",", "."))
st.warning(TX.exp4_falle())

# =====================================================================================================================================================
# Experiment 5
# =====================================================================================================================================================
st.markdown("---")
st.subheader("🔬 Experiment 5: dieselbe Reihe, zwei Fragen")
st.markdown(TX.exp5_frage())
res5 = None
try:
    with st.spinner(f"Rechne {S1.r5} Reihen online (CUSUM) und offline (optimale Zerlegung) …"):
        res5 = _exp5(S1.delta5, S1.nu5, S1.r5, S1.seed)
except SC.LiveLimitError as e:
    st.warning(f"Live-Lauf nicht gerechnet: {e}.")
if res5 is not None:
    c1_, c2_, c3_, c4_ = st.columns(4)
    c1_.metric("Alarm vor ν (exakt)", pct(res5.p_early_exact, 2), delta=f"Simulation: {pct(res5.p_early_sim, 1)} ({res5.R} Läufe)", delta_color="off")
    c2_.metric("Verzögerung nach ν (exakt)", de(res5.delay_exact, 4), delta=(f"Simulation: {de(res5.delay_sim, 2)}" if res5.delay_sim is not None else "kein Alarm nach ν"), delta_color="off")
    c3_.metric("Rückblick: genau ein Bruch", pct(res5.off_one, 1), delta="alle 200 Beobachtungen, c = 2, Mindestlänge 5", delta_color="off")
    c4_.metric("Lagefehler online / offline", f"{de(res5.online_lage_mean, 2) if res5.online_lage_mean is not None else '–'} / {de(res5.off_lage_mean, 2) if res5.off_lage_mean is not None else '–'}",
               delta=f"gleiche Läufe: {res5.same_n} von {res5.R}", delta_color="off")
    st.plotly_chart(V.build_online(res5.x0, res5.C0, res5.alarm0, res5.nu, res5.h, res5.dp_breaks0), width="stretch", key="online_chart")
    st.caption(f"Erste der Live-Reihen (Zufalls-Seed {S1.seed}): wahrer Bruch bei ν = {res5.nu}, Rückblick findet {EV.lst(res5.dp_breaks0)}, CUSUM-Alarm "
               f"{'bei t = ' + str(res5.alarm0) if res5.alarm0 else 'keiner bis 200'}. k = 0,5, h = {de(res5.h, 4)} (ARL₀ = 370,398, zero-state).")
    st.plotly_chart(V.build_delay(res5.delay_sf, res5.delay_values), width="stretch", key="delay_chart")
    kind, text = EV.verdict_exp5(res5.delta, res5.nu, res5.R, res5.p_early_sim, res5.p_early_exact, res5.delay_sim, res5.delay_exact, res5.off_one, res5.off_lage_mean, res5.online_lage_mean, res5.online_n, res5.same_n)
    getattr(st, kind)(text)
    st.caption(f"Live gerechnet in {de(res5.seconds, 2)} s (Rechenzeit dieser Maschine, kein Befund). Verzögerung: Beobachtungen nach dem Sprung bis zum Alarm, gegeben kein Alarm vor ν, Start 0 und ν Schritte "
               f"Einschwingen (für den CUSUM praktisch das steady-state-Maß, quasistationär; nicht das Lorden- und nicht das Shiryaev-Maß). Lagefehler: mittlerer Betrag; online nur Läufe mit Alarm nach ν.")
ol = REF.online()
tab_t, tab_p = st.tabs(["Gefroren: ν = 100, R = 4000", "Phase I als Offline-Suche"])
with tab_t:
    rows = []
    for d, s in ol["simulation"].items():
        ex = ol["exakt"]["verzoegerung"][d]
        rows.append({"δ": d.replace(".", ","), "Alarm vor ν: exakt / Simulation": f"{pct(ex['p_alarm_vor_nu'], 2)} / {pct(s['p_alarm_vor_nu'], 2)}", "Verzögerung: exakt / Simulation": f"{de(ex['mittel'], 4)} / {de(s['verzoegerung_mittel'], 3)}",
                     "Lagefehler online (Läufe mit Alarm nach ν)": f"{de(s['online_lage_betrag'], 2)} ({pct(s['online_lage_in5'], 1)} innerhalb ±5)",
                     "Rückblick: genau ein Bruch": pct(s["offline_genau_ein_bruch"], 1), "Lagefehler offline (genau ein Bruch)": f"{de(s['offline_lage_betrag'], 2)} ({pct(s['offline_lage_in5'], 1)} innerhalb ±5)",
                     "gleiche Läufe: offline Lagefehler": de(s["gleiche_offline_lage_betrag"], 2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    ex = ol["exakt"]
    st.caption(f"h = {de(ex['h'], 5)} für ARL₀ = 370,398: Integralgleichung {de(ex['arl0_integral'], 4)}, Brook/Evans-Kette {de(ex['arl0_kette_400'], 4)} (400 Zellen) und {de(ex['arl0_kette_800'], 4)} (800 Zellen). "
               f"ARL₁ (zero-state) δ = 1: {de(ex['arl1']['1.0']['integral'], 4)} / {de(ex['arl1']['1.0']['kette_400'], 4)}; δ = 2: {de(ex['arl1']['2.0']['integral'], 4)} / {de(ex['arl1']['2.0']['kette_400'], 4)} (beide Wege). "
               f"Zufalls-Seed {ol['simulation']['1.0']['seed']}, R = {ol['simulation']['1.0']['R']}; Stand {ol['meta']['stand']}. Zahlen des Stücks 2 (ARL, Lorden-Näherung) stehen dort.")
with tab_p:
    rows = []
    for d, v in ol["phase1"].items():
        rows.append({"δ": d.replace(".", ","), "Sprung in der Mitte (bei 50 von 100)": pct(v["mitte"], 1), "Sprung am Rand (bei 10 von 100)": pct(v["rand"], 1), "R": v["R"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Phase-I-Umfang 100 Beobachtungen, ein verdeckter Sprung der Höhe δ, optimale Zerlegung mit c = 2 und Mindestlänge 5: Anteil der Läufe mit mindestens einem gefundenen Bruch (bei δ = 0 ist jeder Fund ein Fehlbruch, der Anteil liegt bei etwa 5 %: 4,8 bis 5,9). "
               "Ein Sprung in der Mitte hebt in der Phase-I-Stichprobe das Mittel um δ/2 und die Streuung auf √(1 + δ²/4) (δ = 1: 0,5 und 1,118). **Die Folge für die Karte (bedingte ARL₀) ist nicht gemessen**: sie gehört zu Stück 4.")
st.markdown(TX.exp5_negativ())

# =====================================================================================================================================================
# Kernfragen, Grenzen und Formeln
# =====================================================================================================================================================
st.markdown("---")
st.subheader("❓ Kernfragen")
bul = []
if res1 is not None:
    bul.append(f"1. **Wo liegen die Brüche, und wie viele sind es?** In der Reihe aus Experiment 1 {EV.zahl_brueche(len(res1.breaks))}" + (f" bei {EV.lst(res1.breaks)}" if res1.breaks else "") + f" ({res1.method}, β = {de(res1.beta, 2)}).")
if res2 is not None:
    bul.append(f"2. **Ist die Zahl der Brüche ein Messergebnis oder eine Strafwahl?** Eine Strafwahl: im Szenario aus Experiment 2 erreichen {len(res2.window)} von {len(C.C_GRID)} Werten von c mindestens 90 % richtige Bruchzahl "
               f"({res2.R} Läufe).")
if res4 is not None:
    bul.append(f"3. **Findet das schnelle Verfahren dasselbe wie das optimale?** In Experiment 4: PELT weicht in {res4.pelt_diff} von {res4.R} Läufen ab, die binäre Segmentierung liefert in {res4.bin_other} eine andere Bruchmenge (nie eine bessere).")
if res5 is not None:
    bul.append(f"4. **Was sagt der Rückblick, was der Alarm nicht sagt?** Der Alarm sagt früh, dass sich etwas geändert hat (Verzögerung {de(res5.delay_exact, 2)} Beobachtungen, exakt), der Rückblick sagt, wo und wie oft "
               f"(genau ein Bruch in {pct(res5.off_one, 1)} der Läufe); verschiedene Maße.")
st.markdown("\n".join(bul))

st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(TX.ASSUMPTIONS_MARKDOWN)

st.markdown("---")
with st.expander("📐 Mathematische Formulierung"):
    st.markdown(TX.MATH_INTRO)
    st.markdown(TX.CONVENTIONS_MARKDOWN)
    st.markdown(TX.MATH_MARKDOWN)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Änderungserkennung: Alarm und Lauflänge](https://sebastianhanisch.net/konzepte-aenderungserkennung.html)."
)
