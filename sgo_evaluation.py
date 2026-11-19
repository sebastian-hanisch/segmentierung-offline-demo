"""Einstellungen, Zahlenformat und Verdikte. Jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text).

Maß in jedem Satz: **Bruchzahl** (richtig oder falsch), **Bruchlage** (Fehler in Beobachtungen), **Zielwert**, **F1 mit Toleranz M**; Zähleinheiten Läufe, Zerlegungen,
Kostenauswertungen. Eine ARL gibt es für die Zerlegung nicht; sie kommt nur in Experiment 5 vor, dort mit ihrem Maß. Ein **Fehlbruch** ist ein gefundener Bruch ohne wahren
Bruch, ein **Fehlalarm** nur ein Alarm der Online-Karte vor ν."""

import math
from dataclasses import dataclass

import sgo_constants as C


@dataclass(frozen=True)
class Settings:
    szen1: str = C.SZ_EIN
    reihe: str = C.DEFAULT_OWN
    delta1: float = C.DEFAULT_DELTA1
    n1: int = C.DEFAULT_N1
    modus: str = C.DEFAULT_MODUS
    c1: float = C.DEFAULT_C1
    beta1: float = C.DEFAULT_BETA1
    minlen1: int = C.DEFAULT_MINLEN1
    methode: str = C.DEFAULT_METHODE
    szen2: str = C.DEFAULT_SZEN2
    delta2: float = C.DEFAULT_DELTA2
    n2: int = C.DEFAULT_N2
    minlen2: int = C.DEFAULT_MINLEN2
    r2: int = C.DEFAULT_R2
    sigma2: str = C.DEFAULT_SIGMA
    n3: int = C.DEFAULT_N3
    c3: float = C.DEFAULT_C3
    r3: int = C.DEFAULT_R3
    szen4: str = C.DEFAULT_SZEN4
    minlen4: int = C.DEFAULT_MINLEN4
    c4: float = C.DEFAULT_C4
    r4: int = C.DEFAULT_R4
    delta5: float = C.DEFAULT_DELTA5
    nu5: int = C.DEFAULT_NU5
    r5: int = C.DEFAULT_R5
    seed: int = C.DEFAULT_SEED


# ------------------------------------------------------------------------------------------------------------------ Zahlenformat (Dezimalkomma)
def de(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def de_s(x: float, nd: int = 1) -> str:
    """Mit Vorzeichen (ASCII-Minus, damit st.metric die Farbe des Deltas erkennt): mein Wert minus Referenz."""
    return ("+" if x >= 0 else "-") + de(abs(x), nd)


def pct(share: float, nd: int = 1) -> str:
    """Anteil (0 bis 1) als Prozentzahl mit Dezimalkomma."""
    return de(100.0 * share, nd) + " %"


def se_share(share: float, R: int) -> float:
    """Standardfehler eines Anteils aus R Läufen (mindestens der Wert für einen halben Treffer, damit Nullen keinen Standardfehler 0 liefern)."""
    return max(math.sqrt(max(share * (1 - share), 0.0) / R), 0.5 / R)


def lst(vals) -> str:
    """Liste von Brüchen als Text: „3 und 6“, „50, 100 und 150“, „keiner“."""
    v = [str(int(x)) for x in vals]
    if not v:
        return "keiner"
    if len(v) == 1:
        return v[0]
    return ", ".join(v[:-1]) + " und " + v[-1]


def zahl_brueche(k: int) -> str:
    return "keinen Bruch" if k == 0 else ("einen Bruch" if k == 1 else f"{k} Brüche")


def mit_brueche(breaks) -> str:
    """„ohne Bruch“, „mit einem Bruch (bei 2)“, „mit 3 Brüchen (bei 50, 100 und 150)“."""
    k = len(breaks)
    if k == 0:
        return "ohne Bruch"
    return ("mit einem Bruch" if k == 1 else f"mit {k} Brüchen") + f" (bei {lst(breaks)})"


def c_text(c: float) -> str:
    return de(c, 1) if abs(c - round(c)) > 1e-9 else str(int(round(c)))


# ------------------------------------------------------------------------------------------------------------------ Verdikt Experiment 1
def verdict_exp1(n, true, breaks, method, beta, minlen, objectives, obj_none, dp_breaks, bin_steps, no_break_possible, loc_tol=C.LOC_TOL):
    """Eine Reihe im Rückblick. `true` = wahre Brüche (None bei eigener Reihe), `objectives` = Zielwerte der vier Verfahren, `obj_none` = Zielwert ohne Bruch."""
    if no_break_possible:
        return "warning", (f"Die Reihe hat n = {n} Werte und die Mindestlänge ist {minlen}: eine Zerlegung in zwei Strecken braucht mindestens 2 · {minlen} = {2 * minlen} Werte, "
                           f"also ist **kein Bruch möglich**. Zielwert ohne Bruch: {de(obj_none, 2)}.")
    dp_obj = objectives[C.M_OPT]
    # Verfahren-Befunde zuerst (Anker: Puls von Hand, PELT-Gegenbeispiel)
    if method == C.M_BIN and not breaks and dp_breaks:
        top = max((s[3] for s in bin_steps), default=0.0)
        return "warning", (f"Die optimale Zerlegung findet {lst(dp_breaks)} (Zielwert {de(dp_obj, 2)} statt {de(obj_none, 2)} ohne Bruch), die **binäre Segmentierung findet nichts**: "
                           f"der größte Einzelgewinn einer Teilung ist {de(top, 2)}, und sie teilt nur bei einem Gewinn über β = {de(beta, 2)}. Die Brüche zusammen entfernen aber mehr Kosten "
                           f"als jeder einzeln. Zähleinheit: eine Reihe (n = {n}), Zerlegungen mit Mindestlänge {minlen}.")
    if method in (C.M_PELT, C.M_PELT_V) and objectives[method] > dp_obj + 1e-9:
        return "warning", (f"{method} endet bei Zielwert **{de(objectives[method], 2)}** {mit_brueche(breaks)}, die optimale Zerlegung bei **{de(dp_obj, 2)}** "
                           f"{mit_brueche(dp_breaks)}. Mindestlänge {minlen}: bei Mindestlänge 1 ist PELT exakt, bei größerer Mindestlänge nicht "
                           f"(in dieser Implementierung der Beschneidungsregel mit K = 0).")
    if method == C.M_BIN and objectives[method] > dp_obj + 1e-9:
        return "info", (f"Die binäre Segmentierung endet bei Zielwert {de(objectives[method], 2)} {mit_brueche(breaks)}, die optimale Zerlegung "
                        f"bei {de(dp_obj, 2)} {mit_brueche(dp_breaks)}. Die binäre Segmentierung ist nie besser, hier schlechter.")
    head = f"{method}: gefunden {zahl_brueche(len(breaks))}" + (f" (bei {lst(breaks)})" if breaks else "") + f"; Zielwert {de(objectives[method], 2)}, β = {de(beta, 2)}, Mindestlänge {minlen}. "
    if true is None:
        return "info", head + "Eigene Reihe: es gibt keine wahren Brüche zum Vergleichen; die Zahl der Brüche folgt aus der Strafe, nicht aus einer Messung."
    if not true:
        if breaks:
            return "warning", head + f"Die Reihe hat **keinen wahren Bruch**: {'der gefundene Bruch ist ein Fehlbruch' if len(breaks) == 1 else f'alle {len(breaks)} gefundenen Brüche sind Fehlbrüche'}."
        return "success", head + "Die Reihe hat keinen wahren Bruch: richtig, es wird auch keiner gefunden (ein einzelner Lauf, keine Wahrscheinlichkeit; Experiment 2 und 3 zählen über viele Läufe)."
    if len(breaks) == len(true):
        errs = [abs(e - t) for e, t in zip(sorted(breaks), sorted(true))]
        kind = "success" if max(errs) <= loc_tol else "info"
        return kind, head + f"Wahre Brüche bei {lst(true)}: Zahl richtig, Lagefehler {lst([round(e) for e in errs]).replace(' und ', ' und ')} Beobachtungen (ein Lauf)."
    return "warning", head + f"Wahre Brüche bei {lst(true)} ({len(true)}): die **Zahl stimmt nicht** (gefunden {len(breaks)}). Ein Lauf; Experiment 2 zählt über viele Läufe."


# ------------------------------------------------------------------------------------------------------------------ Verdikt Experiment 2
def verdict_exp2(scen_key, delta, n, minlen, R, c_grid, p_right, p_any, mean_count, window, sigma_robust=False):
    """Die Strafe ist ein Fenster: Anteil der Läufe mit richtiger Bruchzahl je c (Live-Läufe)."""
    se = max(se_share(p, R) for p in p_right)
    tail = f" Zähleinheit Läufe (R = {R}, Standardfehler bis {de(100 * se, 1)} Prozentpunkte), n = {n}, Mindestlänge {minlen}" + (", Streuung robust geschätzt." if sigma_robust else ", Streuung 1 bekannt.")
    if scen_key == "null":
        i1, i2 = c_grid.index(1.0), c_grid.index(2.0)
        return "warning", (f"Nullreihe: bei c = 1 haben {pct(p_any[i1])} der Läufe mindestens einen Fehlbruch (mittlere Zahl {de(mean_count[i1], 2)}), bei c = 2 {pct(p_any[i2])}. "
                           f"Eine zu kleine Strafe findet Brüche, die es nicht gibt; die untere Kante des Fensters ist scharf." + tail)
    if window:
        w = f"c = {c_text(min(window))} bis {c_text(max(window))}" if len(window) > 1 else f"nur c = {c_text(window[0])}"
        return "success", (f"{scen_text(scen_key)} δ = {de(delta, 1)}: richtige Bruchzahl in mindestens 90 % der Läufe bei {w} (in der Tabelle mit c = {', '.join(c_text(c) for c in c_grid)}). "
                           f"Außerhalb fällt der Anteil: zu kleine Strafe gibt Fehlbrüche, zu große lässt wahre Brüche weg." + tail)
    j = max(range(len(p_right)), key=lambda i: p_right[i])
    return "warning", (f"{scen_text(scen_key)} δ = {de(delta, 1)}: **kein c in der Tabelle erreicht 90 %** richtige Bruchzahl; am besten ist c = {c_text(c_grid[j])} mit {pct(p_right[j])}. "
                       f"Die Strafe ist nicht aus den Daten abzulesen: das Fenster hängt an Sprunghöhe und Zahl der Brüche." + tail)


def scen_text(scen_key: str) -> str:
    return {"null": "Nullreihe", "ein": "Ein Bruch", "treppe": "Treppe", "puls": "Puls", "zickzack": "Zickzack"}[scen_key]


# ------------------------------------------------------------------------------------------------------------------ Verdikt Experiment 3
def verdict_exp3(n, c, beta, minlen, R, p_sim, se, exact):
    """Nullreihe: Anteil Läufe mit mindestens einem Fehlbruch; für n = 2 und n = 3 mit exaktem Wert."""
    base = f"n = {n}, c = {c_text(c)} (β = {de(beta, 4)}), Mindestlänge {minlen}: in {R} Läufen haben {pct(p_sim, 2)} (± {de(100 * se, 2)}) mindestens einen Fehlbruch."
    if exact is not None:
        inside = abs(p_sim - exact) <= 4 * se
        return ("success" if inside else "warning"), (base + f" Exakt: **{pct(exact, 3)}** (n = {n}: " + ("geschlossene Formel 2·Φ̄(√β)" if n == 2 else "Winkelintegral, geprüft gegen ein zweidimensionales Integral")
                                                       + f"); die Simulation liegt {'innerhalb' if inside else 'außerhalb'} von vier Standardfehlern.")
    return "info", (base + " Für n ≥ 4 gibt es keine geschlossene Form (nicht hergeleitet): die Simulation ist die Referenz. Die gefrorene Tabelle (R = 1000) zeigt, wie der Anteil mit n "
                    "bei c = 1 steigt und bei c = 2 fällt.")


# ------------------------------------------------------------------------------------------------------------------ Verdikt Experiment 4
def verdict_exp4(scen_key, c, minlen, R, n, pelt_diff, delayed_diff, bin_other, bin_worse, bin_better, ev_dp, ev_pelt, ev_delayed):
    if bin_better > 0:
        return "error", f"Die binäre Segmentierung war in {bin_better} Läufen besser als die optimale Zerlegung: das kann nicht sein, die Rechnung ist fehlerhaft."
    pelt_txt = (f"PELT hat in **{pelt_diff} von {R}** Läufen einen größeren Zielwert als die optimale Zerlegung" if pelt_diff else f"PELT hat in **keinem der {R} Läufe** einen anderen Zielwert als die optimale Zerlegung")
    delayed_txt = f"das verzögerte Streichen in {delayed_diff}"
    kind = "warning" if (pelt_diff or bin_other) else "success"
    return kind, (f"{scen_text(scen_key)}, n = {n}, c = {c_text(c)}, Mindestlänge {minlen}: {pelt_txt} (Mindestlänge 1: PELT ist beweisbar exakt), {delayed_txt}. "
                  f"Die binäre Segmentierung liefert in **{bin_other} von {R}** Läufen eine andere Bruchmenge (in {bin_worse} mit größerem Zielwert), nie eine bessere. "
                  f"Kostenauswertungen im Verhältnis zur optimalen Zerlegung: PELT {de(ev_pelt / ev_dp, 3)}, verzögert {de(ev_delayed / ev_dp, 3)} (Zähleinheit zulässige Strecken; kein Zeitmaß).")


# ------------------------------------------------------------------------------------------------------------------ Verdikt Experiment 5
def verdict_exp5(delta, nu, R, p_early_sim, p_early_exact, delay_sim, delay_exact, off_one, off_lage, on_lage, on_n, same_n):
    d_sim = f"{de(delay_sim, 2)} im Mittel" if delay_sim is not None else "keine Läufe mit Alarm nach ν"
    on_txt = f"online (letzter Zeitpunkt mit C = 0 vor dem Alarm, {on_n} Läufe mit Alarm nach ν) mittlerer Lagefehler {de(on_lage, 2)}" if on_lage is not None else "online ohne Lageschätzung (kein Alarm nach ν)"
    off_txt = f"offline bei genau einem Bruch {de(off_lage, 2)}" if off_lage is not None else "offline ohne Lagefehler (nie genau ein Bruch)"
    return "info", (f"δ = {de(delta, 1)}, ν = {nu}: Der CUSUM schlägt in **{pct(p_early_exact, 2)}** der Läufe vor ν an (exakt aus der Kette; Simulation mit {R} Läufen {pct(p_early_sim, 1)}); "
                    f"danach nach {de(delay_exact, 4)} Beobachtungen (exakt, gegeben kein Alarm vor ν; Simulation {d_sim}). Der Rückblick findet mit allen {C.N5} Beobachtungen genau einen Bruch in "
                    f"{pct(off_one, 1)} der Läufe; mittlerer Lagefehler: {on_txt}; {off_txt}. Das sind verschiedene Maße (Verzögerung bei fester Lage gegen Zahl und Lage der Brüche): "
                    f"der Alarm sagt früh, dass sich etwas geändert hat, der Rückblick sagt, wo und wie oft. Der CUSUM ist einseitig (nur nach oben), der Rückblick unterscheidet die Richtung nicht.")
