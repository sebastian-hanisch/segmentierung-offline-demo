"""Plotly-Abbildungen (alle Achsen fest, fixedrange)."""

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sgo_constants as C


def _layout(fig, title, xtitle=None, ytitle=None, height=380, logx=False, xrange=None, yrange=None):
    fig.update_layout(title=dict(text=title, font=dict(size=14), x=0.01, xanchor="left", y=0.97, yanchor="top"), height=height, margin=dict(l=55, r=15, t=85, b=50),
                      legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0), hovermode="closest")
    fig.update_xaxes(title=xtitle, fixedrange=True, type="log" if logx else "linear", range=xrange)
    fig.update_yaxes(title=ytitle, fixedrange=True, range=yrange)
    return fig


def build_series(x, true, found, segments, title="Die Reihe, die wahren und die gefundenen Brüche"):
    """Die Reihe als Punkte und Linie, Segmentmittel (grün), wahre Brüche (grau gestrichelt, nur bei Szenario), gefundene Brüche (rot). Ein Bruch bei s liegt zwischen
    Beobachtung s und s + 1 (x-Wert s + 0,5)."""
    n = len(x)
    t = list(range(1, n + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=list(x), mode="lines+markers", name="Reihe", line=dict(color=C.COLOR_SERIES, width=1.2), marker=dict(size=4 if n <= 100 else 3)))
    for i, (lo, hi, m) in enumerate(segments):
        fig.add_trace(go.Scatter(x=[lo + 0.5, hi + 0.5], y=[m, m], mode="lines", name="Mittel je Strecke", showlegend=i == 0, line=dict(color=C.COLOR_MEAN, width=3), hoverinfo="skip"))
    ymin, ymax = float(np.min(x)), float(np.max(x))
    pad = 0.08 * (ymax - ymin) + 0.3
    y0, y1 = ymin - pad, ymax + pad
    for i, s in enumerate(true or []):
        fig.add_trace(go.Scatter(x=[s + 0.5, s + 0.5], y=[y0, y1], mode="lines", name="wahrer Bruch", showlegend=i == 0, line=dict(color=C.COLOR_TRUE, dash="dash", width=2), hoverinfo="skip"))
    for i, s in enumerate(found):
        fig.add_trace(go.Scatter(x=[s + 0.5, s + 0.5], y=[y0, y1], mode="lines", name="gefundener Bruch", showlegend=i == 0, line=dict(color=C.COLOR_FOUND, width=2), hoverinfo="skip"))
    return _layout(fig, title, "Beobachtung", "Wert", height=360, xrange=[0.5, n + 0.5], yrange=[y0, y1])


def build_window(c_grid, frozen, live, live_R, title="Anteil richtige Bruchzahl gegen die Strafkonstante c"):
    """Anteil Läufe mit richtiger Bruchzahl je c. frozen: {mean, lo, hi} (drei Zufalls-Seeds, R = 1000 je Seed) oder None; live: Anteile (R Läufe, Band ± 1,96 Standardfehler)."""
    fig = go.Figure()
    if frozen is not None:
        fig.add_trace(go.Scatter(x=list(c_grid) + list(c_grid)[::-1], y=[100 * v for v in frozen["hi"]] + [100 * v for v in frozen["lo"]][::-1], fill="toself", mode="lines",
                                 line=dict(width=0), fillcolor="rgba(90,90,90,0.18)", name="gefroren: Spanne der drei Zufalls-Seeds", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=list(c_grid), y=[100 * v for v in frozen["mean"]], mode="lines+markers", name="gefroren: Mittel (R = 1000 je Zufalls-Seed)",
                                 line=dict(color=C.COLOR_FROZEN, dash="dot", width=2), marker=dict(size=5)))
    if live is not None:
        se = [1.96 * 100 * max(math.sqrt(max(p * (1 - p), 0.0) / live_R), 0.5 / live_R) for p in live]
        fig.add_trace(go.Scatter(x=list(c_grid), y=[100 * v for v in live], mode="lines+markers", name=f"live ({live_R} Läufe, Band ± 1,96 Standardfehler)",
                                 line=dict(color=C.COLOR_LIVE, width=2.5), marker=dict(size=7), error_y=dict(type="data", array=se, visible=True, color=C.COLOR_LIVE, thickness=1.2)))
    fig.add_hline(y=100 * C.WINDOW_SHARE, line=dict(color="rgba(90,90,90,0.9)", dash="dash", width=1.5))
    fig.add_annotation(x=math.log10(c_grid[-1]), y=100 * C.WINDOW_SHARE, text="90 %", showarrow=False, yanchor="bottom", xanchor="right", font=dict(size=11))
    _layout(fig, title, "Strafkonstante c (β = c · ln n, logarithmisch)", "Läufe mit richtiger Bruchzahl in %", height=400, logx=True,
            xrange=[math.log10(c_grid[0]) - 0.05, math.log10(c_grid[-1]) + 0.05], yrange=[-3, 103])
    fig.update_xaxes(tickvals=list(c_grid), ticktext=[str(c).replace(".", ",") for c in c_grid])
    return fig


def build_f1(ms, frozen, live, half_distance=None):
    """F1 mit Toleranz M gegen M (logarithmisch). Gestrichelt: M = halber Abstand wahrer Brüche (eigene Verschärfung; rechts davon sagt F1 nichts)."""
    fig = go.Figure()
    if frozen is not None:
        fig.add_trace(go.Scatter(x=list(ms), y=[frozen[str(m)] for m in ms], mode="lines+markers", name="gefroren (3 Zufalls-Seeds, c = 2)", line=dict(color=C.COLOR_FROZEN, dash="dot", width=2)))
    if live is not None:
        fig.add_trace(go.Scatter(x=list(ms), y=[live[m] for m in ms], mode="lines+markers", name="live (c = 2)", line=dict(color=C.COLOR_LIVE, width=2.5)))
    if half_distance:
        fig.add_vline(x=math.log10(half_distance), line=dict(color="rgba(90,90,90,0.9)", dash="dash", width=1.5))
        fig.add_annotation(x=math.log10(half_distance), y=1.0, text=f"M = {half_distance:g}: halber Abstand wahrer Brüche", showarrow=False, yanchor="bottom", xanchor="right", font=dict(size=11))
    _layout(fig, "Mittleres F1 gegen die Toleranz M", "Toleranz M in Beobachtungen (logarithmisch)", "mittleres F1", height=360, logx=True, xrange=[math.log10(0.9), math.log10(max(ms) * 1.1)], yrange=[0, 1.08])
    fig.update_xaxes(tickvals=list(ms), ticktext=[str(m) for m in ms])
    return fig


def build_null_n(rows):
    """Anteil Läufe mit mindestens einem Fehlbruch gegen die Reihenlänge n (gefroren, R = 1000 je Punkt), je c. rows: {n: {"1.0": {...}, ...}}."""
    fig = go.Figure()
    ns = sorted(int(k) for k in rows)
    for c, col in (("1.0", "#c0392b"), ("2.0", "#2F6B65"), ("3.0", "#4c78a8")):
        fig.add_trace(go.Scatter(x=ns, y=[100 * rows[str(n)][c]["p_fehlbruch"] for n in ns], mode="lines+markers", name=f"c = {c[0]}", line=dict(color=col, width=2), marker=dict(size=6)))
    _layout(fig, "Nullreihe: Läufe mit mindestens einem Fehlbruch gegen n (gefroren, R = 1000)", "Reihenlänge n (logarithmisch)", "Läufe mit Fehlbruch in %", height=360, logx=True,
            xrange=[math.log10(ns[0]) - 0.05, math.log10(ns[-1]) + 0.05], yrange=[-2, 82])
    fig.update_xaxes(tickvals=ns, ticktext=[str(n) for n in ns])
    return fig


def build_online(x, C0, alarm, nu, h, dp_breaks):
    """Oben die Reihe mit wahrem Bruch ν (grau) und dem Bruch der optimalen Zerlegung (rot); unten der CUSUM mit Schwelle h und Alarmzeit (orange)."""
    n = len(x)
    t = list(range(1, n + 1))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Die Reihe: wahrer Bruch und Bruch der optimalen Zerlegung (Rückblick)", "Der CUSUM C_t: Alarm bei C_t ≥ h (online)"))
    fig.add_trace(go.Scatter(x=t, y=list(x), mode="lines+markers", name="Reihe", line=dict(color=C.COLOR_SERIES, width=1.2), marker=dict(size=3)), row=1, col=1)
    lo, hi = float(min(x)) - 0.3, float(max(x)) + 0.3
    fig.add_trace(go.Scatter(x=[nu + 0.5] * 2, y=[lo, hi], mode="lines", name="wahrer Bruch ν", line=dict(color=C.COLOR_TRUE, dash="dash", width=2), hoverinfo="skip"), row=1, col=1)
    for i, s in enumerate(dp_breaks):
        fig.add_trace(go.Scatter(x=[s + 0.5] * 2, y=[lo, hi], mode="lines", name="Rückblick: Bruch", showlegend=i == 0, line=dict(color=C.COLOR_OFFLINE, width=2.5), hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(0, n + 1)), y=list(C0), mode="lines", name="C_t", line=dict(color=C.COLOR_ONLINE, width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=[0, n], y=[h, h], mode="lines", name=f"Schwelle h = {h:.4f}".replace(".", ","), line=dict(color=C.COLOR_TRUE, dash="dash", width=1.5), hoverinfo="skip"), row=2, col=1)
    if alarm is not None:
        fig.add_trace(go.Scatter(x=[alarm, alarm], y=[0, max(h, max(C0)) * 1.05], mode="lines", name=f"Alarm bei t = {alarm}", line=dict(color=C.COLOR_ONLINE, width=2.5, dash="dot"), hoverinfo="skip"), row=2, col=1)
    fig.add_trace(go.Scatter(x=[nu + 0.5] * 2, y=[0, max(h, max(C0)) * 1.05], mode="lines", name="wahrer Bruch ν", showlegend=False, line=dict(color=C.COLOR_TRUE, dash="dash", width=2), hoverinfo="skip"), row=2, col=1)
    fig.update_layout(height=640, margin=dict(l=55, r=15, t=70, b=130), legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0), hovermode="closest")
    fig.update_xaxes(fixedrange=True, range=[0, n + 1])
    fig.update_yaxes(fixedrange=True)
    fig.update_xaxes(title="Beobachtung", row=2, col=1)
    return fig


def build_delay(sf_exact, delays):
    """Verzögerung (Beobachtungen nach ν bis zum Alarm, gegeben kein Alarm vor ν): exakte Wahrscheinlichkeiten aus der Kette (Linie) und die simulierten Läufe (Balken)."""
    sf = list(sf_exact)
    pmf = [sf[d] - sf[d + 1] for d in range(len(sf) - 1)]      # P(Verzögerung = d + 1)
    ds = list(range(1, len(pmf) + 1))
    fig = go.Figure()
    if len(delays):
        vals = np.asarray(delays, int)
        cnt = np.bincount(np.clip(vals, 0, len(ds)), minlength=len(ds) + 1)[1:len(ds) + 1]
        fig.add_trace(go.Bar(x=ds, y=list(cnt / len(vals)), name=f"Simulation ({len(vals)} Läufe mit Alarm nach ν)", marker=dict(color="rgba(214,137,16,0.55)")))
    fig.add_trace(go.Scatter(x=ds, y=pmf, mode="lines", name="exakt (Kette)", line=dict(color="#555", width=2)))
    return _layout(fig, "Verteilung der Verzögerung des CUSUM (Beobachtungen nach ν bis zum Alarm)", "Beobachtungen nach ν", "Wahrscheinlichkeit", height=340, xrange=[0.5, 40.5])
