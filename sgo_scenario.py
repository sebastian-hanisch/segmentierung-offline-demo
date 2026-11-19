"""Szenarien und Live-Läufe. Alle Zufallszahlen kommen aus dem SplitMix64-Strom der Demo (`sgo_normal.Stream`); Läufe sind Zeilen: die ersten R Zeilen bleiben gleich,
wenn R wächst. Zähleinheit überall: Läufe (eine simulierte Reihe = ein Lauf)."""

import math
import re
import time
from dataclasses import dataclass, field

import numpy as np

import sgo_constants as C
import sgo_online as ON
import sgo_segmentation as S
from sgo_normal import Stream


class LiveLimitError(Exception):
    """Der Live-Lauf überschreitet die Obergrenze (Zufallszahlen oder Läufe)."""


class SeriesError(ValueError):
    """Die eigene Reihe ist unbrauchbar (leer, nicht numerisch, zu lang, NaN oder unendlich)."""


# --------------------------------------------------------------------------------------------------------------------------------- Szenarien
def true_breaks(scen_key: str, n: int) -> list[int]:
    """Wahre Brüche des Szenarios bei Länge n (Anteile von n, gerundet; bei n = 200 die Lagen der Vormessung)."""
    return [int(round(f * n)) for f in C.SCEN_FRAC[scen_key]]


def mean_profile(scen_key: str, n: int, delta: float) -> np.ndarray:
    """Mittelwert je Beobachtung: ein Bruch (+δ ab b1), Treppe (+δ ab b1 und nochmals +δ ab b2), Puls (+δ von b1 bis b2), Zickzack (+δ von b1 bis b2 und +δ ab b3)."""
    b = true_breaks(scen_key, n)
    mu = np.zeros(n)
    if scen_key == "ein":
        mu[b[0]:] += delta
    elif scen_key == "treppe":
        mu[b[0]:] += delta
        mu[b[1]:] += delta
    elif scen_key == "puls":
        mu[b[0]:b[1]] += delta
    elif scen_key == "zickzack":
        mu[b[0]:b[1]] += delta
        mu[b[2]:] += delta
    return mu


def draw_series(scen_key: str, n: int, delta: float, R: int, seed: int):
    """R Reihen der Länge n: Normalrauschen (Streuung 1) plus Mittelwertprofil. Rückgabe (X, wahre Brüche)."""
    Z = Stream(seed).normal((R, n))
    return Z + mean_profile(scen_key, n, delta)[None, :], true_breaks(scen_key, n)


def parse_series(text: str) -> list[float]:
    """Eigene Reihe: Zahlen getrennt durch Leerzeichen, Semikolon oder Zeilenumbruch; das Komma ist das **Dezimalkomma**."""
    if text is None or not text.strip():
        raise SeriesError("Die Reihe ist leer: bitte Zahlen eintragen (getrennt durch Leerzeichen, Semikolon oder Zeilenumbruch).")
    toks = [t for t in re.split(r"[\s;]+", text.strip()) if t]
    if len(toks) > C.OWN_MAX:
        raise SeriesError(f"Die Reihe hat {len(toks)} Werte, erlaubt sind höchstens {C.OWN_MAX}.")
    vals = []
    for t in toks:
        try:
            v = float(t.replace(",", "."))
        except ValueError:
            raise SeriesError(f"„{t}“ ist keine Zahl. Zahlen mit Leerzeichen, Semikolon oder Zeilenumbruch trennen; das Komma gilt als Dezimalkomma.") from None
        if not math.isfinite(v):
            raise SeriesError(f"„{t}“ ist keine endliche Zahl (NaN und unendliche Werte sind nicht erlaubt).")
        vals.append(v)
    return vals


# --------------------------------------------------------------------------------------------------------------------------------- Experiment 1
@dataclass
class Exp1Result:
    x: list
    true: list | None                  # None bei eigener Reihe
    n: int
    beta: float
    minlen: int
    method: str
    breaks: list                       # Brüche des gewählten Verfahrens
    objective: float                   # Zielwert des gewählten Verfahrens
    dp_breaks: list
    dp_objective: float
    f_table: list                      # F(0..n) der optimalen Zerlegung
    pelt: S.PeltResult
    pelt_delayed: S.PeltResult
    bin_breaks: list
    bin_steps: list
    bin_objective: float
    objectives: dict                   # Verfahren -> Zielwert
    obj_none: float                    # Zielwert ohne Bruch
    segments: list                     # (lo, hi, Mittel) des gewählten Verfahrens
    enumeration: S.Enumeration | None
    no_break_possible: bool
    seconds: float = 0.0
    notes: list = field(default_factory=list)


def run_exp1(x, true, beta: float, minlen: int, method: str) -> Exp1Result:
    """Eine Reihe, alle vier Verfahren (für den Vergleich der Zielwerte), das gewählte wird gezeigt; bei n <= 14 die Aufzählung als Zweitentscheider.
    Weicht die optimale Zerlegung von der Aufzählung ab, wird ein Fehler gemeldet statt einer Zahl."""
    t0 = time.perf_counter()
    x = [float(v) for v in x]
    n = len(x)
    if n == 0:
        raise SeriesError("Die Reihe ist leer.")
    if n < minlen:
        raise SeriesError(f"Die Reihe hat {n} Werte, die Mindestlänge ist {minlen}: keine Zerlegung möglich.")
    arr = np.array(x)
    dp = S.dp_batch(arr, [beta], minlen)
    dp_breaks = dp.breaks(0, 0)
    dp_obj = float(dp.F[0, 0])
    pe = S.pelt(arr, beta, minlen, trace=n <= 40)
    pd_ = S.pelt(arr, beta, minlen, delayed=True, trace=n <= 40)
    bb, steps = S.binseg(arr, beta, minlen)
    bobj = S.objective(arr, bb, beta)
    enum = None
    if n <= S.ENUM_LIMIT:
        enum = S.enumerate_best(arr, beta, minlen)
        if abs(enum.best - dp_obj) > 1e-9 or abs(S.objective(arr, dp_breaks, beta) - enum.best) > 1e-9:
            raise AssertionError(f"Die optimale Zerlegung weicht von der Aufzählung ab (Zielwert {dp_obj} gegen {enum.best}): Fehler, keine Zahl.")
    chosen = {C.M_OPT: dp_breaks, C.M_PELT: pe.breaks, C.M_PELT_V: pd_.breaks, C.M_BIN: bb}[method]
    objs = {C.M_OPT: dp_obj, C.M_PELT: pe.objective, C.M_PELT_V: pd_.objective, C.M_BIN: bobj}
    edges = [0] + list(chosen) + [n]
    segs = [(lo, hi, float(arr[lo:hi].mean())) for lo, hi in zip(edges[:-1], edges[1:])]
    return Exp1Result(x, true, n, beta, minlen, method, list(chosen), objs[method], dp_breaks, dp_obj, [float(v) for v in dp.table[0, 0]], pe, pd_, bb, steps, bobj, objs, S.objective(arr, [], beta),
                      segs, enum, n < 2 * minlen, time.perf_counter() - t0)


# --------------------------------------------------------------------------------------------------------------------------------- Experiment 2
@dataclass
class Exp2Result:
    scen: str
    n: int
    delta: float
    minlen: int
    R: int
    sigma_mode: str
    true: list
    c_grid: tuple
    p_right: list                      # Anteil Läufe mit richtiger Bruchzahl je c
    p_any: list                        # Anteil Läufe mit mindestens einem Bruch je c
    mean_count: list
    p_loc: list                        # Anteil Läufe mit richtiger Zahl und allen Lagen innerhalb ±5 je c
    loc_error: list                    # mittlerer Lagefehler bei richtiger Zahl je c (None, wenn keine)
    window: list                       # c-Werte mit Anteil >= 90 %
    f1: dict                           # M -> mittleres F1 (c = 2), None ohne wahren Bruch
    f1_zero: dict                      # M -> Anteil Läufe mit F1 = 0
    sigma_hat_mean: float | None
    seconds: float = 0.0


def run_exp2(scen_key: str, n: int, delta: float, minlen: int, R: int, sigma_mode: str, seed: int) -> Exp2Result:
    if R > C.R2_MAX or R < 1:
        raise LiveLimitError(f"R = {R} liegt außerhalb von 1 bis {C.R2_MAX}")
    t0 = time.perf_counter()
    X, true = draw_series(scen_key, n, delta, R, seed)
    sig = None
    if sigma_mode == C.SIGMA_ROBUST:
        sig = S.robust_sigma(X)
        Xe = X / sig[:, None]
    else:
        Xe = X
    betas = np.array([C.beta_of(c, n) for c in C.C_GRID])
    res = S.dp_batch(Xe, betas, minlen)
    P = len(C.C_GRID)
    p_right, p_any, mean_count, p_loc, loc_err = [], [], [], [], []
    cnt = res.counts()
    f1 = {m: [] for m in C.TOL_M}
    j2 = C.C_GRID.index(2.0)
    for j in range(P):
        right = loc_ok = 0
        errs = []
        for r in range(R):
            est = res.breaks(r, j)
            if len(est) == len(true):
                right += 1
                e = S.location_errors(est, true)
                errs += e
                loc_ok += all(v <= C.LOC_TOL for v in e)
        p_right.append(right / R)
        p_any.append(float((cnt[:, j] > 0).mean()))
        mean_count.append(float(cnt[:, j].mean()))
        p_loc.append(loc_ok / R)
        loc_err.append(float(np.mean(errs)) if errs else None)
    f1z = {}
    if true:
        for m in C.TOL_M:
            vals = [S.f1_tolerance(res.breaks(r, j2), true, m)[2] for r in range(R)]
            f1[m] = float(np.mean(vals))
            f1z[m] = float(np.mean([v == 0.0 for v in vals]))
    else:
        f1 = {m: None for m in C.TOL_M}
        f1z = {m: None for m in C.TOL_M}
    window = [c for c, v in zip(C.C_GRID, p_right) if v >= C.WINDOW_SHARE]
    return Exp2Result(scen_key, n, delta, minlen, R, sigma_mode, true, C.C_GRID, p_right, p_any, mean_count, p_loc, loc_err, window, f1, f1z,
                      float(sig.mean()) if sig is not None else None, time.perf_counter() - t0)


# --------------------------------------------------------------------------------------------------------------------------------- Experiment 3
@dataclass
class Exp3Result:
    n: int
    c: float
    beta: float
    minlen: int
    R: int
    p_any: float                       # Anteil Läufe mit mindestens einem Fehlbruch (Simulation)
    se: float
    mean_count: float
    exact_p_any: float | None          # nur n = 2, 3 (exakt)
    seconds: float = 0.0


def run_exp3(n: int, c: float, R: int, seed: int) -> Exp3Result:
    import sgo_exact as EX
    if n * R > C.LIVE_LIMIT_NULL:
        raise LiveLimitError(f"n · R = {n * R:,} übersteigt die Obergrenze {C.LIVE_LIMIT_NULL:,} Zufallszahlen".replace(",", "."))
    t0 = time.perf_counter()
    minlen = C.null_minlen(n)
    beta = C.beta_of(c, n)
    X = Stream(seed).normal((R, n))
    res = S.dp_batch(X, [beta], minlen)
    cnt = res.counts()[:, 0]
    p = float((cnt > 0).mean())
    se = max(math.sqrt(max(p * (1 - p), 0.0) / R), 0.5 / R)
    exact = None
    if n == 2:
        exact = EX.p_break_n2(beta)
    elif n == 3:
        exact = 1.0 - EX.p_nobreak_n3(beta)
    return Exp3Result(n, c, beta, minlen, R, p, se, float(cnt.mean()), exact, time.perf_counter() - t0)


# --------------------------------------------------------------------------------------------------------------------------------- Experiment 4
@dataclass
class Exp4Result:
    scen: str
    c: float
    minlen: int
    R: int
    n: int
    pelt_diff: int                     # Läufe, in denen PELT einen größeren Zielwert hat als die optimale Zerlegung
    delayed_diff: int
    bin_other: int                     # Läufe, in denen die binäre Segmentierung eine andere Bruchmenge liefert
    bin_worse: int                     # davon mit größerem Zielwert
    bin_better: int                    # muss 0 sein (nie besser als die optimale Zerlegung)
    ev_dp: int
    ev_pelt: int
    ev_delayed: int
    seconds: float = 0.0


def run_exp4(scen_key: str, minlen: int, c: float, R: int, seed: int, delta: float = 1.0) -> Exp4Result:
    if R > C.R4_MAX or R < 1:
        raise LiveLimitError(f"R = {R} liegt außerhalb von 1 bis {C.R4_MAX}")
    t0 = time.perf_counter()
    n = C.N4
    X, _ = draw_series(scen_key, n, delta, R, seed)
    beta = C.beta_of(c, n)
    res = S.dp_batch(X, [beta], minlen)
    pd_n = dd_n = bo = bw = bb_ = ev_p = ev_d = 0
    for r in range(R):
        dpv = float(res.F[r, 0])
        pe = S.pelt(X[r], beta, minlen)
        de = S.pelt(X[r], beta, minlen, delayed=True)
        bs, _ = S.binseg(X[r], beta, minlen)
        pd_n += abs(pe.objective - dpv) > 1e-8
        dd_n += abs(de.objective - dpv) > 1e-8 or de.breaks != res.breaks(r, 0)
        ev_p += pe.evaluations
        ev_d += de.evaluations
        bov = S.objective(X[r], bs, beta)
        bo += bs != res.breaks(r, 0)
        bw += bov > dpv + 1e-9
        bb_ += bov < dpv - 1e-9
    return Exp4Result(scen_key, c, minlen, R, n, int(pd_n), int(dd_n), int(bo), int(bw), int(bb_), S.dp_evaluations(n, minlen) * R, int(ev_p), int(ev_d), time.perf_counter() - t0)


# --------------------------------------------------------------------------------------------------------------------------------- Experiment 5
@dataclass
class Exp5Result:
    delta: float
    nu: int
    R: int
    n: int
    h: float
    x0: list                           # erste Reihe
    C0: list                           # CUSUM der ersten Reihe
    alarm0: int | None
    dp_breaks0: list
    p_early_sim: float                 # Alarm vor oder bei ν (Simulation)
    p_early_exact: float
    delay_sim: float | None            # Verzögerung der Läufe mit Alarm nach ν (gegeben kein Alarm vor ν)
    delay_sim_se: float | None
    delay_exact: float
    delay_sf: list                     # exakt: P(Verzögerung > d), d = 0 …
    delay_values: list                 # simulierte Verzögerungen (Alarm nach ν)
    online_lage_mean: float | None     # mittlerer Betrag des Lagefehlers (nur Läufe mit Alarm nach ν)
    online_lage_in5: float | None
    online_n: int                      # Zahl dieser Läufe
    off_one: float                     # Anteil Läufe mit genau einem Bruch (alle Läufe)
    off_lage_mean: float | None        # mittlerer Lagefehler bei genau einem Bruch (alle Läufe)
    off_lage_in5: float | None
    same_n: int                        # Läufe ohne Alarm vor ν und mit Alarm nach ν (gleiche Läufe für beide Seiten)
    same_off_one: float | None
    same_off_lage_mean: float | None
    same_online_lage_mean: float | None
    seconds: float = 0.0


def run_exp5(delta: float, nu: int, R: int, seed: int, h: float, limit: bool = True) -> Exp5Result:
    if (R > C.R5_MAX and limit) or R < 1:
        raise LiveLimitError(f"R = {R} liegt außerhalb von 1 bis {C.R5_MAX}")
    n = C.N5
    if not 1 <= nu <= n - 1:
        raise ValueError("ν muss zwischen 1 und n − 1 liegen")
    t0 = time.perf_counter()
    Z = Stream(seed).normal((R, n))
    X = Z.copy()
    X[:, nu:] += delta
    Cm, T = ON.cusum_paths(X, C.CUSUM_K, h)
    beta = C.beta_of(C.BREAK_PENALTY_C5, n)
    res = S.dp_batch(X, [beta], C.MINLEN5)
    Q0 = ON.chain_matrix(0.0, C.CUSUM_K, h)
    Q1 = ON.chain_matrix(delta, C.CUSUM_K, h)
    mean_ex, sf, p_early_ex = ON.delay_exact(Q0, Q1, nu)
    early = T <= nu
    late = (~early) & (T <= n)
    dl = (T[late] - nu).astype(float)
    est_on = np.array([ON.last_zero_before(Cm[i], int(T[i])) for i in np.flatnonzero(late)], dtype=int)
    err_on = np.abs(est_on - nu)
    cnt = res.counts()[:, 0]
    one = cnt == 1
    off_err = np.array([abs(res.breaks(i, 0)[0] - nu) for i in np.flatnonzero(one)], dtype=float)
    same = late  # kein Alarm vor ν und Alarm nach ν
    same_idx = np.flatnonzero(same)
    same_one = [i for i in same_idx if cnt[i] == 1]
    same_off_err = np.array([abs(res.breaks(i, 0)[0] - nu) for i in same_one], dtype=float)
    return Exp5Result(
        delta, nu, R, n, h, [float(v) for v in X[0]], [float(v) for v in Cm[0]], int(T[0]) if T[0] <= n else None, res.breaks(0, 0),
        float(early.mean()), p_early_ex,
        float(dl.mean()) if dl.size else None, float(dl.std(ddof=1) / math.sqrt(dl.size)) if dl.size > 1 else None, mean_ex, sf, [float(v) for v in dl],
        float(err_on.mean()) if err_on.size else None, float((err_on <= C.LOC_TOL).mean()) if err_on.size else None, int(err_on.size),
        float(one.mean()), float(off_err.mean()) if off_err.size else None, float((off_err <= C.LOC_TOL).mean()) if off_err.size else None,
        int(same_idx.size), (len(same_one) / same_idx.size) if same_idx.size else None, float(same_off_err.mean()) if same_off_err.size else None,
        float(err_on.mean()) if err_on.size else None, time.perf_counter() - t0)
