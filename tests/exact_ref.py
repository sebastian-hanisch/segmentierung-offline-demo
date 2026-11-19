"""Exakte Referenz für die Tests (Testhilfe, kein Teil der App): Dynamische Programmierung, Aufzählung aller Zerlegungen, PELT, binäre Segmentierung und der CUSUM
mit `fractions.Fraction` auf ganzzahligen Reihen, damit „gleich“ wirklich gleich heißt; dazu die Nullverteilung (n = 2 und n = 3) auf zwei unabhängigen Wegen.

Herkunft: die Funktionen sind aus `handbeispiele_07.py` der Planung abgeschrieben (Plan 07, Z3: sie wandern aus dem Scratch-Lauf nach `tests/`); sie rufen
**keinen** Code der App auf. Die Demo-Verfahren in `sgo_segmentation.py` sind eine andere Implementierung (Gleitkomma, vektorisiert, zentriert).
Konventionen: Reihe x[0..n-1]; Bruch bei s heißt: x[0..s-1] liegen vor der Änderung. Kosten einer Strecke (s, t] = Summe der Abweichungsquadrate um das Mittel.
"""
import math
import random  # noqa: F401
from fractions import Fraction as Fr

import numpy as np
from scipy import integrate
from scipy.stats import chi2


# ------------------------------------------------------------------ exakte Verfahren (Fraction)
def prefix_sums(x):
    s1, s2 = [0], [0]
    for v in x:
        s1.append(s1[-1] + v)
        s2.append(s2[-1] + v * v)
    return s1, s2


def seg_cost(sums, s, t):
    """Kosten der Strecke x[s..t-1]: Summe der Quadrate minus Quadrat der Summe durch Länge."""
    s1, s2 = sums
    return (s2[t] - s2[s]) - Fr((s1[t] - s1[s]) ** 2, t - s)


def objective_exact(x, breaks, beta):
    sums = prefix_sums(x)
    b = [0] + list(breaks) + [len(x)]
    return sum(seg_cost(sums, s, t) for s, t in zip(b[:-1], b[1:])) + beta * len(breaks)


def dp_exact(x, beta, minlen):
    """Optimale Zerlegung (Optimal Partitioning) ohne Abkürzung. Rückgabe (Zielwert, Brüche, F-Tabelle); F[0] = -beta."""
    n = len(x)
    sums = prefix_sums(x)
    F = [None] * (n + 1)
    last = [0] * (n + 1)
    F[0] = -Fr(beta)
    for t in range(minlen, n + 1):
        best = None
        for s in range(0, t - minlen + 1):
            if F[s] is None:
                continue
            v = F[s] + seg_cost(sums, s, t) + beta
            if best is None or v < best[0]:
                best = (v, s)
        if best is not None:
            F[t], last[t] = best
    out, t = [], n
    while t > 0:
        s = last[t]
        if s > 0:
            out.append(s)
        t = s
    return F[n], sorted(out), F


def dp_float(x, beta, minlen):
    """Optimal Partitioning mit Gleitkomma (für die Simulation der Nullreihe); Rückgabe (Zielwert, Brüche)."""
    n = len(x)
    s1 = np.concatenate([[0.0], np.cumsum(x)])
    s2 = np.concatenate([[0.0], np.cumsum(np.square(x))])
    F = [None] * (n + 1)
    last = [0] * (n + 1)
    F[0] = -beta
    for t in range(minlen, n + 1):
        best = None
        for s in range(0, t - minlen + 1):
            if F[s] is None:
                continue
            v = F[s] + (s2[t] - s2[s]) - (s1[t] - s1[s]) ** 2 / (t - s) + beta
            if best is None or v < best[0]:
                best = (v, s)
        F[t], last[t] = best
    out, t = [], n
    while t > 0:
        s = last[t]
        if s > 0:
            out.append(s)
        t = s
    return F[n], sorted(out)


def compositions(n, minlen):
    """Alle Zerlegungen von 0..n in Strecken der Länge >= minlen, als Listen der Bruchstellen."""
    res = []

    def rec(start, chosen):
        if n - start >= minlen:
            res.append(list(chosen))  # Rest als letzte Strecke
        for p in range(start + minlen, n - minlen + 1):
            rec(p, chosen + [p])

    if n >= minlen:
        rec(0, [])
    return res


def brute_exact(x, beta, minlen):
    """Aufzählung aller Zerlegungen; Rückgabe (kleinster Zielwert, Menge aller Zerlegungen mit diesem Wert)."""
    best, arg = None, []
    for br in compositions(len(x), minlen):
        v = objective_exact(x, br, beta)
        if best is None or v < best:
            best, arg = v, [br]
        elif v == best:
            arg.append(br)
    return best, arg


def pelt_exact(x, beta, minlen, delayed=False, trace=False):
    """PELT mit K = 0 (Beschneidungsregel: Kandidat s fällt, wenn F(s) + C(s,t) >= F(t)). Mit delayed=True werden als beschneidbar
    markierte Kandidaten erst nach `minlen - 1` weiteren Zeitpunkten gestrichen (eigener Vorschlag, hier gegen DP geprüft).
    Rückgabe (Zielwert, Brüche) bzw. mit trace=True zusätzlich die Kandidatenliste nach jedem Zeitpunkt."""
    n = len(x)
    sums = prefix_sums(x)
    F = [None] * (n + 1)
    last = [0] * (n + 1)
    F[0] = -Fr(beta)
    cand, flagged, log = [], {}, {}
    for t in range(minlen, n + 1):
        s_new = t - minlen
        if s_new == 0 or s_new >= minlen:
            cand.append(s_new)
        vals = {s: F[s] + seg_cost(sums, s, t) for s in cand}
        sbest = min(vals, key=lambda s: vals[s])
        F[t] = vals[sbest] + beta
        last[t] = sbest
        fl = [s for s in cand if vals[s] >= F[t]]
        if not delayed:
            cand = [s for s in cand if s not in fl]
        else:
            flagged[t] = fl
            tt = t - minlen + 1
            if tt in flagged:
                rm = set(flagged.pop(tt))
                cand = [s for s in cand if s not in rm]
        log[t] = list(cand)
    out, t = [], n
    while t > 0:
        s = last[t]
        if s > 0:
            out.append(s)
        t = s
    res = (F[n], sorted(out))
    return (res + (log,)) if trace else res


def gain_exact(x, lo, hi, tau):
    sums = prefix_sums(x)
    return seg_cost(sums, lo, hi) - seg_cost(sums, lo, tau) - seg_cost(sums, tau, hi)


def binseg_exact(x, beta, minlen):
    """Binäre Segmentierung: auf jeder Strecke die Teilung mit dem größten Kostengewinn; teilen, wenn der Gewinn größer als beta ist."""
    res, stack = [], [(0, len(x))]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2 * minlen:
            continue
        taus = list(range(lo + minlen, hi - minlen + 1))
        gains = [gain_exact(x, lo, hi, t) for t in taus]
        j = max(range(len(taus)), key=lambda i: (gains[i], -i))  # erstes Maximum
        if gains[j] > beta:
            res.append(taus[j])
            stack.append((lo, taus[j]))
            stack.append((taus[j], hi))
    return sorted(res)


def cusum_up_trace(x, k, h):
    """Einseitiger oberer CUSUM C_t = max(0, C_{t-1} + x_t - k); Alarm beim ersten t mit C_t >= h (Konvention der Linie, wie Plan 08 und die V1-Bibliothek).
    Rückgabe (C-Folge, Alarmzeit oder None)."""
    c, cs, alarm = Fr(0), [], None
    for t, v in enumerate(x, 1):
        c = max(Fr(0), c + v - k)
        cs.append(c)
        if alarm is None and c >= h:
            alarm = t
    return cs, alarm


def f1_tolerance(est, true, m):
    """Genauigkeit und Ausbeute mit Toleranz m (ein Bruch gilt als gefunden, wenn ein anderer näher als m Beobachtungen liegt)."""
    hit_est = sum(1 for e in est if any(abs(e - t) < m for t in true))
    hit_true = sum(1 for t in true if any(abs(e - t) < m for e in est))
    p = Fr(hit_est, len(est)) if est else None
    r = Fr(hit_true, len(true)) if true else None
    if not true:
        f = None  # ohne wahren Bruch ist F1 nicht definiert
    elif p is None or p + r == 0:
        f = Fr(0)  # nichts gefunden oder nichts getroffen: F1 = 0 (Festlegung dieses Plans)
    else:
        f = 2 * p * r / (p + r)
    return p, r, f


# ------------------------------------------------------------------ Nullverteilung (Gauss, bekannte Streuung 1)
def p_break_n2(beta):
    """n = 2, Mindestlänge 1: der Gewinn der Teilung ist (x1 - x2)^2 / 2, also chi2 mit einem Freiheitsgrad; Fehlbruch, wenn er beta übersteigt."""
    return float(chi2.sf(beta, 1))


def p_nobreak_n3_polar(beta):
    """n = 3, Mindestlänge 1: kein Bruch genau dann, wenn beide Teilungsgewinne <= beta und die Gesamtkosten <= 2 beta sind.
    Die Abweichungen von den Mittel liegen in einer Ebene, dort ist der Abstand r^2 ~ chi2(2), die beiden Teilungsrichtungen schneiden sich
    unter 60 Grad. Einfaches Integral über den Winkel."""
    a = math.sqrt(beta)
    th = math.pi / 3

    def inner(phi):
        r2 = 2 * beta
        for ang in (phi, phi - th):
            c = abs(math.cos(ang))
            if c > 1e-14:
                r2 = min(r2, a * a / (c * c))
        return 1 - math.exp(-r2 / 2)

    val, _ = integrate.quad(inner, 0, 2 * math.pi, limit=400, points=[math.pi / 2, math.pi / 2 + th, 3 * math.pi / 2, 3 * math.pi / 2 + th])
    return val / (2 * math.pi)


def p_nobreak_n3_grid(beta):
    """Zweiter, unabhängiger Weg für dieselbe Größe: zweidimensionales Integral über (z1, z2), den standardisierten Teilungsgewinnen
    (Korrelation 1/2), Gebiet |z1|, |z2| <= sqrt(beta) und (4/3)(z1^2 + z2^2 - z1 z2) <= 2 beta."""
    a = math.sqrt(beta)
    s = 1.0 / (1 - 0.25)

    def dens(z2, z1):
        q = (z1 * z1 + z2 * z2 - 2 * 0.5 * z1 * z2) * s
        return math.exp(-q / 2) / (2 * math.pi * math.sqrt(1 - 0.25))

    def zlim(z1):
        # (4/3)(z1^2 + z2^2 - z1 z2) <= 2 beta  ->  z2^2 - z1 z2 + z1^2 - 1.5 beta <= 0
        disc = z1 * z1 - 4 * (z1 * z1 - 1.5 * beta)
        if disc <= 0:
            return None
        rt = math.sqrt(disc)
        lo, hi = (z1 - rt) / 2, (z1 + rt) / 2
        return max(lo, -a), min(hi, a)

    def outer(z1):
        lim = zlim(z1)
        if lim is None or lim[0] >= lim[1]:
            return 0.0
        v, _ = integrate.quad(dens, lim[0], lim[1], args=(z1,), epsabs=1e-13, epsrel=1e-12)
        return v

    val, _ = integrate.quad(outer, -a, a, limit=400, epsabs=1e-12, epsrel=1e-11)
    return val


def mc_nobreak_n3(beta, runs, seed):
    """Simulation derselben Größe mit der Aufzählung aller vier Zerlegungen (floating point, ohne die Formel)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((runs, 3))
    s1 = np.concatenate([np.zeros((runs, 1)), np.cumsum(x, 1)], 1)
    s2 = np.concatenate([np.zeros((runs, 1)), np.cumsum(x * x, 1)], 1)

    def c(a, b):
        return (s2[:, b] - s2[:, a]) - (s1[:, b] - s1[:, a]) ** 2 / (b - a)

    obj_none = c(0, 3)
    obj_1 = c(0, 1) + c(1, 3) + beta
    obj_2 = c(0, 2) + c(2, 3) + beta
    obj_12 = c(0, 1) + c(1, 2) + c(2, 3) + 2 * beta
    none = (obj_none <= obj_1) & (obj_none <= obj_2) & (obj_none <= obj_12)
    return none.mean(), math.sqrt(none.mean() * (1 - none.mean()) / runs)

