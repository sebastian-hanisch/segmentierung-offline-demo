"""Kern: Brüche (Änderungspunkte) im Rückblick finden. Mittelwertsprünge, bekannte Streuung 1, Gauß-Kosten.

Konventionen (wie im Plan):
* Reihe x[0..n-1]. Ein **Bruch bei s** (1 <= s <= n-1) heißt: x[0..s-1] liegen vor der Änderung, x[s..] danach; s = ν = Zahl der Beobachtungen vor der Änderung.
* **Kosten einer Strecke** (s, t] = Summe der Abweichungsquadrate um das Streckenmittel. **Strafe β** je Bruch, **Zielwert** = Summe der Kosten + β · (Zahl der Brüche),
  **Mindestlänge m** je Strecke.
* **Optimale Zerlegung** (Optimal Partitioning): F(0) = -β, F(t) = min über s von F(s) + C(s, t) + β. **Gleichstand:** der kleinste Startpunkt s der letzten Strecke gewinnt
  (`argmin` nimmt das erste Minimum).
* Jede Reihe wird vor der Rechnung um ihr Mittel verschoben (Zentrieren). Mathematisch ändert das nichts; die Summenformel S2 - S1²/n verliert sonst bei großem Mittel
  Stellen (Messung V16 J: Verschiebung um 10⁷ ändert die Zerlegung in 84 von 200 Läufen, nach dem Zentrieren in keinem).

Zähleinheiten: **Kostenauswertungen** (zulässige Strecken (s, t], deren Kosten ein Verfahren ausrechnet), **Zerlegungen** (Aufzählung), Läufe (simulierte Reihen)."""

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------------------------------------------------------------- Hilfen
def center(X) -> np.ndarray:
    """Jede Zeile um ihr Mittel verschieben (Zentrieren): ändert die Zerlegung mathematisch nicht, verhindert Stellenverlust bei großem Mittel."""
    X = np.atleast_2d(np.asarray(X, float))
    return X - X.mean(axis=1, keepdims=True)


def _cums(X: np.ndarray):
    z = np.zeros((X.shape[0], 1))
    return np.concatenate([z, np.cumsum(X, 1)], 1), np.concatenate([z, np.cumsum(X * X, 1)], 1)


def seg_cost(s1, s2, s, t):
    """Kosten der Strecke (s, t] aus Präfixsummen: Summe der Quadrate minus Quadrat der Summe durch die Länge."""
    return (s2[t] - s2[s]) - (s1[t] - s1[s]) ** 2 / (t - s)


def robust_sigma(X) -> np.ndarray:
    """Robuste Streuungsschätzung je Zeile: Median der Beträge der Differenzen durch (√2 · 0,6745)."""
    X = np.atleast_2d(np.asarray(X, float))
    return np.median(np.abs(np.diff(X, axis=1)), axis=1) / (np.sqrt(2.0) * 0.6745)


def count_decompositions(n: int, minlen: int) -> int:
    """Zahl der Zerlegungen von n Beobachtungen in Strecken der Länge >= minlen: 2^(n-1) bei minlen 1, Fibonacci-Zahlen bei minlen 2."""
    if n < minlen:
        return 0
    c = [0] * (n + 1)
    c[0] = 1
    for t in range(1, n + 1):
        c[t] = sum(c[t - k] for k in range(minlen, t + 1))
    return c[n]


def allowed_starts(t: int, minlen: int) -> list[int]:
    """Startpunkte s der letzten Strecke (s, t], von denen aus eine zulässige Zerlegung von (0, s] existiert: 0 oder s >= minlen, und t - s >= minlen."""
    return [s for s in range(0, t - minlen + 1) if s == 0 or s >= minlen]


def dp_evaluations(n: int, minlen: int) -> int:
    """Kostenauswertungen der optimalen Zerlegung ohne Abkürzung (eine Reihe, eine Strafe): alle zulässigen Strecken (s, t]."""
    return sum(len(allowed_starts(t, minlen)) for t in range(minlen, n + 1))


# ------------------------------------------------------------------------------------------------------------------- optimale Zerlegung (DP)
@dataclass
class DPBatch:
    """Ergebnis der optimalen Zerlegung für R Reihen und P Strafen. F: Zielwerte (R, P); table: F(0..n) (R, P, n+1); mask: Bruchstellen (R, P, n+1), True bei Bruch s."""

    F: np.ndarray
    table: np.ndarray
    mask: np.ndarray
    minlen: int

    def breaks(self, r: int = 0, p: int = 0) -> list[int]:
        return [int(s) for s in np.flatnonzero(self.mask[r, p])]

    def counts(self) -> np.ndarray:
        return self.mask.sum(axis=2)


def dp_batch(X, betas, minlen: int, do_center: bool = True) -> DPBatch:
    """Optimale Zerlegung (Optimal Partitioning, ohne Abkürzung) für alle Zeilen von X (R, n) und alle Strafen betas (P,) zugleich."""
    X = np.atleast_2d(np.asarray(X, float))
    R, n = X.shape
    betas = np.atleast_1d(np.asarray(betas, float))
    P = betas.size
    if minlen < 1:
        raise ValueError("Mindestlänge muss mindestens 1 sein")
    if n < minlen:
        raise ValueError(f"Die Reihe hat {n} Werte, die Mindestlänge ist {minlen}: keine Zerlegung möglich")
    if not np.all(np.isfinite(X)):
        raise ValueError("Die Reihe enthält NaN oder unendliche Werte")
    if do_center:
        X = center(X)
    S1, S2 = _cums(X)
    F = np.full((R, P, n + 1), np.inf)
    F[:, :, 0] = -betas[None, :]
    last = np.zeros((R, P, n + 1), dtype=np.int64)
    for t in range(minlen, n + 1):
        smax = t - minlen
        s = np.arange(smax + 1)
        c = (S2[:, t, None] - S2[:, : smax + 1]) - (S1[:, t, None] - S1[:, : smax + 1]) ** 2 / (t - s)[None, :]
        tot = F[:, :, : smax + 1] + c[:, None, :]
        am = tot.argmin(2)  # Gleichstand: kleinster Startpunkt s
        F[:, :, t] = np.take_along_axis(tot, am[:, :, None], 2)[:, :, 0] + betas[None, :]
        last[:, :, t] = am
    mask = np.zeros((R, P, n + 1), dtype=bool)
    t = np.full((R, P), n, dtype=np.int64)
    ri, pi = np.meshgrid(np.arange(R), np.arange(P), indexing="ij")
    while True:
        active = t > 0
        if not active.any():
            break
        s = np.take_along_axis(last, t[:, :, None], 2)[:, :, 0]
        hit = active & (s > 0)
        mask[ri[hit], pi[hit], s[hit]] = True
        t = np.where(active, s, 0)
    return DPBatch(F[:, :, n].copy(), F, mask, minlen)


def objective(x, breaks, beta: float) -> float:
    """Zielwert einer Zerlegung: Summe der Kosten der Strecken plus β mal Zahl der Brüche."""
    x = center(x)[0]
    b = [0] + sorted(int(s) for s in breaks) + [x.size]
    tot = 0.0
    for s, t in zip(b[:-1], b[1:]):
        seg = x[s:t]
        tot += float(((seg - seg.mean()) ** 2).sum())
    return tot + beta * len(b[1:-1])


def gain(x, lo: int, hi: int, tau: int) -> float:
    """Kostengewinn der Teilung der Strecke (lo, hi] bei tau: (Länge links · Länge rechts / Länge gesamt) · (Mittel links − Mittel rechts)²."""
    x = center(x)[0]
    S1, S2 = _cums(x[None, :])
    S1, S2 = S1[0], S2[0]
    return float(seg_cost(S1, S2, lo, hi) - seg_cost(S1, S2, lo, tau) - seg_cost(S1, S2, tau, hi))


# ---------------------------------------------------------------------------------------------------------------------------------- PELT
@dataclass
class PeltResult:
    breaks: list[int]
    objective: float
    evaluations: int
    log: dict = field(default_factory=dict)   # t -> Kandidatenliste nach dem Beschneiden bei t (nur mit trace=True)
    f_table: list = field(default_factory=list)


def pelt(x, beta: float, minlen: int, delayed: bool = False, trace: bool = False) -> PeltResult:
    """PELT (Killick u. a. 2012) mit K = 0: Kandidat s fällt bei t, wenn F(s) + C(s, t) >= F(t). Mit `delayed=True` (eigener Vorschlag dieses Stücks, keine
    Quellenaussage) wird ein als beschneidbar markierter Kandidat erst nach `minlen - 1` weiteren Zeitpunkten gestrichen; bei minlen = 1 ist das dasselbe."""
    x = center(x)[0]
    n = x.size
    if n < minlen:
        raise ValueError(f"Die Reihe hat {n} Werte, die Mindestlänge ist {minlen}: keine Zerlegung möglich")
    S1, S2 = _cums(x[None, :])
    S1, S2 = S1[0], S2[0]
    F = np.full(n + 1, np.inf)
    F[0] = -beta
    last = np.zeros(n + 1, dtype=np.int64)
    cand = np.array([], dtype=np.int64)
    ev = 0
    flagged: dict = {}
    log: dict = {}
    for t in range(minlen, n + 1):
        s_new = t - minlen
        if s_new == 0 or s_new >= minlen:
            cand = np.append(cand, s_new)
        c = (S2[t] - S2[cand]) - (S1[t] - S1[cand]) ** 2 / (t - cand)
        vals = F[cand] + c + beta
        j = int(np.argmin(vals))
        F[t] = vals[j]
        last[t] = cand[j]
        ev += cand.size
        # Beschneidungsregel F(s) + C(s, t) >= F(t). Bei β = 0 würde sie den besten Kandidaten selbst streichen (F(t) = F(s) + C(s, t) für den besten s):
        # dort gilt die strenge Ungleichung (β <= 0 ist kein Fall der Quelle, nur ein Randfall des Reglers).
        mark = ((vals - beta) >= F[t]) if beta > 0 else ((vals - beta) > F[t])
        marked = cand[mark]
        if not delayed:
            cand = cand[~mark]
        else:
            flagged[t] = marked
            tt = t - minlen + 1
            if tt in flagged:
                rm = flagged.pop(tt)
                if rm.size:
                    cand = cand[~np.isin(cand, rm)]
        if trace:
            log[t] = [int(v) for v in cand]
    out, t = [], n
    while t > 0:
        s = int(last[t])
        if s > 0:
            out.append(s)
        t = s
    return PeltResult(sorted(out), float(F[n]), ev, log, [float(v) for v in F])


# ------------------------------------------------------------------------------------------------------------------------ binäre Segmentierung
def binseg(x, beta: float, minlen: int):
    """Binäre Segmentierung: auf jeder Strecke die Teilung mit dem größten Kostengewinn; geteilt wird nur, wenn der Gewinn **größer als β** ist.
    Rückgabe (Brüche, Schritte); Schritt = (lo, hi, tau, Gewinn, geteilt). Gleichstand: kleinstes tau."""
    x = center(x)[0]
    n = x.size
    S1, S2 = _cums(x[None, :])
    S1, S2 = S1[0], S2[0]
    res, steps = [], []
    stack = [(0, n)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2 * minlen:
            continue
        taus = np.arange(lo + minlen, hi - minlen + 1)
        gains = seg_cost(S1, S2, lo, hi) - seg_cost(S1, S2, lo, taus) - seg_cost(S1, S2, taus, hi)
        j = int(np.argmax(gains))
        split = bool(gains[j] > beta)
        steps.append((lo, hi, int(taus[j]), float(gains[j]), split))
        if split:
            res.append(int(taus[j]))
            stack.append((lo, int(taus[j])))
            stack.append((int(taus[j]), hi))
    return sorted(res), steps


# ----------------------------------------------------------------------------------------------------------- Aufzählung aller Zerlegungen
ENUM_LIMIT = 14   # größte Reihenlänge der Aufzählung (2^13 = 8 192 Zerlegungen bei Mindestlänge 1)


@dataclass
class Enumeration:
    best: float
    optimal: list      # alle Zerlegungen (Listen von Brüchen) mit Zielwert innerhalb 1e-9 des besten
    count: int         # Zahl aller Zerlegungen


def enumerate_best(x, beta: float, minlen: int) -> Enumeration:
    """Zweitentscheider: alle Zerlegungen in Strecken der Länge >= minlen aufzählen (nur n <= ENUM_LIMIT)."""
    x = center(x)[0]
    n = x.size
    if n > ENUM_LIMIT:
        raise ValueError(f"Aufzählung nur bis n = {ENUM_LIMIT} (2^{ENUM_LIMIT - 1} Zerlegungen)")
    if n < minlen:
        raise ValueError(f"Die Reihe hat {n} Werte, die Mindestlänge ist {minlen}: keine Zerlegung möglich")
    S1, S2 = _cums(x[None, :])
    S1, S2 = S1[0], S2[0]
    allv: list = []

    def rec(start, chosen, acc):
        if n - start >= minlen:
            allv.append((acc + seg_cost(S1, S2, start, n) + beta * len(chosen), list(chosen)))
        for p in range(start + minlen, n - minlen + 1):
            rec(p, chosen + [p], acc + seg_cost(S1, S2, start, p))

    rec(0, [], 0.0)
    best = min(v for v, _ in allv)
    opt = [b for v, b in allv if v <= best + 1e-9]
    return Enumeration(float(best), opt, len(allv))


# --------------------------------------------------------------------------------------------------------------------------- Bewertung
def f1_tolerance(est, true, m: float):
    """Genauigkeit und Ausbeute mit Toleranz m: ein Bruch gilt als gefunden, wenn ein anderer **näher als m** Beobachtungen liegt.
    F1 = 0, wenn nichts getroffen oder nichts gefunden wird; ohne wahren Bruch nicht definiert (None). Rückgabe (Genauigkeit, Ausbeute, F1)."""
    est, true = list(est), list(true)
    hit_est = sum(1 for e in est if any(abs(e - t) < m for t in true))
    hit_true = sum(1 for t in true if any(abs(e - t) < m for e in est))
    p = hit_est / len(est) if est else None
    r = hit_true / len(true) if true else None
    if not true:
        return p, r, None
    if p is None or p + r == 0:
        return p, r, 0.0
    return p, r, 2 * p * r / (p + r)


def location_errors(est, true) -> list[float]:
    """Lagefehler je wahrem Bruch, nur bei gleicher Zahl gefundener und wahrer Brüche: |gefunden − wahr| in Beobachtungen, der Reihe nach gepaart."""
    est, true = sorted(est), sorted(true)
    if len(est) != len(true):
        return []
    return [abs(e - t) for e, t in zip(est, true)]
