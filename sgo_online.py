"""Die Online-Seite für Experiment 5: einseitiger CUSUM (nach oben), k = 0,5, Schwelle h für ARL₀ = 370,398 (zero-state), Brook/Evans-Kette und Integralgleichung.

Konventionen der Linie: Alarm bei C >= h; **zero-state** heißt Start bei C = 0; ARL₀-Ziel **370,398** = 1/(2 Φ(−3)), nicht das gesetzte 370 (V1 nahm 370,000 und h = 4,0954).
Die Verzögerung in Experiment 5 ist das Pollak-Summand-Maß bei fester Lage ν **gegeben kein Alarm vor ν** (Start 0, ν Schritte Einschwingen); für den CUSUM praktisch
das steady-state-Maß (quasistationär), nicht das Lorden- und nicht das Shiryaev-Maß."""

import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.special import ndtr

K = 0.5
ARL0_TARGET = 1.0 / (2.0 * float(ndtr(-3.0)))   # 370,398…
CELLS = 400


def chain_matrix(delta: float, k: float, h: float, cells: int = CELLS) -> np.ndarray:
    """Brook/Evans-Kette für C = max(0, C + x − k), x ~ N(delta, 1): Zustand 0 und `cells` Zellen über (0, h]; Übergangsmatrix auf die nicht-absorbierten Zustände."""
    w = h / cells
    reps = np.concatenate([[0.0], (np.arange(cells) + 0.5) * w])
    edges = np.arange(cells + 1) * w
    F = ndtr(k - reps[:, None] + edges[None, :] - delta)
    Q = np.empty((cells + 1, cells + 1))
    Q[:, 0] = ndtr(k - reps - delta)
    Q[:, 1:] = F[:, 1:] - F[:, :-1]
    return np.maximum(Q, 0.0)


def chain_arl(Q: np.ndarray, start: int = 0) -> float:
    return float(np.linalg.solve(np.eye(Q.shape[0]) - Q, np.ones(Q.shape[0]))[start])


def arl_integral_equation(delta: float, k: float, h: float, nodes: int = 48) -> float:
    """Zweiter Rechenweg: Integralgleichung (Nyström, Gauß-Legendre) für die ARL des oberen CUSUM, Start 0, x ~ N(delta, 1)."""
    x, w = leggauss(nodes)
    v, wv = 0.5 * h * x + 0.5 * h, 0.5 * h * w
    u = np.concatenate([[0.0], v])
    A = np.empty((nodes + 1, nodes + 1))
    A[:, 0] = ndtr(k - u - delta)
    A[:, 1:] = wv[None, :] * np.exp(-0.5 * (v[None, :] + k - u[:, None] - delta) ** 2) / math.sqrt(2 * math.pi)
    L = np.linalg.solve(np.eye(nodes + 1) - A, np.ones(nodes + 1))
    return float(L[0])


def solve_h(target: float = ARL0_TARGET, k: float = K, nodes: int = 48) -> float:
    return float(brentq(lambda h: arl_integral_equation(0.0, k, h, nodes) - target, 0.5, 20.0, xtol=1e-10))


def prob_alarm_by(Q0: np.ndarray, nu: int) -> float:
    """Exakt: Wahrscheinlichkeit eines Alarms in den ersten ν Beobachtungen (Fehlalarm vor dem Sprung, zero-state)."""
    v = np.zeros(Q0.shape[0])
    v[0] = 1.0
    for _ in range(nu):
        v = v @ Q0
    return float(1.0 - v.sum())


def delay_exact(Q0: np.ndarray, Q1: np.ndarray, nu: int, dmax: int = 80):
    """Exakt aus der Kette: Verteilung der Verzögerung (Zahl der Beobachtungen nach dem Sprung bis zum Alarm), gegeben kein Alarm vor ν.
    Rückgabe (Mittelwert, P(Verzögerung > d) für d = 0 … dmax, Wahrscheinlichkeit eines Alarms vor ν)."""
    v = np.zeros(Q0.shape[0])
    v[0] = 1.0
    for _ in range(nu):
        v = v @ Q0
    surv = float(v.sum())
    pi = v / surv
    L = np.linalg.solve(np.eye(Q1.shape[0]) - Q1, np.ones(Q1.shape[0]))
    mean = float(pi @ L)
    sf = []
    u = pi.copy()
    sf.append(float(u.sum()))
    for _ in range(dmax):
        u = u @ Q1
        sf.append(float(u.sum()))
    return mean, sf, 1.0 - surv


def cusum_paths(X: np.ndarray, k: float, h: float):
    """CUSUM-Pfade C_0 … C_n (C_0 = 0) und Alarmzeit je Zeile (erstes t mit C_t >= h, 1-basiert; n + 1 heißt kein Alarm)."""
    X = np.atleast_2d(np.asarray(X, float))
    R, n = X.shape
    C = np.zeros((R, n + 1))
    for t in range(1, n + 1):
        C[:, t] = np.maximum(0.0, C[:, t - 1] + X[:, t - 1] - k)
    over = C >= h
    T = np.where(over.any(1), over.argmax(1), n + 1)
    return C, T


def last_zero_before(C_row: np.ndarray, T: int) -> int:
    """Lage-Schätzung des CUSUM: letzter Zeitpunkt mit C = 0 bis zum Alarm (Zahl der Beobachtungen vor der Änderung)."""
    seg = C_row[: T + 1]
    return int(np.flatnonzero(seg == 0.0).max())
