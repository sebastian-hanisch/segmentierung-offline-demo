"""Exakte Wahrscheinlichkeiten der Nullreihe (kein wahrer Bruch, Normalrauschen, Streuung 1, Mindestlänge 1) für n = 2 und n = 3.

Der Gewinn der Teilung an einer **festen** Stelle ist χ² mit einem Freiheitsgrad. Für n = 2 ist ein Fehlbruch genau dann, wenn (x₁ − x₂)² / 2 > β; also
P = P(χ²₁ > β) = 2 · Φ̄(√β) = erfc(√(β / 2)). Für n = 3 liegt „kein Bruch“ genau dann vor, wenn beide Teilungsgewinne höchstens β sind und die Gesamtkosten höchstens 2β:
ein Winkelintegral (Gauß-Legendre über Teilintervalle mit den Knickstellen der Integranden; die Knickstellen sind unabhängig von β). Für n >= 4 gibt es keine
geschlossene Form (nicht hergeleitet): dort bleibt die Simulation."""

import math

import numpy as np
from numpy.polynomial.legendre import leggauss


def p_break_n2(beta: float) -> float:
    """n = 2, Mindestlänge 1: Wahrscheinlichkeit eines Fehlbruchs = P(χ²₁ > β) = erfc(√(β/2))."""
    return math.erfc(math.sqrt(beta / 2.0))


def p_fixed_site(beta: float) -> float:
    """Wahrscheinlichkeit, dass der Gewinn der Teilung an einer **festen** Stelle β übersteigt: dieselbe Formel (χ² mit einem Freiheitsgrad)."""
    return p_break_n2(beta)


_ANGLES = [0.0] + [math.radians(a) for a in (15, 30, 45, 105, 120, 135)] + [math.pi]


def p_nobreak_n3(beta: float, nodes: int = 40) -> float:
    """n = 3, Mindestlänge 1: P(kein Bruch) = (1/π) ∫₀^π [1 − exp(−min(2β, β/cos²φ, β/cos²(φ − 60°)) / 2)] dφ.

    Herleitung: die Abweichungen vom Mittel liegen in einer Ebene (Abstand² ~ χ²₂), die beiden Teilungsrichtungen schneiden sich unter 60°; kein Bruch heißt Abstand²
    <= 2β und beide Projektionen höchstens √β. Der Integrand hat Periode π und Knickstellen bei 15°, 30°, 45°, 105°, 120°, 135° (Gleichheit zweier der drei Terme)."""
    if beta <= 0:
        return 0.0
    x, w = leggauss(nodes)
    tot = 0.0
    for a, b in zip(_ANGLES[:-1], _ANGLES[1:]):
        phi = 0.5 * (b - a) * x + 0.5 * (b + a)
        wt = 0.5 * (b - a) * w
        c1 = np.cos(phi) ** 2
        c2 = np.cos(phi - math.pi / 3.0) ** 2
        r2 = np.minimum(2.0 * beta, np.minimum(np.where(c1 > 1e-300, beta / np.maximum(c1, 1e-300), np.inf), np.where(c2 > 1e-300, beta / np.maximum(c2, 1e-300), np.inf)))
        tot += float(np.sum(wt * (1.0 - np.exp(-r2 / 2.0))))
    return tot / math.pi


def n_sites(n: int, minlen: int) -> int:
    """Zahl möglicher Bruchstellen s mit minlen <= s <= n − minlen (jede Strecke mindestens minlen lang)."""
    return max(0, n - 2 * minlen + 1)


def sites_times_single(n: int, c: float, minlen: int) -> float:
    """„Zahl der Stellen × Einzelwert“: **keine Schranke** (die Gewinne benachbarter Stellen sind stark korreliert; der Wert kann über 1 liegen)."""
    return n_sites(n, minlen) * p_fixed_site(c * math.log(n))
