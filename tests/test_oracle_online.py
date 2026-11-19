"""Orakel für die Online-Seite (Experiment 5): CUSUM-ARL aus der Integralgleichung gegen die Brook/Evans-Kette (zwei Wege), exakte Verzögerung gegen Simulation (Band)."""

import math

import numpy as np
import pytest

import sgo_online as ON
from sgo_normal import Stream


@pytest.fixture(scope="module")
def h():
    return ON.solve_h()


def test_schwelle_trifft_arl0_auf_beiden_wegen(h):
    assert abs(h - 4.0965) < 5e-5
    assert abs(ON.ARL0_TARGET - 370.398) < 5e-4
    assert abs(ON.arl_integral_equation(0.0, 0.5, h) - ON.ARL0_TARGET) < 1e-6
    assert abs(ON.chain_arl(ON.chain_matrix(0.0, 0.5, h, 800)) - ON.ARL0_TARGET) < 0.01
    assert abs(ON.chain_arl(ON.chain_matrix(0.0, 0.5, h, 400)) - 370.3887) < 5e-4        # Kette nähert sich von unten


@pytest.mark.parametrize("delta,soll", [(1.0, 8.5751), (2.0, 3.4068)])
def test_arl1_zero_state_auf_zwei_wegen(h, delta, soll):
    a = ON.arl_integral_equation(delta, 0.5, h)
    b = ON.chain_arl(ON.chain_matrix(delta, 0.5, h))
    assert abs(a - b) < 5e-5 and abs(a - soll) < 5e-4


def test_exakte_alarmwahrscheinlichkeit_und_verzoegerung(h):
    Q0 = ON.chain_matrix(0.0, 0.5, h)
    assert abs(ON.prob_alarm_by(Q0, 100) - 0.2302) < 5e-5
    m1, sf1, pe = ON.delay_exact(Q0, ON.chain_matrix(1.0, 0.5, h), 100)
    m2, _, _ = ON.delay_exact(Q0, ON.chain_matrix(2.0, 0.5, h), 100)
    assert abs(m1 - 7.9058) < 5e-4 and abs(m2 - 3.1090) < 5e-4 and abs(pe - 0.2302) < 5e-5
    assert sf1[0] == pytest.approx(1.0) and all(a >= b - 1e-15 for a, b in zip(sf1, sf1[1:]))
    # Mittelwert = Summe der Überlebenswahrscheinlichkeiten P(Verzögerung > d), d >= 0 (Rest jenseits von d = 80 ist winzig)
    assert abs(sum(sf1) - m1) < 1e-3


def test_exakte_verzoegerung_gleich_simulation_im_band(h):
    delta, nu, R, n = 1.0, 100, 4000, 200
    X = Stream(21).normal((R, n))
    X[:, nu:] += delta
    _, T = ON.cusum_paths(X, 0.5, h)
    late = (T > nu) & (T <= n)
    dl = (T[late] - nu).astype(float)
    Q0 = ON.chain_matrix(0.0, 0.5, h)
    mean, _, pe = ON.delay_exact(Q0, ON.chain_matrix(delta, 0.5, h), nu)
    se = dl.std(ddof=1) / math.sqrt(dl.size)
    assert abs(dl.mean() - mean) < 4 * se
    assert abs((T <= nu).mean() - pe) < 4 * math.sqrt(pe * (1 - pe) / R)


def test_lageschaetzung_online_ist_der_letzte_nullpunkt():
    C = np.array([0, 0.2, 0.0, 0.4, 0.9, 1.6])
    assert ON.last_zero_before(C, 5) == 2 and ON.last_zero_before(C, 1) == 0
