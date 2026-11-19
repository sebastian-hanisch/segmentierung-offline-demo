"""Erzeugt `sgo_online.json` (Experiment 5: dieselbe Reihe, zwei Fragen; Tab „Phase I“).

    python tools/sgo_make_online.py --voll     # neu rechnen und schreiben (R = 4000 für Experiment 5, R = 2000 für Phase I; rund 1 Minute)
    python tools/sgo_make_online.py --check    # Schwelle h und die exakten Teile neu rechnen (Integralgleichung gegen Kette), Simulation im Band (rund 5 s)

Der einseitige CUSUM (k = 0,5, nach oben) hat die Schwelle h für ARL₀ = 370,398 (zero-state, Alarm bei C >= h), gerechnet mit der Integralgleichung (48 Knoten);
die Brook/Evans-Kette (400 Zellen) ist der zweite Weg. Exakt (grün): h, ARL₀, ARL₁, Wahrscheinlichkeit eines Alarms vor ν, mittlere Verzögerung gegeben kein Alarm
vor ν. Simulation (gelb): Lagefehler beider Seiten und Anteile; **Verzögerung bei fester Lage ν, gegeben kein Alarm vor ν** (nicht das Lorden- und nicht das Shiryaev-Maß).
Offline: optimale Zerlegung, c = 2, Mindestlänge 5, n = 200."""

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import scipy  # noqa: E402

import sgo_constants as C  # noqa: E402
import sgo_online as ON  # noqa: E402
import sgo_scenario as SC  # noqa: E402
import sgo_segmentation as S  # noqa: E402
from sgo_normal import Stream  # noqa: E402

STAND = time.strftime("%Y-%m-%d")
SEED_ONLINE = 47501
SEED_PHASE1 = 47601
R_ONLINE, R_PHASE1, NU = 4000, 2000, 100
PHASE_N = 100


def exact_part() -> dict:
    h = ON.solve_h()
    Q0 = ON.chain_matrix(0.0, C.CUSUM_K, h)
    out = dict(h=h, k=C.CUSUM_K, arl0_ziel=ON.ARL0_TARGET, arl0_integral=ON.arl_integral_equation(0.0, C.CUSUM_K, h), arl0_kette_400=ON.chain_arl(Q0),
               arl0_kette_800=ON.chain_arl(ON.chain_matrix(0.0, C.CUSUM_K, h, 800)), arl1={}, verzoegerung={})
    for d in (1.0, 2.0):
        Q1 = ON.chain_matrix(d, C.CUSUM_K, h)
        out["arl1"][str(d)] = dict(integral=ON.arl_integral_equation(d, C.CUSUM_K, h), kette_400=ON.chain_arl(Q1))
        mean, sf, pe = ON.delay_exact(Q0, Q1, NU)
        out["verzoegerung"][str(d)] = dict(nu=NU, mittel=mean, p_alarm_vor_nu=pe, sf=sf)
    return out


def sim_part(R: int, h: float) -> dict:
    out = {}
    for d in (1.0, 2.0):
        r = SC.run_exp5(d, NU, R, SEED_ONLINE, h, limit=False)
        out[str(d)] = dict(seed=SEED_ONLINE, R=R, nu=NU, p_alarm_vor_nu=r.p_early_sim, verzoegerung_mittel=r.delay_sim, verzoegerung_se=r.delay_sim_se,
                           online_n=r.online_n, online_lage_betrag=r.online_lage_mean, online_lage_in5=r.online_lage_in5, offline_genau_ein_bruch=r.off_one,
                           offline_lage_betrag=r.off_lage_mean, offline_lage_in5=r.off_lage_in5, gleiche_laeufe_n=r.same_n, gleiche_offline_genau_ein=r.same_off_one,
                           gleiche_offline_lage_betrag=r.same_off_lage_mean, gleiche_online_lage_betrag=r.same_online_lage_mean)
    return out


def phase1_part(R: int) -> dict:
    out = {}
    beta = C.beta_of(2.0, PHASE_N)
    for i, d in enumerate((0.0, 0.5, 1.0, 1.5, 2.0, 3.0)):
        row = {}
        for pos, name in ((50, "mitte"), (10, "rand")):
            X = Stream(SEED_PHASE1 + 10 * i + (0 if pos == 50 else 1)).normal((R, PHASE_N))
            X[:, pos:] += d
            res = S.dp_batch(X, [beta], 5)
            row[name] = float((res.counts()[:, 0] >= 1).mean())
        out[str(d)] = dict(R=R, N=PHASE_N, c=2.0, minlen=5, seed=SEED_PHASE1 + 10 * i, **row)
    return out


def build(quick=False) -> dict:
    ex = exact_part()
    return dict(meta=dict(stand=STAND, quelle="tools/sgo_make_online.py", python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                          hinweis="exakt (grün): h, ARL, Kette; Simulation (gelb): Lagefehler und Anteile; Zähleinheit Läufe"),
                exakt=ex, simulation=sim_part(400 if quick else R_ONLINE, ex["h"]), phase1=phase1_part(200 if quick else R_PHASE1))


def check(path: Path) -> int:
    z = json.loads(path.read_text(encoding="utf-8"))
    bad = 0
    ex = exact_part()
    for key in ("h", "arl0_integral", "arl0_kette_400", "arl0_kette_800"):
        if abs(ex[key] - z["exakt"][key]) > 1e-9 * max(1.0, abs(ex[key])):
            bad += 1
            print("weicht ab:", key)
    if abs(ex["arl0_integral"] - ON.ARL0_TARGET) > 1e-6:
        bad += 1
        print("h trifft ARL0 nicht")
    if abs(ex["arl0_kette_800"] - ex["arl0_integral"]) > 0.01:
        bad += 1
        print("Kette (800 Zellen) und Integralgleichung weichen um mehr als 0,01 ab")
    for d in ("1.0", "2.0"):
        if abs(ex["verzoegerung"][d]["mittel"] - z["exakt"]["verzoegerung"][d]["mittel"]) > 1e-9:
            bad += 1
            print("Verzögerung weicht ab", d)
    sim = sim_part(400, z["exakt"]["h"])
    for d in ("1.0", "2.0"):
        a, b = sim[d]["p_alarm_vor_nu"], z["exakt"]["verzoegerung"][d]["p_alarm_vor_nu"]
        se = math.sqrt(b * (1 - b) / 400)
        if abs(a - b) > 4 * se:
            bad += 1
            print("Alarm vor nu außerhalb des Bandes", d, a, b)
    print(f"--check: {bad} Abweichungen")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voll", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    path = ROOT / "sgo_online.json"
    if a.check:
        sys.exit(1 if check(path) else 0)
    if not a.voll:
        ap.error("--voll oder --check angeben")
    t = time.time()
    path.write_text(json.dumps(build(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"geschrieben: {path.name} in {time.time() - t:.0f} s")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
