"""Erzeugt `sgo_verfahren.json` (Experiment 4: schnell gegen optimal).

    python tools/sgo_make_verfahren.py --voll     # neu rechnen und schreiben (rund 3 bis 5 Minuten, ein Thread)
    python tools/sgo_make_verfahren.py --check    # kleine Neurechnung der generatorfreien exakten Teile (n = 5) und eines n = 200-Blocks (rund 15 s)

Zähleinheiten: **Instanzen** (eine Reihe mit einer Strafe), **Läufe** (simulierte Reihen), **Kostenauswertungen** (zulässige Strecken, deren Kosten ein Verfahren ausrechnet).
Exakt (grün, `fractions`, ganzzahlige Reihen): die Zählungen für n = 5 (alle 1024 Reihen mit Werten 0 bis 3) und die zufälligen ganzzahligen Reihen n <= 16 (Zufallsquelle der
Demo; ein anderer Generator gibt andere Zahlen). Simulation (gelb): n = 200, die 13 Zellen der Vormessung (ein Bruch, Treppe, Puls, Zickzack je δ = 1, 2, 3, plus Nullreihe)."""

import argparse
import itertools
import json
import os
import platform
import sys
import time
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import exact_ref as E  # noqa: E402

import sgo_constants as C  # noqa: E402
import sgo_scenario as SC  # noqa: E402
import sgo_segmentation as S  # noqa: E402
from sgo_normal import Stream  # noqa: E402

STAND = time.strftime("%Y-%m-%d")
N = 200
CELLS = [("null", 0.0)] + [(s, d) for s in ("ein", "treppe", "puls", "zickzack") for d in (1.0, 2.0, 3.0)]
SEED_ZUFALL = 47701
SEEDS_PELT = {1: (47810,), 5: tuple(range(47811, 47819)), 2: (47820, 47821)}
SEED_BIN = 47830
SEED_BIN_EXAKT = 47840


def exact_n5() -> dict:
    out = {}
    for ml in (1, 2):
        cnt = {1: 0, 2: 0}
        dl = 0
        for row in itertools.product(range(4), repeat=5):
            for be in (1, 2):
                a = E.dp_exact(list(row), be, ml)[0]
                cnt[be] += a != E.pelt_exact(list(row), be, ml)[0]
                dl += a != E.pelt_exact(list(row), be, ml, delayed=True)[0]
        out[str(ml)] = dict(reihen=1024, pelt_abw_beta1=cnt[1], pelt_abw_beta2=cnt[2], verzoegert_abw=dl, instanzen=2048)
    return out


class Draw:
    """Ganzzahlen aus dem SplitMix64-Strom der Demo, der Reihe nach verbraucht."""

    def __init__(self, seed: int, count: int = 400_000):
        self.u = Stream(seed).uniform(count)
        self.i = 0

    def below(self, bound: int) -> int:
        v = int(self.u[self.i] * bound)
        self.i += 1
        return v

    def rng(self, lo: int, hi: int) -> int:
        return lo + self.below(hi - lo + 1)

    def choice(self, seq):
        return seq[self.below(len(seq))]


def make_int_series(d: Draw, ml: int, lo_n: int, hi_n: int, base: int, jump: int, maxjumps: int):
    n = d.rng(max(lo_n, 2 * ml + 1), hi_n)
    x = [d.rng(-base, base) for _ in range(n)]
    for _ in range(d.rng(0, maxjumps)):
        p = d.rng(1, n - 1)
        h = d.rng(-jump, jump)
        x = [v + (h if i >= p else 0) for i, v in enumerate(x)]
    be = Fr(d.choice([1, 2, 3, 5, 8]), d.choice([1, 2]))
    return x, be


def exact_random(count=1500) -> dict:
    out = {}
    for ml in (1, 2, 3, 5):
        d = Draw(SEED_ZUFALL + ml)
        dif = dd = 0
        for _ in range(count):
            x, be = make_int_series(d, ml, 6, 16, 3, 4, 2)
            a = E.dp_exact(x, be, ml)[0]
            dif += a != E.pelt_exact(x, be, ml)[0]
            dd += a != E.pelt_exact(x, be, ml, delayed=True)[0]
        out[str(ml)] = dict(instanzen=count, pelt_abw=dif, verzoegert_abw=dd, seed=SEED_ZUFALL + ml)
    return out


def exact_binseg(count=3000) -> dict:
    d = Draw(SEED_BIN_EXAKT)
    dif = worse = better = 0
    for _ in range(count):
        ml = d.choice([1, 2])
        x, be = make_int_series(d, ml, 4, 14, 2, 4, 3)
        a = E.dp_exact(x, be, ml)
        bs = E.binseg_exact(x, be, ml)
        vb = E.objective_exact(x, bs, be)
        dif += bs != a[1]
        worse += vb > a[0]
        better += vb < a[0]
    return dict(instanzen=count, andere_bruchmenge=dif, zielwert_groesser=worse, zielwert_kleiner=better, seed=SEED_BIN_EXAKT)


def pelt_n200(ml: int, seeds, R_per_cell=40) -> dict:
    inst = dpp = dpd = 0
    ev = {"dp": 0, "pelt": 0, "del": 0}
    for seed in seeds:
        for ci, (scen, dlt) in enumerate(CELLS):
            X, _ = SC.draw_series(scen, N, dlt, R_per_cell, seed * 100 + ci)
            betas = [C.beta_of(c, N) for c in (1.0, 2.0)]
            res = S.dp_batch(X, betas, ml)
            for i in range(R_per_cell):
                for j, b in enumerate(betas):
                    pe = S.pelt(X[i], b, ml)
                    de = S.pelt(X[i], b, ml, delayed=True)
                    inst += 1
                    dpp += bool(abs(pe.objective - res.F[i, j]) > 1e-8)
                    dpd += bool(abs(de.objective - res.F[i, j]) > 1e-8) or de.breaks != res.breaks(i, j)
                    ev["dp"] += S.dp_evaluations(N, ml)
                    ev["pelt"] += pe.evaluations
                    ev["del"] += de.evaluations
    return dict(minlen=ml, seeds=list(seeds), instanzen=inst, pelt_abw=int(dpp), verzoegert_abw=int(dpd), ev_dp=ev["dp"], ev_pelt=ev["pelt"], ev_verzoegert=ev["del"],
                verhaeltnis_pelt=ev["pelt"] / ev["dp"], verhaeltnis_verzoegert=ev["del"] / ev["dp"])


def binseg_n200(R=400) -> dict:
    cs = (1.0, 2.0, 4.0)
    cells = {}
    tot_d = tot_w = tot_b = tot_i = 0
    per_c = {c: [0, 0] for c in cs}
    for ci, (scen, dlt) in enumerate(CELLS):
        X, _ = SC.draw_series(scen, N, dlt, R, SEED_BIN * 100 + ci)
        betas = [C.beta_of(c, N) for c in cs]
        res = S.dp_batch(X, betas, 5)
        row = {}
        for j, c in enumerate(cs):
            dif = wor = bet = 0
            for i in range(R):
                bs, _ = S.binseg(X[i], betas[j], 5)
                v = S.objective(X[i], bs, betas[j])
                dif += bs != res.breaks(i, j)
                wor += bool(v > res.F[i, j] + 1e-9)
                bet += bool(v < res.F[i, j] - 1e-9)
            row[str(c)] = dict(andere=dif / R, schlechter=wor / R, besser=int(bet))
            tot_d += dif
            tot_w += wor
            tot_b += bet
            tot_i += R
            per_c[c][0] += dif
            per_c[c][1] += R
        cells[f"{scen}|{dlt:.1f}"] = dict(seed=SEED_BIN * 100 + ci, R=R, **row)
        print(f"  binäre Segmentierung, Zelle {scen}|{dlt:.1f} fertig", flush=True)
    return dict(cells=cells, gesamt=dict(laeufe_mal_strafen=tot_i, andere=tot_d, schlechter=tot_w, besser=int(tot_b), anteil_andere=tot_d / tot_i,
                                          je_c={str(c): v[0] / v[1] for c, v in per_c.items()}), minlen=5, n=N)


def build() -> dict:
    t = time.time()
    n5 = exact_n5()
    print(f"  n = 5 exakt fertig ({time.time() - t:.0f} s)", flush=True)
    zuf = exact_random()
    print(f"  zufällige exakte Reihen fertig ({time.time() - t:.0f} s)", flush=True)
    be = exact_binseg()
    pelt = {str(ml): pelt_n200(ml, seeds) for ml, seeds in SEEDS_PELT.items()}
    print(f"  PELT n = 200 fertig ({time.time() - t:.0f} s)", flush=True)
    bn = binseg_n200()
    return dict(meta=dict(stand=STAND, quelle="tools/sgo_make_verfahren.py", python=platform.python_version(), numpy=np.__version__,
                          hinweis="exakt (grün): n = 5 und ganzzahlige Reihen (Zufallsquelle der Demo); Simulation (gelb): n = 200. Zähleinheiten Instanzen, Läufe, Kostenauswertungen."),
                n5_exakt=n5, zufall_exakt=zuf, binseg_exakt=be, pelt_n200=pelt, binseg_n200=bn)


def check(path: Path) -> int:
    z = json.loads(path.read_text(encoding="utf-8"))
    bad = 0
    n5 = exact_n5()
    if n5 != z["n5_exakt"]:
        bad += 1
        print("n = 5 exakt weicht ab", n5, z["n5_exakt"])
    r = pelt_n200(5, (47811,))
    old = z["pelt_n200"]["5"]
    if r["verhaeltnis_pelt"] and abs(r["verhaeltnis_pelt"] - old["verhaeltnis_pelt"]) > 0.02:
        bad += 1
        print("Kostenauswertungen PELT gegen DP weichen ab", r["verhaeltnis_pelt"], old["verhaeltnis_pelt"])
    if r["pelt_abw"] > 3 or r["verzoegert_abw"] != 0:
        bad += 1
        print("n = 200: PELT/verzögert weichen zu oft ab", r)
    print(f"--check: {bad} Abweichungen")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voll", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    path = ROOT / "sgo_verfahren.json"
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
