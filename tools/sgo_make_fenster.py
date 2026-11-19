"""Erzeugt die gefrorenen Tabellen `sgo_fenster.json` (Experiment 2: die Strafe ist ein Fenster) und `sgo_null_n.json` (Experiment 3: die Nullreihe).

    python tools/sgo_make_fenster.py --voll     # beide Tabellen neu rechnen und schreiben (rund 1 Minute, ein Thread)
    python tools/sgo_make_fenster.py --check    # kleine Neurechnung (R = 100), vergleicht im Band mit den gefrorenen Zahlen (rund 20 s)

Zähleinheit: Läufe (eine simulierte Reihe = ein Lauf). Zufall: SplitMix64 der Demo (`sgo_normal.Stream`), je Zeile ein eigener Zufalls-Seed. Jede Zelle nennt
Zufalls-Seed, R, Stand und Quelle. Die Anteile sind Simulation (gelb, Standardfehler eines Anteils bis 1,6 Prozentpunkte bei R = 1000), die Wahrscheinlichkeiten
für n = 2 und n = 3 sind exakt (grün) und stehen nur zum Vergleich daneben."""

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
import sgo_exact as EX  # noqa: E402
import sgo_scenario as SC  # noqa: E402
import sgo_segmentation as S  # noqa: E402
from sgo_normal import Stream  # noqa: E402

STAND = time.strftime("%Y-%m-%d")
N = 200
MINLEN = 5
R_FULL = 1000
SEEDS_FENSTER = (47101, 47102, 47103)
SEED_MINLEN, SEED_PULS = 47201, 47301
NULL_SEEDS = {50: 47450, 100: 47500, 200: 47600, 400: 47700, 800: 47800}
SEED_N2, SEED_N3 = 47901, 47902
CELLS = [("null", 0.0)] + [(s, d) for s in ("ein", "treppe", "puls", "zickzack") for d in (1.0, 2.0, 3.0)]


def meta(**extra) -> dict:
    return dict(stand=STAND, quelle="tools/sgo_make_fenster.py", python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                hinweis="Simulation (gelb): Anteile mit Standardfehler bis 1,6 Prozentpunkte bei R = 1000; n = 2 und n = 3 exakt (grün). Zähleinheit Läufe.", **extra)


def cell_stats(X, true, minlen, sigma_robust):
    Xe = X / S.robust_sigma(X)[:, None] if sigma_robust else X
    betas = np.array([C.beta_of(c, X.shape[1]) for c in C.C_GRID])
    res = S.dp_batch(Xe, betas, minlen)
    R = X.shape[0]
    cnt = res.counts()
    out = dict(p_right=[], p_any=[], mean_count=[], p_loc=[], loc_error=[])
    for j in range(len(C.C_GRID)):
        right = loc_ok = 0
        errs = []
        for r in range(R):
            est = res.breaks(r, j)
            if len(est) == len(true):
                right += 1
                e = S.location_errors(est, true)
                errs += e
                loc_ok += all(v <= C.LOC_TOL for v in e)
        out["p_right"].append(right / R)
        out["p_any"].append(float((cnt[:, j] > 0).mean()))
        out["mean_count"].append(float(cnt[:, j].mean()))
        out["p_loc"].append(loc_ok / R)
        out["loc_error"].append(float(np.mean(errs)) if errs else None)
    j2 = C.C_GRID.index(2.0)
    if true:
        f1, f1z = {}, {}
        for m in C.TOL_M:
            vals = [S.f1_tolerance(res.breaks(r, j2), true, m)[2] for r in range(R)]
            f1[str(m)] = float(np.mean(vals))
            f1z[str(m)] = float(np.mean([v == 0.0 for v in vals]))
        out["f1"], out["f1_nullanteil"] = f1, f1z
        out["p_genau_wahre_menge"] = float(np.mean([res.breaks(r, j2) == true for r in range(R)]))
    return out


def make_fenster(R: int, seeds=SEEDS_FENSTER, quick=False) -> dict:
    cells = {}
    for scen, d in CELLS:
        key = f"{scen}|{d:.1f}"
        per_seed = []
        for si, seed in enumerate(seeds):
            X, true = SC.draw_series(scen, N, d, R, seed * 100 + CELLS.index((scen, d)))
            row = {"seed": seed * 100 + CELLS.index((scen, d)), "R": R, "bekannt": cell_stats(X, true, MINLEN, False)}
            if not quick:
                row["robust"] = cell_stats(X, true, MINLEN, True)
            per_seed.append(row)
        cells[key] = {"scen": scen, "delta": d, "wahre_brueche": SC.true_breaks(scen, N), "per_seed": per_seed}
        print(f"  Zelle {key} fertig", flush=True)
    # Mindestlänge als zweite Stellschraube: Nullreihen
    ml = {}
    Xn = Stream(SEED_MINLEN).normal((R, N))
    for m in (1, 2, 5, 10, 20):
        res = S.dp_batch(Xn, [c * math.log(N) for c in (1.0, 2.0, 3.0)], m)
        ml[str(m)] = [float((res.counts()[:, j] > 0).mean()) for j in range(3)]
    # kurzer Puls gegen Mindestlänge 5
    puls = {}
    for w in (2, 4, 5, 8, 20):
        X = Stream(SEED_PULS + w).normal((R, N))
        X[:, 100:100 + w] += 3.0
        res = S.dp_batch(X, [2.0 * math.log(N)], MINLEN)
        cnt = res.counts()[:, 0]
        puls[str(w)] = dict(p_right=float((cnt == 2).mean()), mean_count=float(cnt.mean()))
    return dict(meta=meta(n=N, minlen=MINLEN, R=R, c_grid=list(C.C_GRID), seeds=list(seeds), tol_m=list(C.TOL_M),
                          mindestlaenge=dict(seed=SEED_MINLEN, c=[1.0, 2.0, 3.0], R=R), puls=dict(seed=SEED_PULS, delta=3.0, c=2.0, R=R, minlen=MINLEN)),
                cells=cells, mindestlaenge=ml, puls_breite=puls)


def enum_nobreak_n3(beta: float, runs: int, seed: int):
    """Zweiter Weg (ohne `dp_batch`): Aufzählung aller vier Zerlegungen von drei Werten in numpy."""
    x = Stream(seed).normal((runs, 3))
    s1 = np.concatenate([np.zeros((runs, 1)), np.cumsum(x, 1)], 1)
    s2 = np.concatenate([np.zeros((runs, 1)), np.cumsum(x * x, 1)], 1)

    def c(a, b):
        return (s2[:, b] - s2[:, a]) - (s1[:, b] - s1[:, a]) ** 2 / (b - a)

    o0 = c(0, 3)
    o1 = c(0, 1) + c(1, 3) + beta
    o2 = c(0, 2) + c(2, 3) + beta
    o12 = c(0, 1) + c(1, 2) + c(2, 3) + 2 * beta
    none = (o0 <= o1) & (o0 <= o2) & (o0 <= o12)
    return float(none.mean())


def make_null(R: int, quick=False) -> dict:
    out = {"n2": {}, "n3": {}, "n_reihe": {}}
    runs_f = 200_000 if not quick else 20_000
    runs2 = 2_000_000 if not quick else 200_000
    runs3 = 1_000_000 if not quick else 100_000
    for c in (0.5, 1.0, 2.0, 3.0):
        b = C.beta_of(c, 2)
        Z = Stream(SEED_N2).normal((runs2, 2))
        g = (Z[:, 0] - Z[:, 1]) ** 2 / 2
        res = S.dp_batch(Z[:runs_f], [b], 1)
        out["n2"][str(c)] = dict(beta=b, exakt=EX.p_break_n2(b), sim_formel=float((g > b).mean()), runs_formel=runs2, sim_dp=float((res.counts()[:, 0] > 0).mean()), runs_dp=runs_f,
                                 seed=SEED_N2)
        b3 = C.beta_of(c, 3)
        X3 = Stream(SEED_N3).normal((runs3, 3))
        res3 = S.dp_batch(X3, [b3], 1)
        out["n3"][str(c)] = dict(beta=b3, exakt_kein_bruch=EX.p_nobreak_n3(b3), sim_aufzaehlung=enum_nobreak_n3(b3, runs3, SEED_N3 + 1), runs_aufzaehlung=runs3,
                                 sim_dp=float((res3.counts()[:, 0] == 0).mean()), runs_dp=runs3, seed_dp=SEED_N3, seed_aufzaehlung=SEED_N3 + 1)
    for n, seed in NULL_SEEDS.items():
        X = Stream(seed).normal((R, n))
        res = S.dp_batch(X, [c * math.log(n) for c in (1.0, 2.0, 3.0)], MINLEN)
        cnt = res.counts()
        row = dict(seed=seed, R=R, minlen=MINLEN, n_stellen=EX.n_sites(n, MINLEN))
        for j, c in enumerate((1.0, 2.0, 3.0)):
            row[str(c)] = dict(p_fehlbruch=float((cnt[:, j] > 0).mean()), feste_stelle=EX.p_fixed_site(c * math.log(n)), stellen_mal_einzel=EX.sites_times_single(n, c, MINLEN))
        out["n_reihe"][str(n)] = row
        print(f"  Nullreihe n = {n} fertig", flush=True)
    out["meta"] = meta(R=R, minlen=MINLEN, hinweis_n2_n3="n = 2: Mindestlänge 1, Fehlbruch = (x1 - x2)^2 / 2 > beta; n = 3: Mindestlänge 1, P(kein Bruch) als Winkelintegral.")
    return out


def check(fen_path: Path, null_path: Path) -> int:
    """Kleine Neurechnung (R = 100, Zufalls-Seeds der Demo, ohne robuste Variante) im Band gegen die gefrorenen Tabellen; der Rückgabewert ist die Zahl der Abweichungen."""
    fz = json.loads(fen_path.read_text(encoding="utf-8"))
    nz = json.loads(null_path.read_text(encoding="utf-8"))
    bad = 0
    # exakte Teile: genau gleich
    for c, row in nz["n2"].items():
        if abs(row["exakt"] - EX.p_break_n2(row["beta"])) > 1e-12:
            bad += 1
            print("n = 2 exakt weicht ab", c)
    for c, row in nz["n3"].items():
        if abs(row["exakt_kein_bruch"] - EX.p_nobreak_n3(row["beta"])) > 1e-12:
            bad += 1
            print("n = 3 exakt weicht ab", c)
    # Simulationsteile: kleine Neurechnung im Band (4 Standardfehler beider Stichproben)
    for key in ("ein|1.0", "zickzack|1.0", "null|0.0"):
        cell = fz["cells"][key]
        R = 100
        X, true = SC.draw_series(cell["scen"], N, cell["delta"], R, 55555)
        new = cell_stats(X, true, MINLEN, False)
        old = np.mean([ps["bekannt"]["p_right"] for ps in cell["per_seed"]], axis=0)
        for j, (a, b) in enumerate(zip(new["p_right"], old)):
            se = math.sqrt(max(b * (1 - b), 0.01) / R + max(b * (1 - b), 0.01) / (R_FULL * len(cell["per_seed"])))
            if abs(a - b) > 4 * se:
                bad += 1
                print(f"{key}: c = {C.C_GRID[j]} neu {a:.3f} gegen gefroren {b:.3f} (4 SE = {4 * se:.3f})")
    print(f"--check: {bad} Abweichungen")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voll", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--schnell", action="store_true", help="nur R = 200 und ohne robuste Variante (Probelauf, schreibt NICHT)")
    a = ap.parse_args()
    fen, nul = ROOT / "sgo_fenster.json", ROOT / "sgo_null_n.json"
    if a.check:
        sys.exit(1 if check(fen, nul) else 0)
    if a.schnell:
        t = time.time()
        make_fenster(200, seeds=SEEDS_FENSTER[:1], quick=True)
        print(f"Probelauf in {time.time() - t:.0f} s")
        return
    if not a.voll:
        ap.error("--voll oder --check angeben")
    t = time.time()
    d1 = make_fenster(R_FULL)
    d2 = make_null(R_FULL)
    fen.write_text(json.dumps(d1, ensure_ascii=False, indent=1), encoding="utf-8")
    nul.write_text(json.dumps(d2, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"geschrieben: {fen.name}, {nul.name} in {time.time() - t:.0f} s")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
