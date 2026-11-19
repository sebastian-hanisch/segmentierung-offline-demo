"""Zugriff auf die gefrorenen Tabellen (Quelle und Stand je Datei im Feld `meta`):

* `sgo_fenster.json`   (tools/sgo_make_fenster.py): Experiment 2, drei Zufalls-Seeds je Zelle, R = 1000 Läufe je Seed;
* `sgo_null_n.json`    (tools/sgo_make_fenster.py): Experiment 3, n = 2 und n = 3 exakt neben Simulation, Fehlbruch-Anteil gegen n;
* `sgo_online.json`    (tools/sgo_make_online.py): Experiment 5, h, ARL, exakte Verzögerung, Simulation R = 4000, Phase I;
* `sgo_verfahren.json` (tools/sgo_make_verfahren.py): Experiment 4, PELT und binäre Segmentierung gegen die optimale Zerlegung.

Simulationsanteile sind Bänder (gelb), exakte Teile genau (grün)."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

import sgo_constants as C

HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def fenster() -> dict:
    return _load("sgo_fenster.json")


def null_n() -> dict:
    return _load("sgo_null_n.json")


def online() -> dict:
    return _load("sgo_online.json")


def verfahren() -> dict:
    return _load("sgo_verfahren.json")


def h_cusum() -> float:
    return float(online()["exakt"]["h"])


def cell_key(scen_key: str, delta: float) -> str:
    return f"{scen_key}|{delta:.1f}"


def fenster_cell(scen_key: str, delta: float):
    """Zelle der Fenster-Tabelle (None, wenn δ nicht gemessen ist; die Nullreihe hat nur δ = 0)."""
    d = 0.0 if scen_key == "null" else float(delta)
    return fenster()["cells"].get(cell_key(scen_key, d))


def fenster_pooled(scen_key: str, delta: float, mode: str = "bekannt"):
    """Zusammenfassung über die drei Zufalls-Seeds: Mittel, kleinster und größter Wert je c für Anteil richtige Bruchzahl, Anteil mit Fehlbruch und mittlere Bruchzahl."""
    cell = fenster_cell(scen_key, delta)
    if cell is None:
        return None
    out = {"c_grid": fenster()["meta"]["c_grid"], "R": fenster()["meta"]["R"], "seeds": [ps["seed"] for ps in cell["per_seed"]], "wahre_brueche": cell["wahre_brueche"]}
    for key in ("p_right", "p_any", "mean_count", "p_loc"):
        arr = np.array([ps[mode][key] for ps in cell["per_seed"]], float)
        out[key] = dict(mean=arr.mean(axis=0).tolist(), lo=arr.min(axis=0).tolist(), hi=arr.max(axis=0).tolist())
    out["loc_error"] = []
    for j in range(len(out["c_grid"])):
        vals = [ps[mode]["loc_error"][j] for ps in cell["per_seed"] if ps[mode]["loc_error"][j] is not None]
        out["loc_error"].append(float(np.mean(vals)) if vals else None)
    if "f1" in cell["per_seed"][0][mode]:
        out["f1"] = {m: float(np.mean([ps[mode]["f1"][m] for ps in cell["per_seed"]])) for m in cell["per_seed"][0][mode]["f1"]}
    out["window"] = [c for c, v in zip(out["c_grid"], out["p_right"]["mean"]) if v >= C.WINDOW_SHARE]
    return out
