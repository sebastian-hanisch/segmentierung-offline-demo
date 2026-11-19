"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Schnellstarts.

Modell: Mittelwertsprünge, Normalrauschen mit **bekannter Streuung 1**, Beobachtungen unabhängig; Zähleinheiten Läufe (simulierte Reihen), Zerlegungen,
Kostenauswertungen. Ein Bruch bei s heißt: s Beobachtungen liegen vor der Änderung (ν-Konvention)."""

import math

# --- Szenarien (Vehikel A: ausgedachte Normalreihen) -----------------------------------------------------------------------------------------
SZ_NULL = "Nullreihe (kein Bruch)"
SZ_EIN = "Ein Bruch (in der Mitte)"
SZ_TREPPE = "Treppe (zwei Sprünge nach oben)"
SZ_PULS = "Puls (hoch und zurück)"
SZ_ZICK = "Zickzack (drei Sprünge)"
SZ_EIGEN = "Eigene Reihe (Feld darunter)"
SCEN_FIX = (SZ_NULL, SZ_EIN, SZ_TREPPE, SZ_PULS, SZ_ZICK)
SCEN_ALL = SCEN_FIX + (SZ_EIGEN,)
SCEN_KEY = {SZ_NULL: "null", SZ_EIN: "ein", SZ_TREPPE: "treppe", SZ_PULS: "puls", SZ_ZICK: "zickzack"}
# Lage der wahren Brüche als Anteil von n (bei n = 200: 100 / 70, 140 / 80, 120 / 50, 100, 150, wie in der Vormessung)
SCEN_FRAC = {"null": (), "ein": (0.5,), "treppe": (0.35, 0.7), "puls": (0.4, 0.6), "zickzack": (0.25, 0.5, 0.75)}

# --- Experiment 1: eine Reihe im Rückblick -----------------------------------------------------------------------------------------------------
DELTA1_MIN, DELTA1_MAX, DELTA1_STEP, DEFAULT_DELTA1 = 0.5, 4.0, 0.5, 3.0
N1_MIN, N1_MAX, N1_STEP, DEFAULT_N1 = 20, 400, 10, 200
OWN_MAX = 400                                    # Länge der eigenen Reihe
OWN_TEXT_MAX = 6000                              # Zeichen des Eingabefeldes
DEFAULT_OWN = "0 0 0 2 2 2 0 0 0"
MODUS_C, MODUS_BETA = "c · ln n", "β absolut"
MODI = (MODUS_C, MODUS_BETA)
C1_MIN, C1_MAX, C1_STEP, DEFAULT_C1 = 0.5, 10.0, 0.5, 2.0
BETA1_MIN, BETA1_MAX, BETA1_STEP, DEFAULT_BETA1 = 0.0, 20.0, 0.5, 3.0
MINLEN_MIN, MINLEN_MAX, DEFAULT_MINLEN1 = 1, 20, 5
M_OPT, M_PELT, M_PELT_V, M_BIN = "optimal (DP)", "PELT", "PELT verzögert", "binär"
METHODEN = (M_OPT, M_PELT, M_PELT_V, M_BIN)
DEFAULT_MODUS, DEFAULT_METHODE = MODUS_C, M_OPT

# --- Experiment 2: die Strafe ist ein Fenster --------------------------------------------------------------------------------------------------
DELTA2_OPTIONS, DEFAULT_DELTA2 = (1.0, 2.0, 3.0), 1.0
N2_OPTIONS, DEFAULT_N2 = (50, 100, 150, 200), 200
MINLEN2_OPTIONS, DEFAULT_MINLEN2 = (1, 2, 5, 10, 20), 5
R2_MIN, R2_MAX, R2_STEP, DEFAULT_R2 = 50, 200, 50, 100
SIGMA_BEKANNT, SIGMA_ROBUST = "bekannt (1)", "robust geschätzt"
SIGMAS = (SIGMA_BEKANNT, SIGMA_ROBUST)
DEFAULT_SZEN2, DEFAULT_SIGMA = SZ_ZICK, SIGMA_BEKANNT
C_GRID = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0)    # Strafkonstanten der Fenster-Tabelle (β = c · ln n)
TOL_M = (1, 2, 3, 5, 10, 20, 30, 40, 60)         # Toleranzen M des F1
WINDOW_SHARE = 0.9                               # „richtige Bruchzahl“ mindestens 90 % der Läufe

# --- Experiment 3: die Nullreihe exakt ---------------------------------------------------------------------------------------------------------
N3_OPTIONS, DEFAULT_N3 = (2, 3, 4, 10, 50, 100, 200, 400, 800), 3
C3_OPTIONS, DEFAULT_C3 = (0.5, 1.0, 2.0, 3.0), 2.0
R3_OPTIONS, DEFAULT_R3 = (50, 100, 200, 500, 1000, 5000, 10000), 5000
LIVE_LIMIT_NULL = 40_000                         # Obergrenze n · R (Zufallszahlen eines Live-Laufs der Nullreihe)


def null_minlen(n: int) -> int:
    """Mindestlänge der Nullreihen-Läufe: 1 bis n = 10 (dort gilt die exakte Rechnung für n = 2, 3), sonst 5 (wie V16 D3)."""
    return 1 if n <= 10 else 5


# --- Experiment 4: schnell gegen optimal --------------------------------------------------------------------------------------------------------
DEFAULT_SZEN4 = SZ_PULS
MINLEN4_OPTIONS, DEFAULT_MINLEN4 = (1, 2, 3, 5), 5
C4_OPTIONS, DEFAULT_C4 = (1.0, 2.0, 4.0), 2.0
R4_MIN, R4_MAX, R4_STEP, DEFAULT_R4 = 20, 100, 20, 100
N4 = 200                                         # Reihenlänge der Läufe in Experiment 4

# --- Experiment 5: dieselbe Reihe, zwei Fragen --------------------------------------------------------------------------------------------------
DELTA5_OPTIONS, DEFAULT_DELTA5 = (1.0, 2.0), 1.0
NU5_MIN, NU5_MAX, NU5_STEP, DEFAULT_NU5 = 20, 180, 10, 100
R5_MIN, R5_MAX, R5_STEP, DEFAULT_R5 = 50, 200, 50, 100
N5 = 200                                         # Reihenlänge in Experiment 5
BREAK_PENALTY_C5, MINLEN5 = 2.0, 5               # offline: c = 2, Mindestlänge 5
CUSUM_K = 0.5
ARL0_TARGET = 370.398

SEED_MAX, DEFAULT_SEED = 999999, 35
LOC_TOL = 5                                      # „Lage innerhalb ±5 Beobachtungen“

COLOR_SERIES = "#4c78a8"
COLOR_TRUE = "rgba(90,90,90,0.9)"
COLOR_FOUND = "#c0392b"
COLOR_MEAN = "#2F6B65"
COLOR_FROZEN = "rgba(90,90,90,0.9)"
COLOR_LIVE = "#c0392b"
COLOR_ONLINE = "#d68910"
COLOR_OFFLINE = "#2F6B65"


def beta_of(c: float, n: int) -> float:
    """Strafe β = c · ln n (c = 2 heißt hier „BIC-Wahl“: eine Setzung dieses Berichts, keine Quellenaussage)."""
    return c * math.log(n) if n > 1 else 0.0


# --- Schnellstarts ----------------------------------------------------------------------------------------------------------------------------
# Jeder Schnellstart nennt nur die Regler, die er ändert; alle übrigen kehren zum Standard zurück (= erster Eintrag).
PRESETS = {
    "Ein Bruch, klar (δ = 3)": {},
    "Ein Bruch, schwach (δ = 1)": {"delta1": 1.0},
    "Nullreihe: Fehlbrüche (c = 1)": {"szen1": SZ_NULL, "c1": 1.0, "szen2": SZ_NULL, "n3": 200, "c3": 1.0, "r3": 200},
    "Zickzack δ = 1: kein c passt": {"szen1": SZ_ZICK, "delta1": 1.0},
    "Puls von Hand: binäre Segmentierung blind (β = 3)": {"szen1": SZ_EIGEN, "reihe": DEFAULT_OWN, "modus": MODUS_BETA, "beta1": 3.0, "minlen1": 1, "methode": M_BIN},
    "PELT-Gegenbeispiel (n = 5, Mindestlänge 2)": {"szen1": SZ_EIGEN, "reihe": "2 1 0 0 2", "modus": MODUS_BETA, "beta1": 1.0, "minlen1": 2, "methode": M_PELT, "minlen4": 2},
    "Online gegen offline (δ = 1)": {"delta1": 1.0, "delta2": 1.0, "delta5": 1.0, "r5": 200},
    "Streuung unbekannt: Zickzack δ = 1": {"szen1": SZ_ZICK, "delta1": 1.0, "sigma2": SIGMA_ROBUST},
}
PRESET_HELP = {
    "Ein Bruch, klar (δ = 3)": "Voreinstellung: eine Reihe mit n = 200 Beobachtungen und einem Sprung der Höhe 3 in der Mitte; optimale Zerlegung mit c = 2 und Mindestlänge 5.",
    "Ein Bruch, schwach (δ = 1)": "Derselbe Bruch mit Sprunghöhe 1: im Rauschen der Streuung 1 ist er schwerer zu finden, die Lage streut stärker.",
    "Nullreihe: Fehlbrüche (c = 1)": "Eine Reihe ohne jeden Bruch und die kleine Strafe c = 1: die Suche meldet trotzdem Brüche (Experiment 1, 2 und 3).",
    "Zickzack δ = 1: kein c passt": "Drei Sprünge der Höhe 1: kein c in der Tabelle erreicht 90 % richtige Bruchzahl (Experiment 2, Zickzack; n = 200, Mindestlänge 5).",
    "Puls von Hand: binäre Segmentierung blind (β = 3)": "Die Reihe 0 0 0 2 2 2 0 0 0 mit absoluter Strafe 3: die optimale Zerlegung findet beide Brüche, die binäre Segmentierung keinen (Experiment 1).",
    "PELT-Gegenbeispiel (n = 5, Mindestlänge 2)": "Die Reihe 2 1 0 0 2 mit Mindestlänge 2 und β = 1: das gewöhnliche PELT weicht von der optimalen Zerlegung ab (Experiment 1 und 4).",
    "Online gegen offline (δ = 1)": "Sprung δ = 1 bei ν = 100: der CUSUM-Alarm und der Rückblick auf dieselbe Reihe, mit 200 Live-Läufen (Experiment 5).",
    "Streuung unbekannt: Zickzack δ = 1": "Zickzack mit Sprung 1, die Streuung wird robust aus den Differenzen geschätzt statt als 1 vorausgesetzt (Experiment 2).",
}
