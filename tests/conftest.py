"""Gemeinsame Einrichtung: Importpfad der Demo und der Testhilfe `exact_ref`; die Mini-Reihen der Handbeispiele (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


@pytest.fixture
def puls():
    """Puls (0 0 0 2 2 2 0 0 0): Kosten 8, größter Einzelgewinn 2 (Handbeispiel 4)."""
    return [0, 0, 0, 2, 2, 2, 0, 0, 0]


@pytest.fixture
def sprung():
    """(0 0 0 2 2 2): ein Bruch bei 3, Kosten der ganzen Reihe 6 (Handbeispiel 1)."""
    return [0, 0, 0, 2, 2, 2]
