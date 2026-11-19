"""Recalibração post hoc do Estudo 3 (analysis/recalibration.py): empates de q."""

import importlib.util
import random
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "analysis" / "recalibration.py"
    spec = importlib.util.spec_from_file_location("recalibration", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tied_q_are_pooled():
    """q empatados com desfechos mistos recebem a taxa do grupo, não a dos negativos."""
    rec = _module()
    train = [(0.1, i % 50 == 0) for i in range(5000)]
    assert rec.apply(rec.isotonic(train), 0.1) == pytest.approx(100.5 / 5001)


def test_one_edge_per_block():
    """Cada limite de q pertence a um único bloco (sem blocos duplicados por empate)."""
    rec = _module()
    rng = random.Random(7)
    pts = [(round(rng.random(), 2), rng.random() < 0.3) for _ in range(300)]
    edges, _ = rec.isotonic(pts)
    assert edges == sorted(edges) and len(set(edges)) == len(edges)
