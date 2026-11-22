"""Test keputusan bisnis di evaluate.py: cost curve dan pemilihan threshold."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate import cost_curve, pick_threshold


def _config(min_precision: float = 0.0) -> dict:
    return {
        "business": {
            "cost_false_negative": 5.0,
            "cost_false_positive": 1.0,
            "min_precision": min_precision,
        }
    }


# Skor 0,31 / 0,30 sengaja berdekatan supaya biaya minimum jatuh tepat di satu
# titik grid (0,31) dan tidak menyebar ke rentang threshold lain.
def _tiny_case() -> tuple[np.ndarray, np.ndarray]:
    return np.array([1, 0]), np.array([0.31, 0.30])


def test_cost_curve_counts_and_cost_formula():
    y_true, y_score = _tiny_case()
    curve = cost_curve(y_true, y_score, _config())

    assert len(curve) == 91  # grid 0,05 .. 0,95

    strict = curve.loc[curve["threshold"] == 0.31].iloc[0]
    assert (strict["tp"], strict["fp"], strict["fn"], strict["tn"]) == (1, 0, 0, 1)
    assert strict["cost"] == pytest.approx(0.0)

    loose = curve.loc[curve["threshold"] == 0.30].iloc[0]
    assert (loose["tp"], loose["fp"], loose["fn"], loose["tn"]) == (1, 1, 0, 0)
    assert loose["cost"] == pytest.approx(1.0)  # 0 FN x 5 + 1 FP x 1


def test_pick_threshold_returns_minimum_cost_row():
    y_true, y_score = _tiny_case()
    curve = cost_curve(y_true, y_score, _config())

    chosen = pick_threshold(curve, _config())

    assert chosen["threshold"] == pytest.approx(0.31)
    assert chosen["cost"] == pytest.approx(curve["cost"].min())


def test_min_precision_filters_candidates():
    """Baris termurah presisinya 0,20; batas 0,50 harus memaksa baris presisi 0,80."""
    curve = pd.DataFrame(
        [
            {"threshold": 0.05, "cost": 10.0, "precision": 0.20},
            {"threshold": 0.30, "cost": 50.0, "precision": 0.80},
        ]
    )

    assert pick_threshold(curve, _config())["threshold"] == pytest.approx(0.05)
    assert pick_threshold(curve, _config(0.50))["threshold"] == pytest.approx(0.30)


def test_min_precision_impossible_falls_back_to_cheapest():
    curve = pd.DataFrame(
        [
            {"threshold": 0.05, "cost": 10.0, "precision": 0.20},
            {"threshold": 0.30, "cost": 50.0, "precision": 0.40},
        ]
    )

    assert pick_threshold(curve, _config(0.99))["threshold"] == pytest.approx(0.05)
