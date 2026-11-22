"""Test perhitungan PSI dan skenario drift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build import FEATURE_COLUMNS
from src.monitor.drift import drift_report, psi_categorical, psi_numeric


@pytest.fixture(scope="module")
def reference():
    from src.config import load_config, project_path

    config = load_config()
    return pd.read_csv(project_path(config, "splits_dir") / "train.csv").head(1000)


@pytest.fixture(scope="module")
def project_config():
    from src.config import load_config

    return load_config()


def test_psi_zero_for_identical_data(reference):
    psi = psi_numeric(reference["tenure"], reference["tenure"])
    assert psi == pytest.approx(0.0, abs=1e-3)


def test_psi_low_for_reshuffled_same_distribution(reference):
    """Sampel acak dari distribusi yang sama harus tetap stabil (< 0.10)."""
    shuffled = reference.sample(frac=1.0, random_state=7)
    psi = psi_numeric(reference["tenure"], shuffled["tenure"])
    assert psi < 0.10


def test_psi_high_for_shifted_numeric(reference):
    shifted = reference.copy()
    shifted["MonthlyCharges"] = shifted["MonthlyCharges"] * 2
    psi = psi_numeric(reference["MonthlyCharges"], shifted["MonthlyCharges"])
    assert psi > 0.20


def test_psi_flags_batch_with_no_numeric_values(reference, project_config):
    """Regresi: batch yang kolom numeriknya kosong semua dulu dilaporkan PSI 0 (OK)."""
    blank = reference.copy()
    blank["TotalCharges"] = np.nan
    report = drift_report(reference, blank, project_config).set_index("feature")
    assert report.loc["TotalCharges", "status"] == "ALERT"


def test_psi_high_for_changed_categorical(reference):
    shifted = reference.copy()
    shifted["Contract"] = "Month-to-month"
    psi = psi_categorical(reference["Contract"], shifted["Contract"])
    assert psi > 0.20


def test_psi_detects_shift_from_constant_reference():
    """Regresi: referensi konstan dulu membuat PSI selalu 0 (drift tak terdeteksi)."""
    constant = pd.Series([5.0] * 200)
    shifted = pd.Series([50.0] * 200)
    assert psi_numeric(constant, constant) == pytest.approx(0.0, abs=1e-3)
    assert psi_numeric(constant, shifted) > 0.20


def test_report_rejects_missing_columns(reference, project_config):
    """Batch tanpa kolom wajib harus memberi pesan jelas, bukan KeyError mentah."""
    broken = reference.drop(columns=["Contract"])
    with pytest.raises(ValueError, match="Contract"):
        drift_report(reference, broken, project_config)


def test_report_covers_all_features(reference, project_config):
    batch = reference.copy()
    batch["tenure"] = 0
    report = drift_report(reference, batch, project_config)

    assert set(report["feature"]) == set(FEATURE_COLUMNS)
    assert set(report["status"]) <= {"OK", "WARN", "ALERT"}
    flagged = report[report["status"] == "ALERT"]
    assert "tenure" in set(flagged["feature"])
