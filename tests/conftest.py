"""Fixture bersama untuk test suite.

Test butuh artefak model. Kalau belum ada, model dilatih sekali di awal
supaya `pytest` tetap bisa dijalankan dari repo bersih.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, project_path  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def ensure_artifact():
    config = load_config()
    if not project_path(config, "model_file").exists():
        from src.train import main as train_main

        assert train_main() == 0, "Gagal melatih model untuk test"


@pytest.fixture(scope="session")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def sample_payloads(config: dict) -> list[dict]:
    """20 pelanggan asli dari test split, dalam bentuk payload API."""
    test_df = pd.read_csv(project_path(config, "splits_dir") / "test.csv").head(20)
    payloads = json.loads(test_df.to_json(orient="records"))
    allowed = {
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
    }
    return [{key: value for key, value in payload.items() if key in allowed} for payload in payloads]
