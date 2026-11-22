"""Test kontrak & konsistensi API."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.config import project_path
from src.features.build import FEATURE_COLUMNS
from src.train import load_artifact


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info_exposes_threshold_and_metrics(client, config):
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert 0 < body["threshold"] < 1
    assert body["model_name"] in {"logistic_regression", "lightgbm", "dummy"}
    assert body["environment"]["python"]


def test_predict_valid_customer(client, sample_payloads):
    response = client.post("/predict", json=sample_payloads[0])
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)
    assert body["customer_id"] == sample_payloads[0]["customerID"]


@pytest.mark.parametrize(
    "field, value",
    [
        ("tenure", -1),
        ("tenure", 73),
        ("MonthlyCharges", 0),
        ("TotalCharges", -5),
        ("Contract", "Three year"),
        ("PaymentMethod", "Cash"),
        ("SeniorCitizen", 2),
    ],
)
def test_predict_rejects_invalid_input(client, sample_payloads, field, value):
    payload = copy.deepcopy(sample_payloads[0])
    payload[field] = value
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, f"{field}={value} seharusnya ditolak"


def test_predict_rejects_extra_field(client, sample_payloads):
    payload = copy.deepcopy(sample_payloads[0])
    payload["unknown_column"] = 1
    assert client.post("/predict", json=payload).status_code == 422


def test_batch_matches_single(client, sample_payloads):
    batch = client.post("/predict/batch", json={"customers": sample_payloads[:5]})
    assert batch.status_code == 200
    body = batch.json()
    assert body["count"] == 5
    assert len(body["predictions"]) == 5

    single = client.post("/predict", json=sample_payloads[0]).json()
    assert body["predictions"][0]["churn_probability"] == pytest.approx(
        single["churn_probability"], abs=1e-6
    )


def test_api_matches_pipeline_predictions(client, config, sample_payloads):
    """Prediksi lewat API harus identik dengan pipeline yang dipanggil langsung."""
    model, meta = load_artifact(config)
    frame = pd.DataFrame(sample_payloads)[FEATURE_COLUMNS]
    direct = model.predict_proba(frame)[:, 1]

    response = client.post("/predict/batch", json={"customers": sample_payloads})
    api_scores = np.array([item["churn_probability"] for item in response.json()["predictions"]])

    np.testing.assert_allclose(api_scores, direct, rtol=0, atol=1e-6)
    assert all(item["threshold"] == meta["threshold"] for item in response.json()["predictions"])


def test_predict_logs_written(client, config, sample_payloads):
    """Setiap prediksi dicatat ke logs/predictions.jsonl untuk audit & drift."""
    log_path = project_path(config, "predictions_log")
    before = len(log_path.read_text(encoding="utf-8").splitlines()) if log_path.exists() else 0

    assert client.post("/predict", json=sample_payloads[0]).status_code == 200

    assert log_path.exists(), "log prediksi tidak dibuat"
    after = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(after) == before + 1


def test_ui_page_is_served(client):
    """UI statis disajikan dari origin yang sama dengan API (tanpa CORS)."""
    response = client.get("/ui/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Churn" in response.text
    assert 'src="app.js"' in response.text


def test_ui_assets_are_served(client):
    assert client.get("/ui/styles.css").status_code == 200
    assert client.get("/ui/app.js").status_code == 200
    assert client.get("/ui/fonts/silkscreen-400-latin.woff2").status_code == 200


def test_root_redirects_to_ui(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/ui/"


def test_predict_accepts_valid_high_total_charges(client, sample_payloads):
    """Regresi: batas TotalCharges 10.000 dulu menolak input yang sah (72 bln)."""
    payload = copy.deepcopy(sample_payloads[0])
    payload["tenure"] = 72
    payload["MonthlyCharges"] = 118.75
    payload["TotalCharges"] = 12000.0
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
