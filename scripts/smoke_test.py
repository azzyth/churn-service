"""Smoke test HTTP untuk container hasil build (dipakai CI, bisa juga lokal).

    python scripts/smoke_test.py http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import sys
import urllib.request

PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.7,
    "TotalCharges": 151.65,
}


def _open(base: str, path: str, data: dict | None = None) -> bytes:
    request = urllib.request.Request(
        base + path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.read()


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

    health = json.loads(_open(base, "/health"))
    assert health["status"] == "ok" and health["model_loaded"] is True, health

    info = json.loads(_open(base, "/model/info"))
    assert 0 < info["threshold"] < 1, info

    page = _open(base, "/ui/").decode("utf-8")
    assert "Churn" in page and "Konsol putusan" in page, "UI tidak tersaji"

    body = json.loads(_open(base, "/predict", PAYLOAD))
    assert abs(body["churn_probability"] - 0.691377) < 1e-4, body
    assert body["churn_prediction"] == 1, body

    print(
        f"smoke ok: model={info['model_name']} threshold={info['threshold']} "
        f"probability={body['churn_probability']} ui=200"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
