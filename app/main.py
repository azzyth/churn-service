"""API inferensi churn (FastAPI).

Jalankan:
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

Model dimuat sekali saat startup. Kalau artefak tidak ada, aplikasi gagal
startup dengan pesan yang jelas (bukan mengembalikan prediksi acak).
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import (
    BatchRequest,
    BatchResponse,
    CustomerFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from src.config import ensure_parent, load_config, project_path
from src.features.build import FEATURE_COLUMNS
from src.train import load_artifact

logger = logging.getLogger("churn-service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CONFIG = load_config()
DEFAULT_THRESHOLD = 0.5


def _frame(records: list[dict]) -> pd.DataFrame:
    """Ubah record tervalidasi menjadi DataFrame dengan urutan fitur yang benar."""
    frame = pd.DataFrame(records)
    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Kolom wajib hilang: {missing}")
    return frame[FEATURE_COLUMNS]


def _predict(records: list[dict]) -> list[PredictionResponse]:
    state = app.state
    probabilities = state.model.predict_proba(_frame(records))[:, 1]
    threshold = float(state.threshold)

    responses = [
        PredictionResponse(
            customer_id=record.get("customerID"),
            churn_probability=round(float(probability), 6),
            churn_prediction=int(probability >= threshold),
            threshold=threshold,
            model_version=str(state.meta.get("version", "unknown")),
        )
        for record, probability in zip(records, probabilities)
    ]
    _log_predictions(responses)
    return responses


def _log_predictions(responses: list[PredictionResponse]) -> None:
    """Simpan log prediksi (JSONL) untuk analisis drift & audit.

    shortcut: file tumbuh tanpa rotasi (~200 byte per prediksi); tambahkan
    rotasi saat file melewati ~1 GB atau log mulai dipakai untuk retraining.
    """
    if not responses:
        return
    try:
        path = ensure_parent(project_path(CONFIG, "predictions_log"))
        with path.open("a", encoding="utf-8") as handle:
            for response in responses:
                handle.write(
                    json.dumps(
                        {
                            "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            **response.model_dump(),
                        }
                    )
                    + "\n"
                )
    except OSError as exc:  # logging tidak boleh menggagalkan prediksi
        logger.warning("Gagal menulis log prediksi: %s", exc)


@asynccontextmanager
async def lifespan(application: FastAPI):
    model_path = project_path(CONFIG, "model_file")
    try:
        model, meta = load_artifact(CONFIG)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{exc} Artefak tidak ditemukan di {model_path}; jalankan `python -m src.train` lalu "
            "`python -m src.evaluate` sebelum menyalakan API."
        ) from exc
    except Exception as exc:  # artefak rusak / beda versi library
        raise RuntimeError(
            f"Gagal memuat artefak model dari {model_path}: {exc.__class__.__name__}: {exc}. "
            "Latih ulang dengan `python -m src.train`."
        ) from exc

    if meta.get("threshold") is None:
        logger.warning(
            "model_meta.json belum punya threshold (evaluate belum dijalankan); "
            "API memakai default %.2f.",
            DEFAULT_THRESHOLD,
        )

    application.state.model = model
    application.state.meta = meta
    application.state.threshold = float(meta.get("threshold") or DEFAULT_THRESHOLD)
    logger.info(
        "Model dimuat: %s v%s (threshold=%.2f)",
        meta.get("model_name"),
        meta.get("version"),
        application.state.threshold,
    )
    yield
    application.state.model = None


app = FastAPI(
    title="Churn Prediction Service",
    description="Prediksi probabilitas churn pelanggan telekomunikasi (Telco Customer Churn).",
    version=CONFIG["project"]["version"],
    lifespan=lifespan,
)

# Satu origin untuk UI dan API: tidak perlu CORS, dan Docker tidak perlu service kedua.
app.mount("/ui", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")


@app.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    return RedirectResponse(url="/ui/", status_code=307)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=getattr(app.state, "model", None) is not None)


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    meta = app.state.meta
    return ModelInfoResponse(
        model_name=str(meta.get("model_name", "unknown")),
        version=str(meta.get("version", "unknown")),
        trained_at=meta.get("trained_at"),
        threshold=float(app.state.threshold),
        selection_reason=meta.get("selection_reason"),
        training_rows=meta.get("training_rows"),
        environment=meta.get("environment", {}),
        test_metrics=meta.get("test_metrics"),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures) -> PredictionResponse:
    return _predict([customer.model_dump(mode="json")])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest) -> BatchResponse:
    records = [customer.model_dump(mode="json") for customer in request.customers]
    responses = _predict(records)
    return BatchResponse(count=len(responses), predictions=responses)
