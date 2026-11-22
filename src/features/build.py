"""Definisi fitur dan pipeline preprocessing.

Semua transformasi dibungkus di dalam sklearn Pipeline sehingga:
- tidak ada informasi dari data validasi/test yang bocor ke training,
- artefak yang disimpan sudah berisi preprocessing + model (satu file).
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# SeniorCitizen sudah 0/1; diperlakukan sebagai numerik agar pipeline sederhana.
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
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
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def numeric_pipeline(scale: bool = True) -> Pipeline:
    """Imputasi median, lalu scaling bila model membutuhkannya."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps=steps)


def categorical_pipeline() -> Pipeline:
    """Imputasi modus + one-hot. Kategori tak dikenal menjadi vektor nol."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # handle_unknown="ignore" -> kategori baru saat inference tidak bikin error,
            # hanya menghasilkan vektor nol untuk kolom itu.
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def build_preprocessor(*, scale_numeric: bool = True) -> ColumnTransformer:
    """ColumnTransformer: numerik (imputasi + scaling opsional) dan kategorikal (one-hot)."""
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline(scale_numeric), NUMERIC_FEATURES),
            ("cat", categorical_pipeline(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator, *, scale_numeric: bool = True) -> Pipeline:
    """Gabungkan preprocessing + estimator menjadi satu Pipeline.

    `scale_numeric=False` dipakai untuk model berbasis pohon (tidak butuh scaling),
    tetapi urutan kolom tetap identik karena memakai definisi fitur yang sama.
    """
    preprocessor = build_preprocessor(scale_numeric=scale_numeric)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])


def make_xy(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Pisahkan fitur dan target dari sebuah split."""
    target = config["data"]["target"]
    missing = [c for c in FEATURE_COLUMNS + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan di split: {missing}")
    X = df[FEATURE_COLUMNS].copy()
    y = df[target].astype(int)
    return X, y


def feature_hash() -> str:
    """Hash definisi fitur, dicatat di model_meta.json supaya perubahan terdeteksi."""
    payload = json.dumps(
        {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
