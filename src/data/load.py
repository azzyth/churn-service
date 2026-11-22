"""Muat data mentah dan bersihkan ke bentuk yang siap divalidasi.

Dua masalah nyata dataset ini ditangani di sini:
1. `TotalCharges` disimpan sebagai teks dan berisi string kosong untuk
   pelanggan dengan tenure = 0 (pelanggan baru).
2. `Churn` berupa "Yes"/"No" sehingga perlu dipetakan ke 1/0.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.config import load_config, project_path

RAW_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_raw(config: dict | None = None) -> pd.DataFrame:
    """Baca CSV mentah apa adanya (tanpa transformasi)."""
    config = config or load_config()
    raw_file = project_path(config, "raw_file")
    if not raw_file.exists():
        raise FileNotFoundError(
            f"Data mentah tidak ditemukan: {raw_file}. Jalankan `python -m src.data.download`."
        )
    return pd.read_csv(raw_file)


def clean(raw: pd.DataFrame, config: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Bersihkan data mentah dan kembalikan (dataframe, ringkasan pembersihan)."""
    config = config or load_config()
    target = config["data"]["target"]
    id_column = config["data"]["id_column"]

    # Validasi kolom dijalankan setelah clean, jadi clean sendiri harus gagal
    # dengan pesan jelas (bukan KeyError mentah) kalau sumbernya cacat.
    required_here = [id_column, target, "tenure", "TotalCharges"]
    missing = [column for column in required_here if column not in raw.columns]
    if missing:
        raise ValueError(f"Data mentah tidak punya kolom wajib: {missing}")

    df = raw.copy()

    # 1. Rapikan whitespace pada kolom teks (pandas 3 memakai dtype `str`).
    text_columns = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for column in text_columns:
        df[column] = df[column].astype("string").str.strip()

    # 2. TotalCharges -> numerik. String kosong menjadi NaN, lalu diisi 0
    #    hanya untuk pelanggan dengan tenure = 0 (mereka belum ditagih).
    blank_total = int((df["TotalCharges"].astype("string") == "").sum())
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blank_tenure0 = int(df.loc[df["TotalCharges"].isna(), "tenure"].eq(0).sum())
    df.loc[df["tenure"].eq(0), "TotalCharges"] = df.loc[df["tenure"].eq(0), "TotalCharges"].fillna(0.0)

    # 3. Target -> 0/1.
    mapping = {"Yes": 1, "No": 0}
    df[target] = df[target].map(mapping).astype("Int64")

    stats = {
        "rows_in": int(len(raw)),
        "blank_total_charges": blank_total,
        "blank_total_charges_with_zero_tenure": blank_tenure0,
        "total_charges_imputed_zero": blank_tenure0,
        "remaining_total_charges_na": int(df["TotalCharges"].isna().sum()),
        "remaining_target_na": int(df[target].isna().sum()),
        "id_unique": int(df[id_column].nunique()),
    }
    return df, stats


def sha256_file(path: str | Path) -> str:
    """SHA-256 satu file, dibaca per 1 MiB agar CSV besar tidak masuk memori."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
