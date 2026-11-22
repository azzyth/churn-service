"""Validasi skema dataset sebelum masuk pipeline modelling.

Setiap pemeriksaan mengembalikan satu baris hasil. Pipeline berhenti kalau
ada pelanggaran, supaya data buruk tidak diam-diam dipakai melatih model.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = [
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
    "Churn",
]

CATEGORY_VALUES: dict[str, set[str]] = {
    "gender": {"Female", "Male"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}

RANGES: dict[str, tuple[float, float]] = {
    "tenure": (0, 72),
    "MonthlyCharges": (0.0, 200.0),
    # Batas domain: 72 bulan x 200 = 14.400, dibulatkan ke atas.
    "TotalCharges": (0.0, 15_000.0),
    "SeniorCitizen": (0, 1),
}


def _result(check: str, violations: int, detail: str) -> dict:
    return {
        "check": check,
        "violations": int(violations),
        "status": "PASS" if violations == 0 else "FAIL",
        "detail": detail,
    }


def validate(df: pd.DataFrame, target: str = "Churn", id_column: str = "customerID") -> list[dict]:
    """Jalankan semua pemeriksaan, kembalikan daftar hasil per check."""
    results: list[dict] = []

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    results.append(
        _result("kolom_wajib", len(missing_columns), f"hilang: {missing_columns or '-'}")
    )

    duplicate_ids = int(df[id_column].duplicated().sum()) if id_column in df.columns else 0
    results.append(_result("id_unik", duplicate_ids, f"{id_column} duplikat"))

    total_missing = int(df.isna().sum().sum())
    missing_detail = {k: int(v) for k, v in df.isna().sum().items() if v}
    results.append(
        _result("tanpa_missing_value", total_missing, f"per kolom: {missing_detail or '-'}")
    )

    if target in df.columns:
        bad_target = int((~df[target].isin([0, 1])).sum())
    else:
        bad_target = len(df)
    results.append(_result("target_biner", bad_target, f"{target} harus 0/1"))

    for column, (low, high) in RANGES.items():
        if column not in df.columns:
            results.append(_result(f"range_{column}", len(df), "kolom tidak ada"))
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        out_of_range = int(((series < low) | (series > high) | series.isna()).sum())
        results.append(
            _result(f"range_{column}", out_of_range, f"di luar [{low}, {high}]")
        )

    for column, allowed in CATEGORY_VALUES.items():
        if column not in df.columns:
            results.append(_result(f"kategori_{column}", len(df), "kolom tidak ada"))
            continue
        unknown = sorted(set(df[column].dropna().unique()) - allowed)
        count = int(df[column].isin(unknown).sum()) if unknown else 0
        results.append(
            _result(f"kategori_{column}", count, f"nilai asing: {unknown or '-'}")
        )

    # Konsistensi: pelanggan tenure 0 tidak punya tagihan total.
    if {"tenure", "TotalCharges"}.issubset(df.columns):
        inconsistent = int(((df["tenure"] == 0) & (df["TotalCharges"] > 0)).sum())
        results.append(
            _result("konsistensi_tagihan", inconsistent, "tenure=0 tetapi TotalCharges>0")
        )

    return results


def total_violations(results: list[dict]) -> int:
    return sum(r["violations"] for r in results)


def format_results(results: list[dict]) -> str:
    """Tabel teks sederhana untuk dicetak di CLI."""
    check_width = max(len(r["check"]) for r in results) + 2
    lines = [
        f"{'check'.ljust(check_width)}{'violations'.ljust(12)}status   detail",
        "-" * (check_width + 12 + 9 + 40),
    ]
    for row in results:
        lines.append(
            f"{row['check'].ljust(check_width)}"
            f"{str(row['violations']).ljust(12)}"
            f"{row['status'].ljust(9)}"
            f"{row['detail']}"
        )
    return "\n".join(lines)
