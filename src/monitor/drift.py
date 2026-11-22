"""Monitoring drift ringan dengan Population Stability Index (PSI).

PSI membandingkan distribusi batch baru terhadap distribusi data referensi
(train split). Tidak butuh library tambahan.

Ambang yang dipakai (config.monitoring):
    PSI < 0.10  -> stabil
    0.10 - 0.20 -> warning
    >= 0.20     -> alert (distribusi bergeser, model perlu dievaluasi ulang)

Pemakaian:
    python -m src.monitor.drift                       # jalankan skenario normal + drifted
    python -m src.monitor.drift --batch path/ke.csv   # satu batch saja
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src.config import dump_json, load_config, project_path, set_seed
from src.features.build import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

EPSILON = 1e-6


def _shares(values: np.ndarray, breakpoints: np.ndarray) -> np.ndarray:
    counts = np.histogram(values, bins=breakpoints)[0].astype(float)
    total = counts.sum()
    if total == 0:
        return np.full(len(counts), EPSILON)
    return np.clip(counts / total, EPSILON, None)


def _require_columns(frame: pd.DataFrame, label: str) -> None:
    """Pastikan semua fitur model ada sebelum menghitung apa pun."""
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} tidak punya kolom wajib: {missing}")


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI untuk fitur numerik memakai bin kuantil dari data referensi."""
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if ref.size == 0:
        return 0.0
    if cur.size == 0:
        # Batch tanpa satu pun nilai numerik bukan "stabil", hanya tak bisa
        # dibandingkan. Kembalikan divergensi tak hingga -> ALERT, bukan OK.
        return float("inf")

    if float(np.ptp(ref)) < 1e-9:
        # Referensi konstan: bin kuantil akan kolaps menjadi satu bin, dan PSI
        # jadi selalu 0 walaupun batch bergeser jauh. Pakai bin sempit di
        # sekitar nilai konstan itu supaya pergeseran tetap terdeteksi.
        center = float(ref[0])
        epsilon = max(abs(center) * 1e-6, 1e-9)
        breakpoints = np.array([-np.inf, center - epsilon, center + epsilon, np.inf])
    else:
        breakpoints = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        if breakpoints.size < 3:  # data nyaris konstan (mis. fitur biner)
            breakpoints = np.unique(np.concatenate([breakpoints, [ref.max() + 1e-9]]))
        breakpoints = np.concatenate([[-np.inf], breakpoints[1:-1], [np.inf]])

    ref_share = _shares(ref, breakpoints)
    cur_share = _shares(cur, breakpoints)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    """PSI untuk fitur kategorikal: bandingkan proporsi tiap kategori."""
    ref_counts = reference.astype("string").fillna("__missing__").value_counts(normalize=True)
    cur_counts = current.astype("string").fillna("__missing__").value_counts(normalize=True)
    categories = ref_counts.index.union(cur_counts.index)
    ref = ref_counts.reindex(categories, fill_value=0).to_numpy(dtype=float)
    cur = cur_counts.reindex(categories, fill_value=0).to_numpy(dtype=float)
    ref = np.clip(ref, EPSILON, None)
    cur = np.clip(cur, EPSILON, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def status_for(psi: float, config: dict) -> str:
    monitoring = config["monitoring"]
    if psi >= float(monitoring["psi_alert"]):
        return "ALERT"
    if psi >= float(monitoring["psi_warn"]):
        return "WARN"
    return "OK"


def drift_report(reference: pd.DataFrame, batch: pd.DataFrame, config: dict) -> pd.DataFrame:
    _require_columns(reference, "Data referensi")
    _require_columns(batch, "Batch")
    bins = int(config["monitoring"]["psi_bins"])
    rows = []
    for feature in NUMERIC_FEATURES:
        psi = psi_numeric(reference[feature], batch[feature], bins)
        rows.append({"feature": feature, "tipe": "numerik", "psi": round(psi, 4)})
    for feature in CATEGORICAL_FEATURES:
        psi = psi_categorical(reference[feature], batch[feature])
        rows.append({"feature": feature, "tipe": "kategorikal", "psi": round(psi, 4)})

    report = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    report["status"] = report["psi"].apply(lambda value: status_for(value, config))
    return report


def positive_rate(batch: pd.DataFrame, config: dict) -> dict:
    """Proporsi prediksi churn positif pada batch (butuh artefak model)."""
    _require_columns(batch, "Batch")
    try:
        from src.train import load_artifact
    except Exception:  # noqa: BLE001
        return {"available": False}

    try:
        model, meta = load_artifact(config)
    except FileNotFoundError:
        return {"available": False}

    threshold = float(meta.get("threshold") or 0.5)
    frame = batch[FEATURE_COLUMNS]
    probabilities = model.predict_proba(frame)[:, 1]
    return {
        "available": True,
        "threshold": threshold,
        "positive_rate": round(float((probabilities >= threshold).mean()), 4),
        "mean_probability": round(float(probabilities.mean()), 4),
    }


def make_samples(config: dict) -> dict[str, str]:
    """Buat batch contoh: normal (dari test split) dan drifted (digeser)."""
    set_seed(config["seed"])
    splits_dir = project_path(config, "splits_dir")
    processed_dir = project_path(config, "processed_dir")
    processed_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(splits_dir / "test.csv")
    train_df = pd.read_csv(splits_dir / "train.csv")

    reference = train_df.sample(n=min(1500, len(train_df)), random_state=config["seed"])
    normal = test_df.sample(n=min(800, len(test_df)), random_state=config["seed"] + 1)

    drifted = normal.copy()
    drifted["tenure"] = (drifted["tenure"] * 0.25).round().clip(lower=0)
    drifted["MonthlyCharges"] = (drifted["MonthlyCharges"] * 1.6).clip(upper=200)
    drifted["Contract"] = "Month-to-month"
    drifted["PaymentMethod"] = "Electronic check"
    drifted["InternetService"] = "Fiber optic"

    paths = {
        "reference": processed_dir / "reference_sample.csv",
        "normal": processed_dir / "normal_sample.csv",
        "drifted": processed_dir / "drifted_sample.csv",
    }
    reference.to_csv(paths["reference"], index=False)
    normal.to_csv(paths["normal"], index=False)
    drifted.to_csv(paths["drifted"], index=False)
    return {name: str(path) for name, path in paths.items()}


def _print_report(name: str, report: pd.DataFrame) -> None:
    print(f"\nBatch: {name}")
    print(f"{'feature':22}{'tipe':12}{'psi':>8}  status")
    print("-" * 52)
    for _, row in report.iterrows():
        print(f"{row['feature']:22}{row['tipe']:12}{row['psi']:>8.4f}  {row['status']}")
    counts = report["status"].value_counts().to_dict()
    print(f"Ringkasan: {counts}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PSI drift report")
    parser.add_argument("--batch", help="CSV batch yang ingin dibandingkan dengan data referensi")
    parser.add_argument(
        "--reference", help="CSV referensi (default: data/processed/reference_sample.csv)"
    )
    parser.add_argument("--make-samples", action="store_true", help="Buat ulang batch contoh lalu keluar")
    args = parser.parse_args(argv)

    config = load_config()
    processed_dir = project_path(config, "processed_dir")

    if args.make_samples or not (processed_dir / "drifted_sample.csv").exists():
        paths = make_samples(config)
        print(f"Sampel dibuat: {paths}")
        if args.make_samples:
            return 0

    reference_path = args.reference or str(processed_dir / "reference_sample.csv")
    reference = pd.read_csv(reference_path)

    batches = (
        {"custom": args.batch}
        if args.batch
        else {
            "normal (test split)": str(processed_dir / "normal_sample.csv"),
            "drifted (sengaja digeser)": str(processed_dir / "drifted_sample.csv"),
        }
    )

    output = {"reference": reference_path, "batches": {}}
    for name, path in batches.items():
        batch = pd.read_csv(path)
        report = drift_report(reference, batch, config)
        _print_report(name, report)
        output["batches"][name] = {
            "path": path,
            "rows": int(len(batch)),
            "features": report.to_dict(orient="records"),
            "positive_rate": positive_rate(batch, config),
        }

    dump_json(output, project_path(config, "drift_report"))
    print(f"\nLaporan JSON: {project_path(config, 'drift_report')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
