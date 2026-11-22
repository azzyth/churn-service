"""Bangun dataset: validasi -> split 70/15/15 -> EDA -> manifest reproducible.

Pemakaian:
    python -m src.data.make_dataset

Output:
    data/splits/{train,val,test}.csv
    data/splits/split_manifest.json      (hash + seed + proporsi + jumlah baris)
    reports/figures/*.png                (4 plot EDA)
    reports/data_findings.md
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")  # backend non-interaktif supaya bisa jalan tanpa GUI

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.config import dump_json, load_config, project_path, set_seed
from src.data.load import clean, load_raw, sha256_file
from src.data.validate import format_results, total_violations, validate

sns.set_theme(style="whitegrid", palette="deep")


def make_splits(df: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    """Split stratifikasi 70/15/15. Test set tidak dipakai sampai fase evaluasi."""
    target = config["data"]["target"]
    seed = config["seed"]
    fractions = config["data"]["split"]

    train_df, remainder = train_test_split(
        df,
        test_size=1 - float(fractions["train"]),
        stratify=df[target].astype(int),
        random_state=seed,
    )
    val_share_of_remainder = float(fractions["val"]) / (
        float(fractions["val"]) + float(fractions["test"])
    )
    val_df, test_df = train_test_split(
        remainder,
        test_size=1 - val_share_of_remainder,
        stratify=remainder[target].astype(int),
        random_state=seed,
    )
    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def save_splits(splits: dict[str, pd.DataFrame], config: dict) -> dict:
    target = config["data"]["target"]
    out_dir = project_path(config, "splits_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": config["seed"],
        "fractions": config["data"]["split"],
        "raw_sha256": sha256_file(project_path(config, "raw_file")),
        "splits": {},
    }
    for name, frame in splits.items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        manifest["splits"][name] = {
            "rows": int(len(frame)),
            "positive_rate": round(float(frame[target].astype(int).mean()), 6),
            "sha256": sha256_file(path),
        }
    return manifest


def plot_eda(df: pd.DataFrame, config: dict) -> list[str]:
    """Buat 4 figure EDA ke reports/figures. Kembalikan daftar path."""
    target = config["data"]["target"]
    figures_dir = project_path(config, "figures_dir")
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # 1. Distribusi target
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[target].value_counts().sort_index()
    labels = ["Tidak churn (0)", "Churn (1)"]
    bars = ax.bar(labels, counts.values, color=["#4C72B0", "#C44E52"])
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value}\n({value / len(df):.1%})",
            ha="center",
            va="bottom",
        )
    ax.set_title("Distribusi target Churn")
    ax.set_ylabel("Jumlah pelanggan")
    path = figures_dir / "01_target_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    # 2. Distribusi numerik per kelas churn
    numeric = ["tenure", "MonthlyCharges", "TotalCharges"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, column in zip(axes, numeric):
        sns.histplot(
            data=df, x=column, hue=target, bins=30, element="step", common_norm=False, ax=ax
        )
        ax.set_title(column)
    fig.suptitle("Distribusi fitur numerik (0 = tidak churn, 1 = churn)")
    path = figures_dir / "02_numeric_distributions.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    # 3. Churn rate per kategori penting
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, column in zip(axes, ["Contract", "InternetService"]):
        rate = df.groupby(column, observed=True)[target].mean().sort_values(ascending=False)
        bars = ax.bar(rate.index.astype(str), rate.values, color="#55A868")
        for bar, value in zip(bars, rate.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, value, f"{value:.1%}", ha="center", va="bottom"
            )
        ax.set_title(f"Churn rate berdasarkan {column}")
        ax.set_ylabel("Proporsi churn")
        ax.tick_params(axis="x", rotation=15)
    path = figures_dir / "03_churn_rate_by_category.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    # 4. Korelasi numerik
    fig, ax = plt.subplots(figsize=(6, 5))
    correlation = df[numeric + ["SeniorCitizen", target]].astype(float).corr()
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Korelasi fitur numerik")
    path = figures_dir / "04_numeric_correlation.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    return written


def write_findings(df: pd.DataFrame, stats: dict, config: dict, manifest: dict) -> str:
    """Tulis 3 temuan data + keputusan handling (dipakai README & laporan)."""
    target = config["data"]["target"]
    reports_dir = project_path(config, "reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "data_findings.md"

    churn_rate = float(df[target].astype(int).mean())
    month_to_month = df[df["Contract"] == "Month-to-month"]
    rate_mtm = float(month_to_month[target].astype(int).mean())
    rate_two_year = float(df[df["Contract"] == "Two year"][target].astype(int).mean())

    text = f"""# Temuan Data — Telco Customer Churn

Digenerate oleh `python -m src.data.make_dataset`.
Raw file SHA-256: `{manifest['raw_sha256']}` · {manifest['splits']['train']['rows']} baris train,
{manifest['splits']['val']['rows']} val, {manifest['splits']['test']['rows']} test.

## Temuan 1 — `TotalCharges` bukan angka
Kolom ini bertipe teks di CSV. Ada **{stats['blank_total_charges']} baris berisi string kosong**,
dan semuanya milik pelanggan dengan `tenure = 0` (pelanggan baru yang belum pernah ditagih).

**Keputusan:** `pd.to_numeric(errors="coerce")` lalu isi **0.0 hanya untuk `tenure = 0`**.
Tidak ada baris yang dihapus, dan tidak ada imputasi median yang mengarang tagihan.
Sisa nilai kosong setelah pembersihan: {stats['remaining_total_charges_na']}.

## Temuan 2 — Kelas tidak seimbang
Proporsi churn **{churn_rate:.1%}** ({stats['rows_in']} baris total). Baseline "semua tidak churn"
sudah menghasilkan akurasi {1 - churn_rate:.1%} tanpa belajar apa pun.

**Keputusan:** akurasi tidak dipakai sebagai metrik utama. Metrik yang dipakai ROC-AUC dan PR-AUC,
ditambah pemilihan threshold berbasis biaya (FN = {config['business']['cost_false_negative']}×,
FP = {config['business']['cost_false_positive']}×). Data tidak di-resample/oversample.

## Temuan 3 — `customerID` adalah identitas, bukan fitur
{stats['id_unique']} nilai unik untuk {stats['rows_in']} baris (unik, tanpa duplikat).

**Keputusan:** dibuang dari matriks fitur, tetapi tetap disimpan di file split untuk keperluan
batch scoring dan pelacakan error per pelanggan.

## Sinyal tambahan (bukan temuan teknis)
Rata-rata churn pada kontrak `Month-to-month` **{rate_mtm:.1%}** vs `Two year` **{rate_two_year:.1%}**.
`Contract` diperkirakan menjadi fitur paling kuat, dan ini masuk akal secara bisnis: pelanggan
tanpa ikatan kontrak paling mudah pindah.
"""
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
    return str(path)


def main() -> int:
    config = load_config()
    set_seed(config["seed"])

    print("Memuat data mentah...")
    raw = load_raw(config)
    df, stats = clean(raw, config)
    print(f"Bersih: {stats}")

    results = validate(df, config["data"]["target"], config["data"]["id_column"])
    print("\n" + format_results(results) + "\n")
    violations = total_violations(results)
    if violations:
        print(f"BERHENTI: {violations} pelanggaran data. Perbaiki sumber data dulu.", file=sys.stderr)
        return 1

    splits = make_splits(df, config)
    manifest = save_splits(splits, config)
    dump_json(manifest, project_path(config, "splits_dir") / "split_manifest.json")

    target = config["data"]["target"]
    print("Split (stratified):")
    for name, frame in splits.items():
        rate = float(frame[target].astype(int).mean())
        print(
            f"  {name:5} rows={len(frame):5d}  churn={rate:.2%}  "
            f"sha256={manifest['splits'][name]['sha256'][:12]}..."
        )

    figures = plot_eda(df, config)
    print(f"\nFigur EDA: {len(figures)} file di {project_path(config, 'figures_dir')}")
    findings = write_findings(df, stats, config, manifest)
    print(f"Temuan data: {findings}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
