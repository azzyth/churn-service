"""Evaluasi di val + test, pilih threshold berbasis biaya, dan tulis laporan.

Pemakaian:
    python -m src.evaluate

Urutan yang dijaga:
1. Threshold dipilih HANYA dari val split (cost curve FN=5, FP=1).
2. Test split dievaluasi sekali dengan threshold hasil langkah 1.
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import dump_json, load_config, project_path, set_seed
from src.features.build import make_xy
from src.train import load_artifact

sns.set_theme(style="whitegrid")

THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def cost_curve(y_true: np.ndarray, y_score: np.ndarray, config: dict) -> pd.DataFrame:
    """Total biaya pada tiap threshold: FN x cost_FN + FP x cost_FP."""
    cost_fn = float(config["business"]["cost_false_negative"])
    cost_fp = float(config["business"]["cost_false_positive"])
    rows = []
    for threshold in THRESHOLD_GRID:
        prediction = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "cost": float(fn * cost_fn + fp * cost_fp),
                "precision": float(precision_score(y_true, prediction, zero_division=0)),
                "recall": float(recall_score(y_true, prediction, zero_division=0)),
                "f1": float(f1_score(y_true, prediction, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def pick_threshold(curve: pd.DataFrame, config: dict) -> pd.Series:
    """Threshold dengan biaya terendah; kalau diminta, batasi ke presisi minimum."""
    min_precision = float(config["business"].get("min_precision", 0.0) or 0.0)
    candidates = curve
    if min_precision > 0:
        filtered = curve[curve["precision"] >= min_precision]
        if filtered.empty:
            print(
                f"PERINGATAN: tidak ada threshold dengan precision >= {min_precision}; "
                "batas presisi diabaikan dan dipilih biaya terendah."
            )
        else:
            candidates = filtered
    return candidates.loc[candidates["cost"].idxmin()]


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    prediction = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
        "brier": round(float(brier_score_loss(y_true, y_score)), 4),
        "precision": round(float(precision_score(y_true, prediction, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, prediction, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, prediction, zero_division=0)), 4),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
    }


def plot_cost_curve(curve: pd.DataFrame, chosen: pd.Series, figures_dir, config: dict):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["threshold"], curve["cost"], color="#C44E52", label="total biaya relatif")
    ax.axvline(chosen["threshold"], color="#4C72B0", linestyle="--", label=f"dipilih = {chosen['threshold']:.2f}")
    ax.axvline(0.5, color="gray", linestyle=":", label="default = 0.50")
    ax.set_xlabel("Threshold")
    ax.set_ylabel(
        f"Biaya (FN×{config['business']['cost_false_negative']} + FP×{config['business']['cost_false_positive']})"
    )
    ax.set_title("Cost curve pada val split")
    ax.legend()
    path = figures_dir / "05_cost_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_confusion(metrics: dict, figures_dir):
    matrix = np.array(
        [[metrics["true_negative"], metrics["false_positive"]], [metrics["false_negative"], metrics["true_positive"]]]
    )
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Prediksi: tidak churn", "Prediksi: churn"],
        yticklabels=["Aktual: tidak churn", "Aktual: churn"],
        ax=ax,
    )
    ax.set_title(f"Confusion matrix (test, threshold={metrics['threshold']})")
    path = figures_dir / "06_confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_calibration(y_test: np.ndarray, scores: np.ndarray, figures_dir):
    fraction_positive, mean_predicted = calibration_curve(y_test, scores, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], linestyle=":", color="gray", label="kalibrasi sempurna")
    ax.plot(mean_predicted, fraction_positive, marker="o", color="#4C72B0", label="model")
    ax.set_xlabel("Probabilitas prediksi rata-rata")
    ax.set_ylabel("Proporsi churn aktual")
    ax.set_title("Reliability curve (test)")
    ax.legend()
    path = figures_dir / "07_calibration.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def compute_importance(model, X_test, y_test, seed: int) -> pd.DataFrame:
    """Permutation importance pada ROC-AUC, tanpa dependency tambahan."""
    result = permutation_importance(
        model, X_test, y_test, n_repeats=5, random_state=seed, scoring="roc_auc", n_jobs=-1
    )
    return (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_importance(importance: pd.DataFrame, figures_dir):
    top = importance.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color="#55A868")
    ax.set_xlabel("Penurunan ROC-AUC saat fitur diacak")
    ax.set_title("Permutation importance (test, top 10)")
    path = figures_dir / "08_permutation_importance.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    body = ["| " + " | ".join(str(row[column]) for column in columns) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def main() -> int:
    config = load_config()
    seed = set_seed(config["seed"])
    figures_dir = project_path(config, "figures_dir")
    figures_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = project_path(config, "splits_dir")

    for name in ("val.csv", "test.csv"):
        if not (splits_dir / name).exists():
            print(f"Split {name} tidak ada. Jalankan `python -m src.data.make_dataset`.", file=sys.stderr)
            return 1

    model, meta = load_artifact(config)
    print(f"Model: {meta.get('model_name')} (dilatih {meta.get('trained_at')})")

    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")
    X_val, y_val = make_xy(val_df, config)
    X_test, y_test = make_xy(test_df, config)

    scores_val = model.predict_proba(X_val)[:, 1]
    curve = cost_curve(y_val.to_numpy(), scores_val, config)
    chosen = pick_threshold(curve, config)
    threshold = float(chosen["threshold"])

    cost_rows = []
    for th in (0.5, threshold):
        row = curve.iloc[(curve["threshold"] - th).abs().idxmin()]
        cost_rows.append(
            {
                "threshold": f"{row['threshold']:.2f}",
                "precision": f"{row['precision']:.3f}",
                "recall": f"{row['recall']:.3f}",
                "FN": int(row["fn"]),
                "FP": int(row["fp"]),
                "biaya_relatif": f"{row['cost']:.0f}",
            }
        )
    print("\nPerbandingan threshold di val split:")
    print(markdown_table(cost_rows, list(cost_rows[0].keys())))

    # Test set: dievaluasi sekali, memakai threshold dari val.
    scores_test = model.predict_proba(X_test)[:, 1]
    y_test_array = y_test.to_numpy()
    metrics = classification_metrics(y_test_array, scores_test, threshold)
    metrics_default = classification_metrics(y_test_array, scores_test, 0.5)

    importance = compute_importance(model, X_test, y_test, seed)
    plot_cost_curve(curve, chosen, figures_dir, config)
    plot_confusion(metrics, figures_dir)
    plot_calibration(y_test_array, scores_test, figures_dir)
    plot_importance(importance, figures_dir)

    cv = meta.get("cv", {})
    cv_rows = []
    for name, result in cv.items():
        cv_rows.append(
            {
                "model": name,
                "ROC-AUC (CV)": f"{result['roc_auc']['mean']:.4f} ± {result['roc_auc']['std']:.4f}",
                "PR-AUC (CV)": f"{result['pr_auc']['mean']:.4f} ± {result['pr_auc']['std']:.4f}",
                "F1@0.5 (CV)": f"{result['f1']['mean']:.4f} ± {result['f1']['std']:.4f}",
            }
        )

    # Contoh error: 2 false negative + 1 false positive
    error_frame = test_df.copy()
    error_frame["probabilitas"] = scores_test
    error_frame["prediksi"] = (scores_test >= threshold).astype(int)
    error_frame["aktual"] = y_test_array
    false_negatives = error_frame[(error_frame["aktual"] == 1) & (error_frame["prediksi"] == 0)]
    false_positives = error_frame[(error_frame["aktual"] == 0) & (error_frame["prediksi"] == 1)]
    picked_errors = pd.concat([false_negatives.head(2), false_positives.head(1)])
    error_columns = [
        "customerID",
        "aktual",
        "prediksi",
        "probabilitas",
        "Contract",
        "tenure",
        "MonthlyCharges",
    ]
    error_rows = []
    for _, row in picked_errors.iterrows():
        error_rows.append(
            {
                "customerID": row["customerID"],
                "aktual": int(row["aktual"]),
                "prediksi": int(row["prediksi"]),
                "probabilitas": f"{row['probabilitas']:.3f}",
                "Contract": row["Contract"],
                "tenure": int(row["tenure"]),
                "MonthlyCharges": f"{row['MonthlyCharges']:.2f}",
            }
        )

    # Metrik yang tidak bergantung threshold hanya ditulis sekali; menampilkannya
    # di dua kolom threshold membuat pembaca mengira angkanya berubah.
    independent_rows = [
        {"metrik": "ROC-AUC", "nilai": metrics["roc_auc"]},
        {"metrik": "PR-AUC", "nilai": metrics["pr_auc"]},
        {"metrik": "Brier", "nilai": metrics["brier"]},
    ]

    def _counts(result: dict) -> str:
        return (
            f"{result['true_positive']} / {result['false_positive']} / "
            f"{result['false_negative']} / {result['true_negative']}"
        )

    threshold_rows = [
        {
            "metrik": "Precision",
            "test @0.50": metrics_default["precision"],
            f"test @{threshold:.2f}": metrics["precision"],
        },
        {
            "metrik": "Recall",
            "test @0.50": metrics_default["recall"],
            f"test @{threshold:.2f}": metrics["recall"],
        },
        {
            "metrik": "F1",
            "test @0.50": metrics_default["f1"],
            f"test @{threshold:.2f}": metrics["f1"],
        },
        {
            "metrik": "TP / FP / FN / TN",
            "test @0.50": _counts(metrics_default),
            f"test @{threshold:.2f}": _counts(metrics),
        },
    ]

    error_table = (
        markdown_table(error_rows, error_columns)
        if error_rows
        else "_Tidak ada error pada threshold ini._"
    )

    report = f"""# Evaluasi Model — {meta.get('model_name', 'unknown')}

Dibuat oleh `python -m src.evaluate`.
Model artifact: `{meta.get('artifact')}` · dilatih {meta.get('trained_at')} · seed {meta.get('seed')}.

## 1. Perbandingan kandidat (cross-validation 5-fold di train)

{markdown_table(cv_rows, list(cv_rows[0].keys()))}

Model terpilih: **{meta.get('model_name')}**.
Alasan: {meta.get('selection_reason')}

## 2. Pemilihan threshold (val split, bukan test)

Asumsi biaya: satu false negative (pelanggan churn yang tidak ditangkap) =
**{config['business']['cost_false_negative']}×** biaya satu false positive (diskon retensi yang salah sasaran).

{markdown_table(cost_rows, list(cost_rows[0].keys()))}

Threshold terpilih: **{threshold:.2f}**. Cost curve lengkap ada di `reports/figures/05_cost_curve.png`.
Menurunkan threshold menaikkan recall (lebih banyak churn tertangkap) dengan konsekuensi lebih banyak
pelanggan loyal yang diberi diskon. Keputusan ini murni konsekuensi dari asumsi biaya di `config.yaml`
— kalau rasio biayanya berubah, threshold harus dihitung ulang.

## 3. Performa di test set

Metrik yang tidak bergantung pada threshold (satu nilai, bukan per threshold):

{markdown_table(independent_rows, list(independent_rows[0].keys()))}

Metrik yang bergantung pada threshold:

{markdown_table(threshold_rows, list(threshold_rows[0].keys()))}

Confusion matrix pada threshold {threshold:.2f}:
TP={metrics['true_positive']} · FP={metrics['false_positive']} ·
FN={metrics['false_negative']} · TN={metrics['true_negative']}.
Gambar: `reports/figures/06_confusion_matrix.png`.

Kalibrasi (Brier score {metrics['brier']:.4f}): `reports/figures/07_calibration.png`.
Semakin dekat titik ke garis diagonal, semakin jujur probabilitasnya sebagai probabilitas.

## 4. Fitur paling berpengaruh (permutation importance, test)

{markdown_table([{'peringkat': i + 1, 'fitur': row['feature'], 'delta ROC-AUC': f"{row['importance']:.4f} ± {row['std']:.4f}"} for i, row in importance.head(10).iterrows()], ['peringkat', 'fitur', 'delta ROC-AUC'])}

Gambar: `reports/figures/08_permutation_importance.png`.
Permutation importance mengukur *dampak ke prediksi*, bukan kausalitas.

## 5. Contoh error

{error_table}

Dua baris pertama (bila ada) adalah false negative — churn tidak terdeteksi, paling mahal menurut
asumsi biaya. Baris terakhir (bila ada) adalah false positive — pelanggan loyal yang menerima
diskon tanpa perlu.

## 6. Keterbatasan

- **Dataset kecil dan sintetis**: 7.043 baris dari satu perusahaan telekomunikasi; hasil tidak otomatis
  berlaku untuk perusahaan lain dengan struktur harga dan kontrak berbeda.
- **Tanpa dimensi waktu**: split acak, bukan temporal. Model tidak menguji apakah performa bertahan
  saat perilaku pelanggan bergeser dari waktu ke waktu.
- **Asumsi biaya adalah asumsi**: rasio FN:FP {config['business']['cost_false_negative']}:{config['business']['cost_false_positive']}
  tidak diukur dari data biaya nyata. Threshold yang dipilih adalah konsekuensi langsung dari asumsi itu.
- **Test set dipakai sekali**: angka di atas belum punya confidence interval; dengan {len(test_df)} baris,
  selisih kecil antar model tidak bisa dianggap signifikan.
- **Tidak ada data pasca-churn**: model memprediksi niat berhenti, bukan efektivitas program retensi.
- **Artefak dilatih pada train split saja**; sebelum produksi, latih ulang pada train+val dengan
  threshold yang sudah dipilih di sini.
"""
    report_path = project_path(config, "reports_dir") / "evaluation.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    meta["threshold"] = threshold
    meta["threshold_selection"] = {
        "source_split": "val",
        "grid": [float(THRESHOLD_GRID.min()), float(THRESHOLD_GRID.max())],
        "cost_false_negative": config["business"]["cost_false_negative"],
        "cost_false_positive": config["business"]["cost_false_positive"],
        "expected_cost_relative": round(float(chosen["cost"]), 2),
        "precision_at_threshold": round(float(chosen["precision"]), 4),
        "recall_at_threshold": round(float(chosen["recall"]), 4),
    }
    meta["test_metrics"] = metrics
    dump_json(meta, project_path(config, "meta_file"))

    print(f"\nTest @{threshold:.2f}: ROC-AUC={metrics['roc_auc']} PR-AUC={metrics['pr_auc']} "
          f"P={metrics['precision']} R={metrics['recall']} F1={metrics['f1']}")
    print(f"Laporan: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
