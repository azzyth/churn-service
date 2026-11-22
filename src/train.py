"""Latih dan bandingkan model: dummy baseline -> logistic regression -> LightGBM.

Pemakaian:
    python -m src.train

Prinsip:
- Cross-validation hanya di dalam train split (val dipakai untuk threshold, test tidak disentuh).
- Kalau model kompleks tidak menang jelas, model paling sederhana yang dipakai.
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src import __version__
from src.config import dump_json, load_config, project_path, read_json, set_seed
from src.features.build import build_pipeline, feature_hash, make_xy

SELECTION_MIN_GAIN = 0.01  # PR-AUC minimum agar model kompleks dianggap lebih baik


def recall_at_precision(y_true: np.ndarray, y_score: np.ndarray, min_precision: float) -> float:
    """Recall tertinggi yang masih mempertahankan presisi >= min_precision."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    eligible = recall[precision >= min_precision]
    if eligible.size == 0:
        return float("nan")
    return float(np.max(eligible))


def build_candidates(config: dict) -> dict[str, object]:
    lr_params = dict(config["models"]["logistic_regression"])
    lgb_params = dict(config["models"]["lightgbm"])
    return {
        "dummy": build_pipeline(DummyClassifier(strategy="prior")),
        "logistic_regression": build_pipeline(
            LogisticRegression(**lr_params), scale_numeric=True
        ),
        "lightgbm": build_pipeline(
            lightgbm.LGBMClassifier(random_state=config["seed"], **lgb_params),
            scale_numeric=False,
        ),
    }


def cross_validate(pipeline, X, y, config: dict) -> dict:
    """5-fold stratified CV; semua metrik dihitung di fold validasi."""
    cv_config = config["cv"]
    splitter = StratifiedKFold(
        n_splits=int(cv_config["n_splits"]),
        shuffle=bool(cv_config["shuffle"]),
        random_state=config["seed"],
    )
    per_fold: dict[str, list[float]] = {
        "roc_auc": [],
        "pr_auc": [],
        "f1": [],
        "recall_at_precision_80": [],
    }

    for train_idx, valid_idx in splitter.split(X, y):
        model = clone(pipeline)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        scores = model.predict_proba(X.iloc[valid_idx])[:, 1]
        y_valid = y.iloc[valid_idx].to_numpy()

        per_fold["roc_auc"].append(float(roc_auc_score(y_valid, scores)))
        per_fold["pr_auc"].append(float(average_precision_score(y_valid, scores)))
        per_fold["f1"].append(float(f1_score(y_valid, (scores >= 0.5).astype(int), zero_division=0)))
        per_fold["recall_at_precision_80"].append(
            recall_at_precision(y_valid, scores, 0.80)
        )

    summary = {}
    for metric, values in per_fold.items():
        clean = [v for v in values if not np.isnan(v)]
        summary[metric] = {
            "mean": float(np.mean(clean)) if clean else float("nan"),
            "std": float(np.std(clean)) if clean else float("nan"),
            "folds": [float(v) for v in values],
        }
    return summary


def choose_model(cv_results: dict[str, dict]) -> tuple[str, str]:
    """Pilih model: PR-AUC tertinggi, tapi butuh selisih jelas vs logistic regression."""
    pr_auc = {name: res["pr_auc"]["mean"] for name, res in cv_results.items() if name != "dummy"}
    ordered = sorted(pr_auc.items(), key=lambda item: item[1], reverse=True)
    best_name, best_score = ordered[0]
    lr_score = pr_auc.get("logistic_regression", float("-inf"))

    if best_name == "lightgbm" and (best_score - lr_score) < SELECTION_MIN_GAIN:
        gap = best_score - lr_score
        reason = (
            f"LightGBM hanya unggul {gap:+.4f} PR-AUC (< ambang {SELECTION_MIN_GAIN}) "
            "dibanding logistic regression, jadi model paling sederhana yang dipakai."
        )
        return "logistic_regression", reason

    reason = (
        f"{best_name} unggul {best_score - lr_score:+.4f} PR-AUC dibanding logistic regression "
        f"(>= ambang {SELECTION_MIN_GAIN})."
        if best_name != "logistic_regression"
        else "Logistic regression sudah model terbaik menurut PR-AUC."
    )
    return best_name, reason


def print_cv_table(cv_results: dict[str, dict]) -> None:
    header = f"{'model':22}{'ROC-AUC':>18}{'PR-AUC':>18}{'F1@0.5':>18}{'Recall@P80':>18}"
    print(header)
    print("-" * len(header))
    for name, res in cv_results.items():
        cells = []
        for metric in ("roc_auc", "pr_auc", "f1", "recall_at_precision_80"):
            mean = res[metric]["mean"]
            std = res[metric]["std"]
            cells.append("nan".rjust(18) if np.isnan(mean) else f"{mean:.4f} +/- {std:.4f}".rjust(18))
        print(f"{name:22}" + "".join(cells))


def environment_info() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "joblib": joblib.__version__,
    }


def load_artifact(config: dict | None = None):
    """Muat (model, meta) dari disk. Dipakai API, evaluasi, dan drift."""
    config = config or load_config()
    model_path = project_path(config, "model_file")
    meta_path = project_path(config, "meta_file")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Artefak model belum ada: {model_path}. Jalankan `python -m src.train` dulu."
        )
    model = joblib.load(model_path)
    meta = read_json(meta_path) if meta_path.exists() else {}
    return model, meta


def main() -> int:
    config = load_config()
    seed = set_seed(config["seed"])
    splits_dir = project_path(config, "splits_dir")
    train_path = splits_dir / "train.csv"
    if not train_path.exists():
        print(
            f"Split belum ada: {train_path}. Jalankan `python -m src.data.make_dataset` dulu.",
            file=sys.stderr,
        )
        return 1

    train_df = pd.read_csv(train_path)
    X, y = make_xy(train_df, config)
    print(f"Train: {len(X)} baris, {X.shape[1]} fitur, churn={y.mean():.2%}, seed={seed}")

    candidates = build_candidates(config)
    cv_results: dict[str, dict] = {}
    for name, pipeline in candidates.items():
        print(f"CV {name} ...")
        cv_results[name] = cross_validate(pipeline, X, y, config)

    print("\nHasil cross-validation (5-fold, train split):")
    print_cv_table(cv_results)

    selected_name, reason = choose_model(cv_results)
    print(f"\nModel terpilih : {selected_name}")
    print(f"Alasan         : {reason}")

    final_model = clone(candidates[selected_name])
    final_model.fit(X, y)

    model_path = project_path(config, "model_file")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)

    meta = {
        "project": config["project"]["name"],
        "version": config["project"]["version"],
        "code_version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_name": selected_name,
        "selection_rule": (
            f"PR-AUC tertinggi, tetapi LightGBM hanya dipakai bila unggul "
            f">= {SELECTION_MIN_GAIN} dari logistic regression."
        ),
        "selection_reason": reason,
        "seed": seed,
        "training_rows": int(len(X)),
        "positive_rate": round(float(y.mean()), 6),
        "n_features": int(len(X.columns)),
        "feature_hash": feature_hash(),
        "cv": cv_results,
        "environment": environment_info(),
        "artifact": str(model_path.relative_to(project_path(config, "models_dir").parent)),
        "threshold": None,
    }
    meta_path = dump_json(meta, project_path(config, "meta_file"))

    print(f"\nArtefak  : {model_path}")
    print(f"Metadata : {meta_path}")
    print("Test set belum disentuh. Threshold dipilih di fase evaluasi (val split).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
