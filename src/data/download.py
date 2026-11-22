"""Unduh dataset Telco Customer Churn lewat Kaggle CLI.

Kalau unduhan gagal, kegagalan dilaporkan eksplisit (bukan diam-diam
memakai data sintetis).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from src.config import load_config, project_path

DATASET_SLUG = "blastchar/telco-customer-churn"


def dataset_exists(config: dict) -> bool:
    return project_path(config, "raw_file").exists()


def download(config: dict | None = None) -> Path:
    """Unduh + unzip dataset ke data/raw. Kembalikan path CSV-nya."""
    config = config or load_config()
    target = project_path(config, "raw_file")
    if target.exists():
        print(f"Dataset sudah ada: {target}")
        return target

    raw_dir = project_path(config, "raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)

    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise RuntimeError(
            "Kaggle CLI tidak ditemukan di PATH. Pasang dengan "
            "`python -m pip install -r requirements-dev.txt` "
            "(atau `python -m pip install kaggle`), lalu taruh token di "
            "~/.kaggle/kaggle.json, "
            f"atau unduh manual dataset '{DATASET_SLUG}' ke {raw_dir}."
        )

    print(f"Mengunduh {DATASET_SLUG} -> {raw_dir}")
    result = subprocess.run(
        [kaggle, "datasets", "download", "-d", DATASET_SLUG, "-p", str(raw_dir), "--unzip"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Unduhan Kaggle gagal.\n"
            f"stdout: {result.stdout.strip()}\n"
            f"stderr: {result.stderr.strip()}\n"
            f"Alternatif: unduh manual dataset '{DATASET_SLUG}' lalu simpan CSV-nya di {raw_dir}."
        )

    if not target.exists():
        candidates = sorted(raw_dir.glob("*.csv"))
        raise RuntimeError(
            f"Unduhan selesai tetapi {target.name} tidak ditemukan. "
            f"CSV yang ada: {[c.name for c in candidates]}"
        )

    print(f"Selesai: {target} ({target.stat().st_size / 1024:.0f} KB)")
    return target


def main() -> int:
    try:
        download()
    except Exception as exc:  # noqa: BLE001 - laporkan ke CLI
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
