"""Loader konfigurasi terpusat.

Semua modul membaca konfigurasi dari sini supaya tidak ada path atau
hyperparameter yang di-hardcode di banyak tempat.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

_cache: dict[str, Any] | None = None


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Muat config.yaml (di-cache) dan kembalikan sebagai dict biasa."""
    global _cache
    if path is None and _cache is not None:
        return _cache

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config tidak ditemukan: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Config harus berupa mapping: {config_path}")

    if path is None:
        _cache = config
    return config


def project_path(config: dict[str, Any], key: str) -> Path:
    """Ubah entri paths.<key> menjadi path absolut di dalam project."""
    try:
        relative = config["paths"][key]
    except KeyError as exc:
        raise KeyError(f"paths.{key} tidak ada di config.yaml") from exc
    return PROJECT_ROOT / relative


def ensure_parent(path: str | Path) -> Path:
    """Buat folder induk kalau belum ada, kembalikan Path-nya."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def set_seed(seed: int | None = None) -> int:
    """Set seed untuk random & numpy agar hasil bisa direproduksi."""
    if seed is None:
        seed = int(load_config()["seed"])
    random.seed(seed)
    np.random.seed(seed)
    return seed


def dump_json(payload: Any, path: str | Path) -> Path:
    """Tulis JSON (indent 2, UTF-8) ke path, buat folder bila perlu."""
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
    return target


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
