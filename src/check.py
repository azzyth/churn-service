"""Cek kesehatan kerangka project: import semua modul + tampilkan config.

Dipakai oleh `run.ps1 check` (Fase 0).
"""

from __future__ import annotations

import importlib
import sys

MODULES = [
    "src.config",
    "src.data.download",
    "src.data.load",
    "src.data.validate",
    "src.data.make_dataset",
    "src.features.build",
    "src.train",
    "src.evaluate",
    "src.monitor.drift",
    "app.schemas",
    "app.main",
]


def main() -> int:
    from src.config import load_config, project_path

    config = load_config()
    print(f"project     : {config['project']['name']} v{config['project']['version']}")
    print(f"seed        : {config['seed']}")
    print(f"target      : {config['data']['target']}")
    print(
        "split       : "
        f"train={config['data']['split']['train']} "
        f"val={config['data']['split']['val']} "
        f"test={config['data']['split']['test']}"
    )
    print(
        "biaya       : "
        f"FN={config['business']['cost_false_negative']} "
        f"FP={config['business']['cost_false_positive']}"
    )
    print(f"raw file    : {project_path(config, 'raw_file')}")
    print(f"model file  : {project_path(config, 'model_file')}")

    failures = []
    for module_name in MODULES:
        try:
            importlib.import_module(module_name)
            print(f"[ok]   {module_name}")
        except Exception as exc:  # noqa: BLE001 - ingin laporkan semua kegagalan
            failures.append((module_name, exc))
            print(f"[FAIL] {module_name}: {exc.__class__.__name__}: {exc}")

    if failures:
        print(f"\n{len(failures)} modul gagal di-import")
        return 1

    print(f"\n{len(MODULES)} modul ter-import, config ter-load. Siap.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
