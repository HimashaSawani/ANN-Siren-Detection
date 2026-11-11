"""Package ANN-Siren-Detection Phase 3 files for Google Colab Pro GPU training.

Module: NB6057CEM — Artificial Neural Networks
Portfolio Upgrade: Phase 3 — Temporal Deep Learning (CNN + GRU Hybrid)

This script bundles the exact minimal assets required to run Phase 3 training
and evaluation on Google Colab Pro with CUDA GPU acceleration, while excluding
unnecessary files (.venv, .git, raw 6GB audio WAVs, reports, and large caches).

Usage:
    python scripts/package_colab_phase3.py [--output-zip colab_phase3_package.zip] [--stage-dir colab_phase3_package]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import zipfile
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("package_colab_phase3")

REQUIRED_CODE_DIRS = ("configs", "src")
REQUIRED_DATA_FILES = (
    "data/processed/train_val_mel_cache.npz",
    "data/processed/test_mel_cache.npz",
    "data/processed/mel_norm_stats.json",
    "data/processed/preprocessing_config.json",
    "data/splits/train.csv",
    "data/splits/val.csv",
    "data/splits/test.csv",
    "data/splits/split_summary.json",
)
OPTIONAL_METRICS_FILES = (
    "results/metrics/dataset_audit.json",
    "results/metrics/t3_deep_mlp_val_seed42.json",
    "results/metrics/t3_cnn_2d_val_seed42.json",
    "results/metrics/phase2_comparison_summary.json",
    "results/metrics/phase2_augmented_eval_results.json",
)
RUNNER_SCRIPTS = (
    "scripts/run_phase3_cnn_gru.py",
    "scripts/import_colab_artifacts.py",
    "scripts/smoke_test.py",
    "scripts/coursework_demo.py",
)
ROOT_FILES = (
    "requirements.txt",
    "pyproject.toml",
)


def get_transfer_manifest(project_root: Path = PROJECT_ROOT) -> list[Path]:
    """Compile the list of files to be included in the transfer package."""
    manifest: list[Path] = []

    # 1. Source and config directories
    for dir_name in REQUIRED_CODE_DIRS:
        dir_path = project_root / dir_name
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Required directory '{dir_path}' not found.")
        for p in sorted(dir_path.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith((".pyc", ".pyo")):
                manifest.append(p.relative_to(project_root))

    # 2. Selected scripts
    for script_rel in RUNNER_SCRIPTS:
        p = project_root / script_rel
        if p.is_file():
            manifest.append(Path(script_rel))
        else:
            logger.warning("Runner script '%s' not found; skipping.", script_rel)

    # 3. Root configuration files
    for root_file in ROOT_FILES:
        p = project_root / root_file
        if p.is_file():
            manifest.append(Path(root_file))
        else:
            logger.warning("Root file '%s' not found; skipping.", root_file)

    # 4. Required preprocessed Mel spectrogram caches & splits
    for data_rel in REQUIRED_DATA_FILES:
        p = project_root / data_rel
        if not p.is_file():
            raise FileNotFoundError(f"Critical Phase 3 data artifact missing: '{p}'.")
        manifest.append(Path(data_rel))

    # 5. Coursework and Phase 2 baseline metrics (for comparison table)
    for metrics_rel in OPTIONAL_METRICS_FILES:
        p = project_root / metrics_rel
        if p.is_file():
            manifest.append(Path(metrics_rel))

    return sorted(set(manifest))


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file with streaming chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def create_package_zip(
    manifest: list[Path],
    output_zip: Path,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Create compressed transfer ZIP archive."""
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Building transfer ZIP archive: '%s' (%d files)...", output_zip, len(manifest))

    total_bytes = 0
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for rel in manifest:
            src = project_root / rel
            size = src.stat().st_size
            total_bytes += size
            logger.info("  Adding: %-45s [%8.2f MB]", rel.as_posix(), size / (1024 * 1024))
            zf.write(src, arcname=rel.as_posix())

    zip_size = output_zip.stat().st_size
    zip_sha = compute_file_sha256(output_zip)

    logger.info("Successfully created '%s'", output_zip.name)
    logger.info("  Total uncompressed size : %.2f MB", total_bytes / (1024 * 1024))
    logger.info("  ZIP package size        : %.2f MB", zip_size / (1024 * 1024))
    logger.info("  SHA256 Checksum         : %s", zip_sha)

    manifest_summary = {
        "package_name": output_zip.name,
        "total_files": len(manifest),
        "uncompressed_bytes": total_bytes,
        "uncompressed_mb": round(total_bytes / (1024 * 1024), 2),
        "zip_size_bytes": zip_size,
        "zip_size_mb": round(zip_size / (1024 * 1024), 2),
        "sha256": zip_sha,
        "files": [m.as_posix() for m in manifest],
    }

    manifest_path = output_zip.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest_summary, indent=2), encoding="utf-8")
    logger.info("Saved manifest summary to '%s'.", manifest_path.name)

    return manifest_summary


def stage_package_directory(
    manifest: list[Path],
    stage_dir: Path,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Copy files to a staged folder matching Google Drive structure."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Staging transfer package in directory: '%s'...", stage_dir)

    total_bytes = 0
    for rel in manifest:
        src = project_root / rel
        dst = stage_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        total_bytes += src.stat().st_size

    summary = {
        "stage_directory": str(stage_dir),
        "total_files": len(manifest),
        "uncompressed_bytes": total_bytes,
        "uncompressed_mb": round(total_bytes / (1024 * 1024), 2),
        "files": [m.as_posix() for m in manifest],
    }

    manifest_path = stage_dir / "package_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Successfully staged %d files (%.2f MB) to '%s'.", len(manifest), total_bytes / (1024 * 1024), stage_dir)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Package ANN-Siren-Detection Phase 3 files for Google Colab Pro GPU")
    parser.add_argument("--output-zip", type=Path, default=PROJECT_ROOT / "colab_phase3_package.zip",
                        help="Path for destination ZIP archive")
    parser.add_argument("--stage-dir", type=Path, default=None,
                        help="Optional directory to stage uncompressed files for direct Drive sync")
    args = parser.parse_args()

    manifest = get_transfer_manifest()
    logger.info("Resolved %d files for Phase 3 Colab transfer.", len(manifest))

    if args.stage_dir is not None:
        stage_package_directory(manifest, args.stage_dir)

    create_package_zip(manifest, args.output_zip)

    print("\n" + "=" * 80)
    print("PHASE 3 GOOGLE COLAB PRO TRANSFER PACKAGE READY")
    print("=" * 80)
    print(f"Archive File       : {args.output_zip}")
    print(f"Archive Size       : {args.output_zip.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"Notebook File      : notebooks/colab_phase3_cnn_gru.ipynb")
    print("\nNext steps:")
    print("1. Upload 'colab_phase3_package.zip' to Google Drive:")
    print("   MyDrive/ANN-Siren-Detection/colab_phase3_package.zip")
    print("2. Open 'notebooks/colab_phase3_cnn_gru.ipynb' in Google Colab Pro.")
    print("3. Set Runtime -> Change runtime type -> T4 GPU / V100 GPU / A100 GPU.")
    print("4. Execute cells sequentially. Checkpoints will be saved directly to Drive.")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())

