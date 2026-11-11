"""Safely import and verify real-data Colab artifacts into D:\\ANN-Siren-Detection.

Verifies Phase 3 & Phase 4 requirements:
- Never overwrites source code (`src/`, `scripts/`, `configs/`, `tests/`, `notebooks/`).
- Extracts only whitelisted artifact paths (`checkpoints/`, `results/metrics/`,
  `results/figures/`, `data/splits/`, `data/processed/`, `data/raw/DATASET_PROVENANCE.json`).
- Verifies that all outputs originate from the real 8,732-clip UrbanSound8K dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ALLOWED_PREFIXES = (
    "checkpoints/",
    "results/metrics/",
    "results/figures/",
    "data/splits/",
    "data/processed/",
    "data/raw/DATASET_PROVENANCE.json",
)

REQUIRED_REAL_DATA_ARTIFACTS = (
    "results/metrics/dataset_audit.json",
    "data/splits/train.csv",
    "data/splits/val.csv",
    "data/splits/test.csv",
    "data/processed/mfcc_scaler.json",
    "data/processed/mel_normalizer.json",
    "results/metrics/t2_real_data_training.json",
    "results/metrics/t3_experiment_status.json",
    "results/metrics/t3_rf_tuning_seed42.json",
    "results/metrics/t3_deep_mlp_val_seed42.json",
    "results/metrics/t3_cnn_2d_val_seed42.json",
    "checkpoints/numpy_mlp_t2_seed42.npz",
    "checkpoints/deep_mlp_t3_seed42.pt",
    "checkpoints/cnn_2d_t3_seed42.pt",
    "results/figures/fig_t2_real_data_loss_curves.png",
    "results/figures/fig_t3_deep_mlp_curves_seed42.png",
    "results/figures/fig_t3_cnn_2d_curves_seed42.png",
)


def verify_real_data_artifacts(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Verify whether all Phase 3 real-data artifacts exist and originate from real 8,732-clip data."""
    present: list[str] = []
    missing: list[str] = []
    for rel in REQUIRED_REAL_DATA_ARTIFACTS:
        p = project_root / rel
        if p.is_file() and p.stat().st_size > 0:
            present.append(rel)
        else:
            missing.append(rel)

    audit_ok = False
    total_recordings = 0
    audit_path = project_root / "results" / "metrics" / "dataset_audit.json"
    if audit_path.is_file():
        audit_data = json.loads(audit_path.read_text(encoding="utf-8"))
        total_recordings = int(audit_data.get("total_audio_recordings", 0))
        audit_ok = total_recordings == 8732

    t3_status_ok = False
    t3_val_macro_f1: dict[str, Any] = {}
    t3_path = project_root / "results" / "metrics" / "t3_experiment_status.json"
    if t3_path.is_file():
        t3_data = json.loads(t3_path.read_text(encoding="utf-8"))
        t3_status_ok = t3_data.get("real_data_training_status") == "COMPLETED_ON_REAL_URBANSOUND8K"
        t3_val_macro_f1 = t3_data.get("validation_macro_f1", {})

    return {
        "all_required_files_present": len(missing) == 0,
        "verified_8732_real_dataset": audit_ok,
        "total_audio_recordings": total_recordings,
        "t3_real_training_completed": t3_status_ok,
        "t3_validation_macro_f1": t3_val_macro_f1,
        "present_count": len(present),
        "missing_files": missing,
    }


def import_artifacts_from_zip(zip_path: Path, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Extract only whitelisted artifact paths from `ann_siren_t1_t2_t3_artifacts.zip`."""
    if not zip_path.is_file():
        raise FileNotFoundError(f"Artifact zip not found: {zip_path}")

    extracted: list[str] = []
    skipped: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            norm = member.replace("\\", "/").lstrip("/")
            if not norm or norm.endswith("/"):
                continue
            if any(norm.startswith(prefix) for prefix in ALLOWED_PREFIXES):
                dest = (project_root / norm).resolve()
                if not str(dest).startswith(str(project_root.resolve())):
                    raise ValueError(f"Unsafe zip member path: {member}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src_f, dest.open("wb") as out_f:
                    shutil.copyfileobj(src_f, out_f)
                extracted.append(norm)
            else:
                skipped.append(norm)

    report = verify_real_data_artifacts(project_root)
    report["extracted_files_count"] = len(extracted)
    report["skipped_non_artifact_files"] = skipped
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import and verify Colab training artifacts.")
    parser.add_argument("--zip", type=Path, default=None, help="Path to ann_siren_t1_t2_t3_artifacts.zip")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing local artifacts.")
    args = parser.parse_args(argv)

    if args.zip is not None:
        report = import_artifacts_from_zip(args.zip, PROJECT_ROOT)
    else:
        report = verify_real_data_artifacts(PROJECT_ROOT)

    print(json.dumps(report, indent=2))
    return 0 if (report["all_required_files_present"] and report["verified_8732_real_dataset"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())

