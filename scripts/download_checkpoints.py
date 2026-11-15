"""Reproducible Model Checkpoint Verification & Download Utility.

Module: NB6057CEM — Artificial Neural Networks
Portfolio Integration: Checkpoint Acquisition & Integrity Verification

Verifies all 6 trained model checkpoints against strict SHA256 checksums.
Supports downloading from GitHub Releases or verifying local artifacts.

Usage:
    python scripts/download_checkpoints.py [--verify-only] [--url <archive_url>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import urllib.request
import zipfile
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_checkpoints")

CHECKPOINT_MANIFEST: dict[str, dict[str, Any]] = {
    "numpy_mlp_t2_seed42.npz": {
        "rel_path": "checkpoints/numpy_mlp_t2_seed42.npz",
        "size_bytes": 23298,
        "sha256": "568ea4cc58fd2c41ab944a746339b6a44c283e5f4e5012324a274b021e440c73",
        "architecture": "NumPy Scratch MLP (80 -> 32 -> 1, CW1 T2)",
        "required_for_basic_demo": True,
    },
    "random_forest_t3_seed42.joblib": {
        "rel_path": "checkpoints/random_forest_t3_seed42.joblib",
        "size_bytes": 2853161,
        "sha256": "e7eb080f8a2dc4873930e8c4c6dca71f89d1e8ad14a41af905721c4db12bce3d",
        "architecture": "Random Forest Baseline (100 estimators, CW1 T3)",
        "required_for_basic_demo": True,
    },
    "deep_mlp_t3_seed42.pt": {
        "rel_path": "checkpoints/deep_mlp_t3_seed42.pt",
        "size_bytes": 79897,
        "sha256": "b1ebf2a2531c616c0a00a143374f7bd6f2782dd8eea804755de43d62c4643d50",
        "architecture": "Deep PyTorch MLP (80 -> 128 -> 64 -> 1, CW1 T3 Recommended)",
        "required_for_basic_demo": True,
    },
    "cnn_2d_t3_seed42.pt": {
        "rel_path": "checkpoints/cnn_2d_t3_seed42.pt",
        "size_bytes": 104867,
        "sha256": "3d84ec72d59b0884a04b5fe04ed6034c124c99347526bc28ecd26fd2af33602a",
        "architecture": "2D CNN Spatial Baseline (Log-Mel, CW1 T3)",
        "required_for_basic_demo": True,
    },
    "deep_mlp_augmented_seed42.pt": {
        "rel_path": "checkpoints/deep_mlp_augmented_seed42.pt",
        "size_bytes": 80109,
        "sha256": "c5329d2b2f4ce3c323d93f567b14ab152759b3582eb47971a67539e36860d857",
        "architecture": "Noise-Augmented Deep MLP (Phase 2 Portfolio Upgrade)",
        "required_for_basic_demo": True,
    },
    "cnn_gru_seed42.pt": {
        "rel_path": "checkpoints/cnn_gru_seed42.pt",
        "size_bytes": 323857,
        "sha256": "2bfed8e88f58818712cc4a31531358fc1bc63b9a4fea8a534ac0901968fadddb",
        "architecture": "CNN + GRU Hybrid (CRNN, Phase 3 Portfolio Upgrade)",
        "required_for_basic_demo": True,
    },
}

SUPPORTING_DATA_FILES: dict[str, dict[str, Any]] = {
    "mfcc_scaler.json": {
        "rel_path": "data/processed/mfcc_scaler.json",
        "sha256": "dd6b163ffdbb8f68a51cb1ff94a89dec4d092f5f0e9cff47cbd82fac7e1289d8",
    },
    "mel_norm_stats.json": {
        "rel_path": "data/processed/mel_norm_stats.json",
        "sha256": "37ca452a2d3451a0db6d4fb3f46c61c54580bcfae6cb47cf86b33aac13a794f8",
    },
    "preprocessing_config.json": {
        "rel_path": "data/processed/preprocessing_config.json",
        "sha256": "1ac2b1b7a9f3b684995b0e580527fd8963dfedf13d85954a27a3579004ab64d0",
    },
}


def compute_sha256(file_path: Path) -> str:
    """Compute streaming SHA256 checksum."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoints(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Verify presence and checksums of all model checkpoints."""
    present: list[str] = []
    missing: list[str] = []
    mismatch: list[str] = []

    for name, info in CHECKPOINT_MANIFEST.items():
        p = project_root / info["rel_path"]
        if not p.is_file():
            missing.append(name)
            continue
        actual_sha = compute_sha256(p)
        if actual_sha == info["sha256"]:
            present.append(name)
        else:
            mismatch.append(name)
            logger.warning("Checksum mismatch for '%s': expected %s, got %s", name, info["sha256"], actual_sha)

    return {
        "all_valid": len(missing) == 0 and len(mismatch) == 0,
        "present_count": len(present),
        "total_count": len(CHECKPOINT_MANIFEST),
        "present": present,
        "missing": missing,
        "mismatch": mismatch,
    }


def download_checkpoints_archive(
    archive_url: str,
    project_root: Path = PROJECT_ROOT,
) -> bool:
    """Download and extract checkpoints archive from URL."""
    dest_zip = project_root / "checkpoints_bundle.zip"
    logger.info("Downloading checkpoint bundle from '%s'...", archive_url)
    try:
        urllib.request.urlretrieve(archive_url, dest_zip)
        logger.info("Extracting checkpoints into '%s'...", project_root / "checkpoints")
        with zipfile.ZipFile(dest_zip, "r") as zf:
            zf.extractall(project_root)
        dest_zip.unlink(missing_ok=True)
        return True
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify or download ANN-Siren-Detection checkpoints")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing checkpoints without downloading")
    parser.add_argument("--url", type=str, default=None, help="Custom URL to download checkpoints archive from")
    args = parser.parse_args()

    logger.info("Checking model checkpoints in '%s'...", PROJECT_ROOT / "checkpoints")
    status = verify_checkpoints()

    print("\n" + "=" * 80)
    print("ANN-SIREN-DETECTION MODEL CHECKPOINT INTEGRITY REPORT")
    print("=" * 80)
    for name, info in CHECKPOINT_MANIFEST.items():
        p = PROJECT_ROOT / info["rel_path"]
        if p.is_file():
            actual_sha = compute_sha256(p)
            match = actual_sha == info["sha256"]
            flag = "[VERIFIED]" if match else "[CORRUPT]"
            print(f"{flag:11s} {name:35s} ({p.stat().st_size:,} bytes) - {info['architecture']}")
        else:
            print(f"{'[MISSING]':11s} {name:35s} (expected {info['size_bytes']:,} bytes)")
    print("=" * 80)

    if status["all_valid"]:
        print("\nAll 6 model checkpoints are present and verified against official SHA256 checksums.")
        print("Ready for Streamlit execution: streamlit run app.py")
        return 0

    print(f"\nMissing checkpoints: {len(status['missing'])} / {len(CHECKPOINT_MANIFEST)}")
    if args.url:
        success = download_checkpoints_archive(args.url)
        if success:
            post_status = verify_checkpoints()
            if post_status["all_valid"]:
                print("Checkpoints successfully downloaded and verified.")
                return 0
        return 1

    print("\n[NOTE FOR FRESH CLONES]")
    print("Trained checkpoints are gitignored to comply with GitHub repository size best practices.")
    print("To run inference on a fresh clone, you can either:")
    print("1. Download the pre-packaged release zip from the GitHub Releases tab:")
    print("   https://github.com/HimashaSawani/ANN-Siren-Detection/releases/latest")
    print("   and extract it into the project root.")
    print("2. Or retrain the models deterministically using:")
    print("   python scripts/run_competing_models_t3.py")
    print("   python scripts/run_phase2_noise_augmentation.py --seeds 42")
    print("=" * 80)
    return 1


if __name__ == "__main__":
    sys.exit(main())

