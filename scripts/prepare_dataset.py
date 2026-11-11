"""Leakage-Safe Split Manifest Generation and Training-Only Preprocessing Preparation — T1(c).

Module: NB6057CEM — Artificial Neural Networks
Step: 2 — Dataset Acquisition, Data Audit and Leakage-Safe Preparation (CW1 T1)

Workflow:
1. Validate UrbanSound8K dataset structure and audio file availability.
2. Enrich metadata with validated binary labels (1=siren, 0=non_siren).
3. Partition deterministically by official predefined folds:
   - Train: Folds 1–8
   - Validation: Fold 9
   - Held-out Test: Fold 10 (reserved strictly for final evaluation)
4. Verify zero source-recording (fsID) overlap and zero duplicate clips across splits.
5. Save `data/splits/train.csv`, `data/splits/val.csv`, `data/splits/test.csv`,
   and `data/splits/split_summary.json`.
6. Fit `MFCCStandardScaler` and `MelSpectrogramNormalizer` on the training split ONLY
   and save normalization artifacts under `data/processed/`.

Usage:
    python scripts/prepare_dataset.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import (  # noqa: E402
    DatasetNotFoundError,
    load_config,
    load_data_config,
    resolve_project_paths,
    setup_logger,
)
from src.audio_features import (  # noqa: E402
    extract_log_mel_spectrogram,
    extract_mfcc_features,
    load_and_standardize_audio,
)
from src.preprocessing import (  # noqa: E402
    create_leakage_safe_splits,
    fit_training_preprocessors,
    load_and_enrich_metadata,
    save_split_manifests,
    validate_urbansound8k_dataset,
)


def run_dataset_preparation(
    dataset_root: str | Path,
    splits_dir: str | Path,
    processed_dir: str | Path,
    data_config: dict[str, Any] | None = None,
    fit_preprocessors: bool = True,
) -> dict[str, Any]:
    """Execute leakage-safe split generation and training-only scaler/normalizer fitting.

    Raises:
        DatasetNotFoundError: If the dataset is not present at ``dataset_root``.
        DatasetValidationError: If metadata, audio files, or leakage checks fail.
    """
    d_cfg = data_config if data_config is not None else load_data_config()
    root_path = Path(dataset_root).resolve()
    meta_path = root_path / "metadata" / "UrbanSound8K.csv"
    audio_dir = root_path / "audio"

    if not root_path.is_dir() or not meta_path.is_file() or not audio_dir.is_dir():
        raise DatasetNotFoundError(
            f"Dataset preparation is blocked: UrbanSound8K dataset not found at '{root_path}'."
        )

    # Full structural validation before splitting
    val_report = validate_urbansound8k_dataset(
        dataset_root=root_path, strict=True, check_audio_files=True
    )

    enriched_df = load_and_enrich_metadata(meta_path, audio_dir=audio_dir)
    split_cfg = d_cfg.get("split_protocol", {})
    train_folds = list(split_cfg.get("train_folds", [1, 2, 3, 4, 5, 6, 7, 8]))
    val_folds = list(split_cfg.get("val_folds", [9]))
    test_folds = list(split_cfg.get("test_folds", [10]))

    train_df, val_df, test_df, split_report = create_leakage_safe_splits(
        metadata_df=enriched_df,
        train_folds=train_folds,
        val_folds=val_folds,
        test_folds=test_folds,
        auto_resolve_group_leakage=True,
    )

    saved_manifests = save_split_manifests(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        report=split_report,
        splits_dir=splits_dir,
    )

    preproc_artifacts: dict[str, Any] = {}
    if fit_preprocessors:
        import numpy as np

        preproc_artifacts = fit_training_preprocessors(
            train_df=train_df,
            audio_dir=audio_dir,
            data_config=d_cfg,
            output_dir=processed_dir,
        )
        # Cache scaled train & val features (never touching test_df / Fold 10) for fast T2 & T3 training
        mfcc_scaler = preproc_artifacts["mfcc_scaler"]
        mel_normalizer = preproc_artifacts["mel_normalizer"]
        train_mfcc_raw = preproc_artifacts.pop("train_mfcc_raw")
        train_mel_raw = preproc_artifacts.pop("train_mel_raw")

        X_tr_mfcc = mfcc_scaler.transform(train_mfcc_raw).astype(np.float32)
        y_tr = train_df["binary_label"].to_numpy(dtype=np.int64)
        X_tr_mel = mel_normalizer.transform(train_mel_raw).astype(np.float32)

        audio_cfg = d_cfg["audio_signal"]
        mfcc_cfg = d_cfg["features"]["mfcc"]
        mel_cfg = d_cfg["features"]["mel_spectrogram"]
        target_sr = int(audio_cfg.get("target_sample_rate", 22050))
        target_dur = float(audio_cfg.get("target_duration_sec", 4.0))

        val_mfcc_list: list[np.ndarray] = []
        val_mel_list: list[np.ndarray] = []
        for row in val_df.itertuples(index=False):
            wav_path = audio_dir / f"fold{int(row.fold)}" / str(row.slice_file_name)
            waveform = load_and_standardize_audio(
                wav_path, target_sr=target_sr, target_duration_sec=target_dur
            )
            val_mfcc_list.append(
                extract_mfcc_features(
                    waveform,
                    sample_rate=target_sr,
                    n_mfcc=int(mfcc_cfg.get("n_mfcc", 40)),
                    n_fft=int(mfcc_cfg.get("n_fft", 2048)),
                    hop_length=int(mfcc_cfg.get("hop_length", 512)),
                    win_length=int(mfcc_cfg.get("win_length", 2048)),
                    n_mels=int(mfcc_cfg.get("n_mels", 128)),
                    f_min=float(mfcc_cfg.get("f_min", 0.0)),
                    f_max=float(mfcc_cfg.get("f_max", 11025.0)),
                )
            )
            val_mel_list.append(
                extract_log_mel_spectrogram(
                    waveform,
                    sample_rate=target_sr,
                    n_mels=int(mel_cfg.get("n_mels", 128)),
                    n_fft=int(mel_cfg.get("n_fft", 2048)),
                    hop_length=int(mel_cfg.get("hop_length", 512)),
                    win_length=int(mel_cfg.get("win_length", 2048)),
                    f_min=float(mel_cfg.get("f_min", 0.0)),
                    f_max=float(mel_cfg.get("f_max", 11025.0)),
                    top_db=float(mel_cfg.get("top_db", 80.0)),
                )
            )
        X_va_mfcc = mfcc_scaler.transform(np.stack(val_mfcc_list, axis=0)).astype(np.float32)
        y_va = val_df["binary_label"].to_numpy(dtype=np.int64)
        X_va_mel = mel_normalizer.transform(np.stack(val_mel_list, axis=0)).astype(np.float32)

        proc_path = Path(processed_dir)
        np.savez(
            proc_path / "train_val_mfcc_cache.npz",
            X_train=X_tr_mfcc,
            y_train=y_tr,
            X_val=X_va_mfcc,
            y_val=y_va,
        )
        np.savez(
            proc_path / "train_val_mel_cache.npz",
            X_train=X_tr_mel,
            y_train=y_tr,
            X_val=X_va_mel,
            y_val=y_va,
        )

    return {
        "validation_report": val_report,
        "split_report": split_report,
        "manifest_paths": saved_manifests,
        "preprocessor_artifacts": preproc_artifacts,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for dataset split and preprocessing preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare leakage-safe train/val/test split manifests and training-only scalers."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Path to UrbanSound8K directory (defaults to data/raw/UrbanSound8K).",
    )
    parser.add_argument(
        "--skip-scaler-fit",
        action="store_true",
        help="Generate split manifests only without fitting feature scalers.",
    )
    args = parser.parse_args(argv)

    logger = setup_logger("prepare_dataset")
    cfg = load_config()
    data_cfg = load_data_config()
    paths = resolve_project_paths(cfg)

    dataset_root = (
        Path(args.dataset_root).resolve()
        if args.dataset_root
        else paths["urbansound8k_root"]
    )

    try:
        result = run_dataset_preparation(
            dataset_root=dataset_root,
            splits_dir=paths["splits_dir"],
            processed_dir=paths["processed_data_dir"],
            data_config=data_cfg,
            fit_preprocessors=not args.skip_scaler_fit,
        )
    except DatasetNotFoundError as exc:
        logger.warning("%s", exc)
        logger.warning(
            "No split manifests or scaler artifacts were generated because the real dataset "
            "is not present yet."
        )
        return 2
    except Exception as exc:  # noqa: BLE001
        logger.error("Dataset preparation failed: %s", exc)
        return 1

    s_rep = result["split_report"]
    logger.info(
        "Splits prepared (leakage_free=%s): train=%d (siren=%d), val=%d (siren=%d), test=%d (siren=%d).",
        s_rep.is_leakage_free,
        s_rep.train_count,
        s_rep.train_siren_count,
        s_rep.val_count,
        s_rep.val_siren_count,
        s_rep.test_count,
        s_rep.test_siren_count,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

