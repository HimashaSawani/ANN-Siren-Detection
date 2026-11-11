"""Unit tests for leakage-safe fold partitioning, fsID source isolation, and split reproducibility.

Module: NB6057CEM — Artificial Neural Networks
Step: 2 — Dataset Acquisition, Data Audit and Leakage-Safe Preparation (CW1 T1)

Covers:
- Deterministic split reproducibility (Folds 1-8 Train, Fold 9 Val, Fold 10 Test).
- Source-recording (fsID) group isolation across train, validation, and test splits.
- Detection of cross-partition fsID leakage and duplicate slice filenames.
- End-to-end dataset preparation (`run_dataset_preparation`) on synthetic test fixtures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io.wavfile as wavfile

from scripts.prepare_dataset import run_dataset_preparation
from src.preprocessing import (
    DataLeakageError,
    EXPECTED_CLASSES,
    EXPECTED_FOLDS,
    create_leakage_safe_splits,
    load_and_enrich_metadata,
)


def _create_synthetic_urbansound_fixture(root_dir: Path) -> Path:
    """Create a complete 10-fold synthetic UrbanSound8K fixture inside `root_dir`."""
    us_root = root_dir / "UrbanSound8K"
    meta_dir = us_root / "metadata"
    audio_dir = us_root / "audio"
    meta_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for fold in EXPECTED_FOLDS:
        fold_dir = audio_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        t = np.linspace(0.0, 1.5, int(22050 * 1.5), endpoint=False, dtype=np.float32)
        pcm = np.clip(0.4 * np.sin(2.0 * np.pi * 800.0 * t) * 32767.0, -32768, 32767).astype(
            np.int16
        )

        siren_fsid = 10000 + fold * 10 + 1
        siren_name = f"{siren_fsid}-8-0-0.wav"
        wavfile.write(str(fold_dir / siren_name), 22050, pcm)
        rows.append(
            {
                "slice_file_name": siren_name,
                "fsID": siren_fsid,
                "start": 0.0,
                "end": 1.5,
                "salience": 1,
                "fold": fold,
                "classID": 8,
                "class": "siren",
            }
        )

        other_cid = (fold - 1) % 8
        other_cname = EXPECTED_CLASSES[other_cid]
        other_fsid = 20000 + fold * 10 + 2
        other_name = f"{other_fsid}-{other_cid}-0-0.wav"
        wavfile.write(str(fold_dir / other_name), 22050, pcm)
        rows.append(
            {
                "slice_file_name": other_name,
                "fsID": other_fsid,
                "start": 0.0,
                "end": 1.5,
                "salience": 1,
                "fold": fold,
                "classID": other_cid,
                "class": other_cname,
            }
        )

    pd.DataFrame(rows).to_csv(meta_dir / "UrbanSound8K.csv", index=False)
    return us_root


def test_split_reproducibility_and_zero_source_overlap(tmp_path: Path) -> None:
    """Verify repeated split creation is 100% deterministic and has zero fsID overlap."""
    us_root = _create_synthetic_urbansound_fixture(tmp_path)
    df = load_and_enrich_metadata(
        us_root / "metadata" / "UrbanSound8K.csv", audio_dir=us_root / "audio"
    )

    train_1, val_1, test_1, rep_1 = create_leakage_safe_splits(df)
    train_2, val_2, test_2, rep_2 = create_leakage_safe_splits(df)

    assert rep_1.is_leakage_free is True
    assert rep_1.has_valid_class_representation is True
    assert rep_1 == rep_2

    assert train_1.equals(train_2)
    assert val_1.equals(val_2)
    assert test_1.equals(test_2)

    # Verify folds 1..8 in train, fold 9 in val, fold 10 in test
    assert sorted(train_1["fold"].unique().tolist()) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert sorted(val_1["fold"].unique().tolist()) == [9]
    assert sorted(test_1["fold"].unique().tolist()) == [10]

    # Verify zero fsID overlap across partitions
    fsid_train = set(train_1["fsID"])
    fsid_val = set(val_1["fsID"])
    fsid_test = set(test_1["fsID"])
    assert fsid_train.isdisjoint(fsid_val)
    assert fsid_train.isdisjoint(fsid_test)
    assert fsid_val.isdisjoint(fsid_test)


def test_source_recording_leakage_detection_and_resolution(tmp_path: Path) -> None:
    """Verify cross-partition fsID overlap raises DataLeakageError unless auto-resolved."""
    us_root = _create_synthetic_urbansound_fixture(tmp_path)
    df = load_and_enrich_metadata(us_root / "metadata" / "UrbanSound8K.csv")

    # Inject a cross-fold fsID collision between fold 1 (train) and fold 10 (test)
    leaking_df = df.copy()
    shared_fsid = int(leaking_df.loc[leaking_df["fold"] == 1, "fsID"].iloc[0])
    leaking_df.loc[leaking_df["fold"] == 10, "fsID"] = shared_fsid

    with pytest.raises(DataLeakageError, match="Source recording \\(fsID\\) or clip leakage"):
        create_leakage_safe_splits(leaking_df, auto_resolve_group_leakage=False)

    # If auto_resolve_group_leakage=True, related recordings are grouped into one partition
    # (as long as all partitions retain class representation, let's leak only one clip in fold 9)
    leaking_single = df.copy()
    fold9_siren_idx = leaking_single.index[
        (leaking_single["fold"] == 9) & (leaking_single["binary_label"] == 0)
    ][0]
    # Add an extra non-siren row in fold 9 first so fold 9 still has both classes after reassignment
    extra_row = leaking_single.loc[fold9_siren_idx].copy()
    extra_row["slice_file_name"] = "99999-0-0-0.wav"
    extra_row["fsID"] = 99999
    leaking_single = pd.concat([leaking_single, pd.DataFrame([extra_row])], ignore_index=True)
    leaking_single.loc[fold9_siren_idx, "fsID"] = shared_fsid

    _, _, _, resolved_rep = create_leakage_safe_splits(
        leaking_single, auto_resolve_group_leakage=True
    )
    assert resolved_rep.is_leakage_free is True
    assert shared_fsid in resolved_rep.reassigned_fsids


def test_duplicate_slice_filename_rejected(tmp_path: Path) -> None:
    """Verify duplicate slice_file_name entries raise DataLeakageError."""
    us_root = _create_synthetic_urbansound_fixture(tmp_path)
    df = load_and_enrich_metadata(us_root / "metadata" / "UrbanSound8K.csv")
    dup_df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    with pytest.raises(DataLeakageError, match="Duplicate slice_file_name"):
        create_leakage_safe_splits(dup_df)


def test_end_to_end_dataset_preparation_on_fixture(tmp_path: Path) -> None:
    """Verify run_dataset_preparation saves manifests and training-only scalers."""
    us_root = _create_synthetic_urbansound_fixture(tmp_path)
    splits_out = tmp_path / "splits"
    proc_out = tmp_path / "processed"

    result = run_dataset_preparation(
        dataset_root=us_root,
        splits_dir=splits_out,
        processed_dir=proc_out,
        fit_preprocessors=True,
    )

    assert (splits_out / "train.csv").is_file()
    assert (splits_out / "val.csv").is_file()
    assert (splits_out / "test.csv").is_file()
    assert (splits_out / "split_summary.json").is_file()
    assert (proc_out / "mfcc_scaler.json").is_file()
    assert (proc_out / "mel_norm_stats.json").is_file()
    assert (proc_out / "preprocessing_config.json").is_file()

    split_rep = result["split_report"]
    assert split_rep.train_count == 16
    assert split_rep.val_count == 2
    assert split_rep.test_count == 2
    assert split_rep.is_leakage_free is True
