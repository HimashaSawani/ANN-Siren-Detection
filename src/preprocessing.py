"""Dataset validation, binary label mapping, leakage-safe splitting, and preprocessing pipeline.

Module: NB6057CEM — Artificial Neural Networks
Step: 2 — Dataset Acquisition, Data Audit and Leakage-Safe Preparation (CW1 T1)

Preserves all Step 1 validation utilities while adding:
- Binary label verification and mapping (Label 1: siren, Label 0: non-siren urban sound).
- Predefined fold partitioning (Folds 1–8 Train, Fold 9 Val, Fold 10 Test).
- Source-recording (fsID) group isolation and duplicate leakage checks.
- Training-only fitting of MFCC standard scaler and Mel spectrogram normalizer.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import (
    DatasetNotFoundError,
    DatasetValidationError,
    load_config,
    resolve_project_paths,
)
from src.audio_features import (
    MFCCStandardScaler,
    MelSpectrogramNormalizer,
    extract_log_mel_spectrogram,
    extract_mfcc_features,
    load_and_standardize_audio,
    save_preprocessing_metadata,
)


URBANSOUND8K_SOURCE_URL: str = "https://urbansounddataset.weebly.com/urbansound8k.html"

REQUIRED_METADATA_COLUMNS: tuple[str, ...] = (
    "slice_file_name",
    "fsID",
    "start",
    "end",
    "salience",
    "fold",
    "classID",
    "class",
)

EXPECTED_FOLDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

EXPECTED_CLASSES: dict[int, str] = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music",
}

SIREN_CLASS_ID: int = 8
SIREN_CLASS_NAME: str = "siren"


class DataLeakageError(DatasetValidationError):
    """Raised when source-recording overlap, duplicate clips, or fold leakage is detected."""


@dataclass(frozen=True)
class DatasetValidationReport:
    """Structured report produced by the UrbanSound8K dataset validation utility."""

    dataset_root: Path
    metadata_path: Path
    audio_dir: Path
    dataset_root_exists: bool
    metadata_file_exists: bool
    audio_dir_exists: bool
    required_columns_present: bool
    missing_columns: list[str] = field(default_factory=list)
    available_classes: list[str] = field(default_factory=list)
    available_folds: list[int] = field(default_factory=list)
    missing_fold_dirs: list[str] = field(default_factory=list)
    total_metadata_rows: int = 0
    verified_audio_files: int = 0
    missing_audio_files: list[str] = field(default_factory=list)
    invalid_audio_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return True only if all structural, metadata, and file checks succeeded."""
        return (
            self.dataset_root_exists
            and self.metadata_file_exists
            and self.audio_dir_exists
            and self.required_columns_present
            and len(self.missing_columns) == 0
            and len(self.missing_fold_dirs) == 0
            and len(self.missing_audio_files) == 0
            and len(self.invalid_audio_files) == 0
            and len(self.errors) == 0
            and self.total_metadata_rows > 0
        )


@dataclass(frozen=True)
class SplitVerificationReport:
    """Verification report for train/validation/test partition integrity and leakage prevention."""

    train_folds: list[int]
    val_folds: list[int]
    test_folds: list[int]
    train_count: int
    val_count: int
    test_count: int
    train_siren_count: int
    val_siren_count: int
    test_siren_count: int
    train_non_siren_count: int
    val_non_siren_count: int
    test_non_siren_count: int
    unique_fsid_train: int
    unique_fsid_val: int
    unique_fsid_test: int
    overlapping_fsids_train_val: list[int]
    overlapping_fsids_train_test: list[int]
    overlapping_fsids_val_test: list[int]
    overlapping_clips: list[str]
    reassigned_fsids: list[int] = field(default_factory=list)

    @property
    def is_leakage_free(self) -> bool:
        """Return True if zero source-recording IDs or clip names cross partitions."""
        return (
            len(self.overlapping_fsids_train_val) == 0
            and len(self.overlapping_fsids_train_test) == 0
            and len(self.overlapping_fsids_val_test) == 0
            and len(self.overlapping_clips) == 0
        )

    @property
    def has_valid_class_representation(self) -> bool:
        """Return True if every split contains both siren (1) and non-siren (0) examples."""
        return (
            self.train_siren_count > 0
            and self.val_siren_count > 0
            and self.test_siren_count > 0
            and self.train_non_siren_count > 0
            and self.val_non_siren_count > 0
            and self.test_non_siren_count > 0
        )


def validate_urbansound8k_dataset(
    dataset_root: str | Path | None = None,
    metadata_path: str | Path | None = None,
    audio_dir: str | Path | None = None,
    config: dict[str, Any] | None = None,
    strict: bool = False,
    check_audio_files: bool = True,
) -> DatasetValidationReport:
    """Validate UrbanSound8K directory structure, metadata schema, classes, folds, and audio paths."""
    cfg = config if config is not None else load_config()
    resolved_paths = resolve_project_paths(cfg)

    if dataset_root is not None:
        root_path = Path(dataset_root).resolve()
        meta_path = (
            Path(metadata_path).resolve()
            if metadata_path is not None
            else root_path / "metadata" / "UrbanSound8K.csv"
        )
        aud_path = (
            Path(audio_dir).resolve()
            if audio_dir is not None
            else root_path / "audio"
        )
    else:
        root_path = resolved_paths["urbansound8k_root"]
        meta_path = (
            Path(metadata_path).resolve()
            if metadata_path is not None
            else resolved_paths["metadata_csv"]
        )
        aud_path = (
            Path(audio_dir).resolve()
            if audio_dir is not None
            else resolved_paths["audio_dir"]
        )

    errors: list[str] = []
    root_exists = root_path.is_dir()
    meta_exists = meta_path.is_file()
    audio_exists = aud_path.is_dir()

    if not root_exists:
        msg = (
            f"UrbanSound8K dataset root directory not found at '{root_path}'. "
            f"Please obtain the official dataset from {URBANSOUND8K_SOURCE_URL} "
            f"and place it under '{root_path}'."
        )
        errors.append(msg)
        if strict:
            raise DatasetNotFoundError(msg)

    if not meta_exists:
        msg = (
            f"UrbanSound8K metadata file not found at '{meta_path}'. "
            f"Expected 'metadata/UrbanSound8K.csv' inside '{root_path}'."
        )
        errors.append(msg)
        if strict:
            raise DatasetNotFoundError(msg)

    if not audio_exists:
        msg = (
            f"UrbanSound8K audio directory not found at '{aud_path}'. "
            f"Expected 'audio/fold1' .. 'audio/fold10' inside '{root_path}'."
        )
        errors.append(msg)
        if strict:
            raise DatasetNotFoundError(msg)

    missing_fold_dirs: list[str] = []
    if audio_exists:
        for fold_num in EXPECTED_FOLDS:
            fold_dir = aud_path / f"fold{fold_num}"
            if not fold_dir.is_dir():
                missing_fold_dirs.append(f"fold{fold_num}")
        if missing_fold_dirs:
            errors.append(
                f"Missing expected fold directories inside '{aud_path}': {missing_fold_dirs}"
            )

    if not meta_exists:
        return DatasetValidationReport(
            dataset_root=root_path,
            metadata_path=meta_path,
            audio_dir=aud_path,
            dataset_root_exists=root_exists,
            metadata_file_exists=meta_exists,
            audio_dir_exists=audio_exists,
            required_columns_present=False,
            missing_columns=list(REQUIRED_METADATA_COLUMNS),
            missing_fold_dirs=missing_fold_dirs,
            errors=errors,
        )

    try:
        df = pd.read_csv(meta_path)
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to read metadata CSV at '{meta_path}': {exc}"
        errors.append(msg)
        if strict:
            raise DatasetValidationError(msg) from exc
        return DatasetValidationReport(
            dataset_root=root_path,
            metadata_path=meta_path,
            audio_dir=aud_path,
            dataset_root_exists=root_exists,
            metadata_file_exists=meta_exists,
            audio_dir_exists=audio_exists,
            required_columns_present=False,
            missing_columns=list(REQUIRED_METADATA_COLUMNS),
            missing_fold_dirs=missing_fold_dirs,
            errors=errors,
        )

    missing_columns = [
        col for col in REQUIRED_METADATA_COLUMNS if col not in df.columns
    ]
    required_columns_present = len(missing_columns) == 0
    if not required_columns_present:
        msg = (
            f"Metadata CSV '{meta_path}' is missing required columns: {missing_columns}. "
            f"Found columns: {list(df.columns)}"
        )
        errors.append(msg)
        if strict:
            raise DatasetValidationError(msg)

    total_rows = int(len(df))
    if total_rows == 0:
        errors.append(f"Metadata CSV '{meta_path}' contains 0 rows.")

    available_classes: list[str] = []
    available_folds: list[int] = []
    missing_audio_files: list[str] = []
    invalid_audio_files: list[str] = []
    verified_audio_files = 0

    if required_columns_present and total_rows > 0:
        available_classes = sorted(df["class"].dropna().astype(str).unique().tolist())
        if "siren" not in available_classes:
            errors.append(
                f"Target positive class 'siren' not found in metadata classes: {available_classes}"
            )

        try:
            available_folds = sorted(
                df["fold"].dropna().astype(int).unique().tolist()
            )
        except (ValueError, TypeError) as exc:
            errors.append(f"Invalid non-integer fold identifiers in metadata: {exc}")

        invalid_folds = [f for f in available_folds if f not in EXPECTED_FOLDS]
        if invalid_folds:
            errors.append(
                f"Metadata contains fold identifiers outside expected range 1..10: {invalid_folds}"
            )

        if check_audio_files and audio_exists:
            for row in df.itertuples(index=False):
                fold_val = getattr(row, "fold")
                file_name = str(getattr(row, "slice_file_name"))
                rel_audio_path = f"fold{fold_val}/{file_name}"
                candidate_file = aud_path / f"fold{fold_val}" / file_name

                if not candidate_file.exists() or not candidate_file.is_file():
                    missing_audio_files.append(rel_audio_path)
                else:
                    try:
                        if candidate_file.stat().st_size <= 0:
                            invalid_audio_files.append(rel_audio_path)
                        else:
                            verified_audio_files += 1
                    except OSError:
                        invalid_audio_files.append(rel_audio_path)

            if missing_audio_files:
                errors.append(
                    f"{len(missing_audio_files)} audio file(s) referenced in metadata are missing."
                )
            if invalid_audio_files:
                errors.append(
                    f"{len(invalid_audio_files)} audio file(s) are empty (0 bytes) or unreadable."
                )

    report = DatasetValidationReport(
        dataset_root=root_path,
        metadata_path=meta_path,
        audio_dir=aud_path,
        dataset_root_exists=root_exists,
        metadata_file_exists=meta_exists,
        audio_dir_exists=audio_exists,
        required_columns_present=required_columns_present,
        missing_columns=missing_columns,
        available_classes=available_classes,
        available_folds=available_folds,
        missing_fold_dirs=missing_fold_dirs,
        total_metadata_rows=total_rows,
        verified_audio_files=verified_audio_files,
        missing_audio_files=missing_audio_files,
        invalid_audio_files=invalid_audio_files,
        errors=errors,
    )

    if strict and not report.is_valid:
        raise DatasetValidationError(
            "UrbanSound8K dataset validation failed:\n- " + "\n- ".join(report.errors)
        )

    return report


def map_binary_siren_label(class_id: int, class_name: str) -> int:
    """Validate an UrbanSound8K (classID, class) pair and map to binary label {0, 1}.

    Args:
        class_id: Integer class ID from UrbanSound8K metadata (`0..9`).
        class_name: String class name from UrbanSound8K metadata.

    Returns:
        ``1`` if ``class_id == 8`` and ``class_name == 'siren'``, else ``0``.

    Raises:
        DatasetValidationError: If ``class_id`` is unknown or conflicts with ``class_name``.
    """
    try:
        cid = int(class_id)
    except (ValueError, TypeError) as exc:
        raise DatasetValidationError(f"Invalid non-integer classID: {class_id!r}") from exc

    cname = str(class_name).strip()
    if cid not in EXPECTED_CLASSES:
        raise DatasetValidationError(
            f"Unknown UrbanSound8K classID={cid} (expected 0..9)."
        )
    expected_name = EXPECTED_CLASSES[cid]
    if cname != expected_name:
        raise DatasetValidationError(
            f"Inconsistent metadata label: classID={cid} expects class='{expected_name}', "
            f"but found class='{cname}'."
        )

    return 1 if cid == SIREN_CLASS_ID else 0


def load_and_enrich_metadata(
    metadata_path: str | Path,
    audio_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load UrbanSound8K metadata CSV, validate required columns and labels, and add binary target.

    Adds columns:
    - ``binary_label``: ``1`` for ``siren`` (`classID == 8`), ``0`` for ``non_siren``.
    - ``binary_class``: ``"siren"`` or ``"non_siren"``.
    - ``metadata_duration_sec``: ``end - start``.
    - ``relative_audio_path``: ``fold{fold}/{slice_file_name}``.
    - ``audio_path``: Resolved path inside ``audio_dir`` if ``audio_dir`` is provided.

    Raises:
        DatasetNotFoundError: If ``metadata_path`` does not exist.
        DatasetValidationError: If required columns are missing or label consistency fails.
    """
    meta_file = Path(metadata_path)
    if not meta_file.is_file():
        raise DatasetNotFoundError(f"Metadata CSV file not found at '{meta_file}'.")

    df = pd.read_csv(meta_file)
    missing_cols = [c for c in REQUIRED_METADATA_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DatasetValidationError(
            f"Metadata CSV '{meta_file}' is missing required columns: {missing_cols}"
        )
    if df.empty:
        raise DatasetValidationError(f"Metadata CSV '{meta_file}' is empty.")

    if df[list(REQUIRED_METADATA_COLUMNS)].isna().any().any():
        na_cols = (
            df[list(REQUIRED_METADATA_COLUMNS)]
            .columns[df[list(REQUIRED_METADATA_COLUMNS)].isna().any()]
            .tolist()
        )
        raise DatasetValidationError(
            f"Metadata CSV '{meta_file}' contains missing/null values in columns: {na_cols}"
        )

    binary_labels = [
        map_binary_siren_label(cid, cname)
        for cid, cname in zip(df["classID"], df["class"])
    ]
    enriched = df.copy()
    enriched["binary_label"] = np.asarray(binary_labels, dtype=np.int64)
    enriched["binary_class"] = np.where(
        enriched["binary_label"] == 1, "siren", "non_siren"
    )
    enriched["metadata_duration_sec"] = (
        enriched["end"].astype(float) - enriched["start"].astype(float)
    )
    if (enriched["metadata_duration_sec"] <= 0.0).any():
        raise DatasetValidationError(
            "Metadata contains non-positive slice duration (end <= start)."
        )

    enriched["relative_audio_path"] = [
        f"fold{int(f)}/{fname}"
        for f, fname in zip(enriched["fold"], enriched["slice_file_name"])
    ]
    if audio_dir is not None:
        base_audio = Path(audio_dir).resolve()
        enriched["audio_path"] = [
            str(base_audio / f"fold{int(f)}" / str(fname))
            for f, fname in zip(enriched["fold"], enriched["slice_file_name"])
        ]

    return enriched


def verify_partition_integrity(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_folds: list[int],
    val_folds: list[int],
    test_folds: list[int],
    reassigned_fsids: list[int] | None = None,
) -> SplitVerificationReport:
    """Verify source-recording (fsID) isolation, clip uniqueness, and class balance across splits."""
    fsid_train = set(train_df["fsID"].astype(int).unique())
    fsid_val = set(val_df["fsID"].astype(int).unique())
    fsid_test = set(test_df["fsID"].astype(int).unique())

    overlap_tv = sorted(fsid_train.intersection(fsid_val))
    overlap_tt = sorted(fsid_train.intersection(fsid_test))
    overlap_vt = sorted(fsid_val.intersection(fsid_test))

    clips_train = set(train_df["slice_file_name"].astype(str))
    clips_val = set(val_df["slice_file_name"].astype(str))
    clips_test = set(test_df["slice_file_name"].astype(str))
    clip_overlaps = sorted(
        clips_train.intersection(clips_val)
        | clips_train.intersection(clips_test)
        | clips_val.intersection(clips_test)
    )

    return SplitVerificationReport(
        train_folds=list(train_folds),
        val_folds=list(val_folds),
        test_folds=list(test_folds),
        train_count=int(len(train_df)),
        val_count=int(len(val_df)),
        test_count=int(len(test_df)),
        train_siren_count=int((train_df["binary_label"] == 1).sum()),
        val_siren_count=int((val_df["binary_label"] == 1).sum()),
        test_siren_count=int((test_df["binary_label"] == 1).sum()),
        train_non_siren_count=int((train_df["binary_label"] == 0).sum()),
        val_non_siren_count=int((val_df["binary_label"] == 0).sum()),
        test_non_siren_count=int((test_df["binary_label"] == 0).sum()),
        unique_fsid_train=len(fsid_train),
        unique_fsid_val=len(fsid_val),
        unique_fsid_test=len(fsid_test),
        overlapping_fsids_train_val=overlap_tv,
        overlapping_fsids_train_test=overlap_tt,
        overlapping_fsids_val_test=overlap_vt,
        overlapping_clips=clip_overlaps,
        reassigned_fsids=sorted(reassigned_fsids) if reassigned_fsids else [],
    )


def create_leakage_safe_splits(
    metadata_df: pd.DataFrame,
    train_folds: list[int] | None = None,
    val_folds: list[int] | None = None,
    test_folds: list[int] | None = None,
    auto_resolve_group_leakage: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitVerificationReport]:
    """Create deterministic train/val/test partitions from official UrbanSound8K folds.

    Args:
        metadata_df: Enriched metadata DataFrame containing ``fold``, ``fsID``,
            ``slice_file_name``, and ``binary_label``.
        train_folds: Folds assigned to training (default: ``[1, 2, 3, 4, 5, 6, 7, 8]``).
        val_folds: Folds assigned to validation (default: ``[9]``).
        test_folds: Folds assigned to held-out test (default: ``[10]``).
        auto_resolve_group_leakage: If ``True`` and any ``fsID`` spans multiple folds,
            consolidates all clips of that ``fsID`` into its majority partition before
            re-verifying. If ``False`` (default), raises ``DataLeakageError`` immediately.

    Returns:
        Tuple of ``(train_df, val_df, test_df, report)``.

    Raises:
        DataLeakageError: If fold sets overlap, duplicate clips exist, ``fsID`` crosses
            partitions, or any split lacks positive/negative class representation.
    """
    tr_folds = list(train_folds) if train_folds is not None else [1, 2, 3, 4, 5, 6, 7, 8]
    va_folds = list(val_folds) if val_folds is not None else [9]
    te_folds = list(test_folds) if test_folds is not None else [10]

    if (
        set(tr_folds).intersection(va_folds)
        or set(tr_folds).intersection(te_folds)
        or set(va_folds).intersection(te_folds)
    ):
        raise DataLeakageError(
            f"Fold definitions overlap: train={tr_folds}, val={va_folds}, test={te_folds}."
        )

    df = metadata_df.copy()
    if "binary_label" not in df.columns:
        df["binary_label"] = [
            map_binary_siren_label(cid, cname)
            for cid, cname in zip(df["classID"], df["class"])
        ]

    # Check exact duplicate clip filenames in dataset
    if df["slice_file_name"].duplicated().any():
        dups = df.loc[df["slice_file_name"].duplicated(), "slice_file_name"].unique().tolist()
        raise DataLeakageError(
            f"Duplicate slice_file_name entries detected in metadata: {dups[:10]}"
        )

    # Assign initial partition by fold
    fold_to_split: dict[int, str] = {}
    for f in tr_folds:
        fold_to_split[int(f)] = "train"
    for f in va_folds:
        fold_to_split[int(f)] = "val"
    for f in te_folds:
        fold_to_split[int(f)] = "test"

    unknown_folds = sorted(set(df["fold"].astype(int).unique()) - set(fold_to_split.keys()))
    if unknown_folds:
        raise DataLeakageError(
            f"Metadata contains fold numbers not covered by split protocol: {unknown_folds}"
        )

    df["partition"] = df["fold"].astype(int).map(fold_to_split)

    # Check if any fsID spans multiple partitions
    reassigned: list[int] = []
    fsid_partition_counts = df.groupby("fsID")["partition"].nunique()
    leaking_fsids = fsid_partition_counts[fsid_partition_counts > 1].index.astype(int).tolist()

    if leaking_fsids and auto_resolve_group_leakage:
        for fsid in leaking_fsids:
            sub = df.loc[df["fsID"] == fsid, "partition"]
            # Assign to majority partition (preferring train > val > test on ties)
            counts = sub.value_counts()
            majority_split = str(counts.index[0])
            df.loc[df["fsID"] == fsid, "partition"] = majority_split
            reassigned.append(int(fsid))

    # Sort deterministically by (fold, slice_file_name)
    df = df.sort_values(by=["fold", "slice_file_name"], kind="mergesort").reset_index(drop=True)

    train_df = df[df["partition"] == "train"].reset_index(drop=True)
    val_df = df[df["partition"] == "val"].reset_index(drop=True)
    test_df = df[df["partition"] == "test"].reset_index(drop=True)

    report = verify_partition_integrity(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_folds=tr_folds,
        val_folds=va_folds,
        test_folds=te_folds,
        reassigned_fsids=reassigned,
    )

    if not report.is_leakage_free:
        raise DataLeakageError(
            f"Source recording (fsID) or clip leakage detected across splits! "
            f"train-val fsID overlap={report.overlapping_fsids_train_val}, "
            f"train-test fsID overlap={report.overlapping_fsids_train_test}, "
            f"val-test fsID overlap={report.overlapping_fsids_val_test}, "
            f"clip overlap={report.overlapping_clips}."
        )

    if not report.has_valid_class_representation:
        raise DataLeakageError(
            "One or more splits lack positive (siren) or negative (non_siren) examples: "
            f"train(siren={report.train_siren_count}, non_siren={report.train_non_siren_count}), "
            f"val(siren={report.val_siren_count}, non_siren={report.val_non_siren_count}), "
            f"test(siren={report.test_siren_count}, non_siren={report.test_non_siren_count})."
        )

    return train_df, val_df, test_df, report


def save_split_manifests(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    report: SplitVerificationReport,
    splits_dir: str | Path,
) -> dict[str, Path]:
    """Save deterministic train.csv, val.csv, test.csv, and split_summary.json."""
    out_dir = Path(splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"
    summary_path = out_dir / "split_summary.json"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary_payload = {
        "protocol_name": "fixed_fold_coursework_split",
        "note": (
            "Coursework evaluation protocol using official UrbanSound8K predefined folds "
            "(Folds 1-8 train, Fold 9 val, Fold 10 test). Held-out test partition is reserved "
            "for final evaluation only."
        ),
        "is_leakage_free": report.is_leakage_free,
        "has_valid_class_representation": report.has_valid_class_representation,
        "verification": asdict(report),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return {
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "summary": summary_path,
    }


def fit_training_preprocessors(
    train_df: pd.DataFrame,
    audio_dir: str | Path,
    data_config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Fit MFCCStandardScaler and MelSpectrogramNormalizer exclusively on training audio clips.

    Args:
        train_df: Training partition manifest DataFrame (must have ``partition == 'train'``).
        audio_dir: Root directory containing ``fold1``..``fold10`` audio files.
        data_config: Loaded ``data_config.yaml`` dictionary.
        output_dir: Directory to save fitted scaler/normalizer JSON artifacts.

    Returns:
        Dictionary containing fitted ``mfcc_scaler``, ``mel_normalizer``, and artifact paths.

    Raises:
        DataLeakageError: If ``train_df`` contains any non-training partition rows.
    """
    if "partition" in train_df.columns:
        non_train = set(train_df["partition"].astype(str).unique()) - {"train"}
        if non_train:
            raise DataLeakageError(
                f"Cannot fit preprocessors: manifest contains non-train partitions {non_train}."
            )

    audio_cfg = data_config["audio_signal"]
    mfcc_cfg = data_config["features"]["mfcc"]
    mel_cfg = data_config["features"]["mel_spectrogram"]

    target_sr = int(audio_cfg.get("target_sample_rate", 22050))
    target_dur = float(audio_cfg.get("target_duration_sec", 4.0))
    base_audio = Path(audio_dir).resolve()

    mfcc_list: list[np.ndarray] = []
    mel_list: list[np.ndarray] = []

    for row in train_df.itertuples(index=False):
        fold_num = int(getattr(row, "fold"))
        fname = str(getattr(row, "slice_file_name"))
        wav_path = base_audio / f"fold{fold_num}" / fname
        waveform = load_and_standardize_audio(
            wav_path, target_sr=target_sr, target_duration_sec=target_dur
        )
        mfcc_vec = extract_mfcc_features(
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
        log_mel = extract_log_mel_spectrogram(
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
        mfcc_list.append(mfcc_vec)
        mel_list.append(log_mel)

    mfcc_matrix = np.stack(mfcc_list, axis=0)
    mel_batch = np.stack(mel_list, axis=0)

    mfcc_scaler = MFCCStandardScaler().fit(mfcc_matrix, split_name="train")
    mel_normalizer = MelSpectrogramNormalizer().fit(mel_batch, split_name="train")

    proc_dir = Path(output_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)
    mfcc_scaler_path = mfcc_scaler.save(proc_dir / "mfcc_scaler.json")
    mel_norm_path = mel_normalizer.save(proc_dir / "mel_norm_stats.json")
    preproc_meta_path = save_preprocessing_metadata(
        output_path=proc_dir / "preprocessing_config.json",
        data_config=data_config,
        mfcc_scaler_path=mfcc_scaler_path,
        mel_norm_path=mel_norm_path,
    )

    return {
        "mfcc_scaler": mfcc_scaler,
        "mel_normalizer": mel_normalizer,
        "mfcc_scaler_path": mfcc_scaler_path,
        "mel_norm_path": mel_norm_path,
        "preprocessing_config_path": preproc_meta_path,
        "train_mfcc_raw": mfcc_matrix,
        "train_mel_raw": mel_batch,
    }
