"""PyTorch-compatible Dataset and DataLoader utilities for UrbanSound8K siren detection.

Module: NB6057CEM — Artificial Neural Networks
Step: 2 — Dataset Acquisition, Data Audit and Leakage-Safe Preparation (CW1 T1)

Supports:
- Lazy on-demand audio loading from fixed split manifests (avoids loading raw dataset into RAM).
- Feature representations: 'mfcc' (80-D vector for RF/MLP) and 'mel' (1x128x173 tensor for 2D CNN).
- Binary labels: 1 (siren) vs. 0 (non-siren urban sound).
- Training-only waveform augmentation (strictly disabled on 'val' and 'test' splits).
- Training-fitted normalization (MFCCStandardScaler / MelSpectrogramNormalizer).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src import DatasetNotFoundError, DatasetValidationError
from src.audio_features import (
    MFCCStandardScaler,
    MelSpectrogramNormalizer,
    extract_log_mel_spectrogram,
    extract_mfcc_features,
    load_and_standardize_audio,
)
from src.augmentation import AugmentationPolicyError, TrainingAudioAugmenter
from src.preprocessing import REQUIRED_METADATA_COLUMNS, map_binary_siren_label


VALID_REPRESENTATIONS: tuple[str, ...] = ("mfcc", "mel", "waveform")
VALID_SPLITS: tuple[str, ...] = ("train", "val", "test")


class UrbanSoundSirenDataset(Dataset):
    """Lazy-loading PyTorch Dataset for binary siren classification on UrbanSound8K.

    Each item corresponds to one audio slice from a fixed partition manifest.
    """

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        audio_dir: str | Path,
        split: str = "train",
        representation: str = "mfcc",
        target_sr: int = 22050,
        target_duration_sec: float = 4.0,
        mfcc_scaler: MFCCStandardScaler | None = None,
        mel_normalizer: MelSpectrogramNormalizer | None = None,
        augmenter: TrainingAudioAugmenter | None = None,
        return_metadata: bool = False,
    ) -> None:
        """Initialize the dataset from a split manifest CSV or DataFrame.

        Args:
            manifest: Path to a partition CSV (`train.csv`, `val.csv`, `test.csv`) or DataFrame.
            audio_dir: Root directory containing `fold1`..`fold10` subdirectories.
            split: Partition identifier (`"train"`, `"val"`, or `"test"`).
            representation: Feature type (`"mfcc"`, `"mel"`, or `"waveform"`).
            target_sr: Canonical sampling rate in Hz (default: 22,050).
            target_duration_sec: Canonical clip duration in seconds (default: 4.0).
            mfcc_scaler: Optional fitted `MFCCStandardScaler` (fitted on train only).
            mel_normalizer: Optional fitted `MelSpectrogramNormalizer` (fitted on train only).
            augmenter: Optional `TrainingAudioAugmenter`. Forbidden if `split != 'train'`.
            return_metadata: If `True`, `__getitem__` returns `(features, label, metadata_dict)`.
                If `False` (default), returns `(features, label)`.
        """
        norm_split = split.strip().lower()
        if norm_split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. Expected one of {VALID_SPLITS}."
            )
        norm_repr = representation.strip().lower()
        if norm_repr not in VALID_REPRESENTATIONS:
            raise ValueError(
                f"Invalid representation '{representation}'. Expected one of {VALID_REPRESENTATIONS}."
            )

        if norm_split != "train" and augmenter is not None and augmenter.enabled:
            raise AugmentationPolicyError(
                f"Cannot attach an enabled TrainingAudioAugmenter to evaluation split='{norm_split}'."
            )

        if isinstance(manifest, (str, Path)):
            manifest_path = Path(manifest)
            if not manifest_path.is_file():
                raise DatasetNotFoundError(
                    f"Split manifest CSV not found at '{manifest_path}'."
                )
            df = pd.read_csv(manifest_path)
        elif isinstance(manifest, pd.DataFrame):
            df = manifest.copy()
        else:
            raise TypeError("manifest must be a file path or pandas DataFrame.")

        missing_cols = [c for c in REQUIRED_METADATA_COLUMNS if c not in df.columns]
        if missing_cols:
            raise DatasetValidationError(
                f"Split manifest is missing required columns: {missing_cols}"
            )
        if df.empty:
            raise DatasetValidationError("Split manifest contains 0 rows.")

        if "binary_label" not in df.columns:
            df["binary_label"] = [
                map_binary_siren_label(cid, cname)
                for cid, cname in zip(df["classID"], df["class"])
            ]

        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir).resolve()
        self.split = norm_split
        self.representation = norm_repr
        self.target_sr = int(target_sr)
        self.target_duration_sec = float(target_duration_sec)
        self.mfcc_scaler = mfcc_scaler
        self.mel_normalizer = mel_normalizer
        self.augmenter = augmenter
        self.return_metadata = bool(return_metadata)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_clip_path(self, row: pd.Series) -> Path:
        fold_num = int(row["fold"])
        slice_name = str(row["slice_file_name"])
        return self.audio_dir / f"fold{fold_num}" / slice_name

    def get_sample_metadata(self, index: int) -> dict[str, Any]:
        """Return metadata dictionary for sample at ``index`` without loading audio."""
        row = self.df.iloc[index]
        clip_path = self._resolve_clip_path(row)
        return {
            "index": int(index),
            "slice_file_name": str(row["slice_file_name"]),
            "fsID": int(row["fsID"]),
            "fold": int(row["fold"]),
            "classID": int(row["classID"]),
            "class": str(row["class"]),
            "binary_label": int(row["binary_label"]),
            "split": self.split,
            "audio_path": str(clip_path),
        }

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if index < 0 or index >= len(self.df):
            raise IndexError(f"Dataset index {index} out of range for length {len(self.df)}.")

        row = self.df.iloc[index]
        clip_path = self._resolve_clip_path(row)
        waveform = load_and_standardize_audio(
            clip_path,
            target_sr=self.target_sr,
            target_duration_sec=self.target_duration_sec,
        )

        # Apply stochastic augmentation ONLY on training split when enabled
        if self.augmenter is not None and self.augmenter.enabled:
            waveform = self.augmenter(
                waveform, split=self.split, sample_index=int(index)
            )

        if self.representation == "mfcc":
            feat_arr = extract_mfcc_features(waveform, sample_rate=self.target_sr)
            if self.mfcc_scaler is not None:
                feat_arr = self.mfcc_scaler.transform(feat_arr)
        elif self.representation == "mel":
            feat_arr = extract_log_mel_spectrogram(waveform, sample_rate=self.target_sr)
            if self.mel_normalizer is not None:
                feat_arr = self.mel_normalizer.transform(feat_arr)
        else:
            feat_arr = waveform

        feature_tensor = torch.from_numpy(np.ascontiguousarray(feat_arr, dtype=np.float32))
        label_val = int(row["binary_label"])
        label_tensor = torch.tensor(label_val, dtype=torch.long)

        if self.return_metadata:
            return feature_tensor, label_tensor, self.get_sample_metadata(index)
        return feature_tensor, label_tensor


def create_dataloaders(
    train_manifest: str | Path | pd.DataFrame,
    val_manifest: str | Path | pd.DataFrame,
    test_manifest: str | Path | pd.DataFrame,
    audio_dir: str | Path,
    representation: str = "mfcc",
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    mfcc_scaler: MFCCStandardScaler | None = None,
    mel_normalizer: MelSpectrogramNormalizer | None = None,
    train_augmenter: TrainingAudioAugmenter | None = None,
    return_metadata: bool = False,
) -> dict[str, DataLoader]:
    """Create reproducible train, validation, and test PyTorch DataLoaders.

    Guarantees:
    - Deterministic seeded shuffling on the ``train`` DataLoader.
    - ``shuffle=False`` and ``augmenter=None`` on ``val`` and ``test`` DataLoaders.
    """
    train_ds = UrbanSoundSirenDataset(
        manifest=train_manifest,
        audio_dir=audio_dir,
        split="train",
        representation=representation,
        mfcc_scaler=mfcc_scaler,
        mel_normalizer=mel_normalizer,
        augmenter=train_augmenter,
        return_metadata=return_metadata,
    )
    val_ds = UrbanSoundSirenDataset(
        manifest=val_manifest,
        audio_dir=audio_dir,
        split="val",
        representation=representation,
        mfcc_scaler=mfcc_scaler,
        mel_normalizer=mel_normalizer,
        augmenter=None,
        return_metadata=return_metadata,
    )
    test_ds = UrbanSoundSirenDataset(
        manifest=test_manifest,
        audio_dir=audio_dir,
        split="test",
        representation=representation,
        mfcc_scaler=mfcc_scaler,
        mel_normalizer=mel_normalizer,
        augmenter=None,
        return_metadata=return_metadata,
    )

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

