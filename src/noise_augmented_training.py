"""Phase 2 — Controlled Noise-Augmented Deep MLP Training & Robustness Pipeline.

Module: NB6057CEM — Artificial Neural Networks
Portfolio Upgrade: Phase 2 — Noise-Augmented Audio Training

Protocol:
1. Augmented Partition: Strictly training split (UrbanSound8K Folds 1–8, N = 7,095).
2. Leakage Controls:
   - Validation split (Fold 9, N = 800) is kept strictly clean and unaugmented for early stopping.
   - Held-out test split (Fold 10, N = 837) is reserved strictly for evaluation under controlled test conditions.
   - Feature scaling uses the frozen, training-fitted MFCCStandardScaler (no stats refitted).
3. Audio-Level Perturbations:
   - Injected directly into the 22,050 Hz standardized 1D waveform before feature extraction.
   - Documented noise types:
     a) Zero-mean White Gaussian noise at SNR uniform in [10.0, 25.0] dB.
     b) Non-siren urban ambient clutter (from training folds) at SNR uniform in [10.0, 25.0] dB.
   - Augmentation probability: p = 0.5 per training sample (50% augmented, 50% clean).
   - Recomputes 80 MFCC summary statistics (40 means, 40 standard deviations) from augmented audio.
4. Model Training:
   - Identical architecture: DeepMLP (80 -> 128 -> 64 -> 1, dropout=0.3).
   - BCEWithLogitsLoss with positive class weight (pos_weight = 8.29).
   - Optimizer: Adam (lr = 1e-3, weight_decay = 1e-4).
   - Monitored on clean Validation Macro-F1 with patience = 10 epochs.
5. Checkpoint:
   - Saved to `checkpoints/deep_mlp_augmented_seed{seed}.pt` (preserving clean checkpoint).
"""

from __future__ import annotations

import concurrent.futures
import gc
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

# Ensure persistent, writable Matplotlib cache directory
_mpl_dir = os.path.join(tempfile.gettempdir(), "matplotlib_cache")
os.makedirs(_mpl_dir, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_dir)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src import (
    load_config,
    load_data_config,
    load_model_config,
    resolve_project_paths,
    set_random_seeds,
    setup_logger,
)
from src.audio_features import (
    MFCCStandardScaler,
    extract_mfcc_features,
    load_and_standardize_audio,
)
from src.augmentation import (
    AugmentationPolicyError,
    add_controlled_background_noise,
)
from src.evaluate import (
    compute_classification_metrics,
    enforce_no_test_split_access,
)
from src.models import DeepMLP
from src.train import (
    EarlyStoppingTracker,
    compute_training_pos_weight,
    load_pytorch_checkpoint,
    make_tensor_dataloaders,
    save_pytorch_checkpoint,
)

logger = logging.getLogger(__name__)


def generate_augmented_training_mfcc(
    dataset_root: str | Path,
    splits_dir: str | Path,
    processed_dir: str | Path,
    scaler_path: str | Path | None = None,
    seed: int = 42,
    p_augment: float = 0.5,
    snr_range: tuple[float, float] = (10.0, 25.0),
    max_workers: int = 4,
    force_recompute: bool = False,
) -> Path:
    """Generate and cache recomputed 80-D MFCCs from audio-level noise augmented training split.

    Uses clean pre-extracted features for unaugmented samples (1 - p = 0.5) and recomputes
    MFCC features from perturbed waveforms for augmented samples (p = 0.5).

    Args:
        dataset_root: Path to UrbanSound8K dataset root.
        splits_dir: Path containing `train.csv`.
        processed_dir: Path to save the cached `train_mfcc_augmented_seed{seed}.npz`.
        scaler_path: Path to frozen `mfcc_scaler.json`.
        seed: Random seed for deterministic reproducibility.
        p_augment: Per-clip noise augmentation probability in [0.0, 1.0].
        snr_range: (min_snr_db, max_snr_db) for additive noise injection.
        max_workers: Worker threads for audio decoding.
        force_recompute: Whether to overwrite existing cache.

    Returns:
        Path to the saved `.npz` file containing `X_train` and `y_train`.
    """
    proc_dir = Path(processed_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)
    cache_path = proc_dir / f"train_mfcc_augmented_seed{seed}.npz"

    if cache_path.is_file() and not force_recompute:
        logger.info("Using cached noise-augmented training MFCCs from '%s'.", cache_path)
        return cache_path

    splits_path = Path(splits_dir)
    train_csv = splits_path / "train.csv"
    if not train_csv.is_file():
        raise FileNotFoundError(f"Train manifest not found at '{train_csv}'.")

    train_df = pd.read_csv(train_csv)
    n_samples = len(train_df)

    # Load clean training cache as base
    clean_cache_path = proc_dir / "train_val_mfcc_cache.npz"
    if not clean_cache_path.is_file():
        raise FileNotFoundError(f"Clean train_val cache not found at '{clean_cache_path}'.")
    clean_cache = np.load(clean_cache_path)
    X_train_aug = np.copy(clean_cache["X_train"].astype(np.float32))
    y_train = clean_cache["y_train"].astype(np.int64)

    # Resolve audio directory
    ds_root = Path(dataset_root)
    audio_dir = ds_root / "audio" if (ds_root / "audio").is_dir() else ds_root

    # Load frozen scaler
    if scaler_path is None:
        scaler_path = proc_dir / "mfcc_scaler.json"
    scaler = MFCCStandardScaler.load(scaler_path)

    # Collect non-siren clips for environmental noise mixing
    non_siren_rows = train_df[train_df["binary_label"] == 0]
    non_siren_tuples = list(non_siren_rows[["fold", "slice_file_name"]].itertuples(index=False))

    # Identify samples selected for audio-level augmentation
    aug_tasks: list[tuple[int, Any, float, bool, int]] = []
    for idx, row in enumerate(train_df.itertuples(index=False)):
        rng = np.random.default_rng([int(seed), int(idx)])
        if rng.random() < p_augment:
            snr = float(rng.uniform(snr_range[0], snr_range[1]))
            use_urban = (rng.random() >= 0.70) and len(non_siren_tuples) > 0
            bg_pick = int(rng.integers(0, len(non_siren_tuples))) if use_urban else 0
            aug_tasks.append((idx, row, snr, use_urban, bg_pick))

    n_augmented = len(aug_tasks)
    logger.info(
        "Applying audio-level noise augmentation to %d of %d training clips (%.1f%%, seed=%d, SNR=%s)...",
        n_augmented,
        n_samples,
        (n_augmented / n_samples) * 100.0,
        seed,
        snr_range,
    )

    def _augment_and_extract(task: tuple[int, Any, float, bool, int]) -> tuple[int, np.ndarray]:
        idx, row, snr, use_urban, bg_pick = task
        rng = np.random.default_rng([int(seed), int(idx)])
        wav_path = audio_dir / f"fold{int(row.fold)}" / str(row.slice_file_name)

        # 1. Load canonical waveform
        waveform = load_and_standardize_audio(
            wav_path, target_sr=22050, target_duration_sec=4.0
        )

        # 2. Add controlled background noise
        if use_urban:
            bg_sample = non_siren_tuples[bg_pick]
            bg_path = audio_dir / f"fold{int(bg_sample.fold)}" / str(bg_sample.slice_file_name)
            try:
                bg_wav = load_and_standardize_audio(
                    bg_path, target_sr=22050, target_duration_sec=4.0
                )
                waveform = add_controlled_background_noise(
                    waveform, snr_db=snr, rng=rng, noise_waveform=bg_wav
                )
            except Exception:
                waveform = add_controlled_background_noise(
                    waveform, snr_db=snr, rng=rng
                )
        else:
            waveform = add_controlled_background_noise(
                waveform, snr_db=snr, rng=rng
            )

        # 3. Recompute 80 MFCC summary statistics from the augmented waveform
        mfcc_feat = extract_mfcc_features(
            waveform,
            sample_rate=22050,
            n_mfcc=40,
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            n_mels=128,
            f_min=0.0,
            f_max=11025.0,
        )
        return idx, mfcc_feat

    t0 = time.time()
    # Process in chunks with 6 worker threads to maintain strict low memory footprint
    chunk_size = 200
    completed = 0
    print(f"Starting parallel audio augmentation ({len(aug_tasks)} clips)...", flush=True)
    for chunk_start in range(0, len(aug_tasks), chunk_size):
        chunk = aug_tasks[chunk_start : chunk_start + chunk_size]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            chunk_results = list(executor.map(_augment_and_extract, chunk))

        # Standardize and assign into training array
        for idx, raw_mfcc in chunk_results:
            norm_mfcc = scaler.transform(raw_mfcc.reshape(1, -1)).astype(np.float32)
            X_train_aug[idx] = norm_mfcc[0]

        completed += len(chunk)
        gc.collect()
        print(f"  [Progress] {completed}/{len(aug_tasks)} augmented clips processed ({time.time() - t0:.1f} s elapsed)...", flush=True)

    # Save to disk
    np.savez_compressed(
        cache_path,
        X_train=X_train_aug,
        y_train=y_train,
        seed=seed,
        p_augment=p_augment,
        n_augmented=n_augmented,
        snr_min=snr_range[0],
        snr_max=snr_range[1],
        elapsed_sec=round(time.time() - t0, 2),
    )
    logger.info(
        "Successfully extracted and saved augmented MFCCs to '%s' in %.2f s.",
        cache_path,
        time.time() - t0,
    )
    return cache_path


def train_noise_augmented_deep_mlp(
    dataset_root: str | Path,
    splits_dir: str | Path,
    processed_dir: str | Path,
    checkpoints_dir: str | Path,
    results_dir: str | Path,
    seed: int = 42,
    p_augment: float = 0.5,
    snr_range: tuple[float, float] = (10.0, 25.0),
    max_epochs: int = 50,
    patience: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    preferred_device: str = "auto",
) -> dict[str, Any]:
    """Train DeepMLP on noise-augmented training split with clean validation early stopping.

    Returns:
        Dictionary containing training history, validation metrics, and checkpoint metadata.
    """
    set_random_seeds(seed)
    proc_dir = Path(processed_dir)
    ckpt_dir = Path(checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir = Path(results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    # 1. Acquire augmented training data
    aug_cache = generate_augmented_training_mfcc(
        dataset_root=dataset_root,
        splits_dir=splits_dir,
        processed_dir=processed_dir,
        seed=seed,
        p_augment=p_augment,
        snr_range=snr_range,
    )
    aug_data = np.load(aug_cache)
    X_train = aug_data["X_train"].astype(np.float32)
    y_train = aug_data["y_train"].astype(np.int64)

    # 2. Acquire clean validation data (strictly from original cache)
    clean_cache = proc_dir / "train_val_mfcc_cache.npz"
    if not clean_cache.is_file():
        raise FileNotFoundError(f"Clean train_val cache not found at '{clean_cache}'.")
    val_data = np.load(clean_cache)
    X_val = val_data["X_val"].astype(np.float32)
    y_val = val_data["y_val"].astype(np.int64)

    # 3. Build model and DataLoaders
    train_loader, val_loader = make_tensor_dataloaders(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=batch_size,
        seed=seed,
    )

    model = DeepMLP(
        input_dim=80,
        hidden_dims=(128, 64),
        output_dim=1,
        dropout=0.3,
        seed=seed,
    )

    checkpoint_path = ckpt_dir / f"deep_mlp_augmented_seed{seed}.pt"

    # 4. Train model using identical optimization protocol
    from src.train import train_pytorch_model
    train_out = train_pytorch_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        preferred_device=preferred_device,
        use_train_pos_weight=True,
        checkpoint_path=checkpoint_path,
        extra_metadata={
            "training_type": "noise_augmented",
            "p_augment": p_augment,
            "snr_range_db": list(snr_range),
            "noise_types": ["white_gaussian", "urban_environmental_mix"],
            "base_clean_checkpoint": f"deep_mlp_t3_seed{seed}.pt",
        },
    )

    # Save summary metadata
    meta_path = res_dir / "metrics" / f"augmented_deep_mlp_metadata_seed{seed}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    t_meta = train_out.get("training_metadata", {})
    meta_payload = {
        "model_id": "deep_mlp_augmented",
        "seed": seed,
        "checkpoint_path": str(checkpoint_path),
        "val_macro_f1": float(t_meta.get("best_val_macro_f1", 0.0)),
        "best_epoch": int(t_meta.get("best_epoch", 0)),
        "actual_epochs_completed": int(t_meta.get("actual_epochs_completed", 0)),
        "training_duration_sec": float(t_meta.get("training_duration_sec", 0.0)),
        "p_augment": p_augment,
        "snr_range_db": list(snr_range),
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    logger.info(
        "Augmented Deep MLP Seed %d finished: best epoch %d, val macro-F1: %.4f, time: %.2f s.",
        seed,
        t_meta.get("best_epoch", 0),
        t_meta.get("best_val_macro_f1", 0.0),
        t_meta.get("training_duration_sec", 0.0),
    )
    return train_out


def evaluate_clean_vs_augmented(
    dataset_root: str | Path,
    splits_dir: str | Path,
    processed_dir: str | Path,
    clean_checkpoint_path: str | Path,
    augmented_checkpoint_path: str | Path,
    results_dir: str | Path,
    seed: int = 42,
    device: str = "cpu",
) -> pd.DataFrame:
    """Evaluate Clean vs. Noise-Augmented Deep MLP on Held-Out Fold 10 across 4 SNR conditions.

    Conditions tested:
    1. Clean (unperturbed, canonical Fold 10)
    2. 20 dB SNR (moderate street ambient)
    3. 10 dB SNR (heavy urban noise)
    4. 5 dB SNR (extreme acoustic interference)

    Returns:
        DataFrame comparing both models across all conditions on held-out test data.
    """
    dev = torch.device(device)
    paths_root = Path(dataset_root)
    audio_dir = paths_root / "audio" if (paths_root / "audio").is_dir() else paths_root

    test_csv = Path(splits_dir) / "test.csv"
    if not test_csv.is_file():
        raise FileNotFoundError(f"Held-out test manifest not found at '{test_csv}'.")

    test_df = pd.read_csv(test_csv)
    y_test = test_df["binary_label"].to_numpy(dtype=np.int64)
    n_test = len(test_df)

    # Load frozen scaler
    scaler = MFCCStandardScaler.load(Path(processed_dir) / "mfcc_scaler.json")

    # Load models
    clean_model, clean_meta = load_pytorch_checkpoint(clean_checkpoint_path, device=dev)
    aug_model, aug_meta = load_pytorch_checkpoint(augmented_checkpoint_path, device=dev)
    clean_model.eval()
    aug_model.eval()

    # Load clean test cache if available
    clean_test_cache_path = Path(processed_dir) / "test_mfcc_cache.npz"
    clean_test_mfcc = (
        np.load(clean_test_cache_path)["X_test"].astype(np.float32)
        if clean_test_cache_path.is_file()
        else None
    )

    conditions = [
        ("Clean", None),
        ("20 dB", 20.0),
        ("10 dB", 10.0),
        ("5 dB", 5.0),
    ]

    records: list[dict[str, Any]] = []

    for cond_label, snr_db in conditions:
        print(f"Evaluating Held-Out Test Fold 10 under condition: {cond_label}...", flush=True)
        if snr_db is None and clean_test_mfcc is not None:
            X_test_cond = clean_test_mfcc
        else:
            def _extract_noisy_test(item: tuple[int, Any]) -> tuple[int, np.ndarray]:
                i, row = item
                rng = np.random.default_rng([int(seed), int(i), int(round(snr_db if snr_db is not None else 0))])
                wav_path = audio_dir / f"fold{int(row.fold)}" / str(row.slice_file_name)
                wav = load_and_standardize_audio(wav_path, target_sr=22050, target_duration_sec=4.0)
                if snr_db is not None:
                    wav = add_controlled_background_noise(wav, snr_db=snr_db, rng=rng)
                mfcc_raw = extract_mfcc_features(
                    wav,
                    sample_rate=22050,
                    n_mfcc=40,
                    n_fft=2048,
                    hop_length=512,
                    win_length=2048,
                    n_mels=128,
                    f_min=0.0,
                    f_max=11025.0,
                )
                return i, mfcc_raw

            items = list(enumerate(test_df.itertuples(index=False)))
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                res_items = list(executor.map(_extract_noisy_test, items))
            res_items.sort(key=lambda x: x[0])
            raw_cond_mfcc = np.stack([r[1] for r in res_items], axis=0)
            X_test_cond = scaler.transform(raw_cond_mfcc).astype(np.float32)

        X_tensor = torch.from_numpy(X_test_cond).float().to(dev)

        # Run inference for Clean Deep MLP
        t_start = time.perf_counter()
        with torch.no_grad():
            logits_clean = clean_model(X_tensor).cpu().numpy().reshape(-1)
            probs_clean = 1.0 / (1.0 + np.exp(-logits_clean))
        t_clean_ms = ((time.perf_counter() - t_start) / n_test) * 1000.0

        metrics_clean = compute_classification_metrics(
            y_true=y_test,
            y_prob=probs_clean,
            split_name="test",
            allow_final_test=True,
        )

        # Run inference for Noise-Augmented Deep MLP
        t_start = time.perf_counter()
        with torch.no_grad():
            logits_aug = aug_model(X_tensor).cpu().numpy().reshape(-1)
            probs_aug = 1.0 / (1.0 + np.exp(-logits_aug))
        t_aug_ms = ((time.perf_counter() - t_start) / n_test) * 1000.0

        metrics_aug = compute_classification_metrics(
            y_true=y_test,
            y_prob=probs_aug,
            split_name="test",
            allow_final_test=True,
        )

        records.append({
            "condition": cond_label,
            "snr_db": "Clean" if snr_db is None else f"{snr_db:.0f} dB",
            "model": "Clean Deep MLP",
            "macro_f1": round(float(metrics_clean["macro_f1"]), 4),
            "siren_f1": round(float(metrics_clean["f1_siren"]), 4),
            "siren_recall": round(float(metrics_clean["recall_siren"]), 4),
            "siren_precision": round(float(metrics_clean["precision_siren"]), 4),
            "pr_auc": round(float(metrics_clean["pr_auc"]), 4),
            "roc_auc": round(float(metrics_clean["roc_auc"]), 4),
            "balanced_accuracy": round(float(metrics_clean["balanced_accuracy"]), 4),
            "inference_latency_ms": round(t_clean_ms, 3),
            "tp": int(metrics_clean["confusion_matrix"]["tp"]),
            "fp": int(metrics_clean["confusion_matrix"]["fp"]),
            "tn": int(metrics_clean["confusion_matrix"]["tn"]),
            "fn": int(metrics_clean["confusion_matrix"]["fn"]),
        })

        records.append({
            "condition": cond_label,
            "snr_db": "Clean" if snr_db is None else f"{snr_db:.0f} dB",
            "model": "Noise-Augmented Deep MLP",
            "macro_f1": round(float(metrics_aug["macro_f1"]), 4),
            "siren_f1": round(float(metrics_aug["f1_siren"]), 4),
            "siren_recall": round(float(metrics_aug["recall_siren"]), 4),
            "siren_precision": round(float(metrics_aug["precision_siren"]), 4),
            "pr_auc": round(float(metrics_aug["pr_auc"]), 4),
            "roc_auc": round(float(metrics_aug["roc_auc"]), 4),
            "balanced_accuracy": round(float(metrics_aug["balanced_accuracy"]), 4),
            "inference_latency_ms": round(t_aug_ms, 3),
            "tp": int(metrics_aug["confusion_matrix"]["tp"]),
            "fp": int(metrics_aug["confusion_matrix"]["fp"]),
            "tn": int(metrics_aug["confusion_matrix"]["tn"]),
            "fn": int(metrics_aug["confusion_matrix"]["fn"]),
        })

    df = pd.DataFrame(records)

    # Save artifacts
    res_path = Path(results_dir)
    csv_out = res_path / "metrics" / "phase2_clean_vs_augmented_comparison.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)

    json_out = res_path / "metrics" / "phase2_comparison_summary.json"
    json_out.write_text(json.dumps(records, indent=2), encoding="utf-8")

    logger.info("Saved comparison metrics to '%s'.", csv_out)
    return df
