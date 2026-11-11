"""Phase 3 Training and Evaluation Pipeline for CNN + GRU Hybrid Audio Classifier.

Module: NB6057CEM — Artificial Neural Networks
Portfolio Upgrade: Phase 3 — Temporal Deep Learning (CNN + GRU Hybrid)

Features:
- Deterministic training on official UrbanSound8K Folds 1–8 (Log-Mel representations).
- Validation-based early stopping strictly on clean Fold 9 (tracking Macro-F1).
- Held-out test evaluation strictly once on Fold 10 (N=837).
- Multi-model benchmarking against Deep MLP, 2D CNN, and Noise-Augmented Deep MLP.
- Diagnostic learning curves and comparative performance figures.
"""

from __future__ import annotations

import copy
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import (
    load_config,
    resolve_project_paths,
    select_device,
    set_random_seeds,
)
from src.evaluate import (
    compute_classification_metrics,
    enforce_no_test_split_access,
)
from src.hybrid_model import CNNGRUHybrid
from src.train import (
    EarlyStoppingTracker,
    compute_training_pos_weight,
    evaluate_pytorch_model_on_loader,
    load_pytorch_checkpoint,
    make_tensor_dataloaders,
)

logger = logging.getLogger("phase3_train_hybrid")


def plot_hybrid_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    val_macro_f1s: list[float],
    best_epoch: int,
    output_path: Path,
) -> Path:
    """Generate dual-panel training loss and validation Macro-F1 curve plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=150)
    epochs = list(range(1, len(train_losses) + 1))

    # Panel 1: Loss curves
    ax1.plot(epochs, train_losses, label="Train Loss (Weighted BCE)", color="#2563EB", linewidth=1.8)
    ax1.plot(epochs, val_losses, label="Val Loss (Clean Fold 9)", color="#DC2626", linewidth=1.8, linestyle="--")
    ax1.axvline(best_epoch, color="#10B981", linestyle=":", label=f"Best Epoch ({best_epoch})", linewidth=1.5)
    ax1.set_title("CNN + GRU Hybrid — Loss Progression", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=9.5)
    ax1.set_ylabel("Loss", fontsize=9.5)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8.5, loc="upper right")

    # Panel 2: Validation Macro-F1
    ax2.plot(epochs, val_macro_f1s, label="Val Macro-F1", color="#7C3AED", linewidth=2.0)
    ax2.axvline(best_epoch, color="#10B981", linestyle=":", label=f"Best Epoch ({best_epoch})", linewidth=1.5)
    ax2.set_title("Validation Macro-F1 Progression (Fold 9)", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=9.5)
    ax2.set_ylabel("Macro-F1 Score", fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8.5, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


import zipfile


class MMapMelDataset(torch.utils.data.Dataset):
    """Memory-mapped PyTorch Dataset for Log-Mel Spectrogram arrays."""

    def __init__(self, x_data: Any, y_data: Any) -> None:
        self.x = x_data
        self.y = np.asarray(y_data, dtype=np.int64).reshape(-1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = np.array(self.x[idx], dtype=np.float32, copy=True)
        return torch.from_numpy(sample), torch.tensor(self.y[idx], dtype=torch.int64)


def ensure_mmap_mel_files(processed_dir: Path) -> tuple[Path, Path, Path]:
    """Ensure raw uncompressed .npy files exist for OS-level zero-copy memory mapping."""
    train_mmap = processed_dir / "mmap_mel_train.npy"
    val_mmap = processed_dir / "mmap_mel_val.npy"
    test_mmap = processed_dir / "mmap_mel_test.npy"

    cache_path = processed_dir / "train_val_mel_cache.npz"
    if not train_mmap.is_file() or not val_mmap.is_file():
        logger.info("Extracting uncompressed mmap buffers from '%s'...", cache_path)
        with zipfile.ZipFile(cache_path) as z:
            if not train_mmap.is_file():
                with z.open("X_train.npy") as src, open(train_mmap, "wb") as dst:
                    while chunk := src.read(8 * 1024 * 1024):
                        dst.write(chunk)
            if not val_mmap.is_file():
                with z.open("X_val.npy") as src, open(val_mmap, "wb") as dst:
                    while chunk := src.read(8 * 1024 * 1024):
                        dst.write(chunk)

    if not test_mmap.is_file():
        test_cache = processed_dir / "test_mel_cache.npz"
        logger.info("Extracting uncompressed test mmap buffer from '%s'...", test_cache)
        with zipfile.ZipFile(test_cache) as z:
            with z.open("X_test.npy") as src, open(test_mmap, "wb") as dst:
                while chunk := src.read(8 * 1024 * 1024):
                    dst.write(chunk)

    return train_mmap, val_mmap, test_mmap


def train_cnn_gru_model(
    processed_dir: Path,
    checkpoints_dir: Path,
    figures_dir: Path,
    metrics_dir: Path,
    seed: int = 42,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 25,
    patience: int = 6,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train the CNN + GRU Hybrid architecture on pre-extracted Log-Mel representations.

    Uses strictly training Folds 1–8 and validation Fold 9 for early stopping.
    """
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    cache_path = processed_dir / "train_val_mel_cache.npz"
    if not cache_path.is_file():
        raise FileNotFoundError(f"Normalized Log-Mel cache not found at '{cache_path}'.")

    train_mmap, val_mmap, _ = ensure_mmap_mel_files(processed_dir)
    logger.info("Opening OS memory-mapped Log-Mel arrays from '%s' and '%s'...", train_mmap.name, val_mmap.name)
    x_train_arr = np.load(train_mmap, mmap_mode="r")
    x_val_arr = np.load(val_mmap, mmap_mode="r")

    with zipfile.ZipFile(cache_path) as z:
        with z.open("y_train.npy") as f:
            y_train = np.load(f)
        with z.open("y_val.npy") as f:
            y_val = np.load(f)

    set_random_seeds(seed)
    if device is None:
        device = select_device().device
    logger.info("Using compute device: %s", device)

    # 1. Training-only positive class weight calculation
    pos_weight_val = compute_training_pos_weight(y_train, split_name="train")
    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    logger.info("Training positive class loss weight: %.4f (pos_weight)", pos_weight_val)

    # 2. DataLoaders using MMapMelDataset
    gen = torch.Generator()
    gen.manual_seed(seed)
    train_dataset = MMapMelDataset(x_train_arr, y_train)
    val_dataset = MMapMelDataset(x_val_arr, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=gen,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # 3. Model Architecture
    model = CNNGRUHybrid(
        in_channels=1,
        conv_channels=(16, 32, 64),
        gru_hidden_size=64,
        gru_num_layers=1,
        bidirectional=True,
        n_mels=128,
        time_frames=173,
        dropout=0.3,
        output_dim=1,
        seed=seed,
    )
    model.to(device)
    param_count = model.count_trainable_parameters()
    logger.info("Instantiated CNNGRUHybrid (Seed %d): %d trainable parameters", seed, param_count)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    tracker = EarlyStoppingTracker(patience=patience, min_delta=0.0001)

    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    val_macro_f1_history: list[float] = []

    logger.info("Starting CNN + GRU training: max_epochs=%d, patience=%d, batch_size=%d", max_epochs, patience, batch_size)
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        n_samples = 0

        rng = np.random.default_rng(seed + epoch)
        perm = rng.permutation(len(y_train))

        for start_idx in range(0, len(y_train), batch_size):
            end_idx = min(start_idx + batch_size, len(y_train))
            b_idx = perm[start_idx:end_idx]
            x_b = torch.from_numpy(np.array(x_train_arr[b_idx], dtype=np.float32, copy=True)).to(device)
            y_b = torch.tensor(y_train[b_idx], dtype=torch.float32, device=device).reshape(-1, 1)

            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            bs = int(x_b.shape[0])
            running_loss += float(loss.item()) * bs
            n_samples += bs

        epoch_train_loss = running_loss / max(1, n_samples)
        train_loss_history.append(epoch_train_loss)

        # Validation step on clean Fold 9 (vectorized batched inference)
        model.eval()
        v_running_loss = 0.0
        v_samples = 0
        all_probs: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []

        with torch.no_grad():
            for v_start in range(0, len(y_val), batch_size):
                v_end = min(v_start + batch_size, len(y_val))
                x_va_b = torch.from_numpy(np.array(x_val_arr[v_start:v_end], dtype=np.float32, copy=True)).to(device)
                y_va_b = torch.tensor(y_val[v_start:v_end], dtype=torch.float32, device=device).reshape(-1, 1)
                va_logits = model(x_va_b)
                va_loss = criterion(va_logits, y_va_b)
                probs = torch.sigmoid(va_logits)

                bs_v = int(x_va_b.shape[0])
                v_running_loss += float(va_loss.item()) * bs_v
                v_samples += bs_v
                all_probs.append(probs.cpu().numpy().reshape(-1))
                all_targets.append(y_va_b.cpu().numpy().reshape(-1))

        val_loss = v_running_loss / max(1, v_samples)
        y_true_va = np.concatenate(all_targets, axis=0).astype(np.int64)
        y_prob_va = np.concatenate(all_probs, axis=0).astype(np.float64)
        val_metrics = compute_classification_metrics(
            y_true=y_true_va,
            y_prob=y_prob_va,
            split_name="val",
            allow_final_test=False,
        )

        val_macro_f1 = float(val_metrics["macro_f1"])
        val_loss_history.append(val_loss)
        val_macro_f1_history.append(val_macro_f1)

        epoch_duration = time.time() - epoch_start
        logger.info(
            "Epoch %02d/%02d [%.1fs] — Train Loss: %.4f | Val Loss: %.4f | Val Macro-F1: %.4f (Siren Recall: %.2f%%)",
            epoch,
            max_epochs,
            epoch_duration,
            epoch_train_loss,
            val_loss,
            val_macro_f1,
            val_metrics.get("recall_siren", 0.0) * 100,
        )

        improved = tracker.update(
            epoch=epoch,
            val_loss=val_loss,
            val_metrics=val_metrics,
            model=model,
        )
        if improved:
            logger.info("  -> New best validation Macro-F1: %.4f at Epoch %d (saving checkpoint...)", val_macro_f1, epoch)
            # Checkpoint immediately to preserve best weights
            ckpt_path = checkpoints_dir / f"cnn_gru_seed{seed}.pt"
            ckpt_payload = {
                "model_id": "cnn_gru",
                "framework": "pytorch",
                "architecture_config": model.architecture_summary(),
                "trainable_parameters": param_count,
                "state_dict": tracker.best_state_dict,
                "training_metadata": {
                    "model_name": "CNN + GRU Hybrid (CRNN)",
                    "seed": seed,
                    "best_epoch": tracker.best_epoch,
                    "best_val_macro_f1": round(tracker.best_val_macro_f1, 4),
                    "total_epochs_trained": len(train_loss_history),
                    "total_training_duration_sec": round(time.time() - start_time, 2),
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "pos_weight": round(pos_weight_val, 4),
                    "train_loss_history": [round(x, 4) for x in train_loss_history],
                    "val_loss_history": [round(x, 4) for x in val_loss_history],
                    "val_macro_f1_history": [round(x, 4) for x in val_macro_f1_history],
                },
            }
            torch.save(ckpt_payload, ckpt_path)

        gc.collect()

        if tracker.should_stop:
            logger.info("Early stopping triggered at Epoch %d (Patience=%d exhausted)", epoch, patience)
            break

    total_training_duration = time.time() - start_time
    best_state = tracker.best_state_dict or copy.deepcopy(model.state_dict())
    best_epoch = tracker.best_epoch
    best_val_macro_f1 = tracker.best_val_macro_f1

    logger.info(
        "Training completed in %.1fs. Best Epoch: %d (Val Macro-F1: %.4f)",
        total_training_duration,
        best_epoch,
        best_val_macro_f1,
    )

    # 4. Save Checkpoint
    checkpoint_path = checkpoints_dir / f"cnn_gru_seed{seed}.pt"
    checkpoint_payload = {
        "model_id": "cnn_gru",
        "framework": "pytorch",
        "architecture_config": model.architecture_summary(),
        "trainable_parameters": param_count,
        "state_dict": best_state,
        "training_metadata": {
            "model_name": "CNN + GRU Hybrid (CRNN)",
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_macro_f1": round(best_val_macro_f1, 4),
            "total_epochs_trained": len(train_loss_history),
            "total_training_duration_sec": round(total_training_duration, 2),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "pos_weight": round(pos_weight_val, 4),
            "train_loss_history": [round(x, 4) for x in train_loss_history],
            "val_loss_history": [round(x, 4) for x in val_loss_history],
            "val_macro_f1_history": [round(x, 4) for x in val_macro_f1_history],
        },
    }
    torch.save(checkpoint_payload, checkpoint_path)
    logger.info("Saved Phase 3 checkpoint to '%s'.", checkpoint_path)

    # 5. Save Learning Curves
    fig_path = figures_dir / f"fig_phase3_cnn_gru_curves_seed{seed}.png"
    plot_hybrid_training_curves(
        train_losses=train_loss_history,
        val_losses=val_loss_history,
        val_macro_f1s=val_macro_f1_history,
        best_epoch=best_epoch,
        output_path=fig_path,
    )
    logger.info("Saved learning curve figure to '%s'.", fig_path)

    # 6. Save Metadata JSON
    meta_path = metrics_dir / f"phase3_cnn_gru_train_seed{seed}.json"
    meta_path.write_text(json.dumps(checkpoint_payload["training_metadata"], indent=2), encoding="utf-8")

    return checkpoint_payload["training_metadata"]


def evaluate_cnn_gru_model(
    checkpoint_path: Path,
    processed_dir: Path,
    metrics_dir: Path,
    seed: int = 42,
    threshold: float = 0.50,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate trained CNN + GRU Hybrid model strictly on held-out test fold 10 (N=837)."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    test_cache_path = processed_dir / "test_mel_cache.npz"
    if not test_cache_path.is_file():
        raise FileNotFoundError(f"Test Mel cache not found at '{test_cache_path}'.")

    _, _, test_mmap = ensure_mmap_mel_files(processed_dir)
    logger.info("Opening OS memory-mapped test Mel array from '%s'...", test_mmap.name)
    x_test_arr = np.load(test_mmap, mmap_mode="r")
    with zipfile.ZipFile(test_cache_path) as z:
        with z.open("y_test.npy") as f:
            y_test = np.load(f)

    if device is None:
        device = select_device().device
    logger.info("Evaluating on compute device: %s", device)
    model, payload = load_pytorch_checkpoint(checkpoint_path, device=device)
    model.eval()

    test_dataset = MMapMelDataset(x_test_arr, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    n_samples = len(y_test)

    # Warmup and batched inference
    all_probs: list[np.ndarray] = []
    t0 = time.time()
    with torch.no_grad():
        for x_b, _ in test_loader:
            x_b = x_b.to(device)
            logits = model(x_b)
            probs_b = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            all_probs.append(probs_b)
    total_inf_time = time.time() - t0
    probs = np.concatenate(all_probs, axis=0)

    latency_per_sample_ms = (total_inf_time / max(1, n_samples)) * 1000.0

    # Binary predictions at operating threshold
    preds = (probs >= threshold).astype(np.int64)
    tp = int(np.sum((preds == 1) & (y_test == 1)))
    fp = int(np.sum((preds == 1) & (y_test == 0)))
    tn = int(np.sum((preds == 0) & (y_test == 0)))
    fn = int(np.sum((preds == 0) & (y_test == 1)))

    metrics = compute_classification_metrics(
        y_true=y_test,
        y_prob=probs,
        split_name="test",
        allow_final_test=True,
    )

    results = {
        "model_id": "cnn_gru",
        "model_name": "CNN + GRU Hybrid (Spectro-Temporal)",
        "seed": seed,
        "input_representation": "Log-Mel Spectrogram (1, 128, 173)",
        "trainable_parameters": model.count_trainable_parameters(),
        "macro_f1": round(float(metrics["macro_f1"]), 4),
        "siren_f1": round(float(metrics["f1_siren"]), 4),
        "siren_recall": round(float(metrics["recall_siren"]), 4),
        "siren_precision": round(float(metrics["precision_siren"]), 4),
        "pr_auc": round(float(metrics["pr_auc"]), 4),
        "roc_auc": round(float(metrics["roc_auc"]), 4),
        "balanced_accuracy": round(float(metrics["balanced_accuracy"]), 4),
        "inference_latency_ms": round(latency_per_sample_ms, 3),
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
    }

    out_json = metrics_dir / f"phase3_cnn_gru_test_seed{seed}.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Saved Phase 3 test evaluation metrics to '%s'.", out_json)
    return results


def build_phase3_comparison_table(
    metrics_dir: Path,
    figures_dir: Path,
    cnn_gru_test_metrics: dict[str, Any],
) -> pd.DataFrame:
    """Generate side-by-side comparison table and degradation comparison figure across all architectures."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Base benchmarks from existing evaluated models (CW1 T4 verified metrics)
    rows = [
        {
            "model_id": "deep_mlp",
            "model_name": "Deep MLP (Baseline)",
            "representation": "80-D MFCC (Summary)",
            "trainable_parameters": 18689,
            "val_macro_f1": 0.9655,
            "test_macro_f1": 0.7736,
            "siren_recall": 0.5542,
            "siren_precision": 0.6350,
            "siren_f1": 0.5896,
            "pr_auc": 0.6602,
            "inference_latency_ms": 0.010,
        },
        {
            "model_id": "cnn_2d",
            "model_name": "2D CNN (Log-Mel Spatial)",
            "representation": "(1, 128, 173) Log-Mel",
            "trainable_parameters": 23585,
            "val_macro_f1": 0.9151,
            "test_macro_f1": 0.7362,
            "siren_recall": 0.4458,
            "siren_precision": 0.6167,
            "siren_f1": 0.5175,
            "pr_auc": 0.6745,
            "inference_latency_ms": 0.710,
        },
    ]

    # Check if Phase 2 Augmented MLP summary is available
    phase2_json = metrics_dir / "phase2_comparison_summary.json"
    if phase2_json.is_file():
        try:
            p2_data = json.loads(phase2_json.read_text(encoding="utf-8"))
            for entry in p2_data:
                if entry.get("condition") == "Clean" and entry.get("model") == "Noise-Augmented Deep MLP":
                    rows.append({
                        "model_id": "deep_mlp_augmented",
                        "model_name": "Deep MLP (Noise-Augmented)",
                        "representation": "80-D MFCC (Summary)",
                        "trainable_parameters": 18689,
                        "val_macro_f1": 0.9620,
                        "test_macro_f1": round(float(entry["macro_f1"]), 4),
                        "siren_recall": round(float(entry["siren_recall"]), 4),
                        "siren_precision": round(float(entry["siren_precision"]), 4),
                        "siren_f1": round(float(entry["siren_f1"]), 4),
                        "pr_auc": round(float(entry["pr_auc"]), 4),
                        "inference_latency_ms": round(float(entry.get("inference_latency_ms", 0.015)), 3),
                    })
                    break
        except Exception:
            pass

    # Add Phase 3 CNN + GRU Hybrid
    rows.append({
        "model_id": "cnn_gru",
        "model_name": "CNN + GRU Hybrid (Phase 3)",
        "representation": "(1, 128, 173) Log-Mel",
        "trainable_parameters": cnn_gru_test_metrics["trainable_parameters"],
        "val_macro_f1": round(float(cnn_gru_test_metrics.get("val_macro_f1", 0.0)), 4),
        "test_macro_f1": round(float(cnn_gru_test_metrics["macro_f1"]), 4),
        "siren_recall": round(float(cnn_gru_test_metrics["siren_recall"]), 4),
        "siren_precision": round(float(cnn_gru_test_metrics["siren_precision"]), 4),
        "siren_f1": round(float(cnn_gru_test_metrics["siren_f1"]), 4),
        "pr_auc": round(float(cnn_gru_test_metrics["pr_auc"]), 4),
        "inference_latency_ms": round(float(cnn_gru_test_metrics["inference_latency_ms"]), 3),
    })

    df = pd.DataFrame(rows)
    csv_path = metrics_dir / "phase3_model_comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved Phase 3 architectural comparison table to '%s'.", csv_path)

    # Plot comparative bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=150)
    model_labels = [r["model_name"] for r in rows]
    x_pos = np.arange(len(model_labels))
    colors = ["#1f77b4", "#f59e0b", "#e11d48", "#8b5cf6"][:len(model_labels)]

    macro_f1s = [r["test_macro_f1"] for r in rows]
    recalls = [r["siren_recall"] * 100 for r in rows]

    bars1 = ax1.bar(x_pos, macro_f1s, color=colors, width=0.55, edgecolor="#000", linewidth=0.5)
    ax1.set_title("Test Macro-F1 across Architectures (Fold 10)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Macro-F1 Score", fontsize=9)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_labels, rotation=18, ha="right", fontsize=8.5)
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    for b in bars1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2.0, h + 0.02, f"{h:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    bars2 = ax2.bar(x_pos, recalls, color=colors, width=0.55, edgecolor="#000", linewidth=0.5)
    ax2.set_title("Siren Class Recall across Architectures (Fold 10)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Recall (%)", fontsize=9)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_labels, rotation=18, ha="right", fontsize=8.5)
    ax2.set_ylim(0.0, 100.0)
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    for b in bars2:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.tight_layout()
    plot_path = figures_dir / "phase3_architecture_comparison.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Phase 3 architectural comparison figure to '%s'.", plot_path)

    return df
