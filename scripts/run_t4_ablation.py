"""Controlled Ablation Study on Dropout Regularization — Step 5 (CW1 T4c).

Module: NB6057CEM — Artificial Neural Networks
Step: 5 — Complete CW1 T4: Controlled Experiments and Model Evaluation (14 marks)

Investigates:
Design Claim: Incorporating Dropout (p=0.3) in the fully-connected hidden layers of Deep MLP
prevents co-adaptation and mitigates overfitting on the dominant non-siren classes under
8.4:1 class imbalance, yielding superior validation macro-F1 and lower validation loss.

Single-Factor Ablation:
- Baseline: Deep MLP with Dropout (p = 0.3)
- Ablation: Deep MLP without Dropout (p = 0.0)
- Fixed protocol: Same 7,095 training samples (Folds 1-8), same 800 validation samples (Fold 9),
  same three seeds (42, 123, 2026), same Adam optimizer (lr=1e-3, weight_decay=1e-4),
  same batch size (32), same early stopping patience (10 epochs).
- Evaluated strictly on validation data ONLY (held-out test split is untouched).

Outputs:
- `results/metrics/t4_ablation_results.csv`
- `results/figures/t4_ablation_comparison.png`

Usage:
    python scripts/run_t4_ablation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import (
    load_config,
    load_model_config,
    resolve_project_paths,
    set_random_seeds,
    setup_logger,
)
from src.evaluate import compute_classification_metrics
from src.models import DeepMLP
from src.train import make_tensor_dataloaders, train_pytorch_model


def plot_ablation_comparison(
    df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Generate bar chart comparing validation macro-F1 and loss with vs. without dropout."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5), dpi=150)

    conditions = ["Baseline (Dropout p=0.3)", "Ablation (No Dropout p=0.0)"]
    f1_means = [
        float(df[df["condition"] == "with_dropout_p0.3"]["val_macro_f1"].mean()),
        float(df[df["condition"] == "without_dropout_p0.0"]["val_macro_f1"].mean()),
    ]
    f1_stds = [
        float(df[df["condition"] == "with_dropout_p0.3"]["val_macro_f1"].std(ddof=1)),
        float(df[df["condition"] == "without_dropout_p0.0"]["val_macro_f1"].std(ddof=1)),
    ]
    loss_means = [
        float(df[df["condition"] == "with_dropout_p0.3"]["val_loss"].mean()),
        float(df[df["condition"] == "without_dropout_p0.0"]["val_loss"].mean()),
    ]
    loss_stds = [
        float(df[df["condition"] == "with_dropout_p0.3"]["val_loss"].std(ddof=1)),
        float(df[df["condition"] == "without_dropout_p0.0"]["val_loss"].std(ddof=1)),
    ]

    x = np.arange(len(conditions))
    bars1 = ax1.bar(x, f1_means, yerr=f1_stds, capsize=6, color=["#1f77b4", "#ff7f0e"], edgecolor="black", alpha=0.85)
    ax1.set_ylabel("Validation Macro-F1", fontsize=11)
    ax1.set_title("Validation Macro-F1 (Higher is Better)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, fontsize=9.5)
    ax1.set_ylim(0.90, 1.0)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:.4f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    bars2 = ax2.bar(x, loss_means, yerr=loss_stds, capsize=6, color=["#1f77b4", "#ff7f0e"], edgecolor="black", alpha=0.85)
    ax2.set_ylabel("Validation BCE Loss", fontsize=11)
    ax2.set_title("Validation BCE Loss (Lower is Better)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions, fontsize=9.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    for b in bars2:
        h = b.get_height()
        ax2.annotate(f"{h:.4f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.suptitle("CW1 T4(c) — Deep MLP Dropout Regularization Ablation (3-Seed Validation Mean ± 1 SD)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def run_ablation_study(seeds: list[int] = [42, 123, 2026]) -> dict[str, Any]:
    """Execute single-factor dropout ablation across three random seeds on validation data."""
    logger = setup_logger("t4_ablation")
    cfg = load_config()
    model_cfg = load_model_config()
    paths = resolve_project_paths(cfg)
    metrics_dir = paths["metrics_dir"]
    figures_dir = paths["figures_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Starting CW1 T4(c) Controlled Dropout Ablation Study ===")

    mfcc_cache = np.load(paths["processed_data_dir"] / "train_val_mfcc_cache.npz")
    X_train, y_train = mfcc_cache["X_train"].astype(np.float32), mfcc_cache["y_train"].astype(np.int64)
    X_val, y_val = mfcc_cache["X_val"].astype(np.float32), mfcc_cache["y_val"].astype(np.int64)

    mlp_cfg = model_cfg["deep_mlp"]
    dropout_levels = [
        ("with_dropout_p0.3", 0.3),
        ("without_dropout_p0.0", 0.0),
    ]

    records: list[dict[str, Any]] = []

    for cond_name, p_val in dropout_levels:
        for seed in seeds:
            set_random_seeds(seed)
            t0 = time.time()
            tr_loader, va_loader = make_tensor_dataloaders(
                X_train, y_train, X_val, y_val,
                batch_size=int(mlp_cfg["training"]["batch_size"]), seed=seed,
            )
            model = DeepMLP(
                input_dim=80,
                hidden_dims=tuple(mlp_cfg["architecture"]["hidden_dims"]),
                output_dim=1,
                dropout=p_val,
                seed=seed,
            )
            res = train_pytorch_model(
                model=model,
                train_loader=tr_loader,
                val_loader=va_loader,
                learning_rate=float(mlp_cfg["training"]["learning_rate"]),
                weight_decay=float(mlp_cfg["training"]["weight_decay"]),
                max_epochs=int(mlp_cfg["training"]["max_epochs"]),
                patience=int(mlp_cfg["training"]["early_stopping_patience"]),
                seed=seed,
            )
            train_time = time.time() - t0
            meta = res["training_metadata"]

            model.eval()
            with torch.no_grad():
                val_probs = torch.sigmoid(model(torch.from_numpy(X_val))).squeeze(-1).numpy().astype(np.float64)
            val_metrics = compute_classification_metrics(
                y_true=y_val, y_prob=val_probs, split_name="val", allow_final_test=False
            )

            records.append({
                "condition": cond_name,
                "dropout_rate": p_val,
                "seed": seed,
                "best_epoch": meta["best_epoch"],
                "total_epochs": meta["actual_epochs_completed"],
                "early_stopped": meta["early_stopping_triggered"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_siren_f1": val_metrics["f1_siren"],
                "val_non_siren_f1": val_metrics["f1_non_siren"],
                "val_precision_siren": val_metrics["precision_siren"],
                "val_recall_siren": val_metrics["recall_siren"],
                "val_loss": meta["best_val_loss"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "training_time_sec": round(train_time, 2),
            })
            logger.info("[%s] Seed %d: Best Epoch=%d, Val Macro-F1=%.4f, Val Loss=%.4f (%.2fs)", cond_name, seed, meta["best_epoch"], val_metrics["macro_f1"], meta["best_val_loss"], train_time)

    df_ablation = pd.DataFrame(records)
    csv_path = metrics_dir / "t4_ablation_results.csv"
    df_ablation.to_csv(csv_path, index=False)
    logger.info("Saved ablation results table to '%s'.", csv_path)

    fig_path = plot_ablation_comparison(df_ablation, figures_dir / "t4_ablation_comparison.png")
    logger.info("Saved ablation comparison figure to '%s'.", fig_path)
    return {"records": records, "csv": str(csv_path), "figure": str(fig_path)}


if __name__ == "__main__":
    run_ablation_study()

