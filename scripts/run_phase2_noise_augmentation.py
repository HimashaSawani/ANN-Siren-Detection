"""Phase 2 Experiment Runner — Controlled Noise-Augmented Deep MLP Training & Evaluation.

Module: NB6057CEM — Artificial Neural Networks
Portfolio Upgrade: Phase 2 — Noise-Augmented Audio Training

Usage:
    python scripts/run_phase2_noise_augmentation.py [--seeds 42 123 2026]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

# Set persistent Matplotlib cache before importing matplotlib
_mpl_dir = os.path.join(tempfile.gettempdir(), "matplotlib_cache")
os.makedirs(_mpl_dir, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import load_config, resolve_project_paths, setup_logger
from src.noise_augmented_training import (
    evaluate_clean_vs_augmented,
    train_noise_augmented_deep_mlp,
)

logger = logging.getLogger("phase2_runner")


def plot_clean_vs_augmented_comparison(
    df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Generate side-by-side performance degradation curves (Macro-F1 & Siren Recall)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=150)

    snr_order = ["Clean", "20 dB", "10 dB", "5 dB"]
    models = ["Clean Deep MLP", "Noise-Augmented Deep MLP"]
    colors = {"Clean Deep MLP": "#1f77b4", "Noise-Augmented Deep MLP": "#e11d48"}
    markers = {"Clean Deep MLP": "o", "Noise-Augmented Deep MLP": "s"}

    for m in models:
        sub = df[df["model"] == m].set_index("condition").reindex(snr_order).reset_index()
        ax1.plot(
            sub["condition"],
            sub["macro_f1"],
            label=m,
            color=colors[m],
            marker=markers[m],
            linewidth=2.0,
            markersize=7,
        )
        ax2.plot(
            sub["condition"],
            sub["siren_recall"],
            label=m,
            color=colors[m],
            marker=markers[m],
            linewidth=2.0,
            markersize=7,
        )

    ax1.set_title("Test Macro-F1 across Noise Conditions (Fold 10)", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Noise Condition (Signal-to-Noise Ratio)", fontsize=9.5)
    ax1.set_ylabel("Macro-F1 Score", fontsize=9.5)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9, loc="lower left")

    ax2.set_title("Siren Recall across Noise Conditions (Fold 10)", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Noise Condition (Signal-to-Noise Ratio)", fontsize=9.5)
    ax2.set_ylabel("Siren Class Recall", fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9, loc="lower left")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 Noise-Augmented Deep MLP Training & Evaluation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Random seeds to evaluate")
    parser.add_argument("--force-recompute", action="store_true", help="Force re-extraction of augmented features")
    args = parser.parse_args()

    cfg = load_config()
    paths = resolve_project_paths(cfg)

    logger.info("==========================================================================")
    logger.info("Starting Phase 2: Controlled Noise-Augmented Deep MLP Training")
    logger.info("==========================================================================")

    trained_checkpoints = {}
    for seed in args.seeds:
        logger.info("\n--- Training Noise-Augmented Deep MLP (Seed %d) ---", seed)
        t_start = time.time()
        train_res = train_noise_augmented_deep_mlp(
            dataset_root=paths["urbansound8k_root"],
            splits_dir=paths["splits_dir"],
            processed_dir=paths["processed_data_dir"],
            checkpoints_dir=paths["checkpoints_dir"],
            results_dir=paths["results_dir"],
            seed=seed,
            p_augment=0.5,
            snr_range=(10.0, 25.0),
            max_epochs=50,
            patience=10,
        )
        trained_checkpoints[seed] = paths["checkpoints_dir"] / f"deep_mlp_augmented_seed{seed}.pt"
        logger.info("Seed %d completed in %.2f s.", seed, time.time() - t_start)

    # Evaluate primary seed (42) against baseline clean checkpoint
    primary_seed = args.seeds[0]
    clean_ckpt = paths["checkpoints_dir"] / f"deep_mlp_t3_seed{primary_seed}.pt"
    aug_ckpt = trained_checkpoints[primary_seed]

    logger.info("\n==========================================================================")
    logger.info("Evaluating Clean vs. Noise-Augmented Deep MLP on Held-Out Fold 10 (Seed %d)", primary_seed)
    logger.info("==========================================================================")

    df_results = evaluate_clean_vs_augmented(
        dataset_root=paths["urbansound8k_root"],
        splits_dir=paths["splits_dir"],
        processed_dir=paths["processed_data_dir"],
        clean_checkpoint_path=clean_ckpt,
        augmented_checkpoint_path=aug_ckpt,
        results_dir=paths["results_dir"],
        seed=primary_seed,
    )

    # Plot comparison
    plot_path = paths["figures_dir"] / "phase2_noise_robustness_comparison.png"
    plot_clean_vs_augmented_comparison(df_results, plot_path)
    logger.info("Saved comparison figure to '%s'.", plot_path)

    # Print markdown table
    print("\n" + "=" * 95)
    print("PHASE 2 EMPIRICAL EVALUATION RESULTS: CLEAN VS. NOISE-AUGMENTED DEEP MLP (FOLD 10)")
    print("=" * 95)
    print(df_results[["condition", "model", "macro_f1", "siren_f1", "siren_recall", "siren_precision", "pr_auc", "inference_latency_ms"]].to_markdown(index=False))
    print("=" * 95)

    return 0


if __name__ == "__main__":
    setup_logger("phase2_runner")
    sys.exit(main())
