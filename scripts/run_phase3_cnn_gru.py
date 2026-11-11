"""Phase 3 Experiment Runner — CNN + GRU Hybrid Training & Evaluation.

Module: NB6057CEM — Artificial Neural Networks
Portfolio Upgrade: Phase 3 — Temporal Deep Learning (CNN + GRU Hybrid)

Usage:
    python scripts/run_phase3_cnn_gru.py [--seeds 42] [--batch-size 64] [--epochs 20] [--patience 6]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

# Persistent Matplotlib cache
_mpl_dir = os.path.join(tempfile.gettempdir(), "matplotlib_cache")
os.makedirs(_mpl_dir, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_dir)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import load_config, resolve_project_paths, setup_logger
from src.train_hybrid import (
    build_phase3_comparison_table,
    evaluate_cnn_gru_model,
    train_cnn_gru_model,
)

logger = logging.getLogger("phase3_runner")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3: CNN + GRU Hybrid Training & Evaluation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Random seeds to evaluate (e.g. 42 123 2026)")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience (epochs)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    cfg = load_config()
    paths = resolve_project_paths(cfg)

    logger.info("==========================================================================")
    logger.info("Starting Phase 3: CNN + GRU Hybrid (CRNN) Training & Evaluation")
    logger.info("==========================================================================")

    trained_checkpoints = {}
    train_summaries = {}
    test_results = {}

    for seed in args.seeds:
        logger.info("\n--- Training CNN + GRU Hybrid (Seed %d) ---", seed)
        t_start = time.time()
        train_res = train_cnn_gru_model(
            processed_dir=paths["processed_data_dir"],
            checkpoints_dir=paths["checkpoints_dir"],
            figures_dir=paths["figures_dir"],
            metrics_dir=paths["metrics_dir"],
            seed=seed,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=1e-4,
            max_epochs=args.epochs,
            patience=args.patience,
        )
        duration = time.time() - t_start
        ckpt_path = paths["checkpoints_dir"] / f"cnn_gru_seed{seed}.pt"
        trained_checkpoints[seed] = ckpt_path
        train_summaries[seed] = train_res
        logger.info("Seed %d training completed in %.1f s. Best Val Macro-F1: %.4f (Epoch %d)",
                    seed, duration, train_res["best_val_macro_f1"], train_res["best_epoch"])

    # Evaluate primary seed (args.seeds[0]) on held-out Fold 10
    primary_seed = args.seeds[0]
    primary_ckpt = trained_checkpoints[primary_seed]

    logger.info("\n==========================================================================")
    logger.info("Evaluating CNN + GRU Hybrid on Held-Out Fold 10 (Seed %d, N=837)", primary_seed)
    logger.info("==========================================================================")

    test_metrics = evaluate_cnn_gru_model(
        checkpoint_path=primary_ckpt,
        processed_dir=paths["processed_data_dir"],
        metrics_dir=paths["metrics_dir"],
        seed=primary_seed,
        threshold=0.50,
    )
    test_metrics["val_macro_f1"] = train_summaries[primary_seed]["best_val_macro_f1"]

    # Build architectural comparison table
    logger.info("\nGenerating architectural comparison table and figures...")
    comparison_df = build_phase3_comparison_table(
        metrics_dir=paths["metrics_dir"],
        figures_dir=paths["figures_dir"],
        cnn_gru_test_metrics=test_metrics,
    )

    print("\n" + "=" * 105)
    print("PHASE 3 ARCHITECTURAL BENCHMARK: DEEP MLP vs. 2D CNN vs. CNN + GRU HYBRID (FOLD 10)")
    print("=" * 105)
    print(comparison_df.to_markdown(index=False))
    print("=" * 105)
    print(f"Confusion Matrix (CNN + GRU): TP={test_metrics['confusion_matrix']['tp']}, "
          f"FP={test_metrics['confusion_matrix']['fp']}, "
          f"TN={test_metrics['confusion_matrix']['tn']}, "
          f"FN={test_metrics['confusion_matrix']['fn']}")
    print(f"Inference Latency: {test_metrics['inference_latency_ms']:.3f} ms/sample on CPU")
    print("=" * 105)

    return 0


if __name__ == "__main__":
    setup_logger("phase3_runner")
    setup_logger("phase3_train_hybrid")
    sys.exit(main())
