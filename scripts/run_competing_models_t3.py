"""Unified Competing Models Experiment Runner — Step 4 (CW1 T3).

Module: NB6057CEM — Artificial Neural Networks
Step: 4 — Implement and Train Three Competing Models (CW1 T3)

Supports:
1. Non-Neural Baseline: `RandomForestBaseline` (scikit-learn) on 80-D MFCCs.
2. Primary Neural Model 1: `DeepMLP` (PyTorch) on 80-D MFCCs.
3. Primary Neural Model 2: `Conv2DCNN` (PyTorch) on (1, 128, 173) Log-Mel spectrograms.

Enforces:
- Identical recording-level train (Folds 1-8) and validation (Fold 9) partitions.
- Held-out test partition (Fold 10) is NEVER loaded or evaluated in Step 4.
- Training-only preprocessing statistics and training-only class weights.
- Validation macro-F1 model selection and early stopping.

Usage:
    python scripts/run_competing_models_t3.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import (  # noqa: E402
    get_environment_info,
    load_config,
    load_data_config,
    load_model_config,
    resolve_project_paths,
    setup_logger,
)
from src.audio_features import (  # noqa: E402
    MFCCStandardScaler,
    MelSpectrogramNormalizer,
)
from src.dataset import create_dataloaders  # noqa: E402
from src.models import (  # noqa: E402
    Conv2DCNN,
    DeepMLP,
    RandomForestBaseline,
)
from src.preprocessing import validate_urbansound8k_dataset  # noqa: E402
from src.train import (  # noqa: E402
    make_tensor_dataloaders,
    plot_training_curves,
    train_pytorch_model,
    tune_and_select_random_forest,
)


def export_architecture_comparison_artifacts(metrics_dir: Path) -> dict[str, Any]:
    """Programmatically inspect all three competing models and export architecture tables."""
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rf = RandomForestBaseline(n_estimators=200, max_depth=16, random_state=42)
    mlp = DeepMLP(input_dim=80, hidden_dims=(128, 64), output_dim=1, dropout=0.3, seed=42)
    cnn = Conv2DCNN(
        in_channels=1,
        conv_channels=(16, 32, 64),
        n_mels=128,
        time_frames=173,
        dropout=0.3,
        output_dim=1,
        seed=42,
    )

    mlp_summary = mlp.architecture_summary()
    cnn_summary = cnn.architecture_summary()

    comparison_table = [
        {
            "model_id": "random_forest",
            "model_name": "Random Forest (Non-Neural Baseline)",
            "framework": "scikit-learn",
            "input_representation": "80-D MFCC Summary (40 mean + 40 std)",
            "input_tensor_shape": "(N, 80)",
            "layer_structure": "Ensemble of 100-300 Decision Trees (max_depth in {12, 16, 20, None})",
            "hidden_dimensions": "N/A (axis-aligned recursive splits)",
            "activations": "Non-parametric step splits + leaf probability averaging",
            "output_shape": "(N, 1)",
            "loss_or_criterion": "Gini Impurity (class_weight='balanced' from train split)",
            "optimizer": "Greedy CART bootstrap aggregation",
            "trainable_parameters": "Non-parametric tree nodes (data-dependent)",
            "regularization": "max_depth, min_samples_leaf, bootstrap subsampling",
        },
        {
            "model_id": "deep_mlp",
            "model_name": "Deep MLP (Tabular Neural Architecture)",
            "framework": "PyTorch",
            "input_representation": "80-D MFCC Summary (40 mean + 40 std)",
            "input_tensor_shape": "(B, 80)",
            "layer_structure": "Linear(80->128) -> ReLU -> Dropout(0.3) -> Linear(128->64) -> ReLU -> Dropout(0.3) -> Linear(64->1)",
            "hidden_dimensions": "[128, 64]",
            "activations": "ReLU (hidden), Linear Logit + Sigmoid (output)",
            "output_shape": "(B, 1)",
            "loss_or_criterion": "BCEWithLogitsLoss (pos_weight from train split only)",
            "optimizer": "Adam (lr=1e-3, weight_decay=1e-4)",
            "trainable_parameters": str(mlp_summary["trainable_parameters"]),
            "regularization": "Dropout(p=0.3), L2 weight_decay=1e-4, Validation Early Stopping (patience=10)",
        },
        {
            "model_id": "cnn_2d",
            "model_name": "2D CNN (Spectro-Temporal Neural Architecture)",
            "framework": "PyTorch",
            "input_representation": "2D Log-Mel Spectrogram",
            "input_tensor_shape": "(B, 1, 128, 173)",
            "layer_structure": "3x [Conv2d(3x3, pad=1) + BatchNorm2d + ReLU + MaxPool2d(2x2 on blocks 1-2)] -> AdaptiveAvgPool2d(1x1) -> Dropout(0.3) -> Linear(64->1)",
            "hidden_dimensions": "Channels [16, 32, 64], Spatial [(64, 86), (32, 43), (1, 1)]",
            "activations": "ReLU (conv blocks), Linear Logit + Sigmoid (output)",
            "output_shape": "(B, 1)",
            "loss_or_criterion": "BCEWithLogitsLoss (pos_weight from train split only)",
            "optimizer": "Adam (lr=1e-3, weight_decay=1e-4)",
            "trainable_parameters": str(cnn_summary["trainable_parameters"]),
            "regularization": "BatchNorm2d, Global Average Pooling, Dropout(p=0.3), L2 weight_decay=1e-4, Validation Early Stopping (patience=8)",
        },
    ]

    payload = {
        "comparison_table": comparison_table,
        "random_forest_default": rf.get_hyperparameters(),
        "deep_mlp_detail": mlp_summary,
        "cnn_2d_detail": cnn_summary,
    }

    json_path = metrics_dir / "t3_architecture_comparison.json"
    csv_path = metrics_dir / "t3_architecture_comparison_table.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(comparison_table).to_csv(csv_path, index=False)
    return payload


def _extract_numpy_from_loader(loader: Any) -> tuple[Any, Any]:
    import numpy as np

    xs, ys = [], []
    for xb, yb in loader:
        xs.append(xb.detach().cpu().numpy())
        ys.append(yb.detach().cpu().numpy())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def run_step4_pipeline(seed: int = 42) -> dict[str, Any]:
    """Execute the Step 4 (CW1 T3) competing models pipeline."""
    logger = setup_logger("competing_models_t3")
    cfg = load_config()
    data_cfg = load_data_config()
    model_cfg = load_model_config()
    paths = resolve_project_paths(cfg)
    env_info = get_environment_info(cfg)

    metrics_dir = paths["metrics_dir"]
    figures_dir = paths["figures_dir"]
    checkpoints_dir = paths["checkpoints_dir"]

    arch_payload = export_architecture_comparison_artifacts(metrics_dir)
    logger.info(
        "Exported architecture comparison tables: DeepMLP=%d params, Conv2DCNN=%d params.",
        arch_payload["deep_mlp_detail"]["trainable_parameters"],
        arch_payload["cnn_2d_detail"]["trainable_parameters"],
    )

    dataset_report = validate_urbansound8k_dataset(config=cfg, strict=False)
    train_manifest = paths["splits_dir"] / "train.csv"
    val_manifest = paths["splits_dir"] / "val.csv"
    test_manifest = paths["splits_dir"] / "test.csv"
    mfcc_scaler_path = paths["processed_data_dir"] / "mfcc_scaler.json"
    mel_norm_path = paths["processed_data_dir"] / "mel_norm_stats.json"

    real_data_ready = (
        dataset_report.is_valid
        and train_manifest.is_file()
        and val_manifest.is_file()
        and test_manifest.is_file()
        and mfcc_scaler_path.is_file()
        and mel_norm_path.is_file()
    )

    if not real_data_ready:
        logger.warning(
            "Real-data training of Random Forest, Deep MLP, and 2D CNN is BLOCKED because "
            "the official UrbanSound8K dataset and Step 2 split artifacts are not yet present "
            "at '%s'. Zero fake checkpoints, metrics, or learning curves were created.",
            dataset_report.dataset_root,
        )
        status_payload = {
            "coursework_stage": "Step 4 — CW1 T3 (Three Competing Models)",
            "seed": int(seed),
            "python_version": env_info.python_version,
            "torch_version": env_info.torch_version,
            "selected_device": env_info.device_info.selected_device,
            "cuda_available": env_info.device_info.cuda_available,
            "real_data_training_status": "BLOCKED_MISSING_URBANSOUND8K_DATASET",
            "held_out_test_split_touched": False,
            "deep_mlp_trainable_parameters": arch_payload["deep_mlp_detail"]["trainable_parameters"],
            "cnn_2d_trainable_parameters": arch_payload["cnn_2d_detail"]["trainable_parameters"],
            "tuning_budgets_configured": {
                "random_forest_candidates": len(model_cfg["random_forest"]["tuning_budget_candidates"]),
                "deep_mlp_candidates": len(model_cfg["deep_mlp"]["tuning_budget_candidates"]),
                "cnn_2d_candidates": len(model_cfg["cnn_2d"]["tuning_budget_candidates"]),
            },
        }
        (metrics_dir / "t3_experiment_status.json").write_text(
            json.dumps(status_payload, indent=2), encoding="utf-8"
        )
        return status_payload

    # ------------------------------------------------------------------
    # Real-Data Training & Validation Selection (Executed Only When Data Exists)
    # ------------------------------------------------------------------
    logger.info("Real UrbanSound8K dataset and splits verified. Starting Step 4 model training...")
    import numpy as np

    mfcc_scaler = MFCCStandardScaler.load(mfcc_scaler_path)
    mel_normalizer = MelSpectrogramNormalizer.load(mel_norm_path)

    mfcc_cache_path = paths["processed_data_dir"] / "train_val_mfcc_cache.npz"
    if mfcc_cache_path.is_file():
        mfcc_cached = np.load(mfcc_cache_path)
        X_tr_mfcc = mfcc_cached["X_train"].astype(np.float32)
        y_tr_mfcc = mfcc_cached["y_train"].astype(np.int64)
        X_va_mfcc = mfcc_cached["X_val"].astype(np.float32)
        y_va_mfcc = mfcc_cached["y_val"].astype(np.int64)
    else:
        mfcc_loaders = create_dataloaders(
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            test_manifest=test_manifest,
            audio_dir=paths["audio_dir"],
            representation="mfcc",
            batch_size=int(model_cfg["deep_mlp"]["training"]["batch_size"]),
            seed=seed,
            mfcc_scaler=mfcc_scaler,
        )
        X_tr_mfcc, y_tr_mfcc = _extract_numpy_from_loader(mfcc_loaders["train"])
        X_va_mfcc, y_va_mfcc = _extract_numpy_from_loader(mfcc_loaders["val"])

    # 1. Random Forest Baseline
    logger.info("1/3 Tuning and selecting Random Forest Baseline across %d candidates...", len(model_cfg["random_forest"]["tuning_budget_candidates"]))
    rf_out = tune_and_select_random_forest(
        X_train=X_tr_mfcc,
        y_train=y_tr_mfcc,
        X_val=X_va_mfcc,
        y_val=y_va_mfcc,
        candidates=model_cfg["random_forest"]["tuning_budget_candidates"],
        seed=seed,
        checkpoint_path=checkpoints_dir / f"random_forest_t3_seed{seed}.joblib",
        scaler_ref=str(mfcc_scaler_path),
    )
    rf_json_payload = json.dumps(
        {
            "selected_trial": rf_out["selected_trial"],
            "all_trials": rf_out["all_trials"],
            "checkpoint_metadata": rf_out["checkpoint_metadata"],
        },
        indent=2,
    )
    (metrics_dir / f"t3_random_forest_val_seed{seed}.json").write_text(rf_json_payload, encoding="utf-8")
    (metrics_dir / f"t3_rf_tuning_seed{seed}.json").write_text(rf_json_payload, encoding="utf-8")
    logger.info("Random Forest best validation macro-F1: %.4f", rf_out["selected_trial"]["val_macro_f1"])

    # 2. Deep MLP
    logger.info("2/3 Training Deep MLP on 80-D MFCC features...")
    mlp_cfg = model_cfg["deep_mlp"]
    mlp_tr_loader, mlp_va_loader = make_tensor_dataloaders(
        X_train=X_tr_mfcc,
        y_train=y_tr_mfcc,
        X_val=X_va_mfcc,
        y_val=y_va_mfcc,
        batch_size=int(mlp_cfg["training"]["batch_size"]),
        seed=seed,
    )
    mlp_model = DeepMLP(
        input_dim=80,
        hidden_dims=tuple(mlp_cfg["architecture"]["hidden_dims"]),
        output_dim=1,
        dropout=float(mlp_cfg["architecture"]["dropout"]),
        seed=seed,
    )
    mlp_res = train_pytorch_model(
        model=mlp_model,
        train_loader=mlp_tr_loader,
        val_loader=mlp_va_loader,
        learning_rate=float(mlp_cfg["training"]["learning_rate"]),
        weight_decay=float(mlp_cfg["training"]["weight_decay"]),
        max_epochs=int(mlp_cfg["training"]["max_epochs"]),
        patience=int(mlp_cfg["training"]["early_stopping_patience"]),
        seed=seed,
        checkpoint_path=checkpoints_dir / f"deep_mlp_t3_seed{seed}.pt",
        extra_metadata={"preprocessing_scaler_ref": str(mfcc_scaler_path)},
    )
    mlp_meta = mlp_res["training_metadata"]
    plot_training_curves(
        train_losses=mlp_meta["train_loss_history"],
        val_losses=mlp_meta["val_loss_history"],
        val_macro_f1s=mlp_meta["val_macro_f1_history"],
        best_epoch=mlp_meta["best_epoch"],
        model_title="Deep MLP (80-D MFCC)",
        output_path=figures_dir / f"fig_t3_deep_mlp_curves_seed{seed}.png",
    )
    (metrics_dir / f"t3_deep_mlp_val_seed{seed}.json").write_text(
        json.dumps(mlp_meta, indent=2), encoding="utf-8"
    )
    logger.info("Deep MLP best validation macro-F1: %.4f (Epoch %d)", mlp_meta["best_val_macro_f1"], mlp_meta["best_epoch"])

    # 3. 2D CNN
    logger.info("3/3 Training 2D CNN on (1, 128, 173) Log-Mel spectrograms...")
    cnn_cfg = model_cfg["cnn_2d"]
    mel_cache_path = paths["processed_data_dir"] / "train_val_mel_cache.npz"
    if mel_cache_path.is_file():
        mel_cached = np.load(mel_cache_path)
        X_tr_mel = mel_cached["X_train"].astype(np.float32)
        y_tr_mel = mel_cached["y_train"].astype(np.int64)
        X_va_mel = mel_cached["X_val"].astype(np.float32)
        y_va_mel = mel_cached["y_val"].astype(np.int64)
        cnn_tr_loader, cnn_va_loader = make_tensor_dataloaders(
            X_train=X_tr_mel,
            y_train=y_tr_mel,
            X_val=X_va_mel,
            y_val=y_va_mel,
            batch_size=int(cnn_cfg["training"]["batch_size"]),
            seed=seed,
        )
    else:
        mel_loaders = create_dataloaders(
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            test_manifest=test_manifest,
            audio_dir=paths["audio_dir"],
            representation="mel",
            batch_size=int(cnn_cfg["training"]["batch_size"]),
            seed=seed,
            mel_normalizer=mel_normalizer,
        )
        cnn_tr_loader = mel_loaders["train"]
        cnn_va_loader = mel_loaders["val"]

    cnn_model = Conv2DCNN(
        in_channels=1,
        conv_channels=tuple(cnn_cfg["architecture"]["conv_channels"]),
        n_mels=128,
        time_frames=173,
        dropout=float(cnn_cfg["architecture"]["dropout"]),
        output_dim=1,
        seed=seed,
    )
    cnn_res = train_pytorch_model(
        model=cnn_model,
        train_loader=cnn_tr_loader,
        val_loader=cnn_va_loader,
        learning_rate=float(cnn_cfg["training"]["learning_rate"]),
        weight_decay=float(cnn_cfg["training"]["weight_decay"]),
        max_epochs=int(cnn_cfg["training"]["max_epochs"]),
        patience=int(cnn_cfg["training"]["early_stopping_patience"]),
        seed=seed,
        checkpoint_path=checkpoints_dir / f"cnn_2d_t3_seed{seed}.pt",
        extra_metadata={"preprocessing_norm_ref": str(mel_norm_path)},
    )
    cnn_meta = cnn_res["training_metadata"]
    plot_training_curves(
        train_losses=cnn_meta["train_loss_history"],
        val_losses=cnn_meta["val_loss_history"],
        val_macro_f1s=cnn_meta["val_macro_f1_history"],
        best_epoch=cnn_meta["best_epoch"],
        model_title="2D CNN (1x128x173 Log-Mel)",
        output_path=figures_dir / f"fig_t3_cnn_2d_curves_seed{seed}.png",
    )
    (metrics_dir / f"t3_cnn_2d_val_seed{seed}.json").write_text(
        json.dumps(cnn_meta, indent=2), encoding="utf-8"
    )
    logger.info("2D CNN best validation macro-F1: %.4f (Epoch %d)", cnn_meta["best_val_macro_f1"], cnn_meta["best_epoch"])

    status_payload = {
        "coursework_stage": "Step 4 — CW1 T3 (Three Competing Models)",
        "seed": int(seed),
        "real_data_training_status": "COMPLETED_ON_REAL_URBANSOUND8K",
        "held_out_test_split_touched": False,
        "validation_macro_f1": {
            "random_forest": rf_out["selected_trial"]["val_macro_f1"],
            "deep_mlp": mlp_meta["best_val_macro_f1"],
            "cnn_2d": cnn_meta["best_val_macro_f1"],
        },
    }
    (metrics_dir / "t3_experiment_status.json").write_text(
        json.dumps(status_payload, indent=2), encoding="utf-8"
    )
    return status_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Step 4 (CW1 T3) competing models architecture export and training."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    args = parser.parse_args(argv)
    run_step4_pipeline(seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())

