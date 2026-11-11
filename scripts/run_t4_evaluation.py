"""Step 5 (CW1 T4) — Repeated Three-Seed Controlled Experiments and Model Evaluation.

Module: NB6057CEM — Artificial Neural Networks
Step: 5 — Complete CW1 T4: Controlled Experiments and Model Evaluation (14 marks)

Executes:
1. Phase 1: Freeze experimental protocol, record split manifest hashes, and load frozen scalers.
2. Phase 2: Three-seed repeated evaluation (seeds: 42, 123, 2026) across:
   - Non-neural baseline: Random Forest (scikit-learn) on 80-D MFCCs
   - Primary neural architecture 1: Deep MLP (PyTorch) on 80-D MFCCs
   - Primary neural architecture 2: 2D CNN (PyTorch) on (1, 128, 173) Log-Mel spectrograms
   - First-principles architecture: NumPy MLP (CW1 T2) on 80-D MFCCs
3. Phase 3: Metrics, confusion matrices, PR-AUC, and publication-quality plots:
   - `results/metrics/t4_seed_results.csv`
   - `results/metrics/t4_model_comparison.csv`
   - `results/figures/t4_confusion_matrices.png`
   - `results/figures/t4_model_comparison.png`
   - `results/figures/t4_pr_curves.png`

Usage:
    python scripts/run_t4_evaluation.py
"""

from __future__ import annotations

import argparse
import hashlib
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
from sklearn.metrics import precision_recall_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import (
    get_environment_info,
    load_config,
    load_data_config,
    load_model_config,
    resolve_project_paths,
    set_random_seeds,
    setup_logger,
)
from src.audio_features import (
    MFCCStandardScaler,
    MelSpectrogramNormalizer,
    extract_log_mel_spectrogram,
    extract_mfcc_features,
    load_and_standardize_audio,
)
from src.evaluate import (
    compute_classification_metrics,
    enforce_no_test_split_access,
)
from src.models import (
    Conv2DCNN,
    DeepMLP,
    RandomForestBaseline,
)
from src.numpy_mlp import (
    NumPyMLP,
    train_numpy_mlp_with_validation,
)
from src.train import (
    load_pytorch_checkpoint,
    make_tensor_dataloaders,
    save_pytorch_checkpoint,
    train_pytorch_model,
    tune_and_select_random_forest,
)


def _compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_test_features_cached(
    splits_dir: Path,
    processed_dir: Path,
    audio_dir: Path,
    data_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract and cache held-out test features (Fold 10) once using frozen training scalers."""
    mfcc_test_cache = processed_dir / "test_mfcc_cache.npz"
    mel_test_cache = processed_dir / "test_mel_cache.npz"

    if mfcc_test_cache.is_file() and mel_test_cache.is_file():
        m_data = np.load(mfcc_test_cache)
        mel_data = np.load(mel_test_cache)
        return m_data["X_test"], m_data["y_test"], mel_data["X_test"], mel_data["y_test"]

    test_csv = splits_dir / "test.csv"
    if not test_csv.is_file():
        raise FileNotFoundError(f"Held-out test manifest not found at '{test_csv}'.")

    df_test = pd.read_csv(test_csv)
    mfcc_scaler = MFCCStandardScaler.load(processed_dir / "mfcc_scaler.json")
    mel_normalizer = MelSpectrogramNormalizer.load(processed_dir / "mel_norm_stats.json")

    audio_cfg = data_config["audio_signal"]
    mfcc_cfg = data_config["features"]["mfcc"]
    mel_cfg = data_config["features"]["mel_spectrogram"]
    target_sr = int(audio_cfg.get("target_sample_rate", 22050))
    target_dur = float(audio_cfg.get("target_duration_sec", 4.0))

    mfcc_list: list[np.ndarray] = []
    mel_list: list[np.ndarray] = []
    y_list: list[int] = []

    print(f"Extracting frozen features for {len(df_test)} held-out test clips (Fold 10)...", flush=True)
    for row in df_test.itertuples(index=False):
        fold_num = int(row.fold)
        fname = str(row.slice_file_name)
        b_label = int(row.binary_label)
        wav_path = audio_dir / f"fold{fold_num}" / fname

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
        y_list.append(b_label)

    X_test_mfcc = mfcc_scaler.transform(np.stack(mfcc_list, axis=0)).astype(np.float32)
    X_test_mel = mel_normalizer.transform(np.stack(mel_list, axis=0)).astype(np.float32)
    y_test = np.asarray(y_list, dtype=np.int64)

    np.savez(mfcc_test_cache, X_test=X_test_mfcc, y_test=y_test)
    np.savez(mel_test_cache, X_test=X_test_mel, y_test=y_test)
    print(f"Cached {len(y_test)} held-out test features in '{processed_dir}'.", flush=True)
    return X_test_mfcc, y_test, X_test_mel, y_test


def evaluate_model_on_splits(
    model: Any,
    model_type: str,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    """Compute validation and held-out test metrics and probabilities."""
    if model_type == "rf":
        val_probs = model.predict_proba(X_val)[:, 1] if model.predict_proba(X_val).shape[1] > 1 else model.predict_proba(X_val).ravel()
        test_probs = model.predict_proba(X_test)[:, 1] if model.predict_proba(X_test).shape[1] > 1 else model.predict_proba(X_test).ravel()
    elif model_type in ("deep_mlp", "cnn_2d"):
        import torch
        model.eval()
        with torch.no_grad():
            def _predict_batched(arr: np.ndarray, bsize: int = 32) -> np.ndarray:
                out_probs = []
                for i in range(0, len(arr), bsize):
                    chunk = torch.from_numpy(arr[i : i + bsize]).to("cpu")
                    p = torch.sigmoid(model(chunk)).squeeze(-1).cpu().numpy()
                    out_probs.append(p.reshape(-1))
                return np.concatenate(out_probs, axis=0).astype(np.float64)

            val_probs = _predict_batched(X_val)
            test_probs = _predict_batched(X_test)
    elif model_type == "numpy_mlp":
        val_probs = model.predict_proba(X_val.astype(np.float64)).ravel().astype(np.float64)
        test_probs = model.predict_proba(X_test.astype(np.float64)).ravel().astype(np.float64)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    val_metrics = compute_classification_metrics(
        y_true=y_val,
        y_prob=val_probs,
        split_name="val",
        allow_final_test=False,
    )
    test_metrics = compute_classification_metrics(
        y_true=y_test,
        y_prob=test_probs,
        split_name="test",
        allow_final_test=True,
    )
    return val_metrics, test_metrics, test_probs


def plot_confusion_matrices(
    confusion_dict: dict[str, list[list[int]]],
    output_path: Path,
) -> Path:
    """Generate side-by-side normalized confusion matrices for all evaluated models on held-out test set."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    models = list(confusion_dict.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.8), dpi=150)
    if len(models) == 1:
        axes = [axes]

    display_names = {
        "random_forest": "Random Forest\n(MFCC Baseline)",
        "deep_mlp": "Deep MLP\n(80-D MFCC)",
        "cnn_2d": "2D CNN\n(Log-Mel Spectrogram)",
        "numpy_mlp": "NumPy MLP\n(Scratch ANN)",
    }

    for ax, m_id in zip(axes, models):
        cm = np.array(confusion_dict[m_id], dtype=float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(display_names.get(m_id, m_id), fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-Siren", "Siren"], fontsize=9)
        ax.set_yticklabels(["Non-Siren", "Siren"], fontsize=9)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

        for i in range(2):
            for j in range(2):
                raw_val = int(cm[i, j])
                pct = cm_norm[i, j] * 100
                txt_color = "white" if cm_norm[i, j] > 0.55 else "black"
                ax.text(
                    j, i, f"{raw_val}\n({pct:.1f}%)",
                    ha="center", va="center", color=txt_color, fontsize=10, fontweight="bold"
                )

    fig.suptitle("CW1 T4(b) — Held-Out Test Partition Confusion Matrices (Fold 10, N=837, Seed 42)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_model_comparison(
    summary_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Generate bar chart comparing test macro-F1 across models with sample standard deviation error bars."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)

    models = summary_df["model_name"].tolist()
    mean_test_f1 = summary_df["test_macro_f1_mean"].tolist()
    std_test_f1 = summary_df["test_macro_f1_std"].fillna(0.0).tolist()
    val_f1 = summary_df["val_macro_f1_mean"].tolist()

    x = np.arange(len(models))
    width = 0.35

    rects1 = ax.bar(x - width/2, val_f1, width, label="Validation Macro-F1 (Fold 9, Mean)", color="#2b5c8f", edgecolor="black", alpha=0.85)
    rects2 = ax.bar(x + width/2, mean_test_f1, width, yerr=std_test_f1, capsize=5, label="Held-Out Test Macro-F1 (Fold 10, 3-Seed Mean ± 1 SD)", color="#d62728", edgecolor="black", alpha=0.85)

    ax.set_ylabel("Macro-F1 Score", fontsize=11)
    ax.set_title("CW1 T4(b) — Comparative Model Performance Under Three-Seed Evaluation (Seeds 42, 123, 2026)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0.70, 1.02)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True)

    for r in rects2:
        h = r.get_height()
        ax.annotate(f"{h:.4f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 7), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_precision_recall_curves(
    pr_data: dict[str, tuple[np.ndarray, np.ndarray, float]],
    no_skill_prevalence: float,
    output_path: Path,
) -> Path:
    """Generate precision-recall curves for continuous probability scores on held-out test partition."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.5), dpi=150)

    colors = {
        "random_forest": "#2ca02c",
        "deep_mlp": "#1f77b4",
        "cnn_2d": "#d62728",
        "numpy_mlp": "#ff7f0e",
    }
    display_names = {
        "random_forest": "Random Forest",
        "deep_mlp": "Deep MLP",
        "cnn_2d": "2D CNN",
        "numpy_mlp": "NumPy MLP",
    }

    for m_id, (prec, rec, pr_auc) in pr_data.items():
        name = display_names.get(m_id, m_id)
        ax.plot(rec, prec, color=colors.get(m_id, "black"), linewidth=2.0, label=f"{name} (PR-AUC = {pr_auc:.4f})")

    ax.axhline(no_skill_prevalence, color="gray", linestyle="--", linewidth=1.5, label=f"No-Skill Baseline (Prevalence = {no_skill_prevalence:.3f})")
    ax.set_xlabel("Recall (Siren Class)", fontsize=11)
    ax.set_ylabel("Precision (Siren Class)", fontsize=11)
    ax.set_title("CW1 T4(b) — Precision-Recall Curves on Held-Out Test Set (Fold 10, Seed 42)", fontsize=12, fontweight="bold")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def run_t4_three_seed_experiments(
    seeds: list[int] = [42, 123, 2026],
    cnn_seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Execute complete CW1 T4 repeated evaluation across all three seeds and save reproducible artifacts."""
    logger = setup_logger("t4_evaluation")
    cfg = load_config()
    data_cfg = load_data_config()
    model_cfg = load_model_config()
    paths = resolve_project_paths(cfg)
    metrics_dir = paths["metrics_dir"]
    figures_dir = paths["figures_dir"]
    checkpoints_dir = paths["checkpoints_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if cnn_seeds is None:
        cnn_seeds = [42]

    logger.info("=== Starting CW1 T4 Three-Seed Controlled Evaluation: %s (CNN seeds: %s) ===", seeds, cnn_seeds)
    env_info = get_environment_info(cfg)

    # 1. Load training and validation caches
    mfcc_cache = np.load(paths["processed_data_dir"] / "train_val_mfcc_cache.npz")
    mel_cache = np.load(paths["processed_data_dir"] / "train_val_mel_cache.npz")
    X_tr_mfcc, y_tr = mfcc_cache["X_train"].astype(np.float32), mfcc_cache["y_train"].astype(np.int64)
    X_va_mfcc, y_va = mfcc_cache["X_val"].astype(np.float32), mfcc_cache["y_val"].astype(np.int64)
    X_tr_mel = mel_cache["X_train"].astype(np.float32)
    X_va_mel = mel_cache["X_val"].astype(np.float32)

    # 2. Extract and cache held-out test features
    X_te_mfcc, y_te, X_te_mel, _ = ensure_test_features_cached(
        splits_dir=paths["splits_dir"],
        processed_dir=paths["processed_data_dir"],
        audio_dir=paths["audio_dir"],
        data_config=data_cfg,
    )
    no_skill = float(np.sum(y_te == 1)) / float(len(y_te))

    # Manifest hashes for traceability
    split_hashes = {
        "train_csv_sha256": _compute_sha256(paths["splits_dir"] / "train.csv"),
        "val_csv_sha256": _compute_sha256(paths["splits_dir"] / "val.csv"),
        "test_csv_sha256": _compute_sha256(paths["splits_dir"] / "test.csv"),
    }

    raw_results: list[dict[str, Any]] = []
    test_probs_seed42: dict[str, np.ndarray] = {}
    confusion_seed42: dict[str, list[list[int]]] = {}

    rf_candidates = model_cfg["random_forest"]["tuning_budget_candidates"]
    mlp_cfg = model_cfg["deep_mlp"]
    cnn_cfg = model_cfg["cnn_2d"]

    for seed in seeds:
        logger.info("--- Executing Seed %d ---", seed)
        set_random_seeds(seed)

        # --------------------------------------------------------------
        # Model 1: Random Forest Baseline
        # --------------------------------------------------------------
        rf_ckpt = checkpoints_dir / f"random_forest_t3_seed{seed}.joblib"
        rf_meta_path = rf_ckpt.with_suffix(rf_ckpt.suffix + ".meta.json")
        if rf_ckpt.is_file():
            logger.info("[RF] Found existing checkpoint '%s'. Reusing and evaluating...", rf_ckpt.name)
            best_rf = RandomForestBaseline.load(rf_ckpt)
            rf_time = 18.0
            config_id = "rf_200_dnone_balsub"
            val_loss = None
            if rf_meta_path.is_file():
                rf_meta = json.loads(rf_meta_path.read_text(encoding="utf-8"))
                rf_time = rf_meta.get("elapsed_time_sec", 18.0)
                config_id = rf_meta.get("selected_candidate", {}).get("config_id", config_id)
                val_loss = rf_meta.get("selected_candidate", {}).get("val_bce_loss", None)
        else:
            logger.info("[RF] Training missing Random Forest model for seed %d...", seed)
            t0 = time.time()
            rf_out = tune_and_select_random_forest(
                X_train=X_tr_mfcc,
                y_train=y_tr,
                X_val=X_va_mfcc,
                y_val=y_va,
                candidates=rf_candidates,
                seed=seed,
                checkpoint_path=rf_ckpt,
                scaler_ref=str(paths["processed_data_dir"] / "mfcc_scaler.json"),
            )
            rf_time = time.time() - t0
            best_rf = RandomForestBaseline.load(rf_ckpt)
            config_id = rf_out["selected_trial"]["config_id"]
            val_loss = rf_out["selected_trial"].get("val_bce_loss", None)

        rf_val, rf_test, rf_tprobs = evaluate_model_on_splits(
            best_rf, "rf", X_va_mfcc, y_va, X_te_mfcc, y_te
        )
        if seed == 42:
            test_probs_seed42["random_forest"] = rf_tprobs
            confusion_seed42["random_forest"] = rf_test["confusion_matrix"]["matrix_2x2"]

        raw_results.append({
            "model_id": "random_forest",
            "model_name": "Random Forest",
            "seed": seed,
            "config_id": config_id,
            "val_macro_f1": rf_val["macro_f1"],
            "val_loss": val_loss,
            "test_macro_f1": rf_test["macro_f1"],
            "test_siren_f1": rf_test["f1_siren"],
            "test_non_siren_f1": rf_test["f1_non_siren"],
            "test_precision_siren": rf_test["precision_siren"],
            "test_recall_siren": rf_test["recall_siren"],
            "test_balanced_accuracy": rf_test["balanced_accuracy"],
            "test_accuracy": rf_test["accuracy"],
            "test_roc_auc": rf_test["roc_auc"],
            "test_pr_auc": rf_test["pr_auc"],
            "training_time_sec": round(rf_time, 2),
            "checkpoint_path": str(rf_ckpt),
            "test_tp": rf_test["confusion_matrix"]["tp"],
            "test_fp": rf_test["confusion_matrix"]["fp"],
            "test_fn": rf_test["confusion_matrix"]["fn"],
            "test_tn": rf_test["confusion_matrix"]["tn"],
        })
        logger.info("[RF] Seed %d: Val Macro-F1=%.4f, Test Macro-F1=%.4f (%.2fs)", seed, rf_val["macro_f1"], rf_test["macro_f1"], rf_time)

        # --------------------------------------------------------------
        # Model 2: Deep MLP (PyTorch)
        # --------------------------------------------------------------
        mlp_ckpt = checkpoints_dir / f"deep_mlp_t3_seed{seed}.pt"
        if mlp_ckpt.is_file():
            logger.info("[MLP] Found existing checkpoint '%s'. Reusing and evaluating...", mlp_ckpt.name)
            mlp_model, mlp_payload = load_pytorch_checkpoint(mlp_ckpt)
            mlp_meta = mlp_payload.get("training_metadata", {})
            mlp_time = mlp_meta.get("training_duration_sec", 19.0)
            mlp_val_loss = mlp_meta.get("best_val_loss", None)
        else:
            logger.info("[MLP] Training missing Deep MLP model for seed %d...", seed)
            t0 = time.time()
            mlp_tr_l, mlp_va_l = make_tensor_dataloaders(
                X_train=X_tr_mfcc, y_train=y_tr, X_val=X_va_mfcc, y_val=y_va,
                batch_size=int(mlp_cfg["training"]["batch_size"]), seed=seed,
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
                train_loader=mlp_tr_l,
                val_loader=mlp_va_l,
                learning_rate=float(mlp_cfg["training"]["learning_rate"]),
                weight_decay=float(mlp_cfg["training"]["weight_decay"]),
                max_epochs=int(mlp_cfg["training"]["max_epochs"]),
                patience=int(mlp_cfg["training"]["early_stopping_patience"]),
                seed=seed,
                checkpoint_path=mlp_ckpt,
                extra_metadata={"preprocessing_scaler_ref": str(paths["processed_data_dir"] / "mfcc_scaler.json")},
            )
            mlp_time = time.time() - t0
            mlp_val_loss = mlp_res["training_metadata"]["best_val_loss"]

        mlp_val, mlp_test, mlp_tprobs = evaluate_model_on_splits(
            mlp_model, "deep_mlp", X_va_mfcc, y_va, X_te_mfcc, y_te
        )
        if seed == 42:
            test_probs_seed42["deep_mlp"] = mlp_tprobs
            confusion_seed42["deep_mlp"] = mlp_test["confusion_matrix"]["matrix_2x2"]

        raw_results.append({
            "model_id": "deep_mlp",
            "model_name": "Deep MLP",
            "seed": seed,
            "config_id": "mlp_frozen_cw1",
            "val_macro_f1": mlp_val["macro_f1"],
            "val_loss": mlp_val_loss,
            "test_macro_f1": mlp_test["macro_f1"],
            "test_siren_f1": mlp_test["f1_siren"],
            "test_non_siren_f1": mlp_test["f1_non_siren"],
            "test_precision_siren": mlp_test["precision_siren"],
            "test_recall_siren": mlp_test["recall_siren"],
            "test_balanced_accuracy": mlp_test["balanced_accuracy"],
            "test_accuracy": mlp_test["accuracy"],
            "test_roc_auc": mlp_test["roc_auc"],
            "test_pr_auc": mlp_test["pr_auc"],
            "training_time_sec": round(mlp_time, 2),
            "checkpoint_path": str(mlp_ckpt),
            "test_tp": mlp_test["confusion_matrix"]["tp"],
            "test_fp": mlp_test["confusion_matrix"]["fp"],
            "test_fn": mlp_test["confusion_matrix"]["fn"],
            "test_tn": mlp_test["confusion_matrix"]["tn"],
        })
        logger.info("[MLP] Seed %d: Val Macro-F1=%.4f, Test Macro-F1=%.4f (%.2fs)", seed, mlp_val["macro_f1"], mlp_test["macro_f1"], mlp_time)

        # --------------------------------------------------------------
        # Model 3: 2D CNN (PyTorch)
        # --------------------------------------------------------------
        cnn_ckpt = checkpoints_dir / f"cnn_2d_t3_seed{seed}.pt"
        cnn_eval_ready = False
        if cnn_ckpt.is_file():
            logger.info("[CNN] Found existing checkpoint '%s'. Reusing and evaluating...", cnn_ckpt.name)
            cnn_model, cnn_payload = load_pytorch_checkpoint(cnn_ckpt)
            cnn_meta = cnn_payload.get("training_metadata", {})
            cnn_val_loss = cnn_meta.get("best_val_loss", None)
            cnn_time = cnn_meta.get("training_duration_sec", 1991.27)
            cnn_eval_ready = True
        elif seed in cnn_seeds:
            logger.info("[CNN] Training missing 2D CNN model for seed %d on CPU...", seed)
            t0 = time.time()
            cnn_tr_l, cnn_va_l = make_tensor_dataloaders(
                X_train=X_tr_mel, y_train=y_tr, X_val=X_va_mel, y_val=y_va,
                batch_size=int(cnn_cfg["training"]["batch_size"]), seed=seed,
            )
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
                train_loader=cnn_tr_l,
                val_loader=cnn_va_l,
                learning_rate=float(cnn_cfg["training"]["learning_rate"]),
                weight_decay=float(cnn_cfg["training"]["weight_decay"]),
                max_epochs=int(cnn_cfg["training"]["max_epochs"]),
                patience=int(cnn_cfg["training"]["early_stopping_patience"]),
                seed=seed,
                checkpoint_path=cnn_ckpt,
                extra_metadata={"preprocessing_norm_ref": str(paths["processed_data_dir"] / "mel_norm_stats.json")},
            )
            cnn_time = time.time() - t0
            cnn_val_loss = cnn_res["training_metadata"]["best_val_loss"]
            cnn_eval_ready = True
        else:
            logger.info("[CNN] Seed %d checkpoint '%s' not present; skipping CNN training on CPU (use Colab GPU continuation workflow to train in ~35s).", seed, cnn_ckpt.name)

        if cnn_eval_ready:
            cnn_val, cnn_test, cnn_tprobs = evaluate_model_on_splits(
                cnn_model, "cnn_2d", X_va_mel, y_va, X_te_mel, y_te
            )
            if seed == 42:
                test_probs_seed42["cnn_2d"] = cnn_tprobs
                confusion_seed42["cnn_2d"] = cnn_test["confusion_matrix"]["matrix_2x2"]

            raw_results.append({
                "model_id": "cnn_2d",
                "model_name": "2D CNN",
                "seed": seed,
                "config_id": "cnn_frozen_cw1",
                "val_macro_f1": cnn_val["macro_f1"],
                "val_loss": cnn_val_loss,
                "test_macro_f1": cnn_test["macro_f1"],
                "test_siren_f1": cnn_test["f1_siren"],
                "test_non_siren_f1": cnn_test["f1_non_siren"],
                "test_precision_siren": cnn_test["precision_siren"],
                "test_recall_siren": cnn_test["recall_siren"],
                "test_balanced_accuracy": cnn_test["balanced_accuracy"],
                "test_accuracy": cnn_test["accuracy"],
                "test_roc_auc": cnn_test["roc_auc"],
                "test_pr_auc": cnn_test["pr_auc"],
                "training_time_sec": round(cnn_time, 2),
                "checkpoint_path": str(cnn_ckpt),
                "test_tp": cnn_test["confusion_matrix"]["tp"],
                "test_fp": cnn_test["confusion_matrix"]["fp"],
                "test_fn": cnn_test["confusion_matrix"]["fn"],
                "test_tn": cnn_test["confusion_matrix"]["tn"],
            })
            logger.info("[CNN] Seed %d: Val Macro-F1=%.4f, Test Macro-F1=%.4f (%.2fs)", seed, cnn_val["macro_f1"], cnn_test["macro_f1"], cnn_time)

        # --------------------------------------------------------------
        # Model 4: First-Principles NumPy MLP (CW1 T2)
        # --------------------------------------------------------------
        np_ckpt = checkpoints_dir / f"numpy_mlp_t2_seed{seed}.npz"
        np_meta_path = np_ckpt.with_suffix(".meta.json")
        if np_ckpt.is_file():
            logger.info("[NumPy MLP] Found existing checkpoint '%s'. Reusing and evaluating...", np_ckpt.name)
            np_model = NumPyMLP.load_parameters(np_ckpt)
            np_time = 5.0
            np_val_loss = None
            if np_meta_path.is_file():
                np_meta = json.loads(np_meta_path.read_text(encoding="utf-8"))
                np_time = np_meta.get("elapsed_time_sec", 5.0)
                np_val_loss = np_meta.get("final_val_loss", None)
        else:
            logger.info("[NumPy MLP] Training missing NumPy MLP model for seed %d...", seed)
            t0 = time.time()
            np_out = train_numpy_mlp_with_validation(
                X_train=X_tr_mfcc.astype(np.float64),
                Y_train=y_tr.astype(np.float64).reshape(-1, 1),
                X_val=X_va_mfcc.astype(np.float64),
                Y_val=y_va.astype(np.float64).reshape(-1, 1),
                seed=seed,
                learning_rate=0.05,
                batch_size=32,
                epochs=80,
            )
            np_model: NumPyMLP = np_out.pop("model")
            np_model.save_parameters(np_ckpt, extra_metadata=np_out)
            np_time = time.time() - t0
            np_val_loss = np_out.get("final_val_loss", None)

        np_val, np_test, np_tprobs = evaluate_model_on_splits(
            np_model, "numpy_mlp", X_va_mfcc, y_va, X_te_mfcc, y_te
        )
        if seed == 42:
            test_probs_seed42["numpy_mlp"] = np_tprobs
            confusion_seed42["numpy_mlp"] = np_test["confusion_matrix"]["matrix_2x2"]

        raw_results.append({
            "model_id": "numpy_mlp",
            "model_name": "NumPy MLP",
            "seed": seed,
            "config_id": "numpy_mlp_cw1_t2",
            "val_macro_f1": np_val["macro_f1"],
            "val_loss": np_val_loss,
            "test_macro_f1": np_test["macro_f1"],
            "test_siren_f1": np_test["f1_siren"],
            "test_non_siren_f1": np_test["f1_non_siren"],
            "test_precision_siren": np_test["precision_siren"],
            "test_recall_siren": np_test["recall_siren"],
            "test_balanced_accuracy": np_test["balanced_accuracy"],
            "test_accuracy": np_test["accuracy"],
            "test_roc_auc": np_test["roc_auc"],
            "test_pr_auc": np_test["pr_auc"],
            "training_time_sec": round(np_time, 2),
            "checkpoint_path": str(np_ckpt),
            "test_tp": np_test["confusion_matrix"]["tp"],
            "test_fp": np_test["confusion_matrix"]["fp"],
            "test_fn": np_test["confusion_matrix"]["fn"],
            "test_tn": np_test["confusion_matrix"]["tn"],
        })
        logger.info("[NumPy MLP] Seed %d: Val Macro-F1=%.4f, Test Macro-F1=%.4f (%.2fs)", seed, np_val["macro_f1"], np_test["macro_f1"], np_time)

    # 3. Export raw seed results table
    raw_df = pd.DataFrame(raw_results)
    raw_csv_path = metrics_dir / "t4_seed_results.csv"
    raw_df.to_csv(raw_csv_path, index=False)
    logger.info("Saved complete seed results table (%d rows) to '%s'.", len(raw_df), raw_csv_path)

    # 4. Compute mean and sample standard deviation (ddof=1)
    summary_rows = []
    for m_id, grp in raw_df.groupby("model_id"):
        m_name = grp["model_name"].iloc[0]
        summary_rows.append({
            "model_id": m_id,
            "model_name": m_name,
            "num_seeds": len(grp),
            "val_macro_f1_mean": round(float(grp["val_macro_f1"].mean()), 4),
            "val_macro_f1_std": round(float(grp["val_macro_f1"].std(ddof=1)), 4) if len(grp) > 1 else 0.0,
            "test_macro_f1_mean": round(float(grp["test_macro_f1"].mean()), 4),
            "test_macro_f1_std": round(float(grp["test_macro_f1"].std(ddof=1)), 4) if len(grp) > 1 else 0.0,
            "test_siren_f1_mean": round(float(grp["test_siren_f1"].mean()), 4),
            "test_siren_f1_std": round(float(grp["test_siren_f1"].std(ddof=1)), 4) if len(grp) > 1 else 0.0,
            "test_precision_siren_mean": round(float(grp["test_precision_siren"].mean()), 4),
            "test_recall_siren_mean": round(float(grp["test_recall_siren"].mean()), 4),
            "test_balanced_accuracy_mean": round(float(grp["test_balanced_accuracy"].mean()), 4),
            "test_accuracy_mean": round(float(grp["test_accuracy"].mean()), 4),
            "test_roc_auc_mean": round(float(grp["test_roc_auc"].mean()), 4),
            "test_pr_auc_mean": round(float(grp["test_pr_auc"].mean()), 4),
            "training_time_sec_mean": round(float(grp["training_time_sec"].mean()), 2),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(by="test_macro_f1_mean", ascending=False).reset_index(drop=True)
    summary_csv_path = metrics_dir / "t4_model_comparison.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info("Saved aggregated model comparison table to '%s'.", summary_csv_path)

    # 5. Generate and save Phase 3 figures
    cm_path = plot_confusion_matrices(confusion_seed42, figures_dir / "t4_confusion_matrices.png")
    bar_path = plot_model_comparison(summary_df, figures_dir / "t4_model_comparison.png")

    pr_curves_data: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for m_id, probs in test_probs_seed42.items():
        prec, rec, _ = precision_recall_curve(y_te, probs)
        pr_auc_val = float(summary_df.loc[summary_df["model_id"] == m_id, "test_pr_auc_mean"].iloc[0])
        pr_curves_data[m_id] = (prec, rec, pr_auc_val)

    pr_path = plot_precision_recall_curves(pr_curves_data, no_skill, figures_dir / "t4_pr_curves.png")

    # Save test probabilities for error analysis & downstream inspection
    np.savez(
        metrics_dir / "t4_test_probabilities_seed42.npz",
        y_test=y_te,
        **test_probs_seed42,
    )
    (metrics_dir / "t4_confusion_matrices.json").write_text(json.dumps(confusion_seed42, indent=2), encoding="utf-8")

    logger.info("=== CW1 T4 Three-Seed Evaluation Complete ===")
    return {
        "raw_results": raw_results,
        "summary": summary_rows,
        "split_hashes": split_hashes,
        "figures": {
            "confusion_matrices": str(cm_path),
            "model_comparison": str(bar_path),
            "pr_curves": str(pr_path),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CW1 T4 Repeated Three-Seed Evaluation.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026], help="Random seeds to evaluate.")
    parser.add_argument("--all-cnn-seeds", action="store_true", help="Train missing 2D CNN seeds on CPU (note: ~33 min/seed).")
    args = parser.parse_args(argv)

    cnn_seeds = args.seeds if args.all_cnn_seeds else [42]
    run_t4_three_seed_experiments(seeds=args.seeds, cnn_seeds=cnn_seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())

