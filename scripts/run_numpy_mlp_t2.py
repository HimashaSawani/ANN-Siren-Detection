"""Step 3 (CW1 T2) — NumPy MLP Gradient Verification, Tiny-Batch Diagnosis, and Real-Data Runner.

Module: NB6057CEM — Artificial Neural Networks
Step: 3 — Complete Real-Data Preparation and Implement NumPy ANN From Scratch

Executes:
1. Phase 3: Central finite-difference gradient verification (epsilon = 1e-5, float64)
   across 11 parameters spanning W1, b1, W2, and b2.
   - Saves `results/metrics/t2_gradient_verification.json`
   - Saves `results/metrics/t2_gradient_verification_table.csv`
   - Saves `results/figures/fig_t2_gradient_verification.png`
2. Phase 4: Controlled tiny-batch overfitting numerical diagnosis (seed = 42, N = 16).
   - Saves `results/metrics/t2_tiny_batch_diagnosis.json`
   - Saves `results/figures/fig_t2_tiny_batch_overfitting.png`
3. Phase 5: Real-data MFCC training and validation evaluation (executed ONLY if the
   real UrbanSound8K dataset and Step 2 split manifests exist and pass validation).
4. Phase 6: Saves architecture/parameter tables and experiment metadata.

Usage:
    python scripts/run_numpy_mlp_t2.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import (  # noqa: E402
    get_environment_info,
    load_config,
    load_data_config,
    resolve_project_paths,
    setup_logger,
)
from src.audio_features import (  # noqa: E402
    MFCCStandardScaler,
    extract_mfcc_features,
    load_and_standardize_audio,
)
from src.numpy_mlp import (  # noqa: E402
    GradientCheckReport,
    HIDDEN_DIM,
    INPUT_DIM,
    NumPyMLP,
    OUTPUT_DIM,
    TOTAL_PARAMETERS,
    TinyBatchDiagnosisReport,
    run_tiny_batch_diagnosis,
    train_numpy_mlp_with_validation,
    verify_gradients_finite_difference,
)
from src.preprocessing import validate_urbansound8k_dataset  # noqa: E402


def plot_gradient_verification(
    report: GradientCheckReport,
    output_path: Path,
) -> Path:
    """Plot analytical vs. numerical gradients and log10 relative discrepancies."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{e.parameter_name}{list(e.index)}" for e in report.entries]
    ana_vals = [e.analytical_gradient for e in report.entries]
    num_vals = [e.numerical_gradient for e in report.entries]
    rel_diffs = [max(e.relative_difference, 1e-16) for e in report.entries]

    x = np.arange(len(labels))
    width = 0.36

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.0, 7.5), dpi=150, sharex=True)

    ax1.bar(
        x - width / 2,
        ana_vals,
        width=width,
        color="#1f77b4",
        edgecolor="black",
        label="Analytical Backpropagation Gradient",
    )
    ax1.bar(
        x + width / 2,
        num_vals,
        width=width,
        color="#ff7f0e",
        edgecolor="black",
        alpha=0.85,
        label=f"Central Finite-Difference Gradient (eps={report.epsilon:.0e})",
    )
    ax1.axhline(0.0, color="black", linewidth=0.8)
    ax1.set_title(
        "CW1 T2(c) — Analytical vs. Central Finite-Difference Gradients Across W1, b1, W2, b2",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_ylabel("Gradient Value", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", frameon=True)

    log_rel = np.log10(rel_diffs)
    log_tol = np.log10(report.tolerance_relative)
    ax2.bar(
        x,
        log_rel,
        color="#2ca02c",
        edgecolor="black",
        width=0.5,
        label="log10(Relative Gradient Discrepancy)",
    )
    ax2.axhline(
        log_tol,
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        label=f"Relative Tolerance Threshold (1e-6 -> log10 = {log_tol:.1f})",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("log10(Relative Difference)", fontsize=10)
    ax2.set_xlabel("Inspected Parameter Entry", fontsize=10)
    ax2.set_ylim(-16.0, -4.0)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_tiny_batch_overfitting(
    report: TinyBatchDiagnosisReport,
    output_path: Path,
) -> Path:
    """Plot training Binary Cross-Entropy loss vs. epochs for the tiny-batch diagnosis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(len(report.loss_history))

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)
    ax.plot(
        epochs,
        report.loss_history,
        color="#1f77b4",
        linewidth=2.2,
        label=(
            f"Tiny-Batch Training BCE Loss (N={report.batch_size}, lr={report.learning_rate}, "
            f"seed={report.seed})"
        ),
    )
    ax.axhline(
        report.target_loss_threshold,
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label=f"Stopping Loss Threshold ({report.target_loss_threshold:.3f})",
    )
    ax.scatter(
        [0, report.actual_epochs_completed],
        [report.initial_loss, report.final_loss],
        color="#2ca02c",
        zorder=5,
        s=55,
        label=(
            f"Endpoints: Initial={report.initial_loss:.4f} -> "
            f"Final={report.final_loss:.4f} (Epoch {report.actual_epochs_completed})"
        ),
    )
    ax.set_title(
        "CW1 T2(d) — Controlled Tiny-Batch Overfitting Numerical Diagnosis Curve",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Optimization Epoch", fontsize=11)
    ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_real_data_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    output_path: Path,
) -> Path:
    """Plot training and validation BCE loss curves when real UrbanSound8K training runs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)
    ax.plot(epochs, train_losses, color="#1f77b4", linewidth=2.0, label="Training BCE Loss (Folds 1-8)")
    ax.plot(epochs, val_losses, color="#d62728", linewidth=2.0, label="Validation BCE Loss (Fold 9)")
    ax.set_title("NumPy MLP Real-Data Training and Validation Loss Curves", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _extract_split_mfcc_matrix(
    manifest_csv: Path,
    audio_dir: Path,
    scaler: MFCCStandardScaler,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract and scale 80-D MFCC feature matrix and binary labels from a split manifest."""
    df = pd.read_csv(manifest_csv)
    features_list: list[np.ndarray] = []
    labels_list: list[int] = []

    for row in df.itertuples(index=False):
        fold_num = int(getattr(row, "fold"))
        fname = str(getattr(row, "slice_file_name"))
        b_label = int(getattr(row, "binary_label"))
        wav_path = audio_dir / f"fold{fold_num}" / fname
        waveform = load_and_standardize_audio(wav_path, target_sr=22050, target_duration_sec=4.0)
        mfcc_vec = extract_mfcc_features(waveform, sample_rate=22050)
        features_list.append(mfcc_vec)
        labels_list.append(b_label)

    raw_X = np.stack(features_list, axis=0)
    scaled_X = scaler.transform(raw_X).astype(np.float64)
    Y = np.asarray(labels_list, dtype=np.float64).reshape(-1, 1)
    return scaled_X, Y


def run_step3_pipeline() -> dict[str, Any]:
    """Execute Phases 3, 4, 5, and 6 for CW1 T2 and return structured summary."""
    logger = setup_logger("numpy_mlp_t2")
    cfg = load_config()
    data_cfg = load_data_config()
    paths = resolve_project_paths(cfg)
    metrics_dir = paths["metrics_dir"]
    figures_dir = paths["figures_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Architecture & Parameter Summary Table (Phase 1, 2, 6)
    # ------------------------------------------------------------------
    model = NumPyMLP(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, seed=42)
    arch_summary = {
        "model_id": "numpy_mlp",
        "framework": "numpy (pure first-principles, no autograd)",
        "sample_dimension_convention": "X in R^(N x 80) (rows = audio clips, cols = 80 MFCC summary features); Y in {0,1}^(N x 1)",
        "layers": [
            {
                "layer": "Input Layer",
                "input_shape": "(N, 80)",
                "output_shape": "(N, 80)",
                "activation": "Identity (Standardized MFCC summary vector)",
                "parameters": 0,
            },
            {
                "layer": "Hidden Dense Layer 1",
                "input_shape": "(N, 80)",
                "output_shape": "(N, 32)",
                "equation": "A1 = tanh(X @ W1 + b1)",
                "weight_shape": [80, 32],
                "bias_shape": [32],
                "parameters": 80 * 32 + 32,
            },
            {
                "layer": "Output Dense Layer 2",
                "input_shape": "(N, 32)",
                "output_shape": "(N, 1)",
                "equation": "Y_hat = sigmoid(A1 @ W2 + b2)",
                "weight_shape": [32, 1],
                "bias_shape": [1],
                "parameters": 32 * 1 + 1,
            },
        ],
        "parameter_breakdown": model.parameter_summary(),
        "total_trainable_parameters": TOTAL_PARAMETERS,
    }
    arch_path = metrics_dir / "t2_architecture_summary.json"
    arch_path.write_text(json.dumps(arch_summary, indent=2), encoding="utf-8")
    logger.info("Saved architecture and parameter summary (%d total params) to '%s'.", TOTAL_PARAMETERS, arch_path)

    # ------------------------------------------------------------------
    # 2. Phase 3: Central Finite-Difference Gradient Verification
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    X_grad_batch = rng.normal(0.0, 1.0, size=(8, INPUT_DIM)).astype(np.float64)
    Y_grad_batch = np.array([[0.0], [1.0], [0.0], [1.0], [1.0], [0.0], [1.0], [0.0]], dtype=np.float64)

    grad_report = verify_gradients_finite_difference(
        model=model,
        X_batch=X_grad_batch,
        Y_batch=Y_grad_batch,
        epsilon=1e-5,
        tolerance_relative=1e-6,
        tolerance_absolute=1e-8,
    )
    if not grad_report.all_passed:
        raise RuntimeError(
            f"Numerical gradient verification FAILED! Max relative diff: {grad_report.max_relative_difference}"
        )

    grad_json_path = metrics_dir / "t2_gradient_verification.json"
    grad_json_path.write_text(
        json.dumps(dataclasses.asdict(grad_report), indent=2), encoding="utf-8"
    )

    grad_table_rows = [
        {
            "parameter": e.parameter_name,
            "index": str(tuple(e.index)),
            "parameter_value": f"{e.parameter_value:.10e}",
            "analytical_gradient": f"{e.analytical_gradient:.10e}",
            "numerical_gradient": f"{e.numerical_gradient:.10e}",
            "absolute_difference": f"{e.absolute_difference:.10e}",
            "relative_difference": f"{e.relative_difference:.10e}",
            "passed": e.passed,
        }
        for e in grad_report.entries
    ]
    grad_csv_path = metrics_dir / "t2_gradient_verification_table.csv"
    pd.DataFrame(grad_table_rows).to_csv(grad_csv_path, index=False)

    grad_fig_path = plot_gradient_verification(
        grad_report, figures_dir / "fig_t2_gradient_verification.png"
    )
    logger.info(
        "Phase 3 Gradient Verification PASSED (%d/%d checks): max_abs_diff=%.3e, max_rel_diff=%.3e.",
        len(grad_report.entries),
        len(grad_report.entries),
        grad_report.max_absolute_difference,
        grad_report.max_relative_difference,
    )

    # ------------------------------------------------------------------
    # 3. Phase 4: Controlled Numerical Diagnosis (Tiny-Batch Overfitting)
    # ------------------------------------------------------------------
    tiny_report = run_tiny_batch_diagnosis(
        seed=42,
        batch_size=16,
        learning_rate=0.1,
        max_epochs=300,
        target_loss_threshold=0.02,
        data_source="synthetic_diagnostic_batch (fixed 16-sample 80-D batch, seed=42)",
    )
    tiny_json_path = metrics_dir / "t2_tiny_batch_diagnosis.json"
    tiny_json_path.write_text(
        json.dumps(dataclasses.asdict(tiny_report), indent=2), encoding="utf-8"
    )
    tiny_fig_path = plot_tiny_batch_overfitting(
        tiny_report, figures_dir / "fig_t2_tiny_batch_overfitting.png"
    )
    logger.info(
        "Phase 4 Tiny-Batch Diagnosis complete: initial_loss=%.6f -> final_loss=%.6f in %d epochs (acc: %.2f -> %.2f).",
        tiny_report.initial_loss,
        tiny_report.final_loss,
        tiny_report.actual_epochs_completed,
        tiny_report.initial_accuracy,
        tiny_report.final_accuracy,
    )

    # ------------------------------------------------------------------
    # 4. Phase 5: Real-Data Training on MFCC Features (If Dataset Present)
    # ------------------------------------------------------------------
    dataset_report = validate_urbansound8k_dataset(config=cfg, strict=False)
    train_manifest_path = paths["splits_dir"] / "train.csv"
    val_manifest_path = paths["splits_dir"] / "val.csv"
    scaler_path = paths["processed_data_dir"] / "mfcc_scaler.json"

    real_data_status = "BLOCKED_MISSING_URBANSOUND8K_DATASET"
    real_data_results: dict[str, Any] | None = None

    if (
        dataset_report.is_valid
        and train_manifest_path.is_file()
        and val_manifest_path.is_file()
        and scaler_path.is_file()
    ):
        logger.info("Real UrbanSound8K dataset and split manifests detected. Running Phase 5 training...")
        mfcc_cache_path = paths["processed_data_dir"] / "train_val_mfcc_cache.npz"
        if mfcc_cache_path.is_file():
            cached = np.load(mfcc_cache_path)
            X_train = cached["X_train"].astype(np.float64)
            Y_train = cached["y_train"].astype(np.float64).reshape(-1, 1)
            X_val = cached["X_val"].astype(np.float64)
            Y_val = cached["y_val"].astype(np.float64).reshape(-1, 1)
        else:
            scaler = MFCCStandardScaler.load(scaler_path)
            X_train, Y_train = _extract_split_mfcc_matrix(
                train_manifest_path, paths["audio_dir"], scaler
            )
            X_val, Y_val = _extract_split_mfcc_matrix(
                val_manifest_path, paths["audio_dir"], scaler
            )
        train_out = train_numpy_mlp_with_validation(
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            seed=42,
            learning_rate=0.05,
            batch_size=32,
            epochs=80,
        )
        trained_model: NumPyMLP = train_out.pop("model")
        ckpt_path = paths["checkpoints_dir"] / "numpy_mlp_t2_seed42.npz"
        trained_model.save_parameters(ckpt_path, extra_metadata=train_out)

        real_loss_fig = plot_real_data_loss_curves(
            train_out["train_loss_history"],
            train_out["val_loss_history"],
            figures_dir / "fig_t2_real_data_loss_curves.png",
        )
        train_out["checkpoint_path"] = str(ckpt_path)
        train_out["loss_curve_figure"] = str(real_loss_fig)
        (metrics_dir / "t2_real_data_training.json").write_text(
            json.dumps(train_out, indent=2), encoding="utf-8"
        )
        real_data_status = "COMPLETED_ON_REAL_URBANSOUND8K"
        real_data_results = train_out
    else:
        logger.warning(
            "Phase 5 real-data training is BLOCKED because the official UrbanSound8K dataset "
            "is not yet present at '%s'. No real-data checkpoints or curves were fabricated.",
            dataset_report.dataset_root,
        )

    # ------------------------------------------------------------------
    # 5. Phase 6: Experiment Configuration and Environment Metadata
    # ------------------------------------------------------------------
    env_info = get_environment_info(cfg)
    exp_meta = {
        "coursework_stage": "Step 3 — CW1 T2 (NumPy ANN From Scratch)",
        "python_version": env_info.python_version,
        "platform": f"{env_info.platform_system} {env_info.platform_release} ({env_info.platform_machine})",
        "numpy_version": env_info.numpy_version,
        "seed": 42,
        "gradient_verification_passed": grad_report.all_passed,
        "gradient_verification_max_rel_diff": grad_report.max_relative_difference,
        "tiny_batch_loss_reduction_achieved": tiny_report.loss_reduction_achieved,
        "tiny_batch_initial_loss": tiny_report.initial_loss,
        "tiny_batch_final_loss": tiny_report.final_loss,
        "tiny_batch_epochs_completed": tiny_report.actual_epochs_completed,
        "real_data_training_status": real_data_status,
        "artifacts_generated": {
            "architecture_summary": str(arch_path),
            "gradient_verification_json": str(grad_json_path),
            "gradient_verification_csv": str(grad_csv_path),
            "gradient_verification_figure": str(grad_fig_path),
            "tiny_batch_diagnosis_json": str(tiny_json_path),
            "tiny_batch_overfitting_figure": str(tiny_fig_path),
        },
        "data_config_snapshot": data_cfg.get("features", {}).get("mfcc", {}),
    }
    meta_out_path = metrics_dir / "t2_experiment_metadata.json"
    meta_out_path.write_text(json.dumps(exp_meta, indent=2), encoding="utf-8")

    return {
        "architecture_summary": arch_summary,
        "gradient_report": grad_report,
        "tiny_batch_report": tiny_report,
        "real_data_status": real_data_status,
        "real_data_results": real_data_results,
    }


if __name__ == "__main__":
    run_step3_pipeline()

