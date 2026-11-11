"""Controlled Audio Noise Robustness Experiment — Step 5 (CW1 T4d).

Module: NB6057CEM — Artificial Neural Networks
Step: 5 — Complete CW1 T4: Controlled Experiments and Model Evaluation (14 marks)

Protocol:
1. Evaluation subset: Strictly the 837 held-out test audio clips (Fold 10).
2. Noise conditions:
   - Clean (unperturbed, canonical test audio)
   - 20 dB SNR (moderate urban ambient noise)
   - 10 dB SNR (heavy street / traffic noise)
   - 5 dB SNR (extreme severe acoustic interference)
3. Noise injection: Additive Gaussian background noise generated via `add_controlled_background_noise`
   with a fixed random seed (seed = 42).
4. Feature extraction: Frozen original preprocessing pipeline using training-only fitted
   `MFCCStandardScaler` and `MelSpectrogramNormalizer`. No normalization stats are refitted.
5. Evaluation: Evaluate all trained checkpoints on the perturbed feature sets.
6. Record metrics: Macro-F1, siren recall, siren precision, balanced accuracy, and degradation curve.

Outputs:
- `results/metrics/t4_noise_robustness.csv`
- `results/figures/t4_noise_robustness.png`

Usage:
    python scripts/run_t4_robustness.py
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
from src.augmentation import add_controlled_background_noise
from src.evaluate import compute_classification_metrics
from src.models import (
    Conv2DCNN,
    DeepMLP,
    RandomForestBaseline,
)
from src.numpy_mlp import NumPyMLP
from src.train import load_pytorch_checkpoint


def plot_noise_robustness(
    df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot performance degradation curves across SNR noise conditions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=150)

    snr_order = ["Clean", "20 dB", "10 dB", "5 dB"]
    colors = {
        "Random Forest": "#2ca02c",
        "Deep MLP": "#1f77b4",
        "2D CNN": "#d62728",
        "NumPy MLP": "#ff7f0e",
    }
    markers = {
        "Random Forest": "s",
        "Deep MLP": "o",
        "2D CNN": "^",
        "NumPy MLP": "D",
    }

    for model_name, grp in df.groupby("model_name"):
        grp_sorted = grp.set_index("snr_condition").reindex(snr_order).reset_index()
        ax1.plot(
            grp_sorted["snr_condition"],
            grp_sorted["macro_f1"],
            label=model_name,
            color=colors.get(model_name, "black"),
            marker=markers.get(model_name, "o"),
            linewidth=2.0,
            markersize=7,
        )
        ax2.plot(
            grp_sorted["snr_condition"],
            grp_sorted["recall_siren"],
            label=model_name,
            color=colors.get(model_name, "black"),
            marker=markers.get(model_name, "o"),
            linewidth=2.0,
            markersize=7,
        )

    ax1.set_xlabel("Noise Condition (Additive Gaussian Perturbation)", fontsize=10)
    ax1.set_ylabel("Macro-F1 Score", fontsize=10)
    ax1.set_title("Overall Robustness: Macro-F1 vs. Noise Severity", fontsize=11, fontweight="bold")
    ax1.set_ylim(0.40, 1.02)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower left", frameon=True)

    ax2.set_xlabel("Noise Condition (Additive Gaussian Perturbation)", fontsize=10)
    ax2.set_ylabel("Siren Recall (True Positive Rate)", fontsize=10)
    ax2.set_title("Safety-Critical Metric: Siren Recall vs. Noise Severity", fontsize=11, fontweight="bold")
    ax2.set_ylim(0.40, 1.02)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="lower left", frameon=True)

    fig.suptitle("CW1 T4(d) — Empirical Noise Robustness Degradation on Held-Out Test Set (Fold 10, N=837)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def run_noise_robustness_experiment(seed: int = 42) -> dict[str, Any]:
    """Execute evaluation across Clean, 20 dB, 10 dB, and 5 dB SNR conditions on held-out test data."""
    logger = setup_logger("t4_robustness")
    cfg = load_config()
    data_cfg = load_data_config()
    paths = resolve_project_paths(cfg)
    metrics_dir = paths["metrics_dir"]
    figures_dir = paths["figures_dir"]
    checkpoints_dir = paths["checkpoints_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    test_csv = paths["splits_dir"] / "test.csv"
    if not test_csv.is_file():
        raise FileNotFoundError(f"Held-out test manifest not found at '{test_csv}'.")

    df_test = pd.read_csv(test_csv)
    y_test = df_test["binary_label"].to_numpy(dtype=np.int64)

    # Load frozen scalers
    mfcc_scaler = MFCCStandardScaler.load(paths["processed_data_dir"] / "mfcc_scaler.json")
    mel_normalizer = MelSpectrogramNormalizer.load(paths["processed_data_dir"] / "mel_norm_stats.json")

    # Load trained models (seed 42 checkpoints)
    rf_ckpt = checkpoints_dir / f"random_forest_t3_seed{seed}.joblib"
    mlp_ckpt = checkpoints_dir / f"deep_mlp_t3_seed{seed}.pt"
    cnn_ckpt = checkpoints_dir / f"cnn_2d_t3_seed{seed}.pt"
    np_ckpt = checkpoints_dir / f"numpy_mlp_t2_seed{seed}.npz"

    rf_model = RandomForestBaseline.load(rf_ckpt)
    mlp_model, _ = load_pytorch_checkpoint(mlp_ckpt, device="cpu")
    cnn_model, _ = load_pytorch_checkpoint(cnn_ckpt, device="cpu")
    np_model = NumPyMLP.load_parameters(np_ckpt)

    audio_cfg = data_cfg["audio_signal"]
    mfcc_cfg = data_cfg["features"]["mfcc"]
    mel_cfg = data_cfg["features"]["mel_spectrogram"]
    target_sr = int(audio_cfg.get("target_sample_rate", 22050))
    target_dur = float(audio_cfg.get("target_duration_sec", 4.0))

    # Preload unperturbed waveforms for held-out test set
    logger.info("Preloading %d test audio waveforms for controlled perturbation...", len(df_test))
    clean_waveforms: list[np.ndarray] = []
    for row in df_test.itertuples(index=False):
        wav_path = paths["audio_dir"] / f"fold{int(row.fold)}" / str(row.slice_file_name)
        waveform = load_and_standardize_audio(
            wav_path, target_sr=target_sr, target_duration_sec=target_dur
        )
        clean_waveforms.append(waveform)

    snr_conditions = [
        ("Clean", None),
        ("20 dB", 20.0),
        ("10 dB", 10.0),
        ("5 dB", 5.0),
    ]

    robustness_records: list[dict[str, Any]] = []

    for label, snr_val in snr_conditions:
        logger.info("Extracting features under condition: %s (SNR=%s)", label, snr_val)
        mfcc_feats: list[np.ndarray] = []
        mel_feats: list[np.ndarray] = []

        rng_noise = np.random.default_rng(seed)

        for w in clean_waveforms:
            if snr_val is not None:
                # Add controlled noise with fixed seed RNG
                w_perturbed = add_controlled_background_noise(
                    waveform=w,
                    snr_db=float(snr_val),
                    rng=rng_noise,
                )
            else:
                w_perturbed = w

            m_vec = extract_mfcc_features(
                w_perturbed,
                sample_rate=target_sr,
                n_mfcc=int(mfcc_cfg.get("n_mfcc", 40)),
                n_fft=int(mfcc_cfg.get("n_fft", 2048)),
                hop_length=int(mfcc_cfg.get("hop_length", 512)),
                win_length=int(mfcc_cfg.get("win_length", 2048)),
                n_mels=int(mfcc_cfg.get("n_mels", 128)),
                f_min=float(mfcc_cfg.get("f_min", 0.0)),
                f_max=float(mfcc_cfg.get("f_max", 11025.0)),
            )
            l_mel = extract_log_mel_spectrogram(
                w_perturbed,
                sample_rate=target_sr,
                n_mels=int(mel_cfg.get("n_mels", 128)),
                n_fft=int(mel_cfg.get("n_fft", 2048)),
                hop_length=int(mel_cfg.get("hop_length", 512)),
                win_length=int(mel_cfg.get("win_length", 2048)),
                f_min=float(mel_cfg.get("f_min", 0.0)),
                f_max=float(mel_cfg.get("f_max", 11025.0)),
                top_db=float(mel_cfg.get("top_db", 80.0)),
            )
            mfcc_feats.append(m_vec)
            mel_feats.append(l_mel)

        X_mfcc = mfcc_scaler.transform(np.stack(mfcc_feats, axis=0)).astype(np.float32)
        X_mel = mel_normalizer.transform(np.stack(mel_feats, axis=0)).astype(np.float32)

        # Evaluate Models
        models_to_test = [
            ("Random Forest", "rf", rf_model, X_mfcc),
            ("Deep MLP", "deep_mlp", mlp_model, X_mfcc),
            ("2D CNN", "cnn_2d", cnn_model, X_mel),
            ("NumPy MLP", "numpy_mlp", np_model, X_mfcc),
        ]

        for m_name, m_type, m_obj, feats in models_to_test:
            if m_type == "rf":
                probs = m_obj.predict_proba(feats)[:, 1] if m_obj.predict_proba(feats).shape[1] > 1 else m_obj.predict_proba(feats).ravel()
            elif m_type in ("deep_mlp", "cnn_2d"):
                m_obj.eval()
                with torch.no_grad():
                    out_probs = []
                    bsize = 32
                    for i in range(0, len(feats), bsize):
                        chunk = torch.from_numpy(feats[i : i + bsize]).to("cpu")
                        p = torch.sigmoid(m_obj(chunk)).squeeze(-1).cpu().numpy()
                        out_probs.append(p.reshape(-1))
                    probs = np.concatenate(out_probs, axis=0).astype(np.float64)
            elif m_type == "numpy_mlp":
                probs = m_obj.predict_proba(feats.astype(np.float64)).ravel().astype(np.float64)

            metrics = compute_classification_metrics(
                y_true=y_test,
                y_prob=probs,
                split_name="test",
                allow_final_test=True,
            )

            robustness_records.append({
                "snr_condition": label,
                "snr_db": snr_val if snr_val is not None else float("inf"),
                "model_name": m_name,
                "model_type": m_type,
                "num_samples": len(y_test),
                "macro_f1": metrics["macro_f1"],
                "f1_siren": metrics["f1_siren"],
                "f1_non_siren": metrics["f1_non_siren"],
                "recall_siren": metrics["recall_siren"],
                "precision_siren": metrics["precision_siren"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "accuracy": metrics["accuracy"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
            })
            logger.info("  [%s | %s] Macro-F1=%.4f, Siren Recall=%.4f, Siren Precision=%.4f", label, m_name, metrics["macro_f1"], metrics["recall_siren"], metrics["precision_siren"])

    df_rob = pd.DataFrame(robustness_records)
    csv_path = metrics_dir / "t4_noise_robustness.csv"
    df_rob.to_csv(csv_path, index=False)
    logger.info("Saved noise robustness metrics table to '%s'.", csv_path)

    fig_path = plot_noise_robustness(df_rob, figures_dir / "t4_noise_robustness.png")
    logger.info("Saved noise robustness degradation figure to '%s'.", fig_path)

    return {"records": robustness_records, "csv": str(csv_path), "figure": str(fig_path)}


if __name__ == "__main__":
    run_noise_robustness_experiment()

