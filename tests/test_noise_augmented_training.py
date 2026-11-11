"""Tests for Phase 2 Noise-Augmented Deep MLP Training & Evaluation Pipeline.

Module: NB6057CEM — Artificial Neural Networks
Portfolio Upgrade: Phase 2 — Noise-Augmented Audio Training
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.audio_features import MFCCStandardScaler, extract_mfcc_features
from src.augmentation import (
    AugmentationPolicyError,
    TrainingAudioAugmenter,
    add_controlled_background_noise,
    apply_amplitude_scaling,
    apply_time_shift,
)
from src.models import DeepMLP


def test_augmentation_policy_enforcement() -> None:
    """Verify strict leak-prevention: stochastic augmentation forbidden on val/test."""
    augmenter = TrainingAudioAugmenter(enabled=True, seed=42)
    dummy_wav = np.zeros(88200, dtype=np.float32)

    # Permitted on train
    augmented_train = augmenter(dummy_wav, split="train", sample_index=0)
    assert isinstance(augmented_train, np.ndarray)

    # Strictly forbidden on validation
    with pytest.raises(AugmentationPolicyError, match="forbidden on split='val'"):
        augmenter(dummy_wav, split="val", sample_index=0)

    # Strictly forbidden on test
    with pytest.raises(AugmentationPolicyError, match="forbidden on split='test'"):
        augmenter(dummy_wav, split="test", sample_index=0)


def test_controlled_noise_injection_properties() -> None:
    """Verify mathematical properties of controlled background noise injection."""
    rng = np.random.default_rng(42)
    t = np.linspace(0.0, 4.0, 88200, dtype=np.float32)
    clean_sig = 0.5 * np.sin(2 * np.pi * 1000 * t)

    # 1. Shape and range preservation
    noisy_sig = add_controlled_background_noise(clean_sig, snr_db=15.0, rng=rng)
    assert noisy_sig.shape == clean_sig.shape
    assert noisy_sig.dtype == np.float32
    assert np.all(noisy_sig >= -1.0) and np.all(noisy_sig <= 1.0)

    # 2. Perturbation occurred
    diff = np.abs(noisy_sig - clean_sig)
    assert np.mean(diff) > 0.001

    # 3. Determinism with identical seed
    rng1 = np.random.default_rng(12345)
    rng2 = np.random.default_rng(12345)
    out1 = add_controlled_background_noise(clean_sig, snr_db=20.0, rng=rng1)
    out2 = add_controlled_background_noise(clean_sig, snr_db=20.0, rng=rng2)
    np.testing.assert_allclose(out1, out2, rtol=1e-6)

    # 4. Too severe SNR raises AugmentationPolicyError
    with pytest.raises(AugmentationPolicyError, match="too low to guarantee label preservation"):
        add_controlled_background_noise(clean_sig, snr_db=-10.0, rng=rng)


def test_recomputed_mfcc_from_augmented_waveform() -> None:
    """Verify 80-D MFCC feature recomputation directly from perturbed audio."""
    rng = np.random.default_rng(42)
    t = np.linspace(0.0, 4.0, 88200, dtype=np.float32)
    clean_wav = 0.4 * np.sin(2 * np.pi * 800 * t)

    # Inject noise into waveform
    noisy_wav = add_controlled_background_noise(clean_wav, snr_db=10.0, rng=rng)

    # Recompute MFCCs from both
    clean_mfcc = extract_mfcc_features(clean_wav, sample_rate=22050, n_mfcc=40)
    noisy_mfcc = extract_mfcc_features(noisy_wav, sample_rate=22050, n_mfcc=40)

    assert clean_mfcc.shape == (80,)
    assert noisy_mfcc.shape == (80,)
    assert np.all(np.isfinite(noisy_mfcc))

    # Features must differ due to audio-level perturbation
    assert not np.allclose(clean_mfcc, noisy_mfcc, rtol=1e-3)


def test_deep_mlp_augmented_forward_pass() -> None:
    """Verify DeepMLP operates seamlessly on 80-D MFCC features."""
    model = DeepMLP(input_dim=80, hidden_dims=(128, 64), output_dim=1, dropout=0.3, seed=42)
    model.eval()

    dummy_input = torch.randn(4, 80, dtype=torch.float32)
    with torch.no_grad():
        logits = model(dummy_input)
        probs = torch.sigmoid(logits)

    assert logits.shape == (4, 1)
    assert probs.shape == (4, 1)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


def test_saved_augmented_checkpoint_loading_and_inference() -> None:
    """Verify saved deep_mlp_augmented_seed42.pt checkpoint structure and weights."""
    ckpt_path = Path("checkpoints/deep_mlp_augmented_seed42.pt")
    if not ckpt_path.is_file():
        pytest.skip("deep_mlp_augmented_seed42.pt checkpoint not present in sandbox.")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "state_dict" in ckpt
    assert "architecture_config" in ckpt
    assert "training_metadata" in ckpt

    meta = ckpt["training_metadata"]
    assert "best_val_macro_f1" in meta
    assert "best_epoch" in meta

    model = DeepMLP(input_dim=80, hidden_dims=(128, 64), output_dim=1, dropout=0.3, seed=42)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dummy_x = torch.zeros(1, 80, dtype=torch.float32)
    with torch.no_grad():
        out = model(dummy_x)
    assert out.shape == (1, 1)


def test_inference_engine_includes_augmented_deep_mlp() -> None:
    """Verify SirenInferenceEngine loads and compares deep_mlp_augmented."""
    from src.inference import SirenInferenceEngine

    engine = SirenInferenceEngine()
    sample_wav = Path("demo/sample_audio/siren_sample_165166-8-0-0.wav")
    assert sample_wav.is_file(), f"Sample audio missing at {sample_wav}"

    comparison = engine.compare_all_models(sample_wav)

    assert "models" in comparison
    models = comparison["models"]
    assert "deep_mlp" in models
    assert "deep_mlp_augmented" in models
    aug_res = models["deep_mlp_augmented"]
    assert "probability" in aug_res
    assert 0.0 <= aug_res["probability"] <= 1.0
    assert "predicted_class" in aug_res
    assert "is_siren" in aug_res
    assert aug_res["model_name"] == "Deep MLP (Noise-Augmented, Phase 2)"


def test_phase2_metrics_artifacts_integrity() -> None:
    """Verify that generated Phase 2 metrics CSV and JSON summary are complete and valid."""
    csv_path = Path("results/metrics/phase2_clean_vs_augmented_comparison.csv")
    json_path = Path("results/metrics/phase2_comparison_summary.json")

    if not csv_path.is_file() or not json_path.is_file():
        pytest.skip("Phase 2 metric artifacts not yet generated.")

    df = pd.read_csv(csv_path)
    expected_cols = {"condition", "model", "macro_f1", "siren_recall", "siren_precision", "pr_auc", "inference_latency_ms"}
    assert expected_cols.issubset(set(df.columns))

    conditions = set(df["condition"].unique())
    assert {"Clean", "20 dB", "10 dB", "5 dB"}.issubset(conditions)

    # Check metrics boundaries
    assert (df["macro_f1"] >= 0.0).all() and (df["macro_f1"] <= 1.0).all()
    assert (df["siren_recall"] >= 0.0).all() and (df["siren_recall"] <= 1.0).all()
    assert (df["siren_precision"] >= 0.0).all() and (df["siren_precision"] <= 1.0).all()
    assert (df["pr_auc"] >= 0.0).all() and (df["pr_auc"] <= 1.0).all()

