"""Tests for Phase 3 CNN + GRU Hybrid (CRNN) Spectro-Temporal Architecture.

Module: NB6057CEM — Artificial Neural Networks
Portfolio Upgrade: Phase 3 — Temporal Deep Learning (CNN + GRU Hybrid)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from src.hybrid_model import CNNGRUHybrid
from src.train import load_pytorch_checkpoint


def test_cnn_gru_initialization_and_parameter_count() -> None:
    """Verify CNNGRUHybrid parameter count, architecture dimensions, and summary contract."""
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
        seed=42,
    )
    param_count = model.count_trainable_parameters()
    # 77,601 trainable parameters in the default architecture
    assert 70000 <= param_count <= 85000, f"Expected ~77,601 parameters, got {param_count}"

    summary = model.architecture_summary()
    assert summary["model_id"] == "cnn_gru"
    assert summary["input_shape"] == [1, 128, 173]
    assert summary["conv_channels"] == [16, 32, 64]
    assert summary["gru_config"]["bidirectional"] is True
    assert summary["gru_config"]["output_dim"] == 128
    assert summary["trainable_parameters"] == param_count


def test_cnn_gru_forward_shape_and_intermediate_features() -> None:
    """Verify tensor dimension progression through Conv blocks, GRU, and classification head."""
    model = CNNGRUHybrid(seed=42)
    model.eval()

    batch_size = 3
    dummy_x = torch.randn(batch_size, 1, 128, 173, dtype=torch.float32)

    # 1. Feature extraction check
    with torch.no_grad():
        seq_in, gru_out = model.extract_features(dummy_x)

    # Frequency compressed to 1, time frames preserved (173 -> 86 -> 43)
    assert seq_in.shape == (batch_size, 43, 64)
    # Bidirectional GRU with hidden_size 64 produces 128 features per time step
    assert gru_out.shape == (batch_size, 43, 128)

    # 2. Forward logits check
    with torch.no_grad():
        logits = model(dummy_x)

    assert logits.shape == (batch_size, 1)
    assert torch.all(torch.isfinite(logits))


def test_cnn_gru_input_validation_and_guards() -> None:
    """Verify robust input tensor validation and exception handling."""
    model = CNNGRUHybrid(seed=42)
    model.eval()

    # Non-tensor raises TypeError
    with pytest.raises(TypeError, match="Expected torch.Tensor"):
        model(np.zeros((2, 1, 128, 173)))  # type: ignore

    # Incorrect rank (3D instead of 4D) raises ValueError
    with pytest.raises(ValueError, match="expected 4D tensor"):
        model(torch.randn(1, 128, 173))

    # Incorrect mel channels (e.g. 80 instead of 128)
    with pytest.raises(ValueError, match="expected 4D tensor"):
        model(torch.randn(2, 1, 80, 173))

    # NaN / Inf detection
    nan_tensor = torch.randn(2, 1, 128, 173)
    nan_tensor[0, 0, 10, 10] = float("nan")
    with pytest.raises(ValueError, match="contains NaN or Inf"):
        model(nan_tensor)


def test_cnn_gru_predict_proba_and_predict() -> None:
    """Verify inference prediction and probability calibration bounds."""
    model = CNNGRUHybrid(seed=42)
    model.eval()

    dummy_x = torch.randn(4, 1, 128, 173, dtype=torch.float32)
    probs = model.predict_proba(dummy_x)
    assert probs.shape == (4, 1)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)

    preds = model.predict(dummy_x, threshold=0.5)
    assert preds.shape == (4, 1)
    assert torch.all((preds == 0) | (preds == 1))


def test_cnn_gru_deterministic_seeding() -> None:
    """Verify deterministic weight initialization under identical random seeds."""
    m1 = CNNGRUHybrid(seed=12345)
    m2 = CNNGRUHybrid(seed=12345)

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        torch.testing.assert_close(p1, p2)


def test_checkpoint_serialization_and_loader() -> None:
    """Verify that load_pytorch_checkpoint correctly reconstructs CNNGRUHybrid."""
    model = CNNGRUHybrid(seed=42)
    model.eval()

    dummy_path = Path("checkpoints/test_temp_cnn_gru.pt")
    dummy_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = {
            "model_id": "cnn_gru",
            "framework": "pytorch",
            "architecture_config": model.architecture_summary(),
            "trainable_parameters": model.count_trainable_parameters(),
            "state_dict": model.state_dict(),
            "training_metadata": {"model_name": "CNN + GRU Hybrid (CRNN)", "seed": 42},
        }
        torch.save(payload, dummy_path)

        loaded_model, loaded_payload = load_pytorch_checkpoint(dummy_path, device="cpu")
        assert isinstance(loaded_model, CNNGRUHybrid)
        assert loaded_payload["model_id"] == "cnn_gru"

        dummy_x = torch.zeros(1, 1, 128, 173, dtype=torch.float32)
        with torch.no_grad():
            out1 = model(dummy_x)
            out2 = loaded_model(dummy_x)
        torch.testing.assert_close(out1, out2)
    finally:
        if dummy_path.is_file():
            dummy_path.unlink()

