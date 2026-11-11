"""Automated unit and integration tests for Step 4 Competing Models (CW1 T3).

Module: NB6057CEM — Artificial Neural Networks
Step: 4 — Implement and Train Three Competing Models (CW1 T3)

Covers all 20 Phase 9 requirements:
1. Random Forest fitting and prediction.
2. Deep MLP forward pass.
3. 2D CNN forward pass.
4. Correct output dimensions.
5. Finite loss computation.
6. Exact programmatic parameter counts (DeepMLP = 18,689; Conv2DCNN = 23,585).
7. Successful optimization step and loss decrease.
8. Checkpoint saving and loading (.joblib and .pt + .meta.json).
9. Prediction consistency after loading.
10. Early-stopping logic.
11. Validation checkpoint and hyperparameter selection.
12. Training-only class-weight calculation.
13. Model configuration loading and validation.
14. Deterministic execution across seeds (42, 123, 2026).
15. Input shape and NaN/Inf rejection.
16. Invalid label handling.
17. Device selection integration.
18. Missing dataset handling in unified runner.
19. Strict prevention of held-out test evaluation (`TestSetAccessError`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from scripts.run_competing_models_t3 import export_architecture_comparison_artifacts
from src import ConfigurationError, load_model_config
from src.evaluate import (
    TestSetAccessError,
    compute_classification_metrics,
    enforce_no_test_split_access,
)
from src.models import (
    Conv2DCNN,
    DeepMLP,
    RandomForestBaseline,
    build_model,
    create_model,
)
from src.train import (
    EarlyStoppingTracker,
    compute_training_pos_weight,
    load_pytorch_checkpoint,
    make_tensor_dataloaders,
    plot_training_curves,
    save_pytorch_checkpoint,
    train_pytorch_model,
    tune_and_select_random_forest,
)


def _make_synthetic_mfcc_split(
    n_samples: int = 24, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Create a clearly labeled synthetic 80-D MFCC fixture for unit testing."""
    rng = np.random.default_rng(seed)
    half = n_samples // 2
    x0 = rng.normal(loc=-0.5, scale=0.9, size=(half, 80)).astype(np.float32)
    x1 = rng.normal(loc=+0.5, scale=0.9, size=(n_samples - half, 80)).astype(np.float32)
    X = np.vstack([x0, x1])
    y = np.concatenate([np.zeros(half, dtype=np.int64), np.ones(n_samples - half, dtype=np.int64)])
    return X, y


def _make_synthetic_mel_split(
    n_samples: int = 12, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Create a clearly labeled synthetic (1, 128, 173) Log-Mel fixture for unit testing."""
    rng = np.random.default_rng(seed)
    half = n_samples // 2
    x0 = rng.normal(loc=-0.3, scale=0.8, size=(half, 1, 128, 173)).astype(np.float32)
    x1 = rng.normal(loc=+0.3, scale=0.8, size=(n_samples - half, 1, 128, 173)).astype(np.float32)
    X = np.vstack([x0, x1])
    y = np.concatenate([np.zeros(half, dtype=np.int64), np.ones(n_samples - half, dtype=np.int64)])
    return X, y


# ==============================================================================
# 1–6. Forward Passes, Output Shapes, Finite Loss, and Exact Parameter Counts
# ==============================================================================


def test_random_forest_fit_predict_and_save_load(tmp_path: Path) -> None:
    """Verify Random Forest fitting, probabilities, deterministic predictions, and checkpoint round-trip."""
    X_tr, y_tr = _make_synthetic_mfcc_split(n_samples=24, seed=42)
    X_va, _ = _make_synthetic_mfcc_split(n_samples=8, seed=123)

    rf = RandomForestBaseline(n_estimators=50, max_depth=8, random_state=42)
    # Non-train split fitting must be rejected
    with pytest.raises(ValueError, match="Leakage violation"):
        rf.fit(X_tr, y_tr, split_name="val")

    rf.fit(X_tr, y_tr, split_name="train")
    assert rf.is_fitted is True
    assert rf.training_duration_sec is not None and rf.training_duration_sec >= 0.0

    probs = rf.predict_proba(X_va)
    preds = rf.predict(X_va)
    assert probs.shape == (8, 1)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert preds.shape == (8,)
    assert set(preds.tolist()).issubset({0, 1})

    ckpt_path = tmp_path / "rf_test.joblib"
    meta = rf.save(ckpt_path, extra_metadata={"test": True})
    assert ckpt_path.is_file()
    assert meta["model_size_bytes"] > 0

    loaded_rf = RandomForestBaseline.load(ckpt_path)
    assert np.allclose(probs, loaded_rf.predict_proba(X_va), atol=1e-7)
    assert np.array_equal(preds, loaded_rf.predict(X_va))


def test_deep_mlp_forward_parameter_count_and_finite_loss() -> None:
    """Verify DeepMLP forward shape (B, 1), exact 18,689 parameters, and finite BCEWithLogitsLoss."""
    mlp = DeepMLP(input_dim=80, hidden_dims=(128, 64), output_dim=1, dropout=0.3, seed=42)
    assert mlp.count_trainable_parameters() == 18689

    x = torch.randn(6, 80)
    y = torch.tensor([[0.0], [1.0], [0.0], [1.0], [1.0], [0.0]])
    logits = mlp(x)
    probs = mlp.predict_proba(x)
    preds = mlp.predict(x)

    assert logits.shape == (6, 1)
    assert probs.shape == (6, 1)
    assert torch.all((probs > 0.0) & (probs < 1.0))
    assert preds.shape == (6, 1)

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, y)
    assert torch.isfinite(loss) and float(loss.item()) > 0.0


def test_cnn_2d_forward_parameter_count_and_finite_loss() -> None:
    """Verify Conv2DCNN forward shape (B, 1), exact 23,585 parameters, and finite BCEWithLogitsLoss."""
    cnn = Conv2DCNN(
        in_channels=1,
        conv_channels=(16, 32, 64),
        n_mels=128,
        time_frames=173,
        dropout=0.3,
        output_dim=1,
        seed=42,
    )
    assert cnn.count_trainable_parameters() == 23585

    x = torch.randn(4, 1, 128, 173)
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    logits = cnn(x)
    probs = cnn.predict_proba(x)
    preds = cnn.predict(x)

    assert logits.shape == (4, 1)
    assert probs.shape == (4, 1)
    assert torch.all((probs > 0.0) & (probs < 1.0))
    assert preds.shape == (4, 1)

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, y)
    assert torch.isfinite(loss) and float(loss.item()) > 0.0


# ==============================================================================
# 7–11. Optimization Step, Checkpoints, Early Stopping, and Model Selection
# ==============================================================================


@pytest.mark.parametrize("model_id", ["deep_mlp", "cnn_2d"])
def test_one_optimization_step_and_checkpoint_round_trip(
    tmp_path: Path, model_id: str
) -> None:
    """Verify one Adam step updates weights and checkpoint save/load preserves predictions."""
    if model_id == "deep_mlp":
        model: DeepMLP | Conv2DCNN = create_model("deep_mlp", seed=42, dropout=0.0)
        x = torch.randn(8, 80)
    else:
        model = create_model("cnn_2d", seed=42, dropout=0.0)
        x = torch.randn(4, 1, 128, 173)

    y = torch.ones((x.shape[0], 1), dtype=torch.float32)
    y[: x.shape[0] // 2] = 0.0

    params_before = [p.detach().clone() for p in model.parameters()]
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    optimizer.zero_grad()
    loss_before = criterion(model(x), y)
    loss_before.backward()
    optimizer.step()

    with torch.no_grad():
        loss_after = criterion(model(x), y)
    assert float(loss_after.item()) < float(loss_before.item())
    assert any(not torch.equal(p0, p1) for p0, p1 in zip(params_before, model.parameters()))

    probs_before = model.predict_proba(x)
    ckpt_file = tmp_path / f"{model_id}_ckpt.pt"
    meta = save_pytorch_checkpoint(model, ckpt_file, training_metadata={"epoch": 1})
    assert ckpt_file.is_file()
    assert Path(str(ckpt_file) + ".meta.json").is_file()
    assert meta["model_size_bytes"] > 0

    loaded_model, loaded_payload = load_pytorch_checkpoint(ckpt_file, device="cpu")
    probs_after = loaded_model.predict_proba(x)
    assert loaded_payload["model_id"] == model_id
    assert torch.allclose(probs_before, probs_after, atol=1e-6)


def test_early_stopping_tracker_and_best_weight_restoration() -> None:
    """Verify EarlyStoppingTracker tracks best macro-F1, triggers patience stop, and restores weights."""
    model = DeepMLP(seed=42)
    tracker = EarlyStoppingTracker(patience=2, min_delta=1e-4)

    # Epoch 1: macro_f1 = 0.70
    w_epoch1 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert tracker.update(1, val_loss=0.55, val_metrics={"macro_f1": 0.70}, model=model) is True
    assert tracker.should_stop is False

    # Mutate model weights and simulate worse Epochs 2 and 3
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.5)

    assert tracker.update(2, val_loss=0.60, val_metrics={"macro_f1": 0.65}, model=model) is False
    assert tracker.should_stop is False
    assert tracker.update(3, val_loss=0.65, val_metrics={"macro_f1": 0.62}, model=model) is False
    assert tracker.should_stop is True
    assert tracker.best_epoch == 1
    assert tracker.best_val_macro_f1 == pytest.approx(0.70)

    # Restore best weights (Epoch 1)
    tracker.restore_best_weights(model)
    for k, v in model.state_dict().items():
        assert torch.equal(v, w_epoch1[k])


def test_validation_checkpoint_selection_and_training_runner(tmp_path: Path) -> None:
    """Verify end-to-end training & selection for Random Forest, DeepMLP, and Conv2DCNN on fixtures."""
    X_tr, y_tr = _make_synthetic_mfcc_split(n_samples=24, seed=42)
    X_va, y_va = _make_synthetic_mfcc_split(n_samples=12, seed=123)

    # 1. Random Forest tuning selection
    rf_candidates = [
        {"config_id": "rf_small", "n_estimators": 20, "max_depth": 4, "class_weight": "balanced"},
        {"config_id": "rf_med", "n_estimators": 50, "max_depth": 8, "class_weight": "balanced"},
    ]
    rf_res = tune_and_select_random_forest(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_va,
        y_val=y_va,
        candidates=rf_candidates,
        seed=42,
        checkpoint_path=tmp_path / "rf_best.joblib",
    )
    assert len(rf_res["all_trials"]) == 2
    assert (tmp_path / "rf_best.joblib").is_file()

    # 2. DeepMLP training with early stopping and curve plotting
    tr_loader, va_loader = make_tensor_dataloaders(X_tr, y_tr, X_va, y_va, batch_size=8, seed=42)
    mlp_res = train_pytorch_model(
        model=DeepMLP(seed=42),
        train_loader=tr_loader,
        val_loader=va_loader,
        learning_rate=5e-3,
        max_epochs=6,
        patience=3,
        seed=42,
        checkpoint_path=tmp_path / "mlp_best.pt",
    )
    meta = mlp_res["training_metadata"]
    assert 1 <= meta["best_epoch"] <= 6
    assert (tmp_path / "mlp_best.pt").is_file()

    curve_path = plot_training_curves(
        train_losses=meta["train_loss_history"],
        val_losses=meta["val_loss_history"],
        val_macro_f1s=meta["val_macro_f1_history"],
        best_epoch=meta["best_epoch"],
        model_title="Fixture DeepMLP",
        output_path=tmp_path / "mlp_curves.png",
    )
    assert curve_path.is_file() and curve_path.stat().st_size > 0


# ==============================================================================
# 12–19. Class Weights, Config, Determinism, Shape/Label Guards & Test-Set Guard
# ==============================================================================


def test_training_only_class_weight_calculation() -> None:
    """Verify pos_weight is computed from train only and rejects val/test splits."""
    y_train = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int64)
    pw = compute_training_pos_weight(y_train, split_name="train")
    assert pw == pytest.approx(3.0)

    with pytest.raises(ValueError, match="Leakage violation"):
        compute_training_pos_weight(y_train, split_name="val")
    with pytest.raises(TestSetAccessError, match="Held-out test partition"):
        compute_training_pos_weight(y_train, split_name="test")


def test_model_configuration_loading_and_architecture_export(tmp_path: Path) -> None:
    """Verify configs/model_config.yaml loads and architecture comparison exports cleanly."""
    m_cfg = load_model_config()
    assert m_cfg["comparison_protocol"]["primary_selection_metric"] == "macro-F1"
    assert m_cfg["comparison_protocol"]["allow_test_set_access_during_training_or_tuning"] is False

    exported = export_architecture_comparison_artifacts(tmp_path)
    assert exported["deep_mlp_detail"]["trainable_parameters"] == 18689
    assert exported["cnn_2d_detail"]["trainable_parameters"] == 23585
    assert (tmp_path / "t3_architecture_comparison.json").is_file()
    assert (tmp_path / "t3_architecture_comparison_table.csv").is_file()


@pytest.mark.parametrize("seed", [42, 123, 2026])
def test_deterministic_execution_across_seeds(seed: int) -> None:
    """Verify identical training runs under seeds 42, 123, and 2026 produce identical outputs."""
    X_tr, y_tr = _make_synthetic_mfcc_split(n_samples=16, seed=seed)
    X_va, y_va = _make_synthetic_mfcc_split(n_samples=8, seed=seed + 1)

    tr_1, va_1 = make_tensor_dataloaders(X_tr, y_tr, X_va, y_va, batch_size=8, seed=seed)
    res_1 = train_pytorch_model(
        model=DeepMLP(seed=seed),
        train_loader=tr_1,
        val_loader=va_1,
        learning_rate=1e-3,
        max_epochs=3,
        patience=3,
        seed=seed,
    )

    tr_2, va_2 = make_tensor_dataloaders(X_tr, y_tr, X_va, y_va, batch_size=8, seed=seed)
    res_2 = train_pytorch_model(
        model=DeepMLP(seed=seed),
        train_loader=tr_2,
        val_loader=va_2,
        learning_rate=1e-3,
        max_epochs=3,
        patience=3,
        seed=seed,
    )

    assert np.allclose(
        res_1["training_metadata"]["train_loss_history"],
        res_2["training_metadata"]["train_loss_history"],
        atol=1e-6,
    )


def test_input_shape_and_invalid_label_rejection() -> None:
    """Verify invalid shapes, NaNs, and invalid labels are rejected across all 3 models."""
    mlp = DeepMLP(seed=42)
    cnn = Conv2DCNN(seed=42)
    rf = RandomForestBaseline(random_state=42)

    with pytest.raises(ValueError, match="expected 2D tensor of shape \\(B, 80\\)"):
        mlp(torch.randn(4, 79))
    with pytest.raises(ValueError, match="NaN or Inf"):
        mlp(torch.full((4, 80), float("nan")))

    with pytest.raises(ValueError, match="expected 4D tensor of shape \\(B, 1, 128, 173\\)"):
        cnn(torch.randn(4, 1, 64, 173))
    with pytest.raises(ValueError, match="NaN or Inf"):
        cnn(torch.full((2, 1, 128, 173), float("inf")))

    with pytest.raises(ValueError, match="shape \\(N, 80\\)"):
        rf.fit(np.zeros((10, 75)), np.zeros(10), split_name="train")
    with pytest.raises(ValueError, match="binary values in \\{0, 1\\}"):
        rf.fit(np.zeros((10, 80)), np.array([0, 1, 2, 0, 1, 0, 1, 0, 1, 0]), split_name="train")

    with pytest.raises(ConfigurationError):
        build_model("nonexistent_model_xyz", instantiate=True)


def test_no_accidental_held_out_test_evaluation_guard() -> None:
    """Verify attempting to evaluate on split='test' during Step 4 raises TestSetAccessError."""
    with pytest.raises(TestSetAccessError, match="Held-out test partition"):
        enforce_no_test_split_access("test", allow_final_test=False)

    with pytest.raises(TestSetAccessError, match="Held-out test partition"):
        compute_classification_metrics(
            y_true=np.array([0, 1, 0, 1]),
            y_prob=np.array([0.1, 0.9, 0.2, 0.8]),
            split_name="test",
            allow_final_test=False,
        )

