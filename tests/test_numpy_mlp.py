"""Automated unit tests for the First-Principles NumPy MLP (CW1 T2).

Module: NB6057CEM — Artificial Neural Networks
Step: 3 — Complete Real-Data Preparation and Implement NumPy ANN From Scratch

Covers all 13 Phase 7 requirements:
1. Weight and bias shapes (W1: (80, 32), b1: (32,), W2: (32, 1), b2: (1,)).
2. Forward output shapes.
3. Valid probability bounds and numerical stability for extreme logits.
4. Finite scalar Binary Cross-Entropy loss.
5. Analytical gradient shapes (dW1, db1, dW2, db2).
6. Central finite-difference gradient verification (float64, epsilon=1e-5).
7. Parameter updates via mini-batch gradient descent.
8. Reproducibility with fixed seeds (42, 123, 2026).
9. Tiny-batch overfitting loss reduction.
10. Invalid input dimension and label rejection.
11. NaN and Inf input rejection.
12. Parameter save/load round trip (.npz and .json).
13. Prediction consistency after loading.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.numpy_mlp import (
    HIDDEN_DIM,
    INPUT_DIM,
    NumPyMLP,
    OUTPUT_DIM,
    TOTAL_PARAMETERS,
    compute_bce_loss,
    compute_binary_classification_metrics,
    run_tiny_batch_diagnosis,
    stable_sigmoid,
    train_numpy_mlp_with_validation,
    verify_gradients_finite_difference,
)


def _make_test_batch(n: int = 10, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, INPUT_DIM)).astype(np.float64)
    Y = (rng.random(size=(n, 1)) >= 0.5).astype(np.float64)
    return X, Y


# ==============================================================================
# 1–5. Shapes, Probability Bounds, Finite Loss, and Analytical Gradient Shapes
# ==============================================================================


def test_weight_and_bias_shapes_and_parameter_count() -> None:
    """Verify W1 (80, 32), b1 (32,), W2 (32, 1), b2 (1,) and 2,625 total parameters."""
    model = NumPyMLP(input_dim=80, hidden_dim=32, output_dim=1, seed=42)
    assert model.W1.shape == (80, 32)
    assert model.b1.shape == (32,)
    assert model.W2.shape == (32, 1)
    assert model.b2.shape == (1,)

    summary = model.parameter_summary()
    assert summary["total_parameters"] == TOTAL_PARAMETERS == 2625


def test_forward_shapes_probability_bounds_and_finite_loss() -> None:
    """Verify forward cache shapes, (0, 1) probability bounds, and finite BCE loss."""
    model = NumPyMLP(seed=42)
    X, Y = _make_test_batch(n=12, seed=42)

    cache = model.forward(X)
    assert cache.X.shape == (12, INPUT_DIM)
    assert cache.Z1.shape == (12, HIDDEN_DIM)
    assert cache.A1.shape == (12, HIDDEN_DIM)
    assert cache.Z2.shape == (12, OUTPUT_DIM)
    assert cache.Y_hat.shape == (12, OUTPUT_DIM)

    # tanh bounds in (-1, 1) and sigmoid bounds in (0, 1)
    assert np.all((cache.A1 > -1.0) & (cache.A1 < 1.0))
    assert np.all((cache.Y_hat > 0.0) & (cache.Y_hat < 1.0))

    # Numerical stability of stable_sigmoid on extreme logits
    extreme_z = np.array([-1000.0, 0.0, 1000.0], dtype=np.float64)
    sig_vals = stable_sigmoid(extreme_z)
    assert np.all(np.isfinite(sig_vals))
    assert sig_vals[0] == pytest.approx(0.0, abs=1e-15)
    assert sig_vals[1] == pytest.approx(0.5, abs=1e-15)
    assert sig_vals[2] == pytest.approx(1.0, abs=1e-15)

    loss = model.compute_loss(X, Y)
    assert isinstance(loss, float)
    assert np.isfinite(loss) and loss > 0.0

    prob_loss = compute_bce_loss(Y, cache.Y_hat)
    assert np.isfinite(prob_loss) and abs(loss - prob_loss) < 1e-10


def test_analytical_gradient_shapes() -> None:
    """Verify analytical backward pass produces gradients matching parameter shapes."""
    model = NumPyMLP(seed=42)
    X, Y = _make_test_batch(n=8, seed=42)
    cache = model.forward(X)
    grads = model.backward(cache, Y)

    assert grads["dW1"].shape == (80, 32)
    assert grads["db1"].shape == (32,)
    assert grads["dW2"].shape == (32, 1)
    assert grads["db2"].shape == (1,)
    for g in grads.values():
        assert np.all(np.isfinite(g))


# ==============================================================================
# 6. Central Finite-Difference Gradient Verification
# ==============================================================================


def test_central_difference_gradient_verification_passes() -> None:
    """Verify analytical gradients match central finite-difference to < 1e-9 relative error."""
    model = NumPyMLP(seed=42)
    X, Y = _make_test_batch(n=8, seed=42)

    report = verify_gradients_finite_difference(
        model=model,
        X_batch=X,
        Y_batch=Y,
        epsilon=1e-5,
        tolerance_relative=1e-6,
        tolerance_absolute=1e-8,
    )
    assert report.all_passed is True
    assert len(report.entries) >= 5
    checked_params = {e.parameter_name for e in report.entries}
    assert checked_params == {"W1", "b1", "W2", "b2"}
    assert report.max_relative_difference < 1e-8
    assert report.max_absolute_difference < 1e-9


# ==============================================================================
# 7–9. Parameter Updates, Seed Reproducibility, and Tiny-Batch Overfitting
# ==============================================================================


def test_parameter_update_step_and_loss_decrease() -> None:
    """Verify a gradient step modifies all parameters and reduces single-batch loss."""
    model = NumPyMLP(seed=42)
    X, Y = _make_test_batch(n=10, seed=42)

    w1_before = model.W1.copy()
    b1_before = model.b1.copy()
    w2_before = model.W2.copy()
    b2_before = model.b2.copy()

    loss_before = model.compute_loss(X, Y)
    cache = model.forward(X)
    grads = model.backward(cache, Y)
    model.step(grads, learning_rate=0.05)
    loss_after = model.compute_loss(X, Y)

    assert not np.array_equal(model.W1, w1_before)
    assert not np.array_equal(model.b1, b1_before)
    assert not np.array_equal(model.W2, w2_before)
    assert not np.array_equal(model.b2, b2_before)
    assert loss_after < loss_before


@pytest.mark.parametrize("seed", [42, 123, 2026])
def test_reproducibility_with_fixed_seeds(seed: int) -> None:
    """Verify two NumPyMLP models initialized with the same seed train identically."""
    X, Y = _make_test_batch(n=16, seed=seed)

    m1 = NumPyMLP(seed=seed)
    m2 = NumPyMLP(seed=seed)
    assert np.array_equal(m1.W1, m2.W1)
    assert np.array_equal(m1.W2, m2.W2)

    loss_1 = m1.train_epoch(X, Y, learning_rate=0.05, batch_size=8, shuffle=True, rng=np.random.default_rng(seed))
    loss_2 = m2.train_epoch(X, Y, learning_rate=0.05, batch_size=8, shuffle=True, rng=np.random.default_rng(seed))
    assert loss_1 == pytest.approx(loss_2, abs=1e-14)
    assert np.allclose(m1.W1, m2.W1, atol=1e-14)


def test_tiny_batch_overfitting_loss_reduction() -> None:
    """Verify Phase 4 tiny-batch diagnosis converges to <= 0.02 loss and 100% accuracy."""
    report = run_tiny_batch_diagnosis(
        seed=42,
        batch_size=16,
        learning_rate=0.1,
        max_epochs=300,
        target_loss_threshold=0.02,
    )
    assert report.loss_reduction_achieved is True
    assert report.converged_to_threshold is True
    assert report.final_loss <= 0.02
    assert report.final_loss < report.initial_loss * 0.05
    assert report.final_accuracy == 1.0


# ==============================================================================
# 10–11. Invalid Input Dimensions and NaN/Inf Rejection
# ==============================================================================


def test_invalid_input_dimensions_and_labels_rejected() -> None:
    """Verify mismatched feature dimensions, empty batches, and non-binary labels raise ValueError."""
    model = NumPyMLP(seed=42)

    with pytest.raises(ValueError, match="expected \\(N, 80\\)"):
        model.forward(np.zeros((5, 79)))
    with pytest.raises(ValueError, match="expected \\(N, 80\\)"):
        model.forward(np.zeros((0, 80)))
    with pytest.raises(ValueError, match="2D array"):
        model.forward(np.zeros((2, 4, 80)))

    X_ok = np.zeros((4, 80))
    with pytest.raises(ValueError, match="Batch size mismatch"):
        model.compute_loss(X_ok, np.zeros((3, 1)))
    with pytest.raises(ValueError, match="binary values in \\{0, 1\\}"):
        model.compute_loss(X_ok, np.array([[0], [1], [2], [0]]))


def test_nan_and_inf_inputs_rejected() -> None:
    """Verify NaN or Inf values in X or Y raise ValueError."""
    model = NumPyMLP(seed=42)
    X_nan = np.zeros((4, 80))
    X_nan[1, 10] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        model.forward(X_nan)

    X_inf = np.zeros((4, 80))
    X_inf[0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        model.forward(X_inf)

    Y_nan = np.array([[0.0], [np.nan], [1.0], [0.0]])
    with pytest.raises(ValueError, match="NaN or Inf"):
        model.compute_loss(np.zeros((4, 80)), Y_nan)


# ==============================================================================
# 12–13. Parameter Serialization Round Trip and Prediction Consistency
# ==============================================================================


@pytest.mark.parametrize("ext", [".npz", ".json"])
def test_parameter_save_load_round_trip_and_prediction_consistency(
    tmp_path: Path, ext: str
) -> None:
    """Verify saving and loading preserves exact parameters and predictions."""
    X, Y = _make_test_batch(n=12, seed=123)
    model = NumPyMLP(seed=123)
    model.train_epoch(X, Y, learning_rate=0.08, batch_size=4, shuffle=False)

    probs_before = model.predict_proba(X)
    preds_before = model.predict(X)

    save_path = model.save_parameters(tmp_path / f"numpy_mlp_test{ext}", extra_metadata={"step": 3})
    loaded = NumPyMLP.load_parameters(save_path)

    assert np.allclose(model.W1, loaded.W1, atol=1e-14)
    assert np.allclose(model.b1, loaded.b1, atol=1e-14)
    assert np.allclose(model.W2, loaded.W2, atol=1e-14)
    assert np.allclose(model.b2, loaded.b2, atol=1e-14)

    probs_after = loaded.predict_proba(X)
    preds_after = loaded.predict(X)
    assert np.allclose(probs_before, probs_after, atol=1e-14)
    assert np.array_equal(preds_before, preds_after)


def test_train_with_validation_and_metrics_helper() -> None:
    """Verify train_numpy_mlp_with_validation and binary classification metrics."""
    X_tr, Y_tr = _make_test_batch(n=24, seed=42)
    X_va, Y_va = _make_test_batch(n=8, seed=123)

    res = train_numpy_mlp_with_validation(
        X_train=X_tr,
        Y_train=Y_tr,
        X_val=X_va,
        Y_val=Y_va,
        seed=42,
        learning_rate=0.05,
        batch_size=8,
        epochs=10,
    )
    assert len(res["train_loss_history"]) == 10
    assert len(res["val_loss_history"]) == 10
    assert res["final_train_loss"] < res["initial_train_loss"]
    metrics = res["validation_metrics"]
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert "confusion_matrix" in metrics

    perfect = compute_binary_classification_metrics(
        np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1])
    )
    assert perfect["macro_f1"] == pytest.approx(1.0)
    assert perfect["confusion_matrix"]["tp"] == 2
    assert perfect["confusion_matrix"]["tn"] == 2

