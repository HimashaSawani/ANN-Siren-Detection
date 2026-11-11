"""First-Principles One-Hidden-Layer Neural Network (NumPy MLP) From Scratch.

Module: NB6057CEM — Artificial Neural Networks
Step: 3 — CW1 T2: NumPy ANN Implementation, Analytical Backpropagation,
          Central-Difference Gradient Verification, and Numerical Diagnosis.

Strict Constraints:
- Implemented strictly in NumPy without PyTorch, TensorFlow, JAX, or autograd.
- Sample and Feature Dimension Convention:
  * Input matrix X has shape (N, D) = (N, 80), where rows N index audio observations
    and columns D = 80 index standardized MFCC summary features.
  * Target vector Y has shape (N, 1) (or 1D (N,) reshaped to (N, 1)), with binary
    values in {0, 1} (1 = siren, 0 = non-siren)
- Trainable Parameters (Total = 2,625 parameters):
  * W1: (80, 32)  — Input-to-hidden weight matrix (2,560 parameters)
  * b1: (32,)     — Hidden layer bias vector (32 parameters)
  * W2: (32, 1)   — Hidden-to-output weight matrix (32 parameters)
  * b2: (1,)      — Output layer bias scalar/vector (1 parameter)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


INPUT_DIM: int = 80
HIDDEN_DIM: int = 32
OUTPUT_DIM: int = 1
TOTAL_PARAMETERS: int = INPUT_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * OUTPUT_DIM + OUTPUT_DIM


@dataclass(frozen=True)
class ForwardCache:
    """Intermediate activations cached during the forward pass for analytical backprop."""

    X: np.ndarray      # (N, 80)
    Z1: np.ndarray     # (N, 32)
    A1: np.ndarray     # (N, 32)
    Z2: np.ndarray     # (N, 1)
    Y_hat: np.ndarray  # (N, 1)


@dataclass(frozen=True)
class GradientCheckEntry:
    """Single parameter entry comparison between analytical and numerical gradients."""

    parameter_name: str
    index: tuple[int, ...]
    parameter_value: float
    analytical_gradient: float
    numerical_gradient: float
    absolute_difference: float
    relative_difference: float
    passed: bool


@dataclass(frozen=True)
class GradientCheckReport:
    """Complete report of central finite-difference gradient verification."""

    epsilon: float
    tolerance_relative: float
    tolerance_absolute: float
    batch_size: int
    seed: int
    dtype: str
    all_passed: bool
    max_absolute_difference: float
    max_relative_difference: float
    entries: list[GradientCheckEntry] = field(default_factory=list)


@dataclass(frozen=True)
class TinyBatchDiagnosisReport:
    """Report from the controlled tiny-batch overfitting numerical diagnosis experiment."""

    seed: int
    batch_size: int
    learning_rate: float
    max_epochs: int
    actual_epochs_completed: int
    target_loss_threshold: float
    stopping_rule: str
    initial_loss: float
    final_loss: float
    loss_reduction_achieved: bool
    converged_to_threshold: bool
    initial_accuracy: float
    final_accuracy: float
    loss_history: list[float] = field(default_factory=list)
    data_source: str = "synthetic_diagnostic_batch"


def stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute logistic sigmoid 1 / (1 + exp(-z)) without exponential overflow.

    Uses piecewise evaluation:
    - For z >= 0: 1 / (1 + exp(-z))
    - For z <  0: exp(z) / (1 + exp(z))
    """
    z_arr = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z_arr, dtype=np.float64)
    pos_mask = z_arr >= 0.0
    neg_mask = ~pos_mask

    out[pos_mask] = 1.0 / (1.0 + np.exp(-z_arr[pos_mask]))
    exp_z = np.exp(z_arr[neg_mask])
    out[neg_mask] = exp_z / (1.0 + exp_z)
    return out


def compute_bce_loss(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Compute scalar mean Binary Cross-Entropy loss from probabilities.

    Args:
        y_true: Binary targets in {0, 1} of shape ``(N, 1)`` or ``(N,)``.
        y_prob: Predicted probabilities in ``(0, 1)`` of shape ``(N, 1)`` or ``(N,)``.
        eps: Probability clipping bound to avoid ``log(0)``.

    Returns:
        Scalar mean BCE loss as a Python ``float``.
    """
    y_t = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)
    y_p = np.asarray(y_prob, dtype=np.float64).reshape(-1, 1)
    if y_t.shape != y_p.shape or y_t.shape[0] == 0:
        raise ValueError(
            f"Shape mismatch or empty inputs in compute_bce_loss: y_true={y_t.shape}, y_prob={y_p.shape}"
        )
    if not (np.all(np.isfinite(y_t)) and np.all(np.isfinite(y_p))):
        raise ValueError("NaN or Inf detected in compute_bce_loss inputs.")

    p_clipped = np.clip(y_p, eps, 1.0 - eps)
    bce = -np.mean(y_t * np.log(p_clipped) + (1.0 - y_t) * np.log(1.0 - p_clipped))
    return float(bce)


def compute_bce_loss_from_logits(y_true: np.ndarray, z_logits: np.ndarray) -> float:
    """Compute exact, overflow-free mean Binary Cross-Entropy directly from logits Z2.

    Uses the analytical identity:
        BCE(z, y) = max(z, 0) - y * z + log(1 + exp(-|z|))
    which is C-infinity smooth everywhere and avoids probability clipping bias during
    finite-difference gradient checking.
    """
    y_t = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)
    z = np.asarray(z_logits, dtype=np.float64).reshape(-1, 1)
    if y_t.shape != z.shape or y_t.shape[0] == 0:
        raise ValueError(
            f"Shape mismatch or empty inputs: y_true={y_t.shape}, z_logits={z.shape}"
        )
    loss_vec = np.maximum(z, 0.0) - y_t * z + np.log1p(np.exp(-np.abs(z)))
    return float(np.mean(loss_vec))


def compute_binary_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Compute confusion matrix, macro-F1, siren F1, precision, recall, and accuracy in pure NumPy."""
    yt = np.asarray(y_true, dtype=np.int64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if yt.shape != yp.shape or yt.size == 0:
        raise ValueError("y_true and y_pred must have identical non-empty 1D shapes.")

    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))

    # Positive class (Label 1: siren)
    prec_1 = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec_1 = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1_1 = (
        float(2.0 * prec_1 * rec_1 / (prec_1 + rec_1))
        if (prec_1 + rec_1) > 0.0
        else 0.0
    )

    # Negative class (Label 0: non_siren)
    prec_0 = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    rec_0 = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1_0 = (
        float(2.0 * prec_0 * rec_0 / (prec_0 + rec_0))
        if (prec_0 + rec_0) > 0.0
        else 0.0
    )

    macro_f1 = 0.5 * (f1_0 + f1_1)
    accuracy = float((tp + tn) / len(yt))

    return {
        "macro_f1": float(macro_f1),
        "f1_siren": float(f1_1),
        "f1_non_siren": float(f1_0),
        "precision_siren": float(prec_1),
        "recall_siren": float(rec_1),
        "accuracy": float(accuracy),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix_2x2": [[tn, fp], [fn, tp]],
        },
    }


class NumPyMLP:
    """One-hidden-layer Multi-Layer Perceptron implemented from first principles in NumPy.

    Architecture:
        Input X (N, 80)
          -> Linear1: Z1 = X @ W1 + b1       [W1: (80, 32), b1: (32,)]
          -> Hidden Activation: A1 = tanh(Z1)
          -> Linear2: Z2 = A1 @ W2 + b2      [W2: (32, 1),  b2: (1,)]
          -> Output Activation: Y_hat = sigmoid(Z2)  [(N, 1)]
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        output_dim: int = OUTPUT_DIM,
        seed: int = 42,
        init_scheme: str = "xavier_normal",
    ) -> None:
        if input_dim <= 0 or hidden_dim <= 0 or output_dim != 1:
            raise ValueError(
                f"Invalid dimensions: input_dim={input_dim}, hidden_dim={hidden_dim}, "
                f"output_dim={output_dim} (must be 1)."
            )
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.seed = int(seed)
        self.init_scheme = str(init_scheme)

        self.W1: np.ndarray
        self.b1: np.ndarray
        self.W2: np.ndarray
        self.b2: np.ndarray
        self.initialize_parameters(seed=self.seed, scheme=self.init_scheme)

    def initialize_parameters(
        self, seed: int | None = None, scheme: str = "xavier_normal"
    ) -> None:
        """Explicitly initialize W1, b1, W2, b2 using a seeded NumPy Generator."""
        active_seed = self.seed if seed is None else int(seed)
        rng = np.random.default_rng(active_seed)

        if scheme == "xavier_normal":
            std1 = np.sqrt(2.0 / (self.input_dim + self.hidden_dim))
            std2 = np.sqrt(2.0 / (self.hidden_dim + self.output_dim))
            self.W1 = rng.normal(0.0, std1, size=(self.input_dim, self.hidden_dim)).astype(
                np.float64
            )
            self.W2 = rng.normal(0.0, std2, size=(self.hidden_dim, self.output_dim)).astype(
                np.float64
            )
        elif scheme == "xavier_uniform":
            lim1 = np.sqrt(6.0 / (self.input_dim + self.hidden_dim))
            lim2 = np.sqrt(6.0 / (self.hidden_dim + self.output_dim))
            self.W1 = rng.uniform(-lim1, lim1, size=(self.input_dim, self.hidden_dim)).astype(
                np.float64
            )
            self.W2 = rng.uniform(-lim2, lim2, size=(self.hidden_dim, self.output_dim)).astype(
                np.float64
            )
        else:
            raise ValueError(
                f"Unknown weight initialization scheme '{scheme}'. "
                "Use 'xavier_normal' or 'xavier_uniform'."
            )

        self.b1 = np.zeros((self.hidden_dim,), dtype=np.float64)
        self.b2 = np.zeros((self.output_dim,), dtype=np.float64)

    def _validate_features(self, X: np.ndarray) -> np.ndarray:
        """Validate input feature matrix X has shape (N, input_dim) and finite values."""
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            if arr.shape[0] != self.input_dim:
                raise ValueError(
                    f"1D input must have length {self.input_dim}, got {arr.shape[0]}."
                )
            arr = arr.reshape(1, self.input_dim)
        elif arr.ndim != 2:
            raise ValueError(
                f"Input X must be a 2D array of shape (N, {self.input_dim}), got ndim={arr.ndim}."
            )

        if arr.shape[0] == 0 or arr.shape[1] != self.input_dim:
            raise ValueError(
                f"Invalid input shape {arr.shape}; expected (N, {self.input_dim}) with N > 0."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("Input features X contain NaN or Inf values.")
        return arr

    def _validate_labels(self, Y: np.ndarray, expected_n: int) -> np.ndarray:
        """Validate binary target labels Y have shape (N, 1) and values in {0, 1}."""
        arr = np.asarray(Y, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim != 2 or arr.shape[1] != 1:
            raise ValueError(
                f"Target Y must have shape (N,) or (N, 1), got {arr.shape}."
            )
        if arr.shape[0] != expected_n:
            raise ValueError(
                f"Batch size mismatch: X has {expected_n} samples, Y has {arr.shape[0]}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("Target labels Y contain NaN or Inf values.")
        if not np.all(np.isin(arr, [0.0, 1.0])):
            raise ValueError("Target labels Y must be binary values in {0, 1}.")
        return arr

    def forward(self, X: np.ndarray) -> ForwardCache:
        """Execute forward propagation and return all intermediate activations.

        Equations:
            Z1 = X @ W1 + b1       shape: (N, 32)
            A1 = tanh(Z1)          shape: (N, 32)
            Z2 = A1 @ W2 + b2      shape: (N, 1)
            Y_hat = sigmoid(Z2)    shape: (N, 1)
        """
        X_val = self._validate_features(X)
        Z1 = X_val @ self.W1 + self.b1
        A1 = np.tanh(Z1)
        Z2 = A1 @ self.W2 + self.b2
        Y_hat = stable_sigmoid(Z2)

        if not (np.all(np.isfinite(Z1)) and np.all(np.isfinite(Z2)) and np.all(np.isfinite(Y_hat))):
            raise FloatingPointError("NaN or Inf encountered during forward pass.")

        return ForwardCache(X=X_val, Z1=Z1, A1=A1, Z2=Z2, Y_hat=Y_hat)

    def compute_loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Run forward pass and compute exact mean Binary Cross-Entropy loss."""
        cache = self.forward(X)
        Y_val = self._validate_labels(Y, expected_n=cache.X.shape[0])
        return compute_bce_loss_from_logits(Y_val, cache.Z2)

    def backward(
        self, cache: ForwardCache, Y: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Compute analytical gradients dW1, db1, dW2, db2 via chain rule.

        Chain-Rule Equations (batch size N):
            dZ2 = (Y_hat - Y) / N                  shape: (N, 1)
            dW2 = A1.T @ dZ2                       shape: (32, 1)
            db2 = sum(dZ2, axis=0)                 shape: (1,)
            dA1 = dZ2 @ W2.T                       shape: (N, 32)
            dZ1 = dA1 * (1 - A1^2)                 shape: (N, 32)
            dW1 = X.T @ dZ1                        shape: (80, 32)
            db1 = sum(dZ1, axis=0)                 shape: (32,)
        """
        N = cache.X.shape[0]
        Y_val = self._validate_labels(Y, expected_n=N)

        dZ2 = (cache.Y_hat - Y_val) / float(N)          # (N, 1)
        dW2 = cache.A1.T @ dZ2                          # (32, 1)
        db2 = np.sum(dZ2, axis=0)                       # (1,)

        dA1 = dZ2 @ self.W2.T                           # (N, 32)
        dZ1 = dA1 * (1.0 - np.square(cache.A1))         # (N, 32)
        dW1 = cache.X.T @ dZ1                           # (80, 32)
        db1 = np.sum(dZ1, axis=0)                       # (32,)

        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        for name, g in grads.items():
            if not np.all(np.isfinite(g)):
                raise FloatingPointError(f"Non-finite analytical gradient in {name}.")
        return grads

    def step(self, grads: dict[str, np.ndarray], learning_rate: float) -> None:
        """Apply one mini-batch gradient descent parameter update step."""
        if learning_rate <= 0.0 or not np.isfinite(learning_rate):
            raise ValueError(f"learning_rate must be a positive finite float, got {learning_rate}.")

        self.W1 = self.W1 - learning_rate * grads["dW1"]
        self.b1 = self.b1 - learning_rate * grads["db1"]
        self.W2 = self.W2 - learning_rate * grads["dW2"]
        self.b2 = self.b2 - learning_rate * grads["db2"]

    def train_epoch(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        learning_rate: float = 0.05,
        batch_size: int = 32,
        shuffle: bool = True,
        rng: np.random.Generator | None = None,
    ) -> float:
        """Run one epoch of mini-batch gradient descent and return full-split post-epoch loss."""
        X_val = self._validate_features(X_train)
        Y_val = self._validate_labels(Y_train, expected_n=X_val.shape[0])
        n_samples = X_val.shape[0]

        if shuffle:
            gen = rng if rng is not None else np.random.default_rng(self.seed)
            indices = gen.permutation(n_samples)
        else:
            indices = np.arange(n_samples)

        for start_idx in range(0, n_samples, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            xb = X_val[batch_idx]
            yb = Y_val[batch_idx]
            cache = self.forward(xb)
            grads = self.backward(cache, yb)
            self.step(grads, learning_rate=learning_rate)

        return self.compute_loss(X_val, Y_val)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict siren probabilities Y_hat in (0, 1) of shape (N, 1)."""
        return self.forward(X).Y_hat.copy()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels in {0, 1} of shape (N, 1)."""
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"Classification threshold must be in (0, 1), got {threshold}.")
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(np.int64)

    def parameter_summary(self) -> dict[str, Any]:
        """Return parameter shapes and counts for architectural reporting."""
        w1_count = int(self.W1.size)
        b1_count = int(self.b1.size)
        w2_count = int(self.W2.size)
        b2_count = int(self.b2.size)
        return {
            "W1": {"shape": list(self.W1.shape), "count": w1_count},
            "b1": {"shape": list(self.b1.shape), "count": b1_count},
            "W2": {"shape": list(self.W2.shape), "count": w2_count},
            "b2": {"shape": list(self.b2.shape), "count": b2_count},
            "total_parameters": w1_count + b1_count + w2_count + b2_count,
        }

    def save_parameters(self, path: str | Path, extra_metadata: dict[str, Any] | None = None) -> Path:
        """Serialize model weights, biases, and architecture metadata to `.npz` or `.json`."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.suffix.lower() == ".npz":
            np.savez(
                target,
                W1=self.W1,
                b1=self.b1,
                W2=self.W2,
                b2=self.b2,
                input_dim=np.array(self.input_dim),
                hidden_dim=np.array(self.hidden_dim),
                output_dim=np.array(self.output_dim),
                seed=np.array(self.seed),
                init_scheme=np.array(self.init_scheme),
            )
            if extra_metadata is not None:
                meta_path = target.with_suffix(".meta.json")
                meta_path.write_text(json.dumps(extra_metadata, indent=2), encoding="utf-8")
        else:
            payload = {
                "model_type": "NumPyMLP",
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "seed": self.seed,
                "init_scheme": self.init_scheme,
                "W1": self.W1.tolist(),
                "b1": self.b1.tolist(),
                "W2": self.W2.tolist(),
                "b2": self.b2.tolist(),
                "metadata": extra_metadata or {},
            }
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load_parameters(cls, path: str | Path) -> NumPyMLP:
        """Load a serialized NumPyMLP instance from `.npz` or `.json`."""
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"Saved NumPyMLP file not found at '{target}'.")

        if target.suffix.lower() == ".npz":
            with np.load(target, allow_pickle=False) as data:
                model = cls(
                    input_dim=int(data["input_dim"]),
                    hidden_dim=int(data["hidden_dim"]),
                    output_dim=int(data["output_dim"]),
                    seed=int(data["seed"]),
                    init_scheme=str(data["init_scheme"]),
                )
                model.W1 = np.asarray(data["W1"], dtype=np.float64)
                model.b1 = np.asarray(data["b1"], dtype=np.float64)
                model.W2 = np.asarray(data["W2"], dtype=np.float64)
                model.b2 = np.asarray(data["b2"], dtype=np.float64)
                return model

        payload = json.loads(target.read_text(encoding="utf-8"))
        model = cls(
            input_dim=int(payload["input_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            output_dim=int(payload["output_dim"]),
            seed=int(payload.get("seed", 42)),
            init_scheme=str(payload.get("init_scheme", "xavier_normal")),
        )
        model.W1 = np.asarray(payload["W1"], dtype=np.float64)
        model.b1 = np.asarray(payload["b1"], dtype=np.float64)
        model.W2 = np.asarray(payload["W2"], dtype=np.float64)
        model.b2 = np.asarray(payload["b2"], dtype=np.float64)
        return model


def verify_gradients_finite_difference(
    model: NumPyMLP,
    X_batch: np.ndarray,
    Y_batch: np.ndarray,
    epsilon: float = 1e-5,
    tolerance_relative: float = 1e-6,
    tolerance_absolute: float = 1e-8,
    custom_checks: list[tuple[str, tuple[int, ...]]] | None = None,
) -> GradientCheckReport:
    """Verify analytical backpropagation against central finite-difference numerical gradients.

    Formula:
        g_num(theta_k) = [L(theta_k + eps) - L(theta_k - eps)] / (2 * eps)

    Relative difference formula:
        |g_ana - g_num| / max(|g_ana| + |g_num|, 1e-12)
    """
    X_f64 = np.asarray(X_batch, dtype=np.float64)
    Y_f64 = np.asarray(Y_batch, dtype=np.float64).reshape(-1, 1)

    # Compute analytical gradients before any parameter updates
    cache = model.forward(X_f64)
    analytical_grads = model.backward(cache, Y_f64)

    default_checks: list[tuple[str, tuple[int, ...]]] = [
        ("W1", (0, 0)),
        ("W1", (17, 9)),
        ("W1", (45, 21)),
        ("W1", (79, 31)),
        ("b1", (0,)),
        ("b1", (15,)),
        ("b1", (31,)),
        ("W2", (0, 0)),
        ("W2", (16, 0)),
        ("W2", (31, 0)),
        ("b2", (0,)),
    ]
    checks_to_run = custom_checks if custom_checks is not None else default_checks

    param_map = {
        "W1": (model.W1, analytical_grads["dW1"]),
        "b1": (model.b1, analytical_grads["db1"]),
        "W2": (model.W2, analytical_grads["dW2"]),
        "b2": (model.b2, analytical_grads["db2"]),
    }

    entries: list[GradientCheckEntry] = []
    for param_name, idx in checks_to_run:
        param_arr, grad_arr = param_map[param_name]
        orig_val = float(param_arr[idx])
        ana_val = float(grad_arr[idx])

        # Perturb +epsilon
        param_arr[idx] = orig_val + epsilon
        loss_plus = model.compute_loss(X_f64, Y_f64)

        # Perturb -epsilon
        param_arr[idx] = orig_val - epsilon
        loss_minus = model.compute_loss(X_f64, Y_f64)

        # Restore exact original parameter value
        param_arr[idx] = orig_val

        num_val = float((loss_plus - loss_minus) / (2.0 * epsilon))
        abs_diff = float(abs(ana_val - num_val))
        denom = max(abs(ana_val) + abs(num_val), 1e-12)
        rel_diff = float(abs_diff / denom)

        passed = bool((rel_diff <= tolerance_relative) or (abs_diff <= tolerance_absolute))
        entries.append(
            GradientCheckEntry(
                parameter_name=param_name,
                index=tuple(int(i) for i in idx),
                parameter_value=orig_val,
                analytical_gradient=ana_val,
                numerical_gradient=num_val,
                absolute_difference=abs_diff,
                relative_difference=rel_diff,
                passed=passed,
            )
        )

    max_abs = max((e.absolute_difference for e in entries), default=0.0)
    max_rel = max((e.relative_difference for e in entries), default=0.0)
    all_ok = all(e.passed for e in entries)

    return GradientCheckReport(
        epsilon=float(epsilon),
        tolerance_relative=float(tolerance_relative),
        tolerance_absolute=float(tolerance_absolute),
        batch_size=int(X_f64.shape[0]),
        seed=int(model.seed),
        dtype="float64",
        all_passed=all_ok,
        max_absolute_difference=float(max_abs),
        max_relative_difference=float(max_rel),
        entries=entries,
    )


def run_tiny_batch_diagnosis(
    X_tiny: np.ndarray | None = None,
    Y_tiny: np.ndarray | None = None,
    seed: int = 42,
    batch_size: int = 16,
    learning_rate: float = 0.1,
    max_epochs: int = 300,
    target_loss_threshold: float = 0.02,
    data_source: str = "synthetic_diagnostic_batch",
) -> TinyBatchDiagnosisReport:
    """Execute the Phase 4 controlled tiny-batch overfitting experiment.

    Trains a fresh NumPyMLP on a fixed tiny batch (without shuffling across other batches)
    to verify that analytical gradients and parameter updates monotonically reduce training
    loss and memorize a small batch.
    """
    rng = np.random.default_rng(seed)
    if X_tiny is None or Y_tiny is None:
        half = batch_size // 2
        # Class 0 and Class 1 synthetic standardized 80-D feature clusters
        x0 = rng.normal(loc=-0.4, scale=0.8, size=(half, INPUT_DIM))
        x1 = rng.normal(loc=+0.4, scale=0.8, size=(batch_size - half, INPUT_DIM))
        X_batch = np.vstack([x0, x1]).astype(np.float64)
        Y_batch = np.vstack(
            [np.zeros((half, 1)), np.ones((batch_size - half, 1))]
        ).astype(np.float64)
    else:
        X_batch = np.asarray(X_tiny, dtype=np.float64)
        Y_batch = np.asarray(Y_tiny, dtype=np.float64).reshape(-1, 1)

    model = NumPyMLP(input_dim=X_batch.shape[1], hidden_dim=HIDDEN_DIM, output_dim=1, seed=seed)

    initial_loss = model.compute_loss(X_batch, Y_batch)
    initial_preds = model.predict(X_batch)
    initial_acc = float(np.mean(initial_preds == Y_batch.astype(np.int64)))

    loss_history: list[float] = [initial_loss]
    epochs_run = 0
    stopping_rule = (
        f"Stop when BCE loss <= {target_loss_threshold} or epoch == {max_epochs}"
    )

    for epoch in range(1, max_epochs + 1):
        cache = model.forward(X_batch)
        grads = model.backward(cache, Y_batch)
        model.step(grads, learning_rate=learning_rate)
        current_loss = model.compute_loss(X_batch, Y_batch)
        loss_history.append(current_loss)
        epochs_run = epoch
        if current_loss <= target_loss_threshold:
            break

    final_loss = loss_history[-1]
    final_preds = model.predict(X_batch)
    final_acc = float(np.mean(final_preds == Y_batch.astype(np.int64)))

    return TinyBatchDiagnosisReport(
        seed=int(seed),
        batch_size=int(X_batch.shape[0]),
        learning_rate=float(learning_rate),
        max_epochs=int(max_epochs),
        actual_epochs_completed=int(epochs_run),
        target_loss_threshold=float(target_loss_threshold),
        stopping_rule=stopping_rule,
        initial_loss=float(initial_loss),
        final_loss=float(final_loss),
        loss_reduction_achieved=bool(final_loss < initial_loss),
        converged_to_threshold=bool(final_loss <= target_loss_threshold),
        initial_accuracy=initial_acc,
        final_accuracy=final_acc,
        loss_history=loss_history,
        data_source=data_source,
    )


def train_numpy_mlp_with_validation(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    seed: int = 42,
    learning_rate: float = 0.05,
    batch_size: int = 32,
    epochs: int = 100,
) -> dict[str, Any]:
    """Train NumPyMLP on training MFCC features and evaluate on validation features per epoch."""
    model = NumPyMLP(input_dim=X_train.shape[1], hidden_dim=HIDDEN_DIM, output_dim=1, seed=seed)
    rng = np.random.default_rng(seed)

    train_losses: list[float] = []
    val_losses: list[float] = []
    t0 = time.perf_counter()

    for _ in range(1, epochs + 1):
        tr_loss = model.train_epoch(
            X_train,
            Y_train,
            learning_rate=learning_rate,
            batch_size=batch_size,
            shuffle=True,
            rng=rng,
        )
        va_loss = model.compute_loss(X_val, Y_val)
        train_losses.append(float(tr_loss))
        val_losses.append(float(va_loss))

    duration_sec = float(time.perf_counter() - t0)
    val_preds = model.predict(X_val)
    val_metrics = compute_binary_classification_metrics(Y_val, val_preds)

    return {
        "model": model,
        "seed": int(seed),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "actual_epochs": int(epochs),
        "training_duration_sec": round(duration_sec, 4),
        "initial_train_loss": float(train_losses[0]),
        "final_train_loss": float(train_losses[-1]),
        "initial_val_loss": float(val_losses[0]),
        "final_val_loss": float(val_losses[-1]),
        "train_loss_history": train_losses,
        "val_loss_history": val_losses,
        "validation_metrics": val_metrics,
    }

