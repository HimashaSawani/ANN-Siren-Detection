"""Competing Model Architectures for ANN-Siren-Detection (CW1 T3).

Module: NB6057CEM — Artificial Neural Networks
Step: 4 — Implement and Train Three Competing Models (CW1 T3)

Implements:
1. Non-Neural Baseline: `RandomForestBaseline` (scikit-learn Random Forest on 80-D MFCCs).
2. Neural Architecture 1: `DeepMLP` (PyTorch 2-hidden-layer MLP: 80 -> 128 -> 64 -> 1 on 80-D MFCCs).
3. Neural Architecture 2: `Conv2DCNN` (PyTorch 3-block 2D CNN with BatchNorm, MaxPool,
   Global Average Pooling, Dropout, and Linear head on (1, 128, 173) Log-Mel spectrograms).
4. First-Principles Reference: `NumPyMLP` (imported from `src/numpy_mlp.py` for T2).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

from src import ConfigurationError, load_config, set_random_seeds


@dataclass(frozen=True)
class ModelSpecification:
    """Metadata specification for a configured model architecture."""

    model_id: str
    name: str
    category: str
    framework: str
    enabled: bool
    is_first_principles: bool = False


def get_model_specifications(
    config: dict[str, Any] | None = None,
) -> dict[str, ModelSpecification]:
    """Return all configured candidate and first-principles model specifications."""
    cfg = config if config is not None else load_config()
    models_section = cfg.get("models", {})
    if not isinstance(models_section, dict):
        raise ConfigurationError("Configuration section 'models' must be a dictionary.")

    specs: dict[str, ModelSpecification] = {}
    for entry in models_section.get("candidate_models", []):
        if not isinstance(entry, dict) or "id" not in entry:
            raise ConfigurationError(f"Invalid candidate model entry: {entry!r}")
        model_id = str(entry["id"])
        specs[model_id] = ModelSpecification(
            model_id=model_id,
            name=str(entry.get("name", model_id)),
            category=str(entry.get("category", "candidate")),
            framework=str(entry.get("framework", "unknown")),
            enabled=bool(entry.get("enabled", True)),
            is_first_principles=False,
        )

    fp_entry = models_section.get("first_principles_model")
    if isinstance(fp_entry, dict) and "id" in fp_entry:
        fp_id = str(fp_entry["id"])
        specs[fp_id] = ModelSpecification(
            model_id=fp_id,
            name=str(fp_entry.get("name", fp_id)),
            category=str(fp_entry.get("category", "first_principles_ann")),
            framework=str(fp_entry.get("framework", "numpy")),
            enabled=bool(fp_entry.get("enabled", True)),
            is_first_principles=True,
        )

    return specs


def count_pytorch_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters of a PyTorch ``nn.Module`` programmatically."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# ==============================================================================
# PHASE 2 — Non-Neural Baseline: Random Forest Classifier (scikit-learn)
# ==============================================================================


class RandomForestBaseline:
    """Scikit-learn Random Forest classifier wrapper for 80-D MFCC summary features.

    Enforces:
    - Training-only fitting guard (`split_name == "train"`).
    - Input shape `(N, 80)` and finite value validation.
    - Binary target label `{0, 1}` validation.
    - Deterministic execution via `random_state`.
    - Serialization of model and metadata (`training_duration_sec`, `model_size_bytes`).
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = 16,
        min_samples_split: int = 2,
        min_samples_leaf: int = 2,
        class_weight: str | dict[int, float] | None = "balanced",
        random_state: int = 42,
        n_jobs: int = 1,
        input_dim: int = 80,
    ) -> None:
        if n_estimators <= 0:
            raise ValueError(f"n_estimators must be positive, got {n_estimators}.")
        if max_depth is not None and max_depth <= 0:
            raise ValueError(f"max_depth must be positive or None, got {max_depth}.")
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")

        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth) if max_depth is not None else None
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.class_weight = class_weight
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.input_dim = int(input_dim)

        self.estimator = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.is_fitted: bool = False
        self.training_duration_sec: float | None = None
        self.num_train_samples: int = 0
        self.train_class_counts: dict[str, int] = {}

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 1:
            if arr.shape[0] != self.input_dim:
                raise ValueError(
                    f"Expected 1D feature vector of length {self.input_dim}, got {arr.shape[0]}."
                )
            arr = arr.reshape(1, self.input_dim)
        elif arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected 2D MFCC array of shape (N, {self.input_dim}) with N > 0, got {arr.shape}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("Input MFCC features contain NaN or Inf values.")
        return arr

    def _validate_y(self, y: np.ndarray, expected_n: int) -> np.ndarray:
        arr = np.asarray(y)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.reshape(-1)
        elif arr.ndim != 1:
            raise ValueError(f"Expected 1D label array, got shape {arr.shape}.")
        if arr.shape[0] != expected_n:
            raise ValueError(
                f"Sample count mismatch: X has {expected_n} rows, y has {arr.shape[0]}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("Labels y contain NaN or Inf values.")
        if not np.all(np.isin(arr, [0, 1])):
            raise ValueError("Labels y must be binary values in {0, 1}.")
        uniques = np.unique(arr.astype(np.int64))
        if len(uniques) < 2:
            raise ValueError(
                f"Training labels must contain both classes {{0, 1}}, found only {uniques.tolist()}."
            )
        return arr.astype(np.int64)

    def fit(
        self, X_train: np.ndarray, y_train: np.ndarray, split_name: str = "train"
    ) -> RandomForestBaseline:
        """Fit the Random Forest classifier strictly on the training split."""
        if split_name.strip().lower() != "train":
            raise ValueError(
                f"Leakage violation: RandomForestBaseline.fit() allows split='train' only, "
                f"got split='{split_name}'."
            )
        X_val = self._validate_X(X_train)
        y_val = self._validate_y(y_train, expected_n=X_val.shape[0])

        t0 = time.perf_counter()
        self.estimator.fit(X_val, y_val)
        self.training_duration_sec = round(float(time.perf_counter() - t0), 6)
        self.is_fitted = True
        self.num_train_samples = int(X_val.shape[0])
        self.train_class_counts = {
            "non_siren_0": int(np.sum(y_val == 0)),
            "siren_1": int(np.sum(y_val == 1)),
        }
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict positive class (`siren = 1`) probabilities of shape `(N, 1)`."""
        if not self.is_fitted:
            raise RuntimeError("RandomForestBaseline must be fitted before calling predict_proba().")
        X_val = self._validate_X(X)
        probs = self.estimator.predict_proba(X_val)
        # Positive class 1 is at column index 1
        pos_idx = list(self.estimator.classes_).index(1)
        return probs[:, pos_idx : pos_idx + 1].astype(np.float32)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels `{0, 1}` of shape `(N,)`."""
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"threshold must be in (0, 1), got {threshold}.")
        probs = self.predict_proba(X).reshape(-1)
        return (probs >= threshold).astype(np.int64)

    def get_hyperparameters(self) -> dict[str, Any]:
        """Return dictionary of configured hyperparameters."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "input_dim": self.input_dim,
        }

    def save(
        self, path: str | Path, extra_metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Save the fitted Random Forest model via joblib and export metadata JSON."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted RandomForestBaseline.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "hyperparameters": self.get_hyperparameters(),
            "estimator": self.estimator,
            "training_duration_sec": self.training_duration_sec,
            "num_train_samples": self.num_train_samples,
            "train_class_counts": self.train_class_counts,
        }
        joblib.dump(payload, target)
        size_bytes = int(target.stat().st_size)

        meta = {
            "model_id": "random_forest",
            "framework": "scikit-learn",
            "checkpoint_path": str(target),
            "model_size_bytes": size_bytes,
            "training_duration_sec": self.training_duration_sec,
            "num_train_samples": self.num_train_samples,
            "train_class_counts": self.train_class_counts,
            "hyperparameters": self.get_hyperparameters(),
            "extra_metadata": extra_metadata or {},
        }
        meta_path = target.with_suffix(target.suffix + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    @classmethod
    def load(cls, path: str | Path) -> RandomForestBaseline:
        """Load a fitted `RandomForestBaseline` from a joblib checkpoint."""
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"RandomForest checkpoint not found at '{target}'.")
        payload = joblib.load(target)
        hp = payload["hyperparameters"]
        instance = cls(
            n_estimators=hp["n_estimators"],
            max_depth=hp["max_depth"],
            min_samples_split=hp["min_samples_split"],
            min_samples_leaf=hp["min_samples_leaf"],
            class_weight=hp["class_weight"],
            random_state=hp["random_state"],
            n_jobs=hp["n_jobs"],
            input_dim=hp["input_dim"],
        )
        instance.estimator = payload["estimator"]
        instance.is_fitted = True
        instance.training_duration_sec = payload.get("training_duration_sec")
        instance.num_train_samples = int(payload.get("num_train_samples", 0))
        instance.train_class_counts = dict(payload.get("train_class_counts", {}))
        return instance


# ==============================================================================
# PHASE 3 — Neural Architecture 1: PyTorch Deep MLP (80 -> 128 -> 64 -> 1)
# ==============================================================================


class DeepMLP(nn.Module):
    """Two-hidden-layer PyTorch Multi-Layer Perceptron for 80-D MFCC summary features.

    Default Architecture (18,689 trainable parameters):
        Input (B, 80)
          -> fc1: Linear(80, 128)   [10,368 params]
          -> act1: ReLU()
          -> drop1: Dropout(p=0.3)
          -> fc2: Linear(128, 64)   [8,256 params]
          -> act2: ReLU()
          -> drop2: Dropout(p=0.3)
          -> fc_out: Linear(64, 1)  [65 params]
          -> Output Logits (B, 1)   (for BCEWithLogitsLoss)
    """

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        output_dim: int = 1,
        dropout: float = 0.3,
        seed: int | None = 42,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if len(hidden_dims) < 2 or any(h <= 0 for h in hidden_dims):
            raise ValueError(
                f"DeepMLP requires at least 2 positive hidden dimensions, got {hidden_dims}."
            )
        if output_dim != 1:
            raise ValueError(f"Binary DeepMLP requires output_dim=1, got {output_dim}.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}.")

        if seed is not None:
            set_random_seeds(int(seed))

        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.output_dim = int(output_dim)
        self.dropout_rate = float(dropout)
        self.seed = seed

        layers: list[nn.Module] = []
        in_features = self.input_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(in_features, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.dropout_rate))
            in_features = h_dim

        self.hidden_network = nn.Sequential(*layers)
        self.fc_out = nn.Linear(in_features, self.output_dim)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply Kaiming normal initialization to ReLU layers and Xavier normal to output."""
        for module in self.hidden_network.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.xavier_normal_(self.fc_out.weight)
        if self.fc_out.bias is not None:
            nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw binary classification logits of shape `(B, 1)`."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor input, got {type(x).__name__}.")
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"DeepMLP expected 2D tensor of shape (B, {self.input_dim}) with B > 0, "
                f"got {tuple(x.shape)}."
            )
        if not torch.isfinite(x).all():
            raise ValueError("DeepMLP input tensor contains NaN or Inf values.")

        h = self.hidden_network(x)
        logits = self.fc_out(h)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute calibrated siren probabilities in `(0, 1)` of shape `(B, 1)`."""
        was_training = self.training
        self.eval()
        try:
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        finally:
            self.train(was_training)
        return probs

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Predict binary labels `{0, 1}` of shape `(B, 1)`."""
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"threshold must be in (0, 1), got {threshold}.")
        probs = self.predict_proba(x)
        return (probs >= threshold).to(dtype=torch.long)

    def count_trainable_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return count_pytorch_parameters(self, trainable_only=True)

    def architecture_summary(self) -> dict[str, Any]:
        """Return detailed layer-by-layer shape and parameter breakdown."""
        layer_rows: list[dict[str, Any]] = []
        in_dim = self.input_dim
        for idx, h_dim in enumerate(self.hidden_dims, start=1):
            linear_layer = self.hidden_network[(idx - 1) * 3]
            params = count_pytorch_parameters(linear_layer)
            layer_rows.append(
                {
                    "layer_name": f"fc{idx} + ReLU + Dropout({self.dropout_rate})",
                    "input_shape": f"(B, {in_dim})",
                    "output_shape": f"(B, {h_dim})",
                    "activation": "ReLU",
                    "trainable_parameters": params,
                }
            )
            in_dim = h_dim

        out_params = count_pytorch_parameters(self.fc_out)
        layer_rows.append(
            {
                "layer_name": "fc_out (Binary Logit Head)",
                "input_shape": f"(B, {in_dim})",
                "output_shape": f"(B, {self.output_dim})",
                "activation": "Linear Logit (Sigmoid in BCEWithLogitsLoss / inference)",
                "trainable_parameters": out_params,
            }
        )
        return {
            "model_id": "deep_mlp",
            "input_representation": "80-D Standardized MFCC Summary Vector",
            "input_shape": [self.input_dim],
            "hidden_dims": list(self.hidden_dims),
            "output_shape": [self.output_dim],
            "dropout": self.dropout_rate,
            "loss_function": "BCEWithLogitsLoss",
            "optimizer": "Adam",
            "trainable_parameters": self.count_trainable_parameters(),
            "layers": layer_rows,
        }


# ==============================================================================
# PHASE 4 — Neural Architecture 2: PyTorch 2D CNN on (1, 128, 173) Log-Mel
# ==============================================================================


class Conv2DCNN(nn.Module):
    """Three-block 2D Convolutional Neural Network for Log-Mel Spectrograms `(B, 1, 128, 173)`.

    Default Architecture (23,585 trainable parameters):
        Input: (B, 1, 128, 173)
        Block 1:
          - Conv2d(1, 16, kernel_size=3, stride=1, padding=1)   -> (B, 16, 128, 173) [160 params]
          - BatchNorm2d(16)                                     -> (B, 16, 128, 173) [32 params]
          - ReLU()
          - MaxPool2d(kernel_size=2, stride=2)                  -> (B, 16, 64, 86)
        Block 2:
          - Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  -> (B, 32, 64, 86)   [4,640 params]
          - BatchNorm2d(32)                                     -> (B, 32, 64, 86)   [64 params]
          - ReLU()
          - MaxPool2d(kernel_size=2, stride=2)                  -> (B, 32, 32, 43)
        Block 3:
          - Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  -> (B, 64, 32, 43)   [18,496 params]
          - BatchNorm2d(64)                                     -> (B, 64, 32, 43)   [128 params]
          - ReLU()
          - AdaptiveAvgPool2d((1, 1)) (Global Average Pooling)  -> (B, 64, 1, 1)
        Classifier Head:
          - Flatten()                                           -> (B, 64)
          - Dropout(p=0.3)                                      -> (B, 64)
          - Linear(64, 1)                                       -> (B, 1)             [65 params]
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: tuple[int, int, int] | list[int] = (16, 32, 64),
        n_mels: int = 128,
        time_frames: int = 173,
        dropout: float = 0.3,
        output_dim: int = 1,
        seed: int | None = 42,
    ) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError(f"Conv2DCNN expects in_channels=1, got {in_channels}.")
        if len(conv_channels) != 3 or any(c <= 0 for c in conv_channels):
            raise ValueError(
                f"Conv2DCNN requires 3 positive conv channel sizes, got {conv_channels}."
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}.")
        if output_dim != 1:
            raise ValueError(f"Binary Conv2DCNN requires output_dim=1, got {output_dim}.")

        if seed is not None:
            set_random_seeds(int(seed))

        self.in_channels = int(in_channels)
        self.conv_channels = tuple(int(c) for c in conv_channels)
        self.n_mels = int(n_mels)
        self.time_frames = int(time_frames)
        self.dropout_rate = float(dropout)
        self.output_dim = int(output_dim)
        self.seed = seed

        c1, c2, c3 = self.conv_channels

        self.block1 = nn.Sequential(
            nn.Conv2d(self.in_channels, c1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc_out = nn.Linear(c3, self.output_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply Kaiming normal init to Conv2d, unit/zero init to BatchNorm2d, Xavier to Linear."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw binary classification logits of shape `(B, 1)`."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor input, got {type(x).__name__}.")
        if (
            x.ndim != 4
            or x.shape[0] == 0
            or x.shape[1] != self.in_channels
            or x.shape[2] != self.n_mels
            or x.shape[3] != self.time_frames
        ):
            raise ValueError(
                f"Conv2DCNN expected 4D tensor of shape (B, {self.in_channels}, "
                f"{self.n_mels}, {self.time_frames}) with B > 0, got {tuple(x.shape)}."
            )
        if not torch.isfinite(x).all():
            raise ValueError("Conv2DCNN input tensor contains NaN or Inf values.")

        x1 = self.block1(x)                # (B, 16, 64, 86)
        x2 = self.block2(x1)               # (B, 32, 32, 43)
        x3 = self.block3(x2)               # (B, 64, 32, 43)
        pooled = self.global_pool(x3)      # (B, 64, 1, 1)
        flat = pooled.flatten(start_dim=1) # (B, 64)
        dropped = self.dropout(flat)       # (B, 64)
        logits = self.fc_out(dropped)      # (B, 1)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute calibrated siren probabilities in `(0, 1)` of shape `(B, 1)`."""
        was_training = self.training
        self.eval()
        try:
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        finally:
            self.train(was_training)
        return probs

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Predict binary labels `{0, 1}` of shape `(B, 1)`."""
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"threshold must be in (0, 1), got {threshold}.")
        probs = self.predict_proba(x)
        return (probs >= threshold).to(dtype=torch.long)

    def count_trainable_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return count_pytorch_parameters(self, trainable_only=True)

    def architecture_summary(self) -> dict[str, Any]:
        """Return complete tensor-shape progression and parameter breakdown."""
        c1, c2, c3 = self.conv_channels
        h1, w1 = self.n_mels // 2, self.time_frames // 2
        h2, w2 = h1 // 2, w1 // 2
        return {
            "model_id": "cnn_2d",
            "input_representation": "2D Log-Mel Spectrogram (Normalized)",
            "input_shape": [self.in_channels, self.n_mels, self.time_frames],
            "conv_channels": list(self.conv_channels),
            "kernel_size": [3, 3],
            "pool_size": [2, 2],
            "global_pooling": "AdaptiveAvgPool2d((1, 1))",
            "dropout": self.dropout_rate,
            "output_shape": [self.output_dim],
            "loss_function": "BCEWithLogitsLoss",
            "optimizer": "Adam",
            "trainable_parameters": self.count_trainable_parameters(),
            "layers": [
                {
                    "layer_name": "Block 1: Conv2d(1->16, 3x3, pad=1) + BN + ReLU + MaxPool2d(2x2)",
                    "input_shape": f"(B, 1, {self.n_mels}, {self.time_frames})",
                    "output_shape": f"(B, {c1}, {h1}, {w1})",
                    "activation": "ReLU",
                    "trainable_parameters": count_pytorch_parameters(self.block1),
                },
                {
                    "layer_name": "Block 2: Conv2d(16->32, 3x3, pad=1) + BN + ReLU + MaxPool2d(2x2)",
                    "input_shape": f"(B, {c1}, {h1}, {w1})",
                    "output_shape": f"(B, {c2}, {h2}, {w2})",
                    "activation": "ReLU",
                    "trainable_parameters": count_pytorch_parameters(self.block2),
                },
                {
                    "layer_name": "Block 3: Conv2d(32->64, 3x3, pad=1) + BN + ReLU",
                    "input_shape": f"(B, {c2}, {h2}, {w2})",
                    "output_shape": f"(B, {c3}, {h2}, {w2})",
                    "activation": "ReLU",
                    "trainable_parameters": count_pytorch_parameters(self.block3),
                },
                {
                    "layer_name": f"GlobalAvgPool2d(1x1) + Flatten + Dropout({self.dropout_rate})",
                    "input_shape": f"(B, {c3}, {h2}, {w2})",
                    "output_shape": f"(B, {c3})",
                    "activation": "Identity",
                    "trainable_parameters": 0,
                },
                {
                    "layer_name": "fc_out: Linear(64->1)",
                    "input_shape": f"(B, {c3})",
                    "output_shape": f"(B, {self.output_dim})",
                    "activation": "Linear Logit (Sigmoid in BCEWithLogitsLoss / inference)",
                    "trainable_parameters": count_pytorch_parameters(self.fc_out),
                },
            ],
        }


def create_model(
    model_id: str,
    seed: int = 42,
    **kwargs: Any,
) -> RandomForestBaseline | DeepMLP | Conv2DCNN:
    """Instantiate one of the three competing coursework models (`random_forest`, `deep_mlp`, `cnn_2d`)."""
    norm_id = model_id.strip().lower()
    if norm_id == "random_forest":
        return RandomForestBaseline(random_state=seed, **kwargs)
    if norm_id == "deep_mlp":
        return DeepMLP(seed=seed, **kwargs)
    if norm_id == "cnn_2d":
        return Conv2DCNN(seed=seed, **kwargs)
    if norm_id in ("cnn_gru", "crnn"):
        from src.hybrid_model import CNNGRUHybrid
        return CNNGRUHybrid(seed=seed, **kwargs)
    raise ConfigurationError(
        f"Unsupported competing model_id '{model_id}'. Expected 'random_forest', 'deep_mlp', 'cnn_2d', or 'cnn_gru'."
    )


def build_model(
    model_id: str,
    config: dict[str, Any] | None = None,
    *,
    instantiate: bool = False,
    seed: int = 42,
    **kwargs: Any,
) -> Any:
    """Validate `model_id` against configuration; instantiate only when `instantiate=True`.

    Preserves Step 1 guard behavior when called as `build_model("deep_mlp")` without
    `instantiate=True`, while allowing `build_model("deep_mlp", instantiate=True)` or
    `create_model("deep_mlp")` in Step 4+.
    """
    specs = get_model_specifications(config)
    if model_id not in specs:
        raise ConfigurationError(
            f"Unknown model_id '{model_id}'. Configured models: {sorted(specs.keys())}."
        )
    if not instantiate:
        raise NotImplementedError(
            f"Model '{model_id}' ({specs[model_id].name}) is registered in configuration. "
            "Pass instantiate=True or call create_model() to construct the Step 4 model instance."
        )
    return create_model(model_id=model_id, seed=seed, **kwargs)
