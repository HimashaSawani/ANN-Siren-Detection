"""Shared Training Infrastructure, Early Stopping, and Checkpoint Manager (CW1 T3).

Module: NB6057CEM — Artificial Neural Networks
Step: 4 — Implement and Train Three Competing Models (CW1 T3)

Provides:
- `compute_training_pos_weight`: Training-only positive class weight calculation for `BCEWithLogitsLoss`.
- `EarlyStoppingTracker`: Validation-based early stopping and best-state tracking.
- `save_pytorch_checkpoint` / `load_pytorch_checkpoint`: Traceable checkpoint serialization.
- `train_pytorch_model`: Unified mini-batch trainer for `DeepMLP` and `Conv2DCNN`.
- `tune_and_select_random_forest`: Validation macro-F1 tuning for `RandomForestBaseline`.
- `tune_and_select_pytorch_model`: Validation macro-F1 tuning for `DeepMLP` and `Conv2DCNN`.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src import (
    DeviceInfo,
    load_config,
    resolve_project_paths,
    select_device,
    set_random_seeds,
)
from src.evaluate import compute_classification_metrics, enforce_no_test_split_access
from src.models import Conv2DCNN, DeepMLP, RandomForestBaseline


def validate_training_readiness(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Verify configuration, seeds, paths, and compute device readiness without training."""
    cfg = config if config is not None else load_config()
    repro_cfg = cfg["reproducibility"]
    hw_cfg = cfg["hardware"]
    paths = resolve_project_paths(cfg)

    default_seed = int(repro_cfg.get("default_seed", 42))
    seed_state = set_random_seeds(
        seed=default_seed,
        deterministic_cudnn=bool(repro_cfg.get("deterministic_cudnn", True)),
    )
    device_info: DeviceInfo = select_device(
        preferred=str(hw_cfg.get("preferred_device", "auto")),
        allow_gpu=bool(hw_cfg.get("allow_gpu", True)),
        fallback_to_cpu=bool(hw_cfg.get("fallback_to_cpu", True)),
    )

    return {
        "training_executed": False,
        "default_seed": default_seed,
        "configured_seeds": list(repro_cfg.get("random_seeds", [])),
        "seed_state": seed_state,
        "selected_device": device_info.selected_device,
        "cuda_available": device_info.cuda_available,
        "checkpoints_dir": paths["checkpoints_dir"],
    }


def compute_training_pos_weight(
    y_train: np.ndarray | torch.Tensor,
    split_name: str = "train",
) -> float:
    """Compute positive class weight (`num_neg / num_pos`) strictly from training labels.

    Args:
        y_train: Binary labels `{0, 1}` from the training split.
        split_name: Must be `"train"`; raises `ValueError` if `"val"` or `"test"`.

    Returns:
        Positive float ratio `N_0 / N_1`.
    """
    norm_split = split_name.strip().lower()
    enforce_no_test_split_access(norm_split, allow_final_test=False)
    if norm_split != "train":
        raise ValueError(
            f"Leakage violation: class weights must be calculated from split='train' only, "
            f"got split='{split_name}'."
        )

    if isinstance(y_train, torch.Tensor):
        arr = y_train.detach().cpu().numpy().reshape(-1)
    else:
        arr = np.asarray(y_train).reshape(-1)

    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("Training labels must be non-empty and finite.")
    if not np.all(np.isin(arr, [0, 1])):
        raise ValueError("Training labels must be binary values in {0, 1}.")

    num_pos = int(np.sum(arr == 1))
    num_neg = int(np.sum(arr == 0))
    if num_pos == 0 or num_neg == 0:
        raise ValueError(
            f"Training labels must contain both positive and negative examples "
            f"(found num_neg={num_neg}, num_pos={num_pos})."
        )
    return float(num_neg) / float(num_pos)


@dataclass
class EarlyStoppingTracker:
    """Validation-based early stopping and best-checkpoint state tracker.

    Selects the best epoch primarily by `val_macro_f1` (higher is better), using
    `val_loss` (lower is better) when `val_macro_f1` ties within `min_delta`.
    """

    patience: int = 10
    min_delta: float = 1e-4
    best_epoch: int = 0
    best_val_macro_f1: float = -1.0
    best_val_loss: float = float("inf")
    best_val_metrics: dict[str, Any] = field(default_factory=dict)
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_without_improvement: int = 0
    should_stop: bool = False

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError(f"patience must be positive, got {self.patience}.")
        if self.min_delta < 0.0:
            raise ValueError(f"min_delta must be non-negative, got {self.min_delta}.")

    def update(
        self,
        epoch: int,
        val_loss: float,
        val_metrics: dict[str, Any],
        model: nn.Module,
    ) -> bool:
        """Record validation metrics for `epoch`; return True if this epoch is the new best."""
        val_f1 = float(val_metrics["macro_f1"])

        f1_improved = val_f1 > (self.best_val_macro_f1 + self.min_delta)
        f1_tied_and_loss_improved = (
            abs(val_f1 - self.best_val_macro_f1) <= self.min_delta
            and val_loss < (self.best_val_loss - self.min_delta)
        )

        if self.best_state_dict is None or f1_improved or f1_tied_and_loss_improved:
            self.best_epoch = int(epoch)
            self.best_val_macro_f1 = val_f1
            self.best_val_loss = float(val_loss)
            self.best_val_metrics = copy.deepcopy(val_metrics)
            self.best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            self.epochs_without_improvement = 0
            return True

        self.epochs_without_improvement += 1
        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True
        return False

    def restore_best_weights(self, model: nn.Module) -> None:
        """Restore the best recorded validation weights into `model`."""
        if self.best_state_dict is None:
            raise RuntimeError("No best_state_dict recorded in EarlyStoppingTracker.")
        model.load_state_dict(self.best_state_dict)


def save_pytorch_checkpoint(
    model: DeepMLP | Conv2DCNN,
    checkpoint_path: str | Path,
    training_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Save a PyTorch model checkpoint (`.pt`) and companion metadata JSON (`.pt.meta.json`)."""
    target = Path(checkpoint_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model, DeepMLP):
        model_id = "deep_mlp"
        arch_config = {
            "input_dim": model.input_dim,
            "hidden_dims": list(model.hidden_dims),
            "output_dim": model.output_dim,
            "dropout": model.dropout_rate,
            "seed": model.seed,
        }
    elif isinstance(model, Conv2DCNN):
        model_id = "cnn_2d"
        arch_config = {
            "in_channels": model.in_channels,
            "conv_channels": list(model.conv_channels),
            "n_mels": model.n_mels,
            "time_frames": model.time_frames,
            "dropout": model.dropout_rate,
            "output_dim": model.output_dim,
            "seed": model.seed,
        }
    else:
        raise TypeError(f"Unsupported model class for checkpointing: {type(model).__name__}")

    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload = {
        "model_id": model_id,
        "framework": "pytorch",
        "architecture_config": arch_config,
        "trainable_parameters": model.count_trainable_parameters(),
        "state_dict": cpu_state,
        "training_metadata": training_metadata,
    }
    torch.save(payload, target)
    size_bytes = int(target.stat().st_size)

    meta_json = {
        "model_id": model_id,
        "framework": "pytorch",
        "checkpoint_path": str(target),
        "model_size_bytes": size_bytes,
        "trainable_parameters": model.count_trainable_parameters(),
        "architecture_config": arch_config,
        "training_metadata": training_metadata,
    }
    meta_path = target.with_suffix(target.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta_json, indent=2), encoding="utf-8")
    return meta_json


def load_pytorch_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[DeepMLP | Conv2DCNN, dict[str, Any]]:
    """Reconstruct a `DeepMLP` or `Conv2DCNN` from a saved `.pt` checkpoint."""
    target = Path(checkpoint_path)
    if not target.is_file():
        raise FileNotFoundError(f"PyTorch checkpoint not found at '{target}'.")

    payload = torch.load(target, map_location=device, weights_only=False)
    model_id = str(payload["model_id"])
    arch = payload["architecture_config"]

    if model_id == "deep_mlp":
        model: DeepMLP | Conv2DCNN = DeepMLP(
            input_dim=int(arch["input_dim"]),
            hidden_dims=tuple(arch["hidden_dims"]),
            output_dim=int(arch["output_dim"]),
            dropout=float(arch["dropout"]),
            seed=arch.get("seed", 42),
        )
    elif model_id == "cnn_2d":
        model = Conv2DCNN(
            in_channels=int(arch["in_channels"]),
            conv_channels=tuple(arch["conv_channels"]),
            n_mels=int(arch["n_mels"]),
            time_frames=int(arch["time_frames"]),
            dropout=float(arch["dropout"]),
            output_dim=int(arch["output_dim"]),
            seed=arch.get("seed", 42),
        )
    elif model_id in ("cnn_gru", "crnn"):
        from src.hybrid_model import CNNGRUHybrid
        model = CNNGRUHybrid(
            in_channels=int(arch.get("in_channels", 1)),
            conv_channels=tuple(arch.get("conv_channels", (16, 32, 64))),
            gru_hidden_size=int(arch.get("gru_hidden_size", 64)),
            gru_num_layers=int(arch.get("gru_num_layers", 1)),
            bidirectional=bool(arch.get("bidirectional", True)),
            n_mels=int(arch.get("n_mels", 128)),
            time_frames=int(arch.get("time_frames", 173)),
            dropout=float(arch.get("dropout", 0.3)),
            output_dim=int(arch.get("output_dim", 1)),
            seed=arch.get("seed", 42),
        )
    else:
        raise ValueError(f"Unknown model_id '{model_id}' inside checkpoint '{target}'.")

    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def make_tensor_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Wrap pre-extracted numpy arrays into deterministic train and validation DataLoaders."""
    x_tr_t = torch.from_numpy(np.ascontiguousarray(X_train, dtype=np.float32))
    y_tr_t = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.int64).reshape(-1))
    x_va_t = torch.from_numpy(np.ascontiguousarray(X_val, dtype=np.float32))
    y_va_t = torch.from_numpy(np.ascontiguousarray(y_val, dtype=np.int64).reshape(-1))

    gen = torch.Generator()
    gen.manual_seed(int(seed))

    train_loader = DataLoader(
        TensorDataset(x_tr_t, y_tr_t),
        batch_size=int(batch_size),
        shuffle=True,
        generator=gen,
    )
    val_loader = DataLoader(
        TensorDataset(x_va_t, y_va_t),
        batch_size=int(batch_size),
        shuffle=False,
    )
    return train_loader, val_loader


def _extract_labels_from_loader(loader: DataLoader) -> np.ndarray:
    labels: list[np.ndarray] = []
    for batch in loader:
        y_b = batch[1]
        labels.append(y_b.detach().cpu().numpy().reshape(-1))
    return np.concatenate(labels, axis=0)


@torch.no_grad()
def evaluate_pytorch_model_on_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str = "val",
    allow_final_test: bool = False,
) -> tuple[float, dict[str, Any]]:
    """Evaluate a PyTorch model deterministically on a DataLoader."""
    enforce_no_test_split_access(split_name=split_name, allow_final_test=allow_final_test)
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    try:
        for batch in loader:
            x_b = batch[0].to(device)
            y_b = batch[1].to(device=device, dtype=torch.float32).reshape(-1, 1)
            logits = model(x_b)
            loss = criterion(logits, y_b)
            probs = torch.sigmoid(logits)

            batch_n = int(x_b.shape[0])
            total_loss += float(loss.item()) * batch_n
            total_samples += batch_n
            all_probs.append(probs.detach().cpu().numpy().reshape(-1))
            all_targets.append(y_b.detach().cpu().numpy().reshape(-1))
    finally:
        model.train(was_training)

    mean_loss = total_loss / max(1, total_samples)
    y_true_arr = np.concatenate(all_targets, axis=0).astype(np.int64)
    y_prob_arr = np.concatenate(all_probs, axis=0).astype(np.float64)
    metrics = compute_classification_metrics(
        y_true=y_true_arr,
        y_prob=y_prob_arr,
        split_name=split_name,
        allow_final_test=allow_final_test,
    )
    return float(mean_loss), metrics


def train_pytorch_model(
    model: DeepMLP | Conv2DCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 50,
    patience: int = 10,
    min_delta: float = 1e-4,
    seed: int = 42,
    preferred_device: str = "auto",
    use_train_pos_weight: bool = True,
    checkpoint_path: str | Path | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train `DeepMLP` or `Conv2DCNN` with validation-based early stopping and checkpointing."""
    if learning_rate <= 0.0 or not np.isfinite(learning_rate):
        raise ValueError(f"learning_rate must be positive and finite, got {learning_rate}.")
    if max_epochs <= 0:
        raise ValueError(f"max_epochs must be positive, got {max_epochs}.")

    set_random_seeds(int(seed))
    device_info = select_device(preferred=preferred_device)
    device = device_info.torch_device
    model.to(device)

    y_train_all = _extract_labels_from_loader(train_loader)
    pos_weight_val = (
        compute_training_pos_weight(y_train_all, split_name="train")
        if use_train_pos_weight
        else 1.0
    )
    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )

    tracker = EarlyStoppingTracker(patience=patience, min_delta=min_delta)
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    val_macro_f1_history: list[float] = []

    t0 = time.perf_counter()
    actual_epochs = 0

    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        running_loss = 0.0
        seen_samples = 0

        for batch in train_loader:
            x_b = batch[0].to(device)
            y_b = batch[1].to(device=device, dtype=torch.float32).reshape(-1, 1)

            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss encountered at epoch {epoch}.")
            loss.backward()
            optimizer.step()

            batch_n = int(x_b.shape[0])
            running_loss += float(loss.item()) * batch_n
            seen_samples += batch_n

        epoch_train_loss = running_loss / max(1, seen_samples)
        epoch_val_loss, val_metrics = evaluate_pytorch_model_on_loader(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            split_name="val",
            allow_final_test=False,
        )

        train_loss_history.append(float(epoch_train_loss))
        val_loss_history.append(float(epoch_val_loss))
        val_macro_f1_history.append(float(val_metrics["macro_f1"]))
        actual_epochs = epoch

        tracker.update(
            epoch=epoch,
            val_loss=epoch_val_loss,
            val_metrics=val_metrics,
            model=model,
        )
        if tracker.should_stop:
            break

    training_duration_sec = round(float(time.perf_counter() - t0), 6)
    tracker.restore_best_weights(model)
    model.eval()

    training_meta = {
        "seed": int(seed),
        "device": device_info.selected_device,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "max_epochs": int(max_epochs),
        "actual_epochs_completed": int(actual_epochs),
        "early_stopping_patience": int(patience),
        "early_stopping_triggered": bool(tracker.should_stop),
        "best_epoch": int(tracker.best_epoch),
        "best_val_macro_f1": float(tracker.best_val_macro_f1),
        "best_val_loss": float(tracker.best_val_loss),
        "pos_weight_from_train": float(pos_weight_val),
        "training_duration_sec": training_duration_sec,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "val_macro_f1_history": val_macro_f1_history,
        "best_validation_metrics": tracker.best_val_metrics,
        "extra_metadata": extra_metadata or {},
    }

    saved_ckpt_meta: dict[str, Any] | None = None
    if checkpoint_path is not None:
        saved_ckpt_meta = save_pytorch_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            training_metadata=training_meta,
        )

    return {
        "model": model,
        "training_metadata": training_meta,
        "checkpoint_metadata": saved_ckpt_meta,
    }


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    val_macro_f1s: list[float],
    best_epoch: int,
    model_title: str,
    output_path: str | Path,
) -> Path:
    """Plot and save two-panel training/validation loss and validation macro-F1 curves."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(train_losses) + 1)

    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=150)

    ax_loss.plot(epochs, train_losses, color="#1f77b4", linewidth=2.0, label="Train BCE Loss")
    ax_loss.plot(epochs, val_losses, color="#d62728", linewidth=2.0, label="Validation BCE Loss")
    if 1 <= best_epoch <= len(epochs):
        ax_loss.axvline(
            best_epoch,
            color="#2ca02c",
            linestyle="--",
            linewidth=1.6,
            label=f"Selected Best Checkpoint (Epoch {best_epoch})",
        )
    ax_loss.set_title(f"{model_title} — Loss Curves", fontsize=12, fontweight="bold")
    ax_loss.set_xlabel("Epoch", fontsize=10)
    ax_loss.set_ylabel("Weighted Binary Cross-Entropy Loss", fontsize=10)
    ax_loss.grid(True, linestyle="--", alpha=0.4)
    ax_loss.legend(loc="upper right", frameon=True)

    ax_f1.plot(epochs, val_macro_f1s, color="#2ca02c", linewidth=2.0, label="Validation Macro-F1")
    if 1 <= best_epoch <= len(epochs):
        ax_f1.axvline(
            best_epoch,
            color="#d62728",
            linestyle="--",
            linewidth=1.6,
            label=f"Best Epoch ({best_epoch}): Macro-F1={val_macro_f1s[best_epoch - 1]:.4f}",
        )
    ax_f1.set_title(f"{model_title} — Validation Macro-F1", fontsize=12, fontweight="bold")
    ax_f1.set_xlabel("Epoch", fontsize=10)
    ax_f1.set_ylabel("Validation Macro-F1", fontsize=10)
    ax_f1.set_ylim(0.0, 1.05)
    ax_f1.grid(True, linestyle="--", alpha=0.4)
    ax_f1.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)
    return target


def tune_and_select_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    candidates: list[dict[str, Any]],
    seed: int = 42,
    checkpoint_path: str | Path | None = None,
    scaler_ref: str | None = None,
) -> dict[str, Any]:
    """Fit candidate Random Forest configurations on train and select best by validation macro-F1."""
    if not candidates:
        raise ValueError("candidates list for Random Forest tuning must not be empty.")

    trial_records: list[dict[str, Any]] = []
    best_model: RandomForestBaseline | None = None
    best_f1 = -1.0
    best_trial: dict[str, Any] = {}

    for idx, cand in enumerate(candidates):
        cfg_id = str(cand.get("config_id", f"rf_cand_{idx + 1}"))
        rf = RandomForestBaseline(
            n_estimators=int(cand.get("n_estimators", 200)),
            max_depth=cand.get("max_depth", 16),
            min_samples_split=int(cand.get("min_samples_split", 2)),
            min_samples_leaf=int(cand.get("min_samples_leaf", 2)),
            class_weight=cand.get("class_weight", "balanced"),
            random_state=int(seed),
            n_jobs=int(cand.get("n_jobs", 1)),
            input_dim=int(X_train.shape[1]),
        )
        rf.fit(X_train, y_train, split_name="train")
        val_probs = rf.predict_proba(X_val)
        val_metrics = compute_classification_metrics(
            y_true=y_val,
            y_prob=val_probs,
            split_name="val",
            allow_final_test=False,
        )

        record = {
            "config_id": cfg_id,
            "hyperparameters": rf.get_hyperparameters(),
            "training_duration_sec": rf.training_duration_sec,
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_metrics": val_metrics,
        }
        trial_records.append(record)

        if float(val_metrics["macro_f1"]) > best_f1:
            best_f1 = float(val_metrics["macro_f1"])
            best_model = rf
            best_trial = record

    assert best_model is not None
    ckpt_meta: dict[str, Any] | None = None
    if checkpoint_path is not None:
        ckpt_meta = best_model.save(
            checkpoint_path,
            extra_metadata={
                "selected_config_id": best_trial["config_id"],
                "val_macro_f1": best_f1,
                "preprocessing_scaler_ref": scaler_ref,
                "attempted_candidates_count": len(trial_records),
            },
        )

    return {
        "best_model": best_model,
        "selected_trial": best_trial,
        "all_trials": trial_records,
        "checkpoint_metadata": ckpt_meta,
    }


def run_training(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Dispatch training when arguments are provided; preserve Step 1 guard when called empty."""
    if not args and not kwargs:
        raise NotImplementedError(
            "Model training requires model and DataLoader arguments (Step 1 guard active when called with no arguments)."
        )
    return train_pytorch_model(*args, **kwargs)
