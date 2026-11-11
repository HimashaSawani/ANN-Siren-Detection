"""Evaluation metrics and held-out test set guard for ANN-Siren-Detection (CW1 T3).

Module: NB6057CEM — Artificial Neural Networks
Step: 4 — Implement and Train Three Competing Models (CW1 T3)

Provides:
- Primary metric (`macro-F1`) and supporting binary classification metrics
  (`f1_siren`, `precision_siren`, `recall_siren`, `balanced_accuracy`, `roc_auc`, `pr_auc`).
- Strict `TestSetAccessError` guard preventing accidental use of the held-out test
  partition (`fold 10`) during Step 4 model training and hyperparameter tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import ConfigurationError, load_config, resolve_project_paths


class TestSetAccessError(RuntimeError):
    """Raised when code attempts to access or evaluate the held-out test split prematurely."""


@dataclass(frozen=True)
class EvaluationSpec:
    """Specification of primary and supporting evaluation metrics."""

    primary_metric: str
    supporting_metrics: tuple[str, ...]
    metrics_output_dir: str
    figures_output_dir: str


def get_evaluation_spec(config: dict[str, Any] | None = None) -> EvaluationSpec:
    """Retrieve and validate the configured evaluation metric specifications."""
    cfg = config if config is not None else load_config()
    eval_cfg = cfg.get("evaluation", {})
    primary = str(eval_cfg.get("primary_metric", ""))
    if primary != "macro-F1":
        raise ConfigurationError(
            f"Primary evaluation metric must be 'macro-F1', got '{primary}'."
        )

    supporting = tuple(str(m) for m in eval_cfg.get("supporting_metrics", []))
    paths = resolve_project_paths(cfg)

    return EvaluationSpec(
        primary_metric=primary,
        supporting_metrics=supporting,
        metrics_output_dir=str(paths["metrics_dir"]),
        figures_output_dir=str(paths["figures_dir"]),
    )


def enforce_no_test_split_access(split_name: str, allow_final_test: bool = False) -> None:
    """Raise `TestSetAccessError` if `split_name` is `'test'` and `allow_final_test` is False."""
    norm = split_name.strip().lower()
    if norm == "test" and not allow_final_test:
        raise TestSetAccessError(
            "Held-out test partition ('test' / fold 10) access is strictly prohibited "
            "during Step 4 model training and validation-based hyperparameter selection."
        )


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    split_name: str = "val",
    allow_final_test: bool = False,
) -> dict[str, Any]:
    """Compute comprehensive binary siren classification metrics.

    Args:
        y_true: Ground-truth binary labels `{0, 1}` of shape `(N,)` or `(N, 1)`.
        y_prob: Predicted positive-class (`siren = 1`) probabilities in `[0, 1]`.
        threshold: Decision threshold (default: `0.5`).
        split_name: Partition name (`"train"`, `"val"`, or `"test"`).
        allow_final_test: Must be `False` during Step 4 development/tuning.

    Returns:
        Dictionary with `macro-F1` (`macro_f1`), `f1_siren`, `precision_siren`,
        `recall_siren`, `balanced_accuracy`, `roc_auc`, `pr_auc`, and `confusion_matrix`.
    """
    enforce_no_test_split_access(split_name=split_name, allow_final_test=allow_final_test)

    yt = np.asarray(y_true, dtype=np.int64).reshape(-1)
    yp_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if yt.shape != yp_prob.shape or yt.size == 0:
        raise ValueError(
            f"Shape mismatch or empty arrays in compute_classification_metrics: "
            f"y_true={yt.shape}, y_prob={yp_prob.shape}"
        )
    if not (np.all(np.isfinite(yt)) and np.all(np.isfinite(yp_prob))):
        raise ValueError("NaN or Inf detected in evaluation inputs.")
    if not np.all(np.isin(yt, [0, 1])):
        raise ValueError("y_true must contain binary labels in {0, 1}.")

    yp_pred = (yp_prob >= threshold).astype(np.int64)

    macro_f1 = float(f1_score(yt, yp_pred, average="macro", zero_division=0))
    f1_siren = float(f1_score(yt, yp_pred, pos_label=1, average="binary", zero_division=0))
    f1_non_siren = float(f1_score(yt, yp_pred, pos_label=0, average="binary", zero_division=0))
    prec_siren = float(precision_score(yt, yp_pred, pos_label=1, average="binary", zero_division=0))
    rec_siren = float(recall_score(yt, yp_pred, pos_label=1, average="binary", zero_division=0))
    bal_acc = float(balanced_accuracy_score(yt, yp_pred))
    acc = float(np.mean(yt == yp_pred))

    cm = confusion_matrix(yt, yp_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    has_both_classes = len(np.unique(yt)) == 2
    roc_auc: float | None = None
    pr_auc: float | None = None
    if has_both_classes:
        roc_auc = float(roc_auc_score(yt, yp_prob))
        pr_auc = float(average_precision_score(yt, yp_prob))

    return {
        "split": split_name,
        "num_samples": int(yt.size),
        "macro-F1": macro_f1,
        "macro_f1": macro_f1,
        "f1_siren": f1_siren,
        "f1_non_siren": f1_non_siren,
        "precision_siren": prec_siren,
        "recall_siren": rec_siren,
        "balanced_accuracy": bal_acc,
        "accuracy": acc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix_2x2": [[tn, fp], [fn, tp]],
        },
    }


def run_evaluation(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run evaluation when arguments are provided; preserve Step 1 guard when called empty."""
    if not args and not kwargs:
        raise NotImplementedError(
            "Model evaluation requires y_true and y_prob inputs (Step 1 guard active when called with no arguments)."
        )
    return compute_classification_metrics(*args, **kwargs)
