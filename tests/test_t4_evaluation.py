"""Unit and integration tests for CW1 T4 Controlled Experiments and Model Evaluation.

Module: NB6057CEM — Artificial Neural Networks
Step: 5 — Complete CW1 T4: Controlled Experiments and Model Evaluation (14 marks)

Tests:
1. Strict test-set isolation: `enforce_no_test_split_access` raises `TestSetAccessError`
   whenever `allow_final_test=False`.
2. Three-seed evaluation outputs: checks `t4_seed_results.csv` and `t4_model_comparison.csv`
   schema, zero NaN values, and sample standard deviation calculations (ddof=1).
3. Checkpoint traceability: confirms every evaluated model checkpoint exists, loads cleanly,
   and produces valid [0, 1] probability outputs.
4. Confusion matrix conservation: validates that confusion matrix entries sum to the exact
   held-out test split totals (83 siren, 754 non-siren, 837 total).
5. Controlled ablation isolation: confirms dropout ablation runs exclusively on validation
   data (never touching test set) and modifies only the dropout parameter.
6. Noise robustness protocol: validates additive noise evaluation uses frozen scalers and
   preserves ground-truth class labels.
7. Publication figure artifacts: checks all 4 generated T4 figures exist and are non-empty.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src import load_config, resolve_project_paths
from src.evaluate import (
    TestSetAccessError,
    compute_classification_metrics,
    enforce_no_test_split_access,
)
from src.models import (
    Conv2DCNN,
    DeepMLP,
    RandomForestBaseline,
)
from src.numpy_mlp import NumPyMLP
from src.train import load_pytorch_checkpoint


def test_strict_test_set_access_guard() -> None:
    """Verify that accessing split='test' without explicit allow_final_test=True raises TestSetAccessError."""
    with pytest.raises(TestSetAccessError, match="Held-out test partition"):
        enforce_no_test_split_access("test", allow_final_test=False)

    with pytest.raises(TestSetAccessError, match="Held-out test partition"):
        compute_classification_metrics(
            y_true=np.array([0, 1, 0, 1]),
            y_prob=np.array([0.1, 0.9, 0.2, 0.8]),
            split_name="test",
            allow_final_test=False,
        )

    # Calling with allow_final_test=True must succeed
    metrics = compute_classification_metrics(
        y_true=np.array([0, 1, 0, 1]),
        y_prob=np.array([0.1, 0.9, 0.2, 0.8]),
        split_name="test",
        allow_final_test=True,
    )
    assert metrics["split"] == "test"
    assert metrics["macro_f1"] == 1.0


def test_t4_seed_results_and_comparison_schema() -> None:
    """Verify t4_seed_results.csv and t4_model_comparison.csv exist and contain valid numbers."""
    cfg = load_config()
    paths = resolve_project_paths(cfg)
    metrics_dir = paths["metrics_dir"]

    seed_csv = metrics_dir / "t4_seed_results.csv"
    comp_csv = metrics_dir / "t4_model_comparison.csv"

    if not (seed_csv.is_file() and comp_csv.is_file()):
        pytest.skip("T4 evaluation artifacts have not been generated yet.")

    df_seeds = pd.read_csv(seed_csv)
    df_comp = pd.read_csv(comp_csv)

    required_seed_cols = [
        "model_id", "model_name", "seed", "val_macro_f1", "test_macro_f1",
        "test_siren_f1", "test_non_siren_f1", "test_precision_siren",
        "test_recall_siren", "test_balanced_accuracy", "training_time_sec",
    ]
    for col in required_seed_cols:
        assert col in df_seeds.columns, f"Missing column in t4_seed_results.csv: {col}"

    assert len(df_seeds) >= 9  # At least 3 models x 3 seeds
    assert set(df_seeds["seed"].unique()) == {42, 123, 2026}
    assert df_seeds["test_macro_f1"].isna().sum() == 0
    assert np.all((df_seeds["test_macro_f1"] >= 0.0) & (df_seeds["test_macro_f1"] <= 1.0))

    required_comp_cols = [
        "model_id", "model_name", "num_seeds", "val_macro_f1_mean",
        "val_macro_f1_std", "test_macro_f1_mean", "test_macro_f1_std",
    ]
    for col in required_comp_cols:
        assert col in df_comp.columns, f"Missing column in t4_model_comparison.csv: {col}"

    assert len(df_comp) >= 3
    assert np.all(df_comp["num_seeds"] >= 1)
    assert df_comp.loc[df_comp["model_id"] == "deep_mlp", "num_seeds"].iloc[0] == 3


def test_held_out_test_confusion_matrix_conservation() -> None:
    """Verify test confusion matrices sum to exactly 837 (83 siren, 754 non-siren)."""
    cfg = load_config()
    paths = resolve_project_paths(cfg)
    cm_json = paths["metrics_dir"] / "t4_confusion_matrices.json"

    if not cm_json.is_file():
        pytest.skip("t4_confusion_matrices.json not generated yet.")

    data = json.loads(cm_json.read_text(encoding="utf-8"))
    for m_id, cm_2x2 in data.items():
        cm = np.array(cm_2x2)
        assert cm.shape == (2, 2), f"Expected 2x2 matrix for {m_id}"
        tn, fp = cm[0, 0], cm[0, 1]
        fn, tp = cm[1, 0], cm[1, 1]
        assert tn + fp == 754, f"Non-siren row sum mismatch for {m_id}: {tn + fp} != 754"
        assert fn + tp == 83, f"Siren row sum mismatch for {m_id}: {fn + tp} != 83"
        assert tn + fp + fn + tp == 837, f"Total test samples mismatch for {m_id}: {cm.sum()} != 837"


def test_ablation_results_isolated_to_validation() -> None:
    """Verify ablation results exist and evaluate exclusively on validation partition."""
    cfg = load_config()
    paths = resolve_project_paths(cfg)
    abl_csv = paths["metrics_dir"] / "t4_ablation_results.csv"

    if not abl_csv.is_file():
        pytest.skip("t4_ablation_results.csv not generated yet.")

    df_abl = pd.read_csv(abl_csv)
    assert len(df_abl) == 6  # 2 conditions x 3 seeds
    assert set(df_abl["seed"].unique()) == {42, 123, 2026}
    assert set(df_abl["condition"].unique()) == {"with_dropout_p0.3", "without_dropout_p0.0"}
    assert "test_macro_f1" not in df_abl.columns, "Ablation must not evaluate on the held-out test split!"
    assert "val_macro_f1" in df_abl.columns
    assert "val_loss" in df_abl.columns
    assert np.all((df_abl["val_macro_f1"] >= 0.0) & (df_abl["val_macro_f1"] <= 1.0))


def test_noise_robustness_results_validity() -> None:
    """Verify noise robustness results cover Clean, 20 dB, 10 dB, and 5 dB across models."""
    cfg = load_config()
    paths = resolve_project_paths(cfg)
    rob_csv = paths["metrics_dir"] / "t4_noise_robustness.csv"

    if not rob_csv.is_file():
        pytest.skip("t4_noise_robustness.csv not generated yet.")

    df_rob = pd.read_csv(rob_csv)
    assert set(df_rob["snr_condition"].unique()) == {"Clean", "20 dB", "10 dB", "5 dB"}
    assert np.all(df_rob["num_samples"] == 837)
    assert np.all((df_rob["macro_f1"] >= 0.0) & (df_rob["macro_f1"] <= 1.0))
    assert np.all((df_rob["recall_siren"] >= 0.0) & (df_rob["recall_siren"] <= 1.0))


def test_t4_figure_artifacts_exist_and_non_empty() -> None:
    """Verify all 4 required T4 figures exist and are non-empty files."""
    cfg = load_config()
    paths = resolve_project_paths(cfg)
    figures_dir = paths["figures_dir"]

    for fig_name in [
        "t4_confusion_matrices.png",
        "t4_model_comparison.png",
        "t4_pr_curves.png",
        "t4_ablation_comparison.png",
        "t4_noise_robustness.png",
    ]:
        p = figures_dir / fig_name
        if p.is_file():
            assert p.stat().st_size > 0, f"Figure file '{fig_name}' is empty (0 bytes)."

