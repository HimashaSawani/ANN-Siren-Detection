"""Unit and integration tests for Evidence-Based SHAP Explainability.

Module: NB6057CEM — Artificial Neural Networks
Stage: Portfolio Upgrade Phase 1 (Explainable AI / SHAP Attributions)
"""

from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import pytest
import torch

from src.explainability import (
    SirenSHAPExplainer,
    create_shap_attribution_plotly,
    describe_mfcc_feature,
    explain_audio_prediction,
    generate_human_interpretation,
    get_mfcc_feature_names,
    is_shap_available,
)
from src.inference import SirenInferenceEngine


def test_shap_availability() -> None:
    """Verify that SHAP is installed and detected as available."""
    assert is_shap_available() is True


def test_feature_names_and_descriptions() -> None:
    """Verify 80-D MFCC feature naming contract and semantic descriptors."""
    names = get_mfcc_feature_names(n_mfcc=40)
    assert len(names) == 80
    assert names[0] == "mfcc_mean_0"
    assert names[39] == "mfcc_mean_39"
    assert names[40] == "mfcc_std_0"
    assert names[79] == "mfcc_std_39"

    # Test semantic descriptors
    desc_mean0 = describe_mfcc_feature("mfcc_mean_0")
    assert "broadband power" in desc_mean0.lower() or "log energy" in desc_mean0.lower()

    desc_mean6 = describe_mfcc_feature("mfcc_mean_6")
    assert "harmonic" in desc_mean6.lower() or "formant" in desc_mean6.lower()

    desc_std10 = describe_mfcc_feature("mfcc_std_10")
    assert "Temporal modulation" in desc_std10 or "variance" in desc_std10

    desc_unknown = describe_mfcc_feature("arbitrary_unknown")
    assert "Unknown" in desc_unknown or "descriptor" in desc_unknown


def test_shap_explainer_initialization() -> None:
    """Verify SirenSHAPExplainer loads Deep MLP and background samples correctly."""
    explainer = SirenSHAPExplainer(num_background_samples=50, random_seed=42)
    assert explainer.model is not None
    assert explainer.model.training is False  # Must be in eval mode
    assert explainer.background.shape == (50, 80)
    assert np.all(np.isfinite(explainer.background))
    assert isinstance(explainer.base_value, float)
    assert np.isfinite(explainer.base_value)


def test_shap_explanation_additivity() -> None:
    """Verify exact SHAP efficiency / additivity property on Deep MLP logits:

    sum(phi_i) == logit(x) - base_value
    """
    explainer = SirenSHAPExplainer(num_background_samples=50, random_seed=42)
    rng = np.random.RandomState(123)
    dummy_x = rng.randn(1, 80).astype(np.float32)

    res = explainer.explain_features(dummy_x, top_k=10)
    shap_vals = res["shap_values"]
    assert shap_vals.shape == (80,)

    sum_shap = float(np.sum(shap_vals))
    expected_diff = res["model_logit"] - res["base_value"]

    # DeepLIFT attributions on linear layers + ReLU satisfy additivity with tight tolerance
    assert abs(sum_shap - expected_diff) < 1e-4
    assert res["explanation_space"] == "logit"


def test_shap_siren_audio_explanation() -> None:
    """Verify local SHAP attribution on an authentic held-out emergency siren clip."""
    explainer = SirenSHAPExplainer(num_background_samples=50, random_seed=42)
    siren_path = Path("demo/sample_audio/siren_sample_165166-8-0-0.wav")
    assert siren_path.is_file()

    res = explainer.explain_audio(siren_path, top_k=10)

    assert res["predicted_class"] == "Siren"
    assert res["is_siren"] is True
    assert res["estimated_probability"] >= 0.90
    assert len(res["top_features"]) == 10

    # Verify descending sort order by absolute magnitude
    magnitudes = [abs(f["attribution"]) for f in res["top_features"]]
    for i in range(len(magnitudes) - 1):
        assert magnitudes[i] >= magnitudes[i + 1]

    # Verify at least one positive attribution pushing toward siren
    pos_feats = [f for f in res["top_features"] if f["is_positive"]]
    assert len(pos_feats) > 0
    assert any("mfcc" in f["feature_name"] for f in pos_feats)

    # Verify caveats exist and do not claim causality
    assert len(res["caveats"]) >= 3
    assert any("causation" in c.lower() for c in res["caveats"])


def test_shap_non_siren_audio_explanation() -> None:
    """Verify local SHAP attribution on an authentic held-out non-siren clip."""
    explainer = SirenSHAPExplainer(num_background_samples=50, random_seed=42)
    non_siren_path = Path("demo/sample_audio/non_siren_sample_100795-3-0-0.wav")
    assert non_siren_path.is_file()

    res = explainer.explain_audio(non_siren_path, top_k=10)

    assert res["predicted_class"] == "Non-Siren"
    assert res["is_siren"] is False
    assert res["estimated_probability"] <= 0.10
    assert len(res["top_features"]) == 10

    # Negative attribution should push toward Non-Siren
    neg_feats = [f for f in res["top_features"] if not f["is_positive"]]
    assert len(neg_feats) > 0


def test_shap_input_validation() -> None:
    """Verify robust error handling for invalid input dimensions and non-finite values."""
    explainer = SirenSHAPExplainer(num_background_samples=50, random_seed=42)

    # Invalid feature dimension
    with pytest.raises(ValueError, match="Expected input shape"):
        explainer.explain_features(np.zeros((1, 50), dtype=np.float32))

    # NaN / Inf in features
    nan_features = np.zeros((1, 80), dtype=np.float32)
    nan_features[0, 5] = np.nan
    with pytest.raises(ValueError, match="contain NaN or Inf"):
        explainer.explain_features(nan_features)


def test_shap_plotly_chart_generation() -> None:
    """Verify Plotly attribution horizontal bar chart builds cleanly."""
    explainer = SirenSHAPExplainer(num_background_samples=50, random_seed=42)
    siren_path = Path("demo/sample_audio/siren_sample_165166-8-0-0.wav")
    explanation = explainer.explain_audio(siren_path, top_k=10)

    fig = create_shap_attribution_plotly(explanation, top_k=10)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"
    assert fig.data[0].orientation == "h"
    assert len(fig.data[0].y) == 10
    assert fig.layout.plot_bgcolor == "#FFFFFF"
    assert fig.layout.paper_bgcolor == "#FFFFFF"


def test_human_interpretation_synthesis() -> None:
    """Verify evidence-based human interpretation does not assert causation."""
    top_feats = [
        {
            "rank": 1,
            "feature_name": "mfcc_mean_6",
            "attribution": 3.5,
            "normalized_value": 1.4,
            "direction": "Toward Siren (+)",
            "is_positive": True,
            "description": "Mid-frequency harmonic sweeps & formants",
        },
        {
            "rank": 2,
            "feature_name": "mfcc_std_10",
            "attribution": 2.1,
            "normalized_value": 1.1,
            "direction": "Toward Siren (+)",
            "is_positive": True,
            "description": "Temporal modulation variance",
        },
    ]

    interp = generate_human_interpretation(
        prob=0.985,
        top_features=top_feats,
        base_value=-10.83,
        logit=15.2,
    )
    assert "Siren" in interp
    assert "0.9850" in interp
    assert "harmonic" in interp.lower() or "modulation" in interp.lower()
    assert "cause" not in interp.lower()  # Never claim physical causation
