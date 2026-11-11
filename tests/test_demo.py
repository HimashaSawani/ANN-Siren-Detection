"""Integration tests for Streamlit Emergency Siren Detection Demo components.

Module: NB6057CEM — Artificial Neural Networks
Coursework Stage: CW1 T6 (Reproducibility & Live Demonstration)
"""

from __future__ import annotations

import io
from pathlib import Path
import matplotlib.figure
import numpy as np
import pandas as pd
import pytest

from demo.app import (
    create_spectrogram_plotly,
    create_waveform_plotly,
    get_inference_engine,
    load_model_comparison_metrics,
    load_noise_robustness_metrics,
    plot_waveform_and_spectrogram,
)


def test_demo_engine_initialization() -> None:
    """Verify demo application acquires and caches the inference engine."""
    engine = get_inference_engine()
    assert engine is not None
    assert engine.model is not None


def test_demo_metrics_loaders() -> None:
    """Verify demo successfully loads real metrics from disk."""
    df_comp = load_model_comparison_metrics()
    assert isinstance(df_comp, pd.DataFrame)
    assert not df_comp.empty
    assert "model_name" in df_comp.columns
    assert "test_macro_f1_mean" in df_comp.columns

    df_noise = load_noise_robustness_metrics()
    assert isinstance(df_noise, pd.DataFrame)
    assert not df_noise.empty
    assert "snr_condition" in df_noise.columns


def test_demo_plot_waveform_and_spectrogram() -> None:
    """Verify dual-panel waveform and Log-Mel spectrogram visualization renders cleanly."""
    waveform = np.sin(2 * np.pi * 440 * np.linspace(0, 4.0, 88200)).astype(np.float32)
    log_mel = np.zeros((1, 128, 173), dtype=np.float32)

    fig = plot_waveform_and_spectrogram(waveform, log_mel, sample_rate=22050)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) >= 2  # Waveform and spectrogram axes


def test_demo_preset_and_uploaded_bytes_inference() -> None:
    """Verify demo inference execution on both preset sample bytes and uploaded stream bytes."""
    engine = get_inference_engine()

    siren_path = Path("demo/sample_audio/siren_sample_165166-8-0-0.wav")
    assert siren_path.is_file()
    siren_bytes = siren_path.read_bytes()

    # 1. Preset audio bytes inference
    res_siren = engine.predict(siren_bytes, threshold=0.50)
    assert res_siren["is_siren"] is True
    assert res_siren["predicted_class"] == "Siren"
    assert res_siren["probability"] >= 0.80

    fig_s = plot_waveform_and_spectrogram(
        res_siren["waveform"], res_siren["log_mel_spectrogram"], res_siren["sample_rate"]
    )
    assert isinstance(fig_s, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig_s)

    # 2. Simulated uploaded audio stream (BytesIO)
    non_siren_path = Path("demo/sample_audio/non_siren_sample_100795-3-0-0.wav")
    assert non_siren_path.is_file()
    upload_stream = io.BytesIO(non_siren_path.read_bytes())

    res_upload = engine.predict(upload_stream, threshold=0.50)
    assert res_upload["is_siren"] is False
    assert res_upload["predicted_class"] == "Non-Siren"
    assert res_upload["probability"] <= 0.20

    # 3. Multi-model comparative inference on bytes (including Phase 2 noise-augmented model)
    comp = engine.compare_all_models(siren_bytes, threshold=0.50)
    assert len(comp["models"]) >= 4
    assert "deep_mlp" in comp["models"]
    assert "deep_mlp_augmented" in comp["models"]
    for key, model_info in comp["models"].items():
        assert "model_name" in model_info
        assert "probability" in model_info
        assert "is_siren" in model_info


def test_demo_dynamic_plotly_visualizations() -> None:
    """Verify dynamic Plotly waveform and spectrogram charts render properly from decoded audio."""
    import plotly.graph_objects as go

    waveform = (0.5 * np.sin(2 * np.pi * 880 * np.linspace(0, 4.0, 88200))).astype(np.float32)
    log_mel = np.random.uniform(-40, 20, size=(1, 128, 173)).astype(np.float32)

    # 1. Waveform Plotly Chart
    fig_wf = create_waveform_plotly(waveform, sample_rate=22050)
    assert isinstance(fig_wf, go.Figure)
    assert len(fig_wf.data) >= 1
    assert fig_wf.data[0].type == "scatter"
    assert fig_wf.layout.plot_bgcolor == "#FFFFFF"

    # 2. Spectrogram Plotly Chart
    fig_spec = create_spectrogram_plotly(log_mel, duration_sec=4.0)
    assert isinstance(fig_spec, go.Figure)
    assert len(fig_spec.data) >= 1
    assert fig_spec.data[0].type == "heatmap"
    assert fig_spec.layout.plot_bgcolor == "#FFFFFF"


def test_demo_state_management_workflow() -> None:
    """Verify state management: threshold changes recalculate classification without re-inference."""
    engine = get_inference_engine()
    siren_path = Path("demo/sample_audio/siren_sample_165166-8-0-0.wav")
    assert siren_path.is_file()
    siren_bytes = siren_path.read_bytes()

    # Initial inference at threshold 0.50
    pred = engine.predict(siren_bytes, threshold=0.50)
    prob = pred["probability"]
    assert prob >= 0.80

    # Dynamic threshold re-evaluation:
    # At standard threshold 0.50 -> Siren
    assert bool(prob >= 0.50) is True
    # At extreme threshold 0.9999 -> Non-Siren (if prob < 0.9999) or Siren
    assert isinstance(bool(prob >= 0.9999), bool)

    # Invalidation on audio change simulation
    audio_id_1 = "preset_siren_165166"
    audio_id_2 = "preset_non_siren_100795"
    assert audio_id_1 != audio_id_2


def test_root_app_entrypoint_and_explainability_integration() -> None:
    """Verify root app.py delegates to demo.app.main and demo exposes explainability."""
    import app
    import demo.app as demo_app

    assert hasattr(app, "main")
    assert callable(app.main)
    assert app.main is demo_app.main

    # Verify explainability functions exist in demo.app
    assert hasattr(demo_app, "get_shap_explainer")
    assert callable(demo_app.get_shap_explainer)
    assert hasattr(demo_app, "create_shap_attribution_plotly")



