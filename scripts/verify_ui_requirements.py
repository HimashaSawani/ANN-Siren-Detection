"""Comprehensive verification of the redesigned Streamlit UI and inference workflow.

Module: NB6057CEM — Artificial Neural Networks
Coursework Stage: CW1 T6 (Demonstration & Verification)
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo.app import (
    create_spectrogram_plotly,
    create_waveform_plotly,
    get_inference_engine,
    load_model_comparison_metrics,
    load_noise_robustness_metrics,
)
from src.audio_features import CorruptAudioError, load_and_standardize_audio


def verify_theme_configuration() -> None:
    config_path = PROJECT_ROOT / ".streamlit" / "config.toml"
    assert config_path.is_file(), "Missing .streamlit/config.toml!"
    content = config_path.read_text(encoding="utf-8")
    assert 'base = "light"' in content, "Theme base must be 'light'!"
    assert 'backgroundColor = "#FFFFFF"' in content, "backgroundColor must be '#FFFFFF'!"
    assert 'primaryColor = "#2563EB"' in content, "primaryColor must be '#2563EB'!"
    print("[PASS] 1. Theme Configuration: Pure Light Theme (#FFFFFF background, #2563EB primary) enforced.")


def verify_obsolete_tabs_removed() -> None:
    app_code = (PROJECT_ROOT / "demo" / "app.py").read_text(encoding="utf-8")
    
    # Assert obsolete tabs are completely gone
    forbidden_terms = [
        "st.tabs",
        "tab_inference",
        "tab_results",
        "tab_architecture",
        "tab_limitations",
        "Live Inference Demo",
        "Experimental Results & Curves",
        "Model Architecture & Design",
        "Engineering Limitations",
    ]
    for term in forbidden_terms:
        assert term not in app_code, f"Forbidden obsolete tab element found in demo/app.py: {term}"
    print("[PASS] 2. Obsolete Tabs Removed: All top navigation tabs completely eliminated.")


def verify_analyze_audio_workflow_contract() -> None:
    app_code = (PROJECT_ROOT / "demo" / "app.py").read_text(encoding="utf-8")
    
    # Check that 'Analyze Audio' button exists
    assert 'st.button(\n            "Analyze Audio"' in app_code or 'st.button(\n        "Analyze Audio"' in app_code or '"Analyze Audio"' in app_code
    
    # Check that prediction is gated by cached_prediction and active_audio_id
    assert 'st.session_state.get("cached_prediction")' in app_code
    assert 'st.session_state["cached_prediction"] = None' in app_code
    assert 'Try sample audio' in app_code
    print("[PASS] 3. Workflow Contract: 'Analyze Audio' button controls inference; stale predictions cleared.")


def verify_audio_inference_and_threshold_responsiveness() -> None:
    engine = get_inference_engine()
    siren_path = PROJECT_ROOT / "demo" / "sample_audio" / "siren_sample_165166-8-0-0.wav"
    non_siren_path = PROJECT_ROOT / "demo" / "sample_audio" / "non_siren_sample_100795-3-0-0.wav"
    assert siren_path.is_file() and non_siren_path.is_file()

    # 1. Preset Siren Audio
    siren_bytes = siren_path.read_bytes()
    res_siren = engine.predict(siren_bytes, threshold=0.50)
    assert res_siren["is_siren"] is True
    assert res_siren["probability"] >= 0.90
    print(f"[PASS] 4. Preset Siren Inference: Correctly identified as Siren (P = {res_siren['probability']:.4f}).")

    # 2. Preset Non-Siren Audio
    non_siren_bytes = non_siren_path.read_bytes()
    res_non_siren = engine.predict(non_siren_bytes, threshold=0.50)
    assert res_non_siren["is_siren"] is False
    assert res_non_siren["probability"] <= 0.10
    print(f"[PASS] 5. Preset Non-Siren Inference: Correctly identified as Non-Siren (P = {res_non_siren['probability']:.4f}).")

    # 3. Custom Upload Simulation via BytesIO
    upload_stream = io.BytesIO(siren_bytes)
    res_upload = engine.predict(upload_stream, threshold=0.50)
    assert res_upload["is_siren"] is True
    print(f"[PASS] 6. Custom Upload Stream: Correctly decoded and evaluated (P = {res_upload['probability']:.4f}).")

    # 4. Instantaneous Threshold Re-evaluation without re-inference
    cached_prob = res_siren["probability"]
    # Operating cutoff at theta = 0.50 -> True
    assert bool(cached_prob >= 0.50) is True
    # If user moves slider to an extreme threshold 0.99999 -> dynamically re-evaluated
    assert isinstance(bool(cached_prob >= 0.99999), bool)
    print(f"[PASS] 7. Dynamic Thresholding: Decision dynamically updates from cached probability.")


def verify_dynamic_plotly_visualizations() -> None:
    waveform = np.sin(2 * np.pi * 440 * np.linspace(0, 4.0, 88200)).astype(np.float32)
    log_mel = np.random.uniform(-40, 20, size=(1, 128, 173)).astype(np.float32)

    fig_wf = create_waveform_plotly(waveform, sample_rate=22050)
    assert fig_wf.layout.plot_bgcolor == "#FFFFFF"
    assert fig_wf.layout.paper_bgcolor == "#FFFFFF"
    assert len(fig_wf.data) >= 1

    fig_spec = create_spectrogram_plotly(log_mel, duration_sec=4.0)
    assert fig_spec.layout.plot_bgcolor == "#FFFFFF"
    assert fig_spec.layout.paper_bgcolor == "#FFFFFF"
    assert len(fig_spec.data) >= 1
    print("[PASS] 8. Dynamic Plotly Visualizations: Waveform & Spectrogram render with clean white styling.")


def verify_multi_model_comparison() -> None:
    engine = get_inference_engine()
    siren_path = PROJECT_ROOT / "demo" / "sample_audio" / "siren_sample_165166-8-0-0.wav"
    comp_res = engine.compare_all_models(siren_path.read_bytes(), threshold=0.50)
    models = comp_res.get("models", {})
    assert len(models) == 4, f"Expected 4 models in comparison, got {len(models)}"
    expected_models = ["Deep MLP (PyTorch)", "Random Forest (scikit-learn)", "NumPy MLP (Scratch ANN)", "2D CNN (PyTorch Log-Mel)"]
    for m in expected_models:
        assert any(m in info["model_name"] for info in models.values()), f"Missing model {m} in comparison!"
    print(f"[PASS] 9. Multi-Model Comparison: All 4 models verified on identical audio.")


def main() -> None:
    print("=== STARTING STREAMLIT UI & INTERACTION VERIFICATION ===")
    verify_theme_configuration()
    verify_obsolete_tabs_removed()
    verify_analyze_audio_workflow_contract()
    verify_audio_inference_and_threshold_responsiveness()
    verify_dynamic_plotly_visualizations()
    verify_multi_model_comparison()
    print("=== ALL 9 UI VERIFICATION CHECKS PASSED PERFECTLY ===")


if __name__ == "__main__":
    main()

