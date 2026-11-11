"""Emergency Siren Detection Web Application.

Module: NB6057CEM — Artificial Neural Networks
Coursework Stage: CW1 T5 (Model Recommendation and Inference) & T6 (Demonstration)
Portfolio Upgrade: Phase 1 — Evidence-Based Explainable AI (SHAP Attributions)

Simple, cohesive, functional audio detection application.
Features user-controlled "Analyze Audio" execution, dynamic Plotly charts,
pure light theme, real multi-model inference, and local SHAP feature attributions.
"""

from __future__ import annotations

import html
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# Ensure persistent, writable Matplotlib cache directory in sandbox
mpl_cache = Path(tempfile.gettempdir()) / "mpl_cache"
mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Setup Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_features import CorruptAudioError
from src.explainability import (
    SirenSHAPExplainer,
    create_shap_attribution_plotly,
    is_shap_available,
)
from src.inference import SirenInferenceEngine


@st.cache_resource(show_spinner=False)
def get_inference_engine(_version: str = "2026.11.10.v6_light_analyze_workflow") -> SirenInferenceEngine:
    """Initialize and cache the production inference engine."""
    from src.inference import SirenInferenceEngine
    return SirenInferenceEngine()


@st.cache_resource(show_spinner=False)
def get_shap_explainer(_version: str = "2026.11.11.v1_shap") -> SirenSHAPExplainer | None:
    """Initialize and cache the SHAP explainer for Deep MLP (optional feature)."""
    if not is_shap_available():
        return None
    try:
        return SirenSHAPExplainer()
    except Exception:
        return None


@st.cache_data
def load_model_comparison_metrics() -> pd.DataFrame:
    """Load genuine multi-seed evaluation metrics from disk."""
    csv_path = PROJECT_ROOT / "results" / "metrics" / "t4_model_comparison.csv"
    if csv_path.is_file():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


@st.cache_data
def load_noise_robustness_metrics() -> pd.DataFrame:
    """Load genuine noise degradation metrics from disk."""
    csv_path = PROJECT_ROOT / "results" / "metrics" / "t4_noise_robustness.csv"
    if csv_path.is_file():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def create_waveform_plotly(
    waveform: np.ndarray,
    sample_rate: int = 22050,
) -> go.Figure:
    """Create a compact, responsive Plotly interactive line chart for audio waveform."""
    n_samples = len(waveform)
    step = max(1, n_samples // 2000)
    sampled_indices = np.arange(0, n_samples, step)
    time_axis = sampled_indices / float(sample_rate)
    sampled_wf = waveform[sampled_indices]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=sampled_wf,
            mode="lines",
            line=dict(color="#2563EB", width=1.1),
            hoverinfo="x+y",
            hovertemplate="Time: %{x:.3f} s<br>Amplitude: %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text="Standardized Waveform (22,050 Hz, 4.0 s Mono)",
            font=dict(size=11, color="#0F172A", family="sans-serif"),
            x=0.01,
            y=0.96,
        ),
        margin=dict(l=35, r=15, t=28, b=26),
        height=165,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        xaxis=dict(
            title=dict(text="Time (seconds)", font=dict(size=9.5, color="#64748B")),
            range=[0.0, float(n_samples / sample_rate)],
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=True,
            zerolinecolor="#E2E8F0",
            tickfont=dict(size=8.5, color="#64748B"),
        ),
        yaxis=dict(
            title=dict(text="Amplitude", font=dict(size=9.5, color="#64748B")),
            range=[-1.05, 1.05],
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=True,
            zerolinecolor="#E2E8F0",
            tickfont=dict(size=8.5, color="#64748B"),
        ),
    )
    return fig


def create_spectrogram_plotly(
    log_mel: np.ndarray,
    duration_sec: float = 4.0,
) -> go.Figure:
    """Create a compact, responsive Plotly interactive heatmap for Log-Mel spectrogram."""
    mel_2d = log_mel.squeeze()
    n_mels, n_frames = mel_2d.shape
    time_axis = np.linspace(0.0, duration_sec, n_frames)
    mel_axis = np.arange(n_mels)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=mel_2d,
            x=time_axis,
            y=mel_axis,
            colorscale="Viridis",
            colorbar=dict(
                title=dict(text="dB", font=dict(size=8, color="#64748B")),
                thickness=8,
                len=0.85,
                tickfont=dict(size=7.5, color="#64748B"),
                xpad=4,
                ypad=4,
            ),
            hovertemplate="Time: %{x:.2f} s<br>Mel: %{y}<br>Power: %{z:.1f} dB<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text="Log-Mel Spectrogram (128 Mels, 2048 FFT, 512 Hop)",
            font=dict(size=11, color="#0F172A", family="sans-serif"),
            x=0.01,
            y=0.96,
        ),
        margin=dict(l=35, r=15, t=28, b=26),
        height=185,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        xaxis=dict(
            title=dict(text="Time (seconds)", font=dict(size=9.5, color="#64748B")),
            range=[0.0, duration_sec],
            showgrid=False,
            tickfont=dict(size=8.5, color="#64748B"),
        ),
        yaxis=dict(
            title=dict(text="Mel Band", font=dict(size=9.5, color="#64748B")),
            range=[0, n_mels],
            showgrid=False,
            tickfont=dict(size=8.5, color="#64748B"),
        ),
    )
    return fig


def plot_waveform_and_spectrogram(
    waveform: np.ndarray,
    log_mel: np.ndarray,
    sample_rate: int = 22050,
) -> Any:
    """Render dual-panel waveform and Log-Mel spectrogram figure (Matplotlib export / fallback)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 4.4), dpi=120)
    time_axis = np.linspace(0.0, len(waveform) / sample_rate, len(waveform))
    ax1.plot(time_axis, waveform, color="#2563EB", linewidth=0.75, alpha=0.9)
    ax1.set_ylabel("Amplitude", fontsize=8.5)
    ax1.set_title("Standardized Audio Waveform (22,050 Hz, 4.0 s Mono)", fontsize=9.5, fontweight="600")
    ax1.set_xlim(0.0, time_axis[-1])
    ax1.set_ylim(-1.05, 1.05)
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax1.tick_params(labelsize=8)

    mel_2d = log_mel.squeeze()
    im = ax2.imshow(
        mel_2d,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[0.0, time_axis[-1], 0, 128],
    )
    ax2.set_xlabel("Time (seconds)", fontsize=8.5)
    ax2.set_ylabel("Mel Band (0–128)", fontsize=8.5)
    ax2.set_title("Log-Mel Spectrogram (128 Mels, 2048 FFT, 512 Hop)", fontsize=9.5, fontweight="600")
    ax2.tick_params(labelsize=8)

    cbar = fig.colorbar(im, ax=ax2, orientation="horizontal", pad=0.28, shrink=0.6)
    cbar.set_label("Decibels (dB relative to peak)", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    return fig


def main() -> None:
    """Render the compact, card-based inference experience."""
    st.set_page_config(
        page_title="Siren Detection AI",
        page_icon="🔊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root { --blue:#0B73E0; --navy:#10204F; --muted:#60708F; --line:#DCE7F5; --card:#F7FAFF; }
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            background:#FFFFFF !important; color:var(--navy) !important;
            font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif !important;
        }
        [data-testid="stSidebar"] { background:#F5F9FF !important; border-right:1px solid var(--line); }
        [data-testid="stSidebar"] * { color:var(--navy) !important; }
        .block-container { max-width:1240px !important; padding:1.15rem 2rem 2.5rem !important; }
        .hero { display:flex; align-items:center; gap:1rem; padding:.2rem 0 1.1rem; border-bottom:1px solid var(--line); }
        .hero-mark { width:48px; height:48px; display:grid; place-items:center; border-radius:14px; color:#fff; font-size:1.45rem; background:linear-gradient(145deg,#1685F5,#0758C9); box-shadow:0 7px 18px #0B73E033; }
        .hero h1 { margin:0; color:var(--navy); font-size:2rem; line-height:1.05; letter-spacing:-.04em; }
        .hero p { margin:.3rem 0 0; color:var(--muted); font-size:.92rem; }
        .eyebrow { color:var(--blue); font-size:.72rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; margin:0 0 .35rem; }
        .card { background:var(--card); border:1px solid var(--line); border-radius:16px; padding:1.15rem; box-shadow:0 8px 24px #1D4ED80D; margin-bottom:1rem; }
        .card h3 { color:var(--navy); font-size:1.05rem; margin:0 0 .85rem; }
        .upload-note { text-align:center; color:var(--muted); font-size:.82rem; padding:.5rem 0 .15rem; }
        .file-pill { display:flex; justify-content:space-between; align-items:center; background:#E7F7EF; border:1px solid #BCE8CF; border-radius:10px; padding:.6rem .75rem; margin:.6rem 0 .8rem; color:#166534; font-size:.82rem; font-weight:650; }
        .preview-meta { display:flex; flex-wrap:wrap; gap:.45rem; margin:.65rem 0 .8rem; }
        .meta-pill { background:#EAF3FF; color:#1D4ED8; border-radius:999px; padding:.38rem .62rem; font-size:.75rem; font-weight:650; }
        .result { border-radius:16px; padding:1rem 1.15rem; margin:.15rem 0 1rem; border:1px solid; }
        .result.siren { background:#FFF1F2; border-color:#FECDD3; }
        .result.clear { background:#EFFAF3; border-color:#BBE7C9; }
        .result-label { color:var(--muted); font-size:.73rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
        .result-title { font-size:1.45rem; font-weight:800; margin:.25rem 0 .25rem; }
        .siren .result-title { color:#D61F45; } .clear .result-title { color:#15803D; }
        .result-copy { color:var(--muted); font-size:.82rem; }
        .metric-card { background:#fff; border:1px solid var(--line); border-radius:13px; padding:.8rem; height:100%; }
        .metric-name { color:var(--muted); font-size:.72rem; font-weight:700; }
        .metric-value { color:var(--blue); font-size:1.45rem; font-weight:800; margin-top:.15rem; }
        div.stButton > button[kind="primary"] { background:linear-gradient(90deg,#0877E8,#075DCE) !important; color:#fff !important; border:0 !important; border-radius:11px !important; min-height:2.8rem; font-weight:750 !important; box-shadow:0 7px 16px #0B73E033; }
        div.stButton > button[kind="primary"]:hover { background:#064EB5 !important; }
        [data-testid="stExpander"] { border:1px solid var(--line) !important; border-radius:12px !important; background:#fff !important; }
        [data-testid="stExpander"] summary p { color:var(--navy) !important; font-weight:700 !important; }
        [data-testid="stFileUploaderDropzone"] { background:#fff !important; border:1.5px dashed #8DBCF2 !important; border-radius:12px !important; }
        [data-testid="stAudio"] { margin-top:.3rem; }
        .footer { color:#8492AA; text-align:center; font-size:.72rem; border-top:1px solid var(--line); padding-top:1rem; margin-top:1.3rem; }
        @media (max-width: 720px) { .block-container { padding:1rem .85rem 2rem !important; } .hero h1 { font-size:1.65rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero"><div class="hero-mark">✦</div><div><h1>Siren Detection AI</h1><p>AI-Powered Emergency Siren Recognition</p></div></div>',
        unsafe_allow_html=True,
    )

    metrics = load_model_comparison_metrics()
    primary_metric = metrics.loc[metrics["model_id"] == "deep_mlp"].iloc[0] if not metrics.empty else None
    with st.sidebar:
        st.markdown("### ⚙ Controls")
        threshold = st.slider(
            "Decision threshold (θ)", 0.05, 0.95, 0.50, 0.05,
            help="The operating cutoff: P(siren) >= θ produces a siren decision.",
        )
        st.caption("Adjusting this cutoff re-evaluates an existing model output; it does not retrain the model.")
        st.divider()
        st.markdown("### ▣ Primary model")
        st.markdown("**Deep MLP (PyTorch)**")
        st.caption("80-D MFCC · Seed 42 · 18,689 parameters")
        if primary_metric is not None:
            st.markdown(
                f"Validation Macro-F1  **{primary_metric['val_macro_f1_mean']:.4f} ± {primary_metric['val_macro_f1_std']:.4f}**\n\n"
                f"Held-out test Macro-F1  **{primary_metric['test_macro_f1_mean']:.4f} ± {primary_metric['test_macro_f1_std']:.4f}**"
            )
        st.caption("Protocol: UrbanSound8K Folds 1–8 train, Fold 9 validation, Fold 10 test.")

    engine = get_inference_engine()
    sample_dir = PROJECT_ROOT / "demo" / "sample_audio"
    siren_path = sample_dir / "siren_sample_165166-8-0-0.wav"
    non_siren_path = sample_dir / "non_siren_sample_100795-3-0-0.wav"

    upload_col, preview_col = st.columns([1.02, 1], gap="medium")
    with upload_col:
        st.markdown('<div class="card"><p class="eyebrow">01 · Input</p><h3>Upload audio</h3>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a WAV recording",
            type=["wav"],
            help="Application limit: 15 MB. Audio is standardized to 22,050 Hz, 4.0 seconds, mono.",
            label_visibility="collapsed",
        )
        st.markdown('<div class="upload-note">WAV format · application limit 15 MB · 22,050 Hz processing</div>', unsafe_allow_html=True)
        with st.expander("Try sample audio · verified Fold 10", expanded=False):
            st.caption("Held-out UrbanSound8K Fold 10 examples for demonstration only.")
            sample_choice = st.radio(
                "Sample audio",
                ["None (use uploaded file)", "Emergency siren · 165166-8-0-0.wav", "Non-siren · 100795-3-0-0.wav"],
                label_visibility="collapsed",
            )
        st.markdown("</div>", unsafe_allow_html=True)

    audio_bytes: bytes | None = None
    audio_label = ""
    provenance = ""
    active_audio_id: str | None = None
    if sample_choice != "None (use uploaded file)":
        selected_path = siren_path if "Emergency" in sample_choice else non_siren_path
        if selected_path.is_file():
            audio_bytes = selected_path.read_bytes()
            audio_label = selected_path.name
            provenance = "Verified Fold 10 benchmark sample"
            active_audio_id = f"sample:{selected_path.name}"
    elif uploaded_file is not None:
        if uploaded_file.size > 15 * 1024 * 1024:
            st.error("This file is larger than the 15 MB application limit and cannot be analyzed.")
        else:
            audio_bytes = uploaded_file.getvalue()
            audio_label = uploaded_file.name
            provenance = "Custom WAV upload"
            active_audio_id = f"upload:{uploaded_file.name}:{uploaded_file.size}"

    for key, default in (
        ("current_audio_id", None),
        ("cached_prediction", None),
        ("cached_comparison", None),
        ("cached_explanation", None),
    ):
        st.session_state.setdefault(key, default)

    if active_audio_id != st.session_state["current_audio_id"]:
        st.session_state["current_audio_id"] = active_audio_id
        st.session_state["cached_prediction"] = None
        st.session_state["cached_comparison"] = None
        st.session_state["cached_explanation"] = None

    with preview_col:
        st.markdown('<div class="card"><p class="eyebrow">02 · Preview</p><h3>Audio preview</h3>', unsafe_allow_html=True)
        if audio_bytes is None:
            st.info("Select a WAV file or choose a sample to preview it here.")
        else:
            safe_label = html.escape(audio_label)
            st.markdown(f'<div class="file-pill"><span>✓ &nbsp;{safe_label}</span><span>{len(audio_bytes) / 1024:.0f} KB</span></div>', unsafe_allow_html=True)
            st.audio(audio_bytes, format="audio/wav")
            st.markdown(
                '<div class="preview-meta"><span class="meta-pill">Mono output</span><span class="meta-pill">4.0 s window</span><span class="meta-pill">22,050 Hz</span></div>',
                unsafe_allow_html=True,
            )
            st.caption(provenance)
            analyze_clicked = st.button("▶  Analyze Audio", type="primary", use_container_width=True)
            if analyze_clicked:
                try:
                    with st.spinner("Extracting features, computing inference, and generating explanations…"):
                        st.session_state["cached_prediction"] = engine.predict(audio_bytes, threshold=threshold)
                        st.session_state["cached_comparison"] = engine.compare_all_models(audio_bytes, threshold=threshold)
                        
                        # Generate SHAP explanation if available
                        explainer = get_shap_explainer()
                        if explainer is not None:
                            st.session_state["cached_explanation"] = explainer.explain_audio(
                                audio_bytes, engine=engine, top_k=10
                            )
                        else:
                            st.session_state["cached_explanation"] = None
                except (CorruptAudioError, OSError, RuntimeError, ValueError) as err:
                    st.session_state["cached_prediction"] = None
                    st.session_state["cached_comparison"] = None
                    st.session_state["cached_explanation"] = None
                    st.error(f"Unable to analyze this recording: {err}")
        st.markdown("</div>", unsafe_allow_html=True)

    cached_pred = st.session_state.get("cached_prediction")
    if audio_bytes is not None and cached_pred is not None and st.session_state["current_audio_id"] == active_audio_id:
        prob = float(cached_pred["probability"])
        is_siren = prob >= threshold
        result_class = "siren" if is_siren else "clear"
        result_title = "Siren Detected" if is_siren else "No Siren Detected"
        st.markdown(
            f'<div class="result {result_class}"><div class="result-label">03 · Detection result</div>'
            f'<div class="result-title">{"🚨" if is_siren else "✓"} {result_title}</div>'
            f'<div class="result-copy">Model output for <strong>{html.escape(audio_label)}</strong> at operating threshold θ = {threshold:.2f}. '
            'This is a model-estimated probability, not guaranteed ground-truth certainty.</div></div>',
            unsafe_allow_html=True,
        )
        metric_cols = st.columns(4, gap="small")
        metric_values = [
            ("Model Estimated Siren Probability", f"{prob:.4f}"),
            ("Primary model", "Deep MLP"),
            ("Operating threshold", f"{threshold:.2f}"),
            ("Input representation", "80-D MFCC"),
        ]
        for col, (label, value) in zip(metric_cols, metric_values):
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-name">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
        st.progress(min(max(prob, 0.0), 1.0), text=f"P(siren | audio) = {prob:.4f}")

        st.markdown('<p class="eyebrow" style="margin-top:1.2rem;">04 · Audio visualization</p>', unsafe_allow_html=True)
        chart_cols = st.columns(2, gap="medium")
        with chart_cols[0]:
            st.plotly_chart(create_waveform_plotly(cached_pred["waveform"], cached_pred["sample_rate"]), use_container_width=True, config={"displayModeBar": False})
        with chart_cols[1]:
            st.plotly_chart(create_spectrogram_plotly(cached_pred["log_mel_spectrogram"], cached_pred["duration_sec"]), use_container_width=True, config={"displayModeBar": False})

        # 05 · Explain Prediction (SHAP Feature Attributions)
        cached_expl = st.session_state.get("cached_explanation")
        if cached_expl is not None:
            st.markdown('<p class="eyebrow" style="margin-top:1.4rem;">05 · Explain prediction (Evidence-Based SHAP)</p>', unsafe_allow_html=True)
            
            # Grounded acoustic interpretation card
            st.markdown(
                f'<div class="card" style="margin-bottom:0.75rem; background:#FFFFFF; border:1px solid #DCE7F5;">'
                f'<div style="font-size:0.75rem; font-weight:800; color:#0B73E0; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.35rem;">Model Decision Rationale</div>'
                f'<div style="font-size:0.88rem; color:#10204F; line-height:1.55;">'
                f'{cached_expl["interpretation"]}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Dual-column: SHAP horizontal attribution bar chart & Top 10 features table
            expl_cols = st.columns([1.1, 0.9], gap="medium")
            with expl_cols[0]:
                st.plotly_chart(
                    create_shap_attribution_plotly(cached_expl, top_k=10),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            with expl_cols[1]:
                st.markdown('<div style="font-size:0.82rem; font-weight:700; color:#10204F; margin-bottom:0.4rem;">Top 10 Contributing MFCC Coefficients</div>', unsafe_allow_html=True)
                top_rows = [
                    {
                        "Rank": f["rank"],
                        "Feature": f["feature_name"],
                        "Attribution (Logit)": f"{f['attribution']:+.4f}",
                        "Direction": f["direction"],
                        "Acoustic Descriptor": f["description"],
                    }
                    for f in cached_expl.get("top_features", [])[:10]
                ]
                st.dataframe(pd.DataFrame(top_rows), use_container_width=True, hide_index=True)

            # Scientific Caveat / Scope Callout
            st.markdown(
                """
                <div style="background:#FFFBEB; border:1px solid #FDE68A; border-radius:10px; padding:0.65rem 0.85rem; font-size:0.75rem; color:#92400E; margin-top:0.4rem; line-height:1.4;">
                    <strong>Scope & Limitations:</strong> SHAP values quantify additive local feature contributions to the pre-sigmoid logit based on DeepLIFT relative to the training split background. Attributions indicate internal model feature weighting, not physical real-world causation. Model-estimated scores are statistical outputs with class loss weighting and are not necessarily calibrated probabilities.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("Advanced details · multi-model comparison & noise robustness", expanded=False):
            cached_comp = st.session_state.get("cached_comparison")
            if cached_comp is not None:
                st.markdown("#### Live Model Comparison on Selected Audio")
                rows = []
                for model_info in cached_comp.get("models", {}).values():
                    model_prob = float(model_info["probability"])
                    rows.append({
                        "Architecture": model_info["model_name"],
                        "Siren probability": f"{model_prob:.4f}",
                        "Decision": "Siren" if model_prob >= threshold else "Non-Siren",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.caption("Comparison is unavailable for this recording.")

            # Phase 2 Empirical Robustness Benchmark Table & Figure
            phase2_csv = PROJECT_ROOT / "results" / "metrics" / "phase2_clean_vs_augmented_comparison.csv"
            phase2_img = PROJECT_ROOT / "results" / "figures" / "phase2_noise_robustness_comparison.png"
            if phase2_csv.is_file():
                st.divider()
                st.markdown("#### Phase 2 Empirical Benchmark: Clean vs. Noise-Augmented (Fold 10)")
                df_phase2 = pd.read_csv(phase2_csv)
                display_cols = ["condition", "model", "macro_f1", "siren_recall", "siren_precision", "pr_auc", "inference_latency_ms"]
                display_df = df_phase2[display_cols].copy()
                display_df.columns = ["Condition", "Model", "Macro-F1", "Siren Recall", "Precision", "PR-AUC", "Latency (ms)"]
                for c in ["Macro-F1", "Siren Recall", "Precision", "PR-AUC"]:
                    display_df[c] = display_df[c].apply(lambda v: f"{v:.4f}")
                display_df["Latency (ms)"] = display_df["Latency (ms)"].apply(lambda v: f"{v:.3f}")
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                if phase2_img.is_file():
                    st.image(str(phase2_img), caption="Noise Degradation Curves across SNR Levels (Fold 10, N=837)", use_container_width=True)

            # Phase 3 Spectro-Temporal Architecture Benchmark Table & Figure
            phase3_csv = PROJECT_ROOT / "results" / "metrics" / "phase3_model_comparison.csv"
            phase3_img = PROJECT_ROOT / "results" / "figures" / "phase3_architecture_comparison.png"
            if phase3_csv.is_file():
                st.divider()
                st.markdown("#### Phase 3 Architectural Benchmark: CNN + GRU Hybrid vs. Baselines (Fold 10)")
                df_phase3 = pd.read_csv(phase3_csv)
                p3_display_cols = ["model_name", "representation", "trainable_parameters", "test_macro_f1", "siren_recall", "siren_precision", "pr_auc", "inference_latency_ms"]
                p3_df = df_phase3[p3_display_cols].copy()
                p3_df.columns = ["Architecture", "Representation", "Parameters", "Test Macro-F1", "Siren Recall", "Precision", "PR-AUC", "Latency (ms)"]
                for c in ["Test Macro-F1", "Siren Recall", "Precision", "PR-AUC"]:
                    p3_df[c] = p3_df[c].apply(lambda v: f"{v:.4f}")
                p3_df["Latency (ms)"] = p3_df["Latency (ms)"].apply(lambda v: f"{v:.3f}")
                p3_df["Parameters"] = p3_df["Parameters"].apply(lambda v: f"{int(v):,}")
                st.dataframe(p3_df, use_container_width=True, hide_index=True)

                if phase3_img.is_file():
                    st.image(str(phase3_img), caption="Comparative Performance: Macro-F1 and Siren Class Recall (Fold 10, N=837)", use_container_width=True)



    st.markdown(
        '<div class="footer">Academic prototype for NB6057CEM · Fixed 4-second preprocessing window · '
        'Evaluation protocol: UrbanSound8K Folds 1–8 train, Fold 9 validation, Fold 10 test.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
