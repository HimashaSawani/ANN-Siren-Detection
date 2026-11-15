"""Explainable AI (XAI) module for emergency siren detection.

Module: NB6057CEM — Artificial Neural Networks
Capability: Model Explainability via SHAP (SHapley Additive exPlanations)

Provides:
- `SirenSHAPExplainer`: Local feature attribution engine for the trained Deep MLP.
- `explain_audio_prediction`: Functional API for explaining a single audio recording.
- `create_shap_attribution_plotly`: Interactive Plotly horizontal attribution chart.
- `generate_human_interpretation`: Grounded acoustic interpretation synthesis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import plotly.graph_objects as go
import torch

from src import load_config, resolve_project_paths
from src.audio_features import MFCCStandardScaler
from src.models import DeepMLP
from src.train import load_pytorch_checkpoint

logger = logging.getLogger(__name__)

# Ensure persistent, writable Matplotlib cache directory before shap imports matplotlib
import os
import tempfile
_mpl_dir = os.path.join(tempfile.gettempdir(), "matplotlib_cache")
os.makedirs(_mpl_dir, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_dir)

# Try importing shap as an optional dependency
try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    shap = None
    _SHAP_AVAILABLE = False


def is_shap_available() -> bool:
    """Return True if the SHAP library is installed and importable."""
    return _SHAP_AVAILABLE


def get_mfcc_feature_names(n_mfcc: int = 40) -> list[str]:
    """Generate canonical ordered feature names for the 80-D MFCC summary representation."""
    means = [f"mfcc_mean_{i}" for i in range(n_mfcc)]
    stds = [f"mfcc_std_{i}" for i in range(n_mfcc)]
    return means + stds


def describe_mfcc_feature(feature_name: str) -> str:
    """Provide a scientifically accurate acoustic interpretation for an MFCC summary coefficient.

    Note: MFCC coefficients represent discrete cosine transform (DCT-II) basis
    projections over the entire log-mel filterbank (spectral envelope quefrency),
    rather than localized, isolated physical frequency bands.
    """
    if not (feature_name.startswith("mfcc_mean_") or feature_name.startswith("mfcc_std_")):
        return "Unknown acoustic descriptor"

    is_mean = feature_name.startswith("mfcc_mean_")
    try:
        idx = int(feature_name.split("_")[-1])
    except ValueError:
        return "MFCC descriptor"

    stat_type = "Mean spectral envelope" if is_mean else "Temporal modulation variance"

    if idx == 0:
        return f"{stat_type}: Overall log energy / broadband power"
    elif 1 <= idx <= 4:
        return f"{stat_type}: Low-order quefrency (macro spectral tilt & gross spectral slope)"
    elif 5 <= idx <= 15:
        return f"{stat_type}: Intermediate quefrency (spectral formant structure & harmonic peak patterns)"
    elif 16 <= idx <= 25:
        return f"{stat_type}: Upper-mid quefrency (fine spectral envelope timbre & overtone distribution)"
    else:
        return f"{stat_type}: High quefrency (rapid spectral ripples across mel channels)"


class SirenSHAPExplainer:
    """Local feature attribution engine for the trained Deep MLP siren classifier.

    Uses DeepLIFT-based SHAP (SHapley Additive exPlanations) to compute exact additive
    attributions on the network's pre-sigmoid logit space:
        logit(x) = base_value + sum(phi_i)

    Attributes:
        model: Loaded PyTorch DeepMLP instance in eval mode.
        background: Representative background dataset from the training split (Folds 1-8).
        feature_names: List of 80 canonical feature names.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        background: np.ndarray | None = None,
        num_background_samples: int = 100,
        device: str = "cpu",
        random_seed: int = 42,
    ) -> None:
        if not _SHAP_AVAILABLE:
            raise ImportError(
                "The 'shap' library is required for SirenSHAPExplainer. "
                "Install it via 'pip install shap'."
            )

        cfg = load_config()
        paths = resolve_project_paths(cfg)
        self.device = torch.device(device)
        self.random_seed = random_seed

        # 1. Resolve and load Deep MLP Checkpoint
        if checkpoint_path is None:
            checkpoint_path = paths["checkpoints_dir"] / "deep_mlp_t3_seed42.pt"
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Deep MLP checkpoint not found at '{self.checkpoint_path}'."
            )

        self.model, self.checkpoint_meta = load_pytorch_checkpoint(
            self.checkpoint_path, device=self.device
        )
        self.model.eval()

        # 2. Resolve background samples strictly from the training split
        if background is not None:
            bg_arr = np.asarray(background, dtype=np.float32)
            if bg_arr.ndim != 2 or bg_arr.shape[1] != 80:
                raise ValueError(
                    f"Background data must have shape (N, 80), got {bg_arr.shape}."
                )
            self.background = bg_arr
        else:
            self.background = self._load_or_create_background(
                processed_dir=paths["processed_data_dir"],
                num_samples=num_background_samples,
                seed=self.random_seed,
            )

        # 3. Initialize SHAP DeepExplainer
        bg_tensor = torch.from_numpy(self.background).float().to(self.device)
        self.explainer = shap.DeepExplainer(self.model, bg_tensor)
        self.feature_names = get_mfcc_feature_names(n_mfcc=40)

        # Base expected value (mean logit over background)
        ev = self.explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            self.base_value = float(np.array(ev).squeeze())
        else:
            self.base_value = float(ev)

    def _load_or_create_background(
        self,
        processed_dir: Path,
        num_samples: int = 100,
        seed: int = 42,
    ) -> np.ndarray:
        """Load cached background samples or sample deterministically from the training split."""
        bg_cache_file = processed_dir / "shap_background.npz"
        if bg_cache_file.is_file():
            try:
                data = np.load(bg_cache_file)
                if "background" in data:
                    cached_bg = data["background"].astype(np.float32)
                    if len(cached_bg) >= num_samples:
                        return cached_bg[:num_samples]
            except Exception as exc:
                logger.warning("Failed to load cached background from %s: %s", bg_cache_file, exc)

        # Sample from train_val_mfcc_cache.npz (strictly training partition X_train)
        cache_path = processed_dir / "train_val_mfcc_cache.npz"
        if not cache_path.is_file():
            logger.info(
                "Training cache '%s' not present; initializing zero-mean standardized reference background.",
                cache_path.name,
            )
            return np.zeros((num_samples, 80), dtype=np.float32)

        cache = np.load(cache_path)
        X_tr = cache["X_train"]
        y_tr = cache["y_train"]

        # Stratified sampling matching training class distribution
        siren_idx = np.where(y_tr == 1)[0]
        non_siren_idx = np.where(y_tr == 0)[0]

        n_siren = max(1, int(round(num_samples * (len(siren_idx) / len(y_tr)))))
        n_non = num_samples - n_siren

        rng = np.random.RandomState(seed)
        sel_siren = rng.choice(siren_idx, size=min(n_siren, len(siren_idx)), replace=False)
        sel_non = rng.choice(non_siren_idx, size=min(n_non, len(non_siren_idx)), replace=False)
        all_sel = np.concatenate([sel_siren, sel_non])
        rng.shuffle(all_sel)

        bg_samples = X_tr[all_sel].astype(np.float32)

        # Cache for subsequent runs
        try:
            np.savez_compressed(
                bg_cache_file,
                background=bg_samples,
                feature_names=get_mfcc_feature_names(40),
            )
        except Exception as exc:
            logger.warning("Could not cache SHAP background: %s", exc)

        return bg_samples

    def explain_features(
        self,
        normalized_mfcc: np.ndarray,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Compute local SHAP feature attributions for a standardized 80-D MFCC feature vector.

        Args:
            normalized_mfcc: Array of shape ``(80,)`` or ``(1, 80)`` with float32 values.
            top_k: Number of highest-magnitude contributing features to summarize.

        Returns:
            Dictionary containing attributions, base value, logit, probability, and rankings.
        """
        arr = np.asarray(normalized_mfcc, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape != (1, 80):
            raise ValueError(f"Expected input shape (1, 80), got {arr.shape}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Input features contain NaN or Inf values.")

        tensor_x = torch.from_numpy(arr).to(self.device)
        with torch.no_grad():
            logit = float(self.model(tensor_x).item())
            prob = float(torch.sigmoid(torch.tensor(logit)).item())

        # Compute SHAP values via DeepLIFT
        shap_values_raw = self.explainer.shap_values(tensor_x)
        sv = np.array(shap_values_raw).squeeze()  # (80,)

        # Sort features by absolute contribution magnitude
        abs_sv = np.abs(sv)
        sorted_indices = np.argsort(abs_sv)[::-1]

        top_features = []
        for rank, idx in enumerate(sorted_indices[:top_k], start=1):
            attr = float(sv[idx])
            feat_name = self.feature_names[idx]
            feat_val = float(arr[0, idx])
            direction = "Toward Siren (+)" if attr > 0 else "Toward Non-Siren (-)"
            top_features.append({
                "rank": rank,
                "feature_index": int(idx),
                "feature_name": feat_name,
                "attribution": round(attr, 4),
                "normalized_value": round(feat_val, 4),
                "direction": direction,
                "is_positive": attr > 0,
                "description": describe_mfcc_feature(feat_name),
            })

        interpretation = generate_human_interpretation(
            prob=prob,
            top_features=top_features,
            base_value=self.base_value,
            logit=logit,
        )

        return {
            "shap_values": sv,
            "feature_names": self.feature_names,
            "base_value": round(self.base_value, 4),
            "model_logit": round(logit, 4),
            "estimated_probability": round(prob, 4),
            "predicted_class": "Siren" if prob >= 0.50 else "Non-Siren",
            "is_siren": prob >= 0.50,
            "top_features": top_features,
            "top_k": top_k,
            "interpretation": interpretation,
            "explanation_space": "logit",
            "caveats": [
                "Explanations operate additively on pre-sigmoid model logits: logit(x) = base_value + sum(attributions).",
                "SHAP attributions describe the internal decision logic of the trained Deep MLP, not physical real-world causation.",
                "Model-estimated scores are statistical outputs with class weighting, not calibrated objective probabilities.",
            ],
        }

    def explain_audio(
        self,
        audio_source: str | Path | BinaryIO | bytes,
        engine: Any = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Convenience method to preprocess audio and compute SHAP explanation in one call."""
        if engine is None:
            from src.inference import SirenInferenceEngine
            engine = SirenInferenceEngine()

        _, norm_mfcc, _ = engine.preprocess_audio(audio_source)
        return self.explain_features(norm_mfcc, top_k=top_k)


def generate_human_interpretation(
    prob: float,
    top_features: list[dict[str, Any]],
    base_value: float,
    logit: float,
) -> str:
    """Synthesize an evidence-based, scientifically grounded interpretation from attributions."""
    is_siren = prob >= 0.50
    class_label = "Siren" if is_siren else "Non-Siren"

    positive_feats = [f for f in top_features if f["is_positive"]]
    negative_feats = [f for f in top_features if not f["is_positive"]]

    # Analyze predominant acoustic mechanisms
    harmonic_pos = [f for f in positive_feats if "mfcc_mean_5" <= f["feature_name"] <= "mfcc_mean_15"]
    modulation_pos = [f for f in positive_feats if "mfcc_std" in f["feature_name"]]
    energy_neg = [f for f in negative_feats if f["feature_name"] == "mfcc_mean_0"]

    sentences = []
    sentences.append(
        f"The Deep MLP classified this recording as **{class_label}** with a model-estimated score of **{prob:.4f}** "
        f"(logit: {logit:+.2f} relative to training baseline {base_value:+.2f})."
    )

    if is_siren:
        if harmonic_pos:
            h_names = ", ".join(f["feature_name"] for f in harmonic_pos[:2])
            sentences.append(
                f"Elevated values in intermediate spectral envelope coefficients ({h_names}) provided positive attribution towards Siren, "
                f"capturing prominent harmonic peak patterns characteristic of periodic siren sounds."
            )
        if modulation_pos:
            m_names = ", ".join(f["feature_name"] for f in modulation_pos[:2])
            sentences.append(
                f"High temporal modulation variance ({m_names}) provided positive attribution, "
                f"reflecting dynamic pitch oscillation (wail/yelp cycle)."
            )
        if not harmonic_pos and not modulation_pos:
            top_pos = positive_feats[0]["feature_name"] if positive_feats else "spectral features"
            sentences.append(f"Attribution is driven primarily by positive contributions from {top_pos}.")
    else:
        if negative_feats:
            n_names = ", ".join(f["feature_name"] for f in negative_feats[:2])
            sentences.append(
                f"Absence of siren harmonic structures and damping contributions from {n_names} "
                f"pushed the logit downwards into the Non-Siren decision boundary."
            )
        if energy_neg:
            sentences.append(
                "Broadband acoustic energy without characteristic periodic pitch sweeps indicates ambient urban clutter."
            )

    return " ".join(sentences)


def create_shap_attribution_plotly(
    explanation: dict[str, Any],
    top_k: int = 10,
) -> go.Figure:
    """Create a responsive Plotly horizontal bar chart displaying top SHAP attributions."""
    top_feats = explanation.get("top_features", [])[:top_k]
    if not top_feats:
        fig = go.Figure()
        fig.update_layout(title="No SHAP attributions available")
        return fig

    # Reverse order so highest magnitude is at top of vertical axis
    reversed_feats = top_feats[::-1]

    names = [f["feature_name"] for f in reversed_feats]
    values = [f["attribution"] for f in reversed_feats]
    colors = ["#EF4444" if v > 0 else "#2563EB" for v in values]
    hover_texts = [
        f"<b>{f['feature_name']}</b><br>"
        f"Attribution: {f['attribution']:+.4f} (Logit)<br>"
        f"Normalized Value: {f['normalized_value']:+.2f}<br>"
        f"Direction: {f['direction']}<br>"
        f"<i>{f['description']}</i>"
        for f in reversed_feats
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=names,
            x=values,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="#CBD5E1", width=1),
            ),
            hoverinfo="text",
            hovertext=hover_texts,
        )
    )

    base_val = explanation.get("base_value", 0.0)
    logit = explanation.get("model_logit", 0.0)

    fig.update_layout(
        title=dict(
            text=f"Top {len(top_feats)} Feature Attributions on Pre-Sigmoid Logit (Base: {base_val:+.2f} &rarr; Logit: {logit:+.2f})",
            font=dict(size=11, color="#0F172A", family="sans-serif"),
            x=0.01,
            y=0.96,
        ),
        margin=dict(l=95, r=20, t=32, b=28),
        height=240,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        xaxis=dict(
            title=dict(
                text="SHAP Attribution &phi;<sub>i</sub> (Logit Contribution)",
                font=dict(size=9.5, color="#64748B"),
            ),
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=True,
            zerolinecolor="#94A3B8",
            zerolinewidth=1.5,
            tickfont=dict(size=8.5, color="#64748B"),
        ),
        yaxis=dict(
            title=dict(text="Feature", font=dict(size=9.5, color="#64748B")),
            tickfont=dict(size=8.5, color="#0F172A"),
            showgrid=False,
        ),
    )
    return fig


def explain_audio_prediction(
    audio_source: str | Path | BinaryIO | bytes,
    checkpoint_path: str | Path | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Functional API: preprocess an audio recording and return local SHAP attributions."""
    explainer = SirenSHAPExplainer(checkpoint_path=checkpoint_path)
    return explainer.explain_audio(audio_source, top_k=top_k)
