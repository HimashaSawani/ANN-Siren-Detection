"""Inference engine and deployment utilities for emergency siren detection.

Module: NB6057CEM — Artificial Neural Networks
Coursework Stage: CW1 T5 (Model Recommendation and Inference) & T6 (Reproducibility)

Provides:
- `check_inference_readiness`: Diagnostic check for model checkpoints and hardware.
- `SirenInferenceEngine`: Production-grade inference engine executing frozen Deep MLP
  (and optional comparative architectures: Random Forest, NumPy MLP, 2D CNN).
- `predict_audio`: Functional API for single-clip prediction.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch

from src import (
    load_config,
    load_data_config,
    resolve_project_paths,
    select_device,
)
from src.audio_features import (
    MFCCStandardScaler,
    MelSpectrogramNormalizer,
    extract_log_mel_spectrogram,
    extract_mfcc_features,
    load_and_standardize_audio,
)
from src.models import Conv2DCNN, DeepMLP, RandomForestBaseline
from src.numpy_mlp import NumPyMLP
from src.train import load_pytorch_checkpoint


def check_inference_readiness(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Inspect inference environment and verify checkpoint directory status.

    Args:
        config: Optional loaded configuration dictionary.

    Returns:
        Dictionary containing device status, checkpoint directory path, and existing checkpoints.
    """
    cfg = config if config is not None else load_config()
    paths = resolve_project_paths(cfg)
    hw_cfg = cfg.get("hardware", {})
    device_info = select_device(
        preferred=str(hw_cfg.get("preferred_device", "auto")),
        allow_gpu=bool(hw_cfg.get("allow_gpu", True)),
        fallback_to_cpu=bool(hw_cfg.get("fallback_to_cpu", True)),
    )

    checkpoints_dir: Path = paths["checkpoints_dir"]
    checkpoint_files = (
        sorted(
            p.name
            for p in checkpoints_dir.iterdir()
            if p.is_file() and p.name != ".gitkeep"
        )
        if checkpoints_dir.is_dir()
        else []
    )

    return {
        "device": device_info.selected_device,
        "checkpoints_dir": checkpoints_dir,
        "checkpoints_dir_exists": checkpoints_dir.is_dir(),
        "available_checkpoints": checkpoint_files,
        "ready_for_model_inference": len(checkpoint_files) > 0,
    }


class SirenInferenceEngine:
    """Production inference engine for emergency siren detection.

    Uses strictly frozen preprocessing parameters and scalers fitted on the
    training partition (Folds 1-8). Never refits scalers on inference inputs.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        scaler_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        cfg = load_config()
        paths = resolve_project_paths(cfg)
        self.device = torch.device(device)

        # 1. Resolve Primary Model (Deep MLP Seed 42)
        if checkpoint_path is None:
            checkpoint_path = paths["checkpoints_dir"] / "deep_mlp_t3_seed42.pt"
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found at '{self.checkpoint_path}'. "
                f"Execute training before running inference."
            )

        self.model, self.checkpoint_meta = load_pytorch_checkpoint(
            self.checkpoint_path, device=self.device
        )
        self.model.eval()

        # 2. Resolve Frozen Feature Scaler
        if scaler_path is None:
            scaler_path = paths["processed_data_dir"] / "mfcc_scaler.json"
        self.scaler_path = Path(scaler_path)
        if not self.scaler_path.is_file():
            raise FileNotFoundError(
                f"Frozen MFCC scaler not found at '{self.scaler_path}'."
            )
        self.mfcc_scaler = MFCCStandardScaler.load(self.scaler_path)

        # 3. Cache Data Configurations
        data_cfg = load_data_config()
        self.target_sr = int(data_cfg["audio_signal"].get("target_sample_rate", 22050))
        self.target_duration = float(data_cfg["audio_signal"].get("target_duration_sec", 4.0))
        self.mfcc_cfg = data_cfg["features"]["mfcc"]
        self.mel_cfg = data_cfg["features"]["mel_spectrogram"]

        # 4. Optional Scalers for Mel Spectrograms
        mel_stat_path = paths["processed_data_dir"] / "mel_norm_stats.json"
        self.mel_normalizer = (
            MelSpectrogramNormalizer.load(mel_stat_path)
            if mel_stat_path.is_file()
            else None
        )
        self.checkpoints_dir = paths["checkpoints_dir"]

    def preprocess_audio(
        self,
        audio_source: str | Path | BinaryIO | bytes,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load audio, standardize waveform, and extract normalized MFCC and Mel features.

        Returns:
            Tuple of (waveform_canonical, normalized_mfcc_vector, raw_mel_spectrogram).
        """
        waveform = load_and_standardize_audio(
            audio_source,
            target_sr=self.target_sr,
            target_duration_sec=self.target_duration,
        )

        # 80-D MFCC summary vector (40 mean + 40 std)
        mfcc_vec = extract_mfcc_features(
            waveform,
            sample_rate=self.target_sr,
            n_mfcc=int(self.mfcc_cfg.get("n_mfcc", 40)),
            n_fft=int(self.mfcc_cfg.get("n_fft", 2048)),
            hop_length=int(self.mfcc_cfg.get("hop_length", 512)),
            win_length=int(self.mfcc_cfg.get("win_length", 2048)),
            n_mels=int(self.mfcc_cfg.get("n_mels", 128)),
            f_min=float(self.mfcc_cfg.get("f_min", 0.0)),
            f_max=float(self.mfcc_cfg.get("f_max", 11025.0)),
        )
        norm_mfcc = self.mfcc_scaler.transform(mfcc_vec.reshape(1, -1)).astype(np.float32)

        # Log-Mel Spectrogram for visualization
        log_mel = extract_log_mel_spectrogram(
            waveform,
            sample_rate=self.target_sr,
            n_mels=int(self.mel_cfg.get("n_mels", 128)),
            n_fft=int(self.mel_cfg.get("n_fft", 2048)),
            hop_length=int(self.mel_cfg.get("hop_length", 512)),
            win_length=int(self.mel_cfg.get("win_length", 2048)),
            f_min=float(self.mel_cfg.get("f_min", 0.0)),
            f_max=float(self.mel_cfg.get("f_max", 11025.0)),
            top_db=float(self.mel_cfg.get("top_db", 80.0)),
        )
        return waveform, norm_mfcc, log_mel

    @torch.no_grad()
    def predict(
        self,
        audio_source: str | Path | BinaryIO | bytes,
        threshold: float = 0.50,
    ) -> dict[str, Any]:
        """Perform reliable binary inference on an audio input.

        Args:
            audio_source: File path, Path, file-like buffer, or raw bytes.
            threshold: Probability threshold for siren decision (default: 0.50).

        Returns:
            Comprehensive prediction metadata dictionary.
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}.")

        waveform, norm_mfcc, log_mel = self.preprocess_audio(audio_source)

        tensor_x = torch.from_numpy(norm_mfcc).to(self.device)
        logit = self.model(tensor_x)
        prob = float(torch.sigmoid(logit).item())

        is_siren = bool(prob >= threshold)
        pred_label = 1 if is_siren else 0
        class_name = "Siren" if is_siren else "Non-Siren"
        confidence = prob if is_siren else (1.0 - prob)

        return {
            "is_siren": is_siren,
            "predicted_class": class_name,
            "binary_label": pred_label,
            "probability": round(prob, 4),
            "confidence_percentage": round(confidence * 100.0, 2),
            "threshold": threshold,
            "model_id": "deep_mlp",
            "model_name": "Deep MLP (80-D MFCC, Seed 42)",
            "checkpoint_path": str(self.checkpoint_path),
            "waveform": waveform,
            "log_mel_spectrogram": log_mel,
            "sample_rate": self.target_sr,
            "duration_sec": self.target_duration,
            "num_samples": len(waveform),
            "feature_dim": norm_mfcc.shape[1],
        }

    def compare_all_models(
        self,
        audio_source: str | Path | BinaryIO | bytes,
        threshold: float = 0.50,
    ) -> dict[str, Any]:
        """Run multi-model comparative inference across all available checkpoints."""
        waveform, norm_mfcc, log_mel = self.preprocess_audio(audio_source)
        results: dict[str, Any] = {}

        # 1. Deep MLP (Seed 42)
        with torch.no_grad():
            t_x = torch.from_numpy(norm_mfcc).to(self.device)
            p_mlp = float(torch.sigmoid(self.model(t_x)).item())
            results["deep_mlp"] = {
                "model_name": "Deep MLP (PyTorch)",
                "probability": round(p_mlp, 4),
                "is_siren": p_mlp >= threshold,
                "predicted_class": "Siren" if p_mlp >= threshold else "Non-Siren",
            }

        # 2. Random Forest (Seed 42)
        rf_path = self.checkpoints_dir / "random_forest_t3_seed42.joblib"
        if rf_path.is_file():
            rf_model = RandomForestBaseline.load(rf_path)
            rf_probs = rf_model.predict_proba(norm_mfcc)
            p_rf = float(rf_probs[0, 1] if rf_probs.shape[1] > 1 else rf_probs[0, 0])
            results["random_forest"] = {
                "model_name": "Random Forest (scikit-learn)",
                "probability": round(p_rf, 4),
                "is_siren": p_rf >= threshold,
                "predicted_class": "Siren" if p_rf >= threshold else "Non-Siren",
            }

        # 3. NumPy MLP (CW1 T2, Seed 42)
        np_path = self.checkpoints_dir / "numpy_mlp_t2_seed42.npz"
        if np_path.is_file():
            np_model = NumPyMLP.load_parameters(np_path)
            p_np = float(np_model.predict_proba(norm_mfcc.astype(np.float64))[0, 0])
            results["numpy_mlp"] = {
                "model_name": "NumPy MLP (Scratch ANN)",
                "probability": round(p_np, 4),
                "is_siren": p_np >= threshold,
                "predicted_class": "Siren" if p_np >= threshold else "Non-Siren",
            }

        # 4. 2D CNN (Seed 42)
        cnn_path = self.checkpoints_dir / "cnn_2d_t3_seed42.pt"
        if cnn_path.is_file() and self.mel_normalizer is not None:
            cnn_model, _ = load_pytorch_checkpoint(cnn_path, device=self.device)
            cnn_model.eval()
            norm_mel = self.mel_normalizer.transform(log_mel.reshape(1, 1, 128, 173)).astype(np.float32)
            with torch.no_grad():
                p_cnn = float(torch.sigmoid(cnn_model(torch.from_numpy(norm_mel))).item())
                results["cnn_2d"] = {
                    "model_name": "2D CNN (PyTorch Log-Mel)",
                    "probability": round(p_cnn, 4),
                    "is_siren": p_cnn >= threshold,
                    "predicted_class": "Siren" if p_cnn >= threshold else "Non-Siren",
                }

        return {
            "models": results,
            "waveform": waveform,
            "log_mel_spectrogram": log_mel,
            "sample_rate": self.target_sr,
        }


def predict_audio(
    audio_source: str | Path | BinaryIO | bytes | None = None,
    checkpoint_path: str | Path | None = None,
    scaler_path: str | Path | None = None,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """Convenience functional API for single-clip emergency siren inference."""
    if audio_source is None:
        raise NotImplementedError("Audio inference requires an audio source input.")
    engine = SirenInferenceEngine(
        checkpoint_path=checkpoint_path, scaler_path=scaler_path
    )
    return engine.predict(audio_source=audio_source, threshold=threshold)
