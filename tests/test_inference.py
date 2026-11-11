"""Unit and integration tests for CW1 T5 Inference Engine and Streamlit Demo.

Module: NB6057CEM — Artificial Neural Networks
Coursework Stage: CW1 T5 (Model Recommendation and Inference) & T6 (Reproducibility)
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from src.audio_features import CorruptAudioError, load_and_standardize_audio
from src.inference import (
    SirenInferenceEngine,
    check_inference_readiness,
    predict_audio,
)


def test_inference_readiness_with_checkpoints() -> None:
    """Verify inference readiness confirms existing checkpoints and hardware."""
    info = check_inference_readiness()
    assert info["checkpoints_dir_exists"] is True
    assert info["ready_for_model_inference"] is True
    assert len(info["available_checkpoints"]) >= 4
    assert any("deep_mlp" in f for f in info["available_checkpoints"])
    assert any("random_forest" in f for f in info["available_checkpoints"])
    assert any("numpy_mlp" in f for f in info["available_checkpoints"])
    assert any("cnn_2d" in f for f in info["available_checkpoints"])


def test_siren_inference_engine_preprocessing() -> None:
    """Verify audio preprocessing produces canonical 22,050 Hz 4s audio and 80-D MFCCs."""
    engine = SirenInferenceEngine()
    siren_path = Path("demo/sample_audio/siren_sample_165166-8-0-0.wav")
    assert siren_path.is_file(), f"Sample clip missing at {siren_path}"

    waveform, norm_mfcc, log_mel = engine.preprocess_audio(siren_path)
    assert waveform.ndim == 1
    assert len(waveform) == 88200  # 4.0 sec @ 22,050 Hz
    assert norm_mfcc.shape == (1, 80)
    assert log_mel.shape in ((1, 128, 173), (128, 173))
    assert not np.isnan(norm_mfcc).any()
    assert not np.isinf(norm_mfcc).any()


def test_siren_inference_engine_prediction_on_known_samples() -> None:
    """Verify engine reliably classifies authentic held-out siren and non-siren samples."""
    engine = SirenInferenceEngine()

    siren_file = Path("demo/sample_audio/siren_sample_165166-8-0-0.wav")
    non_siren_file = Path("demo/sample_audio/non_siren_sample_100795-3-0-0.wav")

    res_s = engine.predict(siren_file, threshold=0.50)
    assert res_s["is_siren"] is True
    assert res_s["predicted_class"] == "Siren"
    assert res_s["binary_label"] == 1
    assert res_s["probability"] >= 0.80  # High confidence siren detection
    assert res_s["confidence_percentage"] >= 80.0

    res_ns = engine.predict(non_siren_file, threshold=0.50)
    assert res_ns["is_siren"] is False
    assert res_ns["predicted_class"] == "Non-Siren"
    assert res_ns["binary_label"] == 0
    assert res_ns["probability"] <= 0.20  # Confident non-siren detection


def test_predict_audio_functional_wrapper() -> None:
    """Verify predict_audio functional wrapper works identically."""
    siren_file = Path("demo/sample_audio/siren_sample_165166-8-0-0.wav")
    res = predict_audio(siren_file, threshold=0.50)
    assert res["is_siren"] is True
    assert "probability" in res
    assert "waveform" in res


def test_threshold_sensitivity_and_validation() -> None:
    """Verify decision threshold parameter enforcement and boundary conditions."""
    engine = SirenInferenceEngine()
    siren_file = Path("demo/sample_audio/siren_sample_165166-8-0-0.wav")

    # Invalid thresholds
    with pytest.raises(ValueError, match="threshold must be in"):
        engine.predict(siren_file, threshold=-0.1)
    with pytest.raises(ValueError, match="threshold must be in"):
        engine.predict(siren_file, threshold=1.5)

    # Extreme valid thresholds
    res_low = engine.predict(siren_file, threshold=0.01)
    assert res_low["is_siren"] is True

    res_high = engine.predict(siren_file, threshold=0.99999)
    assert isinstance(res_high["is_siren"], bool)


def test_compare_all_models_execution() -> None:
    """Verify multi-model comparative inference produces predictions from all 4 models."""
    engine = SirenInferenceEngine()
    siren_file = Path("demo/sample_audio/siren_sample_165166-8-0-0.wav")

    comp = engine.compare_all_models(siren_file, threshold=0.50)
    models = comp["models"]

    assert "deep_mlp" in models
    assert "random_forest" in models
    assert "numpy_mlp" in models
    assert "cnn_2d" in models

    for m_id, m_data in models.items():
        assert 0.0 <= m_data["probability"] <= 1.0
        assert m_data["predicted_class"] in ("Siren", "Non-Siren")
        assert isinstance(m_data["is_siren"], bool)


# ==============================================================================
# Comprehensive Audio Input Format Regression Tests (Path, Bytes, Stream, Errors)
# ==============================================================================


def test_inference_with_raw_bytes_input() -> None:
    """Verify inference engine cleanly accepts raw WAV bytes (Streamlit UploadedFile contract)."""
    engine = SirenInferenceEngine()
    siren_path = Path("demo/sample_audio/siren_sample_165166-8-0-0.wav")
    siren_bytes = siren_path.read_bytes()

    res = engine.predict(siren_bytes, threshold=0.50)
    assert res["is_siren"] is True
    assert res["predicted_class"] == "Siren"
    assert res["binary_label"] == 1
    assert res["probability"] >= 0.80
    assert len(res["waveform"]) == 88200


def test_inference_path_vs_bytes_consistency() -> None:
    """Verify Path input and in-memory bytes input produce bit-for-bit identical results."""
    engine = SirenInferenceEngine()
    siren_path = Path("demo/sample_audio/siren_sample_165166-8-0-0.wav")
    siren_bytes = siren_path.read_bytes()

    res_path = engine.predict(siren_path, threshold=0.50)
    res_bytes = engine.predict(siren_bytes, threshold=0.50)

    # Identical predictions and probabilities
    assert res_path["is_siren"] == res_bytes["is_siren"]
    assert res_path["predicted_class"] == res_bytes["predicted_class"]
    assert np.isclose(res_path["probability"], res_bytes["probability"], atol=1e-6)

    # Identical standardized waveforms and spectrograms
    np.testing.assert_allclose(res_path["waveform"], res_bytes["waveform"], atol=1e-7)
    np.testing.assert_allclose(res_path["log_mel_spectrogram"], res_bytes["log_mel_spectrogram"], atol=1e-6)


def test_inference_with_bytesio_stream() -> None:
    """Verify inference engine accepts io.BytesIO file-like stream objects."""
    engine = SirenInferenceEngine()
    non_siren_path = Path("demo/sample_audio/non_siren_sample_100795-3-0-0.wav")
    stream = io.BytesIO(non_siren_path.read_bytes())

    res = engine.predict(stream, threshold=0.50)
    assert res["is_siren"] is False
    assert res["predicted_class"] == "Non-Siren"
    assert res["binary_label"] == 0
    assert res["probability"] <= 0.20


def test_compare_all_models_with_raw_bytes() -> None:
    """Verify multi-model comparison functions correctly with raw bytes input."""
    engine = SirenInferenceEngine()
    siren_path = Path("demo/sample_audio/siren_sample_165166-8-0-0.wav")
    siren_bytes = siren_path.read_bytes()

    comp = engine.compare_all_models(siren_bytes, threshold=0.50)
    assert len(comp["models"]) == 4
    for _, m_info in comp["models"].items():
        assert 0.0 <= m_info["probability"] <= 1.0


def test_inference_rejects_empty_input() -> None:
    """Verify engine and preprocessor raise CorruptAudioError on empty bytes or empty stream."""
    engine = SirenInferenceEngine()

    with pytest.raises(CorruptAudioError, match="empty \\(0 bytes\\)"):
        engine.predict(b"")

    with pytest.raises(CorruptAudioError, match="empty \\(0 bytes\\)"):
        engine.predict(io.BytesIO(b""))


def test_inference_rejects_unsupported_audio_formats() -> None:
    """Verify clear CorruptAudioError with descriptive messages for MP3, OGG, and FLAC."""
    engine = SirenInferenceEngine()

    # Mock MP3 with ID3 header
    mock_mp3_id3 = b"ID3\x03\x00\x00\x00\x00\x00\x10" + b"\x00" * 200
    with pytest.raises(CorruptAudioError, match="Unsupported audio format: MP3"):
        engine.predict(mock_mp3_id3)

    # Mock MP3 with sync frame
    mock_mp3_sync = b"\xff\xfb\x90\x44" + b"\x00" * 200
    with pytest.raises(CorruptAudioError, match="Unsupported audio format: MP3"):
        engine.predict(mock_mp3_sync)

    # Mock OGG
    mock_ogg = b"OggS\x00\x02" + b"\x00" * 200
    with pytest.raises(CorruptAudioError, match="Unsupported audio format: OGG"):
        engine.predict(mock_ogg)

    # Mock FLAC
    mock_flac = b"fLaC\x00\x00" + b"\x00" * 200
    with pytest.raises(CorruptAudioError, match="Unsupported audio format: FLAC"):
        engine.predict(mock_flac)


def test_inference_rejects_invalid_corrupted_bytes() -> None:
    """Verify engine raises CorruptAudioError on corrupted/truncated non-audio bytes."""
    engine = SirenInferenceEngine()

    # Random corrupt bytes
    corrupt_bytes = b"CORRUPTED_NON_AUDIO_STREAM_DATA_ABC_123"
    with pytest.raises(CorruptAudioError, match="Unreadable or corrupt"):
        engine.predict(corrupt_bytes)

    # Truncated RIFF header
    truncated_wav = b"RIFF\x20\x00\x00\x00WAVEfmt "
    with pytest.raises(CorruptAudioError):
        engine.predict(truncated_wav)


def test_inference_rejects_invalid_types() -> None:
    """Verify TypeError when unsupported non-buffer types are provided."""
    engine = SirenInferenceEngine()

    with pytest.raises(TypeError, match="Unsupported audio source type"):
        engine.predict(12345)  # type: ignore

    with pytest.raises(TypeError, match="Unsupported audio source type"):
        engine.predict(["not", "an", "audio", "path"])  # type: ignore


def test_load_and_standardize_audio_direct_bytes_no_type_error() -> None:
    """Explicitly verify load_and_standardize_audio with raw bytes, BytesIO, str, and Path.

    Guarantees that TypeError (argument should be a str or an os.PathLike, not 'bytes')
    is permanently eradicated across all direct loader invocations.
    """
    siren_path = Path("demo/sample_audio/siren_sample_165166-8-0-0.wav")
    assert siren_path.is_file()
    siren_bytes = siren_path.read_bytes()

    # 1. Pure bytes input
    wf_bytes = load_and_standardize_audio(siren_bytes)
    assert isinstance(wf_bytes, np.ndarray)
    assert wf_bytes.shape == (88200,)
    assert wf_bytes.dtype == np.float32

    # 2. File-like BytesIO stream
    wf_stream = load_and_standardize_audio(io.BytesIO(siren_bytes))
    assert isinstance(wf_stream, np.ndarray)
    assert wf_stream.shape == (88200,)
    assert wf_stream.dtype == np.float32

    # 3. Pathlib Path
    wf_path = load_and_standardize_audio(siren_path)
    assert isinstance(wf_path, np.ndarray)

    # 4. Filesystem path string
    wf_str = load_and_standardize_audio(str(siren_path))
    assert isinstance(wf_str, np.ndarray)

    # 5. Strict equality check
    np.testing.assert_allclose(wf_bytes, wf_path, atol=1e-7)
    np.testing.assert_allclose(wf_bytes, wf_stream, atol=1e-7)
    np.testing.assert_allclose(wf_bytes, wf_str, atol=1e-7)


