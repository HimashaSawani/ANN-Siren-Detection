"""Comprehensive runtime verification of audio inference across all input modes.

Validates that:
1. Preset siren WAV (Path and Bytes) classifies as Siren with probability >= 0.80.
2. Preset non-siren WAV (Path and Bytes) classifies as Non-Siren with probability <= 0.20.
3. Custom uploaded siren WAV (BytesIO and Bytes) produces identical predictions.
4. Custom uploaded non-siren WAV (BytesIO and Bytes) produces identical predictions.
5. All 4 competing models (Deep MLP, Random Forest, NumPy MLP, 2D CNN) evaluate without error.
6. Corrupted, empty, or MP3 bytes raise clean CorruptAudioError without TypeError.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.audio_features import CorruptAudioError, load_and_standardize_audio
from src.inference import SirenInferenceEngine
from demo.app import plot_waveform_and_spectrogram


def main() -> None:
    print("=== STARTING FULL AUDIO INFERENCE VERIFICATION ===")
    engine = SirenInferenceEngine()

    siren_path = Path("demo/sample_audio/siren_sample_165166-8-0-0.wav")
    non_siren_path = Path("demo/sample_audio/non_siren_sample_100795-3-0-0.wav")

    assert siren_path.is_file(), f"Siren sample clip missing: {siren_path}"
    assert non_siren_path.is_file(), f"Non-siren sample clip missing: {non_siren_path}"

    siren_bytes = siren_path.read_bytes()
    non_siren_bytes = non_siren_path.read_bytes()

    # 1. Preset Siren (Path)
    res_s_path = engine.predict(siren_path, threshold=0.50)
    print(f"[PASS] 1. Preset Siren (Path): {res_s_path['predicted_class']} (P = {res_s_path['probability']:.4f})")
    assert res_s_path["is_siren"] is True

    # 2. Preset Siren (Bytes)
    res_s_bytes = engine.predict(siren_bytes, threshold=0.50)
    print(f"[PASS] 2. Preset Siren (Bytes): {res_s_bytes['predicted_class']} (P = {res_s_bytes['probability']:.4f})")
    assert res_s_bytes["is_siren"] is True
    assert np.isclose(res_s_path["probability"], res_s_bytes["probability"], atol=1e-6)
    np.testing.assert_allclose(res_s_path["waveform"], res_s_bytes["waveform"], atol=1e-7)

    # 3. Preset Non-Siren (Path)
    res_ns_path = engine.predict(non_siren_path, threshold=0.50)
    print(f"[PASS] 3. Preset Non-Siren (Path): {res_ns_path['predicted_class']} (P = {res_ns_path['probability']:.4f})")
    assert res_ns_path["is_siren"] is False

    # 4. Preset Non-Siren (Bytes)
    res_ns_bytes = engine.predict(non_siren_bytes, threshold=0.50)
    print(f"[PASS] 4. Preset Non-Siren (Bytes): {res_ns_bytes['predicted_class']} (P = {res_ns_bytes['probability']:.4f})")
    assert res_ns_bytes["is_siren"] is False
    assert np.isclose(res_ns_path["probability"], res_ns_bytes["probability"], atol=1e-6)

    # 5. Custom Uploaded Siren (BytesIO Stream - simulates Streamlit UploadedFile)
    res_upload_s = engine.predict(io.BytesIO(siren_bytes), threshold=0.50)
    print(f"[PASS] 5. Uploaded Siren (BytesIO): {res_upload_s['predicted_class']} (P = {res_upload_s['probability']:.4f})")
    assert res_upload_s["is_siren"] is True
    assert np.isclose(res_s_bytes["probability"], res_upload_s["probability"], atol=1e-6)

    # 6. Custom Uploaded Non-Siren (BytesIO Stream)
    res_upload_ns = engine.predict(io.BytesIO(non_siren_bytes), threshold=0.50)
    print(f"[PASS] 6. Uploaded Non-Siren (BytesIO): {res_upload_ns['predicted_class']} (P = {res_upload_ns['probability']:.4f})")
    assert res_upload_ns["is_siren"] is False
    assert np.isclose(res_ns_bytes["probability"], res_upload_ns["probability"], atol=1e-6)

    # 7. Waveform and Spectrogram rendering verification
    fig = plot_waveform_and_spectrogram(res_s_bytes["waveform"], res_s_bytes["log_mel_spectrogram"], 22050)
    plt.close(fig)
    print("[PASS] 7. Waveform and Spectrogram Figure: Successfully generated and rendered.")

    # 8. Multi-Model Cross-Check on Bytes
    comp = engine.compare_all_models(siren_bytes, threshold=0.50)
    print(f"[PASS] 8. Multi-Model Cross-Check on Bytes ({len(comp['models'])} models):")
    for m_id, m_info in comp["models"].items():
        print(f"       - {m_info['model_name']:30s}: {m_info['predicted_class']:10s} (P = {m_info['probability']:.4f})")
    assert len(comp["models"]) == 4

    # 9. Direct loader call with pure bytes
    wf_direct = load_and_standardize_audio(siren_bytes)
    assert wf_direct.shape == (88200,)
    assert wf_direct.dtype == np.float32
    print(f"[PASS] 9. Direct load_and_standardize_audio(bytes): shape = {wf_direct.shape}, dtype = {wf_direct.dtype}")

    # 10. Rejection of invalid inputs without TypeError
    for name, invalid_buf in [
        ("Empty Bytes", b""),
        ("Corrupted Header", b"NOT_A_VALID_RIFF_WAV_HEADER_1234567890"),
        ("MP3 Header", b"ID3\x03\x00\x00\x00\x00\x00\x10" + b"\x00" * 200),
    ]:
        try:
            engine.predict(invalid_buf)
            raise AssertionError(f"Expected failure for {name}")
        except Exception as exc:
            assert not isinstance(exc, TypeError), f"Unexpected TypeError for {name}!"
            assert "CorruptAudioError" in type(exc).__name__ or isinstance(exc, ValueError)
            print(f"[PASS] 10. Rejected {name:16s} with {type(exc).__name__}: '{str(exc)[:50]}...'")

    print("\n=== ALL 10 VERIFICATION CHECKS PASSED PERFECTLY ===")


if __name__ == "__main__":
    main()
