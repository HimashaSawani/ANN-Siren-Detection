"""Unit tests for audio preprocessing, corrupt audio handling, feature dimensions, scalers, and augmentation.

Module: NB6057CEM — Artificial Neural Networks
Step: 2 — Dataset Acquisition, Data Audit and Leakage-Safe Preparation (CW1 T1)

Covers:
- Corrupt and empty audio file detection.
- Canonical waveform shape consistency (mono, 22,050 Hz, 4.0 s = 88,200 samples).
- MFCC feature vector dimensions (80-D) and deterministic extraction.
- Log-Mel spectrogram dimensions (1 x 128 x 173) and deterministic extraction.
- NaN and infinity prevention across waveforms, features, and scalers.
- Training-only scaler and normalizer fitting enforcement.
- Strict prevention of stochastic augmentation during validation/test evaluation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io.wavfile as wavfile

from src.audio_features import (
    CorruptAudioError,
    MFCCStandardScaler,
    MelSpectrogramNormalizer,
    extract_log_mel_spectrogram,
    extract_mfcc_features,
    inspect_audio_file,
    load_and_standardize_audio,
    standardize_waveform,
)
from src.augmentation import (
    AugmentationPolicyError,
    TrainingAudioAugmenter,
    add_controlled_background_noise,
    apply_amplitude_scaling,
    apply_time_shift,
)


def _create_wav_fixture(
    path: Path, sr: int = 22050, duration_sec: float = 2.0, stereo: bool = False
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(round(sr * duration_sec))
    t = np.linspace(0.0, duration_sec, n, endpoint=False, dtype=np.float32)
    sig = (0.5 * np.sin(2.0 * np.pi * 600.0 * t)).astype(np.float32)
    if stereo:
        sig = np.stack([sig, -sig * 0.5], axis=1)
    wavfile.write(str(path), sr, sig)
    return path


# ==============================================================================
# 1. Corrupt Audio & Missing File Handling
# ==============================================================================


def test_corrupt_and_empty_audio_handling(tmp_path: Path) -> None:
    """Verify clear CorruptAudioError on missing, 0-byte, and malformed audio files."""
    missing_wav = tmp_path / "missing.wav"
    with pytest.raises(CorruptAudioError, match="does not exist"):
        load_and_standardize_audio(missing_wav)
    header_missing = inspect_audio_file(missing_wav)
    assert header_missing.exists is False
    assert header_missing.is_corrupt is True

    empty_wav = tmp_path / "empty.wav"
    empty_wav.write_bytes(b"")
    with pytest.raises(CorruptAudioError, match="empty \\(0 bytes\\)"):
        load_and_standardize_audio(empty_wav)
    header_empty = inspect_audio_file(empty_wav)
    assert header_empty.exists is True
    assert header_empty.is_corrupt is True

    corrupt_wav = tmp_path / "corrupt.wav"
    corrupt_wav.write_bytes(b"NOT_A_VALID_RIFF_WAV_HEADER_1234567890")
    with pytest.raises(CorruptAudioError, match="Unreadable or corrupt"):
        load_and_standardize_audio(corrupt_wav)
    header_corrupt = inspect_audio_file(corrupt_wav)
    assert header_corrupt.is_corrupt is True


# ==============================================================================
# 2. Audio Shape Consistency, MFCC Dimensions, Mel Dimensions & Determinism
# ==============================================================================


@pytest.mark.parametrize(
    "orig_sr,duration_sec,stereo",
    [
        (16000, 1.2, False),   # Short mono clip requiring upsampling + zero-padding
        (44100, 3.5, True),    # Short stereo clip requiring downsampling + mono + padding
        (22050, 4.0, False),   # Exact canonical duration
        (48000, 5.2, True),    # Long stereo clip requiring downsampling + truncation
    ],
)
def test_audio_shape_consistency_across_formats(
    tmp_path: Path, orig_sr: int, duration_sec: float, stereo: bool
) -> None:
    """Verify all audio inputs standardize to mono (88200,) float32 at 22,050 Hz."""
    wav_path = _create_wav_fixture(
        tmp_path / f"clip_{orig_sr}_{duration_sec}_{stereo}.wav",
        sr=orig_sr,
        duration_sec=duration_sec,
        stereo=stereo,
    )
    waveform = load_and_standardize_audio(
        wav_path, target_sr=22050, target_duration_sec=4.0
    )
    assert waveform.shape == (88200,)
    assert waveform.dtype == np.float32
    assert np.all(np.isfinite(waveform))


def test_mfcc_and_mel_feature_dimensions_and_determinism(tmp_path: Path) -> None:
    """Verify MFCC (80,) and Log-Mel (1, 128, 173) shapes and deterministic repeatability."""
    wav_path = _create_wav_fixture(tmp_path / "deterministic_fixture.wav", sr=22050, duration_sec=4.0)
    waveform_1 = load_and_standardize_audio(wav_path)
    waveform_2 = load_and_standardize_audio(wav_path)
    assert np.array_equal(waveform_1, waveform_2)

    mfcc_1 = extract_mfcc_features(waveform_1)
    mfcc_2 = extract_mfcc_features(waveform_2)
    assert mfcc_1.shape == (80,)
    assert mfcc_1.dtype == np.float32
    assert np.all(np.isfinite(mfcc_1))
    assert np.allclose(mfcc_1, mfcc_2, atol=1e-7)

    mel_1 = extract_log_mel_spectrogram(waveform_1)
    mel_2 = extract_log_mel_spectrogram(waveform_2)
    assert mel_1.shape == (1, 128, 173)
    assert mel_1.dtype == np.float32
    assert np.all(np.isfinite(mel_1))
    assert np.allclose(mel_1, mel_2, atol=1e-7)


# ==============================================================================
# 3. NaN/Inf Prevention & Training-Only Scaler/Normalizer Fitting
# ==============================================================================


def test_nan_and_infinity_prevention() -> None:
    """Verify NaN or Inf values are rejected across waveform, feature, and scaler functions."""
    bad_wave = np.zeros(88200, dtype=np.float32)
    bad_wave[10] = np.nan
    with pytest.raises(CorruptAudioError, match="NaN or Inf"):
        standardize_waveform(bad_wave, orig_sr=22050)
    with pytest.raises(ValueError, match="NaN or Inf"):
        extract_mfcc_features(bad_wave)
    with pytest.raises(ValueError, match="NaN or Inf"):
        extract_log_mel_spectrogram(bad_wave)

    bad_features = np.ones((10, 80), dtype=np.float32)
    bad_features[0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        MFCCStandardScaler().fit(bad_features, split_name="train")


def test_training_only_scaler_and_normalizer_fitting(tmp_path: Path) -> None:
    """Verify scalers fit ONLY on 'train' split and reject 'val'/'test' fitting."""
    rng = np.random.default_rng(42)
    train_mfcc = rng.normal(loc=5.0, scale=2.0, size=(16, 80)).astype(np.float32)

    scaler = MFCCStandardScaler()
    with pytest.raises(ValueError, match="Leakage violation"):
        scaler.fit(train_mfcc, split_name="val")
    with pytest.raises(ValueError, match="Leakage violation"):
        scaler.fit(train_mfcc, split_name="test")

    scaled = scaler.fit_transform(train_mfcc, split_name="train")
    assert scaled.shape == (16, 80)
    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(scaled.std(axis=0), 1.0, atol=1e-5)

    saved_scaler_path = scaler.save(tmp_path / "mfcc_scaler.json")
    loaded_scaler = MFCCStandardScaler.load(saved_scaler_path)
    assert loaded_scaler.fitted_on_split == "train"
    assert np.allclose(scaler.transform(train_mfcc[0]), loaded_scaler.transform(train_mfcc[0]))

    # MelSpectrogramNormalizer check
    train_mel = rng.normal(loc=-30.0, scale=10.0, size=(8, 1, 128, 173)).astype(np.float32)
    mel_norm = MelSpectrogramNormalizer()
    with pytest.raises(ValueError, match="Leakage violation"):
        mel_norm.fit(train_mel, split_name="test")

    normed_mel = mel_norm.fit_transform(train_mel, split_name="train")
    assert normed_mel.shape == (8, 1, 128, 173)
    saved_mel_path = mel_norm.save(tmp_path / "mel_norm.json")
    loaded_mel = MelSpectrogramNormalizer.load(saved_mel_path)
    assert loaded_mel.fitted_on_split == "train"
    assert np.allclose(mel_norm.transform(train_mel[0]), loaded_mel.transform(train_mel[0]))


# ==============================================================================
# 4. Training-Only Augmentation & Evaluation Guard
# ==============================================================================


def test_training_only_augmentation_and_evaluation_guard() -> None:
    """Verify augmentation is reproducible on 'train' and strictly blocked on 'val'/'test'."""
    t = np.linspace(0.0, 4.0, 88200, endpoint=False, dtype=np.float32)
    clean_wave = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)

    augmenter = TrainingAudioAugmenter(
        enabled=True,
        probability=1.0,
        snr_db_range=(15.0, 20.0),
        max_time_shift_fraction=0.05,
        amplitude_scale_range=(0.9, 1.1),
        seed=42,
    )

    # Reproducible on train split for same sample_index
    aug_a = augmenter(clean_wave, split="train", sample_index=7)
    aug_b = augmenter(clean_wave, split="train", sample_index=7)
    assert aug_a.shape == clean_wave.shape
    assert np.allclose(aug_a, aug_b)
    assert not np.allclose(aug_a, clean_wave)

    # Strictly forbidden on val and test when enabled=True
    with pytest.raises(AugmentationPolicyError, match="forbidden on split='val'"):
        augmenter(clean_wave, split="val", sample_index=0)
    with pytest.raises(AugmentationPolicyError, match="forbidden on split='test'"):
        augmenter(clean_wave, split="test", sample_index=0)

    # Individual transform parameter guards
    with pytest.raises(AugmentationPolicyError):
        add_controlled_background_noise(clean_wave, snr_db=-20.0)
    with pytest.raises(AugmentationPolicyError):
        apply_time_shift(clean_wave, max_shift_fraction=0.6)
    with pytest.raises(AugmentationPolicyError):
        apply_amplitude_scaling(clean_wave, scale_range=(0.01, 5.0))

