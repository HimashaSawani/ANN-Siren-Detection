"""Audio loading, canonical standardization, MFCC extraction, and Log-Mel spectrogram pipeline.

Module: NB6057CEM — Artificial Neural Networks
Step: 2 — Dataset Acquisition, Data Audit and Leakage-Safe Preparation (CW1 T1)

Provides:
1. Canonical audio loading (mono conversion, 22,050 Hz resampling, 4.0 s / 88,200-sample
   zero-padding or truncation) with corrupt/empty audio detection.
2. MFCC feature extraction (40 MFCCs -> 80-D summary vector: temporal mean + std)
   for Random Forest and Deep MLP models, with training-only standard scaling.
3. Fixed-size 2D Log-Mel spectrogram extraction (shape: 1 x 128 x 173) for 2D CNN,
   with training-only normalization statistics.
"""

from __future__ import annotations

import functools
import io
import json
import math
import struct
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as spsignal
import torch
import torchaudio


class CorruptAudioError(ValueError):
    """Raised when an audio file is missing, empty, unreadable, zero-length, or non-finite."""


@dataclass(frozen=True)
class AudioFileHeader:
    """Metadata inspected from an audio file without full feature extraction."""

    path: str
    exists: bool
    file_size_bytes: int
    sample_rate: int | None
    num_channels: int | None
    num_frames: int | None
    duration_sec: float | None
    is_corrupt: bool
    error_message: str | None = None


@dataclass
class MFCCStandardScaler:
    """Training-only standard scaler for 80-dimensional MFCC summary vectors."""

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    fitted_on_split: str | None = None
    num_samples_seen: int = 0
    feature_dim: int = 80
    eps: float = 1e-8

    def fit(self, features: np.ndarray, split_name: str = "train") -> MFCCStandardScaler:
        """Fit mean and standard deviation strictly on training features.

        Args:
            features: 2D array of shape ``(N, feature_dim)``.
            split_name: Partition identifier; must be ``"train"`` to prevent leakage.

        Returns:
            Self after fitting.

        Raises:
            ValueError: If ``split_name != "train"`` or features have invalid shape/NaNs.
        """
        if split_name.strip().lower() != "train":
            raise ValueError(
                f"Leakage violation: MFCCStandardScaler.fit() allows split='train' only, "
                f"got split='{split_name}'."
            )
        arr = np.asarray(features, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise ValueError(
                f"Expected non-empty 2D feature matrix (N, D), got shape {arr.shape}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("Cannot fit MFCCStandardScaler on features containing NaN or Inf.")

        self.feature_dim = int(arr.shape[1])
        self.mean_ = arr.mean(axis=0).astype(np.float32)
        raw_std = arr.std(axis=0)
        self.scale_ = np.where(raw_std < self.eps, 1.0, raw_std).astype(np.float32)
        self.fitted_on_split = "train"
        self.num_samples_seen = int(arr.shape[0])
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Standardize 1D or 2D MFCC features using fitted training statistics."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("MFCCStandardScaler must be fitted before calling transform().")
        arr = np.asarray(features, dtype=np.float32)
        if arr.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.feature_dim}, got {arr.shape[-1]}."
            )
        scaled = (arr - self.mean_) / self.scale_
        if not np.all(np.isfinite(scaled)):
            raise ValueError("Scaled MFCC features contain NaN or Inf values.")
        return scaled.astype(np.float32)

    def fit_transform(self, features: np.ndarray, split_name: str = "train") -> np.ndarray:
        """Fit on training data and return standardized features."""
        return self.fit(features, split_name=split_name).transform(features)

    def save(self, path: str | Path) -> Path:
        """Persist fitted scaler parameters to a JSON file."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Cannot save an unfitted MFCCStandardScaler.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact_type": "MFCCStandardScaler",
            "fitted_on_split": self.fitted_on_split,
            "num_samples_seen": self.num_samples_seen,
            "feature_dim": self.feature_dim,
            "eps": self.eps,
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> MFCCStandardScaler:
        """Load a fitted ``MFCCStandardScaler`` from a JSON file."""
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"MFCC scaler artifact not found at '{target}'.")
        payload = json.loads(target.read_text(encoding="utf-8"))
        scaler = cls(
            mean_=np.asarray(payload["mean"], dtype=np.float32),
            scale_=np.asarray(payload["scale"], dtype=np.float32),
            fitted_on_split=str(payload.get("fitted_on_split", "train")),
            num_samples_seen=int(payload.get("num_samples_seen", 0)),
            feature_dim=int(payload.get("feature_dim", 80)),
            eps=float(payload.get("eps", 1e-8)),
        )
        return scaler


@dataclass
class MelSpectrogramNormalizer:
    """Training-only normalizer for 2D Log-Mel spectrograms (per-Mel-bin mean and std)."""

    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    fitted_on_split: str | None = None
    num_samples_seen: int = 0
    n_mels: int = 128
    eps: float = 1e-8

    def fit(self, spectrograms: np.ndarray, split_name: str = "train") -> MelSpectrogramNormalizer:
        """Estimate per-Mel-bin mean and standard deviation strictly from training examples.

        Args:
            spectrograms: Array of shape ``(N, 1, n_mels, T)`` or ``(N, n_mels, T)``.
            split_name: Partition identifier; must be ``"train"``.

        Returns:
            Self after fitting.

        Raises:
            ValueError: If ``split_name != "train"`` or input shape/values are invalid.
        """
        if split_name.strip().lower() != "train":
            raise ValueError(
                f"Leakage violation: MelSpectrogramNormalizer.fit() allows split='train' only, "
                f"got split='{split_name}'."
            )
        arr = np.asarray(spectrograms, dtype=np.float64)
        if arr.ndim == 3:
            arr = arr[:, np.newaxis, :, :]
        if arr.ndim != 4 or arr.shape[0] == 0:
            raise ValueError(
                f"Expected 4D spectrogram batch (N, 1, n_mels, T), got shape {arr.shape}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("Cannot fit MelSpectrogramNormalizer on NaN or Inf values.")

        self.n_mels = int(arr.shape[2])
        # Compute mean and std across (N, channel, time) for each mel frequency bin -> (1, n_mels, 1)
        mel_mean = arr.mean(axis=(0, 1, 3), keepdims=False)
        mel_std = arr.std(axis=(0, 1, 3), keepdims=False)
        mel_std = np.where(mel_std < self.eps, 1.0, mel_std)

        self.mean_ = mel_mean.reshape(1, self.n_mels, 1).astype(np.float32)
        self.std_ = mel_std.reshape(1, self.n_mels, 1).astype(np.float32)
        self.fitted_on_split = "train"
        self.num_samples_seen = int(arr.shape[0])
        return self

    def transform(self, spectrogram: np.ndarray) -> np.ndarray:
        """Normalize a single spectrogram ``(1, n_mels, T)`` or batch ``(N, 1, n_mels, T)``."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("MelSpectrogramNormalizer must be fitted before calling transform().")
        arr = np.asarray(spectrogram, dtype=np.float32)
        if arr.shape[-2] != self.n_mels:
            raise ValueError(
                f"Mel bin mismatch: expected {self.n_mels}, got {arr.shape[-2]}."
            )
        normalized = (arr - self.mean_) / self.std_
        if not np.all(np.isfinite(normalized)):
            raise ValueError("Normalized Mel spectrogram contains NaN or Inf values.")
        return normalized.astype(np.float32)

    def fit_transform(
        self, spectrograms: np.ndarray, split_name: str = "train"
    ) -> np.ndarray:
        """Fit on training spectrograms and return normalized batch."""
        return self.fit(spectrograms, split_name=split_name).transform(spectrograms)

    def save(self, path: str | Path) -> Path:
        """Save fitted normalizer statistics to JSON."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Cannot save an unfitted MelSpectrogramNormalizer.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact_type": "MelSpectrogramNormalizer",
            "fitted_on_split": self.fitted_on_split,
            "num_samples_seen": self.num_samples_seen,
            "n_mels": self.n_mels,
            "eps": self.eps,
            "mean": self.mean_.squeeze().tolist(),
            "std": self.std_.squeeze().tolist(),
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> MelSpectrogramNormalizer:
        """Load fitted ``MelSpectrogramNormalizer`` from JSON."""
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"Mel normalizer artifact not found at '{target}'.")
        payload = json.loads(target.read_text(encoding="utf-8"))
        n_mels = int(payload.get("n_mels", 128))
        mean_arr = np.asarray(payload["mean"], dtype=np.float32).reshape(1, n_mels, 1)
        std_arr = np.asarray(payload["std"], dtype=np.float32).reshape(1, n_mels, 1)
        return cls(
            mean_=mean_arr,
            std_=std_arr,
            fitted_on_split=str(payload.get("fitted_on_split", "train")),
            num_samples_seen=int(payload.get("num_samples_seen", 0)),
            n_mels=n_mels,
            eps=float(payload.get("eps", 1e-8)),
        )


_IMA_INDEX_TABLE: tuple[int, ...] = (
    -1, -1, -1, -1, 2, 4, 6, 8,
    -1, -1, -1, -1, 2, 4, 6, 8,
)
_IMA_STEP_TABLE: tuple[int, ...] = (
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
    19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
    50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
    130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
    337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
    876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
    2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
    5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
    15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
)
_MS_ADAPTATION_TABLE: tuple[int, ...] = (
    230, 230, 230, 230, 307, 409, 512, 614,
    768, 614, 512, 409, 307, 230, 230, 230,
)
_MS_DEFAULT_COEFFS: list[tuple[int, int]] = [
    (256, 0), (512, -256), (0, 0), (192, 64), (240, 0), (460, -208), (392, -232)
]


def _decode_adpcm_wav(
    audio_input: Path | str | bytes | bytearray | BinaryIO,
) -> tuple[np.ndarray, int, int] | None:
    """Decode Microsoft ADPCM (0x0002) or IMA/DVI ADPCM (0x0011) WAV data to float32."""
    if isinstance(audio_input, (bytes, bytearray)):
        raw = bytes(audio_input)
    elif hasattr(audio_input, "read"):
        if hasattr(audio_input, "seek"):
            try:
                audio_input.seek(0)
            except Exception:
                pass
        raw_read = audio_input.read()
        if hasattr(audio_input, "seek"):
            try:
                audio_input.seek(0)
            except Exception:
                pass
        raw = raw_read.encode("latin-1") if isinstance(raw_read, str) else bytes(raw_read)
    elif isinstance(audio_input, (str, Path)):
        target = Path(audio_input)
        if not target.is_file():
            return None
        raw = target.read_bytes()
    else:
        return None

    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        return None

    pos = 12
    fmt_tag = 0
    channels = 1
    sample_rate = 0
    block_align = 0
    bits_per_sample = 0
    ms_coeffs = list(_MS_DEFAULT_COEFFS)
    data_bytes = b""

    while pos + 8 <= len(raw):
        chunk_id = raw[pos : pos + 4]
        chunk_size = struct.unpack_from("<I", raw, pos + 4)[0]
        chunk_data = raw[pos + 8 : pos + 8 + chunk_size]
        if chunk_id == b"fmt " and len(chunk_data) >= 16:
            fmt_tag, channels, sample_rate, _, block_align, bits_per_sample = struct.unpack_from(
                "<HHIIHH", chunk_data, 0
            )
            if fmt_tag == 2 and len(chunk_data) >= 22:
                num_coeff = struct.unpack_from("<H", chunk_data, 20)[0]
                if len(chunk_data) >= 22 + 4 * num_coeff and num_coeff > 0:
                    ms_coeffs = [
                        struct.unpack_from("<hh", chunk_data, 22 + 4 * i)
                        for i in range(num_coeff)
                    ]
        elif chunk_id == b"data":
            data_bytes = chunk_data
            break
        pos += 8 + chunk_size + (chunk_size & 1)

    if not data_bytes or sample_rate <= 0 or block_align <= 0 or channels != 1 or bits_per_sample != 4:
        return None

    samples: list[int] = []
    if fmt_tag == 17:  # IMA / DVI ADPCM (0x0011)
        for offset in range(0, len(data_bytes), block_align):
            block = data_bytes[offset : offset + block_align]
            if len(block) < 4:
                break
            pred, step_idx, _ = struct.unpack_from("<hBB", block, 0)
            step_idx = max(0, min(88, step_idx))
            samples.append(pred)
            for b in block[4:]:
                for nib in (b & 0x0F, (b >> 4) & 0x0F):
                    step = _IMA_STEP_TABLE[step_idx]
                    diff = step >> 3
                    if nib & 1:
                        diff += step >> 2
                    if nib & 2:
                        diff += step >> 1
                    if nib & 4:
                        diff += step
                    pred = max(-32768, pred - diff) if (nib & 8) else min(32767, pred + diff)
                    step_idx = max(0, min(88, step_idx + _IMA_INDEX_TABLE[nib]))
                    samples.append(pred)
    elif fmt_tag == 2:  # Microsoft ADPCM (0x0002)
        for offset in range(0, len(data_bytes), block_align):
            block = data_bytes[offset : offset + block_align]
            if len(block) < 7:
                break
            bpred, idelta, samp1, samp2 = struct.unpack_from("<Bhhh", block, 0)
            c1, c2 = ms_coeffs[bpred] if bpred < len(ms_coeffs) else (256, 0)
            samples.append(samp2)
            samples.append(samp1)
            for b in block[7:]:
                for u_nib in ((b >> 4) & 0x0F, b & 0x0F):
                    s_nib = u_nib - 16 if u_nib >= 8 else u_nib
                    pred = ((samp1 * c1) + (samp2 * c2)) >> 8
                    pred = max(-32768, min(32767, pred + s_nib * idelta))
                    samp2 = samp1
                    samp1 = pred
                    idelta = max(16, (_MS_ADAPTATION_TABLE[u_nib] * idelta) >> 8)
                    samples.append(pred)
    else:
        return None

    if not samples:
        return None
    floats = (np.asarray(samples, dtype=np.float32) / 32768.0).astype(np.float32)
    return np.clip(floats, -1.0, 1.0), int(sample_rate), int(channels)


def _decode_raw_wav(
    audio_source: Path | str | bytes | bytearray | BinaryIO,
) -> tuple[np.ndarray, int, int]:
    """Read a WAV audio source into a float32 array in [-1.0, 1.0], returning (samples, sr, channels).

    Supports standard PCM, IEEE float, and Microsoft/IMA ADPCM WAV formats from
    paths, raw bytes, or in-memory binary streams.
    """
    raw_bytes: bytes
    source_name: str

    if isinstance(audio_source, (bytes, bytearray)):
        raw_bytes = bytes(audio_source)
        source_name = f"<in-memory buffer ({len(raw_bytes)} bytes)>"
        if len(raw_bytes) == 0:
            raise CorruptAudioError("Audio input buffer is empty (0 bytes).")
    elif hasattr(audio_source, "read"):
        if hasattr(audio_source, "seek"):
            try:
                audio_source.seek(0)
            except Exception:
                pass
        raw_read = audio_source.read()
        if hasattr(audio_source, "seek"):
            try:
                audio_source.seek(0)
            except Exception:
                pass
        raw_bytes = raw_read.encode("latin-1") if isinstance(raw_read, str) else bytes(raw_read)
        s_name = getattr(audio_source, "name", "stream")
        source_name = f"<{s_name} ({len(raw_bytes)} bytes)>"
        if len(raw_bytes) == 0:
            raise CorruptAudioError(f"Audio stream '{source_name}' is empty (0 bytes).")
    elif isinstance(audio_source, (str, Path)):
        file_path = Path(audio_source)
        source_name = str(file_path)
        if not file_path.exists():
            raise CorruptAudioError(f"Audio file does not exist: '{file_path}'")
        if not file_path.is_file():
            raise CorruptAudioError(f"Audio path is not a file: '{file_path}'")
        try:
            size_bytes = file_path.stat().st_size
        except OSError as exc:
            raise CorruptAudioError(f"Cannot stat audio file '{file_path}': {exc}") from exc
        if size_bytes == 0:
            raise CorruptAudioError(f"Audio file is empty (0 bytes): '{file_path}'")
        raw_bytes = file_path.read_bytes()
    else:
        raise TypeError(
            f"Unsupported audio source type: {type(audio_source).__name__}. "
            "Expected str, Path, bytes, or file-like binary stream."
        )

    # Check for non-WAV formats (MP3, OGG, FLAC) to provide clear user feedback
    if raw_bytes.startswith(b"ID3") or (
        len(raw_bytes) >= 2 and raw_bytes[0] == 0xFF and (raw_bytes[1] & 0xE0) == 0xE0
    ):
        raise CorruptAudioError(
            f"Unsupported audio format: MP3 detected in '{source_name}'. "
            "The model requires WAV format audio recordings (PCM, IEEE float, or ADPCM)."
        )
    if raw_bytes.startswith(b"OggS"):
        raise CorruptAudioError(
            f"Unsupported audio format: OGG Vorbis detected in '{source_name}'. "
            "The model requires WAV format audio recordings."
        )
    if raw_bytes.startswith(b"fLaC"):
        raise CorruptAudioError(
            f"Unsupported audio format: FLAC detected in '{source_name}'. "
            "The model requires WAV format audio recordings."
        )

    # Primary decoder: scipy.io.wavfile using BytesIO stream
    bio = io.BytesIO(raw_bytes)
    try:
        sample_rate, data = wavfile.read(bio)
    except Exception as scipy_exc:  # noqa: BLE001
        # Fallback 1: Microsoft ADPCM (0x0002) or IMA/DVI ADPCM (0x0011) in official UrbanSound8K
        try:
            adpcm_decoded = _decode_adpcm_wav(raw_bytes)
            if adpcm_decoded is not None:
                return adpcm_decoded
        except Exception:  # noqa: BLE001
            pass
        # Fallback 2: standard wave module for 24-bit PCM WAV files if scipy raises
        try:
            bio.seek(0)
            with wave.open(bio, "rb") as wf:
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                sampwidth = wf.getsampwidth()
                nframes = wf.getnframes()
                if nframes <= 0 or sample_rate <= 0:
                    raise CorruptAudioError(
                        f"Audio file has zero frames or invalid sample rate: '{source_name}'"
                    )
                frame_bytes = wf.readframes(nframes)
                if sampwidth == 3:
                    raw_u8 = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(-1, 3)
                    signed = (
                        raw_u8[:, 0].astype(np.int32)
                        | (raw_u8[:, 1].astype(np.int32) << 8)
                        | (raw_u8[:, 2].astype(np.int32) << 16)
                    )
                    mask = signed & 0x800000
                    signed = signed - (mask << 1)
                    floats = (signed / float(1 << 23)).astype(np.float32)
                    if channels > 1:
                        floats = floats.reshape(-1, channels)
                    return floats, int(sample_rate), int(channels)
        except Exception:  # noqa: BLE001
            pass
        raise CorruptAudioError(
            f"Unreadable or corrupt audio file '{source_name}': {scipy_exc}"
        ) from scipy_exc

    if sample_rate <= 0:
        raise CorruptAudioError(f"Invalid sample rate ({sample_rate}) in '{source_name}'.")
    if data.size == 0 or data.shape[0] == 0:
        raise CorruptAudioError(f"Zero-duration audio recording in '{source_name}'.")

    num_channels = 1 if data.ndim == 1 else int(data.shape[1])

    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        max_abs = float(max(abs(info.min), abs(info.max)))
        if data.dtype == np.uint8:
            samples = ((data.astype(np.float32) - 128.0) / 128.0).astype(np.float32)
        else:
            samples = data.astype(np.float32)
            samples /= np.float32(max_abs)
    elif np.issubdtype(data.dtype, np.floating):
        samples = data.astype(np.float32)
    else:
        raise CorruptAudioError(f"Unsupported audio sample dtype {data.dtype} in '{source_name}'.")

    if not np.all(np.isfinite(samples)):
        raise CorruptAudioError(f"Audio file contains NaN or Inf samples: '{source_name}'.")

    return samples, int(sample_rate), num_channels


def inspect_audio_file(file_path: str | Path) -> AudioFileHeader:
    """Inspect an audio file for audit reporting without failing the caller.

    Args:
        file_path: Path to the candidate audio file.

    Returns:
        ``AudioFileHeader`` containing sample rate, channels, duration, and corruption status.
    """
    target = Path(file_path)
    if not target.exists() or not target.is_file():
        return AudioFileHeader(
            path=str(target),
            exists=False,
            file_size_bytes=0,
            sample_rate=None,
            num_channels=None,
            num_frames=None,
            duration_sec=None,
            is_corrupt=True,
            error_message="File does not exist",
        )

    size_bytes = int(target.stat().st_size)
    try:
        samples, sr, channels = _decode_raw_wav(target)
        num_frames = int(samples.shape[0])
        duration = float(num_frames) / float(sr)
        if duration <= 0.0:
            return AudioFileHeader(
                path=str(target),
                exists=True,
                file_size_bytes=size_bytes,
                sample_rate=sr,
                num_channels=channels,
                num_frames=num_frames,
                duration_sec=duration,
                is_corrupt=True,
                error_message="Zero-duration recording",
            )
        return AudioFileHeader(
            path=str(target),
            exists=True,
            file_size_bytes=size_bytes,
            sample_rate=sr,
            num_channels=channels,
            num_frames=num_frames,
            duration_sec=duration,
            is_corrupt=False,
            error_message=None,
        )
    except CorruptAudioError as exc:
        return AudioFileHeader(
            path=str(target),
            exists=True,
            file_size_bytes=size_bytes,
            sample_rate=None,
            num_channels=None,
            num_frames=None,
            duration_sec=None,
            is_corrupt=True,
            error_message=str(exc),
        )


def standardize_waveform(
    samples: np.ndarray,
    orig_sr: int,
    target_sr: int = 22050,
    target_duration_sec: float = 4.0,
) -> np.ndarray:
    """Convert raw samples to mono, resample to ``target_sr``, and pad/truncate to fixed length.

    Args:
        samples: Raw 1D ``(T,)`` or 2D ``(T, C)`` float32 waveform.
        orig_sr: Original sampling rate in Hz.
        target_sr: Target sampling rate in Hz (default: 22,050 Hz).
        target_duration_sec: Target clip duration in seconds (default: 4.0 s).

    Returns:
        1D ``np.ndarray`` of shape ``(int(target_sr * target_duration_sec),)`` with ``float32`` dtype.

    Raises:
        CorruptAudioError: If inputs are empty, non-finite, or have non-positive sample rates.
    """
    if orig_sr <= 0 or target_sr <= 0 or target_duration_sec <= 0:
        raise CorruptAudioError("Sample rates and target duration must be strictly positive.")

    arr = np.asarray(samples, dtype=np.float32)
    if arr.size == 0 or arr.shape[0] == 0:
        raise CorruptAudioError("Cannot standardize an empty waveform.")
    if not np.all(np.isfinite(arr)):
        raise CorruptAudioError("Input waveform contains NaN or Inf values.")

    # 1. Consistent mono conversion by channel averaging
    if arr.ndim == 2:
        mono = arr.mean(axis=1, dtype=np.float32)
    elif arr.ndim == 1:
        mono = arr
    else:
        raise CorruptAudioError(f"Unsupported waveform dimensionality: {arr.ndim}")

    # 2. Resample to target_sr if needed
    if orig_sr != target_sr:
        gcd = math.gcd(int(orig_sr), int(target_sr))
        up = int(target_sr) // gcd
        down = int(orig_sr) // gcd
        mono = spsignal.resample_poly(mono, up=up, down=down).astype(np.float32)

    # 3. Pad or truncate to exact target_num_samples
    target_len = int(round(target_sr * target_duration_sec))
    current_len = int(mono.shape[0])
    if current_len < target_len:
        pad_width = target_len - current_len
        standardized = np.pad(mono, (0, pad_width), mode="constant", constant_values=0.0)
    elif current_len > target_len:
        standardized = mono[:target_len]
    else:
        standardized = mono

    standardized = np.ascontiguousarray(standardized, dtype=np.float32)
    if not np.all(np.isfinite(standardized)):
        raise CorruptAudioError("Standardized waveform contains NaN or Inf values.")
    return standardized


def load_and_standardize_audio(
    audio_source: str | Path | bytes | bytearray | BinaryIO,
    target_sr: int = 22050,
    target_duration_sec: float = 4.0,
) -> np.ndarray:
    """Load an audio file, raw byte buffer, or stream and standardize to fixed-length mono float32 waveform.

    Args:
        audio_source: Path to the WAV audio file, or raw bytes / stream buffer.
        target_sr: Target sample rate in Hz (default: 22,050 Hz).
        target_duration_sec: Fixed duration in seconds (default: 4.0 s -> 88,200 samples).

    Returns:
        1D ``np.ndarray`` of shape ``(88200,)`` with ``float32`` dtype.
    """
    samples, orig_sr, _ = _decode_raw_wav(audio_source)
    return standardize_waveform(
        samples=samples,
        orig_sr=orig_sr,
        target_sr=target_sr,
        target_duration_sec=target_duration_sec,
    )


@functools.lru_cache(maxsize=8)
def _get_cached_mfcc_transform(
    sample_rate: int,
    n_mfcc: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    n_mels: int,
    f_min: float,
    f_max: float,
) -> torchaudio.transforms.MFCC:
    return torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
            "n_mels": n_mels,
            "f_min": f_min,
            "f_max": f_max,
            "center": True,
            "power": 2.0,
        },
    )


@functools.lru_cache(maxsize=8)
def _get_cached_mel_db_transforms(
    sample_rate: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    top_db: float,
) -> tuple[torchaudio.transforms.MelSpectrogram, torchaudio.transforms.AmplitudeToDB]:
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        power=2.0,
        center=True,
    )
    db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=top_db)
    return mel_transform, db_transform


def extract_mfcc_features(
    waveform: np.ndarray,
    sample_rate: int = 22050,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = 2048,
    n_mels: int = 128,
    f_min: float = 0.0,
    f_max: float = 11025.0,
) -> np.ndarray:
    """Extract an 80-dimensional MFCC summary vector (40 temporal means + 40 temporal stds).

    Designed for the Random Forest baseline and Deep MLP architecture.

    Args:
        waveform: 1D standardized mono float32 waveform of shape ``(N,)``.
        sample_rate: Sampling rate in Hz (default: 22,050).
        n_mfcc: Number of MFCC coefficients (default: 40).
        n_fft: FFT window size (default: 2048).
        hop_length: STFT hop length in samples (default: 512).
        win_length: Window length in samples (default: 2048).
        n_mels: Number of Mel filterbanks (default: 128).
        f_min: Minimum frequency in Hz (default: 0.0).
        f_max: Maximum frequency in Hz (default: 11,025.0).

    Returns:
        1D ``np.ndarray`` of shape ``(2 * n_mfcc,)`` (i.e. ``(80,)``) and dtype ``float32``.

    Raises:
        ValueError: If input waveform is invalid or output contains NaN/Inf.
    """
    arr = np.asarray(waveform, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"Expected non-empty 1D waveform, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input waveform contains NaN or Inf values.")

    tensor_wave = torch.from_numpy(arr).unsqueeze(0)  # (1, N)
    mfcc_transform = _get_cached_mfcc_transform(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
    )
    with torch.no_grad():
        mfcc_matrix = mfcc_transform(tensor_wave).squeeze(0).numpy()  # (n_mfcc, T)

    mfcc_mean = mfcc_matrix.mean(axis=1, dtype=np.float64).astype(np.float32)
    mfcc_std = mfcc_matrix.std(axis=1, dtype=np.float64).astype(np.float32)
    summary_vec = np.concatenate([mfcc_mean, mfcc_std], axis=0).astype(np.float32)

    expected_dim = 2 * n_mfcc
    if summary_vec.shape != (expected_dim,):
        raise ValueError(
            f"MFCC feature dimension mismatch: expected ({expected_dim},), got {summary_vec.shape}."
        )
    if not np.all(np.isfinite(summary_vec)):
        raise ValueError("Extracted MFCC features contain NaN or Inf values.")

    return summary_vec


def extract_log_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = 2048,
    f_min: float = 0.0,
    f_max: float = 11025.0,
    top_db: float = 80.0,
) -> np.ndarray:
    """Extract a fixed-size 2D Log-Mel spectrogram tensor of shape ``(1, n_mels, time_frames)``.

    Designed for the 2D CNN architecture. For a 4.0 s clip at 22,050 Hz (88,200 samples)
    with ``hop_length=512``, ``time_frames = 173``, yielding shape ``(1, 128, 173)``.

    Args:
        waveform: 1D standardized mono float32 waveform of shape ``(N,)``.
        sample_rate: Sampling rate in Hz (default: 22,050).
        n_mels: Number of Mel frequency bins (default: 128).
        n_fft: FFT size (default: 2048).
        hop_length: STFT hop length (default: 512).
        win_length: STFT window length (default: 2048).
        f_min: Minimum frequency in Hz (default: 0.0).
        f_max: Maximum frequency in Hz (default: 11,025.0).
        top_db: Dynamic range threshold in dB (default: 80.0).

    Returns:
        3D ``np.ndarray`` of shape ``(1, n_mels, T)`` and dtype ``float32``.

    Raises:
        ValueError: If input waveform is invalid or output contains NaN/Inf.
    """
    arr = np.asarray(waveform, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"Expected non-empty 1D waveform, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input waveform contains NaN or Inf values.")

    tensor_wave = torch.from_numpy(arr).unsqueeze(0)  # (1, N)
    mel_transform, db_transform = _get_cached_mel_db_transforms(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        top_db=top_db,
    )

    with torch.no_grad():
        mel_power = mel_transform(tensor_wave)  # (1, n_mels, T)
        log_mel = db_transform(mel_power).numpy().astype(np.float32)

    if log_mel.ndim != 3 or log_mel.shape[0] != 1 or log_mel.shape[1] != n_mels:
        raise ValueError(f"Unexpected Log-Mel spectrogram shape: {log_mel.shape}")
    if not np.all(np.isfinite(log_mel)):
        raise ValueError("Extracted Log-Mel spectrogram contains NaN or Inf values.")

    return log_mel


def save_preprocessing_metadata(
    output_path: str | Path,
    data_config: dict[str, Any],
    mfcc_scaler_path: str | Path | None = None,
    mel_norm_path: str | Path | None = None,
) -> Path:
    """Save a reproducible record of preprocessing settings and fitted artifact locations."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "audio_signal": data_config.get("audio_signal", {}),
        "mfcc_config": data_config.get("features", {}).get("mfcc", {}),
        "mel_spectrogram_config": data_config.get("features", {}).get("mel_spectrogram", {}),
        "mfcc_scaler_artifact": str(mfcc_scaler_path) if mfcc_scaler_path else None,
        "mel_norm_stats_artifact": str(mel_norm_path) if mel_norm_path else None,
        "fitted_exclusively_on_split": "train",
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target

