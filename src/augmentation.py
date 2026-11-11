"""Training-only waveform augmentation pipeline for ANN-Siren-Detection.

Module: NB6057CEM — Artificial Neural Networks
Step: 2 — Dataset Acquisition, Data Audit and Leakage-Safe Preparation (CW1 T1)

Supported label-preserving waveform transformations:
1. Controlled additive background noise at a specified Signal-to-Noise Ratio (SNR in dB).
2. Small circular or zero-padded time shifts.
3. Controlled amplitude (gain) scaling.

Strict Policy:
- Stochastic augmentation is permitted on the 'train' split ONLY.
- Applying stochastic augmentation to 'val' or 'test' splits during normal evaluation
  is strictly prohibited and guarded by ``AugmentationPolicyError``.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


class AugmentationPolicyError(ValueError):
    """Raised when stochastic augmentation is invoked on a non-training split or invalid parameter."""


def add_controlled_background_noise(
    waveform: np.ndarray,
    snr_db: float = 15.0,
    rng: np.random.Generator | None = None,
    noise_waveform: np.ndarray | None = None,
) -> np.ndarray:
    """Add controlled Gaussian or supplied background noise at a target SNR (dB).

    Args:
        waveform: 1D float32 audio signal of shape ``(N,)``.
        snr_db: Target signal-to-noise ratio in decibels. Must be ``>= -5.0`` dB to
            preserve class label semantics.
        rng: Optional ``np.random.Generator`` for reproducible noise synthesis.
        noise_waveform: Optional 1D background noise clip of length ``N``. If omitted,
            zero-mean unit-variance white Gaussian noise is generated via ``rng``.

    Returns:
        1D ``np.ndarray`` of shape ``(N,)`` and dtype ``float32``.

    Raises:
        AugmentationPolicyError: If ``snr_db < -5.0`` (which could mask the target class).
    """
    if snr_db < -5.0:
        raise AugmentationPolicyError(
            f"snr_db={snr_db} is too low to guarantee label preservation (minimum allowed: -5.0 dB)."
        )

    sig = np.asarray(waveform, dtype=np.float32)
    if sig.ndim != 1 or sig.size == 0:
        raise ValueError(f"Expected non-empty 1D waveform, got shape {sig.shape}.")

    gen = rng if rng is not None else np.random.default_rng(42)
    if noise_waveform is not None:
        noise = np.asarray(noise_waveform, dtype=np.float32)
        if noise.shape != sig.shape:
            raise ValueError(
                f"noise_waveform shape {noise.shape} does not match signal shape {sig.shape}."
            )
    else:
        noise = gen.standard_normal(size=sig.shape).astype(np.float32)

    sig_power = float(np.mean(sig.astype(np.float64) ** 2))
    noise_power = float(np.mean(noise.astype(np.float64) ** 2))
    if sig_power <= 1e-12 or noise_power <= 1e-12:
        return sig.copy()

    target_noise_power = sig_power / (10.0 ** (float(snr_db) / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)
    augmented = sig + (noise * float(scale))
    return np.clip(augmented, -1.0, 1.0).astype(np.float32)


def apply_time_shift(
    waveform: np.ndarray,
    max_shift_fraction: float = 0.1,
    rng: np.random.Generator | None = None,
    mode: str = "roll",
) -> np.ndarray:
    """Apply a small random time shift within ``[-max_shift_fraction, +max_shift_fraction]``.

    Args:
        waveform: 1D float32 audio signal of shape ``(N,)``.
        max_shift_fraction: Maximum fraction of total clip length to shift (must be in ``[0.0, 0.25]``).
        rng: Optional ``np.random.Generator`` for reproducible shift sampling.
        mode: ``"roll"`` (circular shift preserving total energy) or ``"zero_pad"``.

    Returns:
        1D ``np.ndarray`` of shape ``(N,)`` and dtype ``float32``.

    Raises:
        AugmentationPolicyError: If ``max_shift_fraction`` is outside ``[0.0, 0.25]``.
    """
    if not (0.0 <= max_shift_fraction <= 0.25):
        raise AugmentationPolicyError(
            f"max_shift_fraction must be within [0.0, 0.25] to preserve label semantics, "
            f"got {max_shift_fraction}."
        )

    sig = np.asarray(waveform, dtype=np.float32)
    if sig.ndim != 1 or sig.size == 0:
        raise ValueError(f"Expected non-empty 1D waveform, got shape {sig.shape}.")

    if max_shift_fraction == 0.0:
        return sig.copy()

    gen = rng if rng is not None else np.random.default_rng(42)
    max_samples = int(round(sig.shape[0] * max_shift_fraction))
    if max_samples == 0:
        return sig.copy()

    shift = int(gen.integers(-max_samples, max_samples + 1))
    if shift == 0:
        return sig.copy()

    if mode == "roll":
        return np.roll(sig, shift).astype(np.float32)
    if mode == "zero_pad":
        result = np.zeros_like(sig)
        if shift > 0:
            result[shift:] = sig[:-shift]
        else:
            result[:shift] = sig[-shift:]
        return result
    raise ValueError(f"Unsupported time shift mode '{mode}'. Use 'roll' or 'zero_pad'.")


def apply_amplitude_scaling(
    waveform: np.ndarray,
    scale_range: tuple[float, float] = (0.85, 1.15),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Scale waveform amplitude by a random factor drawn uniformly from ``scale_range``.

    Args:
        waveform: 1D float32 audio signal of shape ``(N,)``.
        scale_range: ``(min_gain, max_gain)`` tuple. Must satisfy ``0.2 <= min_gain <= max_gain <= 2.5``.
        rng: Optional ``np.random.Generator``.

    Returns:
        1D ``np.ndarray`` of shape ``(N,)`` and dtype ``float32``.

    Raises:
        AugmentationPolicyError: If ``scale_range`` bounds are invalid.
    """
    low, high = float(scale_range[0]), float(scale_range[1])
    if low < 0.2 or high > 2.5 or low > high:
        raise AugmentationPolicyError(
            f"Invalid amplitude scale_range ({low}, {high}); expected 0.2 <= min <= max <= 2.5."
        )

    sig = np.asarray(waveform, dtype=np.float32)
    if sig.ndim != 1 or sig.size == 0:
        raise ValueError(f"Expected non-empty 1D waveform, got shape {sig.shape}.")

    gen = rng if rng is not None else np.random.default_rng(42)
    gain = float(gen.uniform(low, high))
    return np.clip(sig * gain, -1.0, 1.0).astype(np.float32)


@dataclass
class TrainingAudioAugmenter:
    """Composable, seed-reproducible waveform augmenter restricted to the training split.

    Attributes:
        enabled: Whether stochastic augmentation is active.
        probability: Per-transform application probability in ``[0.0, 1.0]``.
        snr_db_range: ``(min_snr_db, max_snr_db)`` for additive background noise.
        max_time_shift_fraction: Maximum fractional time shift in ``[0.0, 0.25]``.
        amplitude_scale_range: ``(min_gain, max_gain)`` for amplitude scaling.
        seed: Base integer random seed for deterministic reproducibility.
    """

    enabled: bool = False
    probability: float = 0.5
    snr_db_range: tuple[float, float] = (10.0, 25.0)
    max_time_shift_fraction: float = 0.1
    amplitude_scale_range: tuple[float, float] = (0.85, 1.15)
    seed: int = 42

    def __post_init__(self) -> None:
        if not (0.0 <= self.probability <= 1.0):
            raise AugmentationPolicyError(
                f"probability must be in [0.0, 1.0], got {self.probability}."
            )
        if self.snr_db_range[0] < -5.0 or self.snr_db_range[0] > self.snr_db_range[1]:
            raise AugmentationPolicyError(f"Invalid snr_db_range: {self.snr_db_range}")
        if not (0.0 <= self.max_time_shift_fraction <= 0.25):
            raise AugmentationPolicyError(
                f"Invalid max_time_shift_fraction: {self.max_time_shift_fraction}"
            )
        if (
            self.amplitude_scale_range[0] < 0.2
            or self.amplitude_scale_range[1] > 2.5
            or self.amplitude_scale_range[0] > self.amplitude_scale_range[1]
        ):
            raise AugmentationPolicyError(
                f"Invalid amplitude_scale_range: {self.amplitude_scale_range}"
            )

    def __call__(
        self,
        waveform: np.ndarray,
        split: str = "train",
        sample_index: int = 0,
    ) -> np.ndarray:
        """Apply stochastic augmentation if ``enabled=True`` and ``split == 'train'``.

        Args:
            waveform: 1D float32 standardized waveform.
            split: Dataset partition name (``"train"``, ``"val"``, or ``"test"``).
            sample_index: Non-negative integer index mixed with ``self.seed`` for
                reproducible per-example augmentation.

        Returns:
            1D float32 waveform (augmented only when ``enabled=True`` and ``split=='train'``).

        Raises:
            AugmentationPolicyError: If ``enabled=True`` and ``split`` is ``"val"`` or ``"test"``.
        """
        normalized_split = split.strip().lower()
        if normalized_split != "train":
            if self.enabled:
                raise AugmentationPolicyError(
                    f"Stochastic augmentation is forbidden on split='{split}'. "
                    "Augmentation may only be enabled for the 'train' split."
                )
            return np.asarray(waveform, dtype=np.float32).copy()

        if not self.enabled:
            return np.asarray(waveform, dtype=np.float32).copy()

        # Deterministic per-sample RNG derived from (self.seed, sample_index)
        rng = np.random.default_rng([int(self.seed), int(sample_index)])
        out = np.asarray(waveform, dtype=np.float32).copy()

        if rng.random() < self.probability:
            out = apply_time_shift(
                out, max_shift_fraction=self.max_time_shift_fraction, rng=rng
            )
        if rng.random() < self.probability:
            out = apply_amplitude_scaling(
                out, scale_range=self.amplitude_scale_range, rng=rng
            )
        if rng.random() < self.probability:
            snr_db = float(rng.uniform(self.snr_db_range[0], self.snr_db_range[1]))
            out = add_controlled_background_noise(out, snr_db=snr_db, rng=rng)

        return out

