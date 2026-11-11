"""CNN + GRU Hybrid (CRNN) Architecture for Spectro-Temporal Emergency Siren Detection.

Module: NB6057CEM — Artificial Neural Networks
Portfolio Upgrade: Phase 3 — Temporal Deep Learning (CNN + GRU Hybrid)

Architecture Overview:
    Input: (B, 1, 128, 173)  [Log-Mel Spectrogram: Batch, Channel, Freq, Time]
    1. Local Spectral Feature Extraction (2D CNN Blocks):
       - Block 1: Conv2d(1, 16, 3x3, pad=1) + BatchNorm + ReLU + MaxPool2d(2, 2)  -> (B, 16, 64, 86)
       - Block 2: Conv2d(16, 32, 3x3, pad=1) + BatchNorm + ReLU + MaxPool2d(2, 2) -> (B, 32, 32, 43)
       - Block 3: Conv2d(32, 64, 3x3, pad=1) + BatchNorm + ReLU                   -> (B, 64, 32, 43)
    2. Spectral Compression & Temporal Preservation:
       - AdaptiveAvgPool2d((1, None)) collapses frequency (32 -> 1) while preserving 43 time steps -> (B, 64, 1, 43)
    3. Reshape for Sequence Modeling:
       - Squeeze frequency and permute to (B, T=43, D=64)
    4. Bidirectional Temporal Modeling:
       - Bidirectional GRU(64 -> 64 per direction) -> (B, 43, 128)
    5. Temporal Context Pooling:
       - Global Mean Pooling across time frames -> (B, 128)
    6. Classification Head:
       - Dropout(0.3) -> Linear(128, 32) -> ReLU -> Dropout(0.3) -> Linear(32, 1) -> Binary Logits (B, 1)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src import set_random_seeds
from src.models import count_pytorch_parameters


class CNNGRUHybrid(nn.Module):
    """Convolutional Recurrent Neural Network (CNN + GRU) for Audio Spectrograms.

    Combines 2D spatial convolutions for local acoustic/spectral pattern extraction
    with a recurrent GRU layer for temporal sequence modeling over time frames.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: tuple[int, int, int] | list[int] = (16, 32, 64),
        gru_hidden_size: int = 64,
        gru_num_layers: int = 1,
        bidirectional: bool = True,
        n_mels: int = 128,
        time_frames: int = 173,
        dropout: float = 0.3,
        output_dim: int = 1,
        seed: int | None = 42,
    ) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError(f"CNNGRUHybrid expects in_channels=1, got {in_channels}.")
        if len(conv_channels) != 3 or any(c <= 0 for c in conv_channels):
            raise ValueError(
                f"CNNGRUHybrid requires exactly 3 positive conv channel sizes, got {conv_channels}."
            )
        if gru_hidden_size <= 0:
            raise ValueError(f"gru_hidden_size must be positive, got {gru_hidden_size}.")
        if gru_num_layers <= 0:
            raise ValueError(f"gru_num_layers must be positive, got {gru_num_layers}.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}.")
        if output_dim != 1:
            raise ValueError(f"Binary CNNGRUHybrid requires output_dim=1, got {output_dim}.")

        if seed is not None:
            set_random_seeds(int(seed))

        self.in_channels = int(in_channels)
        self.conv_channels = tuple(int(c) for c in conv_channels)
        self.gru_hidden_size = int(gru_hidden_size)
        self.gru_num_layers = int(gru_num_layers)
        self.bidirectional = bool(bidirectional)
        self.n_mels = int(n_mels)
        self.time_frames = int(time_frames)
        self.dropout_rate = float(dropout)
        self.output_dim = int(output_dim)
        self.seed = seed

        c1, c2, c3 = self.conv_channels

        # 1. 2D Convolutional Blocks (Local Spectro-Temporal Feature Extraction)
        self.block1 = nn.Sequential(
            nn.Conv2d(self.in_channels, c1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
        )

        # 2. Spectral Compression: collapse frequency to 1 while preserving all temporal steps
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        # 3. Recurrent Temporal Layer (GRU)
        self.gru = nn.GRU(
            input_size=c3,
            hidden_size=self.gru_hidden_size,
            num_layers=self.gru_num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        gru_out_dim = self.gru_hidden_size * (2 if self.bidirectional else 1)

        # 4. Dense Classification Head
        self.head = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(gru_out_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(32, self.output_dim),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply Kaiming normal init to Conv2d, unit/zero init to BatchNorm, Xavier to Linear, and Orthogonal to GRU."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU):
                for param_name, param in module.named_parameters():
                    if "weight_ih" in param_name:
                        nn.init.xavier_normal_(param.data)
                    elif "weight_hh" in param_name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in param_name:
                        param.data.fill_(0)

    def extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract intermediate CNN representation and temporal GRU sequence.

        Args:
            x: Input tensor `(B, 1, 128, 173)`.

        Returns:
            Tuple of (cnn_temporal_features `(B, T, C3)`, gru_sequence `(B, T, D_gru)`).
        """
        x1 = self.block1(x)                                    # (B, C1, 64, 86)
        x2 = self.block2(x1)                                   # (B, C2, 32, 43)
        x3 = self.block3(x2)                                   # (B, C3, 32, 43)
        pooled = self.freq_pool(x3)                            # (B, C3, 1, 43)
        seq_in = pooled.squeeze(2).permute(0, 2, 1)            # (B, 43, C3)
        gru_out, _ = self.gru(seq_in)                          # (B, 43, D_gru)
        return seq_in, gru_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute binary classification logits `(B, 1)` from Log-Mel Spectrograms.

        Args:
            x: Spectrogram tensor `(B, 1, 128, 173)`.

        Returns:
            Raw classification logits `(B, 1)`.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor input, got {type(x).__name__}.")
        if (
            x.ndim != 4
            or x.shape[0] == 0
            or x.shape[1] != self.in_channels
            or x.shape[2] != self.n_mels
            or x.shape[3] != self.time_frames
        ):
            raise ValueError(
                f"CNNGRUHybrid expected 4D tensor of shape (B, {self.in_channels}, "
                f"{self.n_mels}, {self.time_frames}) with B > 0, got {tuple(x.shape)}."
            )
        if not torch.isfinite(x).all():
            raise ValueError("CNNGRUHybrid input tensor contains NaN or Inf values.")

        _, gru_out = self.extract_features(x)                  # (B, T, D_gru)
        # Global mean pooling across temporal dimension
        temporal_context = gru_out.mean(dim=1)                 # (B, D_gru)
        logits = self.head(temporal_context)                   # (B, 1)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute calibrated siren probabilities in `(0, 1)` of shape `(B, 1)`."""
        was_training = self.training
        self.eval()
        try:
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        finally:
            self.train(was_training)
        return probs

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Predict binary labels `{0, 1}` of shape `(B, 1)`."""
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"threshold must be in (0, 1), got {threshold}.")
        probs = self.predict_proba(x)
        return (probs >= threshold).to(dtype=torch.long)

    def count_trainable_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return count_pytorch_parameters(self, trainable_only=True)

    def architecture_summary(self) -> dict[str, Any]:
        """Return complete tensor-shape progression and parameter breakdown."""
        c1, c2, c3 = self.conv_channels
        gru_out_dim = self.gru_hidden_size * (2 if self.bidirectional else 1)
        return {
            "model_id": "cnn_gru",
            "model_name": "CNN + GRU Hybrid (CRNN)",
            "input_representation": "2D Log-Mel Spectrogram (Normalized)",
            "input_shape": [self.in_channels, self.n_mels, self.time_frames],
            "conv_channels": list(self.conv_channels),
            "spatial_pool_progression": [
                f"(B, {c1}, 64, 86)",
                f"(B, {c2}, 32, 43)",
                f"(B, {c3}, 32, 43)",
            ],
            "freq_pool": "AdaptiveAvgPool2d((1, None)) -> (B, 64, 1, 43)",
            "sequence_shape": f"(B, 43, {c3})",
            "gru_config": {
                "input_size": c3,
                "hidden_size": self.gru_hidden_size,
                "num_layers": self.gru_num_layers,
                "bidirectional": self.bidirectional,
                "output_dim": gru_out_dim,
            },
            "temporal_pooling": "Global Mean Pooling across time -> (B, 128)",
            "classification_head": f"Dropout({self.dropout_rate}) -> Linear({gru_out_dim}, 32) -> ReLU -> Dropout({self.dropout_rate}) -> Linear(32, 1)",
            "output_shape": [self.output_dim],
            "loss_function": "BCEWithLogitsLoss",
            "optimizer": "Adam",
            "trainable_parameters": self.count_trainable_parameters(),
        }
