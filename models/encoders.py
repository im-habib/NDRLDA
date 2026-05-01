"""
Shared feature encoders for the actor-critic architecture.

Three encoder variants for experimental comparison:
    1. MLPEncoder   — simple, fast, baseline
    2. CNNEncoder   — captures local feature correlations via 1D convolution
    3. AttentionEncoder — self-attention over feature groups, best for multimodal fusion

All encoders: (batch, input_dim) → (batch, output_dim)
"""

import math

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """
    Multi-layer perceptron encoder.

    Architecture:
        input → [Linear → LayerNorm → ReLU → Dropout] × L → output

    Simple baseline. Works well when feature dimension is moderate.
    """

    def __init__(self, input_dim: int, hidden_dims: tuple = (256, 128), dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        self.net = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNEncoder(nn.Module):
    """
    1D CNN encoder over feature dimension.

    Treats the feature vector as a 1D signal and applies conv layers
    to capture local correlations between adjacent features
    (e.g., neighboring EEG channels within a frequency band).

    Architecture:
        (B, D) → reshape(B, 1, D) → [Conv1d → BN → ReLU → Dropout] × L → AdaptivePool → flatten
    """

    def __init__(
        self,
        input_dim: int,
        channels: tuple = (64, 128, 64),
        kernel_sizes: tuple = (3, 3, 3),
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_ch = 1
        for out_ch, ks in zip(channels, kernel_sizes):
            padding = ks // 2
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding=padding),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) → (B, 1, D)
        x = x.unsqueeze(1)
        x = self.conv(x)      # (B, C, D')
        x = self.pool(x)      # (B, C, 1)
        return x.squeeze(-1)   # (B, C)


class AttentionEncoder(nn.Module):
    """
    Self-attention encoder for multimodal feature fusion.

    Splits the input feature vector into groups (e.g., EEG channels × bands, EOG),
    treats each group as a token, and applies multi-head self-attention.

    This lets the model learn cross-modal and cross-channel relationships.

    Architecture:
        (B, D) → split into N tokens of dim d → positional encoding
        → TransformerEncoder → mean pool → Linear → output
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        d_model: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Project each feature into d_model space
        # We treat each feature as a "token" and project it
        self.input_proj = nn.Linear(1, d_model)

        # Learnable positional encoding
        # We'll cap at a reasonable max length
        max_tokens = max(input_dim, 512)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_tokens, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, d_model)
        self.output_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        # Each feature becomes a token: (B, D) → (B, D, 1) → (B, D, d_model)
        tokens = self.input_proj(x.unsqueeze(-1))
        tokens = tokens + self.pos_encoding[:, :D, :]

        # Self-attention
        encoded = self.transformer(tokens)  # (B, D, d_model)

        # Mean pool over tokens
        pooled = encoded.mean(dim=1)  # (B, d_model)
        return self.output_proj(pooled)


def create_encoder(encoder_type: str, input_dim: int, config) -> nn.Module:
    """Factory function to create encoder from config."""
    if encoder_type == "mlp":
        return MLPEncoder(input_dim, config.mlp_hidden_dims, config.mlp_dropout)
    elif encoder_type == "cnn":
        return CNNEncoder(input_dim, config.cnn_channels, config.cnn_kernel_sizes, config.cnn_dropout)
    elif encoder_type == "attention":
        return AttentionEncoder(
            input_dim, config.attention_heads, config.attention_dim,
            config.attention_layers, config.attention_dropout,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
