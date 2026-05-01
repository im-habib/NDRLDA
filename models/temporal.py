"""
Temporal modules for capturing vigilance evolution patterns.

The temporal module processes a sequence of encoded feature vectors
across the observation window to produce a state representation s_t
that captures:
    - Current physiological state
    - Historical fatigue trend
    - Vigilance transition dynamics

Two architectures:
    1. LSTMTemporal      — recurrent, efficient, good for sequential patterns
    2. TransformerTemporal — attention-based, captures long-range dependencies

Both: (batch, seq_len, feature_dim) → (batch, output_dim)
"""

import math

import torch
import torch.nn as nn


class LSTMTemporal(nn.Module):
    """
    LSTM-based temporal encoder.

    Processes windowed feature sequences to capture fatigue progression.
    Uses the final hidden state as the temporal state representation.

    Architecture:
        (B, T, D) → LSTM(hidden_size, num_layers) → last hidden → Linear → s_t
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)
        self.output_dim = hidden_size * self.num_directions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) — sequence of feature vectors

        Returns:
            (B, output_dim) — temporal state representation
        """
        # output: (B, T, hidden * directions), (h_n, c_n)
        output, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            # Concatenate final hidden from both directions
            # h_n shape: (num_layers * 2, B, hidden)
            forward_h = h_n[-2]   # last layer, forward
            backward_h = h_n[-1]  # last layer, backward
            hidden = torch.cat([forward_h, backward_h], dim=-1)
        else:
            hidden = h_n[-1]  # (B, hidden)

        return self.layer_norm(hidden)


class TransformerTemporal(nn.Module):
    """
    Transformer-based temporal encoder.

    Uses multi-head self-attention over the time dimension to capture
    long-range temporal dependencies in vigilance progression.

    Architecture:
        (B, T, D) → Linear projection → positional encoding
        → TransformerEncoder → [CLS] token or mean pool → s_t

    Uses a learnable [CLS] token prepended to the sequence for aggregation.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # [CLS] token for sequence-level representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Sinusoidal positional encoding
        max_len = 512
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])  # handle odd d_model
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.output_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) — sequence of feature vectors

        Returns:
            (B, d_model) — temporal state representation
        """
        B, T, _ = x.shape

        # Project to d_model
        x = self.input_proj(x)  # (B, T, d_model)

        # Add positional encoding
        x = x + self.pe[:, :T, :]

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, d_model)

        # Transformer encoding
        encoded = self.transformer(x)  # (B, T+1, d_model)

        # Extract [CLS] token output
        cls_output = encoded[:, 0, :]  # (B, d_model)
        return self.output_norm(cls_output)


def create_temporal(temporal_type: str, input_dim: int, config) -> nn.Module:
    """Factory function to create temporal module from config."""
    if temporal_type == "lstm":
        return LSTMTemporal(
            input_dim,
            config.lstm_hidden_size,
            config.lstm_num_layers,
            config.lstm_dropout,
            config.lstm_bidirectional,
        )
    elif temporal_type == "transformer":
        return TransformerTemporal(
            input_dim,
            config.transformer_d_model,
            config.transformer_nhead,
            config.transformer_num_layers,
            config.transformer_dim_feedforward,
            config.transformer_dropout,
        )
    else:
        raise ValueError(f"Unknown temporal type: {temporal_type}")
