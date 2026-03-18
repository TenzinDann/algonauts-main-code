"""Model architectures for multimodal brain encoding.

Three encoder families providing architectural diversity for ensemble learning:
  - TRIBEEncoder: Transformer-based multimodal fusion (inspired by TRIBE [1])
  - MedARCEncoder: Deep linear feedforward branch (inspired by MedARC [6])
  - WideLinearEncoder: Wide nonlinear projection branch
"""

import random

import torch
import torch.nn as nn


class ModalityDropout(nn.Module):
    """Drop entire modalities during training to improve robustness.

    Prevents over-reliance on any single modality, following the technique
    introduced by TRIBE [1]. At least one modality is always kept active.

    Parameters
    ----------
    modality_dims : list of (int, int)
        List of (start, end) index pairs for each modality in the input.
    p : float
        Probability of dropping each modality.
    """

    def __init__(self, modality_dims, p=0.2):
        super().__init__()
        self.modality_dims = modality_dims
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = torch.ones_like(x)
        drop_flags = [random.random() < self.p for _ in self.modality_dims]
        # Never drop all modalities
        if all(drop_flags):
            keep_idx = random.randint(0, len(self.modality_dims) - 1)
            drop_flags[keep_idx] = False
        for i, (start, end) in enumerate(self.modality_dims):
            if drop_flags[i]:
                mask[:, start:end] = 0
        return x * mask


class TRIBEEncoder(nn.Module):
    """Transformer-based multimodal encoder inspired by TRIBE [1].

    Decomposes the concatenated stimulus window into per-modality token
    sequences, projects each to a shared dimension, adds modality type
    embeddings and positional embeddings, then applies bidirectional
    Transformer self-attention across all tokens.

    Parameters
    ----------
    input_dim : int
        Total input feature dimension (vis_base*window + aud_base*window + lang_base).
    output_dim : int
        Number of brain parcels to predict (default: 1000).
    d_model : int
        Transformer hidden dimension (default: 256).
    nhead : int
        Number of attention heads (default: 8).
    num_layers : int
        Number of Transformer encoder layers (default: 3).
    dropout : float
        Dropout rate (default: 0.3).
    modality_dims : list of (int, int) or None
        Modality index ranges for ModalityDropout.
    stimulus_window : int
        Number of temporal frames per modality (default: 5).
    vis_base, aud_base, lang_base : int
        Per-TR feature dimensions for visual, audio, language modalities.
    """

    def __init__(self, input_dim, output_dim=1000, d_model=256,
                 nhead=8, num_layers=3, dropout=0.3,
                 modality_dims=None, stimulus_window=5,
                 vis_base=250, aud_base=20, lang_base=250):
        super().__init__()
        self.modality_dims = modality_dims
        self.stimulus_window = stimulus_window
        self.vis_base = vis_base
        self.aud_base = aud_base
        self.lang_base = lang_base
        self.d_model = d_model

        # Per-modality projections to shared d_model
        self.vis_proj = nn.Linear(vis_base, d_model)
        self.aud_proj = nn.Linear(aud_base, d_model)
        self.lang_proj = nn.Linear(lang_base, d_model)

        # Modality type embeddings (0=visual, 1=audio, 2=language)
        self.modality_embed = nn.Embedding(3, d_model)

        # Learnable positional embeddings
        max_seq = stimulus_window * 2 + 1
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq, d_model) * 0.02)

        # Modality dropout
        self.modality_dropout = ModalityDropout(
            modality_dims or [(0, input_dim)], p=0.2)

        # Bidirectional Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        self.readout = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)
        )

    def forward(self, x):
        B = x.size(0)
        x = self.modality_dropout(x)

        # Split concatenated input into modality segments
        vis_end = self.vis_base * self.stimulus_window
        aud_end = vis_end + self.aud_base * self.stimulus_window

        vis_flat = x[:, :vis_end]
        aud_flat = x[:, vis_end:aud_end]
        lang_flat = x[:, aud_end:]

        # Reshape windowed features into token sequences
        vis_seq = vis_flat.reshape(B, self.stimulus_window, self.vis_base)
        aud_seq = aud_flat.reshape(B, self.stimulus_window, self.aud_base)
        lang_seq = lang_flat.unsqueeze(1)

        # Project to shared dimension
        vis_tokens = self.vis_proj(vis_seq)
        aud_tokens = self.aud_proj(aud_seq)
        lang_tokens = self.lang_proj(lang_seq)

        # Add modality type embeddings
        dev = x.device
        vis_tokens = vis_tokens + self.modality_embed(
            torch.zeros(B, self.stimulus_window, dtype=torch.long, device=dev))
        aud_tokens = aud_tokens + self.modality_embed(
            torch.ones(B, self.stimulus_window, dtype=torch.long, device=dev))
        lang_tokens = lang_tokens + self.modality_embed(
            torch.full((B, 1), 2, dtype=torch.long, device=dev))

        # Concatenate all tokens -> (B, 2*window+1, d_model)
        tokens = torch.cat([vis_tokens, aud_tokens, lang_tokens], dim=1)
        seq_len = tokens.size(1)
        tokens = tokens + self.pos_embed[:, :seq_len, :]

        # Transformer self-attention
        tokens = self.transformer(tokens)

        # Mean pooling -> readout
        pooled = tokens.mean(dim=1)
        return self.readout(pooled)


class MedARCEncoder(nn.Module):
    """Deep linear feedforward encoder (inspired by MedARC [6]).

    A lightweight stacked linear architecture with batch normalization and
    dropout. Unlike the actual MedARC model (which uses 1D temporal
    convolutions), this branch is purely feedforward for maximum architectural
    diversity within the ensemble.

    Parameters
    ----------
    input_dim : int
        Total input feature dimension.
    output_dim : int
        Number of brain parcels (default: 1000).
    hidden_dim : int
        Hidden layer width (default: 512).
    dropout : float
        Dropout rate (default: 0.3).
    modality_dims : list of (int, int) or None
        Modality index ranges for ModalityDropout.
    """

    def __init__(self, input_dim, output_dim=1000, hidden_dim=512,
                 dropout=0.3, modality_dims=None):
        super().__init__()
        self.modality_dropout = ModalityDropout(
            modality_dims or [(0, input_dim)], p=0.2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = self.modality_dropout(x)
        return self.net(x)


class WideLinearEncoder(nn.Module):
    """Wide nonlinear projection encoder for ensemble diversity.

    Uses wider hidden dimensions with GELU activation to provide a different
    inductive bias from the Transformer and Deep Linear branches.

    Parameters
    ----------
    input_dim : int
        Total input feature dimension.
    output_dim : int
        Number of brain parcels (default: 1000).
    hidden_dim : int
        First hidden layer width (default: 2048).
    dropout : float
        Dropout rate (default: 0.4).
    modality_dims : list of (int, int) or None
        Modality index ranges for ModalityDropout.
    """

    def __init__(self, input_dim, output_dim=1000, hidden_dim=2048,
                 dropout=0.4, modality_dims=None):
        super().__init__()
        self.modality_dropout = ModalityDropout(
            modality_dims or [(0, input_dim)], p=0.2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        x = self.modality_dropout(x)
        return self.net(x)
