"""
Staleness-Aware Visual Encoder with Temporal Offset Positional Encoding (TOPE).
Filename: visual_encoder.py

Fuses a rolling buffer of T=8 depth feature vectors into a single 64-dim
context vector c_v, explicitly encoding the staleness of each token so the
Transformer can discount outdated visual observations.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING

from depth_stub import DepthCNNEmbedder

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =============================================================================
# Temporal Offset Positional Encoding (TOPE)
# =============================================================================

class TOPE(nn.Module):
    """
    PE(i) = sinusoidal_pe(i) + learned_offset_embed(d)

    where i is the buffer index (0 = oldest, T-1 = newest) and d is the
    integer visual delay assigned to this environment episode.
    """

    MAX_DELAY: int = 8   # maximum delay in steps

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        self.d_model = d_model

        # Fixed sinusoidal PE for buffer index
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("_sinusoidal_pe", pe)    # (T, d_model)

        # Learned offset embedding for delay value d in {0, ..., MAX_DELAY}
        self.offset_embed = nn.Embedding(self.MAX_DELAY + 1, d_model)

    def forward(
        self,
        x: torch.Tensor,      # (B, T, d_model)
        delay: torch.Tensor,  # (B,) integer delay per environment
    ) -> torch.Tensor:
        """Returns x + sinusoidal_pe + learned_offset_embed, shape (B, T, d_model)."""
        T = x.shape[1]
        sin_pe = self._sinusoidal_pe[:T].unsqueeze(0)            # (1, T, d_model)

        # Clamp delay to valid range
        d_clamped = delay.clamp(0, self.MAX_DELAY).long()
        off_emb   = self.offset_embed(d_clamped).unsqueeze(1)   # (B, 1, d_model)

        return x + sin_pe + off_emb


# =============================================================================
# Staleness-Aware Visual Encoder
# =============================================================================

class StalenessAwareVisualEncoder(nn.Module):
    """
    Transformer-based encoder that fuses T=8 depth feature tokens into
    a single 64-dim context vector c_v.

    Each token is the output of DepthCNNEmbedder applied to one buffered
    depth image frame.  The delay d is incorporated via TOPE so the model
    can learn to down-weight stale frames.
    """

    T: int = 8          # rolling buffer length
    D_MODEL: int = 64   # must match DepthCNNEmbedder.OUTPUT_DIM

    def __init__(self) -> None:
        super().__init__()

        self.cnn = DepthCNNEmbedder(in_channels=4)

        self.tope = TOPE(d_model=self.D_MODEL, max_len=self.T)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.D_MODEL,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Learnable [CLS] token to pool the sequence into one vector
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.D_MODEL) * 0.02)

    def forward(
        self,
        img_buffer: torch.Tensor,   # (B, T, 4, 64, 64) rolled depth frames
        delay: torch.Tensor,        # (B,) integer delay
    ) -> torch.Tensor:
        """
        Returns:
            c_v: (B, 64) context vector.
        """
        B, T, C, H, W = img_buffer.shape

        # Extract CNN features for each frame independently
        imgs_flat  = img_buffer.reshape(B * T, C, H, W)
        feats_flat = self.cnn(imgs_flat)                      # (B*T, 64)
        feats      = feats_flat.reshape(B, T, self.D_MODEL)   # (B, T, 64)

        # Apply TOPE
        feats = self.tope(feats, delay)                       # (B, T, 64)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)               # (B, 1, 64)
        seq = torch.cat([cls, feats], dim=1)                  # (B, T+1, 64)

        out = self.transformer(seq)                           # (B, T+1, 64)
        c_v = out[:, 0, :]                                    # CLS → (B, 64)
        return c_v

    # -------------------------------------------------------------------------
    # Attention weight extraction (used in unit test)
    # -------------------------------------------------------------------------

    def get_attention_weights(
        self,
        img_buffer: torch.Tensor,
        delay: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns average self-attention weights of shape (B, T+1, T+1)
        from the first Transformer layer.

        Used only for analysis / unit testing; not called during training.
        """
        B, T, C, H, W = img_buffer.shape
        imgs_flat  = img_buffer.reshape(B * T, C, H, W)
        feats_flat = self.cnn(imgs_flat)
        feats      = feats_flat.reshape(B, T, self.D_MODEL)
        feats      = self.tope(feats, delay)

        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, feats], dim=1)         # (B, T+1, 64)

        # Access the first layer's self-attention directly
        layer = self.transformer.layers[0]
        attn_output, attn_weights = layer.self_attn(
            seq, seq, seq, average_attn_weights=True
        )
        return attn_weights   # (B, T+1, T+1)


# =============================================================================
# Environment-facing ring-buffer manager
# =============================================================================

class VisualBufferManager:
    """
    Maintains the per-environment rolling image feature buffer and
    manages the delayed read-out.

    Lifecycle
    ---------
    - Instantiated in Go2RetrievalEnv.__init__().
    - push() called every policy step with the latest depth image batch.
    - get_delayed_buffer() returns the (B, T, 4, 64, 64) tensor for the encoder.
    - reset() clears the buffer and re-samples the delay for given env_ids.
    """

    T: int = 8
    D_MIN: int = 3
    D_MAX: int = 8

    def __init__(
        self,
        num_envs: int,
        img_channels: int,
        img_h: int,
        img_w: int,
        device: str | torch.device,
    ) -> None:
        self.num_envs = num_envs
        self.device   = device
        self.img_shape = (img_channels, img_h, img_w)

        # Ring buffer: (N, T, C, H, W)
        self._buffer = torch.zeros(
            num_envs, self.T, *self.img_shape, device=device
        )
        # Per-environment write index
        self._write_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        # Per-environment delay (integer steps)
        self.delay = torch.randint(self.D_MIN, self.D_MAX + 1, (num_envs,), device=device)

    def push(self, imgs: torch.Tensor) -> None:
        """
        Push a new batch of depth images into the ring buffer.

        Args:
            imgs: (N, C, H, W) current depth images.
        """
        for n in range(self.num_envs):
            self._buffer[n, self._write_idx[n]] = imgs[n]
        self._write_idx = (self._write_idx + 1) % self.T

    def get_delayed_buffer(self) -> torch.Tensor:
        """
        Return the (N, T, C, H, W) buffer ordered from oldest-delayed to newest,
        applying the per-environment delay by reading from
        (write_idx - delay) % T as the most-recent valid frame.
        """
        N = self.num_envs
        T = self.T

        ordered = torch.zeros(N, T, *self.img_shape, device=self.device)
        for n in range(N):
            d   = int(self.delay[n].item())
            # The "most-recent valid" frame is d steps behind the write pointer
            latest_valid = (self._write_idx[n] - d) % T
            for t in range(T):
                src_idx = (latest_valid - (T - 1 - t)) % T
                ordered[n, t] = self._buffer[n, src_idx]

        return ordered

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is not None:
            self._buffer[env_ids] = 0.0
            self._write_idx[env_ids] = 0
            self.delay[env_ids] = torch.randint(
                self.D_MIN, self.D_MAX + 1, (len(env_ids),), device=self.device
            )
        else:
            self._buffer.zero_()
            self._write_idx.zero_()
            self.delay = torch.randint(
                self.D_MIN, self.D_MAX + 1, (self.num_envs,), device=self.device
            )