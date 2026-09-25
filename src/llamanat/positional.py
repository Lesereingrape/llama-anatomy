"""The two ways a decoder gets a sense of order, side by side.

RoPE (Su et al., arXiv 2104.09864; adopted by LLaMA-1) never adds anything to
the embedding: it rotates the query and key vectors by an angle proportional
to the token's position, so only *relative* offsets survive the dot product
and the same weights apply at any index. Learned absolute positions (Vaswani's
sinusoidal idea turned into a trainable table, as in GPT-2) instead look up a
row per index -- which means an index that was never sampled during training
has an embedding that was never updated. That difference is the whole of the
train-short / test-long experiment in this repository.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def rope_frequencies(head_dim: int, theta: float = 10_000.0) -> torch.Tensor:
    """1 / theta^(2i/d) for i in [0, head_dim/2) -- LLaMA-1 uses theta=1e4."""
    pairs = torch.arange(0, head_dim, 2).float()
    return theta ** (-pairs / head_dim)


def longest_wavelength(head_dim: int, theta: float) -> float:
    """How far apart two tokens can be before the slowest channel pair wraps.

    Channel pair ``i`` rotates once every ``2*pi*theta^(2i/head_dim)``, so the
    slowest pair has the longest reach: past its period that pair reads two
    different offsets as the same angle, even while the faster pairs still tell
    them apart. The fastest pair has period ``2*pi`` whatever theta is -- theta
    buys range, never precision. Arithmetic on the geometry, not a measurement.
    """
    return 2 * math.pi * theta ** (1.0 - 2.0 / head_dim)


def rope_angles(
    freqs: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    angles = positions.float().outer(freqs)          # (T, head_dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, freqs: torch.Tensor,
               positions: torch.Tensor) -> torch.Tensor:
    """Rotate a (B, H, T, head_dim) tensor by its position.

    The two halves of each channel *pair* are rotated together, which is the
    LLaMA-1 interleaving; the dot product of a rotated query and a rotated key
    then depends only on how far apart the two positions are.
    """
    cos, sin = rope_angles(freqs, positions)
    cos, sin = cos[None, None], sin[None, None]
    first, second = x[..., 0::2], x[..., 1::2]
    return torch.stack(
        [first * cos - second * sin, first * sin + second * cos], dim=-1
    ).flatten(-2)


class AbsolutePositionEmbedding(nn.Module):
    """A trainable position table, sized for the longest sequence we will *run*.

    Sizing it past the training length is deliberate: the extra rows exist so
    the control can be evaluated out of range at all, and they stay at their
    random initialisation because no training batch ever reaches them.
    """

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.table = nn.Embedding(max_len, d_model)
        self.max_len = max_len

    def forward(self, seq_len: int) -> torch.Tensor:
        if seq_len > self.max_len:
            raise ValueError(f"sequence of {seq_len} exceeds {self.max_len}")
        return self.table(torch.arange(seq_len, device=self.table.weight.device))
