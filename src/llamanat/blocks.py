"""The building blocks of a LLaMA decoder layer, written from the paper.

Every choice here is the one Touvron et al. describe (arXiv 2302.13971 for
LLaMA-1, 2305.11276 for LLaMA-2): pre-norm with RMSNorm, RoPE applied to the
query and key vectors, a SwiGLU feed-forward, attention with no biases, and
residual-stream output projections initialised at 1/sqrt(2*n_layer).

The controls (LayerNorm, learned absolute positions, the plain GeLU
feed-forward, full multi-head attention) sit next to the real thing, so an
ablation is one field on a config rather than a second model file.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional import apply_rope, rope_frequencies

NORM_KINDS = ("rms", "ln")
MLP_KINDS = ("swiglu", "gelu")


class RMSNorm(nn.Module):
    """Root-mean-square layer norm: rescale by the RMS, no mean shift, no bias."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


def make_norm(kind: str, dim: int) -> nn.Module:
    if kind == "rms":
        return RMSNorm(dim)
    if kind == "ln":
        return nn.LayerNorm(dim)
    raise ValueError(f"unknown norm {kind!r}, expected one of {NORM_KINDS}")


class CausalSelfAttention(nn.Module):
    """Grouped-query causal attention.

    ``n_kv_head < n_head`` shares each key/value projection across a group of
    query heads. That is the whole of GQA: the query side is unchanged, while
    k/v hold fewer parameters and, more importantly, a proportionally smaller
    KV cache at inference time.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_kv_head: int,
        rope: bool,
        rope_theta: float = 10_000.0,
    ):
        super().__init__()
        if d_model % n_head:
            raise ValueError("d_model must be divisible by n_head")
        if n_head % n_kv_head:
            raise ValueError("n_head must be divisible by n_kv_head")
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.group = n_head // n_kv_head
        self.head_dim = d_model // n_head
        self.rope = rope
        self.q_proj = nn.Linear(d_model, n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_head * self.head_dim, d_model, bias=False)
        if rope:
            self.freqs = rope_frequencies(self.head_dim, rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape

        def split(t: torch.Tensor, heads: int) -> torch.Tensor:
            return t.view(batch, seq, heads, self.head_dim).transpose(1, 2)

        q = split(self.q_proj(x), self.n_head)
        k = split(self.k_proj(x), self.n_kv_head)
        v = split(self.v_proj(x), self.n_kv_head)
        if self.rope:
            positions = torch.arange(seq, device=x.device)
            freqs = self.freqs.to(x.device)
            q = apply_rope(q, freqs, positions)
            k = apply_rope(k, freqs, positions)
        if self.group > 1:
            k = k.repeat_interleave(self.group, dim=1)
            v = v.repeat_interleave(self.group, dim=1)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        causal = torch.ones(seq, seq, dtype=torch.bool, device=x.device).triu(1)
        att = scores.masked_fill(causal, float("-inf")).softmax(dim=-1)
        out = (att @ v).transpose(1, 2).reshape(batch, seq, self.n_head * self.head_dim)
        return self.o_proj(out)


def swiglu_hidden(d_model: int, multiple: int = 64) -> int:
    """LLaMA's feed-forward width: 8/3 * d_model, rounded up to a clean multiple.

    The paper rounds to a multiple of 256, which this reproduces at its own
    scale (``swiglu_hidden(4096, 256) == 11008``, the 7B width). At d_model 64 a
    256-grain would jump to 256 -- four times the model rather than 2.67 -- so
    the tiny reference model keeps the ratio and rounds at 64.
    """
    return int(math.ceil(8 * d_model / 3 / multiple) * multiple)


def gelu_hidden(swiglu_ff: int) -> int:
    """Width that puts the same number of weights into a two-matrix MLP.

    SwiGLU spends three d x ff matrices and a plain MLP two d x h ones, so
    h = 1.5 * ff is the equal-parameter translation. Without it the GeLU
    control would be a third smaller and the ablation would be measuring size.
    """
    return round(1.5 * swiglu_ff)


class SwiGLUMLP(nn.Module):
    """down( silu(gate(x)) * up(x) ): a gated feed-forward with three matrices."""

    def __init__(self, d_model: int, ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, ff, bias=False)
        self.up = nn.Linear(d_model, ff, bias=False)
        self.down = nn.Linear(ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))

    @property
    def residual_matrix(self) -> nn.Module:
        return self.down


class GeLUMLP(nn.Module):
    """The control feed-forward: two matrices, tanh-approximation gelu."""

    def __init__(self, d_model: int, ff: int):
        super().__init__()
        hidden = gelu_hidden(ff)
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x), approximate="tanh"))

    @property
    def residual_matrix(self) -> nn.Module:
        return self.w2


def make_mlp(kind: str, d_model: int, ff: int) -> nn.Module:
    if kind == "swiglu":
        return SwiGLUMLP(d_model, ff)
    if kind == "gelu":
        return GeLUMLP(d_model, ff)
    raise ValueError(f"unknown mlp {kind!r}, expected one of {MLP_KINDS}")


class DecoderBlock(nn.Module):
    """Pre-norm residual block: x += attn(norm1(x)), then x += mlp(norm2(x))."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_kv_head: int,
        ff: int,
        norm: str,
        rope: bool,
        mlp: str,
        rope_theta: float = 10_000.0,
    ):
        super().__init__()
        self.attn_norm = make_norm(norm, d_model)
        self.attn = CausalSelfAttention(d_model, n_head, n_kv_head, rope,
                                        rope_theta)
        self.mlp_norm = make_norm(norm, d_model)
        self.mlp = make_mlp(mlp, d_model, ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.mlp(self.mlp_norm(x))
