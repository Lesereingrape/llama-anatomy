"""A ~100k-parameter LLaMA decoder, assembled from a config instead of a fork.

``Config`` carries the four choices that define the LLaMA block -- RMSNorm,
RoPE, SwiGLU, grouped-query attention -- and ``VARIANTS`` is the ablation:
each entry is the base config with exactly one field replaced, so any two rows
of the results table differ in one mechanism and nothing else.

Everything shares one initialisation (normal, std 0.02, with the residual
output projections scaled by 1/sqrt(2*n_layer)), so a variant that starts
worse off is not being punished for its seed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
import torch.nn as nn

from .blocks import DecoderBlock, make_norm, swiglu_hidden
from .positional import AbsolutePositionEmbedding
from .tasks import VOCAB

INIT_STD = 0.02


@dataclass(frozen=True)
class Config:
    d_model: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_kv_head: int = 2
    ff: int = 0
    norm: str = "rms"
    rope: bool = True
    #: the rotation base. Unlike every other field here it costs nothing: it
    #: adds no parameter and changes no shape, only which relative offsets the
    #: rotation can tell apart. LLaMA-1 uses 1e4; the sweep in ``study.py`` is
    #: the one place this repository measures a parameter-free axis.
    rope_theta: float = 10_000.0
    mlp: str = "swiglu"
    max_len: int = 48
    tie_embeddings: bool = True
    vocab: int = VOCAB

    def __post_init__(self) -> None:
        if not self.ff:
            object.__setattr__(self, "ff", swiglu_hidden(self.d_model))


LLAMA = Config()

#: name -> the single field that is *not* LLaMA's.
VARIANTS: dict[str, dict] = {
    "llama": {},
    "layernorm": {"norm": "ln"},
    "abs_pos": {"rope": False},
    "gelu_mlp": {"mlp": "gelu"},
    "mha": {"n_kv_head": 4},
    "mqa": {"n_kv_head": 1},
}


def variant_config(name: str, **over: int | str | bool) -> Config:
    if name not in VARIANTS:
        raise KeyError(f"unknown variant {name!r}, expected one of {VARIANTS}")
    return replace(LLAMA, **VARIANTS[name], **over)


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def output_proj_scale(n_layer: int) -> float:
    """GPT-2/LLaMA's residual scaling: each block feeds the stream at 1/2*n_layer."""
    return 1.0 / math.sqrt(2 * n_layer)


class TinyLLaMA(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos_emb = (
            None if cfg.rope else AbsolutePositionEmbedding(cfg.max_len, cfg.d_model)
        )
        self.blocks = nn.ModuleList(
            DecoderBlock(
                d_model=cfg.d_model,
                n_head=cfg.n_head,
                n_kv_head=cfg.n_kv_head,
                ff=cfg.ff,
                norm=cfg.norm,
                rope=cfg.rope,
                mlp=cfg.mlp,
                rope_theta=cfg.rope_theta,
            )
            for _ in range(cfg.n_layer)
        )
        self.final_norm = make_norm(cfg.norm, cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        scale = output_proj_scale(cfg.n_layer)
        for block in self.blocks:
            with torch.no_grad():
                block.attn.o_proj.weight.mul_(scale)
                block.mlp.residual_matrix.weight.mul_(scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(ids)
        if self.pos_emb is not None:
            x = x + self.pos_emb(ids.size(1))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))

    def logits_at(self, ids: torch.Tensor, positions: list[int]) -> torch.Tensor:
        """Rows of the vocabulary distribution at the requested positions."""
        full = self.forward(ids)
        index = torch.as_tensor(positions, device=ids.device, dtype=torch.long)
        return full.index_select(1, index)

    def greedy_at(self, ids: torch.Tensor, position: int) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(ids)[:, position].argmax(dim=-1)
