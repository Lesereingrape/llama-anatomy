"""Exact arithmetic about what grouped-query attention buys.

Nothing in this module is measured. It is arithmetic on shapes, and saying so
matters: the interesting claim about GQA is not "it is better", it is "it
holds k/n_kv times less state per generated token", and that follows from the
definition rather than from an experiment. The experiments in this repository
are about how much accuracy that saving costs, which is a separate question.
"""

from __future__ import annotations

from dataclasses import replace

from .model import LLAMA, Config

BYTES = {"fp32": 4, "fp16": 2, "int8": 1}


def head_dim(cfg: Config) -> int:
    return cfg.d_model // cfg.n_head


def kv_cache_bytes(cfg: Config, seq_len: int, precision: str = "fp16",
                   batch: int = 1) -> int:
    """Size of the key/value cache for one sequence: 2 tensors per layer."""
    rows = cfg.n_kv_head * head_dim(cfg)
    return 2 * cfg.n_layer * rows * BYTES[precision] * seq_len * batch


def kv_projection_params(cfg: Config) -> int:
    """Weights that exist only to produce cached state (k and v projections)."""
    return 2 * cfg.n_layer * cfg.d_model * cfg.n_kv_head * head_dim(cfg)


def cache_report(cfg: Config = LLAMA, seq_len: int = 2048,
                 precision: str = "fp16") -> dict:
    """Compare the attention variants at three KV-grouping settings."""
    out: dict = {"seq_len": seq_len, "precision": precision, "variants": {}}
    full = kv_cache_bytes(_with_kv_heads(cfg, cfg.n_head), seq_len, precision)
    for name, n_kv in (("mha", cfg.n_head), ("gqa", cfg.n_kv_head), ("mqa", 1)):
        variant = _with_kv_heads(cfg, n_kv)
        raw = kv_cache_bytes(variant, seq_len, precision)
        out["variants"][name] = {
            "n_kv_head": n_kv,
            "cache_bytes": raw,
            "cache_kib": round(raw / 1024, 2),
            "kv_projection_params": kv_projection_params(variant),
            "share_of_mha": round(raw / full, 4),
            "saved_vs_mha": round(1 - raw / full, 4),
        }
    return out


def _with_kv_heads(cfg: Config, n_kv_head: int) -> Config:
    return replace(cfg, n_kv_head=n_kv_head)
