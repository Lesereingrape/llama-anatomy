"""The cache table is arithmetic on shapes, so it is testable exactly.

Nothing here measures accuracy: the point of the module is the part of the GQA
argument that does not need an experiment.
"""

from __future__ import annotations

from dataclasses import asdict, replace

from llamanat.kv import (
    BYTES,
    cache_report,
    head_dim,
    kv_cache_bytes,
    kv_projection_params,
)
from llamanat.model import LLAMA, Config


def test_head_dim_is_d_over_heads():
    assert head_dim(Config(d_model=64, n_head=4)) == 16
    assert head_dim(Config(d_model=4096, n_head=32)) == 128


def test_cache_formula_counts_two_tensors_per_layer_per_token():
    cfg = Config(d_model=16, n_layer=2, n_head=4, n_kv_head=2)
    # 2 (K and V) x layers x kv rows x bytes x positions
    assert kv_cache_bytes(cfg, 10, "fp16") == 2 * 2 * (2 * 4) * 2 * 10
    assert kv_cache_bytes(cfg, 10, "fp32") == 2 * kv_cache_bytes(cfg, 10, "fp16")
    assert kv_cache_bytes(cfg, 10, "int8") == kv_cache_bytes(cfg, 10, "fp16") // 2


def test_every_precision_is_a_byte_count():
    assert BYTES == {"fp32": 4, "fp16": 2, "int8": 1}


def test_cache_scales_linearly_with_length_and_batch():
    one = kv_cache_bytes(LLAMA, 128)
    assert kv_cache_bytes(LLAMA, 256) == 2 * one
    assert kv_cache_bytes(LLAMA, 128, batch=8) == 8 * one


def test_only_kv_heads_move_the_cache():
    """Query heads multiply the attention output, not the cached state."""
    same_kv = replace(LLAMA, n_head=8, d_model=LLAMA.d_model * 2)
    assert head_dim(same_kv) == head_dim(LLAMA)
    assert kv_cache_bytes(same_kv, 64) == kv_cache_bytes(LLAMA, 64)
    assert kv_cache_bytes(replace(LLAMA, n_kv_head=1), 64) == \
        kv_cache_bytes(LLAMA, 64) // LLAMA.n_kv_head


def test_kv_projection_params_track_the_cached_rows():
    assert kv_projection_params(LLAMA) == 2 * LLAMA.n_layer * LLAMA.d_model * \
        LLAMA.n_kv_head * head_dim(LLAMA)
    mqa = replace(LLAMA, n_kv_head=1)
    assert kv_projection_params(mqa) == \
        kv_projection_params(LLAMA) // LLAMA.n_kv_head


def test_report_shares_follow_the_head_grouping():
    report = cache_report(LLAMA, seq_len=2048, precision="fp16")
    rows = report["variants"]
    assert rows["mha"]["share_of_mha"] == 1.0
    assert rows["gqa"]["share_of_mha"] == \
        round(LLAMA.n_kv_head / LLAMA.n_head, 4)
    assert rows["mqa"]["share_of_mha"] == round(1 / LLAMA.n_head, 4)
    for row in rows.values():
        assert row["saved_vs_mha"] == round(1 - row["share_of_mha"], 4)


def test_report_is_the_same_model_at_three_groupings():
    report = cache_report(LLAMA, seq_len=1024)
    rows = report["variants"]
    assert rows["mha"]["cache_bytes"] == \
        kv_cache_bytes(replace(LLAMA, n_kv_head=LLAMA.n_head), 1024)
    assert rows["gqa"]["cache_bytes"] == kv_cache_bytes(LLAMA, 1024)
    assert rows["mha"]["cache_bytes"] == 2 * rows["gqa"]["cache_bytes"]
    assert rows["gqa"]["cache_bytes"] == LLAMA.n_kv_head * rows["mqa"]["cache_bytes"]


def test_kib_matches_bytes():
    rows = cache_report(LLAMA, seq_len=512)["variants"]
    for row in rows.values():
        assert row["cache_kib"] == round(row["cache_bytes"] / 1024, 2)


def test_report_does_not_mutate_the_reference_config():
    before = asdict(LLAMA)
    cache_report(LLAMA, seq_len=4096)
    assert asdict(LLAMA) == before
