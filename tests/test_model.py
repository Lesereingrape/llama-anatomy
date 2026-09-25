import math

import torch

from llamanat.blocks import RMSNorm
from llamanat.model import (
    INIT_STD,
    LLAMA,
    TinyLLaMA,
    n_params,
    output_proj_scale,
    variant_config,
)
from llamanat.tasks import VOCAB


def test_llama_block_param_budget():
    model = TinyLLaMA(LLAMA)
    params = n_params(model)
    assert 90_000 < params < 130_000
    assert params == LLAMA_PARAMS


LLAMA_PARAMS = n_params(TinyLLaMA(LLAMA))


def test_every_variant_builds_and_is_registered_once():
    from llamanat.model import VARIANTS

    assert list(VARIANTS) == [
        "llama", "layernorm", "abs_pos", "gelu_mlp", "mha", "mqa"
    ]
    for name in VARIANTS:
        out = TinyLLaMA(variant_config(name))(torch.ones(2, 6, dtype=torch.long))
        assert out.shape == (2, 6, variant_config(name).vocab)


def test_gelu_control_has_identical_parameter_count():
    assert n_params(TinyLLaMA(variant_config("gelu_mlp"))) == LLAMA_PARAMS


def test_rmsnorm_control_is_larger_only_by_its_bias_terms():
    norms = 2 * LLAMA.n_layer + 1          # two per block plus the final one
    expect = LLAMA_PARAMS + norms * LLAMA.d_model
    assert n_params(TinyLLaMA(variant_config("layernorm"))) == expect


def test_absolute_positions_cost_a_table_that_rope_does_not():
    cfg = variant_config("abs_pos")
    extra = cfg.max_len * cfg.d_model
    assert n_params(TinyLLaMA(cfg)) == LLAMA_PARAMS + extra
    assert TinyLLaMA(variant_config("llama")).pos_emb is None


def test_kv_sharing_only_touches_the_key_and_value_projections():
    mha = TinyLLaMA(variant_config("mha"))
    mqa = TinyLLaMA(variant_config("mqa"))
    gqa = TinyLLaMA(variant_config("llama"))
    assert n_params(mha) > n_params(gqa) > n_params(mqa)
    heads = LLAMA.n_head
    per_layer_delta = 2 * LLAMA.d_model * (heads - 1) * (LLAMA.d_model // heads)
    assert n_params(mha) - n_params(mqa) == per_layer_delta * LLAMA.n_layer


def test_the_rotation_base_reaches_every_block_and_costs_nothing():
    """The sweep table is only meaningful if this field is actually plumbed.

    A ``rope_theta`` that never reached ``CausalSelfAttention`` would leave four
    identical rows in the README and no test would notice by itself, so the
    frequencies are checked directly -- and the parameter count is checked to
    stay put, which is the claim the section makes about the axis.
    """
    other = variant_config("llama", rope_theta=100.0)
    base = TinyLLaMA(variant_config("llama"))
    swept = TinyLLaMA(other)
    slowest = -2 * 7 / (LLAMA.d_model // LLAMA.n_head)
    assert not math.isclose(float(base.blocks[0].attn.freqs[-1]),
                            float(swept.blocks[0].attn.freqs[-1]), rel_tol=1e-3)
    for block in swept.blocks:
        assert math.isclose(float(block.attn.freqs[-1]), 100.0 ** slowest,
                            rel_tol=1e-6)
    assert n_params(swept) == LLAMA_PARAMS


def test_embeddings_are_tied_by_default():
    model = TinyLLaMA(LLAMA)
    assert model.lm_head.weight is model.tok_emb.weight


def test_untied_variant_keeps_two_matrices():
    model = TinyLLaMA(variant_config("llama", tie_embeddings=False))
    assert model.lm_head.weight is not model.tok_emb.weight
    assert n_params(model) == LLAMA_PARAMS + LLAMA.vocab * LLAMA.d_model


def test_all_projections_initialise_at_the_standard_scale():
    torch.manual_seed(0)
    model = TinyLLaMA(variant_config("llama"))
    plain = model.blocks[0].attn.q_proj.weight.std().item()
    assert math.isclose(plain, INIT_STD, rel_tol=0.2)


def test_residual_output_projections_are_scaled_down():
    model = TinyLLaMA(variant_config("llama"))
    scale = output_proj_scale(LLAMA.n_layer)
    assert math.isclose(scale, 1 / math.sqrt(4), rel_tol=1e-9)
    block = model.blocks[0]
    attn_std = block.attn.o_proj.weight.std().item()
    mlp_std = block.mlp.down.weight.std().item()
    q_std = block.attn.q_proj.weight.std().item()
    assert attn_std < 0.7 * q_std
    assert mlp_std < 0.7 * q_std


def test_no_bias_parameters_exist_in_the_attention_or_mlp():
    model = TinyLLaMA(LLAMA)
    biases = [name for name, p in model.named_parameters() if p.ndim == 1
              and "norm" not in name]
    assert biases == []


def test_forward_returns_one_distribution_per_position():
    ids = torch.randint(0, VOCAB, (3, 9))
    out = TinyLLaMA(LLAMA)(ids)
    assert out.shape == (3, 9, VOCAB)


def test_logits_at_selects_the_requested_rows():
    ids = torch.randint(0, VOCAB, (2, 9))
    model = TinyLLaMA(LLAMA)
    picked = model.logits_at(ids, [1, 4])
    full = model(ids)
    assert torch.allclose(picked, torch.stack([full[:, 1], full[:, 4]], dim=1))


def test_final_norm_is_the_same_kind_as_the_block_norm():
    model = TinyLLaMA(variant_config("layernorm"))
    assert not isinstance(model.final_norm, RMSNorm)
    assert isinstance(TinyLLaMA(LLAMA).final_norm, RMSNorm)


def test_feed_forward_width_comes_from_the_config_rule():
    assert variant_config("llama").ff == swiglu_expectation()


def swiglu_expectation() -> int:
    return math.ceil(8 * LLAMA.d_model / 3 / 64) * 64
