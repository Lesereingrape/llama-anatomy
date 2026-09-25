import math

import torch
import torch.nn as nn

from llamanat.blocks import (
    CausalSelfAttention,
    DecoderBlock,
    GeLUMLP,
    RMSNorm,
    SwiGLUMLP,
    gelu_hidden,
    make_mlp,
    make_norm,
    swiglu_hidden,
)
from llamanat.model import LLAMA, n_params


def test_rmsnorm_matches_its_definition():
    norm = RMSNorm(4)
    x = torch.tensor([[3.0, 4.0, 0.0, 0.0]])
    # RMS = sqrt((9 + 16 + 0 + 0) / 4) = 2.5, and the weight starts at one
    expect = x / math.sqrt(6.25 + 1e-6)
    assert torch.allclose(norm(x), expect, atol=1e-5)


def test_rmsnorm_has_no_bias_and_no_mean_shift():
    x = torch.full((1, 8), 5.0)
    out = RMSNorm(8)(x)
    assert torch.allclose(out, torch.ones(1, 8) * 5.0 / math.sqrt(25.0 + 1e-6),
                          atol=1e-4)
    assert not any(isinstance(m, nn.LayerNorm) for m in RMSNorm(8).modules())


def test_layer_norm_control_centres_and_biases():
    ln = make_norm("ln", 6)
    x = torch.randn(2, 6) + 7.0
    out = ln(x)
    assert abs(float(out.mean())) < 1e-5
    assert ln.bias is not None and float(ln.bias.abs().sum()) == 0.0


def test_unknown_norm_and_mlp_are_rejected():
    for bad in (make_norm, lambda k, d: make_mlp(k, d, 8)):
        try:
            bad("nope", 4)
        except ValueError as exc:
            assert "unknown" in str(exc)
        else:
            raise AssertionError("expected a ValueError")


def test_swiglu_width_is_eight_thirds_of_d_rounded_up_to_a_multiple():
    assert swiglu_hidden(64) == 192
    assert swiglu_hidden(128) == 384
    # the paper's geometry, from the same one-line rule with its own rounding
    assert swiglu_hidden(4096, multiple=256) == 11_008     # LLaMA-7B
    assert swiglu_hidden(5120, multiple=256) == 13_824     # LLaMA-13B
    assert swiglu_hidden(8192, multiple=256) == 22_016     # LLaMA-65B


def test_gelu_control_is_the_equal_parameter_translation():
    swiglu = SwiGLUMLP(64, 192)
    gelu = GeLUMLP(64, 192)
    assert n_params(swiglu) == n_params(gelu)
    assert gelu_hidden(192) == 288


def test_swiglu_actually_gates():
    mlp = SwiGLUMLP(4, 8)
    x = torch.randn(1, 4)
    with torch.no_grad():
        mlp.up.weight.copy_(torch.zeros_like(mlp.up.weight))
        assert float(mlp(x).abs().sum()) == 0.0


def test_attention_is_causal():
    attn = CausalSelfAttention(16, 4, 4, rope=True)
    a = torch.randn(1, 5, 16)
    b = a.clone()
    b[:, -1] += 10.0
    out_a, out_b = attn(a), attn(b)
    assert torch.allclose(out_a[:, :4], out_b[:, :4], atol=1e-5)
    assert not torch.allclose(out_a[:, -1], out_b[:, -1], atol=1e-5)


def test_attention_has_no_biases_anywhere():
    attn = CausalSelfAttention(16, 4, 2, rope=True)
    assert all(m.bias is None for m in attn.modules() if isinstance(m, nn.Linear))


def test_grouped_query_shares_kv_heads():
    attn = CausalSelfAttention(16, 4, 2, rope=True)
    assert attn.group == 2
    assert attn.k_proj.out_features == 8
    x = torch.randn(1, 6, 16)
    assert attn(x).shape == (1, 6, 16)


def test_multi_query_and_multi_head_build():
    for kv in (1, 4):
        attn = CausalSelfAttention(16, 4, kv, rope=True)
        assert attn(torch.randn(1, 4, 16)).shape == (1, 4, 16)


def test_head_count_must_divide():
    for n_head, n_kv_head in ((5, 2), (4, 3)):
        try:
            CausalSelfAttention(20, n_head, n_kv_head, rope=True)
        except ValueError:
            continue
        raise AssertionError("expected a ValueError")


def test_decoder_block_is_pre_norm_and_residual():
    block = DecoderBlock(LLAMA.d_model, 2, 2, 32, norm="rms", rope=True,
                         mlp="swiglu")
    x = torch.randn(1, 4, LLAMA.d_model)
    with torch.no_grad():
        block.attn.o_proj.weight.zero_()
        block.mlp.down.weight.zero_()
        assert torch.allclose(block(x), x)


def test_two_norms_per_block_and_one_at_the_end():
    block = DecoderBlock(8, 2, 2, 16, norm="rms", rope=True, mlp="gelu")
    norms = [m for m in block.modules() if isinstance(m, RMSNorm)]
    assert len(norms) == 2
    assert isinstance(block.mlp, GeLUMLP)
