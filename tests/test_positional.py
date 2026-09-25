import math

import torch

from llamanat.blocks import CausalSelfAttention
from llamanat.model import LLAMA
from llamanat.positional import (
    AbsolutePositionEmbedding,
    apply_rope,
    longest_wavelength,
    rope_angles,
    rope_frequencies,
)


def test_frequencies_follow_the_formula():
    freqs = rope_frequencies(8, theta=10_000.0)
    expect = [10_000 ** (-2 * i / 8) for i in range(4)]
    assert torch.allclose(freqs, torch.tensor(expect), rtol=1e-5)


def test_rotation_only_depends_on_the_relative_offset():
    torch.manual_seed(0)
    freqs = rope_frequencies(8)
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)
    base = float((apply_rope(q, freqs, torch.tensor([7]))
                  * apply_rope(k, freqs, torch.tensor([3]))).sum())
    same = float((apply_rope(q, freqs, torch.tensor([40]))
                  * apply_rope(k, freqs, torch.tensor([36]))).sum())
    other = float((apply_rope(q, freqs, torch.tensor([40]))
                   * apply_rope(k, freqs, torch.tensor([4]))).sum())
    assert abs(base - same) < 1e-4
    assert abs(base - other) > 1e-3


def test_rotation_preserves_vector_norm():
    torch.manual_seed(1)
    x = torch.randn(2, 3, 5, 8)
    freqs = rope_frequencies(8)
    rotated = apply_rope(x, freqs, torch.arange(5))
    assert torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-4)


def test_position_zero_is_the_identity():
    x = torch.randn(1, 1, 1, 8)
    freqs = rope_frequencies(8)
    assert torch.allclose(apply_rope(x, freqs, torch.tensor([0])), x, atol=1e-6)


def test_angles_have_half_the_head_dim_per_position():
    cos, sin = rope_angles(rope_frequencies(16), torch.arange(4))
    assert cos.shape == (4, 8)
    assert torch.allclose(cos.pow(2) + sin.pow(2), torch.ones_like(cos), atol=1e-6)


def test_absolute_table_refuses_to_run_past_its_length():
    table = AbsolutePositionEmbedding(6, 4)
    assert table(6).shape == (6, 4)
    try:
        table(7)
    except ValueError as exc:
        assert "exceeds" in str(exc)
    else:
        raise AssertionError("expected a ValueError")


def test_absolute_rows_are_independent():
    """Unshared parameters per slot -- the thing RoPE does not have.

    A learned table can store "column 4 owns the units digit" only because row
    4 has nothing to do with row 1: one training step that reaches row 1 leaves
    every other row exactly as it was.
    """
    table = AbsolutePositionEmbedding(6, 4)
    table(6)[1].sum().backward()
    grad = table.table.weight.grad
    assert grad[1].abs().sum() > 0
    assert torch.equal(grad[:1], torch.zeros_like(grad[:1]))
    assert torch.equal(grad[2:], torch.zeros_like(grad[2:]))


def test_rope_theta_matches_the_paper_default():
    attn = CausalSelfAttention(LLAMA.d_model, LLAMA.n_head, LLAMA.n_kv_head, True)
    assert float(attn.freqs[0]) == 1.0
    assert math.isclose(float(attn.freqs[-1]), 10_000 ** (-2 * 7 / 16), rel_tol=1e-4)


def test_the_reach_is_the_slowest_pairs_period():
    """theta buys range and not precision, so the two ends differ by design.

    ``longest_wavelength`` is the number the README's sweep table prints next to
    each base, so it has to be the period of the frequencies actually in use --
    the last entry of ``rope_frequencies`` -- and not a restatement of the
    formula that would keep agreeing with itself if the formula changed.
    """
    head_dim, theta = 16, 10_000.0
    freqs = rope_frequencies(head_dim, theta)
    assert math.isclose(2 * math.pi / float(freqs[-1]),
                        longest_wavelength(head_dim, theta), rel_tol=1e-6)
    assert math.isclose(2 * math.pi / float(freqs[0]), 2 * math.pi, rel_tol=1e-6)
    assert longest_wavelength(16, 10.0) < 48  # cannot span this model's window
    assert longest_wavelength(16, 10.0) < longest_wavelength(16, 1_000.0)
