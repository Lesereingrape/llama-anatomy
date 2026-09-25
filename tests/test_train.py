"""Training, scoring and the guards that keep the two honest.

The rule these tests protect: a number enters the README through exactly one
path, and re-asking the same question gives the same answer.
"""

from __future__ import annotations

import math
import random

import pytest
import torch

from llamanat.model import TinyLLaMA, n_params, variant_config
from llamanat.tasks import (
    VOCAB,
    Addition,
    Recall,
    answer_token,
    sample_recall,
)
from llamanat.train import (
    TaskSpec,
    build_model,
    collate,
    evaluate_loss,
    fit,
    generate_addition,
    perplexity,
    probe,
    score,
    splits,
    supervised_loss,
    train_run,
)

RECALL = TaskSpec("recall", "recall", n_pairs=2, n_train=256, n_eval=64)
#: One digit, so the sums are learnable in the ~150 steps these guards spend.
#: That also caps the split: a one-digit operand space holds 9*9 pairs, and
#: ``splits`` draws train+val+eval from it without repetition.
ADD1 = TaskSpec("add1", "addition", width=1, n_train=48, n_eval=8)


def test_splits_are_a_stream_not_three_draws():
    a_train, a_val, a_eval = splits(RECALL, 0)
    assert (len(a_train), len(a_val), len(a_eval)) == (256, 64, 64)
    b_train, _, b_eval = splits(RECALL, 0)
    assert [x.ids() for x in a_train] == [x.ids() for x in b_train]
    assert [x.ids() for x in a_eval] == [x.ids() for x in b_eval]


def test_a_different_seed_changes_the_data():
    _, _, e0 = splits(RECALL, 0)
    _, _, e1 = splits(RECALL, 1)
    assert [x.ids() for x in e0] != [x.ids() for x in e1]


def test_addition_slices_cannot_share_an_operand_pair():
    train, val, ev = splits(ADD1, 3)
    keys = {(x.a, x.b) for x in train}
    assert keys.isdisjoint({(x.a, x.b) for x in val})
    assert keys.isdisjoint({(x.a, x.b) for x in ev})


def test_the_answer_mode_reaches_the_training_data():
    """Whichever alphabet a run is graded in, that is the alphabet it learns.

    A ``TaskSpec`` that forgot to pass its mode through would train one
    formulation and report the other, which is the quietest way to make an
    answer-alphabet control like ``study.run_answer_alphabet`` meaningless.
    """
    named = TaskSpec("recall", "recall", n_pairs=2, n_train=32, n_eval=8)
    copied = TaskSpec("recall", "recall", n_pairs=2, n_train=32, n_eval=8,
                      answer="copy")
    for spec, in_the_input in ((named, False), (copied, True)):
        train, _, ev = splits(spec, 0)
        for item in (*train, *ev):
            assert (item.target_token in item.ids()) is in_the_input


def test_collate_stacks_one_layout():
    items = sample_recall(random.Random(0), 5, n_pairs=2)
    ids, positions, targets = collate(items)
    assert ids.dtype == torch.long and targets.dtype == torch.long
    assert ids.shape == (5, items[0].length)
    assert targets.shape == (5, 1)
    assert positions == [items[0].length - 1]


def test_collate_refuses_mixed_lengths():
    items = [Addition(12, 30, width=2), Addition(5, 6, width=1)]
    with pytest.raises(ValueError, match="length"):
        collate(items)


def test_supervised_loss_is_cross_entropy_at_the_supervised_rows():
    model = build_model("llama", RECALL)
    items = sample_recall(random.Random(1), 4, n_pairs=2)
    ids, positions, targets = collate(items)
    logits = model.logits_at(ids, positions)
    manual = torch.nn.functional.cross_entropy(
        logits.reshape(-1, VOCAB), targets.reshape(-1)
    )
    assert torch.allclose(supervised_loss(model, ids, positions, targets), manual)


def test_an_untrained_model_is_near_chance_on_recall():
    torch.manual_seed(0)
    model = TinyLLaMA(variant_config("llama"))
    items = sample_recall(random.Random(2), 200, n_pairs=2)
    # one answer symbol per value, out of a 61-token alphabet: chance is ~1/61
    assert score(model, items) < 0.10


class _Scripted:
    """Emits one prescribed token per row, so grading can be tested alone."""

    def __init__(self, tokens: list[int]):
        self.tokens = tokens

    def eval(self) -> None:
        return None

    def logits_at(self, ids: torch.Tensor, positions: list[int]) -> torch.Tensor:
        out = torch.zeros(ids.size(0), len(positions), VOCAB)
        for row, token in enumerate(self.tokens):
            out[row, :, token] = 1.0
        return out


def test_grading_is_a_strict_symbol_match_not_a_pair_check():
    """The right pair, written in the wrong alphabet, is graded wrong.

    A copier that re-emits the value token has recalled the answer, but the
    matrix task asks for its private symbol, so the two runs are never scored
    with each other's key -- that mix-up is the one way these numbers could be
    quietly inflated.
    """
    named = sample_recall(random.Random(0), 12, n_pairs=2, answer="name")
    assert score(_Scripted([answer_token(i.target) for i in named]), named) == 1.0
    assert score(_Scripted([Recall.value_token(i.target) for i in named]),
                 named) == 0.0
    copied = sample_recall(random.Random(0), 12, n_pairs=2, answer="copy")
    assert score(_Scripted([Recall.value_token(i.target) for i in copied]),
                 copied) == 1.0


def test_score_counts_only_exact_targets():
    items = sample_recall(random.Random(1), 10, n_pairs=2, answer="name")
    half = [answer_token(i.target) for i in items[:5]]
    wrong = [answer_token((i.target + 1) % 16) for i in items[5:]]
    assert score(_Scripted(half + wrong), items) == pytest.approx(0.5)
    fixed = items[0].target_token
    expect = sum(i.target_token == fixed for i in items) / len(items)
    assert score(_Scripted([fixed] * len(items)), items) == pytest.approx(expect)


def test_generate_addition_emits_one_token_per_chain_slot():
    torch.manual_seed(0)
    model = TinyLLaMA(variant_config("llama"))
    chain = generate_addition(model, Addition(37, 64, width=3))
    assert len(chain) == 2 * 3
    prompt_len = len(Addition(37, 64, width=3).prompt())
    assert prompt_len == 8


def test_fit_scores_only_the_checkpoints_it_was_given():
    fitted = fit(ADD1, "llama", 0, steps=6, checkpoints=(2, 5, 9))
    assert sorted(fitted.curve) == [2, 5]
    assert fitted.steps == 6
    assert fitted.params > 0


def test_the_same_seed_reproduces_the_same_trajectory():
    one = fit(RECALL, "llama", 0, steps=20, checkpoints=(10, 20))
    two = fit(RECALL, "llama", 0, steps=20, checkpoints=(10, 20))
    assert one.curve == two.curve
    assert one.val_loss == two.val_loss
    assert one.params == two.params


def test_a_different_seed_gives_a_different_trajectory():
    one = fit(RECALL, "llama", 0, steps=20, checkpoints=(20,))
    two = fit(RECALL, "llama", 7, steps=20, checkpoints=(20,))
    assert one.curve != two.curve or one.val_loss != two.val_loss


def test_run_accuracy_is_the_last_checkpoint_not_a_second_measurement():
    steps = 30
    run = train_run(ADD1, "llama", 0, steps=steps, checkpoints=(steps,))
    assert run.accuracy == run.curve[steps]
    assert run.steps == steps
    assert run.curve is not None


def test_evaluate_loss_is_the_mean_per_row_whatever_the_batch_size():
    items = sample_recall(random.Random(4), 7, n_pairs=2)
    model = build_model("llama", RECALL)
    whole = evaluate_loss(model, items, batch=200)
    assert whole == pytest.approx(evaluate_loss(model, items, batch=3), rel=1e-4)
    assert 0.0 < whole < math.log(VOCAB) * 1.5


def test_model_overrides_reach_the_built_model():
    tied = build_model("llama", RECALL)
    open_ = build_model("llama", RECALL, tie_embeddings=False)
    assert tied.lm_head.weight is tied.tok_emb.weight
    assert open_.lm_head.weight is not open_.tok_emb.weight
    assert n_params(open_) - n_params(tied) == RECALL.vocab * tied.cfg.d_model


def test_fit_records_an_overridden_model():
    plain = n_params(build_model("llama", RECALL))
    fitted = fit(RECALL, "llama", 0, steps=2, model_over={"tie_embeddings": False})
    assert fitted.params == plain + RECALL.vocab * fitted.model.cfg.d_model


def test_training_lowers_the_loss_and_raises_the_score():
    fitted = fit(ADD1, "llama", 0, steps=150, checkpoints=(5, 150))
    assert fitted.curve[150] > fitted.curve[5]
    assert fitted.val_loss < 2.0


def test_probe_refuses_to_measure_the_trained_length():
    fitted = fit(RECALL, "llama", 0, steps=2)
    with pytest.raises(ValueError, match="beyond the trained length"):
        probe(fitted, RECALL.n_pairs, RECALL)
    with pytest.raises(ValueError, match="beyond the trained length"):
        probe(fitted, RECALL.n_pairs - 1, RECALL)


def test_probe_refuses_a_task_it_cannot_lengthen():
    fitted = fit(ADD1, "llama", 0, steps=2)
    with pytest.raises(ValueError, match="recall"):
        probe(fitted, 8, ADD1)


def test_probe_never_reuses_the_trained_slice():
    fitted = fit(RECALL, "llama", 0, steps=2)
    items = sample_recall(random.Random(7000), 5, n_pairs=8)
    assert [x.ids() for x in items] != [x.ids() for x in fitted.eval_items]


def test_perplexity_is_capped_and_monotone():
    assert perplexity(math.log(2)) == pytest.approx(2.0)
    assert perplexity(20.0) == perplexity(50.0)
    assert perplexity(0.5) < perplexity(1.5)
