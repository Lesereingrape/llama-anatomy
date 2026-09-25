"""The two tasks are graded by construction, so the grader is testable too.

If ``Addition.parse`` and ``Addition.chain`` disagreed the reported accuracy
would be measuring the parser. These tests pin the label side of every number in
the README.
"""

from __future__ import annotations

import random

import pytest

from llamanat.tasks import (
    ANS0,
    ANSWER_MODES,
    EQ,
    KEY0,
    N_KEYS,
    PAD,
    SEP,
    VAL0,
    VOCAB,
    Addition,
    Recall,
    answer_token,
    digit_token,
    recall_length,
    sample_addition,
    sample_recall,
    token_digit,
)


def test_chain_reproduces_the_sum():
    rng = random.Random(0)
    for _ in range(200):
        a, b = rng.randint(100, 999), rng.randint(100, 999)
        ex = Addition(a, b, width=3)
        assert ex.parse(ex.chain()) == a + b


def test_chain_is_a_digit_carry_alternation():
    ex = Addition(487, 268, width=3)
    chain = ex.chain()
    assert len(chain) == 2 * ex.width
    assert all(token_digit(t) in range(10) for t in chain)
    # carries between 3-digit numbers are only ever 0 or 1
    assert {token_digit(chain[i]) for i in range(1, len(chain), 2)} <= {0, 1}


def test_chain_runs_least_significant_column_first():
    # 487 + 268 = 755: units 7+8=15 -> digit 5 carry 1; tens 8+6+1=15 -> 5, 1;
    # hundreds 4+2+1=7 -> 7, 0
    assert [token_digit(t) for t in Addition(487, 268).chain()] == [5, 1, 5, 1, 7, 0]


def test_carry_out_becomes_the_extra_digit():
    ex = Addition(999, 1, width=3)
    assert ex.target == 1000
    assert ex.parse(ex.chain()) == 1000
    assert token_digit(ex.chain()[-1]) == 1


def test_operand_is_padded_most_significant_first():
    ex = Addition(7, 3, width=3)
    assert ex.prompt()[:3] == [digit_token(0), digit_token(0), digit_token(7)]
    assert ex.prompt()[3] == SEP
    assert ex.prompt()[4:7] == [digit_token(0), digit_token(0), digit_token(3)]
    assert len(ex.prompt()) == 2 * ex.width + 2
    assert ex.prompt()[-1] == EQ


def test_example_supervises_exactly_the_chain():
    ex = Addition(123, 456, width=3)
    ids, positions, targets = ex.example()
    assert ids == ex.prompt() + ex.chain()
    assert targets == ex.chain()
    assert len(positions) == len(targets)
    # the token due after position p is chain[p - (len(prompt) - 1)]
    assert positions == list(range(len(ex.prompt()) - 1, len(ids) - 1))


def test_parse_rejects_malformed_output():
    ex = Addition(111, 222, width=3)
    assert ex.parse(ex.chain()[:-1]) is None
    bad = list(ex.chain())
    bad[2] = SEP
    assert ex.parse(bad) is None
    assert ex.parse([]) is None


def test_recall_target_is_the_pair_it_held():
    pairs = ((0, 5), (3, 11), (7, 2))
    for probe in range(3):
        key = pairs[probe][0]
        item = Recall(pairs, key)
        assert item.target == pairs[probe][1]


def test_recall_layout_and_length():
    item = Recall(((0, 5), (3, 11)), 3)
    ids = item.ids()
    assert ids == [Recall.key_token(0), Recall.value_token(5), SEP,
                   Recall.key_token(3), Recall.value_token(11), EQ,
                   Recall.key_token(3)]
    assert item.length == len(ids) == 7
    ids2, positions, targets = item.example()
    assert ids2 == ids
    assert positions == [len(ids) - 1]
    assert targets == [answer_token(11)]


def test_the_length_formula_is_the_layout_not_a_restatement():
    """``recall_length`` is what the README's reach column is compared against.

    The formula has to agree with the tokens ``Recall.ids`` really emits at every
    length the study probes, otherwise the sweep table's window is a fiction.
    """
    rng = random.Random(11)
    for n_pairs in (1, 2, 4, 8, 12):
        item = sample_recall(rng, 1, n_pairs)[0]
        assert item.length == recall_length(n_pairs) == len(item.ids())
        assert recall_length(n_pairs) <= 48, "every probed length fits max_len"


def test_everything_stays_inside_the_alphabet():
    rng = random.Random(5)
    tokens: list[int] = []
    for ex in sample_addition(rng, 50, width=3):
        tokens += ex.example()[0]
    for item in sample_recall(rng, 50, 6):
        tokens += item.ids()
    assert set(tokens) <= set(range(VOCAB))
    assert PAD not in tokens
    assert max(t for t in tokens if t >= VAL0) < VOCAB
    assert min(t for t in tokens if KEY0 <= t < VAL0) >= KEY0


def test_samples_are_disjoint_across_splits():
    rng = random.Random(11)
    seen: set[tuple[int, int]] = set()
    train = sample_addition(rng, 300, width=3, seen=seen)
    val = sample_addition(rng, 50, width=3, seen=seen)
    keys = {(e.a, e.b) for e in train}
    assert len(keys) == len(train)
    assert keys.isdisjoint({(e.a, e.b) for e in val})


def test_an_impossible_addition_sample_fails_instead_of_spinning():
    """A one-digit split can ask for more pairs than the task contains.

    ``sample_addition`` draws without replacement, so 384 requests against an
    81-pair space used to loop forever -- which is how the training guards came
    to hang rather than fail. The capacity is now checked before the first draw.
    """
    with pytest.raises(ValueError, match="still unused"):
        sample_addition(random.Random(0), 400, width=1)
    seen = {(1, 1)}
    with pytest.raises(ValueError, match=r"80 of 81"):
        sample_addition(random.Random(0), 81, width=1, seen=seen)
    assert len(sample_addition(random.Random(0), 80, width=1, seen=seen)) == 80


def test_recall_items_are_well_formed():
    rng = random.Random(3)
    for item in sample_recall(rng, 100, n_pairs=6):
        keys = [k for k, _ in item.pairs]
        values = [v for _, v in item.pairs]
        assert len(set(keys)) == 6, "a repeated key would make the answer ambiguous"
        assert len(set(values)) == 6
        assert item.probe in keys
        assert all(0 <= k < N_KEYS for k in keys + values)


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_chain_and_parse_agree_at_every_width(width):
    rng = random.Random(width)
    lo, hi = 10 ** (width - 1), 10**width - 1
    for _ in range(50):
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        ex = Addition(a, b, width=width)
        assert ex.parse(ex.chain()) == a + b


def test_recall_label_is_never_a_raw_index():
    """The supervised token must name the value, not point at its slot.

    A raw value index lands in the digit and key half of the alphabet, so a model
    trained on it could answer "key 8" to a question about a value and still be
    counted correct by a grader that made the same mistake. This was the bug
    behind an earlier, much better-looking set of recall numbers.
    """
    for mode in ANSWER_MODES:
        item = Recall(((0, 5), (3, 11)), 3, mode)
        _, _, targets = item.example()
        assert targets == [item.target_token]
        assert targets[0] >= VAL0, "a raw index would label a key/digit token"
        assert item.target < VAL0


def test_every_recall_label_names_the_pair_it_held():
    rng = random.Random(17)
    for mode in ANSWER_MODES:
        for item in sample_recall(rng, 200, n_pairs=5, answer=mode):
            _, _, targets = item.example()
            due = (answer_token(item.target) if mode == "name"
                   else Recall.value_token(item.target))
            assert len(targets) == 1
            assert targets[0] == item.target_token == due < VOCAB


def test_the_two_answer_alphabets_label_the_same_pair():
    """Only the symbol differs between the modes; the ground truth does not."""
    named = Recall(((0, 5), (3, 11)), 3, "name")
    copied = Recall(((0, 5), (3, 11)), 3, "copy")
    assert named.ids() == copied.ids()
    assert named.target == copied.target == 11
    assert named.target_token == answer_token(11)
    assert copied.target_token == Recall.value_token(11)
    assert copied.target_token in copied.ids()
    assert named.target_token not in named.ids()


def test_the_answer_alphabet_is_disjoint_from_every_input():
    rng = random.Random(23)
    seen: set[int] = set()
    for item in sample_recall(rng, 400, n_pairs=N_KEYS, answer="name"):
        seen |= set(item.ids())
    for item in sample_recall(rng, 400, n_pairs=N_KEYS, answer="copy"):
        seen |= set(item.ids())
    assert seen.isdisjoint(set(range(ANS0, VOCAB)))
    assert ANS0 + N_KEYS == VOCAB


def test_an_unknown_answer_mode_is_rejected():
    item = Recall(((0, 5),), 0, "sideways")
    with pytest.raises(ValueError, match="unknown answer mode"):
        _ = item.target_token
