"""Two verifiable sequence tasks over one 61-token alphabet.

``Addition`` solves ``a + b`` digit by digit, least significant column first,
emitting output digit then carry. It asks whether a block can run a
two-variable recurrence across positions.

``Recall`` is associative memory: ``n`` key-value pairs, then a probe key, and
the model must report the value that pair held. It asks for long-range attention
at a distance the model has never seen, which is exactly the axis RoPE and
learned absolute positions disagree about.

The value can be reported two ways, and it turns out to matter (see
``study.run_answer_alphabet``):

``name``   the value's own symbol from a dedicated 16-token answer alphabet,
           which never appears anywhere in the input sequence.
``copy``   the value token itself, i.e. a literal copy of something the model
           has already seen.

``name`` is what the ablation matrix and the extrapolation table use, because a
~100k-parameter, two-layer block will not fit ``copy`` at four pairs. That is a
measured constraint, not a preference, so both are implemented and both are
reported.

Both tasks are graded by construction -- the answer is either the right number or
the value the pair actually held -- and neither ever pads: every example in a
batch has the same length, so a shape bug cannot hide behind a pad token.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

PAD, SEP, EQ = 0, 1, 2
DIGIT0 = 3
N_KEYS = 16
KEY0 = DIGIT0 + 10
VAL0 = KEY0 + N_KEYS
ANS0 = VAL0 + N_KEYS
VOCAB = ANS0 + N_KEYS

#: how the probed value is reported: a private symbol, or the token itself
ANSWER_MODES = ("name", "copy")


def digits_of(value: int) -> list[int]:
    return [int(c) for c in str(value)]


def digit_token(d: int) -> int:
    return DIGIT0 + d


def token_digit(t: int) -> int:
    return t - DIGIT0


def answer_token(v: int) -> int:
    """The private symbol for value ``v``, in the alphabet no input uses."""
    return ANS0 + v


def recall_length(n_pairs: int) -> int:
    """Tokens in a recall sequence of ``n_pairs`` pairs: ``k v`` per pair, one
    separator between pairs, then ``= k`` for the probe."""
    return 3 * n_pairs + 1


@dataclass(frozen=True)
class Addition:
    a: int
    b: int
    width: int = 3

    @property
    def target(self) -> int:
        return self.a + self.b

    def _operand(self, value: int) -> list[int]:
        digs = digits_of(value)
        padded = (self.width - len(digs)) * [0] + digs
        return [digit_token(d) for d in padded]

    def prompt(self) -> list[int]:
        """Both operands most-significant-first, the way the sum is written."""
        return [*self._operand(self.a), SEP, *self._operand(self.b), EQ]

    def chain(self) -> list[int]:
        """``o_0 c_1 o_1 c_2 ... o_{k-1} c_k``, least significant column first.

        Column i's digit can only be written once column i-1's carry exists, so
        reading the chain left to right is causal: the model has to carry its
        own state forward rather than peek at the answer.
        """
        ad = digits_of(self.a)[::-1]
        bd = digits_of(self.b)[::-1]
        out: list[int] = []
        carry = 0
        for i in range(self.width):
            da = ad[i] if i < len(ad) else 0
            db = bd[i] if i < len(bd) else 0
            total = da + db + carry
            out.append(digit_token(total % 10))
            carry = total // 10
            out.append(digit_token(carry))
        return out

    def example(self) -> tuple[list[int], list[int], list[int]]:
        """(ids, supervised positions, the token due at each of them)."""
        prompt = self.prompt()
        chain = self.chain()
        ids = [*prompt, *chain]
        positions = list(range(len(prompt) - 1, len(ids) - 1))
        return ids, positions, chain

    def parse(self, tokens: list[int]) -> int | None:
        """Rebuild a number from a generated chain, or None if it is malformed."""
        if len(tokens) < 2 * self.width:
            return None
        pairs = tokens[: 2 * self.width]
        if any(not (DIGIT0 <= t < KEY0) for t in pairs):
            return None
        digs = [token_digit(t) for t in pairs]
        total = sum(digs[2 * i] * 10**i for i in range(self.width))
        return total + digs[-1] * 10**self.width


@dataclass(frozen=True)
class Recall:
    """``k1 v1 | k2 v2 | ... ? kq`` -- report the value that was paired with kq."""

    pairs: tuple[tuple[int, int], ...]
    probe: int
    answer: str = "name"

    @property
    def target(self) -> int:
        return dict(self.pairs)[self.probe]

    @property
    def target_token(self) -> int:
        """The answer, in whichever reporting mode this item uses.

        ``name`` points at a private symbol for the value; ``copy`` re-emits the
        value token the sequence already contains. Both identify the same pair,
        so both are graded against the same ground truth.
        """
        if self.answer == "name":
            return answer_token(self.target)
        if self.answer == "copy":
            return self.value_token(self.target)
        raise ValueError(f"unknown answer mode {self.answer!r}")

    @staticmethod
    def key_token(k: int) -> int:
        return KEY0 + k

    @staticmethod
    def value_token(v: int) -> int:
        return VAL0 + v

    def ids(self) -> list[int]:
        out: list[int] = []
        for i, (k, v) in enumerate(self.pairs):
            if i:
                out.append(SEP)
            out += [self.key_token(k), self.value_token(v)]
        out += [EQ, self.key_token(self.probe)]
        return out

    def example(self) -> tuple[list[int], list[int], list[int]]:
        ids = self.ids()
        return ids, [len(ids) - 1], [self.target_token]

    @property
    def length(self) -> int:
        return recall_length(len(self.pairs))


def sample_addition(rng: random.Random, n: int, width: int = 3,
                    seen: set[tuple[int, int]] | None = None) -> list[Addition]:
    """Distinct operands. Pass one ``seen`` set across splits to keep them apart.

    The space is finite: ``width`` digits leave ``(9 * 10**(width-1))**2`` operand
    pairs, so an over-large request is impossible rather than merely slow. Failing
    here is the difference between a clear error and a draw loop that spins
    forever -- which is what a one-digit split of 384 pairs used to do.
    """
    if seen is None:
        seen = set()
    lo, hi = 10 ** (width - 1), 10**width - 1
    space = (hi - lo + 1) ** 2
    used = sum(1 for a, b in seen if lo <= a <= hi and lo <= b <= hi)
    if n > space - used:
        raise ValueError(
            f"{n} distinct {width}-digit operand pairs requested, "
            f"{space - used} of {space} still unused; widen the operands or "
            "shrink the split")
    out: list[Addition] = []
    while len(out) < n:
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append(Addition(a, b, width))
    return out


def sample_recall(rng: random.Random, n: int, n_pairs: int,
                  answer: str = "name") -> list[Recall]:
    """Keys and values are unique inside an item, and the probe is a live key."""
    out: list[Recall] = []
    for _ in range(n):
        keys = rng.sample(range(N_KEYS), n_pairs)
        values = rng.sample(range(N_KEYS), n_pairs)
        pairs = tuple(zip(keys, values, strict=True))
        out.append(Recall(pairs, rng.choice(keys), answer))
    return out
