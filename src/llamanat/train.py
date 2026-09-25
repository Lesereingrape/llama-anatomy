"""Training and scoring for both tasks, under one identical budget per variant.

One AdamW loop, one batch size, one step count, one learning rate for every
config in the ablation. That is the point: if a variant loses, it loses with
the same compute as the winner, so the gap can be attributed to the single
mechanism that was switched and to nothing else.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import TinyLLaMA, n_params, variant_config
from .tasks import VOCAB, Addition, sample_addition, sample_recall

DEFAULT_STEPS = 1200
DEFAULT_LR = 1e-3
DEFAULT_BATCH = 64


@dataclass(frozen=True)
class TaskSpec:
    """What the runner needs to know about one task family."""

    name: str
    kind: str                      # "addition" | "recall"
    n_pairs: int = 0
    width: int = 3
    n_train: int = 4000
    n_eval: int = 400
    vocab: int = VOCAB
    #: how recall reports its answer: a private name symbol, or the value token
    answer: str = "name"

    @property
    def is_addition(self) -> bool:
        return self.kind == "addition"

    @property
    def recall(self) -> bool:
        return self.kind == "recall"


def splits(spec: TaskSpec, seed: int) -> tuple[list, list, list]:
    """(train, val, eval) drawn from one stream, kept disjoint by construction.

    Addition shares a ``seen`` set across the three slices, because 3-digit
    operands are few enough to collide. Recall cannot collide in the way that
    matters: its eval items are the same length as training but drawn from a
    continuing stream, and the long-sequence probes are built at lengths the
    model has never trained on at all.
    """
    rng = random.Random(1000 + seed)
    if spec.is_addition:
        seen: set[tuple[int, int]] = set()
        train = sample_addition(rng, spec.n_train, spec.width, seen)
        val = sample_addition(rng, spec.n_eval, spec.width, seen)
        evals = sample_addition(rng, spec.n_eval, spec.width, seen)
        return train, val, evals
    train = sample_recall(rng, spec.n_train, spec.n_pairs, spec.answer)
    val = sample_recall(rng, spec.n_eval, spec.n_pairs, spec.answer)
    evals = sample_recall(rng, spec.n_eval, spec.n_pairs, spec.answer)
    return train, val, evals


def collate(items: list) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    """Stack same-layout examples into (ids, supervised positions, targets)."""
    rows = [it.example() for it in items]
    ids = torch.tensor([r[0] for r in rows], dtype=torch.long)
    positions = rows[0][1]
    if any(r[1] != positions for r in rows):
        raise ValueError("examples in one batch must share their layout")
    if len({len(r[0]) for r in rows}) != 1:
        raise ValueError("examples in one batch must share their length")
    targets = torch.tensor([r[2] for r in rows], dtype=torch.long)
    return ids, positions, targets


def supervised_loss(model: TinyLLaMA, ids: torch.Tensor, positions: list[int],
                    targets: torch.Tensor) -> torch.Tensor:
    logits = model.logits_at(ids, positions)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           targets.reshape(-1))


@torch.no_grad()
def evaluate_loss(model: TinyLLaMA, items: list, batch: int = 200) -> float:
    """Mean cross-entropy per supervised position, on held-out examples."""
    model.eval()
    ids, positions, targets = collate(items)
    total = 0.0
    for start in range(0, ids.size(0), batch):
        rows = slice(start, start + batch)
        total += float(supervised_loss(model, ids[rows], positions,
                                       targets[rows])) * ids[rows].size(0)
    return total / ids.size(0)


def generate_addition(model: TinyLLaMA, ex: Addition) -> list[int]:
    """Free-run the chain one token at a time -- no teacher forcing anywhere."""
    prompt = ex.prompt()
    ids = list(prompt)
    for _ in range(2 * ex.width):
        nxt = model.greedy_at(torch.tensor([ids], dtype=torch.long), len(ids) - 1)
        ids.append(int(nxt[0]))
    return ids[len(prompt):]


@torch.no_grad()
def score(model: TinyLLaMA, items: list) -> float:
    """End-task accuracy: exact sum for addition, exact slot for recall."""
    if not items:
        return 0.0
    model.eval()
    if isinstance(items[0], Addition):
        hits = sum(
            it.parse(generate_addition(model, it)) == it.target for it in items
        )
        return hits / len(items)
    ids, positions, targets = collate(items)
    predicted = model.logits_at(ids, positions)[:, -1].argmax(dim=-1)
    return float((predicted == targets[:, -1]).float().mean())


@dataclass
class Fitted:
    model: TinyLLaMA
    variant: str
    task: str
    seed: int
    params: int
    val_loss: float
    steps: int
    eval_items: list
    curve: dict


def build_model(variant: str, spec: TaskSpec, max_len: int = 48,
                **over: int | float | str | bool) -> TinyLLaMA:
    """One ablation variant, with any extra config overrides applied last."""
    return TinyLLaMA(variant_config(variant, vocab=spec.vocab, max_len=max_len,
                                    **over))


def fit(spec: TaskSpec, variant: str, seed: int, steps: int = DEFAULT_STEPS,
        batch: int = 64, lr: float = DEFAULT_LR, max_len: int = 48,
        checkpoints: tuple[int, ...] = (),
        model_over: dict | None = None) -> Fitted:
    """Train one variant on one task and return the model, not a summary.

    ``checkpoints`` scores the same weights, on the same eval slice, at the
    listed step counts, without restarting the optimizer. That is what makes
    "how long did this variant need" comparable: the rows of a curve are
    points on one trajectory, not separate runs that could land on either side
    of a phase transition by luck.
    """
    torch.manual_seed(seed)
    model = build_model(variant, spec, max_len, **(model_over or {}))
    train_items, val_items, eval_items = splits(spec, seed)
    ids, positions, targets = collate(train_items)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = ids.size(0)
    wanted = set(checkpoints)
    curve: dict[int, float] = {}
    for step in range(1, steps + 1):
        model.train()
        rows = torch.arange((step - 1) * batch, step * batch) % n
        loss = supervised_loss(model, ids[rows], positions, targets[rows])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in wanted:
            curve[step] = round(score(model, eval_items), 4)
    return Fitted(
        model=model,
        variant=variant,
        task=spec.name,
        seed=seed,
        params=n_params(model),
        val_loss=round(evaluate_loss(model, val_items), 4),
        steps=steps,
        eval_items=eval_items,
        curve=curve,
    )


@dataclass
class Run:
    variant: str
    task: str
    seed: int
    params: int
    val_loss: float
    accuracy: float
    steps: int
    curve: dict
    n_pairs: int = 0


def train_run(spec: TaskSpec, variant: str, seed: int, **kw: int | float) -> Run:
    """One cell of the in-domain ablation matrix."""
    fitted = fit(spec, variant, seed, **kw)
    return Run(
        variant=variant,
        task=spec.name,
        seed=seed,
        params=fitted.params,
        val_loss=fitted.val_loss,
        accuracy=round(score(fitted.model, fitted.eval_items), 4),
        steps=fitted.steps,
        curve=dict(fitted.curve),
        n_pairs=spec.n_pairs,
    )


def probe(fitted: Fitted, n_pairs: int, spec: TaskSpec, n: int = 400) -> float:
    """Score already-fitted weights on longer sequences than it trained on.

    Restricted to lengths above the training length on purpose: an in-domain
    number here would come from a different random slice than the one the
    matrix reports, and two sources for the same cell is how tables lie.
    """
    if not spec.recall or n_pairs <= spec.n_pairs:
        raise ValueError("probe only measures recall beyond the trained length")
    items = sample_recall(random.Random(7000 + fitted.seed), n, n_pairs,
                          spec.answer)
    return round(score(fitted.model, items), 4)


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))
