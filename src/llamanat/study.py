"""The measurement plan: an acquisition curve per variant, an extrapolation
probe, a rotation-base sweep, and an exact parameter/cache table.

``build_results`` is the only origin of the JSON the README renders from. Within
one process the plan is deterministic -- the four call sites that re-fit the
reference model agree bit for bit, and a rerun of that process reproduces this
file -- but determinism is not claimed *across* environments, so the run records
the environment it was measured in. See ``_environment``.
"""

from __future__ import annotations

import platform
import statistics
import sys
import time
from dataclasses import asdict

import torch

from .kv import cache_report
from .model import LLAMA, VARIANTS
from .train import (
    DEFAULT_BATCH,
    DEFAULT_LR,
    DEFAULT_STEPS,
    TaskSpec,
    fit,
    probe,
    score,
    train_run,
)

SEEDS = (0, 1, 2)
STEPS = DEFAULT_STEPS
BATCH = DEFAULT_BATCH
LR = DEFAULT_LR
CHECKPOINTS = (300, 600, 900, 1200)

TASKS = {
    "addition": TaskSpec("addition", "addition", width=3, n_eval=200),
    "recall": TaskSpec("recall", "recall", n_pairs=4, n_eval=200, answer="name"),
}

VARIANT_ORDER = ("llama", "layernorm", "abs_pos", "gelu_mlp", "mha", "mqa")

TRAIN_PAIRS = 4
PROBE_PAIRS = (4, 8, 12)
POSITION_VARIANTS = ("llama", "abs_pos")

#: the control on the task itself: does the alphabet the answer is written in
#: change what a block of this size can fit? One seed each, both positional
#: variants at the same budget as the matrix, plus the obvious explanation for
#: the difference (input and output share one embedding table) tested directly.
ALPHABET_SEED = 0
ALPHABET_PROBE_PAIRS = 8
ALPHABET_ROWS = (
    ("name", "llama", {}),
    ("name", "abs_pos", {}),
    ("copy", "llama", {}),
    ("copy", "abs_pos", {}),
    ("copy", "llama", {"tie_embeddings": False}),
)

#: RoPE's rotation base, swept on the recall task at the matrix's own budget.
#: Every other row in the README changes a mechanism that costs parameters;
#: ``rope_theta`` costs exactly none, so this is the ablation of the geometry
#: alone. The extremes are chosen to bracket the paper's 1e4: at head_dim 16 a
#: base of 10 stops being able to represent an offset much beyond ~47 positions,
#: and 1e6 is fifty times the whole context window in reach.
THETA_SWEEP = (10.0, 100.0, 10_000.0, 1_000_000.0)
THETA_PROBE_PAIRS = PROBE_PAIRS[1:]


def run_theta_sweep() -> dict:
    """The reference model at four rotation bases, one trajectory each per seed.

    ``rope_theta`` is the only axis in this repository that adds no parameter and
    changes no tensor shape, so a difference here cannot be a size difference
    dressed up as a mechanism. The in-domain column is the last checkpoint of the
    same fits whose curves are reported; the rest is the same
    train-short/test-long probe as the extrapolation table, run on weights that
    were never re-tuned.
    """
    spec = TaskSpec("recall", "recall", n_pairs=TRAIN_PAIRS, n_eval=200)
    rows = []
    for theta in THETA_SWEEP:
        fitted = [
            fit(spec, "llama", seed, steps=STEPS, batch=BATCH, lr=LR,
                checkpoints=CHECKPOINTS, model_over={"rope_theta": theta})
            for seed in SEEDS
        ]
        curves = {
            str(step): _summary([f.curve[step] for f in fitted])
            for step in CHECKPOINTS
        }
        out: dict[str, dict] = {}
        for pairs in THETA_PROBE_PAIRS:
            out[str(pairs)] = _summary(
                [probe(f, pairs, spec, n=200) for f in fitted])
        rows.append({
            "rope_theta": theta,
            "params": fitted[0].params,
            "val_loss": _summary([f.val_loss for f in fitted]),
            "curve": curves,
            "final_accuracy": curves[str(CHECKPOINTS[-1])],
            "out_of_length": out,
        })
    return {
        "task": spec.name,
        "variant": "llama",
        "seeds": list(SEEDS),
        "trained_on_pairs": TRAIN_PAIRS,
        "rows": rows,
    }


def run_answer_alphabet() -> dict:
    """The same sequences, graded through two different answer alphabets.

    This is a control on the *task*, not one of the matrix rows: it records why
    the reported recall numbers name the answer with a private symbol instead of
    asking the model to copy the value token back out. Copying is the
    formulation a reader would expect, so the reason it is not used has to be
    measured rather than asserted -- including the tempting explanation in
    terms of the tied embedding table, which the last row tests.
    """
    rows = []
    for mode, variant, over in ALPHABET_ROWS:
        spec = TaskSpec("recall", "recall", n_pairs=TRAIN_PAIRS, n_eval=200,
                        answer=mode)
        fitted = fit(spec, variant, ALPHABET_SEED, steps=STEPS, batch=BATCH,
                     lr=LR, checkpoints=CHECKPOINTS, model_over=over)
        rows.append({
            "mode": mode,
            "variant": variant,
            "config_overrides": dict(over),
            "params": fitted.params,
            "val_loss": fitted.val_loss,
            "curve": {str(step): _summary([acc])
                      for step, acc in sorted(fitted.curve.items())},
            "probe": {
                "pairs": ALPHABET_PROBE_PAIRS,
                "mean": probe(fitted, ALPHABET_PROBE_PAIRS, spec, n=200),
            },
        })
    return {
        "seeds": [ALPHABET_SEED],
        "probe_pairs": ALPHABET_PROBE_PAIRS,
        "trained_on_pairs": TRAIN_PAIRS,
        "rows": rows,
    }


def _summary(values: list[float]) -> dict:
    return {
        "mean": round(statistics.fmean(values), 4),
        "std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
        "seeds": [round(v, 4) for v in values],
    }


def _environment() -> dict:
    """The origin of the run, recorded next to the numbers.

    A fixed process reproduces this file exactly, and it does not matter to the
    conclusions which one produced it -- but float reduction order over a batch
    is thread-dependent, so a rerun under another build or thread count can move
    a seed that sits within rounding of an acquisition threshold. Rather than
    assert reproducibility the README cannot deliver, the artifact states which
    environment delivered it.
    """
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "device": "cpu",
    }


def run_task(task: TaskSpec) -> dict:
    """Accuracy at each checkpoint of one shared trajectory, over all seeds."""
    cells: dict[str, dict] = {}
    for variant in VARIANT_ORDER:
        runs = [
            train_run(task, variant, seed, steps=STEPS, batch=BATCH, lr=LR,
                      checkpoints=CHECKPOINTS)
            for seed in SEEDS
        ]
        curves = {
            str(step): _summary([r.curve[step] for r in runs])
            for step in CHECKPOINTS
        }
        cells[variant] = {
            "params": runs[0].params,
            "val_loss": _summary([r.val_loss for r in runs]),
            "curve": curves,
            "final_accuracy": curves[str(CHECKPOINTS[-1])],
        }
    return cells


def run_extrapolation() -> dict:
    """Train short on 4 pairs, then score the same weights at 4, 8 and 12.

    One fit per (variant, seed) feeds all three lengths, so a row of the table
    differs from its neighbour only in how far the sequence has run past
    training.
    """
    spec = TaskSpec("recall", "recall", n_pairs=TRAIN_PAIRS, n_eval=200)
    out: dict[str, dict] = {}
    for variant in POSITION_VARIANTS:
        fitted = [fit(spec, variant, seed, steps=STEPS, batch=BATCH, lr=LR)
                  for seed in SEEDS]
        curves: dict[str, dict] = {}
        for pairs in PROBE_PAIRS:
            if pairs == TRAIN_PAIRS:
                accs = [round(score(f.model, f.eval_items), 4) for f in fitted]
            else:
                accs = [probe(f, pairs, spec, n=200) for f in fitted]
            curves[str(pairs)] = _summary(accs)
        out[variant] = {
            "trained_on_pairs": TRAIN_PAIRS,
            "val_loss": _summary([f.val_loss for f in fitted]),
            "curves": curves,
        }
    return out


def variant_notes() -> dict:
    """What each row actually changed, read straight out of the config."""
    base = asdict(LLAMA)
    return {
        name: {"overrides": dict(over), "replaces": {k: base[k] for k in over}}
        for name, over in ((n, VARIANTS[n]) for n in VARIANT_ORDER)
    }


def build_results() -> dict:
    start = time.perf_counter()
    environment = _environment()
    in_domain = {name: run_task(spec) for name, spec in TASKS.items()}
    results = {
        "environment": environment,
        "settings": {
            "seeds": list(SEEDS),
            "steps": STEPS,
            "checkpoints": list(CHECKPOINTS),
            "batch": BATCH,
            "lr": LR,
            "train_items": {n: t.n_train for n, t in TASKS.items()},
            "eval_items": {n: t.n_eval for n, t in TASKS.items()},
            "recall_answer_mode": TASKS["recall"].answer,
            "base_config": asdict(LLAMA),
        },
        "variants": variant_notes(),
        "in_domain": in_domain,
        "extrapolation": run_extrapolation(),
        "answer_alphabet": run_answer_alphabet(),
        "theta_sweep": run_theta_sweep(),
        "kv_cache": cache_report(LLAMA),
        "n_params_llama": in_domain["addition"]["llama"]["params"],
        "runtime_sec": round(time.perf_counter() - start, 1),
    }
    return results
