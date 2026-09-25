"""Command line: a demo you can read, and the entry point that writes the JSON.

    llamanat demo               # seed 0 at the published budget: rows of tables
    llamanat alphabet           # the alphabet table at its 600-step column
    llamanat study              # runs the full matrix, prints the JSON
    llamanat cache              # the exact KV-cache arithmetic, no training

Both training commands take their task, batch and learning rate from
``llamanat.study``, so a printed number is a smaller view of a published table
rather than a second measurement of a loosely related experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace

from .kv import cache_report
from .model import LLAMA
from .study import CHECKPOINTS, PROBE_N, STEPS, TASKS, build_results
from .tasks import ANSWER_MODES, DIGIT0, KEY0, token_digit
from .train import TaskSpec, fit, generate_addition, probe, score


def _show_addition(spec: TaskSpec, seed: int, steps: int) -> None:
    fitted = fit(spec, "llama", seed, steps=steps)
    print(f"llama block, {spec.name}, {steps} steps -> "
          f"in-domain accuracy {score(fitted.model, fitted.eval_items):.3f}, "
          f"val loss {fitted.val_loss:.4f}")
    for ex in fitted.eval_items[:3]:
        chain = generate_addition(fitted.model, ex)
        print(f"  {ex.a} + {ex.b} = {ex.target}   model said {ex.parse(chain)}"
              f"   chain {_digits(chain)}")


def _digits(chain: list[int]) -> str:
    return " ".join(
        str(token_digit(t)) if DIGIT0 <= t < KEY0 else f"<{t}>" for t in chain
    )


def _recall_spec(mode: str = "name") -> TaskSpec:
    """The published recall task rather than a smaller stand-in for it.

    This used to train on 1000 sequences and grade 64, so the numbers printed
    here sat below the README's alphabet table for the same seed and budget --
    which is the one thing a demo command should not do to a reader who is
    checking the table.
    """
    return replace(TASKS["recall"], answer=mode)


def _show_recall(seed: int, steps: int) -> None:
    spec = _recall_spec()
    fitted = fit(spec, "llama", seed, steps=steps)
    print(f"llama block, recall trained on {spec.n_pairs} pairs -> in-domain "
          f"{score(fitted.model, fitted.eval_items):.3f}")
    for pairs in (4, 8, 12):
        value = (score(fitted.model, fitted.eval_items) if pairs == 4
                 else probe(fitted, pairs, spec, n=PROBE_N))
        print(f"  probed at {pairs:>2} pairs: {value:.3f}")


def alphabet(seed: int = 0, steps: int = CHECKPOINTS[1]) -> int:
    """Train the same four pairs twice, once per answer alphabet.

    The shortest version of the result the README's "How the answer is asked"
    section reports: same task, same model, same batch and learning rate, and
    only the symbol the answer is written in decides whether it gets learned at
    all. The default budget is the table's 600-step checkpoint, so the printed
    accuracy is the one that column publishes.
    """
    for mode in ANSWER_MODES:
        spec = _recall_spec(mode)
        fitted = fit(spec, "llama", seed, steps=steps)
        words = ("name the value with a private symbol" if mode == "name"
                 else "copy the value token out of the context")
        print(f"{words:<40} in-domain "
              f"{score(fitted.model, fitted.eval_items):.3f}  "
              f"val loss {fitted.val_loss:.4f}  "
              f"params {fitted.params:,}")
    return 0


def demo(seed: int = 0, steps: int = STEPS) -> int:
    _show_addition(TASKS["addition"], seed, steps)
    print()
    _show_recall(seed, steps)
    return 0


def study(out_path: str | None = None) -> int:
    payload = build_results()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if out_path:
        with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text + "\n")
        print(f"wrote {out_path}", file=sys.stderr)
    else:
        sys.stdout.write(text + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llamanat")
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="train one model and print its traces")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--steps", type=int, default=STEPS)
    a = sub.add_parser("alphabet", help="the same task in two answer alphabets")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--steps", type=int, default=CHECKPOINTS[1])
    s = sub.add_parser("study", help="run the full ablation matrix")
    s.add_argument("--out", default=None)
    sub.add_parser("cache", help="exact KV-cache arithmetic")
    args = parser.parse_args(argv)
    if args.cmd == "demo":
        return demo(args.seed, args.steps)
    if args.cmd == "alphabet":
        return alphabet(args.seed, args.steps)
    if args.cmd == "study":
        return study(args.out)
    print(json.dumps(cache_report(LLAMA), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
