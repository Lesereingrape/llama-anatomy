"""Command line: a demo you can read, and the entry point that writes the JSON.

    llamanat demo               # trains one small model, prints its traces
    llamanat alphabet           # the same task, two answer alphabets
    llamanat study              # runs the full matrix, prints the JSON
    llamanat cache              # the exact KV-cache arithmetic, no training
"""

from __future__ import annotations

import argparse
import json
import sys

from .kv import cache_report
from .model import LLAMA
from .study import build_results
from .tasks import ANSWER_MODES, DIGIT0, KEY0, token_digit
from .train import TaskSpec, fit, generate_addition, probe, score


def _show_addition(spec: TaskSpec, seed: int, steps: int) -> None:
    fitted = fit(spec, "llama", seed, steps=steps, lr=1e-3)
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
    return TaskSpec("recall", "recall", n_pairs=4, n_train=1000, n_eval=64,
                    answer=mode)


def _show_recall(seed: int, steps: int) -> None:
    spec = _recall_spec()
    fitted = fit(spec, "llama", seed, steps=steps, lr=1e-3)
    print(f"llama block, recall trained on 4 pairs -> in-domain "
          f"{score(fitted.model, fitted.eval_items):.3f}")
    for pairs in (4, 8, 12):
        value = (score(fitted.model, fitted.eval_items) if pairs == 4
                 else probe(fitted, pairs, spec))
        print(f"  probed at {pairs:>2} pairs: {value:.3f}")


def alphabet(seed: int = 0, steps: int = 600) -> int:
    """Train the same four pairs twice, once per answer alphabet.

    The shortest version of the result the README's "How the answer is asked"
    section reports: same task, same model, same budget, and only the symbol the
    answer is written in decides whether it gets learned at all.
    """
    for mode in ANSWER_MODES:
        spec = _recall_spec(mode)
        fitted = fit(spec, "llama", seed, steps=steps, lr=1e-3)
        words = ("name the value with a private symbol" if mode == "name"
                 else "copy the value token out of the context")
        print(f"{words:<40} in-domain "
              f"{score(fitted.model, fitted.eval_items):.3f}  "
              f"val loss {fitted.val_loss:.4f}  "
              f"params {fitted.params:,}")
    return 0


def demo(seed: int = 0, steps: int = 400) -> int:
    _show_addition(
        TaskSpec("addition", "addition", width=3, n_train=1000, n_eval=64),
        seed, steps)
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
    d.add_argument("--steps", type=int, default=400)
    a = sub.add_parser("alphabet", help="the same task in two answer alphabets")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--steps", type=int, default=600)
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
