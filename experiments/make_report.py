"""Render the README results block directly from results/anatomy.json.

Every number and every superlative below is computed from the committed
artifact: run ``python experiments/run_study.py``, then
``python experiments/make_report.py --write``, and the README block between the
RESULTS markers is this output verbatim -- ``tests/test_readme.py``
byte-compares them. A rerun that changes the ordering changes the prose too,
instead of leaving behind a sentence that used to be true.

What that guarantees is that the block and the JSON cannot disagree, not that any
two runs of the study produce the same JSON. Near-threshold seeds move between
environments, which is why the tables publish the per-seed lists and this
renderer derives its sentences from them.

Why curves and not one snapshot: these tasks are learned by phase transition, so
a single fixed step records where each variant happens to sit relative to its own
acquisition threshold, not what it is capable of. Each row is therefore one
training trajectory scored at checkpoints.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from llamanat.blocks import gelu_hidden
from llamanat.positional import longest_wavelength
from llamanat.tasks import recall_length

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "anatomy.json"
README = ROOT / "README.md"

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"

#: accuracy gap below which two variants are reported as "not separated"
TOL = 0.01

#: seed standard deviation above which a mean no longer describes the runs
SPREAD = 0.2


def _separates(gap: float, spread: float) -> bool:
    """Whether an accuracy gap is large enough to call, given the noise beside it."""
    return gap > TOL and gap > spread


def _ordered(cells: dict, notes: dict) -> list[str]:
    """Reference model first, controls after it.

    Derived from the variant notes -- the row with no overrides is the reference
    -- rather than from a list of names that could outlive the matrix.
    """
    return sorted(cells, key=lambda v: (bool(notes[v]["overrides"]), v))


def _cell(summary: dict) -> str:
    mean, std = summary["mean"], summary["std"]
    return f"{mean:.3f}" if not std else f"{mean:.3f} ± {std:.3f}"


def _curve_table(cells: dict, notes: dict, checkpoints: list[int]) -> str:
    header = "| variant | params | " + " | ".join(str(s) for s in checkpoints) + " |"
    lines = [header, "|---|---:|" + "---:|" * len(checkpoints)]
    for variant in _ordered(cells, notes):
        curve = cells[variant]["curve"]
        accs = " | ".join(_cell(curve[str(s)]) for s in checkpoints)
        lines.append(f"| `{variant}` | {cells[variant]['params']:,} | {accs} |")
    return "\n".join(lines)


def _tie_group(cells: dict, last: str) -> list[str]:
    """Variants within TOL of the leader at the last checkpoint."""
    top = max(c["curve"][last]["mean"] for c in cells.values())
    return sorted(v for v in cells if cells[v]["curve"][last]["mean"] >= top - TOL)


def _ranking(cells: dict, last: str) -> tuple[str, float, float]:
    """Leader, its score, and the nearest other score at one checkpoint."""
    ranked = sorted(cells, key=lambda v: (-cells[v]["curve"][last]["mean"], v))
    top = cells[ranked[0]]["curve"][last]["mean"]
    second = cells[ranked[1]]["curve"][last]["mean"] if len(ranked) > 1 else top
    return ranked[0], top, second


def _crossings(cells: dict, checkpoints: list[int]) -> list[tuple[str, str]]:
    """Pairs whose order at the first checkpoint is reversed at the last."""
    first, last = str(checkpoints[0]), str(checkpoints[-1])
    early = {v: cells[v]["curve"][first]["mean"] for v in cells}
    late = {v: cells[v]["curve"][last]["mean"] for v in cells}
    names = sorted(cells)
    return [
        (x, y)
        for i, x in enumerate(names)
        for y in names[i + 1:]
        if (early[x] - early[y]) * (late[x] - late[y]) < 0
    ]


def _spread_note(cells: dict, last: str) -> str:
    """Rows whose seeds did not agree, spelled out instead of averaged away.

    On a task learned by transition a mean over three seeds can sit between two
    outcomes that neither seed produced, so the per-seed list is printed.
    """
    loose = sorted(
        (v for v in cells if cells[v]["curve"][last]["std"] > SPREAD),
        key=lambda v: -cells[v]["curve"][last]["std"],
    )
    if not loose:
        return (
            f"No row is split across its seeds at {last} steps: every standard "
            "deviation is small enough that the mean describes all three runs."
        )
    listed = "; ".join(
        f"`{v}` {_numbers(cells[v]['curve'][last]['seeds'])}" for v in loose
    )
    return (
        f"{len(loose)} of {len(cells)} rows are not a mean of three similar runs "
        f"at {last} steps -- {listed}. On a task learned by transition a seed "
        "either crosses the threshold or it does not, so those means describe "
        "which seeds finished rather than what the variant is capable of, and "
        "the spread has to be read with the number."
    )


def _numbers(values: list[float]) -> str:
    return "[" + ", ".join(f"{v:.3f}" for v in sorted(values)) + "]"


def _acquisition_note(cells: dict, checkpoints: list[int], reading: str) -> str:
    """What the last column says, and whether the curve says something else."""
    last = str(checkpoints[-1])
    leader, top, second = _ranking(cells, last)
    group = _tie_group(cells, last)
    parts = []
    if top - second <= TOL:
        parts.append(
            f"At {last} steps the top of the table is not separated: "
            f"{_names(group)} all land within {TOL:.2f} of {top:.3f}"
        )
    else:
        parts.append(
            f"`{leader}` leads at {last} steps on {top:.3f}, "
            f"{top - second:+.3f} clear of the nearest row"
        )
    crossed = _crossings(cells, checkpoints)
    if crossed:
        pairs = ", ".join(f"`{x}` / `{y}`" for x, y in crossed[:3])
        n_pairs = len(cells) * (len(cells) - 1) // 2
        moved = (
            f"The order on {pairs} at {checkpoints[0]} steps is not the order at "
            f"{last}"
            if len(crossed) <= 3
            else f"{len(crossed)} of the {n_pairs} pairings rank differently at "
            f"{checkpoints[0]} steps than at {last}"
        )
        parts.append(
            moved + " -- at this budget the rows are ranked by *when* the task "
            "arrives as much as by whether it can"
        )
    note = ". ".join(parts) + ". " + _spread_note(cells, last) + " "
    if (flat := _flat_note(cells, checkpoints)):
        note += flat + " "
    return note + reading


def _names(values: list[str]) -> str:
    return ", ".join(f"`{v}`" for v in values)


def _changed(note: dict) -> str:
    overrides = note["overrides"]
    if not overrides:
        return "the reference model"
    return "; ".join(
        f"`{key}` {overrides[key]} (instead of {note['replaces'][key]})"
        for key in sorted(overrides)
    )


def _in_domain_relation(rope_in: float, absolute_in: float, out_lead: str) -> str:
    """How the in-domain scores relate, given who wins out of length.

    The two in-domain numbers are printed by the caller, so only the relation is
    spelled out here.
    """
    if abs(rope_in - absolute_in) <= TOL:
        return "in domain the two are level"
    leader = "llama" if rope_in > absolute_in else "abs_pos"
    if leader == out_lead:
        return ("the same model also leads in domain, so transfer is not "
                "rescuing a weaker fit")
    return (f"it reverses in domain, where `{leader}` is the easier of the two "
            "to fit")


def _transfer(rope: dict, absolute: dict, length: str,
              rope_in: float, absolute_in: float) -> tuple[str, str]:
    """State the out-of-length finding, and what it means, from the numbers."""
    r, a = rope[length]["mean"], absolute[length]["mean"]
    if abs(r - a) <= TOL:
        return (
            f"they are not separated at {length} pairs ({r - a:+.3f})",
            "No transfer difference showed up at this scale, so the in-domain "
            "tables above are the whole result and the rotary choice is free "
            "here.",
        )
    lead, trail = ("llama", "abs_pos") if r > a else ("abs_pos", "llama")
    inside = _in_domain_relation(rope_in, absolute_in, lead)
    return (
        f"`{lead}` holds {max(r, a):.3f} while `{trail}` falls to {min(r, a):.3f}",
        f"A rotary block has nothing to memorise about absolute index, so "
        "relative distance is all it ever learns; a position table is the "
        "cheapest place in the model to hide a positional shortcut, and running "
        f"past the trained length is where it gets taken away. Note that this is "
        f"a *transfer* result and {inside}.",
    )


def _theta_table(sweep: dict, checkpoints: list[int], head_dim: int) -> str:
    rows = sweep["rows"]
    lengths = sorted(rows[0]["out_of_length"], key=int)
    lines = [
        "| RoPE base | slowest period (positions) | parameters | "
        + " | ".join(str(s) for s in checkpoints) + " | "
        + " | ".join(f"{n} pairs" for n in lengths) + " |",
        "|---|---:|---:|" + "---:|" * (len(checkpoints) + len(lengths)),
    ]
    for row in rows:
        theta = row["rope_theta"]
        reach = longest_wavelength(head_dim, theta)
        lines.append(
            f"| {_theta(theta)} | {reach:,.0f} | {row['params']:,} | "
            + " | ".join(_cell(row["curve"][str(s)]) for s in checkpoints)
            + " | "
            + " | ".join(_cell(row["out_of_length"][n]) for n in lengths) + " |")
    return "\n".join(lines)


def _theta(theta: float) -> str:
    return f"{theta:,.0f}"


def _ranked(source: dict) -> list[tuple[float, dict]]:
    """Bases ordered by mean accuracy, ties broken toward the smaller base."""
    return sorted(source.items(), key=lambda kv: (-kv[1]["mean"], kv[0]))


def _paired(row: dict, other: dict) -> list[float]:
    """Per-seed differences against another row of the same sweep.

    The swept fits share their seed, so they share their initialisation and their
    data stream and differ only in the angles RoPE rotates them through. That
    makes the paired list the strongest evidence available here, and a pooled
    standard deviation a weaker restatement of it.
    """
    return [a - b for a, b in zip(row["seeds"], other["seeds"], strict=True)]


def _moved(deltas: list[float]) -> int:
    """How many of the paired differences point the claimed way past the noise."""
    return sum(1 for d in deltas if d > TOL)


def _decided(row: dict, other: dict) -> tuple[float, bool]:
    """The mean gap against a row, and whether that gap is callable as a lead.

    A lead needs both the pooled criterion the rest of the README uses -- the gap
    has to clear the larger seed spread -- and unanimity across the paired seeds,
    so a single lucky seed cannot carry a sentence.
    """
    gap = row["mean"] - other["mean"]
    deltas = _paired(row, other)
    return gap, _separates(gap, max(row["std"], other["std"])) and (
        _moved(deltas) == len(deltas))


def _theta_verdict(name: str, ranked: list[tuple[float, dict]],
                   base_theta: float) -> str:
    """One column of the sweep, stated as a relation to the paper's base.

    A lead is claimed only when it clears the seed spread the table prints beside
    it *and* every paired seed moves the same way, in both directions -- including
    when the paper's own base is the row on top.
    """
    n = len(ranked[0][1]["seeds"])
    top_theta, top = ranked[0]
    ref = next(row for theta, row in ranked if theta == base_theta)
    if top_theta == base_theta:
        rival_theta, rival = ranked[1]
        margin, called = _decided(top, rival)
        if called:
            return (f"{name} the paper's base leads on {top['mean']:.3f}, "
                    f"{margin:+.3f} clear of {_theta(rival_theta)} at "
                    f"{rival['mean']:.3f} on every seed")
        return (f"{name} the paper's base is the highest of the row at "
                f"{top['mean']:.3f}, but its {margin:+.3f} over "
                f"{_theta(rival_theta)} is inside the noise beside it "
                f"({_moved(_paired(top, rival))} of {n} seeds), so the two are "
                "not separated")
    gap, called = _decided(top, ref)
    if gap <= TOL:
        return (f"{name} the reference base is within {TOL:.2f} of the best "
                f"({ref['mean']:.3f} against {top['mean']:.3f}), so the two are "
                "not separated")
    if not called:
        return (f"{name} {_theta(top_theta)} prints the best number "
                f"({top['mean']:.3f} against the paper's {ref['mean']:.3f}), but "
                f"the {gap:+.3f} gap is inside the "
                f"{max(top['std'], ref['std']):.3f} seed spread beside it and only "
                f"{_moved(_paired(top, ref))} of {n} paired seeds point that way, "
                "so it is not a lead")
    return (f"{name} {_theta(top_theta)} leads on {top['mean']:.3f} while the "
            f"paper's {_theta(base_theta)} sits at {ref['mean']:.3f} -- a gap of "
            f"{gap:+.3f} outside the spread beside it, on all {n} seeds")


def _theta_note(sweep: dict, last: str, head_dim: int,
                base_theta: float) -> str:
    """What the rotation base decides at a window this short, from the numbers."""
    rows = sweep["rows"]
    lengths = sorted(rows[0]["out_of_length"], key=int)
    longest = lengths[-1]
    inside = {r["rope_theta"]: r["curve"][last] for r in rows}
    outside = {r["rope_theta"]: r["out_of_length"][longest] for r in rows}
    rank_in = _ranked(inside)
    rank_out = _ranked(outside)
    best_in, best_out = rank_in[0], rank_out[0]
    tokens = recall_length(int(longest))
    reach = {r["rope_theta"]: longest_wavelength(head_dim, r["rope_theta"])
             for r in rows}
    sizes = sorted({r["params"] for r in rows})
    size_clause = (
        f"all {len(rows)} of them carry exactly {sizes[0]:,} parameters, because "
        "a rotation base adds none"
        if len(sizes) == 1
        else f"the rows are not even the same size "
             f"({', '.join(f'{s:,}' for s in sizes)})"
    )
    tight = sorted(theta for theta, value in reach.items() if value < tokens)
    tight_names = ", ".join(_theta(t) for t in tight)
    reach_clause = (
        f"{len(tight)} of the {len(rows)} bases complete a turn inside the "
        f"{tokens} tokens this table scores ({tight_names}), so their far columns "
        "are read through a wrapped pair as well as a longer sequence"
        if tight else
        f"every base here is still inside its first turn at {tokens} tokens, so "
        "what the sweep varies is how much of the window the slowest pair "
        "resolves, not whether the sequence wraps"
    )
    parts = [
        f"At {sweep['trained_on_pairs']} pairs trained and {last} steps the rows "
        f"differ in one float and nothing else: {size_clause}.",
        _theta_verdict("In domain,", rank_in, base_theta) + ".",
        _theta_verdict(f"At {longest} pairs,", rank_out, base_theta) + ".",
        f"The slowest channel pair turns once every {min(reach.values()):,.0f} "
        f"positions at {_theta(min(reach))} and {max(reach.values()):,.0f} at "
        f"{_theta(max(reach))} -- {reach_clause}.",
    ]
    if best_in[0] != best_out[0]:
        parts.append(
            f"The in-domain leader {_theta(best_in[0])} is not the leader out of "
            f"length ({_theta(best_out[0])}), which is the same shape as the "
            "position-table result above: fitting the trained length and "
            "generalising past it are two different questions.")
    parts.append(_theta_asymmetry(inside, base_theta, rank_in))
    means = [v["mean"] for v in inside.values()]
    move = max(means) - min(means)
    verdict = (
        "What it does show is that an axis with no parameters and no shapes at "
        f"all still moves the measured accuracy -- the {len(rows)} bases spread "
        f"{move:.3f} in domain -- which is the cleanest case in this repository "
        "for 'the geometry is doing something' rather than 'the extra weights "
        "are'."
        if move > TOL else
        f"The {len(rows)} bases span only {move:.3f} in domain, so at a window "
        "this short the geometry is free: nothing about the rotation is being "
        "paid for in parameters, and nothing is being bought either."
    )
    parts.append(
        "None of this argues for a different base in a real model -- that number "
        "is chosen for the context the model has to run, and a window of "
        f"{tokens} tokens is not the trade LLaMA was making. " + verdict)
    return " ".join(parts)


def _theta_asymmetry(inside: dict, base_theta: float,
                     ranked: list[tuple[float, dict]]) -> str:
    """Whether the base has a measurable ceiling, a measurable floor, or neither.

    The two ends of the sweep are not the same experiment, so each is reported on
    its own terms. The comparison is *paired*: two rows at the same seed share
    their initialisation and their data stream and differ only in the angles
    RoPE rotates them through, which makes "every seed moved this way" the
    available evidence and a pooled standard deviation the weaker version of it.
    """
    ref = inside[base_theta]
    top_theta, top = ranked[0]
    worst_theta, worst = ranked[-1]
    up = _paired(top, ref)
    down = _paired(ref, worst)
    behind = [-d for d in down]  # the same deltas from the worst row's side
    n = len(ref["seeds"])
    head = (
        "Read seed against seed rather than mean against mean -- the swept fits "
        "share their initialisation and their data stream, so two rows at the "
        "same seed differ only in the angles -- and "
    )
    if _moved(down) == n and _moved(up) < n:
        return head + (
            f"only one end of the sweep is decided: {_theta(worst_theta)} lands "
            f"below the paper's base on all {n} seeds ({_signed(behind)}), while the "
            f"top row here, {_theta(top_theta)}, is ahead of it on "
            f"{_moved(up)} of {n}. So this window measures a floor for the "
            "rotation base and not a ceiling: running short of range costs real "
            "accuracy, and buying more of it than the sequence needs is free."
        )
    if _moved(up) == n and _moved(down) < n:
        return head + (
            f"only the upper end is decided: {_theta(top_theta)} beats the "
            f"paper's base on all {n} seeds ({_signed(up)}), while "
            f"{_theta(worst_theta)} is behind on {_moved(down)} of {n} "
            f"({_signed(behind)})."
        )
    if _moved(up) == n and _moved(down) == n:
        return head + (
            f"both ends are decided: {_theta(top_theta)} is ahead of the paper's "
            f"base on all {n} seeds ({_signed(up)}) and {_theta(worst_theta)} "
            f"behind on all {n} ({_signed(behind)}), so at this window the base has "
            "a measurable ceiling as well as a measurable floor."
        )
    return head + (
        f"neither end of the sweep is decided: {_theta(top_theta)} is ahead of "
        f"the paper's base on {_moved(up)} of {n} seeds ({_signed(up)}) and "
        f"{_theta(worst_theta)} trails it on {_moved(down)} of {n} "
        f"({_signed(behind)})."
    )


def _signed(deltas: list[float]) -> str:
    return ", ".join(f"{d:+.3f}" for d in deltas)


def _row_cells(row: dict) -> tuple[str, str]:
    alphabet = ("copy the value token out of the context"
                if row["mode"] == "copy" else "name the value")
    extra = "".join(f", {k}={v}" for k, v in sorted(row["config_overrides"].items()))
    return alphabet, f"`{row['variant']}`{extra}"


def _alphabet_table(alpha: dict, checkpoints: list[int]) -> str:
    lines = ["| answer alphabet | model | parameters | "
             + " | ".join(str(s) for s in checkpoints)
             + f" | {alpha['probe_pairs']} pairs |",
             "|---|---|---:|" + "---:|" * (len(checkpoints) + 1)]
    for row in alpha["rows"]:
        alphabet, model = _row_cells(row)
        lines.append(
            f"| {alphabet} | {model} | {row['params']:,} | "
            + " | ".join(_cell(row['curve'][str(s)]) for s in checkpoints)
            + f" | {row['probe']['mean']:.3f} |")
    return "\n".join(lines)


def _find(rows: list[dict], mode: str, variant: str, **over: object) -> dict:
    for row in rows:
        same = row["mode"] == mode and row["variant"] == variant
        if same and all(row["config_overrides"].get(k) == v
                        for k, v in over.items()):
            return row
    raise KeyError(f"no alphabet row for {mode}/{variant}/{over}")


def _alphabet_note(alpha: dict, last: str) -> str:
    rows = alpha["rows"]
    named = _find(rows, "name", "llama")
    copied = _find(rows, "copy", "llama")
    untied = _find(rows, "copy", "llama", tie_embeddings=False)
    gap = named["curve"][last]["mean"] - copied["curve"][last]["mean"]
    rescue = untied["curve"][last]["mean"] - copied["curve"][last]["mean"]
    shared = (
        f"Same sequences, same architecture, same budget; only the alphabet the "
        f"answer is written in changes. Naming the value reaches "
        f"{named['curve'][last]['mean']:.3f} in domain and "
        f"{named['probe']['mean']:.3f} at {alpha['probe_pairs']} pairs, while "
        f"copying the value token back out of the context reaches "
        f"{copied['curve'][last]['mean']:.3f} and {copied['probe']['mean']:.3f}."
    )
    if gap > TOL:
        readout = (
            " The lookup those two share is identical, so the difference is "
            "confined to the readout, "
        )
        if abs(rescue) <= TOL:
            readout += (
                "and the obvious explanation for it -- an output layer that has "
                "to double as the input table -- does not account for the size of "
                f"it: untying the embeddings moves copy by {rescue:+.3f}. This "
                "repository therefore reports the constraint and not a mechanism "
                "for it."
            )
        elif rescue > 0:
            readout += (
                "and the shared embedding table is part of the cost: untying it, "
                f"so the output layer stops doubling as the input table, recovers "
                f"{rescue:+.3f} of the {gap:.3f} gap."
            )
        else:
            readout += (
                f"and untying the shared embedding table moves it the wrong way "
                f"({rescue:+.3f}), so the shared table is not the bottleneck and "
                "no mechanism for the wall is claimed here."
            )
        return shared + readout
    if -gap > TOL:
        return shared + (
            " Copying is the easier of the two here, so the recall tables above "
            f"use the harder alphabet; untying the embeddings moves it by "
            f"{rescue:+.3f}."
        )
    return shared + (
        " At this budget the two alphabets are not separated, so the readout is "
        f"not what limits the task -- untying the embeddings moves copy by "
        f"{rescue:+.3f}."
    )


def _flat_note(cells: dict, checkpoints: list[int]) -> str:
    """Rows that never left the floor: a flat curve is not a slow transition."""
    first, last = str(checkpoints[0]), str(checkpoints[-1])
    top = max(c["curve"][last]["mean"] for c in cells.values())
    stuck = [
        v for v in sorted(cells)
        if abs(cells[v]["curve"][last]["mean"]
             - cells[v]["curve"][first]["mean"]) <= TOL
        and cells[v]["curve"][last]["mean"] < top - TOL
    ]
    if not stuck:
        return ""
    listed = "; ".join(
        f"`{v}` {cells[v]['curve'][first]['mean']:.3f} at {checkpoints[0]} steps, "
        f"{cells[v]['curve'][last]['mean']:.3f} at {last}"
        for v in stuck
    )
    return (
        f"{len(stuck)} row{'s' if len(stuck) > 1 else ''} never left the floor of "
        f"this table -- {listed}. The first checkpoint and the last print the same "
        "number, so that is a flat curve rather than a slow one, and the honest "
        "reading is that the variant does not do this task at this budget."
    )


KV_BY_HEADS = ("mha", "llama", "mqa")  # 4, 2 (reference), 1 KV heads


def _cache_learning_note(in_domain: dict, last: str) -> str:
    """Whether the axis this table prices also moves the measured accuracy."""
    ranks = {
        task: sorted(KV_BY_HEADS,
                     key=lambda v: (-cells[v]["curve"][last]["mean"], v))
        for task, cells in in_domain.items()
    }
    ordered = {
        task: " > ".join(f"`{v}` ({in_domain[task][v]['curve'][last]['mean']:.3f})"
                         for v in order)
        for task, order in ranks.items()
    }
    joined = "; ".join(f"{task} runs {ordered[task]}" for task in sorted(ordered))
    if len({tuple(order) for order in ranks.values()}) == 1:
        return (
            f"The three rows that differ only in KV heads rank the same way on "
            f"every task at {last} steps ({joined}), so here the cache saving and "
            "the accuracy move together."
        )
    return (
        "What this table prices is the cache. The accuracy side of the very same "
        f"axis is in the tables above, and at this budget it does not order "
        f"itself the same way twice: {joined}. Three seeds at one step budget is "
        "not enough to say why, and no reason is offered here."
    )


def _alphabet_verdict(alpha: dict, last: str) -> str:
    """Whether the answer alphabet decides what fits -- stated from the numbers."""
    rows = alpha["rows"]
    named = _find(rows, "name", "llama")["curve"][last]["mean"]
    copied = _find(rows, "copy", "llama")["curve"][last]["mean"]
    gap = named - copied
    if gap > TOL:
        return (
            "the formulation nobody would question -- emit the token you were "
            "shown -- is the one a block this size cannot fit, while naming the "
            f"value with a private symbol reaches {named:.3f}."
        )
    if -gap > TOL:
        return (
            "copying the value token fits more easily here than naming it, so the "
            "recall tables above use the harder of the two alphabets."
        )
    return (
        "the two alphabets are not separated at this budget, so either "
        "formulation would have produced the recall tables above."
    )


def build(data: dict) -> str:
    settings = data["settings"]
    checkpoints = settings["checkpoints"]
    last = str(checkpoints[-1])
    seeds = settings["seeds"]
    in_domain = data["in_domain"]
    extra = data["extrapolation"]
    alpha = data["answer_alphabet"]
    sweep = data["theta_sweep"]
    kv = data["kv_cache"]
    notes = data["variants"]
    base = settings["base_config"]
    head_dim = base["d_model"] // base["n_head"]
    base_theta = base["rope_theta"]

    add_cells = in_domain["addition"]
    rec_cells = in_domain["recall"]

    lengths = sorted(extra["llama"]["curves"], key=int)
    trained = str(extra["llama"]["trained_on_pairs"])
    rope = extra["llama"]["curves"]
    absolute = extra["abs_pos"]["curves"]
    happened, meaning = _transfer(rope, absolute, lengths[-1],
                                  rope[trained]["mean"], absolute[trained]["mean"])

    swiglu_p = add_cells["llama"]["params"]
    gelu_p = add_cells["gelu_mlp"]["params"]
    kv_rows = kv["variants"]
    gqa, mqa = kv_rows["gqa"], kv_rows["mqa"]

    out: list[str] = []
    env = data["environment"]
    seeds_word = "seed" if len(seeds) == 1 else "seeds"
    out.append(
        f"*{len(seeds)} {seeds_word} "
        f"({', '.join(map(str, seeds))}); one training trajectory per row, "
        f"scored on held-out examples at checkpoints "
        f"{checkpoints}; batch {settings['batch']}, lr {settings['lr']}, "
        f"{settings['steps']} steps. Produced on {env['device']} by "
        "`experiments/run_study.py`"
        f" under python {env['python']} / torch {env['torch']} / "
        f"{env['threads']} threads ({env['platform']}), committed as "
        "[`results/anatomy.json`](results/anatomy.json), and rendered by "
        "`experiments/make_report.py` -- a test fails if this block and that file "
        "disagree, and only a rerun *in that environment* is expected to "
        "reproduce the numbers bit for bit.*"
    )
    out.append("")
    out.append(
        f"Reference model: d_model {base['d_model']}, {base['n_layer']} layers, "
        f"{base['n_head']} query heads sharing {base['n_kv_head']} KV heads "
        f"(head_dim {head_dim}), ff {base['ff']}, "
        f"{base['norm']} norm, "
        f"{'RoPE' if base['rope'] else 'no rotary'} -- "
        f"**{data['n_params_llama']:,} parameters** trained on "
        f"{settings['train_items']['addition']:,} examples."
    )
    out.append("")

    out.append("### What each row replaces")
    out.append("")
    out.append("| variant | change against the reference | parameters |")
    out.append("|---|---|---:|")
    for variant in _ordered(notes, notes):
        out.append(f"| `{variant}` | {_changed(notes[variant])} | "
                   f"{add_cells[variant]['params']:,} |")
    out.append("")
    size_note = (
        "the same size to the parameter"
        if swiglu_p == gelu_p
        else f"{gelu_p - swiglu_p:+,} parameters apart"
    )
    pos_rows = base["max_len"] * base["d_model"]
    pos_note = add_cells["abs_pos"]["params"] - swiglu_p
    pos_clause = (
        f"exactly its position table ({pos_note:,} parameters = "
        f"{base['max_len']} rows x {base['d_model']})"
        if pos_note == pos_rows
        else f"{pos_note:+,} parameters, of which the {pos_rows:,} in its "
        f"{base['max_len']}-row position table is the intended part"
    )
    ratio = gelu_hidden(base["ff"]) / base["ff"]
    out.append(
        "The control that makes the `gelu_mlp` row mean anything: it is "
        f"{size_note} -- {gelu_p:,} against SwiGLU's {swiglu_p:,} -- because a "
        "two-matrix gelu network matches three-matrix SwiGLU at "
        f"`hidden = {ratio:.1f} * ff`. That row therefore measures *gating*, not "
        f"size. `abs_pos` is larger by {pos_clause}, and `mha` / `mqa` differ "
        "only in how many KV heads the projections write.")
    out.append("")

    out.append("### In-domain acquisition")
    out.append("")
    out.append(
        f"**addition** -- {settings['train_items']['addition']:,} three-digit sums "
        "written as a least-significant-column-first carry chain; graded by the "
        "exact sum after free-running generation, no teacher forcing, on "
        f"{settings['eval_items']['addition']} unseen pairs.")
    out.append("")
    out.append(_curve_table(add_cells, notes, checkpoints))
    out.append("")
    add_leader, _, _ = _ranking(add_cells, last)
    add_reading = (
        "Read this column as acquisition at one fixed length, not as capability: "
        "a task whose length never changes lets a position table memorise which "
        "slot owns which column, and that is the shortcut the next table takes "
        "away."
        if add_leader == "abs_pos"
        else "At this budget no positional shortcut showed up as a speed "
        "advantage on the chain, so the fixed-length column and the transfer "
        "column have to be read together."
    )
    out.append(_acquisition_note(add_cells, checkpoints, add_reading))
    out.append("")
    out.append(
        f"**recall** -- {trained} key/value pairs then a probe key; graded by "
        f"whether the emitted token is the value that pair actually held, on "
        f"{settings['eval_items']['recall']} unseen sequences. The answer is "
        f"named with a private symbol (`{settings['recall_answer_mode']}` mode); "
        "the section below measures what the other formulation costs.")
    out.append("")
    out.append(_curve_table(rec_cells, notes, checkpoints))
    out.append("")
    out.append(_acquisition_note(
        rec_cells, checkpoints,
        "That is why the curve is reported rather than a single number: on a task "
        "with a threshold, one snapshot records where each row happens to sit "
        "relative to its own transition, not what it is capable of."))
    out.append("")

    out.append("### Train short, test long")
    out.append("")
    out.append(
        f"The same fitted weights, scored on sequences longer than the {trained} "
        "pairs they trained on, in the same answer alphabet as the table above. "
        "Nothing was retrained or retuned between columns. "
        f"The absolute table is sized to {base['max_len']} positions, so rows "
        "outside the trained range exist and were simply never updated -- that is "
        "the comparison, not a handicap.")
    out.append("")
    out.append("| variant | " + " | ".join(
        f"{n} pairs" + (" *(trained)*" if n == trained else "") for n in lengths)
        + " |")
    out.append("|---|" + "---:|" * len(lengths))
    for variant in _ordered(extra, notes):
        curve = extra[variant]["curves"]
        out.append(f"| `{variant}` | "
                   + " | ".join(_cell(curve[n]) for n in lengths) + " |")
    out.append("")
    out.append(
        f"At the trained length the two score {rope[trained]['mean']:.3f} / "
        f"{absolute[trained]['mean']:.3f}, and out of length {happened}. "
        + meaning)
    out.append("")

    out.append("### How fast the rotation runs")
    out.append("")
    out.append(
        "RoPE arrives with one number attached to it, the base of the frequency "
        f"series, and the reference model runs at {_theta(base_theta)}. It is the "
        "only axis in this repository that costs nothing: no parameter, no bias, "
        "no change of shape, just a different set of angles on the same weights. "
        "So the sweep runs in both directions from it, on the recall task at the "
        "matrix's own budget. The second column is arithmetic rather than "
        "measurement: how far two tokens can sit apart before the *slowest* "
        "channel pair completes a turn, past which that pair reads two offsets "
        "as one angle.")
    out.append("")
    out.append(_theta_table(sweep, checkpoints, head_dim))
    out.append("")
    out.append(_theta_note(sweep, last, head_dim, base_theta))
    out.append("")

    out.append("### How the answer is asked")
    out.append("")
    out.append(
        f"One seed ({', '.join(map(str, alpha['seeds']))}), "
        f"{alpha['trained_on_pairs']} pairs trained, the same "
        f"{settings['steps']}-step budget and the same checkpoints as the tables "
        "above. This is a control on the task rather than on the model, and it is "
        "the reason the recall rows above are worth reading: "
        + _alphabet_verdict(alpha, last)
    )
    out.append("")
    out.append(_alphabet_table(alpha, checkpoints))
    out.append("")
    out.append(_alphabet_note(alpha, last))
    out.append("")

    out.append("### Where the KV cache goes")
    out.append("")
    out.append(
        "Arithmetic rather than measurement, so it carries no seed noise: "
        f"{base['n_layer']} layers, batch 1, {kv['seq_len']:,} positions, "
        f"{kv['precision']}, K and V both stored.")
    out.append("")
    out.append("| attention | KV heads | cache per sequence | share of MHA | "
               "KV projection parameters |")
    out.append("|---|---:|---:|---:|---:|")
    label = {"mha": "MHA", "gqa": "GQA *(reference)*", "mqa": "MQA"}
    for key in ("mha", "gqa", "mqa"):
        row = kv_rows[key]
        out.append(f"| {label[key]} | {row['n_kv_head']} | "
                   f"{row['cache_kib']:,.1f} KiB | {row['share_of_mha']:.2f} | "
                   f"{row['kv_projection_params']:,} |")
    out.append("")
    out.append(
        f"Half the KV heads costs {gqa['kv_projection_params']:,} projection "
        f"parameters -- {gqa['kv_projection_params'] / swiglu_p:.1%} of this model "
        f"-- and buys {(1 - gqa['share_of_mha']):.0%} less cache; MQA buys "
        f"{(1 - mqa['share_of_mha']):.0%}. Cache grows with sequence length and "
        "KV-head count while the weights "
        "stay put, so sharing KV heads becomes worth more the longer the context "
        "is -- and it is the one axis here where the parameter budget genuinely "
        "moves.")
    out.append("")
    out.append(_cache_learning_note(in_domain, last))
    return "\n".join(out)


def splice(readme: str, block: str) -> str:
    """Replace what sits between the RESULTS markers, leaving the prose alone."""
    head, start, rest = readme.partition(START)
    _, end, tail = rest.partition(END)
    if not start or not end:
        raise ValueError("README.md lost its RESULTS markers")
    return f"{head}{START}\n{block}\n{END}{tail}"


def main(argv: list[str] | None = None) -> None:
    """Print the block, or splice it into README.md with ``--write``."""
    write = "--write" in (sys.argv[1:] if argv is None else argv)
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    block = build(data)
    if not write:
        print(block)
        return
    text = README.read_text(encoding="utf-8")
    README.write_text(splice(text, block), encoding="utf-8")
    print(f"spliced {len(block.splitlines())} rendered lines into {README}")


if __name__ == "__main__":
    main()
