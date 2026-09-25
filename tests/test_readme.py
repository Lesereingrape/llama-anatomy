"""The published numbers must be the committed numbers.

Three separate failures are being guarded against: a README that drifted from
`results/anatomy.json`, a `results/anatomy.json` that drifted from the study
plan in `llamanat.study`, and a rendered block that silently lost a variant.
"""

from __future__ import annotations

import inspect
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import make_report  # noqa: E402

from llamanat.study import (  # noqa: E402
    CHECKPOINTS,
    SEEDS,
    STEPS,
    TASKS,
    THETA_SWEEP,
    TRAIN_PAIRS,
    VARIANT_ORDER,
)

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


@pytest.fixture(scope="module")
def data() -> dict:
    path = ROOT / "results" / "anatomy.json"
    if not path.exists():  # pragma: no cover - the artifact is committed
        pytest.fail("results/anatomy.json is missing; run experiments/run_study.py")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def _block(readme: str) -> str:
    assert START in readme and END in readme, "the README lost its RESULTS markers"
    body = readme.split(START, 1)[1].split(END, 1)[0]
    return body.strip("\n")


def test_results_block_is_exactly_the_rendered_artifact(data, readme):
    assert _block(readme) == make_report.build(data)


def test_no_marker_text_leaks_into_the_rendered_block(data, readme):
    body = _block(readme)
    assert "RESULTS:" not in body
    assert "<!--" not in body


def test_the_renderer_can_write_the_block_it_prints(data, readme):
    """``--write`` is the documented way to re-splice, so it must be idempotent.

    Splicing has to replace exactly the generated region and nothing else, or the
    command the README tells people to run would eat the prose around the tables.
    """
    block = make_report.build(data)
    once = make_report.splice(readme, block)
    assert _block(once) == block
    assert make_report.splice(once, block) == once
    head, _, rest = once.partition(make_report.START)
    tail = rest.split(make_report.END, 1)[1]
    assert head == readme.split(make_report.START, 1)[0]
    assert tail == readme.split(make_report.END, 1)[1]


def test_the_artifact_came_from_the_committed_plan(data):
    settings = data["settings"]
    assert settings["seeds"] == list(SEEDS)
    assert settings["steps"] == STEPS
    assert settings["checkpoints"] == list(CHECKPOINTS)
    assert sorted(data["in_domain"]) == sorted(TASKS)


def test_the_artifact_names_the_environment_it_reproduces_in(data):
    """Bit-exact reruns are a claim about one environment, so the block says which.

    ``tests`` cannot re-fit a nine-variant matrix to check that, and the README
    must not imply that any machine reproduces these seeds. What is pinned here
    is that the artifact records its own origin and the renderer quotes that
    record rather than whatever interpreter happens to be running it.
    """
    env = data["environment"]
    assert sorted(env) == ["device", "platform", "python", "threads", "torch"]
    assert env["device"] == "cpu"
    assert isinstance(env["threads"], int) and env["threads"] >= 1
    block = make_report.build(data)
    assert f"torch {env['torch']}" in block
    assert f"{env['threads']} threads" in block
    assert "Produced on cpu" in block


def test_every_variant_and_checkpoint_is_published(data):
    for task, cells in data["in_domain"].items():
        assert sorted(cells) == sorted(VARIANT_ORDER), task
        for variant, cell in cells.items():
            assert sorted(int(s) for s in cell["curve"]) == list(CHECKPOINTS), (
                f"{task}/{variant}")
            for step in cell["curve"].values():
                assert len(step["seeds"]) == len(SEEDS)
                assert 0.0 <= step["mean"] <= 1.0


def test_extrapolation_rows_start_at_the_trained_length(data):
    extra = data["extrapolation"]
    assert sorted(extra) == ["abs_pos", "llama"]
    trained = str(extra["llama"]["trained_on_pairs"])
    for row in extra.values():
        lengths = sorted(int(n) for n in row["curves"])
        assert lengths[0] == int(trained)
        assert lengths == sorted(set(lengths))
        assert len(row["curves"][trained]["seeds"]) == len(SEEDS)


def _published_accuracies(data: dict) -> set[str]:
    """Every 3-decimal accuracy the artifact contains, however it is laid out."""
    out: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"mean", "std", "seeds"} and isinstance(value, list):
                    out.update(f"{v:.3f}" for v in value)
                elif isinstance(value, float) and 0.0 <= value <= 1.0:
                    out.add(f"{value:.3f}")
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk({k: data[k] for k in
          ("in_domain", "extrapolation", "answer_alphabet", "theta_sweep")})
    return out


def _prose(readme: str) -> str:
    """Everything the human wrote, i.e. outside the generated block."""
    return readme.split(START, 1)[0] + readme.split(END, 1)[1]


def test_the_prose_quotes_only_numbers_from_the_artifact(data, readme):
    decimals = set(re.findall(r"\b0\.\d{3}\b", _prose(readme)))
    allowed = _published_accuracies(data)
    assert decimals <= allowed, f"uncited numbers: {sorted(decimals - allowed)}"


def test_the_renderer_hardcodes_no_measurement():
    """`build()` is a function of the JSON, with no measurement of its own.

    A decimal literal in the renderer is a reporting threshold or a leaked
    number, so the only place one may appear is a module-level constant that is
    named like one -- ``TOL`` and ``SPREAD`` -- and everything the prose says
    about ratios is recomputed from ``llamanat``.
    """
    src = (ROOT / "experiments" / "make_report.py").read_text(encoding="utf-8")
    offenders = [
        line.strip() for line in src.splitlines()
        if not line.lstrip().startswith("#")
        and re.search(r"(?<![\w.])\d\.\d+(?![\w.])", line)
        and not re.fullmatch(r"[A-Z][A-Z_]* = \d+\.\d+", line.strip())
    ]
    assert not offenders, sorted(offenders)
    assert (make_report.TOL, make_report.SPREAD) == (0.01, 0.2)


def test_the_theta_sweep_publishes_every_base_at_the_matrix_budget(data):
    sweep = data["theta_sweep"]
    assert [row["rope_theta"] for row in sweep["rows"]] == list(THETA_SWEEP)
    assert sweep["trained_on_pairs"] == TRAIN_PAIRS
    assert sweep["variant"] == "llama"
    base = data["settings"]["base_config"]["rope_theta"]
    assert base in [row["rope_theta"] for row in sweep["rows"]], (
        "the README states the reference base is one of the rows")
    for row in sweep["rows"]:
        assert row["params"] == data["n_params_llama"], (
            f"theta {row['rope_theta']} changed the parameter count, so the "
            "sweep is no longer a comparison of geometry alone")
        assert sorted(int(s) for s in row["curve"]) == list(CHECKPOINTS)
        assert row["final_accuracy"] == row["curve"][str(CHECKPOINTS[-1])]
        assert 0.0 <= row["final_accuracy"]["mean"] <= 1.0
        for step in row["curve"].values():
            assert len(step["seeds"]) == len(SEEDS)


def test_the_sweep_claims_only_the_asymmetry_its_seeds_support(data):
    """A floor without a ceiling is a paired claim, so it has to stay paired.

    The rotation section is allowed to say the base has a measurable floor and no
    measurable ceiling only while the worst base loses to the reference on every
    seed and the best fails to beat it on every seed. If a rerun made both ends
    decide, or neither, the renderer owes a different sentence -- and this test
    fails if it keeps the old one.
    """
    sweep = data["theta_sweep"]
    base = data["settings"]["base_config"]["rope_theta"]
    inside = {r["rope_theta"]: r["curve"][str(CHECKPOINTS[-1])]
              for r in sweep["rows"]}
    ref = inside[base]
    ranked = make_report._ranked(inside)
    down = make_report._paired(ref, ranked[-1][1])
    up = make_report._paired(ranked[0][1], ref)
    floor = all(d > make_report.TOL for d in down)
    ceiling = all(d > make_report.TOL for d in up)
    claimed = "measures a floor for the rotation base and not a ceiling"
    assert (claimed in make_report.build(data)) == (floor and not ceiling)


def test_the_reference_base_is_the_matrix_run_a_second_time(data):
    """One of the swept rows has to reproduce the tables above it exactly.

    The sweep re-fits the reference model on the same task, seed stream and
    budget, so its row is not a fourth opinion about ``llama`` -- it is the same
    three runs, and if the two disagreed the README would be publishing one
    question answered two ways.
    """
    base = data["settings"]["base_config"]["rope_theta"]
    row = next(r for r in data["theta_sweep"]["rows"] if r["rope_theta"] == base)
    matrix = data["in_domain"]["recall"]["llama"]["curve"]
    for step in map(str, CHECKPOINTS):
        assert row["curve"][step]["seeds"] == pytest.approx(
            matrix[step]["seeds"]), step
    longest = max(row["out_of_length"], key=int)
    assert row["out_of_length"][longest]["seeds"] == pytest.approx(
        data["extrapolation"]["llama"]["curves"][longest]["seeds"])


def test_the_alphabet_control_reuses_the_matrix_runs(data):
    """Several call sites re-fit the same model, so their cells must be identical.

    ``run_task``, ``run_extrapolation`` and ``run_answer_alphabet`` are separate
    code paths over the same seed and the same data stream (``run_theta_sweep``
    adds a fourth for the reference base, tested just above). If they disagreed
    the README would be publishing one question answered three different ways,
    and this is the test that would notice.
    """
    alphabet = data["answer_alphabet"]
    pairs = str(alphabet["probe_pairs"])
    for variant in ("llama", "abs_pos"):
        row = make_report._find(alphabet["rows"], "name", variant)
        for step in map(str, CHECKPOINTS):
            assert row["curve"][step]["mean"] == pytest.approx(
                data["in_domain"]["recall"][variant]["curve"][step]["seeds"][0]
            ), (variant, step)
        assert row["probe"]["mean"] == pytest.approx(
            data["extrapolation"][variant]["curves"][pairs]["seeds"][0]
        ), variant


def test_prose_does_not_claim_a_transfer_result_the_data_withholds(data, readme):
    body = _block(readme)
    rope = data["extrapolation"]["llama"]["curves"]
    absolute = data["extrapolation"]["abs_pos"]["curves"]
    longest = max(rope, key=int)
    gap = rope[longest]["mean"] - absolute[longest]["mean"]
    claims_transfer = "holds" in body and "falls to" in body
    assert claims_transfer == (abs(gap) > make_report.TOL), (
        "the sentence and the gap disagree")
    assert "Arithmetic rather than measurement" in body, \
        "the cache table must declare itself non-empirical"


def test_the_demo_commands_grade_the_published_splits():
    """The Quickstart points at ``llamanat alphabet`` to see the alphabet table.

    The command used to train on 1000 sequences and grade 64 of them, so it
    printed 0.344 where the table says 0.905 for the same seed, model and step
    count: a smaller split, not a smaller model. A demo that undercuts the table
    it is quoting is worse than no demo, so the specs have to come from the study.
    """
    from llamanat import cli
    from llamanat.tasks import ANSWER_MODES

    assert cli.TASKS is TASKS
    published = TASKS["recall"]
    assert published.n_eval == cli.PROBE_N
    for mode in ANSWER_MODES:
        spec = cli._recall_spec(mode)
        assert spec == replace(published, answer=mode)
        assert (spec.n_train, spec.n_eval) == (published.n_train,
                                               published.n_eval)
    source = inspect.getsource(cli)
    assert "lr=" not in source, \
        "the demo must not retune an optimiser the study already fixed"
