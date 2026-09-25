"""The prose outside the generated block may describe only this repository.

``test_readme.py`` byte-pins the results block against ``results/anatomy.json``, which
leaves the hand-written numbers *above and below* that block unguarded: the parameter
count in the opening paragraph, the seed count, the "half-hour CPU job" the Quickstart
promises. Those are the claims that go quietly stale when a config changes or a sweep
grows, so they are measured here rather than remembered.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from llamanat.model import TinyLLaMA, n_params, variant_config

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "results" / "anatomy.json").read_text(encoding="utf-8"))
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _prose() -> str:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    return readme.split(START, 1)[0] + readme.split(END, 1)[1]


def _claimed_params() -> int:
    m = re.search(r"\b([\d,]{6,7})\s+parameters", _prose())
    assert m, "the README no longer states its parameter count in the prose"
    return int(m.group(1).replace(",", ""))


def test_the_prose_parameter_count_is_the_model_the_study_fits():
    actual = n_params(TinyLLaMA(variant_config("llama")))
    assert _claimed_params() == actual, (
        f"README says {_claimed_params():,}, the reference model has {actual:,}")
    assert actual == DATA["n_params_llama"], (
        "the prose, the artifact and the code disagree about the model size")


def test_the_title_order_of_magnitude_is_within_ten_percent():
    m = re.search(r"a ([\d,]{1,3})[kK]-parameter\s+LLaMA", _prose())
    assert m, "the title no longer states an order of magnitude"
    claimed = int(m.group(1).replace(",", "")) * 1000
    actual = n_params(TinyLLaMA(variant_config("llama")))
    assert abs(claimed - actual) / actual < 0.10, f"title says {claimed}, model is {actual}"


def test_the_prose_fraction_of_a_million_is_the_model_size():
    """The motivation paragraph rounds the model into a phrase, so check the rounding."""
    words = {"half": 2, "third": 3, "quarter": 4, "fifth": 5, "sixth": 6,
             "eighth": 8, "tenth": 10, "twentieth": 20, "hundredth": 100}
    m = re.search(rf"a ({'|'.join(words)}) of a million parameters", _prose())
    assert m, "the README no longer describes its budget as a fraction of a million"
    claimed = 1_000_000 / words[m.group(1)]
    actual = n_params(TinyLLaMA(variant_config("llama")))
    assert abs(claimed - actual) / actual < 0.10, (
        f"the prose says a {m.group(1)} of a million ({claimed:,.0f}) but the model "
        f"has {actual:,} parameters")


def test_the_prose_seed_count_matches_the_study():
    m = re.search(r"\b(one|two|three|four|five)\s+seeds?\s+per\s+row", _prose())
    assert m, "the README no longer states its seed count in words"
    assert ["one", "two", "three", "four", "five"].index(m.group(1)) + 1 == len(
        DATA["settings"]["seeds"]), m.group(0)


def test_the_promised_runtime_is_the_runtime_that_was_measured():
    m = re.search(r"a (half-hour|hour|([\d.]+)-hour|([\d]+)-minute)\s+CPU\s+job", _prose())
    assert m, "the README no longer states how long the study takes"
    if m.group(2):
        claimed = float(m.group(2)) * 60.0
    elif m.group(3):
        claimed = float(m.group(3))
    else:
        claimed = 30.0 if m.group(1) == "half-hour" else 60.0
    minutes = DATA["runtime_sec"] / 60.0
    assert 0.5 * claimed <= minutes <= 2.0 * claimed, (
        f"README promises a ~{claimed:.0f}-minute job; the committed artifact took "
        f"{minutes:.1f} minutes")


def test_readme_cites_each_paper_once():
    """A repeated arXiv link is the fingerprint of a spliced-in duplicate sentence.

    The prose here was left with a dangling fragment and a second copy of the
    Vaswani citation; nothing else in the suite reads the reference list, so this
    is the guard that a future edit does not paste one in again.
    """
    cited = re.findall(r"arxiv\.org/abs/([0-9.v]+)", _prose())
    assert cited, "the README no longer cites its own sources"
    repeats = {x for x in cited if cited.count(x) > 1}
    assert not repeats, f"cited more than once, probably a duplicated fragment: {repeats}"


def test_readme_names_the_std_convention_the_tables_use():
    """`+/-` is ambiguous unless the file says which divisor produced it.

    The published spreads are the population standard deviation over seeds, so the
    README has to use that word: a reader who recomputed the other convention would
    land on a different number and conclude the tables were wrong.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert re.search("population[^.]{0,60}standard\\s+deviation", readme), (
        "the README no longer states which standard-deviation convention its "
        "`+/-` columns use")
