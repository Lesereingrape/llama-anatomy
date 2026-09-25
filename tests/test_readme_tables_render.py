"""Every pipe table in README.md must be structurally valid GitHub markdown.

The byte-comparison drift test pins the results block to the committed artifact, so it
cannot notice a renderer that emits a separator row with its cells glued together —
GitHub then silently shows the whole table as literal text. This checks the *shape* of
each table instead: consistent column counts, and a separator row of ``-``/``:`` cells
with one cell per column.
"""

from __future__ import annotations

import re
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def _cells(line: str) -> list[str] | None:
    s = line.strip()
    if not (s.startswith("|") and s.endswith("|")):
        return None
    return [c.strip() for c in s[1:-1].split("|")]


def _tables(text: str):
    block: list[tuple[int, list[str]]] = []
    for n, line in enumerate(text.splitlines(), 1):
        cells = _cells(line)
        if cells is None:
            if block:
                yield block
            block = []
        else:
            block.append((n, cells))
    if block:
        yield block


def test_readme_contains_tables():
    assert list(_tables(README.read_text(encoding="utf-8"))), (
        "README has no pipe tables; this test would pass vacuously")


def test_every_table_is_structurally_valid():
    problems = []
    for block in _tables(README.read_text(encoding="utf-8")):
        header_line, header = block[0]
        widths = {len(cells) for _, cells in block}
        if len(widths) > 1:
            problems.append(f"line {header_line}: ragged column counts {sorted(widths)}")
        if len(block) < 2:
            problems.append(f"line {header_line}: table has no separator row")
            continue
        sep_line, sep = block[1]
        if not all(re.fullmatch(r":?-+:?", cell) for cell in sep):
            problems.append(f"line {sep_line}: separator row is not all dashes: {sep}")
        elif len(sep) != len(header):
            problems.append(f"line {sep_line}: separator has {len(sep)} cells, "
                            f"header {header_line} has {len(header)}")
    assert not problems, "README tables GitHub cannot render:\n" + "\n".join(problems)
