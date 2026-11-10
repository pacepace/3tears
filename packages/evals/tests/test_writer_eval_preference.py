"""The writer eval's preference reads on one scale: the share of the judge's preference each page won, the candidate's
beside the control's, against even (50%), so a reader sees at once which page is preferred over which, by how much,
against what, and whether it is settled.

Rung 5's replay, from its recordings (the judge's recorded again on 2026-11-10, over the C5b pages), with no key: the
judge gave lead-first 70% of its preference over shipped, so shipped has 30%, and the 95% interval on lead-first's share
runs 33% to 100% (it is printed on the share, never on the difference of the two shares); its two orders agreed on 6
of 7 cases. The judge does not read this page, so its recording keys stand.

Mutations that turn this file red: a Difference column back on the table; the control's share missing, or not 100 less
the candidate's as printed; the interval printed on the difference's scale; the figure missing from the head of the
Scorecard, its even line not at 50 or unlabelled, or its dot and interval not the table's figures; the reading line
under it missing the control's share or the even point; the strip's marks drawn without their texts; a rate in the
readings table read as a fraction.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest

from packages.evals.tests.ladder_support import load, offline_copy
from threetears.evals.analysis.report import ChartBlock, DisclosureBlock, DotIntervalChart, TableBlock
from threetears.evals.harness import WriterResult


@pytest.fixture(scope="module")
def rung5(tmp_path_factory: pytest.TempPathFactory) -> Iterator[WriterResult]:
    """Rung 5's writer eval replayed once, from its recordings, with no key."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        copy = offline_copy("5-writer", tmp_path_factory.mktemp("rung5"), monkeypatch)
        yield asyncio.run(load(copy / "writer.py", "writer", monkeypatch).writer_eval.run())


def _scorecard(result: WriterResult) -> list[object]:
    return [block for block in result.page.blocks if block.section == "scorecard"]


def test_the_side_by_side_table_prints_both_shares_and_no_difference(rung5: WriterResult, tmp_path: Path) -> None:
    (side,) = [block for block in _scorecard(rung5) if isinstance(block, TableBlock) and block.name == "side_by_side"]
    headers = [column.header for column in side.columns]
    assert headers == [
        "Candidate",
        "Pairs",
        "Reads",
        "Candidate's share (%)",
        "Control's share (%)",
        "95% interval (%)",
    ]
    every = side.values()[0]
    # The two shares as printed sum to 100, and the interval is on the candidate's share, the scale the reader sees.
    assert (every["share"], every["control_share"], every["interval"]) == (70, 30, "33 to 100")
    assert "| lead-first | every pair | cannot tell apart | 70 | 30 | 33 to 100 |" in rung5.pairwise_text().splitlines()
    written = rung5.write(tmp_path / "report")
    for role in ("markdown", "html"):
        page = written[role].read_text(encoding="utf-8")
        assert "Difference" not in page, role


def test_the_scorecard_opens_on_the_preference_drawn_against_even(rung5: WriterResult, tmp_path: Path) -> None:
    figure, line, *_ = _scorecard(rung5)
    assert isinstance(figure, ChartBlock) and isinstance(figure.data, DotIntervalChart)
    chart = figure.data
    # One row, the candidate, at the table's figures: its share and the interval on it, on the share scale.
    assert [(row.label, row.value, row.lo, row.hi, row.hollow) for row in chart.rows or []] == [
        ("lead-first", 70, 33, 100, None)
    ]
    assert (chart.domain, chart.fmt, chart.level) == ((0, 100), "pct:0", "95%")
    # Its mark carries the table's figures as text, so its tooltip and values table print them, not the kit's rule.
    assert [(row.value_text, row.lo_text, row.hi_text) for row in chart.rows or []] == [("70%", "33", "100")]
    # The baseline is drawn and named: even, at 50.
    assert chart.reference is not None
    assert (chart.reference.value, chart.reference.label) == (50, "even (50%)")
    assert figure.title == "Share of the judge's preference against shipped"
    # Under the figure, one line: the candidate's share, the control's, the even point, the interval and the reading.
    assert isinstance(line, DisclosureBlock)
    assert line.text == "lead-first 70% · shipped 30% (50% is even), 95% interval 33 to 100%; cannot tell apart."
    html = rung5.write(tmp_path / "report")["html"].read_text(encoding="utf-8")
    assert "even (50%)" in html and "lead-first 70% · shipped 30% (50% is even)" in html


def test_the_readings_table_reads_a_rate_as_a_percent_and_the_count_it_is_of(rung5: WriterResult) -> None:
    """Report finished read ``0.43`` and ``0.86``, a rate written as a fraction at two decimals; it reads as a
    percent and the cases it counts, at the precision its row's figures give. A mean that is no rate reads as the
    aspect page prints its arms' levels where that page states a contrast of it (one form per measure, number rule 1:
    Must-fix findings open reads 0.4 there and here, never 0.43), else at two decimals (Should-fix findings open)."""
    lines = rung5.render().splitlines()

    assert "| Report finished | 43% (3 of 7) | 86% (6 of 7) |" in lines
    assert "| Bounces | 0.9 | 1.0 |" in lines
    assert "| Must-fix findings open | 0.4 | 0.3 |" in lines
    assert "| Should-fix findings open | 2.43 | 1.57 |" in lines
