"""A figure is never split from what it is printed with, and never printed two ways in one row.

An outcome bar's note (``21% passed · 96 of 456``) wraps at its `` · `` or not at all, on the static page and in the
kit alike: a greedy wrap at its spaces set ``21% passed · 96`` over ``of 456``, a count apart from its denominator, and
sized the margin on the longer line. A change's glyph title states its interval as the row's interval column prints
it, each end by the column's own rule: the title read ``-7 pts, interval -15.09 to -0.03`` beside a column reading
``-15 to -0.03 pts``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from threetears.evals.analysis import ChartBlock
from threetears.evals.analysis.quantities import POINTS
from threetears.evals.analysis.report.chart_svg import chart_drawing
from threetears.evals.analysis.report.figures import SHARE, UNITLESS, ContrastPlaces, MeasureRuler
from threetears.evals.analysis.report.forms import PageForms
from packages.evals.tests.report_kit_support import chromium_unavailable, rendered, report_page

#: Outcome bars whose notes each wrap at a phone's width, each a share passed and its count of its arm's trials.
_NOTED_BARS = {
    "rows": [
        {"label": "control", "value": 21.0, "note": "21% passed · 96 of 456"},
        {"label": "candidate-a", "value": 23.0, "note": "23% passed · 87 of 380"},
    ],
    "domain": [0, 100],
    "fmt": "pct",
}

#: Each drawn text set in lines (a label, a note), its lines in order.
_LINES = re.compile(r"<text[^>]*>((?:<tspan[^>]*>[^<]*</tspan>)+)</text>")
_SPAN = re.compile(r"<tspan[^>]*>([^<]*)</tspan>")


def _note_lines(drawn: list[list[str]]) -> list[list[str]]:
    """The drawn texts that are a note, by their lines."""
    notes = {row["note"] for row in _NOTED_BARS["rows"]}  # type: ignore[index]
    return [lines for lines in drawn if " ".join(lines) in notes]


def _broken_at_the_dot_or_whole(lines: list[str]) -> bool:
    if len(lines) == 1:
        return True
    return len(lines) == 2 and lines[0].endswith(" ·") and re.fullmatch(r"\d+ of \d+", lines[1]) is not None


def test_a_static_outcome_note_breaks_at_its_dot_never_inside_its_count() -> None:
    """At a phone's 360 px the note's room is 98 px: ``21% passed · 96`` fits it and was set over ``of 456``. The note
    breaks at its dot there, and at the wider drawings it fits whole on one line."""
    block = ChartBlock.model_validate({"section": "memo", "title": "Outcomes", "chart": "bars", "data": _NOTED_BARS})
    drawing = chart_drawing(block)
    assert drawing is not None
    drawn = [_SPAN.findall(spans) for spans in _LINES.findall(drawing)]
    notes = _note_lines(drawn)
    # Each of the two notes at each of the three widths.
    assert len(notes) == 6, drawn
    assert all(_broken_at_the_dot_or_whole(lines) for lines in notes), notes
    assert ["21% passed ·", "96 of 456"] in notes, notes
    assert ["21% passed · 96 of 456"] in notes, notes


#: Each bar chart text set in lines, its lines in order.
_KIT_LINES = """() => [...document.querySelectorAll('div.chart[data-kind="bars"] svg text')]
  .filter(t => t.querySelector('tspan'))
  .map(t => [...t.querySelectorAll('tspan')].map(s => s.textContent))"""


@pytest.mark.browser
@pytest.mark.parametrize("width", [1200, 390])
def test_the_kit_breaks_an_outcome_note_at_its_dot_never_inside_its_count(tmp_path: Path, width: int) -> None:
    """The kit wraps a note as the static page does (``kWrapNote``): at a phone's width at its dot, never between a
    count and its denominator."""
    reason = chromium_unavailable()
    if reason is not None:
        pytest.skip(f"headless Chromium cannot launch: {reason}")
    page = tmp_path / "outcomes.html"
    block = {"kind": "chart", "section": "memo", "chart": "bars", "title": "Outcomes", "data": _NOTED_BARS}
    page.write_text(report_page([block]), encoding="utf-8")
    drawn, errors = rendered(page, _KIT_LINES, width=width)
    assert errors == []
    notes = _note_lines(drawn)
    assert len(notes) == 2, drawn
    assert all(_broken_at_the_dot_or_whole(lines) for lines in notes), notes
    if width == 390:
        assert ["21% passed ·", "96 of 456"] in notes, notes


@pytest.mark.parametrize(
    ("ruler", "delta", "interval", "places", "unit", "title", "column"),
    [
        # A rate's change in points at whole points, its high end kept off zero on its own.
        (SHARE, -0.074, (-0.1509, -0.0003), 0, POINTS, "-7 pts, interval -15 to -0.03", "-15 to -0.03 pts"),
        # A score's change at two decimals, its high end kept off zero on its own.
        (UNITLESS, -0.15, (-0.299, -0.004), 2, "", "-0.15, interval -0.30 to -0.004", "-0.30 to -0.004"),
    ],
)
def test_a_glyph_title_states_the_interval_as_its_rows_interval_column_prints_it(
    ruler: MeasureRuler,
    delta: float,
    interval: tuple[float, float],
    places: int,
    unit: str,
    title: str,
    column: str,
) -> None:
    """A contrasts row's change cell and its interval column, at the set's places for changes: the glyph's title
    states the column's own ends, never both at the finer precision one end needs."""
    forms = PageForms.off_page()
    cell = forms.ruled_change_cell(ruler, delta, interval=interval, places=places)
    printed = forms.ruled_change_interval(ruler, *interval, places=ContrastPlaces(level=None, change=places))
    assert printed.text == column
    assert (cell.low_text, cell.high_text) == printed.ends_beside
    assert forms.glyph_said(cell, unit).text == title
