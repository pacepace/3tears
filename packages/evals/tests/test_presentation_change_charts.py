"""Change charts name their zero, show the level the change is from, and mark which way is better (visual pass, V3).

The owner, 2026-11-08: "charts showing deltas for treatments as -8 or +6 or whatever, but control was not obvious".
Held here, offline, on the prototype-5 writer pages (Kairo, r2 with one gate, r3 without its configs) as their
accepted records draw them, and on rung 5's replayed writer eval:

- every change chart names its zero: no change against the control as a chart row names it, or against the second of
  two arms compared, or the arm a lever's move starts at;
- the level the change is from is on the figure: in the zero label where every row shares it, in each group's head
  where each measure's rows do, else in each row's note, and it is the level the input states;
- a change chart over a measure with a direction marks which way is better under its axis, on the chart and on the
  static page; the writer eval's preference figure says whose page each side prefers;
- changes per slice, per place and per lever draw as dots, a row with no interval an open mark (never solid while the
  row that has an interval is muted), and the chart says so;
- a factor panel's labels are never cut so that two rows read alike: a label too long for one line wraps to two, then
  takes its short label;
- a forest's shaded band is named, and a family-corrected level says what it holds.

Mutations that turn this file red: a change chart with no zero label, or one naming no reference; the level missing
from the zero label, the group heads or the notes, or not the input's figure; ``better`` missing, or the words absent
from the drawn SVG; a slice or lever chart drawn as bars, or a row with no interval drawn solid; a factor label cut to
"Brisk-S · reason…" twice; the margin band unlabelled; the writer eval's ends unnamed.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.ladder_support import load, offline_copy
from packages.evals.tests.report_doc_support import REPORT_DOCS, snapshot_of
from threetears.evals.analysis.bundle.schema import TRIAL_PASS_NAME
from threetears.evals.analysis.report import ChartBlock, DotIntervalChart
from threetears.evals.analysis.report.chart_svg import chart_drawing
from threetears.evals.analysis.report.block_spec import block_look
from threetears.evals.analysis.report.charts import BarsChart, ForestChart
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.model import DocPage
from threetears.evals.analysis.report.parts.full_results import every_table
from threetears.evals.analysis.report_arms import fact_arms
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.harness import WriterResult
from threetears.evals.kernel.report_doc import ReportDoc, ReportFinding

#: The writer pages held here.
_PAGES = ("prototype5/kairo", "prototype5/r2-with-one-gate", "prototype5/r3-without-its-configs")

#: The sources a chart draws as changes, by the kind the document asks for.
_CHANGE_SOURCES = {
    "forest": {"readings", "contrasts"},
    "diverging_bars": {
        "readings",
        "contrasts",
        "measures",
        "judged",
        "strata",
        "tasks",
        "configs",
        "compare",
        "levers",
    },
    "dot_interval": {"readings", "contrasts", "compare", "levers"},
}

#: Every drawn text node's text, its spans joined: ``<text ...><tspan>a</tspan><tspan>b</tspan></text>``.
_TEXT = re.compile(r"<text[^>]*>(.*?)</text>", re.S)
_TAG = re.compile(r"<[^>]+>")


@cache
def _record(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((REPORT_DOCS / name / "record.json").read_text())
    return loaded


@cache
def _input(name: str) -> ReportInput:
    return ReportInput.model_validate(snapshot_of(name).report_input)


@cache
def _page(name: str) -> DocPage:
    record = _record(name)
    return build(
        ReportDoc.model_validate(record["doc"]),
        _input(name),
        strict=True,
        findings=[ReportFinding.model_validate(finding) for finding in record.get("findings", [])],
        draft=record["ending"] == "stored_with_findings",
    )


def _changes(name: str) -> Iterator[tuple[str, str, ChartBlock, list[str]]]:
    """Each change chart on a page: the kind the document asks for, its source, the block drawn and its notes."""
    page = _page(name)
    for claim, asked in zip(page.claims, _record(name)["doc"]["claims"], strict=True):
        for evidence, block in zip(claim.evidence, asked["evidence"], strict=True):
            source = block["query"]["source"]
            if source not in _CHANGE_SOURCES.get(block["block"], set()) or evidence.chart is None:
                continue
            drawn = evidence.chart.chart
            assert isinstance(drawn, ChartBlock), (name, block["block"], source)
            yield block["block"], source, drawn, evidence.chart.notes


def _one(name: str, source: str, index: int = 0) -> tuple[ChartBlock, list[str]]:
    found = [(drawn, notes) for _, kind, drawn, notes in _changes(name) if kind == source]
    return found[index]


def _drawn_texts(block: ChartBlock) -> list[str]:
    """Every text the static page draws for a chart, each text node's spans joined by a space."""
    drawing = chart_drawing(block)
    assert drawing is not None
    return [" ".join(_TAG.sub(" ", found).split()) for found in _TEXT.findall(drawing)]


@pytest.mark.parametrize("name", _PAGES)
def test_every_change_chart_names_its_zero(name: str) -> None:
    """r2's compare charts (each place against the second arm) drew an unnamed zero; every chart now names it."""
    charts = list(_changes(name))
    assert charts, name
    for asked, source, drawn, _ in charts:
        data = drawn.data
        assert isinstance(data, BarsChart | ForestChart | DotIntervalChart), (asked, source)
        assert data.zero_label is not None and data.zero_label.startswith("no change "), (asked, source, data)
        assert any(text.startswith("no change ") for text in _drawn_texts(drawn)), (asked, source)


def test_a_forest_shows_the_controls_level_where_its_rows_share_one() -> None:
    """Kairo's cost forest read zero as "P · September" alone: the control's level per 1,000 replies is in its zero
    label now; its pass-rate forest reads two measures, so each measure's head carries the control's level there."""
    board = _input("prototype5/kairo").scoreboard["September"]
    assert board.passed is not None and board.usable
    rate = (
        PageForms.of(_input("prototype5/kairo")).level(TRIAL_PASS_NAME, board.passed / board.usable, kind="trial").text
    )
    (passes, _), (cost, cost_notes) = _one("prototype5/kairo", "contrasts"), _one("prototype5/kairo", "contrasts", 1)
    assert isinstance(passes.data, ForestChart) and isinstance(cost.data, ForestChart)
    assert passes.data.zero_label == "no change vs P · September"
    heads = list(dict.fromkeys(row.group for row in passes.data.rows))
    # Each measure's level at its one precision on the page (number policy, rule 1).
    assert heads == [f"Pass rate · September {rate}", "Persona holds character · September 91%"], heads
    assert rate == "85%"
    assert cost.data.zero_label == "no change vs P · September ($1.89, mean of 294 cases)"
    assert "Production cost, per 1,000 replies, in dollars." in cost_notes


def test_each_place_carries_the_level_its_change_is_from() -> None:
    """r2's first step drew "+1.7 pts" per task with no baseline; each row now reads from the second arm's own figure
    in that task, the overall row from its overall figure."""
    drawn, _ = _one("prototype5/r2-with-one-gate", "compare")
    data = drawn.data
    assert isinstance(data, BarsChart) and data.rows is not None
    assert data.zero_label == "no change vs J · Brisk-S · reasoning low · 1 retry"
    report_input = _input("prototype5/r2-with-one-gate")
    second = "brisk-s.low.r1"
    levels = {task.id: task.per_arm[second]["pass_rate"].mean for task in report_input.tasks}
    levels["overall"] = report_input.measures["pass_rate"].per_arm[second].mean
    places = PageForms.of(report_input).measure("pass_rate").places
    assert places is not None
    for row in data.rows:
        assert row.note is not None and row.id is not None
        level = levels[row.id]
        assert level is not None
        printed = re.fullmatch(r"[+-][\d.]+ pts from ([\d.]+)%", row.note)
        assert printed is not None, (row.id, row.note)
        # The level printed is the second arm's own figure there, at its measure's one precision on the page.
        assert abs(float(printed[1]) - float(level)) <= 0.5 * 10**-places, (row.id, row.note, level)


def test_a_change_chart_marks_which_way_is_better() -> None:
    """No change chart said which way was better: pass rate is better right, cost better left, on the chart's data
    and under its drawn axis."""
    (passes, _), (cost, _) = _one("prototype5/kairo", "contrasts"), _one("prototype5/kairo", "contrasts", 1)
    assert passes.data.better == ("worse", "better")  # type: ignore[union-attr]
    assert cost.data.better == ("better", "worse")  # type: ignore[union-attr]
    assert {"← worse", "better →"} <= set(_drawn_texts(passes))
    assert {"← better", "worse →"} <= set(_drawn_texts(cost))
    for name in _PAGES:
        for asked, source, drawn, _ in _changes(name):
            assert drawn.data.better is not None, (name, asked, source)  # type: ignore[union-attr]


def test_changes_per_place_and_per_lever_are_dots_and_a_row_with_no_interval_is_open() -> None:
    """r2's compare chart drew its tasks as solid bars (no interval) and its one row with an interval muted: the
    evidence inverted. Still the diverging chart it was asked as, it now draws dots, each task an open mark (drawn
    hollow), the overall row filled with its interval, no row muted, and says what an open mark is; r3's lever effects
    draw as dots under a heading per setting."""
    drawn, notes = _one("prototype5/r2-with-one-gate", "compare")
    assert drawn.chart == "diverging_bars" and isinstance(drawn.data, BarsChart) and drawn.data.dots
    rows = drawn.data.rows or []
    assert [(row.id, row.lo is not None, row.muted) for row in rows] == [
        ("overall", True, None),
        ("single", False, None),
        ("multihop", False, None),
        ("datemath", False, None),
        ("disamb", False, None),
    ]
    assert "Open marks: no interval." in notes
    drawing = chart_drawing(drawn)
    assert drawing is not None
    marks = re.findall(r"<circle[^>]*style=\"([^\"]*)\"", drawing.split("<svg", 2)[1])
    # Per drawing: the overall row's mark filled, each task's hollow (filled with the surface, ringed in its colour).
    assert [style.startswith("fill:var(--surface)") for style in marks] == [False, True, True, True, True], marks
    levers, _ = _one("prototype5/r3-without-its-configs", "levers")
    assert levers.chart == "diverging_bars" and isinstance(levers.data, BarsChart) and levers.data.dots
    groups = list(dict.fromkeys(row.group for row in levers.data.rows or []))
    assert groups[:2] == ["model", "prompt"], groups


def test_a_factor_label_is_never_cut_so_two_rows_read_alike() -> None:
    """r2's factor panels drew "Brisk-S · reason…" twice in one panel. At a panel's width the static page now draws
    each label whole (on one line or two), or every row of a panel as its short label, never cut, and no two rows of a
    panel read alike."""
    drawn, _ = _one("prototype5/r2-with-one-gate", "levers")
    data = drawn.data
    assert isinstance(data, DotIntervalChart) and data.panels
    drawing = chart_drawing(drawn)
    assert drawing is not None
    svgs = re.findall(r"<svg.*?</svg>", drawing, re.S)
    assert len(svgs) == len(data.panels)
    for panel, svg in zip(data.panels, svgs, strict=True):
        texts = [" ".join(_TAG.sub(" ", found).split()) for found in _TEXT.findall(svg)]
        rows = panel.rows or []
        whole = [row.label in texts for row in rows]
        short = [row.short_label in texts for row in rows]
        assert all(whole) or all(short), (panel.title, texts)
        shown = [row.label if all(whole) else row.short_label for row in rows]
        assert len(set(shown)) == len(shown), (panel.title, shown)
        assert not any("…" in text for text in texts), texts


def test_the_margin_band_is_named_and_the_family_level_said() -> None:
    """Kairo's forest shaded its margin band with no word for it, and printed 99.69% with no word for why: its head
    says the guarantee the family's intervals hold at together, "95% intervals (adjusted for 16 comparisons)"."""
    drawn, _ = _one("prototype5/kairo", "contrasts")
    assert isinstance(drawn.data, ForestChart) and drawn.data.reference is not None
    # The margin is declared, so it reads as declared (number policy, rule 4), never ``-15.0``.
    assert drawn.data.reference.label == "margin -15 pts; shaded: no worse than the margin"
    assert drawn.data.level == "95% (adjusted for 16 comparisons)"
    assert "95% intervals (adjusted for 16 comparisons)" in _drawn_texts(drawn)


def _levels_read(held: ReportInput) -> tuple[set[str], set[str]]:
    """Kairo's page built on ``held``: the level each contrasts forest's head prints, and each Full results contrast
    row's."""
    record = _record("prototype5/kairo")
    page = build(
        ReportDoc.model_validate(record["doc"]),
        held,
        strict=True,
        findings=[ReportFinding.model_validate(finding) for finding in record.get("findings", [])],
        draft=record["ending"] == "stored_with_findings",
    )
    heads = {
        evidence.chart.chart.data.level
        for claim in page.claims
        for evidence in claim.evidence
        if evidence.chart is not None
        and isinstance(evidence.chart.chart, ChartBlock)
        and isinstance(evidence.chart.chart.data, ForestChart)
        and evidence.chart.chart.data.level
    }
    (contrasts,) = [
        table
        for table in every_table(held, block_look(held, fact_arms(held)), forms=PageForms.of(held))
        if table.name == "contrasts"
    ]
    return heads, {row["level"].text for row in contrasts.rows}


@pytest.mark.parametrize(
    ("stored", "read"),
    [
        ("99.69%", "95% (adjusted for 16 comparisons)"),
        ("99.95%", "95% (adjusted for 16 comparisons)"),
        ("99.951%", "95% (adjusted for 16 comparisons)"),
    ],
)
def test_a_family_level_reads_alike_in_a_chart_head_and_in_full_results(stored: str, read: str) -> None:
    """The chart head and the Full results table each formatted the family's level, by two rules, so one level could
    read two ways on one page (``99.951%`` in the head, ``99.95%`` in the table). One function reads it for both: the
    joint guarantee the family holds at, whatever its per-interval level, which never reads ``100%``."""
    held = _input("prototype5/kairo")
    if stored != held.contrast_families[0].level:
        held = held.model_copy(
            update={
                "contrasts": [contrast.model_copy(update={"level": stored}) for contrast in held.contrasts],
                "contrast_families": [family.model_copy(update={"level": stored}) for family in held.contrast_families],
            }
        )
    heads, rows = _levels_read(held)
    assert heads == rows == {read}, (heads, rows)


@pytest.fixture(scope="module")
def rung5(tmp_path_factory: pytest.TempPathFactory) -> Iterator[WriterResult]:
    """Rung 5's writer eval replayed once, from its recordings, with no key."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        copy = offline_copy("5-writer", tmp_path_factory.mktemp("rung5"), monkeypatch)
        yield asyncio.run(load(copy / "writer.py", "writer", monkeypatch).writer_eval.run())


def test_the_preference_figure_says_whose_page_each_side_prefers(rung5: WriterResult, tmp_path: Path) -> None:
    """Rung 5's preference dot sat on 0 to 100% with only "even" named: each end now says whose page it prefers."""
    (figure,) = [
        block for block in rung5.page.blocks if isinstance(block, ChartBlock) and block.chart == "dot_interval"
    ]
    assert isinstance(figure.data, DotIntervalChart)
    assert figure.data.better == ("prefers shipped", "prefers lead-first")
    assert {"← prefers shipped", "prefers lead-first →"} <= set(_drawn_texts(figure))
    html = rung5.write(tmp_path / "report")["html"].read_text(encoding="utf-8")
    assert '"better":["prefers shipped","prefers lead-first"]' in html.replace("&quot;", '"')
