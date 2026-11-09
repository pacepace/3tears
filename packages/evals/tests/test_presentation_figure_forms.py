"""Figures: one money form, one precision, charts that read cold (presentation build, chunk 3).

The review's A4 holds a page to one money form per figure type and one precision per figure. It found money in five
forms on one page (P17), a total of $0.00413 spelled ``$4.13/1000`` (P16), rounding that printed an interval's two ends
alike (P19), scale lines in raw dollars and padded four-figure ends under cells in another form (W3, P44, P45), a change
chart whose zero was not named (H4), a scatter on zero-based axes with no intervals (P46), forest end labels in two
forms (P47), a quick print of four-decimal means (P43), and one arm tagged A on the code page and B on the document
page (P40). Held here, offline, on the ladder's replayed rungs, the Kairo code page and its prototype-5 document page,
and on the toy shapes:

- for every chart kind a page draws and every grid table, the scale line's money is the cells' form;
- every per-reply cost reads per 1,000 replies in its header or axis, and every total reads plain dollars;
- no interval prints two equal ends unless its stored ends are equal;
- a figure prints at one precision in its chart label, its values table and the print: a forest's end label is its
  note, never the axis's coarser format; the cost chart's cells, tips and axis read one money form; the full print
  states every figure as the quiet print and the page do (a writer page's forest values table still reads four
  figures: changing it moves rung 5's recorded judge evidence, which waits on a re-recording);
- a change chart names its zero;
- the interval glyph draws a zero and a margin tick;
- the same arm has the same tag letter and colour on the code page, the aspect page and the document page, for a
  two-arm and a three-arm campaign.
"""

from __future__ import annotations

import asyncio
import html
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from threetears.evals.analysis import (
    CampaignReport,
    campaign_report,
    campaign_report_html,
    campaign_report_markdown,
    report_html,
    report_markdown,
)
from threetears.evals.analysis.bundle.schema import TRIAL_PASS_NAME
from threetears.evals.analysis.analyst.blocks import values_table
from threetears.evals.analysis.report.block_spec import block_look, complete_chart
from threetears.evals.analysis.report.blocks import ChartBlock, TableBlock, TableCell
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.parts.resolve import resolve_chart
from threetears.evals.kernel.chart_query import ChartQuery, PlannedChart
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.html import doc_page_html
from threetears.evals.analysis.report.doc_page.text import doc_page_text
from threetears.evals.analysis.report.chart_svg import chart_drawing
from threetears.evals.analysis.report.html_parts import KIT_JS, grid_html
from threetears.evals.analysis.report.parts.look import arm_token
from threetears.evals.analysis.report.svg import interval_glyph_svg
from threetears.evals.analysis.report_arms import fact_arms
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.analysis.service import assemble_campaign_bundle
from threetears.evals.kernel.numbers import money_form
from threetears.evals.kernel.report_doc import ReportDoc, ReportFinding
from threetears.evals import Answer, Eval
from threetears.evals.quick.files import load_evals
from threetears.evals.quick.result import Result

from packages.evals.tests.ladder_support import load, offline_copy
from packages.evals.tests.report_doc_support import REPORT_DOCS, snapshot_of
from packages.evals.tests.report_shapes_support import ArmTrials, AspectTrials, s2_choice, shape_of

#: The rungs held here: their directory, the file declaring their eval, and the attribute holding it (None for TOML).
_RUNGS: dict[str, tuple[str, str, str | None]] = {
    "rung0": ("1-triage", "triage.toml", None),
    "rung2": ("2-replies", "replies.py", "replies_eval"),
    "rung3": ("3-invoices", "extract.py", "invoices"),
}

#: An amount of money as a page writes it, its digits after the point captured: ``$0.046``, ``-$1,400.00``.
_MONEY = re.compile(r"[+\-]?\$(\d[\d,]*)(?:\.(\d+))?")

#: The retired per-1,000 mark on an amount (Q4).
_MARK = re.compile(r"\$[\d.,]+/1000")

#: An interval whose two printed ends are alike: ``[$0.045, $0.045]``, ``[4.2%, 4.2%]``.
_EQUAL_ENDS = re.compile(r"\[(-?\$?[\d.,]+%?), \1\]")

#: A scale line under a table of glyphs.
_AXIS = re.compile(r'<p class="axis">(.*?)</p>')

#: A chart's data on a page.
_CHART = re.compile(r'<div class="chart"[^>]*data-kind="([^"]+)"[^>]*data-chart="([^"]*)"')


@dataclass(frozen=True)
class _Rung:
    result: Result
    page: CampaignReport


@pytest.fixture(scope="module")
def rungs(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, _Rung]]:
    """Every rung held here, replayed once from its recordings, with no key."""
    replayed: dict[str, _Rung] = {}
    for name, (directory, file, attribute) in _RUNGS.items():
        with pytest.MonkeyPatch.context() as monkeypatch:
            copy = offline_copy(directory, tmp_path_factory.mktemp(name), monkeypatch)
            if attribute is None:
                (runnable,) = load_evals(copy / file)
            else:
                runnable = getattr(load(copy / file, Path(file).stem, monkeypatch), attribute)
            result = asyncio.run(runnable.run(store=":memory:"))
        replayed[name] = _Rung(result=result, page=result.page)
    yield replayed


@pytest.fixture(scope="module")
def kairo_document() -> str:
    """The Kairo prototype-5 document's page, as its accepted record draws it, its HTML."""
    record = json.loads((REPORT_DOCS / "prototype5" / "kairo" / "record.json").read_text())
    page = build(
        ReportDoc.model_validate(record["doc"]),
        ReportInput.model_validate(snapshot_of("prototype5/kairo").report_input),
        strict=False,
        findings=[ReportFinding.model_validate(finding) for finding in record.get("findings", [])],
        draft=record["ending"] == "stored_with_findings",
    )
    return doc_page_html(page)


def _tables(page: CampaignReport) -> Iterator[TableBlock]:
    """Every table the campaign's page and its aspects' pages lay out."""
    yield from (block for block in page.blocks if isinstance(block, TableBlock))
    for aspect in page.aspects:
        yield from (block for block in aspect.report.blocks if isinstance(block, TableBlock))


def _charts(page: CampaignReport) -> Iterator[ChartBlock]:
    """Every chart the campaign's page and its aspects' pages draw."""
    yield from (block for block in page.blocks if isinstance(block, ChartBlock))
    for aspect in page.aspects:
        yield from (block for block in aspect.report.blocks if isinstance(block, ChartBlock))


def _decimals(text: str) -> set[int]:
    """The decimals every amount of money in ``text`` is written to."""
    return {len(found.group(2) or "") for found in _MONEY.finditer(text)}


def _money_cells(block: TableBlock) -> list[TableCell]:
    return [cell for row in block.rows for cell in row.values() if cell.decimals is not None and cell.is_number]


#: The page's one script, the kit's, whose own comments name the retired mark it still reads.
_SCRIPT = re.compile(r"<script>.*?</script>", re.DOTALL)


def _page_texts(rung: _Rung) -> list[str]:
    """Every form a rung's pages are read in, as a reader reads them: the kit's script left out."""
    return [
        _SCRIPT.sub("", text)
        for text in [
            campaign_report_markdown(rung.page),
            campaign_report_html(rung.page),
            campaign_report_html(rung.page, static=True),
            report_markdown(rung.result.report),
            report_html(rung.result.report),
        ]
    ]


# --- one money form ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_RUNGS))
def test_every_grid_table_s_scale_line_spells_its_money_and_its_ends_as_its_cells_do(
    rungs: dict[str, _Rung], name: str
) -> None:
    """W3, P44, P45: the line under a table of glyphs states the scale's data ends in the cells' own money form, with
    the unit on every span, and a rate's span inside 0 to 100."""
    checked = 0
    for block in _tables(rungs[name].page):
        lines = _AXIS.findall(grid_html(block))
        if not lines:
            continue
        (line,) = lines
        money = _money_cells(block)
        if "$" in line:
            checked += 1
            assert _decimals(line) <= {cell.decimals for cell in money}, (block.name, line)
            assert "/1000" not in line
        for span in re.findall(r"drawn from (.*?)(?= and from | · |$)", line):
            low, _, high = span.partition(" to ")
            assert low and high, line
            # Every span carries its unit: money and a percent on each end, any other unit once after the high end.
            assert ("$" in low and "$" in high) or ("%" in low and "%" in high) or re.search(r" [a-z]+$", high), span
            if "%" in high:
                assert 0 <= float(low.rstrip("%")) and float(high.rstrip("%")) <= 100, span
    # Rung 2 measures one arm, whose scorecard is turned and draws no glyph (visual pass V2); rung 3's compares.
    if name == "rung3":
        assert checked, f"{name} draws money glyphs"


@pytest.mark.parametrize("name", sorted(_RUNGS))
def test_every_chart_s_money_ticks_read_in_its_values_table_s_form(rungs: dict[str, _Rung], name: str) -> None:
    """Every chart a page draws states its money in one form: an axis or value format with digits (``usd:3``) writes
    every amount its values table holds to those digits, and no chart carries the retired per-1,000 format."""
    for chart in _charts(rungs[name].page):
        drawn = chart.kit_chart()
        if drawn is None:
            continue
        _, data = drawn
        formats = [str(data[key]) for key in ("fmt", "x_fmt", "y_fmt") if key in data]
        assert not any(spec.lstrip("+").startswith("usdpk") for spec in formats), formats
        for spec in formats:
            found = re.fullmatch(r"\+?usd:(\d)", spec)
            if found is None:
                continue
            headers, rows = chart.values_table()
            amounts = _decimals(" ".join(cell for row in rows for cell in row) + " ".join(headers))
            assert amounts <= {int(found.group(1))}, (chart.title, spec, rows)


@pytest.mark.parametrize("name", sorted(_RUNGS))
def test_per_reply_cost_reads_per_1000_replies_and_totals_read_plain_dollars(
    rungs: dict[str, _Rung], name: str
) -> None:
    """Q4: the ``/1000`` mark is retired; the count is in the header (``Cost per 1,000 replies``). Q3: a total is plain
    dollars, to two significant figures under a dollar."""
    rung = rungs[name]
    for text in [*_page_texts(rung), str(rung.result)]:
        assert not _MARK.search(text), _MARK.search(text)
    for block in _tables(rung.page):
        heads = {column.key: column.header for column in block.columns}
        for row in block.rows:
            for key, cell in row.items():
                # A money measure's cell in a scorecard table reads per 1,000 replies, as its header says, or in one
                # arm's turned matrix, as its row's measure says.
                if cell.decimals is not None and cell.unit == "usd" and block.name == "scorecard":
                    named = row["measure"].text if "measure" in row else heads[key]
                    assert named.endswith(", per 1,000 replies"), named
    markdown = campaign_report_markdown(rung.page)
    (test_cost,) = [line for line in markdown.splitlines() if line.startswith("| Test cost |")]
    for found in _MONEY.finditer(test_cost):
        whole, cents = found.group(1), found.group(2) or ""
        significant = (whole.lstrip("0,") + cents).lstrip("0") if whole in ("0", "") else whole + cents
        assert len(significant.replace(",", "")) <= 2 or (whole != "0" and len(cents) == 2), test_cost


def test_an_interval_takes_one_figure_more_only_where_two_would_print_its_ends_alike() -> None:
    """P19: ``$0.045 [$0.045, $0.045]`` hid the interval; one figure more tells its ends apart, and no more, for that
    interval alone (owner ruling Q3: "one more only where"): the set's other amounts keep two figures."""
    plain = money_form([0.0451, 0.0448, 0.0453, 0.11])
    assert plain.spell(0.0448) == plain.spell(0.0453) == "$0.045"
    assert plain.spell_ends(0.0448, 0.0453) == ("$0.0448", "$0.0453")
    assert plain.places_of([0.0451, 0.0448, 0.0453], ends=(0.0448, 0.0453)) == 4
    assert plain.spell(0.11) == "$0.11", "the set's other amounts keep two figures"
    # An interval one more figure does not tell apart is as narrow as the figures can say: no further.
    assert plain.spell_ends(0.04512, 0.04513) == ("$0.0451", "$0.0451")
    # An interval whose stored ends are equal is printed alike, as stored.
    assert plain.spell_ends(0.045, 0.045) == ("$0.045", "$0.045")
    assert plain.places_of([0.045], ends=(0.045, 0.045)) == 3


@pytest.mark.parametrize("name", sorted(_RUNGS))
def test_no_interval_prints_two_equal_ends(rungs: dict[str, _Rung], name: str) -> None:
    rung = rungs[name]
    for text in [*_page_texts(rung), str(rung.result)]:
        assert not _EQUAL_ENDS.search(text), _EQUAL_ENDS.search(text)


# --- one precision per figure -------------------------------------------------------------------------------------------


def test_rung_3_s_test_costs_read_alike_on_the_page_and_in_the_print(rungs: dict[str, _Rung]) -> None:
    """P16, A4: each arm's test cost is one figure, so the page's Methods row and the print's total spend print it at
    one precision, in plain dollars."""
    rung = rungs["rung3"]
    (row,) = [line for line in campaign_report_markdown(rung.page).splitlines() if line.startswith("| Test cost |")]
    paged = dict(re.findall(r"(v\d) (\$[\d.]+)", row))
    printed = str(rung.result)
    for arm, amount in paged.items():
        block = printed.split(f"\n{arm}")[1]
        assert f"  total spend: {amount}, the results' cost" in block, (arm, amount)
    assert paged == {"v1": "$0.0041", "v2": "$0.0063"}


def test_the_quick_print_states_rates_in_percent_with_their_trials_and_intervals_as_the_page_does(
    rungs: dict[str, _Rung],
) -> None:
    """P43: no four-decimal means, no ``0.4377-1`` intervals, cases and trials named."""
    printed = str(rungs["rung0"].result)
    assert "  Accuracy: 90%, 81 of 90 trials" in printed
    assert "precision 100% (15 of 15 trials over 5 cases, [43.8%, 100%])" in printed
    assert "top misses (5 of 28 failed trials):" in printed
    assert not re.search(r"(?<!\$)\b0\.\d{4}\b|CI \d|\d-1\b", printed), printed


def test_the_cost_chart_s_tips_and_values_table_print_a_cost_alike_and_its_axes_are_set_to_the_data() -> None:
    """P46: Kairo's scatter clustered three arms in one corner of a zero-based axis and drew no intervals. A campaign's
    cost/quality scatter draws each arm's intervals and sets its axes to its data; a cost reads alike in its tip and in
    its values table."""
    page = s2_choice().page()
    (chart,) = [block for block in _charts(page) if block.chart == "cost_quality"]
    drawn = chart.kit_chart()
    assert drawn is not None
    _, data = drawn
    points: list[dict[str, Any]] = data["points"]  # type: ignore[assignment]
    lows = [point["x_lo"] for point in points]
    highs = [point["x_hi"] for point in points]
    x_domain: list[float] = data["x_domain"]  # type: ignore[assignment]
    y_domain: list[float] = data["y_domain"]  # type: ignore[assignment]
    assert x_domain[0] > 0 and x_domain[0] <= min(lows) and x_domain[1] >= max(highs)
    assert y_domain[0] > 0 and y_domain[1] <= 100
    assert data["level"] == "95%"
    headers, rows = chart.values_table()
    for point, row in zip(points, rows, strict=True):
        # P17: every cost carries its dollar sign, as the scoreboard writes it, its count in the head.
        tip_cost = point["tip"][1]
        assert tip_cost.startswith(f"{row[1]} ({row[2]}) "), (tip_cost, row)
        assert row[1].startswith("$") and re.fullmatch(r"\$[\d.,]+ to \$[\d.,]+", row[2]), row
    assert headers[1:3] == ["Cost per 1,000 replies", "Interval"]
    assert data["x_title"] == "Cost per 1,000 replies"


# --- charts that read cold -----------------------------------------------------------------------------------------------


def test_a_change_chart_names_its_zero_as_the_control(kairo_document: str) -> None:
    """H4: "Change against September (P)" never said that zero is September. Every change chart against the control
    names its zero line with the control as its rows name arms, on the drawn page and on the static one: no change
    against it, with its level where every row shares one (the cost forest's), the pass-rate forest's two measures
    each carrying theirs in its group's head."""
    changes = [
        json.loads(html.unescape(data))
        for kind, data in _CHART.findall(kairo_document)
        if kind in ("forest", "diverging_bars")
    ]
    assert changes, "the Kairo document draws a change chart"
    assert [chart.get("zero_label") for chart in changes] == [
        "no change vs P · September",
        "no change vs P · September ($1.89, mean of 294 cases)",
    ], [c.get("zero_label") for c in changes]
    assert "zero_label" in KIT_JS and "zeroLabel" in KIT_JS


def test_the_interval_glyph_draws_a_zero_tick_and_a_tick_at_each_end_of_its_margin() -> None:
    """The host's iframe pass (§7 finding 5): a glyph across zero draws a zero tick, and under a margin a tick at each
    of its ends, so a margin as wide as the track still reads."""
    drawn = interval_glyph_svg(-7.6, -10.8, -3.4, unit="", scale=(-15.0, 15.0), margin=15.0)
    assert drawn.count('class="viz-zero"') == 1
    assert drawn.count('class="viz-margin"') == 2
    crossing = interval_glyph_svg(3.0, -2.0, 8.0, unit="", scale=(-2.0, 8.0))
    assert crossing.count('class="viz-zero"') == 1 and 'class="viz-margin"' not in crossing


def test_a_margin_s_ticks_sit_strictly_inside_the_glyph_s_track() -> None:
    """Kairo's ±15-point margin over data from -10.8 to 1.1: the track runs past the margin's ends, so its two ticks
    and its band sit inside it and the zero tick stands apart from them; the line under the table states the data's
    and the margin's own ends."""
    block = TableBlock(
        section="scorecard",
        name="scorecard",
        title="By aspect",
        shared_scale=True,
        columns=[{"key": "aspect", "header": "Aspect"}, {"key": "arm:b", "header": "B (pts)"}],  # type: ignore[list-item]
        order="aspect by aspect",
        rows=[
            {
                "aspect": TableCell(value="character"),
                "arm:b": TableCell(value=-7.6, low=-10.8, high=1.1, margin=15.0, zero_line=True),
            }
        ],
        total_rows=1,
    )
    drawn = grid_html(block)
    track = re.search(r'class="viz-track" x1="([\d.]+)" y1="[\d.]+" x2="([\d.]+)"', drawn)
    ticks = [float(x) for x in re.findall(r'class="viz-margin" x1="([\d.]+)"', drawn)]
    assert track is not None and len(ticks) == 2
    start, end = float(track.group(1)), float(track.group(2))
    assert all(start < tick < end for tick in ticks), (start, ticks, end)
    (zero,) = [float(x) for x in re.findall(r'class="viz-zero" x1="([\d.]+)"', drawn)]
    assert ticks[0] < zero < ticks[1]
    # The margin's ends read at a point's one decimal, as the chart's margin line and the glyph's title read it.
    assert '<p class="axis">Each column drawn from -15.0 to 15.0 pts</p>' in drawn


def test_the_line_under_a_matrix_names_the_columns_each_range_is_drawn_for() -> None:
    """Discodon's 0.94.0 pass: a one-arm scorecard read two scales besides money, a percent's and a turn time's, and
    its line said "Each column drawn from 47.2% to 100% and from 973 to 1029", leaving a reader to guess which column
    is which range. Each range names its columns; "Each column" stays only where one range is every glyph column's."""
    columns = [
        ("pass", "Pass rate (%)", TableCell(value=70.0, low=47.2, high=86.0)),
        ("check", "Reply cites policy check (%)", TableCell(value=100.0, low=94.0, high=100.0)),
        ("turn", "Turn time", TableCell(value=1001.0, low=973.0, high=1029.0)),
        ("spend", "Measuring spend", TableCell(value=2.7, low=2.57, high=2.83, unit="usd")),
    ]
    block = TableBlock(
        section="scorecard",
        name="scorecard",
        title="solo by aspect",
        shared_scale=True,
        columns=[
            {"key": "aspect", "header": "Aspect"},
            *({"key": key, "header": header} for key, header, _ in columns),
        ],  # type: ignore[list-item]
        order="aspect by aspect",
        rows=[{"aspect": TableCell(value="research"), **{key: cell for key, _, cell in columns}}],
        total_rows=1,
    )
    (line,) = _AXIS.findall(grid_html(block))
    said = html.unescape(re.sub(r"<[^>]+>", "", line))
    assert "Each column" not in said and " and from " not in said, said
    ranges = said.split(" · ")
    assert len(ranges) == 3, ranges
    assert ranges[0].startswith("Pass rate (%) and Reply cites policy check (%) drawn from "), ranges
    assert any(part.startswith("Turn time drawn from 973 to 1") for part in ranges), ranges
    assert any(part.startswith("Measuring spend drawn from $") for part in ranges), ranges


def _quality_scorecard(*cells: tuple[str, str, TableCell]) -> TableBlock:
    """A one-arm code-only scorecard of the SearXNG quality campaign's shape (Discodon's re-shoot of host pass 5): one
    aspect's row, a column per measure, on one scale."""
    return TableBlock(
        section="scorecard",
        name="scorecard",
        title="deepseek-v4.1-flash by aspect",
        shared_scale=True,
        columns=[
            {"key": "aspect", "header": "Aspect"},
            *({"key": key, "header": header} for key, header, _ in cells),
        ],  # type: ignore[list-item]
        order="aspect by aspect",
        rows=[{"aspect": TableCell(value="Research quality"), **{key: cell for key, _, cell in cells}}],
        total_rows=1,
    )


def _track(drawn: str) -> list[tuple[float, float]]:
    """Each glyph's track, its two ends' x, in the order drawn."""
    return [
        (float(start), float(end))
        for start, end in re.findall(r'class="viz-track" x1="([\d.]+)" y1="[\d.]+" x2="([\d.]+)"', drawn)
    ]


def test_a_scale_of_levels_never_runs_below_zero_where_its_measure_cannot() -> None:
    """Discodon's re-shoot of host pass 5: four counts that each read 0 on a one-arm scorecard were drawn "from -1 to 1",
    a count no arm can have. A scale of levels (no cell a difference) whose one figure is zero is drawn from 0, money
    floors at 0 though a host's interval on it reaches below, and its glyph's whisker stops at the track; a difference
    still reads either side of zero, and a level whose own interval reaches below zero (a measure not declared
    non-negative) is drawn where it reaches."""
    zeros = _quality_scorecard(
        ("pass", "Pass rate (%)", TableCell(value=0.0, low=0.0, high=20.4)),
        *(
            (key, header, TableCell(value=0.0, low=0.0, high=0.0))
            for key, header in (("items", "Items delivered"), ("notices", "Failed-reply notices"))
        ),
    )
    (line,) = _AXIS.findall(grid_html(zeros))
    assert html.unescape(line).split(" · ")[1] == "Items delivered and Failed-reply notices drawn from 0 to 1", line
    money = _quality_scorecard(("spend", "Spend", TableCell(value=0.4, low=-0.2, high=1.0, unit="usd", decimals=1)))
    drawn = grid_html(money)
    (line,) = _AXIS.findall(drawn)
    assert html.unescape(line).startswith("Spend drawn from $0 to "), line
    ((start, _),) = _track(drawn)
    whisker = re.search(r'class="viz-whisker" x1="([\d.]+)"', drawn)
    assert whisker is not None and float(whisker.group(1)) >= start, "the whisker stops at the track's floor"
    change = _quality_scorecard(("change", "Change (pts)", TableCell(value=0.0, low=0.0, high=0.0, zero_line=True)))
    (line,) = _AXIS.findall(grid_html(change))
    assert html.unescape(line) == "Each column drawn from -1.0 to 1.0 pts", line
    signed = _quality_scorecard(("fields", "Stripped fields per delivery", TableCell(value=4.5, low=-4.9, high=13.9)))
    (line,) = _AXIS.findall(grid_html(signed))
    assert html.unescape(line) == "Each column drawn from -4.9 to 13.9", line


def test_a_scorecard_column_of_measured_figures_reads_at_one_precision() -> None:
    """Discodon's re-shoot of host pass 5: a one-arm scorecard printed a mean of ``2.737`` items beside counts read
    whole. A column of measured figures (neither a rate nor money) reads at its measure's one precision, the one its
    intervals set, so a mean turn time of 99.2345 ms beside one of 100 ms, each about ±27 ms, reads ``99`` and
    ``100``, its interval in its one form."""
    shape = shape_of(
        "measured",
        [
            AspectTrials(aspect, [ArmTrials("solo", passing=15, latency_ms=first)], margin=None)
            for aspect, first in (("billing", 1.2345), ("tone", 2.0))
        ],
        control=False,
        question="How long does solo take?",
    )
    page = shape.page()
    (scorecard,) = [block for block in page.blocks if isinstance(block, TableBlock) and block.name == "scorecard"]
    # One arm's matrix is turned, a row per measure (visual pass V2): the measure's rows are its column of figures, and
    # its interval's ends read in its value's form, as its glyph's title did.
    rows = [row for row in scorecard.rows if row["measure"].text == "Turn time"]
    # The control: 99.2345 is held at the measure's whole places, as its cell prints it.
    assert [row["value"].value for row in rows] == [99.0, 100.0]
    assert [row["value"].text for row in rows] == ["99", "100"]
    assert rows[0]["interval"].text == "72 to 127", rows[0]["interval"].text
    assert all(re.fullmatch(r"\d+ to \d+", row["interval"].text) for row in rows), [
        row["interval"].text for row in rows
    ]


def test_a_forest_s_end_label_never_reads_coarser_than_its_note(kairo_document: str) -> None:
    """P34: a note too wide for the margin (``-3 pts not separated``) once fell back to the axis's whole-number format
    (``-3``, ``-8`` beside ``-7.6`` in the prose). The end label is the note, wrapped at its spaces where it does not
    fit, on the drawn page (the kit) and on the static one."""
    block = ChartBlock(
        section="memo",
        chart="forest",
        title="t",
        data={  # type: ignore[arg-type]
            "rows": [
                {"label": "A", "value": -7.581, "lo": -12.85, "hi": -1.804, "note": "-7.6 pts worse"},
                {"label": "B", "value": -3.01, "lo": -7.392, "hi": 2.64, "note": "-3.0 pts not separated"},
            ],
            "domain": [-16, 4],
            "fmt": "+num:0",
            "level": "99.69%",
        },
    )
    drawn = chart_drawing(block)
    assert drawn is not None
    labels = [
        " ".join(re.findall(r"<tspan[^>]*>([^<]*)</tspan>", spans))
        for spans in re.findall(r"<text[^>]*>((?:<tspan[^>]*>[^<]*</tspan>)+)</text>", drawn)
    ]
    # Each end label is its whole note, on two lines at most where it does not fit.
    assert "-7.6 pts worse" in labels and "-3.0 pts not separated" in labels, labels
    assert all(len(re.findall(r"<tspan", spans)) <= 2 for spans in re.findall(r"<text[^>]*>(.*?)</text>", drawn))
    assert not {"-8", "-3"} & set(labels), "an end label fell back to the axis's whole numbers"
    assert "kWrap(r.note," in KIT_JS, "the kit wraps a forest's note rather than falling back to its format"


#: A change chart of bars whose first note is wider than the margin's 30% at a phone's width: the static page drew
#: ``-8`` in its place, the chart's ``+num:0`` value.
LONG_NOTE_BARS = {
    "rows": [
        {
            "label": "Persona holds character",
            "value": -7.6,
            "note": "-7.6 pts worse than the margin, held on the floor",
        },
        {"label": "Reply cites policy", "value": 4.5, "note": "+4.5 pts not separated"},
    ],
    "domain": [-9, 6],
    "fmt": "+num:0",
}


def test_a_bar_s_note_that_does_not_fit_is_wrapped_never_swapped_for_its_rounded_value() -> None:
    """A note is printed whole (GENERATOR §5.2): at each of the static page's widths the long note reads whole on two
    lines at most, and no bar's end reads the rounded value the note replaced (``-8``, ``+5``)."""
    drawn = chart_drawing(ChartBlock(section="memo", chart="bars", title="t", data=LONG_NOTE_BARS))  # type: ignore[arg-type]
    assert drawn is not None
    texts = re.findall(r"<text[^>]*>((?:<tspan[^>]*>[^<]*</tspan>)+)</text>", drawn)
    labels = [" ".join(re.findall(r"<tspan[^>]*>([^<]*)</tspan>", spans)) for spans in texts]
    widths = drawn.count("<svg")
    for row in LONG_NOTE_BARS["rows"]:  # type: ignore[attr-defined]
        assert labels.count(row["note"]) == widths, (row["note"], labels)
    assert not {"-8", "+5"} & set(labels), labels
    assert all(len(re.findall(r"<tspan", spans)) <= 2 for spans in texts)


def _kairo_text() -> str:
    """The Kairo prototype-5 document's page as a reader who cannot see the picture reads it (``doc_page_text``): what
    the editor and the scoring judge read."""
    record = json.loads((REPORT_DOCS / "prototype5" / "kairo" / "record.json").read_text())
    page = build(
        ReportDoc.model_validate(record["doc"]),
        ReportInput.model_validate(snapshot_of("prototype5/kairo").report_input),
        strict=False,
        findings=[ReportFinding.model_validate(finding) for finding in record.get("findings", [])],
        draft=record["ending"] == "stored_with_findings",
    )
    return doc_page_text(page)


#: The retired words for an undecided reading (owner ruling Q2).
_RETIRED = re.compile(r"\bnot separated\b", re.IGNORECASE)


def test_a_judged_dimension_drawn_as_a_chart_is_named_with_a_capital_and_keeps_its_case_inside_a_sentence() -> None:
    """Kairo's judged dimensions declare no label, so each is named by its humanized key ("persona holds character").
    Where the name is the label (its forest's and its table's title, series and cells) it starts with a capital, as
    every label on a written page does; set inside a sentence (a compare chart's note, "... on persona holds
    character") it keeps its own case."""
    held = ReportInput.model_validate(snapshot_of("prototype5/kairo").report_input)
    key = "persona.holds_character"
    assert held.judged[key].label is None, "the control: the dimension declares no label"
    look = block_look(held, fact_arms(held))
    said: list[str] = []
    # A forest draws one unit (the dimension alone); a table holds the dimension beside another.
    for kind, measures in (("forest", [key]), ("table", [key, "__outcome__"])):
        query = ChartQuery.model_validate(
            {
                "source": "contrasts",
                "aspect": "",
                "measures": measures,
                "arms": [],
                "strata": [],
                "limit": 0,
                "transform": "none",
                "small_multiples_by": "none",
            }
        )
        spec, faults = complete_chart(
            PlannedChart(kind=kind, title="", query=query, reference=[], emphasis=""),
            held,
            "block",
            forms=PageForms.of(held),
        )
        assert not faults, faults
        drawn = resolve_chart(spec, look, title="", fallback_title="", forms=PageForms.of(held))
        said += re.findall(r"'([^']*holds character[^']*)'", str(drawn.chart.model_dump()))
    assert said, "the control: the charts name the dimension"
    assert not [text for text in said if "persona holds character" in text], said
    assert any(text.startswith("Persona holds character") for text in said), said
    arms = [arm.id for arm in held.arms]
    query = ChartQuery.model_validate(
        {
            "source": "compare",
            "aspect": "",
            "measures": [key],
            "arms": [arms[1], arms[0]],
            "strata": ["overall"],
            "limit": 0,
            "transform": "none",
            "small_multiples_by": "none",
        }
    )
    spec, faults = complete_chart(
        PlannedChart(kind="diverging_bars", title="", query=query, reference=[], emphasis=""),
        held,
        "block",
        forms=PageForms.of(held),
    )
    assert not faults, faults
    compared = resolve_chart(spec, look, title="", fallback_title="", forms=PageForms.of(held))
    notes = [note for note in compared.notes if "holds character" in note]
    assert notes, "the control: the compare chart's note names the dimension"
    assert all(" on persona holds character" in note for note in notes), notes


def test_a_written_page_s_tables_say_an_undecided_reading_in_the_glossary_s_words() -> None:
    """Owner ruling Q2, landed with rung 5's judge recorded again (2026-11-07): every table a written page draws, the
    corrected comparisons and a chart's values table among them, says an undecided reading ``cannot tell apart``, never
    ``not separated``. The writer's own sentences are its words, not the package's, so only the rows code lays out are
    held here."""
    text = _kairo_text()
    rows = [line for line in text.splitlines() if line.startswith("|")]
    assert not [row for row in rows if _RETIRED.search(row)], [row for row in rows if _RETIRED.search(row)][:3]
    assert any("| Cannot tell apart |" in row for row in rows), "the corrected comparisons' verdict"
    assert any(row.endswith(" cannot tell apart |") for row in rows), "a chart's note"


def test_a_forest_s_values_table_reads_each_figure_at_its_unit_s_one_precision() -> None:
    """A forest of changes in points prints its values and their intervals' ends at the pass rate's one precision on
    the page (number policy, rule 1: Kairo's half-width of 3.5 points reads them whole), never at the float's four
    figures (``-3.010 | -5.815 to +1.136``)."""
    held = ReportInput.model_validate(snapshot_of("prototype5/kairo").report_input)
    look = block_look(held, fact_arms(held))
    spec, faults = complete_chart(
        PlannedChart(
            kind="forest",
            title="",
            query=ChartQuery.model_validate(
                {
                    "source": "readings",
                    "aspect": "",
                    "measures": [],
                    "arms": [],
                    "strata": [],
                    "limit": 0,
                    "transform": "none",
                    "small_multiples_by": "none",
                }
            ),
            reference=[],
            emphasis="",
        ),
        held,
        "block",
        forms=PageForms.of(held),
    )
    assert not faults, faults
    drawn = resolve_chart(spec, look, title="", fallback_title="t", forms=PageForms.of(held))
    assert isinstance(drawn.chart, ChartBlock)
    _, rows = values_table(drawn.chart)
    figures = [number for row in rows for cell in row[1:3] for number in re.findall(r"[+-]?\d+\.\d+|[+-]?\d+", cell)]
    places = PageForms.of(held).measure(TRIAL_PASS_NAME, kind="trial").places
    assert places == 0
    assert figures and all(len(number.partition(".")[2]) == places for number in figures), rows


def test_a_gate_scatter_s_region_label_never_shares_a_row_with_the_budget_s_label() -> None:
    """A5: at 390 px the r3 page drew ``Meets both gates`` across the latency budget's label, both on the chart's top
    row, the region narrower than its label. Where the region is narrower than its label, the label wraps inside it,
    or sits under the budget's label, which keeps its own row; on the static page and in the kit alike."""
    block = ChartBlock(
        section="memo",
        chart="gate_scatter",
        title="t",
        data={  # type: ignore[arg-type]
            "points": [{"x": 184.0, "y": 90.7, "id": "a"}, {"x": 420.0, "y": 89.8, "id": "b"}],
            "x_domain": [56.0, 1627.5],
            "y_domain": [77.0, 95.0],
            "x_fmt": "ms",
            "y_fmt": "pct",
            "x_log": True,
            "reference": {"value": 90.0, "label": "Overall accuracy gate, at least 90%", "kind": "floor"},
            "x_reference": {"value": 250.0, "label": "Latency budget (p95) gate, at most 250 ms", "kind": "budget"},
            "region_label": "Meets both gates",
        },
    )
    drawn = chart_drawing(block)
    assert drawn is not None
    svgs = re.findall(r"<svg[^>]*>.*?</svg>", drawn, flags=re.DOTALL)
    assert len(svgs) == 3, "the static page draws the chart at three widths"
    for svg in svgs:
        region = re.search(r'<rect x="([\d.]+)" y="[\d.]+" width="([\d.]+)"[^>]*fill:var\(--good-wash\)', svg)
        assert region is not None
        left, room = float(region.group(1)), float(region.group(2)) - 16
        label = re.search(r'<text x="[\d.]+" y="([\d.]+)"[^>]*class="lbl"[^>]*>(.*?)</text>', svg)
        assert label is not None
        lines = re.findall(r"<tspan[^>]*>([^<]*)</tspan>", label.group(2)) or [label.group(2)]
        assert " ".join(lines) == "Meets both gates"
        budget = re.search(r'<text x="([\d.]+)" y="([\d.]+)" text-anchor="start" class="tick">(.*?)</text>', svg)
        assert budget is not None and "Latency budget" in budget.group(3)
        budget_rows = len(re.findall(r"<tspan", budget.group(3))) or 1
        budget_bottom = float(budget.group(2)) + 3 + (budget_rows - 1) * 13
        # Either every line fits the region (the bold fallback face's 7.44 px a character), ending left of the
        # budget's label, or the label starts under the budget label's last row.
        fits = all(len(line) * 7.44 <= room for line in lines) and left + 8 + max(map(len, lines)) * 7.44 < float(
            budget.group(1)
        )
        assert fits or float(label.group(1)) - 10 >= budget_bottom, (svg[:200], lines)
    narrow = svgs[0]
    assert narrow.count("<tspan") > 0 and "Meets both</tspan>" in narrow, "at its narrowest the region label wraps"
    assert "RW(nx)>room" in KIT_JS and "ly=xr[3]+14" in KIT_JS, "the kit wraps and moves the region label alike"


def test_the_full_print_states_its_figures_as_the_quiet_print_and_the_page_do(rungs: dict[str, _Rung]) -> None:
    """P43, A4: the full print (``render(full=True)``) wrote ``mean 0.8333``, ``95% CI 0.5687-1`` and a total of
    ``$0.00413`` beside the page's ``$0.0041``. It writes every figure as the quiet print does."""
    rung = rungs["rung3"]
    full = rung.result.render(full=True)
    quiet = str(rung.result)
    for line in quiet.splitlines():
        if line.startswith("  total spend: ") or line.strip().startswith(("CAD:", "Accuracy:", "Number score:")):
            assert line.strip() in full, line
    assert not re.search(r"(?<!\$)\b0\.\d{4}\b|CI \d|: mean 0\.|\$0\.00\d{3}\b", full.split("\n# ")[0])
    assert "Test cost | v1 $0.0041; v2 $0.0063" in full and "  total spend: $0.0041, the results' cost" in full


def test_a_scorecard_glyph_s_title_and_scale_read_in_the_cells_form() -> None:
    """A change cell in points under a ``(pts)`` header reads its glyph's title and its scale in points at one decimal,
    the margin's ends ticked, the line under the table at the data's own ends."""
    block = TableBlock(
        section="scorecard",
        name="scorecard",
        title="By aspect",
        shared_scale=True,
        columns=[{"key": "aspect", "header": "Aspect"}, {"key": "arm:v2", "header": "v2 (pts)"}],  # type: ignore[list-item]
        order="aspect by aspect",
        rows=[
            {
                "aspect": TableCell(value="classify"),
                "arm:v2": TableCell(value=16.7, low=-7.83, high=35.41, margin=5.0, zero_line=True),
            }
        ],
        total_rows=1,
    )
    drawn = grid_html(block)
    # The margin is declared, so it reads as declared (number rule 4), never at the cells' tenth.
    assert "<title>16.7 pts, interval -7.8 to +35.4 pts, margin ±5 pts</title>" in drawn
    assert '<p class="axis">Each column drawn from -7.8 to 35.4 pts</p>' in drawn
    assert drawn.count('class="viz-margin"') == 2


# --- one tag and one colour per arm ---------------------------------------------------------------------------------------


#: Twelve cases each arm answers, by number.
_CASES = [{"n": index} for index in range(12)]


def _answering(misses: int) -> Any:
    """A runner that answers right on every case past its first ``misses``, at a cost under a cent."""

    async def answer(case: dict[str, int]) -> Answer:
        return Answer("right" if case["n"] >= misses else "wrong", model="m-1", cost_usd=2e-5)

    return answer


def _right(case: dict[str, int], answer: str) -> bool:
    """Whether the answer is right."""
    return answer == "right"


#: Each campaign's arms, the control first, in the order it declares them: in the three-arm one ``zeta`` is declared
#: before ``alpha``, which misses fewer cases.
_CAMPAIGNS: dict[str, dict[str, int]] = {
    "two arms": {"production": 4, "zulu": 2},
    "three arms": {"production": 4, "zeta": 6, "alpha": 1},
}


def _code_page_tags(page: CampaignReport) -> dict[str, tuple[str, str | None]]:
    """Each arm's tag and colour on the campaign's page, read off every aspect's outcome bar."""
    worn: dict[str, set[tuple[str, str | None]]] = {}
    for chart in _charts(page):
        if chart.chart == "outcome_bar":
            for bar in chart.bars:
                assert bar.tag is not None
                worn.setdefault(bar.arm, set()).add((bar.tag, arm_token(bar.tag)))
    assert all(len(held) == 1 for held in worn.values()), worn
    return {arm: next(iter(held)) for arm, held in worn.items()}


@pytest.mark.parametrize("name", sorted(_CAMPAIGNS))
def test_an_arm_is_one_tag_and_one_colour_on_the_code_page_the_aspect_page_and_the_document_page(name: str) -> None:
    """P40: Kairo's arms were ``A · 1017, B · qwen3.8-flash`` on the code page and the reverse on the document page.
    Every page tags arms in one order, the campaign's declared arms first, so the letters and the colours agree."""
    declared = _CAMPAIGNS[name]
    result = asyncio.run(
        Eval(_CASES, {arm: _answering(misses) for arm, misses in declared.items()}, [_right], k=1).run(store=":memory:")
    )
    code = _code_page_tags(campaign_report(result.host, result.campaign_id, result.scope_id))
    bundle = assemble_campaign_bundle(result.host, result.campaign_id, result.scope_id)
    document = {arm.id: (arm.tag, arm_token(arm.tag)) for arm in fact_arms(report_input(bundle))}
    # The aspect's own page, as the eval's result serves it (``result.report``, the page ``print`` and ``--write``
    # read), tagged by the campaign's order.
    aspect_page = {
        bar.arm: (bar.tag, arm_token(bar.tag))
        for chart in result.report.blocks
        if isinstance(chart, ChartBlock) and chart.chart == "outcome_bar"
        for bar in chart.bars
        if bar.tag is not None
    }
    assert set(code) == set(declared) == set(document), (code, document)
    assert code == document, (code, document)
    assert aspect_page == document, (aspect_page, document)
    # The candidates take A, B in the campaign's declared order, whatever order the scoreboard lists them in.
    assert [code[arm][0] for arm in declared] == ["P", "A", "B"][: len(declared)]


@pytest.mark.parametrize("name", sorted(_RUNGS))
def test_no_rate_on_a_rung_s_tables_reads_past_a_tenth_but_to_keep_off_a_boundary(
    rungs: dict[str, _Rung], name: str
) -> None:
    """A rate reads at its measure's places, at most a tenth of a point; it takes more only to keep off its floor or its
    ceiling (``99.97%``, never ``100.0%``), as an interval's level does (``99.69%``). A column may hold two measures'
    rates at their own places (owner ruling Q2, 2026-11-08), so the per-column rule of 2026-11-07 is retired."""
    percent = re.compile(r"(?<![\w.])(\d+)(?:\.(\d+))?%(?!\d)")
    for table in _tables(rungs[name].page):
        for column in table.columns:
            for row in table.rows:
                if column.key not in row:
                    continue
                for match in percent.finditer(row[column.key].text):
                    if match[2] is not None and len(match[2]) > 1:
                        assert not 1 <= float(f"{match[1]}.{match[2]}") <= 99, (table.name, column.key, match[0])
    # A zero written two ways: "0.0%" standing alone (never the tail of "100.0%" or "20.0%") beside a bare "0%".
    for text in _page_texts(rungs[name]):
        assert not (re.search(r"(?<![\d.])0\.0%", text) and re.search(r"(?<![\d.])0%", text)), "0.0% beside 0%"
