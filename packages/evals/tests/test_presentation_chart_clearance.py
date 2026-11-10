"""A chart's labels sit clear of its drawn lines, its interval marks read at 3:1, and a table holds an interval on one
line, at every width a host gives the page.

Discodon's C5b re-shoot (2026-11-10, item 8), on the Character v2 written report and two SearXNG code pages:

- the zero line struck through the margin's label ("margin -15 pts; shaded: no worse than the margin") at 1104 and
  688 px, and at 688 px through a forest's head ("95% intervals (adjusted for 10 comparisons)") and its groups'
  headings;
- at 688 px the scorecard broke its intervals across lines ("0 to / 43%", "$25 to / $34") while its Value column had
  room;
- in light mode the cost chart's interval read faint: drawn at .35 opacity, 1.6:1 against the page, where a graphical
  mark takes 3:1 (WCAG 1.4.11), as the kit's stylesheet states an arm's colour does.

So a vertical line (an axis, the zero line, a reference line, a grid line) stops short of a chart's head, a reference's
label, a group's heading and a bar's note where it would cross one and goes on past it; a group's heading wider than a
phone's chart wraps rather than running off its edge, and a head wider than what the plot leaves it moves right to
start inside the chart; a dot's interval is drawn in its arm's colour at full strength; and a cell holding an interval
holds it on one line. ``scripts/check.py`` now fails each of these, and passes the same charts drawn from the shapes
the re-shoot found, in the kit and on the static page, at 1200, 1104, 688 and 390 px, light and dark.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from threetears.evals.analysis import report_html
from threetears.evals.analysis.report.chart_svg import DRAWERS
from threetears.evals.analysis.report.charts import DotIntervalChart, ForestChart
from threetears.evals.analysis.report.html_parts import table_html
from threetears.evals.analysis.report.model import TableBlock, TableCell, TableColumn

from packages.evals.tests.report_kit_support import chromium_unavailable, render_checked, report_of, rendered

#: The control and the candidate as the Character v2 page names them.
CONTROL, CANDIDATE = "P · deepseek-v4.1-flash", "A · qwen3.8-flash"

#: The page's first chart: one change against a margin, its label wider than the room beside the margin's line.
MARGIN: dict[str, Any] = {
    "rows": [{"label": CANDIDATE, "value": 10, "lo": -29, "hi": 58, "note": "cannot tell apart"}],
    "domain": [-30, 60],
    "fmt": "+pts",
    "level": "95%",
    "reference": {"value": -15, "label": "margin -15 pts; shaded: no worse than the margin", "kind": "margin"},
    "zero_label": f"no change vs {CONTROL} (70%, 7 of 10 trials)",
    "better": "right",
}

#: Its third: each judged dimension's change under its heading, at a level glossed with its correction.
GROUPED: dict[str, Any] = {
    "rows": [
        {
            "label": CANDIDATE,
            "value": value,
            "lo": lo,
            "hi": hi,
            "note": f"{value:+d} pts cannot tell apart" if value else "0 pts cannot tell apart",
            "group": group,
        }
        for value, lo, hi, group in (
            (0, -49, 49, "Persona holds character · deepseek-v4.1-flash 80%"),
            (10, -45, 60, "Persona booth stays offstage · deepseek-v4.1-flash 90%"),
            (10, -45, 68, "Pass rate · deepseek-v4.1-flash 70%"),
        )
    ],
    "domain": [-50, 70],
    "fmt": "+pts",
    "level": "95% (adjusted for 10 comparisons)",
    "zero_label": f"no change vs {CONTROL}",
    "better": "right",
}

#: Its cost chart: one paired change in dollars, its interval a dot strip's.
COST: dict[str, Any] = {
    "rows": [{"label": "Overall", "value": -1.43, "lo": -4.99, "hi": 2.14, "note": "-$1.43"}],
    "domain": [-6, 4],
    "fmt": "+usd",
    "level": "99.5%",
    "zero_label": f"no change vs {CONTROL} ($3.69)",
    "better": "left",
}

#: Bars drawn as dots, grouped, against a margin: the bar chart's own head, reference label and headings.
DOTS: dict[str, Any] = {
    "rows": [
        {
            "label": CANDIDATE,
            "value": 10,
            "lo": -45,
            "hi": 68,
            "group": "Persona holds character · deepseek-v4.1-flash 80%",
        },
        {"label": "B · other-model", "value": -5, "lo": -40, "hi": 30, "group": "Pass rate · deepseek-v4.1-flash 70%"},
    ],
    "domain": [-50, 70],
    "fmt": "+pts",
    "level": "95% (adjusted for 10 comparisons)",
    "dots": True,
    "zero_label": f"no change vs {CONTROL}",
    "better": "right",
    "reference": {"value": -15, "label": "margin -15 pts; shaded: no worse than the margin", "kind": "margin"},
}

#: The SearXNG scorecard's rows: a figure (its value and its text) beside its interval, a long measure name among them.
SCORECARD = [
    ("Pass rate", 0.0, "0% (0 of 15 trials, 5 cases)", "0 to 43%"),
    ("Measuring spend, all roles, per 1,000 replies", 29.0, "$29", "$25 to $34"),
    ("Items delivered", 2.7, "2.7", "0.4 to 5.0"),
    ("Production cost, per 1,000 replies", 20.0, "$20", "$16 to $25"),
    ("Salvaged research rate", 0.1, "10%", "0 to 38%"),
]


def _scorecard() -> TableBlock:
    columns = [
        TableColumn(key="aspect", header="Aspect"),
        TableColumn(key="measure", header="Measure"),
        TableColumn(key="value", header="Value"),
        TableColumn(key="interval", header="Interval", term="interval"),
        TableColumn(key="cases", header="Cases"),
    ]
    rows = [
        {
            "aspect": TableCell(value="Research quality"),
            "measure": TableCell(value=measure),
            "value": TableCell(value=value, value_text=text),
            "interval": TableCell(value=interval),
            "cases": TableCell(value=5),
        }
        for measure, value, text, interval in SCORECARD
    ]
    return TableBlock(
        section="contrasts",
        name="scorecard",
        title="deepseek-v4.1-flash by aspect",
        columns=columns,
        rows=rows,
        order="as built",
        total_rows=len(rows),
    )


def _page(*, static: bool) -> str:
    charts = [("forest", "Margin", MARGIN), ("forest", "Grouped", GROUPED), ("dot_interval", "Cost", COST)]
    charts.append(("bars", "Dots", DOTS))
    blocks: list[Any] = [
        {"kind": "chart", "section": "memo", "chart": kind, "title": title, "data": data}
        for kind, title, data in charts
    ]
    blocks.append(_scorecard())
    return report_html(report_of(blocks), static=static)


# =============================================================================
# The drawing, as the static page writes it
# =============================================================================


def test_a_dot_s_interval_is_drawn_in_its_arm_s_colour_at_full_strength() -> None:
    drawn, _ = DRAWERS["dot_interval"](DotIntervalChart.model_validate(COST), 640)
    (interval,) = re.findall(r'<line [^>]*style="stroke:var\(--s1\)[^"]*"', drawn)
    assert "opacity" not in interval and "stroke-width:2" in interval


@pytest.mark.parametrize("width", [360, 640, 960])
def test_the_zero_line_stops_short_of_each_group_s_heading_it_would_cross(width: int) -> None:
    """The grouped forest at each width the static page draws: wherever a group's heading reaches the zero line (as
    each did at 688 px, the 640 px drawing), the line is drawn in pieces around the heading's row, where at 5757aa3ae
    it ran through it from its label to the axis."""
    drawn, _ = DRAWERS["forest"](ForestChart.model_validate(GROUPED), width)
    axis = re.findall(r'<line x1="([\d.]+)" y1="([\d.]+)" x2="[\d.]+" y2="([\d.]+)" class="axis"', drawn)
    heads = re.findall(
        r'<text x="0" y="([\d.]+)" text-anchor="start" class="lbl-strong">(?:<tspan[^>]*>)?([^<]+)<', drawn
    )
    zero = float(axis[0][0])
    # A heading's width as the drawing estimates a bold 12 px label in the face a host's frame falls back to.
    reaching = [float(y) for y, text in heads if len(text) * 12 * 0.58 * 1.08 >= zero]
    assert len(heads) == 3 and (reaching or width == 960), heads
    for baseline in reaching:
        assert not [seg for seg in axis if float(seg[1]) < baseline and float(seg[2]) > baseline - 9], (baseline, axis)


# =============================================================================
# The interval cell
# =============================================================================


def test_an_interval_cell_holds_on_one_line_and_words_holding_to_do_not() -> None:
    table = _scorecard().model_copy(
        update={
            "rows": [
                *_scorecard().rows,
                *(
                    {
                        "aspect": TableCell(value="Research quality"),
                        "measure": TableCell(value=measure),
                        "value": TableCell(value=0.1, value_text="10"),
                        "interval": TableCell(value=interval),
                        "cases": TableCell(value=5),
                    }
                    for measure, interval in (
                        ("Switch to B1017", "-29 to +58 pts"),
                        ("Cost change", "-$4.99 to +$2.14"),
                    )
                ),
            ],
            "total_rows": len(SCORECARD) + 2,
        }
    )
    written = table_html(table)
    for _, _, _, interval in SCORECARD:
        assert f'<td class="interval"><span class="iv">{interval}</span></td>' in written, interval
    # A unit word after an interval may take the next line; the interval's ends and the word between them never do.
    assert '<td class="interval"><span class="iv">-29 to +58</span> pts</td>' in written
    assert '<td class="interval"><span class="iv">-$4.99 to +$2.14</span></td>' in written
    assert 'class="interval">Switch' not in written
    # A figure with its count after it, in a column of numbers, may set its count on the next line, held whole.
    assert '<td class="num split">0% <span class="aside">(0 of 15 trials, 5 cases)</span></td>' in written


# =============================================================================
# In the browser, at every width
# =============================================================================

#: Each table frame's width and how far it scrolls sideways.
_FRAMES = """() => [...document.querySelectorAll('.tbl')].filter(f => f.querySelector('table') && f.offsetParent)
  .map(f => ({head: f.querySelector('th').textContent.trim(), scrolls: f.scrollWidth - f.clientWidth}))"""


@pytest.mark.integration
@pytest.mark.parametrize("static", [True, False], ids=["static", "kit"])
def test_the_render_check_passes_the_re_shoot_s_charts_and_scorecard(tmp_path: Path, static: bool) -> None:
    """``scripts/check.py`` over the re-shoot's charts and scorecard, drawn by the kit and by the static page: no label
    crossed by a line, no interval mark under 3:1 and no interval broken across lines, at all eight renders; each fails
    at 5757aa3ae."""
    reason = chromium_unavailable()
    if reason is not None:
        pytest.skip(f"headless Chromium cannot launch: {reason}")
    page = tmp_path / "charts.html"
    page.write_text(_page(static=static), encoding="utf-8")
    checked, runs = render_checked(page, tmp_path / "shots")
    assert checked.returncode == 0 and len(runs) == 8, checked.stdout + checked.stderr


@pytest.mark.integration
def test_the_scorecard_holds_its_intervals_without_scrolling_sideways_at_688_px(tmp_path: Path) -> None:
    reason = chromium_unavailable()
    if reason is not None:
        pytest.skip(f"headless Chromium cannot launch: {reason}")
    page = tmp_path / "scorecard.html"
    page.write_text(report_html(report_of([_scorecard()])), encoding="utf-8")
    frames, errors = rendered(page, _FRAMES, width=688)
    assert errors == []
    assert frames == [{"head": "Aspect", "scrolls": 0}], frames


@pytest.mark.integration
def test_the_render_check_fails_a_line_through_a_label_a_faint_mark_and_a_broken_interval(tmp_path: Path) -> None:
    """The check itself: the same page with the zero line let run through every label, the dot's interval faded to
    .35 and the interval cells let wrap, as at 5757aa3ae, fails on each."""
    reason = chromium_unavailable()
    if reason is not None:
        pytest.skip(f"headless Chromium cannot launch: {reason}")
    good = _page(static=False)
    for name, rule, said in (
        ("faint", ".chart svg line[style*='stroke-width:2']{opacity:.35}", "contrast: a mark"),
        ("wrapped", "td .iv{white-space:normal!important;word-spacing:40rem}", "wrapped:"),
    ):
        bad = tmp_path / f"{name}.html"
        bad.write_text(good.replace("</style>", f"{rule}</style>", 1), encoding="utf-8")
        checked, _ = render_checked(bad, tmp_path / "shots")
        assert checked.returncode == 1 and said in checked.stdout, (name, checked.stdout)
    crossed = tmp_path / "crossed.html"
    # A line drawn across the margin chart's head, as the zero line was drawn through it.
    struck = '<line class="axis" x1="0" y1="0" x2="2000" y2="2000" style="stroke:var(--muted)"></line>'
    drawn = _page(static=True).replace("</svg>", f"{struck}</svg>", 1)
    crossed.write_text(drawn, encoding="utf-8")
    checked, _ = render_checked(crossed, tmp_path / "shots")
    assert checked.returncode == 1 and "crossed:" in checked.stdout, checked.stdout
