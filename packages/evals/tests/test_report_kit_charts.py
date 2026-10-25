"""Every chart the five reference memos draw is a ``ChartBlock``, and the block gives back the very data it was read
from.

The reference memos (``docs/eval-report-kit/references``) build each chart's data in page code from their input and
set it on a ``div.chart[data-kind]`` (``setChart``); the kit draws it. Once a reference has rendered, each such chart's
``data-chart`` JSON is read as the ``ChartBlock`` of its kind, and the block's own JSON (:meth:`ChartBlock.kit_chart`,
what the package's page writes into ``data-chart``) is the reference's, value for value. So the package can express
every reference chart, and its page hands the kit the same data, which the kit draws the same way.

Two things are compared as JSON values, not as text: a number is a number (``5`` and ``5.0`` are one value), and a
``null`` the reference wrote is a field it did not set, since the kit reads ``null`` and absent alike.

What the references draw outside the vocabulary is listed below with its reason, and nothing else is: a reference
chart with no ``data-kind`` is a failure here unless it is listed.

The references compute their data in the browser, so this drives headless Chromium and carries the ``integration``
marker; where Chromium cannot launch it skips and says why.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import pytest

from threetears.evals.analysis import KIT_CHARTS, ChartBlock
from packages.evals.tests.report_kit_support import KIT, chromium_unavailable, rendered, report_page

#: Each reference's built page, by name.
REFERENCES = sorted(p.stem for p in (KIT / "references").glob("*.html") if not p.name.endswith(".src.html"))

#: The kinds each reference draws: together, every kind the kit draws.
KINDS_DRAWN = {
    "r1-model-selection": {"bars", "gate_matrix", "scatter_frontier", "dot_interval", "range_bars"},
    "r2-config-sweep": {"bars", "diverging_bars", "gate_scatter", "examples"},
    "r3-classifier": {"bars", "lattice", "heatmap", "diff_heatmap", "confusion", "scatter_frontier"},
    "r4-tradeoff": {"bars", "diverging_bars", "lines", "scatter_frontier"},
    "r5-kairo-round2": {"bars", "diverging_bars", "forest", "examples"},
}

#: Reference charts drawn by page code rather than from a kind, each with why.
PAGE_CODE_CHARTS = {
    ("r2-config-sweep", "e-types"): (
        "the explore view's per-task comparison of the selected configuration against the recommended one (a dot for "
        "each, joined by a line): no kind draws it, and it is explore logic, which plan-declared controls replace"
    ),
}

#: Every top-level chart once the page has drawn (a small multiple's panel is a chart inside its host, and is not
#: one of its own): its id, its kind and its data, or no kind for a chart the page draws itself.
_CHARTS = """() => [...document.querySelectorAll('div.chart')].filter(e => !e.parentElement.closest('div.chart')).map(e => ({
  id: e.id, kind: e.dataset.kind || null, data: e.dataset.chart ? JSON.parse(e.dataset.chart) : null,
  drawn: e.children.length > 0}))"""


def _set(value: Any) -> Any:
    """``value`` without the ``null``s the reference wrote: the kit reads a ``null`` field as one not set."""
    if isinstance(value, dict):
        return {key: _set(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_set(item) for item in value]
    return value


def test_the_references_together_draw_every_kind_the_kit_draws() -> None:
    assert REFERENCES == sorted(KINDS_DRAWN)
    assert set().union(*KINDS_DRAWN.values()) == set(KIT_CHARTS)


@pytest.mark.integration
@pytest.mark.parametrize("name", REFERENCES)
def test_every_chart_a_reference_draws_is_a_chart_block_that_gives_back_its_data(name: str) -> None:
    reason = chromium_unavailable()
    if reason is not None:
        pytest.skip(f"headless Chromium cannot launch: {reason}")
    charts, errors = rendered(KIT / "references" / f"{name}.html", _CHARTS)
    assert errors == []
    page_code = {chart["id"] for chart in charts if chart["kind"] is None}
    assert page_code == {chart for reference, chart in PAGE_CODE_CHARTS if reference == name}
    kit_charts = [chart for chart in charts if chart["kind"] is not None]
    assert {chart["kind"] for chart in kit_charts} == KINDS_DRAWN[name]
    for chart in kit_charts:
        assert chart["drawn"], f"#{chart['id']} was not drawn"
        block = ChartBlock(section="memo", finding=0, chart=chart["kind"], title=chart["id"], data=chart["data"])
        assert block.kit_chart() == (chart["kind"], _set(chart["data"])), f"#{chart['id']} ({chart['kind']})"
        headers, rows = block.values_table()
        assert headers and rows and all(len(row) == len(headers) for row in rows), f"#{chart['id']}'s table"


@pytest.mark.integration
def test_a_package_page_hands_the_kit_the_data_its_blocks_carry(tmp_path: Path) -> None:
    """Every reference chart put on a package page is drawn by the kit from the same JSON, with no error."""
    reason = chromium_unavailable()
    if reason is not None:
        pytest.skip(f"headless Chromium cannot launch: {reason}")
    blocks = []
    for name in REFERENCES:
        charts, _ = rendered(KIT / "references" / f"{name}.html", _CHARTS)
        blocks += [
            ChartBlock(section="memo", finding=0, chart=chart["kind"], title=chart["id"], data=chart["data"])
            for chart in charts
            if chart["kind"] is not None
        ]
    page = tmp_path / "page.html"
    page.write_text(report_page(blocks), encoding="utf-8")
    drawn, errors = rendered(page, _CHARTS)
    assert errors == []
    assert [(chart["kind"], chart["data"]) for chart in drawn] == [block.kit_chart() for block in blocks]
    assert all(chart["drawn"] for chart in drawn)


#: Each chart figure's table: whether a reader sees it (wider than the visually-hidden 1px box) and whether its chart
#: was drawn.
_TABLES = """() => [...document.querySelectorAll('figure.fig')].map(f => ({
  figure: f.dataset.figure, seen: f.querySelector('.chart-values').getBoundingClientRect().width > 1,
  drawn: !!(f.querySelector('div.chart') && f.querySelector('div.chart').children.length)}))"""


@pytest.mark.integration
def test_a_chart_s_table_is_what_a_reader_sees_where_no_chart_is_drawn(tmp_path: Path) -> None:
    """With the kit's script, a drawn chart's table is visually hidden (a screen reader reads it) and a chart the kit
    does not draw (interval rows) shows its table; without the script, every figure shows its table."""
    reason = chromium_unavailable()
    if reason is not None:
        pytest.skip(f"headless Chromium cannot launch: {reason}")
    bars = {"rows": [{"label": "a", "value": 2}], "domain": [0, 4], "fmt": "int"}
    rows = [{"label": "a", "value": 1.0, "low": 0.5, "high": 1.5, "n_cases": 6}]
    markup = report_page(
        [
            {"kind": "chart", "section": "memo", "finding": 0, "chart": "bars", "title": "Bars", "data": bars},
            {"kind": "chart", "section": "memo", "finding": 0, "chart": "interval_rows", "title": "Rows", "rows": rows},
        ]
    )
    with_script, without_script = tmp_path / "with.html", tmp_path / "without.html"
    with_script.write_text(markup, encoding="utf-8")
    script = markup[markup.rindex("<script>") : markup.rindex("</script>") + len("</script>")]
    without_script.write_text(markup.replace(script, ""), encoding="utf-8")

    seen, errors = rendered(with_script, _TABLES)
    assert errors == []
    assert seen == [
        {"figure": "bars", "seen": False, "drawn": True},
        {"figure": "interval_rows", "seen": True, "drawn": False},
    ]
    seen, _ = rendered(without_script, _TABLES)
    assert seen == [
        {"figure": "bars", "seen": True, "drawn": False},
        {"figure": "interval_rows", "seen": True, "drawn": False},
    ]


# =============================================================================
# What the kit refuses, and what a reader sees then
# =============================================================================

#: Each refusal API.md lists, as a hand-written chart: its id, its kind, its raw ``data-chart`` and the words the
#: kit's console error gives as the reason. A bar chart's fields fill the rest so each case breaks one thing.
_BARS_OK = '"domain":[0,4],"fmt":"int","rows":[{"label":"a","value":2}]'
REFUSALS = [
    ("unknown-kind", "pie", "{}", "unknown kind; the kit draws bars"),
    ("inherited-name", "constructor", "{}", "unknown kind; the kit draws bars"),
    ("table-kind", "table", "{}", "a table is written as an HTML table"),
    ("bad-json", "bars", '{"rows":', "data-chart is not JSON"),
    ("not-an-object", "bars", "[1,2]", "data-chart must be a JSON object"),
    ("missing-field", "bars", '{"fmt":"int","rows":[{"label":"a","value":2}]}', '"domain" is required'),
    (
        "css-colour",
        "bars",
        '{"domain":[0,4],"fmt":"int","rows":[{"label":"a","value":2,"color":"--s1);background:url(x)"}]}',
        'color "--s1);background:url(x)" is not a token name',
    ),
    ("unknown-fmt", "bars", '{"domain":[0,4],"fmt":"pie","rows":[{"label":"a","value":2}]}', 'fmt "pie" is not one of'),
    (
        "inherited-fmt",
        "bars",
        '{"domain":[0,4],"fmt":"constructor","rows":[{"label":"a","value":2}]}',
        'fmt "constructor" is not one of',
    ),
    (
        "level-less-intervals",
        "bars",
        '{"domain":[0,4],"fmt":"int","rows":[{"label":"a","value":2,"lo":1,"hi":3}]}',
        'rows carry intervals, so "level" is required',
    ),
    (
        "level-less-forest",
        "forest",
        '{"domain":[-4,4],"fmt":"num","rows":[{"label":"a","value":1,"lo":0,"hi":2}]}',
        '"level" is required',
    ),
    (
        "unknown-pattern",
        "bars",
        '{"domain":[0,4],"fmt":"int","rows":[{"label":"a","segments":[{"value":2,"color":"--bad","pattern":"url(x)"}]}]}',
        'pattern "url(x)" is not one of',
    ),
    (
        "css-path-width",
        "scatter_frontier",
        '{"x_domain":[0,4],"y_domain":[0,4],"x_fmt":"num","y_fmt":"num","points":[{"x":1,"y":1,"id":"a"},'
        '{"x":2,"y":2,"id":"b"}],"paths":[{"ids":["a","b"],"width":"2;cursor:url(https://x/c.png),auto"}]}',
        'path width "2;cursor:url(https://x/c.png),auto" is not a finite number',
    ),
    (
        "css-path-opacity",
        "scatter_frontier",
        '{"x_domain":[0,4],"y_domain":[0,4],"x_fmt":"num","y_fmt":"num","points":[{"x":1,"y":1,"id":"a"},'
        '{"x":2,"y":2,"id":"b"}],"paths":[{"ids":["a","b"],"opacity":"1;cursor:url(x)"}]}',
        'path opacity "1;cursor:url(x)" is not a finite number',
    ),
    (
        "css-legend-opacity",
        "bars",
        "{" + _BARS_OK + ',"legend":[{"label":"l","mark":"line","opacity":"1;cursor:url(x)"}]}',
        'legend opacity "1;cursor:url(x)" is not a finite number',
    ),
    (
        "css-point-size",
        "scatter_frontier",
        '{"x_domain":[0,4],"y_domain":[0,4],"x_fmt":"num","y_fmt":"num","points":[{"x":1,"y":1,"size":"5;x:y"}]}',
        'point size "5;x:y" is not a finite number',
    ),
]

#: The page's own figures and its hand-written charts: whether each drew anything, and, for a figure, whether its
#: table is what a reader sees.
_REFUSED = """() => ({
  charts: [...document.querySelectorAll('div.chart.refusal')].map(e => ({id: e.id, drawn: e.children.length > 0})),
  figures: [...document.querySelectorAll('figure.fig')].map(f => ({
    seen: f.querySelector('.chart-values').getBoundingClientRect().width > 1,
    drawn: !!(f.querySelector('div.chart') && f.querySelector('div.chart').children.length)})),
  styled: [...document.querySelectorAll('[style*="url(x)"], [style*="cursor"]')].length,
})"""


def _with_charts(markup: str, charts: str) -> str:
    """A package page with hand-written chart elements put in before the kit's script, as a host's page would carry
    them."""
    at = markup.rindex("<script>")
    return markup[:at] + charts + markup[at:]


@pytest.mark.integration
def test_the_kit_refuses_each_bad_chart_draws_nothing_and_says_why(tmp_path: Path) -> None:
    """Every refusal kit/API.md lists draws nothing for its element and logs ``kit: chart "<kind>" #<id>: <reason>``;
    nothing the data tried to write into a style reaches one; and a package figure whose data the kit refuses shows
    its table, which is then what the figure shows."""
    reason = chromium_unavailable()
    if reason is not None:
        pytest.skip(f"headless Chromium cannot launch: {reason}")
    bars = {"rows": [{"label": "a", "value": 2, "color": "--s1"}], "domain": [0, 4], "fmt": "int"}
    markup = report_page(
        [{"kind": "chart", "section": "memo", "finding": 0, "chart": "bars", "title": "B", "data": bars}]
    )
    # The package's models refuse a colour that is not a token, so the figure's data is broken in the page itself.
    broken = markup.replace("&quot;--s1&quot;", "&quot;--s1);background:url(x)&quot;")
    assert broken != markup
    charts = "".join(
        f'<div class="chart refusal" id="{chart_id}" data-kind="{kind}" data-chart="{html.escape(data)}"></div>'
        for chart_id, kind, data, _ in REFUSALS
    )
    page = tmp_path / "refusals.html"
    page.write_text(_with_charts(broken, charts), encoding="utf-8")
    seen, errors = rendered(page, _REFUSED)
    assert seen["charts"] == [{"id": chart_id, "drawn": False} for chart_id, *_ in REFUSALS]
    for chart_id, kind, _, words in REFUSALS:
        logged = [error for error in errors if error.startswith(f'kit: chart "{kind}" #{chart_id}: ')]
        assert len(logged) == 1 and words in logged[0], (chart_id, errors)
    assert seen["figures"] == [{"seen": True, "drawn": False}], "the refused figure shows its table"
    assert [error for error in errors if re.match(r'kit: chart "bars" #chart-\d+: color "--s1\);', error)], errors
    assert len(errors) == len(REFUSALS) + 1, errors
    assert seen["styled"] == 0


@pytest.mark.integration
def test_a_null_panel_field_inherits_the_chart_s_value_in_the_kit_as_in_the_model(tmp_path: Path) -> None:
    """The models read ``null`` as unset, so a panel's ``null`` field inherits the chart's; the kit draws it so too:
    the panel whose ``reference`` is ``null`` draws the chart's floor, and a row whose ``group`` is ``null`` sits
    under no heading, as one with none."""
    reason = chromium_unavailable()
    if reason is not None:
        pytest.skip(f"headless Chromium cannot launch: {reason}")
    data: dict[str, Any] = {
        "domain": [0, 4],
        "fmt": "int",
        "reference": {"value": 3, "label": "floor 3", "kind": "floor"},
        "panels": [
            {"title": "own", "rows": [{"label": "a", "value": 2}]},
            {
                "title": "null",
                "reference": None,
                "rows": [{"label": "a", "value": 2, "group": None}, {"label": "b", "value": 1}],
            },
        ],
    }
    block = ChartBlock(section="memo", finding=0, chart="bars", title="t", data=data)
    _, rows = block.values_table()
    assert [row for row in rows if row[1] == "floor 3"] == [["own", "floor 3", "3"], ["null", "floor 3", "3"]]
    page = tmp_path / "nulls.html"
    chart = (
        f'<div class="chart refusal" id="nulls" data-kind="bars" data-chart="{html.escape(json.dumps(data))}"></div>'
    )
    page.write_text(_with_charts(report_page([]), chart), encoding="utf-8")
    probe = """() => [...document.querySelectorAll('#nulls .fig')].map(f => ({
      floors: f.querySelectorAll('line.floor').length, headings: f.querySelectorAll('text.lbl-strong').length}))"""
    seen, errors = rendered(page, probe)
    assert errors == []
    assert seen == [{"floors": 1, "headings": 0}, {"floors": 1, "headings": 0}]


# =============================================================================
# What the converted charts draw: a readable cost axis, and arms told apart
# =============================================================================

#: The tick labels of the page's scatter's x axis: those under the plot, which carry the currency sign.
_COST_TICKS = """() => [...document.querySelectorAll('div.chart[data-kind="scatter_frontier"] text.tick')]
  .map(t => t.textContent).filter(t => t.startsWith('$'))"""


@pytest.mark.integration
def test_a_cost_axis_at_a_real_reply_cost_draws_distinct_ticks_none_reading_as_zero(tmp_path: Path) -> None:
    """At $0.00042 and $0.00018 a reply, the drawn cost axis (per 1,000 replies) prints a distinct figure at each
    tick, none past the origin reading as zero."""
    reason = chromium_unavailable()
    if reason is not None:
        pytest.skip(f"headless Chromium cannot launch: {reason}")
    points = [
        {
            "arm": arm,
            "tag": tag,
            "short_name": arm,
            "cost": cost,
            "cost_low": cost * 0.9,
            "cost_high": cost * 1.1,
            "pass_rate": rate,
            "pass_rate_low": rate - 5,
            "pass_rate_high": rate + 5,
        }
        for arm, tag, cost, rate in (("control arm", "P", 0.00042, 80.0), ("cheap arm", "A", 0.00018, 70.0))
    ]
    block = {"kind": "chart", "section": "memo", "finding": 0, "chart": "cost_quality", "title": "Cost", "unit": "usd"}
    page = tmp_path / "cost.html"
    page.write_text(report_page([{**block, "points": points}]), encoding="utf-8")
    ticks, errors = rendered(page, _COST_TICKS)
    assert errors == []
    assert len(ticks) >= 3 and len(set(ticks)) == len(ticks), ticks
    assert all(re.search(r"[1-9]", tick) for tick in ticks[1:]), ticks


#: Each drawn row of the page's bar chart: the label the kit drew and the tag beside it.
_BAR_LABELS = """() => [...document.querySelectorAll('div.chart[data-kind="bars"] svg g')]
  .filter(g => g.querySelector('rect[data-tip]'))
  .map(g => ({label: g.querySelector('text').textContent, tag: (g.querySelector('text.tag') || {}).textContent || null}))"""


@pytest.mark.integration
@pytest.mark.parametrize("width", [1200, 390])
def test_arms_whose_labels_share_a_long_prefix_are_told_apart_on_the_drawn_outcome_bar(
    tmp_path: Path, width: int
) -> None:
    """Three arms named alike up to their reasoning level: at 1200 and at 390 px each drawn row reads apart from the
    others, by its short name or, where that does not fit, its tag alone; none is cut to a shared prefix."""
    reason = chromium_unavailable()
    if reason is not None:
        pytest.skip(f"headless Chromium cannot launch: {reason}")
    passed = [{"state": "passed", "label": "passed", "count": 4}]
    arms = [
        ("model=claude-haiku-5-5, reasoning=low (control)", "P", "control"),
        ("model=claude-haiku-5-5, reasoning=high", "A", "reasoning high"),
        ("model=claude-haiku-5-5, reasoning=medium", "B", "reasoning medium"),
    ]
    bars = [{"arm": arm, "tag": tag, "short_name": short, "segments": passed} for arm, tag, short in arms]
    page = tmp_path / "arms.html"
    block = {"kind": "chart", "section": "memo", "finding": 0, "chart": "outcome_bar", "title": "Outcomes"}
    page.write_text(report_page([{**block, "bars": bars}]), encoding="utf-8")
    drawn, errors = rendered(page, _BAR_LABELS, width=width)
    assert errors == []
    assert [row["tag"] for row in drawn] == ["P", "A", "B"]
    assert len({(row["label"], row["tag"]) for row in drawn}) == 3
    assert not any("…" in row["label"] for row in drawn), drawn
    assert all(row["label"] in {"", arm, short} for row, (arm, _, short) in zip(drawn, arms, strict=True)), drawn


#: The kit's arm naming over a control and thirty candidates: each arm's tag, in order.
_ARM_TAGS = """() => armLabels(
  [{id: 'c', role: 'control', spec: {model: 'm'}},
   ...Array.from({length: 30}, (_, i) => ({id: 'a' + i, role: 'challenger', spec: {model: 'm' + i}}))],
  [{k: 'model'}]).map(a => a.tag)"""


@pytest.mark.integration
def test_the_kit_tags_no_candidate_p_however_many_there_are(tmp_path: Path) -> None:
    """GENERATOR.md §6.1, in the kit as in the package (``arm_tags``): P is the control's, so thirty candidates are
    lettered A to O, Q to Z, then AA on; the sixteenth is Q, never P."""
    reason = chromium_unavailable()
    if reason is not None:
        pytest.skip(f"headless Chromium cannot launch: {reason}")
    page = tmp_path / "tags.html"
    page.write_text(report_page([]), encoding="utf-8")
    tags, errors = rendered(page, _ARM_TAGS)
    assert errors == []
    assert tags[0] == "P"
    assert "P" not in tags[1:] and len(set(tags)) == len(tags)
    assert tags[1:18] == [*"ABCDEFGHIJKLMNOQR"] and tags[26:] == ["AA", "AB", "AC", "AD", "AE"]
