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
