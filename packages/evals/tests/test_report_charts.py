"""The kit's chart kinds as a ``ChartBlock`` carries them: what the kit would refuse is refused first, a block's JSON
is the kit's contract, a stored report's three chart kinds still load, and every chart reads as a table.

- **Refused where the producer can be told.** A missing required field (on the chart, or on a panel as drawn), a
  colour that is not a token name, a format the kit does not name, intervals without their level, a matrix row of the
  wrong width, a path naming no point, a non-finite number: each is a validation error naming it, never a chart the
  kit would draw as nothing.
- **The kind picks the model.** A block names its kind and carries that kind's data; several kinds share field
  names, so the block reads its data by its kind, and refuses data of another kind or kit data on one of its own
  three.
- **The contract's names.** ``range_bars`` rows say ``from`` in the JSON; an unset field is left out; an explicit
  ``false`` stays.
- **Stored reports.** A chart block written before the kit kinds (no ``data``) loads as it was.
- **A table for every chart.** Markdown prints each chart's values table, a small multiple's rows under its panel.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.analysis import (
    BarsChart,
    ChartBlock,
    ForestChart,
    RangeBarsChart,
    report_markdown,
)
from packages.evals.tests.report_kit_support import report_of


def _block(chart: str, data: dict[str, Any]) -> ChartBlock:
    return ChartBlock(section="memo", finding=0, chart=chart, title="t", data=data)  # type: ignore[arg-type]


_BARS: dict[str, Any] = {"rows": [{"label": "a", "value": 2}], "domain": [0, 4], "fmt": "int"}


@pytest.mark.parametrize(
    ("chart", "data", "refusal"),
    [
        ("bars", {"rows": [{"label": "a", "value": 2}], "fmt": "int"}, "'domain' required"),
        ("bars", {**_BARS, "rows": [{"label": "a", "value": 2, "color": "#f00"}]}, "should match pattern"),
        ("bars", {**_BARS, "fmt": "percent"}, "should match pattern"),
        ("bars", {**_BARS, "rows": [{"label": "a", "value": 2, "lo": 1, "hi": 3}]}, '"level" is required'),
        ("bars", {**_BARS, "rows": [{"label": "a"}]}, "needs a value or segments"),
        ("bars", {**_BARS, "domain": [0, float("inf")]}, "finite number"),
        ("forest", {"rows": [{"label": "a", "value": 1, "lo": 0, "hi": 2}], "domain": [-3, 3], "fmt": "num"}, "level"),
        (
            "lines",
            {
                "categories": ["1", "2"],
                "series": [{"name": "s", "values": [1, 2], "lo": [0, 1], "hi": [2, 3]}],
                "domain": [0, 4],
                "fmt": "num",
            },
            '"level" is required',
        ),
        (
            "gate_matrix",
            {
                "columns": [{"label": "A"}, {"label": "B"}],
                "rows": [{"label": "r", "cells": [{"status": "good", "label": "Pass"}]}],
            },
            "1 cells for 2 columns",
        ),
        (
            "gate_matrix",
            {
                "columns": [{"label": "A"}],
                "rows": [{"label": "r", "cells": [{"status": "good", "label": "Pass", "cost": 1}]}],
            },
            '"cost_fmt" is required',
        ),
        (
            "gate_matrix",
            {"columns": [{"label": "A"}], "rows": [{"label": "r", "cells": [{"status": "ok", "label": "Pass"}]}]},
            "status",
        ),
        (
            "scatter_frontier",
            {
                "points": [{"x": 1, "y": 2, "id": "a"}],
                "x_domain": [0, 2],
                "y_domain": [0, 4],
                "x_fmt": "num",
                "y_fmt": "num",
                "paths": [{"ids": ["a", "b"]}],
            },
            "not in `points`: b",
        ),
        (
            "gate_scatter",
            {
                "points": [{"x": 1, "y": 2}],
                "x_domain": [0, 2],
                "y_domain": [0, 4],
                "x_fmt": "num",
                "y_fmt": "num",
                "reference": {"value": 1, "kind": "floor"},
            },
            "'x_reference' required",
        ),
        ("confusion", {"labels": ["a", "b"], "matrix": [[1, 0]]}, "2 × 2"),
        (
            "dot_interval",
            {"domain": [0, 100], "fmt": "pct", "panels": [{"title": "P", "rows": [{"label": "a", "value": 50}]}]},
            "panel 'P': 'level' required",
        ),
    ],
)
def test_what_the_kit_would_refuse_is_refused_first(chart: str, data: dict[str, Any], refusal: str) -> None:
    with pytest.raises(ValidationError, match=refusal):
        _block(chart, data)


def test_a_panel_takes_the_chart_s_fields_and_a_text_panel_needs_none() -> None:
    data = {
        "domain": [0, 100],
        "fmt": "pct",
        "level": "95%",
        "panels": [
            {"title": "P", "rows": [{"label": "a", "value": 50, "lo": 40, "hi": 60}]},
            {"title": "Q", "subtitle": "Nothing to draw here."},
        ],
    }
    headers, rows = _block("dot_interval", data).values_table()
    assert headers == ["Panel", "Row", "Value", "95% interval"]
    assert rows == [["P", "a", "50%", "40% to 60%"], ["Q", "Nothing to draw here.", "", ""]]


def test_the_block_reads_its_data_as_its_own_kind_and_refuses_another() -> None:
    block = _block("diverging_bars", _BARS)
    assert isinstance(block.data, BarsChart) and block.kit_chart() == ("diverging_bars", _BARS)
    with pytest.raises(ValidationError, match="carries the kit's forest data"):
        ChartBlock(section="memo", chart="forest", title="t", data=BarsChart.model_validate(_BARS))
    with pytest.raises(ValidationError, match="carries the kit's bars data"):
        ChartBlock(section="memo", chart="bars", title="t")
    with pytest.raises(ValidationError, match="not kit data"):
        ChartBlock(section="memo", chart="cost_quality", title="t", data=BarsChart.model_validate(_BARS))


def test_the_json_names_what_the_contract_names_and_keeps_an_explicit_false() -> None:
    ranges = RangeBarsChart.model_validate(
        {"groups": [{"label": "g", "rows": [{"label": "a", "from": 1.2, "to": 2.5}]}], "domain": [0, 3], "fmt": "s"}
    )
    assert ranges.groups[0].rows[0].from_ == 1.2
    assert ranges.kit_data()["groups"] == [{"label": "g", "rows": [{"label": "a", "from": 1.2, "to": 2.5}]}]
    bars = BarsChart.model_validate({**_BARS, "rows": [{"label": "a", "value": 2, "muted": False}]})
    assert bars.kit_data()["rows"] == [{"label": "a", "value": 2, "muted": False}]
    stored = _block("range_bars", ranges.kit_data())
    assert ChartBlock.model_validate_json(stored.model_dump_json()).kit_chart() == stored.kit_chart()


def test_a_chart_block_stored_before_the_kit_kinds_loads_as_it_was() -> None:
    stored = {
        "kind": "chart",
        "section": "scoreboard",
        "chart": "outcome_bar",
        "title": "Outcomes",
        "bars": [{"arm": "a", "segments": [{"state": "passed", "label": "passed", "count": 3}]}],
        "rows": [],
        "points": [],
        "unit": "",
        "zero_line": False,
        "margin": None,
    }
    block = ChartBlock.model_validate(stored)
    assert block.data is None and block.bars[0].total == 3
    assert block.kit_chart() is not None and block.kit_chart()[0] == "bars"  # type: ignore[index]


def test_a_table_spells_each_value_by_the_number_rule_not_the_axis_s_rounding() -> None:
    forest = ForestChart.model_validate(
        {
            "rows": [{"label": "A", "value": -7.62, "lo": -12.9, "hi": -2.4, "note": "worse"}],
            "domain": [-16, 4],
            "fmt": "+num:0",
            "level": "99.69%",
            "reference": {"value": -10, "label": "margin", "kind": "margin"},
        }
    )
    assert forest.values_table() == (
        ["Row", "Change", "99.69% interval", "Note"],
        [["A", "-7.62", "-12.9 to -2.4", "worse"], ["margin", "-10", "", ""]],
    )
    stacked = BarsChart.model_validate(
        {
            "rows": [{"label": "arm", "segments": [{"value": 3, "color": "--s1"}, {"value": 1, "color": "--s2"}]}],
            "domain": [0, 4],
            "fmt": "usd:0",
            "legend": [{"label": "calls", "color": "--s1"}, {"label": "retries", "color": "--s2"}],
        }
    )
    assert stacked.values_table() == (["Row", "Value", "Parts"], [["arm", "$4", "calls $3 · retries $1"]])


def test_markdown_prints_every_chart_s_values_table() -> None:
    panels = {
        "domain": [0, 100],
        "fmt": "pct",
        "columns": [{"label": "x"}],
        "panels": [{"title": "P", "rows": [{"label": "r", "cells": [{"value": 50, "n": 1200}]}]}],
    }
    report = report_of(
        [
            {"kind": "chart", "section": "memo", "finding": 0, "chart": "bars", "title": "Spend", "data": _BARS},
            {"kind": "chart", "section": "memo", "finding": 0, "chart": "heatmap", "title": "Cells", "data": panels},
        ]
    )
    markdown = report_markdown(report)
    assert "**Spend**\n\n| Row | Value |\n" in markdown and "| a | 2 |" in markdown
    assert "| Panel |  | x |" in markdown and "| P | r | 50% (n = 1200) |" in markdown


def test_a_range_bar_s_start_is_spelled_from_in_a_stored_report_as_in_the_kit_s_data() -> None:
    """A stored report spells a range row's start ``from``, as the kit's contract and ``kit_data`` do; one stored as
    ``from_`` before still loads."""
    data = {"domain": [0, 4], "fmt": "s", "groups": [{"label": "g", "rows": [{"label": "a", "from": 1, "to": 2}]}]}
    block = ChartBlock(section="memo", finding=0, chart="range_bars", title="t", data=data)
    stored = report_of([block]).to_dict()
    (row,) = next(b for b in stored["blocks"] if b.get("chart") == "range_bars")["data"]["groups"][0]["rows"]
    assert "from" in row and "from_" not in row
    assert block.kit_chart() == ("range_bars", data)
    old = {**data, "groups": [{"label": "g", "rows": [{"label": "a", "from_": 1, "to": 2}]}]}
    assert RangeBarsChart.model_validate(old) == RangeBarsChart.model_validate(data)
