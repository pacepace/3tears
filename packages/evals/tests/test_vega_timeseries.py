"""The Vega-Lite renderer's ``timeseries`` arm, and its spec gate's rule that a line through categories states its order.

What the chart SAYS — its values as drawn, its disclosures — is the intent's, pinned in
``test_viz_timeseries.py``; this file pins how the renderer draws it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from threetears.evals.vega import check_spec, compile_chart
from packages.evals.tests.chart_examples import timeseries_ci as _ci
from packages.evals.tests.chart_examples import timeseries_payload as _payload


class TestTheArm:
    def test_the_time_axis_states_its_order(self) -> None:
        spec = compile_chart(
            "timeseries",
            _payload(
                basis="release",
                release_label="app_version",
                positions=["0.9", "0.10", "0.11"],
                series=[
                    {
                        "label": "a",
                        "points": [{"position": "0.9", "ci": _ci(1.0, 0.5)}, {"position": "0.10", "ci": _ci(2.0, 0.5)}],
                    }
                ],
                gaps=[{"series": "a", "position": "0.11", "reason": "the cell was not measured there"}],
            ),
        ).spec
        for layer in spec["spec"]["layer"]:
            assert layer["encoding"]["x"]["sort"] == ["0.9", "0.10", "0.11"]

    def test_a_gap_breaks_the_line_rather_than_bridging_it(self) -> None:
        chart = compile_chart("timeseries", _payload())
        segments = {
            row["position"]: row["segment"] for row in chart.spec["data"]["values"] if row.get("series") == "wide"
        }
        assert segments["2026-03-14"] != segments["2026-03-16"]
        narrow = {row["segment"] for row in chart.spec["data"]["values"] if row.get("series") == "narrow"}
        assert len(narrow) == 1

    def test_identity_rides_on_the_row_header_and_never_on_a_hue(self) -> None:
        spec = compile_chart("timeseries", _payload()).spec
        assert spec["facet"]["row"]["sort"] == ["narrow", "wide"]
        assert "color" not in json.dumps(spec["spec"])

    def test_the_compiled_spec_passes_the_gate(self) -> None:
        assert check_spec(compile_chart("timeseries", _payload()).spec) == []


# =============================================================================
# The policy rule
# =============================================================================


def _line(x: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": "latency over builds",
        "data": {"values": [{"p": "0.9", "v": 1}, {"p": "0.10", "v": 2}]},
        "mark": "line",
        "encoding": {
            "x": x,
            "y": {"field": "v", "type": "quantitative", "axis": {"title": "latency (ms)", "titleAngle": 0}},
        },
    }


class TestALineThroughCategoriesStatesItsOrder:
    @pytest.mark.parametrize("kind", ["nominal", "ordinal"])
    def test_an_unstated_categorical_order_is_refused(self, kind: str) -> None:
        violations = check_spec(_line({"field": "p", "type": kind}))
        assert any("no stated order" in violation for violation in violations)

    @pytest.mark.parametrize(
        "stated",
        [{"sort": ["0.9", "0.10"]}, {"scale": {"domain": ["0.9", "0.10"]}}],
        ids=["sort", "domain"],
    )
    def test_a_stated_order_is_admitted(self, stated: dict[str, Any]) -> None:
        assert check_spec(_line({"field": "p", "type": "ordinal"} | stated)) == []

    def test_a_quantitative_or_temporal_line_orders_itself(self) -> None:
        assert check_spec(_line({"field": "p", "type": "temporal", "axis": {"title": "when"}})) == []

    def test_the_rule_reads_an_encoding_the_frame_shares(self) -> None:
        spec = _line({"field": "p", "type": "ordinal"})
        spec["layer"] = [{"mark": spec.pop("mark")}]
        assert any("no stated order" in violation for violation in check_spec(spec))

    def test_a_point_mark_is_not_a_line(self) -> None:
        spec = _line({"field": "p", "type": "ordinal"}) | {"mark": "point"}
        assert check_spec(spec) == []
