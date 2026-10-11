"""A measure reads in one unit on every surface of a report: its tables, its charts, its words and its summary lines.

The decision surface chooses each measure's ruler once (:func:`~threetears.evals.analysis.surface_table.measure_rulers`,
over every cell's mean and every threshold and verdict of a bar on it). A distribution chart that chose its unit over
its own values — its interval ends among them — drew a latency with means of 900 and 950 ms and an interval reaching
1100 ms in s, beside a surface table stating it in ms; a 1500 ms bar threshold put the table in s while the chart,
whose values never reached a second, stayed in ms. Each surface below is held to the ruler instead.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.analysis.cells import cell_ref
from threetears.evals.analysis.prose_refs import render_reference
from threetears.evals.analysis.report.words import evidence_rows
from threetears.evals.analysis.summary import EvalSummary, RunMeasureSummary
from threetears.evals.analysis.surface_table import measure_rulers
from threetears.evals.analysis.viz.intent import chart_intent
from threetears.evals.analysis.viz.payloads import PayloadError
from threetears.evals.analysis.viz_refs import DistributionRef, build_viz_payload
from threetears.evals.kernel.analysis_measures import BarAdjudication, MeasureCollection, MeasureSummary
from threetears.evals.kernel.campaign import EvidenceRow
from threetears.evals.kernel.host.measures import MeasureRegistry
from threetears.evals.kernel.surface import DecisionSurface
from packages.evals.tests.test_surface_table import (
    CANDIDATE,
    INCUMBENT,
    LATENCY,
    analysis,
    cell,
    entry,
    measures,
    verdict,
)


def _latency(mean: float, high: float) -> MeasureSummary:
    """A latency summary over enough cases to draw a band, its interval reaching ``high``."""
    return MeasureSummary(
        population="scored",
        name=LATENCY,
        attribution_scope="end_to_end",
        n=18,
        n_independent=6,
        mean=mean,
        sem=40.0,
        ci_low=2 * mean - high,
        ci_high=high,
    )


def _surface(*, threshold: float | None = None) -> DecisionSurface:
    """Two arms whose latency means are 900 and 950 ms, the slower one's interval reaching 1100 ms."""
    fast = cell(INCUMBENT, n_cases=6, measures=MeasureCollection(measures=[_latency(900.0, 990.0)]))
    slow = cell(CANDIDATE, n_cases=6, measures=MeasureCollection(measures=[_latency(950.0, 1100.0)]))
    cells = sorted([fast, slow], key=lambda c: (c.variant_key, c.apparatus_class_id))
    bars = []
    if threshold is not None:
        bars.append(
            BarAdjudication(
                measure_id=LATENCY,
                threshold=threshold,
                direction="lower_is_better",
                source="declared",
                state="adjudicated",
                verdicts=sorted(
                    [verdict(fast, 900.0, 40.0, True), verdict(slow, 950.0, 40.0, True)],
                    key=lambda v: (v.variant_key, v.apparatus_class_id),
                ),
            )
        )
    return DecisionSurface(cells=cells, bars=bars, measures=measures())


def _chart_unit(surface: DecisionSurface) -> str:
    payload = build_viz_payload(
        DistributionRef(
            cells=[cell_ref(c.variant_key, c.apparatus_class_id) for c in surface.cells],
            measure_id=LATENCY,
            reading="measure",
        ),
        surface,
        [entry(INCUMBENT), entry(CANDIDATE)],
        measures=MeasureRegistry([]),
    )
    return chart_intent("distribution", payload).unit


class TestAChartDrawsOnTheMeasuresRuler:
    def test_an_interval_end_past_a_second_does_not_move_the_chart_off_the_tables_ms(self) -> None:
        surface = _surface()
        assert measure_rulers(surface)[LATENCY].unit == "ms"
        assert _chart_unit(surface) == "ms"

    def test_a_threshold_past_a_second_moves_the_chart_to_the_tables_s(self) -> None:
        surface = _surface(threshold=1500.0)
        assert measure_rulers(surface)[LATENCY].unit == "s"
        assert _chart_unit(surface) == "s"

    def test_a_payload_naming_no_ruler_still_chooses_over_its_own_values(self) -> None:
        """A chart a host builds by hand has no surface to take a ruler from, and chooses as it always did."""
        payload: dict[str, Any] = {
            "groups": [
                {"label": "a", "ci": {"mean": 950, "low": 800, "high": 1100, "level": 0.95, "variability": "cases"}}
            ],
            "unit": "ms",
        }
        assert chart_intent("distribution", payload).unit == "s"

    def test_a_stated_unit_off_the_ladder_is_refused(self) -> None:
        payload: dict[str, Any] = {
            "groups": [{"label": "a", "ci": {"mean": 9, "low": 8, "high": 11, "level": 0.95, "variability": "cases"}}],
            "unit": "ms",
            "stated_unit": "usd",
        }
        with pytest.raises(PayloadError, match="no restatement"):
            chart_intent("distribution", payload)


class TestWordsAndEvidenceReadOnTheRuler:
    def test_a_figure_in_prose_states_the_measures_ruler(self) -> None:
        surface = _surface(threshold=1500.0)
        alias = "c1" if surface.cells[0].variant_key == entry(INCUMBENT).variant_key else "c2"
        assert render_reference(surface, f"{alias}|{LATENCY}|measure|mean") == "0.9 s"
        assert render_reference(_surface(), f"{alias}|{LATENCY}|measure|mean") == "900 ms"

    def test_an_evidence_row_states_its_value_and_spread_on_the_ruler(self) -> None:
        surface = _surface(threshold=1500.0)
        subject = analysis(surface)
        slow = next(c for c in surface.cells if c.variant_key == entry(CANDIDATE).variant_key)
        row = EvidenceRow(
            cell_ref=cell_ref(slow.variant_key, slow.apparatus_class_id),
            measure_id=LATENCY,
            reading="measure",
            value=950.0,
            n=18,
            n_cases=6,
            dispersion="sem 40; 95% CI [800, 1100]",
        )
        (stated,) = evidence_rows(subject, [row], lambda ref: ref)
        assert (stated["value"].value, stated["value"].unit, stated["value"].text) == (
            pytest.approx(0.95),
            "s",
            "0.95 s",
        )
        assert stated["spread"].text == "sem 0.04; 95% CI [0.8, 1.1]"


class TestARunSummaryStatesAMeasureOnOneRuler:
    def test_its_mean_minimum_and_maximum_read_in_one_unit(self) -> None:
        summary = EvalSummary(
            run_id="r",
            scope_id="s",
            template_id=None,
            candidate_model="m",
            status="completed",
            k_runs=1,
            n_cases=3,
            n_results=3,
            n_scored=3,
            n_candidate_failed=0,
            n_excluded=0,
            measures=[RunMeasureSummary(name="total_ms", n=3, mean=950.0, minimum=700.0, maximum=1200.0, unit="ms")],
            confusion=[],
            labels=[],
            errors=[],
        )
        assert summary.measure_lines[0].text == "total_ms: mean 0.95 s (n=3, min 0.7 s, max 1.2 s)"
