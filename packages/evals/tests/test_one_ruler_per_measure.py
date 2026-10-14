"""A measure reads in one unit on every surface of a report: its tables, its words and its summary lines.

The report chooses each measure's ruler once (:func:`~threetears.evals.analysis.report.figures.measure_rulers`, over
every cell's mean and every threshold and verdict of a bar on it), and every table, chart and sentence restates onto
it. A chart that chose its unit over its own values once drew a latency in s beside a table stating it in ms; a
1500 ms bar threshold put the table in s while the chart, whose values never reached a second, stayed in ms. Each
surface below is held to the ruler instead.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from threetears.evals.analysis.cells import cell_ref
from threetears.evals.analysis.prose_refs import render_reference
from threetears.evals.analysis.report.figures import measure_rulers
from threetears.evals.analysis.report.words import evidence_rows
from threetears.evals.analysis.summary import EvalSummary, RunMeasureSummary
from threetears.evals.kernel.analysis_measures import (
    BarAdjudication,
    BarVerdict,
    MeasureCollection,
    MeasureSummary,
)
from threetears.evals.kernel.campaign import EvidenceRow, VariantIndexEntry
from threetears.evals.kernel.identity import compute_variant_key
from threetears.evals.kernel.surface import CellFacts, DecisionSurface, MeasureFacts
from threetears.evals.schema.values import SweepableValue

AXIS = "candidate_model"
CANDIDATE = "model-b-rev2"
INCUMBENT = "model-b"
RIG = "a" * 64
LATENCY = "total_ms"


def _level(model: str) -> SweepableValue:
    return SweepableValue.of(model, display=model)


def entry(model: str) -> VariantIndexEntry:
    """The index entry generation freezes for the arm that ran ``model``."""
    levers = {AXIS: _level(model)}
    return VariantIndexEntry(variant_key=compute_variant_key(levers), levers=levers)


def cell(model: str, *, n_cases: int, measures: MeasureCollection) -> CellFacts:
    """One cell of the arm that ran ``model``, under one rig."""
    return CellFacts(
        variant_key=entry(model).variant_key,
        apparatus_class_id=RIG,
        run_ids=[f"run-{model[:6]}"],
        n_observations=18,
        n_cases=n_cases,
        repeats_per_case_min=3,
        repeats_per_case_max=3,
        measures=measures,
    )


def verdict(of: CellFacts, value: float, sem: float, cleared: bool) -> BarVerdict:
    """A bar verdict on the cell ``of``, decided on an interval around the value."""
    return BarVerdict(
        variant_key=of.variant_key,
        apparatus_class_id=of.apparatus_class_id,
        run_ids=of.run_ids,
        value=value,
        sem=sem,
        n=6,
        n_independent=2,
        ci_low=value - 1.0,
        ci_high=value + 1.0,
        cleared=cleared,
    )


def measures() -> dict[str, MeasureFacts]:
    """The frozen catalogue: a latency in ms."""
    return {LATENCY: MeasureFacts(unit="ms", merit_axis="latency", higher_is_better=False)}


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


class TestTheRulerIsChosenOverEveryFigureTheReportStates:
    def test_an_interval_end_past_a_second_does_not_move_the_ruler_off_ms(self) -> None:
        assert measure_rulers(_surface())[LATENCY].unit == "ms"

    def test_a_threshold_past_a_second_moves_the_ruler_to_s(self) -> None:
        assert measure_rulers(_surface(threshold=1500.0))[LATENCY].unit == "s"


class TestWordsAndEvidenceReadOnTheRuler:
    def test_a_figure_in_prose_states_the_measures_ruler(self) -> None:
        surface = _surface(threshold=1500.0)
        alias = "c1" if surface.cells[0].variant_key == entry(INCUMBENT).variant_key else "c2"
        assert render_reference(surface, f"{alias}|{LATENCY}|measure|mean") == "0.9 s"
        assert render_reference(_surface(), f"{alias}|{LATENCY}|measure|mean") == "900 ms"

    def test_an_evidence_row_states_its_value_and_spread_on_the_ruler(self) -> None:
        surface = _surface(threshold=1500.0)
        # What evidence_rows reads of an analysis: the decision surface it was resolved under.
        subject = SimpleNamespace(decision_surface=surface)
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
        (stated,) = evidence_rows(subject, [row], lambda ref: ref)  # type: ignore[arg-type]
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
