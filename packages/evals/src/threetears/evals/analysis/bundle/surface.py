"""The bundle's decision surface: its per-cell facts frozen into what an analysis or a code-only report reads.

Copied, never recomputed: :func:`bundle_decision_surface` freezes the facts assembly already computed
(:func:`cell_measure_facts`, :func:`cell_dimension_facts`), and :func:`_measure_catalog` names every measure the
surface carries.
"""

from __future__ import annotations

from threetears.evals.analysis.lenses.frontier import FrontierResult
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.metrics import (
    MetricDescriptor,
    describe_reported_measure,
)
from threetears.evals.kernel.surface import (
    CellFacts,
    DecisionSurface,
    FrontierDominance,
    JudgedDimensionFacts,
    MeasureFacts,
)
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.bundle.cell_reads import _cell_collections


def _measure_catalog(bundle: AnalysisContextBundle, *, profile: HostProfile) -> dict[str, MetricDescriptor]:
    """Collect the descriptor for every measure name appearing anywhere in the bundle.

    Built from the assembled bundle rather than alongside it, so the catalog cannot fall
    out of step with what the summaries actually reference: every name here came from a
    summary, and every summary's name is resolvable here.

    Args:
        bundle: The assembled bundle, before its catalog is attached.
        profile: The host whose vocabulary this reads.

    Returns:
        Descriptors keyed by measure name, in name order for a stable fingerprint.
    """
    collections = [
        bundle.telemetry.measures,
        *(summary.measures for summary in bundle.run_summaries),
        *(collection for cell in bundle.cell_measures for collection in _cell_collections(cell)),
        *(cell.measures for cell in _time_axis_cells(bundle)),
    ]
    names = {measure.name for collection in collections for measure in collection.measures}
    return {name: describe_reported_measure(name, profile.measures) for name in sorted(names)}


def _time_axis_cells(bundle: AnalysisContextBundle) -> list[CellFacts]:
    """Every cell on the bundle's time axis, at every position — empty when there is no axis."""
    return [cell for position in bundle.time_axis.positions for cell in position.cells] if bundle.time_axis else []


def cell_measure_facts(bundle: AnalysisContextBundle) -> dict[str, MeasureFacts]:
    """What each measure named in the bundle's cells IS — the catalogue half of a decision surface.

    Read off ``measure_catalog``, which describes every measure collection in the bundle, the cells'
    included — so the unit, axis and direction frozen beside a cell's value are the ones the generator
    was handed for the same name, never a second lookup that could disagree with it.

    Args:
        bundle: An assembled bundle.

    Returns:
        One entry per measure name appearing in any cell — the time axis's included — in name order.
    """
    cells = [*bundle.cell_measures, *_time_axis_cells(bundle)]
    names = sorted(
        {measure.name for cell in cells for collection in _cell_collections(cell) for measure in collection.measures}
    )
    return {
        name: MeasureFacts(
            reader_name=bundle.measure_catalog[name].reader_name,
            unit=bundle.measure_catalog[name].unit,
            merit_axis=bundle.measure_catalog[name].merit_axis,
            higher_is_better=bundle.measure_catalog[name].higher_is_better,
            materiality_threshold=bundle.measure_catalog[name].materiality_threshold,
            population=bundle.measure_catalog[name].population,
            scale=bundle.measure_catalog[name].scale,
            guardrail=bundle.measure_catalog[name].guardrail,
        )
        for name in names
    }


def cell_dimension_facts(bundle: AnalysisContextBundle) -> dict[str, JudgedDimensionFacts]:
    """What each judged dimension scored in the bundle's cells IS — the judged half of a decision surface.

    Read off ``judged_measures``, the one place the bundle carries a dimension's polarity and scale,
    so the direction a merit claim on a judged score is read in, and the scale a chart draws it on,
    are the ones the generator was handed for the same name. Every dimension a cell carries has an
    entry there by construction: a cell's judged readings are ``judged_measures`` transposed
    (:func:`_cell_measures`).

    Args:
        bundle: An assembled bundle.

    Returns:
        One entry per dimension appearing in any cell, in name order.
    """
    described = {measure.name: measure for measure in bundle.judged_measures}
    names = sorted({reading.dimension for cell in bundle.cell_measures for reading in cell.judged})
    return {
        name: JudgedDimensionFacts(
            higher_is_better=described[name].higher_is_better,
            value_range=described[name].value_range,
            scale=described[name].scale,
            axis=described[name].axis,
        )
        for name in names
    }


def bundle_decision_surface(bundle: AnalysisContextBundle) -> DecisionSurface:
    """Freeze the bundle's per-cell facts into the decision surface an analysis — or a code-only report — reads.

    Copied, never recomputed: each per-cell number is the one assembly computed over the evidence. The
    control is the declaration's own variant key — the same value ``declared_design`` carries and the arm
    table marks as the control — so the two cannot name different arms.

    Args:
        bundle: The evidence set.

    Returns:
        The decision surface.
    """
    return DecisionSurface(
        control_variant_key=bundle.declared_design.control if bundle.declared_design else None,
        cells=bundle.cell_measures,
        bars=bundle.bar_adjudications,
        measures=cell_measure_facts(bundle),
        dimensions=cell_dimension_facts(bundle),
        time_axis=bundle.time_axis,
        frontier_dominance=_frontier_dominance(bundle.frontier),
        frontier_disqualified={
            point.variant_key: list(point.disqualified_by)
            for subject in bundle.frontier.subjects
            for point in subject.points
            if point.disqualified_by
        },
        rubric_threshold=bundle.frontier.rubric_threshold,
        guardrails=bundle.guardrails,
    )


def _frontier_dominance(frontier: FrontierResult) -> dict[str, FrontierDominance]:
    """Each variant's standing on the frontier lens — the verdict a frontier chart draws, never recomputed.

    The lens decides domination by test over per-case values (:func:`~threetears.evals.analysis.reporting.compute_frontier`),
    which a frozen surface does not carry, so a chart deciding it again from the surface's means would be a
    second rule for one question, and on means it called one of two identical arms dominated a third of the
    time. Keyed by variant because a cell is one; a variant the lens placed as more than one point (under two
    identity versions, or two subjects) has no one standing and is left out, so a chart reads it as untested.
    A point stored before domination was tested carries no standing either, and is left out the same way.

    Args:
        frontier: The bundle's frontier lens.

    Returns:
        ``{variant_key: dominance}``.
    """
    placed: dict[str, list[FrontierDominance | None]] = {}
    for subject in frontier.subjects:
        for point in subject.points:
            placed.setdefault(point.variant_key, []).append(point.dominance)
    return {key: standings[0] for key, standings in placed.items() if len(standings) == 1 and standings[0] is not None}


__all__ = [
    "bundle_decision_surface",
]
