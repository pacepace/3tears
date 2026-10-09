"""Compile a chart from the model's authored chart — the model picks the chart and the cells, code draws the numbers.

The model authors one closed chart shape for every type (:class:`~threetears.evals.contracts.authored.Chart`:
``cells``, ``measures``, ``axis``, ``note``, ``caption``), already checked strictly by
:func:`~threetears.evals.contracts.authored.validate_authored`. :func:`reference_from_chart` reads its
lists by position into the chart type's own typed reference, refusing — in the words of the chart
the model wrote (:data:`CHART_READINGS`) — any list that is not what the type reads; and
:func:`build_viz_payload` compiles that reference into the payload the renderers read, from the
analysis's frozen :class:`~threetears.evals.contracts.surface.DecisionSurface`. Every label and axis
title is computed here from the cells and measures, because the authored chart carries none.

**Every number goes through** :func:`~threetears.evals.analysis.references.resolve_reading` — the one
lookup every code-filled number on an analysis uses — except a categorical breakdown's counts,
which have no point estimate to resolve and are read off the cell :func:`require_cell` returns.

**What refuses here and what refuses downstream.** A reference that names something a chart cannot
be drawn FROM — an unknown cell, a missing measure, a reading with no interval, an ambiguous
default — raises :class:`UnresolvableReference`, which the repair round feeds back so the model can
name something else. A payload built from valid readings that the payload contract then refuses (a
delta table whose every baseline is zero, a breakdown whose parts miss the total it names) is left
to the generator's own payload pass, which decides between dropping the chart and refusing the
analysis; deciding that here as well would be a second policy for one question.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any, Literal, NamedTuple, NoReturn, overload

from pydantic import Field

from threetears.evals.analysis import stats
from threetears.evals.analysis.arms import (
    applicable_levers,
    arm_label,
    arm_names,
    elide_level,
    multi_rig_variants,
    short_digest,
)
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.references import (
    ReadingRef,
    ResolvedReading,
    cell_index,
    require_cell,
    resolve_reading,
)
from threetears.evals.analysis.viz.payloads import ABSENT_LEVEL
from threetears.evals.contracts.authored import Chart
from threetears.evals.contracts.base import EvalBaseModel
from threetears.evals.contracts.campaign import ReadingKind, VariantIndexEntry
from threetears.evals.contracts.host.measures import MeasureRegistry
from threetears.evals.contracts.metrics import describe_reported_measure, materiality, remainder_withheld_reason
from threetears.evals.contracts.surface import CellFacts, DecisionSurface, TimePosition

#: The dimension a sweep row gains when one arm was measured under more than one rig. Without it
#: the two cells carry identical levels and draw as one configuration holding two ranks.
_RIG_DIMENSION = "rig"

#: The only unit a frontier's latency field is stated in — the payload names it ``latency_ms``.
_LATENCY_UNIT = "ms"


# --- Reference shapes ------------------------------------------------------------------------------


class _Ref(EvalBaseModel):
    """Base for every chart reference — what a chart draws, as code read it off the authored chart.

    Built only by :func:`reference_from_chart`, which has already refused every list the type does
    not read, so a reference carries no validation of its own: it names cells and readings, and
    the builder that compiles it resolves each against the surface.
    """

    caption: str | None = Field(default=None, description="The model's editorial line, copied to the payload verbatim.")


class DeltaTableRef(_Ref):
    """Two cells compared on one or more readings."""

    a_cell: str = Field(description="The baseline cell.")
    b_cell: str = Field(description="The cell compared against it.")
    measures: list[ReadingRef] = Field(description="One row per reading.")


class DistributionRef(_Ref):
    """One reading's interval at each of several cells, on one shared axis."""

    cells: list[str] = Field(description="The cells, one group each, in reading order.")
    measure_id: str = Field(description="The measure (or judged dimension) drawn.")
    reading: ReadingKind = Field(description="Its namespace.")


class NullResultRef(_Ref):
    """Two cells' intervals on one reading, and why the lever cannot act."""

    a_cell: str = Field(description="The first arm's cell.")
    b_cell: str = Field(description="The second arm's cell.")
    measure_id: str = Field(description="The measure (or judged dimension) compared.")
    reading: ReadingKind = Field(description="Its namespace.")
    # Never blank (reference_from_chart refuses an empty `note`), because a published null retires a
    # lever — it tells the reader to stop investigating — and a null the model cannot explain is one
    # it has not established. A campaign once published "synthesis token budget does not move
    # latency" over three means falling 50s → 47s → 34s, with a mechanism written to justify the
    # absence. Interval overlap is deliberately not checked: it neither establishes a null nor refutes one.
    mechanism: str = Field(description="Why the lever cannot act — what makes this an established null.")


class BreakdownRef(_Ref):
    """One cell's quantity divided into parts — by a categorical measure's counts, or by numeric part measures."""

    cell: str = Field(description="The cell whose quantity is broken down.")
    measure_id: str | None = Field(default=None, description="A CATEGORICAL measure — its categories are the parts.")
    part_measure_ids: list[str] | None = Field(
        default=None, description="The numeric measures, two or more — each is a part."
    )


class AttributionRef(_Ref):
    """How an end-to-end measure and one subsystem measure moved between two cells."""

    a_cell: str = Field(description="The baseline cell — level A.")
    b_cell: str = Field(description="The cell compared against it — level B.")
    end_to_end_measure: str = Field(description="The whole-run measure.")
    subsystem_measure: str = Field(description="The isolating measure — the part under test.")
    lever: str | None = Field(default=None, description="The lever whose two levels the cells are.")


class FrontierRef(_Ref):
    """Cells placed on cost against quality."""

    quality: ReadingRef = Field(description="The quality reading — higher must be better.")
    cells: list[str] | None = Field(
        default=None, description="The contestants; every cell on the surface when omitted."
    )
    cost_measure_id: str | None = Field(
        default=None, description="The cost measure; defaults to the one surface measure on the cost axis."
    )
    latency_measure_id: str | None = Field(
        default=None,
        description="The latency measure; defaults to the one surface measure on the latency axis, if any.",
    )


class SweepRankingRef(_Ref):
    """Cells ranked on one reading, with a second shown alongside."""

    ranked: ReadingRef = Field(description="The reading the cells are ranked on — higher must be better.")
    secondary: ReadingRef = Field(description="The companion reading.")
    cells: list[str] | None = Field(
        default=None, description="The configurations; every cell on the surface when omitted."
    )


class TimeseriesRef(_Ref):
    """One reading at each position of the campaign's time axis, one line per cell."""

    cells: list[str] | None = Field(
        default=None, description="The cells, one line each; every cell on the surface when omitted."
    )
    measure_id: str = Field(description="The measure (or judged dimension) drawn.")
    reading: ReadingKind = Field(description="Its namespace.")


def _repeated(values: Sequence[str]) -> str:
    """The values a list names more than once, comma-joined; empty when every value is distinct."""
    return ", ".join(sorted({value for value in values if values.count(value) > 1}))


# --- Labels ----------------------------------------------------------------------------------------


def cell_arm_labels(surface: DecisionSurface, variant_index: list[VariantIndexEntry]) -> dict[str, str]:
    """Name every cell on the surface the way the arm table and decision surface name its arm.

    The words are :func:`~threetears.evals.analysis.arms.arm_label`'s over the analysis's
    :func:`~threetears.evals.analysis.arms.arm_names` — the labeller the MCP render's arm and surface
    tables call — and the facts behind them (the levels, the cut, the multi-rig test, the digest width)
    are the arm module's, so one arm reads alike in a chart, in either table, and in the memo the
    reporter eval's judge reads (:func:`~threetears.evals.analysis.reporter_kind.render_memo_as_written`).

    Args:
        surface: The decision surface whose cells to name.
        variant_index: The analysis's variant index, which says what each cell's arm carried.

    Returns:
        ``cell_ref -> label`` for every cell.
    """
    # Over EVERY cell rather than the ones a chart draws, as the surface table asks it: a label is
    # the arm's name on this analysis, and a name must not change with the chart it appears in.
    multi_rig = multi_rig_variants(surface.cells)
    names = arm_names(variant_index)
    return {
        ref: arm_label(
            cell.variant_key,
            names,
            rig=short_digest(cell.apparatus_class_id) if cell.variant_key in multi_rig else None,
        )
        for ref, cell in cell_index(surface).items()
    }


# --- Shared readers --------------------------------------------------------------------------------


#: What every code-filled interval varies over — worded without the count, so equal sources compare equal.
#:
#: The compiler states one sentence when every interval names the same source and a "not comparable"
#: warning when they differ, so a per-cell count in this text would make every chart read as
#: incomparable. Every reading's interval is computed over its cell's cases, a case's repeats clustered
#: (:func:`threetears.evals.analysis.stats.clustered_standard_error`), so a cell run once per case and
#: one run three times per case draw widths that mean the same thing, and one wording is the truth.
_VARIABILITY = "the cell's cases (repeats of one case clustered)"


def _interval(reading: ResolvedReading) -> dict[str, Any]:
    """A reading's interval in the payload's ``ci`` shape, or a refusal when it has none.

    Raises:
        UnresolvableReference: The reading has no interval (fewer than two observations, or one case).
    """
    if reading.ci_low is None or reading.ci_high is None:
        raise UnresolvableReference(
            f"reference names {reading.measure_id!r} at cell {reading.cell_ref!r}, which has no interval "
            f"({reading.dispersion}) — this chart draws intervals, so name a cell with at least 2 observations of it, "
            "over at least 2 cases"
        )
    return {
        "low": reading.ci_low,
        "high": reading.ci_high,
        "mean": reading.mean,
        "level": stats.INTERVAL_LEVEL,
        "variability": _VARIABILITY,
    }


def _read(surface: DecisionSurface, ref: str, reading: ReadingRef) -> ResolvedReading:
    return resolve_reading(surface, ref, reading.measure_id, reading.reading)


def _shared_unit(readings: Sequence[ResolvedReading], chart: str) -> str:
    """The one unit a chart states every reading in, or a refusal naming the units it was handed.

    Raises:
        UnresolvableReference: The readings are in more than one unit, or in none.
    """
    units = {reading.unit for reading in readings}
    named = sorted({reading.measure_id for reading in readings})
    if len(units) > 1:
        spread = ", ".join(f"{r.measure_id}={r.unit or 'unitless'}" for r in readings)
        raise UnresolvableReference(
            f"a {chart} states every value in one unit, but the referenced readings are in several ({spread}) — "
            "name readings that share a unit"
        )
    unit = next(iter(units))
    if not unit:
        raise UnresolvableReference(
            f"a {chart} needs a unit and {', '.join(named)} carries none — name measures with a declared unit"
        )
    return unit


def _has_measure(cell: CellFacts, name: str) -> bool:
    return any(m.name == name for m in cell.measures.measures)


def _cells_or_all(surface: DecisionSurface, cells: list[str] | None, chart: str) -> list[str]:
    """The cells a reference names, or every cell on the surface; either way at least two.

    Raises:
        UnresolvableReference: Fewer than two cells.
    """
    chosen = cells if cells is not None else list(cell_index(surface))
    if len(chosen) < 2:
        raise UnresolvableReference(f"a {chart} compares at least 2 cells and the decision surface holds {len(chosen)}")
    return chosen


def _require_higher_is_better(reading: ResolvedReading, role: str, chart: str) -> None:
    """Refuse a reading drawn as higher-is-better that is not.

    Raises:
        UnresolvableReference: Its polarity is lower-is-better, or undeclared.
    """
    if reading.higher_is_better is not True:
        polarity = "lower-is-better" if reading.higher_is_better is False else "of undeclared direction"
        raise UnresolvableReference(
            f"a {chart}'s {role} reading must be higher-is-better — the chart draws best at the top — but "
            f"{reading.measure_id!r} is {polarity}; name a higher-is-better reading there"
        )


# --- Builders --------------------------------------------------------------------------------------


def _delta_table(ref: DeltaTableRef, surface: DecisionSurface, labels: dict[str, str]) -> dict[str, Any]:
    rows = []
    for reading_ref in ref.measures:
        a = _read(surface, ref.a_cell, reading_ref)
        b = _read(surface, ref.b_cell, reading_ref)
        # A measure's threshold is frozen on the surface; a judged dimension declares none, so its
        # every difference is material.
        facts = surface.measures.get(reading_ref.measure_id) if reading_ref.reading == "measure" else None
        rows.append(
            {
                "metric": reading_ref.measure_id,
                "data_type": "numeric",
                "a": a.mean,
                "b": b.mean,
                "unit": a.unit,
                "delta": b.mean - a.mean,
                "materiality": materiality(facts.materiality_threshold if facts else None, b.mean - a.mean),
                # The smaller arm bounds any test the pair could support; the bundle carries no
                # paired statistic, so no test was run and the row says so through `significant`.
                "n": min(a.n, b.n),
                "paired": False,
                "d_z": None,
                "p": None,
                "significant": None,
            }
        )
    return {
        "caption": ref.caption,
        "rows": rows,
        "a_label": labels[ref.a_cell],
        "b_label": labels[ref.b_cell],
    }


def _distribution(ref: DistributionRef, surface: DecisionSurface, labels: dict[str, str]) -> dict[str, Any]:
    readings = [resolve_reading(surface, cell, ref.measure_id, ref.reading) for cell in ref.cells]
    return {
        "caption": ref.caption,
        "groups": [
            {"label": labels[reading.cell_ref], "ci": _interval(reading), "n": reading.n} for reading in readings
        ],
        "unit": readings[0].unit,
        # The value axis is the reading drawn, named — without it every distribution titles itself
        # "Distribution", and two on one page cannot be told apart.
        "x_label": f"{ref.measure_id} (judged)" if ref.reading == "judged" else ref.measure_id,
    }


def _null_result(ref: NullResultRef, surface: DecisionSurface, labels: dict[str, str]) -> dict[str, Any]:
    readings = [resolve_reading(surface, cell, ref.measure_id, ref.reading) for cell in (ref.a_cell, ref.b_cell)]
    return {
        "caption": ref.caption,
        "groups": [
            {"label": labels[reading.cell_ref], "ci": _interval(reading), "n": reading.n} for reading in readings
        ],
        "metric": ref.measure_id,
        "unit": readings[0].unit,
        "mechanism": ref.mechanism,
    }


def _breakdown(ref: BreakdownRef, surface: DecisionSurface, labels: dict[str, str]) -> dict[str, Any]:
    if ref.measure_id is not None:
        return _categorical_breakdown(ref, ref.measure_id, surface)
    parts = [resolve_reading(surface, ref.cell, name) for name in ref.part_measure_ids or []]
    return {
        "caption": ref.caption,
        "parts": [{"label": part.measure_id, "value": part.mean, "n": part.n} for part in parts],
        "unit": _shared_unit(parts, "breakdown"),
    }


def _categorical_breakdown(ref: BreakdownRef, measure_id: str, surface: DecisionSurface) -> dict[str, Any]:
    """A categorical measure's counts at one cell — the one number here with no point estimate to resolve.

    Raises:
        UnresolvableReference: The measure is absent or numeric.
    """
    cell = require_cell(surface, ref.cell)
    summary = next((m for m in cell.measures.measures if m.name == measure_id), None)
    if summary is None:
        categorical = sorted(m.name for m in cell.measures.measures if m.categories)
        raise UnresolvableReference(
            f"breakdown names measure {measure_id!r} at cell {ref.cell!r}, which measured no such measure; its "
            f"categorical measures are: {', '.join(categorical) or '(none)'}"
        )
    if not summary.categories:
        raise UnresolvableReference(
            f"breakdown names {measure_id!r} as its one measure, which reads as a categorical measure, but at cell "
            f"{ref.cell!r} it is numeric — break numeric measures down by naming two or more of them as the parts"
        )
    # One category compiles to a one-part chart and is left for the generator's undrawable-chart
    # drop, which keeps the finding and says why the chart went. Refusing here would spend the paid
    # repair round on a chart the model cannot see is undrawable until it is built: which categories
    # a cell carries is the data's answer, not a malformed reference.
    return {
        "caption": ref.caption,
        "parts": [{"label": category, "value": float(count)} for category, count in summary.categories.items()],
        "unit": "observations",
        "measure": measure_id,
        "total": float(sum(summary.categories.values())),
        "total_n": summary.n,
    }


def _attribution(
    ref: AttributionRef, surface: DecisionSurface, labels: dict[str, str], measures: MeasureRegistry
) -> dict[str, Any]:
    movements: dict[str, dict[str, Any]] = {}
    readings = []
    for role, name in (("end_to_end", ref.end_to_end_measure), ("subsystem", ref.subsystem_measure)):
        a = resolve_reading(surface, ref.a_cell, name)
        b = resolve_reading(surface, ref.b_cell, name)
        readings += [a, b]
        movements[role] = {"measure": name, "delta": b.mean - a.mean, "a": a.mean, "b": b.mean, "n": min(a.n, b.n)}
    unit = _shared_unit(readings, "attribution")
    # Whether the remainder may be stated is the one rule the bundle's divergence lens also asks —
    # declared containment AND exhaustion — so a chart can never state a remainder the lens withholds.
    described = describe_reported_measure(ref.subsystem_measure, measures)
    contained_by = described.contained_by
    withheld = remainder_withheld_reason(ref.subsystem_measure, ref.end_to_end_measure, described, measures=measures)
    payload: dict[str, Any] = {
        "caption": ref.caption,
        "end_to_end": movements["end_to_end"],
        "subsystem": movements["subsystem"],
        "unit": unit,
        "contained_by": contained_by,
        "lever": ref.lever,
        "a_label": labels[ref.a_cell],
        "b_label": labels[ref.b_cell],
    }
    if withheld is None:
        payload["unattributed_delta"] = movements["end_to_end"]["delta"] - movements["subsystem"]["delta"]
    else:
        payload["unattributed_withheld"] = withheld
    return payload


@overload
def _axis_default(surface: DecisionSurface, axis: Literal["cost", "latency"], *, required: Literal[True]) -> str: ...


@overload
def _axis_default(surface: DecisionSurface, axis: Literal["cost", "latency"], *, required: bool) -> str | None: ...


def _axis_default(surface: DecisionSurface, axis: Literal["cost", "latency"], *, required: bool) -> str | None:
    """The one surface measure on a merit axis, read off its frozen ``merit_axis``.

    Raises:
        UnresolvableReference: More than one measure is on the axis, or none is and one is required.
    """
    names = sorted(name for name, facts in surface.measures.items() if facts.merit_axis == axis)
    # The chart names the cost measure second and the latency measure third (CHART_READINGS).
    position = "second" if axis == "cost" else "third"
    if len(names) > 1:
        raise UnresolvableReference(
            f"frontier names no {axis} measure and the surface holds several measures on the {axis} axis "
            f"({', '.join(names)}) — name one as the chart's {position} measure"
        )
    if not names and required:
        raise UnresolvableReference(
            f"frontier names no cost measure and the surface holds no measure on the cost axis — name the measure to "
            f"place on x as the chart's second measure; the surface's measures are: "
            f"{', '.join(sorted(surface.measures)) or '(none)'}"
        )
    return names[0] if names else None


def _optional_reading(surface: DecisionSurface, ref: str, measure_id: str | None) -> ResolvedReading | None:
    """A measure a contestant MAY carry — absent at a cell reads as unpriced (or untimed), never as zero."""
    if measure_id is None or not _has_measure(require_cell(surface, ref), measure_id):
        return None
    return resolve_reading(surface, ref, measure_id)


def dominated_flags(points: Sequence[tuple[float, float | None]]) -> list[bool]:
    """Which contestants another beats on both axes — quality higher, cost lower.

    A point is dominated when some other point is at least as good on quality AND on cost and
    strictly better on one of them. A point with no cost cannot be placed on the trade-off, so it
    neither dominates nor is dominated — "never priced" is not "priced high".

    Args:
        points: ``(quality, cost)`` per contestant, higher quality and lower cost better.

    Returns:
        One flag per point, in order.
    """
    flags = []
    for i, (quality, cost) in enumerate(points):
        flags.append(
            cost is not None
            and any(
                j != i
                and other_cost is not None
                and other_quality >= quality
                and other_cost <= cost
                and (other_quality > quality or other_cost < cost)
                for j, (other_quality, other_cost) in enumerate(points)
            )
        )
    return flags


def _frontier(ref: FrontierRef, surface: DecisionSurface, labels: dict[str, str]) -> dict[str, Any]:
    cells = _cells_or_all(surface, ref.cells, "frontier")
    cost_id = ref.cost_measure_id or _axis_default(surface, "cost", required=True)
    latency_id = ref.latency_measure_id or _axis_default(surface, "latency", required=False)

    qualities = [_read(surface, cell, ref.quality) for cell in cells]
    _require_higher_is_better(qualities[0], "quality", "frontier")
    costs = [_optional_reading(surface, cell, cost_id) for cell in cells]
    latencies = [_optional_reading(surface, cell, latency_id) for cell in cells]
    for reading in (r for r in costs if r is not None):
        if reading.higher_is_better is True:
            raise UnresolvableReference(
                f"frontier places {cost_id!r} on the cost axis, where lower is better, but it is higher-is-better — "
                "name a cost measure"
            )
    for reading in (r for r in latencies if r is not None):
        if reading.unit != _LATENCY_UNIT:
            raise UnresolvableReference(
                f"a frontier states latency in {_LATENCY_UNIT}, and {latency_id!r} is in {reading.unit or 'no unit'} — "
                "name a latency measure in ms"
            )

    flags = dominated_flags([(q.mean, c.mean if c else None) for q, c in zip(qualities, costs, strict=True)])
    points = [
        {
            "label": labels[cell],
            "quality": quality.mean,
            "cost": cost.mean if cost else None,
            "latency_ms": latency.mean if latency else None,
            "dominated": dominated,
            "disqualified": False,
        }
        for cell, quality, cost, latency, dominated in zip(cells, qualities, costs, latencies, flags, strict=True)
    ]
    cost_unit = surface.measures[cost_id].unit if cost_id in surface.measures else None
    return {
        "caption": ref.caption,
        "points": points,
        "bar": _quality_bar(surface, ref.quality),
        "cost_label": _titled(cost_id, cost_unit),
        "quality_label": _titled(ref.quality.measure_id, qualities[0].unit),
    }


def _timeseries(ref: TimeseriesRef, surface: DecisionSurface, labels: dict[str, str]) -> dict[str, Any]:
    """One line per cell across the time axis, each point the cell's reading over that position's runs.

    The whole surface resolves the reading first, so a reading no position could draw — a categorical or
    text measure, a name no cell measured — is refused in the resolver's own words. Then each position is
    read as a surface of its own (its cells, the analysis's facts), through the same resolver, and a point
    that position cannot give — the cell was not measured there, nothing was scored, or one observation left
    it no interval — is a GAP named in the payload, never a guessed point.

    Raises:
        UnresolvableReference: The analysis has no time axis; a cell is unknown; the reading cannot be drawn
            as a point; or no cell has an interval at two positions, so there is no line to draw.
    """
    axis = surface.time_axis
    if axis is None:
        raise UnresolvableReference(
            "a timeseries draws the campaign's time axis and this analysis has none — its runs share one build and "
            "one day; draw the comparison with another type"
        )
    cells = ref.cells if ref.cells is not None else list(cell_index(surface))
    whole = [resolve_reading(surface, cell, ref.measure_id, ref.reading) for cell in cells]
    series: list[dict[str, Any]] = []
    gaps: list[dict[str, str]] = []
    for cell in cells:
        points = []
        for position in axis.positions:
            point, why = _position_point(position, surface, cell, ref)
            if point is None:
                gaps.append({"series": labels[cell], "position": position.key, "reason": why})
            else:
                points.append(point)
        if points:
            series.append({"label": labels[cell], "points": points})
    if not any(len(line["points"]) >= 2 for line in series):
        raise UnresolvableReference(
            f"a timeseries draws {ref.measure_id!r} with an interval at two or more positions for at least one cell, "
            f"and none of {', '.join(cells)} has one at two — name cells measured at two or more of "
            f"{', '.join(position.key for position in axis.positions)}"
        )
    # Positions are ordered by their earliest run, so a build interleaves with the next exactly when its latest
    # run was made after the next one's earliest.
    interleaved = [
        position.key
        for position, following in zip(axis.positions, axis.positions[1:], strict=False)
        if datetime.fromisoformat(position.last_run_at) > datetime.fromisoformat(following.first_run_at)
    ]
    return {
        "caption": ref.caption,
        "metric": ref.measure_id,
        "unit": whole[0].unit,
        "basis": axis.basis,
        "release_label": axis.release_label,
        "positions": [position.key for position in axis.positions],
        "series": series,
        "gaps": gaps,
        "interleaved": interleaved if axis.basis == "release" else [],
    }


def _position_point(
    position: TimePosition, surface: DecisionSurface, cell: str, ref: TimeseriesRef
) -> tuple[dict[str, Any] | None, str]:
    """One cell's point at one time position, or why it has none — the reason taken from the branch that found it."""
    at = DecisionSurface(cells=position.cells, measures=surface.measures, dimensions=surface.dimensions)
    facts = cell_index(at).get(cell)
    if facts is None:
        return None, "the cell was not measured there"
    if ref.reading == "judged":
        judged = next((j for j in facts.judged if j.dimension == ref.measure_id), None)
        if judged is None or judged.mean is None:
            return None, "nothing was scored on it there"
    else:
        summary = next((m for m in facts.measures.measures if m.name == ref.measure_id), None)
        if summary is None or (summary.rate if summary.rate is not None else summary.mean) is None:
            return None, "it was not observed there"
    reading = resolve_reading(at, cell, ref.measure_id, ref.reading)
    if reading.ci_low is None or reading.ci_high is None:
        return None, f"{reading.n} observation{'' if reading.n == 1 else 's'} there, too few for an interval"
    return {"position": position.key, "ci": _interval(reading), "n": reading.n}, ""


def _quality_bar(surface: DecisionSurface, quality: ReadingRef) -> float | None:
    """The adjudicated bar on the quality measure, when the surface holds exactly one threshold for it.

    A judged dimension has no bar (bars are declared on measures), and two bars at different
    thresholds on one measure have no single rule to draw — the chart then draws none rather than
    picking one a reader would take as THE bar.
    """
    if quality.reading != "measure":
        return None
    thresholds = {
        bar.threshold for bar in surface.bars if bar.state == "adjudicated" and bar.measure_id == quality.measure_id
    }
    return next(iter(thresholds)) if len(thresholds) == 1 else None


def _titled(name: str, unit: str | None) -> str:
    return f"{name} ({unit})" if unit else name


def _sweep_ranking(
    ref: SweepRankingRef, surface: DecisionSurface, variant_index: list[VariantIndexEntry]
) -> dict[str, Any]:
    cells = _cells_or_all(surface, ref.cells, "sweep_ranking")
    ranked = [_read(surface, cell, ref.ranked) for cell in cells]
    secondary = [_read(surface, cell, ref.secondary) for cell in cells]
    _require_higher_is_better(ranked[0], "ranked", "sweep_ranking")
    configs, dimensions = _sweep_configs(surface, cells, variant_index)

    unsorted: list[dict[str, Any]] = [
        {"config": config, "ranked_value": r.mean, "secondary_value": s.mean, "n": r.n}
        for config, r, s in zip(configs, ranked, secondary, strict=True)
    ]
    rows = sorted(unsorted, key=lambda row: -row["ranked_value"])
    return {
        "caption": ref.caption,
        "ranked": {"measure": ref.ranked.measure_id, "unit": ranked[0].unit},
        "secondary": {"measure": ref.secondary.measure_id, "unit": secondary[0].unit},
        "rows": rows,
        "dimensions": dimensions,
    }


def _sweep_configs(
    surface: DecisionSurface, cells: list[str], variant_index: list[VariantIndexEntry]
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Each cell's configuration — its arm's levels — over the levers that actually vary among them.

    A lever one arm carries and another does not reads as the absence sentinel the payload already
    knows — and so does a lever that does not apply to an arm's kind, which that arm did not run. A
    lever every row holds at the same level is dropped: a barcode column that never changes says
    nothing and still spends a hue. Orderedness is DECLARED from each level's scale rather than left
    to the compiler's text inference, because the scale is what the host said. Each level is cut as
    an arm's name cuts it (:func:`~threetears.evals.analysis.arms.elide_level`) wherever the cut keeps
    that column's levels apart, so no two configurations come to read alike.

    Raises:
        UnresolvableReference: A cell's levels cannot be described, or the cells differ on nothing.
    """
    index = {entry.variant_key: entry for entry in variant_index}
    chosen = [require_cell(surface, cell) for cell in cells]
    levers: list[dict[str, Any]] = []
    for ref, cell in zip(cells, chosen, strict=True):
        entry = index.get(cell.variant_key)
        named = applicable_levers(entry)
        if entry is None or entry.levels_unavailable or not named:
            raise UnresolvableReference(
                f"sweep_ranking places cell {ref!r} by its levels, and this analysis cannot say what its arm ran — "
                "leave it out of `cells`"
            )
        levers.append(dict(named))

    names = sorted({name for lever_map in levers for name in lever_map})
    configs: list[dict[str, str]] = [
        {name: lever_map[name].display if name in lever_map else ABSENT_LEVEL for name in names} for lever_map in levers
    ]
    ordered: dict[str, bool] = {
        name: all(lever_map[name].scale.kind != "nominal" for lever_map in levers if name in lever_map)
        for name in names
    }
    # Over the cells this chart ranks, not the whole surface: the rig column exists to split two of
    # ITS rows that would otherwise hold identical levels, and a variant with a second rig the chart
    # does not draw has no row to be split from.
    if multi_rig_variants(chosen):
        for config, cell in zip(configs, chosen, strict=True):
            config[_RIG_DIMENSION] = short_digest(cell.apparatus_class_id)
        ordered[_RIG_DIMENSION] = False

    varying = sorted(name for name in ordered if len({config[name] for config in configs}) > 1)
    if not varying:
        raise UnresolvableReference(
            f"sweep_ranking's cells ({', '.join(cells)}) run at identical levels — there is no configuration to rank"
        )
    for name in varying:
        column = {config[name] for config in configs}
        if len({elide_level(level) for level in column}) == len(column):
            for config in configs:
                config[name] = elide_level(config[name])
    return (
        [{name: config[name] for name in varying} for config in configs],
        [{"name": name, "ordered": ordered[name]} for name in varying],
    )


# --- Entry point -----------------------------------------------------------------------------------


class _Chart(NamedTuple):
    """One chart type code can compile: the reference its authored chart is read into, and what builds it."""

    ref: type[_Ref]
    build: Callable[[Any, DecisionSurface, dict[str, str], list[VariantIndexEntry], MeasureRegistry], dict[str, Any]]


#: The one table of chart types a chart can be — the menu, the reference each is read into and the
#: builder each is compiled by, so none of the three can name a type the others do not.
_CHARTS: dict[str, _Chart] = {
    "delta_table": _Chart(
        DeltaTableRef, lambda r, surface, labels, _index, _measures: _delta_table(r, surface, labels)
    ),
    "distribution": _Chart(
        DistributionRef, lambda r, surface, labels, _index, _measures: _distribution(r, surface, labels)
    ),
    "null_result": _Chart(
        NullResultRef, lambda r, surface, labels, _index, _measures: _null_result(r, surface, labels)
    ),
    "breakdown": _Chart(BreakdownRef, lambda r, surface, labels, _index, _measures: _breakdown(r, surface, labels)),
    "attribution": _Chart(
        AttributionRef, lambda r, surface, labels, _index, measures: _attribution(r, surface, labels, measures)
    ),
    "frontier": _Chart(FrontierRef, lambda r, surface, labels, _index, _measures: _frontier(r, surface, labels)),
    "sweep_ranking": _Chart(
        SweepRankingRef, lambda r, surface, _labels, index, _measures: _sweep_ranking(r, surface, index)
    ),
    "timeseries": _Chart(TimeseriesRef, lambda r, surface, labels, _index, _measures: _timeseries(r, surface, labels)),
}

#: Each builder, keyed by the reference type it compiles — derived from :data:`_CHARTS`.
_BUILDERS = {chart.ref: chart.build for chart in _CHARTS.values()}

#: The chart types the model may author.
REFERENCEABLE_VIZ_TYPES: frozenset[str] = frozenset(_CHARTS)

#: The chart types that draw the campaign's time axis — offered only to a bundle that has one
#: (:func:`threetears.evals.analysis.generator.first_request`), and refused by their builder on a surface
#: without one.
TIME_VIZ_TYPES: frozenset[str] = frozenset({"timeseries"})


#: How each chart type reads the unified chart's positional lists — the contract the model is told, and
#: the shape :func:`reference_from_chart` builds. One sentence per type, so a refusal can quote it.
CHART_READINGS: dict[str, str] = {
    "delta_table": "two cells (baseline first) and one or more measures, one row each",
    "distribution": "two or more cells, one group each, and exactly one measure",
    "null_result": "two cells, exactly one measure, and the mechanism in `note`",
    "breakdown": "one cell, and either one categorical measure or two or more numeric measures as the parts",
    "attribution": "two cells (baseline first), the end-to-end measure then the subsystem measure, and the lever in `axis`",
    "frontier": "the quality measure, then optionally the cost and the latency measures; cells, or none for every cell",
    "sweep_ranking": "the ranked measure then the secondary measure; cells, or none for every cell",
    "timeseries": "exactly one measure, drawn at every position of the time axis; cells, one line each, or none for every cell",
}


def reference_from_chart(chart: Chart) -> _Ref:
    """The typed reference an authored chart stands for, read by position — or a refusal about that chart.

    The model authors one chart shape for every type (``cells``, ``measures``, ``axis``, ``note``,
    ``caption``), and each type reads those lists by position (:data:`CHART_READINGS`). Mapping here
    rather than giving each type its own authored shape is what keeps the authored schema small
    enough for every writer. Every refusal quotes the type's reading and names what the chart held,
    in the chart's own terms, because the chart is the only shape the model ever wrote.

    Args:
        chart: An authored chart whose type is on the menu — :func:`validate_authored` has refused
            any other, and ``none`` means there is no chart to read.

    Returns:
        The chart type's reference.

    Raises:
        UnresolvableReference: The lists are not what the type reads — a count, a repeated cell or
            measure, a judged dimension where the type reads a measure, or an empty null-result
            mechanism.
        ValueError: The type is not on the menu — a caller that bypassed :func:`validate_authored`,
            which no repair round can correct.
    """
    kind, cells, measures = chart.type, list(chart.cells), chart.measures
    if kind not in _CHARTS:
        raise ValueError(
            f"chart type {kind!r} is not one of {sorted(_CHARTS)} — validate the chart with validate_authored"
        )

    def refuse(problem: str) -> NoReturn:
        raise UnresolvableReference(f"a {kind} chart reads {CHART_READINGS[kind]}; {problem}")

    def need(ok: bool) -> None:
        if not ok:
            refuse(f"got {len(cells)} cell(s) and {len(measures)} measure(s)")

    def reading(index: int) -> ReadingRef:
        return ReadingRef(measure_id=measures[index].measure_id, reading=measures[index].reading)

    def distinct_measures(*indexes: int) -> None:
        if repeated := _repeated([measures[i].measure_id for i in indexes]):
            refuse(f"`measures` names {repeated} more than once")

    def measure_only(*indexes: int) -> None:
        # These positions name a MEASURE by id and nothing else; a judged dimension there would be
        # read as a measure of the same name, so it is refused rather than quietly re-read.
        if judged := [
            measures[i].measure_id for i in indexes if i < len(measures) and measures[i].reading != "measure"
        ]:
            refuse(f"it reads measures, not judged dimensions, at those positions; got judged {judged}")

    def cells_or_every_cell() -> list[str] | None:
        # An empty list means every cell; one cell compares nothing.
        if len(cells) == 1:
            refuse("got 1 cell — name two or more, or none for every cell")
        return cells or None

    if repeated := _repeated(cells):
        refuse(f"`cells` names {repeated} more than once — a chart compares different cells")
    caption = chart.caption or None

    if kind == "delta_table":
        need(len(cells) == 2 and len(measures) >= 1)
        distinct_measures(*range(len(measures)))
        return DeltaTableRef(
            a_cell=cells[0], b_cell=cells[1], measures=[reading(i) for i in range(len(measures))], caption=caption
        )
    if kind == "distribution":
        need(len(cells) >= 2 and len(measures) == 1)
        return DistributionRef(
            cells=cells, measure_id=measures[0].measure_id, reading=measures[0].reading, caption=caption
        )
    if kind == "null_result":
        need(len(cells) == 2 and len(measures) == 1)
        if not chart.note.strip():
            refuse("`note` is empty — a null result states why the lever cannot act, or it is not an established null")
        return NullResultRef(
            a_cell=cells[0],
            b_cell=cells[1],
            measure_id=measures[0].measure_id,
            reading=measures[0].reading,
            mechanism=chart.note,
            caption=caption,
        )
    if kind == "breakdown":
        need(len(cells) == 1 and len(measures) >= 1)
        measure_only(*range(len(measures)))
        distinct_measures(*range(len(measures)))
        if len(measures) == 1:
            return BreakdownRef(cell=cells[0], measure_id=measures[0].measure_id, caption=caption)
        return BreakdownRef(cell=cells[0], part_measure_ids=[m.measure_id for m in measures], caption=caption)
    if kind == "attribution":
        need(len(cells) == 2 and len(measures) == 2)
        measure_only(0, 1)
        if measures[0].measure_id == measures[1].measure_id:
            refuse(
                f"the end-to-end and subsystem measures are both {measures[0].measure_id!r} — a measure cannot diverge from itself"
            )
        return AttributionRef(
            a_cell=cells[0],
            b_cell=cells[1],
            end_to_end_measure=measures[0].measure_id,
            subsystem_measure=measures[1].measure_id,
            lever=chart.axis or None,
            caption=caption,
        )
    if kind == "frontier":
        need(1 <= len(measures) <= 3)
        measure_only(1, 2)
        return FrontierRef(
            quality=reading(0),
            cells=cells_or_every_cell(),
            cost_measure_id=measures[1].measure_id if len(measures) >= 2 else None,
            latency_measure_id=measures[2].measure_id if len(measures) == 3 else None,
            caption=caption,
        )
    if kind == "timeseries":
        # One cell is a line: a single arm followed across builds is the case this chart exists for.
        need(len(measures) == 1)
        return TimeseriesRef(
            cells=cells or None, measure_id=measures[0].measure_id, reading=measures[0].reading, caption=caption
        )
    need(len(measures) == 2)  # sweep_ranking, the last type on the menu
    return SweepRankingRef(ranked=reading(0), secondary=reading(1), cells=cells_or_every_cell(), caption=caption)


def build_viz_payload(
    reference: _Ref, surface: DecisionSurface, variant_index: list[VariantIndexEntry], *, measures: MeasureRegistry
) -> dict[str, Any]:
    """Compile ``reference`` into the payload its chart type's renderer reads.

    Args:
        reference: The chart's reference, from :func:`reference_from_chart`.
        surface: The decision surface every number is read from.
        variant_index: The analysis's variant index — how a cell is labelled and placed on the axes.
        measures: The host's measure registry: an attribution states a remainder only where the
            whole measure space declares the part inside, and exhausting, the whole.

    Returns:
        The payload, in the shape :data:`threetears.evals.analysis.viz.payloads.PAYLOAD_MODELS` validates.

    Raises:
        UnresolvableReference: The reference names a cell or reading the surface does not hold, or
            one the chart cannot be drawn from.
    """
    # No up-front cell check: every builder reaches a cell through `resolve_reading` or `require_cell`,
    # which refuse an unknown one naming the cells that exist, before any label is read.
    return _BUILDERS[type(reference)](
        reference, surface, cell_arm_labels(surface, variant_index), variant_index, measures
    )


__all__ = [
    "CHART_READINGS",
    "REFERENCEABLE_VIZ_TYPES",
    "TIME_VIZ_TYPES",
    "build_viz_payload",
    "cell_arm_labels",
    "dominated_flags",
    "reference_from_chart",
]
