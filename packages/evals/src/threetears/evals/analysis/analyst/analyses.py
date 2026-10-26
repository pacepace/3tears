"""The analyst's analyses: what code computes over the frozen report input that no single figure states.

Three analyses, each over the snapshot's report input alone and each returning every number as the figure key that
names it (:mod:`~threetears.evals.analysis.analyst.cited`), so a sentence the analyst writes from one cites keys the
renderer prints the same:

- :func:`dominance`: which options survive the gates and which dominates, on the decision's own axes (every gate,
  cost per 1,000 replies, the asked aspects) as :mod:`~threetears.evals.analysis.report_decision` reads them, with
  the extra axes the analyst names compared beside them on point estimates, descriptively.
- :func:`sensitivity`: how the decision moves as one declared criterion moves (an aspect's margin, a floor's margin,
  a gate's threshold): the value each arm's status flips at, and the call on every piece of the range between them.
- :func:`breakdown`: where a candidate's difference from the control sits, slice by slice (strata, tasks or
  target-matrix cells), and, across strata, which slice carries its loss.

**Code's call is a reading, not a constraint.** The modes and the dominant option these return are what code's rule
reads off the declared criteria: interpretive facts for the analyst and the editor to weigh, never a bound on what a
report may say. Each result says so where it carries one.

**Every number is a key.** Segment ends are the flip figures themselves (``aspects.<aspect>.flip_margin.<arm>``, a
floor's ``held_above`` and ``broken_at``, a criterion's ``value`` at a place); a range end the caller asked for is
never echoed as a number, since no key states it. Counts and levels are words or keys, never bare numbers.

**No branch on the kind of eval.** Every condition is on data: a control exists, an aspect is asked, a slice holds the
measure, a stratum id repeats.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.analysis.analyst.cited import CitedFigure, cite, cite_if
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report_decision import (
    AskedStanding,
    DecisionAt,
    StandingWord,
    asked_standing,
    control_arm,
    decision_at,
    dominant_options,
    separating_contrasts,
    surviving_options,
)
from threetears.evals.analysis.report_gates import GateStatus, gate_check
from threetears.evals.analysis.report_input import Aspect, Contrast, Criterion, ReportInput
from threetears.evals.analysis.report_margins import CriterionFigure, criterion_figures
from threetears.evals.kernel.errors import RefusedInput

__all__ = [
    "BREAKDOWN_PAGE",
    "SENSITIVITY_PAGE",
    "Breakdown",
    "Dominance",
    "Sensitivity",
    "SliceBy",
    "breakdown",
    "dominance",
    "sensitivity",
]

#: What every code call these analyses carry reads as, so neither the analyst nor the editor takes it for a rule.
CODE_CALL_READS_AS = (
    "code's reading of the declared criteria: an interpretive fact for the analyst and the editor to weigh, never a "
    "constraint on what the report says"
)

#: The criterion kinds that gate an option, as the gate check reads them.
_GATING = frozenset({"gate", "guardrail"})

#: The slices one breakdown page holds unless the caller asks for more: a slice reads one line per figure, per arm.
BREAKDOWN_PAGE = 12

#: The flips one sensitivity page lists unless the caller asks for more.
SENSITIVITY_PAGE = 50

#: How many valid choices a refusal lists before it stops.
_SHOWN = 24

#: How a slice is cut: by stratum, by task, or by target-matrix cell.
SliceBy = Literal["stratum", "task", "cell"]

#: Which side of a flip a value exactly at it falls on: the piece below it, or the piece above.
Boundary = Literal["below", "above"]

_MODEL = ConfigDict(frozen=True, extra="forbid")


def _arm(arm_id: str) -> str:
    """An arm as a result names it: ``arm:<id>``, which renders as the arm's name and tag."""
    return f"arm:{arm_id}"


def _listed(names: Sequence[str]) -> str:
    shown = ", ".join(repr(name) for name in names[:_SHOWN])
    return shown + (f", and {len(names) - _SHOWN} more" if len(names) > _SHOWN else "") if names else "none"


def _number(figure: CitedFigure) -> float:
    """A number figure's value."""
    if not isinstance(figure.value, int | float):
        raise TypeError(f"{figure.key} is a {figure.kind}, not a number")
    return float(figure.value)


def _number_or_none(key: str, report_input: ReportInput) -> CitedFigure | None:
    """A figure an arm may lack, by the same rule the decision reads it with: any refusal is no figure."""
    try:
        return cite(key, report_input)
    except UnresolvableReference:
        # NOSILENT: an arm whose cost no source states has no cost, which is how the dominance rule reads it
        # (report_decision's cost rule treats every refusal of this key so); the result shows the figure as absent.
        return None


# =============================================================================
# Dominance
# =============================================================================


class Call(BaseModel):
    """What code's rule decides at one value: the allowed modes, and the one option that dominates."""

    model_config = _MODEL

    modes: list[str] = Field(description="The allowed modes, of decide, tradeoff and no_call.")
    dominant: list[str] = Field(description="The one option that dominates there, as arm:<id>; empty for none.")


class CodeCall(BaseModel):
    """Code's call on the declared criteria, labelled as code's reading."""

    model_config = _MODEL

    reads_as: str = Field(description="What this call is: code's reading, never a constraint.")
    modes: list[str] = Field(description="The modes code's rule allows.")
    dominant: list[str] = Field(description="The one option that dominates alone, as arm:<id>; empty for none.")
    ranking_rule: list[str] = Field(
        description="The merit axes a contrast family declares, which rank survivors none dominates; empty for none."
    )


class SurvivorRow(BaseModel):
    """One option that survives the gates, its standing on the decision's axes, and the figures behind it."""

    model_config = _MODEL

    arm: str = Field(description="The option, as arm:<id>.")
    gate_status: GateStatus = Field(description="pass or borderline: borderline counts as passing.")
    borders: list[str] = Field(description="Why it is borderline, one reason per figure; empty when it passes.")
    standing: StandingWord = Field(description="Its standing on the asked aspects.")
    behind: list[str] = Field(description="In a choice, the arms an asked pair shows ahead of it, as arm:<id>.")
    why: list[str] = Field(description="Why a candidate is not shown no worse on the asked aspects.")
    dominated_by: list[str] = Field(
        description="The survivors at least as good on every axis while it is not as good as them, as arm:<id>."
    )
    figures: list[CitedFigure] = Field(
        description="Its cost per 1,000 replies, its gap from each gate, and its asked readings, pairs and contrasts."
    )


class OutRow(BaseModel):
    """One option the gates or a guardrail removed, and why."""

    model_config = _MODEL

    arm: str = Field(description="The option, as arm:<id>.")
    why: list[str] = Field(description="Each failing gate figure, and each breached guardrail.")


class AxisArm(BaseModel):
    """One survivor on an extra axis."""

    model_config = _MODEL

    arm: str = Field(description="The survivor, as arm:<id>.")
    figure: CitedFigure | None = Field(description="Its mean on the axis; None where the input states none.")


class AxisRow(BaseModel):
    """An extra axis the survivors are compared on: descriptive, on point estimates, no interval and no test."""

    model_config = _MODEL

    axis: str = Field(description="The measure or judged dimension, by key.")
    direction: Literal["higher_is_better", "lower_is_better", "unstated"] = Field(
        description="Which way is better; an axis with no stated direction is listed and not compared."
    )
    descriptive: Literal[True] = Field(default=True, description="Always true: point estimates, no interval or test.")
    best: list[str] = Field(description="The survivors with the best figure on it, as arm:<id>; ties all listed.")
    arms: list[AxisArm] = Field(description="Each survivor's figure, in the input's arm order.")


class Dominance(BaseModel):
    """Which options survive, which dominates, and what extra axes would say beside them."""

    model_config = _MODEL

    survivors: list[SurvivorRow] = Field(description="The options that survive the gates, in the input's arm order.")
    out: list[OutRow] = Field(description="The options the gates or a guardrail removed.")
    dominant: list[str] = Field(
        description="The survivors each at least as good as every other on every axis; two or more tie, none: none "
        "dominates.",
    )
    axes: list[AxisRow] = Field(description="The extra axes asked for, each compared on point estimates.")
    dominant_with_axes: list[str] = Field(
        description="The survivors at least as good as every other on the decision's axes and every extra axis with "
        "a direction; descriptive. Equal to dominant when no axis is asked for.",
    )
    code_call: CodeCall = Field(description="Code's call on the declared criteria: its reading, never a constraint.")


def _contrast_key(contrast: Contrast) -> str:
    """A contrast's key, by its aspect where it states one (the form every aspect's contrast reads by)."""
    if contrast.aspect is not None:
        return f"aspects.{contrast.aspect}.contrast.{contrast.arm}.{contrast.measure}"
    return f"contrast.{contrast.arm}.{contrast.measure}"


def _place(report_input: ReportInput, figure: CriterionFigure) -> str | None:
    """Where a criterion's figure sits, as its key names it: a task's id, a cell's ``<row>.<column>``, or None."""
    if figure.cell is not None:
        cell = report_input.cells[figure.cell]
        return f"{cell.row}.{cell.column}"
    return figure.task


def _criterion_keys(
    report_input: ReportInput, criterion: Criterion, arm: str, field: str
) -> list[tuple[str, str | None]]:
    """The key of each finite figure a criterion reads for an arm, at ``field``, with the place the key names (None
    where the criterion reads one figure for the arm)."""
    held = criterion_figures(report_input, criterion, arm)
    keyed: list[tuple[str, str | None]] = []
    for figure in held:
        if not math.isfinite(figure.value):
            continue
        place = _place(report_input, figure)
        if len(held) == 1 or place is None:
            keyed.append((f"criteria.{criterion.id}.{arm}.{field}", None))
        else:
            keyed.append((f"criteria.{criterion.id}.{arm}.{place}.{field}", place))
    return keyed


def _asked(report_input: ReportInput) -> list[Aspect]:
    return [aspect for aspect in report_input.aspects if aspect.asked]


def _standing_figures(report_input: ReportInput, arm: str) -> list[CitedFigure]:
    """The figures an arm's standing on the asked aspects rests on: its readings and their flips, the pairs it is in
    that show an arm ahead, and the contrasts on an asked aspect that separate it from the control."""
    found: list[CitedFigure] = []
    asked = _asked(report_input)
    for aspect in asked:
        if any(reading.arm == arm for reading in aspect.readings):
            found.append(cite(f"aspects.{aspect.id}.reading.{arm}.change", report_input))
            flip = cite_if(f"aspects.{aspect.id}.flip_margin.{arm}", report_input)
            if flip is not None:
                found.append(flip)
        for pair in aspect.pairs:
            if arm in pair.arms and pair.ahead is not None:
                found.append(cite(f"aspects.{aspect.id}.pair.{pair.arms[0]}.{pair.arms[1]}.change", report_input))
    for contrast in separating_contrasts(report_input, arm):
        if contrast.delta is not None:
            found.append(cite(_contrast_key(contrast) + ".delta", report_input))
    return found


def _gate_figures(report_input: ReportInput, arm: str) -> list[CitedFigure]:
    """An arm's gap from each gate and guardrail, at each place it reads, signed toward passing."""
    return [
        cite(key, report_input)
        for criterion in report_input.criteria
        if criterion.kind in _GATING
        for key, _place_named in _criterion_keys(report_input, criterion, arm, "gap")
    ]


def _cost_at_least(costs: Mapping[str, float], first: str, second: str) -> bool:
    """The dominance rule's cost axis: lower is better; two arms with no cost tie, and one without never beats one
    with."""
    if first not in costs or second not in costs:
        return first not in costs and second not in costs
    return costs[first] <= costs[second]


def _quality_at_least(standing: Mapping[str, AskedStanding], first: str, second: str) -> bool:
    """The dominance rule's asked axis, read off each arm's standing (:class:`AskedStanding`)."""
    mine, theirs = standing[first], standing[second]
    if mine.level is not None and theirs.level is not None:
        return mine.level >= theirs.level
    return second not in mine.behind


def _axis_at_least(values: Mapping[str, float], higher: bool, first: str, second: str) -> bool:
    """An extra axis, on point estimates: two arms with no figure tie, and one without never beats one with."""
    if first not in values or second not in values:
        return first not in values and second not in values
    return values[first] >= values[second] if higher else values[first] <= values[second]


def _axis_figure(report_input: ReportInput, axis: str, arm: str) -> CitedFigure | None:
    """A survivor's mean on an extra axis: a measure's or a judged dimension's, where the input holds the arm."""
    if axis in report_input.judged:
        held = axis in report_input.judged and arm in report_input.judged[axis].per_arm
        return cite_if(f"judged.{axis}.{arm}.mean", report_input) if held else None
    if arm not in report_input.measures[axis].per_arm:
        return None
    return cite_if(f"measure.{axis}.{arm}.mean", report_input)


def _axis(report_input: ReportInput, axis: str, survivors: Sequence[str]) -> tuple[AxisRow, dict[str, float] | None]:
    """One extra axis over the survivors, and the values it compares by (None for an axis with no direction)."""
    figures = {arm: _axis_figure(report_input, axis, arm) for arm in survivors}
    higher: bool | None = True if axis in report_input.judged else report_input.measures[axis].higher_is_better
    values = {
        arm: float(figure.value)
        for arm, figure in figures.items()
        if figure is not None and isinstance(figure.value, int | float) and math.isfinite(figure.value)
    }
    best: list[str] = []
    if higher is not None and values:
        top = max(values.values()) if higher else min(values.values())
        best = [_arm(arm) for arm in survivors if values.get(arm) == top]
    row = AxisRow(
        axis=axis,
        direction="unstated" if higher is None else "higher_is_better" if higher else "lower_is_better",
        best=best,
        arms=[AxisArm(arm=_arm(arm), figure=figures[arm]) for arm in survivors],
    )
    return row, (None if higher is None else values)


def _out_reasons(report_input: ReportInput, arm: str) -> list[str]:
    reasons = list(gate_check(report_input, arm).fails)
    for guardrail in report_input.guardrails:
        if str(guardrail.get("arm")) == arm and guardrail.get("decision") == "breached":
            named = guardrail.get("dimension") or guardrail.get("measure") or guardrail.get("id")
            reasons.append(f"guardrail {named} read breached" if named else "a guardrail read breached")
    return reasons


def dominance(report_input: ReportInput, axes: Sequence[str] = ()) -> Dominance:
    """Which options survive the gates, which dominates on the decision's axes, and what extra axes say beside them.

    The survivors, the dominant options and code's call are :mod:`~threetears.evals.analysis.report_decision`'s own
    (:func:`~threetears.evals.analysis.report_decision.surviving_options`,
    :func:`~threetears.evals.analysis.report_decision.dominant_options`,
    :func:`~threetears.evals.analysis.report_decision.decision_at`). ``dominated_by`` applies the same pairwise rule,
    cost per 1,000 replies and the asked standing, to each pair of survivors.

    :param report_input: The snapshot's report input.
    :ptype report_input: ReportInput
    :param axes: Extra measures or judged dimensions, by key, to compare the survivors on, descriptively.
    :ptype axes: Sequence[str]
    :return: The survivors, the options removed, the dominant ones, the extra axes and code's call.
    :rtype: Dominance
    :raises RefusedInput: An axis names no measure or judged dimension the input holds.
    """
    known = [*report_input.measures, *report_input.judged]
    unknown = [axis for axis in axes if axis not in known]
    if unknown:
        raise RefusedInput(
            f"axes name {_listed(unknown)}, which the snapshot holds as no measure or judged dimension",
            remedy=f"name measures or judged dimensions by key: {_listed(known)}",
            parameters=("axes",),
            valid_values=known,
        )
    survivors = surviving_options(report_input)
    standing = asked_standing(report_input)
    costs: dict[str, float] = {}
    cost_figures: dict[str, CitedFigure] = {}
    for arm in survivors:
        figure = _number_or_none(f"cost_per_1k.{arm}", report_input)
        if figure is not None and isinstance(figure.value, int | float):
            costs[arm], cost_figures[arm] = float(figure.value), figure

    def at_least(first: str, second: str) -> bool:
        return _cost_at_least(costs, first, second) and _quality_at_least(standing, first, second)

    rows: list[SurvivorRow] = []
    for arm in survivors:
        check = gate_check(report_input, arm)
        held = standing[arm]
        figures = [*([cost_figures[arm]] if arm in cost_figures else []), *_gate_figures(report_input, arm)]
        rows.append(
            SurvivorRow(
                arm=_arm(arm),
                gate_status=check.status,
                borders=list(check.borders),
                standing=held.standing,
                behind=[_arm(other) for other in held.behind],
                why=list(held.why),
                dominated_by=[
                    _arm(other)
                    for other in survivors
                    if other != arm and at_least(other, arm) and not at_least(arm, other)
                ],
                figures=[*figures, *_standing_figures(report_input, arm)],
            )
        )
    kept = set(survivors)
    out = [
        OutRow(arm=_arm(arm.id), why=_out_reasons(report_input, arm.id))
        for arm in report_input.arms
        if arm.id not in kept
    ]
    axis_rows: list[AxisRow] = []
    compared: list[tuple[dict[str, float], bool]] = []
    for axis in dict.fromkeys(axes):
        row, values = _axis(report_input, axis, survivors)
        axis_rows.append(row)
        if values is not None:
            compared.append((values, row.direction == "higher_is_better"))
    dominant = dominant_options(report_input)
    with_axes = [
        arm
        for arm in survivors
        if all(
            at_least(arm, other) and all(_axis_at_least(values, higher, arm, other) for values, higher in compared)
            for other in survivors
            if other != arm
        )
    ]
    declared = decision_at(report_input)
    rule = list(dict.fromkeys(axis for family in report_input.contrast_families for axis in family.merit_axes))
    return Dominance(
        survivors=rows,
        out=out,
        dominant=[_arm(arm) for arm in dominant],
        axes=axis_rows,
        dominant_with_axes=[_arm(arm) for arm in with_axes] if axis_rows else [_arm(arm) for arm in dominant],
        code_call=CodeCall(
            reads_as=CODE_CALL_READS_AS,
            modes=sorted(declared.modes),
            dominant=[_arm(arm) for arm in declared.dominant],
            ranking_rule=rule,
        ),
    )


# =============================================================================
# Sensitivity
# =============================================================================


class Flip(BaseModel):
    """One value an arm's status flips at as the criterion moves."""

    model_config = _MODEL

    arm: str = Field(description="The arm whose status flips, as arm:<id>.")
    where: str | None = Field(
        description="Which flip it is: a floor's held_above or broken_at, a criterion's task or cell; None otherwise."
    )
    at: CitedFigure = Field(description="The value it flips at, as the key that states it.")
    call_changes: bool = Field(description="Whether code's call differs on the two sides of this value.")


class SegmentEnd(BaseModel):
    """One end of a segment: the flip it sits at."""

    model_config = _MODEL

    at: CitedFigure = Field(description="The flip, as the key that states it.")


class Segment(BaseModel):
    """A stretch of the range swept where code's call is the same throughout."""

    model_config = _MODEL

    low: SegmentEnd | None = Field(description="Where it starts: a flip; None at the low end of the range swept.")
    high: SegmentEnd | None = Field(description="Where it ends: a flip; None at the high end of the range swept.")
    modes: list[str] = Field(description="The modes code's rule allows throughout it.")
    dominant: list[str] = Field(description="The one option that dominates throughout it, as arm:<id>; empty: none.")


class Sensitivity(BaseModel):
    """How code's call moves as one declared criterion moves over a range."""

    model_config = _MODEL

    parameter: str = Field(description="The criterion swept, as asked.")
    swept: str = Field(description="The range swept, in words.")
    declared: CitedFigure | None = Field(description="The criterion's declared value; None where none is declared.")
    boundary_belongs_to: Boundary = Field(
        description="Which piece a value exactly at a flip falls in: the piece below it, or the piece above."
    )
    declared_call: Call = Field(description="Code's call at the criteria as declared.")
    reads_as: str = Field(description="What every call here is: code's reading, never a constraint.")
    segments: list[Segment] = Field(description="The range swept, cut where code's call changes, low to high.")
    holds: bool = Field(description="Whether every segment's call is the declared call.")
    flips: list[Flip] = Field(description="The values an arm's status flips at inside the range, low to high.")
    total: int = Field(description="The flips inside the range.")
    offset: int = Field(description="The first flip this page holds.")
    next_offset: int | None = Field(description="The offset of the next page of flips; None when this is the last.")
    notes: list[str] = Field(description="What the sweep could not read, and why.")


@dataclass(frozen=True)
class _Cut:
    """One value an arm's status flips at, as its key states it."""

    arm: str
    where: str | None
    at: CitedFigure

    @property
    def value(self) -> float:
        return _number(self.at)


@dataclass(frozen=True)
class _Sweep:
    """One criterion to sweep: where each status flips, how to read the call at a value, and the range."""

    swept: str
    declared: CitedFigure | None
    boundary: Boundary
    cuts: list[_Cut]
    call_at: Callable[[float], DecisionAt]
    low: float | None
    high: float | None
    notes: list[str]


@dataclass(frozen=True)
class _Piece:
    """A piece of the range between two adjacent flips (None: the range's end), and the call throughout it."""

    low: _Cut | None
    high: _Cut | None
    call: DecisionAt


def _call(decision: DecisionAt) -> Call:
    return Call(modes=sorted(decision.modes), dominant=[_arm(arm) for arm in decision.dominant])


def _inside(low: float | None, high: float | None) -> float:
    """A value strictly inside the stretch from ``low`` to ``high`` (None: without end)."""
    if low is not None and high is not None:
        return (low + high) / 2
    if high is not None:
        return high - 1.0
    if low is not None:
        return low + 1.0
    return 0.0


def _pieces(sweep: _Sweep, cuts: Sequence[_Cut]) -> list[_Piece]:
    """The range cut at every flip inside it, each piece with the call read at a value inside it.

    A value exactly at a flip falls in the piece below it or above it (``sweep.boundary``); where a flip sits at the
    range's own end, that end is a piece of one value.
    """
    first: dict[float, _Cut] = {}
    for cut in cuts:
        first.setdefault(cut.value, cut)
    values = sorted(first)
    pieces: list[_Piece] = []
    start, start_cut = sweep.low, None
    for value in values:
        at_low_end = start_cut is None and sweep.low is not None and value == sweep.low
        if sweep.boundary == "below" and at_low_end:
            pieces.append(_Piece(first[value], first[value], sweep.call_at(value)))
        elif not at_low_end:
            pieces.append(_Piece(start_cut, first[value], sweep.call_at(_inside(start, value))))
        start, start_cut = value, first[value]
    at_high_end = start_cut is not None and sweep.high is not None and start == sweep.high
    if sweep.boundary == "above" and start_cut is not None and at_high_end:
        pieces.append(_Piece(start_cut, start_cut, sweep.call_at(start_cut.value)))
    elif not at_high_end:
        pieces.append(_Piece(start_cut, None, sweep.call_at(_inside(start, sweep.high))))
    return pieces


def _segments(pieces: Sequence[_Piece]) -> list[Segment]:
    """Adjacent pieces with the same call, joined."""
    segments: list[Segment] = []
    run: list[_Piece] = []
    for piece in [*pieces, None]:
        if run and (piece is None or piece.call != run[0].call):
            segments.append(
                Segment(
                    low=SegmentEnd(at=run[0].low.at) if run[0].low is not None else None,
                    high=SegmentEnd(at=run[-1].high.at) if run[-1].high is not None else None,
                    modes=sorted(run[0].call.modes),
                    dominant=[_arm(arm) for arm in run[0].call.dominant],
                )
            )
            run = []
        if piece is not None:
            run.append(piece)
    return segments


def _changes(pieces: Sequence[_Piece], value: float) -> bool:
    """Whether the call differs on the two sides of a flip: the piece ending at it, and the one after."""
    for index, piece in enumerate(pieces[:-1]):
        if piece.high is not None and piece.high.value == value:
            return piece.call != pieces[index + 1].call
    return False


def _parameters(report_input: ReportInput) -> list[str]:
    """Every parameter this input can be swept on."""
    floors = dict.fromkeys(f"floor:{read.aspect}/{read.floor}" for read in report_input.floors)
    return [
        *(f"margin:{aspect.id}" for aspect in report_input.aspects),
        *floors,
        *(f"threshold:{criterion.id}" for criterion in report_input.criteria),
    ]


def _range_words(kind: str, low: float | None, high: float | None, given_low: bool, given_high: bool) -> str:
    start = "from range_low" if given_low else "from zero" if low == 0.0 else "from no lower end"
    end = "to range_high" if given_high else "up, without end"
    return f"every {kind} {start} {end}"


def _margin_sweep(report_input: ReportInput, aspect: Aspect, low: float, high: float | None, words: str) -> _Sweep:
    cuts: list[_Cut] = []
    notes: list[str] = []
    for arm in dict.fromkeys(reading.arm for reading in aspect.readings):
        flip = cite_if(f"aspects.{aspect.id}.flip_margin.{arm}", report_input)
        if flip is None:
            notes.append(f"{_arm(arm)}'s reading states no interval, so it is within no margin and never flips")
            continue
        cuts.append(_Cut(arm, None, flip))
    return _Sweep(
        swept=words,
        declared=cite_if(f"aspects.{aspect.id}.margin", report_input),
        boundary="below",
        cuts=cuts,
        call_at=lambda margin: decision_at(report_input, margins={aspect.id: margin}),
        low=low,
        high=high,
        notes=notes,
    )


def _floor_sweep(report_input: ReportInput, floor: str, low: float, high: float | None, words: str) -> _Sweep:
    cuts: list[_Cut] = []
    notes: list[str] = []
    declared: CitedFigure | None = None
    for read in report_input.floors:
        if read.floor != floor:
            continue
        prefix = f"aspects.{read.aspect}.floor.{read.floor}.{read.arm}"
        if declared is None and read.margin is not None:
            declared = cite(f"{prefix}.margin", report_input)
        if read.interval is None:
            notes.append(f"{_arm(read.arm)}'s floor read states no interval, so its status stays {read.status}")
            continue
        cuts.append(_Cut(read.arm, "held_above", cite(f"{prefix}.held_above", report_input)))
        cuts.append(_Cut(read.arm, "broken_at", cite(f"{prefix}.broken_at", report_input)))
    return _Sweep(
        swept=words,
        declared=declared,
        boundary="below",
        cuts=cuts,
        call_at=lambda margin: decision_at(report_input, floors={floor: margin}),
        low=low,
        high=high,
        notes=notes,
    )


def _moved(report_input: ReportInput, criterion_id: str, threshold: float) -> ReportInput:
    """The input with one criterion's threshold moved, and nothing else."""
    criteria = [
        criterion.model_copy(update={"threshold": threshold}) if criterion.id == criterion_id else criterion
        for criterion in report_input.criteria
    ]
    return report_input.model_copy(update={"criteria": criteria})


def _threshold_sweep(
    report_input: ReportInput, criterion: Criterion, low: float | None, high: float | None, words: str
) -> _Sweep:
    cuts = [
        _Cut(arm.id, place, cite(key, report_input))
        for arm in report_input.arms
        for key, place in _criterion_keys(report_input, criterion, arm.id, "value")
    ]
    notes = (
        []
        if criterion.kind in _GATING
        else [f"a {criterion.kind} criterion warns and never gates, so moving its threshold moves no option"]
    )
    return _Sweep(
        swept=words,
        declared=cite(f"criteria.{criterion.id}.threshold", report_input),
        boundary="below" if criterion.direction == "higher_is_better" else "above",
        cuts=cuts,
        call_at=lambda threshold: decision_at(_moved(report_input, criterion.id, threshold)),
        low=low,
        high=high,
        notes=notes,
    )


def _refuse_parameter(report_input: ReportInput, parameter: str, fault: str) -> RefusedInput:
    return RefusedInput(
        f"parameter {parameter!r} {fault}",
        remedy=f"sweep one of this snapshot's criteria: {_listed(_parameters(report_input))}",
        parameters=("parameter",),
        valid_values=_parameters(report_input),
    )


def _sweep_of(report_input: ReportInput, parameter: str, range_low: float | None, range_high: float | None) -> _Sweep:
    kind, _colon, name = parameter.partition(":")
    if range_low is not None and range_high is not None and range_low > range_high:
        raise RefusedInput(
            "range_low is above range_high",
            remedy="give a range_low at or below range_high, or omit either",
            parameters=("range_low", "range_high"),
        )
    if kind in ("margin", "floor"):
        if range_low is not None and range_low < 0:
            raise RefusedInput(
                f"a {kind} is swept from zero up, and range_low is below zero",
                remedy="give a range_low of zero or more",
                parameters=("range_low",),
            )
        if range_high is not None and range_high < 0:
            raise RefusedInput(
                f"a {kind} is swept from zero up, and range_high is below zero",
                remedy="give a range_high of zero or more",
                parameters=("range_high",),
            )
        low = 0.0 if range_low is None else range_low
        words = _range_words(kind, low, range_high, range_low is not None, range_high is not None)
        if kind == "margin":
            aspect = next((aspect for aspect in report_input.aspects if aspect.id == name), None)
            if aspect is None:
                raise _refuse_parameter(report_input, parameter, "names no aspect this snapshot holds")
            return _margin_sweep(report_input, aspect, low, range_high, f"{words}, on aspect {aspect.id}")
        spelled = {f"{read.aspect}/{read.floor}": read.floor for read in report_input.floors}
        floors = (
            {spelled[name]} if name in spelled else {read.floor for read in report_input.floors if read.floor == name}
        )
        if len(floors) != 1:
            raise _refuse_parameter(report_input, parameter, "names no floor this snapshot reads")
        (floor,) = floors
        return _floor_sweep(report_input, floor, low, range_high, f"{words}, on floor {floor}")
    if kind == "threshold":
        criterion = next((held for held in report_input.criteria if held.id == name), None)
        if criterion is None:
            raise _refuse_parameter(report_input, parameter, "names no criterion this snapshot declares")
        words = _range_words("threshold", range_low, range_high, range_low is not None, range_high is not None)
        return _threshold_sweep(report_input, criterion, range_low, range_high, f"{words}, on criterion {name}")
    raise _refuse_parameter(
        report_input, parameter, "is none of margin:<aspect>, floor:<aspect>/<floor> or threshold:<criterion>"
    )


def sensitivity(
    report_input: ReportInput,
    parameter: str,
    *,
    range_low: float | None = None,
    range_high: float | None = None,
    limit: int = SENSITIVITY_PAGE,
    offset: int = 0,
) -> Sensitivity:
    """How code's call moves as one declared criterion moves: where each arm's status flips, and the call between.

    The range is cut at every value some arm's status flips at (a reading's flip margin, a floor's ``held_above`` and
    ``broken_at``, a criterion's figure for an arm at each place it reads), and code's call is read inside each piece
    (:func:`~threetears.evals.analysis.report_decision.decision_at`; a threshold on the input with that one threshold
    moved). Adjacent pieces with the same call are one segment, so a criterion the call never rests on is one segment.

    :param report_input: The snapshot's report input.
    :ptype report_input: ReportInput
    :param parameter: ``margin:<aspect>``, ``floor:<aspect>/<floor>`` or ``threshold:<criterion>``.
    :ptype parameter: str
    :param range_low: Where the sweep starts; omitted, zero for a margin or a floor, and no lower end for a threshold.
    :ptype range_low: float | None
    :param range_high: Where the sweep ends; omitted, no upper end.
    :ptype range_high: float | None
    :param limit: The most flips one page lists.
    :ptype limit: int
    :param offset: The first flip the page lists.
    :ptype offset: int
    :return: The segments, the flips on this page, and whether the call holds across the range.
    :rtype: Sensitivity
    :raises RefusedInput: The parameter names nothing this input holds, or the range is not one.
    """
    sweep = _sweep_of(report_input, parameter, range_low, range_high)
    inside = sorted(
        (
            cut
            for cut in sweep.cuts
            if (sweep.low is None or cut.value >= sweep.low) and (sweep.high is None or cut.value <= sweep.high)
        ),
        key=lambda cut: cut.value,
    )
    pieces = _pieces(sweep, inside)
    declared = decision_at(report_input)
    page = inside[offset : offset + limit]
    end = offset + len(page)
    notes = list(sweep.notes)
    if len(inside) < len(sweep.cuts):
        notes.append("flips outside the range swept are not listed")
    return Sensitivity(
        parameter=parameter,
        swept=sweep.swept,
        declared=sweep.declared,
        boundary_belongs_to=sweep.boundary,
        declared_call=_call(declared),
        reads_as=CODE_CALL_READS_AS,
        segments=_segments(pieces),
        holds=all(piece.call == declared for piece in pieces),
        flips=[
            Flip(arm=_arm(cut.arm), where=cut.where, at=cut.at, call_changes=_changes(pieces, cut.value))
            for cut in page
        ],
        total=len(inside),
        offset=offset,
        next_offset=end if end < len(inside) else None,
        notes=notes,
    )


# =============================================================================
# Breakdown
# =============================================================================


class SliceArm(BaseModel):
    """One arm in one slice, against the control there."""

    model_config = _MODEL

    arm: str = Field(description="The arm, as arm:<id>.")
    mean: CitedFigure | None = Field(description="Its mean on the measure in the slice.")
    vs_control: CitedFigure | None = Field(description="Its mean minus the control's there; derived, descriptive.")
    loss_share: CitedFigure | None = Field(
        description="The share of its loss against the control its stratification carries in this stratum; derived. "
        "None for a task or cell, or where the loss is not stated."
    )


class Slice(BaseModel):
    """One slice: what it is, how large, the control's figure there, and each arm's."""

    model_config = _MODEL

    id: str = Field(description="The stratum or task id, or the cell as <row>.<column>.")
    aspect: str | None = Field(description="The aspect whose cases a stratum slices; None otherwise.")
    name: CitedFigure = Field(description="What a reader calls it.")
    n_cases: CitedFigure | None = Field(description="The cases in it, where stated.")
    share: CitedFigure | None = Field(description="Its share of traffic, where stated; a cell states none.")
    control: CitedFigure | None = Field(description="The control's mean on the measure there.")
    arms: list[SliceArm] = Field(description="Each arm shown that holds the measure there.")


class Concentration(BaseModel):
    """Where one arm's loss across one stratification sits most."""

    model_config = _MODEL

    arm: str = Field(description="The arm, as arm:<id>.")
    aspect: str | None = Field(description="The aspect whose cases the strata slice.")
    dimension: str | None = Field(description="What the strata are cut by.")
    slice: str = Field(description="The stratum carrying the largest share of the loss.")
    loss_share: CitedFigure = Field(description="The share it carries; derived, descriptive.")


class Breakdown(BaseModel):
    """A measure slice by slice: each arm against the control in each one. Exploratory: no interval, no test."""

    model_config = _MODEL

    measure: str = Field(description="The measure or judged dimension, by key.")
    control: str = Field(description="The control every slice is read against, as arm:<id>.")
    slice_by: SliceBy = Field(description="How the slices are cut: stratum, task or cell.")
    arms: list[str] = Field(description="The arms shown, as arm:<id>.")
    slices: list[Slice] = Field(description="The slices holding the measure, in the input's order.")
    total: int = Field(description="The slices holding the measure.")
    offset: int = Field(description="The first slice this page holds.")
    next_offset: int | None = Field(description="The offset of the next page; None when this is the last.")
    concentrated_in: list[Concentration] = Field(
        description="Per arm and stratification, the stratum carrying the largest share of its loss, over every "
        "slice, not only this page; strata only."
    )
    exploratory: Literal[True] = Field(default=True, description="Always true: descriptive, no interval or test.")
    notes: list[str] = Field(description="What the breakdown could not read, and why.")


#: What a breakdown with no control is refused with, naming the read that needs none.
_NO_CONTROL_REMEDY = "read the measure in each stratum with measure_read (by=stratum), which needs no control"


@dataclass(frozen=True)
class _SliceKey:
    """One slice, as the keys that read it name it."""

    id: str
    prefix: str
    aspect: str | None
    dimension: str | None
    stratum: bool
    share: bool
    per_arm: Mapping[str, Mapping[str, object]]


def _stratum_slices(report_input: ReportInput, aspect: str | None, notes: list[str]) -> list[_SliceKey]:
    """The strata (on one aspect, where named), each by the key prefix that names it alone."""
    ids = [stratum.id for stratum in report_input.strata]
    on_aspect = [(stratum.aspect, stratum.id) for stratum in report_input.strata]
    found: list[_SliceKey] = []
    for stratum in report_input.strata:
        if aspect is not None and stratum.aspect != aspect:
            continue
        if ids.count(stratum.id) == 1:
            prefix = f"strata.{stratum.id}"
        elif stratum.aspect is not None and on_aspect.count((stratum.aspect, stratum.id)) == 1:
            prefix = f"aspects.{stratum.aspect}.strata.{stratum.id}"
        else:
            notes.append(f"stratum {stratum.id} is held more than once on one aspect, so no key names it; left out")
            continue
        found.append(_SliceKey(stratum.id, prefix, stratum.aspect, stratum.dimension, True, True, stratum.per_arm))
    return found


def _slices(report_input: ReportInput, slice_by: SliceBy, aspect: str | None, notes: list[str]) -> list[_SliceKey]:
    if slice_by == "stratum":
        return _stratum_slices(report_input, aspect, notes)
    if aspect is not None:
        raise RefusedInput(
            f"aspect_filter narrows strata, and a {slice_by} is not held on an aspect",
            remedy="omit aspect_filter, or slice by stratum",
            parameters=("aspect_filter",),
        )
    if slice_by == "task":
        return [
            _SliceKey(task.id, f"task.{task.id}", None, None, False, True, task.per_arm) for task in report_input.tasks
        ]
    spellings = [f"{cell.row}.{cell.column}" for cell in report_input.cells.values()]
    found: list[_SliceKey] = []
    for cell, spelled in zip(report_input.cells.values(), spellings, strict=True):
        if spellings.count(spelled) > 1:
            notes.append(f"cell {spelled} is spelled alike by more than one cell, so no key names it; left out")
            continue
        found.append(_SliceKey(spelled, f"cells.{spelled}", None, None, False, False, cell.per_arm))
    return found


def _sliced(report_input: ReportInput, slice_by: SliceBy, aspect: str | None, arms: Sequence[str]) -> list[str]:
    """The measures and judged dimensions a breakdown can read this way: those a slice holds for one of ``arms``.

    Empty where the slicing itself is refused (an aspect named for tasks or cells, :func:`_slices`), whose own refusal
    says so.
    """
    if aspect is not None and slice_by != "stratum":
        return []
    known = {*report_input.measures, *report_input.judged}
    every = _slices(report_input, slice_by, aspect, [])
    held = (key for found in every for name in arms for key in found.per_arm.get(name, {}))
    return [key for key in dict.fromkeys(held) if key in known]


def _loss_share(
    report_input: ReportInput, held: _SliceKey, arm: str, measure: str, notes: dict[str, str]
) -> CitedFigure | None:
    """An arm's loss share in one stratum, or None with the reason noted once per arm."""
    try:
        return cite(f"{held.prefix}.loss_share.{arm}.{measure}", report_input)
    except UnresolvableReference as refused:
        # NOSILENT: a loss share the input cannot state (no loss overall, a stratum without the figure) is shown absent
        # and the refusal's own words go to the result's notes, so the caller reads why.
        notes.setdefault(arm, f"no loss share for {_arm(arm)}: {refused}")
        return None


def _slice_row(
    report_input: ReportInput, held: _SliceKey, control: str, arms: Sequence[str], measure: str, notes: dict[str, str]
) -> Slice:
    def mean(arm: str) -> CitedFigure | None:
        return cite_if(f"{held.prefix}.{arm}.{measure}.mean", report_input)

    control_holds = measure in held.per_arm.get(control, {})
    rows = [
        SliceArm(
            arm=_arm(arm),
            mean=mean(arm),
            vs_control=cite_if(f"{held.prefix}.{arm}.{measure}.vs_control", report_input) if control_holds else None,
            loss_share=_loss_share(report_input, held, arm, measure, notes) if held.stratum else None,
        )
        for arm in arms
        if measure in held.per_arm.get(arm, {})
    ]
    return Slice(
        id=held.id,
        aspect=held.aspect,
        name=cite(f"{held.prefix}.name", report_input),
        n_cases=cite_if(f"{held.prefix}.n_cases", report_input),
        share=cite_if(f"{held.prefix}.share", report_input) if held.share else None,
        control=mean(control) if control_holds else None,
        arms=rows,
    )


def _concentrated(slices: Sequence[tuple[_SliceKey, Slice]], arms: Sequence[str]) -> list[Concentration]:
    """Per arm and stratification, the stratum carrying the largest share of the arm's loss."""
    found: list[Concentration] = []
    for arm in arms:
        groups: dict[tuple[str | None, str | None], list[tuple[str, CitedFigure]]] = {}
        for held, row in slices:
            for shown in row.arms:
                if shown.arm == _arm(arm) and shown.loss_share is not None:
                    groups.setdefault((held.aspect, held.dimension), []).append((held.id, shown.loss_share))
        for (aspect, dimension), shares in groups.items():
            top_id, top = max(shares, key=lambda pair: _number(pair[1]))
            found.append(Concentration(arm=_arm(arm), aspect=aspect, dimension=dimension, slice=top_id, loss_share=top))
    return found


def breakdown(
    report_input: ReportInput,
    measure: str,
    *,
    arm: str | None = None,
    slice_by: SliceBy = "stratum",
    aspect: str | None = None,
    limit: int = BREAKDOWN_PAGE,
    offset: int = 0,
) -> Breakdown:
    """A measure slice by slice: each candidate's mean, its difference from the control, and its share of the loss.

    Exploratory, on point estimates: every figure is the slice's own (``strata.*``, ``task.*``, ``cells.*``), its
    difference from the control (``vs_control``) and, across one stratification's strata, the share of the candidate's
    loss each carries (``loss_share``), all descriptive.

    :param report_input: The snapshot's report input.
    :ptype report_input: ReportInput
    :param measure: The measure or judged dimension, by key.
    :ptype measure: str
    :param arm: The one candidate to show; None for every candidate.
    :ptype arm: str | None
    :param slice_by: ``stratum``, ``task`` or ``cell``.
    :ptype slice_by: SliceBy
    :param aspect: For strata, only those slicing this aspect's cases.
    :ptype aspect: str | None
    :param limit: The most slices one page holds.
    :ptype limit: int
    :param offset: The first slice the page holds.
    :ptype offset: int
    :return: The slices on this page, and where each candidate's loss sits most.
    :rtype: Breakdown
    :raises RefusedInput: The input has no control; the arm is no candidate; the measure is none the input holds, or
        no slice holds it; or an aspect is named for tasks or cells.
    """
    control = control_arm(report_input)
    if control is None:
        raise RefusedInput(
            "measure_breakdown reads each slice against the control, and this snapshot has no control",
            remedy=_NO_CONTROL_REMEDY,
        )
    known = [*report_input.measures, *report_input.judged]
    candidates = list(dict.fromkeys(held.id for held in report_input.arms if held.id != control))
    if measure not in known:
        raise RefusedInput(
            f"measure_key {measure!r} names no measure or judged dimension this snapshot holds",
            remedy=f"name one by key: {_listed(known)}",
            parameters=("measure_key",),
            valid_values=_sliced(report_input, slice_by, aspect, (control, *candidates)) or known,
        )
    if arm is not None and arm not in candidates:
        fault = "is the control every slice is read against" if arm == control else "names no candidate"
        raise RefusedInput(
            f"arm_filter {arm!r} {fault}",
            remedy=f"name a candidate: {_listed(candidates)}",
            parameters=("arm_filter",),
            valid_values=candidates,
        )
    arms = [arm] if arm is not None else candidates
    notes: list[str] = []
    every = _slices(report_input, slice_by, aspect, notes)
    holding = [held for held in every if any(measure in held.per_arm.get(name, {}) for name in (control, *arms))]
    if not holding:
        keys = list(dict.fromkeys(key for held in every for figures in held.per_arm.values() for key in figures))
        raise RefusedInput(
            f"no {slice_by} in this snapshot holds {measure!r}" + ("" if every else f": it holds no {slice_by}"),
            remedy=f"name a measure the slices hold ({_listed(keys)}), or slice another way",
            parameters=("measure_key",),
            valid_values=_sliced(report_input, slice_by, aspect, (control, *arms)),
        )
    loss_notes: dict[str, str] = {}
    rows = [(held, _slice_row(report_input, held, control, arms, measure, loss_notes)) for held in holding]
    page = rows[offset : offset + limit]
    end = offset + len(page)
    return Breakdown(
        measure=measure,
        control=_arm(control),
        slice_by=slice_by,
        arms=[_arm(name) for name in arms],
        slices=[row for _held, row in page],
        total=len(rows),
        offset=offset,
        next_offset=end if end < len(rows) else None,
        concentrated_in=_concentrated(rows, arms),
        notes=[*notes, *loss_notes.values()],
    )
