"""The analyst's analyses: what code computes over the frozen report input that no single figure states.

Three analyses, each over the snapshot's report input alone and each returning every number as the figure key that
names it (:mod:`~threetears.evals.analysis.analyst.cited`), so a sentence the analyst writes from one cites keys the
renderer prints the same:

- :func:`dominance`: which options survive the gates and which dominates, on the decision's own axes (every gate,
  cost per 1,000 of what the cost counts, the asked aspects) as :mod:`~threetears.evals.analysis.report_decision`
  reads them, with the extra axes the analyst names compared beside them on point estimates, descriptively.
- :func:`sensitivity`: how the decision moves as one declared criterion moves (an aspect's margin, a floor's margin,
  a gate's threshold), or as a hypothetical line the snapshot does not declare moves: the value each arm's status
  flips at, and the call and each routable task's pick on every piece of the range between them.
- :func:`breakdown`: where a candidate's difference from the control sits, slice by slice (strata, tasks or
  target-matrix cells), and, across strata, which slice carries its loss.

**Code's call is a reading, not a constraint.** The modes and the dominant option these return are what code's rule
reads off the declared criteria: interpretive facts for the analyst and the editor to weigh, never a bound on what a
report may say. Each result says so where it carries one.

**No dominance from cost alone.** Where the snapshot declares nothing but cost (no gate, guardrail, control, asked
aspect or ranking rule: :func:`~threetears.evals.analysis.report_decision.weighed_axes`), the dominance rule reduces
to a cost ranking, and the cheapest option would read as dominant though nothing declared says it is good enough.
These analyses make no dominance call there: no option is dominant or dominated, code's call is a tradeoff, and the
result says why (``withheld``). The extra axes the caller names are weighed beside cost, descriptively
(``dominant_with_axes``, each survivor's ``dominated_with_axes``). Each axis weighed is one the snapshot declares or
the caller names; none is added.

**Per-slice picks.** Where the snapshot holds routable tasks, a sweep reads each one's pick beside the one-option
call (:func:`~threetears.evals.analysis.report_gates.routing_picks`: the cheapest arm passing in the task and in every
cross-cutting task), so a gate that moves a routed plan without moving the one-option call shows it.

**A sweep's reach.** With no range, a sweep walks every flip: a threshold or a hypothetical line from no lower end
to no upper end, a margin or a floor from zero up. A range the caller gives can stop short of a flip where the call or
a routable task's pick changes; the result then says so for each side it cut (``unswept``): the nearest such flip past
the range's end, each such flip by its key (a page's worth), the call past it, and whether that side loosens the line.
A flip exactly on the range's end is past it where the end's value falls in the piece inside the range, since the
change at it lies just beyond (the line at a pass-rate gate's flip, swept up to it, still reads the arm passing; the
call changes just above). Every flip's ``call_changes`` and ``picks_change`` read the two sides of its value whatever
range is given, so a flip on the range's end says what happens just past it. It never leaves the caller to infer it
from a note. A margin sweep over an aspect with no reading of some candidate says so too: code's rule
reads a margin against a reading alone, so no margin moves the call on that candidate, and a contrast on the aspect is
not read against a margin.

**Which arms the per-task picks hold back.** A task that runs inside every other task (cross-cutting) bars an arm
failing a gate there from every routed task, even one whose own gates it passes. Where the snapshot holds routable and
cross-cutting tasks, the sweep lists each arm so barred at the criteria as declared (``barred``): the cross-cutting
task, each gate it fails there by its keys, and the routable tasks whose own gates it passes, so a per-task pick is
never silently cheaper or dearer than a reader expects.

**Hypothetical lines.** ``hypothetical:<measure>`` (or ``hypothetical:<measure>/<statistic>``) sweeps a gate the
snapshot does not declare, on one measure, in its stated direction, read where the snapshot states the statistic
(overall, else in each task, else in each cell): the call at every line, as if that gate held. The result is marked
hypothetical; its flips are the arms' own figures.

**How bars are read is the caller's choice** (``bars_as``). A declared bar warns and never removes an option, as the
snapshot declares it (``warnings``, the default); the analyst may read the bars as gates instead (``gates``), when
the question it answers makes them the condition, and every result says which reading it took. Code never chooses
the reading, from the question or from anything else: the parameter is the analyst's. Read as gates, a bar removes
an option that misses it and its threshold moves the call in a sweep; read as warnings, dominance lists each bar (and
each floor criterion) a survivor misses or borders as a warning beside its gates, and a bar's sweep lists where each
arm's figure crosses its line while the call holds.

**Unknown is not a fail.** A gate with no figure for an option at a place it reads is unknown there
(:mod:`~threetears.evals.analysis.report_margins`): the option is not shown to pass, so it does not survive, and it is
not removed by a figure either. Dominance lists such an option as undecided, with each place it is unknown at; its
removed options list only the figures they fail on, and their unknown places beside. Code never infers a figure the
snapshot does not state. Where a gate read in each task or cell has no figure there, and the option's figure over
every case misses the line, the result hands the analyst and the editor the implied fact as an interpretive note,
labelled inferred and conditional (if the places split the cases between them, some place misses too), never as a
verdict.

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
    weighed_axes,
)
from threetears.evals.analysis.report_gates import GateFinding, GateStatus, gate_check, gate_findings, routing_picks
from threetears.evals.analysis.report_input import Aspect, Contrast, Criterion, ReportInput
from threetears.evals.analysis.report_margins import (
    CriterionFigure,
    criterion_aspect,
    criterion_figures,
    criterion_place,
    criterion_unknowns,
    lead_aspect,
    undecided_arms,
)
from threetears.evals.kernel.errors import RefusedInput

__all__ = [
    "BREAKDOWN_PAGE",
    "DOMINANCE_PAGE",
    "SENSITIVITY_PAGE",
    "BarsAs",
    "Breakdown",
    "Dominance",
    "GateRead",
    "Sensitivity",
    "SliceBy",
    "breached_guardrails",
    "breakdown",
    "dominance",
    "gate_reads",
    "sensitivity",
]

#: What every code call these analyses carry reads as, so neither the analyst nor the editor takes it for a rule.
CODE_CALL_READS_AS = (
    "code's reading of the declared criteria: an interpretive fact for the analyst and the editor to weigh, never a "
    "constraint on what the report says"
)

#: The criterion kinds that gate an option, as the gate check reads them.
_GATING = frozenset({"gate", "guardrail"})

#: The criterion kinds that warn and never gate, as the snapshot declares them.
_WARNING = frozenset({"bar", "floor"})

#: How the caller reads the declared bars: as the warnings the snapshot declares them, or as gates.
BarsAs = Literal["warnings", "gates"]

#: What an implied fact is, said on every one.
_IMPLIED_READS_AS = (
    "inferred, not stated: an interpretive fact for the analyst and the editor to weigh, conditional on the places "
    "splitting the cases between them; never a verdict, and no gate reads it"
)

#: The slices one breakdown page holds unless the caller asks for more: a slice reads one line per figure, per arm.
BREAKDOWN_PAGE = 12

#: The options one dominance page holds unless the caller asks for more.
DOMINANCE_PAGE = 50

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
    withheld: str = Field(
        description="Why code makes no dominance call where its rule alone would name the cheapest option: cost is "
        "the only axis the snapshot declares. Empty where it makes its call."
    )


class GateRead(BaseModel):
    """One gate or guardrail figure an option fails or holds borderline, at one place, by the keys that state it."""

    model_config = _MODEL

    criterion: str = Field(description="The gate or guardrail, by id, as `criteria.<criterion>.` keys name it.")
    place: str | None = Field(
        description="Where the figure sits, as its key names it: a task's id or a cell's <row>.<column>; None overall."
    )
    status: Literal["fail", "borderline", "unknown"] = Field(
        description="fail: the figure misses the threshold; borderline: its interval crosses it, which counts as "
        "passing; unknown: the input states no figure there, so the option is neither shown to pass nor shown to "
        "fail there."
    )
    figures: list[CitedFigure] = Field(
        description="The threshold, then the figure, its interval's ends where stated, and its gap toward passing."
    )


def _criterion_key(
    report_input: ReportInput, criterion: Criterion, arm: str, figure: CriterionFigure, field: str
) -> str:
    """The key of one figure a criterion reads for an arm, at ``field``: by its place where the criterion reads the
    arm at more than one (the form ``_criterion_keys`` writes)."""
    held = criterion_figures(report_input, criterion, arm)
    place = _place(report_input, figure)
    if len(held) == 1 or place is None:
        return f"criteria.{criterion.id}.{arm}.{field}"
    return f"criteria.{criterion.id}.{arm}.{place}.{field}"


def _gate_read(report_input: ReportInput, arm: str, finding: GateFinding) -> GateRead:
    criterion = next(held for held in report_input.criteria if held.id == finding.criterion)
    figures = [cite(f"criteria.{criterion.id}.threshold", report_input)]
    figure = finding.figure
    if figure is not None:
        keys = [
            _criterion_key(report_input, criterion, arm, figure, field) for field in ("value", "low", "high", "gap")
        ]
        figures += _stated(keys, report_input)
        where = _place(report_input, figure)
    else:
        where = _finding_place(report_input, finding)
    return GateRead(criterion=criterion.id, place=where, status=finding.status, figures=figures)


def _finding_place(report_input: ReportInput, finding: GateFinding) -> str | None:
    """Where a finding with no figure sits, as a key would name it."""
    if finding.cell is not None:
        cell = report_input.cells[finding.cell]
        return f"{cell.row}.{cell.column}"
    return finding.task


def _stated(keys: Sequence[str], report_input: ReportInput) -> list[CitedFigure]:
    """The figures the keys name that the input states, in the keys' order."""
    return [figure for key in keys if (figure := cite_if(key, report_input)) is not None]


def gate_reads(report_input: ReportInput, arm: str, status: Literal["fail", "borderline", "unknown"]) -> list[GateRead]:
    """Every figure an arm fails, holds borderline or has none at, on the gates and guardrails, each by the keys that
    state it: the gate check's own findings (:func:`~threetears.evals.analysis.report_gates.gate_findings`), keyed, so
    a reason's numbers reach a reader as figures.

    :param report_input: The snapshot's report input.
    :ptype report_input: ReportInput
    :param arm: The arm's id.
    :ptype arm: str
    :param status: ``fail`` for the figures it fails on, ``borderline`` for the ones it holds borderline, ``unknown``
        for the places with no figure.
    :ptype status: Literal["fail", "borderline", "unknown"]
    :return: Each, in the criteria's order.
    :rtype: list[GateRead]
    """
    return [
        _gate_read(report_input, arm, finding)
        for finding in gate_findings(report_input, arm)
        if finding.status == status
    ]


def _read_as_gates(report_input: ReportInput, kinds: frozenset[str]) -> ReportInput:
    """The input with every criterion of ``kinds`` read as a gate, and nothing else changed: each criterion's figures,
    and so every key citing them, read as they did."""
    criteria = [
        criterion.model_copy(update={"kind": "gate"}) if criterion.kind in kinds else criterion
        for criterion in report_input.criteria
    ]
    return report_input.model_copy(update={"criteria": criteria})


def _read_bars_as(report_input: ReportInput, bars_as: BarsAs) -> ReportInput:
    """The input as the caller reads its bars (this module's How bars are read): as declared for ``warnings``, every
    bar read as a gate for ``gates``. A bar's figures, and every key citing them, read the same either way.

    :param report_input: The snapshot's report input.
    :ptype report_input: ReportInput
    :param bars_as: ``warnings`` or ``gates``, as the caller chose.
    :ptype bars_as: BarsAs
    :return: The input read so.
    :rtype: ReportInput
    """
    return report_input if bars_as == "warnings" else _read_as_gates(report_input, frozenset({"bar"}))


def _warnings(report_input: ReportInput, arm: str) -> list[GateRead]:
    """Each figure an arm misses, holds borderline or has none at on the criteria that warn (bars and floors, as the
    input reads them), by the keys that state it: what the gate check would find were they gates."""
    warning = {criterion.id for criterion in report_input.criteria if criterion.kind in _WARNING}
    if not warning:
        return []
    as_gates = _read_as_gates(report_input, _WARNING)
    return [
        _gate_read(report_input, arm, found) for found in gate_findings(as_gates, arm) if found.criterion in warning
    ]


class ImpliedFact(BaseModel):
    """What the figures an option does have imply about a place a gate has none for it: inferred, never a verdict."""

    model_config = _MODEL

    criterion: str = Field(description="The gate or guardrail, by id, whose places hold no figure for the option.")
    reads_as: str = Field(description="What this is: inferred and conditional, an interpretive fact, never a verdict.")
    statement: str = Field(description="The inference, in words.")
    figures: list[CitedFigure] = Field(
        description="The threshold, then the option's figure over every case that the inference rests on."
    )


#: What a place is called in an implied fact, by where the criterion reads.
_PLACE_WORDS = {"task": "task", "cell": "target-matrix cell"}


def _implied(report_input: ReportInput, arm: str) -> list[ImpliedFact]:
    """For each gate read in every task or every cell with no figure for the arm somewhere, where the arm's figure over
    every case (the lead aspect's, at the criterion's statistic) misses the line: the implied fact that some place
    misses it too, if the places split the cases between them (this module's Unknown is not a fail)."""
    found: list[ImpliedFact] = []
    lead = lead_aspect(report_input)
    for criterion in report_input.criteria:
        place = criterion_place(criterion)
        if (
            criterion.kind not in _GATING
            or place == "overall"
            or criterion.task is not None
            or criterion_aspect(report_input, criterion) != lead
            or not criterion_unknowns(report_input, criterion, arm)
        ):
            continue
        overall = criterion.model_copy(update={"scope": "overall", "task": None})
        figures = criterion_figures(report_input, overall, arm)
        if not figures:
            continue
        threshold = float(criterion.threshold)
        value = figures[0].value
        if not (value < threshold if criterion.direction == "higher_is_better" else value > threshold):
            continue
        cited = cite_if(
            _figure_key(report_input, criterion.measure, arm, criterion.statistic, figures[0]), report_input
        )
        if cited is None:
            continue
        words = _PLACE_WORDS[place]
        found.append(
            ImpliedFact(
                criterion=criterion.id,
                reads_as=_IMPLIED_READS_AS,
                statement=f"{_arm(arm)}'s {criterion.statistic} over every case misses the line; if the {words}s "
                f"split the cases between them (none in two, none in none), at least one {words}'s {criterion.statistic} "
                "misses it too, so the gate would not pass in every one",
                figures=[cite(f"criteria.{criterion.id}.threshold", report_input), cited],
            )
        )
    return found


def breached_guardrails(report_input: ReportInput, arm: str) -> list[str]:
    """The guardrails the pipeline read breached for an arm, by the name the input gives each (its dimension,
    measure or id); ``guardrail`` where it gives none."""
    return [
        str(guardrail.get("dimension") or guardrail.get("measure") or guardrail.get("id") or "guardrail")
        for guardrail in report_input.guardrails
        if str(guardrail.get("arm")) == arm and guardrail.get("decision") == "breached"
    ]


class SurvivorRow(BaseModel):
    """One option that survives the gates, its standing on the decision's axes, and the figures behind it."""

    model_config = _MODEL

    arm: str = Field(description="The option, as arm:<id>.")
    gate_status: GateStatus = Field(description="pass or borderline: borderline counts as passing.")
    borders: list[GateRead] = Field(
        description="Each figure it holds borderline on, by the keys that state it; empty when it passes."
    )
    warnings: list[GateRead] = Field(
        description="Each figure it misses, holds borderline or has none at on a criterion that warns (a bar read as "
        "a warning, a floor criterion), on the criterion's own aspect, by the keys that state it; it stays a survivor."
    )
    standing: StandingWord = Field(description="Its standing on the asked aspects.")
    behind: list[str] = Field(description="In a choice, the arms an asked pair shows ahead of it, as arm:<id>.")
    why: list[str] = Field(
        description="Why a candidate is not shown no worse on the asked aspects, each floor and aspect by its id; the "
        "figures of each floor it is not shown to hold are among its figures."
    )
    dominated_by: list[str] = Field(
        description="The survivors at least as good on every axis while it is not as good as them, as arm:<id>; "
        "empty for every survivor where cost is the only axis declared."
    )
    dominated_with_axes: list[str] = Field(
        description="The survivors at least as good on cost, the declared axes and every extra axis with a direction, "
        "while it is not as good as them; descriptive. Equal to dominated_by when no axis is asked for."
    )
    figures: list[CitedFigure] = Field(
        description="Its cost per 1,000 of what the cost counts, its gap from each gate, and its asked readings, "
        "pairs and contrasts."
    )


class OutRow(BaseModel):
    """One option the gates or a guardrail removed, and why."""

    model_config = _MODEL

    arm: str = Field(description="The option, as arm:<id>.")
    gates: list[GateRead] = Field(description="Each gate or guardrail figure it fails on, by the keys that state it.")
    breached: list[str] = Field(description="Each guardrail the pipeline read breached for it, by name.")
    unknown: list[GateRead] = Field(
        description="Each place a gate or guardrail has no figure for it: unknown there, not among what removed it."
    )


class UndecidedRow(BaseModel):
    """One option neither shown to pass nor removed: no gate figure fails it, and a gate has no figure for it."""

    model_config = _MODEL

    arm: str = Field(description="The option, as arm:<id>.")
    unknown: list[GateRead] = Field(
        description="Each place a gate or guardrail has no figure for it, with the threshold it would be read against."
    )
    borders: list[GateRead] = Field(description="Each figure it holds borderline on, by the keys that state it.")
    warnings: list[GateRead] = Field(
        description="Each figure it misses, holds borderline or has none at on a criterion that warns, by its keys."
    )
    implied: list[ImpliedFact] = Field(
        description="What its figures over every case imply about the places with none, inferred and labelled so; "
        "never a verdict, and empty where nothing is implied."
    )


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
    arms: list[AxisArm] = Field(description="Each survivor's figure on this page, in the input's arm order.")


class Dominance(BaseModel):
    """Which options survive, which dominates, and what extra axes would say beside them."""

    model_config = _MODEL

    survivors: list[SurvivorRow] = Field(
        description="The options on this page that survive the gates, in the input's arm order."
    )
    out: list[OutRow] = Field(description="The options on this page the gates or a guardrail removed.")
    undecided: list[UndecidedRow] = Field(
        description="The options on this page neither shown to pass nor removed: a gate has no figure for them "
        "somewhere and none they have fails. Code's call reads only the survivors."
    )
    bars_as: BarsAs = Field(
        description="How the declared bars were read, as the caller chose: warnings (they never remove an option) or "
        "gates."
    )
    weighs: list[str] = Field(
        description="The axes code's decision weighs, as the snapshot declares them: gates, control, asked, ranking, "
        "cost."
    )
    dominant: list[str] = Field(
        description="The survivors each at least as good as every other on every axis; two or more tie, none: none "
        "dominates, or cost is the only axis declared (code_call.withheld says so).",
    )
    axes: list[AxisRow] = Field(description="The extra axes asked for, each compared on point estimates.")
    dominant_with_axes: list[str] = Field(
        description="The survivors at least as good as every other on the decision's axes and every extra axis with "
        "a direction; descriptive. Equal to dominant when no axis is asked for.",
    )
    code_call: CodeCall = Field(description="Code's call on the declared criteria: its reading, never a constraint.")
    total: int = Field(
        description="The options the snapshot holds; a page lists the survivors, the removed and the undecided among "
        "them."
    )
    offset: int = Field(description="The first option this page holds, in the input's arm order.")
    next_offset: int | None = Field(description="The offset of the next page of options; None when this is the last.")
    notes: list[str] = Field(description="How the result was read, where that is not the default, and why.")


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
    that show an arm ahead, the contrasts on an asked aspect that separate it from the control, and each asked floor
    it is not read to hold."""
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
    asked_ids = {aspect.id for aspect in asked}
    for read in report_input.floors:
        if read.arm == arm and read.aspect in asked_ids and read.status != "held":
            prefix = f"aspects.{read.aspect}.floor.{read.floor}.{arm}"
            found += [figure for field in _FLOOR_FIELDS if (figure := cite_if(f"{prefix}.{field}", report_input))]
    return found


#: What a floor an arm is not shown to hold is cited by: its status, the margin, how far the drop could go and by how
#: much it misses, and the input's own reason.
_FLOOR_FIELDS = ("status", "margin", "worst_drop", "short_by", "reason")


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


#: What code's call says where cost is the only axis the snapshot declares.
_COST_ALONE = (
    "cost is the only axis this snapshot declares (no gate, guardrail, control, asked aspect or ranking rule), so "
    "code makes no dominance call: its rule would name the cheapest option, which nothing declared shows good "
    "enough. Name quality axes in axes to weigh them beside cost (dominant_with_axes), descriptively"
)


def _cost_alone(report_input: ReportInput) -> bool:
    """Whether cost is the only axis the snapshot declares, so the dominance rule would be a cost ranking."""
    return set(weighed_axes(report_input)) <= {"cost"}


def _code_reading(report_input: ReportInput, decision: DecisionAt) -> DecisionAt:
    """Code's call as the analyses read it: the decision, save that where cost is the only axis declared and more
    than one option survives, no option dominates and the call is a tradeoff."""
    if decision.dominant and _cost_alone(report_input) and len(surviving_options(report_input)) > 1:
        return DecisionAt(modes=frozenset({"tradeoff"}), dominant=())
    return decision


#: What a dominance result says where some options are undecided.
_UNDECIDED_NOTE = (
    "undecided: {arms}. A gate or guardrail has no figure for them at some place it reads and none they have fails, "
    "so they are neither shown to pass nor removed; code's call reads only the survivors. Each one's unknown places "
    "are listed, with any fact its figures over every case imply, labelled inferred"
)

#: What a sweep says where some options are undecided at the criteria as declared.
_SWEEP_UNDECIDED_NOTE = (
    "undecided at the criteria as declared: {arms}. A gate or guardrail has no figure for them at some place it reads "
    "and none they have fails, so no value swept passes them there; code's call reads only the survivors"
)


def _named_arms(arms: Sequence[str]) -> str:
    """Arms as a note names them, as arm:<id>, the first few and how many more."""
    named = ", ".join(_arm(arm) for arm in arms[:_SHOWN])
    return named + (f", and {len(arms) - _SHOWN} more" if len(arms) > _SHOWN else "")


#: What a result says where the caller read the bars as gates.
_BARS_AS_GATES_NOTE = (
    "bars read as gates, as the caller chose (bars_as=gates): an option missing a declared bar is removed, which the "
    "snapshot declares as a warning"
)


def dominance(
    report_input: ReportInput,
    axes: Sequence[str] = (),
    *,
    limit: int | None = None,
    offset: int = 0,
    bars_as: BarsAs = "warnings",
) -> Dominance:
    """Which options survive the gates, which dominates on the decision's axes, and what extra axes say beside them.

    A page holds the options from ``offset``, ``limit`` of them in the input's arm order (every one where ``limit`` is
    None): each survivor's row, each removed option's, each undecided one's, and each survivor's figure on every extra
    axis. What is read over every option (the dominant ones, the best on each axis, code's call) is on every page.
    The declared bars are read as ``bars_as`` says (this module's How bars are read), every figure cited as the
    snapshot states it.

    The survivors, the dominant options and code's call are :mod:`~threetears.evals.analysis.report_decision`'s own
    (:func:`~threetears.evals.analysis.report_decision.surviving_options`,
    :func:`~threetears.evals.analysis.report_decision.dominant_options`,
    :func:`~threetears.evals.analysis.report_decision.decision_at`), save where cost is the only axis the snapshot
    declares: there no option is dominant or dominated and the call is a tradeoff (this module's No dominance from
    cost alone). ``dominated_by`` applies the same pairwise rule, cost per 1,000 of what the cost counts and the asked
    standing, to each pair of survivors; ``dominated_with_axes`` adds the extra axes.

    :param report_input: The snapshot's report input.
    :ptype report_input: ReportInput
    :param axes: Extra measures or judged dimensions, by key, to compare the survivors on, descriptively.
    :ptype axes: Sequence[str]
    :param limit: The most options one page holds; None for every one.
    :ptype limit: int | None
    :param offset: The first option the page holds, in the input's arm order.
    :ptype offset: int
    :param bars_as: How to read the declared bars: ``warnings`` as declared, or ``gates``.
    :ptype bars_as: BarsAs
    :return: The survivors, the options removed and undecided, the dominant ones, the extra axes and code's call.
    :rtype: Dominance
    :raises RefusedInput: An axis names no measure or judged dimension the input holds.
    """
    report_input = _read_bars_as(report_input, bars_as)
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
    alone = _cost_alone(report_input) and len(survivors) > 1
    costs: dict[str, float] = {}
    cost_figures: dict[str, CitedFigure] = {}
    for arm in survivors:
        figure = _number_or_none(f"cost_per_1k.{arm}", report_input)
        if figure is not None and isinstance(figure.value, int | float):
            costs[arm], cost_figures[arm] = float(figure.value), figure

    def at_least(first: str, second: str) -> bool:
        return _cost_at_least(costs, first, second) and _quality_at_least(standing, first, second)

    axis_rows: list[AxisRow] = []
    compared: list[tuple[dict[str, float], bool]] = []
    for axis in dict.fromkeys(axes):
        row, values = _axis(report_input, axis, survivors)
        axis_rows.append(row)
        if values is not None:
            compared.append((values, row.direction == "higher_is_better"))

    def with_axes(first: str, second: str) -> bool:
        return at_least(first, second) and all(
            _axis_at_least(values, higher, first, second) for values, higher in compared
        )

    def dominated(arm: str, by: Callable[[str, str], bool]) -> list[str]:
        return [_arm(other) for other in survivors if other != arm and by(other, arm) and not by(arm, other)]

    order = [arm.id for arm in report_input.arms]
    end = len(order) if limit is None else min(offset + limit, len(order))
    window = set(order[offset:end])
    rows: list[SurvivorRow] = []
    for arm in survivors:
        if arm not in window:
            continue
        check = gate_check(report_input, arm)
        held = standing[arm]
        figures = [*([cost_figures[arm]] if arm in cost_figures else []), *_gate_figures(report_input, arm)]
        dominated_by = [] if alone else dominated(arm, at_least)
        rows.append(
            SurvivorRow(
                arm=_arm(arm),
                gate_status=check.status,
                borders=gate_reads(report_input, arm, "borderline"),
                warnings=_warnings(report_input, arm),
                standing=held.standing,
                behind=[_arm(other) for other in held.behind],
                why=list(held.why),
                dominated_by=dominated_by,
                dominated_with_axes=dominated(arm, with_axes) if axis_rows else dominated_by,
                figures=[*figures, *_standing_figures(report_input, arm)],
            )
        )
    undecided = undecided_arms(report_input)
    kept = set(survivors) | set(undecided)
    out = [
        OutRow(
            arm=_arm(arm.id),
            gates=gate_reads(report_input, arm.id, "fail"),
            breached=breached_guardrails(report_input, arm.id),
            unknown=gate_reads(report_input, arm.id, "unknown"),
        )
        for arm in report_input.arms
        if arm.id not in kept and arm.id in window
    ]
    undecided_rows = [
        UndecidedRow(
            arm=_arm(arm),
            unknown=gate_reads(report_input, arm, "unknown"),
            borders=gate_reads(report_input, arm, "borderline"),
            warnings=_warnings(report_input, arm),
            implied=_implied(report_input, arm),
        )
        for arm in undecided
        if arm in window
    ]
    notes = [_BARS_AS_GATES_NOTE] if bars_as == "gates" else []
    if undecided:
        notes.append(_UNDECIDED_NOTE.format(arms=_named_arms(undecided)))
    dominant = [] if alone else dominant_options(report_input)
    dominant_with_axes = [arm for arm in survivors if all(with_axes(arm, other) for other in survivors if other != arm)]
    declared = _code_reading(report_input, decision_at(report_input))
    rule = list(dict.fromkeys(axis for family in report_input.contrast_families for axis in family.merit_axes))
    return Dominance(
        survivors=rows,
        out=out,
        undecided=undecided_rows,
        bars_as=bars_as,
        weighs=list(weighed_axes(report_input)),
        dominant=[_arm(arm) for arm in dominant],
        axes=[
            row.model_copy(update={"arms": [held for held in row.arms if held.arm.removeprefix("arm:") in window]})
            for row in axis_rows
        ],
        dominant_with_axes=[_arm(arm) for arm in dominant_with_axes] if axis_rows else [_arm(arm) for arm in dominant],
        code_call=CodeCall(
            reads_as=CODE_CALL_READS_AS,
            modes=sorted(declared.modes),
            dominant=[_arm(arm) for arm in declared.dominant],
            ranking_rule=rule,
            withheld=_COST_ALONE if alone else "",
        ),
        total=len(order),
        offset=offset,
        next_offset=end if end < len(order) else None,
        notes=notes,
    )


# =============================================================================
# Sensitivity
# =============================================================================


class Pick(BaseModel):
    """One routable task's pick: the cheapest arm passing in it and in every cross-cutting task."""

    model_config = _MODEL

    task: str = Field(description="The routable task, by id.")
    arm: str | None = Field(description="Its pick, as arm:<id>; None where no arm with a cost there passes.")


class Flip(BaseModel):
    """One value an arm's status flips at as the criterion moves."""

    model_config = _MODEL

    arm: str = Field(description="The arm whose status flips, as arm:<id>.")
    where: str | None = Field(
        description="Which flip it is: a floor's held_above or broken_at, a criterion's task or cell; None otherwise."
    )
    at: CitedFigure = Field(description="The value it flips at, as the key that states it.")
    call_changes: bool = Field(
        description="Whether code's call differs on the two sides of this value, read past the range swept too: a "
        "flip on the range's own end says whether the call changes just past it."
    )
    picks_change: bool = Field(
        description="Whether a routable task's pick differs on the two sides of this value, read past the range swept "
        "too."
    )


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
    picks: list[Pick] = Field(
        description="Each routable task's pick throughout it, where the snapshot holds routable tasks; else empty."
    )


#: Which end of the caller's range a flip lies past.
UnsweptSide = Literal["below_range_low", "above_range_high"]


class Unswept(BaseModel):
    """One side of the range the caller gave, past which the call or a routable task's pick changes."""

    model_config = _MODEL

    side: UnsweptSide = Field(description="Which end of the range swept the change lies past.")
    loosens: bool = Field(
        description="Whether this side loosens the line: a lower threshold where higher is better (a higher one "
        "where lower is), or a wider margin or floor; a line loosened admits options the range swept never reads."
    )
    nearest: Flip = Field(description="The nearest flip past that end where the call or a pick changes.")
    past: list[CitedFigure] = Field(
        description="Each flip past that end where the call or a pick changes, nearest first, as the key that states "
        "it (the first is nearest's); at most a page's worth (limit), more_past saying whether more lie beyond."
    )
    more_past: bool = Field(description="Whether flips past the ones listed in past also change the call or a pick.")
    call_past: Call = Field(description="Code's call just past the nearest flip, away from the range.")


class Barred(BaseModel):
    """One arm a cross-cutting task's gate keeps out of every routable task's pick, at the criteria as declared."""

    model_config = _MODEL

    arm: str = Field(description="The arm barred, as arm:<id>.")
    task: str = Field(description="The cross-cutting task it fails a gate in, by id: it runs inside every task.")
    gates: list[GateRead] = Field(description="Each gate or guardrail it fails there, by the keys that state it.")
    passes_tasks: list[str] = Field(
        description="The routable tasks whose own gates it passes, which it is held back from all the same."
    )


class Sensitivity(BaseModel):
    """How code's call moves as one declared criterion moves over a range."""

    model_config = _MODEL

    parameter: str = Field(description="The criterion swept, as asked.")
    hypothetical: bool = Field(
        description="True where the line swept is one the snapshot does not declare: every call is code's reading as "
        "if a gate held there."
    )
    swept: str = Field(description="The range swept, in words.")
    declared: CitedFigure | None = Field(description="The criterion's declared value; None where none is declared.")
    boundary_belongs_to: Boundary = Field(
        description="Which piece a value exactly at a flip falls in: the piece below it, or the piece above."
    )
    declared_call: Call = Field(description="Code's call at the criteria as declared.")
    reads_as: str = Field(description="What every call here is: code's reading, never a constraint.")
    declared_picks: list[Pick] = Field(
        description="Each routable task's pick at the criteria as declared; empty where the snapshot holds none."
    )
    segments: list[Segment] = Field(
        description="The range swept, cut where code's call or a routable task's pick changes, low to high: on each "
        "page, the segments starting at a flip the page lists, and on the first the one starting at the range's low "
        "end."
    )
    holds: bool = Field(description="Whether every segment's call is the declared call.")
    picks_hold: bool = Field(
        description="Whether every segment's picks are the declared picks: a call that holds can still move a routed "
        "plan."
    )
    flips: list[Flip] = Field(description="The values an arm's status flips at inside the range, low to high.")
    total: int = Field(description="The flips inside the range.")
    offset: int = Field(description="The first flip this page holds.")
    next_offset: int | None = Field(description="The offset of the next page of flips; None when this is the last.")
    notes: list[str] = Field(description="What the sweep could not read, and why.")
    bars_as: BarsAs = Field(
        description="How the declared bars were read, as the caller chose: warnings (they never move the call) or "
        "gates."
    )
    unswept: list[Unswept] = Field(
        default_factory=list,
        description="Each side of the range given past which the call or a routable task's pick changes; empty "
        "where the range walks every such flip (always, with no range given).",
    )
    barred: list[Barred] = Field(
        default_factory=list,
        description="Each arm a cross-cutting task's gate keeps out of every routable task's pick at the criteria as "
        "declared, which the per-task picks hold back; empty where the snapshot holds no routable and cross-cutting "
        "tasks, or none is barred.",
    )


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
    input_at: Callable[[float], ReportInput]
    call_at: Callable[[float], DecisionAt]
    low: float | None
    high: float | None
    notes: list[str]
    hypothetical: bool = False

    def read_at(self, value: float) -> _Read:
        """Code's call and the routable tasks' picks with the criterion at ``value``."""
        return _Read(self.call_at(value), _picks(self.input_at(value)))


#: Each routable task's pick, as (task, arm or None), in the tasks' order.
_Picks = tuple[tuple[str, str | None], ...]


@dataclass(frozen=True)
class _Read:
    """What a sweep reads at one value: code's call, and each routable task's pick."""

    call: DecisionAt
    picks: _Picks


def _picks(report_input: ReportInput) -> _Picks:
    """Each routable task's pick (:func:`~threetears.evals.analysis.report_gates.routing_picks`) over every arm, in
    the tasks' order; empty where the input holds no routable task."""
    routable = [task.id for task in report_input.tasks if task.routable]
    if not routable:
        return ()
    picks = routing_picks(report_input, [arm.id for arm in report_input.arms])
    return tuple((task, picks.get(task)) for task in routable)


def _pick_rows(picks: _Picks) -> list[Pick]:
    return [Pick(task=task, arm=None if arm is None else _arm(arm)) for task, arm in picks]


@dataclass(frozen=True)
class _Piece:
    """A piece of the range between two adjacent flips (None: the range's end), and what is read throughout it."""

    low: _Cut | None
    high: _Cut | None
    read: _Read

    @property
    def call(self) -> DecisionAt:
        return self.read.call


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
            pieces.append(_Piece(first[value], first[value], sweep.read_at(value)))
        elif not at_low_end:
            pieces.append(_Piece(start_cut, first[value], sweep.read_at(_inside(start, value))))
        start, start_cut = value, first[value]
    at_high_end = start_cut is not None and sweep.high is not None and start == sweep.high
    if sweep.boundary == "above" and start_cut is not None and at_high_end:
        pieces.append(_Piece(start_cut, start_cut, sweep.read_at(start_cut.value)))
    elif not at_high_end:
        pieces.append(_Piece(start_cut, None, sweep.read_at(_inside(start, sweep.high))))
    return pieces


def _segments(pieces: Sequence[_Piece]) -> list[tuple[_Cut | None, Segment]]:
    """Adjacent pieces with the same call and picks, joined; each with the flip it starts at (None: the range's low
    end)."""
    segments: list[tuple[_Cut | None, Segment]] = []
    run: list[_Piece] = []
    for piece in [*pieces, None]:
        if run and (piece is None or piece.read != run[0].read):
            segments.append(
                (
                    run[0].low,
                    Segment(
                        low=SegmentEnd(at=run[0].low.at) if run[0].low is not None else None,
                        high=SegmentEnd(at=run[-1].high.at) if run[-1].high is not None else None,
                        modes=sorted(run[0].call.modes),
                        dominant=[_arm(arm) for arm in run[0].call.dominant],
                        picks=_pick_rows(run[0].read.picks),
                    ),
                )
            )
            run = []
        if piece is not None:
            run.append(piece)
    return segments


def _changes(pieces: Sequence[_Piece], value: float, read: Callable[[_Piece], object]) -> bool:
    """Whether what ``read`` reads differs on the two sides of a flip: the piece ending at it, and the one after."""
    for index, piece in enumerate(pieces[:-1]):
        if piece.high is not None and piece.high.value == value:
            return read(piece) != read(pieces[index + 1])
    return False


def _read_call(report_input: ReportInput) -> DecisionAt:
    """Code's call on an input, as these analyses read it (:func:`_code_reading`)."""
    return _code_reading(report_input, decision_at(report_input))


def _directed_keys(report_input: ReportInput) -> list[str]:
    """The measures and judged dimensions a hypothetical line can be swept on: those stating which way is better."""
    measures = [key for key, measure in report_input.measures.items() if measure.higher_is_better is not None]
    return list(dict.fromkeys([*report_input.judged, *measures]))


def _parameters(report_input: ReportInput) -> list[str]:
    """Every parameter this input can be swept on: its declared criteria, then a hypothetical line on each measure."""
    floors = dict.fromkeys(f"floor:{read.aspect}/{read.floor}" for read in report_input.floors)
    return [
        *(f"margin:{aspect.id}" for aspect in report_input.aspects),
        *floors,
        *(f"threshold:{criterion.id}" for criterion in report_input.criteria),
        *(f"{_HYPOTHETICAL}:{key}" for key in _directed_keys(report_input)),
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
        input_at=lambda _margin: report_input,
        call_at=lambda margin: _code_reading(report_input, decision_at(report_input, margins={aspect.id: margin})),
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
        input_at=lambda _margin: report_input,
        call_at=lambda margin: _code_reading(report_input, decision_at(report_input, floors={floor: margin})),
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
    notes: list[str] = []
    if criterion.kind == "bar":
        notes.append(
            "a bar read as a warning never gates, so moving its threshold moves no option: each flip is where an arm's "
            "figure crosses the line, its status on the bar changing there; read the bars as gates (bars_as=gates) "
            "for the call as if the bar gated"
        )
    elif criterion.kind not in _GATING:
        notes.append(f"a {criterion.kind} criterion warns and never gates, so moving its threshold moves no option")
    unknown = [arm.id for arm in report_input.arms if criterion_unknowns(report_input, criterion, arm.id)]
    if unknown:
        notes.append(
            f"{_named_arms(unknown)}: no figure at some place this criterion reads (unknown "
            "there): no threshold passes or fails them there, so no flip is listed for those places"
        )
    return _Sweep(
        swept=words,
        declared=cite(f"criteria.{criterion.id}.threshold", report_input),
        boundary="below" if criterion.direction == "higher_is_better" else "above",
        cuts=cuts,
        input_at=lambda threshold: _moved(report_input, criterion.id, threshold),
        call_at=lambda threshold: _read_call(_moved(report_input, criterion.id, threshold)),
        low=low,
        high=high,
        notes=notes,
    )


#: The parameter kind that sweeps a line the snapshot does not declare.
_HYPOTHETICAL = "hypothetical"

#: The statistics a hypothetical line may read.
_LINE_STATISTICS = ("mean", "p50", "p95")


def _line_scope(report_input: ReportInput, key: str, statistic: str) -> Literal["overall", "each_task", "each_cell"]:
    """Where a hypothetical line reads: overall where an arm states the statistic overall, else in each task, else in
    each cell, as the snapshot holds it.

    :raises RefusedInput: No arm states the statistic on the measure anywhere.
    """
    table = report_input.measures.get(key) or report_input.judged.get(key)
    if table is not None and any(getattr(stat, statistic) is not None for stat in table.per_arm.values()):
        return "overall"
    places = (
        ("each_task", [task.per_arm for task in report_input.tasks]),
        ("each_cell", [cell.per_arm for cell in report_input.cells.values()]),
    )
    for scope, held in places:
        if any(
            getattr(figures[key], statistic) is not None
            for per_arm in held
            for figures in per_arm.values()
            if key in figures
        ):
            return "each_task" if scope == "each_task" else "each_cell"
    raise RefusedInput(
        f"no arm states the {statistic} of {key!r} overall, in a task or in a cell, so no line on it reads anything",
        remedy=f"name the statistic the measure states, as hypothetical:{key}/<statistic> with one of "
        f"{', '.join(_LINE_STATISTICS)}; measure_read lists what it holds",
        parameters=("parameter",),
    )


def _with_line(report_input: ReportInput, line: Criterion, threshold: float) -> ReportInput:
    """The input with a hypothetical gate added at ``threshold``, and nothing else changed."""
    return report_input.model_copy(
        update={"criteria": [*report_input.criteria, line.model_copy(update={"threshold": threshold})]}
    )


def _figure_key(report_input: ReportInput, key: str, arm: str, statistic: str, figure: CriterionFigure) -> str:
    """The key of an arm's own figure a hypothetical line reads, at its place."""
    place = _place(report_input, figure)
    if place is None:
        if key in report_input.measures:
            return f"measure.{key}.{arm}.{statistic}"
        return f"judged.{key}.{arm}.{statistic}"
    return f"{'task' if figure.task is not None else 'cells'}.{place}.{arm}.{key}.{statistic}"


def _line_sweep(report_input: ReportInput, spec: str, low: float | None, high: float | None, words: str) -> _Sweep:
    """A sweep of a gate the snapshot does not declare, on one measure, at the statistic named (the mean unless
    ``<measure>/<statistic>``), in the measure's direction."""
    key, statistic = spec, "mean"
    head, slash, tail = spec.rpartition("/")
    if slash and tail in _LINE_STATISTICS and head:
        key, statistic = head, tail
    directed = _directed_keys(report_input)
    if key not in directed:
        known = key in report_input.measures
        fault = "states no direction (higher or lower is better)" if known else "names no measure the snapshot holds"
        raise RefusedInput(
            f"parameter {_HYPOTHETICAL}:{spec!r} {fault}",
            remedy=f"sweep a hypothetical line on a measure with a direction: {_listed(directed)}",
            parameters=("parameter",),
            valid_values=[f"{_HYPOTHETICAL}:{held}" for held in directed],
        )
    higher = True if key in report_input.judged else bool(report_input.measures[key].higher_is_better)
    taken = {criterion.id for criterion in report_input.criteria}
    line_id = next(
        f"{_HYPOTHETICAL}{index or ''}"
        for index in range(len(taken) + 1)
        if f"{_HYPOTHETICAL}{index or ''}" not in taken
    )
    line = Criterion.model_validate(
        {
            "id": line_id,
            "kind": "gate",
            "measure": key,
            "statistic": statistic,
            "direction": "higher_is_better" if higher else "lower_is_better",
            "threshold": 0.0,
            "scope": _line_scope(report_input, key, statistic),
            "label": f"hypothetical line on {key}",
        }
    )
    cuts = [
        _Cut(
            arm.id,
            _place(report_input, figure),
            cite(_figure_key(report_input, key, arm.id, statistic, figure), report_input),
        )
        for arm in report_input.arms
        for figure in criterion_figures(report_input, line, arm.id)
    ]
    return _Sweep(
        swept=words,
        declared=None,
        boundary="below" if higher else "above",
        cuts=cuts,
        input_at=lambda threshold: _with_line(report_input, line, threshold),
        call_at=lambda threshold: _read_call(_with_line(report_input, line, threshold)),
        low=low,
        high=high,
        notes=[
            f"a hypothetical line on the {statistic} of {key!r}, read {line.scope.replace('_', ' ')}, which the "
            f"snapshot does not declare{_declared_on(report_input, key)}; every call is code's reading as if a gate "
            "held at the line, the declared criteria unchanged"
        ],
        hypothetical=True,
    )


def _declared_on(report_input: ReportInput, key: str) -> str:
    """The criteria the snapshot declares on a measure, as a hypothetical sweep's note names them beside its line."""
    declared = [
        f"{criterion.id} ({criterion.kind}, {criterion.statistic}, {criterion.scope.replace('_', ' ')})"
        for criterion in report_input.criteria
        if criterion.measure == key
    ]
    return f" (it declares {', '.join(declared)} on it)" if declared else ""


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
    if kind == _HYPOTHETICAL:
        words = _range_words("hypothetical line", range_low, range_high, range_low is not None, range_high is not None)
        return _line_sweep(
            report_input, name, range_low, range_high, f"{words}, on {name}, which the snapshot does not declare"
        )
    raise _refuse_parameter(
        report_input,
        parameter,
        "is none of margin:<aspect>, floor:<aspect>/<floor>, threshold:<criterion> or hypothetical:<measure>",
    )


#: What a sweep says where the caller's range stops short of a change, once per side (``unswept`` holds the keys).
_UNSWEPT_NOTE = (
    "the call or a routable task's pick changes past the range swept ({sides}): unswept names the nearest such flip "
    "and the call past it; omit range_low and range_high (or widen the range) to walk every flip"
)

#: What a margin sweep says of candidates the aspect holds no reading of.
_NO_READING_NOTE = (
    "aspect {aspect} holds no reading of {arms}: code's rule reads a margin only against a reading, so no margin moves "
    "the call on them, and it never reads a contrast against a margin; a contrast's interval is within any margin "
    "wider than minus its low end, past any at or below minus its high end, and open between (contrasts_list holds "
    "the intervals)"
)

#: What a sweep says where per-task picks hold arms back.
_BARRED_NOTE = (
    "the per-task picks hold back {arms}: each fails a gate in a cross-cutting task, which runs inside every task, so "
    "it takes no task, even one whose own gates it passes (barred names the gates by their keys)"
)


def _loosens(parameter: str, sweep: _Sweep, side: UnsweptSide) -> bool:
    """Whether moving the line past ``side`` loosens it: a margin or a floor wider; a threshold or a hypothetical line
    toward worse figures (lower where higher is better, higher where lower is)."""
    if parameter.partition(":")[0] in ("margin", "floor"):
        return side == "above_range_high"
    return side == ("below_range_low" if sweep.boundary == "below" else "above_range_high")


def _piece_past(pieces: Sequence[_Piece], value: float, side: UnsweptSide) -> _Piece:
    """The piece just past a flip, away from the range: the one ending at it below the range, starting at it above."""
    for piece in pieces:
        end = piece.high if side == "below_range_low" else piece.low
        if end is not None and end.value == value:
            return piece
    raise ValueError(f"no piece of the sweep meets the flip at {value}")


def _past_end(value: float, sweep: _Sweep, side: UnsweptSide) -> bool:
    """Whether a flip lies past the caller's range on ``side``: beyond its end, or exactly on it where the end's value
    falls in the piece inside the range (``sweep.boundary``), so that the change at the flip lies just past the end,
    where the range never reads."""
    if side == "below_range_low":
        return sweep.low is not None and (value < sweep.low or (value == sweep.low and sweep.boundary == "above"))
    return sweep.high is not None and (value > sweep.high or (value == sweep.high and sweep.boundary == "below"))


@dataclass(frozen=True)
class _Whole:
    """A sweep over its default range (every flip): its flips low to high, and its pieces."""

    cuts: list[_Cut]
    pieces: list[_Piece]


def _whole(report_input: ReportInput, parameter: str) -> _Whole:
    """The sweep of ``parameter`` over its default range, which every flip's reading of the call on its two sides and
    every unswept side are read on, whatever range the caller gave."""
    whole = _sweep_of(report_input, parameter, None, None)
    cuts = sorted(
        (
            cut
            for cut in whole.cuts
            if (whole.low is None or cut.value >= whole.low) and (whole.high is None or cut.value <= whole.high)
        ),
        key=lambda cut: cut.value,
    )
    return _Whole(cuts, _pieces(whole, cuts))


def _flip(cut: _Cut, pieces: Sequence[_Piece]) -> Flip:
    """One flip, the call and the picks on its two sides read on ``pieces``."""
    return Flip(
        arm=_arm(cut.arm),
        where=cut.where,
        at=cut.at,
        call_changes=_changes(pieces, cut.value, lambda piece: piece.call),
        picks_change=_changes(pieces, cut.value, lambda piece: piece.read.picks),
    )


def _unswept(parameter: str, sweep: _Sweep, whole: _Whole, limit: int) -> list[Unswept]:
    """Each side of the caller's range past which the call or a routable task's pick changes, read on the sweep over
    its default range (every flip): a flip beyond the range's end, or on it where the change lies just past
    (:func:`_past_end`). Read only where the caller gives a range: with none, the sweep walks every flip."""

    def moves(cut: _Cut) -> bool:
        return _changes(whole.pieces, cut.value, lambda piece: piece.call) or _changes(
            whole.pieces, cut.value, lambda piece: piece.read.picks
        )

    found: list[Unswept] = []
    sides: tuple[UnsweptSide, ...] = ("below_range_low", "above_range_high")
    for side in sides:
        past = [cut for cut in whole.cuts if _past_end(cut.value, sweep, side) and moves(cut)]
        if side == "below_range_low":
            past.reverse()
        if not past:
            continue
        nearest = past[0]
        beyond = _piece_past(whole.pieces, nearest.value, side)
        found.append(
            Unswept(
                side=side,
                loosens=_loosens(parameter, sweep, side),
                nearest=_flip(nearest, whole.pieces),
                past=[cut.at for cut in past[:limit]],
                more_past=len(past) > limit,
                call_past=_call(beyond.call),
            )
        )
    return found


def _unread_arms(report_input: ReportInput, parameter: str) -> list[str]:
    """On a margin sweep, the candidates the aspect holds no reading of, where the snapshot has a control; else none."""
    kind, _colon, name = parameter.partition(":")
    control = control_arm(report_input)
    aspect = next((aspect for aspect in report_input.aspects if aspect.id == name), None)
    if kind != "margin" or control is None or aspect is None:
        return []
    read = {reading.arm for reading in aspect.readings}
    return [arm.id for arm in report_input.arms if arm.id != control and arm.id not in read]


def _barred(report_input: ReportInput) -> list[Barred]:
    """Each arm a cross-cutting task's gate keeps out of every routable task's pick, at the criteria as declared, with
    the gates it fails there and the routable tasks whose own gates it passes."""
    routable = [task.id for task in report_input.tasks if task.routable]
    cross = [task.id for task in report_input.tasks if task.cross_cutting]
    if not routable or not cross:
        return []
    barred: list[Barred] = []
    for arm in report_input.arms:
        for task in cross:
            if gate_check(report_input, arm.id, task=task).status != "fail":
                continue
            reads = [
                _gate_read(report_input, arm.id, finding)
                for finding in gate_findings(report_input, arm.id, task=task)
                if finding.status == "fail"
            ]
            passes = [held for held in routable if gate_check(report_input, arm.id, task=held).status != "fail"]
            barred.append(Barred(arm=_arm(arm.id), task=task, gates=reads, passes_tasks=passes))
            break
    return barred


def sensitivity(
    report_input: ReportInput,
    parameter: str,
    *,
    range_low: float | None = None,
    range_high: float | None = None,
    limit: int = SENSITIVITY_PAGE,
    offset: int = 0,
    bars_as: BarsAs = "warnings",
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
    :param bars_as: How to read the declared bars: ``warnings`` as declared, or ``gates`` (this module's How bars are
        read); a bar's threshold sweeps either way.
    :ptype bars_as: BarsAs
    :return: The segments, the flips on this page, and whether the call holds across the range.
    :rtype: Sensitivity
    :raises RefusedInput: The parameter names nothing this input holds, or the range is not one.
    """
    report_input = _read_bars_as(report_input, bars_as)
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
    declared = _read_call(report_input)
    declared_picks = _picks(report_input)
    page = inside[offset : offset + limit]
    end = offset + len(page)
    at = {id(cut): index for index, cut in enumerate(inside)}
    segments = [
        segment for low, segment in _segments(pieces) if (offset == 0 if low is None else offset <= at[id(low)] < end)
    ]
    notes = [*([_BARS_AS_GATES_NOTE] if bars_as == "gates" else []), *sweep.notes]
    if len(inside) < len(sweep.cuts):
        notes.append("flips outside the range swept are not listed")
    ranged = range_low is not None or range_high is not None
    whole = _whole(report_input, parameter) if ranged else _Whole(inside, pieces)
    unswept = _unswept(parameter, sweep, whole, limit) if ranged else []
    if unswept:
        notes.append(_UNSWEPT_NOTE.format(sides=" and ".join(side.side.replace("_", " ") for side in unswept)))
    unread = _unread_arms(report_input, parameter)
    if unread:
        notes.append(_NO_READING_NOTE.format(aspect=parameter.partition(":")[2], arms=_named_arms(unread)))
    barred = _barred(report_input)
    if barred:
        notes.append(_BARRED_NOTE.format(arms=_named_arms([row.arm.removeprefix("arm:") for row in barred])))
    undecided = undecided_arms(report_input)
    if undecided:
        notes.append(_SWEEP_UNDECIDED_NOTE.format(arms=_named_arms(undecided)))
    return Sensitivity(
        parameter=parameter,
        hypothetical=sweep.hypothetical,
        swept=sweep.swept,
        declared=sweep.declared,
        boundary_belongs_to=sweep.boundary,
        declared_call=_call(declared),
        reads_as=CODE_CALL_READS_AS,
        declared_picks=_pick_rows(declared_picks),
        segments=segments,
        holds=all(piece.call == declared for piece in pieces),
        picks_hold=all(piece.read.picks == declared_picks for piece in pieces),
        flips=[_flip(cut, whole.pieces) for cut in page],
        total=len(inside),
        offset=offset,
        next_offset=end if end < len(inside) else None,
        notes=notes,
        bars_as=bars_as,
        unswept=unswept,
        barred=barred,
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
