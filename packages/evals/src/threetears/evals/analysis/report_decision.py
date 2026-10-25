"""What code decides before the planner writes: the caveats a report input carries, which of them are immaterial, the
options that survive, which one dominates, and so the modes a plan may take.

The planner chooses the mode only from :func:`allowed_modes`, states its confidence against the caveats
:func:`report_caveats` lists, and never mentions a caveat :func:`caveat_materiality` proved immaterial; code renders
that one's checked line. :mod:`threetears.evals.analysis.report_plan_checks` refuses a plan that departs from any of
it.

**Caveat ids.** Every entry of the report input's ``caveats`` gets one stable id, the same vocabulary the plan's flag,
confidence clauses and landing lists use:

========================================  ==========================================================================
``late:<criterion>``                      a criterion set after the results, unnamed (``late:aspect``,
                                          ``late:control``, ``late:margin``)
``late:<criterion>:<name>``               a named one (``late:floor:<floor key>``, ``late:bar:<measure>``,
                                          ``late:guardrail:<dimension>``)
``judge_uncalibrated``                    every judged dimension's tier and calibration are undetermined or
                                          insufficient
``confound:<id>``                         a confound, by its key in ``caveats.confounds``
``apparatus:<dimension>``                 an apparatus confound; a confound of the same dimension is this caveat
``short_run:<run id>``                    a run short of usable measurements
``latency``                               latency is not compared
``unreported:<observation>``              an observation recorded but not reported
========================================  ==========================================================================

**Scope**, which decides where a caveat may land in the plan. *Report-wide*: every late criterion, the uncalibrated
judge, and a confound, apparatus confound or short run that touches every comparison the report makes (each
candidate against the control in a change, every pair in a choice, the one arm in a measurement; a short run touches
the comparisons its arm is in, every one when it names no arm; a confound touches the comparisons whose contrasts name
it). Report-wide caveats are what the decision's confidence is conditional on. *Scoped*: a confound or short run
touching only some comparisons, said beside the claim it qualifies. *Method-level*: latency, an unreported
observation, and an apparatus confound that is not report-wide.

**Immateriality** (:func:`caveat_materiality`). A late criterion is immaterial when the decision, the allowed modes
and the one option that dominates, is the same across the criterion's whole plausible range. Code checks each piece
of the range between the points where some reading's status flips, plus each flip point and a value above the last, so
the check covers the range exactly:

- *A late margin*: every margin from zero up, on each aspect that declares one. A reading flips at its
  :func:`~threetears.evals.analysis.report.figure_keys.flip_margin` (``-lo``): within any margin above it, outside
  any at or below it. Immaterial when the call holds at every margin.
- *A late floor*: every margin the floor could take, from zero up. A floor read on an interval ``[lo, hi]`` of the
  change is held at a margin ``m`` above ``-lo``, broken at or below ``-hi``, unclear between.
- *A late bar*: a bar only warns and decides nothing, so the call holds at any bar.
- *A late aspect, control or guardrail margin*: open. No range of them can be checked from the data: another aspect or
  control is another campaign, and a guardrail's reading carries no interval here.

Every other caveat is open: an uncalibrated judge, a confound, a short run, latency and an unreported observation are
not a criterion with a range.

**Survivors** (:func:`surviving_options`, GENERATOR §3). An option survives when it passes every gate: each criterion
of kind ``gate`` or ``guardrail``, read on its point estimate (at least the threshold where higher is better, at most
where lower is), overall, in its one task, in every task or in every cell as its scope says; and no guardrail the
pipeline read on it is ``breached``. A gate with no figure for the arm is not shown passed, so the arm does not
survive it. Borderline counts as passing for the decision (GENERATOR §3). Floors and bars warn and never remove an
option; a margin is the asked aspect's own axis, not a gate.

**Dominance** (:func:`dominant_options`). One surviving option dominates when it is at least as good as every other
survivor on every axis that matters:

- *Every gate*: every survivor passes every gate, so the gates tie among survivors.
- *Cost*: cost per 1,000 replies (the figure key's source; else the one dollar measure where lower is better; else
  that measure blended over the tasks by traffic share), lower is better. Two arms with no cost tie; an arm with no
  cost is not at least as good as one with a cost, nor the reverse.
- *The asked aspect*: in a change, each arm takes a level: a candidate shown better (a confirmatory ``better``
  reading, or an ``improved`` contrast on an asked aspect) is 2; one shown no worse (every floor on every asked
  aspect held, each declaring one, and within the margin on each asked reading that declares one) is 1; any other
  candidate is 0; the control is 1. An arm is at least as good as another at the same level or above. In a choice, an
  arm is at least as good as another unless a pair reading on an asked aspect shows the other ahead of it. With no
  asked reading, pair or floor, the aspect ties.

**Modes** (:func:`allowed_modes`, GENERATOR §1.1), read in this order:

1. *No call* when the report measures one arm (there is nothing to choose), when no option survives, or, in a change,
   when every surviving candidate's asked reading is not separable: its interval crosses zero and it is outside the
   margin (or no margin is declared).
2. *Decide* when exactly one survivor dominates; the plan recommends that one.
3. *Decide* when two or more survive, none dominates alone, and the campaign declared a rule that ranks the axes (a
   contrast family's merit priority): the rule picks, among the survivors.
4. *Trade off* otherwise: two or more survive, none dominates alone, and no rule ranks them.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal

from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.figure_keys import flip_margin, resolve
from threetears.evals.analysis.report_input import (
    Aspect,
    Criterion,
    FloorRead,
    LateCriterion,
    Reading,
    ReportInput,
    Stat,
)

__all__ = [
    "Caveat",
    "Materiality",
    "allowed_modes",
    "caveat_materiality",
    "control_arm",
    "dominant_options",
    "floors_not_held",
    "report_caveats",
    "report_kind",
    "shown_better",
    "surviving_options",
]

#: Judge tiers and calibration states that mean no human calibration stands behind the judge.
_UNCALIBRATED = frozenset({"undetermined", "insufficient"})

#: Where a caveat may land: report-wide, scoped to some comparisons, or method-level.
CaveatScope = Literal["report", "scoped", "method"]
#: What a report is, from its arms.
ReportKind = Literal["measurement", "change", "choice"]

#: The checked line code renders for a late criterion it proved immaterial, by the criterion's kind.
_CHECKED = {
    "margin": "The late margin was checked: the call holds at every margin.",
    "floor": "The late floor was checked: the call holds at every margin the floor could take.",
    "bar": "The late bar was checked: a bar only warns, so the call holds at any bar.",
}


@dataclass(frozen=True)
class Caveat:
    """One entry of the report input's ``caveats``, by the id the plan names it by, and where it may land.

    :ivar id: The caveat's id (this module's id table).
    :ivar kind: The id's kind: ``late``, ``judge_uncalibrated``, ``confound``, ``apparatus``, ``short_run``,
        ``latency`` or ``unreported``.
    :ivar scope: ``report`` (the confidence is conditional on it), ``scoped`` (said beside a claim) or ``method``.
    """

    id: str
    kind: str
    scope: CaveatScope


@dataclass(frozen=True)
class Materiality:
    """Whether code proved a caveat immaterial.

    :ivar id: The caveat's id.
    :ivar immaterial: True when the decision holds across the caveat's whole plausible range.
    :ivar checked: The one line code renders for an immaterial caveat; None for an open one.
    """

    id: str
    immaterial: bool
    checked: str | None


@dataclass(frozen=True)
class _Overrides:
    """Hypothetical values of late criteria, for the sensitivity check: a margin per aspect, a margin per floor key."""

    margins: Mapping[str, float] = field(default_factory=dict)
    floors: Mapping[str, float] = field(default_factory=dict)


#: The decision a report input supports: the allowed modes, and the one option that dominates (empty for none).
_Outcome = tuple[frozenset[str], tuple[str, ...]]


# --- the arms ------------------------------------------------------------------------------------------------------


def control_arm(report_input: ReportInput) -> str | None:
    """The control's id, or None in a report with no control.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The control's id; None when no arm is the control.
    :rtype: str | None
    """
    return next((arm.id for arm in report_input.arms if arm.role == "control"), None)


def report_kind(report_input: ReportInput) -> ReportKind:
    """What the report is, from its arms: one arm measured, candidates against a control, or a choice among arms.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: ``measurement``, ``change`` or ``choice``.
    :rtype: ReportKind
    """
    if len({arm.id for arm in report_input.arms}) <= 1:
        return "measurement"
    return "change" if control_arm(report_input) is not None else "choice"


def _comparisons(report_input: ReportInput) -> list[frozenset[str]]:
    """The comparisons the report makes, each as the arms in it: candidates against the control, pairs, or one arm."""
    arms = [arm.id for arm in report_input.arms]
    control = control_arm(report_input)
    if len(set(arms)) <= 1:
        return [frozenset(arms)] if arms else []
    if control is not None:
        return [frozenset({control, arm}) for arm in arms if arm != control]
    return [frozenset(pair) for pair in combinations(arms, 2)]


# --- the caveats ---------------------------------------------------------------------------------------------------


def _late_id(late: LateCriterion) -> str:
    return f"late:{late.criterion}" if late.name is None else f"late:{late.criterion}:{late.name}"


def _judge_uncalibrated(report_input: ReportInput) -> bool:
    tiers = report_input.caveats.judge_tiers
    dimensions = [*report_input.judged, *(name for name in tiers if name not in report_input.judged)]
    if not dimensions:
        return False
    for dimension in dimensions:
        judged = report_input.judged.get(dimension)
        stated = {tiers.get(dimension)}
        if judged is not None:
            stated |= {judged.tier, judged.calibration}
        stated.discard(None)
        if not stated or not stated <= _UNCALIBRATED:
            return False
    return True


def _touches_every_comparison(report_input: ReportInput, arms: set[str] | None) -> bool:
    """Whether a caveat on ``arms`` (None: on every arm) touches every comparison the report makes."""
    comparisons = _comparisons(report_input)
    if arms is None:
        return True
    return bool(comparisons) and all(comparison & arms for comparison in comparisons)


def _confound_arms(report_input: ReportInput, dimension: str) -> set[str]:
    return {contrast.arm for contrast in report_input.contrasts if dimension in contrast.confounds}


def report_caveats(report_input: ReportInput) -> list[Caveat]:
    """Every caveat the report input carries, each once, by id, with where it may land (this module's id table).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: Late criteria, the judge, confounds, apparatus confounds, short runs, latency, then unreported
        observations, each in the input's order.
    :rtype: list[Caveat]
    """
    caveats = report_input.caveats
    found: dict[str, Caveat] = {}

    def add(caveat: Caveat) -> None:
        found.setdefault(caveat.id, caveat)

    for late in caveats.late_criteria:
        add(Caveat(_late_id(late), "late", "report"))
    if _judge_uncalibrated(report_input):
        add(Caveat("judge_uncalibrated", "judge_uncalibrated", "report"))
    apparatus = [confound.dimension for confound in caveats.apparatus_confounds]
    for dimension in caveats.confounds:
        if dimension not in apparatus:
            wide = _touches_every_comparison(report_input, _confound_arms(report_input, dimension))
            add(Caveat(f"confound:{dimension}", "confound", "report" if wide else "scoped"))
    for dimension in apparatus:
        wide = _touches_every_comparison(report_input, _confound_arms(report_input, dimension))
        add(Caveat(f"apparatus:{dimension}", "apparatus", "report" if wide else "method"))
    for run in caveats.short_runs:
        wide = _touches_every_comparison(report_input, None if run.arm is None else {run.arm})
        add(Caveat(f"short_run:{run.run}", "short_run", "report" if wide else "scoped"))
    if caveats.latency:
        add(Caveat("latency", "latency", "method"))
    for observation in caveats.unreported:
        add(Caveat(f"unreported:{observation}", "unreported", "method"))
    return list(found.values())


# --- gates and survivors -------------------------------------------------------------------------------------------


def _criterion_figures(report_input: ReportInput, criterion: Criterion, arm: str) -> list[float]:
    """The arm's figures a criterion reads, at its statistic, where its scope reads them."""
    holders: list[Mapping[str, Stat]] = []
    if criterion.scope == "each_cell":
        holders = [cell.per_arm.get(arm, {}) for cell in report_input.cells.values()]
    elif criterion.scope == "each_task" or criterion.task is not None:
        holders = [
            task.per_arm.get(arm, {}) for task in report_input.tasks if criterion.task in (None, task.id)
        ]
    else:
        for table in (report_input.measures, report_input.judged):
            figures = table.get(criterion.measure)
            if figures is not None and arm in figures.per_arm:
                holders = [{criterion.measure: figures.per_arm[arm]}]
                break
    values: list[float] = []
    for holder in holders:
        stat = holder.get(criterion.measure)
        value = getattr(stat, criterion.statistic) if stat is not None else None
        if value is not None:
            values.append(float(value))
    return values


def _passes(report_input: ReportInput, criterion: Criterion, arm: str) -> bool:
    values = _criterion_figures(report_input, criterion, arm)
    if not values:
        return False
    if criterion.direction == "higher_is_better":
        return all(value >= criterion.threshold for value in values)
    return all(value <= criterion.threshold for value in values)


def surviving_options(report_input: ReportInput) -> list[str]:
    """The arms that pass every gate and breach no guardrail (this module's Survivors).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The surviving arms' ids, in the input's arm order.
    :rtype: list[str]
    """
    gates = [criterion for criterion in report_input.criteria if criterion.kind in ("gate", "guardrail")]
    breached = {
        str(guardrail.get("arm"))
        for guardrail in report_input.guardrails
        if guardrail.get("decision") == "breached"
    }
    return [
        arm.id
        for arm in report_input.arms
        if arm.id not in breached and all(_passes(report_input, gate, arm.id) for gate in gates)
    ]


# --- the asked aspect ----------------------------------------------------------------------------------------------


def _asked(report_input: ReportInput) -> list[Aspect]:
    return [aspect for aspect in report_input.aspects if aspect.asked]


def shown_better(report_input: ReportInput, arm: str) -> bool:
    """Whether a candidate is shown better: a confirmatory ``better`` reading, or an ``improved`` contrast on an asked
    aspect.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param arm: The candidate's id.
    :ptype arm: str
    :return: True when the evidence shows it better than the control.
    :rtype: bool
    """
    if any(
        reading.arm == arm and reading.reading == "better" and reading.confirmatory
        for aspect in report_input.aspects
        for reading in aspect.readings
    ):
        return True
    asked = {aspect.id for aspect in _asked(report_input)}
    return any(
        contrast.arm == arm and contrast.verdict == "improved" and contrast.aspect in asked
        for contrast in report_input.contrasts
    )


def _floor_status(read: FloorRead, overrides: _Overrides) -> str:
    """A floor's status: the pipeline's, or at a hypothetical margin, read on its interval."""
    margin = overrides.floors.get(read.floor)
    if margin is None or read.interval is None:
        return read.status
    low, high = read.interval
    if -low < margin:
        return "held"
    return "broken" if -high >= margin else "unclear"


def _floors_not_held(
    report_input: ReportInput, arm: str, asked: Sequence[Aspect], overrides: _Overrides
) -> list[str]:
    clauses: list[str] = []
    for aspect in asked:
        floors = list(dict.fromkeys(floor.floor for floor in report_input.floors if floor.aspect == aspect.id))
        if not floors:
            clauses.append(f"aspect {aspect.id!r} declares no floor, so nothing there shows it no worse")
        for key in floors:
            read = next((f for f in report_input.floors if f.floor == key and f.arm == arm), None)
            if read is None:
                clauses.append(f"floor {key} was not read on it, so it is not shown held")
                continue
            status = _floor_status(read, overrides)
            if status != "held":
                reason = f" ({read.reason.rstrip('.')})" if read.reason and status == read.status else ""
                clauses.append(f"floor {key} is {status}{reason}")
    return clauses


def floors_not_held(report_input: ReportInput, arm: str, asked: Sequence[Aspect]) -> list[str]:
    """Per asked aspect, why a candidate is not shown no worse there; empty when every floor reads held.

    An aspect's floors are those read on it for any candidate; one with none declares no floor.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param arm: The candidate's id.
    :ptype arm: str
    :param asked: The asked aspects.
    :ptype asked: Sequence[Aspect]
    :return: One clause per floor not held, or per aspect declaring none.
    :rtype: list[str]
    """
    return _floors_not_held(report_input, arm, asked, _Overrides())


def _margin(aspect: Aspect, reading: Reading, overrides: _Overrides) -> float | None:
    if aspect.id in overrides.margins:
        return overrides.margins[aspect.id]
    margin = reading.margin if reading.margin is not None else aspect.margin
    return None if margin is None else float(margin)


def _within(aspect: Aspect, reading: Reading, overrides: _Overrides) -> bool:
    margin = _margin(aspect, reading, overrides)
    flip = flip_margin(reading)
    return margin is not None and flip is not None and flip < margin


def _shown_no_worse(report_input: ReportInput, arm: str, overrides: _Overrides) -> bool:
    asked = _asked(report_input)
    if not asked or _floors_not_held(report_input, arm, asked, overrides):
        return False
    for aspect in asked:
        for reading in aspect.readings:
            if reading.arm == arm and _margin(aspect, reading, overrides) is not None:
                if not _within(aspect, reading, overrides):
                    return False
    return True


def _level(report_input: ReportInput, arm: str, control: str, overrides: _Overrides) -> int:
    """A change's arm on the asked aspect: 2 shown better, 1 shown no worse (the control), 0 otherwise."""
    if arm == control:
        return 1
    if shown_better(report_input, arm):
        return 2
    return 1 if _shown_no_worse(report_input, arm, overrides) else 0


def _separable(report_input: ReportInput, arm: str, overrides: _Overrides) -> bool | None:
    """Whether a candidate's asked readings separate it: an interval clear of zero, or within the margin, on any one.
    None where it has no asked reading with an interval."""
    seen = False
    for aspect in _asked(report_input):
        for reading in aspect.readings:
            if reading.arm != arm or reading.within_interval is None:
                continue
            seen = True
            low, high = reading.within_interval
            if low > 0 or high < 0 or _within(aspect, reading, overrides):
                return True
    return False if seen else None


# --- cost ----------------------------------------------------------------------------------------------------------


def _costs(report_input: ReportInput) -> dict[str, float]:
    """Each arm's cost, from the first source that states one (this module's Dominance, cost)."""
    costs: dict[str, float] = {}
    for arm in report_input.arms:
        try:
            figure = resolve(f"cost_per_1k.{arm.id}", report_input)
        except UnresolvableReference:
            continue  # no figure-key cost for this arm: the next source may hold one
        if isinstance(figure.value, (int, float)):
            costs[arm.id] = float(figure.value)
    if costs:
        return costs
    dollars = [
        key
        for key, measure in report_input.measures.items()
        if (measure.unit or "").startswith("usd") and measure.higher_is_better is False
    ]
    if len(dollars) != 1:
        return {}
    key = dollars[0]
    stated = {arm: float(stat.mean) for arm, stat in report_input.measures[key].per_arm.items() if stat.mean is not None}
    if stated:
        return stated
    blended: dict[str, float] = {}
    for arm in report_input.arms:
        parts = [
            (task.share, task.per_arm[arm.id][key].mean)
            for task in report_input.tasks
            if task.share and arm.id in task.per_arm and key in task.per_arm[arm.id]
        ]
        if parts and all(mean is not None for _, mean in parts):
            blended[arm.id] = sum(float(share or 0) * float(mean or 0) for share, mean in parts)
    return blended


# --- dominance and modes -------------------------------------------------------------------------------------------


def _quality_at_least(report_input: ReportInput, first: str, second: str, overrides: _Overrides) -> bool:
    control = control_arm(report_input)
    if control is not None:
        return _level(report_input, first, control, overrides) >= _level(report_input, second, control, overrides)
    return not any(
        set(pair.arms) == {first, second} and pair.ahead == second
        for aspect in _asked(report_input)
        for pair in aspect.pairs
    )


def _cost_at_least(costs: Mapping[str, float], first: str, second: str) -> bool:
    if first not in costs or second not in costs:
        return first not in costs and second not in costs
    return costs[first] <= costs[second]


def _dominant(report_input: ReportInput, survivors: Sequence[str], overrides: _Overrides) -> list[str]:
    costs = _costs(report_input)
    return [
        arm
        for arm in survivors
        if all(
            _cost_at_least(costs, arm, other) and _quality_at_least(report_input, arm, other, overrides)
            for other in survivors
            if other != arm
        )
    ]


def dominant_options(report_input: ReportInput) -> list[str]:
    """The surviving options each at least as good as every other survivor on every axis that matters (this module's
    Dominance). One means that one dominates; two or more tie on every axis; none means no option dominates.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The dominant survivors' ids, in the input's arm order.
    :rtype: list[str]
    """
    return _dominant(report_input, surviving_options(report_input), _Overrides())


def _ranking_rule(report_input: ReportInput) -> bool:
    return any(family.merit_axes for family in report_input.contrast_families)


def _outcome(report_input: ReportInput, overrides: _Overrides) -> _Outcome:
    if report_kind(report_input) == "measurement":
        return frozenset({"no_call"}), ()
    survivors = surviving_options(report_input)
    if not survivors:
        return frozenset({"no_call"}), ()
    control = control_arm(report_input)
    if control is not None:
        separable = [_separable(report_input, arm, overrides) for arm in survivors if arm != control]
        if separable and all(found is False for found in separable):
            return frozenset({"no_call"}), ()
    dominant = _dominant(report_input, survivors, overrides)
    if len(dominant) == 1:
        return frozenset({"decide"}), (dominant[0],)
    if _ranking_rule(report_input):
        return frozenset({"decide"}), ()
    return frozenset({"tradeoff"}), ()


def allowed_modes(report_input: ReportInput) -> frozenset[str]:
    """The modes a plan for this input may take (this module's Modes).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The allowed modes, of ``decide``, ``tradeoff`` and ``no_call``.
    :rtype: frozenset[str]
    """
    return _outcome(report_input, _Overrides())[0]


# --- immateriality -------------------------------------------------------------------------------------------------


def _probes(flips: Sequence[float]) -> list[float]:
    """Values covering every piece of ``[0, inf)`` that ``flips`` cut it into: zero, each flip, each midpoint, and one
    past the last."""
    points = sorted({0.0, *flips})
    middles = [(low + high) / 2 for low, high in zip(points, points[1:], strict=False)]
    return sorted({*points, *middles, points[-1] + 1.0})


def _holds(report_input: ReportInput, base: _Outcome, trials: Iterator[_Overrides]) -> bool:
    return all(_outcome(report_input, overrides) == base for overrides in trials)


def _margin_holds(report_input: ReportInput, base: _Outcome) -> bool:
    aspects = [
        aspect
        for aspect in report_input.aspects
        if aspect.margin is not None or any(reading.margin is not None for reading in aspect.readings)
    ]
    if not aspects:
        return False
    for aspect in aspects:
        flips = [flip_margin(reading) for reading in aspect.readings]
        if any(flip is None for flip in flips):
            return False
        probes = _probes([flip for flip in flips if flip is not None])
        if not _holds(report_input, base, (_Overrides(margins={aspect.id: m}) for m in probes)):
            return False
    return True


def _floor_holds(report_input: ReportInput, base: _Outcome, key: str | None) -> bool:
    reads = [read for read in report_input.floors if read.floor == key]
    if not reads or any(read.interval is None or read.margin is None for read in reads):
        return False
    flips = [max(0.0, -float(end)) for read in reads if read.interval is not None for end in read.interval]
    return _holds(report_input, base, (_Overrides(floors={key: m}) for m in _probes(flips) if key is not None))


def _late_immaterial(report_input: ReportInput, late: LateCriterion, base: _Outcome) -> bool:
    if late.criterion == "margin":
        return _margin_holds(report_input, base)
    if late.criterion == "floor":
        return _floor_holds(report_input, base, late.name)
    return late.criterion == "bar"


def caveat_materiality(report_input: ReportInput) -> dict[str, Materiality]:
    """For every caveat, whether code proved it immaterial, with the checked line it renders (this module's
    Immateriality).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: Each caveat's materiality, by id, in :func:`report_caveats`' order.
    :rtype: dict[str, Materiality]
    """
    base = _outcome(report_input, _Overrides())
    lates = {_late_id(late): late for late in report_input.caveats.late_criteria}
    found: dict[str, Materiality] = {}
    for caveat in report_caveats(report_input):
        late = lates.get(caveat.id)
        if late is not None and _late_immaterial(report_input, late, base):
            found[caveat.id] = Materiality(caveat.id, True, _CHECKED[late.criterion])
        else:
            found[caveat.id] = Materiality(caveat.id, False, None)
    return found
