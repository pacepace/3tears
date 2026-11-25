"""What code decides about a report input: the options that survive, which one dominates, each arm's standing on
the asked aspects, which readings are left unsettled, and whether a late criterion could have changed the call.

The numbers side of the call: it reads the report input as arithmetic and prints nothing, so no wording reaches a
decision. The caveats over it are raised in :mod:`~threetears.evals.analysis.caveats`, and the page words both
(:mod:`~threetears.evals.analysis.report.call`).

**Unsettled past the margin.** A reading is left unsettled past its margin when it is not within the margin (the one
rule: its :func:`~threetears.evals.analysis.report_margins.flip_margin` at or above the margin), its interval reaches a
loss (its low end below zero), and its high end sits above the margin's line (``-margin``): the evidence allows a loss
past the margin and a change short of it. An interval wholly past the line is settled, not unsettled. Asked and
unasked aspects alike: an asked one's reading decides a candidate's level only where nothing else shows it better, so
a candidate shown better on one asked aspect, or left the best of the survivors when the control fails a gate, can be
the call while another asked reading stays unsettled; an unasked one's the decision never reads.

**Immateriality** (:func:`late_immaterial`). A late criterion is immaterial when the decision, the allowed modes
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
not a criterion with a range. An unsettled aspect is open too: the sweep varies a margin, not the change a reading's
interval leaves open, and the decision reads a reading only as within its margin or not, so a call that holds at
every margin would show only that code's rule does not weigh the reading (it never reads an unasked aspect), which is
what the caveat says. Whether a loss past the margin would change the call is the plan's ``material_if`` to state.

**Survivors** (:func:`surviving_options`, GENERATOR §3). An option survives when the gate check
(:func:`~threetears.evals.analysis.criteria.gate_status`) does not fail it, and no guardrail the pipeline read on it
is ``breached``. The check reads each criterion of kind ``gate`` or ``guardrail`` on its point estimate (at least the
threshold where higher is better, at most where lower is), overall, in its one task, in every task or in every cell as
its scope says. A gate with no figure for the arm is not shown passed, so the arm does not survive it. Borderline
counts as passing for the decision (GENERATOR §3). Floors and bars never remove an option: a bar warns, and a floor
read broken on an asked aspect caps the option's standing there (Dominance); a margin is the asked aspect's own axis,
not a gate.

**Dominance** (:func:`dominant_options`). One surviving option dominates when it is at least as good as every other
survivor on every axis that matters:

- *Every gate*: every survivor passes every gate, so the gates tie among survivors.
- *Cost*: cost per 1,000 of what the cost counts as the one cost rule reads it
  (:func:`~threetears.evals.analysis.costs.costs_per_thousand`: a dollar measure's overall figures or the
  scoreboard's, else a dollar measure blended over the tasks by traffic share), lower is better. Two arms with no cost tie; an arm with
  no cost is not at least as good as one with a cost, nor the reverse.
- *The asked aspect*: in a change, each arm takes a level: a candidate shown better (a confirmatory ``better``
  reading, or an ``improved`` contrast on an asked aspect) with no floor on an asked aspect read ``broken`` is 2; one
  shown no worse (every floor on every asked aspect held, each declaring one, and within the margin on each asked
  reading that declares one) is 1; any other candidate is 0; the control is 1. A floor read broken (its whole
  interval past the floor) caps a candidate shown better at 0, not shown no worse, since a broken floor is not held;
  an unclear one caps nothing. An ``improved`` contrast on a judged dimension whose judge the input states
  uncalibrated (its tier and calibration undetermined or insufficient, the reading ``judge_uncalibrated`` makes) is
  set aside: it does not raise the level, and the candidate's other improvements count as usual. Only those two
  states set aside an improvement: hosts often make a judge ad hoc, without the hours of human rating calibration
  takes, and a judge measured and found short (``separation``, ``incidental``) still counts. An arm is at least
  as good as another at the same level or above. In a choice, an arm is at least as good as another unless a pair
  reading on an asked aspect shows the other ahead of it. With no asked reading, pair or floor, the aspect ties.

**Modes** (:func:`decision_at`'s ``modes``, GENERATOR §1.1), read in this order:

1. *No call* when the report measures one arm (there is nothing to choose), when no option survives, or, in a change,
   when no surviving candidate is separated: its asked readings' intervals cross zero and sit outside the margin (or
   no margin is declared), and no contrast on an asked aspect separated it (``improved``, ``regressed`` or
   ``equivalent``, as its family's correction reads it: corrected only where that names a method). A
   candidate with neither an asked reading nor a contrast on an asked aspect forces nothing.
2. *Decide* when exactly one survivor dominates; the plan recommends that one.
3. *Decide* when two or more survive, none dominates alone, and the campaign declared a rule that ranks the axes (a
   contrast family's merit priority): the rule picks, among the survivors.
4. *Trade off* otherwise: two or more survive, none dominates alone, and no rule ranks them.

**The same rules, one point at a time**, for the analyst's analyses (:mod:`threetears.evals.analysis.analyst.analyses`):
:func:`decision_at` is the decision at hypothetical margins and floor margins, one point of the Immateriality
sweep; :func:`asked_standing` is each arm's standing on the asked aspects, the quality axis
dominance compares beside cost; :func:`weighed_axes` names the axes the snapshot declares, so an analysis can say
when the rule has nothing but cost to weigh.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal

from threetears.evals.analysis.bundle.schema import SEPARATED_VERDICTS
from threetears.evals.analysis.costs import costs_per_thousand
from threetears.evals.analysis.criteria import GateStatus, gate_status
from threetears.evals.analysis.per_input import kept_for
from threetears.evals.analysis.report_input import (
    Aspect,
    Contrast,
    FloorRead,
    LateCriterion,
    Reading,
    ReportInput,
)
from threetears.evals.analysis.report_margins import flip_margin

#: ``report_kind`` is this module's own step, and ``DecisionAxis`` and ``ReportKind`` the closed words its reads
#: return: nothing outside it names them.
__all__ = [
    "AskedStanding",
    "DecisionAt",
    "StandingWord",
    "asked_aspects",
    "asked_standing",
    "comparisons",
    "control_arm",
    "decision_at",
    "dominant_options",
    "floors_not_held",
    "judge_described",
    "judged_dimensions",
    "late_immaterial",
    "separating_contrasts",
    "shown_better",
    "stated_uncalibrated",
    "surviving_options",
    "unsettled_arms",
    "weighed_axes",
]

#: Judge tiers and calibration states that mean no human calibration stands behind the judge.
_UNCALIBRATED = frozenset({"undetermined", "insufficient"})

#: What a report is, from its arms.
ReportKind = Literal["measurement", "change", "choice"]


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
    :return: The control's id; None when no arm is the control.
    """
    return next((arm.id for arm in report_input.arms if arm.role == "control"), None)


def report_kind(report_input: ReportInput) -> ReportKind:
    """What the report is, from its arms: one arm measured, candidates against a control, or a choice among arms.

    :param report_input: The report input.
    :return: ``measurement``, ``change`` or ``choice``.
    """
    if len({arm.id for arm in report_input.arms}) <= 1:
        return "measurement"
    return "change" if control_arm(report_input) is not None else "choice"


def comparisons(report_input: ReportInput) -> list[frozenset[str]]:
    """The comparisons the report makes, each as the arms in it: candidates against the control, pairs, or one arm.

    :param report_input: The report input.
    :return: Each comparison's arms, in the input's order.
    """
    arms = [arm.id for arm in report_input.arms]
    control = control_arm(report_input)
    if len(set(arms)) <= 1:
        return [frozenset(arms)] if arms else []
    if control is not None:
        return [frozenset({control, arm}) for arm in arms if arm != control]
    return [frozenset(pair) for pair in combinations(arms, 2)]


def _judge_standing(report_input: ReportInput, dimension: str) -> set[str]:
    """What the input states of how far one judged dimension's judge can be trusted: its tier and calibration state."""
    judged = report_input.judged.get(dimension)
    stated = [report_input.caveats.judge_tiers.get(dimension)]
    if judged is not None:
        stated += [judged.tier, judged.calibration]
    return {value for value in stated if value is not None}


def judge_described(report_input: ReportInput, dimension: str) -> bool:
    """Whether the input describes one judged dimension's judge at all: its model, tier or calibration state.

    :param report_input: The report input.
    :param dimension: The judged dimension, by key.
    :return: True where it states the judge's model, tier or calibration state.
    """
    judged = report_input.judged.get(dimension)
    return bool(_judge_standing(report_input, dimension)) or (judged is not None and judged.judge_model is not None)


def stated_uncalibrated(report_input: ReportInput, dimension: str) -> bool:
    """Whether the input states one judged dimension's judge uncalibrated: it states the judge's tier or calibration
    state, and every one it states is undetermined or insufficient. The one reading of a judge's calibration the
    caveats and the asked standing share.

    :param report_input: The report input.
    :param dimension: The judged dimension, by key.
    :return: True where every tier or calibration state it states means no human calibration stands behind it.
    """
    standing = _judge_standing(report_input, dimension)
    return bool(standing) and standing <= _UNCALIBRATED


def judged_dimensions(report_input: ReportInput) -> list[str]:
    """Every judged dimension the input names: those it describes, then those only a tier names, in the input's
    order."""
    tiers = report_input.caveats.judge_tiers
    return [*report_input.judged, *(name for name in tiers if name not in report_input.judged)]


# --- gates and survivors -------------------------------------------------------------------------------------------


#: The gate statuses an option survives: borderline counts as passing for the decision (GENERATOR §3).
_SURVIVING: frozenset[GateStatus] = frozenset({"pass", "borderline"})


def _passes(report_input: ReportInput, arm: str) -> bool:
    return gate_status(report_input, arm) in _SURVIVING


def surviving_options(report_input: ReportInput) -> list[str]:
    """The arms that pass every gate and breach no guardrail (this module's Survivors).

    :param report_input: The report input.
    :return: The surviving arms' ids, in the input's arm order.
    """
    breached = {
        str(guardrail.get("arm")) for guardrail in report_input.guardrails if guardrail.get("decision") == "breached"
    }
    return [arm.id for arm in report_input.arms if arm.id not in breached and _passes(report_input, arm.id)]


# --- the asked aspect ----------------------------------------------------------------------------------------------


def asked_aspects(report_input: ReportInput) -> list[Aspect]:
    """The aspects the campaign's question asks about, in the input's order.

    :param report_input: The report input.
    :return: The asked aspects.
    """
    return [aspect for aspect in report_input.aspects if aspect.asked]


@dataclass(frozen=True)
class _Better:
    """What shows a candidate better than the control, and what is set aside.

    :ivar counted: Each improvement that counts, as a clause names it: a confirmatory ``better`` reading by its
        aspect, an ``improved`` contrast on an asked aspect by its measure.
    :ivar set_aside: Each judged dimension an ``improved`` contrast on an asked aspect reads whose judge the input
        states uncalibrated (:func:`_stated_uncalibrated`), by key: it does not count.
    """

    counted: tuple[str, ...]
    set_aside: tuple[str, ...]


def _better(report_input: ReportInput, arm: str) -> _Better:
    counted = [
        f"its confirmatory reading on aspect {aspect.id!r}"
        for aspect in report_input.aspects
        for reading in aspect.readings
        if reading.arm == arm and reading.reading == "better" and reading.confirmatory
    ]
    set_aside: list[str] = []
    asked = {aspect.id for aspect in asked_aspects(report_input)}
    for contrast in report_input.contrasts:
        if contrast.arm != arm or contrast.verdict != "improved" or contrast.aspect not in asked:
            continue
        # Only a judged dimension's judge carries a tier or calibration state, so a measure is never set aside.
        if stated_uncalibrated(report_input, contrast.measure):
            set_aside.append(contrast.measure)
        else:
            counted.append(f"its improved contrast on {contrast.measure!r}")
    return _Better(tuple(dict.fromkeys(counted)), tuple(dict.fromkeys(set_aside)))


def shown_better(report_input: ReportInput, arm: str) -> bool:
    """Whether a candidate is shown better: a confirmatory ``better`` reading, or an ``improved`` contrast on an asked
    aspect, save one on a judged dimension whose judge the input states uncalibrated (tier and calibration undetermined
    or insufficient), which is set aside. Only those two states set aside an improvement: hosts often make a judge ad
    hoc, without the hours of human rating calibration takes, and a judge measured and found short (``separation``,
    ``incidental``) still counts.

    :param report_input: The report input.
    :param arm: The candidate's id.
    :return: True when the evidence shows it better than the control.
    """
    return bool(_better(report_input, arm).counted)


def _floor_status(read: FloorRead, overrides: _Overrides) -> str:
    """A floor's status: the pipeline's, or at a hypothetical margin, read on its interval."""
    margin = overrides.floors.get(read.floor)
    if margin is None or read.interval is None:
        return read.status
    low, high = read.interval
    if -low < margin:
        return "held"
    return "broken" if -high >= margin else "unclear"


def _broken_floors(report_input: ReportInput, arm: str, overrides: _Overrides) -> list[str]:
    """The floors on the asked aspects a candidate's read breaks (its whole interval past the floor), by key: each caps
    its standing below shown better."""
    asked = {aspect.id for aspect in asked_aspects(report_input)}
    return list(
        dict.fromkeys(
            read.floor
            for read in report_input.floors
            if read.arm == arm and read.aspect in asked and _floor_status(read, overrides) == "broken"
        )
    )


def _floors_not_held(
    report_input: ReportInput,
    arm: str,
    asked: Sequence[Aspect],
    overrides: _Overrides,
    *,
    reasons: bool = True,
    capped: bool = False,
) -> list[str]:
    """One clause per floor not held on the asked aspects, and per aspect declaring none; with ``reasons``, a floor's
    clause ends with the input's own reason where it speaks for the status read. With ``capped`` (the candidate is shown
    better), a broken floor's clause says it caps the standing."""
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
            if status == "broken" and capped:
                clauses.append(
                    f"floor {key} is broken, its whole interval past the floor, which caps it at not shown no worse "
                    "though it is shown better"
                )
            elif status != "held":
                reason = f" ({read.reason.rstrip('.')})" if reasons and read.reason and status == read.status else ""
                clauses.append(f"floor {key} is {status}{reason}")
    return clauses


def floors_not_held(report_input: ReportInput, arm: str, asked: Sequence[Aspect]) -> list[str]:
    """Per asked aspect, why a candidate is not shown no worse there; empty when every floor reads held.

    An aspect's floors are those read on it for any candidate; one with none declares no floor.

    :param report_input: The report input.
    :param arm: The candidate's id.
    :param asked: The asked aspects.
    :return: One clause per floor not held, or per aspect declaring none.
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


def _unsettled(aspect: Aspect, reading: Reading) -> bool:
    """Whether a reading is left unsettled past its declared margin (this module's Unsettled past the margin)."""
    margin = _margin(aspect, reading, _Overrides())
    if margin is None or reading.within_interval is None or _within(aspect, reading, _Overrides()):
        return False
    low, high = (float(end) for end in reading.within_interval)
    return low < 0 and high > -margin


def unsettled_arms(report_input: ReportInput) -> dict[str, tuple[str, ...]]:
    """Each aspect on which a surviving candidate's reading is left unsettled past its declared margin (this module's
    Unsettled past the margin), by its caveat id ``unsettled:<aspect>``, with those candidates.

    :param report_input: The report input.
    :return: The candidates left unsettled, in the readings' order, by caveat id in the aspects' order; empty for none.
    """
    survivors = set(surviving_options(report_input))
    control = control_arm(report_input)
    found: dict[str, tuple[str, ...]] = {}
    for aspect in report_input.aspects:
        arms = tuple(
            dict.fromkeys(
                reading.arm
                for reading in aspect.readings
                if reading.arm in survivors and reading.arm != control and _unsettled(aspect, reading)
            )
        )
        if arms:
            found[f"unsettled:{aspect.id}"] = arms
    return found


def _shown_no_worse(report_input: ReportInput, arm: str, overrides: _Overrides) -> bool:
    asked = asked_aspects(report_input)
    if not asked or _floors_not_held(report_input, arm, asked, overrides):
        return False
    for aspect in asked:
        for reading in aspect.readings:
            if reading.arm != arm:
                continue
            if _margin(aspect, reading, overrides) is None:
                # A superiority question (asked, no margin) reads the drop against zero: shown worse is never no worse.
                if reading.reading == "worse":
                    return False
            elif not _within(aspect, reading, overrides):
                return False
    return True


def _level(report_input: ReportInput, arm: str, control: str, overrides: _Overrides) -> int:
    """A change's arm on the asked aspect: 2 shown better with no asked floor broken, 1 shown no worse (the control),
    0 otherwise. A broken floor is never held, so it caps a candidate shown better at 0."""
    if arm == control:
        return 1
    if shown_better(report_input, arm) and not _broken_floors(report_input, arm, overrides):
        return 2
    return 1 if _shown_no_worse(report_input, arm, overrides) else 0


def separating_contrasts(report_input: ReportInput, arm: str) -> list[Contrast]:
    """The contrasts on an asked aspect that separate a candidate from the control: verdict ``improved``,
    ``regressed`` or ``equivalent``, each as its family's correction reads it (uncorrected where that is ``none``), in
    the input's order. How a question a measure
    answers (a latency aspect's p95) separates a candidate.

    :param report_input: The report input.
    :param arm: The candidate's id.
    :return: The separating contrasts; empty for none.
    """
    asked = {aspect.id for aspect in asked_aspects(report_input)}
    return [
        contrast
        for contrast in report_input.contrasts
        if contrast.arm == arm and contrast.aspect in asked and contrast.verdict in SEPARATED_VERDICTS
    ]


def _reading_separates(report_input: ReportInput, arm: str, overrides: _Overrides) -> bool | None:
    """Whether a candidate's asked readings separate it: an interval clear of zero, or within the margin, on any one.
    None where it has no asked reading with an interval."""
    seen = False
    for aspect in asked_aspects(report_input):
        for reading in aspect.readings:
            if reading.arm != arm or reading.within_interval is None:
                continue
            seen = True
            low, high = reading.within_interval
            if low > 0 or high < 0 or _within(aspect, reading, overrides):
                return True
    return False if seen else None


def _separable(report_input: ReportInput, arm: str, overrides: _Overrides) -> bool | None:
    """Whether a candidate's asked evidence separates it: a separating contrast on an asked aspect
    (:func:`separating_contrasts`), or its asked readings (at the margins ``overrides`` sets). None where it has
    neither an asked reading with an interval nor a separating contrast."""
    if separating_contrasts(report_input, arm):
        return True
    return _reading_separates(report_input, arm, overrides)


# --- cost ----------------------------------------------------------------------------------------------------------


def _costs(report_input: ReportInput) -> dict[str, float]:
    """Each arm's cost per 1,000 of what it counts, by the one cost rule (this module's Dominance, cost)."""
    costs = costs_per_thousand(report_input)
    return {arm.id: float(costs[arm.id]) for arm in report_input.arms if arm.id in costs}


# --- dominance and modes -------------------------------------------------------------------------------------------


def _quality_at_least(report_input: ReportInput, first: str, second: str, overrides: _Overrides) -> bool:
    control = control_arm(report_input)
    if control is not None:
        return _level(report_input, first, control, overrides) >= _level(report_input, second, control, overrides)
    return not any(
        set(pair.arms) == {first, second} and pair.ahead == second
        for aspect in asked_aspects(report_input)
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
    :return: The dominant survivors' ids, in the input's arm order.
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


#: An axis code's decision weighs, as the snapshot declares it (:func:`weighed_axes`).
DecisionAxis = Literal["gates", "control", "asked", "ranking", "cost"]


def weighed_axes(report_input: ReportInput) -> tuple[DecisionAxis, ...]:
    """The axes code's decision weighs, as the snapshot declares them, in this order: ``gates`` (a gate or guardrail
    criterion, or a guardrail the pipeline read), ``control`` (a control, which a candidate must be shown no worse
    than), ``asked`` (an asked aspect), ``ranking`` (a contrast family's merit priority) and ``cost`` (an arm whose
    cost the cost rule states).

    Where nothing but cost is declared, this module's dominance rule reduces to a cost ranking: every survivor ties
    on the other axes, so the cheapest dominates. The analyst's analyses read that as no dominance call
    (:mod:`~threetears.evals.analysis.analyst.analyses`), since nothing declared says the cheapest is good enough; the
    modes :func:`decision_at` reads apply the rule as it stands.

    :param report_input: The report input.
    :return: The axes declared, in that order; empty where none is.
    """
    found: list[DecisionAxis] = []
    if any(criterion.kind in ("gate", "guardrail") for criterion in report_input.criteria) or report_input.guardrails:
        found.append("gates")
    if control_arm(report_input) is not None:
        found.append("control")
    if asked_aspects(report_input):
        found.append("asked")
    if _ranking_rule(report_input):
        found.append("ranking")
    if _costs(report_input):
        found.append("cost")
    return tuple(found)


@dataclass(frozen=True)
class DecisionAt:
    """The decision code makes at given values of the late criteria it can sweep: the allowed modes, and the one option
    that dominates.

    :ivar modes: The allowed modes at those values (this module's Modes).
    :ivar dominant: The one option that dominates there; empty when none does alone.
    """

    modes: frozenset[str]
    dominant: tuple[str, ...]


def decision_at(
    report_input: ReportInput,
    *,
    margins: Mapping[str, float] | None = None,
    floors: Mapping[str, float] | None = None,
) -> DecisionAt:
    """The decision at hypothetical margins and floor margins: what this module's Immateriality sweeps, one point at a
    time. With neither, the decision as declared.

    A margin set on an aspect replaces the margin every reading on it is read against; a margin set on a floor reads
    that floor on each candidate's interval (held above minus its low end, broken at or below minus its high end), as
    the late-floor check does. Everything else is as the input states it.

    :param report_input: The report input.
    :param margins: A margin per aspect id, in points.
    :param floors: A margin per floor key, in points.
    :return: The allowed modes and the dominant option at those values.
    :raises ValueError: A margin names an aspect the input does not hold, or a floor margin a floor no candidate was
        read on.
    """
    margins = dict(margins or {})
    floors = dict(floors or {})
    aspects = {aspect.id for aspect in report_input.aspects}
    unknown = sorted(set(margins) - aspects)
    if unknown:
        raise ValueError(f"the input holds no aspect {', '.join(repr(name) for name in unknown)}")
    read = {floor.floor for floor in report_input.floors}
    unread = sorted(set(floors) - read)
    if unread:
        raise ValueError(f"the input reads no floor {', '.join(repr(name) for name in unread)}")
    overrides = _Overrides(margins=margins, floors=floors)
    modes, dominant = _declared(report_input) if not margins and not floors else _outcome(report_input, overrides)
    return DecisionAt(modes=modes, dominant=dominant)


#: How an arm stands on the asked aspects, as this module's Dominance reads it. In a change: ``shown_better`` (level 2,
#: no asked floor broken), ``control`` or ``shown_no_worse`` (level 1), ``not_shown_no_worse`` (level 0). In a choice,
#: by the asked pair readings: ``behind`` when one shows another arm ahead of it, else ``not_behind``.
StandingWord = Literal["shown_better", "control", "shown_no_worse", "not_shown_no_worse", "behind", "not_behind"]


@dataclass(frozen=True)
class AskedStanding:
    """One arm's standing on the asked aspects, the axis dominance compares beside cost (this module's Dominance).

    :ivar arm: The arm's id.
    :ivar standing: Its standing (:data:`StandingWord`).
    :ivar level: In a change, its level: 2 shown better, 1 shown no worse or the control, 0 otherwise; one arm is at
        least as good as another on the asked aspects exactly when its level is at least the other's. None in a choice.
    :ivar behind: In a choice, the arms an asked pair reading shows ahead of it; one arm is at least as good as another
        exactly when the other is not among them. Empty in a change.
    :ivar why: In a change, for a candidate: why it is not shown no worse (each floor not held, a broken one saying it
        caps a candidate shown better, each aspect declaring none, and each asked reading not within its margin), and
        at any standing each improvement set aside because its judge is not calibrated, in words that name each by its
        id and state no figure (a floor read's own reason, and its figures, are read by its keys). Empty otherwise.
    :ivar set_aside: In a change, for a candidate: each judged dimension an ``improved`` contrast on an asked aspect
        reads whose judge the input states uncalibrated, by key; set aside, it does not raise the standing. Empty
        otherwise.
    """

    arm: str
    standing: StandingWord
    level: int | None
    behind: tuple[str, ...]
    why: tuple[str, ...]
    set_aside: tuple[str, ...] = ()


def _not_within(report_input: ReportInput, arm: str) -> list[str]:
    """Each asked reading of a candidate that declares a margin and is not within it, as a clause."""
    return [
        f"its reading on aspect {aspect.id!r} is not within the margin"
        for aspect in asked_aspects(report_input)
        for reading in aspect.readings
        if reading.arm == arm
        and _margin(aspect, reading, _Overrides()) is not None
        and not _within(aspect, reading, _Overrides())
    ]


def asked_standing(report_input: ReportInput) -> dict[str, AskedStanding]:
    """Every arm's standing on the asked aspects: the quality axis dominance compares (this module's Dominance).

    :param report_input: The report input.
    :return: Each arm's standing, by id, in the input's arm order.
    """
    control = control_arm(report_input)
    asked = asked_aspects(report_input)
    found: dict[str, AskedStanding] = {}
    for arm in dict.fromkeys(held.id for held in report_input.arms):
        if control is None:
            behind = tuple(
                dict.fromkeys(
                    pair.ahead
                    for aspect in asked
                    for pair in aspect.pairs
                    if arm in pair.arms and pair.ahead is not None and pair.ahead != arm
                )
            )
            found[arm] = AskedStanding(arm, "behind" if behind else "not_behind", None, behind, ())
            continue
        level = _level(report_input, arm, control, _Overrides())
        if arm == control:
            found[arm] = AskedStanding(arm, "control", level, (), ())
            continue
        better = _better(report_input, arm)
        set_aside = tuple(
            f"its improvement on {dimension!r} is set aside: that dimension's judge is not calibrated, so it does not "
            "raise the standing"
            for dimension in better.set_aside
        )
        if level == 2:
            found[arm] = AskedStanding(arm, "shown_better", level, (), set_aside, better.set_aside)
        elif level == 1:
            found[arm] = AskedStanding(arm, "shown_no_worse", level, (), set_aside, better.set_aside)
        else:
            floors = _floors_not_held(
                report_input, arm, asked, _Overrides(), reasons=False, capped=bool(better.counted)
            )
            why = [*floors, *_not_within(report_input, arm), *set_aside]
            if not asked:
                why.append("no aspect is asked, so nothing shows it no worse")
            found[arm] = AskedStanding(arm, "not_shown_no_worse", level, (), tuple(why), better.set_aside)
    return found


# --- immateriality -------------------------------------------------------------------------------------------------


def _probes(flips: Sequence[float]) -> list[float]:
    """Values covering every piece of ``[0, inf)`` that ``flips`` cut it into: zero, each flip, each midpoint, and one
    past the last."""
    points = sorted({0.0, *flips})
    middles = [(low + high) / 2 for low, high in zip(points, points[1:], strict=False)]
    return sorted({*points, *middles, points[-1] + 1.0})


def _declared(report_input: ReportInput) -> _Outcome:
    """The decision as declared: the one every sweep is held against, read once per input."""
    return kept_for(report_input).get(_declared, lambda: _outcome(report_input, _Overrides()))


def _holds(report_input: ReportInput, base: _Outcome, trials: Iterator[_Overrides]) -> bool:
    return all(_outcome(report_input, overrides) == base for overrides in trials)


def _margined(report_input: ReportInput) -> list[Aspect]:
    """The aspects that declare a margin, on the aspect or on one of its readings."""
    return [
        aspect
        for aspect in report_input.aspects
        if aspect.margin is not None or any(reading.margin is not None for reading in aspect.readings)
    ]


def _margin_holds(report_input: ReportInput, base: _Outcome) -> bool:
    aspects = _margined(report_input)
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


def late_immaterial(report_input: ReportInput, late: LateCriterion) -> bool:
    """Whether code proves a late criterion immaterial: the call holds across its whole plausible range (this
    module's Immateriality). A late margin, floor or bar can be swept; any other late criterion is open.

    :param report_input: The report input.
    :param late: The late criterion.
    :return: True when the decision is the same at every value it could take.
    """
    base = _declared(report_input)
    if late.criterion == "margin":
        return _margin_holds(report_input, base)
    if late.criterion == "floor":
        return _floor_holds(report_input, base, late.name)
    return late.criterion == "bar"
