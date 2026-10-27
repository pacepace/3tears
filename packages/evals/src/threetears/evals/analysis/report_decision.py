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
``unsettled:<aspect>``                    a surviving candidate's reading on an aspect declaring a margin is left
                                          unsettled past it (:func:`unsettled_arms`)
``latency``                               latency is not compared
``unreported:<observation>``              an observation recorded but not reported
========================================  ==========================================================================

**Implied caveats** (:func:`implied_caveats`). The ids above are the caveats the input states as caveats, the ones a
plan's flag and confidence clauses are checked against. What the input holds implies more, and an analyst's report
rests on these too; each is raised from what the input holds, never from what kind of eval it is, with one stable id
and the condition under which it matters (``matters_if``). They are not plan caveat ids: a stored plan never names
one.

========================================  ==========================================================================
``judge_unstated``                        a judged dimension whose judge the input does not describe: no model, no
                                          evidence tier, no calibration state
``judge_partly_uncalibrated``             some judged dimensions' judges are stated uncalibrated (tier and
                                          calibration undetermined or insufficient) while ``judge_uncalibrated``,
                                          which needs every one so, is not raised
``no_control``                            more than one arm and none is the control: no reading or contrast is paired
                                          against a baseline
``nothing_declared``                      no gate, guardrail or floor criterion, no margin and no floor read: nothing
                                          says what is good enough, so code ranks options on cost alone (a bar only
                                          warns)
``point_estimate:<criterion>``            a gate or guardrail a surviving option clears only on a point estimate:
                                          the figure it reads holds no interval there, stated or derived from its
                                          cells (:mod:`~threetears.evals.analysis.cell_mean`; a percentile never
                                          holds one)
``borderline:<criterion>``                a gate or guardrail a surviving option clears on its point estimate while
                                          the figure's interval reaches the wrong side of the line (the gate check's
                                          borderline, which counts as passing)
``best_of_n``                             a pick among three candidates or more: the best of many reads
                                          high by chance
========================================  ==========================================================================

**Scope**, which decides where a caveat may land in the plan. *Report-wide*: every late criterion, the uncalibrated
judge, and a confound, apparatus confound or short run that touches every comparison the report makes (each
candidate against the control in a change, every pair in a choice, the one arm in a measurement; a short run touches
the comparisons its arm is in, every one when it names no arm; a confound touches the comparisons whose contrasts name
it), and an unsettled aspect whose unsettled candidates include the option code's decision names (decide mode with
one option dominating, which the plan must recommend). Report-wide caveats are what the decision's confidence is
conditional on. *Scoped*: a confound or short run touching only some comparisons, and an unsettled aspect on no
option code's decision names, said beside the claim it qualifies; where the plan itself recommends an unsettled
candidate (a pick by a ranking rule, which code does not make), its confidence is conditional on that caveat too
(:mod:`~threetears.evals.analysis.report_plan_checks`). *Method-level*: latency, an unreported observation, and an
apparatus confound that is not report-wide.

**Unsettled past the margin.** A reading is left unsettled past its margin when it is not within the margin (the one
rule: its :func:`~threetears.evals.analysis.report_margins.flip_margin` at or above the margin), its interval reaches a
loss (its low end below zero), and its high end sits above the margin's line (``-margin``): the evidence allows a loss
past the margin and a change short of it. An interval wholly past the line is settled, not unsettled. Asked and
unasked aspects alike: an asked one's reading decides a candidate's level only where nothing else shows it better, so
a candidate shown better on one asked aspect, or left the best of the survivors when the control fails a gate, can be
the call while another asked reading stays unsettled; an unasked one's the decision never reads.

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
not a criterion with a range. An unsettled aspect is open too: the sweep varies a margin, not the change a reading's
interval leaves open, and the decision reads a reading only as within its margin or not, so a call that holds at
every margin would show only that code's rule does not weigh the reading (it never reads an unasked aspect), which is
what the caveat says. Whether a loss past the margin would change the call is the plan's ``material_if`` to state.

**Survivors** (:func:`surviving_options`, GENERATOR §3). An option survives when the gate check
(:func:`~threetears.evals.analysis.report_gates.gate_check`) does not fail it, and no guardrail the pipeline read on it
is ``breached``. The check reads each criterion of kind ``gate`` or ``guardrail`` on its point estimate (at least the
threshold where higher is better, at most where lower is), overall, in its one task, in every task or in every cell as
its scope says. A gate with no figure for the arm is not shown passed, so the arm does not survive it. Borderline
counts as passing for the decision (GENERATOR §3). Floors and bars warn and never remove an option; a margin is the
asked aspect's own axis, not a gate.

**Dominance** (:func:`dominant_options`). One surviving option dominates when it is at least as good as every other
survivor on every axis that matters:

- *Every gate*: every survivor passes every gate, so the gates tie among survivors.
- *Cost*: cost per 1,000 of what the cost counts as the ``cost_per_1k.<arm>`` figure key reads it, the one cost rule
  (:mod:`~threetears.evals.analysis.report.figure_keys`: a dollar measure's overall figures or the scoreboard's, else
  a dollar measure blended over the tasks by traffic share), lower is better. Two arms with no cost tie; an arm with
  no cost is not at least as good as one with a cost, nor the reverse.
- *The asked aspect*: in a change, each arm takes a level: a candidate shown better (a confirmatory ``better``
  reading, or an ``improved`` contrast on an asked aspect) is 2; one shown no worse (every floor on every asked
  aspect held, each declaring one, and within the margin on each asked reading that declares one) is 1; any other
  candidate is 0; the control is 1. An arm is at least as good as another at the same level or above. In a choice, an
  arm is at least as good as another unless a pair reading on an asked aspect shows the other ahead of it. With no
  asked reading, pair or floor, the aspect ties.

**Modes** (:func:`allowed_modes`, GENERATOR §1.1), read in this order:

1. *No call* when the report measures one arm (there is nothing to choose), when no option survives, or, in a change,
   when no surviving candidate is separated: its asked readings' intervals cross zero and sit outside the margin (or
   no margin is declared), and no contrast on an asked aspect separated it (``improved``, ``regressed`` or
   ``equivalent``, as its family's correction reads it: corrected only where that names a method). A candidate with neither an asked reading nor a contrast on an asked
   aspect forces nothing.
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

from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.figure_keys import resolve
from threetears.evals.analysis.bundle.schema import SEPARATED_VERDICTS
from threetears.evals.analysis.report_gates import GateCheck, GateStatus, gate_check, gate_findings
from threetears.evals.analysis.report_input import (
    Aspect,
    Contrast,
    FloorRead,
    LateCriterion,
    Reading,
    ReportInput,
)
from threetears.evals.analysis.report_margins import criterion_figures, flip_margin

__all__ = [
    "AskedStanding",
    "Caveat",
    "CandidateMargin",
    "DecisionAt",
    "ImpliedCaveat",
    "MarginSensitivity",
    "Materiality",
    "allowed_modes",
    "asked_standing",
    "caveat_materiality",
    "control_arm",
    "decision_at",
    "implied_caveats",
    "dominant_options",
    "every_caveat",
    "floors_not_held",
    "margin_sensitivity",
    "report_caveats",
    "report_kind",
    "separated_by",
    "separating_contrasts",
    "shown_better",
    "StandingWord",
    "surviving_options",
    "unsettled_arms",
    "weighed_axes",
]

#: Judge tiers and calibration states that mean no human calibration stands behind the judge.
_UNCALIBRATED = frozenset({"undetermined", "insufficient"})

#: Where a caveat may land: report-wide, scoped to some comparisons, or method-level.
CaveatReach = Literal["report", "scoped", "method"]
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
        ``unsettled``, ``latency`` or ``unreported``.
    :ivar reach: ``report`` (the confidence is conditional on it), ``scoped`` (said beside a claim) or ``method``.
    """

    id: str
    kind: str
    reach: CaveatReach


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
    :return: Late criteria, the judge, confounds, apparatus confounds, short runs, unsettled aspects, latency, then
        unreported observations, each in the input's order.
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
    unsettled = unsettled_arms(report_input)
    if unsettled:
        called = set(_outcome(report_input, _Overrides())[1])
        for caveat_id, arms in unsettled.items():
            add(Caveat(caveat_id, "unsettled", "report" if called & set(arms) else "scoped"))
    if caveats.latency:
        add(Caveat("latency", "latency", "method"))
    for observation in caveats.unreported:
        add(Caveat(f"unreported:{observation}", "unreported", "method"))
    return list(found.values())


# --- implied caveats -----------------------------------------------------------------------------------------------

#: How many candidates make a pick the best of many (``best_of_n``): from three, the best of the candidates is a
#: selection among several, whose winner reads high by chance.
_MANY_OPTIONS = 3

#: The criterion kinds that decide which options survive or are shown no worse: a bar only warns.
_DECIDING = frozenset({"gate", "guardrail", "floor"})


@dataclass(frozen=True)
class ImpliedCaveat:
    """One caveat what the input holds implies (this module's Implied caveats), and what it touches.

    :ivar id: Its stable id.
    :ivar kind: The id's kind: ``judge_unstated``, ``judge_partly_uncalibrated``, ``no_control``,
        ``nothing_declared``, ``point_estimate`` or ``best_of_n``.
    :ivar reach: ``report`` (a call may rest on it) or ``scoped`` (said beside the claims it touches).
    :ivar matters_if: When it matters, in one sentence: the condition a report's ``material_if`` weighs.
    :ivar arms: The arms it touches, in the input's order; empty for every arm.
    :ivar measures: The measures and judged dimensions it touches, by key.
    :ivar at: For a point estimate: each surviving arm's figure that clears with no interval, as ``(arm, task, cell)``
        (the task or cell id where the criterion reads one there, else None).
    """

    id: str
    kind: str
    reach: CaveatReach
    matters_if: str
    arms: tuple[str, ...] = ()
    measures: tuple[str, ...] = ()
    at: tuple[tuple[str, str | None, str | None], ...] = ()


def _judge_standing(report_input: ReportInput, dimension: str) -> set[str]:
    """What the input states of how far one judged dimension's judge can be trusted: its tier and calibration state."""
    judged = report_input.judged.get(dimension)
    stated = [report_input.caveats.judge_tiers.get(dimension)]
    if judged is not None:
        stated += [judged.tier, judged.calibration]
    return {value for value in stated if value is not None}


def _judge_described(report_input: ReportInput, dimension: str) -> bool:
    """Whether the input describes one judged dimension's judge at all: its model, tier or calibration state."""
    judged = report_input.judged.get(dimension)
    return bool(_judge_standing(report_input, dimension)) or (judged is not None and judged.judge_model is not None)


def _judged_dimensions(report_input: ReportInput) -> list[str]:
    tiers = report_input.caveats.judge_tiers
    return [*report_input.judged, *(name for name in tiers if name not in report_input.judged)]


def _deciding_reach(report_input: ReportInput, measures: Sequence[str]) -> CaveatReach:
    """``report`` where a deciding criterion or the primary judged dimension reads one of the measures, else
    ``scoped``."""
    read = {criterion.measure for criterion in report_input.criteria if criterion.kind in _DECIDING}
    read.add(report_input.primary_judged or "")
    return "report" if read & set(measures) else "scoped"


def _clears(threshold: float, direction: str, value: float) -> bool:
    return value >= threshold if direction == "higher_is_better" else value <= threshold


def _point_estimates(report_input: ReportInput) -> Iterator[ImpliedCaveat]:
    """A ``point_estimate:<criterion>`` caveat for each gate or guardrail a surviving option clears at some place on a
    figure that holds no interval there: none stated, and none derived from its cells
    (:func:`~threetears.evals.analysis.report_margins.criterion_figures`)."""
    survivors = surviving_options(report_input)
    for criterion in report_input.criteria:
        if criterion.kind not in ("gate", "guardrail"):
            continue
        at = tuple(
            (arm, figure.task, figure.cell)
            for arm in survivors
            for figure in criterion_figures(report_input, criterion, arm)
            if figure.ci is None
            and figure.half_width is None
            and _clears(float(criterion.threshold), criterion.direction, figure.value)
        )
        if at:
            yield ImpliedCaveat(
                f"point_estimate:{criterion.id}",
                "point_estimate",
                "report",
                "it matters where a surviving option clears the line by little: no interval says whether the "
                "clearance is more than noise, so a clearance inside its noise reads as a pass",
                arms=tuple(dict.fromkeys(arm for arm, _task, _cell in at)),
                measures=(criterion.measure,),
                at=at,
            )


def _borderlines(report_input: ReportInput) -> Iterator[ImpliedCaveat]:
    """A ``borderline:<criterion>`` caveat for each gate or guardrail a surviving option clears at some place on its
    point estimate while the figure's interval there reaches the wrong side of the line: the gate check's borderline
    (:func:`~threetears.evals.analysis.report_gates.gate_findings`), which the decision counts as passing."""
    survivors = surviving_options(report_input)
    for criterion in report_input.criteria:
        if criterion.kind not in ("gate", "guardrail"):
            continue
        at = tuple(
            (arm, finding.task, finding.cell)
            for arm in survivors
            for finding in gate_findings(report_input, arm, criterion=criterion.id)
            if finding.status == "borderline"
        )
        if at:
            yield ImpliedCaveat(
                f"borderline:{criterion.id}",
                "borderline",
                "report",
                "it matters where the call rests on an option clearing this line: its figure clears it, but the "
                "interval reaches past it, so the evidence allows a true figure on the wrong side and a re-run of the "
                "same size can read short of the line",
                arms=tuple(dict.fromkeys(arm for arm, _task, _cell in at)),
                measures=(criterion.measure,),
                at=at,
            )


def implied_caveats(report_input: ReportInput) -> list[ImpliedCaveat]:
    """The caveats what the report input holds implies, beyond the ones it states (this module's Implied caveats).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The judge's, then no control, nothing declared, each gate cleared on a point estimate, each gate cleared
        on a borderline figure, then a pick among many; each once, in that order.
    :rtype: list[ImpliedCaveat]
    """
    found: list[ImpliedCaveat] = []
    dimensions = _judged_dimensions(report_input)
    unstated = tuple(dimension for dimension in dimensions if not _judge_described(report_input, dimension))
    if unstated:
        found.append(
            ImpliedCaveat(
                "judge_unstated",
                "judge_unstated",
                _deciding_reach(report_input, unstated),
                "it matters where a call rests on a judged figure: nothing says whose scores they are or how far they "
                "can be trusted, so a judge that is lenient or harsh on one arm moves the call",
                measures=unstated,
            )
        )
    stated = {caveat.id for caveat in report_caveats(report_input)}
    uncalibrated = tuple(
        dimension
        for dimension in dimensions
        if (standing := _judge_standing(report_input, dimension)) and standing <= _UNCALIBRATED
    )
    if uncalibrated and "judge_uncalibrated" not in stated:
        found.append(
            ImpliedCaveat(
                "judge_partly_uncalibrated",
                "judge_partly_uncalibrated",
                _deciding_reach(report_input, uncalibrated),
                "it matters where a call rests on one of these dimensions: no human ratings stand behind their judge",
                measures=uncalibrated,
            )
        )
    if len(report_input.arms) > 1 and control_arm(report_input) is None:
        found.append(
            ImpliedCaveat(
                "no_control",
                "no_control",
                "report",
                "it matters for any claim that one arm beats another: with no control nothing is read paired, so two "
                "arms compare only by their own figures and intervals",
            )
        )
    margins = any(
        aspect.margin is not None or any(reading.margin is not None for reading in aspect.readings)
        for aspect in report_input.aspects
    )
    deciding = any(criterion.kind in _DECIDING for criterion in report_input.criteria)
    if not deciding and not margins and not report_input.floors:
        found.append(
            ImpliedCaveat(
                "nothing_declared",
                "nothing_declared",
                "report",
                "it matters for any call: with no gate, floor or margin declared, nothing says what is good enough, "
                "so code's dominance ranks options on cost alone and the trade is the owner's to price",
            )
        )
    found += _point_estimates(report_input)
    found += _borderlines(report_input)
    candidates = [arm.id for arm in report_input.arms if arm.role == "challenger"]
    if len(candidates) >= _MANY_OPTIONS:
        found.append(
            ImpliedCaveat(
                "best_of_n",
                "best_of_n",
                "report",
                "it matters where the pick leads the next option, or clears a line, by little: the best of many "
                "candidates reads high by chance, the more so the more there are, so a re-run on fresh cases tends "
                "to read it lower",
                arms=tuple(candidates),
            )
        )
    return found


def every_caveat(report_input: ReportInput) -> list[Caveat | ImpliedCaveat]:
    """Every caveat code assigns on a report input: the ones it states, then the ones what it holds implies.

    The one set a caveat id is known by: the analyst's ``caveats_list`` pages it, and a report document's validity
    (:func:`~threetears.evals.kernel.report_doc.report_doc_validity`) reads its ids, so the two cannot disagree.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: :func:`report_caveats`, then :func:`implied_caveats`, each in its own order.
    :rtype: list[Caveat | ImpliedCaveat]
    """
    return [*report_caveats(report_input), *implied_caveats(report_input)]


# --- gates and survivors -------------------------------------------------------------------------------------------


#: The gate statuses an option survives: borderline counts as passing for the decision (GENERATOR §3).
_SURVIVING: frozenset[GateStatus] = frozenset({"pass", "borderline"})


def _passes(report_input: ReportInput, arm: str) -> bool:
    check: GateCheck = gate_check(report_input, arm)
    return check.status in _SURVIVING


def surviving_options(report_input: ReportInput) -> list[str]:
    """The arms that pass every gate and breach no guardrail (this module's Survivors).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The surviving arms' ids, in the input's arm order.
    :rtype: list[str]
    """
    breached = {
        str(guardrail.get("arm")) for guardrail in report_input.guardrails if guardrail.get("decision") == "breached"
    }
    return [arm.id for arm in report_input.arms if arm.id not in breached and _passes(report_input, arm.id)]


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
    report_input: ReportInput, arm: str, asked: Sequence[Aspect], overrides: _Overrides, *, reasons: bool = True
) -> list[str]:
    """One clause per floor not held on the asked aspects, and per aspect declaring none; with ``reasons``, a floor's
    clause ends with the input's own reason where it speaks for the status read."""
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
                reason = f" ({read.reason.rstrip('.')})" if reasons and read.reason and status == read.status else ""
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
    :ptype report_input: ReportInput
    :return: The candidates left unsettled, in the readings' order, by caveat id in the aspects' order; empty for none.
    :rtype: dict[str, tuple[str, ...]]
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


def separating_contrasts(report_input: ReportInput, arm: str) -> list[Contrast]:
    """The contrasts on an asked aspect that separate a candidate from the control: verdict ``improved``,
    ``regressed`` or ``equivalent``, each as its family's correction reads it (uncorrected where that is ``none``), in
    the input's order. How a question a measure
    answers (a latency aspect's p95) separates a candidate.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param arm: The candidate's id.
    :ptype arm: str
    :return: The separating contrasts; empty for none.
    :rtype: list[Contrast]
    """
    asked = {aspect.id for aspect in _asked(report_input)}
    return [
        contrast
        for contrast in report_input.contrasts
        if contrast.arm == arm and contrast.aspect in asked and contrast.verdict in SEPARATED_VERDICTS
    ]


def _reading_separates(report_input: ReportInput, arm: str, overrides: _Overrides) -> bool | None:
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


def separated_by(report_input: ReportInput, arm: str) -> frozenset[str]:
    """What separates a candidate from the control on the asked aspects, as declared: ``reading`` where an asked
    reading does (its interval clear of zero, or within the margin), ``contrast`` where a contrast on an asked aspect
    does (:func:`separating_contrasts`). Empty where nothing does. The one test :func:`allowed_modes` reads a change's
    separation by, and the report plan's spine too.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param arm: The candidate's id.
    :ptype arm: str
    :return: ``reading``, ``contrast``, both or neither.
    :rtype: frozenset[str]
    """
    found = {"contrast"} if separating_contrasts(report_input, arm) else set()
    if _reading_separates(report_input, arm, _Overrides()):
        found.add("reading")
    return frozenset(found)


def _separable(report_input: ReportInput, arm: str, overrides: _Overrides) -> bool | None:
    """Whether a candidate's asked evidence separates it: a separating contrast on an asked aspect
    (:func:`separating_contrasts`), or its asked readings (at the margins ``overrides`` sets). None where it has
    neither an asked reading with an interval nor a separating contrast."""
    if separating_contrasts(report_input, arm):
        return True
    return _reading_separates(report_input, arm, overrides)


# --- cost ----------------------------------------------------------------------------------------------------------


def _costs(report_input: ReportInput) -> dict[str, float]:
    """Each arm's cost per 1,000 of what it counts, as its ``cost_per_1k`` key reads it (this module's Dominance,
    cost)."""
    costs: dict[str, float] = {}
    for arm in report_input.arms:
        try:
            figure = resolve(f"cost_per_1k.{arm.id}", report_input)
        except UnresolvableReference:
            # NOSILENT: an arm whose cost no source states is an expected absence, not a failure: it has no cost, which
            # the dominance rule reads as such.
            continue
        if isinstance(figure.value, (int, float)):
            costs[arm.id] = float(figure.value)
    return costs


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
    planner's :func:`allowed_modes` reads the rule as it stands.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The axes declared, in that order; empty where none is.
    :rtype: tuple[DecisionAxis, ...]
    """
    found: list[DecisionAxis] = []
    if any(criterion.kind in ("gate", "guardrail") for criterion in report_input.criteria) or report_input.guardrails:
        found.append("gates")
    if control_arm(report_input) is not None:
        found.append("control")
    if _asked(report_input):
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

    :ivar modes: The allowed modes, as :func:`allowed_modes` reads them at those values.
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
    :ptype report_input: ReportInput
    :param margins: A margin per aspect id, in points.
    :ptype margins: Mapping[str, float] | None
    :param floors: A margin per floor key, in points.
    :ptype floors: Mapping[str, float] | None
    :return: The allowed modes and the dominant option at those values.
    :rtype: DecisionAt
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
    modes, dominant = _outcome(report_input, _Overrides(margins=margins, floors=floors))
    return DecisionAt(modes=modes, dominant=dominant)


#: How an arm stands on the asked aspects, as this module's Dominance reads it. In a change: ``shown_better`` (level 2),
#: ``control`` or ``shown_no_worse`` (level 1), ``not_shown_no_worse`` (level 0). In a choice, by the asked pair
#: readings: ``behind`` when one shows another arm ahead of it, else ``not_behind``.
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
    :ivar why: Why a change's candidate is not shown no worse: each floor not held, each aspect declaring none, and each
        asked reading not within its margin, in words that name each by its id and state no figure (a floor read's
        own reason, and its figures, are read by its keys). Empty otherwise.
    """

    arm: str
    standing: StandingWord
    level: int | None
    behind: tuple[str, ...]
    why: tuple[str, ...]


def _not_within(report_input: ReportInput, arm: str) -> list[str]:
    """Each asked reading of a candidate that declares a margin and is not within it, as a clause."""
    return [
        f"its reading on aspect {aspect.id!r} is not within the margin"
        for aspect in _asked(report_input)
        for reading in aspect.readings
        if reading.arm == arm
        and _margin(aspect, reading, _Overrides()) is not None
        and not _within(aspect, reading, _Overrides())
    ]


def asked_standing(report_input: ReportInput) -> dict[str, AskedStanding]:
    """Every arm's standing on the asked aspects: the quality axis dominance compares (this module's Dominance).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: Each arm's standing, by id, in the input's arm order.
    :rtype: dict[str, AskedStanding]
    """
    control = control_arm(report_input)
    asked = _asked(report_input)
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
        elif level == 2:
            found[arm] = AskedStanding(arm, "shown_better", level, (), ())
        elif level == 1:
            found[arm] = AskedStanding(arm, "shown_no_worse", level, (), ())
        else:
            floors = _floors_not_held(report_input, arm, asked, _Overrides(), reasons=False)
            why = [*floors, *_not_within(report_input, arm)]
            if not asked:
                why.append("no aspect is asked, so nothing shows it no worse")
            found[arm] = AskedStanding(arm, "not_shown_no_worse", level, (), tuple(why))
    return found


# --- immateriality -------------------------------------------------------------------------------------------------


def _probes(flips: Sequence[float]) -> list[float]:
    """Values covering every piece of ``[0, inf)`` that ``flips`` cut it into: zero, each flip, each midpoint, and one
    past the last."""
    points = sorted({0.0, *flips})
    middles = [(low + high) / 2 for low, high in zip(points, points[1:], strict=False)]
    return sorted({*points, *middles, points[-1] + 1.0})


def _holds(report_input: ReportInput, base: _Outcome, trials: Iterator[_Overrides]) -> bool:
    return all(_outcome(report_input, overrides) == base for overrides in trials)


@dataclass(frozen=True)
class CandidateMargin:
    """One reading against its margin: where it flips, and whether it is within the margin declared.

    :ivar arm: The reading's arm.
    :ivar aspect: The aspect it is read on.
    :ivar flip_margin: The margin it flips at (``-lo``); None where it states no interval.
    :ivar within: Whether it is within the declared margin.
    """

    arm: str
    aspect: str
    flip_margin: float | None
    within: bool


@dataclass(frozen=True)
class MarginSensitivity:
    """How the call rests on the margins (:func:`margin_sensitivity`): each reading's flip, and whether the call holds
    at every margin.

    :ivar candidates: Each reading on an aspect that declares a margin, in the input's order.
    :ivar holds: Whether the decision is the same at every margin from zero up, on each such aspect.
    """

    candidates: tuple[CandidateMargin, ...]
    holds: bool


def _margined(report_input: ReportInput) -> list[Aspect]:
    """The aspects that declare a margin, on the aspect or on one of its readings."""
    return [
        aspect
        for aspect in report_input.aspects
        if aspect.margin is not None or any(reading.margin is not None for reading in aspect.readings)
    ]


def margin_sensitivity(report_input: ReportInput) -> MarginSensitivity:
    """Each reading's flip margin and whether it is within the declared margin, and whether the call holds at every
    margin (this module's Immateriality, for a late margin). A late margin's materiality reads ``holds``; a page's
    explore part, each candidate.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: Each reading on an aspect that declares a margin, and whether the call holds at every margin; no
        candidates, and not holding, where no aspect declares one.
    :rtype: MarginSensitivity
    """
    candidates = tuple(
        CandidateMargin(reading.arm, aspect.id, flip_margin(reading), _within(aspect, reading, _Overrides()))
        for aspect in _margined(report_input)
        for reading in aspect.readings
    )
    return MarginSensitivity(candidates, _margin_holds(report_input, _outcome(report_input, _Overrides())))


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
