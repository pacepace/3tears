"""The caveats code assigns on a report input, as the page reads them: thin reads of the one caveat list
(:mod:`~threetears.evals.analysis.caveats`), each read off the input as it stands with the words the page says it in.

What code decides (the options that survive, which one dominates, the modes a report may take) is the numbers side's
(:mod:`~threetears.evals.analysis.answer`), and every name of it is read here from there for the pages that import it
from this module.

A report states its confidence against the caveats :func:`report_caveats` lists, and never mentions a caveat
:func:`caveat_materiality` proved immaterial; code renders that one's checked line. The report document's invariants
and the editor's facts
(:mod:`~threetears.evals.analysis.report.doc_page.facts`) read the same decisions.

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
``nothing_declared``                      no gate, guardrail or floor criterion, no margin, no floor read and no aspect
                                          a question asks of a control (one with no margin declares shown better
                                          than the control as good enough): nothing says what is good enough, so code
                                          ranks options on cost alone (a bar only warns)
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

**Scope and immateriality** are the list's: each entry's reach (:mod:`~threetears.evals.analysis.caveats`) and
whether code proved a late criterion immaterial (:func:`~threetears.evals.analysis.answer.late_immaterial`). Here a
caveat proved immaterial gets the one checked line code renders for it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from threetears.evals.analysis.answer import (
    AskedStanding,
    DecisionAt,
    StandingWord,
    asked_standing,
    control_arm,
    decision_at,
    dominant_options,
    floors_not_held,
    separating_contrasts,
    shown_better,
    surviving_options,
    unsettled_arms,
    weighed_axes,
)
from threetears.evals.analysis.caveats import (
    cleared_borderline,
    from_legacy,
    implied,
    cleared_on_point_estimate,
    late_id,
    late_subject,
    uncalibrated_judges,
    unstated_judges,
)
from threetears.evals.analysis.report_input import CaveatEntry, ReportInput
from threetears.evals.schema.reading_states import CaveatReach

__all__ = [
    "AskedStanding",
    "Caveat",
    "CaveatReach",
    "DecisionAt",
    "ImpliedCaveat",
    "Materiality",
    "StandingWord",
    "asked_standing",
    "caveat_materiality",
    "control_arm",
    "decision_at",
    "dominant_options",
    "every_caveat",
    "floors_not_held",
    "implied_caveats",
    "late_subject",
    "report_caveats",
    "separating_contrasts",
    "shown_better",
    "surviving_options",
    "unsettled_arms",
    "weighed_axes",
]

#: The checked line code renders for a late criterion it proved immaterial, by the criterion's kind.
_CHECKED = {
    "margin": "The late margin was checked: the call holds at every margin.",
    "floor": "The late floor was checked: the call holds at every margin the floor could take.",
    "bar": "The late bar was checked: a bar only warns, so the call holds at any bar.",
}

#: When each implied caveat matters, in one sentence: the condition a report's ``material_if`` weighs, by kind.
_MATTERS_IF = {
    "judge_unstated": (
        "it matters where a call rests on a judged figure: nothing says whose scores they are or how far they can be "
        "trusted, so a judge that is lenient or harsh on one arm moves the call"
    ),
    "judge_partly_uncalibrated": (
        "it matters where a call rests on one of these dimensions: no human ratings stand behind their judge"
    ),
    "no_control": (
        "it matters for any claim that one arm beats another: with no control nothing is read paired, so two arms "
        "compare only by their own figures and intervals"
    ),
    "nothing_declared": (
        "it matters for any call: with no gate, floor or margin declared, nothing says what is good enough, so "
        "code's dominance ranks options on cost alone and the trade is the owner's to price"
    ),
    "point_estimate": (
        "it matters where a surviving option clears the line by little: no interval says whether the clearance is "
        "more than noise, so a clearance inside its noise reads as a pass"
    ),
    "borderline": (
        "it matters where the call rests on an option clearing this line: its figure clears it, but the interval "
        "reaches past it, so the evidence allows a true figure on the wrong side and a re-run of the same size can "
        "read short of the line"
    ),
    "best_of_n": (
        "it matters where the pick leads the next option, or clears a line, by little: the best of many candidates "
        "reads high by chance, the more so the more there are, so a re-run on fresh cases tends to read it lower"
    ),
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
class ImpliedCaveat:
    """One caveat what the input holds implies (this module's Implied caveats), and what it touches.

    :ivar id: Its stable id.
    :ivar kind: The id's kind: ``judge_unstated``, ``judge_partly_uncalibrated``, ``no_control``,
        ``nothing_declared``, ``point_estimate``, ``borderline`` or ``best_of_n``.
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


def report_caveats(report_input: ReportInput) -> list[Caveat]:
    """Every caveat the report input states, each once, by id, with where it may land: the list's stated
    raiser (:func:`~threetears.evals.analysis.caveats.from_legacy`), read off the input as it stands, so a copy of an
    input with a part changed reads its own.

    :param report_input: The report input.
    :return: Late criteria, the judge, confounds, apparatus confounds, short runs, unsettled aspects, latency, then
        unreported observations, each in the input's order.
    """
    return [Caveat(entry.id, entry.kind, entry.reach) for entry in from_legacy(report_input)]


def _implied(report_input: ReportInput, entry: CaveatEntry) -> ImpliedCaveat:
    """One implied entry of the list as the page reads it: its sentence, and what it touches."""
    measures: tuple[str, ...] = ()
    at: tuple[tuple[str, str | None, str | None], ...] = ()
    if entry.kind == "judge_unstated":
        measures = unstated_judges(report_input)
    elif entry.kind == "judge_partly_uncalibrated":
        measures = uncalibrated_judges(report_input)
    else:
        for criterion in report_input.criteria:
            if entry.id == f"point_estimate:{criterion.id}":
                measures, at = (criterion.measure,), cleared_on_point_estimate(report_input, criterion.id)
            elif entry.id == f"borderline:{criterion.id}":
                measures, at = (criterion.measure,), cleared_borderline(report_input, criterion.id)
    return ImpliedCaveat(
        entry.id, entry.kind, entry.reach, _MATTERS_IF[entry.kind], tuple(entry.arms), measures=measures, at=at
    )


def implied_caveats(report_input: ReportInput) -> list[ImpliedCaveat]:
    """The caveats what the report input holds implies, beyond the ones it states: the list's
    implied raiser (:func:`~threetears.evals.analysis.caveats.implied`), read off the input as it stands.

    :param report_input: The report input.
    :return: The judge's, then no control, nothing declared, each gate cleared on a point estimate, each gate cleared
        on a borderline figure, then a pick among many; each once, in that order.
    """
    return [_implied(report_input, entry) for entry in implied(report_input, from_legacy(report_input))]


def every_caveat(report_input: ReportInput) -> list[Caveat | ImpliedCaveat]:
    """Every caveat code assigns on a report input: the ones it states, then the ones what it holds implies.

    The one set a caveat id is known by: the analyst's ``caveats_list`` pages it, and a report document's validity
    (:func:`~threetears.evals.kernel.report_doc.report_doc_validity`) reads its ids, so the two cannot disagree.

    :param report_input: The report input.
    :return: :func:`report_caveats`, then :func:`implied_caveats`, each in its own order.
    """
    return [*report_caveats(report_input), *implied_caveats(report_input)]


def _lates(report_input: ReportInput) -> Iterator[tuple[str, str]]:
    for late in report_input.caveats.late_criteria:
        yield late_id(late), late.criterion


def caveat_materiality(report_input: ReportInput) -> dict[str, Materiality]:
    """For every stated caveat, whether code proved it immaterial, with the checked line it renders: the list's
    ``immaterial``, worded.

    :param report_input: The report input.
    :return: Each caveat's materiality, by id, in :func:`report_caveats`' order.
    """
    kinds = dict(_lates(report_input))
    found: dict[str, Materiality] = {}
    for entry in from_legacy(report_input):
        if entry.immaterial is True and entry.id in kinds:
            found[entry.id] = Materiality(entry.id, True, _CHECKED[kinds[entry.id]])
        else:
            found[entry.id] = Materiality(entry.id, False, None)
    return found
