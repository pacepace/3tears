"""Each trial and how it ended, and each arm's census of them — the countable half of a report.

A **trial** is one case run once on one arm: one stored result. It ends in exactly one of three states, decided here
once for every reader:

- **lost** — the test setup failed, so it says nothing about the candidate: the harness faulted it, the judge could
  not tell on a dimension it needs, or the judge's reply was cut off at its output cap
  (:func:`~threetears.evals.kernel.result_condition.trial_exclusion` names which);
- **failed** — the candidate failed: its own model or budget failed it
  (:func:`~threetears.evals.kernel.result_condition.candidate_failure_cause`), a check it was graded on failed (a goal
  check, or a pass/fail scorer on the wrong side), or a judged dimension scored under the bar;
- **passed** — none of that. A trial that carried no pass criterion at all (no check, no scorer, no judge) passes
  because nothing could fail it, and the census counts those apart (:attr:`ArmScoreboard.no_criterion`).

A judge that could not tell on one dimension does not lose a trial that a failed check or a sub-bar dimension has
already decided: that trial is the fail it is, as pass^k reads it. That is the census's one departure from the
bundle's ``exclusions`` (counted per arm and rig by the assembler), which count every
trial ``trial_exclusion`` leaves out of the measures needing every dimension, that one included; every other cause
(``infra``, ``truncated``) loses the same trials in both, since both read the one classification.

The census (:class:`ArmScoreboard`) is a count, never a score: it says how many trials each arm ran, lost, could
use, passed and failed. The rows (:class:`TrialRow`) carry what a reader needs to see why — the failed checks, the
judge's scores and reasons, what the case asked and the tail of what the judge was handed — each capped by
:mod:`~threetears.evals.analysis.bundle.caps`, and the rows themselves capped with the lost and failed kept first.
Part of the bundle assembler (:mod:`threetears.evals.analysis.bundle.assemble`).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from threetears.evals.analysis.bundle.caps import (
    MAX_ERROR_CHARS,
    MAX_EXCERPT_CHARS,
    MAX_INPUT_CHARS,
    MAX_LABEL_INPUT_CHARS,
    MAX_REASON_CHARS,
    MAX_TRIALS,
    _capped,
    capped_text,
    tail_text,
)
from threetears.evals.analysis.bundle.schema import (
    ArmScoreboard,
    FailCause,
    TrialCheck,
    TrialJudgement,
    TrialRow,
    TrialState,
)
from threetears.evals.kernel.metrics import MATCH_MEASURE, MetricDescriptor, goal_check_measure
from threetears.evals.kernel.result_condition import (
    candidate_failure_cause,
    delivered_a_turn,
    trial_exclusion,
)
from threetears.evals.kernel.scoring import capability_scores, has_pass_criterion
from threetears.evals.kernel.usage_capture import count_substituted_deliveries, production_replicating_cost
from threetears.evals.schema.models import OUTCOME_DIM_ID, TRANSCRIPT_DIM_ID

if TYPE_CHECKING:  # runtime models — TYPE_CHECKING-only to keep the runtime import graph minimal.
    from threetears.evals.schema.models import EvalResult, EvalTrace, RubricScore


#: How a reserved judge axis is named on a trial, where a reader sees it: never by its stored id.
_AXIS_NAMES: dict[str, str] = {TRANSCRIPT_DIM_ID: "transcript", OUTCOME_DIM_ID: "outcome"}

#: How much worse a lost or failed trial is than a passed one, for keeping trials under the cap: lost first.
_KEEP_WEIGHT: dict[TrialState, int] = {"lost": 2, "failed": 1, "passed": 0}

#: A case id ending in a number, as a generated ``{template}-{index}`` id does.
_NUMBERED_ID = re.compile(r"(.*)-(\d+)")


def trial_order(case_id: str, arm_position: int, repeat: int) -> tuple[str, int, str, int, int]:
    """A trial's place in the order the page's Trials table lists them and the console lists its misses: case by
    case, then arm, then repeat.

    A case id ending in ``-<number>`` sorts by that number under its prefix, so a generated ``{template}-2``
    precedes ``{template}-10``; any other id sorts as its text.

    :param case_id: The trial's stored test case id.
    :ptype case_id: str
    :param arm_position: Its arm's place among the arms as the bundle lists them.
    :ptype arm_position: int
    :param repeat: Which repeat of the case it is.
    :ptype repeat: int
    :return: The sort key.
    :rtype: tuple[str, int, str, int, int]
    """
    numbered = _NUMBERED_ID.fullmatch(case_id)
    if numbered is None:
        return (case_id, -1, case_id, arm_position, repeat)
    return (numbered.group(1), int(numbered.group(2)), case_id, arm_position, repeat)


@dataclass(frozen=True)
class TrialReading:
    """How one trial ended: its state, why, and the checks it failed."""

    state: TrialState
    cause: str | None
    failed_checks: tuple[TrialCheck, ...] = ()
    no_criterion: bool = False


def read_trial(result: EvalResult, *, rubric_threshold: int, catalog: Mapping[str, MetricDescriptor]) -> TrialReading:
    """Decide how one trial ended: lost, failed or passed, and why.

    :param result: The stored result.
    :ptype result: EvalResult
    :param rubric_threshold: The 1–5 level a judged dimension must reach to pass, as pass^k reads it.
    :ptype rubric_threshold: int
    :param catalog: The bundle's measure catalog, which names each check and says which scorers are pass/fail.
    :ptype catalog: Mapping[str, MetricDescriptor]
    :return: The trial's reading.
    :rtype: TrialReading
    """
    checks = _failed_checks(result, catalog)
    below = [score for score in capability_scores(result) if not score.clears(rubric_threshold)]
    decided: FailCause | None = "check" if checks else "judged" if below else None
    exclusion = trial_exclusion(result)
    if exclusion is not None and not (exclusion == "judge_cannot_tell" and decided is not None):
        # A judge that could not tell on one dimension cannot rescue a trial a failed check or a sub-bar
        # dimension has already decided; every other cause loses the trial, whatever its scores say.
        return TrialReading(state="lost", cause=exclusion)
    if (cause := candidate_failure_cause(result)) is not None:
        return TrialReading(state="failed", cause=cause, failed_checks=tuple(checks))
    if decided is not None:
        return TrialReading(state="failed", cause=decided, failed_checks=tuple(checks))
    return TrialReading(state="passed", cause=None, no_criterion=not _has_criterion(result, catalog))


def _failed_checks(result: EvalResult, catalog: Mapping[str, MetricDescriptor]) -> list[TrialCheck]:
    """Every check the trial failed: each goal check, then a wrong label and each check measure on its wrong side
    (:attr:`~threetears.evals.kernel.metrics.MetricDescriptor.is_check`: a boolean, or a scorer graded 0 or 1)."""
    failed = [
        TrialCheck(name=_reader_name(goal_check_measure(check.expression), catalog), expression=check.expression)
        for check in result.goal_state_outcomes
        if not check.passed
    ]
    for name, value in sorted(result.host_measures.items()):
        descriptor = catalog.get(name)
        if name == MATCH_MEASURE:
            if value is False:
                failed.append(TrialCheck(name=_reader_name(name, catalog), expression=f"{name} == expected label"))
            continue
        if descriptor is not None and descriptor.check_failed(value):
            # A pass/fail scorer lands 0 or 1: named by the side it fell on, as a boolean's is.
            fell = value if isinstance(value, bool) else not descriptor.higher_is_better
            failed.append(TrialCheck(name=_reader_name(name, catalog), expression=f"{name} is {fell}"))
    return failed


def _has_criterion(result: EvalResult, catalog: Mapping[str, MetricDescriptor]) -> bool:
    """Whether anything could have failed the trial: a goal check, a judge, a label, or a pass/fail scorer."""
    if has_pass_criterion(result) or MATCH_MEASURE in result.host_measures:
        return True
    return any((descriptor := catalog.get(name)) is not None and descriptor.is_check for name in result.host_measures)


def _reader_name(name: str, catalog: Mapping[str, MetricDescriptor]) -> str:
    """A measure by the name a reader knows it by: its declared reader name, else its key."""
    descriptor = catalog.get(name)
    return (descriptor.reader_name if descriptor is not None else None) or name


@dataclass(frozen=True)
class TrialsAndCensus:
    """What :func:`trials_and_census` reads off the results: the bundle fields it fills."""

    scoreboard: list[ArmScoreboard]
    trials: list[TrialRow]
    trials_omitted: int


def trials_and_census(
    results: Sequence[EvalResult],
    *,
    arm_order: Sequence[str],
    strata: Mapping[str, str | None],
    criteria: Mapping[str, str],
    rubric_threshold: int,
    catalog: Mapping[str, MetricDescriptor],
    load_trace: Callable[[EvalResult], EvalTrace | None],
    names: Mapping[str, str] | None = None,
) -> TrialsAndCensus:
    """Every trial's row and every arm's census, over the campaign's results.

    :param results: Every result of the resolved member runs, after the bundle's own rewrites (a classifier's failure
        read as a miss).
    :ptype results: Sequence[EvalResult]
    :param arm_order: The arms' variant keys in the order the bundle lists them; an arm absent here is listed after,
        by key.
    :ptype arm_order: Sequence[str]
    :param strata: Each test case's declared stratum, by case id.
    :ptype strata: Mapping[str, str | None]
    :param names: The name each test case's author gave it, by case id, for the cases that have one.
    :ptype names: Mapping[str, str] | None
    :param criteria: Each judged dimension's rubric criterion as the judge was asked it, by dimension name.
    :ptype criteria: Mapping[str, str]
    :param rubric_threshold: The 1–5 level a judged dimension must reach to pass.
    :ptype rubric_threshold: int
    :param catalog: The bundle's measure catalog.
    :ptype catalog: Mapping[str, MetricDescriptor]
    :param load_trace: Reads one result's stored trace, or None; called only for a result that has one and is kept.
    :ptype load_trace: Callable[[EvalResult], EvalTrace | None]
    :return: The census, the kept rows and how many were left out. The losses by cause and the differential-loss flag
        are the bundle's ``exclusions`` and ``differential_exclusions``, which the assembler counts off the same
        :func:`~threetears.evals.kernel.result_condition.trial_exclusion`, never counted again here.
    :rtype: TrialsAndCensus
    """
    readings = {result.id: read_trial(result, rubric_threshold=rubric_threshold, catalog=catalog) for result in results}
    position = {variant: index for index, variant in enumerate(arm_order)}
    arms = sorted({result.variant_key for result in results}, key=lambda key: (position.get(key, len(position)), key))
    by_arm = {arm: [result for result in results if result.variant_key == arm] for arm in arms}
    scoreboard = [_census(arm, by_arm[arm], readings) for arm in arms]
    ordered = sorted(
        results,
        key=lambda result: (
            trial_order(result.test_case_id, position.get(result.variant_key, len(position)), result.k_iteration),
            result.eval_run_id,
            result.id,
        ),
    )
    kept, omitted = _capped(ordered, MAX_TRIALS, weight=lambda result: _KEEP_WEIGHT[readings[result.id].state])
    traces = {result.id: load_trace(result) if result.has_trace else None for result in kept}
    labels = _case_labels(kept, strata, traces, names or {})
    rows = [
        _row(f"T{index}", result, readings[result.id], labels[result.test_case_id], strata, criteria, traces[result.id])
        for index, result in enumerate(kept, start=1)
    ]
    return TrialsAndCensus(
        scoreboard=scoreboard,
        trials=rows,
        trials_omitted=omitted,
    )


def _census(arm: str, results: Sequence[EvalResult], readings: Mapping[str, TrialReading]) -> ArmScoreboard:
    """One arm's census over its trials, with what they cost."""
    states = [readings[result.id] for result in results]
    lost = Counter(reading.cause or "unknown" for reading in states if reading.state == "lost")
    failed = Counter(reading.cause or "unknown" for reading in states if reading.state == "failed")
    usable = [result for result, reading in zip(results, states, strict=True) if reading.state != "lost"]
    priced = [result.cost_usd for result in results if result.cost_usd is not None]
    replies = [
        cost
        for result in results
        if delivered_a_turn(result)
        and (
            cost := production_replicating_cost(
                result.usage, substituted_deliveries=count_substituted_deliveries(result)
            )
        )
        is not None
    ]
    return ArmScoreboard(
        variant_key=arm,
        ran=len(results),
        lost=dict(sorted(lost.items(), key=lambda item: (-item[1], item[0]))),
        usable=len(usable),
        passed=sum(1 for reading in states if reading.state == "passed"),
        failed=dict(sorted(failed.items(), key=lambda item: (-item[1], item[0]))),
        cases=len({result.test_case_id for result in results}),
        usable_cases=len({result.test_case_id for result in usable}),
        no_criterion=sum(1 for reading in states if reading.state == "passed" and reading.no_criterion),
        test_cost_usd=sum(priced) if priced else None,
        n_cost_unpriced=len(results) - len(priced),
        reply_cost_usd=sum(replies) / len(replies) if replies else None,
    )


def _case_labels(
    kept: Sequence[EvalResult],
    strata: Mapping[str, str | None],
    traces: Mapping[str, EvalTrace | None],
    names: Mapping[str, str],
) -> dict[str, str]:
    """Each kept case's label: the name its author gave it; else its stratum, then the first input its first kept
    trial's transcript recorded, else its id.

    A given name is the one label a person chose, and the report names the case by it. The rest is derived rather
    than declared: a generated test case carries no human label, and the first thing it asked, as the judge was shown
    it, is evidence of what it is, where a guessed name would not be.
    """
    labels: dict[str, str] = {}
    for result in kept:
        if result.test_case_id in labels:
            continue
        if (given := names.get(result.test_case_id)) is not None:
            labels[result.test_case_id] = given
            continue
        trace = traces.get(result.id)
        asked = trace.judge_evidence.case_material if trace is not None and trace.judge_evidence is not None else ""
        words = " ".join(asked.split())
        lead = capped_text(words, MAX_LABEL_INPUT_CHARS) if words else result.test_case_id
        stratum = strata.get(result.test_case_id)
        labels[result.test_case_id] = f"{stratum}: {lead}" if stratum else lead
    return labels


def _row(
    trial_id: str,
    result: EvalResult,
    reading: TrialReading,
    label: str,
    strata: Mapping[str, str | None],
    criteria: Mapping[str, str],
    trace: EvalTrace | None,
) -> TrialRow:
    """One trial's row, every text in it capped."""
    evidence = trace.judge_evidence if trace is not None else None
    scores: list[RubricScore] = [
        *result.rubric_scores,
        *(score for score in (result.transcript_score, result.outcome_score) if score is not None),
    ]
    judge = [
        TrialJudgement(
            dim=_AXIS_NAMES.get(score.dim, score.dim),
            criterion=criteria.get(score.dim),
            score=score.score,
            scale=score.scale,
            reason=capped_text(" ".join(score.reasoning.split()), MAX_REASON_CHARS),
        )
        for score in scores
    ]
    judge += [
        TrialJudgement(
            dim=_AXIS_NAMES.get(dim, dim),
            criterion=criteria.get(dim),
            score=None,
            scale="ordinal",
            reason=capped_text("could not tell: " + " ".join(reason.split()), MAX_REASON_CHARS),
        )
        for dim, reason in sorted(result.judge_cannot_tell.items())
    ]
    errors = [
        f"{who}: {error}"
        for who, error in (
            ("candidate", result.candidate_error),
            ("test setup", result.infra_error),
            ("judge", result.judge_error),
            ("runner", result.runner_error),
        )
        if error
    ]
    return TrialRow(
        trial_id=trial_id,
        case_id=result.test_case_id,
        case_label=label,
        stratum=strata.get(result.test_case_id),
        arm=result.variant_key,
        repeat=result.k_iteration,
        state=reading.state,
        cause=reading.cause,
        failed_checks=list(reading.failed_checks),
        judged={_AXIS_NAMES.get(score.dim, score.dim): score.score for score in scores},
        judge=judge,
        first_input=capped_text(evidence.case_material, MAX_INPUT_CHARS) if evidence is not None else "",
        excerpt=tail_text(evidence.artifact, MAX_EXCERPT_CHARS) if evidence is not None else "",
        error_words=capped_text("; ".join(errors), MAX_ERROR_CHARS) if errors and reading.state != "passed" else None,
        result_id=result.id,
    )


__all__: list[str] = []
