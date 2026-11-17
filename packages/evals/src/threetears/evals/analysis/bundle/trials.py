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
    capped,
    capped_text,
    tail_text,
)
from threetears.evals.analysis.depth import depth_of
from threetears.evals.analysis.bundle.schema import (
    AnalysisContextBundle,
    ArmPricing,
    ArmScoreboard,
    FailCause,
    RoleUnanswered,
    TrialCheck,
    TrialJudgement,
    TrialRow,
    TrialState,
)
from threetears.evals.kernel.candidate_kind import QUICK_PATH_FILER, QUICK_PATH_KINDS
from threetears.evals.kernel.errors import ConflictError
from threetears.evals.kernel.metrics import MetricDescriptor
from threetears.evals.kernel.result_condition import (
    candidate_failure_cause,
    delivered_a_turn,
    trial_exclusion,
)
from threetears.evals.kernel.scoring import capability_scores, failed_checks, has_pass_criterion
from threetears.evals.kernel.usage_capture import (
    count_substituted_deliveries,
    production_replicating_cost,
    production_rows,
)
from threetears.evals.kernel.provider import sum_optional_tokens
from threetears.evals.schema.models import USAGE_ROLES, rubric_dim_scale

if TYPE_CHECKING:  # runtime models — TYPE_CHECKING-only to keep the runtime import graph minimal.
    from threetears.evals.kernel.result_reading import ResultView, UsageView
    from threetears.evals.schema.models import EvalResult, EvalRun, JudgeEvidence, RubricScale, RubricScore


#: How much worse a lost or failed trial is than a passed one, for keeping trials under the cap: lost first.
_KEEP_WEIGHT: dict[TrialState, int] = {"lost": 2, "failed": 1, "passed": 0}

#: What a caller does when a kept trial's result is gone by the time it is read whole: the runs changed under the read.
_STALE_TRIALS_REMEDY = "a member run changed while the report was read; read it again"

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


#: The roles a trial's recorded errors are written under, in the order :attr:`TrialRow.error_words` lists them.
ERROR_ROLES: tuple[str, ...] = ("candidate", "test setup", "judge", "runner")

_ROLE_BREAK = re.compile("; (?=(?:" + "|".join(map(re.escape, ERROR_ROLES)) + "): )")


def first_recorded_error(error_words: str) -> tuple[str | None, str]:
    """The first error a trial's :attr:`~threetears.evals.analysis.bundle.schema.TrialRow.error_words` records: the
    role it was written under (one of :data:`ERROR_ROLES`, or None where it names none) and its own words, the errors
    after it (often the same words under another role) left out. Split only where a role opens, so an error that
    itself holds ``; `` is kept whole.

    :param error_words: The trial's recorded errors, each under its role, as a trial row joins them.
    :ptype error_words: str
    :return: The first error's role and words.
    :rtype: tuple[str | None, str]
    """
    first = _ROLE_BREAK.split(error_words, maxsplit=1)[0]
    for role in ERROR_ROLES:
        if first.startswith(f"{role}: "):
            return role, first.removeprefix(f"{role}: ")
    return None, first


#: The cause of a trial the candidate failed without answering: the runner raised, or its model call failed.
_UNANSWERED: FailCause = "model_failed"


@dataclass(frozen=True)
class PassCount:
    """One arm's pass count over every trial that ran, as the engine's census holds it: the one count a first line, a
    page's lead and a listing state, with the trials nothing graded beside it.

    The pass rate, its interval and every verdict stay over the graded trials (``usable``); this says how many ran
    beside them, so no reader states a share over the graded trials alone as if they were every trial.

    :ivar passed: Graded trials that passed.
    :ivar graded: Trials graded (the census's ``usable``).
    :ivar ran: Trials that ran.
    :ivar cases: Distinct cases that ran.
    :ivar graded_cases: Distinct cases with a graded trial.
    :ivar ungraded_cases: The cases of the trials not graded that the bundle holds, in the trials' order (a bundle
        that holds only some trials keeps the lost ones first); ``ran - graded`` counts them all.
    :ivar ungraded_role: The role the first held ungraded trial's error was recorded under (``judge``, ``test setup``,
        ...); None where none is held, or it names none.
    :ivar ungraded_error: That error, in its own words; None where none is held.
    :ivar ungraded_causes: The census's causes of the trials not graded (``infra``, ``judge_truncated``, ...), the
        most common first: what is said where no trial's own error is held.
    :ivar candidate_failed: Graded trials the candidate failed by failing to answer (the runner raised, its model call
        failed): each counts against it, and no grader had an answer to read on it.
    :ivar candidate_error: The first such trial's recorded error, in its own words; None where none is held.
    """

    passed: int
    graded: int
    ran: int
    cases: int
    graded_cases: int
    ungraded_cases: tuple[str, ...]
    ungraded_role: str | None
    ungraded_error: str | None
    ungraded_causes: tuple[str, ...] = ()
    candidate_failed: int = 0
    candidate_error: str | None = None

    @property
    def ungraded(self) -> int:
        """Trials that ran and were not graded: lost to the test setup (a grader that raised, a declined call)."""
        return self.ran - self.graded

    @property
    def none_answered(self) -> bool:
        """Whether the candidate failed every graded trial without answering, so no grader read an answer on any."""
        return self.graded > 0 and self.candidate_failed == self.graded


def pass_count(bundle: AnalysisContextBundle, board: ArmScoreboard) -> PassCount:
    """One arm's :class:`PassCount`, read off its census (``board``) and its trials in ``bundle``.

    :param bundle: The aspect's evidence.
    :ptype bundle: AnalysisContextBundle
    :param board: The arm's scoreboard row in it.
    :ptype board: ArmScoreboard
    :return: The count.
    :rtype: PassCount
    """
    # Lost trials are kept first where a bundle holds only some trials, so the held ones are the lost ones it has.
    lost = [trial for trial in bundle.trials if trial.arm == board.variant_key and trial.state == "lost"]
    recorded = next((trial.error_words for trial in lost if trial.error_words), None)
    role, words = first_recorded_error(recorded) if recorded is not None else (None, None)
    unanswered = next(
        (
            trial.error_words
            for trial in bundle.trials
            if trial.arm == board.variant_key
            and trial.state == "failed"
            and trial.cause == _UNANSWERED
            and trial.error_words
        ),
        None,
    )
    return PassCount(
        passed=board.passed,
        graded=board.usable,
        ran=board.ran,
        cases=board.cases,
        graded_cases=board.usable_cases,
        ungraded_cases=tuple(dict.fromkeys(trial.case_id for trial in lost)),
        ungraded_role=role,
        ungraded_error=words,
        ungraded_causes=tuple(sorted(board.lost, key=lambda cause: (-board.lost[cause], cause))),
        candidate_failed=board.failed.get(_UNANSWERED, 0),
        candidate_error=first_recorded_error(unanswered)[1] if unanswered is not None else None,
    )


@dataclass(frozen=True)
class TrialReading:
    """How one trial ended: its state, why, and the checks it failed."""

    state: TrialState
    cause: str | None
    failed_checks: tuple[TrialCheck, ...] = ()
    no_criterion: bool = False


def read_trial(result: ResultView, *, rubric_threshold: int, catalog: Mapping[str, MetricDescriptor]) -> TrialReading:
    """Decide how one trial ended: lost, failed or passed, and why.

    :param result: The stored result.
    :ptype result: ResultView
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
    if exclusion is not None and not (exclusion in ("judge_cannot_tell", "cost_cap") and decided is not None):
        # A judge that could not tell on one dimension, or was kept from being asked by the cost cap, cannot rescue a
        # trial a failed check or a sub-bar dimension has already decided; every other cause loses the trial,
        # whatever its scores say.
        return TrialReading(state="lost", cause=exclusion)
    if (cause := candidate_failure_cause(result)) is not None:
        return TrialReading(state="failed", cause=cause, failed_checks=tuple(checks))
    if decided is not None:
        return TrialReading(state="failed", cause=decided, failed_checks=tuple(checks))
    return TrialReading(state="passed", cause=None, no_criterion=not has_pass_criterion(result, catalog=catalog))


def _failed_checks(result: ResultView, catalog: Mapping[str, MetricDescriptor]) -> list[TrialCheck]:
    """Every check the trial failed, as pass^k reads them (:func:`~threetears.evals.kernel.scoring.failed_checks`),
    each by its measure's key: a page names it when it is rendered, never the bundle."""
    return [TrialCheck(name=check.measure, expression=check.expression) for check in failed_checks(result, catalog)]


@dataclass(frozen=True)
class TrialsAndCensus:
    """What :func:`trials_and_census` reads off the results: the bundle fields it fills."""

    scoreboard: list[ArmScoreboard]
    trials: list[TrialRow]
    trials_omitted: int
    #: Whether the member runs' case lists differ, so a case's place is where it first stands in their union, in run
    #: order, and no one run's list need hold it at that line (``AnalysisContextBundle.case_lists_differ``).
    case_lists_differ: bool = False


def trials_and_census(
    results: Sequence[ResultView],
    *,
    arm_order: Sequence[str],
    strata: Mapping[str, str | None],
    criteria: Mapping[str, str],
    rubric_threshold: int,
    catalog: Mapping[str, MetricDescriptor],
    load_judge_evidence: Callable[[Sequence[str]], Mapping[str, JudgeEvidence]],
    load_results: Callable[[Sequence[str]], Sequence[EvalResult]],
    rubric_scales: Mapping[str, Mapping[str, RubricScale]],
    names: Mapping[str, str] | None = None,
    case_lists: Sequence[Sequence[str]] = (),
) -> TrialsAndCensus:
    """Every trial's row and every arm's census, over the campaign's results.

    :param results: Every result of the resolved member runs, after the bundle's own rewrites (a classifier's failure
        read as a miss), each as the readers read it: its reading or the result. How each trial ended, the census and
        which trials are kept are read off these.
    :ptype results: Sequence[ResultView]
    :param arm_order: The arms' variant keys in the order the bundle lists them; an arm absent here is listed after,
        by key.
    :ptype arm_order: Sequence[str]
    :param strata: Each test case's declared stratum, by case id.
    :ptype strata: Mapping[str, str | None]
    :param names: The name each test case's author gave it, by case id, for the cases that have one.
    :ptype names: Mapping[str, str] | None
    :param case_lists: The member runs' frozen case lists (``EvalRun.test_case_ids``), in the bundle's run order: each
        case's place is where it first stands in them, from 1 (:func:`case_places`), carried on its rows
        (:attr:`~threetears.evals.analysis.bundle.schema.TrialRow.case_place`), and whether they differ is
        ``case_lists_differ``. Empty carries no place.
    :ptype case_lists: Sequence[Sequence[str]]
    :param criteria: Each judged dimension's rubric criterion as the judge was asked it, by dimension name.
    :ptype criteria: Mapping[str, str]
    :param rubric_threshold: The 1–5 level a judged dimension must reach to pass.
    :ptype rubric_threshold: int
    :param catalog: The bundle's measure catalog.
    :ptype catalog: Mapping[str, MetricDescriptor]
    :param load_judge_evidence: Reads what each named result's judge was handed, by result id, omitting a result with
        none. Called once, with every kept result that records a trace (``EvalResult.has_trace``), so the trials cost
        one read however many are kept; not called when none does.
    :ptype load_judge_evidence: Callable[[Sequence[str]], Mapping[str, JudgeEvidence]]
    :param load_results: Reads the named results whole, by id, in one read. Called once, with the kept trials (at most
        :data:`~threetears.evals.analysis.bundle.caps.MAX_TRIALS`), whose rows show the judge's reasons and its "could
        not tell" notes, which a reading leaves out; not called when none is kept. Only those fields of a row are read
        off the whole result, and the bundle's rewrites touch none of them.
    :ptype load_results: Callable[[Sequence[str]], Sequence[EvalResult]]
    :param rubric_scales: The scale each run's template dims were answered on (``EvalRun.rubric_scales``), by run id:
        what a dim the judge could not tell on is shown on, since that answer carries no score of its own.
    :ptype rubric_scales: Mapping[str, Mapping[str, RubricScale]]
    :return: The census, the kept rows and how many were left out. The losses by cause and the differential-loss flag
        are the bundle's ``exclusions`` and ``differential_exclusions``, which the assembler counts off the same
        :func:`~threetears.evals.kernel.result_condition.trial_exclusion`, never counted again here.
    :rtype: TrialsAndCensus
    :raises ConflictError: A kept trial's result was no longer stored when it was read whole (409).
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
    kept, omitted = capped(ordered, MAX_TRIALS, weight=lambda result: _KEEP_WEIGHT[readings[result.id].state])
    traced = [result.id for result in kept if result.has_trace]
    evidence = load_judge_evidence(traced) if traced else {}
    labels = _case_labels(kept, strata, evidence, names or {})
    whole = _whole(kept, load_results)
    places = case_places(case_lists)
    rows = [
        _row(
            f"T{index}",
            whole[result.id],
            readings[result.id],
            labels[result.test_case_id],
            strata,
            criteria,
            evidence.get(result.id),
            rubric_scales.get(result.eval_run_id, {}),
            places.get(result.test_case_id),
        )
        for index, result in enumerate(kept, start=1)
    ]
    return TrialsAndCensus(
        scoreboard=scoreboard,
        trials=rows,
        trials_omitted=omitted,
        case_lists_differ=len({tuple(cases) for cases in case_lists}) > 1,
    )


def case_places(case_lists: Sequence[Sequence[str]]) -> dict[str, int]:
    """Each case's place from 1: where it first stands in ``case_lists``, the member runs' frozen case lists chained
    in the bundle's run order (created, then id), each case kept where it first appears. Where every run holds one
    list (an ``Eval``'s every arm, the usual host campaign), that is the case's line in it, the line of a cases file.
    Where the lists differ (a case added or dropped between runs, a run reordered, runs sharded), it is the order the
    campaign first launched its cases in, so no two cases share a number, and no one run's list need hold the case at
    that line (a run over cases 1 to 38 and the next over 39 to 76 numbers the second's first case 39, and a case
    dropped from a later run keeps its number). Never read off a case's id.

    :param case_lists: The member runs' ``test_case_ids``, in the bundle's run order.
    :ptype case_lists: Sequence[Sequence[str]]
    :return: Each case's place, by case id.
    :rtype: dict[str, int]
    """
    ordered = dict.fromkeys(case for cases in case_lists for case in cases)
    return {case: place for place, case in enumerate(ordered, start=1)}


def bet_places(
    created_by: str, runs: Sequence[EvalRun], results: Sequence[ResultView], *, planned_looks: int | None
) -> dict[str, int] | None:
    """Each case's place in the order a quick-path comparison recorded its cases whole, which the mixture bet reads its
    paired contrasts in; ``None`` where the campaign is not one, and every reading is as an engine campaign's.

    **The order is the order the cases were completed in, as a person watching the results saw them.** A case is
    complete once every arm (each variant key the campaign's results carry) has recorded every repeat of it, each
    ``k_iteration`` from 1 to the runs' ``k_runs``, over any of its runs; its time is its last cell's: the launch that
    recorded it (the runs' launch groups, in the bundle's run order) and, within that launch, its ``scored_at``, the
    stamp a cell's result takes as it is recorded. Cases are placed by that time, ties by their place in the cases
    file (:func:`case_places` over the runs' lists). A case not yet complete holds no place, so the bet does not read
    it until it is: then its values are final, and every reading is a prefix of one sequence that only grows, the cases
    of a launch that stopped among it where they were completed, and the cells it never recorded, which a later launch
    of the comparison runs (:mod:`threetears.evals.quick.continuation`), completing their cases after them.

    A stop of a launch, whoever chose it after seeing whatever readings, is then a stopping time on that one sequence,
    and the bet's p keeps its error under it, valid if a case's latency does not depend on its outcome: where a wrong
    answer takes longer, which cases a stop finds completed depends on their values, and no reading order keeps it.

    A quick-path comparison is decided from stored facts only, so every reader of a stored campaign reads it alike:

    - its origin: the quick path filed it (``EvalCampaign.created_by`` is
      :data:`~threetears.evals.kernel.candidate_kind.QUICK_PATH_FILER`), so a host's campaign over the quick path's
      runs (``campaign_create``, filed as its caller) is not one;
    - every run is of a kind only the quick path launches
      (:data:`~threetears.evals.kernel.candidate_kind.QUICK_PATH_KINDS`), and it holds a result;
    - no arm recorded a cell (a case at a repeat) in two runs, which is how the quick path continues one comparison: a
      recorded cell is never run again, so the bet is one process;
    - no plan of looks is in force (``planned_looks``), which reads α/L instead.

    :param created_by: Who filed the campaign.
    :ptype created_by: str
    :param runs: The member runs, in the bundle's run order.
    :ptype runs: Sequence[EvalRun]
    :param results: Their results.
    :ptype results: Sequence[ResultView]
    :param planned_looks: The plan of looks in force, or ``None``.
    :ptype planned_looks: int | None
    :return: Each complete case's place, by case id, or ``None``.
    :rtype: dict[str, int] | None
    """
    if (
        planned_looks is not None
        or created_by != QUICK_PATH_FILER
        or not results
        or any(run.candidate_kind not in QUICK_PATH_KINDS for run in runs)
    ):
        return None
    # Each launch's index, in the bundle's run order; a run launched in no group is its own launch.
    launches: dict[str, int] = {}
    launch_of: dict[str, int] = {}
    for run in runs:
        launch_of[run.id] = launches.setdefault(run.launch_group_id or run.id, len(launches))
    held: set[tuple[str, str, int]] = set()
    repeats: dict[str, dict[str, set[int]]] = {}
    last: dict[str, tuple[int, str]] = {}
    for result in results:
        cell = (result.variant_key, result.test_case_id, result.k_iteration)
        if cell in held:
            return None
        held.add(cell)
        repeats.setdefault(result.test_case_id, {}).setdefault(result.variant_key, set()).add(result.k_iteration)
        # Stamped by one stamper with a +00:00 offset, so the strings order as the instants do.
        recorded = (launch_of.get(result.eval_run_id, len(launches)), result.scored_at)
        last[result.test_case_id] = max(recorded, last.get(result.test_case_id, recorded))
    arms = {result.variant_key for result in results}
    every_repeat = set(range(1, max((run.k_runs for run in runs), default=1) + 1))
    filed = case_places([run.test_case_ids for run in runs])
    complete = [
        case
        for case, on_arms in repeats.items()
        if set(on_arms) == arms and all(ran == every_repeat for ran in on_arms.values())
    ]
    complete.sort(key=lambda case: (last[case], filed.get(case, len(filed) + 1), case))
    return {case: place for place, case in enumerate(complete, start=1)}


def taken_up(created_by: str, runs: Sequence[EvalRun], results: Sequence[ResultView]) -> frozenset[str]:
    """The member runs of a quick-path comparison that did not finish and that a later launch of the comparison took up:
    every run that did not reach ``completed`` in any launch before the newest, where the bet reads the comparison
    (:func:`bet_places`, with no plan of looks; empty otherwise).

    The quick path continues a comparison by running every cell its launches never recorded, after the cells they
    recorded, which it keeps (:mod:`threetears.evals.quick.continuation`). So a launch that stopped is taken up once a
    later launch has run: that launch ran every cell it left, and whether the comparison opens with a verdict is the
    newest launch's ending alone. A launch is a launch group (``EvalRun.launch_group_id``); a run with none takes up
    nothing and is taken up by nothing.

    :param created_by: Who filed the campaign.
    :ptype created_by: str
    :param runs: The member runs, in the bundle's run order.
    :ptype runs: Sequence[EvalRun]
    :param results: Their results.
    :ptype results: Sequence[ResultView]
    :return: The runs taken up, by id.
    :rtype: frozenset[str]
    """
    if not runs or bet_places(created_by, runs, results, planned_looks=None) is None:
        return frozenset()
    newest = runs[-1].launch_group_id
    if newest is None:
        return frozenset()
    return frozenset(
        run.id
        for run in runs
        if run.status != "completed" and run.launch_group_id is not None and run.launch_group_id != newest
    )


def _whole(
    kept: Sequence[ResultView], load_results: Callable[[Sequence[str]], Sequence[EvalResult]]
) -> dict[str, EvalResult]:
    """The kept trials' results, whole, by id, in one read; none when no trial is kept.

    :raises ConflictError: A kept trial's result was no longer stored: its row would show no reason the judge gave, as
        though it gave none.
    """
    if not kept:
        return {}
    whole = {result.id: result for result in load_results([result.id for result in kept])}
    if missing := [result.id for result in kept if result.id not in whole]:
        raise ConflictError(
            f"the kept trials' results {missing!r} were no longer stored when read whole for their rows",
            remedy=_STALE_TRIALS_REMEDY,
        )
    return whole


def _census(arm: str, results: Sequence[ResultView], readings: Mapping[str, TrialReading]) -> ArmScoreboard:
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
                result.usage,
                substituted_deliveries=count_substituted_deliveries(result),
                candidate_kind=result.candidate_kind,
            )
        )
        is not None
    ]
    return ArmScoreboard(
        variant_key=arm,
        ran=len(results),
        # The census orders its causes itself (most trials first), on assembly and on every read back.
        lost=dict(lost),
        usable=len(usable),
        passed=sum(1 for reading in states if reading.state == "passed"),
        failed=dict(failed),
        cases=len({result.test_case_id for result in results}),
        usable_cases=len({result.test_case_id for result in usable}),
        no_criterion=sum(1 for reading in states if reading.state == "passed" and reading.no_criterion),
        test_cost_usd=sum(priced) if priced else None,
        n_cost_unpriced=len(results) - len(priced),
        reply_cost_usd=sum(replies) / len(replies) if replies else None,
        depth=depth_of(Counter(result.test_case_id for result in usable).values()),
        unanswered=_unanswered(results),
        pricing=_pricing(results),
    )


#: The usage role a tool the model called is metered under, in its provider's units: every other role is a model's.
_TOOL_ROLE = "external"


def _pricing(results: Sequence[ResultView]) -> ArmPricing | None:
    """How one arm's production cost was priced, off the rows of the roles a production deployment pays for
    (:func:`~threetears.evals.kernel.usage_capture.production_rows`): the price sources the priced rows name, the
    providers the model rows name (every role but ``external``), the providers the external rows name apart from them,
    and the calls counted and priced over every row, each None once a row did not count its calls.

    A tool the model called (a web-search API, metered as an ``external`` row) never reads as a provider of the model:
    its spend is the cost's and its calls are counted, but it is named as tool spend (``tool_providers``).

    A row holding only unanswered calls (:attr:`~threetears.evals.schema.models.RoleUsage.answers_nothing`)
    priced nothing and counted no answered call, so it is no row of the cost's: never one that left its calls
    uncounted, and never one whose provider went unrecorded. Every other model row names who served it or leaves it
    unrecorded, a row that reported no count included.

    :param results: The arm's trials.
    :ptype results: Sequence[ResultView]
    :return: The pricing; None where no trial recorded a production row of an answered call.
    :rtype: ArmPricing | None
    """
    rows = [
        row
        for result in results
        for row in production_rows(result.usage or [], candidate_kind=result.candidate_kind)
        if not row.answers_nothing
    ]
    if not rows:
        return None
    counted = all(row.call_count is not None for row in rows)
    model_rows = [row for row in rows if row.role != _TOOL_ROLE]
    providers = sorted({name for row in model_rows if (name := _served_by(row))})
    return ArmPricing(
        sources=sorted({row.price_source for row in rows if row.cost_usd is not None and row.price_source}),
        providers=providers,
        providers_in_part=bool(providers) and any(_served_by(row) is None for row in model_rows),
        tool_providers=sorted({row.provider for row in rows if row.provider and row.role == _TOOL_ROLE}),
        calls=sum(row.call_count or 0 for row in rows) if counted else None,
        priced_calls=sum(row.call_count or 0 for row in rows if row.cost_usd is not None) if counted else None,
    )


def _served_by(row: UsageView) -> str | None:
    """Who served a model row's calls: the upstream its replies named (``served_provider``), else the provider a host
    named on the row; ``None`` = not recorded, never a guess."""
    return row.served_provider or row.provider or None


def _unanswered(results: Sequence[ResultView]) -> list[RoleUnanswered]:
    """Each role's unanswered calls over one arm's trials, for every role a usage row counted them in.

    Sums what each row counted (``RoleUsage.unanswered_retried``, ``unanswered_lost``, ``unanswered_reasons``) and
    never adds a row that counted nothing as zero: a role no row counted in is absent, and a trial whose row in a
    counting role counted nothing (any of its rows there) is ``results_uncounted``. ``calls`` sums the counting rows' answered calls (none on a row
    holding only unanswered calls), and is
    None once one of them did not count its calls, since a share over a partial denominator would overstate.

    :param results: The arm's trials.
    :ptype results: Sequence[ResultView]
    :return: One entry per counting role, the roles in their known order, then any other by name.
    :rtype: list[RoleUnanswered]
    """
    counting: dict[str, list[UsageView]] = {}
    for result in results:
        for row in result.usage or ():
            if row.unanswered_retried is not None or row.unanswered_lost is not None:
                counting.setdefault(row.role, []).append(row)
    order = {role: index for index, role in enumerate(USAGE_ROLES)}
    entries: list[RoleUnanswered] = []
    for role in sorted(counting, key=lambda name: (order.get(name, len(order)), name)):
        rows = counting[role]
        retried: int | None = None
        lost: int | None = None
        calls: int | None = 0
        reasons: Counter[str] = Counter()
        for row in rows:
            retried = sum_optional_tokens(retried, row.unanswered_retried)
            lost = sum_optional_tokens(lost, row.unanswered_lost)
            reasons.update(row.unanswered_reasons or {})
            # A row holding only unanswered calls answered none (``RoleUsage.answers_nothing``): 0 answered calls, never
            # a row that left its answered calls uncounted.
            answered = 0 if row.answers_nothing else row.call_count
            calls = None if calls is None or answered is None else calls + answered
        uncounted = sum(
            1
            for result in results
            if any(
                row.role == role and row.unanswered_retried is None and row.unanswered_lost is None
                for row in result.usage or ()
            )
        )
        unanswered = (retried or 0) + (lost or 0)
        # A share over both kinds, so a row that counted only one of them leaves the share not counted, never a zero.
        both = all(row.unanswered_retried is not None and row.unanswered_lost is not None for row in rows)
        sent = None if calls is None or not both else calls + unanswered
        entries.append(
            RoleUnanswered(
                role=role,
                retried=retried,
                lost=lost,
                reasons=dict(reasons) or None,
                calls=calls,
                share=unanswered / sent if sent else None,
                results_uncounted=uncounted,
            )
        )
    return entries


def _case_labels(
    kept: Sequence[ResultView],
    strata: Mapping[str, str | None],
    evidence: Mapping[str, JudgeEvidence],
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
        shown = evidence.get(result.id)
        asked = shown.case_material if shown is not None else ""
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
    evidence: JudgeEvidence | None,
    scales: Mapping[str, RubricScale],
    place: int | None,
) -> TrialRow:
    """One trial's row, every text in it capped; a dim the judge could not tell on is shown on the scale its run
    recorded for it (:func:`~threetears.evals.schema.models.rubric_dim_scale`), never assumed."""
    scores: list[RubricScore] = [
        *result.rubric_scores,
        *(score for score in (result.transcript_score, result.outcome_score) if score is not None),
    ]
    judge = [
        TrialJudgement(
            dim=score.dim,
            criterion=criteria.get(score.dim),
            score=score.score,
            scale=score.scale,
            reason=capped_text(" ".join(score.reasoning.split()), MAX_REASON_CHARS),
        )
        for score in scores
    ]
    judge += [
        TrialJudgement(
            dim=dim,
            criterion=criteria.get(dim),
            score=None,
            scale=rubric_dim_scale(dim, scales),
            reason=capped_text("could not tell: " + " ".join(reason.split()), MAX_REASON_CHARS),
        )
        for dim, reason in sorted(result.judge_cannot_tell.items())
    ]
    errors = [
        f"{who}: {error}"
        for who, error in zip(
            ERROR_ROLES,
            (result.candidate_error, result.infra_error, result.judge_error, result.runner_error),
            strict=True,
        )
        if error
    ]
    return TrialRow(
        trial_id=trial_id,
        case_id=result.test_case_id,
        case_label=label,
        case_place=place,
        stratum=strata.get(result.test_case_id),
        arm=result.variant_key,
        repeat=result.k_iteration,
        state=reading.state,
        cause=reading.cause,
        failed_checks=list(reading.failed_checks),
        judged={score.dim: score.score for score in scores},
        judge=judge,
        first_input=capped_text(evidence.case_material, MAX_INPUT_CHARS) if evidence is not None else "",
        excerpt=tail_text(evidence.artifact, MAX_EXCERPT_CHARS) if evidence is not None else "",
        error_words=capped_text("; ".join(errors), MAX_ERROR_CHARS) if errors and reading.state != "passed" else None,
        result_id=result.id,
    )


__all__: list[str] = []
