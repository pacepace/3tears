"""A finished run as a short summary: how it ended, how its cells came out, and each measure's mean.

What :func:`~threetears.evals.quick.run_eval` returns, what the CLI prints after a launch and what the
``run_get`` action reads. It is read from the store, never from the job that ran, so a summary of a run finished in another process
says the same thing as one made the moment it ended.

Every count is over the run's stored results, and each result is classified once by
:func:`~threetears.evals.contracts.classify_result`: scored normally, failed by the candidate, or
excluded as a fault of the rig. A measure's mean is over the results that carry it, and its ``n``
says how many did, so a mean over two results of five is never mistaken for one over five.

**A classifier's run is read as one.** A kind that classifies lands ``match`` and ``confusion_cell``,
core measures no host declares, and the summary reads them wherever a result carries them: ``match``
as a measure whose mean is the share of answers that matched, and the ``confusion_cell`` counts as the
confusion matrix and each label's precision, recall and F1, counted by
:func:`~threetears.evals.analysis.confusion.label_statistics` as the analysis bundle counts them.

**A cost or latency measure is read over the turns the candidate took**
(:func:`~threetears.evals.contracts.delivered_a_turn`), as every analysis surface reads it
(:func:`~threetears.evals.contracts.metrics.summary_population`): a call the model refused or errored on
carries a round trip and an empty spend, and averaged in they read as a fast, free run. What was left out
is counted beside the mean — the failures that took no turn apart from the results excluded as a fault of
the rig — and a run none of whose results took a turn says ``no successful results`` instead of a number.

**A judged run's rubric is read too.** Each dimension a judge scored is summarised over the results that
carry its score, beside how many the judge could not tell on, and the judge's spend is the sum of the
results' ``judge`` usage rows — unknown, never zero, when any judge call went unpriced. So is the
template's intent, which the judge reads beside every answer: an unjudged run's is read by nothing that
grades it, so its summary leaves it out.

**So is what the candidate reported spending.** A kind's own ``candidate`` usage rows — a
:func:`~threetears.evals.quick.run_eval` candidate's :class:`~threetears.evals.quick.Answer` — are summed
the same way, unknown rather than zero when any went unpriced. A run whose candidate reported nothing
carries none of it.

**So are its goal-state checks.** Each check the results carry is counted as every per-check rate counts
it (:func:`~threetears.evals.contracts.counted_goal_verdicts`): passed as it evaluated, failed on every
check of a result the candidate failed, and in no count for a result excluded as a fault of the rig.
"""

from __future__ import annotations

import math

from collections import Counter
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from threetears.evals.analysis.confusion import ConfusionCount, LabelStatistics, confusion_matrix, label_statistics
from threetears.evals.analysis.stats import INTERVAL_LEVEL
from threetears.evals.analysis.surface_table import NO_SUCCESSFUL_RESULTS
from threetears.evals.contracts import (
    CONFUSION_CELL_MEASURE,
    MATCH_MEASURE,
    CostCapOrigin,
    EvalResult,
    EvalRun,
    GoalCheckProof,
    ResultOutcome,
    RubricScale,
    UsageRole,
    classify_result,
    counted_goal_verdicts,
    delivered_a_turn,
)
from threetears.evals.contracts.host import EvalHost
from threetears.evals.contracts.host.sweepables import UNCAPPED_SPEND
from threetears.evals.contracts.models import (
    CHECK_REFUSED_UNDER_CURRENT_GRAMMAR,
    goal_check_proofs_as_read,
    stale_goal_check_proofs,
)
from threetears.evals.contracts.metrics import describe_measure, summary_population
from threetears.evals.contracts.usage_capture import blended_cost
from threetears.evals.run import get_run, list_results


def dollars_text(amount: float) -> str:
    """Spend as a person reads it: dollars and cents from ten cents up, three significant figures below.

    A cheap model's call costs a few hundred-thousandths of a dollar, so a fixed number of decimals either
    shows it as $0.000000 or pads every larger amount with noise, and a general format shows it in
    exponent notation ($3e-05), which no one reads as money; the stored value is never rounded. From ten
    cents up, two decimals already carry two significant figures or more, and a third digit ($0.500) reads as
    noise; below, rounding to the cent would erase the difference between two cheap runs ($0.0123 and
    $0.0149 are not both $0.01). Public because two surfaces print spend — a run's summary and the
    agent-facing rendering of its results — and one rule for both is what keeps one amount from reading two
    ways.

    Args:
        amount: Dollars, as stored.

    Returns:
        The amount with its dollar sign: ``$0`` for zero, two decimals from ten cents up (``$0.50``,
        ``$1.20``), three significant figures below (``$0.0123``, ``$0.000160``).
    """
    if amount == 0:
        return "$0"
    if abs(amount) >= 0.1:
        return f"${amount:.2f}"
    decimals = 2 - math.floor(math.log10(abs(amount)))
    return f"${amount:.{decimals}f}"


#: Said beside a guardrail's level in one run's summary, which has no control to decide it against.
_GUARDRAIL_ALONE = "a guardrail: held, breached or undecided is decided only against a control, in a comparison"


class MeasureSummary(BaseModel):
    """One measure over a run's results.

    Attributes:
        name: The measure, as the host declares it.
        n: How many results carry it.
        mean: Their mean — a boolean measure's is its rate; ``None`` when none does, or for a text
            measure, whose words are listed by the analysis bundle and never averaged.
        minimum: The lowest value; ``None`` when none does.
        maximum: The highest value; ``None`` when none does.
        n_no_turn: How many results carrying it were left out of ``n`` and the mean because the candidate's
            model refused or errored and took no turn. Only a cost or latency measure leaves any out; 0 for
            every other.
        n_faulted: How many results carrying it were left out of ``n`` and the mean as a fault of the rig. Only
            a cost or latency measure leaves any out; 0 for every other.
        guardrail: Whether the host declares it a guardrail, something no arm may get worse on. A run alone has no
            control to hold it against, so its level here is no verdict: held, breached or undecided is decided
            only against a control, in a comparison.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    n: int
    mean: float | None
    minimum: float | None
    maximum: float | None
    n_no_turn: int = 0
    n_faulted: int = 0
    guardrail: bool = False


class GoalCheckSummary(BaseModel):
    """One goal-state check over a run's results.

    Attributes:
        check: The check, as the template states it.
        passed: How many results it counts as passed.
        n: How many results count for it: every result carrying it but those excluded as a fault of the rig.
        proof: Whether the check was shown, when the run launched, to tell its outcomes apart
            (``EvalRun.goal_check_proofs``); ``None`` for a run that recorded none, which reads as unproven.
            Only ``proven`` reads as measuring the behaviour: an unproven or refuted check's pass rate may be
            what a candidate that did nothing would score, and :meth:`render` says so beside it.
        did_nothing_passed: Of ``did_nothing_cases``, how many cases a candidate that did nothing passes — the
            check graded against each case's untouched starting state with no calls made. Set where each case's
            seed is in hand (:func:`~threetears.evals.quick.run_eval`'s world path); ``None`` elsewhere.
        did_nothing_cases: The cases that baseline was graded over; ``None`` with it.
        stale_proof: The run recorded the check ``proven`` under an older proof rule
            (:func:`~threetears.evals.contracts.models.goal_check_proofs_as_read`), so ``proof`` reads
            ``unproven`` and the check needs re-proving by a new launch.
        refused: Why the grammar refused the check when the run launched, for a check a template stored before
            the rule still carried; ``None`` otherwise. Such a check is graded on none of the run's results,
            which are not rig faults for it: ``excluded`` counts them.
        excluded: The results a refused check was not graded on — every result of the run; 0 otherwise.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    check: str
    passed: int
    n: int
    proof: GoalCheckProof | None = None
    did_nothing_passed: int | None = None
    did_nothing_cases: int | None = None
    stale_proof: bool = False
    refused: str | None = None
    excluded: int = 0

    @property
    def proven(self) -> bool:
        """Whether the pass rate measures the behaviour: shown at launch to beat doing nothing."""
        return self.proof == "proven"

    def line(self) -> str:
        """The check as :meth:`EvalSummary.render` prints it: its pass count, and, unless proven, why it is not one.

        Returns:
            One line, without indentation.
        """
        if self.refused is not None:
            return (
                f"goal check {self.check}: {CHECK_REFUSED_UNDER_CURRENT_GRAMMAR} ({self.refused}); graded on none "
                f"of the {self.excluded} result(s) — rewrite the check and launch again"
            )
        head = f"goal check {self.check}: passed {self.passed}/{self.n}"
        baseline = None
        if self.did_nothing_passed is not None and self.did_nothing_cases:
            baseline = f"a candidate that did nothing passes it in {self.did_nothing_passed} of {self.did_nothing_cases} case(s)"
        if self.proven:
            return head if baseline is None else f"{head} ({baseline})"
        if baseline is not None and self.did_nothing_passed == self.did_nothing_cases:
            return f"{head} — NOT A MEASUREMENT: {baseline}, so this pass rate does not beat doing nothing"
        reason = {
            "refuted": "refuted: its control does not show it tells acting from doing nothing",
            "unproven": (
                "unproven: its proof was recorded under an earlier proof rule and needs re-proving by a new launch"
                if self.stale_proof
                else "unproven: no control shows it tells acting from doing nothing"
            ),
            None: "unproven: this run recorded no proof that it tells acting from doing nothing",
        }[self.proof]
        return f"{head} — {reason}" + ("" if baseline is None else f"; {baseline}")


class DimensionSummary(BaseModel):
    """One judged rubric dimension over a run's results.

    Attributes:
        name: The dimension, as the template's rubric names it.
        scale: How it was answered: ``ordinal`` (1 to 5) or ``pass_fail`` (1 pass, 0 fail); ``None`` when
            no result carries a score to say.
        n: How many results carry a score on it.
        mean: Their mean; ``None`` when none does.
        minimum: The lowest score; ``None`` when none does.
        maximum: The highest score; ``None`` when none does.
        cannot_tell: How many results the judge answered it could not score on it — not failures, and in no mean.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    scale: RubricScale | None
    n: int
    mean: float | None
    minimum: float | None
    maximum: float | None
    cannot_tell: int


#: How one result came out, as :func:`~threetears.evals.contracts.classify_result` classifies it: graded normally,
#: failed by the candidate (it counts against the candidate), or excluded as a fault of the rig (it counts for nothing).
CaseOutcome = Literal["scored", "failed", "excluded"]

_CASE_OUTCOMES: dict[ResultOutcome, CaseOutcome] = {
    ResultOutcome.OK: "scored",
    ResultOutcome.CANDIDATE_FAIL: "failed",
    ResultOutcome.INFRA_EXCLUDE: "excluded",
}


class JudgeGrade(BaseModel):
    """One rubric dimension's score on one answer, with the judge's reason.

    Attributes:
        dimension: The dimension, as the rubric names it.
        scale: ``ordinal`` (1 to 5) or ``pass_fail`` (1 pass, 0 fail).
        score: The score.
        reasoning: What the judge said about the answer when it scored it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dimension: str
    scale: RubricScale
    score: int
    reasoning: str


def _misses(value: float, higher_is_better: bool | None) -> bool:
    """Whether a scorer's value is a miss for its direction: 0 or less where higher is better, above 0 where lower
    is, and never where the measure declares no direction."""
    if higher_is_better is None:
        return False
    return value <= 0 if higher_is_better else value > 0


class CaseResult(BaseModel):
    """One case's answer on one repeat, every grade it got, and why it failed or was excluded.

    What :meth:`EvalSummary.results` and :meth:`EvalSummary.misses` return, one per result, so a run's answers
    can be read after it ends without the store it ran in.

    Attributes:
        case: The case's name: its own ``id`` when the case carries one, else its position in the list given
            (``"0"`` is the first). The summary's :attr:`EvalSummary.errors` name cases the same way.
        input: The case, as given.
        expected: A classifier's expected label for the case; ``None`` when the run classified nothing.
        repeat: Which repeat of the case this is, from 1 to ``k``.
        outcome: ``scored``, ``failed`` (the candidate's failure, counted against it) or ``excluded`` (a fault
            of the rig — a scorer that raised, a judge that failed — counted for nothing).
        answer: The candidate's answer as returned (its ``repr`` when JSON could not hold it); ``None`` when it
            gave none.
        scores: Each grade the result carries, by measure: every scorer's value, and a classifier's ``match``
            and ``confusion_cell``.
        judged: Each rubric dimension the judge scored, with its reason; empty for an unjudged run.
        judge_cannot_tell: Each dimension the judge said it could not score, with its reason. Not a failure.
        goal_checks: Each goal-state check, by its expression, and whether the end state passed it.
        errors: Why the result failed or was excluded, as the run recorded it; empty for a scored result.
        missed_because: Why the result is a miss, one line per reason; empty when it is not one. A miss is a
            result the candidate failed, a classifier answer that is not the expected label, a scorer on the
            wrong side of 0 for its direction — 0 or less where higher is better (``False`` counts as 0), more
            than 0 where lower is better, and never for a measure that declares no direction — a goal check the end
            state failed, or a pass/fail dimension the judge failed. An excluded result is never a miss: it says
            nothing about the candidate.
        cost_usd: What the result spent, as reported and priced; ``None`` when any of it went unpriced.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    case: str
    input: JsonValue
    expected: str | None = None
    repeat: int
    outcome: CaseOutcome
    answer: JsonValue = None
    scores: dict[str, bool | float | str] = {}
    judged: list[JudgeGrade] = []
    judge_cannot_tell: dict[str, str] = {}
    goal_checks: dict[str, bool] = {}
    errors: list[str] = []
    missed_because: list[str] = []
    cost_usd: float | None = None

    @property
    def missed(self) -> bool:
        """Whether the result is a miss (:attr:`missed_because` says why)."""
        return bool(self.missed_because)

    @classmethod
    def of(
        cls,
        result: EvalResult,
        *,
        case: str,
        given: JsonValue,
        expected: str | None,
        answer: JsonValue,
        higher_is_better: Mapping[str, bool | None] | None = None,
    ) -> CaseResult:
        """One stored result, read as a case result.

        Args:
            result: The stored result.
            case: The case's name.
            given: The case, as given.
            expected: A classifier's expected label, or ``None``.
            answer: The candidate's answer, as its kind stored it.
            higher_is_better: Each measure's declared direction, by name. Lower-is-better (``leaked``, a count
                of errors) misses above 0 and not at 0, guardrail or not; a measure that declares no direction
                misses never, since "worse" needs one; a measure not named reads as higher-is-better, a scorer's
                default.

        Returns:
            The case result, its miss reasons decided by the rule :attr:`missed_because` states.
        """
        outcome = _CASE_OUTCOMES[classify_result(result)]
        errors = (
            []
            if outcome == "scored"
            else [
                error
                for error in (
                    result.runner_error,
                    None if result.judge_error is None else f"judge: {result.judge_error}",
                )
                if error
            ]
        )
        judged = [
            JudgeGrade(dimension=score.dim, scale=score.scale, score=score.score, reasoning=score.reasoning)
            for score in result.rubric_scores
        ]
        goal_checks = {check.expression: check.passed for check in result.goal_state_outcomes}
        missed: list[str] = []
        if outcome == "failed":
            missed.extend(f"failed: {error}" for error in errors or ["the candidate failed"])
        elif outcome == "scored":
            if result.host_measures.get(MATCH_MEASURE) is False:
                missed.append(f"answered {_shown_value(answer)}, expected {_shown_value(expected)}")
            missed.extend(
                f"{name} gave {value:g}"
                for name, value in result.host_measures.items()
                if name not in (MATCH_MEASURE, CONFUSION_CELL_MEASURE)
                and not isinstance(value, str)
                and _misses(float(value), (higher_is_better or {}).get(name, True))
            )
            missed.extend(f"goal check {check} failed" for check, passed in goal_checks.items() if not passed)
            missed.extend(
                f"the judge failed it on {grade.dimension}: {' '.join(grade.reasoning.split())}"
                for grade in judged
                if grade.scale == "pass_fail" and grade.score == 0
            )
        return cls(
            case=case,
            input=given,
            expected=expected,
            repeat=result.k_iteration,
            outcome=outcome,
            answer=answer,
            scores=dict(result.host_measures),
            judged=judged,
            judge_cannot_tell=dict(result.judge_cannot_tell),
            goal_checks=goal_checks,
            errors=errors,
            missed_because=missed,
            cost_usd=result.cost_usd,
        )

    def render(self) -> str:
        """The result as a few lines of text: the case, the answer, the grades, and why it missed or failed.

        Returns:
            The text, without a trailing newline.
        """
        head = f"case {self.case} (repeat {self.repeat}, {self.outcome}): answered {_shown_value(self.answer)}"
        if self.expected is not None:
            head += f", expected {_shown_value(self.expected)}"
        lines = [head]
        grades = [
            f"{name} {value if isinstance(value, str | bool) else f'{value:g}'}"
            for name, value in self.scores.items()
            if name != CONFUSION_CELL_MEASURE
        ]
        if grades:
            lines.append(f"  grades: {', '.join(grades)}")
        lines.extend(
            f"  judged {grade.dimension}: {grade.score} — {' '.join(grade.reasoning.split())}" for grade in self.judged
        )
        lines.extend(
            f"  judged {dimension}: could not tell — {' '.join(reason.split())}"
            for dimension, reason in self.judge_cannot_tell.items()
        )
        lines.extend(
            f"  goal check {check}: {'passed' if passed else 'failed'}" for check, passed in self.goal_checks.items()
        )
        lines.extend(f"  error: {error}" for error in self.errors)
        lines.extend(f"  missed: {reason}" for reason in self.missed_because)
        return "\n".join(lines)


def _shown_value(value: Any) -> str:
    """A value as a result's line prints it: a string quoted, anything else as written."""
    return repr(value) if isinstance(value, str) or value is None else str(value)


class EvalSummary(BaseModel):
    """One run, summarised.

    Attributes:
        run_id: The run.
        scope_id: The scope it is stored in.
        template_id: The template it ran; ``None`` for an ad-hoc run of explicit cases.
        candidate_model: The arm's candidate model.
        arm: The arm's name, where a comparison named its arms apart from their model
            (:func:`~threetears.evals.quick.compare`, every arm at one shared model); ``None`` elsewhere, and for a
            summary read back from the store.
        status: How the run ended, as stored (``completed``, ``failed``, ``cancelled``, ...).
        k_runs: Repeats per case.
        n_cases: Cases in the run's frozen case set.
        n_results: Results stored.
        n_scored: Results scored normally.
        n_candidate_failed: Results the candidate failed: they count against it.
        n_excluded: Results excluded as a fault of the rig: they count for nothing.
        measures: Each measure the host declares, over the results that carry it, then ``match`` and
            ``confusion_cell`` when a result carries them and the host does not declare them.
        confusion: The confusion matrix of the results' ``confusion_cell`` values, by expected then
            predicted label; empty for a run that classified nothing.
        labels: Each label's precision, recall and F1 from the same observations, by label, each interval
            over the cases behind it (a case classified k times is one draw); empty with the matrix.
        judged: Each rubric dimension a judge scored or could not tell on, in the order first met; empty
            for an unjudged run.
        goal_checks: Each goal-state check the results carry, in the order first met; empty for a run
            with none.
        intent: The template's intent, which a judged run's judge read beside every answer; ``None`` for an
            unjudged run, and for a template edited since the run launched, whose intent is no longer the one
            the judge read.
        intent_source: Where the intent came from, as the caller that wrote the template says it
            (:func:`~threetears.evals.quick.run_eval`: ``"from <candidate>'s docstring"`` or a generic
            default); ``None`` for an intent stated outright, and for a summary read back from the store,
            which keeps the intent but not its source.
        judge_calls: How many judge calls the results' ``judge`` usage rows count.
        judge_cost_usd: What those calls cost, as their client priced them; ``None`` when any went
            unpriced, and for a run no judge was called in.
        candidate_calls: How many calls the results' ``candidate`` usage rows count; 0 when the candidate
            reported no spend.
        candidate_cost_usd: What those calls cost, as the candidate priced them; ``None`` when any went
            unpriced, and for a run whose candidate reported no spend.
        errors: Each failed or excluded result's error, prefixed by its case, then the run's own. A case is named
            as the caller that summarised the run named it (:func:`~threetears.evals.quick.run_eval`: its own
            ``id``, else its position), else by its stored test case id.
        max_cost_usd: The spend ceiling the run was held to, in US dollars; ``None`` when none bound it.
        max_cost_usd_origin: Where that ceiling came from, as the run records it: ``chosen`` (the launch named
            it), ``inherited`` (the host's default), ``uncapped`` (ceiling enforcement was off, so nothing bounded
            the run), or ``None`` for a run whose writer recorded none.
        judge_shares_candidate_model: Each model that both judged the run's answers and produced them — the
            judge's model (as requested, or as a score says it was served) matching the run's candidate model or
            a model the candidate's usage rows name — sorted; empty when they share none, or for an unjudged run.
            A model tends to rate its own output higher, so a judged score from one of these may favour the
            candidate. Compared on the ids as recorded, so an alias one side spells differently is not caught.
        case_results: Every result, read for a person (:class:`CaseResult`), in case order then repeat; ``None``
            when the run was summarised without its cases, as the CLI and ``run_get`` summarise one.
            :meth:`results` and :meth:`misses` read it.
        stopped_because: Why a designed stop ended the run short, as the run records it: the reason an
            operator gave for a cancel, or which budget stopped it (its cost cap, or its wall-clock budget).
            ``None`` for a run nothing stopped, and for a cancel given no reason. A stop is never one of
            :attr:`errors`, which are faults.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    scope_id: str
    template_id: str | None
    candidate_model: str
    arm: str | None = None
    status: str
    k_runs: int
    n_cases: int
    n_results: int
    n_scored: int
    n_candidate_failed: int
    n_excluded: int
    measures: list[MeasureSummary]
    confusion: list[ConfusionCount]
    labels: list[LabelStatistics]
    judged: list[DimensionSummary] = []
    judge_calls: int = 0
    judge_cost_usd: float | None = None
    candidate_calls: int = 0
    candidate_cost_usd: float | None = None
    goal_checks: list[GoalCheckSummary] = []
    intent: str | None = None
    intent_source: str | None = None
    errors: list[str]
    stopped_because: str | None = None
    max_cost_usd: float | None = None
    max_cost_usd_origin: CostCapOrigin | None = None
    judge_shares_candidate_model: list[str] = []
    case_results: list[CaseResult] | None = None

    def results(self) -> list[CaseResult]:
        """Every result: each case's answer on each repeat, its grades, and why it failed or was excluded.

        Returns:
            One :class:`CaseResult` per result, in case order then repeat.

        Raises:
            ValueError: The summary carries no results — it was summarised without its cases (the CLI's and
                ``run_get``'s summaries are); read the stored results with
                :func:`~threetears.evals.run.list_results` instead.
        """
        if self.case_results is None:
            raise ValueError(
                f"this summary of run {self.run_id} carries no per-case results: it was summarised without its "
                "cases, as the CLI and run_get summarise a run; read them with threetears.evals.run.list_results"
            )
        return list(self.case_results)

    def misses(self) -> list[CaseResult]:
        """The results the candidate missed, each saying why (:attr:`CaseResult.missed_because`).

        A miss is a result the candidate failed, a classifier answer that is not the expected label, a scorer
        on the wrong side of 0 for its direction (:attr:`CaseResult.missed_because`), a goal check
        the end state failed, or a pass/fail dimension the judge failed. An excluded result is not a miss — it says nothing about the candidate —
        so read :meth:`results` for those; the summary's :attr:`n_excluded` counts them.

        Returns:
            The missed results, in case order then repeat.

        Raises:
            ValueError: The summary carries no results (:meth:`results`).
        """
        return [result for result in self.results() if result.missed]

    def render(self) -> str:
        """The summary as a few lines of text for a terminal.

        The results line counts the failures and the rig's exclusions only when there are some, or a judge is in
        play; the spend-cap line prints only when a cap bounded the run, a judge ran, or the candidate reported
        spend — a first run of a free function is not handed the vocabulary of a rig it does not have.

        Returns:
            The text, without a trailing newline.
        """
        # A judge is the rig a first run can meet, and the spend it bills is what a cap bounds: with neither in
        # play, the rig's exclusions and the cap say nothing a newcomer can act on, so they print only when they do.
        judged = bool(self.judged) or self.judge_calls > 0
        counts = [f"{self.n_scored} scored"]
        if self.n_candidate_failed:
            counts.append(f"{self.n_candidate_failed} failed by the candidate")
        if self.n_excluded or judged:
            counts.append(f"{self.n_excluded} excluded")
        lines = [
            f"run {self.run_id} {self.status}: "
            + (self.candidate_model if self.arm is None else f"arm {self.arm} (model {self.candidate_model})")
            + f" over {self.n_cases} case(s) x k={self.k_runs}",
            f"  {self.n_results} result(s): {', '.join(counts)}",
        ]
        if self.stopped_because is not None:
            lines.append(f"  stopped: {self.stopped_because}")
        if (self.max_cost_usd is not None or judged or self.candidate_calls) and (
            cap := _spend_cap_line(self)
        ) is not None:
            lines.append(f"  {cap}")
        for measure in self.measures:
            left_out = _left_out(measure)
            if measure.n == 0 and measure.n_no_turn:
                lines.append(f"  {measure.name}: {NO_SUCCESSFUL_RESULTS}{left_out}")
            elif measure.n == 0 and measure.n_faulted:
                lines.append(f"  {measure.name}: every result carrying it was excluded{left_out}")
            elif measure.n == 0:
                lines.append(f"  {measure.name}: no result carries it")
            elif measure.name == CONFUSION_CELL_MEASURE and self.confusion:
                lines.append(f"  {measure.name}: n={measure.n}, counted in the confusion matrix below")
            elif measure.mean is None:
                lines.append(f"  {measure.name}: n={measure.n}, not a number")
            else:
                lines.append(
                    f"  {measure.name}: mean {measure.mean:.3g} (n={measure.n}, "
                    f"min {measure.minimum:.3g}, max {measure.maximum:.3g}{left_out})"
                    + (f"; {_GUARDRAIL_ALONE}" if measure.guardrail else "")
                )
        if self.confusion:
            lines.append("  confusion (expected → predicted):")
            lines.extend(
                f"    {_shown(cell.expected)} → {_shown(cell.predicted)}: {cell.count}" for cell in self.confusion
            )
            lines.append("  per label:")
            lines.extend(f"    {_label_line(statistics)}" for statistics in self.labels)
        if self.intent is not None:
            source = "" if self.intent_source is None else f" ({self.intent_source})"
            lines.append(f"  intent{source}: {' '.join(self.intent.split())}")
        lines.extend(f"  {_dimension_line(dimension)}" for dimension in self.judged)
        if self.judge_shares_candidate_model:
            lines.append(f"  {self_judging_text(self.judge_shares_candidate_model, 'the candidate')}")
        if self.judged:
            spend = (
                "unknown: a judge call went unpriced"
                if self.judge_cost_usd is None
                else dollars_text(self.judge_cost_usd)
            )
            lines.append(f"  judge spend: {spend} over {self.judge_calls} call(s)")
        if self.candidate_calls:
            spend = (
                "unknown: a candidate call went unpriced"
                if self.candidate_cost_usd is None
                else dollars_text(self.candidate_cost_usd)
            )
            lines.append(f"  candidate spend: {spend} over {self.candidate_calls} call(s)")
        lines.extend(f"  {goal.line()}" for goal in self.goal_checks)
        lines.extend(f"  error: {error}" for error in self.errors)
        return "\n".join(lines)


def self_judging_text(models: list[str], whose: str) -> str:
    """The disclosure that a judge graded answers its own model produced, as every quick surface words it.

    Public because two surfaces say it — a run's summary and a judged comparison's report — and one wording
    for both keeps the warning from reading two ways.

    Args:
        models: The models that both judged and answered.
        whose: Whose answers they produced (``"the candidate"``, ``"arm 'candidate'"``).

    Returns:
        One sentence.
    """
    return (
        f"self-judging: the judge's model {', '.join(models)} also produced {whose}'s answers, and a model tends "
        "to rate its own output higher, so its judged scores may favour them; judge with another model to rule "
        "that out"
    )


def _self_judging(run: EvalRun, results: list[EvalResult]) -> list[str]:
    """The models that both judged a run's answers and produced them (:attr:`EvalSummary.judge_shares_candidate_model`)."""
    judges = {*(run.effective_judges or {}).values(), *([run.judge_model] if run.judge_model else [])}
    judges |= {score.served_model for result in results for score in result.rubric_scores if score.served_model}
    if not judges:
        return []
    candidates = {run.candidate_model}
    candidates |= {
        model
        for result in results
        for row in result.usage
        if row.role == "candidate"
        for model in (row.model, row.served_model)
        if model
    }
    return sorted(judges & candidates)


def _spend_cap_line(summary: EvalSummary) -> str | None:
    """The spend ceiling the run was held to, or that it had none; nothing for a run that recorded neither.

    A cap counts the spend results report, so a capped run none of whose candidate calls reported any says
    that its cap counted only what else was spent (a judge's calls, say).
    """
    if summary.max_cost_usd_origin == "uncapped":
        return f"spend cap: {UNCAPPED_SPEND}"
    if summary.max_cost_usd is None:
        return None
    line = f"spend cap: {dollars_text(summary.max_cost_usd)} for this run"
    if summary.candidate_calls == 0:
        line += "; it counts only spend a result reports, and the candidate reported none"
    return line


def _left_out(measure: MeasureSummary) -> str:
    """What a cost or latency measure's mean left out, each kind named for what it is — or nothing."""
    parts = [
        *([f"{measure.n_no_turn} refused or errored with no turn taken"] if measure.n_no_turn else []),
        *([f"{measure.n_faulted} excluded as a fault of the rig"] if measure.n_faulted else []),
    ]
    return f", left out: {' and '.join(parts)}" if parts else ""


def _shown(label: str) -> str:
    """A label as printed: as written, or quoted when whitespace at its ends would otherwise be invisible."""
    return repr(label) if label != label.strip() else label


def _proportion(
    name: str, rate: float | None, hits: int, n: int, cases: int, interval: tuple[float, float] | None
) -> str:
    """A rate with its count and interval; where cases repeat, the count says over how many cases it rests on."""
    if rate is None:
        return f"{name} none (n=0)"
    shown = f"{name} {rate:.3g} ({hits}/{n}"
    if cases < n:
        shown += f" over {cases} case{'' if cases == 1 else 's'}"
    if interval is not None:
        shown += f", {INTERVAL_LEVEL:.0%} CI {interval[0]:.2g}-{interval[1]:.2g}"
    return shown + ")"


def _dimension_line(dimension: DimensionSummary) -> str:
    """One judged dimension's line: its scale, its mean and range, and how often the judge could not tell."""
    scale = {"ordinal": "judged 1-5", "pass_fail": "judged pass/fail", None: "judged"}[dimension.scale]
    line = f"{dimension.name} ({scale}): "
    if dimension.mean is None or dimension.minimum is None or dimension.maximum is None:
        line += "no result carries a score"
    else:
        line += f"mean {dimension.mean:.3g} (n={dimension.n}, min {dimension.minimum:.3g}, max {dimension.maximum:.3g})"
    if dimension.cannot_tell:
        line += f", the judge could not tell on {dimension.cannot_tell}"
    return line


def _label_line(statistics: LabelStatistics) -> str:
    """One label's line: its precision and recall with their counts and intervals, and its F1."""
    precision = _proportion(
        "precision",
        statistics.precision,
        statistics.correct,
        statistics.predicted,
        statistics.predicted_cases,
        statistics.precision_interval,
    )
    recall = _proportion(
        "recall",
        statistics.recall,
        statistics.correct,
        statistics.expected,
        statistics.expected_cases,
        statistics.recall_interval,
    )
    f1 = "f1 none" if statistics.f1 is None else f"f1 {statistics.f1:.3g}"
    return f"{_shown(statistics.label)}: {precision}, {recall}, {f1}"


def summarize_run(
    host: EvalHost, run_id: str, scope_id: str, *, case_names: Mapping[str, str] | None = None
) -> EvalSummary:
    """Summarise one stored run and its results.

    Args:
        host: The host whose store holds the run, and whose measures are summarised.
        run_id: The run.
        scope_id: The scope it lives in.
        case_names: What each case is called in the summary's errors, by stored test case id; a case it does
            not name, and every case when it is ``None``, is called by its id.

    Returns:
        The summary.

    Raises:
        NotFoundError: No run with that id in the scope.
    """
    run = get_run(host.storage, run_id, scope_id)
    results = list_results(host.storage, run_id, scope_id)
    outcomes = [classify_result(result) for result in results]
    declared = host.profile.measures.names
    classified = [
        name
        for name in (MATCH_MEASURE, CONFUSION_CELL_MEASURE)
        if name not in declared and any(name in result.host_measures for result in results)
    ]
    measures = []
    for name in (*declared, *classified):
        carrying = [result for result in results if name in result.host_measures]
        # A cost or latency reading is a turn's: a call that took none carries a round trip and an empty spend,
        # and a faulted one measured the rig. `delivered_a_turn` is the one predicate every surface reads.
        turns_only = summary_population(describe_measure(name, host.profile.measures), "all_observed") == "delivered"
        left = [result for result in carrying if turns_only and not delivered_a_turn(result)]
        carried = [result.host_measures[name] for result in carrying if not turns_only or delivered_a_turn(result)]
        # A text observation is words, never a number; a boolean counts as 1 or 0, so its mean is its rate.
        values = [float(value) for value in carried if not isinstance(value, str)]
        measures.append(
            MeasureSummary(
                name=name,
                n=len(carried),
                mean=sum(values) / len(values) if values else None,
                minimum=min(values) if values else None,
                maximum=max(values) if values else None,
                n_no_turn=sum(1 for result in left if classify_result(result) is ResultOutcome.CANDIDATE_FAIL),
                n_faulted=sum(1 for result in left if classify_result(result) is ResultOutcome.INFRA_EXCLUDE),
                guardrail=describe_measure(name, host.profile.measures).guardrail,
            )
        )
    names = case_names or {}
    errors = [
        f"case {names.get(result.test_case_id, result.test_case_id)}: {error}"
        for result, outcome in zip(results, outcomes, strict=True)
        if outcome is not ResultOutcome.OK
        for error in (result.runner_error, None if result.judge_error is None else f"judge: {result.judge_error}")
        if error
    ]
    errors.extend(run.error_details)
    cells = Counter(
        cell for result in results if isinstance(cell := result.host_measures.get(CONFUSION_CELL_MEASURE), str)
    )
    confusion = confusion_matrix(cells)
    # Each observation beside its case, so a label's interval counts a case classified k times as one draw.
    classifications = [
        (cell, result.test_case_id)
        for result in results
        if isinstance(cell := result.host_measures.get(CONFUSION_CELL_MEASURE), str)
    ]
    judge_rows = [row for result in results for row in result.usage if row.role == "judge"]
    template = None
    if run.judge_model is not None and run.template_id is not None:
        template = host.storage.load_template(run.template_id, scope_id)
    candidate_rows = [row for result in results for row in result.usage if row.role == "candidate"]
    return EvalSummary(
        run_id=run.id,
        scope_id=scope_id,
        template_id=run.template_id,
        candidate_model=run.candidate_model,
        status=run.status,
        k_runs=run.k_runs,
        n_cases=len(run.test_case_ids),
        n_results=len(results),
        n_scored=outcomes.count(ResultOutcome.OK),
        n_candidate_failed=outcomes.count(ResultOutcome.CANDIDATE_FAIL),
        n_excluded=outcomes.count(ResultOutcome.INFRA_EXCLUDE),
        measures=measures,
        confusion=confusion,
        labels=label_statistics(classifications),
        judged=_judged_dimensions(results),
        judge_calls=sum(row.call_count or 0 for row in judge_rows),
        judge_cost_usd=blended_cost(judge_rows, _JUDGE_ROLE) if judge_rows else None,
        candidate_calls=sum(row.call_count or 0 for row in candidate_rows),
        candidate_cost_usd=blended_cost(candidate_rows, _CANDIDATE_ROLE) if candidate_rows else None,
        goal_checks=_goal_checks(results, run),
        # Templates are edited in place: one edited since the launch no longer holds what the judge read.
        intent=template.intent if template is not None and template.updated_at <= run.created_at else None,
        errors=errors,
        stopped_because=run.cancellation_reason or run.budget_stop_reason,
        max_cost_usd=run.max_cost_usd,
        max_cost_usd_origin=run.max_cost_usd_origin,
        judge_shares_candidate_model=_self_judging(run, results),
    )


#: The one role a run's judge spends under, which the summary's judge spend sums.
_JUDGE_ROLE: tuple[UsageRole, ...] = ("judge",)

#: The role a kind's own calls spend under, which the summary's candidate spend sums.
_CANDIDATE_ROLE: tuple[UsageRole, ...] = ("candidate",)


def _judged_dimensions(results: list[EvalResult]) -> list[DimensionSummary]:
    """Each rubric dimension the results carry a judge's score or a judge's "cannot tell" on, in the order first met."""
    scores: dict[str, list[float]] = {}
    scales: dict[str, RubricScale] = {}
    cannot_tell: Counter[str] = Counter()
    for result in results:
        for score in result.rubric_scores:
            scores.setdefault(score.dim, []).append(float(score.score))
            scales.setdefault(score.dim, score.scale)
        for dim in result.judge_cannot_tell:
            scores.setdefault(dim, [])
            cannot_tell[dim] += 1
    return [
        DimensionSummary(
            name=name,
            scale=scales.get(name),
            n=len(values),
            mean=sum(values) / len(values) if values else None,
            minimum=min(values) if values else None,
            maximum=max(values) if values else None,
            cannot_tell=cannot_tell[name],
        )
        for name, values in scores.items()
    ]


def _goal_checks(results: list[EvalResult], run: EvalRun) -> list[GoalCheckSummary]:
    """Each goal-state check the results carry, counted as every per-check rate counts it, in the order first met.

    Each carries the proof its run froze at launch, as read under the current proof rules
    (:func:`~threetears.evals.contracts.models.goal_check_proofs_as_read`); a check the run recorded none for is
    unproven. Each check the grammar refused at launch follows, graded on no result and counted as excluded.
    """
    proofs = goal_check_proofs_as_read(run)
    stale = set(stale_goal_check_proofs(run))
    counted: dict[str, list[bool]] = {}
    for result in results:
        for outcome, passed in counted_goal_verdicts(result) or []:
            counted.setdefault(outcome.expression, []).append(passed)
    graded = [
        GoalCheckSummary(
            check=check,
            passed=sum(verdicts),
            n=len(verdicts),
            proof=None if proofs is None else proofs.get(check, "unproven"),
            stale_proof=check in stale,
        )
        for check, verdicts in counted.items()
    ]
    refused = [
        GoalCheckSummary(check=check, passed=0, n=0, proof="refuted", refused=reason, excluded=len(results))
        for check, reason in (run.refused_goal_checks or {}).items()
        if check not in counted
    ]
    return graded + refused


__all__ = [
    "CaseOutcome",
    "CaseResult",
    "DimensionSummary",
    "EvalSummary",
    "GoalCheckSummary",
    "JudgeGrade",
    "MeasureSummary",
    "dollars_text",
    "self_judging_text",
    "summarize_run",
]
