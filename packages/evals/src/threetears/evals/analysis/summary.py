"""A finished run as a short summary: how it ended, how its cells came out, and each measure's mean.

What :func:`~threetears.evals.quick.run_eval` returns, what the CLI prints after a launch and what the
``run_get`` action reads. It is read from the store, never from the job that ran, so a summary of a run finished in another process
says the same thing as one made the moment it ended.

Every count is over the run's stored results, and each result is classified once by
:func:`~threetears.evals.kernel.classify_result`: scored normally, failed by the candidate, or
excluded as a fault of the rig. A measure's mean is over the results that carry it, and its ``n``
says how many did, so a mean over two results of five is never mistaken for one over five.

**A classifier's run is read as one.** A kind that classifies lands ``match`` and ``confusion_cell``,
core measures no host declares, and the summary reads them wherever a result carries them: ``match``
as a measure whose mean is the share of answers that matched, and the ``confusion_cell`` counts as the
confusion matrix and each label's precision, recall and F1, counted by
:func:`~threetears.evals.analysis.confusion.label_statistics` as the analysis bundle counts them.

**A cost or latency measure is read over the turns the candidate took**
(:func:`~threetears.evals.kernel.delivered_a_turn`), as every analysis surface reads it
(:func:`~threetears.evals.kernel.metrics.summary_population`): a call the model refused or errored on
carries a round trip and an empty spend, and averaged in they read as a fast, free run. What was left out
is counted beside the mean — the failures that took no turn apart from the results excluded as a fault of
the rig — and a run none of whose results took a turn says ``no successful results`` instead of a number.

**A judged run's rubric is read too.** Each dimension a judge scored is summarised over the results that
carry its score, beside how many the judge could not tell on, and the judge's spend is the sum of the
results' ``judge`` usage rows — unknown, never zero, when any judge call went unpriced. So is the
template's intent, which the judge reads beside every answer: an unjudged run's is read by nothing that
grades it, so its summary leaves it out.

**So is what every role spent.** Each role a result's usage rows name — the candidate, the judge, a simulator,
an inner agent the candidate called, a paid external API — is summed on its own line (:class:`RoleSpend`), unknown
rather than zero when any of its calls went unpriced, with an external API's own units (search credits, quota
units) beside it per provider, and the results' blended cost below as the total. A line per role rather than a
line for the candidate and one for a judge: a run whose inner agent spent most of its money printed the
other two and hid it, and a judge campaign — every trial bills the ``judge`` role — printed no spend at all. A run
whose results reported nothing carries none of it.

**So are its goal-state checks.** Each check the results carry is counted as every per-check rate counts
it (:func:`~threetears.evals.kernel.counted_goal_verdicts`): passed as it evaluated, failed on every
check of a result the candidate failed, and in no count for a result excluded as a fault of the rig.

**So is each kind of case.** When any case of the run declares a stratum (``EvalTestCase.stratum``), every
measure is read again over each stratum's results alone (:class:`StratumSummary`), the cases declaring none
their own stratum, and a stratum of fewer than ``STRATUM_MIN_CASES`` cases is said to be too few to read alone.
A run none of whose cases declares one carries no strata and reads as before.

**Its words are served with it.** Every line :meth:`EvalSummary.render` prints is a field of the summary's JSON
(each a ``computed_field``): the run's line, its results line, why it stopped, its spend cap, each measure's
state and line, the confusion and per-label lines, each stratum's lines, the intent, each judged dimension, the self-judging warning,
each role's spend and the total, and each goal check (each error is ``errors`` itself, printed as written) —
and ``text``, all of it as one string.
So a browser shows the words this package wrote, and ``render`` is those fields joined, never a second spelling.
A summary read back from its JSON re-derives them and discards what the document said.
"""

from __future__ import annotations

import math

from collections import Counter
from collections.abc import Mapping
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, JsonValue, computed_field, model_validator

from threetears.evals.analysis.agreement import InterJudgeDimension, inter_judge_agreement
from threetears.evals.analysis.confusion import ConfusionCount, LabelStatistics, confusion_matrix, label_statistics
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.reporting import (
    UnconfirmedModelIdentity,
    model_identity_disclosure,
    unconfirmed_model_identities,
)
from threetears.evals.analysis.stats import INTERVAL_LEVEL
from threetears.evals.analysis.surface_table import NO_SUCCESSFUL_RESULTS
from threetears.evals.analysis.viz.quantities import display_scale, with_unit
from threetears.evals.kernel import (
    NotFoundError,
    CONFUSION_CELL_MEASURE,
    MATCH_MEASURE,
    ResultOutcome,
    classify_result,
    counted_goal_verdicts,
    delivered_a_turn,
)
from threetears.evals.schema import CostCapOrigin, EvalResult, EvalRun, GoalCheckProof, RubricScale, UsageRole
from threetears.evals.schema.base import without_computed_field_echo
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.host.sweepables import UNCAPPED_SPEND
from threetears.evals.schema.models import (
    CHECK_REFUSED_UNDER_CURRENT_GRAMMAR,
    goal_check_proofs_as_read,
    stale_goal_check_proofs,
)
from threetears.evals.kernel.metrics import describe_measure, summary_population
from threetears.evals.kernel.surface import STRATUM_MIN_CASES
from threetears.evals.analysis.report.build import NO_STRATUM, TOO_FEW_CASES
from threetears.evals.kernel.remedies import FIND_RUN


#: What an amount no client could price reads as. Never ``$0``: an unpriced call may have been billed.
UNPRICED = "unpriced"


def dollars_text(amount: float | None) -> str:
    """Spend as a person reads it: dollars and cents from ten cents up, three significant figures below.

    A cheap model's call costs a few hundred-thousandths of a dollar, so a fixed number of decimals either
    shows it as $0.000000 or pads every larger amount with noise, and a general format shows it in
    exponent notation ($3e-05), which no one reads as money; the stored value is never rounded. From ten
    cents up, two decimals already carry two significant figures or more, and a third digit ($0.500) reads as
    noise; below, rounding to the cent would erase the difference between two cheap runs ($0.0123 and
    $0.0149 are not both $0.01). Public because two surfaces print spend — a run's summary and the
    agent-facing rendering of its results — and one rule for both is what keeps one amount from reading two
    ways.

    An amount that is not there — a cost no client could price — is :data:`UNPRICED`, never ``$0``, which would say
    the spend was measured and was nothing. Said here rather than by each caller, so every surface that prints a
    possibly-unpriced amount says it in one word.

    Args:
        amount: Dollars, as stored; ``None`` when the amount went unpriced.

    Returns:
        The amount with its dollar sign: ``$0`` for zero, two decimals from ten cents up (``$0.50``,
        ``$1.20``), three significant figures below (``$0.0123``, ``$0.000160``); :data:`UNPRICED` for ``None``.
    """
    if amount is None:
        return UNPRICED
    if amount == 0:
        return "$0"
    if abs(amount) >= 0.1:
        return f"${amount:.2f}"
    decimals = 2 - math.floor(math.log10(abs(amount)))
    return f"${amount:.{decimals}f}"


#: How one measure came out over a run, as the summary states it: ``measured`` (a mean), ``not_a_number`` (it is
#: carried, as words, and never averaged), ``in_confusion_matrix`` (a classifier's cells, counted in the matrix),
#: ``no_successful_results`` (every result carrying it took no turn), ``all_excluded`` (every result carrying it
#: was excluded as a fault of the rig) or ``not_carried`` (no result carries it).
MeasureState = Literal[
    "measured", "not_a_number", "in_confusion_matrix", "no_successful_results", "all_excluded", "not_carried"
]


class MeasureLine(BaseModel):
    """One measure's state over a run, and the line :meth:`EvalSummary.render` prints for it.

    Attributes:
        name: The measure.
        state: How it came out (:data:`MeasureState`).
        text: The line, without indentation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    state: MeasureState
    text: str


#: Said beside a guardrail's level in one run's summary, which has no control to decide it against.
_GUARDRAIL_ALONE = "a guardrail: held, breached or undecided is decided only against a control, in a comparison"


class RunMeasureSummary(BaseModel):
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
        unit: The unit the host declares its values in; empty when it declares none. Its line states the mean, the
            minimum and the maximum on one ruler chosen over the three, as a report states a measure on one ruler.
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
    unit: str = ""


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
            (:func:`~threetears.evals.schema.models.goal_check_proofs_as_read`), so ``proof`` reads
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
        second_judges: How far each second judge asked about this run agreed with its judge on this dimension
            (:func:`~threetears.evals.analysis.inter_judge_agreement`): n, exact agreement and kappa, beside the score
            it qualifies. Empty when no second judge was asked — agreement between judges is then unmeasured.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    scale: RubricScale | None
    n: int
    mean: float | None
    minimum: float | None
    maximum: float | None
    cannot_tell: int
    second_judges: list[InterJudgeDimension] = []


#: How one result came out, as :func:`~threetears.evals.kernel.classify_result` classifies it: graded normally,
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
        stratum: The kind of case it is, as its stored test case declares it (``EvalTestCase.stratum``); ``None``
            when it declares none.
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
    stratum: str | None = None
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
        stratum: str | None = None,
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
            stratum: The stratum the case declares, or ``None``.

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
                f"{name} gave {format_number(value)}"
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
            stratum=stratum,
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
        kind = "" if self.stratum is None else f"{self.stratum}, "
        head = f"case {self.case} ({kind}repeat {self.repeat}, {self.outcome}): answered {_shown_value(self.answer)}"
        if self.expected is not None:
            head += f", expected {_shown_value(self.expected)}"
        lines = [head]
        grades = [
            f"{name} {value if isinstance(value, str | bool) else format_number(value)}"
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


class ProviderUnits(BaseModel):
    """What an external API metered over a run, in its own unit, for one provider.

    Attributes:
        provider: Who was called, as the host names it; ``None`` where the caller could not say.
        unit: The provider's unit (``credits``, ``quota_units``).
        units: How many the results' rows report, summed within this provider and unit only — two providers'
            units are never added together.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str | None
    unit: str
    units: int


#: Why a model-metered role's call carries no dollars: its client priced nothing for it (a local model, a client that
#: prices nothing) — the one way such a row comes to hold none.
MODEL_CALL_UNPRICED = "its client reported no price for them"

#: Why an external call carries no dollars in a run that declared external rates: neither a charge its provider
#: reported nor a rate the run declared for its provider and unit priced it (a call the candidate's own tools made is
#: counted as volume and never priced).
EXTERNAL_CALL_UNPRICED = "neither a charge their provider reported nor a rate the run declared priced them"

#: Why an external call carries no dollars in a run that declared no external rate, whose total sums none.
EXTERNAL_NOT_COUNTED = "the run declared no external rate, so its total sums no external dollars"


class RoleSpend(BaseModel):
    """What one role spent over a run, summed from its results' usage rows.

    **Priced and unpriced are stated side by side, never folded.** A role some of whose calls went unpriced spent the
    priced calls' dollars AND something nobody can put a number on, so it states both: its priced dollars and a count
    of its unpriced calls, with why they are unpriced. A role read as wholly unknown the moment one call went
    unpriced dropped dollars the run's total still counted, so its lines no longer added up to it.

    Attributes:
        role: The role (:data:`~threetears.evals.schema.UsageRole`).
        calls: How many calls its rows count.
        uncounted_rows: How many of its usage rows reported no call count. Their calls are not in ``calls``, and
            how many there were is unknown, never 0.
        cost_usd: What its priced calls cost, summed as priced when they were made; ``None`` when none was priced.
        unpriced_calls: How many of its calls carry no dollars.
        counted: Whether the run's total (each result's blended cost) sums this role. Every model-metered role is;
            the ``external`` role is only in a run that declared external rates.
        units: The ``external`` role's own metering, per provider and unit; empty for every other role.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: UsageRole
    calls: int
    cost_usd: float | None
    unpriced_calls: int = 0
    uncounted_rows: int = 0
    counted: bool = True
    units: list[ProviderUnits] = []

    @property
    def unpriced_because(self) -> str | None:
        """Why its unpriced calls carry no dollars, as true for its run; ``None`` when every call was priced."""
        if not self.unpriced_calls and self.cost_usd is not None:
            return None
        if self.role != "external":
            return MODEL_CALL_UNPRICED
        return EXTERNAL_CALL_UNPRICED if self.counted else EXTERNAL_NOT_COUNTED

    def calls_text(self) -> str:
        """How many calls its rows count, and how many rows counted none: an uncounted row is never read as 0 calls."""
        if not self.uncounted_rows:
            return f"{self.calls} call(s)"
        rows = f"{self.uncounted_rows} usage row(s) that reported no call count"
        return f"{self.calls} call(s) and {rows}" if self.calls else rows

    def money_text(self) -> str:
        """Its dollars over its calls: the priced calls' cost (or :data:`UNPRICED`), and how many are unpriced and why."""
        money = UNPRICED if self.cost_usd is None else dollars_text(self.cost_usd)
        text = f"{money} over {self.calls_text()}"
        if (because := self.unpriced_because) is None:
            return text
        if self.cost_usd is None:
            return f"{text}: {because}"
        return f"{text}, {self.unpriced_calls} of them {UNPRICED}: {because}"

    def line(self) -> str:
        """The role's spend line: its priced dollars and unpriced calls, and whether the run's total leaves it out."""
        outside = "" if self.counted or self.unpriced_because == EXTERNAL_NOT_COUNTED else "; not in the total"
        if self.role != "external":
            return f"{self.role} spend: {self.money_text()}{outside}"
        metered = ", ".join(
            f"{units.units} {units.unit}" + (f" ({units.provider})" if units.provider else "") for units in self.units
        )
        return f"external spend: {metered or 'no units counted'}, {self.money_text()}{outside}"


def _shown_value(value: Any) -> str:
    """A value as a result's line prints it: a string quoted, anything else as written."""
    return repr(value) if isinstance(value, str) or value is None else str(value)


class StratumSummary(BaseModel):
    """One stratum of a run: its cases' results, each measure read over them alone.

    A run's pooled mean is dominated by its commonest kind of case, so a run whose cases declare strata
    (``EvalTestCase.stratum``) is read again per stratum, by the same rules as the pooled figures.

    Attributes:
        stratum: The stratum, as its cases declare it; ``None`` for the cases of the run that declare none.
        n_cases: Cases of the run's frozen case set in it.
        n_results: Results stored over them.
        measures: Each measure over those results alone, as :attr:`EvalSummary.measures` reads it over all of
            them — every one but ``confusion_cell``, whose counts are the run's confusion matrix.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    stratum: str | None
    n_cases: int
    n_results: int
    measures: list[RunMeasureSummary]

    @property
    def too_few(self) -> bool:
        """Whether it holds fewer cases than a stratum needs to be read on its own (:data:`STRATUM_MIN_CASES`)."""
        return self.n_cases < STRATUM_MIN_CASES


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
        strata: The run read again per stratum of its cases (:class:`StratumSummary`), named strata in name order
            then the cases declaring none; empty when none of its cases declares a stratum, so an unstratified
            run reads pooled, with nothing per stratum.
        goal_checks: Each goal-state check the results carry, in the order first met; empty for a run
            with none.
        intent: The template's intent, which a judged run's judge read beside every answer; ``None`` for an
            unjudged run, and for a template edited since the run launched, whose intent is no longer the one
            the judge read.
        intent_source: Where the intent came from, as the caller that wrote the template says it
            (:func:`~threetears.evals.quick.run_eval`: ``"from <candidate>'s docstring"`` or a generic
            default); ``None`` for an intent stated outright, and for a summary read back from the store,
            which keeps the intent but not its source.
        second_judge_calls: How many calls second judges asked about this run made, as the out-of-run ledger records
            them under purpose ``second_judge``. Measurement cost: never in :attr:`spend` or :attr:`total_cost_usd`.
        second_judge_cost_usd: What those calls cost; ``None`` when any went unpriced, and when none was made.
        spend: What each role spent (:class:`RoleSpend`), in :data:`~threetears.evals.schema.UsageRole`'s order, for
            every role the results' usage rows name; empty for a run whose results reported none.
        total_cost_usd: The results' cost summed (each result's blended ``cost_usd``) — what the counted roles'
            dollars add up to, external units aside; ``None`` when any result's cost is unknown (its line then states
            the counted roles' priced dollars and their unpriced calls), and for a run whose results reported no
            spend.
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
        unconfirmed_models: Each role — the candidate, then the judge — some of whose calls' responses named no
            model (:class:`~threetears.evals.analysis.reporting.UnconfirmedModelIdentity`): the record holds only
            the id asked for there, so which model ran is unconfirmed. Empty when every response named its model,
            and for a run that called nothing.
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
    measures: list[RunMeasureSummary]
    confusion: list[ConfusionCount]
    labels: list[LabelStatistics]
    judged: list[DimensionSummary] = []
    strata: list[StratumSummary] = []
    spend: list[RoleSpend] = []
    total_cost_usd: float | None = None
    second_judge_calls: int = 0
    second_judge_cost_usd: float | None = None
    goal_checks: list[GoalCheckSummary] = []
    intent: str | None = None
    intent_source: str | None = None
    errors: list[str]
    stopped_because: str | None = None
    max_cost_usd: float | None = None
    max_cost_usd_origin: CostCapOrigin | None = None
    judge_shares_candidate_model: list[str] = []
    unconfirmed_models: list[UnconfirmedModelIdentity] = []
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

    def role_spend(self, role: UsageRole) -> RoleSpend | None:
        """What one role spent over the run, or ``None`` when its results name no row of it.

        Args:
            role: The role.

        Returns:
            Its :class:`RoleSpend`, or ``None``.
        """
        return next((spend for spend in self.spend if spend.role == role), None)

    @model_validator(mode="before")
    @classmethod
    def _discard_served_words(cls, data: Any) -> Any:
        """Drop the served words a summary's JSON carries: they are re-derived from the fields beside them."""
        return without_computed_field_echo(cls, data)

    @property
    def _judged_in_play(self) -> bool:
        """Whether a judge is in play — the rig a first run can meet, and whose spend a cap bounds.

        With neither a judge nor a cap in play, the rig's exclusions and the cap say nothing a newcomer can act on,
        so they print only when they do.
        """
        judge = self.role_spend("judge")
        return bool(self.judged) or (judge is not None and (judge.calls > 0 or judge.uncounted_rows > 0))

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description="The run's first line: its id, how it ended, the candidate (or arm), its cases and repeats."
    )
    @property
    def headline(self) -> str:
        """The run's line, as :meth:`render` opens with it."""
        who = self.candidate_model if self.arm is None else f"arm {self.arm} (model {self.candidate_model})"
        return f"run {self.run_id} {self.status}: {who} over {self.n_cases} case(s) x k={self.k_runs}"

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description=(
            "How the results came out: scored, then failed by the candidate and excluded as a fault of the rig where "
            "there are some (excluded always once a judge is in play)."
        )
    )
    @property
    def results_text(self) -> str:
        """The results line, without indentation."""
        counts = [f"{self.n_scored} scored"]
        if self.n_candidate_failed:
            counts.append(f"{self.n_candidate_failed} failed by the candidate")
        if self.n_excluded or self._judged_in_play:
            counts.append(f"{self.n_excluded} excluded")
        return f"{self.n_results} result(s): {', '.join(counts)}"

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description="Why a designed stop ended the run short (`stopped: <reason>`); null when nothing stopped it."
    )
    @property
    def stopped_text(self) -> str | None:
        """The stop line, without indentation, or None."""
        return None if self.stopped_because is None else f"stopped: {self.stopped_because}"

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description=(
            "The spend ceiling the run was held to, or that nothing bounded it; null when the summary prints no cap "
            "line — no cap recorded, or none of a cap, a judge or any role's spend in play."
        )
    )
    @property
    def spend_cap_text(self) -> str | None:
        """The spend-cap line, without indentation, or None."""
        if self.max_cost_usd is None and not self._judged_in_play and not self.spend:
            return None
        return _spend_cap_line(self)

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description="Each measure, in `measures` order: its state and the line printed for it."
    )
    @property
    def measure_lines(self) -> list[MeasureLine]:
        """Each measure's state and line."""
        return [self._measure_line(measure) for measure in self.measures]

    def _measure_line(self, measure: RunMeasureSummary) -> MeasureLine:
        """One measure's state, and its line."""
        left_out = _left_out(measure)
        state: MeasureState
        if measure.n == 0 and measure.n_no_turn:
            state, said = "no_successful_results", f"{NO_SUCCESSFUL_RESULTS}{left_out}"
        elif measure.n == 0 and measure.n_faulted:
            state, said = "all_excluded", f"every result carrying it was excluded{left_out}"
        elif measure.n == 0:
            state, said = "not_carried", "no result carries it"
        elif measure.name == CONFUSION_CELL_MEASURE and self.confusion:
            state, said = "in_confusion_matrix", f"n={measure.n}, counted in the confusion matrix below"
        elif measure.mean is None:
            state, said = "not_a_number", f"n={measure.n}, not a number"
        else:
            state = "measured"
            # One ruler for the line's three figures, chosen over all of them: a mean in s beside a max in ms is two
            # rulers read as one.
            figures = [value for value in (measure.mean, measure.minimum, measure.maximum) if value is not None]
            factor, unit = display_scale(figures, measure.unit)

            def stated(value: float | None) -> str:
                return with_unit(None if value is None else value * factor, unit)

            said = (
                f"mean {stated(measure.mean)} (n={measure.n}, "
                f"min {stated(measure.minimum)}, max {stated(measure.maximum)}{left_out})"
                + (f"; {_GUARDRAIL_ALONE}" if measure.guardrail else "")
            )
        return MeasureLine(name=measure.name, state=state, text=f"{measure.name}: {said}")

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description="The confusion matrix, a line per cell (`expected → predicted: count`); empty with `confusion`."
    )
    @property
    def confusion_lines(self) -> list[str]:
        """Each confusion cell's line."""
        return [f"{_shown(cell.expected)} → {_shown(cell.predicted)}: {cell.count}" for cell in self.confusion]

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description="Each label's precision and recall with their counts and intervals, and its F1; empty with `labels`."
    )
    @property
    def label_lines(self) -> list[str]:
        """Each label's line."""
        return [_label_line(statistics) for statistics in self.labels]

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description=(
            "Each stratum's line (its cases and results, and whether it holds too few cases to read alone), each "
            "followed by its measures' lines, indented; empty with `strata`."
        )
    )
    @property
    def stratum_lines(self) -> list[str]:
        """Each stratum's line, then its measures' lines indented beneath it."""
        lines: list[str] = []
        for stratum in self.strata:
            said = f"{NO_STRATUM if stratum.stratum is None else stratum.stratum}: {stratum.n_cases} case(s), "
            said += f"{stratum.n_results} result(s)"
            if stratum.too_few:
                said += f" — {TOO_FEW_CASES} (fewer than {STRATUM_MIN_CASES})"
            lines.append(said)
            lines += [f"  {self._measure_line(measure).text}" for measure in stratum.measures]
        return lines

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description="The intent the judge read, with where it came from; null when the summary carries no intent."
    )
    @property
    def intent_text(self) -> str | None:
        """The intent line, without indentation, or None."""
        if self.intent is None:
            return None
        source = "" if self.intent_source is None else f" ({self.intent_source})"
        return f"intent{source}: {' '.join(self.intent.split())}"

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description="Each judged dimension: its scale, mean and range, and how often the judge could not tell."
    )
    @property
    def dimension_lines(self) -> list[str]:
        """Each judged dimension's line."""
        return [_dimension_line(dimension) for dimension in self.judged]

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description="The warning that a judge graded answers its own model produced; null when none did."
    )
    @property
    def self_judging_warning(self) -> str | None:
        """The self-judging line, or None."""
        if not self.judge_shares_candidate_model:
            return None
        return self_judging_text(self.judge_shares_candidate_model, "the candidate")

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description=(
            "That the record cannot say which model answered some of the candidate's calls or the judge's scores — "
            "their responses named none, so only the id asked for is recorded; null when every response named one."
        )
    )
    @property
    def model_identity_warning(self) -> str | None:
        """The unconfirmed-model line (:func:`~threetears.evals.analysis.reporting.model_identity_disclosure`), or None."""
        return model_identity_disclosure(self.unconfirmed_models)

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description=(
            "Each role's spend line, in `spend` order: what it cost, or why that is unknown, over how many calls — an "
            "external API's units per provider beside it. Empty for a run whose results reported no spend."
        )
    )
    @property
    def spend_lines(self) -> list[str]:
        """Each role's spend line (:meth:`RoleSpend.line`)."""
        return [spend.line() for spend in self.spend]

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description=(
            "The results' total cost — what the role lines add up to — or that it is unknown and why; null for a "
            "run whose results reported no spend."
        )
    )
    @property
    def total_spend_text(self) -> str | None:
        """The total-spend line, without indentation, or None."""
        if not self.spend:
            return None
        if self.total_cost_usd is not None:
            return f"total spend: {dollars_text(self.total_cost_usd)}, the results' cost"
        # Unknown in full: what the counted roles' priced calls cost, which their lines add up to, and how many of
        # their calls are unpriced — never a bare "unknown" that drops the dollars the lines state.
        counted = [spend for spend in self.spend if spend.counted]
        priced = [spend.cost_usd for spend in counted if spend.cost_usd is not None]
        unpriced = sum(spend.unpriced_calls for spend in counted)
        stated = f"{dollars_text(math.fsum(priced))} over the priced calls" if priced else "no call priced"
        return f"total spend: unknown in full: {stated}, plus {unpriced} call(s) {UNPRICED}"

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description="Each goal check's pass count and, unless proven, why it is not a measurement."
    )
    @property
    def goal_check_lines(self) -> list[str]:
        """Each goal check's line (:meth:`GoalCheckSummary.line`)."""
        return [goal.line() for goal in self.goal_checks]

    @computed_field(  # type: ignore[prop-decorator]  # pydantic's documented form; mypy cannot type a decorator above @property
        description="The whole summary as `render()` prints it for a terminal: every line above, indented."
    )
    @property
    def text(self) -> str:
        """:meth:`render`'s text."""
        return self.render()

    def render(self) -> str:
        """The summary as a few lines of text for a terminal: the served lines, in order, indented.

        The results line counts the failures and the rig's exclusions only when there are some, or a judge is in
        play; the spend-cap line prints only when a cap bounded the run, a judge ran, or any role reported
        spend — a first run of a free function is not handed the vocabulary of a rig it does not have.

        Returns:
            The text, without a trailing newline.
        """
        lines = [self.headline, f"  {self.results_text}"]
        lines += [f"  {line}" for line in (self.stopped_text, self.spend_cap_text) if line is not None]
        lines += [f"  {measure.text}" for measure in self.measure_lines]
        if self.confusion:
            lines.append("  confusion (expected → predicted):")
            lines += [f"    {line}" for line in self.confusion_lines]
            lines.append("  per label:")
            lines += [f"    {line}" for line in self.label_lines]
        if self.strata:
            lines.append("  by stratum:")
            lines += [f"    {line}" for line in self.stratum_lines]
        if self.intent_text is not None:
            lines.append(f"  {self.intent_text}")
        lines += [f"  {line}" for line in self.dimension_lines]
        if self.self_judging_warning is not None:
            lines.append(f"  {self.self_judging_warning}")
        if self.model_identity_warning is not None:
            lines.append(f"  {self.model_identity_warning}")
        lines += [f"  {line}" for line in self.spend_lines]
        if self.total_spend_text is not None:
            lines.append(f"  {self.total_spend_text}")
        lines += [f"  {line}" for line in self.goal_check_lines]
        lines += [f"  error: {error}" for error in self.errors]
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

    A cap counts the spend results report, so a capped run none of whose results reported any says that its cap
    had nothing to count.
    """
    if summary.max_cost_usd_origin == "uncapped":
        return f"spend cap: {UNCAPPED_SPEND}"
    if summary.max_cost_usd is None:
        return None
    line = f"spend cap: {dollars_text(summary.max_cost_usd)} for this run"
    if not summary.spend:
        line += "; it counts only spend a result reports, and no result reported any"
    return line


def _left_out(measure: RunMeasureSummary) -> str:
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
    shown = f"{name} {format_number(rate)} ({hits}/{n}"
    if cases < n:
        shown += f" over {cases} case{'' if cases == 1 else 's'}"
    if interval is not None:
        shown += f", {INTERVAL_LEVEL:.0%} CI {format_number(interval[0])}-{format_number(interval[1])}"
    return shown + ")"


def _dimension_line(dimension: DimensionSummary) -> str:
    """One judged dimension's line: its scale, its mean and range, and how often the judge could not tell."""
    scale = {"ordinal": "judged 1-5", "pass_fail": "judged pass/fail", None: "judged"}[dimension.scale]
    line = f"{dimension.name} ({scale}): "
    if dimension.mean is None or dimension.minimum is None or dimension.maximum is None:
        line += "no result carries a score"
    else:
        line += (
            f"mean {format_number(dimension.mean)} (n={dimension.n}, "
            f"min {format_number(dimension.minimum)}, max {format_number(dimension.maximum)})"
        )
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
    f1 = "f1 none" if statistics.f1 is None else f"f1 {format_number(statistics.f1)}"
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
    run = host.storage.load_eval_run(run_id, scope_id)
    if run is None:
        raise NotFoundError("run", run_id, remedy=FIND_RUN)
    results = host.storage.query_eval_results_by_run(run_id, scope_id)
    outcomes = [classify_result(result) for result in results]
    measures = _measure_summaries(host, results)
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
    template = None
    if run.judge_model is not None and run.template_id is not None:
        template = host.storage.load_template(run.template_id, scope_id)
    # A second judge's spend is the out-of-run ledger's, stamped with the run — never a result's usage row.
    second_rows = [
        row for row in host.storage.query_out_of_run_spend(scope_id, purpose="second_judge") if row.run_id == run.id
    ]
    second_costs = [row.cost_usd for row in second_rows]
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
        strata=_strata(host, run, results),
        spend=_role_spend(results),
        total_cost_usd=_total_cost(results),
        second_judge_calls=len(second_rows),
        second_judge_cost_usd=(
            math.fsum(cost for cost in second_costs if cost is not None)
            if second_rows and None not in second_costs
            else None
        ),
        goal_checks=_goal_checks(results, run),
        # Templates are edited in place: one edited since the launch no longer holds what the judge read.
        intent=template.intent if template is not None and template.updated_at <= run.created_at else None,
        errors=errors,
        stopped_because=run.cancellation_reason or run.budget_stop_reason,
        max_cost_usd=run.max_cost_usd,
        max_cost_usd_origin=run.max_cost_usd_origin,
        judge_shares_candidate_model=_self_judging(run, results),
        unconfirmed_models=unconfirmed_model_identities([run], results),
    )


def _measure_summaries(host: EvalHost, results: list[EvalResult]) -> list[RunMeasureSummary]:
    """Each measure the host declares over the results carrying it, then ``match`` and ``confusion_cell`` where carried."""
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
            RunMeasureSummary(
                name=name,
                n=len(carried),
                mean=sum(values) / len(values) if values else None,
                minimum=min(values) if values else None,
                maximum=max(values) if values else None,
                n_no_turn=sum(1 for result in left if classify_result(result) is ResultOutcome.CANDIDATE_FAIL),
                n_faulted=sum(1 for result in left if classify_result(result) is ResultOutcome.INFRA_EXCLUDE),
                guardrail=describe_measure(name, host.profile.measures).guardrail,
                unit=describe_measure(name, host.profile.measures).unit or "",
            )
        )
    return measures


def _strata(host: EvalHost, run: EvalRun, results: list[EvalResult]) -> list[StratumSummary]:
    """The run read again per stratum of its cases — or nothing, when none of its cases declares one.

    The strata partition the run: the cases declaring none are their own stratum, last, so they add up to it.
    """
    declared = {case.id: case.stratum for case in host.storage.load_case_strata(list(run.test_case_ids), run.scope_id)}
    if not any(stratum is not None for stratum in declared.values()):
        return []
    cases: dict[str | None, list[str]] = {}
    for case_id in run.test_case_ids:
        cases.setdefault(declared.get(case_id), []).append(case_id)
    strata = []
    for stratum in sorted(cases, key=lambda name: (name is None, name or "")):
        members = set(cases[stratum])
        carried = [result for result in results if result.test_case_id in members]
        strata.append(
            StratumSummary(
                stratum=stratum,
                n_cases=len(members),
                n_results=len(carried),
                measures=[
                    measure for measure in _measure_summaries(host, carried) if measure.name != CONFUSION_CELL_MEASURE
                ],
            )
        )
    return strata


def _role_spend(results: list[EvalResult]) -> list[RoleSpend]:
    """What each role the results' usage rows name spent, in :data:`~threetears.evals.schema.UsageRole`'s order.

    A role's dollars are its priced rows' costs summed, and its unpriced calls are counted beside them, so a role
    partly priced states what was priced rather than reading wholly unknown: the run's total
    (:func:`~threetears.evals.kernel.usage_capture.blended_cost`, per result) sums exactly the counted roles' priced
    rows, and the role lines add up to it. Whether a role is counted is the results' own composition
    (``cost_roles``): the ``external`` role is in it only where the run declared external rates. Its units are summed
    per provider and unit, never across two.
    """
    spent: list[RoleSpend] = []
    for role in get_args(UsageRole):
        carrying = [result for result in results if any(row.role == role for row in result.usage)]
        rows = [row for result in carrying for row in result.usage if row.role == role]
        if not rows:
            continue
        priced = [row.cost_usd for row in rows if row.cost_usd is not None]
        metered: dict[tuple[str | None, str], int] = {}
        for row in rows:
            if row.provider_unit is not None and row.provider_units is not None:
                key = (row.provider, row.provider_unit)
                metered[key] = metered.get(key, 0) + row.provider_units
        spent.append(
            RoleSpend(
                role=role,
                calls=sum(row.call_count or 0 for row in rows),
                cost_usd=math.fsum(priced) if priced else None,
                unpriced_calls=sum(row.call_count or 0 for row in rows if row.cost_usd is None),
                uncounted_rows=sum(1 for row in rows if row.call_count is None),
                counted=all(role in result.cost_roles for result in carrying),
                units=[
                    ProviderUnits(provider=provider, unit=unit, units=units)
                    for (provider, unit), units in sorted(
                        metered.items(), key=lambda item: (item[0][0] or "", item[0][1])
                    )
                ],
            )
        )
    return spent


def _total_cost(results: list[EvalResult]) -> float | None:
    """The results' cost summed, or ``None`` when any result's is unknown or no result's rows observed spend."""
    if not any(row.cost_usd is not None for result in results for row in result.usage):
        return None
    costs = [result.cost_usd for result in results]
    return None if None in costs else math.fsum(cost for cost in costs if cost is not None)


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
    agreement = inter_judge_agreement(results)
    return [
        DimensionSummary(
            name=name,
            scale=scales.get(name),
            n=len(values),
            mean=sum(values) / len(values) if values else None,
            minimum=min(values) if values else None,
            maximum=max(values) if values else None,
            cannot_tell=cannot_tell[name],
            second_judges=[row for row in agreement.dimensions if row.rubric_dim == name],
        )
        for name, values in scores.items()
    ]


def _goal_checks(results: list[EvalResult], run: EvalRun) -> list[GoalCheckSummary]:
    """Each goal-state check the results carry, counted as every per-check rate counts it, in the order first met.

    Each carries the proof its run froze at launch, as read under the current proof rules
    (:func:`~threetears.evals.schema.models.goal_check_proofs_as_read`); a check the run recorded none for is
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
    "UNPRICED",
    "CaseOutcome",
    "CaseResult",
    "DimensionSummary",
    "EvalSummary",
    "GoalCheckSummary",
    "JudgeGrade",
    "MeasureLine",
    "MeasureState",
    "ProviderUnits",
    "RoleSpend",
    "RunMeasureSummary",
    "dollars_text",
    "self_judging_text",
    "summarize_run",
]
