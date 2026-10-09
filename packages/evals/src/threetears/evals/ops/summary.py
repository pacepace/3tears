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

from pydantic import BaseModel, ConfigDict

from threetears.evals.analysis.confusion import ConfusionCount, LabelStatistics, confusion_matrix, label_statistics
from threetears.evals.analysis.stats import INTERVAL_LEVEL
from threetears.evals.analysis.surface_table import NO_SUCCESSFUL_RESULTS
from threetears.evals.contracts import (
    CONFUSION_CELL_MEASURE,
    MATCH_MEASURE,
    EvalResult,
    ResultOutcome,
    RubricScale,
    UsageRole,
    classify_result,
    counted_goal_verdicts,
    delivered_a_turn,
)
from threetears.evals.contracts.host import EvalHost
from threetears.evals.contracts.metrics import describe_measure, summary_population
from threetears.evals.contracts.usage_capture import blended_cost
from threetears.evals.run import get_run, list_results


def _dollars(amount: float) -> str:
    """Spend as a person reads it: cents for whole calls' worth, three significant figures below a cent.

    A cheap model's call costs a few hundred-thousandths of a dollar, so a fixed number of decimals either
    shows it as $0.000000 or pads every larger amount with noise; the stored value is never rounded.
    """
    if amount == 0:
        return "$0"
    decimals = max(2, 2 - math.floor(math.log10(abs(amount))))
    return f"${amount:.{decimals}f}"


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
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    n: int
    mean: float | None
    minimum: float | None
    maximum: float | None
    n_no_turn: int = 0
    n_faulted: int = 0


class GoalCheckSummary(BaseModel):
    """One goal-state check over a run's results.

    Attributes:
        check: The check, as the template states it.
        passed: How many results it counts as passed.
        n: How many results count for it: every result carrying it but those excluded as a fault of the rig.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    check: str
    passed: int
    n: int


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


class EvalSummary(BaseModel):
    """One run, summarised.

    Attributes:
        run_id: The run.
        scope_id: The scope it is stored in.
        template_id: The template it ran; ``None`` for an ad-hoc run of explicit cases.
        candidate_model: The arm's candidate model.
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
        labels: Each label's precision, recall and F1 from that matrix, by label; empty with it.
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
        errors: Each failed or excluded result's error, prefixed by its case, then the run's own.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    scope_id: str
    template_id: str | None
    candidate_model: str
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

    def render(self) -> str:
        """The summary as a few lines of text for a terminal.

        Returns:
            The text, without a trailing newline.
        """
        lines = [
            f"run {self.run_id} {self.status}: {self.candidate_model} over {self.n_cases} case(s) x k={self.k_runs}",
            f"  {self.n_results} result(s): {self.n_scored} scored, {self.n_candidate_failed} failed by the "
            f"candidate, {self.n_excluded} excluded",
        ]
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
        if self.judged:
            spend = (
                "unknown: a judge call went unpriced" if self.judge_cost_usd is None else _dollars(self.judge_cost_usd)
            )
            lines.append(f"  judge spend: {spend} over {self.judge_calls} call(s)")
        if self.candidate_calls:
            spend = (
                "unknown: a candidate call went unpriced"
                if self.candidate_cost_usd is None
                else _dollars(self.candidate_cost_usd)
            )
            lines.append(f"  candidate spend: {spend} over {self.candidate_calls} call(s)")
        lines.extend(f"  goal check {goal.check}: passed {goal.passed}/{goal.n}" for goal in self.goal_checks)
        lines.extend(f"  error: {error}" for error in self.errors)
        return "\n".join(lines)


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


def _proportion(name: str, rate: float | None, hits: int, n: int, interval: tuple[float, float] | None) -> str:
    if rate is None:
        return f"{name} none (n=0)"
    shown = f"{name} {rate:.3g} ({hits}/{n}"
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
        "precision", statistics.precision, statistics.correct, statistics.predicted, statistics.precision_interval
    )
    recall = _proportion(
        "recall", statistics.recall, statistics.correct, statistics.expected, statistics.recall_interval
    )
    f1 = "f1 none" if statistics.f1 is None else f"f1 {statistics.f1:.3g}"
    return f"{_shown(statistics.label)}: {precision}, {recall}, {f1}"


def summarize_run(host: EvalHost, run_id: str, scope_id: str) -> EvalSummary:
    """Summarise one stored run and its results.

    Args:
        host: The host whose store holds the run, and whose measures are summarised.
        run_id: The run.
        scope_id: The scope it lives in.

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
            )
        )
    errors = [
        f"case {result.test_case_id}: {error}"
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
        labels=label_statistics(confusion),
        judged=_judged_dimensions(results),
        judge_calls=sum(row.call_count or 0 for row in judge_rows),
        judge_cost_usd=blended_cost(judge_rows, _JUDGE_ROLE) if judge_rows else None,
        candidate_calls=sum(row.call_count or 0 for row in candidate_rows),
        candidate_cost_usd=blended_cost(candidate_rows, _CANDIDATE_ROLE) if candidate_rows else None,
        goal_checks=_goal_checks(results),
        # Templates are edited in place: one edited since the launch no longer holds what the judge read.
        intent=template.intent if template is not None and template.updated_at <= run.created_at else None,
        errors=errors,
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


def _goal_checks(results: list[EvalResult]) -> list[GoalCheckSummary]:
    """Each goal-state check the results carry, counted as every per-check rate counts it, in the order first met."""
    counted: dict[str, list[bool]] = {}
    for result in results:
        for outcome, passed in counted_goal_verdicts(result) or []:
            counted.setdefault(outcome.expression, []).append(passed)
    return [GoalCheckSummary(check=check, passed=sum(verdicts), n=len(verdicts)) for check, verdicts in counted.items()]


__all__ = ["DimensionSummary", "EvalSummary", "GoalCheckSummary", "MeasureSummary", "summarize_run"]
