"""A regression found the moment a run completes, handed to the host's sink: :class:`RegressionWatch`.

:func:`~threetears.evals.ops.scope_history` finds a regression only when someone reads for it. A
:class:`RegressionWatch` reads for it when a run ends: handed to a
:class:`~threetears.evals.run.launch.LaunchHost` as its ``on_run_end`` hook, it checks each measure the
host names against the completed run's contestant history, and hands every regression on the step INTO that
run to the host's :class:`RegressionSink`. The engine delivers nothing itself: no watch, or a watch whose
sink drops what it is given, and nobody hears.

**The test is the history's own.** Each measure is read through :func:`~threetears.evals.ops.scope_history`
with the watch's thresholds and its default ``completed`` population, so an alert is exactly the flag that
lens would show on that point (:class:`~threetears.evals.analysis.RegressionFlag`, carried whole: its label,
delta, p, test and thresholds). Only a ``regressed`` label is delivered.

**Which runs are checked.** Only a run whose recorded status is ``completed``: the history series completed
runs, so a run that ended any other way is not a point on it and has no step to check. Only the step into
the completed run is delivered — the step out of it into a later run, where one exists, was that later run's
to report.

**Which measures fire.** A code-graded measure (:func:`~threetears.evals.kernel.metrics.is_code_graded`)
fires on its flag. A judged measure — the composite, either dual axis, or a host measure whose family a judge
grades — fires only when every judge behind the step's scores, on both runs, is ``calibrated``
(:mod:`~threetears.evals.kernel.evidence_tiers`, read over the scope's calibration ratings and judge
repeats). An alert on an uncalibrated score is worse than none, so the check withholds it and logs that it did.
A judge-graded host measure names no rubric dimension the engine can look a judge up by, so it never fires.

**The history is read for the run's own subject.** Every series the watch alerts on is one of the run's
contestants, so it reads that subject's history: in a scope past the read tier's budget
(:data:`~threetears.evals.analysis.READ_TIER_ROW_BUDGET`) the run is still checked when its subject fits, reading
that subject's runs alone. A judged step's judges are then tiered on that subject's results alone, and its alert
says so (``narrowed_read_disclosure``). When the subject alone is past the budget, the history is refused and the
watch checks nothing on the run: :meth:`RegressionWatch.check` raises the refusal to its caller, and the hook, which
nobody waits on, logs at ERROR that nothing was checked and that nothing narrower exists (#996). The run is
untouched.

**A sink that fails changes nothing.** Each alert is delivered on its own: a sink that raises is logged and
the next alert is still delivered, and neither can reach the run, whose status was recorded before any check
began (:class:`~threetears.evals.run.jobs.RunEndHook`).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import Field

from threetears.evals.analysis.agreement import (
    JudgeKey,
    judge_agreement,
    judge_evidence_tiers,
    judge_key,
    judge_self_agreement,
    tier_for_judges,
)
from threetears.evals.analysis.reads import hold_to_read_budget, narrowed_read_disclosure
from threetears.evals.analysis.reporting import METRIC_COMPOSITE, METRIC_OUTCOME, METRIC_TRANSCRIPT
from threetears.evals.analysis.lenses.history import (
    HistoryError,
    HistoryResult,
    MeasureSeries,
    RegressionFlag,
    compute_history,
)
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.kernel.evidence_tiers import JudgedEvidenceTier
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.metrics import GradedBy, is_code_graded
from threetears.evals.schema.models import EvalResult
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.ops.lenses import scope_history
from threetears.evals.run.reads import list_runs
from threetears.evals.schema.logs import contain, get_logger

log = get_logger(__name__)

#: The history measures a judge's scores make, each read off the dimensions named here: the composite off every
#: dimension the result was scored on (``None``), a dual axis off its own.
_JUDGED_MEASURES: dict[str, str | None] = {
    METRIC_COMPOSITE: None,
    METRIC_TRANSCRIPT: METRIC_TRANSCRIPT,
    METRIC_OUTCOME: METRIC_OUTCOME,
}


class RegressionAlert(EvalBaseModel):
    """One regression the watch found on the step into a completed run: who, which measure, which step, and why.

    ``regression`` is the history's own flag for the step, whole, so the verdict arrives with the test and the
    thresholds it was decided under. ``evidence_tier`` is the weakest tier among the judges behind the step's
    scores on a judged measure (always ``calibrated``: nothing weaker is delivered), and ``None`` on a
    code-graded one, which has no judge.
    """

    scope_id: str = Field(description="The scope the run is in.")
    run_id: str = Field(description="The completed run whose step regressed.")
    previous_run_id: str = Field(description="The run the step is measured from: the series' preceding point.")
    subject_id: str = Field(description="The contestant's subject.")
    subject_label: str = Field(description="The subject as people read it.")
    variant_key: str = Field(description="The contestant's identity within the subject.")
    variant_identity_version: int = Field(description="The identity predicate version that minted the key.")
    model: str = Field(description="The contestant's model label.")
    measure: str = Field(description="The measure that regressed, as the history names it.")
    regression: RegressionFlag = Field(description="The history's verdict on the step, with its test and thresholds.")
    graded_by: GradedBy = Field(description="`code` when code produced the measure; `judge` when a judge did.")
    evidence_tier: JudgedEvidenceTier | None = Field(
        description="On a judged measure, the weakest tier among the step's judges; None on a code-graded one."
    )
    narrowed_read_disclosure: str | None = Field(
        default=None,
        description="On a judged measure whose history was read for the run's subject alone (its scope past the read "
        "tier's budget): that the step's judges were tiered on that subject's results alone. None otherwise.",
    )


class RegressionSink(Protocol):
    """The host's delivery of a regression: a page, a message, a ticket. The engine ships none."""

    async def __call__(self, alert: RegressionAlert) -> None:
        """Deliver one regression.

        :param alert: What regressed, on which step, and the verdict behind it.
        :ptype alert: RegressionAlert
        """
        ...  # pragma: no cover — protocol


def _graded_by(history: HistoryResult, host: EvalHost) -> GradedBy:
    """Who produced the measure's numbers: a judge for the judged history measures, else the measure's family."""
    if history.metric in _JUDGED_MEASURES:
        return "judge"
    return "code" if is_code_graded(history.measure, host.profile.measures) else "judge"


def _step_judges(metric: str, results: Iterable[EvalResult]) -> set[JudgeKey]:
    """Every judge behind ``metric`` on ``results``: every scored dimension for the composite, else its own axis."""
    dimension = _JUDGED_MEASURES.get(metric)
    keys: set[JudgeKey] = set()
    for result in results:
        dims = [score.dim for score in result.judge_scores()] if dimension is None else [dimension]
        keys.update(key for dim in dims if (key := judge_key(result, dim)) is not None)
    return keys


@dataclass(frozen=True, kw_only=True)
class RegressionWatch:
    """Checks a completed run's measures against its contestant's history, and delivers each regression.

    A :class:`~threetears.evals.run.jobs.RunEndHook`: hand it to the
    :class:`~threetears.evals.run.launch.LaunchHost` that launches the host's runs (``on_run_end=``). See the
    module docstring for which runs, steps and measures fire.

    :ivar host: The host whose store and vocabulary are read — the launching host's own ``eval_host``.
    :ivar sink: Where each regression is delivered.
    :ivar measures: The measures to track, each one :func:`~threetears.evals.ops.scope_history` accepts. Refused
        when empty or when one cannot be seriesed, here rather than at the first completed run.
    :ivar min_absolute_change: The smallest move a regression counts, in the measure's unit, as the history takes it.
    :ivar min_relative_change: The smallest move relative to the baseline a regression counts, as a fraction.
    """

    host: EvalHost
    sink: RegressionSink
    measures: Sequence[str]
    min_absolute_change: float = 0.0
    min_relative_change: float = 0.0

    def __post_init__(self) -> None:
        """Refuse a watch that tracks nothing, a measure the history cannot series, or a negative threshold.

        :raises ValueError: As above.
        """
        if not self.measures:
            raise ValueError("a regression watch tracks at least one measure; name the measures to check")
        if self.min_absolute_change < 0 or self.min_relative_change < 0:
            raise ValueError("a regression threshold is a size of move, so it cannot be negative")
        object.__setattr__(self, "measures", tuple(self.measures))
        for measure in self.measures:
            try:
                compute_history([], [], metric=measure, archived_run_ids=None, profile=self.host.profile)
            except HistoryError as e:
                raise ValueError(f"a regression watch cannot track {measure!r}: {e}") from e

    async def __call__(self, run_id: str, scope_id: str, status: str) -> None:
        """Check a run that ended, and deliver each regression found, each on its own.

        :param run_id: The run.
        :ptype run_id: str
        :param scope_id: Its scope.
        :ptype scope_id: str
        :param status: The terminal status recorded. Anything but ``completed`` is not a point on the history.
        :ptype status: str
        """
        if status != "completed":
            return
        try:
            alerts = await run_blocking(self.host.blocking_executor, self.check, run_id, scope_id)
        except ValidationFailedError as refused:
            # The history refused the read: no measure of this run was checked, and nobody waits on a run-end hook,
            # so the operator hears it here, at ERROR, every time. In the watch's own words: the history's remedy
            # names a filter the watch already applies, and nothing narrower exists for it.
            if "result_budget" in refused.details:
                log.error(
                    "eval.regression watch checked nothing on run=%s scope_id=%s: the history it reads for this run "
                    "holds about %s results, past READ_TIER_ROW_BUDGET=%s, so no regression on this run was looked "
                    "for; there is nothing an operator can narrow until the read tier is computed in the store (#996)",
                    run_id,
                    scope_id,
                    refused.details.get("results_returned", refused.details.get("results_estimated")),
                    refused.details["result_budget"],
                )
            else:
                log.error(
                    "eval.regression watch checked nothing on run=%s scope_id=%s: its history was refused: %s",
                    run_id,
                    scope_id,
                    refused.message,
                )
            return
        for alert in alerts:
            try:
                await self.sink(alert)
            except (
                Exception
            ) as undelivered:  # prawduct:ok-broad-except — a sink's failure must not reach the run or the next alert
                # NOSILENT: the run is completed and its record final; a sink failing to deliver is the host's
                # delivery breaking, not the run, and an error_details entry would count it as one of the run's faults.
                contain(
                    undelivered,
                    log=log,
                    what=(
                        f"eval.regression sink failed delivering run={run_id} measure={alert.measure}; "
                        "the next alert is still delivered"
                    ),
                    level="error",
                )

    def check(self, run_id: str, scope_id: str) -> list[RegressionAlert]:
        """The regressions on the step into ``run_id``, over every tracked measure, that may fire.

        Synchronous and side-effect free — it reads the store and delivers nothing — so a host can ask it
        directly of a run that has already ended.

        :param run_id: The run whose step to check.
        :ptype run_id: str
        :param scope_id: Its scope.
        :ptype scope_id: str
        :return: One alert per tracked measure and contestant whose step into the run reads ``regressed`` and may
            fire, in the order the measures were named.
        :rtype: list[RegressionAlert]
        :raises ValidationFailedError: The history was refused: the run's subject alone is past the read tier's
            budget; or, for a judged step, the subject's results its evidence tiers read are past it.
        """
        alerts: list[RegressionAlert] = []
        tiers = None
        results: list[EvalResult] | None = None
        # Every series the watch alerts on is one of this run's contestants, all of the run's own subject, so the
        # history is read for that subject: a scope past the read tier's budget is still checked wherever the subject
        # fits. A run no longer stored reads the whole scope's history.
        run = self.host.storage.load_eval_run(run_id, scope_id)
        subject_id = run.subject_snapshot.subject_id if run is not None else None
        for measure in self.measures:
            history = scope_history(
                self.host,
                scope_id,
                metric=measure,
                min_absolute_change=self.min_absolute_change,
                min_relative_change=self.min_relative_change,
                subject_id=subject_id,
            )
            graded_by = _graded_by(history, self.host)
            for series in history.series:
                step = _step_into(series, run_id)
                if step is None:
                    continue
                previous_run_id, flag = step
                tier: JudgedEvidenceTier | None = None
                if graded_by == "judge":
                    if results is None:
                        # The read the history made: the whole scope when it fit the budget, else the subject's runs
                        # alone, which the history found within it.
                        results = self._judged_results(scope_id, subject_id, narrowed=history.narrowed_read_disclosure)
                    if tiers is None:
                        tiers = judge_evidence_tiers(
                            judge_agreement(self.host.storage.query_calibration_ratings(scope_id), results),
                            judge_self_agreement(results),
                            _step_judges(METRIC_COMPOSITE, results),
                        )
                    step_results = [
                        result
                        for result in results
                        if result.eval_run_id in (previous_run_id, run_id) and result.variant_key == series.variant_key
                    ]
                    judges = _step_judges(history.metric, step_results) if history.metric in _JUDGED_MEASURES else set()
                    tier = tier_for_judges(tiers, judges) if judges else "undetermined"
                    if tier != "calibrated":
                        log.info(
                            "eval.regression withheld run=%s measure=%s: its judges stand on %s, not calibrated",
                            run_id,
                            history.metric,
                            tier,
                        )
                        continue
                alerts.append(
                    RegressionAlert(
                        scope_id=scope_id,
                        run_id=run_id,
                        previous_run_id=previous_run_id,
                        subject_id=series.subject_id,
                        subject_label=series.subject_label,
                        variant_key=series.variant_key,
                        variant_identity_version=series.variant_identity_version,
                        model=series.model,
                        measure=history.metric,
                        regression=flag,
                        graded_by=graded_by,
                        evidence_tier=tier,
                        narrowed_read_disclosure=(
                            narrowed_read_disclosure(subject_id, tiers=True)
                            if graded_by == "judge" and history.narrowed_read_disclosure and subject_id is not None
                            else None
                        ),
                    )
                )
        return alerts

    def _judged_results(self, scope_id: str, subject_id: str | None, *, narrowed: str | None) -> list[EvalResult]:
        """The results a judged step's tiers are read over: the scope's, or, when the history read one subject's runs
        alone (``narrowed`` is its disclosure), that subject's runs' results, every status and archived ones too."""
        if narrowed is None or subject_id is None:
            # The scope the history read whole: its listing's estimate and what its read returned were both within
            # the read tier's budget, or the history would have read the subject's runs instead (below) or refused.
            return self.host.storage.query_eval_results(scope_id)
        runs = list_runs(self.host, scope_id, include_archived=True)
        subject_runs = [run for run in runs if run.subject_snapshot.subject_id == subject_id]
        # The subject's runs the history read: held to the budget here too, off the listing above and before any
        # result is read, so the tiers are bounded by their own read and never by the history's alone.
        hold_to_read_budget(_JUDGED_TIERS, scope_id, subject_runs, how=_NARROW_THE_JUDGED_TIERS)
        results = self.host.storage.query_eval_results_by_runs([run.id for run in subject_runs], scope_id)
        hold_to_read_budget(_JUDGED_TIERS, scope_id, subject_runs, how=_NARROW_THE_JUDGED_TIERS, returned=len(results))
        return results


#: The read a judged step's evidence tiers are refused as past the read budget.
_JUDGED_TIERS = "regression_watch_judged_tiers"

#: How that read narrows: nothing narrower than the run's subject exists for a watch.
_NARROW_THE_JUDGED_TIERS = (
    "nothing narrower than the run's own subject exists for a regression watch (it reads every run of that subject, "
    "archived ones too); delete that subject's runs it need not compare, or wait for the store to compute the read "
    "tier (#996)"
)


def _step_into(series: MeasureSeries, run_id: str) -> tuple[str, RegressionFlag] | None:
    """The preceding run and the regressed flag on the point ``run_id`` holds in ``series``, or None."""
    for index, point in enumerate(series.points):
        if point.run_id != run_id:
            continue
        if index == 0 or point.regression is None or point.regression.label != "regressed":
            return None
        return series.points[index - 1].run_id, point.regression
    return None


__all__ = ["RegressionAlert", "RegressionSink", "RegressionWatch"]
