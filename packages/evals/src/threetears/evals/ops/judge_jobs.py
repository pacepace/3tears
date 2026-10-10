"""A second judge's pass, a drift check and a judge temperature comparison, each started as a job.

Each re-judges a finished run's stored evidence and makes many judge calls: one per scored dim of every result asked
about for a second judge (each with its parse retries), every selected dim ``repeats`` times at each of two
temperatures for a comparison. A host that budgets a call's wall time by the model requests it makes while its
caller waits cannot hold that inside one call, so each follows the job contract (:mod:`threetears.evals.ops.jobs`)
as :func:`~threetears.evals.ops.analysis_generate` does: the call decides, the job spends.

**Every refusal is the call's, by the spend's own code.** The call runs the work's own estimate
(:func:`~threetears.evals.run.estimate_second_judge`, :func:`~threetears.evals.run.estimate_judge_temperature_comparison`)
— the same collection, sample or selection, calls and admission the spend makes — and refuses on what it refuses:
an unknown run, one still running, nothing eligible, the #633 reproducibility checks, calls the out-of-run cap
cannot admit. No model request is made in the call; the job starts only once nothing stands in its way. The job then
runs the spend itself (:func:`~threetears.evals.run.ask_second_judge`,
:func:`~threetears.evals.run.judge_at_two_temperatures`), which admits its calls again before the first: a run that
changed between the call and the job is refused there, and the job ends ``failed`` saying why, nothing spent.

**The job's record says how it ended, and what it handed back** (:class:`~threetears.evals.kernel.EvalJudgeJob`,
``judge:<id>``): written when it starts, after each result is asked, and when it ends however it ends. ``job_poll``
on an ended job reads the record into what the action used to return in the call — :class:`SecondJudgeRead` or
:class:`~threetears.evals.analysis.JudgeTemperatureComparison` — and ``job_cancel`` stops it before its next judge
call: what it paid for stays on the ledger, and the partial report says it was cancelled.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from threetears.evals.kernel.errors import StorageError, ValidationFailedError
from threetears.evals.kernel.judge_jobs import EvalJudgeJob, JudgeJobAction
from threetears.evals.kernel.judge_second import SecondJudgeReport
from threetears.evals.kernel.judge_temperature import (
    DEFAULT_TEMPERATURE_REPEATS,
    JudgeTemperatureAnswers,
    TemperatureSelection,
)
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.jobs import JobHandle, JobsStarted, judge_job_id, judge_key
from threetears.evals.run.jobs import JOB_TIMEOUT_CAP_S
from threetears.evals.run.judge_second import DEFAULT_SECOND_JUDGE_SEED, ask_second_judge, estimate_second_judge
from threetears.evals.run.judge_temperature import estimate_judge_temperature_comparison, judge_at_two_temperatures
from threetears.evals.schema.models import SecondJudge, utc_now_iso
from threetears.observe import get_logger

log = get_logger(__name__)


async def judge_second(
    host: OpsHost,
    run_id: str,
    scope_id: str,
    *,
    judge: SecondJudge,
    sample_fraction: float = 1.0,
    seed: int = DEFAULT_SECOND_JUDGE_SEED,
    result_ids: list[str] | None = None,
) -> JobsStarted:
    """Check a second judge's pass over a finished run, admit its calls against the host's out-of-run cap, then start it as a job.

    The pass asks the second judge about a seeded share of the run's judged results, records each answer beside the
    first score it pairs with, and ledgers every call under purpose ``second_judge`` — measurement cost on its own
    line, never the candidate's. The scores the run was judged with are never changed. The call makes no model
    request; ``job_poll`` on the ended job carries the :class:`SecondJudgeRead` (the pass, and the agreement and drift
    its pairs read).

    Args:
        host: The host: its store, its judge clients, its out-of-run cap and the job manager the task runs under.
        run_id: The finished run.
        scope_id: The scope it lives in.
        judge: The second judge: its model, its prompts, its temperature.
        sample_fraction: The share of the run's judgeable results to ask about, in ``(0, 1]``.
        seed: The seed the share is drawn with.
        result_ids: The results to draw from; ``None`` for every result of the run.

    Returns:
        The pass's one job (``judge:<id>``), whose target is the run.

    Raises:
        NotFoundError: No run with that id, or a record it names does not load.
        ValidationFailedError: The run cannot be asked about, or its calls are priced above the cap or cannot be
            priced under it — before the job starts, with nothing spent.
        ValueError: The host supplies no completion clients.
    """
    return await _start_second_judge(
        host, "judge_second", run_id, scope_id, judge=judge, sample_fraction=sample_fraction, seed=seed, ids=result_ids
    )


async def judge_drift_check(
    host: OpsHost, run_id: str, scope_id: str, *, judge: SecondJudge, result_ids: list[str] | None = None
) -> JobsStarted:
    """Start re-scoring every judged result of a finished run under a changed judge as a job, refusing up front.

    :func:`judge_second` over the whole set (``sample_fraction`` 1): the run's judge is the configuration it was
    scored under, as recorded; ``judge`` is the new one. The stored scores are never changed. The drift its
    :class:`SecondJudgeRead` carries detects movement between the two judges, never which one is right.

    Args:
        host: The host: its store, its judge clients, its out-of-run cap and the job manager the task runs under.
        run_id: The finished run whose stored evidence is re-scored.
        scope_id: The scope it lives in.
        judge: The changed judge.
        result_ids: The frozen set to re-score; ``None`` for every result of the run.

    Returns:
        The check's one job (``judge:<id>``), whose target is the run.

    Raises:
        NotFoundError: No run with that id, or a record it names does not load.
        ValidationFailedError: As :func:`judge_second`.
        ValueError: The host supplies no completion clients.
    """
    return await _start_second_judge(
        host,
        "judge_drift_check",
        run_id,
        scope_id,
        judge=judge,
        sample_fraction=1.0,
        seed=DEFAULT_SECOND_JUDGE_SEED,
        ids=result_ids,
    )


async def judge_temperature(
    host: OpsHost,
    run_id: str,
    scope_id: str,
    *,
    selection: TemperatureSelection = "borderline",
    repeats: int = DEFAULT_TEMPERATURE_REPEATS,
    result_ids: list[str] | None = None,
) -> JobsStarted:
    """Check a judge temperature comparison over a finished run, admit its calls, then start it as a job (#633).

    The comparison re-judges the run's borderline cases ``repeats`` times at the pinned temperature and as many at
    the provider default, every call ledgered under purpose ``judge``. Nothing is written to the results: the
    answers are kept on the job's record, and ``job_poll`` on the ended job reads them into the
    :class:`~threetears.evals.analysis.JudgeTemperatureComparison` (:func:`~threetears.evals.analysis.read_judge_temperatures`).
    The call makes no model request.

    Args:
        host: The host: its store, its judge clients, its out-of-run cap and the job manager the task runs under.
        run_id: The finished run.
        scope_id: The scope it lives in.
        selection: ``borderline`` (the default) or ``all`` scored dims.
        repeats: Calls per dim at each temperature.
        result_ids: The results to draw from; ``None`` for every result of the run.

    Returns:
        The comparison's one job (``judge:<id>``), whose target is the run.

    Raises:
        NotFoundError: No run with that id, or a record it names does not load.
        ValidationFailedError: The comparison cannot be made, or its calls are priced above the cap or cannot be
            priced under it — before the job starts, with nothing spent.
        ValueError: The host supplies no completion clients.
    """
    eval_host = host.eval_host
    cap = host.out_of_run_cap()
    estimate = await estimate_judge_temperature_comparison(
        eval_host, run_id, scope_id, out_of_run_cap_usd=cap, selection=selection, repeats=repeats, result_ids=result_ids
    )
    if not estimate.would_start:
        raise ValidationFailedError(estimate.refusal or "the comparison's calls were not admitted")
    record = EvalJudgeJob(
        scope_id=scope_id,
        action="judge_temperature",
        run_id=estimate.run_id,
        results_total=estimate.results,
        max_calls=estimate.max_calls,
    )

    async def ask() -> None:
        async def progress(answers: JudgeTemperatureAnswers) -> None:
            record.temperature_answers = answers
            record.results_done = len({case.result_id for case in answers.answers})
            record.calls_made = answers.calls_made
            await _save_progress(host, record)

        answers = await judge_at_two_temperatures(
            eval_host,
            record.run_id,
            scope_id,
            out_of_run_cap_usd=cap,
            selection=selection,
            repeats=repeats,
            result_ids=result_ids,
            on_progress=progress,
        )
        record.temperature_answers = answers
        record.results_done = len({case.result_id for case in answers.answers})
        record.calls_made = answers.calls_made

    return await _start(host, record, ask, label=f"judge_temperature {estimate.judge_model}")


async def _start_second_judge(
    host: OpsHost,
    action: JudgeJobAction,
    run_id: str,
    scope_id: str,
    *,
    judge: SecondJudge,
    sample_fraction: float,
    seed: int,
    ids: list[str] | None,
) -> JobsStarted:
    """Refuse what the pass would refuse, by its own estimate, then start the pass as a job."""
    eval_host = host.eval_host
    cap = host.out_of_run_cap()
    estimate = await estimate_second_judge(
        eval_host,
        run_id,
        scope_id,
        judge=judge,
        out_of_run_cap_usd=cap,
        sample_fraction=sample_fraction,
        seed=seed,
        result_ids=ids,
    )
    if not estimate.would_start:
        raise ValidationFailedError(estimate.refusal or "the pass's calls were not admitted")
    record = EvalJudgeJob(
        scope_id=scope_id,
        action=action,
        run_id=estimate.run_id,
        results_total=len(estimate.sampled),
        max_calls=estimate.max_calls,
    )

    def tally(report: SecondJudgeReport) -> None:
        record.second_judge = report
        record.results_done = len(report.judged) + len(report.unwritten)
        record.calls_made = report.calls_made

    async def ask() -> None:
        async def progress(report: SecondJudgeReport) -> None:
            tally(report)
            await _save_progress(host, record)

        tally(
            await ask_second_judge(
                eval_host,
                record.run_id,
                scope_id,
                judge=judge,
                out_of_run_cap_usd=cap,
                sample_fraction=sample_fraction,
                seed=seed,
                result_ids=ids,
                on_progress=progress,
            )
        )

    return await _start(host, record, ask, label=f"{action} {judge.model}")


async def _save(host: OpsHost, record: EvalJudgeJob) -> None:
    eval_host = host.eval_host
    await run_blocking(eval_host.blocking_executor, eval_host.storage.save_judge_job, record)


async def _save_progress(host: OpsHost, record: EvalJudgeJob) -> None:
    """Write a progress tick; one that does not land is superseded by the next, or by the record's ending."""
    try:
        await _save(host, record)
    except StorageError:
        log.warning("eval.judge_job job=%s: a progress write did not land", record.id, exc_info=True)


async def _start(host: OpsHost, record: EvalJudgeJob, ask: Callable[[], Awaitable[None]], *, label: str) -> JobsStarted:
    """Write the record, start the task that runs ``ask`` and records how it ended, and hand back its job."""
    await _save(host, record)

    async def work() -> None:
        await _run(host, record, ask)

    # The engine's job ceiling, as a sweep's arm has: the work's own calls are each bounded by the host's client, and
    # a job is a backstop against one that never returns, not a measurement of how long the calls take.
    host.launch.job_manager.start_task(
        record.id, work, budget_s=JOB_TIMEOUT_CAP_S, key=judge_key(record.id, record.scope_id)
    )
    return JobsStarted(
        jobs=[JobHandle(job_id=judge_job_id(record.id), kind="judge", target_id=record.run_id, label=label)]
    )


async def _run(host: OpsHost, record: EvalJudgeJob, ask: Callable[[], Awaitable[None]]) -> None:
    """Run the work, then record how it ended, however it ended: completed, failed, or cancelled with its partial report."""
    try:
        await ask()
    except asyncio.CancelledError:
        partial = record.second_judge.stopped if record.second_judge is not None else None
        if record.temperature_answers is not None:
            partial = record.temperature_answers.stopped
        record.outcome, record.ended_at = "cancelled", utc_now_iso()
        record.detail = partial or "cancelled before its first judge call; nothing was spent"
        await _save(host, record)
        raise
    # prawduct:ok-broad-except — the job's record is where its ending is stated; the task boundary logs it
    except Exception as broke:
        log.exception(
            "eval.judge_job job=%s action=%s run=%s ended on an error", record.id, record.action, record.run_id
        )
        record.outcome, record.detail, record.ended_at = "failed", str(broke), utc_now_iso()
        await _save(host, record)
        return
    record.outcome, record.ended_at = "completed", utc_now_iso()
    await _save(host, record)


__all__ = [
    "judge_drift_check",
    "judge_second",
    "judge_temperature",
]
