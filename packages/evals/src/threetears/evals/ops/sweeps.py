"""A sweep: one call launching a template's arms that differ beyond the model, one after another, into one campaign.

``run_launch`` takes one set of overlays and apparatus settings for every arm and starts every arm at once. A
comparison whose arms differ in more than the model (two prompts, two retrieval settings) was N launches,
polled by hand, with the run ids carried to ``campaign_create`` afterwards; and running every arm at once is not
neutral, because shared provider rate limits inflate the latency each arm measures.

:func:`sweep_launch` declares the template, the subject, ``k`` and the settings the arms share, and a list of
arms, each with its own model, overlays and apparatus settings. It refuses up front whatever any arm's launch
would refuse (each arm is quoted by the launch's own rule, :func:`~threetears.evals.run.quote_launch`), creates
the named campaign, and starts one background job that launches the arms in order — its record written carrying
the job's lease first, so every process answers the job alike. By default one arm runs at a
time (``max_concurrent_arms=1``), and a caller can raise it. Arms one after another do not share provider load,
but a run's own cells still run side by side unless the sweep declares latency under test (``measure_latency``,
passed to every arm's launch as ``run_launch`` takes it), which latency comparisons need. Each arm's run joins the campaign as it is created,
so a sweep stopped part-way leaves its finished runs already members. The job is polled and cancelled through
the ordinary job contract (``sweep:<sweep id>``). Its progress counts the arms launched and finished, and one
cancel stops the arm in flight and launches none after it.

**What the arms hold fixed.** The judge and simulator are the sweep's (``judge_model`` / ``simulator_model``)
unless an arm names its own. Where the sweep names none, the first arm's run resolves the role's default, and
every later arm is pinned to what that run recorded. A default that moves part-way through a sweep therefore
cannot give two arms different instruments. The one exception is an arm whose own candidate is that judge: it
is left to the launch's own pin resolution, so no model grades its own output.

**Its campaign is exploratory.** A sweep takes no design, so the campaign it creates declares none, and says so
(``exploratory``) rather than leaving the caller to: what the arms were swept to learn is declared afterwards, once
the arms are known, with ``campaign_update``.

**When it stops early.** If an arm is refused at its launch (the host's admission ceiling, say), or an arm's run
ends ``failed``, no further arm is launched and the sweep ends ``failed``, naming it. A harness fault is not
measured N more times overnight. Runs already launched finish and stay members.
"""

from __future__ import annotations

import logging
import asyncio
import json
from functools import partial
from typing import Annotated, Any

from pydantic import Field, model_validator

from threetears.evals.analysis.campaigns import add_runs_to_campaign
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.kernel.campaign import EvalSweep, SweepArmRecord
from threetears.evals.kernel.errors import EvalServiceError, JobFenced, ValidationFailedError
from threetears.evals.kernel.refusals import job_refusal
from threetears.evals.schema.refusal import JobRefusal
from threetears.evals.kernel.job_timeout import EvalJobTimeout, timed_out_detail
from threetears.evals.schema.models import DEFAULT_LAUNCH_K_RUNS, CaseSetRef, utc_now_iso
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.ops.analysis import CampaignDefinition, campaign_create
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.jobs import JobHandle, JobsStarted, sweep_job_id, sweep_key
from threetears.evals.ops.runs import LaunchArguments
from threetears.evals.run.jobs import JOB_TIMEOUT_CAP_S
from threetears.evals.run.launch import quote_launch, start_run
from threetears.evals.kernel.job_leases import SWEEP_JOB, require_held, write_held_record
from threetears.evals.run.lifecycle import get_run
from threetears.evals.schema.logs import contain, get_logger

log = get_logger(__name__)

#: The reason recorded on an arm's run a sweep's cancel stopped.
SWEEP_CANCEL_REASON = "its sweep was cancelled"

#: The reason recorded on an arm's run a sweep's time budget stopped.
SWEEP_TIMEOUT_REASON = "its sweep's time budget ran out"

#: Each arm's share of a sweep's wall-clock budget, in seconds: the engine's cap on one run's job. A sweep's budget is
#: one more share than it has arms, since its arms may run one after another and launching them takes time too.
SWEEP_ARM_BUDGET_S = JOB_TIMEOUT_CAP_S


class SweepArm(EvalBaseModel):
    """One arm of a sweep: its model, and what it sets differently from its siblings."""

    label: str | None = Field(
        default=None, min_length=1, description="The arm's name; defaults to its model, numbered when models repeat."
    )
    model: str = Field(min_length=1, description="The arm's candidate model.")
    overlays: dict[str, Any] | None = Field(default=None, description="The knobs this arm turns, by field.")
    apparatus_settings: dict[str, Any] | None = Field(
        default=None, description="Host-declared apparatus values this arm's rig is set up with."
    )
    judge_model: str | None = Field(default=None, min_length=1, description="This arm's own judge, if it differs.")
    simulator_model: str | None = Field(
        default=None, min_length=1, description="This arm's own simulated user's model, if it differs."
    )


class SweepSettings(EvalBaseModel):
    """What every arm of a sweep shares, and the campaign its runs join — a sweep's arguments but its arms."""

    template_id: Annotated[str, LaunchArguments.model_fields["template_id"]]
    subject_id: Annotated[str, LaunchArguments.model_fields["subject_id"]]
    campaign_name: str = Field(
        min_length=1,
        description="The campaign every arm's run joins as it is created. It is created exploratory, since a sweep "
        "takes no design; declare one afterwards with campaign_update.",
    )
    campaign_behavior: str = Field(min_length=1, description="What the sweep's campaign measures.")
    campaign_description: str = Field(default="", description="The sweep's campaign's description.")
    k_runs: Annotated[int, LaunchArguments.model_fields["k_runs"]] = DEFAULT_LAUNCH_K_RUNS
    max_concurrent_arms: int = Field(
        default=1,
        ge=1,
        description="How many of a sweep's arms run at once. 1 (the default) runs them one after another, so no "
        "arm shares another's provider load. To compare latency, also declare measure_latency.",
    )
    judge_model: Annotated[str | None, LaunchArguments.model_fields["judge_model"]] = None
    simulator_model: Annotated[str | None, LaunchArguments.model_fields["simulator_model"]] = None
    max_cost_usd: Annotated[float | None, LaunchArguments.model_fields["max_cost_usd"]] = None
    cell_timeout_s: Annotated[float | None, LaunchArguments.model_fields["cell_timeout_s"]] = None
    case_set_name: Annotated[str | None, LaunchArguments.model_fields["case_set_name"]] = None
    case_set_version: Annotated[int | None, LaunchArguments.model_fields["case_set_version"]] = None
    measure_latency: Annotated[bool, LaunchArguments.model_fields["measure_latency"]] = False

    @property
    def case_set(self) -> CaseSetRef | None:
        """The case set every arm runs, or ``None``."""
        if self.case_set_name is None or self.case_set_version is None:
            return None
        return CaseSetRef(name=self.case_set_name, version=self.case_set_version)


class SweepArguments(SweepSettings):
    """What a sweep names: its shared settings and campaign (:class:`SweepSettings`), and its arms in order."""

    arms: list[SweepArm] = Field(min_length=1, description="The arms, launched in this order.")

    @model_validator(mode="after")
    def _distinct_arms(self) -> SweepArguments:
        """Name every arm, and refuse two arms that are one condition, or two arms under one name.

        :raises ValueError: Two arms set every setting alike, two share a label, or a case set is named without its
            version (or the reverse).
        """
        if (self.case_set_name is None) != (self.case_set_version is None):
            raise ValueError("case_set_name and case_set_version name one version of one set together")
        models = [arm.model for arm in self.arms]
        for index, arm in enumerate(self.arms):
            if arm.label is None:
                arm.label = arm.model if models.count(arm.model) == 1 else f"{arm.model} #{index + 1}"
        labels = [arm.label for arm in self.arms]
        if repeated := sorted({label for label in labels if label is not None and labels.count(label) > 1}):
            raise ValueError(f"each arm has its own label; {', '.join(map(repr, repeated))} repeat")
        conditions = [json.dumps(arm.model_dump(exclude={"label"}), sort_keys=True, default=str) for arm in self.arms]
        if len(set(conditions)) != len(conditions):
            raise ValueError("two arms set every setting alike, so they are one arm measured twice; raise k_runs")
        return self


def _arm_label(arm: SweepArm) -> str:
    assert arm.label is not None  # SweepArguments names every arm
    return arm.label


async def _refuse_what_any_arm_would_refuse(host: OpsHost, arguments: SweepArguments, scope_id: str) -> None:
    """Quote every arm by the launch's own rule before anything is created, and refuse the sweep on any refusal.

    :param host: The launching host.
    :ptype host: OpsHost
    :param arguments: The sweep.
    :ptype arguments: SweepArguments
    :param scope_id: The scope it runs in.
    :ptype scope_id: str
    :raises ValidationFailedError: An arm's launch would be refused, named with its arm.
    :raises NotFoundError: The template, or the case set, is not in the scope.
    """
    settings = host.launch.settings()
    if arguments.max_concurrent_arms > settings.max_admitted_runs:
        raise ValidationFailedError(
            f"max_concurrent_arms={arguments.max_concurrent_arms} is above the runs this host admits at once "
            f"({settings.name_of('max_admitted_runs')}={settings.max_admitted_runs})",
            remedy=f"pass max_concurrent_arms at or below {settings.max_admitted_runs}",
        )
    for arm in arguments.arms:
        try:
            quote = await quote_launch(
                host.launch,
                template_id=arguments.template_id,
                subject_id=arguments.subject_id,
                models=[arm.model],
                k_runs=arguments.k_runs,
                judge_model=arm.judge_model or arguments.judge_model,
                simulator_model=arm.simulator_model or arguments.simulator_model,
                overlays=arm.overlays,
                apparatus_settings=arm.apparatus_settings,
                max_cost_usd=arguments.max_cost_usd,
                scope_id=scope_id,
                cell_timeout_s=arguments.cell_timeout_s,
                case_set=arguments.case_set,
            )
        except ValidationFailedError as refused:
            raise ValidationFailedError(
                f"arm {_arm_label(arm)!r} would be refused: {refused.message}", remedy=refused.remedy
            ) from refused
        if refused_arms := [verdict for verdict in quote.arms if verdict.refusal is not None]:
            raise ValidationFailedError(
                f"arm {_arm_label(arm)!r} would be refused: "
                + "; ".join(str(verdict.refusal) for verdict in refused_arms),
                remedy="; ".join(dict.fromkeys(str(verdict.remedy) for verdict in refused_arms)),
            )


async def sweep_launch(host: OpsHost, arguments: SweepArguments, scope_id: str, *, created_by: str) -> JobsStarted:
    """Create the sweep's campaign and start the job that launches its arms in order.

    :param host: The launching host.
    :ptype host: OpsHost
    :param arguments: The sweep.
    :ptype arguments: SweepArguments
    :param scope_id: The scope the template is read in and the runs and campaign live in.
    :ptype scope_id: str
    :param created_by: Who is starting it, recorded on the campaign.
    :ptype created_by: str
    :return: The sweep's one job (``sweep:<sweep id>``), whose target is the campaign its runs join; polling it reports
        the arms launched and finished.
    :rtype: JobsStarted
    :raises ValidationFailedError: An arm's launch would be refused, concurrency above the host's admission ceiling,
        or the campaign definition is refused.
    :raises NotFoundError: The template or the case set is not in the scope.
    """
    await _refuse_what_any_arm_would_refuse(host, arguments, scope_id)
    eval_host, manager = host.eval_host, host.launch.job_manager
    campaign = await run_blocking(
        eval_host.blocking_executor,
        lambda: campaign_create(
            eval_host,
            CampaignDefinition(
                name=arguments.campaign_name,
                subject_id=arguments.subject_id,
                behavior=arguments.campaign_behavior,
                description=arguments.campaign_description,
                exploratory=True,
            ),
            scope_id,
            created_by=created_by,
        ),
    )
    sweep = EvalSweep(
        scope_id=scope_id,
        campaign_id=campaign.id,
        template_id=arguments.template_id,
        subject_id=arguments.subject_id,
        arms=[SweepArmRecord(label=_arm_label(arm), model=arm.model) for arm in arguments.arms],
        max_concurrent_arms=arguments.max_concurrent_arms,
    )
    leases = host.launch.job_leases

    def save_started() -> None:
        # Written carrying its lease, so no reader ever finds the sweep running and unleased, which reads lost.
        sweep.lease = leases.grant()
        eval_host.storage.save_sweep(sweep)

    await run_blocking(eval_host.blocking_executor, save_started)

    async def work() -> None:
        await _run_sweep(host, arguments, sweep, budget_s=(len(arguments.arms) + 1) * SWEEP_ARM_BUDGET_S)

    # The sweep enters its own budget (_run_sweep), so its record can say it timed out: one entered around the task
    # would reach the sweep as a cancel.
    manager.start_task(sweep.id, work, budget_s=None, key=sweep_key(sweep.id, scope_id), record=(SWEEP_JOB, scope_id))
    return JobsStarted(
        jobs=[JobHandle(job_id=sweep_job_id(sweep.id), kind="sweep", target_id=campaign.id, label=campaign.name)]
    )


async def _run_sweep(host: OpsHost, arguments: SweepArguments, sweep: EvalSweep, *, budget_s: float) -> None:
    """Launch each arm in order, at most ``max_concurrent_arms`` at once, joining each run to the campaign.

    The sweep's record is written as each arm's run is created and when the sweep ends, however it ends: a
    cancel — here, or asked on the record from another process — stops the arms in flight (each records
    ``cancelled``) and launches none after. Every write is the job's own guarded write
    (:func:`~threetears.evals.kernel.job_leases.write_held_record`), and every arm's launch is preceded by the same
    check (:func:`~threetears.evals.kernel.job_leases.require_held`): once the record reads lost — the lease lapsed and
    a reader somewhere wrote so, or this process shut down, or this process let go of its lease — the sweep
    launches nothing more and writes nothing more, leaving the arms it launched to their own jobs.

    **Its own budget.** ``budget_s`` is entered here, around the launching and the wait, through the host's job
    timeout; when it runs out the arms in flight are stopped (:data:`SWEEP_TIMEOUT_REASON`), none is launched after,
    and the sweep ends ``timed_out``, its detail naming the budget — never ``cancelled``, which is someone deciding.

    :param host: The launching host.
    :ptype host: OpsHost
    :param arguments: The sweep.
    :ptype arguments: SweepArguments
    :param sweep: Its record, as started.
    :ptype sweep: EvalSweep
    :param budget_s: The sweep's wall-clock budget.
    :ptype budget_s: float
    """
    eval_host, manager = host.eval_host, host.launch.job_manager
    scope_id = sweep.scope_id
    slots = asyncio.Semaphore(sweep.max_concurrent_arms)
    in_flight: dict[str, asyncio.Task[None]] = {}
    failure: list[str] = []
    # The engine refusal the sweep ended on, kept as data beside its words: set only for the failure the sweep's
    # detail states (the first), since a refusal after another failure did not end it.
    refusal: JobRefusal | None = None
    judge_pin, simulator_pin = arguments.judge_model, arguments.simulator_model

    leases = host.launch.job_leases

    async def save() -> None:
        await run_blocking(
            eval_host.blocking_executor, partial(write_held_record, leases=leases), eval_host.storage, SWEEP_JOB, sweep
        )

    async def stop_arms(reason: str) -> None:
        for run_id, waiter in in_flight.items():
            manager.cancel_job(run_id, reason=reason)
            waiter.cancel()
        await manager.wait_for(list(in_flight))
        await asyncio.gather(*in_flight.values(), return_exceptions=True)

    def fenced(why: JobFenced) -> None:
        log.warning("eval.sweep sweep=%s scope=%s stopped: %s", sweep.id, scope_id, why)

    async def settle(label: str, run_id: str) -> None:
        try:
            await manager.wait_for([run_id])
            run = await run_blocking(eval_host.blocking_executor, get_run, eval_host.storage, run_id, scope_id)
            if run.status == "failed":
                failure.append(f"arm {label!r}'s run {run_id} ended failed, so no further arm was launched")
        finally:
            slots.release()

    try:
        async with manager.job_timeout_factory(budget_s):
            for spec, record in zip(arguments.arms, sweep.arms, strict=True):
                await slots.acquire()
                if failure:
                    slots.release()
                    break
                # Fenced BEFORE the arm is launched, not only at the write after it: a sweep whose record reads lost
                # would otherwise pay for one more arm that nothing records, whose run no sweep and no campaign names.
                try:
                    await run_blocking(
                        eval_host.blocking_executor,
                        partial(require_held, leases=leases),
                        eval_host.storage,
                        SWEEP_JOB,
                        sweep.id,
                        scope_id,
                    )
                except JobFenced:
                    slots.release()
                    raise
                judge = spec.judge_model or (judge_pin if judge_pin != spec.model else None)
                simulator = spec.simulator_model or simulator_pin
                try:
                    (run,) = await start_run(
                        host.launch,
                        template_id=arguments.template_id,
                        subject_id=arguments.subject_id,
                        models=[spec.model],
                        k_runs=arguments.k_runs,
                        judge_model=judge,
                        simulator_model=simulator,
                        overlays=spec.overlays,
                        apparatus_settings=spec.apparatus_settings,
                        max_cost_usd=arguments.max_cost_usd,
                        cell_timeout_s=arguments.cell_timeout_s,
                        case_set=arguments.case_set,
                        measure_latency=arguments.measure_latency,
                        scope_id=scope_id,
                    )
                # prawduct:ok-broad-except — any refusal of one arm ends the sweep, recorded on its record
                except Exception as refused:
                    contain(refused, log=log, what=f"eval.sweep arm {record.label!r} was refused at its launch")
                    slots.release()
                    failure.append(
                        f"arm {record.label!r} was refused at its launch, so no further arm was launched: {refused}"
                    )
                    if isinstance(refused, EvalServiceError) and len(failure) == 1:
                        refusal = job_refusal(refused)
                    break
                in_flight[run.id] = asyncio.create_task(settle(record.label, run.id))
                record.run_id = run.id
                await run_blocking(
                    eval_host.blocking_executor,
                    add_runs_to_campaign,
                    eval_host.storage,
                    sweep.campaign_id,
                    scope_id,
                    [run.id],
                )
                await save()
                # The first arm's resolved instruments hold for every later arm that names none of its own.
                judge_pin = judge_pin or run.judge_model
                simulator_pin = simulator_pin or run.simulator_model
            await asyncio.gather(*in_flight.values())
    except asyncio.CancelledError:
        await stop_arms(SWEEP_CANCEL_REASON)
        sweep.outcome, sweep.detail, sweep.ended_at = "cancelled", "cancelled before every arm ran", utc_now_iso()
        try:
            await save()
        except JobFenced as why:
            # NOSILENT: the sweep's record, written lost elsewhere, is its ending; this process writes nothing over it.
            fenced(why)
        raise
    except EvalJobTimeout as timed_out:
        log.warning("eval.sweep sweep=%s scope=%s timed out: %s", sweep.id, scope_id, timed_out)
        await stop_arms(SWEEP_TIMEOUT_REASON)
        sweep.outcome, sweep.detail, sweep.ended_at = "timed_out", timed_out_detail(timed_out), utc_now_iso()
        try:
            await save()
        except JobFenced as why:
            # NOSILENT: the sweep's record, written lost elsewhere, is its ending; this process writes nothing over it.
            fenced(why)
        return
    except JobFenced as why:
        # The record reads lost: this process no longer holds the sweep. Launch nothing more and write nothing
        # more; the arms already launched are their own jobs, and finish or are lost as their own records say.
        fenced(why)
        return
    # prawduct:ok-broad-except — the sweep's record is where its ending is stated; the task boundary logs it
    except Exception as broke:
        # NOSILENT: written below on the sweep's own record: outcome ``failed``, its detail this error, its refusal kept.
        contain(
            broke,
            log=log,
            what=f"eval.sweep sweep={sweep.id} scope={scope_id} ended on an error",
            level=logging.ERROR,
        )
        failure.append(f"the sweep ended on an error: {broke}")
        if isinstance(broke, EvalServiceError) and len(failure) == 1:
            refusal = job_refusal(broke)
        # Arms already launched are measurements in their own right: they finish and stay members.
        await asyncio.gather(*in_flight.values(), return_exceptions=True)
    sweep.outcome = "failed" if failure else "completed"
    sweep.detail = failure[0] if failure else None
    sweep.refusal = refusal
    sweep.ended_at = utc_now_iso()
    try:
        await save()
    except JobFenced as why:
        # NOSILENT: the sweep's record, written lost elsewhere, is its ending; this process writes nothing over it.
        fenced(why)


__all__ = [
    "SWEEP_ARM_BUDGET_S",
    "SWEEP_CANCEL_REASON",
    "SWEEP_TIMEOUT_REASON",
    "SweepArguments",
    "SweepArm",
    "sweep_launch",
]
