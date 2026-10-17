"""A sweep: one call launching a template's arms that differ beyond the model, into one campaign.

``run_launch`` takes one set of overlays and apparatus settings for every arm. A comparison whose arms differ in
more than the model (two prompts, two retrieval settings) was N launches, polled by hand, with the run ids carried
to ``campaign_create`` afterwards.

:func:`sweep_launch` declares the template, the subject, ``k`` and the settings the arms share, and a list of
arms, each with its own model, overlays and apparatus settings. It refuses up front whatever any arm's launch
would refuse (each arm is quoted by the launch's own rule, :func:`~threetears.evals.run.quote_launch`), creates
the named campaign, and starts one background job that launches the arms — its record written carrying the job's
lease first, so every process answers the job alike.

**How the arms share time is the launch's own argument** (``arm_schedule``, declared once on
:class:`~threetears.evals.ops.LaunchArguments`, so the sweep and ``run_launch`` cannot mean two things by it), because
running every arm at once is not neutral: shared provider capacity rations one arm's calls and not another's, and
inflates the latency each arm measures (:mod:`threetears.evals.run.arm_schedule`). By default the arms are
interleaved: every arm is prepared, then all start together as one launch whose cells are taken in turn, case by
case, under one width, so a provider's bad minute lands on every arm alike. ``sequential`` launches the arms one
after another, each its own launch, so no arm shares another's provider load; ``concurrent`` launches them all at
once, each side by side at its own width. Latency under test (``measure_latency``, passed to every arm's launch as
``run_launch`` takes it) runs them sequential, the only schedule it takes. Each arm's run joins the campaign as it
is created, so a sweep stopped part-way leaves its finished runs already members. The job is polled and cancelled
through the ordinary job contract (``sweep:<sweep id>``). Its progress counts the arms launched and finished, and
one cancel stops the arms in flight and launches none after.

**What the arms hold fixed.** The judge and simulator are the sweep's (``judge_model`` / ``simulator_model``)
unless an arm names its own. Where the sweep names none, the role's default is resolved once for every arm: an
interleaved sweep's arms are one launch, which resolves it once for all of them, and a sweep launching arm after
arm pins every later arm to what the first arm's run recorded. A default that moves part-way through a sweep
therefore cannot give two arms different instruments. The one exception is an arm whose own candidate is that
judge: it is left to the launch's own pin resolution, so no model grades its own output.

**Its campaign is exploratory.** A sweep takes no design, so the campaign it creates declares none, and says so
(``exploratory``) rather than leaving the caller to: what the arms were swept to learn is declared afterwards, once
the arms are known, with ``campaign_update``.

**When it stops early.** If an arm is refused at its launch (the host's admission ceiling, say), or an arm's run
ends ``failed``, no further arm is launched and the sweep ends ``failed``, naming it. A harness fault is not
measured N more times overnight. Runs already launched finish and stay members. An interleaved sweep's arms are
launched together, so an arm refused at its launch starts none of them, and an arm whose run fails leaves its
siblings to finish.
"""

from __future__ import annotations

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
from threetears.evals.schema.models import DEFAULT_LAUNCH_K_RUNS, ArmSchedule, CaseSetRef, EvalRun, utc_now_iso
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.ops.analysis import CampaignDefinition, campaign_create
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.jobs import JobHandle, JobsStarted, sweep_job_id, sweep_key
from threetears.evals.ops.runs import LaunchArguments
from threetears.evals.run.arm_schedule import resolve_arm_schedule
from threetears.evals.run.jobs import JOB_TIMEOUT_CAP_S
from threetears.evals.run.launch import LaunchGroup, launch_as_group, quote_launch, start_run
from threetears.evals.kernel.job_leases import SWEEP_JOB, require_held, write_held_record
from threetears.evals.run.lifecycle import get_run
from threetears.evals.schema.logs import get_logger

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
    arm_schedule: Annotated[ArmSchedule | None, LaunchArguments.model_fields["arm_schedule"]] = None
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

    @property
    def schedule(self) -> ArmSchedule:
        """The schedule the arms run under: the one named, or the default for whether latency is under test.

        :raises ValidationFailedError: Latency is under test and the arms are asked to run at once.
        """
        return resolve_arm_schedule(self.arm_schedule, measure_latency=self.measure_latency)


class SweepArguments(SweepSettings):
    """What a sweep names: its shared settings and campaign (:class:`SweepSettings`), and its arms in order."""

    arms: list[SweepArm] = Field(
        min_length=1, description="The arms, in order: the order they are launched in, and take their turns in."
    )

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


def _arms_at_once(arguments: SweepArguments) -> int:
    """How many of the sweep's arms run at once: one arm after another, or every arm."""
    return 1 if arguments.schedule == "sequential" else len(arguments.arms)


async def _refuse_what_any_arm_would_refuse(host: OpsHost, arguments: SweepArguments, scope_id: str) -> None:
    """Quote every arm by the launch's own rule before anything is created, and refuse the sweep on any refusal.

    :param host: The launching host.
    :ptype host: OpsHost
    :param arguments: The sweep.
    :ptype arguments: SweepArguments
    :param scope_id: The scope it runs in.
    :ptype scope_id: str
    :raises ValidationFailedError: An arm's launch would be refused, named with its arm; the schedule is one latency
        cannot be read under; or it runs more arms at once than the host admits, or than one launch may start.
    :raises NotFoundError: The template, or the case set, is not in the scope.
    """
    settings = host.launch.settings()
    schedule = arguments.schedule
    at_once = _arms_at_once(arguments)
    if at_once > settings.max_admitted_runs:
        raise ValidationFailedError(
            f"arm_schedule={schedule!r} runs all {at_once} arms at once, above the runs this host admits at once "
            f"({settings.name_of('max_admitted_runs')}={settings.max_admitted_runs})",
            remedy=f"pass arm_schedule='sequential', or sweep at most {settings.max_admitted_runs} arms",
        )
    if schedule == "interleaved" and at_once > settings.max_launch_arms:
        raise ValidationFailedError(
            f"arm_schedule='interleaved' starts all {at_once} arms as one launch, above the runs one launch may start "
            f"({settings.name_of('max_launch_arms')}={settings.max_launch_arms})",
            remedy=f"pass arm_schedule='sequential', or sweep at most {settings.max_launch_arms} arms",
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
    """Create the sweep's campaign and start the job that launches its arms, as its arm schedule says.

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
    :raises ValidationFailedError: An arm's launch would be refused, a schedule latency cannot be read under, arms at
        once above the host's admission ceiling or its launch ceiling, or the campaign definition is refused.
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
        max_concurrent_arms=_arms_at_once(arguments),
        arm_schedule=arguments.schedule,
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
    """Launch the arms as the sweep's schedule says, joining each run to the campaign, and wait for them.

    An interleaved sweep prepares every arm into one launch and starts them together
    (:meth:`_SweepRun.launch_together`); any other launches each arm as its own launch, in order,
    ``max_concurrent_arms`` at once (:meth:`_SweepRun.launch_in_order`).

    The sweep's record is written as its arms' runs are created and when the sweep ends, however it ends: a
    cancel — here, or asked on the record from another process — stops the arms in flight (each records
    ``cancelled``) and launches none after. Every write is the job's own guarded write
    (:func:`~threetears.evals.kernel.job_leases.write_held_record`), and every launch is preceded by the same
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
    manager = host.launch.job_manager
    scope_id = sweep.scope_id
    run = _SweepRun(host, arguments, sweep)

    def fenced(why: JobFenced) -> None:
        log.warning("eval.sweep sweep=%s scope=%s stopped: %s", sweep.id, scope_id, why)

    try:
        async with manager.job_timeout_factory(budget_s):
            if sweep.arm_schedule == "interleaved":
                await run.launch_together()
            else:
                await run.launch_in_order()
            await asyncio.gather(*run.in_flight.values())
    except asyncio.CancelledError:
        await run.stop_arms(SWEEP_CANCEL_REASON)
        sweep.outcome, sweep.detail, sweep.ended_at = "cancelled", "cancelled before every arm ran", utc_now_iso()
        try:
            await run.save()
        except JobFenced as why:
            # NOSILENT: the sweep's record, written lost elsewhere, is its ending; this process writes nothing over it.
            fenced(why)
        raise
    except EvalJobTimeout as timed_out:
        log.warning("eval.sweep sweep=%s scope=%s timed out: %s", sweep.id, scope_id, timed_out)
        await run.stop_arms(SWEEP_TIMEOUT_REASON)
        sweep.outcome, sweep.detail, sweep.ended_at = "timed_out", timed_out_detail(timed_out), utc_now_iso()
        try:
            await run.save()
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
        log.exception("eval.sweep sweep=%s scope=%s ended on an error", sweep.id, scope_id)
        run.fail(f"the sweep ended on an error: {broke}", broke)
        # Arms already launched are measurements in their own right: they finish and stay members.
        await asyncio.gather(*run.in_flight.values(), return_exceptions=True)
    sweep.outcome = "failed" if run.failure else "completed"
    sweep.detail = run.failure[0] if run.failure else None
    sweep.refusal = run.refusal
    sweep.ended_at = utc_now_iso()
    try:
        await run.save()
    except JobFenced as why:
        # NOSILENT: the sweep's record, written lost elsewhere, is its ending; this process writes nothing over it.
        fenced(why)


class _SweepRun:
    """One sweep's job as it runs: its arms in flight, what failed, and how each schedule launches its arms."""

    def __init__(self, host: OpsHost, arguments: SweepArguments, sweep: EvalSweep) -> None:
        self.host = host
        self.arguments = arguments
        self.sweep = sweep
        self.in_flight: dict[str, asyncio.Task[None]] = {}
        self.failure: list[str] = []
        # The engine refusal the sweep ended on, kept as data beside its words: set only for the failure the sweep's
        # detail states (the first), since a refusal after another failure did not end it.
        self.refusal: JobRefusal | None = None

    def fail(self, why: str, error: BaseException | None = None) -> None:
        """Record a failure, keeping the engine refusal behind the first as data."""
        self.failure.append(why)
        if isinstance(error, EvalServiceError) and len(self.failure) == 1:
            self.refusal = job_refusal(error)

    async def save(self) -> None:
        eval_host = self.host.eval_host
        await run_blocking(
            eval_host.blocking_executor,
            partial(write_held_record, leases=self.host.launch.job_leases),
            eval_host.storage,
            SWEEP_JOB,
            self.sweep,
        )

    async def require_held(self) -> None:
        """Refuse to launch more once the sweep's record reads lost: a launch nothing records would be paid for."""
        eval_host = self.host.eval_host
        await run_blocking(
            eval_host.blocking_executor,
            partial(require_held, leases=self.host.launch.job_leases),
            eval_host.storage,
            SWEEP_JOB,
            self.sweep.id,
            self.sweep.scope_id,
        )

    async def stop_arms(self, reason: str) -> None:
        manager = self.host.launch.job_manager
        for run_id, waiter in self.in_flight.items():
            manager.cancel_job(run_id, reason=reason)
            waiter.cancel()
        await manager.wait_for(list(self.in_flight))
        await asyncio.gather(*self.in_flight.values(), return_exceptions=True)

    async def settle(self, label: str, run_id: str, *, done: asyncio.Semaphore | None = None) -> None:
        """Wait for one arm's run, recording it failed if it ended so, and free its place when it holds one."""
        eval_host = self.host.eval_host
        try:
            await self.host.launch.job_manager.wait_for([run_id])
            run = await run_blocking(
                eval_host.blocking_executor, get_run, eval_host.storage, run_id, self.sweep.scope_id
            )
            if run.status == "failed":
                after = "so no further arm was launched" if done is not None else "beside its siblings"
                self.failure.append(f"arm {label!r}'s run {run_id} ended failed, {after}")
        finally:
            if done is not None:
                done.release()

    async def join_campaign(self, runs: list[EvalRun]) -> None:
        eval_host = self.host.eval_host
        await run_blocking(
            eval_host.blocking_executor,
            add_runs_to_campaign,
            eval_host.storage,
            self.sweep.campaign_id,
            self.sweep.scope_id,
            [run.id for run in runs],
        )

    def _judge(self, spec: SweepArm, pin: str | None) -> str | None:
        # An arm whose own candidate is the pinned judge is left to the launch's pin resolution: no model grades itself.
        return spec.judge_model or (pin if pin != spec.model else None)

    async def launch_in_order(self) -> None:
        """Launch each arm as its own launch, in order, at most ``max_concurrent_arms`` at once."""
        arguments, sweep = self.arguments, self.sweep
        slots = asyncio.Semaphore(sweep.max_concurrent_arms)
        judge_pin, simulator_pin = arguments.judge_model, arguments.simulator_model
        for spec, record in zip(arguments.arms, sweep.arms, strict=True):
            await slots.acquire()
            if self.failure:
                slots.release()
                return
            # Fenced BEFORE the arm is launched, not only at the write after it: a sweep whose record reads lost
            # would otherwise pay for one more arm that nothing records, whose run no sweep and no campaign names.
            try:
                await self.require_held()
            except JobFenced:
                slots.release()
                raise
            try:
                (run,) = await start_run(
                    self.host.launch,
                    **self._arm_launch(spec, judge_model=self._judge(spec, judge_pin)),
                    simulator_model=spec.simulator_model or simulator_pin,
                    arm_schedule=sweep.arm_schedule,
                )
            # prawduct:ok-broad-except — any refusal of one arm ends the sweep, recorded on its record
            except Exception as refused:
                slots.release()
                self.fail(
                    f"arm {record.label!r} was refused at its launch, so no further arm was launched: {refused}",
                    refused,
                )
                return
            self.in_flight[run.id] = asyncio.create_task(self.settle(record.label, run.id, done=slots))
            record.run_id = run.id
            await self.join_campaign([run])
            await self.save()
            # The first arm's resolved instruments hold for every later arm that names none of its own.
            judge_pin = judge_pin or run.judge_model
            simulator_pin = simulator_pin or run.simulator_model

    async def launch_together(self) -> None:
        """Prepare every arm into one interleaved launch and start them together, every run then joining the campaign.

        The launch resolves what its arms share once (the subject's snapshot, the case set, the judge's default), so no
        arm's instruments are pinned from another's run. An arm refused at its launch starts none of them.
        """
        arguments, sweep = self.arguments, self.sweep
        await self.require_held()
        group = LaunchGroup(
            candidate_models=list(dict.fromkeys(spec.model for spec in arguments.arms)), arm_schedule="interleaved"
        )
        preparing: list[str] = []

        async def form() -> tuple[LaunchGroup, None]:
            return group, None

        async def prepare(group: LaunchGroup, _formed: None) -> list[EvalRun]:
            prepared: list[EvalRun] = []
            for spec, record in zip(arguments.arms, sweep.arms, strict=True):
                preparing[:] = [record.label]
                prepared += await start_run(
                    self.host.launch,
                    **self._arm_launch(spec, judge_model=self._judge(spec, arguments.judge_model)),
                    simulator_model=spec.simulator_model or arguments.simulator_model,
                    launch_group=group,
                )
            preparing.clear()
            return prepared

        try:
            runs = await launch_as_group(
                self.host.launch,
                len(arguments.arms),
                settings=self.host.launch.settings(),
                form=form,
                prepare=prepare,
                event="eval.sweep",
            )
        # prawduct:ok-broad-except — any refusal of one arm ends the sweep, recorded on its record
        except Exception as refused:
            if isinstance(refused, JobFenced):
                raise
            which = f"arm {preparing[0]!r} was refused at its launch" if preparing else "the arms' launch was refused"
            self.fail(f"{which}, so no arm was launched: {refused}", refused)
            return
        for record, run in zip(sweep.arms, runs, strict=True):
            self.in_flight[run.id] = asyncio.create_task(self.settle(record.label, run.id))
            record.run_id = run.id
        # Members once they exist, as an arm launched on its own joins once its run is created.
        await self.join_campaign(runs)
        await self.save()

    def _arm_launch(self, spec: SweepArm, *, judge_model: str | None) -> dict[str, Any]:
        """What every launch of one arm names, whichever schedule launches it."""
        arguments = self.arguments
        return {
            "template_id": arguments.template_id,
            "subject_id": arguments.subject_id,
            "models": [spec.model],
            "k_runs": arguments.k_runs,
            "judge_model": judge_model,
            "overlays": spec.overlays,
            "apparatus_settings": spec.apparatus_settings,
            "max_cost_usd": arguments.max_cost_usd,
            "cell_timeout_s": arguments.cell_timeout_s,
            "case_set": arguments.case_set,
            "measure_latency": arguments.measure_latency,
            "scope_id": self.sweep.scope_id,
        }


__all__ = [
    "SWEEP_ARM_BUDGET_S",
    "SWEEP_CANCEL_REASON",
    "SWEEP_TIMEOUT_REASON",
    "SweepArguments",
    "SweepArm",
    "sweep_launch",
]
