"""A sweep: one call launching a template's arms that differ beyond the model, one after another, into one campaign.

``run_launch`` takes one set of overlays and apparatus settings for every arm and starts every arm at once. A
comparison whose arms differ in more than the model (two prompts, two retrieval settings) was N launches,
polled by hand, with the run ids carried to ``campaign_create`` afterwards; and running every arm at once is not
neutral, because shared provider rate limits inflate the latency each arm measures.

:func:`sweep_launch` declares the template, the subject, ``k`` and the settings the arms share, and a list of
arms, each with its own model, overlays and apparatus settings. It refuses up front whatever any arm's launch
would refuse (each arm is quoted by the launch's own rule, :func:`~threetears.evals.run.quote_launch`), creates
the named campaign, and starts one background job that launches the arms in order. By default one arm runs at a
time (``max_concurrent_arms=1``), and a caller can raise it. Each arm's run joins the campaign as it is created,
so a sweep stopped part-way leaves its finished runs already members. The job is polled and cancelled through
the ordinary job contract (``sweep:<sweep id>``). Its progress counts the arms launched and finished, and one
cancel stops the arm in flight and launches none after it.

**What the arms hold fixed.** The judge and simulator are the sweep's (``judge_model`` / ``simulator_model``)
unless an arm names its own. Where the sweep names none, the first arm's run resolves the role's default, and
every later arm is pinned to what that run recorded. A default that moves part-way through a sweep therefore
cannot give two arms different instruments. The one exception is an arm whose own candidate is that judge: it
is left to the launch's own pin resolution, so no model grades its own output.

**When it stops early.** If an arm is refused at its launch (the host's admission ceiling, say), or an arm's run
ends ``failed``, no further arm is launched and the sweep ends ``failed``, naming it. A harness fault is not
measured N more times overnight. Runs already launched finish and stay members.
"""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any

from pydantic import Field, model_validator

from threetears.evals.analysis.campaigns import add_runs_to_campaign
from threetears.evals.contracts.base import EvalBaseModel
from threetears.evals.contracts.campaign import EvalSweep, SweepArmRecord
from threetears.evals.contracts.errors import ValidationFailedError
from threetears.evals.contracts.models import DEFAULT_LAUNCH_K_RUNS, CaseSetRef, utc_now_iso
from threetears.evals.contracts.offload import run_blocking
from threetears.evals.ops.analysis import CampaignDefinition, campaign_create
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.jobs import JobHandle, JobsStarted, sweep_job_id, sweep_key
from threetears.evals.ops.runs import LaunchArguments
from threetears.evals.run.jobs import JOB_TIMEOUT_CAP_S
from threetears.evals.run.launch import quote_launch, start_run
from threetears.evals.run.lifecycle import get_run
from threetears.observe import get_logger

log = get_logger(__name__)

#: The reason recorded on an arm's run a sweep's cancel stopped.
SWEEP_CANCEL_REASON = "its sweep was cancelled"


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
    campaign_name: str = Field(min_length=1, description="The campaign every arm's run joins as it is created.")
    campaign_behavior: str = Field(min_length=1, description="What the sweep's campaign measures.")
    campaign_description: str = Field(default="", description="The sweep's campaign's description.")
    k_runs: Annotated[int, LaunchArguments.model_fields["k_runs"]] = DEFAULT_LAUNCH_K_RUNS
    max_concurrent_arms: int = Field(
        default=1,
        ge=1,
        description="How many of a sweep's arms run at once. 1 (the default) runs them one after another, so no "
        "arm's latency is measured under another's provider load.",
    )
    judge_model: Annotated[str | None, LaunchArguments.model_fields["judge_model"]] = None
    simulator_model: Annotated[str | None, LaunchArguments.model_fields["simulator_model"]] = None
    max_cost_usd: Annotated[float | None, LaunchArguments.model_fields["max_cost_usd"]] = None
    cell_timeout_s: Annotated[float | None, LaunchArguments.model_fields["cell_timeout_s"]] = None
    case_set_name: Annotated[str | None, LaunchArguments.model_fields["case_set_name"]] = None
    case_set_version: Annotated[int | None, LaunchArguments.model_fields["case_set_version"]] = None

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

        Raises:
            ValueError: Two arms set every setting alike, two share a label, or a case set is named without its
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

    Args:
        host: The launching host.
        arguments: The sweep.
        scope_id: The scope it runs in.

    Raises:
        ValidationFailedError: An arm's launch would be refused, named with its arm.
        NotFoundError: The template, or the case set, is not in the scope.
    """
    settings = host.launch.settings()
    if arguments.max_concurrent_arms > settings.max_admitted_runs:
        raise ValidationFailedError(
            f"max_concurrent_arms={arguments.max_concurrent_arms} is above the runs this host admits at once "
            f"({settings.name_of('max_admitted_runs')}={settings.max_admitted_runs})"
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
            raise ValidationFailedError(f"arm {_arm_label(arm)!r} would be refused: {refused}") from refused
        if refusals := [verdict.refusal for verdict in quote.arms if verdict.refusal is not None]:
            raise ValidationFailedError(f"arm {_arm_label(arm)!r} would be refused: {'; '.join(refusals)}")


async def sweep_launch(host: OpsHost, arguments: SweepArguments, scope_id: str, *, created_by: str) -> JobsStarted:
    """Create the sweep's campaign and start the job that launches its arms in order.

    Args:
        host: The launching host.
        arguments: The sweep.
        scope_id: The scope the template is read in and the runs and campaign live in.
        created_by: Who is starting it, recorded on the campaign.

    Returns:
        The sweep's one job (``sweep:<sweep id>``), whose target is the campaign its runs join; polling it reports
        the arms launched and finished.

    Raises:
        ValidationFailedError: An arm's launch would be refused, concurrency above the host's admission ceiling,
            or the campaign definition is refused.
        NotFoundError: The template or the case set is not in the scope.
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
    await run_blocking(eval_host.blocking_executor, eval_host.storage.save_sweep, sweep)

    async def work() -> None:
        await _run_sweep(host, arguments, sweep)

    # The task's own ceiling sits above every arm's: each arm's job is bounded by the engine's job cap, and the
    # arms may run one after another.
    budget_s = (len(arguments.arms) + 1) * JOB_TIMEOUT_CAP_S
    manager.start_task(sweep.id, work, budget_s=budget_s, key=sweep_key(sweep.id, scope_id))
    return JobsStarted(
        jobs=[JobHandle(job_id=sweep_job_id(sweep.id), kind="sweep", target_id=campaign.id, label=campaign.name)]
    )


async def _run_sweep(host: OpsHost, arguments: SweepArguments, sweep: EvalSweep) -> None:
    """Launch each arm in order, at most ``max_concurrent_arms`` at once, joining each run to the campaign.

    The sweep's record is written as each arm's run is created and when the sweep ends, however it ends: a
    cancel stops the arms in flight (each records ``cancelled``) and launches none after.

    Args:
        host: The launching host.
        arguments: The sweep.
        sweep: Its record, as started.
    """
    eval_host, manager = host.eval_host, host.launch.job_manager
    scope_id = sweep.scope_id
    slots = asyncio.Semaphore(sweep.max_concurrent_arms)
    in_flight: dict[str, asyncio.Task[None]] = {}
    failure: list[str] = []
    judge_pin, simulator_pin = arguments.judge_model, arguments.simulator_model

    async def save() -> None:
        await run_blocking(eval_host.blocking_executor, eval_host.storage.save_sweep, sweep)

    async def settle(label: str, run_id: str) -> None:
        try:
            await manager.wait_for([run_id])
            run = await run_blocking(eval_host.blocking_executor, get_run, eval_host.storage, run_id, scope_id)
            if run.status == "failed":
                failure.append(f"arm {label!r}'s run {run_id} ended failed, so no further arm was launched")
        finally:
            slots.release()

    try:
        for spec, record in zip(arguments.arms, sweep.arms, strict=True):
            await slots.acquire()
            if failure:
                slots.release()
                break
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
                    scope_id=scope_id,
                )
            # prawduct:ok-broad-except — any refusal of one arm ends the sweep, recorded on its record
            except Exception as refused:
                slots.release()
                failure.append(
                    f"arm {record.label!r} was refused at its launch, so no further arm was launched: {refused}"
                )
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
        for run_id, waiter in in_flight.items():
            manager.cancel_job(run_id, reason=SWEEP_CANCEL_REASON)
            waiter.cancel()
        await manager.wait_for(list(in_flight))
        await asyncio.gather(*in_flight.values(), return_exceptions=True)
        sweep.outcome, sweep.detail, sweep.ended_at = "cancelled", "cancelled before every arm ran", utc_now_iso()
        await save()
        raise
    # prawduct:ok-broad-except — the sweep's record is where its ending is stated; the task boundary logs it
    except Exception as broke:
        log.exception("eval.sweep sweep=%s scope=%s ended on an error", sweep.id, scope_id)
        failure.append(f"the sweep ended on an error: {broke}")
        # Arms already launched are measurements in their own right: they finish and stay members.
        await asyncio.gather(*in_flight.values(), return_exceptions=True)
    sweep.outcome = "failed" if failure else "completed"
    sweep.detail = failure[0] if failure else None
    sweep.ended_at = utc_now_iso()
    await save()


__all__ = [
    "SWEEP_CANCEL_REASON",
    "SweepArguments",
    "SweepArm",
    "sweep_launch",
]
