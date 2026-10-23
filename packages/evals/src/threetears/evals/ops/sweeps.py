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
(``exploratory``) rather than leaving the caller to. A campaign's declaration is frozen once it holds a run, so
``campaign_update`` refuses to add one; what the arms were swept to learn is declared with ``campaign_create``: an
aspect over the sweep's ``run_ids`` (its job result names them), its arms named with ``from_run_ids``. A question
asked after the runs reads as asked, with ``question_after_runs`` disclosing when.

**Taken up again.** The sweep's record keeps what each arm launches with, and its job launches every arm from
that record alone. A sweep lost (its process gone), cancelled, or stopped by its own time budget is taken up again
as the same sweep (:func:`sweep_resume`, which ``job_resume`` calls for ``sweep:<sweep id>``): each arm whose run was
lost, cancelled, or stopped by its budget or its account is taken up as that same run, each arm that never launched
is launched as the sweep would have launched it, an arm whose run ended is left as it ended, and the arm schedule is
kept — an interleaved sweep's arms run in one group under the sweep's original launch group, so they take their
remaining cells in turn in the order they always had. Every arm is checked in the call, so one arm that would
measure otherwise refuses the whole take-up, naming every difference, before anything is spent.

**When it stops early.** If an arm is refused at its launch (the host's admission ceiling, say), or an arm's run
ends ``failed``, no further arm is launched and the sweep ends ``failed``, naming it. A harness fault is not
measured N more times overnight. Runs already launched finish and stay members. An interleaved sweep's arms are
launched together, so an arm refused at its launch starts none of them, and an arm whose run fails leaves its
siblings to finish.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from functools import partial
from typing import Annotated, Any

from pydantic import Field, model_validator

from threetears.evals.analysis.campaigns import add_runs_to_campaign
from threetears.evals.schema.aspect_label import aspect_label_for
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.kernel.campaign import (
    EvalSweep,
    SweepArmArguments,
    SweepArmRecord,
    SweepLaunchArguments,
    SweepLifetime,
    SweepOutcome,
    ended_sweep_lifetimes,
)
from threetears.evals.kernel.errors import (
    ConflictError,
    EvalServiceError,
    JobFenced,
    NotFoundError,
    ValidationFailedError,
)
from threetears.evals.kernel.refusals import job_refusal
from threetears.evals.schema.refusal import JobRefusal
from threetears.evals.kernel.job_timeout import EvalJobTimeout, timed_out_detail
from threetears.evals.schema.models import DEFAULT_LAUNCH_K_RUNS, ArmSchedule, CaseSetRef, EvalRun, utc_now_iso
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.ops.analysis import CampaignDefinition, campaign_create
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.jobs import JobHandle, JobsStarted, sweep_job_id, sweep_key
from threetears.evals.kernel.remedies import FIND_JOB
from threetears.evals.ops.runs import LaunchArguments
from threetears.evals.run.arm_schedule import resolve_arm_schedule
from threetears.evals.run.jobs import JOB_TIMEOUT_CAP_S
from threetears.evals.run.launch import LaunchGroup, admit_resumed_run, launch_as_group, quote_launch, start_run
from threetears.evals.kernel.job_leases import JobLeases, SWEEP_JOB, require_held, settle_job_record, write_held_record
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.run.lifecycle import get_run
from threetears.evals.run.resume import (
    RunResumption,
    next_epoch,
    refuse_a_job_not_taken_up,
    refused_as_one,
    start_take_up,
    take_up_arms,
    write_taken_up,
)
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
        "takes no design; declare one with campaign_create, an aspect over the sweep's run_ids and arms from "
        "from_run_ids.",
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

    @model_validator(mode="after")
    def _a_schedule_latency_can_be_read_under(self) -> SweepSettings:
        """Refuse arms asked to run at once beside latency under test, wherever the arguments are used.

        :raises ValueError: ``measure_latency`` with ``arm_schedule`` ``interleaved`` or ``concurrent``.
        """
        try:
            resolve_arm_schedule(self.arm_schedule, measure_latency=self.measure_latency)
        except ValidationFailedError as refused:
            raise ValueError(f"{refused.message}; {refused.remedy}") from refused
        return self

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


def _what_it_launches(arguments: SweepArguments) -> tuple[SweepLaunchArguments, list[SweepArmRecord]]:
    """What a sweep records of its launch: what every arm shares, and each arm with what it launches with.

    The sweep's job launches every arm from these alone, so a sweep taken up again launches an arm that never
    launched exactly as the sweep would have.
    """
    shared = SweepLaunchArguments(
        k_runs=arguments.k_runs,
        judge_model=arguments.judge_model,
        simulator_model=arguments.simulator_model,
        max_cost_usd=arguments.max_cost_usd,
        cell_timeout_s=arguments.cell_timeout_s,
        case_set=arguments.case_set,
        measure_latency=arguments.measure_latency,
    )
    arms = [
        SweepArmRecord(
            label=_arm_label(arm),
            model=arm.model,
            arguments=SweepArmArguments(
                overlays=arm.overlays,
                apparatus_settings=arm.apparatus_settings,
                judge_model=arm.judge_model,
                simulator_model=arm.simulator_model,
            ),
        )
        for arm in arguments.arms
    ]
    return shared, arms


async def _refuse_what_any_arm_would_refuse(
    host: OpsHost,
    arguments: SweepArguments,
    shared: SweepLaunchArguments,
    arms: list[SweepArmRecord],
    scope_id: str,
) -> None:
    """Quote every arm by the launch's own rule before anything is created, and refuse the sweep on any refusal.

    :param host: The launching host.
    :ptype host: OpsHost
    :param arguments: The sweep.
    :ptype arguments: SweepArguments
    :param shared: What its arms share, as the sweep records it.
    :ptype shared: SweepLaunchArguments
    :param arms: Its arms, as the sweep records them.
    :ptype arms: list[SweepArmRecord]
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
    for arm in arms:
        await _refuse_what_an_arm_would_refuse(
            host,
            template_id=arguments.template_id,
            subject_id=arguments.subject_id,
            shared=shared,
            arm=arm,
            scope_id=scope_id,
        )


async def _refuse_what_an_arm_would_refuse(
    host: OpsHost,
    *,
    template_id: str,
    subject_id: str,
    shared: SweepLaunchArguments,
    arm: SweepArmRecord,
    scope_id: str,
) -> None:
    """Quote one arm by the launch's own rule, from what the sweep records it launches with, and refuse it on any
    refusal, naming the arm.

    :raises ValidationFailedError: The arm's launch would be refused.
    :raises NotFoundError: The template, or the case set, is not in the scope.
    """
    own = arm.arguments if arm.arguments is not None else SweepArmArguments()
    try:
        quote = await quote_launch(
            host.launch,
            template_id=template_id,
            subject_id=subject_id,
            models=[arm.model],
            k_runs=shared.k_runs,
            judge_model=own.judge_model or shared.judge_model,
            simulator_model=own.simulator_model or shared.simulator_model,
            overlays=own.overlays,
            apparatus_settings=own.apparatus_settings,
            max_cost_usd=shared.max_cost_usd,
            scope_id=scope_id,
            cell_timeout_s=shared.cell_timeout_s,
            case_set=shared.case_set,
        )
    except ValidationFailedError as refused:
        raise ValidationFailedError(
            f"arm {arm.label!r} would be refused: {refused.message}", remedy=refused.remedy
        ) from refused
    if refused_arms := [verdict for verdict in quote.arms if verdict.refusal is not None]:
        raise ValidationFailedError(
            f"arm {arm.label!r} would be refused: " + "; ".join(str(verdict.refusal) for verdict in refused_arms),
            remedy="; ".join(dict.fromkeys(str(verdict.remedy) for verdict in refused_arms)),
        )


async def sweep_launch(host: OpsHost, arguments: SweepArguments, scope_id: str, *, created_by: str) -> JobsStarted:
    """Create the sweep's campaign and start the job that launches its arms, as its arm schedule says.

    The sweep's record keeps what each arm launches with (``EvalSweep.arguments``, ``SweepArmRecord.arguments``),
    and its job launches every arm from that record alone, so a sweep taken up again (:func:`sweep_resume`) launches
    an arm that never launched exactly as it would have.

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
    shared, arms = _what_it_launches(arguments)
    await _refuse_what_any_arm_would_refuse(host, arguments, shared, arms, scope_id)
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
                # One aspect, the sweep's template, which every arm's run joins as it is created.
                aspects=[
                    {
                        "label": aspect_label_for(arguments.campaign_behavior),
                        "template_id": arguments.template_id,
                    }
                ],
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
        arms=arms,
        max_concurrent_arms=_arms_at_once(arguments),
        arm_schedule=arguments.schedule,
        arguments=shared,
    )
    leases = host.launch.job_leases

    def save_started() -> None:
        # Written carrying its lease, so no reader ever finds the sweep running and unleased, which reads lost.
        sweep.lease = leases.grant()
        eval_host.storage.save_sweep(sweep)

    await run_blocking(eval_host.blocking_executor, save_started)

    async def work() -> None:
        await _run_sweep(host, sweep, budget_s=_budget_s(sweep))

    # The sweep enters its own budget (_run_sweep), so its record can say it timed out: one entered around the task
    # would reach the sweep as a cancel.
    manager.start_task(sweep.id, work, budget_s=None, key=sweep_key(sweep.id, scope_id), record=(SWEEP_JOB, scope_id))
    return JobsStarted(
        jobs=[JobHandle(job_id=sweep_job_id(sweep.id), kind="sweep", target_id=campaign.id, label=campaign.name)]
    )


def _budget_s(sweep: EvalSweep) -> float:
    """A sweep's wall-clock budget, for each holding of it: one arm's share per arm, and one for launching them."""
    return (len(sweep.arms) + 1) * SWEEP_ARM_BUDGET_S


@dataclass(frozen=True)
class _TakingUp:
    """What a sweep taken up again holds of its arms: each arm's run taken up again, and each arm that had ended.

    :ivar resumptions: Each arm's run taken up again, by the arm's label, as its take-up planned it in the call.
    :ivar ended: Each arm whose run had ended ``completed`` or ``failed``, by label: left alone, and read as it ended.
    """

    resumptions: dict[str, RunResumption] = field(default_factory=dict)
    ended: dict[str, EvalRun] = field(default_factory=dict)


def _never_launched(sweep: EvalSweep, arm: SweepArmRecord) -> ValidationFailedError:
    """The refusal of an arm that never launched, of a sweep stored before sweeps recorded their arms' arguments."""
    return ValidationFailedError(
        f"arm {arm.label!r} never launched, and sweep '{sweep.id}' was stored before sweeps recorded what each arm "
        "launches with, so it cannot be launched as the sweep would have launched it",
        remedy=(
            f"launch arm {arm.label!r} with run_launch, and resume each other arm by its run's job id ('run:<run id>', "
            "which job_poll on the sweep names)"
        ),
    )


async def sweep_resume(
    host: OpsHost, sweep_id: str, scope_id: str, *, max_cost_usd: float | None = None
) -> JobsStarted:
    """Take an interrupted sweep up again as the same sweep, and start its job, which keeps the arm schedule.

    A sweep lost (its process gone), cancelled by an operator, or stopped by its own time budget (``timed_out``: a
    ceiling, not a defect) is taken up again; one still running, completed or failed is not. In the call, before
    anything is spent:

    - each arm whose run ended ``completed`` or ``failed`` is left alone, and read as it ended;
    - each arm whose run was lost, cancelled, or stopped by its budget or its account is planned and its launch
      replayed and compared (:func:`~threetears.evals.run.resume.take_up_arms`);
    - each arm that never launched is quoted by the launch's own rule, from what the sweep recorded it launches with
      — one of a sweep stored before sweeps recorded that is refused, naming it.

    Any refusal of any arm refuses the whole take-up, naming every arm's every refusal, with nothing written. Then the
    sweep's record is reopened under its lease's next epoch, conditionally on the record as read, a new holding opens
    on it (``EvalSweep.lifetimes``), and its own job runs the arms as its schedule says: an interleaved sweep's in one
    group under the sweep's original launch group, so the arms take their remaining cells in turn in the order they
    always had; a sequential sweep's one after another, in order; a concurrent sweep's side by side. Its job supervises
    every arm as it supervises a launched one, and the sweep ends completed or failed as it would have.

    :param host: The host, whose launcher wires the arms and whose job manager holds them.
    :ptype host: OpsHost
    :param sweep_id: The sweep.
    :ptype sweep_id: str
    :param scope_id: Its scope.
    :ptype scope_id: str
    :param max_cost_usd: The cost cap every arm is taken up or launched under; ``None`` keeps each arm's own. Not
        part of the measurement: the sweep records it, and each holding records the cap it ran under.
    :ptype max_cost_usd: float | None
    :return: The sweep's one job (``sweep:<sweep id>``), whose target is its campaign.
    :rtype: JobsStarted
    :raises NotFoundError: There is no such sweep, or an arm's template is gone.
    :raises ValidationFailedError: The sweep cannot be taken up, it has no arm left to take up or launch, or an arm
        cannot be (every arm's refusal named).
    :raises ConflictError: Another take-up of it won, or another writer moved it, while this one prepared; nothing was
        started.
    """
    eval_host, launch = host.eval_host, host.launch
    storage, manager, leases = eval_host.storage, launch.job_manager, launch.job_leases
    sweep, etag = await run_blocking(eval_host.blocking_executor, _settled_sweep, storage, sweep_id, scope_id)
    if sweep is None:
        raise NotFoundError("sweep", sweep_id, remedy=FIND_JOB)
    refuse_a_job_not_taken_up(
        "sweep",
        sweep.id,
        sweep.outcome,
        sweep.detail,
        live_here=manager.is_task_active(sweep.id),
        start_again="sweep_launch",
    )
    launched = [arm.run_id for arm in sweep.arms if arm.run_id is not None]
    runs = {
        run.id: run
        for run in await run_blocking(eval_host.blocking_executor, storage.load_eval_runs, launched, scope_id)
    }
    shared = sweep.arguments
    if shared is not None and max_cost_usd is not None:
        shared = shared.model_copy(update={"max_cost_usd": max_cost_usd})
    refusals: list[ValidationFailedError] = []
    ended: dict[str, EvalRun] = {}
    unfinished: list[tuple[str, str]] = []
    never_launched: list[SweepArmRecord] = []
    for arm in sweep.arms:
        run = runs.get(arm.run_id) if arm.run_id is not None else None
        if arm.run_id is None and (shared is None or arm.arguments is None):
            refusals.append(_never_launched(sweep, arm))
        elif arm.run_id is None:
            never_launched.append(arm)
        elif run is None:
            refusals.append(
                ValidationFailedError(
                    f"arm {arm.label!r}'s run '{arm.run_id}' is not in scope {scope_id!r}",
                    remedy="launch the sweep again (sweep_launch)",
                )
            )
        elif run.status in ("completed", "failed"):
            ended[arm.label] = run
        else:
            unfinished.append((arm.label, run.id))
    planned, refused = await take_up_arms(launch, unfinished, scope_id, max_cost_usd=max_cost_usd)
    refusals += refused
    for arm in never_launched:
        assert shared is not None  # an arm the sweep holds no arguments for was refused above
        try:
            await _refuse_what_an_arm_would_refuse(
                host,
                template_id=sweep.template_id,
                subject_id=sweep.subject_id,
                shared=shared,
                arm=arm,
                scope_id=scope_id,
            )
        except ValidationFailedError as refusal:
            refusals.append(refusal)
    if refusals:
        raise refused_as_one(f"sweep '{sweep.id}'", refusals)
    if not planned and not never_launched:
        raise ValidationFailedError(
            f"every arm of sweep '{sweep.id}' has ended completed or failed, so there is nothing to take up",
            remedy="read how each arm ended with job_poll on the sweep",
        )
    epoch = next_epoch(sweep.lease, sweep.lost, (lifetime.epoch for lifetime in sweep.lifetimes))
    # An interleaved sweep stored before it recorded its group carries it on every arm's run.
    group_id = sweep.launch_group_id
    if group_id is None and sweep.arm_schedule == "interleaved":
        group_id = next((run.launch_group_id for run in runs.values() if run.launch_group_id is not None), None)
    reopened = _reopened(
        sweep, epoch=epoch, holder=leases.holder_id, shared=shared, group_id=group_id, cap=max_cost_usd
    )
    await run_blocking(eval_host.blocking_executor, _write_reopened_sweep, storage, leases, reopened, etag, epoch)
    taking_up = _TakingUp(resumptions=planned, ended=ended)

    async def work() -> None:
        await _run_sweep(host, reopened, budget_s=_budget_s(reopened), taking_up=taking_up)

    manager.start_task(
        reopened.id,
        work,
        budget_s=None,
        key=sweep_key(reopened.id, scope_id),
        record=(SWEEP_JOB, scope_id),
        epoch=epoch,
    )
    log.info(
        "eval.sweep_resume sweep=%s scope=%s taken up at epoch %d: %d arm(s) taken up, %d to launch, %d ended",
        sweep.id,
        scope_id,
        epoch,
        len(planned),
        len(never_launched),
        len(ended),
    )
    campaign = await run_blocking(eval_host.blocking_executor, storage.load_campaign, sweep.campaign_id, scope_id)
    return JobsStarted(
        jobs=[
            JobHandle(
                job_id=sweep_job_id(sweep.id),
                kind="sweep",
                target_id=sweep.campaign_id,
                label=campaign.name if campaign is not None else sweep.campaign_id,
            )
        ]
    )


def _settled_sweep(storage: EvalStorage, sweep_id: str, scope_id: str) -> tuple[EvalSweep | None, str | None]:
    """A sweep's record as every reader settles it, then read with the token its reopen is written conditionally on."""
    settle_job_record(storage, SWEEP_JOB, sweep_id, scope_id)
    return storage.load_job_record(EvalSweep, sweep_id, scope_id)


def _reopened(
    sweep: EvalSweep,
    *,
    epoch: int,
    holder: str,
    shared: SweepLaunchArguments | None,
    group_id: str | None,
    cap: float | None,
) -> EvalSweep:
    """A sweep as a take-up reopens it: ``running`` again, its ending moved to the holding that ended, a new one open.

    :param sweep: The sweep as stored.
    :ptype sweep: EvalSweep
    :param epoch: The lease epoch it is taken up under.
    :ptype epoch: int
    :param holder: The process taking it up.
    :ptype holder: str
    :param shared: What its arms share, a raised cap carried; ``None`` for a sweep stored before it recorded that.
    :ptype shared: SweepLaunchArguments | None
    :param group_id: Its interleaved arms' launch group.
    :ptype group_id: str | None
    :param cap: The cap the take-up named, or ``None``.
    :ptype cap: float | None
    :return: The reopened record, its lease still the stored one (the take-up's write grants the next).
    :rtype: EvalSweep
    """
    recorded_cap = sweep.arguments.max_cost_usd if sweep.arguments is not None else None
    first = SweepLifetime(
        holder=sweep.lease.holder if sweep.lease is not None else None,
        epoch=epoch - 1,
        started_at=sweep.created_at,
        max_cost_usd=recorded_cap,
    )
    ended = ended_sweep_lifetimes(
        sweep.lifetimes or [first],
        outcome=sweep.outcome,
        at=sweep.ended_at or utc_now_iso(),
        detail=sweep.detail,
        lost=sweep.lost,
    )
    holding = SweepLifetime(
        holder=holder,
        epoch=epoch,
        started_at=utc_now_iso(),
        max_cost_usd=cap if cap is not None else recorded_cap,
    )
    return sweep.model_copy(
        update={
            "outcome": "running",
            "detail": None,
            "ended_at": None,
            "lost": None,
            "refusal": None,
            "cancel_requested_at": None,
            "cancel_request_reason": None,
            "arguments": shared,
            "launch_group_id": group_id,
            "lifetimes": [*ended, holding],
        }
    )


def _write_reopened_sweep(
    storage: EvalStorage, leases: JobLeases, sweep: EvalSweep, etag: str | None, epoch: int
) -> None:
    """Write the reopened sweep under a lease at its next epoch, conditionally on the record the take-up read.

    :raises ConflictError: The record moved since it was read: another take-up won, or another writer wrote it.
    """
    sweep.lease = leases.grant(epoch)
    try:
        storage.save_sweep(sweep, if_match=etag)
    except ConflictError as moved:
        raise ConflictError(
            f"sweep '{sweep.id}' changed while it was being taken up — another take-up of it won, or another writer "
            "moved it — so this one started nothing",
            remedy="read it with job_poll; call job_resume again if it is still not running",
        ) from moved


async def _run_sweep(host: OpsHost, sweep: EvalSweep, *, budget_s: float, taking_up: _TakingUp | None = None) -> None:
    """Launch the arms as the sweep's schedule says, joining each run to the campaign, and wait for them.

    An interleaved sweep prepares every arm into one launch and starts them together
    (:meth:`_SweepRun.launch_together`); any other launches each arm as its own launch, in order,
    ``max_concurrent_arms`` at once (:meth:`_SweepRun.launch_in_order`). Every arm launches from what the sweep's
    record says it launches with. A sweep taken up again (``taking_up``) runs the same way: an arm whose run had ended
    is read as it ended, an arm whose run is taken up again is started as that run, and an arm that never launched is
    launched.

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
    :param sweep: Its record, as started or as reopened.
    :ptype sweep: EvalSweep
    :param budget_s: The sweep's wall-clock budget.
    :ptype budget_s: float
    :param taking_up: For a sweep taken up again, what it holds of its arms; ``None`` for a sweep starting fresh.
    :ptype taking_up: _TakingUp | None
    """
    manager = host.launch.job_manager
    scope_id = sweep.scope_id
    run = _SweepRun(host, sweep, taking_up if taking_up is not None else _TakingUp())

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
        run.end("cancelled", "cancelled before every arm ran")
        try:
            await run.save()
        except JobFenced as why:
            # NOSILENT: the sweep's record, written lost elsewhere, is its ending; this process writes nothing over it.
            fenced(why)
        raise
    except EvalJobTimeout as timed_out:
        log.warning("eval.sweep sweep=%s scope=%s timed out: %s", sweep.id, scope_id, timed_out)
        await run.stop_arms(SWEEP_TIMEOUT_REASON)
        run.end("timed_out", timed_out_detail(timed_out))
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
        contain(
            broke,
            log=log,
            what=f"eval.sweep sweep={sweep.id} scope={scope_id} ended on an error",
            level="error",
        )
        run.fail(f"the sweep ended on an error: {broke}", broke)
        # Arms already launched are measurements in their own right: they finish and stay members.
        await asyncio.gather(*run.in_flight.values(), return_exceptions=True)
    sweep.refusal = run.refusal
    run.end(
        "failed" if run.failure else "completed",
        run.failure[0] if run.failure else "; ".join(run.measured_nothing) or None,
    )
    try:
        await run.save()
    except JobFenced as why:
        # NOSILENT: the sweep's record, written lost elsewhere, is its ending; this process writes nothing over it.
        fenced(why)


class _SweepRun:
    """One sweep's job as it runs: its arms in flight, what failed, and how each schedule launches its arms."""

    def __init__(self, host: OpsHost, sweep: EvalSweep, taking_up: _TakingUp) -> None:
        self.host = host
        self.sweep = sweep
        self.taking_up = taking_up
        self.in_flight: dict[str, asyncio.Task[None]] = {}
        self.failure: list[str] = []
        #: Each arm that measured nothing, said on the sweep's detail when nothing else failed it.
        self.measured_nothing: list[str] = []
        # The engine refusal the sweep ended on, kept as data beside its words: set only for the failure the sweep's
        # detail states (the first), since a refusal after another failure did not end it.
        self.refusal: JobRefusal | None = None

    def fail(self, why: str, error: BaseException | None = None) -> None:
        """Record a failure, keeping the engine refusal behind the first as data."""
        self.failure.append(why)
        if isinstance(error, EvalServiceError) and len(self.failure) == 1:
            self.refusal = job_refusal(error)

    def end(self, outcome: SweepOutcome, detail: str | None) -> None:
        """Set the sweep's ending on its record; its model closes the holding a take-up opened from it."""
        sweep = self.sweep
        sweep.outcome, sweep.detail, sweep.ended_at = outcome, detail, utc_now_iso()

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
            # An arm that measured nothing (every cell excluded: a model no allowed provider serves, a decline that
            # lasted its run) failed as a measurement of that arm alone, so the sweep goes on to the others and says
            # so; any other failure of an arm's run, a reason this build does not know included, ends the sweep.
            if run.status == "failed" and run.failed_because == "nothing_measured":
                self.measured_nothing.append(f"arm {label!r} measured nothing (its run {run_id} failed)")
            elif run.status == "failed":
                after = "so no further arm was launched" if done is not None else "beside its siblings"
                why = f" ({run.failed_because})" if run.failed_because is not None else ""
                self.failure.append(f"arm {label!r}'s run {run_id} ended failed{why}, {after}")
        finally:
            if done is not None:
                done.release()

    async def join_campaign(self, runs: list[EvalRun]) -> None:
        if not runs:
            return
        eval_host = self.host.eval_host
        await run_blocking(
            eval_host.blocking_executor,
            add_runs_to_campaign,
            eval_host.storage,
            self.sweep.campaign_id,
            self.sweep.scope_id,
            [run.id for run in runs],
        )

    def _recorded(self, arm: SweepArmRecord) -> tuple[SweepLaunchArguments, SweepArmArguments]:
        """What the sweep records an arm launches with.

        :raises ValidationFailedError: The sweep was stored before it recorded that; a take-up refuses such an arm in
            its call, so a launch never reaches here without it.
        """
        if self.sweep.arguments is None or arm.arguments is None:
            raise _never_launched(self.sweep, arm)
        return self.sweep.arguments, arm.arguments

    def _judge(self, arm: SweepArmRecord, pin: str | None) -> str | None:
        # An arm whose own candidate is the pinned judge is left to the launch's pin resolution: no model grades itself.
        own = arm.arguments.judge_model if arm.arguments is not None else None
        return own or (pin if pin != arm.model else None)

    def _simulator(self, arm: SweepArmRecord, pin: str | None) -> str | None:
        own = arm.arguments.simulator_model if arm.arguments is not None else None
        return own or pin

    async def launch_in_order(self) -> None:
        """Launch each arm as its own launch, in order, at most ``max_concurrent_arms`` at once.

        An arm whose run had ended before the sweep was taken up again takes its turn as it ended; an arm whose run is
        taken up again is started as that run, in its turn.
        """
        sweep, taking_up = self.sweep, self.taking_up
        slots = asyncio.Semaphore(sweep.max_concurrent_arms)
        shared = sweep.arguments
        judge_pin = shared.judge_model if shared is not None else None
        simulator_pin = shared.simulator_model if shared is not None else None
        for record in sweep.arms:
            await slots.acquire()
            if self.failure:
                slots.release()
                return
            if (ended := taking_up.ended.get(record.label)) is not None:
                # Read as it ended, as a launched arm's run is once it ends: a failure ends the sweep here, as it did.
                self.in_flight[ended.id] = asyncio.create_task(self.settle(record.label, ended.id, done=slots))
                judge_pin = judge_pin or ended.judge_model
                simulator_pin = simulator_pin or ended.simulator_model
                continue
            # Fenced BEFORE the arm is launched, not only at the write after it: a sweep whose record reads lost
            # would otherwise pay for one more arm that nothing records, whose run no sweep and no campaign names.
            try:
                await self.require_held()
            except JobFenced:
                slots.release()
                raise
            resuming = taking_up.resumptions.get(record.label)
            try:
                if resuming is not None:
                    run = await start_take_up(self.host.launch, resuming)
                else:
                    (run,) = await start_run(
                        self.host.launch,
                        **self._arm_launch(record, judge_model=self._judge(record, judge_pin)),
                        simulator_model=self._simulator(record, simulator_pin),
                        arm_schedule=sweep.arm_schedule,
                    )
            # prawduct:ok-broad-except — any refusal of one arm ends the sweep, recorded on its record
            except Exception as refused:
                contain(refused, log=log, what=f"eval.sweep arm {record.label!r} was refused at its launch")
                slots.release()
                self.fail(
                    f"arm {record.label!r} was refused at its launch, so no further arm was launched: {refused}",
                    refused,
                )
                return
            self.in_flight[run.id] = asyncio.create_task(self.settle(record.label, run.id, done=slots))
            if resuming is None:
                record.run_id = run.id
                await self.join_campaign([run])
            await self.save()
            # The first arm's resolved instruments hold for every later arm that names none of its own.
            judge_pin = judge_pin or run.judge_model
            simulator_pin = simulator_pin or run.simulator_model

    async def launch_together(self) -> None:
        """Prepare every arm into one interleaved launch and start them together, every run then joining the campaign.

        The launch resolves what its arms share once (the subject's snapshot, the case set, the judge's default), so no
        arm's instruments are pinned from another's run. An arm refused at its launch starts none of them. A sweep
        taken up again prepares its arms into a group under its original launch group's id, so its arms' cells keep
        the order they had: an arm whose run is taken up again joins as that run, one that never launched joins anew,
        and one whose run had ended is read as it ended.
        """
        sweep, taking_up = self.sweep, self.taking_up
        await self.require_held()
        group = LaunchGroup(
            candidate_models=list(dict.fromkeys(arm.model for arm in sweep.arms)),
            arm_schedule="interleaved",
            group_id=sweep.launch_group_id,
        )
        sweep.launch_group_id = group.id
        going = [record for record in sweep.arms if record.label not in taking_up.ended]
        shared = sweep.arguments
        preparing: list[str] = []

        async def form() -> tuple[LaunchGroup, None]:
            return group, None

        async def prepare(group: LaunchGroup, _formed: None) -> list[EvalRun]:
            prepared: list[EvalRun] = []
            for record in going:
                preparing[:] = [record.label]
                resuming = taking_up.resumptions.get(record.label)
                if resuming is not None:
                    prepared += await (await admit_resumed_run(self.host.launch, resuming, launch_group=group)).start()
                    continue
                prepared += await start_run(
                    self.host.launch,
                    **self._arm_launch(
                        record, judge_model=self._judge(record, shared.judge_model if shared is not None else None)
                    ),
                    simulator_model=self._simulator(record, shared.simulator_model if shared is not None else None),
                    launch_group=group,
                )
            preparing.clear()
            # Asked again once every arm is prepared and before any starts: preparing can be long (a generating
            # launch writes its cases), and a sweep whose record read lost meanwhile would start every arm unowned.
            # Raised here, it abandons the launch, and nothing starts.
            await self.require_held()
            return prepared

        async def reopen(runs: list[EvalRun]) -> None:
            await write_taken_up(self.host.launch, list(taking_up.resumptions.values()), runs)

        try:
            runs = await launch_as_group(
                self.host.launch,
                len(going),
                settings=self.host.launch.settings(),
                form=form,
                prepare=prepare,
                event="eval.sweep",
                attach=reopen if taking_up.resumptions else None,
            )
        # prawduct:ok-broad-except — any refusal of one arm ends the sweep, recorded on its record
        except Exception as refused:
            if isinstance(refused, JobFenced):
                raise
            which = f"arm {preparing[0]!r} was refused at its launch" if preparing else "the arms' launch was refused"
            contain(refused, log=log, what=f"eval.sweep: {which}")
            self.fail(f"{which}, so no arm was launched: {refused}", refused)
            return
        for record, run in zip(going, runs, strict=True):
            self.in_flight[run.id] = asyncio.create_task(self.settle(record.label, run.id))
            record.run_id = run.id
        for label, ended in taking_up.ended.items():
            self.in_flight[ended.id] = asyncio.create_task(self.settle(label, ended.id))
        # Members once they exist, as an arm launched on its own joins once its run is created.
        await self.join_campaign(
            [run for record, run in zip(going, runs, strict=True) if record.label not in taking_up.resumptions]
        )
        await self.save()

    def _arm_launch(self, arm: SweepArmRecord, *, judge_model: str | None) -> dict[str, Any]:
        """What every launch of one arm names, whichever schedule launches it: what the sweep records it launches
        with."""
        shared, own = self._recorded(arm)
        return {
            "template_id": self.sweep.template_id,
            "subject_id": self.sweep.subject_id,
            "models": [arm.model],
            "k_runs": shared.k_runs,
            "judge_model": judge_model,
            "overlays": own.overlays,
            "apparatus_settings": own.apparatus_settings,
            "max_cost_usd": shared.max_cost_usd,
            "cell_timeout_s": shared.cell_timeout_s,
            "case_set": shared.case_set,
            "measure_latency": shared.measure_latency,
            "scope_id": self.sweep.scope_id,
        }


__all__ = [
    "SWEEP_ARM_BUDGET_S",
    "SWEEP_CANCEL_REASON",
    "SWEEP_TIMEOUT_REASON",
    "SweepArguments",
    "SweepArm",
    "sweep_launch",
    "sweep_resume",
]
