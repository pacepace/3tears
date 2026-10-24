"""The run side's operations: what a scope holds, one run's summary, a launch, and a run's curation.

Each takes the host and the scope and returns a typed value — never a ``dict`` a surface re-types — so
a CLI, an MCP action and a REST route that call one read one shape. A launch, and every operation here
whose work calls a model, follows the job contract (:mod:`threetears.evals.ops.jobs`): the call prices, caps and
admits the work and returns its one job, and the job does the work and writes its record.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field, model_validator

from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.logs import get_logger
from threetears.evals.kernel.errors import EvalServiceError, ValidationFailedError
from threetears.evals.kernel.host import EvalHost
from threetears.evals.schema.models import (
    DEFAULT_LAUNCH_K_RUNS,
    CalibrationRating,
    NON_TERMINAL_RUN_STATUSES,
    ArmSchedule,
    CaseSet,
    CaseSetRef,
    EvalRun,
    SecondJudge,
)
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.jobs import AdmittedJob, JobsStarted, start_job
from threetears.evals.analysis.summary import EvalSummary, summarize_run
from threetears.evals.run.arm_schedule import resolve_arm_schedule
from threetears.evals.run.authoring import list_judge_configs, list_templates
from threetears.evals.run.case_sets import mint_case_set
from threetears.evals.run.curation import delete_run, delete_unreadable_run, set_run_archived
from threetears.evals.run.run_document import set_run_labels
from threetears.evals.run.launch import admit_launch
from threetears.evals.run.launch_job import launch_job_work
from threetears.evals.run.judge_repeat import (
    JudgeRepeatEstimate,
    estimate_judge_repeat,
    JUDGE_REPEAT_ACTION,
    judge_repeat_work,
)
from threetears.evals.run.judge_second import (
    DEFAULT_SECOND_JUDGE_SEED,
    SecondJudgeAskedBy,
    SecondJudgeEstimate,
    estimate_second_judge,
    second_judge_work,
)
from threetears.evals.kernel.judge_temperature import DEFAULT_TEMPERATURE_REPEATS, TemperatureSelection
from threetears.evals.run.judge_temperature import (
    JudgeTemperatureEstimate,
    estimate_judge_temperature_comparison,
    judge_temperature_work,
)
from threetears.evals.run.lifecycle import get_run
from threetears.evals.run.ratings import rate_result
from threetears.evals.run.reads import list_runs_disclosing

log = get_logger(__name__)


class TemplateLine(EvalBaseModel):
    """One template, as a listing shows it."""

    id: str
    name: str
    revision: int = Field(
        description="Which record of its name this is: 1 when first written, one more for each definition reset, "
        "whose new content is a new record. A launch by name runs the newest non-archived one."
    )
    candidate_kind: str
    intent: str
    archived: bool


class TemplateListing(EvalBaseModel):
    """A scope's templates."""

    templates: list[TemplateLine]


class JudgeConfigLine(EvalBaseModel):
    """One judge config, as a listing shows it: the versioned prompt a judge is asked one dimension with."""

    id: str
    name: str
    rubric_dim_id: str = Field(description="The dimension this config asks a judge for.")
    model: str = Field(description="The judge model it pins for its dimension; blank when it pins none.")
    temperature: float
    archived: bool


class JudgeConfigListing(EvalBaseModel):
    """A scope's judge configs."""

    judge_configs: list[JudgeConfigLine]


class RunLine(EvalBaseModel):
    """One run, as a listing shows it."""

    id: str
    status: str
    candidate_model: str
    template_id: str | None
    created_at: str
    archived: bool
    labels: list[str] = Field(default_factory=list, description="The labels an operator gave the run, sorted.")


class RunListing(EvalBaseModel):
    """A scope's runs, newest first as the store lists them."""

    runs: list[RunLine]
    include_archived: bool = Field(description="Whether archived runs were listed; they are left out by default.")
    label: str | None = Field(default=None, description="The label the runs were listed by; None lists every run.")
    unreadable: int = Field(
        ge=0,
        description=(
            "Stored runs the listing matched by status that this build's model cannot read, so they are not "
            "listed; runs_unreadable names each with why. Their archive flag and labels cannot be read, so the "
            "archive and label filters do not narrow this count."
        ),
    )


class UnreadableRunLine(EvalBaseModel):
    """One stored run this build's model cannot read, and why."""

    run_id: str | None = Field(description="The id the stored row states; None when it states none as text.")
    status: str | None = Field(
        description="The status the row was stored with, as stored; None when it states none as text."
    )
    stranded: bool = Field(
        description=(
            "Stored pending or running: the boot sweep counts it unreadable and can never settle it, so it reads "
            "as not ended until run_delete removes it or a build that reads it is deployed."
        )
    )
    reasons: list[str] = Field(
        description="Each distinct refusal the model gave, as `field: what it said`, sorted.", min_length=1
    )


class UnreadableRunsListing(EvalBaseModel):
    """The scope's stored runs that this build's model cannot read, each with why."""

    scope_id: str
    runs_read: int = Field(
        description="Every stored run row in the scope that was read, of any status, archived and unreadable included."
    )
    stranded: int = Field(
        description=(
            "Of the unreadable, those stored pending or running: the count the boot sweep reports as unreadable "
            "for this scope, unless a readable run's lost write was also refused while it ran."
        )
    )
    runs: list[UnreadableRunLine] = Field(description="Every row that does not read, stranded ones first, then by id.")


class LaunchArguments(EvalBaseModel):
    """What a launch names: the template, the subject, one arm per model, and the run's own limits.

    The one declaration of a launch's arguments — their types, bounds and descriptions. The ``run_launch``
    and ``launch_estimate`` actions' parameters derive from it, so the operation and what an agent is
    offered cannot drift apart.
    """

    template_id: str = Field(min_length=1, description="A template's id, as templates_list names it.")
    subject_id: str = Field(min_length=1, description="The subject the runs measure, as the host names it.")
    models: list[str] = Field(
        default_factory=list,
        description="Candidate models, one arm and one run each; empty runs the kind's own default.",
    )
    k_runs: int = Field(default=DEFAULT_LAUNCH_K_RUNS, ge=1, description="Repeats of every case, for pass^k.")
    n_variations: int = Field(
        default=0,
        ge=0,
        description="New cases to generate from the template's variation axes; 0 runs its stored cases.",
    )
    variation_model: str | None = Field(
        default=None,
        description="The model that writes the template's llm variation axes' values; required when n_variations "
        "generates for such an axis, refused otherwise.",
    )
    overlays: dict[str, Any] | None = Field(
        default=None, description="The knobs this launch turns on the template's kind, by field."
    )
    apparatus_settings: dict[str, Any] | None = Field(
        default=None,
        description="Host-declared apparatus values to set the runs' rig up with, by apparatus dimension (e.g. who sits "
        "in an adjudicator's seat) — each a string, a bool or a number, and one the template's kind reads; refused "
        "otherwise. Recorded on every run and part of its measurement context, so one template can be compared at two.",
    )
    max_cost_usd: float | None = Field(
        default=None,
        gt=0,
        description="A per-run cost cap in dollars, at or below the host's ceiling; it can only lower that ceiling, "
        "and a value above it is refused.",
    )
    judge_model: str | None = Field(default=None, description="The judge model, where the kind is model-judged.")
    simulator_model: str | None = Field(default=None, description="The simulated user's model, where the kind has one.")
    cell_timeout_s: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description="The deadline each cell runs under, in seconds, in place of the kind's own; at or below the "
        "host's ceiling (a host declaring none allows only lowering the kind's deadline). Recorded on every run.",
    )
    case_set_name: str | None = Field(
        default=None,
        min_length=1,
        description="A named case set of the template to run, as case_sets_list names it; every arm runs exactly its "
        "cases and records it. Named with case_set_version, and refused beside n_variations.",
    )
    case_set_version: int | None = Field(
        default=None, ge=1, description="The version of case_set_name to run; required with it."
    )
    measure_latency: bool = Field(
        default=False,
        description=(
            "Declare latency under test: each run executes its cases one at a time, with no other run beside it. "
            "False runs several cases at once — far faster — and the latency then recorded is marked read under "
            "concurrency and never compared."
        ),
    )
    arm_schedule: ArmSchedule | None = Field(
        default=None,
        description=(
            "How the arms share time and provider capacity. interleaved (the default): every arm at once, their "
            "cases taken in turn case by case under one width shared by the launch, which costs nothing and absorbs a "
            "provider's bad minute across every arm. sequential: one arm after another. concurrent: every arm side by side at its own "
            "width, the fastest, loading the providers with every arm at once, so one arm can take a provider's "
            "rationing. Latency under test runs the arms sequential and takes no other. Recorded on every run."
        ),
    )

    @model_validator(mode="after")
    def _a_schedule_latency_can_be_read_under(self) -> LaunchArguments:
        """Refuse arms asked to run at once beside latency under test, wherever the arguments are used.

        :raises ValueError: ``measure_latency`` with ``arm_schedule`` ``interleaved`` or ``concurrent``.
        """
        try:
            resolve_arm_schedule(self.arm_schedule, measure_latency=self.measure_latency)
        except ValidationFailedError as refused:
            raise ValueError(f"{refused.message}; {refused.remedy}") from refused
        return self

    @model_validator(mode="after")
    def _a_case_set_names_its_version(self) -> LaunchArguments:
        """Refuse a case set's name without its version, or a version without a name."""
        if (self.case_set_name is None) != (self.case_set_version is None):
            raise ValueError(
                "case_set_name and case_set_version name one version of one set together; a set's versions hold "
                "different cases, so neither is read without the other"
            )
        return self

    @property
    def case_set(self) -> CaseSetRef | None:
        """The case set the launch targets, or ``None``."""
        if self.case_set_name is None or self.case_set_version is None:
            return None
        return CaseSetRef(name=self.case_set_name, version=self.case_set_version)


class CaseSetLine(EvalBaseModel):
    """One version of a named case set."""

    name: str
    version: int
    label: str
    template_id: str
    test_case_ids: list[str]
    tracked: bool
    created_at: str


class CaseSetListing(EvalBaseModel):
    """A scope's case sets, every version, newest version first within a name."""

    case_sets: list[CaseSetLine]


class CaseSetMint(EvalBaseModel):
    """What minting a case set's next version names: the set, its template and its cases in order."""

    case_set: str = Field(min_length=1, description="The case set's name; a new name starts at version 1.")
    template_id: Annotated[str, LaunchArguments.model_fields["template_id"]]
    test_case_ids: list[str] = Field(
        min_length=1,
        description="The cases, in order, each a stored case of the template; a list the latest version already "
        "holds is refused.",
    )
    tracked: bool = Field(
        default=True, description="Whether the set is a standing suite followed over time, or made for one launch."
    )


def _case_set_line(case_set: CaseSet) -> CaseSetLine:
    return CaseSetLine(
        name=case_set.name,
        version=case_set.version,
        label=case_set.ref.label,
        template_id=case_set.template_id,
        test_case_ids=list(case_set.test_case_ids),
        tracked=case_set.tracked,
        created_at=case_set.created_at,
    )


def case_set_mint(host: EvalHost, arguments: CaseSetMint, scope_id: str) -> CaseSetLine:
    """Store the next version of a named case set: append-only, so a change is a new version, never an edit.

    :param host: The host whose store holds the set.
    :ptype host: EvalHost
    :param arguments: The set, its template and its cases.
    :ptype arguments: CaseSetMint
    :param scope_id: The scope the set lives in.
    :ptype scope_id: str
    :return: The stored version.
    :rtype: CaseSetLine
    :raises ValidationFailedError: Any refusal :func:`~threetears.evals.run.case_sets.mint_case_set` makes.
    :raises ConflictError: Another writer stored this version first.
    """
    return _case_set_line(
        mint_case_set(
            host.storage,
            scope_id=scope_id,
            name=arguments.case_set,
            template_id=arguments.template_id,
            test_case_ids=arguments.test_case_ids,
            tracked=arguments.tracked,
        )
    )


def case_sets_list(host: EvalHost, scope_id: str, *, name: str | None = None) -> CaseSetListing:
    """The scope's case sets — every version, or every version of one name.

    :param host: The host whose store is read.
    :ptype host: EvalHost
    :param scope_id: The scope.
    :ptype scope_id: str
    :param name: Only this set's versions.
    :ptype name: str | None
    :return: The listing.
    :rtype: CaseSetListing
    """
    return CaseSetListing(case_sets=[_case_set_line(s) for s in host.storage.query_case_sets(scope_id, name=name)])


class RunDeleted(EvalBaseModel):
    """What deleting a run removed."""

    run_id: str
    results_deleted: int
    campaigns_detached: list[str]


def _line(run: EvalRun) -> RunLine:
    return RunLine(
        id=run.id,
        status=run.status,
        candidate_model=run.candidate_model,
        template_id=run.template_id,
        created_at=run.created_at,
        archived=run.archived,
        labels=list(run.labels),
    )


def templates_list(host: EvalHost, scope_id: str, *, archived: bool = False) -> TemplateListing:
    """The scope's templates, each with its revision: a reset template's older revisions are archived, so the active
    listing shows each name once, at the revision a launch by name runs.

    :param host: The host whose store is read.
    :ptype host: EvalHost
    :param scope_id: The scope.
    :ptype scope_id: str
    :param archived: List the archived templates instead of the active ones (a reset template's older revisions among
        them).
    :ptype archived: bool
    :return: The listing.
    :rtype: TemplateListing
    """
    return TemplateListing(
        templates=[
            TemplateLine(
                id=template.id,
                name=template.name,
                revision=template.revision,
                candidate_kind=template.candidate_kind,
                intent=template.intent,
                archived=template.archived,
            )
            for template in list_templates(host.storage, scope_id, archived=archived)
        ]
    )


def judge_configs_list(
    host: EvalHost, scope_id: str, *, rubric_dim_id: str | None = None, archived: bool = False
) -> JudgeConfigListing:
    """The scope's judge configs — what a second judge's ``config_ids`` and a launch's ``judge_config_ids`` name.

    :param host: The host whose store is read.
    :ptype host: EvalHost
    :param scope_id: The scope.
    :ptype scope_id: str
    :param rubric_dim_id: Only the configs asking for this dimension; None for every dimension's.
    :ptype rubric_dim_id: str | None
    :param archived: List the archived (superseded) versions instead of the active ones.
    :ptype archived: bool
    :return: The listing.
    :rtype: JudgeConfigListing
    """
    return JudgeConfigListing(
        judge_configs=[
            JudgeConfigLine(
                id=config.id,
                name=config.name,
                rubric_dim_id=config.rubric_dim_id,
                model=config.model,
                temperature=config.temperature,
                archived=config.archived,
            )
            for config in list_judge_configs(host.storage, scope_id, rubric_dim_id=rubric_dim_id, archived=archived)
        ]
    )


def runs_list(
    host: EvalHost,
    scope_id: str,
    *,
    status: str | None = None,
    include_archived: bool = False,
    label: str | None = None,
) -> RunListing:
    """The scope's runs, and how many it matched that this build cannot read.

    An inventory, so a run this build's model cannot read is left out and counted rather than refusing the listing
    (:func:`~threetears.evals.run.list_runs_disclosing`): the operator opening it is looking for a run, and one row
    nothing can read must not hide every other. ``unreadable`` says how many, and ``runs_unreadable`` names them.

    :param host: The host whose store is read.
    :ptype host: EvalHost
    :param scope_id: The scope.
    :ptype scope_id: str
    :param status: Only runs with this stored status.
    :ptype status: str | None
    :param include_archived: List archived runs too.
    :ptype include_archived: bool
    :param label: Only runs carrying this label (:func:`run_label`); None lists every run.
    :ptype label: str | None
    :return: The listing.
    :rtype: RunListing
    :raises ValidationFailedError: ``status`` is not a run status.
    :raises UnreadableRunError: A run read, but writing it lost was refused by a model.
    """
    label = None if label is None else label.strip()
    read = list_runs_disclosing(host, scope_id, status=status, include_archived=include_archived, label=label)
    return RunListing(
        runs=[_line(run) for run in read.runs],
        include_archived=include_archived,
        label=label,
        unreadable=len(read.unreadable),
    )


def runs_unreadable(host: EvalHost, scope_id: str) -> UnreadableRunsListing:
    """The scope's stored runs this build's model cannot read, each with why, and how many of them are stranded.

    A run's document is what is read: a run whose own document reads but one of whose results does not is not
    here, as the boot sweep does not count it either — settling such a run reads past the result, writing its
    completeness unknown and naming the result in its ``error_details``, so it ends and says why. Every run in the scope
    is read, of any status, archived ones included, through the store's isolating scan
    (:meth:`~threetears.evals.kernel.EvalStorage.scan_eval_runs`): the same refusal the boot sweep's scan counts, so
    ``stranded`` here and the sweep's ``unreadable`` for the scope are the same rows. A sweep that also had a readable
    run's lost write refused counts that one too, and it is not here, since it reads.

    Derived on every call and never stored, as :func:`~threetears.evals.ops.analyses_undescribable` is. One row that
    does not read never stops the rest from being read, and every such row is named; a store that fails to answer
    raises rather than reading as a scope with none.

    :param host: The host whose store is read; its listing elisions are left out of every row, which never changes
        whether a row reads.
    :ptype host: EvalHost
    :param scope_id: The scope.
    :ptype scope_id: str
    :return: The listing.
    :rtype: UnreadableRunsListing
    :raises StorageError: The store failed to answer.
    """
    scan = host.storage.scan_eval_runs(scope_id, elide_payload=host.profile.listing_elisions)
    lines = [
        UnreadableRunLine(
            run_id=row.row_id,
            status=row.status,
            stranded=row.status in NON_TERMINAL_RUN_STATUSES,
            reasons=row.reasons,
        )
        for row in scan.unreadable
    ]
    lines.sort(key=lambda line: (not line.stranded, line.run_id is None, line.run_id or ""))
    return UnreadableRunsListing(
        scope_id=scope_id,
        runs_read=scan.readable + len(scan.unreadable),
        stranded=sum(line.stranded for line in lines),
        runs=lines,
    )


def run_get(host: EvalHost, run_id: str, scope_id: str) -> EvalSummary:
    """One run, summarised from its stored results.

    :param host: The host whose store holds the run.
    :ptype host: EvalHost
    :param run_id: The run.
    :ptype run_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :return: The summary.
    :rtype: EvalSummary
    :raises NotFoundError: No run with that id in the scope.
    """
    return summarize_run(host, run_id, scope_id)


async def run_launch(host: OpsHost, arguments: LaunchArguments, scope_id: str) -> JobsStarted:
    """Admit a template's launch — one run per model — and start its arms and runs as one job.

    The call makes every refusal the launch makes before it spends and prices every arm against the cap its run is
    held to (:func:`~threetears.evals.run.admit_launch`), reserving the runs' room against the host's admission
    ceiling. The job (:func:`~threetears.evals.run.launch_job_work`) then does what spends: a generating launch's
    case generation, each arm's launcher, the runs started together and waited for to their end. Its record is the
    launch (:class:`~threetears.evals.run.EvalLaunch`), whose poll names its runs once they start.

    :param host: The launching host.
    :ptype host: OpsHost
    :param arguments: What the launch names.
    :ptype arguments: LaunchArguments
    :param scope_id: The scope the template is read in and the runs live in.
    :ptype scope_id: str
    :return: The launch's one job (``launch:<launch id>``); its target is the template.
    :rtype: JobsStarted
    :raises NotFoundError: The template is not in the scope.
    :raises AdmissionRefusedError: The runs would pass the host's admission ceiling.
    :raises ValidationFailedError: Any refusal :func:`~threetears.evals.run.admit_launch` makes. A refusal a kind's
        launcher makes is the job's: it ends ``failed`` saying why.
    """
    admitted = await admit_launch(
        host.launch,
        template_id=arguments.template_id,
        subject_id=arguments.subject_id,
        models=list(arguments.models),
        k_runs=arguments.k_runs,
        n_variations=arguments.n_variations,
        variation_model=arguments.variation_model,
        overlays=arguments.overlays,
        apparatus_settings=arguments.apparatus_settings,
        max_cost_usd=arguments.max_cost_usd,
        judge_model=arguments.judge_model,
        simulator_model=arguments.simulator_model,
        scope_id=scope_id,
        cell_timeout_s=arguments.cell_timeout_s,
        case_set=arguments.case_set,
        measure_latency=arguments.measure_latency,
        arm_schedule=arguments.arm_schedule,
    )
    work = launch_job_work(
        host.launch,
        admitted,
        template_id=arguments.template_id,
        subject_id=arguments.subject_id,
        models=list(arguments.models),
        k_runs=arguments.k_runs,
        n_variations=arguments.n_variations,
        scope_id=scope_id,
        arm_schedule=resolve_arm_schedule(arguments.arm_schedule, measure_latency=arguments.measure_latency),
    )
    label = ", ".join(arguments.models) or "the kind's default model"
    return await start_job(host, AdmittedJob(work=work, kind="launch", target_id=arguments.template_id, label=label))


def run_archive(host: EvalHost, run_id: str, scope_id: str, *, archived: bool) -> RunLine:
    """Archive or restore a run — reversible exclusion from every cohort.

    :param host: The host whose store holds the run.
    :ptype host: EvalHost
    :param run_id: The run.
    :ptype run_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param archived: ``True`` retires it, ``False`` restores it.
    :ptype archived: bool
    :return: The run as persisted.
    :rtype: RunLine
    :raises NotFoundError: No run with that id in the scope.
    """
    return _line(set_run_archived(host.storage, run_id, scope_id, archived=archived, profile=host.profile))


def run_label(host: EvalHost, run_id: str, scope_id: str, *, labels: list[str]) -> RunLine:
    """Set a run's labels, replacing any it has; an empty list clears them (#648).

    A label names a run (a baseline, a production snapshot) so it can be referred to, and listed by
    (``runs_list`` with ``label``), by something other than its id. It is curation, like archiving: no variant key,
    context key or fingerprint is computed over it, so labelling a run moves none of them.

    :param host: The host whose store holds the run.
    :ptype host: EvalHost
    :param run_id: The run.
    :ptype run_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param labels: Its labels: each 1 to 64 characters once stripped, at most 16, stored sorted and each once.
    :ptype labels: list[str]
    :return: The run as persisted.
    :rtype: RunLine
    :raises ValidationFailedError: A label is blank or too long, or there are too many.
    :raises NotFoundError: No run with that id in the scope.
    :raises StorageError: The write did not land.
    """
    return _line(set_run_labels(host.storage, run_id, scope_id, labels=labels))


class ResultRated(EvalBaseModel):
    """A rating written through an action: what it rated, whose it is, and who relayed it when it is a person's."""

    rating_id: str
    result_id: str
    rubric_dim: str
    score: int
    rater: str
    rater_kind: str = Field(
        description=(
            "`agent` through result_rate: the agent rated, whatever account it acts for. `person` through "
            "result_rate_person, which the host attested."
        )
    )
    source: str | None = Field(
        default=None, description="Where the person gave the rating; None on an agent's own rating."
    )
    collected_by: str | None = Field(
        default=None, description="The caller that relayed the person's rating; None on an agent's own rating."
    )
    rated_at: str = Field(description="When the rating was given: the person's own time on a relay that names one.")
    replaced: bool = Field(
        default=False,
        description=(
            "Whether this write replaced the same rater's earlier rating of the same dimension of the same result, "
            "which is gone: one rater holds one rating of one thing."
        ),
    )
    replaced_source: str | None = Field(
        default=None,
        description="Where the replaced rating was given; None when nothing was replaced or it named no source.",
    )
    output_fingerprint: str | None = Field(
        default=None,
        description=(
            "The fingerprint of the judged output rated, copied from the judge score: the rating holds for every "
            "judgement of a byte-identical output on the same criterion. None when the score carries no stamp (a "
            "score made outside a judge phase)."
        ),
    )
    criterion_fingerprint: str | None = Field(
        default=None, description="The fingerprint of the rated dimension's definition, beside output_fingerprint."
    )


def result_rate(
    host: EvalHost, result_id: str, scope_id: str, *, rubric_dim: str, score: int, reason: str, rater: str
) -> ResultRated:
    """Record an agent's rating of one judged dimension of one result — kept beside people's, never pooled with them.

    The operation an agent-facing surface rates through, so ``rater_kind`` is fixed here rather than taken from
    the caller: an agent writing through a tool is an ``agent`` whatever account it acts for, and only a person's
    rating is judge-versus-human agreement (:func:`~threetears.evals.run.rate_result`). A person's rating relayed by
    a caller goes through :func:`result_rate_person`, which the host attests; a host recording a person's rating in
    its own code calls :func:`~threetears.evals.run.rate_result` with ``rater_kind="person"`` itself.

    :param host: The host whose store holds the result.
    :ptype host: EvalHost
    :param result_id: The result rated.
    :ptype result_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param rubric_dim: The judged dimension, spelled as the result's score spells it.
    :ptype rubric_dim: str
    :param score: The score, on the dimension's scale.
    :ptype score: int
    :param reason: The agent's own words for the score.
    :ptype reason: str
    :param rater: Who rated, as the calling surface names the agent.
    :ptype rater: str
    :return: What was written.
    :rtype: ResultRated
    :raises NotFoundError: No such result in the scope.
    :raises ValidationFailedError: The judge scored no such dimension, or the score is off its scale.
    :raises StorageError: The write failed.
    """
    prior = _prior(host, scope_id, result_id, rubric_dim, rater, "agent")
    rating = rate_result(
        host.storage,
        result_id=result_id,
        scope_id=scope_id,
        rubric_dim=rubric_dim,
        rater=rater,
        rater_kind="agent",
        score=score,
        reason=reason,
    )
    return _rated(rating, prior)


#: What ``person`` is, said once for every action that records a person's verdict on a caller's word
#: (``result_rate_person``, ``judge_case_author``): one parameter name, one meaning, across a tool.
PERSON_DESCRIPTION = (
    "Whose verdict it is, as you name them; the host must vouch that you relay it for them, and records it under its "
    "own name for them."
)

#: What a caller relaying a person's rating does when the host vouches for no one.
_PERSON_UNCHECKED_REMEDY = (
    "rate it as yourself with result_rate; a host that relays people's ratings is built with attests_person=<its check>"
)

#: What a caller relaying a person's rating does when the host does not attest it for that person.
_PERSON_REFUSED_REMEDY = (
    "rate it as yourself with result_rate, or relay it as a caller the host attests for that person"
)


def result_rate_person(
    host: EvalHost,
    result_id: str,
    scope_id: str,
    *,
    rubric_dim: str,
    score: int,
    reason: str,
    person: str,
    source: str,
    collected_by: str,
    rated_at: str | None = None,
) -> ResultRated:
    """Record a person's rating of one judged dimension of one result, relayed by a caller the host attests for them.

    A person's verdicts (a blind read, a review session) reach the store through someone else: an agent, or a host's
    import. Recorded as the person's (``rater_kind="person"``), the rating enters the judge's agreement with people,
    follows its output into judge cases and profiles by its label key, and names its ``source`` and the caller that
    relayed it (``collected_by``). So the engine asks the host first (``EvalHost.attests_person``) and records nothing
    unless the host vouches that this caller relays this person's rating: a caller cannot make its own opinion a
    person's by naming one. Checked before the result is read, so a refused caller learns nothing of it. The rating is
    stored under the name the host returns, so two spellings the host knows as one person are one rater.

    One person holds one rating of one dimension of one result, and the one they gave last stands. So the relay
    may carry when the person rated (``rated_at``): dated, it replaces their stored rating (given directly, or
    relayed from another source) when that was given at the same time or earlier, and when the stored rating was
    given later the relay is refused (409, naming that rating's time and source) and nothing is written. Undated, it
    is the person's word now and replaces the stored rating. Either way the result says what it replaced
    (``replaced``, ``replaced_source``), as a log line does.

    A host check that raises is a defect in the host, not a refusal: it propagates, and nothing is written.

    :param host: The host whose store holds the result, and whose check attests the person.
    :ptype host: EvalHost
    :param result_id: The result rated.
    :ptype result_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param rubric_dim: The judged dimension, spelled as the result's score spells it.
    :ptype rubric_dim: str
    :param score: The person's score, on the dimension's scale.
    :ptype score: int
    :param reason: The person's own words for the score.
    :ptype reason: str
    :param person: Who gave the rating, as the host names its people.
    :ptype person: str
    :param source: Where the person gave it (a blind read's id).
    :ptype source: str
    :param collected_by: The calling identity that relays it, as the host resolved it.
    :ptype collected_by: str
    :param rated_at: When the person gave the rating, ISO-8601 with an offset and not in the future; ``None`` for
        now, the relay's own time.
    :ptype rated_at: str | None
    :return: What was written, and what it replaced.
    :rtype: ResultRated
    :raises EvalServiceError: Status 403: the host supplies no ``attests_person``, or it does not attest this caller
        for this person. Nothing is written.
    :raises ConflictError: ``rated_at`` is named, and the person's stored rating of the same thing was given later;
        it stands, and nothing is written.
    :raises TypeError: The host's check returned something other than a non-blank name or ``None``: a defect in the
        host.
    :raises NotFoundError: No such result in the scope.
    :raises ValidationFailedError: The judge scored no such dimension, the score is off its scale, a name is blank, or
        ``rated_at`` is not an ISO-8601 time with an offset at or before now.
    :raises StorageError: The write failed.
    """
    canonical = attested_person(
        host,
        scope_id=scope_id,
        collected_by=collected_by,
        person=person,
        operation="result_rate_person",
        what="rating",
        unchecked_remedy=_PERSON_UNCHECKED_REMEDY,
        refused_remedy=_PERSON_REFUSED_REMEDY,
    )
    prior = _prior(host, scope_id, result_id, rubric_dim, canonical, "person")
    rating = rate_result(
        host.storage,
        result_id=result_id,
        scope_id=scope_id,
        rubric_dim=rubric_dim,
        rater=canonical,
        rater_kind="person",
        score=score,
        reason=reason,
        source=source,
        collected_by=collected_by,
        rated_at=rated_at,
    )
    return _rated(rating, prior)


def attested_person(
    host: EvalHost,
    *,
    scope_id: str,
    collected_by: str,
    person: str,
    operation: str,
    what: str,
    unchecked_remedy: str,
    refused_remedy: str,
) -> str:
    """The person's name as the host knows them, when the host attests ``collected_by`` relaying their ``what``.

    The one trust boundary every operation that records something as a person's, on a caller's word, passes: a
    person's rating (``result_rate_person``), an authored judge case, whose verdict is read as its author's label
    (``judge_case_author``), and a reporter case's reader verdicts, each asked per reader (``reporter_case_freeze``).
    The engine never decides it: the host's check (``EvalHost.attests_person``) does. A refusal is logged as a
    warning, so an operator sees it, and raised to the caller; the operation writes nothing.

    A host check that raises is a defect in the host, not a refusal: it propagates.

    :param host: The host whose check attests the person.
    :ptype host: EvalHost
    :param scope_id: The scope the write would land in, as the host resolved it for the call.
    :ptype scope_id: str
    :param collected_by: The calling identity that relays it, as the host resolved it.
    :ptype collected_by: str
    :param person: Who the caller says it is from, in the caller's spelling.
    :ptype person: str
    :param operation: The operation asking, as its log event names it (``eval.<operation> refused ...``).
    :ptype operation: str
    :param what: What is relayed, as a refusal words it after "no" and "a" (``rating``, ``judge case``,
        ``reader verdict``).
    :ptype what: str
    :param unchecked_remedy: The refusal's remedy when the host supplies no check.
    :ptype unchecked_remedy: str
    :param refused_remedy: The refusal's remedy when the host's check refuses.
    :ptype refused_remedy: str
    :return: The host's own name for the person, non-blank.
    :rtype: str
    :raises EvalServiceError: Status 403: the host supplies no ``attests_person``, or it does not attest this caller
        for this person.
    :raises TypeError: The host's check returned something other than a non-blank name or ``None``: a defect in the
        host.
    """
    attests = host.attests_person
    if attests is None:
        log.warning(
            "eval.%s refused caller=%s person=%s scope=%s: the host supplies no attests_person",
            operation,
            collected_by,
            person,
            scope_id,
        )
        raise EvalServiceError(
            f"host {host.profile.host_id!r} vouches for no person (it supplies no attests_person), so no {what} is "
            "recorded as a person's through an action",
            status_code=403,
            details={"person": person, "collected_by": collected_by},
            remedy=unchecked_remedy,
        )
    canonical = attests(scope_id=scope_id, caller=collected_by, person=person)
    if canonical is not None and (not isinstance(canonical, str) or not canonical.strip()):
        raise TypeError(
            f"host {host.profile.host_id!r}'s attests_person returned {canonical!r}; it returns the person's name as "
            "the host knows them, or None to refuse"
        )
    if canonical is None:
        log.warning(
            "eval.%s refused caller=%s person=%s scope=%s: the host does not attest the caller for the person",
            operation,
            collected_by,
            person,
            scope_id,
        )
        raise EvalServiceError(
            f"the host does not attest {collected_by!r} relaying a {what} for {person!r}, so it is not recorded as "
            "that person's",
            status_code=403,
            details={"person": person, "collected_by": collected_by},
            remedy=refused_remedy,
        )
    # one person, one name: checked stripped above, so returned (and stored) stripped too
    return canonical.strip()


def _prior(
    host: EvalHost, scope_id: str, result_id: str, rubric_dim: str, rater: str, rater_kind: str
) -> CalibrationRating | None:
    """The rating a write by this rater, of this kind, on this dimension of this result would replace, if any.

    Read before the write, not with it: two concurrent writes of one rating can both report what each saw, so
    ``replaced`` says what this write found, and the store keeps the right rating either way.
    """
    return next(
        (
            rating
            for rating in host.storage.query_calibration_ratings(scope_id, result_id=result_id)
            if (rating.rubric_dim, rating.rater, rating.rater_kind) == (rubric_dim, rater, rater_kind)
        ),
        None,
    )


def _rated(rating: CalibrationRating, prior: CalibrationRating | None) -> ResultRated:
    if prior is not None:
        log.info(
            "eval.rate_result replaced rating=%s rater=%s rater_kind=%s replaced_source=%s replaced_rated_at=%s "
            "source=%s",
            rating.id,
            rating.rater,
            rating.rater_kind,
            prior.source,
            prior.rated_at,
            rating.source,
        )
    return ResultRated(
        rating_id=rating.id,
        result_id=rating.result_id,
        rubric_dim=rating.rubric_dim,
        score=rating.score,
        rater=rating.rater,
        rater_kind=rating.rater_kind,
        source=rating.source,
        collected_by=rating.collected_by,
        rated_at=rating.rated_at,
        replaced=prior is not None,
        replaced_source=None if prior is None else prior.source,
        output_fingerprint=rating.output_fingerprint,
        criterion_fingerprint=rating.criterion_fingerprint,
    )


async def judge_repeat_estimate(
    host: OpsHost, run_id: str, scope_id: str, *, result_ids: list[str] | None = None
) -> JudgeRepeatEstimate:
    """What repeating a finished run's judge scores would be priced at, against the host's out-of-run cap — no call
    made.

    The repeat's own collection and admission (:func:`judge_repeat` refuses by the same rule), so
    ``would_start`` is its answer.

    :param host: The host: its store, its judge clients and its out-of-run cap.
    :ptype host: OpsHost
    :param run_id: The finished run.
    :ptype run_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param result_ids: The results to repeat; ``None`` for every result of the run.
    :ptype result_ids: list[str] | None
    :return: The estimate.
    :rtype: JudgeRepeatEstimate
    :raises NotFoundError: No run with that id, or a record it names does not load.
    :raises ValidationFailedError: The run cannot be repeated.
    """
    return await estimate_judge_repeat(
        host.eval_host, run_id, scope_id, out_of_run_cap_usd=host.out_of_run_cap(), result_ids=result_ids
    )


async def judge_repeat(
    host: OpsHost, run_id: str, scope_id: str, *, result_ids: list[str] | None = None
) -> JobsStarted:
    """Price and admit repeating a finished run's judge scores, then start it as a job — the ``separation`` tier's read.

    Every call is priced and admitted against the host's out-of-run cap here, and refused here with nothing spent;
    the job makes them, each ledgered under purpose ``judge`` (:func:`~threetears.evals.run.judge_repeat_work`), so
    the call returns at once. Poll the job: its result, once completed, is the
    :class:`~threetears.evals.run.JudgeRepeatReport`.

    :param host: The host: its store, its judge clients and its out-of-run cap.
    :ptype host: OpsHost
    :param run_id: The finished run.
    :ptype run_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param result_ids: The results to repeat; ``None`` for every result of the run.
    :ptype result_ids: list[str] | None
    :return: The repeat's one job (``action:<id>``, its record an :class:`~threetears.evals.run.EvalActionJob` under
        ``judge_repeat``); its label names the run.
    :rtype: JobsStarted
    :raises NotFoundError: No run with that id, or a record it names does not load.
    :raises ValidationFailedError: The run cannot be repeated, or its calls are priced above the cap or cannot be
        priced under it — before any call.
    """
    work = await judge_repeat_work(
        host.eval_host, run_id, scope_id, out_of_run_cap_usd=host.out_of_run_cap(), result_ids=result_ids
    )
    return await start_job(
        host, AdmittedJob(work=work, kind="action", target_id=JUDGE_REPEAT_ACTION, label=f"run {run_id}")
    )


async def judge_temperature_estimate(
    host: OpsHost,
    run_id: str,
    scope_id: str,
    *,
    selection: TemperatureSelection = "borderline",
    repeats: int = DEFAULT_TEMPERATURE_REPEATS,
    result_ids: list[str] | None = None,
) -> JudgeTemperatureEstimate:
    """What comparing a finished run's judge at the pinned temperature and the provider default would cost — no call.

    The comparison's own collection, selection and admission (:func:`judge_temperature` refuses by the same rule), so
    ``would_start`` is its answer.

    :param host: The host: its store, its judge clients and its out-of-run cap.
    :ptype host: OpsHost
    :param run_id: The finished run.
    :ptype run_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param selection: ``borderline`` (the default) or ``all`` scored dims.
    :ptype selection: TemperatureSelection
    :param repeats: Calls per dim at each temperature.
    :ptype repeats: int
    :param result_ids: The results to draw from; ``None`` for every result of the run.
    :ptype result_ids: list[str] | None
    :return: The estimate.
    :rtype: JudgeTemperatureEstimate
    :raises NotFoundError: No run with that id, or a record it names does not load.
    :raises ValidationFailedError: The comparison cannot be made.
    """
    return await estimate_judge_temperature_comparison(
        host.eval_host,
        run_id,
        scope_id,
        out_of_run_cap_usd=host.out_of_run_cap(),
        selection=selection,
        repeats=repeats,
        result_ids=result_ids,
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
    """Admit re-judging a finished run's borderline cases at the pinned temperature and the provider default (#633), as
    a job.

    The measurement the judge temperature policy rests on: per dimension, score variance across repeats and the
    judge's self-agreement at each setting, side by side. Every call is priced and admitted against the host's
    out-of-run cap in the call, which refuses with nothing spent; the job
    (:func:`~threetears.evals.run.judge_temperature_work`) makes the calls, ledgered under purpose ``judge``, and
    writes every answer on its record (:class:`~threetears.evals.run.EvalJudgeTemperatureCheck`). Nothing is written
    to the results. Polling the job returns the comparison
    (:func:`~threetears.evals.analysis.read_judge_temperatures` over the record's answers) once it has completed.

    :param host: The host: its store, its judge clients, its out-of-run cap and its job manager.
    :ptype host: OpsHost
    :param run_id: The finished run.
    :ptype run_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param selection: ``borderline`` (the default) or ``all`` scored dims.
    :ptype selection: TemperatureSelection
    :param repeats: Calls per dim at each temperature.
    :ptype repeats: int
    :param result_ids: The results to draw from; ``None`` for every result of the run.
    :ptype result_ids: list[str] | None
    :return: The comparison's one job (``judge-temperature:<id>``); its target is the run.
    :rtype: JobsStarted
    :raises NotFoundError: No run with that id, or a record it names does not load.
    :raises ValidationFailedError: The comparison cannot be made, or its calls are priced above the cap or cannot be
        priced under it — before any call.
    """
    work = await judge_temperature_work(
        host.eval_host,
        run_id,
        scope_id,
        out_of_run_cap_usd=host.out_of_run_cap(),
        selection=selection,
        repeats=repeats,
        result_ids=result_ids,
    )
    return await start_job(
        host, AdmittedJob(work=work, kind="judge_temperature", target_id=run_id, label=f"{selection} x{repeats}")
    )


async def judge_second_estimate(
    host: OpsHost,
    run_id: str,
    scope_id: str,
    *,
    judge: SecondJudge,
    sample_fraction: float = 1.0,
    seed: int = DEFAULT_SECOND_JUDGE_SEED,
    result_ids: list[str] | None = None,
) -> SecondJudgeEstimate:
    """What asking a second judge about a finished run would be priced at, against the host's out-of-run cap — no call.

    The pass's own collection, sample and admission (:func:`judge_second` refuses by the same rule), so
    ``would_start`` is its answer.

    :param host: The host: its store, its judge clients and its out-of-run cap.
    :ptype host: OpsHost
    :param run_id: The finished run.
    :ptype run_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param judge: The second judge.
    :ptype judge: SecondJudge
    :param sample_fraction: The share of the run's judgeable results to ask about, in ``(0, 1]``.
    :ptype sample_fraction: float
    :param seed: The seed the share is drawn with.
    :ptype seed: int
    :param result_ids: The results to draw from; ``None`` for every result of the run.
    :ptype result_ids: list[str] | None
    :return: The estimate.
    :rtype: SecondJudgeEstimate
    :raises NotFoundError: No run with that id, or a record it names does not load.
    :raises ValidationFailedError: The run cannot be asked about.
    """
    return await estimate_second_judge(
        host.eval_host,
        run_id,
        scope_id,
        judge=judge,
        out_of_run_cap_usd=host.out_of_run_cap(),
        sample_fraction=sample_fraction,
        seed=seed,
        result_ids=result_ids,
    )


async def judge_second(
    host: OpsHost,
    run_id: str,
    scope_id: str,
    *,
    judge: SecondJudge,
    sample_fraction: float = 1.0,
    seed: int = DEFAULT_SECOND_JUDGE_SEED,
    result_ids: list[str] | None = None,
    asked_by: SecondJudgeAskedBy = "judge_second",
) -> JobsStarted:
    """Admit asking a second judge to score a seeded share of a finished run's judged results, as a job.

    Every call is priced and admitted against the host's out-of-run cap in the call, which refuses with nothing
    spent; the job (:func:`~threetears.evals.run.second_judge_work`) makes the calls, ledgered under purpose
    ``second_judge`` — measurement cost on its own line, never the candidate's — records each second score beside
    the first it pairs with, and writes the pass's record (:class:`~threetears.evals.run.EvalSecondJudgePass`). The
    scores the run was judged with are never changed. Polling the job returns the read — the pass's report, and the
    agreement and drift its pairs read (:class:`~threetears.evals.ops.SecondJudgeRead`) — once it has completed.

    :param host: The host: its store, its judge clients, its out-of-run cap and its job manager.
    :ptype host: OpsHost
    :param run_id: The finished run.
    :ptype run_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param judge: The second judge: its model, its prompts, its temperature.
    :ptype judge: SecondJudge
    :param sample_fraction: The share of the run's judgeable results to ask about, in ``(0, 1]``.
    :ptype sample_fraction: float
    :param seed: The seed the share is drawn with.
    :ptype seed: int
    :param result_ids: The results to draw from; ``None`` for every result of the run.
    :ptype result_ids: list[str] | None
    :param asked_by: The operation asking, recorded on the pass.
    :ptype asked_by: SecondJudgeAskedBy
    :return: The pass's one job (``second-judge:<pass id>``); its target is the run.
    :rtype: JobsStarted
    :raises NotFoundError: No run with that id, or a record it names does not load.
    :raises ValidationFailedError: The run cannot be asked about, or its calls are priced above the cap or cannot be
        priced under it — before any call.
    """
    work = await second_judge_work(
        host.eval_host,
        run_id,
        scope_id,
        judge=judge,
        out_of_run_cap_usd=host.out_of_run_cap(),
        asked_by=asked_by,
        sample_fraction=sample_fraction,
        seed=seed,
        result_ids=result_ids,
    )
    return await start_job(host, AdmittedJob(work=work, kind="second_judge", target_id=run_id, label=judge.model))


async def judge_drift_check(
    host: OpsHost, run_id: str, scope_id: str, *, judge: SecondJudge, result_ids: list[str] | None = None
) -> JobsStarted:
    """Admit re-scoring every judged result of a finished run under a changed judge, as a job; poll it for the drift.

    :func:`judge_second` over the whole set (``sample_fraction`` 1): the run's judge is the configuration it was
    scored under, as recorded; ``judge`` is the new one. The stored scores are never changed. The drift detects
    movement between the two judges, never which one is right.

    :param host: The host: its store, its judge clients, its out-of-run cap and its job manager.
    :ptype host: OpsHost
    :param run_id: The finished run whose stored evidence is re-scored.
    :ptype run_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param judge: The changed judge.
    :ptype judge: SecondJudge
    :param result_ids: The frozen set to re-score; ``None`` for every result of the run.
    :ptype result_ids: list[str] | None
    :return: The pass's one job, as :func:`judge_second` returns it.
    :rtype: JobsStarted
    :raises NotFoundError: No run with that id, or a record it names does not load.
    :raises ValidationFailedError: As :func:`judge_second`.
    """
    return await judge_second(
        host,
        run_id,
        scope_id,
        judge=judge,
        sample_fraction=1.0,
        result_ids=result_ids,
        asked_by="judge_drift_check",
    )


def run_delete(host: EvalHost, run_id: str, scope_id: str, *, confirm: str | None) -> RunDeleted:
    """Destroy a run, its results and its campaign memberships — unrecoverable; archive is the safe answer.

    A run this build's model cannot read is destroyed from its stored row
    (:func:`~threetears.evals.run.curation.delete_unreadable_run`), removing exactly what a readable run's delete
    removes: the way out of a corpus the model has outlived, which ``runs_unreadable`` lists. Its results are found
    by the run id they carry, never read, and it is refused only while a process may still hold it.

    :param host: The host whose store holds the run.
    :ptype host: EvalHost
    :param run_id: The run.
    :ptype run_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param confirm: Must echo ``run_id``.
    :ptype confirm: str | None
    :return: What was removed.
    :rtype: RunDeleted
    :raises NotFoundError: No run with that id in the scope — refused before ``confirm`` is read.
    :raises ValidationFailedError: ``confirm`` does not echo the id, or the run is still running (one this build
        cannot read: stored running under a lease that has not lapsed).
    :raises StorageError: The cascade stopped partway; the message says how far it got.
    """
    found = host.storage.scan_eval_run(run_id, scope_id)
    if found.unreadable:
        # The way out of a run this build cannot read: every read of it refuses, so it is deleted from its row.
        (row,) = found.unreadable
        removed = delete_unreadable_run(host.storage, run_id, row, scope_id, confirm=confirm)
    else:
        removed = delete_run(host.storage, get_run(host.storage, run_id, scope_id), scope_id, confirm=confirm)
    return RunDeleted(
        run_id=removed["run_id"],
        results_deleted=removed["results_deleted"],
        campaigns_detached=list(removed["campaigns_detached"]),
    )


__all__ = [
    "CaseSetLine",
    "CaseSetListing",
    "CaseSetMint",
    "JudgeConfigLine",
    "JudgeConfigListing",
    "LaunchArguments",
    "ResultRated",
    "RunDeleted",
    "RunLine",
    "RunListing",
    "TemplateLine",
    "TemplateListing",
    "UnreadableRunLine",
    "UnreadableRunsListing",
    "run_archive",
    "run_label",
    "run_delete",
    "run_get",
    "result_rate",
    "result_rate_person",
    "run_launch",
    "runs_list",
    "runs_unreadable",
    "templates_list",
    "case_set_mint",
    "case_sets_list",
    "judge_configs_list",
]
