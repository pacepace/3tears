"""The run side's operations: what a scope holds, one run's summary, a launch, and a run's curation.

Each takes the host and the scope and returns a typed value — never a ``dict`` a surface re-types — so
a CLI, an MCP action and a REST route that call one read one shape. A launch is long work and follows
the job contract (:mod:`threetears.evals.ops.jobs`): it returns one job per arm, and each job's record is
its run.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field, model_validator

from threetears.evals.analysis.agreement import InterJudgeAgreement, inter_judge_agreement
from threetears.evals.analysis.judge_drift import JudgeDrift, judge_drift
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.kernel.host import EvalHost
from threetears.evals.schema.models import DEFAULT_LAUNCH_K_RUNS, CaseSet, CaseSetRef, EvalRun, SecondJudge
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.jobs import JobHandle, JobsStarted, run_job_id
from threetears.evals.analysis.summary import EvalSummary, summarize_run
from threetears.evals.run.authoring import list_templates
from threetears.evals.run.case_sets import mint_case_set
from threetears.evals.run.curation import delete_run, set_run_archived
from threetears.evals.run.launch import start_run
from threetears.evals.run.judge_repeat import (
    JudgeRepeatEstimate,
    JudgeRepeatReport,
    estimate_judge_repeat,
    repeat_judge_scores,
)
from threetears.evals.run.judge_second import (
    DEFAULT_SECOND_JUDGE_SEED,
    SecondJudgeEstimate,
    SecondJudgeReport,
    ask_second_judge,
    estimate_second_judge,
)
from threetears.evals.analysis.judge_temperature import JudgeTemperatureComparison, read_judge_temperatures
from threetears.evals.kernel.judge_temperature import DEFAULT_TEMPERATURE_REPEATS, TemperatureSelection
from threetears.evals.run.judge_temperature import (
    JudgeTemperatureEstimate,
    estimate_judge_temperature_comparison,
    judge_at_two_temperatures,
)
from threetears.evals.run.lifecycle import get_run
from threetears.evals.run.ratings import rate_result
from threetears.evals.run.reads import list_runs


class TemplateLine(EvalBaseModel):
    """One template, as a listing shows it."""

    id: str
    name: str
    candidate_kind: str
    intent: str
    archived: bool


class TemplateListing(EvalBaseModel):
    """A scope's templates."""

    templates: list[TemplateLine]


class RunLine(EvalBaseModel):
    """One run, as a listing shows it."""

    id: str
    status: str
    candidate_model: str
    template_id: str | None
    created_at: str
    archived: bool


class RunListing(EvalBaseModel):
    """A scope's runs, newest first as the store lists them."""

    runs: list[RunLine]
    include_archived: bool = Field(description="Whether archived runs were listed; they are left out by default.")


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

    Args:
        host: The host whose store holds the set.
        arguments: The set, its template and its cases.
        scope_id: The scope the set lives in.

    Returns:
        The stored version.

    Raises:
        ValidationFailedError: Any refusal :func:`~threetears.evals.run.case_sets.mint_case_set` makes.
        ConflictError: Another writer stored this version first.
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

    Args:
        host: The host whose store is read.
        scope_id: The scope.
        name: Only this set's versions.

    Returns:
        The listing.
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
    )


def templates_list(host: EvalHost, scope_id: str, *, archived: bool = False) -> TemplateListing:
    """The scope's templates.

    Args:
        host: The host whose store is read.
        scope_id: The scope.
        archived: List the archived templates instead of the active ones.

    Returns:
        The listing.
    """
    return TemplateListing(
        templates=[
            TemplateLine(
                id=template.id,
                name=template.name,
                candidate_kind=template.candidate_kind,
                intent=template.intent,
                archived=template.archived,
            )
            for template in list_templates(host.storage, scope_id, archived=archived)
        ]
    )


def runs_list(
    host: EvalHost, scope_id: str, *, status: str | None = None, include_archived: bool = False
) -> RunListing:
    """The scope's runs.

    Args:
        host: The host whose store is read.
        scope_id: The scope.
        status: Only runs with this stored status.
        include_archived: List archived runs too.

    Returns:
        The listing.

    Raises:
        ValidationFailedError: ``status`` is not a run status.
    """
    runs = list_runs(host, scope_id, status=status, include_archived=include_archived)
    return RunListing(runs=[_line(run) for run in runs], include_archived=include_archived)


def run_get(host: EvalHost, run_id: str, scope_id: str) -> EvalSummary:
    """One run, summarised from its stored results.

    Args:
        host: The host whose store holds the run.
        run_id: The run.
        scope_id: The scope it lives in.

    Returns:
        The summary.

    Raises:
        NotFoundError: No run with that id in the scope.
    """
    return summarize_run(host, run_id, scope_id)


async def run_launch(host: OpsHost, arguments: LaunchArguments, scope_id: str) -> JobsStarted:
    """Launch a template's runs, one per model, and return a job per run.

    Args:
        host: The launching host.
        arguments: What the launch names.
        scope_id: The scope the template is read in and the runs live in.

    Returns:
        One job per launched run, in the order the models were named.

    Raises:
        NotFoundError: The template is not in the scope.
        AdmissionRefusedError: The runs would pass the host's admission ceiling.
        ValidationFailedError: Any refusal :func:`~threetears.evals.run.start_run` makes.
    """
    runs = await start_run(
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
    )
    return JobsStarted(
        jobs=[
            JobHandle(job_id=run_job_id(run.id), kind="run", target_id=run.id, label=run.candidate_model)
            for run in runs
        ]
    )


def run_archive(host: EvalHost, run_id: str, scope_id: str, *, archived: bool) -> RunLine:
    """Archive or restore a run — reversible exclusion from every cohort.

    Args:
        host: The host whose store holds the run.
        run_id: The run.
        scope_id: The scope it lives in.
        archived: ``True`` retires it, ``False`` restores it.

    Returns:
        The run as persisted.

    Raises:
        NotFoundError: No run with that id in the scope.
    """
    return _line(set_run_archived(host.storage, run_id, scope_id, archived=archived, profile=host.profile))


class ResultRated(EvalBaseModel):
    """A rating an agent wrote: what it rated, and that it is an agent's, never read as a person's."""

    rating_id: str
    result_id: str
    rubric_dim: str
    score: int
    rater: str
    rater_kind: str = Field(
        description="Always `agent` through an action: the agent rated, whatever account it acts for."
    )


def result_rate(
    host: EvalHost, result_id: str, scope_id: str, *, rubric_dim: str, score: int, reason: str, rater: str
) -> ResultRated:
    """Record an agent's rating of one judged dimension of one result — kept beside people's, never pooled with them.

    The operation an agent-facing surface rates through, so ``rater_kind`` is fixed here rather than taken from
    the caller: an agent writing through a tool is an ``agent`` whatever account it acts for, and only a person's
    rating is judge-versus-human agreement (:func:`~threetears.evals.run.rate_result`). A host recording a
    person's rating calls :func:`~threetears.evals.run.rate_result` with ``rater_kind="person"`` itself.

    Args:
        host: The host whose store holds the result.
        result_id: The result rated.
        scope_id: The scope it lives in.
        rubric_dim: The judged dimension, spelled as the result's score spells it.
        score: The score, on the dimension's scale.
        reason: The agent's own words for the score.
        rater: Who rated, as the calling surface names the agent.

    Returns:
        What was written.

    Raises:
        NotFoundError: No such result in the scope.
        ValidationFailedError: The judge scored no such dimension, or the score is off its scale.
        StorageError: The write failed.
    """
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
    return ResultRated(
        rating_id=rating.id,
        result_id=rating.result_id,
        rubric_dim=rating.rubric_dim,
        score=rating.score,
        rater=rating.rater,
        rater_kind=rating.rater_kind,
    )


async def judge_repeat_estimate(
    host: OpsHost, run_id: str, scope_id: str, *, result_ids: list[str] | None = None
) -> JudgeRepeatEstimate:
    """What repeating a finished run's judge scores would be priced at, against the host's out-of-run cap — no call made.

    The repeat's own collection and admission (:func:`judge_repeat` refuses by the same rule), so
    ``would_start`` is its answer.

    Args:
        host: The host: its store, its judge clients and its out-of-run cap.
        run_id: The finished run.
        scope_id: The scope it lives in.
        result_ids: The results to repeat; ``None`` for every result of the run.

    Returns:
        The estimate.

    Raises:
        NotFoundError: No run with that id, or a record it names does not load.
        ValidationFailedError: The run cannot be repeated.
    """
    return await estimate_judge_repeat(
        host.eval_host, run_id, scope_id, out_of_run_cap_usd=host.out_of_run_cap(), result_ids=result_ids
    )


async def judge_repeat(
    host: OpsHost, run_id: str, scope_id: str, *, result_ids: list[str] | None = None
) -> JudgeRepeatReport:
    """Repeat a finished run's judge scores — the measurement the ``separation`` evidence tier reads.

    Every call is priced and admitted against the host's out-of-run cap before the first is sent, and
    ledgered under purpose ``judge`` (:func:`~threetears.evals.run.repeat_judge_scores`).

    Args:
        host: The host: its store, its judge clients and its out-of-run cap.
        run_id: The finished run.
        scope_id: The scope it lives in.
        result_ids: The results to repeat; ``None`` for every result of the run.

    Returns:
        What was repeated, what it cost, and what was left out.

    Raises:
        NotFoundError: No run with that id, or a record it names does not load.
        ValidationFailedError: The run cannot be repeated, or its calls are priced above the cap or cannot be
            priced under it — before any call.
    """
    return await repeat_judge_scores(
        host.eval_host, run_id, scope_id, out_of_run_cap_usd=host.out_of_run_cap(), result_ids=result_ids
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

    Args:
        host: The host: its store, its judge clients and its out-of-run cap.
        run_id: The finished run.
        scope_id: The scope it lives in.
        selection: ``borderline`` (the default) or ``all`` scored dims.
        repeats: Calls per dim at each temperature.
        result_ids: The results to draw from; ``None`` for every result of the run.

    Returns:
        The estimate.

    Raises:
        NotFoundError: No run with that id, or a record it names does not load.
        ValidationFailedError: The comparison cannot be made.
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
) -> JudgeTemperatureComparison:
    """Re-judge a finished run's borderline cases at the pinned temperature and at the provider default (#633).

    The measurement the judge temperature policy rests on: per dimension, score variance across repeats and the
    judge's self-agreement at each setting, side by side: :func:`~threetears.evals.run.judge_at_two_temperatures` asks,
    :func:`~threetears.evals.analysis.read_judge_temperatures` reads.
    Every call is priced and admitted against the host's out-of-run cap before the first is sent, and ledgered under
    purpose ``judge``. Nothing is written to the results.

    Args:
        host: The host: its store, its judge clients and its out-of-run cap.
        run_id: The finished run.
        scope_id: The scope it lives in.
        selection: ``borderline`` (the default) or ``all`` scored dims.
        repeats: Calls per dim at each temperature.
        result_ids: The results to draw from; ``None`` for every result of the run.

    Returns:
        The comparison.

    Raises:
        NotFoundError: No run with that id, or a record it names does not load.
        ValidationFailedError: The comparison cannot be made, or its calls are priced above the cap or cannot be
            priced under it — before any call.
    """
    answers = await judge_at_two_temperatures(
        host.eval_host,
        run_id,
        scope_id,
        out_of_run_cap_usd=host.out_of_run_cap(),
        selection=selection,
        repeats=repeats,
        result_ids=result_ids,
    )
    return read_judge_temperatures(answers)


class SecondJudgeRead(EvalBaseModel):
    """A second judge's pass over a run, and what its pairs say: agreement between the judges, and drift.

    Both readings are over this pass's pairs only, read off the stored records by the code every other surface reads
    them with (:func:`~threetears.evals.analysis.inter_judge_agreement`, :func:`~threetears.evals.analysis.judge_drift`).
    """

    report: SecondJudgeReport = Field(description="What was asked, what was left out, and what it cost.")
    agreement: InterJudgeAgreement = Field(
        description="Per-dimension agreement between the run's judge and the second."
    )
    drift: JudgeDrift = Field(description="Per-dimension movement from the run's judge to the second, with intervals.")


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

    Args:
        host: The host: its store, its judge clients and its out-of-run cap.
        run_id: The finished run.
        scope_id: The scope it lives in.
        judge: The second judge.
        sample_fraction: The share of the run's judgeable results to ask about, in ``(0, 1]``.
        seed: The seed the share is drawn with.
        result_ids: The results to draw from; ``None`` for every result of the run.

    Returns:
        The estimate.

    Raises:
        NotFoundError: No run with that id, or a record it names does not load.
        ValidationFailedError: The run cannot be asked about.
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
) -> SecondJudgeRead:
    """Ask a second judge to score a seeded share of a finished run's judged results, and read agreement and drift.

    Every call is priced and admitted against the host's out-of-run cap before the first is sent, and ledgered under
    purpose ``second_judge`` — measurement cost on its own line, never the candidate's
    (:func:`~threetears.evals.run.ask_second_judge`). The scores the run was judged with are never changed.

    Args:
        host: The host: its store, its judge clients and its out-of-run cap.
        run_id: The finished run.
        scope_id: The scope it lives in.
        judge: The second judge: its model, its prompts, its temperature.
        sample_fraction: The share of the run's judgeable results to ask about, in ``(0, 1]``.
        seed: The seed the share is drawn with.
        result_ids: The results to draw from; ``None`` for every result of the run.

    Returns:
        The pass, and the agreement and drift its pairs read.

    Raises:
        NotFoundError: No run with that id, or a record it names does not load.
        ValidationFailedError: The run cannot be asked about, or its calls are priced above the cap or cannot be
            priced under it — before any call.
    """
    eval_host = host.eval_host
    report = await ask_second_judge(
        eval_host,
        run_id,
        scope_id,
        judge=judge,
        out_of_run_cap_usd=host.out_of_run_cap(),
        sample_fraction=sample_fraction,
        seed=seed,
        result_ids=result_ids,
    )
    results = await run_blocking(
        eval_host.blocking_executor, eval_host.storage.query_eval_results_by_run, run_id, scope_id
    )
    return SecondJudgeRead(
        report=report,
        agreement=inter_judge_agreement(results, pass_id=report.pass_id),
        drift=judge_drift(results, pass_id=report.pass_id),
    )


async def judge_drift_check(
    host: OpsHost, run_id: str, scope_id: str, *, judge: SecondJudge, result_ids: list[str] | None = None
) -> SecondJudgeRead:
    """Re-score every judged result of a finished run under a changed judge, and read how far each dimension moved.

    :func:`judge_second` over the whole set (``sample_fraction`` 1): the run's judge is the configuration it was
    scored under, as recorded; ``judge`` is the new one. The stored scores are never changed. The drift detects
    movement between the two judges, never which one is right.

    Args:
        host: The host: its store, its judge clients and its out-of-run cap.
        run_id: The finished run whose stored evidence is re-scored.
        scope_id: The scope it lives in.
        judge: The changed judge.
        result_ids: The frozen set to re-score; ``None`` for every result of the run.

    Returns:
        The pass, and the drift (and agreement) its pairs read.

    Raises:
        NotFoundError: No run with that id, or a record it names does not load.
        ValidationFailedError: As :func:`judge_second`.
    """
    return await judge_second(host, run_id, scope_id, judge=judge, sample_fraction=1.0, result_ids=result_ids)


def run_delete(host: EvalHost, run_id: str, scope_id: str, *, confirm: str | None) -> RunDeleted:
    """Destroy a run, its results and its campaign memberships — unrecoverable; archive is the safe answer.

    Args:
        host: The host whose store holds the run.
        run_id: The run.
        scope_id: The scope it lives in.
        confirm: Must echo ``run_id``.

    Returns:
        What was removed.

    Raises:
        NotFoundError: No run with that id in the scope — refused before ``confirm`` is read.
        ValidationFailedError: ``confirm`` does not echo the id, or the run is still running.
        StorageError: The cascade stopped partway; the message says how far it got.
    """
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
    "LaunchArguments",
    "ResultRated",
    "RunDeleted",
    "RunLine",
    "RunListing",
    "TemplateLine",
    "TemplateListing",
    "run_archive",
    "run_delete",
    "run_get",
    "result_rate",
    "run_launch",
    "runs_list",
    "templates_list",
    "case_set_mint",
    "case_sets_list",
]
