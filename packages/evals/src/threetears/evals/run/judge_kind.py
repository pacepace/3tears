"""The judge kind: a judge configuration evaluated as a campaign's subject, over outputs frozen from stored results (#628).

What a judge case is, what a trial records and the kind's contract are declared in the kernel
(:mod:`threetears.evals.kernel.judge_cases`). This module is the run side of it:

- :class:`JudgeKind` — the :class:`~threetears.evals.kernel.candidate_kind.CandidateKind`. One instance is one
  judge: a model (the run's candidate model), the prompt per criterion (versioned
  :class:`~threetears.evals.schema.models.JudgeConfig` records, or the built-in prompt) and, optionally, a
  temperature every call is requested at. A trial rebuilds the judge context the case froze and sends the
  criterion's call through the engine's own :class:`~threetears.evals.run.JudgeService` — the request builders,
  :func:`~threetears.evals.run.run_judge_llm`'s parse and retry, the cost record — so the judge under test is asked
  exactly as a run's judge is asked, and nothing but the judge is called. Its grade is code: the trial lands parse
  validity and agreement with the case's labels on ``host_measures``, and the whole answer on ``kind_payload``,
  where :func:`~threetears.evals.analysis.judge_kind_readings` pools trials per judge and criterion.
- :func:`freeze_judge_cases` — freezes (output, criterion, labels) cases from a host's stored judged results into a
  judge template, and mints a case set over them. What each case freezes is what the result's judge read, rebuilt
  under the one check a re-judge and a second judge rebuild it under
  (:func:`~threetears.evals.run.rejudge.reproducible_judge_inputs`).
- :func:`judge_case_labels` — the labels a frozen case carries. **The seam output-bound labels plug into** (#628):
  today it reads the person ratings keyed to the result.
- :func:`launchable_judge_kind` — the kind's launch-registry entry, for a host that launches judge campaigns
  through :func:`~threetears.evals.run.start_run`.

**Spend.** Every call a trial makes is recorded on the trial's result as a ``judge``-role usage row and nothing
else, so a judge campaign's metered spend has no ``candidate`` role at all. It is in-run spend — the run's cost
cap counts it as it counts any cell's — and is deliberately NOT written to the out-of-run ledger the way a judge
repeat's or a second judge's is: those calls score a finished run from outside it, and these are the run.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from threetears.evals.kernel.candidate_kind import (
    CandidateOutput,
    CandidatePreparationFailed,
    CandidateTelemetry,
    CellSink,
    CellSpanWindow,
    VariantConfig,
)
from threetears.evals.kernel.errors import NotFoundError, ValidationFailedError
from threetears.evals.kernel.judge_cases import (
    JUDGE_CASE_KEY,
    JUDGE_KIND,
    JUDGE_KIND_ROLE,
    JUDGE_LABEL_AGREEMENT_MEASURE,
    JUDGE_PARSE_VALID_MEASURE,
    JudgeCase,
    JudgeCaseLabel,
    JudgeCaseSource,
    JudgeKindOverlays,
    JudgeTrial,
    JudgeTrialOutcome,
    judge_case_of,
)
from threetears.evals.kernel.usage_capture import CallUsage, RoleUsageLedger
from threetears.evals.run.case_sets import mint_case_set
from threetears.evals.run.judge import sent_temperature
from threetears.evals.run.judge_service import JudgeContext, JudgeRequest, JudgeService
from threetears.evals.run.launch import ArmPlan, KindWiring, LaunchableKind, launch_run, require_candidate_model
from threetears.evals.run.rejudge import reproducible_judge_inputs
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.models import (
    OUTCOME_DIM_ID,
    TRANSCRIPT_DIM_ID,
    CaseSetRef,
    EvalTestCase,
    JudgedArtifact,
)
from threetears.evals.schema.subject import SubjectSnapshot
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.evals.kernel.cassettes import CellCassettes
    from threetears.evals.kernel.host.eval_host import CompletionClients
    from threetears.evals.kernel.storage import EvalStorage
    from threetears.evals.run.judge_service import JudgeOutcome
    from threetears.evals.run.launch import LaunchHost, LaunchRequest
    from threetears.evals.schema.completion import ProviderFailureDescriber
    from threetears.evals.schema.models import CalibrationRating, EvalResult, EvalRun, EvalTemplate, JudgeConfig

log = get_logger(__name__)

__all__ = [
    "FrozenJudgeCase",
    "JudgeCaseFreezeReport",
    "JudgeCaseSkip",
    "JudgeKind",
    "PreparedJudge",
    "freeze_judge_cases",
    "judge_case_labels",
    "judge_kind",
    "launchable_judge_kind",
]

#: The namespace a frozen judge case's id is derived in, so freezing the same output, criterion and labels into
#: the same template twice names one case rather than minting a second.
_CASE_ID_NAMESPACE = uuid.UUID("4557ddbf-f5d3-4ed0-9459-1b883d5f1f70")


# =============================================================================
# Labels — the seam output-bound labels plug into
# =============================================================================


def judge_case_labels(
    storage: EvalStorage, scope_id: str, result: EvalResult, dim: str, *, scale: str
) -> list[CalibrationRating]:
    """The person ratings a judge case frozen from ``result`` on ``dim`` carries as its labels.

    **SEAM (#628, output-bound labels).** Today a label is found by the RESULT it was written on: the person
    ratings keyed to ``result.id`` on ``dim``. Once a rating can be keyed by a fingerprint of the output a person
    read plus the criterion, this is the one function that changes — it looks the label up by that fingerprint, so
    a label given on any result whose judged output is byte-identical follows the output into its case. Nothing
    else in the freeze reads ratings.

    Only a person's rating is a label: an agent's is another model's opinion, as agreement with people reads it
    (:func:`~threetears.evals.analysis.judge_agreement`). A rating on another scale than the criterion is asked on
    is left out, since it answers a different question.

    Args:
        storage: The store the ratings live in.
        scope_id: The scope the result lives in.
        result: The judged result the case replays.
        dim: The criterion.
        scale: The scale the case asks the criterion on.

    Returns:
        The labels, oldest first.
    """
    return [
        rating
        for rating in storage.query_calibration_ratings(scope_id, result_id=result.id)
        if rating.rubric_dim == dim and rating.rater_kind == "person" and rating.scale == scale
    ]


# =============================================================================
# Freezing cases from stored results
# =============================================================================


class FrozenJudgeCase(EvalBaseModel):
    """One case a freeze produced: the test case, the output and criterion it replays, and how many labels it has."""

    test_case_id: str
    source_run_id: str
    source_result_id: str
    dim: str
    labels: int
    created: bool = False
    """Whether this freeze wrote the case; ``False`` when an identical case was already stored."""


class JudgeCaseSkip(EvalBaseModel):
    """A stored result the freeze could not make a case of, and why."""

    result_id: str
    reason: str


class JudgeCaseFreezeReport(EvalBaseModel):
    """What a freeze froze, the case set listing it, and what it could not freeze."""

    template_id: str
    cases: list[FrozenJudgeCase]
    case_set: CaseSetRef | None = None
    """The case set version listing exactly these cases, in order; ``None`` when no set was asked for."""
    skipped: list[JudgeCaseSkip] = []


def _judge_template(template: EvalTemplate) -> EvalTemplate:
    """The template, refused unless it is of the judge kind — the only kind that reads a judge case."""
    if template.candidate_kind != JUDGE_KIND:
        raise ValidationFailedError(
            f"template {template.id!r} is of kind {template.candidate_kind!r}, not {JUDGE_KIND!r}; a judge case is "
            f"frozen into a template of the {JUDGE_KIND!r} kind, whose launch is what runs it"
        )
    return template


def _cases_of_result(
    storage: EvalStorage,
    template: EvalTemplate,
    run: EvalRun,
    result: EvalResult,
    scope_id: str,
    dims: Collection[str] | None,
) -> list[tuple[str, JudgeCase]]:
    """Every case one stored result freezes into: one per judged dim, refusing what its run cannot reproduce.

    Raises:
        ValidationFailedError: The result's judge input cannot be rebuilt from what its run recorded.
        NotFoundError: A record the run names does not load.
    """
    # The configs are the arm's, not the source's, so none needs to load; everything the judge READ must.
    inputs = reproducible_judge_inputs(storage, result, run, scope_id, request_settings="as_recorded", config_dims=())
    criteria = {dim.name: dim for dim in inputs.template.rubric}
    frozen: list[tuple[str, JudgeCase]] = []
    for dim in inputs.dims:
        if dims is not None and dim not in dims:
            continue
        criterion = criteria.get(dim)
        if criterion is None and dim not in (TRANSCRIPT_DIM_ID, OUTCOME_DIM_ID):
            continue  # recorded_judged_dims names only rubric dims and the axes; unreachable, but never guessed at
        scale = criterion.scale if criterion is not None else "ordinal"
        first = result.judge_score(dim)
        labels = judge_case_labels(storage, scope_id, result, dim, scale=scale)
        case = JudgeCase(
            dim=dim,
            scale=scale,
            criterion=criterion,
            intent=inputs.template.intent,
            variation=dict(inputs.test_case.variation_params),
            goal_outcomes=list(result.goal_state_outcomes),
            judged_artifact=inputs.judged_artifact,
            evidence=inputs.judge_evidence,
            labels=[
                JudgeCaseLabel(rater=rating.rater, score=rating.score, reason=rating.reason, rating_id=rating.id)
                for rating in labels
            ],
            source=JudgeCaseSource(
                run_id=run.id,
                result_id=result.id,
                test_case_id=result.test_case_id,
                first_score=first.score if first is not None else None,
                first_served_model=first.served_model if first is not None else None,
                first_judge_config_id=result.judge_config_ids.get(dim),
                first_judge_temperature=first.judge_temperature if first is not None else None,
            ),
        )
        frozen.append((dim, case))
    return frozen


def freeze_judge_cases(
    storage: EvalStorage,
    *,
    template: EvalTemplate,
    run_ids: Sequence[str],
    scope_id: str,
    dims: Collection[str] | None = None,
    result_ids: Collection[str] | None = None,
    case_set: str | None = None,
) -> JudgeCaseFreezeReport:
    """Freeze (output, criterion, labels) cases from stored judged results into a judge template.

    One case per judged dim of each result: the evidence the result's judge read, rebuilt from what its run
    recorded (:func:`~threetears.evals.run.rejudge.reproducible_judge_inputs` — a result whose template was edited
    after its run, or whose run did not record its judging, is skipped with that reason), the criterion as its
    template worded it, and the person labels :func:`judge_case_labels` finds for it. A case's id is derived from
    the template, the result, the dim and everything the case freezes, so freezing the same thing twice names the
    stored case, and a re-freeze after a label changed mints a new case beside the old one.

    With ``case_set``, the freeze then mints the next version of that case set listing exactly this freeze's cases,
    in order — the set a launch targets, so cases a later freeze superseded are not run by it. A freeze whose cases
    the latest version already lists exactly is answered with that version.

    Args:
        storage: The store the runs, their results and ratings live in, and the cases are written to.
        template: The judge template the cases belong to, loaded.
        run_ids: The judged runs whose results to freeze.
        scope_id: The scope the runs live in, and the cases are stored in.
        dims: The criteria to freeze; ``None`` for every dim each result was judged on.
        result_ids: The results to freeze; ``None`` for every result of the runs.
        case_set: The case set to mint over the frozen cases, or ``None`` for none.

    Returns:
        The cases, the set, and every result skipped with why.

    Raises:
        ValidationFailedError: The template is not of the judge kind; no run is named; ``result_ids`` names a result
            outside the runs; or nothing could be frozen.
        NotFoundError: A run does not load.
        StorageError: A case or the set could not be persisted.
    """
    _judge_template(template)
    if not run_ids:
        raise ValidationFailedError("name at least one judged run whose results to freeze")
    cases: list[FrozenJudgeCase] = []
    skipped: list[JudgeCaseSkip] = []
    seen: set[str] = set()
    wanted = None if result_ids is None else set(result_ids)
    found: set[str] = set()
    for run_id in run_ids:
        run = storage.load_eval_run(run_id, scope_id)
        if run is None:
            raise NotFoundError("run", run_id)
        for result in storage.query_eval_results_by_run(run.id, scope_id):
            if wanted is not None and result.id not in wanted:
                continue
            found.add(result.id)
            try:
                frozen = _cases_of_result(storage, template, run, result, scope_id, dims)
            except (ValidationFailedError, NotFoundError) as refused:
                skipped.append(JudgeCaseSkip(result_id=result.id, reason=str(refused)))
                continue
            if not frozen:
                skipped.append(JudgeCaseSkip(result_id=result.id, reason="none of the dims asked for was judged on it"))
            for dim, case in frozen:
                digest = case.content_digest()
                case_id = str(uuid.uuid5(_CASE_ID_NAMESPACE, f"{template.id}|{result.id}|{dim}|{digest}"))
                if case_id in seen:
                    continue
                seen.add(case_id)
                stored = storage.load_test_case(case_id, scope_id)
                if stored is None:
                    storage.save_test_case(
                        EvalTestCase(
                            id=case_id,
                            scope_id=scope_id,
                            template_id=template.id,
                            host_payload={JUDGE_CASE_KEY: case.model_dump(mode="json")},
                            # The criterion is what a judge campaign's readings split by, so every measure of a cell is
                            # summarised per criterion beside the pooled figure.
                            stratum=dim,
                            content_hash=digest,
                        )
                    )
                cases.append(
                    FrozenJudgeCase(
                        test_case_id=case_id,
                        source_run_id=run.id,
                        source_result_id=result.id,
                        dim=dim,
                        labels=len(case.labels),
                        created=stored is None,
                    )
                )
    if wanted is not None and (stray := sorted(wanted - found)):
        raise ValidationFailedError(f"result(s) {stray} are not results of the named runs in this scope")
    if not cases:
        why = "; ".join(f"{skip.result_id}: {skip.reason}" for skip in skipped) or "the runs have no results"
        raise ValidationFailedError(f"no judge case could be frozen — {why}")
    minted: CaseSetRef | None = None
    if case_set is not None:
        ids = [case.test_case_id for case in cases]
        versions = storage.query_case_sets(scope_id, name=case_set)
        latest = versions[0] if versions else None
        if latest is not None and latest.template_id == template.id and latest.test_case_ids == ids:
            minted = latest.ref
        else:
            minted = mint_case_set(
                storage, scope_id=scope_id, name=case_set, template_id=template.id, test_case_ids=ids
            ).ref
    log.info(
        "eval.judge_cases_freeze template=%s runs=%d cases=%d created=%d skipped=%d case_set=%s",
        template.id,
        len(run_ids),
        len(cases),
        sum(1 for case in cases if case.created),
        len(skipped),
        minted.label if minted is not None else "none",
    )
    return JudgeCaseFreezeReport(template_id=template.id, cases=cases, case_set=minted, skipped=skipped)


# =============================================================================
# The kind
# =============================================================================


@dataclass
class _CellClient:
    """One cell's view of the judge's client: every completion it returned, and the call that raised.

    Per cell, so what it records is this trial's. It owns nothing — the client it forwards to is the kind's, and
    is released with the kind — so releasing it is a no-op.
    """

    inner: Any
    completions: list[Any] = field(default_factory=list)
    failure: BaseException | None = None

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
        """Forward one call and keep its completion, or the exception it raised.

        Args:
            system: The system prompt.
            user: The user message.
            response_format: The provider directive.

        Returns:
            The provider's completion, unchanged.
        """
        try:
            completion = await self.inner.generate(system=system, user=user, response_format=response_format)
        # prawduct:ok-broad-except — not swallowed: recorded so invoke can tell a call that failed from a reply that broke the protocol, then re-raised
        except Exception as exc:
            self.failure = exc
            raise
        self.completions.append(completion)
        return completion

    async def aclose(self) -> None:
        """Release nothing: the client this forwards to is the kind's."""


@dataclass
class PreparedJudge:
    """One cell's judge: the model it binds, and the cell's tracing windows."""

    model: str
    span_window: CellSpanWindow


class JudgeKind:
    """A judge configuration as a candidate: asks each frozen case's criterion of one judge, and grades the answer by code.

    One instance is one judge — its model, its prompt per criterion and its temperature — and drives every cell of
    a run; a cell's state lives on what :meth:`prepare` returns and on the per-cell client view :meth:`invoke`
    builds. The clients it builds are its own: release them with :meth:`aclose` (or use it as an async context
    manager) once its run ends.
    """

    #: Graded by code against what the case froze: no judge phase runs over a judge kind's cells.
    judged_artifact = JudgedArtifact.UNJUDGED

    def __init__(
        self,
        *,
        clients: CompletionClients,
        model: str,
        failure_describer: ProviderFailureDescriber,
        configs: Mapping[str, JudgeConfig] | None = None,
        temperature: float | None = None,
    ) -> None:
        """Bind the judge under test.

        Args:
            clients: The host's client factory; every call is asked of a ``judge``-role client bound to ``model``.
            model: The judge's model — the run's candidate model, which every criterion is sent to.
            failure_describer: The host's reading of what a judge client raises — the only thing that can say a call
                was refused for the calling account.
            configs: criterion -> the versioned config whose prompt asks it; a criterion absent from the map is
                asked with the built-in prompt.
            temperature: The temperature every call is requested at; ``None`` for what each criterion's prompt asks
                for (its config's, else the default an unconfigured criterion is judged at).

        Raises:
            ValueError: A config is keyed by another criterion than the one it asks, or names a model of its own other
                than ``model`` — the judge under test is the one named here, and a config pinning another would
                score that criterion with a judge the run does not name.
        """
        configs = dict(configs or {})
        for dim, config in configs.items():
            if config.rubric_dim_id != dim:
                raise ValueError(f"judge config {config.id!r} asks {config.rubric_dim_id!r}, not {dim!r}")
            if config.model and config.model != model:
                raise ValueError(
                    f"judge config {config.id!r} pins model {config.model!r}, and this judge is {model!r}; a judge "
                    "campaign's arm names its model once, as its candidate model"
                )
        self._clients = clients
        self._model = model
        self._failure_describer = failure_describer
        self._configs = configs
        self._temperature = temperature
        self._built: dict[float | None, Any] = {}

    @property
    def model(self) -> str:
        """The judge's model."""
        return self._model

    def _client(self, temperature: float | None) -> Any:
        """The kind's judge client at ``temperature``, built on first ask and kept for every later cell."""
        sent = self._temperature if self._temperature is not None else temperature
        if sent not in self._built:
            self._built[sent] = self._clients("judge", self._model, temperature=sent)
        return self._built[sent]

    async def aclose(self) -> None:
        """Release every client this kind built, the first failure re-raised after all were offered a release."""
        built = list(self._built.values())
        self._built.clear()
        first: BaseException | None = None
        for client in built:
            try:
                await client.aclose()
            except (
                Exception
            ) as exc:  # prawduct:allow prawduct/broad-except -- one client's teardown must not strand the others
                log.warning("releasing a judge kind's client failed; its pool may leak", exc_info=True)
                first = first or exc
        if first is not None:
            raise first

    async def __aenter__(self) -> JudgeKind:
        """Enter a scope that releases the kind's clients on exit."""
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Release the kind's clients."""
        await self.aclose()

    async def prepare(
        self,
        *,
        subject_snapshot: Any,
        variant_config: VariantConfig,
        world_seed: Any,
        span_window: CellSpanWindow,
        cassettes: CellCassettes | None,
        world: Any,
    ) -> PreparedJudge:
        """Bind this cell's windows, refusing a cell recorded against another model than this judge's.

        Args:
            subject_snapshot: Unread — the judge under test is the one this kind holds.
            variant_config: This cell's contestant stack; its model is the judge's.
            world_seed: Unread — a judge perceives no world.
            span_window: This cell's tracing windows, opened in :meth:`invoke`.
            cassettes: Unwired — a judge calls no tool a cassette could record, so the engine refuses a cassette run.
            world: Unread.

        Returns:
            The cell's bound model and windows.

        Raises:
            CandidatePreparationFailed: The cell's model is not this judge's.
        """
        if variant_config.candidate_model != self._model:
            raise CandidatePreparationFailed(
                f"apparatus: this cell is recorded against model {variant_config.candidate_model!r} but the judge "
                f"under test is {self._model!r}, so running it would report one judge's answers under another's "
                "name — wire one judge kind per judge model",
                termination="factory_failed",
            )
        return PreparedJudge(model=self._model, span_window=span_window)

    def _request(self, service: JudgeService, case: JudgeCase, context: JudgeContext) -> JudgeRequest:
        """The call the case's criterion is asked in, built by the judge service exactly as a run's judge builds it."""
        if case.dim == TRANSCRIPT_DIM_ID:
            return service.transcript_request(context)
        if case.dim == OUTCOME_DIM_ID:
            return service.outcome_request(context)
        if case.criterion is None:
            raise ValueError(f"judge case on {case.dim!r} carries no criterion to ask")
        return service.dimension_request(case.criterion, context)

    async def invoke(self, instance: PreparedJudge, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        """Ask the judge under test the case's criterion over the frozen evidence, and grade its answer.

        Never raises. A case that cannot be read is the harness's (``infra_errors``, the cell excluded). The judge's
        own call failing is the candidate's (``candidate_errors``) — unless the host's describer says the call was
        refused for the calling account, which excludes the cell and stops the run (``account_refused``). A reply
        that came back is a measurement whatever it says: a score, a "can't tell", or a reply that broke the
        protocol, which is what parse validity counts.

        Args:
            instance: What :meth:`prepare` bound.
            test_case: The cell's case, carrying the judge case in its ``host_payload``.
            sink: Unused — one call, whose spend there is nothing to report until it returns.

        Returns:
            The answer, its code grade on ``host_measures``, the trial on ``kind_payload``, and the call's spend as
            the ``judge`` role's usage.
        """
        scopes = instance.span_window
        with scopes.identity():
            try:
                case = judge_case_of(test_case)
            except ValueError as exc:
                return CandidateOutput(infra_errors=[f"apparatus: {exc}"])
            if case is None:
                return CandidateOutput(
                    infra_errors=[
                        f"apparatus: test case {test_case.id} carries no judge case under host_payload[{JUDGE_CASE_KEY!r}]"
                    ]
                )
            cell = _CellClient(inner=None)

            def client_for(_model: str | None, temperature: float | None) -> _CellClient:
                cell.inner = self._client(temperature)
                return cell

            service = JudgeService(
                client_factory=client_for, configs=self._configs, failure_describer=self._failure_describer
            )
            try:
                context = JudgeContext(
                    case_id=test_case.id,
                    intent=case.intent,
                    variation=dict(case.variation),
                    goal_outcomes=list(case.goal_outcomes),
                    judged_artifact=case.judged_artifact,
                    judge_evidence=case.evidence,
                )
                request = self._request(service, case, context)
            except ValueError as exc:
                return CandidateOutput(infra_errors=[f"apparatus: test case {test_case.id}'s judge case: {exc}"])
            with scopes.collecting():
                outcome = await service.score_request(request, case_id=test_case.id)
        return self._graded(case, test_case, request, outcome, cell)

    def _graded(
        self, case: JudgeCase, test_case: EvalTestCase, request: JudgeRequest, outcome: JudgeOutcome, cell: _CellClient
    ) -> CandidateOutput:
        """The cell's output: the trial, its code grade, and its spend under the judge role."""
        ledger = RoleUsageLedger(role=JUDGE_KIND_ROLE)
        if outcome.usage is not None:
            usage: CallUsage = outcome.usage
            ledger.add_llm_result(replace(usage, model=usage.model or self._model))
        telemetry = CandidateTelemetry(usage=ledger.rows(), turns_delivered=len(cell.completions))
        last = cell.completions[-1] if cell.completions else None
        outcome_kind: JudgeTrialOutcome
        if outcome.score is not None:
            outcome_kind = "scored"
        elif outcome.cannot_tell is not None:
            outcome_kind = "cannot_tell"
        elif cell.failure is not None:
            outcome_kind = "no_reply"
        else:
            outcome_kind = "invalid"
        score = outcome.score
        trial = JudgeTrial(
            test_case_id=test_case.id,
            dim=case.dim,
            scale=case.scale,
            criterion_digest=case.criterion_digest(),
            case_digest=case.content_digest(),
            source_result_id=case.source.result_id,
            labels=case.labels,
            outcome=outcome_kind,
            score=score.score if score is not None else None,
            reasoning=score.reasoning if score is not None else None,
            cannot_tell=outcome.cannot_tell,
            error=outcome.error,
            served_model=(score.served_model if score is not None else (getattr(last, "served_model", None) or None)),
            judge_config_id=request.config.id if request.config is not None else None,
            judge_temperature=(
                score.judge_temperature if score is not None else (sent_temperature(last) if last is not None else None)
            ),
        )
        payload = trial.model_dump(mode="json")
        if outcome_kind == "no_reply":
            error = outcome.error or "the judge's call failed"
            if outcome.account_refused:
                return CandidateOutput(
                    infra_errors=[f"apparatus: judge: the call was refused for the calling account: {error}"],
                    account_refused=True,
                    telemetry=telemetry,
                    kind_payload=payload,
                )
            log.warning("Judge kind's call failed for case %s: %s", test_case.id, error)
            return CandidateOutput(candidate_errors=[f"judge: {error}"], telemetry=telemetry, kind_payload=payload)
        measures: dict[str, bool | float | str] = {JUDGE_PARSE_VALID_MEASURE: outcome_kind != "invalid"}
        if score is not None and case.labels:
            measures[JUDGE_LABEL_AGREEMENT_MEASURE] = sum(
                1 for label in case.labels if label.score == score.score
            ) / len(case.labels)
        if outcome_kind == "invalid":
            log.info("Judge kind's reply broke the protocol for case %s: %s", test_case.id, outcome.error)
        return CandidateOutput(
            output=[
                {
                    "outcome": outcome_kind,
                    "score": trial.score,
                    "reasoning": trial.reasoning,
                    "cannot_tell": trial.cannot_tell,
                    "error": trial.error,
                }
            ],
            host_measures=measures,
            telemetry=telemetry,
            kind_payload=payload,
        )


def judge_kind(
    storage: EvalStorage,
    clients: CompletionClients,
    *,
    model: str,
    scope_id: str,
    failure_describer: ProviderFailureDescriber,
    overlays: JudgeKindOverlays | None = None,
) -> JudgeKind:
    """The judge kind for one arm, its configs loaded from the store: what a launcher builds per arm.

    Args:
        storage: The store the configs live in.
        clients: The host's client factory.
        model: The arm's judge model (its candidate model).
        scope_id: The scope the configs live in.
        failure_describer: The host's reading of what a judge client raises.
        overlays: The arm's overlays — its prompt per criterion and its temperature; ``None`` for the built-in
            prompt on every criterion at each prompt's own temperature.

    Returns:
        The kind, owning the clients it builds.

    Raises:
        NotFoundError: A config the overlays name does not load.
        ValidationFailedError: A config asks another criterion than it is named for, or pins another model.
    """
    overlays = overlays if overlays is not None else JudgeKindOverlays()
    configs: dict[str, JudgeConfig] = {}
    for dim, config_id in overlays.config_ids.items():
        config = storage.load_judge_config(config_id, scope_id)
        if config is None:
            raise NotFoundError("judge_config", config_id)
        configs[dim] = config
    try:
        return JudgeKind(
            clients=clients,
            model=model,
            failure_describer=failure_describer,
            configs=configs,
            temperature=overlays.temperature,
        )
    except ValueError as exc:
        raise ValidationFailedError(str(exc)) from exc


def _template_cases(storage: EvalStorage, template: EvalTemplate, scope_id: str) -> list[EvalTestCase]:
    """The judge template's stored live cases, in storage order — what an arm runs when its launch names no set."""
    return [
        case
        for case in storage.query_test_cases(scope_id, template_id=template.id)
        if not case.archived and JUDGE_CASE_KEY in (case.host_payload or {})
    ]


def launchable_judge_kind(launch_host: Callable[[], LaunchHost]) -> LaunchableKind:
    """The judge kind's entry in a host's launch registry: each arm one judge over the template's frozen cases.

    A host launching judge campaigns through :func:`~threetears.evals.run.start_run` registers this under
    :data:`~threetears.evals.kernel.JUDGE_KIND` and declares :data:`~threetears.evals.kernel.JUDGE_KIND_CONTRACT`
    on its profile, so each arm's prompt per criterion and temperature arrive as its overlays. An arm's model is
    its candidate model. The arm runs its launch's case set when it names one, else every live judge case of the
    template. No simulator, no judge of its own (the judge is the candidate), no cassette, and no generated case.

    Args:
        launch_host: Returns the launch host the entry is registered on, which the launcher hands the launch tail;
            a callable because the host is built with its registry, this entry in it.

    Returns:
        The registry entry.
    """

    def plan(request: LaunchRequest) -> ArmPlan:
        host = launch_host().eval_host
        cases = request.cases_or(_template_cases(host.storage, request.template, request.scope_id))
        if not cases:
            raise ValidationFailedError(
                f"template {request.template.id!r} holds no judge case to run; freeze some with judge_cases_freeze"
            )
        return ArmPlan(
            case_count=len(cases),
            candidate_model=require_candidate_model(request, None),
            judge=None,
            simulator_model=None,
        )

    async def launch(request: LaunchRequest) -> Any:
        launching = launch_host()
        host = launching.eval_host
        model = require_candidate_model(request, None)
        overlays = request.overlays_as(JudgeKindOverlays) if request.overlays is not None else JudgeKindOverlays()
        kind = judge_kind(
            host.storage,
            host.completion_clients("a judge campaign"),
            model=model,
            scope_id=request.scope_id,
            failure_describer=host.failure_describer,
            overlays=overlays,
        )
        teardown = contextlib.AsyncExitStack()
        teardown.push_async_callback(kind.aclose)
        cases = request.cases_or(_template_cases(host.storage, request.template, request.scope_id))
        return await launch_run(
            launching,
            request,
            KindWiring(
                kind_factory=lambda _cell: kind,
                subject=SubjectSnapshot(subject_id=request.subject_id, subject_label=request.subject_id, state={}),
                test_cases=cases,
                teardown=teardown,
            ),
        )

    return LaunchableKind(
        launch=launch,
        unhonoured_launch_arguments=frozenset(
            {"n_variations", "judge_model", "judge_config_ids", "simulator_model", "cassette_mode"}
        ),
        plan_arm=plan,
    )
