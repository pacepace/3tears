"""Ask a second judge to score a finished run's stored evidence: inter-judge agreement, and judge drift.

A run's judged scores come from one judge — a model, the prompt per dimension and a temperature. Two questions
need a SECOND judge's answer to the same evidence, and this module asks it:

- **How far would another judge agree?** (#646) The cheap check for what a stored calibration profile cannot
  cover: drift since the judge was profiled, a criterion with no human labels yet, outputs unlike the profiling
  set. A seeded share of the run's judged results is enough; per-dimension agreement is read by the same code as
  agreement with people (:func:`~threetears.evals.analysis.inter_judge_agreement`).
- **How far did the scores move when the judge changed?** (#597) Every judged result re-scored under the new judge,
  and each dimension's movement read with an interval over the run's cases
  (:func:`~threetears.evals.analysis.judge_drift`). It detects movement; it cannot say which judge is right.

Both are one operation, :func:`ask_second_judge`, over the machinery a judge repeat uses
(:mod:`threetears.evals.run.judge_repeat`):

- **The evidence the first judge read, or no second opinion.** Each result's judge input is rebuilt from what its
  run recorded (:func:`~threetears.evals.run.rejudge.reproducible_judge_inputs`) — the template's intent and rubric,
  the case, and the evidence the cell's kind rendered for its first judge, read back off the stored trace. Only the
  judge differs, and it is named in full (:class:`~threetears.evals.schema.models.SecondJudge`).
- **Priced before it is paid for.** Every call, parse retries included, is priced on the client it will be made on
  and admitted against the host's out-of-run cap before the first is sent. Each call made is written to the
  out-of-run ledger under purpose ``second_judge``, stamped with the run: measurement cost on its own line, never
  added to the candidate's ``cost_usd``.
- **A measurement of the judge, never a change to the result.** The second judge's answers are recorded on the
  result as a :class:`~threetears.evals.schema.models.SecondJudging` beside the first scores they pair with; the
  scores every lens reads stay the ones the cell was judged with.
- **A seeded sample, recorded.** The share asked is drawn from the run's judgeable results, sorted by id, with the
  seed given, so the same seed draws the same results; the fraction, the seed and the pass's id are written on every
  entry it records.
"""

from __future__ import annotations

import math
import random
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import ValidationError as PydanticValidationError

from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.kernel.errors import ConflictError, NotFoundError, StorageError, ValidationFailedError
from threetears.evals.schema.models import (
    NON_TERMINAL_RUN_STATUSES,
    EvalResult,
    EvalRun,
    SecondJudge,
    SecondJudgeScore,
    SecondJudging,
)
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel.out_of_run import OutOfRunBudget, PlannedCall
from threetears.evals.run.judge import JUDGE_CALL_ATTEMPTS, JUDGE_REQUEST_SETTINGS
from threetears.evals.run.judge_repeat import BudgetedJudgeClient, planned_judge_call
from threetears.evals.run.judge_service import JudgeService
from threetears.evals.run.rejudge import (
    ReproducibleJudgeInputs,
    judged_template,
    recorded_judge_pins,
    reproducible_judge_inputs,
)
from threetears.evals.run.runner import build_judge_context, judge_dims, judge_requests
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.evals.kernel.host.eval_host import EvalHost
    from threetears.evals.schema.models import JudgeConfig
    from threetears.evals.kernel.storage import EvalStorage
    from threetears.evals.run.judge_service import JudgeContext, JudgeOutcome

log = get_logger(__name__)

#: How many times a result's record is re-read and re-applied when another writer changed it between the read and
#: the write. The record only appends, so re-applying it to the newer result is safe.
_WRITE_ATTEMPTS = 3

#: The seed a pass is drawn with when the caller names none — fixed, so an unseeded call is still reproducible.
DEFAULT_SECOND_JUDGE_SEED = 0


class SecondJudgeSkip(EvalBaseModel):
    """A result of the run that the second judge is not asked about, and why."""

    result_id: str
    reason: str


class SecondJudgeEstimate(EvalBaseModel):
    """What asking a second judge about a run would be priced at, against the cap it would be held to — no call made.

    Attributes:
        run_id: The run.
        judge: The second judge.
        eligible: The run's results the second judge could be asked about.
        sampled: The results the seeded sample draws, sorted.
        dims: The judge calls it would send, one per scored dim per sampled result.
        max_calls: The most calls it can make: each dim's first attempt and every parse retry.
        ceiling_usd: The most those calls can cost together, as the clients price them; ``None`` when a client
            cannot say.
        cap_usd: The out-of-run cap it would be held to; ``None`` when the host enforces none.
        would_start: Whether every call would be admitted.
        refusal: Why it would not, when it would not.
        skipped: The run's results that could not be asked about, each with why.
    """

    run_id: str
    judge: SecondJudge
    eligible: int
    sampled: list[str]
    dims: int
    max_calls: int
    ceiling_usd: float | None
    cap_usd: float | None
    would_start: bool
    refusal: str | None = None
    skipped: list[SecondJudgeSkip]


class SecondJudgeReport(EvalBaseModel):
    """What a second judge's pass did, what it spent apart from the candidate's cost, and what it read.

    Attributes:
        run_id: The run.
        pass_id: The pass, as every entry it recorded names it.
        judge: The second judge.
        sample_fraction: The share of the eligible results it drew.
        sample_seed: The seed it drew them with.
        eligible: The run's results it could have asked about.
        sampled: The results the sample drew, sorted by id.
        judged: The results a second score was recorded on, in the order they were asked.
        scores_paired: Dims the second judge scored.
        scores_unanswered: Dims whose call failed or answered it could not tell.
        skipped: The run's results not asked about, each with why — decided before anything was spent.
        unwritten: Results asked and paid for whose record did not land. Their spend is in the ledger.
        stopped: Why the pass stopped before its last result, when it did — the account behind the judge refused a
            call, so every later one would be refused too.
        calls_made: Calls made, as the ledger recorded them under purpose ``second_judge``.
        cost_usd: What they cost together — measurement cost, never the candidate's; ``None`` when any went unpriced.
        cap_usd: The out-of-run cap the calls were admitted under; ``None`` when the host enforces none.
    """

    run_id: str
    pass_id: str
    judge: SecondJudge
    sample_fraction: float
    sample_seed: int
    eligible: int
    sampled: list[str]
    judged: list[str]
    scores_paired: int
    scores_unanswered: int
    skipped: list[SecondJudgeSkip]
    unwritten: list[str]
    stopped: str | None = None
    calls_made: int
    cost_usd: float | None
    cap_usd: float | None


@dataclass(frozen=True)
class _Planned:
    """One result to ask about: what its first judge read, and the dims it holds a score on, in dimension order."""

    result: EvalResult
    inputs: ReproducibleJudgeInputs
    dims: list[str]


@dataclass(frozen=True)
class _Collected:
    """What a pass reads before it builds anything."""

    run: EvalRun
    eligible: int
    planned: list[_Planned]
    configs: dict[str, JudgeConfig]
    skipped: list[SecondJudgeSkip]


def _scored_dims(result: EvalResult) -> set[str]:
    """The dims ``result`` holds a judge score on — the only ones a second judge has a first score to pair with."""
    return {score.dim for score in result.judge_scores()}


def _same_judge(run: EvalRun, judge: SecondJudge) -> bool:
    """Whether ``judge`` names exactly the run's own judge: its pin, its prompts and its sampling."""
    return judge.model == run.judge_model and judge.config_ids is None and judge.temperature is None


def _sample(eligible: Sequence[str], fraction: float, seed: int) -> list[str]:
    """The seeded share of ``eligible`` a pass asks about: ``ceil(fraction · n)`` of them, drawn without replacement.

    The ids are sorted before the draw, so the same seed over the same results draws the same ones whatever order the
    store returned them in. At least one is drawn from a non-empty set.
    """
    ordered = sorted(eligible)
    count = min(len(ordered), max(1, math.ceil(fraction * len(ordered))))
    return sorted(random.Random(seed).sample(ordered, count))


def _collect(
    storage: EvalStorage,
    run_id: str,
    scope_id: str,
    judge: SecondJudge,
    result_ids: Sequence[str] | None,
    fraction: float,
    seed: int,
) -> _Collected:
    """Load the run and every result to ask about, refusing what cannot be asked — blocking, so off the loop.

    Raises:
        NotFoundError: No run with that id, its template, or a judge config the second judge names, does not load.
        ValidationFailedError: The run is still running or was not judged; its judging apparatus was not recorded;
            its judge request settings are not the ones a call sends now; the second judge is the run's own;
            ``result_ids`` names a result not in the run; or no result can be asked about.
    """
    run = storage.load_eval_run(run_id, scope_id)
    if run is None:
        raise NotFoundError("run", run_id)
    if run.status in NON_TERMINAL_RUN_STATUSES:
        raise ValidationFailedError(f"run '{run_id}' is {run.status} — its results are still being written")
    # The run's judge is the FIRST judge, so its apparatus must be recorded; a second judge may be sampled at another
    # temperature (that is part of its identity), but not asked with other request settings, which it does not name.
    recorded_judge_pins(run, request_settings="as_recorded")
    if run.judge_request_settings != JUDGE_REQUEST_SETTINGS:
        raise ValidationFailedError(
            f"run '{run_id}' was judged with request settings {run.judge_request_settings!r} and a judge call now "
            f"sends {JUDGE_REQUEST_SETTINGS!r}; the second judge would differ from the first by more than it names"
        )
    if _same_judge(run, judge):
        raise ValidationFailedError(
            f"the second judge names run '{run_id}''s own judge ({judge.model}, its recorded prompts and sampling); "
            "asking it again measures the judge against itself — that is a judge repeat"
        )
    judged_template(storage, run, scope_id)
    configs: dict[str, JudgeConfig] = {}
    if judge.config_ids is not None:
        for dim, config_id in judge.config_ids.items():
            config = storage.load_judge_config(config_id, scope_id)
            if config is None:
                raise NotFoundError("judge_config", config_id)
            if config.rubric_dim_id != dim:
                raise ValidationFailedError(
                    f"judge config '{config_id}' asks for {config.rubric_dim_id!r}, not {dim!r}, so it cannot ask the "
                    "second judge for that dimension"
                )
            configs[dim] = config
    results = storage.query_eval_results_by_run(run.id, scope_id)
    if result_ids is not None:
        known = {result.id for result in results}
        if stray := sorted(set(result_ids) - known):
            raise ValidationFailedError(f"result(s) {stray} are not results of run '{run_id}' in this scope")
        wanted = set(result_ids)
        results = [result for result in results if result.id in wanted]
    eligible: list[_Planned] = []
    skipped: list[SecondJudgeSkip] = []
    for result in results:
        scored = _scored_dims(result)
        if not scored:
            skipped.append(SecondJudgeSkip(result_id=result.id, reason="it holds no judge score to pair with"))
            continue
        try:
            inputs = reproducible_judge_inputs(
                storage, result, run, scope_id, request_settings="as_recorded", config_dims=scored
            )
        except (ValidationFailedError, NotFoundError) as refused:
            skipped.append(SecondJudgeSkip(result_id=result.id, reason=str(refused)))
            continue
        dims = [dim for dim in inputs.dims if dim in scored]
        if not dims:
            skipped.append(SecondJudgeSkip(result_id=result.id, reason="none of its scored dims is one its run judged"))
            continue
        eligible.append(_Planned(result=result, inputs=inputs, dims=dims))
    if not eligible:
        why = "; ".join(f"{skip.result_id}: {skip.reason}" for skip in skipped) or "it has no results"
        raise ValidationFailedError(f"no result of run '{run_id}' can be asked about — {why}")
    drawn = set(_sample([planned.result.id for planned in eligible], fraction, seed))
    planned = [one for one in eligible if one.result.id in drawn]
    if judge.config_ids is None:
        for one in planned:
            configs.update(one.inputs.configs)
    return _Collected(run=run, eligible=len(eligible), planned=planned, configs=configs, skipped=skipped)


@dataclass
class _Prepared:
    """A pass built and priced: the judge service over budgeted clients, and each result's context."""

    collected: _Collected
    budget: OutOfRunBudget
    service: JudgeService
    contexts: list[JudgeContext]
    calls: list[tuple[BudgetedJudgeClient, list[PlannedCall]]]


async def _prepare(
    host: EvalHost,
    run_id: str,
    scope_id: str,
    *,
    judge: SecondJudge,
    out_of_run_cap_usd: float | None,
    result_ids: Sequence[str] | None,
    sample_fraction: float,
    seed: int,
) -> _Prepared:
    """Collect the run, build the second judge over budgeted clients, and plan every call — admitting nothing yet.

    The caller owns ``service`` and enters it, which releases every client built here.

    Raises:
        ValidationFailedError: ``sample_fraction`` is outside ``(0, 1]``, or :func:`_collect` refuses the run.
        ValueError: The host supplies no completion clients.
    """
    if not 0.0 < sample_fraction <= 1.0:
        raise ValidationFailedError(f"sample_fraction {sample_fraction!r} is outside (0, 1]")
    clients = host.completion_clients("a second judge")
    collected = await run_blocking(
        host.blocking_executor, _collect, host.storage, run_id, scope_id, judge, result_ids, sample_fraction, seed
    )
    budget = OutOfRunBudget(
        store=host.storage,
        scope_id=scope_id,
        cap_usd=out_of_run_cap_usd,
        template_id=collected.run.template_id,
        run_id=collected.run.id,
        blocking_executor=host.blocking_executor,
    )

    def budgeted(_model: str | None, temperature: float | None) -> BudgetedJudgeClient:
        # Every dim goes to the second judge's model, whatever its prompt's config names, at its temperature when set.
        sent = judge.temperature if judge.temperature is not None else temperature
        return BudgetedJudgeClient(clients("judge", judge.model, temperature=sent), budget, "second_judge")

    service = JudgeService(client_factory=budgeted, configs=collected.configs, failure_describer=host.failure_describer)
    contexts: list[JudgeContext] = []
    by_client: dict[int, tuple[BudgetedJudgeClient, list[PlannedCall]]] = {}
    for planned in collected.planned:
        context = build_judge_context(
            template=planned.inputs.template,
            test_case=planned.inputs.test_case,
            goal_outcomes=planned.result.goal_state_outcomes,
            judged_artifact=planned.inputs.judged_artifact,
            judge_evidence=planned.inputs.judge_evidence,
        )
        contexts.append(context)
        requests = judge_requests(
            template=planned.inputs.template, judge_service=service, context=context, only=frozenset(planned.dims)
        )
        for request in requests:
            client: BudgetedJudgeClient = service.client_for(request)
            by_client.setdefault(id(client), (client, []))[1].extend(
                [planned_judge_call(request)] * JUDGE_CALL_ATTEMPTS
            )
    return _Prepared(
        collected=collected, budget=budget, service=service, contexts=contexts, calls=list(by_client.values())
    )


async def estimate_second_judge(
    host: EvalHost,
    run_id: str,
    scope_id: str,
    *,
    judge: SecondJudge,
    out_of_run_cap_usd: float | None,
    sample_fraction: float = 1.0,
    seed: int = DEFAULT_SECOND_JUDGE_SEED,
    result_ids: Sequence[str] | None = None,
) -> SecondJudgeEstimate:
    """Price asking a second judge about a run against the cap it would be held to, and make no call.

    The same collection, sample, calls and admission :func:`ask_second_judge` makes, so ``would_start`` is its answer.

    Args:
        host: The host: the run's store and the judge clients.
        run_id: The finished run.
        scope_id: The scope it lives in.
        judge: The second judge.
        out_of_run_cap_usd: The cap it would be held to; ``None`` when the host enforces none.
        sample_fraction: The share of the run's judgeable results to ask about, in ``(0, 1]``.
        seed: The seed the share is drawn with.
        result_ids: The results to draw from; ``None`` for every result of the run.

    Returns:
        The estimate.

    Raises:
        NotFoundError: No run with that id, or a record it names does not load.
        ValidationFailedError: The run cannot be asked about (see :func:`ask_second_judge`).
        ValueError: The host supplies no completion clients.
    """
    prepared = await _prepare(
        host,
        run_id,
        scope_id,
        judge=judge,
        out_of_run_cap_usd=out_of_run_cap_usd,
        result_ids=result_ids,
        sample_fraction=sample_fraction,
        seed=seed,
    )
    async with prepared.service:
        ceilings = [
            client.inner.price_ceiling(system=call.system, user=call.user, response_format=call.response_format)
            for client, calls in prepared.calls
            for call in calls
        ]
        refusal: str | None = None
        try:
            for client, calls in prepared.calls:
                client.admit(calls)
        except ValidationFailedError as refused:
            refusal = str(refused)
    collected = prepared.collected
    return SecondJudgeEstimate(
        run_id=collected.run.id,
        judge=judge,
        eligible=collected.eligible,
        sampled=sorted(planned.result.id for planned in collected.planned),
        dims=sum(len(planned.dims) for planned in collected.planned),
        max_calls=len(ceilings),
        ceiling_usd=None if None in ceilings else math.fsum(c for c in ceilings if c is not None),
        cap_usd=out_of_run_cap_usd,
        would_start=refusal is None,
        refusal=refusal,
        skipped=collected.skipped,
    )


async def ask_second_judge(
    host: EvalHost,
    run_id: str,
    scope_id: str,
    *,
    judge: SecondJudge,
    out_of_run_cap_usd: float | None,
    sample_fraction: float = 1.0,
    seed: int = DEFAULT_SECOND_JUDGE_SEED,
    result_ids: Sequence[str] | None = None,
) -> SecondJudgeReport:
    """Ask a second judge to score a seeded share of a finished run's judged results, and record each beside its scores.

    Every call is priced and admitted against ``out_of_run_cap_usd`` before the first is sent, so a pass the cap
    refuses has spent nothing. Results are asked one after another, each one's record written as soon as its calls
    return, so a pass cut off partway keeps what it paid for. What the pairs say — agreement
    (:func:`~threetears.evals.analysis.inter_judge_agreement`) and drift (:func:`~threetears.evals.analysis.judge_drift`)
    — is read off the stored pairs, by the same code wherever the results are read
    (:func:`~threetears.evals.ops.judge_second` reads both over the pass it ran).

    Args:
        host: The host: the run's store, its judge clients, and how a judge call that raised reads.
        run_id: The finished run.
        scope_id: The scope it lives in.
        judge: The second judge: the model every dim is sent to, the prompts, the temperature.
        out_of_run_cap_usd: The most the pass's calls may be priced at together; ``None`` when the host enforces
            no out-of-run cap.
        sample_fraction: The share of the run's judgeable results to ask about, in ``(0, 1]``; 1 asks about every
            one, as a drift check does.
        seed: The seed the share is drawn with.
        result_ids: The results to draw from; ``None`` for every result of the run.

    Returns:
        What was asked, and what it cost apart from the candidate.

    Raises:
        NotFoundError: No run with that id, or a record it names does not load.
        ValidationFailedError: The run is still running or was not judged; its judging apparatus was not recorded,
            or its judge request settings are not the ones a call sends now; the second judge is the run's own; a
            config it names asks for another dim; ``sample_fraction`` is outside ``(0, 1]``; ``result_ids`` names a
            result outside the run; no result can be asked about; or the calls cannot be priced under the enforced
            cap or are priced above it. All before any call.
        ValueError: The host supplies no completion clients.
    """
    prepared = await _prepare(
        host,
        run_id,
        scope_id,
        judge=judge,
        out_of_run_cap_usd=out_of_run_cap_usd,
        result_ids=result_ids,
        sample_fraction=sample_fraction,
        seed=seed,
    )
    collected = prepared.collected
    pass_id = str(uuid.uuid7())
    judged: list[str] = []
    unwritten: list[str] = []
    paired = unanswered = 0
    stopped: str | None = None
    async with prepared.service:
        for client, calls in prepared.calls:
            client.admit(calls)
        for planned, context in zip(collected.planned, prepared.contexts, strict=True):
            outcomes = await judge_dims(
                template=planned.inputs.template,
                judge_service=prepared.service,
                context=context,
                only=frozenset(planned.dims),
            )
            entry = _judging_of(
                planned.result, outcomes, pass_id=pass_id, judge=judge, fraction=sample_fraction, seed=seed
            )
            paired += sum(1 for score in entry.scores if score.second is not None)
            unanswered += sum(1 for score in entry.scores if score.second is None)
            if await run_blocking(host.blocking_executor, _record, host.storage, planned.result.id, scope_id, entry):
                judged.append(planned.result.id)
            else:
                unwritten.append(planned.result.id)
            if any(outcome.account_refused for _, outcome in outcomes):
                stopped = (
                    f"the account behind the second judge refused a call while asking about result "
                    f"'{planned.result.id}', so every later call would be refused too"
                )
                break
    spends = prepared.budget.recorded
    costs = [spend.cost_usd for spend in spends]
    report = SecondJudgeReport(
        run_id=collected.run.id,
        pass_id=pass_id,
        judge=judge,
        sample_fraction=sample_fraction,
        sample_seed=seed,
        eligible=collected.eligible,
        sampled=sorted(planned.result.id for planned in collected.planned),
        judged=judged,
        scores_paired=paired,
        scores_unanswered=unanswered,
        skipped=collected.skipped,
        unwritten=unwritten,
        stopped=stopped,
        calls_made=len(spends),
        cost_usd=None if None in costs else math.fsum(c for c in costs if c is not None),
        cap_usd=out_of_run_cap_usd,
    )
    log.info(
        "eval.judge_second run=%s pass=%s model=%s sampled=%d judged=%d paired=%d unanswered=%d calls=%d cost=%s",
        report.run_id,
        report.pass_id,
        judge.model,
        len(report.sampled),
        len(report.judged),
        report.scores_paired,
        report.scores_unanswered,
        report.calls_made,
        "unpriced" if report.cost_usd is None else f"${report.cost_usd:.6f}",
    )
    return report


def _judging_of(
    result: EvalResult,
    outcomes: list[tuple[str, JudgeOutcome]],
    *,
    pass_id: str,
    judge: SecondJudge,
    fraction: float,
    seed: int,
) -> SecondJudging:
    """The entry to record: each dim's first score beside the second judge's answer to it."""
    entries = []
    for dim, outcome in outcomes:
        first = result.judge_score(dim)
        assert first is not None  # only scored dims are planned, from this very result
        entries.append(
            SecondJudgeScore(
                dim=dim,
                scale=first.scale,
                first_score=first.score,
                first_served_model=first.served_model,
                first_judge_config_id=result.judge_config_ids.get(dim),
                first_judge_temperature=first.judge_temperature,
                second=outcome.score,
                error=(outcome.error or "no answer") if outcome.score is None and outcome.cannot_tell is None else None,
                cannot_tell=outcome.cannot_tell,
            )
        )
    return SecondJudging(
        pass_id=pass_id,
        judge=judge,
        sample_fraction=fraction,
        sample_seed=seed,
        scores=entries,
        judge_config_ids={dim: outcome.config_id for dim, outcome in outcomes if outcome.config_id is not None},
    )


def _record(storage: EvalStorage, result_id: str, scope_id: str, entry: SecondJudging) -> bool:
    """Append ``entry`` to the stored result, re-reading and re-applying it when another writer got there first.

    Blocking, so it runs on the host's executor. The read is inside the guard as well as the write: the calls are
    paid for and in the ledger by now, so a result that cannot be read costs this one result its record — named in
    the report's ``unwritten`` — and never the report of everything else the pass paid for.

    Returns:
        True once the entry is stored; False when the result is gone, could not be read, or every write was
        refused or failed.
    """
    for _ in range(_WRITE_ATTEMPTS):
        try:
            result, etag = storage.load_eval_result_with_etag(result_id, scope_id)
        except StorageError, PydanticValidationError:
            log.exception("eval.judge_second result=%s could not be read back; its entry is not stored", result_id)
            return False
        if result is None:
            log.error("eval.judge_second result=%s was deleted while asked about; its entry is not stored", result_id)
            return False
        updated = result.model_copy(update={"judge_seconds": [*result.judge_seconds, entry]})
        try:
            storage.replace_eval_result(updated, if_match=etag)
        except ConflictError:
            log.info("eval.judge_second result=%s changed between the read and the write; re-applying", result_id)
            continue
        except StorageError:
            log.exception("eval.judge_second result=%s: its entry could not be written", result_id)
            return False
        return True
    log.error(
        "eval.judge_second result=%s changed under every one of %d writes; its entry is not stored",
        result_id,
        _WRITE_ATTEMPTS,
    )
    return False


__all__ = [
    "DEFAULT_SECOND_JUDGE_SEED",
    "SecondJudgeEstimate",
    "SecondJudgeReport",
    "SecondJudgeSkip",
    "estimate_second_judge",
    "ask_second_judge",
]
