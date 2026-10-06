"""Repeat a finished run's judge scores: the same judge asked the same question again, to measure its agreement with itself.

The ``separation`` evidence tier (:mod:`threetears.evals.contracts.evidence_tiers`) reads how often a
judge gives the same score when it scores the same evidence twice. Nothing in a run measures that: a
cell is judged once. This module takes the measurement, on a run that has finished:

- **The same judge setup, or no repeat.** Every input a repeat sends is the one the run recorded —
  the judge pin, the config per dim, the request settings as the client sends them today, the
  template's intent and rubric, the test case, and the evidence the cell's kind rendered for its first
  judge, read back off the stored trace — through
  :func:`~threetears.evals.run.rejudge.reproducible_judge_inputs`, the one check a re-judge and a judge
  freeze make too. A run whose judging cannot be reproduced is refused; a result that cannot be is
  named and left out.
- **Priced before it is paid for, never paid for and then refused.** Every call the repeat can make —
  each scored dim's first attempt and every parse retry
  (:data:`~threetears.evals.run.judge.JUDGE_CALL_ATTEMPTS`) — is priced on the client it will be made
  on and admitted against the host's out-of-run cap
  (:class:`~threetears.evals.contracts.out_of_run.OutOfRunBudget`) before the first one is sent. Over
  the cap, or unpriceable under an enforced cap, the whole repeat is refused with nothing spent. Each
  call made is written to the out-of-run ledger under purpose ``judge``, stamped with the run, whether
  it returns or raises.
- **A measurement of the judge, never a change to the result.** The repeat's answers are recorded on
  the result as a :class:`~threetears.evals.contracts.models.JudgeRepeat` beside the first scores they
  repeat; the scores every lens reads stay the ones the cell was judged with.

Only dims the result holds a SCORE on are repeated: a dim the judge could not tell on, or whose call
failed, has no first score to agree with.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from threetears.evals.contracts.base import EvalBaseModel
from threetears.evals.contracts.errors import ConflictError, NotFoundError, StorageError, ValidationFailedError
from threetears.evals.contracts.models import (
    NON_TERMINAL_RUN_STATUSES,
    EvalResult,
    EvalRun,
    JudgeRepeat,
    RepeatedScore,
)
from threetears.evals.contracts.offload import run_blocking
from threetears.evals.contracts.out_of_run import AdmittedCall, OutOfRunBudget, PlannedCall
from threetears.evals.contracts.provider import JSON_OBJECT_RESPONSE_FORMAT
from threetears.evals.run.judge import JUDGE_CALL_ATTEMPTS
from threetears.evals.run.judge_service import JudgeService, judge_clients_for_run
from threetears.evals.run.rejudge import (
    ReproducibleJudgeInputs,
    judged_template,
    recorded_judge_pins,
    reproducible_judge_inputs,
)
from threetears.evals.run.runner import build_judge_context, judge_dims, judge_requests
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.evals.contracts.host.eval_host import EvalHost
    from threetears.evals.contracts.models import JudgeConfig
    from threetears.evals.contracts.provider import BoundCompletionClient, CompletionResult
    from threetears.evals.contracts.storage import EvalStorage
    from threetears.evals.run.judge_service import JudgeContext, JudgeOutcome, JudgeRequest

log = get_logger(__name__)

#: How many times a result's repeat is re-read and re-applied when another writer changed the result
#: between the read and the write. The repeat only appends, so re-applying it to the newer result is safe.
_WRITE_ATTEMPTS = 3


class JudgeRepeatSkip(EvalBaseModel):
    """A result of the run that is not repeated, and why."""

    result_id: str
    reason: str


class JudgeRepeatEstimate(EvalBaseModel):
    """What repeating a run's judge scores would be priced at, against the cap it would be held to — no call made.

    Attributes:
        run_id: The run.
        judge_model: The run's judge pin.
        results: The results that would be repeated.
        dims: The judge calls a repeat would send, one per scored dim per result.
        max_calls: The most calls it can make: each dim's first attempt and every parse retry.
        ceiling_usd: The most those calls can cost together, as the clients price them; ``None`` when a
            client cannot say.
        cap_usd: The out-of-run cap the repeat would be held to; ``None`` when the host enforces none.
        would_start: Whether every call would be admitted.
        refusal: Why it would not, when it would not.
        skipped: The run's results that would not be repeated, each with why.
    """

    run_id: str
    judge_model: str
    results: int
    dims: int
    max_calls: int
    ceiling_usd: float | None
    cap_usd: float | None
    would_start: bool
    refusal: str | None = None
    skipped: list[JudgeRepeatSkip]


class JudgeRepeatReport(EvalBaseModel):
    """What a repeat did: which results it recorded a repeat on, what it spent, and what it left out.

    Attributes:
        run_id: The run.
        judge_model: The run's judge pin.
        repeated: The results a repeat was recorded on, in the order they were repeated.
        scores_repeated: Dims the judge scored again.
        scores_unanswered: Dims whose repeat failed or answered it could not tell.
        skipped: The run's results not repeated, each with why — decided before anything was spent.
        unwritten: Results repeated and paid for whose record did not land (deleted meanwhile, or the
            write failed). Their spend is in the ledger.
        stopped: Why the repeat stopped before its last result, when it did — the account behind the
            judge refused a call, so every later call would be refused too.
        calls_made: Judge calls made, as the ledger recorded them.
        cost_usd: What they cost together, as reported; ``None`` when any call went unpriced.
        cap_usd: The out-of-run cap the calls were admitted under; ``None`` when the host enforces none.
    """

    run_id: str
    judge_model: str
    repeated: list[str]
    scores_repeated: int
    scores_unanswered: int
    skipped: list[JudgeRepeatSkip]
    unwritten: list[str]
    stopped: str | None = None
    calls_made: int
    cost_usd: float | None
    cap_usd: float | None


class _BudgetedJudgeClient:
    """A judge client whose every call goes through the repeat's out-of-run budget, as admitted.

    The judge service is handed these in place of the host's clients, so the call
    :func:`~threetears.evals.run.judge.run_judge_llm` sends — first attempt and parse retry alike — is made
    only if an admission priced for exactly that prompt is waiting, and is ledgered however it ends.
    """

    def __init__(self, inner: BoundCompletionClient, budget: OutOfRunBudget) -> None:
        """Wrap ``inner`` so its calls are made through ``budget``.

        Args:
            inner: The host's judge client, which this owns from here and releases.
            budget: The repeat's budget.
        """
        self._inner = inner
        self._budget = budget
        self._pending: list[AdmittedCall] = []

    @property
    def inner(self) -> BoundCompletionClient:
        """The host's client the calls are priced on and made through."""
        return self._inner

    @property
    def model_name(self) -> str:
        """The model the host's client calls."""
        return self._inner.model_name

    def admit(self, calls: Sequence[PlannedCall]) -> list[AdmittedCall]:
        """Price and admit ``calls`` against the budget, or refuse them all before any is made.

        Raises:
            ValidationFailedError: See :meth:`OutOfRunBudget.admit`.
        """
        admitted = self._budget.admit(self._inner, "judge", calls)
        self._pending.extend(admitted)
        return admitted

    async def generate(
        self, *, system: str, user: str, response_format: dict[str, Any] | None = None
    ) -> CompletionResult:
        """Make one admitted call through the budget.

        Raises:
            ValueError: No admission for exactly this prompt is waiting — a call that would go unpriced.
        """
        call = PlannedCall(system=system, user=user, response_format=response_format)
        waiting = next((admitted for admitted in self._pending if admitted.call == call), None)
        if waiting is None:
            raise ValueError("a judge repeat call was sent without being admitted against its budget")
        self._pending.remove(waiting)
        result: CompletionResult = (await self._budget.generate(self._inner, waiting)).result
        return result

    async def aclose(self) -> None:
        """Release the host's client."""
        await self._inner.aclose()


@dataclass(frozen=True)
class _Planned:
    """One result to repeat: what its judge read, and the dims it holds a score on, in dimension order."""

    result: EvalResult
    inputs: ReproducibleJudgeInputs
    dims: list[str]


@dataclass(frozen=True)
class _Collected:
    """What a repeat of a run reads before it builds anything."""

    run: EvalRun
    judge_model: str
    planned: list[_Planned]
    skipped: list[JudgeRepeatSkip]


def _scored_dims(result: EvalResult) -> set[str]:
    """The dims ``result`` holds a judge score on — the only ones a repeat has a first score to agree with."""
    return {
        score.dim
        for score in (*result.rubric_scores, result.transcript_score, result.outcome_score)
        if score is not None
    }


def _collect(storage: EvalStorage, run_id: str, scope_id: str, result_ids: Sequence[str] | None) -> _Collected:
    """Load the run and every result to repeat, refusing what cannot be reproduced — blocking, so off the loop.

    Raises:
        NotFoundError: No run with that id, or the run's template does not load.
        ValidationFailedError: The run is still running, its judging apparatus was not recorded or cannot be
            sent today, ``result_ids`` names a result not in the run, or no result of the run can be repeated.
    """
    run = storage.load_eval_run(run_id, scope_id)
    if run is None:
        raise NotFoundError("run", run_id)
    if run.status in NON_TERMINAL_RUN_STATUSES:
        raise ValidationFailedError(f"run '{run_id}' is {run.status} — its results are still being written")
    # Refusals that hold for every result, raised once rather than named per result.
    judge_model = recorded_judge_pins(run, request_settings="today")
    judged_template(storage, run, scope_id)
    results = storage.query_eval_results_by_run(run.id, scope_id)
    if result_ids is not None:
        known = {result.id for result in results}
        if stray := sorted(set(result_ids) - known):
            raise ValidationFailedError(f"result(s) {stray} are not results of run '{run_id}' in this scope")
        wanted = set(result_ids)
        results = [result for result in results if result.id in wanted]
    planned: list[_Planned] = []
    skipped: list[JudgeRepeatSkip] = []
    for result in results:
        scored = _scored_dims(result)
        if not scored:
            skipped.append(JudgeRepeatSkip(result_id=result.id, reason="it holds no judge score to repeat"))
            continue
        try:
            inputs = reproducible_judge_inputs(
                storage, result, run, scope_id, request_settings="today", config_dims=scored
            )
        except (ValidationFailedError, NotFoundError) as refused:
            skipped.append(JudgeRepeatSkip(result_id=result.id, reason=str(refused)))
            continue
        dims = [dim for dim in inputs.dims if dim in scored]
        if not dims:
            skipped.append(JudgeRepeatSkip(result_id=result.id, reason="none of its scored dims is one its run judged"))
            continue
        planned.append(_Planned(result=result, inputs=inputs, dims=dims))
    if not planned:
        why = "; ".join(f"{skip.result_id}: {skip.reason}" for skip in skipped) or "it has no results"
        raise ValidationFailedError(f"no result of run '{run_id}' can be repeated — {why}")
    return _Collected(run=run, judge_model=judge_model, planned=planned, skipped=skipped)


@dataclass
class _Prepared:
    """A repeat built and priced: the judge service over budgeted clients, and each result's context and calls."""

    collected: _Collected
    budget: OutOfRunBudget
    service: JudgeService
    contexts: list[JudgeContext]
    #: Per budgeted client, every call it may make, each as many times as one dim can be attempted.
    calls: list[tuple[_BudgetedJudgeClient, list[PlannedCall]]]


async def _prepare(
    host: EvalHost,
    run_id: str,
    scope_id: str,
    *,
    out_of_run_cap_usd: float | None,
    result_ids: Sequence[str] | None,
) -> _Prepared:
    """Collect the run, build the judge over budgeted clients, and plan every call — admitting nothing yet.

    The caller owns ``service`` and enters it, which releases every client built here.
    """
    clients = host.completion_clients("a judge repeat")
    collected = await run_blocking(host.blocking_executor, _collect, host.storage, run_id, scope_id, result_ids)
    budget = OutOfRunBudget(
        store=host.storage,
        scope_id=scope_id,
        cap_usd=out_of_run_cap_usd,
        template_id=collected.run.template_id,
        run_id=collected.run.id,
        blocking_executor=host.blocking_executor,
    )
    inner = judge_clients_for_run(clients, collected.judge_model)

    def budgeted(model: str | None, temperature: float | None) -> _BudgetedJudgeClient:
        return _BudgetedJudgeClient(inner(model, temperature), budget)

    configs: dict[str, JudgeConfig] = {}
    for planned in collected.planned:
        configs.update(planned.inputs.configs)
    service = JudgeService(client_factory=budgeted, configs=configs, failure_describer=host.failure_describer)
    contexts: list[JudgeContext] = []
    by_client: dict[int, tuple[_BudgetedJudgeClient, list[PlannedCall]]] = {}
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
            client: _BudgetedJudgeClient = service.client_for(request)
            by_client.setdefault(id(client), (client, []))[1].extend([_planned_call(request)] * JUDGE_CALL_ATTEMPTS)
    return _Prepared(
        collected=collected, budget=budget, service=service, contexts=contexts, calls=list(by_client.values())
    )


def _planned_call(request: JudgeRequest) -> PlannedCall:
    """The call a judge request sends, as :func:`~threetears.evals.run.judge.run_judge_llm` sends it."""
    return PlannedCall(
        system=request.system_prompt, user=request.user_prompt, response_format=JSON_OBJECT_RESPONSE_FORMAT
    )


async def estimate_judge_repeat(
    host: EvalHost,
    run_id: str,
    scope_id: str,
    *,
    out_of_run_cap_usd: float | None,
    result_ids: Sequence[str] | None = None,
) -> JudgeRepeatEstimate:
    """Price repeating a run's judge scores against the cap it would be held to, and make no call.

    The same collection, the same calls and the same admission :func:`repeat_judge_scores` makes, so
    ``would_start`` is its answer.

    Args:
        host: The host: the run's store and the judge clients.
        run_id: The finished run.
        scope_id: The scope it lives in.
        out_of_run_cap_usd: The cap the repeat would be held to; ``None`` when the host enforces none.
        result_ids: The results to repeat; ``None`` for every result of the run.

    Returns:
        The estimate.

    Raises:
        NotFoundError: No run with that id, or a record it names does not load.
        ValidationFailedError: The run cannot be repeated (see :func:`repeat_judge_scores`).
        ValueError: The host supplies no completion clients.
    """
    prepared = await _prepare(host, run_id, scope_id, out_of_run_cap_usd=out_of_run_cap_usd, result_ids=result_ids)
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
    return JudgeRepeatEstimate(
        run_id=collected.run.id,
        judge_model=collected.judge_model,
        results=len(collected.planned),
        dims=sum(len(planned.dims) for planned in collected.planned),
        max_calls=len(ceilings),
        ceiling_usd=None if None in ceilings else math.fsum(c for c in ceilings if c is not None),
        cap_usd=out_of_run_cap_usd,
        would_start=refusal is None,
        refusal=refusal,
        skipped=collected.skipped,
    )


async def repeat_judge_scores(
    host: EvalHost,
    run_id: str,
    scope_id: str,
    *,
    out_of_run_cap_usd: float | None,
    result_ids: Sequence[str] | None = None,
) -> JudgeRepeatReport:
    """Ask a finished run's judge to score its results' scored dims again, and record each repeat on its result.

    Every call the repeat can make is priced and admitted against ``out_of_run_cap_usd`` before the first is
    sent, so a repeat the cap refuses has spent nothing. Results are repeated one after another, each one's
    repeat written as soon as its calls return, so a repeat cut off partway keeps what it paid for.

    Args:
        host: The host: the run's store, the judge clients — through the same run-pinned resolution the run
            scored through — and how a judge call that raised reads.
        run_id: The finished run.
        scope_id: The scope it lives in.
        out_of_run_cap_usd: The most the repeat's calls may be priced at together; ``None`` when the host
            enforces no out-of-run cap.
        result_ids: The results to repeat; ``None`` for every result of the run.

    Returns:
        What was repeated, what it cost, and what was left out.

    Raises:
        NotFoundError: No run with that id, or a record it names does not load.
        ValidationFailedError: The run is still running; its judging apparatus was not recorded or would be
            asked differently today; ``result_ids`` names a result outside it; no result can be repeated; or
            the calls cannot be priced under the enforced cap or are priced above it. All before any call.
        ValueError: The host supplies no completion clients.
    """
    prepared = await _prepare(host, run_id, scope_id, out_of_run_cap_usd=out_of_run_cap_usd, result_ids=result_ids)
    collected = prepared.collected
    repeated: list[str] = []
    unwritten: list[str] = []
    scored = unanswered = 0
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
            repeat = _repeat_of(planned.result, outcomes, judge_model=collected.judge_model)
            scored += sum(1 for entry in repeat.scores if entry.repeat is not None)
            unanswered += sum(1 for entry in repeat.scores if entry.repeat is None)
            if await run_blocking(host.blocking_executor, _record, host.storage, planned.result.id, scope_id, repeat):
                repeated.append(planned.result.id)
            else:
                unwritten.append(planned.result.id)
            if any(outcome.account_refused for _, outcome in outcomes):
                stopped = (
                    f"the account behind the judge refused a call while repeating result '{planned.result.id}', "
                    "so every later call would be refused too"
                )
                break
    spends = prepared.budget.recorded
    costs = [spend.cost_usd for spend in spends]
    report = JudgeRepeatReport(
        run_id=collected.run.id,
        judge_model=collected.judge_model,
        repeated=repeated,
        scores_repeated=scored,
        scores_unanswered=unanswered,
        skipped=collected.skipped,
        unwritten=unwritten,
        stopped=stopped,
        calls_made=len(spends),
        cost_usd=None if None in costs else math.fsum(c for c in costs if c is not None),
        cap_usd=out_of_run_cap_usd,
    )
    log.info(
        "eval.repeat_judge_scores run=%s repeated=%d scored=%d unanswered=%d skipped=%d unwritten=%d calls=%d cost=%s",
        report.run_id,
        len(report.repeated),
        report.scores_repeated,
        report.scores_unanswered,
        len(report.skipped),
        len(report.unwritten),
        report.calls_made,
        "unpriced" if report.cost_usd is None else f"${report.cost_usd:.6f}",
    )
    return report


def _repeat_of(result: EvalResult, outcomes: list[tuple[str, JudgeOutcome]], *, judge_model: str) -> JudgeRepeat:
    """The repeat to record: each dim's first score beside the judge's answer to it this time."""
    entries = []
    for dim, outcome in outcomes:
        first = result.judge_score(dim)
        assert first is not None  # only scored dims are planned, from this very result
        entries.append(
            RepeatedScore(
                dim=dim,
                scale=first.scale,
                first_score=first.score,
                first_served_model=first.served_model,
                repeat=outcome.score,
                error=outcome.error if outcome.score is None and outcome.cannot_tell is None else None,
                cannot_tell=outcome.cannot_tell,
            )
        )
    return JudgeRepeat(
        judge_model=judge_model,
        scores=entries,
        judge_config_ids={dim: outcome.config_id for dim, outcome in outcomes if outcome.config_id is not None},
    )


def _record(storage: EvalStorage, result_id: str, scope_id: str, repeat: JudgeRepeat) -> bool:
    """Append ``repeat`` to the stored result, re-reading and re-applying it when another writer got there first.

    Blocking, so it runs on the host's executor.

    Returns:
        True once the repeat is stored; False when the result is gone or every write was refused or failed —
        the calls are paid for and in the ledger either way, so this never raises.
    """
    for _ in range(_WRITE_ATTEMPTS):
        result, etag = storage.load_eval_result_with_etag(result_id, scope_id)
        if result is None:
            log.error(
                "eval.repeat_judge_scores result=%s was deleted while repeated; its repeat is not stored", result_id
            )
            return False
        updated = result.model_copy(update={"judge_repeats": [*result.judge_repeats, repeat]})
        try:
            storage.replace_eval_result(updated, if_match=etag)
        except ConflictError:
            continue
        except StorageError:
            log.exception("eval.repeat_judge_scores result=%s: its repeat could not be written", result_id)
            return False
        return True
    log.error(
        "eval.repeat_judge_scores result=%s changed under every one of %d writes; its repeat is not stored",
        result_id,
        _WRITE_ATTEMPTS,
    )
    return False


__all__ = [
    "JudgeRepeatEstimate",
    "JudgeRepeatReport",
    "JudgeRepeatSkip",
    "estimate_judge_repeat",
    "repeat_judge_scores",
]
