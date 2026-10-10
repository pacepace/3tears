"""Measure what temperature does to a judge: the same borderline evidence re-judged at the pinned temperature and at the provider's default.

Every judge call is requested at :data:`~threetears.evals.schema.models.DEFAULT_JUDGE_TEMPERATURE` unless a config
says otherwise (#633). The issue asked that policy to rest on a measurement, not an assertion: how far a judge's
scores of the SAME evidence vary across repeats at that temperature, against how far they vary at the provider's
default (temperature not sent), on the borderline cases temperature is expected to move. This module takes it, on a
run that has finished, over the machinery a judge repeat uses (:mod:`threetears.evals.run.judge_repeat`):

- **The same judge setup, two samplings.** Each result's judge input is rebuilt from what its run recorded
  (:func:`~threetears.evals.run.judge_repeat.collect_repeatable`, the judge repeat's own collection and refusals):
  the judge pin, the prompt per dim, the template, the case and the evidence the cell's kind rendered. Only the
  temperature differs: every call on the ``pinned`` side is requested at ``DEFAULT_JUDGE_TEMPERATURE``, every call on
  the ``provider_default`` side is sent none. A config that pins its own temperature is overridden on both sides —
  this compares the two settings — while its prompt and model are kept.
- **Borderline by default.** A scored dim of a result is borderline when its stored score sits inside the scale (2-4
  on 1-5), or when a recorded judge repeat or second judge of it answered differently (a "can't tell" included).
  ``selection="all"`` takes every scored dim instead.
- **Repeated at each setting, interleaved.** Each borderline dim is judged ``repeats`` times at each setting, the two
  settings alternating call round by call round, so a provider changing under the measurement moves both alike.
- **Spent here, read there.** This module asks and returns every answer
  (:class:`~threetears.evals.kernel.JudgeTemperatureAnswers`); the reading — per-case variance and self-agreement at
  each setting, side by side — is :func:`~threetears.evals.analysis.read_judge_temperatures`, which spends nothing.
  The ``judge_temperature`` operation and the ``judge-temperature`` command do both.
- **The temperature sent is checked, never assumed.** Each answer records the temperature its client reports sending
  (:func:`~threetears.evals.run.judge.sent_temperature`). A run whose every borderline score records that its model
  was sent none (a model that refuses a temperature) is refused before anything is spent: both sides would be the
  same sampling. An answer recorded at anything other than its side's setting — a client that dropped the
  temperature, or reports none — is left out of that side's figures by the reading, and the comparison is marked not
  comparable, with the reason.
- **Priced before it is paid for.** Every call, parse retries included, is priced on the client it will be made on
  and admitted against the host's out-of-run cap before the first is sent. Each call made is written to the
  out-of-run ledger under purpose ``judge``, stamped with the run, whether it returns or raises.
- **A measurement of the judge, never a change to the result.** Nothing is written to the results: the forced
  temperatures are not the run's judge, so recording them as repeats would split the run's own self-agreement. The
  returned answers, and the comparison read off them, are the record; keep them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.kernel.judge_temperature import (
    DEFAULT_TEMPERATURE_REPEATS,
    MIN_TEMPERATURE_REPEATS,
    TEMPERATURE_SETTINGS,
    JudgeTemperatureAnswers,
    TemperatureAnswer,
    TemperatureCaseAnswers,
    TemperatureSelection,
    TemperatureSetting,
    TemperatureSkip,
)
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel.out_of_run import OutOfRunBudget, PlannedCall
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.models import MODEL_DEFAULT_TEMPERATURE, SCALES, EvalResult
from threetears.evals.run.judge import JUDGE_CALL_ATTEMPTS
from threetears.evals.run.judge_repeat import (
    BudgetedJudgeClient,
    CollectedRepeat,
    PlannedRepeat,
    collect_repeatable,
    planned_judge_call,
)
from threetears.evals.run.judge_service import JudgeService, judge_clients_for_run
from threetears.evals.run.runner import build_judge_context, judge_dims, judge_requests
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.evals.kernel.host.eval_host import EvalHost
    from threetears.evals.kernel.storage import EvalStorage
    from threetears.evals.run.judge_service import JudgeContext, JudgeOutcome
    from threetears.evals.schema.models import JudgeConfig

log = get_logger(__name__)


class JudgeTemperatureEstimate(EvalBaseModel):
    """What comparing a run's judge at the two temperatures would be priced at, against its cap — no call made.

    Attributes:
        run_id: The run.
        judge_model: The run's judge pin.
        selection: Which scored dims would be re-judged.
        repeats: Calls per dim per setting.
        results: The results that would be re-judged.
        cases: The (result, dim) pairs that would be re-judged.
        max_calls: The most calls it can make: both settings, every repeat, each attempt and parse retry.
        ceiling_usd: The most those calls can cost together, as the clients price them; ``None`` when a client
            cannot say.
        cap_usd: The out-of-run cap it would be held to; ``None`` when the host enforces none.
        would_start: Whether every call would be admitted.
        refusal: Why it would not, when it would not.
        skipped: The run's results that would not be re-judged, each with why.
    """

    run_id: str
    judge_model: str
    selection: TemperatureSelection
    repeats: int
    results: int
    cases: int
    max_calls: int
    ceiling_usd: float | None
    cap_usd: float | None
    would_start: bool
    refusal: str | None = None
    skipped: list[TemperatureSkip]

    def render(self) -> str:
        """The estimate in a line: what would be re-judged, the most it costs, the cap, and whether it would start."""
        ceiling = "unpriceable" if self.ceiling_usd is None else f"${self.ceiling_usd:.4f}"
        cap = "none enforced" if self.cap_usd is None else f"${self.cap_usd:.2f}"
        verdict = "would start" if self.would_start else f"would be refused: {self.refusal}"
        return (
            f"estimate: judge temperature comparison on run {self.run_id} (judge {self.judge_model}) — "
            f"{self.cases} {self.selection} case(s) over {self.results} result(s), {self.repeats} repeat(s) at each "
            f"of 2 temperatures, at most {self.max_calls} call(s) priced at up to {ceiling}; out-of-run cap {cap}; "
            f"{verdict}."
        )


def _disagreed(result: EvalResult, dim: str, stored: int) -> bool:
    """Whether a recorded repeat or second judge of ``dim`` answered other than ``stored`` — a "can't tell" included."""
    for repeat in result.judge_repeats:
        for entry in repeat.scores:
            if entry.dim == dim and entry.first_score == stored:
                if entry.cannot_tell is not None or (entry.repeat is not None and entry.repeat.score != stored):
                    return True
    for second in result.judge_seconds:
        for score in second.scores:
            if score.dim == dim and score.first_score == stored:
                if score.cannot_tell is not None or (score.second is not None and score.second.score != stored):
                    return True
    return False


def borderline_dims(result: EvalResult) -> set[str]:
    """The scored dims of ``result`` that are borderline: the cases temperature is expected to move.

    A dim is borderline when its stored score sits strictly inside its scale (2, 3 or 4 on 1-5; never on pass/fail,
    whose two scores are both its ends), or when a judge repeat or a second judge recorded on the result answered it
    differently from the stored score, or could not tell. A pass/fail dim is therefore borderline only once something
    has disagreed on it: run a judge repeat first, or select every dim.

    Args:
        result: The result.

    Returns:
        The borderline dims, as their scores spell them.
    """
    chosen: set[str] = set()
    for score in result.judge_scores():
        low, high = SCALES[score.scale].scores
        if low < score.score < high or _disagreed(result, score.dim, score.score):
            chosen.add(score.dim)
    return chosen


def _select(
    collected: CollectedRepeat, selection: TemperatureSelection
) -> tuple[list[PlannedRepeat], list[TemperatureSkip]]:
    """Narrow each planned result to the dims ``selection`` takes, skipping a result left with none.

    Raises:
        ValidationFailedError: No result has a dim ``selection`` takes, or every one of their stored scores
            records that its model was sent no temperature.
    """
    planned: list[PlannedRepeat] = []
    skipped = [TemperatureSkip(result_id=skip.result_id, reason=skip.reason) for skip in collected.skipped]
    for one in collected.planned:
        if selection == "all":
            planned.append(one)
            continue
        borderline = borderline_dims(one.result)
        dims = [dim for dim in one.dims if dim in borderline]
        if not dims:
            skipped.append(
                TemperatureSkip(
                    result_id=one.result.id,
                    reason="no borderline dim: every stored score is at an end of its scale and nothing disagreed",
                )
            )
            continue
        planned.append(PlannedRepeat(result=one.result, inputs=one.inputs, dims=dims))
    run_id = collected.run.id
    if not planned:
        raise ValidationFailedError(
            f"no result of run '{run_id}' has a borderline dim to compare temperatures on — every stored score is "
            "at an end of its scale and no repeat or second judge disagreed with one. Select every dim "
            "(selection='all'), or run a judge repeat first so disagreement can mark a case borderline"
        )
    recorded = [one.result.judge_score(dim) for one in planned for dim in one.dims]
    if all(score is not None and score.judge_temperature == MODEL_DEFAULT_TEMPERATURE for score in recorded):
        raise ValidationFailedError(
            f"every score the comparison would re-judge on run '{run_id}' records that its judge was sent no "
            "temperature — a model that refuses one, so a call at the pinned temperature would be sent none too and "
            "both sides would be the same sampling. There is nothing to compare on this judge"
        )
    return planned, skipped


@dataclass
class _Prepared:
    """A comparison built and priced: a judge service per setting over budgeted clients, and each result's context."""

    run_id: str
    judge_model: str
    planned: list[PlannedRepeat]
    skipped: list[TemperatureSkip]
    budget: OutOfRunBudget
    services: dict[TemperatureSetting, JudgeService]
    contexts: list[JudgeContext]
    calls: list[tuple[BudgetedJudgeClient, list[PlannedCall]]]


async def _prepare(
    host: EvalHost,
    run_id: str,
    scope_id: str,
    *,
    out_of_run_cap_usd: float | None,
    result_ids: Sequence[str] | None,
    selection: TemperatureSelection,
    repeats: int,
) -> _Prepared:
    """Collect the run, build a judge per setting over budgeted clients, and plan every call — admitting nothing yet.

    The caller enters every service, which releases every client built here.

    Raises:
        ValidationFailedError: ``repeats`` is under :data:`MIN_TEMPERATURE_REPEATS`, or the run cannot be compared.
        ValueError: The host supplies no completion clients.
    """
    if repeats < MIN_TEMPERATURE_REPEATS:
        raise ValidationFailedError(
            f"repeats {repeats} is under {MIN_TEMPERATURE_REPEATS}: one answer per setting has no variance to compare"
        )
    clients = host.completion_clients("a judge temperature comparison")

    def collect(storage: EvalStorage) -> tuple[CollectedRepeat, list[PlannedRepeat], list[TemperatureSkip]]:
        collected = collect_repeatable(storage, run_id, scope_id, result_ids)
        return (collected, *_select(collected, selection))

    collected, planned, skipped = await run_blocking(host.blocking_executor, collect, host.storage)
    budget = OutOfRunBudget(
        store=host.storage,
        scope_id=scope_id,
        cap_usd=out_of_run_cap_usd,
        template_id=collected.run.template_id,
        run_id=collected.run.id,
        blocking_executor=host.blocking_executor,
    )
    inner = judge_clients_for_run(clients, collected.judge_model)
    configs: dict[str, JudgeConfig] = {}
    for one in planned:
        configs.update(one.inputs.configs)
    services: dict[TemperatureSetting, JudgeService] = {}
    for setting, requested, _ in TEMPERATURE_SETTINGS:

        def budgeted(
            model: str | None, _temperature: float | None, sent: float | None = requested
        ) -> BudgetedJudgeClient:
            # The setting's temperature, whatever the dim's config asks: this compares the two settings.
            return BudgetedJudgeClient(inner(model, sent), budget)

        services[setting] = JudgeService(
            client_factory=budgeted, configs=configs, failure_describer=host.failure_describer
        )
    contexts: list[JudgeContext] = []
    by_client: dict[int, tuple[BudgetedJudgeClient, list[PlannedCall]]] = {}
    for one in planned:
        context = build_judge_context(
            template=one.inputs.template,
            test_case=one.inputs.test_case,
            goal_outcomes=one.result.goal_state_outcomes,
            judged_artifact=one.inputs.judged_artifact,
            judge_evidence=one.inputs.judge_evidence,
        )
        contexts.append(context)
        for service in services.values():
            for request in judge_requests(
                template=one.inputs.template, judge_service=service, context=context, only=frozenset(one.dims)
            ):
                client: BudgetedJudgeClient = service.client_for(request)
                by_client.setdefault(id(client), (client, []))[1].extend(
                    [planned_judge_call(request)] * (JUDGE_CALL_ATTEMPTS * repeats)
                )
    return _Prepared(
        run_id=collected.run.id,
        judge_model=collected.judge_model,
        planned=planned,
        skipped=skipped,
        budget=budget,
        services=services,
        contexts=contexts,
        calls=list(by_client.values()),
    )


async def _close(services: dict[TemperatureSetting, JudgeService]) -> None:
    """Release every client the services minted."""
    for service in services.values():
        await service.aclose()


async def estimate_judge_temperature_comparison(
    host: EvalHost,
    run_id: str,
    scope_id: str,
    *,
    out_of_run_cap_usd: float | None,
    selection: TemperatureSelection = "borderline",
    repeats: int = DEFAULT_TEMPERATURE_REPEATS,
    result_ids: Sequence[str] | None = None,
) -> JudgeTemperatureEstimate:
    """Price comparing a run's judge at the two temperatures against the cap it would be held to, and make no call.

    The same collection, selection, calls and admission :func:`judge_at_two_temperatures` makes, so ``would_start``
    is its answer.

    Args:
        host: The host: the run's store and the judge clients.
        run_id: The finished run.
        scope_id: The scope it lives in.
        out_of_run_cap_usd: The cap it would be held to; ``None`` when the host enforces none.
        selection: ``borderline`` (the default) or ``all`` scored dims.
        repeats: Calls per dim per setting, at least :data:`MIN_TEMPERATURE_REPEATS`.
        result_ids: The results to draw from; ``None`` for every result of the run.

    Returns:
        The estimate.

    Raises:
        NotFoundError: No run with that id, or a record it names does not load.
        ValidationFailedError: The comparison cannot be made (see :func:`judge_at_two_temperatures`).
        ValueError: The host supplies no completion clients.
    """
    prepared = await _prepare(
        host,
        run_id,
        scope_id,
        out_of_run_cap_usd=out_of_run_cap_usd,
        result_ids=result_ids,
        selection=selection,
        repeats=repeats,
    )
    try:
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
    finally:
        await _close(prepared.services)
    return JudgeTemperatureEstimate(
        run_id=prepared.run_id,
        judge_model=prepared.judge_model,
        selection=selection,
        repeats=repeats,
        results=len(prepared.planned),
        cases=sum(len(one.dims) for one in prepared.planned),
        max_calls=len(ceilings),
        ceiling_usd=None if None in ceilings else math.fsum(c for c in ceilings if c is not None),
        cap_usd=out_of_run_cap_usd,
        would_start=refusal is None,
        refusal=refusal,
        skipped=prepared.skipped,
    )


async def judge_at_two_temperatures(
    host: EvalHost,
    run_id: str,
    scope_id: str,
    *,
    out_of_run_cap_usd: float | None,
    selection: TemperatureSelection = "borderline",
    repeats: int = DEFAULT_TEMPERATURE_REPEATS,
    result_ids: Sequence[str] | None = None,
) -> JudgeTemperatureAnswers:
    """Re-judge a finished run's borderline cases ``repeats`` times at each temperature, and return every answer.

    Every call is priced and admitted against ``out_of_run_cap_usd`` before the first is sent, so a comparison the
    cap refuses has spent nothing. Nothing is written to the results. Read the answers with
    :func:`~threetears.evals.analysis.read_judge_temperatures`.

    Args:
        host: The host: the run's store, its judge clients — through the run-pinned resolution the run scored
            through — and how a judge call that raised reads.
        run_id: The finished run.
        scope_id: The scope it lives in.
        out_of_run_cap_usd: The most the calls may be priced at together; ``None`` when the host enforces none.
        selection: ``borderline`` (the default: see :func:`borderline_dims`) or ``all`` scored dims.
        repeats: Calls per dim per setting, at least :data:`~threetears.evals.kernel.MIN_TEMPERATURE_REPEATS`.
        result_ids: The results to draw from; ``None`` for every result of the run.

    Returns:
        Per case and setting every answer in call order, the case count and the spend.

    Raises:
        NotFoundError: No run with that id, or a record it names does not load.
        ValidationFailedError: The run is still running; its judging cannot be reproduced; ``result_ids`` names a
            result outside it; ``repeats`` is under the floor; no result has a dim ``selection`` takes; every score to
            re-judge records a model that refuses a temperature; or the calls cannot be priced under the enforced cap
            or are priced above it. All before any call.
        ValueError: The host supplies no completion clients.
    """
    prepared = await _prepare(
        host,
        run_id,
        scope_id,
        out_of_run_cap_usd=out_of_run_cap_usd,
        result_ids=result_ids,
        selection=selection,
        repeats=repeats,
    )
    answered: dict[tuple[str, str, TemperatureSetting], list[JudgeOutcome]] = {}
    stopped: str | None = None
    try:
        for client, calls in prepared.calls:
            client.admit(calls)
        for one, context in zip(prepared.planned, prepared.contexts, strict=True):
            for _ in range(repeats):
                for setting, _, _ in TEMPERATURE_SETTINGS:
                    outcomes = await judge_dims(
                        template=one.inputs.template,
                        judge_service=prepared.services[setting],
                        context=context,
                        only=frozenset(one.dims),
                    )
                    for dim, outcome in outcomes:
                        answered.setdefault((one.result.id, dim, setting), []).append(outcome)
                    if any(outcome.account_refused for _, outcome in outcomes):
                        stopped = (
                            f"the account behind the judge refused a call while re-judging result '{one.result.id}', "
                            "so every later call would be refused too"
                        )
                        break
                if stopped:
                    break
            if stopped:
                break
    finally:
        await _close(prepared.services)
    cases: list[TemperatureCaseAnswers] = []
    for one in prepared.planned:
        for dim in one.dims:
            stored = one.result.judge_score(dim)
            assert stored is not None  # only scored dims are planned, from this very result
            for setting, _, _ in TEMPERATURE_SETTINGS:
                if (outcomes_of := answered.get((one.result.id, dim, setting))) is None:
                    continue
                cases.append(
                    TemperatureCaseAnswers(
                        result_id=one.result.id,
                        rubric_dim=dim,
                        scale=stored.scale,
                        stored_score=stored.score,
                        setting=setting,
                        answers=[
                            TemperatureAnswer(
                                score=outcome.score,
                                config_id=outcome.config_id,
                                cannot_tell=outcome.cannot_tell,
                                error=(outcome.error or "no answer")
                                if outcome.score is None and outcome.cannot_tell is None
                                else None,
                            )
                            for outcome in outcomes_of
                        ],
                    )
                )
    spends = prepared.budget.recorded
    costs = [spend.cost_usd for spend in spends]
    answers = JudgeTemperatureAnswers(
        run_id=prepared.run_id,
        judge_model=prepared.judge_model,
        selection=selection,
        repeats=repeats,
        results=len(prepared.planned),
        cases=sum(len(one.dims) for one in prepared.planned),
        answers=cases,
        skipped=prepared.skipped,
        stopped=stopped,
        calls_made=len(spends),
        cost_usd=None if None in costs else math.fsum(c for c in costs if c is not None),
        cap_usd=out_of_run_cap_usd,
    )
    log.info(
        "eval.judge_temperature run=%s cases=%d repeats=%d calls=%d cost=%s",
        answers.run_id,
        answers.cases,
        repeats,
        answers.calls_made,
        "unpriced" if answers.cost_usd is None else f"${answers.cost_usd:.6f}",
    )
    return answers


__all__ = [
    "JudgeTemperatureEstimate",
    "borderline_dims",
    "estimate_judge_temperature_comparison",
    "judge_at_two_temperatures",
]
