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
- **Read by the code every self-agreement is read by.** Per setting, each case's answers pair against its first
  answer at that setting and go through :func:`~threetears.evals.analysis.judge_self_agreement` — exact agreement and
  kappa, a "can't tell" a disagreement — and each case's score variance across its repeats is reported beside it.
- **The temperature sent is checked, never assumed.** Each answer records the temperature its client reports sending
  (:func:`~threetears.evals.run.judge.sent_temperature`). A run whose every borderline score records that its model
  was sent none (a model that refuses a temperature) is refused before anything is spent: both sides would be the
  same sampling. An answer recorded at anything other than its side's setting — a client that dropped the
  temperature, or reports none — is left out of that side's variance and counted, and the comparison is then marked
  not comparable, with the reason.
- **Priced before it is paid for.** Every call, parse retries included, is priced on the client it will be made on
  and admitted against the host's out-of-run cap before the first is sent. Each call made is written to the
  out-of-run ledger under purpose ``judge``, stamped with the run, whether it returns or raises.
- **A measurement of the judge, never a change to the result.** Nothing is written to the results: the forced
  temperatures are not the run's judge, so recording them as repeats would split the run's own self-agreement. The
  returned :class:`JudgeTemperatureComparison` is the record; keep it.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from threetears.evals.analysis.agreement import JudgeSelfAgreement, judge_self_agreement
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel.out_of_run import OutOfRunBudget, PlannedCall
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.models import (
    DEFAULT_JUDGE_TEMPERATURE,
    MODEL_DEFAULT_TEMPERATURE,
    SCALES,
    EvalResult,
    JudgeRepeat,
    JudgeTemperature,
    RepeatedScore,
    RubricScale,
)
from threetears.evals.run.judge import JUDGE_CALL_ATTEMPTS
from threetears.evals.run.judge_repeat import (
    BudgetedJudgeClient,
    CollectedRepeat,
    JudgeRepeatSkip,
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

#: The two samplings compared. ``pinned``: every call requested at ``DEFAULT_JUDGE_TEMPERATURE`` (the policy).
#: ``provider_default``: every call sent no temperature, so the provider's own default applies.
TemperatureSetting = Literal["pinned", "provider_default"]

#: The settings in the order a call round makes them, and each one's request and the record its answers must carry.
_SETTINGS: tuple[tuple[TemperatureSetting, float | None, JudgeTemperature], ...] = (
    ("pinned", DEFAULT_JUDGE_TEMPERATURE, DEFAULT_JUDGE_TEMPERATURE),
    ("provider_default", None, MODEL_DEFAULT_TEMPERATURE),
)

#: Which scored dims are re-judged: ``borderline`` (stored score inside its scale, or a recorded repeat or second
#: judge disagreed on it — :func:`borderline_dims`) or ``all``.
TemperatureSelection = Literal["borderline", "all"]

#: How many times each dim is judged at each setting when the caller names no number.
DEFAULT_TEMPERATURE_REPEATS = 5

#: The fewest repeats per setting: one answer has no variance and nothing to agree with.
MIN_TEMPERATURE_REPEATS = 2


class TemperatureCase(EvalBaseModel):
    """One result's dimension, judged ``repeats`` times at one setting: its answers and their spread."""

    result_id: str
    rubric_dim: str
    scale: RubricScale
    stored_score: int = Field(description="The score the run's judge stored, which made the case borderline or not.")
    scores: list[int] = Field(description="The scores answered at this setting's temperature, in call order.")
    cannot_tell: int = Field(ge=0, description="Answers that said the judge could not tell.")
    failed: int = Field(ge=0, description="Calls that failed — an infrastructure fault, saying nothing of the judge.")
    off_setting: int = Field(
        ge=0, description="Scores recorded at a temperature other than this setting's, left out of `scores`."
    )
    variance: float | None = Field(
        description="The sample variance of `scores`; None under two scores. 0 = the judge gave one score every time."
    )
    stable: bool | None = Field(
        description=(
            'Whether every answer was the same, a "can\'t tell" counted as an answer of its own; None under two answers.'
        )
    )


class TemperatureSide(EvalBaseModel):
    """One dimension at one setting: how much its cases' scores moved across repeats, and the judge's self-agreement."""

    cases: int = Field(ge=0, description="Cases with at least two answers at this setting.")
    mean_variance: float | None = Field(
        description="The mean over cases of each case's score variance; None when no case has two scores."
    )
    max_variance: float | None = Field(description="The largest case variance; None when no case has two scores.")
    unstable_cases: int = Field(ge=0, description="Cases whose answers were not all the same.")
    exact_agreement: float | None = Field(
        description=(
            "The share of repeats answering what the case's first answer at this setting did, as "
            "`judge_self_agreement` reads it; None when it read no pair, or split the dim across judges."
        )
    )
    kappa: float | None = Field(description="Cohen's kappa of the same pairs; None when undefined or not read.")
    weighted_kappa: float | None = Field(description="Quadratic-weighted kappa on 1-5; None on pass/fail or undefined.")


class TemperatureDimension(EvalBaseModel):
    """One dimension, the two settings side by side."""

    rubric_dim: str
    scale: RubricScale
    pinned: TemperatureSide
    provider_default: TemperatureSide


class TemperatureSettingRead(EvalBaseModel):
    """Everything one setting's answers say, case by case."""

    setting: TemperatureSetting
    requested: float | None = Field(description="The temperature each call was requested at; None = sent none.")
    recorded: list[str] = Field(
        description=(
            "The temperatures the answers recorded being sent at, sorted, as text: a number, 'model_default' (sent "
            "none), or 'unrecorded' (the client reports nothing)."
        )
    )
    off_setting: int = Field(ge=0, description="Scores recorded at a temperature other than `requested`.")
    self_agreement: JudgeSelfAgreement = Field(
        description="Each case's later answers paired with its first at this setting, read by judge_self_agreement."
    )
    cases: list[TemperatureCase]


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
    skipped: list[JudgeRepeatSkip]

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


class JudgeTemperatureComparison(EvalBaseModel):
    """The measurement: a run's borderline cases re-judged at the pinned temperature and at the provider's default.

    Attributes:
        run_id: The run.
        judge_model: The run's judge pin.
        selection: Which scored dims were re-judged.
        repeats: Calls per dim per setting.
        results: The results re-judged.
        cases: The (result, dim) pairs re-judged.
        dimensions: Per dimension, the two settings side by side.
        settings: Per setting, every case and the full self-agreement read.
        comparable: Whether every score on each side was recorded at that side's temperature. False means the
            figures do not compare the two settings as named, and ``incomparable`` says why.
        incomparable: Why not, when not.
        skipped: The run's results not re-judged, each with why — decided before anything was spent.
        stopped: Why the comparison stopped before its last result, when it did.
        calls_made: Judge calls made, as the ledger recorded them under purpose ``judge``.
        cost_usd: What they cost together; ``None`` when any went unpriced.
        cap_usd: The out-of-run cap they were admitted under; ``None`` when the host enforces none.
    """

    run_id: str
    judge_model: str
    selection: TemperatureSelection
    repeats: int
    results: int
    cases: int
    dimensions: list[TemperatureDimension]
    settings: list[TemperatureSettingRead]
    comparable: bool
    incomparable: str | None = None
    skipped: list[JudgeRepeatSkip]
    stopped: str | None = None
    calls_made: int
    cost_usd: float | None
    cap_usd: float | None

    def render(self) -> str:
        """The comparison as text: what was asked and spent, then each dimension's two settings side by side."""
        cost = "unpriced" if self.cost_usd is None else f"${self.cost_usd:.4f}"
        requested = {read.setting: read for read in self.settings}
        pinned = requested["pinned"].requested
        lines = [
            f"judge temperature comparison on run {self.run_id} (judge {self.judge_model}): {self.cases} "
            f"{self.selection} case(s) over {self.results} result(s), {self.repeats} repeat(s) at temperature "
            f"{pinned:g} and at the provider default; {self.calls_made} call(s), {cost} — measurement cost, "
            "ledgered under judge, never the candidate's",
        ]
        if not self.comparable:
            lines.append(f"NOT COMPARABLE: {self.incomparable}")
        if self.stopped:
            lines.append(f"stopped: {self.stopped}")
        lines += [f"skipped {skip.result_id}: {skip.reason}" for skip in self.skipped]
        for read in self.settings:
            lines.append(f"{read.setting}: answers recorded at {', '.join(read.recorded) or 'nothing'}")
        lines.append(
            "per dimension — cases, mean / max score variance across repeats, unstable cases, exact agreement, kappa"
        )
        for row in self.dimensions:
            lines.append(f"- {row.rubric_dim} ({row.scale})")
            lines.append(f"    temperature {pinned:g}:      {_side_text(row.pinned)}")
            lines.append(f"    provider default:   {_side_text(row.provider_default)}")
        return "\n".join(lines)


def _figure(value: float | None, spec: str = ".3g") -> str:
    """A figure, or ``n/a`` where there is none."""
    return "n/a" if value is None else format(value, spec)


def _side_text(side: TemperatureSide) -> str:
    """One setting of one dimension, in a line."""
    agreement = "n/a" if side.exact_agreement is None else f"{side.exact_agreement:.0%}"
    kappa = side.weighted_kappa if side.weighted_kappa is not None else side.kappa
    return (
        f"{side.cases} case(s), variance {_figure(side.mean_variance)} / {_figure(side.max_variance)}, "
        f"{side.unstable_cases} unstable, exact agreement {agreement}, kappa {_figure(kappa)}"
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
) -> tuple[list[PlannedRepeat], list[JudgeRepeatSkip]]:
    """Narrow each planned result to the dims ``selection`` takes, skipping a result left with none.

    Raises:
        ValidationFailedError: No result has a dim ``selection`` takes, or every one of their stored scores
            records that its model was sent no temperature.
    """
    planned: list[PlannedRepeat] = []
    skipped = list(collected.skipped)
    for one in collected.planned:
        if selection == "all":
            planned.append(one)
            continue
        borderline = borderline_dims(one.result)
        dims = [dim for dim in one.dims if dim in borderline]
        if not dims:
            skipped.append(
                JudgeRepeatSkip(
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
    skipped: list[JudgeRepeatSkip]
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

    def collect(storage: EvalStorage) -> tuple[CollectedRepeat, list[PlannedRepeat], list[JudgeRepeatSkip]]:
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
    for setting, requested, _ in _SETTINGS:

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

    The same collection, selection, calls and admission :func:`compare_judge_temperatures` makes, so ``would_start``
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
        ValidationFailedError: The comparison cannot be made (see :func:`compare_judge_temperatures`).
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


async def compare_judge_temperatures(
    host: EvalHost,
    run_id: str,
    scope_id: str,
    *,
    out_of_run_cap_usd: float | None,
    selection: TemperatureSelection = "borderline",
    repeats: int = DEFAULT_TEMPERATURE_REPEATS,
    result_ids: Sequence[str] | None = None,
) -> JudgeTemperatureComparison:
    """Re-judge a finished run's borderline cases ``repeats`` times at each temperature, and read the two side by side.

    Every call is priced and admitted against ``out_of_run_cap_usd`` before the first is sent, so a comparison the
    cap refuses has spent nothing. Nothing is written to the results; the returned comparison is the record.

    Args:
        host: The host: the run's store, its judge clients — through the run-pinned resolution the run scored
            through — and how a judge call that raised reads.
        run_id: The finished run.
        scope_id: The scope it lives in.
        out_of_run_cap_usd: The most the calls may be priced at together; ``None`` when the host enforces none.
        selection: ``borderline`` (the default: see :func:`borderline_dims`) or ``all`` scored dims.
        repeats: Calls per dim per setting, at least :data:`MIN_TEMPERATURE_REPEATS`.
        result_ids: The results to draw from; ``None`` for every result of the run.

    Returns:
        Per dimension, score variance across repeats and self-agreement at each setting, the case count and the spend.

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
    answers: dict[TemperatureSetting, dict[tuple[str, str], list[JudgeOutcome]]] = {
        setting: {} for setting, _, _ in _SETTINGS
    }
    stopped: str | None = None
    try:
        for client, calls in prepared.calls:
            client.admit(calls)
        for one, context in zip(prepared.planned, prepared.contexts, strict=True):
            for _ in range(repeats):
                for setting, _, _ in _SETTINGS:
                    outcomes = await judge_dims(
                        template=one.inputs.template,
                        judge_service=prepared.services[setting],
                        context=context,
                        only=frozenset(one.dims),
                    )
                    for dim, outcome in outcomes:
                        answers[setting].setdefault((one.result.id, dim), []).append(outcome)
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
    spends = prepared.budget.recorded
    costs = [spend.cost_usd for spend in spends]
    comparison = _read(prepared, answers, selection=selection, repeats=repeats, stopped=stopped).model_copy(
        update={
            "calls_made": len(spends),
            "cost_usd": None if None in costs else math.fsum(c for c in costs if c is not None),
            "cap_usd": out_of_run_cap_usd,
        }
    )
    log.info(
        "eval.judge_temperature run=%s cases=%d repeats=%d comparable=%s calls=%d cost=%s",
        comparison.run_id,
        comparison.cases,
        repeats,
        comparison.comparable,
        comparison.calls_made,
        "unpriced" if comparison.cost_usd is None else f"${comparison.cost_usd:.6f}",
    )
    return comparison


def _recorded_text(temperature: JudgeTemperature | None) -> str:
    """A recorded temperature as the report spells it."""
    if temperature is None:
        return "unrecorded"
    return temperature if isinstance(temperature, str) else f"{temperature:g}"


def _case(
    result: EvalResult, dim: str, outcomes: list[JudgeOutcome], expected: JudgeTemperature
) -> tuple[TemperatureCase, set[str]]:
    """One case's answers at one setting, and the temperatures they recorded."""
    stored = result.judge_score(dim)
    assert stored is not None  # only scored dims are planned, from this very result
    scores: list[int] = []
    answers: list[int | None] = []  # None = "can't tell", an answer of its own
    recorded: set[str] = set()
    cannot_tell = failed = off = 0
    for outcome in outcomes:
        if outcome.score is not None:
            recorded.add(_recorded_text(outcome.score.judge_temperature))
            if outcome.score.judge_temperature != expected:
                off += 1
                continue
            scores.append(outcome.score.score)
            answers.append(outcome.score.score)
        elif outcome.cannot_tell is not None:
            cannot_tell += 1
            answers.append(None)
        else:
            failed += 1
    case = TemperatureCase(
        result_id=result.id,
        rubric_dim=dim,
        scale=stored.scale,
        stored_score=stored.score,
        scores=scores,
        cannot_tell=cannot_tell,
        failed=failed,
        off_setting=off,
        variance=statistics.variance(scores) if len(scores) >= 2 else None,
        stable=len(set(answers)) == 1 if len(answers) >= 2 else None,
    )
    return case, recorded


def _paired(result: EvalResult, by_dim: dict[str, list[JudgeOutcome]], judge_model: str) -> EvalResult | None:
    """``result`` carrying one setting's answers as repeats of its first scored answer there, for judge_self_agreement.

    Each dim's anchor is its first answer that is a score; every later answer is recorded as a repeat of it, so the
    rounds are ``repeat 1`` (the second answer) onward. An in-memory copy, never stored. None when no dim has an anchor.
    """
    repeats: list[JudgeRepeat] = []
    for dim, outcomes in by_dim.items():
        anchor_at = next((index for index, outcome in enumerate(outcomes) if outcome.score is not None), None)
        if anchor_at is None:
            continue
        anchor = outcomes[anchor_at]
        assert anchor.score is not None
        for outcome in outcomes[anchor_at + 1 :]:
            entry = RepeatedScore(
                dim=dim,
                scale=anchor.score.scale,
                first_score=anchor.score.score,
                first_served_model=anchor.score.served_model,
                first_judge_config_id=anchor.config_id,
                first_judge_temperature=anchor.score.judge_temperature,
                repeat=outcome.score,
                error=(outcome.error or "no answer") if outcome.score is None and outcome.cannot_tell is None else None,
                cannot_tell=outcome.cannot_tell,
            )
            repeats.append(
                JudgeRepeat(
                    judge_model=judge_model,
                    scores=[entry],
                    judge_config_ids={dim: outcome.config_id} if outcome.config_id is not None else {},
                )
            )
    return result.model_copy(update={"judge_repeats": repeats}) if repeats else None


def _side(cases: list[TemperatureCase], agreement: JudgeSelfAgreement, dim: str) -> TemperatureSide:
    """One dimension at one setting, from its cases and that setting's self-agreement."""
    answered = [case for case in cases if case.stable is not None]
    variances = [case.variance for case in cases if case.variance is not None]
    groups = [group for group in agreement.dimensions if group.rubric_dim == dim]
    # One judge (model, config, temperature) answered every pair, or the figures would pool two judges: none then.
    group = groups[0] if len(groups) == 1 else None
    return TemperatureSide(
        cases=len(answered),
        mean_variance=statistics.fmean(variances) if variances else None,
        max_variance=max(variances) if variances else None,
        unstable_cases=sum(1 for case in answered if not case.stable),
        exact_agreement=None if group is None else group.exact_agreement,
        kappa=None if group is None else group.kappa,
        weighted_kappa=None if group is None else group.weighted_kappa,
    )


def _read(
    prepared: _Prepared,
    answers: dict[TemperatureSetting, dict[tuple[str, str], list[JudgeOutcome]]],
    *,
    selection: TemperatureSelection,
    repeats: int,
    stopped: str | None,
) -> JudgeTemperatureComparison:
    """Every answer read, per setting and per dimension; the spend is filled in by the caller."""
    reads: list[TemperatureSettingRead] = []
    cases_by: dict[TemperatureSetting, list[TemperatureCase]] = {}
    agreement_by: dict[TemperatureSetting, JudgeSelfAgreement] = {}
    for setting, requested, expected in _SETTINGS:
        cases: list[TemperatureCase] = []
        recorded: set[str] = set()
        paired: list[EvalResult] = []
        for one in prepared.planned:
            by_dim = {
                dim: answers[setting][(one.result.id, dim)]
                for dim in one.dims
                if (one.result.id, dim) in answers[setting]
            }
            for dim, outcomes in by_dim.items():
                case, seen = _case(one.result, dim, outcomes, expected)
                cases.append(case)
                recorded |= seen
            if (copy := _paired(one.result, by_dim, prepared.judge_model)) is not None:
                paired.append(copy)
        agreement = judge_self_agreement(paired)
        cases_by[setting], agreement_by[setting] = cases, agreement
        reads.append(
            TemperatureSettingRead(
                setting=setting,
                requested=requested,
                recorded=sorted(recorded),
                off_setting=sum(case.off_setting for case in cases),
                self_agreement=agreement,
                cases=cases,
            )
        )
    dims: dict[str, RubricScale] = {}
    for one in prepared.planned:
        for dim in one.dims:
            stored = one.result.judge_score(dim)
            assert stored is not None
            dims.setdefault(dim, stored.scale)
    dimensions = [
        TemperatureDimension(
            rubric_dim=dim,
            scale=scale,
            pinned=_side([c for c in cases_by["pinned"] if c.rubric_dim == dim], agreement_by["pinned"], dim),
            provider_default=_side(
                [c for c in cases_by["provider_default"] if c.rubric_dim == dim], agreement_by["provider_default"], dim
            ),
        )
        for dim, scale in sorted(dims.items())
    ]
    problems = [
        f"{read.off_setting} score(s) at the {read.setting} setting were recorded at "
        f"{', '.join(sorted(set(read.recorded) - {_recorded_text(expected)})) or 'another temperature'}, not "
        f"{_recorded_text(expected)}"
        for read, (_, _, expected) in zip(reads, _SETTINGS, strict=True)
        if read.off_setting
    ]
    incomparable = None
    if problems:
        incomparable = (
            "; ".join(problems)
            + " — the judge's client did not send (or does not report sending) the temperature each side names, so "
            "those scores are left out and the two sides do not compare the settings as named"
        )
    return JudgeTemperatureComparison(
        run_id=prepared.run_id,
        judge_model=prepared.judge_model,
        selection=selection,
        repeats=repeats,
        results=len(prepared.planned),
        cases=sum(len(one.dims) for one in prepared.planned),
        dimensions=dimensions,
        settings=reads,
        comparable=incomparable is None,
        incomparable=incomparable,
        skipped=prepared.skipped,
        stopped=stopped,
        calls_made=0,
        cost_usd=None,
        cap_usd=None,
    )


__all__ = [
    "DEFAULT_TEMPERATURE_REPEATS",
    "MIN_TEMPERATURE_REPEATS",
    "JudgeTemperatureComparison",
    "JudgeTemperatureEstimate",
    "TemperatureCase",
    "TemperatureDimension",
    "TemperatureSelection",
    "TemperatureSetting",
    "TemperatureSettingRead",
    "TemperatureSide",
    "borderline_dims",
    "compare_judge_temperatures",
    "estimate_judge_temperature_comparison",
]
