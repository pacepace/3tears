"""The answer call layer's own jobs: budget, pacing, dispositions, failure mapping, spend (§10.3).

Written against a deliberately thin provider double, as ``test_call.py`` is,
because the call layer's guarantees have to hold for an adapter that gets
them wrong -- and once over the real Brave adapter and its recording.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from threetears.search.adapters.brave_answers import BRAVE_ANSWERS_CAPABILITIES, BraveAnswersAdapter
from threetears.search.answer import DEFAULT_ANSWER_TIMEOUT_SECONDS, answer
from threetears.search.call import PACING_BURST_SCOPE
from threetears.search.contracts import (
    AnswerCapabilities,
    AnswerProvider,
    AnswerRequest,
    BudgetDecision,
    Criterion,
    CriterionDisposition,
    GroundedAnswer,
    LocalCapExceeded,
    QuotaExhausted,
    RateLimitDecision,
    RateLimited,
    Spend,
    TransportFailed,
)
from threetears.search.testing import FakeBudgetPort, FakeRateLimiterPort
from packages.search.tests.recorded_streams import RecordedStream, RecordedStreamingTransport
from packages.search.tests.brave_answers_payloads import SINGLE, SINGLE_TOTAL_COST
from packages.search.tests.search_instances import GROUNDED_ANSWER

ESTIMATE = Spend(money=Decimal("0.004"), calls=1, provider_units=Decimal(1), provider_unit="brave:searches")


class _RecordingAnswerProvider:
    """An answers provider that records what the call layer sent it and answers as scripted.

    Not a ``Fake*``: it is the subject's counterpart, satisfying the seam
    structurally, and what it was handed is what these tests assert.
    """

    def __init__(self, *, result: GroundedAnswer | None = None, failure: BaseException | None = None) -> None:
        """Configure the double.

        :param result: what to answer with
        :ptype result: GroundedAnswer | None
        :param failure: what to raise instead
        :ptype failure: BaseException | None
        """
        self._result = result if result is not None else GROUNDED_ANSWER
        self._failure = failure
        self.requests: list[AnswerRequest] = []
        self.timeouts: list[float | None] = []

    @property
    def provider(self) -> str:
        """The product name.

        :return: ``brave``
        :rtype: str
        """
        return "brave"

    @property
    def provider_instance(self) -> str:
        """The configured deployment.

        :return: ``brave-live``
        :rtype: str
        """
        return "brave-live"

    @property
    def capabilities(self) -> AnswerCapabilities:
        """Brave's declaration.

        :return: the registered declaration
        :rtype: AnswerCapabilities
        """
        return BRAVE_ANSWERS_CAPABILITIES

    def estimate(self, request: AnswerRequest) -> Spend:
        """A fixed, priced estimate.

        :param request: ignored
        :ptype request: AnswerRequest
        :return: :data:`ESTIMATE`
        :rtype: Spend
        """
        return ESTIMATE

    async def answer(self, request: AnswerRequest, *, timeout_seconds: float | None = None) -> GroundedAnswer:
        """Record the call, then answer or fail as scripted.

        :param request: what the call layer sent
        :ptype request: AnswerRequest
        :param timeout_seconds: the bound it handed down
        :ptype timeout_seconds: float | None
        :return: the scripted answer
        :rtype: GroundedAnswer
        """
        self.requests.append(request)
        self.timeouts.append(timeout_seconds)
        if self._failure is not None:
            raise self._failure
        return self._result


class _Clock:
    """A monotonic clock a test advances by hand."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        self.now += 0.5
        return self.now


async def test_the_double_satisfies_the_seam() -> None:
    assert isinstance(_RecordingAnswerProvider(), AnswerProvider)


async def test_a_caller_naming_no_bound_gets_the_answer_default_not_the_search_one() -> None:
    provider = _RecordingAnswerProvider()
    await answer(AnswerRequest(question="q"), provider=provider)
    assert provider.timeouts == [DEFAULT_ANSWER_TIMEOUT_SECONDS]
    assert DEFAULT_ANSWER_TIMEOUT_SECONDS >= BRAVE_ANSWERS_CAPABILITIES.minimum_timeout_seconds  # type: ignore[operator]


async def test_the_budget_is_checked_with_the_provider_s_estimate_before_and_debited_after() -> None:
    """AR-D1, AR-C3: check with the provider's own estimate, record exactly what the caller is told."""
    budget = FakeBudgetPort()
    provider = _RecordingAnswerProvider()
    result = await answer(
        AnswerRequest(question="q", budget_scope_tags=("run:1",)), provider=provider, budget=budget, clock=_Clock()
    )

    assert budget.checks == [(ESTIMATE, ("run:1",))]
    assert budget.records == [(result.spend, ("run:1",))]
    assert result.spend.wall_clock_seconds == pytest.approx(0.5), "the call layer's own wall-clock replaces it"


async def test_a_budget_refusal_is_a_local_cap_and_the_provider_is_never_asked() -> None:
    """AR-D2: a local refusal is LocalCapExceeded, never the provider's QuotaExhausted."""
    consumed = Spend(money=Decimal("9.99"), calls=40)
    budget = FakeBudgetPort(BudgetDecision(allowed=False, scope="run:1", consumed=consumed, reason="spent"))
    limiter = FakeRateLimiterPort()
    provider = _RecordingAnswerProvider()
    with pytest.raises(LocalCapExceeded) as caught:
        await answer(AnswerRequest(question="q"), provider=provider, budget=budget, limiter=limiter)

    assert caught.value.scope == "run:1"
    assert caught.value.spend == consumed
    assert not isinstance(caught.value, QuotaExhausted)
    assert provider.requests == [] and limiter.acquisitions == [] and budget.records == []


async def test_pacing_keys_on_instance_and_egress_and_spends_the_caller_s_bound() -> None:
    limiter = FakeRateLimiterPort()
    provider = _RecordingAnswerProvider()
    await answer(
        AnswerRequest(question="q"),
        provider=provider,
        limiter=limiter,
        egress="warp",
        timeout_seconds=40.0,
        clock=_Clock(),
    )

    ((instance, egress, tokens, max_wait),) = limiter.acquisitions
    assert (instance, egress, tokens, max_wait) == ("brave-live", "warp", 1.0, 40.0)
    assert provider.timeouts == [pytest.approx(39.5)], "what pacing waited is taken from the provider's bound"


async def test_a_pacing_denial_is_rate_limited_and_the_provider_is_never_asked() -> None:
    limiter = FakeRateLimiterPort(RateLimitDecision(acquired=False, retry_after_seconds=0.5))
    provider = _RecordingAnswerProvider()
    with pytest.raises(RateLimited):
        await answer(AnswerRequest(question="q"), provider=provider, limiter=limiter)
    assert provider.requests == []


async def test_a_never_grantable_pacing_ask_is_the_burst_cap() -> None:
    class _Refusing(FakeRateLimiterPort):
        async def acquire(
            self, *, provider_instance: str, egress: str, tokens: float = 1.0, max_wait_seconds: float = 0.0
        ) -> RateLimitDecision:
            raise ValueError("burst below one token")

    with pytest.raises(LocalCapExceeded) as caught:
        await answer(AnswerRequest(question="q"), provider=_RecordingAnswerProvider(), limiter=_Refusing())
    assert caught.value.scope == PACING_BURST_SCOPE


async def test_a_typed_failure_passes_through_and_its_spend_is_debited() -> None:
    """AR-E1: a broken stream was billed, so its spend reaches the budget."""
    billed = Spend(calls=1, bytes_transferred=120)
    budget = FakeBudgetPort()
    provider = _RecordingAnswerProvider(failure=TransportFailed("broke mid-body", spend=billed))
    with pytest.raises(TransportFailed) as caught:
        await answer(AnswerRequest(question="q"), provider=provider, budget=budget, clock=_Clock())

    assert caught.value.spend.calls == 1
    assert caught.value.spend.wall_clock_seconds == pytest.approx(0.5)
    assert budget.records == [(caught.value.spend, ())]


async def test_an_untyped_adapter_escape_becomes_a_typed_failure_and_is_debited_honestly() -> None:
    budget = FakeBudgetPort()
    provider = _RecordingAnswerProvider(failure=KeyError("choices"))
    with pytest.raises(TransportFailed) as caught:
        await answer(AnswerRequest(question="q"), provider=provider, budget=budget)

    assert "unmapped KeyError" in caught.value.message
    assert caught.value.spend.calls == 0, "the call layer claims no bill it cannot see"
    assert budget.records == [(caught.value.spend, ())]


async def test_a_criterion_the_adapter_forgot_is_answered_from_the_declaration() -> None:
    forgetful = _RecordingAnswerProvider(
        result=GROUNDED_ANSWER.model_copy(
            update={"dispositions": (CriterionDisposition(criterion_key="language", disposition="pushdown"),)}
        )
    )
    criteria = (
        Criterion.language("en"),
        Criterion.time_range(end=datetime(2026, 1, 1, tzinfo=UTC)),
        Criterion.namespaced("brave", "country", "us"),
    )
    result = await answer(AnswerRequest(question="q", criteria=criteria), provider=forgetful)

    assert [(d.criterion_key, d.disposition) for d in result.dispositions] == [
        ("language", "pushdown"),
        ("time-range", "unsatisfied"),
        ("brave:country", "pushdown"),
    ]
    assert "capability declaration" in (result.dispositions[1].detail or "")


async def test_the_call_layer_over_the_real_adapter_reads_the_recorded_bill() -> None:
    budget = FakeBudgetPort()
    transport = RecordedStreamingTransport((RecordedStream.of(SINGLE),))
    adapter = BraveAnswersAdapter(api_key="k", transport=transport)
    result = await answer(AnswerRequest(question="largest rodent?"), provider=adapter, budget=budget)

    assert result.spend.money == SINGLE_TOTAL_COST
    assert budget.records[0][0].money == SINGLE_TOTAL_COST
    assert budget.checks[0][0].provider_unit == "brave:searches"
    assert transport.calls[0]["timeout_seconds"] == DEFAULT_ANSWER_TIMEOUT_SECONDS
