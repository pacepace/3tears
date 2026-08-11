"""What Call does with the two injected ports, and in what order.

``test_call.py`` covers Call's own jobs -- bounds, negotiation, failure
mapping, spend. This file covers the wiring those jobs sit between: the budget
consulted before the provider and debited after it (D4, D5), and the pacing
slot taken in between, keyed on the pair D8 and D20 name. The pins are written
against the declared witnesses in ``threetears.search.testing`` rather than
against a ledger or a token bucket -- what is under test here is *whether Call
called them, in what order, and with what*, and the ports' own behaviour is
already pinned in ``test_ports.py`` and ``test_limiter.py``.

Ordering across two separate witnesses is not visible in either one's list, so
the two journalling subclasses below write into one shared sequence alongside
the provider. That sequence is the D4/D8 ordering claim, stated once.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from threetears.search.call import DEFAULT_TIMEOUT_SECONDS, search
from threetears.search.contracts import (
    CRITERION_LANGUAGE,
    CRITERION_MAX_RESULTS,
    EGRESS_DIRECT,
    PRICING_FREE_SELF_HOSTED,
    PRICING_PER_WEIGHTED_UNIT,
    BudgetDecision,
    CandidateSet,
    LocalCapExceeded,
    ProviderCapabilities,
    QuotaExhausted,
    RateLimitDecision,
    RateLimited,
    SearchProvider,
    SearchRequest,
    Spend,
    TransportFailed,
)
from threetears.search.adapters.searxng import SearxngAdapter
from threetears.search.testing import FakeBudgetPort, FakeRateLimiterPort, ScriptedTransport, TransportScript
from _searxng_payloads import TWO_RESULTS_BODY

_DECLARATION = ProviderCapabilities(
    provider="wiring",
    pushdown_criteria=(CRITERION_LANGUAGE,),
    local_criteria=(CRITERION_MAX_RESULTS,),
    pricing_model=PRICING_FREE_SELF_HOSTED,
)

_WEIGHTED_DECLARATION = ProviderCapabilities(provider="wiring", pricing_model=PRICING_PER_WEIGHTED_UNIT)


class _JournallingBudget(FakeBudgetPort):
    """The declared budget witness, writing its calls into a shared sequence.

    Subclasses rather than replaces :class:`FakeBudgetPort`, so the parity the
    witness declares against ``BudgetPort`` still holds here: this adds one
    line of bookkeeping to each method and no behaviour.
    """

    def __init__(self, journal: list[str], decision: BudgetDecision | None = None) -> None:
        """Bind the witness to a shared sequence.

        :param journal: the sequence every wired participant appends to
        :ptype journal: list[str]
        :param decision: the decision every ``check`` returns
        :ptype decision: BudgetDecision | None
        """
        super().__init__(decision)
        self._journal = journal

    async def check(self, estimate: Spend, *, scope_tags: tuple[str, ...]) -> BudgetDecision:
        """Note the consultation, then answer as the witness would.

        :param estimate: what the prospective call is expected to consume
        :ptype estimate: Spend
        :param scope_tags: the scopes this call would debit
        :ptype scope_tags: tuple[str, ...]
        :return: the configured decision
        :rtype: BudgetDecision
        """
        self._journal.append("check")
        return await super().check(estimate, scope_tags=scope_tags)

    async def record(self, spend: Spend, *, scope_tags: tuple[str, ...]) -> None:
        """Note the debit, then record it as the witness would.

        :param spend: what the call consumed
        :ptype spend: Spend
        :param scope_tags: the scopes to debit
        :ptype scope_tags: tuple[str, ...]
        :return: nothing
        :rtype: None
        """
        self._journal.append("record")
        await super().record(spend, scope_tags=scope_tags)


class _JournallingLimiter(FakeRateLimiterPort):
    """The declared limiter witness, writing its calls into a shared sequence."""

    def __init__(self, journal: list[str], decision: RateLimitDecision | None = None) -> None:
        """Bind the witness to a shared sequence.

        :param journal: the sequence every wired participant appends to
        :ptype journal: list[str]
        :param decision: the decision every ``acquire`` returns
        :ptype decision: RateLimitDecision | None
        """
        super().__init__(decision)
        self._journal = journal

    async def acquire(
        self,
        *,
        provider_instance: str,
        egress: str,
        tokens: float = 1.0,
        max_wait_seconds: float = 0.0,
    ) -> RateLimitDecision:
        """Note the acquisition, then answer as the witness would.

        :param provider_instance: the deployment about to be called
        :ptype provider_instance: str
        :param egress: the exit the call will leave by
        :ptype egress: str
        :param tokens: how much of the key's allowance this call consumes
        :ptype tokens: float
        :param max_wait_seconds: how long the caller will block for permission
        :ptype max_wait_seconds: float
        :return: the configured decision
        :rtype: RateLimitDecision
        """
        self._journal.append("acquire")
        return await super().acquire(
            provider_instance=provider_instance,
            egress=egress,
            tokens=tokens,
            max_wait_seconds=max_wait_seconds,
        )


class _SlowLimiter(FakeRateLimiterPort):
    """A limiter that grants only after pacing the caller for a real interval.

    The witness answers instantly, which cannot show that a pacing wait is
    taken out of the caller's bound rather than added to it. This one sleeps
    before granting, which is what a real limiter with room to wait does.
    """

    def __init__(self, *, wait_seconds: float) -> None:
        """Fix how long every acquisition takes.

        :param wait_seconds: seconds to pace before granting
        :ptype wait_seconds: float
        """
        super().__init__(RateLimitDecision(acquired=True))
        self._wait_seconds = wait_seconds

    async def acquire(
        self,
        *,
        provider_instance: str,
        egress: str,
        tokens: float = 1.0,
        max_wait_seconds: float = 0.0,
    ) -> RateLimitDecision:
        """Pace, then grant.

        :param provider_instance: the deployment about to be called
        :ptype provider_instance: str
        :param egress: the exit the call will leave by
        :ptype egress: str
        :param tokens: how much of the key's allowance this call consumes
        :ptype tokens: float
        :param max_wait_seconds: how long the caller will block for permission
        :ptype max_wait_seconds: float
        :return: a grant, after the wait
        :rtype: RateLimitDecision
        """
        await asyncio.sleep(self._wait_seconds)
        return await super().acquire(
            provider_instance=provider_instance,
            egress=egress,
            tokens=tokens,
            max_wait_seconds=max_wait_seconds,
        )


class _WiringProvider:
    """A provider that answers as configured and marks where it sat in the order.

    Not a ``Fake*``: it satisfies the provider seam structurally and is the
    subject of these pins rather than a stub for something else. What it adds
    over ``test_call.py``'s double is the journal entry it writes when it is
    invoked -- which is what makes "before the call" and "after the call"
    observable facts rather than an inference from two witnesses' final counts.
    """

    def __init__(
        self,
        *,
        journal: list[str] | None = None,
        capabilities: ProviderCapabilities = _DECLARATION,
        result: CandidateSet | None = None,
        failure: BaseException | None = None,
    ) -> None:
        """Configure the double.

        :param journal: the shared ordering sequence, when one is in use
        :ptype journal: list[str] | None
        :param capabilities: the declaration Call negotiates and estimates against
        :ptype capabilities: ProviderCapabilities
        :param result: what to answer with, when it answers
        :ptype result: CandidateSet | None
        :param failure: what to raise instead of answering
        :ptype failure: BaseException | None
        """
        self._journal = journal
        self._capabilities = capabilities
        self._result = result if result is not None else CandidateSet(spend=Spend(calls=1))
        self._failure = failure
        self.requests: list[SearchRequest] = []
        self.timeouts: list[float | None] = []

    @property
    def provider(self) -> str:
        """Product name.

        :return: a fixed product name
        :rtype: str
        """
        return "wiring"

    @property
    def provider_instance(self) -> str:
        """Instance name -- half of D8's pacing key.

        :return: a fixed instance name
        :rtype: str
        """
        return "wiring-1"

    @property
    def capabilities(self) -> ProviderCapabilities:
        """The declaration Call negotiates and estimates against.

        :return: the configured declaration
        :rtype: ProviderCapabilities
        """
        return self._capabilities

    async def search(self, request: SearchRequest, *, timeout_seconds: float | None = None) -> CandidateSet:
        """Note the call and answer as configured.

        :param request: the request Call built
        :ptype request: SearchRequest
        :param timeout_seconds: the bound Call had left for the provider
        :ptype timeout_seconds: float | None
        :return: the configured result
        :rtype: CandidateSet
        :raises BaseException: the configured failure, when there is one
        """
        if self._journal is not None:
            self._journal.append("provider")
        self.requests.append(request)
        self.timeouts.append(timeout_seconds)
        if self._failure is not None:
            raise self._failure
        return self._result


# --- ordering (D4, D8) ----------------------------------------------------


async def test_the_ports_are_consulted_around_the_call_in_the_ruled_order() -> None:
    """D4/D8: check, then pace, then call, then record what it actually cost."""
    journal: list[str] = []
    budget = _JournallingBudget(journal)
    limiter = _JournallingLimiter(journal)
    provider = _WiringProvider(journal=journal)

    await search(SearchRequest(query="capybara"), provider=provider, budget=budget, limiter=limiter)

    assert journal == ["check", "acquire", "provider", "record"]


async def test_a_call_with_no_ports_is_unchanged() -> None:
    """A consumer without budgets passes nothing and gets nothing injected."""
    provider = _WiringProvider()
    result = await search(SearchRequest(query="capybara"), provider=provider)

    assert result.spend.calls == 1
    assert provider.timeouts[0] == DEFAULT_TIMEOUT_SECONDS


# --- the estimate (SR-D1, SR-E2, D6) --------------------------------------


async def test_the_estimate_is_one_call_for_a_provider_that_bills_nothing() -> None:
    """D6/SR-D6: a free self-hosted call is estimable without synthetic pricing."""
    budget = FakeBudgetPort()
    await search(SearchRequest(query="capybara"), provider=_WiringProvider(), budget=budget)

    estimate, _ = budget.checks[0]
    assert estimate == Spend(calls=1)


async def test_a_provider_that_declares_weighted_pricing_estimates_a_unit_floor() -> None:
    """SR-E4: a credit budget needs a number to refuse on, read off the declaration."""
    budget = FakeBudgetPort()
    provider = _WiringProvider(capabilities=_WEIGHTED_DECLARATION)
    await search(SearchRequest(query="capybara"), provider=provider, budget=budget)

    estimate, _ = budget.checks[0]
    assert estimate.calls == 1
    assert estimate.provider_units == Decimal(1)


async def test_the_estimate_names_no_money_and_no_wall_clock() -> None:
    """An invented price and a ceiling-as-expectation are both dishonest estimates."""
    budget = FakeBudgetPort()
    await search(SearchRequest(query="capybara"), provider=_WiringProvider(), budget=budget)

    estimate, _ = budget.checks[0]
    assert estimate.money == Decimal(0)
    assert estimate.wall_clock_seconds == 0.0


async def test_scope_tags_travel_from_the_request_to_both_halves() -> None:
    """SR-D2: the scopes a call debits are the scopes it was checked against."""
    budget = FakeBudgetPort()
    tags = ("run:7", "persona:ana/2026-08-11")
    await search(SearchRequest(query="capybara", budget_scope_tags=tags), provider=_WiringProvider(), budget=budget)

    assert budget.checks[0][1] == tags
    assert budget.records[0][1] == tags


# --- refusal (D5, SR-D3) --------------------------------------------------


async def test_a_refused_call_reaches_neither_the_limiter_nor_the_provider() -> None:
    """A call that will not be made needs no pacing slot and sends no request."""
    budget = FakeBudgetPort(BudgetDecision(allowed=False, scope="run:7"))
    limiter = FakeRateLimiterPort()
    provider = _WiringProvider()

    with pytest.raises(LocalCapExceeded):
        await search(SearchRequest(query="capybara"), provider=provider, budget=budget, limiter=limiter)

    assert provider.requests == []
    assert limiter.acquisitions == []


async def test_a_refusal_carries_the_decisions_own_facts() -> None:
    """SR-E3/SR-D2: which authority said no, what it had spent, how to fix it."""
    consumed = Spend(calls=42, money=Decimal("1.25"))
    budget = FakeBudgetPort(
        BudgetDecision(
            allowed=False,
            scope="persona:ana/2026-08-11",
            reason="the day's 40-call allowance is spent",
            remediation="raise the per-persona daily allowance, or wait for the day to roll",
            consumed=consumed,
        )
    )

    with pytest.raises(LocalCapExceeded) as raised:
        await search(SearchRequest(query="capybara"), provider=_WiringProvider(), budget=budget, egress="corp-proxy")

    assert raised.value.scope == "persona:ana/2026-08-11"
    assert raised.value.message == "the day's 40-call allowance is spent"
    assert raised.value.remediation is not None
    assert raised.value.spend == consumed
    assert raised.value.provider_instance == "wiring-1"
    assert raised.value.egress == "corp-proxy"


async def test_a_budget_refusal_is_not_the_providers_quota_refusal() -> None:
    """SR-D3: the local authority's no and the provider's no stay distinguishable."""
    budget = FakeBudgetPort(BudgetDecision(allowed=False, scope="run:7"))

    with pytest.raises(LocalCapExceeded) as raised:
        await search(SearchRequest(query="capybara"), provider=_WiringProvider(), budget=budget)

    assert not isinstance(raised.value, QuotaExhausted)
    assert raised.value.failure_class == "local-cap-exceeded"


async def test_a_refused_call_records_nothing() -> None:
    """D4: budget follows the bill, and a call nobody made has no bill."""
    budget = FakeBudgetPort(BudgetDecision(allowed=False, scope="run:7"))

    with pytest.raises(LocalCapExceeded):
        await search(SearchRequest(query="capybara"), provider=_WiringProvider(), budget=budget)

    assert budget.records == []


# --- pacing (D8, D20) -----------------------------------------------------


async def test_pacing_is_keyed_on_the_provider_instance_and_the_egress() -> None:
    """D8/D20: two deployments, and two exits out of one, are separate subjects."""
    limiter = FakeRateLimiterPort()
    await search(SearchRequest(query="capybara"), provider=_WiringProvider(), limiter=limiter, egress="corp-proxy")

    instance, egress, tokens, _ = limiter.acquisitions[0]
    assert (instance, egress) == ("wiring-1", "corp-proxy")
    assert tokens == 1.0, "one call is one token; a provider's weight is a billing fact, not a pacing one"


async def test_the_default_egress_is_the_named_direct_value() -> None:
    """D20: ``direct`` is a value like any other, never an absence."""
    limiter = FakeRateLimiterPort()
    await search(SearchRequest(query="capybara"), provider=_WiringProvider(), limiter=limiter)

    assert limiter.acquisitions[0][1] == EGRESS_DIRECT


async def test_the_callers_bound_is_what_the_limiter_may_wait() -> None:
    """SR-G2: a caller under a deadline passes what remains of it."""
    limiter = FakeRateLimiterPort()
    await search(SearchRequest(query="capybara"), provider=_WiringProvider(), limiter=limiter, timeout_seconds=1.5)

    assert limiter.acquisitions[0][3] == 1.5


async def test_a_pacing_wait_comes_out_of_the_bound_rather_than_on_top_of_it() -> None:
    """The bound the caller stated bounds the whole call, pacing included."""
    provider = _WiringProvider()
    await search(
        SearchRequest(query="capybara"), provider=provider, limiter=_SlowLimiter(wait_seconds=0.05), timeout_seconds=1.0
    )

    passed_down = provider.timeouts[0]
    assert passed_down is not None
    assert 0.9 < passed_down < 1.0, "the provider gets what pacing left, not the whole bound again"


async def test_a_denial_the_caller_will_not_wait_out_is_a_typed_rate_limit() -> None:
    """SR-J1: the caller meets the taxonomy, carrying the limiter's own backoff."""
    limiter = FakeRateLimiterPort(RateLimitDecision(acquired=False, retry_after_seconds=2.5))
    provider = _WiringProvider()

    with pytest.raises(RateLimited) as raised:
        await search(SearchRequest(query="capybara"), provider=provider, limiter=limiter, egress="corp-proxy")

    assert raised.value.retry_after_seconds == 2.5
    assert raised.value.provider_instance == "wiring-1"
    assert raised.value.egress == "corp-proxy"
    assert provider.requests == [], "the call was never made"


async def test_a_denied_call_bills_nothing_and_records_nothing() -> None:
    """The call never happened: no bill to carry, and none to debit (D4, SR-E3)."""
    budget = FakeBudgetPort()
    limiter = FakeRateLimiterPort(RateLimitDecision(acquired=False, retry_after_seconds=2.5))

    with pytest.raises(RateLimited) as raised:
        await search(SearchRequest(query="capybara"), provider=_WiringProvider(), budget=budget, limiter=limiter)

    assert raised.value.spend.calls == 0
    assert raised.value.spend.money == Decimal(0)
    assert raised.value.spend.bytes_transferred == 0
    assert raised.value.spend.wall_clock_seconds >= 0.0
    assert budget.records == []
    assert budget.checks != [], "the budget was still asked -- the refusal came from pacing, not from it"


# --- recording (SR-E2, SR-E3, D4) -----------------------------------------


async def test_the_recorded_spend_is_the_one_the_caller_receives() -> None:
    """SR-E2: never a locally re-derived tally, and never two different numbers."""
    reported = CandidateSet(spend=Spend(calls=1, money=Decimal("0.02"), bytes_transferred=4096))
    budget = FakeBudgetPort()
    result = await search(SearchRequest(query="capybara"), provider=_WiringProvider(result=reported), budget=budget)

    assert budget.records[0][0] == result.spend
    assert budget.records[0][0].bytes_transferred == 4096


async def test_a_typed_failure_is_recorded_before_it_propagates() -> None:
    """SR-E3: a failure carries the spend it incurred, so that spend debits."""
    failure = QuotaExhausted("out of credits", spend=Spend(calls=1, provider_units=Decimal(2)))
    budget = FakeBudgetPort()

    with pytest.raises(QuotaExhausted) as raised:
        await search(SearchRequest(query="capybara"), provider=_WiringProvider(failure=failure), budget=budget)

    recorded, _ = budget.records[0]
    assert recorded.calls == 1
    assert recorded.provider_units == Decimal(2)
    assert recorded == raised.value.spend, "the budget hears exactly what the caller hears"


async def test_an_adapter_defect_records_what_call_could_measure() -> None:
    """A defective adapter reported no spend; Call claims no bill it cannot see."""
    budget = FakeBudgetPort()
    provider = _WiringProvider(failure=ZeroDivisionError("adapter bug"))

    with pytest.raises(TransportFailed) as raised:
        await search(SearchRequest(query="capybara"), provider=provider, budget=budget)

    recorded, _ = budget.records[0]
    assert recorded.calls == 0
    assert recorded.wall_clock_seconds > 0
    assert recorded == raised.value.spend


async def test_transport_retries_below_the_seam_are_consulted_for_once() -> None:
    """D4: an attempt that was retried and never billed never debits a budget."""
    adapter = SearxngAdapter(
        base_url="https://searx.example.org",
        transport=ScriptedTransport([TransportScript(body=TWO_RESULTS_BODY, attempts=3)]),
    )
    budget = FakeBudgetPort()
    limiter = FakeRateLimiterPort()

    await search(SearchRequest(query="capybara"), provider=adapter, budget=budget, limiter=limiter)

    assert len(budget.checks) == 1
    assert len(budget.records) == 1
    assert len(limiter.acquisitions) == 1
    assert budget.records[0][0].calls == 1, "three attempts, one billed call (SR-E2)"
    assert limiter.acquisitions[0][0] == "searx.example.org"


async def test_the_wiring_doubles_satisfy_the_seams_they_stand_in_for() -> None:
    """If they did not, every pin above would be testing something else."""
    assert isinstance(_WiringProvider(), SearchProvider)
