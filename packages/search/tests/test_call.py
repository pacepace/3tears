"""Call's own jobs: bounds, negotiation, failure mapping, spend.

The adapter suite covers what SearXNG does. This covers what Call adds on top
of *any* provider, so the pins are written against a deliberately thin and
occasionally badly-behaved provider double rather than against SearXNG --
Call's guarantees have to hold for the adapter that gets it wrong.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from threetears.search.adapters.searxng import SearxngAdapter
from threetears.search.call import DEFAULT_MAX_RESULTS, DEFAULT_TIMEOUT_SECONDS, MAX_RESULTS_CEILING, search
from threetears.search.contracts import (
    CRITERION_LANGUAGE,
    CRITERION_MAX_RESULTS,
    CRITERION_TIME_RANGE,
    CandidateSet,
    Criterion,
    CriterionDisposition,
    LocalCapExceeded,
    ProviderCapabilities,
    QuotaExhausted,
    RateLimited,
    SearchProvider,
    SearchRequest,
    Spend,
    TransportFailed,
)
from threetears.search.testing import ScriptedTransport, TransportScript
from _searxng_payloads import TWO_RESULTS_BODY, ZERO_RESULTS_BODY


class _RecordingProvider:
    """A provider that records what Call sent it and answers as scripted.

    Not a ``Fake*``: it stands in for the provider seam by satisfying it
    structurally, and it is the subject of these tests rather than a stub for
    something else -- what Call handed down is exactly what is being asserted.
    """

    def __init__(
        self,
        *,
        capabilities: ProviderCapabilities,
        result: CandidateSet | None = None,
        failure: BaseException | None = None,
    ) -> None:
        """Configure the double.

        :param capabilities: the declaration Call negotiates against
        :ptype capabilities: ProviderCapabilities
        :param result: what to answer with, when it answers
        :ptype result: CandidateSet | None
        :param failure: what to raise instead of answering
        :ptype failure: BaseException | None
        """
        self._capabilities = capabilities
        self._result = result if result is not None else CandidateSet()
        self._failure = failure
        self.requests: list[SearchRequest] = []
        self.timeouts: list[float | None] = []

    @property
    def provider(self) -> str:
        """Product name.

        :return: a fixed product name
        :rtype: str
        """
        return "recording"

    @property
    def provider_instance(self) -> str:
        """Instance name.

        :return: a fixed instance name
        :rtype: str
        """
        return "recording-1"

    @property
    def capabilities(self) -> ProviderCapabilities:
        """The declaration Call negotiates against.

        :return: the configured declaration
        :rtype: ProviderCapabilities
        """
        return self._capabilities

    async def search(self, request: SearchRequest, *, timeout_seconds: float | None = None) -> CandidateSet:
        """Record the call and answer as configured.

        :param request: the request Call built
        :ptype request: SearchRequest
        :param timeout_seconds: the bound Call chose
        :ptype timeout_seconds: float | None
        :return: the configured result
        :rtype: CandidateSet
        :raises BaseException: the configured failure, when there is one
        """
        self.requests.append(request)
        self.timeouts.append(timeout_seconds)
        if self._failure is not None:
            raise self._failure
        return self._result


_DECLARATION = ProviderCapabilities(
    provider="recording",
    pushdown_criteria=(CRITERION_LANGUAGE,),
    local_criteria=(CRITERION_MAX_RESULTS,),
    unsatisfiable_criteria=(CRITERION_TIME_RANGE,),
)


def _cap(criteria: Sequence[Criterion]) -> int | None:
    """Read the result cap out of a criteria sequence.

    :param criteria: the criteria Call sent down
    :ptype criteria: Sequence[Criterion]
    :return: the cap's value, or None when there is no cap
    :rtype: int | None
    """
    caps = [criterion.value for criterion in criteria if criterion.key == CRITERION_MAX_RESULTS]
    return caps[-1] if caps and isinstance(caps[-1], int) else None


def _searxng(*steps: TransportScript) -> SearxngAdapter:
    """Build a real SearXNG adapter over a scripted transport.

    :param steps: the exchanges the transport answers with
    :ptype steps: TransportScript
    :return: the adapter
    :rtype: SearxngAdapter
    """
    return SearxngAdapter(base_url="https://searx.example.org", transport=ScriptedTransport(steps))


# --- safe default bounds (SR-L6) -----------------------------------------


async def test_a_caller_who_tunes_nothing_gets_a_bounded_call() -> None:
    """SR-L6: the default is what a MemoryMax cap actually enforces."""
    provider = _RecordingProvider(capabilities=_DECLARATION)
    await search(SearchRequest(query="capybara"), provider=provider)

    assert _cap(provider.requests[0].criteria) == DEFAULT_MAX_RESULTS
    assert provider.timeouts[0] == DEFAULT_TIMEOUT_SECONDS


async def test_a_stated_cap_is_left_alone() -> None:
    """The default is a default, not an override."""
    provider = _RecordingProvider(capabilities=_DECLARATION)
    await search(SearchRequest(query="capybara", criteria=(Criterion.max_results(3),)), provider=provider)

    assert _cap(provider.requests[0].criteria) == 3


async def test_a_stated_deadline_is_passed_down() -> None:
    """SR-G2: a caller under its own deadline bounds the call with what remains."""
    provider = _RecordingProvider(capabilities=_DECLARATION)
    await search(SearchRequest(query="capybara"), provider=provider, timeout_seconds=1.5)

    assert provider.timeouts[0] == 1.5


async def test_asking_past_the_ceiling_is_refused_not_clamped() -> None:
    """D5: a local cap bounds the run's shape, and an overrun is a defect."""
    provider = _RecordingProvider(capabilities=_DECLARATION)
    with pytest.raises(LocalCapExceeded) as raised:
        await search(
            SearchRequest(query="capybara", criteria=(Criterion.max_results(MAX_RESULTS_CEILING + 1),)),
            provider=provider,
        )

    assert raised.value.scope == CRITERION_MAX_RESULTS
    assert raised.value.remediation is not None
    assert provider.requests == [], "the refusal happens before the provider is touched"


async def test_the_ceiling_is_a_parameter_a_run_can_raise() -> None:
    """A cap nobody can raise is a wall; this one is deployment-tunable."""
    provider = _RecordingProvider(capabilities=_DECLARATION)
    await search(
        SearchRequest(query="capybara", criteria=(Criterion.max_results(200),)),
        provider=provider,
        max_results_ceiling=500,
    )

    assert _cap(provider.requests[0].criteria) == 200


async def test_a_local_cap_refusal_is_not_a_quota_exhaustion() -> None:
    """SR-D3/D5: two refusal authorities, and merging them hides which said no."""
    provider = _RecordingProvider(capabilities=_DECLARATION)
    with pytest.raises(LocalCapExceeded) as raised:
        await search(SearchRequest(query="capybara", criteria=(Criterion.max_results(9999),)), provider=provider)

    assert not isinstance(raised.value, QuotaExhausted)
    assert raised.value.failure_class == "local-cap-exceeded"


# --- negotiation (SR-B2, SR-B4) ------------------------------------------


async def test_a_criterion_the_adapter_forgot_is_answered_from_the_declaration() -> None:
    """SR-B2 holds at this boundary whatever the adapter reported."""
    provider = _RecordingProvider(capabilities=_DECLARATION, result=CandidateSet())
    result = await search(
        SearchRequest(query="capybara", criteria=(Criterion.language("en"), Criterion.rights_class("cc0"))),
        provider=provider,
    )

    answers = {disposition.criterion_key: disposition for disposition in result.dispositions}
    assert answers[CRITERION_LANGUAGE].disposition == "pushdown"
    assert answers["rights-class"].disposition == "ignored-unknown"
    assert "capability declaration" in (answers[CRITERION_LANGUAGE].detail or "")


async def test_the_adapters_own_answer_wins_over_the_declaration() -> None:
    """The adapter knows what it actually did, including precedence rules."""
    reported = CandidateSet(
        dispositions=(
            CriterionDisposition(
                criterion_key=CRITERION_LANGUAGE,
                disposition="unsatisfied",
                detail="this instance has no engine for that language",
            ),
        )
    )
    provider = _RecordingProvider(capabilities=_DECLARATION, result=reported)
    result = await search(SearchRequest(query="capybara", criteria=(Criterion.language("en"),)), provider=provider)

    answers = {disposition.criterion_key: disposition for disposition in result.dispositions}
    assert answers[CRITERION_LANGUAGE].disposition == "unsatisfied"
    assert answers[CRITERION_LANGUAGE].detail == "this instance has no engine for that language"


async def test_the_injected_default_cap_is_itself_answered_for() -> None:
    """A criterion Call added is a criterion Call owes an answer for."""
    provider = _RecordingProvider(capabilities=_DECLARATION)
    result = await search(SearchRequest(query="capybara"), provider=provider)

    answers = {disposition.criterion_key: disposition.disposition for disposition in result.dispositions}
    assert answers[CRITERION_MAX_RESULTS] == "local"


async def test_no_criterion_is_answered_twice() -> None:
    """One answer per criterion; a duplicate would make the set unreadable."""
    provider = _RecordingProvider(capabilities=_DECLARATION)
    criteria = (Criterion.language("en"), Criterion.max_results(4), Criterion.rights_class("cc0"))
    result = await search(SearchRequest(query="capybara", criteria=criteria), provider=provider)

    keys = [disposition.criterion_key for disposition in result.dispositions]
    assert len(keys) == len(set(keys)) == len(criteria)


# --- failure mapping and spend -------------------------------------------


async def test_a_typed_failure_passes_through_carrying_spend() -> None:
    """SR-E3: the class is preserved and the accounting survives."""
    failure = RateLimited("slow down", spend=Spend(calls=1, bytes_transferred=12), retry_after_seconds=9.0)
    provider = _RecordingProvider(capabilities=_DECLARATION, failure=failure)
    with pytest.raises(RateLimited) as raised:
        await search(SearchRequest(query="capybara"), provider=provider)

    assert raised.value.retry_after_seconds == 9.0
    assert raised.value.spend.calls == 1
    assert raised.value.spend.bytes_transferred == 12
    assert raised.value.spend.wall_clock_seconds > 0, "Call's own measurement replaces the provider's"


async def test_an_untyped_adapter_escape_becomes_a_typed_failure() -> None:
    """No layer above Call is written against arbitrary exceptions (SR-J1)."""
    provider = _RecordingProvider(capabilities=_DECLARATION, failure=ZeroDivisionError("adapter bug"))
    with pytest.raises(TransportFailed, match="unmapped ZeroDivisionError"):
        await search(SearchRequest(query="capybara"), provider=provider)


async def test_wall_clock_is_replaced_rather_than_summed() -> None:
    """Call's measurement contains the transport's; summing would double it."""
    provider = _RecordingProvider(
        capabilities=_DECLARATION, result=CandidateSet(spend=Spend(wall_clock_seconds=42.0, calls=1))
    )
    result = await search(SearchRequest(query="capybara"), provider=provider)

    assert result.spend.calls == 1
    assert result.spend.wall_clock_seconds < 42.0


# --- end to end through the real adapter ---------------------------------


async def test_zero_results_reaches_the_caller_as_a_success() -> None:
    """SR-J2, through Call rather than only at the adapter."""
    result = await search(SearchRequest(query="nothing"), provider=_searxng(TransportScript(body=ZERO_RESULTS_BODY)))

    assert result.candidates == ()
    assert result.spend.calls == 1


async def test_call_over_the_real_adapter_bounds_and_answers() -> None:
    """The two layers compose: Call's bound, the adapter's dispositions."""
    adapter = _searxng(TransportScript(body=TWO_RESULTS_BODY))
    result = await search(SearchRequest(query="capybara", criteria=(Criterion.language("en"),)), provider=adapter)

    assert len(result.candidates) == 2
    answers = {disposition.criterion_key: disposition.disposition for disposition in result.dispositions}
    assert answers == {CRITERION_LANGUAGE: "pushdown", CRITERION_MAX_RESULTS: "local"}


async def test_the_provider_double_satisfies_the_seam() -> None:
    """If it did not, these pins would be testing something else."""
    assert isinstance(_RecordingProvider(capabilities=_DECLARATION), SearchProvider)
