"""The provider conformance suite -- one suite, every adapter (SR-O5).

Five pins, and each one exists because a consumer's guarantee rests on it
holding for *whichever* provider it was handed:

1. **contract shape** -- a candidate carries identity, locators, provenance
   naming the instance and the egress, and named scores rather than a single
   ``score`` field (SR-C1, SR-A3, D1, D20); and the whole set JSON
   round-trips (SR-L4).
2. **spend on failure** -- a broken call still says what it consumed
   (SR-E3). A failure with no accounting is how a run overspends with
   nobody able to say where.
3. **error taxonomy** -- the distinguishable classes stay distinguishable
   (SR-J1): a rate limit is not an auth failure is not a malformed
   response, because the correct response to each differs.
4. **disposition honesty** -- exactly one answer per criterion the request
   carried, each matching what the provider declared it could do, and a
   criterion reported ``pushdown`` actually put something on the wire
   (SR-B2, SR-B3, SR-B4, P8).
5. **zero results is a success** -- an empty candidate set, never an
   exception (SR-J2).

**How to run it.** Subclass :class:`ProviderConformanceSuite` in a test
module, set the ``case`` class attribute, and name the subclass so the
runner collects it::

    class TestSearxngConformance(ProviderConformanceSuite):
        case = ProviderConformanceCase(...)

The base class is deliberately not named ``Test*``, so it is not collected
on its own -- a suite with no case configured would fail for the wrong
reason.

No pytest import: the methods are ``async def test_*`` over plain
assertions, so the suite runs under whatever runner the consumer already
has and this package's install weight does not grow a test framework.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import ClassVar

from threetears.search.contracts import (
    PRODUCER_API_PROVIDER,
    AuthFailed,
    CandidateSet,
    Criterion,
    MalformedResponse,
    RateLimited,
    SearchFailure,
    SearchProvider,
    SearchRequest,
    SearchTransport,
)
from threetears.search.testing.fakes import ScriptedTransport, TransportScript

__all__ = ["ProviderConformanceCase", "ProviderConformanceSuite"]


@dataclass(frozen=True, slots=True)
class ProviderConformanceCase:
    """One provider, plus the fixtures only its own API can supply.

    Everything derivable from the provider's declaration is derived rather
    than restated here -- a case that had to list its own dispositions could
    agree with itself while disagreeing with the adapter.
    """

    #: builds the provider under test over an injected transport. Takes the
    #: transport because the whole suite drives the provider through it.
    provider_factory: Callable[[SearchTransport], SearchProvider]
    #: a response body carrying at least one result, in this provider's own
    #: JSON shape.
    success_body: bytes
    #: a response body carrying no results.
    zero_results_body: bytes
    #: a body that is well-formed JSON but not the shape the API promises.
    malformed_body: bytes
    #: a criterion this provider pushes down to its API.
    pushdown_criterion: Criterion
    #: a criterion this provider honours by filtering locally.
    local_criterion: Criterion
    #: a criterion this provider cannot honour at all.
    unsatisfiable_criterion: Criterion
    #: the wire parameter name ``pushdown_criterion`` should produce.
    pushdown_parameter: str
    #: status this provider answers a rate limit with.
    rate_limited_status: int = 429
    #: status this provider answers an auth rejection with.
    auth_failed_status: int = 401
    #: headers to send with the rate-limited response.
    rate_limited_headers: Mapping[str, str] = field(default_factory=lambda: {"retry-after": "30"})


async def _expect_failure(provider: SearchProvider, request: SearchRequest) -> SearchFailure:
    """Run a search that must fail, and return the failure it raised.

    :param provider: the provider under test
    :ptype provider: SearchProvider
    :param request: the request to run
    :ptype request: SearchRequest
    :return: the typed failure
    :rtype: SearchFailure
    :raises AssertionError: when the search succeeded, or failed with
        something outside the taxonomy
    """
    try:
        result = await provider.search(request)
    except SearchFailure as failure:
        return failure
    except Exception as exc:
        # an untyped escape from an adapter IS the finding this pin exists for, so it
        # is converted to a named assertion failure rather than propagated as a
        # mystery from inside the provider.
        raise AssertionError(f"provider raised {type(exc).__name__} outside the typed taxonomy: {exc}") from exc
    raise AssertionError(f"expected a typed failure, got a successful set of {len(result.candidates)} candidate(s)")


class ProviderConformanceSuite:
    """The five pins every provider adapter must pass.

    Subclass it, set :attr:`case`, and name the subclass ``Test*``.
    """

    #: the provider under test and its fixtures. Set by the subclass.
    case: ClassVar[ProviderConformanceCase]

    def _provider(self, script: tuple[TransportScript, ...]) -> tuple[SearchProvider, ScriptedTransport]:
        """Build the provider over a scripted transport.

        :param script: the exchanges the transport will answer with
        :ptype script: tuple[TransportScript, ...]
        :return: the provider and the transport it will speak through
        :rtype: tuple[SearchProvider, ScriptedTransport]
        """
        transport = ScriptedTransport(script)
        return self.case.provider_factory(transport), transport

    async def test_contract_shape(self) -> None:
        """A candidate is identifiable, locatable, provenanced and scored."""
        provider, _ = self._provider((TransportScript(body=self.case.success_body),))
        result = await provider.search(SearchRequest(query="conformance shape"))

        assert result.candidates, "the success fixture must yield at least one candidate"
        for candidate in result.candidates:
            assert candidate.identity, "every candidate carries an identity"
            assert candidate.locators, "every candidate carries at least one locator"
            assert candidate.provenance.query == "conformance shape"
            assert candidate.provenance.provider_instance == provider.provider_instance
            assert candidate.provenance.egress, "egress is a named value, never an absence (D20)"
            assert candidate.provenance.producer == PRODUCER_API_PROVIDER
            assert candidate.provenance.retrieved_at.tzinfo is not None
            for score in candidate.scores:
                assert score.name, "a score is named (D1)"
                assert score.scale, "a score states its scale semantics"
                assert score.comparable is False, "provider-native scores are never cross-provider comparable"
        assert CandidateSet.model_validate_json(result.model_dump_json()) == result

    async def test_zero_results_is_success(self) -> None:
        """An empty answer is a value, not an error (SR-J2)."""
        provider, _ = self._provider((TransportScript(body=self.case.zero_results_body),))
        result = await provider.search(SearchRequest(query="conformance emptiness"))

        assert result.candidates == ()
        assert result.spend.calls >= 1, "the provider served the request, so it counts as a call"

    async def test_spend_survives_the_failure_path(self) -> None:
        """A broken call still reports what it consumed (SR-E3)."""
        provider, _ = self._provider(
            (
                TransportScript(
                    status_code=self.case.rate_limited_status,
                    body=b"",
                    headers=dict(self.case.rate_limited_headers),
                    elapsed_seconds=0.5,
                ),
            )
        )
        failure = await _expect_failure(provider, SearchRequest(query="conformance spend"))

        assert failure.spend.wall_clock_seconds > 0, "a failure that took time says so"
        assert failure.provider_instance == provider.provider_instance
        record = failure.to_record()
        assert record.spend == failure.spend
        assert record.to_failure().spend == failure.spend, "the failure round-trips through its wire record"

    async def test_error_taxonomy_stays_distinguishable(self) -> None:
        """Different causes raise different classes (SR-J1)."""
        limited, _ = self._provider(
            (
                TransportScript(
                    status_code=self.case.rate_limited_status, body=b"", headers=dict(self.case.rate_limited_headers)
                ),
            )
        )
        denied, _ = self._provider((TransportScript(status_code=self.case.auth_failed_status, body=b""),))
        garbled, _ = self._provider((TransportScript(body=self.case.malformed_body),))

        request = SearchRequest(query="conformance taxonomy")
        rate_failure = await _expect_failure(limited, request)
        auth_failure = await _expect_failure(denied, request)
        shape_failure = await _expect_failure(garbled, request)

        assert isinstance(rate_failure, RateLimited)
        assert isinstance(auth_failure, AuthFailed)
        assert isinstance(shape_failure, MalformedResponse)
        classes = {rate_failure.failure_class, auth_failure.failure_class, shape_failure.failure_class}
        assert len(classes) == 3, f"three causes collapsed into {classes}"

    async def test_dispositions_are_honest(self) -> None:
        """Every criterion gets one answer, and it matches the declaration."""
        unknown = Criterion.namespaced("conformance", "not-a-real-criterion", True)
        criteria = (
            self.case.pushdown_criterion,
            self.case.local_criterion,
            self.case.unsatisfiable_criterion,
            unknown,
        )
        provider, transport = self._provider((TransportScript(body=self.case.success_body),))
        result = await provider.search(SearchRequest(query="conformance dispositions", criteria=criteria))

        answers = {disposition.criterion_key: disposition for disposition in result.dispositions}
        assert len(result.dispositions) == len(answers), "one answer per criterion, never two"
        assert set(answers) == {criterion.key for criterion in criteria}, "no criterion is silently dropped (SR-B3)"
        assert answers[self.case.pushdown_criterion.key].disposition == "pushdown"
        assert answers[self.case.local_criterion.key].disposition == "local"
        assert answers[self.case.unsatisfiable_criterion.key].disposition == "unsatisfied"
        assert answers[unknown.key].disposition == "ignored-unknown"
        assert answers[self.case.unsatisfiable_criterion.key].detail, "an unsatisfiable criterion says why (SR-B3)"

        for criterion in criteria:
            declared = provider.capabilities.disposition_for(criterion.key)
            assert answers[criterion.key].disposition == declared, (
                f"{criterion.key} was reported {answers[criterion.key].disposition} but declared {declared}"
            )

        sent = transport.calls[-1]["params"]
        assert isinstance(sent, dict)
        assert self.case.pushdown_parameter in sent, (
            f"{self.case.pushdown_criterion.key} claims pushdown but sent no {self.case.pushdown_parameter!r} parameter"
        )

    async def test_the_success_fixture_is_this_provider_s_own_shape(self) -> None:
        """The fixtures are JSON, so a mis-specified case fails here not later."""
        for name, body in (
            ("success_body", self.case.success_body),
            ("zero_results_body", self.case.zero_results_body),
            ("malformed_body", self.case.malformed_body),
        ):
            parsed = json.loads(body)
            assert isinstance(parsed, dict | list), f"{name} must be a JSON document, got {type(parsed).__name__}"
