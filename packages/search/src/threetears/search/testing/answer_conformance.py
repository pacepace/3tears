"""The answers conformance suite -- one suite, every answers adapter (AR-I2, SR-O5).

The provider suite's discipline (:mod:`threetears.search.testing.conformance`)
applied to the answer seam, run against a provider's own recorded streams
rather than invented ones. Seven pins, each a guarantee a consumer relies on
whichever answers provider it was handed:

1. **contract shape** -- the text arrives tag-free, every source is a
   candidate whose page is unread (AD7, AR-A2), every anchor lands inside the
   text and cites a source it has (AR-A4), provenance names the instance and
   the egress, and the answer JSON round-trips (SR-L4);
2. **spend is the provider's bill** -- a served answer counts one call, and a
   provider pricing per weighted unit reports its units under its own label
   (AD10, SR-E4);
3. **zero sources is a success that says so** (AR-A3);
4. **spend survives the failure path** (SR-E3);
5. **the error taxonomy stays distinguishable** (AD9, SR-J1);
6. **a stream broken after it began is a typed failure counting the call**,
   never retried (AR-E1);
7. **disposition honesty** -- one answer per criterion, matching the
   declaration, and a pushdown that reached the wire (SR-B2, SR-B3).

**How to run it.** Subclass :class:`AnswerConformanceSuite` in a test module,
set ``case``, and name the subclass ``Test*``. No pytest import, for the
reason the provider suite has none.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from threetears.search.contracts import (
    PRICING_PER_WEIGHTED_UNIT,
    PRODUCER_API_PROVIDER,
    AnswerProvider,
    AnswerRequest,
    AuthFailed,
    Criterion,
    GroundedAnswer,
    MalformedResponse,
    RateLimited,
    SearchFailure,
    Spend,
    StreamingTransport,
    TransportFailed,
)
from threetears.search.testing.recorded import RecordedStream, RecordedStreamingTransport

__all__ = ["AnswerConformanceCase", "AnswerConformanceSuite"]


@dataclass(frozen=True, slots=True)
class AnswerConformanceCase:
    """One answers provider, plus the recordings only its own API can supply."""

    #: builds the provider under test over an injected streaming transport.
    provider_factory: Callable[[StreamingTransport], AnswerProvider]
    #: a recorded answer citing at least one source, in the provider's own stream.
    cited_stream: bytes
    #: a stream answering with no citations.
    uncited_stream: bytes
    #: a stream that is well-formed SSE but not the shape the API promises.
    malformed_stream: bytes
    #: the provider's rate-limit refusal: status and body.
    rate_limited: tuple[int, bytes]
    #: the provider's refusal of the key: status and body.
    auth_failed: tuple[int, bytes]
    #: a criterion this provider pushes down to its API.
    pushdown_criterion: Criterion
    #: a criterion this provider cannot honour.
    unsatisfiable_criterion: Criterion
    #: the request body field ``pushdown_criterion`` should produce.
    pushdown_parameter: str


async def _expect_failure(provider: AnswerProvider, request: AnswerRequest) -> SearchFailure:
    """Ask a question that must fail, and return the failure it raised.

    :param provider: the provider under test
    :ptype provider: AnswerProvider
    :param request: the question
    :ptype request: AnswerRequest
    :return: the typed failure
    :rtype: SearchFailure
    :raises AssertionError: when the answer succeeded, or failed outside the taxonomy
    """
    try:
        result = await provider.answer(request)
    except SearchFailure as failure:
        return failure
    except Exception as exc:
        # an untyped escape is the finding this pin exists for, so it is named.
        raise AssertionError(f"provider raised {type(exc).__name__} outside the typed taxonomy: {exc}") from exc
    raise AssertionError(f"expected a typed failure, got an answer citing {len(result.sources)} source(s)")


class AnswerConformanceSuite:
    """The pins every answers adapter must pass, against its own recordings.

    Subclass it, set :attr:`case`, and name the subclass ``Test*``.
    """

    #: the provider under test and its recordings. Set by the subclass.
    case: ClassVar[AnswerConformanceCase]

    def _provider(self, *streams: RecordedStream) -> tuple[AnswerProvider, RecordedStreamingTransport]:
        """Build the provider over a transport replaying ``streams``.

        :param streams: the recordings the transport answers with, in order
        :ptype streams: RecordedStream
        :return: the provider and the transport it reads through
        :rtype: tuple[AnswerProvider, RecordedStreamingTransport]
        """
        transport = RecordedStreamingTransport(streams)
        return self.case.provider_factory(transport), transport

    async def _cited(self, question: str) -> tuple[AnswerProvider, GroundedAnswer]:
        """Answer ``question`` from the cited recording.

        :param question: the question
        :ptype question: str
        :return: the provider and its answer
        :rtype: tuple[AnswerProvider, GroundedAnswer]
        """
        provider, _ = self._provider(RecordedStream.of(self.case.cited_stream))
        return provider, await provider.answer(AnswerRequest(question=question))

    async def test_contract_shape(self) -> None:
        """Tag-free text, unread candidate sources, anchors inside the text, full provenance."""
        provider, result = await self._cited("conformance shape")

        assert result.text, "the cited recording must yield answer text"
        assert "<citation>" not in result.text and "</" not in result.text, "a tag reached the answer text"
        assert result.sources, "the cited recording must yield at least one source"
        for source in result.sources:
            assert source.identity and source.locators
            assert source.content is None, "a source's page is unread until Extract reads it (AD7)"
            assert source.provenance.query == "conformance shape"
            assert source.provenance.provider_instance == provider.provider_instance
            assert source.provenance.egress, "egress is a named value, never an absence (D20)"
            assert source.provenance.producer == PRODUCER_API_PROVIDER
        for anchor in result.anchors:
            assert anchor.end <= len(result.text)
            assert anchor.source_index < len(result.sources)
        assert result.provenance.provider_instance == provider.provider_instance
        assert result.provenance.query == "conformance shape"
        assert GroundedAnswer.model_validate_json(result.model_dump_json()) == result

    async def test_spend_is_the_provider_s_bill(self) -> None:
        """A served answer counts one call; weighted pricing reports its own labelled units."""
        provider, result = await self._cited("conformance spend")

        assert result.spend.calls == 1
        if provider.capabilities.pricing_model != PRICING_PER_WEIGHTED_UNIT:
            return
        assert result.spend.provider_units > 0, "a weighted provider billed nothing on a served answer"
        assert result.spend.provider_unit == provider.capabilities.qualified_unit

    async def test_zero_sources_is_a_success_that_says_so(self) -> None:
        """An uncited answer is a value carrying a notice (AR-A3)."""
        provider, _ = self._provider(RecordedStream.of(self.case.uncited_stream))
        result = await provider.answer(AnswerRequest(question="conformance emptiness"))

        assert result.sources == ()
        assert result.notices, "an uncited answer must say so"

    async def test_spend_survives_the_failure_path(self) -> None:
        """A refused call still reports what it consumed (SR-E3)."""
        status, body = self.case.rate_limited
        provider, _ = self._provider(RecordedStream.of(body, status_code=status, headers={}))
        failure = await _expect_failure(provider, AnswerRequest(question="conformance spend failure"))

        assert failure.provider_instance == provider.provider_instance
        assert failure.to_record().to_failure().spend == failure.spend, "the failure round-trips its spend"

    async def test_error_taxonomy_stays_distinguishable(self) -> None:
        """A rate limit, a refused key and a garbled stream are three classes (SR-J1)."""
        rate_status, rate_body = self.case.rate_limited
        auth_status, auth_body = self.case.auth_failed
        limited, _ = self._provider(RecordedStream.of(rate_body, status_code=rate_status, headers={}))
        denied, _ = self._provider(RecordedStream.of(auth_body, status_code=auth_status, headers={}))
        garbled, _ = self._provider(RecordedStream.of(self.case.malformed_stream))

        request = AnswerRequest(question="conformance taxonomy")
        assert isinstance(await _expect_failure(limited, request), RateLimited)
        assert isinstance(await _expect_failure(denied, request), AuthFailed)
        assert isinstance(await _expect_failure(garbled, request), MalformedResponse)

    async def test_a_stream_broken_after_it_began_counts_the_call_and_is_not_retried(self) -> None:
        """Once the provider began answering it billed the search (AR-E1)."""
        half = self.case.cited_stream[: len(self.case.cited_stream) // 2]
        broken = TransportFailed("the connection broke mid-body", spend=Spend())
        provider, transport = self._provider(RecordedStream.of(half, broken=broken), RecordedStream.of(b""))
        failure = await _expect_failure(provider, AnswerRequest(question="conformance break"))

        assert isinstance(failure, TransportFailed)
        assert failure.spend.calls == 1, "a stream that began was billed, so the call counts"
        assert len(transport.calls) == 1, "a stream broken after its first byte was retried"

    async def test_dispositions_are_honest(self) -> None:
        """Every criterion gets one answer, matching the declaration, and a pushdown reached the wire."""
        unknown = Criterion.namespaced("conformance", "not-a-real-criterion", True)
        criteria = (self.case.pushdown_criterion, self.case.unsatisfiable_criterion, unknown)
        provider, transport = self._provider(RecordedStream.of(self.case.cited_stream))
        result = await provider.answer(AnswerRequest(question="conformance dispositions", criteria=criteria))

        answers = {disposition.criterion_key: disposition for disposition in result.dispositions}
        assert len(result.dispositions) == len(answers), "one answer per criterion, never two"
        assert set(answers) == {criterion.key for criterion in criteria}, "no criterion is silently dropped"
        for criterion in criteria:
            assert answers[criterion.key].disposition == provider.capabilities.disposition_for(criterion.key)
        assert answers[self.case.unsatisfiable_criterion.key].detail, "an unsatisfiable criterion says why"
        body = transport.calls[-1]["json_body"]
        assert isinstance(body, dict) and self.case.pushdown_parameter in body, (
            f"{self.case.pushdown_criterion.key} claims pushdown but sent no {self.case.pushdown_parameter!r}"
        )
