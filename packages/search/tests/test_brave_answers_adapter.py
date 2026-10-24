"""The Brave Answers adapter (answers-requirements-design.md §10.2), against what Brave really sent.

The recorded streams pin the reading -- text, anchors in code points, sources,
spend -- and synthetic streams in the same framing pin the edges no capture
exhibits: boundaries mid-tag and mid-character, an unknown tag, a missing or
unreadable bill, a stream that breaks after it began, and each refusal.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from threetears.search.adapters.brave_answers import (
    BRAVE_ANSWERS_CAPABILITIES,
    BRAVE_PARAM_COUNTRY,
    BRAVE_PARAM_MAX_COMPLETION_TOKENS,
    BRAVE_PARAM_SAFESEARCH,
    BRAVE_PARAM_SEARCH_CONTEXT_SIZE,
    BRAVE_VALIDATION_REMEDIATION,
    BraveAnswersAdapter,
)
from threetears.search.contracts import (
    FIDELITY_SNIPPET,
    AnswerProvider,
    AnswerRequest,
    AuthFailed,
    Criterion,
    GroundedAnswer,
    LocalCapExceeded,
    MalformedResponse,
    QuotaExhausted,
    RateLimited,
    Spend,
    StreamingTransport,
    TimedOut,
    TransportFailed,
    get_capabilities,
)
from packages.search.tests.recorded_streams import RecordedStream, RecordedStreamingTransport
from packages.search.tests.brave_answers_payloads import (
    INVALID_TOKEN_BODY,
    NON_ASCII,
    NON_ASCII_CITATIONS,
    NON_ASCII_POINT,
    NON_ASCII_UTF8_LENGTH,
    RATE_LIMITED_BODY,
    SINGLE,
    SINGLE_CITATIONS,
    SINGLE_POINTS,
    SINGLE_TEXT_LENGTH,
    SINGLE_TOTAL_COST,
    VALIDATION_BODY,
    citation,
    envelope,
    stream,
    usage_tag,
)

QUESTION = "what is the largest rodent?"


def _adapter(transport: StreamingTransport, **overrides: object) -> BraveAnswersAdapter:
    """A Brave adapter over ``transport``, with a test key.

    :param transport: the streaming transport
    :ptype transport: StreamingTransport
    :param overrides: constructor arguments to change
    :ptype overrides: object
    :return: the adapter
    :rtype: BraveAnswersAdapter
    """
    arguments: dict[str, object] = {"api_key": "test-key", "transport": transport, "provider_instance": "brave-live"}
    arguments.update(overrides)
    return BraveAnswersAdapter(**arguments)  # type: ignore[arg-type]


async def _answer(
    body: bytes, *, chunk_size: int | None = None, request: AnswerRequest | None = None
) -> tuple[GroundedAnswer, RecordedStreamingTransport]:
    """Answer from one recorded body.

    :param body: the stream
    :ptype body: bytes
    :param chunk_size: bytes per chunk; None replays one event per chunk
    :ptype chunk_size: int | None
    :param request: the request; a plain question when None
    :ptype request: AnswerRequest | None
    :return: the answer and the transport it read through
    :rtype: tuple[GroundedAnswer, RecordedStreamingTransport]
    """
    transport = RecordedStreamingTransport((RecordedStream.of(body, chunk_size=chunk_size),))
    result = await _adapter(transport).answer(request or AnswerRequest(question=QUESTION))
    return result, transport


async def _failure(stream_: RecordedStream, request: AnswerRequest | None = None) -> Exception:
    """Answer from one recording that must fail, and return the failure.

    :param stream_: the recording
    :ptype stream_: RecordedStream
    :param request: the request; a plain question when None
    :ptype request: AnswerRequest | None
    :return: what the adapter raised
    :rtype: Exception
    """
    transport = RecordedStreamingTransport((stream_,))
    with pytest.raises(Exception) as caught:
        await _adapter(transport).answer(request or AnswerRequest(question=QUESTION))
    return caught.value


# --- the seam ----------------------------------------------------------------


def test_the_adapter_is_an_answers_provider_registered_under_the_answer_kind() -> None:
    """Brave's answers declare themselves before contact, under their own kind (AR-B2)."""
    adapter = _adapter(RecordedStreamingTransport((RecordedStream.of(SINGLE),)))
    assert isinstance(adapter, AnswerProvider)
    assert get_capabilities("brave", kind="answer") == BRAVE_ANSWERS_CAPABILITIES
    assert BRAVE_ANSWERS_CAPABILITIES.modes == ("single",)
    assert BRAVE_ANSWERS_CAPABILITIES.citations_in_modes == ("single",)
    assert BRAVE_ANSWERS_CAPABILITIES.qualified_unit == "brave:searches"
    assert BRAVE_ANSWERS_CAPABILITIES.minimum_timeout_seconds == 30.0


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"api_key": ""}, "host-supplied api_key"),
        ({"base_url": "ftp://brave.example"}, "absolute http"),
        ({"usd_per_search": Decimal("0.004")}, "together or not at all"),
        ({"usd_per_search": Decimal("-1"), "usd_per_million_tokens": Decimal("5")}, "cannot be negative"),
        ({"timeout_seconds": 0}, "timeout_seconds"),
        ({"max_stream_bytes": 0}, "max_stream_bytes"),
    ],
)
def test_construction_refuses_what_a_deployment_must_state(overrides: dict[str, object], match: str) -> None:
    """A key, a sane base URL, a whole rate card or none, positive bounds (AR-B3, AR-C3)."""
    with pytest.raises(ValueError, match=match):
        _adapter(RecordedStreamingTransport((RecordedStream.of(SINGLE),)), **overrides)


async def test_the_request_is_one_streamed_cited_user_message_under_the_subscription_header() -> None:
    """AD5, AR-K1: always streamed with citations; the question is the one message, wrapped in nothing."""
    _, transport = await _answer(SINGLE)

    (call,) = transport.calls
    assert call["method"] == "POST"
    assert call["url"] == "https://api.search.brave.com/res/v1/chat/completions"
    assert call["headers"] == {"Accept": "text/event-stream", "X-Subscription-Token": "test-key"}
    assert call["json_body"] == {
        "model": "brave",
        "messages": [{"role": "user", "content": QUESTION}],
        "stream": True,
        "enable_citations": True,
    }
    assert call["timeout_seconds"] == 45.0, "no call bound means the adapter's own"


async def test_an_empty_question_is_refused_before_any_stream_is_opened() -> None:
    transport = RecordedStreamingTransport((RecordedStream.of(SINGLE),))
    with pytest.raises(LocalCapExceeded) as caught:
        await _adapter(transport).answer(AnswerRequest(question="   "))
    assert caught.value.scope == "question-empty"
    assert transport.calls == []


# --- the recorded streams ------------------------------------------------------


async def test_the_single_capture_reads_as_its_text_sources_anchors_and_bill() -> None:
    """Text tag-free; sources in citation order, unread, snippet carried; anchors at the recorded points."""
    result, transport = await _answer(SINGLE)

    assert len(result.text) == SINGLE_TEXT_LENGTH
    assert result.text.startswith("The **capybara** (*Hydrochoerus hydrochaeris*) is the largest living rodent")
    assert "<" not in result.text
    assert len(result.sources) == SINGLE_CITATIONS
    assert [source.provenance.provider_ids["number"] for source in result.sources] == [
        str(number) for number in range(1, SINGLE_CITATIONS + 1)
    ], "sources are in citation order"
    first = result.sources[0]
    assert first.identity == "https://grokipedia.com/page/List_of_largest_rodents"
    assert first.locators[0].url == first.identity
    assert first.content is None, "Extract reads the page; the snippet is not content (AD7)"
    assert first.fidelity_available == FIDELITY_SNIPPET
    assert first.snippet is not None and first.snippet.startswith("# List of largest rodents")
    assert set(first.provenance.provider_ids) == {"number", "start_index", "end_index", "favicon"}
    assert first.provenance.provider_ids["favicon"].startswith("https://imgs.search.brave.com/")
    assert [anchor.start for anchor in result.anchors] == list(SINGLE_POINTS)
    assert all(anchor.start == anchor.end for anchor in result.anchors), "Brave cites zero-width points"
    assert [anchor.source_index for anchor in result.anchors] == list(range(SINGLE_CITATIONS))
    assert result.text[: SINGLE_POINTS[0]].endswith("world."), "the first point follows the first claim"
    assert result.spend.money == SINGLE_TOTAL_COST, "money is the usage tag's own total, through no float"
    assert result.spend.provider_units == Decimal(1)
    assert result.spend.provider_unit == "brave:searches"
    assert result.spend.calls == 1
    assert result.spend.bytes_transferred == len(SINGLE)
    assert (result.prompt_tokens, result.completion_tokens) == (10463, 87)
    assert result.provenance.provider_ids["model"] == "brave-pro", "the served model, not the one named"
    assert result.provenance.provider_instance == "brave-live"
    assert result.provenance.egress == transport.egress_name
    assert result.notices == (), "a whole, cited, billed answer has nothing to warn about"


async def test_the_non_ascii_capture_counts_its_anchors_in_code_points_not_bytes() -> None:
    """Every point is 83 -- the text's length in characters, though it is 94 in UTF-8."""
    result, _ = await _answer(NON_ASCII)

    assert len(result.text) == NON_ASCII_POINT
    assert len(result.text.encode()) == NON_ASCII_UTF8_LENGTH
    assert "Schrödinger" in result.text
    assert len(result.sources) == NON_ASCII_CITATIONS
    assert {anchor.start for anchor in result.anchors} == {NON_ASCII_POINT}
    assert len(result.anchors) == NON_ASCII_CITATIONS


@pytest.mark.parametrize("chunk_size", [1, 7, 64, 1000])
async def test_any_chunking_of_a_capture_reads_the_same_answer(chunk_size: int) -> None:
    """Chunk boundaries fall mid-tag, mid-event and mid-character; none of them changes the answer (AR-F2)."""
    whole, _ = await _answer(NON_ASCII)
    cut, _ = await _answer(NON_ASCII, chunk_size=chunk_size)

    assert cut.model_dump(exclude={"provenance", "sources"}) == whole.model_dump(exclude={"provenance", "sources"})
    assert [source.identity for source in cut.sources] == [source.identity for source in whole.sources]


# --- the edges no capture shows ---------------------------------------------------


async def test_a_tag_split_across_events_and_a_character_split_across_chunks_still_read() -> None:
    """The tag reader buffers across events, and the decoder across chunks (AR-F2)."""
    tag = citation(1, 6)
    body = stream("Café ", "ok.", tag[:9], tag[9:30], tag[30:], usage_tag()[:12], usage_tag()[12:])
    result, _ = await _answer(body, chunk_size=3)

    assert result.text == "Café ok."
    assert [anchor.start for anchor in result.anchors] == [6]
    assert result.spend.money == Decimal("0.01567")
    assert result.spend.provider_units == Decimal(2)


async def test_a_less_than_sign_that_opens_no_tag_is_answer_text() -> None:
    """A tag is ``<name>`` followed by ``{``; a ``<`` in prose stays prose."""
    body = stream("2 < 3 and a <br> break and <b", "old> text ", "<", usage_tag())
    result, _ = await _answer(body)

    assert result.text == "2 < 3 and a <br> break and <bold> text <"


async def test_an_unknown_tag_leaves_the_text_and_is_kept_verbatim_with_a_notice() -> None:
    """AR-A5: provider machinery is not answer text, and dropping it silently would lose it."""
    unknown = '<entities>{"names": ["capybara"]}</entities>'
    body = stream("Capybaras graze.", citation(1, 16), unknown, usage_tag())
    result, _ = await _answer(body)

    assert result.text == "Capybaras graze."
    assert result.provenance.provider_ids["unknown-tag-0"] == unknown
    assert any("<entities>" in notice for notice in result.notices)


async def test_an_unknown_tag_whose_body_is_not_json_is_the_answer_s_own_text() -> None:
    """An answer quoting markup keeps it verbatim, and an anchor after it lands on the right code point."""
    quoted = "Write <div>{x}</div> for a box. "
    claim = "Schrödinger won."
    text = quoted + claim
    split = citation(1, len(text))
    body = stream("Write <di", "v>{x}</d", "iv> for a box. ", claim, split[:20], split[20:], usage_tag())
    result, _ = await _answer(body, chunk_size=5)

    assert result.text == text, "the quoted markup stays where it stood"
    assert [anchor.start for anchor in result.anchors] == [len(text)]
    assert result.text[: result.anchors[0].start].endswith(claim)
    assert not any("<div>" in notice for notice in result.notices), "quoted text is not provider machinery"
    assert not any(key.startswith("unknown-tag") for key in result.provenance.provider_ids)


async def test_an_unknown_tag_never_closed_is_text_too() -> None:
    result, _ = await _answer(stream("Use <span>{ as an opener.", usage_tag()))

    assert result.text == "Use <span>{ as an opener."


async def test_an_unclosed_unknown_tag_never_swallows_a_citation_or_the_bill() -> None:
    """A stray ``<span>{`` before a citation hands itself back as text, so the source and bill still read."""
    text = "Use <span>{ as an opener."
    result, _ = await _answer(stream(text, citation(1, len(text)), usage_tag()), chunk_size=4)

    assert result.text == text
    assert [anchor.start for anchor in result.anchors] == [len(text)]
    assert result.spend.money == Decimal("0.01567"), "the bill after the stray opener was read"


async def test_a_known_tag_with_a_body_that_is_not_json_still_fails_closed() -> None:
    """AR-E2 holds for the two tags the adapter reads, whatever the rule for unknown names."""
    failure = await _failure(RecordedStream.of(stream("Capybaras graze.", "<usage>{oops}</usage>")))

    assert isinstance(failure, MalformedResponse)
    assert "<usage>{oops}</usage>" in failure.message


async def test_a_stream_without_a_usage_report_records_the_call_and_says_the_bill_is_unknown() -> None:
    """AR-C1: never an estimate wearing an observation's clothes."""
    result, _ = await _answer(stream("Capybaras graze.", citation(1, 16), final_usage=False))

    assert result.spend.calls == 1
    assert result.spend.money == Decimal(0)
    assert result.spend.provider_units == Decimal(0)
    assert result.spend.provider_unit is None
    assert result.prompt_tokens is None
    assert any("bill is unknown" in notice for notice in result.notices)


async def test_token_counts_fall_back_to_the_bill_when_the_final_event_has_none() -> None:
    result, _ = await _answer(stream("Capybaras graze.", citation(1, 16), usage_tag(), final_usage=False))

    assert (result.prompt_tokens, result.completion_tokens) == (1234, 300)


async def test_zero_citations_is_a_success_with_a_notice() -> None:
    """AR-A3: an uncited answer is a value, and it says so."""
    result, _ = await _answer(stream("Capybaras graze.", usage_tag()))

    assert result.sources == ()
    assert result.anchors == ()
    assert any("cited no sources" in notice for notice in result.notices)


async def test_a_citation_with_no_usable_offset_anchors_nowhere_and_says_so() -> None:
    """AR-A4: no offset, or one past the text, yields no anchor -- and nothing is inferred from the words."""
    body = stream(
        "Capybaras graze.",
        citation(1, 16),
        citation(2, 16, start_index=None, end_index=None),
        citation(3, 400),
        usage_tag(),
    )
    result, _ = await _answer(body)

    assert len(result.sources) == 3
    assert [anchor.source_index for anchor in result.anchors] == [0]
    assert any("2 of brave's citations gave no usable offset" in notice for notice in result.notices)


async def test_a_source_cited_twice_is_one_source_at_two_points() -> None:
    body = stream("One. Two.", citation(1, 4), citation(1, 9), usage_tag())
    result, _ = await _answer(body)

    assert len(result.sources) == 1
    assert [(anchor.start, anchor.source_index) for anchor in result.anchors] == [(4, 0), (9, 0)]


@pytest.mark.parametrize(
    ("tag", "why"),
    [
        ("<citation>{not json</citation>", "does not parse"),
        ('<citation>{"number": 1, "start_index": 0}</citation>', "names no url"),
        ('<usage>{"X-Request-Queries": 1}</usage>', "X-Request-Total-Cost"),
        ('<usage>{"X-Request-Total-Cost": "0.01", "X-Request-Queries": 1}</usage>', "X-Request-Total-Cost"),
        ('<usage>{"X-Request-Total-Cost": 0.01}</usage>', "X-Request-Queries"),
    ],
    ids=["citation-not-json", "citation-no-url", "usage-no-total", "usage-cost-a-string", "usage-no-queries"],
)
async def test_an_unreadable_tag_fails_closed_carrying_the_raw_tag(tag: str, why: str) -> None:
    """AR-E2: a citation we cannot read, or a bill we cannot state, is MalformedResponse."""
    failure = await _failure(RecordedStream.of(stream("Capybaras graze.", tag)))

    assert isinstance(failure, MalformedResponse)
    assert why in failure.message
    assert tag in failure.message, "the raw tag rides the failure record"
    assert failure.spend.calls == 1, "the provider began answering, so the call counts"


@pytest.mark.parametrize(
    ("body", "why"),
    [
        (stream("Capybaras graze.", usage_tag(), done=False), "without [DONE]"),
        (stream("Capybaras graze.", '<citation>{"number": 1'), "inside a tag"),
        (b"data: not json\n\ndata: [DONE]\n\n", "not JSON"),
        (b"data: [1, 2]\n\ndata: [DONE]\n\n", "not a JSON object"),
        (b"data: [DONE]\n\n" + stream("late", done=False), "after [DONE]"),
        (b'data: {"choices": [{"delta": {"content": "\xff"}}]}\n\n', "not UTF-8"),
    ],
    ids=["no-done", "unterminated-tag", "event-not-json", "event-not-object", "event-after-done", "not-utf8"],
)
async def test_a_stream_that_is_not_the_promised_shape_is_malformed(body: bytes, why: str) -> None:
    failure = await _failure(RecordedStream.of(body))

    assert isinstance(failure, MalformedResponse)
    assert why in failure.message


async def test_a_stream_broken_after_its_first_byte_keeps_its_class_and_counts_the_call() -> None:
    """AR-E1: the transport never retries a begun stream, and the provider billed the search."""
    broken = TimedOut(
        "stream passed its deadline after 120 body byte(s)", spend=Spend(wall_clock_seconds=3.0, bytes_transferred=120)
    )
    transport = RecordedStreamingTransport(
        (RecordedStream.of(stream("Capybaras graze.", citation(1, 16), final_usage=False, done=False), broken=broken),)
    )
    with pytest.raises(TimedOut) as caught:
        await _adapter(transport).answer(AnswerRequest(question=QUESTION))

    failure = caught.value
    assert failure.spend.calls == 1
    assert failure.spend.money == Decimal(0)
    assert failure.spend.wall_clock_seconds == 3.0
    assert "bill is unknown" in failure.message
    assert failure.provider_instance == "brave-live"
    assert len(transport.calls) == 1


async def test_a_stream_broken_after_the_bill_arrived_carries_the_bill() -> None:
    broken = TransportFailed("connection reset mid-body", spend=Spend())
    body = stream("Capybaras graze.", usage_tag(), final_usage=False, done=False)
    failure = await _failure(RecordedStream.of(body, broken=broken))

    assert isinstance(failure, TransportFailed)
    assert failure.spend.money == Decimal("0.01567")
    assert failure.spend.provider_unit == "brave:searches"
    assert "bill is unknown" not in failure.message


async def test_a_raw_timeout_mid_body_is_timed_out_counting_the_call_and_never_retried() -> None:
    transport = RecordedStreamingTransport(
        (RecordedStream.of(stream("Capybaras", done=False), broken=TimeoutError()), RecordedStream.of(SINGLE))
    )
    with pytest.raises(TimedOut) as caught:
        await _adapter(transport).answer(AnswerRequest(question=QUESTION))

    assert caught.value.spend.calls == 1
    assert caught.value.spend.bytes_transferred > 0
    assert "bill is unknown" in caught.value.message
    assert len(transport.calls) == 1


async def test_an_untyped_failure_mid_body_is_a_transport_failure() -> None:
    failure = await _failure(RecordedStream.of(stream("Capybaras", done=False), broken=ConnectionResetError("reset")))

    assert isinstance(failure, TransportFailed)
    assert "ConnectionResetError" in failure.message
    assert failure.spend.calls == 1


async def test_a_stream_past_its_cap_is_refused_as_the_local_cap() -> None:
    transport = RecordedStreamingTransport((RecordedStream.of(SINGLE),))
    with pytest.raises(LocalCapExceeded) as caught:
        await _adapter(transport, max_stream_bytes=4096).answer(AnswerRequest(question=QUESTION))
    assert caught.value.scope == "response-bytes"
    assert transport.calls[0]["max_bytes"] == 4096


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (TimeoutError(), TimedOut),
        (OSError("connect refused"), TransportFailed),
        (AuthFailed("typed already", spend=Spend()), AuthFailed),
    ],
)
async def test_a_failure_before_the_first_byte_is_typed_and_attributed(
    raised: BaseException, expected: type[Exception]
) -> None:
    failure = await _failure(RecordedStream(failure=raised))

    assert isinstance(failure, expected)
    assert getattr(failure, "provider_instance", None) == "brave-live"
    assert getattr(failure, "egress", None) == "direct"


# --- refusals ----------------------------------------------------------------------


async def test_the_captured_invalid_key_is_an_auth_refusal_though_it_is_a_422() -> None:
    """SUBSCRIPTION_TOKEN_INVALID decides the class before the status does."""
    failure = await _failure(RecordedStream.of(INVALID_TOKEN_BODY, status_code=422, headers={}))

    assert isinstance(failure, AuthFailed)
    assert "SUBSCRIPTION_TOKEN_INVALID" in failure.message
    assert failure.spend.calls == 1
    assert failure.spend.money == Decimal(0), "a refused answer bought nothing"


async def test_the_captured_validation_failure_names_its_fields_without_echoing_the_question() -> None:
    failure = await _failure(RecordedStream.of(VALIDATION_BODY, status_code=422, headers={}))

    assert isinstance(failure, TransportFailed)
    assert failure.status == 422
    assert failure.remediation == BRAVE_VALIDATION_REMEDIATION
    assert "body.messages" in failure.message
    assert '"content"' not in failure.message and "'b'" not in failure.message, "the request input is user content"


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (429, RATE_LIMITED_BODY, RateLimited),
        (429, envelope("QUOTA_LIMITED", 429, "monthly credit limit reached"), QuotaExhausted),
        (402, b"", QuotaExhausted),
        (401, b"not json", AuthFailed),
        (429, b"", RateLimited),
        (504, b"", TimedOut),
        (500, envelope("INTERNAL", 500), TransportFailed),
    ],
)
async def test_each_refusal_maps_to_its_own_class(status: int, body: bytes, expected: type[Exception]) -> None:
    """AR-D2: the provider's credit limit, its pace and a key refusal stay distinguishable."""
    failure = await _failure(RecordedStream.of(body, status_code=status, headers={"retry-after": "2"}))

    assert type(failure) is expected
    if isinstance(failure, RateLimited):
        assert failure.retry_after_seconds == 2.0


# --- criteria and the estimate --------------------------------------------------


async def test_every_criterion_gets_one_honest_answer_and_pushdowns_reach_the_body() -> None:
    criteria = (
        Criterion.language("de"),
        Criterion.namespaced("brave", "country", "DE"),
        Criterion.namespaced("brave", "safesearch", "strict"),
        Criterion.namespaced("brave", "search-context-size", "high"),
        Criterion.namespaced("brave", "max-completion-tokens", 500),
        Criterion.domains_include(["example.org"]),
        Criterion.namespaced("brave", "safesearch", "very"),
        Criterion.namespaced("other", "thing", 1),
    )
    result, transport = await _answer(SINGLE, request=AnswerRequest(question=QUESTION, criteria=criteria))

    body = transport.calls[0]["json_body"]
    assert isinstance(body, dict)
    assert body["language"] == "de"
    assert body["country"] == "DE"
    assert body["safesearch"] == "strict"
    assert body["web_search_options"] == {"search_context_size": "high"}
    assert body["max_completion_tokens"] == 500
    assert [(d.criterion_key, d.disposition) for d in result.dispositions] == [
        ("language", "pushdown"),
        (BRAVE_PARAM_COUNTRY, "pushdown"),
        (BRAVE_PARAM_SAFESEARCH, "pushdown"),
        (BRAVE_PARAM_SEARCH_CONTEXT_SIZE, "pushdown"),
        (BRAVE_PARAM_MAX_COMPLETION_TOKENS, "pushdown"),
        ("domains-include", "unsatisfied"),
        (BRAVE_PARAM_SAFESEARCH, "unsatisfied"),
        ("other:thing", "ignored-unknown"),
    ]


def test_the_estimate_is_one_call_and_one_search_and_money_only_from_a_stated_rate_card() -> None:
    """AR-C3: unpriced is not free; priced, the search plus the most a capped reply can bill."""
    transport = RecordedStreamingTransport((RecordedStream.of(SINGLE),))
    unpriced = _adapter(transport).estimate(AnswerRequest(question=QUESTION))
    assert (unpriced.calls, unpriced.provider_units, unpriced.provider_unit) == (1, Decimal(1), "brave:searches")
    assert unpriced.money == Decimal(0)

    priced = _adapter(transport, usd_per_search=Decimal("0.004"), usd_per_million_tokens=Decimal("5"))
    assert priced.estimate(AnswerRequest(question=QUESTION)).money == Decimal("0.004")
    capped = AnswerRequest(question=QUESTION, criteria=(Criterion.namespaced("brave", "max-completion-tokens", 300),))
    assert priced.estimate(capped).money == Decimal("0.0055")


def test_the_recordings_are_the_shipped_captures() -> None:
    """The fixture is read from the package, so it ships in the wheel with the code that reads it."""
    events = [line for line in SINGLE.decode().splitlines() if line.startswith("data: ")]
    assert events[-1] == "data: [DONE]"
    assert json.loads(INVALID_TOKEN_BODY)["error"]["code"] == "SUBSCRIPTION_TOKEN_INVALID"
