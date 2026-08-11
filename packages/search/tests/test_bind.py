"""Bind: the prose callers already read, the structure they could not.

The prose pin is a compatibility pin, and it is written literally: the exact
string the tool being replaced produced, for the same input. A regression here
is a prompt regression in every consumer at once, which no unit test elsewhere
would catch.
"""

from __future__ import annotations

from decimal import Decimal

from threetears.search.adapters.searxng import SearxngAdapter
from threetears.search.bind import (
    NO_RESULTS_PROSE,
    RenderedSearch,
    bind_candidate_set,
    bind_failure,
    bind_search,
    project_metadata,
    render_prose,
)
from threetears.search.contracts import (
    SEARCH_RESULTS_METADATA_KEY,
    CandidateSet,
    Criterion,
    QuotaExhausted,
    RateLimited,
    SearchRequest,
    SearchResultsMetadata,
    Spend,
)
from threetears.search.testing import ScriptedTransport, TransportScript
from _search_instances import CANDIDATE, CANDIDATE_SET, DISPOSITION, SPEND
from _searxng_payloads import MALFORMED_BODY, TWO_RESULTS_BODY, ZERO_RESULTS_BODY, body

#: byte-for-byte what ``_format_results`` produced for the two-result fixture.
#: The old renderer read SearXNG JSON directly; this one reads typed
#: candidates, and the output must not move.
EXPECTED_PROSE = (
    "1. Capybara\n"
    "   URL: https://example.org/capybara\n"
    "   The capybara is the largest living rodent.\n"
    "\n"
    "2. Capybara at dusk\n"
    "   URL: https://images.example.net/page/capy"
)


def _searxng(*steps: TransportScript) -> SearxngAdapter:
    """Build a SearXNG adapter over a scripted transport.

    :param steps: the exchanges the transport answers with
    :ptype steps: TransportScript
    :return: the adapter
    :rtype: SearxngAdapter
    """
    return SearxngAdapter(base_url="https://searx.example.org", transport=ScriptedTransport(steps))


# --- prose ----------------------------------------------------------------


async def test_prose_is_byte_identical_to_the_renderer_it_replaces() -> None:
    """The migration is structure-preserving: no consumer's prompt moves."""
    result = await _searxng(TransportScript(body=TWO_RESULTS_BODY)).search(SearchRequest(query="capybara"))

    assert render_prose(result) == EXPECTED_PROSE


def test_an_untitled_candidate_renders_the_old_placeholder() -> None:
    """The word the old renderer used, kept for the same reason."""
    untitled = CANDIDATE.model_copy(update={"title": None, "snippet": None})
    prose = render_prose(CandidateSet(candidates=(untitled,)))

    assert prose == f"1. Untitled\n   URL: {untitled.identity}"


def test_zero_results_renders_the_success_sentence() -> None:
    """SR-J2: found-nothing is a result, and it reads like one."""
    assert render_prose(CandidateSet()) == NO_RESULTS_PROSE


def test_prose_is_bounded_because_it_goes_into_a_context_window() -> None:
    """The projection carries everything; the prose carries what fits."""
    many = CandidateSet(
        candidates=tuple(
            CANDIDATE.model_copy(update={"identity": f"https://example.org/{index}"}) for index in range(25)
        )
    )
    prose = render_prose(many, max_candidates=3)

    assert prose.count("   URL: ") == 3
    assert len(project_metadata("q", many)[SEARCH_RESULTS_METADATA_KEY]["candidates"]) == 25


async def test_an_unsatisfied_criterion_is_named_in_the_prose() -> None:
    """A model handed results has no other way to learn what was ignored."""
    result = await _searxng(TransportScript(body=TWO_RESULTS_BODY)).search(
        SearchRequest(query="capybara", criteria=(Criterion.rights_class("public-domain"),))
    )
    prose = render_prose(result)

    assert prose.startswith(EXPECTED_PROSE)
    assert "rights-class was not applied (unsatisfied)" in prose


async def test_a_provider_degradation_is_named_in_the_prose() -> None:
    """An outage that reads as an empty topic is the reasoning error P8 targets."""
    result = await _searxng(TransportScript(body=body(unresponsive=(["wikidata", "timeout"],)))).search(
        SearchRequest(query="capybara")
    )
    prose = render_prose(result)

    assert "wikidata" in prose


async def test_a_fully_honoured_request_gets_no_note() -> None:
    """Nothing to report means nothing appended -- the note is not decoration."""
    result = await _searxng(TransportScript(body=TWO_RESULTS_BODY)).search(
        SearchRequest(query="capybara", criteria=(Criterion.language("en"),))
    )

    assert render_prose(result) == EXPECTED_PROSE


# --- the metadata projection (D22) ---------------------------------------


def test_the_projection_rides_one_named_key() -> None:
    """D22: structure crosses the border under SEARCH_RESULTS_METADATA_KEY."""
    metadata = project_metadata("capybara habitat range", CANDIDATE_SET)

    assert list(metadata) == [SEARCH_RESULTS_METADATA_KEY]
    payload = metadata[SEARCH_RESULTS_METADATA_KEY]
    assert payload["schema_version"] == 1
    assert payload["query"] == "capybara habitat range"


def test_the_projection_round_trips_back_to_typed_candidates() -> None:
    """A consumer reads structure rather than re-parsing prose."""
    metadata = project_metadata("capybara habitat range", CANDIDATE_SET)
    rebuilt = SearchResultsMetadata.from_metadata(metadata[SEARCH_RESULTS_METADATA_KEY])

    assert rebuilt.candidates == CANDIDATE_SET.candidates
    assert rebuilt.dispositions == (DISPOSITION,)
    assert rebuilt.spend == SPEND
    assert rebuilt.notices == CANDIDATE_SET.notices


def test_the_projection_carries_every_disposition_and_notice() -> None:
    """SR-B2 and P8 survive the border, not just the in-process call."""
    payload = project_metadata("q", CANDIDATE_SET)[SEARCH_RESULTS_METADATA_KEY]

    assert len(payload["dispositions"]) == 1
    assert payload["notices"] == list(CANDIDATE_SET.notices)


# --- failures (D10) -------------------------------------------------------


def test_a_failure_renders_as_a_failed_result_carrying_spend() -> None:
    """SR-E3 across the border: a broken call still says what it cost."""
    rendered = bind_failure("capybara", RateLimited("429 from the instance", spend=SPEND, retry_after_seconds=30.0))

    assert rendered.success is False
    assert rendered.error == rendered.content
    payload = rendered.metadata[SEARCH_RESULTS_METADATA_KEY]
    assert payload["failure"]["failure_class"] == "rate-limited"
    assert payload["failure"]["retry_after_seconds"] == 30.0
    assert Decimal(payload["spend"]["money"]) == SPEND.money
    assert payload["spend"]["calls"] == SPEND.calls


def test_a_failure_names_its_class_rather_than_prefixing_a_string() -> None:
    """String-prefix error detection is one of the defects being retired."""
    rendered = bind_failure("capybara", QuotaExhausted("out of credit", spend=Spend()))

    assert "quota-exhausted" in rendered.content
    assert not rendered.content.startswith("[TOOL ERROR]")


def test_a_failure_with_remediation_puts_the_fix_in_the_prose() -> None:
    """The reader of that sentence is usually the person who can fix it."""
    rendered = bind_failure(
        "capybara",
        RateLimited("429", spend=Spend(), remediation="lower the configured rate for this instance"),
    )

    assert "lower the configured rate" in rendered.content


def test_the_failed_projection_reconstructs_the_typed_exception() -> None:
    """The far side branches on the class, having received no exception."""
    rendered = bind_failure("capybara", RateLimited("429", spend=SPEND, retry_after_seconds=12.5))
    payload = SearchResultsMetadata.from_metadata(rendered.metadata[SEARCH_RESULTS_METADATA_KEY])

    assert payload.failure is not None
    rebuilt = payload.failure.to_failure()
    assert isinstance(rebuilt, RateLimited)
    assert rebuilt.retry_after_seconds == 12.5


# --- the whole path -------------------------------------------------------


async def test_bind_search_renders_a_success_in_both_registers() -> None:
    """One path, both faces: prose and structure cannot disagree (check 14)."""
    rendered = await bind_search(
        SearchRequest(query="capybara"), provider=_searxng(TransportScript(body=TWO_RESULTS_BODY))
    )

    assert isinstance(rendered, RenderedSearch)
    assert rendered.success is True
    assert rendered.error is None
    assert rendered.content.startswith("1. Capybara")
    payload = rendered.metadata[SEARCH_RESULTS_METADATA_KEY]
    assert len(payload["candidates"]) == 2
    assert payload["failure"] is None


async def test_bind_search_treats_zero_results_as_a_success() -> None:
    """SR-J2 all the way out to the rendered result."""
    rendered = await bind_search(
        SearchRequest(query="nothing"), provider=_searxng(TransportScript(body=ZERO_RESULTS_BODY))
    )

    assert rendered.success is True
    assert rendered.content == NO_RESULTS_PROSE
    assert rendered.error is None


async def test_nothing_raises_past_bind_for_a_provider_failure() -> None:
    """D10: an exception cannot cross the wire, so it never reaches it."""
    rendered = await bind_search(
        SearchRequest(query="capybara"), provider=_searxng(TransportScript(status_code=403, body=b""))
    )

    assert rendered.success is False
    assert "search.formats" in rendered.content
    assert rendered.metadata[SEARCH_RESULTS_METADATA_KEY]["failure"]["failure_class"] == "malformed-response"


async def test_nothing_raises_past_bind_for_a_malformed_payload() -> None:
    """The second-most-likely real failure, through the same guarantee."""
    rendered = await bind_search(
        SearchRequest(query="capybara"), provider=_searxng(TransportScript(body=MALFORMED_BODY))
    )

    assert rendered.success is False
    assert rendered.metadata[SEARCH_RESULTS_METADATA_KEY]["spend"]["calls"] == 1


async def test_nothing_raises_past_bind_for_a_local_cap_refusal() -> None:
    """A refusal Call raised before contact still arrives as a failed result."""
    rendered = await bind_search(
        SearchRequest(query="capybara", criteria=(Criterion.max_results(10_000),)),
        provider=_searxng(TransportScript(body=TWO_RESULTS_BODY)),
    )

    assert rendered.success is False
    payload = rendered.metadata[SEARCH_RESULTS_METADATA_KEY]
    assert payload["failure"]["failure_class"] == "local-cap-exceeded"
    assert payload["failure"]["scope"] == "max-results"


def test_bind_candidate_set_and_project_metadata_agree() -> None:
    """The two entry points are one projection, not two that can drift."""
    rendered = bind_candidate_set("capybara habitat range", CANDIDATE_SET)

    assert rendered.metadata == project_metadata("capybara habitat range", CANDIDATE_SET)
