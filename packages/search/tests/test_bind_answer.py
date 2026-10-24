"""Bind for answers: the first line, the sources, the metadata projection, and failures (§10.4).

The prose is what a model reads, so its first line must say whose synthesis
follows (AR-G2) and whether anything is cited (AR-A3); the projection is what
a program reads, so it must round-trip and refuse a version it does not know.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from threetears.search.adapters.brave_answers import BraveAnswersAdapter
from threetears.search.bind_answer import bind_answer, project_answer_failure_metadata, project_answer_metadata
from threetears.search.contracts import (
    ANSWER_METADATA_KEY,
    ANSWER_SCHEMA_VERSION,
    SEARCH_RESULTS_METADATA_KEY,
    AnswerMetadata,
    AnswerRequest,
    BudgetDecision,
    Criterion,
    QuotaExhausted,
    Spend,
)
from threetears.search.testing import FakeBudgetPort
from packages.search.tests.recorded_streams import RecordedStream, RecordedStreamingTransport
from packages.search.tests.brave_answers_payloads import (
    INVALID_TOKEN_BODY,
    SINGLE,
    SINGLE_CITATIONS,
    SINGLE_TOTAL_COST,
    stream,
    usage_tag,
)
from packages.search.tests.search_instances import FAILURE_RECORD, GROUNDED_ANSWER

QUESTION = "what is the largest rodent?"


def _brave(*streams: RecordedStream) -> BraveAnswersAdapter:
    """A Brave adapter replaying ``streams``.

    :param streams: the recordings
    :ptype streams: RecordedStream
    :return: the adapter
    :rtype: BraveAnswersAdapter
    """
    return BraveAnswersAdapter(api_key="k", transport=RecordedStreamingTransport(streams))


async def test_the_first_line_says_whose_synthesis_follows_and_how_many_sources_it_cites() -> None:
    """AR-G2: the attribution leads, so a model cannot take the provider's prose for retrieved text."""
    rendered = await bind_answer(AnswerRequest(question=QUESTION), provider=_brave(RecordedStream.of(SINGLE)))

    assert rendered.success and rendered.error is None
    lines = rendered.content.splitlines()
    assert lines[0] == f"Answer from brave, {SINGLE_CITATIONS} sources:"
    assert lines[2].startswith("The **capybara**"), "the text follows the first line, unchanged"
    assert "Sources:" in lines
    sources = lines[lines.index("Sources:") + 1 :]
    assert sources[0] == "1. https://grokipedia.com/page/List_of_largest_rodents"
    assert sources[1] == "   no date given", "each source carries the search prose's facts line"
    assert sources[2].startswith("   # List of largest rodents"), "the snippet sits under its own source"
    assert f"{SINGLE_CITATIONS}. https://" in rendered.content


async def test_an_uncited_answer_says_so_in_its_first_line_not_a_footnote() -> None:
    """AR-A3."""
    rendered = await bind_answer(
        AnswerRequest(question=QUESTION), provider=_brave(RecordedStream.of(stream("Capybaras graze.", usage_tag())))
    )

    first = rendered.content.splitlines()[0]
    assert first.startswith("Answer from brave, citing no sources")
    assert "Sources:" not in rendered.content
    assert "What this answer could not do:" in rendered.content


async def test_an_unhonoured_criterion_and_a_notice_are_named_for_the_prose_reader() -> None:
    request = AnswerRequest(question=QUESTION, criteria=(Criterion.domains_include(["example.org"]),))
    body = stream("Capybaras graze.", '<entities>{"x": 1}</entities>', usage_tag())
    rendered = await bind_answer(request, provider=_brave(RecordedStream.of(body)))

    note = rendered.content.split("What this answer could not do:\n", 1)[1]
    assert "- the domains-include constraint was not applied:" in note
    assert "<entities>" in note


async def test_the_metadata_round_trips_the_whole_answer_under_its_own_key() -> None:
    """AD8: the answer rides its own key, never the search key, at the current version."""
    rendered = await bind_answer(AnswerRequest(question=QUESTION), provider=_brave(RecordedStream.of(SINGLE)))

    assert set(rendered.metadata) == {ANSWER_METADATA_KEY}
    assert SEARCH_RESULTS_METADATA_KEY not in rendered.metadata
    payload = AnswerMetadata.from_metadata(rendered.metadata[ANSWER_METADATA_KEY])
    assert payload.schema_version == ANSWER_SCHEMA_VERSION
    assert payload.question == QUESTION
    assert payload.failure is None
    assert payload.answer is not None and len(payload.answer.sources) == SINGLE_CITATIONS
    assert payload.spend.money == SINGLE_TOTAL_COST


async def test_a_failure_renders_as_a_failed_result_that_still_accounts() -> None:
    """D10: nothing raises past bind; the failure and its spend ride the same key."""
    refused = RecordedStream.of(INVALID_TOKEN_BODY, status_code=422, headers={})
    rendered = await bind_answer(AnswerRequest(question=QUESTION), provider=_brave(refused))

    assert not rendered.success
    assert rendered.content == rendered.error
    assert rendered.content.startswith("answer failed (auth-failed):")
    assert "Brave rejected the API key" in rendered.content
    payload = AnswerMetadata.from_metadata(rendered.metadata[ANSWER_METADATA_KEY])
    assert payload.answer is None
    assert payload.failure is not None and payload.failure.failure_class == "auth-failed"
    assert payload.spend.calls == 1


async def test_a_budget_refusal_is_rendered_with_the_refusing_scope_s_spend() -> None:
    consumed = Spend(money=Decimal("5"), calls=9)
    budget = FakeBudgetPort(BudgetDecision(allowed=False, scope="run:1", consumed=consumed))
    rendered = await bind_answer(
        AnswerRequest(question=QUESTION), provider=_brave(RecordedStream.of(SINGLE)), budget=budget
    )

    payload = AnswerMetadata.from_metadata(rendered.metadata[ANSWER_METADATA_KEY])
    assert payload.failure is not None and payload.failure.scope == "run:1"
    assert payload.spend == consumed


async def test_an_escape_past_the_taxonomy_is_still_rendered() -> None:
    class _Broken(BraveAnswersAdapter):
        def estimate(self, request: AnswerRequest) -> Spend:
            raise RuntimeError("estimate broke")

    adapter = _Broken(api_key="k", transport=RecordedStreamingTransport((RecordedStream.of(SINGLE),)))
    rendered = await bind_answer(AnswerRequest(question=QUESTION), provider=adapter, budget=FakeBudgetPort())

    assert not rendered.success
    assert "unmapped RuntimeError" in rendered.content


def test_the_projections_are_the_one_construction_site_and_stamp_their_version() -> None:
    success = project_answer_metadata(QUESTION, GROUNDED_ANSWER)[ANSWER_METADATA_KEY]
    failure = project_answer_failure_metadata(QUESTION, FAILURE_RECORD)[ANSWER_METADATA_KEY]

    assert success["schema_version"] == failure["schema_version"] == ANSWER_SCHEMA_VERSION
    assert AnswerMetadata.from_metadata(success).answer == GROUNDED_ANSWER
    assert AnswerMetadata.from_metadata(failure).failure == FAILURE_RECORD


def test_a_newer_payload_is_refused_by_version_before_its_shape_is_read() -> None:
    """D13: the refusal names the version, not whatever field a newer writer added."""
    payload = project_answer_metadata(QUESTION, GROUNDED_ANSWER)[ANSWER_METADATA_KEY]
    newer = {**payload, "schema_version": ANSWER_SCHEMA_VERSION + 1, "field_from_the_future": True}

    with pytest.raises(ValueError, match=f"schema_version {ANSWER_SCHEMA_VERSION + 1}"):
        AnswerMetadata.from_metadata(newer)


def test_a_payload_is_exactly_one_of_an_answer_and_a_failure() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        AnswerMetadata(question=QUESTION)
    with pytest.raises(ValidationError, match="exactly one"):
        AnswerMetadata(question=QUESTION, answer=GROUNDED_ANSWER, failure=FAILURE_RECORD)


def test_a_provider_quota_refusal_and_a_local_one_stay_distinct_in_the_record() -> None:
    record = QuotaExhausted("credit limit", spend=Spend(calls=1)).to_record()
    payload = AnswerMetadata.from_metadata(project_answer_failure_metadata(QUESTION, record)[ANSWER_METADATA_KEY])
    assert payload.failure is not None and payload.failure.failure_class == "quota-exhausted"
