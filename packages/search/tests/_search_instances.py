"""Fully-populated instances of every wire-crossing contract type.

Shared by the round-trip and canonical-serialization suites so "every
contract type" means one list, checked for completeness against the
package's exported surface rather than maintained by hand.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from threetears.search.contracts import (
    FACET_EXTRACTION_STATUS,
    FACET_HAS_DOWNLOADABLE_DATA,
    FACET_MEDIA_CATEGORY,
    PRODUCER_API_PROVIDER,
    SCALE_UNIT_INTERVAL,
    Candidate,
    CandidateSet,
    ContentSlot,
    Criterion,
    CriterionDisposition,
    FailureRecord,
    Locator,
    Provenance,
    ScoreEntry,
    SearchRequest,
    SearchResultsMetadata,
    Spend,
)

RETRIEVED_AT = datetime(2026, 8, 10, 12, 30, 0, tzinfo=UTC)

SPEND = Spend(
    money=Decimal("0.02"),
    currency="USD",
    wall_clock_seconds=1.25,
    calls=2,
    provider_units=Decimal("2"),
    bytes_transferred=40960,
)

PROVENANCE = Provenance(
    query="capybara habitat range",
    provider_instance="searxng-main",
    provider_ids={"engine": "duckduckgo", "positions": "1"},
    retrieved_at=RETRIEVED_AT,
    egress="warp",
    producer=PRODUCER_API_PROVIDER,
)

SCORE = ScoreEntry.provider_native(
    name="relevance",
    value=0.87,
    scale=SCALE_UNIT_INTERVAL,
    provider_instance="searxng-main",
)

CRITERION = Criterion.time_range(
    start=datetime(2026, 1, 1, tzinfo=UTC),
    end=datetime(2026, 8, 1, tzinfo=UTC),
)

DISPOSITION = CriterionDisposition(
    criterion_key="time-range",
    disposition="pushdown",
    detail="mapped to provider date filters; absolute dates win over relative ranges",
)

LOCATOR = Locator(url="https://example.org/capybaras", rel="canonical")

CONTENT = ContentSlot(
    text="Capybaras range across most of South America...",
    origin="provider-response",
    mime_type="text/html",
    size_bytes=18231,
)

CANDIDATE = Candidate(
    identity="https://example.org/capybaras",
    locators=(LOCATOR, Locator(url="https://example.org/capybaras.pdf", rel="direct-file")),
    provenance=PROVENANCE,
    title="Capybara habitats",
    snippet="Where capybaras live, and why.",
    scores=(SCORE,),
    fidelity_available="content",
    fidelity_achieved="content",
    content=CONTENT,
    facets={FACET_MEDIA_CATEGORY: "document", FACET_EXTRACTION_STATUS: "complete", FACET_HAS_DOWNLOADABLE_DATA: True},
)

CANDIDATE_SET = CandidateSet(
    candidates=(CANDIDATE,),
    dispositions=(DISPOSITION,),
    spend=SPEND,
)

REQUEST = SearchRequest(
    query="capybara habitat range",
    criteria=(CRITERION, Criterion.language("en")),
    fidelity="content",
    record=True,
    budget_scope_tags=("persona:capy", "run:eval-042"),
)

FAILURE_RECORD = FailureRecord(
    failure_class="rate-limited",
    message="provider returned 429",
    spend=SPEND,
    provider_instance="searxng-main",
    remediation="lower the configured rate for this instance",
    retry_after_seconds=30.0,
)

METADATA = SearchResultsMetadata.from_candidate_set(
    query="capybara habitat range",
    candidate_set=CANDIDATE_SET,
)

#: one fully-populated instance per exported wire type. The round-trip
#: suite asserts this list covers every exported ContractModel subclass.
ALL_INSTANCES = [
    SPEND,
    PROVENANCE,
    SCORE,
    CRITERION,
    DISPOSITION,
    LOCATOR,
    CONTENT,
    CANDIDATE,
    CANDIDATE_SET,
    REQUEST,
    FAILURE_RECORD,
    METADATA,
]
