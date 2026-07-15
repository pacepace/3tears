"""Unit tests for threetears.scrape.enrichment -- the secondary, separate LLM pass
(mocking approach mirrors tests/unit/test_query_agent_matching.py's
create_chat_model pattern; real sidecar + real LLM proof lives in
tests/e2e/test_scrape_enrichment_live.py).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from threetears.scrape.collections import ScrapeExtractionCollection
from threetears.scrape.enrichment import _EnrichmentResult, enrich_extraction, run_enrichment
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

_test_registry = CollectionRegistry()
_test_config = DefaultCoreConfig()


def get_registry() -> CollectionRegistry:
    return _test_registry


def get_config() -> DefaultCoreConfig:
    return _test_config


_PAGE_HTML = "<html><body><p>Acme Corp is closing its plant in Q3.</p></body></html>"
_STRUCTURED_FIELDS = {"employer": "Acme Corp", "affected_count": 42}


def _fake_structured_model(result=None, *, side_effect=None):
    ainvoke_mock = AsyncMock(return_value=result, side_effect=side_effect)
    structured = SimpleNamespace(ainvoke=ainvoke_mock)
    return SimpleNamespace(with_structured_output=lambda schema, **kwargs: structured), ainvoke_mock


class TestRunEnrichment:
    async def test_success_returns_notes(self):
        parsed = _EnrichmentResult(notes={"context": "closure tied to Q3 restructuring"})
        fake_model, ainvoke_mock = _fake_structured_model(parsed)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            notes = await run_enrichment(_PAGE_HTML, _STRUCTURED_FIELDS, api_key="k")
        assert notes == {"context": "closure tied to Q3 restructuring"}
        assert ainvoke_mock.await_count == 1

    async def test_retries_before_succeeding(self):
        parsed = _EnrichmentResult(notes={"note": "ok"})
        fake_model, ainvoke_mock = _fake_structured_model(side_effect=[RuntimeError("transient"), parsed])
        with (
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model),
            patch("threetears.scrape.llm_retry.asyncio.sleep", AsyncMock()),
        ):
            notes = await run_enrichment(_PAGE_HTML, _STRUCTURED_FIELDS, api_key="k")
        assert ainvoke_mock.await_count == 2
        assert notes == {"note": "ok"}

    async def test_total_failure_returns_empty_dict_not_a_crash(self):
        fake_model, _ = _fake_structured_model(side_effect=RuntimeError("boom"))
        with (
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model),
            patch("threetears.scrape.llm_retry.asyncio.sleep", AsyncMock()),
        ):
            notes = await run_enrichment(_PAGE_HTML, _STRUCTURED_FIELDS, api_key="k")
        assert notes == {}

    async def test_genuinely_nothing_noteworthy_returns_empty_dict(self):
        parsed = _EnrichmentResult(notes={})
        fake_model, _ = _fake_structured_model(parsed)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            notes = await run_enrichment(_PAGE_HTML, _STRUCTURED_FIELDS, api_key="k")
        assert notes == {}


class TestEnrichExtraction:
    async def test_enrichment_notes_stored_separately_from_structured_fields(self):
        extraction_collection = ScrapeExtractionCollection(get_registry(), get_config(), nats_client=None)
        original = extraction_collection.create(
            {
                "target_id": "warn_act_ca",
                "source_url": "https://edd.ca.gov/warn",
                "structured_fields": _STRUCTURED_FIELDS,
            }
        )
        await extraction_collection.save_entity(original)
        assert original.enrichment_notes is None  # Chunk 1's pre-enrichment default

        parsed = _EnrichmentResult(notes={"context": "closure tied to Q3 restructuring"})
        fake_model, _ = _fake_structured_model(parsed)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            enriched = await enrich_extraction(
                original,
                _PAGE_HTML,
                extraction_collection=extraction_collection,
                api_key="k",
            )

        # Same row, updated in place -- not a second extraction for the same fetch.
        assert enriched.id == original.id
        assert enriched.enrichment_notes == {"context": "closure tied to Q3 restructuring"}
        assert enriched.structured_fields == _STRUCTURED_FIELDS  # untouched by the enrichment pass
        assert enriched.enrichment_notes != enriched.structured_fields

        # Re-fetch from the collection to confirm the write actually persisted, not just
        # the returned in-memory object.
        refetched = await extraction_collection.get(original.id)
        assert refetched is not None
        assert refetched.enrichment_notes == {"context": "closure tied to Q3 restructuring"}
        assert refetched.structured_fields == _STRUCTURED_FIELDS

    async def test_enrichment_failure_still_persists_empty_notes_not_structured_fields(self):
        extraction_collection = ScrapeExtractionCollection(get_registry(), get_config(), nats_client=None)
        original = extraction_collection.create(
            {
                "target_id": "warn_act_ca",
                "source_url": "https://edd.ca.gov/warn",
                "structured_fields": _STRUCTURED_FIELDS,
            }
        )
        await extraction_collection.save_entity(original)

        fake_model, _ = _fake_structured_model(side_effect=RuntimeError("boom"))
        with (
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model),
            patch("threetears.scrape.llm_retry.asyncio.sleep", AsyncMock()),
        ):
            enriched = await enrich_extraction(
                original,
                _PAGE_HTML,
                extraction_collection=extraction_collection,
                api_key="k",
            )

        assert enriched.enrichment_notes == {}
        assert enriched.structured_fields == _STRUCTURED_FIELDS
