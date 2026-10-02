"""Unit tests for threetears.scrape.enrichment -- the secondary, separate LLM pass.

Every LLM call is mocked at ``create_chat_model``. This package ships no
live-LLM suite of its own: exercising this path against a real model is a
consuming application's job, since it owns the API keys and the target.

The contract these pin: a pass whose every attempt failed is recorded as FAILED, with the
reason, and never as ``{}``. ``{}`` means the model answered and had nothing to add, and a
reader must be able to tell the two apart from the stored row alone.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, patch

import pytest

from threetears.scrape.collections import ENRICHMENT_STATUSES, ScrapeExtractionCollection
from threetears.scrape.migrations import LEGACY_EMPTY_ENRICHMENT_FAILURE
from uuid_utils import uuid7
from threetears.scrape.enrichment import EnrichmentFailedError, enrich_extraction, run_enrichment
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from packages.scrape.tests.structured_output_fakes import fake_structured_model

_test_registry = CollectionRegistry()
_test_config = DefaultCoreConfig()


def get_registry() -> CollectionRegistry:
    return _test_registry


def get_config() -> DefaultCoreConfig:
    return _test_config


_PAGE_HTML = "<html><body><p>Acme Corp is closing its plant in Q3.</p></body></html>"
_STRUCTURED_FIELDS = {"employer": "Acme Corp", "affected_count": 42}


async def _persisted_extraction(collection: ScrapeExtractionCollection):
    original = collection.create(
        {
            "target_id": "warn_act_ca",
            "source_url": "https://edd.ca.gov/warn",
            "structured_fields": _STRUCTURED_FIELDS,
        }
    )
    await collection.save_entity(original)
    return original


def _errors(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= logging.ERROR]


class TestRunEnrichment:
    async def test_success_returns_notes(self):
        parsed = {"notes": {"context": "closure tied to Q3 restructuring"}}
        fake_model, ainvoke_mock = fake_structured_model(parsed)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            notes = await run_enrichment(_PAGE_HTML, _STRUCTURED_FIELDS, api_key="k")
        assert notes == {"context": "closure tied to Q3 restructuring"}
        assert ainvoke_mock.await_count == 1

    async def test_retries_before_succeeding(self):
        parsed = {"notes": {"note": "ok"}}
        fake_model, ainvoke_mock = fake_structured_model(side_effect=[RuntimeError("transient"), parsed])
        with (
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model),
            patch("threetears.scrape.llm_retry.asyncio.sleep", AsyncMock()),
        ):
            notes = await run_enrichment(_PAGE_HTML, _STRUCTURED_FIELDS, api_key="k")
        assert ainvoke_mock.await_count == 2
        assert notes == {"note": "ok"}

    async def test_total_failure_raises_with_the_reason_never_returns_empty_notes(self):
        boom = RuntimeError("boom")
        fake_model, ainvoke_mock = fake_structured_model(side_effect=boom)
        with (
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model),
            patch("threetears.scrape.llm_retry.asyncio.sleep", AsyncMock()),
            pytest.raises(EnrichmentFailedError) as exc_info,
        ):
            await run_enrichment(_PAGE_HTML, _STRUCTURED_FIELDS, api_key="k", attempts=3)
        assert ainvoke_mock.await_count == 3
        assert exc_info.value.attempts == 3
        assert exc_info.value.reason == "RuntimeError: boom"
        assert exc_info.value.__cause__ is boom

    async def test_total_failure_is_logged_once_with_its_cause(self, caplog: pytest.LogCaptureFixture):
        fake_model, _ = fake_structured_model(side_effect=RuntimeError("provider down"))
        with (
            caplog.at_level(logging.WARNING),
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model),
            patch("threetears.scrape.llm_retry.asyncio.sleep", AsyncMock()),
            pytest.raises(EnrichmentFailedError),
        ):
            await run_enrichment(_PAGE_HTML, _STRUCTURED_FIELDS, api_key="k", attempts=2)
        # One failure, one ERROR line: the retry helper logs it where it happens, with the
        # cause, and the enrichment pass does not log it a second time.
        errors = _errors(caplog)
        assert len(errors) == 1
        assert errors[0].name == "threetears.scrape.llm_retry"
        assert "scrape enrichment" in errors[0].getMessage()
        assert "RuntimeError: provider down" in errors[0].getMessage()

    async def test_cancellation_propagates_and_is_not_retried(self):
        fake_model, ainvoke_mock = fake_structured_model(side_effect=asyncio.CancelledError())
        with (
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model),
            patch("threetears.scrape.llm_retry.asyncio.sleep", AsyncMock()),
            pytest.raises(asyncio.CancelledError),
        ):
            await run_enrichment(_PAGE_HTML, _STRUCTURED_FIELDS, api_key="k")
        assert ainvoke_mock.await_count == 1

    async def test_genuinely_nothing_noteworthy_returns_empty_dict(self):
        parsed = {"notes": {}}
        fake_model, _ = fake_structured_model(parsed)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            notes = await run_enrichment(_PAGE_HTML, _STRUCTURED_FIELDS, api_key="k")
        assert notes == {}


class TestEnrichExtraction:
    async def test_a_row_never_enriched_says_so(self):
        extraction_collection = ScrapeExtractionCollection(get_registry(), get_config(), nats_client=None)
        original = await _persisted_extraction(extraction_collection)
        assert original.enrichment_notes is None
        assert original.enrichment_status is None
        assert original.enrichment_failure is None

    async def test_enrichment_notes_stored_separately_from_structured_fields(self):
        extraction_collection = ScrapeExtractionCollection(get_registry(), get_config(), nats_client=None)
        original = await _persisted_extraction(extraction_collection)

        parsed = {"notes": {"context": "closure tied to Q3 restructuring"}}
        fake_model, _ = fake_structured_model(parsed)
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            enriched = await enrich_extraction(
                original,
                _PAGE_HTML,
                extraction_collection=extraction_collection,
                api_key="k",
            )

        # Same row, updated in place -- not a second extraction for the same fetch.
        assert enriched.id == original.id
        assert enriched.enrichment_status == "enriched"
        assert enriched.enrichment_failure is None
        assert enriched.enrichment_notes == {"context": "closure tied to Q3 restructuring"}
        assert enriched.structured_fields == _STRUCTURED_FIELDS  # untouched by the enrichment pass
        assert enriched.enrichment_notes != enriched.structured_fields

        # Re-fetch from the collection to confirm the write actually persisted, not just
        # the returned in-memory object.
        refetched = await extraction_collection.get(original.id)
        assert refetched is not None
        assert refetched.enrichment_status == "enriched"
        assert refetched.enrichment_notes == {"context": "closure tied to Q3 restructuring"}
        assert refetched.structured_fields == _STRUCTURED_FIELDS

    async def test_nothing_to_add_is_recorded_as_enriched_with_empty_notes(self):
        extraction_collection = ScrapeExtractionCollection(get_registry(), get_config(), nats_client=None)
        original = await _persisted_extraction(extraction_collection)

        fake_model, _ = fake_structured_model({"notes": {}})
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model):
            enriched = await enrich_extraction(
                original, _PAGE_HTML, extraction_collection=extraction_collection, api_key="k"
            )

        assert enriched.enrichment_status == "enriched"
        assert enriched.enrichment_notes == {}
        assert enriched.enrichment_failure is None

    async def test_total_failure_is_recorded_as_failed_with_the_reason_never_as_empty_notes(self):
        extraction_collection = ScrapeExtractionCollection(get_registry(), get_config(), nats_client=None)
        original = await _persisted_extraction(extraction_collection)

        fake_model, _ = fake_structured_model(side_effect=RuntimeError("boom"))
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

        assert enriched.id == original.id
        assert enriched.enrichment_status == "failed"
        assert enriched.enrichment_failure == "RuntimeError: boom"
        assert enriched.enrichment_notes is None
        assert enriched.structured_fields == _STRUCTURED_FIELDS

        refetched = await extraction_collection.get(original.id)
        assert refetched is not None
        assert refetched.enrichment_status == "failed"
        assert refetched.enrichment_failure == "RuntimeError: boom"
        assert refetched.enrichment_notes is None

    async def test_a_failed_row_can_be_enriched_again(self):
        extraction_collection = ScrapeExtractionCollection(get_registry(), get_config(), nats_client=None)
        original = await _persisted_extraction(extraction_collection)

        failing_model, _ = fake_structured_model(side_effect=RuntimeError("boom"))
        with (
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=failing_model),
            patch("threetears.scrape.llm_retry.asyncio.sleep", AsyncMock()),
        ):
            failed = await enrich_extraction(
                original, _PAGE_HTML, extraction_collection=extraction_collection, api_key="k"
            )
        assert failed.enrichment_status == "failed"

        working_model, _ = fake_structured_model({"notes": {"context": "second try"}})
        with patch("threetears.scrape.llm_retry.create_chat_model", return_value=working_model):
            retried = await enrich_extraction(
                failed, _PAGE_HTML, extraction_collection=extraction_collection, api_key="k"
            )

        assert retried.id == original.id
        assert retried.enrichment_status == "enriched"
        assert retried.enrichment_failure is None
        assert retried.enrichment_notes == {"context": "second try"}

    async def test_cancellation_propagates_and_persists_nothing(self):
        extraction_collection = ScrapeExtractionCollection(get_registry(), get_config(), nats_client=None)
        original = await _persisted_extraction(extraction_collection)

        fake_model, _ = fake_structured_model(side_effect=asyncio.CancelledError())
        with (
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=fake_model),
            pytest.raises(asyncio.CancelledError),
        ):
            await enrich_extraction(original, _PAGE_HTML, extraction_collection=extraction_collection, api_key="k")

        refetched = await extraction_collection.get(original.id)
        assert refetched is not None
        assert refetched.enrichment_status is None
        assert refetched.enrichment_notes is None


class TestEnrichmentStatusField:
    def test_the_vocabulary_is_exactly_enriched_and_failed(self):
        assert ENRICHMENT_STATUSES == frozenset({"enriched", "failed"})

    def test_an_unknown_stored_status_is_refused_rather_than_read_as_either(self):
        extraction_collection = ScrapeExtractionCollection(get_registry(), get_config(), nats_client=None)
        row = extraction_collection.create(
            {"target_id": "warn_act_ca", "source_url": "https://edd.ca.gov/warn", "enrichment_status": "partial"}
        )
        with pytest.raises(ValueError, match="partial"):
            _ = row.enrichment_status


class TestARowWrittenBeforeTheStatusExisted:
    """A copy of a row cached in L1 or L2 before migration v013 ran is out of the migration's
    reach. It is read through the same constructor every tier uses, which applies v013's rules,
    so it never reads as "never ran" beside ``{}``.

    Each row goes in through the collection's own store and comes back out through ``get``,
    the read path a cached copy takes.
    """

    @staticmethod
    async def _read_back(raw_notes):
        collection = ScrapeExtractionCollection(get_registry(), get_config(), nats_client=None)
        row_id = str(uuid7())
        await collection.save_to_store(
            {
                "id": row_id,
                "target_id": "warn_act_ca",
                "source_url": "https://edd.ca.gov/warn",
                "enrichment_notes": raw_notes,
            }
        )
        stored = await collection.get(row_id)
        assert stored is not None
        return stored

    async def test_empty_notes_with_no_status_read_as_failed_with_the_legacy_reason(self):
        stored = await self._read_back({})
        assert stored.enrichment_status == "failed"
        assert stored.enrichment_notes is None
        assert stored.enrichment_failure == LEGACY_EMPTY_ENRICHMENT_FAILURE

    async def test_empty_notes_as_json_text_read_as_failed(self):
        stored = await self._read_back("{}")
        assert stored.enrichment_status == "failed"
        assert stored.enrichment_failure == LEGACY_EMPTY_ENRICHMENT_FAILURE

    async def test_empty_notes_double_encoded_read_as_failed(self):
        stored = await self._read_back('"{}"')
        assert stored.enrichment_status == "failed"
        assert stored.enrichment_failure == LEGACY_EMPTY_ENRICHMENT_FAILURE

    async def test_notes_with_content_and_no_status_read_as_enriched(self):
        stored = await self._read_back({"context": "q3"})
        assert stored.enrichment_status == "enriched"
        assert stored.enrichment_notes == {"context": "q3"}
        assert stored.enrichment_failure is None

    async def test_no_notes_and_no_status_still_reads_as_never_ran(self):
        stored = await self._read_back(None)
        assert stored.enrichment_status is None
        assert stored.enrichment_notes is None
        assert stored.enrichment_failure is None

    def test_a_row_that_has_a_status_is_never_translated(self):
        collection = ScrapeExtractionCollection(get_registry(), get_config(), nats_client=None)
        row = collection.create({"target_id": "warn_act_ca", "enrichment_notes": {}, "enrichment_status": "enriched"})
        assert row.enrichment_status == "enriched"
        assert row.enrichment_notes == {}
        assert row.enrichment_failure is None
