"""tests for LLM usage tracking and purpose classification."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timezone
from decimal import Decimal
from enum import StrEnum

import threading
from typing import Sequence

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from threetears.models.enums import ModelTier
from threetears.models.tracking import LlmPurpose, UsageRecord, UsageTracker


class _InMemorySpanExporter(SpanExporter):
    """in-memory span exporter for testing.

    collects finished spans in list for assertion.
    """

    def __init__(self) -> None:
        self._spans: list[ReadableSpan] = []
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """stores spans in memory.

        :param spans: batch of finished spans to store
        :ptype spans: Sequence[ReadableSpan]
        :return: success result
        :rtype: SpanExportResult
        """
        with self._lock:
            self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def get_finished_spans(self) -> list[ReadableSpan]:
        """returns all collected spans.

        :return: list of finished spans
        :rtype: list[ReadableSpan]
        """
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        """removes all collected spans."""
        with self._lock:
            self._spans.clear()

    def shutdown(self) -> None:
        """no-op shutdown."""

    def force_flush(self, timeout_millis: int = 0) -> bool:
        """no-op flush.

        :param timeout_millis: unused timeout
        :ptype timeout_millis: int
        :return: always true
        :rtype: bool
        """
        return True


class TestLlmPurpose:
    """tests for LlmPurpose enum."""

    def test_is_str_enum(self) -> None:
        """LlmPurpose inherits from StrEnum."""
        assert issubclass(LlmPurpose, StrEnum)

    def test_values(self) -> None:
        """LlmPurpose contains all expected members with correct values."""
        assert LlmPurpose.CHAT == "chat"
        assert LlmPurpose.TOOL_SELECTION == "tool_selection"
        assert LlmPurpose.EMBEDDING == "embedding"
        assert LlmPurpose.TRANSCRIPTION == "transcription"
        assert LlmPurpose.IMAGE_GENERATION == "image_generation"
        assert LlmPurpose.SUMMARIZATION == "summarization"
        assert LlmPurpose.EXTRACTION == "extraction"
        assert LlmPurpose.UTILITY == "utility"

    def test_member_count(self) -> None:
        """LlmPurpose has exactly eight members."""
        assert len(LlmPurpose) == 8


class TestUsageRecord:
    """tests for UsageRecord dataclass."""

    def test_is_dataclass(self) -> None:
        """UsageRecord is a dataclass."""
        assert dataclasses.is_dataclass(UsageRecord)

    def test_is_not_pydantic(self) -> None:
        """UsageRecord is not a Pydantic BaseModel."""
        from pydantic import BaseModel

        assert not issubclass(UsageRecord, BaseModel)

    def test_required_fields(self) -> None:
        """UsageRecord requires model_name, provider_name, purpose, token counts, latency_ms."""
        record = UsageRecord(
            model_name="claude-3-opus",
            provider_name="anthropic",
            purpose=LlmPurpose.CHAT,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            latency_ms=500,
        )
        assert record.model_name == "claude-3-opus"
        assert record.provider_name == "anthropic"
        assert record.purpose == LlmPurpose.CHAT
        assert record.input_tokens == 100
        assert record.output_tokens == 50
        assert record.total_tokens == 150
        assert record.latency_ms == 500

    def test_defaults(self) -> None:
        """UsageRecord defaults tier to None, cost_usd to None, date_created is set."""
        record = UsageRecord(
            model_name="claude-3-opus",
            provider_name="anthropic",
            purpose=LlmPurpose.CHAT,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            latency_ms=500,
        )
        assert record.tier is None
        assert record.cost_usd is None
        assert record.date_created is not None

    def test_cost_usd_is_decimal(self) -> None:
        """cost_usd preserves Decimal type."""
        record = UsageRecord(
            model_name="claude-3-opus",
            provider_name="anthropic",
            purpose=LlmPurpose.CHAT,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            latency_ms=500,
            cost_usd=Decimal("0.0015"),
        )
        assert isinstance(record.cost_usd, Decimal)
        assert record.cost_usd == Decimal("0.0015")

    def test_date_created_is_utc_aware(self) -> None:
        """date_created has UTC timezone info."""
        record = UsageRecord(
            model_name="claude-3-opus",
            provider_name="anthropic",
            purpose=LlmPurpose.CHAT,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            latency_ms=500,
        )
        assert record.date_created.tzinfo is not None
        assert record.date_created.tzinfo in (UTC, timezone.utc)

    def test_all_fields(self) -> None:
        """UsageRecord can be constructed with all fields."""
        now = datetime.now(UTC)
        record = UsageRecord(
            model_name="gpt-4",
            provider_name="openai",
            purpose=LlmPurpose.TOOL_SELECTION,
            input_tokens=200,
            output_tokens=100,
            total_tokens=300,
            latency_ms=1200,
            tier=ModelTier.LARGE,
            cost_usd=Decimal("0.009"),
            date_created=now,
        )
        assert record.model_name == "gpt-4"
        assert record.provider_name == "openai"
        assert record.purpose == LlmPurpose.TOOL_SELECTION
        assert record.input_tokens == 200
        assert record.output_tokens == 100
        assert record.total_tokens == 300
        assert record.latency_ms == 1200
        assert record.tier == ModelTier.LARGE
        assert record.cost_usd == Decimal("0.009")
        assert record.date_created == now


class TestUsageTracker:
    """tests for UsageTracker OpenTelemetry integration."""

    _exporter: _InMemorySpanExporter
    _provider: TracerProvider

    @classmethod
    def setup_class(cls) -> None:
        """configures shared OTel TracerProvider with in-memory exporter."""
        cls._exporter = _InMemorySpanExporter()
        cls._provider = TracerProvider()
        cls._provider.add_span_processor(SimpleSpanProcessor(cls._exporter))
        trace.set_tracer_provider(cls._provider)

    def setup_method(self) -> None:
        """clears collected spans before each test."""
        self._exporter.clear()

    def _make_usage(
        self,
        tier: ModelTier | None = ModelTier.LARGE,
        cost_usd: Decimal | None = Decimal("0.005"),
    ) -> UsageRecord:
        """creates usage record for testing.

        :param tier: model tier for record
        :ptype tier: ModelTier | None
        :param cost_usd: cost for record
        :ptype cost_usd: Decimal | None
        :return: usage record instance
        :rtype: UsageRecord
        """
        return UsageRecord(
            model_name="claude-3-opus",
            provider_name="anthropic",
            purpose=LlmPurpose.CHAT,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            latency_ms=500,
            tier=tier,
            cost_usd=cost_usd,
        )

    def test_record_no_op_without_otel(self) -> None:
        """record completes without error when OTel is available but unconfigured."""
        tracker = UsageTracker()
        usage = self._make_usage()
        tracker.record(usage)

    def test_record_creates_span(self) -> None:
        """record creates OTel span with correct attributes."""
        tracker = UsageTracker()
        usage = self._make_usage()
        tracker.record(usage)

        spans = self._exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "llm.usage"

        attrs = dict(span.attributes or {})
        assert attrs["llm.model"] == "claude-3-opus"
        assert attrs["llm.provider"] == "anthropic"
        assert attrs["llm.purpose"] == "chat"
        assert attrs["llm.input_tokens"] == 100
        assert attrs["llm.output_tokens"] == 50
        assert attrs["llm.total_tokens"] == 150
        assert attrs["llm.latency_ms"] == 500
        assert attrs["llm.tier"] == "large"
        assert attrs["llm.cost_usd"] == "0.005"

    def test_record_with_none_tier_omits_attribute(self) -> None:
        """record omits llm.tier attribute when tier is None."""
        tracker = UsageTracker()
        usage = self._make_usage(tier=None)
        tracker.record(usage)

        spans = self._exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})
        assert "llm.tier" not in attrs

    def test_record_with_none_cost_omits_attribute(self) -> None:
        """record omits llm.cost_usd attribute when cost_usd is None."""
        tracker = UsageTracker()
        usage = self._make_usage(cost_usd=None)
        tracker.record(usage)

        spans = self._exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})
        assert "llm.cost_usd" not in attrs
