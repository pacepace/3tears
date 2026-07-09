"""gu-task-05: tests for the metering dimensions added to ``UsageRecord``.

Covers GU-05-01..06: the new ``bytes_in`` / ``bytes_out`` / ``api_calls``
counters, the dedicated ``correlation_id`` and ``origin_invocation_ref``
UUID fields, their ``llm.*`` span emission, and the cardinality contract
that keeps every new field out of the Prometheus label set.
"""

from __future__ import annotations

import threading
from typing import Sequence
from uuid import uuid4

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from threetears.models import DEFAULT_LARGE_MODEL
from threetears.models.tracking import (
    LlmPurpose,
    UsageRecord,
    UsageTracker,
)


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


class TestUsageRecordDimensionDefaults:
    """GU-05-01..03: the new fields default so every existing caller is unchanged."""

    def test_construct_minimal_still_works(self) -> None:
        """a bare LLM record constructs with zero token counts (defaults intact)."""
        record = UsageRecord(
            model_name=DEFAULT_LARGE_MODEL,
            provider_name="anthropic",
            purpose=LlmPurpose.CHAT,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            latency_ms=0,
        )
        assert record.model_name == DEFAULT_LARGE_MODEL

    def test_byte_and_api_call_fields_default_zero(self) -> None:
        """bytes_in / bytes_out / api_calls default to 0."""
        record = UsageRecord(
            model_name=DEFAULT_LARGE_MODEL,
            provider_name="anthropic",
            purpose=LlmPurpose.CHAT,
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            latency_ms=10,
        )
        assert record.bytes_in == 0
        assert record.bytes_out == 0
        assert record.api_calls == 0

    def test_correlation_fields_default_none(self) -> None:
        """correlation_id / origin_invocation_ref default to None."""
        record = UsageRecord(
            model_name=DEFAULT_LARGE_MODEL,
            provider_name="anthropic",
            purpose=LlmPurpose.CHAT,
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            latency_ms=10,
        )
        assert record.correlation_id is None
        assert record.origin_invocation_ref is None

    def test_invocation_ref_still_str_and_default_none(self) -> None:
        """invocation_ref stays a str field defaulting to None (no retype)."""
        record = UsageRecord(
            model_name=DEFAULT_LARGE_MODEL,
            provider_name="anthropic",
            purpose=LlmPurpose.CHAT,
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            latency_ms=10,
        )
        assert record.invocation_ref is None
        record.invocation_ref = "tool-llm-7c1a"
        assert record.invocation_ref == "tool-llm-7c1a"

    def test_dimensions_round_trip(self) -> None:
        """the new fields persist on the dataclass with their given values."""
        correlation_id = uuid4()
        origin_invocation_ref = uuid4()
        record = UsageRecord(
            model_name=DEFAULT_LARGE_MODEL,
            provider_name="anthropic",
            purpose=LlmPurpose.CHAT,
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            latency_ms=10,
            bytes_in=2048,
            bytes_out=4096,
            api_calls=3,
            correlation_id=correlation_id,
            origin_invocation_ref=origin_invocation_ref,
        )
        assert record.bytes_in == 2048
        assert record.bytes_out == 4096
        assert record.api_calls == 3
        assert record.correlation_id == correlation_id
        assert record.origin_invocation_ref == origin_invocation_ref


class TestUsageTrackerDimensionSpanAttrs:
    """GU-05-05: the new fields emit as ``llm.*`` span attributes when populated."""

    _exporter: _InMemorySpanExporter
    _provider: TracerProvider

    @classmethod
    def setup_class(cls) -> None:
        """configures shared OTel TracerProvider with in-memory exporter."""
        cls._exporter = _InMemorySpanExporter()
        cls._provider = TracerProvider()
        cls._provider.add_span_processor(SimpleSpanProcessor(cls._exporter))
        try:
            trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
        except AttributeError:
            pass
        trace.set_tracer_provider(cls._provider)

    def setup_method(self) -> None:
        """clears collected spans before each test."""
        self._exporter.clear()

    def test_span_carries_dimension_attrs_when_populated(self) -> None:
        """populated dimensions land as llm.* span attributes."""
        correlation_id = uuid4()
        origin_invocation_ref = uuid4()
        tracker = UsageTracker()
        usage = UsageRecord(
            model_name=DEFAULT_LARGE_MODEL,
            provider_name="anthropic",
            purpose=LlmPurpose.CHAT,
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            latency_ms=1,
            bytes_in=2048,
            bytes_out=4096,
            api_calls=3,
            correlation_id=correlation_id,
            origin_invocation_ref=origin_invocation_ref,
        )
        tracker.record(usage)

        spans = self._exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})
        assert attrs["llm.correlation_id"] == str(correlation_id)
        assert attrs["llm.origin_invocation_ref"] == str(origin_invocation_ref)
        assert attrs["llm.bytes_in"] == 2048
        assert attrs["llm.bytes_out"] == 4096
        assert attrs["llm.api_calls"] == 3

    def test_span_omits_dimension_attrs_when_absent(self) -> None:
        """a pure-LLM record (zero/None dimensions) omits every new attribute."""
        tracker = UsageTracker()
        usage = UsageRecord(
            model_name=DEFAULT_LARGE_MODEL,
            provider_name="anthropic",
            purpose=LlmPurpose.CHAT,
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            latency_ms=1,
        )
        tracker.record(usage)

        spans = self._exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})
        assert "llm.correlation_id" not in attrs
        assert "llm.origin_invocation_ref" not in attrs
        assert "llm.bytes_in" not in attrs
        assert "llm.bytes_out" not in attrs
        assert "llm.api_calls" not in attrs

    def test_dimension_attrs_stay_out_of_tenant_namespace(self) -> None:
        """call dimensions are NOT emitted under the llm.tenant.* namespace."""
        tracker = UsageTracker()
        usage = UsageRecord(
            model_name=DEFAULT_LARGE_MODEL,
            provider_name="anthropic",
            purpose=LlmPurpose.CHAT,
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            latency_ms=1,
            bytes_in=2048,
            correlation_id=uuid4(),
            origin_invocation_ref=uuid4(),
        )
        tracker.record(usage)

        attrs = dict(self._exporter.get_finished_spans()[0].attributes or {})
        assert "llm.tenant.correlation_id" not in attrs
        assert "llm.tenant.origin_invocation_ref" not in attrs
        assert "llm.tenant.bytes_in" not in attrs


class TestDimensionCardinalityContract:
    """GU-05-06: no new dimension may become a Prometheus label."""

    def test_prom_labels_frozen(self) -> None:
        """the locked Prometheus label set stays exactly (model, provider, purpose)."""
        from threetears.models.tracking import _PROM_LABELS

        assert _PROM_LABELS == ("model", "provider", "purpose")

    def test_new_dimensions_not_labels(self) -> None:
        """none of the new caller-scoped fields leak into the label set."""
        from threetears.models.tracking import _PROM_LABELS

        assert "correlation_id" not in _PROM_LABELS
        assert "origin_invocation_ref" not in _PROM_LABELS
        assert "bytes_in" not in _PROM_LABELS
        assert "bytes_out" not in _PROM_LABELS
        assert "api_calls" not in _PROM_LABELS
