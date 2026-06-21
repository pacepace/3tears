"""tests for LLM usage tracking and purpose classification."""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import UTC, datetime, timezone
from decimal import Decimal
from enum import StrEnum
from uuid import uuid4

import threading
from typing import Sequence

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from threetears.models import DEFAULT_LARGE_MODEL
from threetears.models.enums import ModelTier
from threetears.models.tracking import (
    LlmPurpose,
    UsageAuditSink,
    UsageCounterSink,
    UsageRecord,
    UsageTracker,
    _reset_prom_emitter_for_testing,
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
            model_name=DEFAULT_LARGE_MODEL,
            provider_name="anthropic",
            purpose=LlmPurpose.CHAT,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            latency_ms=500,
        )
        assert record.model_name == DEFAULT_LARGE_MODEL
        assert record.provider_name == "anthropic"
        assert record.purpose == LlmPurpose.CHAT
        assert record.input_tokens == 100
        assert record.output_tokens == 50
        assert record.total_tokens == 150
        assert record.latency_ms == 500

    def test_defaults(self) -> None:
        """UsageRecord defaults tier to None, cost_usd to None, date_created is set."""
        record = UsageRecord(
            model_name=DEFAULT_LARGE_MODEL,
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
            model_name=DEFAULT_LARGE_MODEL,
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
            model_name=DEFAULT_LARGE_MODEL,
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
        """configures shared OTel TracerProvider with in-memory exporter.

        OTel's ``set_tracer_provider`` is a set-once operation; if another
        test class earlier in the run already set a provider, this call
        is silently ignored and our spans land in their exporter. Reset
        the internal sentinel and rebind the cached UsageTracker tracer
        so this test class always sees its own provider.
        """
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
            model_name=DEFAULT_LARGE_MODEL,
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
        assert attrs["llm.model"] == DEFAULT_LARGE_MODEL
        assert attrs["llm.provider"] == "anthropic"
        assert attrs["llm.purpose"] == "chat"
        assert attrs["llm.input_tokens"] == 100
        assert attrs["llm.output_tokens"] == 50
        assert attrs["llm.total_tokens"] == 150
        assert attrs["llm.latency_ms"] == 500
        assert attrs["llm.tier"] == "large"
        assert attrs["llm.cost_usd"] == "0.005"

    def test_record_omits_cache_attrs_when_zero(self) -> None:
        """no prompt-cache tokens -> no llm.cache_* attributes (clean span)."""
        tracker = UsageTracker()
        tracker.record(self._make_usage())
        attrs = dict(self._exporter.get_finished_spans()[0].attributes or {})
        assert "llm.cache_read_tokens" not in attrs
        assert "llm.cache_creation_tokens" not in attrs

    def test_record_emits_cache_attrs_when_set(self) -> None:
        """prompt-cache tokens flow to the span as llm.cache_* attributes."""
        tracker = UsageTracker()
        usage = UsageRecord(
            model_name=DEFAULT_LARGE_MODEL,
            provider_name="anthropic",
            purpose=LlmPurpose.CHAT,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            latency_ms=500,
            cache_read_tokens=80,
            cache_creation_tokens=20,
        )
        tracker.record(usage)
        attrs = dict(self._exporter.get_finished_spans()[0].attributes or {})
        assert attrs["llm.cache_read_tokens"] == 80
        assert attrs["llm.cache_creation_tokens"] == 20

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


class TestUsageRecordTenantFields:
    """task-07.5: tests for the optional tenant context fields."""

    def test_tenant_fields_default_none(self) -> None:
        """tenant context fields default to None for backward compat."""
        record = UsageRecord(
            model_name=DEFAULT_LARGE_MODEL,
            provider_name="anthropic",
            purpose=LlmPurpose.CHAT,
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            latency_ms=10,
        )
        assert record.agent_id is None
        assert record.customer_id is None
        assert record.user_id is None
        assert record.conversation_id is None
        assert record.invocation_ref is None
        assert record.category is None
        assert record.model_id is None

    def test_tenant_fields_round_trip(self) -> None:
        """tenant context fields persist on the dataclass."""
        agent_id = uuid4()
        customer_id = uuid4()
        user_id = uuid4()
        conversation_id = uuid4()
        model_id = uuid4()
        record = UsageRecord(
            model_name=DEFAULT_LARGE_MODEL,
            provider_name="anthropic",
            purpose=LlmPurpose.CHAT,
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            latency_ms=10,
            agent_id=agent_id,
            customer_id=customer_id,
            user_id=user_id,
            conversation_id=conversation_id,
            invocation_ref="tool-llm-7c1a",
            category="chat",
            model_id=model_id,
        )
        assert record.agent_id == agent_id
        assert record.customer_id == customer_id
        assert record.user_id == user_id
        assert record.conversation_id == conversation_id
        assert record.invocation_ref == "tool-llm-7c1a"
        assert record.category == "chat"
        assert record.model_id == model_id


class _RecordingSink:
    """test sink that records every call into a list."""

    def __init__(self) -> None:
        self.calls: list[UsageRecord] = []

    async def record(self, record: UsageRecord) -> None:
        """records the incoming record into ``self.calls``.

        :param record: usage record forwarded from the tracker
        :ptype record: UsageRecord
        """
        self.calls.append(record)


class _FailingSink:
    """test sink that always raises to verify the swallow contract."""

    async def record(self, record: UsageRecord) -> None:
        """raises every call so callers can assert the swallow contract.

        :param record: usage record (unused)
        :ptype record: UsageRecord
        """
        _ = record
        raise RuntimeError("sink intentionally failed")


class TestUsageTrackerSinks:
    """task-07.5: tests for audit + counter sink fanout."""

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

    def _make_usage(self, **overrides: object) -> UsageRecord:
        """builds a usage record with sane defaults for sink tests.

        :param overrides: keyword overrides for the record fields
        :ptype overrides: object
        :return: usage record instance
        :rtype: UsageRecord
        """
        defaults: dict[str, object] = {
            "model_name": DEFAULT_LARGE_MODEL,
            "provider_name": "anthropic",
            "purpose": LlmPurpose.CHAT,
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "latency_ms": 500,
        }
        defaults.update(overrides)
        return UsageRecord(**defaults)  # type: ignore[arg-type]

    def test_protocols_runtime_checkable(self) -> None:
        """sinks satisfy the runtime-checkable Protocols."""
        audit_sink = _RecordingSink()
        counter_sink = _RecordingSink()
        assert isinstance(audit_sink, UsageAuditSink)
        assert isinstance(counter_sink, UsageCounterSink)

    def test_record_without_sinks_no_op(self) -> None:
        """tracker without sinks just emits OTel + Prom (backward compat)."""
        tracker = UsageTracker()
        tracker.record(self._make_usage())
        spans = self._exporter.get_finished_spans()
        assert len(spans) == 1

    def test_audit_sink_invoked_on_async(self) -> None:
        """audit sink fires when called from inside an asyncio loop."""
        audit_sink = _RecordingSink()
        tracker = UsageTracker(audit_sink=audit_sink)

        async def _drive() -> None:
            tracker.record(self._make_usage(category="chat"))
            # let the scheduled task run
            await asyncio.sleep(0)

        asyncio.run(_drive())
        assert len(audit_sink.calls) == 1
        assert audit_sink.calls[0].category == "chat"

    def test_counter_sink_invoked_on_async(self) -> None:
        """counter sink fires when called from inside an asyncio loop."""
        counter_sink = _RecordingSink()
        tracker = UsageTracker(counter_sink=counter_sink)

        async def _drive() -> None:
            tracker.record(self._make_usage())
            await asyncio.sleep(0)

        asyncio.run(_drive())
        assert len(counter_sink.calls) == 1

    def test_fanout_order_is_otel_prom_audit_counter(self) -> None:
        """sink fanout fires audit before counter (documented order)."""
        events: list[str] = []

        class _OrderingSink:
            def __init__(self, name: str) -> None:
                self._name = name

            async def record(self, record: UsageRecord) -> None:
                _ = record
                events.append(self._name)

        audit_sink = _OrderingSink("audit")
        counter_sink = _OrderingSink("counter")
        tracker = UsageTracker(audit_sink=audit_sink, counter_sink=counter_sink)

        async def _drive() -> None:
            tracker.record(self._make_usage())
            await asyncio.sleep(0)

        asyncio.run(_drive())
        assert events == ["audit", "counter"]

    def test_failing_audit_sink_does_not_propagate(self) -> None:
        """a raising audit sink is swallowed; OTel + Prom + counter still fire."""
        counter_sink = _RecordingSink()
        tracker = UsageTracker(
            audit_sink=_FailingSink(),
            counter_sink=counter_sink,
        )

        async def _drive() -> None:
            tracker.record(self._make_usage())
            # let scheduled tasks run; the failing sink raises but
            # the counter sink should still run.
            await asyncio.sleep(0)

        # asyncio.run must not raise even though the audit sink raises
        asyncio.run(_drive())
        assert len(counter_sink.calls) == 1
        # OTel side effect still fired
        spans = self._exporter.get_finished_spans()
        assert len(spans) == 1

    def test_failing_counter_sink_does_not_propagate(self) -> None:
        """a raising counter sink is swallowed; audit sink still fires."""
        audit_sink = _RecordingSink()
        tracker = UsageTracker(
            audit_sink=audit_sink,
            counter_sink=_FailingSink(),
        )

        async def _drive() -> None:
            tracker.record(self._make_usage())
            await asyncio.sleep(0)

        asyncio.run(_drive())
        assert len(audit_sink.calls) == 1


class TestUsageTrackerTenantSpanAttrs:
    """task-07.5: tests for llm.tenant.* span attribute emission."""

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

    def test_span_carries_tenant_attrs_when_populated(self) -> None:
        """populated tenant fields land as llm.tenant.* span attributes."""
        agent_id = uuid4()
        customer_id = uuid4()
        user_id = uuid4()
        conversation_id = uuid4()
        tracker = UsageTracker()
        usage = UsageRecord(
            model_name=DEFAULT_LARGE_MODEL,
            provider_name="anthropic",
            purpose=LlmPurpose.CHAT,
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            latency_ms=1,
            agent_id=agent_id,
            customer_id=customer_id,
            user_id=user_id,
            conversation_id=conversation_id,
            invocation_ref="tool-llm-7c1a",
            category="chat",
        )
        tracker.record(usage)

        spans = self._exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})
        assert attrs["llm.tenant.agent_id"] == str(agent_id)
        assert attrs["llm.tenant.customer_id"] == str(customer_id)
        assert attrs["llm.tenant.user_id"] == str(user_id)
        assert attrs["llm.tenant.conversation_id"] == str(conversation_id)
        assert attrs["llm.tenant.invocation_ref"] == "tool-llm-7c1a"
        assert attrs["llm.tenant.category"] == "chat"

    def test_span_omits_tenant_attrs_when_absent(self) -> None:
        """tenant span attrs are absent when fields are None."""
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
        assert "llm.tenant.agent_id" not in attrs
        assert "llm.tenant.customer_id" not in attrs
        assert "llm.tenant.user_id" not in attrs
        assert "llm.tenant.conversation_id" not in attrs
        assert "llm.tenant.invocation_ref" not in attrs
        assert "llm.tenant.category" not in attrs


class TestPrometheusLabelDiscipline:
    """task-07.5: enforce that tenant fields never leak to Prometheus labels."""

    def test_prom_labels_unchanged(self) -> None:
        """the locked Prometheus label set is exactly {model, provider, purpose}.

        adding tenant fields here would explode user_id / conversation_id
        cardinality and bloat the time-series database.
        """
        from threetears.models.tracking import _PROM_LABELS

        assert set(_PROM_LABELS) == {"model", "provider", "purpose"}
        assert "user_id" not in _PROM_LABELS
        assert "customer_id" not in _PROM_LABELS
        assert "agent_id" not in _PROM_LABELS
        assert "conversation_id" not in _PROM_LABELS
        assert "invocation_ref" not in _PROM_LABELS
        assert "category" not in _PROM_LABELS


class TestUsageTrackerCustomRegistry:
    """task-01 follow-up: ``UsageTracker(prom_registry=...)`` targets a custom
    ``CollectorRegistry`` so consumers (like the 14-eng-ai-bot gateway) can
    expose ``threetears_llm_*`` instruments on their own metrics endpoint.

    The default-registry path MUST remain unchanged for backward compat.
    Each test resets the per-registry emitter cache in setup so registrations
    don't bleed across tests.
    """

    def setup_method(self) -> None:
        """clears the per-registry emitter cache so each test gets fresh
        instruments without raising "Duplicated timeseries" against any
        registry it touches.
        """
        _reset_prom_emitter_for_testing()

    def teardown_method(self) -> None:
        """clears the cache again so the next test class starts clean."""
        _reset_prom_emitter_for_testing()

    def _make_usage(self) -> UsageRecord:
        """builds a usage record carrying a non-zero cost so the
        ``threetears_llm_cost_usd_total`` counter increments.

        :return: usage record instance
        :rtype: UsageRecord
        """
        return UsageRecord(
            model_name=DEFAULT_LARGE_MODEL,
            provider_name="anthropic",
            purpose=LlmPurpose.CHAT,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            latency_ms=500,
            cost_usd=Decimal("0.0042"),
        )

    def test_records_to_custom_registry(self) -> None:
        """tracker built with a custom registry emits the locked
        ``threetears_llm_*`` instruments against THAT registry.
        """
        try:
            from prometheus_client import CollectorRegistry, generate_latest
        except ImportError:
            import pytest

            pytest.skip("prometheus_client not installed in this environment")

        custom = CollectorRegistry()
        tracker = UsageTracker(prom_registry=custom)
        tracker.record(self._make_usage())

        scraped = generate_latest(custom).decode("utf-8")
        assert "threetears_llm_cost_usd_total" in scraped
        assert "threetears_llm_input_tokens_total" in scraped
        assert "threetears_llm_output_tokens_total" in scraped
        assert "threetears_llm_calls_total" in scraped
        assert "threetears_llm_latency_seconds" in scraped
        # the labels survived
        assert f'model="{DEFAULT_LARGE_MODEL}"' in scraped
        assert 'provider="anthropic"' in scraped
        assert 'purpose="chat"' in scraped

    def test_default_registry_path_unchanged(self) -> None:
        """tracker built without ``prom_registry`` still uses the default
        global registry — backward compat guarantee.

        Constructs a tracker with no kwarg and asserts it is wired to the
        same emitter that ``_get_prom_emitter()`` (no arg) returns.
        """
        from threetears.models.tracking import _get_prom_emitter

        tracker = UsageTracker()
        assert tracker._prom is _get_prom_emitter()

    def test_custom_registry_does_not_pollute_default(self) -> None:
        """instruments registered on a custom registry are NOT exposed via
        the default global registry — registry isolation guarantee.
        """
        try:
            from prometheus_client import REGISTRY, CollectorRegistry, generate_latest
        except ImportError:
            import pytest

            pytest.skip("prometheus_client not installed in this environment")

        custom = CollectorRegistry()
        tracker = UsageTracker(prom_registry=custom)
        tracker.record(self._make_usage())

        # the default-registry emitter has not been built (we only used
        # a custom one), so the locked instruments should not appear in
        # the default scrape.
        default_scraped = generate_latest(REGISTRY).decode("utf-8")
        # if some other test already built a default-registry emitter,
        # the metric NAMES will be present, but the metric SAMPLES from
        # this test's custom-registry record() must not show up. We
        # check for the cost_usd value 0.0042 by way of its decoded
        # representation, which is the smoking-gun for "samples leaked
        # into the default registry".
        assert "0.0042" not in default_scraped

    def test_per_registry_emitter_caching(self) -> None:
        """two ``UsageTracker(prom_registry=same)`` calls share the same
        cached ``_PrometheusEmitter`` and do NOT double-register
        instruments (which would raise in prometheus_client).
        """
        try:
            from prometheus_client import CollectorRegistry
        except ImportError:
            import pytest

            pytest.skip("prometheus_client not installed in this environment")

        custom = CollectorRegistry()
        tracker_a = UsageTracker(prom_registry=custom)
        tracker_b = UsageTracker(prom_registry=custom)
        # both trackers share the cached emitter for that registry
        assert tracker_a._prom is tracker_b._prom

    def test_distinct_registries_get_distinct_emitters(self) -> None:
        """two trackers with two different registries do NOT share the
        cached emitter (each registry holds its own instruments).
        """
        try:
            from prometheus_client import CollectorRegistry
        except ImportError:
            import pytest

            pytest.skip("prometheus_client not installed in this environment")

        reg_a = CollectorRegistry()
        reg_b = CollectorRegistry()
        tracker_a = UsageTracker(prom_registry=reg_a)
        tracker_b = UsageTracker(prom_registry=reg_b)
        assert tracker_a._prom is not tracker_b._prom
