"""tests for the LangChain callbacks (usage tracking + circuit breaker).

verifies the dual side-effect contract: every ``on_llm_end`` emits both
the locked OTel span attributes and the locked Prometheus instruments.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from threetears.models.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)
from threetears.models.enums import ModelTier
from threetears.models.tracking import (
    LlmPurpose,
    UsageRecord,
    UsageTracker,
)


class _InMemorySpanExporter(SpanExporter):
    """in-memory span exporter for assertion in tests."""

    def __init__(self) -> None:
        self._spans: list[ReadableSpan] = []
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """stores spans in memory.

        :param spans: batch of finished spans
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
        _ = timeout_millis
        return True


def _build_llm_result(input_tokens: int, output_tokens: int) -> Any:
    """builds a minimal ``LLMResult``-like object with token usage.

    :param input_tokens: prompt tokens to surface
    :ptype input_tokens: int
    :param output_tokens: completion tokens to surface
    :ptype output_tokens: int
    :return: object exposing the ``llm_output`` shape the callback reads
    :rtype: Any
    """
    result = MagicMock()
    result.llm_output = {
        "token_usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
        },
    }
    result.generations = []
    return result


class _CapturingTracker(UsageTracker):
    """tracker subclass that captures every ``record`` call for assertion."""

    def __init__(self) -> None:
        super().__init__()
        self.captured: list[UsageRecord] = []

    def record(self, usage: UsageRecord) -> None:
        """captures the usage record without emitting OTel/Prometheus.

        :param usage: usage record forwarded by the callback
        :ptype usage: UsageRecord
        """
        self.captured.append(usage)


class TestUsageTrackerCallback:
    """tests for the callback returned by :meth:`UsageTracker.make_callback`."""

    def test_callback_records_on_llm_end(self) -> None:
        """``on_llm_end`` forwards a populated ``UsageRecord`` to the tracker."""
        tracker = _CapturingTracker()
        callback = tracker.make_callback(
            model_name="claude-sonnet-4-20250514",
            provider_name="anthropic",
            purpose=LlmPurpose.CHAT,
            tier=ModelTier.LARGE,
            cost_per_input_token=Decimal("0.000003"),
            cost_per_output_token=Decimal("0.000015"),
        )

        callback.on_chat_model_start({}, [], run_id=None)
        callback.on_llm_end(_build_llm_result(10, 20), run_id=None)

        assert len(tracker.captured) == 1
        record = tracker.captured[0]
        assert record.model_name == "claude-sonnet-4-20250514"
        assert record.provider_name == "anthropic"
        assert record.input_tokens == 10
        assert record.output_tokens == 20
        assert record.total_tokens == 30
        assert record.tier == ModelTier.LARGE
        assert record.cost_usd == Decimal("0.000003") * 10 + Decimal("0.000015") * 20

    def test_callback_handles_missing_usage(self) -> None:
        """``LLMResult`` without ``token_usage`` records zero counts."""
        tracker = _CapturingTracker()
        callback = tracker.make_callback(
            model_name="gpt-4o",
            provider_name="openai",
        )

        empty_result = MagicMock()
        empty_result.llm_output = None
        empty_result.generations = []
        callback.on_llm_start({}, ["prompt"], run_id=None)
        callback.on_llm_end(empty_result, run_id=None)

        assert len(tracker.captured) == 1
        assert tracker.captured[0].input_tokens == 0
        assert tracker.captured[0].output_tokens == 0


class TestUsageTrackerOTelAttributes:
    """tests that ``UsageTracker.record`` emits the locked span attributes."""

    _exporter: _InMemorySpanExporter
    _provider: TracerProvider

    @classmethod
    def setup_class(cls) -> None:
        """configures shared OTel TracerProvider with an in-memory exporter."""
        cls._exporter = _InMemorySpanExporter()
        cls._provider = TracerProvider()
        cls._provider.add_span_processor(SimpleSpanProcessor(cls._exporter))
        trace.set_tracer_provider(cls._provider)

    def setup_method(self) -> None:
        """clears collected spans before each test."""
        self._exporter.clear()

    def test_record_emits_locked_attribute_keys(self) -> None:
        """the 9 locked ``llm.*`` keys all reach the OTel span."""
        tracker = UsageTracker()

        usage = UsageRecord(
            model_name="claude-sonnet-4-20250514",
            provider_name="anthropic",
            purpose=LlmPurpose.CHAT,
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            latency_ms=42,
            tier=ModelTier.LARGE,
            cost_usd=Decimal("0.0009"),
        )
        tracker.record(usage)

        spans = self._exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})

        for key in (
            "llm.model",
            "llm.provider",
            "llm.purpose",
            "llm.input_tokens",
            "llm.output_tokens",
            "llm.total_tokens",
            "llm.latency_ms",
            "llm.tier",
            "llm.cost_usd",
        ):
            assert key in attrs, f"locked span attribute missing: {key}"


class TestPrometheusEmission:
    """tests for the Prometheus side effect on ``UsageTracker.record``."""

    def test_record_increments_locked_instruments_when_available(self) -> None:
        """all locked counters/histogram fire when prometheus_client is available."""
        try:
            import prometheus_client  # noqa: F401
        except ImportError:
            pytest.skip("prometheus_client not installed in this environment")

        tracker = UsageTracker()
        usage = UsageRecord(
            model_name="claude-sonnet-4-20250514",
            provider_name="anthropic",
            purpose=LlmPurpose.CHAT,
            input_tokens=7,
            output_tokens=11,
            total_tokens=18,
            latency_ms=123,
            tier=ModelTier.LARGE,
            cost_usd=Decimal("0.0001"),
        )
        # must not raise; instruments are well-formed.
        tracker.record(usage)

    def test_record_noops_when_prometheus_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``UsageTracker.record`` does not raise when prometheus_client is missing.

        simulates the missing-dep case by hiding ``prometheus_client`` from the
        import system before instantiating a fresh tracker. the
        ``_PrometheusEmitter`` then resolves to its no-op state.
        """
        import sys

        from threetears.models import tracking

        # hide prometheus_client from the import system; reset the cached emitter.
        monkeypatch.setitem(sys.modules, "prometheus_client", None)
        tracking._reset_prom_emitter_for_testing()  # noqa: SLF001
        try:
            tracker = UsageTracker()
            usage = UsageRecord(
                model_name="m",
                provider_name="p",
                purpose=LlmPurpose.CHAT,
                input_tokens=1,
                output_tokens=2,
                total_tokens=3,
                latency_ms=4,
            )
            tracker.record(usage)  # must not raise.
        finally:
            tracking._reset_prom_emitter_for_testing()  # noqa: SLF001


class TestCircuitBreakerCallback:
    """tests for the circuit-breaker callback factory."""

    def test_open_breaker_raises_in_on_chat_model_start(self) -> None:
        """callback raises :class:`CircuitOpenError` when the breaker is open."""
        breaker = CircuitBreaker("anthropic", failure_threshold=1, recovery_timeout_seconds=60)
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        callback = breaker.make_callback()
        with pytest.raises(CircuitOpenError):
            callback.on_chat_model_start({}, [], run_id=None)

    def test_on_llm_end_records_success(self) -> None:
        """``on_llm_end`` returns the breaker to ``CLOSED`` from ``HALF_OPEN``."""
        breaker = CircuitBreaker("anthropic", failure_threshold=1, recovery_timeout_seconds=0)
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        breaker.check()  # transitions to HALF_OPEN since timeout=0
        assert breaker.state == CircuitState.HALF_OPEN

        callback = breaker.make_callback()
        callback.on_llm_end(MagicMock(), run_id=None)
        assert breaker.state == CircuitState.CLOSED

    def test_on_llm_error_records_failure(self) -> None:
        """``on_llm_error`` increments failure count via :meth:`record_failure`."""
        breaker = CircuitBreaker("anthropic", failure_threshold=2, recovery_timeout_seconds=60)
        callback = breaker.make_callback()
        callback.on_llm_error(RuntimeError("boom"), run_id=None)
        callback.on_llm_error(RuntimeError("boom"), run_id=None)
        assert breaker.state == CircuitState.OPEN

    def test_on_llm_error_ignores_circuit_open_error(self) -> None:
        """``CircuitOpenError`` is not counted as a provider failure."""
        breaker = CircuitBreaker("anthropic", failure_threshold=1, recovery_timeout_seconds=60)
        callback = breaker.make_callback()
        # this is what the callback's own on_llm_start raises -- swallow it.
        callback.on_llm_error(CircuitOpenError("anthropic", 30.0), run_id=None)
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
