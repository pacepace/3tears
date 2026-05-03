"""LLM usage tracking with OpenTelemetry spans and Prometheus metrics.

`UsageTracker.record()` emits a single ``llm.usage`` OTel span with locked
attribute names that the platform's Grafana dashboards consume. The
companion ``make_callback()`` factory returns a LangChain
``BaseCallbackHandler`` that records both the OTel span AND a parallel set
of Prometheus instruments on every ``on_llm_end`` event.

The Prometheus side effect is required because Tempo's ``span_metrics``
processor cannot sum custom span attributes — task-08 dashboards depend on
the counters this callback emits. Both the span attribute names and the
Prometheus instrument names are LOCKED. Adding new attributes/labels is
fine; renaming existing ones breaks the dashboards.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from threetears.observe import get_logger

from threetears.models.enums import ModelTier

__all__ = [
    "LlmPurpose",
    "UsageRecord",
    "UsageTracker",
    "UsageTrackingCallback",
]

logger = get_logger(__name__)

if TYPE_CHECKING:
    from langchain_core.outputs import LLMResult
    from opentelemetry.trace import Tracer


class LlmPurpose(StrEnum):
    """classification of LLM invocation by intended use.

    :cvar CHAT: conversational interaction with user
    :cvar TOOL_SELECTION: selecting or routing tool calls
    :cvar EMBEDDING: generating text embeddings
    :cvar TRANSCRIPTION: audio-to-text conversion
    :cvar IMAGE_GENERATION: text-to-image synthesis
    :cvar SUMMARIZATION: condensing text content
    :cvar EXTRACTION: extracting structured data from text
    :cvar UTILITY: general-purpose or uncategorized usage
    """

    CHAT = "chat"
    TOOL_SELECTION = "tool_selection"
    EMBEDDING = "embedding"
    TRANSCRIPTION = "transcription"
    IMAGE_GENERATION = "image_generation"
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"
    UTILITY = "utility"


@dataclass
class UsageRecord:
    """single LLM usage event with token counts, cost, and latency.

    :param model_name: identifier of model used
    :ptype model_name: str
    :param provider_name: name of provider serving model
    :ptype provider_name: str
    :param purpose: classification of invocation purpose
    :ptype purpose: LlmPurpose
    :param input_tokens: number of input tokens consumed
    :ptype input_tokens: int
    :param output_tokens: number of output tokens generated
    :ptype output_tokens: int
    :param total_tokens: total tokens consumed (input + output)
    :ptype total_tokens: int
    :param latency_ms: request latency in milliseconds
    :ptype latency_ms: int
    :param tier: model size/cost tier
    :ptype tier: ModelTier | None
    :param cost_usd: estimated cost in USD
    :ptype cost_usd: Decimal | None
    :param date_created: timestamp when usage was recorded
    :ptype date_created: datetime
    """

    model_name: str
    provider_name: str
    purpose: LlmPurpose
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: int
    tier: ModelTier | None = None
    cost_usd: Decimal | None = None
    date_created: datetime = field(default_factory=lambda: datetime.now(UTC))


# Locked Prometheus instrument names (do not rename — task-08 dashboards
# read these). Counters track tokens, calls, and cost; the histogram
# tracks per-call latency. All carry bounded labels: model, provider,
# purpose. Do not add high-cardinality labels (conversation_id, etc.).
_PROM_LABELS = ("model", "provider", "purpose")
_PROM_COUNTER_INPUT_TOKENS_NAME = "threetears_llm_input_tokens_total"
_PROM_COUNTER_OUTPUT_TOKENS_NAME = "threetears_llm_output_tokens_total"
_PROM_COUNTER_CALLS_NAME = "threetears_llm_calls_total"
_PROM_COUNTER_COST_USD_NAME = "threetears_llm_cost_usd_total"
_PROM_HISTOGRAM_LATENCY_NAME = "threetears_llm_latency_seconds"


class _PrometheusEmitter:
    """lazily-initialised holder for the locked Prometheus instruments.

    ``prometheus_client`` is an optional dependency. when not installed
    this emitter no-ops every call gracefully — matching the OTel side
    effect's behaviour when OTel is not available.
    """

    def __init__(self) -> None:
        self._available = False
        self._counter_input: Any = None
        self._counter_output: Any = None
        self._counter_calls: Any = None
        self._counter_cost: Any = None
        self._histogram_latency: Any = None
        self._initialise()

    def _initialise(self) -> None:
        """imports prometheus_client and registers the locked instruments.

        on import failure (extra not installed) leaves ``_available`` as
        False so subsequent calls become no-ops.
        """
        try:
            from prometheus_client import Counter, Histogram
        except ImportError:
            logger.debug(
                "prometheus_client not installed -- LLM Prometheus emission disabled",
            )
            return

        self._counter_input = Counter(
            _PROM_COUNTER_INPUT_TOKENS_NAME,
            "Total LLM input tokens consumed",
            labelnames=_PROM_LABELS,
        )
        self._counter_output = Counter(
            _PROM_COUNTER_OUTPUT_TOKENS_NAME,
            "Total LLM output tokens generated",
            labelnames=_PROM_LABELS,
        )
        self._counter_calls = Counter(
            _PROM_COUNTER_CALLS_NAME,
            "Total LLM calls observed",
            labelnames=_PROM_LABELS,
        )
        self._counter_cost = Counter(
            _PROM_COUNTER_COST_USD_NAME,
            "Total LLM cost in USD",
            labelnames=_PROM_LABELS,
        )
        self._histogram_latency = Histogram(
            _PROM_HISTOGRAM_LATENCY_NAME,
            "LLM call latency in seconds",
            labelnames=_PROM_LABELS,
        )
        self._available = True

    @property
    def available(self) -> bool:
        """whether the prometheus instruments were registered.

        :return: True when prometheus_client is installed and instruments are live
        :rtype: bool
        """
        return self._available

    def emit(self, usage: UsageRecord) -> None:
        """emits the locked counters and histogram for one usage record.

        :param usage: usage record to emit as Prometheus samples
        :ptype usage: UsageRecord
        """
        if not self._available:
            return

        labels = {
            "model": usage.model_name,
            "provider": usage.provider_name,
            "purpose": str(usage.purpose),
        }
        self._counter_input.labels(**labels).inc(usage.input_tokens)
        self._counter_output.labels(**labels).inc(usage.output_tokens)
        self._counter_calls.labels(**labels).inc()
        if usage.cost_usd is not None:
            self._counter_cost.labels(**labels).inc(float(usage.cost_usd))
        self._histogram_latency.labels(**labels).observe(usage.latency_ms / 1000.0)


# module-level emitter so the prometheus instruments are registered exactly
# once per process (registering the same metric twice raises in
# prometheus_client). lazy-instantiated on first UsageTracker construction.
_PROM_EMITTER: _PrometheusEmitter | None = None


def _get_prom_emitter() -> _PrometheusEmitter:
    """returns the singleton Prometheus emitter, instantiating it once.

    :return: shared emitter (no-op when prometheus_client is missing)
    :rtype: _PrometheusEmitter
    """
    global _PROM_EMITTER
    if _PROM_EMITTER is None:
        _PROM_EMITTER = _PrometheusEmitter()
    return _PROM_EMITTER


def _reset_prom_emitter_for_testing() -> None:
    """resets the singleton emitter so unit tests can re-register metrics.

    only intended for use by the test suite — production code never calls
    this. clears the module-level cache so the next ``_get_prom_emitter``
    call rebuilds against the current process state.
    """
    global _PROM_EMITTER
    _PROM_EMITTER = None


class UsageTracker:
    """records LLM usage metrics as OpenTelemetry spans and Prometheus samples.

    degrades gracefully to no-op when OpenTelemetry is not configured. the
    ``make_callback()`` method returns a LangChain ``BaseCallbackHandler``
    that fires both side effects on every ``on_llm_end`` event.
    """

    def __init__(self) -> None:
        self._tracer: Tracer | None = None
        self._otel_available = False
        self._prom = _get_prom_emitter()
        try:
            from opentelemetry import trace

            self._tracer = trace.get_tracer("threetears.models.tracking")
            self._otel_available = True
        except ImportError:
            logger.debug(
                "opentelemetry not installed -- LLM span emission disabled",
            )

    def record(self, usage: UsageRecord) -> None:
        """records usage as an OTel span and Prometheus samples.

        creates span named ``llm.usage`` with the 9 locked attributes
        (``llm.{model,provider,purpose,input_tokens,output_tokens,total_tokens,latency_ms,tier,cost_usd}``)
        and increments the locked Prometheus counters/histogram. either side
        effect no-ops gracefully when its backing library is missing.

        :param usage: usage record to emit
        :ptype usage: UsageRecord
        """
        self._emit_otel_span(usage)
        self._prom.emit(usage)

    def _emit_otel_span(self, usage: UsageRecord) -> None:
        """creates the locked ``llm.usage`` OTel span.

        :param usage: usage record to emit as span
        :ptype usage: UsageRecord
        """
        if not self._otel_available or self._tracer is None:
            return

        with self._tracer.start_as_current_span("llm.usage") as span:
            span.set_attribute("llm.model", usage.model_name)
            span.set_attribute("llm.provider", usage.provider_name)
            span.set_attribute("llm.purpose", str(usage.purpose))
            span.set_attribute("llm.input_tokens", usage.input_tokens)
            span.set_attribute("llm.output_tokens", usage.output_tokens)
            span.set_attribute("llm.total_tokens", usage.total_tokens)
            span.set_attribute("llm.latency_ms", usage.latency_ms)

            if usage.tier is not None:
                span.set_attribute("llm.tier", str(usage.tier))
            if usage.cost_usd is not None:
                span.set_attribute("llm.cost_usd", str(usage.cost_usd))

    def make_callback(
        self,
        *,
        model_name: str,
        provider_name: str,
        purpose: LlmPurpose = LlmPurpose.CHAT,
        tier: ModelTier | None = None,
        cost_per_input_token: Decimal | None = None,
        cost_per_output_token: Decimal | None = None,
    ) -> BaseCallbackHandler:
        """builds a LangChain callback that records usage on every ``on_llm_end``.

        :param model_name: identifier of model used
        :ptype model_name: str
        :param provider_name: name of provider serving model
        :ptype provider_name: str
        :param purpose: classification of invocation purpose
        :ptype purpose: LlmPurpose
        :param tier: model size/cost tier
        :ptype tier: ModelTier | None
        :param cost_per_input_token: USD per input token (used to derive cost_usd)
        :ptype cost_per_input_token: Decimal | None
        :param cost_per_output_token: USD per output token (used to derive cost_usd)
        :ptype cost_per_output_token: Decimal | None
        :return: callback handler suitable for ``model.with_config(callbacks=[...])``
        :rtype: BaseCallbackHandler
        """
        return UsageTrackingCallback(
            tracker=self,
            model_name=model_name,
            provider_name=provider_name,
            purpose=purpose,
            tier=tier,
            cost_per_input_token=cost_per_input_token,
            cost_per_output_token=cost_per_output_token,
        )


class UsageTrackingCallback(BaseCallbackHandler):
    """LangChain callback that records ``UsageRecord`` on every ``on_llm_end``.

    captures wall-clock latency between ``on_llm_start`` (or
    ``on_chat_model_start``) and ``on_llm_end``, extracts token counts from
    the standard LangChain ``LLMResult.llm_output`` or per-generation
    ``usage_metadata``, and forwards a fully-populated :class:`UsageRecord`
    to :meth:`UsageTracker.record`.
    """

    def __init__(
        self,
        *,
        tracker: UsageTracker,
        model_name: str,
        provider_name: str,
        purpose: LlmPurpose = LlmPurpose.CHAT,
        tier: ModelTier | None = None,
        cost_per_input_token: Decimal | None = None,
        cost_per_output_token: Decimal | None = None,
    ) -> None:
        """initialises the callback with the metadata needed for each record.

        :param tracker: backing tracker that emits OTel + Prometheus
        :ptype tracker: UsageTracker
        :param model_name: identifier of model used
        :ptype model_name: str
        :param provider_name: name of provider serving model
        :ptype provider_name: str
        :param purpose: classification of invocation purpose
        :ptype purpose: LlmPurpose
        :param tier: model size/cost tier
        :ptype tier: ModelTier | None
        :param cost_per_input_token: USD per input token
        :ptype cost_per_input_token: Decimal | None
        :param cost_per_output_token: USD per output token
        :ptype cost_per_output_token: Decimal | None
        """
        super().__init__()
        self._tracker = tracker
        self._model_name = model_name
        self._provider_name = provider_name
        self._purpose = purpose
        self._tier = tier
        self._cost_per_input_token = cost_per_input_token
        self._cost_per_output_token = cost_per_output_token
        # run_id -> monotonic start timestamp for latency calculation
        self._starts: dict[Any, float] = {}

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """records start timestamp so ``on_llm_end`` can compute latency.

        :param serialized: serialized LLM definition (unused)
        :ptype serialized: dict[str, Any]
        :param prompts: prompt strings (unused)
        :ptype prompts: list[str]
        :param run_id: optional run identifier supplied by LangChain
        :ptype run_id: UUID | None
        :param kwargs: additional LangChain context (ignored)
        :ptype kwargs: Any
        """
        _ = serialized
        _ = prompts
        _ = kwargs
        key = run_id if run_id is not None else id(self)
        self._starts[key] = time.monotonic()

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """records start timestamp for chat-model invocations.

        LangChain emits ``on_chat_model_start`` (not ``on_llm_start``) for
        chat models, so we mirror the same bookkeeping there.

        :param serialized: serialized model definition (unused)
        :ptype serialized: dict[str, Any]
        :param messages: input messages (unused)
        :ptype messages: list[list[Any]]
        :param run_id: optional run identifier supplied by LangChain
        :ptype run_id: UUID | None
        :param kwargs: additional LangChain context (ignored)
        :ptype kwargs: Any
        """
        _ = serialized
        _ = messages
        _ = kwargs
        key = run_id if run_id is not None else id(self)
        self._starts[key] = time.monotonic()

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """builds a ``UsageRecord`` and forwards it to the tracker.

        :param response: LangChain ``LLMResult`` carrying token usage
        :ptype response: LLMResult
        :param run_id: optional run identifier supplied by LangChain
        :ptype run_id: UUID | None
        :param kwargs: additional LangChain context (ignored)
        :ptype kwargs: Any
        """
        _ = kwargs
        key = run_id if run_id is not None else id(self)
        start = self._starts.pop(key, None)
        latency_ms = int((time.monotonic() - start) * 1000) if start is not None else 0

        input_tokens, output_tokens = self._extract_token_counts(response)
        total_tokens = input_tokens + output_tokens

        cost_usd: Decimal | None = None
        if self._cost_per_input_token is not None and self._cost_per_output_token is not None:
            cost_usd = self._cost_per_input_token * Decimal(input_tokens) + self._cost_per_output_token * Decimal(
                output_tokens
            )

        usage = UsageRecord(
            model_name=self._model_name,
            provider_name=self._provider_name,
            purpose=self._purpose,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            tier=self._tier,
            cost_usd=cost_usd,
        )
        self._tracker.record(usage)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """drops any pending start timestamp on error.

        the failure path is intentionally untracked here — the circuit
        breaker callback is the right place for failure observability.

        :param error: raised exception (logged at debug level)
        :ptype error: BaseException
        :param run_id: optional run identifier supplied by LangChain
        :ptype run_id: UUID | None
        :param kwargs: additional LangChain context (ignored)
        :ptype kwargs: Any
        """
        _ = kwargs
        key = run_id if run_id is not None else id(self)
        self._starts.pop(key, None)
        logger.debug(
            "LLM call errored, usage record skipped: %s",
            error,
        )

    def _extract_token_counts(self, response: LLMResult) -> tuple[int, int]:
        """extracts (input, output) token counts from a LangChain ``LLMResult``.

        checks ``llm_output['token_usage']`` (OpenAI-style) first, then
        falls back to ``Generation.message.usage_metadata`` (Anthropic and
        newer OpenAI / OpenRouter shapes).

        :param response: LangChain LLM result
        :ptype response: LLMResult
        :return: tuple of (input_tokens, output_tokens)
        :rtype: tuple[int, int]
        """
        input_tokens = 0
        output_tokens = 0

        llm_output = getattr(response, "llm_output", None) or {}
        token_usage = llm_output.get("token_usage") if isinstance(llm_output, dict) else None
        if isinstance(token_usage, dict):
            input_tokens = int(token_usage.get("prompt_tokens", 0) or token_usage.get("input_tokens", 0))
            output_tokens = int(token_usage.get("completion_tokens", 0) or token_usage.get("output_tokens", 0))

        if input_tokens == 0 and output_tokens == 0:
            generations = getattr(response, "generations", None) or []
            for batch in generations:
                for gen in batch:
                    message = getattr(gen, "message", None)
                    usage_metadata = getattr(message, "usage_metadata", None) if message is not None else None
                    if isinstance(usage_metadata, dict):
                        input_tokens = int(usage_metadata.get("input_tokens", 0))
                        output_tokens = int(usage_metadata.get("output_tokens", 0))
                        break
                if input_tokens or output_tokens:
                    break

        return input_tokens, output_tokens
