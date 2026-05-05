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

task-07.5 unified the per-invocation audit and aggregated counter side
effects behind two Protocols (``UsageAuditSink``, ``UsageCounterSink``)
that consumers attach to ``UsageTracker``. Sinks are awaited via
``asyncio.create_task(...)`` so a slow database write never blocks the
LLM caller; each sink invocation is wrapped in its own try/except that
logs failures with ``exc_info=True`` and never re-raises.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from threetears.observe import get_logger

from threetears.models.enums import ModelTier

__all__ = [
    "LlmPurpose",
    "UsageAuditSink",
    "UsageCounterSink",
    "UsageRecord",
    "UsageTracker",
    "UsageTrackingCallback",
]

logger = get_logger(__name__)

if TYPE_CHECKING:
    from langchain_core.outputs import LLMResult
    from opentelemetry.trace import Tracer
    from prometheus_client import CollectorRegistry


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

    The tenant fields (``agent_id``, ``customer_id``, ``user_id``,
    ``conversation_id``, ``invocation_ref``, ``category``) are optional
    so that pure in-memory callers (no audit / counter sinks attached)
    continue to work unchanged. They flow only to OTel spans (as
    ``llm.tenant.*`` attributes) and to attached sinks; they MUST NOT
    be added as Prometheus labels (cardinality discipline -- ``user_id``
    alone would explode the time-series database).

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
    :param agent_id: optional agent UUID (14-eng-ai-bot tenant context)
    :ptype agent_id: UUID | None
    :param customer_id: optional customer UUID (14-eng-ai-bot tenant context)
    :ptype customer_id: UUID | None
    :param user_id: optional user UUID (metallm + 14-eng-ai-bot tenant context)
    :ptype user_id: UUID | None
    :param conversation_id: optional conversation UUID (metallm tenant context)
    :ptype conversation_id: UUID | None
    :param invocation_ref: optional free-form reference to the underlying
        invocation (metallm passes ``str(tool_llm_id)``; 14-eng-ai-bot
        typically passes ``None``)
    :ptype invocation_ref: str | None
    :param category: optional consumer-defined usage category (metallm:
        ``chat`` / ``sycophancy`` / ``tool`` / ``embedding`` /
        ``image_generation`` / ``naming`` / ``reasoning`` /
        ``tool_llm_dispatch``)
    :ptype category: str | None
    :param model_id: optional consumer-side model UUID. metallm
        populates this from its ``models`` table primary key so
        :class:`MetaLLMTokenAuditSink` can write the FK column;
        14-eng-ai-bot leaves it ``None``.
    :ptype model_id: UUID | None
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
    # task-07.5: optional tenant + consumer context. flows to OTel spans
    # (as ``llm.tenant.*``) and to attached sinks; never to Prometheus
    # labels (cardinality discipline).
    agent_id: UUID | None = None
    customer_id: UUID | None = None
    user_id: UUID | None = None
    conversation_id: UUID | None = None
    invocation_ref: str | None = None
    category: str | None = None
    model_id: UUID | None = None


@runtime_checkable
class UsageAuditSink(Protocol):
    """Protocol for per-invocation audit sinks.

    An audit sink writes one row per LLM invocation to a durable store
    suitable for cost reconciliation and per-conversation cost
    attribution. metallm's :class:`MetaLLMTokenAuditSink` (writes the
    ``token_usage_logs`` table) is the canonical implementation.

    Implementations MUST be coroutine-safe and MUST NOT raise into the
    caller -- ``UsageTracker`` already wraps each sink call in a
    try/except, but defense in depth keeps the contract clear.
    """

    async def record(self, record: UsageRecord) -> None:
        """persists one usage record as a per-invocation audit row.

        :param record: usage record to persist
        :ptype record: UsageRecord
        """
        ...


@runtime_checkable
class UsageCounterSink(Protocol):
    """Protocol for aggregated counter sinks.

    A counter sink increments rolling-window counters (e.g. daily /
    monthly tokens-per-customer) suitable for quota enforcement and
    billing rollups. 14-eng-ai-bot's :class:`AggregateUsageCounterSink`
    (increments the ``platform.usage_records`` table) is the canonical
    implementation.

    Implementations MUST be coroutine-safe and MUST NOT raise into the
    caller -- ``UsageTracker`` already wraps each sink call in a
    try/except, but defense in depth keeps the contract clear.
    """

    async def record(self, record: UsageRecord) -> None:
        """increments aggregated usage counters for one event.

        :param record: usage record carrying the counter increments
        :ptype record: UsageRecord
        """
        ...


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

    Optionally targets a non-default ``CollectorRegistry`` so consumers
    that scrape from their own registry (e.g. the 14-eng-ai-bot gateway,
    which exposes a dedicated ``GATEWAY_METRICS_REGISTRY`` on its
    ``:8001/metrics`` endpoint) can still surface ``threetears_llm_*``
    instruments. When ``registry`` is ``None`` the default global
    registry is used as before.

    :param registry: optional Prometheus collector registry to register
        the locked instruments against; ``None`` uses the default
        global registry
    :ptype registry: CollectorRegistry | None
    """

    def __init__(self, registry: "CollectorRegistry | None" = None) -> None:
        self._registry = registry
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
        False so subsequent calls become no-ops. when ``self._registry``
        is ``None`` the instruments register against the default global
        registry; otherwise they register against the supplied one.
        """
        try:
            from prometheus_client import Counter, Histogram
        except ImportError:
            logger.debug(
                "prometheus_client not installed -- LLM Prometheus emission disabled",
            )
            return

        # prometheus_client treats ``registry=None`` as "use the default
        # registry"; pass through verbatim so the default-registry path
        # behaves identically to before this knob existed.
        kwargs: dict[str, Any] = {"labelnames": _PROM_LABELS}
        if self._registry is not None:
            kwargs["registry"] = self._registry

        self._counter_input = Counter(
            _PROM_COUNTER_INPUT_TOKENS_NAME,
            "Total LLM input tokens consumed",
            **kwargs,
        )
        self._counter_output = Counter(
            _PROM_COUNTER_OUTPUT_TOKENS_NAME,
            "Total LLM output tokens generated",
            **kwargs,
        )
        self._counter_calls = Counter(
            _PROM_COUNTER_CALLS_NAME,
            "Total LLM calls observed",
            **kwargs,
        )
        self._counter_cost = Counter(
            _PROM_COUNTER_COST_USD_NAME,
            "Total LLM cost in USD",
            **kwargs,
        )
        self._histogram_latency = Histogram(
            _PROM_HISTOGRAM_LATENCY_NAME,
            "LLM call latency in seconds",
            **kwargs,
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


# per-registry emitter cache so the prometheus instruments are registered
# exactly once per (process, registry) pair (registering the same metric
# twice on the same registry raises in prometheus_client). The key is
# ``id(registry)``; the sentinel ``0`` denotes "default global registry".
# Multiple consumers in the same process can each pass their own
# ``CollectorRegistry`` and get independent emitters back.
_PROM_EMITTERS: dict[int, _PrometheusEmitter] = {}


def _get_prom_emitter(registry: "CollectorRegistry | None" = None) -> _PrometheusEmitter:
    """returns the cached Prometheus emitter for the given registry.

    when ``registry`` is ``None`` the default global registry is used and
    the sentinel key ``0`` indexes the cache. The emitter is instantiated
    on first use per (process, registry) pair.

    :param registry: optional collector registry; ``None`` uses the
        default global registry
    :ptype registry: CollectorRegistry | None
    :return: shared emitter for that registry (no-op when
        prometheus_client is missing)
    :rtype: _PrometheusEmitter
    """
    key = 0 if registry is None else id(registry)
    emitter = _PROM_EMITTERS.get(key)
    if emitter is None:
        emitter = _PrometheusEmitter(registry=registry)
        _PROM_EMITTERS[key] = emitter
    return emitter


def _reset_prom_emitter_for_testing() -> None:
    """resets the per-registry emitter cache so unit tests can re-register metrics.

    only intended for use by the test suite — production code never calls
    this. clears the module-level cache so the next ``_get_prom_emitter``
    call rebuilds against the current process state.
    """
    _PROM_EMITTERS.clear()


class UsageTracker:
    """records LLM usage metrics as OpenTelemetry spans and Prometheus samples.

    degrades gracefully to no-op when OpenTelemetry is not configured. the
    ``make_callback()`` method returns a LangChain ``BaseCallbackHandler``
    that fires both side effects on every ``on_llm_end`` event.

    task-07.5: ``record()`` additionally fans out to optional audit and
    counter sinks. The fanout order is OTel span → Prometheus → audit
    sink → counter sink. Sink invocations are awaited via
    ``asyncio.create_task(...)`` when a running event loop is available
    so a slow sink never blocks the LLM caller; each sink invocation is
    wrapped in a try/except that logs failures with ``exc_info=True``
    and never re-raises.

    :param audit_sink: optional sink that persists per-invocation rows
        (e.g. metallm's ``MetaLLMTokenAuditSink``)
    :ptype audit_sink: UsageAuditSink | None
    :param counter_sink: optional sink that increments rolling-window
        counters (e.g. 14-eng-ai-bot's ``AggregateUsageCounterSink``)
    :ptype counter_sink: UsageCounterSink | None
    :param prom_registry: optional Prometheus ``CollectorRegistry`` to
        register the locked ``threetears_llm_*`` instruments against.
        Consumers that scrape from a non-default registry (e.g. the
        14-eng-ai-bot gateway's ``GATEWAY_METRICS_REGISTRY``) pass their
        registry here so cross-product Grafana panels can sum traffic
        across both products. ``None`` uses the default global registry.
    :ptype prom_registry: CollectorRegistry | None
    """

    def __init__(
        self,
        *,
        audit_sink: UsageAuditSink | None = None,
        counter_sink: UsageCounterSink | None = None,
        prom_registry: "CollectorRegistry | None" = None,
    ) -> None:
        self._tracer: Tracer | None = None
        self._otel_available = False
        self._prom = _get_prom_emitter(prom_registry)
        self._audit_sink = audit_sink
        self._counter_sink = counter_sink
        try:
            from opentelemetry import trace

            self._tracer = trace.get_tracer("threetears.models.tracking")
            self._otel_available = True
        except ImportError:
            logger.debug(
                "opentelemetry not installed -- LLM span emission disabled",
            )

    def record(self, usage: UsageRecord) -> None:
        """records usage as an OTel span, Prometheus samples, and (optional) sinks.

        creates span named ``llm.usage`` with the 9 locked attributes
        (``llm.{model,provider,purpose,input_tokens,output_tokens,total_tokens,latency_ms,tier,cost_usd}``)
        plus optional ``llm.tenant.*`` attributes for tenant context, and
        increments the locked Prometheus counters/histogram. either side
        effect no-ops gracefully when its backing library is missing.

        Audit and counter sinks (when attached) are awaited via
        ``asyncio.create_task(...)`` so slow sinks do not block the
        caller. Sink failures are logged via ``exc_info=True`` and
        swallowed; a billing-audit failure must never surface to the
        LLM caller.

        :param usage: usage record to emit
        :ptype usage: UsageRecord
        """
        self._emit_otel_span(usage)
        self._prom.emit(usage)
        self._dispatch_sink("audit", self._audit_sink, usage)
        self._dispatch_sink("counter", self._counter_sink, usage)

    def _dispatch_sink(
        self,
        kind: str,
        sink: UsageAuditSink | UsageCounterSink | None,
        usage: UsageRecord,
    ) -> None:
        """schedules a fire-and-forget sink invocation on the running loop.

        when no event loop is running (sync callsite without a loop),
        the sink is invoked synchronously via ``asyncio.run()`` -- this
        path is intended for tests and one-off scripts; production
        callers all run inside an asyncio loop already.

        :param kind: human-readable sink kind for log context (``audit`` or ``counter``)
        :ptype kind: str
        :param sink: sink instance or ``None`` (no-op)
        :ptype sink: UsageAuditSink | UsageCounterSink | None
        :param usage: usage record to forward to the sink
        :ptype usage: UsageRecord
        """
        if sink is None:
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            loop.create_task(self._invoke_sink(kind, sink, usage))
            return

        # no running loop -- synchronous fallback for tests and scripts.
        # we still wrap in try/except to honour the "never re-raise"
        # contract; ``asyncio.run()`` may itself raise if a previous loop
        # was left in an inconsistent state.
        try:
            asyncio.run(self._invoke_sink(kind, sink, usage))
        except Exception:  # noqa: BLE001 -- intentional swallow per task-07.5
            logger.warning(
                "%s usage sink failed (sync fallback)",
                kind,
                exc_info=True,
            )

    @staticmethod
    async def _invoke_sink(
        kind: str,
        sink: UsageAuditSink | UsageCounterSink,
        usage: UsageRecord,
    ) -> None:
        """awaits one sink invocation, swallowing any exception.

        :param kind: human-readable sink kind for log context
        :ptype kind: str
        :param sink: sink to invoke
        :ptype sink: UsageAuditSink | UsageCounterSink
        :param usage: usage record to forward
        :ptype usage: UsageRecord
        """
        try:
            await sink.record(usage)
        except Exception:  # noqa: BLE001 -- intentional swallow per task-07.5
            logger.warning(
                "%s usage sink failed",
                kind,
                exc_info=True,
            )

    def _emit_otel_span(self, usage: UsageRecord) -> None:
        """creates the locked ``llm.usage`` OTel span.

        Existing ``llm.{model,provider,purpose,input_tokens,output_tokens,total_tokens,latency_ms,tier,cost_usd}``
        attributes are NEVER renamed or removed (task-08 dashboards
        reference them). Tenant context is added under the ``llm.tenant.*``
        namespace when populated; absent attributes when fields are
        ``None``.

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

            # task-07.5: tenant attributes under the locked llm.tenant.*
            # namespace. Names here are part of the public dashboard
            # contract -- do not rename. Absent attributes when fields
            # are None so consumers can detect "no tenant context".
            if usage.agent_id is not None:
                span.set_attribute("llm.tenant.agent_id", str(usage.agent_id))
            if usage.customer_id is not None:
                span.set_attribute("llm.tenant.customer_id", str(usage.customer_id))
            if usage.user_id is not None:
                span.set_attribute("llm.tenant.user_id", str(usage.user_id))
            if usage.conversation_id is not None:
                span.set_attribute("llm.tenant.conversation_id", str(usage.conversation_id))
            if usage.invocation_ref is not None:
                span.set_attribute("llm.tenant.invocation_ref", usage.invocation_ref)
            if usage.category is not None:
                span.set_attribute("llm.tenant.category", usage.category)

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
