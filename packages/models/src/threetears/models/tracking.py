"""LLM usage tracking with OpenTelemetry span recording."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING

from threetears.models.enums import ModelTier

if TYPE_CHECKING:
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


class UsageTracker:
    """records LLM usage metrics as OpenTelemetry spans.

    degrades gracefully to no-op when OpenTelemetry is not configured.
    """

    def __init__(self) -> None:
        self._tracer: Tracer | None = None
        self._otel_available = False
        try:
            from opentelemetry import trace

            self._tracer = trace.get_tracer("threetears.models.tracking")
            self._otel_available = True
        except ImportError:
            pass

    def record(self, usage: UsageRecord) -> None:
        """records usage metrics as OTel span attributes.

        creates span named "llm.usage" with model, provider, tier, purpose,
        token counts, cost, and latency as semantic attributes. no-op if
        OpenTelemetry is not available.

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
