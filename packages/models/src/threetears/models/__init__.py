"""AI model provider protocols, capabilities, and result types.

public API for the 3tears-models package. re-exports all types needed
by consumers: enums, message types, result types, provider protocols,
and model capabilities.
"""

from __future__ import annotations

from threetears.models.capabilities import ModelCapabilities
from threetears.models.enums import ModelStatus, ModelTier, ModelType
from threetears.models.messages import (
    ChatMessage,
    MessageRole,
    ToolCallRequest,
    ToolDefinition,
)
from threetears.models.protocol import (
    ChatProvider,
    EmbeddingProvider,
    ImageGenerationProvider,
    RerankingProvider,
    SpeechProvider,
    TranscriptionProvider,
)
from threetears.models.results import (
    ChatChunk,
    ChatResult,
    EmbeddingResult,
    RerankResult,
    SpeechResult,
    TranscriptionResult,
    TranscriptionSegment,
)

__all__ = [
    # enums
    "ModelType",
    "ModelStatus",
    "ModelTier",
    # messages
    "MessageRole",
    "ToolCallRequest",
    "ToolDefinition",
    "ChatMessage",
    # results
    "TranscriptionSegment",
    "TranscriptionResult",
    "EmbeddingResult",
    "ChatResult",
    "ChatChunk",
    "SpeechResult",
    "RerankResult",
    # protocols
    "ChatProvider",
    "EmbeddingProvider",
    "TranscriptionProvider",
    "ImageGenerationProvider",
    "SpeechProvider",
    "RerankingProvider",
    # capabilities
    "ModelCapabilities",
]
