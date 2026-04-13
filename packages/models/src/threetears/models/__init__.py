"""AI model provider protocols, capabilities, and result types.

public API for the 3tears-models package. re-exports all types needed
by consumers: enums, message types, result types, provider protocols,
and model capabilities.
"""

from __future__ import annotations

from threetears.models.cache import ModelCache
from threetears.models.capabilities import ModelCapabilities
from threetears.models.enums import ModelStatus, ModelTier, ModelType
from threetears.models.errors import friendly_api_error, identify_provider
from threetears.models.messages import (
    ChatMessage,
    MessageRole,
    ToolCallRequest,
    ToolDefinition,
)
from threetears.models.preprocessing import (
    enforce_alternating_roles,
    format_vision_content,
    preprocess_messages,
)
from threetears.models.protocol import (
    ChatProvider,
    EmbeddingProvider,
    ImageGenerationProvider,
    RerankingProvider,
    SpeechProvider,
    TranscriptionProvider,
)
from threetears.models.registry import ProviderRegistry
from threetears.models.results import (
    ChatChunk,
    ChatResult,
    EmbeddingResult,
    RerankResult,
    SpeechResult,
    TranscriptionResult,
    TranscriptionSegment,
)
from threetears.models.streaming import (
    merge_chunks,
    recover_invalid_tool_calls,
    recover_split_tool_calls,
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
    # preprocessing
    "enforce_alternating_roles",
    "format_vision_content",
    "preprocess_messages",
    # streaming
    "merge_chunks",
    "recover_split_tool_calls",
    "recover_invalid_tool_calls",
    # errors
    "identify_provider",
    "friendly_api_error",
    # cache
    "ModelCache",
    # registry
    "ProviderRegistry",
]
