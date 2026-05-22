"""LangChain-native AI model factories with capability metadata, circuit
breakers, and usage tracking.

3tears v0.6.0+ exposes provider factory functions that return configured
LangChain ``BaseChatModel`` and ``Embeddings`` instances. The legacy
``ChatProvider`` / ``EmbeddingProvider`` / ``TranscriptionProvider`` /
``SpeechProvider`` / ``RerankingProvider`` runtime protocols have been
removed.

Importing this package eagerly loads the builtin provider modules so
their import-time :func:`register_capabilities` calls populate the
shared registry. This makes :func:`create_chat_model` /
:func:`create_embedding_model` work for every builtin model id without
the caller manually importing ``threetears.models.providers.<name>``.
The provider modules themselves keep ``langchain_<provider>`` imports
inside ``TYPE_CHECKING`` and inside their factory bodies, so the
eager-load only pulls capability metadata — actually instantiating a
provider model still imports its langchain backend lazily, preserving
the "install only the providers you use" property for production
callers.

v0.9.0 adds chat-model utility modules used across every consumer:
:mod:`threetears.models.chunk_merging` (merge streamed AIMessageChunks
into a single AIMessage), :mod:`threetears.models.chunk_parsing`
(extract text + reasoning per chunk), and
:mod:`threetears.models.tool_name_validation` (defend against
malformed tool names emitted by misbehaving providers). The
OpenRouter and Anthropic wrappers invoke the validator on every
response so junk tool names never reach downstream dispatch.
"""

from __future__ import annotations

from threetears.models.cache import ModelCache
from threetears.models.capabilities import (
    CapabilityOverride,
    ModelCapabilities,
    clear_capability_overrides,
    get_capabilities,
    get_capability_override,
    list_capabilities,
    register_capabilities,
    register_capability_override,
    unregister_capability_override,
)
from threetears.models.chunk_merging import merge_chunks
from threetears.models.chunk_parsing import ChunkParsed, parse_chunk
from threetears.models.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerCallback,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
)
from threetears.models.enums import ModelStatus, ModelTier, ModelType
from threetears.models.errors import friendly_api_error, identify_provider
from threetears.models.factory import create_chat_model, create_embedding_model
from threetears.models.preprocessing import (
    enforce_alternating_roles,
    format_vision_content,
    preprocess_messages,
)
from threetears.models.registry import BUILTIN_PROVIDERS, ProviderRegistry
from threetears.models.tool_name_validation import (
    ToolNameValidationError,
    filter_invalid_tool_calls,
    is_valid_tool_name,
    validate_tool_name,
)
from threetears.models.tracking import (
    LlmPurpose,
    UsageAuditSink,
    UsageCounterSink,
    UsageRecord,
    UsageTracker,
    UsageTrackingCallback,
)

# Eager-import builtin provider modules so their import-time
# `register_capabilities()` calls populate the shared registry. The
# provider modules themselves do not import their respective
# `langchain_<provider>` package at module scope (those imports live
# inside TYPE_CHECKING + factory bodies) so this is metadata-only.
from threetears.models.providers import (  # noqa: E402, F401
    anthropic as _anthropic_caps,
    openai as _openai_caps,
    openrouter as _openrouter_caps,
    voyageai as _voyageai_caps,
    whisper as _whisper_caps,
)

__all__ = [
    "BUILTIN_PROVIDERS",
    "CapabilityOverride",
    "ChunkParsed",
    "CircuitBreaker",
    "CircuitBreakerCallback",
    "CircuitBreakerRegistry",
    "CircuitOpenError",
    "CircuitState",
    "LlmPurpose",
    "ModelCache",
    "ModelCapabilities",
    "ModelStatus",
    "ModelTier",
    "ModelType",
    "ProviderRegistry",
    "ToolNameValidationError",
    "UsageAuditSink",
    "UsageCounterSink",
    "UsageRecord",
    "UsageTracker",
    "UsageTrackingCallback",
    "clear_capability_overrides",
    "create_chat_model",
    "create_embedding_model",
    "enforce_alternating_roles",
    "filter_invalid_tool_calls",
    "format_vision_content",
    "friendly_api_error",
    "get_capabilities",
    "get_capability_override",
    "identify_provider",
    "is_valid_tool_name",
    "list_capabilities",
    "merge_chunks",
    "parse_chunk",
    "preprocess_messages",
    "register_capabilities",
    "register_capability_override",
    "unregister_capability_override",
    "validate_tool_name",
]
