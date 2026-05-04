"""LangChain-native AI model factories with capability metadata, circuit
breakers, and usage tracking.

3tears v0.6.0+ exposes provider factory functions that return configured
LangChain ``BaseChatModel`` and ``Embeddings`` instances. The legacy
``ChatProvider`` / ``EmbeddingProvider`` / ``TranscriptionProvider`` /
``SpeechProvider`` / ``RerankingProvider`` runtime protocols have been
removed.
"""

from __future__ import annotations

from threetears.models.cache import ModelCache
from threetears.models.capabilities import (
    ModelCapabilities,
    get_capabilities,
    list_capabilities,
    register_capabilities,
)
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
from threetears.models.tracking import (
    LlmPurpose,
    UsageAuditSink,
    UsageCounterSink,
    UsageRecord,
    UsageTracker,
    UsageTrackingCallback,
)

__all__ = [
    # enums
    "ModelType",
    "ModelStatus",
    "ModelTier",
    # capabilities
    "ModelCapabilities",
    "get_capabilities",
    "list_capabilities",
    "register_capabilities",
    # factory
    "create_chat_model",
    "create_embedding_model",
    # preprocessing
    "enforce_alternating_roles",
    "format_vision_content",
    "preprocess_messages",
    # errors
    "identify_provider",
    "friendly_api_error",
    # cache
    "ModelCache",
    # registry
    "BUILTIN_PROVIDERS",
    "ProviderRegistry",
    # circuit breaker
    "CircuitState",
    "CircuitOpenError",
    "CircuitBreaker",
    "CircuitBreakerCallback",
    "CircuitBreakerRegistry",
    # tracking
    "LlmPurpose",
    "UsageAuditSink",
    "UsageCounterSink",
    "UsageRecord",
    "UsageTracker",
    "UsageTrackingCallback",
]
