"""LangChain-native factory entry points for chat and embedding models.

Single public entry point per model class:

- :func:`create_chat_model` returns a configured ``BaseChatModel``
- :func:`create_embedding_model` returns a configured ``Embeddings``

The factory wires the usage tracker callback and the circuit breaker
callback into the returned model by default so consumers do not have to
re-attach instrumentation per call site.

Provider resolution is driven by :func:`get_capabilities`: passing a
``model_id`` known to the capabilities registry tells the factory which
provider's factory function to invoke. Unknown ids require the caller to
pass an explicit ``provider`` kwarg (or to register capabilities first).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from threetears.observe import get_logger

from threetears.models.capabilities import ModelCapabilities, get_capabilities
from threetears.models.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry
from threetears.models.enums import ModelType
from threetears.models.tracking import LlmPurpose, UsageTracker

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel

__all__ = [
    "create_chat_model",
    "create_embedding_model",
]

logger = get_logger(__name__)


# A single shared registry by default — host apps can override by passing
# their own breaker registry or constructing breakers per model id.
_DEFAULT_BREAKER_REGISTRY = CircuitBreakerRegistry()


def _resolve_provider(
    model_id: str,
    *,
    explicit_provider: str | None,
    capabilities: ModelCapabilities | None,
) -> str:
    """resolves the provider name for ``model_id``.

    explicit ``provider`` kwarg wins. otherwise consults the capabilities
    registry. raises a clear ``ValueError`` when neither is available.

    :param model_id: model identifier
    :ptype model_id: str
    :param explicit_provider: provider name passed by the caller (overrides registry)
    :ptype explicit_provider: str | None
    :param capabilities: capability metadata for ``model_id`` if registered
    :ptype capabilities: ModelCapabilities | None
    :return: provider name
    :rtype: str
    :raises ValueError: when no provider can be resolved
    """
    if explicit_provider:
        return explicit_provider
    if capabilities is not None and capabilities.provider_name:
        return capabilities.provider_name
    raise ValueError(
        f"Cannot resolve provider for model_id '{model_id}'. "
        "Pass `provider=...` explicitly or register capabilities for the model first.",
    )


def _build_callbacks(
    *,
    model_id: str,
    provider_name: str,
    capabilities: ModelCapabilities | None,
    purpose: LlmPurpose,
    tracker: UsageTracker | None,
    breaker: CircuitBreaker | None,
    extra_callbacks: list[BaseCallbackHandler] | None,
) -> list[BaseCallbackHandler]:
    """builds the default callback list (tracker + breaker + extras).

    :param model_id: model identifier (used as the tracker's ``model_name``)
    :ptype model_id: str
    :param provider_name: provider name (used as the tracker's ``provider_name``)
    :ptype provider_name: str
    :param capabilities: capability metadata or ``None``
    :ptype capabilities: ModelCapabilities | None
    :param purpose: classification of invocation purpose
    :ptype purpose: LlmPurpose
    :param tracker: usage tracker (defaults to a fresh instance)
    :ptype tracker: UsageTracker | None
    :param breaker: circuit breaker (defaults to one from the shared registry)
    :ptype breaker: CircuitBreaker | None
    :param extra_callbacks: additional callbacks the caller wants attached
    :ptype extra_callbacks: list[BaseCallbackHandler] | None
    :return: ordered list of callbacks to wire into the model
    :rtype: list[BaseCallbackHandler]
    """
    callbacks: list[BaseCallbackHandler] = []

    effective_tracker = tracker if tracker is not None else UsageTracker()
    cost_in = capabilities.cost_per_input_token if capabilities is not None else None
    cost_out = capabilities.cost_per_output_token if capabilities is not None else None
    tier = capabilities.model_tier if capabilities is not None else None
    callbacks.append(
        effective_tracker.make_callback(
            model_name=model_id,
            provider_name=provider_name,
            purpose=purpose,
            tier=tier,
            cost_per_input_token=cost_in,
            cost_per_output_token=cost_out,
        ),
    )

    effective_breaker = breaker if breaker is not None else _DEFAULT_BREAKER_REGISTRY.get(provider_name)
    callbacks.append(effective_breaker.make_callback())

    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    return callbacks


def create_chat_model(
    model_id: str,
    *,
    api_key: str,
    provider: str | None = None,
    purpose: LlmPurpose = LlmPurpose.CHAT,
    tracker: UsageTracker | None = None,
    breaker: CircuitBreaker | None = None,
    extra_callbacks: list[BaseCallbackHandler] | None = None,
    **provider_kwargs: Any,
) -> BaseChatModel:
    """builds a configured ``BaseChatModel`` with instrumentation pre-wired.

    resolves the provider name from the capabilities registry (or the
    explicit ``provider`` kwarg), invokes the appropriate provider factory
    function, and attaches the usage tracker + circuit breaker callbacks
    via ``with_config(callbacks=[...])``.

    :param model_id: model identifier
    :ptype model_id: str
    :param api_key: provider API key
    :ptype api_key: str
    :param provider: optional explicit provider name (overrides registry)
    :ptype provider: str | None
    :param purpose: classification of invocation purpose for tracking
    :ptype purpose: LlmPurpose
    :param tracker: optional shared usage tracker (defaults to a fresh instance)
    :ptype tracker: UsageTracker | None
    :param breaker: optional explicit circuit breaker (defaults to shared registry)
    :ptype breaker: CircuitBreaker | None
    :param extra_callbacks: optional extra callbacks to attach
    :ptype extra_callbacks: list[BaseCallbackHandler] | None
    :param provider_kwargs: additional keyword arguments forwarded to the provider factory
    :ptype provider_kwargs: Any
    :return: configured ``BaseChatModel`` instance with callbacks attached
    :rtype: BaseChatModel
    :raises ValueError: when the provider cannot be resolved or model_type is wrong
    """
    capabilities = get_capabilities(model_id)
    provider_name = _resolve_provider(
        model_id,
        explicit_provider=provider,
        capabilities=capabilities,
    )

    if capabilities is not None and capabilities.model_type != ModelType.CHAT:
        raise ValueError(
            f"Model '{model_id}' is registered as {capabilities.model_type}; "
            f"use create_embedding_model() or another factory.",
        )

    model = _invoke_chat_factory(
        provider_name=provider_name,
        model_name=model_id,
        api_key=api_key,
        provider_kwargs=provider_kwargs,
    )

    callbacks = _build_callbacks(
        model_id=model_id,
        provider_name=provider_name,
        capabilities=capabilities,
        purpose=purpose,
        tracker=tracker,
        breaker=breaker,
        extra_callbacks=extra_callbacks,
    )

    configured: BaseChatModel = model.with_config(callbacks=callbacks)
    return configured


def create_embedding_model(
    model_id: str,
    *,
    api_key: str,
    provider: str | None = None,
    **provider_kwargs: Any,
) -> Embeddings:
    """builds a configured ``Embeddings`` instance.

    embedding models do not get the LangChain ``BaseCallbackHandler`` wiring
    because LangChain's ``Embeddings`` interface predates the callback
    manager. callers that need to track embedding usage should wrap the
    returned instance.

    :param model_id: model identifier
    :ptype model_id: str
    :param api_key: provider API key
    :ptype api_key: str
    :param provider: optional explicit provider name (overrides registry)
    :ptype provider: str | None
    :param provider_kwargs: additional keyword arguments forwarded to the provider factory
    :ptype provider_kwargs: Any
    :return: configured ``Embeddings`` instance
    :rtype: Embeddings
    :raises ValueError: when the provider cannot be resolved or model_type is wrong
    """
    capabilities = get_capabilities(model_id)
    provider_name = _resolve_provider(
        model_id,
        explicit_provider=provider,
        capabilities=capabilities,
    )

    if capabilities is not None and capabilities.model_type != ModelType.EMBEDDING:
        raise ValueError(
            f"Model '{model_id}' is registered as {capabilities.model_type}; "
            f"use create_chat_model() or another factory.",
        )

    model = _invoke_embedding_factory(
        provider_name=provider_name,
        model_name=model_id,
        api_key=api_key,
        provider_kwargs=provider_kwargs,
    )
    return model


def _invoke_chat_factory(
    *,
    provider_name: str,
    model_name: str,
    api_key: str,
    provider_kwargs: dict[str, Any],
) -> BaseChatModel:
    """dispatches to the per-provider chat factory function.

    :param provider_name: provider name
    :ptype provider_name: str
    :param model_name: model identifier
    :ptype model_name: str
    :param api_key: provider API key
    :ptype api_key: str
    :param provider_kwargs: additional keyword arguments forwarded to the factory
    :ptype provider_kwargs: dict[str, Any]
    :return: ``BaseChatModel`` instance from the provider factory
    :rtype: BaseChatModel
    :raises ValueError: when the provider does not expose a chat factory
    """
    if provider_name == "anthropic":
        from threetears.models.providers.anthropic import create_anthropic_chat

        return create_anthropic_chat(model_name, api_key, **provider_kwargs)
    if provider_name == "openai":
        from threetears.models.providers.openai import create_openai_chat

        return create_openai_chat(model_name, api_key, **provider_kwargs)
    if provider_name == "openrouter":
        from threetears.models.providers.openrouter import create_openrouter_chat

        return create_openrouter_chat(model_name, api_key, **provider_kwargs)

    raise ValueError(f"Provider '{provider_name}' has no chat factory")


def _invoke_embedding_factory(
    *,
    provider_name: str,
    model_name: str,
    api_key: str,
    provider_kwargs: dict[str, Any],
) -> Embeddings:
    """dispatches to the per-provider embedding factory function.

    :param provider_name: provider name
    :ptype provider_name: str
    :param model_name: model identifier
    :ptype model_name: str
    :param api_key: provider API key
    :ptype api_key: str
    :param provider_kwargs: additional keyword arguments forwarded to the factory
    :ptype provider_kwargs: dict[str, Any]
    :return: ``Embeddings`` instance from the provider factory
    :rtype: Embeddings
    :raises ValueError: when the provider does not expose an embedding factory
    """
    if provider_name == "openai":
        from threetears.models.providers.openai import create_openai_embedding

        return create_openai_embedding(model_name, api_key, **provider_kwargs)
    if provider_name == "voyageai":
        from threetears.models.providers.voyageai import create_voyageai_embedding

        return create_voyageai_embedding(api_key, model_name=model_name, **provider_kwargs)

    raise ValueError(f"Provider '{provider_name}' has no embedding factory")
