"""OpenRouter chat factory backed by ``langchain_openrouter``.

LangChain-native shape (3tears v0.6.0+): :func:`create_openrouter_chat`
returns a fully-configured ``ChatOpenRouter`` instance. ``ChatOpenRouter``
expects the request timeout in milliseconds — the factory accepts seconds
to match the rest of the API surface and converts internally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from threetears.models.capabilities import ModelCapabilities, register_capabilities
from threetears.models.enums import ModelStatus, ModelTier, ModelType

if TYPE_CHECKING:
    from langchain_openrouter import ChatOpenRouter

__all__ = [
    "OPENROUTER_PROVIDER_NAME",
    "create_openrouter_chat",
]


OPENROUTER_PROVIDER_NAME = "openrouter"


def create_openrouter_chat(
    model_name: str,
    api_key: str,
    *,
    timeout: int = 120,
    max_retries: int = 2,
    **extra_kwargs: object,
) -> ChatOpenRouter:
    """creates a configured ``ChatOpenRouter`` for OpenRouter-routed models.

    :param model_name: OpenRouter model identifier (e.g. ``deepseek/deepseek-chat-v3-0324``)
    :ptype model_name: str
    :param api_key: OpenRouter API key
    :ptype api_key: str
    :param timeout: request timeout in seconds (converted to ms internally)
    :ptype timeout: int
    :param max_retries: maximum retry attempts for failed requests
    :ptype max_retries: int
    :param extra_kwargs: additional keyword arguments forwarded to ``ChatOpenRouter``
    :ptype extra_kwargs: object
    :return: configured ``ChatOpenRouter`` instance
    :rtype: ChatOpenRouter
    """
    from langchain_openrouter import ChatOpenRouter

    # langchain-openrouter 0.1.0 defaults app_title="langchain" and forwards
    # it as `x_title` to the underlying openrouter SDK. openrouter 0.8+
    # renamed that kwarg to `x_open_router_title`, so the old name now
    # raises TypeError. setting both to None restores compatibility until
    # langchain-openrouter ships a fix; callers that need attribution can
    # pass app_title/app_url via extra_kwargs.
    kwargs: dict[str, object] = {
        "model": model_name,
        "api_key": api_key,
        "timeout": timeout * 1000,
        "max_retries": max_retries,
        "app_title": None,
        "app_url": None,
    }
    kwargs.update(extra_kwargs)

    model: ChatOpenRouter = ChatOpenRouter(**kwargs)
    return model


# -- capability registration -------------------------------------------------

# representative OpenRouter ids. additional ids can be registered by host
# apps at boot via register_capabilities().
_OPENROUTER_CAPABILITIES: dict[str, ModelCapabilities] = {
    "deepseek/deepseek-chat-v3-0324": ModelCapabilities(
        model_name="deepseek/deepseek-chat-v3-0324",
        provider_name=OPENROUTER_PROVIDER_NAME,
        model_type=ModelType.CHAT,
        model_tier=ModelTier.LARGE,
        model_status=ModelStatus.ACTIVE,
        context_window=64_000,
        max_output_tokens=8_192,
        supports_streaming=True,
        supports_tools=True,
        supports_vision=False,
        requires_alternating_roles=True,
    ),
    "deepseek/deepseek-r1": ModelCapabilities(
        model_name="deepseek/deepseek-r1",
        provider_name=OPENROUTER_PROVIDER_NAME,
        model_type=ModelType.CHAT,
        model_tier=ModelTier.LARGE,
        model_status=ModelStatus.ACTIVE,
        context_window=64_000,
        max_output_tokens=8_192,
        supports_streaming=True,
        supports_tools=False,
        supports_vision=False,
        requires_alternating_roles=True,
    ),
}


for _model_id, _caps in _OPENROUTER_CAPABILITIES.items():
    register_capabilities(_model_id, _caps)
