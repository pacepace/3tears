"""anthropic chat factory returning a configured ``langchain_anthropic.ChatAnthropic``.

LangChain-native shape (3tears v0.6.0+): :func:`create_anthropic_chat`
returns a fully-configured ``ChatAnthropic`` instance. Capability metadata
for known Anthropic model ids is registered with the module-level
:func:`~threetears.models.capabilities.register_capabilities` registry at
import time so consumers can ``get_capabilities(model_id)`` without
instantiating the provider.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from threetears.models.capabilities import ModelCapabilities, register_capabilities
from threetears.models.enums import ModelStatus, ModelTier, ModelType

if TYPE_CHECKING:
    from langchain_anthropic import ChatAnthropic

__all__ = [
    "ANTHROPIC_PROVIDER_NAME",
    "create_anthropic_chat",
    "strip_v1_suffix",
]


ANTHROPIC_PROVIDER_NAME = "anthropic"


def strip_v1_suffix(url: str) -> str:
    """strips trailing ``/v1`` or ``/v1/`` from URL.

    Anthropic SDK auto-appends ``/v1``, so passing a URL ending with
    ``/v1`` would cause doubled path segments.

    :param url: base URL to clean
    :ptype url: str
    :return: URL with ``/v1`` suffix removed if present
    :rtype: str
    """
    if url.endswith("/v1"):
        return url[:-3]
    if url.endswith("/v1/"):
        return url[:-4]
    return url


def create_anthropic_chat(
    model_name: str,
    api_key: str,
    *,
    base_url: str | None = None,
    timeout: int = 120,
    max_retries: int = 2,
    **extra_kwargs: object,
) -> ChatAnthropic:
    """creates a configured ``ChatAnthropic`` for Anthropic models.

    :param model_name: Anthropic model identifier (e.g. ``claude-sonnet-4-20250514``)
    :ptype model_name: str
    :param api_key: Anthropic API key
    :ptype api_key: str
    :param base_url: optional custom API base URL; trailing ``/v1`` is stripped
    :ptype base_url: str | None
    :param timeout: request timeout in seconds
    :ptype timeout: int
    :param max_retries: maximum retry attempts for failed requests
    :ptype max_retries: int
    :param extra_kwargs: additional keyword arguments forwarded to ``ChatAnthropic``
    :ptype extra_kwargs: object
    :return: configured ``ChatAnthropic`` instance
    :rtype: ChatAnthropic
    """
    from langchain_anthropic import ChatAnthropic

    cleaned_base_url = strip_v1_suffix(base_url) if base_url else None

    kwargs: dict[str, object] = {
        "model_name": model_name,
        "api_key": api_key,
        "timeout": timeout,
        "max_retries": max_retries,
    }
    if cleaned_base_url is not None:
        kwargs["base_url"] = cleaned_base_url
    kwargs.update(extra_kwargs)

    model: ChatAnthropic = ChatAnthropic(**kwargs)
    return model


# -- capability registration -------------------------------------------------

# canonical Anthropic chat models. extend with additional ids by calling
# register_capabilities() externally at host-app boot time.
_ANTHROPIC_CAPABILITIES: dict[str, ModelCapabilities] = {
    "claude-opus-4-5-20251101": ModelCapabilities(
        model_name="claude-opus-4-5-20251101",
        provider_name=ANTHROPIC_PROVIDER_NAME,
        model_type=ModelType.CHAT,
        model_tier=ModelTier.LARGE,
        model_status=ModelStatus.ACTIVE,
        context_window=200_000,
        max_output_tokens=8_192,
        supports_streaming=True,
        supports_tools=True,
        supports_vision=True,
        requires_alternating_roles=False,
        cost_per_input_token=Decimal("0.000015"),
        cost_per_output_token=Decimal("0.000075"),
    ),
    "claude-sonnet-4-20250514": ModelCapabilities(
        model_name="claude-sonnet-4-20250514",
        provider_name=ANTHROPIC_PROVIDER_NAME,
        model_type=ModelType.CHAT,
        model_tier=ModelTier.LARGE,
        model_status=ModelStatus.ACTIVE,
        context_window=200_000,
        max_output_tokens=8_192,
        supports_streaming=True,
        supports_tools=True,
        supports_vision=True,
        requires_alternating_roles=False,
        cost_per_input_token=Decimal("0.000003"),
        cost_per_output_token=Decimal("0.000015"),
    ),
    "claude-3-5-haiku-20241022": ModelCapabilities(
        model_name="claude-3-5-haiku-20241022",
        provider_name=ANTHROPIC_PROVIDER_NAME,
        model_type=ModelType.CHAT,
        model_tier=ModelTier.SMALL,
        model_status=ModelStatus.ACTIVE,
        context_window=200_000,
        max_output_tokens=8_192,
        supports_streaming=True,
        supports_tools=True,
        supports_vision=True,
        requires_alternating_roles=False,
        cost_per_input_token=Decimal("0.0000008"),
        cost_per_output_token=Decimal("0.000004"),
    ),
}


for _model_id, _caps in _ANTHROPIC_CAPABILITIES.items():
    register_capabilities(_model_id, _caps)
