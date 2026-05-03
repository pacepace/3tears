"""OpenAI-compatible chat and embedding factories backed by ``langchain_openai``.

LangChain-native shape (3tears v0.6.0+): :func:`create_openai_chat` returns
a configured ``ChatOpenAI`` instance and :func:`create_openai_embedding`
returns a configured ``OpenAIEmbeddings`` instance. Capability metadata
for known OpenAI model ids is registered with the module-level
:func:`~threetears.models.capabilities.register_capabilities` registry at
import time.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from threetears.models.capabilities import ModelCapabilities, register_capabilities
from threetears.models.enums import ModelStatus, ModelTier, ModelType

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

__all__ = [
    "OPENAI_PROVIDER_NAME",
    "create_openai_chat",
    "create_openai_embedding",
]


OPENAI_PROVIDER_NAME = "openai"


def create_openai_chat(
    model_name: str,
    api_key: str,
    *,
    base_url: str | None = None,
    timeout: int = 120,
    max_retries: int = 2,
    stream_usage: bool = True,
    **extra_kwargs: object,
) -> ChatOpenAI:
    """creates a configured ``ChatOpenAI`` for OpenAI-compatible providers.

    :param model_name: OpenAI model identifier (e.g. ``gpt-4o``)
    :ptype model_name: str
    :param api_key: API key
    :ptype api_key: str
    :param base_url: optional custom API base URL (passed through unchanged)
    :ptype base_url: str | None
    :param timeout: request timeout in seconds
    :ptype timeout: int
    :param max_retries: maximum retry attempts for failed requests
    :ptype max_retries: int
    :param stream_usage: enable streaming usage metadata (token counts)
    :ptype stream_usage: bool
    :param extra_kwargs: additional keyword arguments forwarded to ``ChatOpenAI``
    :ptype extra_kwargs: object
    :return: configured ``ChatOpenAI`` instance
    :rtype: ChatOpenAI
    """
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, object] = {
        "model": model_name,
        "api_key": api_key,
        "timeout": timeout,
        "max_retries": max_retries,
        "stream_usage": stream_usage,
    }
    if base_url is not None:
        kwargs["base_url"] = base_url
    kwargs.update(extra_kwargs)

    model: ChatOpenAI = ChatOpenAI(**kwargs)
    return model


def create_openai_embedding(
    model_name: str,
    api_key: str,
    *,
    base_url: str | None = None,
    embedding_dimensions: int | None = None,
    **extra_kwargs: object,
) -> OpenAIEmbeddings:
    """creates a configured ``OpenAIEmbeddings`` for OpenAI-compatible providers.

    :param model_name: OpenAI embedding model identifier (e.g. ``text-embedding-3-small``)
    :ptype model_name: str
    :param api_key: API key
    :ptype api_key: str
    :param base_url: optional custom API base URL (passed through unchanged)
    :ptype base_url: str | None
    :param embedding_dimensions: optional output vector dimensionality (only honoured by models that support it)
    :ptype embedding_dimensions: int | None
    :param extra_kwargs: additional keyword arguments forwarded to ``OpenAIEmbeddings``
    :ptype extra_kwargs: object
    :return: configured ``OpenAIEmbeddings`` instance
    :rtype: OpenAIEmbeddings
    """
    from langchain_openai import OpenAIEmbeddings

    kwargs: dict[str, object] = {
        "model": model_name,
        "api_key": api_key,
    }
    if base_url is not None:
        kwargs["base_url"] = base_url
    if embedding_dimensions is not None:
        kwargs["dimensions"] = embedding_dimensions
    kwargs.update(extra_kwargs)

    model: OpenAIEmbeddings = OpenAIEmbeddings(**kwargs)
    return model


# -- capability registration -------------------------------------------------

# canonical OpenAI models. extend by calling register_capabilities() at
# host-app boot time for additional ids.
_OPENAI_CAPABILITIES: dict[str, ModelCapabilities] = {
    "gpt-4o": ModelCapabilities(
        model_name="gpt-4o",
        provider_name=OPENAI_PROVIDER_NAME,
        model_type=ModelType.CHAT,
        model_tier=ModelTier.LARGE,
        model_status=ModelStatus.ACTIVE,
        context_window=128_000,
        max_output_tokens=16_384,
        supports_streaming=True,
        supports_tools=True,
        supports_vision=True,
        requires_alternating_roles=False,
        cost_per_input_token=Decimal("0.0000025"),
        cost_per_output_token=Decimal("0.00001"),
    ),
    "gpt-4o-mini": ModelCapabilities(
        model_name="gpt-4o-mini",
        provider_name=OPENAI_PROVIDER_NAME,
        model_type=ModelType.CHAT,
        model_tier=ModelTier.SMALL,
        model_status=ModelStatus.ACTIVE,
        context_window=128_000,
        max_output_tokens=16_384,
        supports_streaming=True,
        supports_tools=True,
        supports_vision=True,
        requires_alternating_roles=False,
        cost_per_input_token=Decimal("0.00000015"),
        cost_per_output_token=Decimal("0.0000006"),
    ),
    "text-embedding-3-small": ModelCapabilities(
        model_name="text-embedding-3-small",
        provider_name=OPENAI_PROVIDER_NAME,
        model_type=ModelType.EMBEDDING,
        model_tier=ModelTier.SMALL,
        model_status=ModelStatus.ACTIVE,
        embedding_dimensions=1536,
        max_embedding_tokens=8_191,
        supports_batch_embedding=True,
        cost_per_input_token=Decimal("0.00000002"),
    ),
    "text-embedding-3-large": ModelCapabilities(
        model_name="text-embedding-3-large",
        provider_name=OPENAI_PROVIDER_NAME,
        model_type=ModelType.EMBEDDING,
        model_tier=ModelTier.LARGE,
        model_status=ModelStatus.ACTIVE,
        embedding_dimensions=3072,
        max_embedding_tokens=8_191,
        supports_batch_embedding=True,
        cost_per_input_token=Decimal("0.00000013"),
    ),
}


for _model_id, _caps in _OPENAI_CAPABILITIES.items():
    register_capabilities(_model_id, _caps)
