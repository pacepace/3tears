"""OpenAI-compatible chat and embedding factories backed by ``langchain_openai``.

LangChain-native shape (3tears v0.6.0+): :func:`create_openai_chat` returns
a configured ``ChatOpenAI`` instance and :func:`create_openai_embedding`
returns a configured ``OpenAIEmbeddings`` instance. Capability metadata
for known OpenAI model ids is registered with the module-level
:func:`~threetears.models.capabilities.register_capabilities` registry at
import time.

Tool-name translation: the OpenAI tools API validates tool names against
``^[a-zA-Z0-9_-]{1,64}$`` and rejects the dot. Canonical 3tears tool names
use the dotted form, so :func:`create_openai_chat` returns a
:class:`_NameTranslatingChatOpenAI` subclass that translates
dot-to-underscore on outgoing tool specs / history and underscore-to-dot on
incoming ``tool_calls``. The wrapper is a thin binding of the shared
:class:`~threetears.models.providers._name_translation_mixin.NameTranslatingChatMixin`
(identical hooks across the OpenAI / OpenRouter / Anthropic wrappers).
Application code never sees the wire form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import PrivateAttr

from threetears.models.capabilities import ModelCapabilities, register_capabilities
from threetears.models.enums import ModelStatus, ModelTier, ModelType
from threetears.models.providers._name_translation_mixin import NameTranslatingChatMixin

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings


__all__ = [
    "OPENAI_PROVIDER_NAME",
    "create_openai_chat",
    "create_openai_embedding",
]


OPENAI_PROVIDER_NAME = "openai"


def _build_translating_chat_class() -> type[ChatOpenAI]:
    """build the :class:`ChatOpenAI` subclass with name-translation hooks.

    Defined inside a function so ``langchain_openai`` stays a lazy
    import; the openai capability registry can populate without the
    optional dependency.

    :return: name-translating ChatOpenAI subclass
    :rtype: type[ChatOpenAI]
    """
    from langchain_openai import ChatOpenAI

    class _NameTranslatingChatOpenAI(NameTranslatingChatMixin, ChatOpenAI):
        """``ChatOpenAI`` with dot<->underscore tool-name translation at the
        wire boundary (OpenAI validates names against ``^[a-zA-Z0-9_-]{1,64}$``
        and rejects the dot).

        All translation hooks live in :class:`NameTranslatingChatMixin`
        (mixed in ahead of ``ChatOpenAI`` so ``super()`` resolves to it); this
        subclass only supplies the per-instance reverse-map slot.

        :ivar _name_reverse_map: underscored-wire -> canonical-dotted map,
            populated at ``bind_tools`` time.
        :ptype _name_reverse_map: dict[str, str]
        """

        _name_reverse_map: dict[str, str] = PrivateAttr(default_factory=dict)

    return _NameTranslatingChatOpenAI


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

    Returns the :class:`_NameTranslatingChatOpenAI` subclass so dotted
    canonical tool names round-trip through OpenAI's strict tool-name
    validator. Application code interacts with it exactly the same way
    as a vanilla ``ChatOpenAI``.

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
    :return: configured ``ChatOpenAI`` (the name-translating subclass)
    :rtype: ChatOpenAI
    """
    chat_cls = _build_translating_chat_class()

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

    model: ChatOpenAI = chat_cls(**kwargs)
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
        supports_anthropic_cache_control=False,
        supports_openai_auto_cache=True,
        min_cacheable_tokens=0,
        cache_ttl_seconds=0,
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
        supports_anthropic_cache_control=False,
        supports_openai_auto_cache=True,
        min_cacheable_tokens=0,
        cache_ttl_seconds=0,
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
    ),
}


for _model_id, _caps in _OPENAI_CAPABILITIES.items():
    register_capabilities(_model_id, _caps)
