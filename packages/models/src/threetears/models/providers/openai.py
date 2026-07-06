"""OpenAI-compatible chat and embedding factories backed by ``langchain_openai``.

LangChain-native shape (3tears v0.6.0+): :func:`create_openai_chat` returns
a configured ``ChatOpenAI`` instance and :func:`create_openai_embedding`
returns a configured ``OpenAIEmbeddings`` instance. Capability metadata
for known OpenAI model ids is registered with the module-level
:func:`~threetears.models.capabilities.register_capabilities` registry at
import time.

Tool-name translation: the OpenAI tools API validates tool names against
``^[a-zA-Z0-9_-]{1,64}$`` and rejects the dot. Canonical 3tears tool
names use the dotted form, so :func:`create_openai_chat` returns a
:class:`_NameTranslatingChatOpenAI` subclass that translates
dot-to-underscore on outgoing tool specs and underscore-to-dot on
incoming ``tool_calls``. The same wrapper covers OpenRouter accessed
via ``base_url`` (the gateway's standard OpenAI-compatible route).
Application code never sees the wire form. Translation primitives live
in :mod:`threetears.models.tool_name_translation`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator

from pydantic import PrivateAttr

from threetears.models.capabilities import ModelCapabilities, register_capabilities
from threetears.models.enums import ModelStatus, ModelTier, ModelType
from threetears.models.tool_name_translation import (
    build_name_translation,
    forward_translate_input,
    reverse_translate_message,
)
from threetears.models.tool_name_validation import filter_invalid_tool_calls
from threetears.observe import get_logger

if TYPE_CHECKING:
    from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
    from langchain_core.language_models.chat_models import LanguageModelInput
    from langchain_core.messages import AIMessageChunk, BaseMessage
    from langchain_core.outputs import ChatResult
    from langchain_core.runnables import Runnable, RunnableConfig
    from langchain_core.tools import BaseTool
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

__all__ = [
    "OPENAI_PROVIDER_NAME",
    "create_openai_chat",
    "create_openai_embedding",
]


OPENAI_PROVIDER_NAME = "openai"

_logger = get_logger(__name__)


def _drop_junk_invalid_tool_calls(message: Any) -> None:
    """drop ``invalid_tool_calls`` entries whose ``name`` fails validation.

    Parity with the OpenRouter / Anthropic wrappers. Mutates
    ``message.invalid_tool_calls`` in place, keeping only entries whose
    names match the canonical 3tears tool-name regex. Each rejected
    entry is logged once at WARNING (name truncated to 80 characters).
    See :mod:`threetears.models.providers.openrouter` for the prod
    incident write-up (2026-05-19).

    :param message: chat-model response (``AIMessage`` or
        ``AIMessageChunk``); duck-typed via attribute access
    :ptype message: Any
    """
    raw = getattr(message, "invalid_tool_calls", None) or []
    if not raw:
        return
    kept, rejected = filter_invalid_tool_calls(raw)
    if not rejected:
        return
    for entry in rejected:
        name = entry.get("name") if isinstance(entry, dict) else None
        truncated = name[:80] if isinstance(name, str) else repr(name)[:80]
        _logger.warning(
            "openai wrapper dropped invalid_tool_calls entry with junk name: %s",
            truncated,
        )
    raw.clear()
    raw.extend(kept)


def _build_translating_chat_class() -> type[ChatOpenAI]:
    """build the :class:`ChatOpenAI` subclass with name-translation hooks.

    Defined inside a function so ``langchain_openai`` stays a lazy
    import; the openai capability registry can populate without the
    optional dependency.

    :return: name-translating ChatOpenAI subclass
    :rtype: type[ChatOpenAI]
    """
    from langchain_openai import ChatOpenAI

    class _NameTranslatingChatOpenAI(ChatOpenAI):
        """``ChatOpenAI`` that translates tool names dot<->underscore at
        the wire boundary, mirroring the Anthropic/OpenRouter shape.

        :ivar _name_reverse_map: populated at ``bind_tools`` time;
            maps each tool's underscored wire name back to the
            canonical dotted form so ``tool_call`` names in
            streaming responses can be rewritten before they reach
            application code.
        :ptype _name_reverse_map: dict[str, str]
        """

        _name_reverse_map: dict[str, str] = PrivateAttr(default_factory=dict)

        def bind_tools(
            self,
            tools: list[BaseTool],
            **kwargs: Any,
        ) -> Runnable[LanguageModelInput, BaseMessage]:
            """bind tools after dot->underscore name translation for the wire.

            :param tools: application-side tool list (canonical dotted names)
            :ptype tools: list[BaseTool]
            :param kwargs: passthrough to ``super().bind_tools``
            :ptype kwargs: Any
            :return: runnable bound to wire-side proxy tools
            :rtype: Runnable[LanguageModelInput, BaseMessage]
            """
            wire_tools, reverse_map = build_name_translation(tools)
            self._name_reverse_map.clear()
            self._name_reverse_map.update(reverse_map)
            return super().bind_tools(wire_tools, **kwargs)

        async def astream(
            self,
            input: LanguageModelInput,
            config: RunnableConfig | None = None,
            *,
            stop: list[str] | None = None,
            **kwargs: Any,
        ) -> AsyncIterator[AIMessageChunk]:
            """stream AIMessageChunks with tool-call names un-translated.

            Parity fix with the OpenRouter / Anthropic wrappers: we
            override ``astream`` (the public Runnable method), NOT
            ``_astream``. Wrapping ``_astream`` in our own async
            generator -- even as a pass-through -- breaks LangGraph's
            ``astream_events(version="v2")`` event tap: chunks reach
            the consumer's ``async for`` loop but
            ``on_chat_model_stream`` callbacks never fire. See the
            OpenRouter wrapper module for the full incident write-up
            (2026-05-13). Today's path drives
            ``astream`` not ``astream_events``, so this isn't currently
            biting -- this fix lands the same parity contract so the
            next consumer to drive the v2 event tap through the
            OpenAI-compat wrapper (e.g. a consumer switching its OpenAI
            provider to ``create_openai_chat``) doesn't repeat the
            saga.

            :param input: chat input (messages or string)
            :ptype input: LanguageModelInput
            :param config: optional runnable config
            :ptype config: RunnableConfig | None
            :param stop: optional stop sequences
            :ptype stop: list[str] | None
            :param kwargs: passthrough to ``super().astream``
            :ptype kwargs: Any
            :return: async iterator of un-translated AIMessageChunks
            :rtype: AsyncIterator[AIMessageChunk]
            """
            # Pre-merge with the contextvar config so the
            # ``astream_events`` event_streamer (carried in the
            # contextvar's AsyncCallbackManager) survives
            # BaseChatModel.astream's ensure_config replace-by-key step.
            # See the OpenRouter wrapper for the full incident write-up
            # (2026-05-13).
            from langchain_core.runnables.config import ensure_config, merge_configs

            merged_config = merge_configs(ensure_config(None), config)
            # Forward-translate: mangle any dotted tool-call name in the
            # OUTBOUND history (a prior round's AIMessage, or a dotted
            # hallucination / MCP tool) to wire form before it reaches
            # OpenAI's ``^[a-zA-Z0-9_-]{1,64}$`` validator. Non-mutating (a
            # copy), symmetric with ``reverse_translate_message`` below.
            wire_input = forward_translate_input(input)
            async for chunk in super().astream(
                wire_input,
                config=merged_config,
                stop=stop,
                **kwargs,
            ):
                reverse_translate_message(chunk, self._name_reverse_map)
                # Drop junk-name ``invalid_tool_calls`` entries before they
                # reach downstream dispatch / persistence (parity with the
                # OpenRouter / Anthropic wrappers).
                _drop_junk_invalid_tool_calls(chunk)
                yield chunk

        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            """non-streaming generate with tool-call names un-translated.

            :param messages: chat messages
            :ptype messages: list[BaseMessage]
            :param stop: optional stop sequences
            :ptype stop: list[str] | None
            :param run_manager: LangChain run manager
            :ptype run_manager: AsyncCallbackManagerForLLMRun | None
            :param kwargs: passthrough
            :ptype kwargs: Any
            :return: chat result with un-translated tool-call names
            :rtype: ChatResult
            """
            # Forward-translate outbound dotted tool-call names (see the
            # ``astream`` override) before the provider validates them.
            result = await super()._agenerate(
                forward_translate_input(messages),
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
            for generation in result.generations:
                reverse_translate_message(generation.message, self._name_reverse_map)
                # Drop junk-name ``invalid_tool_calls`` entries (parity with
                # the OpenRouter / Anthropic wrappers).
                _drop_junk_invalid_tool_calls(generation.message)
            return result

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
