"""anthropic chat factory returning a configured ``langchain_anthropic.ChatAnthropic``.

LangChain-native shape (3tears v0.6.0+): :func:`create_anthropic_chat`
returns a fully-configured ``ChatAnthropic`` instance. Capability metadata
for known Anthropic model ids is registered with the module-level
:func:`~threetears.models.capabilities.register_capabilities` registry at
import time so consumers can ``get_capabilities(model_id)`` without
instantiating the provider.

Tool-name translation: the Anthropic Messages API validates tool names
against ``^[a-zA-Z0-9_-]{1,128}$`` and rejects the dot. Canonical 3tears
tool names use the dotted form (``threetears.calculator``,
``aibots.admin.agent_management``), so :func:`create_anthropic_chat`
returns a :class:`_NameTranslatingChatAnthropic` subclass that translates
dot-to-underscore on outgoing tool specs and underscore-to-dot on
incoming ``tool_calls``. Application code never sees the wire form. The
translation primitives live in
:mod:`threetears.models.tool_name_translation` and are shared across
every provider whose validator forces the same rename (Anthropic-direct,
OpenRouter-routed Bedrock, etc.).
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any, AsyncIterator

from langchain_core.outputs import ChatGeneration
from pydantic import PrivateAttr

from threetears.models.capabilities import ModelCapabilities, register_capabilities
from threetears.models.enums import ModelStatus, ModelTier, ModelType
from threetears.models.tool_name_translation import (
    build_name_translation,
    reverse_translate_message,
)
from threetears.models.tool_name_validation import filter_invalid_tool_calls
from threetears.observe import get_logger

if TYPE_CHECKING:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
    from langchain_core.language_models import BaseChatModel
    from langchain_core.language_models.chat_models import LanguageModelInput
    from langchain_core.messages import AIMessageChunk, BaseMessage
    from langchain_core.outputs import ChatResult
    from langchain_core.runnables import Runnable, RunnableConfig
    from langchain_core.tools import BaseTool

__all__ = [
    "ANTHROPIC_PROVIDER_NAME",
    "create_anthropic_chat",
    "strip_v1_suffix",
]


ANTHROPIC_PROVIDER_NAME = "anthropic"

_logger = get_logger(__name__)


def _drop_junk_invalid_tool_calls(message: Any) -> None:
    """drop ``invalid_tool_calls`` entries whose ``name`` fails validation.

    Mirror of the OpenRouter wrapper's hook. Mutates
    ``message.invalid_tool_calls`` in place, keeping only entries
    whose names match the canonical 3tears tool-name regex. Each
    rejected entry is logged once at WARNING (name truncated to 80
    characters). See
    :mod:`threetears.models.providers.openrouter` for the prod
    incident write-up (metallm conv
    ``019e3e26-9870-7a03-8f04-8cc6a4f5f418``, 2026-05-19).

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
            "anthropic wrapper dropped invalid_tool_calls entry with junk name: %s",
            truncated,
        )
    raw.clear()
    raw.extend(kept)


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


def _build_translating_chat_class() -> type[ChatAnthropic]:
    """build the :class:`ChatAnthropic` subclass with name-translation hooks.

    Defined inside a function so the langchain-anthropic import stays
    lazy -- :mod:`threetears.models.providers.anthropic` is imported
    eagerly at package load to populate the capability registry, and
    we must not require ``langchain-anthropic`` to be installed for
    that side-effect import to succeed (it is an optional dependency
    selected via ``3tears-models[anthropic]``).

    :return: name-translating ChatAnthropic subclass
    :rtype: type[ChatAnthropic]
    """
    from langchain_anthropic import ChatAnthropic

    class _NameTranslatingChatAnthropic(ChatAnthropic):
        """``ChatAnthropic`` that translates tool names dot<->underscore
        at the wire boundary.

        The Anthropic Messages API rejects tool names that fail
        ``^[a-zA-Z0-9_-]{1,128}$``. Canonical 3tears tool names use
        the dotted form. This subclass mangles names on outgoing
        ``bind_tools`` and reverses them on every streaming /
        non-streaming response so application code only ever sees
        the canonical dotted form. See
        :mod:`threetears.models.tool_name_translation` for the
        primitives + the openrouter integration that wears the
        same translation layer.

        :ivar _name_reverse_map: populated at ``bind_tools`` time;
            maps each tool's underscored wire name back to the
            canonical dotted form so ``tool_call`` names in
            streaming responses can be rewritten before they reach
            the application's dispatch / logging / persistence
            layers.
        :ptype _name_reverse_map: dict[str, str]
        """

        _name_reverse_map: dict[str, str] = PrivateAttr(default_factory=dict)

        def bind_tools(
            self,
            tools: list[BaseTool],
            **kwargs: Any,
        ) -> Runnable[LanguageModelInput, BaseMessage]:
            """bind tools after dot->underscore name translation for the wire.

            :param tools: application-side tool list (canonical
                dotted names)
            :ptype tools: list[BaseTool]
            :param kwargs: passthrough to ``super().bind_tools``
            :ptype kwargs: Any
            :return: runnable bound to wire-side proxy tools
            :rtype: Runnable[LanguageModelInput, BaseMessage]
            """
            wire_tools, reverse_map = build_name_translation(tools)
            # mutate the shared reverse_map rather than reassign so
            # the ``_astream`` closure in concurrently-running
            # streams still sees the same dict object.
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

            We override ``astream`` (the public Runnable method) and
            NOT ``_astream`` (the protected hook). Wrapping ``_astream``
            in our own async generator -- even as a pass-through --
            broke LangGraph's ``astream_events(version="v2")`` event
            tap: chunks reached the consumer's ``async for`` loop but
            the framework's ``on_chat_model_stream`` callbacks never
            fired, leaving event-driven UIs (e.g. metallm's WS handler)
            with the saved DB content but a blank live stream. Same
            failure mode as the OpenRouter wrapper, same root cause,
            same fix (see :mod:`threetears.models.providers.openrouter`
            for the OpenRouter side and the regression-test rationale).

            Overriding ``astream`` means ``BaseChatModel.astream``'s
            callback wiring runs unchanged against the parent's
            untouched ``_astream`` output, and we post-process the
            ``AIMessageChunk`` objects as they're yielded to us.

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
            # (metallm 2026-05-13, conv ``019e2243-de0c``).
            from langchain_core.runnables.config import ensure_config, merge_configs

            merged_config = merge_configs(ensure_config(None), config)
            async for chunk in super().astream(
                input,
                config=merged_config,
                stop=stop,
                **kwargs,
            ):
                reverse_translate_message(chunk, self._name_reverse_map)
                # Drop junk-name ``invalid_tool_calls`` entries
                # before they reach downstream dispatch / persistence
                # (see :func:`_drop_junk_invalid_tool_calls`).
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

            Mirrors :meth:`_astream` for the non-streaming code path
            (``ainvoke`` and friends). Walks every generation in the
            ``ChatResult`` and rewrites ``tool_calls`` /
            ``invalid_tool_calls`` names back to the canonical form
            before returning to the caller.

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
            result = await super()._agenerate(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
            for generation in result.generations:
                reverse_translate_message(generation.message, self._name_reverse_map)
                _drop_junk_invalid_tool_calls(generation.message)
            return result

        async def ainvoke(
            self,
            input: LanguageModelInput,
            config: RunnableConfig | None = None,
            *,
            stop: list[str] | None = None,
            **kwargs: Any,
        ) -> BaseMessage:
            """invoke (non-streaming public API) with names un-translated.

            Overriding ``_agenerate`` (above) is NOT sufficient. When
            streaming callbacks are present -- e.g. ``model.ainvoke`` under
            an outer ``astream_events`` tap (the converged ``agent_node``
            path) -- ``BaseChatModel.ainvoke`` aggregates from the PROTECTED
            ``self._astream`` via ``_agenerate_with_cache`` instead of
            calling ``_agenerate``, bypassing BOTH the public ``astream``
            override AND ``_agenerate``. Tool-call names would then reach the
            caller in their wire (underscored) form and miss the dotted
            dispatch map. We override the PUBLIC ``ainvoke`` (same strategy as
            the ``astream`` override -- wrapping the protected ``_astream``
            would drop ``on_chat_model_stream`` callbacks) and post-process
            the single returned message; ``reverse_translate_message`` keys on
            the underscored wire name, so a second pass is a no-op.

            :param input: chat input (messages or string)
            :ptype input: LanguageModelInput
            :param config: optional runnable config
            :ptype config: RunnableConfig | None
            :param stop: optional stop sequences
            :ptype stop: list[str] | None
            :param kwargs: passthrough to ``super().ainvoke``
            :ptype kwargs: Any
            :return: response message with canonical (dotted) tool-call names
            :rtype: BaseMessage
            """
            from langchain_core.runnables.config import ensure_config, merge_configs

            # Pre-merge like the ``astream`` override: a plain-list ``callbacks``
            # in ``config`` would otherwise overwrite the contextvar's callback
            # manager (carrying the ``astream_events`` event_streamer) inside
            # ``BaseChatModel.ainvoke``'s ``ensure_config``. ``merge_configs``
            # folds the list into the manager instead, preserving the tap.
            merged_config = merge_configs(ensure_config(None), config)
            result = await super().ainvoke(
                input,
                config=merged_config,
                stop=stop,
                **kwargs,
            )
            reverse_translate_message(result, self._name_reverse_map)
            _drop_junk_invalid_tool_calls(result)
            return result

        def invoke(
            self,
            input: LanguageModelInput,
            config: RunnableConfig | None = None,
            *,
            stop: list[str] | None = None,
            **kwargs: Any,
        ) -> BaseMessage:
            """sync mirror of :meth:`ainvoke` (same bypass, same fix).

            The sync path aggregates from the protected ``_stream`` via
            ``_generate_with_cache`` when streaming callbacks are present, so
            post-process the returned message for sync callers too.

            :param input: chat input (messages or string)
            :ptype input: LanguageModelInput
            :param config: optional runnable config
            :ptype config: RunnableConfig | None
            :param stop: optional stop sequences
            :ptype stop: list[str] | None
            :param kwargs: passthrough to ``super().invoke``
            :ptype kwargs: Any
            :return: response message with canonical (dotted) tool-call names
            :rtype: BaseMessage
            """
            from langchain_core.runnables.config import ensure_config, merge_configs

            # Pre-merge to preserve a callback-manager ``callbacks`` (see the
            # ``ainvoke`` override above for the rationale).
            merged_config = merge_configs(ensure_config(None), config)
            result = super().invoke(
                input,
                config=merged_config,
                stop=stop,
                **kwargs,
            )
            reverse_translate_message(result, self._name_reverse_map)
            _drop_junk_invalid_tool_calls(result)
            return result

        async def agenerate(
            self,
            messages: list[list[BaseMessage]],
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            """un-translate tool names on the batch generate surface.

            ``agenerate`` is the chokepoint ``ainvoke`` / ``abatch`` route
            through, and aggregates from the protected ``_astream`` when
            streaming callbacks are present (bypassing ``_agenerate``).
            Post-process every generated message so a direct ``agenerate``
            caller also sees canonical dotted names; idempotent with the other
            overrides.

            :param messages: batch of message lists
            :ptype messages: list[list[BaseMessage]]
            :param args: positional passthrough to ``super().agenerate``
            :ptype args: Any
            :param kwargs: keyword passthrough to ``super().agenerate``
            :ptype kwargs: Any
            :return: LLMResult with canonical (dotted) tool-call names
            :rtype: Any
            """
            result = await super().agenerate(messages, *args, **kwargs)
            for generations in result.generations:
                for generation in generations:
                    # chat models always yield ChatGeneration(Chunk); the
                    # isinstance narrow proves ``.message`` exists (the base
                    # Generation union member has no such attribute).
                    if isinstance(generation, ChatGeneration):
                        reverse_translate_message(generation.message, self._name_reverse_map)
                        _drop_junk_invalid_tool_calls(generation.message)
            return result

        def generate(
            self,
            messages: list[list[BaseMessage]],
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            """sync mirror of :meth:`agenerate` (same bypass, same fix).

            :param messages: batch of message lists
            :ptype messages: list[list[BaseMessage]]
            :param args: positional passthrough to ``super().generate``
            :ptype args: Any
            :param kwargs: keyword passthrough to ``super().generate``
            :ptype kwargs: Any
            :return: LLMResult with canonical (dotted) tool-call names
            :rtype: Any
            """
            result = super().generate(messages, *args, **kwargs)
            for generations in result.generations:
                for generation in generations:
                    # see ``agenerate`` for why the isinstance narrow is needed.
                    if isinstance(generation, ChatGeneration):
                        reverse_translate_message(generation.message, self._name_reverse_map)
                        _drop_junk_invalid_tool_calls(generation.message)
            return result

    return _NameTranslatingChatAnthropic


def create_anthropic_chat(
    model_name: str,
    api_key: str,
    *,
    base_url: str | None = None,
    timeout: int = 120,
    max_retries: int = 2,
    **extra_kwargs: object,
) -> BaseChatModel:
    """creates a configured ``ChatAnthropic`` for Anthropic models.

    Returns the :class:`_NameTranslatingChatAnthropic` subclass, not
    a vanilla ``ChatAnthropic``. Application code interacts with it
    exactly the same way (it IS a ``ChatAnthropic``); the only
    difference is the wire-side tool-name translation that hides
    Anthropic's strict tool-name regex from the rest of the
    codebase.

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
    :return: configured ``ChatAnthropic`` (the name-translating subclass), or a Claude
        **subscription**-backed model when ``api_key`` is an OAuth token (``sk-ant-oat…``).
    :rtype: BaseChatModel
    """
    # A Claude subscription OAuth token (``claude setup-token``) routes to the CLI/Agent-SDK backend
    # instead of the HTTP API — the SAME Anthropic model ids, no separate provider. An API key
    # (``sk-ant-api…``) takes the ChatAnthropic path below. Imported lazily so the optional
    # ``langchain-claude-code`` dep is only pulled when a subscription token is actually used.
    from threetears.models.providers._claude_cli import create_subscription_chat, is_subscription_token

    if is_subscription_token(api_key):
        return create_subscription_chat(model_name, api_key, **extra_kwargs)

    chat_cls = _build_translating_chat_class()

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

    model: ChatAnthropic = chat_cls(**kwargs)
    return model


# -- capability registration -------------------------------------------------

# canonical Anthropic chat models. extend with additional ids by calling
# register_capabilities() externally at host-app boot time.
_ANTHROPIC_CAPABILITIES: dict[str, ModelCapabilities] = {
    "claude-opus-4-8": ModelCapabilities(
        model_name="claude-opus-4-8",
        provider_name=ANTHROPIC_PROVIDER_NAME,
        model_type=ModelType.CHAT,
        model_tier=ModelTier.LARGE,
        model_status=ModelStatus.ACTIVE,
        context_window=200_000,
        max_output_tokens=64_000,
        supports_streaming=True,
        supports_tools=True,
        supports_vision=True,
        requires_alternating_roles=False,
        supports_anthropic_cache_control=True,
        supports_openai_auto_cache=False,
        min_cacheable_tokens=1024,
        cache_ttl_seconds=300,
        cost_per_input_token=Decimal("0.000015"),
        cost_per_output_token=Decimal("0.000075"),
        cost_per_cache_read_token=Decimal("0.0000015"),
        cost_per_cache_write_token=Decimal("0.00001875"),
    ),
    "claude-sonnet-4-6": ModelCapabilities(
        model_name="claude-sonnet-4-6",
        provider_name=ANTHROPIC_PROVIDER_NAME,
        model_type=ModelType.CHAT,
        model_tier=ModelTier.LARGE,
        model_status=ModelStatus.ACTIVE,
        context_window=200_000,
        max_output_tokens=64_000,
        supports_streaming=True,
        supports_tools=True,
        supports_vision=True,
        requires_alternating_roles=False,
        supports_anthropic_cache_control=True,
        supports_openai_auto_cache=False,
        min_cacheable_tokens=1024,
        cache_ttl_seconds=300,
        cost_per_input_token=Decimal("0.000003"),
        cost_per_output_token=Decimal("0.000015"),
        cost_per_cache_read_token=Decimal("0.0000003"),
        cost_per_cache_write_token=Decimal("0.00000375"),
    ),
    "claude-haiku-4-5-20251001": ModelCapabilities(
        model_name="claude-haiku-4-5-20251001",
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
        supports_anthropic_cache_control=True,
        supports_openai_auto_cache=False,
        min_cacheable_tokens=1024,
        cache_ttl_seconds=300,
        cost_per_input_token=Decimal("0.0000008"),
        cost_per_output_token=Decimal("0.000004"),
        cost_per_cache_read_token=Decimal("0.00000008"),
        cost_per_cache_write_token=Decimal("0.000001"),
    ),
}


for _model_id, _caps in _ANTHROPIC_CAPABILITIES.items():
    register_capabilities(_model_id, _caps)
