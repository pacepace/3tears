"""OpenRouter chat factory backed by ``langchain_openrouter``.

LangChain-native shape (3tears v0.6.0+): :func:`create_openrouter_chat`
returns a fully-configured ``ChatOpenRouter`` instance. ``ChatOpenRouter``
expects the request timeout in milliseconds — the factory accepts seconds
to match the rest of the API surface and converts internally.

Tool-name translation: OpenRouter routes some upstream models through
backends with strict tool-name validators (Bedrock requires
``^[a-zA-Z0-9_-]{1,128}$`` -- no dots), but the canonical 3tears tool
name is the dotted ``threetears.X`` form. The factory returns a
:class:`_NameTranslatingChatOpenRouter` subclass that translates
dot-to-underscore on outgoing tool specs and underscore-to-dot on
incoming ``tool_calls``, so application code (3tears core and other
consumers) never sees the wire form. The
translation is keyed by the dotted -> underscored mapping built at
``bind_tools`` time, so it round-trips losslessly even for tools whose
names contain underscores already (e.g. ``threetears.web_search`` ->
``threetears_web_search`` and back).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator

from langchain_core.outputs import ChatGeneration
from langchain_core.tools import BaseTool
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
    from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
    from langchain_core.language_models.chat_models import LanguageModelInput
    from langchain_core.messages import AIMessageChunk, BaseMessage
    from langchain_core.outputs import ChatResult
    from langchain_core.runnables import Runnable, RunnableConfig
    from langchain_openrouter import ChatOpenRouter

__all__ = [
    "OPENROUTER_PROVIDER_NAME",
    "create_openrouter_chat",
]


OPENROUTER_PROVIDER_NAME = "openrouter"

_logger = get_logger(__name__)


def _drop_junk_invalid_tool_calls(message: Any) -> None:
    """drop ``invalid_tool_calls`` entries whose ``name`` fails validation.

    Mutates ``message.invalid_tool_calls`` in place, replacing it with
    only the entries whose names match the canonical 3tears tool-name
    regex. Each rejected entry is logged once at WARNING (name
    truncated to 80 characters to bound output size and prevent
    log-injection). Junk names like the prod-observed
    ``memory_recall" name="memory_recall`` (2026-05-19) cannot
    dispatch and would otherwise propagate through
    consumer tool routing as a recovery target.

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
            "openrouter wrapper dropped invalid_tool_calls entry with junk name: %s",
            truncated,
        )
    # mutate the list in place so consumers holding a reference see
    # the filtered view; AIMessage / AIMessageChunk both back the
    # field with a regular list.
    raw.clear()
    raw.extend(kept)


def _build_translating_chat_class() -> type[ChatOpenRouter]:
    """build the :class:`ChatOpenRouter` subclass with name-translation hooks.

    Defined inside a function so the langchain-openrouter import is
    lazy -- :mod:`threetears.models.providers.openrouter` is imported
    eagerly at package load to populate the capability registry, and we
    must not require ``langchain-openrouter`` to be installed for that
    side-effect import to succeed (it is an optional dependency,
    selected via ``3tears-models[openrouter]``).
    """
    from langchain_openrouter import ChatOpenRouter

    class _NameTranslatingChatOpenRouter(ChatOpenRouter):
        """``ChatOpenRouter`` that translates tool names dot<->underscore
        at the wire boundary, hiding the wire form from application code.

        :ivar _name_reverse_map: populated at ``bind_tools`` time;
            maps each tool's underscored wire name back to the
            canonical dotted form so ``tool_call`` names in streaming
            responses can be rewritten before they reach the
            application's dispatch / logging / persistence layers.
        :ptype _name_reverse_map: dict[str, str]
        """

        _name_reverse_map: dict[str, str] = PrivateAttr(default_factory=dict)

        def bind_tools(
            self,
            tools: list[BaseTool],
            **kwargs: Any,
        ) -> Runnable[LanguageModelInput, BaseMessage]:
            """bind tools after dot->underscore name translation for the wire.

            Application-side tools keep their canonical dotted names;
            the bound runnable holds wire-side proxies whose ``.name``
            is the underscored form. The reverse map for response
            un-translation is stored on this instance so ``_astream``
            and ``_agenerate`` can rewrite tool-call names.

            :param tools: application-side tool list (canonical dotted
                names)
            :ptype tools: list[BaseTool]
            :param kwargs: passthrough to ``super().bind_tools``
            :ptype kwargs: Any
            :return: runnable bound to wire-side proxy tools
            :rtype: Runnable[LanguageModelInput, BaseMessage]
            """
            wire_tools, reverse_map = build_name_translation(tools)
            # Mutate the shared reverse_map rather than reassign so the
            # ``_astream`` closure in concurrently-running streams still
            # sees the same dict object (PrivateAttr is per-instance,
            # and this instance is request-scoped in every consumer
            # observed -- a consumer's ``build_chat_model_from_config``
            # constructs a fresh model per resolve_chat_model call).
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
            NOT ``_astream`` (the protected hook) on purpose. Wrapping
            ``_astream`` in our own async generator -- even as a
            pass-through -- breaks LangGraph's
            ``astream_events(version="v2")`` event tap: chunks reach
            the consumer's ``async for`` loop but the framework's
            ``on_chat_model_stream`` callbacks never fire, leaving
            event-driven UIs (e.g. a consumer's WS handler) with the saved
            DB content but a blank live stream. The cause: the
            callback-firing path lives inside ``BaseChatModel.astream``
            and depends on the unaltered ``self._astream`` async
            generator to drive ``run_manager.on_llm_new_token`` calls
            per chunk; routing chunks through an extra generator layer
            in our override silently dropped those callbacks for some
            downstream consumers (observed in production
            on 2026-05-13, 190 chunks delivered, 0 stream
            events emitted).

            Overriding ``astream`` instead means
            ``BaseChatModel.astream``'s callback wiring runs unchanged
            against the parent's ``_astream`` output, and we post-
            process the ``AIMessageChunk`` objects as they're yielded
            to us. Tool-call name translation still happens on every
            chunk; event emission still works because we're outside
            the callback-firing loop.

            CRITICAL — config merge (2026-05-13 fix): when this method
            is called via ``RunnableBinding.astream`` (the wrapper
            produced by ``model.with_config(callbacks=[...])`` inside
            :func:`threetears.models.factory.create_chat_model`), the
            ``config`` argument we receive holds the bound
            ``callbacks=[UsageTracker, CircuitBreaker]`` as a plain
            list. The contextvar ``var_child_runnable_config`` --
            populated by LangGraph's node wrapper with the parent's
            run-manager-as-``AsyncCallbackManager`` -- carries the
            ``astream_events`` event_streamer. If we forward
            ``config=config`` verbatim to ``super().astream(...)``,
            ``BaseChatModel.astream``'s ``ensure_config(config)``
            performs a plain ``dict.update`` that REPLACES the
            contextvar's manager (with event_streamer inside) with the
            input's list -- silently dropping the event_streamer for
            the entire stream. Result: chunks reach the personality
            node and get persisted, but no ``on_chat_model_stream``
            events fire and the live UI stays blank (the exact
            ``saved_content_length > 0`` /
            ``tokens_dispatched_count == 0`` fingerprint observed in
            production with the post-tool-executor sonnet/openrouter call on
            2026-05-13).

            The fix is to pre-merge with
            :func:`merge_configs(ensure_config(None), config)`, which
            uses the smart per-key callbacks merge in
            ``merge_configs`` -- a list-into-manager merge clones the
            manager and adds each list callback as a handler with
            ``inherit=True``. The resulting merged config has
            ``callbacks`` as a single manager that holds both the
            event_streamer AND the bound tracking callbacks, so
            ``ensure_config(merged)`` inside ``BaseChatModel.astream``
            preserves everything.

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
            from langchain_core.runnables.config import ensure_config, merge_configs

            merged_config = merge_configs(ensure_config(None), config)
            async for chunk in super().astream(
                input,
                config=merged_config,
                stop=stop,
                **kwargs,
            ):
                # ``BaseChatModel.astream`` yields ``AIMessageChunk``
                # directly (it unwraps ``ChatGenerationChunk.message``
                # before yielding). The AIMessageChunk carries the
                # tool-call fields ``reverse_translate_message`` rewrites.
                reverse_translate_message(chunk, self._name_reverse_map)
                # Drop junk-name ``invalid_tool_calls`` entries (e.g.
                # the XML-attribute-leak shape observed in production,
                # 2026-05-19)
                # before they reach downstream dispatch / persistence.
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
            """
            result = await super()._agenerate(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
            for generation in result.generations:
                reverse_translate_message(generation.message, self._name_reverse_map)
                # Drop junk-name ``invalid_tool_calls`` entries
                # before the non-streaming response reaches the
                # caller (mirrors the streaming path above).
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
            streaming callbacks are present -- exactly the converged
            ``agent_node`` path, where ``model.ainvoke`` runs under the
            outer ``astream_events`` tap -- ``BaseChatModel.ainvoke`` routes
            through ``_agenerate_with_cache``, which aggregates from the
            PROTECTED ``self._astream`` (``chat_models.py``: ``elif
            self._should_stream(...): async for chunk in self._astream(...)``)
            instead of calling ``_agenerate``. That bypasses BOTH the public
            ``astream`` override AND ``_agenerate``, so tool-call names would
            reach the caller in their wire (underscored) form and miss the
            dotted dispatch map (observed: a converged loop emitting
            ``threetears_web_search`` that the tool node could not resolve,
            2026-06-22).

            We override the PUBLIC ``ainvoke`` -- the same strategy the
            ``astream`` override uses, and for the same reason: wrapping the
            protected ``_astream`` would drop ``on_chat_model_stream``
            callbacks. Post-processing the single returned message catches
            the result no matter which internal path (``_astream`` aggregate
            or ``_agenerate``) produced it. Double-translation is impossible:
            ``reverse_translate_message`` keys on the underscored wire name,
            so a second pass over already-dotted names is a no-op.

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

            The sync ``invoke`` path has the identical exposure: when
            streaming callbacks are present ``_generate_with_cache``
            aggregates from the protected ``_stream`` rather than calling
            ``_generate``. Post-process the returned message so sync callers
            also see canonical dotted tool-call names.

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

            ``agenerate`` is the chokepoint that ``ainvoke`` / ``abatch`` route
            through, and it too aggregates from the protected ``_astream`` when
            streaming callbacks are present (bypassing ``_agenerate``). A direct
            ``agenerate`` caller would otherwise see wire (underscored) names.
            Post-process every generated message; idempotent with the
            ``ainvoke`` / ``_agenerate`` overrides (``reverse_translate_message``
            keys on the underscored wire name, so a second pass is a no-op).

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

    return _NameTranslatingChatOpenRouter


def create_openrouter_chat(
    model_name: str,
    api_key: str,
    *,
    timeout: int = 120,
    max_retries: int = 2,
    **extra_kwargs: object,
) -> ChatOpenRouter:
    """creates a configured ``ChatOpenRouter`` for OpenRouter-routed models.

    Returns the :class:`_NameTranslatingChatOpenRouter` subclass, not
    a vanilla ``ChatOpenRouter``. Application code interacts with it
    exactly the same way (it IS a ``ChatOpenRouter``); the only
    difference is the wire-side tool-name translation that hides
    Bedrock-style provider quirks from the rest of the codebase.

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
    :return: configured ``ChatOpenRouter`` (the name-translating subclass)
    :rtype: ChatOpenRouter
    """
    chat_cls = _build_translating_chat_class()

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

    model: ChatOpenRouter = chat_cls(**kwargs)
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
        # DeepSeek's direct API runs automatic context caching and surfaces
        # ``cached_tokens`` on the response without an opt-in marker; the
        # ``deepseek/`` slug routed through OpenRouter inherits the same
        # behavior. Same request shape as OpenAI auto-cache.
        supports_anthropic_cache_control=False,
        supports_openai_auto_cache=True,
        min_cacheable_tokens=0,
        cache_ttl_seconds=0,
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
        supports_anthropic_cache_control=False,
        supports_openai_auto_cache=True,
        min_cacheable_tokens=0,
        cache_ttl_seconds=0,
    ),
}


for _model_id, _caps in _OPENROUTER_CAPABILITIES.items():
    register_capabilities(_model_id, _caps)
