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
incoming ``tool_calls``, so application code (3tears core, metallm,
14-eng-ai-bot, 14-eng-ai-bot-agents) never sees the wire form. The
translation is keyed by the dotted -> underscored mapping built at
``bind_tools`` time, so it round-trips losslessly even for tools whose
names contain underscores already (e.g. ``threetears.web_search`` ->
``threetears_web_search`` and back).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator

from langchain_core.tools import BaseTool
from pydantic import Field, PrivateAttr

from threetears.models.capabilities import ModelCapabilities, register_capabilities
from threetears.models.enums import ModelStatus, ModelTier, ModelType

if TYPE_CHECKING:
    from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
    from langchain_core.language_models.chat_models import LanguageModelInput
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatGenerationChunk, ChatResult
    from langchain_core.runnables import Runnable, RunnableConfig
    from langchain_openrouter import ChatOpenRouter

__all__ = [
    "OPENROUTER_PROVIDER_NAME",
    "create_openrouter_chat",
]


OPENROUTER_PROVIDER_NAME = "openrouter"


class _NameMangledToolProxy(BaseTool):
    """LangChain BaseTool whose ``.name`` is the underscored wire form,
    delegating execution to a dotted-named original tool.

    Used by :class:`_NameTranslatingChatOpenRouter.bind_tools` to swap
    each application-side tool for a wire-side proxy. The proxy keeps
    the description and ``args_schema`` of the original (those flow to
    the LLM unchanged) but exposes the dot-replaced name so the bound
    tool spec passes Bedrock-style validators.

    Application code never receives this proxy -- it lives only inside
    the bound runnable's tool list, and the response un-translation
    layer rewrites tool-call names back to the dotted form before any
    consumer code sees them.
    """

    name: str
    description: str
    args_schema: type[Any] | None = None
    _delegate: BaseTool = PrivateAttr()

    def __init__(self, *, delegate: BaseTool, mangled_name: str) -> None:
        # Pydantic v2 BaseModel constructor does not take positional
        # args; route the proxy fields through ``model_construct`` and
        # set ``_delegate`` after to keep this lean.
        super().__init__(
            name=mangled_name,
            description=delegate.description,
            args_schema=delegate.args_schema,
        )
        self._delegate = delegate

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        return await self._delegate._arun(*args, **kwargs)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate._run(*args, **kwargs)


def _mangle_tool_name(name: str) -> str:
    """Translate a dotted canonical tool name to its underscored wire form.

    Applied before sending tool specs to OpenRouter so names pass
    strict provider validators that reject dots.

    :param name: canonical tool name (may contain dots)
    :ptype name: str
    :return: dot-replaced form suitable for ``^[a-zA-Z0-9_-]{1,128}$``
        validators (Bedrock + Anthropic-direct + most OpenAI-compat
        backends accept this)
    :rtype: str
    """
    return name.replace(".", "_")


def _build_name_translation(
    tools: list[BaseTool],
) -> tuple[list[BaseTool], dict[str, str]]:
    """build the wire-side proxy list + reverse map for response un-translation.

    Tools whose canonical name has no dot pass through unchanged (no
    proxy needed). Tools with dots get a :class:`_NameMangledToolProxy`
    wrapper carrying the underscored wire name. The reverse map keys
    on the underscored form so :class:`_NameTranslatingChatOpenRouter`
    can rewrite ``tool_call`` names in streaming responses.

    :param tools: application-side bind_tools input (canonical dotted
        names)
    :ptype tools: list[BaseTool]
    :return: tuple of (wire-side tool list, underscored -> dotted map)
    :rtype: tuple[list[BaseTool], dict[str, str]]
    """
    wire_tools: list[BaseTool] = []
    reverse_map: dict[str, str] = {}
    for tool in tools:
        canonical = tool.name
        if "." not in canonical:
            wire_tools.append(tool)
            continue
        mangled = _mangle_tool_name(canonical)
        wire_tools.append(_NameMangledToolProxy(
            delegate=tool, mangled_name=mangled,
        ))
        reverse_map[mangled] = canonical
    return wire_tools, reverse_map


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
            wire_tools, reverse_map = _build_name_translation(tools)
            # Mutate the shared reverse_map rather than reassign so the
            # ``_astream`` closure in concurrently-running streams still
            # sees the same dict object (PrivateAttr is per-instance,
            # and this instance is request-scoped in every consumer
            # observed -- metallm's ``build_chat_model_from_config``
            # constructs a fresh model per resolve_chat_model call).
            self._name_reverse_map.clear()
            self._name_reverse_map.update(reverse_map)
            return super().bind_tools(wire_tools, **kwargs)

        async def _astream(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> AsyncIterator[ChatGenerationChunk]:
            """stream chunks with tool-call names un-translated to canonical form.

            ``ChatOpenRouter._astream`` yields
            :class:`ChatGenerationChunk` whose ``.message`` is an
            ``AIMessageChunk`` carrying ``tool_call_chunks`` (partial,
            during streaming) and ``tool_calls`` (post-merge). We
            rewrite each tool-call name on ``chunk.message`` back to
            the canonical dotted form using the reverse map populated
            at ``bind_tools`` time, so application code reading
            ``chunk.message.tool_calls[i]["name"]`` (or the merged
            AIMessage that consumers accumulate from chunks) never
            sees the wire form.

            :param messages: chat messages
            :ptype messages: list[BaseMessage]
            :param stop: optional stop sequences
            :ptype stop: list[str] | None
            :param run_manager: LangChain run manager
            :ptype run_manager: AsyncCallbackManagerForLLMRun | None
            :param kwargs: passthrough
            :ptype kwargs: Any
            :return: async iterator of un-translated chunks
            :rtype: AsyncIterator[ChatGenerationChunk]
            """
            async for chunk in super()._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs,
            ):
                # ``ChatGenerationChunk.message`` is the AIMessageChunk
                # carrying tool-call fields. Some upstream paths wrap
                # the message in a different shape (LangChain has
                # historically vacillated between yielding raw
                # AIMessageChunks vs ChatGenerationChunks); fall back
                # to the chunk itself when ``.message`` is absent so
                # we cover both shapes without a special case.
                target = getattr(chunk, "message", chunk)
                self._reverse_translate_message(target)
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
                messages, stop=stop, run_manager=run_manager, **kwargs,
            )
            for generation in result.generations:
                self._reverse_translate_message(generation.message)
            return result

        def _reverse_translate_message(self, message: Any) -> None:
            """rewrite tool-call names on ``message`` from wire to canonical form.

            Mutates in place. Called for every chunk yielded from
            ``_astream`` and every message in ``_agenerate``'s result.
            Touches three name-bearing fields:

            * ``tool_call_chunks`` -- partial streamed tool calls; the
              ``name`` field arrives once at the start of each call,
              subsequent chunks carry only ``args``.
            * ``tool_calls`` -- the merged, well-formed tool calls
              consumers iterate.
            * ``invalid_tool_calls`` -- the recovery target for
              malformed streaming; metallm and aibots-agents both
              attempt to re-parse them.

            No-op when the message has no tool-call fields or when no
            name matches the reverse map (empty map = no tools were
            bound, so nothing to translate).
            """
            if not self._name_reverse_map:
                return
            for tc_chunk in getattr(message, "tool_call_chunks", None) or []:
                name = tc_chunk.get("name")
                if name and name in self._name_reverse_map:
                    tc_chunk["name"] = self._name_reverse_map[name]
            for tc in getattr(message, "tool_calls", None) or []:
                name = tc.get("name")
                if name and name in self._name_reverse_map:
                    tc["name"] = self._name_reverse_map[name]
            for tc in getattr(message, "invalid_tool_calls", None) or []:
                name = tc.get("name")
                if name and name in self._name_reverse_map:
                    tc["name"] = self._name_reverse_map[name]

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
