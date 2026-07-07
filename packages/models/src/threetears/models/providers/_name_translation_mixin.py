"""Shared tool-name-translation hooks for provider chat wrappers.

Every provider chat subclass (`_NameTranslatingChatOpenAI` /
`_NameTranslatingChatOpenRouter` / `_NameTranslatingChatAnthropic`) needs the
identical dot<->underscore tool-name translation at the wire boundary. Rather
than three copies of the same seven method overrides (and three copies of the
junk-name filter), each provider now builds
``class _NameTranslatingChatX(NameTranslatingChatMixin, ChatX)`` and declares
only the ``_name_reverse_map`` ``PrivateAttr``; all the behaviour lives here,
once.

Translation happens in two directions at the wire boundary:

- **Outbound (forward)** — dotted tool-call names in the ``messages`` being
  SENT (a prior round's ``AIMessage`` re-sent next round, or a model
  hallucination / dotted MCP tool) are mangled to the underscored wire form via
  :func:`threetears.models.tool_name_translation.forward_translate_input` before
  ``super()``, so they satisfy every provider's ``^[a-zA-Z0-9_-]`` tool-name
  validator. Non-mutating (copy-on-rename) so application history keeps the
  canonical dotted name for dispatch / logging / persistence.
- **Inbound (reverse)** — tool-call names in the RESPONSE are rewritten back
  from wire to canonical dotted form via
  :func:`threetears.models.tool_name_translation.reverse_translate_message`, and
  junk-named ``invalid_tool_calls`` entries (e.g. the XML-attribute-leak shape,
  prod 2026-05-19) are dropped, before any of it reaches application dispatch.

**Why the PUBLIC ``astream`` / ``ainvoke`` / ``invoke`` (not the protected
``_astream`` / ``_generate``) are overridden.** Wrapping the protected
``_astream`` in another async generator silently drops ``on_chat_model_stream``
callbacks (prod 2026-05-13: 190 chunks delivered, 0 stream events — the live UI
stayed blank while the DB content saved fine). And ``BaseChatModel.ainvoke`` /
``invoke`` route through ``_agenerate_with_cache`` -> the protected ``_astream``
aggregate whenever a streaming callback is attached (the converged ``agent_node``
path under an ``astream_events`` tap), bypassing BOTH ``astream`` AND
``_agenerate`` (prod 2026-06-22: a converged loop leaking ``threetears_web_search``
that the tool node could not resolve). So the public methods are overridden and
their single/streamed result post-processed no matter which internal route
produced it. ``agenerate`` / ``generate`` (the batch chokepoints) are covered for
the same bypass. All translation is idempotent — ``reverse_translate_message``
keys on the underscored wire name and ``mangle`` only touches dotted names, so the
overlapping internal routings between these entry points cannot double-translate.

**The ``merge_configs`` pre-merge (prod 2026-05-13).** When called via
``RunnableBinding`` (``model.with_config(callbacks=[...])``), ``config`` carries
the bound tracking callbacks as a plain list; forwarding it verbatim makes
``BaseChatModel``'s ``ensure_config`` REPLACE the contextvar's callback manager
(which holds the ``astream_events`` event_streamer) with that list, dropping the
event stream. ``merge_configs(ensure_config(None), config)`` folds the list into
the manager instead, preserving both.

Mix in BEFORE the concrete base (``(NameTranslatingChatMixin, ChatX)``) so
``super()`` in each hook resolves to the provider class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator

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

__all__ = ["NameTranslatingChatMixin", "drop_junk_invalid_tool_calls"]

_logger = get_logger(__name__)


def drop_junk_invalid_tool_calls(message: Any) -> None:
    """drop ``invalid_tool_calls`` entries whose ``name`` fails validation.

    Mutates ``message.invalid_tool_calls`` in place, keeping only entries whose
    names match the canonical 3tears tool-name regex; each rejected entry is
    logged once at WARNING (name truncated to 80 chars). Guards against a
    junk-named fragment (e.g. an XML-attribute leak, prod 2026-05-19) reaching
    downstream dispatch / persistence.

    :param message: chat-model response (``AIMessage`` or ``AIMessageChunk``);
        duck-typed via attribute access
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
            "name-translating wrapper dropped invalid_tool_calls entry with junk name: %s",
            truncated,
        )
    raw.clear()
    raw.extend(kept)


class NameTranslatingChatMixin:
    """Provider-agnostic dot<->underscore tool-name translation hooks.

    See the module docstring for the direction model and the incident history
    behind overriding the public entry points. The consuming subclass must
    declare ``_name_reverse_map: dict[str, str] = PrivateAttr(default_factory=dict)``
    (pydantic collects private attrs from the concrete model, not a plain-object
    mixin base). Every hook calls ``super()`` to reach the concrete provider
    class, so the mixin MUST precede that class in the bases.
    """

    _name_reverse_map: dict[str, str]

    def bind_tools(
        self,
        tools: list[BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """bind tools after dot->underscore name translation for the wire.

        Application-side tools keep their canonical dotted names; the bound
        runnable holds wire-side proxies whose ``.name`` is the underscored
        form. The reverse map is stored on this instance for response
        un-translation. Mutated (clear + update) rather than reassigned so a
        concurrently-running stream's closure keeps the same dict object.

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
        return super().bind_tools(wire_tools, **kwargs)  # type: ignore[misc]

    async def astream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        """stream AIMessageChunks with tool-call names translated both ways.

        Forward-translates the outbound history, then un-translates + filters
        each yielded chunk. Overrides the PUBLIC ``astream`` (not ``_astream``)
        and pre-merges the config — see the module docstring for the two
        production incidents (2026-05-13) that dictate both choices.

        :param input: chat input (messages or string)
        :ptype input: LanguageModelInput
        :param config: optional runnable config
        :ptype config: RunnableConfig | None
        :param stop: optional stop sequences
        :ptype stop: list[str] | None
        :param kwargs: passthrough to ``super().astream``
        :ptype kwargs: Any
        :return: async iterator of translated AIMessageChunks
        :rtype: AsyncIterator[AIMessageChunk]
        """
        from langchain_core.runnables.config import ensure_config, merge_configs

        merged_config = merge_configs(ensure_config(None), config)
        wire_input = forward_translate_input(input)
        async for chunk in super().astream(  # type: ignore[misc]
            wire_input,
            config=merged_config,
            stop=stop,
            **kwargs,
        ):
            reverse_translate_message(chunk, self._name_reverse_map)
            drop_junk_invalid_tool_calls(chunk)
            yield chunk

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """non-streaming generate with tool-call names translated both ways.

        :param messages: chat messages
        :ptype messages: list[BaseMessage]
        :param stop: optional stop sequences
        :ptype stop: list[str] | None
        :param run_manager: LangChain run manager
        :ptype run_manager: AsyncCallbackManagerForLLMRun | None
        :param kwargs: passthrough
        :ptype kwargs: Any
        :return: chat result with translated tool-call names
        :rtype: ChatResult
        """
        result = await super()._agenerate(  # type: ignore[misc]
            forward_translate_input(messages),
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )
        for generation in result.generations:
            reverse_translate_message(generation.message, self._name_reverse_map)
            drop_junk_invalid_tool_calls(generation.message)
        return result

    async def agenerate(
        self,
        messages: list[list[BaseMessage]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """un-translate tool names on the batch generate surface.

        ``agenerate`` is the chokepoint ``ainvoke`` / ``abatch`` route through,
        and it aggregates from the protected ``_astream`` when streaming
        callbacks are present (bypassing ``_agenerate``). Post-process every
        generated message; idempotent with the other overrides.

        :param messages: batch of message lists
        :ptype messages: list[list[BaseMessage]]
        :param args: positional passthrough to ``super().agenerate``
        :ptype args: Any
        :param kwargs: keyword passthrough to ``super().agenerate``
        :ptype kwargs: Any
        :return: LLMResult with canonical (dotted) tool-call names
        :rtype: Any
        """
        from langchain_core.outputs import ChatGeneration

        result = await super().agenerate(messages, *args, **kwargs)  # type: ignore[misc]
        for generations in result.generations:
            for generation in generations:
                # chat models always yield ChatGeneration(Chunk); the isinstance
                # narrow proves ``.message`` exists (the base Generation union
                # member has no such attribute).
                if isinstance(generation, ChatGeneration):
                    reverse_translate_message(generation.message, self._name_reverse_map)
                    drop_junk_invalid_tool_calls(generation.message)
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
        from langchain_core.outputs import ChatGeneration

        result = super().generate(messages, *args, **kwargs)  # type: ignore[misc]
        for generations in result.generations:
            for generation in generations:
                if isinstance(generation, ChatGeneration):
                    reverse_translate_message(generation.message, self._name_reverse_map)
                    drop_junk_invalid_tool_calls(generation.message)
        return result

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> BaseMessage:
        """invoke (non-streaming public API) with names translated both ways.

        Overriding ``astream`` + ``_agenerate`` is not sufficient: under a
        streaming tap ``ainvoke`` aggregates from the protected ``_astream``,
        bypassing both (module docstring, prod 2026-06-22). Forward-translate the
        input and post-process the returned message so BOTH directions are
        covered regardless of the internal route.

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

        merged_config = merge_configs(ensure_config(None), config)
        result = await super().ainvoke(  # type: ignore[misc]
            forward_translate_input(input),
            config=merged_config,
            stop=stop,
            **kwargs,
        )
        reverse_translate_message(result, self._name_reverse_map)
        drop_junk_invalid_tool_calls(result)
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

        merged_config = merge_configs(ensure_config(None), config)
        result = super().invoke(  # type: ignore[misc]
            forward_translate_input(input),
            config=merged_config,
            stop=stop,
            **kwargs,
        )
        reverse_translate_message(result, self._name_reverse_map)
        drop_junk_invalid_tool_calls(result)
        return result
