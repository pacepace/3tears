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
  :func:`threetears.models.tool_name_translation.reverse_translate_message`
  before any of it reaches application dispatch.

**Junk-named tool calls** (the XML-attribute-leak shape, prod 2026-05-19) are
dropped where every consumer is downstream of the drop
(:mod:`threetears.models.providers._junk_tool_calls`). On a stream that is the
protected ``_astream`` / ``_stream``: ``BaseChatModel`` reports each chunk to the
callbacks (``astream_events``, a streaming ``ainvoke``'s handlers, LangGraph's
messages stream) and builds the ``on_chat_model_end`` aggregate only AFTER this
hook yields it, and every streaming route -- ``astream``, ``ainvoke`` under a
streaming callback, the v2 protocol events, a deadline-collected
``_agenerate`` -- reads it. The run manager is kept from the provider's own
stream, which would otherwise report each raw chunk before the filter saw it,
and each released chunk is reported here instead. A finished message is
filtered again at every public entry point, which covers the non-streamed
answers.

**Why the PUBLIC ``astream`` / ``ainvoke`` / ``invoke`` carry the name
translation.** Prod 2026-05-13 lost every ``on_chat_model_stream`` event (190
chunks delivered, 0 stream events -- the live UI stayed blank while the DB content
saved fine) with translation in an ``_astream`` override. The ``_astream`` here
keeps the event chain whole because ``BaseChatModel`` reports what it yields;
``test_astream_events_emits_on_chat_model_stream`` and
``test_astream_events_survives_with_config_callbacks`` pin that chain for each
provider, and a change to this hook that breaks it fails them. And ``BaseChatModel.ainvoke`` /
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

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import TYPE_CHECKING, Any

from threetears.models.errors import ModelCallTimeout
from threetears.models.providers._junk_tool_calls import JunkToolCallStreamFilter, drop_junk_tool_calls
from threetears.models.tool_name_translation import (
    build_name_translation,
    forward_translate_input,
    reverse_translate_message,
)

if TYPE_CHECKING:
    from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
    from langchain_core.outputs import ChatGenerationChunk, ChatResult
    from langchain_core.runnables import Runnable, RunnableConfig

__all__ = ["NameTranslatingChatMixin"]


class NameTranslatingChatMixin:
    """Provider-agnostic dot<->underscore tool-name translation hooks.

    See the module docstring for the direction model and the incident history
    behind overriding the public entry points. The consuming subclass must
    declare ``_name_reverse_map: dict[str, str] = PrivateAttr(default_factory=dict)``
    (pydantic collects private attrs from the concrete model, not a plain-object
    mixin base). Every hook calls ``super()`` to reach the concrete provider
    class, so the mixin MUST precede that class in the bases.
    """

    if TYPE_CHECKING:
        # Type-checker-only declaration so this mixin's methods can reference
        # ``self._name_reverse_map``. At runtime each concrete provider subclass
        # owns the real ``PrivateAttr`` (pydantic collects private attrs from a
        # concrete BaseModel, not a plain-object mixin base). Guarding it keeps
        # the annotation out of the runtime class body, so the REQUIRED subclass
        # declarations are not flagged as shadowing a base private.
        _name_reverse_map: dict[str, str]

    def bind_tools(
        self,
        tools: Sequence[Any],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """bind tools after dot->underscore name translation for the wire.

        Application-side tools keep their canonical dotted names; the bound
        runnable holds wire-side proxies whose ``.name`` is the underscored
        form. The reverse map is stored on this instance for response
        un-translation. Mutated (clear + update) rather than reassigned so a
        concurrently-running stream's closure keeps the same dict object.

        :param tools: application-side bind_tools input -- BaseTool objects or
            provider-native tool-spec dicts, both of which the translator handles.
            Deliberately ``Sequence[Any]``: this mixin sits over ChatOpenAI,
            ChatAnthropic and ChatOpenRouter, whose own ``bind_tools`` signatures
            disagree with each other (``dict`` vs ``Mapping`` elements, and three
            different ``tool_choice`` types), so no narrower annotation can be a
            valid override of all three. The mixin never inspects a tool itself --
            it translates names and forwards -- so ``Any`` is what it actually means
        :ptype tools: Sequence[Any]
        :param kwargs: passthrough to ``super().bind_tools``
        :ptype kwargs: Any
        :return: runnable bound to wire-side proxy tools
        :rtype: Runnable[LanguageModelInput, AIMessage]
        """
        wire_tools, reverse_map = build_name_translation(list(tools))
        self._name_reverse_map.clear()
        self._name_reverse_map.update(reverse_map)
        if "tool_choice" in kwargs:
            kwargs["tool_choice"] = _wire_tool_choice(kwargs["tool_choice"], reverse_map)
        bound: Runnable[LanguageModelInput, AIMessage] = super().bind_tools(wire_tools, **kwargs)  # type: ignore[misc]
        return bound

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
            yield chunk

    def call_deadline_s(self) -> float | None:
        """How long one call may run, or ``None`` for no limit beyond the SDK's own.

        ``None`` here: a provider whose SDK already holds its timeout to the whole
        call needs nothing more. A wrapper whose SDK does not says so by
        overriding this (OpenRouter).

        :return: seconds, or ``None``
        :rtype: float | None
        """
        return None

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """The parent's stream without junk-named tool calls, ended when nothing arrives within the deadline.

        A long reply may stream for longer than :meth:`call_deadline_s`; what cannot
        happen is a wait that long with nothing arriving. Each chunk goes through
        :class:`~threetears.models.providers._junk_tool_calls.JunkToolCallStreamFilter`
        (the module docstring says why here). The parent gets no run manager, so it
        cannot report a chunk before the filter has seen it; each chunk this yields is
        reported to the caller's run manager instead, as the parent would have.
        Translation stays on the public ``astream``.

        :param messages: chat messages
        :ptype messages: list[BaseMessage]
        :param stop: optional stop sequences
        :ptype stop: list[str] | None
        :param run_manager: the caller's run manager, when it wants each chunk reported
        :ptype run_manager: AsyncCallbackManagerForLLMRun | None
        :param kwargs: keyword passthrough to the parent's ``_astream``
        :ptype kwargs: Any
        :return: the parent's chunks, junk-named tool calls removed
        :rtype: AsyncIterator[ChatGenerationChunk]
        :raises ModelCallTimeout: when no chunk arrives within the deadline
        """
        stream = super()._astream(messages, stop=stop, **kwargs)  # type: ignore[misc]
        junk = JunkToolCallStreamFilter()
        try:
            ended = False
            while not ended:
                released: list[ChatGenerationChunk]
                try:
                    async with asyncio.timeout(self.call_deadline_s()):
                        chunk = await anext(stream)
                    released = junk.feed(chunk)
                except StopAsyncIteration:
                    # NOSILENT: the parent's stream ended; what the filter still holds goes out last.
                    ended = True
                    released = junk.finish()
                except TimeoutError as exc:
                    raise ModelCallTimeout(f"no chunk within {self.call_deadline_s()} s") from exc
                for out in released:
                    if run_manager is not None:
                        await run_manager.on_llm_new_token(out.text, chunk=out, **_token_extras(out))
                    yield out
        finally:
            await stream.aclose()

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """The parent's sync stream without junk-named tool calls, as :meth:`_astream` does it.

        The sync ``stream`` and a sync ``invoke`` under a streaming callback read this
        hook. It has no deadline: the sync path never had one.

        :param messages: chat messages
        :ptype messages: list[BaseMessage]
        :param stop: optional stop sequences
        :ptype stop: list[str] | None
        :param run_manager: the caller's run manager, when it wants each chunk reported
        :ptype run_manager: CallbackManagerForLLMRun | None
        :param kwargs: keyword passthrough to the parent's ``_stream``
        :ptype kwargs: Any
        :return: the parent's chunks, junk-named tool calls removed
        :rtype: Iterator[ChatGenerationChunk]
        """
        junk = JunkToolCallStreamFilter()
        stream: Iterator[ChatGenerationChunk] = super()._stream(messages, stop=stop, **kwargs)  # type: ignore[misc]

        def released() -> Iterator[ChatGenerationChunk]:
            """the filter's output for each source chunk, then what it still holds.

            :return: the chunks to emit, in order
            :rtype: Iterator[ChatGenerationChunk]
            """
            for chunk in stream:
                yield from junk.feed(chunk)
            yield from junk.finish()

        for out in released():
            if run_manager is not None:
                run_manager.on_llm_new_token(out.text, chunk=out, **_token_extras(out))
            yield out

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """non-streaming generate with tool-call names translated both ways.

        With a :meth:`call_deadline_s`, the answer is collected from :meth:`_astream` and merged by
        LangChain's own ``agenerate_from_stream``, so the deadline limits silence, as it does for a
        streamed call, rather than the whole call. As a whole-call limit it cut off every plain
        ``ainvoke`` at 120 s while the model was still writing: a reasoning model's memory
        extraction runs 14 to 64 s and a resolution longer, and it fell back to storing every
        candidate as new. A call that sends nothing for the deadline still ends at it (the 218 s
        stall :meth:`call_deadline_s` was added for). Without a deadline the provider's own
        ``_agenerate`` runs, as before.

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
        :raises ModelCallTimeout: when no chunk arrives within the deadline
        """
        wire = forward_translate_input(messages)
        if self.call_deadline_s() is None:
            result = await super()._agenerate(  # type: ignore[misc]
                wire,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
        else:
            from langchain_core.language_models.chat_models import agenerate_from_stream

            result = await agenerate_from_stream(
                self._astream(wire, stop=stop, run_manager=run_manager, **kwargs),
            )
        for generation in result.generations:
            reverse_translate_message(generation.message, self._name_reverse_map)
            drop_junk_tool_calls(generation.message)
        translated: ChatResult = result
        return translated

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
                    drop_junk_tool_calls(generation.message)
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
                    drop_junk_tool_calls(generation.message)
        return result

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
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
        :rtype: AIMessage
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
        drop_junk_tool_calls(result)
        translated: AIMessage = result
        return translated

    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
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
        :rtype: AIMessage
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
        drop_junk_tool_calls(result)
        translated: AIMessage = result
        return translated


def _token_extras(chunk: ChatGenerationChunk) -> dict[str, Any]:
    """The extra arguments a provider passes with a chunk it reports: its logprobs, when it has any.

    :param chunk: a chunk about to be reported
    :ptype chunk: ChatGenerationChunk
    :return: ``{"logprobs": ...}`` or nothing
    :rtype: dict[str, Any]
    """
    logprobs = (chunk.generation_info or {}).get("logprobs")
    return {"logprobs": logprobs} if logprobs else {}


def _wire_tool_choice(tool_choice: Any, reverse_map: dict[str, str]) -> Any:
    """A ``tool_choice`` that names one tool, naming it by its wire name.

    The bound tools carry wire names, so a choice naming the canonical dotted
    name would name a tool the provider was never given. Keywords (``auto``,
    ``any``, ``required``, ``none``) and every other shape pass through.

    :param tool_choice: the caller's choice: a keyword, a tool name, or a
        provider dict (``{"type": "tool", "name": ...}`` or
        ``{"type": "function", "function": {"name": ...}}``)
    :ptype tool_choice: Any
    :param reverse_map: wire name -> canonical name, from this bind
    :ptype reverse_map: dict[str, str]
    :return: the choice, with a canonical tool name swapped for its wire name
    :rtype: Any
    """
    forward = {canonical: wire for wire, canonical in reverse_map.items()}
    result = tool_choice
    if isinstance(tool_choice, str) and tool_choice in forward:
        result = forward[tool_choice]
    elif isinstance(tool_choice, dict):
        if tool_choice.get("name") in forward:
            result = {**tool_choice, "name": forward[tool_choice["name"]]}
        function = tool_choice.get("function")
        if isinstance(function, dict) and function.get("name") in forward:
            result = {**tool_choice, "function": {**function, "name": forward[function["name"]]}}
    return result
