"""tests for ``create_openrouter_chat`` factory and capability registration."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.tools import BaseTool

from threetears.models.capabilities import get_capabilities
from threetears.models.enums import ModelTier, ModelType
from threetears.models.providers.openrouter import (
    OPENROUTER_PROVIDER_NAME,
    create_openrouter_chat,
)
from threetears.models.tool_name_translation import (
    NameMangledToolProxy,
    build_name_translation,
    mangle_tool_name,
    reverse_translate_message,
)


class TestCreateOpenRouterChat:
    """tests for ``create_openrouter_chat`` factory."""

    def test_returns_base_chat_model(self) -> None:
        """factory returns a ``BaseChatModel`` subclass instance."""
        model = create_openrouter_chat("deepseek/deepseek-chat-v3-0324", "sk-test")
        assert isinstance(model, BaseChatModel)


class TestOpenRouterCapabilityRegistration:
    """tests that openrouter canonical models register at import time."""

    def test_deepseek_chat_registered(self) -> None:
        """``deepseek/deepseek-chat-v3-0324`` resolves to openrouter chat capabilities."""
        caps = get_capabilities("deepseek/deepseek-chat-v3-0324")
        assert caps is not None
        assert caps.provider_name == OPENROUTER_PROVIDER_NAME
        assert caps.model_type == ModelType.CHAT
        assert caps.model_tier == ModelTier.LARGE
        assert caps.requires_alternating_roles is True


# -- name translation --------------------------------------------------------


class _DottedTool(BaseTool):
    """Minimal :class:`BaseTool` whose ``.name`` carries a dot, for tests."""

    name: str = "threetears.calculator"
    description: str = "test calculator"
    invoked_with: list[dict[str, Any]] = []

    def _run(self, **kwargs: Any) -> str:
        self.invoked_with.append(dict(kwargs))
        return "ok"

    async def _arun(self, **kwargs: Any) -> str:
        self.invoked_with.append(dict(kwargs))
        return "ok"


class TestMangleToolName:
    """``mangle_tool_name`` produces wire-safe names."""

    def test_dot_replaced_with_underscore(self) -> None:
        assert mangle_tool_name("threetears.calculator") == "threetears_calculator"

    def test_nested_dots_all_replaced(self) -> None:
        assert (
            mangle_tool_name("threetears.workspace.fs_read")
            == "threetears_workspace_fs_read"
        )

    def test_no_dots_passes_through(self) -> None:
        assert mangle_tool_name("plain_name") == "plain_name"

    def test_existing_underscores_preserved(self) -> None:
        """Underscores in the source are preserved -- the round-trip
        relies on the reverse map (built per-bind) rather than on a
        symmetrical underscore<->dot inverse so ``threetears.web_search``
        and any future tool ``threetears_web_search`` (no dot) coexist
        unambiguously when one's wire form happens to collide with the
        other's canonical name.
        """
        assert mangle_tool_name("threetears.web_search") == "threetears_web_search"


class TestBuildNameTranslation:
    """``build_name_translation`` returns proxies + reverse map."""

    def test_dotted_tool_gets_proxy(self) -> None:
        tool = _DottedTool()
        wire_tools, reverse_map = build_name_translation([tool])
        assert len(wire_tools) == 1
        assert wire_tools[0] is not tool
        assert isinstance(wire_tools[0], NameMangledToolProxy)
        assert wire_tools[0].name == "threetears_calculator"
        assert reverse_map == {"threetears_calculator": "threetears.calculator"}

    def test_dotless_tool_passes_through(self) -> None:
        """Tools with no dot in their name pass through unchanged --
        the proxy is unnecessary and the reverse map stays empty for
        them, so the response un-translation no-ops on their tool calls.
        """

        class _Plain(BaseTool):
            name: str = "plain_tool"
            description: str = "no dots here"

            def _run(self, **kwargs: Any) -> str:
                return "ok"

            async def _arun(self, **kwargs: Any) -> str:
                return "ok"

        tool = _Plain()
        wire_tools, reverse_map = build_name_translation([tool])
        assert wire_tools == [tool]
        assert reverse_map == {}

    def test_proxy_preserves_description_and_args_schema(self) -> None:
        tool = _DottedTool()
        wire_tools, _ = build_name_translation([tool])
        proxy = wire_tools[0]
        assert proxy.description == tool.description
        assert proxy.args_schema is tool.args_schema

    @pytest.mark.asyncio
    async def test_proxy_arun_delegates_to_original(self) -> None:
        """Calling the proxy's ``_arun`` runs the dotted-named original."""
        tool = _DottedTool()
        wire_tools, _ = build_name_translation([tool])
        proxy = wire_tools[0]
        result = await proxy._arun(expression="1+1")
        assert result == "ok"
        assert tool.invoked_with == [{"expression": "1+1"}]


class TestNameTranslatingChatOpenRouter:
    """End-to-end name-translation via the ``ChatOpenRouter`` subclass."""

    def test_factory_returns_translating_subclass(self) -> None:
        """The factory builds the translating subclass, not vanilla
        ``ChatOpenRouter``. The class name carries ``Translating`` so a
        debugger / log line shows the active behaviour.
        """
        model = create_openrouter_chat("deepseek/deepseek-chat-v3-0324", "sk-test")
        assert "Translating" in type(model).__name__

    def test_bind_tools_populates_reverse_map(self) -> None:
        """``bind_tools`` mutates the instance's reverse map so a
        subsequent ``_astream`` / ``_agenerate`` can rewrite tool-call
        names.
        """
        model = create_openrouter_chat("deepseek/deepseek-chat-v3-0324", "sk-test")
        tool = _DottedTool()
        model.bind_tools([tool])
        # PrivateAttr: access via the standard pydantic shape
        reverse = model._name_reverse_map
        assert reverse == {"threetears_calculator": "threetears.calculator"}

    def test_reverse_translate_message_rewrites_tool_calls(self) -> None:
        """A finished ``AIMessage`` with underscored tool-call names
        gets its names rewritten back to the canonical dotted form.
        """
        model = create_openrouter_chat("deepseek/deepseek-chat-v3-0324", "sk-test")
        model.bind_tools([_DottedTool()])
        msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "threetears_calculator",
                    "args": {"expression": "2+2"},
                    "id": "call_1",
                },
            ],
        )
        reverse_translate_message(msg, model._name_reverse_map)
        assert msg.tool_calls[0]["name"] == "threetears.calculator"

    def test_reverse_translate_message_rewrites_tool_call_chunks(self) -> None:
        """Streaming chunks carry partial ``tool_call_chunks``; the name
        field arrives once at the start of each call. The reverse
        translation rewrites that first chunk so consumers accumulating
        tool calls see canonical names from the start.
        """
        model = create_openrouter_chat("deepseek/deepseek-chat-v3-0324", "sk-test")
        model.bind_tools([_DottedTool()])
        chunk = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "name": "threetears_calculator",
                    "args": "",
                    "id": "call_1",
                    "index": 0,
                },
            ],
        )
        reverse_translate_message(chunk, model._name_reverse_map)
        assert chunk.tool_call_chunks[0]["name"] == "threetears.calculator"

    def test_reverse_translate_message_rewrites_invalid_tool_calls(self) -> None:
        """Malformed streamed tool calls land in ``invalid_tool_calls``
        and metallm / aibots-agents both inspect them when ``tool_calls``
        is empty. Translate those names too so the recovery code sees
        canonical form.
        """
        model = create_openrouter_chat("deepseek/deepseek-chat-v3-0324", "sk-test")
        model.bind_tools([_DottedTool()])
        msg = AIMessage(
            content="",
            invalid_tool_calls=[
                {
                    "name": "threetears_calculator",
                    "args": "{partial",
                    "id": "call_1",
                    "error": "JSONDecodeError",
                },
            ],
        )
        reverse_translate_message(msg, model._name_reverse_map)
        assert msg.invalid_tool_calls[0]["name"] == "threetears.calculator"

    def test_reverse_translate_message_noop_when_no_tools_bound(self) -> None:
        """Without any prior ``bind_tools`` call the reverse map is
        empty; ``_reverse_translate_message`` short-circuits without
        mutating the message.
        """
        model = create_openrouter_chat("deepseek/deepseek-chat-v3-0324", "sk-test")
        msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "external_tool",
                    "args": {},
                    "id": "call_1",
                },
            ],
        )
        reverse_translate_message(msg, model._name_reverse_map)
        # No bind_tools means no translation map; the name stays as-is.
        assert msg.tool_calls[0]["name"] == "external_tool"

    def test_reverse_translate_message_noop_for_unmatched_name(self) -> None:
        """A tool-call name not in the reverse map (e.g. from a tool
        that was already underscored, or an LLM hallucination) passes
        through unchanged.
        """
        model = create_openrouter_chat("deepseek/deepseek-chat-v3-0324", "sk-test")
        model.bind_tools([_DottedTool()])
        msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "some_other_tool",
                    "args": {},
                    "id": "call_1",
                },
            ],
        )
        reverse_translate_message(msg, model._name_reverse_map)
        assert msg.tool_calls[0]["name"] == "some_other_tool"

    def test_rebind_replaces_reverse_map(self) -> None:
        """A second ``bind_tools`` call with a different tool set
        replaces the reverse map wholesale. Otherwise stale entries
        from a prior bind would translate names that don't belong to
        the current bind.
        """

        class _OtherTool(_DottedTool):
            name: str = "threetears.web_search"

        model = create_openrouter_chat("deepseek/deepseek-chat-v3-0324", "sk-test")
        model.bind_tools([_DottedTool()])
        model.bind_tools([_OtherTool()])
        assert model._name_reverse_map == {
            "threetears_web_search": "threetears.web_search",
        }

    @pytest.mark.asyncio
    async def test_astream_translates_chat_generation_chunk_message(self) -> None:
        """``_astream`` must walk ``chunk.message`` to translate names,
        not ``chunk`` itself.

        ``ChatOpenRouter._astream`` yields :class:`ChatGenerationChunk`
        instances whose ``.message`` (an :class:`AIMessageChunk`) carries
        the actual ``tool_calls`` / ``tool_call_chunks``. An earlier
        version of the wrapper invoked the translation on ``chunk``
        directly, which produced no-op rewrites (the ChatGenerationChunk
        has no ``tool_calls`` attribute) and let the underscored wire
        names leak to consumers -- the metallm date-tool e2e caught
        this in production-shape testing on 2026-05-09. This test pins
        the chunk-shape contract so the bug class cannot regress.
        """
        from langchain_core.outputs import ChatGenerationChunk
        from langchain_core.messages import AIMessageChunk

        async def _fake_super_astream(
            self: Any,
            messages: Any,
            stop: Any = None,
            run_manager: Any = None,
            **kwargs: Any,
        ):
            """Stand in for ``ChatOpenRouter._astream`` -- yields a single
            ChatGenerationChunk whose nested AIMessageChunk carries the
            wire-form tool_call name the LLM emitted.

            Takes ``self`` because we patch it onto the class -- the
            method-binding semantics need the receiver slot or our
            override's ``super()._astream(messages, ...)`` call ends
            up passing ``messages`` as the bound ``self`` and
            ``stop``-as-kwarg trips a "multiple values" TypeError.
            """
            del self, messages, stop, run_manager, kwargs
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": "threetears_calculator",
                            "args": "",
                            "id": "call_1",
                            "index": 0,
                        },
                    ],
                ),
            )

        model = create_openrouter_chat(
            "deepseek/deepseek-chat-v3-0324", "sk-test",
        )
        model.bind_tools([_DottedTool()])

        # Patch the parent ``_astream`` so the test does not need a
        # real OpenRouter HTTP call.
        from langchain_openrouter import ChatOpenRouter

        original_astream = ChatOpenRouter._astream
        try:
            ChatOpenRouter._astream = _fake_super_astream  # type: ignore[method-assign]
            chunks: list[ChatGenerationChunk] = []
            async for chunk in model._astream([], None, None):
                chunks.append(chunk)
        finally:
            ChatOpenRouter._astream = original_astream  # type: ignore[method-assign]

        assert len(chunks) == 1
        nested = chunks[0].message
        assert nested.tool_call_chunks[0]["name"] == "threetears.calculator", (
            "_astream did not translate ``chunk.message.tool_call_chunks[i][\"name\"]``;"
            " regression to operating on the ChatGenerationChunk wrapper"
            " instead of its nested AIMessageChunk."
        )
