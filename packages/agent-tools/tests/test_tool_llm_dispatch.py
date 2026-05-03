"""Tests for ``invoke_tool_llm`` service tool."""

from __future__ import annotations

import pytest

from threetears.agent.tools import (
    InvokeToolLlmInput,
    ToolLlmInvocation,
    ToolLlmResolver,
    load_tool_llm_dispatch,
)


class _Resolver:
    """Minimal :class:`ToolLlmResolver` impl for tests."""

    def __init__(
        self,
        *,
        result: ToolLlmInvocation | None = None,
        raise_with: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[str, str]] = []
        self._result = result
        self._raise = raise_with

    async def resolve_and_invoke(
        self,
        tool_name: str,
        input_text: str,
    ) -> ToolLlmInvocation | None:
        self.calls.append((tool_name, input_text))
        if self._raise is not None:
            raise self._raise
        return self._result


def test_resolver_protocol_runtime_check() -> None:
    """``_Resolver`` satisfies :class:`ToolLlmResolver` at runtime."""
    resolver = _Resolver()
    assert isinstance(resolver, ToolLlmResolver)


def test_load_returns_single_tool() -> None:
    """The factory produces exactly one ``invoke_tool_llm`` tool."""
    resolver = _Resolver()
    tools = load_tool_llm_dispatch(resolver)
    assert len(tools) == 1
    assert tools[0].name == "invoke_tool_llm"
    assert tools[0].args_schema is InvokeToolLlmInput


@pytest.mark.asyncio
async def test_dispatch_returns_resolver_output() -> None:
    """A successful resolve passes the output through to the LLM."""
    resolver = _Resolver(result=ToolLlmInvocation(
        tool_name="translator",
        output="Bonjour le monde",
        duration_ms=42,
    ))
    tool = load_tool_llm_dispatch(resolver)[0]

    result = await tool.ainvoke({
        "tool_name": "translator",
        "input_text": "translate Hello world to French",
    })
    assert result == "Bonjour le monde"
    assert resolver.calls == [
        ("translator", "translate Hello world to French"),
    ]


@pytest.mark.asyncio
async def test_resolver_returning_none_surfaces_not_found() -> None:
    """``None`` becomes an LLM-facing not-found error."""
    resolver = _Resolver(result=None)
    tool = load_tool_llm_dispatch(resolver)[0]

    result = await tool.ainvoke({
        "tool_name": "nonexistent",
        "input_text": "do something",
    })
    assert "[TOOL ERROR]" in result
    assert "not found" in result.lower()
    assert "'nonexistent'" in result


@pytest.mark.asyncio
async def test_resolver_exception_caught() -> None:
    """A resolver raising becomes an LLM-facing tool error."""
    resolver = _Resolver(raise_with=RuntimeError("upstream blew up"))
    tool = load_tool_llm_dispatch(resolver)[0]

    result = await tool.ainvoke({
        "tool_name": "translator",
        "input_text": "translate something",
    })
    assert "[TOOL ERROR]" in result
    assert "upstream blew up" in result


@pytest.mark.asyncio
async def test_recall_intent_short_circuits_before_resolver() -> None:
    """Inputs that look like recall requests redirect without dispatching."""
    resolver = _Resolver(result=ToolLlmInvocation(
        tool_name="translator",
        output="should not run",
    ))
    tool = load_tool_llm_dispatch(resolver)[0]

    # "show me" + ctx-id pattern is the canonical recall-intent shape
    # the heuristic catches.
    result = await tool.ainvoke({
        "tool_name": "translator",
        "input_text": "show me the previous result [ctx:abc123]",
    })
    assert "[REDIRECT]" in result
    assert "recall_context" in result
    # Resolver was not called at all
    assert resolver.calls == []


@pytest.mark.asyncio
async def test_custom_description_override() -> None:
    """``description=`` kwarg overrides the platform-standard wording."""
    resolver = _Resolver()
    custom = "Custom dispatch description for app-specific framing."
    tool = load_tool_llm_dispatch(resolver, description=custom)[0]
    assert tool.description == custom


@pytest.mark.asyncio
async def test_invocation_metadata_preserved_internally() -> None:
    """The resolver's invocation metadata is not surfaced to the LLM but is loggable."""
    resolver = _Resolver(result=ToolLlmInvocation(
        tool_name="extractor",
        output="extracted content",
        duration_ms=1234,
        input_tokens=100,
        output_tokens=50,
        metadata={"tool_llm_id": "abc-123"},
    ))
    tool = load_tool_llm_dispatch(resolver)[0]

    # The LLM gets only the output text.
    result = await tool.ainvoke({
        "tool_name": "extractor",
        "input_text": "extract dates from text",
    })
    assert result == "extracted content"
