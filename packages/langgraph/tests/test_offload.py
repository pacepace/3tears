"""tests for the large tool-result offload contract + the ``tool_node`` seam.

covers the framework half of Path-1 / 2a:

- :class:`OffloadResult` value shape + the
  :const:`DEFAULT_OFFLOAD_THRESHOLD_CHARS` default.
- the ``tool_node`` seam: offload fires above the threshold (model sees
  ``summary + [ctx:<id>]``), does NOT fire below it, is byte-for-byte
  unchanged with no offloader injected, soft-fails to the full content
  when the offloader raises, skips failed tool results, skips when no
  ``conversation_id`` is known, and never swallows ``GraphBubbleUp``.
"""

from __future__ import annotations

import dataclasses
from typing import Any
from uuid import UUID, uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from threetears.langgraph.nodes import tool_node
from threetears.langgraph.offload import (
    DEFAULT_OFFLOAD_THRESHOLD_CHARS,
    NEVER_OFFLOAD_TOOLS,
    OffloadResult,
    ToolResultOffloader,
    format_offload_handle,
    has_offload_handle,
    is_never_offload_tool,
)


# parity-with: threetears.langgraph.offload.ToolResultOffloader
class _FakeOffloader:
    """records offload calls and returns a canned :class:`OffloadResult`.

    stands in for a concrete :class:`ToolResultOffloader` (e.g. the SDK's
    ``ContextItemOffloader``). ``handle`` defaults to a fixed id so the
    seam's ``[ctx:<id>]`` shaping is assertable; ``raises`` / ``returns``
    let individual tests exercise the soft-fail and decline branches.
    """

    def __init__(
        self,
        *,
        handle: str = "ctxid-1",
        summary: str = "stored",
        raises: Exception | None = None,
        returns_none: bool = False,
    ) -> None:
        self.handle = handle
        self.summary = summary
        self._raises = raises
        self._returns_none = returns_none
        self.calls: list[dict[str, Any]] = []

    async def offload(
        self,
        *,
        tool_name: str,
        content: str,
        conversation_id: UUID,
        user_id: UUID | None,
    ) -> OffloadResult | None:
        """record the call, then raise / decline / return per construction."""
        self.calls.append(
            {
                "tool_name": tool_name,
                "content": content,
                "conversation_id": conversation_id,
                "user_id": user_id,
            },
        )
        if self._raises is not None:
            raise self._raises
        if self._returns_none:
            return None
        return OffloadResult(summary=self.summary, handle=self.handle)


def _ai_with_tool_call(name: str = "scan", args: dict[str, Any] | None = None) -> AIMessage:
    """build an AIMessage carrying a single tool_call for the seam."""
    msg = AIMessage(content="")
    msg.tool_calls = [{"id": "tc1", "name": name, "args": args or {}}]
    return msg


class _StubTool:
    """minimal langchain-tool-shaped object returning a fixed payload."""

    def __init__(self, name: str, payload: str) -> None:
        self.name = name
        self._payload = payload

    async def ainvoke(self, args: Any, config: Any = None) -> str:
        """return the canned payload regardless of args."""
        return self._payload


def _config(
    tool: Any,
    *,
    offloader: Any = None,
    threshold: int | None = None,
    conversation_id: UUID | None = None,
    user_id: UUID | None = None,
) -> RunnableConfig:
    """assemble a ``configurable`` with the offload seam inputs."""
    configurable: dict[str, Any] = {
        "tools": [tool],
        "_hook_heartbeat_seconds": 0.0,
    }
    if offloader is not None:
        configurable["tool_result_offloader"] = offloader
    if threshold is not None:
        configurable["offload_threshold_chars"] = threshold
    if conversation_id is not None:
        configurable["conversation_id"] = conversation_id
    if user_id is not None:
        configurable["user_id"] = user_id
    return {"configurable": configurable}


class TestOffloadResult:
    """the value object + module default."""

    def test_default_threshold_is_8192(self) -> None:
        """the inline ceiling default is 8K chars (~2K tokens)."""
        assert DEFAULT_OFFLOAD_THRESHOLD_CHARS == 8192

    def test_offload_result_carries_summary_and_handle(self) -> None:
        """OffloadResult exposes summary + handle."""
        result = OffloadResult(summary="s", handle="h")
        assert result.summary == "s"
        assert result.handle == "h"

    def test_offload_result_is_frozen(self) -> None:
        """OffloadResult is an immutable value object."""
        result = OffloadResult(summary="s", handle="h")
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.summary = "mutated"  # type: ignore[misc]


class TestToolNodeOffloadSeam:
    """the ``tool_node`` offload seam behavior."""

    @pytest.mark.asyncio
    async def test_offload_fires_above_threshold(self) -> None:
        """large result -> ToolMessage content becomes summary + [ctx:id]."""
        big = "x" * 100
        tool = _StubTool("scan", big)
        offloader = _FakeOffloader(handle="abc123", summary="scan stored")
        conv = uuid4()
        user = uuid4()
        config = _config(
            tool,
            offloader=offloader,
            threshold=10,
            conversation_id=conv,
            user_id=user,
        )
        state: dict[str, Any] = {"messages": [_ai_with_tool_call("scan")]}
        result = await tool_node(state, config)  # type: ignore[arg-type]
        message = result["messages"][0]
        assert message.content == "scan stored\n\n[ctx:abc123]"
        # the offloader received the FULL content + verified identity.
        assert offloader.calls == [
            {
                "tool_name": "scan",
                "content": big,
                "conversation_id": conv,
                "user_id": user,
            },
        ]

    @pytest.mark.asyncio
    async def test_no_offload_below_threshold(self) -> None:
        """small result stays inline; offloader is never called."""
        small = "tiny"
        tool = _StubTool("scan", small)
        offloader = _FakeOffloader()
        config = _config(
            tool,
            offloader=offloader,
            threshold=100,
            conversation_id=uuid4(),
            user_id=uuid4(),
        )
        state: dict[str, Any] = {"messages": [_ai_with_tool_call("scan")]}
        result = await tool_node(state, config)  # type: ignore[arg-type]
        assert result["messages"][0].content == small
        assert offloader.calls == []

    @pytest.mark.asyncio
    async def test_backward_compatible_without_offloader(self) -> None:
        """no offloader injected -> content is the raw str(tool_result)."""
        big = "y" * 9000
        tool = _StubTool("scan", big)
        config = _config(tool, conversation_id=uuid4(), user_id=uuid4())
        state: dict[str, Any] = {"messages": [_ai_with_tool_call("scan")]}
        result = await tool_node(state, config)  # type: ignore[arg-type]
        assert result["messages"][0].content == big

    @pytest.mark.asyncio
    async def test_soft_fail_falls_back_to_full_content(self) -> None:
        """offloader raising -> full content is preserved (degrade gracefully)."""
        big = "z" * 100
        tool = _StubTool("scan", big)
        offloader = _FakeOffloader(raises=RuntimeError("store down"))
        config = _config(
            tool,
            offloader=offloader,
            threshold=10,
            conversation_id=uuid4(),
            user_id=uuid4(),
        )
        state: dict[str, Any] = {"messages": [_ai_with_tool_call("scan")]}
        result = await tool_node(state, config)  # type: ignore[arg-type]
        assert result["messages"][0].content == big
        assert len(offloader.calls) == 1

    @pytest.mark.asyncio
    async def test_decline_returns_full_content(self) -> None:
        """offloader returning None -> full content is preserved."""
        big = "q" * 100
        tool = _StubTool("scan", big)
        offloader = _FakeOffloader(returns_none=True)
        config = _config(
            tool,
            offloader=offloader,
            threshold=10,
            conversation_id=uuid4(),
            user_id=uuid4(),
        )
        state: dict[str, Any] = {"messages": [_ai_with_tool_call("scan")]}
        result = await tool_node(state, config)  # type: ignore[arg-type]
        assert result["messages"][0].content == big

    @pytest.mark.asyncio
    async def test_failed_tool_result_not_offloaded(self) -> None:
        """a tool that errors keeps its (large) error text inline for the model."""
        offloader = _FakeOffloader()

        class _BoomTool:
            name = "scan"

            async def ainvoke(self, args: Any, config: Any = None) -> str:
                raise ValueError("x" * 100)

        config = _config(
            _BoomTool(),
            offloader=offloader,
            threshold=10,
            conversation_id=uuid4(),
            user_id=uuid4(),
        )
        state: dict[str, Any] = {"messages": [_ai_with_tool_call("scan")]}
        result = await tool_node(state, config)  # type: ignore[arg-type]
        # error text rode inline; offloader was never consulted.
        assert "Tool error" in result["messages"][0].content
        assert offloader.calls == []

    @pytest.mark.asyncio
    async def test_no_conversation_id_skips_offload(self) -> None:
        """without a conversation_id the seam cannot store; content stays inline."""
        big = "w" * 100
        tool = _StubTool("scan", big)
        offloader = _FakeOffloader()
        config = _config(tool, offloader=offloader, threshold=10, user_id=uuid4())
        state: dict[str, Any] = {"messages": [_ai_with_tool_call("scan")]}
        result = await tool_node(state, config)  # type: ignore[arg-type]
        assert result["messages"][0].content == big
        assert offloader.calls == []

    @pytest.mark.asyncio
    async def test_uses_default_threshold_when_unset(self) -> None:
        """absent offload_threshold_chars -> the 8192 default applies."""
        under = "a" * (DEFAULT_OFFLOAD_THRESHOLD_CHARS - 1)
        tool = _StubTool("scan", under)
        offloader = _FakeOffloader()
        config = _config(
            tool,
            offloader=offloader,
            conversation_id=uuid4(),
            user_id=uuid4(),
        )
        state: dict[str, Any] = {"messages": [_ai_with_tool_call("scan")]}
        result = await tool_node(state, config)  # type: ignore[arg-type]
        # one char under the default -> stays inline.
        assert result["messages"][0].content == under
        assert offloader.calls == []

    @pytest.mark.asyncio
    async def test_graph_bubble_up_still_propagates(self) -> None:
        """a GraphInterrupt raised inside a tool propagates through the seam."""
        from langgraph.errors import GraphBubbleUp, GraphInterrupt

        class _InterruptTool:
            name = "scan"

            async def ainvoke(self, args: Any, config: Any = None) -> str:
                raise GraphInterrupt(())

        offloader = _FakeOffloader()
        config = _config(
            _InterruptTool(),
            offloader=offloader,
            threshold=1,
            conversation_id=uuid4(),
            user_id=uuid4(),
        )
        state: dict[str, Any] = {"messages": [_ai_with_tool_call("scan")]}
        with pytest.raises(GraphBubbleUp):
            await tool_node(state, config)  # type: ignore[arg-type]

    def test_protocol_is_importable(self) -> None:
        """the ToolResultOffloader Protocol is part of the public surface."""
        assert ToolResultOffloader is not None

    @pytest.mark.asyncio
    async def test_recall_tool_result_is_never_offloaded(self) -> None:
        """B1: context_recall's OWN result must ride inline, never re-offloaded.

        context_recall is in the standard tool surface, so the model calls
        it THROUGH the graph; its result is the full recalled content
        (larger than the threshold by definition). re-offloading it would
        hand the model a fresh ``[ctx:<id>]`` instead of the bytes -- an
        infinite recall loop. drive it through the REAL seam (not a direct
        get_context_item call) and assert the full content is returned
        inline with no handle wrapping.
        """
        recalled = "z" * 20000
        tool = _StubTool("threetears.context_recall", recalled)
        offloader = _FakeOffloader()
        config = _config(
            tool,
            offloader=offloader,
            threshold=10,
            conversation_id=uuid4(),
            user_id=uuid4(),
        )
        state: dict[str, Any] = {"messages": [_ai_with_tool_call("threetears.context_recall")]}
        result = await tool_node(state, config)  # type: ignore[arg-type]
        assert result["messages"][0].content == recalled
        assert "[ctx:" not in result["messages"][0].content
        assert offloader.calls == []

    @pytest.mark.asyncio
    async def test_recall_tool_sanitized_name_also_excluded(self) -> None:
        """B1: the underscore-sanitized recall name is excluded too (robust match)."""
        recalled = "z" * 20000
        tool = _StubTool("threetears_context_recall", recalled)
        offloader = _FakeOffloader()
        config = _config(
            tool,
            offloader=offloader,
            threshold=10,
            conversation_id=uuid4(),
            user_id=uuid4(),
        )
        state: dict[str, Any] = {"messages": [_ai_with_tool_call("threetears_context_recall")]}
        result = await tool_node(state, config)  # type: ignore[arg-type]
        assert result["messages"][0].content == recalled
        assert offloader.calls == []

    @pytest.mark.asyncio
    async def test_offloader_graph_bubble_up_propagates(self) -> None:
        """N3: a GraphBubbleUp raised by offload(...) propagates, not swallowed."""
        from langgraph.errors import GraphBubbleUp, GraphInterrupt

        big = "x" * 100
        tool = _StubTool("scan", big)
        offloader = _FakeOffloader(raises=GraphInterrupt(()))
        config = _config(
            tool,
            offloader=offloader,
            threshold=10,
            conversation_id=uuid4(),
            user_id=uuid4(),
        )
        state: dict[str, Any] = {"messages": [_ai_with_tool_call("scan")]}
        with pytest.raises(GraphBubbleUp):
            await tool_node(state, config)  # type: ignore[arg-type]


class TestOffloadHelpers:
    """the never-offload + handle-format helpers (contract single-source)."""

    def test_recall_tool_is_never_offload(self) -> None:
        """the recall tool's canonical name is in the never-offload set."""
        assert is_never_offload_tool("threetears.context_recall") is True

    def test_recall_tool_sanitized_is_never_offload(self) -> None:
        """the underscore-sanitized recall name normalizes to the same key."""
        assert is_never_offload_tool("threetears_context_recall") is True

    def test_ordinary_tool_is_offloadable(self) -> None:
        """an ordinary tool is not in the never-offload set."""
        assert is_never_offload_tool("threetears.nmap") is False

    def test_never_offload_set_carries_recall(self) -> None:
        """the exported set names the recall tool canonically."""
        assert "threetears.context_recall" in NEVER_OFFLOAD_TOOLS

    def test_format_offload_handle(self) -> None:
        """the marker shape is ``[ctx:<id>]``."""
        assert format_offload_handle("abc") == "[ctx:abc]"

    def test_has_offload_handle_true_on_offloaded(self) -> None:
        """a summary + trailing handle is detected as already-offloaded."""
        text = f"nmap returned 9000 chars\n\n{format_offload_handle('id-1')}"
        assert has_offload_handle(text) is True

    def test_has_offload_handle_false_on_plain(self) -> None:
        """a plain tool result is not flagged as offloaded."""
        assert has_offload_handle("just a normal 12-line nmap result") is False

    def test_has_offload_handle_false_on_midtext_bracket(self) -> None:
        """a bracketed token mid-text (not a trailing handle) is not flagged."""
        assert has_offload_handle("see [ctx:foo] then more text after it") is False
