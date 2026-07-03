"""Tests for the produced-object catalog middleware seam (Path-2 P2.2-C).

When a producing tool returns an ObjectHandle in its result artifact (under
``OBJECT_HANDLE_METADATA_KEY``), the
:class:`~threetears.langgraph.middleware_catalog.ObjectCatalogMiddleware`
``wrap_tool_call`` seam (the ``create_agent`` successor to the old ``tool_node``
catalog branch) persists a catalog record via the injected
:class:`~threetears.langgraph.catalog.ObjectCataloger`, using the VERIFIED call
identity. The catalog is a soft-fail side-effect: a missing cataloger, a missing
identity, a malformed handle, or a catalog error must never break the tool result
(an uncataloged object is reconciled later).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from langchain.agents.middleware import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.errors import GraphBubbleUp

from threetears.langgraph.middleware_catalog import ObjectCatalogMiddleware
from threetears.media.contracts import OBJECT_HANDLE_METADATA_KEY, ObjectHandle

_CUSTOMER = UUID("06a41d51-a6d5-7824-8000-29ab66754fc0")
_OBJECT = UUID("019f1924-1a31-72d3-81b4-855415bd34ba")


# parity-with: threetears.langgraph.catalog.ObjectCataloger
class _FakeCataloger:
    """Records catalog() calls; optionally raises to exercise the soft-fail."""

    def __init__(self, *, raises: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._raises = raises

    async def catalog(
        self,
        handle: ObjectHandle,
        *,
        conversation_id: UUID,
        customer_id: UUID,
        user_id: UUID | None,
    ) -> None:
        self.calls.append(
            {
                "handle": handle,
                "conversation_id": conversation_id,
                "customer_id": customer_id,
                "user_id": user_id,
            }
        )
        if self._raises is not None:
            raise self._raises


def _ai_with_tool_call(name: str = "scan") -> AIMessage:
    """An AIMessage carrying one tool_call for the seam to dispatch."""
    msg = AIMessage(content="")
    msg.tool_calls = [{"id": "tc1", "name": name, "args": {}}]
    return msg


def _producing_tool(name: str, artifact: Any) -> StructuredTool:
    """A content_and_artifact tool returning a small summary + the artifact."""

    async def _impl(**kwargs: Any) -> tuple[str, Any]:
        return "2 hosts, 5 open ports", artifact

    return StructuredTool.from_function(
        coroutine=_impl,
        name=name,
        description="producing tool stub",
        response_format="content_and_artifact",
    )


def _handle() -> ObjectHandle:
    """A representative produced-object handle."""
    return ObjectHandle(
        object_id=_OBJECT,
        s3_key=f"{_CUSTOMER}/conversation-x/scans/2026/06/30/{_OBJECT}/scan.xml",
        mime_type="application/xml",
        size_bytes=4096,
        summary="2 hosts, 5 open ports",
        category="scans",
    )


def _config(
    tool: Any,
    *,
    cataloger: Any = None,
    call_context: Any = None,
) -> RunnableConfig:
    """Assemble a ``configurable`` with the catalog seam inputs."""
    configurable: dict[str, Any] = {"tools": [tool], "_hook_heartbeat_seconds": 0.0}
    if cataloger is not None:
        configurable["object_cataloger"] = cataloger
    if call_context is not None:
        configurable["call_context"] = call_context
    return {"configurable": configurable}


def _ctx(*, customer_id: UUID | None = _CUSTOMER, conversation_id: UUID | None = None) -> SimpleNamespace:
    """A minimal call_context stub carrying the identity the seam reads."""
    return SimpleNamespace(
        customer_id=customer_id,
        conversation_id=conversation_id if conversation_id is not None else uuid4(),
        user_id=uuid4(),
    )


def _make_handler(tool_map: dict[str, Any], config: RunnableConfig) -> Any:
    """Build an async tool-dispatch handler mirroring LangGraph's ``ToolNode``.

    Invokes a registered tool ToolCall-style (so a ``content_and_artifact`` tool
    yields a ToolMessage carrying its ``artifact``, which the catalog seam reads);
    a ``GraphBubbleUp`` propagates; any other error becomes a ``status="error"``
    ToolMessage. This is the ``handler`` LangGraph hands to ``awrap_tool_call``.
    """

    async def _handler(request: ToolCallRequest) -> ToolMessage:
        call = request.tool_call
        tool = tool_map.get(call["name"])
        if tool is None:
            return ToolMessage(
                content=f"tool '{call['name']}' not found",
                tool_call_id=call["id"],
                status="error",
            )
        try:
            invoked = await tool.ainvoke(
                {"type": "tool_call", "id": call["id"], "name": call["name"], "args": call["args"]},
                config,
            )
        except GraphBubbleUp:
            raise
        except Exception as exc:  # noqa: BLE001 - ToolNode-parity: surface as an error ToolMessage
            return ToolMessage(
                content=f"Tool error: {call['name']}: {exc}",
                tool_call_id=call["id"],
                status="error",
            )
        if isinstance(invoked, ToolMessage):
            return invoked
        return ToolMessage(content=invoked, tool_call_id=call["id"])

    return _handler


async def _run_seam(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """Drive :class:`ObjectCatalogMiddleware` over the last AI message's tool calls.

    Stands in for the removed ``tool_node``: builds the real
    :class:`~langchain.agents.middleware.ToolCallRequest`, hands the middleware a
    ToolNode-parity handler, and returns ``{"messages": [...]}`` so every existing
    assertion (``result["messages"][0].content``) reads unchanged.
    """
    middleware = ObjectCatalogMiddleware()
    configurable = config["configurable"]
    tool_map = {t.name: t for t in configurable.get("tools", [])}
    last = state["messages"][-1]
    messages: list[Any] = []
    for call in last.tool_calls:
        request = ToolCallRequest(
            tool_call=call,
            tool=tool_map.get(call["name"]),
            state=state,
            runtime=SimpleNamespace(config=config),
        )
        result = await middleware.awrap_tool_call(request, _make_handler(tool_map, config))
        messages.append(result)
    return {"messages": messages}


@pytest.mark.asyncio
async def test_catalog_fires_on_object_handle_artifact() -> None:
    """A handle in the artifact -> the cataloger persists it with verified identity."""
    handle = _handle()
    tool = _producing_tool("scan", {OBJECT_HANDLE_METADATA_KEY: handle.to_metadata()})
    cataloger = _FakeCataloger()
    ctx = _ctx()
    config = _config(tool, cataloger=cataloger, call_context=ctx)
    state: dict[str, Any] = {"messages": [_ai_with_tool_call("scan")]}
    result = await _run_seam(state, config)  # type: ignore[arg-type]
    # the tool result still flows.
    assert result["messages"][0].content == "2 hosts, 5 open ports"
    # the cataloger received the reconstructed handle + the verified identity.
    assert len(cataloger.calls) == 1
    call = cataloger.calls[0]
    assert call["handle"].object_id == handle.object_id
    assert call["handle"].s3_key == handle.s3_key
    assert call["handle"].category == "scans"
    assert call["handle"].size_bytes == 4096
    assert call["customer_id"] == _CUSTOMER
    assert call["conversation_id"] == ctx.conversation_id


@pytest.mark.asyncio
async def test_no_cataloger_is_noop() -> None:
    """No cataloger injected -> tool result flows, nothing cataloged, no error."""
    tool = _producing_tool("scan", {OBJECT_HANDLE_METADATA_KEY: _handle().to_metadata()})
    config = _config(tool, call_context=_ctx())
    state: dict[str, Any] = {"messages": [_ai_with_tool_call("scan")]}
    result = await _run_seam(state, config)  # type: ignore[arg-type]
    assert result["messages"][0].content == "2 hosts, 5 open ports"


@pytest.mark.asyncio
async def test_artifact_without_handle_is_not_cataloged() -> None:
    """An artifact carrying no object handle -> the cataloger is not called."""
    tool = _producing_tool("scan", {"summary": "no object here", "rows": 3})
    cataloger = _FakeCataloger()
    config = _config(tool, cataloger=cataloger, call_context=_ctx())
    state: dict[str, Any] = {"messages": [_ai_with_tool_call("scan")]}
    await _run_seam(state, config)  # type: ignore[arg-type]
    assert cataloger.calls == []


@pytest.mark.asyncio
async def test_missing_customer_skips_catalog() -> None:
    """No verified customer_id -> the object is not cataloged (untenanted)."""
    tool = _producing_tool("scan", {OBJECT_HANDLE_METADATA_KEY: _handle().to_metadata()})
    cataloger = _FakeCataloger()
    config = _config(tool, cataloger=cataloger, call_context=_ctx(customer_id=None))
    state: dict[str, Any] = {"messages": [_ai_with_tool_call("scan")]}
    result = await _run_seam(state, config)  # type: ignore[arg-type]
    assert cataloger.calls == []
    # the tool result still flows.
    assert result["messages"][0].content == "2 hosts, 5 open ports"


@pytest.mark.asyncio
async def test_malformed_handle_soft_fails() -> None:
    """A handle dict missing required fields -> logged + skipped, result flows."""
    tool = _producing_tool("scan", {OBJECT_HANDLE_METADATA_KEY: {"summary": "incomplete"}})
    cataloger = _FakeCataloger()
    config = _config(tool, cataloger=cataloger, call_context=_ctx())
    state: dict[str, Any] = {"messages": [_ai_with_tool_call("scan")]}
    result = await _run_seam(state, config)  # type: ignore[arg-type]
    assert cataloger.calls == []
    assert result["messages"][0].content == "2 hosts, 5 open ports"


@pytest.mark.asyncio
async def test_catalog_failure_soft_fails() -> None:
    """A cataloger error is swallowed -> the tool result still flows (orphan -> reconciler)."""
    tool = _producing_tool("scan", {OBJECT_HANDLE_METADATA_KEY: _handle().to_metadata()})
    cataloger = _FakeCataloger(raises=RuntimeError("db down"))
    config = _config(tool, cataloger=cataloger, call_context=_ctx())
    state: dict[str, Any] = {"messages": [_ai_with_tool_call("scan")]}
    result = await _run_seam(state, config)  # type: ignore[arg-type]
    assert len(cataloger.calls) == 1  # attempted
    assert result["messages"][0].content == "2 hosts, 5 open ports"  # but result survives
