"""per-call concurrency cap + hard execution timeout on :class:`ToolServer`.

Two pod-level guards protect a tool pod from a burst of heavy tools (several
scanners at once, a runaway that ignores its own budget):

* ``max_concurrent_calls`` bounds how many ``tool.run`` bodies execute at once.
* ``max_call_seconds`` is a HARD ceiling that force-ends a call running past it
  and invokes the call's registered cleanup hooks -- so a tool that spawned a
  subprocess reaps it rather than orphaning it.

These exercise both guards through the dispatch's guarded runner with a fake
tool, plus the two properties that keep the timeout honest: a tool's OWN
``TimeoutError`` is NOT reclassified as the server's, and hooks fire only when
the ceiling actually trips.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.call_scope import ToolCallScope, register_call_cleanup
from threetears.agent.tools.server import CallRequest, HardCallTimeout, ToolServer


class _FakeTool(TearsTool):
    """a tool whose ``execute`` body is supplied per test."""

    def __init__(self, body: Any) -> None:
        self._body = body

    async def execute(self, **kwargs: Any) -> ToolResult:
        return await self._body()

    def mcp_schema(self) -> MCPToolDefinition:
        return MCPToolDefinition(name="test.fake", version="1.0.0", description="t", input_schema={})

    def mcp_name(self) -> str:
        return "test.fake"

    def mcp_version(self) -> str:
        return "1.0.0"


def _request() -> CallRequest:
    """a minimal call request the guarded runner will pass to ``tool.run``."""
    return CallRequest(tool_name="test.fake", tool_version="1.0.0", arguments={})


def _server(**kwargs: Any) -> ToolServer:
    """a tool server with no NATS wired, configured with the given guards."""
    return ToolServer(nats_url="nats://stub", namespace_collection=None, **kwargs)


async def test_concurrency_cap_bounds_simultaneous_runs() -> None:
    """with the cap at 2, only 2 of 5 launched calls run at once; the rest queue."""
    active = 0
    peak = 0
    started = 0
    gate = asyncio.Event()

    async def body() -> ToolResult:
        nonlocal active, peak, started
        started += 1
        active += 1
        peak = max(peak, active)
        try:
            await gate.wait()
        finally:
            active -= 1
        return ToolResult(success=True, content="ok")

    server = _server(max_concurrent_calls=2)
    tool = _FakeTool(body)
    tasks = [
        # noqa seam: whitebox-tests the per-call concurrency + hard-timeout wrapper directly.
        asyncio.create_task(server._run_tool_guarded(tool, _request(), ToolCallScope()))  # noqa: SLF001
        for _ in range(5)
    ]
    # let every task reach the semaphore; only the cap may be inside `body`.
    await asyncio.sleep(0.05)
    assert active == 2, "more than the cap ran at once"
    assert started == 2, "queued calls entered the tool body before a slot freed"

    gate.set()
    results = await asyncio.gather(*tasks)
    assert all(r.success for r in results)
    assert peak == 2


async def test_hard_timeout_raises_and_runs_cleanup_hooks() -> None:
    """a call past the ceiling raises HardCallTimeout after invoking its hooks."""
    reaped: list[str] = []

    async def body() -> ToolResult:
        register_call_cleanup(lambda: reaped.append("killed"))
        await asyncio.sleep(10)  # far past the ceiling
        return ToolResult(success=True, content="never")

    server = _server(max_call_seconds=0.05)
    scope = ToolCallScope()
    with pytest.raises(HardCallTimeout):
        await server._run_tool_guarded(_FakeTool(body), _request(), scope)  # noqa: SLF001 -- guard seam
    assert reaped == ["killed"], "cleanup hook did not run on hard timeout"


async def test_tool_own_timeout_error_is_not_reclassified() -> None:
    """a tool raising its OWN TimeoutError inside the ceiling stays a plain failure."""
    reaped: list[str] = []

    async def body() -> ToolResult:
        register_call_cleanup(lambda: reaped.append("killed"))
        raise TimeoutError("the tool's own per-scan budget")

    server = _server(max_call_seconds=5.0)
    scope = ToolCallScope()
    with pytest.raises(TimeoutError) as excinfo:
        await server._run_tool_guarded(_FakeTool(body), _request(), scope)  # noqa: SLF001 -- guard seam
    assert not isinstance(excinfo.value, HardCallTimeout)
    assert reaped == [], "cleanup ran for a tool-owned timeout that was not the server's"


async def test_hooks_do_not_run_on_success() -> None:
    """a call that returns within the ceiling never invokes its cleanup hooks."""
    reaped: list[str] = []

    async def body() -> ToolResult:
        register_call_cleanup(lambda: reaped.append("killed"))
        return ToolResult(success=True, content="ok")

    server = _server(max_call_seconds=5.0)
    result = await server._run_tool_guarded(_FakeTool(body), _request(), ToolCallScope())  # noqa: SLF001 -- guard seam
    assert result.success
    assert reaped == []


async def test_unguarded_server_is_pass_through() -> None:
    """with neither guard set the runner just runs the tool (prior behaviour)."""

    async def body() -> ToolResult:
        return ToolResult(success=True, content="ok")

    server = _server()
    result = await server._run_tool_guarded(_FakeTool(body), _request(), ToolCallScope())  # noqa: SLF001 -- guard seam
    assert result.success and result.content == "ok"


def test_rejects_nonpositive_guard_values() -> None:
    """the constructor refuses guard values that could not bound anything."""
    with pytest.raises(ValueError, match="max_concurrent_calls"):
        _server(max_concurrent_calls=0)
    with pytest.raises(ValueError, match="max_call_seconds"):
        _server(max_call_seconds=0)


def test_register_call_cleanup_outside_scope_raises() -> None:
    """registering a hook with no active call scope is a programming error."""
    with pytest.raises(RuntimeError, match="outside a ToolServer call scope"):
        register_call_cleanup(lambda: None)
