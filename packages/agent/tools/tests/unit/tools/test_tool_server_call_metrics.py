"""a tool pod's /metrics: the calls in progress, and every finished call counted with its duration.

The in-flight gauge alone cannot show a pod's load: it is read once per scrape, and a call that starts and
ends between two scrapes (a cached report answer, a tool that hands its work to a background operation) is
never seen. On cobalt dev on 2026-10-30 the reports pods answered thousands of calls in six hours and the
gauge read 0 at every scrape. ``threetears_tools_calls_total`` and
``threetears_tools_call_duration_seconds`` count every call by tool and outcome, so a dashboard reads the
rate, and the concurrency as rate times duration, whatever the scrape interval.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.server import ToolServer
from threetears.core.testing.replay_guard import FakeReplayGuard
from threetears.nats import IncomingMessage

from packages.agent.tools.tests.unit.tools.pod_auth import (
    jwks_provider,
    recording_tool_server,
    signed_call_payload,
)

_POD_ID = "metrics-pod"


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


def _serving(body: Any) -> ToolServer:
    """a tool server holding one fake tool, driven by hand.

    :param body: the coroutine function the fake tool's ``execute`` awaits
    :ptype body: Any
    :return: the server
    :rtype: ToolServer
    """
    server, _rec = recording_tool_server(
        pod_id=_POD_ID, jwks_provider=jwks_provider, assertion_replay_guard=FakeReplayGuard()
    )
    server.register(_FakeTool(body))
    return server


async def _call(server: ToolServer, data: bytes | None = None) -> None:
    """deliver one call to the pod's call handler: an authenticated one for the fake tool by default.

    :param server: the pod
    :ptype server: ToolServer
    :param data: the message body, or ``None`` for a signed call of the fake tool
    :ptype data: bytes | None
    :return: nothing
    :rtype: None
    """
    if data is None:
        payload = signed_call_payload(pod_id=_POD_ID, tool_name="test.fake", tool_version="1.0.0")
        data = json.dumps(payload).encode("utf-8")
    await server.handle_call(
        IncomingMessage(data=data, reply_subject="_INBOX.metrics", subject=f"3tears.tools.internal.{_POD_ID}")
    )


def _sample(server: ToolServer, name: str, **labels: str) -> float | None:
    """one sample of the pod's /metrics body, as a scrape reads it.

    :param server: the pod
    :ptype server: ToolServer
    :param name: the sample's name, suffix included (``_total``, ``_count``)
    :ptype name: str
    :param labels: the sample's labels, all of them
    :ptype labels: str
    :return: the value, or ``None`` when the body has no such sample
    :rtype: float | None
    """
    _content_type, body = server.render_metrics()
    wanted = name + ("{" + ",".join(f'{k}="{v}"' for k, v in sorted(labels.items())) + "}" if labels else "")
    for line in body.decode("utf-8").splitlines():
        if line.startswith("#"):
            continue
        sample, _, value = line.rpartition(" ")
        if sample == wanted:
            return float(value)
    return None


@pytest.mark.asyncio
async def test_a_call_in_progress_reads_one_in_flight_on_metrics() -> None:
    """Flips if a call the pod is running is not counted in flight on /metrics while it runs."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def body() -> ToolResult:
        started.set()
        await release.wait()
        return ToolResult(success=True, content="ok")

    server = _serving(body)
    call = asyncio.create_task(_call(server))
    await asyncio.wait_for(started.wait(), 5)
    assert _sample(server, "threetears_tools_inflight_requests") == 1.0
    release.set()
    await asyncio.wait_for(call, 5)
    assert _sample(server, "threetears_tools_inflight_requests") == 0.0


@pytest.mark.asyncio
async def test_a_finished_call_is_counted_by_tool_and_outcome_with_its_duration() -> None:
    """Flips if a call that ends between two scrapes leaves no trace on /metrics: the count and the duration
    are what a dashboard reads the pod's load from."""

    async def ok() -> ToolResult:
        await asyncio.sleep(0.01)
        return ToolResult(success=True, content="ok")

    server = _serving(ok)
    await _call(server)
    await _call(server)
    assert _sample(server, "threetears_tools_calls_total", tool="test.fake", outcome="success") == 2.0
    assert _sample(server, "threetears_tools_call_duration_seconds_count", tool="test.fake", outcome="success") == 2.0
    total = _sample(server, "threetears_tools_call_duration_seconds_sum", tool="test.fake", outcome="success")
    assert total is not None and total >= 0.02


@pytest.mark.asyncio
async def test_each_outcome_is_counted_apart() -> None:
    """Flips if a failed or raising call is counted as a success, or not at all."""

    async def fails() -> ToolResult:
        return ToolResult(success=False, content="", error="no")

    async def raises() -> ToolResult:
        raise RuntimeError("boom")

    failing = _serving(fails)
    await _call(failing)
    assert _sample(failing, "threetears_tools_calls_total", tool="test.fake", outcome="failure") == 1.0
    raising = _serving(raises)
    await _call(raising)
    assert _sample(raising, "threetears_tools_calls_total", tool="test.fake", outcome="error") == 1.0
    assert _sample(raising, "threetears_tools_calls_total", tool="test.fake", outcome="success") is None


@pytest.mark.asyncio
async def test_a_call_refused_before_any_tool_is_named_is_counted_under_no_tool() -> None:
    """Flips if a call the pod could not read goes uncounted: a refused call is load, and a refusal rate is
    worth seeing."""
    server = _serving(lambda: None)
    await _call(server, data=b"not json")
    assert _sample(server, "threetears_tools_calls_total", tool="", outcome="failure") == 1.0
