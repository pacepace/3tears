"""Two tool servers in one process each build their subjects from their own namespace, never the process's.

**The bug this pins.** ``ToolServer.serve`` and every ``publish_registration`` wrote the process-wide subject
namespace (``set_default_namespace``), and the server's other subjects read it back. With two servers in one process
on namespaces A and B, whichever registered last won the global, and every subject built after it without an explicit
namespace landed in that one: the hub met it as ``NoRespondersError`` on ``intg_*.tools.discover``, a pod left over
from an earlier test rebinding the global under a later one (2026-11-18). A server knows its namespace; it builds
every subject from it, and never reads or writes the global. (``NatsClient.connect`` still binds the process default:
a process's one connection is its deployment's namespace, see :mod:`threetears.nats.subjects`.)
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.server import DiscoveryProbeResponse, RegistrationResponse, ToolServer
from threetears.nats import get_default_namespace, set_default_namespace
from threetears.nats.testing import reset_default_namespace

_POD_A = "01947100-0000-7000-8000-00000000aa01"
_POD_B = "01947100-0000-7000-8000-00000000bb01"

#: what the process's default namespace is before either server starts: neither may change it
_AMBIENT = "ambient"


class _StubTool(TearsTool):
    """the smallest tool a manifest carries."""

    async def execute(self, **kwargs: Any) -> ToolResult:
        """no-op body.

        :param kwargs: ignored
        :ptype kwargs: Any
        :return: a trivial result
        :rtype: ToolResult
        """
        return ToolResult(success=True, content="")

    def mcp_schema(self) -> MCPToolDefinition:
        """the stub's schema.

        :return: schema with an empty-object input
        :rtype: MCPToolDefinition
        """
        return MCPToolDefinition(
            name="stub.tool", version="1.0", description="stub", input_schema={"type": "object", "properties": {}}
        )

    def mcp_name(self) -> str:
        """the stub's name.

        :return: the name
        :rtype: str
        """
        return "stub.tool"

    def mcp_version(self) -> str:
        """the stub's version.

        :return: the version
        :rtype: str
        """
        return "1.0"


def _recording_nc() -> AsyncMock:
    """a NATS client double that records every subject it is handed: enough to drive ``serve``.

    :return: the double; ``subjects`` lists each subject path, in order
    :rtype: AsyncMock
    """
    nc = AsyncMock()
    nc.is_connected = True
    nc.renew_credential = MagicMock()
    nc.subjects = []

    def _seen(kwargs: dict[str, Any]) -> None:
        subject = kwargs.get("subject")
        if subject is not None:
            nc.subjects.append(subject.path)

    async def _subscribe(*args: Any, **kwargs: Any) -> MagicMock:
        _seen(kwargs)
        return MagicMock()

    async def _publish(*args: Any, **kwargs: Any) -> None:
        _seen(kwargs)

    async def _request(*args: Any, **kwargs: Any) -> DiscoveryProbeResponse | RegistrationResponse:
        _seen(kwargs)
        # a registry that admits the registration and reports no tool ready
        if kwargs.get("response_type") is RegistrationResponse:
            return RegistrationResponse(success=True, pod_id=kwargs["message"].pod_id)
        return DiscoveryProbeResponse(agent_id="x", tools=[])

    async def _request_raw(*args: Any, **kwargs: Any) -> bytes:
        return json.dumps({"keys": []}).encode("utf-8")

    nc.subscribe = AsyncMock(side_effect=_subscribe)
    nc.publish = AsyncMock(side_effect=_publish)
    nc.request = AsyncMock(side_effect=_request)
    nc.request_raw = AsyncMock(side_effect=_request_raw)
    return nc


@asynccontextmanager
async def _serving(server: ToolServer) -> AsyncIterator[None]:
    """the server serving for the body of the block.

    :param server: the server
    :ptype server: ToolServer
    :return: nothing
    :rtype: AsyncIterator[None]
    """
    task = asyncio.create_task(server.serve())
    try:
        await server.wait_ready(timeout=2.0)
        await asyncio.sleep(0)
        yield
    finally:
        await server.shutdown()
        await asyncio.wait_for(task, timeout=2.0)


@pytest.fixture(autouse=True)
def _ambient_namespace() -> Iterator[None]:
    """the process's default namespace set to :data:`_AMBIENT` for the test, and cleared after it.

    :return: nothing
    :rtype: Iterator[None]
    """
    set_default_namespace(_AMBIENT)
    yield
    reset_default_namespace()


@pytest.mark.asyncio
async def test_a_server_s_subjects_stay_in_its_namespace_after_another_server_registers_last() -> None:
    """Flips if a tool server builds any subject from the process's namespace rather than its own, or writes the
    process's namespace: B serves, then A serves and registers last, and B's registration and discovery still go to
    B's namespace, A's to A's, and the process default is what it was."""
    nc_a, nc_b = _recording_nc(), _recording_nc()
    server_a = ToolServer(nats_client=nc_a, namespace="nsa", pod_id=_POD_A)
    server_b = ToolServer(nats_client=nc_b, namespace="nsb", pod_id=_POD_B)
    server_a.register(_StubTool())
    server_b.register(_StubTool())
    async with _serving(server_b), _serving(server_a):
        await server_a.publish_registration()
        # A registered last: B's discovery probe, then its own registration, are built after it
        assert await server_b.wait_until_ready(timeout=0.2) is False  # the double's registry reports nothing ready
        await server_b.publish_registration()
        assert await server_a.wait_until_ready(timeout=0.2) is False
        ambient = get_default_namespace()

    assert nc_b.subjects and nc_a.subjects
    assert [s for s in nc_b.subjects if not s.startswith("nsb.")] == [], nc_b.subjects
    assert [s for s in nc_a.subjects if not s.startswith("nsa.")] == [], nc_a.subjects
    assert "nsb.tools.register" in nc_b.subjects and "nsb.tools.discover" in nc_b.subjects
    assert "nsa.tools.register" in nc_a.subjects and "nsa.tools.discover" in nc_a.subjects
    assert f"nsb.tools.internal.{_POD_B}" in nc_b.subjects and f"nsb.tools.probe.{_POD_B}" in nc_b.subjects
    assert ambient == _AMBIENT, "a tool server wrote the process's namespace"
