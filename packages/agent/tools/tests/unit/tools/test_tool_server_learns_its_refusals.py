"""A tool server learns which of its own tools the registry refused, and says so.

A refused tool used to be a warning on the REGISTRY and silence on the pod: the pod's readiness
counted any available copy of the name -- including another pod's -- so a pod whose own copy was
refused came up "ready", serving nothing, and nobody reading its log could tell. Now:

* the registration reply names every refused tool (``RegistrationResponse.refused_tools``), and
  the server logs each at ERROR and keeps them on :attr:`ToolServer.refused_tools`;
* :meth:`ToolServer.wait_until_ready` names this pod in its discovery poll and waits for THIS pod's
  copy of every tool (``requester_copy_status == "available"``), and raises
  :class:`ToolRegistrationRefused` the moment a refusal is known rather than timing out;
* :meth:`ToolServer.register_tool` on a serving pod awaits the reply and raises for the tool it
  just added when that tool is refused.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid7

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.server import (
    DiscoveryProbeRequest,
    DiscoveryProbeResponse,
    DiscoveryProbeResultEntry,
    RefusedTool,
    RegistrationManifest,
    RegistrationResponse,
    ToolRegistrationRefused,
    ToolServer,
)

_POD = "01947100-0000-7000-8000-00000000ab02"


class _Tool(TearsTool):
    """a stub tool with a configurable name."""

    def __init__(self, name: str = "threetears.calculator") -> None:
        """set the tool's mcp name.

        :param name: the mcp name
        :ptype name: str
        :return: nothing
        :rtype: None
        """
        super().__init__()
        self._name = name

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

        :return: an empty-object schema
        :rtype: MCPToolDefinition
        """
        return MCPToolDefinition(
            name=self._name, version="1.0", description="stub", input_schema={"type": "object", "properties": {}}
        )

    def mcp_name(self) -> str:
        """the stub's mcp name.

        :return: the name
        :rtype: str
        """
        return self._name

    def mcp_version(self) -> str:
        """the stub's version.

        :return: the version
        :rtype: str
        """
        return "1.0"


def _refusal(name: str = "threetears.calculator", code: str = "NOT_PLATFORM_SHARED") -> RefusedTool:
    """one refusal of a 1.0 tool.

    :param name: the refused tool
    :ptype name: str
    :param code: the refusal code
    :ptype code: str
    :return: the refusal
    :rtype: RefusedTool
    """
    return RefusedTool(name=name, version="1.0", code=code, reason="only the platform serves this to everyone")


def _server(*names: str) -> ToolServer:
    """a server holding the named tools, with no live connection.

    :param names: tool names; one calculator when omitted
    :ptype names: str
    :return: the server
    :rtype: ToolServer
    """
    server = ToolServer(agent_id=uuid7(), nats_url="nats://test:4222", pod_id=_POD)
    for name in names or ("threetears.calculator",):
        server.register(_Tool(name))
    return server


def _replying(*replies: Any) -> AsyncMock:
    """a NATS double whose successive requests answer with ``replies``.

    :param replies: the replies, in order
    :ptype replies: Any
    :return: the double
    :rtype: AsyncMock
    """
    nc = AsyncMock()
    nc.request = AsyncMock(side_effect=list(replies))
    return nc


class TestTheReplyNamesTheRefusals:
    """the pod holds and logs what the registry refused."""

    async def test_refusals_are_kept_and_logged_at_error_one_per_tool(self, caplog: pytest.LogCaptureFixture) -> None:
        """two refused tools, two ERROR lines, both on the property.

        :return: none
        :rtype: None
        """
        server = _server("threetears.calculator", "threetears.dictionary")
        server._nc = _replying(  # noqa: SLF001
            RegistrationResponse(
                success=False,
                pod_id=_POD,
                refused_tools=[_refusal(), _refusal("threetears.dictionary", "OWNED_ELSEWHERE")],
                error="no tools authorized",
                error_code="NO_TOOLS_ADMITTED",
            )
        )
        with caplog.at_level(logging.ERROR, logger="threetears.agent.tools.server"):
            await server.publish_registration(await_reply=True)

        assert {(r.name, r.code) for r in server.refused_tools} == {
            ("threetears.calculator", "NOT_PLATFORM_SHARED"),
            ("threetears.dictionary", "OWNED_ELSEWHERE"),
        }
        errors = [r for r in caplog.records if r.levelno == logging.ERROR and "refused" in r.getMessage()]
        assert len(errors) == 2

    async def test_a_later_clean_reply_clears_them(self) -> None:
        """the latest reply is the truth; a refusal fixed upstream stops being reported.

        :return: none
        :rtype: None
        """
        server = _server()
        server._nc = _replying(  # noqa: SLF001
            RegistrationResponse(success=True, pod_id=_POD, refused_tools=[_refusal("threetears.other")]),
            RegistrationResponse(success=True, pod_id=_POD, registered_tools=["threetears.calculator@1.0"]),
        )
        await server.publish_registration(await_reply=True)
        assert len(server.refused_tools) == 1
        await server.publish_registration(await_reply=True)
        assert server.refused_tools == ()

    async def test_a_publish_that_does_not_await_learns_nothing(self) -> None:
        """the heartbeat's plain publish neither asks nor clears.

        :return: none
        :rtype: None
        """
        server = _server()
        nc = AsyncMock()
        server._nc = nc  # noqa: SLF001
        await server.publish_registration()
        nc.request.assert_not_awaited()
        assert server.refused_tools == ()


class TestReadinessIsThisPodsOwnCopy:
    """wait_until_ready waits for its own copy, and gives up at once on a refusal."""

    async def test_a_refusal_raises_before_any_poll(self) -> None:
        """no timeout to wait out: the refusal is already known.

        :return: none
        :rtype: None
        """
        server = _server()
        nc = _replying(RegistrationResponse(success=False, pod_id=_POD, refused_tools=[_refusal()]))
        server._nc = nc  # noqa: SLF001
        await server.publish_registration(await_reply=True)
        nc.request.reset_mock()

        with pytest.raises(ToolRegistrationRefused) as excinfo:
            await server.wait_until_ready(timeout=5.0)

        assert [r.name for r in excinfo.value.refused] == ["threetears.calculator"]
        assert "NOT_PLATFORM_SHARED" in str(excinfo.value)
        nc.request.assert_not_awaited()

    async def test_another_pods_available_copy_does_not_make_this_pod_ready(self) -> None:
        """status available with this pod's copy absent is not ready.

        :return: none
        :rtype: None
        """
        server = _server()
        absent = DiscoveryProbeResponse(
            agent_id=_POD,
            tools=[
                DiscoveryProbeResultEntry(
                    name="threetears.calculator", version="1.0", status="available", requester_copy_status="absent"
                )
            ],
        )
        nc = AsyncMock()
        nc.request = AsyncMock(return_value=absent)
        server._nc = nc  # noqa: SLF001
        assert await server.wait_until_ready(timeout=0.3) is False

    async def test_its_own_available_copy_makes_it_ready_and_the_poll_names_the_pod(self) -> None:
        """the request carries pod_id; the answer about THIS copy decides.

        :return: none
        :rtype: None
        """
        server = _server()
        ready = DiscoveryProbeResponse(
            agent_id=_POD,
            tools=[
                DiscoveryProbeResultEntry(
                    name="threetears.calculator", version="1.0", status="available", requester_copy_status="available"
                )
            ],
        )
        nc = AsyncMock()
        nc.request = AsyncMock(return_value=ready)
        server._nc = nc  # noqa: SLF001
        assert await server.wait_until_ready(timeout=2.0) is True
        request = nc.request.await_args.kwargs["message"]
        assert isinstance(request, DiscoveryProbeRequest)
        assert request.pod_id == _POD


class TestADynamicRegistrationHearsItsAnswer:
    """register_tool on a serving pod awaits the registry's reply."""

    async def test_a_refused_new_tool_raises(self) -> None:
        """the tool just added was refused: the caller learns it here.

        :return: none
        :rtype: None
        """
        server = _server()
        nc = _replying(
            RegistrationResponse(success=True, pod_id=_POD, refused_tools=[_refusal("threetears.dictionary")])
        )
        server._nc = nc  # noqa: SLF001
        server._ready_event.set()  # noqa: SLF001

        with pytest.raises(ToolRegistrationRefused):
            await server.register_tool(_Tool("threetears.dictionary"))

        assert isinstance(nc.request.await_args.kwargs["message"], RegistrationManifest)

    async def test_an_admitted_new_tool_does_not_raise_for_an_older_refusal(self) -> None:
        """only the tool being added is this call's business.

        :return: none
        :rtype: None
        """
        server = _server()
        server._nc = _replying(  # noqa: SLF001
            RegistrationResponse(success=True, pod_id=_POD, refused_tools=[_refusal("threetears.calculator")])
        )
        server._ready_event.set()  # noqa: SLF001
        await server.register_tool(_Tool("threetears.dictionary"))

    async def test_before_serving_the_manifest_is_published_without_waiting(self) -> None:
        """no probe subject is bound yet, so waiting would only wait out the registry's probe.

        :return: none
        :rtype: None
        """
        server = _server()
        nc = AsyncMock()
        server._nc = nc  # noqa: SLF001
        await server.register_tool(_Tool("threetears.dictionary"))
        nc.request.assert_not_awaited()
        nc.publish.assert_awaited()


class TestTheProbeRequestNeverSendsANullPodId:
    """``DiscoveryProbeRequest.pod_id`` is omitted when unset."""

    def test_unset_is_absent(self) -> None:
        """no key, not a null.

        :return: none
        :rtype: None
        """
        assert "pod_id" not in DiscoveryProbeRequest(agent_id="a", tool_manifest=[]).model_dump(mode="json")
