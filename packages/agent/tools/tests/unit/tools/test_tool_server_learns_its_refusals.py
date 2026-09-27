"""A tool server learns which of its own tools the registry refused, and says so.

A refused tool used to be a warning on the REGISTRY and silence on the pod: the pod's readiness
counted any available copy of the name -- including another pod's -- so a pod whose own copy was
refused came up "ready", serving nothing, and nobody reading its log could tell. Now:

* the registration reply names every refused tool (``RegistrationResponse.refused_tools``), and
  the server logs each at ERROR and keeps them on :attr:`ToolServer.refused_tools`;
* :meth:`ToolServer.wait_until_ready` names this pod in its discovery poll and waits for THIS pod's
  copy of every tool (``requester_copy_status == "available"``), and raises
  :class:`ToolRegistrationRefused` the moment a FINAL refusal is known rather than timing out;
* :meth:`ToolServer.register_tool` on a serving pod awaits the reply and raises for the tool it
  just added when that tool is refused finally.

A temporary refusal -- the registry could not read its ownership graph, or an older registry's
failed reply carrying no code -- is waited out while the heartbeat re-offers the manifest
(:data:`FINAL_REFUSAL_CODES` is the one classification).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid7

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.server import (
    FINAL_REFUSAL_CODES,
    DiscoveryProbeRequest,
    DiscoveryProbeResponse,
    DiscoveryProbeResultEntry,
    RefusedTool,
    RegistrationManifest,
    RegistrationResponse,
    ToolRegistrationRefused,
    ToolServer,
    refusal_is_final,
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


# ---------------------------------------------------------------------------
# a temporary refusal is waited out; only a final one ends readiness
# ---------------------------------------------------------------------------
#
# every refusal used to be fatal to readiness. two are not verdicts at all:
# OWNERSHIP_GRAPH_UNAVAILABLE (the registry could not read its graph and says
# the next heartbeat retries), and a failed reply carrying no code -- what an
# OLDER registry sends, e.g. "invalid bootstrap token", while a deploy is
# mid-roll. raising on those failed a pod the next heartbeat would have admitted.

_REGISTRY_FINAL = {
    "OWNED_ELSEWHERE",
    "NOT_PLATFORM_SHARED",
    "POD_ID_MISMATCH",
    "INVALID_TOOL_NAME",
    "INVALID_MANIFEST",
    "NO_TOOLS_ADMITTED",
}


class _ScriptedRegistry:
    """a registry double: answers manifests from a script, and discovery from what it last admitted.

    a manifest counts as registered whether it was requested or plainly published, as on the real
    registry. discovery reports this pod's copy available only once a manifest was admitted.
    """

    def __init__(self, *replies: RegistrationResponse, copy_status_field: bool = True) -> None:
        """script the registration replies; the last one repeats.

        :param replies: the registration replies, in order
        :ptype replies: RegistrationResponse
        :param copy_status_field: whether discovery reports ``requester_copy_status`` -- an older
            registry does not
        :ptype copy_status_field: bool
        :return: nothing
        :rtype: None
        """
        self._replies = list(replies)
        self._copy_status_field = copy_status_field
        self.admitted = False
        self.manifests: list[RegistrationManifest] = []
        self.awaited: list[bool] = []
        self.nc = AsyncMock()
        self.nc.is_closed = False
        self.nc.is_healthy = True
        self.nc.request = AsyncMock(side_effect=self._request)
        self.nc.publish = AsyncMock(side_effect=self._publish)

    def _register(self, manifest: RegistrationManifest, *, awaited: bool) -> RegistrationResponse:
        """register one manifest and return the scripted verdict.

        :param manifest: the manifest
        :ptype manifest: RegistrationManifest
        :param awaited: whether the pod asked for the reply
        :ptype awaited: bool
        :return: the verdict
        :rtype: RegistrationResponse
        """
        self.manifests.append(manifest)
        self.awaited.append(awaited)
        reply = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]
        self.admitted = reply.success and not reply.refused_tools
        return reply

    async def _publish(self, *, subject: Any, message: Any) -> None:
        """a plain publish: a manifest still registers, a heartbeat is ignored.

        :param subject: the subject
        :ptype subject: Any
        :param message: the message
        :ptype message: Any
        :return: nothing
        :rtype: None
        """
        if isinstance(message, RegistrationManifest):
            self._register(message, awaited=False)

    async def _request(self, *, subject: Any, message: Any, response_type: Any, timeout: Any) -> Any:
        """answer a registration or a discovery poll.

        :param subject: the subject
        :ptype subject: Any
        :param message: the request
        :ptype message: Any
        :param response_type: the expected reply type
        :ptype response_type: Any
        :param timeout: the request timeout
        :ptype timeout: Any
        :return: the reply
        :rtype: Any
        """
        if isinstance(message, RegistrationManifest):
            return self._register(message, awaited=True)
        status = "available" if self.admitted else "pending"
        entry = DiscoveryProbeResultEntry(
            name="threetears.calculator",
            version="1.0",
            status=status if self._copy_status_field or self.admitted else "unavailable",
            requester_copy_status=status if self._copy_status_field else None,
        )
        return DiscoveryProbeResponse(agent_id=_POD, tools=[entry])


def _heartbeating_server(registry: _ScriptedRegistry) -> ToolServer:
    """a server wired to ``registry`` with a fast heartbeat, not yet heartbeating.

    :param registry: the registry double
    :ptype registry: _ScriptedRegistry
    :return: the server
    :rtype: ToolServer
    """
    server = ToolServer(agent_id=uuid7(), nats_url="nats://test:4222", pod_id=_POD, heartbeat_interval=0.05)
    server.register(_Tool())
    server._nc = registry.nc  # noqa: SLF001
    return server


def _graph_unavailable(*, success: bool) -> RegistrationResponse:
    """the registry could not read its ownership graph.

    :param success: whether the reply as a whole succeeded (another tool admitted)
    :ptype success: bool
    :return: the reply
    :rtype: RegistrationResponse
    """
    return RegistrationResponse(
        success=success,
        pod_id=_POD,
        owned_namespaces=["tools.calc"] if success else [],
        refused_tools=[_refusal(code="OWNERSHIP_GRAPH_UNAVAILABLE")],
        error=None if success else "ownership graph unavailable; retried on the pod's next heartbeat",
        error_code=None if success else "OWNERSHIP_GRAPH_UNAVAILABLE",
    )


_ADMITTED = RegistrationResponse(
    success=True, pod_id=_POD, owned_namespaces=["tools.calc"], registered_tools=["threetears.calculator@1.0"]
)


class TestTheClassificationIsOneSet:
    def test_the_final_codes_are_the_registrys_permanent_verdicts(self) -> None:
        """the six codes that no retry changes; nothing else ends readiness.

        :return: none
        :rtype: None
        """
        assert FINAL_REFUSAL_CODES == _REGISTRY_FINAL

    @pytest.mark.parametrize("code", ["OWNERSHIP_GRAPH_UNAVAILABLE", "UNVERIFIED_PUBLISHER", None, "A_NEWER_CODE"])
    def test_everything_else_including_no_code_is_temporary(self, code: str | None) -> None:
        """an absent or unknown code is not a verdict this pod can act on by stopping.

        :param code: the refusal code
        :ptype code: str | None
        :return: none
        :rtype: None
        """
        assert refusal_is_final(code) is False

    @pytest.mark.parametrize("code", sorted(_REGISTRY_FINAL))
    def test_each_final_code_is_final(self, code: str) -> None:
        """a code no retry changes.

        :param code: the refusal code
        :ptype code: str
        :return: none
        :rtype: None
        """
        assert refusal_is_final(code) is True


class TestATemporaryRefusalIsWaitedOut:
    async def test_a_temporary_refusal_is_admitted_on_a_later_heartbeat_with_no_restart(self) -> None:
        """the registry refuses for want of its graph, the heartbeat re-offers, the pod is ready.

        :return: none
        :rtype: None
        """
        registry = _ScriptedRegistry(_graph_unavailable(success=False), _ADMITTED)
        server = _heartbeating_server(registry)
        await server.publish_registration(await_reply=True)
        server._running = True  # noqa: SLF001
        heartbeat = asyncio.create_task(server._heartbeat_loop())  # noqa: SLF001
        try:
            assert await server.wait_until_ready(timeout=3.0) is True
        finally:
            server._running = False  # noqa: SLF001
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
        assert len(registry.manifests) >= 2
        assert server.refused_tools == ()

    async def test_the_heartbeat_keeps_asking_while_a_refusal_stands(self) -> None:
        """a reply that admitted other tools set the pod's identity; the refusal still gets re-read.

        :return: none
        :rtype: None
        """
        registry = _ScriptedRegistry(_graph_unavailable(success=True), _ADMITTED)
        server = _heartbeating_server(registry)
        await server.publish_registration(await_reply=True)
        assert server.owned_namespaces == ("tools.calc",)
        server._running = True  # noqa: SLF001
        heartbeat = asyncio.create_task(server._heartbeat_loop())  # noqa: SLF001
        try:
            for _ in range(100):
                if len(registry.manifests) >= 2:
                    break
                await asyncio.sleep(0.01)
        finally:
            server._running = False  # noqa: SLF001
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
        assert registry.awaited[1] is True
        assert server.refused_tools == ()

    async def test_a_temporary_refusal_is_warned_once_naming_its_cause(self, caplog: pytest.LogCaptureFixture) -> None:
        """many polls, one WARNING naming the code; no raise.

        :return: none
        :rtype: None
        """
        registry = _ScriptedRegistry(_graph_unavailable(success=False))
        server = _heartbeating_server(registry)
        await server.publish_registration(await_reply=True)
        with caplog.at_level(logging.WARNING, logger="threetears.agent.tools.server"):
            assert await server.wait_until_ready(timeout=0.3) is False
        named = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "OWNERSHIP_GRAPH_UNAVAILABLE" in r.getMessage()
        ]
        assert len(named) == 1

    async def test_a_temporary_refusal_that_clears_and_returns_is_warned_again(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """once per EPISODE, not once per process: signing keys rotate again and hub reads fail again.

        a cause logged once for the pod's lifetime made the second outage silent on the pod, and
        nothing marked the first one clearing.

        :return: none
        :rtype: None
        """
        admitted = RegistrationResponse(success=True, pod_id=_POD, registered_tools=["threetears.calculator@1.0"])
        server = _server()
        server._nc = _replying(  # noqa: SLF001
            _graph_unavailable(success=False), admitted, _graph_unavailable(success=False)
        )
        with caplog.at_level(logging.INFO, logger="threetears.agent.tools.server"):
            for _ in range(3):
                await server.publish_registration(await_reply=True)

        warned = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "OWNERSHIP_GRAPH_UNAVAILABLE" in r.getMessage()
        ]
        cleared = [
            r for r in caplog.records if r.levelno == logging.INFO and "after a temporary refusal" in r.getMessage()
        ]
        assert len(warned) == 2
        assert len(cleared) == 1

    async def test_a_failed_reply_with_no_code_is_temporary(self, caplog: pytest.LogCaptureFixture) -> None:
        """what an older registry sends mid-roll: no code, no tools named. waited, never raised.

        :return: none
        :rtype: None
        """
        older = RegistrationResponse(success=False, pod_id=_POD, error="invalid bootstrap token")
        registry = _ScriptedRegistry(older)
        server = _heartbeating_server(registry)
        await server.publish_registration(await_reply=True)
        with caplog.at_level(logging.WARNING, logger="threetears.agent.tools.server"):
            assert await server.wait_until_ready(timeout=0.3) is False
        named = [
            r for r in caplog.records if r.levelno == logging.WARNING and "invalid bootstrap token" in r.getMessage()
        ]
        assert len(named) == 1

    @pytest.mark.parametrize("code", sorted(_REGISTRY_FINAL))
    async def test_each_final_code_still_raises_at_once(self, code: str) -> None:
        """a final verdict is not waited out.

        :param code: the refusal code
        :ptype code: str
        :return: none
        :rtype: None
        """
        registry = _ScriptedRegistry(
            RegistrationResponse(success=False, pod_id=_POD, refused_tools=[_refusal(code=code)], error_code=code)
        )
        server = _heartbeating_server(registry)
        await server.publish_registration(await_reply=True)
        with pytest.raises(ToolRegistrationRefused, match=code):
            await server.wait_until_ready(timeout=5.0)

    async def test_a_manifest_refused_whole_with_a_final_code_raises(self) -> None:
        """a reply-level code with no tools named still names every tool the manifest offered.

        :return: none
        :rtype: None
        """
        registry = _ScriptedRegistry(
            RegistrationResponse(
                success=False, pod_id=_POD, error="tools[0].name is empty", error_code="INVALID_MANIFEST"
            )
        )
        server = _heartbeating_server(registry)
        await server.publish_registration(await_reply=True)
        with pytest.raises(ToolRegistrationRefused, match="INVALID_MANIFEST") as excinfo:
            await server.wait_until_ready(timeout=5.0)
        assert [r.name for r in excinfo.value.refused] == ["threetears.calculator"]

    async def test_a_temporary_refusal_of_a_new_tool_does_not_raise(self) -> None:
        """register_tool on a serving pod raises only on a final refusal.

        :return: none
        :rtype: None
        """
        server = _server()
        server._nc = _replying(  # noqa: SLF001
            RegistrationResponse(
                success=True,
                pod_id=_POD,
                refused_tools=[_refusal("threetears.dictionary", "OWNERSHIP_GRAPH_UNAVAILABLE")],
            )
        )
        server._ready_event.set()  # noqa: SLF001
        await server.register_tool(_Tool("threetears.dictionary"))
        assert [r.code for r in server.refused_tools] == ["OWNERSHIP_GRAPH_UNAVAILABLE"]


class TestAnOlderRegistrysDiscovery:
    """an older registry answers discovery with no ``requester_copy_status`` at all."""

    async def test_readiness_falls_back_to_the_tools_status(self, caplog: pytest.LogCaptureFixture) -> None:
        """the older registry's own answer decides, and the pod says it is reading one.

        :return: none
        :rtype: None
        """
        registry = _ScriptedRegistry(_ADMITTED, copy_status_field=False)
        server = _heartbeating_server(registry)
        await server.publish_registration(await_reply=True)
        with caplog.at_level(logging.WARNING, logger="threetears.agent.tools.server"):
            assert await server.wait_until_ready(timeout=2.0) is True
        assert any("older" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)

    async def test_an_unavailable_tool_on_an_older_registry_is_not_ready_and_does_not_raise(self) -> None:
        """not ready is False at the timeout, for the caller to retry; nothing crashes.

        :return: none
        :rtype: None
        """
        registry = _ScriptedRegistry(
            RegistrationResponse(success=False, pod_id=_POD, error="invalid bootstrap token"),
            copy_status_field=False,
        )
        server = _heartbeating_server(registry)
        await server.publish_registration(await_reply=True)
        assert await server.wait_until_ready(timeout=0.3) is False
