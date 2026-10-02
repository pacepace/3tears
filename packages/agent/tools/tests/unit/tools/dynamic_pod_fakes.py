"""the fakes the :class:`DynamicToolPod` lifecycle tests share.

a stub tool, a closeable resource, a fake :class:`ToolServer` that records register /
unregister / publish / shutdown, and a concrete pod over it with injectable specs. the
fake subclasses :class:`ToolServer` -- that subclass declaration is its
fake-protocol-parity declaration (mypy enforces the method surface).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.dynamic_pod import BuiltSpec, DynamicToolPod
from threetears.agent.tools.server import (
    RefusedTool,
    RegistrationResponse,
    ToolServer,
)


# --- test tools ---


class StubTool(TearsTool):
    """minimal TearsTool used to populate a spec's tool list."""

    def __init__(self, name: str, version: str = "1.0") -> None:
        """initialize stub tool.

        :param name: namespaced tool name
        :ptype name: str
        :param version: version string
        :ptype version: str
        """
        self._name = name
        self._version = version

    async def execute(self, **kwargs: Any) -> ToolResult:
        """echo arguments as success result.

        :param kwargs: tool input parameters
        :ptype kwargs: Any
        :return: success result
        :rtype: ToolResult
        """
        return ToolResult(success=True, content=json.dumps(kwargs))

    def mcp_schema(self) -> MCPToolDefinition:
        """return stub schema.

        :return: tool definition
        :rtype: MCPToolDefinition
        """
        return MCPToolDefinition(
            name=self._name,
            version=self._version,
            description="stub tool",
            input_schema={"type": "object", "properties": {}},
        )

    def mcp_name(self) -> str:
        """return namespaced tool name.

        :return: namespaced tool name
        :rtype: str
        """
        return self._name

    def mcp_version(self) -> str:
        """return version string.

        :return: version string
        :rtype: str
        """
        return self._version


class StubResource:
    """closeable resource that records how many times it was closed."""

    def __init__(self, *, fail_close: bool = False) -> None:
        """initialize the resource with a zeroed close counter.

        :param fail_close: whether :meth:`close` raises, as a driver whose pool close times out does
        :ptype fail_close: bool
        """
        self.close_count = 0
        self._fail_close = fail_close

    async def close(self) -> None:
        """record a close call.

        :return: nothing
        :rtype: None
        :raises TimeoutError: when built with ``fail_close``
        """
        self.close_count += 1
        if self._fail_close:
            raise TimeoutError("pool close timed out")


# --- fake tool server (parity via subclass declaration) ---


class FakeToolServer(ToolServer):
    """records register / unregister / publish / shutdown calls.

    subclasses :class:`ToolServer` purely as the fake-protocol-parity
    declaration; it does NOT call ``super().__init__`` and overrides the
    only surface the pod touches. ``serve`` blocks on an event so the
    spawned serve task stays alive until ``stop`` cancels it.
    """

    def __init__(self) -> None:
        """initialize the fake with empty call records."""
        self.registered: list[TearsTool] = []
        self.unregistered: list[str] = []
        self.publish_count = 0
        # how many tools each published manifest carried, in publish order.
        self.published_tool_counts: list[int] = []
        self.shutdown_count = 0
        self.serve_count = 0
        self._connected = False
        self._ready = False
        self._serve_gate = asyncio.Event()
        # whether each publish awaited the registry's reply, in publish order.
        self.awaited_replies: list[bool] = []
        # the refusals the next awaited reply names.
        self.next_refusals: list[RefusedTool] = []

    def set_connected(self, connected: bool) -> None:
        """flip the fake's connected state for publish-gating tests.

        :param connected: value returned by :attr:`is_connected`
        :ptype connected: bool
        :return: nothing
        :rtype: None
        """
        self._connected = connected

    def register(self, tool: TearsTool) -> None:
        """record a tool registration.

        :param tool: tool being registered
        :ptype tool: TearsTool
        :return: nothing
        :rtype: None
        """
        self.registered.append(tool)

    def unregister(self, mcp_name: str) -> bool:
        """remove every registered tool matching ``mcp_name``.

        :param mcp_name: namespaced tool name to remove
        :ptype mcp_name: str
        :return: true when at least one tool was removed
        :rtype: bool
        """
        self.unregistered.append(mcp_name)
        before = len(self.registered)
        self.registered = [t for t in self.registered if t.mcp_name() != mcp_name]
        return len(self.registered) < before

    async def publish_registration(self, *, await_reply: bool = False) -> RegistrationResponse | None:
        """record a manifest publish and, when awaited, answer with the scripted refusals.

        :param await_reply: whether the caller awaits the registry's reply
        :ptype await_reply: bool
        :return: the scripted reply when awaited, else ``None``
        :rtype: RegistrationResponse | None
        """
        self.publish_count += 1
        self.published_tool_counts.append(len(self.registered))
        self.awaited_replies.append(await_reply)
        reply: RegistrationResponse | None = None
        if await_reply:
            reply = RegistrationResponse(success=True, pod_id="pod-test", refused_tools=list(self.next_refusals))
        return reply

    async def shutdown(self) -> None:
        """record a shutdown and release the serve gate.

        :return: nothing
        :rtype: None
        """
        self.shutdown_count += 1
        self._serve_gate.set()

    async def serve(self) -> None:
        """report ready as the real loop does once its subjects are bound, then block.

        :return: nothing
        :rtype: None
        """
        self.serve_count += 1
        self._ready = True
        await self._serve_gate.wait()

    @property
    def is_ready(self) -> bool:
        """return whether the fake serve loop has reached its bound state.

        :return: ready state
        :rtype: bool
        """
        return self._ready

    @property
    def is_connected(self) -> bool:
        """return the fake's connected flag.

        :return: connected state
        :rtype: bool
        """
        return self._connected

    @property
    def tools_count(self) -> int:
        """return number of registered tools.

        :return: registered tool count
        :rtype: int
        """
        return len(self.registered)


# --- test pod ---


class StubSpec:
    """spec carrying a key, tools, and an optional resource."""

    def __init__(
        self,
        key: str,
        tool_count: int = 2,
        with_resource: bool = True,
        *,
        fail_close: bool = False,
        built_key: str | None = None,
        fail_build: bool = False,
        build_gate: asyncio.Event | None = None,
    ) -> None:
        """initialize the stub spec.

        :param key: spec key
        :ptype key: str
        :param tool_count: number of tools to build for this spec
        :ptype tool_count: int
        :param with_resource: whether the spec owns a closeable resource
        :ptype with_resource: bool
        :param fail_close: whether the spec's resource raises on close
        :ptype fail_close: bool
        :param built_key: the key the build reports, when it should disagree with ``key``
        :ptype built_key: str | None
        :param fail_build: whether building this spec raises, as a bad OpenAPI spec does
        :ptype fail_build: bool
        :param build_gate: when set, the build waits on it, so a test can overlap two builds
        :ptype build_gate: asyncio.Event | None
        """
        self.key = key
        self.tool_count = tool_count
        self.resource: StubResource | None = StubResource(fail_close=fail_close) if with_resource else None
        self.built_key = built_key if built_key is not None else key
        self.fail_build = fail_build
        self.build_gate = build_gate


class StubPod(DynamicToolPod[StubSpec]):
    """concrete pod over a fake server with injectable specs."""

    def __init__(self, specs: list[StubSpec], fake_server: FakeToolServer) -> None:
        """initialize the stub pod.

        :param specs: specs returned by :meth:`load_specs`
        :ptype specs: list[StubSpec]
        :param fake_server: fake tool server returned by :meth:`build_tool_server`
        :ptype fake_server: FakeToolServer
        """
        super().__init__(
            nats_url="nats://ignored",
            nats_client=object(),
            namespace="3tears",
            pod_id="pod-test",
        )
        self._specs = specs
        self._fake_server = fake_server
        self.on_started_calls = 0
        # builds and closes in the order they happened, so a test can pin the ordering.
        self.events: list[str] = []

    def build_tool_server(self) -> ToolServer:
        """return the injected fake server.

        :return: fake tool server
        :rtype: ToolServer
        """
        return self._fake_server

    async def load_specs(self) -> list[StubSpec]:
        """return the injected specs.

        :return: specs to build tools for
        :rtype: list[StubSpec]
        """
        return list(self._specs)

    def spec_key(self, spec: StubSpec) -> str:
        """return the stub spec's key.

        :param spec: the spec
        :ptype spec: StubSpec
        :return: its key
        :rtype: str
        """
        return spec.key

    async def build_tools(self, spec: StubSpec) -> BuiltSpec:
        """build stub tools for ``spec``.

        :param spec: spec to build tools for
        :ptype spec: StubSpec
        :return: built spec
        :rtype: BuiltSpec
        """
        self.events.append(f"build:{spec.key}")
        if spec.build_gate is not None:
            await spec.build_gate.wait()
        if spec.fail_build:
            raise ValueError("the spec has no server url")
        tools: list[TearsTool] = [StubTool(f"{spec.key}.tool{i}") for i in range(spec.tool_count)]
        return BuiltSpec(key=spec.built_key, tools=tools, resource=spec.resource)

    async def close_resource(self, resource: object) -> None:
        """record the close, then close as the base does.

        :param resource: resource to close
        :ptype resource: object
        :return: nothing
        :rtype: None
        """
        self.events.append("close")
        await super().close_resource(resource)

    async def on_started(self) -> None:
        """record on_started invocation.

        :return: nothing
        :rtype: None
        """
        self.on_started_calls += 1


__all__ = [
    "FakeToolServer",
    "StubPod",
    "StubResource",
    "StubSpec",
    "StubTool",
]
