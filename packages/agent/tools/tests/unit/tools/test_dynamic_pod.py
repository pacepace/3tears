"""lifecycle tests for the generic :class:`DynamicToolPod` base.

exercises register / deregister / publish / stop over a fake
:class:`ToolServer` (``_FakeToolServer``) so no live NATS is required.
the fake subclasses :class:`ToolServer` -- that subclass declaration is
its fake-protocol-parity declaration (mypy enforces the method surface).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.dynamic_pod import BuiltSpec, DynamicToolPod
from threetears.agent.tools.server import ToolServer


# --- test tools ---


class _StubTool(TearsTool):
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


class _StubResource:
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


class _FakeToolServer(ToolServer):
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

    async def publish_registration(self) -> None:
        """record a manifest publish.

        :return: nothing
        :rtype: None
        """
        self.publish_count += 1
        self.published_tool_counts.append(len(self.registered))

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


class _StubSpec:
    """spec carrying a key, tools, and an optional resource."""

    def __init__(
        self,
        key: str,
        tool_count: int = 2,
        with_resource: bool = True,
        *,
        fail_close: bool = False,
        built_key: str | None = None,
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
        """
        self.key = key
        self.tool_count = tool_count
        self.resource: _StubResource | None = _StubResource(fail_close=fail_close) if with_resource else None
        self.built_key = built_key if built_key is not None else key


class _StubPod(DynamicToolPod[_StubSpec]):
    """concrete pod over a fake server with injectable specs."""

    def __init__(self, specs: list[_StubSpec], fake_server: _FakeToolServer) -> None:
        """initialize the stub pod.

        :param specs: specs returned by :meth:`load_specs`
        :ptype specs: list[_StubSpec]
        :param fake_server: fake tool server returned by :meth:`build_tool_server`
        :ptype fake_server: _FakeToolServer
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

    async def load_specs(self) -> list[_StubSpec]:
        """return the injected specs.

        :return: specs to build tools for
        :rtype: list[_StubSpec]
        """
        return list(self._specs)

    def spec_key(self, spec: _StubSpec) -> str:
        """return the stub spec's key.

        :param spec: the spec
        :ptype spec: _StubSpec
        :return: its key
        :rtype: str
        """
        return spec.key

    async def build_tools(self, spec: _StubSpec) -> BuiltSpec:
        """build stub tools for ``spec``.

        :param spec: spec to build tools for
        :ptype spec: _StubSpec
        :return: built spec
        :rtype: BuiltSpec
        """
        self.events.append(f"build:{spec.key}")
        tools: list[TearsTool] = [_StubTool(f"{spec.key}.tool{i}") for i in range(spec.tool_count)]
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


# --- tests ---


@pytest.mark.asyncio
async def test_start_registers_all_tools_and_spawns_one_serve() -> None:
    """start registers every spec's tools and spawns exactly one serve task."""
    fake = _FakeToolServer()
    pod = _StubPod([_StubSpec("ds_a"), _StubSpec("ds_b")], fake)

    await pod.start()
    # let the spawned serve task reach its first await so serve() runs
    await asyncio.sleep(0)

    assert len(fake.registered) == 4
    assert fake.serve_count == 1
    assert pod.on_started_calls == 1

    await pod.stop()


@pytest.mark.asyncio
async def test_start_with_no_specs_spawns_no_serve() -> None:
    """a pod whose load_specs returns [] starts with no serve task."""
    fake = _FakeToolServer()
    pod = _StubPod([], fake)

    await pod.start()

    assert fake.serve_count == 0
    assert fake.tools_count == 0

    await pod.stop()


@pytest.mark.asyncio
async def test_register_spec_while_disconnected_does_not_publish() -> None:
    """registering while is_connected is False must not publish."""
    fake = _FakeToolServer()
    fake.set_connected(False)
    pod = _StubPod([], fake)
    await pod.start()

    await pod.register_spec(_StubSpec("ds_late"))

    assert len(fake.registered) == 2
    assert fake.publish_count == 0

    await pod.stop()


@pytest.mark.asyncio
async def test_register_spec_on_a_serving_pod_publishes_once() -> None:
    """registering on a pod whose serve loop is bound registers tools and publishes once."""
    fake = _FakeToolServer()
    pod = _StubPod([_StubSpec("ds_first", tool_count=1)], fake)
    await pod.start()
    await asyncio.sleep(0)
    fake.set_connected(True)

    await pod.register_spec(_StubSpec("ds_live", tool_count=2))

    assert len(fake.registered) == 3
    assert fake.publish_count == 1

    await pod.stop()


@pytest.mark.asyncio
async def test_register_spec_that_starts_the_serve_loop_leaves_the_publish_to_it() -> None:
    """a spec that gives an empty pod its first tools must not publish ahead of the loop.

    the registry probes a pod the moment a manifest names a new endpoint, and does not probe
    an endpoint it already holds. a publish issued before the serve loop has bound the probe
    subject therefore loses the probe AND stops the loop's own publish from probing again, so
    the tools wait a whole heartbeat for promotion. the loop publishes the current manifest,
    new tools included, as soon as its subjects are bound.
    """
    fake = _FakeToolServer()
    pod = _StubPod([], fake)
    await pod.start()
    fake.set_connected(True)

    await pod.register_spec(_StubSpec("ds_first", tool_count=2))

    assert fake.publish_count == 0
    await asyncio.sleep(0)
    assert fake.serve_count == 1

    await pod.stop()


@pytest.mark.asyncio
async def test_register_spec_that_builds_no_tools_publishes_nothing() -> None:
    """a spec whose build failed changes nothing a manifest carries, so none is published."""
    fake = _FakeToolServer()
    pod = _StubPod([_StubSpec("ds_first", tool_count=1)], fake)
    await pod.start()
    await asyncio.sleep(0)
    fake.set_connected(True)

    await pod.register_spec(_StubSpec("ds_broken", tool_count=0))

    assert fake.publish_count == 0

    await pod.stop()


@pytest.mark.asyncio
async def test_deregister_spec_unregisters_closes_and_publishes() -> None:
    """deregister removes tools by mcp_name, closes resource once, publishes once, returns True."""
    fake = _FakeToolServer()
    spec = _StubSpec("ds_x", tool_count=2)
    resource = spec.resource
    assert resource is not None
    pod = _StubPod([spec], fake)
    await pod.start()
    fake.set_connected(True)

    result = await pod.deregister_spec("ds_x")

    assert result is True
    assert fake.unregistered == ["ds_x.tool0", "ds_x.tool1"]
    assert fake.registered == []
    assert resource.close_count == 1
    assert fake.publish_count == 1

    await pod.stop()


async def _serving(pod: _StubPod, fake: _FakeToolServer) -> None:
    """start the pod and let its serve loop bind, as a live pod has.

    :param pod: the pod
    :ptype pod: _StubPod
    :param fake: its server
    :ptype fake: _FakeToolServer
    :return: nothing
    :rtype: None
    """
    await pod.start()
    await asyncio.sleep(0)
    fake.set_connected(True)


@pytest.mark.asyncio
async def test_a_rebuilt_spec_is_announced_once_and_never_as_an_empty_manifest() -> None:
    """a refreshed spec swaps its tools with ONE publish.

    refreshing as deregister-then-register published the reduced manifest in between, and for a
    pod whose only tools are that spec's it was empty -- which the registry refuses, moments
    before the real one lands, on every credential refresh.
    """
    fake = _FakeToolServer()
    old = _StubSpec("ds_only", tool_count=2)
    old_resource = old.resource
    assert old_resource is not None
    pod = _StubPod([old], fake)
    await _serving(pod, fake)

    await pod.register_spec(_StubSpec("ds_only", tool_count=3))

    assert fake.published_tool_counts == [3]
    assert fake.unregistered == ["ds_only.tool0", "ds_only.tool1"]
    assert len(fake.registered) == 3
    assert old_resource.close_count == 1

    await pod.stop()


@pytest.mark.asyncio
async def test_a_rebuild_closes_the_old_resource_before_building_the_new_one() -> None:
    """a rebuild never holds two resources: a driver's pool counts against the connection limit."""
    fake = _FakeToolServer()
    pod = _StubPod([_StubSpec("ds_only", tool_count=1)], fake)
    await _serving(pod, fake)
    pod.events.clear()

    await pod.register_spec(_StubSpec("ds_only", tool_count=1))

    assert pod.events == ["close", "build:ds_only"]

    await pod.stop()


@pytest.mark.asyncio
async def test_a_failed_close_does_not_cost_the_rebuilt_tools() -> None:
    """the old resource's close failing is logged; the rebuilt spec is still registered and announced."""
    fake = _FakeToolServer()
    pod = _StubPod([_StubSpec("ds_only", tool_count=1, fail_close=True)], fake)
    await _serving(pod, fake)
    rebuilt = _StubSpec("ds_only", tool_count=2)

    await pod.register_spec(rebuilt)

    assert len(fake.registered) == 2
    assert fake.published_tool_counts == [2]
    await pod.stop()
    # the rebuilt resource was tracked, so stop() reached it.
    assert rebuilt.resource is not None
    assert rebuilt.resource.close_count == 1


@pytest.mark.asyncio
async def test_a_rebuild_that_now_builds_nothing_announces_the_reduced_manifest() -> None:
    """losing a spec's tools changes the manifest, so it is published -- unlike a first build of none."""
    fake = _FakeToolServer()
    pod = _StubPod([_StubSpec("ds_a", tool_count=1), _StubSpec("ds_b", tool_count=2)], fake)
    await _serving(pod, fake)

    await pod.register_spec(_StubSpec("ds_b", tool_count=0))

    assert fake.published_tool_counts == [1]

    await pod.stop()


@pytest.mark.asyncio
async def test_a_new_key_is_registered_without_forgetting_anything() -> None:
    fake = _FakeToolServer()
    pod = _StubPod([_StubSpec("ds_a", tool_count=1)], fake)
    await _serving(pod, fake)

    await pod.register_spec(_StubSpec("ds_new", tool_count=2))

    assert fake.published_tool_counts == [3]
    assert fake.unregistered == []

    await pod.stop()


@pytest.mark.asyncio
async def test_a_build_that_reports_another_key_is_refused() -> None:
    """spec_key and the built key must agree, or a rebuild forgets one spec and replaces another."""
    fake = _FakeToolServer()
    pod = _StubPod([], fake)
    await _serving(pod, fake)

    with pytest.raises(ValueError, match="spec_key"):
        await pod.register_spec(_StubSpec("ds_a", built_key="ds_b"))

    await pod.stop()


@pytest.mark.asyncio
async def test_deregister_unknown_key_returns_false_and_no_publish() -> None:
    """deregistering an unknown key returns False and publishes nothing."""
    fake = _FakeToolServer()
    pod = _StubPod([], fake)
    await pod.start()
    fake.set_connected(True)

    result = await pod.deregister_spec("does-not-exist")

    assert result is False
    assert fake.publish_count == 0

    await pod.stop()


@pytest.mark.asyncio
async def test_stop_shuts_down_cancels_and_closes_resources() -> None:
    """stop shuts the server, cancels serve, closes every tracked resource."""
    fake = _FakeToolServer()
    spec_a = _StubSpec("ds_a")
    spec_b = _StubSpec("ds_b")
    res_a = spec_a.resource
    res_b = spec_b.resource
    assert res_a is not None and res_b is not None
    pod = _StubPod([spec_a, spec_b], fake)
    await pod.start()

    await pod.stop()

    assert fake.shutdown_count == 1
    assert res_a.close_count == 1
    assert res_b.close_count == 1


@pytest.mark.asyncio
async def test_second_stop_is_noop() -> None:
    """a second stop is a no-op (no extra shutdown / no extra close)."""
    fake = _FakeToolServer()
    spec = _StubSpec("ds_a")
    resource = spec.resource
    assert resource is not None
    pod = _StubPod([spec], fake)
    await pod.start()

    await pod.stop()
    await pod.stop()

    assert fake.shutdown_count == 1
    assert resource.close_count == 1


@pytest.mark.asyncio
async def test_register_spec_before_serve_connects_is_safe() -> None:
    """register_spec is safe to call before serve connects (no publish, tools kept)."""
    fake = _FakeToolServer()
    pod = _StubPod([], fake)
    await pod.start()

    # never connected: is_connected stays False
    await pod.register_spec(_StubSpec("ds_pre", tool_count=1))

    assert len(fake.registered) == 1
    assert fake.publish_count == 0

    await pod.stop()
