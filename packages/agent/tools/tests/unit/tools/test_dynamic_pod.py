"""lifecycle tests for the generic :class:`DynamicToolPod` base.

exercises register / deregister / publish / stop over a fake
:class:`ToolServer` (``FakeToolServer``) so no live NATS is required.
the fake subclasses :class:`ToolServer` -- that subclass declaration is
its fake-protocol-parity declaration (mypy enforces the method surface).
"""

from __future__ import annotations

import asyncio

import pytest

from threetears.agent.tools.dynamic_pod import BuiltSpec, DynamicToolPod
from threetears.agent.tools.server import RefusedTool, ToolRegistrationRefused

from packages.agent.tools.tests.unit.tools.dynamic_pod_fakes import (
    FakeToolServer,
    StubPod,
    StubSpec,
    StubTool,
)

# --- tests ---


@pytest.mark.asyncio
async def test_start_registers_all_tools_and_spawns_one_serve() -> None:
    """start registers every spec's tools and spawns exactly one serve task."""
    fake = FakeToolServer()
    pod = StubPod([StubSpec("ds_a"), StubSpec("ds_b")], fake)

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
    fake = FakeToolServer()
    pod = StubPod([], fake)

    await pod.start()

    assert fake.serve_count == 0
    assert fake.tools_count == 0

    await pod.stop()


@pytest.mark.asyncio
async def test_register_spec_while_disconnected_does_not_publish() -> None:
    """registering while is_connected is False must not publish."""
    fake = FakeToolServer()
    fake.set_connected(False)
    pod = StubPod([], fake)
    await pod.start()

    await pod.register_spec(StubSpec("ds_late"))

    assert len(fake.registered) == 2
    assert fake.publish_count == 0

    await pod.stop()


@pytest.mark.asyncio
async def test_register_spec_on_a_serving_pod_publishes_once() -> None:
    """registering on a pod whose serve loop is bound registers tools and publishes once."""
    fake = FakeToolServer()
    pod = StubPod([StubSpec("ds_first", tool_count=1)], fake)
    await pod.start()
    await asyncio.sleep(0)
    fake.set_connected(True)

    await pod.register_spec(StubSpec("ds_live", tool_count=2))

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
    fake = FakeToolServer()
    pod = StubPod([], fake)
    await pod.start()
    fake.set_connected(True)

    await pod.register_spec(StubSpec("ds_first", tool_count=2))

    assert fake.publish_count == 0
    await asyncio.sleep(0)
    assert fake.serve_count == 1

    await pod.stop()


@pytest.mark.asyncio
async def test_register_spec_that_builds_no_tools_publishes_nothing() -> None:
    """a spec whose build failed changes nothing a manifest carries, so none is published."""
    fake = FakeToolServer()
    pod = StubPod([StubSpec("ds_first", tool_count=1)], fake)
    await pod.start()
    await asyncio.sleep(0)
    fake.set_connected(True)

    await pod.register_spec(StubSpec("ds_broken", tool_count=0))

    assert fake.publish_count == 0

    await pod.stop()


@pytest.mark.asyncio
async def test_deregister_spec_unregisters_closes_and_publishes() -> None:
    """deregister removes tools by mcp_name, closes resource once, publishes once, returns True."""
    fake = FakeToolServer()
    spec = StubSpec("ds_x", tool_count=2)
    resource = spec.resource
    assert resource is not None
    pod = StubPod([spec], fake)
    await pod.start()
    fake.set_connected(True)

    result = await pod.deregister_spec("ds_x")

    assert result is True
    assert fake.unregistered == ["ds_x.tool0", "ds_x.tool1"]
    assert fake.registered == []
    assert resource.close_count == 1
    assert fake.publish_count == 1

    await pod.stop()


async def _serving(pod: StubPod, fake: FakeToolServer) -> None:
    """start the pod and let its serve loop bind, as a live pod has.

    :param pod: the pod
    :ptype pod: StubPod
    :param fake: its server
    :ptype fake: FakeToolServer
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
    fake = FakeToolServer()
    old = StubSpec("ds_only", tool_count=2)
    old_resource = old.resource
    assert old_resource is not None
    pod = StubPod([old], fake)
    await _serving(pod, fake)

    await pod.register_spec(StubSpec("ds_only", tool_count=3))

    assert fake.published_tool_counts == [3]
    assert fake.unregistered == ["ds_only.tool0", "ds_only.tool1"]
    assert len(fake.registered) == 3
    assert old_resource.close_count == 1

    await pod.stop()


@pytest.mark.asyncio
async def test_a_rebuild_closes_the_old_resource_before_building_the_new_one() -> None:
    """a rebuild never holds two resources: a driver's pool counts against the connection limit."""
    fake = FakeToolServer()
    pod = StubPod([StubSpec("ds_only", tool_count=1)], fake)
    await _serving(pod, fake)
    pod.events.clear()

    await pod.register_spec(StubSpec("ds_only", tool_count=1))

    assert pod.events == ["close", "build:ds_only"]

    await pod.stop()


@pytest.mark.asyncio
async def test_a_failed_close_does_not_cost_the_rebuilt_tools() -> None:
    """the old resource's close failing is logged; the rebuilt spec is still registered and announced."""
    fake = FakeToolServer()
    pod = StubPod([StubSpec("ds_only", tool_count=1, fail_close=True)], fake)
    await _serving(pod, fake)
    rebuilt = StubSpec("ds_only", tool_count=2)

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
    fake = FakeToolServer()
    pod = StubPod([StubSpec("ds_a", tool_count=1), StubSpec("ds_b", tool_count=2)], fake)
    await _serving(pod, fake)

    await pod.register_spec(StubSpec("ds_b", tool_count=0))

    assert fake.published_tool_counts == [1]

    await pod.stop()


@pytest.mark.asyncio
async def test_a_new_key_is_registered_without_forgetting_anything() -> None:
    fake = FakeToolServer()
    pod = StubPod([StubSpec("ds_a", tool_count=1)], fake)
    await _serving(pod, fake)

    await pod.register_spec(StubSpec("ds_new", tool_count=2))

    assert fake.published_tool_counts == [3]
    assert fake.unregistered == []

    await pod.stop()


@pytest.mark.asyncio
async def test_a_build_that_reports_another_key_is_refused() -> None:
    """spec_key and the built key must agree, or a rebuild forgets one spec and replaces another."""
    fake = FakeToolServer()
    pod = StubPod([], fake)
    await _serving(pod, fake)

    mismatched = StubSpec("ds_a", built_key="ds_b")

    with pytest.raises(ValueError, match="spec_key"):
        await pod.register_spec(mismatched)

    # nothing tracks what the build returned, so the pod closes it rather than leak it.
    assert mismatched.resource is not None
    assert mismatched.resource.close_count == 1

    await pod.stop()


@pytest.mark.asyncio
async def test_start_refuses_a_build_that_reports_another_key() -> None:
    """start() goes through the same replace as register_spec, key check and all."""
    fake = FakeToolServer()
    mismatched = StubSpec("ds_a", built_key="ds_b")
    pod = StubPod([mismatched], fake)

    with pytest.raises(ValueError, match="spec_key"):
        await pod.start()

    assert mismatched.resource is not None
    assert mismatched.resource.close_count == 1
    assert fake.registered == []

    await pod.stop()


@pytest.mark.asyncio
async def test_a_rebuild_that_raises_forgets_the_spec_and_says_so() -> None:
    """the old tools are gone -- a narrowed spec must not keep its wider ones -- and the registry is told."""
    fake = FakeToolServer()
    old = StubSpec("ds_only", tool_count=2)
    pod = StubPod([old], fake)
    await _serving(pod, fake)

    with pytest.raises(ValueError, match="server url"):
        await pod.register_spec(StubSpec("ds_only", fail_build=True))

    assert fake.registered == []
    assert fake.published_tool_counts == [0]
    assert old.resource is not None
    assert old.resource.close_count == 1

    await pod.stop()


@pytest.mark.asyncio
async def test_a_retried_start_closes_what_the_failed_one_built() -> None:
    """a host retries start() after a build fails part-way; the retry must not leak the first attempt."""
    fake = FakeToolServer()
    first_a = StubSpec("ds_a", tool_count=1)
    specs = [first_a, StubSpec("ds_b", tool_count=1, fail_build=True)]
    pod = StubPod(specs, fake)

    with pytest.raises(ValueError, match="server url"):
        await pod.start()

    second_a = StubSpec("ds_a", tool_count=1)
    specs[:] = [second_a, StubSpec("ds_b", tool_count=1)]
    await pod.start()

    assert first_a.resource is not None and second_a.resource is not None
    assert first_a.resource.close_count == 1
    assert second_a.resource.close_count == 0
    assert sorted(t.mcp_name() for t in fake.registered) == ["ds_a.tool0", "ds_b.tool0"]

    await pod.stop()


@pytest.mark.asyncio
async def test_a_deregister_racing_a_rebuild_waits_for_it() -> None:
    """the deregister runs after the rebuild it raced, so it removes what the rebuild registered."""
    fake = FakeToolServer()
    pod = StubPod([], fake)
    await _serving(pod, fake)
    gate = asyncio.Event()
    rebuilt = StubSpec("ds_race", tool_count=1, build_gate=gate)

    racing = asyncio.gather(pod.register_spec(rebuilt), pod.deregister_spec("ds_race"))
    await asyncio.sleep(0)
    gate.set()
    await racing

    assert rebuilt.resource is not None
    assert rebuilt.resource.close_count == 1
    assert fake.registered == []

    await pod.stop()


@pytest.mark.asyncio
async def test_overlapping_rebuilds_of_one_spec_leave_one_resource() -> None:
    """two rebuilds racing on one key: the first's resource is closed by the second, never leaked."""
    fake = FakeToolServer()
    pod = StubPod([], fake)
    await _serving(pod, fake)
    gate = asyncio.Event()
    first = StubSpec("ds_race", tool_count=1, build_gate=gate)
    second = StubSpec("ds_race", tool_count=1)

    racing = asyncio.gather(pod.register_spec(first), pod.register_spec(second))
    await asyncio.sleep(0)
    gate.set()
    await racing

    assert first.resource is not None and second.resource is not None
    assert first.resource.close_count == 1
    assert second.resource.close_count == 0
    assert [t.mcp_name() for t in fake.registered] == ["ds_race.tool0"]

    await pod.stop()
    assert second.resource.close_count == 1


@pytest.mark.asyncio
async def test_deregister_unknown_key_returns_false_and_no_publish() -> None:
    """deregistering an unknown key returns False and publishes nothing."""
    fake = FakeToolServer()
    pod = StubPod([], fake)
    await pod.start()
    fake.set_connected(True)

    result = await pod.deregister_spec("does-not-exist")

    assert result is False
    assert fake.publish_count == 0

    await pod.stop()


@pytest.mark.asyncio
async def test_stop_shuts_down_cancels_and_closes_resources() -> None:
    """stop shuts the server, cancels serve, closes every tracked resource."""
    fake = FakeToolServer()
    spec_a = StubSpec("ds_a")
    spec_b = StubSpec("ds_b")
    res_a = spec_a.resource
    res_b = spec_b.resource
    assert res_a is not None and res_b is not None
    pod = StubPod([spec_a, spec_b], fake)
    await pod.start()

    await pod.stop()

    assert fake.shutdown_count == 1
    assert res_a.close_count == 1
    assert res_b.close_count == 1


@pytest.mark.asyncio
async def test_second_stop_is_noop() -> None:
    """a second stop is a no-op (no extra shutdown / no extra close)."""
    fake = FakeToolServer()
    spec = StubSpec("ds_a")
    resource = spec.resource
    assert resource is not None
    pod = StubPod([spec], fake)
    await pod.start()

    await pod.stop()
    await pod.stop()

    assert fake.shutdown_count == 1
    assert resource.close_count == 1


@pytest.mark.asyncio
async def test_register_spec_before_serve_connects_is_safe() -> None:
    """register_spec is safe to call before serve connects (no publish, tools kept)."""
    fake = FakeToolServer()
    pod = StubPod([], fake)
    await pod.start()

    # never connected: is_connected stays False
    await pod.register_spec(StubSpec("ds_pre", tool_count=1))

    assert len(fake.registered) == 1
    assert fake.publish_count == 0

    await pod.stop()


@pytest.mark.asyncio
async def test_a_live_registration_awaits_the_registry_and_raises_on_refusal() -> None:
    """a spec whose tool the registry refuses fails its register_spec, naming the tool."""
    fake = FakeToolServer()
    pod = StubPod([StubSpec("ds_first", tool_count=1)], fake)
    await pod.start()
    await asyncio.sleep(0)
    fake.set_connected(True)
    fake.next_refusals = [
        RefusedTool(name="ds_live.tool1", version="1.0", code="OWNED_ELSEWHERE", reason="another pod owns it")
    ]

    with pytest.raises(ToolRegistrationRefused) as excinfo:
        await pod.register_spec(StubSpec("ds_live", tool_count=2))

    assert [r.name for r in excinfo.value.refused] == ["ds_live.tool1"]
    assert fake.awaited_replies[-1] is True

    await pod.stop()


@pytest.mark.asyncio
async def test_a_temporary_refusal_does_not_fail_the_registration() -> None:
    """a registry that could not read its ownership graph retries on the next heartbeat.

    the spec's tools stay registered on the server and the heartbeat re-offers them, so failing
    the registration would fail a spec the registry is about to admit.
    """
    fake = FakeToolServer()
    pod = StubPod([StubSpec("ds_first", tool_count=1)], fake)
    await pod.start()
    await asyncio.sleep(0)
    fake.set_connected(True)
    fake.next_refusals = [
        RefusedTool(name="ds_live.tool0", version="1.0", code="OWNERSHIP_GRAPH_UNAVAILABLE", reason="graph unreadable")
    ]

    await pod.register_spec(StubSpec("ds_live", tool_count=1))

    assert fake.awaited_replies[-1] is True
    await pod.stop()


@pytest.mark.asyncio
async def test_a_refusal_of_another_specs_tool_does_not_fail_this_spec() -> None:
    """only this spec's tools are this registration's business."""
    fake = FakeToolServer()
    pod = StubPod([StubSpec("ds_first", tool_count=1)], fake)
    await pod.start()
    await asyncio.sleep(0)
    fake.set_connected(True)
    fake.next_refusals = [RefusedTool(name="ds_first.tool0", version="1.0", code="OWNED_ELSEWHERE", reason="x")]

    await pod.register_spec(StubSpec("ds_live", tool_count=1))

    await pod.stop()


@pytest.mark.asyncio
async def test_the_pods_identity_token_rides_every_manifest() -> None:
    """a pod given an identity presents it, re-minted per manifest, on the default server."""
    from unittest.mock import AsyncMock

    minted: list[str] = []

    def _mint() -> str:
        minted.append(f"token-{len(minted)}")
        return minted[-1]

    class _BarePod(DynamicToolPod[StubSpec]):
        """a pod using the base's own ToolServer construction."""

        async def load_specs(self) -> list[StubSpec]:
            return []

        def spec_key(self, spec: StubSpec) -> str:
            return spec.key

        async def build_tools(self, spec: StubSpec) -> BuiltSpec:
            return BuiltSpec(key=spec.key, tools=[])

    nc = AsyncMock()
    pod = _BarePod(nats_url="", nats_client=nc, namespace="3tears", pod_id="hub-internal-pod", identity_token=_mint)
    server = pod.build_tool_server()
    server.register(StubTool("addrnorm.normalize"))

    await server.publish_registration()
    await server.publish_registration()

    tokens = [call.kwargs["message"].bootstrap_token for call in nc.publish.await_args_list]
    assert tokens == ["token-0", "token-1"]
