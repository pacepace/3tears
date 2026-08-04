"""a pod that starts empty must serve once its first spec arrives.

THE BUG THIS CLOSES. :meth:`DynamicToolPod.start` spawned the serve loop only
when the server already had tools:

    if server.tools_count > 0:
        self._serve_task = spawn_background(server.serve(), ...)

``serve()`` is what subscribes to the pod's call AND probe subjects. A pod that
starts empty therefore never subscribes -- and nothing ever revisits that
decision, so it stays unreachable for the rest of the process's life no matter
how many specs it later gains.

Measured on a live stack. The Hub's dataset pod starts before any
``access_mode='build'`` datasource exists, so:

    dynamic tool pod started with no tools: pod_id=019fc549-abda-...

An operator then created the build datasource through the admin website. The pod
built its eight ``ripple.audience_*`` tools and published its manifest, and the
Hub said so:

    dataset tools registered: datasource=influencers-build pod_id=019fc549-abda-...

That message was false. The registry accepted the registration and then could
not reach the pod:

    tool pod reachability probe failed or reply was malformed; endpoints remain
    pending  probe_subject=aibots.tools.probe.019fc549-abda-...
    error="no responders for subject"

so the tools stayed PENDING, never became available, and the agent -- whose RBAC
grants were all correctly materialized -- discovered 17 tools and none of the
eight. The failure was one WARNING in a different container, two minutes after a
success message in this one.

This is the ordinary steady-state path, not an edge case: adding a datasource to
a running cluster is what the runbooks describe as the normal operation.
"""

from __future__ import annotations

import asyncio

import pytest

from unit.tools.test_dynamic_pod import (
    _FakeToolServer,
    _StubPod,
    _StubSpec,
)


def _spec(key: str) -> _StubSpec:
    """build a spec carrying one tool.

    :param key: spec key
    :ptype key: str
    :return: the spec
    :rtype: _StubSpec
    """
    return _StubSpec(key=key, tool_count=1)


class TestServesWhenFirstSpecArrives:
    """an empty pod is not a permanently deaf pod."""

    @pytest.mark.asyncio
    async def test_a_pod_that_starts_empty_serves_once_a_spec_registers(self) -> None:
        """without this the pod never subscribes and is unreachable forever."""
        server = _FakeToolServer()
        pod = _StubPod(specs=[], fake_server=server)
        await pod.start()
        await asyncio.sleep(0)
        assert server.serve_count == 0, "nothing to serve yet"

        server.set_connected(True)
        await pod.register_spec(_spec("influencers-build"))

        await asyncio.sleep(0)
        assert server.serve_count == 1, (
            "a pod that gained its first tool must start serving; otherwise its "
            "probe subject has no responders and the registry leaves every tool pending"
        )
        await pod.stop()

    @pytest.mark.asyncio
    async def test_serving_is_not_started_twice(self) -> None:
        """a second spec must not spawn a second serve loop.

        Two serve loops on one subject is a duplicate-delivery bug, and the
        cheapest way to introduce one is to re-check ``tools_count > 0`` on
        every registration.
        """
        server = _FakeToolServer()
        pod = _StubPod(specs=[], fake_server=server)
        await pod.start()
        server.set_connected(True)

        await pod.register_spec(_spec("one"))
        await pod.register_spec(_spec("two"))

        await asyncio.sleep(0)
        assert server.serve_count == 1
        await pod.stop()

    @pytest.mark.asyncio
    async def test_a_pod_that_starts_with_tools_still_serves_once(self) -> None:
        """the pre-existing path is unchanged."""
        server = _FakeToolServer()
        pod = _StubPod(specs=[_spec("central-reporting")], fake_server=server)

        await pod.start()

        await asyncio.sleep(0)
        assert server.serve_count == 1
        await pod.stop()

    @pytest.mark.asyncio
    async def test_a_spec_that_builds_no_tools_does_not_start_serving(self) -> None:
        """serving is gated on having something to serve, not on being asked."""
        server = _FakeToolServer()
        pod = _StubPod(specs=[], fake_server=server)
        await pod.start()
        server.set_connected(True)

        await pod.register_spec(_StubSpec(key="empty", tool_count=0))

        await asyncio.sleep(0)
        assert server.serve_count == 0
        await pod.stop()
