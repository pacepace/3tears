"""unit tests for ``threetears.agent.tools.bootstrap.ToolServerBootstrap``."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import socket
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import Column, MetaData, String, Table

from threetears.agent.tools.bootstrap import (
    EX_CONFIG,
    EX_SOFTWARE,
    OWNER_PID_ENV,
    ToolPodConfigError,
    ToolPodShutdownError,
    ToolServerBootstrap,
    resolve_owner_pid,
)
from threetears.agent.tools.config import OWNER_POLL_INTERVAL_ENV, SHUTDOWN_TIMEOUT_ENV
from threetears.agent.tools.object_resolution_collection import (
    OBJECT_RESOLUTIONS_TABLE,
    ObjectResolutionCollection,
)
from threetears.core.coordination.replay_anchor import CollectionReplayAnchor
from threetears.core.testing.kv import FakeNatsClient
from threetears.nats import Principal, Subjects, kv_key_scope_for


def _collection_tables() -> MetaData:
    """one trivial table so a bootstrap can opt into the collection stack.

    :return: metadata carrying a single-column table
    :rtype: MetaData
    """
    metadata = MetaData()
    Table("widgets", metadata, Column("id", String(64), primary_key=True))
    return metadata


def _stack_nats_client() -> MagicMock:
    """a wrapper-client stand-in whose bucket bind and typed subscribe both succeed.

    :return: mock NATS client
    :rtype: MagicMock
    """
    client = MagicMock()
    client.raw = MagicMock()
    client.ensure_kv_bucket = AsyncMock(return_value=MagicMock())
    client.subscribe_typed = AsyncMock(return_value=MagicMock())
    client.unsubscribe = AsyncMock()
    return client


# parity-exempt: TearsTool subset shim covering the tool-server registration flow under unit test; full TearsTool surface includes mcp_schema/mcp_name/etc. but the test only exercises register+call
class _FakeToolServer:
    """minimal ToolServer stand-in for bootstrap behavioral tests.

    not a SLF001 violation surface: ``_serve_event`` is owned by
    this class and accessed only by its own methods plus the test
    that constructs it. tests below NEVER touch that field directly.
    """

    def __init__(self, pod_id: str = "01947100-0000-7000-8000-0000000000aa") -> None:
        self.tools_count = 2
        self.serve_called = False
        self.shutdown_called = False
        self.serve_event = asyncio.Event()
        self.pod_id = pod_id
        self.connected_callbacks: list[Any] = []
        self.object_resolution_cache: Any = None
        self.assertion_replay_anchor: Any = None

    def add_connected_callback(self, callback: Any) -> None:
        self.connected_callbacks.append(callback)

    def attach_object_resolution_cache(self, cache: Any) -> None:
        self.object_resolution_cache = cache

    def attach_assertion_replay_anchor(self, anchor: Any) -> None:
        self.assertion_replay_anchor = anchor

    async def serve(self) -> None:
        self.serve_called = True
        await self.serve_event.wait()

    async def shutdown(self) -> None:
        self.shutdown_called = True
        self.serve_event.set()

    def render_metrics(self) -> tuple[str, bytes]:
        return ("text/plain; version=0.0.4; charset=utf-8", b"")


class _ConcreteBootstrap(ToolServerBootstrap):
    """subclass used to drive ``run`` / ``run_async`` paths."""

    def __init__(
        self,
        *,
        server: _FakeToolServer,
        register_log: list[bool],
        collection_tables: MetaData | None = None,
    ) -> None:
        super().__init__("test-pod", collection_tables=collection_tables)
        self.server = server
        self.register_log = register_log

    async def build_server(self) -> Any:
        return self.server

    async def register_tools(self, server: Any) -> None:
        self.register_log.append(True)


class TestRunAsync:
    """``run_async`` builds, registers, and serves until shutdown."""

    async def test_lifecycle_completes_when_serve_returns(self) -> None:
        server = _FakeToolServer()
        register_log: list[bool] = []
        bootstrap = _ConcreteBootstrap(server=server, register_log=register_log)

        # release serve immediately so run_async returns
        server.serve_event.set()
        await bootstrap.run_async()

        assert register_log == [True]
        assert server.serve_called is True

    async def test_signal_handler_triggers_shutdown(self) -> None:
        server = _FakeToolServer()
        register_log: list[bool] = []
        bootstrap = _ConcreteBootstrap(server=server, register_log=register_log)

        async def trigger_signal_then_run() -> None:
            run_task = asyncio.create_task(bootstrap.run_async())
            await asyncio.sleep(0)  # let bootstrap install handlers
            await asyncio.sleep(0)
            # simulate the signal handler firing (do not raise actual SIGTERM)
            handler = bootstrap.make_signal_handler(server, "test")
            handler()
            await run_task

        await asyncio.wait_for(trigger_signal_then_run(), timeout=2.0)
        assert server.shutdown_called is True


class TestTheCollectionStackRidesTheLifecycle:
    """``coll-task-07c`` TP-04 / TP-06: the pod's tiers start and stop with the pod.

    The stack is built on a CONNECTED-callback rather than in ``run_async`` directly, because the
    connection does not exist until ``ToolServer.serve()`` opens it -- and it must be built BEFORE
    the pod subscribes its call subject and publishes its registration manifest, or the pod is
    discoverable while its own collections are still unwired.
    """

    async def test_a_pod_that_declares_no_tables_still_gets_the_runtime_stack(self) -> None:
        """the runtime's own collections do not wait for the host to declare one of its own.

        Every pod reads the shared bucket whether it declared tables or not: its proxy-assertion
        guard's replay anchor lives there. Gating the stack on host tables left every pod that
        declared none -- every SDK tool pod, and the built-in tool server -- with no anchor, so
        the first proxied call after each cold start was refused as a replay that never happened.
        """
        server = _FakeToolServer()
        server.serve_event.set()
        bootstrap = _ConcreteBootstrap(server=server, register_log=[])

        run_task = asyncio.create_task(bootstrap.run_async())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert len(server.connected_callbacks) == 1
        await server.connected_callbacks[0](_stack_nats_client())
        registry = bootstrap.collection_registry
        await run_task

        assert registry is not None
        assert isinstance(server.assertion_replay_anchor, CollectionReplayAnchor)
        assert isinstance(server.object_resolution_cache, ObjectResolutionCollection)

    async def test_a_bare_pods_anchor_answers_over_the_real_collection_path(self) -> None:
        """the anchor a pod with no tables is handed actually records and re-reads a birth time.

        Driven through the real collection over an in-memory KV rather than a mock, because the
        wiring existed for a release without ever running: no pod reached it, and the only test
        over it asserted the anchor's TYPE. The shared bucket is declared first, as the hub
        declares it -- a pod only ever binds it.
        """
        nats = FakeNatsClient()
        nats.ensure_kv_bucket = AsyncMock(return_value=MagicMock())  # type: ignore[method-assign]
        nats.subscribe_typed = AsyncMock(return_value=MagicMock())  # type: ignore[attr-defined]
        nats.unsubscribe = AsyncMock()  # type: ignore[method-assign]
        await nats.kv_bucket(name="collections", create_if_missing=True)
        server = _FakeToolServer(pod_id=str(uuid.uuid4()))
        server.serve_event.set()
        bootstrap = _ConcreteBootstrap(server=server, register_log=[])

        run_task = asyncio.create_task(bootstrap.run_async())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await server.connected_callbacks[0](nats)
        anchor = server.assertion_replay_anchor
        first = datetime.now(UTC)
        recorded = await anchor.first_existed("proxy_assertion_nonces", now=first)
        later = await anchor.first_existed("proxy_assertion_nonces", now=first + timedelta(minutes=5))
        await run_task

        assert recorded == first
        assert later == first, "a later reader must get the ledger's birth time, not its own clock"

    async def test_an_in_process_pod_gets_no_tool_pod_stack(self) -> None:
        """a pod running inside an agent process is not a tool-pod principal.

        It rides the agent's injected connection, authenticated as the AGENT, and its pod id
        is ``{agent_id}.{instance}`` rather than a ``tool_pods.id``. No tool-pod key scope can
        be derived from that id, and the connection's grant carries the agent's scope, not
        ``tool_pod-<hex>``. Building the stack for it raised inside the connected callback on
        every start, which exited the process and fed a supervisor restart loop.
        """
        server = _FakeToolServer(pod_id=Subjects.agent_inprocess_pod_id(uuid.uuid4(), uuid.uuid4()))
        server.serve_event.set()
        bootstrap = _ConcreteBootstrap(server=server, register_log=[])

        await bootstrap.run_async()

        assert server.serve_called is True
        assert server.connected_callbacks == []
        assert bootstrap.collection_registry is None
        assert bootstrap.object_resolutions is None
        assert server.assertion_replay_anchor is None

    async def test_an_in_process_pod_that_declares_tables_is_refused_before_it_serves(self) -> None:
        """declared tables need a tool-pod identity to scope them, and this pod has none.

        Refused at wiring with the terminal config type, so ``run`` exits ``EX_CONFIG`` once
        instead of the connected callback raising on every restart.
        """
        server = _FakeToolServer(pod_id=Subjects.agent_inprocess_pod_id(uuid.uuid4(), uuid.uuid4()))
        server.serve_event.set()
        bootstrap = _ConcreteBootstrap(
            server=server,
            register_log=[],
            collection_tables=_collection_tables(),
        )

        with pytest.raises(ToolPodConfigError) as caught:
            await bootstrap.run_async()

        assert caught.value.variable == "collection_tables"
        assert server.pod_id in str(caught.value)
        assert server.serve_called is False
        assert server.connected_callbacks == []

    async def test_the_runtime_collection_is_built_and_handed_to_the_server(self) -> None:
        """the stack carries a payload: the resolver's cache is wired without the host asking.

        A pod that had to construct this itself is a pod that will not, and the resolver
        then silently keeps its per-process dict -- a cache that looks live and is never
        shared with the replica beside it.
        """
        server = _FakeToolServer()
        server.serve_event.set()
        bootstrap = _ConcreteBootstrap(
            server=server,
            register_log=[],
            collection_tables=_collection_tables(),
        )

        run_task = asyncio.create_task(bootstrap.run_async())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await server.connected_callbacks[0](_stack_nats_client())
        collection = bootstrap.object_resolutions
        await run_task

        assert isinstance(collection, ObjectResolutionCollection)
        assert server.object_resolution_cache is collection
        # the pod's L1 tier carries the runtime's table, not only the host's own.
        registry = bootstrap.collection_registry
        assert registry is not None
        l1 = registry.get_l1_backend(OBJECT_RESOLUTIONS_TABLE)
        assert l1 is not None
        assert l1.has_table(OBJECT_RESOLUTIONS_TABLE) is True

    async def test_the_stack_is_built_when_the_connection_arrives(self) -> None:
        server = _FakeToolServer()
        server.serve_event.set()
        bootstrap = _ConcreteBootstrap(
            server=server,
            register_log=[],
            collection_tables=_collection_tables(),
        )
        client = _stack_nats_client()

        run_task = asyncio.create_task(bootstrap.run_async())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert len(server.connected_callbacks) == 1
        await server.connected_callbacks[0](client)
        await run_task

        client.ensure_kv_bucket.assert_awaited_once_with(name="collections", create_if_missing=False)

    async def test_the_stack_is_scoped_to_the_servers_own_pod_id(self) -> None:
        """the key side and the grant side must derive the scope from the SAME authenticated id.

        The pod id comes off the ``ToolServer`` the subclass built -- the same value the pod
        presents at connect and the auth callout pins as ``claims.sub`` -- rather than from a
        second piece of bootstrap configuration that could drift from it.
        """
        server = _FakeToolServer(pod_id="01947100-0000-7000-8000-0000000000cc")
        server.serve_event.set()
        bootstrap = _ConcreteBootstrap(
            server=server,
            register_log=[],
            collection_tables=_collection_tables(),
        )

        run_task = asyncio.create_task(bootstrap.run_async())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await server.connected_callbacks[0](_stack_nats_client())
        registry = bootstrap.collection_registry
        await run_task

        assert registry is not None
        assert registry.kv_key_scope == kv_key_scope_for(Principal.TOOL_POD, pod_id=server.pod_id)

    async def test_the_listener_is_stopped_when_serve_returns(self) -> None:
        """a pod that shut down without this left a subscription on a client it no longer owns."""
        server = _FakeToolServer()
        server.serve_event.set()
        bootstrap = _ConcreteBootstrap(
            server=server,
            register_log=[],
            collection_tables=_collection_tables(),
        )
        client = _stack_nats_client()

        run_task = asyncio.create_task(bootstrap.run_async())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await server.connected_callbacks[0](client)
        await run_task

        client.unsubscribe.assert_awaited_once()

    async def test_the_replay_anchor_is_wired_without_the_host_asking(self) -> None:
        """the proxy-assertion guard's durable first-existence record, wired here or nowhere.

        Without an anchor the guard cannot tell a bucket it never had from one it lost, so
        it applies its creation-time watermark to both -- and `proxy_assertion_nonces` is
        memory-backed, so it dies with the broker. Every cold start then refused its first
        proxied call, naming a replay that had not happened.

        A pod CANNOT do this for itself: an anchor reads through the collection registry,
        which needs a connected NATS client, and the thing that connects is the very
        ToolServer the pod has finished constructing by then. So it is wired from the
        connected callback, which runs before `serve` builds the guard. An anchor a host
        had to remember to build is one a host will forget to build.
        """
        server = _FakeToolServer()
        server.serve_event.set()
        bootstrap = _ConcreteBootstrap(
            server=server,
            register_log=[],
            collection_tables=_collection_tables(),
        )

        run_task = asyncio.create_task(bootstrap.run_async())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await server.connected_callbacks[0](_stack_nats_client())
        await run_task

        assert isinstance(server.assertion_replay_anchor, CollectionReplayAnchor)


class TestUnoverriddenHooks:
    """default ``build_server`` and ``register_tools`` raise NotImplementedError."""

    async def test_build_server_default_raises(self) -> None:
        bootstrap = ToolServerBootstrap("x")
        with pytest.raises(NotImplementedError):
            await bootstrap.build_server()

    async def test_register_tools_default_raises(self) -> None:
        bootstrap = ToolServerBootstrap("x")
        with pytest.raises(NotImplementedError):
            await bootstrap.register_tools(MagicMock())


class TestRunServe:
    """default ``run_serve`` awaits ``server.serve()``."""

    async def test_run_serve_awaits_serve(self) -> None:
        server = AsyncMock()
        server.serve = AsyncMock(return_value=None)
        bootstrap = ToolServerBootstrap("x")
        await bootstrap.run_serve(server)
        server.serve.assert_awaited_once()


class _ReadinessFakeServer(_FakeToolServer):
    """a ToolServer stand-in exposing the three probe surfaces the bootstrap health server reads;
    ``jwks_warmed`` is flipped by the test to drive the NOT-READY -> READY transition. ``is_healthy``
    (not ``is_connected``) is the nats liveness surface the probe now reads -- real NATS health, so a
    dead connection trips liveness instead of reporting healthy forever."""

    def __init__(self) -> None:
        super().__init__()
        self.is_healthy = True
        self.tools_count = 1
        self.jwks_warmed = False


def _free_port() -> int:
    """a TCP port nothing is listening on right now.

    :return: the port
    :rtype: int
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
    return port


async def _probe(port: int, path: str) -> tuple[int, dict[str, Any]]:
    """GET a health route as a kube probe would, asking for the JSON body.

    :param port: the pod's health port
    :ptype port: int
    :param path: the route, e.g. ``/healthz/ready``
    :ptype path: str
    :return: the HTTP status and the decoded status body
    :rtype: tuple[int, dict[str, Any]]
    """
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        writer.write(f"GET {path}?format=json HTTP/1.1\r\nHost: pod\r\nConnection: close\r\n\r\n".encode())
        await writer.drain()
        raw = await reader.read()
    finally:
        writer.close()
        await writer.wait_closed()
    head, _, body = raw.partition(b"\r\n\r\n")
    status = int(head.split()[1])
    return status, json.loads(body)


@contextlib.asynccontextmanager
async def _serving_pod(server: _ReadinessFakeServer, *, version: str | None = None) -> AsyncIterator[int]:
    """run a tool pod over ``server`` through ``run_async`` until the block exits.

    the health listener is the one ``run_async`` starts, probed over HTTP exactly as kube probes it.

    :param server: the server the pod builds
    :ptype server: _ReadinessFakeServer
    :param version: the release version the pod is constructed with
    :ptype version: str | None
    :return: an async iterator yielding the pod's health port once it is listening
    :rtype: AsyncIterator[int]
    """
    port = _free_port()

    class _HealthBootstrap(ToolServerBootstrap):
        async def build_server(self) -> Any:
            return server

        async def register_tools(self, server: Any) -> None:
            return None

    run_task = asyncio.create_task(_HealthBootstrap("test-pod", health_port=port, version=version).run_async())
    try:
        for _ in range(500):
            if run_task.done():
                run_task.result()
                raise AssertionError("run_async returned before its health listener came up")
            try:
                _reader, writer = await asyncio.open_connection("127.0.0.1", port)
            except OSError:
                await asyncio.sleep(0.01)
                continue
            writer.close()
            await writer.wait_closed()
            break
        else:
            raise AssertionError(f"the pod's health listener never bound port {port}")
        yield port
    finally:
        server.serve_event.set()
        await asyncio.wait_for(run_task, timeout=5.0)


def _components(body: dict[str, Any]) -> dict[str, bool]:
    """the per-component verdicts of a health status body.

    :param body: the decoded status body
    :ptype body: dict[str, Any]
    :return: component name -> healthy
    :rtype: dict[str, bool]
    """
    return {component["name"]: component["healthy"] for component in body["components"]}


class TestHealthServerReadinessGate:
    """B5: the tool-pod health server gates readiness on the JWKS being warm -- it must report
    NOT-READY (503) until the pod's JWKS provider can actually verify a token."""

    async def test_reports_not_ready_until_jwks_warmed(self) -> None:
        srv = _ReadinessFakeServer()
        async with _serving_pod(srv) as port:
            # before the JWKS warms: the jwks_warmed component is unhealthy -> overall NOT-READY.
            status, before = await _probe(port, "/healthz/ready")
            comps = _components(before)
            assert "jwks_warmed" in comps, "the tool-pod health server must wire a jwks_warmed readiness gate"
            assert comps["jwks_warmed"] is False
            assert before["healthy"] is False
            assert status == 503
            # after the first successful JWKS fetch: the gate clears -> READY.
            srv.jwks_warmed = True
            status, after = await _probe(port, "/healthz/ready")
            assert all(_components(after).values())
            assert after["healthy"] is True
            assert status == 200

    async def test_cold_jwks_cache_does_not_fail_liveness(self) -> None:
        """the readiness gate must be invisible to the liveness verdict.

        this is what buys the tool pods a livenessProbe: under the old aliased
        contract a cold JWKS cache took /healthz down too, so the only safe
        deployment was to ship with no livenessProbe and no restart-on-wedge net.
        """
        srv = _ReadinessFakeServer()
        srv.jwks_warmed = False
        srv.tools_count = 0
        async with _serving_pod(srv) as port:
            live_status, live = await _probe(port, "/healthz/live")
            ready_status, ready = await _probe(port, "/healthz/ready")
        assert live["healthy"] is True, "a cold JWKS cache must never restart the pod"
        assert live_status == 200
        assert [c["name"] for c in live["components"]] == ["nats"]
        assert ready["healthy"] is False
        assert ready_status == 503

    async def test_dead_nats_fails_both_tiers(self) -> None:
        """a terminally wedged data plane restarts the pod AND pulls it from rotation."""
        srv = _ReadinessFakeServer()
        srv.is_healthy = False
        srv.jwks_warmed = True
        async with _serving_pod(srv) as port:
            live_status, live = await _probe(port, "/healthz/live")
            ready_status, ready = await _probe(port, "/healthz/ready")
        assert live["healthy"] is False
        assert ready["healthy"] is False
        assert live_status == 503
        assert ready_status == 503


class TestTheToolPodHealthServerCarriesItsVersion:
    """the tool pod's probe body names the release answering it, read from its distribution."""

    async def test_the_constructor_version_reaches_the_health_status(self) -> None:
        srv = _ReadinessFakeServer()
        async with _serving_pod(srv, version="9.9.9") as port:
            _status, live = await _probe(port, "/healthz/live")
            _status, ready = await _probe(port, "/healthz/ready")
        assert live["version"] == "9.9.9"
        assert ready["version"] == "9.9.9"

    async def test_a_subclass_passing_no_version_still_serves(self) -> None:
        srv = _ReadinessFakeServer()
        async with _serving_pod(srv) as port:
            status, live = await _probe(port, "/healthz/live")
        assert status == 200
        assert live["version"] is None

    def test_the_builtin_tool_server_passes_the_installed_distribution_version(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import importlib.metadata

        from threetears.agent.tools import serve as serve_module

        captured: dict[str, Any] = {}

        def _capture_run(self: ToolServerBootstrap) -> None:
            captured["version"] = self.version

        monkeypatch.setattr(ToolServerBootstrap, "run", _capture_run)
        serve_module.main()

        installed = importlib.metadata.version("3tears-agent-tools")
        assert installed
        assert captured["version"] == installed

    async def test_the_starting_line_names_the_version(self, caplog: pytest.LogCaptureFixture) -> None:
        server = _FakeToolServer()

        class _VersionedBootstrap(ToolServerBootstrap):
            def __init__(self) -> None:
                super().__init__("versioned-pod", health_port=0, version="9.9.9")

            async def build_server(self) -> Any:
                return server

            async def register_tools(self, server: Any) -> None:
                return None

        server.serve_event.set()
        with caplog.at_level(logging.INFO, logger="threetears.agent.tools.bootstrap"):
            await _VersionedBootstrap().run_async()

        starting = [r for r in caplog.records if r.getMessage() == "versioned-pod starting"]
        assert len(starting) == 1
        assert starting[0].__dict__["extra_data"]["version"] == "9.9.9"


class _StartupFailureBootstrap(ToolServerBootstrap):
    """subclass whose ``build_server`` raises, to drive ``run``'s failure classification."""

    def __init__(self, exc: BaseException) -> None:
        # health_port=0 -> OS-assigned ephemeral port, so nothing collides in CI.
        super().__init__("test-pod", health_port=0)
        self.exc = exc

    async def build_server(self) -> Any:
        raise self.exc

    async def register_tools(self, server: Any) -> None:  # pragma: no cover - startup never gets here
        raise AssertionError("register_tools must not run after build_server raised")


class _SucceedingBootstrap(ToolServerBootstrap):
    """subclass that reaches and returns from the serve loop, driving ``run``'s happy path."""

    def __init__(self, server: _FakeToolServer) -> None:
        super().__init__("test-pod", health_port=0)
        self.server = server

    async def build_server(self) -> Any:
        return self.server

    async def register_tools(self, server: Any) -> None:
        pass


def _error_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """the ERROR-or-worse records the bootstrap emitted, in order."""
    return [r for r in caplog.records if r.levelno >= logging.ERROR]


class TestConfigFaultIsTerminal:
    """a permanent config fault must stop the pod, not feed a restart loop.

    bluelabsio/14-eng-ai-bot#235: a built-in tool pod started with a renamed identity
    env var raised, exited 1, and was restarted 9,580 times over 8 days without ever
    reaching ``running``. no healthcheck could fire (the container never started), no
    neighbour declared ``depends_on`` it, and the 278,110 identical traceback lines it
    wrote went to a stderr nobody tails. the fail-loud intent was right; "loud once"
    is what the deployment turned into "invisible forever".

    the fix has two halves and both are asserted here: a distinct exit STATUS, which
    is the only thing a supervisor can branch on, and ONE structured record naming the
    variable an operator has to go change.
    """

    def test_config_fault_exits_with_ex_config(self) -> None:
        """the status is 78, ``EX_CONFIG`` from sysexits.h -- not the generic 1."""
        bootstrap = _StartupFailureBootstrap(
            ToolPodConfigError("SOME_VAR is required", variable="SOME_VAR"),
        )

        with pytest.raises(SystemExit) as exit_info:
            bootstrap.run()

        assert exit_info.value.code == EX_CONFIG
        assert EX_CONFIG == 78, "the value is the contract: compose / k8s alerting branches on it"

    def test_config_fault_logs_exactly_one_error_naming_the_variable(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """one record, through the repo logger, carrying the offending variable.

        one, because the incident's cost was repetition. through ``get_logger``, because
        a traceback on container stderr never reaches the observability pipeline and so
        was never queryable. naming the variable, because an operator who reads the line
        and still has to open the source has not been told anything.
        """
        bootstrap = _StartupFailureBootstrap(
            ToolPodConfigError(
                "THREETEARS_TOOL_POD_ID is missing (the minter kid must be the pod id)",
                variable="THREETEARS_TOOL_POD_ID",
            ),
        )

        with caplog.at_level(logging.ERROR), pytest.raises(SystemExit):
            bootstrap.run()

        errors = _error_records(caplog)
        assert len(errors) == 1, f"expected exactly one ERROR record, got {[r.getMessage() for r in errors]}"
        extra = getattr(errors[0], "extra_data", {})
        assert extra.get("variable") == "THREETEARS_TOOL_POD_ID"
        assert extra.get("exit_code") == EX_CONFIG
        assert "THREETEARS_TOOL_POD_ID" in extra.get("error", "")

    def test_unrelated_value_error_is_not_treated_as_config(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """the reason the terminal path keys on a distinct type, not on ``ValueError``.

        a ``ValueError`` out of a tool's own business logic during registration is a bug
        that a restart may well clear. catching the base type would make it terminal and
        keep a recoverable pod down.
        """
        bootstrap = _StartupFailureBootstrap(ValueError("bad payload from some tool's own parsing"))

        with caplog.at_level(logging.ERROR), pytest.raises(ValueError, match="bad payload"):
            bootstrap.run()

        assert _error_records(caplog) == [], "a non-config failure must not emit the terminal record"

    def test_transient_failure_still_propagates_and_stays_retryable(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """a NATS server that is not up yet is exactly what supervisor restarts are for."""
        bootstrap = _StartupFailureBootstrap(ConnectionRefusedError("nats://nats:4222 refused"))

        with caplog.at_level(logging.ERROR), pytest.raises(ConnectionRefusedError):
            bootstrap.run()

        assert _error_records(caplog) == []

    def test_successful_run_is_unaffected(self) -> None:
        """the guard is inert on the path that actually serves."""
        server = _FakeToolServer()
        server.serve_event.set()  # release serve immediately so run returns

        _SucceedingBootstrap(server).run()

        assert server.serve_called is True


def test_signal_handler_uses_service_name_in_task_name() -> None:
    bootstrap = ToolServerBootstrap("my-svc")
    server = MagicMock()
    handler = bootstrap.make_signal_handler(server, "sigterm")
    # closure captured the service name in the task name template
    assert callable(handler)


class _FailingShutdownServer(_FakeToolServer):
    """a server whose shutdown raises the way a reconnecting NATS client's drain did.

    Its ``shutdown`` does NOT release ``serve`` -- exactly the real failure: ``ToolServer.shutdown``
    raised before it set the event ``serve`` waits on, so ``serve`` never returned and two tool
    pods stayed alive for two days after SIGTERM.
    """

    def __init__(self) -> None:
        super().__init__()
        self.on_serve: Any = None

    async def serve(self) -> None:
        self.serve_called = True
        if self.on_serve is not None:
            self.on_serve()
        await self.serve_event.wait()

    async def shutdown(self) -> None:
        self.shutdown_called = True
        raise ConnectionResetError("NATS drain failed; connection reset by peer")


class _FailingShutdownBootstrap(ToolServerBootstrap):
    """drives ``run`` with a server whose shutdown raises, signalled from inside ``serve``."""

    def __init__(self, server: _FailingShutdownServer) -> None:
        super().__init__("failing-shutdown-pod", health_port=0)
        self.server = server
        as_server: Any = server
        server.on_serve = lambda: self.make_signal_handler(as_server, "sigterm")()

    async def build_server(self) -> Any:
        return self.server

    async def register_tools(self, server: Any) -> None:
        pass


class TestAFailedShutdownStillExits:
    """SIGTERM ends the process even when the server's own shutdown raises."""

    async def test_serve_is_left_and_the_failure_is_raised(self, caplog: pytest.LogCaptureFixture) -> None:
        """``run_async`` returns control -- by raising the typed failure -- instead of waiting forever.

        :param caplog: the log capture
        :ptype caplog: pytest.LogCaptureFixture
        :return: nothing
        :rtype: None
        """
        server = _FailingShutdownServer()
        bootstrap = _FailingShutdownBootstrap(server)

        with caplog.at_level(logging.ERROR), pytest.raises(ToolPodShutdownError) as raised:
            await asyncio.wait_for(bootstrap.run_async(), timeout=5.0)

        assert server.shutdown_called is True
        assert isinstance(raised.value.__cause__, ConnectionResetError)
        errors = _error_records(caplog)
        assert len(errors) == 1, [r.getMessage() for r in errors]
        extra = getattr(errors[0], "extra_data", {})
        assert extra.get("error_type") == "ConnectionResetError"
        assert "connection reset by peer" in extra.get("error", "")
        assert extra.get("reason") == "sigterm"

    def test_the_process_exits_non_zero(self) -> None:
        """``run`` owns the exit status: a shutdown that failed is not a clean exit.

        :return: nothing
        :rtype: None
        """
        bootstrap = _FailingShutdownBootstrap(_FailingShutdownServer())

        with pytest.raises(SystemExit) as exit_info:
            bootstrap.run()

        assert exit_info.value.code == EX_SOFTWARE
        assert EX_SOFTWARE not in (0, EX_CONFIG)

    async def test_a_shutdown_that_hangs_is_cut_off_at_the_bound(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """a drain that never returns is the same failure as one that raises, once the bound passes.

        :param monkeypatch: the test's patcher
        :ptype monkeypatch: pytest.MonkeyPatch
        :return: nothing
        :rtype: None
        """
        monkeypatch.setenv(SHUTDOWN_TIMEOUT_ENV, "0.1")
        server = _FailingShutdownServer()

        async def _hang() -> None:
            server.shutdown_called = True
            await asyncio.Event().wait()

        server.shutdown = _hang  # type: ignore[method-assign]
        bootstrap = _FailingShutdownBootstrap(server)

        with pytest.raises(ToolPodShutdownError) as raised:
            await asyncio.wait_for(bootstrap.run_async(), timeout=5.0)

        assert isinstance(raised.value.__cause__, TimeoutError)

    async def test_a_second_signal_does_not_start_a_second_shutdown(self) -> None:
        """SIGTERM then SIGINT (or the owner going too) drives one shutdown, not two.

        :return: nothing
        :rtype: None
        """
        server = _FakeToolServer()
        calls: list[int] = []
        original = server.shutdown

        async def _counted() -> None:
            calls.append(1)
            await original()

        server.shutdown = _counted  # type: ignore[method-assign]
        bootstrap = _ConcreteBootstrap(server=server, register_log=[])
        run_task = asyncio.create_task(bootstrap.run_async())
        await asyncio.sleep(0.01)
        as_server: Any = server
        await asyncio.gather(
            bootstrap.shutdown_server(as_server, reason="sigterm"),
            bootstrap.shutdown_server(as_server, reason="sigint"),
        )
        await asyncio.wait_for(run_task, timeout=2.0)

        assert calls == [1]


class TestTheOwnerPidIsValidatedAtStartup:
    """``THREETEARS_TOOL_POD_OWNER_PID`` is refused at startup unless it can name a real owner."""

    @pytest.mark.parametrize("value", ["0", "-5", "1", "abc", "", "12.5"])
    def test_an_invalid_value_is_a_config_fault(self, value: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """each is refused with the variable named, and the process exits ``EX_CONFIG``.

        :param value: the invalid value
        :ptype value: str
        :param monkeypatch: the test's patcher
        :ptype monkeypatch: pytest.MonkeyPatch
        :return: nothing
        :rtype: None
        """
        monkeypatch.setenv(OWNER_PID_ENV, value)
        server = _FakeToolServer()
        server.serve_event.set()

        with pytest.raises(SystemExit) as exit_info:
            _SucceedingBootstrap(server).run()

        assert exit_info.value.code == EX_CONFIG
        assert server.serve_called is False

    def test_the_pods_own_pid_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """a pod watching itself would never notice anything.

        :param monkeypatch: the test's patcher
        :ptype monkeypatch: pytest.MonkeyPatch
        :return: nothing
        :rtype: None
        """
        monkeypatch.setenv(OWNER_PID_ENV, str(os.getpid()))
        with pytest.raises(ToolPodConfigError) as raised:
            resolve_owner_pid()
        assert raised.value.variable == OWNER_PID_ENV

    def test_unset_watches_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """opt-in: no variable, no watch.

        :param monkeypatch: the test's patcher
        :ptype monkeypatch: pytest.MonkeyPatch
        :return: nothing
        :rtype: None
        """
        monkeypatch.delenv(OWNER_PID_ENV, raising=False)
        assert resolve_owner_pid() is None

    def test_a_live_pid_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """the parent of this test process is a real, live owner.

        :param monkeypatch: the test's patcher
        :ptype monkeypatch: pytest.MonkeyPatch
        :return: nothing
        :rtype: None
        """
        monkeypatch.setenv(OWNER_PID_ENV, f" {os.getppid()} ")
        assert resolve_owner_pid() == os.getppid()


class TestAToolPodExitsWhenItsOwnerIsGone:
    """the owner watch, against a real process that really exits."""

    async def test_the_pod_shuts_down_when_the_owner_exits(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """a short-lived owner subprocess exits; the pod shuts down through the normal path.

        :param monkeypatch: the test's patcher
        :ptype monkeypatch: pytest.MonkeyPatch
        :param caplog: the log capture
        :ptype caplog: pytest.LogCaptureFixture
        :return: nothing
        :rtype: None
        """
        owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.3)"])  # noqa: S603
        monkeypatch.setenv(OWNER_PID_ENV, str(owner.pid))
        monkeypatch.setenv(OWNER_POLL_INTERVAL_ENV, "0.05")
        server = _FakeToolServer()
        bootstrap = _ConcreteBootstrap(server=server, register_log=[])

        with caplog.at_level(logging.WARNING):
            run_task = asyncio.create_task(bootstrap.run_async())
            await asyncio.sleep(0.1)
            assert not run_task.done(), "the pod must keep serving while its owner lives"
            # reap the owner, as its own parent would; an unreaped child still answers kill(pid, 0).
            await asyncio.to_thread(owner.wait)
            await asyncio.wait_for(run_task, timeout=5.0)

        assert server.shutdown_called is True
        gone = [r for r in caplog.records if r.levelno == logging.WARNING and str(owner.pid) in r.getMessage()]
        assert len(gone) == 1

    async def test_a_pod_whose_owner_lives_keeps_serving(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """the negative control: a live owner is polled and nothing happens.

        :param monkeypatch: the test's patcher
        :ptype monkeypatch: pytest.MonkeyPatch
        :return: nothing
        :rtype: None
        """
        owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])  # noqa: S603
        try:
            monkeypatch.setenv(OWNER_PID_ENV, str(owner.pid))
            monkeypatch.setenv(OWNER_POLL_INTERVAL_ENV, "0.02")
            server = _FakeToolServer()
            bootstrap = _ConcreteBootstrap(server=server, register_log=[])
            run_task = asyncio.create_task(bootstrap.run_async())
            await asyncio.sleep(0.3)

            assert not run_task.done()
            assert server.shutdown_called is False
            await server.shutdown()
            await asyncio.wait_for(run_task, timeout=2.0)
        finally:
            owner.kill()
            owner.wait()

    async def test_an_owner_gone_with_a_failing_shutdown_still_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """item 5's guarantee holds on this path too: the owner is gone and the drain raises.

        :param monkeypatch: the test's patcher
        :ptype monkeypatch: pytest.MonkeyPatch
        :return: nothing
        :rtype: None
        """
        owner = subprocess.Popen([sys.executable, "-c", "pass"])  # noqa: S603
        owner.wait()
        monkeypatch.setenv(OWNER_PID_ENV, str(owner.pid))
        monkeypatch.setenv(OWNER_POLL_INTERVAL_ENV, "0.05")
        server = _FailingShutdownServer()
        bootstrap = _ConcreteBootstrap(server=server, register_log=[])

        with pytest.raises(ToolPodShutdownError):
            await asyncio.wait_for(bootstrap.run_async(), timeout=5.0)
        assert server.shutdown_called is True
