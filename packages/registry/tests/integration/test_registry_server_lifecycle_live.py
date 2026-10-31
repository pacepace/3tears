"""integration: a real :class:`RegistryServer` started with ``serve()`` on a real bus.

Two properties of the running registry that only its own startup and shutdown can show:

* the health listener ``serve()`` starts names the release that is answering -- the version
  the server was constructed with, or ``null`` for an embedding caller that passed none -- on
  the probe body kube and the devx preflight read;
* ``shutdown()`` releases what ``serve()`` wired into the heartbeat collection registry: it stops
  the cross-pod invalidation listener and closes the collections, so neither the subscription
  nor a collection's own background work outlives the server that started it.

Everything is the production path: the server connects to a NATS testcontainer, binds the shared
collections bucket (declared first, as the hub declares it), loads its catalog, starts every
handler and its health server, and is probed over HTTP.

requires docker; marked integration. run with::

    ./scripts/test-integration.sh registry -rs
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import socket
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid7

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from threetears.core.collections.bucket import COLLECTIONS_BUCKET_SUFFIX
from threetears.core.collections.registry import CollectionRegistry
from threetears.nats import NatsClient
from threetears.registry import server as server_module
from threetears.registry.auth import AllowAllAuthorizer
from threetears.registry.catalog import ToolEndpoint
from threetears.registry.server import RegistryServer

from ..unit.registry.copy_entries import uniform_entry

pytestmark = pytest.mark.integration


def _free_port() -> int:
    """a TCP port nothing is listening on right now.

    :return: the port
    :rtype: int
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
    return port


async def _probe(port: int, path: str) -> tuple[int, dict[str, Any]] | None:
    """GET a health route as a kube probe would, asking for the JSON body.

    :param port: the registry's health port
    :ptype port: int
    :param path: the route, e.g. ``/healthz/live``
    :ptype path: str
    :return: the HTTP status and decoded body, or ``None`` while nothing is listening
    :rtype: tuple[int, dict[str, Any]] | None
    """
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
    except OSError:
        return None
    try:
        writer.write(f"GET {path}?format=json HTTP/1.1\r\nHost: registry\r\nConnection: close\r\n\r\n".encode())
        await writer.drain()
        raw = await reader.read()
    finally:
        writer.close()
        await writer.wait_closed()
    head, _, body = raw.partition(b"\r\n\r\n")
    return int(head.split()[1]), json.loads(body)


@contextlib.asynccontextmanager
async def _running_registry(
    nats_url: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    version: str | None,
    heartbeat_timeout: float | None = None,
    catalog: dict[str, bytes] | None = None,
) -> AsyncIterator[tuple[RegistryServer, int]]:
    """a registry started through ``serve()``, yielded once its health listener answers.

    the namespace is fresh per use so one test's buckets and streams never meet another's. the
    collections bucket is declared first because the registry only BINDS it -- the hub is its
    declaring identity in production.

    :param nats_url: the bus
    :ptype nats_url: str
    :param monkeypatch: supplies the proxy-assertion signing key the registry requires to start
    :ptype monkeypatch: pytest.MonkeyPatch
    :param version: the release version the server is constructed with
    :ptype version: str | None
    :param heartbeat_timeout: the server's heartbeat timeout; its default when None
    :ptype heartbeat_timeout: float | None
    :param catalog: catalog entries the bucket holds before the registry starts, by KV key
    :ptype catalog: dict[str, bytes] | None
    :return: the serving registry and its health port
    :rtype: AsyncIterator[tuple[RegistryServer, int]]
    """
    seed = base64.urlsafe_b64encode(Ed25519PrivateKey.generate().private_bytes_raw()).decode("ascii")
    monkeypatch.setenv("THREETEARS_PROXY_ASSERTION_SIGNING_KEY", seed)
    namespace = f"lifecycle{uuid7().hex[-12:]}"
    async with await NatsClient.connect(
        nats_url=nats_url, nats_subject_namespace=namespace, client_name="bucket-declarer"
    ) as declarer:
        await declarer.ensure_kv_bucket(name=COLLECTIONS_BUCKET_SUFFIX)
        if catalog:
            # as the registry declares its catalog bucket (catalog_persistence.catalog_bucket)
            persisted = await declarer.ensure_kv_bucket(
                name=f"tool_catalog_{namespace}", storage="file", history=1, direct=True, prefix_namespace=False
            )
            for key, value in catalog.items():
                await persisted.put(key=key, value=value)

    port = _free_port()
    server = RegistryServer(
        nats_url=nats_url,
        namespace=namespace,
        authorizer=AllowAllAuthorizer(),
        health_port=port,
        version=version,
        heartbeat_timeout=heartbeat_timeout,
        kv_bucket=f"tool_catalog_{namespace}",
    )
    serving = asyncio.create_task(server.serve())
    try:
        for _ in range(600):
            if serving.done():
                serving.result()
                raise AssertionError("serve() returned before the registry's health listener answered")
            if await _probe(port, "/healthz/live") is not None:
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError(f"the registry's health listener never answered on port {port}")
        yield server, port
    finally:
        await server.shutdown()
        await asyncio.wait_for(serving, timeout=10.0)


class TestTheRunningRegistryNamesItsVersion:
    """the probe body carries the version the server was given, read off the live listener."""

    async def test_the_constructor_version_reaches_the_probe_body(
        self, nats_container: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """the status the probe renders carries the version the server was given.

        :param nats_container: the bus
        :ptype nats_container: str
        :param monkeypatch: pytest monkeypatch fixture
        :ptype monkeypatch: pytest.MonkeyPatch
        :return: nothing
        :rtype: None
        """
        async with _running_registry(nats_container, monkeypatch, version="9.9.9") as (_server, port):
            live = await _probe(port, "/healthz/live")
            ready = await _probe(port, "/healthz/ready")
        assert live is not None and ready is not None
        assert live[1]["version"] == "9.9.9"
        assert ready[1]["version"] == "9.9.9"

    async def test_an_embedding_caller_that_passes_no_version_still_serves(
        self, nats_container: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """the parameter is optional: an embedding host passing none keeps working, version null.

        :param nats_container: the bus
        :ptype nats_container: str
        :param monkeypatch: pytest monkeypatch fixture
        :ptype monkeypatch: pytest.MonkeyPatch
        :return: nothing
        :rtype: None
        """
        async with _running_registry(nats_container, monkeypatch, version=None) as (_server, port):
            live = await _probe(port, "/healthz/live")
        assert live is not None
        status, body = live
        assert status == 200
        assert body["version"] is None


class TestShutdownReleasesTheCollectionRegistry:
    """every subscription `serve()` makes through the collection registry is released by `shutdown()`.

    `startup` calls `start_invalidation_listener`, and `shutdown` tore down six other components
    while skipping it -- so the listener outlived the server that owned it. A collection can also
    start background work of its own -- a write-behind coordination collection runs a periodic
    flusher -- and it owes one last flush on the way out.
    """

    async def test_shutdown_stops_the_invalidation_listener_and_closes_the_collections(
        self, nats_container: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """both halves of the registry teardown run against the registry ``serve()`` built.

        :param nats_container: the bus
        :ptype nats_container: str
        :param monkeypatch: pytest monkeypatch fixture
        :ptype monkeypatch: pytest.MonkeyPatch
        :return: nothing
        :rtype: None
        """
        built: list[CollectionRegistry] = []
        released: list[str] = []
        real_builder = server_module.build_heartbeat_collection_registry

        def _recording_builder(**kwargs: Any) -> Any:
            collection_registry, heartbeat_collection = real_builder(**kwargs)
            stop_listener = collection_registry.stop_invalidation_listener
            close_collections = collection_registry.close_collections

            async def _stop_listener() -> None:
                released.append("stop_invalidation_listener")
                await stop_listener()

            async def _close_collections() -> None:
                released.append("close_collections")
                await close_collections()

            monkeypatch.setattr(collection_registry, "stop_invalidation_listener", _stop_listener)
            monkeypatch.setattr(collection_registry, "close_collections", _close_collections)
            built.append(collection_registry)
            return collection_registry, heartbeat_collection

        monkeypatch.setattr(server_module, "build_heartbeat_collection_registry", _recording_builder)

        # leaving the block shuts the server down, once, and lets serve() return.
        async with _running_registry(nats_container, monkeypatch, version=None):
            assert len(built) == 1, "serve() must wire exactly one heartbeat collection registry"
            assert released == []

        assert released == ["stop_invalidation_listener", "close_collections"]


class TestAWarmLoadedReplicaHoldsItsCalls:
    """a replica joins its call group only once the copies its catalog warm-loaded are routable again, or the
    heartbeat timeout has passed: on cobalt dev on 2026-10-30 a new replica took a scheduled call during a rollout
    and refused it TOOL_UNAVAILABLE while every other replica could route it."""

    async def test_the_proxy_joins_and_readiness_turns_only_after_the_loaded_copies_settle(
        self, nats_container: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """the health listener answers not-ready (catalog_revived) and the call proxy has not started while a copy
        whose pod never comes back is waited for; both proceed once the window has passed.

        :param nats_container: the bus
        :ptype nats_container: str
        :param monkeypatch: pytest monkeypatch fixture
        :ptype monkeypatch: pytest.MonkeyPatch
        :return: nothing
        :rtype: None
        """
        window = 4.0
        started: list[float] = []

        class _RecordingProxy(server_module.CallProxy):  # type: ignore[misc,name-defined]
            async def start(self, nc: Any) -> None:
                joining = asyncio.get_running_loop().time()
                await super().start(nc)
                started.append(joining)

        monkeypatch.setattr(server_module, "CallProxy", _RecordingProxy)
        entry = uniform_entry(
            tool_name="enr.reload",
            tool_version="1.0",
            description="reload",
            input_schema={"type": "object", "properties": {}},
            endpoints=[ToolEndpoint(pod_id="pod-gone", status="available")],
        )
        loop = asyncio.get_running_loop()
        began = loop.time()
        async with _running_registry(
            nats_container,
            monkeypatch,
            version=None,
            heartbeat_timeout=window,
            catalog={"enr_reload_AT_1_0": json.dumps(entry.to_dict()).encode("utf-8")},
        ) as (_server, port):
            ready = await _probe(port, "/healthz/ready")
            assert ready is not None
            held = {component["name"]: component["healthy"] for component in ready[1]["components"]}
            assert ready[0] == 503 and held.get("catalog_revived") is False, ready
            assert started == [], "the call proxy joined while a warm-loaded copy was still unrouted"
            for _ in range(400):
                if started:
                    break
                await asyncio.sleep(0.05)
            assert started, "the call proxy never joined"
            assert started[0] - began >= window * 0.9, "the call proxy joined before the window passed"
            ready = await _probe(port, "/healthz/ready")
            assert ready is not None
            settled = {component["name"]: component["healthy"] for component in ready[1]["components"]}
            assert settled.get("catalog_revived") is True, ready

    async def test_a_shutdown_while_the_replica_holds_its_calls_ends_it(
        self, nats_container: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """a replica can hold its calls for a whole heartbeat timeout: a shutdown in that time stops the start-up
        rather than letting it start the call proxy on the connection being closed (``serve`` then never ended).

        :param nats_container: the bus
        :ptype nats_container: str
        :param monkeypatch: pytest monkeypatch fixture
        :ptype monkeypatch: pytest.MonkeyPatch
        :return: nothing
        :rtype: None
        """
        entry = uniform_entry(
            tool_name="enr.reload",
            tool_version="1.0",
            description="reload",
            input_schema={"type": "object", "properties": {}},
            endpoints=[ToolEndpoint(pod_id="pod-gone", status="available")],
        )
        async with _running_registry(
            nats_container,
            monkeypatch,
            version=None,
            heartbeat_timeout=600.0,
            catalog={"enr_reload_AT_1_0": json.dumps(entry.to_dict()).encode("utf-8")},
        ) as (_server, port):
            ready = await _probe(port, "/healthz/ready")
            assert ready is not None and ready[0] == 503
        # leaving the block shut the server down and waited for serve() to end, within its bound
