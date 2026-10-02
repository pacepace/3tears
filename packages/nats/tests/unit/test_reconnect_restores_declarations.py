"""a NATS restart wipes memory storage; the client puts back what it declared, and nothing else.

Found live in the devx bring-up: the tool registry declared its memory-backed result stream once,
at startup. A single-node NATS restart deleted it, and from then on every tool call failed with
``stream not found`` until someone restarted the registry by hand. Every memory-storage stream a
service declared through :class:`~threetears.nats.NatsClient` had the same hole, and so did every
memory KV bucket it declared and every durable consumer it bound on one: the server forgets all
three at once.

These run the client's real reconnect path -- the ``reconnected_cb`` nats-py is handed at connect --
against a scripted JetStream that holds streams and consumers the way a server does, so a "restart"
is the server forgetting them. The live counterpart, against a real broker, is
``tests/integration/test_a_restart_wiped_stream_comes_back_live.py``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from nats.js.api import StorageType, StreamConfig
from nats.js.errors import NotFoundError

from threetears.nats import NatsClient, Subjects, set_default_namespace
from threetears.nats.kv import build_kv_stream_config

_NS = "3tears"

#: how long a test waits for the background restoration to reach the state it expects
_SETTLE_SECONDS = 5.0


class _ApiError(Exception):
    """a JetStream API refusal carrying the server's error code, as nats-py's ``APIError`` does.

    Not a fake of a protocol: the client classifies on ``err_code`` alone, so an exception carrying
    that attribute IS the input shape.
    """

    def __init__(self, err_code: int, description: str) -> None:
        super().__init__(f"nats: err_code={err_code} description={description!r}")
        self.err_code = err_code


# parity-exempt: scripted JetStream server for the stream, consumer and bind calls the client makes on a reconnect; the real JetStreamContext surface is far larger and unrelated
class _ScriptedServer:
    """a JetStream context over server-side state a test can wipe, as a restart does.

    ``streams`` maps a stream name to the config it was created with; ``consumers`` maps
    ``(stream, durable)`` to the consumer's config. Adding a stream that exists with the SAME
    config succeeds, as JetStream's create is idempotent; with a different config it is refused
    with err_code 10058, which is what the server answers.
    """

    def __init__(self) -> None:
        self.streams: dict[str, StreamConfig] = {}
        self.consumers: dict[tuple[str, str], Any] = {}
        self.added: list[StreamConfig] = []
        self.updated: list[StreamConfig] = []
        self.push_binds: list[str] = []
        self.pull_binds: list[str] = []
        self.add_failures: list[Exception] = []
        self.stream_by_subject: dict[str, str] = {}

    def restart(self) -> None:
        """forget every stream and every consumer, as a memory-storage server restart does."""
        self.streams.clear()
        self.consumers.clear()

    async def add_stream(self, config: StreamConfig) -> Any:
        self.added.append(dataclasses.replace(config))
        if self.add_failures:
            raise self.add_failures.pop(0)
        existing = self.streams.get(config.name or "")
        if existing is not None and existing != config:
            raise _ApiError(10058, "stream name already in use with a different configuration")
        self.streams[config.name or ""] = dataclasses.replace(config)
        return object()

    async def update_stream(self, config: StreamConfig) -> Any:
        self.updated.append(dataclasses.replace(config))
        self.streams[config.name or ""] = dataclasses.replace(config)
        return object()

    async def stream_info(self, name: str) -> Any:
        if name not in self.streams:
            raise NotFoundError()
        return type("_Info", (), {"config": self.streams[name]})()

    async def key_value(self, name: str) -> Any:
        if f"KV_{name}" not in self.streams:
            raise NotFoundError()
        return MagicMock(name=f"kv:{name}")

    async def consumer_info(self, stream: str, durable: str) -> Any:
        if stream not in self.streams or (stream, durable) not in self.consumers:
            raise NotFoundError()
        return type("_Info", (), {"config": self.consumers[(stream, durable)]})()

    async def add_consumer(self, stream: str, config: Any) -> Any:
        self.consumers[(stream, config.durable_name)] = config
        return object()

    async def find_stream_name_by_subject(self, subject: str) -> str:
        return self.stream_by_subject[subject]

    def _bind(self, stream: str | None, durable: str, config: Any) -> None:
        name = stream or ""
        if name not in self.streams:
            raise NotFoundError()
        self.consumers.setdefault((name, durable), config)

    async def subscribe(self, subject: str, *, durable: str, stream: str | None, config: Any, **_: Any) -> Any:
        self._bind(stream, durable, config)
        self.push_binds.append(durable)
        sub = MagicMock(name=f"push:{durable}")
        sub.unsubscribe = AsyncMock()
        return sub

    async def pull_subscribe(self, subject: str, *, durable: str, stream: str | None, config: Any) -> Any:
        self._bind(stream, durable, config)
        self.pull_binds.append(durable)
        psub = MagicMock(name=f"pull:{durable}")
        psub.unsubscribe = AsyncMock()
        psub.fetch = AsyncMock(return_value=[])
        return psub


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    set_default_namespace(_NS)


async def _connected(server: _ScriptedServer) -> tuple[NatsClient, Callable[[], Awaitable[None]]]:
    """connect a client over ``server``, returning it and the reconnect slot nats-py would call.

    :param server: the scripted JetStream server
    :ptype server: _ScriptedServer
    :return: the client and its ``reconnected_cb``
    :rtype: tuple[NatsClient, Callable[[], Awaitable[None]]]
    """
    captured: dict[str, Any] = {}
    raw = MagicMock(name="nats-py-client")
    raw.jetstream = MagicMock(return_value=server)
    raw.is_closed = False
    raw.is_connected = True
    raw.drain = AsyncMock()
    raw.close = AsyncMock()

    async def _establish(servers: list[str], options: dict[str, Any], nats_url: str) -> Any:
        captured["options"] = options
        return raw

    client = await NatsClient.connect(
        establish_connection=_establish,
        nats_url="nats://localhost:4222",
        nats_subject_namespace=_NS,
        client_name="registry",
        verify_jetstream=False,
    )
    return client, captured["options"]["reconnected_cb"]


async def _until(condition: Callable[[], bool]) -> None:
    """wait until ``condition`` holds, failing the test if it never does.

    :param condition: the state the background restoration should reach
    :ptype condition: Callable[[], bool]
    :return: nothing
    :rtype: None
    """
    deadline = asyncio.get_running_loop().time() + _SETTLE_SECONDS
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("the client did not restore what it declared in time")
        await asyncio.sleep(0.01)


_RESULTS = f"{_NS}-tools-results"


async def _declare_results_stream(client: NatsClient) -> None:
    await client.ensure_jetstream_stream(
        name="tools-results",
        subjects=[f"{_NS}.tools.result.>", f"{_NS}.tools.reply.>"],
        max_age_seconds=900.0,
        max_msgs_per_subject=1,
    )


@pytest.mark.asyncio
async def test_a_memory_stream_wiped_by_a_restart_is_declared_again_with_its_own_config() -> None:
    """the registry's defect: its result stream must come back without the registry restarting."""
    server = _ScriptedServer()
    client, reconnected = await _connected(server)
    await _declare_results_stream(client)
    declared = server.streams[_RESULTS]

    server.restart()
    await reconnected()
    await _until(lambda: _RESULTS in server.streams)

    assert server.streams[_RESULTS] == declared
    assert declared.storage == StorageType.MEMORY
    assert declared.max_age == 900.0 and declared.max_msgs_per_subject == 1
    assert server.updated == [], "a re-declaration creates; it never updates"
    await client.shutdown()


@pytest.mark.asyncio
async def test_a_file_stream_is_not_declared_again() -> None:
    """file storage survives a restart, so re-declaring it would only risk clobbering it."""
    server = _ScriptedServer()
    client, reconnected = await _connected(server)
    await client.ensure_jetstream_stream(name="audit", subjects=[f"{_NS}.audit.>"], storage="file")
    adds_before = len(server.added)

    await reconnected()
    for _ in range(20):
        await asyncio.sleep(0)

    assert len(server.added) == adds_before
    await client.shutdown()


@pytest.mark.asyncio
async def test_a_stream_another_process_changed_is_left_as_it_is(caplog: pytest.LogCaptureFixture) -> None:
    """a reconnect that was only a network blip finds the stream alive; it is never reconciled back.

    another declarer of the same name may have changed it since; this client restores what a
    restart took, it does not fight over a stream that is still there.
    """
    server = _ScriptedServer()
    client, reconnected = await _connected(server)
    await _declare_results_stream(client)
    changed = dataclasses.replace(server.streams[_RESULTS], max_age=60.0)
    server.streams[_RESULTS] = changed
    adds_before = len(server.added)

    with caplog.at_level(logging.INFO, logger="threetears.nats.client"):
        await reconnected()
        await _until(lambda: len(server.added) > adds_before)
        for _ in range(20):
            await asyncio.sleep(0)

    assert server.streams[_RESULTS] == changed
    assert server.updated == []
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(server.added) == adds_before + 1, "an answered refusal is final, not retried"
    await client.shutdown()


@pytest.mark.asyncio
async def test_a_failed_redeclaration_is_logged_at_error_and_retried(caplog: pytest.LogCaptureFixture) -> None:
    """a broker still coming back must not leave the stream missing for good, nor fail silently."""
    server = _ScriptedServer()
    client, reconnected = await _connected(server)
    await _declare_results_stream(client)

    server.restart()
    server.add_failures.append(RuntimeError("nats: timeout"))
    with caplog.at_level(logging.ERROR, logger="threetears.nats.client"):
        await reconnected()
        await _until(lambda: _RESULTS in server.streams)

    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any(_RESULTS in message and "nats: timeout" in message for message in errors), errors
    await client.shutdown()


@pytest.mark.asyncio
async def test_a_declared_memory_kv_bucket_is_declared_again() -> None:
    """a declarer's bucket comes back after a restart even when nothing in its own process touches it.

    the bucket handle self-heals only when an operation on it fails, and a process that only binds
    the bucket waits for its declarer -- so a declarer that does not use its own bucket left every
    binder waiting.
    """
    server = _ScriptedServer()
    client, reconnected = await _connected(server)
    await client.ensure_kv_bucket(name="sessions", ttl=timedelta(minutes=5), history=1)
    kv_stream = f"KV_{_NS}-sessions"
    declared = server.streams[kv_stream]

    server.restart()
    await reconnected()
    await _until(lambda: kv_stream in server.streams)

    assert server.streams[kv_stream] == declared
    assert declared.allow_direct is True
    await client.shutdown()


@pytest.mark.asyncio
async def test_a_kv_bucket_only_bound_is_not_declared_by_the_binder() -> None:
    """a process that BINDS a bucket has no authority to create it, after a restart or ever."""
    server = _ScriptedServer()
    binder, reconnected = await _connected(server)
    server.streams[f"KV_{_NS}-shared"] = build_kv_stream_config(
        bucket=f"{_NS}-shared", ttl_seconds=0, history=1, storage_type=StorageType.MEMORY, direct=True
    )
    await binder.kv_bucket(name="shared", create_if_missing=False)

    server.restart()
    await reconnected()
    for _ in range(20):
        await asyncio.sleep(0)

    assert server.added == []
    await binder.shutdown()


async def _ack(msg: Any) -> None:
    await msg.ack()


@pytest.mark.asyncio
async def test_a_pull_durable_on_a_wiped_stream_is_created_again() -> None:
    """a durable lives on its stream: the restart that took the stream took the durable too."""
    server = _ScriptedServer()
    client, reconnected = await _connected(server)
    await _declare_results_stream(client)
    consumer = await client.jetstream_pull_subscribe(
        subject=Subjects.audit_wildcard(), durable="persister", cb=_ack, max_deliver=3, stream=_RESULTS
    )

    server.restart()
    await reconnected()
    await _until(lambda: _RESULTS in server.streams)
    await consumer.fetch_and_process()

    assert (_RESULTS, "persister") in server.consumers
    assert server.pull_binds == ["persister", "persister"]
    await consumer.stop()
    await client.shutdown()


@pytest.mark.asyncio
async def test_a_pull_durable_that_survived_is_not_rebound() -> None:
    """a reconnect over a network blip leaves the durable in place; rebinding it is churn."""
    server = _ScriptedServer()
    client, reconnected = await _connected(server)
    await _declare_results_stream(client)
    consumer = await client.jetstream_pull_subscribe(
        subject=Subjects.audit_wildcard(), durable="persister", cb=_ack, max_deliver=3, stream=_RESULTS
    )
    adds_before = len(server.added)

    await reconnected()
    await _until(lambda: len(server.added) > adds_before)
    for _ in range(20):
        await asyncio.sleep(0)
    await consumer.fetch_and_process()

    assert server.pull_binds == ["persister"]
    await consumer.stop()
    await client.shutdown()


@pytest.mark.asyncio
async def test_a_push_durable_on_a_wiped_stream_is_bound_again() -> None:
    """a push durable's subscription is replayed by nats-py, but nothing delivers to it until the
    durable exists again."""
    server = _ScriptedServer()
    client, reconnected = await _connected(server)
    await _declare_results_stream(client)
    consumer = await client.jetstream_subscribe_durable(
        subject=Subjects.audit_wildcard(), durable="listener", cb=_ack, max_deliver=3, stream=_RESULTS
    )
    first = consumer.raw_subscription

    server.restart()
    await reconnected()
    await _until(lambda: server.push_binds == ["listener", "listener"])

    assert (_RESULTS, "listener") in server.consumers
    assert consumer.raw_subscription is not first
    first.unsubscribe.assert_awaited()
    await consumer.stop()
    await client.shutdown()


@pytest.mark.asyncio
async def test_a_durable_whose_stream_is_not_back_yet_is_retried(caplog: pytest.LogCaptureFixture) -> None:
    """the stream may be another process's to declare; until it is back, the rebind fails and retries."""
    server = _ScriptedServer()
    client, reconnected = await _connected(server)
    server.streams["3tears-elsewhere"] = StreamConfig(name="3tears-elsewhere", subjects=[f"{_NS}.audit.>"])
    consumer = await client.jetstream_subscribe_durable(
        subject=Subjects.audit_wildcard(), durable="listener", cb=_ack, max_deliver=3, stream="3tears-elsewhere"
    )
    declared = server.streams["3tears-elsewhere"]

    server.restart()
    with caplog.at_level(logging.ERROR, logger="threetears.nats.client"):
        await reconnected()
        await _until(lambda: any(r.levelno == logging.ERROR for r in caplog.records))
        server.streams["3tears-elsewhere"] = declared  # its own declarer brings it back
        await _until(lambda: server.push_binds == ["listener", "listener"])

    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any("listener" in message for message in errors), errors
    await consumer.stop()
    await client.shutdown()


@pytest.mark.asyncio
async def test_shutdown_stops_a_restoration_still_retrying() -> None:
    """a restoration that cannot finish must not outlive its client."""
    server = _ScriptedServer()
    client, reconnected = await _connected(server)
    await _declare_results_stream(client)

    server.restart()
    server.add_failures.extend(RuntimeError("nats: timeout") for _ in range(1000))
    await reconnected()
    await _until(lambda: len(server.add_failures) < 1000)

    await asyncio.wait_for(client.shutdown(), timeout=2.0)
    remaining = len(server.add_failures)
    await asyncio.sleep(1.0)

    assert len(server.add_failures) == remaining, "the restoration kept running after shutdown"
