"""moving a stream live on file storage onto memory keeps what it still owes, and nothing a consumer read."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from nats.js.api import ConsumerConfig, StorageType, StreamConfig

from threetears.nats.stream_storage import StreamMoveRefused, moves_wanted, stream_move, stream_move_plan

_NOW = datetime(2026, 11, 16, 6, 0, tzinfo=UTC)


@dataclasses.dataclass
class _Stored:
    seq: int
    subject: str
    data: bytes
    headers: dict[str, str] | None
    time: datetime | None


@dataclasses.dataclass
class _State:
    messages: int
    bytes: int
    first_seq: int
    last_seq: int


@dataclasses.dataclass
class _Info:
    config: Any
    state: _State


@dataclasses.dataclass
class _Consumer:
    name: str
    config: ConsumerConfig
    num_pending: int = 0
    num_ack_pending: int = 0


class _Server:
    """one stream's JetStream, as far as a move reads and changes it."""

    def __init__(self, config: StreamConfig, messages: list[_Stored], consumers: list[_Consumer]) -> None:
        self.config = config
        self.messages = messages
        self.consumers = consumers
        self.calls: list[str] = []
        self.published: list[tuple[str, bytes, dict[str, str] | None]] = []
        self.added_consumers: list[ConsumerConfig] = []

    async def stream_info(self, name: str) -> _Info:
        seqs = [m.seq for m in self.messages]
        return _Info(
            self.config,
            _State(
                len(self.messages), sum(len(m.data) for m in self.messages), min(seqs, default=0), max(seqs, default=0)
            ),
        )

    async def consumers_info(self, stream: str, offset: int | None = None) -> list[_Consumer]:
        return list(self.consumers)

    async def get_msg(
        self,
        stream_name: str,
        seq: int | None = None,
        subject: str | None = None,
        direct: bool | None = False,
        next: bool | None = False,
    ) -> _Stored:
        return next_or_raise([m for m in self.messages if m.seq >= (seq or 0)])

    async def delete_stream(self, name: str) -> bool:
        self.calls.append("delete")
        self.messages, self.consumers = [], []
        return True

    async def add_stream(self, config: Any = None, **params: Any) -> Any:
        self.calls.append("add")
        self.config = config
        return object()

    async def add_consumer(self, stream: str, config: Any = None, **params: Any) -> Any:
        self.calls.append("consumer")
        self.added_consumers.append(config)
        return object()

    async def publish(
        self,
        subject: str,
        payload: bytes = b"",
        timeout: float | None = None,
        stream: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> Any:
        self.calls.append("publish")
        self.published.append((subject, payload, headers))
        return object()


def next_or_raise(found: list[_Stored]) -> _Stored:
    if not found:
        raise AssertionError("read past the stream's last message")
    return found[0]


def _audit(*, storage: StorageType = StorageType.FILE) -> StreamConfig:
    return StreamConfig(
        name="ns-audit", subjects=["ns.audit.>", "ns.audit-deadletter"], storage=storage, num_replicas=3
    )


def _live_consumer(**counts: int) -> _Consumer:
    return _Consumer(
        "unified-audit-consumer",
        ConsumerConfig(durable_name="unified-audit-consumer", filter_subject="ns.audit.>"),
        **counts,
    )


def _audit_messages() -> list[_Stored]:
    return [
        _Stored(1, "ns.audit.tool.call", b"read", None, _NOW - timedelta(days=1)),
        _Stored(
            2,
            "ns.audit-deadletter",
            b"parked-old",
            {"Threetears-Original-Subject": "ns.audit.x"},
            _NOW - timedelta(days=30),
        ),
        _Stored(
            3,
            "ns.audit-deadletter",
            b"parked",
            {"Threetears-Original-Subject": "ns.audit.y", "Nats-Msg-Id": "m"},
            _NOW - timedelta(hours=2),
        ),
        _Stored(4, "ns.audit.rbac.group.create", b"read too", None, _NOW),
    ]


@pytest.mark.asyncio
async def test_a_plain_stream_keeps_only_the_parked_messages_inside_its_new_age_and_its_consumers() -> None:
    server = _Server(_audit(), _audit_messages(), [_live_consumer()])

    moved = await stream_move(server, "ns-audit", replicas=3, max_age_seconds=7 * 86400.0, max_bytes=1 << 20, now=_NOW)

    assert moved.moved and moved.copied == 1
    assert dict(moved.skipped) == {"consumed": 2, "older than max_age": 1}
    assert server.calls == ["delete", "add", "consumer", "publish"], "the order loses or duplicates something"
    assert (server.config.storage, server.config.num_replicas, server.config.max_age, server.config.max_bytes) == (
        StorageType.MEMORY,
        3,
        7 * 86400.0,
        1 << 20,
    )
    # the dead letter keeps the subject it was authorised on; the publish id of the old stream is dropped
    assert server.published == [("ns.audit-deadletter", b"parked", {"Threetears-Original-Subject": "ns.audit.y"})]
    assert [c.durable_name for c in server.added_consumers] == ["unified-audit-consumer"]


@pytest.mark.asyncio
async def test_a_move_is_refused_while_a_consumer_still_owes_a_message() -> None:
    server = _Server(_audit(), _audit_messages(), [_live_consumer(num_ack_pending=2)])

    plan = await stream_move_plan(server, "ns-audit", replicas=3, now=_NOW)
    assert plan.refused is not None and "unified-audit-consumer 0/2" in plan.refused
    with pytest.raises(StreamMoveRefused):
        await stream_move(server, "ns-audit", replicas=3, now=_NOW)
    assert server.calls == [], "a refused move changed the stream"


@pytest.mark.asyncio
async def test_a_stream_already_on_memory_is_not_moved() -> None:
    server = _Server(_audit(storage=StorageType.MEMORY), _audit_messages(), [_live_consumer()])

    plan = await stream_move_plan(server, "ns-audit", replicas=3, now=_NOW)

    assert plan.refused == "it is on memory storage already"


@pytest.mark.asyncio
async def test_a_kv_bucket_keeps_the_latest_value_of_each_key_and_drops_markers() -> None:
    config = StreamConfig(
        name="KV_tool_catalog", subjects=["$KV.tool_catalog.>"], storage=StorageType.FILE, num_replicas=1
    )
    messages = [
        _Stored(10, "$KV.tool_catalog.a", b"A", None, _NOW - timedelta(days=90)),
        _Stored(12, "$KV.tool_catalog.b", b"", {"KV-Operation": "DEL"}, _NOW),
        _Stored(15, "$KV.tool_catalog.c", b"C", None, _NOW),
    ]
    watcher = _Consumer("kw_1", ConsumerConfig(filter_subject="$KV.tool_catalog.>"))
    server = _Server(config, messages, [watcher])

    moved = await stream_move(server, "KV_tool_catalog", replicas=3, now=_NOW)

    # a KV value is copied whatever its age and whoever watches it
    assert [(s, d) for s, d, _h in server.published] == [("$KV.tool_catalog.a", b"A"), ("$KV.tool_catalog.c", b"C")]
    assert dict(moved.skipped) == {"kv marker": 1}
    assert moved.ephemeral_consumers == ("kw_1",) and server.added_consumers == []
    assert server.config.num_replicas == 3, "a bucket on one server was not raised to three"


def test_the_streams_wanted_are_the_ones_on_file_storage() -> None:
    listed = [_Info(_audit(), _State(0, 0, 0, 0)), _Info(_audit(storage=StorageType.MEMORY), _State(0, 0, 0, 0))]
    listed[1].config = dataclasses.replace(listed[1].config, name="ns-other")

    assert moves_wanted(listed) == ["ns-audit"]
