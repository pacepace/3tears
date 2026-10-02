"""a cache-bypassing write evicts every row it touched from every tier, to completion, however it ended.

``BaseCollection.bypassing_write`` is the one owner of the rule for a targeted UPDATE that skips
``save_entity`` on a table ``get`` serves: on the collection's own pool, every touched key is
evicted from L1 and L2 and the eviction broadcast once the body ends -- raised, cancelled or
returned -- shielded so a cancellation cannot stop it partway; joined to a caller's connection,
the keys are enrolled in the enclosing :class:`CallerTransaction` and settled when it ends, and a
connection no ``CallerTransaction`` opened is refused before anything is written.

Asserted on the invalidation broadcast, which is what reaches every other replica, not on a local
re-read.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from types import TracebackType
from typing import Any

import pytest

from threetears.core.collections import CallerTransaction
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.registry import CacheInvalidationMessage, CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.kv import FakeNatsClient

_TABLE = "bypassed_rows"


class _Row(BaseEntity):
    primary_key_field = "id"


class _Rows(BaseCollection[_Row]):
    """a collection whose own L3 is never reached; the tests drive only the eviction."""

    @property
    def table_name(self) -> str:
        return _TABLE

    @property
    def entity_class(self) -> type[_Row]:
        return _Row

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        del entity_id
        return None

    async def save_to_store(
        self, data: dict[str, Any], original_timestamp: datetime | None = None, *, conn: Any = None
    ) -> int:
        del data, original_timestamp, conn
        return 1

    async def delete_from_store(self, entity_id: Any) -> None:
        del entity_id

    def serialize(self, data: dict[str, Any]) -> bytes:
        return json.dumps(data, default=str).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        row: dict[str, Any] = json.loads(data)
        return row


class _SuspendingNats(FakeNatsClient):
    """a NATS client whose first publish suspends until released, where a broker acknowledgement would."""

    def __init__(self) -> None:
        super().__init__()
        self.first_publish_started = asyncio.Event()
        self.release = asyncio.Event()

    async def publish(self, *, subject: Any, message: Any, reply_to: Any = None) -> None:
        if not self.first_publish_started.is_set():
            self.first_publish_started.set()
            await self.release.wait()
        await super().publish(subject=subject, message=message, reply_to=reply_to)


class _Transaction:
    async def __aenter__(self) -> _Transaction:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> bool:
        return False


class _Conn:
    def transaction(self) -> _Transaction:
        return _Transaction()


def _rows(nats: FakeNatsClient) -> _Rows:
    registry = CollectionRegistry()
    registry.configure(l3_pool=object(), kv_key_scope="bypass-principal")  # type: ignore[arg-type]
    return _Rows(registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""), nats_client=nats)  # type: ignore[arg-type]


def _evicted(nats: FakeNatsClient) -> list[str]:
    return [
        message.ids[0]
        for message in nats.published
        if isinstance(message, CacheInvalidationMessage) and message.table == _TABLE
    ]


class TestOnTheCollectionsPool:
    async def test_a_write_that_raises_still_evicts_every_row(self) -> None:
        nats = FakeNatsClient()
        rows = _rows(nats)

        with pytest.raises(ConnectionError):
            async with rows.bypassing_write("a", "b"):
                raise ConnectionError("connection lost after the statement was sent")

        assert sorted(_evicted(nats)) == ["a", "b"]

    async def test_rows_named_during_the_write_are_evicted_with_the_rest(self) -> None:
        nats = FakeNatsClient()
        rows = _rows(nats)

        async with rows.bypassing_write("a") as write:
            write.touches("b", "c")

        assert sorted(_evicted(nats)) == ["a", "b", "c"]

    async def test_a_cancellation_cannot_stop_the_eviction_partway(self) -> None:
        nats = _SuspendingNats()
        rows = _rows(nats)

        async def _write() -> None:
            async with rows.bypassing_write("a", "b", "c"):
                pass

        task = asyncio.create_task(_write())
        await asyncio.wait_for(nats.first_publish_started.wait(), timeout=5.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        nats.release.set()
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        await asyncio.gather(*pending)

        assert sorted(_evicted(nats)) == ["a", "b", "c"]

    async def test_a_write_known_to_have_changed_nothing_evicts_nothing(self) -> None:
        nats = FakeNatsClient()
        rows = _rows(nats)

        async with rows.bypassing_write("a") as write:
            write.unchanged()

        assert _evicted(nats) == []

    async def test_unchanged_does_not_excuse_a_write_that_then_raised(self) -> None:
        nats = FakeNatsClient()
        rows = _rows(nats)

        with pytest.raises(RuntimeError):
            async with rows.bypassing_write("a") as write:
                write.unchanged()
                raise RuntimeError("the outcome is unknown again")

        assert _evicted(nats) == ["a"]


class TestJoinedToACallersTransaction:
    async def test_a_connection_no_caller_transaction_opened_is_refused_before_the_write(self) -> None:
        rows = _rows(FakeNatsClient())
        ran = False

        with pytest.raises(ValueError, match="CallerTransaction"):
            async with rows.bypassing_write("a", conn=_Conn()):
                ran = True

        assert not ran

    async def test_the_rows_are_settled_when_the_transaction_ends_not_before(self) -> None:
        nats = FakeNatsClient()
        rows = _rows(nats)
        conn = _Conn()

        async with CallerTransaction(conn):
            async with rows.bypassing_write("a", conn=conn) as write:
                write.touches("b")
            assert _evicted(nats) == [], "a row joined to an open transaction was settled before it ended"

        assert sorted(_evicted(nats)) == ["a", "b"]


class TestJoiningNamesTheWriter:
    def test_join_refuses_a_bare_connection_naming_the_writer(self) -> None:
        with pytest.raises(ValueError, match=r"Rows\.resume\(conn=\.\.\.\) needs the connection's transaction"):
            CallerTransaction.join(_Conn(), writer="Rows.resume")

    async def test_join_finds_the_open_transaction(self) -> None:
        conn = _Conn()
        async with CallerTransaction(conn) as transaction:
            assert CallerTransaction.join(conn, writer="Rows.resume") is transaction
