"""Integration test: following a collection table's write generation against a real broker.

What a fake cannot prove, proved here with two ``NatsClient`` connections standing in for a writer
pod and a follower pod, each with its own registry and the real cache-invalidation listener, over
the real epoch bucket:

- a row broadcast the follower never received is caught by one catch-up pass;
- a bucket the broker lost is caught by one pass, through the new incarnation the next write mints
  -- a ``FakeKvBucket`` has no stream that can be deleted out from under it;
- a transaction of many rows is one advance, and a follower that received every broadcast drops
  nothing;
- the key watcher is pushed each advance by a real named consumer on the generation key, drops
  nothing for a write whose broadcast arrived, and drops the table for one whose broadcast did not.

Uses the session-scoped ``nats_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, ClassVar

import pytest
from sqlalchemy import Column, DateTime, MetaData, String, Table

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import (
    WRITE_GENERATION,
    BaseCollection,
    CallerTransaction,
    CollectionRegistry,
    GenerationVerdict,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.epoch import (
    EpochGenerationReader,
    EpochGenerationSource,
    follow_generation_key,
    generation_catchup_tick,
)
from threetears.nats import NatsClient, set_default_namespace

pytestmark = pytest.mark.integration

_TABLE = "live_group_members"

#: how long a test waits for something the broker delivers asynchronously.
_DELIVERY_SECONDS = 10.0


def _metadata() -> MetaData:
    metadata = MetaData()
    Table(
        _TABLE,
        metadata,
        Column("id", String(255), primary_key=True),
        Column("member_id", String(255)),
        Column("date_created", DateTime(timezone=True)),
        Column("date_updated", DateTime(timezone=True)),
    )
    return metadata


class _Member(BaseEntity):
    primary_key_field = "id"


class _LiveMembers(BaseCollection[_Member]):
    """a switched-on table over an in-process L3 both pods share."""

    datetime_columns: ClassVar[frozenset[str]] = frozenset({"date_created", "date_updated"})
    write_generation = WRITE_GENERATION
    invalidation_columns: ClassVar[tuple[str, ...]] = ("member_id",)

    def __init__(self, registry: CollectionRegistry, rows: dict[str, dict[str, Any]]) -> None:
        self._rows = rows
        super().__init__(registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""))

    @property
    def table_name(self) -> str:
        return _TABLE

    @property
    def entity_class(self) -> type[_Member]:
        return _Member

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        row = self._rows.get(str(entity_id))
        return dict(row) if row is not None else None

    async def save_to_store(
        self, data: dict[str, Any], original_timestamp: datetime | None = None, *, conn: Any = None
    ) -> int:
        self._rows[str(data["id"])] = dict(data)
        return 1

    async def delete_from_store(self, entity_id: Any) -> None:
        self._rows.pop(str(entity_id), None)

    def serialize(self, data: dict[str, Any]) -> bytes:
        return json.dumps(data, default=str).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        row: dict[str, Any] = json.loads(data)
        return row


class _Pod:
    """one pod: its own L1, registry, listener and reader; the shared broker and L3."""

    def __init__(self, nc: NatsClient, rows: dict[str, dict[str, Any]], *, scope: str) -> None:
        l1 = SQLiteBackend(db_name=f"live_generation_{uuid.uuid4().hex[:8]}")
        l1.initialize(_metadata())
        self.nc = nc
        self.registry = CollectionRegistry()
        self.registry.configure(l1_backend=l1, l2_client=nc, l3_pool=object(), kv_key_scope=scope)  # type: ignore[arg-type]
        self.registry.set_generation_source(EpochGenerationSource(nc))
        self.reader = EpochGenerationReader(nc)
        self.members = _LiveMembers(self.registry, rows)
        # every row message this pod's listener has handled, so a test can wait for one to arrive
        self.heard: list[str] = []
        self.registry.register_derived_cache(
            _TABLE, on_row=lambda message: self.heard.append(message.ids[0]), on_table_dropped=lambda: None
        )

    async def has_heard(self, *entity_ids: str) -> None:
        """wait until this pod's listener has handled a row message for each of ``entity_ids``."""
        await _until(lambda: set(entity_ids) <= set(self.heard), f"the broadcasts for {entity_ids}")

    async def listen(self) -> None:
        await self.registry.start_invalidation_listener(self.nc)

    async def go_deaf(self) -> None:
        await self.registry.stop_invalidation_listener()

    async def tick(self) -> int:
        return await generation_catchup_tick(self.registry, self.reader)

    async def close(self) -> None:
        await self.registry.stop_invalidation_listener()


async def _save(pod: _Pod, entity_id: str, member_id: str) -> None:
    await pod.members.save_entity(pod.members.create({"id": entity_id, "member_id": member_id}))


async def _member_id(pod: _Pod, entity_id: str) -> str | None:
    row = await pod.members.ensure(entity_id)
    return None if row is None else str(row["member_id"])


async def _until(condition: Callable[[], bool], what: str) -> None:
    """wait for something the broker delivers asynchronously; fail rather than wait for ever."""
    deadline = asyncio.get_running_loop().time() + _DELIVERY_SECONDS
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.02)


async def _heard_everything(pod: _Pod) -> None:
    """wait until the follower has received every broadcast up to the table's current generation."""
    token = await pod.reader.read(_TABLE)
    marks = pod.registry.generation_marks
    await _until(lambda: marks.judge(_TABLE, token) is GenerationVerdict.CURRENT, "the follower to hear every row")


@asynccontextmanager
async def _pods(url: str) -> AsyncIterator[tuple[_Pod, _Pod, Callable[[], Any]]]:
    """a writer and a follower in a namespace of their own, the follower listening.

    The third value loses the epoch bucket, as a broker restart on memory storage does.
    """
    namespace = f"genf{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    rows: dict[str, dict[str, Any]] = {}
    async with (
        await NatsClient.connect(nats_url=url, nats_subject_namespace=namespace, client_name="writer") as writer_nc,
        await NatsClient.connect(nats_url=url, nats_subject_namespace=namespace, client_name="follower") as follower_nc,
    ):
        writer, follower = _Pod(writer_nc, rows, scope="writer"), _Pod(follower_nc, rows, scope="follower")
        await follower.listen()
        try:

            async def lose_epoch_bucket() -> None:
                await writer_nc.jetstream_context().delete_stream(f"KV_{namespace}-epochs")

            yield writer, follower, lose_epoch_bucket
        finally:
            await writer.close()
            await follower.close()


class _Conn:
    """a connection whose transaction does nothing, for a CallerTransaction to open."""

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[None]:
        yield

    def transaction(self, **options: Any) -> Any:
        return self._transaction()


async def test_a_broadcast_the_follower_never_received_is_caught_by_one_pass(nats_container: str) -> None:
    async with _pods(nats_container) as (writer, follower, _):
        await _save(writer, "m1", "ada")
        await follower.has_heard("m1")
        follower.registry.follow_generation(_TABLE)
        await follower.tick()  # first sight: the mark is taken
        assert await _member_id(follower, "m1") == "ada"

        # a heard write first, to show the pass is not simply dropping every time
        await _save(writer, "m2", "grace")
        await _heard_everything(follower)
        assert await follower.tick() == 0
        assert follower.members.exists_in_cache_sync("m1")

        # the follower's subscription is gone for exactly one write, and stays gone until the pass
        # has run: resubscribing sooner could still catch a publish the broker had not routed yet.
        await follower.go_deaf()
        await _save(writer, "m1", "edsger")
        assert await _member_id(follower, "m1") == "ada", "nothing told the follower, so it serves what it cached"

        assert await follower.tick() == 1

        assert not follower.members.exists_in_cache_sync("m1")
        assert await _member_id(follower, "m1") == "edsger"
        await follower.listen()
        assert await follower.tick() == 0


async def test_a_bucket_the_broker_lost_is_caught_by_one_pass(nats_container: str) -> None:
    async with _pods(nats_container) as (writer, follower, lose_epoch_bucket):
        await _save(writer, "m1", "ada")
        await follower.has_heard("m1")
        follower.registry.follow_generation(_TABLE)
        await follower.tick()
        assert await _member_id(follower, "m1") == "ada"
        before = follower.registry.generation_marks.recorded(_TABLE)
        assert before is not None

        # the broker loses the bucket, as a restart on memory storage does, and the follower's
        # subscription is gone with it for the write that follows.
        await follower.go_deaf()
        await lose_epoch_bucket()
        await _save(writer, "m1", "edsger")

        assert await follower.tick() == 1

        after = follower.registry.generation_marks.recorded(_TABLE)
        assert after is not None
        assert after.rpartition(":")[0] != before.rpartition(":")[0], "the new bucket minted a new incarnation"
        assert await _member_id(follower, "m1") == "edsger"


async def test_a_transaction_is_one_advance_and_a_follower_that_heard_it_all_drops_nothing(
    nats_container: str,
) -> None:
    async with _pods(nats_container) as (writer, follower, _):
        await _save(writer, "kept", "ada")
        await follower.has_heard("kept")
        follower.registry.follow_generation(_TABLE)
        await follower.tick()
        assert await _member_id(follower, "kept") == "ada"
        before = await follower.reader.read(_TABLE)
        assert before is not None

        conn = _Conn()
        async with CallerTransaction(conn):
            for index in range(8):
                entity = writer.members.create({"id": f"m{index}", "member_id": "grace"})
                await writer.members.save_entity(entity, conn=conn)

        after = await follower.reader.read(_TABLE)
        assert after is not None
        assert int(after.rpartition(":")[2]) == int(before.rpartition(":")[2]) + 1
        await _heard_everything(follower)
        assert await follower.tick() == 0
        assert follower.members.exists_in_cache_sync("kept")


async def test_a_key_watcher_is_pushed_each_advance_and_drops_only_for_a_missed_broadcast(
    nats_container: str,
) -> None:
    async with _pods(nats_container) as (writer, follower, _):
        await _save(writer, "m1", "ada")
        await follower.has_heard("m1")
        marks = follower.registry.generation_marks
        emptied: list[str] = []
        follower.registry.register_derived_cache(
            _TABLE, on_row=lambda message: None, on_table_dropped=lambda: emptied.append("ALL")
        )
        watcher = asyncio.create_task(
            follow_generation_key(follower.registry, follower.reader, _TABLE, grace=timedelta(seconds=2))
        )
        try:
            # the key's latest value is pushed first: this pod's first sight of the table
            await _until(lambda: marks.follows(_TABLE) and marks.recorded(_TABLE) is not None, "the first push")
            assert emptied == ["ALL"]
            assert await _member_id(follower, "m1") == "ada"

            await _save(writer, "m2", "grace")
            await _heard_everything(follower)
            # long enough for the watcher to have judged the push it was sent for that write
            await asyncio.sleep(0.5)
            assert emptied == ["ALL"], "a write whose broadcast arrived dropped the table"
            assert follower.members.exists_in_cache_sync("m1")

            await follower.go_deaf()
            await _save(writer, "m1", "edsger")
            await _until(lambda: len(emptied) == 2, "the watcher to drop the table for the missed broadcast")
            assert await _member_id(follower, "m1") == "edsger"
        finally:
            watcher.cancel()
            with pytest.raises(asyncio.CancelledError):
                await watcher
