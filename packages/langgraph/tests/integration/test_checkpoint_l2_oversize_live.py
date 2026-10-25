"""Integration: a checkpoint too big for NATS KV leaves no older checkpoint cached for another worker.

The production failure (2026-10-23): a turn's checkpoint was over the NATS server's
``max_payload``, the KV put failed, and the saver's L2 key -- per thread, not per checkpoint --
kept the previous checkpoint. A second worker whose L1 missed read that from L2 and never
reached Postgres. Here two savers stand for two workers: they share one real NATS KV bucket and
one real Postgres schema, and neither has an L1, so the second reads only what the tiers hold.

Uses the session-scoped ``nats_container`` and ``db_container`` fixtures; a checkout without
docker skips cleanly.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from threetears.core.data.migrations import MigrationRunner
from threetears.langgraph.checkpoint import ThreeTierCheckpointSaver
from threetears.langgraph.checkpoint_scope import CheckpointScope
from threetears.langgraph.migrations import register as register_langgraph
from threetears.langgraph.protocols import (
    AsyncpgPoolAdapter,
    CheckpointL2Cache,
    CheckpointL2RevisionCache,
    CheckpointL2ValueLimit,
)
from threetears.nats import NatsClient, set_default_namespace

pytestmark = pytest.mark.integration

_UNSCOPED = CheckpointScope.unscoped(reason="integration test of the un-tenanted keyspace")

_BUCKET = "ckpt"

#: over the 1 MB ``max_payload`` an untuned NATS server advertises, which the test server is.
_OVERSIZED_VALUE = "x" * 1_200_000


class _MigrationStore:
    """the two calls the migration runner makes, over one asyncpg connection."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        """wrap ``conn``.

        :param conn: connection whose search_path selects the test schema
        :ptype conn: asyncpg.Connection
        """
        self.conn = conn

    async def execute(self, sql: str, *params: Any) -> str:
        """run a statement.

        :param sql: SQL text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: status tag
        :rtype: str
        """
        result: str = await self.conn.execute(sql, *params)
        return result

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """fetch rows as dicts.

        :param sql: SQL text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: rows
        :rtype: list[dict[str, Any]]
        """
        return [dict(r) for r in await self.conn.fetch(sql, *params)]


class _NatsKvL2Cache(CheckpointL2Cache):
    """the checkpoint L2 over a NatsClient's KV bucket, the shape a host wires (no stated limit)."""

    def __init__(self, nc: NatsClient) -> None:
        """capture the client.

        :param nc: connected client
        :ptype nc: NatsClient
        """
        self.nc = nc

    async def get(self, bucket: str, key: str) -> bytes | None:
        """read a value.

        :param bucket: bucket suffix
        :ptype bucket: str
        :param key: cache key
        :ptype key: str
        :return: value or None
        :rtype: bytes | None
        """
        return await (await self.nc.kv_bucket(name=bucket)).get(key=key)

    async def put(self, bucket: str, key: str, value: bytes) -> None:
        """write a value.

        :param bucket: bucket suffix
        :ptype bucket: str
        :param key: cache key
        :ptype key: str
        :param value: bytes to store
        :ptype value: bytes
        :return: nothing
        :rtype: None
        """
        await (await self.nc.kv_bucket(name=bucket)).put(key=key, value=value)

    async def delete(self, bucket: str, key: str) -> None:
        """delete a key.

        :param bucket: bucket suffix
        :ptype bucket: str
        :param key: cache key
        :ptype key: str
        :return: nothing
        :rtype: None
        """
        await (await self.nc.kv_bucket(name=bucket)).delete(key=key)


class _FencedNatsKvL2Cache(_NatsKvL2Cache, CheckpointL2RevisionCache, CheckpointL2ValueLimit):
    """the same cache with the bucket's revision-fenced write and the server's ``max_payload`` as its limit."""

    async def get_latest(self, bucket: str, key: str) -> tuple[bytes | None, int]:
        """the value and the key's latest revision, a deletion included.

        :param bucket: bucket suffix
        :ptype bucket: str
        :param key: cache key
        :ptype key: str
        :return: value or None, and the revision
        :rtype: tuple[bytes | None, int]
        """
        return await (await self.nc.kv_bucket(name=bucket)).get_latest(key=key)

    async def update(self, bucket: str, key: str, value: bytes, revision: int) -> int | None:
        """write only while the key's latest revision is ``revision``.

        :param bucket: bucket suffix
        :ptype bucket: str
        :param key: cache key
        :ptype key: str
        :param value: bytes to store
        :ptype value: bytes
        :param revision: expected revision
        :ptype revision: int
        :return: the new revision, or None when the key changed
        :rtype: int | None
        """
        return await (await self.nc.kv_bucket(name=bucket)).update(key=key, value=value, revision=revision)

    @property
    def max_value_bytes(self) -> int | None:
        """the connected server's ``max_payload``.

        :return: the limit, or None while disconnected
        :rtype: int | None
        """
        return self.nc.max_payload


class _InterleavingExecutor:
    """the real executor, running ``between`` after the first latest-checkpoint query and before it answers.

    Makes the refill race deterministic: the reader's query has read the older checkpoint, a writer
    lands a newer one in Postgres and L2, and only then does the reader go on to refill.
    """

    def __init__(self, inner: AsyncpgPoolAdapter) -> None:
        """wrap ``inner``.

        :param inner: the real executor
        :ptype inner: AsyncpgPoolAdapter
        """
        self.inner = inner
        self.between: Any = None

    async def fetch(self, query: str, *args: object) -> list[dict[str, Any]]:
        """delegate.

        :param query: SQL
        :ptype query: str
        :param args: parameters
        :ptype args: object
        :return: rows
        :rtype: list[dict[str, Any]]
        """
        return await self.inner.fetch(query, *args)

    async def fetchrow(self, query: str, *args: object) -> dict[str, Any] | None:
        """delegate, then run ``between`` once if this was the latest-checkpoint query.

        :param query: SQL
        :ptype query: str
        :param args: parameters
        :ptype args: object
        :return: the row the real query returned
        :rtype: dict[str, Any] | None
        """
        row = await self.inner.fetchrow(query, *args)
        if self.between is not None and "ORDER BY checkpoint_id DESC" in query:
            between, self.between = self.between, None
            await between()
        return row

    async def execute(self, query: str, *args: object) -> str:
        """delegate.

        :param query: SQL
        :ptype query: str
        :param args: parameters
        :ptype args: object
        :return: status tag
        :rtype: str
        """
        return await self.inner.execute(query, *args)


@pytest.fixture
async def executor(db_container: str) -> AsyncIterator[AsyncpgPoolAdapter]:
    schema = f"ckpt_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        runner = MigrationRunner()
        register_langgraph(runner)
        await runner.apply_for_agent_schema(_MigrationStore(conn))  # type: ignore[arg-type]
    finally:
        await conn.close()
    pool = await asyncpg.create_pool(db_container, min_size=1, max_size=2, server_settings={"search_path": schema})
    assert pool is not None
    try:
        yield AsyncpgPoolAdapter(pool)
    finally:
        await pool.close()
        admin = await asyncpg.connect(db_container)
        try:
            await admin.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await admin.close()


@pytest.fixture
async def nc(nats_container: str) -> AsyncIterator[NatsClient]:
    namespace = f"ckpt{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="ckpt-oversize"
    ) as client:
        yield client


def _checkpoint(checkpoint_id: str, value: str = "small") -> dict[str, Any]:
    """a LangGraph checkpoint carrying ``value`` in one channel.

    :param checkpoint_id: the checkpoint's id
    :ptype checkpoint_id: str
    :param value: the channel's value; its size sets the blob's
    :ptype value: str
    :return: the checkpoint mapping
    :rtype: dict[str, Any]
    """
    return {
        "id": checkpoint_id,
        "ts": "2026-10-23T06:06:55Z",
        "channel_values": {"state": value},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }


_CONFIG: dict[str, Any] = {"configurable": {"thread_id": "t-1", "checkpoint_ns": ""}}


@pytest.mark.parametrize("cache", [_FencedNatsKvL2Cache, _NatsKvL2Cache], ids=["limit-stated", "put-refused"])
async def test_the_next_worker_reads_the_oversized_checkpoint_from_postgres(
    executor: AsyncpgPoolAdapter, nc: NatsClient, cache: type[_NatsKvL2Cache]
) -> None:
    """stated or not, the limit leaves no older checkpoint under the key: the next worker gets the new one."""
    writer = ThreeTierCheckpointSaver(executor, l2_cache=cache(nc), l2_bucket=_BUCKET, scope=_UNSCOPED)
    await writer.aput(_CONFIG, _checkpoint("cp-1"), {}, {})
    bucket = await nc.kv_bucket(name=_BUCKET)
    assert await bucket.get(key="t-1") is not None, "the small checkpoint never reached L2"

    await writer.aput(_CONFIG, _checkpoint("cp-2", _OVERSIZED_VALUE), {}, {})

    reader = ThreeTierCheckpointSaver(executor, l2_cache=cache(nc), l2_bucket=_BUCKET, scope=_UNSCOPED)
    tup = await reader.aget_tuple(_CONFIG)
    assert tup is not None
    assert tup.checkpoint["id"] == "cp-2", "the next worker was served the older checkpoint from L2"
    assert tup.checkpoint["channel_values"]["state"] == _OVERSIZED_VALUE
    assert await bucket.get(key="t-1") is None, "the reader's warm cached a blob the bucket cannot hold"


async def test_a_read_pinned_to_an_older_checkpoint_leaves_the_latest_cached(
    executor: AsyncpgPoolAdapter, nc: NatsClient
) -> None:
    writer = ThreeTierCheckpointSaver(executor, l2_cache=_FencedNatsKvL2Cache(nc), l2_bucket=_BUCKET, scope=_UNSCOPED)
    first = await writer.aput(_CONFIG, _checkpoint("cp-1"), {}, {})
    await writer.aput(first, _checkpoint("cp-2"), {}, {})

    pinned_reader = ThreeTierCheckpointSaver(
        executor, l2_cache=_FencedNatsKvL2Cache(nc), l2_bucket=_BUCKET, scope=_UNSCOPED
    )
    pinned = await pinned_reader.aget_tuple({"configurable": {**_CONFIG["configurable"], "checkpoint_id": "cp-1"}})
    assert pinned is not None and pinned.checkpoint["id"] == "cp-1"

    cached = await (await nc.kv_bucket(name=_BUCKET)).get(key="t-1")
    assert cached is not None
    assert writer.deserialize_checkpoint_tuple(cached)["checkpoint"]["id"] == "cp-2"
    latest_reader = ThreeTierCheckpointSaver(
        executor, l2_cache=_FencedNatsKvL2Cache(nc), l2_bucket=_BUCKET, scope=_UNSCOPED
    )
    latest = await latest_reader.aget_tuple(_CONFIG)
    assert latest is not None and latest.checkpoint["id"] == "cp-2"


async def test_an_older_refill_cannot_replace_a_newer_checkpoint(executor: AsyncpgPoolAdapter, nc: NatsClient) -> None:
    """the race: a reader's query reads cp-1, a writer lands cp-2, then the reader refills; cp-2 must stand."""
    writer = ThreeTierCheckpointSaver(executor, l2_cache=_FencedNatsKvL2Cache(nc), l2_bucket=_BUCKET, scope=_UNSCOPED)
    first = await writer.aput(_CONFIG, _checkpoint("cp-1"), {}, {})
    bucket = await nc.kv_bucket(name=_BUCKET)
    await bucket.delete(key="t-1")  # an evicted entry: the reader misses L2 and reads Postgres

    interleaving = _InterleavingExecutor(executor)

    async def _a_newer_checkpoint_lands() -> None:
        await writer.aput(first, _checkpoint("cp-2"), {}, {})

    interleaving.between = _a_newer_checkpoint_lands
    reader = ThreeTierCheckpointSaver(
        interleaving,  # type: ignore[arg-type]
        l2_cache=_FencedNatsKvL2Cache(nc),
        l2_bucket=_BUCKET,
        scope=_UNSCOPED,
    )
    read = await reader.aget_tuple(_CONFIG)
    assert read is not None and read.checkpoint["id"] == "cp-1", "the interleaving did not happen"

    fresh = ThreeTierCheckpointSaver(executor, l2_cache=_FencedNatsKvL2Cache(nc), l2_bucket=_BUCKET, scope=_UNSCOPED)
    latest = await fresh.aget_tuple(_CONFIG)
    assert latest is not None and latest.checkpoint["id"] == "cp-2", "the older refill replaced the newer checkpoint"
    cached = await bucket.get(key="t-1")
    assert cached is not None and writer.deserialize_checkpoint_tuple(cached)["checkpoint"]["id"] == "cp-2"
