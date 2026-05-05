"""multi-pod integration test: two listeners stay coherent on bumps.

verifies the full push-plus-pull design end-to-end against a real
Postgres + NATS testcontainer pair:

- happy path: a writer bumps :class:`EpochClient.bump`; both
  subscribed listeners receive the broadcast and advance their
  last-seen.
- pull-on-stale: with one listener's NATS subscription deliberately
  unsubscribed BEFORE a bump, the missed-broadcast listener still
  catches up via :meth:`EpochListener.catch_up` (the periodic-tick
  shape) and via :meth:`EpochListener.echo` (the per-message-echo
  shape). proves Postgres is the source of truth and a missed
  broadcast does not leak forward.
- monotonicity under contention: 50 random bumps from two writers
  yield strictly-monotonic last-seen on every listener.

requires docker; gated by ``pytest.mark.integration``.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from threetears.core.data.migrations import MigrationRunner
from threetears.epoch import EpochClient, EpochListener
from threetears.epoch.migrations import register as register_epoch
from threetears.nats import NatsClient, set_default_namespace
from threetears.nats.subjects import Subject

pytestmark = pytest.mark.integration


class _AsyncpgStore:
    """DataStore-shape wrapper for migration-runner integration tests.

    mirrors :class:`tests.integration.conftest._AsyncpgStore` from
    every other 3tears integration suite. one connection per test
    function.

    :param conn: asyncpg connection with search_path pre-set
    :ptype conn: asyncpg.Connection
    """

    def __init__(self, conn: asyncpg.Connection) -> None:
        """capture connection.

        :param conn: connection with search_path bound
        :ptype conn: asyncpg.Connection
        """
        self._conn = conn

    async def execute(self, sql: str, *params: Any) -> str:
        """proxy execute through asyncpg."""
        result: str = await self._conn.execute(sql, *params)
        return result

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """proxy fetch through asyncpg, return as dicts."""
        rows = await self._conn.fetch(sql, *params)
        return [dict(r) for r in rows]


@pytest.fixture
async def pg_schema(db_container: str) -> AsyncIterator[tuple[str, str]]:
    """create a fresh schema, run the epoch migration, yield (url, schema).

    each test gets a clean schema with only the ``config_epochs``
    table provisioned. teardown drops the schema.
    """
    schema = f"epoch_it_{id(object())}".lower().replace("-", "_")
    conn = await asyncpg.connect(db_container)
    try:
        await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        runner = MigrationRunner()
        register_epoch(runner)
        store = _AsyncpgStore(conn)
        count = await runner.apply_for_platform_schema(store)  # type: ignore[arg-type]
        assert count == 1, f"expected 1 epoch migration, applied {count}"
    finally:
        await conn.close()

    yield (db_container, schema)

    conn = await asyncpg.connect(db_container)
    try:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    finally:
        await conn.close()


@pytest.fixture
async def pg_pool(pg_schema: tuple[str, str]) -> AsyncIterator[asyncpg.Pool]:
    """asyncpg pool whose connections inherit the schema's search_path.

    the ``server_settings`` kwarg sets search_path on every new
    connection so :class:`EpochClient` queries hit the per-test
    schema's ``config_epochs`` table.
    """
    url, schema = pg_schema
    pool = await asyncpg.create_pool(
        dsn=url,
        server_settings={"search_path": f"{schema}"},
        min_size=1,
        max_size=3,
    )
    assert pool is not None
    try:
        yield pool
    finally:
        await pool.close()


def _subject() -> Subject:
    """canonical test subject."""
    return Subject(path="itest.epoch.unit", kind="point")


async def _connect_pod(nats_url: str, name: str) -> NatsClient:
    """connect one pod's NatsClient bound to the test namespace."""
    return await NatsClient.connect(
        nats_url=nats_url,
        nats_subject_namespace="itest",
        client_name=name,
    )


@pytest.mark.asyncio
async def test_two_pods_receive_bump_via_broadcast(
    pg_pool: asyncpg.Pool,
    nats_container: str,
) -> None:
    """happy path: bump from a writer fires both listeners' callbacks."""
    set_default_namespace("itest")

    async with await _connect_pod(nats_container, "writer") as writer_nc, \
            await _connect_pod(nats_container, "pod-a") as pod_a_nc, \
            await _connect_pod(nats_container, "pod-b") as pod_b_nc:

        writer = EpochClient(pg_pool, writer_nc)
        listener_a = EpochListener(pod_a_nc, EpochClient(pg_pool, pod_a_nc))
        listener_b = EpochListener(pod_b_nc, EpochClient(pg_pool, pod_b_nc))

        cb_a_calls: list[int] = []
        cb_b_calls: list[int] = []

        async def cb_a(epoch: int, _payload: dict[str, object] | None) -> None:
            cb_a_calls.append(epoch)

        async def cb_b(epoch: int, _payload: dict[str, object] | None) -> None:
            cb_b_calls.append(epoch)

        subject = _subject()
        await listener_a.subscribe(subject, cb_a)
        await listener_b.subscribe(subject, cb_b)
        # let subscriptions register server-side before publishing
        await pod_a_nc.flush()
        await pod_b_nc.flush()

        new_epoch = await writer.bump(subject, payload={"hint": "first"})
        assert new_epoch == 1

        # let dispatch fire
        await asyncio.sleep(0.5)

        assert cb_a_calls == [1]
        assert cb_b_calls == [1]
        assert listener_a.last_seen(subject) == 1
        assert listener_b.last_seen(subject) == 1


@pytest.mark.asyncio
async def test_pull_on_stale_recovers_missed_broadcast(
    pg_pool: asyncpg.Pool,
    nats_container: str,
) -> None:
    """pod B never sees the broadcast; catches up on the next pull.

    proves the durable Postgres write is the recovery mechanism. a
    missed broadcast becomes a < 1-tick delay before any consumer
    notices the staleness and reloads.
    """
    set_default_namespace("itest")

    async with await _connect_pod(nats_container, "writer") as writer_nc, \
            await _connect_pod(nats_container, "pod-b-no-sub") as pod_b_nc:

        writer = EpochClient(pg_pool, writer_nc)
        listener_b = EpochListener(pod_b_nc, EpochClient(pg_pool, pod_b_nc))

        subject = _subject()
        cb_b_calls: list[int] = []

        async def cb_b(epoch: int, _payload: dict[str, object] | None) -> None:
            cb_b_calls.append(epoch)

        # deliberately DO NOT subscribe pod B to the broadcast subject.
        # prime its last-seen manually as if cold-started after an outage.
        primed = await listener_b._epoch_client.current(subject)  # noqa: SLF001
        listener_b._last_seen[subject.path] = primed  # noqa: SLF001
        assert primed == 0

        # writer bumps; broadcast goes nowhere reachable to pod B.
        await writer.bump(subject)
        await writer.bump(subject)

        assert cb_b_calls == []  # broadcast missed

        # next periodic catch-up tick discovers the stale state.
        result = await listener_b.catch_up(subject, cb_b)
        assert result == 2
        assert cb_b_calls == [2]
        assert listener_b.last_seen(subject) == 2

        # subsequent catch-up at same epoch is a no-op.
        cb_b_calls.clear()
        result = await listener_b.catch_up(subject, cb_b)
        assert result == 2
        assert cb_b_calls == []


@pytest.mark.asyncio
async def test_per_message_echo_recovers_missed_broadcast(
    pg_pool: asyncpg.Pool,
    nats_container: str,
) -> None:
    """per-message echo path is the second recovery channel after pull-tick.

    consumer receives a response whose envelope echoes a higher
    epoch; :meth:`EpochListener.echo` confirms via L3 and fires.
    """
    set_default_namespace("itest")

    async with await _connect_pod(nats_container, "writer") as writer_nc, \
            await _connect_pod(nats_container, "pod") as pod_nc:

        writer = EpochClient(pg_pool, writer_nc)
        listener = EpochListener(pod_nc, EpochClient(pg_pool, pod_nc))

        subject = _subject()
        cb_calls: list[int] = []

        async def cb(epoch: int, _payload: dict[str, object] | None) -> None:
            cb_calls.append(epoch)

        # deliberately do NOT subscribe; rely on the echo path entirely.
        listener._last_seen[subject.path] = 0  # noqa: SLF001

        new_epoch = await writer.bump(subject)
        assert new_epoch == 1

        await listener.echo(subject, echoed_epoch=1, on_bump=cb)

        assert cb_calls == [1]
        assert listener.last_seen(subject) == 1


@pytest.mark.asyncio
async def test_monotonicity_under_concurrent_writers(
    pg_pool: asyncpg.Pool,
    nats_container: str,
) -> None:
    """50 interleaved bumps from two writers yield strict monotonicity.

    proves the row-lock serialization of ``ON CONFLICT DO UPDATE
    SET epoch = epoch + 1`` plus the listener's dedupe-on-equal
    discipline together produce a monotonically-increasing sequence
    on every subscriber.
    """
    set_default_namespace("itest")

    async with await _connect_pod(nats_container, "writer-1") as w1_nc, \
            await _connect_pod(nats_container, "writer-2") as w2_nc, \
            await _connect_pod(nats_container, "pod") as pod_nc:

        w1 = EpochClient(pg_pool, w1_nc)
        w2 = EpochClient(pg_pool, w2_nc)
        listener = EpochListener(pod_nc, EpochClient(pg_pool, pod_nc))

        subject = _subject()
        observed: list[int] = []

        async def cb(epoch: int, _payload: dict[str, object] | None) -> None:
            observed.append(epoch)

        await listener.subscribe(subject, cb)
        await pod_nc.flush()

        rng = random.Random(42)
        async def _flurry(client: EpochClient, count: int) -> None:
            """fire `count` bumps with random small jitter."""
            for _ in range(count):
                await client.bump(subject)
                await asyncio.sleep(rng.uniform(0, 0.005))

        await asyncio.gather(_flurry(w1, 25), _flurry(w2, 25))
        # let any in-flight broadcasts drain
        await asyncio.sleep(0.5)

        # final last-seen must equal 50 (all bumps committed durably)
        durable = await listener._epoch_client.current(subject)  # noqa: SLF001
        assert durable == 50

        # observed deliveries must be strictly monotonic (some may have
        # been deduped or merged via gap-jump; the contract is no
        # decreases, never that every value 1..50 was seen).
        assert observed, "listener saw no broadcasts at all"
        assert all(b > a for a, b in zip(observed, observed[1:], strict=False))
        assert observed[-1] == 50
        assert listener.last_seen(subject) == 50
