"""multi-pod RBAC propagation integration test for the MCP framework.

closes spec MCP-11's hard success criterion ("Multi-pod test: revoking a grant on pod A propagates
to pod B in < 5 seconds"), against real Postgres + NATS.

``mcp_tool_grants`` is a switched-on table: every committed grant write advances its write generation
and records the grant it changed in the epoch change log. each pod's :class:`LocalGrantAuthorizer` is
a cache derived from the table on that pod's registry, and follows it:

- a grant written on pod A reaches pod B as a row broadcast, and pod B reads that one grant by key;
- a broadcast pod B never received -- its listener down for the write -- is found by the generation
  walk and read by key, with no restart and no refusal in between; the same holds for a write whose
  ``mcp.rbac`` bump named no grant, since the authorizer no longer reads bumps at all;
- every grant is read only by the cold start and a whole-table change
  (``McpToolGrantCollection.load_all_grants``), which a test here proves;
- speed: the cold-start fill, and one walked change's apply, each held to a multiple of the bare read
  on the same database.

requires docker; gated by ``pytest.mark.integration``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import asyncpg
import pytest

from threetears.core.collections.epoch_change import ChangedRow, EpochChange
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner
from threetears.core.testing.calibrate import best_seconds, budget_problem, plain_read_seconds
from threetears.epoch import EpochClient, EpochGenerationReader, EpochGenerationSource
from threetears.epoch.migrations import register as register_epoch
from threetears.mcp import (
    Identity,
    LocalGrantAuthorizer,
    McpToolGrantCollection,
)
from threetears.mcp.migrations import register as register_mcp
from threetears.nats import NatsClient, Subjects, set_default_namespace

pytestmark = pytest.mark.integration

#: how long a test waits for something the broker delivers asynchronously
_DELIVERY_SECONDS = 10.0


# ---------------------------------------------------------------------
# Migration-runner store wrapper (matches every other 3tears
# integration suite; centralizing this would cut paste-load but the
# duplication is shallow and stable across tests)
# ---------------------------------------------------------------------


class _AsyncpgStore:
    """DataStore-shape wrapper for migration-runner integration tests."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        """capture connection bound to the target schema.

        :param conn: connection with search_path set
        :ptype conn: asyncpg.Connection
        :return: nothing
        :rtype: None
        """
        self._conn = conn

    async def execute(self, sql: str, *params: Any) -> str:
        """proxy execute."""
        result: str = await self._conn.execute(sql, *params)
        return result

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """proxy fetch -> dict rows."""
        rows = await self._conn.fetch(sql, *params)
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
async def pg_schema(db_container: str) -> AsyncIterator[tuple[str, str]]:
    """fresh schema with epoch + mcp migrations applied; teardown drops.

    yields ``(url, schema)``. tests pass the URL to ``asyncpg.create_pool``
    with ``server_settings={"search_path": schema}`` so every connection
    inherits the schema bind.
    """
    schema = f"mcp_it_{id(object())}".lower().replace("-", "_")
    conn = await asyncpg.connect(db_container)
    try:
        await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        runner = MigrationRunner()
        register_epoch(runner)
        register_mcp(runner)
        store = _AsyncpgStore(conn)
        count = await runner.apply_for_platform_schema(store)  # type: ignore[arg-type]
        # epoch v001 + v002 (previous_epoch, da9e89fb) + mcp v001 = 3 migrations.
        assert count == 3, f"expected 3 platform migrations, applied {count}"
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
    """asyncpg pool whose connections inherit the schema bind."""
    url, schema = pg_schema
    pool = await asyncpg.create_pool(
        dsn=url,
        server_settings={"search_path": schema},
        min_size=1,
        max_size=4,
    )
    assert pool is not None
    try:
        yield pool
    finally:
        await pool.close()


class _Counting:
    """a grant loader or reader that counts its calls, so a test can say when every grant was read."""

    def __init__(self, inner: Callable[..., Awaitable[Any]]) -> None:
        """wrap the call.

        :param inner: the loader or reader
        :ptype inner: Callable[..., Awaitable[Any]]
        :return: nothing
        :rtype: None
        """
        self.inner = inner
        self.calls = 0
        self.landed = asyncio.Event()

    async def __call__(self, *args: Any) -> Any:
        """count, then call through.

        :param args: the call's arguments
        :ptype args: Any
        :return: what the inner call returns
        :rtype: Any
        """
        self.calls += 1
        try:
            return await self.inner(*args)
        finally:
            self.landed.set()


class _Pod:
    """one pod: its own NATS connection, registry, listener, collection and authorizer; the shared broker and L3."""

    def __init__(self, nc: NatsClient, pool: asyncpg.Pool, *, scope: str) -> None:
        self.nc = nc
        self.registry = CollectionRegistry()
        self.registry.configure(l3_pool=pool, kv_key_scope=scope)
        self.source = EpochGenerationSource(nc)
        self.registry.set_generation_source(self.source)
        self.grants = McpToolGrantCollection(self.registry, DefaultCoreConfig(), nc)
        self.fills = _Counting(self.grants.load_all_grants)
        self.reads = _Counting(self.grants.read_grant)
        self.authorizer = LocalGrantAuthorizer(
            grant_loader=self.fills,
            grant_reader=self.reads,
            registry=self.registry,
            generation_reader=EpochGenerationReader(nc),
        )

    async def listen(self) -> None:
        await self.registry.start_invalidation_listener(self.nc)

    async def go_deaf(self) -> None:
        """the pod's subscription is gone: every row broadcast from here is lost to it."""
        await self.registry.stop_invalidation_listener()

    async def close(self) -> None:
        await self.authorizer.stop()
        await self.registry.stop_invalidation_listener()


@asynccontextmanager
async def _pods(url: str, pool: asyncpg.Pool) -> AsyncIterator[tuple[_Pod, _Pod]]:
    """a writer and a follower in a namespace of their own, both listening and both authorizers started."""
    namespace = f"mcpg{uuid4().hex[:6]}"
    set_default_namespace(namespace)
    async with (
        await NatsClient.connect(nats_url=url, nats_subject_namespace=namespace, client_name="pod-a") as a_nc,
        await NatsClient.connect(nats_url=url, nats_subject_namespace=namespace, client_name="pod-b") as b_nc,
    ):
        pod_a, pod_b = _Pod(a_nc, pool, scope="pod-a"), _Pod(b_nc, pool, scope="pod-b")
        try:
            # the namespace's change log is declared before anyone reads it, as the hub's keeper does at its start
            await pod_a.source.start_log()
            for pod in (pod_a, pod_b):
                await pod.listen()
                await pod.authorizer.start()
            yield pod_a, pod_b
        finally:
            await pod_a.close()
            await pod_b.close()


async def _until_answers(pod: _Pod, identity: Identity, permission: str, expected: bool) -> int:
    """wait until ``pod`` answers ``expected``, checking every 20 ms; every check must answer, never refuse.

    :param pod: the pod
    :ptype pod: _Pod
    :param identity: who asks
    :ptype identity: Identity
    :param permission: what for
    :ptype permission: str
    :param expected: the answer waited for
    :ptype expected: bool
    :return: how many checks were made
    :rtype: int
    :raises AssertionError: when it does not answer so within :data:`_DELIVERY_SECONDS`
    :raises GrantCacheUnavailableError: when a check is refused -- the steady state never fails closed
    """
    deadline = time.monotonic() + _DELIVERY_SECONDS
    checks = 0
    while True:
        checks += 1
        if await pod.authorizer.allows(identity, permission) is expected:
            return checks
        if time.monotonic() > deadline:
            raise AssertionError(f"pod never answered {expected} for {permission}")
        await asyncio.sleep(0.02)


_PERMISSION = "mcp.test.read"


async def _grant(pod: _Pod, identity: Identity) -> UUID:
    entity = await pod.grants.add_grant(
        principal_type="user", principal_id=identity.principal_id, tool_name="probe", permission=_PERMISSION
    )
    return UUID(str(entity.grant_id))


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_grant_added_on_pod_a_reaches_pod_b_by_key(pg_pool: asyncpg.Pool, nats_container: str) -> None:
    async with _pods(nats_container, pg_pool) as (pod_a, pod_b):
        identity = Identity(principal_type="user", principal_id=uuid4())
        assert await pod_b.authorizer.allows(identity, _PERMISSION) is False
        grant_id = await _grant(pod_a, identity)
        await _until_answers(pod_b, identity, _PERMISSION, True)
        await _until_answers(pod_a, identity, _PERMISSION, True)
        assert pod_b.fills.calls == 1, "a changed grant read every grant"
        assert pod_b.reads.calls >= 1
        assert grant_id


@pytest.mark.asyncio
async def test_a_grant_removed_on_pod_a_is_revoked_on_pod_b(pg_pool: asyncpg.Pool, nats_container: str) -> None:
    """revocation is the security-critical direction: a revoked operator loses access on every pod, fast."""
    async with _pods(nats_container, pg_pool) as (pod_a, pod_b):
        identity = Identity(principal_type="user", principal_id=uuid4())
        grant_id = await _grant(pod_a, identity)
        await _until_answers(pod_b, identity, _PERMISSION, True)
        assert await pod_a.grants.remove_grant(grant_id) is True
        await _until_answers(pod_b, identity, _PERMISSION, False)
        await _until_answers(pod_a, identity, _PERMISSION, False)
        assert pod_b.fills.calls == 1


@pytest.mark.asyncio
async def test_a_lost_broadcast_converges_by_key_with_no_restart_and_no_refusal(
    pg_pool: asyncpg.Pool, nats_container: str
) -> None:
    """pod B hears nothing of the revocation; the generation walk finds it and pod B reads it by key.

    every check in between answers from what pod B holds -- never a refusal -- and nothing reads every grant.
    """
    async with _pods(nats_container, pg_pool) as (pod_a, pod_b):
        identity = Identity(principal_type="user", principal_id=uuid4())
        grant_id = await _grant(pod_a, identity)
        await _until_answers(pod_b, identity, _PERMISSION, True)
        reads_before = pod_b.reads.calls

        await pod_b.go_deaf()
        assert await pod_a.grants.remove_grant(grant_id) is True
        await _until_answers(pod_b, identity, _PERMISSION, False)

        assert pod_b.reads.calls > reads_before, "the walk read the revoked grant by key"
        assert pod_b.fills.calls == 1, "a lost broadcast read every grant again"


@pytest.mark.asyncio
async def test_a_write_whose_bump_named_no_grant_is_applied_from_the_walk(
    pg_pool: asyncpg.Pool, nats_container: str
) -> None:
    """the rbac bump says nothing the follow needs: a bump naming no grant, broadcast lost, still converges."""
    async with _pods(nats_container, pg_pool) as (pod_a, pod_b):
        identity = Identity(principal_type="user", principal_id=uuid4())
        await pod_b.go_deaf()
        await _grant(pod_a, identity)
        await EpochClient(pg_pool, pod_a.nc).bump(Subjects.mcp_rbac_epoch(), payload=None)
        await _until_answers(pod_b, identity, _PERMISSION, True)
        assert pod_b.fills.calls == 1


@pytest.mark.asyncio
async def test_load_all_grants_is_reached_only_at_the_cold_start_and_a_whole_table_change(
    pg_pool: asyncpg.Pool, nats_container: str
) -> None:
    """every grant is read to fill the empty cache at start, and when the whole table changed -- never for a grant.

    pod A adds, revokes and adds again; pod B applies each by key, heard or walked. only a whole-table
    change -- a truncate or a load, which drops the table here -- reaches ``load_all_grants`` again.
    """
    async with _pods(nats_container, pg_pool) as (pod_a, pod_b):
        assert pod_b.fills.calls == 1, "the cold start fills the cache once"
        identity = Identity(principal_type="user", principal_id=uuid4())
        first = await _grant(pod_a, identity)
        await _until_answers(pod_b, identity, _PERMISSION, True)
        await pod_a.grants.remove_grant(first)
        await _until_answers(pod_b, identity, _PERMISSION, False)
        await pod_b.go_deaf()
        await _grant(pod_a, identity)
        await _until_answers(pod_b, identity, _PERMISSION, True)
        assert pod_b.fills.calls == 1, "a changed grant, heard or walked, read every grant"

        pod_b.registry.drop_table("mcp_tool_grants", reason="a truncate")
        deadline = time.monotonic() + _DELIVERY_SECONDS
        while pod_b.fills.calls < 2 and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        assert pod_b.fills.calls == 2, "a whole-table change fills the cache again"
        await _until_answers(pod_b, identity, _PERMISSION, True)


#: grants seeded for the speed tests: past any deployment's operator-written grants
_SPEED_GRANTS = 20_000

_GRANT_COLUMNS = "grant_id, principal_type, principal_id, tool_name, permission, date_created"


async def _seed_grants(pool: asyncpg.Pool, count: int) -> list[UUID]:
    """``count`` grants, written as one COPY.

    :param pool: the pool
    :ptype pool: asyncpg.Pool
    :param count: how many
    :ptype count: int
    :return: their keys
    :rtype: list[UUID]
    """
    now = datetime.now(UTC)
    rows = [(uuid4(), "user", uuid4(), f"tool_{i % 50}", f"mcp.speed.{i % 200}", now) for i in range(count)]
    async with pool.acquire() as conn:
        await conn.copy_records_to_table(
            "mcp_tool_grants",
            records=rows,
            columns=["grant_id", "principal_type", "principal_id", "tool_name", "permission", "date_created"],
        )
        await conn.execute("ANALYZE mcp_tool_grants")
    return [row[0] for row in rows]


@pytest.mark.asyncio
async def test_load_all_grants_speed_at_real_size(pg_pool: asyncpg.Pool) -> None:
    """the cold-start fill reads the grants as fast as a bare read of the same rows, within the budget."""
    await _seed_grants(pg_pool, _SPEED_GRANTS)
    registry = CollectionRegistry()
    registry.configure(l3_pool=pg_pool)
    collection = McpToolGrantCollection(registry, DefaultCoreConfig(), None)
    async with pg_pool.acquire() as conn:
        baseline = await plain_read_seconds(conn, f"SELECT {_GRANT_COLUMNS} FROM mcp_tool_grants")  # noqa: S608
    took = await best_seconds(collection.load_all_grants)
    print(f"load_all_grants {took * 1000:.1f} ms, bare {baseline * 1000:.1f} ms, {took / baseline:.2f}x")
    problem = budget_problem(f"load_all_grants of {_SPEED_GRANTS} grants", took, baseline)
    assert problem is None, problem


@pytest.mark.asyncio
async def test_a_walked_change_applies_as_fast_as_a_bare_keyed_read(pg_pool: asyncpg.Pool, nats_container: str) -> None:
    """one epoch the walk applies -- its row message, the keyed read, the cache update -- against a bare
    ``WHERE grant_id = $1`` on the same database."""
    keys = await _seed_grants(pg_pool, _SPEED_GRANTS)
    async with _pods(nats_container, pg_pool) as (_pod_a, pod_b):
        target = keys[len(keys) // 2]
        change = EpochChange(table="mcp_tool_grants", token="speed:1", rows=[ChangedRow(ids=[str(target)])])

        async def _walked_one() -> None:
            pod_b.reads.landed.clear()
            await pod_b.registry.apply_epoch_change(change)
            await pod_b.reads.landed.wait()

        async def _bare() -> None:
            await pg_pool.fetchrow(
                f"SELECT {_GRANT_COLUMNS} FROM mcp_tool_grants WHERE grant_id = $1",  # noqa: S608
                target,
            )

        baseline = await best_seconds(_bare, runs=50)
        took = await best_seconds(_walked_one, runs=50)
        print(f"walked change {took * 1e6:.0f} us, bare keyed read {baseline * 1e6:.0f} us, {took / baseline:.2f}x")
        problem = budget_problem("one walked change's keyed apply", took, baseline)
        assert problem is None, problem
