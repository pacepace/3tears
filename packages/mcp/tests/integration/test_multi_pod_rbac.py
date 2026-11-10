"""multi-pod RBAC propagation integration test for the MCP framework.

closes spec MCP-11's hard success criterion ("Multi-pod test:
revoking a grant on pod A propagates to pod B in < 5 seconds via
the ``mcp.rbac`` epoch broadcast"). without this end-to-end test
the cross-pod RBAC promise is asserted only at unit-test level
(LocalGrantAuthorizer + EpochListener tested with mocks); the
testcontainer pair here proves the wiring stays coherent against
real Postgres + NATS.

shape mirrors :mod:`threetears.epoch.tests.integration.test_multi_pod`:

- two :class:`LocalGrantAuthorizer` instances against shared
  Postgres + NATS, wired through real :class:`EpochClient` /
  :class:`EpochListener`. each pod has its own
  :class:`McpToolGrantCollection` reading from the same row store.
- a writer pod calls :meth:`McpToolGrantCollection.add_grant` then
  bumps :func:`Subjects.mcp_rbac_epoch`; the receiver pod's
  authorizer reloads its in-memory cache and ``allows`` returns
  True for the new grant.
- a revoke path covers the inverse: ``remove_grant`` then bump;
  receiver's ``allows`` flips to False.
- each bump names its grant and the receiver reads that one grant by
  key; every grant is read only by the cold start and an operator reset
  (``McpToolGrantCollection.load_all_grants``), which a test here proves.
- a missed-broadcast path covers the periodic catch-up tick: the
  receiver's subscription never delivers the broadcast, and the
  authorizer's own catch-up tick finds the higher epoch. The grants are in
  Postgres; the EPOCH is a NATS KV counter that keeps no record of what a
  bump changed -- ``mcp.rbac.epoch`` is not in the durable tile family --
  so the grant is unknown: the receiver refuses non-admin checks, reads
  nothing whole, and an operator reset fills it again.
- speed: the cold-start fill and one bump's keyed apply, each held to a
  multiple of the bare read on the same database.

requires docker; gated by ``pytest.mark.integration``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import asyncpg
import pytest

from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner
from threetears.core.testing.calibrate import best_seconds, budget_problem, plain_read_seconds
from threetears.epoch import BumpCallback, EpochClient, EpochListener, ResetCallback
from threetears.epoch.migrations import register as register_epoch
from threetears.mcp import (
    Identity,
    LocalGrantAuthorizer,
    McpToolGrantCollection,
)
from threetears.mcp.auth import GrantCacheUnavailableError
from threetears.mcp.migrations import register as register_mcp
from threetears.nats import NatsClient, Subject, Subjects, set_default_namespace

pytestmark = pytest.mark.integration


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


async def _connect_pod(nats_url: str, name: str) -> NatsClient:
    """connect one pod's NatsClient bound to the test namespace.

    matches the pattern from :mod:`threetears.epoch.tests.integration
    .test_multi_pod` so the propagation behaviour observed here is
    structurally identical to the epoch package's own test.
    """
    return await NatsClient.connect(
        nats_url=nats_url,
        nats_subject_namespace="itest",
        client_name=name,
    )


def _build_collection(pool: asyncpg.Pool) -> McpToolGrantCollection:
    """construct a :class:`McpToolGrantCollection` against ``pool``.

    the registry is minimal: only the L3 pool is wired (no L1, no NATS-
    KV). load_all_grants reads via ``self.l3_pool`` directly; the test
    exercises the public surface only.
    """
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    return McpToolGrantCollection(
        registry,
        DefaultCoreConfig(),
        None,
        None,
    )


async def _build_started_authorizer(
    *,
    pool: asyncpg.Pool,
    nats_client: NatsClient,
    catchup_interval_seconds: float = 3600.0,
) -> tuple[LocalGrantAuthorizer, McpToolGrantCollection, EpochListener]:
    """wire EpochClient/EpochListener + collection + authorizer; start authorizer.

    returns the authorizer for ``allows`` calls, plus the collection
    and listener for tests that need to mutate / probe directly.
    catchup_interval defaults to 1h so the periodic tick never fires
    during a test (deterministic) -- tests that exercise the catch-up
    path call ``listener.catch_up(...)`` explicitly.
    """
    epoch_client = EpochClient(pool, nats_client)
    epoch_listener = EpochListener(nats_client, epoch_client)
    collection = _build_collection(pool)
    authorizer = LocalGrantAuthorizer(
        grant_loader=collection.load_all_grants,
        grant_reader=collection.read_grant,
        epoch_client=epoch_client,
        epoch_listener=epoch_listener,
        catchup_interval_seconds=catchup_interval_seconds,
    )
    await authorizer.start()
    return authorizer, collection, epoch_listener


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grant_added_on_pod_a_propagates_to_pod_b(
    pg_pool: asyncpg.Pool,
    nats_container: str,
) -> None:
    """add_grant on pod A; pod B's allows returns True within ~2 seconds.

    this is the spec MCP-11 hard success criterion: cross-pod
    RBAC propagation via the ``mcp.rbac`` epoch broadcast.
    """
    set_default_namespace("itest")
    async with (
        await _connect_pod(nats_container, "pod-a") as pod_a_nc,
        await _connect_pod(nats_container, "pod-b") as pod_b_nc,
    ):
        pod_a, pod_a_collection, _ = await _build_started_authorizer(
            pool=pg_pool,
            nats_client=pod_a_nc,
        )
        pod_b, _, _ = await _build_started_authorizer(
            pool=pg_pool,
            nats_client=pod_b_nc,
        )
        try:
            principal_id = uuid4()
            permission = "mcp.test.read"
            identity = Identity(
                principal_type="user",
                principal_id=principal_id,
            )

            # baseline: neither pod allows the grant.
            assert await pod_a.allows(identity, permission) is False
            assert await pod_b.allows(identity, permission) is False

            # pod A adds + bumps. EpochClient.bump publishes
            # mcp.rbac broadcast; pod B's listener receives it
            # and the authorizer reads the named grant from L3 by key.
            grant_entity = await pod_a_collection.add_grant(
                principal_type="user",
                principal_id=principal_id,
                tool_name="probe",
                permission=permission,
            )
            # bump is the caller's responsibility (matches the REST
            # endpoint pattern, payload included); here the test does it explicitly.
            await EpochClient(pg_pool, pod_a_nc).bump(
                Subjects.mcp_rbac_epoch(),
                payload={"grant_id": str(grant_entity.grant_id), "action": "create"},
            )

            # poll for propagation (NATS dispatch is async). 2s is
            # well under the spec's 5s budget.
            for _ in range(20):
                await asyncio.sleep(0.1)
                if await pod_b.allows(identity, permission):
                    break
            assert await pod_b.allows(identity, permission) is True
            assert await pod_a.allows(identity, permission) is True
        finally:
            await pod_a.stop()
            await pod_b.stop()


@pytest.mark.asyncio
async def test_grant_removed_on_pod_a_propagates_to_pod_b(
    pg_pool: asyncpg.Pool,
    nats_container: str,
) -> None:
    """remove_grant on pod A; pod B's allows flips False within ~2 seconds.

    revocation propagation is the security-critical inverse: a
    revoked operator must lose access on every pod, fast.
    """
    set_default_namespace("itest")
    async with (
        await _connect_pod(nats_container, "pod-a") as pod_a_nc,
        await _connect_pod(nats_container, "pod-b") as pod_b_nc,
    ):
        pod_a, pod_a_collection, _ = await _build_started_authorizer(
            pool=pg_pool,
            nats_client=pod_a_nc,
        )
        pod_b, _, _ = await _build_started_authorizer(
            pool=pg_pool,
            nats_client=pod_b_nc,
        )
        try:
            principal_id = uuid4()
            permission = "mcp.test.read"
            identity = Identity(
                principal_type="user",
                principal_id=principal_id,
            )

            # seed: a grant exists, both pods allow.
            grant_entity = await pod_a_collection.add_grant(
                principal_type="user",
                principal_id=principal_id,
                tool_name="probe",
                permission=permission,
            )
            await EpochClient(pg_pool, pod_a_nc).bump(
                Subjects.mcp_rbac_epoch(),
                payload={"grant_id": str(grant_entity.grant_id), "action": "create"},
            )
            for _ in range(20):
                await asyncio.sleep(0.1)
                if await pod_b.allows(identity, permission):
                    break
            assert await pod_b.allows(identity, permission) is True

            # revoke + bump.
            removed = await pod_a_collection.remove_grant(grant_entity.grant_id)
            assert removed is True
            await EpochClient(pg_pool, pod_a_nc).bump(
                Subjects.mcp_rbac_epoch(),
                payload={"grant_id": str(grant_entity.grant_id), "action": "delete"},
            )

            # poll for revocation.
            for _ in range(20):
                await asyncio.sleep(0.1)
                if not await pod_b.allows(identity, permission):
                    break
            assert await pod_b.allows(identity, permission) is False
            assert await pod_a.allows(identity, permission) is False
        finally:
            await pod_a.stop()
            await pod_b.stop()


class _BroadcastDroppingNats:
    """a pod's NATS client whose subscriptions never receive anything.

    every call is forwarded to the real connected client except
    ``subscribe_typed``, which accepts the registration and drops it --
    the "every broadcast was lost" condition (subscriber blip, dropped
    message) made deterministic. KV reads still reach the real broker, so
    the epoch counter the catch-up tick reads is the real one.
    """

    def __init__(self, inner: NatsClient) -> None:
        """wrap a connected client.

        :param inner: the pod's real NATS client
        :ptype inner: NatsClient
        :return: nothing
        :rtype: None
        """
        self.inner = inner

    async def subscribe_typed(self, **_kwargs: Any) -> None:
        """accept the subscription and never deliver to it.

        :return: nothing
        :rtype: None
        """

    def __getattr__(self, name: str) -> Any:
        """forward everything else to the real client.

        :param name: attribute name
        :ptype name: str
        :return: the real client's attribute
        :rtype: Any
        """
        return getattr(self.inner, name)


@pytest.mark.asyncio
async def test_missed_broadcast_refuses_until_an_operator_reset_and_never_reads_every_grant(
    pg_pool: asyncpg.Pool,
    nats_container: str,
) -> None:
    """grant added during a NATS outage; pod B's tick finds the epoch, and its grant is unknown.

    pod B is started normally -- subscribe, prime, catch-up loop -- but its
    subscription is wired to a client that never delivers. pod A mutates +
    bumps; pod B's catch-up tick sees the higher epoch on the KV counter
    after one interval's grace. nothing durable says which grant that epoch
    changed, so pod B refuses non-admin checks rather than answer from a
    cache that may hold a revoked grant -- and reads nothing whole. the
    operator reset fills it again. epochs are compared relative to the
    counter's value at the start, because the broker, and with it the
    counter for this subject, is shared by the session.
    """
    set_default_namespace("itest")
    async with (
        await _connect_pod(nats_container, "pod-a") as pod_a_nc,
        await _connect_pod(nats_container, "pod-b") as pod_b_nc,
    ):
        pod_a_epoch_client = EpochClient(pg_pool, pod_a_nc)
        pod_a_collection = _build_collection(pg_pool)

        pod_b_epoch_client = EpochClient(pg_pool, pod_b_nc)
        pod_b_listener = EpochListener(_BroadcastDroppingNats(pod_b_nc), pod_b_epoch_client)  # type: ignore[arg-type]
        pod_b_collection = _build_collection(pg_pool)
        fills = _Counting(pod_b_collection.load_all_grants)
        pod_b_authorizer = LocalGrantAuthorizer(
            grant_loader=fills,
            grant_reader=pod_b_collection.read_grant,
            epoch_client=pod_b_epoch_client,
            epoch_listener=pod_b_listener,
            catchup_interval_seconds=0.05,
        )

        principal_id = uuid4()
        permission = "mcp.test.read"
        identity = Identity(
            principal_type="user",
            principal_id=principal_id,
        )

        epoch_before = await pod_a_epoch_client.current(Subjects.mcp_rbac_epoch())
        await pod_b_authorizer.start()
        try:
            # baseline: cache empty, deny.
            assert await pod_b_authorizer.allows(identity, permission) is False

            # pod A adds + bumps. the broadcast goes out; pod B's
            # subscription never delivers it.
            grant_entity = await pod_a_collection.add_grant(
                principal_type="user",
                principal_id=principal_id,
                tool_name="probe",
                permission=permission,
            )
            new_epoch = await pod_a_epoch_client.bump(
                Subjects.mcp_rbac_epoch(),
                payload={"grant_id": str(grant_entity.grant_id), "action": "create"},
            )
            assert new_epoch == epoch_before + 1

            # the catch-up tick gives the broadcast one interval, then advances last_seen:
            # the epoch's grant is unknown, so non-admin checks are refused.
            for _ in range(40):
                if pod_b_listener.last_seen(Subjects.mcp_rbac_epoch()) >= new_epoch:
                    break
                await asyncio.sleep(0.05)
            assert pod_b_listener.last_seen(Subjects.mcp_rbac_epoch()) >= new_epoch
            with pytest.raises(GrantCacheUnavailableError, match="unheard"):
                await pod_b_authorizer.allows(identity, permission)
            assert fills.calls == 1, "a lost broadcast read every grant again"

            # the operator reset: the one fill after the cold start.
            await pod_b_listener.signal_reset()
            assert fills.calls == 2
            assert await pod_b_authorizer.allows(identity, permission) is True
        finally:
            await pod_b_authorizer.stop()


class _Counting:
    """a grant loader that counts its calls, so a test can say when every grant was read."""

    def __init__(self, inner: Callable[[], Awaitable[list[dict[str, Any]]]]) -> None:
        """wrap the loader.

        :param inner: the loader
        :ptype inner: Callable[[], Awaitable[list[dict[str, Any]]]]
        :return: nothing
        :rtype: None
        """
        self.inner = inner
        self.calls = 0

    async def __call__(self) -> list[dict[str, Any]]:
        """count, then load.

        :return: the rows
        :rtype: list[dict[str, Any]]
        """
        self.calls += 1
        return await self.inner()


class _CapturingListener(EpochListener):
    """a real listener that also hands its test the bump callback, to time one bump's apply without NATS."""

    on_bump: BumpCallback | None = None

    async def subscribe(
        self,
        subject: Subject,
        on_bump: BumpCallback,
        primed_epoch: int | None = None,
        on_reset: ResetCallback | None = None,
    ) -> None:
        """record the callback, then subscribe.

        :param subject: the subject
        :ptype subject: Subject
        :param on_bump: the bump callback
        :ptype on_bump: BumpCallback
        :param primed_epoch: as the listener takes it
        :ptype primed_epoch: int | None
        :param on_reset: as the listener takes it
        :ptype on_reset: ResetCallback | None
        :return: nothing
        :rtype: None
        """
        self.on_bump = on_bump
        await super().subscribe(subject, on_bump, primed_epoch=primed_epoch, on_reset=on_reset)


@pytest.mark.asyncio
async def test_load_all_grants_is_reached_only_at_the_cold_start_and_an_operator_reset(
    pg_pool: asyncpg.Pool,
    nats_container: str,
) -> None:
    """every grant is read to fill the empty cache at start, and after an operator reset -- never for a bump.

    pod A adds a grant, revokes it, and bumps each time with the grant named; pod B applies each by
    key. a bump naming no grant leaves its grant unknown, and is not read around either. only
    ``signal_reset`` -- the counter replaced -- reaches ``load_all_grants`` again.
    """
    set_default_namespace("itest")
    async with (
        await _connect_pod(nats_container, "pod-a") as pod_a_nc,
        await _connect_pod(nats_container, "pod-b") as pod_b_nc,
    ):
        pod_a_epoch_client = EpochClient(pg_pool, pod_a_nc)
        pod_a_collection = _build_collection(pg_pool)
        pod_b_epoch_client = EpochClient(pg_pool, pod_b_nc)
        pod_b_listener = EpochListener(pod_b_nc, pod_b_epoch_client)
        pod_b_collection = _build_collection(pg_pool)
        fills = _Counting(pod_b_collection.load_all_grants)
        pod_b = LocalGrantAuthorizer(
            grant_loader=fills,
            grant_reader=pod_b_collection.read_grant,
            epoch_client=pod_b_epoch_client,
            epoch_listener=pod_b_listener,
            catchup_interval_seconds=0.05,
        )
        identity = Identity(principal_type="user", principal_id=uuid4())
        permission = "mcp.test.read"
        await pod_b.start()
        try:
            assert fills.calls == 1, "the cold start fills the cache once"
            grant = await pod_a_collection.add_grant(
                principal_type="user", principal_id=identity.principal_id, tool_name="probe", permission=permission
            )
            await pod_a_epoch_client.bump(
                Subjects.mcp_rbac_epoch(), payload={"grant_id": str(grant.grant_id), "action": "create"}
            )
            for _ in range(40):
                if await pod_b.allows(identity, permission):
                    break
                await asyncio.sleep(0.05)
            assert await pod_b.allows(identity, permission) is True
            await pod_a_collection.remove_grant(grant.grant_id)
            await pod_a_epoch_client.bump(
                Subjects.mcp_rbac_epoch(), payload={"grant_id": str(grant.grant_id), "action": "delete"}
            )
            for _ in range(40):
                if not await pod_b.allows(identity, permission):
                    break
                await asyncio.sleep(0.05)
            assert await pod_b.allows(identity, permission) is False
            # a few catch-up ticks pass with nothing unheard
            await asyncio.sleep(0.3)
            assert fills.calls == 1, "a bump or a catch-up tick read every grant"

            await pod_a_epoch_client.bump(Subjects.mcp_rbac_epoch(), payload=None)
            for _ in range(40):
                try:
                    await pod_b.allows(identity, permission)
                except GrantCacheUnavailableError:
                    break
                await asyncio.sleep(0.05)
            with pytest.raises(GrantCacheUnavailableError):
                await pod_b.allows(identity, permission)
            assert fills.calls == 1, "a bump naming no grant read every grant"

            await pod_b_listener.signal_reset()
            assert fills.calls == 2, "the operator reset fills the cache"
            assert await pod_b.allows(identity, permission) is False
        finally:
            await pod_b.stop()


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
    collection = _build_collection(pg_pool)
    async with pg_pool.acquire() as conn:
        baseline = await plain_read_seconds(conn, f"SELECT {_GRANT_COLUMNS} FROM mcp_tool_grants")  # noqa: S608
    took = await best_seconds(collection.load_all_grants)
    print(f"load_all_grants {took * 1000:.1f} ms, bare {baseline * 1000:.1f} ms, {took / baseline:.2f}x")
    problem = budget_problem(f"load_all_grants of {_SPEED_GRANTS} grants", took, baseline)
    assert problem is None, problem


@pytest.mark.asyncio
async def test_a_bump_applies_its_grant_as_fast_as_a_bare_keyed_read(
    pg_pool: asyncpg.Pool,
    nats_container: str,
) -> None:
    """one bump's apply -- the keyed read and the cache update -- against a bare ``WHERE grant_id = $1``."""
    set_default_namespace("itest")
    keys = await _seed_grants(pg_pool, _SPEED_GRANTS)
    async with await _connect_pod(nats_container, "pod-speed") as nc:
        epoch_client = EpochClient(pg_pool, nc)
        listener = _CapturingListener(nc, epoch_client)
        collection = _build_collection(pg_pool)
        authorizer = LocalGrantAuthorizer(
            grant_loader=collection.load_all_grants,
            grant_reader=collection.read_grant,
            epoch_client=epoch_client,
            epoch_listener=listener,
            catchup_interval_seconds=3600.0,
        )
        await authorizer.start()
        try:
            on_bump = listener.on_bump
            assert on_bump is not None
            epoch = [await epoch_client.current(Subjects.mcp_rbac_epoch())]
            target = keys[len(keys) // 2]

            async def _apply_one() -> None:
                epoch[0] += 1
                await on_bump(epoch[0], {"grant_id": str(target), "action": "create"})

            async def _bare() -> None:
                await pg_pool.fetchrow(
                    f"SELECT {_GRANT_COLUMNS} FROM mcp_tool_grants WHERE grant_id = $1",  # noqa: S608
                    target,
                )

            baseline = await best_seconds(_bare, runs=50)
            took = await best_seconds(_apply_one, runs=50)
            print(f"bump apply {took * 1e6:.0f} us, bare keyed read {baseline * 1e6:.0f} us, {took / baseline:.2f}x")
            problem = budget_problem("one bump's keyed apply", took, baseline)
            assert problem is None, problem
            assert await authorizer.allows(Identity(principal_type="user", principal_id=uuid4()), "x") is False
        finally:
            await authorizer.stop()
