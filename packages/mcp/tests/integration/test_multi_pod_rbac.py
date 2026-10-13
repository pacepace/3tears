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
- a missed-broadcast path covers the periodic catch-up tick: the
  receiver's subscription never delivers the broadcast, but the
  authorizer's own catch-up tick discovers the higher epoch and
  reloads the cache. The grants are in Postgres; the EPOCH
  is a NATS KV counter -- ``mcp.rbac.epoch`` is not in the durable
  tile family.

requires docker; gated by ``pytest.mark.integration``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import asyncpg
import pytest

from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner
from threetears.epoch import EpochClient, EpochListener
from threetears.epoch.migrations import register as register_epoch
from threetears.mcp import (
    Identity,
    LocalGrantAuthorizer,
    McpToolGrantCollection,
)
from threetears.mcp.migrations import register as register_mcp
from threetears.nats import NatsClient, Subjects, set_default_namespace

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
            # and the authorizer reloads the cache from L3.
            await pod_a_collection.add_grant(
                principal_type="user",
                principal_id=principal_id,
                tool_name="probe",
                permission=permission,
            )
            # bump is the caller's responsibility (matches the REST
            # endpoint pattern); here the test does it explicitly.
            await EpochClient(pg_pool, pod_a_nc).bump(
                Subjects.mcp_rbac_epoch(),
                payload={"grant_id": "test-grant", "action": "create"},
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
async def test_missed_broadcast_recovers_via_catchup(
    pg_pool: asyncpg.Pool,
    nats_container: str,
) -> None:
    """grant added during a NATS outage; pod B catches up via the periodic tick.

    proves the safety net: even if every NATS broadcast dropped
    (subscriber blip, JetStream redelivery edge), the authorizer's own
    periodic catch-up tick reads the epoch counter and reloads.

    deterministic simulation: pod B is started normally -- subscribe,
    prime, catch-up loop -- but its subscription is wired to a client
    that never delivers, so the broadcast cannot be what recovers it.
    pod A mutates + bumps; pod B is stale until its catch-up tick sees
    the higher epoch on the KV counter and reloads. epochs are compared
    relative to the counter's value at the start, because the broker,
    and with it the counter for this subject, is shared by the session.
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
        pod_b_authorizer = LocalGrantAuthorizer(
            grant_loader=pod_b_collection.load_all_grants,
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

            # the catch-up tick sees current > last_seen, advances
            # last_seen, and reloads the cache -> grant visible.
            for _ in range(40):
                if await pod_b_authorizer.allows(identity, permission):
                    break
                await asyncio.sleep(0.05)
            assert await pod_b_authorizer.allows(identity, permission) is True
            assert pod_b_listener.last_seen(Subjects.mcp_rbac_epoch()) >= new_epoch
        finally:
            await pod_b_authorizer.stop()
