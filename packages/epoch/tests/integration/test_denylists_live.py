"""Integration: the durable denylists against a real broker, database and epoch bucket.

What a fake cannot prove: a standing revocation and a spent redemption both survive the broker
restart that used to forget them, with the real epoch-backed write generation deciding when a
recorded absence stops answering. Lives in the epoch package because it needs
``EpochGenerationSource``, which core cannot import (epoch depends on core, not the reverse).

Uses the session-scoped ``db_container`` and ``nats_container`` fixtures; a checkout without
docker skips cleanly.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import asyncpg
import pytest

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.coordination.migrations.v001_create_coordination_tables import create_coordination_tables
from threetears.core.coordination.revocation import RedemptionLedger, RevocationGuard
from threetears.core.data.store import DataStore
from threetears.epoch import EpochGenerationSource
from threetears.nats import NatsClient, set_default_namespace

pytestmark = pytest.mark.integration

_SCOPE = "live-denylist"


async def _migrated_pool(db_url: str) -> asyncpg.Pool:
    """a fresh schema carrying the coordination tables, and a pool bound to it.

    :param db_url: the container's connection URL
    :ptype db_url: str
    :return: the pool
    :rtype: asyncpg.Pool
    """
    schema = f"denylist_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(db_url)
    try:
        await admin.execute(f"CREATE SCHEMA {schema}")
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(db_url, min_size=1, max_size=4, server_settings={"search_path": schema})
    assert pool is not None
    migration_registry = CollectionRegistry()
    migration_registry.configure(l3_pool=SqlL3Backend(pool))
    await create_coordination_tables(DataStore(uuid.uuid4(), migration_registry))
    return pool


def _registry(nats: NatsClient, pool: asyncpg.Pool) -> CollectionRegistry:
    """one replica's registry, with the real epoch-backed write generation."""
    l1 = SQLiteBackend(db_name=f"live_denylist_{uuid.uuid4().hex[:8]}")
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l2_client=nats, l3_pool=SqlL3Backend(pool), kv_key_scope=_SCOPE)
    registry.set_generation_source(EpochGenerationSource(nats))
    return registry


async def test_a_revocation_and_a_redemption_survive_a_broker_wipe(db_container: str, nats_container: str) -> None:
    namespace = f"coord{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    pool = await _migrated_pool(db_container)
    try:
        async with await NatsClient.connect(
            nats_url=nats_container, nats_subject_namespace=namespace, client_name="denylist"
        ) as nats:
            registry = _registry(nats, pool)
            guard = RevocationGuard(registry, purpose="standing", ttl_seconds=3600)
            ledger = RedemptionLedger(registry, purpose="refresh_jti", ttl_seconds=3600)
            revoked_at = datetime.now(UTC)
            await guard.record_revocation("sub:p1", revoked_at=revoked_at)
            assert await ledger.record_unique("jti-1") is True

            # both write L3 before returning, so nothing is buffered and the wipe is the whole test
            js = nats.jetstream_context()
            await js.delete_stream(f"KV_{namespace}-collections")

            fresh_registry = _registry(nats, pool)
            fresh_guard = RevocationGuard(fresh_registry, purpose="standing", ttl_seconds=3600)
            fresh_ledger = RedemptionLedger(fresh_registry, purpose="refresh_jti", ttl_seconds=3600)
            assert await fresh_guard.is_revoked_before("sub:p1", moment=revoked_at - timedelta(minutes=5)) is True, (
                "a wipe made a revoked session valid again"
            )
            assert await fresh_ledger.record_unique("jti-1") is False, "a wipe made a spent token spendable"
            assert await fresh_ledger.record_unique("jti-2") is True, "a wipe refused an unspent token"
    finally:
        await pool.close()
