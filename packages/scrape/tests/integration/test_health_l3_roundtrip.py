"""Real-Postgres round-trip for ``scrape_target_health``, through the real migrations.

This package's first integration test, and it exists because of a specific failure.
``ScrapeTarget.link_selector`` shipped as a persisted field with no DDL column and every
test in the package stayed green, because ``ScrapeCollection`` falls back to an in-memory
dict when no L3 pool is configured, and a dict has no schema to violate. The bug only
appeared against a real database, which nothing here ever touched.

``test_migrations_drift.py`` closes that gap offline by reading column names back out of
the migration SQL, which is fast and runs everywhere. It is still a check against a
*parse* of the DDL rather than against a database that actually accepted it. This suite
is the other half: apply the real migrations to a real Postgres, then write and read a
real row through the same collection production uses.

Guarded by ``@pytest.mark.integration``. The full sweep (``./scripts/test.sh`` with no
package) passes ``-m "not integration"`` and deselects it outright; ``./scripts/test.sh
scrape`` does collect it, and it then skips on the ``db_container`` fixture when docker is
absent. Either way nobody needs docker to run the package's tests, and neither path is
what proves this suite ran -- select it explicitly with ``-m integration``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import asyncpg
import pytest
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from threetears.scrape.health import ScrapeTargetHealthCollection, content_fingerprint, record_validated_fetch
from threetears.scrape.migrations import apply_migrations

pytestmark = pytest.mark.integration


@pytest.fixture
async def pg_pool(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    """A plain asyncpg pool with every 3tears-scrape migration applied.

    Deliberately the real ``apply_migrations`` rather than hand-written DDL: the thing
    under test is whether the migrations this package ships actually provision what its
    entities read, so writing the schema by hand here would test a copy of the answer.
    """
    pool: asyncpg.Pool = await asyncpg.create_pool(db_container, min_size=1, max_size=4)
    try:
        await apply_migrations(pool)
        yield pool
    finally:
        await pool.close()


@pytest.fixture
def health(pg_pool: asyncpg.Pool) -> ScrapeTargetHealthCollection:
    """L3 only, no L1 backend wired.

    Deliberate: an L1 SQLite backend has to be initialized per table (normally by
    ``DataStore.create_table``, which a collection built directly never calls), and
    caching is not what this suite is testing. With no L1, every read goes to the real
    Postgres, which is precisely the path that needs proving.
    """
    registry = CollectionRegistry()
    registry.configure(l3_pool=pg_pool)
    return ScrapeTargetHealthCollection(registry, DefaultCoreConfig(collection_flush="ALWAYS"), nats_client=None)


def _target(name: str) -> str:
    """Unique target id per test: the container is session-scoped and rows persist."""
    return f"{name}_{uuid.uuid4().hex[:8]}"


async def test_every_health_field_round_trips_through_real_postgres(
    health: ScrapeTargetHealthCollection,
) -> None:
    """Write every column, evict L1, read it back from Postgres.

    Writing ALL fields at once is the point rather than a convenience: a single field
    with no column raises ``asyncpg.UndefinedColumnError`` on the upsert, so this fails
    loudly for exactly the bug class that motivated the suite. The cache invalidation
    keeps the read honest even if a caching tier is wired in later.
    """
    target_id = _target("warn_oh")
    blocked_at = datetime(2026, 7, 25, 3, 30, tzinfo=UTC)
    entity = health.create(
        {
            "target_id": target_id,
            "content_fingerprint": "a" * 64,
            "fingerprint_updated_at": blocked_at,
            "consecutive_fetch_failures": 3,
            "circuit_state": "open",
            "blocked_until": blocked_at + timedelta(hours=6),
            "last_blocked_at": blocked_at,
            "last_block_kind": "interstitial",
            "session_state_sealed": "sealed-ciphertext-token",
            "session_state_expires_at": blocked_at + timedelta(days=1),
        }
    )
    await entity.save()

    await health.invalidate_cache(target_id)
    stored = await health.get(target_id)

    assert stored is not None
    assert stored.content_fingerprint == "a" * 64
    assert stored.consecutive_fetch_failures == 3
    assert stored.circuit_state == "open"
    assert stored.last_block_kind == "interstitial"
    assert stored.session_state_sealed == "sealed-ciphertext-token"
    assert stored.last_blocked_at == blocked_at
    assert stored.blocked_until == blocked_at + timedelta(hours=6)
    assert stored.session_state_expires_at == blocked_at + timedelta(days=1)


async def test_a_health_row_needs_only_a_target_id(health: ScrapeTargetHealthCollection) -> None:
    """Every column is nullable or defaulted, so a first observation can be minimal.

    The blocked path writes health for a target that may never have succeeded, so it must
    be able to create a row without inventing values for columns it knows nothing about.
    """
    target_id = _target("warn_new")
    entity = health.create({"target_id": target_id})
    await entity.save()

    await health.invalidate_cache(target_id)
    stored = await health.get(target_id)

    assert stored is not None
    assert stored.content_fingerprint is None
    assert stored.consecutive_fetch_failures == 0
    assert stored.circuit_state == "closed"


async def test_record_validated_fetch_works_against_a_real_schema(health: ScrapeTargetHealthCollection) -> None:
    """The production write path, not a hand-built entity, against real Postgres."""
    target_id = _target("warn_md")
    page = "<html><body><table><tr><td>Acme Corp</td></tr></table></body></html>"

    await record_validated_fetch(health, target_id=target_id, html=page)

    await health.invalidate_cache(target_id)
    stored = await health.get(target_id)
    assert stored is not None
    assert stored.content_fingerprint == content_fingerprint(page)
    assert stored.fingerprint_updated_at is not None


async def test_a_second_success_merges_rather_than_replacing(health: ScrapeTargetHealthCollection) -> None:
    """The read-then-update path, against a real schema.

    Distinct from the fresh-row write above: the second call takes the branch that loads
    the existing row and saves it back as an existing entity, which is where a real
    database can disagree with an in-memory dict (column types, the compare-and-swap
    fence, and the creation timestamp that must not be reset by a later success).
    """
    target_id = _target("warn_merge")
    seeded = health.create(
        {
            "target_id": target_id,
            "consecutive_fetch_failures": 2,
            "circuit_state": "half_open",
            "last_block_kind": "interstitial",
        }
    )
    await seeded.save()

    await health.invalidate_cache(target_id)
    before = await health.get(target_id)
    assert before is not None
    created_at = before.date_created

    page = "<html><body><table><tr><td>Acme Corp</td></tr></table></body></html>"
    await record_validated_fetch(health, target_id=target_id, html=page)

    await health.invalidate_cache(target_id)
    stored = await health.get(target_id)
    assert stored is not None
    assert stored.content_fingerprint == content_fingerprint(page)
    # Unrelated health survives the success that proves the target recovered.
    assert stored.consecutive_fetch_failures == 2
    assert stored.circuit_state == "half_open"
    assert stored.last_block_kind == "interstitial"
    # And the row's creation time is not reset by a later fetch.
    assert stored.date_created == created_at
