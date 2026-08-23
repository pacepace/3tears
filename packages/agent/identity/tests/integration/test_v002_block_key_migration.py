"""live proof that v002 adds wants / needs to identity_block_key.

The case that actually matters is the UPGRADE path, not the fresh one: v001
shipped in 3tears v0.26.1, so every already-deployed agent schema has the
five-label enum and has recorded v001 as applied. Those schemas will never
re-run v001's ``CREATE TYPE``, which is precisely why the new labels need a
forward migration -- and why ``test_upgrade_from_v001_only_schema`` below
builds a runner registered with v001 ALONE, applies it, and only then
applies the full chain. A test that only ever exercises a fresh schema
would pass just as happily if v002 did nothing and someone had edited
v001's DDL instead, which is the mistake this file exists to catch.

Everything runs against a real Postgres (testcontainers), because the
constraint being verified is a PostgreSQL one: ``ALTER TYPE ... ADD VALUE``
is the statement that cannot run inside a transaction block, and no
in-memory double reproduces that.
"""

from __future__ import annotations

import asyncpg
import pytest
from threetears.agent.identity.types import IDENTITY_BLOCK_KEY_VALUES

from .conftest import AsyncpgStore, apply_migrations

# integration, not asyncio: asyncio_mode is "auto" repo-wide, while the
# marker is what keeps the no-docker CI job (-m "not integration") from
# running these against a Postgres that isn't there.
pytestmark = pytest.mark.integration


async def _enum_labels(url: str, schema: str) -> list[str]:
    """read identity_block_key's labels, in enum sort order, from ``schema``."""
    conn = await asyncpg.connect(url)
    try:
        rows = await conn.fetch(
            """
            SELECT e.enumlabel
              FROM pg_enum e
              JOIN pg_type t ON t.oid = e.enumtypid
              JOIN pg_namespace n ON n.oid = t.typnamespace
             WHERE t.typname = 'identity_block_key'
               AND n.nspname = $1
             ORDER BY e.enumsortorder
            """,
            schema,
        )
        return [r["enumlabel"] for r in rows]
    finally:
        await conn.close()


async def _apply_v001_only(url: str, schema: str) -> None:
    """apply ONLY v001, leaving the schema exactly as a v0.26.1 deploy left it."""
    from threetears.agent.identity.migrations.v001_create_identity_versions import (
        create_identity_versions_table,
    )
    from threetears.core.data.migrations import (
        MigrationRunner,
        MigrationScope,
        PackageMigrations,
    )

    conn = await asyncpg.connect(url)
    try:
        await conn.execute(f'SET search_path TO "{schema}", public')
        runner = MigrationRunner()
        pkg = PackageMigrations(name="agent_identity", scope=MigrationScope.AGENT)
        pkg.version(1)(create_identity_versions_table)
        runner.register(pkg)
        await runner.apply_for_agent_schema(AsyncpgStore(conn))  # type: ignore[arg-type]
    finally:
        await conn.close()


async def test_fresh_schema_has_every_declared_block_key(pg_schema: tuple[str, str]) -> None:
    """A fresh chain yields exactly the Python enum's members, in declaration order."""
    url, schema = pg_schema
    await apply_migrations(url, schema)

    assert tuple(await _enum_labels(url, schema)) == IDENTITY_BLOCK_KEY_VALUES


async def test_upgrade_from_v001_only_schema(pg_schema: tuple[str, str]) -> None:
    """A schema already at v001 gains wants / needs when the chain advances.

    This is the deployed-fleet path. Asserting the five-label starting state
    first is what makes the post-condition meaningful -- without it the test
    could not tell an upgrade from a fresh create.
    """
    url, schema = pg_schema
    await _apply_v001_only(url, schema)
    assert await _enum_labels(url, schema) == [
        "personality",
        "reinforcement",
        "anti_sycophant",
        "self_improvement",
        "presence",
    ]

    await apply_migrations(url, schema)

    assert tuple(await _enum_labels(url, schema)) == IDENTITY_BLOCK_KEY_VALUES


async def test_replay_is_a_no_op(pg_schema: tuple[str, str]) -> None:
    """Re-applying the chain neither errors nor duplicates a label."""
    url, schema = pg_schema
    await apply_migrations(url, schema)
    await apply_migrations(url, schema)

    labels = await _enum_labels(url, schema)
    assert len(labels) == len(set(labels))
    assert tuple(labels) == IDENTITY_BLOCK_KEY_VALUES


@pytest.mark.parametrize("block_key", ["wants", "needs"])
async def test_new_labels_are_usable_in_a_row(pg_schema: tuple[str, str], block_key: str) -> None:
    """The added labels bind as a real block_key value, not just as catalog rows.

    A label present in pg_enum but uncommitted would still fail here, so
    this is the assertion that proves the runner's no-transaction contract
    holds for ALTER TYPE ADD VALUE.
    """
    url, schema = pg_schema
    await apply_migrations(url, schema)

    conn = await asyncpg.connect(url)
    try:
        await conn.execute(f'SET search_path TO "{schema}", public')
        await conn.execute(
            """
            INSERT INTO identity_versions
                (version_id, agent_id, block_key, content, content_hash, status,
                 date_created, date_updated)
            VALUES (gen_random_uuid(), gen_random_uuid(), $1::identity_block_key,
                    'body', 'hash', 'active'::identity_version_status,
                    now(), now())
            """,
            block_key,
        )
        stored = await conn.fetchval("SELECT block_key FROM identity_versions WHERE content_hash = 'hash'")
        assert stored == block_key
    finally:
        await conn.close()
