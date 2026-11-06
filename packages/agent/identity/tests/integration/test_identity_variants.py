"""live proof of block variants: v003 upgrades a deployed schema, and each variant keeps its own chain.

The upgrade case builds a schema at v002, writes a row, and only then applies
the chain, because a schema that recorded v002 never re-runs v001's DDL --
a fresh-schema test would pass just as well if v003 did nothing and v001
had been edited instead.

The lifecycle cases run the real collection over real rows: the one-active
rule is a partial unique index, and only Postgres enforces it.
"""

from __future__ import annotations

import uuid
from typing import Any

import asyncpg
import pytest

from threetears.agent.identity import lifecycle

from .conftest import AsyncpgStore, InMemoryNatsBus, apply_migrations, build_collection, make_pool, owner_authorizer

pytestmark = pytest.mark.integration

_AGENT = uuid.UUID("00000000-0000-0000-0000-000000000001")
_USER = uuid.UUID("00000000-0000-0000-0000-00000000000b")
_CUST = _USER
_BLOCK = "presence"  # tier 1: a proposal waits for consent
_ROUTINE = "self_improvement"  # tier 2: a proposal applies at once


async def _apply_through_v002(url: str, schema: str) -> None:
    """apply v001 and v002 only, as a deploy before variants left the schema."""
    from threetears.agent.identity.migrations.v001_create_identity_versions import create_identity_versions_table
    from threetears.agent.identity.migrations.v002_add_wants_needs_block_keys import add_wants_needs_block_keys
    from threetears.core.data.migrations import MigrationRunner, MigrationScope, PackageMigrations

    conn = await asyncpg.connect(url)
    try:
        await conn.execute(f'SET search_path TO "{schema}", public')
        runner = MigrationRunner()
        pkg = PackageMigrations(name="agent_identity", scope=MigrationScope.AGENT)
        pkg.version(1)(create_identity_versions_table)
        pkg.version(2)(add_wants_needs_block_keys)
        runner.register(pkg)
        await runner.apply_for_agent_schema(AsyncpgStore(conn))  # type: ignore[arg-type]
    finally:
        await conn.close()


async def _indexes(url: str, schema: str) -> dict[str, str]:
    conn = await asyncpg.connect(url)
    try:
        rows = await conn.fetch(
            "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = $1 AND tablename = 'identity_versions'",
            schema,
        )
        return {r["indexname"]: r["indexdef"] for r in rows}
    finally:
        await conn.close()


async def test_upgrade_keeps_every_row_as_the_block_itself(pg_schema: tuple[str, str]) -> None:
    url, schema = pg_schema
    await _apply_through_v002(url, schema)
    before = await _indexes(url, schema)
    assert "uq_identity_active_per_block" in before and "uq_identity_active_per_variant" not in before
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(f'SET search_path TO "{schema}", public')
        await conn.execute(
            """
            INSERT INTO identity_versions
                (version_id, agent_id, customer_id, user_id, block_key, content, content_hash, status,
                 date_created, date_updated)
            VALUES (gen_random_uuid(), $1, $2, $2, 'presence', 'before variants', 'h', 'active', now(), now())
            """,
            _AGENT,
            _USER,
        )
    finally:
        await conn.close()

    await apply_migrations(url, schema)
    await apply_migrations(url, schema)  # a replay is a no-op

    after = await _indexes(url, schema)
    assert "uq_identity_active_per_block" not in after and "idx_identity_block_history" not in after
    assert "(agent_id, customer_id, user_id, block_key, variant)" in after["uq_identity_active_per_variant"]
    assert "WHERE (status = 'active'" in after["uq_identity_active_per_variant"]
    assert "(agent_id, user_id, block_key, variant, date_created DESC)" in after["idx_identity_variant_history"]

    pool = await make_pool(url, schema)
    try:
        coll, _l1 = build_collection(pool, InMemoryNatsBus())
        kept = await coll.resolve_active(agent_id=_AGENT, customer_id=_CUST, user_id=_USER, block_key=_BLOCK)
        assert kept is not None and kept.content == "before variants" and kept.variant == ""
    finally:
        await pool.close()


async def _stack(pg_schema: tuple[str, str]) -> tuple[Any, Any]:
    url, schema = pg_schema
    await apply_migrations(url, schema)
    pool = await make_pool(url, schema)
    coll, _l1 = build_collection(pool, InMemoryNatsBus())
    return coll, pool


def _kw(**extra: Any) -> dict[str, Any]:
    base: dict[str, Any] = dict(agent_id=_AGENT, customer_id=_CUST, user_id=_USER, caller_agent_id=_AGENT)
    base.update(extra)
    return base


async def _active(coll: Any, block: str, variant: str) -> Any:
    return await coll.resolve_active(
        agent_id=_AGENT, customer_id=_CUST, user_id=_USER, block_key=block, variant=variant
    )


async def _propose_and_consent(coll: Any, authz: Any, content: str, variant: str) -> Any:
    proposed = await lifecycle.propose(
        coll,
        authz,
        block_key=_BLOCK,
        content=content,
        rationale="r",
        proposer_agent_id=_AGENT,
        variant=variant,
        **_kw(),
    )
    assert proposed is not None and proposed.status == "proposed"
    return await lifecycle.consent(coll, authz, version_id=proposed.version_id, consenter_user_id=_USER, **_kw())


async def test_each_variant_keeps_its_own_active_version_and_chain(pg_schema: tuple[str, str]) -> None:
    coll, pool = await _stack(pg_schema)
    authz = owner_authorizer()
    try:
        base1 = await _propose_and_consent(coll, authz, "base one", "")
        alt1 = await _propose_and_consent(coll, authz, "alt one", "alt")
        assert alt1.parent_version_id is None  # the variant's chain starts on its own, not under the block's

        alt2 = await _propose_and_consent(coll, authz, "alt two", "alt")
        assert alt2.parent_version_id == alt1.version_id

        assert (await _active(coll, _BLOCK, "")).version_id == base1.version_id  # untouched by alt's consent
        assert (await _active(coll, _BLOCK, "alt")).version_id == alt2.version_id
        assert (
            await pool.fetchval(
                "SELECT count(*) FROM identity_versions WHERE block_key = $1 AND status = 'active'", _BLOCK
            )
            == 2
        )

        history = await coll.find_versions_for_block(
            agent_id=_AGENT, customer_id=_CUST, user_id=_USER, block_key=_BLOCK, variant="alt"
        )
        assert [v.content for v in history] == ["alt two", "alt one"]
        assert [
            v.content
            for v in await coll.find_versions_for_block(
                agent_id=_AGENT, customer_id=_CUST, user_id=_USER, block_key=_BLOCK
            )
        ] == ["base one"]

        clone = await lifecycle.rollback(
            coll, authz, target_version_id=alt1.version_id, consenter_user_id=_USER, **_kw()
        )
        assert clone is not None and clone.variant == "alt" and clone.content == "alt one"
        assert (await _active(coll, _BLOCK, "alt")).version_id == clone.version_id
        assert (await _active(coll, _BLOCK, "")).version_id == base1.version_id
    finally:
        await pool.close()


async def test_a_routine_variant_applies_at_once_beside_the_block(pg_schema: tuple[str, str]) -> None:
    coll, pool = await _stack(pg_schema)
    authz = owner_authorizer()
    try:
        base = await lifecycle.propose(
            coll, authz, block_key=_ROUTINE, content="base", rationale="r", proposer_agent_id=_AGENT, **_kw()
        )
        alt = await lifecycle.propose(
            coll,
            authz,
            block_key=_ROUTINE,
            content="alt",
            rationale="r",
            proposer_agent_id=_AGENT,
            variant="alt",
            **_kw(),
        )
        assert base is not None and alt is not None and base.status == alt.status == "active"
        assert (await _active(coll, _ROUTINE, "")).content == "base"
        assert (await _active(coll, _ROUTINE, "alt")).content == "alt"
    finally:
        await pool.close()


async def test_seed_imports_each_variant_on_its_own(pg_schema: tuple[str, str]) -> None:
    coll, pool = await _stack(pg_schema)
    authz = owner_authorizer()
    try:
        seed = dict(agent_id=_AGENT, customer_id=_CUST, user_id=_USER, block_key=_BLOCK)
        base = await lifecycle.seed_active(coll, authz, content="base", **seed)
        alt = await lifecycle.seed_active(coll, authz, content="alt", variant="alt", **seed)
        again = await lifecycle.seed_active(coll, authz, content="ignored", variant="alt", **seed)
        assert base.version_id != alt.version_id
        assert again.version_id == alt.version_id  # idempotent per variant
        assert (await _active(coll, _BLOCK, "")).content == "base"
        assert (await _active(coll, _BLOCK, "alt")).content == "alt"
    finally:
        await pool.close()


async def test_the_index_refuses_a_second_active_version_in_one_variant(pg_schema: tuple[str, str]) -> None:
    url, schema = pg_schema
    await apply_migrations(url, schema)
    conn = await asyncpg.connect(url)
    insert = """
        INSERT INTO identity_versions
            (version_id, agent_id, customer_id, user_id, block_key, variant, content, content_hash, status,
             date_created, date_updated)
        VALUES (gen_random_uuid(), $1, $2, $2, 'presence', $3, 'c', 'h', 'active', now(), now())
    """
    try:
        await conn.execute(f'SET search_path TO "{schema}", public')
        await conn.execute(insert, _AGENT, _USER, "alt")
        await conn.execute(insert, _AGENT, _USER, "")  # another variant: allowed
        with pytest.raises(asyncpg.UniqueViolationError):
            await conn.execute(insert, _AGENT, _USER, "alt")
    finally:
        await conn.close()
