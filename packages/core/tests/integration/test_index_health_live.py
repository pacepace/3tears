"""an index a failed build left unusable is built again from its own definition; a usable one is left alone.

A failed build's leftover is made here as the catalog records one: ``pg_index.indisvalid`` turned off (Postgres
allows a superuser that, and YugabyteDB keeps the same columns). Uses the session-scoped ``db_container`` fixture.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from threetears.core.ddl.index_health import (
    IndexDuplicateKeysError,
    rebuild_unusable_indexes,
    unusable_indexes,
)
from threetears.core.testing.plans import on_yugabyte

pytestmark = pytest.mark.integration


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[tuple[asyncpg.Connection, str]]:
    schema = f"idx_{uuid.uuid4().hex[:8]}"
    connection = await asyncpg.connect(db_container)
    try:
        await connection.execute(f"CREATE SCHEMA {schema}")
        await connection.execute(
            f"CREATE TABLE {schema}.datasets (id uuid PRIMARY KEY, customer_id uuid NOT NULL, name text NOT NULL, "
            "is_deleted boolean NOT NULL DEFAULT false)"
        )
        await connection.execute(
            f"CREATE UNIQUE INDEX datasets_customer_id_name_key ON {schema}.datasets (customer_id, name) "
            "WHERE is_deleted = false"
        )
        await connection.execute(f"CREATE INDEX datasets_by_name ON {schema}.datasets (name)")
        yield connection, schema
    finally:
        await connection.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await connection.close()


async def _break(conn: asyncpg.Connection, schema: str, name: str) -> None:
    """mark an index as a failed build leaves it: not valid, not ready (still updated and enforced while ready)."""
    if await on_yugabyte(conn):
        pytest.skip("YugabyteDB does not let the catalog be written to stand in for a failed build")
    await conn.execute(
        "UPDATE pg_index SET indisvalid = false, indisready = false WHERE indexrelid = to_regclass($1)",
        f"{schema}.{name}",
    )


async def _valid(conn: asyncpg.Connection, schema: str, name: str) -> Any:
    return await conn.fetchval(
        "SELECT indisvalid AND indisready AND indislive FROM pg_index WHERE indexrelid = to_regclass($1)",
        f"{schema}.{name}",
    )


async def test_an_unusable_index_is_built_again_from_its_definition(conn: tuple[asyncpg.Connection, str]) -> None:
    connection, schema = conn
    await _break(connection, schema, "datasets_customer_id_name_key")
    oid_before = await connection.fetchval("SELECT to_regclass($1)::oid", f"{schema}.datasets_customer_id_name_key")

    rebuilt = await rebuild_unusable_indexes(connection, [schema])

    assert [index.name for index in rebuilt] == ["datasets_customer_id_name_key"]
    assert await _valid(connection, schema, "datasets_customer_id_name_key")
    assert (
        await connection.fetchval("SELECT to_regclass($1)::oid", f"{schema}.datasets_customer_id_name_key")
        != oid_before
    )
    # built from its own definition: still unique, still partial
    definition = await connection.fetchval(
        "SELECT pg_get_indexdef(to_regclass($1))", f"{schema}.datasets_customer_id_name_key"
    )
    assert "UNIQUE" in definition and "is_deleted = false" in definition


async def test_duplicate_keys_refuse_the_rebuild_loudly_and_drop_nothing(
    conn: tuple[asyncpg.Connection, str],
) -> None:
    connection, schema = conn
    await _break(connection, schema, "datasets_customer_id_name_key")
    customer = uuid.uuid4()
    for _ in range(2):
        # the unusable index enforces nothing, as on prod: the duplicate lands
        await connection.execute(
            f"INSERT INTO {schema}.datasets (id, customer_id, name) VALUES ($1, $2, 'turnout')", uuid.uuid4(), customer
        )

    with pytest.raises(IndexDuplicateKeysError, match="turnout") as refused:
        await rebuild_unusable_indexes(connection, [schema])

    assert refused.value.duplicates == (((customer, "turnout"), 2),)
    assert [index.name for index in await unusable_indexes(connection, [schema])] == ["datasets_customer_id_name_key"]


async def test_usable_indexes_are_left_alone(conn: tuple[asyncpg.Connection, str]) -> None:
    connection, schema = conn
    before = await connection.fetchval("SELECT to_regclass($1)::oid", f"{schema}.datasets_by_name")

    assert await rebuild_unusable_indexes(connection, [schema]) == []
    assert await connection.fetchval("SELECT to_regclass($1)::oid", f"{schema}.datasets_by_name") == before
