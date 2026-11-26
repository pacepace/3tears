"""integration: ``make_unique_index_live_only`` makes a stamped table's unique name hold its live rows only.

On a real Postgres: after the swap a gone row keeps its name and a re-create by that name is a new row with a new id,
two live rows of one name are still refused, the index keeps its name, predicate and comment, a ``UNIQUE``
constraint swaps the same way, a second run does nothing, a run stopped after the drop finishes with its rename, and a
foreign key's target, the primary key and an unstamped table's index are refused, left whole.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest
from threetears.core.data.migrations import (
    LiveUniqueRefusedError,
    make_unique_index_live_only,
    make_unique_indexes_live_only,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def db_image() -> str:
    """pin pgvector/pg16 to match the rest of the core integration suite."""
    return "pgvector/pgvector:pg16"


class _Store:
    """the migration store surface the helper reads and writes through, over one connection."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def execute(self, sql: str, *params: Any) -> str:
        status: str = await self._conn.execute(sql, *params)
        return status

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        return [dict(row) for row in await self._conn.fetch(sql, *params)]


class _StoppedError(RuntimeError):
    """the run stopped, as a migration's process stops mid-run."""


class _StoppedAtRename(_Store):
    """the store of a run that stops at its rename: every statement before it ran, the rename never does."""

    async def execute(self, sql: str, *params: Any) -> str:
        if " RENAME TO " in sql:
            raise _StoppedError(sql)
        return await super().execute(sql, *params)


async def _indexes(conn: asyncpg.Connection) -> set[str]:
    """the names of the ``things`` table's indexes in the test's schema."""
    rows = await conn.fetch(
        "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema() AND tablename = 'things'"
    )
    return {str(row["indexname"]) for row in rows}


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    """a connection on a fresh schema holding a stamped ``things`` table, named uniquely, and its parent."""
    schema = f"live_unique_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(db_container)
    await connection.execute(f'CREATE SCHEMA "{schema}"')
    await connection.execute(f'SET search_path TO "{schema}"')
    await connection.execute(
        "CREATE TABLE things (id uuid PRIMARY KEY, customer_id uuid, name text NOT NULL, "
        "gone boolean NOT NULL DEFAULT false)"
    )
    try:
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await connection.close()


async def _add(conn: asyncpg.Connection, name: str, customer: uuid.UUID | None = None) -> uuid.UUID:
    row_id = uuid.uuid4()
    await conn.execute("INSERT INTO things (id, customer_id, name) VALUES ($1, $2, $3)", row_id, customer, name)
    return row_id


async def _predicate(conn: asyncpg.Connection, index: str) -> str | None:
    value: str | None = await conn.fetchval(
        "SELECT pg_get_expr(i.indpred, i.indrelid) FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
        "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relname = $1 AND n.nspname = current_schema()",
        index,
    )
    return value


async def test_a_deleted_name_is_created_again_as_a_new_row(conn: asyncpg.Connection) -> None:
    await conn.execute("CREATE UNIQUE INDEX things_name ON things (name)")
    first = await _add(conn, "alpha")
    await conn.execute("UPDATE things SET gone = true WHERE id = $1", first)
    with pytest.raises(asyncpg.UniqueViolationError):
        await _add(conn, "alpha")

    assert await make_unique_index_live_only(_Store(conn), "things_name") is True

    second = await _add(conn, "alpha")
    assert second != first
    assert await conn.fetchval("SELECT count(*) FROM things WHERE name = 'alpha'") == 2
    assert await _predicate(conn, "things_name") == "(NOT gone)"


async def test_two_live_rows_of_one_name_are_still_refused(conn: asyncpg.Connection) -> None:
    await conn.execute("CREATE UNIQUE INDEX things_name ON things (name)")
    await make_unique_index_live_only(_Store(conn), "things_name")
    await _add(conn, "alpha")
    with pytest.raises(asyncpg.UniqueViolationError):
        await _add(conn, "alpha")


async def test_an_index_keeps_its_own_predicate_and_its_comment(conn: asyncpg.Connection) -> None:
    await conn.execute("CREATE UNIQUE INDEX things_platform_name ON things (name) WHERE customer_id IS NULL")
    await conn.execute("COMMENT ON INDEX things_platform_name IS 'one platform thing a name'")
    await make_unique_index_live_only(_Store(conn), "things_platform_name")
    assert await _predicate(conn, "things_platform_name") == "((customer_id IS NULL) AND (NOT gone))"
    assert (
        await conn.fetchval("SELECT obj_description('things_platform_name'::regclass, 'pg_class')")
        == "one platform thing a name"
    )
    # a customer's thing never met the index, and still does not
    customer = uuid.uuid4()
    await _add(conn, "alpha", customer)
    await _add(conn, "alpha", uuid.uuid4())


async def test_a_unique_constraint_swaps_for_a_live_only_index(conn: asyncpg.Connection) -> None:
    await conn.execute("ALTER TABLE things ADD CONSTRAINT things_customer_name_key UNIQUE (customer_id, name)")
    customer = uuid.uuid4()
    first = await _add(conn, "alpha", customer)
    await conn.execute("UPDATE things SET gone = true WHERE id = $1", first)
    await make_unique_index_live_only(_Store(conn), "things_customer_name_key")
    assert await conn.fetchval("SELECT count(*) FROM pg_constraint WHERE conname = 'things_customer_name_key'") == 0
    await _add(conn, "alpha", customer)
    with pytest.raises(asyncpg.UniqueViolationError):
        await _add(conn, "alpha", customer)


async def test_an_on_conflict_names_the_live_predicate(conn: asyncpg.Connection) -> None:
    await conn.execute("CREATE UNIQUE INDEX things_name ON things (name)")
    # the clause holds on the whole index, before the swap
    upsert = (
        "INSERT INTO things (id, name) VALUES ($1, 'alpha') ON CONFLICT (name) WHERE NOT gone "
        "DO UPDATE SET customer_id = EXCLUDED.customer_id RETURNING id"
    )
    first = await conn.fetchval(upsert, uuid.uuid4())
    assert await conn.fetchval(upsert, uuid.uuid4()) == first
    await make_unique_index_live_only(_Store(conn), "things_name")
    # and after it: the live row is the arbiter; once it is gone, the name is a new row
    assert await conn.fetchval(upsert, uuid.uuid4()) == first
    await conn.execute("UPDATE things SET gone = true WHERE id = $1", first)
    assert await conn.fetchval(upsert, uuid.uuid4()) != first
    # a target without the predicate no longer finds the live-only index
    with pytest.raises(asyncpg.InvalidColumnReferenceError):
        await conn.execute(
            "INSERT INTO things (id, name) VALUES ($1, 'beta') ON CONFLICT (name) DO NOTHING", uuid.uuid4()
        )


async def test_a_second_run_does_nothing(conn: asyncpg.Connection) -> None:
    await conn.execute("CREATE UNIQUE INDEX things_name ON things (name)")
    assert await make_unique_indexes_live_only(_Store(conn), ["things_name"]) == 1
    assert await make_unique_indexes_live_only(_Store(conn), ["things_name"]) == 0
    assert await _predicate(conn, "things_name") == "(NOT gone)"


async def test_a_run_stopped_after_the_drop_finishes_with_the_rename(conn: asyncpg.Connection) -> None:
    await conn.execute("CREATE UNIQUE INDEX things_name ON things (name)")
    before = await _indexes(conn)
    with pytest.raises(_StoppedError):
        await make_unique_indexes_live_only(_StoppedAtRename(conn), ["things_name"])
    # stopped after the drop: the live-only index stands under its temporary name alone
    during = await _indexes(conn)
    assert "things_name" not in during and len(during) == len(before), during
    assert await make_unique_indexes_live_only(_Store(conn), ["things_name"]) == 1
    assert await _predicate(conn, "things_name") == "(NOT gone)"
    # the temporary name is gone with the rename
    assert await _indexes(conn) == before


async def test_a_foreign_keys_target_is_refused_and_left_whole(conn: asyncpg.Connection) -> None:
    await conn.execute("CREATE UNIQUE INDEX things_name ON things (name)")
    await conn.execute("CREATE UNIQUE INDEX things_customer_name ON things (customer_id, name)")
    await conn.execute("CREATE TABLE owned (id uuid PRIMARY KEY, owner_name text REFERENCES things (name))")
    with pytest.raises(LiveUniqueRefusedError, match="foreign key's target"):
        await make_unique_indexes_live_only(_Store(conn), ["things_customer_name", "things_name"])
    # nothing changed: the refusal came before any swap
    assert await _predicate(conn, "things_name") is None
    assert await _predicate(conn, "things_customer_name") is None


async def test_the_primary_key_and_an_unstamped_table_are_refused(conn: asyncpg.Connection) -> None:
    with pytest.raises(LiveUniqueRefusedError, match="primary key"):
        await make_unique_index_live_only(_Store(conn), "things_pkey")
    await conn.execute("CREATE TABLE plain (id uuid PRIMARY KEY, name text)")
    await conn.execute("CREATE UNIQUE INDEX plain_name ON plain (name)")
    with pytest.raises(LiveUniqueRefusedError, match="not stamped"):
        await make_unique_index_live_only(_Store(conn), "plain_name")
    with pytest.raises(LiveUniqueRefusedError, match="no such index"):
        await make_unique_index_live_only(_Store(conn), "no_such_index")
