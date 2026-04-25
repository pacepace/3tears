"""
integration tests for ``threetears.core.data.migrations.helpers``.

drives every helper end-to-end against a real testcontainers postgres
container. proves:

- ``add_column_with_backfill`` actually adds the column AND writes the
  backfill rows (the v054 footgun: yugabyte silently no-ops the
  UPDATE inside a DDL+DML DO block; postgres does not reproduce that
  exact behaviour, but we still prove the UPDATE runs and respects
  the replay guard)
- ``replace_check_constraint`` swaps the constraint and short-circuits
  on re-run
- ``replace_primary_key`` swaps the PK, preserves UNIQUE on the id
  column, and round-trips inbound FK ON DELETE clauses
- ``add_partition_column`` lands the column + backfill + check in one
  declarative call

guarded by ``@pytest.mark.integration`` and skips when docker is
unavailable.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import asyncpg
import pytest
from threetears.core.data.migrations.helpers import (
    InboundFk,
    add_check_constraint,
    add_column_with_backfill,
    add_partition_column,
    replace_check_constraint,
    replace_primary_key,
)

pytestmark = pytest.mark.integration

POSTGRES_IMAGE = "pgvector/pgvector:pg16"


@pytest.fixture(scope="module")
def pg_url() -> Iterator[str]:
    """spin up a postgres container and yield an asyncpg-compatible URL."""
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:
        pytest.skip("testcontainers not installed")

    container = PostgresContainer(POSTGRES_IMAGE)
    try:
        container.start()
    except Exception as exc:
        pytest.skip(f"docker unavailable: {exc}")
    try:
        url = container.get_connection_url()
        if url.startswith("postgresql+psycopg2://"):
            url = url.replace("postgresql+psycopg2://", "postgresql://", 1)
        yield url
    finally:
        container.stop()


class _AsyncpgStore:
    """``MigrationStore``-shape wrapper over an asyncpg connection."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        """initialize wrapper.

        :param conn: asyncpg connection
        :ptype conn: asyncpg.Connection
        """
        self._conn = conn

    async def execute(self, sql: str, *params: Any) -> str:
        """delegate to the underlying connection.

        :param sql: SQL text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: status tag from asyncpg
        :rtype: str
        """
        result: str = await self._conn.execute(sql, *params)
        return result


@pytest.fixture
async def pg_conn(pg_url: str) -> AsyncIterator[asyncpg.Connection]:
    """yield a connection bound to a fresh per-test schema."""
    schema = f"helpers_it_{id(object())}".lower().replace("-", "_")
    conn = await asyncpg.connect(pg_url)
    await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    await conn.execute(f'SET search_path TO "{schema}"')
    yield conn
    await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    await conn.close()


# ---------------------------------------------------------------------------
# add_column_with_backfill
# ---------------------------------------------------------------------------


async def test_add_column_with_backfill_lands_column_and_rows(
    pg_conn: asyncpg.Connection,
) -> None:
    """ADD COLUMN + UPDATE flow lands the column AND backfills matching rows."""
    await pg_conn.execute("CREATE TABLE w (id SERIAL PRIMARY KEY, c VARCHAR(8))")
    # seed three rows: two with c IS NULL, one already at 'b'.
    await pg_conn.execute("INSERT INTO w (c) VALUES (NULL), (NULL), ('b')")
    store = _AsyncpgStore(pg_conn)

    await add_column_with_backfill(
        store,
        table="w",
        column="row_scope",
        column_type="VARCHAR(8)",
        default="'customer'",
        not_null=False,
        backfill_value_sql="'platform'",
        backfill_predicate="c IS NULL",
    )

    rows = await pg_conn.fetch("SELECT id, c, row_scope FROM w ORDER BY id")
    assert len(rows) == 3
    assert rows[0]["row_scope"] == "platform"
    assert rows[1]["row_scope"] == "platform"
    assert rows[2]["row_scope"] == "customer"


async def test_add_column_with_backfill_replay_is_noop(
    pg_conn: asyncpg.Connection,
) -> None:
    """re-running the helper does not double-update rows."""
    await pg_conn.execute("CREATE TABLE w (id SERIAL PRIMARY KEY, c VARCHAR(8))")
    await pg_conn.execute("INSERT INTO w (c) VALUES (NULL)")
    store = _AsyncpgStore(pg_conn)

    for _ in range(2):
        await add_column_with_backfill(
            store,
            table="w",
            column="row_scope",
            column_type="VARCHAR(8)",
            default="'customer'",
            backfill_value_sql="'platform'",
            backfill_predicate="c IS NULL",
        )

    rows = await pg_conn.fetch("SELECT row_scope FROM w")
    # the second run sees row_scope='platform' (not the default), so the
    # replay guard skips it.
    assert all(r["row_scope"] == "platform" for r in rows)


# ---------------------------------------------------------------------------
# add_check_constraint
# ---------------------------------------------------------------------------


async def test_add_check_constraint_blocks_invalid_inserts(
    pg_conn: asyncpg.Connection,
) -> None:
    """CHECK constraint rejects values outside the closed set."""
    await pg_conn.execute("CREATE TABLE w (c VARCHAR(8) NOT NULL)")
    store = _AsyncpgStore(pg_conn)

    await add_check_constraint(
        store,
        table="w",
        constraint_name="w_c_ck",
        expression="c IN ('a', 'b')",
    )

    await pg_conn.execute("INSERT INTO w (c) VALUES ('a')")
    with pytest.raises(asyncpg.exceptions.CheckViolationError):
        await pg_conn.execute("INSERT INTO w (c) VALUES ('z')")


async def test_add_check_constraint_replay_is_noop(
    pg_conn: asyncpg.Connection,
) -> None:
    """second call does not raise (constraint already exists)."""
    await pg_conn.execute("CREATE TABLE w (c VARCHAR(8))")
    store = _AsyncpgStore(pg_conn)

    for _ in range(2):
        await add_check_constraint(
            store,
            table="w",
            constraint_name="w_c_ck",
            expression="c IN ('a', 'b')",
        )

    # constraint visible exactly once.
    count = await pg_conn.fetchval(
        "SELECT count(*) FROM pg_constraint WHERE conname = 'w_c_ck'",
    )
    assert count == 1


# ---------------------------------------------------------------------------
# replace_check_constraint
# ---------------------------------------------------------------------------


async def test_replace_check_constraint_swaps_definition(
    pg_conn: asyncpg.Connection,
) -> None:
    """re-installing a CHECK with a wider set admits the new value."""
    await pg_conn.execute(
        "CREATE TABLE w (c VARCHAR(8) NOT NULL, "
        "CONSTRAINT w_c_ck CHECK (c IN ('a', 'b')))",
    )
    store = _AsyncpgStore(pg_conn)

    # original constraint rejects 'c'.
    with pytest.raises(asyncpg.exceptions.CheckViolationError):
        await pg_conn.execute("INSERT INTO w (c) VALUES ('c')")

    await replace_check_constraint(
        store,
        table="w",
        constraint_name="w_c_ck",
        new_expression="c IN ('a', 'b', 'c')",
    )
    await pg_conn.execute("INSERT INTO w (c) VALUES ('c')")


async def test_replace_check_constraint_short_circuits_with_engine_form(
    pg_conn: asyncpg.Connection,
) -> None:
    """providing ``engine_normalized_def`` makes replay an OID-stable no-op.

    postgres normalizes CHECK expressions on storage (``c IN ('a',
    'b')`` round-trips as ``((c)::text = ANY ((ARRAY['a'::character
    varying, 'b'::character varying])::text[]))``). callers who want
    OID-stable replay supply the engine form explicitly; the v037 /
    v042 / v045 / v048 hub migrations are the canonical example of
    this pattern.
    """
    await pg_conn.execute(
        "CREATE TABLE w (c VARCHAR(8) NOT NULL, "
        "CONSTRAINT w_c_ck CHECK (c IN ('a', 'b')))",
    )
    store = _AsyncpgStore(pg_conn)
    # capture the engine-stored form once so we can supply it back.
    engine_form: str = await pg_conn.fetchval(
        "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
        "WHERE conname = 'w_c_ck'",
    )

    oid_before = await pg_conn.fetchval(
        "SELECT oid FROM pg_constraint WHERE conname = 'w_c_ck'",
    )
    # second call: engine form supplied, helper short-circuits.
    await replace_check_constraint(
        store,
        table="w",
        constraint_name="w_c_ck",
        new_expression="c IN ('a', 'b')",
        engine_normalized_def=engine_form,
    )
    oid_after = await pg_conn.fetchval(
        "SELECT oid FROM pg_constraint WHERE conname = 'w_c_ck'",
    )
    assert oid_before == oid_after


async def test_replace_check_constraint_replay_without_engine_form_still_idempotent(
    pg_conn: asyncpg.Connection,
) -> None:
    """without engine_normalized_def, replay DROP+ADDs but still leaves a working constraint."""
    await pg_conn.execute(
        "CREATE TABLE w (c VARCHAR(8) NOT NULL)",
    )
    store = _AsyncpgStore(pg_conn)

    for _ in range(3):
        await replace_check_constraint(
            store,
            table="w",
            constraint_name="w_c_ck",
            new_expression="c IN ('a', 'b')",
        )

    # constraint exists exactly once at the end.
    count = await pg_conn.fetchval(
        "SELECT count(*) FROM pg_constraint WHERE conname = 'w_c_ck'",
    )
    assert count == 1
    # and it actually rejects 'z'.
    with pytest.raises(asyncpg.exceptions.CheckViolationError):
        await pg_conn.execute("INSERT INTO w (c) VALUES ('z')")


# ---------------------------------------------------------------------------
# replace_primary_key
# ---------------------------------------------------------------------------


async def test_replace_primary_key_basic_swap(
    pg_conn: asyncpg.Connection,
) -> None:
    """basic PK swap: new composite PK + UNIQUE on the preserved id column."""
    await pg_conn.execute(
        "CREATE TABLE w (id UUID PRIMARY KEY, customer_id UUID NOT NULL)",
    )
    store = _AsyncpgStore(pg_conn)

    await replace_primary_key(
        store,
        table="w",
        new_columns=("customer_id", "id"),
    )

    pk_def = await pg_conn.fetchval(
        "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
        "WHERE conrelid = 'w'::regclass AND contype = 'p'",
    )
    assert pk_def is not None
    assert "PRIMARY KEY (customer_id, id)" in pk_def
    unique_count = await pg_conn.fetchval(
        "SELECT count(*) FROM pg_constraint WHERE conname = 'w_id_unique'",
    )
    assert unique_count == 1


async def test_replace_primary_key_with_inbound_fk(
    pg_conn: asyncpg.Connection,
) -> None:
    """inbound FK is dropped + recreated against UNIQUE(id) post-swap."""
    await pg_conn.execute(
        "CREATE TABLE w (id UUID PRIMARY KEY, customer_id UUID NOT NULL)",
    )
    await pg_conn.execute(
        "CREATE TABLE child (child_id UUID PRIMARY KEY, "
        "parent_id UUID NOT NULL REFERENCES w(id) ON DELETE CASCADE)",
    )
    store = _AsyncpgStore(pg_conn)

    fk_name = await pg_conn.fetchval(
        "SELECT conname FROM pg_constraint "
        "WHERE conrelid = 'child'::regclass AND contype = 'f'",
    )
    assert fk_name is not None

    await replace_primary_key(
        store,
        table="w",
        new_columns=("customer_id", "id"),
        inbound_fks=(
            InboundFk(
                source_table="child",
                constraint_name=str(fk_name),
                source_column="parent_id",
                on_delete="ON DELETE CASCADE",
            ),
        ),
    )

    # FK still present after the swap, with CASCADE preserved.
    fk_def = await pg_conn.fetchval(
        "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
        "WHERE conrelid = 'child'::regclass AND contype = 'f'",
    )
    assert fk_def is not None
    assert "ON DELETE CASCADE" in fk_def
    # cascade actually fires.
    parent_id = "11111111-1111-1111-1111-111111111111"
    customer_id = "22222222-2222-2222-2222-222222222222"
    child_id = "33333333-3333-3333-3333-333333333333"
    await pg_conn.execute(
        "INSERT INTO w (id, customer_id) VALUES ($1::uuid, $2::uuid)",
        parent_id,
        customer_id,
    )
    await pg_conn.execute(
        "INSERT INTO child (child_id, parent_id) VALUES ($1::uuid, $2::uuid)",
        child_id,
        parent_id,
    )
    await pg_conn.execute("DELETE FROM w WHERE id = $1::uuid", parent_id)
    children = await pg_conn.fetch("SELECT child_id FROM child")
    assert children == []


async def test_replace_primary_key_replay_is_noop(
    pg_conn: asyncpg.Connection,
) -> None:
    """re-running the swap on a table already on the target form is a no-op."""
    await pg_conn.execute(
        "CREATE TABLE w (id UUID PRIMARY KEY, customer_id UUID NOT NULL)",
    )
    store = _AsyncpgStore(pg_conn)

    await replace_primary_key(
        store,
        table="w",
        new_columns=("customer_id", "id"),
    )
    pk_oid_first = await pg_conn.fetchval(
        "SELECT oid FROM pg_constraint "
        "WHERE conrelid = 'w'::regclass AND contype = 'p'",
    )
    await replace_primary_key(
        store,
        table="w",
        new_columns=("customer_id", "id"),
    )
    pk_oid_second = await pg_conn.fetchval(
        "SELECT oid FROM pg_constraint "
        "WHERE conrelid = 'w'::regclass AND contype = 'p'",
    )
    assert pk_oid_first == pk_oid_second


# ---------------------------------------------------------------------------
# add_partition_column
# ---------------------------------------------------------------------------


async def test_add_partition_column_lands_column_backfill_and_check(
    pg_conn: asyncpg.Connection,
) -> None:
    """add_partition_column composes the three-step partition-add pattern."""
    await pg_conn.execute(
        "CREATE TABLE w (id SERIAL PRIMARY KEY, customer_id UUID)",
    )
    await pg_conn.execute("INSERT INTO w (customer_id) VALUES (NULL)")
    await pg_conn.execute(
        "INSERT INTO w (customer_id) VALUES "
        "('22222222-2222-2222-2222-222222222222'::uuid)",
    )
    store = _AsyncpgStore(pg_conn)

    await add_partition_column(
        store,
        table="w",
        column="row_scope",
        column_type="VARCHAR(8)",
        default="'customer'",
        backfill_value_sql="'platform'",
        backfill_predicate="customer_id IS NULL",
        add_check_allowed_values=("platform", "customer"),
    )

    rows = await pg_conn.fetch("SELECT customer_id, row_scope FROM w ORDER BY id")
    assert len(rows) == 2
    assert rows[0]["row_scope"] == "platform"
    assert rows[1]["row_scope"] == "customer"

    # CHECK rejects non-allowed values.
    with pytest.raises(asyncpg.exceptions.CheckViolationError):
        await pg_conn.execute(
            "INSERT INTO w (customer_id, row_scope) "
            "VALUES (NULL, 'invalid')",
        )
