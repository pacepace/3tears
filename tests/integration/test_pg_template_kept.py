"""integration: on a server that copies no template (YugabyteDB), each migration set's schema is migrated once a
process and handed to every test as the migrations left it (``tests.pg_template``).

Each case enters ``migrated_copy`` as consecutive tests do, on the session's server with the kept path asked for
(``copies_templates=False``: a PostgreSQL session takes it as a YugabyteDB one does): the rows, the sequence and the schema the next test is handed,
and a fresh migration whenever a test left more than rows.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import asyncpg
import pytest

from tests.pg_template import drop_kept, kept_schema, migrated_copy

pytestmark = pytest.mark.integration

#: the migration set every case migrates: a parent and a child referring to it, a serial key, a table the migrations
#: write a row into, and a function
_MIGRATION = """
CREATE TABLE parents (id serial PRIMARY KEY, name text NOT NULL);
CREATE TABLE children (id serial PRIMARY KEY, parent_id integer NOT NULL REFERENCES parents (id), note text);
CREATE INDEX ix_children_parent ON children (parent_id);
CREATE TABLE applied (version integer PRIMARY KEY);
INSERT INTO applied VALUES (1);
CREATE FUNCTION doubled(x integer) RETURNS integer LANGUAGE sql AS 'SELECT x * 2';
"""


class _Counted:
    """the migration set, counting each time it runs."""

    def __init__(self) -> None:
        self.runs = 0

    async def __call__(self, conn: asyncpg.Connection, _schema: str) -> None:
        self.runs += 1
        await conn.execute(_MIGRATION)


@pytest.fixture
async def kept_key(db_container: str) -> AsyncIterator[str]:
    """a migration-set name of this test's own; the schema kept for it is dropped after."""
    key = f"kept_{uuid.uuid4().hex[:8]}"
    try:
        yield key
    finally:
        await drop_kept(db_container, key)


async def _schema_exists(url: str, schema: str) -> bool:
    conn = await asyncpg.connect(url)
    try:
        return bool(await conn.fetchval("SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = $1)", schema))
    finally:
        await conn.close()


async def _connect(url: str, schema: str) -> asyncpg.Connection:
    conn = await asyncpg.connect(url)
    await conn.execute(f'SET search_path TO "{schema}", public')
    return conn


async def test_the_next_test_gets_the_same_schema_without_the_rows_and_with_the_sequences_set_back(
    db_container: str, kept_key: str
) -> None:
    migrate = _Counted()
    async with migrated_copy(db_container, key=kept_key, migrate=migrate, copies_templates=False) as (url, first):
        conn = await _connect(url, first)
        try:
            parent = await conn.fetchval("INSERT INTO parents (name) VALUES ('a') RETURNING id")
            await conn.execute("INSERT INTO children (parent_id, note) VALUES ($1, 'x'), ($1, 'y')", parent)
            assert parent == 1
        finally:
            await conn.close()
    async with migrated_copy(db_container, key=kept_key, migrate=migrate, copies_templates=False) as (url, second):
        conn = await _connect(url, second)
        try:
            assert second == first and migrate.runs == 1
            assert await conn.fetchval("SELECT count(*) FROM parents") == 0
            assert await conn.fetchval("SELECT count(*) FROM children") == 0
            assert await conn.fetch("SELECT version FROM applied") == [(1,)]
            # the serial key starts where a freshly migrated schema's does
            assert await conn.fetchval("INSERT INTO parents (name) VALUES ('b') RETURNING id") == 1
            assert await conn.fetchval("SELECT doubled(4)") == 8
        finally:
            await conn.close()


@pytest.mark.parametrize(
    "left",
    [
        pytest.param("CREATE TABLE extra (id integer)", id="a-new-table"),
        pytest.param("CREATE INDEX ix_children_note ON children (note)", id="a-new-index"),
        pytest.param("ALTER TABLE parents ADD COLUMN born date", id="a-new-column"),
        pytest.param("ALTER TABLE children ALTER COLUMN note SET DEFAULT 'n'", id="a-new-default"),
        pytest.param(
            "CREATE OR REPLACE FUNCTION doubled(x integer) RETURNS integer LANGUAGE sql AS 'SELECT x * 3'",
            id="a-changed-function",
        ),
        pytest.param("INSERT INTO applied VALUES (2)", id="a-migration-row-added"),
        pytest.param("DELETE FROM applied", id="a-migration-row-deleted"),
        pytest.param("ANALYZE parents", id="an-analyze"),
    ],
)
async def test_a_test_that_left_more_than_rows_hands_the_next_a_fresh_schema(
    db_container: str, kept_key: str, left: str
) -> None:
    migrate = _Counted()
    async with migrated_copy(db_container, key=kept_key, migrate=migrate, copies_templates=False) as (url, first):
        conn = await _connect(url, first)
        try:
            await conn.execute("INSERT INTO parents (name) VALUES ('a')")
            await conn.execute(left)
        finally:
            await conn.close()
    assert not await _schema_exists(db_container, first), "the changed schema was kept"
    async with migrated_copy(db_container, key=kept_key, migrate=migrate, copies_templates=False) as (url, second):
        conn = await _connect(url, second)
        try:
            assert second != first and migrate.runs == 2
            assert await conn.fetchval("SELECT count(*) FROM parents") == 0
            assert await conn.fetch("SELECT version FROM applied") == [(1,)]
            assert await conn.fetchval("SELECT doubled(4)") == 8
        finally:
            await conn.close()


async def test_a_migration_that_fails_keeps_nothing(db_container: str, kept_key: str) -> None:
    async def failing(conn: asyncpg.Connection, _schema: str) -> None:
        await conn.execute("CREATE TABLE half (id integer)")
        raise RuntimeError("the migration failed")

    with pytest.raises(RuntimeError, match="the migration failed"):
        async with migrated_copy(db_container, key=kept_key, migrate=failing, copies_templates=False):
            pass
    assert kept_schema(db_container, kept_key) is None
