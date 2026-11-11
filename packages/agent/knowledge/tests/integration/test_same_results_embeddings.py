"""integration: the vector read by id (each collection's ``fetch_embeddings``) reads what the statement it replaced read.

``SELECT id, embedding::text AS embedding FROM <table> WHERE id = ANY($1)`` became a keyed read by primary key
(:func:`threetears.core.keyed.read_keyed`, the customer scope on each statement through
:func:`~threetears.core.keyed.with_options`). The equivalence harness seeds each table, as declared plus the hub's
``embedding`` vector column, for every case and compares the old statement's rows with the rows the code's own
statement returned (captured under the proxy pool it reads through), row for row. On the session's ``db_container``
(PostgreSQL), or YugabyteDB through ``THREETEARS_TEST_POSTGRES_URL``; a schema of its own.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest
from sqlalchemy import MetaData
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from threetears.agent.knowledge.collections import ConceptCollection, PlaybookEntryCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.same_results import Seeded, assert_same_results

pytestmark = pytest.mark.integration

_START = datetime(2026, 11, 1, tzinfo=UTC)

#: the seeds the read names (and one no row holds): every case holds some of them, the large ones not all their rows
_NAMED = [uuid.uuid5(uuid.NAMESPACE_OID, f"knowledge-{index}") for index in (0, 1, 2, 4, 7, 11, 999)]


def _id(seed: Seeded) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_OID, f"knowledge-{seed.index}")


def _common(seed: Seeded) -> dict[str, Any]:
    stamp = _START + timedelta(minutes=seed.tie)
    return {
        "id": _id(seed),
        "datasource_id": uuid.uuid5(uuid.NAMESPACE_OID, f"ds-{seed.tenant}"),
        "date_created": stamp,
        "date_updated": stamp,
        "embedding": None if seed.null else f"[{seed.index}, {seed.tie}, 0.5]",
    }


def _entry(seed: Seeded) -> Mapping[str, Any]:
    return {
        **_common(seed),
        "playbook_id": uuid.uuid5(uuid.NAMESPACE_OID, "pb"),
        "title": f"t{seed.index}",
        "body": "b",
    }


def _concept(seed: Seeded) -> Mapping[str, Any]:
    return {**_common(seed), "name": f"n{seed.index}", "definition": "d"}


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    """A connection on a fresh schema holding the two knowledge tables as declared, plus the hub's vector column."""
    schema = f"knowvec_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(db_container)
    try:
        await connection.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}", public')
        vector_schema = await connection.fetchval(
            "SELECT n.nspname FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace WHERE t.typname = 'vector'"
        )
        await connection.set_type_codec("vector", schema=vector_schema, encoder=str, decoder=str, format="text")
        dialect = postgresql.dialect()  # type: ignore[no-untyped-call]
        for declared in (PlaybookEntryCollection.schema, ConceptCollection.schema):
            table = declared.to_sqlalchemy_table(MetaData())
            await connection.execute(str(CreateTable(table).compile(dialect=dialect)))
            await connection.execute(f'ALTER TABLE "{table.name}" ADD COLUMN embedding vector(3)')
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


class _Proxy:
    """the rbac-read proxy pool as ``fetch_embeddings`` reads through it (a customer scope on each statement), over
    the test connection; keeps the rows the statement returned."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn
        self.rows: list[dict[str, Any]] = []
        self.scopes: list[Any] = []

    async def fetch(self, query: str, *params: Any, customer_scope: Any) -> list[Any]:
        self.scopes.append(customer_scope)
        answered = await self._conn.fetch(query, *params)
        self.rows.extend(dict(row) for row in answered)
        return list(answered)


@pytest.mark.parametrize(
    ("collection", "row"),
    [(PlaybookEntryCollection, _entry), (ConceptCollection, _concept)],
    ids=["playbook_entries", "concepts"],
)
async def test_the_vector_read_reads_what_the_statement_it_replaced_read(
    conn: asyncpg.Connection, collection: Any, row: Any
) -> None:
    """Flips if the keyed vector read returns other rows, ids or vector text than ``id = ANY($1)`` did, or drops the
    customer scope from its statement."""
    scope = uuid.uuid4()
    table = collection.schema.name

    async def helper(c: asyncpg.Connection) -> list[dict[str, Any]]:
        proxy = _Proxy(c)
        config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
        reading = collection(registry=CollectionRegistry(), config=config)
        # the rbac-read proxy pool itself, as an agent pod's collection holds it (not wrapped as a plain L3 backend)
        reading.l3_pool = proxy
        await reading.fetch_embeddings(list(_NAMED), customer_scope=scope)
        assert proxy.scopes and set(proxy.scopes) == {scope}
        return proxy.rows

    await assert_same_results(
        conn,
        table=table,
        row=row,
        old_sql=f"SELECT id, embedding::text AS embedding FROM {table} WHERE id = ANY($1)",  # noqa: S608 - the test's table
        params=[list(_NAMED)],
        helper=helper,
    )
