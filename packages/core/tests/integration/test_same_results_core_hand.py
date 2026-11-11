"""integration: core's hand conversions onto the keyed helpers read and write what the statements they replaced did.

Each site names its table at run time (a child a delete's cascade reaches, a declared collection's table), so it
cannot sit in the use-the-helper manifest, whose source check reads a literal table. Each is held here instead by the
shared equivalence harness (:func:`threetears.core.testing.same_results.assert_same_results`): every seed case (empty,
one, many, NULLs, ties, a tenant boundary, a page boundary and one past it), the old statement against the converted
code, on the session's ``db_container`` (PostgreSQL), or YugabyteDB through ``THREETEARS_TEST_POSTGRES_URL``; a schema
of its own.

- ``delete_cascade._children`` (read through :func:`~threetears.core.collections.delete_cascade.read_delete_cascade`,
  the rows a shelf's ``SET NULL`` reaches): ``SELECT <key> FROM <child> WHERE <column> = ANY($1)`` paged by the key under the
  rail's cap, by hand -> :func:`~threetears.core.keyed.keyset_rows` with the set and the key bound untyped
  (``AnyOf(cast=None)``, ``Sort(cast=None)``), a bigint key and a uuid parent so an untyped bind is typed by its column.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from threetears.core.collections import delete_cascade as delete_cascade_module
from threetears.core.testing.same_results import Seeded, assert_same_results

pytestmark = pytest.mark.integration

#: the parents a cascade read names: tenant 0's
_PARENTS = [uuid.UUID(int=1), uuid.UUID(int=3)]


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    """A superuser connection on a fresh schema holding a child table keyed by a bigint, naming a uuid parent."""
    schema = f"corehand_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(db_container)
    try:
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}"')
        await connection.execute("CREATE TABLE o27_books (id bigint PRIMARY KEY, shelf_id uuid, title text)")
        await connection.execute("CREATE INDEX o27_books_shelf ON o27_books (shelf_id)")
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


class _Books:
    """the child collection a shelf's cascade reaches: keyed by its id, its messages carrying the shelf."""

    primary_key_columns = ("id",)
    invalidation_columns = ("shelf_id",)


class _Registry:
    def get_collection(self, table: str) -> Any:
        return _Books() if table == "o27_books" else None


class _Shelves:
    """the parent collection: a shelf's books' shelf_id is SET NULL when it goes (one level, no recursion)."""

    delete_cascades = (delete_cascade_module.DeleteCascade("o27_books", "shelf_id", "SET NULL"),)
    primary_key_columns = ("id",)
    registry = _Registry()

    def __init__(self, l3_pool: Any) -> None:
        self.l3_pool = l3_pool

    def normalize_pk(self, key: Any) -> tuple[Any, ...]:
        return (key,)


async def _cascaded_books(c: Any) -> list[dict[str, Any]]:
    """the books a delete of the shelves would reach, as the cascade reads them before the delete."""
    reached = await delete_cascade_module.read_delete_cascade(_Shelves(c), _PARENTS)
    return [dict(row) for cascaded in reached for row in cascaded.rows]


def _book(seed: Seeded) -> dict[str, Any]:
    shelf = None if seed.null else uuid.UUID(int=1 + 2 * seed.tenant + seed.index % 2)
    return {"id": 1000 - seed.tie * 7 - seed.index, "shelf_id": shelf, "title": f"b{seed.index}"}


async def test_a_cascade_s_children_are_the_rows_the_any_named(
    conn: asyncpg.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if the cascade's keyed pages name other children than the one ``= ANY`` it replaced, or lose or double
    one across a page."""
    # a page of 5 under the cap, where the harness puts its page boundary
    monkeypatch.setattr(delete_cascade_module, "L3_RAIL_ROW_CAP", 7)
    await assert_same_results(
        conn,
        table="o27_books",
        row=_book,
        old_sql="SELECT id, shelf_id FROM o27_books WHERE shelf_id = ANY($1) ORDER BY id",
        params=[_PARENTS],
        helper=lambda c: _cascaded_books(c),
        page_size=5,
    )
