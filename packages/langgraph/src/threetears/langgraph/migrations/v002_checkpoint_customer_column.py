"""
3tears-langgraph v002: a checkpoint row records its customer, so a customer's purge reads its rows by key.

A saver bound to a customer stores every thread under ``"<customer uuid>/<thread id>"``
(:meth:`~threetears.langgraph.checkpoint.ThreeTierCheckpointSaver.storage_thread_id`), and
:meth:`~threetears.langgraph.checkpoint.ThreeTierCheckpointSaver.adelete_customer_threads` used to find a customer's
rows by that prefix: a code-point range of ``thread_id``, which the primary key (in the column's own collation, and
hashed on YugabyteDB) does not order, so every row of the table was read. Now each row carries ``customer_id``, the
customer its ``thread_id`` names, and the purge deletes ``customer_id = $1`` through a partial index.

**Filled by the database, not by the writer, and with no table rewrite** (Pace, 2026-11-11, option B):

- ``customer_id UUID`` is a plain nullable column: ``ADD COLUMN`` changes the catalog only. A ``GENERATED ... STORED``
  column would rewrite the whole table (PostgreSQL holds an ACCESS EXCLUSIVE lock throughout; YugabyteDB rewrites
  too; probe S/o33/lgalter.log).
- A ``BEFORE INSERT OR UPDATE OF thread_id`` trigger sets it from ``thread_id`` on every write, whoever writes: a pod of
  the previous release, a sandboxed agent's statement over the rail, a by-hand re-key (``UPDATE ... SET thread_id =
  customer || '/' || thread_id``, the documented adoption path) all get it.
- Rows written before this migration are filled here, a page at a time in key order (:func:`backfill_customer_ids`;
  YugabyteDB walks the hashed ``thread_id`` lead value by value), each page one keyed ``UPDATE`` per customer. A row
  whose ``thread_id`` names no customer (an unscoped saver's) keeps ``NULL``, so the partial index holds only scoped
  rows.

The customer is the ``thread_id``'s first 36 characters when they are a lowercase UUID followed by ``/`` (the text a
:class:`~uuid.UUID` renders as), else ``NULL``: exactly the rows the prefix ``LIKE '<customer>/%'`` matched.

Idempotent: ``ADD COLUMN IF NOT EXISTS``, ``CREATE OR REPLACE`` function and trigger, ``CREATE INDEX IF NOT EXISTS``;
the backfill writes only a row whose ``customer_id`` differs, so a replay writes nothing. One statement at a time
(YugabyteDB runs no DDL in a transaction). Forward-only: 3tears migrations do not declare downgrades.

Revision ID: 002
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Final
from uuid import UUID

from threetears.core.data.store import DataStore
from threetears.core.keyed import KeyBatch, update_keyed
from threetears.core.keyset import keyset_pages
from threetears.observe import get_logger

__all__ = ["add_checkpoint_customer_column"]

log = get_logger(__name__)

#: a customer-scoped thread id: a lowercase UUID's text, then the separator
_SCOPED: Final = re.compile(r"^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/")

#: the same test in SQL, for the trigger
_SCOPED_SQL: Final = "'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/'"

#: each checkpoint table and its primary key, in key order
CUSTOMER_TABLES: Final[dict[str, tuple[str, ...]]] = {
    "checkpoints": ("thread_id", "checkpoint_ns", "checkpoint_id"),
    "checkpoint_writes": ("thread_id", "checkpoint_ns", "checkpoint_id", "task_id", "idx"),
}

#: each key column's SQL type, as a key batch casts its text back (``idx`` an integer, the rest text)
_KEY_CASTS: Final[dict[str, str]] = {"idx": "integer"}

_CUSTOMER_FUNCTION_SQL: Final = (
    "CREATE OR REPLACE FUNCTION checkpoint_customer_id() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
    f"NEW.customer_id := CASE WHEN NEW.thread_id ~ {_SCOPED_SQL} "
    "THEN CAST(substring(NEW.thread_id FROM 1 FOR 36) AS UUID) END; RETURN NEW; END $$"
)

_DDL: Final[tuple[str, ...]] = (
    "ALTER TABLE checkpoints ADD COLUMN IF NOT EXISTS customer_id UUID",
    "ALTER TABLE checkpoint_writes ADD COLUMN IF NOT EXISTS customer_id UUID",
    _CUSTOMER_FUNCTION_SQL,
    "CREATE OR REPLACE TRIGGER checkpoints_customer_id BEFORE INSERT OR UPDATE OF thread_id ON checkpoints "
    "FOR EACH ROW EXECUTE FUNCTION checkpoint_customer_id()",
    "CREATE OR REPLACE TRIGGER checkpoint_writes_customer_id BEFORE INSERT OR UPDATE OF thread_id ON checkpoint_writes "
    "FOR EACH ROW EXECUTE FUNCTION checkpoint_customer_id()",
    "-- purpose: packages/langgraph/src/threetears/langgraph/checkpoint.py::ThreeTierCheckpointSaver.adelete_customer_threads\n"
    "-- plan_test: packages/langgraph/tests/integration/test_checkpoint_customer.py::test_the_purge_reads_its_customer_s_rows_by_index\n"
    "CREATE INDEX IF NOT EXISTS idx_checkpoints_customer ON checkpoints (customer_id) WHERE customer_id IS NOT NULL",
    "-- purpose: packages/langgraph/src/threetears/langgraph/checkpoint.py::ThreeTierCheckpointSaver.adelete_customer_threads\n"
    "-- plan_test: packages/langgraph/tests/integration/test_checkpoint_customer.py::test_the_purge_reads_its_customer_s_rows_by_index\n"
    "CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_customer ON checkpoint_writes (customer_id) "
    "WHERE customer_id IS NOT NULL",
)

#: rows a backfill page reads
_BACKFILL_PAGE: Final = 500


def customer_of(thread_id: str) -> UUID | None:
    """the customer a stored ``thread_id`` names, as the trigger reads it: its UUID prefix, or None.

    :param thread_id: the stored thread id
    :ptype thread_id: str
    :return: the customer, or None for an unscoped thread
    :rtype: UUID | None
    """
    found = _SCOPED.match(thread_id)
    return None if found is None else UUID(found.group(1))


class _StoreReader:
    """a migration's :class:`DataStore` as the asyncpg-shaped reader the keyed helpers take."""

    def __init__(self, store: DataStore) -> None:
        self._store = store

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        return await self._store.query(query, *params)

    async def execute(self, query: str, *params: Any) -> str:
        return str(await self._store.execute(query, *params))


async def backfill_customer_ids(conn: Any, *, page: int = _BACKFILL_PAGE) -> int:
    """set ``customer_id`` on every checkpoint row whose ``thread_id`` names a customer and whose column differs, a
    page at a time in key order; each page one keyed ``UPDATE`` per customer it holds.

    No cache holds ``customer_id`` (the saver caches checkpoint bundles, which do not carry it), so no row is evicted.

    :param conn: an asyncpg-shaped reader (``fetch`` / ``execute``) bound to the agent's schema
    :ptype conn: Any
    :param page: the rows a page reads
    :ptype page: int
    :return: the rows written
    :rtype: int
    """
    written = 0
    for table, key in CUSTOMER_TABLES.items():
        columns = ", ".join(f'"{column}"' for column in (*key, "customer_id"))
        async for rows in keyset_pages(
            conn, f"SELECT {columns} FROM {table}", key, page_size=page, lead_array="text[]"
        ):
            written += await _fill_page(conn, table, key, rows)
    return written


async def _fill_page(conn: Any, table: str, key: Sequence[str], rows: Sequence[dict[str, Any]]) -> int:
    """write one page's customers, one keyed ``UPDATE`` per customer whose rows differ.

    :param conn: the reader
    :ptype conn: Any
    :param table: the checkpoint table
    :ptype table: str
    :param key: its primary key
    :ptype key: Sequence[str]
    :param rows: the page's rows (key and ``customer_id``)
    :ptype rows: Sequence[dict[str, Any]]
    :return: the rows written
    :rtype: int
    """
    by_customer: dict[UUID, list[tuple[Any, ...]]] = defaultdict(list)
    for row in rows:
        customer = customer_of(row["thread_id"])
        if customer is not None and row["customer_id"] != customer:
            by_customer[customer].append(tuple(row[column] for column in key))
    written = 0
    for customer, keys in by_customer.items():
        casts = tuple(_KEY_CASTS.get(column, "text") for column in key)
        updated = await update_keyed(
            conn, table, values={"customer_id": customer}, keys=KeyBatch(tuple(key), keys, casts=casts)
        )
        written += updated.count
    return written


async def add_checkpoint_customer_column(store: DataStore) -> None:
    """add ``customer_id`` to both checkpoint tables with the trigger that fills it and its partial index, then fill
    the rows written before.

    :param store: DataStore bound to the per-agent schema
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    for statement in _DDL:
        await store.execute(statement)
    written = await backfill_customer_ids(_StoreReader(store))
    log.info("langgraph v002: checkpoint customer_id added, filled on %d existing row(s)", written)
