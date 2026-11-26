"""agent-acl v011: a namespace's owner is stored by id, ``owner_namespace_id``.

**Why** (Pace, 2026-11-26). A namespace named its owner by NAME (``owner_namespace``), and the hub keyed a foreign key
onto ``name`` through it, so its name index had to stay whole (a foreign key's target cannot be a partial index): a
deleted namespace kept its name until quiet-time cleanup removed the row, and a datasource created again under a
removed one's name was refused. The owner is now stored by id, ``owner_namespace_id``, which a deployment holding its
owners' rows keys onto ``namespace_id``; the name index can hold live rows only. ``owner_namespace`` stays the owner's
NAME beside it: the evaluator's ownership short-circuit compares it with the caller's own name, derived in process, the
one owner path that has to answer inside an agent pod that reads no ``namespaces`` row
(:func:`threetears.agent.acl.evaluator._agent_owns_namespace`).

The steps, each idempotent:

1. ``owner_namespace_id UUID``, nullable: a row whose owner names no row (a hub-less deployment registers no agents'
   rows, v001) has none;
2. ``idx_namespaces_owner_namespace_id``: what a namespace owns
   (:meth:`~threetears.agent.acl.collections.NamespaceCollection.list_owned_by`);
3. the backfill (:func:`backfill_owner_namespace_ids`): every live row naming an owner and holding no id takes its
   owner's id, a page of keys at a time (``keyset_pages``, each page's writes keyed updates), never one ``UPDATE`` of
   the table. A gone row is left: nothing reads its owner, and a revive writes every column again
   (``NamespaceCollection.ensure_namespace``);
4. ``idx_namespaces_owner_namespace`` dropped: v003 made it for ``list_owned_by``, which reads the id now, and nothing
   else reads by the owner's name.

A deployment whose own migrations create ``namespaces`` (the hub) runs no agent_acl migration: it takes these steps in a
migration of its own (:data:`ADD_OWNER_ID_SQL`, :func:`backfill_owner_namespace_ids`). Forward-only.

Revision ID: 011
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from threetears.core.data.migrations.backfill import BACKFILL_PAGE
from threetears.core.data.store import DataStore
from threetears.core.ddl.session import SessionReader
from threetears.core.keyed import Fragment, KeyBatch, update_keyed
from threetears.core.keyset import keyset_pages
from threetears.observe import get_logger

__all__ = ["ADD_OWNER_ID_SQL", "OWNER_ID_OF_NAME", "backfill_owner_namespace_ids", "owner_namespace_by_id"]

log = get_logger(__name__)

#: the column (FROZEN)
ADD_OWNER_ID_SQL: Final[str] = "ALTER TABLE namespaces ADD COLUMN IF NOT EXISTS owner_namespace_id UUID"

#: the owner's id, from its name: one row of that name, a live one before a gone one (a name unique among live rows
#: alone may also be held by gone rows), through the name's index. NULL where no row carries it (FROZEN)
OWNER_ID_OF_NAME: Final[str] = (
    "(SELECT o.namespace_id FROM namespaces o WHERE o.name = $1 ORDER BY o.gone, o.namespace_id LIMIT 1)"
)

#: what a namespace owns, by id, and the old index by the owner's name dropped (FROZEN)
_INDEX_DDL: Final[tuple[str, ...]] = (
    "-- purpose: packages/agent/acl/src/threetears/agent/acl/collections.py::NamespaceCollection.list_owned_by\n"
    "-- plan_test: packages/agent/acl/tests/integration/test_index_plans.py::test_namespaces_owner_namespace_id\n"
    "-- reason: what one namespace owns, by its id, each step of a namespace teardown's walk\n"
    "CREATE INDEX IF NOT EXISTS idx_namespaces_owner_namespace_id ON namespaces (owner_namespace_id)",
    "DROP INDEX IF EXISTS idx_namespaces_owner_namespace",
)


def _owners_of_page(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    """a page's rows that name an owner and hold no id, grouped by the owner's name: each group one write.

    :param rows: a page's rows: ``namespace_id``, ``owner_namespace`` and ``owner_namespace_id``
    :ptype rows: Sequence[Mapping[str, Any]]
    :return: owner name -> the keys of the rows it owns
    :rtype: dict[str, list[Any]]
    """
    owners: dict[str, list[Any]] = {}
    for row in rows:
        owner = row["owner_namespace"]
        if owner is not None and row["owner_namespace_id"] is None:
            owners.setdefault(owner, []).append(row["namespace_id"])
    return owners


async def backfill_owner_namespace_ids(conn: Any, *, page: int = BACKFILL_PAGE) -> int:
    """set ``owner_namespace_id`` on every live row that names an owner and holds no id, a page of keys at a time in key
    order (:func:`~threetears.core.keyset.keyset_pages`, ``table=`` so a page reads live rows): each page one keyed
    ``UPDATE`` per owner it names, the id read by name in the statement (:data:`OWNER_ID_OF_NAME`). A row already
    holding one is left, so a re-run writes nothing.

    No cache is touched: a migration runs in the outage window, before any process reads the table.

    :param conn: an asyncpg-shaped reader (``fetch`` / ``execute``) bound where ``namespaces`` resolves
    :ptype conn: Any
    :param page: the rows a page reads
    :ptype page: int
    :return: the rows written
    :rtype: int
    """
    written = 0
    async for rows in keyset_pages(
        conn,
        'SELECT "namespace_id", "owner_namespace", "owner_namespace_id" FROM namespaces',
        ("namespace_id",),
        page_size=page,
        lead_array="uuid[]",
        table="namespaces",
    ):
        for owner, keys in _owners_of_page(rows).items():
            batch = KeyBatch(("namespace_id",), [keys], casts=("uuid",))
            values = {"owner_namespace_id": Fragment(OWNER_ID_OF_NAME, (owner,))}
            written += (await update_keyed(conn, "namespaces", values=values, keys=batch)).count
    return written


async def owner_namespace_by_id(store: DataStore) -> None:
    """add ``owner_namespace_id``, its index, fill it, and drop the index by the owner's name (v011).

    :param store: DataStore bound to the platform schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await store.execute(ADD_OWNER_ID_SQL)
    await store.execute(_INDEX_DDL[0])
    written = await backfill_owner_namespace_ids(SessionReader(store))
    await store.execute(_INDEX_DDL[1])
    log.info("agent_acl v011: a namespace's owner is stored by id", extra={"extra_data": {"written": written}})
