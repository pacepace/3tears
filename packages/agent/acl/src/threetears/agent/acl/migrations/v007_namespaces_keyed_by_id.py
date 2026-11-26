"""agent-acl v007: ``namespaces`` is keyed by ``namespace_id`` alone, and ``row_scope`` is a moveable column.

**Why** (Pace, 2026-11-25). ``namespaces`` was keyed ``(row_scope, namespace_id)``, and a rescope
(:meth:`~threetears.agent.acl.collections.NamespaceCollection.rescope`) changes ``row_scope`` in place: half the key.
With the change-note ring retired, a restore of a lost change log reads a stamped table's rows back by their CURRENT key,
so it never heard the old key go away. Gone-plus-new is not possible: ``namespace_id`` and ``name`` are unique and are
foreign-key targets (``role_assignments.scope_namespace_id``; the hub's ``owner_namespace`` onto ``name``). So the key
is ``namespace_id``, and ``row_scope`` an ordinary column declared moveable (:mod:`~threetears.core.ddl.tenant_moves`):
a rescope is an in-place update whose old scope is recorded on the row and heard, live and by a restore.

The steps, in order, each its own statement (DDL only; YugabyteDB's DDL/DML separation), each idempotent:

1. ``row_scope_moved_from TEXT`` and ``row_scope_moved_epoch BIGINT``, nullable, the moveable column's two L3-only
   columns (:func:`~threetears.core.ddl.tenant_moves.moved_columns`): NULL in every existing row, which never moved.
2. ``idx_namespaces_scope_id (row_scope, namespace_id)``: the old primary key's shape, which
   :meth:`~threetears.agent.acl.collections.NamespaceCollection.page_ids` walks a scope at a time. WHY kept as an
   index: the new key is hashed on YugabyteDB, so a page past a cursor over it alone would read and sort the table.
3. The primary key: ``namespaces_pkey`` dropped and added again on ``(namespace_id)``, only when the catalog says it is
   on other columns. The foreign keys onto ``namespaces(namespace_id)`` reference ``uq_namespaces_namespace_id``, not
   the primary key, and are left as they are; so is that unique constraint (now the key's twin), which they depend on.
4. The restore bucket, rebuilt for the new key, only when the bucket index's declaration does not already name it:
   the index and the stored ``epoch_bucket`` dropped (the bucket is derived from the key's hash: the old one hashed
   ``row_scope`` too), then added again as ``TableSchema.create_statements`` renders them, and the declaration naming
   ``namespace_id`` as the key, ``row_scope`` as moveable and as an invalidation column. The declaration is written
   last, so a run stopped between these statements finds it unwritten and takes the step again.

**The outage-window cutover** (Pace 2026-11-25: one cutover, everything scaled to zero): nothing writes while it runs.
It never truncates and never drops a row; the only things it drops are the old primary key constraint and the derived
bucket column and its index, each added again in this run. Forward-only.

A deployment whose own migrations create ``namespaces`` (the hub's v001) runs no agent_acl migration: it takes these
same steps in a migration of its own (:data:`NAMESPACES`, :func:`rekey_namespaces`).

Revision ID: 007
"""

from __future__ import annotations

import json
from typing import Any, Final

from threetears.core.data.migrations.stamp_tables import StampingTable, bucket_statements, on_yugabyte
from threetears.core.ddl.row_epoch import BUCKET_COLUMN
from threetears.core.ddl.tenant_moves import moved_epoch_column, moved_from_column
from threetears.core.sql_fragments import quote_identifier
from threetears.observe import get_logger

__all__ = ["NAMESPACES", "SCOPE_ID_INDEX_SQL", "namespaces_keyed_by_id", "rekey_namespaces"]

log = get_logger(__name__)

#: ``namespaces`` as v007 leaves it (FROZEN: a later change of its key or declaration is its own migration)
NAMESPACES: Final[StampingTable] = StampingTable(
    "namespaces",
    (("namespace_id", "uuid"),),
    moveable=("row_scope",),
    columns=("row_scope",),
)

#: the key v007 leaves
_KEY: Final[tuple[str, ...]] = ("namespace_id",)

#: the old primary key's shape, as an index :meth:`NamespaceCollection.page_ids` walks a scope at a time
SCOPE_ID_INDEX_SQL: Final[str] = (
    "-- purpose: packages/agent/acl/src/threetears/agent/acl/collections.py::NamespaceCollection.page_ids\n"
    "-- plan_test: packages/agent/acl/tests/integration/test_index_plans.py::test_namespace_ids_page_by_key\n"
    "-- reason: every namespace id a page at a time, a scope at a time: the key alone is hashed on YugabyteDB\n"
    "CREATE INDEX IF NOT EXISTS idx_namespaces_scope_id ON namespaces (row_scope, namespace_id)"
)

#: the primary key's constraint name and columns, in key order, as the catalog holds them (none when it has none)
_PRIMARY_KEY_SQL: Final[str] = (
    "SELECT c.conname AS name, a.attname AS column_name FROM pg_constraint c"
    " CROSS JOIN LATERAL unnest(c.conkey) WITH ORDINALITY AS k(attnum, position)"
    " JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.attnum"
    " WHERE c.contype = 'p' AND c.conrelid = to_regclass('namespaces') ORDER BY k.position"
)

#: the bucket index's declaration, as its comment holds it (NULL when the index or its comment is missing)
_DECLARATION_SQL: Final[str] = f"SELECT obj_description(to_regclass('{NAMESPACES.index}'), 'pg_class') AS declared"


def _declared_keys(declared: Any) -> list[str] | None:
    """the key a bucket index's declaration names, or None when it names none (missing, or not a declaration).

    :param declared: the comment's text, or None
    :ptype declared: Any
    :return: the key's columns
    :rtype: list[str] | None
    """
    keys: list[str] | None = None
    if isinstance(declared, str):
        try:
            parsed = json.loads(declared)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("keys"), list):
            keys = [str(column) for column in parsed["keys"]]
    return keys


async def rekey_namespaces(store: Any) -> None:
    """the four steps: the moved columns, the scope index, the primary key on ``namespace_id``, the bucket rebuilt.

    :param store: the migration's store, bound to the table's schema via ``search_path``
    :ptype store: Any
    :return: nothing
    :rtype: None
    """
    quoted = quote_identifier(NAMESPACES.name)
    await store.execute(f"ALTER TABLE {quoted} ADD COLUMN IF NOT EXISTS {moved_from_column('row_scope')} text")
    await store.execute(f"ALTER TABLE {quoted} ADD COLUMN IF NOT EXISTS {moved_epoch_column('row_scope')} bigint")
    await store.execute(SCOPE_ID_INDEX_SQL)
    primary = await store.query(_PRIMARY_KEY_SQL)
    if tuple(str(row["column_name"]) for row in primary) != _KEY:
        # two statements, not one ALTER of both: each is its own on YugabyteDB, and a run stopped between them finds
        # no primary key and adds it
        name = str(primary[0]["name"]) if primary else "namespaces_pkey"
        await store.execute(f"ALTER TABLE {quoted} DROP CONSTRAINT IF EXISTS {quote_identifier(name)}")
        await store.execute(f"ALTER TABLE {quoted} ADD CONSTRAINT namespaces_pkey PRIMARY KEY ({', '.join(_KEY)})")
    declared = await store.query(_DECLARATION_SQL)
    if _declared_keys(declared[0]["declared"] if declared else None) != list(_KEY):
        await store.execute(f"DROP INDEX IF EXISTS {quote_identifier(NAMESPACES.index)}")
        await store.execute(f"ALTER TABLE {quoted} DROP COLUMN IF EXISTS {BUCKET_COLUMN}")
    for statement in bucket_statements(NAMESPACES, yugabyte=await on_yugabyte(store)):
        await store.execute(statement)


async def namespaces_keyed_by_id(store: Any) -> None:
    """key ``namespaces`` by ``namespace_id`` alone, ``row_scope`` a moveable column (v007).

    :param store: DataStore bound to the platform schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await rekey_namespaces(store)
    log.info("agent_acl v007: namespaces is keyed by namespace_id; row_scope is moveable")
