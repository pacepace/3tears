"""find the indexes a failed build left unusable, and build each again from its own definition.

An index build that fails part way leaves its index behind marked not valid, not ready or not live
(``pg_index.indisvalid`` / ``indisready`` / ``indislive``). The planner never reads it, a unique one enforces
nothing, and ``CREATE [UNIQUE] INDEX IF NOT EXISTS`` takes it as present, so a migration that declared it never
heals it. YugabyteDB's docs say to drop such an index and create it again ("If online CREATE INDEX fails, an
invalid index may be left behind"); on YugabyteDB 2026.1 an online build racing other DDL on the database was also
seen to end that way with no error at all.

:func:`rebuild_unusable_indexes` finds every such index in the named schemas and builds it again from the
definition the catalog keeps for it (``pg_get_indexdef``): the very statement that declared it, so nothing is
re-declared by hand. A unique index whose table now holds duplicate keys under its predicate is refused before
anything is dropped, naming the duplicate keys: building it would fail, and dropping it first would lose even the
definition. An index still unusable after its build raises.

An installer that declares its own indexes (the change-note ring) passes their statements as ``definitions``: an
unusable index of one of those names is built again from the installer's declared statement, not from whatever
definition a failed build left under that name.

The duplicate check reads the table under the index's predicate, grouped by its key: a schema repair's own read,
run only for an index found unusable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from threetears.core.ddl.ddl_lock import DdlLockPolicy, database_ddl_lock
from threetears.core.declared import whole_read
from threetears.core.ddl.session import ConnectionSession
from threetears.core.sql_fragments import quote_identifier
from threetears.observe import get_logger

__all__ = [
    "IndexDuplicateKeysError",
    "IndexStillUnusableError",
    "UnusableIndex",
    "ddl_body",
    "rebuild_unusable_indexes",
    "unusable_indexes",
]

log = get_logger(__name__)


def ddl_body(statement: str) -> str:
    """a DDL statement past the ``--`` comment lines it opens with (the ``-- purpose:`` / ``-- plan_test:`` /
    ``-- reason:`` lines declaring what reads an index): the statement the database runs, as it starts.

    :param statement: the statement
    :ptype statement: str
    :return: its body
    :rtype: str
    """
    return "\n".join(line for line in statement.splitlines() if not line.lstrip().startswith("--")).lstrip()


#: how many duplicate keys a refusal names
_DUPLICATES_NAMED: Final = 10

#: every index of the schemas that is not valid, ready and live, with what rebuilding it needs
_UNUSABLE_SQL: Final = """
SELECT n.nspname AS schema_name,
       t.relname AS table_name,
       c.relname AS index_name,
       i.indisunique AS is_unique,
       pg_get_indexdef(i.indexrelid) AS definition,
       pg_get_expr(i.indpred, i.indrelid) AS predicate,
       ARRAY(
           SELECT pg_get_indexdef(i.indexrelid, k.position::int, true)
             FROM generate_series(1, i.indnkeyatts) AS k(position)
       ) AS key_terms
  FROM pg_index i
  JOIN pg_class c ON c.oid = i.indexrelid
  JOIN pg_class t ON t.oid = i.indrelid
  JOIN pg_namespace n ON n.oid = c.relnamespace
 WHERE n.nspname::text = ANY($1::text[])
   AND ($2::text[] IS NULL OR c.relname::text = ANY($2::text[]))
   AND NOT (i.indisvalid AND i.indisready AND i.indislive)
 ORDER BY n.nspname, c.relname
"""


@dataclass(frozen=True, slots=True)
class UnusableIndex:
    """an index a failed build left not valid, not ready or not live.

    :ivar schema: its schema
    :ivar table: its table
    :ivar name: its name
    :ivar unique: whether it is a unique index
    :ivar definition: the statement that builds it again (``pg_get_indexdef``)
    :ivar predicate: its partial-index predicate, ``None`` for a whole-table index
    :ivar key_terms: its key columns or expressions, in key order
    """

    schema: str
    table: str
    name: str
    unique: bool
    definition: str
    predicate: str | None
    key_terms: tuple[str, ...]


class IndexDuplicateKeysError(RuntimeError):
    """a unique index cannot be built again: its table holds duplicate keys under its predicate.

    :ivar index: the index
    :ivar duplicates: the duplicate keys found (at most :data:`_DUPLICATES_NAMED`), each with its row count
    """

    def __init__(self, index: UnusableIndex, duplicates: Sequence[tuple[tuple[Any, ...], int]]) -> None:
        self.index = index
        self.duplicates = tuple(duplicates)
        named = "; ".join(f"{key!r} x{count}" for key, count in self.duplicates)
        super().__init__(
            f"unique index {index.schema}.{index.name} is unusable and cannot be built again: "
            f"{index.table} holds duplicate keys ({', '.join(index.key_terms)}): {named}. Resolve them, then repeat"
        )


class IndexStillUnusableError(RuntimeError):
    """an index was built again and is still not valid, ready and live."""


async def unusable_indexes(
    conn: Any, schemas: Sequence[str], *, names: Sequence[str] | None = None
) -> list[UnusableIndex]:
    """every index of ``schemas`` that is not valid, ready and live.

    :param conn: a connection to the database, or anything with ``fetch`` answering rows by column name
    :ptype conn: Any
    :param schemas: the schemas to look in
    :ptype schemas: Sequence[str]
    :param names: only the indexes of these names (an installer's own); every index of the schemas when None
    :ptype names: Sequence[str] | None
    :return: the indexes, by schema and name
    :rtype: list[UnusableIndex]
    """
    rows = await conn.fetch(_UNUSABLE_SQL, list(schemas), None if names is None else list(names))
    return [
        UnusableIndex(
            schema=row["schema_name"],
            table=row["table_name"],
            name=row["index_name"],
            unique=bool(row["is_unique"]),
            definition=row["definition"],
            predicate=row["predicate"],
            key_terms=tuple(row["key_terms"]),
        )
        for row in rows
    ]


async def _duplicate_keys(conn: Any, index: UnusableIndex) -> list[tuple[tuple[Any, ...], int]]:
    """the keys a unique index would refuse: its table's rows under its predicate, grouped by its key terms.

    :param conn: a connection to the database
    :ptype conn: Any
    :param index: the unique index
    :ptype index: UnusableIndex
    :return: up to :data:`_DUPLICATES_NAMED` duplicate keys, each with its row count
    :rtype: list[tuple[tuple[Any, ...], int]]
    """
    terms = ", ".join(index.key_terms)
    named = ", ".join(f"{term} AS k{position}" for position, term in enumerate(index.key_terms))
    where = f" WHERE {index.predicate}" if index.predicate else ""
    # NULLs never collide in a unique index, so a key holding one is no duplicate
    not_null = " AND ".join(f"({term}) IS NOT NULL" for term in index.key_terms)
    where = f"{where} AND {not_null}" if where else f" WHERE {not_null}"
    rows = await whole_read(
        conn.fetch(
            f"SELECT {named}, count(*) AS n FROM {quote_identifier(index.schema)}.{quote_identifier(index.table)}"  # noqa: S608
            f"{where} GROUP BY {terms} HAVING count(*) > 1 LIMIT {_DUPLICATES_NAMED}"
        ),
        kind="schema_change",
        reason="a unique index a failed build left unusable is built again only once its table holds no duplicate key",
        ruling="Pace 2026-11-12: a schema change may read whole; coordinator 2026-11-12: schema_change admits unknown-schema",
        proof="packages/core/tests/integration/test_index_health_live.py::test_the_duplicate_key_read_is_reached_only_for_an_unusable_unique_index",
        speed="packages/core/tests/integration/test_index_health_live.py::test_rebuild_speed_at_real_size",
    )
    return [(tuple(row[f"k{i}"] for i in range(len(index.key_terms))), int(row["n"])) for row in rows]


async def rebuild_unusable_indexes(
    conn: Any,
    schemas: Sequence[str],
    *,
    names: Sequence[str] | None = None,
    policy: DdlLockPolicy | None = None,
    definitions: Mapping[str, str] | None = None,
) -> list[UnusableIndex]:
    """build every unusable index of ``schemas`` again from its own definition; say which were rebuilt.

    Every unique index is checked for duplicate keys before any index is dropped, so a refusal leaves every index as
    it was. Each rebuild is said at WARNING. The drops and builds run under the database's DDL lock
    (:func:`~threetears.core.ddl.ddl_lock.database_ddl_lock`), taken here on ``conn``: two index builds in one
    YugabyteDB database leave both invalid. The lock is re-entrant on its session, so a caller already holding it (a
    migration run) loses nothing; a repair that finds nothing to build takes no lock, and one that does looks again
    once it holds it, so two starts racing build each index once.

    :param conn: ONE connection allowed DDL, outside a transaction (YugabyteDB builds an index on a populated table
        online only there), never a pool: the lock is held on its session
    :ptype conn: Any
    :param schemas: the schemas to repair
    :ptype schemas: Sequence[str]
    :param names: only the indexes of these names (an installer's own, as the change-note ring's); every unusable index
        of the schemas when None
    :ptype names: Sequence[str] | None
    :param policy: how long to wait for the DDL lock; the default waits until it is free
    :ptype policy: DdlLockPolicy | None
    :param definitions: the declared statement per index name (an installer's own): an unusable index of one of these
        names is built from it rather than from the definition the catalog holds under that name, which a failed build
        of something else may have left; a declared statement that is not UNIQUE needs no duplicate check
    :ptype definitions: Mapping[str, str] | None
    :return: the indexes built again
    :rtype: list[UnusableIndex]
    :raises IndexDuplicateKeysError: when a unique index's table holds duplicate keys; nothing was dropped
    :raises IndexStillUnusableError: when an index is still unusable after it was built again
    :raises SessionRequiredError: when ``conn`` is a pool
    """
    if not await unusable_indexes(conn, schemas, names=names):
        return []
    # looked at again under the lock: another start may have built them meanwhile
    async with database_ddl_lock(ConnectionSession(conn), policy):
        found = await unusable_indexes(conn, schemas, names=names)
        if not found:
            return []
        declared = definitions or {}
        for index in found:
            unique = (
                ddl_body(declared[index.name]).upper().startswith("CREATE UNIQUE ")
                if index.name in declared
                else index.unique
            )
            if unique:
                duplicates = await whole_read(
                    _duplicate_keys(conn, index),
                    kind="schema_change",
                    reason="an unusable unique index is checked for duplicate keys before any index is dropped and built",
                    ruling="Pace 2026-11-12: a schema change may read whole; coordinator 2026-11-12: schema_change admits unknown-schema",
                    proof="packages/core/tests/integration/test_index_health_live.py::test_the_duplicate_key_read_is_reached_only_for_an_unusable_unique_index",
                    speed="packages/core/tests/integration/test_index_health_live.py::test_rebuild_speed_at_real_size",
                )
                if duplicates:
                    raise IndexDuplicateKeysError(index, duplicates)
        log.warning(
            "indexes left unusable by a failed build; dropping and building each again from its definition",
            extra={"extra_data": {"indexes": [f"{index.schema}.{index.name}" for index in found]}},
        )
        for index in found:
            await conn.execute(f"DROP INDEX IF EXISTS {quote_identifier(index.schema)}.{quote_identifier(index.name)}")
            await conn.execute(declared.get(index.name, index.definition))
        still = {(i.schema, i.name) for i in await unusable_indexes(conn, schemas, names=names)} & {
            (i.schema, i.name) for i in found
        }
        if still:
            raise IndexStillUnusableError(f"indexes still unusable after they were built again: {sorted(still)}")
        return found
