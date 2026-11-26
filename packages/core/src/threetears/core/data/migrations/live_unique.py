"""make a stamped table's unique index hold its live rows only: ``... WHERE <its predicate> AND NOT gone``.

Append-only L3 (``design-append-only-l3.md`` decision 2): a stamped table's delete marks its row gone in place
(``ROW_STAMPED``), and the row stays until quiet-time cleanup removes it below the restore window. A unique index over
a natural name (a group's name, a datasource's name) still holds that gone row, so "delete X, create X" is refused
until cleanup runs. Pace 2026-11-25: **unique names apply to live rows only**. Each such index becomes partial on the
live rows, and a re-create makes a new row with a new id beside the gone one.

**What is refused** (:class:`LiveUniqueRefusedError`), each a reason the index must stay as it is:

- **a foreign key's target**: PostgreSQL (and YugabyteDB) refuse a foreign key into a partial index, so an index a
  key references (``namespaces.name``, ``owner_namespace``'s target) stays whole;
- **the primary key**: a key whose natural key IS the primary key keeps reviving its gone row in place (a grant keyed
  by its group and scope); a new row could never stand beside it;
- **an index that is not unique**, and **a table with no** ``gone`` **column** (not stamped).

**The swap**, each step its own statement (the YugabyteDB DDL rule of :mod:`~threetears.core.data.migrations.helpers`):
the live-only index is built under a temporary name from the index's own definition (``pg_get_indexdef``: its
columns, method and storage hints exactly, ``AND NOT gone`` added to its predicate), the old index (or the ``UNIQUE``
constraint it backs) is dropped, and the new one takes its name and its comment. WHY build first: the table is never
without the uniqueness. A live-only index holds a subset of the rows the whole one held, so its build never meets a
duplicate. **Idempotent**: an index already live-only is left as it is; a run stopped after the drop finds the
temporary index and finishes with its rename.

**ON CONFLICT targets** naming the columns must name the predicate too (``ON CONFLICT (cols) WHERE NOT gone``):
PostgreSQL infers a partial unique index as the arbiter only from a predicate that implies it. A whole index satisfies
that clause as well, so the clause may ship before the migration runs.

It imports nothing but the catalog SQL, the gone column's name and the logger, so ``threetears.core.data.migrations``
exports it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from threetears.core.ddl.row_epoch import GONE_COLUMN
from threetears.core.sql_fragments import quote_identifier
from threetears.observe import get_logger

__all__ = [
    "LIVE_ONLY_PREDICATE",
    "LiveUniqueRefusedError",
    "live_only_definition",
    "make_unique_index_live_only",
    "make_unique_indexes_live_only",
    "temporary_name",
]

log = get_logger(__name__)

#: the predicate a live-only unique index adds: a stamped table's live rows
LIVE_ONLY_PREDICATE: Final[str] = f"NOT {GONE_COLUMN}"

#: a predicate already holding the live rows only, as the catalog prints it (``(NOT gone)``)
_ALREADY_LIVE: Final = re.compile(rf"\bNOT {GONE_COLUMN}\b")

#: the head ``pg_get_indexdef`` prints, its name group replaced by the temporary name
_HEAD: Final = re.compile(r"^CREATE UNIQUE INDEX (?P<name>\S+) ON ")

#: PostgreSQL's identifier limit, bytes
_MAX_IDENTIFIER: Final[int] = 63

#: the suffix the live-only index is built under before it takes the old one's name
_TEMPORARY_SUFFIX: Final[str] = "_liveonly"

#: the index as the catalog holds it, by name within one schema (the migration's ``current_schema()`` when none given)
_INDEX_SQL: Final[str] = f"""
SELECT n.nspname AS schema_name,
       t.relname AS table_name,
       pg_get_indexdef(i.indexrelid) AS definition,
       pg_get_expr(i.indpred, i.indrelid) AS predicate,
       i.indisunique AS is_unique,
       i.indisprimary AS is_primary,
       i.indisvalid AS is_valid,
       (SELECT k.conname FROM pg_constraint k
         WHERE k.conindid = i.indexrelid AND k.conrelid = i.indrelid AND k.contype = 'u') AS constraint_name,
       (SELECT string_agg(f.conrelid::regclass::text || '.' || f.conname, ', ' ORDER BY f.conname)
          FROM pg_constraint f WHERE f.contype = 'f' AND f.conindid = i.indexrelid) AS foreign_keys,
       obj_description(i.indexrelid, 'pg_class') AS note,
       EXISTS (SELECT 1 FROM pg_attribute a
                WHERE a.attrelid = i.indrelid AND a.attname = '{GONE_COLUMN}' AND NOT a.attisdropped) AS stamped
  FROM pg_index i
  JOIN pg_class c ON c.oid = i.indexrelid
  JOIN pg_class t ON t.oid = i.indrelid
  JOIN pg_namespace n ON n.oid = c.relnamespace
 WHERE c.relname = $1 AND n.nspname = COALESCE($2::text, current_schema())
"""


class LiveUniqueRefusedError(ValueError):
    """an index that cannot be made live-only: a foreign key's target, the primary key, not unique, not stamped, or
    missing.

    :ivar index: the index
    """

    def __init__(self, index: str, reason: str) -> None:
        """name the index and why it stays as it is.

        :param index: the index
        :ptype index: str
        :param reason: why
        :ptype reason: str
        :return: nothing
        :rtype: None
        """
        super().__init__(f"unique index {index} cannot hold its live rows only: {reason}")
        self.index = index


@dataclass(frozen=True, slots=True)
class _Held:
    """the index as the catalog holds it (:data:`_INDEX_SQL`)."""

    schema_name: str
    table_name: str
    definition: str
    predicate: str | None
    is_unique: bool
    is_primary: bool
    is_valid: bool
    constraint_name: str | None
    foreign_keys: str | None
    note: str | None
    stamped: bool


def temporary_name(index: str) -> str:
    """the name the live-only index is built under, within the identifier limit.

    :param index: the index
    :ptype index: str
    :return: its temporary name
    :rtype: str
    """
    return f"{index[: _MAX_IDENTIFIER - len(_TEMPORARY_SUFFIX)]}{_TEMPORARY_SUFFIX}"


def live_only_definition(definition: str, predicate: str | None, name: str) -> str:
    """``definition`` (as ``pg_get_indexdef`` prints it) built as ``name``, its predicate the live rows' too.

    :param definition: the index's own definition
    :ptype definition: str
    :param predicate: its predicate as ``pg_get_expr`` prints it, None for none
    :ptype predicate: str | None
    :param name: the name to build it as
    :ptype name: str
    :return: ``CREATE UNIQUE INDEX IF NOT EXISTS <name> ON ... WHERE (<predicate>) AND (NOT gone)``
    :rtype: str
    :raises ValueError: when the definition is not a unique index's, or does not end in its predicate
    """
    if _HEAD.match(definition) is None:
        raise ValueError(f"not a unique index's definition: {definition}")
    body = definition
    if predicate is not None:
        tail = f" WHERE {predicate}"
        if not definition.endswith(tail):
            raise ValueError(f"the definition does not end in its predicate {predicate!r}: {definition}")
        body = definition[: -len(tail)]
    where = f"({predicate}) AND ({LIVE_ONLY_PREDICATE})" if predicate is not None else f"({LIVE_ONLY_PREDICATE})"
    renamed = _HEAD.sub(f"CREATE UNIQUE INDEX IF NOT EXISTS {quote_identifier(name)} ON ", body, count=1)
    return f"{renamed} WHERE {where}"


async def _held(store: Any, index: str, schema: str | None) -> _Held | None:
    """the index by name in ``schema`` (the store's current schema when None), or None when there is none.

    :param store: the migration's store, with its ``query``
    :ptype store: Any
    :param index: the index
    :ptype index: str
    :param schema: its schema
    :ptype schema: str | None
    :return: the index, or None
    :rtype: _Held | None
    """
    rows = await store.query(_INDEX_SQL, index, schema)
    if not rows:
        return None
    row = rows[0]
    return _Held(
        schema_name=str(row["schema_name"]),
        table_name=str(row["table_name"]),
        definition=str(row["definition"]),
        predicate=None if row["predicate"] is None else str(row["predicate"]),
        is_unique=bool(row["is_unique"]),
        is_primary=bool(row["is_primary"]),
        is_valid=bool(row["is_valid"]),
        constraint_name=None if row["constraint_name"] is None else str(row["constraint_name"]),
        foreign_keys=None if row["foreign_keys"] is None else str(row["foreign_keys"]),
        note=None if row["note"] is None else str(row["note"]),
        stamped=bool(row["stamped"]),
    )


def _refusal(held: _Held) -> str | None:
    """why ``held`` cannot be made live-only, or None when it can.

    :param held: the index
    :ptype held: _Held
    :return: the reason, or None
    :rtype: str | None
    """
    if held.is_primary:
        return "it is the primary key: a gone row of that key is revived in place, never stood beside"
    if not held.is_unique:
        return "it is not unique"
    if held.foreign_keys:
        return f"it is a foreign key's target ({held.foreign_keys}), and no foreign key may reference a partial index"
    if not held.stamped:
        return f"its table {held.table_name} carries no {GONE_COLUMN} column (it is not stamped)"
    return None


async def make_unique_index_live_only(store: Any, index: str, *, schema: str | None = None) -> bool:
    """make one unique index hold its live rows only (``AND NOT gone``), keeping its name, columns and comment.

    :param store: the migration's store (``DataStore``), with ``query`` and ``execute``, bound to the schema
    :ptype store: Any
    :param index: the index (or the ``UNIQUE`` constraint's index)
    :ptype index: str
    :param schema: its schema; the store's current schema when None
    :ptype schema: str | None
    :return: whether it was changed (False: already live-only)
    :rtype: bool
    :raises LiveUniqueRefusedError: when it is missing, a foreign key's target, the primary key, not unique, or on a
        table with no gone column
    """
    temporary = temporary_name(index)
    held = await _held(store, index, schema)
    built = await _held(store, temporary, schema)
    if held is None:
        if built is not None and built.is_valid:
            # a run stopped after the drop: the live-only index stands under its temporary name
            await store.execute(
                f"ALTER INDEX {quote_identifier(built.schema_name)}.{quote_identifier(temporary)} "
                f"RENAME TO {quote_identifier(index)}"
            )
            log.info(
                "a unique index holds its live rows only (its rename finished)", extra={"extra_data": {"index": index}}
            )
            return True
        raise LiveUniqueRefusedError(index, "no such index")
    refused = _refusal(held)
    if refused is not None:
        raise LiveUniqueRefusedError(index, refused)
    if held.predicate is not None and _ALREADY_LIVE.search(held.predicate):
        return False
    qualified_schema = quote_identifier(held.schema_name)
    if built is not None and not built.is_valid:
        # a build a failed run left behind holds nothing: built again
        await store.execute(f"DROP INDEX {qualified_schema}.{quote_identifier(temporary)}")
    await store.execute(live_only_definition(held.definition, held.predicate, temporary))
    if held.constraint_name is not None:
        await store.execute(
            f"ALTER TABLE {qualified_schema}.{quote_identifier(held.table_name)} "
            f"DROP CONSTRAINT {quote_identifier(held.constraint_name)}"
        )
    else:
        await store.execute(f"DROP INDEX {qualified_schema}.{quote_identifier(index)}")
    await store.execute(
        f"ALTER INDEX {qualified_schema}.{quote_identifier(temporary)} RENAME TO {quote_identifier(index)}"
    )
    if held.note is not None:
        escaped = held.note.replace("'", "''")
        await store.execute(f"COMMENT ON INDEX {qualified_schema}.{quote_identifier(index)} IS '{escaped}'")
    log.info(
        "a unique index holds its live rows only",
        extra={
            "extra_data": {
                "index": index,
                "table": held.table_name,
                "was_constraint": held.constraint_name is not None,
                "predicate": held.predicate,
            }
        },
    )
    return True


async def make_unique_indexes_live_only(store: Any, indexes: Sequence[str], *, schema: str | None = None) -> int:
    """:func:`make_unique_index_live_only` for each of ``indexes``, in order. Every one is checked before any changes:
    one refused index refuses the whole list, nothing changed.

    :param store: the migration's store, bound to the schema
    :ptype store: Any
    :param indexes: the indexes, frozen as the migration names them
    :ptype indexes: Sequence[str]
    :param schema: their schema; the store's current schema when None
    :ptype schema: str | None
    :return: how many were changed
    :rtype: int
    :raises LiveUniqueRefusedError: as :func:`make_unique_index_live_only`, before any index is changed
    """
    for index in indexes:
        held = await _held(store, index, schema)
        if held is None:
            built = await _held(store, temporary_name(index), schema)
            if built is None or not built.is_valid:
                raise LiveUniqueRefusedError(index, "no such index")
            continue
        refused = _refusal(held)
        if refused is not None:
            raise LiveUniqueRefusedError(index, refused)
    changed = 0
    for index in indexes:
        if await make_unique_index_live_only(store, index, schema=schema):
            changed += 1
    return changed
