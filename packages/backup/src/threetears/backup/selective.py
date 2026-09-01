"""Selective restore — rows back from a backup, not databases.

The request that actually arrives is never "restore the cluster"; it is "restore THIS row", or
"restore what that migration ate between 02:00 and 02:15", or "restore ids in this range". A dump
is an opaque stream with no index, so selection cannot happen inside it — and does not need to.
The tractable shape: restore the relevant database's dump into a SCRATCH database (the same
machinery the verifier trusts), then treat the scratch as a queryable snapshot and move rows from
it into the live database by ordinary SQL.

Selection speaks three vocabularies, all row-grain:

- explicit ids (``ids=...``) — "this one row" and its friends;
- an id RANGE (``id_range=...``) — uuid7 ids are time-ordered and the uuid type compares
  bytewise, so an id range IS a time range over creation, with index locality for free;
- a date range over a named timestamp column (``date_range=...``).

The plan/apply split is the safety model: :meth:`SelectiveRestore.plan` only reads, classifying
every selected row as an INSERT (absent live), an UPDATE (present but different), or IDENTICAL
(present and equal, written by neither path). :meth:`SelectiveRestore.apply` takes a plan and
upserts exactly its insert+update rows in one transaction. A caller that shows the plan to a
human before applying has a dry-run for free.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from threetears.observe import get_logger

__all__ = [
    "RowSelection",
    "SelectionTooLargeError",
    "SelectiveRestore",
    "SelectiveRestorePlan",
    "PlannedRow",
]

log = get_logger(__name__)

#: SQL identifiers are quoted, and anything that could escape the quoting is refused outright.
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")

_PK_SQL = """
    SELECT a.attname
      FROM pg_index i
      JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
     WHERE i.indrelid = ($1 || '.' || $2)::regclass
       AND i.indisprimary
     ORDER BY array_position(i.indkey, a.attnum)
"""

_COLUMNS_SQL = """
    SELECT column_name
      FROM information_schema.columns
     WHERE table_schema = $1 AND table_name = $2
     ORDER BY ordinal_position
"""


class SelectionTooLargeError(RuntimeError):
    """The selection matched more rows than the caller's bound — refuse rather than surprise."""


@runtime_checkable
class _Connection(Protocol):
    async def fetch(self, query: str, *args: object) -> list[Any]: ...
    async def execute(self, query: str, *args: object) -> str: ...
    async def close(self) -> None: ...
    def transaction(self) -> Any: ...


Connect = Callable[[str], Awaitable[_Connection]]


def _quote(name: str) -> str:
    if not _IDENTIFIER.match(name):
        raise ValueError(f"unsafe SQL identifier {name!r}")
    return f'"{name}"'


@dataclass(frozen=True, slots=True)
class RowSelection:
    """What to bring back: one table, one predicate.

    Exactly one of ``ids`` / ``id_range`` / ``date_range`` / ``all_rows`` must be given.

    :param table: the table to select from.
    :param schema: its schema (default ``public``).
    :param id_column: the column the id predicates address. ``None`` means "the table's single
        primary-key column"; a table with a composite key (the platform's ``(customer_id, id)``
        shape) must name it explicitly, because "the id" is ambiguous there.
    :param ids: explicit id values — the "restore this one row" case.
    :param id_range: inclusive ``(low, high)`` over ``id_column``. uuid7 values compare in
        creation order, so this doubles as a creation-time range.
    :param date_range: ``(column, low, high)`` inclusive over a timestamp column.
    :param all_rows: the whole table — the coarsest selection, still row-grain underneath.
    :param where: a raw SQL predicate with ``$1``-style placeholders, bound with
        ``where_params`` — the escape hatch for the selection no fixed vocabulary fits
        ("``status = $1 AND customer_id = $2``"). Safe BY PLACEMENT rather than by parsing:
        it executes only against the scratch snapshot, whose blast radius is a throwaway
        restored copy; the rows it selects still flow through the same plan/apply
        classification, bounds, and structured upsert as every other vocabulary.
    :param where_params: bind parameters for ``where``.
    """

    table: str
    schema: str = "public"
    id_column: str | None = None
    ids: tuple[Any, ...] | None = None
    id_range: tuple[Any, Any] | None = None
    date_range: tuple[str, datetime, datetime] | None = None
    all_rows: bool = False
    where: str | None = None
    where_params: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        chosen = [
            self.ids is not None,
            self.id_range is not None,
            self.date_range is not None,
            self.all_rows,
            self.where is not None,
        ]
        if sum(chosen) != 1:
            raise ValueError("exactly one of ids / id_range / date_range / all_rows / where must be given")
        if self.ids is not None and len(self.ids) == 0:
            # empty is not absent: an empty id list selects nothing, and a selection that selects
            # nothing is a mistake worth refusing loudly rather than a no-op restore.
            raise ValueError("ids must not be empty")
        if self.where is not None and not self.where.strip():
            raise ValueError("where must not be blank")


@dataclass(frozen=True, slots=True)
class PlannedRow:
    """One selected row and what applying it would do."""

    key: tuple[Any, ...]
    action: str  # "insert" | "update" | "identical"
    row: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SelectiveRestorePlan:
    """The read-only outcome of planning a selection — show it to a human, then apply it.

    ``apply`` writes only ``inserts`` and ``updates``; ``identical`` rows are recorded so the
    human can see the selection matched what they expected, and are never written.
    """

    schema: str
    table: str
    pk_columns: tuple[str, ...]
    columns: tuple[str, ...]
    inserts: tuple[PlannedRow, ...] = field(default_factory=tuple)
    updates: tuple[PlannedRow, ...] = field(default_factory=tuple)
    identical: tuple[PlannedRow, ...] = field(default_factory=tuple)

    @property
    def write_count(self) -> int:
        """How many rows applying this plan would write."""
        return len(self.inserts) + len(self.updates)


class SelectiveRestore:
    """Move selected rows from a restored scratch database into the live one.

    :param connect: an ``asyncpg.connect``-shaped callable (injected, as everywhere in this
        package).
    :param scratch_dsn: the database a backup was restored into — the queryable snapshot.
    :param live_dsn: the database rows are restored TO.
    :param max_rows: refuse selections matching more rows than this
        (:class:`SelectionTooLargeError`) — a fat-fingered range should fail loudly, not quietly
        rewrite a table.
    """

    def __init__(self, connect: Connect, scratch_dsn: str, live_dsn: str, *, max_rows: int = 10_000) -> None:
        self._connect = connect
        self._scratch_dsn = scratch_dsn
        self._live_dsn = live_dsn
        self._max_rows = max_rows

    async def plan(self, selection: RowSelection) -> SelectiveRestorePlan:
        """Read-only: select rows from the scratch snapshot and classify them against live.

        :raises SelectionTooLargeError: when the selection exceeds ``max_rows``.
        :raises ValueError: when ``id_column`` is needed but not given, or names are unsafe.
        """
        scratch = await self._connect(self._scratch_dsn)
        try:
            pk_columns = await self._pk_columns(scratch, selection.schema, selection.table)
            columns = await self._columns(scratch, selection.schema, selection.table)
            where, params = self._predicate(selection, pk_columns)
            qualified = f"{_quote(selection.schema)}.{_quote(selection.table)}"
            rows = await scratch.fetch(f"SELECT * FROM {qualified} WHERE {where}", *params)
        finally:
            await scratch.close()
        if len(rows) > self._max_rows:
            raise SelectionTooLargeError(
                f"selection matched {len(rows)} rows; the bound is {self._max_rows} — narrow the "
                f"selection or raise max_rows deliberately"
            )

        live = await self._connect(self._live_dsn)
        try:
            inserts: list[PlannedRow] = []
            updates: list[PlannedRow] = []
            identical: list[PlannedRow] = []
            pk_where = " AND ".join(f"{_quote(col)} = ${i + 1}" for i, col in enumerate(pk_columns))
            for record in rows:
                row = dict(record)
                key = tuple(row[col] for col in pk_columns)
                current = await live.fetch(
                    f"SELECT * FROM {_quote(selection.schema)}.{_quote(selection.table)} WHERE {pk_where}",
                    *key,
                )
                if not current:
                    inserts.append(PlannedRow(key=key, action="insert", row=row))
                elif dict(current[0]) == row:
                    identical.append(PlannedRow(key=key, action="identical", row=row))
                else:
                    updates.append(PlannedRow(key=key, action="update", row=row))
        finally:
            await live.close()

        plan = SelectiveRestorePlan(
            schema=selection.schema,
            table=selection.table,
            pk_columns=pk_columns,
            columns=columns,
            inserts=tuple(inserts),
            updates=tuple(updates),
            identical=tuple(identical),
        )
        log.info(
            "selective restore planned",
            extra={
                "extra_data": {
                    "table": f"{selection.schema}.{selection.table}",
                    "inserts": len(plan.inserts),
                    "updates": len(plan.updates),
                    "identical": len(plan.identical),
                }
            },
        )
        return plan

    async def apply(self, plan: SelectiveRestorePlan) -> int:
        """Upsert the plan's insert+update rows into the live database, in one transaction.

        :return: the number of rows written.
        """
        to_write = [*plan.inserts, *plan.updates]
        if not to_write:
            return 0
        qualified = f"{_quote(plan.schema)}.{_quote(plan.table)}"
        column_list = ", ".join(_quote(c) for c in plan.columns)
        placeholders = ", ".join(f"${i + 1}" for i in range(len(plan.columns)))
        conflict = ", ".join(_quote(c) for c in plan.pk_columns)
        non_key = [c for c in plan.columns if c not in plan.pk_columns]
        if non_key:
            update_set = ", ".join(f"{_quote(c)} = EXCLUDED.{_quote(c)}" for c in non_key)
            on_conflict = f"ON CONFLICT ({conflict}) DO UPDATE SET {update_set}"
        else:
            on_conflict = f"ON CONFLICT ({conflict}) DO NOTHING"
        sql = f"INSERT INTO {qualified} ({column_list}) VALUES ({placeholders}) {on_conflict}"

        live = await self._connect(self._live_dsn)
        try:
            async with live.transaction():
                for planned in to_write:
                    values = [planned.row.get(c) for c in plan.columns]
                    await live.execute(sql, *values)
        finally:
            await live.close()
        log.info(
            "selective restore applied",
            extra={"extra_data": {"table": qualified, "rows_written": len(to_write)}},
        )
        return len(to_write)

    # ------------------------------------------------------------------ internals

    async def _pk_columns(self, conn: _Connection, schema: str, table: str) -> tuple[str, ...]:
        rows = await conn.fetch(_PK_SQL, _quote(schema), _quote(table))
        pk = tuple(row["attname"] for row in rows)
        if not pk:
            raise ValueError(f"{schema}.{table} has no primary key; selective restore needs one")
        return pk

    async def _columns(self, conn: _Connection, schema: str, table: str) -> tuple[str, ...]:
        rows = await conn.fetch(_COLUMNS_SQL, schema, table)
        return tuple(row["column_name"] for row in rows)

    def _predicate(self, selection: RowSelection, pk_columns: tuple[str, ...]) -> tuple[str, list[Any]]:
        if selection.where is not None:
            return selection.where, list(selection.where_params)
        if selection.all_rows:
            return "TRUE", []
        if selection.date_range is not None:
            column, low, high = selection.date_range
            return f"{_quote(column)} >= $1 AND {_quote(column)} <= $2", [low, high]
        id_column = selection.id_column
        if id_column is None:
            if len(pk_columns) != 1:
                raise ValueError(f"table has composite primary key {pk_columns}; name id_column explicitly")
            id_column = pk_columns[0]
        if selection.ids is not None:
            return f"{_quote(id_column)} = ANY($1)", [list(selection.ids)]
        assert selection.id_range is not None
        low, high = selection.id_range
        return f"{_quote(id_column)} >= $1 AND {_quote(id_column)} <= $2", [low, high]
