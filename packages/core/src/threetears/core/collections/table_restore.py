"""a lost change log rebuilt from the versioned tables themselves: every version, gone marker and run marker since X.

Append-only L3 (``design-append-only-l3.md``, decision 3, Pace 2026-11-25): every version of a versioned table
(:func:`~threetears.core.ddl.row_epoch.row_versions`) carries its write's epoch, a hybrid logical clock stamp, so "what
changed since epoch X" is read from the table itself. A keeper's mark of the log (position P at time T) vouches that
every record appended after P carries stamps at or above X = (T - M) << 15
(:func:`~threetears.core.ddl.row_epoch.restore_floor`): so the records a lost log held after P name, between them, no
key that is not among the table's versions with ``epoch >= X``. A restore needs that set of keys, in no order; reading
too much only re-evicts a key, missing one is the only failure.

**Which tables, from the catalog.** The keeper runs in the hub, which imports no consumer's table classes. Each
versioned table's ``(scope, epoch)`` index carries the table's restore declaration as its comment
(:func:`~threetears.core.ddl.row_epoch.restore_declaration`, written by ``TableSchema.create_statements``): the scope
column, the key, the scope epochs table and prefix, the tenancy. :func:`versioned_tables` reads them in one catalog
statement.

**Which scopes, from ScopeEpochs.** Never by scanning a data table: a stamped write records its highest stamp as
``last_epoch`` on every scope it touched (``ScopeEpochs``' "Newest-wins across writers"), so the scopes holding a version
stamped at X or later are exactly those whose ``last_epoch`` is at or above X, read through the epochs table's
``last_epoch`` index (:func:`_changed_scopes`). A scope epochs scope names a table's scope as ``scope_prefix + value``
(reports: ``state:VA`` for ``join_state_code = 'VA'``); a scope of another prefix (a race) names none of this table's.
The writer's contract this rests on: a write of a row of scope value ``v`` touches ``scope_prefix + v``.

**Each scope's versions, through the (scope, epoch) index.** One statement a scope a table (:func:`_scope_versions`):
the scope bound, ``epoch >= X``, one range of the index (hashed by the scope on YugabyteDB), read as ``COPY`` text and
folded into records in the restore's worker process (:class:`~threetears.core.change_note_fold.RestoreFolder`, the fold
the change-note restore uses), so the loop moves bytes only and no row becomes a Python object. A gone marker is a
version like any other: its key comes back, and a follower evicts it. A run marker
(:mod:`~threetears.core.ddl.run_markers`) comes back as the whole-table record of its prefix (:func:`_run_marker_changes`),
read by its own key ``(table_name, scope)``. Every record names X as its ``s_min``: the lowest stamp any row it carries
may hold.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from itertools import groupby
from operator import itemgetter
from typing import Any, Final

from threetears.core.change_note_fold import RestoreFolder
from threetears.core.collections.change_notes import NotedChange
from threetears.core.collections.epoch_change import keyed_record_form
from threetears.core.ddl.change_notes import tenant_sql
from threetears.core.ddl.row_epoch import EPOCH_COLUMN, RESTORE_DECLARATION_MARK
from threetears.core.ddl.run_markers import RUN_MARKERS_TABLE
from threetears.core.keyed import AnyOf, Fragment, Selected, Sort, copy_keyed, keyset_rows
from threetears.core.sql_fragments import quote_identifier
from threetears.core.tenancy import Tenancy, schema_tenant, value_tenant
from threetears.observe import get_logger

__all__ = ["VersionedTable", "TableChanges", "versioned_tables"]

log = get_logger(__name__)

#: the bytes of a scope's ``COPY`` text handed to the worker at once: well under a MiB, so the loop's copy of a page
#: stays far inside a turn (the change-note restore's measure, ``change_note_keeper._RESTORE_PAGE_BYTES``)
_PAGE_BYTES: Final = 1 << 20

#: distinct rows of one table a change holds before it may be cut (as the change-note restore's)
_CHUNK_ROWS: Final = 5000

#: what a run marker's prefix joins its values with (:mod:`~threetears.core.ddl.run_markers`)
_PREFIX_SEPARATOR: Final = "\x1f"

#: a ``timestamptz`` key's text in a record: its UTC instant to the microsecond, as a change note spells it
_UTC_MICROS: Final = '\'YYYY-MM-DD"T"HH24:MI:SS.US"+00:00"\''

#: the row :data:`_SCOPE_EPOCHS_WHOLE` names in a scope epochs table: the data's version, never a scope
_SCOPE_EPOCHS_WHOLE: Final = "*"

#: the CSV columns the fold reads, each a JSON text (:mod:`threetears.core.change_note_fold_job`)
_FOLD_COLUMNS: Final = ("table_json", "tenant_json", "keys_json", "columns_json")

#: every versioned table the database holds, by the restore declaration its (scope, epoch) index carries as its
#: comment; and whether its schema holds the run-marker table. A catalog read, its size the declared indexes
_VERSIONED_TABLES_SQL: Final = f"""
SELECT n.nspname::text AS table_schema, c.relname::text AS table_name, d.description AS declared,
       to_regclass(quote_ident(n.nspname) || '.{RUN_MARKERS_TABLE}') IS NOT NULL AS run_markers
FROM pg_description d
JOIN pg_index i ON i.indexrelid = d.objoid
JOIN pg_class c ON c.oid = i.indrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE d.classoid = 'pg_class'::regclass AND d.objsubid = 0 AND d.description LIKE $1
ORDER BY 1, 2
"""


@dataclass(frozen=True, slots=True)
class VersionedTable:
    """one versioned table as its restore declaration names it.

    :ivar schema: its schema
    :ivar table: its name
    :ivar scope: the scope column its (scope, epoch) index leads with
    :ivar keys: its primary key's columns, in order
    :ivar instants: those of them spelled as their UTC text
    :ivar epochs: the scope epochs table of its schema
    :ivar scope_prefix: how that table names one of its scopes
    :ivar tenancy: its tenancy
    :ivar run_markers: whether its schema holds the run-marker table
    """

    schema: str
    table: str
    scope: str
    keys: tuple[str, ...]
    instants: frozenset[str]
    epochs: str
    scope_prefix: str
    tenancy: Tenancy
    run_markers: bool

    @property
    def relation(self) -> str:
        """the table as the keyed helpers name it, ``schema.table``.

        :return: it
        :rtype: str
        """
        return f"{self.schema}.{self.table}"


def _declared(schema: str, table: str, text: str, run_markers: bool) -> VersionedTable:
    """the table a restore declaration names.

    :param schema: the table's schema
    :ptype schema: str
    :param table: its name
    :ptype table: str
    :param text: its index's comment
    :ptype text: str
    :param run_markers: whether its schema holds the run-marker table
    :ptype run_markers: bool
    :return: the table
    :rtype: VersionedTable
    :raises ValueError: when the comment is not a restore declaration
    """
    value = json.loads(text)
    if not isinstance(value, dict) or value.get(RESTORE_DECLARATION_MARK) != 1:
        raise ValueError(f"{schema}.{table}'s (scope, epoch) index names no restore declaration: {text!r}")
    kind = str(value["tenancy"])
    column = value.get("tenant_column")
    tenancy = Tenancy("column", str(column)) if kind == "column" else Tenancy("schema") if kind == "schema" else None
    return VersionedTable(
        schema=schema,
        table=table,
        scope=str(value["scope"]),
        keys=tuple(str(key) for key in value["keys"]),
        instants=frozenset(str(key) for key in value["instants"]),
        epochs=str(value["epochs"]),
        scope_prefix=str(value["scope_prefix"]),
        # a table of no tenant (none, undeclared) is restored on _all, as its change notes were
        tenancy=tenancy if tenancy is not None else Tenancy("undeclared"),
        run_markers=run_markers,
    )


async def versioned_tables(pool: Any) -> list[VersionedTable]:
    """every versioned table the database holds, from the restore declarations of their (scope, epoch) indexes.

    :param pool: the keeper's L3 pool
    :ptype pool: Any
    :return: the tables, in schema and name order
    :rtype: list[VersionedTable]
    """
    rows = await pool.fetch(_VERSIONED_TABLES_SQL, f'{{"{RESTORE_DECLARATION_MARK}":%')
    return [
        _declared(str(row["table_schema"]), str(row["table_name"]), str(row["declared"]), bool(row["run_markers"]))
        for row in rows
    ]


async def _changed_scopes(pool: Any, schema: str, epochs: str, floor: int) -> list[str]:
    """the scopes of the scope epochs table ``schema.epochs`` a stamped write touched at ``floor`` or later: their
    ``last_epoch`` at or above it, read through the table's ``last_epoch`` index a page at a time.

    :param pool: the keeper's L3 pool
    :ptype pool: Any
    :param schema: the table's schema
    :ptype schema: str
    :param epochs: the scope epochs table
    :ptype epochs: str
    :param floor: X
    :ptype floor: int
    :return: the scopes, :data:`_SCOPE_EPOCHS_WHOLE` left out
    :rtype: list[str]
    """
    rows = await keyset_rows(
        pool,
        f"{schema}.{epochs}",
        columns=("last_epoch", "scope"),
        by={},
        order=[Sort("last_epoch", cast="bigint"), Sort("scope")],
        where=[Fragment('t."last_epoch" >= $1', (floor,))],
    )
    return [str(row["scope"]) for row in rows if row["scope"] != _SCOPE_EPOCHS_WHOLE]


def _literal(text: str) -> str:
    """``text`` as a SQL string literal.

    :param text: a TRUSTED name (a table's, a tenant token)
    :ptype text: str
    :return: the literal
    :rtype: str
    """
    return "'" + text.replace("'", "''") + "'"


def _key_term(table: VersionedTable, column: str) -> str:
    """a key column of a version as a record spells it: its text, a ``timestamptz`` as its UTC instant.

    :param table: the table
    :ptype table: VersionedTable
    :param column: the column
    :ptype column: str
    :return: the expression
    :rtype: str
    """
    value = f"t.{quote_identifier(column)}"
    if column in table.instants:
        return f"to_char({value} AT TIME ZONE 'UTC', {_UTC_MICROS})"
    return f"{value}::text"


def _fold_columns(table: VersionedTable) -> list[str | Selected]:
    """what a scope's ``COPY`` selects: the four JSON texts the restore's fold reads, as a change note's restore
    ``COPY`` writes them (the table, the tenant, the key's values, the invalidation columns).

    :param table: the table
    :ptype table: VersionedTable
    :return: the columns
    :rtype: list[str | Selected]
    """
    tenancy = table.tenancy
    if tenancy.kind == "schema":
        tenant = f"to_json({_literal(schema_tenant(table.schema))}::text)::text"
    elif tenancy.kind == "column":
        tenant = f"to_json({tenant_sql(tenancy, 't')})::text"
    else:
        tenant = "NULL::text"
    keys = ", ".join(_key_term(table, column) for column in table.keys)
    # a column tenancy's tenant column is an invalidation column of every record of the table, as its notes carried it
    columns = (
        f"json_build_object({_literal(str(tenancy.column))}, t.{quote_identifier(str(tenancy.column))}::text)::text"
        if tenancy.kind == "column"
        else "NULL::text"
    )
    expressions = (
        f"to_json({_literal(table.table)}::text)::text",
        tenant,
        f"array_to_json(ARRAY[{keys}])::text",
        columns,
    )
    return [Selected(expression, alias) for expression, alias in zip(expressions, _FOLD_COLUMNS, strict=True)]


async def _scope_versions(pool: Any, table: VersionedTable, value: str, floor: int) -> bytes:
    """every version (gone markers included) of ``table`` in the scope ``value`` stamped at ``floor`` or later, as the
    fold's ``COPY`` text: the scope bound, one range of the ``(scope, epoch)`` index.

    :param pool: the keeper's L3 pool
    :ptype pool: Any
    :param table: the table
    :ptype table: VersionedTable
    :param value: the scope's value
    :ptype value: str
    :param floor: X
    :ptype floor: int
    :return: the text, a line per version
    :rtype: bytes
    """
    return await copy_keyed(
        pool,
        table.relation,
        columns=_fold_columns(table),
        by={table.scope: value},
        # the index's own order after its bound scope: one range, no sort
        order=[Sort(EPOCH_COLUMN, cast="bigint")],
        where=[Fragment(f't."{EPOCH_COLUMN}" >= $1', (floor,))],
        force_quote=_FOLD_COLUMNS[:3],
    )


async def _run_marker_changes(pool: Any, table: VersionedTable, values: Sequence[str], floor: int) -> list[NotedChange]:
    """the runs of ``table`` deleted at ``floor`` or later in the scopes ``values``, as whole-table changes of their
    prefixes, one per tenant: read by the run-marker table's key ``(table_name, scope)``.

    :param pool: the keeper's L3 pool
    :ptype pool: Any
    :param table: the table
    :ptype table: VersionedTable
    :param values: the scopes' values
    :ptype values: Sequence[str]
    :param floor: X, named as the changes' ``s_min``
    :ptype floor: int
    :return: the changes
    :rtype: list[NotedChange]
    """
    if not table.run_markers or not values:
        return []
    rows = await keyset_rows(
        pool,
        f"{table.schema}.{RUN_MARKERS_TABLE}",
        columns=("scope", "prefix", "epoch"),
        by={"table_name": table.table, "scope": AnyOf(list(values))},
        order=[Sort("scope"), Sort("prefix"), Sort("epoch", cast="bigint")],
        where=[Fragment('t."epoch" >= $1', (floor,))],
    )
    tenancy = table.tenancy
    prefixes: dict[str | None, dict[tuple[tuple[str, str], ...], None]] = {}
    for row in rows:
        parts = str(row["prefix"]).split(_PREFIX_SEPARATOR)
        named = dict(zip(table.keys, parts, strict=False))
        if tenancy.kind == "schema":
            tenant: str | None = schema_tenant(table.schema)
        elif tenancy.kind == "column" and str(tenancy.column) in named:
            tenant = value_tenant(named[str(tenancy.column)])
        else:
            tenant = None
        prefixes.setdefault(tenant, {})[tuple(named.items())] = None
    return [
        NotedChange(
            table=table.table,
            tenant=tenant,
            whole_table=True,
            prefixes=tuple(dict(prefix) for prefix in held),
            s_min=floor,
        )
        for tenant, held in prefixes.items()
    ]


class TableChanges:
    """the versioned tables' changes since X, as a stream of changes a restore appends: each table's run markers, then
    its scopes' versions a scope at a time, folded in the restore's worker.

    :ivar tables: the tables a change was streamed of
    :ivar scopes: the scope reads made (one statement each)
    """

    def __init__(
        self,
        pool: Any,
        tables: Sequence[VersionedTable],
        floor: int,
        folder: RestoreFolder,
        database: str | None,
    ) -> None:
        """capture what the stream reads; no I/O.

        :param pool: the keeper's L3 pool
        :ptype pool: Any
        :param tables: the versioned tables
        :ptype tables: Sequence[VersionedTable]
        :param floor: X: every version stamped at or after it is read
        :ptype floor: int
        :param folder: the restore's worker
        :ptype folder: RestoreFolder
        :param database: the database the records name
        :ptype database: str | None
        """
        self._pool = pool
        self._tables = tuple(tables)
        self._floor = floor
        self._folder = folder
        form = keyed_record_form(database, s_min=floor)
        self._form = form
        self._pieces = (form.head, form.middle, form.tail, form.row_head, form.row_columns, form.row_tail)
        self.tables = 0
        self.scopes = 0

    def __aiter__(self) -> AsyncIterator[NotedChange]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[NotedChange]:
        # the scopes each epochs table names changed, read once a table even when several tables share it
        changed: dict[tuple[str, str], list[str]] = {}
        for table in self._tables:
            named = (table.schema, table.epochs)
            if named not in changed:
                changed[named] = await _changed_scopes(self._pool, table.schema, table.epochs, self._floor)
            prefix = table.scope_prefix
            values = [scope[len(prefix) :] for scope in changed[named] if scope.startswith(prefix)]
            streamed = False
            for change in await _run_marker_changes(self._pool, table, values, self._floor):
                streamed = True
                yield change
            for value in values:
                self.scopes += 1
                text = await _scope_versions(self._pool, table, value, self._floor)
                async for change in self._folded(text):
                    streamed = True
                    yield change
            if streamed:
                self.tables += 1

    async def _folded(self, text: bytes) -> AsyncIterator[NotedChange]:
        """one scope's ``COPY`` text folded into changes a page at a time, each page cut at a line's end.

        :param text: the text
        :ptype text: bytes
        :return: the changes
        :rtype: AsyncIterator[NotedChange]
        """
        carried = b""
        offset = 0
        while offset < len(text) or carried:
            room = max(_PAGE_BYTES, 2 * len(carried)) - len(carried)
            if offset + room >= len(text):
                end = len(text)
            else:
                end = text.rfind(b"\n", offset, offset + room) + 1
                if end <= offset:
                    # one line longer than a page: up to its end
                    end = text.find(b"\n", offset + room) + 1 or len(text)
            page = carried + text[offset:end]
            offset = end
            final = offset >= len(text)
            folded, records = await self._folder.fold(
                page, final=final, form=self._pieces, chunk_rows=_CHUNK_ROWS, max_bytes=self._form.max_bytes
            )
            carried = page[folded:]
            for _number, chunk in groupby(records, key=itemgetter(0)):
                held = list(chunk)
                _, name, tenant, _ = held[0]
                yield NotedChange(table=name, tenant=tenant, bodies=tuple(body for *_, body in held), s_min=self._floor)
            if final:
                return
