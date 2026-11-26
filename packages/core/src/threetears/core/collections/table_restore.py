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
``last_epoch`` index (:func:`changed_scopes`). A scope epochs scope names a table's scope as ``scope_prefix + value``
(reports: ``state:VA`` for ``join_state_code = 'VA'``); a scope of another prefix (a race) names none of this table's.
The writer's contract this rests on: a write of a row of scope value ``v`` touches ``scope_prefix + v``.

**Each scope's versions, through the (scope, epoch) index.** One statement a scope a table (:func:`_scope_versions`):
the scope bound, ``epoch >= X``, one range of the index (hashed by the scope on YugabyteDB), read as ``COPY`` text and
folded into records in the restore's worker process (:class:`~threetears.core.change_note_fold.RestoreFolder`, the fold
the change-note restore uses), so the loop moves bytes only and no row becomes a Python object. A gone marker is a
version like any other: its key comes back, and a follower evicts it. A run marker
(:mod:`~threetears.core.ddl.run_markers`) comes back as the whole-table record of its prefix (:func:`run_marker_changes`),
read by its own key ``(table_name, scope)``. Every record names X as its ``s_min``: the lowest stamp any row it carries
may hold.

**Stamped tables, a bucket at a time.** A stamped table (``ROW_STAMPED``: one row a key, updated in place, its epoch
risen on every write) keeps no versions and no scope: its rows written since X are those whose epoch is at or above X,
the rows its deletes marked gone among them. Its ``(bucket, epoch)`` index carries a stamped declaration
(:func:`~threetears.core.ddl.row_epoch.stamped_restore_declaration`); the bucket is a stored column the database derives
from the key's hash, one of :data:`~threetears.core.ddl.row_epoch.RESTORE_BUCKETS`. A restore reads each bucket's
``epoch >= X`` as one range of that index (:func:`_bucket_rows`), every bucket whatever changed (no table says which
did): :data:`~threetears.core.ddl.row_epoch.RESTORE_BUCKETS` statements a table, folded in the same worker, every record
naming X as its ``s_min``. A delete of a stamped table's key marks its row gone in place, its epoch risen past the
delete's stamp (``SqlL3Backend.mark_gone``; a real ``DELETE`` of one is refused), so the bucket read finds it like any
other write (``copy_keyed(with_gone=True)``, where every other read skips it) and the key comes back: a follower
evicts it.

**Every declared invalidation column, as the notes carried them.** A record names a row's key and its declared
invalidation columns (``TableSchema.invalidation_columns``, a column tenancy's tenant column among them), so a follower
evicting by any of them hears the change. The declaration names them (``"columns"``,
:func:`~threetears.core.ddl.row_epoch.invalidation_columns_of`); each read selects every one.

**Every move since X comes back with the values it left.** A stamped table whose invalidation columns (its tenant
column among them) a write may change in place (``TableSchema.moveable``, :mod:`~threetears.core.ddl.tenant_moves`)
has each move appended, by the write that made it, to the moves table in its schema: one row a move, under the table
and the row's bucket, naming the key, the tenant it left and every invalidation column as it left them. Its
declaration names the moveable columns (``"moveable"``); beside each bucket's rows a restore reads the bucket's moves
with ``epoch >= X``, one range of the moves table's key (:func:`_bucket_moves`): the followers of each tenant and value
a move since X left evict the key, as the ring's old-row note had them do. A row moved twice since X (A to B, then B to
C) is two moves, so A's followers and B's both hear it.

**A table reset is heard whole.** A ``TRUNCATE`` leaves no row, so whatever resets a table that carries an epoch writes
its reset marker into the run-marker table first (:func:`~threetears.core.ddl.run_markers.table_reset_marker_sql`), at
the reset's epoch. Each table's reset markers since X are read by their key (:func:`table_resets`) and come back as the
whole table, a reset (:data:`~threetears.core.collections.epoch_change.RESET_REASON`): every follower clears it.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from itertools import groupby
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Final

from threetears.core.collections.noted_change import NotedChange
from threetears.core.collections.epoch_change import RESET_REASON, keyed_record_form
from threetears.core.collections.scope_epochs import changed_scopes
from threetears.core.ddl.row_epoch import EPOCH_COLUMN, RESTORE_DECLARATION_MARK, invalidation_columns_of
from threetears.core.ddl.run_markers import RUN_MARKERS_TABLE, TABLE_RESET_SCOPE
from threetears.core.ddl.tenant_moves import key_text_sql
from threetears.core.keyed import AnyOf, Fragment, Selected, Sort, copy_keyed, keyset_page, keyset_rows
from threetears.core.sql_fragments import quote_identifier
from threetears.core.tenancy import Tenancy, column_tenant_sql, schema_tenant, value_tenant
from threetears.observe import get_logger

if TYPE_CHECKING:
    # the folder is Arrow's (core.change_note_fold imports pyarrow and NumPy); the keeper builds it and hands it to
    # TableChanges, so this module, which epoch_reads (and through it the epoch follow) imports, stays free of Arrow
    # (tests/enforcement/test_core_imports_without_arrow.py)
    from threetears.core.change_note_fold import RestoreFolder

__all__ = [
    "RestoredTable",
    "StampedTable",
    "TableChanges",
    "VersionedTable",
    "changed_scopes",
    "key_term",
    "restored_tables",
    "run_marker_changes",
    "table_resets",
    "versioned_tables",
]

log = get_logger(__name__)

#: the bytes of a scope's ``COPY`` text handed to the worker at once: well under a MiB, so the loop's copy of a page
#: stays far inside a turn (measured for the change-note restore this read replaced, 2026-11-15, L1: 1 MiB a page)
_PAGE_BYTES: Final = 1 << 20

#: distinct rows of one table a change holds before it may be cut (as the change-note restore's)
_CHUNK_ROWS: Final = 5000

#: what a run marker's prefix joins its values with (:mod:`~threetears.core.ddl.run_markers`)
_PREFIX_SEPARATOR: Final = "\x1f"


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
    :ivar columns: its declared invalidation columns, the tenant column of a column tenancy among them
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
    columns: tuple[str, ...] = ()

    @property
    def relation(self) -> str:
        """the table as the keyed helpers name it, ``schema.table``.

        :return: it
        :rtype: str
        """
        return f"{self.schema}.{self.table}"


@dataclass(frozen=True, slots=True)
class StampedTable:
    """one stamped table as its restore declaration names it.

    :ivar schema: its schema
    :ivar table: its name
    :ivar bucket: the bucket column its (bucket, epoch) index leads with
    :ivar buckets: how many buckets: the values ``0`` to ``buckets - 1``
    :ivar keys: its primary key's columns, in order
    :ivar instants: those of them spelled as their UTC text
    :ivar tenancy: its tenancy
    :ivar moveable: the columns a write may move its rows between tenants by (``TableSchema.moveable``)
    :ivar columns: its declared invalidation columns, the tenant column of a column tenancy among them
    :ivar run_markers: whether its schema holds the run-marker table (its reset markers)
    """

    schema: str
    table: str
    bucket: str
    buckets: int
    keys: tuple[str, ...]
    instants: frozenset[str]
    tenancy: Tenancy
    moveable: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()
    run_markers: bool = False

    @property
    def moved_columns(self) -> tuple[str, ...]:
        """its moveable columns a record names: its invalidation columns a write may change in place, whose moves a
        restore reads back with the value a row left (none when no record names a moveable column: a key alone, on
        ``_all`` or its tenant, is what every follower of it hears).

        :return: the columns, in declared order
        :rtype: tuple[str, ...]
        """
        return tuple(column for column in self.columns if column in self.moveable)

    @property
    def relation(self) -> str:
        """the table as the keyed helpers name it, ``schema.table``.

        :return: it
        :rtype: str
        """
        return f"{self.schema}.{self.table}"


#: a table a restore reads: versioned, a scope at a time; or stamped, a bucket at a time
RestoredTable = VersionedTable | StampedTable


def _tenancy(value: dict[str, Any]) -> Tenancy:
    """the tenancy a restore declaration names; a table of no tenant (none, undeclared) is restored on ``_all``, as its
    change notes were.

    :param value: the declaration
    :ptype value: dict[str, Any]
    :return: the tenancy
    :rtype: Tenancy
    """
    kind = str(value["tenancy"])
    column = value.get("tenant_column")
    if kind == "column":
        return Tenancy("column", str(column))
    return Tenancy("schema") if kind == "schema" else Tenancy("undeclared")


def _declared(schema: str, table: str, text: str, run_markers: bool) -> RestoredTable:
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
    :rtype: RestoredTable
    :raises ValueError: when the comment is not a restore declaration
    """
    value = json.loads(text)
    if not isinstance(value, dict) or value.get(RESTORE_DECLARATION_MARK) != 1:
        raise ValueError(f"{schema}.{table}'s restore index names no restore declaration: {text!r}")
    tenancy = _tenancy(value)
    columns = invalidation_columns_of(value)
    if value.get("kind") == "stamped":
        return StampedTable(
            schema=schema,
            table=table,
            bucket=str(value["bucket"]),
            buckets=int(value["buckets"]),
            keys=tuple(str(key) for key in value["keys"]),
            instants=frozenset(str(key) for key in value["instants"]),
            tenancy=tenancy,
            moveable=tuple(str(column) for column in value.get("moveable", ())),
            columns=columns,
            run_markers=run_markers,
        )
    return VersionedTable(
        schema=schema,
        table=table,
        scope=str(value["scope"]),
        keys=tuple(str(key) for key in value["keys"]),
        instants=frozenset(str(key) for key in value["instants"]),
        epochs=str(value["epochs"]),
        scope_prefix=str(value["scope_prefix"]),
        tenancy=tenancy,
        run_markers=run_markers,
        columns=columns,
    )


async def restored_tables(pool: Any) -> list[RestoredTable]:
    """every versioned and stamped table the database holds, from the restore declarations of their restore indexes
    (a versioned table's (scope, epoch), a stamped table's (bucket, epoch)): one catalog statement.

    :param pool: the keeper's L3 pool
    :ptype pool: Any
    :return: the tables, in schema and name order
    :rtype: list[RestoredTable]
    """
    rows = await pool.fetch(_VERSIONED_TABLES_SQL, f'{{"{RESTORE_DECLARATION_MARK}":%')
    return [
        _declared(str(row["table_schema"]), str(row["table_name"]), str(row["declared"]), bool(row["run_markers"]))
        for row in rows
    ]


async def versioned_tables(pool: Any) -> list[VersionedTable]:
    """every versioned table the database holds (:func:`restored_tables`, the stamped ones left out).

    :param pool: the keeper's L3 pool
    :ptype pool: Any
    :return: the tables, in schema and name order
    :rtype: list[VersionedTable]
    """
    return [table for table in await restored_tables(pool) if isinstance(table, VersionedTable)]


def _literal(text: str) -> str:
    """``text`` as a SQL string literal.

    :param text: a TRUSTED name (a table's, a tenant token)
    :ptype text: str
    :return: the literal
    :rtype: str
    """
    return "'" + text.replace("'", "''") + "'"


def key_term(table: RestoredTable, column: str) -> str:
    """a key column of a row as a record spells it: its text, a ``timestamptz`` as its UTC instant.

    :param table: the table
    :ptype table: RestoredTable
    :param column: the column
    :ptype column: str
    :return: the expression
    :rtype: str
    """
    # the moves table's keys are spelled by the same function, so a move's key is the row's (tenant_moves)
    return key_text_sql(column, instant=column in table.instants)


def column_value_sql(table: RestoredTable, column: str) -> str:
    """an invalidation column of a row as a record names it, its text.

    :param table: the table
    :ptype table: RestoredTable
    :param column: the column
    :ptype column: str
    :return: the expression
    :rtype: str
    """
    return f"t.{quote_identifier(column)}::text"


def tenant_value_sql(table: RestoredTable) -> str:
    """a row's tenant token as a record names it (:mod:`threetears.core.tenancy`): its schema's, its tenant column's,
    or ``NULL`` (``_all``).

    :param table: the table
    :ptype table: RestoredTable
    :return: the expression
    :rtype: str
    """
    tenancy = table.tenancy
    if tenancy.kind == "schema":
        return f"{_literal(schema_tenant(table.schema))}::text"
    if tenancy.kind != "column":
        return "NULL::text"
    return column_tenant_sql(str(tenancy.column))


def moved_tenant_sql(table: StampedTable) -> str:
    """a moves row's tenant token as a record names it: the tenant the move left (its ``tenant``) for a column tenancy,
    else the table's own (:func:`tenant_value_sql`).

    :param table: the moved table
    :ptype table: StampedTable
    :return: the expression, over the moves row ``t``
    :rtype: str
    """
    if table.tenancy.kind == "column":
        return 't."tenant"'
    return tenant_value_sql(table)


def _fold_columns(table: RestoredTable) -> list[str | Selected]:
    """what a scope's (or a bucket's) ``COPY`` selects: the four JSON texts the restore's fold reads, as a change note's
    restore ``COPY`` writes them (the table, the tenant, the key's values, every declared invalidation column).

    :param table: the table
    :ptype table: RestoredTable
    :return: the columns
    :rtype: list[str | Selected]
    """
    keys = ", ".join(key_term(table, column) for column in table.keys)
    # every declared invalidation column (a column tenancy's tenant column among them), as its notes carried them
    pairs = ", ".join(f"{_literal(column)}, {column_value_sql(table, column)}" for column in table.columns)
    columns = f"json_build_object({pairs})::text" if table.columns else "NULL::text"
    expressions = (
        f"to_json({_literal(table.table)}::text)::text",
        f"to_json({tenant_value_sql(table)})::text",
        f"array_to_json(ARRAY[{keys}])::text",
        columns,
    )
    return [Selected(expression, alias) for expression, alias in zip(expressions, _FOLD_COLUMNS, strict=True)]


def _move_fold_columns(table: StampedTable) -> list[str | Selected]:
    """what a bucket's moves ``COPY`` selects (:func:`_bucket_moves`): the fold's four JSON texts from a moves row,
    which holds the key and the invalidation columns as JSON already (:mod:`~threetears.core.ddl.tenant_moves`).

    :param table: the moved table
    :ptype table: StampedTable
    :return: the columns
    :rtype: list[str | Selected]
    """
    expressions = (
        f"to_json({_literal(table.table)}::text)::text",
        f"to_json({moved_tenant_sql(table)})::text",
        't."keys"',
        't."columns"',
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
        # every version and gone marker is a change the restore hears: read as stored, never the newest relation
        with_gone=True,
    )


async def _bucket_rows(pool: Any, table: StampedTable, bucket: int, floor: int) -> bytes:
    """every row of the stamped ``table`` in ``bucket`` stamped at ``floor`` or later, as the fold's ``COPY`` text: the
    bucket bound, one range of the ``(bucket, epoch)`` index (its own tablet on YugabyteDB).

    :param pool: the keeper's L3 pool
    :ptype pool: Any
    :param table: the table
    :ptype table: StampedTable
    :param bucket: the bucket, ``0`` to ``table.buckets - 1``
    :ptype bucket: int
    :param floor: X
    :ptype floor: int
    :return: the text, a line per row
    :rtype: bytes
    """
    return await copy_keyed(
        pool,
        table.relation,
        columns=_fold_columns(table),
        by={table.bucket: bucket},
        # the index's own order after its bound bucket: one range, no sort
        order=[Sort(EPOCH_COLUMN, cast="bigint")],
        where=[Fragment(f't."{EPOCH_COLUMN}" >= $1', (floor,))],
        force_quote=_FOLD_COLUMNS[:3],
        # a gone row is a delete the restore hears: read with the live ones
        with_gone=True,
    )


def moves_relation(table: StampedTable) -> str:
    """the moves table of ``table``'s schema, as the keyed helpers name it.

    :param table: the moved table
    :ptype table: StampedTable
    :return: ``schema.threetears_tenant_moves``
    :rtype: str
    """
    # spelled here, not by TENANT_MOVES_TABLE: the no-whole-reads walk reads the relation this answers from its own
    # text, and judges each read of it by the moves table's declared key (test_run_markers_schema holds them equal)
    return f"{table.schema}.threetears_tenant_moves"


async def _bucket_moves(pool: Any, table: StampedTable, bucket: int, floor: int) -> bytes:
    """every move of a row of the stamped ``table`` in ``bucket`` stamped at ``floor`` or later, as the fold's ``COPY``
    text naming the tenant and values the row LEFT: one range of the moves table's key ``(table_name, scope, epoch)``.
    A row moved twice since X is two lines, one for each tenant it left.

    :param pool: the keeper's L3 pool
    :ptype pool: Any
    :param table: the table
    :ptype table: StampedTable
    :param bucket: the bucket, ``0`` to ``table.buckets - 1``
    :ptype bucket: int
    :param floor: X
    :ptype floor: int
    :return: the text, a line per move
    :rtype: bytes
    """
    return await copy_keyed(
        pool,
        moves_relation(table),
        columns=_move_fold_columns(table),
        # the moves table's key prefix, the table and the bucket, then its own order (its own columns, as
        # TENANT_MOVES_SCHEMA spells them): one range, no sort
        by={"table_name": table.table, "scope": str(bucket)},
        order=[Sort("epoch", cast="bigint")],
        where=[Fragment('t."epoch" >= $1', (floor,))],
        force_quote=_FOLD_COLUMNS[:3],
    )


async def table_resets(pool: Any, table: RestoredTable, floor: int, ceiling: int | None = None) -> list[NotedChange]:
    """``table`` reset (emptied by a ``TRUNCATE``) at ``floor`` or later (to ``ceiling``, when named): its reset markers
    (:func:`~threetears.core.ddl.run_markers.table_reset_marker_sql`), read by the run-marker table's key
    ``(table_name, scope)``, as ONE whole-table change, a reset, under the table's tenant (``_all`` for a column
    tenancy, whose every tenant it clears).

    :param pool: the keeper's L3 pool
    :ptype pool: Any
    :param table: the table
    :ptype table: RestoredTable
    :param floor: X, named as the change's ``s_min``
    :ptype floor: int
    :param ceiling: the highest epoch read; ``None`` for every one from ``floor`` on (a restore's)
    :ptype ceiling: int | None
    :return: the change, or none
    :rtype: list[NotedChange]
    """
    if not table.run_markers:
        return []
    bounds = [Fragment('t."epoch" >= $1', (floor,))]
    if ceiling is not None:
        bounds.append(Fragment('t."epoch" <= $1', (ceiling,)))
    # one reset in the range clears the table whole, so one row answers: the first of the key's range
    found = await keyset_page(
        pool,
        f"{table.schema}.{RUN_MARKERS_TABLE}",
        columns=("prefix", "epoch"),
        by={"table_name": table.table, "scope": TABLE_RESET_SCOPE},
        order=[Sort("prefix"), Sort("epoch", cast="bigint")],
        limit=1,
        where=bounds,
    )
    if not found.rows:
        return []
    tenant = schema_tenant(table.schema) if table.tenancy.kind == "schema" else None
    return [NotedChange(table=table.table, tenant=tenant, whole_table=True, reason=RESET_REASON, s_min=floor)]


async def run_marker_changes(
    pool: Any, table: VersionedTable, values: Sequence[str], floor: int, ceiling: int | None = None
) -> list[NotedChange]:
    """the runs of ``table`` deleted at ``floor`` or later (to ``ceiling``, when named) in the scopes ``values``, as
    whole-table changes of their prefixes, one per tenant: read by the run-marker table's key ``(table_name, scope)``.

    :param pool: the keeper's L3 pool
    :ptype pool: Any
    :param table: the table
    :ptype table: VersionedTable
    :param values: the scopes' values
    :ptype values: Sequence[str]
    :param floor: X, named as the changes' ``s_min``
    :ptype floor: int
    :param ceiling: the highest epoch read; ``None`` for every one from ``floor`` on (a restore's)
    :ptype ceiling: int | None
    :return: the changes
    :rtype: list[NotedChange]
    """
    if not table.run_markers or not values:
        return []
    bounds = [Fragment('t."epoch" >= $1', (floor,))]
    if ceiling is not None:
        bounds.append(Fragment('t."epoch" <= $1', (ceiling,)))
    rows = await keyset_rows(
        pool,
        f"{table.schema}.{RUN_MARKERS_TABLE}",
        columns=("scope", "prefix", "epoch"),
        by={"table_name": table.table, "scope": AnyOf(list(values))},
        order=[Sort("scope"), Sort("prefix"), Sort("epoch", cast="bigint")],
        where=bounds,
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
    """the restored tables' changes since X, as a stream of changes a restore appends: each table's resets; each
    versioned table's run markers, then its scopes' versions a scope at a time; each stamped table's rows a bucket at a
    time; all folded in the restore's worker.

    :ivar tables: the tables a change was streamed of
    :ivar scopes: the scope reads made (one statement each)
    :ivar buckets: the bucket reads made (one statement each)
    """

    def __init__(
        self,
        pool: Any,
        tables: Sequence[RestoredTable],
        floor: int,
        folder: RestoreFolder,
        database: str | None,
    ) -> None:
        """capture what the stream reads; no I/O.

        :param pool: the keeper's L3 pool
        :ptype pool: Any
        :param tables: the versioned and stamped tables
        :ptype tables: Sequence[RestoredTable]
        :param floor: X: every version (or stamped row) stamped at or after it is read
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
        self.buckets = 0

    def __aiter__(self) -> AsyncIterator[NotedChange]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[NotedChange]:
        # the scopes each epochs table names changed, read once a table even when several tables share it
        changed: dict[tuple[str, str], list[str]] = {}
        for table in self._tables:
            if isinstance(table, StampedTable):
                async for change in self._stamped(table):
                    yield change
                continue
            streamed = False
            for change in await table_resets(self._pool, table, self._floor):
                streamed = True
                yield change
            named = (table.schema, table.epochs)
            if named not in changed:
                changed[named] = await changed_scopes(self._pool, table.schema, table.epochs, self._floor)
            prefix = table.scope_prefix
            values = [scope[len(prefix) :] for scope in changed[named] if scope.startswith(prefix)]
            for change in await run_marker_changes(self._pool, table, values, self._floor):
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

    async def _stamped(self, table: StampedTable) -> AsyncIterator[NotedChange]:
        """a stamped table's rows stamped since X, a bucket at a time: every bucket, one statement each.

        :param table: the table
        :ptype table: StampedTable
        :return: the changes
        :rtype: AsyncIterator[NotedChange]
        """
        streamed = False
        for change in await table_resets(self._pool, table, self._floor):
            streamed = True
            yield change
        for bucket in range(table.buckets):
            self.buckets += 1
            text = await _bucket_rows(self._pool, table, bucket, self._floor)
            async for change in self._folded(text):
                streamed = True
                yield change
            if not table.moved_columns:
                continue
            # every move since X, with the values (and the tenant) each left: their followers evict the key
            self.buckets += 1
            text = await _bucket_moves(self._pool, table, bucket, self._floor)
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
