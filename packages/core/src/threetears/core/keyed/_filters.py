"""the one spelling of a keyed read's filters, shared by every helper in :mod:`threetears.core.keyed`.

The filters, the names and the vocabulary they are written in are pure, and live in the dependency-free
:mod:`threetears.interfaces.keyed.filters` (re-exported here under the names every helper imports); what stays here
is the one question a filter asks of the connection it is sent on: whether it binds typed arrays natively.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

import asyncpg
from asyncpg.pool import PoolConnectionProxy
from threetears.core.backends.arrow_reply import l3_fetch_arrow
from threetears.core.backends.broker_generation import cleanup_delete_floor
from threetears.core.backends.dialect import known_on_yugabyte
from threetears.core.backends.read_bounds import answers_directly, bounded_call, bounded_read
from threetears.core.ddl.row_epoch import AppendOnlyWriteError
from threetears.core.keyed.inline import InlineDuckDB
from threetears.core.keyed.threaded import InlineSQLite, ThreadedDuckDB, ThreadedSQLite
from threetears.interfaces.keyed.counts import SQL_DIALECT_ATTRIBUTE, Dialect
from threetears.interfaces.keyed.keys import (
    VersionedTable,
    index_hint,
    key_pins_within,
    live_guard,
    live_relation,
    newest_relation,
    newest_settings,
    not_null_guards,
    serving_index,
    stamped_table,
    versioned_table,
)
from threetears.interfaces.keyed.page import Sort
from threetears.interfaces.keyed.filters import (
    IN_SUBQUERY,
    AnyOf,
    Const,
    Derived,
    Fragment,
    Lock,
    Pin,
    Selected,
    batched_join_hint,
    by_conditions,
    SQL_TOKEN,
    clause_guards,
    each_placeholder,
    narrowing,
    one_use_each,
    prefix_conditions,
    quoted_table,
    relation,
    renumbered,
    selected_column,
)

__all__ = [
    "IN_SUBQUERY",
    "SQL_TOKEN",
    "AnyOf",
    "Const",
    "Derived",
    "Fragment",
    "Lock",
    "Pin",
    "Selected",
    "batched_join_hint",
    "binds_native_arrays",
    "by_conditions",
    "clause_guards",
    "each_placeholder",
    "prefix_conditions",
    "quoted_table",
    "reads_l3",
    "reads_yugabyte",
    "selected_column",
    "relation",
    "live_only",
    "live_read",
    "narrowing",
    "newest_of",
    "newest_read",
    "newest_sent",
    "bounded_sent",
    "versions_read",
    "refuse_versioned",
    "refuse_append_only",
    "stamped_names",
    "one_use_each",
    "refuse_in_place",
    "renumbered",
    "sent_by",
    "yugabyte_index_hint",
]


def binds_native_arrays(conn: Any) -> bool:
    """whether ``conn`` binds a typed array natively (``$1::bigint[]`` from a list of ints): asyncpg's own
    connection, pool or pooled connection, or anything saying so (``binds_native_arrays = True``: a wrapper of one).
    The L3 rail carries its parameters as JSON, so a keyed read through it binds text arrays cast to the column's
    type (``$1::text[]::bigint[]``); identity, not truthiness, so a mock's auto-attribute never reads as yes.

    :param conn: the connection a keyed helper reads through
    :ptype conn: Any
    :return: whether its arrays bind natively
    :rtype: bool
    """
    if isinstance(conn, asyncpg.Connection | asyncpg.Pool | PoolConnectionProxy):
        return True
    return getattr(conn, "binds_native_arrays", None) is True


#: the dialects whose drivers take a placeholder named more than once (a reader naming one is sent as written)
_TAKES_REUSE: Final[frozenset[Dialect]] = frozenset({Dialect.POSTGRES, Dialect.YUGABYTE, Dialect.DUCKDB})


class _OneUseEach:
    """a reader naming its engine's dialect, sent every statement with one placeholder a use (:func:`sent_by`)."""

    def __init__(self, reader: Any) -> None:
        self._reader = reader

    def __getattr__(self, name: str) -> Any:
        return getattr(self._reader, name)

    async def fetch(self, sql: str, *params: Any, **options: Any) -> Any:
        """the reader's ``fetch`` of ``sql`` sent one placeholder a use.

        :param sql: the statement
        :ptype sql: str
        :param params: its parameters
        :ptype params: Any
        :param options: the reader's own options, passed on
        :ptype options: Any
        :return: what the reader answers
        :rtype: Any
        """
        statement, sent = one_use_each(sql, params)
        return await self._reader.fetch(statement, *sent, **options)

    async def fetchrow(self, sql: str, *params: Any, **options: Any) -> Any:
        """the reader's ``fetchrow`` of ``sql`` sent one placeholder a use.

        :param sql: the statement
        :ptype sql: str
        :param params: its parameters
        :ptype params: Any
        :param options: the reader's own options, passed on
        :ptype options: Any
        :return: what the reader answers
        :rtype: Any
        """
        statement, sent = one_use_each(sql, params)
        return await self._reader.fetchrow(statement, *sent, **options)

    async def fetchval(self, sql: str, *params: Any, **options: Any) -> Any:
        """the reader's ``fetchval`` of ``sql`` sent one placeholder a use.

        :param sql: the statement
        :ptype sql: str
        :param params: its parameters
        :ptype params: Any
        :param options: the reader's own options, passed on
        :ptype options: Any
        :return: what the reader answers
        :rtype: Any
        """
        statement, sent = one_use_each(sql, params)
        return await self._reader.fetchval(statement, *sent, **options)

    async def fetch_arrow(self, sql: str, *params: Any, **options: Any) -> Any:
        """the reader's ``fetch_arrow`` of ``sql`` sent one placeholder a use.

        :param sql: the statement
        :ptype sql: str
        :param params: its parameters
        :ptype params: Any
        :param options: the reader's own options, passed on
        :ptype options: Any
        :return: what the reader answers
        :rtype: Any
        """
        statement, sent = one_use_each(sql, params)
        return await self._reader.fetch_arrow(statement, *sent, **options)


def sent_by(conn: Any) -> Any:
    """what a keyed read sends its statement through: ``conn`` itself, or, for a reader naming its engine's dialect
    (a warehouse's datasource reader, any but PostgreSQL, YugabyteDB and DuckDB: Redshift binds each placeholder once, and refuses a statement naming ``$1`` twice where
    PostgreSQL takes it), ``conn`` sending each statement :func:`one_use_each`, whatever builder wrote it.

    :param conn: the connection or reader the read was handed
    :ptype conn: Any
    :return: what to send the statement through
    :rtype: Any
    """
    named = getattr(conn, SQL_DIALECT_ATTRIBUTE, None)
    return _OneUseEach(conn) if isinstance(named, Dialect) and named not in _TAKES_REUSE else conn


def reads_l3(conn: Any) -> bool:
    """whether a read on ``conn`` goes to L3, where the declared tables (their keys, their indexes) are: an in-process
    DuckDB or SQLite does not, nor does a reader naming its dialect (a datasource's, a warehouse's, even one spelled as
    PostgreSQL); an asyncpg connection, a pool, the rail's proxy do.

    :param conn: the connection or reader the read was handed
    :ptype conn: Any
    :return: whether it reads L3
    :rtype: bool
    """
    if isinstance(conn, ThreadedDuckDB | ThreadedSQLite | InlineSQLite | InlineDuckDB):
        return False
    # looked up on the class (a reader's property, its class attribute) and the instance's own attributes: a mock
    # answers any name it is asked, and is no datasource
    named = getattr(type(conn), SQL_DIALECT_ATTRIBUTE, None)
    return named is None and getattr(conn, "__dict__", {}).get(SQL_DIALECT_ATTRIBUTE) is None


def newest_of(conn: Any, table: Any) -> VersionedTable | None:
    """the versioned declaration of the L3 table a read on ``conn`` names: None for a read that does not go to L3 (a
    local copy keeps one row a key), a derived relation, or a table no versioned declaration names.

    :param conn: the connection or reader the read was handed
    :ptype conn: Any
    :param table: the relation read
    :ptype table: Any
    :return: the declaration, or None
    :rtype: VersionedTable | None
    """
    return versioned_table(table, l3=reads_l3(conn))


def refuse_versioned(conn: Any, tables: Sequence[Any], read: str) -> None:
    """refuse a read on L3 that names a versioned table in a shape no newest relation reads (a recursive walk, a join
    inside one statement): it would read every version of the table. A local copy (one row a key) is read as held.

    :param conn: the connection or reader the read was handed
    :ptype conn: Any
    :param tables: the tables the read names
    :ptype tables: Sequence[Any]
    :param read: what the read is, for the refusal
    :ptype read: str
    :return: nothing
    :rtype: None
    :raises ValueError: when one of them is a versioned L3 table
    """
    named = [table for table in tables if newest_of(conn, table) is not None]
    if named:
        raise ValueError(
            f"{read} names the versioned table(s) {', '.join(map(str, named))}, every version of which it would read: "
            "read them by keyed reads (read_keyed, read_by), which see each key's newest live version"
        )


def refuse_append_only(conn: Any, tables: Sequence[Any], read: str) -> None:
    """refuse a read on L3 that names an append-only table in a shape no live read reaches (a recursive walk, a join
    inside one statement): a versioned table's every version (:func:`refuse_versioned`), or a stamped table's gone
    rows, which its statement names no live guard for. Such a read is made by keyed reads (``read_by``,
    ``read_keyed``, a walk a level a read), each through :func:`live_read`. A local copy is read as held.

    WHY refused, not guarded in place: the statement's text is written once a call shape (a walk's,
    ``joins._HELD_STATEMENTS``) or its joined tables' rows are read beside the ranked ones before the limit cuts
    (``_pattern_count_statement``), and each spot a gone row could enter is one more place to keep a guard; the keyed
    reads beside it already answer live rows alone.

    :param conn: the connection or reader the read was handed
    :ptype conn: Any
    :param tables: the tables the read names
    :ptype tables: Sequence[Any]
    :param read: what the read is, for the refusal
    :ptype read: str
    :return: nothing
    :rtype: None
    :raises ValueError: when one of them is a versioned or a stamped L3 table
    """
    refuse_versioned(conn, tables, read)
    named = [table for table in tables if stamped_table(table, l3=reads_l3(conn)) is not None]
    if named:
        raise ValueError(
            f"{read} names the stamped table(s) {', '.join(map(str, named))}, whose gone rows it would read: read them "
            "by keyed reads (read_keyed, read_by), which see live rows alone"
        )


def stamped_names(conn: Any, tables: Sequence[Any]) -> frozenset[str]:
    """the tables among ``tables`` a read on ``conn`` reads as stamped L3 tables (``ROW_STAMPED``), which a statement
    naming them inside one statement (a join, a recursive walk) guards in place by their live guard
    (``NOT <alias>."gone"``), so it answers no row a delete marked gone. Empty for a read that does not go to L3 (a
    local copy holds only live rows).

    :param conn: the connection or reader the read was handed
    :ptype conn: Any
    :param tables: the tables the statement names
    :ptype tables: Sequence[Any]
    :return: the stamped ones, by name
    :rtype: frozenset[str]
    """
    l3 = reads_l3(conn)
    return frozenset(str(table) for table in tables if stamped_table(table, l3=l3) is not None)


def versions_read(
    conn: Any, table: str | Derived, *, lock: Any = None, by: Mapping[str, Any] | None = None
) -> str | Derived:
    """what a read of ``table`` on ``conn`` reads in its place when it guards a stamped table itself
    (:func:`live_read`, the keyed counts naming each relation's guard): a versioned L3 table's newest live versions, a
    :class:`Derived` relation answering one row a key (:func:`~threetears.interfaces.keyed.keys.newest_relation`), so
    the read sees only each key's newest version, never an older one, a key whose newest version is a "gone" marker,
    or a key under a newer run marker. ``table`` itself for a read that does not go to L3, a derived relation, or a
    table no versioned declaration names (a stamped one among them: its guard is :func:`live_only`).

    For a read whose conditions on the key are plain ones (pins, ranges: the planner pushes them into the relation's
    ``DISTINCT ON``); a keyed batch, a join and a fill page name their keys or ``LIMIT`` inside it themselves
    (``read_keyed``, ``keyset_pages``), as no planner pushes those.

    :param conn: the connection or reader the read was handed
    :ptype conn: Any
    :param table: the relation read
    :ptype table: str | Derived
    :param lock: the row lock the read takes; a versioned table's versions are never locked
    :ptype lock: Lock | None
    :param by: the read's pins: those on the key's columns alone are named inside the relation's ``DISTINCT ON``
        (:func:`~threetears.interfaces.keyed.keys.key_pins_within`, bound as the derived relation's own parameters),
        so it reads only the pinned scope's versions, never left to the planner to push there; none when None
    :ptype by: Mapping[str, Any] | None
    :return: the relation to read
    :rtype: str | Derived
    :raises ValueError: when a read of a versioned table takes a row lock (``DISTINCT ON`` takes none, and an
        append-only table's versions are never changed in place, so there is nothing to lock against)
    """
    versioned = newest_of(conn, table)
    if versioned is None or not isinstance(table, str):
        return table
    if lock is not None:
        raise ValueError(f"{table} is versioned (append-only): its newest versions are read, never locked")
    within, values = key_pins_within(versioned, by or {})
    return Derived(newest_relation(versioned, table, within=within), tuple(values))


def newest_read(
    conn: Any, table: str | Derived, *, lock: Any = None, by: Mapping[str, Any] | None = None
) -> str | Derived:
    """what a read of ``table`` on ``conn`` reads in its place, so it sees only live rows: a versioned L3 table's
    newest live versions (:func:`versions_read`), a stamped one's live rows (a :class:`Derived` relation of its
    declared columns, :func:`~threetears.interfaces.keyed.keys.live_relation`). ``table`` itself for a read that does
    not go to L3, a derived relation, or a table neither declaration names. What a generic step, or any read naming
    no guard of its own, reads an append-only table through.

    :param conn: the connection or reader the read was handed
    :ptype conn: Any
    :param table: the relation read
    :ptype table: str | Derived
    :param lock: the row lock the read takes; an append-only table's relation is never locked
    :ptype lock: Lock | None
    :param by: the read's pins, named inside a versioned table's relation (:func:`versions_read`)
    :ptype by: Mapping[str, Any] | None
    :return: the relation to read
    :rtype: str | Derived
    :raises ValueError: when a read of a versioned or stamped table's relation takes a row lock
    """
    stamped = stamped_table(table, l3=reads_l3(conn)) if isinstance(table, str) else None
    if stamped is None:
        return versions_read(conn, table, lock=lock, by=by)
    if lock is not None:
        raise ValueError(f"{table} is stamped: its live rows are read through a relation, never locked")
    return Derived(live_relation(stamped, str(table)))


class _NewestSettings:
    """a direct connection or pool sending each newest read in a transaction of its own that first sets the read's
    bounds (:func:`~threetears.interfaces.keyed.keys.newest_settings`, ``SET LOCAL``:
    :func:`~threetears.core.backends.read_bounds.bounded_call`): they end with the read."""

    def __init__(self, conn: Any, settings: Sequence[tuple[str, int]]) -> None:
        self._conn = conn
        self._settings = tuple(settings)

    async def fetch(self, query: str, *params: Any) -> list[Any]:
        """the statement's rows, read under the bounds.

        :param query: the statement
        :ptype query: str
        :param params: its binds
        :ptype params: Any
        :return: the rows
        :rtype: list[Any]
        """
        return list(await bounded_call(self._conn, self._settings, lambda conn: conn.fetch(query, *params)))


class _NewestBounded:
    """any other L3 reader (the store, the rail's proxy, one of its transactions) sending each newest read inside
    :func:`~threetears.core.backends.read_bounds.bounded_read`: the reader that talks to the database reads the bounds
    where it sends, the store setting them on its own direct connection, the rail naming them on the request
    (``read_settings``) for the hub's broker to set, as it sets the read's ``statement_timeout``."""

    def __init__(self, conn: Any, settings: Sequence[tuple[str, int]]) -> None:
        self._conn = conn
        self._settings = tuple(settings)

    async def fetch(self, query: str, *params: Any) -> list[Any]:
        """the statement's rows, read under the bounds.

        :param query: the statement
        :ptype query: str
        :param params: its binds
        :ptype params: Any
        :return: the rows
        :rtype: list[Any]
        """
        with bounded_read(self._settings):
            return list(await self._conn.fetch(query, *params))

    async def fetch_arrow(self, query: str, *params: Any, types: Mapping[str, Any]) -> Any:
        """the statement's rows as one Arrow table, read under the bounds: the reader's own Arrow read (the rail's
        proxy parses its reply off the interpreter lock), else its dict rows made the table on a thread.

        :param query: the statement
        :ptype query: str
        :param params: its binds
        :ptype params: Any
        :param types: column name -> its Arrow type, every column selected, in order
        :ptype types: Mapping[str, pyarrow.DataType]
        :return: the rows
        :rtype: pyarrow.Table
        """
        with bounded_read(self._settings):
            return await l3_fetch_arrow(self._conn, query, *params, types=types)


def newest_sent(conn: Any, newest: VersionedTable | None, yugabyte: bool, *, batch: bool = True) -> Any:
    """what a newest read of a versioned table sends its statement through: each statement under the read's bounds
    (``work_mem``, and YugabyteDB's ``yb_fetch_row_limit``: :func:`~threetears.interfaces.keyed.keys.newest_settings`),
    so its sort stays in memory and its storage requests few: on a direct connection set on it
    (:class:`_NewestSettings`), through any other L3 reader named to it (:class:`_NewestBounded`: the store's direct
    connection sets them, the L3 rail names them on the request for the hub's broker to set); a read of every row named (``newest`` None) through :func:`sent_by`.

    :param conn: the connection or reader the read was handed
    :ptype conn: Any
    :param newest: the table's versioned declaration; None for a read of every row named
    :ptype newest: VersionedTable | None
    :param yugabyte: whether the read runs on YugabyteDB
    :ptype yugabyte: bool
    :param batch: whether the read is a keyed batch (else a fill page, which sets ``work_mem`` alone)
    :ptype batch: bool
    :return: what to send the statement through
    :rtype: Any
    """
    if newest is None:
        return sent_by(conn)
    settings = newest_settings(yugabyte=yugabyte, batch=batch)
    return _NewestSettings(conn, settings) if answers_directly(conn) else _NewestBounded(conn, settings)


def bounded_sent(conn: Any, *tables: Any) -> Any:
    """what a shared keyed read that names its relations itself (a count, an aggregate, a point read, a marks read, a
    changed read) sends its statement through: under a newest read's bounds (``work_mem``,
    :func:`~threetears.core.backends.read_bounds.bounded_read`) when any of ``tables`` is a versioned L3 table, whose
    newest relation sorts the versions of the keys it reads; :func:`sent_by` otherwise, as before.

    WHY a fill's bounds (``batch=False``): such a read is no keyed batch of whole keys, so YugabyteDB's fetch limit is
    not asked (:func:`~threetears.interfaces.keyed.keys.newest_settings`); its relation's sort is the work ``work_mem``
    is sized for. The one place each of those readers asks, so none is left unbounded on another path.

    :param conn: the connection or reader the read was handed
    :ptype conn: Any
    :param tables: the relations the statement reads
    :ptype tables: Any
    :return: what to send the statement through
    :rtype: Any
    """
    for table in tables:
        newest = newest_of(conn, table)
        if newest is not None:
            return newest_sent(conn, newest, False, batch=False)
    return sent_by(conn)


def live_only(conn: Any, table: Any, where: Sequence[Fragment] = (), *, row: str | None = "t") -> tuple[Fragment, ...]:
    """a keyed read's filters, with a stamped L3 table's live guard beside them: the read then answers only live
    rows, never one a delete marked gone (:func:`~threetears.interfaces.keyed.keys.live_guard`). Unchanged for a read
    that does not go to L3 (a local copy holds only live rows), a derived relation, or a table no stamped declaration
    names.

    The stamped half of :func:`live_read`, the one place a shared keyed read asks both: a versioned table is read
    through its newest relation (:func:`versions_read`), a stamped one by this guard, so no read answers a gone row.

    :param conn: the connection or reader the read was handed
    :ptype conn: Any
    :param table: the relation read
    :ptype table: Any
    :param where: the read's own filters
    :ptype where: Sequence[Fragment]
    :param row: how the statement names the row read: its alias, the quoted relation where it has none, or None for
        a statement naming its columns unqualified
    :ptype row: str | None
    :return: the filters to read by
    :rtype: tuple[Fragment, ...]
    """
    declared = stamped_table(table, l3=reads_l3(conn))
    if declared is None:
        return tuple(where)
    return (*where, Fragment(live_guard(declared, row)))


def live_read(
    conn: Any,
    table: str | Derived,
    where: Sequence[Fragment] = (),
    *,
    lock: Lock | None = None,
    row: str | None = "t",
    by: Mapping[str, Any] | None = None,
) -> tuple[str | Derived, tuple[Fragment, ...]]:
    """what a shared keyed read of ``table`` on ``conn`` reads, and the filters it reads by, so it answers only live
    rows: a versioned L3 table's newest live versions in its place (:func:`versions_read`: one row a key, never an older
    version, a "gone" marker or a key under a newer run marker), a stamped one by its live guard beside ``where``
    (:func:`live_only`: never a row its delete marked gone). ``table`` and ``where`` as given for a read that does not
    go to L3 (a local copy holds only live rows), a derived relation, or a table neither declaration names.

    The one mechanism every shared keyed read asks (append-only L3: a delete is a marker everywhere, so a restore of a
    lost change log hears it, and no read sees one). A table is versioned or stamped, never both.

    :param conn: the connection or reader the read was handed
    :ptype conn: Any
    :param table: the relation read
    :ptype table: str | Derived
    :param where: the read's own filters
    :ptype where: Sequence[Fragment]
    :param lock: the row lock the read takes; a versioned table's versions are never locked
    :ptype lock: Lock | None
    :param row: how the statement names the row read (:func:`live_only`)
    :ptype row: str | None
    :param by: the read's pins: those on a versioned table's key are named inside its newest relation
        (:func:`versions_read`), so it reads only the pinned scope's versions
    :ptype by: Mapping[str, Any] | None
    :return: the relation to read, and the filters to read it by
    :rtype: tuple[str | Derived, tuple[Fragment, ...]]
    :raises ValueError: when a read of a versioned table takes a row lock (:func:`versions_read`)
    """
    return versions_read(conn, table, lock=lock, by=by), live_only(conn, table, where, row=row)


def refuse_in_place(conn: Any, table: Any, write: str) -> None:
    """refuse a keyed in-place write of an append-only L3 table, so a restore of a lost change log loses nothing
    (append-only L3, decision 2: "a real delete leaves nothing for a restore to find"):

    - a versioned table (``row_versions``): its rows are versions; a write appends the key's new version and a delete
      a "gone" marker (``SqlL3Backend``, the collection's save and delete), never an ``UPDATE`` or a ``DELETE`` of a
      version a read may be seeing (``threetears.interfaces.keyed.keys.newest_relation``);
    - a stamped table (``ROW_STAMPED``): a delete of one of its keys marks its row gone in place
      (``SqlL3Backend.mark_gone``, the collection's ``delete_rows``); an update in place is its own write. Only
      quiet-time cleanup deletes from it, below the restore window, through its own statements
      (:func:`~threetears.core.collections.stamped_cleanup.clear_stamped_gone`).

    A local copy's writes are its own, and quiet-time cleanup's delete (one made inside
    :func:`~threetears.core.backends.broker_generation.cleanup_deleting`, below its floor F) is the one delete either
    kind takes: it removes only what no restore reads (``QuietCleanup``, ``clear_stamped_gone``).

    :param conn: the connection the write would run on; None for a collection's own L3 pool
    :ptype conn: Any
    :param table: the table written
    :ptype table: Any
    :param write: which write (``update`` / ``delete``), for the refusal
    :ptype write: str
    :return: nothing
    :rtype: None
    :raises AppendOnlyWriteError: when ``table`` is a versioned L3 table, or a stamped one a delete names
    """
    if write == "delete" and cleanup_delete_floor() is not None:
        # quiet-time cleanup's own delete, marked with its floor (the hub's broker bounds it by its own reading)
        return
    l3 = conn is None or reads_l3(conn)
    if versioned_table(table, l3=l3) is not None:
        raise AppendOnlyWriteError(
            f"{table} is versioned (append-only): a keyed {write} would change a version in place; save the key's new "
            "version, or delete it as a gone marker, through its collection or the L3 backend"
        )
    if write == "delete" and stamped_table(table, l3=l3) is not None:
        raise AppendOnlyWriteError(
            f"{table} is stamped (ROW_STAMPED): a keyed delete would leave nothing for a restore to find; delete its "
            "keys as gone marks through its collection (delete_rows) or the L3 backend (mark_gone)"
        )


async def reads_yugabyte(conn: Any) -> bool:
    """whether a read on ``conn`` runs on YugabyteDB, whose keyed batches are spelled for it (no bounds beside a
    batch's keys, :meth:`~threetears.interfaces.keyed.rows._KeyedRead.render`): an in-process DuckDB or SQLite never
    is, a reader naming its dialect is when it names YugabyteDB's, and an L3 says so as it says it for an index hint
    (:func:`~threetears.core.backends.dialect.known_on_yugabyte`: no round trip on an asyncpg connection, one a pool
    or proxy, kept).

    :param conn: the connection or reader the read was handed
    :ptype conn: Any
    :return: whether it runs on YugabyteDB
    :rtype: bool
    """
    if isinstance(conn, ThreadedDuckDB | ThreadedSQLite | InlineSQLite | InlineDuckDB):
        return False
    named = getattr(type(conn), SQL_DIALECT_ATTRIBUTE, None)
    if named is None:
        named = getattr(conn, "__dict__", {}).get(SQL_DIALECT_ATTRIBUTE)
    if named is not None:
        return named is Dialect.YUGABYTE
    return await known_on_yugabyte(conn)


async def yugabyte_index_hint(
    conn: Any,
    table: Any,
    by: Mapping[str, Any],
    order: Sequence[Sort],
    *,
    alias: str = "t",
    where: Sequence[Fragment] = (),
    live: bool = False,
) -> str:
    """the hint a keyed read of ``table`` pinning ``by`` in ``order`` starts with on YugabyteDB: the declared index
    that serves it (:func:`~threetears.interfaces.keyed.keys.serving_index`), named to pg_hint_plan
    (:func:`~threetears.interfaces.keyed.keys.index_hint`). Nothing on any other database, for a read that does not go
    to L3, or where no declared index but the primary key serves it.

    The one place a keyed helper asks it: YugabyteDB read one agent's conversations of one status through the primary
    key filtering the status, on every one of 15 fresh worlds, rather than the declared ``(agent_id, status,
    conversation_id)`` (3tears #87, #88); statistics did not move it.

    :param conn: the reader
    :ptype conn: Any
    :param table: the relation read
    :ptype table: Any
    :param by: the read's pins
    :ptype by: Mapping[str, Any]
    :param order: the read's sort
    :ptype order: Sequence[Sort]
    :param alias: the alias the statement reads the table by (a keyed count reads it as ``r``)
    :ptype alias: str
    :param where: the read's guards beside its pins: each ``<col> IS NOT NULL`` of them states that guard to a
        ``col IS NOT NULL`` partial index (:func:`~threetears.interfaces.keyed.keys.not_null_guards`; WHY at
        :func:`~threetears.interfaces.keyed.keys.partial_predicate`)
    :ptype where: Sequence[Fragment]
    :param live: the read answers live rows alone (it carries :func:`live_only`'s guard, as every shared keyed read
        of a stamped L3 table does): a stamped table's live-only index (``WHERE NOT gone``) then serves it. False for
        a read of the table as stored (``with_gone``: the restore, quiet cleanup), which no live-only index serves: it
        holds none of the gone rows such a read answers (WHY at
        :func:`~threetears.interfaces.keyed.keys.partial_predicate`)
    :ptype live: bool
    :return: the hint and a space, or the empty string
    :rtype: str
    """
    declared = stamped_table(table, l3=reads_l3(conn)) if live else None
    flags = frozenset() if declared is None else frozenset({declared.gone})
    guarded = not_null_guards(fragment.sql for fragment in where)
    index = serving_index(table, by, order, guarded=guarded, live=flags)
    if index is None or index.primary or not reads_l3(conn):
        return ""
    return index_hint(index, alias) if await known_on_yugabyte(conn) else ""
