"""keyed reads: the rows a batch of keys (or key prefixes) names, a bounded batch a statement.

The one spelling of the keyed batch read. Each key column's values are bound as one text array, the array cast as a
whole to the column's type, so the column is compared with values of its own type and its index is read by them::

    SELECT <columns> FROM <table> AS t WHERE t."id" = ANY($1::text[]::uuid[]) AND t."tenant" = $2

    /*+ YbBatchedNL(ANY_subquery t) */ SELECT <columns> FROM <table> AS t
     WHERE (t."a", t."b") IN (SELECT unnest($1::text[]) AS k0, unnest($2::text[]::bigint[]) AS k1)

One key column is an index condition (``= ANY``). Several are joined to the table as rows, a nested loop over the
key's index (batched on YugabyteDB by the hint). Never a column = ANY its values beside the keys when those values'
cross product names many more rows than the keys (600 keys over 50 states, 200 geographies and 30 ids name the whole
table, and the planner reads it whole); but when it names few more (prefixes sharing their lead, a page's 200
geographies of one state), each column is bound to its distinct values too (``AND t."a" = ANY($3::text[])``): the
index's own condition, where the keys alone, each naming many rows, plan on PostgreSQL as a hash join over the whole
table (300,000 rows read for 6,000).

With the key columns a leading part of the table's primary key (or of an index declared for the caller's read),
PostgreSQL, YugabyteDB and DuckDB each answer it by the key: a lookup whose cost is the keys', whatever the table
holds. ``by`` pins columns ahead of the batch (a tenant, a scope) or adds a set filter
(:class:`~threetears.core.keyed.AnyOf`).

A batch of ``batch`` keys a statement keeps a statement's parameters and its answer bounded. A transport that cuts
an answer without saying so (the L3 rail, :data:`~threetears.core.backends.protocol.L3_RAIL_ROW_CAP`) is named by
``row_cap``: an answer that reaches it is read again in halves, and one key that alone reaches it is refused
(:class:`KeyTooWide`): such a key is a prefix with many rows, read a page at a time
(:func:`threetears.core.keyed.keyset_page` or :func:`threetears.core.keyset.keyset_pages`).

IO only: each statement is awaited on the caller's connection; no row is touched in Python beyond the answer's own
dicts (:func:`read_keyed`) or the Arrow table (:func:`read_keyed_arrow`, parsed off the interpreter lock by
:func:`~threetears.core.backends.arrow_reply.l3_fetch_arrow`).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from operator import itemgetter
from typing import Any, Literal, overload

from threetears.core.backends.arrow_reply import (
    counted_type,
    flag_type,
    l3_fetch_arrow,
    without_counted,
    without_full_groups,
)
from threetears.core.backends.protocol import L3AnswerCutError
from threetears.core.keyed.binding import binder_for, key_chunks_off_loop
from threetears.core.keyed.page import newest_listing
from threetears.core.keyed._filters import (
    Derived,
    Fragment,
    Lock,
    Selected,
    binds_native_arrays,
    by_conditions,
    live_only,
    live_read,
    narrowing,
    newest_of,
    newest_sent,
    reads_yugabyte,
    relation,
    renumbered,
    sent_by,
)
from threetears.core.sql_fragments import quote_identifier
from threetears.interfaces.keyed.rows import (
    ANY_ROWS_CAP,
    DEFAULT_KEY_BATCH,
    DIRECT_KEY_BATCH,
    DeclaredKey,
    DeclaredScope,
    KeyBatch,
    KeyChunk,
    KeyColumn,
    KeyColumns,
    KeyValues,
    KeyTooWide,
    declared_key,
    keyed_read_statement,
)
from threetears.interfaces.keyed.keys import (
    NEWEST_KEPT,
    NEWEST_LIVE,
    NEWEST_VERSIONS,
    newest_page_statement,
    next_newest_keys,
)
from threetears.interfaces.keyed.page import Sort, next_past

__all__ = [
    "ANY_ROWS_CAP",
    "DEFAULT_KEY_BATCH",
    "DIRECT_KEY_BATCH",
    "DeclaredKey",
    "DeclaredScope",
    "KeyBatch",
    "KeyColumn",
    "KeyColumns",
    "KeyValues",
    "KeyTooWide",
    "declared_key",
    "any_rows",
    "key_batch_for",
    "read_joined",
    "read_keyed",
    "read_keyed_arrow",
]


def key_batch_for(conn: Any) -> int:
    """the keys a statement names on ``conn`` when the caller does not say.

    :param conn: the connection a keyed helper reads through
    :ptype conn: Any
    :return: :data:`DIRECT_KEY_BATCH` where its arrays bind natively, else :data:`DEFAULT_KEY_BATCH`
    :rtype: int
    """
    return DIRECT_KEY_BATCH if binds_native_arrays(conn) else DEFAULT_KEY_BATCH


@dataclass
class _Fit:
    """the most keys a statement of one read has been seen to answer whole: after a cut, every later batch of the
    read is cut to it before it is sent, so one too-wide answer costs one statement, not one a batch."""

    keys: int | None = None


def _sliced(keys: KeyChunk, start: int, stop: int | None = None) -> KeyChunk:
    """keys ``start`` to ``stop`` of a chunk, each column sliced (an Arrow column's slice copies nothing)."""
    return tuple(column[start:stop] for column in keys)


#: the SQL types whose key values a prefix batch pins as they are (a prefix read a range at a time,
#: :func:`_prefix_scope`): a text array's value is the column's own; any other column's value is its type's (an Arrow
#: column of the type), never text cast back
_TEXT_CASTS = frozenset({"text", "varchar", "character varying"})


def _reads_prefix_scopes(newest: Any, keys: KeyBatch, yugabyte: bool) -> bool:
    """whether a keyed read is a batch of a versioned table's key prefixes on YugabyteDB, bound as arrays: each of
    its keys a scope of many keys (a state's geographies), whose batched ``DISTINCT ON`` sorts every version of the
    scopes it names. WHY bounded apart (2026-11-25): sized by keys like a batch of whole keys, its first statement
    named :data:`~threetears.interfaces.keyed.keys._NEWEST_FIRST_KEYS` prefixes and sorted all their scopes' versions
    in one statement, past any ``work_mem`` (pods have no scratch volume: no disk spill, by design)."""
    width = len(keys.columns)
    return (
        newest is not None
        and yugabyte
        and not keys.as_list
        and not keys.null_safe
        and width < len(newest.key)
        and tuple(newest.key[:width]) == tuple(keys.columns)
    )


def _prefix_pins(keys: KeyBatch, chunk: KeyChunk, at: int = 0) -> dict[str, Any] | None:
    """the prefix a chunk names at ``at`` as pins (column -> value), or None for a value bound as text for a column
    of another type (a read that names it so is read as a batch)."""
    pins: dict[str, Any] = {}
    for index, column in enumerate(keys.columns):
        value = chunk[index][at]
        value = value.as_py() if hasattr(value, "as_py") else value
        if isinstance(value, str) and keys.cast(index).lower() not in _TEXT_CASTS:
            return None
        pins[column] = value
    return pins


def _chunk_prefixes(keys: KeyBatch, chunk: KeyChunk) -> list[dict[str, Any]] | None:
    """every prefix a chunk names, each as pins (:func:`_prefix_pins`), in order; None when one is bound as text for
    a column of another type. Function-local, one a prefix of a chunk already sized (:class:`_NewestSizing`): a
    statement's branches are written from them, one a prefix (:func:`_scopes_statement`)."""
    prefixes: list[dict[str, Any]] = []
    for at in range(len(chunk[0])):
        pins = _prefix_pins(keys, chunk, at)
        if pins is None:
            return None
        prefixes.append(pins)
    return prefixes


def _scopes_statement(
    newest: Any,
    table: str,
    columns: Sequence[str | Selected],
    by: Mapping[str, Any],
    prefixes: Sequence[Mapping[str, Any]],
    where: Sequence[Fragment],
    keys: int,
) -> tuple[str, list[Any]]:
    """several prefixes' first listing statements in one, each under its own ``LIMIT``: ``(<prefix 1's>) UNION ALL
    (<prefix 2's>) ...``, each :func:`~threetears.interfaces.keyed.keys.newest_page_statement` with its prefix pinned
    inside the ``DISTINCT ON`` and ``keys`` its keys at most, so the statement reads at most ``keys`` keys of any
    prefix, however big, never one prefix's whole scope sorted beside the rest.

    :return: the statement and its parameters
    :rtype: tuple[str, list[Any]]
    """
    order = [Sort(column, cast=None) for column in newest.key]
    branches: list[str] = []
    params: list[Any] = []
    for pins in prefixes:
        built = newest_page_statement(
            newest,
            table,
            columns=columns,
            by={**by, **pins},
            order=order,
            keys=keys,
            past=None,
            by_row=False,
            where=where,
        )
        assert built is not None  # noqa: S101 -- a key's own order is always the listing's
        sql, values = built
        branches.append(f"({renumbered(Fragment(sql, tuple(values)), len(params))})")
        params += values
    return " UNION ALL ".join(branches), params


async def _prefix_scopes(
    conn: Any,
    newest: Any,
    table: str,
    columns: Sequence[str | Selected],
    by: Mapping[str, Any],
    prefixes: Sequence[Mapping[str, Any]],
    where: Sequence[Fragment],
    page: int,
) -> tuple[list[dict[str, Any]], int]:
    """several prefixes' rows, a prefix's keys at most ``page`` in one statement (:func:`_scopes_statement`), each
    prefix that filled its ``page`` then read alone a range at a time (:func:`_prefix_scope`): so a batch of prefixes
    is sized per prefix, and one big prefix among small ones is paged, never sorted whole beside them. WHY
    (2026-11-25): a batch of prefixes sized by the versions its first prefix read named as many as that held; one
    big prefix among them was read whole in the batch's one ``DISTINCT ON``, past any ``work_mem``. And the versions
    read, which size the next batch."""
    prefix = list(prefixes[0])
    named = {column if isinstance(column, str) else column.alias for column in columns}
    hidden = frozenset(column for column in newest.key if column not in named)
    read = [*columns, *sorted(hidden)]
    sql, params = _scopes_statement(newest, table, read, by, prefixes, where, page)
    answered = await newest_sent(conn, newest, True, batch=False).fetch(sql, *params)
    held: dict[tuple[Any, ...], int] = {}
    for row in answered:
        named_by = tuple(row[column] for column in prefix)
        held[named_by] = held.get(named_by, 0) + 1
    full = {named_by for named_by, keys in held.items() if keys >= page}
    dropped = _NEWEST_ROWS | hidden
    rows: list[dict[str, Any]] = []
    versions = 0
    for row in answered:
        if tuple(row[column] for column in prefix) in full:
            continue
        versions += int(row[NEWEST_VERSIONS])
        if row[NEWEST_KEPT]:
            rows.append({name: value for name, value in dict(row).items() if name not in dropped})
    for pins in prefixes:
        if tuple(pins[column] for column in prefix) in full:
            listed = await _prefix_scope(conn, newest, table, columns, by, pins, where, page)
            assert listed is not None  # noqa: S101 -- a key's own order is always the listing's
            rows.extend(listed[0])
            versions += listed[1]
    return rows, versions


async def _prefix_scopes_arrow(
    conn: Any,
    newest: Any,
    table: str,
    columns: Sequence[str | Selected],
    types: Mapping[str, Any],
    by: Mapping[str, Any],
    prefixes: Sequence[Mapping[str, Any]],
    where: Sequence[Fragment],
    page: int,
) -> tuple[list[Any], int] | None:
    """:func:`_prefix_scopes`, the statement's rows one Arrow table (each full prefix told apart by Arrow's own group
    count and anti-join, :func:`~threetears.core.backends.arrow_reply.without_full_groups`; no row a Python object);
    None when the read's columns leave out a key column, which tells a row's prefix apart and seeks past its last key
    (such a read is read as a batch)."""
    named = [column if isinstance(column, str) else column.alias for column in columns]
    if any(column not in named or column not in types for column in newest.key):
        return None
    prefix = list(prefixes[0])
    fetched = {**types, NEWEST_KEPT: flag_type(), NEWEST_VERSIONS: counted_type()}
    sql, params = _scopes_statement(newest, table, columns, by, prefixes, where, page)
    answered = await l3_fetch_arrow(newest_sent(conn, newest, True, batch=False), sql, *params, types=fetched)
    rest, full = without_full_groups(answered, prefix, page)
    counted, versions = without_counted(rest, NEWEST_VERSIONS)
    tables = [counted.filter(counted.column(NEWEST_KEPT)).drop_columns([NEWEST_KEPT])]
    # the full prefixes alone made pins: one a prefix that filled its page, each read again on its own
    for pins in full.to_pylist():
        listed = await _prefix_scope_arrow(conn, newest, table, columns, types, by, pins, where, page)
        if listed is None:
            return None
        tables.extend(listed[0])
        versions += listed[1]
    return tables, versions


async def _prefix_scope(
    conn: Any,
    newest: Any,
    table: str,
    columns: Sequence[str | Selected],
    by: Mapping[str, Any],
    pins: Mapping[str, Any],
    where: Sequence[Fragment],
    page: int,
) -> tuple[list[dict[str, Any]], int] | None:
    """one prefix's rows read a range of its keys at a time, in the key's order
    (:func:`~threetears.core.keyed.page.newest_listing`: the prefix pinned and the cursor inside the ``DISTINCT ON``,
    each statement's ``LIMIT`` the keys that hold :data:`~threetears.interfaces.keyed.keys._NEWEST_ROWS_A_READ`
    versions at the versions a key the one before read), so a scope of any size is read in bounded statements, each
    seeking past the pinned lead; and the versions read, which size the next batch."""
    rest = [column for column in newest.key if column not in pins]
    named = {column if isinstance(column, str) else column.alias for column in columns}
    hidden = frozenset(column for column in rest if column not in named)
    order = [Sort(column, cast=None) for column in newest.key]
    read = [*columns, *sorted(hidden)]
    return await newest_listing(conn, newest, table, read, {**by, **pins}, order, None, None, (), where, page, hidden)


async def _prefix_scope_arrow(
    conn: Any,
    newest: Any,
    table: str,
    columns: Sequence[str | Selected],
    types: Mapping[str, Any],
    by: Mapping[str, Any],
    pins: Mapping[str, Any],
    where: Sequence[Fragment],
    page: int,
) -> tuple[list[Any], int] | None:
    """:func:`_prefix_scope`, each statement's rows one Arrow table (kept by Arrow's own filter, its versions by
    Arrow's sum, no row a Python object); None when the read's columns leave out a key column past the prefix, whose
    Arrow type it cannot name for its cursor (such a read is read as a batch)."""
    rest = [column for column in newest.key if column not in pins]
    if any(column not in types for column in rest):
        return None
    order = [Sort(column, cast=None) for column in newest.key]
    fetched = {**types, NEWEST_KEPT: flag_type(), NEWEST_VERSIONS: counted_type()}
    sender = newest_sent(conn, newest, True, batch=False)
    tables: list[Any] = []
    total = 0
    past: list[Any] | None = None
    asked = next_newest_keys(0, 0, page)
    while True:
        built = newest_page_statement(
            newest,
            table,
            columns=columns,
            by={**by, **pins},
            order=order,
            keys=asked,
            past=past,
            by_row=False,
            where=where,
        )
        assert built is not None  # noqa: S101 -- a key's own order is always the listing's
        sql, params = built
        answered = await l3_fetch_arrow(sender, sql, *params, types=fetched)
        counted, versions = without_counted(answered, NEWEST_VERSIONS)
        total += versions
        tables.append(counted.filter(counted.column(NEWEST_KEPT)).drop_columns([NEWEST_KEPT]))
        if answered.num_rows < asked:
            return tables, total
        # past the last key read, kept or not: one row's key, read off the table
        last = answered.slice(answered.num_rows - 1, 1).select(rest).to_pylist()[0]
        past = next_past(order, {**pins, **last})
        asked = next_newest_keys(answered.num_rows, versions, page)


#: the columns a newest listing statement answers beside each row, never handed to the caller
_NEWEST_ROWS = frozenset({NEWEST_KEPT, NEWEST_LIVE, NEWEST_VERSIONS})


class _NewestSizing:
    """how many keys a versioned batch on YugabyteDB asks for: its first
    :data:`~threetears.interfaces.keyed.keys._NEWEST_FIRST_KEYS` (the budget at the measured worst, 50 versions a key),
    each later one as many as hold
    :data:`~threetears.interfaces.keyed.keys._NEWEST_ROWS_A_READ` versions at the versions a key the one before read
    (:func:`~threetears.interfaces.keyed.keys.next_newest_keys`), so the batch's sort of every version of its keys
    never outgrows its ``work_mem``. Local to one read."""

    def __init__(self, asked: int, *, first: int | None = None) -> None:
        self._asked = asked
        # a batch of key prefixes knows nothing yet of how many keys a prefix holds: its first statement names one
        self.keys = next_newest_keys(0, 0, asked) if first is None else first

    def read(self, keys: int, versions: int) -> None:
        """learn from a batch: ``keys`` named, ``versions`` read of the keys it answered."""
        self.keys = next_newest_keys(keys, versions, self._asked)


async def _chunks(keys: KeyBatch, batch: int, binder: Any, sizing: _NewestSizing | None) -> Iterable[KeyChunk]:
    """the read's chunks of distinct keys: ``batch`` a chunk (:func:`key_chunks_off_loop`), or, for a versioned batch
    sized by its versions, as many as ``sizing`` asks for each in turn. Either way a large Arrow batch's distinct keys
    are found in a worker process, never on the loop (``key_chunks_off_loop``)."""
    if sizing is None:
        return await key_chunks_off_loop(keys, batch, binder)
    # every distinct key as one chunk, off the loop as any other read's; the sized chunks are zero-copy slices of it
    whole = await key_chunks_off_loop(keys, max(1, len(keys.keys[0]) if keys.keys else 1), binder)
    return _sized(whole[0], sizing) if whole else ()


def _sized(unique: KeyChunk, sizing: _NewestSizing) -> Iterator[KeyChunk]:
    """``unique``'s keys, as many a chunk as ``sizing`` asks for when each chunk is taken."""
    total = len(unique[0])
    start = 0
    while start < total:
        size = sizing.keys
        yield tuple(column[start : start + size] for column in unique)
        start += size


async def _read_split(
    fetch: Callable[[KeyChunk], Awaitable[Any]],
    size: Callable[[Any], int],
    keys: KeyChunk,
    row_cap: int | None,
    table: str | Derived,
    fit: _Fit | None = None,
) -> list[Any]:
    """one batch's answers: the batch read whole, or in halves while an answer reaches ``row_cap`` or the transport
    cuts it (:class:`~threetears.core.backends.protocol.L3AnswerCutError`: its row cap, or a reply's bytes, as the
    hub's datasource replies are cut): each half is another statement, never the cut answer taken as every row."""
    fit = fit if fit is not None else _Fit()
    count = len(keys[0])
    if fit.keys is not None and count > fit.keys:
        return [
            *await _read_split(fetch, size, _sliced(keys, 0, fit.keys), row_cap, table, fit),
            *await _read_split(fetch, size, _sliced(keys, fit.keys), row_cap, table, fit),
        ]
    try:
        answered = await fetch(keys)
    except L3AnswerCutError:
        if count == 1:
            raise
        cut = True
    else:
        if row_cap is None or size(answered) < row_cap:
            return [answered]
        cut = False
    if count == 1:
        raise KeyTooWide(
            f"{table}: the key {tuple(column[0] for column in keys)!r} alone reaches the transport's {row_cap} rows a statement; page it "
            "(threetears.core.keyed.keyset_page)"
        )
    half = count // 2
    if cut:
        fit.keys = half if fit.keys is None else min(fit.keys, half)
    return [
        *await _read_split(fetch, size, _sliced(keys, 0, half), row_cap, table, fit),
        *await _read_split(fetch, size, _sliced(keys, half), row_cap, table, fit),
    ]


@overload
async def read_keyed(
    conn: Any,
    table: str | Derived,
    *,
    keys: KeyBatch,
    columns: Sequence[str | Selected],
    by: Mapping[str, Any] | None = ...,
    batch: int | None = ...,
    row_cap: int | None = ...,
    lock: Lock | None = ...,
    where: Sequence[Fragment] = ...,
    records: Literal[False] = ...,
) -> list[dict[str, Any]]: ...


@overload
async def read_keyed(
    conn: Any,
    table: str | Derived,
    *,
    keys: KeyBatch,
    columns: Sequence[str | Selected],
    by: Mapping[str, Any] | None = ...,
    batch: int | None = ...,
    row_cap: int | None = ...,
    lock: Lock | None = ...,
    where: Sequence[Fragment] = ...,
    records: Literal[True],
) -> list[Any]: ...


async def read_keyed(
    conn: Any,
    table: str | Derived,
    *,
    keys: KeyBatch,
    columns: Sequence[str | Selected],
    by: Mapping[str, Any] | None = None,
    batch: int | None = None,
    row_cap: int | None = None,
    lock: Lock | None = None,
    where: Sequence[Fragment] = (),
    records: bool = False,
) -> list[dict[str, Any]] | list[Any]:
    """the rows of ``table`` the keys name (within ``by``), ``batch`` keys a statement.

    :param conn: anything with an asyncpg-shaped ``fetch(query, *params)``: an L3 backend, the rail's proxy, a
        connection, or a DuckDB connection behind :class:`~threetears.core.keyed.ThreadedDuckDB`
    :ptype conn: Any
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param keys: the keys, a leading part of the table's key (or of an index declared for the caller's read)
    :ptype keys: KeyBatch
    :param columns: the columns to read, TRUSTED identifiers, or :class:`~threetears.core.keyed.Selected` expressions
        under their own names (the warehouse's text of a timestamp: ``Selected("CAST(copied_at AS VARCHAR)",
        "copied_at")``)
    :ptype columns: Sequence[str | Selected]
    :param by: columns pinned beside the keys (a tenant), each a value or an :class:`~threetears.core.keyed.AnyOf`
    :ptype by: Mapping[str, Any] | None
    :param batch: the most keys a statement names; None: :func:`key_batch_for` the connection
    :ptype batch: int | None
    :param row_cap: the most rows the transport answers a statement without saying it cut; None when it never cuts
    :ptype row_cap: int | None
    :param lock: the row lock taken on the rows read (the caller's transaction), in the key's order; none when None
    :ptype lock: Lock | None
    :param where: filters beside the key, each a fragment written as ``t."col"`` from ``$1`` (a status the row must
        not hold, a range); none when empty
    :ptype where: Sequence[Fragment]
    :param records: answer the rows as the connection answers them (asyncpg ``Record``s on a direct connection:
        read by name or place, never written), with no copy of each into a dict (2,000 rows: 0.45 ms); dicts when
        False
    :ptype records: bool
    :return: the rows, in no promised order (in the key's order when locked)
    :rtype: list[dict[str, Any]] | list[Any]
    :raises ValueError: when no column is named, ``batch`` is under one, or a key does not match the columns
    :raises KeyTooWide: when one key alone reaches ``row_cap``
    """
    binder = binder_for(conn, keys)
    newest = newest_of(conn, table)
    # a versioned table's newest live versions (the statement's newest form), a stamped one's live rows alone
    statement = keyed_read_statement(
        table, columns, keys, by or {}, lock, live_only(conn, table, where), binder, newest
    )
    natively = binds_native_arrays(conn)
    # asked only where it changes the statement: a batch of several key columns (its bounds, PostgreSQL's alone), or
    # any batch of a versioned table (a batched DISTINCT ON on YugabyteDB, a LATERAL seek a key on PostgreSQL)
    yugabyte = (newest is not None or len(keys.columns) > 1) and not keys.as_list and await reads_yugabyte(conn)
    counted = statement.counted(yugabyte)

    async def fetch(chunk: KeyChunk) -> list[Any]:
        sql, params = statement.render(chunk, natively, yugabyte)
        return list(await newest_sent(conn, newest, yugabyte).fetch(sql, *params))

    rows: list[Any] = []
    fit = _Fit()
    asked = key_batch_for(conn) if batch is None else batch
    scopes = _reads_prefix_scopes(newest, keys, yugabyte) and isinstance(table, str)
    sizing = _NewestSizing(asked, first=1 if scopes else None) if counted else None
    for chunk in await _chunks(keys, asked, binder, sizing):
        prefixes = _chunk_prefixes(keys, chunk) if scopes else None
        if prefixes and sizing is not None and isinstance(table, str):
            page = max(1, (row_cap - 1) if row_cap is not None else asked)
            if len(prefixes) == 1:
                # one prefix alone: read a range of its keys at a time, never its whole scope in one sort
                listed = await _prefix_scope(conn, newest, table, columns, by or {}, prefixes[0], where, page)
                assert listed is not None  # noqa: S101 -- a key's own order is always the listing's
            else:
                # several: each under its own LIMIT, a prefix that fills it read alone a range at a time
                listed = await _prefix_scopes(conn, newest, table, columns, by or {}, prefixes, where, page)
            rows.extend(listed[0])
            sizing.read(len(prefixes), listed[1])
            continue
        versions = 0
        for answered in await _read_split(fetch, len, chunk, row_cap, table, fit):
            if counted:
                # the versions each key read sizes the next batch; the count is the read's own, never a column of
                # the rows answered
                versions += sum(row[NEWEST_VERSIONS] for row in answered)
                rows.extend(
                    {name: value for name, value in dict(row).items() if name != NEWEST_VERSIONS} for row in answered
                )
            elif records:
                rows.extend(answered)
            else:
                # mapped, not a generator: a row a call less, at a page's 6,000 rows a millisecond
                rows.extend(map(dict, answered))
        if sizing is not None:
            sizing.read(len(chunk[0]), versions)
    return rows


async def read_keyed_arrow(
    conn: Any,
    table: str | Derived,
    *,
    keys: KeyBatch,
    types: Mapping[str, Any],
    by: Mapping[str, Any] | None = None,
    batch: int | None = None,
    row_cap: int | None = None,
    columns: Sequence[str | Selected] | None = None,
    where: Sequence[Fragment] = (),
    with_gone: bool = False,
) -> list[Any]:
    """:func:`read_keyed`, each statement's answer one Arrow table: the same statements, the rows never Python
    objects (the rail's proxy parses its reply off the interpreter lock).

    :param conn: the rail's proxy (``fetch_arrow``), or anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param keys: the keys
    :ptype keys: KeyBatch
    :param types: column name -> its Arrow type, every column read, in order
    :ptype types: Mapping[str, pyarrow.DataType]
    :param by: columns pinned beside the keys
    :ptype by: Mapping[str, Any] | None
    :param batch: the most keys a statement names; None: :func:`key_batch_for` the connection
    :ptype batch: int | None
    :param row_cap: the most rows the transport answers a statement without saying it cut
    :ptype row_cap: int | None
    :param columns: the select list, as :func:`read_keyed` takes it (a :class:`~threetears.core.keyed.Selected`
        expression under its name); None selects the columns ``types`` names. Its names are ``types``' keys, in order
    :ptype columns: Sequence[str | Selected] | None
    :param where: filters beside the key, as :func:`read_keyed` takes them
    :ptype where: Sequence[Fragment]
    :param with_gone: read the table as stored, as ``copy_keyed(with_gone=True)`` does: a versioned table's every
        version and gone marker, a stamped one's gone rows too. Quiet-time cleanup alone, which reads the versions it
        removes (:mod:`~threetears.core.collections.quiet_cleanup`); every other read sees live rows alone
    :ptype with_gone: bool
    :return: one ``pyarrow.Table`` a statement, in the order read
    :rtype: list[pyarrow.Table]
    :raises ValueError: as :func:`read_keyed`
    :raises KeyTooWide: when one key alone reaches ``row_cap``
    """
    binder = binder_for(conn, keys)
    # a versioned table's newest live versions (the statement's newest form), a stamped one's live rows alone; as
    # stored for a read of the versions themselves
    newest = None if with_gone else newest_of(conn, table)
    statement = keyed_read_statement(
        table,
        list(types) if columns is None else columns,
        keys,
        by or {},
        None,
        tuple(where) if with_gone else live_only(conn, table, where),
        binder,
        newest,
    )
    natively = binds_native_arrays(conn)
    # asked only where it changes the statement, as read_keyed asks it
    yugabyte = (newest is not None or len(keys.columns) > 1) and not keys.as_list and await reads_yugabyte(conn)
    counted = statement.counted(yugabyte)
    fetched = {**types, NEWEST_VERSIONS: counted_type()} if counted else types

    async def fetch(chunk: KeyChunk) -> Any:
        sql, params = statement.render(chunk, natively, yugabyte)
        return await l3_fetch_arrow(newest_sent(conn, newest, yugabyte), sql, *params, types=fetched)

    tables: list[Any] = []
    fit = _Fit()
    asked = key_batch_for(conn) if batch is None else batch
    scopes = _reads_prefix_scopes(newest, keys, yugabyte) and isinstance(table, str)
    sizing = _NewestSizing(asked, first=1 if scopes else None) if counted else None
    for chunk in await _chunks(keys, asked, binder, sizing):
        prefixes = _chunk_prefixes(keys, chunk) if scopes else None
        if prefixes and sizing is not None and isinstance(table, str):
            page = max(1, (row_cap - 1) if row_cap is not None else asked)
            selected = list(types) if columns is None else columns
            # one prefix alone read a range of its keys at a time; several each under its own LIMIT, a prefix that
            # fills it read alone (_prefix_scopes)
            listed = (
                await _prefix_scope_arrow(conn, newest, table, selected, types, by or {}, prefixes[0], where, page)
                if len(prefixes) == 1
                else await _prefix_scopes_arrow(conn, newest, table, selected, types, by or {}, prefixes, where, page)
            )
            if listed is not None:
                tables.extend(listed[0])
                sizing.read(len(prefixes), listed[1])
                continue
        versions = 0
        for answered in await _read_split(fetch, lambda answered: int(answered.num_rows), chunk, row_cap, table, fit):
            if counted:
                answered, read = without_counted(answered, NEWEST_VERSIONS)
                versions += read
            tables.append(answered)
        if sizing is not None:
            sizing.read(len(chunk[0]), versions)
    return tables


async def read_joined(
    conn: Any,
    rows: Sequence[Mapping[str, Any]],
    table: str | Derived,
    *,
    on: Mapping[str, str],
    columns: Sequence[str],
    casts: Mapping[str, str] | None = None,
    by: Mapping[str, Any] | None = None,
    batch: int | None = None,
) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    """the rows of ``table`` joined to ``rows`` (a page already read), by a keyed read of the values they name.

    A listing with joins, as a page and keyed reads: the page is read by its own key (:func:`keyset_page`), then each
    joined table by the values the page's rows hold (``on``: its column -> the page's), one keyed batch, never a join
    the planner may drive from the joined side. The page's rows are matched to theirs in Python, by those values.

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param rows: the page's rows
    :ptype rows: Sequence[Mapping[str, Any]]
    :param table: the joined table, a TRUSTED identifier
    :ptype table: str
    :param on: its columns -> the page's columns they equal, a leading part of its key (or of an index declared for
        the caller's read)
    :ptype on: Mapping[str, str]
    :param columns: the joined table's columns read; each of ``on``'s among them
    :ptype columns: Sequence[str]
    :param casts: each ``on`` column's SQL type; ``text`` when absent
    :ptype casts: Mapping[str, str] | None
    :param by: the joined table's columns pinned beside (a tenant)
    :ptype by: Mapping[str, Any] | None
    :param batch: the most values a statement names; None: :func:`key_batch_for` the connection
    :ptype batch: int | None
    :return: the page's values (in ``on``'s order) -> the joined rows holding them, read-only, as the connection
        answers them (asyncpg ``Record``s on a direct connection: read by name, never written); a value no row holds
        is absent. WHY not dicts (2026-11-18, brooks-5950x, 6,000 rows): the bare statement 3.54 ms, the helper's own
        statement 3.32 ms, but a dict of each row 2.25 ms more, and the read sat at 2.0-2.1x its bar
    :rtype: dict[tuple[Any, ...], list[Mapping[str, Any]]]
    :raises ValueError: when ``on`` is empty or a column of it is not read
    """
    if not on:
        raise ValueError(f"a joined read of {table} names how it joins the page")
    inner = tuple(on)
    missing = [column for column in inner if column not in columns]
    if missing:
        raise ValueError(f"a joined read of {table} reads its join columns, missing {missing}")
    cast_of = dict(casts or {})
    # the page's rows are Python rows (a page read): their join values taken a column at a time, a row naming a NULL
    # among them dropped
    named: list[list[Any]] = [list(map(itemgetter(on[column]), rows)) for column in inner]
    if any(value is None for column in named for value in column):
        kept = [key for key in zip(*named, strict=True) if None not in key]
        named = [list(column) for column in zip(*kept, strict=True)] if kept else [[] for _ in inner]
    joined: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    if not named[0]:
        return joined
    keys = KeyBatch(inner, named, tuple(cast_of.get(column, "text") for column in inner))
    # each row's join values by one itemgetter, not a tuple built a column at a time (6,000 rows: 0.6 ms against 1.8)
    getter = itemgetter(*inner)
    key_of = getter if len(inner) > 1 else (lambda row: (getter(row),))
    for row in await read_keyed(conn, table, keys=keys, columns=columns, by=by, batch=batch, records=True):
        key = key_of(row)
        held = joined.get(key)
        if held is None:
            joined[key] = [row]
        else:
            held.append(row)
    return joined


async def any_rows(
    conn: Any,
    table: str | Derived,
    columns: Sequence[str],
    *,
    limit: int,
    by: Mapping[str, Any] | None = None,
    where: Sequence[Fragment] = (),
) -> list[dict[str, Any]]:
    """at most ``limit`` rows of ``table`` that ``by`` and ``where`` name, in no order: a bounded look.

    ``SELECT <columns> FROM <table> AS t WHERE <by> AND <where> LIMIT <limit>``, no ``ORDER BY``: the database
    stops at the bound wherever its scan stands, so the look costs about ``limit`` rows. For an operator command that
    needs any rows, not given ones (take any 500 to empty a table a statement at a time, any one run to name): which
    rows come is the database's choice and may differ call to call.

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param table: the table (``schema.table``), or a derived relation
    :ptype table: str | Derived
    :param columns: the columns read, TRUSTED identifiers
    :ptype columns: Sequence[str]
    :param limit: the most rows read, 1 to :data:`ANY_ROWS_CAP`
    :ptype limit: int
    :param by: columns pinned (each a value, None or an ``AnyOf``)
    :ptype by: Mapping[str, Any] | None
    :param where: filters beside ``by``, each a fragment from ``$1`` written as ``t."col"``
    :ptype where: Sequence[Fragment]
    :return: the rows, in no order
    :rtype: list[dict[str, Any]]
    :raises ValueError: when ``limit`` is under one or over :data:`ANY_ROWS_CAP`, or no column is read
    """
    if not 1 <= limit <= ANY_ROWS_CAP:
        raise ValueError(f"a bounded look takes 1 to {ANY_ROWS_CAP} rows, got {limit}")
    if not columns:
        raise ValueError(f"a bounded look of {table} reads at least one column")
    conditions, params = by_conditions(by or {}, alias="t")
    read_from, guarded = live_read(conn, table, where)
    for fragment in guarded:
        conditions.append(narrowing(fragment, len(params)))
        params += list(fragment.params)
    clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    selected = ", ".join(f"t.{quote_identifier(column)}" for column in columns)
    rows = await sent_by(conn).fetch(
        f"SELECT {selected} FROM {relation(read_from, 't', params)}{clause} LIMIT {limit}", *params
    )
    return [dict(row) for row in rows]
