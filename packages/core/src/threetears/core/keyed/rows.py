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

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from operator import itemgetter
from typing import Any, Final, Literal, overload

from threetears.core.backends.arrow_reply import l3_fetch_arrow
from threetears.core.backends.protocol import L3AnswerCutError
from threetears.core.keyed._filters import (
    IN_SUBQUERY,
    Derived,
    Fragment,
    Lock,
    Selected,
    batched_join_hint,
    binds_native_arrays,
    by_conditions,
    relation,
    renumbered,
    selected_column,
)
from threetears.core.sql_fragments import quote_identifier

__all__ = [
    "ANY_ROWS_CAP",
    "DEFAULT_KEY_BATCH",
    "DIRECT_KEY_BATCH",
    "KeyBatch",
    "KeyTooWide",
    "any_rows",
    "key_batch_for",
    "read_joined",
    "read_keyed",
    "read_keyed_arrow",
]

#: the most rows one :func:`any_rows` look takes
ANY_ROWS_CAP: Final = 1000

#: keys a statement names when the caller does not say, through a transport that carries its parameters as JSON
#: and bounds its replies (the L3 rail, a datasource): well under a statement's parameter and reply bounds
DEFAULT_KEY_BATCH: Final = 1000

#: keys a statement names when the caller does not say, on a connection that binds arrays natively (asyncpg's, to
#: PostgreSQL or YugabyteDB): each key column is ONE array parameter, so the batch is bounded by the statement's work,
#: not by its parameters, and a second statement costs more than the keys it splits off (2,000 keys: 0.5-0.65 ms
#: more as two statements than as one)
DIRECT_KEY_BATCH: Final = 10_000


def key_batch_for(conn: Any) -> int:
    """the keys a statement names on ``conn`` when the caller does not say.

    :param conn: the connection a keyed helper reads through
    :ptype conn: Any
    :return: :data:`DIRECT_KEY_BATCH` where its arrays bind natively, else :data:`DEFAULT_KEY_BATCH`
    :rtype: int
    """
    return DIRECT_KEY_BATCH if binds_native_arrays(conn) else DEFAULT_KEY_BATCH


#: a batch's key columns are each bound to their distinct values when those name at most this many times its keys
_TIGHT_BOUND_FACTOR: Final = 2

#: the key column types bound as text arrays, with no cast
_TEXT_TYPES: Final = frozenset({"text", "varchar"})

#: the most statement shapes whose text is kept rendered; past it the oldest is let go
_RENDERED_SHAPES: Final = 4096

#: an array statement's text by its shape (:attr:`_Statement._shape`, which columns bind natively, whether bounded):
#: a shape read again (the same caller's next read) binds its batch's arrays to the text already rendered
_RENDERED: dict[tuple[Any, ...], str] = {}


def _remember(shape: tuple[Any, ...], sql: str) -> None:
    """keep ``sql`` as ``shape``'s text, letting the oldest shape go past :data:`_RENDERED_SHAPES`."""
    if len(_RENDERED) >= _RENDERED_SHAPES:
        _RENDERED.pop(next(iter(_RENDERED)), None)
    _RENDERED[shape] = sql


class KeyTooWide(ValueError):
    """one key alone holds as many rows as the transport answers a statement: it is a prefix to page through."""


@dataclass(frozen=True)
class KeyBatch:
    """the keys a read names: a leading part of a key, one tuple a key.

    :param columns: the key columns, TRUSTED identifiers, in key order
    :ptype columns: tuple[str, ...]
    :param values: one tuple a key, a value a column; each key is read once however often named
    :ptype values: Sequence[Sequence[Any]]
    :param casts: each column's SQL type its text is cast back to; ``text`` for every column when empty
    :ptype casts: tuple[str, ...]
    :param null_safe: a ``None`` in a key matches the rows holding none in that column (``IS NOT DISTINCT FROM``):
        a scope column's NULL bucket
    :ptype null_safe: bool
    :param as_list: bind the keys as a list of placeholders (``(c1, c2) IN (($1, $2), ...)``), each value as it is,
        for a database with no arrays (Redshift, SQLite); a statement's text then follows its batch's size
    :ptype as_list: bool
    """

    columns: tuple[str, ...]
    values: Sequence[Sequence[Any]]
    casts: tuple[str, ...] = field(default=())
    null_safe: bool = False
    as_list: bool = False

    def __post_init__(self) -> None:
        if not self.columns:
            raise ValueError("a key batch names at least one key column")
        if self.as_list and self.null_safe:
            raise ValueError("a key batch bound as a list matches by equality; a null-safe batch binds arrays")
        if self.casts and len(self.casts) != len(self.columns):
            raise ValueError(f"a key batch casts each of its {len(self.columns)} columns, got {len(self.casts)}")
        for key in self.values:
            if len(key) != len(self.columns):
                raise ValueError(f"a key of {self.columns} has {len(self.columns)} values, got {tuple(key)!r}")

    def cast(self, index: int) -> str:
        """the SQL type column ``index``'s text is cast back to.

        :param index: the column's place in :attr:`columns`
        :ptype index: int
        :return: the type
        :rtype: str
        """
        return self.casts[index] if self.casts else "text"


def _text(value: Any) -> str | None:
    """a key value as the text its array binds: its ISO form for a date or time, ``str`` otherwise."""
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    return str(isoformat()) if callable(isoformat) else str(value)


@dataclass(frozen=True)
class _Statement:
    """the statement one keyed read sends a batch: built once for arrays, once a batch for a list of keys."""

    table: str | Derived
    selected: str
    keys: KeyBatch
    by: Mapping[str, Any]
    lock: Lock | None = None
    where: Sequence[Fragment] = ()

    def _guards(self, before: int) -> tuple[list[str], list[Any]]:
        """the ``where`` fragments, renumbered past the ``before`` parameters the statement already binds."""
        conditions: list[str] = []
        params: list[Any] = []
        for fragment in self.where:
            conditions.append(renumbered(fragment, before + len(params)))
            params += list(fragment.params)
        return conditions, params

    def _locked(self, sql: str) -> str:
        if self.lock is None:
            return sql
        # locked in the key's order, so two lockers of the same keys wait on each other rather than deadlock
        ordered = ", ".join(f"t.{quote_identifier(column)}" for column in self.keys.columns)
        return f"{sql} ORDER BY {ordered}{self.lock.of('t')}"

    def render(self, chunk: Sequence[tuple[Any, ...]], natively: bool = False) -> tuple[str, tuple[Any, ...]]:
        """the statement naming ``chunk``'s keys, and its parameters.

        :param chunk: the keys
        :ptype chunk: Sequence[tuple[Any, ...]]
        :param natively: whether the connection binds typed arrays natively (:func:`binds_native_arrays`): a column
            whose values are of its type (no str) is then bound as ``$n::<type>[]`` from the values themselves
        :ptype natively: bool
        :return: the statement, and its parameters
        :rtype: tuple[str, tuple[Any, ...]]
        """
        width = len(self.keys.columns)
        if self.keys.as_list:
            head = f"SELECT {self.selected} FROM {relation(self.table, 't')}"
            columns = [f"t.{quote_identifier(column)}" for column in self.keys.columns]
            if width == 1:
                named = f"{columns[0]} IN ({', '.join(f'${place + 1}' for place in range(len(chunk)))})"
            else:
                rows = ", ".join(
                    "(" + ", ".join(f"${row * width + index + 1}" for index in range(width)) + ")"
                    for row in range(len(chunk))
                )
                named = f"({', '.join(columns)}) IN ({rows})"
            conditions, pinned = by_conditions(self.by, first=len(chunk) * width + 1, alias="t")
            guards, guarded = self._guards(len(chunk) * width + len(pinned))
            where = " AND ".join([named, *conditions, *guards])
            return self._locked(f"{head} WHERE {where}"), (
                *(value for key in chunk for value in key),
                *pinned,
                *guarded,
            )
        # each key column's values bound as one array: of the column's type where the connection binds it natively
        # and the values are of it, else a text array cast as a whole to it. The column is compared with values of its
        # own type, never wrapped or compared element by element through a cast, so the planner reads the key's index
        # by them (an Index Cond), as a hand-written "col = ANY($1::uuid[])" would
        held = [list(map(itemgetter(index), chunk)) for index in range(width)]
        native = self._native(held, natively)
        arrays = tuple(
            values if native[index] else [_text(value) for value in values] for index, values in enumerate(held)
        )
        bounded: tuple[list[Any], ...] = ()
        if width > 1 and not self.keys.null_safe:
            bounded = self._tight_values(arrays, len(chunk))
        guarded = [param for fragment in self.where for param in fragment.params]
        # the statement's text follows only its shape (which columns bind natively, whether bounded): rendered once a
        # shape, here or by an earlier read of the same shape, and each batch binds its own arrays to it
        shape = (self._shape, native, bool(bounded))
        sql = _RENDERED.get(shape)
        if sql is None:
            sql = self._array_text(native, bool(bounded))
            _remember(shape, sql)
        return sql, (*arrays, *self._by[1], *bounded, *guarded)

    @cached_property
    def _by(self) -> tuple[list[str], list[Any]]:
        """``by``'s conditions and values in an array statement, its placeholders following the key columns'."""
        return by_conditions(self.by, first=len(self.keys.columns) + 1, alias="t")

    @cached_property
    def _shape(self) -> tuple[Any, ...]:
        """everything an array statement's text follows but the batch: its relation, columns, keys' columns and
        types, ``by``'s conditions, lock and filters (each filter's text and how many parameters it binds)."""
        return (
            self.table,
            self.selected,
            self.keys.columns,
            self.keys.casts,
            self.keys.null_safe,
            self.lock,
            tuple(self._by[0]),
            tuple((fragment.sql, len(fragment.params)) for fragment in self.where),
        )

    def _array_text(self, native: tuple[bool, ...], bounded: bool) -> str:
        """the array statement's text: ``= ANY`` for one key column, else the keys joined as rows (null-safe, or by
        ``IN (SELECT unnest ...)``, each key column bounded to its distinct values when ``bounded``)."""
        width = len(self.keys.columns)
        head = f"SELECT {self.selected} FROM {relation(self.table, 't')}"
        typed = [self._typed_array(index + 1, index, native[index]) for index in range(width)]
        conditions, pinned = self._by
        columns = [f"t.{quote_identifier(column)}" for column in self.keys.columns]
        if width == 1 and not self.keys.null_safe:
            guards, _ = self._guards(width + len(pinned))
            where = " AND ".join([f"{columns[0]} = ANY({typed[0]})", *conditions, *guards])
            return self._locked(f"{head} WHERE {where}")
        unnested = ", ".join(f"unnest({array}) AS k{index}" for index, array in enumerate(typed))
        if self.keys.null_safe:
            # a NULL bucket matches only by IS NOT DISTINCT FROM, which = ANY cannot say: joined to the keys
            joined = " AND ".join(f"{column} IS NOT DISTINCT FROM k.k{index}" for index, column in enumerate(columns))
            guards, _ = self._guards(width + len(pinned))
            kept = [*conditions, *guards]
            clause = f" WHERE {' AND '.join(kept)}" if kept else ""
            return self._locked(f"{head} JOIN (SELECT {unnested}) AS k ON {joined}{clause}")
        # the keys joined to the table as rows: a nested loop over the key's index, each key one lookup. Never a
        # column = ANY its values beside it: those name the values' cross product (600 keys over 50 states, 200
        # geographies and 30 ids name the whole table), and the planner reads it whole. YugabyteDB is told to batch
        # the loop (batched_join_hint): one storage request for the batch, not one a key
        named = f"({', '.join(columns)}) IN (SELECT {unnested})"
        first = width + len(pinned) + 1
        bounds = (
            [
                f"t.{quote_identifier(column)} = ANY({self._typed_array(first + index, index, native[index])})"
                for index, column in enumerate(self.keys.columns)
            ]
            if bounded
            else []
        )
        guards, _ = self._guards(width + len(pinned) + len(bounds))
        where = " AND ".join([named, *conditions, *bounds, *guards])
        return self._locked(f"{batched_join_hint(IN_SUBQUERY, 't')}{head} WHERE {where}")

    def _tight_values(self, arrays: tuple[list[Any], ...], keys: int) -> tuple[list[Any], ...]:
        """each key column's distinct values, bound beside the keys (``t."a" = ANY($n)``) when their cross product
        names at most :data:`_TIGHT_BOUND_FACTOR` times the keys; none otherwise.

        A batch of prefixes that share their lead (a page's 200 geographies of one state) names its rows by those
        bounds exactly: PostgreSQL reads them as the index's own condition, where the keys alone, each naming many
        rows, plan as a hash join over the whole table (300,000 rows read for 6,000). A batch of scattered keys
        (600 keys over 50 states, 200 geographies and 30 ids) names the whole table by them, so it gets none and is
        read by its keys, one lookup a key.
        """
        distinct = [list(dict.fromkeys(values)) for values in arrays]
        product = 1
        for values in distinct:
            product *= len(values)
        if product > _TIGHT_BOUND_FACTOR * keys:
            return ()
        return tuple(distinct)

    def _native(self, held: Sequence[list[Any]], natively: bool) -> tuple[bool, ...]:
        """which key columns bind their values as they are: on a connection binding typed arrays natively, each
        column of a type other than text whose values hold no str (a caller's text form stays text, cast in SQL).
        Judged by the values' distinct types, not value by value (2,000 keys: a few types)."""
        if not natively:
            return (False,) * len(self.keys.columns)
        return tuple(
            self.keys.cast(index).lower() not in _TEXT_TYPES
            and not any(issubclass(kind, str) for kind in set(map(type, values)))
            for index, values in enumerate(held)
        )

    def _typed_array(self, place: int, column: int, native: bool) -> str:
        """placeholder ``place``'s array of key column ``column``: of the column's type, bound natively, or a text
        array cast as a whole to it."""
        cast = self.keys.cast(column)
        if cast.lower() in _TEXT_TYPES:
            return f"${place}::text[]"
        return f"${place}::{cast}[]" if native else f"${place}::text[]::{cast}[]"


def _statement(
    table: str | Derived,
    columns: Sequence[str | Selected],
    keys: KeyBatch,
    by: Mapping[str, Any],
    lock: Lock | None = None,
    where: Sequence[Fragment] = (),
) -> _Statement:
    if not columns:
        raise ValueError(f"a keyed read of {table} names the columns it reads")
    selected = ", ".join(selected_column(column) for column in columns)
    return _Statement(table=table, selected=selected, keys=keys, by=by, lock=lock, where=tuple(where))


def _batches(keys: KeyBatch, batch: int) -> list[list[tuple[Any, ...]]]:
    if batch < 1:
        raise ValueError(f"a keyed read names at least one key a statement, got {batch}")
    unique = list(dict.fromkeys(map(tuple, keys.values)))
    return [unique[start : start + batch] for start in range(0, len(unique), batch)]


@dataclass
class _Fit:
    """the most keys a statement of one read has been seen to answer whole: after a cut, every later batch of the
    read is cut to it before it is sent, so one too-wide answer costs one statement, not one a batch."""

    keys: int | None = None


async def _read_split(
    fetch: Callable[[Sequence[tuple[Any, ...]]], Awaitable[Any]],
    size: Callable[[Any], int],
    keys: Sequence[tuple[Any, ...]],
    row_cap: int | None,
    table: str | Derived,
    fit: _Fit | None = None,
) -> list[Any]:
    """one batch's answers: the batch read whole, or in halves while an answer reaches ``row_cap`` or the transport
    cuts it (:class:`~threetears.core.backends.protocol.L3AnswerCutError`: its row cap, or a reply's bytes, as the
    hub's datasource replies are cut): each half is another statement, never the cut answer taken as every row."""
    fit = fit if fit is not None else _Fit()
    if fit.keys is not None and len(keys) > fit.keys:
        return [
            *await _read_split(fetch, size, keys[: fit.keys], row_cap, table, fit),
            *await _read_split(fetch, size, keys[fit.keys :], row_cap, table, fit),
        ]
    try:
        answered = await fetch(keys)
    except L3AnswerCutError:
        if len(keys) == 1:
            raise
        cut = True
    else:
        if row_cap is None or size(answered) < row_cap:
            return [answered]
        cut = False
    if len(keys) == 1:
        raise KeyTooWide(
            f"{table}: the key {keys[0]!r} alone reaches the transport's {row_cap} rows a statement; page it "
            "(threetears.core.keyed.keyset_page)"
        )
    half = len(keys) // 2
    if cut:
        fit.keys = half if fit.keys is None else min(fit.keys, half)
    return [
        *await _read_split(fetch, size, keys[:half], row_cap, table, fit),
        *await _read_split(fetch, size, keys[half:], row_cap, table, fit),
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
    statement = _statement(table, columns, keys, by or {}, lock, where)
    natively = binds_native_arrays(conn)

    async def fetch(chunk: Sequence[tuple[Any, ...]]) -> list[Any]:
        sql, params = statement.render(chunk, natively)
        return list(await conn.fetch(sql, *params))

    rows: list[Any] = []
    fit = _Fit()
    for chunk in _batches(keys, key_batch_for(conn) if batch is None else batch):
        for answered in await _read_split(fetch, len, chunk, row_cap, table, fit):
            if records:
                rows.extend(answered)
            else:
                # mapped, not a generator: a row a call less, at a page's 6,000 rows a millisecond
                rows.extend(map(dict, answered))
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
    :return: one ``pyarrow.Table`` a statement, in the order read
    :rtype: list[pyarrow.Table]
    :raises ValueError: as :func:`read_keyed`
    :raises KeyTooWide: when one key alone reaches ``row_cap``
    """
    statement = _statement(table, list(types), keys, by or {})
    natively = binds_native_arrays(conn)

    async def fetch(chunk: Sequence[tuple[Any, ...]]) -> Any:
        sql, params = statement.render(chunk, natively)
        return await l3_fetch_arrow(conn, sql, *params, types=types)

    tables: list[Any] = []
    fit = _Fit()
    for chunk in _batches(keys, key_batch_for(conn) if batch is None else batch):
        tables += await _read_split(fetch, lambda answered: int(answered.num_rows), chunk, row_cap, table, fit)
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
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
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
    :return: the page's values (in ``on``'s order) -> the joined rows holding them; a value no row holds is absent
    :rtype: dict[tuple[Any, ...], list[dict[str, Any]]]
    :raises ValueError: when ``on`` is empty or a column of it is not read
    """
    if not on:
        raise ValueError(f"a joined read of {table} names how it joins the page")
    inner = tuple(on)
    missing = [column for column in inner if column not in columns]
    if missing:
        raise ValueError(f"a joined read of {table} reads its join columns, missing {missing}")
    cast_of = dict(casts or {})
    values = [tuple(row[on[column]] for column in inner) for row in rows]
    named = [value for value in values if all(part is not None for part in value)]
    joined: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    if not named:
        return joined
    keys = KeyBatch(inner, named, tuple(cast_of.get(column, "text") for column in inner))
    # each row's join values by one itemgetter, not a tuple built a column at a time (6,000 rows: 0.6 ms against 1.8)
    getter = itemgetter(*inner)
    key_of = getter if len(inner) > 1 else (lambda row: (getter(row),))
    for row in await read_keyed(conn, table, keys=keys, columns=columns, by=by, batch=batch):
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
    for fragment in where:
        conditions.append(renumbered(fragment, len(params)))
        params += list(fragment.params)
    clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    selected = ", ".join(f"t.{quote_identifier(column)}" for column in columns)
    rows = await conn.fetch(f"SELECT {selected} FROM {relation(table, 't')}{clause} LIMIT {limit}", *params)
    return [dict(row) for row in rows]
