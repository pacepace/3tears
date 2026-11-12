"""keyed reads' statements: the rows a batch of keys (or key prefixes) names, a bounded batch a statement.

The one spelling of the keyed batch read. Each key column's values are bound as one text array, the array cast as a
whole to the column's type, so the column is compared with values of its own type and its index is read by them::

    SELECT <columns> FROM <table> AS t WHERE t."id" = ANY($1::text[]::uuid[]) AND t."tenant" = $2

    /*+ YbBatchedNL(ANY_subquery t) */ SELECT <columns> FROM <table> AS t
     WHERE (t."a", t."b") IN (SELECT unnest($1::text[]) AS k0, unnest($2::text[]::bigint[]) AS k1)

or, for a database with no arrays (Redshift, SQLite), as a list of placeholders (``KeyBatch(..., as_list=True)``).
Pure: :func:`keyed_read_statement` renders a batch's text and binds, :func:`key_chunks` cuts a batch, and
:func:`read_keyed_statements` writes every statement a read sends; ``threetears.core.keyed.read_keyed`` runs them
(with its transport's row cap), and a reader with no 3tears core runs them on its own connection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from operator import itemgetter
from typing import Any, Final

from threetears.interfaces.keyed.filters import (
    IN_SUBQUERY,
    Derived,
    Fragment,
    Lock,
    Selected,
    batched_join_hint,
    by_conditions,
    narrowing,
    quote_identifier,
    relation,
    selected_column,
)

__all__ = [
    "ANY_ROWS_CAP",
    "DEFAULT_KEY_BATCH",
    "DIRECT_KEY_BATCH",
    "KeyBatch",
    "KeyTooWide",
    "key_chunks",
    "keyed_read_statement",
    "read_keyed_statements",
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


#: a batch's key columns are each bound to their distinct values when those name at most this many times its keys
_TIGHT_BOUND_FACTOR: Final = 2

#: the key column types bound as text arrays, with no cast
_TEXT_TYPES: Final = frozenset({"text", "varchar"})

#: the most statement shapes whose text is kept rendered; past it the oldest is let go
_RENDERED_SHAPES: Final = 4096

#: an array statement's text by its shape (:attr:`_KeyedRead._shape`, which columns bind natively, whether bounded):
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
    :param as_text: with ``as_list``, each key column's text as a TRUSTED SQL expression over ``t`` (its canonical
        text: ``CAST(t."n" AS VARCHAR)``); the key is then compared as ONE text, each column's text written as its
        length, ``:`` and the text, against one bind a key, ``values`` giving each key's column texts. One scalar
        ``IN`` of texts, where Redshift plans a list of tuples key by key (528 four-column keys: 4.4 s of planning
        against 0.11 s). The length marks keep it exact: no two keys of different texts join alike. A read's only;
        a NULL part is no text (pin it ``IS NULL`` in ``by``). Empty: the columns are compared as they are
    :ptype as_text: tuple[str, ...]
    """

    columns: tuple[str, ...]
    values: Sequence[Sequence[Any]]
    casts: tuple[str, ...] = field(default=())
    null_safe: bool = False
    as_list: bool = False
    as_text: tuple[str, ...] = field(default=())

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
        if self.as_text:
            if not self.as_list:
                raise ValueError("a key batch spelled as text is bound as a list of texts (as_list)")
            if len(self.as_text) != len(self.columns):
                raise ValueError(
                    f"a key batch spelled as text names the text of each of its {len(self.columns)} columns, "
                    f"got {len(self.as_text)}"
                )
            for key in self.values:
                if not all(isinstance(part, str) for part in key):
                    raise ValueError(f"a key batch spelled as text gives each key's column texts, got {tuple(key)!r}")

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
class _KeyedRead:
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
            conditions.append(narrowing(fragment, before + len(params)))
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
            if self.keys.as_text:
                return self._text_list(head, chunk)
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

    def _text_list(self, head: str, chunk: Sequence[tuple[Any, ...]]) -> tuple[str, tuple[Any, ...]]:
        """a list batch spelled as text (:attr:`KeyBatch.as_text`): the key's length-marked text in one scalar ``IN``
        of one bind a key."""
        spelled = " || ".join(f"CAST(LENGTH({text}) AS VARCHAR) || ':' || {text}" for text in self.keys.as_text)
        named = f"({spelled}) IN ({', '.join(f'${place + 1}' for place in range(len(chunk)))})"
        conditions, pinned = by_conditions(self.by, first=len(chunk) + 1, alias="t")
        guards, guarded = self._guards(len(chunk) + len(pinned))
        where = " AND ".join([named, *conditions, *guards])
        texts = ("".join(f"{len(part)}:{part}" for part in key) for key in chunk)
        return self._locked(f"{head} WHERE {where}"), (*texts, *pinned, *guarded)

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


def keyed_read_statement(
    table: str | Derived,
    columns: Sequence[str | Selected],
    keys: KeyBatch,
    by: Mapping[str, Any],
    lock: Lock | None = None,
    where: Sequence[Fragment] = (),
) -> _KeyedRead:
    """the statement a keyed read of ``columns`` sends each batch of ``keys``: its ``render(chunk, natively)`` writes
    a batch's text and binds (``threetears.core.keyed.read_keyed`` renders each batch as it reads it, halving one its
    transport cut).

    :param table: the table, a TRUSTED identifier, or a derived relation
    :ptype table: str | Derived
    :param columns: the columns read, TRUSTED identifiers or :class:`Selected` expressions
    :ptype columns: Sequence[str | Selected]
    :param keys: the keys' shape (the batch itself is rendered a chunk at a time)
    :ptype keys: KeyBatch
    :param by: columns pinned beside the keys
    :ptype by: Mapping[str, Any]
    :param lock: the row lock taken, in the key's order; none when None
    :ptype lock: Lock | None
    :param where: filters beside the key, each a fragment written as ``t."col"`` from ``$1``
    :ptype where: Sequence[Fragment]
    :return: the statement
    :rtype: _KeyedRead
    :raises ValueError: when no column is named
    """
    if not columns:
        raise ValueError(f"a keyed read of {table} names the columns it reads")
    selected = ", ".join(selected_column(column) for column in columns)
    return _KeyedRead(table=table, selected=selected, keys=keys, by=by, lock=lock, where=tuple(where))


def key_chunks(keys: KeyBatch, batch: int) -> list[list[tuple[Any, ...]]]:
    """``keys``' distinct keys, in the order first named, ``batch`` of them a chunk: one statement each.

    :param keys: the keys
    :ptype keys: KeyBatch
    :param batch: the most keys a statement names
    :ptype batch: int
    :return: the chunks
    :rtype: list[list[tuple[Any, ...]]]
    :raises ValueError: when ``batch`` is under one
    """
    if batch < 1:
        raise ValueError(f"a keyed read names at least one key a statement, got {batch}")
    unique = list(dict.fromkeys(map(tuple, keys.values)))
    return [unique[start : start + batch] for start in range(0, len(unique), batch)]


def read_keyed_statements(
    table: str | Derived,
    *,
    keys: KeyBatch,
    columns: Sequence[str | Selected],
    by: Mapping[str, Any] | None = None,
    batch: int = DEFAULT_KEY_BATCH,
    natively: bool = False,
    lock: Lock | None = None,
    where: Sequence[Fragment] = (),
) -> list[tuple[str, tuple[Any, ...]]]:
    """every statement a keyed read of ``table`` by ``keys`` (within ``by``) sends, ``batch`` keys each, and its
    binds: what ``threetears.core.keyed.read_keyed`` sends a transport that never cuts an answer.

    :param table: the table, a TRUSTED identifier, or a derived relation
    :ptype table: str | Derived
    :param keys: the keys, a leading part of the table's key (or of an index declared for the caller's read)
    :ptype keys: KeyBatch
    :param columns: the columns read, TRUSTED identifiers or :class:`Selected` expressions
    :ptype columns: Sequence[str | Selected]
    :param by: columns pinned beside the keys, each a value, None or an ``AnyOf``
    :ptype by: Mapping[str, Any] | None
    :param batch: the most keys a statement names
    :ptype batch: int
    :param natively: whether the connection binds typed arrays natively (asyncpg's own)
    :ptype natively: bool
    :param lock: the row lock taken, in the key's order; none when None
    :ptype lock: Lock | None
    :param where: filters beside the key, each a fragment written as ``t."col"`` from ``$1``
    :ptype where: Sequence[Fragment]
    :return: each statement and its binds, a chunk of keys each; none for no keys
    :rtype: list[tuple[str, tuple[Any, ...]]]
    :raises ValueError: when no column is named, ``batch`` is under one, or a key does not match the columns
    """
    statement = keyed_read_statement(table, columns, keys, by or {}, lock, where)
    return [statement.render(chunk, natively) for chunk in key_chunks(keys, batch)]
