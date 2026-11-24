"""keyed reads' statements: the rows a batch of keys (or key prefixes) names, a bounded batch a statement.

The one spelling of the keyed batch read. Each key column's values are bound as ONE array (a plain column's list, or
an Arrow column's literal text written in Arrow's kernels: ``$1::text::uuid[]``), the array cast as a whole to the
column's type, so the column is compared with values of its own type and its index is read by them::

    SELECT <columns> FROM <table> AS t WHERE t."id" = ANY($1::text[]::uuid[]) AND t."tenant" = $2

    /*+ YbBatchedNL(ANY_subquery t) */ SELECT <columns> FROM <table> AS t
     WHERE (t."a", t."b") IN (SELECT unnest($1::text[]) AS k0, unnest($2::text[]::bigint[]) AS k1)

or, for a database with no arrays (Redshift, SQLite), as a list of placeholders (``KeyBatch(..., as_list=True)``).
Pure: :func:`keyed_read_statement` renders a batch's text and binds, :func:`key_chunks` cuts a batch, and
:func:`read_keyed_statements` writes every statement a read sends; ``threetears.core.keyed.read_keyed`` runs them
(with its transport's row cap), and a reader with no 3tears core runs them on its own connection.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Annotated, Any, Final, Literal, Protocol, TypeAlias

from threetears.interfaces.keyed.filters import (
    IN_SUBQUERY,
    Derived,
    Fragment,
    Lock,
    Selected,
    batched_join_hint,
    by_conditions,
    each_placeholder,
    narrowing,
    quote_identifier,
    relation,
    selected_column,
)

__all__ = [
    "ANY_ROWS_CAP",
    "DEFAULT_KEY_BATCH",
    "DIRECT_KEY_BATCH",
    "PLAIN_KEYS",
    "BindKind",
    "SplitText",
    "KeyBatch",
    "KeyBinder",
    "KeyChunk",
    "KeyTooWide",
    "NullGroup",
    "key_chunks",
    "keyed_read_statement",
    "null_pattern_code",
    "null_split",
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


class DeclaredKey:
    """the mark of a declared key source, in an annotation: ``Annotated[Sequence[str], DeclaredKey]`` for a key's
    columns held in another shape than :data:`KeyColumns`. A parameter, attribute or function annotated with it (or
    with :data:`KeyColumn` / :data:`KeyColumns` / :data:`KeyValues`) holds a table's key as its declaration names it:
    the no_whole_reads rule reads a column interpolated from one (or from a local written from one) as the key, and any
    other column as a predicate it cannot see, whatever the column's name says."""


class DeclaredScope:
    """the mark of a declared state column, in an annotation: ``Annotated[str, DeclaredScope]`` for the row column a
    table's own declaration names as the state its rows belong to (``TableSchema.delta_scope``, a followed table's
    ``scope``), set only from that declaration. The no_whole_reads rule reads a local copy's read pinned by a column
    from one (``threetears.core.keyed.read_state_arrow``, never L3) as bounded by the state, and any other column as a
    predicate it cannot see."""


#: a table's key column, as its declaration names it: a declared key source (:class:`DeclaredKey`)
KeyColumn: TypeAlias = Annotated[str, DeclaredKey]

#: a table's key columns, in key order, as its declaration names them: a declared key source (:class:`DeclaredKey`)
KeyColumns: TypeAlias = Annotated[tuple[str, ...], DeclaredKey]

#: one key's values by its columns (column -> value): a declared key source (:class:`DeclaredKey`)
KeyValues: TypeAlias = Annotated[Mapping[str, Any], DeclaredKey]


def declared_key(statement: str, *, key: Sequence[str]) -> str:
    """a statement over a relation named at run time, with the key its reads are bound by declared where it is
    written: ``declared_key(f"SELECT ... FROM {q(layer.table)} WHERE {q(layer.version_column)} = $1 ...",
    key=(layer.version_column,))``. The no_whole_reads rule reads each interpolation of a ``key`` expression in the
    statement (itself, or quoted by a one-argument call) as the relation's key, as it reads a :class:`DeclaredKey`
    source, and judges the read by it: a read pinning the key is keyed; one that does not is the finding it is. It
    declares a key, never a whole read; the relation's owner holds the index its key leads (and its plan test).

    :param statement: the statement, returned unchanged
    :ptype statement: str
    :param key: the key's columns, as the statement interpolates them
    :ptype key: Sequence[str]
    :return: ``statement``
    :rtype: str
    """
    del key
    return statement


class KeyTooWide(ValueError):
    """one key alone holds as many rows as the transport answers a statement: it is a prefix to page through."""


@dataclass(frozen=True)
class SplitText:
    """a key column bound as ONE text, its values' texts joined by ``delimiter`` (a code point none of them holds),
    split back in SQL: ``string_to_array($1::text, chr(31))::uuid[]``. ``string_to_array`` and ``chr`` are immutable,
    so a custom plan folds the array into a constant, as it does a bound list (``$1::text[]::uuid[]``): the planner
    sees the same array and plans the same; an array literal's parse (``array_in``) is stable, evaluated again at each
    estimate and never folded (2,000 UUIDs: 1.2-2.3 ms of planning against 0.3-0.7, measured 2026-11-23).

    :param delimiter: the code point between two values
    :ptype delimiter: int
    :param null: the code point a null value is written as (none of the values holds it); None when none is null
    :ptype null: int | None
    """

    delimiter: int
    null: int | None = None


#: how one key column's values are bound (:meth:`KeyBinder.array`): as they are, of the column's type (asyncpg's own
#: typed arrays); as a list of texts the statement casts (``$n::text[]::<type>[]``); as ONE text split back in SQL
#: (:class:`SplitText`: an Arrow column's texts joined in Arrow's kernels, so no value of it is ever a Python object on
#: the way in); or as ONE text, the array's literal (``$n::text::<type>[]``: a database with no ``string_to_array``,
#: DuckDB behind a reader, or the rare column no delimiter fits)
BindKind: TypeAlias = Literal["native", "text", "literal"] | SplitText

#: one statement's keys: one sequence a key column, in key order, all of one length (a plain list's slice, or an Arrow
#: array's zero-copy slice)
KeyChunk: TypeAlias = tuple[Sequence[Any], ...]


@dataclass(frozen=True)
class KeyBatch:
    """the keys a read names: a leading part of a key, one array a key column (column-major, never one tuple a key).

    Each of ``keys`` is one key column's values, in key order: a ``pyarrow`` array (a serving path's keys, bound as one
    text written in Arrow's kernels, ``threetears.core.keyed.binding``) or a plain sequence (a small key list program
    logic holds, or any key list in a process with no Arrow: its values bound as one list). The i-th key is each
    column's i-th value.

    :param columns: the key columns, TRUSTED identifiers, in key order
    :ptype columns: tuple[str, ...]
    :param keys: one array a key column, in ``columns``' order, all of one length; each key is read once however
        often named
    :ptype keys: Sequence[pyarrow.Array | Sequence[Any]]
    :param casts: each column's SQL type its text is cast back to; ``text`` for every column when empty
    :ptype casts: tuple[str, ...]
    :param null_safe: a ``None`` in a key matches the rows holding none in that column (as ``IS NOT DISTINCT FROM``
        would): a scope column's NULL bucket. Sent split by which parts are NULL (:func:`null_split`), each NULL part
        pinned ``IS NULL`` beside equality on the rest, so the key's index reads every key
    :ptype null_safe: bool
    :param as_list: bind the keys as a list of placeholders (``(c1, c2) IN (($1, $2), ...)``), each value as it is,
        for a database with no arrays (Redshift, SQLite); a statement's text then follows its batch's size. Its
        driver binds a placeholder a value, so each value is a Python object at the bind (an Arrow column's too)
    :ptype as_list: bool
    :param as_text: with ``as_list``, each key column's text as a TRUSTED SQL expression over ``t`` (its canonical
        text: ``CAST(t."n" AS VARCHAR)``); the key is then compared as ONE text, each column's text written as its
        length, ``:`` and the text, against one bind a key, ``keys`` giving each column's texts. One scalar
        ``IN`` of texts, where Redshift plans a list of tuples key by key (528 four-column keys: 4.4 s of planning
        against 0.11 s). The length marks keep it exact: no two keys of different texts join alike. A read's only;
        a NULL part is no text (pin it ``IS NULL`` in ``by``). Empty: the columns are compared as they are
    :ptype as_text: tuple[str, ...]
    """

    columns: tuple[str, ...]
    keys: Sequence[Any]
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
        if len(self.keys) != len(self.columns):
            raise ValueError(
                f"a key batch of {self.columns} holds one array a column ({len(self.columns)}), got {len(self.keys)}"
            )
        lengths = {len(column) for column in self.keys}
        if len(lengths) > 1:
            raise ValueError(f"a key batch of {self.columns} holds columns of one length, got {sorted(lengths)}")
        if self.as_text:
            if not self.as_list:
                raise ValueError("a key batch spelled as text is bound as a list of texts (as_list)")
            if len(self.as_text) != len(self.columns):
                raise ValueError(
                    f"a key batch spelled as text names the text of each of its {len(self.columns)} columns, "
                    f"got {len(self.as_text)}"
                )
            for column in self.keys:
                kind = getattr(column, "type", None)
                texts = str(kind) in _ARROW_TEXT_TYPES if kind is not None else all(isinstance(v, str) for v in column)
                if not texts:
                    raise ValueError(f"a key batch spelled as text gives each column's texts, got {kind or column!r}")

    def __len__(self) -> int:
        """the keys named, a repeat counted each time it is named."""
        return len(self.keys[0])

    def cast(self, index: int) -> str:
        """the SQL type column ``index``'s text is cast back to.

        :param index: the column's place in :attr:`columns`
        :ptype index: int
        :return: the type
        :rtype: str
        """
        return self.casts[index] if self.casts else "text"


#: the Arrow types whose values are already the texts a text-spelled batch compares
_ARROW_TEXT_TYPES: Final = frozenset({"string", "large_string", "string_view"})


class KeyBinder(Protocol):
    """how a batch's key columns become a statement's binds: :data:`PLAIN_KEYS` for plain sequences (here, with no
    third-party import), ``threetears.core.keyed``'s Arrow binder for Arrow columns (its kernels, never a Python
    object a key)."""

    def distinct(self, keys: KeyBatch) -> KeyChunk:
        """the batch's distinct keys, in the order first named, as columns."""
        ...

    def array(self, values: Sequence[Any], cast: str, natively: bool) -> tuple[Any, BindKind]:
        """one key column's values as ONE bind, and how the statement casts it."""
        ...

    def distinct_values(self, values: Sequence[Any]) -> Sequence[Any]:
        """one key column's distinct values."""
        ...

    def listed(self, values: Sequence[Any]) -> list[Any]:
        """one key column's values as Python values, a placeholder each (``as_list``: the driver binds them so)."""
        ...

    def null_groups(self, chunk: KeyChunk) -> list[tuple[int, KeyChunk]]:
        """``chunk``'s keys grouped by which of their parts are NULL (:func:`null_pattern_code`), each group its code
        and its keys (every column; a NULL part's column all None), in no promised order."""
        ...


def null_pattern_code(nulls: Sequence[bool]) -> int:
    """which parts of a key are NULL, as one number: bit ``i`` set when part ``i`` is (``0``: no part NULL).

    :param nulls: each part's NULL-ness, in key order
    :ptype nulls: Sequence[bool]
    :return: the code
    :rtype: int
    """
    return sum(1 << index for index, null in enumerate(nulls) if null)


def _text(value: Any) -> str | None:
    """a key value as the text its array binds: its ISO form for a date or time, ``str`` otherwise."""
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    return str(isoformat()) if callable(isoformat) else str(value)


class _PlainKeys:
    """:class:`KeyBinder` of plain sequences: a batch of plain columns, or any in a process with no Arrow. Each column
    bound as one list, of its values where the connection binds the type natively, else of their texts (a column of
    str bound as it is)."""

    def distinct(self, keys: KeyBatch) -> KeyChunk:
        columns = tuple(self._plain(column) for column in keys.keys)
        if len(columns) == 1:
            return (list(dict.fromkeys(columns[0])),)
        unique = list(dict.fromkeys(zip(*columns, strict=True)))
        if not unique:
            return tuple([] for _ in columns)
        return tuple(map(list, zip(*unique, strict=True)))

    def array(self, values: Sequence[Any], cast: str, natively: bool) -> tuple[Any, BindKind]:
        # judged by the values' distinct types, not value by value (2,000 keys: a few types)
        kinds = set(map(type, values))
        if natively and cast.lower() not in _TEXT_TYPES and not any(issubclass(kind, str) for kind in kinds):
            return list(values), "native"
        if kinds <= {str, type(None)}:
            return list(values), "text"
        return [_text(value) for value in values], "text"

    def distinct_values(self, values: Sequence[Any]) -> Sequence[Any]:
        return list(dict.fromkeys(values))

    def listed(self, values: Sequence[Any]) -> list[Any]:
        return list(self._plain(values))

    def null_groups(self, chunk: KeyChunk) -> list[tuple[int, KeyChunk]]:
        columns = [self._plain(column) for column in chunk]
        if not any(value is None for column in columns for value in column):
            return [(0, chunk)]
        groups: dict[int, list[tuple[Any, ...]]] = {}
        for key in zip(*columns, strict=True):
            groups.setdefault(null_pattern_code([value is None for value in key]), []).append(key)
        return [(code, tuple(map(list, zip(*keys, strict=True)))) for code, keys in groups.items()]

    @staticmethod
    def _plain(values: Sequence[Any]) -> Sequence[Any]:
        if hasattr(values, "to_pylist"):
            raise TypeError(
                "an Arrow key column binds through threetears.core.keyed (its Arrow binder), never value by value here"
            )
        return values


#: the binder of plain sequences, the one a pure statement builder uses when given none
PLAIN_KEYS: Final[KeyBinder] = _PlainKeys()


@dataclass(frozen=True)
class NullGroup:
    """one NULL pattern's keys of a null-safe batch (:func:`null_split`).

    :param nulled: the key columns its keys hold NULL in, each pinned ``IS NULL``
    :ptype nulled: tuple[str, ...]
    :param keys: its keys' other columns, a batch named by equality as any is (its own chunk: distinct, one
        statement's); None when every part is NULL
    :ptype keys: KeyBatch | None
    """

    nulled: tuple[str, ...]
    keys: KeyBatch | None


def null_split(keys: KeyBatch, chunk: KeyChunk, binder: KeyBinder = PLAIN_KEYS) -> list[NullGroup]:
    """a null-safe batch's chunk as the statements against PostgreSQL and YugabyteDB name it: grouped by which parts
    of each key are NULL, the keys with none first, then each pattern in :func:`null_pattern_code` order (the order a
    locked read takes the patterns' locks, alike in every reader). Each pattern's NULL parts are pinned ``IS NULL``
    and its keys' other parts named by equality, so the index reads every pattern by its own condition; together
    they name exactly the rows ``IS NOT DISTINCT FROM`` each key would. The one split of a null-safe batch: keyed
    reads, keyed writes and the changed-rows read's warehouse keys (``threetears.core.keyed.changed``).

    :param keys: the batch's shape (its columns and casts)
    :ptype keys: KeyBatch
    :param chunk: one statement's keys, one sequence a key column, every column of the batch
    :ptype chunk: KeyChunk
    :param binder: how the key columns are read (:data:`PLAIN_KEYS`, or ``threetears.core.keyed``'s Arrow binder)
    :ptype binder: KeyBinder
    :return: the patterns, each its pinned columns and its batch of the rest
    :rtype: list[NullGroup]
    """
    width = len(keys.columns)
    split: list[NullGroup] = []
    for code, grouped in sorted(binder.null_groups(chunk), key=lambda group: group[0]):
        present = [index for index in range(width) if not code >> index & 1]
        nulled = tuple(keys.columns[index] for index in range(width) if code >> index & 1)
        batch = (
            KeyBatch(
                columns=tuple(keys.columns[index] for index in present),
                keys=[grouped[index] for index in present],
                casts=tuple(keys.cast(index) for index in present) if keys.casts else (),
            )
            if present
            else None
        )
        split.append(NullGroup(nulled, batch))
    return split


def _united(branches: Sequence[tuple[str, tuple[Any, ...]]], *, locked: bool) -> tuple[str, tuple[Any, ...]]:
    """several reads as one statement: each renumbered past the binds before it, joined by ``UNION ALL`` (a locked
    read's each a ``WITH`` query, read in turn: a union takes no lock), the join hint they carry said once, first
    (YugabyteDB applies it to every branch's join).

    :param branches: each read's text and binds, in the order read
    :ptype branches: Sequence[tuple[str, tuple[Any, ...]]]
    :param locked: whether the reads lock their rows
    :ptype locked: bool
    :return: the statement and its binds
    :rtype: tuple[str, tuple[Any, ...]]
    """
    if len(branches) == 1:
        return branches[0]
    hint = ""
    texts: list[str] = []
    params: list[Any] = []
    for sql, bound in branches:
        text = sql
        if text.startswith(_HINT_OPEN):
            end = text.index(_HINT_CLOSE) + len(_HINT_CLOSE)
            hint, text = text[:end], text[end:]
        texts.append(each_placeholder(text, _past(len(params))))
        params += bound
    if locked:
        named = ", ".join(f"b{index} AS ({text})" for index, text in enumerate(texts))
        read = " UNION ALL ".join(f"SELECT * FROM b{index}" for index in range(len(texts)))
        return f"{hint}WITH {named} {read}", tuple(params)
    return hint + " UNION ALL ".join(texts), tuple(params)


def _past(before: int) -> Callable[[int], str]:
    """a placeholder's number moved past the ``before`` parameters bound ahead of it."""

    def moved(number: int) -> str:
        return f"${number + before}"

    return moved


#: how a statement's leading hint opens and closes (:func:`batched_join_hint`)
_HINT_OPEN: Final = "/*+ "
_HINT_CLOSE: Final = "*/ "


def _typed_array(place: int, cast: str, kind: BindKind, once: bool = False) -> str:
    """placeholder ``place``'s array of a key column of SQL type ``cast``, bound as ``kind``: of the column's type
    natively, a text array cast as a whole to it, or one text (the array's literal) cast to it.

    One text's array is made from it in SQL, so where a generic plan may evaluate it a row at a time (a bound beside
    the keys, on the inner side of a nested loop: 10,000 keys split or parsed 10,000 times, 84-97 ms against 41 for a
    bound list on PostgreSQL 16, measured 2026-11-23) it is made ``once``: a scalar subquery, an InitPlan run once a
    statement (35 ms).

    :param place: the placeholder's number
    :ptype place: int
    :param cast: the column's SQL type
    :ptype cast: str
    :param kind: how its values are bound
    :ptype kind: BindKind
    :param once: one text's array made once a statement (an InitPlan), not where it stands
    :ptype once: bool
    :return: the placeholder's SQL
    :rtype: str
    """
    textual = cast.lower() in _TEXT_TYPES
    array = "text[]" if textual else f"{cast}[]"
    if isinstance(kind, SplitText):
        null = "" if kind.null is None else f", chr({kind.null})"
        split = f"string_to_array(${place}::text, chr({kind.delimiter}){null})"
        typed = split if textual else f"{split}::{array}"
        return f"(SELECT {typed})::{array}" if once else typed
    if kind == "literal":
        return f"(SELECT ${place}::text::{array})::{array}" if once else f"${place}::text::{array}"
    if textual:
        return f"${place}::text[]"
    return f"${place}::{cast}[]" if kind == "native" else f"${place}::text[]::{cast}[]"


@dataclass(frozen=True)
class _KeyedRead:
    """the statement one keyed read sends a batch: built once for arrays, once a batch for a list of keys."""

    table: str | Derived
    selected: str
    keys: KeyBatch
    by: Mapping[str, Any]
    lock: Lock | None = None
    where: Sequence[Fragment] = ()
    binder: KeyBinder = PLAIN_KEYS

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

    def render(self, chunk: KeyChunk, natively: bool = False) -> tuple[str, tuple[Any, ...]]:
        """the statement naming ``chunk``'s keys, and its parameters.

        :param chunk: the keys, one sequence a key column (:func:`key_chunks`)
        :ptype chunk: KeyChunk
        :param natively: whether the connection binds typed arrays natively (:func:`binds_native_arrays`): a plain
            column whose values are of its type (no str) is then bound as ``$n::<type>[]`` from the values themselves
        :ptype natively: bool
        :return: the statement, and its parameters
        :rtype: tuple[str, tuple[Any, ...]]
        """
        if self.keys.null_safe:
            return self._null_split(chunk, natively)
        width = len(self.keys.columns)
        count = len(chunk[0])
        if self.keys.as_list:
            head = f"SELECT {self.selected} FROM {relation(self.table, 't')}"
            columns = [f"t.{quote_identifier(column)}" for column in self.keys.columns]
            listed = [self.binder.listed(column) for column in chunk]
            if self.keys.as_text:
                return self._text_list(head, listed, count)
            if width == 1:
                named = f"{columns[0]} IN ({', '.join(f'${place + 1}' for place in range(count))})"
            else:
                rows = ", ".join(
                    "(" + ", ".join(f"${row * width + index + 1}" for index in range(width)) + ")"
                    for row in range(count)
                )
                named = f"({', '.join(columns)}) IN ({rows})"
            conditions, pinned = by_conditions(self.by, first=count * width + 1, alias="t")
            guards, guarded = self._guards(count * width + len(pinned))
            where = " AND ".join([named, *conditions, *guards])
            values = listed[0] if width == 1 else [value for key in zip(*listed, strict=True) for value in key]
            return self._locked(f"{head} WHERE {where}"), (*values, *pinned, *guarded)
        # each key column's values bound as ONE array: of the column's type where the connection binds it natively,
        # a text array, or the array's literal text, each cast as a whole to the column's type. The column is compared
        # with values of its own type, never wrapped or compared element by element through a cast, so the planner
        # reads the key's index by them (an Index Cond), as a hand-written "col = ANY($1::uuid[])" would
        bound = [self.binder.array(values, self.keys.cast(index), natively) for index, values in enumerate(chunk)]
        arrays = tuple(value for value, _ in bound)
        kinds = tuple(kind for _, kind in bound)
        tight: tuple[tuple[Any, BindKind], ...] = ()
        if width > 1:
            tight = self._tight_values(chunk, count, natively)
        guarded = [param for fragment in self.where for param in fragment.params]
        # the statement's text follows only its shape (how each column binds, whether bounded): rendered once a
        # shape, here or by an earlier read of the same shape, and each batch binds its own arrays to it
        tight_kinds = tuple(kind for _, kind in tight)
        shape = (self._shape, kinds, tight_kinds)
        sql = _RENDERED.get(shape)
        if sql is None:
            sql = self._array_text(kinds, tight_kinds)
            _remember(shape, sql)
        return sql, (*arrays, *self._by[1], *(value for value, _ in tight), *guarded)

    def _null_split(self, chunk: KeyChunk, natively: bool) -> tuple[str, tuple[Any, ...]]:
        """a null-safe batch's statement: its keys a NULL pattern at a time (:func:`null_split`), each pattern's NULL
        parts pinned ``IS NULL`` and the rest named by equality as any batch's keys are, the patterns' reads one
        statement (``UNION ALL``; each locked one a ``WITH`` query of its own, as a union takes no lock).

        Never ``IS NOT DISTINCT FROM`` a joined key: no index reads by it, so PostgreSQL and YugabyteDB read the
        table whole once a key (1,000 keys against 200,000 rows: 3.9 s on PostgreSQL, 5.3 s on YugabyteDB; split, 3-13 ms); pinned ``IS NULL`` beside the rest, each
        pattern is its index's own condition (test_keyed_null_safe_live).
        """
        branches: list[tuple[str, tuple[Any, ...]]] = []
        for group in null_split(self.keys, chunk, self.binder):
            by = {**self.by, **dict.fromkeys(group.nulled)}
            if group.keys is None:
                # every part NULL: the pins alone name the key's rows
                conditions, pinned = by_conditions(by, alias="t")
                guards, guarded = self._guards(len(pinned))
                where = " AND ".join([*conditions, *guards])
                head = f"SELECT {self.selected} FROM {relation(self.table, 't')}"
                branches.append((self._locked(f"{head} WHERE {where}"), (*pinned, *guarded)))
                continue
            branches.append(replace(self, keys=group.keys, by=by).render(tuple(group.keys.keys), natively))
        return _united(branches, locked=self.lock is not None)

    def _text_list(self, head: str, listed: Sequence[list[Any]], count: int) -> tuple[str, tuple[Any, ...]]:
        """a list batch spelled as text (:attr:`KeyBatch.as_text`): the key's length-marked text in one scalar ``IN``
        of one bind a key."""
        spelled = " || ".join(f"CAST(LENGTH({text}) AS VARCHAR) || ':' || {text}" for text in self.keys.as_text)
        named = f"({spelled}) IN ({', '.join(f'${place + 1}' for place in range(count))})"
        conditions, pinned = by_conditions(self.by, first=count + 1, alias="t")
        guards, guarded = self._guards(count + len(pinned))
        where = " AND ".join([named, *conditions, *guards])
        texts = ("".join(f"{len(part)}:{part}" for part in key) for key in zip(*listed, strict=True))
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

    def _array_text(self, kinds: tuple[BindKind, ...], tight: tuple[BindKind, ...]) -> str:
        """the array statement's text: ``= ANY`` for one key column, else the keys joined as rows by
        ``IN (SELECT unnest ...)``, each key column bounded to its distinct values when ``tight`` names their binds."""
        width = len(self.keys.columns)
        head = f"SELECT {self.selected} FROM {relation(self.table, 't')}"
        typed = [_typed_array(index + 1, self.keys.cast(index), kinds[index]) for index in range(width)]
        conditions, pinned = self._by
        columns = [f"t.{quote_identifier(column)}" for column in self.keys.columns]
        if width == 1:
            guards, _ = self._guards(width + len(pinned))
            where = " AND ".join([f"{columns[0]} = ANY({typed[0]})", *conditions, *guards])
            return self._locked(f"{head} WHERE {where}")
        unnested = ", ".join(f"unnest({array}) AS k{index}" for index, array in enumerate(typed))
        # the keys joined to the table as rows: a nested loop over the key's index, each key one lookup. Never a
        # column = ANY its values beside it: those name the values' cross product (600 keys over 50 states, 200
        # geographies and 30 ids name the whole table), and the planner reads it whole. YugabyteDB is told to batch
        # the loop (batched_join_hint): one storage request for the batch, not one a key
        named = f"({', '.join(columns)}) IN (SELECT {unnested})"
        first = width + len(pinned) + 1
        bounds = (
            [
                f"t.{quote_identifier(column)} = "
                f"ANY({_typed_array(first + index, self.keys.cast(index), tight[index], once=True)})"
                for index, column in enumerate(self.keys.columns)
            ]
            if tight
            else []
        )
        guards, _ = self._guards(width + len(pinned) + len(bounds))
        where = " AND ".join([named, *conditions, *bounds, *guards])
        return self._locked(f"{batched_join_hint(IN_SUBQUERY, 't')}{head} WHERE {where}")

    def _tight_values(self, chunk: KeyChunk, keys: int, natively: bool) -> tuple[tuple[Any, BindKind], ...]:
        """each key column's distinct values, bound beside the keys (``t."a" = ANY($n)``) when their cross product
        names at most :data:`_TIGHT_BOUND_FACTOR` times the keys; none otherwise.

        A batch of prefixes that share their lead (a page's 200 geographies of one state) names its rows by those
        bounds exactly: PostgreSQL reads them as the index's own condition, where the keys alone, each naming many
        rows, plan as a hash join over the whole table (300,000 rows read for 6,000). A batch of scattered keys
        (600 keys over 50 states, 200 geographies and 30 ids) names the whole table by them, so it gets none and is
        read by its keys, one lookup a key.
        """
        distinct = [self.binder.distinct_values(values) for values in chunk]
        product = 1
        for values in distinct:
            product *= len(values)
        if product > _TIGHT_BOUND_FACTOR * keys:
            return ()
        return tuple(
            self.binder.array(values, self.keys.cast(index), natively) for index, values in enumerate(distinct)
        )


def keyed_read_statement(
    table: str | Derived,
    columns: Sequence[str | Selected],
    keys: KeyBatch,
    by: Mapping[str, Any],
    lock: Lock | None = None,
    where: Sequence[Fragment] = (),
    binder: KeyBinder = PLAIN_KEYS,
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
    :param binder: how the key columns bind (:data:`PLAIN_KEYS`, or ``threetears.core.keyed``'s Arrow binder)
    :ptype binder: KeyBinder
    :return: the statement
    :rtype: _KeyedRead
    :raises ValueError: when no column is named
    """
    if not columns:
        raise ValueError(f"a keyed read of {table} names the columns it reads")
    selected = ", ".join(selected_column(column) for column in columns)
    return _KeyedRead(table=table, selected=selected, keys=keys, by=by, lock=lock, where=tuple(where), binder=binder)


def key_chunks(keys: KeyBatch, batch: int, binder: KeyBinder = PLAIN_KEYS) -> list[KeyChunk]:
    """``keys``' distinct keys, in the order first named, ``batch`` of them a chunk (one sequence a key column): one
    statement each.

    :param keys: the keys
    :ptype keys: KeyBatch
    :param batch: the most keys a statement names
    :ptype batch: int
    :param binder: how the key columns are read (:data:`PLAIN_KEYS`, or ``threetears.core.keyed``'s Arrow binder)
    :ptype binder: KeyBinder
    :return: the chunks
    :rtype: list[KeyChunk]
    :raises ValueError: when ``batch`` is under one
    """
    if batch < 1:
        raise ValueError(f"a keyed read names at least one key a statement, got {batch}")
    unique = binder.distinct(keys)
    total = len(unique[0])
    return [tuple(column[start : start + batch] for column in unique) for start in range(0, total, batch)]


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
    binder: KeyBinder = PLAIN_KEYS,
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
    :param binder: how the key columns bind (:data:`PLAIN_KEYS`, or ``threetears.core.keyed``'s Arrow binder)
    :ptype binder: KeyBinder
    :return: each statement and its binds, a chunk of keys each; none for no keys
    :rtype: list[tuple[str, tuple[Any, ...]]]
    :raises ValueError: when no column is named, ``batch`` is under one, or a key does not match the columns
    """
    statement = keyed_read_statement(table, columns, keys, by or {}, lock, where, binder)
    return [statement.render(chunk, natively) for chunk in key_chunks(keys, batch, binder)]
