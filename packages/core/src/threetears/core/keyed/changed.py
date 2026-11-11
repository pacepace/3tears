"""changed rows by hash: the rows of a scope whose content differs from what the caller holds, read by key.

A follower holding a relation's rows with each row's hash beside it reads a scope (``by=`` / ``where=``: a state,
the geographies moved past a mark) in two phases::

    held = {(geo,): stored_hash, ...}            # the caller's rows inside this scope
    changed = await read_changed(reader, "results", key=["geo"], columns=[...], hash_columns=[...],
                                 where=[Fragment('t."state" = $1', ("PA",))], held=held)

1. the scope's keys and each row's hash only (``SELECT key..., <row_hash> AS row_hash``), paged by the key
   (:func:`~threetears.core.keyed.keyset_rows`);
2. the full rows, by key (:func:`~threetears.core.keyed.read_keyed`), of only the keys whose hash differs from
   ``held`` or that ``held`` lacks.

``gone_keys`` are the held keys phase 1 did not answer: rows that left the scope or the relation.

The hash is ONE SQL expression, :func:`row_hash_sql`: an MD5 over a canonical, typed text of the hashed columns,
each column length-prefixed and a NULL marked apart from every value, so the same values give the same hash on
PostgreSQL (and YugabyteDB), DuckDB and Redshift. A full read selects the same expression and stores it beside the
row (:func:`row_hash_selected` among its ``columns=``); the hash is never computed in Python.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from threetears.core.backends.dialect import known_on_yugabyte
from threetears.core.keyed._filters import Derived, Fragment, Selected
from threetears.core.keyed.counts import SQL_DIALECT_ATTRIBUTE, Dialect
from threetears.core.keyed.inline import InlineDuckDB
from threetears.core.keyed.threaded import ThreadedDuckDB
from threetears.core.keyed.page import Sort, keyset_rows
from threetears.core.keyed.rows import KeyBatch, read_keyed
from threetears.core.sql_fragments import quote_identifier

__all__ = [
    "ROW_HASH_COLUMN",
    "ChangedRows",
    "HashColumn",
    "HashKind",
    "dialect_of",
    "read_changed",
    "row_hash_selected",
    "row_hash_sql",
]

#: the name a row's hash is read under, in both phases and in a full read that stores it
ROW_HASH_COLUMN: Final = "row_hash"

#: the most digits a hashed NUMERIC's scale may name (DECIMAL(38, scale) on every dialect)
_MAX_SCALE: Final = 37

#: the scope's keys and hashes a phase-1 statement reads, on a transport that states no cap of its own (a key and a
#: 32-digit hash: about 3.5 MB of JSON)
DEFAULT_HASH_PAGE: Final = 50_000

#: a hashed NULL: every value's text starts with its length's digits, so this mark is no value's
_NULL_MARK: Final = "'~'"


class HashKind(StrEnum):
    """how a hashed column's value is written as canonical text, the same on every dialect."""

    #: the text itself
    TEXT = "text"
    #: an integer's decimal digits
    INTEGER = "integer"
    #: a decimal at a fixed scale (``HashColumn.scale``): ``1.50`` at scale 2, whatever the column's own
    NUMERIC = "numeric"
    #: ``t`` or ``f``
    BOOLEAN = "boolean"
    #: ``YYYY-MM-DD`` (PostgreSQL writes an infinite date as no text: it hashes as NULL there)
    DATE = "date"
    #: a timestamp without time zone, ``YYYY-MM-DD HH:MM:SS.ffffff`` (six fractional digits always)
    TIMESTAMP = "timestamp"


@dataclass(frozen=True)
class HashColumn:
    """one column a row's hash covers.

    A floating-point column is hashed as :attr:`HashKind.NUMERIC` at the scale its values carry; a float's own text
    differs between databases.

    :param column: the column, a TRUSTED identifier
    :ptype column: str
    :param kind: how its value is written before hashing
    :ptype kind: HashKind
    :param scale: a NUMERIC's fractional digits (0 to 37); None for every other kind
    :ptype scale: int | None
    """

    column: str
    kind: HashKind = HashKind.TEXT
    scale: int | None = None

    def __post_init__(self) -> None:
        if self.kind is HashKind.NUMERIC:
            if self.scale is None or not 0 <= self.scale <= _MAX_SCALE:
                raise ValueError(f"a hashed numeric {self.column!r} names its scale, 0 to {_MAX_SCALE}")
        elif self.scale is not None:
            raise ValueError(f"only a hashed numeric takes a scale, {self.column!r} is {self.kind}")


def _value_text(column: HashColumn, dialect: Dialect, alias: str | None) -> str:
    """``column``'s canonical text on ``dialect``; NULL for a NULL."""
    spelled = quote_identifier(column.column)
    named = f"{alias}.{spelled}" if alias else spelled
    duck = dialect is Dialect.DUCKDB
    match column.kind:
        case HashKind.TEXT:
            # Redshift's VARCHAR without a length is VARCHAR(256): a longer text would be cut
            return named if dialect is Dialect.REDSHIFT else f"CAST({named} AS VARCHAR)"
        case HashKind.INTEGER:
            return f"CAST({named} AS VARCHAR)"
        case HashKind.NUMERIC:
            return f"CAST(CAST({named} AS DECIMAL(38, {column.scale})) AS VARCHAR)"
        case HashKind.BOOLEAN:
            return f"CASE WHEN {named} THEN 't' WHEN NOT {named} THEN 'f' END"
        case HashKind.DATE:
            return f"strftime({named}, '%Y-%m-%d')" if duck else f"to_char({named}, 'YYYY-MM-DD')"
        case HashKind.TIMESTAMP:
            if duck:
                return f"strftime({named}, '%Y-%m-%d %H:%M:%S.%f')"
            return f"to_char({named}, 'YYYY-MM-DD HH24:MI:SS.US')"
    raise ValueError(f"no canonical text for {column.kind!r}")  # pragma: no cover - every kind is matched


async def dialect_of(conn: Any) -> Dialect:
    """the dialect ``conn``'s statements run in: a reader's own :data:`SQL_DIALECT_ATTRIBUTE` (a datasource reader
    names its datasource's engine), DuckDB for an in-process DuckDB, else PostgreSQL or YugabyteDB as
    :func:`~threetears.core.backends.dialect.known_on_yugabyte` learns it (no round trip on an asyncpg connection).
    The caller never names it: the hash a full read stored and the hash a later read compares are spelled alike.

    :param conn: the connection or reader
    :ptype conn: Any
    :return: its dialect
    :rtype: Dialect
    :raises ValueError: when a reader names no dialect this module spells (a datasource of another engine)
    """
    named = getattr(conn, SQL_DIALECT_ATTRIBUTE, None)
    if named is not None:
        if not isinstance(named, Dialect):
            raise ValueError(f"{type(conn).__name__} names a dialect no row hash is spelled for: {named!r}")
        return named
    if isinstance(conn, InlineDuckDB | ThreadedDuckDB):
        return Dialect.DUCKDB
    return Dialect.YUGABYTE if await known_on_yugabyte(conn) else Dialect.POSTGRES


def row_hash_sql(columns: Sequence[HashColumn], dialect: Dialect, *, alias: str | None = "t") -> str:
    """the SQL expression of a row's hash over ``columns``: the same text of hex digits on every dialect for the
    same values.

    Each column's canonical text (:class:`HashKind`) is written as its length, ``:`` and the text, a NULL as ``~``,
    and the columns are joined in the order given, so no two rows of different values join to the same text. The
    hash is ``md5`` of it (32 lowercase hex digits).

    :param columns: the hashed columns, in a fixed order
    :ptype columns: Sequence[HashColumn]
    :param dialect: the database the expression runs on (YugabyteDB spells it as PostgreSQL)
    :ptype dialect: Dialect
    :param alias: the alias the row's columns are named through (``t."col"``); bare names when None
    :ptype alias: str | None
    :return: the expression, TRUSTED SQL with no parameters
    :rtype: str
    :raises ValueError: when no column is named, or one is named twice
    """
    if not columns:
        raise ValueError("a row hash covers at least one column")
    named = [column.column for column in columns]
    if len(set(named)) != len(named):
        raise ValueError(f"a row hash names each column once, got {named}")
    parts = []
    for column in columns:
        text = _value_text(column, dialect, alias)
        parts.append(f"COALESCE(CAST(LENGTH({text}) AS VARCHAR) || ':' || {text}, {_NULL_MARK})")
    return f"md5({' || '.join(parts)})"


def row_hash_selected(columns: Sequence[HashColumn], dialect: Dialect, *, alias: str | None = "t") -> Selected:
    """:func:`row_hash_sql` as a select-list item under :data:`ROW_HASH_COLUMN`, for any helper's ``columns=`` (a full
    read or a cold load that stores the hash beside each row: ``read_keyed(..., columns=[*cols,
    row_hash_selected(hashed, dialect)])``).

    :param columns: the hashed columns, in a fixed order
    :ptype columns: Sequence[HashColumn]
    :param dialect: the database the read runs on
    :ptype dialect: Dialect
    :param alias: the alias the row's columns are named through; bare names when None
    :ptype alias: str | None
    :return: the item
    :rtype: Selected
    """
    return Selected(row_hash_sql(columns, dialect, alias=alias), ROW_HASH_COLUMN)


@dataclass(frozen=True)
class ChangedRows:
    """what a read of a scope's changed rows answered.

    :param rows: the full rows whose hash differs from the held one or that were not held, each with its hash
        under :data:`ROW_HASH_COLUMN` (store it beside the row)
    :ptype rows: list[dict[str, Any]]
    :param gone_keys: the held keys the scope no longer holds
    :ptype gone_keys: list[tuple[Any, ...]]
    """

    rows: list[dict[str, Any]]
    gone_keys: list[tuple[Any, ...]]


def _hash_page(conn: Any) -> int:
    """phase 1's page on ``conn``: :data:`DEFAULT_HASH_PAGE`, or one row under the cap its transport states (a page
    asks one row more than it keeps)."""
    stated = getattr(conn, "rows_per_statement", None)
    if isinstance(stated, int) and stated > 1:
        return min(DEFAULT_HASH_PAGE, stated - 1)
    return DEFAULT_HASH_PAGE


def _sorts(key: Sequence[str | Sort]) -> list[Sort]:
    return [part if isinstance(part, Sort) else Sort(part) for part in key]


async def read_changed(
    conn: Any,
    relation: str | Derived,
    *,
    key: Sequence[str | Sort],
    columns: Sequence[str | Selected],
    held: Mapping[tuple[Any, ...], str | None],
    hash_columns: Sequence[HashColumn],
    by: Mapping[str, Any] | None = None,
    where: Sequence[Fragment] = (),
    page: int | None = None,
    batch: int | None = None,
    row_cap: int | None = None,
) -> ChangedRows:
    """the rows of ``relation`` inside the scope ``by`` / ``where`` names whose hash differs from ``held``, read in
    full by key, and the held keys the scope no longer holds.

    :param conn: anything with an asyncpg-shaped ``fetch``: a connection, the L3 rail's proxy, a
        :class:`~threetears.datasources.keyed_reader.DatasourceReader`, an in-process DuckDB; its dialect
        (:func:`dialect_of`) spells the hash, and a warehouse's (Redshift) phase 2 binds its keys as a list
    :ptype conn: Any
    :param relation: the relation, a TRUSTED name, or a derived relation
    :ptype relation: str | Derived
    :param key: the relation's key, in its index's order: a column (its values compared as text) or a
        :class:`~threetears.core.keyed.Sort` naming its SQL type (``Sort("id", cast="bigint")``)
    :ptype key: Sequence[str | Sort]
    :param columns: the columns the full rows carry, as :func:`~threetears.core.keyed.read_keyed` takes them
    :ptype columns: Sequence[str | Selected]
    :param held: each key the caller holds INSIDE this scope (a tuple in ``key``'s order, its values as the read
        answers them) onto the hash stored with it, None for a row stored with none (always read again); a held key outside the scope would be reported gone
    :ptype held: Mapping[tuple[Any, ...], str | None]
    :param hash_columns: the columns the hash covers, as the stored hashes were taken
    :ptype hash_columns: Sequence[HashColumn]
    :param by: the scope's pinned leading columns, as :func:`~threetears.core.keyed.keyset_rows` takes them
    :ptype by: Mapping[str, Any] | None
    :param where: the scope's filters, each a fragment written as ``t."col"`` from ``$1`` (a mark the rows moved
        past); applied in both phases
    :ptype where: Sequence[Fragment]
    :param page: the keys and hashes a phase-1 statement reads; None: :data:`DEFAULT_HASH_PAGE`, or under the cap
        a transport states (``rows_per_statement``: the L3 rail's 1,000). A reader with a pager of its own
        (``read_scope(relation, *, columns, key, by, where)``: a datasource reader, sized from the hub's announced
        reply cap and reading on past a cut reply) pages phase 1 itself, and ``page`` is not used
    :ptype page: int | None
    :param batch: the keys a phase-2 statement names; None: the connection's default
    :ptype batch: int | None
    :param row_cap: the most rows the transport answers a phase-2 statement without saying it cut; None: the
        transport's own ``rows_per_statement``, or none when it states none
    :ptype row_cap: int | None
    :return: the changed rows and the gone keys
    :rtype: ChangedRows
    :raises ValueError: when the key is empty, or as :func:`row_hash_sql`, :func:`~threetears.core.keyed.keyset_rows`
        and :func:`~threetears.core.keyed.read_keyed` refuse
    """
    order = _sorts(key)
    if not order:
        raise ValueError(f"a changed-rows read of {relation} names its key")
    dialect = await dialect_of(conn)
    hashed = row_hash_selected(hash_columns, dialect)
    key_columns = tuple(sort.column for sort in order)
    read_scope = getattr(conn, "read_scope", None)
    if read_scope is not None and isinstance(relation, str):
        # a reader with a pager of its own (a datasource's, sized from the hub's reply cap, reading on past a cut
        # reply) reads the scope's keys and hashes through it
        scope = await read_scope(relation, columns=[*key_columns, hashed], key=key_columns, by=by or {}, where=where)
    else:
        scope = await keyset_rows(
            conn,
            relation,
            columns=[*key_columns, hashed],
            by=by or {},
            order=order,
            page=_hash_page(conn) if page is None else page,
            where=where,
        )
    seen: set[tuple[Any, ...]] = set()
    wanted: list[tuple[Any, ...]] = []
    for row in scope:
        held_key = tuple(row[column] for column in key_columns)
        seen.add(held_key)
        if held.get(held_key) != row[ROW_HASH_COLUMN]:
            wanted.append(held_key)
    gone = [held_key for held_key in held if held_key not in seen]
    if not wanted:
        return ChangedRows(rows=[], gone_keys=gone)
    keys = KeyBatch(
        columns=key_columns,
        values=wanted,
        casts=tuple(sort.cast or "text" for sort in order),
        as_list=dialect is Dialect.REDSHIFT,
    )
    rows = await read_keyed(
        conn,
        relation,
        keys=keys,
        columns=[*columns, hashed],
        by=by,
        batch=batch,
        row_cap=getattr(conn, "rows_per_statement", None) if row_cap is None else row_cap,
        where=where,
    )
    return ChangedRows(rows=rows, gone_keys=gone)
