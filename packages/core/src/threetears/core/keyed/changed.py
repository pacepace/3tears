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

A key column is typed by its :class:`HashKind` (the warehouse's type: from ``hash_columns`` when the hash covers it,
or ``key_kinds``): phase 1 pages it and phase 2 binds it cast to that type, and a held key is matched to the key the
scope answers by both values' canonical text (:func:`key_text`), so a key held in another form than the warehouse
answers it (an integer id held as text, a time held with its zone) is neither read again nor reported gone.

The hash is ONE SQL expression, :func:`row_hash_sql`: an MD5 over a canonical, typed text of the hashed columns,
each column length-prefixed and a NULL marked apart from every value, so the same values give the same hash on
PostgreSQL (and YugabyteDB), DuckDB and Redshift. A full read selects the same expression and stores it beside the
row (:func:`row_hash_selected` among its ``columns=``); the hash is never computed in Python.

Every hash a read answers (a row's, a group's) is its 128 bits as unpadded base64 (:class:`HashEncoding`, 22
characters where hex took 32). A held hash is compared with an answered one by its bytes, so a hash stored as hex
before 3tears 0.95.0 matches the same row's base64: stored hashes move to base64 as their rows are written again,
none rewritten in place.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, Final

from threetears.core.backends.dialect import known_on_yugabyte
from threetears.core.keyed._filters import Derived, Fragment, Selected, by_conditions, narrowing, sent_by
from threetears.core.keyed._filters import relation as _relation
from threetears.core.keyed.counts import SQL_DIALECT_ATTRIBUTE, Dialect
from threetears.core.keyed.held import Held, HeldArrow, HeldMapping
from threetears.core.keyed.held import hash_bytes as _hash_bytes
from threetears.core.keyed.inline import InlineDuckDB
from threetears.core.keyed.page import Sort, keyset_rows
from threetears.core.keyed.rows import KeyBatch, read_keyed
from threetears.core.keyed.threaded import ThreadedDuckDB
from threetears.core.sql_fragments import quote_identifier
from threetears.interfaces.keyed.changed import (
    ROW_HASH_COLUMN,
    ChangedRows,
    GroupHash,
    HashColumn,
    HashEncoding,
    HashKind,
)

try:
    import pyarrow as pa
except ImportError:  # NOSILENT: the snapshot extra is optional; held is then a mapping (an Arrow table needs it)
    pa = None

__all__ = [
    "ROW_HASH_COLUMN",
    "ChangedRows",
    "GroupHash",
    "HashColumn",
    "HashKind",
    "dialect_of",
    "group_hashes",
    "key_text",
    "read_changed",
    "row_hash_selected",
    "row_hash_sql",
]

#: the scope's keys and hashes a phase-1 statement reads, on a transport that states no cap of its own (a key and a
#: 22-character hash: about 3 MB of JSON)
DEFAULT_HASH_PAGE: Final = 50_000

#: a hashed NULL: every value's text starts with its length's digits, so this mark is no value's
_NULL_MARK: Final = "'~'"

#: a group's hash lanes: each the sum, mod 2**32, of one 8-hex-digit slice of every row hash in the group (1-based
#: start in the 32 digits of the md5); four distinct slices, so the group hash covers all 128 bits of every row's
_LANES: Final = (1, 9, 17, 25)

#: each lane is summed modulo this
_LANE_MOD: Final = 2**32

#: the bytes of a hash (an md5's 128 bits; a group's four 32-bit lanes)
_HASH_BYTES: Final = 16

#: the groups a phase-0 statement answers (a group's row is its value, a count and its hash: about 70 bytes of JSON)
_GROUP_PAGE: Final = 10_000

#: the moved groups a phase-1 statement names
_GROUP_BATCH: Final = 1_000


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


#: a key column's SQL types (a Sort's cast) a text-kind key's canonical text is the value itself
_TEXT_CASTS: Final = frozenset({"text", "varchar", "character varying"})

#: a key column's SQL types (a Sort's cast) a text-kind key's canonical text is the digits of
_INTEGER_CASTS: Final = frozenset({"bigint", "integer", "int", "smallint", "int2", "int4", "int8"})


def _key_part_text(kind: HashColumn, sort: Sort) -> str | None:
    """a phase-2 key column's canonical text on Redshift, over ``t``: its kind's, as the hash writes it; a text-kind
    column of an integer type by its digits (``str`` of the value, as :func:`key_text` writes it); None for a
    text-kind column of another type, whose text is no canonical one (its kind belongs in ``key_kinds``): its batch
    is then bound as a list of tuples."""
    cast = (sort.cast or "text").lower().split("(")[0].strip()
    if kind.kind is not HashKind.TEXT or cast in _TEXT_CASTS:
        return _value_text(kind, Dialect.REDSHIFT, "t")
    if cast in _INTEGER_CASTS:
        return f"CAST(t.{quote_identifier(kind.column)} AS VARCHAR)"
    return None


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


def _base64_of_hex(hexed: str, dialect: Dialect) -> str:
    """the SQL of the bytes the hex text ``hexed`` spells, as unpadded base64, on ``dialect``: the same 22 characters
    for 16 bytes everywhere (Redshift's spelling checked live against Python's, 2026-11-11; ``TO_VARCHAR`` refuses
    there)."""
    match dialect:
        case Dialect.REDSHIFT:
            return f"RTRIM(FROM_VARBYTE(FROM_HEX({hexed}), 'base64'), '=')"
        case Dialect.DUCKDB:
            return f"rtrim(to_base64(unhex({hexed})), '=')"
    return f"rtrim(encode(decode({hexed}, 'hex'), 'base64'), '=')"


def row_hash_sql(
    columns: Sequence[HashColumn],
    dialect: Dialect,
    *,
    alias: str | None = "t",
    encoding: HashEncoding = HashEncoding.BASE64,
) -> str:
    """the SQL expression of a row's hash over ``columns``: the same text on every dialect for the same values.

    Each column's canonical text (:class:`HashKind`) is written as its length, ``:`` and the text, a NULL as ``~``,
    and the columns are joined in the order given, so no two rows of different values join to the same text. The
    hash is ``md5`` of it, all 128 bits: as unpadded base64 (22 characters), or as 32 lowercase hex digits.

    :param columns: the hashed columns, in a fixed order
    :ptype columns: Sequence[HashColumn]
    :param dialect: the database the expression runs on (YugabyteDB spells it as PostgreSQL)
    :ptype dialect: Dialect
    :param alias: the alias the row's columns are named through (``t."col"``); bare names when None
    :ptype alias: str | None
    :param encoding: how the hash is written; every read answers base64, hex is the spelling the group lanes slice
    :ptype encoding: HashEncoding
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
    hexed = f"md5({' || '.join(parts)})"
    return hexed if encoding is HashEncoding.HEX else _base64_of_hex(hexed, dialect)


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


def _group_hash(hashes: Sequence[str]) -> GroupHash:
    """the group hash of rows held with these row hashes (the stored text of :func:`row_hash_sql`, either
    spelling): what :func:`_group_hash_sql` answers for the same rows on every dialect.

    :param hashes: each row's hash, 32 hex digits or 22 of unpadded base64
    :ptype hashes: Sequence[str]
    :return: the group hash
    :rtype: GroupHash
    :raises ValueError: for a hash in neither spelling
    """
    sums = [0, 0, 0, 0]
    for hashed in hashes:
        spelled = _hash_bytes(hashed)
        for lane, start in enumerate(_LANES):
            at = (start - 1) // 2
            sums[lane] += int.from_bytes(spelled[at : at + 4], "big")
    return GroupHash(
        rows=len(hashes), lanes=(sums[0] % _LANE_MOD, sums[1] % _LANE_MOD, sums[2] % _LANE_MOD, sums[3] % _LANE_MOD)
    )


def _lane_value(hashed: str, start: int, dialect: Dialect) -> str:
    """one 8-hex-digit slice of a row hash expression as a non-negative integer, on ``dialect``."""
    piece = f"SUBSTRING({hashed}, {start}, 8)"
    match dialect:
        case Dialect.REDSHIFT:
            return f"STRTOL({piece}, 16)"
        case Dialect.DUCKDB:
            return f"CAST('0x' || {piece} AS BIGINT)"
    return f"CAST(CAST('x' || LPAD({piece}, 16, '0') AS BIT(64)) AS BIGINT)"


def _group_hash_sql(hashed: str, dialect: Dialect) -> list[str]:
    """the select items of a group's hash over the hex row hash expression ``hashed`` (:func:`row_hash_sql` in
    :attr:`HashEncoding.HEX`), for a ``GROUP BY``: the count as ``group_rows`` and the four lanes, each 8 hex digits
    in lane order, as the unpadded base64 of their 16 bytes under ``group_hash`` (what :func:`_group_hash` computes
    from the stored hashes, :func:`_answered_group_hash` reads).

    :param hashed: the hex row hash, a TRUSTED expression
    :ptype hashed: str
    :param dialect: the database the statement runs on
    :ptype dialect: Dialect
    :return: the items
    :rtype: list[str]
    """
    # each lane reduced into [0, 2**32) before its 8 hex digits: LPAD cuts a longer text from the right (Redshift
    # and PostgreSQL alike), so a value outside the range would keep its high digits and lose its low ones. A sum of
    # slices is never negative; the second MOD holds the range for any value all the same, on every engine
    lanes = " || ".join(
        f"LPAD(TO_HEX(CAST(MOD(MOD(SUM({_lane_value(hashed, start, dialect)}), {_LANE_MOD}) + {_LANE_MOD}, "
        f"{_LANE_MOD}) AS BIGINT)), 8, '0')"
        for start in _LANES
    )
    return ["count(*) AS group_rows", f"{_base64_of_hex(lanes, dialect)} AS group_hash"]


def _answered_group_hash(row: Mapping[str, Any]) -> GroupHash:
    """a phase-0 row's group hash: its four lanes from the 16 bytes ``group_hash`` spells."""
    spelled = _hash_bytes(str(row["group_hash"]))
    lanes = tuple(int.from_bytes(spelled[at : at + 4], "big") for at in range(0, _HASH_BYTES, 4))
    return GroupHash(rows=int(row["group_rows"]), lanes=lanes)  # type: ignore[arg-type]


#: the SQL type a key column of each kind is cast to where its value is bound as text (a page's cursor, a key array)
_KEY_CASTS: Final[Mapping[HashKind, str]] = {
    HashKind.TEXT: "text",
    HashKind.INTEGER: "bigint",
    HashKind.NUMERIC: "numeric",
    HashKind.BOOLEAN: "boolean",
    HashKind.DATE: "date",
    HashKind.TIMESTAMP: "timestamp",
}

#: a boolean's texts, by the canonical one
_BOOLEAN_TEXTS: Final[Mapping[str, str]] = {
    **dict.fromkeys(("t", "true", "1", "y", "yes"), "t"),
    **dict.fromkeys(("f", "false", "0", "n", "no"), "f"),
}


def _timestamp(value: Any) -> datetime:
    """a timestamp key's value as a naive UTC time: a zoned one moved to UTC, text read as ISO (a date as its
    midnight)."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value.strip())
    elif isinstance(value, date) and not isinstance(value, datetime):
        value = datetime(value.year, value.month, value.day)  # noqa: DTZ001 - a date's midnight, naive as the column
    if not isinstance(value, datetime):
        raise ValueError(f"a timestamp key value is a time or its text, got {value!r}")
    return value.astimezone(UTC).replace(tzinfo=None) if value.tzinfo is not None else value


def key_text(value: Any, kind: HashKind = HashKind.TEXT, scale: int | None = None) -> str | None:
    """a key value's canonical text for ``kind``, whatever form it is held in: the text :func:`row_hash_sql` writes
    for that kind (an integer's digits, a decimal at ``scale``, ``t``/``f``, ``YYYY-MM-DD``,
    ``YYYY-MM-DD HH:MM:SS.ffffff`` in UTC), so a value held as text and the same value as the warehouse answers it
    match. A zoned time is taken in UTC, a naive one as UTC already.

    :param value: the value: of the kind's type, or its text
    :ptype value: Any
    :param kind: the key column's kind
    :ptype kind: HashKind
    :param scale: a NUMERIC's fractional digits; None takes the value's own
    :ptype scale: int | None
    :return: the text; None for a NULL
    :rtype: str | None
    :raises ValueError: when the value is no value of its kind
    """
    if value is None:
        return None
    match kind:
        case HashKind.TEXT:
            return str(value)
        case HashKind.INTEGER:
            if isinstance(value, float) or (isinstance(value, Decimal) and value != value.to_integral_value()):
                raise ValueError(f"an integer key value is whole, got {value!r}")
            return str(int(value.strip() if isinstance(value, str) else value))
        case HashKind.NUMERIC:
            number = Decimal(str(value).strip())
            return format(number if scale is None else number.quantize(Decimal(1).scaleb(-scale)), "f")
        case HashKind.BOOLEAN:
            if isinstance(value, bool):
                return "t" if value else "f"
            text = _BOOLEAN_TEXTS.get(str(value).strip().lower())
            if text is None:
                raise ValueError(f"a boolean key value is true or false, got {value!r}")
            return text
        case HashKind.DATE:
            if isinstance(value, datetime):
                raise ValueError(f"a date key value is a date, got the time {value!r}")
            return (value if isinstance(value, date) else date.fromisoformat(str(value).strip()[:10])).isoformat()
        case HashKind.TIMESTAMP:
            return _timestamp(value).strftime("%Y-%m-%d %H:%M:%S.%f")
    raise ValueError(f"no canonical text for {kind!r}")  # pragma: no cover - every kind is matched


def _hash_page(conn: Any) -> int:
    """phase 1's page on ``conn``: :data:`DEFAULT_HASH_PAGE`, or one row under the cap its transport states (a page
    asks one row more than it keeps)."""
    stated = getattr(conn, "rows_per_statement", None)
    if isinstance(stated, int) and stated > 1:
        return min(DEFAULT_HASH_PAGE, stated - 1)
    return DEFAULT_HASH_PAGE


def _key_kinds(
    key: Sequence[str | Sort], hash_columns: Sequence[HashColumn], key_kinds: Sequence[HashColumn]
) -> list[HashColumn]:
    """each key column's kind, in key order: ``key_kinds``' for it, else the hash's, else text."""
    named = {column.column: column for column in hash_columns} | {column.column: column for column in key_kinds}
    columns = [part.column if isinstance(part, Sort) else part for part in key]
    return [named.get(column, HashColumn(column)) for column in columns]


def _sorts(key: Sequence[str | Sort], kinds: Sequence[HashColumn]) -> list[Sort]:
    """the key's sorts, in key order: a column named bare cast to its kind's type; a Sort as given."""
    return [
        part if isinstance(part, Sort) else Sort(part, cast=_KEY_CASTS[kind.kind])
        for part, kind in zip(key, kinds, strict=True)
    ]


async def read_changed(
    conn: Any,
    relation: str | Derived,
    *,
    key: Sequence[str | Sort],
    columns: Sequence[str | Selected],
    held: Mapping[tuple[Any, ...], str | None] | Any,
    hash_columns: Sequence[HashColumn],
    by: Mapping[str, Any] | None = None,
    where: Sequence[Fragment] = (),
    page: int | None = None,
    batch: int | None = None,
    row_cap: int | None = None,
    key_kinds: Sequence[HashColumn] = (),
    group: str | None = None,
    full_rows_at_most: int | None = None,
    groups: Mapping[str | None, tuple[Any, GroupHash]] | None = None,
    group_rows_at_most: int | None = None,
) -> ChangedRows:
    """the rows of ``relation`` inside the scope ``by`` / ``where`` names whose hash differs from ``held``, read in
    full by key, and the held keys the scope no longer holds.

    :param conn: anything with an asyncpg-shaped ``fetch``: a connection, the L3 rail's proxy, a
        :class:`~threetears.datasources.keyed_reader.DatasourceReader`, an in-process DuckDB; its dialect
        (:func:`dialect_of`) spells the hash, and a warehouse's (Redshift) phase 2 binds its keys as a list
    :ptype conn: Any
    :param relation: the relation, a TRUSTED name, or a derived relation
    :ptype relation: str | Derived
    :param key: the relation's key, in its index's order: a column, cast to its kind's SQL type (``key_kinds``,
        ``hash_columns``; text when neither names it), or a :class:`~threetears.core.keyed.Sort` naming its SQL type
        (``Sort("id", cast="bigint")``)
    :ptype key: Sequence[str | Sort]
    :param columns: the columns the full rows carry, as :func:`~threetears.core.keyed.read_keyed` takes them
    :ptype columns: Sequence[str | Selected]
    :param held: each key the caller holds INSIDE this scope (a tuple in ``key``'s order, each value of its kind or
        its text: matched to the answered key by :func:`key_text`) onto the hash stored with it, in either
        spelling (:class:`HashEncoding`: compared by its bytes), None for a row stored with none (always read again); a
        held key outside the scope would be reported gone. Or an Arrow table of the key columns (named as in
        ``key``, any type) and :data:`ROW_HASH_COLUMN` (NULL: always read again), as L3 answers them: matched and
        compared column by column in ``pyarrow.compute`` and numpy, never a Python row of it built (the gone keys,
        the answer, are the only Python values made of it); the same answer as the mapping
    :ptype held: Mapping[tuple[Any, ...], str | None] | pa.Table
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
    :param key_kinds: the kind of each key column the hash does not cover, or covers as another kind than the
        warehouse holds the key in; a key column named in neither is text
    :ptype key_kinds: Sequence[HashColumn]
    :param group: a key column the scope's rows are grouped by (a geography): first a phase 0 reads one row a group,
        its row count and an order-free hash of its rows' hashes (:func:`_group_hash_sql`), compared with the same
        hash of the held rows' stored hashes (:func:`_group_hash`); phase 1 then reads the keys and hashes of only the
        groups whose hash moved, and a held group the scope no longer answers is gone whole. None: no phase 0
    :ptype group: str | None
    :param groups: phase 0 already answered for this scope, one slice of :func:`group_hashes` (every group the scope
        holds: its canonical text -> its value as answered and its group hash; a group missing from it is a group the
        scope holds no row of); read_changed reads no phase 0 of its own. Needs ``group``. None: phase 0 is read here
    :ptype groups: Mapping[str | None, tuple[Any, GroupHash]] | None
    :param full_rows_at_most: the most changed keys phase 2 reads in full; past it no full row is read and the
        answer is ``capped``, naming the changed keys (the caller reads them its own way, one export). None: no cap
    :ptype full_rows_at_most: int | None
    :param group_rows_at_most: the most rows the groups whose hash moved may hold, as phase 0 counts them, for their
        keys to be read; past it no key's hash and no full row is read, and the answer is ``capped``, naming the moved
        groups (the caller reads them its own way, one export: a move judged broad on phase 0 alone, with no phase 1
        spent). Needs ``group``. None: no cap
    :ptype group_rows_at_most: int | None
    :return: the changed rows (each key as the warehouse answers it) and the gone keys (as ``held`` names them)
    :rtype: ChangedRows
    :raises ValueError: when the key is empty, or as :func:`row_hash_sql`, :func:`~threetears.core.keyed.keyset_rows`
        and :func:`~threetears.core.keyed.read_keyed` refuse, or a held key's value is no value of its kind; when
        ``group`` is no key column or ``full_rows_at_most`` is negative, or ``groups`` or ``group_rows_at_most`` is
        given without ``group``
    """
    kinds = _key_kinds(key, hash_columns, key_kinds)
    order = _sorts(key, kinds)
    if not order:
        raise ValueError(f"a changed-rows read of {relation} names its key")
    if full_rows_at_most is not None and full_rows_at_most < 0:
        raise ValueError(f"full_rows_at_most is a count of rows, got {full_rows_at_most}")
    if group_rows_at_most is not None and (group is None or group_rows_at_most < 0):
        raise ValueError(f"group_rows_at_most is a count of rows of a grouped read, got {group_rows_at_most}")
    dialect = await dialect_of(conn)
    hashed = row_hash_selected(hash_columns, dialect)
    key_columns = tuple(sort.column for sort in order)

    def canonical(values: Sequence[Any]) -> tuple[str | None, ...]:
        return tuple(key_text(value, kind.kind, kind.scale) for value, kind in zip(values, kinds, strict=True))

    # held and answered keys meet in one form: each value's canonical text by its column's kind
    holding: Held = (
        HeldArrow(held, key_columns, kinds, _LANES, _LANE_MOD, key_text)
        if pa is not None and isinstance(held, pa.Table)
        else HeldMapping.of(held, canonical, _LANES, _LANE_MOD)
    )
    vanished: list[tuple[Any, ...]] = []
    moved_groups: list[Any] = []
    moved_group_rows = 0
    scopes: list[Sequence[Fragment]] = [where]
    if groups is not None and group is None:
        raise ValueError(f"a changed-rows read of {relation} given its groups' hashes names its group column")
    if group is not None:
        if group not in key_columns:
            raise ValueError(f"a changed-rows read groups by a key column, {group!r} is none of {key_columns}")
        at = key_columns.index(group)
        answered_groups = (
            groups
            if groups is not None
            else await _group_hashes(
                conn,
                relation,
                group=group,
                order=order[at],
                kind=kinds[at],
                by=by or {},
                where=where,
                hash_columns=hash_columns,
                dialect=dialect,
            )
        )
        held_groups = holding.group_hashes(at)
        moved: list[Any] = []
        moved_texts: set[str | None] = set()
        for group_text, (value, answered) in answered_groups.items():
            if held_groups.get(group_text) != answered:
                moved.append(value)
                moved_texts.add(group_text)
                moved_group_rows += answered.rows
        # a held group the scope answers no row of: every held key of it is gone, read no further
        vanished, gone_groups = holding.outside(at, set(answered_groups))
        moved_groups = [*moved, *gone_groups.values()]
        holding = holding.within(at, moved_texts)
        if not moved:
            return ChangedRows(rows=[], gone_keys=vanished, moved_groups=moved_groups)
        if group_rows_at_most is not None and moved_group_rows > group_rows_at_most:
            # broad on phase 0 alone: the moved groups' rows are past the cap, so none of their keys is read here
            return ChangedRows(
                rows=[],
                gone_keys=vanished,
                capped=True,
                moved_groups=moved_groups,
                moved_group_rows=moved_group_rows,
            )
        scopes = [
            [*where, _in_groups(group, moved[start : start + _GROUP_BATCH])]
            for start in range(0, len(moved), _GROUP_BATCH)
        ]
    read_scope = getattr(conn, "read_scope", None)
    scope: list[Mapping[str, Any]] = []
    for scoped in scopes:
        if read_scope is not None and isinstance(relation, str):
            # a reader with a pager of its own (a datasource's, sized from the hub's reply cap, reading on past a cut
            # reply) reads the scope's keys and hashes through it
            scope.extend(
                await read_scope(relation, columns=[*key_columns, hashed], key=key_columns, by=by or {}, where=scoped)
            )
        else:
            scope.extend(
                await keyset_rows(
                    conn,
                    relation,
                    columns=[*key_columns, hashed],
                    by=by or {},
                    order=order,
                    page=_hash_page(conn) if page is None else page,
                    where=scoped,
                )
            )

    wanted: list[tuple[Any, ...]] = []
    # each wanted key's hash as phase 1 answered it: phase 2 stamps it on the key's full row, never reading it again
    answered_hashes: dict[tuple[str | None, ...], Any] = {}
    answered_keys = [tuple(row[column] for column in key_columns) for row in scope]
    answered_texts = [canonical(answered_key) for answered_key in answered_keys]
    same, unanswered = holding.compare(answered_texts, [row[ROW_HASH_COLUMN] for row in scope])
    for row, answered_key, text, held_same in zip(scope, answered_keys, answered_texts, same, strict=True):
        if not held_same:
            wanted.append(answered_key)
            answered_hashes[text] = row[ROW_HASH_COLUMN]
    gone = [*vanished, *unanswered]
    if not wanted:
        return ChangedRows(rows=[], gone_keys=gone, moved_groups=moved_groups, moved_group_rows=moved_group_rows)
    if full_rows_at_most is not None and len(wanted) > full_rows_at_most:
        return ChangedRows(
            rows=[],
            gone_keys=gone,
            changed_keys=wanted,
            capped=True,
            moved_groups=moved_groups,
            moved_group_rows=moved_group_rows,
        )
    named = {column.alias if isinstance(column, Selected) else column for column in columns}
    # the key read beside the columns asked, to stamp each row with its phase-1 hash
    read_columns = [*columns, *(column for column in key_columns if column not in named)]
    cap = getattr(conn, "rows_per_statement", None) if row_cap is None else row_cap
    # a key with a NULL part is never named by equality: the keys are read a NULL pattern at a time, each NULL part
    # pinned IS NULL beside the batch of the rest (as a keyset page's NULL leg), so no changed row is left unread
    patterns: dict[tuple[bool, ...], list[tuple[Any, ...]]] = {}
    for wanted_key in wanted:
        patterns.setdefault(tuple(value is None for value in wanted_key), []).append(wanted_key)
    rows: list[Mapping[str, Any]] = []
    for nulls, pattern_keys in patterns.items():
        present = [at for at, null in enumerate(nulls) if not null]
        pinned = {**(by or {}), **{key_columns[at]: None for at, null in enumerate(nulls) if null}}
        if not present:
            rows.extend(await keyset_rows(conn, relation, columns=read_columns, by=pinned, order=order, where=where))
            continue
        spelled = [_key_part_text(kinds[at], order[at]) for at in present]
        keys = (
            # a warehouse's composite key: one scalar IN of the key's canonical text (each column's text as the hash
            # writes it, Python's by key_text), where Redshift plans a list of tuples key by key (528 four-column
            # keys: 4.4 s of planning against 0.11 s). The parts are NULL-free: a NULL part is pinned above
            KeyBatch(
                columns=tuple(key_columns[at] for at in present),
                values=[tuple(str(texts[at]) for at in present) for texts in map(canonical, pattern_keys)],
                as_list=True,
                as_text=tuple(text for text in spelled if text is not None),
            )
            if dialect is Dialect.REDSHIFT and len(present) > 1 and None not in spelled
            else KeyBatch(
                columns=tuple(key_columns[at] for at in present),
                values=[tuple(key[at] for at in present) for key in pattern_keys],
                casts=tuple(order[at].cast or "text" for at in present),
                as_list=dialect is Dialect.REDSHIFT,
            )
        )
        rows.extend(
            await read_keyed(
                conn,
                relation,
                keys=keys,
                columns=read_columns,
                by=pinned,
                batch=batch,
                row_cap=cap,
                where=where,
            )
        )
    stamped = [
        {
            **{name: value for name, value in row.items() if name in named},
            ROW_HASH_COLUMN: answered_hashes.get(canonical(tuple(row[column] for column in key_columns))),
        }
        for row in rows
    ]
    return ChangedRows(
        rows=stamped,
        gone_keys=gone,
        changed_keys=wanted,
        moved_groups=moved_groups,
        moved_group_rows=moved_group_rows,
    )


def _in_groups(group: str, values: Sequence[Any]) -> Fragment:
    """a phase-1 filter keeping the rows of ``values``' groups (a NULL group by ``IS NULL``)."""
    quoted = f"t.{quote_identifier(group)}"
    named = [value for value in values if value is not None]
    parts = [f"{quoted} IN ({', '.join(f'${place}' for place in range(1, len(named) + 1))})"] if named else []
    if len(named) < len(values):
        parts.append(f"{quoted} IS NULL")
    return Fragment(f"({' OR '.join(parts)})", tuple(named))


async def _group_hashes(
    conn: Any,
    relation: str | Derived,
    *,
    group: str,
    order: Sort,
    kind: HashColumn,
    by: Mapping[str, Any],
    where: Sequence[Fragment],
    hash_columns: Sequence[HashColumn],
    dialect: Dialect,
) -> dict[str | None, tuple[Any, GroupHash]]:
    """phase 0: each group of the scope -> its value as answered and its group hash, a page of groups a statement in
    the group's order (the NULL group first, on the first page)."""
    answered: dict[str | None, tuple[Any, GroupHash]] = {}
    for row in await _group_hash_rows(
        conn,
        relation,
        levels=[(group, order.cast or "text", "g")],
        by=by,
        where=where,
        hash_columns=hash_columns,
        dialect=dialect,
    ):
        value = row["group_value"]
        answered[key_text(value, kind.kind, kind.scale)] = (value, _answered_group_hash(row))
    return answered


async def group_hashes(
    conn: Any,
    relation: str | Derived,
    *,
    scope: str | Sort,
    scopes: Sequence[Any],
    group: str | Sort,
    hash_columns: Sequence[HashColumn],
    by: Mapping[str, Any] | None = None,
    where: Sequence[Fragment] = (),
    key_kinds: Sequence[HashColumn] = (),
) -> dict[str | None, dict[str | None, tuple[Any, GroupHash]]]:
    """phase 0 of :func:`read_changed` for many scopes at once: each group's row count and order-free hash of its
    rows' hashes, of every scope in ``scopes`` (the states an upload moved), in ONE statement a page (paged by scope
    then group, ``scope IN (...)``), where a read per scope would cost a statement each. Hand each scope's slice to
    ``read_changed(..., group=..., groups=<slice>)``, which then reads no phase 0 of its own.

    :param conn: as :func:`read_changed` takes it
    :ptype conn: Any
    :param relation: the relation, a TRUSTED name, or a derived relation
    :ptype relation: str | Derived
    :param scope: the column the scopes are values of (a key column: the state); a
        :class:`~threetears.core.keyed.Sort` names its SQL type, a bare name is typed as ``group`` is
    :ptype scope: str | Sort
    :param scopes: the scopes read; none: no statement
    :ptype scopes: Sequence[Any]
    :param group: the key column the scope's rows are grouped by (the geography), typed as ``scope`` is
    :ptype group: str | Sort
    :param hash_columns: the columns the hash covers, as the stored hashes were taken (and as read_changed is given)
    :ptype hash_columns: Sequence[HashColumn]
    :param by: pinned columns every scope shares, as :func:`read_changed` takes them
    :ptype by: Mapping[str, Any] | None
    :param where: filters every scope shares, each a fragment written as ``t."col"`` from ``$1`` (the union of the
        scopes' named groups)
    :ptype where: Sequence[Fragment]
    :param key_kinds: the kind of ``scope`` or ``group`` when the hash does not cover it, as :func:`read_changed`
        takes them; a column named in neither is text
    :ptype key_kinds: Sequence[HashColumn]
    :return: each asked scope's canonical text (:func:`key_text`) -> its groups (each group's canonical text -> its
        value as answered and its group hash); a scope holding no row maps to no group
    :rtype: dict[str | None, dict[str | None, tuple[Any, GroupHash]]]
    :raises ValueError: when ``scope`` is ``group``, or a scope is NULL
    """
    scope_kind, group_kind = _key_kinds([scope, group], hash_columns, key_kinds)
    scope_sort, group_sort = _sorts([scope, group], [scope_kind, group_kind])
    if scope_sort.column == group_sort.column:
        raise ValueError(f"group hashes are read by a scope and a group, {scope_sort.column!r} is both")
    if any(value is None for value in scopes):
        raise ValueError("a scope of group hashes is a value, never NULL")
    answered: dict[str | None, dict[str | None, tuple[Any, GroupHash]]] = {
        key_text(value, scope_kind.kind, scope_kind.scale): {} for value in scopes
    }
    if not scopes:
        return answered
    quoted = quote_identifier(scope_sort.column)
    named = ", ".join(f"CAST(${place} AS {scope_sort.cast or 'text'})" for place in range(1, len(scopes) + 1))
    for row in await _group_hash_rows(
        conn,
        relation,
        levels=[
            (scope_sort.column, scope_sort.cast or "text", "sc"),
            (group_sort.column, group_sort.cast or "text", "g"),
        ],
        by=by or {},
        where=[Fragment(f"t.{quoted} IN ({named})", tuple(scopes)), *where],
        hash_columns=hash_columns,
        dialect=await dialect_of(conn),
    ):
        scope_text = key_text(row["scope_value"], scope_kind.kind, scope_kind.scale)
        group_value = row["group_value"]
        answered.setdefault(scope_text, {})[key_text(group_value, group_kind.kind, group_kind.scale)] = (
            group_value,
            _answered_group_hash(row),
        )
    return answered


def _past(levels: Sequence[tuple[str, str, str]], last: Sequence[Any], params: list[Any]) -> str:
    """the keyset condition of the groups after ``last`` in ``levels``' order, each level NULLS FIRST: for each level,
    the levels before it equal to ``last``'s and it past ``last``'s (a NULL is passed by any value)."""
    branches = []
    for at, (column, cast, _) in enumerate(levels):
        parts = []
        for (before, before_cast, _), value in zip(levels[:at], last[:at], strict=True):
            if value is None:
                parts.append(f"t.{quote_identifier(before)} IS NULL")
            else:
                params.append(value)
                parts.append(f"t.{quote_identifier(before)} = CAST(${len(params)} AS {before_cast})")
        if last[at] is None:
            parts.append(f"t.{quote_identifier(column)} IS NOT NULL")
        else:
            params.append(last[at])
            parts.append(f"t.{quote_identifier(column)} > CAST(${len(params)} AS {cast})")
        branches.append(" AND ".join(parts))
    return branches[0] if len(branches) == 1 else "(" + " OR ".join(f"({branch})" for branch in branches) + ")"


async def _group_hash_rows(
    conn: Any,
    relation: str | Derived,
    *,
    levels: Sequence[tuple[str, str, str]],
    by: Mapping[str, Any],
    where: Sequence[Fragment],
    hash_columns: Sequence[HashColumn],
    dialect: Dialect,
) -> list[Mapping[str, Any]]:
    """each group of ``levels`` (column, SQL type, alias: the group as ``g``, or the scope as ``sc`` then the group)
    -> a row of its values (``group_value``, ``scope_value``) and its group hash (:func:`_group_hash_sql`), a page of
    groups a statement in the levels' order, each NULLS FIRST."""
    stated = getattr(conn, "rows_per_statement", None)
    page = min(_GROUP_PAGE, stated - 1) if isinstance(stated, int) and stated > 1 else _GROUP_PAGE
    items = ", ".join(_group_hash_sql("s.h", dialect))
    named = {"g": "group_value", "sc": "scope_value"}
    inner = ", ".join(f"t.{quote_identifier(column)} AS {alias}" for column, _, alias in levels)
    outer = ", ".join(f"s.{alias} AS {named[alias]}" for _, _, alias in levels)
    grouped = ", ".join(f"s.{alias}" for _, _, alias in levels)
    ordered = ", ".join(f"s.{alias} NULLS FIRST" for _, _, alias in levels)
    found: list[Mapping[str, Any]] = []
    last: Sequence[Any] | None = None
    while True:
        conditions, params = by_conditions(by, alias="t")
        for fragment in where:
            conditions.append(narrowing(fragment, len(params)))
            params.extend(fragment.params)
        if last is not None:
            conditions.append(_past(levels, last, params))
        filtered = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        # OFFSET 0 fences the inner select: unfenced, PostgreSQL (and YugabyteDB, Redshift) pull it up and inline the
        # row hash into each lane's SUM, computing every row's md5 once a lane (4x, measured 7-9x slower on PG);
        # fenced, each row is hashed once below the aggregate. Every dialect here accepts it; the answer is the same.
        sql = (
            f"SELECT {outer}, {items} FROM (SELECT {inner}, "
            f"{row_hash_sql(hash_columns, dialect, alias='t', encoding=HashEncoding.HEX)} AS h FROM {_relation(relation, 't')}{filtered}"
            f" OFFSET 0) AS s GROUP BY {grouped} ORDER BY {ordered} LIMIT {page}"
        )
        rows = await sent_by(conn).fetch(sql, *params)
        found.extend(rows)
        if len(rows) < page:
            return found
        last = [rows[-1][named[alias]] for _, _, alias in levels]
