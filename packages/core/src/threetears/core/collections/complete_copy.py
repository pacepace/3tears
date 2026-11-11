"""Reading an L3 table's rows by key, a page at a time, for a copy of it in L1.

The L3 rail answers at most a thousand rows a statement and does not say when it cut, so a read that did not page
would be silently short. :func:`read_l3_rows`, :func:`l3_row_pages` and :func:`read_l3_arrow` page by the table's
key, narrowed by the rows' own key prefix (``where``, ``where_any``): a copy reads the rows a change names, never
the whole table. :func:`rail_arrow_types` says how the rail answers each column as Arrow.

:class:`Unsettled` is a writer's record while a write is in progress (``ScopeEpochs.settled``), and
:class:`IncompleteCopyError` what a reader of a copy not known to hold its rows is told. A copy follows its table's
changes by key (:mod:`~threetears.core.collections.copy_follower`); the whole-table copies this module once built,
read and fingerprinted whole on every refresh, are gone.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from threetears.observe import get_logger

from threetears.core.backends.protocol import L3_RAIL_ROW_CAP
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    BOOL_TYPE,
    DATETIMETZ_TYPE,
    ENUM_TYPE,
    INT_TYPE,
    NUMERIC_TYPE,
    STRING_TYPE,
    TSVECTOR_TYPE,
    UUID_TYPE,
)
from threetears.core.keyset import keyset_arrow_pages, keyset_pages, read_keyset_pages
from threetears.core.sql_fragments import equality_conditions, quote_identifier

try:
    import pyarrow as pa

    _PYARROW = True
except ImportError:  # NOSILENT: the snapshot extra is optional; the Arrow reads refuse without it, saying so
    _PYARROW = False

if TYPE_CHECKING:
    from threetears.core.backends.protocol import L3Reader

__all__ = [
    "DEFAULT_PAGE_SIZE",
    "IncompleteCopyError",
    "Unsettled",
    "l3_row_pages",
    "rail_arrow_types",
    "read_l3_arrow",
    "read_l3_rows",
]

log = get_logger(__name__)

#: rows per page: one under the L3 rail's cap, so a page asks for the row that says more follow
DEFAULT_PAGE_SIZE: Final = L3_RAIL_ROW_CAP - 1


class IncompleteCopyError(RuntimeError):
    """the L1 copy is not proven to hold every row of its table, so nothing may read it."""


def _filtered(where: Mapping[str, Any] | None, first: int) -> tuple[str, list[Any]]:
    """the conditions keeping only ``where``'s rows, their placeholders numbered from ``first``.

    :param where: equality filters, column -> value; columns are TRUSTED identifiers
    :ptype where: Mapping[str, Any] | None
    :param first: the first placeholder's number
    :ptype first: int
    :return: the conditions joined by ``AND`` (empty without filters), and their values
    :rtype: tuple[str, list[Any]]
    """
    return equality_conditions(where, first=first)


def _select_from(table: str, columns: Sequence[str]) -> str:
    """``SELECT <columns> FROM <table>``, quoted: what a whole-table read pages over.

    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param columns: the columns, TRUSTED identifiers
    :ptype columns: Sequence[str]
    :return: the statement's head
    :rtype: str
    """
    selected = ", ".join(quote_identifier(c) for c in columns)
    return f"SELECT {selected} FROM {quote_identifier(table)}"  # noqa: S608 - trusted identifiers


async def read_l3_rows(
    l3: L3Reader,
    table: str,
    columns: Sequence[str],
    key: Sequence[str],
    *,
    where: Mapping[str, Any] | None = None,
    where_any: Mapping[str, tuple[str, Sequence[Any]]] | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    lead_array: str | None = None,
) -> list[dict[str, Any]]:
    """every row of an L3 table (or the part ``where`` and ``where_any`` name), in key order, a page at a time.

    The L3 rail answers at most :data:`~threetears.core.backends.protocol.L3_RAIL_ROW_CAP` rows a
    statement and does not say when it cut, so a read that did not page would be silently short:
    this pages through :func:`~threetears.core.keyset.read_keyset_pages`, asking for one row more
    than a page until a page comes back without it.

    **Page in the order an index answers.** Each page is ``ORDER BY`` ``key``. Given the primary key for
    a read filtered on other columns, every page reads and sorts every row the filters leave. A filtered
    read passes :meth:`TableSchema.keyset_order
    <threetears.core.collections.schema_backed.TableSchema.keyset_order>` of its equality-filtered
    columns, the order of the declared index they lead (with the primary key as that index's tail), and
    each page reads about its own rows (:mod:`threetears.core.keyset`).

    :param l3: the L3 backend
    :ptype l3: L3Reader
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param columns: the columns to read, TRUSTED identifiers
    :ptype columns: Sequence[str]
    :param key: the columns to page on, TRUSTED identifiers; unique among the rows read, so paging steps past
        no row: the table's key, or ``TableSchema.keyset_order`` of the filtered columns
    :ptype key: Sequence[str]
    :param where: equality filters naming the rows; every row when None
    :ptype where: Mapping[str, Any] | None
    :param where_any: set filters beside them, column -> (its array type, the values it may hold), each
        ``"column" = ANY($n::<array type>)`` with every value bound at once (:func:`~threetears.core.keyset.read_keyset_pages`)
    :ptype where_any: Mapping[str, tuple[str, Sequence[Any]]] | None
    :param page_size: rows per page; under the rail's cap
    :ptype page_size: int
    :param lead_array: the SQL array type of ``key``'s first column, named when the table's declaration leaves it
        hashed on YugabyteDB (:func:`~threetears.core.keyset.keyset_pages`); None pages by the cursor alone
    :ptype lead_array: str | None
    :return: the rows
    :rtype: list[dict[str, Any]]
    :raises ValueError: when ``page_size`` is under one
    """
    read = await read_keyset_pages(
        l3,
        _select_from(table, columns),
        key,
        where=where,
        where_any=where_any,
        page_size=page_size,
        lead_array=lead_array,
    )
    return read.rows


def l3_row_pages(
    l3: L3Reader,
    table: str,
    columns: Sequence[str],
    key: Sequence[str],
    *,
    where: Mapping[str, Any] | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    lead_array: str | None = None,
) -> AsyncGenerator[list[dict[str, Any]], None]:
    """the rows :func:`read_l3_rows` reads, handed over a page at a time as each is read.

    For a reader that must not hold every row at once: it keeps what
    it needs of each page before the next is asked for. The same statements as :func:`read_l3_rows`,
    in the same order; the last page may be empty.

    :param l3: the L3 backend
    :ptype l3: L3Reader
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param columns: the columns to read, TRUSTED identifiers
    :ptype columns: Sequence[str]
    :param key: the table's key, TRUSTED identifiers; unique, so paging steps past no row
    :ptype key: Sequence[str]
    :param where: equality filters naming the rows; every row when None
    :ptype where: Mapping[str, Any] | None
    :param page_size: rows per page; under the rail's cap
    :ptype page_size: int
    :param lead_array: the SQL array type of ``key``'s first column, named when the table's declaration leaves it
        hashed on YugabyteDB (:func:`~threetears.core.keyset.keyset_pages`); None pages by the cursor alone
    :ptype lead_array: str | None
    :return: the pages, in key order
    :rtype: AsyncGenerator[list[dict[str, Any]], None]
    :raises ValueError: when ``page_size`` is under one (at the first page)
    """
    return keyset_pages(l3, _select_from(table, columns), key, where=where, page_size=page_size, lead_array=lead_array)


#: the L3 column types the rail's JSON answers as text: ``fetch`` answers them as text (a ``NUMERIC`` arrives as
#: its digits, a timestamp outside ``date_*`` as its ISO text)
_RAIL_TEXT_TYPES: Final = frozenset({STRING_TYPE, UUID_TYPE, ENUM_TYPE, TSVECTOR_TYPE, NUMERIC_TYPE, DATETIMETZ_TYPE})

#: the prefix of the columns the rail's proxy answers as aware datetimes (``nats_proxy._deserialize_row``)
_DATED_PREFIX: Final = "date_"


def rail_arrow_types(columns: Sequence[Any]) -> dict[str, Any]:
    """each column's Arrow type as the L3 rail's reply carries it: what :meth:`NatsProxyL3Backend.fetch_arrow
    <threetears.core.backends.nats_proxy.NatsProxyL3Backend.fetch_arrow>` reads the rows as, so they equal ``fetch``'s.

    Text, a UUID, an enum, a ``NUMERIC`` and a timestamp are text (as ``fetch`` answers them); an integer is
    ``int64``; a boolean is ``bool``; a ``date_*`` timestamp is an instant in UTC (``fetch`` makes those aware
    datetimes). A column ``fetch`` answers otherwise than the reply's JSON (``jsonb``, bytes, a vector), or a
    ``date_*`` column that is not a timestamp (``fetch`` parses any ISO text there), has no such type: read it
    with ``fetch``.

    :param columns: the columns, each with ``name`` and ``column_type`` (a
        :class:`~threetears.core.collections.schema_backed.Column`), in the order they are selected
    :ptype columns: Sequence[Column]
    :return: column name -> its Arrow type, in order
    :rtype: dict[str, pyarrow.DataType]
    :raises ValueError: for a column the rail does not answer as one Arrow type exactly
    :raises ImportError: without pyarrow
    """
    if not _PYARROW:
        raise ImportError("an L3 read as Arrow needs pyarrow: install 3tears[snapshot]")
    types: dict[str, Any] = {}
    for column in columns:
        name, kind = column.name, column.column_type
        dated = name.startswith(_DATED_PREFIX)
        if dated and kind != DATETIMETZ_TYPE:
            raise ValueError(
                f"column {name!r}: fetch reads ISO text in a date_ column as a datetime; read it with fetch"
            )
        if dated:
            types[name] = pa.timestamp("us", tz="UTC")
        elif kind in _RAIL_TEXT_TYPES:
            types[name] = pa.string()
        elif kind in (INT_TYPE, BIGINT_TYPE):
            types[name] = pa.int64()
        elif kind == BOOL_TYPE:
            types[name] = pa.bool_()
        else:
            raise ValueError(
                f"column {name!r} ({kind}): the rail answers it otherwise than one Arrow type; read it with fetch"
            )
    return types


async def read_l3_arrow(
    l3: Any,
    table: str,
    columns: Sequence[Any],
    key: Sequence[str],
    *,
    where: Mapping[str, Any] | None = None,
    where_any: Mapping[str, tuple[str, Sequence[Any]]] | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    lead_array: str | None = None,
) -> Any:
    """:func:`read_l3_rows` as one Arrow table: the same statements, a page at a time, the rows never Python objects.

    For a reader that hands the rows on as columns (a worker process, a DuckDB table): each page is parsed by
    Arrow from the rail's reply, off the interpreter lock (:meth:`~threetears.core.backends.nats_proxy.NatsProxyL3Backend.fetch_arrow`),
    and the pages are joined without a copy. ``read_l3_arrow(...).to_pylist()`` equals :func:`read_l3_rows`'s rows
    on the rail. Any other L3's dict rows are made the same table, value by value
    (:func:`~threetears.core.backends.arrow_reply.rail_arrow_table`). Needs ``pyarrow``.

    :param l3: the L3 backend: the rail's proxy, or anything with ``fetch``
    :ptype l3: Any
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param columns: the columns to read, each with ``name`` and ``column_type``, TRUSTED identifiers (:func:`rail_arrow_types`)
    :ptype columns: Sequence[Column]
    :param key: the table's key, TRUSTED identifiers; unique, so paging steps past no row
    :ptype key: Sequence[str]
    :param where: equality filters naming the rows; every row when None
    :ptype where: Mapping[str, Any] | None
    :param where_any: set filters beside them, column -> (its array type, the values it may hold), each
        ``"column" = ANY($n::<array type>)`` with every value bound at once (:func:`~threetears.core.keyset.read_keyset_pages`)
    :ptype where_any: Mapping[str, tuple[str, Sequence[Any]]] | None
    :param page_size: rows per page; under the rail's cap
    :ptype page_size: int
    :param lead_array: the SQL array type of ``key``'s first column, named when the table's declaration leaves it
        hashed on YugabyteDB (:func:`~threetears.core.keyset.keyset_pages`); None pages by the cursor alone
    :ptype lead_array: str | None
    :return: the rows, in key order
    :rtype: pyarrow.Table
    :raises ValueError: when ``page_size`` is under one, or a column has no rail Arrow type
    """
    pages = [
        page
        async for page in _l3_arrow_pages(
            l3, table, columns, key, where=where, where_any=where_any, page_size=page_size, lead_array=lead_array
        )
    ]
    return pa.concat_tables(pages)


def _l3_arrow_pages(
    l3: Any,
    table: str,
    columns: Sequence[Any],
    key: Sequence[str],
    *,
    where: Mapping[str, Any] | None = None,
    where_any: Mapping[str, tuple[str, Sequence[Any]]] | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    lead_array: str | None = None,
) -> AsyncGenerator[Any, None]:
    """the pages :func:`read_l3_arrow` reads, handed over as each is read: :func:`l3_row_pages` as Arrow tables.

    :param l3: the L3 backend: the rail's proxy, or anything with ``fetch``
    :ptype l3: Any
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param columns: the columns to read, each with ``name`` and ``column_type``
    :ptype columns: Sequence[Column]
    :param key: the table's key, TRUSTED identifiers
    :ptype key: Sequence[str]
    :param where: equality filters naming the rows; every row when None
    :ptype where: Mapping[str, Any] | None
    :param where_any: set filters beside them, column -> (its array type, the values it may hold), each
        ``"column" = ANY($n::<array type>)`` with every value bound at once (:func:`~threetears.core.keyset.read_keyset_pages`)
    :ptype where_any: Mapping[str, tuple[str, Sequence[Any]]] | None
    :param page_size: rows per page; under the rail's cap
    :ptype page_size: int
    :param lead_array: the SQL array type of ``key``'s first column, named when the table's declaration leaves it
        hashed on YugabyteDB (:func:`~threetears.core.keyset.keyset_pages`); None pages by the cursor alone
    :ptype lead_array: str | None
    :return: the pages, in key order, each a ``pyarrow.Table``; the last may be empty
    :rtype: AsyncGenerator[pyarrow.Table, None]
    :raises ValueError: when a column has no rail Arrow type
    """
    types = rail_arrow_types(columns)
    return keyset_arrow_pages(
        l3,
        _select_from(table, list(types)),
        key,
        types=types,
        where=where,
        where_any=where_any,
        page_size=page_size,
        lead_array=lead_array,
    )


@dataclass(frozen=True)
class Unsettled:
    """the writer's record while a write is in progress: no stamp to build against, and why.

    A writer that records the parts of the data its write has touched names them, with every part's
    version as last committed: a reader may then read a part the write has not touched, and keep what it
    read only when that part is still untouched, at the same version, once it has read it.

    :ivar reason: what the writer says of the write (which one, how long it has run)
    :ivar epochs: each part's version as last committed, when the writer records touches; ``None`` when it does not
    :ivar pending: the parts the write in progress has touched (its rows there may be committed part way)
    :ivar version: the last committed write's version, when the writer names it
    """

    reason: str
    epochs: Mapping[str, int] | None = None
    pending: frozenset[str] = frozenset()
    #: the last committed write's version, when the writer's record names it
    version: int | None = None
