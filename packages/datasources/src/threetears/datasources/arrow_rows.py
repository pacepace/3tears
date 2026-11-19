"""A datasource read's rows as Arrow: each column's type as the rail's reply carries it, and pages joined into one table.

**Why the warehouse's own types, not the caller's.** :meth:`~threetears.datasources.query_client.DatasourceQueryClient.query_arrow`
reads a reply's rows as the Arrow types it is given, and those must be the types of the reply's JSON values, which the
warehouse's column decides, never the type a caller keeps the value as: a warehouse ``integer`` a caller keeps as text
arrives as a number, a warehouse ``numeric`` arrives as text (the driver's ``Decimal``, written by pydantic as its
digits) and a ``double precision`` as a number. So :func:`warehouse_arrow_types` asks the warehouse's catalogue
once per relation (:func:`_columns_sql`: ``SVV_COLUMNS`` on Redshift, ``information_schema.columns`` elsewhere), and
maps each column's type to its JSON's. A value of another form than its column's type is refused by the read
(:class:`~threetears.core.backends.arrow_reply.ReplyNotArrowError`), never answered otherwise than the JSON read would.

Needs ``pyarrow`` (the ``export`` extra).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, assert_never

try:
    import pyarrow as pa

    _PYARROW = True
except ImportError:  # NOSILENT: the export extra is optional; an Arrow read says so without it
    _PYARROW = False

from threetears.core.keyed import Dialect
from threetears.datasources.export import sql_string_literal

if TYPE_CHECKING:
    from threetears.datasources.query_client import DatasourceQueryClient

__all__ = ["joined_arrow", "relation_columns_sql", "warehouse_arrow_types"]

#: each warehouse column type (the catalogue's ``data_type``: :func:`_columns_sql`'s, Redshift's and PostgreSQL's
#: spelling) onto
#: the form its value takes in the rail's JSON reply: the hub writes the driver's values with pydantic, so text, a
#: ``Decimal``, a date, a time and a UUID are text, an integer is a number with no point, a float a number with one,
#: a boolean a boolean
_JSON_FORMS: Final[Mapping[str, str]] = {
    "character varying": "string",
    "character": "string",
    "text": "string",
    "uuid": "string",
    "numeric": "string",
    "date": "string",
    "timestamp without time zone": "string",
    "timestamp with time zone": "string",
    "time without time zone": "string",
    "time with time zone": "string",
    # SVV_COLUMNS' spelling of a Redshift table's time with time zone column, where information_schema.columns and a
    # late-binding view's SVV_COLUMNS row spell it long (read on the warehouse, 2026-11-19). Every column the two
    # catalogues both held for the ots login that day had one spelling in both; a table of each type it may read is
    # compared live (test_warehouse_arrow_types_redshift_live.py)
    "timetz": "string",
    "smallint": "int64",
    "integer": "int64",
    "bigint": "int64",
    "real": "float64",
    "double precision": "float64",
    "boolean": "bool",
}


#: a relation's columns and their types, in ordinal order, from the catalogue that holds them on Redshift: SVV_COLUMNS
#: holds a table's, a bound view's and a late-binding view's columns alike. information_schema.columns holds no column
#: of a late-binding view (``WITH NO SCHEMA BINDING``, what dbt's ``+bind: false`` makes of every view, so a table
#: rebuilt dated under a view of its old name), and a read of it there found nothing and refused the relation's every
#: load (read on the warehouse, 2026-11-19: the view's columns in SVV_COLUMNS, none in information_schema.columns).
#: SVV_COLUMNS is also the system view the Redshift driver's own column reads take, which answers in seconds where
#: information_schema.columns is leader-only and slow (threetears.datasources.drivers.redshift_driver)
_REDSHIFT_COLUMNS_SQL: Final = (
    "SELECT column_name, data_type FROM svv_columns "
    "WHERE table_schema = $1 AND table_name = $2 ORDER BY ordinal_position"
)

#: the same, on PostgreSQL, YugabyteDB and DuckDB, which hold no SVV_COLUMNS: the standard catalogue holds a table's
#: and a view's columns alike there
_STANDARD_COLUMNS_SQL: Final = (
    "SELECT column_name, data_type FROM information_schema.columns "
    "WHERE table_schema = $1 AND table_name = $2 ORDER BY ordinal_position"
)


def _columns_sql(dialect: Dialect) -> str:
    """the statement reading one relation's columns and their types (``$1`` its schema, ``$2`` its name), in ordinal
    order, from the catalogue that holds every relation's columns in ``dialect``: the one place a warehouse
    relation's columns are read over the rail.

    Each member is matched by identity, so a plain string (``"redshift"``) never passes for one, and a member added to
    :class:`Dialect` without a catalogue here is a type error (``assert_never``).

    :param dialect: the dialect the datasource's statements are spelled in, from its declared engine
    :ptype dialect: Dialect
    :return: the statement
    :rtype: str
    :raises AssertionError: for anything not a :class:`Dialect` member
    """
    if dialect is Dialect.REDSHIFT:
        return _REDSHIFT_COLUMNS_SQL
    if dialect is Dialect.POSTGRES or dialect is Dialect.YUGABYTE or dialect is Dialect.DUCKDB:
        return _STANDARD_COLUMNS_SQL
    assert_never(dialect)


def relation_columns_sql(relation: str, *, dialect: Dialect) -> str:
    """the statement reading ``relation``'s columns and their types, in ordinal order, from the catalogue that holds
    every relation's columns in ``dialect`` (:func:`_columns_sql`: on Redshift ``SVV_COLUMNS``, which holds a
    late-binding view's columns too), its schema and name written in as quoted literals: for a caller whose query
    binds nothing (an operator's check holding one login's ``fetch``), which would otherwise hand-write
    ``information_schema.columns`` and find no column of a late-binding view.

    :param relation: the relation, ``schema.table`` (``public`` when it names no schema), a TRUSTED identifier
    :ptype relation: str
    :param dialect: the dialect the warehouse's statements are spelled in, from its declared engine
    :ptype dialect: Dialect
    :return: the statement, answering ``column_name`` and ``data_type``
    :rtype: str
    :raises AssertionError: for anything not a :class:`Dialect` member
    :raises ExportRefusedError: when the schema or name holds a backslash or a NUL
    """
    schema, _, table = relation.rpartition(".")
    bound = _columns_sql(dialect)
    return bound.replace("$1", sql_string_literal(schema or "public")).replace("$2", sql_string_literal(table))


def _arrow_of(form: str) -> Any:
    """the Arrow type of a JSON form (:data:`_JSON_FORMS`' values).

    :param form: ``string``, ``int64``, ``float64`` or ``bool``
    :ptype form: str
    :return: the type
    :rtype: pyarrow.DataType
    :raises ImportError: without pyarrow
    """
    if not _PYARROW:
        raise ImportError("a datasource read as Arrow needs pyarrow: install 3tears-datasources[export]")
    kinds = {"string": pa.string(), "int64": pa.int64(), "float64": pa.float64(), "bool": pa.bool_()}
    return kinds[form]


async def warehouse_arrow_types(
    client: DatasourceQueryClient,
    datasource_name: str,
    relation: str,
    columns: Sequence[str] | None,
    *,
    dialect: Dialect,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """each column's Arrow type as a read of ``relation`` through the rail carries it, from the warehouse's catalogue.

    One small read of the warehouse's catalogue (the relation's columns and their types, :func:`_columns_sql`: on
    Redshift ``SVV_COLUMNS``, which holds a late-binding view's columns too). A caller reads it once
    per relation and keeps it while the relation's columns hold their types; a read refused for a column's type
    (:class:`~threetears.core.backends.arrow_reply.ReplyNotArrowError`) is the sign they moved, and the caller asks
    again.

    :param client: the query client
    :ptype client: DatasourceQueryClient
    :param datasource_name: the datasource
    :ptype datasource_name: str
    :param relation: the relation, ``schema.table``, a TRUSTED identifier
    :ptype relation: str
    :param columns: the columns read, in the order selected; None for every column of the relation whose type has one
        JSON form, in the catalogue's order (a caller that keeps one answer per relation for all its reads, so the
        catalogue is read once, not once per select list); a column of another type is then left out, and refused only
        when a read names it
    :ptype columns: Sequence[str] | None
    :param dialect: the dialect the datasource's statements are spelled in, from its declared engine
        (:func:`threetears.core.keyed.dialect_of` over the datasource's
        :class:`~threetears.datasources.keyed_reader.DatasourceReader`, which refuses an engine with none); it names
        the catalogue read, never learned by a statement failing
    :ptype dialect: Dialect
    :param extra: the types of selected expressions the catalogue does not hold (a row hash), by their names, which
        follow ``columns`` in this order
    :ptype extra: Mapping[str, pyarrow.DataType] | None
    :return: column name -> its Arrow type, in order
    :rtype: dict[str, pyarrow.DataType]
    :raises ValueError: when a column is not in the catalogue, or its type has no one JSON form here
    :raises AssertionError: when ``dialect`` is not a :class:`Dialect` member (a plain string)
    :raises ImportError: without pyarrow
    """
    if not _PYARROW:
        raise ImportError("a datasource read as Arrow needs pyarrow: install 3tears-datasources[export]")
    schema, _, table = relation.rpartition(".")
    result = await client.query(datasource_name, _columns_sql(dialect), params=[schema or "public", table])
    held = {str(row["column_name"]): str(row["data_type"]) for row in result.rows}
    types: dict[str, Any] = {}
    if columns is None:
        types = {name: _arrow_of(_JSON_FORMS[kind]) for name, kind in held.items() if kind in _JSON_FORMS}
        types.update(extra or {})
        return types
    for name in columns:
        kind = held.get(name)
        if kind is None:
            raise ValueError(f"{relation}: the warehouse's catalogue holds no column {name!r}")
        form = _JSON_FORMS.get(kind)
        if form is None:
            raise ValueError(f"{relation}.{name}: a {kind!r} column has no one JSON form on the rail; read it as dicts")
        types[name] = _arrow_of(form)
    types.update(extra or {})
    return types


def joined_arrow(pieces: Sequence[Any], types: Mapping[str, Any]) -> Any:
    """pages' Arrow tables as one table, joined without a copy; an empty table of ``types`` when there are none.

    :param pieces: the tables, each of ``types``
    :ptype pieces: Sequence[pyarrow.Table]
    :param types: column name -> its Arrow type, in order
    :ptype types: Mapping[str, pyarrow.DataType]
    :return: the rows
    :rtype: pyarrow.Table
    :raises ImportError: without pyarrow
    """
    if not _PYARROW:
        raise ImportError("a datasource read as Arrow needs pyarrow: install 3tears-datasources[export]")
    if not pieces:
        return pa.schema([pa.field(name, kind) for name, kind in types.items()]).empty_table()
    return pa.concat_tables(pieces)
