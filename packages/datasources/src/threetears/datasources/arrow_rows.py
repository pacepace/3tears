"""A datasource read's rows as Arrow: each column's type as the rail's reply carries it, and pages joined into one table.

**Why the warehouse's own types, not the caller's.** :meth:`~threetears.datasources.query_client.DatasourceQueryClient.query_arrow`
reads a reply's rows as the Arrow types it is given, and those must be the types of the reply's JSON values, which the
warehouse's column decides, never the type a caller keeps the value as: a warehouse ``integer`` a caller keeps as text
arrives as a number, a warehouse ``numeric`` arrives as text (the driver's ``Decimal``, written by pydantic as its
digits) and a ``double precision`` as a number. So :func:`warehouse_arrow_types` asks the warehouse's catalogue
(``information_schema.columns``, which Redshift and PostgreSQL both hold) once per relation, and maps each column's
type to its JSON's. A value of another form than its column's type is refused by the read
(:class:`~threetears.core.backends.arrow_reply.ReplyNotArrowError`), never answered otherwise than the JSON read would.

Needs ``pyarrow`` (the ``export`` extra).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final

try:
    import pyarrow as pa

    _PYARROW = True
except ImportError:  # NOSILENT: the export extra is optional; an Arrow read says so without it
    _PYARROW = False

if TYPE_CHECKING:
    from threetears.datasources.query_client import DatasourceQueryClient

__all__ = ["joined_arrow", "warehouse_arrow_types"]

#: each warehouse column type (``information_schema.columns.data_type``, Redshift's and PostgreSQL's spelling) onto
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
    "smallint": "int64",
    "integer": "int64",
    "bigint": "int64",
    "real": "float64",
    "double precision": "float64",
    "boolean": "bool",
}


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
    columns: Sequence[str],
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """each column's Arrow type as a read of ``relation`` through the rail carries it, from the warehouse's catalogue.

    One small read of ``information_schema.columns`` (the relation's columns and their types). A caller reads it once
    per relation and keeps it while the relation's columns hold their types; a read refused for a column's type
    (:class:`~threetears.core.backends.arrow_reply.ReplyNotArrowError`) is the sign they moved, and the caller asks
    again.

    :param client: the query client
    :ptype client: DatasourceQueryClient
    :param datasource_name: the datasource
    :ptype datasource_name: str
    :param relation: the relation, ``schema.table``, a TRUSTED identifier
    :ptype relation: str
    :param columns: the columns read, in the order selected
    :ptype columns: Sequence[str]
    :param extra: the types of selected expressions the catalogue does not hold (a row hash), by their names, which
        follow ``columns`` in this order
    :ptype extra: Mapping[str, pyarrow.DataType] | None
    :return: column name -> its Arrow type, in order
    :rtype: dict[str, pyarrow.DataType]
    :raises ValueError: when a column is not in the catalogue, or its type has no one JSON form here
    :raises ImportError: without pyarrow
    """
    if not _PYARROW:
        raise ImportError("a datasource read as Arrow needs pyarrow: install 3tears-datasources[export]")
    schema, _, table = relation.rpartition(".")
    result = await client.query(
        datasource_name,
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_schema = $1 AND table_name = $2 ORDER BY ordinal_position",
        params=[schema or "public", table],
    )
    held = {str(row["column_name"]): str(row["data_type"]) for row in result.rows}
    types: dict[str, Any] = {}
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
