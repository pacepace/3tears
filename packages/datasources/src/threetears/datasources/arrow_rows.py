"""A datasource read's rows as Arrow: each column's type as the rail's reply carries it, and pages joined into one table.

**Why the warehouse's own types, not the caller's.** :meth:`~threetears.datasources.query_client.DatasourceQueryClient.query_arrow`
reads a reply's rows as the Arrow types it is given, and those must be the types of the reply's JSON values, which the
warehouse's column decides, never the type a caller keeps the value as: a warehouse ``integer`` a caller keeps as text
arrives as a number, a warehouse ``numeric`` arrives as text (the driver's ``Decimal``, written by pydantic as its
digits) and a ``double precision`` as a number. So :func:`warehouse_arrow_types` asks the warehouse's catalogue
once per relation (:func:`_columns_sql`: ``SVV_COLUMNS`` on Redshift, ``information_schema.columns`` elsewhere), and
maps each column's type to its JSON's. :func:`relation_columns_sql` gives the same catalogue statement with the
relation written in as literals, for a caller whose query binds nothing. A value of another form than its column's type is refused by the read
(:class:`~threetears.core.backends.arrow_reply.ReplyNotArrowError`), never answered otherwise than the JSON read would.

Needs ``pyarrow`` (the ``export`` extra).
"""

from __future__ import annotations

import re

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, assert_never

try:
    import pyarrow as pa

    _PYARROW = True
except ImportError:  # NOSILENT: the export extra is optional; an Arrow read says so without it
    _PYARROW = False

from threetears.core.backends.schema_sql import MAX_STATEMENT_PARAMS
from threetears.core.keyed import Dialect
from threetears.datasources.export import sql_string_literal

if TYPE_CHECKING:
    from threetears.datasources.query_client import DatasourceQueryClient

__all__ = ["joined_arrow", "relation_columns_sql", "warehouse_arrow_types", "warehouse_arrow_types_of"]

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

    Its ``$1``/``$2`` spelling is a contract: :func:`relation_columns_sql` writes literals in for exactly those two
    placeholders, in one pass.

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


#: the relations' columns of a batch read (:func:`warehouse_arrow_types_of`), each relation's in ordinal order: the
#: catalogue :func:`_columns_sql` names (``svv_columns`` on Redshift, ``information_schema.columns`` elsewhere), asked
#: for every relation of the batch in ONE statement. Why one: Redshift's leader answers each catalogue statement in
#: roughly 5-50 s whatever its scope (the ots warehouse, 2026-11-27: one relation's svv_columns read 26.6 s, the
#: catalogue's own pg_attribute rows keyed to one relation median 23.9 s and max 51.7 s over 46 relations, all 46
#: relations' svv_columns in one statement 16.2 s), so a pod that read its relations one statement each paid ~25 s a
#: relation at every cold start. The schemas and names are two IN lists, never the (schema, name) pairs ORed: the
#: same 46 relations' pairs ORed took 153.3 s (2026-11-27), against 16.2 s as IN lists. The lists may pair a schema with
#: a name of another relation; :func:`_held_columns` keeps only the rows of a relation asked
_BATCH_COLUMNS_SQL: Final = (
    "SELECT table_schema, table_name, column_name, data_type FROM {catalogue} "
    "WHERE table_schema IN ({schemas}) AND table_name IN ({names}) "
    "ORDER BY table_schema, table_name, ordinal_position"
)


def _catalogue_of(dialect: Dialect) -> str:
    """the catalogue view holding every relation's columns in ``dialect`` (:func:`_columns_sql`'s).

    :param dialect: the dialect the datasource's statements are spelled in
    :ptype dialect: Dialect
    :return: ``svv_columns`` on Redshift, ``information_schema.columns`` elsewhere
    :rtype: str
    :raises AssertionError: for anything not a :class:`Dialect` member
    """
    if dialect is Dialect.REDSHIFT:
        return "svv_columns"
    if dialect is Dialect.POSTGRES or dialect is Dialect.YUGABYTE or dialect is Dialect.DUCKDB:
        return "information_schema.columns"
    assert_never(dialect)


def _batch_columns_sql(schemas: int, names: int, *, dialect: Dialect) -> str:
    """the statement reading the columns and types of every relation of ``schemas`` schemas and ``names`` names in
    one ask (``$1`` to ``$<schemas>`` the schemas, the names after them), each relation's in ordinal order.

    :param schemas: the distinct schemas asked, at least one
    :ptype schemas: int
    :param names: the distinct names asked, at least one
    :ptype names: int
    :param dialect: the dialect the datasource's statements are spelled in
    :ptype dialect: Dialect
    :return: the statement, answering ``table_schema``, ``table_name``, ``column_name`` and ``data_type``
    :rtype: str
    :raises ValueError: for no schema or name, or more parameters than one statement binds
    """
    if schemas < 1 or names < 1 or schemas + names > MAX_STATEMENT_PARAMS:
        raise ValueError(
            f"a batch catalogue read binds 1 or more schemas and names, {MAX_STATEMENT_PARAMS} at most: "
            f"not {schemas} and {names}"
        )
    return _BATCH_COLUMNS_SQL.format(
        catalogue=_catalogue_of(dialect),
        schemas=", ".join(f"${index + 1}" for index in range(schemas)),
        names=", ".join(f"${schemas + index + 1}" for index in range(names)),
    )


def _statements(pairs: Sequence[tuple[str, str]]) -> list[tuple[list[str], list[str]]]:
    """the batch's statements: each one's distinct schemas and names, as many relations a statement as the wire
    protocol's 16-bit parameter count binds (:data:`~threetears.core.backends.schema_sql.MAX_STATEMENT_PARAMS`, a
    schema and a name each bound once a statement), the one limit that splits a batch.

    :param pairs: each relation's schema and name, each once
    :ptype pairs: Sequence[tuple[str, str]]
    :return: each statement's schemas and names, in the order first asked
    :rtype: list[tuple[list[str], list[str]]]
    """
    statements: list[tuple[list[str], list[str]]] = []
    schemas: dict[str, None] = {}
    names: dict[str, None] = {}
    for schema, name in pairs:
        grown = len(schemas) + (schema not in schemas) + len(names) + (name not in names)
        if grown > MAX_STATEMENT_PARAMS:
            statements.append((list(schemas), list(names)))
            schemas, names = {}, {}
        schemas[schema] = None
        names[name] = None
    if schemas:
        statements.append((list(schemas), list(names)))
    return statements


async def _held_columns(
    client: DatasourceQueryClient, datasource_name: str, relations: Sequence[str], *, dialect: Dialect
) -> dict[str, dict[str, str]]:
    """each relation's columns and their catalogue types, in ordinal order, from one statement a batch
    (:func:`_batch_columns_sql`): the one place :func:`warehouse_arrow_types` and :func:`warehouse_arrow_types_of`
    read the catalogue.

    :param client: the query client
    :ptype client: DatasourceQueryClient
    :param datasource_name: the datasource
    :ptype datasource_name: str
    :param relations: the relations, ``schema.table`` (the engine's default schema when one names none), TRUSTED
        identifiers, each once
    :ptype relations: Sequence[str]
    :param dialect: the dialect the datasource's statements are spelled in
    :ptype dialect: Dialect
    :return: relation -> (column name -> its catalogue type), every relation asked, empty when the catalogue holds none
    :rtype: dict[str, dict[str, str]]
    """
    named = {relation: _schema_and_name(relation, dialect) for relation in relations}
    # by (schema, name), not by relation: two names of one pair (``orders`` and ``public.orders``) each answer its
    # columns, from the one ask of that pair
    by_pair: dict[tuple[str, str], dict[str, str]] = {pair: {} for pair in named.values()}
    for schemas, names in _statements(list(by_pair)):
        result = await client.query(
            datasource_name,
            _batch_columns_sql(len(schemas), len(names), dialect=dialect),
            params=[*schemas, *names],
        )
        for row in result.rows:
            # a row of a schema and name the lists pair but no relation asked is not kept
            held = by_pair.get((str(row["table_schema"]), str(row["table_name"])))
            if held is not None:
                held[str(row["column_name"])] = str(row["data_type"])
    return {relation: dict(by_pair[pair]) for relation, pair in named.items()}


def _schema_and_name(relation: str, dialect: Dialect) -> tuple[str, str]:
    """``relation``'s schema and name: the engine's default schema when it names none (``main`` on DuckDB,
    ``public`` elsewhere), so a bare name reads the schema the engine would resolve it in.

    :param relation: the relation, ``schema.table`` or a bare name
    :ptype relation: str
    :param dialect: the dialect the warehouse's statements are spelled in
    :ptype dialect: Dialect
    :return: the schema and the name
    :rtype: tuple[str, str]
    """
    schema, _, table = relation.rpartition(".")
    return schema or ("main" if dialect is Dialect.DUCKDB else "public"), table


#: the characters a quoted literal cannot carry by doubling quotes alone (with ``standard_conforming_strings`` off a
#: backslash escapes; a NUL ends the string)
_UNQUOTABLE: Final = ("\\", "\x00")

#: the catalogue statement's placeholders, replaced in ONE pass so a literal written in is never scanned again (a
#: schema named ``rpt$2x`` must stay that name, not have ``$2`` inside it replaced by the table's literal)
_PLACEHOLDER: Final = re.compile(r"\$([12])")


def relation_columns_sql(relation: str, *, dialect: Dialect) -> str:
    """the statement reading ``relation``'s columns and their types, in ordinal order, from the catalogue that holds
    every relation's columns in ``dialect`` (:func:`_columns_sql`: on Redshift ``SVV_COLUMNS``, which holds a
    late-binding view's columns too), its schema and name written in as quoted literals: for a caller whose query
    binds nothing (an operator's check holding one login's ``fetch``), which would otherwise hand-write
    ``information_schema.columns`` and find no column of a late-binding view.

    :param relation: the relation, ``schema.table`` (the engine's default schema when it names none), a TRUSTED
        identifier: the caller's own constant, never a reader's input (the literals are quoted and refused when no
        quoting carries them, but a relation name is never taken from a request)
    :ptype relation: str
    :param dialect: the dialect the warehouse's statements are spelled in, from its declared engine
    :ptype dialect: Dialect
    :return: the statement, answering ``column_name`` and ``data_type``
    :rtype: str
    :raises AssertionError: for anything not a :class:`Dialect` member
    :raises ValueError: when the schema or name holds a backslash or a NUL, which no quoted literal carries safely
    """
    schema, table = _schema_and_name(relation, dialect)
    for part, text in (("schema", schema), ("name", table)):
        if any(character in text for character in _UNQUOTABLE):
            raise ValueError(
                f"{relation!r}: the relation's {part} holds a backslash or a NUL, which a quoted literal cannot carry"
            )
    literals = {"1": sql_string_literal(schema), "2": sql_string_literal(table)}
    return _PLACEHOLDER.sub(lambda found: literals[found.group(1)], _columns_sql(dialect))


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
    held = (await _held_columns(client, datasource_name, [relation], dialect=dialect))[relation]
    return _types_of(relation, held, columns, extra)


async def warehouse_arrow_types_of(
    client: DatasourceQueryClient,
    datasource_name: str,
    relations: Sequence[str],
    *,
    dialect: Dialect,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """:func:`warehouse_arrow_types` of every column of many relations (``columns=None`` for each), from ONE catalogue
    statement for them all (:data:`_BATCH_COLUMNS_SQL`'s why: on Redshift each catalogue statement costs ~25 s whatever
    its scope), split only past the wire protocol's parameter count (a statement binds each distinct schema and name
    once, :data:`~threetears.core.backends.schema_sql.MAX_STATEMENT_PARAMS` at most).

    Each relation's answer is the one :func:`warehouse_arrow_types` gives it alone: its columns of one JSON form, in
    the catalogue's order, ``extra``'s after them; a column of another type left out; a relation the catalogue holds
    no column of answers ``extra`` alone.

    :param client: the query client
    :ptype client: DatasourceQueryClient
    :param datasource_name: the datasource
    :ptype datasource_name: str
    :param relations: the relations, ``schema.table``, TRUSTED identifiers (a repeat is asked once)
    :ptype relations: Sequence[str]
    :param dialect: the dialect the datasource's statements are spelled in, from its declared engine
    :ptype dialect: Dialect
    :param extra: the types of selected expressions the catalogue does not hold (a row hash), by their names, after
        each relation's columns
    :ptype extra: Mapping[str, pyarrow.DataType] | None
    :return: relation -> (column name -> its Arrow type, in order), every relation asked
    :rtype: dict[str, dict[str, pyarrow.DataType]]
    :raises AssertionError: when ``dialect`` is not a :class:`Dialect` member (a plain string)
    :raises ImportError: without pyarrow
    """
    if not _PYARROW:
        raise ImportError("a datasource read as Arrow needs pyarrow: install 3tears-datasources[export]")
    unique = list(dict.fromkeys(relations))
    if not unique:
        return {}
    held = await _held_columns(client, datasource_name, unique, dialect=dialect)
    return {relation: _types_of(relation, held[relation], None, extra) for relation in unique}


def _types_of(
    relation: str, held: Mapping[str, str], columns: Sequence[str] | None, extra: Mapping[str, Any] | None
) -> dict[str, Any]:
    """each column's Arrow type from the relation's catalogue rows (:func:`warehouse_arrow_types`' answer).

    :param relation: the relation, for a refusal's message
    :ptype relation: str
    :param held: column name -> its catalogue type, in ordinal order
    :ptype held: Mapping[str, str]
    :param columns: the columns read, in order; None for every column of one JSON form
    :ptype columns: Sequence[str] | None
    :param extra: the types of selected expressions after them
    :ptype extra: Mapping[str, pyarrow.DataType] | None
    :return: column name -> its Arrow type, in order
    :rtype: dict[str, pyarrow.DataType]
    :raises ValueError: when a column is not in the catalogue, or its type has no one JSON form here
    """
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
