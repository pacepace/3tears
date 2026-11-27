"""live proof of ``warehouse_arrow_types`` on a real Redshift, read-only: a late-binding view's columns (``WITH NO SCHEMA
BINDING``, which dbt's ``+bind: false`` makes of every view) are answered from ``SVV_COLUMNS``, where
``information_schema.columns`` holds none of them; and a base table's columns are answered alike from either
catalogue, type for type, for a table of every column type the login can read.

Writes nothing: it reads the catalogue (``pg_get_late_binding_view_cols``, ``SVV_COLUMNS``, ``SVV_TABLES``,
``information_schema.columns``, the privilege functions) and finds there the relations it reads, late-binding views
and tables that already exist and the login may read. It skips, saying why, when the warehouse holds none. Gated on
its own environment-named connection, as the fetch_within proof is (``live_redshift_config``, in this directory's
conftest); it skips (fails in CI) when any part is missing.

run locally, from the repo root, with the password injected by reference:

.. code-block:: bash

    THREETEARS_LIVE_REDSHIFT_HOST=<host> THREETEARS_LIVE_REDSHIFT_DATABASE=<db> \\
    THREETEARS_LIVE_REDSHIFT_USER=<user> \\
    THREETEARS_LIVE_REDSHIFT_PASSWORD=op://<vault>/<item>/password \\
      op run -- ./scripts/test.sh datasources \\
      packages/datasources/tests/integration/test_warehouse_arrow_types_redshift_live.py -v -s
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any
from uuid import uuid7

import pytest

from threetears.core.keyed import Dialect
from threetears.datasources.arrow_rows import warehouse_arrow_types, warehouse_arrow_types_of
from threetears.datasources.config import RedshiftConnectionConfig
from threetears.datasources.drivers.redshift_driver import RedshiftDriver
from threetears.datasources.query_client import DatasourceQueryResult

pytestmark = [pytest.mark.integration, pytest.mark.live, pytest.mark.secrets]

#: the late-binding views read: the first few by name in a schema the login may use
VIEWS = 3

#: every late-binding view's columns in the warehouse, in each view's order: the function lists every view, readable
#: by the login or not
_LATE_BINDING_COLUMNS = (
    "SELECT view_schema, view_name, col_name FROM pg_get_late_binding_view_cols() "
    "cols(view_schema name, view_name name, col_name name, col_type varchar, col_num int) "
    "ORDER BY view_schema, view_name, col_num"
)

#: the schemas the login may use: a privilege asked of a relation in any other is refused outright
_USABLE_SCHEMAS = "SELECT nspname FROM pg_namespace WHERE has_schema_privilege(current_user, nspname, 'USAGE')"

#: each column type's first base table by name that the login may read, in the usable schemas (``{first}`` and
#: ``{second}`` each one placeholder a schema, the schemas bound twice): ONE statement answering a row a type. Why one:
#: Redshift's leader answers each catalogue statement in ~5-50 s whatever its scope (the ots warehouse, 2026-11-27:
#: ``list_columns`` over the 8 usable schemas, 1,033,272 rows, 102.8 s; a keyed one-relation read ~25 s), so the old
#: listing and two reads a type (19 types) ran past the 600 s timeout
_FIRST_TABLE_OF_EACH_TYPE = (
    "SELECT c.data_type, MIN(c.table_schema || '.' || c.table_name) AS relation FROM svv_columns c "
    "JOIN (SELECT table_schema, table_name FROM svv_tables WHERE table_schema IN ({first}) "
    "AND table_type = 'BASE TABLE' AND has_table_privilege(current_user, "
    "quote_ident(table_schema) || '.' || quote_ident(table_name), 'SELECT')) t "
    "ON t.table_schema = c.table_schema AND t.table_name = c.table_name "
    "WHERE c.table_schema IN ({second}) GROUP BY c.data_type"
)

#: the system's own schemas, whose tables no reader reads through the rail
_SYSTEM_SCHEMAS = frozenset({"pg_catalog", "information_schema"})


class _Client:
    """the hub's query ask, answered by the driver."""

    def __init__(self, driver: RedshiftDriver) -> None:
        self._driver = driver

    async def query(self, datasource_name: str, sql: str, params: Sequence[Any] = ()) -> DatasourceQueryResult:
        rows = await self._driver.fetch(sql, *params)
        return DatasourceQueryResult(rows=rows, row_count=len(rows), truncated=False, correlation_id=uuid7())


@pytest.fixture
async def driver(live_redshift_config: RedshiftConnectionConfig) -> AsyncIterator[RedshiftDriver]:
    """the environment-named cluster's driver: the client's, and the catalogue reads' that find the relations.

    :param live_redshift_config: the cluster
    :ptype live_redshift_config: RedshiftConnectionConfig
    :return: the driver
    :rtype: AsyncIterator[RedshiftDriver]
    """
    made = RedshiftDriver(live_redshift_config, datasource_name="live-arrow-types")
    try:
        yield made
    finally:
        await made.close()


async def _types(driver: RedshiftDriver, relation: str, dialect: Dialect) -> dict[str, Any]:
    """every column's Arrow type of ``relation``, from the catalogue ``dialect`` names.

    :param relation: ``schema.name``
    :ptype relation: str
    :param dialect: Redshift's (``SVV_COLUMNS``) or PostgreSQL's (``information_schema.columns``, which Redshift holds
        too)
    :ptype dialect: Dialect
    :return: column name -> its Arrow type
    :rtype: dict[str, pyarrow.DataType]
    """
    return await warehouse_arrow_types(_Client(driver), "w", relation, None, dialect=dialect)  # type: ignore[arg-type]


async def _usable(driver: RedshiftDriver) -> set[str]:
    """the schemas the login may use, the system's own left out.

    :return: their names
    :rtype: set[str]
    """
    return {str(row["nspname"]) for row in await driver.fetch(_USABLE_SCHEMAS)} - _SYSTEM_SCHEMAS


async def test_a_late_binding_view_s_columns_are_read_from_the_catalogue_that_holds_them(
    driver: RedshiftDriver,
) -> None:
    """Flips if a late-binding view's columns are read from ``information_schema.columns``, which holds none of them
    (so a load of the view is refused), or are answered from ``SVV_COLUMNS`` as other columns than the view's own
    (``pg_get_late_binding_view_cols``), or not in its order.

    The views read are the first few in a schema the login may use; it need not read them, the catalogue answering for
    any view in such a schema. ``information_schema.columns`` also leaves out a relation the login may not read, so
    its empty answer is the old read's failure, not by itself the proof that a late-binding view is never there; the
    table compared below shows it answering a relation the login may read."""
    usable = await _usable(driver)
    columns: dict[str, list[str]] = {}
    for row in await driver.fetch(_LATE_BINDING_COLUMNS):
        if row["view_schema"] in usable:
            columns.setdefault(f"{row['view_schema']}.{row['view_name']}", []).append(str(row["col_name"]))
    if not columns:
        pytest.skip(f"no late-binding view in a schema the login may use ({sorted(usable)})")

    for relation in sorted(columns)[:VIEWS]:
        held = await _types(driver, relation, Dialect.REDSHIFT)
        standard = await _types(driver, relation, Dialect.POSTGRES)
        print(f"\nLIVE late-binding view {relation}: {list(held)} from SVV_COLUMNS, {list(standard)} elsewhere")
        assert held, f"{relation}: SVV_COLUMNS answered no column of one JSON form"
        own = [name for name in columns[relation] if name in held]
        assert list(held) == own, (
            f"{relation}: {list(held)} are not the view's own columns in order {columns[relation]}"
        )
        assert standard == {}, f"{relation}: information_schema.columns holds its columns"


class _StandardCatalogue:
    """the standard catalogue's answer (``information_schema.columns``) for the tables compared, read once in ONE
    statement and handed to the PostgreSQL read of each, as the hub would answer it: each ~25 s on Redshift (leader
    only), one read a table ran past the timeout. The PostgreSQL statement itself is pinned by the unit tests and run on
    a real PostgreSQL (``test_warehouse_arrow_types_live.py``); what is compared here is Redshift's two spellings."""

    def __init__(self, rows: Sequence[dict[str, Any]]) -> None:
        self._rows = rows

    async def query(self, datasource_name: str, sql: str, params: Sequence[Any] = ()) -> DatasourceQueryResult:
        assert "information_schema.columns" in sql, sql
        # the statement's two IN lists: the schemas, then the names
        schemas = sql.split("table_schema IN (", 1)[1].split(")", 1)[0].count("$")
        rows = [
            row
            for row in self._rows
            if row["table_schema"] in params[:schemas] and row["table_name"] in params[schemas:]
        ]
        return DatasourceQueryResult(rows=rows, row_count=len(rows), truncated=False, correlation_id=uuid7())


async def test_a_table_s_columns_are_the_same_types_from_either_catalogue(driver: RedshiftDriver) -> None:
    """Flips if ``SVV_COLUMNS`` spells a base table's column type other than ``information_schema.columns`` does in a
    way that changes its JSON form (a column left out, or another Arrow type), for a table of each column type the
    login may read: the Redshift read must answer a table as the standard catalogue always did. The tables are each
    column type's first base table by name the login may read (:data:`_FIRST_TABLE_OF_EACH_TYPE`, one statement);
    they are read from SVV_COLUMNS in one batched read (``warehouse_arrow_types_of``), and from the standard catalogue
    in one statement: four catalogue statements in all, whatever the number of types."""
    usable = sorted(await _usable(driver))
    if not usable:
        pytest.skip("no schema the login may use")
    first_marks = ", ".join(f"${index + 1}" for index in range(len(usable)))
    second_marks = ", ".join(f"${len(usable) + index + 1}" for index in range(len(usable)))
    first = {
        str(row["data_type"]): str(row["relation"])
        for row in await driver.fetch(
            _FIRST_TABLE_OF_EACH_TYPE.format(first=first_marks, second=second_marks), *usable, *usable
        )
    }
    if not first:
        pytest.skip(f"no base table the login may read in {usable}")
    relations = sorted(set(first.values()))
    # two IN lists, never the pairs ORed (the leader took 153.3 s for 46 relations' pairs ORed, 16.2 s as IN lists);
    # the rows of a pair no relation is are left out by the PostgreSQL read below, as the batched read leaves them
    schemas = sorted({relation.split(".", 1)[0] for relation in relations})
    names = sorted({relation.split(".", 1)[1] for relation in relations})
    schema_marks = ", ".join(f"${index + 1}" for index in range(len(schemas)))
    name_marks = ", ".join(f"${len(schemas) + index + 1}" for index in range(len(names)))
    standard_rows = await driver.fetch(
        "SELECT table_schema, table_name, column_name, data_type FROM information_schema.columns "
        f"WHERE table_schema IN ({schema_marks}) AND table_name IN ({name_marks}) "
        "ORDER BY table_schema, table_name, ordinal_position",
        *schemas,
        *names,
    )

    held = await warehouse_arrow_types_of(_Client(driver), "w", relations, dialect=Dialect.REDSHIFT)  # type: ignore[arg-type]
    standard_client = _StandardCatalogue(standard_rows)
    for kind, relation in sorted(first.items()):
        standard = await warehouse_arrow_types(
            standard_client,  # type: ignore[arg-type]
            "w",
            relation,
            None,
            dialect=Dialect.POSTGRES,
        )
        assert standard, f"{relation} ({kind}): information_schema.columns holds no column of one JSON form"
        assert held[relation] == standard, (
            f"{relation} ({kind}): SVV_COLUMNS {held[relation]} != information_schema.columns {standard}"
        )
        assert list(held[relation]) == list(standard), f"{relation} ({kind}): the columns' order differs"
    print(f"\nLIVE column types compared, {len(relations)} tables covering {len(first)} types: {first}")
