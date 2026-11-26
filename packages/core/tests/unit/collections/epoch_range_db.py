"""a database holding one versioned and one stamped table, as an epoch-range reader finds them: for the readers that
read the change-note ring until 0.95 (append-only L3, chunk 7) and now read the tables' own versions.

The data statements run in an in-process DuckDB (the keyed helpers' SQL is one spelling there and on PostgreSQL); the
catalog statements a reader sends (the restore declarations of ``pg_description``, the key types of ``pg_attribute``)
are answered from the tables' own declarations, as ``TableSchema.create_statements`` writes them. Every statement is
recorded with its arguments, for a test's statement spy and the no-whole-reads judgement.
"""

from __future__ import annotations

import re
from typing import Any, Final

import duckdb

from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column, TableSchema
from threetears.core.ddl.row_epoch import ROW_STAMPED, row_versions
from threetears.core.ddl.run_markers import RUN_MARKERS_TABLE
from threetears.core.eviction import Declared, Policy
from threetears.core.keyed import ThreadedDuckDB
from threetears.core.tenancy import TENANT_BY_SCHEMA

#: the platform schema, which holds the run markers' and the scope epochs' tables beside the data
SCHEMA: Final = "platform"
VERSIONED: Final = "range_results"
STAMPED: Final = "range_runs"
EPOCHS: Final = "range_epochs"
SCOPE: Final = "join_state_code"

#: the versioned table a consumer declares: its key, its scope, the scope epochs table naming its states
RESULTS: Final = TableSchema(
    name=VERSIONED,
    primary_key=(SCOPE, "run_id", "candidate"),
    columns=[
        Column(SCOPE, STRING_TYPE),
        Column("run_id", STRING_TYPE),
        Column("candidate", STRING_TYPE),
        Column("votes", BIGINT_TYPE),
    ],
    tenancy=TENANT_BY_SCHEMA,
    row_epoch=row_versions(SCOPE, epochs=EPOCHS, scope_prefix="state:"),
    l1_eviction=Declared(Policy.EVICTABLE, "a test's versioned table"),
)

#: a stamped table a consumer declares: one row a key, updated in place
RUNS: Final = TableSchema(
    name=STAMPED,
    primary_key=(SCOPE, "run_id"),
    columns=[Column(SCOPE, STRING_TYPE), Column("run_id", STRING_TYPE), Column("rows", BIGINT_TYPE)],
    tenancy=TENANT_BY_SCHEMA,
    row_epoch=ROW_STAMPED,
    l1_eviction=Declared(Policy.EVICTABLE, "a test's stamped table"),
)


def declaration(schema: TableSchema) -> str:
    """the restore declaration the table's DDL gives its restore index, as the catalog then holds it.

    :param schema: the table
    :ptype schema: TableSchema
    :return: the declaration's text
    :rtype: str
    """
    [comment] = [statement for statement in schema.create_statements(yugabyte=True) if statement.startswith("COMMENT")]
    found = re.search(r"IS '(.*)'$", comment)
    assert found is not None, comment
    return found.group(1).replace("''", "'")


class RangeDb:
    """the database: the two tables, their scope epochs and run markers in DuckDB, the catalog answered here.

    :ivar sent: every statement sent, with its arguments
    :ivar declared: the tables the catalog declares, by name
    """

    def __init__(self) -> None:
        self._conn = duckdb.connect()
        self._conn.execute(f"CREATE SCHEMA {SCHEMA}")
        self._conn.execute(
            f"CREATE TABLE {SCHEMA}.{VERSIONED} ({SCOPE} TEXT, run_id TEXT, candidate TEXT, votes BIGINT, "
            "epoch BIGINT NOT NULL, gone BOOLEAN NOT NULL DEFAULT false)"
        )
        self._conn.execute(
            f"CREATE TABLE {SCHEMA}.{STAMPED} ({SCOPE} TEXT, run_id TEXT, rows BIGINT, epoch BIGINT NOT NULL, "
            "epoch_bucket SMALLINT NOT NULL, gone BOOLEAN NOT NULL DEFAULT false)"
        )
        self._conn.execute(f"CREATE TABLE {SCHEMA}.{EPOCHS} (scope TEXT, last_epoch BIGINT)")
        self._conn.execute(
            f"CREATE TABLE {SCHEMA}.{RUN_MARKERS_TABLE} (table_name TEXT, scope TEXT, prefix TEXT, epoch BIGINT)"
        )
        self._db = ThreadedDuckDB(self._conn)
        self.sent: list[tuple[str, tuple[Any, ...]]] = []
        self.declared: dict[str, TableSchema] = {VERSIONED: RESULTS, STAMPED: RUNS}

    def version(self, state: str, run: str, candidate: str, epoch: int, *, gone: bool = False) -> None:
        """a version of the versioned table, its scope touched at its epoch.

        :param state: its state
        :ptype state: str
        :param run: its run
        :ptype run: str
        :param candidate: its candidate
        :ptype candidate: str
        :param epoch: its stamp
        :ptype epoch: int
        :param gone: whether it is a gone marker
        :ptype gone: bool
        :return: nothing
        :rtype: None
        """
        self._conn.execute(
            f"INSERT INTO {SCHEMA}.{VERSIONED} VALUES (?, ?, ?, 1, ?, ?)", [state, run, candidate, epoch, gone]
        )
        self.touch(f"state:{state}", epoch)

    def bulk_versions(self, state: str, run: str, count: int, epoch: int) -> None:
        """``count`` versions of the versioned table in one run, candidates ``c0`` on, all at one epoch (one large
        commit's), its scope touched.

        :param state: their state
        :ptype state: str
        :param run: their run
        :ptype run: str
        :param count: how many
        :ptype count: int
        :param epoch: their stamp
        :ptype epoch: int
        :return: nothing
        :rtype: None
        """
        self._conn.execute(
            f"INSERT INTO {SCHEMA}.{VERSIONED} SELECT ?, ?, 'c' || i::text, 1, ?, false FROM range(?) AS r(i)",
            [state, run, epoch, count],
        )
        self.touch(f"state:{state}", epoch)

    def touch(self, scope: str, epoch: int) -> None:
        """a scope's ``last_epoch`` raised to ``epoch``, as a stamped write's commit raises it.

        :param scope: the scope
        :ptype scope: str
        :param epoch: the stamp
        :ptype epoch: int
        :return: nothing
        :rtype: None
        """
        held = self._conn.execute(f"SELECT last_epoch FROM {SCHEMA}.{EPOCHS} WHERE scope = ?", [scope]).fetchall()
        if not held:
            self._conn.execute(f"INSERT INTO {SCHEMA}.{EPOCHS} VALUES (?, ?)", [scope, epoch])
        elif held[0][0] < epoch:
            self._conn.execute(f"UPDATE {SCHEMA}.{EPOCHS} SET last_epoch = ? WHERE scope = ?", [epoch, scope])

    def stamped(self, state: str, run: str, bucket: int, epoch: int, *, gone: bool = False) -> None:
        """a row of the stamped table, in ``bucket``.

        :param state: its state
        :ptype state: str
        :param run: its run
        :ptype run: str
        :param bucket: its restore bucket
        :ptype bucket: int
        :param epoch: its stamp
        :ptype epoch: int
        :param gone: whether it is marked gone
        :ptype gone: bool
        :return: nothing
        :rtype: None
        """
        self._conn.execute(
            f"INSERT INTO {SCHEMA}.{STAMPED} VALUES (?, ?, 1, ?, ?, ?)", [state, run, epoch, bucket, gone]
        )

    def run_marker(self, state: str, run: str, epoch: int) -> None:
        """a run of the versioned table deleted at ``epoch``: its marker, its scope touched.

        :param state: the run's state
        :ptype state: str
        :param run: the run
        :ptype run: str
        :param epoch: the delete's stamp
        :ptype epoch: int
        :return: nothing
        :rtype: None
        """
        self._conn.execute(
            f"INSERT INTO {SCHEMA}.{RUN_MARKERS_TABLE} VALUES (?, ?, ?, ?)",
            [VERSIONED, state, f"{state}\x1f{run}", epoch],
        )
        self.touch(f"state:{state}", epoch)

    def data_reads(self, table: str) -> list[tuple[str, tuple[Any, ...]]]:
        """the statements sent that read ``table``.

        :param table: the table
        :ptype table: str
        :return: them
        :rtype: list[tuple[str, tuple[Any, ...]]]
        """
        return [(query, args) for query, args in self.sent if f'"{table}"' in query]

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        """the statement's rows: the catalog's answered from the declarations, every other one by DuckDB.

        :param query: the statement
        :ptype query: str
        :param args: its parameters
        :ptype args: Any
        :return: the rows
        :rtype: list[Any]
        """
        self.sent.append((query, args))
        if "pg_description" in query:
            return [
                {"table_schema": SCHEMA, "table_name": name, "declared": declaration(schema), "run_markers": True}
                for name, schema in sorted(self.declared.items())
            ]
        if "pg_attribute" in query:
            named = str(args[0]).split(".")[-1].strip('"')
            schema = self.declared[named]
            return [
                {"column_name": column, "column_type": "text"} for column in args[1] if column in schema.primary_key
            ]
        if query.startswith("SELECT version()"):
            return [{"v": "PostgreSQL 16"}]
        return await self._db.fetch(query, *args)

    async def execute(self, query: str, *args: Any) -> str:
        """refused: a reader of an epoch range writes nothing.

        :param query: the statement
        :ptype query: str
        :param args: its parameters
        :ptype args: Any
        :return: never
        :rtype: str
        :raises AssertionError: always
        """
        raise AssertionError(f"an epoch-range reader writes nothing to L3: {query}")
