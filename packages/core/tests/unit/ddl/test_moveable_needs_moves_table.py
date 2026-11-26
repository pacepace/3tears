"""A stamped table that declares a moveable column is refused at start when its schema holds no moves table.

Every write that moves a row between tenants (``TableSchema(moveable=...)``) appends the move to
``threetears_tenant_moves`` in the table's own schema, in the same statement (``threetears.core.ddl.tenant_moves``).
Without that table each such write fails at write time with "relation does not exist". A runtime requirement fails
at start instead: the refusal runs at the end of every migration run and at the keeper's install, one catalog read,
and its error names the table, the schema and the call that creates the moves table.

The fake catalog below answers the refusal's statement from its own state: each index's comment, matched by the
statement's own ``LIKE`` patterns, and which schemas hold the moves table. The comments are the real ones a
``TableSchema``'s DDL writes, so a declaration that stops naming its moveable columns flips these tests. The statement
itself runs on a real database in ``tests/integration/test_tenant_moves_live.py``.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from threetears.core.collections.schema_backed import STRING_TYPE, Column, TableSchema
from threetears.core.data.migrations import MigrationRunner, MigrationScope, PackageMigrations
from threetears.core.ddl.epoch_catalog import (
    MovesTableMissingError,
    moveable_without_moves,
    refuse_moveable_without_moves,
)
from threetears.core.ddl.row_epoch import ROW_STAMPED
from threetears.core.ddl.tenant_moves import TENANT_MOVES_TABLE
from threetears.core.eviction import Declared, Policy
from threetears.core.tenancy import tenant_column

from ..data.migrations.fake_store import FakeDataStore

_EVICTABLE = Declared(Policy.EVICTABLE, "a test's stamped table")

#: a stamped table a write may move between tenants
_MOVEABLE = TableSchema(
    name="mv_widgets",
    primary_key="id",
    columns=[Column("id", STRING_TYPE), Column("customer_id", STRING_TYPE)],
    tenancy=tenant_column("customer_id"),
    row_epoch=ROW_STAMPED,
    moveable=("customer_id",),
    l1_eviction=_EVICTABLE,
)

#: a stamped table no write moves
_FIXED = TableSchema(
    name="mv_fixed",
    primary_key="id",
    columns=[Column("id", STRING_TYPE), Column("customer_id", STRING_TYPE)],
    tenancy=tenant_column("customer_id"),
    row_epoch=ROW_STAMPED,
    l1_eviction=_EVICTABLE,
)

_COMMENT = re.compile(r"^COMMENT ON INDEX \S+ IS '(.*)'$", re.DOTALL)


def _declaration(schema: TableSchema) -> str:
    """the restore declaration the table's own DDL writes as its index comment."""
    found = [m.group(1) for each in schema.create_statements(yugabyte=False) if (m := _COMMENT.match(str(each)))]
    assert len(found) == 1, f"{schema.name}'s DDL comments its restore index once"
    return found[0].replace("''", "'")


def _like(value: str, pattern: str) -> bool:
    """SQL ``LIKE``: ``%`` any run, ``_`` one character, everything else itself."""
    spelled = "".join(".*" if c == "%" else "." if c == "_" else re.escape(c) for c in pattern)
    return re.fullmatch(spelled, value, re.DOTALL) is not None


class _Catalog:
    """the catalog: each table's index comment, and the schemas holding the moves table. Answers the refusal's
    statement by its own patterns and parameters."""

    def __init__(self, tables: tuple[tuple[str, TableSchema], ...], holding: frozenset[str]) -> None:
        self.comments = tuple((schema, table.name, _declaration(table)) for schema, table in tables)
        self.holding = holding
        self.asked: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.asked.append((query, args))
        declared, stamped, moveable, moves_table, only = args
        assert moves_table == TENANT_MOVES_TABLE
        return [
            {"table_schema": schema, "table_name": table, "declaration": comment}
            for schema, table, comment in sorted(self.comments)
            if all(_like(comment, pattern) for pattern in (declared, stamped, moveable))
            and (only is None or schema == only)
            and schema not in self.holding
        ]


async def test_a_moveable_table_without_the_moves_table_is_refused_naming_the_table_schema_and_fix() -> None:
    """Flips if a moveable stamped table whose schema lacks the moves table passes the start check, or if the error
    does not name the table, the schema's missing moves table and the call that creates it."""
    catalog = _Catalog((("agent_reports", _MOVEABLE),), holding=frozenset())
    with pytest.raises(MovesTableMissingError) as refused:
        await refuse_moveable_without_moves(catalog)
    assert refused.value.found == ("agent_reports.mv_widgets (moveable: customer_id)",)
    assert refused.value.schemas == ("agent_reports",)
    text = str(refused.value)
    assert f"agent_reports.{TENANT_MOVES_TABLE}" in text
    assert "TENANT_MOVES_SCHEMA.create_statements(yugabyte=" in text
    assert "How to fix" in text
    assert len(catalog.asked) == 1, "one catalog read"


async def test_a_moveable_table_beside_the_moves_table_passes_and_a_fixed_one_needs_none() -> None:
    """Flips if the check refuses a moveable table whose schema holds the moves table, or a stamped table that
    declares no moveable column."""
    catalog = _Catalog(
        (("agent_reports", _MOVEABLE), ("agent_other", _FIXED)),
        holding=frozenset({"agent_reports"}),
    )
    await refuse_moveable_without_moves(catalog)
    assert await moveable_without_moves(catalog) == ()


async def test_the_check_judges_one_schema_when_given_one() -> None:
    """Flips if a schema-scoped check refuses another schema's table."""
    catalog = _Catalog((("agent_other", _MOVEABLE),), holding=frozenset())
    await refuse_moveable_without_moves(catalog, "agent_reports")
    assert await moveable_without_moves(catalog, "agent_other") == (("agent_other", "mv_widgets", ("customer_id",)),)


class _Store(FakeDataStore):
    """a run's store over schema ``rpt``, its catalog answering the refusal's statement."""

    def __init__(self, catalog: _Catalog) -> None:
        super().__init__(schema="rpt")
        self.catalog = catalog

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        if "pg_description" in sql and TENANT_MOVES_TABLE in params:
            return await self.catalog.fetch(sql, *params)
        return await super().query(sql, *params)


async def test_the_migration_run_refuses_a_moveable_table_without_the_moves_table() -> None:
    """Flips if a migration run ends while its schema's moveable stamped table has no moves table beside it."""

    def runner() -> MigrationRunner:
        made = MigrationRunner()
        package = PackageMigrations(name="rows", scope=MigrationScope.PLATFORM)

        async def nothing(store: Any) -> None:
            await store.execute("SELECT 1")

        package.version(1)(nothing)
        made.register(package)
        return made

    with pytest.raises(MovesTableMissingError, match=r"rpt\.mv_widgets"):
        await runner().apply_for_platform_schema(_Store(_Catalog((("rpt", _MOVEABLE),), holding=frozenset())))
    assert (
        await runner().apply_for_platform_schema(_Store(_Catalog((("rpt", _MOVEABLE),), holding=frozenset({"rpt"}))))
        == 1
    )
