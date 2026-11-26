"""A versioned (page-written) table carries no per-row trigger, and one found on it is refused.

Append-only L3 chunk 7 (``design-append-only-l3.md`` decision 4). A versioned table is its own change record: a restore
reads it back through its ``(scope, epoch)`` index, so it is given no change note and no trigger; a per-row trigger on
it switches off YugabyteDB's write batching (1.58 ms a row against 0.43 without, 2026-11-25). These tests hold:

- its DDL creates no trigger, and no migration run gives one (``test_keeper_marks_ddl``: the ring's step is gone);
- a per-row trigger found on one is refused at the end of every migration run, with a how-to-fix error;
- "switched on" counts it by the epoch it carries (``test_switched_by_trigger``).

No live check yet: the catalog statements (pg_trigger.tgtype, the index comment) are unproven on YugabyteDB.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.core.collections.generation import WRITE_GENERATION
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
)
from threetears.core.data.migrations import MigrationRunner, MigrationScope, PackageMigrations
from threetears.core.ddl.epoch_catalog import refuse_row_triggers
from threetears.core.ddl.row_epoch import row_versions
from threetears.core.entities.base import BaseEntity
from threetears.core.eviction import Declared, Policy
from threetears.core.tenancy import TENANT_BY_SCHEMA

from ..data.migrations.fake_store import FakeDataStore

_TABLE = "versioned_trigger_results"
_SCOPE = "join_state_code"

#: a versioned table a consumer declares: its key, its scope, the scope epochs table naming its states
_RESULTS = TableSchema(
    name=_TABLE,
    primary_key=(_SCOPE, "candidate"),
    columns=[Column(_SCOPE, STRING_TYPE), Column("candidate", STRING_TYPE), Column("votes", BIGINT_TYPE)],
    tenancy=TENANT_BY_SCHEMA,
    row_epoch=row_versions(_SCOPE, epochs="versioned_trigger_epochs", scope_prefix="state:"),
    l1_eviction=Declared(Policy.EVICTABLE, "a test's versioned table"),
)


class _Row(BaseEntity):
    primary_key_field = "candidate"


class _Results(collection_for_schema(_RESULTS, entity_class=_Row)):  # type: ignore[misc]
    """the versioned table, switched on: its writes advance a generation."""

    write_generation = WRITE_GENERATION


def test_a_versioned_table_s_ddl_creates_no_trigger() -> None:
    """Flips if a versioned table's own DDL creates a trigger of any kind."""
    statements = [str(each) for each in _RESULTS.create_statements(yugabyte=False)]
    assert statements, "the table's DDL"
    assert not [each for each in statements if "CREATE TRIGGER" in each.upper()]


def test_a_versioned_table_is_switched_on_by_its_class() -> None:
    """Flips if the versioned table's class stops switching it on: the refusals below judge a switched-on table."""
    assert _Results.write_generation is WRITE_GENERATION


class _Catalog(FakeDataStore):
    """a run's store over a schema holding one versioned table, and the triggers given."""

    def __init__(self, *, triggers: tuple[tuple[str, str, str], ...] = ()) -> None:
        super().__init__(schema="rpt")
        self.row_triggers = triggers

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        if "pg_description" in sql and "tgtype" in sql:
            # the per-row triggers on versioned tables (epoch_catalog._ROW_TRIGGERS_SQL)
            self.queried.append((sql, params))
            return [{"table_schema": s, "table_name": t, "trigger_name": g} for s, t, g in self.row_triggers]
        if "information_schema.tables" in sql:
            self.queried.append((sql, params))
            return [{"table_name": name} for name in params[0]]
        if "current_schema() AS schema" in sql and "schema_name" not in sql:
            self.queried.append((sql, params))
            return [{"schema": "rpt"}]
        return await super().query(sql, *params)


async def test_a_per_row_trigger_on_a_versioned_table_is_refused_by_the_migration_run() -> None:
    """Flips if a migration run ends while a versioned table carries a per-row trigger: a migration, a hand-written
    statement or an older replica would have brought the slow path back silently."""
    runner = MigrationRunner()
    package = PackageMigrations(name="rows", scope=MigrationScope.PLATFORM)

    async def nothing(store: Any) -> None:
        await store.execute("SELECT 1")

    package.version(1)(nothing)
    runner.register(package)
    store = _Catalog(triggers=(("rpt", _TABLE, "results_audit"),))
    with pytest.raises(RuntimeError, match="PER-ROW TRIGGER") as refused:
        await runner.apply_for_platform_schema(store)
    assert f"  - rpt.{_TABLE}: results_audit\n" in str(refused.value)
    assert "How to fix" in str(refused.value)
    assert "DROP TRIGGER" in str(refused.value)


async def test_a_run_over_versioned_tables_with_no_row_trigger_passes() -> None:
    """Flips if the refusal fires with no per-row trigger to refuse."""
    runner = MigrationRunner()
    package = PackageMigrations(name="rows", scope=MigrationScope.PLATFORM)

    async def nothing(store: Any) -> None:
        await store.execute("SELECT 1")

    package.version(1)(nothing)
    runner.register(package)
    assert await runner.apply_for_platform_schema(_Catalog()) == 1


class _Pool:
    """a pool answering the per-row trigger read with what it is given."""

    def __init__(self, found: list[dict[str, str]]) -> None:
        self.found = found
        self.asked: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, str]]:
        self.asked.append((query, args))
        return self.found


async def test_the_keeper_s_refusal_judges_every_schema_and_names_each_trigger() -> None:
    """Flips if the start's refusal judges one schema only, or reads stamped tables' declarations as versioned."""
    pool = _Pool([{"table_schema": "agent_x", "table_name": "rows", "trigger_name": "threetears_change_note"}])
    with pytest.raises(RuntimeError, match="PER-ROW TRIGGER") as refused:
        await refuse_row_triggers(pool)
    assert "  - agent_x.rows: threetears_change_note\n" in str(refused.value)
    ((query, args),) = pool.asked
    declared, stamped, schema = args
    assert declared == '{"threetears_restore":%' and stamped == '%"kind":"stamped"%' and schema is None
    assert "(g.tgtype::int & 1) = 1" in query and "NOT g.tgisinternal" in query
