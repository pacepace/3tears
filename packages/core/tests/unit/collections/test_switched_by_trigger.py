"""a table switched on by its change-note trigger alone: no class here declares it, and its records go under its schema.

A tool pod's own tables are declared by the pod's classes, which the hub never imports. The hub's broker and a lost
write's recovery read the switch off the database instead (the change-note row trigger, installed under the DDL lock;
the write-mark trigger until 0.95), and key such a table's records by the schema it lives in, where its pods read them.
What these hold:

- the catalogue read names the trigger, the schema and the tables, and answers the tables that carry it;
- ``advance_for_write`` given ``TENANT_BY_SCHEMA`` puts an undeclared table's records, rowless ones too, under the
  write's schema; without it they go on ``_all`` as before;
- a recovery's rowless record of an undeclared table goes under each pod schema holding it, and one in the
  platform schema stays on ``_all``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from threetears.core.collections.epoch_change import ChangedRow
from threetears.core.collections.generation import advance_for_write
from threetears.core.collections.lost_writes import announce_window, tables_switched_by_trigger
from threetears.core.ddl.change_notes import NOTE_TRIGGER
from threetears.core.tenancy import TENANT_BY_SCHEMA, schema_tenant

_TABLE = "switched_by_trigger_only_rows"


class _Catalogue:
    """the catalogue as the lookup reads it: which tables of which schema carry the trigger."""

    def __init__(self, triggered: set[tuple[str, str]]) -> None:
        self.triggered = triggered
        self.asked: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.asked.append((query, args))
        if "current_ring" in query:
            # no change-note ring here
            return []
        trigger, schema, names = args
        assert trigger == NOTE_TRIGGER
        return [{"table_name": table} for s, table in sorted(self.triggered) if s == schema and table in names]

    async def execute(self, query: str, *args: Any) -> str:
        return ""


class _Recorder:
    """a generation source that records each advance's change and its tenant."""

    def __init__(self) -> None:
        self.records: list[tuple[str, Sequence[ChangedRow] | None, bool, str | None]] = []

    async def current(self, table_name: str) -> str:
        return "inc:0"

    async def advance(self, table_name: str) -> str:
        return "inc:1"

    async def advance_with_change(
        self,
        table_name: str,
        rows: Sequence[ChangedRow] | None,
        *,
        keys_unknown: bool = False,
        tenant: str | None = None,
    ) -> str:
        self.records.append((table_name, rows, keys_unknown, tenant))
        return f"inc:{len(self.records)}"


async def test_the_catalogue_read_answers_the_tables_of_the_schema_that_carry_the_trigger() -> None:
    catalogue = _Catalogue({("agent_reports", _TABLE), ("agent_other", "other_rows")})
    switched = await tables_switched_by_trigger(catalogue, "agent_reports", [_TABLE, "other_rows", "plain_rows"])
    assert switched == frozenset({_TABLE})
    (query, args), *_ = catalogue.asked
    assert "pg_trigger" in query
    assert args == (NOTE_TRIGGER, "agent_reports", ["other_rows", "plain_rows", _TABLE])


async def test_no_tables_reads_nothing() -> None:
    catalogue = _Catalogue(set())
    assert await tables_switched_by_trigger(catalogue, "agent_reports", []) == frozenset()
    assert catalogue.asked == []


async def test_an_undeclared_tables_records_go_under_the_writes_schema_when_its_tenancy_is_given() -> None:
    source = _Recorder()
    await advance_for_write(source, _TABLE, [ChangedRow(ids=["1"])], schema="agent_reports", tenancy=TENANT_BY_SCHEMA)
    await advance_for_write(source, _TABLE, None, schema="agent_reports", tenancy=TENANT_BY_SCHEMA)
    await advance_for_write(source, _TABLE, None, schema="agent_reports", tenancy=TENANT_BY_SCHEMA, keys_unknown=True)
    assert [tenant for *_, tenant in source.records] == [schema_tenant("agent_reports")] * 3


async def test_without_its_tenancy_an_undeclared_tables_records_stay_on_all() -> None:
    source = _Recorder()
    await advance_for_write(source, _TABLE, None, schema="agent_reports")
    assert [tenant for *_, tenant in source.records] == [None]


async def _announced(triggered: set[tuple[str, str]]) -> list[str | None]:
    """a lost write's window of the table announced with no change-note ring: a rowless record a tenant."""
    source = _Recorder()
    catalogue = _Catalogue(triggered)
    schemas = tuple(sorted(schema for schema, table in triggered if table == _TABLE))
    now = datetime.now(UTC)
    await announce_window(
        catalogue,
        generation_source=source,
        notes_schema="platform",
        since=now,
        until=now,
        why="recovery",
        tables={_TABLE: schemas},
    )
    return [tenant for *_, tenant in source.records]


async def test_a_recovery_keys_an_undeclared_pod_table_by_each_schema_holding_it() -> None:
    assert await _announced({("agent_a", _TABLE), ("agent_b", _TABLE)}) == [
        schema_tenant("agent_a"),
        schema_tenant("agent_b"),
    ]


async def test_a_recovery_keeps_an_undeclared_platform_table_on_all() -> None:
    assert await _announced({("platform", _TABLE)}) == [None]
