"""a table switched on by its write-mark trigger alone: no class here declares it, and its records go under its schema.

A tool pod's own tables are declared by the pod's classes, which the hub never imports. The hub's broker and its
write-mark reconciler read the switch off the database instead (the trigger, installed under the DDL lock), and
key such a table's records by the schema it lives in, where its pods read them. What these hold:

- the catalogue read names the trigger, the schema and the tables, and answers the tables that carry it;
- ``advance_for_write`` given ``TENANT_BY_SCHEMA`` puts an undeclared table's records, rowless ones too, under the
  write's schema; without it they go on ``_all`` as before;
- the reconciler's rowless record of an undeclared table goes under each pod schema holding it, and one in the
  platform schema stays on ``_all``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from threetears.core.collections.epoch_change import ChangedRow
from threetears.core.collections.generation import advance_for_write
from threetears.core.collections.write_marks import WRITE_MARK_TRIGGER, WriteMarkReconciler, tables_switched_by_trigger
from threetears.core.tenancy import TENANT_BY_SCHEMA, schema_tenant

_TABLE = "switched_by_trigger_only_rows"


class _Catalogue:
    """the catalogue as the lookup reads it: which tables of which schema carry the trigger."""

    def __init__(self, triggered: set[tuple[str, str]]) -> None:
        self.triggered = triggered
        self.asked: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.asked.append((query, args))
        if "information_schema.tables" in query:
            return [{"table_schema": schema} for schema, table in sorted(self.triggered) if table == args[0]]
        trigger, schema, names = args
        assert trigger == WRITE_MARK_TRIGGER
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
    assert args == (WRITE_MARK_TRIGGER, "agent_reports", ["other_rows", "plain_rows", _TABLE])


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


class _Queue(_Catalogue):
    """the marks queue beside the catalogue: one mark of the table, and no change-note ring."""

    def __init__(self, triggered: set[tuple[str, str]]) -> None:
        super().__init__(triggered)
        self.marks = {1: _TABLE}

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        if "information_schema.tables" in query:
            return await super().fetch(query, *args)
        if "current_ring" in query:
            return []
        if "count(*)" in query:
            return [{"n": len(self.marks), "age": 1.0}]
        return [
            {"id": key, "table_name": table, "date_created": datetime.now(UTC)} for key, table in self.marks.items()
        ][: int(args[0])]

    async def execute(self, query: str, *args: Any) -> str:
        for key in args[0]:
            self.marks.pop(key, None)
        return "DELETE"


async def _reconciled(triggered: set[tuple[str, str]]) -> list[str | None]:
    source = _Recorder()
    reconciler = WriteMarkReconciler(
        _Queue(triggered),  # type: ignore[arg-type]
        generation_source=source,
        marks_schema="platform",
        tables=lambda: (),
    )
    await reconciler.run_pass()
    return [tenant for *_, tenant in source.records]


async def test_the_reconciler_keys_an_undeclared_pod_table_by_each_schema_holding_it() -> None:
    assert await _reconciled({("agent_a", _TABLE), ("agent_b", _TABLE)}) == [
        schema_tenant("agent_a"),
        schema_tenant("agent_b"),
    ]


async def test_the_reconciler_keeps_an_undeclared_platform_table_on_all() -> None:
    assert await _reconciled({("platform", _TABLE)}) == [None]
