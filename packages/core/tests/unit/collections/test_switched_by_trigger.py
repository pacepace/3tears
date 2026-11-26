"""a table switched on by its change-note trigger alone: no class here declares it, and its records go under its schema.

A tool pod's own tables are declared by the pod's classes, which the hub never imports. The hub's broker and a lost
write's recovery read the switch off the database instead (the change-note row trigger, installed under the DDL lock;
the write-mark trigger until 0.95), and key such a table's records by the schema it lives in, where its pods read them.
What these hold:

- the catalogue read names the trigger, the schema and the tables, and answers the tables that carry it;
- ``advance_for_write`` given ``TENANT_BY_SCHEMA`` puts an undeclared table's records, rowless ones too, under the
  write's schema; without it they go on ``_all`` as before;
- a recovery's rowless record of an undeclared table goes under every tenant it may be read under: each pod schema
  holding it and ``_all`` (one in the platform schema, ``_all`` alone), asking no registry, so nothing is reported
  undeclared; a declared table keeps its declared tenancy.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pytest
from prometheus_client import REGISTRY

from threetears.core.collections import load_declared_collections
from threetears.core.collections.epoch_change import ChangedRow
from threetears.core.collections.generation import advance_for_write
from threetears.core.collections.lost_writes import announce_window, tables_switched_by_trigger
from threetears.core.ddl.change_notes import NOTE_TRIGGER
from threetears.core.ddl.row_epoch import EPOCH_COLUMN
from threetears.core.tenancy import TENANT_BY_SCHEMA, schema_tenant

_TABLE = "switched_by_trigger_only_rows"


class _Catalogue:
    """the catalogue as the lookup reads it: which tables of which schema carry the trigger."""

    def __init__(self, triggered: set[tuple[str, str]], epoch: set[tuple[str, str]] | None = None) -> None:
        self.triggered = triggered
        self.epoch = epoch or set()
        self.asked: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.asked.append((query, args))
        if "current_ring" in query:
            # no change-note ring here
            return []
        if "pg_attribute" in query:
            # the tables carrying the epoch column (threetears.core.ddl.epoch_catalog)
            schema, names, column = args
            assert column == EPOCH_COLUMN
            return [{"table_name": table} for s, table in sorted(self.epoch) if s == schema and table in names]
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


async def test_a_table_carrying_an_epoch_is_switched_on_without_any_trigger() -> None:
    """Flips if "switched on" still means "carries the change-note trigger" alone: a versioned table carries no
    per-row trigger (append-only L3 chunk 7), so the broker would advance nothing for it and its copies would never
    hear a write."""
    catalogue = _Catalogue({("agent_reports", _TABLE)}, epoch={("agent_reports", "versioned_rows")})
    switched = await tables_switched_by_trigger(catalogue, "agent_reports", [_TABLE, "versioned_rows", "plain_rows"])
    assert switched == frozenset({_TABLE, "versioned_rows"})


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


class _WholeCatalogue(_Catalogue):
    """the catalogue as a recovery reads it whole: every table of every schema carrying the trigger."""

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        answer: list[dict[str, Any]]
        if args == (NOTE_TRIGGER,):
            self.asked.append((query, args))
            answer = [{"table_schema": s, "table_name": table} for s, table in sorted(self.triggered)]
        else:
            answer = await super().fetch(query, *args)
        return answer


async def _announced(triggered: set[tuple[str, str]], table: str = _TABLE) -> list[str | None]:
    """a lost write's window of ``table`` announced with no change-note ring, the switched tables read off the
    catalogue as a recovery reads them: a rowless record a tenant. The startup loader has run, as in every hub, here on
    a thread: its first run imports every declared module, seconds that on the loop held it past the loop-stall guard's
    limit (threetears.core.testing.loop_stall)."""
    await asyncio.to_thread(load_declared_collections)
    source = _Recorder()
    now = datetime.now(UTC)
    await announce_window(
        _WholeCatalogue(triggered),
        generation_source=source,
        notes_schema="platform",
        since=now,
        until=now,
        why="recovery",
    )
    return [tenant for name, *_, tenant in source.records if name == table]


def _undeclared_count(table: str) -> float:
    return sum(
        REGISTRY.get_sample_value("threetears_undeclared_table_total", {"table": table, "lookup": lookup}) or 0.0
        for lookup in ("tenancy", "write_generation", "l2_first")
    )


async def test_a_recovery_keys_an_undeclared_pod_table_by_every_tenant_and_reports_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Flips if recovering a table known only from its trigger (a tool pod's own, undeclared on purpose) asks a
    registry -- an undeclared ERROR and ``threetears_undeclared_table_total`` in every hub, and a refusal next
    release -- or invalidates fewer than every tenant it may be read under."""
    table = "switched_by_trigger_only_recovered_rows"
    with caplog.at_level(logging.ERROR, logger="threetears.core.collections.undeclared_tables"):
        tenants = await _announced({("agent_a", table), ("agent_b", table)}, table)
    assert tenants == [schema_tenant("agent_a"), schema_tenant("agent_b"), None]
    assert [r for r in caplog.records if table in r.getMessage()] == []
    assert _undeclared_count(table) == 0.0


async def test_a_recovery_keeps_a_declared_tables_declared_tenancy() -> None:
    """Flips if a declared table (``coordination_counters``, by schema) is recovered under more or fewer tenants
    than its declaration names."""
    tenants = await _announced({("agent_a", "coordination_counters")}, "coordination_counters")
    assert tenants == [schema_tenant("agent_a")]


async def test_a_recovery_keys_an_undeclared_pod_table_by_each_schema_holding_it() -> None:
    assert await _announced({("agent_a", _TABLE), ("agent_b", _TABLE)}) == [
        schema_tenant("agent_a"),
        schema_tenant("agent_b"),
        None,
    ]


async def test_a_recovery_keeps_an_undeclared_platform_table_on_all() -> None:
    assert await _announced({("platform", _TABLE)}) == [None]
