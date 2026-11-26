"""a table switched on by the epoch it carries alone: no class here declares it, and its records go under its schema.

A tool pod's own tables are declared by the pod's classes, which the hub never imports. The hub's broker and a lost
write's recovery read the switch off the database instead (the epoch the table carries, ``epoch BIGINT NOT NULL``;
the change-note row trigger until the ring retired, append-only L3 chunk 7, the write-mark trigger before it), and
key such a table's records by the schema it lives in, where its pods read them. What these hold:

- the catalogue read answers the tables of the schema that carry an epoch, and nothing for a trigger alone;
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
from typing import Any

import pytest
from prometheus_client import REGISTRY

from threetears.core.collections import load_declared_collections
from threetears.core.collections.epoch_change import ChangedRow
from threetears.core.collections.generation import advance_for_write
from threetears.core.collections.lost_writes import tables_switched_by_epoch
from threetears.core.collections.recovery_tenants import recovery_tenants
from threetears.core.ddl.row_epoch import EPOCH_COLUMN
from threetears.core.tenancy import TENANCY_UNDECLARED, TENANT_BY_SCHEMA, schema_tenant

_TABLE = "switched_by_trigger_only_rows"


class _Catalogue:
    """the catalogue as the lookup reads it: which tables of which schema carry the trigger."""

    def __init__(self, triggered: set[tuple[str, str]], epoch: set[tuple[str, str]] | None = None) -> None:
        self.triggered = triggered
        self.epoch = epoch or set()
        self.asked: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.asked.append((query, args))
        if "pg_attribute" in query:
            # the tables carrying the epoch column (threetears.core.ddl.epoch_catalog)
            schema, names, column = args
            assert column == EPOCH_COLUMN
            return [{"table_name": table} for s, table in sorted(self.epoch) if s == schema and table in names]
        raise AssertionError(f"only the epoch column is read: {query}")

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


async def test_a_table_is_switched_on_by_its_epoch_alone_never_by_a_trigger() -> None:
    """Flips if "switched on" still means "carries the change-note trigger": the ring is retired (append-only L3
    chunk 7), so a table with a trigger and no epoch is no longer switched on, and one carrying an epoch is."""
    catalogue = _Catalogue({("agent_reports", _TABLE)}, epoch={("agent_reports", "versioned_rows")})
    switched = await tables_switched_by_epoch(catalogue, "agent_reports", [_TABLE, "versioned_rows", "plain_rows"])
    assert switched == frozenset({"versioned_rows"})
    assert ["pg_attribute" in query for query, _ in catalogue.asked] == [True], "one catalogue read, of the epoch"


async def test_no_tables_reads_nothing() -> None:
    catalogue = _Catalogue(set())
    assert await tables_switched_by_epoch(catalogue, "agent_reports", []) == frozenset()
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


async def _announced(triggered: set[tuple[str, str]], table: str = _TABLE) -> list[str | None]:
    """the tenants a recovery's rowless record of ``table`` goes under (a keys-unknown window), the schemas holding it
    as the catalogue's restore declarations name them, none of which names a tenancy. The startup loader has run, as in
    every hub, here on a thread: its first run imports every declared module, seconds that on the loop held it past
    the loop-stall guard's limit (threetears.core.testing.loop_stall)."""
    await asyncio.to_thread(load_declared_collections)
    schemas = [schema for schema, name in sorted(triggered) if name == table]
    return list(recovery_tenants(table, TENANCY_UNDECLARED, schemas, platform_schema="platform", schema=None))


def _undeclared_count(table: str) -> float:
    return sum(
        REGISTRY.get_sample_value("threetears_undeclared_table_total", {"table": table, "lookup": lookup}) or 0.0
        for lookup in ("tenancy", "write_generation", "l2_first")
    )


async def test_a_recovery_keys_an_undeclared_pod_table_by_every_tenant_and_reports_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Flips if recovering a table known only from the catalogue (a tool pod's own, undeclared on purpose) asks a
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
