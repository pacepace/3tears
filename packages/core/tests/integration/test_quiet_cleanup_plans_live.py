"""Integration: quiet-time cleanup reads and deletes by key on a real database, at size (append-only chunk 6).

What the unit tests (``tests/unit/collections/test_quiet_cleanup.py``, on DuckDB) cannot prove: that on YugabyteDB the
page of a scope's versions is one range of the table's ``(scope, epoch, key...)`` index, that the versions of the keys
it names are read by the primary key, and that the delete reaches each version by its whole stored key, with no
sequential scan of the table anywhere; and what a batch costs, a version removed, at each batch size (logged, for the
batch size cleanup defaults to: ``quiet_cleanup._DEFAULT_BATCH``).

Against the session ``db_container``: Postgres in the ``integration`` suite, YugabyteDB in ``integration_yugabyte``
(or ``THREETEARS_TEST_ENGINE=yugabyte`` with ``THREETEARS_TEST_POSTGRES_URL`` naming a server). The plans are asserted
on both engines.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from typing import Any, cast

import asyncpg
import pytest

from threetears.core.collections.cleanup_progress import CLEANUP_PROGRESS_SCHEMA
from threetears.core.collections.quiet_cleanup import LoadGate, QuietCleanup
from threetears.core.collections.run_markers import RUN_MARKERS_SCHEMA
from threetears.core.collections.schema_backed import INT_TYPE, STRING_TYPE, Column, TableSchema
from threetears.core.collections.scope_epochs import WHOLE, EpochSnapshot, ScopeEpochs
from threetears.core.ddl.row_epoch import row_versions
from threetears.core.testing.l1_tables import TEST_L1_TABLE
from threetears.core.testing.plans import on_yugabyte, plan_uses_index

pytestmark = pytest.mark.integration

_log = logging.getLogger(__name__)

_RESULTS = TableSchema(
    name="qc_results",
    primary_key=("office_key", "candidate"),
    columns=[
        Column("office_key", STRING_TYPE),
        Column("candidate", STRING_TYPE),
        Column("join_state_code", STRING_TYPE),
        Column("votes", INT_TYPE, nullable=True),
    ],
    row_epoch=row_versions("join_state_code"),
    l1_eviction=TEST_L1_TABLE,
)

#: the index cleanup pages a scope by
_INDEX = "qc_results_scope_epoch"
#: states, keys a state, and the versions each key holds: 200,000 rows, so the planner would rather scan if it could
_STATES = 50
_KEYS = 2_000
_VERSIONS = 2
#: an epoch's wall part far from 0; versions are stamped from it up
_BASE = 1_000_000 << 15


class _Epochs:
    """the scope epochs cleanup listens to, nothing loading."""

    def __init__(self, scopes: list[str]) -> None:
        self.snap = EpochSnapshot(version=1, writing=None, epochs={WHOLE: 1, **dict.fromkeys(scopes, 1)}, previous={})

    def on_change(self, listener: Callable[[], None]) -> Callable[[], None]:
        return lambda: None

    async def snapshot(self) -> EpochSnapshot:
        return self.snap


class _Recorder:
    """the connection cleanup runs on, recording each statement and its parameters."""

    def __init__(self, conn: asyncpg.Connection, yugabyte: bool) -> None:
        self._conn = conn
        self._yugabyte = yugabyte
        self.sent: list[tuple[str, tuple[Any, ...]]] = []

    async def answers_from_yugabyte(self) -> bool:
        return self._yugabyte

    async def fetch(self, query: str, /, *params: Any) -> list[dict[str, Any]]:
        self.sent.append((query, params))
        return [dict(row) for row in await self._conn.fetch(query, *params)]

    async def execute(self, query: str, /, *params: Any) -> str:
        self.sent.append((query, params))
        return str(await self._conn.execute(query, *params))


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    schema = f"cleanup_{uuid.uuid4().hex[:8]}"
    held = await asyncpg.connect(db_container, command_timeout=600)
    try:
        yugabyte = await on_yugabyte(held)
        await held.execute(f"CREATE SCHEMA {schema}")
        await held.execute(f"SET search_path = {schema}")
        for declared in (_RESULTS, RUN_MARKERS_SCHEMA, CLEANUP_PROGRESS_SCHEMA):
            for statement in declared.create_statements(yugabyte=yugabyte):
                await held.execute(statement)
        yield held
    finally:
        await held.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await held.close()


async def _seed(conn: asyncpg.Connection) -> None:
    """``_KEYS`` keys in each of ``_STATES`` states, each with ``_VERSIONS`` versions, one write statement a version
    (its rows sharing one epoch, as a page's do)."""
    for version in range(_VERSIONS):
        await conn.execute(
            "INSERT INTO qc_results (office_key, candidate, join_state_code, votes, epoch, gone) "
            "SELECT 'o' || g, 'a', 's' || (g % $1), g, $2::bigint, false FROM generate_series(1, $3) g",
            _STATES,
            _BASE + version,
            _STATES * _KEYS,
        )
    await conn.execute("ANALYZE qc_results")


def _cleanup(recorder: _Recorder, batch: int, floor: int) -> QuietCleanup:
    async def window() -> int | None:
        return floor

    gate = LoadGate(cast(ScopeEpochs, _Epochs([f"s{n}" for n in range(_STATES)])))
    return QuietCleanup(recorder, (_RESULTS,), gate, window, batch=batch, run_markers=None)


def _nodes(plan: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    yield plan
    for child in plan.get("Plans", ()):
        yield from _nodes(child)


async def _plan(conn: asyncpg.Connection, sql: str, params: tuple[Any, ...]) -> Mapping[str, Any]:
    """the statement's plan from estimates (a write is planned, never run here)."""
    explained = await conn.fetchval(f"EXPLAIN (FORMAT JSON) {sql}", *params)
    plan: Mapping[str, Any] = json.loads(explained)[0]["Plan"]
    return plan


async def test_cleanup_pages_a_scope_through_the_scope_epoch_index(conn: asyncpg.Connection) -> None:
    """Flips if a page of a scope's versions, the versions of its keys, or the delete of what they supersede reaches
    the table by anything but its keys: the (scope, epoch, key) index for the page, the primary key for the rest."""
    await _seed(conn)
    recorder = _Recorder(conn, await on_yugabyte(conn))
    swept = await _cleanup(recorder, 500, _BASE + _VERSIONS).clean()
    assert swept.removed == _STATES * _KEYS * (_VERSIONS - 1)
    planned = [(sql, params) for sql, params in recorder.sent if "qc_results" in sql]
    pages = [entry for entry in planned if '"join_state_code" = $1' in entry[0]]
    rest = [entry for entry in planned if entry not in pages]
    assert pages and rest
    for sql, params in pages[:3]:
        plan = await _plan(conn, sql, params)
        assert plan_uses_index(plan, _INDEX), f"the page does not read {_INDEX}: {plan}"
    for sql, params in rest[:4]:
        plan = await _plan(conn, sql, params)
        scans = [node for node in _nodes(plan) if node.get("Relation Name") == "qc_results"]
        assert scans and all(node["Node Type"] != "Seq Scan" for node in scans), f"{sql[:60]} scans: {plan}"


@pytest.mark.parametrize("batch", [100, 500, 2000])
async def test_what_a_cleanup_batch_costs(conn: asyncpg.Connection, batch: int) -> None:
    """Logs ms a version removed at ``batch``, and flips if cleanup removes anything but the superseded versions."""
    await _seed(conn)
    recorder = _Recorder(conn, await on_yugabyte(conn))
    started = time.perf_counter()
    swept = await _cleanup(recorder, batch, _BASE + _VERSIONS).clean()
    elapsed = time.perf_counter() - started
    assert swept.finished
    assert swept.removed == _STATES * _KEYS * (_VERSIONS - 1)
    left = await conn.fetchval("SELECT count(*) FROM qc_results WHERE office_key = ANY($1::text[])", ["o1", "o2"])
    assert left == 2
    statements = len(recorder.sent)
    _log.warning(
        "quiet cleanup batch %d on %s: %d removed in %.2f s, %.3f ms a version, %d statements (%.2f ms each)",
        batch,
        "YugabyteDB" if await on_yugabyte(conn) else "Postgres",
        swept.removed,
        elapsed,
        elapsed * 1000 / max(1, swept.removed),
        statements,
        elapsed * 1000 / max(1, statements),
    )
