"""Integration: a state's newest-version reads of a race-keyed versioned table go by the state's index, never every
state's versions (``VersionedTable.fixed``).

Reports' live tables are keyed by race (and geography), their scope ``join_state_code`` in their rows alone, with an
index leading with the state. Before the scope pin was judged inside the newest relation's ``DISTINCT ON``, a state's
count (``counts_by(by={state})``) chose every key's newest version of the whole table and filtered by state after, and a
state's geographies (reports' ``held_in``: ``read_l3_rows`` by the state and a geography set) bound only the geographies
inside: on both engines a scan of the whole table. The unit tests (``tests/unit/test_keyed_newest_scope_inside.py``, on
DuckDB) prove the rows; these prove the plans, at a size where a planner would rather scan if it could.

Against the session ``db_container``: Postgres in the ``integration`` suite, YugabyteDB in ``integration_yugabyte``.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

import asyncpg
import pytest

from threetears.core.collections.complete_copy import read_l3_rows
from threetears.core.collections.run_markers import RUN_MARKERS_SCHEMA
from threetears.core.collections.schema_backed import INT_TYPE, STRING_TYPE, Column, IndexDef, TableSchema
from threetears.core.ddl.row_epoch import row_versions
from threetears.core.keyed import counts_by
from threetears.core.testing.l1_tables import TEST_L1_TABLE
from threetears.core.testing.plans import on_yugabyte

pytestmark = pytest.mark.integration

_log = logging.getLogger(__name__)

_NAME = "sp_live"

#: the index a state's reads go by, as reports declares one for its live tables
_STATE_INDEX = "sp_live_by_state"

_LIVE = TableSchema(
    name=_NAME,
    primary_key=("race_id", "geo_id"),
    columns=[
        Column("race_id", STRING_TYPE),
        Column("geo_id", STRING_TYPE),
        Column("join_state_code", STRING_TYPE),
        Column("votes", INT_TYPE, nullable=True),
    ],
    indexes=(IndexDef(name=_STATE_INDEX, columns=("join_state_code", "race_id", "geo_id")),),
    row_epoch=row_versions("join_state_code", scope_prefix="state:"),
    l1_eviction=TEST_L1_TABLE,
)

#: states, races a state, geographies a race, versions a key: 200,000 rows, 4,000 a state (2,000 keys)
_STATES = 50
_RACES = 200
_GEOS = 10
_VERSIONS = 2
#: an epoch's wall part far from 0; versions are stamped from it up
_BASE = 1_000_000 << 15


class _Recorder:
    """the connection the reads run on, keeping each statement and its parameters."""

    def __init__(self, conn: asyncpg.Connection, yugabyte: bool) -> None:
        self._conn = conn
        self._yugabyte = yugabyte
        self.sent: list[tuple[str, tuple[Any, ...]]] = []

    async def answers_from_yugabyte(self) -> bool:
        return self._yugabyte

    async def fetch(self, query: str, /, *params: Any) -> list[dict[str, Any]]:
        self.sent.append((query, params))
        return [dict(row) for row in await self._conn.fetch(query, *params)]

    async def fetchrow(self, query: str, /, *params: Any) -> dict[str, Any] | None:
        self.sent.append((query, params))
        row = await self._conn.fetchrow(query, *params)
        return None if row is None else dict(row)


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    schema = f"scopepin_{uuid.uuid4().hex[:8]}"
    held = await asyncpg.connect(db_container, command_timeout=600)
    try:
        yugabyte = await on_yugabyte(held)
        await held.execute(f"CREATE SCHEMA {schema}")
        await held.execute(f"SET search_path = {schema}")
        for declared in (_LIVE, RUN_MARKERS_SCHEMA):
            for statement in declared.create_statements(yugabyte=yugabyte):
                await held.execute(statement)
        # each race in one state (a key's versions all hold its state), each geography of it a key
        for version in range(_VERSIONS):
            await held.execute(
                f"INSERT INTO {_NAME} (race_id, geo_id, join_state_code, votes, epoch, gone) "
                "SELECT 'r' || (g / $1), 'g' || (g % $1), 's' || ((g / $1) % $2), g, $3::bigint, false "
                "FROM generate_series(0, $4 - 1) g",
                _GEOS,
                _STATES,
                _BASE + version,
                _STATES * _RACES * _GEOS,
            )
        await held.execute(f"ANALYZE {_NAME}")
        yield held
    finally:
        await held.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await held.close()


def _nodes(plan: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    yield plan
    for child in plan.get("Plans", ()):
        yield from _nodes(child)


async def _plan(conn: asyncpg.Connection, sql: str, params: tuple[Any, ...]) -> Mapping[str, Any]:
    explained = await conn.fetchval(f"EXPLAIN (FORMAT JSON) {sql}", *params)
    plan: Mapping[str, Any] = json.loads(explained)[0]["Plan"]
    return plan


def _shown(plan: Mapping[str, Any], depth: int = 0) -> str:
    """the plan as indented lines: each node's type, relation, index and condition."""
    parts = [plan.get("Node Type", "?")]
    for field in ("Relation Name", "Index Name", "Index Cond", "Recheck Cond", "Filter"):
        if field in plan:
            parts.append(f"{field}={plan[field]}")
    line = "  " * depth + " ".join(str(part) for part in parts)
    return "\n".join([line, *(_shown(child, depth + 1) for child in plan.get("Plans", ()))])


async def _assert_by_state(conn: asyncpg.Connection, what: str, sql: str, params: tuple[Any, ...]) -> None:
    """the statement reads the table by its state alone: no sequential scan of it, and every read of it bounded by an
    index condition on the state (the state index, or the (scope, epoch) index, both leading with it; a bitmap heap
    scan by its recheck of the bitmap index scan under it)."""
    plan = await _plan(conn, sql, params)
    shown = _shown(plan)
    _log.warning("%s on %s:\n%s", what, "YugabyteDB" if await on_yugabyte(conn) else "Postgres", shown)
    reads = [node for node in _nodes(plan) if node.get("Relation Name") == _NAME]
    assert reads, f"{what} reads no {_NAME}:\n{shown}"
    for node in reads:
        assert "Seq Scan" not in str(node["Node Type"]), f"{what} scans {_NAME} whole:\n{shown}"
        # an index scan's own condition, or a bitmap heap scan's recheck of the bitmap index scan under it
        bound = f"{node.get('Index Cond', '')} {node.get('Recheck Cond', '')}"
        assert "join_state_code" in bound, f"{what} reads {_NAME} past its state:\n{shown}"


async def test_a_state_s_count_reads_by_the_state(conn: asyncpg.Connection) -> None:
    """Flips if a state's count (counts_by, by the state) chooses every key's newest version across the table before
    its state is judged: a whole scan, not the state's index."""
    recorder = _Recorder(conn, await on_yugabyte(conn))
    counted = await counts_by(recorder, _NAME, column="geo_id", by={"join_state_code": "s7"})
    assert counted == {f"g{geo}": _RACES for geo in range(_GEOS)}
    # the count and its count of NULLs, each its own statement
    sent = [entry for entry in recorder.sent if _NAME in entry[0]]
    assert len(sent) == 2
    for sql, params in sent:
        await _assert_by_state(conn, "a state's count", sql, params)


async def test_a_state_s_geographies_read_by_the_state(conn: asyncpg.Connection) -> None:
    """Flips if a state's geographies (reports' held_in: read_l3_rows by the state and a geography set) bind only the
    geographies inside the newest relation: no index leads with a geography, so a whole scan."""
    recorder = _Recorder(conn, await on_yugabyte(conn))
    rows = await read_l3_rows(
        recorder,
        _NAME,
        ["race_id", "geo_id", "join_state_code", "votes"],
        ["race_id", "geo_id"],
        where={"join_state_code": "s7"},
        where_any={"geo_id": ("text[]", ["g1", "g2"])},
        page_size=500,
    )
    assert len(rows) == 2 * _RACES
    assert {row["join_state_code"] for row in rows} == {"s7"}
    sent = [entry for entry in recorder.sent if _NAME in entry[0]]
    assert sent
    for sql, params in sent:
        await _assert_by_state(conn, "a state's geographies", sql, params)
