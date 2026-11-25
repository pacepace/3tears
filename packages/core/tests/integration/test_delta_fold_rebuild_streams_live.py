"""Integration: the writer's rebuild of missing fulls from L3 reads several tables and states at once, at real size.

Why: read one table after another, one statement at a time, a cold rebuild of the reports pod's 17 tables (~20.1M
rows, three LSF tables ~86% of them) waited 532 s on the L3 broker at ~37k rows/s before a pod restarted on an empty
volume served (local stack, 2026-11-24). ``DeltaFolds._fresh_fulls`` now reads up to ``RebuildConcurrency.tables``
tables at once and splits a table of more than a page by its states, ``streams`` reads in flight in all.

The table set is that night's shape scaled to a tenth (``THREETEARS_REBUILD_SPEED_SCALE``, default 0.1): three big
tables of 766k, 699k and 257k rows and fourteen of 20k, ~2.0M rows over 50 states. L3 is the rail as a pod meets it
(:class:`_Rail`): each statement runs on its own connection of the database and its rows come back as Arrow parsed off
the interpreter lock (the rail's proxy parses its Arrow reply so), a page sized to the broker's reply cap after the
first (``fetch_arrow_page``, as the hub's ``max_reply_bytes`` announces it). Timed runs belong under measure-lock.
Against the session ``db_container`` (Postgres); set ``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL to run there.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import uuid
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import asyncpg
import pytest

pa = pytest.importorskip("pyarrow")
pa_csv = pytest.importorskip("pyarrow.csv")
pc = pytest.importorskip("pyarrow.compute")

from threetears.core.backends.protocol import L3Page  # noqa: E402
from threetears.core.collections.complete_copy import rail_arrow_types  # noqa: E402
from threetears.core.collections.delta_fold import DeltaFolds, FoldedTable, RebuildConcurrency  # noqa: E402
from threetears.core.collections.epoch_change import EpochChange, EpochWalk, walk_by_count  # noqa: E402
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column, TableSchema  # noqa: E402
from threetears.core.collections.volume_fulls import write_full_part  # noqa: E402
from threetears.core.keyset import keyset_arrow_pages  # noqa: E402
from threetears.core.testing.calibrate import best_seconds, budget_problem  # noqa: E402
from threetears.core.testing.delta_copies import NO_COPIES  # noqa: E402
from threetears.core.testing.l1_tables import TEST_L1_TABLE  # noqa: E402

pytestmark = pytest.mark.integration

_SCALE = float(os.environ.get("THREETEARS_REBUILD_SPEED_SCALE", "0.1"))
#: the 2026-11-24 rebuild's table set: its three LSF tables, then fourteen small ones (~2.85M rows between them)
_TABLES: tuple[tuple[str, int], ...] = (
    ("lsf_county_projections", int(7_660_000 * _SCALE)),
    ("lsf_projections", int(6_990_000 * _SCALE)),
    ("lsf_feed_why_sources", int(2_570_000 * _SCALE)),
    *((f"small_{index:02d}", int(200_000 * _SCALE)) for index in range(14)),
)
_STATES = 50
_KEY = ("state", "run_id", "geo_id")
_COLUMNS = (
    Column(name="state", column_type=STRING_TYPE),
    Column(name="run_id", column_type=BIGINT_TYPE),
    Column(name="geo_id", column_type=BIGINT_TYPE),
    Column(name="quantity", column_type=STRING_TYPE),
    Column(name="value", column_type=BIGINT_TYPE),
    Column(name="note", column_type=STRING_TYPE),
)
#: the broker's reply cap a page is sized to after the first (the hub's ``l3_reply_max_bytes``)
_REPLY_CAP = 12 * 1024 * 1024
#: the rebuild timed against one stream: the shape's default
_STREAMS = RebuildConcurrency()
#: one stream, one table at a time: the old rebuild's shape
_SERIAL = RebuildConcurrency(tables=1, streams=1)
#: how many times one stream's rows a second the rebuild's read from L3 at the default concurrency must read. Measured
#: on the testament pool's Postgres 16, ~2.0M rows (2026-11-25): one stream (the old shape: one table after another,
#: one statement at a time) 1.40 s, 4 tables and 6 streams 0.52 s, 2.72x; the old code 1.41 s at any setting, 1.01x,
#: failing this
_MARGIN = 2.0
#: what the rebuild logs as its read from L3 ends, with its seconds (DeltaFolds._read_tables)
_READ_LINE = "every table of the rebuild read from L3"


class _Rail:
    """the L3 rail as a pod meets it, over the test's database: each statement on a connection of its own, its rows
    sent as CSV (``COPY``) and parsed into Arrow on a thread, off the interpreter lock, never a Python object a row;
    ``fetch_arrow_page`` answers the reply's bytes and the broker's cap, so the keyset reader sizes its pages to it."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def fetch(self, query: str, *params: Any) -> list[Any]:
        return list(await self._pool.fetch(query, *params))

    async def fetch_arrow(self, query: str, /, *params: Any, types: Mapping[str, Any]) -> Any:
        rows, _ = await self._arrow(query, params, types)
        return rows

    async def fetch_arrow_page(self, query: str, /, *params: Any, types: Mapping[str, Any]) -> L3Page:
        rows, size = await self._arrow(query, params, types)
        return L3Page(rows=rows, truncated=False, reply_bytes=size, reply_cap=_REPLY_CAP)

    async def _arrow(self, query: str, params: tuple[Any, ...], types: Mapping[str, Any]) -> tuple[Any, int]:
        chunks: list[bytes] = []

        async def take(chunk: bytes) -> None:
            chunks.append(bytes(chunk))

        async with self._pool.acquire() as conn:
            await conn.copy_from_query(query, *params, output=take, format="csv")
        data = b"".join(chunks)
        return await asyncio.to_thread(_parsed, data, types), len(data)


def _parsed(data: bytes, types: Mapping[str, Any]) -> Any:
    if not data:
        return pa.table({name: pa.array([], kind) for name, kind in types.items()})
    return pa_csv.read_csv(
        io.BytesIO(data),
        read_options=pa_csv.ReadOptions(column_names=list(types)),
        convert_options=pa_csv.ConvertOptions(
            column_types=dict(types), strings_can_be_null=True, quoted_strings_can_be_null=False
        ),
    )


class _Generations:
    async def current(self, table_name: str) -> str:
        return "inc:1"

    async def advance(self, table_name: str) -> str | None:
        raise AssertionError("a rebuild advances nothing")


class _Log:
    async def change(self, table_name: str, token: str) -> EpochChange | None:
        return None

    async def changes_after(self, table_name: str, after: str, upto: str) -> EpochWalk:
        return await walk_by_count(self, table_name, after, upto)


@dataclass
class _Db:
    rail: _Rail
    rows: int


@pytest.fixture
async def db(db_container: str) -> AsyncIterator[_Db]:
    """the table set at real size (scaled), in its own schema."""
    schema = f"rebuild_streams_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container, command_timeout=1800)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
        for name, rows in _TABLES:
            await conn.execute(
                f"CREATE TABLE {schema}.{name} (state TEXT, run_id BIGINT, geo_id BIGINT, quantity TEXT, value BIGINT, "
                "note TEXT, PRIMARY KEY (state, run_id, geo_id))"
            )
            await conn.execute(
                f"INSERT INTO {schema}.{name} SELECT 'S' || lpad((g % $2)::text, 2, '0'), g % 7, g, "  # noqa: S608
                "'q' || (g % 5), g * 3, 'note-' || (g % 97) FROM generate_series(1, $1) g",
                rows,
                _STATES,
            )
            await conn.execute(f"ANALYZE {schema}.{name}")
        pool = await asyncpg.create_pool(
            db_container, min_size=1, max_size=8, command_timeout=1800, server_settings={"search_path": schema}
        )
        try:
            yield _Db(rail=_Rail(pool), rows=sum(rows for _, rows in _TABLES))
        finally:
            await pool.close()
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


def _folded() -> list[FoldedTable]:
    return [
        FoldedTable.of(
            TableSchema(
                name=name,
                primary_key=_KEY,
                columns=list(_COLUMNS),
                l1_eviction=TEST_L1_TABLE,
                delta_files=True,
            ),
            scope=("state", "state:"),
        )
        for name, _ in _TABLES
    ]


def _states_sets(streams: int) -> list[list[str]]:
    states = [f"S{state:02d}" for state in range(_STATES)]
    return [states[index::streams] for index in range(streams)]


@pytest.mark.loop_timing
async def test_rebuild_streams_speed_at_real_size(db: _Db, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """the writer's rebuild of every missing full (DeltaFolds.start over an empty volume, the whole read declared in
    delta_fold._declared_pages and delta_fold._cold_start_pages, and its twin _schema_change_pages, the same read) with the default RebuildConcurrency reads its rows from
    L3 at least _MARGIN times as many a second as one stream a table at a time (the old rebuild's shape), and the whole
    rebuild, its publish included, takes within 2x of the bare work at the same concurrency: the same states read by
    keyset pages, ``streams`` reads at once, and written as a part a state.

    The read is timed by the rebuild's own line as its read ends: the publish after it (a part a state, each
    fingerprinted, in the fold's worker process) is the same work at any concurrency, and on this stand-in rail, many
    times faster than the broker (here ~870k rows/s a stream; the local stack's broker, ~37k), it would hide the read."""
    caplog.set_level(logging.INFO, logger="threetears.core.collections.delta_fold")
    types = rail_arrow_types(_COLUMNS)
    selected = ", ".join(types)
    runs = iter(range(1_000_000))

    def write_parts(root: Path, name: str, rows: Any) -> None:
        states = rows.column("state")
        for value in pc.unique(states).to_pylist():
            write_full_part(root, name, f"state:{value}", rows.filter(pc.equal(states, value)), "inc:1")

    async def bare() -> None:
        root = tmp_path / f"bare-{next(runs)}"
        streams = asyncio.Semaphore(_STREAMS.streams)

        async def one(name: str, states: list[str]) -> Any:
            async with streams:
                pages = [
                    page
                    async for page in keyset_arrow_pages(
                        db.rail,
                        f"SELECT {selected} FROM {name}",  # noqa: S608 - the test's own names
                        _KEY,
                        types=types,
                        where_any={"state": ("text[]", states)},
                        page_size=1000,
                        lead_array="text[]",
                    )
                ]
            return pa.concat_tables(pages)

        async with asyncio.TaskGroup() as group:
            tables = {
                name: [group.create_task(one(name, states)) for states in _states_sets(_STREAMS.streams)]
                for name, _ in _TABLES
            }
        for name, parts in tables.items():
            await asyncio.to_thread(write_parts, root, name, pa.concat_tables([part.result() for part in parts]))

    def rebuild(concurrency: RebuildConcurrency) -> Any:
        async def run() -> None:
            folds = DeltaFolds(
                tmp_path / f"rebuild-{next(runs)}", _folded(), retention="delete", copies=NO_COPIES, rebuild=concurrency
            )
            try:
                rebuilt = await folds.start(db.rail, _Generations(), changes=_Log(), version=1)
                assert sorted(rebuilt) == sorted(name for name, _ in _TABLES)
            finally:
                await folds.close()

        return run

    def read_seconds() -> float:
        timed = [
            float(record.__dict__["extra_data"]["seconds"])
            for record in caplog.records
            if record.getMessage() == _READ_LINE
        ]
        caplog.clear()
        assert timed, "the rebuild says its read from L3 as it ends"
        return min(timed)

    caplog.clear()
    serial = await best_seconds(rebuild(_SERIAL))
    serial_read = read_seconds()
    streamed = await best_seconds(rebuild(_STREAMS))
    streamed_read = read_seconds()
    baseline = await best_seconds(bare)
    print(
        f"rebuild of {db.rows} rows in {len(_TABLES)} tables: read from L3 by one stream {serial_read:.2f} s "
        f"({db.rows / serial_read:,.0f} rows/s), by {_STREAMS.tables} tables and {_STREAMS.streams} streams "
        f"{streamed_read:.2f} s ({db.rows / streamed_read:,.0f} rows/s, {serial_read / streamed_read:.2f}x); whole "
        f"rebuild {serial:.2f} s and {streamed:.2f} s, bare read and parts at {_STREAMS.streams} streams {baseline:.2f} s"
    )
    assert serial_read >= _MARGIN * streamed_read, (
        f"the rebuild's read at {_STREAMS.streams} streams took {db.rows / streamed_read:,.0f} rows/s, "
        f"{serial_read / streamed_read:.2f}x one stream's {db.rows / serial_read:,.0f} rows/s: under the {_MARGIN}x margin"
    )
    problem = budget_problem(f"a rebuild of {db.rows} rows at {_STREAMS.streams} streams", streamed, baseline)
    assert problem is None, problem
