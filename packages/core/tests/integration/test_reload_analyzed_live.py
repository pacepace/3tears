"""Integration: a table reloaded from files is analysed after its last request, so its first reads plan by its index.

A reload truncates a table and loads it again. Until the table is analysed the planner has no row count for
it, and YugabyteDB planned a read of a state's rows of some geographies (``state = $1 AND geo = ANY($2)``,
the reports pod's ``held_in``) as a primary-key scan and a sort: 157 ms against 0.4 ms analysed (b118). A
load of a reload's rows through a direct store now analyses the table after its last request (on the L3 rail
the broker does). This reloads a table of 356,928 rows and judges that read's plan at once, with the shared
plan check, on YugabyteDB; Postgres checks the rows.

Against the session ``db_container``; set ``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL to run it there.
"""

from __future__ import annotations

import io
import threading
import uuid
from collections.abc import AsyncIterator, Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import asyncpg
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from threetears.core.backends.protocol import LoadFile
from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.asyncpg_init import init_connection
from threetears.core.collections.complete_copy import read_l3_rows
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    DATETIMETZ_TYPE,
    STRING_TYPE,
    Column,
    Index,
    TableSchema,
    collection_for_schema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.plans import PlanRecorder, on_yugabyte, plan_problems

pytestmark = pytest.mark.integration

_STATE, _GEO = "join_state_code", "join_geo_id"

_SCHEMA = TableSchema(
    name="results",
    primary_key=("office_key", "candidate", _GEO),
    columns=[
        Column("office_key", STRING_TYPE),
        Column("candidate", STRING_TYPE),
        Column(_GEO, STRING_TYPE),
        Column(_STATE, STRING_TYPE),
        Column("votes", BIGINT_TYPE, nullable=True),
        Column("date_created", DATETIMETZ_TYPE, immutable=True, nullable=True),
        Column("date_updated", DATETIMETZ_TYPE, nullable=True),
    ],
    indexes=(Index("results_by_state", _STATE, _GEO, "office_key", "candidate", include=("votes",)),),
)


class _Row(BaseEntity):
    primary_key_field = _GEO


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - the stdlib's own name
        del format, args


@pytest.fixture
def served(tmp_path: Path) -> Iterator[tuple[Path, str]]:
    """a directory served over http on localhost, as a bucket answers a presigned GET."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_QuietHandler, directory=str(tmp_path)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield tmp_path, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
async def pool(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    schema = f"reload_{uuid.uuid4().hex[:8]}"
    held = await asyncpg.create_pool(
        db_container, min_size=1, max_size=2, init=init_connection, server_settings={"search_path": schema}
    )
    assert held is not None
    try:
        await held.execute(f"CREATE SCHEMA {schema}")
        await held.execute(
            f"CREATE TABLE results (office_key TEXT, candidate TEXT, {_GEO} TEXT, {_STATE} TEXT NOT NULL, votes BIGINT, "
            f"date_created TIMESTAMPTZ, date_updated TIMESTAMPTZ, PRIMARY KEY (office_key, candidate, {_GEO}))"
        )
        [index] = _SCHEMA.indexes
        await held.execute(
            f"CREATE INDEX {index.name} ON results ({', '.join(index.columns)}) INCLUDE ({', '.join(index.include)})"
        )
        yield held
    finally:
        await held.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await held.close()


def _files(served: tuple[Path, str]) -> list[LoadFile]:
    """the export: Texas's 254 counties and 49 states of 50, 33 offices, 4 candidates each, one file a state."""
    directory, base = served
    schema = pa.schema(
        [
            ("office_key", pa.string()),
            ("candidate", pa.string()),
            (_GEO, pa.string()),
            (_STATE, pa.string()),
            ("votes", pa.int64()),
        ]
    )
    files: list[LoadFile] = []
    for state, geos in [("TX", 254), *((f"S{n:02d}", 50) for n in range(1, 50))]:
        rows = [
            {"office_key": f"o{o}", "candidate": f"c{c}", _GEO: f"{state}-{g:03d}", _STATE: state, "votes": o * c * g}
            for o in range(1, 34)
            for c in range(1, 5)
            for g in range(1, geos + 1)
        ]
        sink = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), sink)
        (directory / f"{state}.parquet").write_bytes(sink.getvalue())
        files.append(LoadFile(url=f"{base}/{state}.parquet", rows=len(rows)))
    return files


async def test_a_reloaded_tables_first_read_of_some_geographies_plans_by_its_index(
    pool: asyncpg.Pool, served: tuple[Path, str]
) -> None:
    """Flips if a reload leaves the table unanalysed, so a read of a state's geographies scans and sorts."""
    registry = CollectionRegistry()
    registry.configure(l3_pool=SqlL3Backend(pool))
    collection = collection_for_schema(_SCHEMA, entity_class=_Row)(registry, DefaultCoreConfig(), None)
    await pool.execute("TRUNCATE results")

    written = await collection.load_files(
        _files(served), columns=["office_key", "candidate", _GEO, _STATE, "votes"], rows_per_transaction=20_000
    )

    assert written == 356_928
    geographies = [f"TX-{g:03d}" for g in range(1, 201)]
    async with pool.acquire() as conn:
        checked = PlanRecorder(conn)
        rows = await read_l3_rows(
            checked,
            "results",
            [column.name for column in _SCHEMA.columns],
            _SCHEMA.keyset_order({_STATE}),
            where={_STATE: "TX"},
            where_any={_GEO: ("text[]", geographies)},
        )
        assert len(rows) == 33 * 4 * 200
        if await on_yugabyte(conn):
            problems = plan_problems(checked.plans, scan_ratio=1.1, together=True)
            assert not problems, problems
