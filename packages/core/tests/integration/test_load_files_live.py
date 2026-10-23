"""Integration: a table loaded from parquet files served over HTTP, on a real Postgres.

What the unit tests' recording connections cannot prove: that a warehouse export's parquet -- an
int64, a decimal, a timestamp with a zone, a uuid written as text -- copies into the table through
asyncpg's binary ``copy_records_to_table`` on a pool initialised as every 3tears pool is; that the
files are fetched over ``http`` as the broker fetches a presigned URL; that the rows commit a
transaction at a time, so a key already held stops the load with the transactions before it kept;
and that the collection's ``load_files`` writes what a read then answers.

Uses the session-scoped ``db_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import io
import threading
import uuid
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from decimal import Decimal
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
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    INT_TYPE,
    NUMERIC_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity

pytestmark = pytest.mark.integration

_SCHEMA = TableSchema(
    name="lsf_rows",
    primary_key=("race", "geo"),
    columns=[
        Column("race", STRING_TYPE),
        Column("geo", STRING_TYPE),
        Column("votes", INT_TYPE, nullable=True),
        Column("source_id", UUID_TYPE, nullable=True),
        Column("share", NUMERIC_TYPE, nullable=True, precision=38, scale=12),
        Column("reported_at", DATETIMETZ_TYPE, nullable=True),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
)

_WHEN = datetime(2026, 10, 22, 12, tzinfo=UTC)
_COLUMNS = ["geo", "votes", "source_id", "share", "reported_at"]


class _Row(BaseEntity):
    primary_key_field = "geo"


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
    schema = f"loadfiles_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(db_container)
    try:
        await admin.execute(f"CREATE SCHEMA {schema}")
    finally:
        await admin.close()
    held = await asyncpg.create_pool(
        db_container, min_size=1, max_size=2, init=init_connection, server_settings={"search_path": schema}
    )
    assert held is not None
    try:
        await held.execute(
            "CREATE TABLE lsf_rows (race TEXT, geo TEXT, votes BIGINT, source_id UUID, share NUMERIC(38, 12), "
            "reported_at TIMESTAMPTZ, date_created TIMESTAMPTZ NOT NULL, date_updated TIMESTAMPTZ NOT NULL, "
            "PRIMARY KEY (race, geo))"
        )
        yield held
    finally:
        await held.close()


def _export_rows(start: int, count: int) -> list[dict[str, Any]]:
    return [
        {
            "geo": f"{i:05d}",
            "votes": i,
            "source_id": str(uuid.UUID(int=i)),
            "share": Decimal(i) / 8,
            "reported_at": datetime(2026, 11, 3, 20, tzinfo=UTC),
        }
        for i in range(start, start + count)
    ]


def _write(served: tuple[Path, str], name: str, rows: list[dict[str, Any]]) -> LoadFile:
    directory, base = served
    schema = pa.schema(
        [
            ("geo", pa.string()),
            ("votes", pa.int64()),
            ("source_id", pa.string()),
            ("share", pa.decimal128(38, 12)),
            ("reported_at", pa.timestamp("us", tz="UTC")),
        ]
    )
    sink = io.BytesIO()
    # reversed: a warehouse writes in no key order, and the load orders by key
    pq.write_table(pa.Table.from_pylist(list(reversed(rows)), schema=schema), sink)
    (directory / name).write_bytes(sink.getvalue())
    return LoadFile(url=f"{base}/{name}", rows=len(rows))


def _collection(pool: asyncpg.Pool) -> Any:
    registry = CollectionRegistry()
    registry.configure(l3_pool=SqlL3Backend(pool))
    return collection_for_schema(_SCHEMA, entity_class=_Row)(registry, DefaultCoreConfig(), None)


async def test_a_table_is_loaded_from_files_fetched_over_http(pool: asyncpg.Pool, served: tuple[Path, str]) -> None:
    collection = _collection(pool)
    files = [_write(served, "0000.parquet", _export_rows(0, 250)), _write(served, "0001.parquet", _export_rows(250, 3))]
    loaded = await collection.load_files(
        files,
        columns=_COLUMNS,
        constants={"race": "va-gov", "date_created": _WHEN, "date_updated": _WHEN},
        rows_per_transaction=100,
        timeout_ms=30_000,
    )
    assert loaded == 253
    assert await pool.fetchval("SELECT count(*) FROM lsf_rows WHERE race = 'va-gov'") == 253
    row = await pool.fetchrow("SELECT * FROM lsf_rows WHERE race = 'va-gov' AND geo = '00007'")
    assert row is not None
    assert row["votes"] == 7
    assert row["source_id"] == uuid.UUID(int=7)
    assert row["share"] == Decimal("0.875")
    assert row["reported_at"] == datetime(2026, 11, 3, 20, tzinfo=UTC)
    assert row["date_created"] == _WHEN
    read = await collection.ensure(("va-gov", "00007"))
    assert read is not None and read["votes"] == 7


async def test_a_key_already_held_stops_the_load_with_the_transactions_before_it_kept(
    pool: asyncpg.Pool, served: tuple[Path, str]
) -> None:
    store = SqlL3Backend(pool)
    store.register_schema("lsf_rows", _SCHEMA)
    await pool.execute(
        "INSERT INTO lsf_rows (race, geo, date_created, date_updated) VALUES ('r', '00150', now(), now())"
    )
    files = [_write(served, "0000.parquet", _export_rows(0, 250))]
    with pytest.raises(asyncpg.UniqueViolationError):
        await store.load_files(
            "lsf_rows",
            files,
            columns=_COLUMNS,
            constants={"race": "r", "date_created": _WHEN, "date_updated": _WHEN},
            rows_per_transaction=100,
            timeout_ms=30_000,
        )
    # rows 0-99 committed; the second transaction, holding 00150, rolled back whole
    assert await pool.fetchval("SELECT count(*) FROM lsf_rows WHERE race = 'r'") == 101


async def test_a_file_holding_other_than_its_declared_rows_writes_nothing(
    pool: asyncpg.Pool, served: tuple[Path, str]
) -> None:
    store = SqlL3Backend(pool)
    store.register_schema("lsf_rows", _SCHEMA)
    file = _write(served, "0000.parquet", _export_rows(0, 5))
    with pytest.raises(ValueError, match="holds 5 rows; the load declares 6"):
        await store.load_files(
            "lsf_rows",
            [LoadFile(url=file.url, rows=6)],
            columns=_COLUMNS,
            constants={"race": "r", "date_created": _WHEN, "date_updated": _WHEN},
            rows_per_transaction=100,
            timeout_ms=30_000,
        )
    assert await pool.fetchval("SELECT count(*) FROM lsf_rows") == 0
