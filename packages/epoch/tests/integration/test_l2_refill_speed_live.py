"""Integration: the L2 refill's one whole read of L3 costs what the bare read and put of the same rows costs.

``L2FirstSupport.refill_from_l3`` (declared ``whole_read(kind="cold_start")``: L2 holds nothing for the table) reads
the table from L3 a keyset page at a time (``SchemaBackedCollection._l3_pages``) and puts each row into an empty L2 at
revision zero. Held within 2x (:func:`threetears.core.testing.calibrate.budget_problem`) of the bare operation timed
on the same database and the same NATS: the same keyset pages read by hand and each row put at revision zero under a
key of its own, into an L2 just as empty. Each round of either side writes under a fresh scope, so every put lands.

Timed runs belong under measure-lock. Against the session ``db_container`` and ``nats_container``;
``THREETEARS_REFILL_SPEED_ROWS`` sets the rows (default 20,000).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import AsyncIterator

import asyncpg
import pytest
from sqlalchemy import MetaData

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import CollectionRegistry
from threetears.core.collections.l2_first import L2_FIRST
from threetears.core.collections.schema_backed import STRING_TYPE, TableSchema, collection_for_schema
from threetears.core.collections.schema_backed import Column as SchemaColumn
from threetears.core.config import DefaultCoreConfig
from threetears.core.keyset import keyset_pages
from threetears.core.testing.calibrate import budget_problem
from threetears.nats import NatsClient, set_default_namespace

pytestmark = pytest.mark.integration

_ROWS = int(os.environ.get("THREETEARS_REFILL_SPEED_ROWS", "20000"))
#: the refiller's page (l2_first._REFILL_PAGE_ROWS)
_PAGE = 1000
_ROUNDS = 3


@pytest.fixture
async def seeded(db_container: str) -> AsyncIterator[tuple[asyncpg.Pool, str]]:
    """a schema holding one table of ``_ROWS`` rows; dropped after."""
    suffix = uuid.uuid4().hex[:8]
    schema, table = f"refill_{suffix}", f"refilled_{suffix}"
    pool = await asyncpg.create_pool(db_container, min_size=1, max_size=4, server_settings={"search_path": schema})
    assert pool is not None
    try:
        await pool.execute(f"CREATE SCHEMA {schema}")
        await pool.execute(f"CREATE TABLE {schema}.{table} (id TEXT PRIMARY KEY, owner TEXT, n TEXT)")
        await pool.execute(
            f"INSERT INTO {schema}.{table} SELECT 'r' || lpad(g::text, 8, '0'), 'o' || (g % 50), g::text "  # noqa: S608
            "FROM generate_series(1, $1) g",
            _ROWS,
        )
        await pool.execute(f"ANALYZE {schema}.{table}")
        yield pool, table
    finally:
        await pool.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await pool.close()


def _schema(table: str) -> TableSchema:
    return TableSchema(
        name=table,
        primary_key="id",
        columns=[
            SchemaColumn("id", STRING_TYPE),
            SchemaColumn("owner", STRING_TYPE, nullable=True),
            SchemaColumn("n", STRING_TYPE, nullable=True),
        ],
        on_conflict="update",
    )


async def test_refill_from_l3_speed_at_real_size(seeded: tuple[asyncpg.Pool, str], nats_container: str) -> None:
    """the refill L2Refiller.run_pass makes through L2Refiller.refill: L2FirstSupport.refill_from_l3, reading
    SchemaBackedCollection._l3_pages whole into an empty L2, within 2x the bare keyset read of the same pages and a
    revision-zero put of each row (run_pass and refill add one read of the head and two marks per table)."""
    pool, table = seeded
    schema = _schema(table)
    metadata = MetaData()
    schema.to_sqlalchemy_table(metadata)
    declared = type("Refilled", (collection_for_schema(schema),), {"l2_first": L2_FIRST})
    namespace = f"refill{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="refill"
    ) as nc:
        kv = await nc.kv_bucket(name="collections", create_if_missing=True)

        async def refill() -> int:
            l1 = SQLiteBackend(db_name=f"refill_{uuid.uuid4().hex[:8]}")
            l1.initialize(metadata)
            registry = CollectionRegistry()
            registry.configure(
                l1_backend=l1, l2_client=nc, l3_pool=SqlL3Backend(pool), kv_key_scope=f"s{uuid.uuid4().hex[:10]}"
            )
            rows: int = await declared(registry, DefaultCoreConfig()).l2_first_support.refill_from_l3(page_size=_PAGE)
            return rows

        async def bare() -> int:
            scope = f"s{uuid.uuid4().hex[:10]}"
            put = 0
            async for page in keyset_pages(pool, f"SELECT id, owner, n FROM {table}", ["id"], page_size=_PAGE):  # noqa: S608
                for row in page:
                    landed = await kv.update(
                        key=f"{scope}.{table}.{row['id']}", value=json.dumps(dict(row)).encode("utf-8"), revision=0
                    )
                    put += landed is not None
            return put

        took = baseline = float("inf")
        for _ in range(_ROUNDS):
            began = time.perf_counter()
            assert await bare() == _ROWS
            baseline = min(baseline, time.perf_counter() - began)
            began = time.perf_counter()
            assert await refill() == _ROWS
            took = min(took, time.perf_counter() - began)
    print(f"refill of {_ROWS} rows {took:.2f} s, bare read and put {baseline:.2f} s, {took / baseline:.2f}x")
    problem = budget_problem(f"an L2 refill (refill_from_l3) of {_ROWS} rows", took, baseline)
    assert problem is None, problem
