"""Integration: a versioned table's run replaced in ONE transaction, its run marker then its new rows, on a real database.

What the unit tests cannot prove: that the database itself, through every reader that judges run markers (the newest
relation a keyed read, a led read, a count and a probe name), hides the old run and shows the new one when both were
written in one commit, the marker first; that the change the commit settles reaches a derived cache as the marker's
prefix and the new rows' prefix; and that a transaction that fails before its COMMIT leaves the old run exactly as it
was, no marker and no new row ("stale OK, missing is P0": reports replaces an LSF state's run this way, so no reader ever
sees the state with no run).

Against the session ``db_container``: Postgres in the ``integration`` suite, YugabyteDB in ``integration_yugabyte``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import asyncpg
import pytest

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.asyncpg_init import init_connection
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.run_markers import RUN_MARKERS_SCHEMA
from threetears.core.collections.schema_backed import INT_TYPE, STRING_TYPE, Column, TableSchema, collection_for_schema
from threetears.core.collections.scope_epochs import ScopeEpochs, scope_epochs_collection
from threetears.core.config import DefaultCoreConfig
from threetears.core.ddl.row_epoch import EPOCH_COLUMN, row_versions
from threetears.core.ddl.run_markers import RUN_MARKERS_TABLE
from threetears.core.entities.base import BaseEntity
from threetears.core.keyed import counts_by, holds_any
from threetears.core.testing.l1_tables import TEST_L1_TABLE
from threetears.core.testing.plans import on_yugabyte

pytestmark = pytest.mark.integration

_EPOCHS = "marker_beside_scope_epochs"

#: an LSF table: its key leads with the scope, then the run
_PROJECTIONS = TableSchema(
    name="mb_projections",
    primary_key=("join_state_code", "run_id", "geo_id"),
    columns=[
        Column("join_state_code", STRING_TYPE),
        Column("run_id", INT_TYPE),
        Column("geo_id", STRING_TYPE),
        Column("value", INT_TYPE),
    ],
    row_epoch=row_versions("join_state_code", epochs=_EPOCHS, scope_prefix="state:"),
    l1_eviction=TEST_L1_TABLE,
)
_ORDER = ["join_state_code", "run_id", "geo_id"]
_GEOS = 50


class _Projection(BaseEntity):
    primary_key_field = "geo_id"


@dataclass
class _Held:
    pool: asyncpg.Pool
    epochs: ScopeEpochs
    projections: Any
    heard: list[Any]


@pytest.fixture
async def held(db_container: str) -> AsyncIterator[_Held]:
    schema = f"marker_beside_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(db_container)
    try:
        await admin.execute(f"CREATE SCHEMA {schema}")
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(
        db_container, min_size=1, max_size=4, init=init_connection, server_settings={"search_path": schema}
    )
    assert pool is not None
    try:
        async with pool.acquire() as conn:
            yugabyte = await on_yugabyte(conn)
        epochs_class = scope_epochs_collection(_EPOCHS)
        for table in (epochs_class.schema, _PROJECTIONS, RUN_MARKERS_SCHEMA):
            for statement in table.create_statements(yugabyte=yugabyte):
                await pool.execute(statement)
        registry = CollectionRegistry()
        registry.configure(l3_pool=SqlL3Backend(pool))
        heard: list[Any] = []
        registry.register_derived_cache(
            _PROJECTIONS.name,
            on_row=lambda message: heard.append(("row", tuple(message.ids))),
            on_table_dropped=lambda: heard.append("dropped"),
            on_partition=lambda prefix: heard.append(dict(prefix)),
        )
        config = DefaultCoreConfig()
        yield _Held(
            pool=pool,
            epochs=ScopeEpochs(epochs_class(registry, config, None)),
            projections=collection_for_schema(_PROJECTIONS, entity_class=_Projection)(registry, config, None),
            heard=heard,
        )
    finally:
        await pool.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await pool.close()


def _run(run: int) -> list[dict[str, Any]]:
    return [
        {"join_state_code": "PA", "run_id": run, "geo_id": f"g{n:03d}", "value": run * 1000 + n} for n in range(_GEOS)
    ]


async def _in_version(held: _Held, step: Callable[[Any], Awaitable[None]]) -> None:
    """``step`` on a transaction that took a fresh write version's turn."""
    async with held.pool.acquire() as conn, CallerTransaction(conn):
        version = await held.epochs.begin(conn=conn)
    async with held.pool.acquire() as conn, CallerTransaction(conn):
        await held.epochs.touch(version, ["state:PA"], conn=conn)
        await step(conn)


async def _replace(held: _Held, old: int, new: int, conn: Any) -> None:
    """the one-transaction replace: the old run's marker, then the new run's rows."""
    assert await held.projections.delete_prefixes([{"join_state_code": "PA", "run_id": old}], conn=conn) == 1
    await held.projections.save_rows(_run(new), conn=conn)


async def _runs_read(held: _Held) -> dict[str, Any]:
    """what each reader that judges run markers answers for the state: led, keyed, counted, probed."""
    led = await held.projections.read_rows_led_by(["PA"])
    return {
        "led": sorted({row["run_id"] for row in led}),
        "led_rows": len(led),
        "keyed_old": await held.projections.fetch_from_store(("PA", 7, "g000")),
        "keyed_new": (await held.projections.fetch_from_store(("PA", 8, "g000")) or {}).get("value"),
        "counts": await counts_by(
            held.pool, _PROJECTIONS.name, column="run_id", by={"join_state_code": "PA"}, cast="integer"
        ),
        "holds_old": await holds_any(
            held.pool, _PROJECTIONS.name, order=_ORDER, by={"join_state_code": "PA", "run_id": 7}
        ),
        "holds_new": await holds_any(
            held.pool, _PROJECTIONS.name, order=_ORDER, by={"join_state_code": "PA", "run_id": 8}
        ),
    }


async def test_a_run_replaced_in_one_transaction_hides_the_old_run_and_shows_the_new_one(held: _Held) -> None:
    await _in_version(held, lambda conn: held.projections.save_rows(_run(7), conn=conn))
    held.heard.clear()
    await _in_version(held, lambda conn: _replace(held, 7, 8, conn))
    assert await _runs_read(held) == {
        "led": [8],
        "led_rows": _GEOS,
        "keyed_old": None,
        "keyed_new": 8000,
        "counts": {8: _GEOS},
        "holds_old": False,
        "holds_new": True,
    }
    # the marker is older than every row the transaction appended after it
    marker = await held.pool.fetchval(f"SELECT max({EPOCH_COLUMN}) FROM {RUN_MARKERS_TABLE}")  # noqa: S608 -- test names
    oldest_new = await held.pool.fetchval(
        f"SELECT min({EPOCH_COLUMN}) FROM {_PROJECTIONS.name} WHERE run_id = 8"  # noqa: S608 -- test names
    )
    assert marker < oldest_new
    # the commit settled once, by the marker's prefix and the new rows', never a key at a time
    assert held.heard == [{"join_state_code": "PA", "run_id": "7"}, {"join_state_code": "PA", "run_id": "8"}]


async def test_a_replace_that_fails_before_its_commit_leaves_the_old_run_intact(held: _Held) -> None:
    await _in_version(held, lambda conn: held.projections.save_rows(_run(7), conn=conn))
    await _in_version(held, lambda conn: _replace(held, 7, 8, conn))
    before = await _runs_read(held)

    async def crashing(conn: Any) -> None:
        await _replace(held, 8, 9, conn)
        raise RuntimeError("the writer died before its COMMIT")

    with pytest.raises(RuntimeError, match="before its COMMIT"):
        await _in_version(held, crashing)
    assert await _runs_read(held) == before
    assert before["led"] == [8] and before["led_rows"] == _GEOS
    assert await held.pool.fetchval(f"SELECT count(*) FROM {RUN_MARKERS_TABLE}") == 1  # noqa: S608 -- test names
    assert await held.pool.fetchval(f"SELECT count(*) FROM {_PROJECTIONS.name} WHERE run_id = 9") == 0  # noqa: S608
