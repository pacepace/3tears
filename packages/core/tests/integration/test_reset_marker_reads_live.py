"""Integration: a versioned table reset in ONE transaction, its reset marker then its new rows, on a real database.

What the unit tests cannot prove: that the database itself, through every reader (a keyed read, a led read, a keyed
batch, a page, a count, a probe), hides every version older than the table's reset marker and shows the rows the same
transaction appended after it; that the commit settles a derived cache as a reset of the whole table; that a restore
hears the reset by the marker's key; and that a transaction that fails before its COMMIT leaves every old row visible,
no marker written ("stale OK, missing is P0": reports replaces a race-keyed live table this way, with no ``TRUNCATE``).
And the plan: the reset is ONE keyed probe of the run-marker table's primary key, read once a statement.

Against the session ``db_container``: Postgres in the ``integration`` suite, YugabyteDB in ``integration_yugabyte``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import asyncpg
import pyarrow as pa
import pytest

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.asyncpg_init import init_connection
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.epoch_change import RESET_REASON
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.run_markers import RUN_MARKERS_SCHEMA
from threetears.core.collections.schema_backed import INT_TYPE, STRING_TYPE, Column, TableSchema, collection_for_schema
from threetears.core.collections.scope_epochs import ScopeEpochs, scope_epochs_collection
from threetears.core.collections.table_restore import restored_tables, table_resets
from threetears.core.config import DefaultCoreConfig
from threetears.core.ddl.row_epoch import EPOCH_COLUMN, row_versions
from threetears.core.ddl.run_markers import RUN_MARKERS_TABLE
from threetears.core.entities.base import BaseEntity
from threetears.core.keyed import KeyBatch, Sort, counts_by, holds_any, keyset_rows, read_keyed_arrow
from threetears.core.testing.l1_tables import TEST_L1_TABLE
from threetears.core.testing.plans import PlanRecorder, on_yugabyte, plan_uses_index

pytestmark = pytest.mark.integration

_EPOCHS = "reset_marker_scope_epochs"

#: a live results table as reports declares one: keyed by the race, its scope (the state) no part of its key, so no
#: run marker can name it
_RESULTS = TableSchema(
    name="rm_results",
    primary_key=("race_id", "candidate"),
    columns=[
        Column("race_id", STRING_TYPE),
        Column("candidate", STRING_TYPE),
        Column("join_state_code", STRING_TYPE),
        Column("votes", INT_TYPE),
    ],
    row_epoch=row_versions("join_state_code", epochs=_EPOCHS, scope_prefix="state:"),
    l1_eviction=TEST_L1_TABLE,
)
_ORDER = ["race_id", "candidate"]
_CANDIDATES = 20


class _Result(BaseEntity):
    primary_key_field = "candidate"


@dataclass
class _Held:
    pool: asyncpg.Pool
    schema: str
    epochs: ScopeEpochs
    results: Any
    heard: list[Any]


@pytest.fixture
async def held(db_container: str) -> AsyncIterator[_Held]:
    schema = f"reset_marker_{uuid.uuid4().hex[:8]}"
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
        for table in (epochs_class.schema, _RESULTS, RUN_MARKERS_SCHEMA):
            for statement in table.create_statements(yugabyte=yugabyte):
                await pool.execute(statement)
        registry = CollectionRegistry()
        registry.configure(l3_pool=SqlL3Backend(pool))
        heard: list[Any] = []
        registry.register_derived_cache(
            _RESULTS.name,
            on_row=lambda message: heard.append(("row", tuple(message.ids))),
            on_table_dropped=lambda: heard.append("dropped"),
            on_partition=lambda prefix: heard.append(dict(prefix)),
            on_table_reset=lambda token: heard.append("reset"),
        )
        config = DefaultCoreConfig()
        yield _Held(
            pool=pool,
            schema=schema,
            epochs=ScopeEpochs(epochs_class(registry, config, None)),
            results=collection_for_schema(_RESULTS, entity_class=_Result)(registry, config, None),
            heard=heard,
        )
    finally:
        await pool.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await pool.close()


def _rows(race: str, votes: int) -> list[dict[str, Any]]:
    return [
        {"race_id": race, "candidate": f"c{n:03d}", "join_state_code": "PA", "votes": votes + n}
        for n in range(_CANDIDATES)
    ]


async def _in_version(held: _Held, step: Callable[[Any], Awaitable[None]]) -> None:
    """``step`` on a transaction that took a fresh write version's turn."""
    async with held.pool.acquire() as conn, CallerTransaction(conn):
        version = await held.epochs.begin(conn=conn)
    async with held.pool.acquire() as conn, CallerTransaction(conn):
        await held.epochs.touch(version, ["state:PA"], conn=conn)
        await step(conn)


async def _reset(held: _Held, rows: list[dict[str, Any]], conn: Any) -> None:
    """the one-transaction replace of the whole table: its reset marker, then its new rows."""
    assert await held.results.reset_table(conn=conn) == 1
    await held.results.save_rows(rows, conn=conn)


async def _read(held: _Held) -> dict[str, Any]:
    """what each reader answers: led, keyed, a keyed batch, a page, counted, probed."""
    led = await held.results.read_rows_led_by(["r1", "r2"], columns=["race_id", "candidate", "votes"])
    batch = await read_keyed_arrow(
        held.pool,
        _RESULTS.name,
        keys=KeyBatch(tuple(_ORDER), [["r1", "r2"], ["c000", "c000"]], ("text", "text")),
        types={"race_id": pa.string(), "votes": pa.int64()},
    )
    paged = await keyset_rows(
        held.pool,
        _RESULTS.name,
        columns=["race_id", "candidate", "votes"],
        by={"race_id": "r1"},
        order=[Sort("candidate")],
        page=7,
    )
    return {
        "led": sorted({(row["race_id"], row["votes"] // 1000) for row in led}),
        "led_rows": len(led),
        "keyed_r1": (await held.results.fetch_from_store(("r1", "c000")) or {}).get("votes"),
        "keyed_r2": (await held.results.fetch_from_store(("r2", "c000")) or {}).get("votes"),
        "batch": sorted((row["race_id"], row["votes"]) for table in batch for row in table.to_pylist()),
        "paged": len(paged),
        "counts": await counts_by(held.pool, _RESULTS.name, column="candidate", by={"race_id": "r1"}),
        "holds_r1": await holds_any(held.pool, _RESULTS.name, order=_ORDER, by={"race_id": "r1"}),
        "holds_r2": await holds_any(held.pool, _RESULTS.name, order=_ORDER, by={"race_id": "r2"}),
    }


async def test_a_reset_in_one_transaction_hides_every_old_row_and_shows_the_new_ones(held: _Held) -> None:
    await _in_version(held, lambda conn: held.results.save_rows(_rows("r1", 1000) + _rows("r2", 2000), conn=conn))
    held.heard.clear()
    await _in_version(held, lambda conn: _reset(held, _rows("r1", 3000), conn))
    read = await _read(held)
    assert read["led"] == [("r1", 3)]
    assert read["led_rows"] == _CANDIDATES
    assert read["keyed_r1"] == 3000 and read["keyed_r2"] is None
    assert read["batch"] == [("r1", 3000)]
    assert read["paged"] == _CANDIDATES
    assert len(read["counts"]) == _CANDIDATES and set(read["counts"].values()) == {1}
    assert read["holds_r1"] and not read["holds_r2"]
    # the marker is older than every row the transaction appended after it
    marker = await held.pool.fetchval(f"SELECT max({EPOCH_COLUMN}) FROM {RUN_MARKERS_TABLE}")  # noqa: S608 -- test names
    oldest_new = await held.pool.fetchval(
        f"SELECT min({EPOCH_COLUMN}) FROM {_RESULTS.name} WHERE votes >= 3000"  # noqa: S608 -- test names
    )
    assert marker < oldest_new
    # nothing was deleted: every old version is still stored, hidden
    assert await held.pool.fetchval(f"SELECT count(*) FROM {_RESULTS.name}") == 3 * _CANDIDATES  # noqa: S608
    # the commit settled once, as a reset of the whole table, never a key at a time
    assert held.heard == ["reset"]
    # a restore hears it by the marker's own key
    tables = [table for table in await restored_tables(held.pool) if table.schema == held.schema]
    (table,) = [table for table in tables if table.table == _RESULTS.name]
    (change,) = await table_resets(held.pool, table, 0)
    assert change.whole_table and change.reason == RESET_REASON


async def test_a_reset_that_fails_before_its_commit_leaves_every_old_row_visible(held: _Held) -> None:
    await _in_version(held, lambda conn: held.results.save_rows(_rows("r1", 1000) + _rows("r2", 2000), conn=conn))
    before = await _read(held)

    async def crashing(conn: Any) -> None:
        await _reset(held, _rows("r1", 3000), conn)
        raise RuntimeError("the writer died before its COMMIT")

    with pytest.raises(RuntimeError, match="before its COMMIT"):
        await _in_version(held, crashing)
    assert await _read(held) == before
    assert before["led"] == [("r1", 1), ("r2", 2)] and before["led_rows"] == 2 * _CANDIDATES
    assert await held.pool.fetchval(f"SELECT count(*) FROM {RUN_MARKERS_TABLE}") == 0  # noqa: S608 -- test names


def _nodes(
    plan: Mapping[str, Any], parents: tuple[str, ...] = ()
) -> Iterator[tuple[Mapping[str, Any], tuple[str, ...]]]:
    """every node of a plan, depth first, with the parent relationships above it."""
    yield plan, parents
    for child in plan.get("Plans", ()):
        yield from _nodes(child, (*parents, str(child.get("Parent Relationship", ""))))


async def test_the_reset_is_one_keyed_probe_of_the_marker_table_read_once_a_statement(held: _Held) -> None:
    """Flips if a read finds the table's newest reset by anything but a probe of the run-marker table's primary key (a
    scan of the markers), or reads it a row at a time (a SubPlan, a loop a row) instead of once a statement."""
    await _in_version(held, lambda conn: held.results.save_rows(_rows("r1", 1000) + _rows("r2", 2000), conn=conn))
    await _in_version(held, lambda conn: _reset(held, _rows("r1", 3000), conn))
    async with held.pool.acquire() as conn:
        checked = PlanRecorder(conn)
        await holds_any(checked, _RESULTS.name, order=_ORDER, by={"race_id": "r1"})
        await counts_by(checked, _RESULTS.name, column="candidate", by={"race_id": "r1"})
        await keyset_rows(
            checked,
            _RESULTS.name,
            columns=["race_id", "candidate", "votes"],
            by={"race_id": "r1"},
            order=[Sort("candidate")],
            page=7,
        )
    assert checked.plans
    for statement in checked.plans:
        probes = [
            (node, parents)
            for node, parents in _nodes(statement.plan)
            if node.get("Relation Name") == RUN_MARKERS_TABLE
        ]
        assert len(probes) == 1, statement.sql
        ((probe, parents),) = probes
        assert plan_uses_index(probe, f"{RUN_MARKERS_TABLE}_pkey"), probe
        assert "InitPlan" in parents and "SubPlan" not in parents, parents
        # once a statement at most (0: the outer read asked no row of it)
        assert probe.get("Actual Loops", 0) <= 1, probe
