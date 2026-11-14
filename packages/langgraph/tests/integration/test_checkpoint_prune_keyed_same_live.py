"""Integration: the keyed prune and the maintenance walks remove and count exactly what the statements they replaced did.

``aprune(keep_latest)``, :func:`keep_latest_counts` and the maintenance walks read by key: guards decided per row's own
namespace, keyed batches of threads, keyset pages of threads. Before, each was one statement over a ``GROUP BY``
survivors join, or over the whole table. These tests keep those statements, verbatim, as the reference, seed
generated estates (threads in several namespaces, history of every length, the DeltaChannel marker on a newest
checkpoint, on an older one only, and no metadata at all, writes against kept, pruned and absent checkpoints,
threads holding writes alone, an owned set naming some), and require the same rows removed and the same figures,
seed after seed. Each seed is checked to hold every case, so no guard passes by never flipping.

Uses the session-scoped ``db_container`` fixture (PostgreSQL or YugabyteDB); a checkout without one skips cleanly.
"""

from __future__ import annotations

import random
import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from threetears.core.data.migrations import MigrationRunner
from threetears.langgraph.checkpoint import (
    DELTA_COUNTERS_METADATA_KEY,
    ThreeTierCheckpointSaver,
    keep_latest_counts,
)
from threetears.langgraph.checkpoint_maintenance import checkpoint_usage, delete_unowned_threads, prune_threads
from threetears.langgraph.checkpoint_scope import CheckpointScope
from threetears.langgraph.migrations import register as register_langgraph
from threetears.langgraph.protocols import AsyncpgPoolAdapter

pytestmark = pytest.mark.integration

_UNSCOPED = CheckpointScope.unscoped(reason="integration test of the un-tenanted keyspace")
_SEEDS = (1, 2, 3, 4, 5)
_MARKER = DELTA_COUNTERS_METADATA_KEY.encode()
_OWNED_SQL = "SELECT thread_id FROM owned"
_CHECKPOINT_KEY = "thread_id, checkpoint_ns, checkpoint_id"
_WRITE_KEY = "thread_id, checkpoint_ns, checkpoint_id, task_id, idx"


def _old_survivors_sql(*, marker_param: int, threads_param: int | None = None, held: bool = False) -> str:
    """the removed ``keep_latest_survivors_sql``, verbatim: per namespace the checkpoint ``keep_latest`` keeps."""
    thread_filter = f"WHERE thread_id = ANY(${threads_param}::text[]) " if threads_param is not None else ""
    marker = f"position(${marker_param}::bytea IN c.metadata_)"
    delta_test = f"{marker} > 0" if held else f"(c.metadata_ IS NULL OR {marker} = 0)"
    return (
        "SELECT c.thread_id, c.checkpoint_ns, c.checkpoint_id FROM checkpoints c "
        "JOIN (SELECT thread_id, checkpoint_ns, max(checkpoint_id) AS checkpoint_id FROM checkpoints "
        f"{thread_filter}GROUP BY thread_id, checkpoint_ns) newest "
        "ON c.thread_id = newest.thread_id AND c.checkpoint_ns = newest.checkpoint_ns "
        "AND c.checkpoint_id = newest.checkpoint_id "
        f"WHERE {delta_test}"
    )


_OLD_MATCH = "p.thread_id = k.thread_id AND p.checkpoint_ns = k.checkpoint_ns AND p.checkpoint_id < k.checkpoint_id"
_OLD_CHECKPOINT_BYTES = "pg_column_size(p.checkpoint) + coalesce(pg_column_size(p.metadata_), 0)"
_OLD_WRITE_BYTES = "pg_column_size(p.blob)"


async def _old_removed(conn: asyncpg.Connection, threads: list[str] | None) -> tuple[set[tuple[Any, ...]], ...]:
    """the keys the old ``DELETE ... USING (survivors)`` statements removed, read as a SELECT of the same join.

    :param conn: the schema's connection
    :ptype conn: asyncpg.Connection
    :param threads: the stored thread ids pruned, or None for every thread (the old report's whole read)
    :ptype threads: list[str] | None
    :return: the checkpoint keys and the write keys
    :rtype: tuple[set[tuple[Any, ...]], ...]
    """
    args: list[Any] = [_MARKER] if threads is None else [_MARKER, threads]
    survivors = _old_survivors_sql(marker_param=1, threads_param=None if threads is None else 2)
    checkpoints = await conn.fetch(
        f"SELECT p.thread_id, p.checkpoint_ns, p.checkpoint_id FROM checkpoints p JOIN ({survivors}) k ON {_OLD_MATCH}",
        *args,
    )
    writes = await conn.fetch(
        "SELECT p.thread_id, p.checkpoint_ns, p.checkpoint_id, p.task_id, p.idx "
        f"FROM checkpoint_writes p JOIN ({survivors}) k ON {_OLD_MATCH}",
        *args,
    )
    return {tuple(row) for row in checkpoints}, {tuple(row) for row in writes}


async def _old_prunable(conn: asyncpg.Connection, threads: list[str] | None) -> tuple[int, int, int, int]:
    """the removed maintenance ``_prunable``, verbatim: rows, write rows, bytes, namespaces held."""
    args: list[Any] = [_MARKER] if threads is None else [_MARKER, threads]
    threads_param = None if threads is None else 2
    survivors = _old_survivors_sql(marker_param=1, threads_param=threads_param)
    checkpoints = await conn.fetchrow(
        f"SELECT count(*) AS row_count, sum({_OLD_CHECKPOINT_BYTES}) AS bytes "
        f"FROM checkpoints p JOIN ({survivors}) k ON {_OLD_MATCH}",
        *args,
    )
    writes = await conn.fetchrow(
        f"SELECT count(*) AS row_count, sum({_OLD_WRITE_BYTES}) AS bytes "
        f"FROM checkpoint_writes p JOIN ({survivors}) k ON {_OLD_MATCH}",
        *args,
    )
    held = await conn.fetchrow(
        "SELECT count(*) AS held FROM ("
        + _old_survivors_sql(marker_param=1, threads_param=threads_param, held=True)
        + ") h",
        *args,
    )
    assert checkpoints is not None and writes is not None and held is not None
    return (
        int(checkpoints["row_count"]),
        int(writes["row_count"]),
        int(checkpoints["bytes"] or 0) + int(writes["bytes"] or 0),
        int(held["held"]),
    )


async def _old_usage(conn: asyncpg.Connection) -> dict[str, int]:
    """the removed ``checkpoint_usage`` whole statements, verbatim, for the figures they counted."""
    threads = await conn.fetchval(
        "SELECT count(*) AS threads FROM (SELECT thread_id FROM checkpoints UNION SELECT thread_id FROM checkpoint_writes) t"
    )
    with_history = await conn.fetchval(
        "SELECT count(DISTINCT thread_id) AS threads FROM "
        "(SELECT thread_id FROM checkpoints GROUP BY thread_id, checkpoint_ns HAVING count(*) > 1) multi"
    )
    owned = f"WITH owned AS (SELECT o.thread_id::text AS thread_id FROM ({_OWNED_SQL}) AS o(thread_id)) "
    not_owned = "NOT EXISTS (SELECT 1 FROM owned WHERE owned.thread_id = {a}.thread_id)"
    unowned_threads = await conn.fetchval(
        f"{owned}SELECT count(*) AS threads FROM (SELECT thread_id FROM checkpoints c WHERE {not_owned.format(a='c')} "
        f"UNION SELECT thread_id FROM checkpoint_writes w WHERE {not_owned.format(a='w')}) t"
    )
    unowned_checkpoints = await conn.fetchrow(
        f"{owned}SELECT count(*) AS row_count, sum(pg_column_size(c.checkpoint) + coalesce(pg_column_size(c.metadata_), 0)) "
        f"AS bytes FROM checkpoints c WHERE {not_owned.format(a='c')}"
    )
    unowned_writes = await conn.fetchrow(
        f"{owned}SELECT count(*) AS row_count, sum(pg_column_size(w.blob)) AS bytes "
        f"FROM checkpoint_writes w WHERE {not_owned.format(a='w')}"
    )
    assert unowned_checkpoints is not None and unowned_writes is not None
    prunable = await _old_prunable(conn, None)
    return {
        "threads": int(threads),
        "with_history": int(with_history),
        "prunable_checkpoints": prunable[0],
        "prunable_writes": prunable[1],
        "prunable_bytes": prunable[2],
        "held": prunable[3],
        "unowned_threads": int(unowned_threads),
        "unowned_checkpoints": int(unowned_checkpoints["row_count"]),
        "unowned_writes": int(unowned_writes["row_count"]),
        "unowned_bytes": (
            int(unowned_checkpoints["bytes"] or 0) + int(unowned_writes["bytes"] or 0)
            if not await _yugabyte(conn)
            else await _bytes_by_key(conn, sorted(await _old_unowned_threads(conn)))
        ),
    }


async def _yugabyte(conn: asyncpg.Connection) -> bool:
    """whether the connection is YugabyteDB's."""
    return "-YB-" in str(await conn.fetchval("SELECT version()"))


async def _bytes_by_key(conn: asyncpg.Connection, threads: list[str]) -> int:
    """the stored blob bytes of every row of ``threads``, read by thread.

    YugabyteDB's ``pg_column_size`` answers a value's 1-byte varlena header through a key lookup and a 4-byte one
    through a sequential scan (5 rows of 60 bytes: 65 against 77), so on YugabyteDB a whole statement's byte sum is
    not the keyed read's; this is the old figure read through the key, the rows the old statement named.
    """
    checkpoints = await conn.fetchval(
        "SELECT coalesce(sum(pg_column_size(checkpoint) + coalesce(pg_column_size(metadata_), 0)), 0) "
        "FROM checkpoints WHERE thread_id = ANY($1::text[])",
        threads,
    )
    writes = await conn.fetchval(
        "SELECT coalesce(sum(pg_column_size(blob)), 0) FROM checkpoint_writes WHERE thread_id = ANY($1::text[])",
        threads,
    )
    return int(checkpoints) + int(writes)


async def _old_unowned_threads(conn: asyncpg.Connection) -> set[str]:
    """the threads the old delete-unowned pager named: in either table, not in the owned set."""
    rows = await conn.fetch(
        "SELECT thread_id FROM checkpoints WHERE thread_id NOT IN (SELECT thread_id FROM owned) "
        "UNION SELECT thread_id FROM checkpoint_writes WHERE thread_id NOT IN (SELECT thread_id FROM owned)"
    )
    return {str(row["thread_id"]) for row in rows}


async def _keys(conn: asyncpg.Connection) -> tuple[set[tuple[Any, ...]], set[tuple[Any, ...]]]:
    """every key of both tables."""
    checkpoints = await conn.fetch(f"SELECT {_CHECKPOINT_KEY} FROM checkpoints")
    writes = await conn.fetch(f"SELECT {_WRITE_KEY} FROM checkpoint_writes")
    return {tuple(row) for row in checkpoints}, {tuple(row) for row in writes}


def _metadata(rng: random.Random, *, marked: bool) -> bytes | None:
    """a checkpoint's stored metadata: carrying the DeltaChannel marker, or not, or none at all."""
    if marked:
        return b'{"step": 3, "' + _MARKER + b'": {"log": 2}}'
    return rng.choice([None, b'{"step": 1}', b'{"source": "loop", "note": "' + b"x" * rng.randint(1, 400) + b'"}'])


async def _seed(conn: asyncpg.Connection, seed: int) -> dict[str, int]:
    """a generated estate, and how many of each case it holds.

    :param conn: the schema's connection
    :ptype conn: asyncpg.Connection
    :param seed: the generator's seed
    :ptype seed: int
    :return: the cases seeded (held namespaces, marker on an older checkpoint only, threads with history, ...)
    :rtype: dict[str, int]
    """
    rng = random.Random(seed)
    names = ["alpha", "Alpha", "beta-1", "beta_2", "Gamma", "delta.x", "epsilon", "zeta/9", "eta", "Theta", "iota"]
    rng.shuffle(names)
    cases = {"held": 0, "older_marked": 0, "history": 0, "single": 0, "orphan_writes": 0, "writes_only": 0}
    checkpoints: list[tuple[Any, ...]] = []
    writes: list[tuple[Any, ...]] = []
    for position, thread in enumerate(names):
        namespaces = rng.sample(["", "inner", "inner|tool:1", "Outer"], rng.randint(1, 3))
        for ns_position, ns in enumerate(namespaces):
            # the first thread's first namespace is held, the second's has the marker on an older checkpoint only
            depth = rng.randint(1, 6) if position > 1 or ns_position else 4
            held = (position == 0 and ns_position == 0) or (position > 1 and rng.random() < 0.2)
            older_marked = position == 1 and ns_position == 0
            ids = [f"cp-{n:02d}" for n in sorted(rng.sample(range(1, 60), depth))]
            for index, checkpoint_id in enumerate(ids):
                newest = index == len(ids) - 1
                marked = (held and newest) or (older_marked and index == 0)
                checkpoints.append(
                    (thread, ns, checkpoint_id, b"c" * rng.randint(1, 900), _metadata(rng, marked=marked))
                )
                for task in range(rng.randint(0, 2)):
                    for idx in range(rng.randint(1, 2)):
                        writes.append((thread, ns, checkpoint_id, f"task-{task}", idx, b"w" * rng.randint(1, 700)))
            if rng.random() < 0.3 or position == 2:
                # writes against no stored checkpoint: one older than any, one newer than the newest
                writes.append((thread, ns, "cp-00", "late", 0, b"o" * 40))
                writes.append((thread, ns, "cp-99", "early", 0, b"o" * 50))
                cases["orphan_writes"] += 1
            cases["held"] += held and depth > 1
            cases["older_marked"] += older_marked
            cases["history"] += depth > 1
            cases["single"] += depth == 1
    for thread in ("writes-only-a", "Writes-only-b"):
        for idx in range(rng.randint(1, 3)):
            writes.append((thread, "", "cp-05", "task-0", idx, b"q" * 30))
        cases["writes_only"] += 1
    await conn.executemany(
        "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, checkpoint, metadata_) "
        "VALUES ($1, $2, $3, $4, $5)",
        checkpoints,
    )
    await conn.executemany(
        "INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, blob) "
        "VALUES ($1, $2, $3, $4, $5, 'state', $6)",
        writes,
    )
    # "Writes-only-b" is never owned, so a thread holding writes alone is always among the unowned
    owned = [name for name in [*names, "writes-only-a"] if rng.random() < 0.5] or [names[0]]
    await conn.execute("CREATE TABLE owned (thread_id text PRIMARY KEY)")
    await conn.executemany("INSERT INTO owned VALUES ($1)", [(name,) for name in owned])
    return cases


class _Store:
    """the two calls the migration runner makes, over one connection."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self.conn = conn

    async def execute(self, sql: str, *params: Any) -> str:
        result: str = await self.conn.execute(sql, *params)
        return result

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        return [dict(r) for r in await self.conn.fetch(sql, *params)]


@pytest.fixture
async def estate(db_container: str, request: pytest.FixtureRequest) -> AsyncIterator[tuple[Any, ...]]:
    """a migrated schema seeded with the parametrized seed's estate: (executor, connection, cases)."""
    schema = f"prune_same_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container)
    pool = None
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        runner = MigrationRunner()
        register_langgraph(runner)
        await runner.apply_for_agent_schema(_Store(conn))
        cases = await _seed(conn, request.param)
        pool = await asyncpg.create_pool(db_container, min_size=1, max_size=2, server_settings={"search_path": schema})
        assert pool is not None
        yield AsyncpgPoolAdapter(pool), conn, cases
    finally:
        if pool is not None:
            await pool.close()
        await conn.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await conn.close()


def _every_case(cases: dict[str, int]) -> None:
    """the seed holds every case a guard decides, so no guard passes by never flipping."""
    assert all(count > 0 for count in cases.values()), cases


@pytest.mark.parametrize("estate", _SEEDS, indirect=True)
class TestKeyedPruneMatchesTheOldStatements:
    async def test_aprune_keep_latest_removes_the_rows_the_old_deletes_removed(self, estate: tuple[Any, ...]) -> None:
        executor, conn, cases = estate
        _every_case(cases)
        rng = random.Random(len(cases) * 7)
        threads = sorted({str(row["thread_id"]) for row in await conn.fetch("SELECT thread_id FROM checkpoints")})
        named = rng.sample(threads, max(1, len(threads) // 2)) + ["writes-only-a", "never-stored"]
        expected_checkpoints, expected_writes = await _old_removed(conn, named)
        _, _, _, old_held = await _old_prunable(conn, named)
        before_checkpoints, before_writes = await _keys(conn)
        assert expected_checkpoints and expected_writes, "the named threads hold nothing to prune"

        result = await ThreeTierCheckpointSaver(executor, scope=_UNSCOPED).aprune(named)

        after_checkpoints, after_writes = await _keys(conn)
        assert before_checkpoints - after_checkpoints == expected_checkpoints
        assert before_writes - after_writes == expected_writes
        assert (result.checkpoints_deleted, result.writes_deleted) == (
            len(expected_checkpoints),
            len(expected_writes),
        )
        assert result.namespaces_held == old_held

    async def test_keep_latest_counts_equal_the_old_report_statements(self, estate: tuple[Any, ...]) -> None:
        executor, conn, cases = estate
        _every_case(cases)
        threads = sorted({str(row["thread_id"]) for row in await conn.fetch("SELECT thread_id FROM checkpoints")})
        for named in (threads, threads[:3], threads[3:], ["writes-only-a"]):
            counts = await keep_latest_counts(executor, named)
            assert (counts.checkpoint_rows, counts.write_rows, counts.bytes, counts.namespaces_held) == (
                await _old_prunable(conn, named)
            ), named

    async def test_checkpoint_usage_counts_what_the_old_whole_statements_counted(self, estate: tuple[Any, ...]) -> None:
        executor, conn, cases = estate
        _every_case(cases)
        old = await _old_usage(conn)
        assert old["held"] > 0 and old["prunable_checkpoints"] > 0 and old["unowned_threads"] > 0

        usage = await checkpoint_usage(executor, owned_threads_sql=_OWNED_SQL)

        assert usage.unowned is not None
        assert {
            "threads": usage.threads,
            "with_history": usage.threads_with_history,
            "prunable_checkpoints": usage.prunable_checkpoint_rows,
            "prunable_writes": usage.prunable_write_rows,
            "prunable_bytes": usage.prunable_bytes,
            "held": usage.held_namespaces,
            "unowned_threads": usage.unowned.threads,
            "unowned_checkpoints": usage.unowned.checkpoint_rows,
            "unowned_writes": usage.unowned.write_rows,
            "unowned_bytes": usage.unowned.bytes,
        } == old

    @pytest.mark.parametrize("batch", [1, 3, 200])
    async def test_prune_threads_removes_what_the_old_whole_prune_removed(
        self, estate: tuple[Any, ...], batch: int
    ) -> None:
        executor, conn, cases = estate
        _every_case(cases)
        expected_checkpoints, expected_writes = await _old_removed(conn, None)
        with_history = (await _old_usage(conn))["with_history"]
        before_checkpoints, before_writes = await _keys(conn)
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED)

        dry = await prune_threads(saver, executor, batch=batch, dry_run=True)
        assert await _keys(conn) == (before_checkpoints, before_writes)
        real = await prune_threads(saver, executor, batch=batch)

        after_checkpoints, after_writes = await _keys(conn)
        assert before_checkpoints - after_checkpoints == expected_checkpoints
        assert before_writes - after_writes == expected_writes
        for sweep in (dry, real):
            assert (sweep.threads, sweep.checkpoint_rows, sweep.write_rows) == (
                with_history,
                len(expected_checkpoints),
                len(expected_writes),
            )

    async def test_delete_unowned_threads_deletes_the_threads_the_old_pager_named(
        self, estate: tuple[Any, ...]
    ) -> None:
        executor, conn, cases = estate
        _every_case(cases)
        gone = await _old_unowned_threads(conn)
        before_checkpoints, before_writes = await _keys(conn)
        assert "Writes-only-b" in gone and len(gone) > 1
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED)

        sweep = await delete_unowned_threads(saver, executor, owned_threads_sql=_OWNED_SQL, batch=2)

        after_checkpoints, after_writes = await _keys(conn)
        assert after_checkpoints == {key for key in before_checkpoints if key[0] not in gone}
        assert after_writes == {key for key in before_writes if key[0] not in gone}
        assert sweep.threads == len(gone)
