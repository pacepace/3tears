"""Integration: a ranked read joined to its parents costs within twice the ranked read alone.

``read_nearest`` with plain ``LEFT`` joins sends one statement: the ranked read as a derived table, each ranked row
``LEFT JOIN LATERAL``-ed to its parent by key (``threetears.core.keyed.point``). It replaced the two-step read (ranked
rows, then a keyed batch of the parents), whose second round trip put memory's FTS search over the hand-written join
it replaced: 5.96 ms against 4.92 at 20k ``media_content`` rows of one user on PostgreSQL. The one statement measured
1.01-1.05x the bare ranked read on PostgreSQL and at most 1.14x on YugabyteDB (2 users x 10k rows, 1024-dim,
2026-11-14). This test is that measurement's guard: the joined read of a 10k-row owner beside another, by text rank
and by vector similarity, timed turn about with the same read unjoined on the same session, within the default 2x.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from threetears.core.data.gin import gin_filter
from threetears.core.keyed import Fragment, Joined, Nearest, TextRank, read_nearest
from threetears.core.testing.calibrate import best_seconds_interleaved, budget_problem

pytestmark = pytest.mark.integration

_DIM = 16
#: rows an owner has; ten share each parent
_ROWS = 10_000
_PER_PARENT = 10
_LIMIT = 20
#: reads a timed run makes: one read is a few milliseconds, under the timer's noise alone
_READS = 20
_TEXT = "apple"
_QUERY = "[" + ",".join(["1", *["0"] * (_DIM - 1)]) + "]"


def _seed(owner: uuid.UUID) -> list[tuple[str, tuple[Any, ...]]]:
    """the statements seeding ``owner``'s parents and rows: one in ten rows says ``apple``."""
    vector = (
        "('[' || array_to_string(ARRAY(SELECT random() + g * 0 FROM generate_series(1, %d)), ',') || ']')::vector"
        % _DIM
    )
    text = "'alpha beta word' || (g % 97) || CASE WHEN g % 10 = 0 THEN ' apple' ELSE '' END"
    return [
        (
            "INSERT INTO parent (owner, parent_id, title) SELECT $1::text::uuid, md5($1::text || g::text)::uuid, 'file ' || g "
            f"FROM generate_series(0, {_ROWS // _PER_PARENT - 1}) AS g",
            (str(owner),),
        ),
        (
            "INSERT INTO child (owner, id, parent_id, body, search_vector, embedding) SELECT $1::text::uuid, gen_random_uuid(), "
            f"md5($1::text || (g / {_PER_PARENT})::text)::uuid, {text}, to_tsvector('english', {text}), {vector} "
            f"FROM generate_series(0, {_ROWS - 1}) AS g",
            (str(owner),),
        ),
    ]


@pytest.fixture
async def world(db_container: str) -> AsyncIterator[tuple[str, uuid.UUID]]:
    """a schema of its own: ``parent (owner, parent_id, title)`` and ``child (owner, id, parent_id, body,
    search_vector, embedding)``, two owners of 10,000 rows each; its name and the owner read."""
    name = f"join_leg_{uuid.uuid4().hex[:10]}"
    owners = [uuid.uuid4(), uuid.uuid4()]
    held = await asyncpg.connect(db_container)
    try:
        await held.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await held.execute(f'CREATE SCHEMA "{name}"')
        await held.execute(f'SET search_path TO "{name}", public')
        await held.execute(
            "CREATE TABLE parent (owner uuid NOT NULL, parent_id uuid NOT NULL, title text NOT NULL,"
            " PRIMARY KEY (owner, parent_id))"
        )
        await held.execute(
            f"CREATE TABLE child (owner uuid NOT NULL, id uuid NOT NULL, parent_id uuid NOT NULL, body text NOT NULL,"
            f" search_vector tsvector NOT NULL, embedding vector({_DIM}) NOT NULL, PRIMARY KEY (owner, id))"
        )
        await held.execute("SELECT setseed(0.3)")
        for owner in owners:
            for sql, params in _seed(owner):
                await held.execute(sql, *params)
        await held.execute("ANALYZE parent")
        await held.execute("ANALYZE child")
        yield name, owners[0]
    finally:
        await held.execute(f'DROP SCHEMA IF EXISTS "{name}" CASCADE')
        await held.close()


def _reads(owner: uuid.UUID) -> dict[str, dict[str, Any]]:
    """each ranked read's arguments, by name: by text rank (matched rows only) and by vector similarity."""
    return {
        "text": {
            "where": [Fragment(gin_filter("t.search_vector @@ websearch_to_tsquery('english', $1)"), (_TEXT,))],
            "nearest": TextRank("search_vector", _TEXT),
        },
        "vector": {"where": [], "nearest": Nearest("embedding", _QUERY, similarity="similarity")},
    }


@pytest.mark.loop_timing
async def test_the_joined_ranked_read_costs_within_twice_the_ranked_read_alone(
    db_container: str, world: tuple[str, uuid.UUID]
) -> None:
    """Flips if ``read_nearest`` joined ``LEFT`` to the rows' parents takes more than twice the same ranked read
    unjoined, on the same session, by text rank or by vector similarity; sends more than one statement (the two-step
    read's second round trip back); or answers other rows than the ranked read, or a row without its parent."""
    schema, owner = world
    conn = await asyncpg.connect(db_container, server_settings={"search_path": f'"{schema}", public'})
    try:
        for name, read in _reads(owner).items():
            columns = ("id", "parent_id")
            joins = [
                Joined(
                    "parent", on={"parent_id": "parent_id"}, by={"owner": owner}, columns={"title": "title"}, left=True
                )
            ]

            async def bare(read: dict[str, Any] = read) -> list[Any]:
                return [
                    await read_nearest(conn, "child", by={"owner": owner}, limit=_LIMIT, columns=columns, **read)
                    for _ in range(_READS)
                ]

            async def joined(read: dict[str, Any] = read) -> list[Any]:
                return [
                    await read_nearest(
                        conn, "child", by={"owner": owner}, limit=_LIMIT, columns=columns, joins=joins, **read
                    )
                    for _ in range(_READS)
                ]

            sent: list[str] = []

            def log(record: Any, sent: list[str] = sent) -> None:
                sent.append(record.query)

            conn.add_query_logger(log)
            got = (await joined())[0]
            await asyncio.sleep(0)  # the connection hands its last statement to the logger on the next loop turn
            conn.remove_query_logger(log)
            assert len(sent) == _READS, f"{name}: {_READS} joined reads sent {len(sent)} statements"
            want = (await bare())[0]
            assert len(got) == len(want) == _LIMIT
            assert all(row["title"] is not None for row in got)
            if name == "vector":
                # random distances: no ties, so the order is the ranked read's own (text ranks tie by design)
                assert [row["id"] for row in got] == [row["id"] for row in want]

            baseline, took = await best_seconds_interleaved(bare, joined)
            problem = budget_problem(f"{_READS} joined {name} reads", took, baseline)
            assert problem is None, problem
    finally:
        await conn.close()
