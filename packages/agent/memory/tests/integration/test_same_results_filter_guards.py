"""the memory reads moved onto the keyed helpers, with the host's read filter as their ``where=`` guards, give the rows
the hand-written statements gave.

Each method below once wrote its own statement and spliced the read filter's clause into it; it now hands its pins to
``read_row`` / ``read_by`` / ``aggregate_keyed`` and the filter to ``where=`` (``filter_guards``,
``_chunk_parent_guards``, ``_media_parent_guards``). Each is compared, through
:func:`~threetears.core.testing.same_results.assert_same_results`, with the statement it replaced: empty, one row,
many, NULLs, ties in the sort, a tenant boundary, a page boundary and one past it. What a filter leaves readable is
held by ``test_read_filter`` for every one of these methods.

Runs on PostgreSQL and YugabyteDB (the session's ``db_container``), a schema of its own.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import timedelta
from typing import Any

import asyncpg
import pytest

from threetears.agent.memory.collections import MediaContentCollection, MemoriesCollection, MemoryChunkCollection
from threetears.agent.memory.read_filter import ReadFilter, filter_clause
from threetears.core.testing.same_results import Seeded, assert_same_results

from .memory_same_results import (
    text_vectors,
    AGENTS,
    CUSTOMER,
    MEMORIES,
    PAGE,
    T0,
    USER,
    chunk_row,
    collection,
    content_row,
    ident,
    memory_row,
    parents,
    conn,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]

__all__ = ["conn"]

#: what the memory listings compare: the columns every seed writes, the embedding left to the entity's codec
_LISTED = ("memory_id", "agent_id", "customer_id", "user_id", "type_memory", "content", "date_created")


def _listed(entities: Sequence[Any]) -> list[dict[str, Any]]:
    return [{column: entity.to_dict()[column] for column in _LISTED} for entity in entities]


async def _same_memories(
    conn: asyncpg.Connection, old_sql: str, params: Sequence[Any], helper: Callable[[Any], Awaitable[Any]]
) -> None:
    await text_vectors(conn)
    await assert_same_results(
        conn,
        table="memories",
        row=memory_row,
        old_sql=old_sql,
        params=params,
        helper=helper,
        columns=_LISTED,
        order_by=["date_created"],
        page_size=PAGE,
    )


async def test_find_by_user_gives_the_hand_written_rows(conn: asyncpg.Connection) -> None:
    """Flips if find_by_user's read_by returns a memory the owner-triple statement did not, misses one, or orders
    them differently."""
    agent = AGENTS[0]

    async def helper(c: Any) -> list[dict[str, Any]]:
        memories = collection(MemoriesCollection, c)
        return _listed(await memories.find_by_user(USER, agent_id=agent, customer_id=CUSTOMER))

    await _same_memories(
        conn,
        f"SELECT {', '.join(_LISTED)} FROM memories WHERE agent_id = $1 AND customer_id = $2 AND user_id = $3"
        " ORDER BY date_created DESC",
        [agent, CUSTOMER, USER],
        helper,
    )


@pytest.mark.parametrize("with_customer", [False, True])
async def test_find_by_scope_gives_the_hand_written_rows(conn: asyncpg.Connection, with_customer: bool) -> None:
    """Flips if find_by_scope's read_by, in either owner form, returns a memory its statement did not, misses one,
    or orders them differently."""
    agent = AGENTS[0]
    customer = CUSTOMER if with_customer else None

    async def helper(c: Any) -> list[dict[str, Any]]:
        memories = collection(MemoriesCollection, c)
        return _listed(await memories.find_by_scope(agent_id=agent, customer_id=customer, user_id=USER))

    if with_customer:
        sql = "WHERE agent_id = $1 AND customer_id = $2 AND user_id = $3"
        params: list[Any] = [agent, CUSTOMER, USER]
    else:
        sql = "WHERE agent_id = $1 AND user_id = $2"
        params = [agent, USER]
    await _same_memories(
        conn, f"SELECT {', '.join(_LISTED)} FROM memories {sql} ORDER BY date_created DESC", params, helper
    )


@pytest.mark.parametrize("with_customer", [False, True])
async def test_find_for_user_in_agents_gives_the_hand_written_rows(
    conn: asyncpg.Connection, with_customer: bool
) -> None:
    """Flips if find_for_user_in_agents' read_by over the agent set, in either owner form, returns a memory its
    statement did not, misses one, or orders them differently."""
    customer = CUSTOMER if with_customer else None

    async def helper(c: Any) -> list[dict[str, Any]]:
        memories = collection(MemoriesCollection, c)
        return _listed(await memories.find_for_user_in_agents(user_id=USER, agent_ids=AGENTS, customer_id=customer))

    if with_customer:
        sql = "WHERE agent_id = ANY($1::uuid[]) AND customer_id = $2 AND user_id = $3"
        params: list[Any] = [list(AGENTS), CUSTOMER, USER]
    else:
        sql = "WHERE agent_id = ANY($1::uuid[]) AND user_id = $2"
        params = [list(AGENTS), USER]
    await _same_memories(
        conn, f"SELECT {', '.join(_LISTED)} FROM memories {sql} ORDER BY date_created DESC", params, helper
    )


def _rows(value: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    return [] if value is None else [value]


async def test_memory_content_for_recall_gives_the_hand_written_row(conn: asyncpg.Connection) -> None:
    """Flips if fetch_content_for_recall's read_row answers another memory's content, or one the owner statement
    did not find."""
    agent, memory = AGENTS[0], ident("memory", 0)

    async def helper(c: Any) -> list[Mapping[str, Any]]:
        content = await collection(MemoriesCollection, c).fetch_content_for_recall(
            memory_id=memory, user_id=USER, agent_id=agent
        )
        return _rows(None if content is None else {"content": content})

    await text_vectors(conn)
    await assert_same_results(
        conn,
        table="memories",
        row=memory_row,
        old_sql="SELECT content FROM memories WHERE agent_id = $1 AND memory_id = $2 AND user_id = $3",
        params=[agent, memory, USER],
        helper=helper,
        page_size=PAGE,
    )


async def test_media_content_for_recall_gives_the_hand_written_row(conn: asyncpg.Connection) -> None:
    """Flips if the media leg's read_row answers another row's content, or one the owner statement did not find (the
    every-fourth content row of another user included)."""
    agent = AGENTS[0]

    async def helper(c: Any) -> list[Mapping[str, Any]]:
        found = []
        for index in (0, 3):
            content = await collection(MediaContentCollection, c).fetch_content_for_recall(
                content_id=ident("content", index), user_id=USER, agent_id=agent
            )
            found.extend(_rows(None if content is None else {"content": content}))
        return found

    async with parents(conn):
        await assert_same_results(
            conn,
            table="media_content",
            row=content_row,
            old_sql="SELECT content FROM media_content WHERE agent_id = $1 AND content_id = ANY($2::uuid[])"
            " AND user_id = $3 ORDER BY content",
            params=[agent, [ident("content", 0), ident("content", 3)], USER],
            helper=helper,
            ordered=False,
            page_size=PAGE,
        )


async def test_chunk_content_for_recall_gives_the_hand_written_row(conn: asyncpg.Connection) -> None:
    """Flips if the passage leg's read_row answers another passage, or one the owner statement did not find (the
    every-fifth passage of another user included)."""
    agent = AGENTS[0]

    async def helper(c: Any) -> list[Mapping[str, Any]]:
        found = []
        for index in (0, 4):
            got = await collection(MemoryChunkCollection, c).fetch_content_for_recall(
                chunk_id=ident("chunk", index), user_id=USER, agent_id=agent
            )
            found.extend(_rows(None if got is None else {"content": got[0], "memory_id": got[1]}))
        return found

    async with parents(conn):
        await assert_same_results(
            conn,
            table="memory_chunks",
            row=chunk_row,
            old_sql="SELECT content, memory_id FROM memory_chunks WHERE agent_id = $1 AND chunk_id = ANY($2::uuid[])"
            " AND user_id = $3",
            params=[agent, [ident("chunk", 0), ident("chunk", 4)], USER],
            helper=helper,
            ordered=False,
            page_size=PAGE,
        )


async def test_count_by_memory_id_gives_the_hand_written_count(conn: asyncpg.Connection) -> None:
    """Flips if count_by_memory_id's aggregate_keyed counts a passage the owner statement did not, or misses one."""
    agent = AGENTS[0]
    memory = MEMORIES[(agent, "mine")][0]

    async def helper(c: Any) -> list[Mapping[str, Any]]:
        counted = await collection(MemoryChunkCollection, c).count_by_memory_id(
            memory, user_id=USER, agent_id=agent, customer_id=CUSTOMER
        )
        return [{"n": counted}]

    async with parents(conn):
        await assert_same_results(
            conn,
            table="memory_chunks",
            row=chunk_row,
            old_sql="SELECT count(*) AS n FROM memory_chunks mc WHERE mc.agent_id = $1 AND mc.memory_id = $2"
            " AND mc.user_id = $3 AND mc.customer_id = $4",
            params=[agent, memory, USER, CUSTOMER],
            helper=helper,
            page_size=PAGE,
        )


async def test_find_by_chunk_indexes_gives_the_hand_written_rows(conn: asyncpg.Connection) -> None:
    """Flips if find_by_chunk_indexes' read_by returns a passage its statement did not, misses one, or orders them
    differently."""
    agent = AGENTS[0]
    memory = MEMORIES[(agent, "mine")][0]
    wanted = [0, 3, 6, 9, 12, 30]

    async def helper(c: Any) -> list[dict[str, Any]]:
        chunks: MemoryChunkCollection = collection(MemoryChunkCollection, c)
        return await chunks.find_by_chunk_indexes(
            memory, user_id=USER, agent_id=agent, customer_id=CUSTOMER, chunk_indexes=wanted
        )

    async with parents(conn):
        await assert_same_results(
            conn,
            table="memory_chunks",
            row=chunk_row,
            old_sql="""
                SELECT mc.chunk_id, mc.chunk_index, mc.content, mc.summary,
                       mc.message_id_start, mc.message_id_end,
                       mc.token_count, mc.date_created
                FROM memory_chunks mc
                WHERE mc.agent_id = $1 AND mc.memory_id = $2 AND mc.user_id = $3 AND mc.customer_id = $4
                  AND mc.chunk_index = ANY($5)
                ORDER BY mc.chunk_index ASC
            """,
            params=[agent, memory, USER, CUSTOMER, wanted],
            helper=helper,
            order_by=["chunk_index"],
            page_size=PAGE,
        )


#: the four scope grains a consolidation run pins exactly (None: IS NULL), each seeded on every fourth memory
_GRAINS = ((None, None), (None, USER), (CUSTOMER, None), (CUSTOMER, USER))
#: the columns a consolidation candidate is compared on (its embedding decoded by the method, so compared by which
#: rows carry one, the NULL ones never being candidates)
_CANDIDATE = (
    "memory_id",
    "customer_id",
    "user_id",
    "conversation_id",
    "type_memory",
    "content",
    "date_created",
    "tags",
)
#: the statement find_active_for_consolidation sent before read_by, one per grain, frozen
_OLD_CANDIDATE_COLUMNS = (
    "m.memory_id, m.customer_id, m.user_id, m.conversation_id, m.type_memory, m.content, "
    "m.embedding::text AS embedding, m.date_created, m.tags"
)
_OLD_CANDIDATE_TAIL = (
    "AND NOT m.evergreen AND m.embedding IS NOT NULL "
    "AND (m.superseded_by IS NULL OR NOT EXISTS ("
    "SELECT 1 FROM memories g WHERE g.agent_id = m.agent_id AND g.memory_id = m.superseded_by)) "
    "AND NOT (COALESCE(m.tags, '[]'::jsonb) ? 'retracted') ORDER BY m.date_created"
)


def _hide(tags: str, bind: Callable[[Any], str]) -> str:
    return f"NOT (COALESCE({tags}, '[]'::jsonb) ? {bind('hide')})"


def _old_candidates(customer: Any, user: Any, read_filter: ReadFilter | None) -> tuple[str, list[Any]]:
    """the hand-written statement of a grain, as the method built it (the filter's clause where it spliced it)."""
    params: list[Any] = [AGENTS[0]]
    head = f"SELECT {_OLD_CANDIDATE_COLUMNS} FROM memories m WHERE m.agent_id = $1"
    if customer is None and user is None:
        allowed = filter_clause(read_filter, "m.tags", params)
        sql = f"{head}{allowed} AND m.customer_id IS NULL AND m.user_id IS NULL {_OLD_CANDIDATE_TAIL}"
    elif customer is None:
        params.append(user)
        allowed = filter_clause(read_filter, "m.tags", params)
        sql = f"{head} AND m.customer_id IS NULL AND m.user_id = $2{allowed} {_OLD_CANDIDATE_TAIL}"
    elif user is None:
        params.append(customer)
        allowed = filter_clause(read_filter, "m.tags", params)
        sql = f"{head} AND m.customer_id = $2{allowed} AND m.user_id IS NULL {_OLD_CANDIDATE_TAIL}"
    else:
        params.extend((customer, user))
        allowed = filter_clause(read_filter, "m.tags", params)
        sql = f"{head} AND m.customer_id = $2 AND m.user_id = $3{allowed} {_OLD_CANDIDATE_TAIL}"
    return sql, params


def _candidate_row(seed: Seeded) -> Mapping[str, Any]:
    """a memory of its tenant's agent in one of the four grains (by its index), its time its tie. Each guard is set
    by the row's place within its grain (``index // 4``), so every grain holds every kind: the second of each five
    superseded by the memory before it (a live gist while that one is seeded), the fourth of each five by a gist that
    does not exist (so the NOT EXISTS keeps it), the sixth of each six pinned (evergreen), the first of each three
    tagged ``hide`` (the filter's) and the fifth of each seven retracted."""
    customer, user = _GRAINS[seed.index % 4]
    place = seed.index // 4
    superseded = (
        ident("memory", seed.index - 1)
        if place % 5 == 1
        else ident("dead-gist", seed.index)
        if place % 5 == 3
        else None
    )
    tags = ["retracted"] if place % 7 == 4 else ["hide"] if place % 3 == 0 else None
    return {
        **memory_row(seed),
        "customer_id": customer,
        "user_id": user,
        "date_created": T0 + timedelta(minutes=seed.tie),
        "evergreen": place % 6 == 5,
        "superseded_by": superseded,
        "tags": None if tags is None else json.dumps(tags),
    }


@pytest.mark.parametrize("read_filter", [None, _hide], ids=["unfiltered", "filtered"])
@pytest.mark.parametrize("grain", range(len(_GRAINS)), ids=["agent", "user-only", "customer", "user"])
async def test_find_active_for_consolidation_gives_the_hand_written_rows(
    conn: asyncpg.Connection, grain: int, read_filter: ReadFilter | None
) -> None:
    """Flips if find_active_for_consolidation's read_by, in any of the four grains (a None one pinned IS NULL), with
    the host's filter or without, returns a candidate its grain's statement did not, misses one, or orders them
    differently: the pins, the evergreen, embedding and retracted guards, and the live-gist NOT EXISTS (true for
    some rows of every grain, false for others)."""
    customer, user = _GRAINS[grain]
    old_sql, params = _old_candidates(customer, user, read_filter)

    async def helper(c: Any) -> list[dict[str, Any]]:
        memories = collection(MemoriesCollection, c)
        memories.set_read_filter(read_filter)
        found = await memories.find_active_for_consolidation(AGENTS[0], customer_id=customer, user_id=user)
        return [{column: candidate[column] for column in _CANDIDATE} for candidate in found]

    await text_vectors(conn)
    answered = await assert_same_results(
        conn,
        table="memories",
        row=_candidate_row,
        old_sql=old_sql,
        params=params,
        helper=helper,
        columns=_CANDIDATE,
        order_by=["date_created"],
        page_size=PAGE,
    )
    # the seeds reach every guard: some case keeps rows, and the many-rows case drops some its grain and agent hold
    assert any(answered), "no case kept a candidate"
