"""a user merge's alias collision reads the children its delete cascades to by key, and finds the rows the tuple-set
reads found.

``merge._delete_alias_collisions`` once read each child table by ``(agent_id, <parent>) IN (SELECT * FROM unnest(...))``;
it now reads them through :func:`~threetears.core.keyed.read_keyed`, the agent pinned and the parents named, through the
index each cascade follows (``test_index_plans``). Each read is compared, through
:func:`~threetears.core.testing.same_results.assert_same_results`, with the statement it replaces: empty, one row,
many, NULLs, ties, a tenant boundary (the other agent's rows under the same parent ids) and a page boundary.

The merge's whole answer -- every row the cascade removed is named, nothing else -- is
``test_merge_names_every_cascaded_row``. Runs on PostgreSQL and YugabyteDB, a schema of its own.
"""

from __future__ import annotations

import asyncpg
import pytest

import uuid
from collections.abc import Mapping
from typing import Any

from threetears.agent.memory.merge import delete_colliding_memories, lock_colliding_memories
from threetears.core.keyed import KeyBatch, read_keyed
from threetears.core.testing.same_results import Seeded, assert_same_results

from .memory_same_results import (
    AGENTS,
    CONVERSATION,
    CUSTOMER,
    FILES,
    MEMORIES,
    NS,
    OTHER_USER,
    PAGE,
    T0,
    USER,
    chunk_row,
    conn,
    content_row,
    ident,
    parents,
)

__all__ = ["conn"]

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]

#: the colliding memories' chunks, as the merge read them before
_CHUNKS_SQL = (
    "SELECT agent_id, chunk_id FROM memory_chunks"
    " WHERE (agent_id, memory_id) IN (SELECT * FROM unnest($1::uuid[], $2::uuid[]))"
)

#: the colliding memories' files' contents, as the merge read them before
_CONTENTS_SQL = (
    "SELECT agent_id, content_id FROM media_content"
    " WHERE (agent_id, media_id) IN (SELECT * FROM unnest($1::uuid[], $2::uuid[]))"
)


async def test_the_collision_s_chunks_are_the_tuple_set_s(conn: asyncpg.Connection) -> None:
    """Flips if the chunks a collision names differ from the tuple-set read's: two of the agent's memories named, the
    other agent's chunks under the same memory ids never."""
    agent = AGENTS[0]
    named = [MEMORIES[(agent, "mine")][0], MEMORIES[(agent, "elsewhere")][0]]
    async with parents(conn):
        await assert_same_results(
            conn,
            table="memory_chunks",
            row=chunk_row,
            old_sql=_CHUNKS_SQL,
            params=[[agent] * len(named), named],
            helper=lambda c: read_keyed(
                c,
                "memory_chunks",
                keys=KeyBatch(("memory_id",), [[memory for memory in named]], casts=("uuid",)),
                by={"agent_id": agent},
                columns=["agent_id", "chunk_id"],
            ),
            page_size=PAGE,
        )


async def test_the_collision_s_contents_are_the_tuple_set_s(conn: asyncpg.Connection) -> None:
    """Flips if the contents a collision names by their files differ from the tuple-set read's: both the agent's files
    named, the other agent's contents never."""
    agent = AGENTS[0]
    named = [FILES[(agent, "mine")][0], FILES[(agent, "theirs")][0]]
    async with parents(conn):
        await assert_same_results(
            conn,
            table="media_content",
            row=content_row,
            old_sql=_CONTENTS_SQL,
            params=[[agent] * len(named), named],
            helper=lambda c: read_keyed(
                c,
                "media_content",
                keys=KeyBatch(("media_id",), [[media for media in named]], casts=("uuid",)),
                by={"agent_id": agent},
                columns=["agent_id", "content_id"],
            ),
            page_size=PAGE,
        )


#: the master a user merge moves the source's (``USER``'s) memories to
_MASTER = uuid.uuid5(NS, "master")

#: the colliding source memories, as the merge locked them before (``$1`` source, ``$2`` master, ``$3`` agent)
_COLLISION_PREDICATE = (
    "m.agent_id = $3 AND m.user_id = $1 AND m.alias IS NOT NULL "
    "AND EXISTS (SELECT 1 FROM memories m2 WHERE m2.user_id = $2 AND m2.agent_id = m.agent_id AND m2.alias = m.alias)"
)
_LOCK_SQL = f"SELECT m.agent_id, m.memory_id FROM memories m WHERE {_COLLISION_PREDICATE} FOR UPDATE OF m"

#: the delete of the locked ones, as the merge sent it before (``$4`` / ``$5`` the locked keys)
_DELETE_SQL = (
    f"DELETE FROM memories m WHERE {_COLLISION_PREDICATE} "
    "AND (m.agent_id, m.memory_id) IN (SELECT * FROM unnest($4::uuid[], $5::uuid[])) RETURNING agent_id, memory_id"
)


def _named_memory(seed: Seeded) -> Mapping[str, Any]:
    """a named memory of its tenant's agent: the source user's, every fifth another user's; NULL-named in the NULLs
    case (no alias collides)."""
    return {
        "memory_id": ident("collide", seed.index),
        "agent_id": AGENTS[seed.tenant],
        "customer_id": CUSTOMER,
        "user_id": OTHER_USER if seed.index % 5 == 4 else USER,
        "conversation_id": CONVERSATION,
        "type_memory": "fact",
        "content": f"memory {seed.index}",
        "date_created": T0,
        "date_updated": T0,
        "alias": None if seed.null else f"alias-{seed.tie}-{seed.index}",
    }


async def _masters(conn: asyncpg.Connection, rows: list[Mapping[str, Any]]) -> None:
    """the master's memories, named as some seeded memories: under the FIRST agent as every other one, so the guard
    holds for some source memories and not others; under the OTHER agent as every third of the rest, so a name the
    master holds only under another agent never collides."""
    await conn.executemany(
        "INSERT INTO memories (memory_id, agent_id, customer_id, user_id, conversation_id, type_memory, content,"
        " date_created, date_updated, alias) VALUES ($1, $2, $3, $4, $5, 'fact', 'the master''s', $6, $6, $7)",
        [
            (
                ident("master", index),
                AGENTS[0 if index % 2 == 0 else 1],
                CUSTOMER,
                _MASTER,
                CONVERSATION,
                T0,
                row["alias"],
            )
            for index, row in enumerate(rows)
            if row["alias"] is not None and (index % 2 == 0 or index % 3 == 1)
        ],
    )


async def test_the_collision_lock_names_the_predicate_s_memories(conn: asyncpg.Connection) -> None:
    """Flips if the memories the collision step locks differ from the hand-written predicate's: the source's named
    memories the master names too under the same agent, never another user's, another agent's, or an unnamed one."""
    agent = AGENTS[0]
    await assert_same_results(
        conn,
        table="memories",
        row=_named_memory,
        old_sql=_LOCK_SQL,
        params=[USER, _MASTER, agent],
        helper=lambda c: lock_colliding_memories(c, from_user_id=USER, to_user_id=_MASTER, agent_id=agent),
        related=_masters,
        page_size=PAGE,
    )


async def test_the_collision_delete_removes_the_predicate_s_locked_memories(conn: asyncpg.Connection) -> None:
    """Flips if the collision delete removes (or returns) other memories than the hand-written delete: of the keys
    named (two of every three seeded, the other agent's same ids among them), only the source's colliding ones, the
    guard re-checked."""
    agent = AGENTS[0]
    named = [ident("collide", index) for index in range(40) if index % 3 != 2]
    await assert_same_results(
        conn,
        table="memories",
        row=_named_memory,
        old_sql=_DELETE_SQL,
        params=[USER, _MASTER, agent, [agent] * len(named), named],
        helper=lambda c: delete_colliding_memories(
            c, from_user_id=USER, to_user_id=_MASTER, agent_id=agent, memory_ids=named
        ),
        related=_masters,
        state_sql="SELECT memory_id, agent_id, user_id, alias FROM memories",
        page_size=PAGE,
    )
