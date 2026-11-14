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

from threetears.core.keyed import KeyBatch, read_keyed
from threetears.core.testing.same_results import assert_same_results

from .memory_same_results import AGENTS, FILES, MEMORIES, PAGE, chunk_row, conn, content_row, parents

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
                keys=KeyBatch(("memory_id",), [(memory,) for memory in named], casts=("uuid",)),
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
                keys=KeyBatch(("media_id",), [(media,) for media in named], casts=("uuid",)),
                by={"agent_id": agent},
                columns=["agent_id", "content_id"],
            ),
            page_size=PAGE,
        )
