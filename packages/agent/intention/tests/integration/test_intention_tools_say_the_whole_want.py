"""integration: what the want tools say, against real pgvector.

- a near-duplicate log answers with the want as it is stored, and keeps its embedding;
- a listing honours its limit up to 100, says "N of M", lists one status or all, and
  carries each want's status, when it was written down and the consumer's note;
- ``intention_get`` reads one want in full, and not another user's;
- ``intention_mark_surfaced`` says the status it moved from, and changes nothing when
  the want already has the status asked for.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import asyncpg
import pytest

from threetears.agent.intention.tools import (
    load_intention_get_tool,
    load_intention_list_tool,
    load_intention_log_tool,
    load_intention_mark_surfaced_tool,
)

from .conftest import InMemoryNatsBus, apply_migrations, build_collection, make_pool, owner_only_authorizer

pytestmark = pytest.mark.integration

_DIM = 1024

#: When every inserted want is said to have been written down.
_WRITTEN = datetime(2026, 10, 9, 22, 0, tzinfo=UTC)


class _NearEmbedder:
    """One axis per text, except the texts named ``near``: 0.97 the same as the text they name."""

    def __init__(self, near: Mapping[str, str] | None = None) -> None:
        self._axes: dict[str, int] = {}
        self._near = dict(near or {})

    async def aembed_query(self, text: str) -> list[float]:
        vec = [0.0] * _DIM
        if text in self._near:
            vec[self._axes.setdefault(self._near[text], len(self._axes))] = 0.97
            vec[self._axes.setdefault(text, len(self._axes))] = math.sqrt(1 - 0.97**2)
            return vec
        vec[self._axes.setdefault(text, len(self._axes))] = 1.0
        return vec


async def _insert(
    pool: asyncpg.Pool,
    agent_id: uuid.UUID,
    customer_id: uuid.UUID,
    user_id: uuid.UUID,
    content: str,
    *,
    status: str = "open",
    salience: float = 0.5,
    last_surfaced: datetime | None = None,
) -> uuid.UUID:
    intention_id = uuid.uuid4()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO intentions (intention_id, agent_id, customer_id, user_id, status, "
            "content, salience, last_surfaced_at, date_created, date_updated) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$9)",
            intention_id,
            agent_id,
            customer_id,
            user_id,
            status,
            content,
            salience,
            last_surfaced,
            _WRITTEN,
        )
    return intention_id


async def _setup(pg_schema: tuple[str, str]) -> tuple[asyncpg.Pool, Any]:
    url, schema = pg_schema
    await apply_migrations(url, schema)
    pool = await make_pool(url, schema)
    coll, _l1 = build_collection(pool, InMemoryNatsBus())
    return pool, coll


class TestANearDuplicateKeepsTheWantAsStored:
    async def test_the_answer_is_the_stored_text_and_nothing_is_rewritten(self, pg_schema: tuple[str, str]) -> None:
        pool, coll = await _setup(pg_schema)
        try:
            agent_id, customer_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            embedder = _NearEmbedder(near={"ask about their trip": "ask about the trip to Lisbon"})
            (log_tool,) = await load_intention_log_tool(
                user_id, embedder, agent_id, customer_id, owner_only_authorizer(), coll
            )
            first = await log_tool.ainvoke({"content": "ask about the trip to Lisbon"})
            want_id = first.split("[intention:")[1].split("]")[0]
            async with pool.acquire() as conn:
                stored_before = await conn.fetchval(
                    "SELECT embedding::text FROM intentions WHERE intention_id = $1", uuid.UUID(want_id)
                )
            second = await log_tool.ainvoke({"content": "ask about their trip"})
            assert second == (
                f"Refreshed existing want [intention:{want_id}] (it was 97% the same), so nothing new was "
                "written down. It is now stronger, and it stays as first written: ask about the trip to Lisbon"
            )
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT content, embedding::text AS embedding FROM intentions WHERE intention_id = $1",
                    uuid.UUID(want_id),
                )
                count = await conn.fetchval("SELECT count(*) FROM intentions WHERE user_id = $1", user_id)
            assert row["content"] == "ask about the trip to Lisbon"
            assert row["embedding"] == stored_before
            assert count == 1
        finally:
            await pool.close()


class TestAListingSaysHowManyOfHowMany:
    async def test_a_limit_past_twenty_is_honoured(self, pg_schema: tuple[str, str]) -> None:
        pool, coll = await _setup(pg_schema)
        try:
            agent_id, customer_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            for n in range(25):
                await _insert(pool, agent_id, customer_id, user_id, f"want {n:02d}", salience=0.9 - n * 0.01)
            (list_tool,) = await load_intention_list_tool(user_id, agent_id, customer_id, owner_only_authorizer(), coll)
            out = await list_tool.ainvoke({"limit": 25})
            lines = out.splitlines()
            assert lines[0] == "25 of 25 open wants, strongest first."
            assert len(lines) == 26 and lines[-1].endswith(": want 24")
        finally:
            await pool.close()

    async def test_the_default_shows_twenty_and_says_how_many_more_there_are(self, pg_schema: tuple[str, str]) -> None:
        pool, coll = await _setup(pg_schema)
        try:
            agent_id, customer_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            for n in range(25):
                await _insert(pool, agent_id, customer_id, user_id, f"want {n:02d}", salience=0.9 - n * 0.01)
            (list_tool,) = await load_intention_list_tool(user_id, agent_id, customer_id, owner_only_authorizer(), coll)
            out = await list_tool.ainvoke({})
            assert out.splitlines()[0] == (
                "20 of 25 open wants, strongest first. Call again with a higher limit (up to 100) to see more."
            )
            assert len(out.splitlines()) == 21
        finally:
            await pool.close()

    async def test_each_line_has_its_status_when_it_was_written_and_the_consumers_note(
        self, pg_schema: tuple[str, str]
    ) -> None:
        pool, coll = await _setup(pg_schema)
        try:
            agent_id, customer_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            carried = await _insert(pool, agent_id, customer_id, user_id, "fix the flaky test", salience=0.9)
            plain = await _insert(pool, agent_id, customer_id, user_id, "learn the bass line", salience=0.4)

            async def _notes(wants: Sequence[Any]) -> Mapping[uuid.UUID, str]:
                return {w.intention_id: "project [project:p1], open" for w in wants if w.intention_id == carried}

            (list_tool,) = await load_intention_list_tool(
                user_id, agent_id, customer_id, owner_only_authorizer(), coll, tz="America/Los_Angeles", notes=_notes
            )
            out = await list_tool.ainvoke({})
            assert out == (
                "2 of 2 open wants, strongest first.\n"
                f"- [intention:{carried}] open, written Oct 9, 2026 at 3:00 PM PDT: fix the flaky test "
                "(project [project:p1], open)\n"
                f"- [intention:{plain}] open, written Oct 9, 2026 at 3:00 PM PDT: learn the bass line"
            )
        finally:
            await pool.close()

    async def test_a_status_lists_only_that_status_and_all_lists_every_want(self, pg_schema: tuple[str, str]) -> None:
        pool, coll = await _setup(pg_schema)
        try:
            agent_id, customer_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            await _insert(pool, agent_id, customer_id, user_id, "still open", salience=0.9)
            await _insert(pool, agent_id, customer_id, user_id, "done already", status="granted", salience=0.8)
            await _insert(pool, agent_id, customer_id, user_id, "let it go", status="dropped", salience=0.7)
            (list_tool,) = await load_intention_list_tool(user_id, agent_id, customer_id, owner_only_authorizer(), coll)
            granted = await list_tool.ainvoke({"status": "granted"})
            assert granted.splitlines()[0] == "1 of 1 granted wants, strongest first."
            assert "done already" in granted and "still open" not in granted and "let it go" not in granted
            every = await list_tool.ainvoke({"status": "all"})
            assert every.splitlines()[0] == "3 of 3 wants, strongest first."
            assert "] granted, written" in every and "] dropped, written" in every
            assert (await list_tool.ainvoke({"status": "asked"})) == "You have no asked wants."
            assert (await list_tool.ainvoke({"status": "done"})).startswith(
                "[TOOL ERROR] intention_list: input failed — invalid status 'done'"
            )
        finally:
            await pool.close()

    async def test_open_wants_held_back_are_counted(self, pg_schema: tuple[str, str]) -> None:
        pool, coll = await _setup(pg_schema)
        try:
            agent_id, customer_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            await _insert(pool, agent_id, customer_id, user_id, "raise me", salience=0.9)
            await _insert(pool, agent_id, customer_id, user_id, "raised", last_surfaced=datetime.now(UTC))
            (list_tool,) = await load_intention_list_tool(
                user_id, agent_id, customer_id, owner_only_authorizer(), coll, cooldown_days=7
            )
            out = await list_tool.ainvoke({})
            assert out.splitlines()[0] == (
                "1 of 1 open wants, strongest first. 1 more open want is held back because it was raised "
                "in the last 7 days."
            )
            assert "raised\n" not in out and not out.endswith(": raised")
        finally:
            await pool.close()

    async def test_a_note_that_fails_leaves_the_listing(self, pg_schema: tuple[str, str]) -> None:
        pool, coll = await _setup(pg_schema)
        try:
            agent_id, customer_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            await _insert(pool, agent_id, customer_id, user_id, "still listed")

            async def _broken(wants: Sequence[Any]) -> Mapping[uuid.UUID, str]:
                raise RuntimeError("projects are down")

            (list_tool,) = await load_intention_list_tool(
                user_id, agent_id, customer_id, owner_only_authorizer(), coll, notes=_broken
            )
            out = await list_tool.ainvoke({})
            assert out.startswith("1 of 1 open wants") and out.endswith(": still listed")
        finally:
            await pool.close()


class TestOneWantInFull:
    async def test_get_reads_a_want_whole(self, pg_schema: tuple[str, str]) -> None:
        pool, coll = await _setup(pg_schema)
        try:
            agent_id, customer_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            want_id = await _insert(pool, agent_id, customer_id, user_id, "learn the bass line", salience=0.62)

            async def _notes(wants: Sequence[Any]) -> Mapping[uuid.UUID, str]:
                return {wants[0].intention_id: "No project carries it."}

            (get_tool,) = await load_intention_get_tool(
                user_id, agent_id, customer_id, owner_only_authorizer(), coll, notes=_notes
            )
            out = await get_tool.ainvoke({"intention_id": f"[intention:{want_id}]"})
            assert out == (
                f"[intention:{want_id}] open\n"
                "Written down: Oct 9, 2026 at 10:00 PM UTC\n"
                "Strength: 0.62\n"
                "No project carries it.\n"
                "Want: learn the bass line"
            )
        finally:
            await pool.close()

    async def test_another_users_want_is_not_found(self, pg_schema: tuple[str, str]) -> None:
        pool, coll = await _setup(pg_schema)
        try:
            agent_id, customer_id = uuid.uuid4(), uuid.uuid4()
            owner, other = uuid.uuid4(), uuid.uuid4()
            want_id = await _insert(pool, agent_id, customer_id, owner, "private")
            (get_tool,) = await load_intention_get_tool(other, agent_id, customer_id, owner_only_authorizer(), coll)
            out = await get_tool.ainvoke({"intention_id": str(want_id)})
            assert out == f"[TOOL ERROR] No want found for [intention:{want_id}]."
        finally:
            await pool.close()


class TestAMarkSaysWhatItWas:
    async def test_a_mark_says_the_status_it_moved_from(self, pg_schema: tuple[str, str]) -> None:
        pool, coll = await _setup(pg_schema)
        try:
            agent_id, customer_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            want_id = await _insert(pool, agent_id, customer_id, user_id, "ship the fix", status="asked")
            (mark_tool,) = await load_intention_mark_surfaced_tool(
                user_id, agent_id, customer_id, owner_only_authorizer(), coll
            )
            out = await mark_tool.ainvoke({"intention_id": str(want_id), "new_status": "granted"})
            assert out == f"Marked [intention:{want_id}] granted; it was asked. The want: ship the fix"
            async with pool.acquire() as conn:
                assert await conn.fetchval("SELECT status FROM intentions WHERE intention_id = $1", want_id) == (
                    "granted"
                )
        finally:
            await pool.close()

    async def test_a_mark_to_the_status_it_has_changes_nothing(self, pg_schema: tuple[str, str]) -> None:
        pool, coll = await _setup(pg_schema)
        try:
            agent_id, customer_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            want_id = await _insert(pool, agent_id, customer_id, user_id, "ship the fix", status="dropped")
            (mark_tool,) = await load_intention_mark_surfaced_tool(
                user_id, agent_id, customer_id, owner_only_authorizer(), coll
            )
            out = await mark_tool.ainvoke({"intention_id": str(want_id), "new_status": "dropped"})
            assert out == f"[NO CHANGE] intention_mark_surfaced: [intention:{want_id}] is already dropped: ship the fix"
            async with pool.acquire() as conn:
                surfaced = await conn.fetchval(
                    "SELECT last_surfaced_at FROM intentions WHERE intention_id = $1", want_id
                )
            assert surfaced is None
        finally:
            await pool.close()
