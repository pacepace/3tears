"""integration: what ambient recall hands an agent, and one event stored once.

Against real pgvector tables, through the real collection, on memories AGED
the way a real history is: months of history beside a burst written in the
last hours. A fixture stamped at ``now`` holds recency at 1.0 for every row
and can never show a recent burst crowding the list.

- the recent cap: how many of the list were written in the last days, and
  that a recent memory that answers the message best still gets in;
- a memory written with nobody present stays out of ambient recall and stays
  searchable, and one whose origin nobody recorded stays in;
- being shown does not raise a memory's salience when the consumer says so,
  and salience never moves a memory's place in the list;
- the same event written twice from one turn or the next is one memory,
  whichever writer wrote first, and a different fact from that turn is not
  merged;
- extraction does not read a reply the host said is not a source;
- a dream never makes a memory permanent, and a gist made only of unattended
  memories is unattended.
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import asyncpg
import pytest

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import MemoriesCollection, MemoryConsolidationsCollection
from threetears.agent.memory.extraction import REPLY_NOT_A_SOURCE, MemoryExtractor
from threetears.agent.memory.migrations import register as register_memory
from threetears.agent.memory.origin import UNATTENDED_TAG, made_by_tag, origin_of
from threetears.agent.memory.retrieval import MemoryRetriever
from threetears.agent.memory.retrieval_scope import RetrievalScope
from threetears.agent.memory.revisions import SAME_EVENT_RATIONALE, write_once
from threetears.agent.memory.tools import (
    PERMANENT_NEEDS_THE_PERSON,
    load_memory_add_tool,
    load_memory_keep_tool,
    load_memory_recall_tool,
)
from threetears.agent.memory.types import MemoryConfig
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore
from .memory_support import DIM, make_pool, make_service

pytestmark = pytest.mark.integration

_WEIGHTS = {"semantic": 0.55, "keyword": 0.15, "recency": 0.30}
#: The query every test asks with: the first axis.
_QUERY = [1.0] + [0.0] * (DIM - 1)


def _at(cosine: float, axis: int) -> list[float]:
    """A unit vector whose cosine to :data:`_QUERY` is ``cosine``; each ``axis`` is its own direction."""
    vector = [0.0] * DIM
    vector[0] = cosine
    vector[axis] = math.sqrt(1.0 - cosine * cosine)
    return vector


@pytest.fixture
async def applied_schema(pg_schema: tuple[str, str]) -> tuple[str, str]:
    """apply conversations + memory migrations into the per-test schema."""
    url, schema = pg_schema
    runner = MigrationRunner()
    register_conversations(runner)
    register_memory(runner)
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(f'SET search_path TO "{schema}", public')
        await runner.apply_for_agent_schema(AsyncpgStore(conn))  # type: ignore[arg-type]
    finally:
        await conn.close()
    return url, schema


class _World:
    """One person's memories on a real database."""

    def __init__(self, pool: asyncpg.Pool, conn: asyncpg.Connection) -> None:
        self.pool, self.conn = pool, conn
        self.agent_id, self.user_id = uuid.uuid4(), uuid.uuid4()
        registry = CollectionRegistry()
        registry.configure(l3_pool=pool)
        config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
        self.memories = MemoriesCollection(
            registry=registry, config=config, authorizer=MagicMock(spec=MemoryAuthorizerDependencies)
        )
        self.edges = MemoryConsolidationsCollection(registry=registry, config=config, nats_client=None)

    async def remember(
        self,
        content: str,
        *,
        cosine: float,
        axis: int,
        age_hours: float = 0.0,
        tags: list[str] | None = None,
        permanent: bool = False,
        message_id: uuid.UUID | None = None,
        salience: float = 0.5,
    ) -> uuid.UUID:
        memory_id = uuid.uuid4()
        created = datetime.now(UTC) - timedelta(hours=age_hours)
        await self.conn.execute(
            "INSERT INTO memories (memory_id, agent_id, customer_id, user_id, conversation_id, message_id_source, "
            "type_memory, content, embedding, salience, evergreen, tags, date_created, date_updated) "
            "VALUES ($1,$2,$3,$3,$4,$5,'fact',$6,$7::text::public.vector,$8,$9,$10::jsonb,$11,$11)",
            memory_id,
            self.agent_id,
            self.user_id,
            uuid.uuid4(),
            message_id,
            content,
            json.dumps(_at(cosine, axis)),
            salience,
            permanent,
            json.dumps(tags) if tags is not None else None,
            created,
        )
        return memory_id

    async def ambient(self, **more: Any) -> list[dict[str, Any]]:
        """The ambient list for the query, as the retriever asks the collection for it."""
        return await self.memories.hybrid_search(
            user_id=self.user_id,
            agent_id=self.agent_id,
            customer_id=self.user_id,
            embedding=_QUERY,
            user_text="zz",  # under the FTS minimum: the semantic arm alone
            top_k=10,
            candidate_limit=30,
            similarity_threshold=0.4,
            recency_half_life_hours=24.0,
            signal_weights=_WEIGHTS,
            salience_ambient_floor=0.2,
            **more,
        )

    async def row(self, memory_id: uuid.UUID) -> dict[str, Any]:
        row = await self.conn.fetchrow(
            "SELECT content, salience, evergreen, superseded_by, tags, message_id_source FROM memories "
            "WHERE memory_id = $1",
            memory_id,
        )
        assert row is not None, "a memory was deleted"
        return {**dict(row), "salience": float(row["salience"])}

    async def live(self) -> list[str]:
        """The text of every memory nothing has replaced, oldest first."""
        rows = await self.conn.fetch(
            "SELECT content FROM memories WHERE agent_id = $1 AND superseded_by IS NULL ORDER BY date_created",
            self.agent_id,
        )
        return [r["content"] for r in rows]

    def fields(self, content: str, *, cosine: float, axis: int, message_id: uuid.UUID, **more: Any) -> dict[str, Any]:
        return {
            "customer_id": self.user_id,
            "user_id": self.user_id,
            "conversation_id": uuid.uuid4(),
            "message_id_source": message_id,
            "type_memory": "fact",
            "content": content,
            "embedding": _at(cosine, axis),
            "salience": 0.5,
            **more,
        }


@pytest.fixture
async def world(applied_schema: tuple[str, str]) -> Any:
    url, schema = applied_schema
    pool, conn = await make_pool(url, schema), await asyncpg.connect(url)
    await conn.execute(f'SET search_path TO "{schema}", public')
    try:
        yield _World(pool, conn)
    finally:
        await conn.close()
        await pool.close()


async def _history_and_burst(world: _World) -> tuple[list[uuid.UUID], list[uuid.UUID], uuid.UUID]:
    """Twelve memories from two to seven months ago, eight written in the last hours, and one from five days ago.

    The old ones answer the query well (cosine 0.70 down to 0.59). The burst
    barely does (0.52 down to 0.45). The five-day-old one answers it best of
    all (0.95), and its recency term is already gone (``exp(-120/24)``).
    """
    old = [
        await world.remember(f"old {i}", cosine=0.70 - 0.01 * i, axis=10 + i, age_hours=24.0 * (60 + 13 * i))
        for i in range(12)
    ]
    burst = [
        await world.remember(f"burst {i}", cosine=0.52 - 0.01 * i, axis=40 + i, age_hours=1.0 + i) for i in range(8)
    ]
    best = await world.remember("the answer", cosine=0.95, axis=60, age_hours=120.0)
    return old, burst, best


def _recent(rows: list[dict[str, Any]], days: float = 7.0) -> list[dict[str, Any]]:
    cutoff = datetime.now(UTC) - timedelta(days=days)
    return [r for r in rows if r["date_created"] >= cutoff]


class TestTheRecentCap:
    async def test_without_a_cap_the_burst_fills_the_list(self, world: _World) -> None:
        """The crowding is the formula's: nine of ten places go to the last week."""
        await _history_and_burst(world)

        rows = await world.ambient()

        assert len(rows) == 10
        assert len(_recent(rows)) == 9

    async def test_a_cap_of_three_leaves_seven_places_to_the_history(self, world: _World) -> None:
        old, _burst, _best = await _history_and_burst(world)

        rows = await world.ambient(recent_cap=3)

        assert len(rows) == 10
        assert len(_recent(rows)) == 3
        # the history's places go to its seven best answers, and the list is still ranked
        assert {r["memory_id"] for r in rows if r not in _recent(rows)} == set(old[:7])
        assert [r["hybrid_score"] for r in rows] == sorted((r["hybrid_score"] for r in rows), reverse=True)

    async def test_the_recent_memory_that_answers_best_gets_in(self, world: _World) -> None:
        """Chosen on evidence, not on newness: the five-day-old best answer beats this morning's near misses."""
        _old, burst, best = await _history_and_burst(world)

        for cap in (1, 2, 3):
            rows = await world.ambient(recent_cap=cap)
            kept = {r["memory_id"] for r in _recent(rows)}
            assert best in kept, f"cap {cap} dropped the best answer"
            # the rest of the capped places go to the burst's best answers, in order
            assert kept - {best} == set(burst[: cap - 1])

    async def test_fewer_recent_than_the_cap_drops_nothing(self, world: _World) -> None:
        mine = [
            await world.remember("old", cosine=0.7, axis=10, age_hours=24.0 * 90),
            await world.remember("new", cosine=0.5, axis=11, age_hours=2.0),
            await world.remember("newer", cosine=0.45, axis=12, age_hours=1.0),
        ]

        rows = await world.ambient(recent_cap=3)

        assert {r["memory_id"] for r in rows} == set(mine)

    async def test_the_window_is_the_callers(self, world: _World) -> None:
        """A memory older than the window is history, whatever the cap."""
        _old, _burst, best = await _history_and_burst(world)

        rows = await world.ambient(recent_cap=0, recent_window_hours=96.0)

        # the burst is inside four days and all of it is capped out; the five-day-old one is history
        assert [r["memory_id"] for r in _recent(rows, days=4.0)] == []
        assert best in {r["memory_id"] for r in rows}

    async def test_no_private_key_reaches_the_caller(self, world: _World) -> None:
        await _history_and_burst(world)

        for rows in (await world.ambient(), await world.ambient(recent_cap=3)):
            assert not [key for row in rows for key in row if key.startswith("_")]


class _Embeds:
    async def aembed_query(self, _text: str) -> list[float]:
        return _QUERY


class _Nothing:
    async def hybrid_search(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return []


class _NoParts:
    """A chunk collection for memories that have no parts."""

    async def count_by_memory_id(self, *_args: Any, **_kwargs: Any) -> int:
        return 0

    async def find_for_recall(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return []

    async def find_by_memory_id(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return []


def _retriever(world: _World, **config: Any) -> MemoryRetriever:
    return MemoryRetriever(
        config=MemoryConfig(**config),
        embedding_provider=_Embeds(),  # type: ignore[arg-type]
        authorizer=MagicMock(spec=MemoryAuthorizerDependencies),
        memories_collection=world.memories,
        media_content_collection=_Nothing(),  # type: ignore[arg-type]
        memory_chunk_collection=_Nothing(),  # type: ignore[arg-type]
    )


async def _retrieved(world: _World, **config: Any) -> list[uuid.UUID]:
    result = await _retriever(world, **config).retrieve_with_candidates(
        world.user_id, "zz", agent_id=world.agent_id, customer_id=world.user_id
    )
    return [m["memory_id"] for m in result.memories]


class TestTheRetrieverPassesItsConfig:
    async def test_the_cap_reaches_the_search(self, world: _World) -> None:
        await _history_and_burst(world)

        capped = await _retrieved(world, ambient_recent_cap=3)
        rows = [await world.conn.fetchrow("SELECT date_created FROM memories WHERE memory_id = $1", m) for m in capped]

        assert len(capped) == 10 and len(_recent([dict(r) for r in rows])) == 3

    async def test_no_cap_is_the_default(self, world: _World) -> None:
        await _history_and_burst(world)

        rows = await world.conn.fetch(
            "SELECT date_created FROM memories WHERE memory_id = ANY($1::uuid[])", await _retrieved(world)
        )

        assert len(_recent([dict(r) for r in rows])) == 9


class TestAMemoryWrittenWithNobodyPresent:
    async def _three(self, world: _World) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
        attended = await world.remember("said to me", cosine=0.8, axis=10, tags=[made_by_tag("agent")])
        unattended = await world.remember(
            "noted on a wake", cosine=0.9, axis=11, tags=[made_by_tag("agent"), UNATTENDED_TAG]
        )
        unknown = await world.remember("from before origins were kept", cosine=0.7, axis=12)
        return attended, unattended, unknown

    async def test_it_stays_out_of_ambient_recall_and_the_others_stay_in(self, world: _World) -> None:
        attended, unattended, unknown = await self._three(world)

        got = await _retrieved(world, ambient_excluded_tags=(UNATTENDED_TAG,))

        assert unattended not in got
        # a memory someone was there for, and one whose origin nobody recorded, are both shown
        assert set(got) == {attended, unknown}

    async def test_it_is_shown_when_the_consumer_excludes_nothing(self, world: _World) -> None:
        _attended, unattended, _unknown = await self._three(world)

        assert unattended in await _retrieved(world)

    async def test_it_stays_searchable(self, world: _World) -> None:
        _attended, unattended, _unknown = await self._three(world)

        found = await world.memories.search_by_semantic(
            user_id=world.user_id, agent_id=world.agent_id, embedding=_QUERY, max_results=10, similarity_threshold=0.2
        )

        assert str(unattended) in {str(row["memory_id"]) for row in found}

    async def test_the_scope_excludes_in_both_arms(self, world: _World) -> None:
        """The keyword arm must not let in what the semantic arm left out."""
        _attended, unattended, _unknown = await self._three(world)

        rows = await world.memories.hybrid_search(
            user_id=world.user_id,
            agent_id=world.agent_id,
            customer_id=world.user_id,
            embedding=_QUERY,
            user_text="noted wake",
            top_k=10,
            candidate_limit=30,
            similarity_threshold=0.4,
            recency_half_life_hours=24.0,
            signal_weights=_WEIGHTS,
            scope=RetrievalScope(tags_none=(UNATTENDED_TAG,)),
        )

        assert unattended not in {r["memory_id"] for r in rows}

    def test_an_origin_nobody_recorded_is_not_read_as_attended(self) -> None:
        assert origin_of(None).unattended is None and origin_of([]).unattended is None
        assert origin_of(json.dumps(["made_by:agent"])).unattended is False
        assert origin_of(["made_by:dream", UNATTENDED_TAG]) == origin_of(json.dumps([UNATTENDED_TAG, "made_by:dream"]))
        assert origin_of(["made_by:dream", UNATTENDED_TAG]).made_by == "dream"


class TestBeingShownAndSalience:
    async def test_ambient_recall_raises_salience_by_default(self, world: _World) -> None:
        shown = await world.remember("shown", cosine=0.8, axis=10, age_hours=24.0 * 90)

        await _retrieved(world)

        assert (await world.row(shown))["salience"] == pytest.approx(0.55)

    async def test_it_does_not_when_the_consumer_turns_it_off(self, world: _World) -> None:
        shown = await world.remember("shown", cosine=0.8, axis=10, age_hours=24.0 * 90)

        assert await _retrieved(world, ambient_bumps_salience=False) == [shown]

        assert (await world.row(shown))["salience"] == pytest.approx(0.5)

    async def test_salience_does_not_move_a_memory_in_the_list(self, world: _World) -> None:
        """Twenty recalls of the last-placed memory leave it last: salience is a floor, not a score."""
        old, _burst, _best = await _history_and_burst(world)
        before = [r["memory_id"] for r in await world.ambient(recent_cap=3)]
        last = before[-1]

        for recalls in (1, 5, 20):
            await world.memories.bump_salience([last] * 1, agent_id=world.agent_id, access_bump=0.05 * recalls)
            assert [r["memory_id"] for r in await world.ambient(recent_cap=3)] == before

        assert (await world.row(last))["salience"] == pytest.approx(1.0)
        assert last in old

    async def test_below_the_floor_a_memory_is_not_shown_and_a_recall_it_was_asked_for_brings_it_back(
        self, world: _World, permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        sunk = await world.remember("sunk", cosine=0.8, axis=10, age_hours=24.0 * 120, salience=0.17)
        assert await _retrieved(world, ambient_bumps_salience=False) == []
        [recall] = await load_memory_recall_tool(
            world.user_id,
            world.agent_id,
            world.user_id,
            permissive_memory_authorizer,
            world.memories,
            MagicMock(),
            _NoParts(),  # type: ignore[arg-type]
            salience_bump=0.05,
        )

        said = await recall.ainvoke({"memory_id": str(sunk)})

        assert "sunk" in said
        assert (await world.row(sunk))["salience"] == pytest.approx(0.22)
        assert await _retrieved(world, ambient_bumps_salience=False) == [sunk]

    async def test_a_recall_does_not_raise_salience_unless_asked_to(
        self, world: _World, permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        opened = await world.remember("opened", cosine=0.8, axis=10)
        [recall] = await load_memory_recall_tool(
            world.user_id,
            world.agent_id,
            world.user_id,
            permissive_memory_authorizer,
            world.memories,
            MagicMock(),
            _NoParts(),  # type: ignore[arg-type]
        )

        assert "opened" in await recall.ainvoke({"memory_id": str(opened)})

        assert (await world.row(opened))["salience"] == pytest.approx(0.5)


_SAME = 0.75


class TestOneEventIsStoredOnce:
    """Vectors: the first write sits at cosine 0.9 to the query on its own axis; a rewrite of the same
    event sits at 0.9 on the SAME axis (cosine 1.0 to it); a different fact sits at 0.6 on another axis
    (cosine 0.54 to it, under the threshold)."""

    async def _write(
        self,
        world: _World,
        content: str,
        *,
        axis: int,
        message_id: uuid.UUID,
        near: Any,
        cosine: float = 0.9,
        **more: Any,
    ):
        return await write_once(
            world.memories,
            world.edges,
            agent_id=world.agent_id,
            fields=world.fields(content, cosine=cosine, axis=axis, message_id=message_id, **more),
            near_message_ids=near,
            threshold=_SAME,
        )

    async def test_the_second_write_from_the_same_turn_replaces_the_first(self, world: _World) -> None:
        turn = uuid.uuid4()
        first = await self._write(world, "first wording", axis=10, message_id=turn, near=[turn])

        second = await self._write(world, "second wording", axis=10, message_id=turn, near=[turn])

        assert first.memory_id is not None and second.memory_id is not None
        assert [row["memory_id"] for row in second.replaced] == [first.memory_id]
        assert (await world.row(first.memory_id))["superseded_by"] == second.memory_id
        assert await world.live() == ["second wording"]
        why = await world.conn.fetchval(
            "SELECT rationale FROM memory_consolidations WHERE consolidated_memory_id = $1", second.memory_id
        )
        assert why == SAME_EVENT_RATIONALE

    async def test_a_write_from_the_next_turn_replaces_it_too(self, world: _World) -> None:
        turn, next_turn = uuid.uuid4(), uuid.uuid4()
        await self._write(world, "first wording", axis=10, message_id=turn, near=[turn])

        await self._write(world, "said again", axis=10, message_id=next_turn, near=[next_turn, turn])

        assert await world.live() == ["said again"]

    async def test_three_writes_of_one_event_leave_one_memory(self, world: _World) -> None:
        """Extraction, the agent's own save, extraction again: the order production stored four times."""
        turn, next_turn = uuid.uuid4(), uuid.uuid4()
        await self._write(world, "extracted", axis=10, message_id=turn, near=[turn])
        await self._write(world, "saved by the agent", axis=10, message_id=turn, near=[turn])

        await self._write(world, "extracted again", axis=10, message_id=next_turn, near=[next_turn, turn])

        assert await world.live() == ["extracted again"]
        assert await world.conn.fetchval("SELECT count(*) FROM memories WHERE agent_id = $1", world.agent_id) == 3

    async def test_a_different_fact_from_the_same_turn_is_not_merged(self, world: _World) -> None:
        turn = uuid.uuid4()
        await self._write(world, "one fact", axis=10, message_id=turn, near=[turn])

        other = await self._write(world, "another fact", axis=11, cosine=0.6, message_id=turn, near=[turn])

        assert other.replaced == ()
        assert await world.live() == ["one fact", "another fact"]

    async def test_the_same_words_from_an_older_turn_are_not_merged(self, world: _World) -> None:
        """Alike is not enough: the earlier memory must come from this turn or the one before."""
        long_ago, turn = uuid.uuid4(), uuid.uuid4()
        await self._write(world, "said in spring", axis=10, message_id=long_ago, near=[long_ago])

        again = await self._write(world, "said in autumn", axis=10, message_id=turn, near=[turn])

        assert again.replaced == ()
        assert await world.live() == ["said in spring", "said in autumn"]

    async def test_a_permanent_first_write_is_never_replaced(self, world: _World) -> None:
        turn = uuid.uuid4()
        kept = await self._write(world, "kept", axis=10, message_id=turn, near=[turn], evergreen=True)
        fields = world.fields("again", cosine=0.9, axis=10, message_id=turn)

        skipped = await write_once(
            world.memories,
            world.edges,
            agent_id=world.agent_id,
            fields=fields,
            near_message_ids=[turn],
            threshold=_SAME,
            when_permanent="skip",
        )
        beside = await write_once(
            world.memories,
            world.edges,
            agent_id=world.agent_id,
            fields=fields,
            near_message_ids=[turn],
            threshold=_SAME,
        )

        assert skipped.memory_id is None and skipped.standing is not None
        assert skipped.standing["memory_id"] == kept.memory_id
        assert beside.memory_id is not None and beside.replaced == ()
        assert (await world.row(kept.memory_id))["superseded_by"] is None  # type: ignore[arg-type]
        assert await world.live() == ["kept", "again"]

    async def test_with_no_threshold_every_write_is_its_own_memory(self, world: _World) -> None:
        turn = uuid.uuid4()
        for wording in ("first", "second"):
            await write_once(
                world.memories,
                world.edges,
                agent_id=world.agent_id,
                fields=world.fields(wording, cosine=0.9, axis=10, message_id=turn),
                near_message_ids=[turn],
            )

        assert await world.live() == ["first", "second"]

    async def test_a_retired_first_write_is_not_brought_back_as_a_source(self, world: _World) -> None:
        turn = uuid.uuid4()
        first = await self._write(world, "wrong", axis=10, message_id=turn, near=[turn])
        assert first.memory_id is not None
        await world.memories.mark_retracted(world.agent_id, memory_id=first.memory_id, reason="not true")

        second = await self._write(world, "right", axis=10, message_id=turn, near=[turn])

        assert second.replaced == ()
        assert (await world.row(first.memory_id))["superseded_by"] is None


class _Model:
    """A chat model that answers with fixed text and keeps what it was asked."""

    def __init__(self, content: str, asked: list[str]) -> None:
        self._content, self._asked = content, asked

    async def ainvoke(self, messages: list[Any], **_kwargs: Any) -> Any:
        self._asked.append("\n".join(str(m.content) for m in messages))
        reply = MagicMock()
        reply.content = self._content
        return reply


class _Models:
    def __init__(self, extracted: str) -> None:
        self.asked: list[str] = []
        self._by_purpose = {
            "worthiness": json.dumps({"worthy": True, "reason": "has facts"}),
            "extraction": json.dumps([{"type": "fact", "content": extracted}]),
        }

    async def create_chat_model(self, purpose: str = "extraction") -> Any:
        return _Model(self._by_purpose[purpose], self.asked)


class _Fixed:
    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    async def aembed_query(self, _text: str) -> list[float]:
        return self._vector


_CONFESSION = "I fabricated that line. I had no source for it and I should not have said it. " * 3
_HER_SISTER = "My sister Jeva moved to Lisbon in May and she is opening a bakery there."


class TestExtraction:
    def _extractor(
        self,
        world: _World,
        authorizer: MemoryAuthorizerDependencies,
        models: _Models,
        *,
        axis: int = 10,
        threshold: float | None = None,
    ) -> MemoryExtractor:
        return MemoryExtractor(
            config=MemoryConfig(same_event_threshold=threshold, similar_memory_threshold=2.0),
            embedding_provider=_Fixed(_at(0.9, axis)),  # type: ignore[arg-type]
            chat_model_factory=models,
            authorizer=authorizer,
            memories_collection=world.memories,
            consolidations_collection=world.edges,
        )

    async def _extract(self, world: _World, extractor: MemoryExtractor, said: str, turn: uuid.UUID, **more: Any):
        return await extractor.extract(
            user_id=world.user_id,
            conversation_id=uuid.uuid4(),
            message_id_source=turn,
            user_message=said,
            assistant_response=_CONFESSION,
            turn_count=10,
            agent_id=world.agent_id,
            customer_id=world.user_id,
            **more,
        )

    async def test_a_reply_that_is_not_a_source_is_shown_to_no_model(
        self, world: _World, permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        """Its own words reach neither the worthiness call nor the extraction call; the person's do."""
        models = _Models("Her sister Jeva moved to Lisbon in May.")
        extractor = self._extractor(world, permissive_memory_authorizer, models)

        result = await self._extract(world, extractor, _HER_SISTER, uuid.uuid4(), reply_is_source=False)

        assert result.stored == 1 and await world.live() == ["Her sister Jeva moved to Lisbon in May."]
        assert len(models.asked) == 2
        for prompt in models.asked:
            assert "fabricated" not in prompt
            assert REPLY_NOT_A_SOURCE in prompt
            assert "Jeva" in prompt

    async def test_when_the_reply_is_the_only_source_nothing_is_written_and_no_model_is_asked(
        self, world: _World, permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        models = _Models("I fabricated a line and had no source for it.")
        extractor = self._extractor(world, permissive_memory_authorizer, models)

        for said in ("", "   ", "why?"):
            result = await self._extract(world, extractor, said, uuid.uuid4(), reply_is_source=False)
            assert result.stored == 0

        assert models.asked == [] and await world.live() == []

    async def test_a_reply_that_is_a_source_is_read(
        self, world: _World, permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        models = _Models("We agreed on it.")
        extractor = self._extractor(world, permissive_memory_authorizer, models)

        await self._extract(world, extractor, _HER_SISTER, uuid.uuid4())

        assert all("fabricated" in prompt for prompt in models.asked) and len(models.asked) == 2

    async def test_extraction_twice_on_one_turn_keeps_one_memory(
        self, world: _World, permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        turn = uuid.uuid4()
        first = self._extractor(world, permissive_memory_authorizer, _Models("first wording"), threshold=_SAME)
        again = self._extractor(world, permissive_memory_authorizer, _Models("second wording"), threshold=_SAME)

        await self._extract(world, first, _HER_SISTER, turn, near_message_ids=[turn])
        result = await self._extract(world, again, _HER_SISTER, turn, near_message_ids=[turn])

        assert await world.live() == ["second wording"]
        assert "updated=1" in (result.reason or "") and result.stored == 1

    async def test_extraction_after_the_agents_own_save_replaces_it_and_a_permanent_one_is_left_alone(
        self, world: _World, permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        turn, other_turn = uuid.uuid4(), uuid.uuid4()
        await world.remember("the agent's own save", cosine=0.9, axis=10, message_id=turn)
        await world.remember("kept by the agent", cosine=0.9, axis=11, message_id=other_turn, permanent=True)
        over_own = self._extractor(world, permissive_memory_authorizer, _Models("extracted"), threshold=_SAME)
        over_kept = self._extractor(
            world, permissive_memory_authorizer, _Models("extracted over kept"), axis=11, threshold=_SAME
        )

        await self._extract(world, over_own, _HER_SISTER, turn, near_message_ids=[turn])
        result = await self._extract(world, over_kept, _HER_SISTER, other_turn, near_message_ids=[other_turn])

        assert await world.live() == ["kept by the agent", "extracted"]
        assert result.stored == 0 and "unchanged=1" in (result.reason or "")

    async def test_a_different_fact_from_the_same_turn_is_added_beside(
        self, world: _World, permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        turn = uuid.uuid4()
        await world.remember("one fact", cosine=0.6, axis=10, message_id=turn)
        extractor = self._extractor(world, permissive_memory_authorizer, _Models("another"), axis=11, threshold=_SAME)

        await self._extract(world, extractor, _HER_SISTER, turn, near_message_ids=[turn])

        assert await world.live() == ["one fact", "another"]


class _Ctx:
    def __init__(self, message_id: uuid.UUID) -> None:
        self.conversation_id, self.correlation_id = uuid.uuid4(), message_id


class TestTheAgentsOwnSave:
    async def _add(
        self, world: _World, authorizer: MemoryAuthorizerDependencies, turn: uuid.UUID, *, axis: int = 10, **more: Any
    ) -> Any:
        async def near() -> list[uuid.UUID]:
            return [turn]

        [add] = await load_memory_add_tool(
            world.user_id,
            _Fixed(_at(0.9, axis)),  # type: ignore[arg-type]
            world.agent_id,
            world.user_id,
            authorizer,
            world.memories,
            context_resolver=lambda: _Ctx(turn),
            consolidations_collection=world.edges,
            **{"near_message_ids": near, "same_event_threshold": _SAME, **more},
        )
        return add

    async def test_a_save_after_extraction_of_the_same_turn_replaces_it_and_says_so(
        self, world: _World, permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        """Below the 90% near-duplicate line, so only the same-turn rule can have merged them."""
        turn = uuid.uuid4()
        extracted = await world.remember("extracted wording", cosine=0.6, axis=10, message_id=turn)
        add = await self._add(world, permissive_memory_authorizer, turn)

        said = await add.ainvoke({"content": "my own wording"})

        assert await world.live() == ["my own wording"]
        assert f"It replaces [memory:{extracted}] (" in said and '"extracted wording"' in said
        assert said.endswith("If they are two facts and not one, store it again with replace=false.")

    async def test_replace_false_writes_beside_it(
        self, world: _World, permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        turn = uuid.uuid4()
        await world.remember("extracted wording", cosine=0.6, axis=10, message_id=turn)
        add = await self._add(world, permissive_memory_authorizer, turn)

        said = await add.ainvoke({"content": "a second fact", "replace": False})

        assert await world.live() == ["extracted wording", "a second fact"] and "replaces" not in said

    async def test_a_different_fact_and_an_older_turn_are_left_alone(
        self, world: _World, permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        turn = uuid.uuid4()
        await world.remember("another fact of this turn", cosine=0.6, axis=11, message_id=turn)
        await world.remember("alike, from long ago", cosine=0.6, axis=10, message_id=uuid.uuid4())
        add = await self._add(world, permissive_memory_authorizer, turn)

        said = await add.ainvoke({"content": "new"})

        assert len(await world.live()) == 3 and "replaces" not in said

    async def test_without_the_rule_a_save_is_beside(
        self, world: _World, permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        turn = uuid.uuid4()
        await world.remember("extracted wording", cosine=0.6, axis=10, message_id=turn)
        add = await self._add(world, permissive_memory_authorizer, turn, same_event_threshold=None)

        await add.ainvoke({"content": "my own wording"})

        assert await world.live() == ["extracted wording", "my own wording"]

    async def test_permanent_is_refused_when_the_host_says_it_needs_the_person(
        self, world: _World, permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        asked: list[str] = []

        async def only_plain_facts(content: str) -> bool:
            asked.append(content)
            return "failed" not in content

        turn = uuid.uuid4()
        add = await self._add(world, permissive_memory_authorizer, turn, permanent_allowed=only_plain_facts)
        other = await self._add(world, permissive_memory_authorizer, turn, axis=11, permanent_allowed=only_plain_facts)

        refused = await add.ainvoke({"content": "I failed to check the record", "permanent": True})
        allowed = await other.ainvoke({"content": "Her sister is Jeva", "permanent": True})

        rows = await world.conn.fetch("SELECT content, evergreen FROM memories ORDER BY date_created")
        assert [(r["content"], r["evergreen"]) for r in rows] == [
            ("I failed to check the record", False),
            ("Her sister is Jeva", True),
        ]
        assert refused.endswith(PERMANENT_NEEDS_THE_PERSON) and "Kept permanently" not in refused
        assert "Kept permanently" in allowed and PERMANENT_NEEDS_THE_PERSON not in allowed
        assert asked == ["I failed to check the record", "Her sister is Jeva"]

    async def test_keep_is_refused_the_same_way_and_allowed_by_default(
        self, world: _World, permissive_memory_authorizer: MemoryAuthorizerDependencies
    ) -> None:
        async def never(_content: str) -> bool:
            return False

        mine = await world.remember("I failed", cosine=0.9, axis=10)
        args = (world.user_id, world.agent_id, world.user_id, permissive_memory_authorizer, world.memories)
        [gated] = load_memory_keep_tool(*args, permanent_allowed=never)
        [plain] = load_memory_keep_tool(*args)

        refused = await gated.ainvoke({"memory_id": str(mine)})
        assert (await world.row(mine))["evergreen"] is False
        assert refused.startswith("[NO CHANGE]") and PERMANENT_NEEDS_THE_PERSON in refused

        assert "is kept permanently" in await plain.ainvoke({"memory_id": str(mine)})
        assert (await world.row(mine))["evergreen"] is True


class TestADream:
    async def _pair(self, world: _World, tags: tuple[list[str] | None, list[str] | None]) -> None:
        for content, tag in zip(("noted once", "noted twice"), tags, strict=True):
            await world.remember(content, cosine=0.9, axis=10, tags=tag)

    async def _dream(self, world: _World, answer: dict[str, Any]) -> dict[str, Any]:
        service = make_service(world.pool, gist_vector=_at(0.5, 20), reflector_content=json.dumps(answer))
        result = await service.run_consolidation(world.agent_id, customer_id=world.user_id, user_id=world.user_id)
        [gist] = result.gist_ids
        return await world.row(uuid.UUID(str(gist)))

    async def test_never_makes_a_memory_permanent_whatever_its_model_answers(self, world: _World) -> None:
        await self._pair(world, (None, None))

        gist = await self._dream(world, {"gist": "noted", "rationale": "x", "one_subject": True, "permanent": True})

        assert gist["evergreen"] is False

    async def test_a_gist_of_only_unattended_memories_is_unattended(self, world: _World) -> None:
        await self._pair(world, ([UNATTENDED_TAG], [made_by_tag("agent"), UNATTENDED_TAG]))

        gist = await self._dream(world, {"gist": "noted", "rationale": "x", "one_subject": True})

        assert origin_of(gist["tags"]).unattended is True
        assert await _retrieved(world, ambient_excluded_tags=(UNATTENDED_TAG,)) == []

    @pytest.mark.parametrize("other", [None, [], ["made_by:agent"]])
    async def test_one_source_someone_was_there_for_keeps_the_gist_in(
        self, world: _World, other: list[str] | None
    ) -> None:
        await self._pair(world, ([UNATTENDED_TAG], other))

        gist = await self._dream(world, {"gist": "noted", "rationale": "x", "one_subject": True})

        assert origin_of(gist["tags"]).unattended is not True
        assert len(await _retrieved(world, ambient_excluded_tags=(UNATTENDED_TAG,))) == 1
