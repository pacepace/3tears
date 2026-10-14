"""integration: what the memory tools hand an agent says which memories still stand.

Every tool is built by its real factory over the real collections on a real
pgvector database. The cases are the ways a result used to mislead:

- a replaced memory and its replacement came back as two equal facts;
- a retired memory came back from a search, unmarked;
- ``memory_add`` named the id it replaced and not the text, and said nothing of
  a permanent near-twin it left standing;
- nothing corrected, retired or restored a memory, and nothing listed them;
- a bare ``date_before`` left out the day it named;
- on a turn nobody was present for, ``memory_add`` replaced a memory;
- a cut preview read as the whole passage, and no result said how many more
  there were.

Each guard that can hide a memory is paired with the memory it must still show.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import asyncpg
import pytest
import uuid_utils

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import (
    MediaContentCollection,
    MemoriesCollection,
    MemoryChunkCollection,
    MemoryConsolidationsCollection,
)
from threetears.agent.memory.migrations import register as register_memory
from threetears.agent.memory.revisions import marks_of, restore, retract, supersede
from threetears.agent.memory.tools import (
    CHANGE_NEEDS_THE_PERSON,
    PERMANENT_IS_THE_PERSONS,
    load_chunk_recall_tool,
    load_chunk_search_tool,
    load_memory_add_tool,
    load_memory_forget_tool,
    load_memory_keep_tool,
    load_memory_list_tool,
    load_memory_recall_tool,
    load_memory_restore_tool,
    load_memory_search_tool,
    load_memory_update_tool,
)
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore
from .memory_support import DIM, make_pool

pytestmark = pytest.mark.integration

#: Each subject has its own direction, so a query about one is near only its own memories.
_AXES = {"tea": 0, "surname": 1, "sleeve": 2, "harbour": 3, "garden": 4}
_ELSEWHERE = 9
_LA = "America/Los_Angeles"


def _axis(index: int) -> list[float]:
    vector = [0.0] * DIM
    vector[index] = 1.0
    return vector


def _vector_for(text: str) -> list[float]:
    for word, index in _AXES.items():
        if word in text.lower():
            return _axis(index)
    return _axis(_ELSEWHERE)


class _SubjectEmbeddings:
    """An embedder that puts a text on the axis of the subject it names."""

    async def aembed_query(self, text: str) -> list[float]:
        return _vector_for(text)

    def embed_query(self, text: str) -> list[float]:
        return _vector_for(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_vector_for(t) for t in texts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_vector_for(t) for t in texts]


class _Ctx:
    def __init__(self) -> None:
        self.conversation_id = uuid.uuid4()
        self.correlation_id = uuid.uuid4()


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


@dataclass
class _World:
    """One person's memories on a real database, and every tool built over them."""

    pool: asyncpg.Pool
    conn: asyncpg.Connection
    authorizer: MemoryAuthorizerDependencies
    memories: MemoriesCollection
    edges: MemoryConsolidationsCollection
    chunks: MemoryChunkCollection
    media_content: MediaContentCollection
    agent_id: uuid.UUID
    user_id: uuid.UUID

    async def memory(
        self,
        content: str,
        *,
        age_days: float = 0,
        at: datetime | None = None,
        kind: str = "fact",
        permanent: bool = False,
        alias: str | None = None,
        user_id: uuid.UUID | None = None,
        salience: float = 0.5,
        source: uuid.UUID | None = None,
    ) -> uuid.UUID:
        memory_id = uuid.UUID(str(uuid_utils.uuid7()))
        created = at or datetime.now(UTC) - timedelta(days=age_days)
        owner = user_id or self.user_id
        await self.conn.execute(
            "INSERT INTO memories (memory_id, agent_id, customer_id, user_id, conversation_id, type_memory, "
            "content, embedding, salience, evergreen, alias, date_created, date_updated, message_id_source) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7,$8::text::public.vector,$9,$10,$11,$12,$12,$13)",
            memory_id,
            self.agent_id,
            owner,
            owner,
            uuid.uuid4(),
            kind,
            content,
            json.dumps(_vector_for(content)),
            salience,
            permanent,
            alias,
            created,
            source,
        )
        return memory_id

    async def chunk(self, memory_id: uuid.UUID, content: str, *, index: int, at: datetime | None = None) -> uuid.UUID:
        chunk_id = uuid.UUID(str(uuid_utils.uuid7()))
        await self.conn.execute(
            "INSERT INTO memory_chunks (chunk_id, memory_id, agent_id, customer_id, user_id, chunk_index, content, "
            "embedding, date_created) VALUES ($1,$2,$3,$4,$4,$5,$6,$7::text::public.vector,$8)",
            chunk_id,
            memory_id,
            self.agent_id,
            self.user_id,
            index,
            content,
            json.dumps(_vector_for(content)),
            at or datetime.now(UTC),
        )
        return chunk_id

    async def file(self, memory_id: uuid.UUID, content: str, *, title: str, at: datetime | None = None) -> uuid.UUID:
        media_id, content_id = uuid.uuid4(), uuid.uuid4()
        when = at or datetime.now(UTC)
        await self.conn.execute(
            "INSERT INTO media (media_id, memory_id, agent_id, customer_id, user_id, media_category, metadata_json, "
            "date_created, date_updated) VALUES ($1,$2,$3,$4,$4,'document',$5::jsonb,$6,$6)",
            media_id,
            memory_id,
            self.agent_id,
            self.user_id,
            json.dumps({"document_title": title}),
            when,
        )
        await self.conn.execute(
            "INSERT INTO media_content (content_id, media_id, agent_id, customer_id, user_id, content_type, content, "
            "embedding, date_created) VALUES ($1,$2,$3,$4,$4,'ocr',$5,$6::text::public.vector,$7)",
            content_id,
            media_id,
            self.agent_id,
            self.user_id,
            content,
            json.dumps(_vector_for(content)),
            when,
        )
        return content_id

    async def row(self, memory_id: uuid.UUID) -> dict[str, Any]:
        row = await self.conn.fetchrow(
            "SELECT content, type_memory, salience, evergreen, superseded_by, tags, alias FROM memories "
            "WHERE memory_id = $1",
            memory_id,
        )
        assert row is not None, "a memory was deleted"
        return dict(row)

    async def count(self) -> int:
        return int(await self.conn.fetchval("SELECT count(*) FROM memories WHERE agent_id = $1", self.agent_id))

    async def replace(self, old: uuid.UUID, content: str) -> uuid.UUID:
        return await supersede(
            self.memories,
            self.edges,
            agent_id=self.agent_id,
            source_ids=[old],
            fields={
                "customer_id": self.user_id,
                "user_id": self.user_id,
                "conversation_id": uuid.uuid4(),
                "message_id_source": uuid.uuid4(),
                "type_memory": "fact",
                "content": content,
                "embedding": _vector_for(content),
                "salience": 0.5,
            },
            rationale="test",
        )

    # -- the tools, each from its real factory ---------------------------------

    def _ids(self) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID, MemoryAuthorizerDependencies]:
        return self.user_id, self.agent_id, self.user_id, self.authorizer

    async def search(self, **kwargs: Any) -> Any:
        [tool] = await load_memory_search_tool(
            self.user_id,
            _SubjectEmbeddings(),  # type: ignore[arg-type]
            self.agent_id,
            self.user_id,
            self.authorizer,
            self.memories,
            self.media_content,
            self.chunks,
            **kwargs,
        )
        return tool

    async def recall(self, **kwargs: Any) -> Any:
        [tool] = await load_memory_recall_tool(
            self.user_id,
            self.agent_id,
            self.user_id,
            self.authorizer,
            self.memories,
            self.media_content,
            self.chunks,
            _SubjectEmbeddings(),  # type: ignore[arg-type]
            consolidations_collection=self.edges,
            **kwargs,
        )
        return tool

    async def add(self, **kwargs: Any) -> Any:
        [tool] = await load_memory_add_tool(
            self.user_id,
            _SubjectEmbeddings(),  # type: ignore[arg-type]
            self.agent_id,
            self.user_id,
            self.authorizer,
            self.memories,
            context_resolver=_Ctx,
            consolidations_collection=self.edges,
            **kwargs,
        )
        return tool

    async def update(self, **kwargs: Any) -> Any:
        [tool] = await load_memory_update_tool(
            self.user_id,
            _SubjectEmbeddings(),  # type: ignore[arg-type]
            self.agent_id,
            self.user_id,
            self.authorizer,
            self.memories,
            self.edges,
            context_resolver=_Ctx,
            **kwargs,
        )
        return tool

    def forget(self, **kwargs: Any) -> Any:
        [tool] = load_memory_forget_tool(*self._ids(), self.memories, **kwargs)
        return tool

    def restore(self, **kwargs: Any) -> Any:
        [tool] = load_memory_restore_tool(*self._ids(), self.memories, **kwargs)
        return tool

    def listing(self, **kwargs: Any) -> Any:
        [tool] = load_memory_list_tool(*self._ids(), self.memories, **kwargs)
        return tool

    def keep(self) -> Any:
        [tool] = load_memory_keep_tool(*self._ids(), self.memories)
        return tool

    async def chunk_search(self, **kwargs: Any) -> Any:
        [tool] = await load_chunk_search_tool(
            self.user_id,
            self.agent_id,
            self.user_id,
            _SubjectEmbeddings(),  # type: ignore[arg-type]
            self.authorizer,
            self.chunks,
            **kwargs,
        )
        return tool

    async def chunk_recall(self, **kwargs: Any) -> Any:
        [tool] = await load_chunk_recall_tool(*self._ids(), self.memories, self.chunks, **kwargs)
        return tool


@pytest.fixture
async def world(
    applied_schema: tuple[str, str], permissive_memory_authorizer: MemoryAuthorizerDependencies
) -> AsyncIterator[_World]:
    url, schema = applied_schema
    pool, conn = await make_pool(url, schema), await asyncpg.connect(url)
    await conn.execute(f'SET search_path TO "{schema}", public')
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    try:
        yield _World(
            pool=pool,
            conn=conn,
            authorizer=permissive_memory_authorizer,
            memories=MemoriesCollection(
                registry=registry, config=config, authorizer=MagicMock(spec=MemoryAuthorizerDependencies)
            ),
            edges=MemoryConsolidationsCollection(registry=registry, config=config, nats_client=None),
            chunks=MemoryChunkCollection(registry=registry, config=config),
            media_content=MediaContentCollection(registry=registry, config=config),
            agent_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
        )
    finally:
        await conn.close()
        await pool.close()


def _line(result: str, memory_id: uuid.UUID) -> str:
    """The one line of a result that carries ``memory_id`` as its own id."""
    hits = [line for line in result.splitlines() if line.lstrip("- ").startswith(f"[memory:{memory_id}]")]
    assert len(hits) == 1, f"expected one line for {memory_id} in:\n{result}"
    return hits[0]


# --------------------------------------------------------------------------
# memory_search: every line says whether the memory still stands
# --------------------------------------------------------------------------


class TestASearchSaysWhichMemoryStands:
    async def test_a_replaced_memory_is_marked_with_what_replaced_it_and_when(self, world: _World) -> None:
        old = await world.memory("the surname is Brannigan", age_days=30)
        new = await world.replace(old, "the surname is Brennan")
        search = await world.search(tz=_LA)

        said = await search.ainvoke({"query": "surname"})

        old_line, new_line = _line(said, old), _line(said, new)
        assert f"(replaced by [memory:{new}] on " in old_line and "; that one is current)" in old_line
        # the mark sits before the text, so a reader that stops early has read it
        assert old_line.index("(replaced by") < old_line.index("the surname is Brannigan")
        assert "replaced" not in new_line and new_line.endswith("the surname is Brennan")

    async def test_a_memory_whose_replacement_is_gone_is_current_again(self, world: _World) -> None:
        old = await world.memory("the surname is Brannigan", age_days=30)
        new = await world.replace(old, "the surname is Brennan")
        await world.conn.execute("DELETE FROM memory_consolidations WHERE consolidated_memory_id = $1", new)
        await world.conn.execute("DELETE FROM memories WHERE memory_id = $1", new)
        search = await world.search()

        said = await search.ainvoke({"query": "surname"})

        assert "replaced" not in _line(said, old)

    async def test_a_permanent_memory_and_a_named_one_say_so(self, world: _World) -> None:
        kept = await world.memory("the surname is Brennan", permanent=True, alias="family-name")
        plain = await world.memory("a surname is a family name")
        search = await world.search()

        said = await search.ainvoke({"query": "surname"})

        assert "(alias: family-name) (permanent) the surname is Brennan" in _line(said, kept)
        assert "(permanent)" not in _line(said, plain) and "(alias" not in _line(said, plain)

    @pytest.mark.parametrize("mode", ["fuzzy", "precise", "balanced"])
    async def test_a_retired_memory_is_not_found_and_the_one_beside_it_is(self, world: _World, mode: str) -> None:
        wrong = await world.memory("the harbour closes at noon")
        right = await world.memory("the harbour closes at dusk")
        assert await retract(world.memories, agent_id=world.agent_id, memory_id=wrong, reason="it closes at dusk")
        search = await world.search()

        said = await search.ainvoke({"query": "harbour", "mode": mode})

        assert str(wrong) not in said, "a retired memory came back from a search"
        assert "the harbour closes at dusk" in _line(said, right)

    async def test_a_retired_memory_still_opens_by_id_and_by_alias_and_says_why(self, world: _World) -> None:
        wrong = await world.memory("the harbour closes at noon", alias="harbour-hours")
        assert await retract(world.memories, agent_id=world.agent_id, memory_id=wrong, reason="it closes at dusk")
        search = await world.search()

        by_id = await search.ainvoke({"ids": [str(wrong)]})
        by_alias = await search.ainvoke({"alias": "harbour-hours", "query": ""})

        for said in (by_id, by_alias):
            line = _line(said, wrong)
            assert "(retired, no longer holds: it closes at dusk)" in line
            assert line.index("(retired") < line.index("the harbour closes at noon")

    async def test_a_correction_beside_a_permanent_memory_shows_on_its_line(self, world: _World) -> None:
        kept = await world.memory("the sleeve was his choice", permanent=True)
        add = await world.add()
        stored = await add.ainvoke({"content": "the sleeve choice was mine", "corrects": [str(kept)]})
        new_id = stored.split("[memory:")[1].split("]")[0]
        search = await world.search()

        said = await search.ainvoke({"query": "sleeve"})

        assert f"(permanent) (corrected by [memory:{new_id}]) the sleeve was his choice" in _line(said, kept)

    async def test_the_score_shows_how_each_memory_matched(self, world: _World) -> None:
        near = await world.memory("green tea in the morning")
        # no subject word the embedder knows, so only its words can match
        by_word = await world.memory("the kettle is for tea-time only")
        await world.conn.execute(
            "UPDATE memories SET embedding = $2::text::public.vector WHERE memory_id = $1",
            by_word,
            json.dumps(_axis(_ELSEWHERE)),
        )
        search = await world.search()

        said = await search.ainvoke({"query": "tea"})

        assert "[match 1.00]" in _line(said, near)
        assert "[word match]" in _line(said, by_word)


class TestASearchSaysWhatItLeftOut:
    async def test_relational_context_can_be_filtered_by(self, world: _World) -> None:
        relational = await world.memory("we talk about the garden every Sunday", kind="relational_context")
        fact = await world.memory("the garden faces south")
        search = await world.search()

        said = await search.ainvoke({"query": "garden", "type_filter": "relational_context"})

        assert str(relational) in said and str(fact) not in said

    async def test_a_type_that_does_not_exist_is_refused_with_the_list(self, world: _World) -> None:
        search = await world.search()
        said = await search.ainvoke({"query": "garden", "type_filter": "gossip"})
        assert said.startswith("[TOOL ERROR]") and "relational_context" in said

    async def test_a_bare_date_before_includes_that_whole_day_where_the_person_lives(self, world: _World) -> None:
        zone = ZoneInfo(_LA)
        # 9 Oct 22:30 in Los Angeles is already 10 Oct in UTC
        late_on_the_9th = datetime(2026, 10, 9, 22, 30, tzinfo=zone)
        early_on_the_10th = datetime(2026, 10, 10, 0, 30, tzinfo=zone)
        inside = await world.memory("tea on the ninth", at=late_on_the_9th)
        outside = await world.memory("tea on the tenth", at=early_on_the_10th)
        search = await world.search(tz=_LA)

        said = await search.ainvoke({"query": "tea", "date_before": "2026-10-09"})

        assert str(inside) in said, "the day named was left out"
        assert str(outside) not in said

    async def test_a_bare_date_after_starts_at_that_days_first_minute_where_the_person_lives(
        self, world: _World
    ) -> None:
        zone = ZoneInfo(_LA)
        inside = await world.memory("tea at half past midnight", at=datetime(2026, 10, 9, 0, 30, tzinfo=zone))
        outside = await world.memory("tea the evening before", at=datetime(2026, 10, 8, 23, 30, tzinfo=zone))
        search = await world.search(tz=_LA)

        said = await search.ainvoke({"query": "tea", "date_after": "2026-10-09"})

        assert str(inside) in said and str(outside) not in said

    async def test_the_date_filter_reaches_files_and_passages_too(self, world: _World) -> None:
        old, new = datetime(2026, 9, 1, 12, tzinfo=UTC), datetime(2026, 10, 9, 12, tzinfo=UTC)
        parent = await world.memory("notes on tea", at=old)
        old_file = await world.file(parent, "tea ledger, September", title="September", at=old)
        new_file = await world.file(parent, "tea ledger, October", title="October", at=new)
        old_part = await world.chunk(parent, "tea passage from September", index=0, at=old)
        new_part = await world.chunk(parent, "tea passage from October", index=1, at=new)
        # A file is listed by the id the media tools take, its media id.
        new_media, old_media = [
            await world.conn.fetchval("SELECT media_id FROM media_content WHERE content_id = $1", c)
            for c in (new_file, old_file)
        ]
        search = await world.search()

        said = await search.ainvoke({"query": "tea", "date_after": "2026-10-01"})

        assert str(new_media) in said and str(new_part) in said
        assert str(old_media) not in said and str(old_part) not in said and str(parent) not in said
        # and with no filter, the older ones are there: the filter is what removed them
        everything = await search.ainvoke({"query": "tea"})
        assert str(old_media) in everything and str(old_part) in everything

    async def test_a_full_page_says_more_may_match_first_and_offset_reads_on(self, world: _World) -> None:
        ids = [await world.memory(f"tea note {i}", age_days=i) for i in range(5)]
        search = await world.search()

        first = await search.ainvoke({"query": "tea", "limit": 2, "mode": "fuzzy"})
        second = await search.ainvoke({"query": "tea", "limit": 2, "offset": 2, "mode": "fuzzy"})
        last = await search.ainvoke({"query": "tea", "limit": 2, "offset": 4, "mode": "fuzzy"})

        assert first.splitlines()[0] == ("(More memories may match than the 2 shown: call again with offset=2.)"), (
            "the note is not the first line"
        )
        assert second.splitlines()[0] == "(More memories may match than the 2 shown: call again with offset=4.)"
        assert last.splitlines()[0] == "(Memories 5-5; no more match.)"
        shown = [m for page in (first, second, last) for m in ids if str(m) in page]
        assert sorted(shown) == sorted(ids), "paging skipped or repeated a memory"

    async def test_a_page_that_holds_everything_says_nothing_about_more(self, world: _World) -> None:
        await world.memory("tea note")
        search = await world.search()
        said = await search.ainvoke({"query": "tea"})
        assert said.startswith("Found 1 relevant memories:") and "More" not in said

    async def test_an_id_that_is_not_one_is_named_first_and_the_rest_are_read(self, world: _World) -> None:
        real = await world.memory("the harbour closes at dusk")
        search = await world.search()

        said = await search.ainvoke({"ids": ["not-an-id", f"[memory:{real}]"]})

        assert said.splitlines()[0].startswith("(Not ids, so not looked up: not-an-id.")
        assert "the harbour closes at dusk" in _line(said, real)

    async def test_a_long_file_is_marked_cut_and_names_what_reads_it_whole(self, world: _World) -> None:
        parent = await world.memory("notes on the harbour")
        content_id = await world.file(parent, "harbour " + "x" * 900, title="Tide tables")
        short_id = await world.file(parent, "harbour, briefly", title="Short")
        media_id = await world.conn.fetchval("SELECT media_id FROM media_content WHERE content_id = $1", content_id)
        short_media = await world.conn.fetchval("SELECT media_id FROM media_content WHERE content_id = $1", short_id)
        search = await world.search(file_reader="media_read_text")

        said = await search.ainvoke({"ids": [str(content_id), str(short_id)]})

        # Each file is listed once, by its media id.
        cut = next(line for line in said.splitlines() if f"[media:{media_id}]" in line)
        assert (
            f"(cut: the first 500 of 908 characters; media_read_text reads it whole by its media id {media_id})" in cut
        )
        whole = next(line for line in said.splitlines() if f"[media:{short_media}]" in line)
        assert whole.endswith("harbour, briefly") and "cut" not in whole


# --------------------------------------------------------------------------
# memory_add: what it replaced, and what it left standing
# --------------------------------------------------------------------------


class TestAddSaysWhatItDidToTheOthers:
    async def test_a_replaced_near_duplicate_is_quoted_with_its_date(self, world: _World) -> None:
        old = await world.memory("likes green tea", at=datetime(2026, 9, 1, 15, tzinfo=UTC))
        add = await world.add(tz=_LA)

        said = await add.ainvoke({"content": "likes black tea now"})

        assert f"It replaces [memory:{old}] (100% the same, written Sep 1, 2026 at 8:00 AM PDT)" in said
        assert 'which said "likes green tea" and stays readable by its id.' in said
        assert (await world.row(old))["superseded_by"] is not None

    async def test_replace_false_writes_beside_and_says_both_stand(self, world: _World) -> None:
        old = await world.memory("likes green tea")
        add = await world.add()

        said = await add.ainvoke({"content": "likes black tea too", "replace": False})

        assert f'[memory:{old}] (100% the same) still stands beside it and says "likes green tea".' in said
        assert (await world.row(old))["superseded_by"] is None and await world.count() == 2

    async def test_a_permanent_near_twin_is_left_standing_and_the_result_says_so(self, world: _World) -> None:
        kept = await world.memory("the surname is Brannigan", permanent=True)
        add = await world.add()

        said = await add.ainvoke({"content": "the surname is Brennan"})

        assert f"[memory:{kept}] (permanent, 100% the same) still stands beside it" in said
        assert 'says "the surname is Brannigan". Both are now in your memory.' in said
        assert said.endswith(PERMANENT_IS_THE_PERSONS)
        assert (await world.row(kept))["superseded_by"] is None

    async def test_an_unrelated_memory_is_stored_plainly(self, world: _World) -> None:
        await world.memory("likes green tea")
        add = await world.add()
        said = await add.ainvoke({"content": "the garden faces south"})
        assert said.startswith("Stored as [memory:") and said.endswith("the garden faces south.")

    async def test_the_type_defaults_to_fact(self, world: _World) -> None:
        add = await world.add()
        said = await add.ainvoke({"content": "the garden faces south"})
        new_id = uuid.UUID(said.split("[memory:")[1].split("]")[0])
        assert (await world.row(new_id))["type_memory"] == "fact"


class TestACorrectionIsLinkedToWhatItCorrects:
    async def test_a_named_memory_is_replaced_whatever_its_wording(self, world: _World) -> None:
        # different subjects, so the near-duplicate check would never have found it
        old = await world.memory("the sleeve was his choice")
        add = await world.add()

        said = await add.ainvoke({"content": "the choice of cover was mine", "corrects": [f"[memory:{old}]"]})

        row = await world.row(old)
        assert row["superseded_by"] is not None and row["content"] == "the sleeve was his choice"
        assert f"It replaces [memory:{old}]" in said and 'which said "the sleeve was his choice"' in said
        assert (
            await world.conn.fetchval("SELECT rationale FROM memory_consolidations WHERE source_memory_id = $1", old)
            == "corrected by memory_add"
        )

    async def test_a_permanent_memory_is_corrected_beside_it_and_linked(self, world: _World) -> None:
        kept = await world.memory("the sleeve was his choice", permanent=True)
        add = await world.add()

        said = await add.ainvoke({"content": "the sleeve choice was mine", "corrects": [str(kept)]})

        row = await world.row(kept)
        assert (row["superseded_by"], row["evergreen"]) == (None, True), "a permanent memory was replaced"
        assert f"[memory:{kept}] is permanent, so it still stands" in said and PERMANENT_IS_THE_PERSONS in said
        new_id = uuid.UUID(said.split("[memory:")[1].split("]")[0])
        assert (
            await world.conn.fetchval(
                "SELECT count(*) FROM memory_consolidations WHERE source_memory_id = $1 AND consolidated_memory_id = $2",
                kept,
                new_id,
            )
            == 1
        )

    async def test_with_the_persons_word_a_permanent_memory_is_replaced_and_its_replacement_is_permanent(
        self, world: _World
    ) -> None:
        kept = await world.memory("the sleeve was his choice", permanent=True)
        asked: list[uuid.UUID] = []

        async def _the_person_asked(memory_id: uuid.UUID) -> bool:
            asked.append(memory_id)
            return True

        add = await world.add(person_asked=_the_person_asked)

        said = await add.ainvoke({"content": "the sleeve choice was mine", "corrects": [str(kept)]})

        row = await world.row(kept)
        assert asked == [kept] and row["superseded_by"] is not None
        assert (await world.row(row["superseded_by"]))["evergreen"] is True
        assert "Kept permanently." in said and "still stands" not in said

    async def test_a_resolver_that_fails_is_a_no(self, world: _World) -> None:
        kept = await world.memory("the sleeve was his choice", permanent=True)

        async def _broken(memory_id: uuid.UUID) -> bool:
            raise RuntimeError("the turn's record could not be read")

        add = await world.add(person_asked=_broken)

        said = await add.ainvoke({"content": "the sleeve choice was mine", "corrects": [str(kept)]})

        assert (await world.row(kept))["superseded_by"] is None and "is permanent, so it still stands" in said

    async def test_a_retired_target_is_refused_and_nothing_is_stored(self, world: _World) -> None:
        gone = await world.memory("the sleeve was his choice")
        assert await retract(world.memories, agent_id=world.agent_id, memory_id=gone, reason="wrong")
        add = await world.add()

        said = await add.ainvoke({"content": "the sleeve choice was mine", "corrects": [str(gone)]})

        assert said.startswith("[NO CHANGE] memory_add:") and "was retired" in said and "Nothing was stored" in said
        assert await world.count() == 1

    async def test_an_already_replaced_target_is_refused_and_names_its_replacement(self, world: _World) -> None:
        old = await world.memory("the sleeve was his choice")
        new = await world.replace(old, "the sleeve choice was mine")
        add = await world.add()

        said = await add.ainvoke({"content": "the sleeve choice was ours", "corrects": [str(old)]})

        assert said.startswith("[NO CHANGE] memory_add:") and f"already replaced by [memory:{new}]" in said
        assert await world.count() == 2

    async def test_another_persons_memory_is_refused_as_not_there(self, world: _World) -> None:
        theirs = await world.memory("the sleeve was his choice", user_id=uuid.uuid4())
        add = await world.add()

        said = await add.ainvoke({"content": "the sleeve choice was mine", "corrects": [str(theirs)]})

        assert said.startswith("[NO CHANGE] memory_add:") and "that is yours" in said
        assert (await world.row(theirs))["superseded_by"] is None and await world.count() == 1


async def _nobody() -> bool:
    return False


async def _somebody() -> bool:
    return True


def _held_says(said: str, old: uuid.UUID, text: str) -> None:
    """The result of a replacement held back: nothing stored, the memory named and quoted, and when it can change."""
    assert said.startswith("[NO CHANGE] memory_add: Nothing was stored. It would replace "), said
    assert f"[memory:{old}]" in said and f'which says "{text}"' in said
    assert said.endswith(CHANGE_NEEDS_THE_PERSON)


class TestWithNobodyPresentNothingIsReplaced:
    """A turn nobody is present for (a wake fire, an injected turn) replaces no memory; the person's turn still does."""

    async def _same_event(self, world: _World, *, present: Any) -> tuple[Any, uuid.UUID]:
        turn = uuid.uuid4()
        old = await world.memory("the harbour wall was rebuilt", source=turn)

        async def _near() -> list[uuid.UUID]:
            return [turn]

        # the near-duplicate check is off, so only the same-event rule can find it
        add = await world.add(
            similarity_dedup_threshold=1.5, near_message_ids=_near, same_event_threshold=0.5, person_present=present
        )
        return add, old

    async def test_a_near_duplicate_is_not_replaced_and_nothing_is_stored(self, world: _World) -> None:
        old = await world.memory("likes green tea")
        add = await world.add(person_present=_nobody)

        said = await add.ainvoke({"content": "likes black tea now"})

        _held_says(said, old, "likes green tea")
        assert (await world.row(old))["superseded_by"] is None and await world.count() == 1

    async def test_a_correction_is_not_made_and_nothing_is_stored(self, world: _World) -> None:
        old = await world.memory("the sleeve was his choice")
        add = await world.add(person_present=_nobody)

        said = await add.ainvoke({"content": "the choice of cover was mine", "corrects": [str(old)]})

        _held_says(said, old, "the sleeve was his choice")
        assert (await world.row(old))["superseded_by"] is None and await world.count() == 1
        assert await world.conn.fetchval("SELECT count(*) FROM memory_consolidations") == 0

    async def test_the_same_event_is_not_replaced_and_nothing_is_stored(self, world: _World) -> None:
        add, old = await self._same_event(world, present=_nobody)

        said = await add.ainvoke({"content": "the harbour wall is new"})

        _held_says(said, old, "the harbour wall was rebuilt")
        assert (await world.row(old))["superseded_by"] is None and await world.count() == 1

    async def test_a_resolver_that_fails_is_nobody(self, world: _World) -> None:
        old = await world.memory("likes green tea")

        async def _broken() -> bool:
            raise RuntimeError("the turn's door could not be read")

        add = await world.add(person_present=_broken)

        said = await add.ainvoke({"content": "likes black tea now"})

        _held_says(said, old, "likes green tea")
        assert (await world.row(old))["superseded_by"] is None

    # -- what must still get through -------------------------------------------

    async def test_a_new_memory_that_replaces_nothing_is_still_stored(self, world: _World) -> None:
        await world.memory("likes green tea")
        add, _old = await self._same_event(world, present=_nobody)

        said = await add.ainvoke({"content": "the garden faces south"})

        assert said.startswith("Stored as [memory:") and said.endswith("the garden faces south."), said
        assert await world.count() == 3

    async def test_a_correction_beside_a_permanent_memory_is_still_written(self, world: _World) -> None:
        kept = await world.memory("the sleeve was his choice", permanent=True)
        add = await world.add(person_present=_nobody)

        said = await add.ainvoke({"content": "the sleeve choice was mine", "corrects": [str(kept)]})

        assert said.startswith("Stored as [memory:") and f"[memory:{kept}] is permanent, so it still stands" in said
        assert (await world.row(kept))["superseded_by"] is None and await world.count() == 2

    async def test_a_correction_naming_one_it_would_replace_stores_nothing(self, world: _World) -> None:
        kept = await world.memory("the sleeve was his choice", permanent=True)
        plain = await world.memory("the cover was blue")
        add = await world.add(person_present=_nobody)

        said = await add.ainvoke({"content": "the sleeve choice was mine", "corrects": [str(kept), str(plain)]})

        _held_says(said, plain, "the cover was blue")
        assert f"[memory:{kept}]" not in said and await world.count() == 2

    async def test_beside_a_near_twin_with_replace_false_is_still_stored(self, world: _World) -> None:
        old = await world.memory("likes green tea")
        add = await world.add(person_present=_nobody)

        said = await add.ainvoke({"content": "likes black tea too", "replace": False})

        assert said.startswith("Stored as [memory:") and "still stands beside it" in said
        assert (await world.row(old))["superseded_by"] is None and await world.count() == 2

    async def test_with_the_person_present_a_near_duplicate_is_replaced(self, world: _World) -> None:
        old = await world.memory("likes green tea")
        add = await world.add(person_present=_somebody)

        said = await add.ainvoke({"content": "likes black tea now"})

        assert said.startswith("Stored as [memory:") and f"It replaces [memory:{old}]" in said
        assert (await world.row(old))["superseded_by"] is not None

    async def test_with_the_person_present_a_correction_is_made(self, world: _World) -> None:
        old = await world.memory("the sleeve was his choice")
        add = await world.add(person_present=_somebody)

        said = await add.ainvoke({"content": "the choice of cover was mine", "corrects": [str(old)]})

        assert f"It replaces [memory:{old}]" in said
        assert (await world.row(old))["superseded_by"] is not None

    async def test_with_the_person_present_the_same_event_is_replaced(self, world: _World) -> None:
        add, old = await self._same_event(world, present=_somebody)

        said = await add.ainvoke({"content": "the harbour wall is new"})

        assert said.startswith("Stored as [memory:") and f"It replaces [memory:{old}]" in said
        assert (await world.row(old))["superseded_by"] is not None

    async def test_the_description_says_replacing_needs_the_person(self, world: _World) -> None:
        add = await world.add()
        assert "Replacing or correcting one needs your person in the conversation." in add.description


# --------------------------------------------------------------------------
# memory_update: a replacement, never an overwrite
# --------------------------------------------------------------------------


class TestUpdateWritesAReplacement:
    async def test_the_old_text_is_kept_and_the_result_shows_both(self, world: _World) -> None:
        old = await world.memory("the harbour closes at noon", kind="fact", salience=0.8)
        update = await world.update()

        said = await update.ainvoke({"memory_id": str(old), "content": "the harbour closes at dusk"})

        row = await world.row(old)
        assert row["content"] == "the harbour closes at noon" and row["superseded_by"] is not None
        new = await world.row(row["superseded_by"])
        assert (new["content"], new["type_memory"], float(new["salience"])) == (
            "the harbour closes at dusk",
            "fact",
            0.8,
        )
        assert said.startswith(f"Updated: [memory:{row['superseded_by']}] replaces [memory:{old}].")
        assert 'It said "the harbour closes at noon" and now says "the harbour closes at dusk".' in said
        assert (
            await world.conn.fetchval("SELECT rationale FROM memory_consolidations WHERE source_memory_id = $1", old)
            == "changed by memory_update"
        )

    async def test_a_name_moves_to_the_replacement(self, world: _World) -> None:
        old = await world.memory("the harbour closes at noon", alias="harbour-hours")
        update = await world.update()

        said = await update.ainvoke({"memory_id": str(old), "content": "the harbour closes at dusk"})

        row = await world.row(old)
        assert row["alias"] is None and (await world.row(row["superseded_by"]))["alias"] == "harbour-hours"
        assert "The name 'harbour-hours' now opens the new one." in said
        search = await world.search()
        assert "the harbour closes at dusk" in await search.ainvoke({"alias": "harbour-hours", "query": ""})

    async def test_only_the_type_changes_and_the_text_is_carried_over(self, world: _World) -> None:
        old = await world.memory("the harbour closes at dusk", kind="preference")
        update = await world.update()

        said = await update.ainvoke({"memory_id": str(old), "memory_type": "fact"})

        new = await world.row((await world.row(old))["superseded_by"])
        assert (new["content"], new["type_memory"]) == ("the harbour closes at dusk", "fact")
        assert "Its type was preference and is now fact." in said and "now says" not in said

    async def test_a_name_taken_by_another_memory_is_refused_and_the_old_name_stays(self, world: _World) -> None:
        await world.memory("the garden faces south", alias="garden")
        old = await world.memory("the harbour closes at noon", alias="harbour-hours")
        update = await world.update()

        said = await update.ainvoke({"memory_id": str(old), "alias": "garden"})

        row = await world.row(old)
        assert said.startswith("[TOOL ERROR] memory_update: alias failed") and "Nothing was changed" in said
        assert (row["alias"], row["superseded_by"]) == ("harbour-hours", None)

    async def test_the_same_text_changes_nothing(self, world: _World) -> None:
        old = await world.memory("the harbour closes at dusk")
        update = await world.update()
        said = await update.ainvoke({"memory_id": str(old), "content": "the harbour closes at dusk"})
        assert said.startswith("[NO CHANGE] memory_update:") and await world.count() == 1

    async def test_a_permanent_memory_is_not_changed_and_the_result_says_who_can(self, world: _World) -> None:
        kept = await world.memory("the surname is Brannigan", permanent=True)
        update = await world.update()

        said = await update.ainvoke({"memory_id": str(kept), "content": "the surname is Brennan"})

        assert said.startswith(f"[NO CHANGE] memory_update: [memory:{kept}] is permanent, so it was not changed.")
        assert 'It says "the surname is Brannigan".' in said and PERMANENT_IS_THE_PERSONS in said
        assert f"corrects=['{kept}']" in said
        assert (await world.row(kept))["superseded_by"] is None and await world.count() == 1

    async def test_with_the_persons_word_a_permanent_memory_is_replaced_and_stays_permanent(
        self, world: _World
    ) -> None:
        kept = await world.memory("the surname is Brannigan", permanent=True)

        async def _the_person_asked(memory_id: uuid.UUID) -> bool:
            return memory_id == kept

        update = await world.update(person_asked=_the_person_asked)

        said = await update.ainvoke({"memory_id": str(kept), "content": "the surname is Brennan"})

        row = await world.row(kept)
        assert row["superseded_by"] is not None and (await world.row(row["superseded_by"]))["evergreen"] is True
        assert said.endswith("It is still permanent.")

    async def test_a_retired_or_replaced_memory_is_not_changed(self, world: _World) -> None:
        retired = await world.memory("the harbour closes at noon")
        assert await retract(world.memories, agent_id=world.agent_id, memory_id=retired, reason="wrong")
        old = await world.memory("the sleeve was his choice")
        new = await world.replace(old, "the sleeve choice was mine")
        update = await world.update()
        before = await world.count()

        was_retired = await update.ainvoke({"memory_id": str(retired), "content": "the harbour closes at dusk"})
        was_replaced = await update.ainvoke({"memory_id": str(old), "content": "the sleeve was ours"})

        assert was_retired.startswith("[NO CHANGE] memory_update:") and "memory_restore" in was_retired
        assert f"already replaced by [memory:{new}]" in was_replaced and "Update that one instead." in was_replaced
        assert await world.count() == before

    async def test_another_persons_memory_is_not_changed(self, world: _World) -> None:
        theirs = await world.memory("the harbour closes at noon", user_id=uuid.uuid4())
        update = await world.update()
        said = await update.ainvoke({"memory_id": str(theirs), "content": "the harbour closes at dusk"})
        assert said.startswith("[NO CHANGE] memory_update:") and "that is yours" in said
        assert (await world.row(theirs))["superseded_by"] is None


# --------------------------------------------------------------------------
# memory_forget and memory_restore: retired, kept, and brought back
# --------------------------------------------------------------------------


class TestForgetRetiresAndRestoreUndoesIt:
    async def test_a_forgotten_memory_is_kept_leaves_search_and_opens_by_id(self, world: _World) -> None:
        wrong = await world.memory("the harbour closes at noon", salience=0.7)
        right = await world.memory("the harbour closes at dusk")
        forget, search, recall = world.forget(), await world.search(), await world.recall()

        said = await forget.ainvoke({"memory_id": f"[memory:{wrong}]", "reason": "it closes   at dusk"})

        row = await world.row(wrong)
        assert row["content"] == "the harbour closes at noon", "the row's text was changed"
        assert float(row["salience"]) == 0.0
        assert said.startswith(f'Retired [memory:{wrong}], which said "the harbour closes at noon".')
        assert "Reason kept: it closes at dusk." in said and "memory_restore brings it back" in said
        found = await search.ainvoke({"query": "harbour"})
        assert str(wrong) not in found and str(right) in found
        opened = await recall.ainvoke({"memory_id": str(wrong)})
        assert opened.splitlines()[0].endswith("(retired, no longer holds: it closes at dusk)")
        assert "the harbour closes at noon" in opened

    async def test_restore_brings_it_back_to_search_with_the_salience_it_had(self, world: _World) -> None:
        memory = await world.memory("the harbour closes at dusk", salience=0.7)
        forget, restore_tool, search = world.forget(), world.restore(), await world.search()
        await forget.ainvoke({"memory_id": str(memory), "reason": "not sure any more"})
        assert str(memory) not in await search.ainvoke({"query": "harbour"})

        said = await restore_tool.ainvoke({"memory_id": str(memory)})

        row = await world.row(memory)
        assert float(row["salience"]) == 0.7
        assert json.loads(row["tags"]) == [], "a retraction tag was left behind"
        assert said.startswith(f'Restored [memory:{memory}], which says "the harbour closes at dusk".')
        assert "It had been retired because: not sure any more." in said and said.endswith("Search finds it again.")
        assert str(memory) in await search.ainvoke({"query": "harbour"})

    async def test_a_memory_retired_before_its_salience_was_recorded_restores_to_the_seed(self, world: _World) -> None:
        memory = await world.memory("the harbour closes at dusk", salience=0.0)
        await world.conn.execute(
            "UPDATE memories SET tags = $2::jsonb WHERE memory_id = $1",
            memory,
            json.dumps(["keep-me", "retracted", "retracted_because:old run"]),
        )
        restore_tool = world.restore(salience_seed=0.4)

        await restore_tool.ainvoke({"memory_id": str(memory)})

        row = await world.row(memory)
        assert float(row["salience"]) == 0.4 and json.loads(row["tags"]) == ["keep-me"]

    async def test_a_reason_is_required(self, world: _World) -> None:
        memory = await world.memory("the harbour closes at dusk")
        forget = world.forget()
        said = await forget.ainvoke({"memory_id": str(memory), "reason": "   "})
        assert said.startswith("[TOOL ERROR] memory_forget: reason failed")
        assert (await world.row(memory))["tags"] is None

    async def test_a_permanent_memory_is_not_retired_and_the_result_says_who_can(self, world: _World) -> None:
        kept = await world.memory("the surname is Brannigan", permanent=True)
        forget = world.forget()

        said = await forget.ainvoke({"memory_id": str(kept), "reason": "it is Brennan"})

        assert said.startswith(f"[NO CHANGE] memory_forget: [memory:{kept}] is permanent, so it was not retired.")
        assert PERMANENT_IS_THE_PERSONS in said and 'It says "the surname is Brannigan".' in said
        assert (await world.row(kept))["tags"] is None

    async def test_with_the_persons_word_a_permanent_memory_is_retired_and_can_be_restored(self, world: _World) -> None:
        kept = await world.memory("the surname is Brannigan", permanent=True, salience=0.9)

        async def _the_person_asked(memory_id: uuid.UUID) -> bool:
            return True

        forget, search = world.forget(person_asked=_the_person_asked), await world.search()

        said = await forget.ainvoke({"memory_id": str(kept), "reason": "it is Brennan"})

        assert said.startswith(f"Retired [memory:{kept}]")
        assert str(kept) not in await search.ainvoke({"query": "surname", "mode": "fuzzy"})
        assert float((await world.row(kept))["salience"]) == 0.9, "a permanent memory's salience was touched"
        await world.restore().ainvoke({"memory_id": str(kept)})
        assert str(kept) in await search.ainvoke({"query": "surname", "mode": "fuzzy"})

    async def test_retiring_twice_and_retiring_a_replaced_memory_change_nothing(self, world: _World) -> None:
        memory = await world.memory("the harbour closes at noon")
        old = await world.memory("the sleeve was his choice")
        new = await world.replace(old, "the sleeve choice was mine")
        forget = world.forget()
        await forget.ainvoke({"memory_id": str(memory), "reason": "wrong hour"})

        again = await forget.ainvoke({"memory_id": str(memory), "reason": "still wrong"})
        replaced = await forget.ainvoke({"memory_id": str(old), "reason": "wrong"})

        assert again == f"[NO CHANGE] memory_forget: [memory:{memory}] was already retired (wrong hour)."
        assert f"already replaced by [memory:{new}]" in replaced and (await world.row(old))["tags"] is None
        assert "retracted_because:wrong hour" in json.loads((await world.row(memory))["tags"])

    async def test_another_persons_memory_is_neither_retired_nor_restored(self, world: _World) -> None:
        theirs = await world.memory("the harbour closes at noon", user_id=uuid.uuid4())
        said = await world.forget().ainvoke({"memory_id": str(theirs), "reason": "wrong"})
        assert said.startswith("[NO CHANGE] memory_forget:") and "that is yours" in said
        assert (await world.row(theirs))["tags"] is None
        assert await retract(world.memories, agent_id=world.agent_id, memory_id=theirs, reason="by their own agent")
        back = await world.restore().ainvoke({"memory_id": str(theirs)})
        assert back.startswith("[NO CHANGE] memory_restore:") and "retracted" in json.loads(
            (await world.row(theirs))["tags"]
        )

    async def test_restoring_a_memory_that_is_not_retired_changes_nothing(self, world: _World) -> None:
        memory = await world.memory("the harbour closes at dusk", salience=0.7)
        said = await world.restore().ainvoke({"memory_id": str(memory)})
        assert said.startswith("[NO CHANGE] memory_restore:") and "show='retired'" in said
        assert float((await world.row(memory))["salience"]) == 0.7

    async def test_the_moves_refuse_a_permanent_memory_unless_told(self, world: _World) -> None:
        kept = await world.memory("the surname is Brannigan", permanent=True)
        assert not await retract(world.memories, agent_id=world.agent_id, memory_id=kept, reason="no")
        assert (await world.row(kept))["tags"] is None
        assert await retract(
            world.memories, agent_id=world.agent_id, memory_id=kept, reason="the person said", allow_permanent=True
        )
        assert not await retract(world.memories, agent_id=world.agent_id, memory_id=kept, reason="again")
        assert await restore(world.memories, agent_id=world.agent_id, memory_id=kept)
        assert not await restore(world.memories, agent_id=world.agent_id, memory_id=kept)


# --------------------------------------------------------------------------
# memory_list: browsing with no query
# --------------------------------------------------------------------------


class TestListBrowsesWithoutAQuery:
    async def test_newest_first_with_the_count_and_the_next_offset(self, world: _World) -> None:
        ids = [await world.memory(f"note {i}", age_days=i) for i in range(5)]
        listing = world.listing()

        first = await listing.ainvoke({"limit": 2})
        rest = await listing.ainvoke({"limit": 3, "offset": 2})

        assert first.splitlines()[0] == "5 memories, newest first. Showing 1-2; offset=2 shows the next."
        assert [str(i) in first for i in ids] == [True, True, False, False, False]
        assert first.index(str(ids[0])) < first.index(str(ids[1]))
        assert rest.splitlines()[0] == "5 memories, newest first. Showing 3-5; that is all of them."

    async def test_the_default_list_is_what_still_holds_and_the_others_can_be_asked_for(self, world: _World) -> None:
        current = await world.memory("the harbour closes at dusk")
        retired = await world.memory("the harbour closes at noon")
        old = await world.memory("the sleeve was his choice")
        new = await world.replace(old, "the sleeve choice was mine")
        await world.forget().ainvoke({"memory_id": str(retired), "reason": "wrong hour"})
        listing = world.listing()

        by_default = await listing.ainvoke({})
        only_retired = await listing.ainvoke({"show": "retired"})
        only_replaced = await listing.ainvoke({"show": "replaced"})
        everything = await listing.ainvoke({"show": "all"})

        assert {m for m in (current, retired, old, new) if str(m) in by_default.split("\n", 1)[1]} == {current, new}
        assert only_retired.splitlines()[0].startswith("1 retired memories")
        assert "(retired, no longer holds: wrong hour)" in _line(only_retired, retired)
        assert f"(replaced by [memory:{new}]" in _line(only_replaced, old)
        assert everything.splitlines()[0].startswith("4 memories of any state")

    async def test_filters_by_date_type_permanence_and_name(self, world: _World) -> None:
        zone = ZoneInfo(_LA)
        on_the_9th = await world.memory("a decision", kind="decision", at=datetime(2026, 10, 9, 22, 30, tzinfo=zone))
        kept = await world.memory("who they are", permanent=True, at=datetime(2026, 10, 1, 9, tzinfo=zone))
        named = await world.memory("a named note", alias="note", at=datetime(2026, 10, 2, 9, tzinfo=zone))
        listing = world.listing(tz=_LA)

        def _shown(said: str) -> set[uuid.UUID]:
            return {m for m in (on_the_9th, kept, named) if str(m) in said}

        assert _shown(await listing.ainvoke({"date_after": "2026-10-09", "date_before": "2026-10-09"})) == {on_the_9th}
        assert _shown(await listing.ainvoke({"memory_type": "decision"})) == {on_the_9th}
        assert _shown(await listing.ainvoke({"permanent": True})) == {kept}
        assert _shown(await listing.ainvoke({"permanent": False})) == {on_the_9th, named}
        assert _shown(await listing.ainvoke({"has_alias": True})) == {named}
        assert "[Oct 9, 2026 at 10:30 PM PDT]" in await listing.ainvoke({"memory_type": "decision"})

    async def test_another_persons_memories_are_not_listed(self, world: _World) -> None:
        mine = await world.memory("mine")
        theirs = await world.memory("theirs", user_id=uuid.uuid4())
        said = await world.listing().ainvoke({})
        assert str(mine) in said and str(theirs) not in said and said.startswith("1 memories")

    async def test_a_long_memory_is_cut_and_marked(self, world: _World) -> None:
        long = await world.memory("the garden " + "y" * 600)
        said = await world.listing().ainvoke({})
        assert f"(cut: the first 300 of 611 characters; memory_recall reads [memory:{long}] whole)" in said

    async def test_nothing_matching_says_so(self, world: _World) -> None:
        await world.memory("a note")
        assert await world.listing().ainvoke({"show": "retired"}) == "0 retired memories match."
        assert await world.listing().ainvoke({"offset": 5}) == (
            "1 memories match, and offset=5 is past the last of them."
        )


# --------------------------------------------------------------------------
# memory_recall, chunk_search and chunk_recall: the facts beside the text
# --------------------------------------------------------------------------


class TestARecallSaysWhatItIsShowing:
    async def test_the_header_says_type_date_name_and_state(self, world: _World) -> None:
        memory = await world.memory(
            "the surname is Brennan",
            permanent=True,
            alias="family-name",
            at=datetime(2026, 10, 9, 15, tzinfo=UTC),
        )
        recall = await world.recall(tz=_LA)

        said = await recall.ainvoke({"alias": "family-name"})

        assert said.splitlines()[0] == (
            f"[memory:{memory}] [fact] written Oct 9, 2026 at 8:00 AM PDT (alias: family-name) (permanent)"
        )
        assert said.splitlines()[1] == "the surname is Brennan" and said.endswith("(no parts)")

    async def test_a_replaced_memory_opens_with_its_replacement_beside_it(self, world: _World) -> None:
        old = await world.memory("the surname is Brannigan")
        new = await world.replace(old, "the surname is Brennan")
        recall = await world.recall()

        of_the_old = await recall.ainvoke({"memory_id": str(old)})
        of_the_new = await recall.ainvoke({"memory_id": str(new)})

        assert f"(replaced by [memory:{new}]" in of_the_old.splitlines()[0]
        assert of_the_old.splitlines()[1] == f'Replaced by [memory:{new}]: "the surname is Brennan"'
        assert "the surname is Brannigan" in of_the_old
        assert f"It replaced or corrects: [memory:{old}]." in of_the_new

    async def test_the_parts_say_which_of_how_many_and_how_to_read_on(self, world: _World) -> None:
        memory = await world.memory("a long talk about the harbour")
        parts = [
            await world.chunk(memory, f"harbour part {i} " + "z" * (300 if i == 1 else 0), index=i) for i in range(7)
        ]
        recall = await world.recall()

        first = await recall.ainvoke({"memory_id": str(memory), "limit": 3})
        middle = await recall.ainvoke({"memory_id": str(memory), "limit": 3, "chunk_id_after": str(parts[2])})
        last = await recall.ainvoke({"memory_id": str(memory), "limit": 3, "chunk_id_after": str(parts[5])})
        picked = await recall.ainvoke({"memory_id": str(memory), "chunk_indexes": [1, 6]})

        assert f"Parts 0-2 of 7. Next: chunk_id_after={parts[2]}." in first
        assert f"Parts 3-5 of 7. Next: chunk_id_after={parts[5]}. Earlier: chunk_id_before={parts[3]}." in middle
        assert f"Part 6 of 7. Earlier: chunk_id_before={parts[6]}." in last
        assert "2 of 7 parts, the ones asked for." in picked
        assert f"(cut: the first 200 of 315 characters; chunk_recall reads [chunk:{parts[1]}] whole)" in first
        whole = next(line for line in first.splitlines() if line.strip().startswith("harbour part 0"))
        assert whole.strip() == "harbour part 0" and f"[chunk:{parts[0]}] (part 0)" in first

    async def test_parts_matching_a_text_say_their_place(self, world: _World) -> None:
        memory = await world.memory("a long talk")
        await world.chunk(memory, "about the garden", index=0)
        tea = await world.chunk(memory, "about tea", index=1)
        recall = await world.recall()

        said = await recall.ainvoke({"memory_id": str(memory), "chunk_query": "tea"})

        assert "1 of 2 parts, the ones matching the text." in said and f"[chunk:{tea}] (part 1)" in said

    async def test_another_persons_memory_does_not_open(self, world: _World) -> None:
        theirs = await world.memory("theirs", user_id=uuid.uuid4())
        said = await (await world.recall()).ainvoke({"memory_id": str(theirs)})
        assert said == "[TOOL ERROR] No memory with that id that you can read."


class TestAPassageSaysWhereItIsFrom:
    async def test_a_hit_is_dated_sourced_and_marked_when_cut(self, world: _World) -> None:
        memory = await world.memory("notes on tea")
        await world.file(memory, "the tea ledger", title="Tea ledger")
        when = datetime(2026, 10, 9, 15, tzinfo=UTC)
        long = await world.chunk(memory, "tea " + "q" * 400, index=0, at=when)
        short = await world.chunk(memory, "tea, briefly", index=1, at=when)
        search = await world.chunk_search(tz=_LA)

        said = await search.ainvoke({"query": "tea"})

        assert f'[chunk:{long}] (from [memory:{memory}], "Tea ledger", written Oct 9, 2026 at 8:00 AM PDT)' in said
        assert f"(cut: the first 200 of 404 characters; chunk_recall reads [chunk:{long}] whole)" in said
        block = said.split(f"[chunk:{short}]")[1].split("\n\n")[0]
        assert block.endswith("tea, briefly") and "cut" not in block

    async def test_more_matching_than_shown_is_said_first(self, world: _World) -> None:
        memory = await world.memory("notes on tea")
        for i in range(4):
            await world.chunk(memory, f"tea passage {i}", index=i)
        search = await world.chunk_search()

        few = await search.ainvoke({"query": "tea", "limit": 2})
        all_of_them = await search.ainvoke({"query": "tea", "limit": 10})

        assert few.splitlines()[0] == (
            "(More passages match than the 2 shown: raise limit (up to 20), or narrow the query.)"
        )
        assert all_of_them.startswith("[chunk:") and all_of_them.count("[chunk:") == 4

    async def test_one_passage_read_whole_says_its_place_date_and_memory(self, world: _World) -> None:
        memory = await world.memory("notes on tea")
        await world.file(memory, "the tea ledger", title="Tea ledger")
        for i in range(3):
            last = await world.chunk(
                memory, f"tea passage {i} " + "w" * 500, index=i, at=datetime(2026, 10, 9, 15, tzinfo=UTC)
            )
        recall = await world.chunk_recall(tz=_LA)

        said = await recall.ainvoke({"chunk_id": f"[chunk:{last}]"})

        header, _, body = said.partition("\n\n")
        assert header == (
            f'[chunk:{last}] part 2 of 3 of [memory:{memory}], written Oct 9, 2026 at 8:00 AM PDT, from "Tea ledger".'
        )
        assert body == "tea passage 2 " + "w" * 500, "the passage was not returned whole"

    async def test_another_persons_passage_does_not_open(self, world: _World) -> None:
        memory = await world.memory("notes on tea")
        part = await world.chunk(memory, "tea", index=0)
        await world.conn.execute("UPDATE memory_chunks SET user_id = $2 WHERE chunk_id = $1", part, uuid.uuid4())
        said = await (await world.chunk_recall()).ainvoke({"chunk_id": str(part)})
        assert said.startswith("[TOOL ERROR] No passage with that id that you can read.")


class TestKeepSaysItIsOneWay:
    async def test_the_description_says_who_can_take_it_back(self, world: _World) -> None:
        assert "only your person can change, retire or un-keep a permanent memory" in world.keep().description


class TestMarksReadARow:
    async def test_the_marks_of_each_state(self, world: _World) -> None:
        old = await world.memory("the surname is Brannigan", alias="was")
        new = await world.replace(old, "the surname is Brennan")
        assert await retract(world.memories, agent_id=world.agent_id, memory_id=new, reason="neither")

        async def _marks(memory_id: uuid.UUID) -> Any:
            row = await world.memories.find_for_recall(
                memory_id=memory_id, user_id=world.user_id, agent_id=world.agent_id
            )
            assert row is not None
            return marks_of(row)

        of_old, of_new = await _marks(old), await _marks(new)
        assert (of_old.replaced_by, of_old.alias, of_old.current) == (new, "was", False)
        assert (of_new.retired, of_new.retired_because, of_new.replaced_by) == (True, "neither", None)
