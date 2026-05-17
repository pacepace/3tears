"""Tests for memory retrieval -- ranking math, MMR, FTS, formatting.

Collection-parameterised (namespace-task-01 phase 8.5b): the retriever
takes the three memory-package Collections as required constructor
parameters and no longer accepts a raw pool. tests build registry-bound
Collections against a simple stub pool and pass them through.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import (
    MediaContentCollection,
    MemoriesCollection,
    MemoryChunkCollection,
)
from threetears.agent.memory.retrieval import (
    MemoryRetriever,
    _build_fts_query,
    _cosine_sim,
    _format_memory_context,
    _get_display_text,
    _mmr_rerank,
    _normalize_fts_scores,
    _recency_decay,
)
from threetears.agent.memory.types import MemoryConfig
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig


# -- _recency_decay -----------------------------------------------------------


class TestRecencyDecay:
    def test_fresh_item_near_one(self) -> None:
        now = datetime.now(timezone.utc)
        result = _recency_decay(now, half_life_hours=24.0)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_one_half_life_ago(self) -> None:
        t = datetime.now(timezone.utc) - timedelta(hours=24)
        result = _recency_decay(t, half_life_hours=24.0)
        assert result == pytest.approx(math.exp(-1), abs=0.01)

    def test_two_half_lives_ago(self) -> None:
        t = datetime.now(timezone.utc) - timedelta(hours=48)
        result = _recency_decay(t, half_life_hours=24.0)
        assert result == pytest.approx(math.exp(-2), abs=0.01)

    def test_naive_datetime_raises_typeerror(self) -> None:
        # collections-task-05: every datetime in the platform is
        # timezone-aware UTC. Passing a naive value to _recency_decay
        # used to be silently coerced; the defensive wrap is gone now,
        # so the platform-wide convention surfaces as a TypeError on
        # the ``now - created`` subtract.
        t = datetime.now(timezone.utc).replace(tzinfo=None)
        with pytest.raises(TypeError):
            _recency_decay(t, half_life_hours=24.0)

    def test_custom_half_life(self) -> None:
        t = datetime.now(timezone.utc) - timedelta(hours=12)
        result = _recency_decay(t, half_life_hours=12.0)
        assert result == pytest.approx(math.exp(-1), abs=0.01)


# -- _build_fts_query ---------------------------------------------------------


class TestBuildFtsQuery:
    def test_short_text_returns_none(self) -> None:
        assert _build_fts_query("ab") is None

    def test_empty_string_returns_none(self) -> None:
        assert _build_fts_query("") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert _build_fts_query("  ") is None

    def test_normal_text_returned_stripped(self) -> None:
        assert _build_fts_query("  hello world  ") == "hello world"

    def test_long_text_truncated(self) -> None:
        text = "x" * 600
        result = _build_fts_query(text)
        assert result is not None
        assert len(result) == 500

    def test_custom_min_len(self) -> None:
        assert _build_fts_query("ab", min_len=2) == "ab"
        assert _build_fts_query("a", min_len=2) is None

    def test_custom_max_len(self) -> None:
        result = _build_fts_query("abcdefgh", max_len=5)
        assert result == "abcde"


# -- _normalize_fts_scores ----------------------------------------------------


class TestNormalizeFtsScores:
    def test_normalizes_range(self) -> None:
        candidates = [{"fts_rank": 0.0}, {"fts_rank": 5.0}, {"fts_rank": 10.0}]
        _normalize_fts_scores(candidates)
        assert candidates[0]["fts_rank"] == pytest.approx(0.0)
        assert candidates[1]["fts_rank"] == pytest.approx(0.5)
        assert candidates[2]["fts_rank"] == pytest.approx(1.0)

    def test_single_item_normalizes_to_one(self) -> None:
        candidates = [{"fts_rank": 3.5}]
        _normalize_fts_scores(candidates)
        assert candidates[0]["fts_rank"] == pytest.approx(1.0)

    def test_all_zeros_stay_zero(self) -> None:
        candidates = [{"fts_rank": 0.0}, {"fts_rank": 0.0}]
        _normalize_fts_scores(candidates)
        assert all(c["fts_rank"] == 0.0 for c in candidates)

    def test_custom_key(self) -> None:
        candidates = [{"score": 2.0}, {"score": 4.0}]
        _normalize_fts_scores(candidates, key="score")
        assert candidates[0]["score"] == pytest.approx(0.0)
        assert candidates[1]["score"] == pytest.approx(1.0)


# -- _cosine_sim ---------------------------------------------------------------


class TestCosineSim:
    def test_identical_vectors(self) -> None:
        v = [1.0, 2.0, 3.0]
        assert _cosine_sim(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert _cosine_sim([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        assert _cosine_sim([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self) -> None:
        assert _cosine_sim([0.0, 0.0], [1.0, 2.0]) == 0.0
        assert _cosine_sim([1.0, 2.0], [0.0, 0.0]) == 0.0

    def test_both_zero_returns_zero(self) -> None:
        assert _cosine_sim([0.0, 0.0], [0.0, 0.0]) == 0.0


# -- _mmr_rerank ---------------------------------------------------------------


class TestMmrRerank:
    def test_empty_candidates(self) -> None:
        assert _mmr_rerank([1.0, 0.0], [], k=3) == []

    def test_fewer_candidates_than_k(self) -> None:
        candidates = [
            {"embedding": [1.0, 0.0], "hybrid_score": 0.9},
            {"embedding": [0.0, 1.0], "hybrid_score": 0.8},
        ]
        result = _mmr_rerank([1.0, 0.0], candidates, k=5)
        assert len(result) == 2

    def test_selects_diverse_results(self) -> None:
        candidates = [
            {"embedding": [1.0, 0.0], "hybrid_score": 0.95},
            {"embedding": [0.99, 0.1], "hybrid_score": 0.90},
            {"embedding": [0.0, 1.0], "hybrid_score": 0.85},
        ]
        result = _mmr_rerank([1.0, 0.0], candidates, k=2, lambda_mult=0.5)
        scores = {r["hybrid_score"] for r in result}
        assert 0.95 in scores
        assert 0.85 in scores

    def test_uses_similarity_fallback(self) -> None:
        candidates = [
            {"embedding": [1.0, 0.0], "similarity": 0.9},
            {"embedding": [0.0, 1.0], "similarity": 0.8},
            {"embedding": [0.5, 0.5], "similarity": 0.7},
        ]
        result = _mmr_rerank([1.0, 0.0], candidates, k=2)
        assert len(result) == 2


# -- _get_display_text ---------------------------------------------------------


class TestGetDisplayText:
    def test_above_threshold_full_content(self) -> None:
        item = {"content": "full content here", "summary": "short", "hybrid_score": 0.9}
        text, detailed = _get_display_text(item, detail_threshold=0.85)
        assert text == "full content here"
        assert detailed is True

    def test_below_threshold_with_summary(self) -> None:
        item = {"content": "full content", "summary": "short summary", "hybrid_score": 0.5}
        text, detailed = _get_display_text(item, detail_threshold=0.85)
        assert text == "short summary"
        assert detailed is False

    def test_below_threshold_no_summary_short_content(self) -> None:
        item = {"content": "short", "summary": None, "hybrid_score": 0.3}
        text, detailed = _get_display_text(item, detail_threshold=0.85)
        assert text == "short"
        assert detailed is False

    def test_below_threshold_no_summary_long_content_truncated(self) -> None:
        long_content = "x" * 200
        item = {"content": long_content, "summary": None, "hybrid_score": 0.3}
        text, detailed = _get_display_text(item, detail_threshold=0.85)
        assert len(text) == 150
        assert text.endswith("...")
        assert detailed is False

    def test_uses_similarity_fallback_when_no_score_key(self) -> None:
        item = {"content": "full", "similarity": 0.9}
        text, detailed = _get_display_text(item, detail_threshold=0.85)
        assert detailed is True


# -- _format_memory_context ----------------------------------------------------


class TestFormatMemoryContext:
    def test_memories_section(self) -> None:
        memories = [
            {"memory_id": uuid.uuid7(), "content": "likes cats", "summary": None, "hybrid_score": 0.5},
        ]
        result = _format_memory_context(memories, detail_threshold=0.85)
        assert "Things you remember about this user:" in result
        assert "likes cats" in result
        assert "memory_recall" in result

    def test_media_section(self) -> None:
        media = [
            {
                "content_id": uuid.uuid7(),
                "content": "photo desc",
                "summary": None,
                "media_id": str(uuid.uuid7()),
                "hybrid_score": 0.5,
            },
        ]
        result = _format_memory_context([], media_content=media, detail_threshold=0.85)
        assert "Relevant media context:" in result

    def test_chunks_section(self) -> None:
        chunks = [
            {
                "chunk_id": uuid.uuid7(),
                "content": "doc text",
                "summary": "doc summary",
                "media_id": None,
                "title": "My Doc",
                "page_number": 5,
                "heading_context": "Chapter 1",
                "hybrid_score": 0.5,
            },
        ]
        result = _format_memory_context([], memory_chunks=chunks, detail_threshold=0.85)
        assert "Relevant document excerpts:" in result
        assert '"My Doc"' in result
        assert "p.5" in result
        assert '"Chapter 1"' in result
        # v0.7.0 Shard D: chunk headlines carry the recall affordance
        # so the agent knows how to escalate from headline to full
        # content.
        assert "chunk_recall(" in result

    def test_ledger_dedup(self) -> None:
        mid = uuid.uuid7()
        memories = [
            {"memory_id": mid, "content": "should be excluded", "summary": None, "hybrid_score": 0.9},
        ]
        result = _format_memory_context(memories, ledgered_ids={str(mid)}, detail_threshold=0.85)
        assert "should be excluded" not in result

    def test_chunk_dedup_by_media_id(self) -> None:
        shared_media_id = str(uuid.uuid7())
        media = [
            {
                "content_id": uuid.uuid7(),
                "content": "media desc",
                "summary": None,
                "media_id": shared_media_id,
                "hybrid_score": 0.5,
            },
        ]
        chunks = [
            {
                "chunk_id": uuid.uuid7(),
                "content": "chunk from same media",
                "summary": None,
                "media_id": shared_media_id,
                "title": "Doc",
                "page_number": None,
                "heading_context": None,
                "hybrid_score": 0.5,
            },
        ]
        result = _format_memory_context([], media_content=media, memory_chunks=chunks, detail_threshold=0.85)
        assert "chunk from same media" not in result
        assert "media desc" in result

    def test_chunk_pull_not_push_high_score_does_not_render_content(self) -> None:
        """v0.7.0 Shard D D-02: chunk SUMMARIES are pushed, chunk
        CONTENT is pulled. Even when a chunk's hybrid_score is at the
        maximum, the retrieval rendering must NEVER include the chunk's
        ``content`` field in the system prompt -- only ``summary`` +
        ``chunk_id`` + the chunk_recall affordance.

        Pins the architectural invariant against a future refactor
        tempted to "include the content when score is super high."
        """
        sentinel = "VERBATIM-CONTENT-MUST-NOT-LEAK-INTO-SYSTEM-PROMPT"
        chunks = [
            {
                "chunk_id": uuid.uuid7(),
                "content": sentinel,
                "summary": "the chunk's safe-to-surface headline",
                "media_id": None,
                "title": "Big Doc",
                "page_number": 1,
                "heading_context": None,
                # Maximum score. detail_threshold=0.85 default; the
                # 1.0 score would tip the legacy ``_get_display_text``
                # branch into returning ``content``. The chunk branch
                # must NOT use that path.
                "hybrid_score": 1.0,
            },
        ]
        result = _format_memory_context([], memory_chunks=chunks, detail_threshold=0.85)
        # Summary surfaces.
        assert "the chunk's safe-to-surface headline" in result
        # chunk_id + affordance surface.
        assert "[chunk:" in result
        assert "chunk_recall(" in result
        # CRITICAL INVARIANT: content sentinel never appears.
        assert sentinel not in result, (
            "Pull-not-push violation: chunk content leaked into the "
            "rendered memory_context. The retrieval rendering must "
            "only push summaries; the agent pulls full content via "
            "chunk_recall(chunk_id)."
        )

    def test_chunk_surfaced_cap_max_three(self) -> None:
        """The retrieval rendering surfaces at most MAX_SURFACED_CHUNKS=3
        chunk headlines per retrieval. A larger result must NOT inflate
        the system prompt; the agent reaches the rest via chunk_search."""
        chunks = [
            {
                "chunk_id": uuid.uuid7(),
                "content": "c",
                "summary": f"headline-{i}",
                "media_id": None,
                "title": None,
                "page_number": None,
                "heading_context": None,
                "hybrid_score": 0.9 - (i * 0.01),
            }
            for i in range(10)
        ]
        result = _format_memory_context([], memory_chunks=chunks, detail_threshold=0.85)
        # First three headlines present.
        assert "headline-0" in result
        assert "headline-1" in result
        assert "headline-2" in result
        # Fourth+ headlines suppressed.
        assert "headline-3" not in result
        assert "headline-9" not in result

    def test_chunk_summary_truncated_at_max_chars(self) -> None:
        """Pinning MAX_CHUNK_SUMMARY_CHARS=150 so a runaway summary
        (e.g., a chunk whose summary callback wrote the full content)
        can't single-handedly blow the prompt budget."""
        long_summary = "x" * 500
        chunks = [
            {
                "chunk_id": uuid.uuid7(),
                "content": "c",
                "summary": long_summary,
                "media_id": None,
                "title": None,
                "page_number": None,
                "heading_context": None,
                "hybrid_score": 0.5,
            },
        ]
        result = _format_memory_context([], memory_chunks=chunks, detail_threshold=0.85)
        # The ``xxxx...`` truncation marker is present + the full
        # 500-char string is not.
        assert "..." in result
        assert long_summary not in result

    def test_footer_present(self) -> None:
        memories = [{"memory_id": uuid.uuid7(), "content": "x", "summary": None, "hybrid_score": 0.5}]
        result = _format_memory_context(memories, detail_threshold=0.85)
        assert "memory_recall" in result

    def test_empty_returns_empty(self) -> None:
        result = _format_memory_context([])
        assert result == ""


# -- MemoryRetriever end-to-end ------------------------------------------------


class StubEmbeddingProvider:
    """Stub LangChain ``Embeddings`` for testing."""

    def __init__(self, embedding: list[float] | None = None) -> None:
        self._embedding = embedding or [1.0, 0.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embedding for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        _ = text
        return self._embedding

    async def aembed_query(self, text: str) -> list[float]:
        _ = text
        return self._embedding

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embedding for _ in texts]

    @property
    def dimensions(self) -> int:
        return len(self._embedding) if self._embedding else 3


def _make_mock_pool(
    memory_rows: list[dict] | None = None,
    media_rows: list[dict] | None = None,
    chunk_rows: list[dict] | None = None,
) -> AsyncMock:
    """Create a mock pool that returns predetermined rows for each fetch call.

    FTS queries (containing ``ts_rank_cd``) return empty -- only vector rows are mocked.
    """
    pool = AsyncMock()
    mem = memory_rows or []
    med = media_rows or []
    chk = chunk_rows or []

    async def _fetch(sql: str, *args: object) -> list[dict]:
        sql_lower = sql.lower().strip()
        if "ts_rank_cd" in sql_lower:
            return []
        if "from memories" in sql_lower:
            return mem
        elif "from media_content" in sql_lower:
            return med
        elif "from memory_chunks" in sql_lower:
            return chk
        return []

    pool.fetch = AsyncMock(side_effect=_fetch)
    pool.execute = AsyncMock(return_value="INSERT 0 1")
    return pool


def _make_retriever(
    pool: Any,
    authorizer: MemoryAuthorizerDependencies,
    config: MemoryConfig | None = None,
) -> MemoryRetriever:
    """build a retriever with registry-bound Collections around the pool.

    mirrors the production wiring shape: configure the registry with
    the pool, then construct each Collection.
    """
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    core_config = DefaultCoreConfig(
        collection_flush="ALWAYS",
        collection_flush_tables="",
    )
    memories = MemoriesCollection(
        registry=registry,
        config=core_config,
        authorizer=authorizer,
    )
    media_content = MediaContentCollection(
        registry=registry,
        config=core_config,
    )
    chunks = MemoryChunkCollection(
        registry=registry,
        config=core_config,
    )
    return MemoryRetriever(
        config=config or MemoryConfig(),
        embedding_provider=StubEmbeddingProvider(),
        authorizer=authorizer,
        memories_collection=memories,
        media_content_collection=media_content,
        memory_chunk_collection=chunks,
    )


class TestMemoryRetrieverE2E:
    async def test_returns_formatted_string(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        mem_id = uuid.uuid7()
        pool = _make_mock_pool(
            memory_rows=[
                {
                    "memory_id": mem_id,
                    "content": "User likes Python",
                    "summary": None,
                    "type_memory": "preference",
                    "date_created": datetime.now(timezone.utc),
                    "embedding": [1.0, 0.0, 0.0],
                    "similarity": 0.9,
                }
            ],
        )
        retriever = _make_retriever(pool, permissive_memory_authorizer)

        result = await retriever.retrieve(
            uuid.uuid7(),
            "Tell me about Python",
            agent_id=uuid.uuid7(),
            customer_id=uuid.uuid7(),
        )
        assert result is not None
        assert "User likes Python" in result
        assert "Things you remember" in result

    async def test_empty_text_returns_none(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        pool = AsyncMock()
        retriever = _make_retriever(pool, permissive_memory_authorizer)

        result = await retriever.retrieve(
            uuid.uuid7(),
            "  ",
            agent_id=uuid.uuid7(),
            customer_id=uuid.uuid7(),
        )
        assert result is None

    async def test_embedding_failure_returns_none(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        class FailingProvider:
            async def aembed_query(self, text: str) -> list[float]:
                _ = text
                raise RuntimeError("embedding service down")

            async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
                _ = texts
                raise RuntimeError("embedding service down")

            def embed_query(self, text: str) -> list[float]:
                _ = text
                return []

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                _ = texts
                return []

            @property
            def dimensions(self) -> int:
                return 3

        pool = AsyncMock()
        registry = CollectionRegistry()
        registry.configure(l3_pool=pool)
        core_config = DefaultCoreConfig(
            collection_flush="ALWAYS",
            collection_flush_tables="",
        )
        memories = MemoriesCollection(
            registry=registry,
            config=core_config,
            authorizer=permissive_memory_authorizer,
        )
        media_content = MediaContentCollection(
            registry=registry,
            config=core_config,
        )
        chunks = MemoryChunkCollection(
            registry=registry,
            config=core_config,
        )
        retriever = MemoryRetriever(
            config=MemoryConfig(),
            embedding_provider=FailingProvider(),
            authorizer=permissive_memory_authorizer,
            memories_collection=memories,
            media_content_collection=media_content,
            memory_chunk_collection=chunks,
        )

        result = await retriever.retrieve(
            uuid.uuid7(),
            "hello world",
            agent_id=uuid.uuid7(),
            customer_id=uuid.uuid7(),
        )
        assert result is None

    async def test_no_results_returns_none(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        pool = _make_mock_pool()
        retriever = _make_retriever(pool, permissive_memory_authorizer)

        result = await retriever.retrieve(
            uuid.uuid7(),
            "hello world",
            agent_id=uuid.uuid7(),
            customer_id=uuid.uuid7(),
        )
        assert result is None
