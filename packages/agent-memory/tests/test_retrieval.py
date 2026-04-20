"""Tests for memory retrieval -- ranking math, MMR, FTS, formatting."""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
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

    def test_naive_datetime_treated_as_utc(self) -> None:
        t = datetime.now(timezone.utc).replace(tzinfo=None)
        result = _recency_decay(t, half_life_hours=24.0)
        assert result == pytest.approx(1.0, abs=0.01)

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
        # Three candidates: two very similar, one different
        candidates = [
            {"embedding": [1.0, 0.0], "hybrid_score": 0.95},
            {"embedding": [0.99, 0.1], "hybrid_score": 0.90},  # near-duplicate of first
            {"embedding": [0.0, 1.0], "hybrid_score": 0.85},  # very different
        ]
        result = _mmr_rerank([1.0, 0.0], candidates, k=2, lambda_mult=0.5)
        # Should pick the top scorer and the diverse one, not the near-duplicate
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
        assert "recall_memory" in result

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
                "summary": None,
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

    def test_footer_present(self) -> None:
        memories = [{"memory_id": uuid.uuid7(), "content": "x", "summary": None, "hybrid_score": 0.5}]
        result = _format_memory_context(memories, detail_threshold=0.85)
        assert "recall_memory" in result

    def test_empty_returns_empty(self) -> None:
        result = _format_memory_context([])
        assert result == ""


# -- MemoryRetriever end-to-end ------------------------------------------------


class StubEmbeddingProvider:
    """Stub embedding provider for testing."""

    def __init__(self, embedding: list[float] | None = None) -> None:
        self._embedding = embedding or [1.0, 0.0, 0.0]

    async def embed_text(self, text: str) -> tuple[list[float] | None, int]:
        return self._embedding, 10

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
        # FTS queries return no results in this mock
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


class TestMemoryRetrieverE2E:
    async def test_returns_formatted_string(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        config = MemoryConfig()
        provider = StubEmbeddingProvider([1.0, 0.0, 0.0])
        retriever = MemoryRetriever(config, provider, permissive_memory_authorizer)

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

        result = await retriever.retrieve(pool, uuid.uuid7(), "Tell me about Python")
        assert result is not None
        assert "User likes Python" in result
        assert "Things you remember" in result

    async def test_empty_text_returns_none(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        config = MemoryConfig()
        provider = StubEmbeddingProvider()
        retriever = MemoryRetriever(config, provider, permissive_memory_authorizer)
        pool = AsyncMock()

        result = await retriever.retrieve(pool, uuid.uuid7(), "  ")
        assert result is None

    async def test_embedding_failure_returns_none(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        config = MemoryConfig()

        class FailingProvider:
            async def embed_text(self, text: str) -> tuple[None, int]:
                return None, 0

            @property
            def dimensions(self) -> int:
                return 3

        retriever = MemoryRetriever(
            config, FailingProvider(), permissive_memory_authorizer,
        )
        pool = AsyncMock()

        result = await retriever.retrieve(pool, uuid.uuid7(), "hello world")
        assert result is None

    async def test_no_results_returns_none(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        config = MemoryConfig()
        provider = StubEmbeddingProvider()
        retriever = MemoryRetriever(config, provider, permissive_memory_authorizer)
        pool = _make_mock_pool()

        result = await retriever.retrieve(pool, uuid.uuid7(), "hello world")
        assert result is None
