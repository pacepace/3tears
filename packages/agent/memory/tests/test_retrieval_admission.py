"""Admission is separate from ranking in the hybrid searches.

These drive the real ``hybrid_search`` bodies against a stub pool, because the
defect they pin lives in the merge-score-filter sequence rather than in any
helper: gating admission on the composite ``hybrid_score`` let an ABSENT signal
lower the attainable ceiling, so a month-old memory was unreachable at any
semantic relevance.

Every ``date_created`` here is deliberately aged. The suite's other retrieval
tests hand-inject ``hybrid_score`` literals or stamp rows at ``now``, which
holds the recency term at 1.0 and hides exactly this class of bug.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import (
    MediaContentCollection,
    MemoriesCollection,
    MemoryChunkCollection,
)
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

# Production defaults, restated so a change to them shows up here as a failing
# test rather than as a silent shift in what these tests actually prove.
WEIGHTS = {"semantic": 0.55, "keyword": 0.15, "recency": 0.30}
CHUNK_WEIGHTS = {"semantic": 0.80, "keyword": 0.20}
HALF_LIFE_HOURS = 24.0
THRESHOLD = 0.4

AGED = datetime.now(UTC) - timedelta(days=30)
FRESH = datetime.now(UTC) - timedelta(hours=1)


def _pool(*, vec_rows: list[dict[str, Any]], fts_rows: list[dict[str, Any]]) -> AsyncMock:
    """Stub L3 pool that answers the vector and FTS legs from fixed rows.

    :param vec_rows: rows the vector-distance leg returns
    :ptype vec_rows: list[dict[str, Any]]
    :param fts_rows: rows the FTS-rank leg returns
    :ptype fts_rows: list[dict[str, Any]]
    :return: mock exposing ``fetch`` / ``execute``
    :rtype: AsyncMock
    """

    async def _fetch(sql: str, *args: object) -> list[dict[str, Any]]:
        _ = args
        return fts_rows if "ts_rank_cd" in sql.lower() else vec_rows

    pool = AsyncMock()
    pool.fetch = AsyncMock(side_effect=_fetch)
    pool.execute = AsyncMock(return_value="UPDATE 1")
    return pool


def _memory_row(
    *,
    similarity: float = 0.0,
    fts_rank: float = 0.0,
    date_created: datetime = AGED,
    content: str = "the Severed Isle and its High College",
) -> dict[str, Any]:
    """Build one ``memories`` row as either leg's SQL returns it.

    :param similarity: cosine similarity from the vector leg
    :ptype similarity: float
    :param fts_rank: raw ts_rank_cd score from the FTS leg
    :ptype fts_rank: float
    :param date_created: row age driver
    :ptype date_created: datetime
    :param content: row content
    :ptype content: str
    :return: row dict
    :rtype: dict[str, Any]
    """
    return {
        "memory_id": uuid.uuid7(),
        "content": content,
        "summary": None,
        "type_memory": "fact",
        "date_created": date_created,
        "embedding": "[1.0, 0.0]",
        "similarity": similarity,
        "fts_rank": fts_rank,
    }


def _media_row(
    *,
    similarity: float = 0.0,
    fts_rank: float = 0.0,
    date_created: datetime = AGED,
) -> dict[str, Any]:
    """Build one ``media_content`` row as either leg's SQL returns it.

    :param similarity: cosine similarity from the vector leg
    :ptype similarity: float
    :param fts_rank: raw ts_rank_cd score from the FTS leg
    :ptype fts_rank: float
    :param date_created: row age driver
    :ptype date_created: datetime
    :return: row dict
    :rtype: dict[str, Any]
    """
    return {
        "content_id": uuid.uuid7(),
        "content": "a map of the Severed Isle",
        "summary": None,
        "content_type": "image_description",
        "media_id": uuid.uuid7(),
        "media_category": "image",
        "metadata_json": None,
        "date_created": date_created,
        "embedding": "[1.0, 0.0]",
        "similarity": similarity,
        "fts_rank": fts_rank,
    }


def _chunk_row(*, similarity: float = 0.0, fts_rank: float = 0.0) -> dict[str, Any]:
    """Build one ``memory_chunks`` row as either leg's SQL returns it.

    :param similarity: cosine similarity from the vector leg
    :ptype similarity: float
    :param fts_rank: raw ts_rank_cd score from the FTS leg
    :ptype fts_rank: float
    :return: row dict
    :rtype: dict[str, Any]
    """
    return {
        "chunk_id": uuid.uuid7(),
        "content": "columnar basalt in the east tower",
        "summary": None,
        "heading_context": None,
        "page_number": None,
        "memory_id": uuid.uuid7(),
        "media_id": None,
        "message_id_start": None,
        "message_id_end": None,
        "metadata_json": None,
        "embedding": "[1.0, 0.0]",
        "similarity": similarity,
        "fts_rank": fts_rank,
    }


def _memories(pool: AsyncMock, authorizer: MemoryAuthorizerDependencies) -> MemoriesCollection:
    """Bind a ``MemoriesCollection`` to the stub pool.

    :param pool: stub L3 pool
    :ptype pool: AsyncMock
    :param authorizer: permissive rbac bundle
    :ptype authorizer: MemoryAuthorizerDependencies
    :return: collection under test
    :rtype: MemoriesCollection
    """
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    return MemoriesCollection(
        registry=registry,
        config=DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""),
        authorizer=authorizer,
    )


async def _search(collection: MemoriesCollection, **overrides: Any) -> list[dict[str, Any]]:
    """Run ``hybrid_search`` with the production-default knobs.

    :param collection: collection under test
    :ptype collection: MemoriesCollection
    :param overrides: per-test kwarg overrides
    :ptype overrides: Any
    :return: admitted rows
    :rtype: list[dict[str, Any]]
    """
    kwargs: dict[str, Any] = {
        "user_id": uuid.uuid7(),
        "agent_id": uuid.uuid7(),
        "customer_id": uuid.uuid7(),
        "embedding": [1.0, 0.0],
        "user_text": "what do you remember about the Severed Isle",
        "top_k": 10,
        "candidate_limit": 30,
        "similarity_threshold": THRESHOLD,
        "recency_half_life_hours": HALF_LIFE_HOURS,
        "signal_weights": WEIGHTS,
    }
    kwargs.update(overrides)
    return await collection.hybrid_search(**kwargs)


class TestAgedMemoryAdmission:
    """A memory's age must not decide whether it can be recalled at all."""

    async def test_aged_relevant_memory_is_admitted(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        """30 days old, strong semantic match, no keyword hit -> admitted.

        The composite for this row is ``0.55*0.50 + 0.15*0 + 0.30*~0 = 0.275``,
        which the old ``hybrid_score > 0.4`` gate rejected. Nothing about the
        row is marginal: 0.50 is at the top of the cosine range this embedding
        model produces for a genuine hit.
        """
        row = _memory_row(similarity=0.50, date_created=AGED)
        collection = _memories(_pool(vec_rows=[row], fts_rows=[]), permissive_memory_authorizer)

        results = await _search(collection)

        assert len(results) == 1
        assert results[0]["memory_id"] == row["memory_id"]
        assert results[0]["hybrid_score"] < THRESHOLD, (
            "guard: this test is only meaningful while the composite sits BELOW "
            "the threshold — otherwise it would pass under the old gate too"
        )

    async def test_aged_irrelevant_memory_is_still_cut(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        """The dropper still drops: weak semantic match, no keyword hit."""
        row = _memory_row(similarity=0.15, date_created=AGED)
        collection = _memories(_pool(vec_rows=[row], fts_rows=[]), permissive_memory_authorizer)

        assert await _search(collection) == []

    async def test_admission_boundary_is_inclusive(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        """``similarity == threshold`` is admitted; just under it is not."""
        at = _memory_row(similarity=THRESHOLD, date_created=AGED)
        under = _memory_row(similarity=THRESHOLD - 0.01, date_created=AGED)
        collection = _memories(_pool(vec_rows=[at, under], fts_rows=[]), permissive_memory_authorizer)

        results = await _search(collection)

        assert [r["memory_id"] for r in results] == [at["memory_id"]]


class TestKeywordOnlyAdmission:
    """A keyword hit is evidence too — admitting on similarity alone loses it."""

    async def test_keyword_only_match_is_admitted(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        """An FTS-only row survives despite its placeholder 0.0 similarity.

        Rows reaching the pool through the FTS leg alone never had a distance
        computed, so their ``similarity`` is a placeholder rather than a
        measurement. Gating them on it would silently retire the keyword leg.
        """
        row = _memory_row(similarity=0.0, fts_rank=0.9, date_created=AGED)
        collection = _memories(_pool(vec_rows=[], fts_rows=[row]), permissive_memory_authorizer)

        results = await _search(collection)

        assert [r["memory_id"] for r in results] == [row["memory_id"]]

    async def test_lowest_ranked_keyword_hit_still_admitted(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        """Min-max normalization drives the weakest FTS hit's rank to 0.0.

        Admission must consult *whether the row matched*, not its normalized
        score, or the weakest keyword hit in every result set is discarded for
        an artifact of the normalization.
        """
        weak = _memory_row(similarity=0.0, fts_rank=0.1, date_created=AGED)
        strong = _memory_row(similarity=0.0, fts_rank=0.9, date_created=AGED)
        collection = _memories(_pool(vec_rows=[], fts_rows=[weak, strong]), permissive_memory_authorizer)

        results = await _search(collection)

        assert {r["memory_id"] for r in results} == {weak["memory_id"], strong["memory_id"]}
        assert min(r["fts_rank"] for r in results) == 0.0, "expected the weak hit to normalize to 0.0"

    async def test_private_fts_marker_is_not_returned(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        """The admission marker is module-private and must not reach callers."""
        rows = [
            _memory_row(similarity=0.50, date_created=AGED),
            _memory_row(similarity=0.0, fts_rank=0.9, date_created=AGED),
        ]
        collection = _memories(_pool(vec_rows=rows[:1], fts_rows=rows[1:]), permissive_memory_authorizer)

        for result in await _search(collection):
            assert not any(key.startswith("_") for key in result), f"private key leaked: {sorted(result)}"


class TestRecencyRanksButDoesNotAdmit:
    """Recency orders relevant candidates; it never decides they exist."""

    async def test_equally_relevant_rows_of_different_ages_both_survive(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        """Same similarity, 30 days apart -> both admitted, newest ranked first."""
        old = _memory_row(similarity=0.50, date_created=AGED)
        new = _memory_row(similarity=0.50, date_created=FRESH)
        collection = _memories(_pool(vec_rows=[old, new], fts_rows=[]), permissive_memory_authorizer)

        results = await _search(collection)

        assert [r["memory_id"] for r in results] == [new["memory_id"], old["memory_id"]]
        assert results[0]["hybrid_score"] > results[1]["hybrid_score"]

    async def test_recent_but_irrelevant_row_is_not_admitted(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        """Freshness alone no longer buys admission.

        Under the old gate a brand-new row needed only ``similarity > 0.18`` to
        clear 0.4 on its recency term, so today's weakest scraps outranked every
        genuinely relevant older memory.
        """
        row = _memory_row(similarity=0.20, date_created=FRESH)
        collection = _memories(_pool(vec_rows=[row], fts_rows=[]), permissive_memory_authorizer)

        assert await _search(collection) == []


class TestMediaContentAdmission:
    """``media_content`` carries the same three-signal weights, and the same bug.

    A shared helper cannot vouch for this path on its own: each collection marks
    its own FTS-matched rows during its own merge, and that per-site line is
    exactly what the helper cannot guarantee.
    """

    @staticmethod
    def _collection(pool: AsyncMock) -> MediaContentCollection:
        """Bind a ``MediaContentCollection`` to the stub pool.

        :param pool: stub L3 pool
        :ptype pool: AsyncMock
        :return: collection under test
        :rtype: MediaContentCollection
        """
        registry = CollectionRegistry()
        registry.configure(l3_pool=pool)
        return MediaContentCollection(
            registry=registry,
            config=DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""),
        )

    @staticmethod
    async def _search(collection: MediaContentCollection) -> list[dict[str, Any]]:
        """Run ``hybrid_search`` with the production-default knobs.

        :param collection: collection under test
        :ptype collection: MediaContentCollection
        :return: admitted rows
        :rtype: list[dict[str, Any]]
        """
        return await collection.hybrid_search(
            user_id=uuid.uuid7(),
            agent_id=uuid.uuid7(),
            customer_id=uuid.uuid7(),
            embedding=[1.0, 0.0],
            user_text="what do you remember about the Severed Isle",
            top_k=5,
            candidate_limit=15,
            similarity_threshold=THRESHOLD,
            recency_half_life_hours=HALF_LIFE_HOURS,
            signal_weights=WEIGHTS,
        )

    async def test_aged_relevant_media_is_admitted(self) -> None:
        """30 days old, strong semantic match, no keyword hit -> admitted."""
        row = _media_row(similarity=0.50, date_created=AGED)

        results = await self._search(self._collection(_pool(vec_rows=[row], fts_rows=[])))

        assert [r["content_id"] for r in results] == [row["content_id"]]
        assert results[0]["hybrid_score"] < THRESHOLD, (
            "guard: only meaningful while the composite sits BELOW the threshold"
        )

    async def test_aged_irrelevant_media_is_still_cut(self) -> None:
        """The dropper still drops on this path too."""
        row = _media_row(similarity=0.15, date_created=AGED)

        assert await self._search(self._collection(_pool(vec_rows=[row], fts_rows=[]))) == []

    async def test_keyword_only_media_is_admitted(self) -> None:
        """An FTS-only row survives its placeholder 0.0 similarity."""
        row = _media_row(similarity=0.0, fts_rank=0.9, date_created=AGED)

        results = await self._search(self._collection(_pool(vec_rows=[], fts_rows=[row])))

        assert [r["content_id"] for r in results] == [row["content_id"]]

    async def test_vector_row_also_matching_fts_is_marked(self) -> None:
        """A row in BOTH legs is admitted on its keyword evidence.

        This is the per-site line a shared helper cannot cover: the merge marks
        an ALREADY-merged vector row when the FTS leg returns it too. Its
        similarity is below the floor, so only the marking can admit it.
        """
        row = _media_row(similarity=0.10, date_created=AGED)
        fts_hit = dict(row, fts_rank=0.9)

        results = await self._search(self._collection(_pool(vec_rows=[row], fts_rows=[fts_hit])))

        assert [r["content_id"] for r in results] == [row["content_id"]]
        assert not any(key.startswith("_") for key in results[0])


class TestChunkAdmission:
    """The chunk path carries the same split, on two-signal weights."""

    async def test_aged_relevant_chunk_is_admitted(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        """Chunks have no recency term, so the composite gate bit them too.

        ``0.80*0.50 + 0.20*0 = 0.40`` is not ``> 0.40``: a chunk needed a
        similarity above 0.5 to survive without a keyword hit.
        """
        _ = permissive_memory_authorizer
        row = _chunk_row(similarity=0.50)
        registry = CollectionRegistry()
        registry.configure(l3_pool=_pool(vec_rows=[row], fts_rows=[]))
        chunks = MemoryChunkCollection(
            registry=registry,
            config=DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""),
        )

        results = await chunks.hybrid_search(
            user_id=uuid.uuid7(),
            agent_id=uuid.uuid7(),
            customer_id=uuid.uuid7(),
            embedding=[1.0, 0.0],
            user_text="what is the east tower made of",
            candidate_k=15,
            similarity_threshold=THRESHOLD,
            chunk_signal_weights=CHUNK_WEIGHTS,
        )

        assert [r["chunk_id"] for r in results] == [row["chunk_id"]]
        assert results[0]["hybrid_score"] == pytest.approx(0.40)
