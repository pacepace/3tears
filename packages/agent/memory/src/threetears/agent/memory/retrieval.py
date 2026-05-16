"""Memory retrieval -- hybrid search across memories, media, and chunks.

Three Collections feed the retrieval pipeline: every SQL site that used
to live raw on a pool is now a call onto
:class:`MemoriesCollection.hybrid_search`,
:class:`MediaContentCollection.hybrid_search`, or
:class:`MemoryChunkCollection.hybrid_search`. The retriever stays
responsible for orchestration (parallel gather across the three tables,
cross-type MMR rerank, context formatting) but never touches SQL
directly.

Surfaced-item bookkeeping -- the ``conversation_memory_refs`` table --
is no longer a retriever concern. callers that want to dedupe already-
surfaced items pass ``surfaced_ids`` to :meth:`MemoryRetriever.retrieve`
and persist new surfacings through :class:`MemoryRefsCollection` at
their own layer. this retires the hand-rolled ``MemoryLedger`` wrapper
under namespace-task-01 phase 8.5l-2.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from threetears.agent.memory.authorize import (
    ACTION_MEMORY_READ,
    MemoryAuthorizerDependencies,
    authorize_memory_access,
)
from threetears.agent.memory.collections import (
    MediaContentCollection,
    MemoriesCollection,
    MemoryChunkCollection,
)
from langchain_core.embeddings import Embeddings

from threetears.agent.memory.embedding_utils import _estimate_tokens, _safe_aembed_query
from threetears.agent.memory.types import MemoryConfig
from threetears.observe import traced

__all__ = [
    "MemoryRetriever",
    "RetrievalResult",
]


def _recency_decay(created: datetime, half_life_hours: float) -> float:
    """Exponential recency decay (1.0 = just created, ~0.37 at half-life).

    :param created: creation timestamp; must be timezone-aware UTC
        (every datetime in the platform is aware-UTC after
        collections-task-05; passing naive raises TypeError on the
        ``now - created`` subtract)
    :ptype created: datetime
    :param half_life_hours: half-life in hours
    :ptype half_life_hours: float
    :return: decay factor in (0, 1]
    :rtype: float
    """
    now = datetime.now(timezone.utc)
    hours_ago = max((now - created).total_seconds() / 3600, 0.0)
    return math.exp(-hours_ago / half_life_hours)


def _build_fts_query(text: str, min_len: int = 3, max_len: int = 500) -> str | None:
    """Prepare text for ``websearch_to_tsquery``. Returns None if too short.

    :param text: raw text
    :ptype text: str
    :param min_len: minimum length to survive
    :ptype min_len: int
    :param max_len: truncation cap
    :ptype max_len: int
    :return: cleaned text or ``None``
    :rtype: str | None
    """
    text = text.strip()
    if len(text) < min_len:
        return None
    return text[:max_len]


def _normalize_fts_scores(
    candidates: list[dict[str, Any]],
    key: str = "fts_rank",
) -> None:
    """Min-max normalize FTS ranks in-place to [0, 1].

    :param candidates: candidate rows (mutated)
    :ptype candidates: list[dict[str, Any]]
    :param key: score column
    :ptype key: str
    :return: nothing
    :rtype: None
    """
    scores = [c.get(key, 0.0) for c in candidates]
    lo, hi = min(scores), max(scores)
    span = hi - lo
    for c in candidates:
        raw = c.get(key, 0.0)
        if span > 0:
            c[key] = (raw - lo) / span
        elif hi > 0:
            c[key] = 1.0


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors.

    :param a: first vector
    :ptype a: list[float]
    :param b: second vector
    :ptype b: list[float]
    :return: similarity in [-1, 1]
    :rtype: float
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _get_display_text(
    item: dict[str, Any],
    detail_threshold: float,
    score_key: str = "hybrid_score",
) -> tuple[str, bool]:
    """Return (display_text, is_detailed) for a memory/media/chunk item.

    :param item: candidate row
    :ptype item: dict[str, Any]
    :param detail_threshold: score above which to show full content
    :ptype detail_threshold: float
    :param score_key: column to consult for scoring
    :ptype score_key: str
    :return: tuple of (text, is_detailed)
    :rtype: tuple[str, bool]
    """
    score = item.get(score_key, item.get("similarity", 0.0))
    if score >= detail_threshold:
        return item["content"], True
    summary = item.get("summary")
    if summary:
        return summary, False
    content = item["content"]
    if len(content) > 150:
        return content[:147] + "...", False
    return content, False


def _mmr_rerank(
    query_embedding: list[float],
    candidates: list[dict[str, Any]],
    k: int = 3,
    lambda_mult: float = 0.7,
) -> list[dict[str, Any]]:
    """Maximal Marginal Relevance reranking for diversity.

    :param query_embedding: query embedding vector
    :ptype query_embedding: list[float]
    :param candidates: candidates to rerank
    :ptype candidates: list[dict[str, Any]]
    :param k: size of output set
    :ptype k: int
    :param lambda_mult: trade-off between relevance and diversity
    :ptype lambda_mult: float
    :return: reranked top-k
    :rtype: list[dict[str, Any]]
    """
    _ = query_embedding
    if not candidates:
        return []
    if len(candidates) <= k:
        return candidates

    selected: list[dict[str, Any]] = []
    remaining = list(candidates)

    for _ in range(min(k, len(remaining))):
        best_idx, best_score = -1, -float("inf")
        for i, cand in enumerate(remaining):
            relevance = cand.get("hybrid_score", cand.get("similarity", 0.0))
            if selected:
                redundancy = max(_cosine_sim(cand["embedding"], s["embedding"]) for s in selected)
            else:
                redundancy = 0.0
            score = lambda_mult * relevance - (1 - lambda_mult) * redundancy
            if score > best_score:
                best_score, best_idx = score, i
        selected.append(remaining.pop(best_idx))

    return selected


def _format_memory_context(
    memories: list[dict[str, Any]],
    media_content: list[dict[str, Any]] | None = None,
    memory_chunks: list[dict[str, Any]] | None = None,
    detail_threshold: float = 0.85,
    ledgered_ids: set[str] | None = None,
) -> str:
    """Format retrieved memories, media content, and chunks as structured context.

    :param memories: ranked memory rows
    :ptype memories: list[dict[str, Any]]
    :param media_content: ranked media content rows
    :ptype media_content: list[dict[str, Any]] | None
    :param memory_chunks: ranked chunk rows
    :ptype memory_chunks: list[dict[str, Any]] | None
    :param detail_threshold: cutoff for showing full content
    :ptype detail_threshold: float
    :param ledgered_ids: IDs to exclude (already surfaced)
    :ptype ledgered_ids: set[str] | None
    :return: formatted context string
    :rtype: str
    """
    lines: list[str] = []
    _ledger = ledgered_ids or set()

    if _ledger:
        memories = [m for m in memories if str(m["memory_id"]) not in _ledger]
        if media_content:
            media_content = [mc for mc in media_content if str(mc["content_id"]) not in _ledger]
        if memory_chunks:
            memory_chunks = [c for c in memory_chunks if str(c["chunk_id"]) not in _ledger]

    media_ids_covered: set[str] = set()
    if media_content:
        for mc in media_content:
            mid = mc.get("media_id")
            if mid:
                media_ids_covered.add(str(mid))

    deduped_chunks: list[dict[str, Any]] = []
    if memory_chunks:
        for chunk in memory_chunks:
            chunk_media_id = chunk.get("media_id")
            if chunk_media_id and str(chunk_media_id) in media_ids_covered:
                continue
            deduped_chunks.append(chunk)

    if memories:
        lines.append("Things you remember about this user:")
        for mem in memories:
            text, detailed = _get_display_text(mem, detail_threshold)
            marker = " (detailed)" if detailed else ""
            mem_id = str(mem["memory_id"])
            lines.append(f"- [mem:{mem_id}] {text}{marker}")

    if media_content:
        if lines:
            lines.append("")
        lines.append("Relevant media context:")
        for mc in media_content:
            text, detailed = _get_display_text(mc, detail_threshold)
            marker = " (detailed)" if detailed else ""
            content_id = str(mc["content_id"])
            lines.append(f"- [media:{content_id}] {text}{marker}")

    if deduped_chunks:
        if lines:
            lines.append("")
        lines.append("Relevant document excerpts:")
        for chunk in deduped_chunks:
            text, detailed = _get_display_text(chunk, detail_threshold)
            marker = " (detailed)" if detailed else ""
            chunk_id = str(chunk["chunk_id"])
            title = chunk.get("title") or "untitled"
            page = chunk.get("page_number")
            heading = chunk.get("heading_context")
            loc_parts = [f'"{title}"']
            if page:
                loc_parts.append(f"p.{page}")
            if heading:
                loc_parts.append(f'"{heading}"')
            location = " ".join(loc_parts)
            lines.append(f"- [chunk:{chunk_id} {location}] {text}{marker}")

    if lines:
        lines.append("")
        lines.append(
            "Items marked (detailed) contain full content above. For summary-only items, "
            "use the memory_recall tool with the item ID and type (memory/media/chunk) "
            "to retrieve full content when needed."
        )

    return "\n".join(lines)


@dataclass
class RetrievalResult:
    """Structured output from memory retrieval.

    Carries both the formatted context string (for prompt injection)
    and the raw candidate lists (for event dispatch, logging, etc.).
    """

    context: str | None = None
    memories: list[dict[str, Any]] = field(default_factory=list)
    media_content: list[dict[str, Any]] = field(default_factory=list)
    memory_chunks: list[dict[str, Any]] = field(default_factory=list)
    embed_tokens: int = 0


class MemoryRetriever:
    """Hybrid memory retrieval with MMR reranking.

    Not a LangGraph node -- a plain callable class. Collection-
    parameterised: the three memory-package Collections
    (:class:`MemoriesCollection`, :class:`MediaContentCollection`,
    :class:`MemoryChunkCollection`) are required at construction; the
    retriever does no raw SQL and holds no pool reference — Collections
    carry the pool internally.
    """

    def __init__(
        self,
        config: MemoryConfig,
        embedding_provider: Embeddings,
        authorizer: MemoryAuthorizerDependencies,
        memories_collection: MemoriesCollection,
        media_content_collection: MediaContentCollection,
        memory_chunk_collection: MemoryChunkCollection,
    ) -> None:
        """initialize the retriever with Collections + rbac authorizer.

        :param config: memory retrieval configuration
        :ptype config: MemoryConfig
        :param embedding_provider: embedding provider for query vectors
        :ptype embedding_provider: Embeddings
        :param authorizer: rbac authorizer dependency bundle; required.
            every retrieval call carrying ``caller_user_id`` evaluates
            ``memory.read`` on the per-(agent, customer) memory
            namespace before any SQL runs. tests inject a permissive
            fixture from ``conftest.py``; production wiring builds
            the real bundle from hub-side loaders + namespace resolver
        :ptype authorizer: MemoryAuthorizerDependencies
        :param memories_collection: three-tier memories collection
        :ptype memories_collection: MemoriesCollection
        :param media_content_collection: three-tier media_content
            collection
        :ptype media_content_collection: MediaContentCollection
        :param memory_chunk_collection: three-tier memory_chunks
            collection
        :ptype memory_chunk_collection: MemoryChunkCollection
        """
        self._config = config
        self._embedding = embedding_provider
        self._authorizer = authorizer
        self._memories = memories_collection
        self._media_content = media_content_collection
        self._chunks = memory_chunk_collection

    @traced(record_args=False)
    async def retrieve(
        self,
        user_id: UUID,
        user_text: str,
        *,
        agent_id: UUID,
        customer_id: UUID,
        surfaced_ids: set[str] | None = None,
        caller_user_id: UUID | None = None,
        caller_agent_id: UUID | None = None,
    ) -> str | None:
        """full retrieval pipeline; returns formatted context or ``None``.

        :param user_id: user whose memories to search (row filter)
        :ptype user_id: UUID
        :param user_text: query text for similarity search
        :ptype user_text: str
        :param agent_id: partition column on the memory tables; required
        :ptype agent_id: UUID
        :param customer_id: required sub-scope
        :ptype customer_id: UUID
        :param surfaced_ids: optional set of item IDs (as strings)
            already shown to the agent this conversation; rows whose
            pk stringifies to one of these are filtered out of the
            formatted context. callers persist these through
            :class:`MemoryRefsCollection` at their own layer.
        :ptype surfaced_ids: set[str] | None
        :param caller_user_id: invoking user UUID for rbac evaluator
        :ptype caller_user_id: UUID | None
        :param caller_agent_id: invoking agent UUID; owner short-
            circuit applies when equal to ``agent_id``
        :ptype caller_agent_id: UUID | None
        :return: formatted context string or ``None``
        :rtype: str | None
        :raises MemoryAccessDenied: when rbac enforcement denies
        """
        result = await self.retrieve_with_candidates(
            user_id,
            user_text,
            agent_id=agent_id,
            customer_id=customer_id,
            surfaced_ids=surfaced_ids,
            caller_user_id=caller_user_id,
            caller_agent_id=caller_agent_id,
        )
        return result.context

    @traced(record_args=False)
    async def retrieve_with_candidates(
        self,
        user_id: UUID,
        user_text: str,
        *,
        agent_id: UUID,
        customer_id: UUID,
        surfaced_ids: set[str] | None = None,
        caller_user_id: UUID | None = None,
        caller_agent_id: UUID | None = None,
    ) -> RetrievalResult:
        """Full retrieval pipeline returning structured results.

        :param user_id: user whose memories to search
        :ptype user_id: UUID
        :param user_text: query text for similarity search
        :ptype user_text: str
        :param agent_id: partition column on the memory tables; required
        :ptype agent_id: UUID
        :param customer_id: required sub-scope
        :ptype customer_id: UUID
        :param surfaced_ids: optional set of item IDs (as strings)
            already surfaced in this conversation; filtered from the
            returned context. callers persist new surfacings through
            :class:`MemoryRefsCollection` at their own layer
        :ptype surfaced_ids: set[str] | None
        :param caller_user_id: invoking user UUID for rbac evaluator
        :ptype caller_user_id: UUID | None
        :param caller_agent_id: invoking agent UUID
        :ptype caller_agent_id: UUID | None
        :return: structured retrieval results
        :rtype: RetrievalResult
        :raises MemoryAccessDenied: when rbac enforcement denies
        """
        empty = RetrievalResult()
        if not user_text.strip():
            return empty

        if caller_user_id is not None:
            await authorize_memory_access(
                action=ACTION_MEMORY_READ,
                agent_id=agent_id,
                customer_id=customer_id,
                caller_user_id=caller_user_id,
                caller_agent_id=caller_agent_id,
                deps=self._authorizer,
            )

        embedding = await _safe_aembed_query(self._embedding, user_text)
        if embedding is None:
            return empty
        embed_tokens = _estimate_tokens(user_text)

        cfg = self._config
        top_k = cfg.top_k
        candidate_limit = top_k * cfg.candidate_multiplier
        media_top_k = cfg.media_top_k
        media_candidate_limit = media_top_k * cfg.media_candidate_multiplier
        chunk_candidate_k = cfg.chunk_candidate_k

        memories, media_content, memory_chunks = await asyncio.gather(
            self._memories.hybrid_search(
                user_id=user_id,
                embedding=embedding,
                user_text=user_text,
                top_k=top_k,
                candidate_limit=candidate_limit,
                similarity_threshold=cfg.similarity_threshold,
                recency_half_life_hours=cfg.recency_half_life_hours,
                signal_weights=cfg.signal_weights,
                agent_id=agent_id,
                customer_id=customer_id,
                fts_min_len=cfg.fts_min_query_len,
                fts_max_len=cfg.fts_max_query_len,
            ),
            self._media_content.hybrid_search(
                user_id=user_id,
                embedding=embedding,
                user_text=user_text,
                top_k=media_top_k,
                candidate_limit=media_candidate_limit,
                similarity_threshold=cfg.similarity_threshold,
                recency_half_life_hours=cfg.recency_half_life_hours,
                signal_weights=cfg.signal_weights,
                agent_id=agent_id,
                customer_id=customer_id,
                fts_min_len=cfg.fts_min_query_len,
                fts_max_len=cfg.fts_max_query_len,
            ),
            self._chunks.hybrid_search(
                user_id=user_id,
                embedding=embedding,
                user_text=user_text,
                candidate_k=chunk_candidate_k,
                similarity_threshold=cfg.similarity_threshold,
                chunk_signal_weights=cfg.chunk_signal_weights,
                agent_id=agent_id,
                customer_id=customer_id,
                fts_min_len=cfg.fts_min_query_len,
                fts_max_len=cfg.fts_max_query_len,
            ),
        )

        all_candidates: list[dict[str, Any]] = []
        for m in memories:
            m["_source_type"] = "memory"
            all_candidates.append(m)
        for mc in media_content:
            mc["_source_type"] = "media"
            all_candidates.append(mc)
        for ch in memory_chunks:
            ch["_source_type"] = "chunk"
            all_candidates.append(ch)

        if not all_candidates:
            return RetrievalResult(embed_tokens=embed_tokens)

        selected = _mmr_rerank(
            embedding,
            all_candidates,
            k=cfg.context_budget,
            lambda_mult=cfg.mmr_lambda,
        )

        memories = [c for c in selected if c["_source_type"] == "memory"]
        media_content = [c for c in selected if c["_source_type"] == "media"]
        memory_chunks = [c for c in selected if c["_source_type"] == "chunk"]

        for c in selected:
            c.pop("_source_type", None)
            c.pop("embedding", None)
            c.pop("fts_rank", None)

        ledgered_ids = surfaced_ids if surfaced_ids is not None else set()

        if not memories and not media_content and not memory_chunks:
            return RetrievalResult(embed_tokens=embed_tokens)

        context = _format_memory_context(
            memories,
            media_content,
            memory_chunks,
            detail_threshold=cfg.detail_threshold,
            ledgered_ids=ledgered_ids,
        )

        return RetrievalResult(
            context=context or None,
            memories=memories,
            media_content=media_content,
            memory_chunks=memory_chunks,
            embed_tokens=embed_tokens,
        )
