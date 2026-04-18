"""Memory retrieval -- hybrid search across memories, media, and chunks."""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from threetears.agent.memory.embedding import EmbeddingProvider
from threetears.agent.memory.ledger import MemoryLedger
from threetears.agent.memory.types import MemoryConfig
from threetears.observe import traced

__all__ = [
    "MemoryRetriever",
    "RetrievalResult",
]


def _recency_decay(created: datetime, half_life_hours: float) -> float:
    """Exponential recency decay (1.0 = just created, ~0.37 at half-life)."""
    now = datetime.now(timezone.utc)
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    hours_ago = max((now - created).total_seconds() / 3600, 0.0)
    return math.exp(-hours_ago / half_life_hours)


def _build_fts_query(text: str, min_len: int = 3, max_len: int = 500) -> str | None:
    """Prepare text for ``websearch_to_tsquery``. Returns None if too short."""
    text = text.strip()
    if len(text) < min_len:
        return None
    return text[:max_len]


def _normalize_fts_scores(
    candidates: list[dict[str, Any]],
    key: str = "fts_rank",
) -> None:
    """Min-max normalize FTS ranks in-place to [0, 1]."""
    scores = [c.get(key, 0.0) for c in candidates]
    lo, hi = min(scores), max(scores)
    span = hi - lo
    for c in candidates:
        raw = c.get(key, 0.0)
        if span > 0:
            c[key] = (raw - lo) / span
        elif hi > 0:
            c[key] = 1.0
        # else: all zero, keep 0.0


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
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
    """Return (display_text, is_detailed) for a memory/media/chunk item."""
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

    Uses ``hybrid_score`` (fallback ``similarity``) as relevance.
    """
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
    """Format retrieved memories, media content, and chunks as structured context."""
    lines: list[str] = []
    _ledger = ledgered_ids or set()

    # Filter out ledgered items
    if _ledger:
        memories = [m for m in memories if str(m["memory_id"]) not in _ledger]
        if media_content:
            media_content = [mc for mc in media_content if str(mc["content_id"]) not in _ledger]
        if memory_chunks:
            memory_chunks = [c for c in memory_chunks if str(c["chunk_id"]) not in _ledger]

    # Build set of media_ids covered by media_content for chunk dedup
    media_ids_covered: set[str] = set()
    if media_content:
        for mc in media_content:
            mid = mc.get("media_id")
            if mid:
                media_ids_covered.add(str(mid))

    # Dedup chunks whose media_id is already in media_content results
    deduped_chunks: list[dict[str, Any]] = []
    if memory_chunks:
        for chunk in memory_chunks:
            chunk_media_id = chunk.get("media_id")
            if chunk_media_id and str(chunk_media_id) in media_ids_covered:
                continue
            deduped_chunks.append(chunk)

    # Section 1: User memories
    if memories:
        lines.append("Things you remember about this user:")
        for mem in memories:
            text, detailed = _get_display_text(mem, detail_threshold)
            marker = " (detailed)" if detailed else ""
            mem_id = str(mem["memory_id"])
            lines.append(f"- [mem:{mem_id}] {text}{marker}")

    # Section 2: Media context
    if media_content:
        if lines:
            lines.append("")
        lines.append("Relevant media context:")
        for mc in media_content:
            text, detailed = _get_display_text(mc, detail_threshold)
            marker = " (detailed)" if detailed else ""
            content_id = str(mc["content_id"])
            lines.append(f"- [media:{content_id}] {text}{marker}")

    # Section 3: Document/transcript excerpts (deduped)
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

    # Footer
    if lines:
        lines.append("")
        lines.append(
            "Items marked (detailed) contain full content above. For summary-only items, "
            "use the recall_memory tool with the item ID and type (memory/media/chunk) "
            "to retrieve full content when needed."
        )

    return "\n".join(lines)


def _build_scope_clause(
    user_id: UUID,
    *,
    agent_id: UUID | None = None,
    customer_id: UUID | None = None,
    start_param: int = 2,
    table_prefix: str = "",
) -> tuple[str, list[UUID], int]:
    """Build WHERE clause fragments for agent/customer/user scoping.

    :param user_id: user ID (always included)
    :ptype user_id: UUID
    :param agent_id: optional agent ID scope
    :ptype agent_id: UUID | None
    :param customer_id: optional customer ID scope
    :ptype customer_id: UUID | None
    :param start_param: starting positional parameter index
    :ptype start_param: int
    :param table_prefix: optional table alias prefix for column names
    :ptype table_prefix: str
    :return: tuple of (conditions string, param values, last param index used)
    :rtype: tuple[str, list[UUID], int]
    """
    prefix = f"{table_prefix}." if table_prefix else ""
    conditions: list[str] = []
    params: list[UUID] = []
    idx = start_param

    if agent_id is not None:
        conditions.append(f"{prefix}agent_id = ${idx}")
        params.append(agent_id)
        idx += 1

    if customer_id is not None:
        conditions.append(f"{prefix}customer_id = ${idx}")
        params.append(customer_id)
        idx += 1

    conditions.append(f"{prefix}user_id = ${idx}")
    params.append(user_id)

    return " AND ".join(conditions), params, idx


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

    Not a LangGraph node -- a plain callable class.
    """

    def __init__(self, config: MemoryConfig, embedding_provider: EmbeddingProvider) -> None:
        self._config = config
        self._embedding = embedding_provider

    @traced(record_args=False)
    async def retrieve(
        self,
        pool: Any,
        user_id: UUID,
        user_text: str,
        ledger: MemoryLedger | None = None,
        *,
        agent_id: UUID | None = None,
        customer_id: UUID | None = None,
    ) -> str | None:
        """Full retrieval pipeline. Returns formatted context string or None.

        :param pool: database connection pool
        :ptype pool: Any
        :param user_id: user whose memories to search
        :ptype user_id: UUID
        :param user_text: query text for similarity search
        :ptype user_text: str
        :param ledger: optional ledger for dedup tracking
        :ptype ledger: MemoryLedger | None
        :param agent_id: optional agent ID scope for filtering results
        :ptype agent_id: UUID | None
        :param customer_id: optional customer ID scope for filtering results
        :ptype customer_id: UUID | None
        :return: formatted context string or None if no results
        :rtype: str | None
        """
        result = await self.retrieve_with_candidates(
            pool,
            user_id,
            user_text,
            ledger,
            agent_id=agent_id,
            customer_id=customer_id,
        )
        return result.context

    @traced(record_args=False)
    async def retrieve_with_candidates(
        self,
        pool: Any,
        user_id: UUID,
        user_text: str,
        ledger: MemoryLedger | None = None,
        *,
        agent_id: UUID | None = None,
        customer_id: UUID | None = None,
    ) -> RetrievalResult:
        """Full retrieval pipeline returning structured results.

        Use this when you need both the formatted context and the raw
        candidate data (e.g. for event dispatch or token tracking).

        :param pool: database connection pool
        :ptype pool: Any
        :param user_id: user whose memories to search
        :ptype user_id: UUID
        :param user_text: query text for similarity search
        :ptype user_text: str
        :param ledger: optional ledger for dedup tracking
        :ptype ledger: MemoryLedger | None
        :param agent_id: optional agent ID scope for filtering results
        :ptype agent_id: UUID | None
        :param customer_id: optional customer ID scope for filtering results
        :ptype customer_id: UUID | None
        :return: structured retrieval results
        :rtype: RetrievalResult
        """
        empty = RetrievalResult()
        if not user_text.strip():
            return empty

        embedding, embed_tokens = await self._embedding.embed_text(user_text)
        if embedding is None:
            return empty

        cfg = self._config

        memories, media_content, memory_chunks = await asyncio.gather(
            self._query_memories(pool, embedding, user_id, user_text, agent_id=agent_id, customer_id=customer_id),
            self._query_media_content(pool, embedding, user_id, user_text, agent_id=agent_id, customer_id=customer_id),
            self._query_chunks(pool, embedding, user_id, user_text, agent_id=agent_id, customer_id=customer_id),
        )

        # Cross-type MMR
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

        ledgered_ids = ledger.ledgered_ids if ledger else set()

        if not memories and not media_content and not memory_chunks:
            return RetrievalResult(embed_tokens=embed_tokens)

        context = _format_memory_context(
            memories,
            media_content,
            memory_chunks,
            detail_threshold=cfg.detail_threshold,
            ledgered_ids=ledgered_ids,
        )

        # Update ledger with newly surfaced items
        if ledger and pool:
            for m in memories:
                mid = str(m["memory_id"])
                if mid not in ledgered_ids:
                    desc = m.get("summary") or m["content"]
                    await ledger.add_ref(pool, UUID(int=0), mid, "memory", desc)
            for mc in media_content:
                cid = str(mc["content_id"])
                if cid not in ledgered_ids:
                    desc = mc.get("title") or mc.get("summary") or mc["content"]
                    await ledger.add_ref(pool, UUID(int=0), cid, "media", desc)
            for chunk in memory_chunks:
                ckid = str(chunk["chunk_id"])
                if ckid not in ledgered_ids:
                    desc = chunk.get("title") or chunk.get("summary") or chunk["content"]
                    await ledger.add_ref(pool, UUID(int=0), ckid, "chunk", desc)

        return RetrievalResult(
            context=context or None,
            memories=memories,
            media_content=media_content,
            memory_chunks=memory_chunks,
            embed_tokens=embed_tokens,
        )

    async def _query_memories(
        self,
        pool: Any,
        embedding: list[float],
        user_id: UUID,
        user_text: str,
        *,
        agent_id: UUID | None = None,
        customer_id: UUID | None = None,
    ) -> list[dict[str, Any]]:
        """Query memories with parallel vector + FTS, three-signal hybrid scoring.

        :param pool: database connection pool
        :ptype pool: Any
        :param embedding: query embedding vector
        :ptype embedding: list[float]
        :param user_id: user whose memories to search
        :ptype user_id: UUID
        :param user_text: raw query text for FTS
        :ptype user_text: str
        :param agent_id: optional agent ID scope for filtering
        :ptype agent_id: UUID | None
        :param customer_id: optional customer ID scope for filtering
        :ptype customer_id: UUID | None
        :return: scored memory candidates
        :rtype: list[dict[str, Any]]
        """
        cfg = self._config
        embedding_str = json.dumps(embedding)
        top_k = cfg.top_k
        candidate_limit = top_k * cfg.candidate_multiplier
        weights = cfg.signal_weights

        scope_conditions, scope_params, param_offset = _build_scope_clause(
            user_id,
            agent_id=agent_id,
            customer_id=customer_id,
            start_param=2,
        )

        vec_where = f"WHERE {scope_conditions} AND is_deleted = false"
        limit_param = f"${param_offset + 1}"

        vec_coro = pool.fetch(
            f"""
            SELECT memory_id, content, summary, type_memory, date_created,
                   embedding,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM memories
            {vec_where}
            ORDER BY embedding <=> $1::vector
            LIMIT {limit_param}
            """,
            embedding_str,
            *scope_params,
            candidate_limit,
        )

        fts_text = _build_fts_query(user_text, cfg.fts_min_query_len, cfg.fts_max_query_len)
        if fts_text:
            fts_scope_conditions, fts_scope_params, fts_param_offset = _build_scope_clause(
                user_id,
                agent_id=agent_id,
                customer_id=customer_id,
                start_param=2,
            )
            fts_where = f"WHERE {fts_scope_conditions} AND is_deleted = false"
            fts_limit_param = f"${fts_param_offset + 1}"
            fts_coro = pool.fetch(
                f"""
                SELECT memory_id, content, summary, type_memory, date_created,
                       embedding,
                       ts_rank_cd(search_vector, websearch_to_tsquery('english', $1)) AS fts_rank
                FROM memories
                {fts_where}
                  AND search_vector @@ websearch_to_tsquery('english', $1)
                ORDER BY fts_rank DESC
                LIMIT {fts_limit_param}
                """,
                fts_text,
                *fts_scope_params,
                candidate_limit,
            )
            vec_rows, fts_rows = await asyncio.gather(vec_coro, fts_coro)
        else:
            vec_rows = await vec_coro
            fts_rows = []

        merged: dict[Any, dict[str, Any]] = {}
        for row in vec_rows:
            mid = row["memory_id"]
            emb = row["embedding"]
            if isinstance(emb, str):
                emb = json.loads(emb)
            merged[mid] = {
                "memory_id": mid,
                "content": row["content"],
                "summary": row["summary"],
                "type_memory": row["type_memory"],
                "date_created": row["date_created"],
                "similarity": float(row["similarity"]),
                "fts_rank": 0.0,
                "embedding": emb,
            }
        for row in fts_rows:
            mid = row["memory_id"]
            if mid in merged:
                merged[mid]["fts_rank"] = float(row["fts_rank"])
            else:
                emb = row["embedding"]
                if isinstance(emb, str):
                    emb = json.loads(emb)
                merged[mid] = {
                    "memory_id": mid,
                    "content": row["content"],
                    "summary": row["summary"],
                    "type_memory": row["type_memory"],
                    "date_created": row["date_created"],
                    "similarity": 0.0,
                    "fts_rank": float(row["fts_rank"]),
                    "embedding": emb,
                }

        candidates = list(merged.values())
        if not candidates:
            return []

        _normalize_fts_scores(candidates)

        for c in candidates:
            recency = _recency_decay(c["date_created"], cfg.recency_half_life_hours)
            c["recency"] = round(recency, 4)
            c["hybrid_score"] = round(
                weights["semantic"] * c["similarity"]
                + weights["keyword"] * c["fts_rank"]
                + weights["recency"] * recency,
                4,
            )

        filtered = [c for c in candidates if c["hybrid_score"] > cfg.similarity_threshold]
        filtered.sort(key=lambda m: m["hybrid_score"], reverse=True)
        return filtered[:top_k]

    async def _query_media_content(
        self,
        pool: Any,
        embedding: list[float],
        user_id: UUID,
        user_text: str,
        *,
        agent_id: UUID | None = None,
        customer_id: UUID | None = None,
    ) -> list[dict[str, Any]]:
        """Query media_content with parallel vector + FTS, three-signal hybrid scoring.

        :param pool: database connection pool
        :ptype pool: Any
        :param embedding: query embedding vector
        :ptype embedding: list[float]
        :param user_id: user whose media content to search
        :ptype user_id: UUID
        :param user_text: raw query text for FTS
        :ptype user_text: str
        :param agent_id: optional agent ID scope for filtering
        :ptype agent_id: UUID | None
        :param customer_id: optional customer ID scope for filtering
        :ptype customer_id: UUID | None
        :return: scored media content candidates
        :rtype: list[dict[str, Any]]
        """
        cfg = self._config
        embedding_str = json.dumps(embedding)
        top_k = cfg.media_top_k
        candidate_limit = top_k * cfg.media_candidate_multiplier
        weights = cfg.signal_weights

        scope_conditions, scope_params, param_offset = _build_scope_clause(
            user_id,
            agent_id=agent_id,
            customer_id=customer_id,
            start_param=2,
            table_prefix="mc",
        )
        limit_param = f"${param_offset + 1}"

        vec_coro = pool.fetch(
            f"""
            SELECT mc.content_id, mc.content, mc.summary, mc.content_type,
                   mc.media_id, mc.date_created, mc.embedding,
                   med.media_category, med.metadata_json,
                   1 - (mc.embedding <=> $1::vector) AS similarity
            FROM media_content mc
            JOIN media med ON mc.media_id = med.media_id
            WHERE {scope_conditions} AND mc.embedding IS NOT NULL
            ORDER BY mc.embedding <=> $1::vector
            LIMIT {limit_param}
            """,
            embedding_str,
            *scope_params,
            candidate_limit,
        )

        fts_text = _build_fts_query(user_text, cfg.fts_min_query_len, cfg.fts_max_query_len)
        if fts_text:
            fts_scope_conditions, fts_scope_params, fts_param_offset = _build_scope_clause(
                user_id,
                agent_id=agent_id,
                customer_id=customer_id,
                start_param=2,
                table_prefix="mc",
            )
            fts_limit_param = f"${fts_param_offset + 1}"
            fts_coro = pool.fetch(
                f"""
                SELECT mc.content_id, mc.content, mc.summary, mc.content_type,
                       mc.media_id, mc.date_created, mc.embedding,
                       med.media_category, med.metadata_json,
                       ts_rank_cd(mc.search_vector, websearch_to_tsquery('english', $1)) AS fts_rank
                FROM media_content mc
                JOIN media med ON mc.media_id = med.media_id
                WHERE {fts_scope_conditions} AND mc.embedding IS NOT NULL
                  AND mc.search_vector @@ websearch_to_tsquery('english', $1)
                ORDER BY fts_rank DESC
                LIMIT {fts_limit_param}
                """,
                fts_text,
                *fts_scope_params,
                candidate_limit,
            )
            vec_rows, fts_rows = await asyncio.gather(vec_coro, fts_coro)
        else:
            vec_rows = await vec_coro
            fts_rows = []

        merged: dict[Any, dict[str, Any]] = {}
        for row in vec_rows:
            cid = row["content_id"]
            meta = row["metadata_json"]
            if isinstance(meta, str):
                meta = json.loads(meta)
            title = (
                (meta or {}).get("document_title") or (meta or {}).get("original_filename") or (meta or {}).get("title")
            )
            emb = row["embedding"]
            if isinstance(emb, str):
                emb = json.loads(emb)
            merged[cid] = {
                "content_id": cid,
                "content": row["content"],
                "summary": row["summary"],
                "content_type": row["content_type"],
                "media_id": str(row["media_id"]),
                "media_category": row["media_category"],
                "title": title,
                "date_created": row["date_created"],
                "similarity": float(row["similarity"]),
                "fts_rank": 0.0,
                "embedding": emb,
            }
        for row in fts_rows:
            cid = row["content_id"]
            if cid in merged:
                merged[cid]["fts_rank"] = float(row["fts_rank"])
            else:
                meta = row["metadata_json"]
                if isinstance(meta, str):
                    meta = json.loads(meta)
                title = (
                    (meta or {}).get("document_title")
                    or (meta or {}).get("original_filename")
                    or (meta or {}).get("title")
                )
                emb = row["embedding"]
                if isinstance(emb, str):
                    emb = json.loads(emb)
                merged[cid] = {
                    "content_id": cid,
                    "content": row["content"],
                    "summary": row["summary"],
                    "content_type": row["content_type"],
                    "media_id": str(row["media_id"]),
                    "media_category": row["media_category"],
                    "title": title,
                    "date_created": row["date_created"],
                    "similarity": 0.0,
                    "fts_rank": float(row["fts_rank"]),
                    "embedding": emb,
                }

        candidates = list(merged.values())
        if not candidates:
            return []

        _normalize_fts_scores(candidates)

        for c in candidates:
            recency = _recency_decay(c["date_created"], cfg.recency_half_life_hours)
            c["recency"] = round(recency, 4)
            c["hybrid_score"] = round(
                weights["semantic"] * c["similarity"]
                + weights["keyword"] * c["fts_rank"]
                + weights["recency"] * recency,
                4,
            )

        filtered = [c for c in candidates if c["hybrid_score"] > cfg.similarity_threshold]
        filtered.sort(key=lambda c: c["hybrid_score"], reverse=True)
        return filtered[:top_k]

    async def _query_chunks(
        self,
        pool: Any,
        embedding: list[float],
        user_id: UUID,
        user_text: str,
        *,
        agent_id: UUID | None = None,
        customer_id: UUID | None = None,
    ) -> list[dict[str, Any]]:
        """Query memory_chunks with parallel vector + FTS, two-signal hybrid scoring.

        :param pool: database connection pool
        :ptype pool: Any
        :param embedding: query embedding vector
        :ptype embedding: list[float]
        :param user_id: user whose chunks to search
        :ptype user_id: UUID
        :param user_text: raw query text for FTS
        :ptype user_text: str
        :param agent_id: optional agent ID scope for filtering
        :ptype agent_id: UUID | None
        :param customer_id: optional customer ID scope for filtering
        :ptype customer_id: UUID | None
        :return: scored chunk candidates
        :rtype: list[dict[str, Any]]
        """
        cfg = self._config
        embedding_str = json.dumps(embedding)
        candidate_k = cfg.chunk_candidate_k
        chunk_weights = cfg.chunk_signal_weights

        scope_conditions, scope_params, param_offset = _build_scope_clause(
            user_id,
            agent_id=agent_id,
            customer_id=customer_id,
            start_param=2,
            table_prefix="mc",
        )
        limit_param = f"${param_offset + 1}"

        vec_coro = pool.fetch(
            f"""
            SELECT mc.chunk_id, mc.content, mc.summary, mc.heading_context,
                   mc.page_number, mc.media_id,
                   mc.embedding, med.metadata_json,
                   1 - (mc.embedding <=> $1::vector) AS similarity
            FROM memory_chunks mc
            LEFT JOIN media med ON mc.media_id = med.media_id
            WHERE {scope_conditions}
            ORDER BY mc.embedding <=> $1::vector
            LIMIT {limit_param}
            """,
            embedding_str,
            *scope_params,
            candidate_k,
        )

        fts_text = _build_fts_query(user_text, cfg.fts_min_query_len, cfg.fts_max_query_len)
        if fts_text:
            fts_scope_conditions, fts_scope_params, fts_param_offset = _build_scope_clause(
                user_id,
                agent_id=agent_id,
                customer_id=customer_id,
                start_param=2,
                table_prefix="mc",
            )
            fts_limit_param = f"${fts_param_offset + 1}"
            fts_coro = pool.fetch(
                f"""
                SELECT mc.chunk_id, mc.content, mc.summary, mc.heading_context,
                       mc.page_number, mc.media_id,
                       mc.embedding, med.metadata_json,
                       ts_rank_cd(mc.search_vector, websearch_to_tsquery('english', $1)) AS fts_rank
                FROM memory_chunks mc
                LEFT JOIN media med ON mc.media_id = med.media_id
                WHERE {fts_scope_conditions}
                  AND mc.search_vector @@ websearch_to_tsquery('english', $1)
                ORDER BY fts_rank DESC
                LIMIT {fts_limit_param}
                """,
                fts_text,
                *fts_scope_params,
                candidate_k,
            )
            vec_rows, fts_rows = await asyncio.gather(vec_coro, fts_coro)
        else:
            vec_rows = await vec_coro
            fts_rows = []

        merged: dict[Any, dict[str, Any]] = {}
        for row in vec_rows:
            ckid = row["chunk_id"]
            title = None
            meta = row["metadata_json"]
            if meta:
                if isinstance(meta, str):
                    meta = json.loads(meta)
                title = meta.get("document_title") or meta.get("original_filename")
            emb = row["embedding"]
            if isinstance(emb, str):
                emb = json.loads(emb)
            merged[ckid] = {
                "chunk_id": ckid,
                "content": row["content"],
                "summary": row["summary"],
                "heading_context": row["heading_context"],
                "page_number": row["page_number"],
                "media_id": str(row["media_id"]) if row["media_id"] else None,
                "title": title,
                "similarity": float(row["similarity"]),
                "fts_rank": 0.0,
                "embedding": emb,
            }
        for row in fts_rows:
            ckid = row["chunk_id"]
            if ckid in merged:
                merged[ckid]["fts_rank"] = float(row["fts_rank"])
            else:
                title = None
                meta = row["metadata_json"]
                if meta:
                    if isinstance(meta, str):
                        meta = json.loads(meta)
                    title = meta.get("document_title") or meta.get("original_filename")
                emb = row["embedding"]
                if isinstance(emb, str):
                    emb = json.loads(emb)
                merged[ckid] = {
                    "chunk_id": ckid,
                    "content": row["content"],
                    "summary": row["summary"],
                    "heading_context": row["heading_context"],
                    "page_number": row["page_number"],
                    "media_id": str(row["media_id"]) if row["media_id"] else None,
                    "title": title,
                    "similarity": 0.0,
                    "fts_rank": float(row["fts_rank"]),
                    "embedding": emb,
                }

        candidates = list(merged.values())
        if not candidates:
            return []

        _normalize_fts_scores(candidates)

        for c in candidates:
            c["hybrid_score"] = round(
                chunk_weights["semantic"] * c["similarity"] + chunk_weights["keyword"] * c["fts_rank"],
                4,
            )

        filtered = [c for c in candidates if c["hybrid_score"] > cfg.similarity_threshold]
        filtered.sort(key=lambda c: c["hybrid_score"], reverse=True)
        return filtered
