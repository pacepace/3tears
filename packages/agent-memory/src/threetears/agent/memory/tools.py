"""Memory tools -- LangChain tools for searching and recalling memories.

Provides ``load_memory_search_tool`` and ``load_recall_memory_tool`` as
reusable factories. Callers supply a connection pool, user_id, and an
optional ``EmbeddingProvider`` for semantic search.

Optional ``ledger_callback`` lets the host application track which items
the LLM has seen (e.g., MetaLLM's ToolContextManager).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field, model_validator

from threetears.agent.memory.authorize import (
    ACTION_MEMORY_READ,
    ACTION_MEMORY_WRITE,
    MemoryAccessDenied,
    MemoryAuthorizerDependencies,
    authorize_memory_access,
)
from threetears.agent.memory.embedding import EmbeddingProvider
from threetears.agent.memory.types import MemoryType
from threetears.observe import get_logger

__all__ = [
    "AddMemoryInput",
    "LedgerCallback",
    "MemorySearchInput",
    "RecallMemoryInput",
    "load_add_memory_tool",
    "load_memory_search_tool",
    "load_recall_memory_tool",
]


"""memory tools bind per-call identity through closure.

unlike workspace / MCP tools that ride remote dispatch envelopes,
the memory LangChain tools are built fresh for each conversation
by the agent bootstrap's MemoryIntegration. the ``user_id``,
``conversation_id``, ``agent_id`` and ``customer_id`` are known
at factory-call time and baked into the closure — every
invocation of the returned tool runs against that bound identity.

authorization therefore runs against the closure-bound
``(agent_id, customer_id, user_id)`` rather than reading from a
remote call scope. the agent identity in ``caller_agent_id`` is
the same agent that owns the memory namespace, so the evaluator's
owner short-circuit fires for agent-internal use and the user
side constrains to the closure-bound user.
"""

log = get_logger(__name__)

# -- Defaults -----------------------------------------------------------------

_MAX_RESULTS = 10
_DEFAULT_SIMILARITY_THRESHOLD = 0.2

# Type alias for the optional ledger callback
LedgerCallback = Callable[[str, str, str], Awaitable[None]]  # (item_id, item_type, description)


# -- Input schemas ------------------------------------------------------------


class MemorySearchInput(BaseModel):
    """Input schema for memory_search tool."""

    query: str = Field(
        default="",
        description="Natural language search query to find relevant memories",
    )
    type_filter: str | None = Field(
        default=None,
        description=(
            "Optional filter by memory type: 'preference', 'fact', 'decision', or 'topical_context'. "
            "If omitted, searches all types."
        ),
    )
    ids: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of item IDs (UUIDs) for direct lookup. "
            "Retrieves memories, media content, and document chunks by exact ID. "
            "Bypasses semantic search — use this when you know specific IDs from the conversation ledger."
        ),
    )

    @model_validator(mode="after")
    def _require_query_or_ids(self) -> "MemorySearchInput":
        if not self.query and not self.ids:
            raise ValueError("At least one of 'query' or 'ids' must be provided.")
        return self


class RecallMemoryInput(BaseModel):
    """Input schema for recall_memory tool."""

    id: str = Field(description="The UUID of the item to recall full content for")
    type: str = Field(description="Type of item: 'memory', 'media', or 'chunk'")


# -- Helpers ------------------------------------------------------------------


def _tool_error(tool_name: str, action: str, error: str) -> str:
    """Format a structured error string for tool results."""
    return f"[TOOL ERROR] {tool_name}: {action} failed — {error}"


def _fmt_dt(dt: Any) -> str:
    """Format a datetime or None into a short timestamp string."""
    if dt is None:
        return ""
    if hasattr(dt, "strftime"):
        return dt.strftime("%b %-d, %Y at %-I:%M %p")  # type: ignore[no-any-return]
    return str(dt)


def _extract_title(metadata_json: Any) -> str | None:
    """Extract title from metadata_json (may be a string or dict)."""
    meta = metadata_json
    if isinstance(meta, str):
        meta = json.loads(meta)
    if not meta:
        return None
    return meta.get("document_title") or meta.get("original_filename") or meta.get("title")  # type: ignore[no-any-return]


async def _ensure_first_write_owner_assignment(
    *,
    pool: Any,
    authorizer: MemoryAuthorizerDependencies,
    ns_row: Any,
    user_id: UUID,
) -> None:
    """ensure MemoryOwner assignment exists on first write by this user.

    checks if the user has any existing memory rows for this
    (agent, customer) pair. if yes → this is not a first write,
    skip the ensurer (grant either exists from earlier, or was
    explicitly revoked and must stay revoked). if no → first
    write, invoke the ensurer which is idempotent on the group
    row and the assignment row.

    the existence check + ensurer call is NOT transactional; a
    concurrent second-first-write will have both callers invoke
    the ensurer, but the ensurer's per-row insert-if-absent
    semantics make the double-call harmless.

    :param pool: asyncpg pool for the existence query
    :ptype pool: Any
    :param authorizer: rbac authorizer dependency bundle
    :ptype authorizer: MemoryAuthorizerDependencies
    :param ns_row: resolved memory namespace row
    :ptype ns_row: Any
    :param user_id: user UUID asking to write their first memory
    :ptype user_id: UUID
    :return: nothing
    :rtype: None
    """
    existing = await pool.fetchval(
        "SELECT EXISTS(SELECT 1 FROM memories WHERE user_id = $1)",
        user_id,
    )
    if existing:
        return None
    await authorizer.assignment_ensurer(user_id, ns_row)
    return None


# -- Memory search by IDs (batch lookup) --------------------------------------


async def _search_by_ids(
    pool: Any,
    user_id: UUID,
    ids: list[str],
    ledger_callback: LedgerCallback | None = None,
) -> str:
    """Batch-lookup memories, media content, and chunks by ID."""
    valid_uuids: list[UUID] = []
    for raw_id in ids:
        try:
            valid_uuids.append(UUID(raw_id))
        except ValueError:
            pass
    if not valid_uuids:
        return "No valid UUIDs provided."

    parts: list[str] = []

    try:
        mem_rows, mc_rows, chunk_rows = await asyncio.gather(
            pool.fetch(
                """SELECT memory_id, type_memory, content, date_created
                   FROM memories
                   WHERE memory_id = ANY($1::uuid[]) AND user_id = $2 AND is_deleted = false""",
                valid_uuids,
                user_id,
            ),
            pool.fetch(
                """SELECT mc.content_id, mc.content, mc.content_type, mc.media_id,
                          med.media_category, med.metadata_json, med.date_created
                   FROM media_content mc
                   JOIN media med ON mc.media_id = med.media_id
                   WHERE mc.content_id = ANY($1::uuid[]) AND mc.user_id = $2""",
                valid_uuids,
                user_id,
            ),
            pool.fetch(
                """SELECT mc.chunk_id, mc.content, mc.heading_context, mc.page_number,
                          med.metadata_json
                   FROM memory_chunks mc
                   LEFT JOIN media med ON mc.media_id = med.media_id
                   WHERE mc.chunk_id = ANY($1::uuid[]) AND mc.user_id = $2""",
                valid_uuids,
                user_id,
            ),
        )
    except Exception as exc:
        return _tool_error("memory_search", "id_lookup", str(exc))

    if mem_rows:
        parts.append(f"Found {len(mem_rows)} memories:")
        for row in mem_rows:
            mid = str(row["memory_id"])
            parts.append(f"- [mem:{mid}] [{row['type_memory']}] {row['content']}")
            if ledger_callback:
                await ledger_callback(mid, "memory", row["content"])

    if mc_rows:
        if parts:
            parts.append("")
        parts.append(f"Found {len(mc_rows)} media items:")
        for row in mc_rows:
            cid = str(row["content_id"])
            title = _extract_title(row["metadata_json"])
            label = title or row.get("media_category", "media")
            content = row["content"]
            if len(content) > 500:
                content = content[:497] + "..."
            parts.append(f"- [media:{cid}] [{label}] {content}")
            if ledger_callback:
                await ledger_callback(cid, "media", label or content)

    if chunk_rows:
        if parts:
            parts.append("")
        parts.append(f"Found {len(chunk_rows)} document chunks:")
        for row in chunk_rows:
            ckid = str(row["chunk_id"])
            title = _extract_title(row["metadata_json"])
            loc_parts = []
            if title:
                loc_parts.append(f'"{title}"')
            if row.get("page_number"):
                loc_parts.append(f"p.{row['page_number']}")
            if row.get("heading_context"):
                loc_parts.append(f'"{row["heading_context"]}"')
            location = " ".join(loc_parts) if loc_parts else "unknown source"
            parts.append(f"- [chunk:{ckid}] [from {location}] {row['content']}")
            if ledger_callback:
                await ledger_callback(ckid, "chunk", title or row["content"])

    if not parts:
        return "No items found for the provided IDs."

    found_ids = set()
    for row in mem_rows:
        found_ids.add(str(row["memory_id"]))
    for row in mc_rows:
        found_ids.add(str(row["content_id"]))
    for row in chunk_rows:
        found_ids.add(str(row["chunk_id"]))
    missing = [str(u) for u in valid_uuids if str(u) not in found_ids]
    if missing:
        parts.append("")
        parts.append(f"Not found: {', '.join(missing)}")

    return "\n".join(parts)


# -- Memory search tool factory -----------------------------------------------


async def load_memory_search_tool(
    pool: Any,
    user_id: UUID,
    embedding_provider: EmbeddingProvider,
    *,
    similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
    max_results: int = _MAX_RESULTS,
    ledger_callback: LedgerCallback | None = None,
    agent_id: UUID | None = None,
    customer_id: UUID | None = None,
    authorizer: MemoryAuthorizerDependencies | None = None,
) -> list[BaseTool]:
    """create a memory_search tool bound to the given user and pool.

    when ``authorizer`` is supplied, every invocation resolves
    the per-(agent, customer) memory namespace and evaluates
    ``memory.read`` for the effective caller. the effective
    caller identity is read from the active
    :class:`~threetears.agent.tools.call_scope.ToolCallScope` at
    call time when available; the closure's ``user_id`` is the
    fallback (kept for tests that pre-date phase 3).

    returns a single-element list (matching LangChain's
    bind_tools convention), or an empty list if preconditions
    aren't met.

    :param pool: asyncpg connection pool
    :ptype pool: Any
    :param user_id: user whose memories to search (row filter
        fallback when no scope is active)
    :ptype user_id: UUID
    :param embedding_provider: embedding provider for query vectors
    :ptype embedding_provider: EmbeddingProvider
    :param similarity_threshold: similarity cutoff for results
    :ptype similarity_threshold: float
    :param max_results: cap on returned results per source
    :ptype max_results: int
    :param ledger_callback: optional coroutine invoked per surfaced
        item with ``(item_id, item_type, description)``
    :ptype ledger_callback: LedgerCallback | None
    :param agent_id: owning agent UUID; required when
        ``authorizer`` is supplied (scopes the memory namespace)
    :ptype agent_id: UUID | None
    :param customer_id: owning customer UUID; required when
        ``authorizer`` is supplied
    :ptype customer_id: UUID | None
    :param authorizer: rbac authorizer dependency bundle enabling
        per-call ``memory.read`` enforcement; ``None`` preserves
        legacy column-filter-only behaviour
    :ptype authorizer: MemoryAuthorizerDependencies | None
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """
    sim_threshold = similarity_threshold

    @tool("memory_search", args_schema=MemorySearchInput)
    async def memory_search(
        query: str = "",
        type_filter: str | None = None,
        ids: list[str] | None = None,
    ) -> str:
        """search the user's memories and document knowledge base."""
        # rbac: evaluate memory.read against the closure-bound
        # (agent_id, customer_id) namespace for the closure-bound
        # user. the owner short-circuit fires for agent-internal
        # reads (caller_agent_id == owner_agent_id == agent_id).
        if authorizer is not None and agent_id is not None and customer_id is not None:
            try:
                await authorize_memory_access(
                    action=ACTION_MEMORY_READ,
                    agent_id=agent_id,
                    customer_id=customer_id,
                    caller_user_id=user_id,
                    caller_agent_id=agent_id,
                    deps=authorizer,
                )
            except MemoryAccessDenied as exc:
                return _tool_error("memory_search", "authorize", str(exc))

        # -- Batch ID lookup path --
        if ids:
            return await _search_by_ids(pool, user_id, ids, ledger_callback)

        if not query:
            return "Either 'query' or 'ids' must be provided."

        # -- Semantic search path --
        embedding, _tokens = await embedding_provider.embed_text(query)
        if embedding is None:
            return _tool_error("memory_search", "embed", "Embedding API request failed")

        embedding_str = json.dumps(embedding)
        fts_text = query.strip()[:500] if len(query.strip()) >= 3 else None

        # 1. Search user memories
        params: list[Any] = [embedding_str, user_id]
        conditions = ["user_id = $2", "is_deleted = false"]
        param_idx = 3

        if type_filter:
            valid_types = {"preference", "fact", "decision", "topical_context"}
            if type_filter not in valid_types:
                return _tool_error(
                    "memory_search",
                    "filter",
                    f"Invalid type_filter '{type_filter}'. Must be one of: {', '.join(sorted(valid_types))}",
                )
            conditions.append(f"type_memory = ${param_idx}")
            params.append(type_filter)
            param_idx += 1

        where_clause = " AND ".join(conditions)
        query_sql = f"""
            SELECT memory_id, type_memory, content, date_created,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM memories
            WHERE {where_clause}
            ORDER BY embedding <=> $1::vector
            LIMIT ${param_idx}
        """
        params.append(max_results)

        try:
            rows = await pool.fetch(query_sql, *params)
        except Exception as exc:
            return _tool_error("memory_search", "query", str(exc))

        memories = [
            {
                "memory_id": str(row["memory_id"]),
                "type": row["type_memory"],
                "content": row["content"],
                "date_created": row["date_created"],
                "similarity": float(row["similarity"]),
            }
            for row in rows
            if float(row["similarity"]) > sim_threshold
        ]
        seen_ids = {m["memory_id"] for m in memories}

        # FTS keyword results
        if fts_text:
            fts_conditions = [
                "user_id = $2",
                "is_deleted = false",
                "search_vector @@ websearch_to_tsquery('english', $1)",
            ]
            fts_params: list[Any] = [fts_text, user_id]
            fts_idx = 3
            if type_filter:
                fts_conditions.append(f"type_memory = ${fts_idx}")
                fts_params.append(type_filter)
                fts_idx += 1
            fts_where = " AND ".join(fts_conditions)
            try:
                fts_rows = await pool.fetch(
                    f"""
                    SELECT memory_id, type_memory, content, date_created,
                           ts_rank_cd(search_vector, websearch_to_tsquery('english', $1)) AS fts_rank
                    FROM memories
                    WHERE {fts_where}
                    ORDER BY fts_rank DESC
                    LIMIT {max_results}
                    """,
                    *fts_params,
                )
                for row in fts_rows:
                    mid = str(row["memory_id"])
                    if mid not in seen_ids:
                        seen_ids.add(mid)
                        memories.append(
                            {
                                "memory_id": mid,
                                "type": row["type_memory"],
                                "content": row["content"],
                                "date_created": row["date_created"],
                                "similarity": None,
                            }
                        )
            except Exception:
                pass  # FTS search is best-effort

        # 2. Search media_content
        media_results: list[dict[str, Any]] = []
        try:
            mc_rows = await pool.fetch(
                """
                SELECT mc.content_id, mc.content, mc.content_type,
                       mc.media_id, med.media_category, med.metadata_json,
                       med.date_created,
                       1 - (mc.embedding <=> $1::vector) AS similarity
                FROM media_content mc
                JOIN media med ON mc.media_id = med.media_id
                WHERE mc.user_id = $2
                  AND mc.embedding IS NOT NULL
                ORDER BY mc.embedding <=> $1::vector
                LIMIT 5
                """,
                embedding_str,
                user_id,
            )
            mc_seen: set[str] = set()
            for row in mc_rows:
                sim = float(row["similarity"])
                if sim <= sim_threshold:
                    continue
                cid = str(row["content_id"])
                mc_seen.add(cid)
                title = _extract_title(row["metadata_json"])
                media_results.append(
                    {
                        "content_id": cid,
                        "content": row["content"],
                        "media_id": str(row["media_id"]),
                        "media_category": row["media_category"],
                        "title": title,
                        "date_created": row["date_created"],
                        "similarity": sim,
                    }
                )

            # FTS for media_content
            if fts_text:
                fts_mc_rows = await pool.fetch(
                    """
                    SELECT mc.content_id, mc.content, mc.content_type,
                           mc.media_id, med.media_category, med.metadata_json,
                           med.date_created,
                           ts_rank_cd(mc.search_vector, websearch_to_tsquery('english', $1)) AS fts_rank
                    FROM media_content mc
                    JOIN media med ON mc.media_id = med.media_id
                    WHERE mc.user_id = $2
                      AND mc.embedding IS NOT NULL
                      AND mc.search_vector @@ websearch_to_tsquery('english', $1)
                    ORDER BY fts_rank DESC
                    LIMIT 5
                    """,
                    fts_text,
                    user_id,
                )
                for row in fts_mc_rows:
                    cid = str(row["content_id"])
                    if cid not in mc_seen:
                        mc_seen.add(cid)
                        title = _extract_title(row["metadata_json"])
                        media_results.append(
                            {
                                "content_id": cid,
                                "content": row["content"],
                                "media_id": str(row["media_id"]),
                                "media_category": row["media_category"],
                                "title": title,
                                "date_created": row["date_created"],
                                "similarity": None,
                            }
                        )
        except Exception:
            pass

        # 3. Search memory_chunks
        doc_chunks: list[dict[str, Any]] = []
        try:
            chunk_rows = await pool.fetch(
                """
                SELECT mc.chunk_id, mc.content, mc.heading_context, mc.page_number,
                       med.metadata_json,
                       1 - (mc.embedding <=> $1::vector) AS similarity
                FROM memory_chunks mc
                LEFT JOIN media med ON mc.media_id = med.media_id
                WHERE mc.user_id = $2
                ORDER BY mc.embedding <=> $1::vector
                LIMIT 5
                """,
                embedding_str,
                user_id,
            )
            for row in chunk_rows:
                sim = float(row["similarity"])
                if sim <= sim_threshold:
                    continue
                title = _extract_title(row["metadata_json"])
                doc_chunks.append(
                    {
                        "chunk_id": str(row["chunk_id"]),
                        "content": row["content"],
                        "title": title,
                        "heading": row["heading_context"],
                        "page": row["page_number"],
                        "similarity": sim,
                    }
                )
        except Exception:
            pass

        if not memories and not media_results and not doc_chunks:
            return "No relevant memories or documents found for this query."

        # Format results
        parts: list[str] = []
        if memories:
            parts.append(f"Found {len(memories)} relevant memories:")
            for m in memories:
                ts = _fmt_dt(m.get("date_created"))
                tag = f"[mem:{m['memory_id']}]"
                if ts:
                    parts.append(f"- {tag} [{m['type']}] [{ts}] {m['content']}")
                else:
                    parts.append(f"- {tag} [{m['type']}] {m['content']}")

        if media_results:
            parts.append("")
            parts.append(f"Found {len(media_results)} relevant media items:")
            for mr in media_results:
                ts = _fmt_dt(mr.get("date_created"))
                ts_suffix = f" (created: {ts})" if ts else ""
                tag = f"[media:{mr['content_id']}]"
                if mr.get("media_category") == "image":
                    parts.append(f"- {tag} {mr['content']}{ts_suffix}")
                else:
                    label = mr.get("title") or mr.get("media_category", "media")
                    parts.append(f"- {tag} [{label}] {mr['content']}{ts_suffix}")

        if doc_chunks:
            parts.append("")
            parts.append(f"Found {len(doc_chunks)} relevant document excerpts:")
            for c in doc_chunks:
                loc_parts = []
                if c.get("title"):
                    loc_parts.append(f'"{c["title"]}"')
                if c.get("page"):
                    loc_parts.append(f"p.{c['page']}")
                if c.get("heading"):
                    loc_parts.append(f'"{c["heading"]}"')
                location = " ".join(loc_parts) if loc_parts else "unknown source"
                tag = f"[chunk:{c['chunk_id']}]"
                parts.append(f"- {tag} [from {location}] {c['content']}")

        # Ledger updates
        if ledger_callback:
            for m in memories:
                await ledger_callback(m["memory_id"], "memory", m["content"])
            for mr in media_results:
                desc = mr.get("title") or mr["content"]
                await ledger_callback(mr["content_id"], "media", desc)
            for c in doc_chunks:
                desc = c.get("title") or c["content"]
                await ledger_callback(c["chunk_id"], "chunk", desc)

        return "\n".join(parts)

    memory_search.description = (
        "Search the user's stored memories and uploaded documents by semantic similarity. "
        "Use this to recall user preferences, past facts, decisions, and document content. "
        "Memories include preferences, facts, decisions, and topical context extracted from previous conversations. "
        "You can also pass 'ids' for direct batch lookup by UUID (bypasses embedding search)."
    )

    return [memory_search]


# -- Recall memory tool factory ------------------------------------------------


async def load_recall_memory_tool(
    pool: Any,
    user_id: UUID,
    *,
    agent_id: UUID | None = None,
    customer_id: UUID | None = None,
    authorizer: MemoryAuthorizerDependencies | None = None,
) -> list[BaseTool]:
    """create a recall_memory tool for full-content retrieval by ID.

    when ``authorizer`` is supplied, every invocation runs
    ``memory.read`` through the evaluator before the SQL fetch.
    identity is resolved as in :func:`load_memory_search_tool`.

    returns a single-element list, or empty list if preconditions
    aren't met.

    :param pool: asyncpg connection pool
    :ptype pool: Any
    :param user_id: user whose memories to recall (row filter fallback)
    :ptype user_id: UUID
    :param agent_id: owning agent UUID for rbac namespace lookup
    :ptype agent_id: UUID | None
    :param customer_id: owning customer UUID for rbac namespace lookup
    :ptype customer_id: UUID | None
    :param authorizer: rbac authorizer dependency bundle
    :ptype authorizer: MemoryAuthorizerDependencies | None
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """

    @tool("recall_memory", args_schema=RecallMemoryInput)
    async def recall_memory(id: str, type: str) -> str:
        """retrieve full content of a memory, media description, or chunk."""
        if authorizer is not None and agent_id is not None and customer_id is not None:
            try:
                await authorize_memory_access(
                    action=ACTION_MEMORY_READ,
                    agent_id=agent_id,
                    customer_id=customer_id,
                    caller_user_id=user_id,
                    caller_agent_id=agent_id,
                    deps=authorizer,
                )
            except MemoryAccessDenied as exc:
                return _tool_error("recall_memory", "authorize", str(exc))

        try:
            item_uuid = UUID(id)
        except ValueError:
            return f"Invalid ID format: {id}"

        item_type = type.lower().strip()

        if item_type == "memory":
            row = await pool.fetchrow(
                "SELECT content FROM memories WHERE memory_id = $1 AND user_id = $2 AND is_deleted = false",
                item_uuid,
                user_id,
            )
            if not row:
                return "Memory not found or access denied."
            return row["content"]  # type: ignore[no-any-return]

        elif item_type == "media":
            row = await pool.fetchrow(
                "SELECT content FROM media_content WHERE content_id = $1 AND user_id = $2",
                item_uuid,
                user_id,
            )
            if not row:
                return "Media content not found or access denied."
            content = row["content"]
            if len(content) > 12000:
                content = content[:12000] + "\n\n[Content truncated — full text is longer]"
            return content  # type: ignore[no-any-return]

        elif item_type == "chunk":
            row = await pool.fetchrow(
                "SELECT content FROM memory_chunks WHERE chunk_id = $1 AND user_id = $2",
                item_uuid,
                user_id,
            )
            if not row:
                return "Document chunk not found or access denied."
            return row["content"]  # type: ignore[no-any-return]

        else:
            return f"Unknown type '{type}'. Use 'memory', 'media', or 'chunk'."

    recall_memory.description = (
        "Retrieve the full content of a memory, media description, or document chunk. "
        "Use this when the system prompt shows a summary-only item (without '(detailed)' marker) "
        "and you need the complete content to answer the user's question accurately. "
        "Pass the ID from the [mem:ID], [media:ID], or [chunk:ID] tag and the type."
    )

    return [recall_memory]


# -- Add memory tool factory --------------------------------------------------

_VALID_MEMORY_TYPES = frozenset(t.value for t in MemoryType)


class AddMemoryInput(BaseModel):
    """Input schema for add_memory tool."""

    content: str = Field(
        description=(
            "The memory to store. Write as a concise, standalone fact about the user. "
            "Example: 'User prefers Rust as their primary programming language.'"
        ),
    )
    memory_type: str = Field(
        default="preference",
        description=(
            "Memory category: 'preference' (likes, dislikes, style choices), "
            "'fact' (biographical facts, skills, background), "
            "'decision' (choices the user has made), "
            "'topical_context' (ongoing projects, current focus areas), "
            "or 'relational_context' (relationship dynamics, communication preferences)."
        ),
    )


async def load_add_memory_tool(
    pool: Any,
    user_id: UUID,
    conversation_id: UUID,
    message_id: UUID,
    embedding_provider: EmbeddingProvider,
    *,
    similarity_dedup_threshold: float = 0.90,
    agent_id: UUID | None = None,
    customer_id: UUID | None = None,
    authorizer: MemoryAuthorizerDependencies | None = None,
) -> list[BaseTool]:
    """create an add_memory tool for explicit memory storage.

    the assistant calls this when the user explicitly asks it to
    remember something. the memory is embedded and stored
    immediately, with dedup against existing similar memories.

    when ``authorizer`` is supplied, every invocation runs
    ``memory.write`` through the evaluator before the SQL
    insert / update. on a successful write authored by a user
    (not the agent-owner short-circuit path), the per-user
    ``MemoryOwner`` assignment is ensured via
    :attr:`MemoryAuthorizerDependencies.assignment_ensurer` so
    subsequent reads from the same user hit a cached grant.

    returns a single-element list (matching LangChain's
    bind_tools convention).

    :param pool: asyncpg connection pool
    :ptype pool: Any
    :param user_id: owning user's UUID (row filter fallback)
    :ptype user_id: UUID
    :param conversation_id: current conversation UUID
    :ptype conversation_id: UUID
    :param message_id: current message UUID (source attribution)
    :ptype message_id: UUID
    :param embedding_provider: provider for embedding vectors
    :ptype embedding_provider: EmbeddingProvider
    :param similarity_dedup_threshold: if an existing memory
        exceeds this similarity score, the existing memory is
        updated instead of creating a duplicate. default 0.90
    :ptype similarity_dedup_threshold: float
    :param agent_id: owning agent UUID; required when
        ``authorizer`` is supplied
    :ptype agent_id: UUID | None
    :param customer_id: owning customer UUID; required when
        ``authorizer`` is supplied
    :ptype customer_id: UUID | None
    :param authorizer: rbac authorizer dependency bundle
    :ptype authorizer: MemoryAuthorizerDependencies | None
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """
    dedup_threshold = similarity_dedup_threshold

    @tool("add_memory", args_schema=AddMemoryInput)
    async def add_memory(content: str, memory_type: str = "preference") -> str:
        """store a memory about the user for future conversations."""
        # rbac: the add_memory tool is USER-initiated (the LLM
        # calls it on behalf of the user who asked to be
        # remembered). the act of writing a memory declares user
        # ownership of that slice of the memory namespace, so the
        # per-user MemoryOwner assignment is ensured BEFORE the
        # authorize check (idempotent — a no-op if the grant
        # already exists from a prior write). the evaluator then
        # sees the grant and authorizes the write; a user whose
        # grant was later revoked (admin action) correctly denies
        # because the ensurer only creates a fresh grant when the
        # user has never written before (not "every write
        # resurrects a revoked grant").
        #
        # first-write path: ensure creates the group + assignment,
        # authorize passes against the fresh grant, insert proceeds.
        # revoked-then-retry path: ensure is idempotent (row stays
        # deleted if the admin explicitly removed it — the ensurer
        # only inserts when absent AND the user has no existing
        # memory rows); authorize denies; caller gets clean error.
        ns_row = None
        if authorizer is not None and agent_id is not None and customer_id is not None:
            # resolve namespace first so the ensurer can bind to it.
            ns_row = await authorizer.namespace_resolver(agent_id, customer_id)
            if ns_row is None:
                return _tool_error(
                    "add_memory",
                    "authorize",
                    f"memory namespace for agent={agent_id} "
                    f"customer={customer_id} could not be resolved",
                )
            try:
                await _ensure_first_write_owner_assignment(
                    pool=pool,
                    authorizer=authorizer,
                    ns_row=ns_row,
                    user_id=user_id,
                )
            except Exception as exc:
                log.warning(
                    "add_memory: first-write assignment ensure failed",
                    extra={"extra_data": {"error": str(exc)}},
                )
            try:
                await authorize_memory_access(
                    action=ACTION_MEMORY_WRITE,
                    agent_id=agent_id,
                    customer_id=customer_id,
                    caller_user_id=user_id,
                    caller_agent_id=None,
                    deps=authorizer,
                )
            except MemoryAccessDenied as exc:
                return _tool_error("add_memory", "authorize", str(exc))

        # Validate type
        mt = memory_type.lower().strip()
        if mt not in _VALID_MEMORY_TYPES:
            return f"Invalid memory_type '{memory_type}'. Valid types: {', '.join(sorted(_VALID_MEMORY_TYPES))}"

        # Generate embedding
        try:
            embedding, _token_count = await embedding_provider.embed_text(content)
            if embedding is None:
                return _tool_error("add_memory", "embed", "embedding provider returned None")
        except Exception as exc:
            log.warning(
                "add_memory: embedding failed",
                extra={"extra_data": {"error": str(exc)}},
            )
            return _tool_error("add_memory", "embed", str(exc))

        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

        # Dedup: check for very similar existing memories
        try:
            similar_rows = await pool.fetch(
                """
                SELECT memory_id, content, type_memory,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM memories
                WHERE user_id = $2 AND is_deleted = false
                ORDER BY embedding <=> $1::vector
                LIMIT 3
                """,
                embedding_str,
                user_id,
            )

            for row in similar_rows:
                if float(row["similarity"]) >= dedup_threshold:
                    # Update existing memory instead of creating duplicate
                    existing_id = row["memory_id"]
                    from datetime import UTC, datetime

                    # YugabyteDB TIMESTAMP columns are timezone-naive; convert at
                    # the WRITE boundary per CLAUDE.md "Datetime Handling".
                    now = datetime.now(UTC).replace(tzinfo=None)
                    await pool.execute(
                        """
                        UPDATE memories
                        SET content = $1, type_memory = $2, embedding = $3::vector,
                            date_updated = $4
                        WHERE memory_id = $5
                        """,
                        content,
                        mt,
                        embedding_str,
                        now,
                        existing_id,
                    )
                    log.info(
                        "add_memory: updated existing similar memory",
                        extra={
                            "extra_data": {
                                "memory_id": str(existing_id),
                                "similarity": round(float(row["similarity"]), 3),
                                "old_content": row["content"][:100],
                            }
                        },
                    )
                    return f"Updated existing memory (was similar at {float(row['similarity']):.0%}): {content}"

        except Exception as exc:
            # Dedup check failed — proceed with insert anyway
            log.warning(
                "add_memory: dedup check failed, inserting new",
                extra={"extra_data": {"error": str(exc)}},
            )

        # Insert new memory
        try:
            from datetime import UTC, datetime
            from uuid import uuid4

            # YugabyteDB TIMESTAMP columns are timezone-naive; convert at the
            # WRITE boundary per CLAUDE.md "Datetime Handling".
            now = datetime.now(UTC).replace(tzinfo=None)
            memory_id = uuid4()
            await pool.execute(
                """
                INSERT INTO memories (
                    memory_id, user_id, conversation_id, message_id_source,
                    type_memory, content, embedding, is_deleted,
                    date_created, date_updated
                ) VALUES ($1, $2, $3, $4, $5, $6, $7::vector, false, $8, $8)
                """,
                memory_id,
                user_id,
                conversation_id,
                message_id,
                mt,
                content,
                embedding_str,
                now,
            )

            log.info(
                "add_memory: stored new memory",
                extra={
                    "extra_data": {
                        "memory_id": str(memory_id),
                        "type": mt,
                        "content": content[:100],
                    }
                },
            )
            return f"Remembered: {content}"

        except Exception as exc:
            log.warning(
                "add_memory: insert failed",
                extra={"extra_data": {"error": str(exc)}},
            )
            return _tool_error("add_memory", "store", str(exc))

    add_memory.description = (
        "Store a memory about the user for future conversations. "
        "Use this when the user explicitly asks you to remember something "
        "(e.g., 'remember that I prefer...', 'my X is Y', 'don't forget...'). "
        "Write the memory as a concise, standalone fact. "
        "Duplicate detection is automatic — if a very similar memory already "
        "exists, it will be updated instead of creating a duplicate."
    )

    return [add_memory]
