"""Memory tools -- LangChain tools for searching and recalling memories.

Provides ``load_memory_search_tool``, ``load_recall_memory_tool``, and
``load_add_memory_tool`` as reusable factories. Callers supply the four
memory-package Collections (:class:`MemoriesCollection`,
:class:`MediaCollection`, :class:`MediaContentCollection`,
:class:`MemoryChunkCollection`) plus user_id, agent_id, customer_id, and
an authorizer. Every SQL site that used to live raw on a pool is now a
method call on one of the Collections; tool factories hold no pool
reference.

Optional ``ledger_callback`` lets the host application track which items
the LLM has seen.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field, model_validator

from threetears.agent.memory.authorize import (
    ACTION_MEMORY_READ,
    ACTION_MEMORY_WRITE,
    MEMORY_NAMESPACE_TYPE,
    MemoryAccessDenied,
    MemoryAuthorizerDependencies,
    authorize_memory_access,
    ensure_memory_owner_assignment,
)
from threetears.agent.memory.collections import (
    MediaCollection,
    MediaContentCollection,
    MemoriesCollection,
    MemoryChunkCollection,
)
from threetears.agent.memory.embedding import EmbeddingProvider
from threetears.agent.memory.embedding_utils import _safe_aembed_query
from threetears.agent.memory.entities import MemoryEntity
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


log = get_logger(__name__)

_MAX_RESULTS = 10
_DEFAULT_SIMILARITY_THRESHOLD = 0.2

# Type alias for the optional ledger callback
LedgerCallback = Callable[[str, str, str], Awaitable[None]]


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
        """raise ValidationError unless at least one of ``query`` / ``ids`` is set.

        :return: self
        :rtype: MemorySearchInput
        """
        if not self.query and not self.ids:
            raise ValueError("At least one of 'query' or 'ids' must be provided.")
        return self


class RecallMemoryInput(BaseModel):
    """Input schema for recall_memory tool."""

    id: str = Field(description="The UUID of the item to recall full content for")
    type: str = Field(description="Type of item: 'memory', 'media', or 'chunk'")


def _tool_error(tool_name: str, action: str, error: str) -> str:
    """Format a structured error string for tool results.

    :param tool_name: tool name emitting the error
    :ptype tool_name: str
    :param action: action that failed
    :ptype action: str
    :param error: error description
    :ptype error: str
    :return: formatted error string
    :rtype: str
    """
    return f"[TOOL ERROR] {tool_name}: {action} failed — {error}"


def _fmt_dt(dt: Any) -> str:
    """Format a datetime or None into a short timestamp string.

    :param dt: datetime or None
    :ptype dt: Any
    :return: formatted string or empty
    :rtype: str
    """
    if dt is None:
        return ""
    if hasattr(dt, "strftime"):
        return dt.strftime("%b %-d, %Y at %-I:%M %p")  # type: ignore[no-any-return]
    return str(dt)


def _extract_title(metadata_json: Any) -> str | None:
    """Extract title from metadata_json (may be a string or dict).

    :param metadata_json: JSONB payload (string or dict)
    :ptype metadata_json: Any
    :return: title or ``None``
    :rtype: str | None
    """
    meta = metadata_json
    if isinstance(meta, str):
        meta = json.loads(meta)
    if not meta:
        return None
    return meta.get("document_title") or meta.get("original_filename") or meta.get("title")  # type: ignore[no-any-return]


async def _ensure_first_write_owner_assignment(
    *,
    memories_collection: MemoriesCollection,
    authorizer: MemoryAuthorizerDependencies,
    ns_entity: Any,
    user_id: UUID,
) -> None:
    """ensure MemoryOwner assignment exists on first write by this user.

    delegates the existence probe to
    :meth:`MemoriesCollection.count_by_user` so the SQL site lives on
    the Collection. semantics match the previous raw-pool helper: if
    the user has ANY existing memory rows, skip the ensure (grant
    exists from earlier or was explicitly revoked). otherwise call
    :func:`ensure_memory_owner_assignment`, which is idempotent on the
    group / membership / assignment rows.

    :param memories_collection: three-tier memories collection
    :ptype memories_collection: MemoriesCollection
    :param authorizer: rbac authorizer dependency bundle
    :ptype authorizer: MemoryAuthorizerDependencies
    :param ns_entity: resolved memory namespace entity
    :ptype ns_entity: Any
    :param user_id: user UUID asking to write their first memory
    :ptype user_id: UUID
    :return: nothing
    :rtype: None
    """
    has_rows = await memories_collection.count_by_user(
        user_id,
        agent_id=ns_entity.owner_agent_id,
    )
    if has_rows:
        return None
    await ensure_memory_owner_assignment(
        user_id=user_id,
        namespace=ns_entity,
        deps=authorizer,
    )
    return None


async def _search_by_ids(
    memories_collection: MemoriesCollection,
    media_content_collection: MediaContentCollection,
    memory_chunk_collection: MemoryChunkCollection,
    user_id: UUID,
    ids: list[str],
    *,
    agent_id: UUID,
    ledger_callback: LedgerCallback | None = None,
) -> str:
    """Batch-lookup memories, media content, and chunks by ID via Collections.

    :param memories_collection: three-tier memories collection
    :ptype memories_collection: MemoriesCollection
    :param media_content_collection: three-tier media_content collection
    :ptype media_content_collection: MediaContentCollection
    :param memory_chunk_collection: three-tier memory_chunks collection
    :ptype memory_chunk_collection: MemoryChunkCollection
    :param user_id: owning user UUID
    :ptype user_id: UUID
    :param ids: list of raw UUID strings from the model
    :ptype ids: list[str]
    :param agent_id: partition column on every Collection touched here;
        required
    :ptype agent_id: UUID
    :param ledger_callback: optional surfaced-item notification
    :ptype ledger_callback: LedgerCallback | None
    :return: formatted tool output
    :rtype: str
    """
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
            memories_collection.search_by_ids(
                valid_uuids,
                user_id,
                agent_id=agent_id,
            ),
            media_content_collection.search_by_ids(
                valid_uuids,
                user_id,
                agent_id=agent_id,
            ),
            memory_chunk_collection.search_by_ids(
                valid_uuids,
                user_id,
                agent_id=agent_id,
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

    found_ids: set[str] = set()
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


async def load_memory_search_tool(
    user_id: UUID,
    embedding_provider: EmbeddingProvider,
    agent_id: UUID,
    customer_id: UUID,
    authorizer: MemoryAuthorizerDependencies,
    memories_collection: MemoriesCollection,
    media_content_collection: MediaContentCollection,
    memory_chunk_collection: MemoryChunkCollection,
    *,
    similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
    max_results: int = _MAX_RESULTS,
    ledger_callback: LedgerCallback | None = None,
) -> list[BaseTool]:
    """create a memory_search tool bound to the given user and Collections.

    :param user_id: user whose memories to search (row filter +
        evaluator ``caller_user_id``)
    :ptype user_id: UUID
    :param embedding_provider: embedding provider for query vectors
    :ptype embedding_provider: EmbeddingProvider
    :param agent_id: owning agent UUID (memory namespace owner)
    :ptype agent_id: UUID
    :param customer_id: owning customer UUID
    :ptype customer_id: UUID
    :param authorizer: rbac authorizer dependency bundle; required
    :ptype authorizer: MemoryAuthorizerDependencies
    :param memories_collection: three-tier memories collection
    :ptype memories_collection: MemoriesCollection
    :param media_content_collection: three-tier media_content collection
    :ptype media_content_collection: MediaContentCollection
    :param memory_chunk_collection: three-tier memory_chunks collection
    :ptype memory_chunk_collection: MemoryChunkCollection
    :param similarity_threshold: similarity cutoff for results
    :ptype similarity_threshold: float
    :param max_results: cap on returned results per source
    :ptype max_results: int
    :param ledger_callback: optional coroutine invoked per surfaced
        item with ``(item_id, item_type, description)``
    :ptype ledger_callback: LedgerCallback | None
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

        if ids:
            return await _search_by_ids(
                memories_collection,
                media_content_collection,
                memory_chunk_collection,
                user_id,
                ids,
                agent_id=agent_id,
                ledger_callback=ledger_callback,
            )

        if not query:
            return "Either 'query' or 'ids' must be provided."

        embedding = await _safe_aembed_query(embedding_provider, query)
        if embedding is None:
            return _tool_error("memory_search", "embed", "Embedding API request failed")

        fts_text = query.strip()[:500] if len(query.strip()) >= 3 else None

        if type_filter:
            valid_types = {"preference", "fact", "decision", "topical_context"}
            if type_filter not in valid_types:
                return _tool_error(
                    "memory_search",
                    "filter",
                    f"Invalid type_filter '{type_filter}'. Must be one of: {', '.join(sorted(valid_types))}",
                )

        try:
            memories = await memories_collection.search_by_semantic(
                user_id=user_id,
                agent_id=agent_id,
                embedding=embedding,
                max_results=max_results,
                similarity_threshold=sim_threshold,
                type_filter=type_filter,
            )
        except Exception as exc:
            return _tool_error("memory_search", "query", str(exc))

        seen_ids = {m["memory_id"] for m in memories}

        if fts_text:
            try:
                fts_rows = await memories_collection.search_by_fts(
                    user_id=user_id,
                    agent_id=agent_id,
                    fts_text=fts_text,
                    max_results=max_results,
                    type_filter=type_filter,
                )
                for row in fts_rows:
                    mid = row["memory_id"]
                    if mid not in seen_ids:
                        seen_ids.add(mid)
                        memories.append(
                            {
                                "memory_id": mid,
                                "type": row["type"],
                                "content": row["content"],
                                "date_created": row["date_created"],
                                "similarity": None,
                            }
                        )
            except Exception:
                pass

        media_results: list[dict[str, Any]] = []
        try:
            mc_rows = await media_content_collection.search_by_semantic(
                user_id=user_id,
                agent_id=agent_id,
                embedding=embedding,
                max_results=5,
                similarity_threshold=sim_threshold,
            )
            mc_seen: set[str] = set()
            for row in mc_rows:
                cid = row["content_id"]
                mc_seen.add(cid)
                title = _extract_title(row["metadata_json"])
                media_results.append(
                    {
                        "content_id": cid,
                        "content": row["content"],
                        "media_id": row["media_id"],
                        "media_category": row["media_category"],
                        "title": title,
                        "date_created": row["date_created"],
                        "similarity": row["similarity"],
                    }
                )
            if fts_text:
                fts_mc_rows = await media_content_collection.search_by_fts(
                    user_id=user_id,
                    agent_id=agent_id,
                    fts_text=fts_text,
                    max_results=5,
                )
                for row in fts_mc_rows:
                    cid = row["content_id"]
                    if cid not in mc_seen:
                        mc_seen.add(cid)
                        title = _extract_title(row["metadata_json"])
                        media_results.append(
                            {
                                "content_id": cid,
                                "content": row["content"],
                                "media_id": row["media_id"],
                                "media_category": row["media_category"],
                                "title": title,
                                "date_created": row["date_created"],
                                "similarity": None,
                            }
                        )
        except Exception:
            pass

        doc_chunks: list[dict[str, Any]] = []
        try:
            chunk_rows = await memory_chunk_collection.search_by_semantic(
                user_id=user_id,
                agent_id=agent_id,
                embedding=embedding,
                max_results=5,
                similarity_threshold=sim_threshold,
            )
            for row in chunk_rows:
                title = _extract_title(row["metadata_json"])
                doc_chunks.append(
                    {
                        "chunk_id": row["chunk_id"],
                        "content": row["content"],
                        "title": title,
                        "heading": row["heading_context"],
                        "page": row["page_number"],
                        "similarity": row["similarity"],
                    }
                )
        except Exception:
            pass

        if not memories and not media_results and not doc_chunks:
            return "No relevant memories or documents found for this query."

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


async def load_recall_memory_tool(
    user_id: UUID,
    agent_id: UUID,
    customer_id: UUID,
    authorizer: MemoryAuthorizerDependencies,
    memories_collection: MemoriesCollection,
    media_content_collection: MediaContentCollection,
    memory_chunk_collection: MemoryChunkCollection,
) -> list[BaseTool]:
    """create a recall_memory tool for full-content retrieval by ID.

    :param user_id: user whose memories to recall (row filter +
        evaluator ``caller_user_id``)
    :ptype user_id: UUID
    :param agent_id: owning agent UUID for rbac namespace lookup
    :ptype agent_id: UUID
    :param customer_id: owning customer UUID for rbac namespace lookup
    :ptype customer_id: UUID
    :param authorizer: rbac authorizer dependency bundle; required
    :ptype authorizer: MemoryAuthorizerDependencies
    :param memories_collection: three-tier memories collection
    :ptype memories_collection: MemoriesCollection
    :param media_content_collection: three-tier media_content collection
    :ptype media_content_collection: MediaContentCollection
    :param memory_chunk_collection: three-tier memory_chunks collection
    :ptype memory_chunk_collection: MemoryChunkCollection
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """

    @tool("recall_memory", args_schema=RecallMemoryInput)
    async def recall_memory(id: str, type: str) -> str:
        """retrieve full content of a memory, media description, or chunk."""
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
            content = await memories_collection.fetch_content_for_recall(
                memory_id=item_uuid,
                user_id=user_id,
                agent_id=agent_id,
            )
            if content is None:
                return "Memory not found or access denied."
            return content

        elif item_type == "media":
            content = await media_content_collection.fetch_content_for_recall(
                content_id=item_uuid,
                user_id=user_id,
                agent_id=agent_id,
            )
            if content is None:
                return "Media content not found or access denied."
            if len(content) > 12000:
                content = content[:12000] + "\n\n[Content truncated — full text is longer]"
            return content

        elif item_type == "chunk":
            content = await memory_chunk_collection.fetch_content_for_recall(
                chunk_id=item_uuid,
                user_id=user_id,
                agent_id=agent_id,
            )
            if content is None:
                return "Document chunk not found or access denied."
            return content

        else:
            return f"Unknown type '{type}'. Use 'memory', 'media', or 'chunk'."

    recall_memory.description = (
        "Retrieve the full content of a memory, media description, or document chunk. "
        "Use this when the system prompt shows a summary-only item (without '(detailed)' marker) "
        "and you need the complete content to answer the user's question accurately. "
        "Pass the ID from the [mem:ID], [media:ID], or [chunk:ID] tag and the type."
    )

    return [recall_memory]


_VALID_MEMORY_TYPES = frozenset(t.value for t in MemoryType)
_ = MediaCollection  # imported for public symbol availability; factory users pass it via __init__ wiring


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
    user_id: UUID,
    embedding_provider: EmbeddingProvider,
    agent_id: UUID,
    customer_id: UUID,
    authorizer: MemoryAuthorizerDependencies,
    memories_collection: MemoriesCollection,
    *,
    context_resolver: Callable[[], Any],
    similarity_dedup_threshold: float = 0.90,
) -> list[BaseTool]:
    """create an add_memory tool for explicit memory storage.

    data-layer-task-01 sub-task 4: ``conversation_id`` + ``message_id``
    are resolved per-call via ``context_resolver`` rather than bound at
    factory time. binding them at construction was the latent multi-
    plex violation surfaced by Group D D4-2: an agent pod multiplexes
    many conversations and a tool minted with one conversation's id
    would attribute every subsequent call's memory to that frozen
    conversation. the per-call resolver mirrors the
    :class:`NatsToolWrapper._arun` shape that already reads identity
    from :class:`CallContext` at invocation time.

    :param user_id: owning user's UUID (row filter + evaluator
        ``caller_user_id``)
    :ptype user_id: UUID
    :param embedding_provider: provider for embedding vectors
    :ptype embedding_provider: EmbeddingProvider
    :param agent_id: owning agent UUID (memory namespace owner)
    :ptype agent_id: UUID
    :param customer_id: owning customer UUID
    :ptype customer_id: UUID
    :param authorizer: rbac authorizer dependency bundle; required
    :ptype authorizer: MemoryAuthorizerDependencies
    :param memories_collection: three-tier memories collection; every
        dedup lookup + insert goes through this collection
    :ptype memories_collection: MemoriesCollection
    :param context_resolver: callable returning the live
        :class:`CallContext` (or any object exposing
        ``conversation_id`` and ``correlation_id`` attributes); read
        at every tool invocation so the tool attributes the memory
        to the actively-running conversation, not the conversation
        the factory was minted under
    :ptype context_resolver: Callable[[], Any]
    :param similarity_dedup_threshold: if an existing memory
        exceeds this similarity score, the existing memory is
        updated instead of creating a duplicate. default 0.90
    :ptype similarity_dedup_threshold: float
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """
    dedup_threshold = similarity_dedup_threshold

    @tool("add_memory", args_schema=AddMemoryInput)
    async def add_memory(content: str, memory_type: str = "preference") -> str:
        """store a memory about the user for future conversations."""
        # data-layer-task-01 sub-task 4: read live identity from
        # CallContext via the resolver. binding these at factory
        # construction would attribute every memory to whatever
        # conversation the pod was first minted under -- a multiplex
        # violation in any pod handling more than one conversation.
        try:
            live_ctx = context_resolver()
        except Exception as exc:
            return _tool_error(
                "add_memory",
                "context",
                f"context_resolver raised {type(exc).__name__}: {exc}",
            )
        live_conversation_id = getattr(live_ctx, "conversation_id", None)
        live_message_id = getattr(live_ctx, "correlation_id", None)
        if live_conversation_id is None:
            return _tool_error(
                "add_memory",
                "context",
                "context_resolver returned no conversation_id",
            )
        if live_message_id is None:
            # source attribution falls back to a fresh UUID when no
            # correlation is available (smoke fixtures, hub-direct
            # calls without an envelope)
            live_message_id = uuid4()
        try:
            ns_entity = await authorizer.namespace_collection.get_by_owner_and_customer(
                namespace_type=MEMORY_NAMESPACE_TYPE,
                owner_agent_id=agent_id,
                customer_id=customer_id,
            )
        except Exception as exc:
            return _tool_error(
                "add_memory",
                "authorize",
                f"memory namespace lookup for agent={agent_id} customer={customer_id} failed: {exc}",
            )
        if ns_entity is None:
            return _tool_error(
                "add_memory",
                "authorize",
                f"memory namespace for agent={agent_id} customer={customer_id} could not be resolved",
            )
        try:
            await _ensure_first_write_owner_assignment(
                memories_collection=memories_collection,
                authorizer=authorizer,
                ns_entity=ns_entity,
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

        mt = memory_type.lower().strip()
        if mt not in _VALID_MEMORY_TYPES:
            return f"Invalid memory_type '{memory_type}'. Valid types: {', '.join(sorted(_VALID_MEMORY_TYPES))}"

        embedding = await _safe_aembed_query(embedding_provider, content)
        if embedding is None:
            return _tool_error("add_memory", "embed", "embedding provider returned None")

        try:
            similar_rows = await memories_collection.find_similar_for_dedup(
                user_id=user_id,
                agent_id=agent_id,
                embedding=embedding,
                top_k=3,
                threshold=0.0,
            )
            for row in similar_rows:
                if float(row["similarity"]) >= dedup_threshold:
                    existing_id = row["memory_id"]
                    existing_entity = await memories_collection.get(
                        (agent_id, existing_id),
                    )
                    if existing_entity is None:
                        continue
                    existing_entity.content = content
                    existing_entity.type_memory = mt
                    existing_entity.embedding = embedding
                    await memories_collection.save_entity(existing_entity)
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
            log.warning(
                "add_memory: dedup check failed, inserting new",
                extra={"extra_data": {"error": str(exc)}},
            )

        try:
            now = datetime.now(UTC)
            memory_id = uuid4()
            new_data: dict[str, Any] = {
                "memory_id": memory_id,
                "agent_id": agent_id,
                "customer_id": customer_id,
                "user_id": user_id,
                "conversation_id": live_conversation_id,
                "message_id_source": live_message_id,
                "type_memory": mt,
                "content": content,
                "embedding": embedding,
                "is_deleted": False,
                "media_id": None,
                "date_created": now,
                "date_deleted": None,
                "date_updated": now,
            }
            entity: MemoryEntity = memories_collection.create(new_data)
            await memories_collection.save_entity(entity)

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
