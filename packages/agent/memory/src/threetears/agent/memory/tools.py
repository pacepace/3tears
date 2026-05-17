"""Memory tools -- LangChain tools for searching, recalling, and adding memories.

Provides ``load_memory_search_tool``, ``load_memory_recall_tool``, and
``load_memory_add_tool`` as reusable factories — all in canonical
``noun_verb`` form (the LangChain tool names are ``memory_search``,
``memory_recall``, ``memory_add``). Callers supply the four memory-
package Collections (:class:`MemoriesCollection`,
:class:`MediaCollection`, :class:`MediaContentCollection`,
:class:`MemoryChunkCollection`) plus user_id, agent_id, customer_id, and
an authorizer. Every SQL site that used to live raw on a pool is now a
method call on one of the Collections; tool factories hold no pool
reference.

Optional ``ledger_callback`` lets the host application track which items
the LLM has seen.

``memory_add`` requires a ``conversation_id`` for every persisted row;
it is read from a per-call ``context_resolver`` (the canonical
``CallContext`` shape) so the live conversation attribution is never
bound at factory construction (a frozen attribution is a multiplex
violation). The LLM never sees ``conversation_id`` in the Pydantic
tool schema — the runtime injects it, full stop.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from uuid_utils import uuid7

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
from langchain_core.embeddings import Embeddings

from threetears.agent.memory.embedding_utils import _safe_aembed_query
from threetears.agent.memory.entities import MemoryEntity
from threetears.agent.memory.types import MemoryType
from threetears.observe import get_logger

__all__ = [
    "ChunkRecallInput",
    "ChunkSearchInput",
    "LedgerCallback",
    "MemoryAddInput",
    "MemoryRecallInput",
    "MemorySearchInput",
    "load_chunk_recall_tool",
    "load_chunk_search_tool",
    "load_memory_add_tool",
    "load_memory_recall_tool",
    "load_memory_search_tool",
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


class MemoryRecallInput(BaseModel):
    """Input schema for ``memory_recall`` tool.

    v0.7.0 (transcript-chunks shard C) rewrote this from the old
    ``(id, type)`` discriminator into a memory-centric surface: the
    tool returns the parent memory + a window into its chunks, with
    four mutually-exclusive chunk-selection modes.

    Mode resolution:

    - none of ``chunk_query`` / ``chunk_indexes`` / ``chunk_id_after``
      / ``chunk_id_before`` set → memory + first ``limit`` chunks in
      ``chunk_id`` ASC order (creation order, since chunk_id is
      UUIDv7).
    - ``chunk_query`` set → memory + top ``limit`` chunks via
      hybrid (vector + FTS) search restricted to this memory.
    - ``chunk_indexes`` set → memory + exactly those chunks (by
      ``chunk_index`` column). Out-of-range indexes silently skipped.
    - ``chunk_id_after`` set → memory + ``limit`` chunks with
      ``chunk_id > cursor`` (forward paging).
    - ``chunk_id_before`` set → memory + ``limit`` chunks with
      ``chunk_id < cursor`` (backward paging; reversed to ASC for
      the caller).

    Passing more than one mode raises ValidationError before the
    tool body runs.

    Recalls of media or by-chunk-id moved to other tools:
    ``chunk_recall(chunk_id)`` is the single-chunk by-id read;
    media is reached via the parent memory's ``media.memory_id`` link.
    """

    memory_id: str = Field(
        description="UUID of the parent memory to recall, with its chunks.",
    )
    chunk_query: str | None = Field(
        default=None,
        description=(
            "Optional hybrid (vector + FTS) search restricted to "
            "this memory's chunks. Mutually exclusive with the "
            "other chunk_* modes."
        ),
    )
    chunk_indexes: list[int] | None = Field(
        default=None,
        description=(
            "Optional list of chunk_index ints to fetch exactly. Mutually exclusive with the other chunk_* modes."
        ),
    )
    chunk_id_after: str | None = Field(
        default=None,
        description=(
            "Optional forward-cursor chunk_id UUID. Returns up to "
            "``limit`` chunks with chunk_id strictly greater. "
            "Mutually exclusive with the other chunk_* modes."
        ),
    )
    chunk_id_before: str | None = Field(
        default=None,
        description=(
            "Optional backward-cursor chunk_id UUID. Returns up to "
            "``limit`` chunks with chunk_id strictly less, reversed "
            "to ASC order before returning. Mutually exclusive with "
            "the other chunk_* modes."
        ),
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum chunks to return (default 5, max 50).",
    )

    @model_validator(mode="after")
    def _modes_are_mutually_exclusive(self) -> "MemoryRecallInput":
        """Reject ambiguous calls that pass more than one chunk mode."""
        modes_set = sum(
            1
            for m in (
                self.chunk_query,
                self.chunk_indexes,
                self.chunk_id_after,
                self.chunk_id_before,
            )
            if m is not None and m != ""
        )
        if modes_set > 1:
            raise ValueError("Pass at most one of chunk_query / chunk_indexes / chunk_id_after / chunk_id_before.")
        return self


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
    embedding_provider: Embeddings,
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
    :ptype embedding_provider: Embeddings
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


async def load_memory_recall_tool(
    user_id: UUID,
    agent_id: UUID,
    customer_id: UUID,
    authorizer: MemoryAuthorizerDependencies,
    memories_collection: MemoriesCollection,
    media_content_collection: MediaContentCollection,
    memory_chunk_collection: MemoryChunkCollection,
    embedding_provider: Embeddings | None = None,
    similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
) -> list[BaseTool]:
    """create a memory_recall tool that returns a memory + its chunks.

    v0.7.0 (transcript-chunks shard C) replaced the old ``(id, type)``
    discriminator surface with a memory-centric one: the tool takes
    one ``memory_id`` plus an optional chunk-selection mode and
    returns the parent memory's content + a window into its chunks.

    Mode resolution (mutually exclusive; enforced by the Pydantic
    ``model_validator`` on :class:`MemoryRecallInput`):

    - none → memory + first ``limit`` chunks in chunk_id ASC order.
    - ``chunk_query`` → memory + top ``limit`` chunks via hybrid
      (vector + FTS) search restricted to this memory. Requires
      ``embedding_provider`` at factory construction; the tool
      gracefully degrades to the default mode (warns once) when no
      embedding is wired.
    - ``chunk_indexes`` → memory + exactly those chunks (by
      ``chunk_index`` column). Out-of-range indexes silently skipped.
    - ``chunk_id_after`` → memory + ``limit`` chunks paged forward.
    - ``chunk_id_before`` → memory + ``limit`` chunks paged
      backward, reversed to ASC for the caller.

    Recalls of media moved to no separate tool: media is reached
    via the parent memory's ``media.memory_id`` link, surfaced by
    the system prompt's media-context block. Single-chunk by-id
    lookup moved to :func:`load_chunk_recall_tool`.

    :param user_id: user whose memories to recall
    :ptype user_id: UUID
    :param agent_id: owning agent UUID for rbac namespace lookup
    :ptype agent_id: UUID
    :param customer_id: owning customer UUID for rbac namespace lookup
    :ptype customer_id: UUID
    :param authorizer: rbac authorizer dependency bundle; required
    :ptype authorizer: MemoryAuthorizerDependencies
    :param memories_collection: three-tier memories collection
    :ptype memories_collection: MemoriesCollection
    :param media_content_collection: kept for signature parity with
        v0.6.x callers; no longer consumed by this tool. media now
        recalls via the parent memory's ``media.memory_id`` link
    :ptype media_content_collection: MediaContentCollection
    :param memory_chunk_collection: three-tier memory_chunks collection
    :ptype memory_chunk_collection: MemoryChunkCollection
    :param embedding_provider: optional embeddings provider for the
        ``chunk_query`` mode. When omitted, ``chunk_query`` requests
        gracefully degrade to default-mode output with a warning
    :ptype embedding_provider: Embeddings | None
    :param similarity_threshold: floor on hybrid score for
        ``chunk_query`` mode
    :ptype similarity_threshold: float
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """
    # kept for signature parity; no consumer in the v0.7.0 body.
    _ = media_content_collection

    @tool("memory_recall", args_schema=MemoryRecallInput)
    async def memory_recall(
        memory_id: str,
        chunk_query: str | None = None,
        chunk_indexes: list[int] | None = None,
        chunk_id_after: str | None = None,
        chunk_id_before: str | None = None,
        limit: int = 5,
    ) -> str:
        """recall a memory + a window into its chunks."""
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
            return _tool_error("memory_recall", "authorize", str(exc))

        try:
            mem_uuid = UUID(memory_id)
        except ValueError:
            return f"Invalid memory_id format: {memory_id}"

        # ── memory content ──────────────────────────────────────
        memory_content = await memories_collection.fetch_content_for_recall(
            memory_id=mem_uuid,
            user_id=user_id,
            agent_id=agent_id,
        )
        if memory_content is None:
            return "Memory not found or access denied."

        # ── chunks (mode dispatch) ──────────────────────────────
        chunks: list[dict[str, Any]] = []
        mode_name = "default"
        try:
            if chunk_query is not None and chunk_query.strip():
                mode_name = "chunk_query"
                if embedding_provider is None:
                    log.warning(
                        "memory_recall chunk_query mode requested without "
                        "embedding_provider; falling back to default mode",
                    )
                    chunks = await memory_chunk_collection.find_by_memory_id(
                        mem_uuid,
                        user_id=user_id,
                        agent_id=agent_id,
                        customer_id=customer_id,
                        limit=limit,
                    )
                else:
                    embedding = await _safe_aembed_query(
                        embedding_provider,
                        chunk_query,
                    )
                    if embedding is None:
                        return _tool_error(
                            "memory_recall",
                            "embed",
                            "embedding unavailable for chunk_query mode",
                        )
                    chunks = await memory_chunk_collection.hybrid_search_within_memory(
                        memory_id=mem_uuid,
                        user_id=user_id,
                        agent_id=agent_id,
                        customer_id=customer_id,
                        embedding=embedding,
                        user_text=chunk_query,
                        candidate_k=max(limit * 4, 20),
                        similarity_threshold=similarity_threshold,
                        chunk_signal_weights={"semantic": 0.6, "keyword": 0.4},
                    )
                    chunks = chunks[:limit]

            elif chunk_indexes is not None and chunk_indexes:
                mode_name = "chunk_indexes"
                # Inline cache-bypass fetch: the by-index access
                # pattern isn't exposed on the Collection today, so
                # the SELECT lives here. Auth triple (agent_id,
                # user_id) pinned at the SQL boundary; chunk_indexes
                # is the only caller-supplied filter.
                l3_pool = getattr(memory_chunk_collection, "l3_pool", None)
                if l3_pool is None:
                    chunks = []
                else:
                    rows = await l3_pool.fetch(
                        "SELECT chunk_id, chunk_index, content, summary, "
                        "       message_id_start, message_id_end, "
                        "       token_count, date_created "
                        "FROM memory_chunks "
                        "WHERE agent_id = $1 AND memory_id = $2 "
                        "  AND user_id = $3 "
                        "  AND chunk_index = ANY($4) "
                        "ORDER BY chunk_index ASC",
                        agent_id,
                        mem_uuid,
                        user_id,
                        list(chunk_indexes),
                    )
                    chunks = [dict(r) for r in rows]

            elif chunk_id_after is not None and chunk_id_after.strip():
                mode_name = "chunk_id_after"
                try:
                    cur = UUID(chunk_id_after)
                except ValueError:
                    return f"Invalid chunk_id_after format: {chunk_id_after}"
                chunks = await memory_chunk_collection.find_by_memory_id(
                    mem_uuid,
                    user_id=user_id,
                    agent_id=agent_id,
                    customer_id=customer_id,
                    limit=limit,
                    chunk_id_after=cur,
                )

            elif chunk_id_before is not None and chunk_id_before.strip():
                mode_name = "chunk_id_before"
                try:
                    cur = UUID(chunk_id_before)
                except ValueError:
                    return f"Invalid chunk_id_before format: {chunk_id_before}"
                chunks = await memory_chunk_collection.find_by_memory_id(
                    mem_uuid,
                    user_id=user_id,
                    agent_id=agent_id,
                    customer_id=customer_id,
                    limit=limit,
                    chunk_id_before=cur,
                )

            else:
                # Default mode: first ``limit`` chunks chronologically.
                chunks = await memory_chunk_collection.find_by_memory_id(
                    mem_uuid,
                    user_id=user_id,
                    agent_id=agent_id,
                    customer_id=customer_id,
                    limit=limit,
                )
        except Exception as exc:
            return _tool_error("memory_recall", f"fetch chunks ({mode_name})", str(exc))

        # ── render ──────────────────────────────────────────────
        lines: list[str] = [
            f"[memory:{mem_uuid}]",
            memory_content,
        ]
        if chunks:
            lines.append("")
            lines.append(f"Chunks ({len(chunks)} returned, mode={mode_name}):")
            for ch in chunks:
                cid = ch.get("chunk_id", "?")
                idx = ch.get("chunk_index", "?")
                preview = (ch.get("content") or "")[:200]
                lines.append(f"  [chunk:{cid}] (index={idx})\n  {preview}")
        else:
            lines.append("")
            lines.append("(no chunks)")
        return "\n".join(lines)

    memory_recall.description = (
        "Recall a memory's full content plus a window into its chunks. "
        "Pass ``memory_id`` (from [memory:ID] tags surfaced by "
        "memory_search / conversation_summarize). Optional chunk-mode "
        "args are mutually exclusive: ``chunk_query`` for hybrid "
        "search within the memory, ``chunk_indexes`` for exact-index "
        "lookup, ``chunk_id_after`` / ``chunk_id_before`` for cursor "
        "paging. Default returns the first ``limit`` chunks (default 5) "
        "in creation order. Use chunk_recall(chunk_id) for a single "
        "chunk's full text."
    )

    return [memory_recall]


_VALID_MEMORY_TYPES = frozenset(t.value for t in MemoryType)
_ = MediaCollection  # imported for public symbol availability; factory users pass it via __init__ wiring


class MemoryAddInput(BaseModel):
    """Input schema for memory_add tool.

    Note: ``conversation_id`` is intentionally NOT in this schema —
    the LLM never sees or sets it. The tool runner reads it per-call
    from the live :class:`CallContext` via the factory's
    ``context_resolver`` so the memory is always attributed to the
    actively-running conversation.
    """

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


async def load_memory_add_tool(
    user_id: UUID,
    embedding_provider: Embeddings,
    agent_id: UUID,
    customer_id: UUID,
    authorizer: MemoryAuthorizerDependencies,
    memories_collection: MemoriesCollection,
    *,
    context_resolver: Callable[[], Any],
    similarity_dedup_threshold: float = 0.90,
) -> list[BaseTool]:
    """create a memory_add tool for explicit memory storage.

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
    :ptype embedding_provider: Embeddings
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

    @tool("memory_add", args_schema=MemoryAddInput)
    async def memory_add(content: str, memory_type: str = "preference") -> str:
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
                "memory_add",
                "context",
                f"context_resolver raised {type(exc).__name__}: {exc}",
            )
        live_conversation_id = getattr(live_ctx, "conversation_id", None)
        live_message_id = getattr(live_ctx, "correlation_id", None)
        if live_conversation_id is None:
            return _tool_error(
                "memory_add",
                "context",
                "context_resolver returned no conversation_id",
            )
        if live_message_id is None:
            # source attribution falls back to a fresh UUID when no
            # correlation is available (smoke fixtures, hub-direct
            # calls without an envelope)
            # uuid_utils.uuid7() returns a uuid_utils.UUID; cast so
            # downstream stores see stdlib UUID (the canonical type
            # across 3tears).
            live_message_id = UUID(str(uuid7()))
        try:
            ns_entity = await authorizer.namespace_collection.get_by_owner_and_customer(
                namespace_type=MEMORY_NAMESPACE_TYPE,
                owner_agent_id=agent_id,
                customer_id=customer_id,
            )
        except Exception as exc:
            return _tool_error(
                "memory_add",
                "authorize",
                f"memory namespace lookup for agent={agent_id} customer={customer_id} failed: {exc}",
            )
        if ns_entity is None:
            return _tool_error(
                "memory_add",
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
                "memory_add: first-write assignment ensure failed",
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
            return _tool_error("memory_add", "authorize", str(exc))

        mt = memory_type.lower().strip()
        if mt not in _VALID_MEMORY_TYPES:
            return f"Invalid memory_type '{memory_type}'. Valid types: {', '.join(sorted(_VALID_MEMORY_TYPES))}"

        embedding = await _safe_aembed_query(embedding_provider, content)
        if embedding is None:
            return _tool_error("memory_add", "embed", "embedding provider returned None")

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
                        "memory_add: updated existing similar memory",
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
                "memory_add: dedup check failed, inserting new",
                extra={"extra_data": {"error": str(exc)}},
            )

        try:
            now = datetime.now(UTC)
            memory_id = UUID(str(uuid7()))
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
                "date_created": now,
                "date_updated": now,
            }
            entity: MemoryEntity = memories_collection.create(new_data)
            await memories_collection.save_entity(entity)

            log.info(
                "memory_add: stored new memory",
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
                "memory_add: insert failed",
                extra={"extra_data": {"error": str(exc)}},
            )
            return _tool_error("memory_add", "store", str(exc))

    memory_add.description = (
        "Store a memory about the user for future conversations. "
        "Use this when the user explicitly asks you to remember something "
        "(e.g., 'remember that I prefer...', 'my X is Y', 'don't forget...'). "
        "Write the memory as a concise, standalone fact. "
        "Duplicate detection is automatic — if a very similar memory already "
        "exists, it will be updated instead of creating a duplicate."
    )

    return [memory_add]


# ---------------------------------------------------------------------------
# chunk_recall + chunk_search (v0.7.0 / transcript-chunks shard C)
# ---------------------------------------------------------------------------


class ChunkRecallInput(BaseModel):
    """Input schema for the ``chunk_recall`` tool.

    Single-chunk lookup by ID. Distinct from ``memory_recall`` which
    returns a memory plus its constituent chunks; this tool is for
    when the agent has already identified one specific chunk (e.g.
    from a prior ``chunk_search`` result) and wants its full content
    plus parent-memory context.
    """

    chunk_id: str = Field(
        description="UUID of the chunk to recall.",
    )


class ChunkSearchInput(BaseModel):
    """Input schema for the ``chunk_search`` tool.

    Cross-memory chunk hybrid search. Distinct from ``memory_search``
    (which searches at the memory level); this tool reaches into the
    chunks belonging to every memory the user owns and returns the
    most relevant chunks regardless of which memory they belong to.
    """

    query: str = Field(
        description=("Natural language search query for finding relevant chunks across every memory the user owns."),
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of chunks to return (default 5, max 20).",
    )


async def load_chunk_recall_tool(
    user_id: UUID,
    agent_id: UUID,
    customer_id: UUID,
    authorizer: MemoryAuthorizerDependencies,
    memories_collection: MemoriesCollection,
    memory_chunk_collection: MemoryChunkCollection,
) -> list[BaseTool]:
    """create a ``chunk_recall`` tool that returns a chunk + parent memory.

    auth: the chunk's parent memory's user_id must match ``user_id``.
    enforced at the SQL level by ``MemoryChunkCollection.fetch_content_for_recall``
    which scopes by ``(agent_id, user_id)``. an unauthorized chunk
    surfaces as ``None`` → tool returns a graceful "not found" message
    rather than leaking existence.

    the return shape is a formatted text block: chunk content +
    ``[parent memory: <memory_id>]`` footer so the agent can chain to
    ``memory_recall`` if it needs the surrounding memory.

    :param user_id: caller's user UUID
    :ptype user_id: UUID
    :param agent_id: agent UUID for rbac namespace lookup
    :ptype agent_id: UUID
    :param customer_id: customer UUID for rbac namespace lookup
    :ptype customer_id: UUID
    :param authorizer: rbac authorizer dependency bundle
    :ptype authorizer: MemoryAuthorizerDependencies
    :param memories_collection: three-tier memories collection (for
        the parent-memory metadata join)
    :ptype memories_collection: MemoriesCollection
    :param memory_chunk_collection: three-tier memory_chunks collection
    :ptype memory_chunk_collection: MemoryChunkCollection
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """

    @tool("chunk_recall", args_schema=ChunkRecallInput)
    async def chunk_recall(chunk_id: str) -> str:
        """retrieve the full content of one chunk plus its parent memory id."""
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
            return _tool_error("chunk_recall", "authorize", str(exc))

        try:
            cid = UUID(chunk_id)
        except ValueError:
            return f"Invalid chunk_id format: {chunk_id}"

        # ``fetch_content_for_recall`` returns both content + memory_id
        # in one SELECT (the parent_memory_id is what makes chunk_recall
        # chainable into memory_recall — the footer below names it so
        # the agent can pull the surrounding memory in a follow-up turn).
        chunk_row = await memory_chunk_collection.fetch_content_for_recall(
            chunk_id=cid,
            user_id=user_id,
            agent_id=agent_id,
        )
        if chunk_row is None:
            return "Chunk not found or access denied."
        content, parent_memory_id = chunk_row
        return f"{content}\n\n[parent memory: {parent_memory_id!s}]"

    chunk_recall.description = (
        "Retrieve the full content of a single document/transcript chunk by "
        "ID, plus the ID of the memory it belongs to. Use this when you "
        "have a specific chunk_id (e.g. from a previous chunk_search result) "
        "and need the full text. The footer ``[parent memory: <id>]`` lets "
        "you chain to memory_recall for the surrounding context."
    )

    return [chunk_recall]


async def load_chunk_search_tool(
    user_id: UUID,
    agent_id: UUID,
    customer_id: UUID,
    embedding_provider: Embeddings,
    authorizer: MemoryAuthorizerDependencies,
    memory_chunk_collection: MemoryChunkCollection,
    ledger_callback: LedgerCallback | None = None,
    similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
) -> list[BaseTool]:
    """create a ``chunk_search`` tool: cross-memory chunk hybrid search.

    delegates to :meth:`MemoryChunkCollection.hybrid_search` which is
    the same path the memory-retrieval node uses for grounding.
    optionally records each surfaced chunk via ``ledger_callback`` so
    the host application can dedup chunks already shown in the
    current conversation (the metallm-ledger pattern).

    :param user_id: caller's user UUID
    :ptype user_id: UUID
    :param agent_id: agent UUID for rbac namespace lookup
    :ptype agent_id: UUID
    :param customer_id: customer UUID for rbac namespace lookup
    :ptype customer_id: UUID
    :param embedding_provider: embedding provider for the query vector
    :ptype embedding_provider: Embeddings
    :param authorizer: rbac authorizer dependency bundle
    :ptype authorizer: MemoryAuthorizerDependencies
    :param memory_chunk_collection: three-tier memory_chunks collection
    :ptype memory_chunk_collection: MemoryChunkCollection
    :param ledger_callback: optional async callback invoked with
        ``(chunk_id, "chunk", content_preview)`` for every surfaced
        chunk; the host application uses this to mark chunks as seen
        within the current conversation so subsequent invocations can
        exclude them
    :ptype ledger_callback: LedgerCallback | None
    :param similarity_threshold: floor on hybrid score
    :ptype similarity_threshold: float
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """

    @tool("chunk_search", args_schema=ChunkSearchInput)
    async def chunk_search(query: str, limit: int = 5) -> str:
        """search across every chunk the user owns by relevance to query."""
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
            return _tool_error("chunk_search", "authorize", str(exc))

        if not query or not query.strip():
            return "Empty query. Provide a search term."

        embedding = await _safe_aembed_query(embedding_provider, query)
        if embedding is None:
            return "Embedding unavailable; cannot search chunks."

        try:
            results = await memory_chunk_collection.hybrid_search(
                user_id=user_id,
                agent_id=agent_id,
                customer_id=customer_id,
                embedding=embedding,
                user_text=query,
                candidate_k=max(limit * 4, 20),
                similarity_threshold=similarity_threshold,
                chunk_signal_weights={"semantic": 0.6, "keyword": 0.4},
            )
        except Exception as exc:
            return _tool_error("chunk_search", "search", str(exc))

        if not results:
            return "No chunks matched."

        results = results[:limit]
        lines: list[str] = []
        for r in results:
            chunk_id = str(r.get("chunk_id", ""))
            memory_id = r.get("memory_id")
            memory_id_str = str(memory_id) if memory_id is not None else "?"
            content = str(r.get("content", "") or "")
            preview = content[:200]
            score = r.get("hybrid_score") or r.get("similarity") or 0.0
            lines.append(f"[chunk:{chunk_id}] (memory:{memory_id_str}, score={score:.3f})\n{preview}")
            if ledger_callback is not None:
                try:
                    await ledger_callback(chunk_id, "chunk", preview[:80])
                except Exception as cb_exc:
                    log.debug(
                        "chunk_search ledger_callback failed for chunk %s: %s",
                        chunk_id,
                        cb_exc,
                    )

        return "\n\n".join(lines)

    chunk_search.description = (
        "Search every document / transcript chunk the user owns for content "
        "relevant to a natural-language query. Returns up to ``limit`` "
        "chunks (default 5) ordered by hybrid score (semantic + keyword). "
        "Each result carries the chunk_id, parent memory_id, score, and a "
        "preview. Use chunk_recall to fetch a chunk's full text."
    )

    return [chunk_search]
