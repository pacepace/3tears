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
import re
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

# v0.7.5: alias validation. Letters, digits, hyphen, underscore. 1-64.
_ALIAS_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

# Type alias for the optional ledger callback
LedgerCallback = Callable[[str, str, str], Awaitable[None]]


class MemorySearchInput(BaseModel):
    """Input schema for memory_search tool."""

    query: str = Field(default="", description="Search text.")
    type_filter: str | None = Field(
        default=None,
        description="Filter by type: preference, fact, decision, topical_context.",
    )
    ids: list[str] | None = Field(
        default=None,
        description="Direct lookup by [memory:<id>]. Bypasses search.",
    )
    alias: str | None = Field(
        default=None,
        description="Direct lookup by named alias set via memory_add.",
    )
    mode: str = Field(
        default="balanced",
        description="precise (exact wording) / balanced (default) / fuzzy (concepts).",
    )
    date_after: str | None = Field(
        default=None,
        description="ISO date or datetime. Only memories on/after.",
    )
    date_before: str | None = Field(
        default=None,
        description="ISO date or datetime. Only memories on/before.",
    )

    @model_validator(mode="after")
    def _validate(self) -> "MemorySearchInput":
        if not self.query and not self.ids and not self.alias:
            raise ValueError("Provide at least one of 'query', 'ids', or 'alias'.")
        if self.mode not in {"precise", "balanced", "fuzzy"}:
            raise ValueError(f"mode must be 'precise', 'balanced', or 'fuzzy'; got {self.mode!r}.")
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
        default="",
        description="[memory:<id>] UUID. Required unless alias is set.",
    )
    alias: str | None = Field(
        default=None,
        description="Named alias set via memory_add. Alternative to memory_id.",
    )
    chunk_query: str | None = Field(
        default=None,
        description="Search inside chunks. Mutually exclusive with chunk_indexes / chunk_id_after / chunk_id_before.",
    )
    chunk_indexes: list[int] | None = Field(
        default=None,
        description="Specific chunk positions (zero-based).",
    )
    chunk_id_after: str | None = Field(default=None, description="Forward cursor. [chunk:<id>] UUID.")
    chunk_id_before: str | None = Field(default=None, description="Backward cursor. Same format.")
    limit: int = Field(default=5, ge=1, le=50, description="Max chunks. 1-50.")

    @model_validator(mode="after")
    def _validate(self) -> "MemoryRecallInput":
        """Require memory_id or alias + reject ambiguous chunk modes."""
        if not self.memory_id and not self.alias:
            raise ValueError("Provide memory_id or alias.")
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
        alias: str | None = None,
        mode: str = "balanced",
        date_after: str | None = None,
        date_before: str | None = None,
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

        # v0.7.5: alias is a direct-lookup short-circuit. Returns at
        # most one memory by its per-user-unique named anchor.
        if alias:
            try:
                row = await memories_collection.find_by_alias(
                    user_id=user_id,
                    agent_id=agent_id,
                    alias=alias,
                )
            except Exception as exc:
                return _tool_error("memory_search", "alias", str(exc))
            if row is None:
                return f"No memory found with alias '{alias}'."
            mid = str(row["memory_id"])
            ts = _fmt_dt(row.get("date_created"))
            preview = f"[mem:{mid}] [{row['type_memory']}]"
            if ts:
                preview += f" [{ts}]"
            preview += f" {row['content']}"
            if ledger_callback:
                await ledger_callback(mid, "memory", row["content"])
            return f"Found memory by alias '{alias}':\n- {preview}"

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
            return "Provide one of 'query', 'ids', or 'alias'."

        # v0.7.5: parse ISO date filters. Either bound may stand alone.
        def _parse_iso(label: str, raw: str | None) -> tuple[datetime | None, str | None]:
            if not raw:
                return None, None
            try:
                value = datetime.fromisoformat(raw)
            except ValueError:
                return None, f"Invalid {label} '{raw}'. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)."
            if value.tzinfo is None:
                value = value.replace(tzinfo=UTC)
            return value, None

        date_after_dt, err_a = _parse_iso("date_after", date_after)
        if err_a:
            return _tool_error("memory_search", "filter", err_a)
        date_before_dt, err_b = _parse_iso("date_before", date_before)
        if err_b:
            return _tool_error("memory_search", "filter", err_b)

        # v0.7.5: ``mode`` selects which legs to run.
        #   precise → FTS only (keyword wins)
        #   fuzzy   → semantic only (concepts win)
        #   balanced (default) → both legs, merged
        if mode not in {"precise", "balanced", "fuzzy"}:
            return _tool_error(
                "memory_search",
                "filter",
                f"Invalid mode '{mode}'. Must be one of: precise, balanced, fuzzy.",
            )
        run_semantic = mode in {"balanced", "fuzzy"}
        run_fts = mode in {"balanced", "precise"}

        embedding = None
        if run_semantic:
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

        memories: list[dict[str, Any]] = []
        if run_semantic and embedding is not None:
            try:
                memories = await memories_collection.search_by_semantic(
                    user_id=user_id,
                    agent_id=agent_id,
                    embedding=embedding,
                    max_results=max_results,
                    similarity_threshold=sim_threshold,
                    type_filter=type_filter,
                    date_after=date_after_dt,
                    date_before=date_before_dt,
                )
            except Exception as exc:
                return _tool_error("memory_search", "query", str(exc))

        seen_ids = {m["memory_id"] for m in memories}

        if run_fts and fts_text:
            try:
                fts_rows = await memories_collection.search_by_fts(
                    user_id=user_id,
                    agent_id=agent_id,
                    fts_text=fts_text,
                    max_results=max_results,
                    type_filter=type_filter,
                    date_after=date_after_dt,
                    date_before=date_before_dt,
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
        mc_seen: set[str] = set()
        try:
            if embedding is not None:
                mc_rows = await media_content_collection.search_by_semantic(
                    user_id=user_id,
                    agent_id=agent_id,
                    embedding=embedding,
                    max_results=5,
                    similarity_threshold=sim_threshold,
                )
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
            if run_fts and fts_text:
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
            if embedding is not None:
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
        "Search stored memories by meaning or keyword. Use first "
        "when resuming a topic. Returns [memory:<id>] matches with "
        "relevance scores.\n\n"
        "- `ids` — direct fetch by [memory:<id>], bypasses search\n"
        "- `alias` — direct fetch by named anchor (set via memory_add)\n"
        "- `mode` — precise (exact wording) / balanced (default) / fuzzy (concepts)\n"
        "- `date_after`, `date_before` — narrow by ISO date range"
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
        memory_id: str = "",
        alias: str | None = None,
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

        # v0.7.5: alias resolves to a memory_id via the per-user-unique
        # ``alias`` column. The validator on MemoryRecallInput already
        # ensures at least one of memory_id / alias is present.
        if alias and not memory_id:
            try:
                row = await memories_collection.find_by_alias(
                    user_id=user_id,
                    agent_id=agent_id,
                    alias=alias,
                )
            except Exception as exc:
                return _tool_error("memory_recall", "alias", str(exc))
            if row is None:
                return f"No memory found with alias '{alias}'."
            memory_id = str(row["memory_id"])

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
            # v0.7.2 visibility hint: tell the agent how to drill INTO
            # the chunks via search rather than re-paging the same
            # window. Only shows when chunks were actually returned --
            # the hint is noise on memories with no chunks.
            lines.append("")
            lines.append(
                "To search inside this memory's chunks by topic, call "
                f"``memory_recall('{mem_uuid}', chunk_query='<your text>')``. "
                "For semantic search across ALL chunks the user owns, "
                "use ``chunk_search``."
            )
        else:
            lines.append("")
            lines.append("(no chunks)")
        return "\n".join(lines)

    memory_recall.description = (
        "Pull full content + chunk previews for a known [memory:<id>] "
        "(or a named alias). Pass chunk_query to narrow inside it, "
        "or chunk_indexes / chunk_id_after / chunk_id_before to page."
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
        description="What's worth preserving. About the user, yourself, or a moment.",
    )
    memory_type: str = Field(
        default="preference",
        description="One of: preference, fact, decision, topical_context, relational_context.",
    )
    alias: str | None = Field(
        default=None,
        description=(
            "Optional named anchor (e.g. 'cave-altar'). 1-64 chars, "
            "letters/digits/hyphens/underscores. Unique per user."
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
    salience_seed: float = 0.5,
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
    :param salience_seed: starting salience for a newly-added memory
        (design §3 per-user knob). default 0.5 matches the DB server
        default; a tuned value is honored on insert
    :ptype salience_seed: float
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """
    dedup_threshold = similarity_dedup_threshold

    @tool("memory_add", args_schema=MemoryAddInput)
    async def memory_add(
        content: str,
        memory_type: str = "preference",
        alias: str | None = None,
    ) -> str:
        """store a memory about the user for future conversations."""
        # v0.7.5: validate alias format early. Per-user uniqueness is
        # enforced at the DB level (alembic 088 partial unique index);
        # we surface a graceful error on UniqueViolation below.
        normalised_alias: str | None = None
        if alias is not None:
            stripped = alias.strip()
            if stripped:
                if not _ALIAS_RE.match(stripped):
                    return _tool_error(
                        "memory_add",
                        "alias",
                        f"Invalid alias '{alias}'. Use 1-64 letters, digits, hyphens, or underscores.",
                    )
                normalised_alias = stripped
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
                    # v0.7.2: surface ``[memory:<id>]`` on the dedup
                    # path so the agent can chain to memory_recall
                    # without a separate memory_search round-trip.
                    return (
                        f"Updated existing memory "
                        f"[memory:{existing_id}] "
                        f"(was similar at {float(row['similarity']):.0%}): "
                        f"{content}. "
                        f"Use ``memory_recall('{existing_id}')`` to "
                        f"read the full updated record."
                    )
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
                "alias": normalised_alias,
                # honor the per-user salience_seed knob (design §3); without
                # it the DB server default (0.5) applies and a tuned seed is
                # a silent no-op.
                "salience": salience_seed,
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
                        "alias": normalised_alias,
                    }
                },
            )
            # v0.7.2: surface ``[memory:<id>]`` so the agent can chain
            # to memory_recall without a follow-up memory_search.
            alias_clause = f" with alias '{normalised_alias}'" if normalised_alias is not None else ""
            return (
                f"Stored as [memory:{memory_id}]{alias_clause}: {content}. "
                f"Use ``memory_recall('{memory_id}')`` to read it back "
                f"in any future conversation."
            )

        except Exception as exc:
            # v0.7.5: surface alias-uniqueness collisions cleanly. The
            # partial unique index ``ix_memories_user_alias`` raises a
            # UniqueViolation from asyncpg; we don't import asyncpg
            # here (the memory pkg stays driver-agnostic), so match by
            # name + by the SQLSTATE substring in the message body.
            exc_name = type(exc).__name__
            exc_msg = str(exc)
            if normalised_alias and (
                exc_name == "UniqueViolationError" or "ix_memories_user_alias" in exc_msg or "23505" in exc_msg
            ):
                return _tool_error(
                    "memory_add",
                    "alias",
                    f"Alias '{normalised_alias}' already used. Pick a different name or "
                    f"call memory_search(alias='{normalised_alias}') to find the existing one.",
                )
            log.warning(
                "memory_add: insert failed",
                extra={"extra_data": {"error": exc_msg}},
            )
            return _tool_error("memory_add", "store", exc_msg)

    memory_add.description = (
        "Store something worth preserving across conversations — "
        "about the user, yourself, or anything you want to remember. "
        "Preferences, facts, decisions, observations, moments — "
        "use liberally. Deduplicates automatically at 90% similarity.\n\n"
        "- `memory_type` — preference / fact / decision / topical_context / relational_context\n"
        "- `alias` — optional named anchor for direct lookup later (e.g. 'cave-altar')\n\n"
        "Returns [memory:<id>]."
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

    chunk_id: str = Field(description="[chunk:<id>] UUID.")


class ChunkSearchInput(BaseModel):
    """Input schema for the ``chunk_search`` tool.

    Cross-memory chunk hybrid search. Distinct from ``memory_search``
    (which searches at the memory level); this tool reaches into the
    chunks belonging to every memory the user owns and returns the
    most relevant chunks regardless of which memory they belong to.
    """

    query: str = Field(description="Search text.")
    limit: int = Field(default=5, ge=1, le=20, description="Max chunks. 1-20.")
    mode: str = Field(
        default="balanced",
        description=("Search blend: precise (favor exact wording), balanced (default), fuzzy (favor concepts)."),
    )

    @model_validator(mode="after")
    def _validate_mode(self) -> "ChunkSearchInput":
        if self.mode not in {"precise", "balanced", "fuzzy"}:
            raise ValueError(f"mode must be 'precise', 'balanced', or 'fuzzy'; got {self.mode!r}.")
        return self


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
            return (
                "Chunk not found or access denied. Verify the "
                "``chunk_id`` came from a recent ``chunk_search`` "
                "result in this user's namespace."
            )
        content, parent_memory_id = chunk_row
        return (
            f"{content}\n\n"
            f"[parent memory: {parent_memory_id!s}]\n"
            f"Next step: call ``memory_recall('{parent_memory_id!s}')`` "
            f"to read the parent memory and its other chunks for "
            f"surrounding context."
        )

    chunk_recall.description = (
        "Read exact text of one chunk by [chunk:<id>]. Returns the slice verbatim plus its parent [memory:<id>]."
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
    current conversation (the ledger pattern).

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
    async def chunk_search(query: str, limit: int = 5, mode: str = "balanced") -> str:
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

        # v0.7.5: ``mode`` remaps the semantic/keyword blend so the
        # agent can self-correct between "I want similar concepts"
        # (fuzzy) and "I want exact wording" (precise). Default
        # ``balanced`` preserves prior 60/40 semantic-leaning behavior.
        weights = {
            "precise": {"semantic": 0.2, "keyword": 0.8},
            "balanced": {"semantic": 0.6, "keyword": 0.4},
            "fuzzy": {"semantic": 0.85, "keyword": 0.15},
        }.get(mode, {"semantic": 0.6, "keyword": 0.4})

        try:
            results = await memory_chunk_collection.hybrid_search(
                user_id=user_id,
                agent_id=agent_id,
                customer_id=customer_id,
                embedding=embedding,
                user_text=query,
                candidate_k=max(limit * 4, 20),
                similarity_threshold=similarity_threshold,
                chunk_signal_weights=weights,
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
            # convert at border: agent-facing tool-result display text
            memory_id_str = str(memory_id) if memory_id is not None else "?"
            content = str(r.get("content", "") or "")
            preview = content[:200]
            score = r.get("hybrid_score") or r.get("similarity") or 0.0
            lines.append(
                f"[chunk:{chunk_id}] (memory:{memory_id_str}, "
                f"score={score:.3f})\n{preview}\n"
                f"Next step: ``chunk_recall('{chunk_id}')`` for full "
                f"text, or ``memory_recall('{memory_id_str}')`` for "
                f"the parent memory."
            )
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
        "Hybrid vector + keyword search across every stored memory's "
        "text. Use when you need a specific line or detail, not just "
        "a topic. Pass mode='precise' to favor exact wording, "
        "'fuzzy' for concepts (default 'balanced'). "
        "Returns [chunk:<id>] (memory:<id>, score=X) previews."
    )

    return [chunk_search]
