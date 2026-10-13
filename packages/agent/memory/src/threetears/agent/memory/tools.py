"""Memory tools -- LangChain tools an agent searches, reads and changes its memories with.

Factories, all in canonical ``noun_verb`` form: ``memory_search``,
``memory_recall``, ``memory_list``, ``memory_add``, ``memory_update``,
``memory_forget``, ``memory_restore``, ``memory_keep``, ``chunk_search`` and
``chunk_recall``. Callers supply the memory-package Collections plus user_id,
agent_id, customer_id, and an authorizer. Every SQL site is a method call on
one of the Collections; tool factories hold no pool reference.

**What every result says.** A memory line carries the memory's state before
its text: permanent, retired and why, replaced and by what, corrected by what.
A date is shown in the zone the caller names (``tz``) and names it. Text that
is cut says so, how long the whole is, and which call reads the rest. A note
about the result as a whole (partial, more may match, ids not read) is its
first line, so a reader that keeps only the head still reads it.

**Nothing here deletes a row.** ``memory_update`` writes a replacement and
keeps the old memory readable; ``memory_forget`` retracts, and
``memory_restore`` undoes it. A permanent memory is changed or retired only
when the host says the person asked (``person_asked``); without that it is
left standing and the result says only the person can.

Optional ``ledger_callback`` lets the host application track which items
the LLM has seen.

``memory_add`` and ``memory_update`` require a ``conversation_id`` for every
persisted row; it is read from a per-call ``context_resolver`` (the canonical
``CallContext`` shape) so the live conversation attribution is never
bound at factory construction (a frozen attribution is a multiplex
violation). The LLM never sees ``conversation_id`` in the Pydantic
tool schema — the runtime injects it, full stop.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, date, datetime, time, tzinfo
from typing import Any, Literal
from uuid import UUID

from uuid_utils import uuid7

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field, model_validator

from threetears.agent.memory.revisions import (
    MemoryMarks,
    correct_beside,
    is_permanent,
    marks_of,
    restore,
    retract,
    supersede,
    write_once,
)
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
    MemoryConsolidationsCollection,
)
from langchain_core.embeddings import Embeddings

from threetears.agent.memory.embedding_utils import _safe_aembed_query
from threetears.agent.memory.types import MemoryType
from threetears.core.utils.local_time import render_local_time, resolve_zone
from threetears.observe import get_logger

__all__ = [
    "ChunkRecallInput",
    "ChunkSearchInput",
    "LedgerCallback",
    "MemoryAddInput",
    "MemoryForgetInput",
    "MemoryKeepInput",
    "MemoryListInput",
    "MemoryRecallInput",
    "MemoryRestoreInput",
    "MemorySearchInput",
    "MemoryUpdateInput",
    "PERMANENT_IS_THE_PERSONS",
    "NearMessageIds",
    "PERMANENT_NEEDS_THE_PERSON",
    "PermanentAllowed",
    "PersonAsked",
    "load_chunk_recall_tool",
    "load_chunk_search_tool",
    "load_memory_add_tool",
    "load_memory_forget_tool",
    "load_memory_keep_tool",
    "load_memory_list_tool",
    "load_memory_recall_tool",
    "load_memory_restore_tool",
    "load_memory_search_tool",
    "load_memory_update_tool",
]


log = get_logger(__name__)

_MAX_RESULTS = 10
_DEFAULT_SIMILARITY_THRESHOLD = 0.2
#: Most memories one search or listing shows.
_MAX_LIMIT = 50
#: Files and passages a memory search shows beside its memories.
_SIDE_RESULTS = 5
#: How much of a passage a list of passages shows.
_PASSAGE_PREVIEW_CHARS = 200
#: How much of a file's text a read by id shows.
_FILE_PREVIEW_CHARS = 500
#: How much of another memory's text a result quotes, and of each memory in a listing.
_QUOTE_CHARS = 300

# v0.7.5: alias validation. Letters, digits, hyphen, underscore. 1-64.
_ALIAS_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

# Type alias for the optional ledger callback
LedgerCallback = Callable[[str, str, str], Awaitable[None]]

#: Whether the person asked, in this turn, for this permanent memory to be
#: changed or retired. The host answers per call, by the memory's id.
PersonAsked = Callable[[UUID], Awaitable[bool]]

#: What a tool says when it leaves a permanent memory standing.
PERMANENT_IS_THE_PERSONS = "Only your person can change or retire a permanent memory."

#: Whether a memory with this text may be made permanent by the agent alone.
#: The host answers per call, from the text: ``False`` for one that needs the
#: person's word first (a memory the agent writes about its own failure).
PermanentAllowed = Callable[[str], Awaitable[bool]]

#: What a tool says when it stores or leaves a memory not permanent because
#: the host said that needs the person's word.
PERMANENT_NEEDS_THE_PERSON = "It is not kept permanently: making this one permanent needs your person's word."

#: The source messages of this turn and the one before it, read per call.
NearMessageIds = Callable[[], Awaitable[Sequence[UUID]]]

_VALID_MEMORY_TYPES = frozenset(t.value for t in MemoryType)
_TYPES_LISTED = ", ".join(sorted(_VALID_MEMORY_TYPES))
_ = MediaCollection  # imported for public symbol availability; factory users pass it via __init__ wiring


class MemorySearchInput(BaseModel):
    """Input schema for memory_search tool."""

    query: str = Field(default="", description="What to look for.")
    type_filter: str | None = Field(
        default=None,
        description="Only memories of this type: preference, fact, decision, topical_context or relational_context.",
    )
    ids: list[str] | None = Field(
        default=None,
        description="Ids to open without searching: the id from a [memory:<id>], [media:<id>] or [chunk:<id>] line.",
    )
    alias: str | None = Field(
        default=None,
        description="Open the memory given this name with memory_add.",
    )
    mode: str = Field(
        default="balanced",
        description="precise matches exact words, fuzzy matches loosely, balanced (default) is between.",
    )
    date_after: str | None = Field(
        default=None,
        description="ISO date or datetime. Only items written on or after it. A date alone starts at the day's first minute.",
    )
    date_before: str | None = Field(
        default=None,
        description="ISO date or datetime. Only items written on or before it. A date alone runs to the end of that day.",
    )
    limit: int = Field(default=_MAX_RESULTS, ge=1, le=_MAX_LIMIT, description="Most memories to show, 1-50.")
    offset: int = Field(default=0, ge=0, description="Memories to skip, to read past the first page.")

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
        description="The id from a [memory:<id>] line. Give this or alias.",
    )
    alias: str | None = Field(
        default=None,
        description="The name given to the memory with memory_add. Give this or memory_id.",
    )
    chunk_query: str | None = Field(
        default=None,
        description="Show only the parts that match this text.",
    )
    chunk_indexes: list[int] | None = Field(
        default=None,
        description="Show only the parts at these positions, counting from 0.",
    )
    chunk_id_after: str | None = Field(default=None, description="Show the parts after this [chunk:<id>].")
    chunk_id_before: str | None = Field(default=None, description="Show the parts before this [chunk:<id>].")
    limit: int = Field(default=5, ge=1, le=50, description="Most parts to show, 1-50.")

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


def _no_change(tool_name: str, what: str) -> str:
    """The result of a call that ran and changed nothing, in the platform's ``[NO CHANGE]`` form.

    The prefix is the one ``threetears.agent.tools.utils.tool_result_status``
    reads; this package does not import that one, so the text is written here.

    :param tool_name: the tool answering
    :ptype tool_name: str
    :param what: why nothing changed, and what stands instead
    :ptype what: str
    :return: the result text
    :rtype: str
    """
    return f"[NO CHANGE] {tool_name}: {what}"


def _fmt_dt(dt: Any, zone: tzinfo | None = None) -> str:
    """Format a datetime or None into a short timestamp string.

    With a zone the time is shown in it and names it (``Oct 9, 2026 at 8:00 AM PDT``).
    Without one the text is what it was before a caller could name a zone.

    :param dt: datetime or None
    :ptype dt: Any
    :param zone: the reader's zone, when the caller named one
    :ptype zone: tzinfo | None
    :return: formatted string or empty
    :rtype: str
    """
    if dt is None:
        return ""
    if zone is not None and isinstance(dt, datetime):
        return render_local_time(dt, zone)
    if hasattr(dt, "strftime"):
        return dt.strftime("%b %-d, %Y at %-I:%M %p")  # type: ignore[no-any-return]
    return str(dt)


def _cut(text: str, limit: int, rest: str) -> str:
    """``text`` whole, or its first ``limit`` characters with a mark saying it is cut and how to read the rest.

    :param text: the text
    :ptype text: str
    :param limit: most characters to show
    :ptype limit: int
    :param rest: which call reads it whole, as a clause
    :ptype rest: str
    :return: the text, marked when cut
    :rtype: str
    """
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}... (cut: the first {limit:,} of {len(text):,} characters; {rest})"


def _parse_bound(label: str, raw: str | None, zone: tzinfo | None) -> tuple[datetime | None, str | None]:
    """Read a date filter. A date alone is the whole day in the reader's zone; a time with no zone is in it too.

    ``date_before`` given as a bare date runs to the end of that day, so the day
    named is included. With no zone named the day is UTC's.

    :param label: ``date_after`` or ``date_before``
    :ptype label: str
    :param raw: what the caller passed
    :ptype raw: str | None
    :param zone: the reader's zone, when the caller named one
    :ptype zone: tzinfo | None
    :return: the moment, or an error to show
    :rtype: tuple[datetime | None, str | None]
    """
    if not raw or not raw.strip():
        return None, None
    text = raw.strip()
    home = zone or UTC
    try:
        day = date.fromisoformat(text) if len(text) <= 10 else None
    except ValueError:
        day = None
    if day is not None:
        edge = time.max if label == "date_before" else time.min
        return datetime.combine(day, edge, tzinfo=home), None
    try:
        value = datetime.fromisoformat(text)
    except ValueError:
        return None, f"Invalid {label} '{raw}'. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)."
    if value.tzinfo is None:
        value = value.replace(tzinfo=home)
    return value, None


def _parse_memory_id(raw: str) -> UUID | None:
    """A memory id as an agent writes one: bare, or the whole ``[memory:<id>]`` mark.

    :param raw: what the caller passed
    :ptype raw: str
    :return: the id, or ``None`` when it is not one
    :rtype: UUID | None
    """
    try:
        return UUID(raw.strip().removeprefix("[memory:").removesuffix("]").strip())
    except ValueError:  # NOSILENT: "not an id" is the answer; every caller turns None into the tool's own error text
        return None


def _mark_text(marks: MemoryMarks, zone: tzinfo | None) -> str:
    """A memory's state as its line says it, each part in brackets, before the memory's text.

    :param marks: the memory's marks
    :ptype marks: MemoryMarks
    :param zone: the reader's zone
    :ptype zone: tzinfo | None
    :return: the marks, each followed by a space; empty for a plain current memory
    :rtype: str
    """
    parts: list[str] = []
    if marks.alias:
        parts.append(f"(alias: {marks.alias})")
    if marks.permanent:
        parts.append("(permanent)")
    if marks.retired:
        why = f": {marks.retired_because}" if marks.retired_because else ""
        parts.append(f"(retired, no longer holds{why})")
    if marks.replaced_by is not None:
        when = _fmt_dt(marks.replaced_at, zone)
        on = f" on {when}" if when else ""
        parts.append(f"(replaced by [memory:{marks.replaced_by}]{on}; that one is current)")
    if marks.corrected_by is not None:
        parts.append(f"(corrected by [memory:{marks.corrected_by}])")
    return "".join(f"{p} " for p in parts)


def _memory_line(row: dict[str, Any], zone: tzinfo | None, *, quote: int | None = None) -> str:
    """One memory as a list line: id, type, when, how it matched, its state, then its text.

    :param row: a memory row with the mark columns
    :ptype row: dict[str, Any]
    :param zone: the reader's zone
    :ptype zone: tzinfo | None
    :param quote: cut the text to this many characters, marked; ``None`` shows it whole
    :ptype quote: int | None
    :return: the line, without a leading dash
    :rtype: str
    """
    mid = str(row["memory_id"])
    kind = row.get("type") or row.get("type_memory")
    line = f"[memory:{mid}] [{kind}] "
    when = _fmt_dt(row.get("date_created"), zone)
    if when:
        line += f"[{when}] "
    if "similarity" in row:
        score = row["similarity"]
        line += "[word match] " if score is None else f"[match {float(score):.2f}] "
    content = str(row.get("content") or "")
    if quote is not None:
        content = _cut(content, quote, f"memory_recall reads [memory:{mid}] whole")
    return f"{line}{_mark_text(marks_of(row), zone)}{content}"


def _quoted(row: dict[str, Any]) -> str:
    """Another memory's text inside a sentence: in quotes, cut and marked when long.

    :param row: the memory's row
    :ptype row: dict[str, Any]
    :return: the quoted text
    :rtype: str
    """
    mid = str(row["memory_id"])
    return f'"{_cut(str(row.get("content") or ""), _QUOTE_CHARS, f"memory_recall reads [memory:{mid}] whole")}"'


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


def _passage_source(*, title: str | None, page: Any, heading: str | None) -> str:
    """Where a passage came from, as far as its row says.

    :param title: the source file's title
    :ptype title: str | None
    :param page: its page number
    :ptype page: Any
    :param heading: the heading it sits under
    :ptype heading: str | None
    :return: e.g. ``"Trip notes" p.3 "Day two"``, or empty when nothing is known
    :rtype: str
    """
    parts = []
    if title:
        parts.append(f'"{title}"')
    if page:
        parts.append(f"p.{page}")
    if heading:
        parts.append(f'"{heading}"')
    return " ".join(parts)


async def _refused(
    action: Literal["memory.read", "memory.write"],
    tool_name: str,
    *,
    agent_id: UUID,
    customer_id: UUID,
    user_id: UUID,
    authorizer: MemoryAuthorizerDependencies,
) -> str | None:
    """Ask whether the caller may read or write these memories; the refusal as a tool result, or ``None``.

    A read is asked as the agent reading its own namespace; a write as the
    person, which is how every write here has always been asked.

    :param action: ``memory.read`` or ``memory.write``
    :ptype action: Literal["memory.read", "memory.write"]
    :param tool_name: the tool answering
    :ptype tool_name: str
    :param agent_id: owning agent UUID
    :ptype agent_id: UUID
    :param customer_id: owning customer UUID
    :ptype customer_id: UUID
    :param user_id: the caller
    :ptype user_id: UUID
    :param authorizer: rbac authorizer dependency bundle
    :ptype authorizer: MemoryAuthorizerDependencies
    :return: the refusal text, or ``None`` when allowed
    :rtype: str | None
    """
    try:
        await authorize_memory_access(
            action=action,
            agent_id=agent_id,
            customer_id=customer_id,
            caller_user_id=user_id,
            caller_agent_id=agent_id if action == ACTION_MEMORY_READ else None,
            deps=authorizer,
        )
    except MemoryAccessDenied as exc:
        return _tool_error(tool_name, "authorize", str(exc))
    return None


async def _may_touch_permanent(person_asked: PersonAsked | None, memory_id: UUID) -> bool:
    """Whether the host says the person asked for this permanent memory to be changed.

    No resolver, or one that fails, is a no: a permanent memory is left alone
    unless something positively says otherwise.

    :param person_asked: the host's resolver, if it gave one
    :ptype person_asked: PersonAsked | None
    :param memory_id: the permanent memory
    :ptype memory_id: UUID
    :return: ``True`` only when the resolver says so
    :rtype: bool
    """
    if person_asked is None:
        return False
    try:
        return bool(await person_asked(memory_id))
    except (
        Exception
    ):  # prawduct:allow prawduct/broad-except -- a host resolver may fail in any way, and every way is a no
        log.exception(
            "person_asked resolver failed; the permanent memory is left alone",
            extra={"extra_data": {"memory_id": str(memory_id)}},
        )
        return False


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
    zone: tzinfo | None = None,
    file_reader: str | None = None,
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
    :param zone: the reader's zone, for every date shown
    :ptype zone: tzinfo | None
    :param file_reader: the host's tool that reads a file whole by its media id, named where a file's text is cut
    :ptype file_reader: str | None
    :return: formatted tool output
    :rtype: str
    """
    valid_uuids: list[UUID] = []
    unread: list[str] = []
    for raw_id in ids:
        text = raw_id.strip()
        for prefix in ("[memory:", "[media:", "[chunk:"):
            text = text.removeprefix(prefix)
        try:
            valid_uuids.append(UUID(text.removesuffix("]").strip()))
        except ValueError:
            unread.append(raw_id[:64])
    not_ids = (
        f"(Not ids, so not looked up: {', '.join(unread)}. An id is the part after the colon in a "
        "[memory:<id>], [media:<id>] or [chunk:<id>] mark.)"
        if unread
        else ""
    )
    if not valid_uuids:
        return f"[TOOL ERROR] No valid UUIDs provided. {not_ids}".rstrip()

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
            parts.append(f"- {_memory_line(row, zone)}")
            if ledger_callback:
                await ledger_callback(mid, "memory", row["content"])

    if mc_rows:
        if parts:
            parts.append("")
        parts.append(f"Found {len(mc_rows)} files:")
        for row in mc_rows:
            cid = str(row["content_id"])
            title = _extract_title(row["metadata_json"])
            label = title or row.get("media_category", "media")
            media_id = row.get("media_id")
            whole = (
                f"{file_reader} reads it whole by its media id {media_id}"
                if file_reader and media_id is not None
                else "the rest is not shown here"
            )
            content = _cut(row["content"], _FILE_PREVIEW_CHARS, whole)
            when = _fmt_dt(row.get("date_created"), zone)
            stamp = f" [{when}]" if when else ""
            parts.append(f"- [media:{cid}] [{label}]{stamp} {content}")
            if ledger_callback:
                await ledger_callback(cid, "media", label or content)

    if chunk_rows:
        if parts:
            parts.append("")
        parts.append(f"Found {len(chunk_rows)} passages:")
        for row in chunk_rows:
            ckid = str(row["chunk_id"])
            title = _extract_title(row["metadata_json"])
            location = (
                _passage_source(title=title, page=row.get("page_number"), heading=row.get("heading_context"))
                or "unknown source"
            )
            when = _fmt_dt(row.get("date_created"), zone)
            stamp = f" [{when}]" if when else ""
            parts.append(f"- [chunk:{ckid}] [from {location}]{stamp} {row['content']}")
            if ledger_callback:
                await ledger_callback(ckid, "chunk", title or row["content"])

    if not parts:
        return f"[TOOL ERROR] No items found for the provided IDs. {not_ids}".rstrip()

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

    body = "\n".join(parts)
    # what was not looked up is said first: a reader that keeps only the head still reads it
    return f"{not_ids}\n\n{body}" if not_ids else body


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
    tz: tzinfo | str | None = None,
    file_reader: str | None = None,
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
    :param max_results: how many memories a call shows when it names no ``limit``
    :ptype max_results: int
    :param ledger_callback: optional coroutine invoked per surfaced
        item with ``(item_id, item_type, description)``
    :ptype ledger_callback: LedgerCallback | None
    :param tz: the zone of the person the results are read to: a ``tzinfo``
        or an IANA name. Every date in a result is shown in it and names it,
        and a date filter given as a bare date is that zone's day.
        ``None`` keeps the dates as stored, with no zone named, and UTC's day
    :ptype tz: tzinfo | str | None
    :param file_reader: the name of the host's tool that reads a file whole by
        its media id; named where a file's text is cut
    :ptype file_reader: str | None
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    :raises ValueError: when ``tz`` is a name that is not a known zone
    """
    sim_threshold = similarity_threshold
    zone = resolve_zone(tz)
    default_limit = max(1, min(max_results, _MAX_LIMIT))

    @tool("memory_search", args_schema=MemorySearchInput)
    async def memory_search(
        query: str = "",
        type_filter: str | None = None,
        ids: list[str] | None = None,
        alias: str | None = None,
        mode: str = "balanced",
        date_after: str | None = None,
        date_before: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> str:
        """search the user's memories and document knowledge base."""
        denied = await _refused(
            ACTION_MEMORY_READ,
            "memory_search",
            agent_id=agent_id,
            customer_id=customer_id,
            user_id=user_id,
            authorizer=authorizer,
        )
        if denied:
            return denied
        shown = default_limit if limit is None else max(1, min(limit, _MAX_LIMIT))
        skip = max(0, offset)

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
                return f"[TOOL ERROR] No memory found with alias '{alias}'."
            mid = str(row["memory_id"])
            if ledger_callback:
                await ledger_callback(mid, "memory", row["content"])
            return f"Found memory by alias '{alias}':\n- {_memory_line({**row, 'alias': None}, zone)}"

        if ids:
            return await _search_by_ids(
                memories_collection,
                media_content_collection,
                memory_chunk_collection,
                user_id,
                ids,
                agent_id=agent_id,
                ledger_callback=ledger_callback,
                zone=zone,
                file_reader=file_reader,
            )

        if not query:
            return "[TOOL ERROR] Provide one of 'query', 'ids', or 'alias'."

        # Either bound may stand alone. A bare date is the reader's whole day.
        date_after_dt, err_a = _parse_bound("date_after", date_after, zone)
        if err_a:
            return _tool_error("memory_search", "filter", err_a)
        date_before_dt, err_b = _parse_bound("date_before", date_before, zone)
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

        if type_filter and type_filter not in _VALID_MEMORY_TYPES:
            return _tool_error(
                "memory_search",
                "filter",
                f"Invalid type_filter '{type_filter}'. Must be one of: {_TYPES_LISTED}",
            )

        # Branches that failed outright. A search that lost a branch must not be reported as a
        # search that found nothing -- the agent would tell the user it has no such memory.
        degraded: list[str] = []

        # One more than the page from each branch, so a page that is full can say whether
        # more match. A retired memory is in neither branch: it no longer holds.
        reach = skip + shown + 1
        memories: list[dict[str, Any]] = []
        if run_semantic and embedding is not None:
            try:
                memories = await memories_collection.search_by_semantic(
                    user_id=user_id,
                    agent_id=agent_id,
                    embedding=embedding,
                    max_results=reach,
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
                    max_results=reach,
                    type_filter=type_filter,
                    date_after=date_after_dt,
                    date_before=date_before_dt,
                )
                for row in fts_rows:
                    mid = row["memory_id"]
                    if mid not in seen_ids:
                        seen_ids.add(mid)
                        memories.append({**row, "similarity": None})
            except Exception:
                degraded.append("memory full-text search")
                log.exception(
                    "memory_search full-text branch failed",
                    extra={"extra_data": {"user_id": user_id, "agent_id": agent_id}},
                )

        more_memories = len(memories) > skip + shown
        memories = memories[skip : skip + shown]

        media_results: list[dict[str, Any]] = []
        mc_seen: set[str] = set()
        try:
            if embedding is not None:
                mc_rows = await media_content_collection.search_by_semantic(
                    user_id=user_id,
                    agent_id=agent_id,
                    embedding=embedding,
                    max_results=_SIDE_RESULTS + 1,
                    similarity_threshold=sim_threshold,
                    date_after=date_after_dt,
                    date_before=date_before_dt,
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
                    max_results=_SIDE_RESULTS + 1,
                    date_after=date_after_dt,
                    date_before=date_before_dt,
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
            degraded.append("media search")
            log.exception(
                "memory_search media branch failed",
                extra={"extra_data": {"user_id": user_id, "agent_id": agent_id}},
            )
        more_files = len(media_results) > _SIDE_RESULTS
        media_results = media_results[:_SIDE_RESULTS]

        doc_chunks: list[dict[str, Any]] = []
        try:
            if embedding is not None:
                chunk_rows = await memory_chunk_collection.search_by_semantic(
                    user_id=user_id,
                    agent_id=agent_id,
                    embedding=embedding,
                    max_results=_SIDE_RESULTS + 1,
                    similarity_threshold=sim_threshold,
                    date_after=date_after_dt,
                    date_before=date_before_dt,
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
                            "date_created": row.get("date_created"),
                            "similarity": row["similarity"],
                        }
                    )
        except Exception:
            degraded.append("document chunk search")
            log.exception(
                "memory_search document-chunk branch failed",
                extra={"extra_data": {"user_id": user_id, "agent_id": agent_id}},
            )
        more_passages = len(doc_chunks) > _SIDE_RESULTS
        doc_chunks = doc_chunks[:_SIDE_RESULTS]

        if not memories and not media_results and not doc_chunks:
            if degraded:
                # Empty because the search broke, not because there is nothing stored. Saying
                # "no memories found" here is the answer that turns an outage into a confident lie.
                return _tool_error("memory_search", "query", f"search incomplete: {', '.join(degraded)} failed")
            if skip:
                return f"No more memories match this query past the first {skip}."
            return "No relevant memories or documents found for this query."

        # What is true of the result as a whole, said first: a reader that keeps only the
        # head of a long result must still read that the recall is incomplete or has more.
        notes: list[str] = []
        if degraded:
            notes.append(f"(Partial results: {', '.join(degraded)} failed, so more may exist.)")
        if more_memories:
            notes.append(
                f"(More memories may match than the {len(memories)} shown: call again with offset={skip + len(memories)}.)"
            )
        elif skip and memories:
            notes.append(f"(Memories {skip + 1}-{skip + len(memories)}; no more match.)")
        if more_files or more_passages:
            which = " and ".join(w for w, more in (("files", more_files), ("passages", more_passages)) if more)
            notes.append(
                f"(More {which} may match than the {_SIDE_RESULTS} shown: narrow the query, or use chunk_search for passages.)"
            )

        sections: list[list[str]] = []
        if notes:
            sections.append(notes)

        parts: list[str] = []
        if memories:
            parts.append(f"Found {len(memories)} relevant memories:")
            parts.extend(f"- {_memory_line(m, zone)}" for m in memories)
            sections.append(parts)

        if media_results:
            parts = [f"Found {len(media_results)} files:"]
            for mr in media_results:
                ts = _fmt_dt(mr.get("date_created"), zone)
                ts_suffix = f" (created: {ts})" if ts else ""
                tag = f"[media:{mr['content_id']}]"
                if mr.get("media_category") == "image":
                    parts.append(f"- {tag} {mr['content']}{ts_suffix}")
                else:
                    label = mr.get("title") or mr.get("media_category", "media")
                    parts.append(f"- {tag} [{label}] {mr['content']}{ts_suffix}")
            sections.append(parts)

        if doc_chunks:
            parts = [f"Found {len(doc_chunks)} passages:"]
            for c in doc_chunks:
                location = (
                    _passage_source(title=c.get("title"), page=c.get("page"), heading=c.get("heading"))
                    or "unknown source"
                )
                tag = f"[chunk:{c['chunk_id']}]"
                when = _fmt_dt(c.get("date_created"), zone)
                stamp = f" [{when}]" if when else ""
                parts.append(f"- {tag} [from {location}]{stamp} {c['content']}")
            sections.append(parts)

        if ledger_callback:
            for m in memories:
                await ledger_callback(m["memory_id"], "memory", m["content"])
            for mr in media_results:
                desc = mr.get("title") or mr["content"]
                await ledger_callback(mr["content_id"], "media", desc)
            for c in doc_chunks:
                desc = c.get("title") or c["content"]
                await ledger_callback(c["chunk_id"], "chunk", desc)

        return "\n\n".join("\n".join(section) for section in sections)

    memory_search.description = (
        "Search your memories, the files you have seen and your past conversations. "
        "Use it first when a topic comes back. Returns [memory:<id>], [media:<id>] "
        "and [chunk:<id>] lines; a line says when a memory is permanent or was replaced, "
        "and a retired memory is not found. Pass ids to open items you already have, "
        "alias to open a named memory, date_after or date_before to limit by date, and "
        "limit and offset to read further. memory_list browses without a query."
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
    *,
    tz: tzinfo | str | None = None,
    consolidations_collection: MemoryConsolidationsCollection | None = None,
    salience_bump: float = 0.0,
) -> list[BaseTool]:
    """create a memory_recall tool that returns a memory + its chunks.

    With ``salience_bump`` above zero, opening a memory raises its salience by
    that much (permanent ones are pinned and skipped): the recall the agent
    asked for is what keeps a memory from sinking out of ambient recall, for a
    consumer whose ambient recall no longer does
    (``MemoryConfig.ambient_bumps_salience``).

    v0.7.0 (transcript-chunks shard C) replaced the old ``(id, type)``
    discriminator surface with a memory-centric one: the tool takes
    one ``memory_id`` plus an optional chunk-selection mode and
    returns the parent memory's content + a window into its chunks.

    The result opens with a header line -- the id, type, when it was written,
    its name, and whether it is permanent, retired, replaced or corrected --
    then, for a replaced or corrected memory, the text of the one that stands
    in its place, then the memory's own text, then its parts: which ones of how
    many, the cursor for the next, and each cut preview marked as cut.

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
    :param similarity_threshold: floor on semantic similarity for
        admission in ``chunk_query`` mode
    :ptype similarity_threshold: float
    :param tz: the reader's zone, a ``tzinfo`` or an IANA name; every date is shown in it
    :ptype tz: tzinfo | str | None
    :param consolidations_collection: the edges between a memory and what it
        replaced; with it the header names the memories this one replaced
    :ptype consolidations_collection: MemoryConsolidationsCollection | None
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    :raises ValueError: when ``tz`` is a name that is not a known zone
    """
    # kept for signature parity; no consumer in the v0.7.0 body.
    _ = media_content_collection
    zone = resolve_zone(tz)

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
        denied = await _refused(
            ACTION_MEMORY_READ,
            "memory_recall",
            agent_id=agent_id,
            customer_id=customer_id,
            user_id=user_id,
            authorizer=authorizer,
        )
        if denied:
            return denied

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
                return f"[TOOL ERROR] No memory found with alias '{alias}'."
            memory_id = str(row["memory_id"])

        mem_uuid = _parse_memory_id(memory_id)
        if mem_uuid is None:
            return f"[TOOL ERROR] Invalid memory_id format: {memory_id}"

        # ── the memory and its state ────────────────────────────
        memory = await memories_collection.find_for_recall(
            memory_id=mem_uuid,
            user_id=user_id,
            agent_id=agent_id,
        )
        if memory is None:
            return "[TOOL ERROR] No memory with that id that you can read."
        marks = marks_of(memory)
        if salience_bump > 0 and marks.current:
            try:
                await memories_collection.bump_salience([mem_uuid], agent_id=agent_id, access_bump=salience_bump)
            except Exception as exc:  # prawduct:allow prawduct/broad-except -- reinforcement is bookkeeping; the read it follows must still answer
                log.warning("memory_recall: salience bump failed", extra={"extra_data": {"error": str(exc)}})

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
                    mode_name = "default"
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
                    # a ranked hit carries no position; read each one's from its row
                    placed = await memory_chunk_collection.search_by_ids(
                        [UUID(str(ch["chunk_id"])) for ch in chunks], user_id, agent_id=agent_id
                    )
                    index_of = {str(r["chunk_id"]): r.get("chunk_index") for r in placed}
                    chunks = [{**ch, "chunk_index": index_of.get(str(ch["chunk_id"]))} for ch in chunks]

            elif chunk_indexes is not None and chunk_indexes:
                mode_name = "chunk_indexes"
                chunks = await memory_chunk_collection.find_by_chunk_indexes(
                    mem_uuid,
                    user_id=user_id,
                    agent_id=agent_id,
                    customer_id=customer_id,
                    chunk_indexes=chunk_indexes,
                )

            elif chunk_id_after is not None and chunk_id_after.strip():
                mode_name = "chunk_id_after"
                try:
                    cur = UUID(chunk_id_after)
                except ValueError:
                    return f"[TOOL ERROR] Invalid chunk_id_after format: {chunk_id_after}"
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
                    return f"[TOOL ERROR] Invalid chunk_id_before format: {chunk_id_before}"
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
            total = await memory_chunk_collection.count_by_memory_id(
                mem_uuid, user_id=user_id, agent_id=agent_id, customer_id=customer_id
            )
        except Exception as exc:
            return _tool_error("memory_recall", f"fetch chunks ({mode_name})", str(exc))

        # ── render: the header, what stands in its place, the text, the parts ──
        written = _fmt_dt(memory.get("date_created"), zone)
        header = f"[memory:{mem_uuid}] [{memory['type_memory']}]"
        if written:
            header += f" written {written}"
        state = _mark_text(marks, zone).rstrip()
        if state:
            header += f" {state}"
        lines: list[str] = [header]
        for label, other_id in (("Replaced by", marks.replaced_by), ("Corrected by", marks.corrected_by)):
            if other_id is None:
                continue
            other = await memories_collection.find_for_recall(memory_id=other_id, user_id=user_id, agent_id=agent_id)
            if other is not None:
                lines.append(f"{label} [memory:{other_id}]: {_quoted(other)}")
        if consolidations_collection is not None:
            try:
                sources = await consolidations_collection.find_sources(agent_id, mem_uuid)
            except (
                Exception
            ):  # prawduct:allow prawduct/broad-except -- the header's last fact is optional; the memory is still read
                log.exception("memory_recall: reading what a memory replaced failed")
                sources = []
            if sources:
                lines.append("It replaced or corrects: " + ", ".join(f"[memory:{s}]" for s in sources) + ".")
        lines.append(str(memory["content"]))
        lines.append("")
        if not chunks:
            lines.append(
                "(no parts)"
                if total == 0
                else f"(none of its {total} parts match that; call again with no part filter for the first ones)"
            )
            return "\n".join(lines)

        positions = [ch.get("chunk_index") for ch in chunks]
        known = [p for p in positions if isinstance(p, int)]
        if mode_name in {"default", "chunk_id_after", "chunk_id_before"} and len(known) == len(chunks):
            span = f"Parts {known[0]}-{known[-1]} of {total}." if len(known) > 1 else f"Part {known[0]} of {total}."
            if known[-1] < total - 1:
                span += f" Next: chunk_id_after={chunks[-1]['chunk_id']}."
            if known[0] > 0:
                span += f" Earlier: chunk_id_before={chunks[0]['chunk_id']}."
        else:
            which = "matching the text" if mode_name == "chunk_query" else "asked for"
            span = f"{len(chunks)} of {total} parts, the ones {which}."
        lines.append(span)
        for ch in chunks:
            cid = ch.get("chunk_id", "?")
            idx = ch.get("chunk_index")
            place = f"part {idx}" if isinstance(idx, int) else "part"
            preview = _cut(
                str(ch.get("content") or ""), _PASSAGE_PREVIEW_CHARS, f"chunk_recall reads [chunk:{cid}] whole"
            )
            lines.append(f"  [chunk:{cid}] ({place})\n  {preview}")
        return "\n".join(lines)

    memory_recall.description = (
        "Read a whole memory by its [memory:<id>] or its alias: when it was written, whether it "
        "is permanent, retired or replaced, and the first parts of what it came from. A retired "
        "or replaced memory still opens here. Pass chunk_query to show only the parts about "
        "something, or chunk_indexes, chunk_id_after or chunk_id_before to show other parts."
    )

    return [memory_recall]


class MemoryListInput(BaseModel):
    """Input schema for memory_list."""

    date_after: str | None = Field(
        default=None,
        description="ISO date or datetime. Only memories written on or after it.",
    )
    date_before: str | None = Field(
        default=None,
        description="ISO date or datetime. Only memories written on or before it. A date alone runs to the end of that day.",
    )
    memory_type: str | None = Field(
        default=None,
        description="Only this type: preference, fact, decision, topical_context or relational_context.",
    )
    permanent: bool | None = Field(default=None, description="True for only permanent memories, false for the rest.")
    has_alias: bool | None = Field(default=None, description="True for only named memories, false for unnamed ones.")
    show: str = Field(
        default="current",
        description="current (default) lists what still holds; retired, replaced or all lists the others.",
    )
    limit: int = Field(default=20, ge=1, le=_MAX_LIMIT, description="Most memories to list, 1-50.")
    offset: int = Field(default=0, ge=0, description="Memories to skip, newest first.")

    @model_validator(mode="after")
    def _validate(self) -> "MemoryListInput":
        if self.show not in {"current", "retired", "replaced", "all"}:
            raise ValueError(f"show must be 'current', 'retired', 'replaced' or 'all'; got {self.show!r}.")
        return self


def load_memory_list_tool(
    user_id: UUID,
    agent_id: UUID,
    customer_id: UUID,
    authorizer: MemoryAuthorizerDependencies,
    memories_collection: MemoriesCollection,
    *,
    tz: tzinfo | str | None = None,
) -> list[BaseTool]:
    """create memory_list: browse memories newest first, by date, type and state, with no query.

    :param user_id: the caller; only their memories are listed
    :ptype user_id: UUID
    :param agent_id: the partition
    :ptype agent_id: UUID
    :param customer_id: the tenant
    :ptype customer_id: UUID
    :param authorizer: rbac authorizer dependency bundle
    :ptype authorizer: MemoryAuthorizerDependencies
    :param memories_collection: the memories collection
    :ptype memories_collection: MemoriesCollection
    :param tz: the reader's zone, a ``tzinfo`` or an IANA name: every date is
        shown in it, and a bare date filter is that zone's day
    :ptype tz: tzinfo | str | None
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    :raises ValueError: when ``tz`` is a name that is not a known zone
    """
    zone = resolve_zone(tz)

    @tool("memory_list", args_schema=MemoryListInput)
    async def memory_list(
        date_after: str | None = None,
        date_before: str | None = None,
        memory_type: str | None = None,
        permanent: bool | None = None,
        has_alias: bool | None = None,
        show: str = "current",
        limit: int = 20,
        offset: int = 0,
    ) -> str:
        """list memories newest first."""
        denied = await _refused(
            ACTION_MEMORY_READ,
            "memory_list",
            agent_id=agent_id,
            customer_id=customer_id,
            user_id=user_id,
            authorizer=authorizer,
        )
        if denied:
            return denied
        if show not in {"current", "retired", "replaced", "all"}:
            return _tool_error("memory_list", "filter", f"show '{show}' is not one of: current, retired, replaced, all")
        if memory_type and memory_type not in _VALID_MEMORY_TYPES:
            return _tool_error(
                "memory_list", "filter", f"Invalid memory_type '{memory_type}'. Must be one of: {_TYPES_LISTED}"
            )
        after, err_a = _parse_bound("date_after", date_after, zone)
        if err_a:
            return _tool_error("memory_list", "filter", err_a)
        before, err_b = _parse_bound("date_before", date_before, zone)
        if err_b:
            return _tool_error("memory_list", "filter", err_b)
        page = max(1, min(limit, _MAX_LIMIT))
        skip = max(0, offset)
        try:
            rows, total = await memories_collection.list_for_user(
                user_id=user_id,
                agent_id=agent_id,
                date_after=after,
                date_before=before,
                type_filter=memory_type or None,
                permanent=permanent,
                has_alias=has_alias,
                state=show,
                limit=page,
                offset=skip,
            )
        except Exception as exc:
            return _tool_error("memory_list", "query", str(exc))
        what = {
            "current": "memories",
            "retired": "retired memories",
            "replaced": "replaced memories",
            "all": "memories of any state",
        }[show]
        if not rows:
            if total:
                return f"{total} {what} match, and offset={skip} is past the last of them."
            return f"0 {what} match."
        last = skip + len(rows)
        head = f"{total} {what}, newest first. Showing {skip + 1}-{last}"
        head += f"; offset={last} shows the next." if last < total else "; that is all of them."
        return "\n".join([head, *(f"- {_memory_line(row, zone, quote=_QUOTE_CHARS)}" for row in rows)])

    memory_list.description = (
        "List your memories newest first, with no query: what you saved in a span of dates, "
        "your permanent ones, your named ones, one type, or the retired and replaced ones. "
        "Says how many there are in all. memory_search finds by meaning."
    )

    return [memory_list]


async def _may_be_permanent(permanent_allowed: PermanentAllowed | None, content: str) -> bool:
    """Whether the host lets the agent make a memory with this text permanent on its own.

    No resolver is a yes: the rule is the host's to turn on. One that fails is
    a no, because a memory wrongly left to fade can be kept later and one
    wrongly made permanent cannot be undone by the agent.

    :param permanent_allowed: the host's resolver, if it gave one
    :ptype permanent_allowed: PermanentAllowed | None
    :param content: the memory's text
    :ptype content: str
    :return: ``False`` only when the resolver says no, or fails
    :rtype: bool
    """
    if permanent_allowed is None:
        return True
    try:
        return bool(await permanent_allowed(content))
    except (
        Exception
    ):  # prawduct:allow prawduct/broad-except -- a host resolver may fail in any way, and every way is a no
        log.exception("permanent_allowed resolver failed; the memory is not made permanent")
        return False


class MemoryAddInput(BaseModel):
    """Input schema for memory_add tool.

    Note: ``conversation_id`` is intentionally NOT in this schema —
    the LLM never sees or sets it. The tool runner reads it per-call
    from the live :class:`CallContext` via the factory's
    ``context_resolver`` so the memory is always attributed to the
    actively-running conversation.
    """

    content: str = Field(
        description="What to remember, in your own words.",
    )
    memory_type: str = Field(
        default="fact",
        description="One of: preference, fact, decision, topical_context, relational_context.",
    )
    alias: str | None = Field(
        default=None,
        description=(
            "Optional name to open it by later, e.g. 'cave-altar'. 1-64 letters, "
            "digits, hyphens or underscores. Each name is used once."
        ),
    )
    permanent: bool = Field(
        default=False,
        description=(
            "True for what must never fade or be merged away: who someone is, a promise, "
            "anything it would be wrong to forget."
        ),
    )
    replace: bool = Field(
        default=True,
        description=(
            "False to keep a nearly identical memory you already have standing beside this one, "
            "when they are two facts and not one."
        ),
    )
    corrects: list[str] | None = Field(
        default=None,
        description=(
            "Ids of the memories this one corrects, from their [memory:<id>] lines. Each is replaced "
            "by this one and stays readable by its id."
        ),
    )


def _live_context(tool_name: str, context_resolver: Callable[[], Any]) -> tuple[UUID, UUID] | str:
    """Read the live conversation and the message a write is attributed to, or the error to return.

    Binding these at factory construction would attribute every memory to
    whatever conversation the pod was first minted under -- a multiplex
    violation in any pod handling more than one conversation.

    :param tool_name: the tool answering
    :ptype tool_name: str
    :param context_resolver: the host's per-call context reader
    :ptype context_resolver: Callable[[], Any]
    :return: ``(conversation_id, message_id)``, or the tool's error text
    :rtype: tuple[UUID, UUID] | str
    """
    try:
        live_ctx = context_resolver()
    except Exception as exc:
        return _tool_error(tool_name, "context", f"context_resolver raised {type(exc).__name__}: {exc}")
    conversation_id = getattr(live_ctx, "conversation_id", None)
    message_id = getattr(live_ctx, "correlation_id", None)
    if conversation_id is None:
        return _tool_error(tool_name, "context", "context_resolver returned no conversation_id")
    if message_id is None:
        # source attribution falls back to a fresh UUID when no correlation is
        # available (smoke fixtures, hub-direct calls without an envelope).
        # uuid_utils.uuid7() returns a uuid_utils.UUID; cast so downstream
        # stores see stdlib UUID (the canonical type across 3tears).
        message_id = UUID(str(uuid7()))
    return conversation_id, message_id


async def _correction_targets(
    tool_name: str,
    memories_collection: MemoriesCollection,
    raw_ids: list[str],
    *,
    user_id: UUID,
    agent_id: UUID,
    zone: tzinfo | None,
) -> list[dict[str, Any]] | str:
    """Read the memories a correction names, or the refusal that stops it before anything is written.

    Refused: an id that is not one, a memory that is not this person's, one
    already retired, and one already replaced (its replacement is the one to
    correct).

    :param tool_name: the tool answering
    :ptype tool_name: str
    :param memories_collection: the memories collection
    :ptype memories_collection: MemoriesCollection
    :param raw_ids: the ids as the caller wrote them
    :ptype raw_ids: list[str]
    :param user_id: the caller
    :ptype user_id: UUID
    :param agent_id: the partition
    :ptype agent_id: UUID
    :param zone: the reader's zone
    :ptype zone: tzinfo | None
    :return: the rows, in the order named and without repeats, or the refusal text
    :rtype: list[dict[str, Any]] | str
    """
    rows: list[dict[str, Any]] = []
    seen: set[UUID] = set()
    for raw in raw_ids:
        target = _parse_memory_id(raw)
        if target is None:
            return _tool_error(tool_name, "memory_id", f"'{raw[:64]}' is not a memory id. Nothing was stored.")
        if target in seen:
            continue
        seen.add(target)
        row = await memories_collection.find_for_recall(memory_id=target, user_id=user_id, agent_id=agent_id)
        if row is None:
            return _no_change(tool_name, f"No memory [memory:{target}] that is yours. Nothing was stored.")
        marks = marks_of(row)
        if marks.retired:
            return _no_change(
                tool_name,
                f"[memory:{target}] was retired and no longer holds, so there is nothing to correct. "
                "Nothing was stored. Store this as a new memory without it, or memory_restore it first.",
            )
        if marks.replaced_by is not None:
            return _no_change(
                tool_name,
                f"[memory:{target}] was already replaced by [memory:{marks.replaced_by}]"
                f"{' on ' + _fmt_dt(marks.replaced_at, zone) if marks.replaced_at else ''}. "
                "Nothing was stored. Correct that one instead.",
            )
        rows.append(row)
    return rows


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
    consolidations_collection: MemoryConsolidationsCollection | None = None,
    tz: tzinfo | str | None = None,
    person_asked: PersonAsked | None = None,
    permanent_allowed: PermanentAllowed | None = None,
    near_message_ids: NearMessageIds | None = None,
    same_event_threshold: float | None = None,
) -> list[BaseTool]:
    """create a memory_add tool for explicit memory storage.

    **One event, written once.** With ``near_message_ids`` and
    ``same_event_threshold`` a memory that says nearly what one from this turn
    or the one before already says replaces it
    (:func:`~threetears.agent.memory.revisions.write_once`): post-turn
    extraction and the agent's own save of the same event are one memory. The
    result says so in the words a near-duplicate's replacement uses.
    ``replace=false`` writes beside it, as for any near-duplicate.

    **Permanent, when the host says that needs the person.** With
    ``permanent_allowed`` a ``permanent=true`` the resolver refuses is stored
    as an ordinary memory, and the result says why.

    data-layer-task-01 sub-task 4: ``conversation_id`` + ``message_id``
    are resolved per-call via ``context_resolver`` rather than bound at
    factory time. binding them at construction was the latent multi-
    plex violation surfaced by Group D D4-2: an agent pod multiplexes
    many conversations and a tool minted with one conversation's id
    would attribute every subsequent call's memory to that frozen
    conversation. the per-call resolver mirrors the
    :class:`NatsToolWrapper._arun` shape that already reads identity
    from :class:`CallContext` at invocation time.

    **What the result says of other memories.** A near-duplicate it replaced
    is quoted with its date. One it left standing -- because it is permanent,
    or the call said ``replace=false`` -- is quoted too, so the caller reads
    that both now stand. With ``corrects`` each named memory is replaced and
    quoted; a permanent one is replaced only when ``person_asked`` says so,
    and otherwise the new memory is written beside it and linked.

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
        replaced instead of creating a duplicate. default 0.90
    :ptype similarity_dedup_threshold: float
    :param salience_seed: starting salience for a newly-added memory
        (design §3 per-user knob). default 0.5 matches the DB server
        default; a tuned value is honored on insert
    :ptype salience_seed: float
    :param consolidations_collection: the edges a near-duplicate is replaced
        through: the new memory is written, the old one linked to it and
        superseded, never overwritten. Without it, and for a permanent old one,
        the new memory is written beside the old, and ``corrects`` is refused
    :ptype consolidations_collection: MemoryConsolidationsCollection | None
    :param tz: the reader's zone, a ``tzinfo`` or an IANA name; every date is shown in it
    :ptype tz: tzinfo | str | None
    :param person_asked: whether the person asked, this turn, for a permanent
        memory to be changed; asked per permanent ``corrects`` target. ``None``
        means never: a permanent memory is corrected beside, not replaced
    :ptype person_asked: PersonAsked | None
    :param permanent_allowed: whether a memory with this text may be made
        permanent without the person's word; ``None`` allows every one
    :ptype permanent_allowed: PermanentAllowed | None
    :param near_message_ids: the source messages of this turn and the one
        before, read per call; ``None`` turns the same-event rule off
    :ptype near_message_ids: NearMessageIds | None
    :param same_event_threshold: the similarity at which a memory from those
        messages is the same event; ``None`` turns the rule off
    :ptype same_event_threshold: float | None
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    :raises ValueError: when ``tz`` is a name that is not a known zone
    """
    dedup_threshold = similarity_dedup_threshold
    zone = resolve_zone(tz)

    @tool("memory_add", args_schema=MemoryAddInput)
    async def memory_add(
        content: str,
        memory_type: str = "fact",
        alias: str | None = None,
        permanent: bool = False,
        replace: bool = True,
        corrects: list[str] | None = None,
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
        live = _live_context("memory_add", context_resolver)
        if isinstance(live, str):
            return live
        live_conversation_id, live_message_id = live
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
        denied = await _refused(
            ACTION_MEMORY_WRITE,
            "memory_add",
            agent_id=agent_id,
            customer_id=customer_id,
            user_id=user_id,
            authorizer=authorizer,
        )
        if denied:
            return denied

        mt = memory_type.lower().strip()
        if mt not in _VALID_MEMORY_TYPES:
            return f"[TOOL ERROR] Invalid memory_type '{memory_type}'. Valid types: {_TYPES_LISTED}"

        targets: list[dict[str, Any]] = []
        if corrects:
            if consolidations_collection is None:
                return _tool_error(
                    "memory_add", "corrects", "this install keeps no links between memories. Nothing was stored."
                )
            read = await _correction_targets(
                "memory_add", memories_collection, corrects, user_id=user_id, agent_id=agent_id, zone=zone
            )
            if isinstance(read, str):
                return read
            targets = read

        embedding = await _safe_aembed_query(embedding_provider, content)
        if embedding is None:
            return _tool_error("memory_add", "embed", "embedding provider returned None")

        not_kept = ""
        if permanent and not await _may_be_permanent(permanent_allowed, content):
            permanent = False
            not_kept = f" {PERMANENT_NEEDS_THE_PERSON}"

        fields: dict[str, Any] = {
            "customer_id": customer_id,
            "user_id": user_id,
            "conversation_id": live_conversation_id,
            "message_id_source": live_message_id,
            "type_memory": mt,
            "content": content,
            "embedding": embedding,
            "alias": normalised_alias,
            # honor the per-user salience_seed knob (design §3); without it the DB
            # server default (0.5) applies and a tuned seed is a silent no-op.
            "salience": salience_seed,
            "evergreen": permanent,
        }
        alias_clause = f" with alias '{normalised_alias}'" if normalised_alias is not None else ""
        kept = " Kept permanently." if permanent else not_kept

        def _alias_taken(exc: Exception) -> str | None:
            # The partial unique index ``ix_memories_user_alias`` raises a
            # UniqueViolation from asyncpg; this package stays driver-agnostic, so
            # match by name + by the SQLSTATE substring in the message body.
            exc_msg = str(exc)
            if normalised_alias and (
                type(exc).__name__ == "UniqueViolationError"
                or "ix_memories_user_alias" in exc_msg
                or "23505" in exc_msg
            ):
                return _tool_error(
                    "memory_add",
                    "alias",
                    f"Alias '{normalised_alias}' already used. Pick a different name or "
                    f"call memory_search(alias='{normalised_alias}') to find the existing one.",
                )
            return None

        if targets:
            assert consolidations_collection is not None  # refused above without it
            replaced: list[dict[str, Any]] = []
            beside: list[dict[str, Any]] = []
            for row in targets:
                target_id = UUID(str(row["memory_id"]))
                if marks_of(row).permanent and not await _may_touch_permanent(person_asked, target_id):
                    beside.append(row)
                else:
                    replaced.append(row)
            # what replaces a permanent memory, on the person's word, is permanent too
            if any(marks_of(row).permanent for row in replaced):
                fields["evergreen"] = True
                kept = " Kept permanently."
            try:
                if replaced:
                    new_id = await supersede(
                        memories_collection,
                        consolidations_collection,
                        agent_id=agent_id,
                        source_ids=[UUID(str(r["memory_id"])) for r in replaced],
                        fields=fields,
                        rationale="corrected by memory_add",
                        allow_permanent=True,
                    )
                    for row in beside:
                        edge = consolidations_collection.create(
                            {
                                "agent_id": agent_id,
                                "consolidated_memory_id": new_id,
                                "source_memory_id": UUID(str(row["memory_id"])),
                                "rationale": "corrected beside a permanent memory by memory_add",
                                "date_created": datetime.now(UTC),
                                "date_updated": datetime.now(UTC),
                            }
                        )
                        await consolidations_collection.save_entity(edge)
                else:
                    new_id = await correct_beside(
                        memories_collection,
                        consolidations_collection,
                        agent_id=agent_id,
                        source_ids=[UUID(str(r["memory_id"])) for r in beside],
                        fields=fields,
                        rationale="corrected beside a permanent memory by memory_add",
                    )
            except Exception as exc:
                taken = _alias_taken(exc)
                if taken:
                    return taken
                log.warning("memory_add: correction failed", extra={"extra_data": {"error": str(exc)}})
                return _tool_error("memory_add", "store", str(exc))
            said = [f"Stored as [memory:{new_id}]{alias_clause}: {content}.{kept}"]
            for row in replaced:
                when = _fmt_dt(row.get("date_created"), zone)
                written = f", written {when}" if when else ""
                said.append(
                    f"It replaces [memory:{row['memory_id']}]{written}, which said {_quoted(row)} "
                    "and stays readable by its id."
                )
            for row in beside:
                said.append(
                    f"[memory:{row['memory_id']}] is permanent, so it still stands and says {_quoted(row)}. "
                    f"The correction is kept beside it and linked to it. {PERMANENT_IS_THE_PERSONS}"
                )
            return " ".join(said)

        # A near-duplicate: replaced, or left standing and said so. A permanent one,
        # a call that said ``replace=false`` and an install with no edges all leave it.
        standing: tuple[dict[str, Any], float] | None = None
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
                    existing = await memories_collection.find_for_recall(
                        memory_id=existing_id, user_id=user_id, agent_id=agent_id
                    )
                    if existing is None:
                        continue
                    similarity = float(row["similarity"])
                    if is_permanent(existing) or consolidations_collection is None or not replace:
                        standing = (existing, similarity)
                        break
                    new_id = await supersede(
                        memories_collection,
                        consolidations_collection,
                        agent_id=agent_id,
                        source_ids=[existing_id],
                        fields=fields,
                        rationale="replaced by memory_add",
                    )
                    log.info(
                        "memory_add: replaced a near-duplicate",
                        extra={
                            "extra_data": {
                                "memory_id": str(new_id),
                                "replaced": str(existing_id),  # convert at border: log extra_data field
                                "similarity": round(similarity, 3),
                                "old_content": row["content"][:100],
                            }
                        },
                    )
                    when = _fmt_dt(existing.get("date_created"), zone)
                    written = f", written {when}" if when else ""
                    return (
                        f"Stored as [memory:{new_id}]{alias_clause}: {content}. It replaces "
                        f"[memory:{existing_id}] ({similarity:.0%} the same{written}), which said "
                        f"{_quoted(existing)} and stays readable by its id.{kept} "
                        "If they are two facts and not one, store it again with replace=false."
                    )
        except Exception as exc:
            taken = _alias_taken(exc)
            if taken:
                return taken
            log.warning(
                "memory_add: dedup check failed, inserting new",
                extra={"extra_data": {"error": str(exc)}},
            )

        try:
            near: Sequence[UUID] = ()
            if replace and standing is None and near_message_ids is not None and same_event_threshold is not None:
                try:
                    near = await near_message_ids()
                except Exception as exc:  # prawduct:allow prawduct/broad-except -- a host resolver may fail in any way; the memory is still stored
                    log.warning(
                        "memory_add: near_message_ids resolver failed, storing without the same-event check",
                        extra={"extra_data": {"error": str(exc)}},
                    )
            wrote = await write_once(
                memories_collection,
                consolidations_collection,
                agent_id=agent_id,
                fields=fields,
                near_message_ids=near,
                threshold=same_event_threshold,
            )
            assert wrote.memory_id is not None  # "beside" always writes
            memory_id = wrote.memory_id
            if wrote.replaced:
                said = [f"Stored as [memory:{memory_id}]{alias_clause}: {content}."]
                for row in wrote.replaced:
                    when = _fmt_dt(row.get("date_created"), zone)
                    written_at = f", written {when}" if when else ""
                    said.append(
                        f"It replaces [memory:{row['memory_id']}] ({float(row['similarity']):.0%} the same"
                        f"{written_at}), which said {_quoted(row)} and stays readable by its id."
                    )
                if kept:
                    said.append(kept.strip())
                said.append("If they are two facts and not one, store it again with replace=false.")
                return " ".join(said)
            if standing is None and wrote.standing is not None:
                standing = (wrote.standing, float(wrote.standing["similarity"]))

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
            stored = f"Stored as [memory:{memory_id}]{alias_clause}: {content}.{kept}"
            if standing is None:
                return stored
            other, similarity = standing
            if is_permanent(other):
                return (
                    f"{stored} [memory:{other['memory_id']}] (permanent, {similarity:.0%} the same) still stands "
                    f"beside it and says {_quoted(other)}. Both are now in your memory. {PERMANENT_IS_THE_PERSONS}"
                )
            return (
                f"{stored} [memory:{other['memory_id']}] ({similarity:.0%} the same) still stands beside it "
                f"and says {_quoted(other)}."
            )

        except Exception as exc:
            taken = _alias_taken(exc)
            if taken:
                return taken
            log.warning(
                "memory_add: insert failed",
                extra={"extra_data": {"error": str(exc)}},
            )
            return _tool_error("memory_add", "store", str(exc))

    memory_add.description = (
        "Remember something for every later conversation: about the person, about you, "
        "or anything else. Use it often. A memory that is nearly the same as one you "
        "have replaces it, and the result shows what the old one said; the old one stays "
        "readable by its id. To correct a memory, pass its id in corrects. Set permanent for "
        "what must never fade. Returns [memory:<id>]."
    )

    return [memory_add]


class MemoryUpdateInput(BaseModel):
    """Input schema for memory_update."""

    memory_id: str = Field(description="The id from a [memory:<id>] line.")
    content: str | None = Field(
        default=None, description="The new text, in your own words. Leave out to keep the text."
    )
    memory_type: str | None = Field(
        default=None,
        description="A new type: preference, fact, decision, topical_context or relational_context.",
    )
    alias: str | None = Field(
        default=None,
        description="A new name to open it by. 1-64 letters, digits, hyphens or underscores.",
    )

    @model_validator(mode="after")
    def _validate(self) -> "MemoryUpdateInput":
        if self.content is None and self.memory_type is None and self.alias is None:
            raise ValueError("Give content, memory_type or alias: at least one thing to change.")
        return self


async def load_memory_update_tool(
    user_id: UUID,
    embedding_provider: Embeddings,
    agent_id: UUID,
    customer_id: UUID,
    authorizer: MemoryAuthorizerDependencies,
    memories_collection: MemoriesCollection,
    consolidations_collection: MemoryConsolidationsCollection,
    *,
    context_resolver: Callable[[], Any],
    tz: tzinfo | str | None = None,
    person_asked: PersonAsked | None = None,
) -> list[BaseTool]:
    """create memory_update: change a memory's text, type or name by writing its replacement.

    The change goes through :func:`~threetears.agent.memory.revisions.supersede`:
    a new memory is written with the changed fields and everything else carried
    over, the old one is linked to it and marked replaced, and the old one stays
    readable by its id. A name moves to the new memory. Refused, with nothing
    written: a memory that is not the caller's, one already retired, one
    already replaced, and a permanent one unless ``person_asked`` says the
    person asked for the change.

    :param user_id: the caller; a memory of anyone else is refused
    :ptype user_id: UUID
    :param embedding_provider: embeds a changed text
    :ptype embedding_provider: Embeddings
    :param agent_id: the partition
    :ptype agent_id: UUID
    :param customer_id: the tenant
    :ptype customer_id: UUID
    :param authorizer: rbac authorizer dependency bundle
    :ptype authorizer: MemoryAuthorizerDependencies
    :param memories_collection: the memories collection
    :ptype memories_collection: MemoriesCollection
    :param consolidations_collection: the edges a memory is replaced through
    :ptype consolidations_collection: MemoryConsolidationsCollection
    :param context_resolver: the live call context, read per call, as ``memory_add`` reads it
    :ptype context_resolver: Callable[[], Any]
    :param tz: the reader's zone, a ``tzinfo`` or an IANA name; every date is shown in it
    :ptype tz: tzinfo | str | None
    :param person_asked: whether the person asked, this turn, for this
        permanent memory to be changed. ``None`` means never
    :ptype person_asked: PersonAsked | None
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    :raises ValueError: when ``tz`` is a name that is not a known zone
    """
    zone = resolve_zone(tz)

    @tool("memory_update", args_schema=MemoryUpdateInput)
    async def memory_update(
        memory_id: str,
        content: str | None = None,
        memory_type: str | None = None,
        alias: str | None = None,
    ) -> str:
        """change a memory by writing its replacement."""
        target = _parse_memory_id(memory_id)
        if target is None:
            return _tool_error("memory_update", "memory_id", f"'{memory_id}' is not a memory id.")
        new_text = content.strip() if content is not None else None
        if content is not None and not new_text:
            return _tool_error(
                "memory_update", "content", "the new text is empty. To retire a memory use memory_forget."
            )
        new_type = memory_type.lower().strip() if memory_type else None
        if new_type is not None and new_type not in _VALID_MEMORY_TYPES:
            return _tool_error(
                "memory_update", "memory_type", f"Invalid memory_type '{memory_type}'. Valid types: {_TYPES_LISTED}"
            )
        new_alias = alias.strip() if alias else None
        if new_alias is not None and not _ALIAS_RE.match(new_alias):
            return _tool_error(
                "memory_update",
                "alias",
                f"Invalid alias '{alias}'. Use 1-64 letters, digits, hyphens, or underscores.",
            )
        if new_text is None and new_type is None and new_alias is None:
            return _tool_error("memory_update", "arguments", "give content, memory_type or alias to change.")
        live = _live_context("memory_update", context_resolver)
        if isinstance(live, str):
            return live
        live_conversation_id, live_message_id = live
        denied = await _refused(
            ACTION_MEMORY_WRITE,
            "memory_update",
            agent_id=agent_id,
            customer_id=customer_id,
            user_id=user_id,
            authorizer=authorizer,
        )
        if denied:
            return denied

        entity = await memories_collection.get((agent_id, target))
        row = await memories_collection.find_for_recall(memory_id=target, user_id=user_id, agent_id=agent_id)
        if entity is None or row is None or entity.user_id != user_id:
            return _no_change("memory_update", f"No memory [memory:{target}] that is yours. Nothing was changed.")
        marks = marks_of(row)
        if marks.retired:
            return _no_change(
                "memory_update",
                f"[memory:{target}] was retired and no longer holds, so it was not changed. "
                "memory_restore brings it back first.",
            )
        if marks.replaced_by is not None:
            return _no_change(
                "memory_update",
                f"[memory:{target}] was already replaced by [memory:{marks.replaced_by}]. "
                "Nothing was changed. Update that one instead.",
            )
        allowed = False
        if marks.permanent:
            allowed = await _may_touch_permanent(person_asked, target)
            if not allowed:
                return _no_change(
                    "memory_update",
                    f"[memory:{target}] is permanent, so it was not changed. It says {_quoted(row)}. "
                    f"{PERMANENT_IS_THE_PERSONS} To record that it is wrong, call memory_add with "
                    f"corrects=['{target}']: the correction is kept beside it and linked.",
                )

        old_text, old_type, old_alias = str(row["content"]), str(row["type_memory"]), marks.alias
        text = new_text if new_text is not None else old_text
        kind = new_type if new_type is not None else old_type
        name = new_alias if new_alias is not None else old_alias
        if (text, kind, name) == (old_text, old_type, old_alias):
            return _no_change("memory_update", f"[memory:{target}] already says that. Nothing was changed.")

        embedding: list[float] | None = (
            entity.embedding if text == old_text else await _safe_aembed_query(embedding_provider, text)
        )
        if not embedding:
            return _tool_error("memory_update", "embed", "embedding provider returned None. Nothing was changed.")

        # A name is unique per person, so it comes off the old memory before the
        # new one takes it, and goes back if the replacement is not written.
        try:
            if old_alias is not None:
                entity.alias = None
                await memories_collection.save_entity(entity)
            try:
                new_id = await supersede(
                    memories_collection,
                    consolidations_collection,
                    agent_id=agent_id,
                    source_ids=[target],
                    fields={
                        "customer_id": customer_id,
                        "user_id": user_id,
                        "conversation_id": live_conversation_id,
                        "message_id_source": live_message_id,
                        "type_memory": kind,
                        "content": text,
                        "embedding": embedding,
                        "alias": name,
                        "salience": entity.salience,
                        "evergreen": marks.permanent,
                    },
                    rationale="changed by memory_update",
                    allow_permanent=allowed,
                )
            except Exception:
                if old_alias is not None:
                    fresh = await memories_collection.get((agent_id, target))
                    if fresh is not None:
                        fresh.alias = old_alias
                        await memories_collection.save_entity(fresh)
                raise
        except Exception as exc:
            exc_msg = str(exc)
            if name and (
                type(exc).__name__ == "UniqueViolationError"
                or "ix_memories_user_alias" in exc_msg
                or "23505" in exc_msg
            ):
                return _tool_error(
                    "memory_update",
                    "alias",
                    f"Alias '{name}' already used by another memory. Nothing was changed. Pick a different name.",
                )
            log.warning("memory_update: replace failed", extra={"extra_data": {"error": exc_msg}})
            return _tool_error("memory_update", "store", f"{exc_msg}. Nothing was changed.")

        changes: list[str] = []
        if text != old_text:
            changes.append(f"It said {_quoted(row)} and now says {_quoted({'memory_id': new_id, 'content': text})}.")
        if kind != old_type:
            changes.append(f"Its type was {old_type} and is now {kind}.")
        if name != old_alias:
            changes.append(
                f"Its name was '{old_alias}' and is now '{name}'." if old_alias else f"It is now named '{name}'."
            )
        elif name:
            changes.append(f"The name '{name}' now opens the new one.")
        when = _fmt_dt(row.get("date_created"), zone)
        written = f", written {when}," if when else ""
        still = " It is still permanent." if marks.permanent else ""
        return (
            f"Updated: [memory:{new_id}] replaces [memory:{target}]. {' '.join(changes)} "
            f"The old one{written} stays readable by its id.{still}"
        )

    memory_update.description = (
        "Change a memory you have: its text, its type or its name. It writes a replacement and "
        "keeps the old one readable by its id, and the result shows both. A permanent memory is "
        "not changed here: only your person can change one."
    )

    return [memory_update]


class MemoryForgetInput(BaseModel):
    """Input schema for memory_forget."""

    memory_id: str = Field(description="The id from a [memory:<id>] line.")
    reason: str = Field(description="Why it no longer holds, in a sentence. It is kept with the memory.")


class MemoryRestoreInput(BaseModel):
    """Input schema for memory_restore."""

    memory_id: str = Field(description="The id of a retired memory, from its [memory:<id>] line.")


def load_memory_forget_tool(
    user_id: UUID,
    agent_id: UUID,
    customer_id: UUID,
    authorizer: MemoryAuthorizerDependencies,
    memories_collection: MemoriesCollection,
    *,
    tz: tzinfo | str | None = None,
    person_asked: PersonAsked | None = None,
) -> list[BaseTool]:
    """create memory_forget: retire a memory, keeping its row, with the reason.

    Retiring is :func:`~threetears.agent.memory.revisions.retract`: the memory
    leaves search and ambient recall, stays readable by its id, and
    ``memory_restore`` brings it back. Nothing is deleted. Refused, with nothing
    changed: no reason, a memory that is not the caller's, one already retired,
    one already replaced, and a permanent one unless ``person_asked`` says the
    person asked.

    :param user_id: the caller; a memory of anyone else is refused
    :ptype user_id: UUID
    :param agent_id: the partition
    :ptype agent_id: UUID
    :param customer_id: the tenant
    :ptype customer_id: UUID
    :param authorizer: rbac authorizer dependency bundle
    :ptype authorizer: MemoryAuthorizerDependencies
    :param memories_collection: the memories collection
    :ptype memories_collection: MemoriesCollection
    :param tz: the reader's zone, a ``tzinfo`` or an IANA name; every date is shown in it
    :ptype tz: tzinfo | str | None
    :param person_asked: whether the person asked, this turn, for this
        permanent memory to be retired. ``None`` means never
    :ptype person_asked: PersonAsked | None
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    :raises ValueError: when ``tz`` is a name that is not a known zone
    """
    zone = resolve_zone(tz)

    @tool("memory_forget", args_schema=MemoryForgetInput)
    async def memory_forget(memory_id: str, reason: str) -> str:
        """retire a memory, keeping it."""
        target = _parse_memory_id(memory_id)
        if target is None:
            return _tool_error("memory_forget", "memory_id", f"'{memory_id}' is not a memory id.")
        why = " ".join(reason.split())
        if not why:
            return _tool_error(
                "memory_forget", "reason", "say why it no longer holds. The reason is kept with the memory."
            )
        denied = await _refused(
            ACTION_MEMORY_WRITE,
            "memory_forget",
            agent_id=agent_id,
            customer_id=customer_id,
            user_id=user_id,
            authorizer=authorizer,
        )
        if denied:
            return denied
        row = await memories_collection.find_for_recall(memory_id=target, user_id=user_id, agent_id=agent_id)
        if row is None:
            return _no_change("memory_forget", f"No memory [memory:{target}] that is yours. Nothing was retired.")
        marks = marks_of(row)
        if marks.retired:
            because = f" ({marks.retired_because})" if marks.retired_because else ""
            return _no_change("memory_forget", f"[memory:{target}] was already retired{because}.")
        if marks.replaced_by is not None:
            when = _fmt_dt(marks.replaced_at, zone)
            return _no_change(
                "memory_forget",
                f"[memory:{target}] was already replaced by [memory:{marks.replaced_by}]"
                f"{' on ' + when if when else ''}, so it is no longer current. Nothing was retired. "
                "If the replacement is wrong too, retire that one.",
            )
        allowed = False
        if marks.permanent:
            allowed = await _may_touch_permanent(person_asked, target)
            if not allowed:
                return _no_change(
                    "memory_forget",
                    f"[memory:{target}] is permanent, so it was not retired. It says {_quoted(row)}. "
                    f"{PERMANENT_IS_THE_PERSONS} To record that it is wrong, call memory_add with "
                    f"corrects=['{target}']: the correction is kept beside it and linked.",
                )
        try:
            done = await retract(
                memories_collection, agent_id=agent_id, memory_id=target, reason=why, allow_permanent=allowed
            )
        except Exception as exc:
            log.warning("memory_forget: retract failed", extra={"extra_data": {"error": str(exc)}})
            return _tool_error("memory_forget", "store", f"{exc}. Nothing was retired.")
        if not done:
            return _no_change(
                "memory_forget", f"[memory:{target}] changed while this ran and was not retired. Read it again."
            )
        return (
            f"Retired [memory:{target}], which said {_quoted(row)}. Reason kept: {why[:200]}. "
            "It is out of search and of what you are handed, and it is not deleted: memory_recall still "
            "opens it by its id, and memory_restore brings it back."
        )

    memory_forget.description = (
        "Retire a memory that no longer holds: it leaves search and what you are handed, and is "
        "kept. Give the reason. memory_restore undoes it. To put something right use memory_update "
        "or memory_add with corrects; a permanent memory is retired only by your person."
    )

    return [memory_forget]


def load_memory_restore_tool(
    user_id: UUID,
    agent_id: UUID,
    customer_id: UUID,
    authorizer: MemoryAuthorizerDependencies,
    memories_collection: MemoriesCollection,
    *,
    salience_seed: float = 0.5,
) -> list[BaseTool]:
    """create memory_restore: bring a retired memory back, the undo of ``memory_forget``.

    :param user_id: the caller; a memory of anyone else is refused
    :ptype user_id: UUID
    :param agent_id: the partition
    :ptype agent_id: UUID
    :param customer_id: the tenant
    :ptype customer_id: UUID
    :param authorizer: rbac authorizer dependency bundle
    :ptype authorizer: MemoryAuthorizerDependencies
    :param memories_collection: the memories collection
    :ptype memories_collection: MemoriesCollection
    :param salience_seed: the salience given to a memory retired before its own was recorded
    :ptype salience_seed: float
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """

    @tool("memory_restore", args_schema=MemoryRestoreInput)
    async def memory_restore(memory_id: str) -> str:
        """bring a retired memory back."""
        target = _parse_memory_id(memory_id)
        if target is None:
            return _tool_error("memory_restore", "memory_id", f"'{memory_id}' is not a memory id.")
        denied = await _refused(
            ACTION_MEMORY_WRITE,
            "memory_restore",
            agent_id=agent_id,
            customer_id=customer_id,
            user_id=user_id,
            authorizer=authorizer,
        )
        if denied:
            return denied
        row = await memories_collection.find_for_recall(memory_id=target, user_id=user_id, agent_id=agent_id)
        if row is None:
            return _no_change("memory_restore", f"No memory [memory:{target}] that is yours. Nothing was restored.")
        marks = marks_of(row)
        if not marks.retired:
            return _no_change(
                "memory_restore",
                f"[memory:{target}] is not retired, so there is nothing to restore. "
                "memory_list with show='retired' lists the ones that are.",
            )
        try:
            done = await restore(memories_collection, agent_id=agent_id, memory_id=target, salience=salience_seed)
        except Exception as exc:
            log.warning("memory_restore: restore failed", extra={"extra_data": {"error": str(exc)}})
            return _tool_error("memory_restore", "store", f"{exc}. Nothing was restored.")
        if not done:
            return _no_change(
                "memory_restore", f"[memory:{target}] changed while this ran and was not restored. Read it again."
            )
        was = f" It had been retired because: {marks.retired_because}." if marks.retired_because else ""
        also = (
            f" It is still replaced by [memory:{marks.replaced_by}], which is the current one."
            if marks.replaced_by is not None
            else " Search finds it again."
        )
        return f"Restored [memory:{target}], which says {_quoted(row)}.{was}{also}"

    memory_restore.description = (
        "Bring back a memory that was retired with memory_forget: search finds it again. "
        "memory_list with show='retired' lists the retired ones and why each was retired."
    )

    return [memory_restore]


class MemoryKeepInput(BaseModel):
    """Input schema for memory_keep."""

    memory_id: str = Field(description="The id from a [memory:<id>] line.")


def load_memory_keep_tool(
    user_id: UUID,
    agent_id: UUID,
    customer_id: UUID,
    authorizer: MemoryAuthorizerDependencies,
    memories_collection: MemoriesCollection,
    *,
    permanent_allowed: PermanentAllowed | None = None,
) -> list[BaseTool]:
    """create memory_keep: pin a stored memory permanent.

    A permanent (``evergreen``) memory never decays, is never merged by dream,
    and is never replaced by a later conversation: a change to it is a new
    memory beside it. The pin is one way; nothing here unpins.

    With ``permanent_allowed`` a memory the resolver refuses is left as it is,
    and the result says making it permanent needs the person's word.

    :param user_id: the caller; a memory of anyone else is refused
    :ptype user_id: UUID
    :param agent_id: the partition
    :ptype agent_id: UUID
    :param customer_id: the tenant
    :ptype customer_id: UUID
    :param authorizer: rbac authorizer dependency bundle
    :ptype authorizer: MemoryAuthorizerDependencies
    :param memories_collection: the memories collection
    :ptype memories_collection: MemoriesCollection
    :param permanent_allowed: whether a memory with this text may be made
        permanent without the person's word; ``None`` allows every one
    :ptype permanent_allowed: PermanentAllowed | None
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """

    @tool("memory_keep", args_schema=MemoryKeepInput)
    async def memory_keep(memory_id: str) -> str:
        """pin a memory permanent."""
        memory_uuid = _parse_memory_id(memory_id)
        if memory_uuid is None:
            return _tool_error("memory_keep", "memory_id", f"'{memory_id}' is not a memory id.")
        denied = await _refused(
            ACTION_MEMORY_WRITE,
            "memory_keep",
            agent_id=agent_id,
            customer_id=customer_id,
            user_id=user_id,
            authorizer=authorizer,
        )
        if denied:
            return denied
        entity = await memories_collection.get((agent_id, memory_uuid))
        if entity is None or entity.user_id != user_id:
            return _tool_error("memory_keep", "memory_id", f"No memory [memory:{memory_uuid}].")
        if entity.evergreen:
            return f"[memory:{memory_uuid}] was already kept permanently."
        if not await _may_be_permanent(permanent_allowed, entity.content or ""):
            return _no_change("memory_keep", f"[memory:{memory_uuid}] is as it was. {PERMANENT_NEEDS_THE_PERSON}")
        entity.evergreen = True
        await memories_collection.save_entity(entity)
        return f"[memory:{memory_uuid}] is kept permanently: {entity.content}"

    memory_keep.description = (
        "Keep a memory permanently: it never fades, is never merged away, and a later "
        "change is written beside it. For who someone is, a promise, anything it would "
        "be wrong to forget. It cannot be taken back here: only your person can change, "
        "retire or un-keep a permanent memory."
    )

    return [memory_keep]


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

    chunk_id: str = Field(description="The id from a [chunk:<id>] line.")


class ChunkSearchInput(BaseModel):
    """Input schema for the ``chunk_search`` tool.

    Cross-memory chunk hybrid search. Distinct from ``memory_search``
    (which searches at the memory level); this tool reaches into the
    chunks belonging to every memory the user owns and returns the
    most relevant chunks regardless of which memory they belong to.
    """

    query: str = Field(description="What to look for.")
    limit: int = Field(default=5, ge=1, le=20, description="Most passages to return, 1-20.")
    mode: str = Field(
        default="balanced",
        description="precise matches exact words, fuzzy matches loosely, balanced (default) is between.",
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
    *,
    tz: tzinfo | str | None = None,
) -> list[BaseTool]:
    """create a ``chunk_recall`` tool that returns a chunk + parent memory.

    auth: the chunk's parent memory's user_id must match ``user_id``.
    enforced at the SQL level by ``MemoryChunkCollection.find_for_recall``
    which scopes by ``(agent_id, user_id)``. an unauthorized chunk
    surfaces as ``None`` → tool returns a graceful "not found" message
    rather than leaking existence.

    the return shape is a formatted text block: one header line -- the
    passage's id, which part of how many it is, the memory it belongs to, when
    it was written and what it came from -- a blank line, then the passage
    whole. The part number is what ``memory_recall(chunk_indexes=[...])`` takes,
    so its neighbours are one call away.

    :param user_id: caller's user UUID
    :ptype user_id: UUID
    :param agent_id: agent UUID for rbac namespace lookup
    :ptype agent_id: UUID
    :param customer_id: customer UUID for rbac namespace lookup
    :ptype customer_id: UUID
    :param authorizer: rbac authorizer dependency bundle
    :ptype authorizer: MemoryAuthorizerDependencies
    :param memories_collection: three-tier memories collection (kept for
        signature parity; the passage's row names its memory)
    :ptype memories_collection: MemoriesCollection
    :param memory_chunk_collection: three-tier memory_chunks collection
    :ptype memory_chunk_collection: MemoryChunkCollection
    :param tz: the reader's zone, a ``tzinfo`` or an IANA name; every date is shown in it
    :ptype tz: tzinfo | str | None
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    :raises ValueError: when ``tz`` is a name that is not a known zone
    """
    _ = memories_collection
    zone = resolve_zone(tz)

    @tool("chunk_recall", args_schema=ChunkRecallInput)
    async def chunk_recall(chunk_id: str) -> str:
        """retrieve the full content of one chunk plus its parent memory id."""
        denied = await _refused(
            ACTION_MEMORY_READ,
            "chunk_recall",
            agent_id=agent_id,
            customer_id=customer_id,
            user_id=user_id,
            authorizer=authorizer,
        )
        if denied:
            return denied

        try:
            cid = UUID(chunk_id.strip().removeprefix("[chunk:").removesuffix("]").strip())
        except ValueError:
            return f"[TOOL ERROR] Invalid chunk_id format: {chunk_id}"

        row = await memory_chunk_collection.find_for_recall(
            chunk_id=cid,
            user_id=user_id,
            agent_id=agent_id,
        )
        if row is None:
            return "[TOOL ERROR] No passage with that id that you can read. Use an id from a [chunk:<id>] line you were shown."
        header = f"[chunk:{cid}] part {row['chunk_index']} of {row['total']} of [memory:{row['memory_id']!s}]"
        when = _fmt_dt(row.get("date_created"), zone)
        if when:
            header += f", written {when}"
        source = _passage_source(
            title=_extract_title(row.get("metadata_json")),
            page=row.get("page_number"),
            heading=row.get("heading_context"),
        )
        if source:
            header += f", from {source}"
        return f"{header}.\n\n{row['content']}"

    chunk_recall.description = (
        "Read the exact words of one passage by its [chunk:<id>], whole. Also says which part it is "
        "of how many, when it was written, and the [memory:<id>] it belongs to."
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
    *,
    tz: tzinfo | str | None = None,
) -> list[BaseTool]:
    """create a ``chunk_search`` tool: cross-memory chunk hybrid search.

    delegates to :meth:`MemoryChunkCollection.hybrid_search` which is
    the same path the memory-retrieval node uses for grounding.
    optionally records each surfaced chunk via ``ledger_callback`` so
    the host application can dedup chunks already shown in the
    current conversation (the ledger pattern).

    each hit is the passage's id, the memory it belongs to, what it came from
    and when it was written, then its first characters; a passage longer than
    that is marked as cut, with its length and the call that reads it whole, so
    a preview is never taken for the exact words.

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
    :param similarity_threshold: floor on semantic similarity for admission
    :ptype similarity_threshold: float
    :param tz: the reader's zone, a ``tzinfo`` or an IANA name; every date is shown in it
    :ptype tz: tzinfo | str | None
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    :raises ValueError: when ``tz`` is a name that is not a known zone
    """
    zone = resolve_zone(tz)

    @tool("chunk_search", args_schema=ChunkSearchInput)
    async def chunk_search(query: str, limit: int = 5, mode: str = "balanced") -> str:
        """search across every chunk the user owns by relevance to query."""
        denied = await _refused(
            ACTION_MEMORY_READ,
            "chunk_search",
            agent_id=agent_id,
            customer_id=customer_id,
            user_id=user_id,
            authorizer=authorizer,
        )
        if denied:
            return denied

        if not query or not query.strip():
            return "Empty query. Provide a search term."

        embedding = await _safe_aembed_query(embedding_provider, query)
        if embedding is None:
            return "Passage search is down; try memory_search."

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
            return "No passages matched."

        more = len(results) > limit
        results = results[:limit]
        # a ranked hit carries neither its date nor its place; both are on its row
        try:
            placed = await memory_chunk_collection.search_by_ids(
                [UUID(str(r["chunk_id"])) for r in results], user_id, agent_id=agent_id
            )
        except (
            Exception
        ):  # prawduct:allow prawduct/broad-except -- the date and place are extra facts; the hits are still shown
            log.exception("chunk_search: reading the hits' dates failed")
            placed = []
        row_of = {str(r["chunk_id"]): r for r in placed}

        lines: list[str] = []
        for r in results:
            chunk_id = str(r.get("chunk_id", ""))
            memory_id = r.get("memory_id")
            # convert at border: agent-facing tool-result display text
            memory_id_str = str(memory_id) if memory_id is not None else "?"
            content = str(r.get("content", "") or "")
            preview = _cut(content, _PASSAGE_PREVIEW_CHARS, f"chunk_recall reads [chunk:{chunk_id}] whole")
            facts = [f"from [memory:{memory_id_str}]"]
            source = _passage_source(title=r.get("title"), page=r.get("page_number"), heading=r.get("heading_context"))
            if source:
                facts.append(source)
            when = _fmt_dt(row_of.get(chunk_id, {}).get("date_created"), zone)
            if when:
                facts.append(f"written {when}")
            lines.append(f"[chunk:{chunk_id}] ({', '.join(facts)})\n{preview}")
            if ledger_callback is not None:
                try:
                    await ledger_callback(chunk_id, "chunk", content[:80])
                except Exception as cb_exc:
                    log.debug(
                        "chunk_search ledger_callback failed for chunk %s: %s",
                        chunk_id,
                        cb_exc,
                    )

        body = "\n\n".join(lines)
        if more:
            hint = "raise limit (up to 20), or narrow the query" if limit < 20 else "narrow the query"
            return f"(More passages match than the {len(results)} shown: {hint}.)\n\n{body}"
        return body

    chunk_search.description = (
        "Search the exact words of your memories, files and past conversations. Use it "
        "for a specific line or detail, not a topic. Returns [chunk:<id>] lines, each with "
        "where it is from and when it was written, and the start of the passage; "
        "chunk_recall reads one whole."
    )

    return [chunk_search]
