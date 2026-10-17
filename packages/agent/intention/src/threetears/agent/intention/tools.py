"""Intention tools -- private LangChain tools for the deliberation loop.

Three verbs, mirroring luminara's intention surface, all **private**
(absent from any outward-facing tool set -- the LLM uses them only on the
agent-internal deliberation path):

- ``intention_log`` -- embed a want, dedup against the user's open wants,
  create an ``open`` intention (or refresh the near-duplicate).
- ``intention_list`` -- the deliberation candidate set: ``open`` wants
  outside the surfacing cooldown, salience-ranked.
- ``intention_mark_surfaced`` -- move a want forward (``open -> asked`` /
  ``asked -> granted`` / ``asked -> dropped``), stamping the cooldown
  clock and emitting the presence-timeline event.

Every factory binds ``user_id`` at load time (the tool is minted per-user)
and never exposes it in the Pydantic schema: user isolation is the
``user_id`` WHERE clause on the collection reads + an ownership check on
``mark_surfaced``, NOT RBAC (RBAC is the agent-owner short-circuit alone).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta, tzinfo
from typing import Any
from uuid import UUID

from uuid_utils import uuid7

from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from threetears.observe.text_block import under
from threetears.core.utils.local_time import render_local_time, resolve_zone
from threetears.langgraph.events import FrameworkEvent, dispatch_event
from threetears.observe import contain, get_logger

from threetears.agent.intention.authorize import (
    ACTION_INTENTION_READ,
    ACTION_INTENTION_WRITE,
    IntentionAccessDenied,
    IntentionAuthorizerDependencies,
    authorize_intention_access,
)
from threetears.agent.intention.collections import IntentionsCollection
from threetears.agent.intention.entities import IntentionEntity
from threetears.agent.intention.events import (
    IntentionResolvedEvent,
    IntentionSurfacedEvent,
)
from threetears.agent.intention.types import INTENTION_STATUS_VALUES, IntentionStatus

__all__ = [
    "IntentionGetInput",
    "IntentionListInput",
    "IntentionLogInput",
    "IntentionMarkSurfacedInput",
    "WantNotes",
    "load_intention_get_tool",
    "load_intention_list_tool",
    "load_intention_log_tool",
    "load_intention_mark_surfaced_tool",
]

#: A consumer's own words about some wants, by id: what a want carries in that
#: product (a project, say). Asked once per listing or read, with the wants shown.
WantNotes = Callable[[Sequence[IntentionEntity]], Awaitable[Mapping[UUID, str]]]

log = get_logger(__name__)

# luminara-grounded defaults, exposed as factory params so metallm can
# resolve per-user config values and pass them in (design §3 knobs).
_DEFAULT_COOLDOWN_DAYS = 7
_DEFAULT_DEDUP_THRESHOLD = 0.90
_DEFAULT_SALIENCE_SEED = 0.5
_DEFAULT_NEAR_DUP_BUMP = 0.05
_DEFAULT_MAX_RESULTS = 20
#: The most wants one listing shows, whatever it is asked for.
_MAX_LIMIT = 100

#: What ``intention_list`` may be asked to show: one status, or every want.
_LIST_STATUSES = (*INTENTION_STATUS_VALUES, "all")

# max chars carried in an event payload's content preview (the producer
# truncates here; consumers needing the full want text re-fetch by id).
_CONTENT_PREVIEW_LEN = 120

# statuses ``intention_mark_surfaced`` may move a want to. ``open`` is
# excluded -- the verb walks a want forward, never back to the start.
_MARK_SURFACED_STATUSES = tuple(s for s in INTENTION_STATUS_VALUES if s != IntentionStatus.OPEN.value)


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


async def _safe_aembed_query(embedder: Embeddings, text: str) -> list[float] | None:
    """embed ``text`` with soft-fail semantics (``None`` on any failure).

    Mirrors ``agent-memory``'s private shim so a transient embedding
    outage degrades the tool to a graceful error rather than a crash;
    kept local so ``agent-intention`` carries no dependency on the
    memory package.

    :param embedder: LangChain embedding model
    :ptype embedder: Embeddings
    :param text: text to embed
    :ptype text: str
    :return: embedding vector, or ``None`` on empty input / failure
    :rtype: list[float] | None
    """
    if not text:
        return None
    try:
        return await embedder.aembed_query(text)
    except Exception as exc:  # prawduct:allow prawduct/broad-except -- soft-fail embedding boundary; upstream API errors surface as None
        contain(exc, log=log, what="intention embedding failed; degrading to soft-fail", extra={"error": str(exc)})
        return None


async def _emit_intention_event(event: FrameworkEvent) -> None:
    """dispatch an intention event, tolerating calls outside a langgraph run.

    ``adispatch_custom_event`` raises ``RuntimeError`` when no run manager
    is in scope (cli / background job / test harness). The intention row
    is already committed by the time this fires, so the stream event is a
    best-effort surface: log at debug and swallow only ``RuntimeError``.
    Any other failure (e.g. a pydantic regression) still propagates.

    :param event: the intention event to dispatch
    :ptype event: FrameworkEvent
    :return: nothing
    :rtype: None
    """
    try:
        await dispatch_event(event)
    except RuntimeError as exc:
        log.debug(
            "intention event dropped: no run manager in scope (type=%s, reason=%s)",
            type(event).model_fields["type"].default,
            exc,
        )


class IntentionLogInput(BaseModel):
    """Input schema for the ``intention_log`` tool."""

    content: str = Field(description="What you want, in your own words, in one sentence.")
    source_memory_id: str | None = Field(
        default=None,
        description="Optional [memory:<id>] that this want came from.",
    )


class IntentionListInput(BaseModel):
    """Input schema for the ``intention_list`` tool."""

    limit: int = Field(
        default=_DEFAULT_MAX_RESULTS,
        ge=1,
        le=_MAX_LIMIT,
        description="Most wants to list, 1-100.",
    )
    status: str = Field(
        default=IntentionStatus.OPEN.value,
        description="Which wants: open (the default), asked, granted, dropped, or all.",
    )


class IntentionGetInput(BaseModel):
    """Input schema for the ``intention_get`` tool."""

    intention_id: str = Field(description="The id from an [intention:<id>] mark.")


class IntentionMarkSurfacedInput(BaseModel):
    """Input schema for the ``intention_mark_surfaced`` tool."""

    intention_id: str = Field(description="The id from an [intention:<id>] line.")
    new_status: str = Field(
        description="asked (you asked the person), granted, or dropped.",
    )


async def load_intention_log_tool(
    user_id: UUID,
    embedding_provider: Embeddings,
    agent_id: UUID,
    customer_id: UUID,
    authorizer: IntentionAuthorizerDependencies,
    intentions_collection: IntentionsCollection,
    *,
    context_resolver: Callable[[], Any] | None = None,
    similarity_dedup_threshold: float = _DEFAULT_DEDUP_THRESHOLD,
    salience_seed: float = _DEFAULT_SALIENCE_SEED,
    near_dup_bump: float = _DEFAULT_NEAR_DUP_BUMP,
    on_logged: Callable[[IntentionEntity], Awaitable[None]] | None = None,
) -> list[BaseTool]:
    """create an ``intention_log`` tool bound to a user + the collection.

    Embeds ``content``, dedups against the user's OPEN wants, and either
    refreshes the near-duplicate (salience bump + cooldown reset so the
    reinforced want re-enters deliberation) or creates a fresh ``open``
    intention. ``user_id`` is bound here, never an LLM field.

    :param user_id: owning user whose want to log (isolation boundary)
    :ptype user_id: UUID
    :param embedding_provider: embedding provider for the dedup vector
    :ptype embedding_provider: Embeddings
    :param agent_id: owning agent UUID (intention namespace owner + partition)
    :ptype agent_id: UUID
    :param customer_id: owning customer UUID (namespace scope grain)
    :ptype customer_id: UUID
    :param authorizer: intention authorizer dependency bundle
    :ptype authorizer: IntentionAuthorizerDependencies
    :param intentions_collection: three-tier intentions collection
    :ptype intentions_collection: IntentionsCollection
    :param context_resolver: optional callable returning the live call
        context; when present, ``conversation_id`` stamps the want's
        ``source_conversation_id`` (timeline link into chat)
    :ptype context_resolver: Callable[[], Any] | None
    :param similarity_dedup_threshold: cosine similarity at/above which an
        existing open want is refreshed instead of duplicated
    :ptype similarity_dedup_threshold: float
    :param salience_seed: starting salience for a fresh want
    :ptype salience_seed: float
    :param near_dup_bump: salience increment applied on a dedup refresh
    :ptype near_dup_bump: float
    :param on_logged: told of each NEW want once it is stored, so a consumer can
        record it (a timeline entry); a refreshed want is the same want and is
        not told again. Its failure is logged and the want stays logged.
    :ptype on_logged: Callable[[IntentionEntity], Awaitable[None]] | None
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """

    @tool("intention_log", args_schema=IntentionLogInput)
    async def intention_log(content: str, source_memory_id: str | None = None) -> str:
        """record a standing want you hold, for the deliberation loop."""
        try:
            await authorize_intention_access(
                action=ACTION_INTENTION_WRITE,
                agent_id=agent_id,
                customer_id=customer_id,
                caller_agent_id=agent_id,
                deps=authorizer,
            )
        except IntentionAccessDenied as exc:
            return _tool_error("intention_log", "authorize", str(exc))

        text = content.strip()
        if not text:
            return _tool_error("intention_log", "input", "content is empty")

        source_memory_uuid: UUID | None = None
        if source_memory_id:
            try:
                source_memory_uuid = UUID(source_memory_id)
            except ValueError:
                return _tool_error(
                    "intention_log",
                    "input",
                    f"invalid source_memory_id '{source_memory_id}'",
                )

        embedding = await _safe_aembed_query(embedding_provider, text)
        if embedding is None:
            return _tool_error("intention_log", "embed", "embedding provider returned None")

        # dedup against the user's OPEN wants: refresh the near-duplicate
        # instead of stacking a second row for the same standing want.
        try:
            similar = await intentions_collection.find_similar_for_dedup(
                user_id=user_id,
                agent_id=agent_id,
                embedding=embedding,
                top_k=3,
                threshold=0.0,
            )
        except Exception as exc:
            contain(
                exc, log=log, what="intention_log: dedup lookup failed, creating new want", extra={"error": str(exc)}
            )
            similar = []

        for row in similar:
            if float(row["similarity"]) >= similarity_dedup_threshold:
                existing_id = row["intention_id"]
                existing = await intentions_collection.get((agent_id, existing_id))
                if existing is None:
                    continue
                # reinforcement: clear the cooldown anchor via the entity save,
                # then bump salience via the raw pass. The stored text and its
                # embedding stay: the want kept is the one written down first,
                # and an embedding of words that were not saved would no longer
                # be the embedding of the text. salience is immutable to the
                # entity-UPDATE generator (like memory), so the save can't carry
                # a stale salience back and revert a concurrent decay; the raw
                # bump_salience is the only salience writer.
                existing.last_surfaced_at = None
                try:
                    await intentions_collection.save_entity(existing)
                    await intentions_collection.bump_salience(
                        [existing_id],
                        agent_id=agent_id,
                        access_bump=near_dup_bump,
                    )
                except Exception as exc:
                    # preserve the tool's soft-fail-to-string contract: a CAS
                    # conflict (a concurrent re-log advancing date_updated
                    # between this get and save) or a transient L3 error must
                    # not raise out of the tool. The create + mark_surfaced
                    # paths catch the same way.
                    contain(
                        exc,
                        log=log,
                        what="intention_log: refresh save failed",
                        extra={"intention_id": str(existing_id), "error": str(exc)},
                    )
                    return _tool_error("intention_log", "refresh", str(exc))
                log.info(
                    "intention_log: refreshed near-duplicate open want",
                    extra={
                        "extra_data": {
                            "intention_id": str(existing_id),
                            "similarity": round(float(row["similarity"]), 3),
                        }
                    },
                )
                return (
                    f"Refreshed existing want [intention:{existing_id}] "
                    f"(it was {float(row['similarity']):.0%} the same), so nothing new was written down. "
                    f"It is now stronger, and it stays as first written: {existing.content}"
                )

        source_conversation_id: UUID | None = None
        if context_resolver is not None:
            try:
                live_ctx = context_resolver()
                raw_conv = getattr(live_ctx, "conversation_id", None)
                if raw_conv is not None:
                    source_conversation_id = raw_conv if isinstance(raw_conv, UUID) else UUID(str(raw_conv))
            except Exception as exc:
                contain(
                    exc,
                    log=log,
                    what=f"intention_log: context_resolver unavailable; source_conversation_id left null ({exc})",
                )

        try:
            now = datetime.now(UTC)
            intention_id = UUID(str(uuid7()))
            new_data: dict[str, Any] = {
                "intention_id": intention_id,
                "agent_id": agent_id,
                "customer_id": customer_id,
                "user_id": user_id,
                "status": IntentionStatus.OPEN.value,
                "content": text,
                "embedding": embedding,
                "salience": salience_seed,
                "source_memory_id": source_memory_uuid,
                "source_conversation_id": source_conversation_id,
                "date_created": now,
                "date_updated": now,
            }
            entity: IntentionEntity = intentions_collection.create(new_data)
            await intentions_collection.save_entity(entity)
        except Exception as exc:
            contain(exc, log=log, what="intention_log: create failed", extra={"error": str(exc)})
            return _tool_error("intention_log", "store", str(exc))

        log.info(
            "intention_log: stored new open want",
            extra={"extra_data": {"intention_id": str(intention_id), "content": text[:100]}},
        )
        if on_logged is not None:
            try:
                await on_logged(entity)
            except Exception as exc:  # the want is stored; a record of it failing must not unlog it
                contain(
                    exc,
                    log=log,
                    what="intention_log: the want is stored but on_logged failed",
                    extra={"intention_id": str(intention_id), "error": str(exc)},
                    level="error",
                )
        return f"Logged as [intention:{intention_id}]: {text}"

    intention_log.description = (
        "Write down something you want to do, ask or come back to, in any later "
        "conversation. Writing down a want you already have makes it stronger instead "
        "of adding it twice. Returns [intention:<id>]."
    )
    return [intention_log]


async def load_intention_list_tool(
    user_id: UUID,
    agent_id: UUID,
    customer_id: UUID,
    authorizer: IntentionAuthorizerDependencies,
    intentions_collection: IntentionsCollection,
    *,
    cooldown_days: float = _DEFAULT_COOLDOWN_DAYS,
    max_results: int = _DEFAULT_MAX_RESULTS,
    tz: tzinfo | str | None = None,
    notes: WantNotes | None = None,
) -> list[BaseTool]:
    """create an ``intention_list`` tool: the deliberation candidate set, or the wants of one status.

    By default returns the user's ``open`` wants that are outside the surfacing
    cooldown (restraint brake #1), salience-ranked (restraint brake #2 --
    decayed wants sink); a ``status`` argument lists the wants of that status, or
    all of them. ``user_id`` is bound here and is the isolation boundary; the
    cooldown filter runs in the query, not by convention. The first line says
    how many are shown of how many there are ("3 of 7"), and how many open wants
    the cooldown held back.

    :param user_id: owning user whose wants to rank (isolation boundary)
    :ptype user_id: UUID
    :param agent_id: owning agent UUID (namespace owner + partition)
    :ptype agent_id: UUID
    :param customer_id: owning customer UUID (namespace scope grain)
    :ptype customer_id: UUID
    :param authorizer: intention authorizer dependency bundle
    :ptype authorizer: IntentionAuthorizerDependencies
    :param intentions_collection: three-tier intentions collection
    :ptype intentions_collection: IntentionsCollection
    :param cooldown_days: exclude wants surfaced within this many days
    :ptype cooldown_days: float
    :param max_results: how many wants a listing shows when the call names no limit
    :ptype max_results: int
    :param tz: the zone of the person the listing is read to; when each want was
        written down is said in it. ``None`` says it in UTC.
    :ptype tz: tzinfo | str | None
    :param notes: a consumer's own words about the wants shown, by id, each put at
        the end of its want's line
    :ptype notes: WantNotes | None
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    :raises ValueError: when ``tz`` is a name that is not a known zone
    """
    zone = resolve_zone(tz)

    @tool("intention_list", args_schema=IntentionListInput)
    async def intention_list(limit: int = max_results, status: str = IntentionStatus.OPEN.value) -> str:
        """list your wants, strongest first: the open ones by default."""
        try:
            await authorize_intention_access(
                action=ACTION_INTENTION_READ,
                agent_id=agent_id,
                customer_id=customer_id,
                caller_agent_id=agent_id,
                deps=authorizer,
            )
        except IntentionAccessDenied as exc:
            return _tool_error("intention_list", "authorize", str(exc))

        wanted = status.strip().lower()
        if wanted not in _LIST_STATUSES:
            return _tool_error(
                "intention_list", "input", f"invalid status '{status}'. Use one of: {', '.join(_LIST_STATUSES)}"
            )
        shown_at_most = max(1, min(limit, _MAX_LIMIT))
        held_back = 0
        try:
            if wanted == IntentionStatus.OPEN.value:
                cooldown_cutoff = datetime.now(UTC) - timedelta(days=cooldown_days)
                wants = await intentions_collection.find_open_for_deliberation(
                    user_id,
                    agent_id=agent_id,
                    cooldown_cutoff=cooldown_cutoff,
                )
                every = await intentions_collection.find_by_user(user_id, agent_id=agent_id)
                held_back = sum(1 for w in every if w.status == IntentionStatus.OPEN.value) - len(wants)
            else:
                every = await intentions_collection.find_by_user(user_id, agent_id=agent_id)
                wants = every if wanted == "all" else [w for w in every if w.status == wanted]
        except Exception as exc:
            contain(exc, log=log, what="intention_list query failed")
            return _tool_error("intention_list", "query", str(exc))

        held = (
            f" {held_back} more open want{'' if held_back == 1 else 's'} {'is' if held_back == 1 else 'are'} "
            f"held back because {'it was' if held_back == 1 else 'they were'} raised in the last "
            f"{cooldown_days:g} days."
            if held_back
            else ""
        )
        kind = "" if wanted == "all" else f"{wanted} "
        if not wants:
            return f"You have no {kind}wants.{held}"

        shown = wants[:shown_at_most]
        said = await _notes_for(notes, shown)
        more = f" Call again with a higher limit (up to {_MAX_LIMIT}) to see more." if len(shown) < len(wants) else ""
        lines = [f"{len(shown)} of {len(wants)} {kind}wants, strongest first.{held}{more}"]
        lines.extend(_want_line(w, zone, said.get(w.intention_id)) for w in shown)
        return "\n".join(lines)

    intention_list.description = (
        "List your wants, strongest first: the open ones unless you name a status (asked, granted, dropped, all). "
        "Each line has the want's [intention:<id>], its status and when you wrote it down."
    )
    return [intention_list]


def _want_line(want: IntentionEntity, zone: tzinfo | None, note: str | None) -> str:
    """One want as a listing shows it: handle, status, when written, the text, and the consumer's note."""
    written = render_local_time(want.date_created, zone)
    tail = f" ({note})" if note else ""
    head = f"- [intention:{want.intention_id}] {want.status}, written {written}: "
    return "\n".join(under(f"{want.content}{tail}", "  ", first=head))


async def _notes_for(notes: WantNotes | None, wants: Sequence[IntentionEntity]) -> Mapping[UUID, str]:
    """The consumer's notes on these wants; none when it has none or cannot say."""
    if notes is None or not wants:
        return {}
    try:
        return await notes(wants)
    except Exception as exc:  # prawduct:allow prawduct/broad-except -- a note is extra; the listing stands without it
        contain(exc, log=log, what="intention notes failed; listing without them", extra={"error": str(exc)})
        return {}


async def load_intention_get_tool(
    user_id: UUID,
    agent_id: UUID,
    customer_id: UUID,
    authorizer: IntentionAuthorizerDependencies,
    intentions_collection: IntentionsCollection,
    *,
    tz: tzinfo | str | None = None,
    notes: WantNotes | None = None,
) -> list[BaseTool]:
    """create an ``intention_get`` tool: one want in full, by its id.

    Ownership is enforced as ``intention_mark_surfaced`` enforces it: a want of
    another user reads back not-found.

    :param user_id: owning user (isolation boundary; ownership check)
    :ptype user_id: UUID
    :param agent_id: owning agent UUID (namespace owner + partition)
    :ptype agent_id: UUID
    :param customer_id: owning customer UUID (namespace scope grain)
    :ptype customer_id: UUID
    :param authorizer: intention authorizer dependency bundle
    :ptype authorizer: IntentionAuthorizerDependencies
    :param intentions_collection: three-tier intentions collection
    :ptype intentions_collection: IntentionsCollection
    :param tz: the zone the want's times are said in; ``None`` says them in UTC
    :ptype tz: tzinfo | str | None
    :param notes: a consumer's own words about the want, put on its last line
    :ptype notes: WantNotes | None
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    :raises ValueError: when ``tz`` is a name that is not a known zone
    """
    zone = resolve_zone(tz)

    @tool("intention_get", args_schema=IntentionGetInput)
    async def intention_get(intention_id: str) -> str:
        """read one of your wants in full."""
        try:
            await authorize_intention_access(
                action=ACTION_INTENTION_READ,
                agent_id=agent_id,
                customer_id=customer_id,
                caller_agent_id=agent_id,
                deps=authorizer,
            )
        except IntentionAccessDenied as exc:
            return _tool_error("intention_get", "authorize", str(exc))
        try:
            want_uuid = UUID(intention_id.strip().removeprefix("[").removesuffix("]").removeprefix("intention:"))
        except ValueError:
            return _tool_error("intention_get", "input", f"invalid intention_id '{intention_id}'")
        entity = await intentions_collection.get((agent_id, want_uuid))
        if entity is None or entity.user_id != user_id:
            return f"[TOOL ERROR] No want found for [intention:{intention_id}]."
        lines = [
            f"[intention:{entity.intention_id}] {entity.status}",
            f"Written down: {render_local_time(entity.date_created, zone)}",
        ]
        if entity.last_surfaced_at is not None:
            lines.append(f"Status last set: {render_local_time(entity.last_surfaced_at, zone)}")
        lines.append(f"Strength: {entity.salience:.2f}")
        if entity.source_memory_id is not None:
            lines.append(f"From: [memory:{entity.source_memory_id}]")
        if entity.source_conversation_id is not None:
            lines.append(f"Written in: [conversation:{entity.source_conversation_id}]")
        carried = (await _notes_for(notes, [entity])).get(entity.intention_id)
        if carried:
            lines.append(carried)
        lines.append(f"Want: {entity.content}")
        return "\n".join(lines)

    intention_get.description = (
        "Read one of your wants by its [intention:<id>]: what it says, its status, when you wrote it down "
        "and what it came from."
    )
    return [intention_get]


async def load_intention_mark_surfaced_tool(
    user_id: UUID,
    agent_id: UUID,
    customer_id: UUID,
    authorizer: IntentionAuthorizerDependencies,
    intentions_collection: IntentionsCollection,
) -> list[BaseTool]:
    """create an ``intention_mark_surfaced`` tool: move a want forward.

    Sets ``status`` and stamps ``last_surfaced_at`` (the cooldown clock),
    then emits the presence-timeline event -- :class:`IntentionSurfacedEvent`
    on ``asked``, :class:`IntentionResolvedEvent` on ``granted`` / ``dropped``.
    Ownership is enforced: a want whose ``user_id`` is not the bound
    ``user_id`` reads back as not-found (no cross-user writes, no
    existence leak), because ``agent_id`` alone does not isolate users.

    :param user_id: owning user (isolation boundary; ownership check)
    :ptype user_id: UUID
    :param agent_id: owning agent UUID (namespace owner + partition)
    :ptype agent_id: UUID
    :param customer_id: owning customer UUID (namespace scope grain)
    :ptype customer_id: UUID
    :param authorizer: intention authorizer dependency bundle
    :ptype authorizer: IntentionAuthorizerDependencies
    :param intentions_collection: three-tier intentions collection
    :ptype intentions_collection: IntentionsCollection
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """

    @tool("intention_mark_surfaced", args_schema=IntentionMarkSurfacedInput)
    async def intention_mark_surfaced(intention_id: str, new_status: str) -> str:
        """mark a want as raised to the user, granted, or dropped."""
        try:
            await authorize_intention_access(
                action=ACTION_INTENTION_WRITE,
                agent_id=agent_id,
                customer_id=customer_id,
                caller_agent_id=agent_id,
                deps=authorizer,
            )
        except IntentionAccessDenied as exc:
            return _tool_error("intention_mark_surfaced", "authorize", str(exc))

        status = new_status.strip().lower()
        if status not in _MARK_SURFACED_STATUSES:
            return _tool_error(
                "intention_mark_surfaced",
                "input",
                f"invalid new_status '{new_status}'. Use one of: {', '.join(_MARK_SURFACED_STATUSES)}",
            )

        try:
            want_uuid = UUID(intention_id)
        except ValueError:
            return _tool_error("intention_mark_surfaced", "input", f"invalid intention_id '{intention_id}'")

        entity = await intentions_collection.get((agent_id, want_uuid))
        # ownership check: agent_id is shared across users, so a bare pk
        # get would let one user mark another's want. treat a foreign /
        # missing want identically -- not-found, no existence leak.
        if entity is None or entity.user_id != user_id:
            return f"[TOOL ERROR] No want found for [intention:{intention_id}]."

        was = entity.status
        if was == status:
            return f"[NO CHANGE] intention_mark_surfaced: [intention:{want_uuid}] is already {status}: {entity.content}"
        now = datetime.now(UTC)
        entity.status = status
        entity.last_surfaced_at = now
        try:
            await intentions_collection.save_entity(entity)
        except Exception as exc:
            contain(exc, log=log, what="intention_mark_surfaced store failed")
            return _tool_error("intention_mark_surfaced", "store", str(exc))

        preview = entity.content[:_CONTENT_PREVIEW_LEN]
        # event payloads carry str-form uuids (FrameworkEvent wire contract,
        # matching MemoryConsolidatedEvent) -- this is the serialization border.
        event_user_id = str(user_id)  # convert at border: event payload wire uuid
        if status == IntentionStatus.ASKED.value:
            await _emit_intention_event(
                IntentionSurfacedEvent(
                    agent_id=str(agent_id),  # convert at border: event payload wire uuid
                    intention_id=str(want_uuid),
                    new_status=status,
                    user_id=event_user_id,
                    content_preview=preview,
                )
            )
        else:
            await _emit_intention_event(
                IntentionResolvedEvent(
                    agent_id=str(agent_id),  # convert at border: event payload wire uuid
                    intention_id=str(want_uuid),
                    new_status=status,
                    user_id=event_user_id,
                    content_preview=preview,
                )
            )

        log.info(
            "intention_mark_surfaced: want moved forward",
            extra={"extra_data": {"intention_id": str(want_uuid), "new_status": status}},
        )
        return f"Marked [intention:{want_uuid}] {status}; it was {was}. The want: {entity.content}"

    intention_mark_surfaced.description = (
        "Mark a want 'asked' once you have asked the person, 'granted' once it is done or given, "
        "or 'dropped' once you no longer want it. Only open wants are in intention_list's default list; "
        "list the others by status."
    )
    return [intention_mark_surfaced]
