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

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from uuid_utils import uuid7

from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from threetears.langgraph.events import FrameworkEvent, dispatch_event
from threetears.observe import get_logger

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
    "IntentionListInput",
    "IntentionLogInput",
    "IntentionMarkSurfacedInput",
    "load_intention_list_tool",
    "load_intention_log_tool",
    "load_intention_mark_surfaced_tool",
]

log = get_logger(__name__)

# luminara-grounded defaults, exposed as factory params so metallm can
# resolve per-user config values and pass them in (design §3 knobs).
_DEFAULT_COOLDOWN_DAYS = 7
_DEFAULT_DEDUP_THRESHOLD = 0.90
_DEFAULT_SALIENCE_SEED = 0.5
_DEFAULT_NEAR_DUP_BUMP = 0.05
_DEFAULT_MAX_RESULTS = 20

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
        log.warning(
            "intention embedding failed; degrading to soft-fail",
            extra={"extra_data": {"error": str(exc)}},
        )
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

    content: str = Field(description="The standing want, in your own words. One clear sentence.")
    source_memory_id: str | None = Field(
        default=None,
        description="Optional [memory:<id>] this want was deliberated from.",
    )


class IntentionListInput(BaseModel):
    """Input schema for the ``intention_list`` tool."""

    limit: int = Field(
        default=_DEFAULT_MAX_RESULTS,
        ge=1,
        le=100,
        description="Max wants to return. 1-100.",
    )


class IntentionMarkSurfacedInput(BaseModel):
    """Input schema for the ``intention_mark_surfaced`` tool."""

    intention_id: str = Field(description="[intention:<id>] UUID to update.")
    new_status: str = Field(
        description="asked (raised to the user) / granted / dropped.",
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
            log.warning(
                "intention_log: dedup lookup failed, creating new want",
                extra={"extra_data": {"error": str(exc)}},
            )
            similar = []

        for row in similar:
            if float(row["similarity"]) >= similarity_dedup_threshold:
                existing_id = row["intention_id"]
                existing = await intentions_collection.get((agent_id, existing_id))
                if existing is None:
                    continue
                # reinforcement: clear the cooldown anchor + refresh the
                # embedding via the entity save, then bump salience via the
                # raw pass. salience is immutable to the entity-UPDATE
                # generator (like memory), so the save can't carry a stale
                # salience back and revert a concurrent decay; the raw
                # bump_salience is the only salience writer.
                existing.last_surfaced_at = None
                existing.embedding = embedding
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
                    log.warning(
                        "intention_log: refresh save failed",
                        extra={"extra_data": {"intention_id": str(existing_id), "error": str(exc)}},
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
                    f"(similar at {float(row['similarity']):.0%}): {text}"
                )

        source_conversation_id: UUID | None = None
        if context_resolver is not None:
            try:
                live_ctx = context_resolver()
                raw_conv = getattr(live_ctx, "conversation_id", None)
                if raw_conv is not None:
                    source_conversation_id = raw_conv if isinstance(raw_conv, UUID) else UUID(str(raw_conv))
            except Exception as exc:
                log.debug(
                    "intention_log: context_resolver unavailable; source_conversation_id left null (%s)",
                    exc,
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
            log.warning(
                "intention_log: create failed",
                extra={"extra_data": {"error": str(exc)}},
            )
            return _tool_error("intention_log", "store", str(exc))

        log.info(
            "intention_log: stored new open want",
            extra={"extra_data": {"intention_id": str(intention_id), "content": text[:100]}},
        )
        return f"Logged as [intention:{intention_id}]: {text}"

    intention_log.description = (
        "Record a standing want you hold — something you'd like to do, ask, "
        "or return to across conversations. Dedups against your open wants: "
        "logging the same want again reinforces it, it doesn't duplicate. "
        "Returns [intention:<id>]."
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
) -> list[BaseTool]:
    """create an ``intention_list`` tool: the deliberation candidate set.

    Returns the user's ``open`` wants that are outside the surfacing
    cooldown (restraint brake #1), salience-ranked (restraint brake #2 --
    decayed wants sink). ``user_id`` is bound here and is the isolation
    boundary; the cooldown filter runs in the query, not by convention.

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
    :param max_results: cap on returned wants
    :ptype max_results: int
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """

    @tool("intention_list", args_schema=IntentionListInput)
    async def intention_list(limit: int = max_results) -> str:
        """list your open wants worth acting on now, most salient first."""
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

        cooldown_cutoff = datetime.now(UTC) - timedelta(days=cooldown_days)
        try:
            wants = await intentions_collection.find_open_for_deliberation(
                user_id,
                agent_id=agent_id,
                cooldown_cutoff=cooldown_cutoff,
            )
        except Exception as exc:
            return _tool_error("intention_list", "query", str(exc))

        if not wants:
            return "No open wants outside the cooldown window."

        wants = wants[: max(1, min(limit, max_results))]
        lines = [f"{len(wants)} open want(s), most salient first:"]
        for w in wants:
            lines.append(f"- [intention:{w.intention_id}] (salience {float(w.salience):.2f}) {w.content}")
        return "\n".join(lines)

    intention_list.description = (
        "List your open standing wants that are worth acting on right now — "
        "outside the recent-surfacing cooldown, most salient first. Use this "
        "when deliberating what (if anything) to raise. Returns [intention:<id>] items."
    )
    return [intention_list]


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
            return f"No want found for [intention:{intention_id}]."

        now = datetime.now(UTC)
        entity.status = status
        entity.last_surfaced_at = now
        try:
            await intentions_collection.save_entity(entity)
        except Exception as exc:
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
        return f"Marked [intention:{want_uuid}] as {status}."

    intention_mark_surfaced.description = (
        "Move a want forward: 'asked' once you've raised it to the user, "
        "'granted' or 'dropped' once resolved. Stamps the cooldown clock so "
        "an asked want isn't re-raised immediately. Takes an [intention:<id>]."
    )
    return [intention_mark_surfaced]
