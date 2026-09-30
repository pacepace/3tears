"""Six wake-schedule CRUD tools + the wake-yield cooperative tool.

Factory functions mint LangChain ``BaseTool`` instances bound to a
``(conversation_id, user_id, agent_id)`` actor triple plus the wake
Collections + a consumer-supplied :class:`WakeRegistryClient` for ACL
probes on ``skill_id`` attachments. The LLM never sees the actor IDs
in the input schema -- the factory closes over them.

The tools are scoped to the agent, not to the conversation they run in:
an agent's wakes live in its wake conversations, and a tool in any of the
agent's conversations lists and changes all of them. A wake of another
agent is never found. ``conversation_id`` is where the tool runs; a new
wake conversation made by ``wake_schedule_create`` has it as its parent.

A protected wake is refused by the pause, delete and update tools with a
plain message; the table's trigger refuses it from anywhere else.

Tools:

- :func:`load_wake_schedule_create_tool` -- ``wake_schedule_create``
- :func:`load_wake_schedule_update_tool` -- ``wake_schedule_update``
- :func:`load_wake_schedule_list_tool`   -- ``wake_schedule_list``
- :func:`load_wake_schedule_pause_tool`  -- ``wake_schedule_pause``
- :func:`load_wake_schedule_resume_tool` -- ``wake_schedule_resume``
- :func:`load_wake_schedule_delete_tool` -- ``wake_schedule_delete``
- :func:`load_wake_yield_tool`           -- ``wake_yield`` (PLACEMENT §8.5.1)

Per PLACEMENT §1.1 the skill-attach/detach tools are dropped; skill
attachment is set via ``wake_schedule_create(..., skill_id=...)`` and
``wake_schedule_update(..., skill_id=...)``.

Spec ref: ``docs/agent-wake/shard-04-agent-tools-and-webhook-adapter.md``
+ ``docs/long_running/PLACEMENT.md`` §1.1 / §1.3 / §1.5 / §1.6 /
§1.9 / §8.5.1.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal, Protocol
from uuid import UUID

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field
from uuid_utils import uuid7

from threetears.agent.wake.collections import WakeScheduleCollection
from threetears.agent.wake.config import (
    DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT as _CONFIG_DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT,
)
from threetears.agent.wake.rate_limit import (
    ScheduleCapExceeded,
    WakeConversationMaker,
    create_schedule_serialized,
    resume_schedule_serialized,
)
from threetears.agent.wake.types import WakeConversations
from threetears.scheduled_jobs import compute_next_fire_at
from threetears.agent.wake.tools.resolve import parse_conversation_id, parse_schedule_id
from threetears.agent.wake.tools.validators import (
    _ChainNode,
    validate_context_from_chain,
    validate_schedule_config,
)
from threetears.observe import get_logger

__all__ = [
    "DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT",
    "PROTECTED_REFUSAL",
    "NAME_MAX_LEN",
    "TASK_PROMPT_MAX_LEN",
    "ScheduleCreateInput",
    "ScheduleDeleteInput",
    "ScheduleIdInput",
    "ScheduleListInput",
    "ScheduleUpdateInput",
    "WakeRegistryClient",
    "WakeYieldProbe",
    "WakeYieldSetter",
    "load_wake_schedule_create_tool",
    "load_wake_schedule_delete_tool",
    "load_wake_schedule_list_tool",
    "load_wake_schedule_pause_tool",
    "load_wake_schedule_resume_tool",
    "load_wake_schedule_update_tool",
    "load_wake_yield_tool",
]


log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Limits + constants
# ---------------------------------------------------------------------------


NAME_MAX_LEN = 256
TASK_PROMPT_MAX_LEN = 4000
# Re-exported from :mod:`threetears.agent.wake.config` so the tool
# layer and the ``WakeConfig`` Protocol share a single source of truth
# for the per-agent cap default.
DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT = _CONFIG_DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT

# What the tools say about a protected wake. It is the agent's own; it keeps
# running whatever the agent does with its other wakes.
PROTECTED_REFUSAL = "this wake is protected: it cannot be paused, deleted or changed with this tool"


_VALID_MISSED_FIRE_POLICIES: frozenset[str] = frozenset({"coalesce", "catch_up"})


# ---------------------------------------------------------------------------
# Consumer-supplied wiring types
# ---------------------------------------------------------------------------


class WakeRegistryClient(Protocol):
    """Thin Protocol the consumer implements over its ACL + skills cache.

    Two surfaces, both async:

    - :meth:`acl_permits_skill` -- can ``(user_id, agent_id)`` attach
      the given ``skill_id`` to a wake? Used by ``wake_schedule_create``
      / ``wake_schedule_update`` and the webhook subscription
      counterparts. Mirrors the skills package's
      :meth:`threetears.agent.skills.tools.SkillRegistryClient.acl_permits`
      shape but takes a ``skill_id`` (UUID) rather than an mcp_name.
    - :meth:`skill_name_for_id` -- look up the human-readable name for a
      ``skill_id`` so the catalog rendering ("[schedule:<id>] ... skill:
      <name>") doesn't have to fetch the skill entity separately. May
      return ``None`` for unknown skills.

    The consumer's typical implementation:

    1. ``acl_permits_skill`` -- map ``skill_id`` -> the underlying
       ``AgentSkillEntity`` (via :class:`AgentSkillCollection.get`),
       check ``user_id`` ownership + ``enabled`` + then ACL via the
       product's ACL evaluator (``skill.invoke`` grant or
       ``tool.call`` ACL aggregated across the skill's
       ``tool_additions``).
    2. ``skill_name_for_id`` -- single-row fetch through the same
       collection.

    Tests substitute a fake -- see
    ``packages/agent/wake/tests/unit/test_tools_factories.py``.
    """

    async def acl_permits_skill(
        self,
        *,
        user_id: UUID,
        agent_id: UUID,
        skill_id: UUID,
    ) -> bool:
        """Return ``True`` iff ``(user_id, agent_id)`` may attach ``skill_id``.

        :param user_id: caller's user UUID
        :ptype user_id: UUID
        :param agent_id: caller's agent UUID
        :ptype agent_id: UUID
        :param skill_id: candidate skill to attach
        :ptype skill_id: UUID
        :return: ``True`` when the actor may attach the skill
        :rtype: bool
        """
        ...

    async def skill_name_for_id(
        self,
        *,
        user_id: UUID,
        agent_id: UUID,
        skill_id: UUID,
    ) -> str | None:
        """Return the human-readable name for ``skill_id`` (or ``None``).

        :param user_id: caller's user UUID
        :ptype user_id: UUID
        :param agent_id: caller's agent UUID
        :ptype agent_id: UUID
        :param skill_id: skill to look up
        :ptype skill_id: UUID
        :return: skill name or ``None`` if unknown/inaccessible
        :rtype: str | None
        """
        ...


# Probe/setter pair the consumer wires for wake-yield state
# (PLACEMENT §8.5.1). The probe returns ``True`` when the in-flight
# turn is wake-driven (i.e. the WakeYieldTool should load); the setter
# flags ``_yield_requested=True`` so the consumer's tool loop exits
# cleanly at the next iteration boundary.
WakeYieldProbe = Callable[[], bool]
WakeYieldSetter = Callable[[], None]


# ---------------------------------------------------------------------------
# Pydantic input schemas
# ---------------------------------------------------------------------------


_ScheduleTypeLiteral = Literal[
    "daily_at",
    "every_n_hours",
    "random_within_window",
    "one_shot_at",
    "cron",
    "relative_delay",
    "interval",
]


class ScheduleCreateInput(BaseModel):
    """Input schema for ``wake_schedule_create``."""

    schedule_type: _ScheduleTypeLiteral = Field(
        description="One of: daily_at, every_n_hours, random_within_window, one_shot_at, cron, relative_delay, interval.",
    )
    schedule_config: dict[str, Any] = Field(
        description="When it fires. The keys for each schedule_type are in the tool description.",
    )
    skill_id: str | None = Field(
        default=None,
        description="Optional skill to use when it fires: a [skill:<id>] or its id.",
    )
    wake_conversation_id: str | None = Field(
        default=None,
        description=(
            "Optional [conversation:<id>] of one of your wake conversations, to add this wake to it. "
            "Leave it out to start a new wake conversation."
        ),
    )
    missed_fire_policy: Literal["coalesce", "catch_up"] = Field(
        default="coalesce",
        description="After missed times: 'coalesce' fires once, 'catch_up' fires once for each missed time.",
    )
    task_prompt: str | None = Field(
        default=None,
        description="Optional instructions for each wake, up to 4000 characters. Write them to stand on their own.",
    )
    name: str | None = Field(
        default=None,
        description="Optional name, up to 256 characters.",
    )
    context_from_schedule_id: str | None = Field(
        default=None,
        description="Optional [schedule:<id>] of any of your wakes. When this one fires, you are given that one's last output.",
    )
    include_conversation_history: bool = Field(
        default=True,
        description="True (default): wake with its wake conversation's recent messages. False: wake without them.",
    )


class ScheduleUpdateInput(BaseModel):
    """Input schema for ``wake_schedule_update``.

    Every non-``schedule_id`` field is optional; only fields the LLM
    passes get applied. Because LangChain ``@tool`` cannot distinguish
    "field absent" from "explicit null" at the JSON layer, detachment
    of nullable references uses explicit boolean fields:

    - ``detach_skill=true`` clears the attached skill (regardless of
      what ``skill_id`` holds; passing both is rejected).
    - ``detach_context_from=true`` clears ``context_from_schedule_id``.
    - ``clear_name=true`` clears the optional human-readable name.
    """

    schedule_id: str = Field(description="The [schedule:<id>] to change.")
    schedule_type: _ScheduleTypeLiteral | None = None
    schedule_config: dict[str, Any] | None = None
    skill_id: str | None = Field(
        default=None,
        description="A skill to attach: a [skill:<id>] or its id. To remove the skill, use detach_skill instead.",
    )
    detach_skill: bool = Field(
        default=False,
        description="True removes the attached skill. Do not also pass skill_id.",
    )
    missed_fire_policy: Literal["coalesce", "catch_up"] | None = None
    task_prompt: str | None = None
    name: str | None = None
    clear_name: bool = Field(
        default=False,
        description="True removes the name. Do not also pass name.",
    )
    context_from_schedule_id: str | None = None
    detach_context_from: bool = Field(
        default=False,
        description="True removes context_from_schedule_id. Do not also pass context_from_schedule_id.",
    )
    include_conversation_history: bool | None = Field(
        default=None,
        description="Change whether it wakes with its wake conversation's recent messages (true) or without them (false).",
    )


class ScheduleListInput(BaseModel):
    """Input schema for ``wake_schedule_list``."""

    include_paused: bool = Field(
        default=True,
        description="Include paused schedules. Default true.",
    )


class ScheduleIdInput(BaseModel):
    """Shared input for pause / resume actions."""

    schedule_id: str = Field(description="The [schedule:<id>].")


class ScheduleDeleteInput(BaseModel):
    """Input schema for ``wake_schedule_delete``."""

    schedule_id: str = Field(description="The [schedule:<id>] to delete.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tool_error(tool_name: str, description: str) -> str:
    """Format the canonical ``[TOOL ERROR] <tool>: <description>``."""
    return f"[TOOL ERROR] {tool_name}: {description}"


def _validate_name(name: str | None) -> str | None:
    """Validate optional schedule/subscription name."""
    if name is None:
        return None
    if not isinstance(name, str):
        return "name must be a string"
    if len(name) > NAME_MAX_LEN:
        return f"name must be at most {NAME_MAX_LEN} characters"
    return None


def _validate_task_prompt(prompt: str | None) -> str | None:
    """Validate optional per-fire task prompt."""
    if prompt is None:
        return None
    if not isinstance(prompt, str):
        return "task_prompt must be a string or null"
    if len(prompt) > TASK_PROMPT_MAX_LEN:
        return f"task_prompt must be at most {TASK_PROMPT_MAX_LEN} characters"
    return None


def _format_next_fire(value: datetime | None) -> str:
    """Render ``next_fire_at`` for the catalog row; ``-`` when unset."""
    if value is None:
        return "-"
    return value.isoformat()


def _format_schedule_line(
    entity: Any,
    *,
    skill_name: str | None,
) -> str:
    """Render a one-line catalog entry for a schedule row."""
    name = entity.name or "untitled"
    skill_segment = f" · skill: {skill_name}" if skill_name else ""
    # Surface the history switch only when OFF -- the default (on) is the
    # common case and would add noise to every catalog line.
    history_segment = "" if entity.include_conversation_history else " · history: off"
    protected_segment = " · protected" if entity.protected else ""
    return (
        f"[schedule:{entity.schedule_id}] · {name} · {entity.schedule_type}"
        f" · next: {_format_next_fire(entity.next_fire_at)}"
        f" · {entity.status} · in [conversation:{entity.conversation_id}]"
        f"{skill_segment}{history_segment}{protected_segment}"
    )


async def _check_skill_acl(
    *,
    registry: WakeRegistryClient,
    user_id: UUID,
    agent_id: UUID,
    skill_id: UUID,
    tool_name: str,
) -> str | None:
    """Probe ACL for one skill attachment; return error string or ``None``."""
    try:
        permitted = await registry.acl_permits_skill(
            user_id=user_id,
            agent_id=agent_id,
            skill_id=skill_id,
        )
    except Exception as exc:  # noqa: BLE001 - boundary; surface as tool error
        log.warning(
            "wake schedule skill ACL probe raised",
            extra={"extra_data": {"skill_id": str(skill_id), "error": str(exc)}},
        )
        return f"skill_id {skill_id} ACL probe failed: {exc}"
    if not permitted:
        return f"skill_id {skill_id} not authorized for this user/agent"
    return None


def _make_chain_resolver(
    schedules_collection: WakeScheduleCollection,
    agent_id: UUID,
) -> Callable[[UUID], Any]:
    """Build the closure :func:`validate_context_from_chain` consumes.

    Wraps :meth:`WakeScheduleCollection.find_for_agent` so the validator
    stays free of DB knowledge. Another agent's schedule is not found, so
    a chain can never reach outside the agent.
    """

    async def resolver(schedule_id: UUID) -> _ChainNode | None:
        entity = await schedules_collection.find_for_agent(agent_id, schedule_id)
        if entity is None:
            return None
        return _ChainNode(
            agent_id=entity.agent_id,
            context_from_schedule_id=entity.context_from_schedule_id,
        )

    return resolver


# ---------------------------------------------------------------------------
# wake_schedule_create
# ---------------------------------------------------------------------------


def load_wake_schedule_create_tool(
    *,
    conversation_id: UUID,
    user_id: UUID,
    agent_id: UUID,
    schedules_collection: WakeScheduleCollection,
    registry: WakeRegistryClient,
    max_active_schedules_per_agent: int = DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT,
    wake_conversations: WakeConversations | None = None,
) -> list[BaseTool]:
    """Build a ``wake_schedule_create`` tool bound to the actor triple.

    Per Requirement TOOL-09 + PLACEMENT §1.18 the consumer decides when
    to include this tool in the loaded set (e.g. a consumer suppresses it
    on wake-driven turns to prevent recursive cron creation).

    With ``wake_conversations`` a wake lives in a wake conversation: the
    one the agent names, which must be one of its wake conversations, or a
    new one made in the create's transaction with the caller's conversation
    as its parent. Without it the wake lives in the caller's conversation.

    :param conversation_id: caller's conversation UUID; the parent of a new
        wake conversation
    :ptype conversation_id: UUID
    :param user_id: caller's user UUID
    :ptype user_id: UUID
    :param agent_id: caller's agent UUID
    :ptype agent_id: UUID
    :param schedules_collection: three-tier schedules collection
    :ptype schedules_collection: WakeScheduleCollection
    :param registry: consumer-supplied registry for ACL probes
    :ptype registry: WakeRegistryClient
    :param max_active_schedules_per_agent: cap on the agent's unprotected
        active schedules, across all its conversations
    :ptype max_active_schedules_per_agent: int
    :param wake_conversations: the consumer's wake-conversation hooks
    :ptype wake_conversations: WakeConversations | None
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """

    @tool("wake_schedule_create", args_schema=ScheduleCreateInput)
    async def wake_schedule_create(
        schedule_type: str,
        schedule_config: dict[str, Any],
        skill_id: str | None = None,
        wake_conversation_id: str | None = None,
        missed_fire_policy: Literal["coalesce", "catch_up"] = "coalesce",
        task_prompt: str | None = None,
        name: str | None = None,
        context_from_schedule_id: str | None = None,
        include_conversation_history: bool = True,
    ) -> str:
        """Create a wake schedule."""
        for err in (
            _validate_name(name),
            _validate_task_prompt(task_prompt),
            validate_schedule_config(schedule_type, schedule_config),
        ):
            if err is not None:
                return _tool_error("wake_schedule_create", err)

        target_conversation_id = conversation_id
        if wake_conversation_id is not None:
            if wake_conversations is None:
                return _tool_error(
                    "wake_schedule_create", "wakes here live in this conversation; omit wake_conversation_id"
                )
            parsed_target = parse_conversation_id(wake_conversation_id)
            if parsed_target is None:
                return _tool_error("wake_schedule_create", f"invalid wake_conversation_id {wake_conversation_id!r}")
            if not await wake_conversations.is_wake_conversation(agent_id=agent_id, conversation_id=parsed_target):
                return _tool_error(
                    "wake_schedule_create",
                    f"[conversation:{parsed_target}] is not one of your wake conversations",
                )
            target_conversation_id = parsed_target

        if missed_fire_policy not in _VALID_MISSED_FIRE_POLICIES:
            return _tool_error(
                "wake_schedule_create",
                f"missed_fire_policy must be 'coalesce' or 'catch_up'; got {missed_fire_policy!r}",
            )

        # The per-agent cap is enforced RACE-PROOF at the persist step below
        # via :func:`create_schedule_serialized`, which takes the agent's
        # advisory lock, re-counts, and inserts inside one transaction -- so
        # two concurrent creates can never both pass a stale count and exceed
        # the cap. The cap-reject event + metric fire on the
        # ``ScheduleCapExceeded`` branch at insert time.

        attached_skill: UUID | None = None
        if skill_id is not None:
            stripped = skill_id.strip()
            # Tag-confusion guard: a [schedule:<id>] tag is NEVER a valid
            # skill_id (it's a schedule identifier). Surface as a clear
            # error so the LLM corrects the call instead of silently
            # ACL-denying a UUID that happens to parse.
            if stripped.startswith("[schedule:"):
                return _tool_error(
                    "wake_schedule_create",
                    f"skill_id received a schedule tag {skill_id!r}; "
                    "did you mean to pass a skill catalog id from "
                    "skill_list/skill_get? Schedule tags are not valid "
                    "skill identifiers.",
                )
            if stripped.startswith("[skill:") and stripped.endswith("]"):
                stripped = stripped[len("[skill:") : -1].strip()
            parsed_skill: UUID | None
            try:
                parsed_skill = UUID(stripped)
            except ValueError:
                parsed_skill = None
            except AttributeError:
                parsed_skill = None
            except TypeError:
                parsed_skill = None
            if parsed_skill is None:
                return _tool_error(
                    "wake_schedule_create",
                    f"invalid skill_id {skill_id!r} (use [skill:<uuid>] or bare UUID)",
                )
            err = await _check_skill_acl(
                registry=registry,
                user_id=user_id,
                agent_id=agent_id,
                skill_id=parsed_skill,
                tool_name="wake_schedule_create",
            )
            if err is not None:
                return _tool_error("wake_schedule_create", err)
            attached_skill = parsed_skill

        parsed_context_from: UUID | None = None
        if context_from_schedule_id is not None:
            parsed_context_from = parse_schedule_id(context_from_schedule_id)
            if parsed_context_from is None:
                return _tool_error(
                    "wake_schedule_create",
                    f"invalid context_from_schedule_id {context_from_schedule_id!r}",
                )

        # Compute next_fire_at via the pure helper from shard 02.
        now = datetime.now(UTC)
        try:
            next_fire_at = compute_next_fire_at(
                schedule_type,
                schedule_config,
                missed_fire_policy,
                last_fired_at=None,
                now=now,
            )
        except ValueError as exc:
            return _tool_error(
                "wake_schedule_create",
                f"schedule_config rejected by reschedule engine: {exc}",
            )

        new_schedule_id = UUID(str(uuid7()))

        # Run cycle detection AFTER the new id is minted so the walker
        # can seed visited={new_id} and catch self-references / loops.
        if parsed_context_from is not None:
            resolver = _make_chain_resolver(schedules_collection, agent_id)
            err = await validate_context_from_chain(
                new_schedule_id=new_schedule_id,
                proposed_context_from=parsed_context_from,
                agent_id=agent_id,
                resolver=resolver,
            )
            if err is not None:
                return _tool_error("wake_schedule_create", err)

        make_wake_conversation: WakeConversationMaker | None = None
        if wake_conversations is not None and wake_conversation_id is None:
            hooks = wake_conversations

            async def _make(conn: Any) -> UUID:
                return await hooks.create(
                    parent_conversation_id=conversation_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    name=name,
                    conn=conn,
                )

            make_wake_conversation = _make

        data: dict[str, Any] = {
            "schedule_id": new_schedule_id,
            "conversation_id": target_conversation_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "skill_id": attached_skill,
            "schedule_type": schedule_type,
            "schedule_config": dict(schedule_config),
            "task_prompt": task_prompt,
            "execution_mode": "spawn",
            "status": "active",
            "next_fire_at": next_fire_at,
            "last_fired_at": None,
            "name": name,
            "missed_fire_policy": missed_fire_policy,
            "context_from_schedule_id": parsed_context_from,
            "include_conversation_history": include_conversation_history,
            "protected": False,
            "date_created": now,
            "date_updated": now,
        }
        try:
            # Race-proof cap + insert: the helper takes the agent's advisory
            # lock, re-counts active schedules, makes the wake conversation
            # when one is wanted, and inserts, inside one transaction. The
            # collection's pool is the serialization substrate.
            entity = await create_schedule_serialized(
                collection=schedules_collection,
                data=data,
                agent_id=agent_id,
                cap=max_active_schedules_per_agent,
                pool=schedules_collection.l3_pool,
                make_wake_conversation=make_wake_conversation,
            )
        except ScheduleCapExceeded:
            # local import keeps the tool layer's cold-import cost the
            # same on the happy path; metrics + events only matter on
            # the rejection branch. The Loki event is intentionally
            # emitted here (not in the helper) so the trigger context
            # -- user_id, agent_id -- lands on the structured event.
            from threetears.agent.wake.events import EVENT_SCHEDULE_CAP_REJECT  # noqa: PLC0415
            from threetears.agent.wake.metrics import get_wake_emitter  # noqa: PLC0415

            get_wake_emitter().inc_schedule_cap_rejection()
            # convert at border: schedule-cap-reject (create) log extra_data fields
            log_user_id = str(user_id)
            log_agent_id = str(agent_id)
            log.info(
                EVENT_SCHEDULE_CAP_REJECT,
                extra={
                    "extra_data": {
                        "conversation_id": str(conversation_id),
                        "user_id": log_user_id,
                        "agent_id": log_agent_id,
                        "cap": max_active_schedules_per_agent,
                    }
                },
            )
            return _tool_error(
                "wake_schedule_create",
                f"you have {max_active_schedules_per_agent} active wakes, the most allowed (pause or delete one first)",
            )
        except Exception as exc:  # noqa: BLE001 - surface persistence failure
            log.warning(
                "wake_schedule_create persist failed",
                extra={"extra_data": {"schedule_id": str(new_schedule_id), "error": str(exc)}},
            )
            return _tool_error("wake_schedule_create", f"persist failed: {exc}")

        skill_name: str | None = None
        if attached_skill is not None:
            try:
                skill_name = await registry.skill_name_for_id(
                    user_id=user_id,
                    agent_id=agent_id,
                    skill_id=attached_skill,
                )
            except Exception as exc:  # noqa: BLE001 - name lookup is best-effort
                log.warning(
                    "wake_schedule_create skill_name lookup raised",
                    extra={"extra_data": {"skill_id": str(attached_skill), "error": str(exc)}},
                )
                skill_name = None

        log.info(
            "wake_schedule_create persisted",
            extra={
                "extra_data": {
                    "schedule_id": str(new_schedule_id),
                    "schedule_type": schedule_type,
                    "wake_conversation_id": str(entity.conversation_id),
                    "skill_id": str(attached_skill) if attached_skill else None,
                    "next_fire_at": next_fire_at.isoformat() if next_fire_at else None,
                }
            },
        )
        return _format_schedule_line(entity, skill_name=skill_name)

    wake_schedule_create.description = (
        (
            "Wake yourself at a set time or on a repeat. Each wake lives in a wake conversation, "
            "and each time it fires it starts a new conversation.\n"
            if wake_conversations is not None
            else "Wake yourself in this conversation at a set time or on a repeat.\n"
        )
        + "schedule_config keys for each schedule_type: daily_at {hour, minute, tz}; "
        "every_n_hours {n}; random_within_window {start_hour, end_hour, tz, fires_per_day}; "
        "one_shot_at {fire_at_iso}; cron {expr}; relative_delay {delay, e.g. '30m'}; "
        "interval {seconds}.\n"
        f"Returns [schedule:<id>]. At most {max_active_schedules_per_agent} active wakes."
    )

    return [wake_schedule_create]


# ---------------------------------------------------------------------------
# wake_schedule_update
# ---------------------------------------------------------------------------


def load_wake_schedule_update_tool(
    *,
    user_id: UUID,
    agent_id: UUID,
    schedules_collection: WakeScheduleCollection,
    registry: WakeRegistryClient,
) -> list[BaseTool]:
    """Build a ``wake_schedule_update`` tool with partial-update semantics.

    Re-runs the same validators ``wake_schedule_create`` enforces on any
    field the caller passes. Detachment of nullable references uses
    explicit boolean fields (``detach_skill``, ``detach_context_from``,
    ``clear_name``) because LangChain ``@tool`` decoration cannot
    distinguish "field absent from JSON input" from "explicit null in
    JSON input". Passing the attach value AND the detach flag together
    is contradictory and rejected. Changes any of the agent's wakes,
    whichever conversation it lives in; a protected wake is refused.

    :param user_id: caller's user UUID
    :ptype user_id: UUID
    :param agent_id: caller's agent UUID
    :ptype agent_id: UUID
    :param schedules_collection: three-tier schedules collection
    :ptype schedules_collection: WakeScheduleCollection
    :param registry: consumer-supplied registry client
    :ptype registry: WakeRegistryClient
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """

    @tool("wake_schedule_update", args_schema=ScheduleUpdateInput)
    async def wake_schedule_update(
        schedule_id: str,
        schedule_type: str | None = None,
        schedule_config: dict[str, Any] | None = None,
        skill_id: str | None = None,
        detach_skill: bool = False,
        missed_fire_policy: Literal["coalesce", "catch_up"] | None = None,
        task_prompt: str | None = None,
        name: str | None = None,
        clear_name: bool = False,
        context_from_schedule_id: str | None = None,
        detach_context_from: bool = False,
        include_conversation_history: bool | None = None,
    ) -> str:
        """Edit a wake schedule in place. Pass only fields to change."""
        # Contradictory-input guards. Attach value + detach flag is
        # ambiguous; reject before touching the DB.
        if skill_id is not None and detach_skill:
            return _tool_error(
                "wake_schedule_update",
                "skill_id and detach_skill=true cannot be combined; "
                "pass exactly one (or neither, to leave the attachment unchanged).",
            )
        if name is not None and clear_name:
            return _tool_error(
                "wake_schedule_update",
                "name and clear_name=true cannot be combined; "
                "pass exactly one (or neither, to leave the name unchanged).",
            )
        if context_from_schedule_id is not None and detach_context_from:
            return _tool_error(
                "wake_schedule_update",
                "context_from_schedule_id and detach_context_from=true "
                "cannot be combined; pass exactly one (or neither).",
            )

        parsed = parse_schedule_id(schedule_id)
        if parsed is None:
            return _tool_error("wake_schedule_update", f"invalid schedule_id {schedule_id!r}")

        entity = await schedules_collection.find_for_agent(agent_id, parsed)
        if entity is None:
            return _tool_error("wake_schedule_update", "schedule not found")
        if entity.protected:
            return _tool_error("wake_schedule_update", PROTECTED_REFUSAL)

        merged_type = schedule_type if schedule_type is not None else entity.schedule_type
        merged_config = dict(schedule_config) if schedule_config is not None else dict(entity.schedule_config)

        # Validation pass over fields the caller changed or that are
        # affected by a changed neighbour.
        validation_errors: list[str | None] = []
        if name is not None:
            validation_errors.append(_validate_name(name))
        if task_prompt is not None:
            validation_errors.append(_validate_task_prompt(task_prompt))
        if schedule_type is not None or schedule_config is not None:
            validation_errors.append(validate_schedule_config(merged_type, merged_config))
        for err in validation_errors:
            if err is not None:
                return _tool_error("wake_schedule_update", err)

        if missed_fire_policy is not None and missed_fire_policy not in _VALID_MISSED_FIRE_POLICIES:
            return _tool_error(
                "wake_schedule_update",
                f"missed_fire_policy must be 'coalesce' or 'catch_up'; got {missed_fire_policy!r}",
            )

        # Resolve skill_id change. detach_skill=True clears the
        # attachment; a non-null skill_id attaches the new skill (after
        # ACL check); both unset is a no-op. skill_change[1] is True
        # when entity.skill_id should be overwritten with skill_change[0].
        skill_change: tuple[UUID | None, bool]
        if detach_skill:
            skill_change = (None, True)
        elif skill_id is not None:
            stripped = skill_id.strip()
            # Tag-confusion guard: schedule tag is not a skill id.
            if stripped.startswith("[schedule:"):
                return _tool_error(
                    "wake_schedule_update",
                    f"skill_id received a schedule tag {skill_id!r}; "
                    "did you mean to pass a skill catalog id from "
                    "skill_list/skill_get? Schedule tags are not valid "
                    "skill identifiers.",
                )
            if stripped.startswith("[skill:") and stripped.endswith("]"):
                stripped = stripped[len("[skill:") : -1].strip()
            new_skill: UUID
            try:
                new_skill = UUID(stripped)
            except ValueError:
                return _tool_error("wake_schedule_update", f"invalid skill_id {skill_id!r}")
            except AttributeError:
                return _tool_error("wake_schedule_update", f"invalid skill_id {skill_id!r}")
            except TypeError:
                return _tool_error("wake_schedule_update", f"invalid skill_id {skill_id!r}")
            err = await _check_skill_acl(
                registry=registry,
                user_id=user_id,
                agent_id=agent_id,
                skill_id=new_skill,
                tool_name="wake_schedule_update",
            )
            if err is not None:
                return _tool_error("wake_schedule_update", err)
            skill_change = (new_skill, True)
        else:
            skill_change = (None, False)

        # context_from_schedule_id change.
        new_context_from: UUID | None
        context_from_changed = False
        if detach_context_from:
            new_context_from = None
            context_from_changed = True
        elif context_from_schedule_id is not None:
            new_context_from = parse_schedule_id(context_from_schedule_id)
            if new_context_from is None:
                return _tool_error(
                    "wake_schedule_update",
                    f"invalid context_from_schedule_id {context_from_schedule_id!r}",
                )
            context_from_changed = True
        else:
            new_context_from = entity.context_from_schedule_id

        if context_from_changed and new_context_from is not None:
            resolver = _make_chain_resolver(schedules_collection, agent_id)
            err = await validate_context_from_chain(
                new_schedule_id=entity.schedule_id,
                proposed_context_from=new_context_from,
                agent_id=agent_id,
                resolver=resolver,
            )
            if err is not None:
                return _tool_error("wake_schedule_update", err)

        # Apply changes via setters.
        recompute_next_fire = False
        if schedule_type is not None:
            entity.schedule_type = schedule_type
            recompute_next_fire = True
        if schedule_config is not None:
            entity.schedule_config = dict(schedule_config)
            recompute_next_fire = True
        if missed_fire_policy is not None:
            entity.missed_fire_policy = missed_fire_policy
        if task_prompt is not None:
            entity.task_prompt = task_prompt if task_prompt else None
        if clear_name:
            entity.name = None
        elif name is not None:
            entity.name = name
        if skill_change[1]:
            entity.skill_id = skill_change[0]
        if context_from_changed:
            entity.context_from_schedule_id = new_context_from
        if include_conversation_history is not None:
            entity.include_conversation_history = include_conversation_history

        # Recompute next_fire_at if WHEN-relevant fields changed and
        # the schedule is active (paused schedules have NULL
        # next_fire_at by design).
        if recompute_next_fire and entity.status == "active":
            try:
                entity.next_fire_at = compute_next_fire_at(
                    entity.schedule_type,
                    entity.schedule_config,
                    entity.missed_fire_policy,
                    last_fired_at=entity.last_fired_at,
                    now=datetime.now(UTC),
                )
            except ValueError as exc:
                return _tool_error(
                    "wake_schedule_update",
                    f"schedule_config rejected by reschedule engine: {exc}",
                )

        entity.date_updated = datetime.now(UTC)
        try:
            await schedules_collection.save_entity(entity)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "wake_schedule_update persist failed",
                extra={"extra_data": {"schedule_id": str(parsed), "error": str(exc)}},
            )
            return _tool_error("wake_schedule_update", f"persist failed: {exc}")

        skill_name: str | None = None
        if entity.skill_id is not None:
            try:
                skill_name = await registry.skill_name_for_id(
                    user_id=user_id,
                    agent_id=agent_id,
                    skill_id=entity.skill_id,
                )
            except Exception:  # noqa: BLE001 - name lookup is best-effort
                skill_name = None
        return _format_schedule_line(entity, skill_name=skill_name)

    wake_schedule_update.description = (
        "Change a wake schedule. Pass only what changes. To remove a skill, name or "
        "context_from, pass detach_skill, clear_name or detach_context_from, not the value "
        "as well. Returns the schedule as it now stands."
    )

    return [wake_schedule_update]


# ---------------------------------------------------------------------------
# wake_schedule_list
# ---------------------------------------------------------------------------


def load_wake_schedule_list_tool(
    *,
    user_id: UUID,
    agent_id: UUID,
    schedules_collection: WakeScheduleCollection,
    registry: WakeRegistryClient,
) -> list[BaseTool]:
    """Build a ``wake_schedule_list`` tool listing the agent's wakes in every conversation.

    :param user_id: caller's user UUID
    :ptype user_id: UUID
    :param agent_id: caller's agent UUID
    :ptype agent_id: UUID
    :param schedules_collection: three-tier schedules collection
    :ptype schedules_collection: WakeScheduleCollection
    :param registry: consumer-supplied registry for skill_name lookups
    :ptype registry: WakeRegistryClient
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """

    @tool("wake_schedule_list", args_schema=ScheduleListInput)
    async def wake_schedule_list(include_paused: bool = True) -> str:
        """List your wake schedules."""
        try:
            visible = await schedules_collection.list_for_agent(agent_id, include_paused=include_paused)
        except Exception as exc:  # noqa: BLE001
            return _tool_error("wake_schedule_list", f"list failed: {exc}")

        if not visible:
            return "You have no wake schedules."

        lines: list[str] = [f"Found {len(visible)} schedules:"]
        for entity in visible:
            skill_name: str | None = None
            if entity.skill_id is not None:
                try:
                    skill_name = await registry.skill_name_for_id(
                        user_id=user_id,
                        agent_id=agent_id,
                        skill_id=entity.skill_id,
                    )
                except Exception:  # noqa: BLE001 - best-effort
                    skill_name = None
            lines.append("- " + _format_schedule_line(entity, skill_name=skill_name))
        return "\n".join(lines)

    wake_schedule_list.description = (
        "List your wake schedules, in every conversation: id, name, type, next time, status "
        "and the wake conversation each lives in."
    )

    return [wake_schedule_list]


# ---------------------------------------------------------------------------
# wake_schedule_pause
# ---------------------------------------------------------------------------


def load_wake_schedule_pause_tool(
    *,
    agent_id: UUID,
    schedules_collection: WakeScheduleCollection,
) -> list[BaseTool]:
    """Build a ``wake_schedule_pause`` tool (status -> 'paused', clear next_fire_at).

    Pauses any of the agent's wakes, whichever conversation it lives in. A
    protected wake is refused with a plain message.

    :param agent_id: caller's agent UUID
    :ptype agent_id: UUID
    :param schedules_collection: three-tier schedules collection
    :ptype schedules_collection: WakeScheduleCollection
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """

    @tool("wake_schedule_pause", args_schema=ScheduleIdInput)
    async def wake_schedule_pause(schedule_id: str) -> str:
        """Pause a wake schedule. Reversible via wake_schedule_resume."""
        parsed = parse_schedule_id(schedule_id)
        if parsed is None:
            return _tool_error("wake_schedule_pause", f"invalid schedule_id {schedule_id!r}")

        entity = await schedules_collection.find_for_agent(agent_id, parsed)
        if entity is None:
            return _tool_error("wake_schedule_pause", "schedule not found")
        if entity.protected:
            return _tool_error("wake_schedule_pause", PROTECTED_REFUSAL)
        if entity.status == "expired":
            return _tool_error(
                "wake_schedule_pause",
                "schedule is expired (one-shot fires cannot be paused); create a new schedule",
            )

        try:
            await schedules_collection.pause(entity.conversation_id, parsed)
        except Exception as exc:  # noqa: BLE001
            return _tool_error("wake_schedule_pause", f"persist failed: {exc}")
        return f"Paused [schedule:{parsed}]."

    wake_schedule_pause.description = "Pause a wake schedule. It does not fire until wake_schedule_resume."
    return [wake_schedule_pause]


# ---------------------------------------------------------------------------
# wake_schedule_resume
# ---------------------------------------------------------------------------


def load_wake_schedule_resume_tool(
    *,
    agent_id: UUID,
    schedules_collection: WakeScheduleCollection,
    max_active_schedules_per_agent: int = DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT,
) -> list[BaseTool]:
    """Build a ``wake_schedule_resume`` tool (status -> 'active'; recompute next_fire_at).

    Resumes any of the agent's wakes, whichever conversation it lives in.
    Re-activation is RACE-PROOF cap-checked: the persist step routes
    through :func:`resume_schedule_serialized`, which takes the agent's
    advisory lock, re-counts active schedules (excluding the one being
    resumed), and flips paused -> active inside one transaction. Without
    this a pause -> create-to-fill -> resume sequence could push the active
    count past the cap.

    :param agent_id: caller's agent UUID
    :ptype agent_id: UUID
    :param schedules_collection: three-tier schedules collection
    :ptype schedules_collection: WakeScheduleCollection
    :param max_active_schedules_per_agent: cap on the agent's unprotected
        active schedules
    :ptype max_active_schedules_per_agent: int
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """

    @tool("wake_schedule_resume", args_schema=ScheduleIdInput)
    async def wake_schedule_resume(schedule_id: str) -> str:
        """Resume a paused wake schedule. Recomputes next_fire_at from now."""
        parsed = parse_schedule_id(schedule_id)
        if parsed is None:
            return _tool_error("wake_schedule_resume", f"invalid schedule_id {schedule_id!r}")

        entity = await schedules_collection.find_for_agent(agent_id, parsed)
        if entity is None:
            return _tool_error("wake_schedule_resume", "schedule not found")
        if entity.status == "expired":
            return _tool_error(
                "wake_schedule_resume",
                "schedule is expired (one-shot already fired); create a new schedule",
            )

        try:
            next_fire_at = compute_next_fire_at(
                entity.schedule_type,
                entity.schedule_config,
                entity.missed_fire_policy,
                last_fired_at=entity.last_fired_at,
                now=datetime.now(UTC),
            )
        except ValueError as exc:
            return _tool_error(
                "wake_schedule_resume",
                f"reschedule failed: {exc}",
            )
        if next_fire_at is None:
            # Terminal one-shot whose last_fired_at is set; should be
            # expired but defend in depth.
            return _tool_error(
                "wake_schedule_resume",
                "schedule has no future fire (one-shot already fired); create a new schedule",
            )

        try:
            # Race-proof cap + flip: the helper takes the agent's advisory
            # lock, re-counts active schedules (excluding this one), and
            # flips paused -> active inside one transaction.
            await resume_schedule_serialized(
                collection=schedules_collection,
                agent_id=agent_id,
                conversation_id=entity.conversation_id,
                schedule_id=parsed,
                next_fire_at=next_fire_at,
                cap=max_active_schedules_per_agent,
                pool=schedules_collection.l3_pool,
            )
        except ScheduleCapExceeded:
            from threetears.agent.wake.events import EVENT_SCHEDULE_CAP_REJECT  # noqa: PLC0415
            from threetears.agent.wake.metrics import get_wake_emitter  # noqa: PLC0415

            get_wake_emitter().inc_schedule_cap_rejection()
            log.info(
                EVENT_SCHEDULE_CAP_REJECT,
                extra={
                    "extra_data": {
                        "agent_id": str(
                            agent_id
                        ),  # convert at border: schedule-cap-reject (resume) log extra_data field
                        "schedule_id": str(parsed),
                        "cap": max_active_schedules_per_agent,
                    }
                },
            )
            return _tool_error(
                "wake_schedule_resume",
                f"you have {max_active_schedules_per_agent} active wakes, the most allowed (pause or delete one first)",
            )
        except Exception as exc:  # noqa: BLE001
            return _tool_error("wake_schedule_resume", f"persist failed: {exc}")
        return f"Resumed [schedule:{parsed}]. Next fires at {next_fire_at.isoformat()}."

    wake_schedule_resume.description = "Resume a paused wake schedule. Its next time is counted from now."
    return [wake_schedule_resume]


# ---------------------------------------------------------------------------
# wake_schedule_delete
# ---------------------------------------------------------------------------


def load_wake_schedule_delete_tool(
    *,
    agent_id: UUID,
    schedules_collection: WakeScheduleCollection,
) -> list[BaseTool]:
    """Build a ``wake_schedule_delete`` tool (hard delete; cascades fires).

    Deletes any of the agent's wakes, whichever conversation it lives in. A
    protected wake is refused with a plain message.

    :param agent_id: caller's agent UUID
    :ptype agent_id: UUID
    :param schedules_collection: three-tier schedules collection
    :ptype schedules_collection: WakeScheduleCollection
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    """

    @tool("wake_schedule_delete", args_schema=ScheduleDeleteInput)
    async def wake_schedule_delete(schedule_id: str) -> str:
        """Delete a wake schedule permanently. Fire history cascades."""
        parsed = parse_schedule_id(schedule_id)
        if parsed is None:
            return _tool_error("wake_schedule_delete", f"invalid schedule_id {schedule_id!r}")

        entity = await schedules_collection.find_for_agent(agent_id, parsed)
        if entity is None:
            return _tool_error("wake_schedule_delete", "schedule not found")
        if entity.protected:
            return _tool_error("wake_schedule_delete", PROTECTED_REFUSAL)

        try:
            await schedules_collection.delete((entity.conversation_id, parsed))
        except Exception as exc:  # noqa: BLE001
            return _tool_error("wake_schedule_delete", f"persist failed: {exc}")
        return f"Deleted [schedule:{parsed}] ({entity.name or 'untitled'})."

    wake_schedule_delete.description = (
        "Delete a wake schedule and its history for good. To keep it for later, use wake_schedule_pause."
    )
    return [wake_schedule_delete]


# ---------------------------------------------------------------------------
# wake_yield (cooperative interrupt -- PLACEMENT §8.5.1)
# ---------------------------------------------------------------------------


class _WakeYieldInput(BaseModel):
    """No-arg input schema for ``wake_yield``."""


def load_wake_yield_tool(
    *,
    is_wake_turn: WakeYieldProbe,
    set_yield_requested: WakeYieldSetter,
) -> list[BaseTool]:
    """Build a ``wake_yield`` tool gated to wake-driven turns only.

    Per PLACEMENT §8.5.1 the tool MUST NOT load on user-driven turns;
    the factory enforces this by checking ``is_wake_turn()`` at
    construction time. The consumer (its personality node) wires
    the probe to read its ``_active_wake_fire_id`` state. When the
    probe returns ``False``, the factory raises so the loaded tool set
    omits the tool entirely -- the LLM never sees an unusable tool.

    When the tool fires on a wake-driven turn, it calls
    ``set_yield_requested()`` so the consumer's tool loop catches the
    flag at the next iteration boundary and exits cleanly. Returns a
    short confirmation string the LLM uses as its final assistant
    message for the turn.

    :param is_wake_turn: closure returning ``True`` when the current
        turn is wake-driven (decided by the consumer's message-source
        signals -- see PLACEMENT §8.5.1)
    :ptype is_wake_turn: WakeYieldProbe
    :param set_yield_requested: closure that flips ``_yield_requested``
        on the consumer's state object
    :ptype set_yield_requested: WakeYieldSetter
    :return: list with one LangChain tool
    :rtype: list[BaseTool]
    :raises RuntimeError: when the probe returns False at load time
        -- the consumer is responsible for ONLY loading this tool on
        wake-driven turns
    """
    if not is_wake_turn():
        raise RuntimeError(
            "load_wake_yield_tool: refusing to load on a non-wake turn; "
            "the consumer must gate this tool to wake-driven turns only",
        )

    @tool("wake_yield", args_schema=_WakeYieldInput)
    async def wake_yield() -> str:
        """Yield this wake so a queued user message processes next."""
        try:
            set_yield_requested()
        except Exception as exc:  # noqa: BLE001 - surface as tool error
            log.warning(
                "wake_yield setter raised",
                extra={"extra_data": {"error": str(exc)}},
            )
            return _tool_error("wake_yield", f"yield setter failed: {exc}")
        log.info("wake_yield fired")
        return "Wake yielded. Your last message ends it, and the person's waiting message is answered next."

    wake_yield.description = (
        "End this wake now so the person's waiting message is answered next. "
        "Use it only when a message from them is waiting."
    )
    return [wake_yield]
