"""Literal types + enum constants for agent-wake.

These mirror the CHECK constraints on the L3 tables: changing the
allowed values here requires a matching migration.

Why ``Literal`` and not ``Enum``: callers (the rest of 3tears, plus
consumers) pass these values through tool input schemas + JSON
boundaries where strings round-trip cleanly. ``Literal`` keeps the
runtime payload a plain ``str`` so JSON encoding doesn't need a custom
serializer and mypy still pins valid value sets at every call site.

Naming follows the agent-skills precedent
(:mod:`threetears.agent.skills.types`).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable
from uuid import UUID

if TYPE_CHECKING:
    from threetears.agent.skills.entities import AgentSkillEntity

__all__ = [
    "ExecutionMode",
    "FireConversationHook",
    "FireLimits",
    "FirePermit",
    "FireSource",
    "FireStatus",
    "HandlerCallback",
    "HandlerCallbackResult",
    "MissedFirePolicy",
    "PreparedWakeContext",
    "ReapedFire",
    "ReapedFiresHook",
    "ScheduleStatus",
    "ScheduleType",
    "VerificationScheme",
    "WakeConversations",
    "WakeDispatchResult",
    "WakeTrigger",
    "WebhookSubscriptionStatus",
]


# ``schedule_type`` column on ``agent_wake_schedules``. CHECK-pinned in
# the L3 schema. Each value maps to a specific ``schedule_config`` JSONB
# shape documented in :mod:`threetears.agent.wake.collections`. The
# ``'cron'`` and ``'relative_delay'`` values were added in the
# 2026-05-19 revision per PLACEMENT §1.5.
ScheduleType = Literal[
    "daily_at",
    "every_n_hours",
    "random_within_window",
    "one_shot_at",
    "cron",
    "relative_delay",
    "interval",
]


# ``status`` column on ``agent_wake_schedules``. ``'paused'`` is set by
# the agent / user via ``WakeScheduleCollection.pause``; ``'expired'``
# is set automatically for one-shot schedules after firing. CHECK-pinned
# in the L3 schema.
ScheduleStatus = Literal["active", "paused", "expired"]


# ``execution_mode`` column on both ``agent_wake_schedules`` and
# ``webhook_subscriptions``. Every fire starts a new conversation, so every
# row written since v007 is ``'spawn'`` and no surface offers a choice.
# ``'inline'`` (the wake turn injected into the wake's own conversation)
# survives only on rows written before v007, until their consumer moves
# them. CHECK-pinned in the L3 schema.
ExecutionMode = Literal["inline", "spawn"]


# ``missed_fire_policy`` column on ``agent_wake_schedules`` (added per
# PLACEMENT §1.7 in the 2026-05-19 revision). ``'coalesce'`` fires
# ONCE for a backlog of missed ticks (recomputed ``next_fire_at``
# forward); ``'catch_up'`` fires once per missed tick. CHECK-pinned in
# the L3 schema.
MissedFirePolicy = Literal["coalesce", "catch_up"]


# ``status`` column on ``wake_fires``. The full enum after the
# wake-yield revision (2026-05-19 evening, PLACEMENT §8.5.1) gained the
# ``'yielded'`` value; v004 added the ``'dispatching'`` placeholder.
# CHECK-pinned in the L3 schema.
#
# - ``'dispatching'`` -- in-flight placeholder written by
#   :meth:`WakeFireCollection.create_dispatching` before the dispatch
#   callback runs; overwritten by ``finalize_success`` /
#   ``finalize_failed`` on the same tick. A row that stays in this
#   status means the dispatcher crashed before finalize (audit
#   evidence the fire was attempted).
# - ``'fired'`` -- wake executed end-to-end, visible assistant response.
# - ``'fired_silent'`` -- wake executed; assistant response started with
#   ``[SILENT]`` and was suppressed (PLACEMENT §1.4).
# - ``'yielded'`` -- agent voluntarily yielded mid-turn via
#   ``wake_yield`` because a user message was waiting (PLACEMENT §8.5.1).
# - ``'skipped_busy'`` -- per-conv lock held; deferred.
# - ``'skipped_rate_limit'`` -- per-conv / per-user cap exceeded.
# - ``'skipped_cap'`` -- per-conv active-schedule cap exceeded
#   (PLACEMENT §1.9).
# - ``'skipped_no_handler'`` -- product did not register a handler
#   callback at dispatch time.
# - ``'skipped_life_off'`` -- the consumer's :class:`FirePermit` answered
#   "not now" (v007). Not a failure: the wake is fine, its agent is not
#   taking fires at the moment.
# - ``'checked_quiet'`` -- the consumer's handler ran a check before any
#   turn and it found nothing (v008). No turn ran; not counted by the fire
#   limits, not a failure, not a ``context_from`` source.
# - ``'failed'`` -- exception raised during dispatch / handler.
FireStatus = Literal[
    "dispatching",
    "fired",
    "fired_silent",
    "yielded",
    "skipped_busy",
    "skipped_rate_limit",
    "skipped_cap",
    "skipped_no_handler",
    "skipped_life_off",
    "checked_quiet",
    "failed",
]


# ``verification_scheme`` column on ``webhook_subscriptions``. v1
# ships ``'generic_hmac_sha256'`` as the platform default; vendor
# schemes (``'github'``, ``'stripe'``, ``'slack_signing'``, ...)
# plug in at runtime via
# :meth:`~threetears.channels.webhook.WebhookReceiver.register_verifier`
# and the schema accepts any slug-shaped value (``^[a-z0-9_]+$``,
# length 1-64; enforced by v005's CHECK constraint).
# ``str`` rather than ``Literal[...]`` because typing every vendor
# scheme up front would defeat the registry's pluggability -- the
# format guard lives at the schema layer + the receiver returns 400
# for any value not in its in-process registry.
VerificationScheme = str


# ``status`` column on ``webhook_subscriptions``. Symmetrical to
# ``ScheduleStatus`` minus the one-shot ``'expired'`` transition --
# webhooks are long-lived. CHECK-pinned in the L3 schema.
WebhookSubscriptionStatus = Literal["active", "paused"]


# Source of a wake fire. ``'scheduled_tick'`` for fires emitted by
# :func:`threetears.agent.wake.tick.wake_tick_job`; ``'webhook'`` for
# inbound HTTP webhook fires (shard 06). Stored verbatim on the
# ``wake_fires`` row so the audit history records the origin without
# requiring a join.
FireSource = Literal["scheduled_tick", "webhook"]


@dataclass(frozen=True)
class WakeTrigger:
    """Immutable envelope describing a single wake fire opportunity.

    Constructed by the tick engine after a schedule is claimed (and by
    the webhook receiver in shard 04 for inbound HTTP fires) and handed
    to the consumer-supplied ``dispatch_callback`` so the handler
    (shard 03) has every field it needs to materialise the fire
    without re-reading the schedule row. Frozen dataclass so the
    callback cannot accidentally mutate fields the platform owns.

    Fields mirror the wake schedule row plus the tick-side decisions
    (``fire_source``, ``fired_at``). The handler decides what to do
    with the fire; the platform decides when it fires.

    ``schedule_id`` is ``None`` for webhook fires (no source schedule
    -- the originator is a :class:`WebhookSubscriptionEntity` row
    keyed off elsewhere); scheduled fires always carry the source
    schedule id. The dispatch flow treats ``None`` as "no chain to
    resolve" and "no per-schedule logging key".

    ``agent_id`` is carried so :func:`dispatch_wake` can resolve the
    attached skill (composite ``(agent_id, skill_id)`` PK on
    ``agent_skills``) without re-fetching the schedule row.

    ``fire_id`` is the ``wake_fires`` row this fire writes, minted before
    the trigger on both the tick and the webhook path. ``protected`` is the
    schedule's flag: a protected wake is exempt from counting against the
    fire caps. ``started_conversation_id`` is the conversation the
    :class:`FireConversationHook` started for this fire; :func:`dispatch_wake`
    sets it before the handler runs, and it is ``None`` when the consumer
    supplies no hook. ``webhook_subscription_id`` is the subscription a
    webhook fire came through, ``None`` on scheduled fires.
    """

    schedule_id: UUID | None
    user_id: UUID
    agent_id: UUID
    conversation_id: UUID
    fire_source: FireSource
    execution_mode: str
    schedule_type: str
    fired_at: datetime
    schedule_name: str | None = None
    task_prompt: str | None = None
    context_from_schedule_id: UUID | None = None
    skill_id: UUID | None = None
    include_conversation_history: bool = True
    fire_id: UUID | None = None
    protected: bool = False
    started_conversation_id: UUID | None = None
    webhook_subscription_id: UUID | None = None


@dataclass(frozen=True)
class FireLimits:
    """How many fires an agent may make in the trailing day.

    Returned by the consumer's :class:`FirePermit` for each fire, so a
    change to the agent's settings takes effect on its next fire. Fires
    that ran count, silent ones included; skipped and failed ones do not.

    :ivar per_wake: fires one wake (or one webhook subscription) may make
    :ivar per_agent: fires all the agent's wakes and subscriptions may make
        together; protected wakes' fires are not counted
    """

    per_wake: int
    per_agent: int


@runtime_checkable
class FirePermit(Protocol):
    """Consumer callback that decides, per fire, whether and how much the agent may fire.

    Called for every fire before anything else runs, protected wakes
    included. Return the agent's :class:`FireLimits` to let the fire go on
    to the cap check, or ``None`` for "not now": the fire is recorded as
    ``'skipped_life_off'``, nothing else runs, and it is not a failure.
    A protected wake is not counted against the limits, but "not now"
    still applies to it.
    """

    async def __call__(self, trigger: WakeTrigger) -> FireLimits | None:
        """Decide for one fire.

        :param trigger: the fire
        :ptype trigger: WakeTrigger
        :return: the agent's limits, or ``None`` for "not now"
        :rtype: FireLimits | None
        """
        ...


@runtime_checkable
class FireConversationHook(Protocol):
    """Consumer callback that starts the conversation a fire runs in.

    Called by :func:`dispatch_wake` after the fire is permitted and before
    the handler runs, inside a transaction on ``conn``. It creates the
    conversation on that connection and returns its id; the platform then
    links the fire row to it in the same transaction, so the two commit
    together. The handler receives the id as
    ``trigger.started_conversation_id``. The transaction is a
    :class:`~threetears.core.collections.CallerTransaction`, so the hook may
    save through a collection with ``save_entity(entity, conn=conn)``.
    """

    async def __call__(self, trigger: WakeTrigger, conn: Any) -> UUID:
        """Create the fire's conversation.

        :param trigger: the fire
        :ptype trigger: WakeTrigger
        :param conn: the transaction's connection
        :ptype conn: Any
        :return: the new conversation's id
        :rtype: UUID
        """
        ...


@runtime_checkable
class WakeConversations(Protocol):
    """Consumer hooks for the conversations wakes live in.

    A wake lives in a wake conversation. Creating a wake either names one
    of the agent's existing wake conversations or has a new one made, whose
    parent is the conversation the wake was created from. Several wakes may
    share one wake conversation.
    """

    async def create(
        self,
        *,
        parent_conversation_id: UUID,
        user_id: UUID,
        agent_id: UUID,
        name: str | None,
        conn: Any,
    ) -> UUID:
        """Create a wake conversation on ``conn`` and return its id.

        Runs inside the transaction that inserts the wake, so a refused
        insert leaves no empty conversation behind. That transaction is a
        :class:`~threetears.core.collections.CallerTransaction`, so the hook
        may save through a collection with ``save_entity(entity, conn=conn)``.

        :param parent_conversation_id: the conversation the wake was created from
        :ptype parent_conversation_id: UUID
        :param user_id: the agent's owner
        :ptype user_id: UUID
        :param agent_id: the agent
        :ptype agent_id: UUID
        :param name: the wake's name, if it has one
        :ptype name: str | None
        :param conn: the transaction's connection
        :ptype conn: Any
        :return: the new wake conversation's id
        :rtype: UUID
        """
        ...

    async def is_wake_conversation(self, *, agent_id: UUID, conversation_id: UUID) -> bool:
        """Whether ``conversation_id`` is one of ``agent_id``'s wake conversations.

        :param agent_id: the agent
        :ptype agent_id: UUID
        :param conversation_id: the conversation the caller named
        :ptype conversation_id: UUID
        :return: ``True`` when a wake may be created into it
        :rtype: bool
        """
        ...


@dataclass(frozen=True)
class ReapedFire:
    """A fire the reaper failed because its dispatcher was lost.

    :ivar conversation_id: the wake's conversation (the fire row's partition)
    :ivar fire_id: the fire
    :ivar started_conversation_id: the conversation the fire had started, or
        ``None`` when it was lost before starting one
    """

    conversation_id: UUID
    fire_id: UUID
    started_conversation_id: UUID | None


# Consumer callback the tick calls with the fires each reaper sweep failed,
# so it can close whatever it opened for them (the conversation a fire
# started, the entry that says the fire is running).
ReapedFiresHook = Callable[[Sequence[ReapedFire]], Awaitable[None]]


@dataclass(frozen=True)
class WakeDispatchResult:
    """Return value from a dispatch callback to the tick engine.

    The platform uses :attr:`status` to write the terminal fire row;
    :attr:`output_text` / :attr:`latency_ms` / :attr:`error` are
    optional fixups the dispatcher may capture. Shard 03 owns the
    real producer; shard 02 only types the callable so the tick body
    can be exercised end-to-end with a stub callback.
    """

    status: FireStatus
    output_text: str | None = None
    latency_ms: int | None = None
    error: str | None = None
    display_suppressed: bool = False


@dataclass(frozen=True)
class PreparedWakeContext:
    """Read-only payload :func:`dispatch_wake` hands to the consumer's handler.

    The platform owns the boundary that resolves "what does this wake
    need at fire time": the attached skill row (or ``None``), the
    optional ``context_from`` chain block, and the trigger envelope
    itself. The handler reads these fields and assembles its product-
    specific prompt + LLM invocation. Frozen so the handler cannot
    mutate fields the platform owns.

    Per PLACEMENT §1.2 (2026-05-19 revision) there is NO pre-check
    output here: the pre-check executor framework was dropped in favor
    of ``tool_eligible=False, skill_eligible=True`` ordinary tools
    surfaced via the attached skill's ``tool_additions``. If a
    diagnostic produces "context for the wake", it now does so via the
    skill body + tool calls, not via a parallel pre-check stage.

    :ivar trigger: the wake trigger envelope
    :ivar attached_skill: resolved skill row (``None`` when
        ``trigger.skill_id`` is ``None``, when the skill row was
        deleted, or when the skill is disabled)
    :ivar context_blocks: zero-or-more labeled context blocks the
        handler may inject into its prompt. v1 source is the
        ``context_from_schedule_id`` upstream-fire output (PLACEMENT
        §1.6, single-hop). Empty tuple when nothing applies.
    """

    trigger: WakeTrigger
    attached_skill: AgentSkillEntity | None
    context_blocks: tuple[str, ...]


@dataclass(frozen=True)
class HandlerCallbackResult:
    """Return value from the consumer's :class:`HandlerCallback`.

    The handler reports the outcome of the wake-driven turn: where the
    assistant message landed (``target_conversation_id`` differs from
    ``trigger.conversation_id`` only in spawn mode -- not yet wired in
    v1, but typed for forward compatibility), the assistant text the
    handler placed in its messages table, and a structural status the
    platform writes onto ``wake_fires``.

    ``status`` semantics:

    - ``'fired'`` -- the handler produced an assistant message visible
      to the user.
    - ``'fired_silent'`` -- the handler produced an assistant message
      whose text started with ``[SILENT]``; the consumer hid it from
      its UI (PLACEMENT §1.4). The platform sets
      ``display_suppressed=True`` on the ``wake_fires`` row regardless
      of which status the handler returns, based on prefix detection.
    - ``'yielded'`` -- the handler exited mid-turn via ``wake_yield``
      because a user message was queued (PLACEMENT §8.5.1).
    - ``'failed'`` -- the handler captured a non-exceptional failure
      (downstream rate-limit, no eligible model, etc.) into
      :attr:`error` rather than raising.
    - ``'skipped_busy'`` -- the handler declined because the conv was
      busy on a parallel turn. Rare; most busy detection lives in the
      platform's per-conv lock acquire upstream.
    - ``'checked_quiet'`` -- the handler ran a check before any turn and
      it found nothing to wake the agent for. No turn ran, and the fire
      limits do not count it.

    The platform inspects :attr:`assistant_message_content` for the
    ``[SILENT]`` prefix and records ``display_suppressed`` on the fire
    row regardless of what :attr:`status` the handler reports. The
    handler is responsible for stripping the prefix from its persisted
    message + setting whatever its UI uses to render hidden (e.g.
    a ``messages.display='hidden'`` column).
    """

    status: FireStatus
    assistant_message_content: str
    target_conversation_id: UUID
    assistant_message_id: UUID | None = None
    latency_ms: int | None = None
    error: str | None = None


@runtime_checkable
class HandlerCallback(Protocol):
    """Consumer-supplied callback invoked by :func:`dispatch_wake`.

    The platform owns "when does the wake fire" + "what skill is
    attached" + "what context blocks accompany the fire". The handler
    owns "how does my product turn that into an assistant message"
    -- prompt assembly, LLM invocation, message persistence.

    A typical implementation builds the system prompt around the
    attached skill's body, injects the wake notice + context blocks
    into the conversation, calls the personality node, and persists
    the assistant response. A future 3tears implementation might
    bypass LangGraph entirely. Same Protocol shape, different body.

    Implementations MUST be async. They SHOULD return a
    :class:`HandlerCallbackResult` -- raising propagates through
    :func:`dispatch_wake` and the platform records the exception on
    the fire row.

    ``pool`` is forwarded so the handler can read its own product
    state (messages table, conversation row) using the same
    connection pool the platform uses; consumers can ignore it if
    they keep their own pool.
    """

    async def __call__(
        self,
        trigger: WakeTrigger,
        prepared_context: PreparedWakeContext,
        pool: Any,
    ) -> HandlerCallbackResult:
        """Handle one wake fire and return the outcome.

        :param trigger: immutable fire envelope
        :ptype trigger: WakeTrigger
        :param prepared_context: platform-resolved skill + context
            blocks
        :ptype prepared_context: PreparedWakeContext
        :param pool: asyncpg-compatible connection pool
        :ptype pool: Any
        :return: outcome of the wake-driven turn
        :rtype: HandlerCallbackResult
        """
        ...
