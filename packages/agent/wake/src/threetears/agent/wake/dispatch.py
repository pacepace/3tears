"""Wake dispatch -- the convergence point that turns a fire into a result.

:func:`dispatch_wake` is the single entry point both the tick engine
(shard 02, scheduled fires) and the webhook receiver (shard 06,
inbound HTTP fires) feed. The platform owns the boundary; the
consumer plugs in:

- a :class:`HandlerCallback` (the product's prompt-assembly + LLM
  invocation + message persistence body)
- a :class:`DeliveryAdapter` per non-``'conversation'`` delivery
  target (e.g. metallm's email adapter)

Per PLACEMENT.md (2026-05-19 revision, §1.2) this shard does NOT
contain a pre-check executor framework. Diagnostic tools that used
to be modelled as pre-checks are now ordinary
``tool_eligible=False, skill_eligible=True`` ``TearsTool`` subclasses
surfaced via the attached skill's ``tool_additions`` and dispatched
through the handler's regular tool loop. There is no
``pre_check_output`` field on :class:`PreparedWakeContext`; there is
no parallel executor framework here.

Flow (revised per PLACEMENT §1.6 / §1.4 / §1.3):

1. Resolve ``context_from`` chain (single hop) -> ``context_blocks``.
2. Resolve attached skill (single ``(agent_id, skill_id)`` lookup) ->
   ``attached_skill`` (or ``None``).
3. Build :class:`PreparedWakeContext`.
4. Invoke the consumer's :class:`HandlerCallback`.
5. Determine silent treatment: ``[SILENT]`` marker on the assistant
   text OR an explicit handler ``status='fired_silent'`` is enough
   (either signal alone is authoritative). When silent, flip
   ``display_suppressed=True`` on the dispatch result.
6. Route delivery: for ``'conversation'`` target the handler already
   placed the message -- no-op. For other targets, invoke the
   matching :class:`DeliveryAdapter`. Skip delivery on silent fires
   -- silent fires have no payload to relay; the per-target
   ``'skipped_silent'`` lands on
   :attr:`WakeDispatchResult.delivery_status`.
7. Build + return :class:`WakeDispatchResult`.

The per-conv NATS lock is acquired by the CALLER (tick / webhook
receiver), not here -- the lock spans claim + dispatch so the row
write and the handler invocation share one mutual-exclusion window.
The fire row UPDATE site is the tick body's
``finalize_success`` / ``finalize_failed`` calls; ``dispatch_wake``
returns a typed result and the tick writes the terminal status.
"""

from __future__ import annotations

import re
import time
from typing import Any, Final
from uuid import UUID

from threetears.observe import get_logger

from threetears.agent.wake.config import DEFAULT_WAKE_CONFIG, WakeConfig
from threetears.agent.wake.events import (
    EVENT_DELIVERY_ATTEMPT,
    EVENT_DELIVERY_FAILED,
    EVENT_DELIVERY_SKIPPED_SILENT,
    EVENT_DELIVERY_SUCCESS,
    EVENT_FIRE_RATE_LIMITED,
    EVENT_FIRE_SILENT,
)
from threetears.agent.wake.metrics import get_wake_emitter
from threetears.agent.wake.rate_limit import _check_rate_limit
from threetears.agent.wake.types import (
    DeliveryAdapter,
    FireStatus,
    HandlerCallback,
    HandlerCallbackResult,
    PreparedWakeContext,
    WakeDispatchResult,
    WakeTrigger,
)

__all__ = [
    "detect_silent_prefix",
    "dispatch_wake",
]


log = get_logger(__name__)


# Compiled once. ``[SILENT]`` prefix with optional leading whitespace +
# optional trailing whitespace / newline. Case-insensitive per
# PLACEMENT §1.4 + the spec body's regex example.
_SILENT_PREFIX: Final[re.Pattern[str]] = re.compile(
    r"^\s*\[SILENT\]\s*",
    re.IGNORECASE,
)


# Combined cap for ``context_from`` context block bytes per PLACEMENT
# §1.6 + the spec body's "context_blocks budget" note. The pre-check
# share was dropped along with the executor framework, so the entire
# 16KB budget belongs to ``context_from``.
_CONTEXT_BLOCK_BUDGET_BYTES: Final[int] = 16 * 1024


def detect_silent_prefix(content: str) -> bool:
    """Return ``True`` when ``content`` starts with the ``[SILENT]`` marker.

    Match is case-insensitive and tolerates leading whitespace /
    trailing whitespace + newline. Compiled once at module load so
    per-call cost is a regex match, not a recompilation.

    Per PLACEMENT §1.4 the platform records ``display_suppressed`` on
    the fire row when this returns ``True``; the consumer's handler
    is responsible for stripping the marker from the persisted message
    + setting whatever UI column it uses for hidden rows (e.g.
    metallm's ``messages.display='hidden'``).

    :param content: candidate assistant-message text
    :ptype content: str
    :return: ``True`` when the marker prefix matches
    :rtype: bool
    """
    if not content:
        return False
    return _SILENT_PREFIX.match(content) is not None


async def dispatch_wake(
    trigger: WakeTrigger,
    fire_id: UUID,
    pool: Any,
    *,
    handler: HandlerCallback,
    delivery_adapters: dict[str, DeliveryAdapter] | None = None,
    wake_config: WakeConfig = DEFAULT_WAKE_CONFIG,
) -> WakeDispatchResult:
    """Drive one wake fire end-to-end and return the typed result.

    The caller (tick body in shard 02; webhook receiver in shard 06)
    is responsible for:

    - acquiring the per-conv ``conv.<id>.llm_active`` lock BEFORE
      calling this function and holding it across the call
    - INSERTing the initial ``wake_fires`` row with
      ``create_dispatching`` before invoking
    - UPDATEing the terminal row state from this function's return
      value via ``finalize_success`` / ``finalize_failed``

    This function:

    1. Runs the rate-limit check via
       :func:`threetears.agent.wake.rate_limit._check_rate_limit`.
       When either the per-conv or per-user cap is exceeded, emits
       :data:`EVENT_FIRE_RATE_LIMITED`, increments the matching
       Prometheus rejection counter, and returns
       ``WakeDispatchResult(status='skipped_rate_limit', ...)``
       without invoking the handler or any delivery adapters.
    2. Resolves the ``context_from`` chain (single hop) into
       :attr:`PreparedWakeContext.context_blocks`.
    3. Resolves the attached skill from
       ``(trigger.agent_id, trigger.skill_id)``. Missing or disabled
       skills resolve to ``None`` with a warning log; the handler
       sees ``attached_skill=None`` and decides how to proceed.
    4. Invokes the consumer's :class:`HandlerCallback`. Exceptions
       propagate to the caller (the tick records them via the
       ``try / except`` in :func:`threetears.agent.wake.tick._dispatch_one`).
    5. Determines silent treatment from the handler's outcome:
       ``is_silent`` is true when EITHER the ``[SILENT]`` marker
       (PLACEMENT §1.4) is detected OR the handler explicitly
       returned ``status='fired_silent'``. Either signal alone is
       authoritative -- when the handler self-reports silent we
       honor it even if the marker is absent. When ``is_silent`` is
       true, ``display_suppressed`` flips and delivery routing is
       skipped (recorded as ``'skipped_silent'`` on
       :attr:`WakeDispatchResult.delivery_status` for shard-05
       metrics).
    6. Routes delivery: ``'conversation'`` is a no-op (handler
       already wrote the message); other targets invoke the matching
       :class:`DeliveryAdapter` from ``delivery_adapters``. Delivery
       failure is logged but does not fail the fire -- the message
       already landed in the conversation, so a failed email side-
       channel does not invalidate the wake. The per-target outcome
       lands on :attr:`WakeDispatchResult.delivery_status`.

    :param trigger: immutable fire envelope from the caller
    :ptype trigger: WakeTrigger
    :param fire_id: the pre-INSERTed ``wake_fires`` row id (forwarded
        to logging so per-fire traces correlate)
    :ptype fire_id: UUID
    :param pool: asyncpg-compatible connection pool the platform +
        handler share
    :ptype pool: Any
    :param handler: consumer-supplied :class:`HandlerCallback`
    :ptype handler: HandlerCallback
    :param delivery_adapters: optional mapping of non-conversation
        delivery targets to their adapters; ``None`` is treated as an
        empty dict (only ``'conversation'`` is routable)
    :ptype delivery_adapters: dict[str, DeliveryAdapter] | None
    :param wake_config: consumer's :class:`WakeConfig` impl supplying
        rate-limit caps; defaults to :data:`DEFAULT_WAKE_CONFIG` so
        the platform invariants are enforced even when the consumer
        forgets to plumb a config
    :ptype wake_config: WakeConfig
    :return: typed dispatch result the caller writes onto
        ``wake_fires``
    :rtype: WakeDispatchResult
    """
    started = time.monotonic()
    adapters = delivery_adapters or {}
    emitter = get_wake_emitter()

    # Step 1 (per OBS-13 / PLACEMENT §1.9): rate-limit check BEFORE
    # any handler work runs. Rejection short-circuits to a
    # ``skipped_rate_limit`` terminal result; the caller writes it
    # onto the ``wake_fires`` row via the usual finalize path. With
    # ``pool=None`` (unit tests without a DB) the helper returns
    # ``None`` so existing handler-flow tests are unchanged.
    rate_limit_scope = await _check_rate_limit(trigger, pool, wake_config)
    if rate_limit_scope is not None:
        cap = (
            wake_config.max_fires_per_conv_per_day
            if rate_limit_scope == "conv"
            else wake_config.max_fires_per_user_per_day
        )
        log.info(
            EVENT_FIRE_RATE_LIMITED,
            extra={
                "extra_data": {
                    "fire_id": str(fire_id),
                    "schedule_id": str(trigger.schedule_id) if trigger.schedule_id else None,
                    "conversation_id": str(trigger.conversation_id),
                    "user_id": str(trigger.user_id),
                    "fire_source": trigger.fire_source,
                    "scope": rate_limit_scope,
                    "cap": cap,
                }
            },
        )
        emitter.inc_rate_limit_rejection(scope=rate_limit_scope)
        emitter.inc_failure(reason="rate_limited")
        return WakeDispatchResult(
            status="skipped_rate_limit",
            error=f"rate limit exceeded ({rate_limit_scope}); cap={cap}",
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    context_blocks = await _resolve_context_from(pool, trigger)
    attached_skill = await _resolve_attached_skill(pool, trigger)
    prepared = PreparedWakeContext(
        trigger=trigger,
        attached_skill=attached_skill,
        context_blocks=context_blocks,
    )

    log.info(
        "dispatch_wake: invoking handler",
        extra={
            "extra_data": {
                "fire_id": str(fire_id),
                "schedule_id": str(trigger.schedule_id),
                "conversation_id": str(trigger.conversation_id),
                "agent_id": str(trigger.agent_id),
                "attached_skill_id": str(attached_skill.skill_id) if attached_skill else None,
                "context_blocks": len(context_blocks),
                "fire_source": trigger.fire_source,
            }
        },
    )

    handler_result = await handler(trigger, prepared, pool)

    latency_ms = handler_result.latency_ms
    if latency_ms is None:
        latency_ms = int((time.monotonic() - started) * 1000)

    marker_detected = detect_silent_prefix(handler_result.assistant_message_content)
    # The handler's explicit ``status='fired_silent'`` is authoritative
    # (per the coherence-asymmetry resolution in
    # `.prawduct/critic-review.md`): when the handler self-reports
    # silent we honor it even if the marker is absent. Otherwise the
    # marker is the canonical signal (PLACEMENT §1.4) and promotes
    # ``'fired'`` to ``'fired_silent'``. The resulting ``is_silent``
    # flag drives BOTH ``display_suppressed`` AND the delivery-skip
    # branch, so the audit row + the suppression flag stay coherent.
    is_silent = marker_detected or handler_result.status == "fired_silent"
    final_status: FireStatus
    if is_silent and handler_result.status == "fired":
        final_status = "fired_silent"
    else:
        final_status = handler_result.status

    delivery_status: dict[str, str] = {}
    if handler_result.status in {"fired", "fired_silent"} and trigger.delivery_target != "conversation":
        if is_silent:
            # Silent fires intentionally skip delivery -- the marker
            # path means the agent decided there's nothing to relay.
            # Record the skip on the result so shard-05 metrics can
            # distinguish "no adapter" from "agent chose silence".
            delivery_status[trigger.delivery_target] = "skipped_silent"
            log.info(
                EVENT_DELIVERY_SKIPPED_SILENT,
                extra={
                    "extra_data": {
                        "fire_id": str(fire_id),
                        "schedule_id": str(trigger.schedule_id) if trigger.schedule_id else None,
                        "conversation_id": str(trigger.conversation_id),
                        "delivery_target": trigger.delivery_target,
                    }
                },
            )
            emitter.inc_delivery(target=trigger.delivery_target, status="skipped_silent")
        else:
            log.info(
                EVENT_DELIVERY_ATTEMPT,
                extra={
                    "extra_data": {
                        "fire_id": str(fire_id),
                        "schedule_id": str(trigger.schedule_id) if trigger.schedule_id else None,
                        "conversation_id": str(trigger.conversation_id),
                        "delivery_target": trigger.delivery_target,
                    }
                },
            )
            delivery_outcome = await _route_delivery(
                trigger=trigger,
                prepared=prepared,
                handler_result=handler_result,
                adapters=adapters,
                pool=pool,
                fire_id=fire_id,
            )
            delivery_status[trigger.delivery_target] = delivery_outcome
            emitter.inc_delivery(target=trigger.delivery_target, status=delivery_outcome)
            event_name = EVENT_DELIVERY_SUCCESS if delivery_outcome == "delivered" else EVENT_DELIVERY_FAILED
            log.info(
                event_name,
                extra={
                    "extra_data": {
                        "fire_id": str(fire_id),
                        "schedule_id": str(trigger.schedule_id) if trigger.schedule_id else None,
                        "conversation_id": str(trigger.conversation_id),
                        "delivery_target": trigger.delivery_target,
                        "delivery_status": delivery_outcome,
                    }
                },
            )

    if is_silent:
        log.info(
            EVENT_FIRE_SILENT,
            extra={
                "extra_data": {
                    "fire_id": str(fire_id),
                    "schedule_id": str(trigger.schedule_id) if trigger.schedule_id else None,
                    "conversation_id": str(trigger.conversation_id),
                    "schedule_type": trigger.schedule_type,
                    "execution_mode": trigger.execution_mode,
                    "fire_source": trigger.fire_source,
                }
            },
        )

    log.info(
        "dispatch_wake: handler complete",
        extra={
            "extra_data": {
                "fire_id": str(fire_id),
                "schedule_id": str(trigger.schedule_id),
                "status": final_status,
                "display_suppressed": is_silent,
                "latency_ms": latency_ms,
                "delivery_target": trigger.delivery_target,
                "delivery_status": delivery_status,
            }
        },
    )

    return WakeDispatchResult(
        status=final_status,
        output_text=handler_result.assistant_message_content,
        latency_ms=latency_ms,
        error=handler_result.error,
        display_suppressed=is_silent,
        delivery_status=delivery_status,
    )


async def _resolve_context_from(
    pool: Any,
    trigger: WakeTrigger,
) -> tuple[str, ...]:
    """Resolve the ``context_from`` chain into a tuple of labeled blocks.

    Per PLACEMENT §1.6 the chain is **single-hop, same-conversation
    only** -- if schedule A's ``context_from = B`` and B's
    ``context_from = C``, A receives B's most recent successful fire
    output, NEVER C's. Cycle detection lives at the agent-tools layer
    (shard 04) where wakes are authored; this resolver simply reads
    the upstream row.

    Returns an empty tuple when:

    - ``trigger.context_from_schedule_id`` is ``None``
    - the upstream schedule has no successful fires yet (logged at
      WARNING for diagnosability)
    - the upstream output_text is empty

    Combined output is capped at :data:`_CONTEXT_BLOCK_BUDGET_BYTES`
    (16 KB) per PLACEMENT §1.6. If a single block would exceed the
    budget it is truncated with a ``[truncated: <total>B -> 16KB]``
    suffix so downstream prompts can stay within the consumer's
    context window.

    :param pool: asyncpg pool the upstream-fire fetch uses
    :ptype pool: Any
    :param trigger: fire envelope (carries the conversation + upstream
        schedule reference)
    :ptype trigger: WakeTrigger
    :return: tuple of zero or more labeled context blocks
    :rtype: tuple[str, ...]
    """
    upstream_id = trigger.context_from_schedule_id
    if upstream_id is None:
        return ()
    if pool is None:
        return ()
    # local imports keep the dispatch module's import cost identical
    # to the no-context-from path -- the registry / config / collection
    # plumbing only loads when a wake actually uses the feature.
    from threetears.agent.wake.collections import (  # noqa: PLC0415
        WakeFireCollection,
        WakeScheduleCollection,
    )
    from threetears.core.collections.registry import CollectionRegistry  # noqa: PLC0415
    from threetears.core.config import DefaultCoreConfig  # noqa: PLC0415

    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")

    fires = WakeFireCollection(registry=registry, config=cfg)
    schedules = WakeScheduleCollection(registry=registry, config=cfg)
    upstream_fire = await fires.latest_for_schedule(
        conversation_id=trigger.conversation_id,
        schedule_id=upstream_id,
    )
    if upstream_fire is None or upstream_fire.status not in {"fired", "fired_silent"}:
        log.warning(
            "dispatch_wake: context_from upstream has no successful fire",
            extra={
                "extra_data": {
                    "downstream_schedule_id": str(trigger.schedule_id),
                    "upstream_schedule_id": str(upstream_id),
                    "conversation_id": str(trigger.conversation_id),
                }
            },
        )
        return ()
    payload = upstream_fire.output_text or ""
    if not payload.strip():
        return ()

    upstream_name: str | None = None
    upstream_schedule = await schedules.get(
        (trigger.conversation_id, upstream_id),
    )
    if upstream_schedule is not None:
        upstream_name = upstream_schedule.name

    label = upstream_name or f"schedule {upstream_id}"
    fired_at_iso = upstream_fire.actual_fired_at.isoformat()
    block = f'Context from upstream schedule "{label}" (fired {fired_at_iso}):\n{payload}\n---'
    encoded = block.encode("utf-8")
    if len(encoded) > _CONTEXT_BLOCK_BUDGET_BYTES:
        truncated = encoded[:_CONTEXT_BLOCK_BUDGET_BYTES].decode("utf-8", errors="ignore")
        suffix = f"\n[truncated: {len(encoded)}B -> {_CONTEXT_BLOCK_BUDGET_BYTES}B]"
        block = truncated + suffix
        log.warning(
            "dispatch_wake: context_from block truncated",
            extra={
                "extra_data": {
                    "downstream_schedule_id": str(trigger.schedule_id),
                    "upstream_schedule_id": str(upstream_id),
                    "original_bytes": len(encoded),
                    "budget_bytes": _CONTEXT_BLOCK_BUDGET_BYTES,
                }
            },
        )
    return (block,)


async def _resolve_attached_skill(
    pool: Any,
    trigger: WakeTrigger,
) -> Any:
    """Resolve ``trigger.skill_id`` to an ``AgentSkillEntity`` (or ``None``).

    Per PLACEMENT §1.3 a wake has at most ONE attached skill. The
    composite ``(agent_id, skill_id)`` PK on ``agent_skills`` requires
    both values; ``trigger.agent_id`` carries the partition key. The
    function returns ``None`` (with a warning log) when:

    - ``trigger.skill_id`` is ``None`` (the schedule has no attached
      skill -- normal case)
    - the skill row was deleted (``ON DELETE SET NULL`` on
      ``agent_wake_schedules.skill_id`` should keep this rare, but
      defence in depth)
    - the skill is disabled (``enabled=false``)

    The handler receives ``attached_skill=None`` and decides how to
    proceed -- typically by falling back to a default system prompt.

    :param pool: asyncpg pool the skills collection reads from
    :ptype pool: Any
    :param trigger: fire envelope (carries the agent_id + skill_id)
    :ptype trigger: WakeTrigger
    :return: resolved skill entity or ``None``
    :rtype: AgentSkillEntity | None
    """
    if trigger.skill_id is None:
        return None
    if pool is None:
        return None
    # local imports keep the no-skill path's import cost identical to
    # the skill path's -- the skills collection only loads when a
    # wake actually has a skill attached.
    from threetears.agent.skills.collections import AgentSkillCollection  # noqa: PLC0415
    from threetears.core.collections.registry import CollectionRegistry  # noqa: PLC0415
    from threetears.core.config import DefaultCoreConfig  # noqa: PLC0415

    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    skills = AgentSkillCollection(registry=registry, config=cfg)
    entity = await skills.get((trigger.agent_id, trigger.skill_id))
    if entity is None:
        log.warning(
            "dispatch_wake: attached skill missing -- handler will see attached_skill=None",
            extra={
                "extra_data": {
                    "schedule_id": str(trigger.schedule_id),
                    "agent_id": str(trigger.agent_id),
                    "skill_id": str(trigger.skill_id),
                }
            },
        )
        return None
    if not entity.enabled:
        log.warning(
            "dispatch_wake: attached skill disabled -- handler will see attached_skill=None",
            extra={
                "extra_data": {
                    "schedule_id": str(trigger.schedule_id),
                    "agent_id": str(trigger.agent_id),
                    "skill_id": str(trigger.skill_id),
                }
            },
        )
        return None
    return entity


async def _route_delivery(
    *,
    trigger: WakeTrigger,
    prepared: PreparedWakeContext,
    handler_result: HandlerCallbackResult,
    adapters: dict[str, DeliveryAdapter],
    pool: Any,
    fire_id: UUID,
) -> str:
    """Invoke the matching :class:`DeliveryAdapter` for the trigger's target.

    ``trigger.delivery_target == 'conversation'`` is handled by the
    caller (the handler already placed the message); this function is
    only invoked for non-conversation targets.

    Failure of the adapter is logged but does NOT raise -- the
    assistant message already landed in the conversation, so an
    email-side-channel failure does not invalidate the wake. The
    returned status string is the value the caller stores in
    :attr:`WakeDispatchResult.delivery_status` under the target key --
    shard-05's Prometheus emit reads these strings off and increments
    the matching counter. Values: ``'no_adapter'`` (no adapter
    registered for this target), ``'failed'`` (adapter raised), or
    whatever short string the adapter itself returned (convention:
    ``'delivered'`` for success). The target name is NOT repeated in
    the status string because the caller already keys the dict by
    target.

    :param trigger: fire envelope (target lookup key)
    :ptype trigger: WakeTrigger
    :param prepared: prepared context forwarded to the adapter
    :ptype prepared: PreparedWakeContext
    :param handler_result: handler outcome to relay
    :ptype handler_result: HandlerCallbackResult
    :param adapters: registry of delivery target -> adapter
    :ptype adapters: dict[str, DeliveryAdapter]
    :param pool: asyncpg pool the adapter may use
    :ptype pool: Any
    :param fire_id: forwarded to logging for correlation
    :ptype fire_id: UUID
    :return: short status string for the delivery_status dict value
    :rtype: str
    """
    target = trigger.delivery_target
    adapter = adapters.get(target)
    if adapter is None:
        log.warning(
            "dispatch_wake: no DeliveryAdapter registered for target",
            extra={
                "extra_data": {
                    "fire_id": str(fire_id),
                    "schedule_id": str(trigger.schedule_id),
                    "delivery_target": target,
                    "registered_targets": sorted(adapters.keys()),
                }
            },
        )
        return "no_adapter"
    try:
        return await adapter.deliver(trigger, prepared, handler_result, pool)
    except Exception as exc:  # noqa: BLE001 - boundary: delivery failure must not raise out of dispatch
        log.exception(
            "dispatch_wake: delivery adapter raised",
            extra={
                "extra_data": {
                    "fire_id": str(fire_id),
                    "schedule_id": str(trigger.schedule_id),
                    "delivery_target": target,
                    "error_type": type(exc).__name__,
                }
            },
        )
        return "failed"
