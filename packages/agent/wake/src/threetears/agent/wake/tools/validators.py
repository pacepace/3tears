"""Pure validators for wake schedule config + context_from chains.

Module-level symbols importable by the product's REST surface so the
exact same validation runs at both the agent-tool boundary and the
HTTP boundary (Requirement TOOL-11 in shard-04 spec). Validators
return ``str | None`` -- a human-readable error or ``None`` on success.

Per PLACEMENT §1.6 the ``context_from`` chain is single-hop +
same-conversation, so cycle detection walks the chain BFS-style up to
:data:`CONTEXT_FROM_MAX_DEPTH` hops and rejects revisits.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Final
from uuid import UUID

__all__ = [
    "CONTEXT_FROM_MAX_DEPTH",
    "SUPPORTED_SCHEDULE_TYPES",
    "validate_context_from_chain",
    "validate_schedule_config",
]


# Max hops the context_from cycle-walker visits before refusing.
# PLACEMENT §1.6 documents the chain as single-hop, but the walker
# tolerates a small chain for defence-in-depth: if a future revision
# allows multi-hop chains we still bound the cost. Eight hops is the
# spec's documented cap.
CONTEXT_FROM_MAX_DEPTH: Final[int] = 8


# Schedule types the validator recognises. Synchronised with
# :data:`threetears.agent.wake.types.ScheduleType`.
SUPPORTED_SCHEDULE_TYPES: Final[frozenset[str]] = frozenset(
    {
        "daily_at",
        "every_n_hours",
        "random_within_window",
        "one_shot_at",
        "cron",
        "relative_delay",
        "interval",
    },
)


def validate_schedule_config(
    schedule_type: str,
    schedule_config: dict[str, Any],
    *,
    now: datetime | None = None,
) -> str | None:
    """Return an error message when ``schedule_config`` is malformed.

    Pure function. Each ``schedule_type`` branch checks the
    config-shape contract documented in
    :mod:`threetears.agent.wake.collections` and rejects with a
    field-level error.

    ``one_shot_at`` enforces the past-time guard (PLACEMENT §3.5
    "one-shots cannot be resurrected"): ``fire_at_iso <= now`` is
    rejected so the schedule can't be created already-past.

    :param schedule_type: one of :data:`SUPPORTED_SCHEDULE_TYPES`
    :ptype schedule_type: str
    :param schedule_config: per-type JSON config payload
    :ptype schedule_config: dict[str, Any]
    :param now: reference instant for past-time guards (defaults to
        ``datetime.now(UTC)``)
    :ptype now: datetime | None
    :return: error string or ``None`` on success
    :rtype: str | None
    """
    if not isinstance(schedule_config, dict):
        return "schedule_config must be a JSON object"

    if schedule_type not in SUPPORTED_SCHEDULE_TYPES:
        allowed = ", ".join(sorted(SUPPORTED_SCHEDULE_TYPES))
        return f"unknown schedule_type {schedule_type!r}; allowed: {allowed}"

    if schedule_type == "daily_at":
        return _validate_daily_at(schedule_config)
    if schedule_type == "every_n_hours":
        return _validate_every_n_hours(schedule_config)
    if schedule_type == "random_within_window":
        return _validate_random_within_window(schedule_config)
    if schedule_type == "one_shot_at":
        return _validate_one_shot_at(schedule_config, now=now)
    if schedule_type == "cron":
        return _validate_cron(schedule_config)
    if schedule_type == "relative_delay":
        return _validate_relative_delay(schedule_config)
    if schedule_type == "interval":
        return _validate_interval(schedule_config)
    # unreachable -- the membership check above guards this
    return f"schedule_type {schedule_type!r} validator missing"


def _validate_daily_at(config: dict[str, Any]) -> str | None:
    """Validate ``daily_at`` config: ``{hour, minute, tz}``."""
    if "hour" not in config:
        return "daily_at requires 'hour' (0-23); got: %s" % sorted(config.keys())
    hour = config["hour"]
    if not isinstance(hour, int) or not 0 <= hour <= 23:
        return f"daily_at 'hour' must be int in [0, 23]; got {hour!r}"
    minute = config.get("minute", 0)
    if not isinstance(minute, int) or not 0 <= minute <= 59:
        return f"daily_at 'minute' must be int in [0, 59]; got {minute!r}"
    tz = config.get("tz")
    if tz is not None and not isinstance(tz, str):
        return f"daily_at 'tz' must be a string; got {type(tz).__name__}"
    return None


def _validate_every_n_hours(config: dict[str, Any]) -> str | None:
    """Validate ``every_n_hours`` config: ``{n}``."""
    if "n" not in config:
        return "every_n_hours requires 'n' (positive int hours)"
    n = config["n"]
    if not isinstance(n, int) or n <= 0:
        return f"every_n_hours 'n' must be a positive int; got {n!r}"
    return None


def _validate_random_within_window(config: dict[str, Any]) -> str | None:
    """Validate ``random_within_window``: ``{start_hour, end_hour, tz, fires_per_day}``."""
    if "start_hour" not in config or "end_hour" not in config:
        return "random_within_window requires 'start_hour' and 'end_hour' (0-23)"
    start = config["start_hour"]
    end = config["end_hour"]
    if not isinstance(start, int) or not 0 <= start <= 23:
        return f"random_within_window 'start_hour' must be int in [0, 23]; got {start!r}"
    if not isinstance(end, int) or not 0 <= end <= 23:
        return f"random_within_window 'end_hour' must be int in [0, 23]; got {end!r}"
    if start == end:
        return "random_within_window requires start_hour != end_hour"
    tz = config.get("tz")
    if tz is not None and not isinstance(tz, str):
        return f"random_within_window 'tz' must be a string; got {type(tz).__name__}"
    fpd = config.get("fires_per_day", 1)
    if not isinstance(fpd, int) or fpd <= 0:
        return f"random_within_window 'fires_per_day' must be positive int; got {fpd!r}"
    return None


def _validate_one_shot_at(
    config: dict[str, Any],
    *,
    now: datetime | None,
) -> str | None:
    """Validate ``one_shot_at`` config: ``{fire_at_iso}``."""
    if "fire_at_iso" not in config:
        return "one_shot_at requires 'fire_at_iso' (ISO 8601 timestamp)"
    iso = config["fire_at_iso"]
    if not isinstance(iso, str):
        return f"one_shot_at 'fire_at_iso' must be a string; got {type(iso).__name__}"
    try:
        fire_at = datetime.fromisoformat(iso)
    except ValueError:
        return f"one_shot_at 'fire_at_iso' must be valid ISO 8601; got {iso!r}"
    if fire_at.tzinfo is None:
        fire_at = fire_at.replace(tzinfo=UTC)
    reference = now if now is not None else datetime.now(UTC)
    if fire_at <= reference:
        return f"one_shot_at 'fire_at_iso' must be in the future; got {iso!r} <= now ({reference.isoformat()})"
    return None


def _validate_cron(config: dict[str, Any]) -> str | None:
    """Validate ``cron`` config: ``{expr}`` (5-field cron, APScheduler-parsed)."""
    if "expr" not in config:
        return "cron requires 'expr' (5-field cron expression)"
    expr = config["expr"]
    if not isinstance(expr, str) or not expr.strip():
        return f"cron 'expr' must be a non-empty string; got {expr!r}"
    # APScheduler parses + validates; we surface its raise as a tool
    # error rather than letting it explode at tick time. Lazy import so
    # the validator costs nothing for non-cron schedule types.
    try:
        from apscheduler.triggers.cron import CronTrigger  # noqa: PLC0415

        CronTrigger.from_crontab(expr)
    except ValueError as exc:
        return f"cron 'expr' invalid: {exc}"
    except Exception as exc:  # noqa: BLE001 - apscheduler raises varied types
        return f"cron 'expr' parse failed: {exc}"
    return None


def _validate_relative_delay(config: dict[str, Any]) -> str | None:
    """Validate ``relative_delay`` config: ``{delay}`` ('30m', '2h', '1d')."""
    if "delay" not in config:
        return "relative_delay requires 'delay' (e.g. '30m', '2h', '1d')"
    delay = config["delay"]
    if not isinstance(delay, str) or len(delay) < 2:
        return f"relative_delay 'delay' must be '<int><unit>' (s/m/h/d); got {delay!r}"
    unit = delay[-1].lower()
    if unit not in {"s", "m", "h", "d"}:
        return f"relative_delay 'delay' unit must be s/m/h/d; got {unit!r}"
    try:
        value = int(delay[:-1])
    except ValueError:
        return f"relative_delay 'delay' integer part invalid; got {delay!r}"
    if value <= 0:
        return f"relative_delay 'delay' must be positive; got {delay!r}"
    return None


def _validate_interval(config: dict[str, Any]) -> str | None:
    """Validate ``interval`` config: ``{seconds}``."""
    if "seconds" not in config:
        return "interval requires 'seconds' (positive int)"
    seconds = config["seconds"]
    if not isinstance(seconds, int) or seconds <= 0:
        return f"interval 'seconds' must be a positive int; got {seconds!r}"
    return None


# ---------------------------------------------------------------------------
# context_from cycle detection
# ---------------------------------------------------------------------------


# Resolver Protocol shape the validator calls back into. The tool layer
# wraps WakeScheduleCollection.get((conversation_id, schedule_id)) so
# this validator stays pure / DB-agnostic and the same surface can be
# stubbed in unit tests.
ContextFromResolver = Callable[[UUID], Awaitable["_ChainNode | None"]]


class _ChainNode:
    """Lightweight result row the resolver returns for one chain hop.

    Carries the upstream schedule's ``context_from_schedule_id`` (the
    next hop in the chain) plus the ``conversation_id`` so the
    cross-conversation guard can fire. Constructed by the tool's
    closure over :class:`WakeScheduleCollection`.
    """

    __slots__ = ("conversation_id", "context_from_schedule_id")

    def __init__(
        self,
        *,
        conversation_id: UUID,
        context_from_schedule_id: UUID | None,
    ) -> None:
        self.conversation_id = conversation_id
        self.context_from_schedule_id = context_from_schedule_id


async def validate_context_from_chain(
    *,
    new_schedule_id: UUID,
    proposed_context_from: UUID,
    conversation_id: UUID,
    resolver: ContextFromResolver,
    max_depth: int = CONTEXT_FROM_MAX_DEPTH,
) -> str | None:
    """Walk ``proposed_context_from`` to detect cycles + cross-conv leaks.

    Per PLACEMENT §1.6:

    - The chain is same-conversation-only. If any hop's
      ``conversation_id`` differs from the new schedule's
      ``conversation_id``, reject.
    - Cycles are forbidden. The walker accumulates visited schedule
      ids; revisiting any (including ``new_schedule_id`` itself, which
      is added to ``visited`` up-front so self-references reject as
      "cycle") yields an error.
    - Chains deeper than :data:`CONTEXT_FROM_MAX_DEPTH` are rejected
      as "too deep".

    Returns the first error encountered as a human-readable string, or
    ``None`` when the chain is clean.

    :param new_schedule_id: the schedule being created/updated;
        already-visited so it can't appear downstream
    :ptype new_schedule_id: UUID
    :param proposed_context_from: the target schedule (chain head)
    :ptype proposed_context_from: UUID
    :param conversation_id: the new schedule's conversation; chain
        hops must match
    :ptype conversation_id: UUID
    :param resolver: async callable mapping ``schedule_id -> _ChainNode``
        (or ``None`` when the schedule is missing)
    :ptype resolver: ContextFromResolver
    :param max_depth: max hops the walker visits before refusing
    :ptype max_depth: int
    :return: error string or ``None``
    :rtype: str | None
    """
    visited: set[UUID] = {new_schedule_id}
    current: UUID | None = proposed_context_from
    depth = 0
    while current is not None:
        if depth >= max_depth:
            return f"context_from chain exceeds max depth {max_depth} (potential cycle or runaway chain)"
        if current in visited:
            return f"context_from chain contains a cycle: schedule {current} revisited"
        visited.add(current)
        node = await resolver(current)
        if node is None:
            return f"context_from target schedule {current} not found"
        if node.conversation_id != conversation_id:
            return (
                f"context_from target schedule {current} belongs to a "
                "different conversation; chain must stay in-conversation"
            )
        current = node.context_from_schedule_id
        depth += 1
    return None
