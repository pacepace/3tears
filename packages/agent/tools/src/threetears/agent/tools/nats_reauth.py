"""timing helpers for the tool-pod NATS-JWT re-auth loop (platform-auth Option A, tool-pod side).

The NATS auth-callout mints each standalone tool pod's connection user JWT with a finite TTL. At
expiry the NATS server closes the connection with an auth ``-ERR`` that nats-py routes STRAIGHT to a
terminal ``_close`` -- it never enters ``_attempt_reconnect``, so the canonical wrapper's
forever-reconnect (which only governs the network-drop path) does NOT cover it. The pod's data plane
then wedges and its heartbeat supervisor eventually ``os._exit``s the process; on a host / docker
daemon with no restart policy that is a permanent tool-count drop.

The fix is **proactive re-auth**: force a NATS reconnect a margin BEFORE expiry, while the current JWT
is still valid, so nats-py re-runs the auth-callout and mints a fresh JWT -- the connection rides on
indefinitely. This module owns ONLY the pure scheduling math so it is unit-testable without the
server; the loop and the reconnect itself live on
:class:`threetears.agent.tools.server.ToolServer`.

This mirrors the agent-runtime module ``aibots_agents.runtime.nats_reauth`` EXACTLY (same constants,
same predicate, same schedule). It is deliberately duplicated rather than imported: the 3tears library
must not depend on the aibots SDK (the dependency runs the other way), so the shared timing contract
is carried as a parallel copy with matching values.

Unlike the agent runtime -- which learns its NATS-JWT TTL from the Hub handshake reply -- a standalone
tool pod receives no such handshake, so the TTL DURATION is sourced from the pod's own config
(:func:`threetears.agent.tools.config.get_nats_user_jwt_ttl_seconds`, the platform default 150s,
env-overridable). The schedule is anchored at the most recent (re)connect: the loop sleeps
``ttl - margin`` and then reconnects, which re-anchors the next cycle.
"""

from __future__ import annotations

__all__ = [
    "REAUTH_LEEWAY_SECONDS",
    "REAUTH_BUFFER_SECONDS",
    "REAUTH_RETRY_SECONDS",
    "REAUTH_MIN_SLEEP_SECONDS",
    "REAUTH_UNKNOWN_TTL_RECHECK_SECONDS",
    "has_schedulable_ttl",
    "seconds_until_reauth",
]

#: reconnect this many seconds before the JWT's notional expiry to cover the NATS server's clock-skew
#: leeway, so the pod never reconnects so late the server has already closed the connection.
REAUTH_LEEWAY_SECONDS = 60
#: extra margin on top of the leeway so the full reconnect round-trip (drop + reopen transport,
#: re-run the auth-callout, re-handshake, JWKS warm) completes well before expiry.
REAUTH_BUFFER_SECONDS = 30
#: on a FAILED re-auth, retry fast (seconds) -- a connection nearing expiry must not wait a full cycle.
REAUTH_RETRY_SECONDS = 5.0
#: floor on the scheduled sleep so a tiny TTL cannot busy-spin the event loop; it re-auths promptly
#: but never hot-loops.
REAUTH_MIN_SLEEP_SECONDS = 1.0
#: when the TTL is unknown (config reports a non-positive / malformed value), re-check on this cadence
#: rather than forcing a reconnect on a guess -- the loop must not churn the connection on an unknown
#: interval; the reactive heartbeat-supervisor backstop covers any terminal close in that window.
REAUTH_UNKNOWN_TTL_RECHECK_SECONDS = 60.0


def has_schedulable_ttl(ttl_seconds: int | None) -> bool:
    """whether ``ttl_seconds`` is usable TTL to schedule a proactive reconnect against.

    THE single predicate for "we know the connection JWT's lifetime": a positive int. shared by
    :func:`seconds_until_reauth` AND the re-auth loop so the loop's "never reconnect on a guess" rule
    and the scheduler's unknown-TTL fallback stay in lockstep.

    :param ttl_seconds: the NATS user JWT TTL from pod config, or ``None`` when unknown
    :ptype ttl_seconds: int | None
    :return: ``True`` when the TTL is positive int the schedule can use
    :rtype: bool
    """
    return ttl_seconds is not None and ttl_seconds > 0


def seconds_until_reauth(ttl_seconds: int | None) -> float:
    """seconds to sleep before forcing the next proactive reconnect, from the connection JWT's TTL.

    re-auth target = ``ttl - leeway - buffer`` from the most recent (re)connect. clamped to at least
    :data:`REAUTH_MIN_SLEEP_SECONDS` so a tiny TTL re-auths promptly without busy-spinning. when the
    TTL is not schedulable (:func:`has_schedulable_ttl`), returns
    :data:`REAUTH_UNKNOWN_TTL_RECHECK_SECONDS` so the loop re-checks soon WITHOUT reconnecting on a
    guess.

    :param ttl_seconds: the NATS user JWT TTL from pod config, or ``None`` when unknown
    :ptype ttl_seconds: int | None
    :return: seconds to sleep before the next re-auth check
    :rtype: float
    """
    if not has_schedulable_ttl(ttl_seconds):
        result = REAUTH_UNKNOWN_TTL_RECHECK_SECONDS
    else:
        assert ttl_seconds is not None  # narrowed by has_schedulable_ttl; for the type checker
        delay = ttl_seconds - REAUTH_LEEWAY_SECONDS - REAUTH_BUFFER_SECONDS
        result = float(delay) if delay > REAUTH_MIN_SLEEP_SECONDS else REAUTH_MIN_SLEEP_SECONDS
    return result
