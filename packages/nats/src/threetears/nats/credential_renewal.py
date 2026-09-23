"""when to renew a connection whose credential expires, and when that cadence is unsafe.

The auth-callout mints each connection's user JWT with a finite TTL. At expiry the NATS
server closes the connection with an auth ``-ERR`` that nats-py routes STRAIGHT to a
terminal ``_close`` -- it never enters ``_attempt_reconnect``, so forever-reconnect (which
governs only the network-drop path) does not cover it. A connection that is not renewed
before then dies on a timer.

The fix is a **proactive reconnect** a margin BEFORE expiry, while the current JWT is still
valid: the reconnect re-runs the auth-callout and mints a fresh one, and the connection
rides on. :meth:`threetears.nats.NatsClient.renew_credential` runs that loop; this module
is only its arithmetic, pure so it is testable without a server, and so the Hub can refuse
to mint a TTL this side could not schedule (:data:`REAUTH_MARGIN_SECONDS`).

The client cannot read its own minted JWT -- it is server-side -- so the schedule is derived
from the TTL DURATION, anchored at the most recent (re)connect: sleep ``ttl - margin``,
reconnect, repeat. The duration comes from wherever the caller learns it: an agent from its
Hub handshake, a standalone pod from its environment (:func:`nats_user_jwt_ttl_seconds`).
"""

from __future__ import annotations

import os
from typing import Final

from threetears.observe import get_logger

__all__ = [
    "NATS_USER_JWT_TTL_ENV",
    "PLATFORM_DEFAULT_NATS_USER_JWT_TTL_SECONDS",
    "REAUTH_BUFFER_SECONDS",
    "REAUTH_LEEWAY_SECONDS",
    "REAUTH_MARGIN_SECONDS",
    "REAUTH_MIN_SLEEP_SECONDS",
    "REAUTH_RETRY_SECONDS",
    "REAUTH_UNKNOWN_TTL_RECHECK_SECONDS",
    "has_schedulable_ttl",
    "nats_user_jwt_ttl_seconds",
    "seconds_until_reauth",
    "unsafe_reauth_delay_reason",
]

log = get_logger(__name__)

#: reconnect this many seconds before the JWT's notional expiry, to cover the server's
#: clock-skew leeway, so the connection is never renewed after the server has closed it.
REAUTH_LEEWAY_SECONDS: Final[int] = 60
#: extra margin on top of the leeway so the whole reconnect round trip -- drop and reopen
#: the transport, re-run the auth-callout -- completes well before expiry.
REAUTH_BUFFER_SECONDS: Final[int] = 30
#: total margin subtracted from the TTL. The Hub imports it to refuse a TTL no client could
#: schedule safely.
REAUTH_MARGIN_SECONDS: Final[int] = REAUTH_LEEWAY_SECONDS + REAUTH_BUFFER_SECONDS
#: after a FAILED renewal, retry this fast: a connection nearing expiry must not wait a cycle.
REAUTH_RETRY_SECONDS: Final[float] = 5.0
#: floor on the scheduled sleep, so a tiny TTL renews promptly without busy-spinning.
REAUTH_MIN_SLEEP_SECONDS: Final[float] = 1.0
#: when the TTL is unknown, re-check on this cadence rather than reconnect on a guess.
REAUTH_UNKNOWN_TTL_RECHECK_SECONDS: Final[float] = 60.0

#: the variable a standalone connection reads its TTL from -- the same one the platform's
#: auth-callout responder mints with, so both sides agree without a handshake.
NATS_USER_JWT_TTL_ENV: Final[str] = "FOURTEENAIBOTS_NATS_USER_JWT_TTL_SECONDS"
#: the TTL the platform mints when that variable is unset.
#:
#: the error is not symmetric. assuming LESS than the minted TTL costs only churn: a still-valid
#: credential is recycled, and a tool pod re-registers its manifest on every reconnect. assuming
#: MORE is fatal: the JWT expires first, and nats-py routes the auth ``-ERR`` to a terminal close
#: forever-reconnect does not cover. so this must never exceed the platform's minted TTL, and must
#: move with it -- it drifted once by keeping a superseded value after the mint was raised. a
#: deployment that tunes the minted TTL sets :data:`NATS_USER_JWT_TTL_ENV` on the client too.
PLATFORM_DEFAULT_NATS_USER_JWT_TTL_SECONDS: Final[int] = 300


def has_schedulable_ttl(ttl_seconds: int | None) -> bool:
    """whether ``ttl_seconds`` is a TTL a renewal can be scheduled against: a positive int.

    the ONE predicate for "the credential's lifetime is known", shared by the schedule and
    the loop so "never reconnect on a guess" holds in both.

    :param ttl_seconds: the credential's TTL, or ``None`` when unknown
    :ptype ttl_seconds: int | None
    :return: whether the schedule can use it
    :rtype: bool
    """
    return ttl_seconds is not None and ttl_seconds > 0


def seconds_until_reauth(ttl_seconds: int | None) -> float:
    """how long to sleep before the next renewal, from the credential's TTL.

    ``ttl - leeway - buffer`` from the latest (re)connect, never below
    :data:`REAUTH_MIN_SLEEP_SECONDS`; :data:`REAUTH_UNKNOWN_TTL_RECHECK_SECONDS` when the TTL
    is not schedulable, so the loop looks again soon without reconnecting.

    :param ttl_seconds: the credential's TTL, or ``None`` when unknown
    :ptype ttl_seconds: int | None
    :return: seconds to sleep
    :rtype: float
    """
    result = REAUTH_UNKNOWN_TTL_RECHECK_SECONDS
    if ttl_seconds is not None and ttl_seconds > 0:
        delay = float(ttl_seconds - REAUTH_MARGIN_SECONDS)
        result = delay if delay > REAUTH_MIN_SLEEP_SECONDS else REAUTH_MIN_SLEEP_SECONDS
    return result


def unsafe_reauth_delay_reason(
    delay_seconds: float,
    ttl_seconds: int | None,
    *,
    longest_request_seconds: float,
) -> str | None:
    """why a renewal cadence would cut off requests in flight, or ``None`` when it is safe.

    A renewal is a real disconnect: every request in flight loses its reply inbox. So the
    renewal interval is a hard ceiling on how long any request on the connection may take.
    When it falls at or below the longest one the caller makes, those requests can never
    finish -- they die at their own timeout, presenting as "it answers quickly, then hangs"
    with nothing in the logs naming the TTL. This names it.

    :param delay_seconds: the scheduled sleep before the next renewal
    :ptype delay_seconds: float
    :param ttl_seconds: the credential's TTL, or ``None`` when unknown
    :ptype ttl_seconds: int | None
    :param longest_request_seconds: the longest request this connection makes
    :ptype longest_request_seconds: float
    :return: the reason, or ``None`` when the cadence is safe
    :rtype: str | None
    """
    result: str | None = None
    if has_schedulable_ttl(ttl_seconds) and delay_seconds <= longest_request_seconds:
        result = (
            f"the NATS credential TTL is {ttl_seconds}s, so the connection is renewed every "
            f"{delay_seconds:.0f}s (ttl-{REAUTH_MARGIN_SECONDS}). A renewal DROPS every request in "
            f"flight, and the longest this connection makes takes up to {longest_request_seconds:.0f}s, "
            f"so a request longer than {delay_seconds:.0f}s can never finish. Raise "
            f"{NATS_USER_JWT_TTL_ENV} above {longest_request_seconds + REAUTH_MARGIN_SECONDS:.0f}."
        )
    return result


def nats_user_jwt_ttl_seconds() -> int | None:
    """the credential TTL a connection with no handshake assumes, from its environment.

    :data:`NATS_USER_JWT_TTL_ENV` when set, :data:`PLATFORM_DEFAULT_NATS_USER_JWT_TTL_SECONDS`
    when not. A non-positive or malformed value is ``None`` -- unknown -- so the renewal loop
    re-checks rather than churning the connection on a guess, and an operator's typo is never
    a crash.

    :return: the TTL, or ``None`` when the configured value is unusable
    :rtype: int | None
    """
    raw = os.environ.get(NATS_USER_JWT_TTL_ENV)
    result: int | None = PLATFORM_DEFAULT_NATS_USER_JWT_TTL_SECONDS
    if raw is not None:
        try:
            parsed = int(raw)
        except ValueError:
            log.warning("invalid %s=%r; treating the NATS credential TTL as unknown", NATS_USER_JWT_TTL_ENV, raw)
            result = None
        else:
            result = parsed if parsed > 0 else None
    return result
