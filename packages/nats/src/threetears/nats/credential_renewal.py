"""when to renew a connection whose credential expires, and when that cadence is unsafe.

The auth-callout mints each connection's user JWT with a finite TTL -- a long one, 24 hours by
default (:data:`PLATFORM_DEFAULT_NATS_USER_JWT_TTL_SECONDS`), because the TTL is a backstop rather
than how access is taken away. At expiry the NATS
server closes the connection (``client.authExpired`` sends ``-ERR 'User Authentication
Expired'`` and closes it), and nats-py routes that ``-ERR`` STRAIGHT to a terminal ``_close``
-- it never enters ``_attempt_reconnect``, so forever-reconnect (which governs only the
network-drop path) does not cover it. A connection that is not renewed before then dies on a
timer. NATS has no in-band re-authentication that keeps a connection's interest: a second
``CONNECT`` on a live connection makes the server drop every subscription it holds
(``processConnect``: ``if !firstConnect { c.clearAccountSubs(false) }``) and blocks its read
loop on the callout -- a gap by another name.

**The renewal is make-before-break.** A credential is per CONNECTION and stays valid until its
own ``exp``, so the client opens a SECOND connection -- the auth-callout mints it a fresh JWT
-- moves every subscription onto it, points new work at it, and keeps the old connection open
until everything it was carrying has finished, then drains it before its JWT expires. Nothing
is ever unsubscribed from both, so no message published during the handover finds no
listener, and a reply to a request in flight still arrives on the connection that asked
(:meth:`threetears.nats.NatsClient.renew_connection`).

That fixes the schedule. The old connection must outlive the longest request it may be
carrying when it is replaced, so the successor is opened ``ttl - leeway - buffer - longest``
after the current connection was established, and the old one is retired ``longest`` later --
still ``buffer`` short of the point the server's clock-skew leeway allows. This module is only
that arithmetic, pure so it is testable without a server, and so the Hub can refuse to mint a
TTL this side could not schedule (:data:`REAUTH_MARGIN_SECONDS`).

The client cannot read its own minted JWT -- it is server-side -- so the schedule is derived
from the TTL DURATION, anchored at the moment the current connection was established. The
duration comes from wherever the caller learns it: an agent from its Hub handshake, a
standalone pod from its environment (:func:`nats_user_jwt_ttl_seconds`).
"""

from __future__ import annotations

import os
from typing import Final, TypeIs

from threetears.observe import get_logger

__all__ = [
    "NATS_USER_JWT_TTL_ENV",
    "PLATFORM_DEFAULT_NATS_USER_JWT_TTL_SECONDS",
    "REAUTH_BUFFER_SECONDS",
    "REAUTH_CONNECT_TIMEOUT_SECONDS",
    "REAUTH_LEEWAY_SECONDS",
    "REAUTH_MARGIN_SECONDS",
    "REAUTH_MIN_SLEEP_SECONDS",
    "REAUTH_RETIRE_DRAIN_SECONDS",
    "REAUTH_RETRY_SECONDS",
    "REAUTH_UNKNOWN_TTL_RECHECK_SECONDS",
    "has_schedulable_ttl",
    "nats_user_jwt_ttl_seconds",
    "seconds_until_reauth",
    "seconds_until_retirement",
    "unsafe_renewal_reason",
]

log = get_logger(__name__)

#: retire a connection this many seconds before the JWT's notional expiry, to cover the server's
#: clock-skew against the minter, so a connection is never still in use after the server has
#: closed it.
REAUTH_LEEWAY_SECONDS: Final[int] = 60
#: extra margin on top of the leeway, spent opening the successor connection
#: (:data:`REAUTH_CONNECT_TIMEOUT_SECONDS`), draining the retired one
#: (:data:`REAUTH_RETIRE_DRAIN_SECONDS`), and one fast retry of a failed renewal
#: (:data:`REAUTH_RETRY_SECONDS`).
REAUTH_BUFFER_SECONDS: Final[int] = 30
#: total margin subtracted from the TTL, before the longest request is. The Hub imports it to
#: refuse a TTL no client could schedule safely.
REAUTH_MARGIN_SECONDS: Final[int] = REAUTH_LEEWAY_SECONDS + REAUTH_BUFFER_SECONDS
#: ceiling on opening the successor connection, auth-callout round trip included.
REAUTH_CONNECT_TIMEOUT_SECONDS: Final[float] = 10.0
#: ceiling on draining the retired connection: its subscriptions' queued messages are handed
#: over and its outbound buffer flushed within this, or it is closed.
REAUTH_RETIRE_DRAIN_SECONDS: Final[float] = 10.0
#: after a FAILED renewal, retry this fast: a connection nearing expiry must not wait a cycle.
REAUTH_RETRY_SECONDS: Final[float] = 5.0
#: floor on the scheduled sleep, so a tiny TTL renews promptly without busy-spinning.
REAUTH_MIN_SLEEP_SECONDS: Final[float] = 1.0
#: when the TTL is unknown, re-check on this cadence rather than renew on a guess.
REAUTH_UNKNOWN_TTL_RECHECK_SECONDS: Final[float] = 60.0

#: the variable a standalone connection reads its TTL from -- the same one the platform's
#: auth-callout responder mints with, so both sides agree without a handshake.
NATS_USER_JWT_TTL_ENV: Final[str] = "FOURTEENAIBOTS_NATS_USER_JWT_TTL_SECONDS"
#: the TTL the platform's auth-callout mints when :data:`NATS_USER_JWT_TTL_ENV` is unset, and
#: so the TTL a connection with no handshake assumes: 24 hours.
#:
#: A BACKSTOP, not the fence. nats-server takes a credential away from a live connection only at
#: its ``exp``, so a short TTL used to be how a revoked or superseded principal was cut off -- at
#: the price of a renewal handover on every pod every few minutes. Access is now taken away when it
#: must be: the consumer kicks the connection (:func:`threetears.nats.kick_connection`) and its
#: auth-callout refuses the reconnect. The TTL remains only to bound a kick that was lost, and to
#: re-verify each principal once a day; the renewal it forces is the same make-before-break
#: handover (:meth:`threetears.nats.NatsClient.renew_connection`).
#:
#: ONE owner: the generic responder's
#: :data:`~threetears.nats.auth_callout_responder.DEFAULT_NATS_USER_JWT_TTL_SECONDS` is this
#: constant, so no minter's default can be shorter than what a client assumes. the error is not
#: symmetric: assuming LESS than the minted TTL costs only churn (a still-valid credential is
#: replaced early), while assuming MORE is fatal (the JWT expires first, and nats-py routes the
#: auth ``-ERR`` to a terminal close forever-reconnect does not cover). a deployment that tunes the
#: minted TTL sets :data:`NATS_USER_JWT_TTL_ENV` on both sides.
PLATFORM_DEFAULT_NATS_USER_JWT_TTL_SECONDS: Final[int] = 86_400


def has_schedulable_ttl(ttl_seconds: int | None) -> TypeIs[int]:
    """whether ``ttl_seconds`` is a TTL a renewal can be scheduled against: a positive int.

    the ONE predicate for "the credential's lifetime is known", shared by the schedule, the
    safety check and the loop so "never renew on a guess" holds in all three. a ``TypeIs`` so
    a caller needs no separate ``None`` check to use the TTL after it.

    :param ttl_seconds: the credential's TTL, or ``None`` when unknown
    :ptype ttl_seconds: int | None
    :return: whether the schedule can use it
    :rtype: TypeIs[int]
    """
    return ttl_seconds is not None and ttl_seconds > 0


def seconds_until_reauth(ttl_seconds: int | None, *, longest_request_seconds: float) -> float:
    """how long after the current connection was established to open its successor.

    ``ttl - leeway - buffer - longest``: late enough to renew as rarely as the TTL allows, early
    enough that the connection being replaced can still carry a request started just before
    the handover to completion before it must be retired. Never below
    :data:`REAUTH_MIN_SLEEP_SECONDS`; :data:`REAUTH_UNKNOWN_TTL_RECHECK_SECONDS` when the TTL is
    not schedulable, so the loop looks again soon without renewing.

    :param ttl_seconds: the credential's TTL, or ``None`` when unknown
    :ptype ttl_seconds: int | None
    :param longest_request_seconds: the longest request the connection makes
    :ptype longest_request_seconds: float
    :return: seconds from the connection's establishment to its successor's
    :rtype: float
    """
    result = REAUTH_UNKNOWN_TTL_RECHECK_SECONDS
    if has_schedulable_ttl(ttl_seconds):
        delay = float(ttl_seconds - REAUTH_MARGIN_SECONDS) - longest_request_seconds
        result = delay if delay > REAUTH_MIN_SLEEP_SECONDS else REAUTH_MIN_SLEEP_SECONDS
    return result


def seconds_until_retirement(
    ttl_seconds: int | None,
    *,
    connection_age_seconds: float,
    longest_request_seconds: float,
) -> float:
    """how long a replaced connection is kept open for the work it was carrying.

    ``longest_request_seconds`` -- anything started on it before the handover is bounded by
    that -- but never past the point where the drain that retires it would still be running at
    the credential's expiry less the leeway. A renewal that ran late, after failed attempts,
    therefore shortens the hold rather than letting the server close the connection under a
    request. ``0`` when that point has already passed. An unknown TTL holds for the longest
    request: nothing is known that says it cannot.

    :param ttl_seconds: the replaced connection's credential TTL, or ``None`` when unknown
    :ptype ttl_seconds: int | None
    :param connection_age_seconds: how long ago the replaced connection was established
    :ptype connection_age_seconds: float
    :param longest_request_seconds: the longest request the connection makes
    :ptype longest_request_seconds: float
    :return: seconds to keep the replaced connection open before draining it
    :rtype: float
    """
    result = longest_request_seconds
    if has_schedulable_ttl(ttl_seconds):
        remaining = float(ttl_seconds - REAUTH_LEEWAY_SECONDS) - REAUTH_RETIRE_DRAIN_SECONDS - connection_age_seconds
        result = max(0.0, min(longest_request_seconds, remaining))
    return result


def unsafe_renewal_reason(ttl_seconds: int | None, *, longest_request_seconds: float) -> str | None:
    """why a TTL is too short to renew without cutting off a request, or ``None`` when it is safe.

    A renewal retires the replaced connection once the longest request it may be carrying has
    had time to finish, and that has to happen before the credential expires. When the TTL
    leaves no room for that -- ``ttl <= longest + leeway + buffer`` -- a request started just
    before a handover can still be cut off when the old connection is retired or expires, which
    presents as "it answers quickly, then hangs" with nothing in the logs naming the TTL. This
    names it.

    The ONE judge of that invariant; the Hub's minimum TTL is the same inequality, from the same
    :data:`REAUTH_MARGIN_SECONDS`.

    :param ttl_seconds: the credential's TTL, or ``None`` when unknown
    :ptype ttl_seconds: int | None
    :param longest_request_seconds: the longest request this connection makes
    :ptype longest_request_seconds: float
    :return: the reason, or ``None`` when the TTL is safe or unknown
    :rtype: str | None
    """
    result: str | None = None
    if has_schedulable_ttl(ttl_seconds) and ttl_seconds - REAUTH_MARGIN_SECONDS <= longest_request_seconds:
        minimum_ttl = longest_request_seconds + REAUTH_MARGIN_SECONDS
        result = (
            f"the NATS credential TTL is {ttl_seconds}s. A renewal keeps the connection it replaces open "
            f"for the longest request it may be carrying ({longest_request_seconds:.0f}s here) and must "
            f"retire it {REAUTH_MARGIN_SECONDS}s before its credential expires, so a TTL of "
            f"{minimum_ttl:.0f}s or less cannot carry that request across a renewal -- it presents as "
            f'"it answers quickly, then hangs". Raise {NATS_USER_JWT_TTL_ENV} above {minimum_ttl:.0f}.'
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
