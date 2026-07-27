"""Which pod owns a session's display, and how it stops being two of them.

A session is one operator working one display. On Kubernetes the display lives in exactly one
pod -- an ``x11vnc`` in the AGPL sidecar sharing that pod's network namespace -- while the
operator's WebSocket lands on whichever pod the ingress happened to route it to. Nothing in that
arrangement stops a second pod deciding it also serves the session, and the cost of it doing so
is not a race that resolves: it is two Xvfb displays, two browsers, and a human driving whichever
one their socket reached while the other collects half a solve.

So a pod claims a session before it acts as that session's owner, and stops acting the moment
the claim goes. :func:`claim_session` is that claim.

**Why this reaches for KVLease and not nats_distributed_lock.** The lock looks like the closer
fit -- it is one context manager, it owns its own heartbeat, and its docstring pairs it with
:func:`threetears.nats.serve_owner`, which is the other half of this surface. It is still the
wrong primitive here, for one specific reason: its heartbeat is an unconditional
``bucket.put``. If a holder stalls long enough for its entry to expire and another pod wins the
key, the first pod's next heartbeat overwrites the winner's entry and neither one ever learns.
Two holders, no error, and for a display that is exactly the divergence being prevented.
:meth:`LeaseHandle.refresh` is a compare-and-swap against the recorded holder and raises
:class:`LeaseLost` instead, which is the property this module is built on. What KVLease does not
carry is the renewal loop, and that loop is short precisely because its entire job is to react
to the exception the lock cannot raise.

**A claim can be lost without anything failing.** Losing it is not an error condition to
retry -- it means another pod is now the owner, and continuing to serve is the fault. The claim
therefore reports loss rather than raising it, because the caller is usually parked in a relay
that has to be interrupted rather than a call that can return.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta

from threetears.core.coordination import KVLease, LeaseHandle, LeaseLost
from threetears.observe import get_logger

__all__ = [
    "SESSION_CLAIM_REFRESH",
    "SESSION_CLAIM_TTL",
    "SessionClaim",
    "claim_session",
    "session_claim_key",
]

log = get_logger(__name__)

#: How long a claim outlives the pod holding it.
#:
#: This is the window in which a session whose pod died is still refused to everybody else, so it
#: is a direct cost to an operator waiting to be let back in. It is not free to shorten: below
#: roughly three renewal intervals, an ordinary scheduling stall starts expiring live claims.
SESSION_CLAIM_TTL = timedelta(seconds=45)

#: How often a live claim is renewed.
#:
#: A third of the TTL, so two consecutive renewals can fail without the claim lapsing. One
#: missed renewal is a blip; three in a row is a pod that cannot defend what it holds.
SESSION_CLAIM_REFRESH = timedelta(seconds=15)


def session_claim_key(session_id: str) -> str:
    """Derive the coordination key for *session_id*.

    Hashed rather than used verbatim, for the same reason
    :meth:`threetears.nats.Subjects.forward` hashes its own: a session id is arbitrary,
    supplied from outside this module, and a JetStream KV key admits only a restricted
    character set. A digest is subject-safe, deterministic so every pod derives the same key
    from the same id, and one-way with nothing lost -- both ends start from the id.

    Deriving both this key and the control subject from the same id the same way is what makes
    "the pod holding the claim is the pod serving the subject" true by construction rather than
    by two pieces of code agreeing.

    :param session_id: the session whose display is being claimed
    :ptype session_id: str
    :return: a key safe to use against a JetStream KV bucket
    :rtype: str
    :raises ValueError: if *session_id* is empty
    """
    if not session_id:
        raise ValueError("session_id must be non-empty")
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


@dataclass
class SessionClaim:
    """This pod's live claim on one session's display.

    :ivar session_id: the session claimed
    :ivar lost: set when this pod is no longer the owner. Never cleared: a claim that has been
        lost stays lost, because the pod that took it has already brought its own display up and
        there is nothing to return to.
    """

    session_id: str
    lost: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def held(self) -> bool:
        """Whether this pod may still act as the session's owner."""
        return not self.lost.is_set()

    async def until_lost(self) -> None:
        """Block until this claim is lost.

        For racing against work that would otherwise run forever -- an operator's relay ends
        when the human leaves, and this ends it when the pod stops being entitled to serve
        them::

            await asyncio.wait(
                [asyncio.create_task(relay()), asyncio.create_task(claim.until_lost())],
                return_when=asyncio.FIRST_COMPLETED,
            )
        """
        await self.lost.wait()


async def _renew(handle: LeaseHandle, claim: SessionClaim, *, refresh: float, ttl: float) -> None:
    """Renew *claim* until it is lost or this task is cancelled.

    Two ways to stop being the owner, and they are told apart deliberately.

    :class:`LeaseLost` is authoritative: somebody else's holder id is on the entry, so this pod
    is demonstrably not the owner and says so immediately.

    Anything else -- an unreachable bucket, a transport fault -- is not evidence of anything,
    and treating one failed renewal as lost would end an operator's session over a blip. But it
    cannot be ignored either, because a claim that has gone un-renewed for longer than its TTL
    has expired whether or not this pod noticed, and another pod may already have taken it. So
    the deadline is what decides: keep trying while the claim could still be alive, and give it
    up once it could not.
    """
    loop = asyncio.get_running_loop()
    expires_at = loop.time() + ttl
    while True:
        await asyncio.sleep(refresh)
        try:
            await handle.refresh()
        except LeaseLost:
            log.warning(
                "operator: another pod now owns this session's display",
                extra={"extra_data": {"session_id": claim.session_id}},
            )
            claim.lost.set()
            return
        except Exception as exc:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- a renewal can fail for any transport reason; the claim's own deadline decides whether that has cost us ownership, and swallowing it silently is what the deadline below prevents
            if loop.time() >= expires_at:
                log.warning(
                    "operator: this session's claim went un-renewed past its TTL and is assumed lost",
                    exc_info=True,
                    extra={"extra_data": {"session_id": claim.session_id, "ttl_seconds": ttl}},
                )
                claim.lost.set()
                return
            log.info(
                "operator: could not renew this session's claim, still inside its TTL",
                extra={"extra_data": {"session_id": claim.session_id, "error_type": type(exc).__name__}},
            )
        else:
            expires_at = loop.time() + ttl


@asynccontextmanager
async def claim_session(
    lease: KVLease | None,
    session_id: str,
    *,
    ttl: timedelta = SESSION_CLAIM_TTL,
    refresh: timedelta = SESSION_CLAIM_REFRESH,
) -> AsyncIterator[SessionClaim]:
    """Claim *session_id*'s display for this pod, and hold it for the body.

    Refuses immediately rather than waiting when another pod holds the session. Waiting would
    be the wrong shape twice over: the holder is a human working a page, so the wait is minutes
    to hours, and a caller that cannot have the display wants to say so to its operator now.

    :param lease: the coordination primitive, constructor-injected by the platform in the same
        style as every other collaborator in this package. ``None`` claims nothing -- see below.
    :ptype lease: KVLease | None
    :param session_id: the session whose display is being claimed
    :ptype session_id: str
    :param ttl: how long the claim outlives this pod
    :ptype ttl: timedelta
    :param refresh: how often the claim is renewed; must be shorter than *ttl*
    :ptype refresh: timedelta
    :return: an async iterator yielding the live claim
    :rtype: AsyncIterator[SessionClaim]
    :raises ValueError: if *refresh* is not shorter than *ttl*, which would let a live claim
        lapse under its own holder; or if *ttl* is not a whole number of seconds of at least
        one, which the coordination layer cannot express
    :raises LeaseUnavailable: if another pod holds this session's display
    """
    if refresh >= ttl:
        raise ValueError(f"refresh {refresh} must be shorter than ttl {ttl}, or a live claim lapses under its holder")
    ttl_seconds = ttl.total_seconds()
    # `acquire` takes whole seconds, so anything finer is truncated -- and a sub-second TTL
    # truncates to ZERO, which writes an entry that is stale the instant it lands and hands the
    # display to whoever asks next. Refusing is the only reading that cannot silently mean
    # something else.
    if ttl_seconds < 1 or ttl_seconds != int(ttl_seconds):
        raise ValueError(f"ttl {ttl} must be a whole number of seconds and at least one second")

    claim = SessionClaim(session_id=session_id)

    if lease is None:
        # A deployment with one pod has nothing to coordinate with, and the compose file in
        # this repo is exactly that. Yielding is right; yielding QUIETLY is not -- a platform
        # that meant to pass a lease and did not gets no mutual exclusion and no signal, and
        # the symptom is two operators on two displays believing they share one.
        log.warning(
            "operator: no lease was supplied, so this session's display is not claimed. "
            "Two pods can serve it at once; supply a KVLease on any deployment running more than one.",
            extra={"extra_data": {"session_id": session_id}},
        )
        yield claim
        return

    handle = await lease.acquire(
        session_claim_key(session_id),
        ttl_seconds=int(ttl_seconds),
        # Fail fast. `acquire` reads 0 as "do not block", and every other value would hold this
        # caller open while a human works.
        max_wait_seconds=0,
    )
    renewal = asyncio.create_task(
        _renew(handle, claim, refresh=refresh.total_seconds(), ttl=ttl_seconds),
        name=f"operator-session-claim:{session_id}",
    )
    try:
        yield claim
    finally:
        renewal.cancel()
        # Awaited rather than left to be collected: an un-awaited cancellation can still be
        # mid-refresh, and a refresh landing after the release below would put the entry back
        # for a session that has ended.
        await asyncio.gather(renewal, return_exceptions=True)
        # Released even when the claim was lost. `release` verifies the holder before deleting,
        # so releasing a claim somebody else now owns is a no-op rather than a theft.
        #
        # Best-effort, and that is the point rather than a shrug. The single most likely reason
        # a release fails is that the coordination layer is unreachable -- which is the same
        # reason the claim was given up moments earlier, so it is the EXPECTED path out of a
        # lost claim, not an exotic one. Raising here would replace whatever ended the session
        # with a cleanup error, and it would do it most often when the session ended badly and
        # the original exception was the informative one. The TTL is the backstop: an entry
        # nobody deleted expires on its own, which is what it is for.
        try:
            await handle.release()
        except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- see above; the TTL frees the entry regardless and a cleanup failure must not replace the body's own outcome. Logged with its traceback
            log.warning(
                "operator: could not release this session's claim; it will expire with its TTL",
                exc_info=True,
                extra={"extra_data": {"session_id": session_id, "ttl_seconds": ttl_seconds}},
            )
