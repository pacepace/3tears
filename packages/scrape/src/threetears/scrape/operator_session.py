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
fit -- it is one context manager over this same :meth:`KVLease.hold`, and it reports a lost hold
through :class:`~threetears.core.coordination.LockHold`. The one property that still rules it out is that it
always has a maximum hold: past it the lock stops renewing so a wedged body cannot starve a fleet,
and an operator session has no such ceiling -- a long solve would lose its display mid-session. :meth:`LeaseHandle.refresh` is a compare-and-swap against the recorded
holder and raises :class:`LeaseLost`, and :meth:`KVLease.hold` renews on that in the background
and turns it into the loss this module reports.

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

from typing import TYPE_CHECKING

from threetears.core.coordination import KVLease
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.nats.kv import KvCapable

__all__ = [
    "SESSION_CLAIM_REFRESH",
    "SESSION_CLAIM_TTL",
    "SessionClaim",
    "claim_session",
    "operator_session_lease",
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


def operator_session_lease(nats_client: KvCapable, *, key_scope: str, pod_id: str | None = None) -> KVLease:
    """The lease a platform hands :func:`claim_session`: bind-only, on the platform's shared leases bucket.

    A display claim runs in a TOOL pod, and a pod holds no stream-management verb -- ``STREAM.CREATE``
    carries ``sources`` in its body, so a pod allowed to create a bucket could copy any stream on the
    bus into it. The bucket is the lease's default ``leases``, which the connection materialises as
    ``{ns}-leases``: the one name a tool pod is granted, and the one the hub declares at startup. A
    lease built any other way either asks for a create the pod's grant refuses -- a JetStream deadline
    on the first claim -- or names a bucket nothing grants, which is the same deadline later.

    **Keyed under the pod's own scope.** Every tool pod binds that one bucket and is granted only the
    keys under its own scope, so each claim is ``{key_scope}.{digest}``: replicas of one pod contend
    for one key, and no pod can read, steal or release another pod's claim.

    :param nats_client: the pod's connected NATS client
    :ptype nats_client: KvCapable
    :param key_scope: this pod's key scope -- ``kv_key_scope_for(Principal.TOOL_POD, pod_id=...)``
        over its ``tool_pods.id``, the scope its grant on the bucket is narrowed to
    :ptype key_scope: str
    :param pod_id: this pod's holder identity; ``None`` lets the lease mint one per process
    :ptype pod_id: str | None
    :return: a lease that binds the hub-declared bucket and never creates one
    :rtype: KVLease
    :raises ValueError: when ``key_scope`` is not one literal subject token
    """
    return KVLease(nats_client, pod_id=pod_id, create_if_missing=False, key_scope=key_scope)


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
        style as every other collaborator in this package -- in a pod, the bind-only lease
        :func:`operator_session_lease` builds. ``None`` claims nothing -- see below.
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

    # KVLease.hold renews the claim by compare-and-swap in the background and reports loss on its
    # own event, which this claim shares -- the loop that used to live here moved into core so
    # every consumer holding a lease across real work gets the same one. Fail fast: every other
    # wait would hold this caller open while a human works.
    held = await lease.hold(
        session_claim_key(session_id),
        ttl=ttl,
        renew_every=refresh,
        max_wait_seconds=0,
        # the key is a one-way digest; the session id is what an operator can find a session by
        log_extra={"session_id": session_id},
    )
    claim.lost = held.lost
    try:
        yield claim
    finally:
        # Released even when the claim was lost -- the delete is fenced on the holder, so releasing
        # a claim somebody else now owns is a no-op rather than a theft -- and best-effort: the
        # likeliest reason a release fails is the unreachable coordination layer that ended the
        # claim, and a cleanup error must not replace the body's own outcome. The TTL is the backstop.
        await held.release()
