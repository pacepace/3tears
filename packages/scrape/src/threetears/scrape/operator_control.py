"""Control messages for a session, routed to the pod that actually holds its display.

The display for a session lives in one pod. Everything that acts on it -- putting a target in
front of the operator, taking a cleared one back, ending the session -- has to reach that pod
and no other. The obvious ways to arrange that are both bad: addressing pods directly makes the
caller track which pod is which and re-track it after every reschedule, and pinning the caller
to a pod with ingress stickiness makes the routing layer responsible for a fact it cannot see.

So the messages are addressed to the SESSION and find their own way. A pod serves its session's
subject for exactly as long as it holds that session's claim, and a caller sends to the subject
without knowing or caring which pod answers.

**What is on the subject, and what is deliberately not.** Four messages act on a live session:
open a tab, complete a tab, close the session, read its state. Opening a session is not one of
them, and the asymmetry is the mechanism rather than an oversight -- taking the claim is what
MAKES a pod the owner, so routing it to an owner would be circular. A caller opens a session by
claiming it (:func:`threetears.scrape.operator_session.claim_session`), and everything after
that is addressed to the session it now owns.

**No cookie jar travels in the clear, in either direction.** A solve is a live credential, and
a bus is a place other subscribers can be granted a read of, so both directions carry ciphertext
and the plaintext exists only inside the pod that holds the display.

Outbound, a completed tab's raw export travels one loopback hop from the sidecar -- which holds no
key by design -- and stops here, where it is sealed before the reply is sent. Inbound, a prior
solve being re-injected arrives sealed and is opened here, immediately before the sidecar is
handed it. The symmetry is the point: sealing the reply while accepting a plaintext jar on the way
in would put exactly the credential this protects onto exactly the bus it protects it from, and
the asymmetry would read as deliberate to whoever found it next.

**Nothing here frames its own errors.** :func:`threetears.nats.forward` already carries a
handler's exception back to the caller as a type name and message, so a handler that raises
produces a :class:`threetears.nats.ForwardedHandlerError` on the other side. A second envelope
on top would be a second thing that can disagree with the first.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Protocol

from threetears.nats import DEFAULT_FORWARD_TIMEOUT, Subjects, forward, serve_owner
from threetears.observe import get_logger

from .session_state import DEFAULT_SESSION_STATE_TTL, open_session_state, seal_session_state

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from pydantic import SecretStr
    from threetears.nats import NatsClient

    from .operator_session import SessionClaim

__all__ = [
    "CLOSE_SESSION",
    "COMPLETE_TAB",
    "OPEN_TAB",
    "READ_STATE",
    "SessionDisplay",
    "UnknownControlMessage",
    "close_session",
    "complete_tab",
    "open_tab",
    "read_state",
    "serve_session",
]

log = get_logger(__name__)

#: Put one target in front of the operator, in its own isolated browser context.
OPEN_TAB = "open_tab"

#: The human says this one is cleared. Replies with the solve, sealed.
COMPLETE_TAB = "complete_tab"

#: Drop every context and release the display.
CLOSE_SESSION = "close_session"

#: Free slots and the live tab set. The one that makes the other three usable: a caller cannot
#: decide to push another target without knowing whether there is room for it.
READ_STATE = "read_state"


class UnknownControlMessage(ValueError):
    """A message arrived that this pod has no operation for.

    Its own type rather than a bare ``ValueError`` because it crosses the wire as a type name:
    a caller seeing this back from :func:`forward` is talking to a pod running a different
    version of this contract, which is a different problem from a malformed request.
    """


class SessionDisplay(Protocol):
    """What the pod holding a display can do to it.

    A protocol because the doing belongs elsewhere. The display is driven by an AGPL-isolated
    sidecar in the same pod, reached over loopback, and this package neither imports it nor
    knows how a platform prefers to talk to it. What this module owns is which pod gets asked
    and what the answer may contain.

    Every method returns plain JSON-able data. Returning richer objects would put a
    serialisation format in the protocol, and the protocol's implementations live in consumers.
    """

    async def open_tab(
        self,
        *,
        target_id: str,
        url: str,
        nav_steps: list[dict[str, Any]] | None,
        session_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Bring one target into the session and return the tab it became.

        *session_state* is a prior solve, RAW: :func:`serve_session` opened it from the sealed
        form the caller sent, because the sidecar behind this protocol holds no key and cannot.
        """
        ...

    async def complete_tab(self, *, tab_id: str) -> dict[str, Any]:
        """Close a cleared tab.

        The returned mapping carries the human's work under ``session_state``, RAW: this is the
        one place it exists unsealed on this side, and :func:`serve_session` seals it before it
        travels. An implementation that sealed it here would be sealing under a key it should
        not have.
        """
        ...

    async def close_session(self) -> dict[str, Any]:
        """Drop every context and release the display."""
        ...

    async def read_state(self) -> dict[str, Any]:
        """Free slots and the tabs currently open."""
        ...


def _decode(payload: bytes) -> tuple[str, dict[str, Any]]:
    """Read the operation and its arguments out of a request payload.

    :raises ValueError: if the payload is not a JSON object carrying a string ``op``
    """
    try:
        message = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"control message is not valid JSON: {exc}") from exc
    if not isinstance(message, dict):
        raise ValueError("control message must be a JSON object")
    op = message.get("op")
    if not isinstance(op, str):
        raise ValueError("control message has no `op`")
    return op, message


def _encode(reply: dict[str, Any]) -> bytes:
    """Render a handler's reply for the wire."""
    return json.dumps(reply).encode("utf-8")


def _require(message: dict[str, Any], field: str) -> str:
    """Read a required string field, or say which one was missing.

    Named rather than generic, because "control message is malformed" reaches the caller as a
    string and is the only thing they will have to go on.
    """
    value = message.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"control message {message.get('op')!r} needs a non-empty {field!r}")
    return value


async def _perform(
    op: str,
    message: dict[str, Any],
    display: SessionDisplay,
    *,
    session_state_key: SecretStr,
    session_state_ttl: timedelta,
) -> dict[str, Any]:
    """Carry out one control message against this pod's display.

    :raises UnknownControlMessage: if *op* names no operation
    :raises ValueError: if a required field is missing
    """
    if op == OPEN_TAB:
        nav_steps = message.get("nav_steps")
        # Opened HERE, at the last moment before the sidecar needs it, so the plaintext exists
        # only inside this pod. `open_session_state` returns None for anything it cannot trust --
        # a wrong key, a tampered token -- and None is exactly right: the tab opens
        # unauthenticated and the human clears the challenge again, which is the same outcome as
        # having no stored solve at all.
        sealed = message.get("session_state_sealed")
        prior = open_session_state(sealed, session_state_key, target_id=message.get("target_id")) if sealed else None
        return await display.open_tab(
            target_id=_require(message, "target_id"),
            url=_require(message, "url"),
            nav_steps=nav_steps if isinstance(nav_steps, list) else None,
            session_state=prior,
        )

    if op == COMPLETE_TAB:
        completed = await display.complete_tab(tab_id=_require(message, "tab_id"))
        # Removed rather than overwritten. Overwriting leaves the key's fate depending on the
        # sealing below succeeding, and a raw jar surviving because an exception unwound past
        # the reassignment is exactly the accident this is here to make impossible.
        raw = completed.pop("session_state", None)
        if raw:
            sealed = seal_session_state(raw, session_state_key, ttl=session_state_ttl)
            completed["session_state_sealed"] = sealed.sealed
            completed["session_state_expires_at"] = sealed.expires_at.isoformat()
        return completed

    if op == CLOSE_SESSION:
        return await display.close_session()

    if op == READ_STATE:
        return await display.read_state()

    raise UnknownControlMessage(f"no control message named {op!r}")


@asynccontextmanager
async def serve_session(
    nats: NatsClient,
    claim: SessionClaim,
    display: SessionDisplay,
    *,
    session_state_key: SecretStr,
    session_state_ttl: timedelta = DEFAULT_SESSION_STATE_TTL,
    tool: str | None = None,
) -> AsyncIterator[None]:
    """Answer this session's control messages for as long as this pod owns it.

    Pairs with :func:`threetears.scrape.operator_session.claim_session`, and the pairing is the
    contract rather than a convention -- the claim is what entitles this pod to answer::

        async with claim_session(lease, session_id) as claim, serve_session(
            nats, claim, display, session_state_key=key
        ):
            await claim.until_lost()

    **A lost claim refuses every message from the moment it is lost.** The subscription is
    dropped when this context exits, which is prompt but not instantaneous, and in between a
    message can still arrive at a pod that has stopped being the owner. Acting on one would put
    a target in front of nobody, or tear down a display the new owner is using. So ownership is
    re-checked per message rather than only at subscribe time: the subscription is the
    optimisation and the check is the mechanism.

    :param nats: the platform's connected client
    :ptype nats: NatsClient
    :param claim: this pod's live claim on the session, which decides both the subject served
        and whether a message may be acted on
    :ptype claim: SessionClaim
    :param display: what this pod can do to the display it holds
    :ptype display: SessionDisplay
    :param session_state_key: operator master key, used to seal a completed tab's solve before
        it leaves this pod
    :ptype session_state_key: SecretStr
    :param session_state_ttl: how long a sealed solve is trusted
    :ptype session_state_ttl: timedelta
    :param tool: the registered namespace name of the tool serving this session, e.g.
        ``tools.scrape-zone_alpha.1-0-0``. Naming it moves the session's control messages onto the
        forward family a tool pod is actually granted; the unscoped default is granted to no
        principal, so a deployment enforcing its grants needs this. ``None`` keeps the previous
        subject and is the compatible default.
    :ptype tool: str | None
    :return: an async iterator yielding while the session is served
    :rtype: AsyncIterator[None]
    :raises RuntimeError: if the claim has already been lost, since serving would be answering
        for a session this pod does not own
    """
    if not claim.held:
        raise RuntimeError(
            f"session {claim.session_id!r} is not held by this pod, so its messages are not ours to answer"
        )

    async def _handle(payload: bytes) -> bytes:
        """Run one control message and return its reply, raising for `forward` to frame."""
        op, message = _decode(payload)
        if not claim.held:
            # Deliberately raised rather than logged and ignored: the caller needs to know its
            # request did not happen, so it can send again and let the new owner take it.
            raise RuntimeError(f"session {claim.session_id!r} moved to another pod before this message was handled")
        log.info(
            "operator: handling a control message",
            extra={"extra_data": {"session_id": claim.session_id, "op": op}},
        )
        return _encode(
            await _perform(
                op,
                message,
                display,
                session_state_key=session_state_key,
                session_state_ttl=session_state_ttl,
            )
        )

    async with serve_owner(nats, claim.session_id, _handle, family=_family_for(tool)):
        try:
            yield
        finally:
            if not claim.held:
                await _release_a_session_this_pod_no_longer_owns(claim, display)


async def _release_a_session_this_pod_no_longer_owns(claim: SessionClaim, display: SessionDisplay) -> None:
    """Tear down the local display after a handover took the session elsewhere.

    **The obligation this closes.** Losing the claim already stops this pod serving: the relay
    ends, control messages are refused, and the lease is released. None of that releases what the
    pod is still HOLDING -- an ``x11vnc`` on a live display, and browser contexts carrying the
    cookies of whatever the human was part way through. Left alone they survive until the sidecar's
    own session TTL reaps them, which is half an hour of a browser holding a target's authenticated
    session for a session this pod is no longer entitled to.

    Safe precisely because the pods do not share a display. Each has its own sidecar on its own
    loopback, so closing this one cannot touch the display the new owner just brought up. On a
    single-pod deployment there is no handover and this never runs.

    Best-effort, and it has to be: this is a `finally` on the way out of a session that has already
    gone wrong, and raising here would replace whatever the caller was dealing with. The sidecar's
    reaper remains the backstop it always was -- this makes it the backstop rather than the
    mechanism.
    """
    try:
        await display.close_session()
    except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- see docstring; the sidecar's own TTL reaper still collects this, and a failure here must not replace the caller's own outcome. Logged with its traceback
        log.warning(
            "operator: could not release the display for a session that moved to another pod; "
            "the sidecar's TTL will collect it",
            exc_info=True,
            extra={"extra_data": {"session_id": claim.session_id}},
        )
    else:
        log.info(
            "operator: released the display for a session that moved to another pod",
            extra={"extra_data": {"session_id": claim.session_id}},
        )


def _family_for(tool: str | None) -> str | None:
    """Derive the forward family a session's control messages ride, or ``None`` for the old one.

    ``None`` keeps the UNSCOPED subject every existing caller uses, so nothing that does not
    name a tool changes behaviour. Naming one moves the session onto
    :meth:`threetears.nats.Subjects.hitl_forward_family`, which is the family a tool pod is
    actually granted -- the unscoped one is granted to no principal at all, so a deployment that
    enforces its grants cannot serve a session without this.

    Both ends derive it here rather than each building its own, because an owner and a caller
    that disagree about the family do not fail loudly: they subscribe and publish different
    subjects, and the caller sees only that nothing answers.
    """
    return None if tool is None else Subjects.hitl_forward_family(tool)


async def _send(
    nats: NatsClient,
    session_id: str,
    message: dict[str, Any],
    *,
    timeout: timedelta,
    tool: str | None = None,
) -> dict[str, Any]:
    """Send one control message to whichever pod holds *session_id*, and read its reply."""
    reply = await forward(nats, session_id, _encode(message), timeout=timeout, family=_family_for(tool))
    decoded: dict[str, Any] = json.loads(reply.decode("utf-8"))
    return decoded


async def open_tab(
    nats: NatsClient,
    session_id: str,
    *,
    target_id: str,
    url: str,
    nav_steps: list[dict[str, Any]] | None = None,
    session_state_sealed: str | None = None,
    timeout: timedelta = DEFAULT_FORWARD_TIMEOUT,
    tool: str | None = None,
) -> dict[str, Any]:
    """Put one target in front of the operator working *session_id*.

    Carries ``url`` and ``nav_steps`` rather than a handle to an earlier fetch, because nothing
    is held while a target waits for a human: it is reported and forgotten, then re-driven from
    these two when somebody actually arrives.

    *session_state_sealed* is a prior solve in the form :func:`complete_tab` handed back and
    :func:`threetears.scrape.session_state.record_session_state` stored -- sealed, never a raw
    jar. Pass it straight through; the owning pod opens it. A caller with plaintext in hand has a
    plaintext credential to keep out of its own logs too, so the sealed form is the only one this
    contract accepts.

    :raises NoOwnerError: if no pod currently holds the session
    :raises ForwardedHandlerError: if the owning pod refused or failed
    """
    return await _send(
        nats,
        session_id,
        {
            "op": OPEN_TAB,
            "target_id": target_id,
            "url": url,
            "nav_steps": nav_steps,
            "session_state_sealed": session_state_sealed,
        },
        timeout=timeout,
        tool=tool,
    )


async def complete_tab(
    nats: NatsClient,
    session_id: str,
    *,
    tab_id: str,
    timeout: timedelta = DEFAULT_FORWARD_TIMEOUT,
    tool: str | None = None,
) -> dict[str, Any]:
    """Take a cleared tab back, and with it the human's work.

    The reply carries ``session_state_sealed`` and ``session_state_expires_at``, never a raw
    export. Open it with :func:`threetears.scrape.session_state.open_session_state` under the
    same key the owning pod sealed it with.

    :raises NoOwnerError: if no pod currently holds the session
    :raises ForwardedHandlerError: if the owning pod refused or failed
    """
    return await _send(nats, session_id, {"op": COMPLETE_TAB, "tab_id": tab_id}, timeout=timeout, tool=tool)


async def close_session(
    nats: NatsClient,
    session_id: str,
    *,
    timeout: timedelta = DEFAULT_FORWARD_TIMEOUT,
    tool: str | None = None,
) -> dict[str, Any]:
    """End the session and release its display.

    :raises NoOwnerError: if no pod currently holds the session
    :raises ForwardedHandlerError: if the owning pod refused or failed
    """
    return await _send(nats, session_id, {"op": CLOSE_SESSION}, timeout=timeout, tool=tool)


async def read_state(
    nats: NatsClient,
    session_id: str,
    *,
    timeout: timedelta = DEFAULT_FORWARD_TIMEOUT,
    tool: str | None = None,
) -> dict[str, Any]:
    """Read free slots and the live tab set from whichever pod holds the session.

    :raises NoOwnerError: if no pod currently holds the session
    :raises ForwardedHandlerError: if the owning pod refused or failed
    """
    return await _send(nats, session_id, {"op": READ_STATE}, timeout=timeout, tool=tool)
