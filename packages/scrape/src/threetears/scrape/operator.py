"""The human-handover surface, as a router a platform mounts rather than a service we run.

Some targets sit behind a wall no unattended fetch will pass, and a person clears one in
seconds. This module is the path for that person: the page they look at, the noVNC client it
loads, and the WebSocket that carries the display's pixels to them.

**A seam, like everything else in this package.** ``ScrapeTool`` is registered by a platform,
``list_walled()`` is called by one, ``ScrapeDriver`` is a protocol a platform picks an
implementation of. Nothing here runs itself either: :func:`build_operator_router` returns an
``APIRouter`` to mount into an app the platform already has, and every dependency is passed in.
No environment reads, no internal resolution, no server.

**Why this is MIT code and the display is not.** The display lives in a nodriver sidecar, which
is AGPL-3.0 and therefore isolated to its own container -- it imports nothing from
``threetears`` and never will. That container cannot use this family's coordination, identity
or NATS primitives, so anything it did with them would be a hand-rolled reimplementation one
boundary away from the real thing. The split is therefore not cosmetic: the MIT half owns the
operator's socket, the session's lease and the control subject, and the AGPL half owns Xvfb,
Chromium and x11vnc and nothing else. They share a Kubernetes pod, so the MIT half reaches the
display over loopback.

**Every URL emitted here is relative.** A platform mounts this under a prefix of its choosing,
at a depth this module can never learn, so a leading slash would resolve against the origin
root and break -- invisibly, and only in the deployment nobody can test from here. The rule is
narrow enough to check: nothing handed to a client begins with ``/``.

**The token is a capability, not an identity.** The platform authenticates its operator and
decides who may be handed a session; this module checks that whoever arrived is carrying
something that entitles them to this display. Those are different claims, and conflating them
would put policy evaluation in a library that cannot see a policy.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from threetears.observe import get_logger

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from fastapi import APIRouter

__all__ = [
    "OPERATOR_ASSETS",
    "OPERATOR_PAGE",
    "DisplayEndpoint",
    "SessionAuthorizer",
    "build_operator_router",
    "relay_stream",
]

log = get_logger(__name__)

#: The page, the vendored noVNC client, and the licence notice that travels with it.
#:
#: Resolved from this module's own location rather than looked up in a configured directory,
#: because it ships inside the wheel: there is no deployment in which it is somewhere else, and
#: a configurable path would be a way to point at a noVNC that is not the one this page was
#: tested against.
OPERATOR_ASSETS = Path(__file__).resolve().parent / "operator_assets"

#: The operator's page. Ours and MIT; a sibling of the vendored tree, never a file inside it.
OPERATOR_PAGE = OPERATOR_ASSETS / "operator.html"

#: How much to move in one direction before yielding. RFB framebuffer updates are large and
#: bursty; a small buffer turns one update into many wakeups and reads to the person watching
#: as a laggy screen, which is the one thing this surface exists to be good at.
_RELAY_CHUNK_BYTES = 65536

#: Subprotocol prefix carrying the session token.
#:
#: A subprotocol because it is the only header a browser can set on a WebSocket upgrade --
#: reachable as the second argument to ``new WebSocket(url, protocols)`` and as noVNC's
#: ``wsProtocols``. The alternatives are worse in specific ways rather than merely different: a
#: cookie is state this platform does not use, and a query parameter writes a live credential
#: into access logs, browser history and referrer headers.
TOKEN_SUBPROTOCOL_PREFIX = "hitl-token."


class DisplayEndpoint(Protocol):
    """Where the RFB server for a session can be reached.

    A protocol rather than a host/port pair because the answer is deployment-shaped: loopback
    inside a pod today, and a caller that arranges otherwise should not have to fork this
    module to say so.
    """

    async def __call__(self, session_id: str) -> tuple[str, int]:
        """Resolve *session_id* to the host and port of its RFB server."""
        ...


class SessionAuthorizer(Protocol):
    """Decides whether a token entitles its bearer to a session's display.

    Injected because the answer belongs to the platform. A deployment verifying a hub-signed
    token does so with :mod:`threetears.core.security.identity_token`; one comparing an opaque
    minted value does that instead. This module needs neither to be true.
    """

    async def __call__(self, token: str) -> str | None:
        """Return the session id the token entitles, or ``None`` to refuse."""
        ...


def token_from_subprotocols(offered: str) -> str:
    """Pull the session token out of the offered WebSocket subprotocols.

    A client offers a list -- noVNC offers ``binary``, and this arrangement adds one more
    carrying the token behind a fixed prefix, so the two are told apart by shape rather than by
    position. Returns an empty string when none is offered, which fails authorization like any
    other wrong value rather than raising.
    """
    for entry in offered.split(","):
        name = entry.strip()
        if name.startswith(TOKEN_SUBPROTOCOL_PREFIX):
            return name[len(TOKEN_SUBPROTOCOL_PREFIX) :]
    return ""


async def relay_stream(
    send: Callable[[bytes], Awaitable[None]],
    receive: Callable[[], Awaitable[bytes]],
    host: str,
    port: int,
) -> None:
    """Carry bytes between a connected client and an RFB server, interpreting nothing.

    Byte-for-byte and deliberately ignorant of RFB's framing. RFB is a stateful binary protocol
    whose structure this has no business knowing: anything it understood, it could get wrong,
    and getting it wrong corrupts a stream that has no way to resync.

    **One seam, kept narrow on purpose.** Bytes in, bytes out, over a plain TCP connection --
    today a loopback one to the sidecar sharing this pod. On Kubernetes a WebSocket pins to
    whichever pod the ingress routed it to, and the display lives in exactly one pod, so landing
    on the wrong one is possible. The cheap answer, and the one taken, is to make any pod the
    right pod: claim the session, bring that pod's display up, and re-open whatever was not
    finished. A completed target is already durable, because its state is exported and sealed
    the moment the human says done, so a reconnect costs one in-flight challenge rather than a
    session.

    **If that proves too expensive**, the alternative is relaying these same bytes to the pod
    that does hold the display, over NATS, keyed on the session id. Only this function changes.
    Know what it costs before reaching for it: core NATS is at-most-once, so one dropped message
    under slow-consumer conditions corrupts the stream permanently and presents to the operator
    as a frozen screen rather than an error; and a full-screen update is megabytes, continuous
    while somebody works, on the bus every other subject shares. The trade is real work saved
    against a worse failure mode, and it should be settled on measurements -- how often these
    sockets drop and land elsewhere, and how long a target stays open -- rather than on taste.

    :raises OSError: when the RFB server cannot be reached.
    """
    reader, writer = await asyncio.open_connection(host, port)

    async def _to_client() -> None:
        while data := await reader.read(_RELAY_CHUNK_BYTES):
            await send(data)

    async def _to_display() -> None:
        while True:
            writer.write(await receive())
            await writer.drain()

    pumps = [asyncio.create_task(_to_client()), asyncio.create_task(_to_display())]
    try:
        # Either direction ending ends the session. A half-open RFB stream is not a state the
        # protocol recovers from, and leaving the other pump running would hold a socket and a
        # task for a connection that can no longer carry anything.
        await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for pump in pumps:
            pump.cancel()
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- the peer is already gone; a close that fails changes nothing and must not mask the disconnect that got us here
            log.debug("operator: RFB connection did not close cleanly", exc_info=True)


def build_operator_router(
    *,
    authorize: SessionAuthorizer,
    display: DisplayEndpoint,
) -> APIRouter:
    """Build the router a platform mounts to give its operators a display.

    :param authorize: decides whether a token entitles its bearer to a session, and to which
        one. The platform's own concern -- see :class:`SessionAuthorizer`.
    :ptype authorize: SessionAuthorizer
    :param display: resolves a session to the RFB server serving it.
    :ptype display: DisplayEndpoint
    :return: a router carrying the operator page, the noVNC client and the RFB WebSocket, ready
        to mount under any prefix
    :rtype: APIRouter
    """
    # Deliberately late, and the LAZINESS IS THE MODULE rather than the import statement. An
    # earlier shape imported FastAPI inside this function instead, which kept the extra optional
    # and silently broke the WebSocket route: this module uses `from __future__ import
    # annotations`, so `websocket: WebSocket` is a string at runtime, and FastAPI resolves a
    # handler's annotations against the DEFINING MODULE's globals -- where a name bound only as a
    # local variable does not exist. FastAPI then treated the parameter as a request field and
    # closed every upgrade with 1008, which is also the code a refused token gets, so a route
    # that never ran looked exactly like a rejected operator.
    #
    # `operator_routes` imports FastAPI at its own module scope, so annotations resolve there.
    # Importing that module here keeps the extra as optional as it ever was, because nothing
    # reaches it until a caller asks for a router.
    from .operator_routes import build_router  # noqa: PLC0415

    return build_router(authorize=authorize, display=display, assets=OPERATOR_ASSETS, page=OPERATOR_PAGE)
