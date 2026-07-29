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
Chromium and x11vnc and nothing else. Where the two share a Kubernetes pod, the MIT half reaches
the display over loopback by connecting to it, and that is what :func:`relay_stream` does with no
transport supplied. Where they do not -- a hub terminating an operator's socket is in neither
container and has no route into that pod at all -- the same function takes a transport that
reaches the display some other way, and moves the same bytes over it.

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
    "ClaimLookup",
    "ClaimWatch",
    "DisplayEndpoint",
    "DisplayReader",
    "DisplayTransport",
    "DisplayWriter",
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
#:
#: **THIS CONSTRAINS WHAT A TOKEN MAY CONTAIN, and a platform mints the token.** RFC 6455
#: subprotocol names are HTTP tokens, so a value carrying ``/``, ``=``, a comma or whitespace is
#: not merely untidy -- it breaks, and it breaks in the worst place. Standard base64 uses ``/``
#: and ``=``; a comma is how the offered list is separated, so one inside a token truncates it
#: server-side. Either way the browser's ``new WebSocket(...)`` raises, or the handler is never
#: entered, and the operator reads "Could not open the display" with NO server-side trace at all.
#:
#: So a token must be URL-safe-alphabet material: ``secrets.token_urlsafe`` (which is what the
#: sidecar uses), a base64URL value with padding stripped, or a hex digest. A JWT is fine -- its
#: three segments are base64URL and its separator is ``.`` -- but a signature encoded with standard
#: base64 is not. This is a constraint on the PLATFORM, which is why it is stated on the constant
#: a platform reads rather than left in a test.
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


class DisplayReader(Protocol):
    """The reading half of an open connection to a display.

    Exactly what :class:`asyncio.StreamReader` already offers and nothing more, so the default
    transport satisfies it without an adapter and a substitute has a short list to implement.
    """

    async def read(self, n: int) -> bytes:
        """Return at most *n* bytes, or empty once the far end has finished."""
        ...


class DisplayWriter(Protocol):
    """The writing half of an open connection to a display.

    The four methods a relay uses on :class:`asyncio.StreamWriter`: two to move bytes and two to
    let go of them. Closing is included because the relay is what closes this -- whatever the
    transport opened is released on the way out, and a substitute that could not be closed would
    leak one per operator session.
    """

    def write(self, data: bytes) -> None:
        """Queue *data* for the display."""
        ...

    async def drain(self) -> None:
        """Wait until the queued bytes have been handed onwards."""
        ...

    def close(self) -> None:
        """Start closing this half."""
        ...

    async def wait_closed(self) -> None:
        """Wait for the close to finish."""
        ...


class DisplayTransport(Protocol):
    """Opens the pair of streams a relay carries bytes over.

    The narrowest substitutable thing: it yields a reader-like and a writer-like pair and
    nothing else. It is handed the endpoint :class:`DisplayEndpoint` resolved and decides what
    reaching that endpoint means -- the default reads it as a TCP address, and one built for a
    deployment where no TCP address can reach the display is free to read it as a key instead.
    The relay itself interprets neither value, and interprets nothing that travels over the
    pair either: RFB is a stateful binary protocol that does not resynchronise, so anything on
    this path that understood it could get it wrong.
    """

    async def __call__(self, host: str, port: int) -> tuple[DisplayReader, DisplayWriter]:
        """Open a connection to the display resolved to *host* and *port*."""
        ...


class ClaimWatch(Protocol):
    """Something that completes when this pod stops owning a session.

    Structural on purpose: :class:`threetears.scrape.operator_session.SessionClaim` satisfies it,
    and a deployment coordinating some other way is not obliged to adopt that class to say when
    a handover happened.
    """

    async def until_lost(self) -> None:
        """Return once this pod is no longer the session's owner."""
        ...


class ClaimLookup(Protocol):
    """Finds the claim this pod holds on a session, if it holds one.

    The router serves a session for as long as this pod owns it, and ownership is decided
    outside the router -- so it has to be able to ASK. Without this the relay outlives the claim
    that entitled it: a pod that lost a session to a handover keeps a live browser and a live
    display in front of an operator for the rest of the session's TTL.

    Injected and optional, like every other collaborator here. A single-pod deployment has no
    handover to survive and passes nothing.
    """

    async def __call__(self, session_id: str) -> ClaimWatch | None:
        """Return this pod's claim on *session_id*, or ``None`` when it holds none."""
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


async def _open_tcp_display(host: str, port: int) -> tuple[DisplayReader, DisplayWriter]:
    """Connect to the display over TCP, which is what a relay with no transport does.

    A named function rather than a lambda over :func:`asyncio.open_connection` so the behaviour
    it preserves has somewhere to be stated: a display sharing this process's network namespace
    is reached by connecting to it, and the transport parameter changes nothing about that for a
    caller who does not pass one.
    """
    return await asyncio.open_connection(host, port)


async def _await_stop(until: Callable[[], Awaitable[None]]) -> None:
    """Await a caller-supplied stop signal, so it can be raced as a task.

    Takes a CALLABLE rather than an awaitable, and that is not style. An awaitable built at the
    call site is a coroutine that exists whether or not this relay ever gets far enough to await
    it -- and the first thing the relay does is open the display connection, whose failure is a
    handled case that returns early. On that path the coroutine was created, never awaited, and
    surfaced as a ``RuntimeWarning`` in whatever ran next. Deferring construction to here means
    there is nothing to leak.
    """
    await until()


async def relay_stream(
    send: Callable[[bytes], Awaitable[None]],
    receive: Callable[[], Awaitable[bytes]],
    host: str,
    port: int,
    *,
    transport: DisplayTransport | None = None,
    until: Callable[[], Awaitable[None]] | None = None,
    benign: tuple[type[BaseException], ...] = (),
) -> None:
    """Carry bytes between a connected client and an RFB server, interpreting nothing.

    Byte-for-byte and deliberately ignorant of RFB's framing. RFB is a stateful binary protocol
    whose structure this has no business knowing: anything it understood, it could get wrong,
    and getting it wrong corrupts a stream that has no way to resync.

    **One seam, kept narrow on purpose.** Bytes in, bytes out, over whatever *transport* opened --
    by default a plain TCP connection to *host* and *port*, which is a loopback one to the sidecar
    sharing this pod. On Kubernetes a WebSocket pins to whichever pod the ingress routed it to, and
    the display lives in exactly one pod, so landing on the wrong one is possible. The cheap
    answer, and the one taken, is to make any pod the right pod: claim the session, bring that
    pod's display up, and re-open whatever was not finished. A completed target is already durable,
    because its state is exported and sealed the moment the human says done, so a reconnect costs
    one in-flight challenge rather than a session.

    **That whole arrangement needs the socket and the display in one pod, and a deployed tool
    pod cannot give it that.** A tool pod carries no Service and no Ingress: it dials out and
    nothing dials in, so a process terminating the operator's WebSocket anywhere else has no
    address to connect to and no way to be given one. Reaching the display over the bus that pod
    already holds open is not a cheaper alternative to be weighed against this one, it is the
    only route -- and *transport* is where it plugs in. What that costs is a property of the
    transport rather than of this function: core NATS is at-most-once, so one message lost under
    slow-consumer conditions corrupts a stream that cannot resynchronise, which is why a
    bus-backed one has to make a gap detectable instead of skippable.

    :param transport: opens the reader and writer this moves bytes between, and is handed *host*
        and *port* to say which display. Defaults to a plain TCP connection, so a caller who
        passes nothing gets exactly the behaviour that shipped before this parameter existed.
        The endpoint is passed through uninterpreted -- see :class:`DisplayTransport`.
    :ptype transport: DisplayTransport | None
    :param until: called and awaited alongside the relay; when it completes first the relay ends.
        This is how a pod that has stopped owning the session stops showing it -- see
        :meth:`threetears.scrape.operator_session.SessionClaim.until_lost`. A callable rather than
        an awaitable so nothing is constructed on a path that never awaits it.
    :ptype until: Callable[[], Awaitable[None]] | None
    :param benign: exception types a pump may raise to mean "the other end went away", which is
        the ORDINARY end of an operator's session rather than a fault. A client library that
        signals disconnection by raising -- as Starlette's ``receive_bytes`` does, for a clean
        close as much as a dirty one -- makes this necessary: without it every operator closing a
        tab is reported as an unreachable display. This function must not import a web framework
        to know that, so the caller names the types.
    :ptype benign: tuple[type[BaseException], ...]
    :raises OSError: when a pump fails mid-stream for a reason not listed in *benign*. Failing to
        reach the display raises whatever the transport raises, which for the default one is the
        ``OSError`` an unreachable or refused TCP endpoint produces.
    """
    open_display: DisplayTransport = transport if transport is not None else _open_tcp_display
    reader, writer = await open_display(host, port)

    async def _to_client() -> None:
        while data := await reader.read(_RELAY_CHUNK_BYTES):
            await send(data)

    async def _to_display() -> None:
        while True:
            writer.write(await receive())
            await writer.drain()

    pumps = [asyncio.create_task(_to_client()), asyncio.create_task(_to_display())]
    # The stop signal rides alongside the pumps rather than in a second `wait`, so losing the
    # session interrupts a relay that is otherwise parked forever on a socket nobody is writing
    # to. Wrapped in a task because `asyncio.wait` refuses a bare coroutine, and tracked
    # separately so it can be cancelled without being mistaken for a pump that ended.
    stop = asyncio.create_task(_await_stop(until)) if until is not None else None
    watched = [*pumps, stop] if stop is not None else pumps
    try:
        # Either direction ending ends the session. A half-open RFB stream is not a state the
        # protocol recovers from, and leaving the other pump running would hold a socket and a
        # task for a connection that can no longer carry anything.
        done, _pending = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
        # RETRIEVE THE OUTCOME, which `asyncio.wait` does not do for you -- but distinguish the
        # ordinary end from a fault, because the two are NOT the same and reporting them
        # identically is its own defect. Left unretrieved, a genuine fault resurfaces later as a
        # bare "Task exception was never retrieved" with no session, host or port attached.
        # Reported indiscriminately, every operator who closes a tab is logged as an unreachable
        # display with a traceback and a host and port that were both fine -- which is worse,
        # because it is the common path and it drowns the rare one it was meant to surface.
        #
        # A pump raising a `benign` type means the other end went away. The stop signal completing
        # means a handover took the session. Neither is a fault; anything else is.
        for finished in done:
            if finished is stop:
                continue
            error = finished.exception()
            if error is None or isinstance(error, benign):
                continue
            raise OSError(f"the display relay failed mid-stream: {error}") from error
    finally:
        for task in watched:
            task.cancel()
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- the peer is already gone; a close that fails changes nothing and must not mask the disconnect that got us here
            log.debug("operator: RFB connection did not close cleanly", exc_info=True)


def build_operator_router(
    *,
    authorize: SessionAuthorizer,
    display: DisplayEndpoint,
    claim: ClaimLookup | None = None,
) -> APIRouter:
    """Build the router a platform mounts to give its operators a display.

    :param authorize: decides whether a token entitles its bearer to a session, and to which
        one. The platform's own concern -- see :class:`SessionAuthorizer`. Note the constraint on
        what the platform may MINT, recorded at :data:`TOKEN_SUBPROTOCOL_PREFIX`: the token rides a
        WebSocket subprotocol name, so it must use the URL-safe alphabet. A standard-base64 token
        fails in the browser before this is ever called, and leaves no server-side trace.
    :ptype authorize: SessionAuthorizer
    :param display: resolves a session to the RFB server serving it.
    :ptype display: DisplayEndpoint
    :param claim: finds this pod's claim on a session, so the operator's stream ends when a
        handover takes the session elsewhere. ``None`` on a deployment with one pod, which has
        no handover to survive -- see :class:`ClaimLookup` for what leaving it out costs.
    :ptype claim: ClaimLookup | None
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

    return build_router(
        authorize=authorize,
        display=display,
        claim=claim,
        assets=OPERATOR_ASSETS,
        page=OPERATOR_PAGE,
    )
