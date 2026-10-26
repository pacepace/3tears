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
reaches the display some other way, and moves the same bytes over it. The pod's own end of that
second arrangement is :mod:`threetears.scrape.operator_pipe`, which serves the display onto a
stream for as long as the pod holds the session's claim.

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

from pathlib import Path
from typing import TYPE_CHECKING

from .operator_relay import (
    ClaimLookup,
    ClaimWatch,
    DisplayEndpoint,
    DisplayReader,
    DisplayTransport,
    DisplayWriter,
    SessionAuthorizer,
    relay_stream,
)

# Released at this module's path, and read by a platform minting tokens (see the constant), so both stay importable
# here; they are defined beside the relay that reads them.
from .operator_relay import TOKEN_SUBPROTOCOL_PREFIX as TOKEN_SUBPROTOCOL_PREFIX
from .operator_relay import token_from_subprotocols as token_from_subprotocols

_HITL_MISSING: ImportError | None = None
try:
    from .operator_routes import build_router
except ImportError as exc:  # NOSILENT: the hitl extra is optional; the router builder says so
    _HITL_MISSING = exc

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

#: The page, the vendored noVNC client, and the licence notice that travels with it.
#:
#: Resolved from this module's own location rather than looked up in a configured directory,
#: because it ships inside the wheel: there is no deployment in which it is somewhere else, and
#: a configurable path would be a way to point at a noVNC that is not the one this page was
#: tested against.
OPERATOR_ASSETS = Path(__file__).resolve().parent / "operator_assets"

#: The operator's page. Ours and MIT; a sibling of the vendored tree, never a file inside it.
OPERATOR_PAGE = OPERATOR_ASSETS / "operator.html"


def build_operator_router(
    *,
    authorize: SessionAuthorizer,
    display: DisplayEndpoint,
    claim: ClaimLookup | None = None,
    transport: DisplayTransport | None = None,
) -> APIRouter:
    """Build the router a platform mounts to give its operators a display.

    :param authorize: decides whether a token entitles its bearer to a session, and to which
        one. The platform's own concern -- see :class:`SessionAuthorizer`. Note the constraint on
        what the platform may MINT, recorded at
        :data:`~threetears.scrape.operator_relay.TOKEN_SUBPROTOCOL_PREFIX`: the token rides a
        WebSocket subprotocol name, so it must use the URL-safe alphabet. A standard-base64 token
        fails in the browser before this is ever called, and leaves no server-side trace.
    :ptype authorize: SessionAuthorizer
    :param display: resolves a session to the RFB server serving it.
    :ptype display: DisplayEndpoint
    :param claim: finds this pod's claim on a session, so the operator's stream ends when a
        handover takes the session elsewhere. ``None`` on a deployment with one pod, which has
        no handover to survive -- see :class:`ClaimLookup` for what leaving it out costs.
    :ptype claim: ClaimLookup | None
    :param transport: opens the connection to the display. ``None`` connects by TCP to whatever
        *display* resolved, which is the co-located arrangement and what every existing caller
        gets. A deployment whose display is in a pod with no inbound network path supplies one
        that reaches it another way, and *display*'s host and port are then passed through
        uninterpreted for that transport to read however it likes.
    :ptype transport: DisplayTransport | None
    :return: a router carrying the operator page, the noVNC client and the RFB WebSocket, ready
        to mount under any prefix
    :rtype: APIRouter
    """
    # `operator_routes` imports FastAPI at its own module scope, and that is load-bearing rather
    # than style. This module uses `from __future__ import annotations`, so `websocket: WebSocket`
    # is a string at runtime, and FastAPI resolves a handler's annotations against the DEFINING
    # MODULE's globals. An earlier shape imported FastAPI inside this function, where the name
    # existed only as a local: FastAPI then treated the parameter as a request field and closed
    # every upgrade with 1008, the same code a refused token gets. The routes module is imported
    # at this module's top, guarded, so a deployment without the extra imports this module and
    # learns what is missing here.
    if _HITL_MISSING is not None:
        raise ImportError("build_operator_router needs the hitl extra: install 3tears-scrape[hitl]") from _HITL_MISSING
    return build_router(
        authorize=authorize,
        display=display,
        claim=claim,
        assets=OPERATOR_ASSETS,
        page=OPERATOR_PAGE,
        transport=transport,
    )
