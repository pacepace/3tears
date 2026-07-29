"""The FastAPI half of the operator surface, in a module that imports FastAPI at the top.

**This module exists because of where annotations are resolved, which is not obvious and cost a
silently broken route to find.** :mod:`threetears.scrape.operator` uses
``from __future__ import annotations``, so every annotation in it is a string at runtime, and it
imports FastAPI *inside* ``build_operator_router`` to keep the ``hitl`` extra genuinely optional.
Those two facts do not compose. FastAPI resolves a handler's annotations against the handler's
own ``__globals__`` -- the defining MODULE's namespace -- so ``websocket: "WebSocket"`` could not
be resolved from a name that only ever existed as a local variable. FastAPI then treated the
parameter as a request field, failed to validate it, and closed every upgrade with 1008.

That is the worst possible failure code for this route, because 1008 is also what a refused token
gets. A broken route and a rejected operator were indistinguishable from the outside, and the
route was never once exercised end to end until a test mounted it.

The fix is structural rather than a workaround: FastAPI is imported at module scope HERE, so
annotations resolve, and ``operator.py`` imports this module lazily instead of importing FastAPI
lazily. The extra stays exactly as optional as before -- nothing reaches this module until a
caller actually asks for a router -- and the shape that caused the bug is gone rather than
worked around.
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect
from threetears.observe import get_logger

from .operator import relay_stream, token_from_subprotocols

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from .operator import ClaimLookup, DisplayEndpoint, DisplayTransport, SessionAuthorizer

__all__ = ["build_router"]

log = get_logger(__name__)


def build_router(
    *,
    authorize: SessionAuthorizer,
    display: DisplayEndpoint,
    claim: ClaimLookup | None,
    assets: Path,
    page: Path,
    transport: DisplayTransport | None = None,
) -> APIRouter:
    """Wire the operator page, the vendored client and the RFB WebSocket onto one router.

    See :func:`threetears.scrape.operator.build_operator_router`, which is the public entry
    point and documents the parameters.
    """
    router = APIRouter()

    @router.websocket("/ws")
    async def stream(websocket: WebSocket) -> None:
        """Relay the display to an operator holding a token that entitles them to it."""
        token = token_from_subprotocols(websocket.headers.get("sec-websocket-protocol", ""))
        session_id = await authorize(token)
        if session_id is None:
            # Closed BEFORE the display is even resolved, so nothing is done on behalf of a
            # caller who never had a capability: an unauthenticated request must not be able to
            # cause work against a display. The reason stays vague for the same purpose the rest
            # of this surface does -- distinguishing "no session" from "wrong token" confirms an id
            # to whoever is guessing.
            #
            # THE CODE DOES NOT REACH A BROWSER, and that is worth knowing rather than assuming.
            # Closing before accepting means no WebSocket is ever established, so the server
            # answers the handshake with HTTP 403 and the 1008 is never transmitted -- verified
            # against a real client, not reasoned about. 1008 stays because it is the correct ASGI
            # close code and a test client does surface it; what an operator actually gets is a
            # failed connection, which is why the page distinguishes never-connected from dropped
            # instead of waiting for RFB's `securityfailure` (which cannot fire this early).
            # Logged, because a refusal that leaves no server-side trace is a refusal nobody can
            # diagnose -- and this branch is reached both by a wrong token and by a token whose
            # session has ended, which look identical to the operator on purpose. The token
            # itself is never logged.
            log.info(
                "operator: refused an operator carrying no usable capability",
                extra={"extra_data": {"offered_token": bool(token)}},
            )
            await websocket.close(code=1008, reason="not authorised for this display")
            return

        host, port = await display(session_id)

        # Asked BEFORE the stream opens, and refused rather than ignored when this pod holds no
        # claim. A relay is the one thing here that runs for hours, so it is the one thing that
        # must not outlive the ownership that entitled it: a pod that lost the session to a
        # handover would otherwise keep a live browser and a live display in front of an operator
        # for the rest of the session's TTL, while the pod that actually owns it serves another.
        held = await claim(session_id) if claim is not None else None
        if claim is not None and held is None:
            log.info(
                "operator: refusing a display this pod does not hold",
                extra={"extra_data": {"session_id": session_id}},
            )
            await websocket.close(code=1008, reason="not authorised for this display")
            return

        try:
            # Accepted only once the display has been located, so a caller learns the difference
            # between "refused" and "the display is down" from the close code rather than from a
            # connection that opens and then dies.
            await websocket.accept(subprotocol="binary")
        except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- an operator who navigated away between authorizing and accepting is ordinary, not a fault. Logged with its traceback below
            log.info("operator: the client went away before the stream opened", exc_info=True)
            return

        try:
            await relay_stream(
                websocket.send_bytes,
                websocket.receive_bytes,
                host,
                port,
                transport=transport,
                until=held.until_lost if held is not None else None,
                # An operator closing their tab is the ORDINARY end of a session, and Starlette
                # signals it by raising -- for a clean close exactly as for a dirty one. Without
                # naming it here, every finished session is logged as an unreachable display,
                # with a traceback and a host and port that were both fine.
                benign=(WebSocketDisconnect,),
            )
        except OSError:
            log.warning(
                "operator: the display could not be reached",
                exc_info=True,
                extra={"extra_data": {"session_id": session_id, "host": host, "port": port}},
            )
            # Suppressed, because the most likely reason this close fails is that the client is
            # already gone -- and a client going away is one of the things that gets us here. The
            # close raising would escape the handler and replace the warning above, which is the
            # only line carrying the session id, with a framework traceback that carries none. The
            # socket is finished either way; telling it so is a courtesy, not a step.
            # NOSILENT: the reason is already logged above with the session id; this close is a
            # courtesy to a peer that is probably gone, and cannot change the outcome either way.
            with suppress(
                Exception
            ):  # prawduct:allow prawduct/broad-except -- see above; the diagnosis is already logged and this close cannot change the outcome
                await websocket.close(code=1011, reason="the display is not available")

    @router.get("/", include_in_schema=False)
    async def operator_page() -> FileResponse:
        """The operator's page, served at the router's own root.

        **At the root, and that is load-bearing rather than tidy.** Every URL the page emits is
        relative, so the browser resolves them against the directory the page was served from.
        Served at ``.../hitl/`` the WebSocket resolves to ``.../hitl/ws``; served at ``.../hitl``
        it resolves to ``.../ws``, one directory too high, and the display never connects. A
        request arriving without the trailing slash is redirected to the one with it by the
        framework, which is what makes both spellings of a handed-out link work.

        ``no-store`` because the page carries the shape of the client it loads -- the token
        itself lives in the fragment and never reaches a server -- and a cached page paired with
        a fresh vendored noVNC is the version skew this package ships the tree to avoid.
        """
        return FileResponse(page, media_type="text/html", headers={"Cache-Control": "no-store"})

    # MOUNTED AS A SIBLING OF THE ROUTES ABOVE, NOT ABOVE THEM, and that is the whole safeguard.
    # A mount matches every path beneath it -- including a WebSocket, which `StaticFiles` cannot
    # serve: it dies on its own `assert scope["type"] == "http"`, which reaches the operator as a
    # 500 on connect and a client saying only "Failed to connect to server", indistinguishable
    # from a dead display. The sidecar hit exactly that, because there the client tree was the
    # PARENT of the socket (`/vnc` and `/vnc/ws`) and only registration order kept them apart.
    #
    # Here `/novnc` is beside `/` and `/ws` rather than over them, so nothing it could claim is a
    # route, and the order it is registered in does not matter. That is the reason not to "tidy"
    # this into a mount at the router's root: doing so would put the client tree back above the
    # socket and make correctness depend on registration order again.
    router.mount("/novnc", StaticFiles(directory=assets / "novnc"), name="novnc")

    return router
