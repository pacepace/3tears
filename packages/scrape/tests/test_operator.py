"""The human-handover router: a seam a platform mounts, not a service this package runs.

Every assertion here is about a property a consuming platform depends on and cannot see from
outside: that importing the module costs it no web framework, that a refused operator never
causes a connection to the display, and that the relay moves bytes without understanding them.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import socket
from collections.abc import Iterator
from urllib.parse import urljoin

import pytest
from threetears.scrape.operator import TOKEN_SUBPROTOCOL_PREFIX, relay_stream, token_from_subprotocols


def test_the_module_imports_without_the_optional_web_framework() -> None:
    """`hitl` is an extra, so a deployment that never needs a human never installs FastAPI.

    The import has to stay clean for that to be true, which is why the router builder imports
    FastAPI inside the function rather than at module scope. A stray top-level import would
    make the extra a lie -- every consumer of this package would start needing a web framework,
    and nothing would fail until an install somewhere had a smaller dependency set than this
    repo's own venv.
    """
    import importlib

    module = importlib.import_module("threetears.scrape.operator")
    source = (module.__file__ or "").replace(".pyc", ".py")
    assert source, "the module has no file to inspect"

    with open(source) as handle:  # noqa: PTH123 - reading this repo's own source, not a data path
        top_level = [line for line in handle if line.startswith(("import fastapi", "from fastapi"))]
    assert not top_level, f"FastAPI is imported at module scope, so the extra is not optional: {top_level}"
    # `sys.modules` is deliberately NOT asserted on. An earlier line here read
    # `assert "fastapi" not in sys.modules or True`, which cannot fail, and the reason it was
    # written that way is real: this suite imports FastAPI elsewhere, so by the time this runs the
    # module is legitimately loaded and the honest form would fail for a reason that says nothing
    # about the extra. The source check above is the whole of the claim; a tautology dressed as a
    # second one only makes the first look weaker than it is.


class TestTheTokenIsReadFromTheSubprotocol:
    """Where the credential lives is a security property, not an implementation detail."""

    def test_it_is_found_regardless_of_position(self) -> None:
        """A client orders its own subprotocol list, so position must not identify the token."""
        assert token_from_subprotocols(f"binary, {TOKEN_SUBPROTOCOL_PREFIX}abc") == "abc"
        assert token_from_subprotocols(f"{TOKEN_SUBPROTOCOL_PREFIX}abc, binary") == "abc"

    def test_absent_is_empty_rather_than_an_error(self) -> None:
        """An absent token fails authorization like a wrong one, rather than raising."""
        assert token_from_subprotocols("binary") == ""
        assert token_from_subprotocols("") == ""


class TestTheRelayMovesBytesAndInterpretsNothing:
    """RFB is a stateful binary stream: anything the relay understood, it could get wrong."""

    async def test_both_directions_carry_bytes_verbatim(self) -> None:
        """A byte pattern that looks like framing must survive untouched in both directions."""
        # Chosen to look like something worth parsing: a length prefix, a null, high bytes.
        payload = b"\x00\x00\x01\x2c\xff\xfe\x00RFB-ish\x00"
        server_received: list[bytes] = []
        client_received: list[bytes] = []

        async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            writer.write(payload)
            await writer.drain()
            server_received.append(await reader.read(len(payload)))
            writer.close()

        server = await asyncio.start_server(_handle, "127.0.0.1", 0)
        host, port = server.sockets[0].getsockname()[:2]

        to_client: asyncio.Queue[bytes] = asyncio.Queue()
        from_client: asyncio.Queue[bytes] = asyncio.Queue()
        await from_client.put(payload)

        async def _send(data: bytes) -> None:
            client_received.append(data)
            await to_client.put(data)

        async def _receive() -> bytes:
            return await from_client.get()

        async with server:
            await asyncio.wait_for(relay_stream(_send, _receive, host, port), timeout=5)

        assert b"".join(client_received) == payload, "the display's bytes reached the client altered"
        assert server_received == [payload], "the operator's bytes reached the display altered"

    async def test_the_other_end_going_away_is_the_ordinary_end_not_a_fault(self) -> None:
        """A client library that signals disconnection by RAISING must not read as broken.

        Starlette's ``receive_bytes`` raises ``WebSocketDisconnect`` for a clean close exactly as
        for a dirty one, so an operator closing their tab completes a pump with an exception.
        Reported as a fault, every finished session logs a warning with a traceback and a host and
        port that were both fine -- drowning the one real failure that log line exists for. This
        is the common path; getting it wrong is worse than not checking at all.
        """

        class _ClientWentAway(Exception):
            """Stands in for the framework's own disconnect signal."""

        async def _receive() -> bytes:
            raise _ClientWentAway

        with _quiet_listener() as port:
            # Returns rather than raising: nothing here is a fault.
            await asyncio.wait_for(
                relay_stream(lambda _d: asyncio.sleep(0), _receive, "127.0.0.1", port, benign=(_ClientWentAway,)),
                timeout=5,
            )

    async def test_a_pump_failing_for_any_other_reason_is_raised(self) -> None:
        """The counterpart, so "benign" cannot quietly become "everything".

        Left unretrieved, a genuine mid-stream fault surfaces later as a bare "Task exception was
        never retrieved" with no session, host or port attached to it.
        """

        async def _receive() -> bytes:
            raise RuntimeError("the transport broke")

        with _quiet_listener() as port:
            with pytest.raises(OSError, match="failed mid-stream"):
                await asyncio.wait_for(
                    relay_stream(lambda _d: asyncio.sleep(0), _receive, "127.0.0.1", port), timeout=5
                )

    async def test_an_unreachable_display_never_reaches_for_the_stop_signal(self) -> None:
        """Nothing is constructed on a path that never awaits it.

        The stop signal used to be passed as an awaitable built at the call site, and the relay's
        first act is to open a socket, whose failure returns early. That left a coroutine created
        and never awaited, surfacing as a ``RuntimeWarning`` in whatever ran next -- attributed to
        a completely unrelated test, and invisible here because no warning filter is configured.
        Taking a callable makes the leak impossible rather than merely unlikely.
        """
        started: list[bool] = []

        async def _stop() -> None:
            started.append(True)
            await asyncio.Event().wait()

        with pytest.raises(OSError):
            # Port 1 on loopback: reserved, and nothing listens there.
            await asyncio.wait_for(
                relay_stream(lambda _d: asyncio.sleep(0), asyncio.Event().wait, "127.0.0.1", 1, until=_stop),
                timeout=5,
            )
        assert started == [], "the stop signal was constructed for a relay that never opened"

    async def test_an_unreachable_display_raises_rather_than_hanging(self) -> None:
        """A caller needs to tell "refused" from "the display is down", so this must not hang."""
        with pytest.raises(OSError):
            # Port 1 on loopback: reserved, and nothing listens there.
            await asyncio.wait_for(
                relay_stream(lambda _d: asyncio.sleep(0), asyncio.Event().wait, "127.0.0.1", 1), timeout=5
            )


# The prefix every routing test mounts under. Deep, and with more than one segment, because the
# failures being caught are all "works at the root, breaks under a prefix" -- a test that mounted
# at `/` would pass under every wrong answer, which is exactly why the requirement exists.
_PREFIX = "/platform/api/v1/scrape/hitl"


@contextlib.contextmanager
def _quiet_listener() -> Iterator[int]:
    """A port that accepts a connection and then says nothing, as an idle RFB server does.

    A plain listening socket that is never accepted from in Python: the OS completes the
    handshake and holds the connection in the backlog, so the relay sees an established socket
    with no data and no EOF. That is the state an idle operator's session is in for most of its
    life, and it is the state a relay must be interruptible in.

    Deliberately NOT `asyncio.start_server` with a handler that blocks. `Server.wait_closed()`
    waits for open handlers to finish, so `async with server` around a handler that never returns
    deadlocks the test rather than the code -- and a handler that DOES return lets its writer be
    collected, which closes the socket and EOFs the relay, so every "it stayed open" claim would
    pass for the wrong reason. Both ways round, the harness decides the result.
    """
    listener = socket.socket()
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        yield listener.getsockname()[1]
    finally:
        listener.close()


class _StubClaim:
    """A claim this pod holds, which can be taken away mid-relay."""

    def __init__(self) -> None:
        self.lost = asyncio.Event()

    async def until_lost(self) -> None:
        """Complete once the claim is gone, as ``SessionClaim.until_lost`` does."""
        await self.lost.wait()


def _mounted_app(
    resolved: list[str] | None = None,
    checked: list[str] | None = None,
    claim: object | None = None,
    *,
    with_claim_lookup: bool = False,
    claim_asked: list[str] | None = None,
    rfb_port: int | None = None,
    prefix: str = _PREFIX,
) -> object:
    """A router mounted the way a platform mounts it: under a prefix it chose, at a depth we
    can never learn.

    Both collaborators are observable, and that is not convenience.

    *checked* records every token authorization was asked about, which is the only evidence the
    handler RAN. The route once closed every upgrade with 1008 without executing a line of it,
    because FastAPI could not resolve the handler's annotation and rejected the request as
    malformed -- and 1008 is also what a refused token gets. Asserting on the close code alone
    cannot tell a dead route from a working refusal, so these tests assert the collaborator was
    reached.

    *resolved* records each session the display was looked up for. Resolving a display is the
    first thing done on a caller's behalf, so an entry for a refused caller means the refusal
    came too late.
    """
    from fastapi import FastAPI

    from threetears.scrape.operator import build_operator_router

    async def _authorize(token: str) -> str | None:
        if checked is not None:
            checked.append(token)
        return "session-1" if token == "good" else None

    async def _display(session_id: str) -> tuple[str, int]:
        if resolved is not None:
            resolved.append(session_id)
        return ("127.0.0.1", rfb_port if rfb_port is not None else 5900)

    async def _claim(session_id: str) -> object | None:
        if claim_asked is not None:
            claim_asked.append(session_id)
        return claim

    app = FastAPI()
    router = build_operator_router(
        authorize=_authorize,
        display=_display,
        claim=_claim if with_claim_lookup else None,
    )
    # `prefix=""` mounts at the origin root, which the requirement names alongside the prefixed
    # case: the same page and the same client with no configuration difference. Everything else
    # here mounts under a prefix, because that is where the failures hide -- at the root every
    # wrong answer works.
    app.include_router(router, prefix=prefix)
    return app


class TestThePageAndItsClientSurviveAPrefix:
    """A platform picks the mount point and this router can never learn it.

    There is no configuration path by which it could, and nothing to fall back on if a leading
    slash is emitted -- so the relative-URL rule is the mechanism here rather than a precaution.
    """

    def test_the_page_is_served_under_the_prefix(self) -> None:
        """The operator has to be able to open it where the platform put it."""
        from fastapi.testclient import TestClient

        with TestClient(_mounted_app()) as client:  # type: ignore[arg-type]
            response = client.get(f"{_PREFIX}/")
        assert response.status_code == 200
        assert "Human handover" in response.text

    def test_a_request_without_the_trailing_slash_is_redirected_to_one_with_it(self) -> None:
        """Everything the page emits is resolved against the directory it was served from.

        At ``.../hitl/`` the WebSocket resolves to ``.../hitl/ws``. At ``.../hitl`` it resolves
        to ``.../ws`` -- one directory too high, no route there, and a display that never
        connects. The operator sees "Failed to connect" and nothing else. The redirect is what
        makes the link a platform hands out work whichever way it was written.
        """
        from fastapi.testclient import TestClient

        with TestClient(_mounted_app()) as client:  # type: ignore[arg-type]
            landed = client.get(_PREFIX)
        assert landed.status_code == 200, "a link without the trailing slash does not reach the page"
        assert str(landed.url).endswith("/"), (
            f"the page was served from {landed.url}, which does not end in a slash, so every "
            f"relative URL on it resolves one directory too high"
        )

    def test_it_works_mounted_at_the_root_as_well_as_under_a_prefix(self) -> None:
        """Both halves of the requirement, which named the root explicitly and had only one tested.

        The acceptance shape was "the whole flow works when mounted at `/` and when mounted under a
        prefix, with no configuration difference visible to the operator's browser". Every other
        test here uses a prefix, deliberately, because that is where a leading slash hides. The
        root case was never exercised and was never descoped -- so it is exercised here rather
        than quietly dropped.
        """
        from fastapi.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect

        with TestClient(_mounted_app(prefix="")) as client:  # type: ignore[arg-type]
            page = client.get("/")
            asset = client.get("/novnc/core/rfb.js")

        assert page.status_code == 200, "the page is not served when the router is mounted at the root"
        assert "Human handover" in page.text
        assert asset.status_code == 200, "the vendored client is not served when mounted at the root"

        # And the SOCKET at the root, which is the third leg of "the whole flow". The page and the
        # client are static; the WebSocket is the one whose path resolution actually changes shape
        # when there is no prefix, and asserting only the two static ones while the docstring
        # claimed the whole flow would have been the same overclaim this test exists to remove.
        checked: list[str] = []
        with TestClient(_mounted_app(checked=checked, prefix="")) as client:  # type: ignore[arg-type]
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect("/ws", subprotocols=["binary", "hitl-token.wrong"]):
                    pytest.fail("a wrong token opened the display at the root")
        assert checked == ["wrong"], "the upgrade never reached the WebSocket route at the root"

    def test_the_novnc_client_the_page_imports_is_served(self) -> None:
        """The page's own import, resolved the way a browser resolves it.

        Asserted by reading the import out of the page rather than by naming the path here: a
        constant in a test would keep passing after somebody changed the page, which is the
        version-skew this package vendors the tree to prevent.
        """
        from fastapi.testclient import TestClient

        from threetears.scrape.operator import OPERATOR_PAGE

        page = OPERATOR_PAGE.read_text()
        match = re.search(r'import RFB from "(\./[^"]+)"', page)
        assert match, "the page no longer imports RFB from a relative path"
        resolved = urljoin(f"http://testserver{_PREFIX}/", match.group(1))

        with TestClient(_mounted_app()) as client:  # type: ignore[arg-type]
            response = client.get(resolved)
        assert response.status_code == 200, f"the page imports {match.group(1)} and nothing serves it"
        assert "RFB" in response.text

    def test_the_static_client_tree_does_not_shadow_the_websocket(self) -> None:
        """A mount matches everything beneath it, including an upgrade it cannot handle.

        ``StaticFiles`` handed a WebSocket dies on its own ``assert scope["type"] == "http"``,
        which reaches the operator as a 500 on connect and a client that says only "Failed to
        connect to server" -- indistinguishable from a dead display. The sidecar hit exactly
        that, because there the client tree was the PARENT of the socket.

        What prevents it here is the SHAPE rather than the order: the tree is mounted beside the
        routes rather than over them, so there is nothing of theirs it could claim. Moving the
        mount to the router's root fails this test, which is the point of it -- registration
        order does not, and must not, be what correctness rests on.

        Refusing the upgrade on a bad token is the WebSocket route answering, which a mount that
        had claimed the path could not do.
        """
        from fastapi.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect

        checked: list[str] = []
        with TestClient(_mounted_app(checked=checked)) as client:  # type: ignore[arg-type]
            with pytest.raises(WebSocketDisconnect) as refused:
                with client.websocket_connect(f"{_PREFIX}/ws", subprotocols=["binary", "hitl-token.wrong"]):
                    pytest.fail("a wrong token opened the display")

        # The close code alone proves nothing: a route that never runs is ALSO closed 1008, by
        # FastAPI's own request validation. What proves the WebSocket route answered is that our
        # authorizer was consulted, with the token the client actually offered.
        assert checked == ["wrong"], (
            "the upgrade never reached the WebSocket route, so either the static mount claimed it "
            "or the framework rejected it before the handler ran"
        )
        assert refused.value.code == 1008

    def test_a_refused_operator_never_causes_the_display_to_be_resolved(self) -> None:
        """Order, not just outcome: refused at the door rather than after reaching for a display.

        Resolving the display is the first thing done on a caller's behalf, and in a real
        deployment it is a step towards an RFB socket in the pod. Doing it before deciding
        whether the caller may have one lets an unauthenticated request cause work against the
        display, which is the thing it must not be able to do. Watching the injected
        collaborator is what makes the ORDER observable -- the outcome alone is identical either
        way, which is how this passed once already.
        """
        from fastapi.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect

        resolved: list[str] = []
        checked: list[str] = []
        with TestClient(_mounted_app(resolved, checked)) as client:  # type: ignore[arg-type]
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect(f"{_PREFIX}/ws", subprotocols=["binary"]):
                    pytest.fail("an upgrade carrying no token reached the display")

        assert checked == [""], "the handler did not run, so this proves nothing about ordering"
        assert resolved == [], "the display was resolved on behalf of a caller who was then refused"

    def test_an_authorised_operator_gets_an_accepted_stream_from_their_display(self) -> None:
        """The counterpart, and it has to go all the way through the handshake.

        An earlier version asserted only that the display was RESOLVED, which happens before
        ``accept``. Accepting is wrapped in a broad catch that returns quietly, so a
        systematically failing handshake would have kept that assertion green -- the same class
        of hole as asserting on a close code the framework can also produce. This connects to a
        real listening socket, completes the upgrade, and checks the negotiated subprotocol is
        ``binary`` and NOT the token-bearing one, which would put the credential in a response
        header.
        """
        from fastapi.testclient import TestClient

        resolved: list[str] = []
        with _quiet_listener() as port:
            with TestClient(_mounted_app(resolved, rfb_port=port)) as client:  # type: ignore[arg-type]
                with client.websocket_connect(f"{_PREFIX}/ws", subprotocols=["binary", "hitl-token.good"]) as stream:
                    negotiated = stream.accepted_subprotocol

        assert resolved == ["session-1"], "an authorised operator did not reach the display they hold"
        assert negotiated == "binary", (
            f"the server echoed {negotiated!r}; echoing the token-bearing subprotocol would put a "
            f"live credential into a response header"
        )


class _Captured(logging.Handler):
    """Collect records emitted anywhere, so a structured ``extra`` survives to be asserted on.

    Not `caplog`, and the honest reason is narrower than the first version of this docstring
    claimed. That version said ``threetears.observe``'s handlers might not propagate to the root
    logger caplog attaches to -- which is not a real distinction: ``get_logger`` never touches
    ``propagate``, so caplog would see these records exactly as this handler does, and would fail
    identically if propagation were ever turned off. Offering that as the reason invented a defect
    in the logging setup to justify a preference.

    The real reason is that this keeps the ``LogRecord`` objects, so a test can read
    ``record.extra_data`` -- the structured payload these assertions are about. ``caplog.records``
    would serve as well; this holds the records without also owning caplog's level handling.
    """

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Keep the record rather than rendering it."""
        self.records.append(record)


class TestTheStreamDoesNotOutliveTheClaimThatEntitledIt:
    """A relay runs for hours, so it is the one thing that must not survive a handover.

    A pod that lost a session otherwise keeps a live browser and a live display in front of an
    operator for the rest of the session's TTL, while the pod that actually owns it serves
    somebody else against the same session id.

    Tested at ``relay_stream`` and at the router separately, because the interesting half is a
    race inside one event loop and ``TestClient`` runs the app in another thread -- an event set
    from the test thread cannot wake a waiter bound to the app's loop, which is a property of the
    harness rather than of the code and would prove nothing either way.
    """

    @staticmethod
    @contextlib.contextmanager
    def _captured(level: int) -> Iterator[list[logging.LogRecord]]:
        """Attach a handler to the root logger for the duration, and hand back what it saw."""
        handler = _Captured()
        root = logging.getLogger()
        previous = root.level
        root.addHandler(handler)
        root.setLevel(level)
        try:
            yield handler.records
        finally:
            root.removeHandler(handler)
            root.setLevel(previous)

    async def test_the_relay_ends_when_the_stop_signal_completes(self) -> None:
        """The mechanism itself: a relay parked on a silent socket still lets go.

        Without the signal riding alongside the pumps, this relay would sit forever -- nothing is
        being written from either end, which is exactly the state an idle operator's session is in
        for most of its life.
        """
        lost = asyncio.Event()
        with _quiet_listener() as port:
            relay = asyncio.create_task(
                relay_stream(lambda _d: asyncio.sleep(0), asyncio.Event().wait, "127.0.0.1", port, until=lost.wait)
            )
            await asyncio.sleep(0.05)
            assert not relay.done(), "the relay ended before the claim was lost"
            lost.set()
            await asyncio.wait_for(relay, timeout=5)

    async def test_a_relay_with_no_stop_signal_still_runs(self) -> None:
        """One pod has no handover to survive and must not need a signal to show a display."""
        with _quiet_listener() as port:
            relay = asyncio.create_task(
                relay_stream(lambda _d: asyncio.sleep(0), asyncio.Event().wait, "127.0.0.1", port)
            )
            await asyncio.sleep(0.05)
            assert not relay.done(), "a relay with no stop signal ended on its own"
            relay.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await relay

    def test_the_router_asks_whether_this_pod_holds_the_session(self) -> None:
        """The seam that was missing: the router had no parameter that could carry a claim.

        ``SessionClaim.until_lost()`` shipped with no consumer, so losing a lease was connected to
        nothing. This asserts the lookup is consulted for the session that was authorised.
        """
        from fastapi.testclient import TestClient

        asked: list[str] = []
        app = _mounted_app(claim=_StubClaim(), with_claim_lookup=True, claim_asked=asked)
        with TestClient(app) as client:  # type: ignore[arg-type]
            with contextlib.suppress(Exception):
                with client.websocket_connect(f"{_PREFIX}/ws", subprotocols=["binary", "hitl-token.good"]):
                    pass

        assert asked == ["session-1"], "the router never asked whether this pod holds the session"

    def test_an_unreachable_display_is_reported_with_enough_to_act_on(self) -> None:
        """The fault path's log line is the only place the session, host and port appear.

        The operator gets 1011 and nothing else, deliberately -- so this line is the whole of the
        diagnosis, and dropping any of the three turns "the display could not be reached" into a
        message nobody can act on.

        The close that follows it is suppressed in the route, for a case this harness cannot
        reproduce: if the client has already gone -- one of the ways a fault arrives -- the close
        raises, escapes the handler, and buries this warning under a framework traceback carrying
        no session id. ``TestClient``'s client never departs, so that branch is defensive and
        stated rather than tested. What IS tested is that the diagnosis is complete and the socket
        is told.
        """
        from fastapi.testclient import TestClient

        with self._captured(logging.WARNING) as records:
            # Port 1 on loopback: reserved, and nothing listens there, which is the real shape of
            # a display that cannot be reached.
            with TestClient(_mounted_app(rfb_port=1)) as client:  # type: ignore[arg-type]
                with client.websocket_connect(f"{_PREFIX}/ws", subprotocols=["binary", "hitl-token.good"]) as stream:
                    # `receive`, not `receive_bytes`: the raw message carries the close frame, where
                    # the typed readers turn it into an exception and lose the code.
                    ended = stream.receive()

        assert ended["type"] == "websocket.close", f"the socket was not closed on a fault: {ended}"
        assert ended["code"] == 1011, "an unreachable display was not reported as a server-side fault"
        reported = [r for r in records if "could not be reached" in r.getMessage()]
        assert reported, "an unreachable display produced no warning at all"
        data = getattr(reported[0], "extra_data", {})
        assert data.get("session_id") == "session-1", f"the warning names no session: {data}"
        assert data.get("host") == "127.0.0.1" and data.get("port") == 1, (
            f"the warning does not say which endpoint failed, so nobody can act on it: {data}"
        )

    def test_a_pod_holding_no_claim_refuses_before_opening_anything(self) -> None:
        """Serving a display this pod does not own is worse than refusing to serve it.

        The token is perfectly valid here -- authorization is not what fails. What fails is this
        pod's entitlement to answer, which is a different question and needs its own refusal.
        """
        from fastapi.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect

        app = _mounted_app(claim=None, with_claim_lookup=True)
        with TestClient(app) as client:  # type: ignore[arg-type]
            with pytest.raises(WebSocketDisconnect) as refused:
                with client.websocket_connect(f"{_PREFIX}/ws", subprotocols=["binary", "hitl-token.good"]):
                    pytest.fail("a pod with no claim served the display anyway")
        assert refused.value.code == 1008


class TestThePageKeepsItsTwoInvisibleGuarantees:
    """The page is code nothing else checks: no linter reads it, no type checker sees it."""

    def test_the_operator_arrives_connected_and_the_desktop_fits_their_screen(self) -> None:
        """Two properties a summoned operator notices immediately if they are missing.

        ``scaleViewport`` rather than a remote resize, because Xvfb creates ONE mode at startup
        and cannot follow a client viewport -- ``resize=remote`` is accepted and silently does
        nothing, which is how an operator ended up scrolling a fixed 1920x1080 desktop to reach
        a taskbar. And the client is constructed on load, so somebody summoned to clear one
        challenge arrives at the display rather than at a settings sidebar.
        """
        from threetears.scrape.operator import OPERATOR_PAGE

        page = OPERATOR_PAGE.read_text()
        assert "scaleViewport = true" in page, (
            "without a scaling mode the operator scrolls a fixed desktop; remote resizing is "
            "silently inert here because Xvfb has a single mode and cannot resize"
        )
        assert "new RFB(" in page, "the page does not connect on load, so an operator arrives at nothing"

    def test_the_page_offers_the_subprotocol_the_server_actually_reads(self) -> None:
        """The prefix is a contract between a Python constant and a string in a JS file.

        Nothing links them: no import, no build step, no type. Change one and the operator gets
        "not authorised" with a perfectly valid token, and the two halves each look right on
        their own.
        """
        from threetears.scrape.operator import TOKEN_SUBPROTOCOL_PREFIX, OPERATOR_PAGE

        page = OPERATOR_PAGE.read_text()
        assert f"`{TOKEN_SUBPROTOCOL_PREFIX}${{token}}`" in page, (
            f"the page does not offer a {TOKEN_SUBPROTOCOL_PREFIX!r}-prefixed subprotocol, so the "
            f"server will read no token from it"
        )

    def test_the_token_comes_from_the_fragment_and_never_the_query_string(self) -> None:
        """A fragment is never sent to a server, so it is in no log, referrer or proxy trace.

        Read from the query string instead -- a one-word edit -- and a live session credential
        is written to all three, on every request, forever.
        """
        from threetears.scrape.operator import OPERATOR_PAGE

        page = OPERATOR_PAGE.read_text()
        assert "location.hash" in page, "the token is no longer read from the URL fragment"
        assert "location.search" not in page, "the page reads the query string, which is logged everywhere"

    def test_no_url_the_page_emits_is_absolute(self) -> None:
        """An absolute path resolves against the origin root, not against the mount point.

        It works perfectly in every local test and fails only in the deployment nobody can test
        from here, which is why this is greppable rather than left to review.
        """
        from threetears.scrape.operator import OPERATOR_PAGE

        page = OPERATOR_PAGE.read_text()
        for absolute in ('href="/', 'src="/', 'from "/', "from '/", 'new URL("/', "new URL('/"):
            assert absolute not in page, (
                f"the page contains an absolute reference ({absolute!r}), which resolves against "
                f"the origin root and breaks the moment this router is mounted under a prefix"
            )

    def test_the_page_is_not_inside_the_vendored_tree(self) -> None:
        """MPL-2.0 requires a modification to be marked, so nothing of ours may look vendored.

        A page living inside ``novnc/`` would read as a noVNC file we changed and did not say
        so. Keeping it a sibling makes that impossible rather than merely untrue today.
        """
        from threetears.scrape.operator import OPERATOR_ASSETS, OPERATOR_PAGE

        assert OPERATOR_PAGE.parent == OPERATOR_ASSETS, "the operator page moved out of the assets root"
        assert (OPERATOR_ASSETS / "novnc") not in OPERATOR_PAGE.parents, (
            "the operator page sits inside the vendored noVNC tree, where it reads as a modified noVNC file"
        )
