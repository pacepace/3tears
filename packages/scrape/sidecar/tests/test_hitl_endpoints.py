"""Contract tests for the three ``/v1/hitl/vnc`` endpoints.

The lifecycle itself is covered in ``test_hitl_lifecycle.py`` against real stub processes.
What is left, and what operator verification could not cover because a human only ever walks
the happy path, is the HTTP surface: the shapes a caller parses, and the failure branch that
turns a ``VncUnavailable`` into a 503 rather than a 500 with a traceback.

``main`` imports nodriver at module scope, so this shares ``test_render_contract.py``'s
approach of importing it directly -- the sidecar's conftest puts it on the path.
"""

from __future__ import annotations

from typing import Any

import httpx
import main
import pytest
import hitl
from hitl import VncSession, VncUnavailable


# parity-exempt: stands in for this sidecar's own hitl.VncLifecycle, mirroring the four members main calls (start/stop/health/display); the sidecar is a standalone deployable whose modules are never installed in the workspace venv, so a parity-with marker cannot resolve there -- same reason as this suite's nodriver stubs
class _FakeLifecycle:
    """Stands in for ``hitl.VncLifecycle`` at the endpoint boundary.

    Mirrors the four members ``main`` actually calls (``start``/``stop``/``health``/
    ``display``). The processes themselves are somebody else's test.
    """

    def __init__(self, *, explode: bool = False) -> None:
        self._explode = explode
        self._running = False
        self.stops = 0

    @property
    def display(self) -> str:
        return ":99"

    @property
    def web_port(self) -> int:
        return 6080

    @property
    def client_path(self) -> str:
        return f"/{hitl.NOVNC_PAGE}?path=websockify"

    def health(self) -> bool:
        return self._running

    async def start(self) -> VncSession:
        if self._explode:
            raise VncUnavailable("x11vnc is not installed in this container")
        self._running = True
        return VncSession(web_port=6080, display=":99", path=f"/{hitl.NOVNC_PAGE}?path=websockify")

    async def stop(self) -> None:
        self.stops += 1
        self._running = False


async def _call(method: str, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://sidecar") as client:
        return await client.request(method, path)


@pytest.fixture()
def fake_vnc(monkeypatch: pytest.MonkeyPatch) -> _FakeLifecycle:
    fake = _FakeLifecycle()
    monkeypatch.setattr(main, "_vnc", fake)
    return fake


async def test_starting_returns_where_to_point_a_browser(fake_vnc: _FakeLifecycle) -> None:
    """The response IS the contract: a caller has to know the port and the path."""
    r = await _call("POST", "/v1/hitl/vnc")
    assert r.status_code == 200
    body: dict[str, Any] = r.json()
    assert body["display"] == ":99"
    assert body["web_port"] == 6080
    assert body["path"].startswith(f"/{hitl.NOVNC_PAGE}")


async def test_status_reports_running_only_once_started(fake_vnc: _FakeLifecycle) -> None:
    assert (await _call("GET", "/v1/hitl/vnc")).json() == {"running": False, "display": ":99"}
    await _call("POST", "/v1/hitl/vnc")
    assert (await _call("GET", "/v1/hitl/vnc")).json() == {"running": True, "display": ":99"}


async def test_deleting_stops_it_and_says_so(fake_vnc: _FakeLifecycle) -> None:
    await _call("POST", "/v1/hitl/vnc")
    r = await _call("DELETE", "/v1/hitl/vnc")
    assert r.status_code == 200
    assert r.json() == {"running": False}
    assert fake_vnc.stops == 1


async def test_an_unavailable_vnc_path_is_a_503_not_a_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    """The branch operator verification structurally cannot reach.

    A human checking this feature walks the success path; nobody hand-breaks an image to see
    what a caller gets back. Unhandled, a ``VncUnavailable`` is a 500 with a traceback, which
    tells a queue "the sidecar is broken" when the truth is "this container has no VNC
    support" -- a distinction that decides whether anyone gets paged.
    """
    monkeypatch.setattr(main, "_vnc", _FakeLifecycle(explode=True))
    r = await _call("POST", "/v1/hitl/vnc")
    assert r.status_code == 503
    assert "not installed" in r.json()["error"]


# --------------------------------------------------------------------------
# Session endpoints. The manager's own behaviour is covered in
# test_hitl_session.py; what is only reachable here is the HTTP surface: the two
# header forms a token may arrive in, and the status-code mapping a caller
# branches on.
# --------------------------------------------------------------------------


# parity-exempt: stands in for this sidecar's own hitl.SessionManager, mirroring the members main calls (open/authorize/open_tab/complete_tab/close/current/vnc); the sidecar is a standalone deployable never installed in the workspace venv, so a parity-with marker cannot resolve there
class _FakeSessions:
    """A session manager whose outcomes are scripted, to drive each status code."""

    def __init__(self) -> None:
        self.vnc = _FakeLifecycle()
        self._session: Any = None
        self.open_error: Exception | None = None
        self.tab_error: Exception | None = None
        self.expired = False
        self.authorized_with: list[tuple[str, str]] = []
        self.session_states: list[Any] = []

    def current(self) -> Any:
        return self._session

    def owns_display(self, **_kw: Any) -> bool:
        return self._session is not None and not self.expired

    def authorize(self, session_id: str, token: str, **_kw: Any) -> Any:
        self.authorized_with.append((session_id, token))
        from hitl import SessionNotFound

        if self._session is None or token != self._session.token or session_id != self._session.session_id:
            raise SessionNotFound("no such session")
        if self.expired:
            raise SessionNotFound("no such session")
        return self._session

    async def open(self, **_kw: Any) -> Any:
        if self.open_error is not None:
            raise self.open_error
        from hitl import HitlSession

        self._session = HitlSession(session_id="sid", token="tok", expires_at=99.0, max_slots=2)
        await self.vnc.start()
        return self._session

    async def open_tab(
        self,
        session: Any,
        *,
        target_id: str,
        url: str,
        nav_steps: Any = None,
        # Recorded, not ignored: the endpoint forwarding this is the whole point of the
        # capability, and it was already shipped once dropping it silently. A fake that does
        # not accept it turns that bug into a 502 from the broad handler rather than a
        # failed assertion, which is how it stayed hidden.
        session_state: Any = None,
    ) -> Any:
        self.session_states.append(session_state)
        if self.tab_error is not None:
            raise self.tab_error
        from hitl import HitlTab

        tab = HitlTab(tab_id="t1", target_id=target_id, url=url, tab=None, context_id="ctx", opened_at=0.0)
        session.tabs[tab.tab_id] = tab
        return tab

    async def complete_tab(self, session: Any, tab_id: str) -> Any:
        from hitl import SessionNotFound

        tab = session.tabs.pop(tab_id, None)
        if tab is None:
            raise SessionNotFound("no such tab")
        return tab

    async def close(self, session: Any = None) -> None:
        self._session = None
        await self.vnc.stop()


@pytest.fixture()
def fake_sessions(monkeypatch: pytest.MonkeyPatch) -> _FakeSessions:
    fake = _FakeSessions()
    monkeypatch.setattr(main, "_sessions", fake)
    monkeypatch.setattr(main, "_vnc", fake.vnc)
    return fake


async def _open(fake: _FakeSessions) -> tuple[str, str]:
    r = await _call("POST", "/v1/hitl/session")
    assert r.status_code == 200
    return r.json()["session_id"], r.json()["token"]


async def test_opening_returns_the_token_once_and_says_where_the_display_is(
    fake_sessions: _FakeSessions,
) -> None:
    r = await _call("POST", "/v1/hitl/session")
    body = r.json()
    assert r.status_code == 200
    assert body["token"] == "tok"
    assert body["max_slots"] == 2
    assert body["vnc_path"].startswith(f"/{hitl.NOVNC_PAGE}")


async def test_a_second_session_is_409_not_a_queue(fake_sessions: _FakeSessions) -> None:
    """A caller has to be able to tell "busy" from "broken" to know whether to retry."""
    from hitl import SessionUnavailable

    await _open(fake_sessions)
    fake_sessions.open_error = SessionUnavailable("a session is already open")
    r = await _call("POST", "/v1/hitl/session")
    assert r.status_code == 409


async def test_no_display_is_503_not_409(fake_sessions: _FakeSessions) -> None:
    """Different cause, different code: 409 says try later, 503 says this container is broken."""
    from hitl import VncUnavailable

    fake_sessions.open_error = VncUnavailable("x11vnc is not installed")
    r = await _call("POST", "/v1/hitl/session")
    assert r.status_code == 503


async def test_the_token_is_accepted_in_either_header_form(fake_sessions: _FakeSessions) -> None:
    """Bearer for anything that speaks HTTP normally, X-HITL-Token for anything that does not."""
    sid, tok = await _open(fake_sessions)
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://sidecar") as client:
        bearer = await client.get(f"/v1/hitl/session/{sid}", headers={"Authorization": f"Bearer {tok}"})
        custom = await client.get(f"/v1/hitl/session/{sid}", headers={"X-HITL-Token": tok})
        lower = await client.get(f"/v1/hitl/session/{sid}", headers={"Authorization": f"bearer {tok}"})
    assert bearer.status_code == 200
    assert custom.status_code == 200
    assert lower.status_code == 200, "the scheme is case-insensitive per RFC 7235"


async def test_a_missing_or_wrong_token_is_404(fake_sessions: _FakeSessions) -> None:
    sid, _ = await _open(fake_sessions)
    assert (await _call("GET", f"/v1/hitl/session/{sid}")).status_code == 404
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://sidecar") as client:
        r = await client.get(f"/v1/hitl/session/{sid}", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 404


async def test_a_tab_that_will_not_open_is_502_and_leaves_the_session_alive(
    fake_sessions: _FakeSessions,
) -> None:
    """One target's nav-step replay failing is that target's problem, not the operator's session.

    A 500 would say the sidecar is broken; the truth is this one URL did not come up, and the
    other tabs are fine.
    """
    sid, tok = await _open(fake_sessions)
    fake_sessions.tab_error = RuntimeError("nav step timed out")
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://sidecar") as client:
        r = await client.post(
            f"/v1/hitl/session/{sid}/tab",
            headers={"Authorization": f"Bearer {tok}"},
            json={"target_id": "a", "url": "https://a.example"},
        )
        still = await client.get(f"/v1/hitl/session/{sid}", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 502
    assert still.status_code == 200, "one failed target tore down the whole session"


async def test_slot_exhaustion_is_409_at_the_http_boundary(fake_sessions: _FakeSessions) -> None:
    from hitl import SessionUnavailable

    sid, tok = await _open(fake_sessions)
    fake_sessions.tab_error = SessionUnavailable("all 2 slots are occupied")
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://sidecar") as client:
        r = await client.post(
            f"/v1/hitl/session/{sid}/tab",
            headers={"Authorization": f"Bearer {tok}"},
            json={"target_id": "a", "url": "https://a.example"},
        )
    assert r.status_code == 409, "a full session must be distinguishable from a broken one"


async def test_completing_an_unknown_tab_is_404(fake_sessions: _FakeSessions) -> None:
    sid, tok = await _open(fake_sessions)
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://sidecar") as client:
        r = await client.post(f"/v1/hitl/session/{sid}/tab/nope/complete", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 404


async def test_the_bare_vnc_endpoints_refuse_while_a_session_owns_the_display(
    fake_sessions: _FakeSessions,
) -> None:
    """Two owners of one display is a state divergence, not a convenience.

    A bare POST would bring the display up outside any TTL; a bare DELETE would kill it under
    a live session that then goes on reporting itself open, with nothing for its operator to
    look at. Both refuse and name the session API.
    """
    await _open(fake_sessions)

    started = await _call("POST", "/v1/hitl/vnc")
    stopped = await _call("DELETE", "/v1/hitl/vnc")

    assert started.status_code == 409
    assert "session" in started.json()["error"]
    assert stopped.status_code == 409
    assert fake_sessions.vnc.health(), "a bare DELETE stopped the display under a live session"


async def test_the_bare_vnc_endpoints_still_work_with_no_session(fake_sessions: _FakeSessions) -> None:
    """The diagnostic case they exist for: look at the unattended browser, no session involved."""
    assert (await _call("POST", "/v1/hitl/vnc")).status_code == 200
    assert (await _call("DELETE", "/v1/hitl/vnc")).status_code == 200


async def test_deleting_a_session_closes_it(fake_sessions: _FakeSessions) -> None:
    """The one endpoint from the previous round's list that still had no HTTP test."""
    sid, tok = await _open(fake_sessions)
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://sidecar") as client:
        r = await client.delete(f"/v1/hitl/session/{sid}", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    assert r.json()["closed"] is True
    assert fake_sessions.current() is None
    assert not fake_sessions.vnc.health(), "the display outlived the session that owned it"


async def test_deleting_a_session_without_the_token_is_404(fake_sessions: _FakeSessions) -> None:
    sid, _ = await _open(fake_sessions)
    assert (await _call("DELETE", f"/v1/hitl/session/{sid}")).status_code == 404
    assert fake_sessions.current() is not None, "an unauthenticated DELETE closed the session"


async def test_an_expired_session_can_still_be_released(monkeypatch: pytest.MonkeyPatch) -> None:
    """The trap the last two fixes built between them.

    `authorize` refuses an expired session, so the session API cannot tear one down. If the
    bare VNC teardown ALSO refused whenever a session object existed, then past the TTL no
    HTTP caller could release the display at all -- leaving only the reaper, which is the very
    task the request-path TTL check exists because it might be dead. The remaining escape
    would have been opening another session on the display in order to shut it down.

    So the bare teardown asks whether a session still OWNS the display, which an expired one
    does not, and releases it properly rather than stopping the processes underneath a
    tracked session.
    """
    fake = _FakeSessions()
    monkeypatch.setattr(main, "_sessions", fake)
    monkeypatch.setattr(main, "_vnc", fake.vnc)
    await _call("POST", "/v1/hitl/session")
    assert fake.vnc.health()

    fake.expired = True

    blocked = await _call("DELETE", f"/v1/hitl/session/{fake.current().session_id}")
    assert blocked.status_code == 404, "an expired session is still refused by the session API"

    released = await _call("DELETE", "/v1/hitl/vnc")
    assert released.status_code == 200, "an expired session made the display unreleasable"
    assert released.json()["released_expired_session"] is True
    assert not fake.vnc.health()
    assert fake.current() is None


async def test_a_session_closed_mid_navigation_is_409_not_502(fake_sessions: _FakeSessions) -> None:
    """An ordinary race, not a fault.

    A tab opening while its session is torn down is expected under concurrency. Reporting it
    as a bad gateway with a traceback sends someone investigating an incident that is just two
    callers arriving in an unlucky order.
    """
    from hitl import SessionNotFound

    sid, tok = await _open(fake_sessions)
    fake_sessions.tab_error = SessionNotFound("the session was closed while this tab was opening")
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://sidecar") as client:
        r = await client.post(
            f"/v1/hitl/session/{sid}/tab",
            headers={"Authorization": f"Bearer {tok}"},
            json={"target_id": "a", "url": "https://a.example"},
        )
    assert r.status_code == 409


async def test_the_tab_endpoint_forwards_the_session_state_it_accepts(fake_sessions: _FakeSessions) -> None:
    """It shipped once accepting this field and discarding it.

    `HitlTabRequest` declared it, documented it, and the manager implemented applying it --
    and the endpoint called `open_tab` without it. The request returned 200, set no cookies,
    and logged nothing, so every symptom pointed at the browser rather than at one missing
    keyword argument. Nothing tested the forwarding, which is why.
    """
    sid, tok = await _open(fake_sessions)
    state = {"cookies": [{"name": "cf_clearance", "value": "earned", "domain": ".example.gov"}]}
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://sidecar") as client:
        r = await client.post(
            f"/v1/hitl/session/{sid}/tab",
            headers={"Authorization": f"Bearer {tok}"},
            json={"target_id": "a", "url": "https://a.example", "session_state": state},
        )

    assert r.status_code == 200
    assert fake_sessions.session_states == [state], (
        "the endpoint accepted a human's solve and dropped it, so the tab opens unauthenticated"
    )
