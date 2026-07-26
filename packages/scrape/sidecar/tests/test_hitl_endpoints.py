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

    def health(self) -> bool:
        return self._running

    async def start(self) -> VncSession:
        if self._explode:
            raise VncUnavailable("x11vnc is not installed in this container")
        self._running = True
        return VncSession(web_port=6080, display=":99", path="/vnc_lite.html?path=websockify")

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
    assert body["path"].startswith("/vnc_lite.html")


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
