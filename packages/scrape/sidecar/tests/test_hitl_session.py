"""Session lifecycle: slots, TTL, tokens, isolation, teardown.

No browser and no VNC processes. Both are injected, so what runs here is the accounting and
the policy -- which is where this chunk's failures actually live. A slot budget that can be
overspent, a TTL that never fires, a token compared with ``==``, or a teardown that drops the
display but leaks contexts are all bugs that a working browser would hide rather than reveal.

The one thing deliberately NOT faked away is concurrency: several of these drive the manager
from parallel tasks, because "read the free-slot count, then take one" is a read-then-write
and the bug it invites only appears when two callers interleave.
"""

from __future__ import annotations

import asyncio
from typing import Any

import hitl
import pytest
from hitl import HitlSession, HitlTab, SessionManager, SessionNotFound, SessionUnavailable, VncSession


# parity-exempt: stands in for this sidecar's own hitl.VncLifecycle, mirroring the members SessionManager calls (start/stop/health); the sidecar is a standalone deployable never installed in the workspace venv, so a parity-with marker cannot resolve there
class _FakeVnc:
    """A display lifecycle that records rather than spawns."""

    def __init__(self) -> None:
        self.starts = 0
        self.stops = 0
        self._running = False

    @property
    def display(self) -> str:
        return ":99"

    def health(self) -> bool:
        return self._running

    async def start(self) -> VncSession:
        self.starts += 1
        self._running = True
        return VncSession(web_port=6080, display=":99", path=f"/{hitl.NOVNC_PAGE}?path=websockify")

    async def stop(self) -> None:
        self.stops += 1
        self._running = False


# parity-exempt: stands in for nodriver's third-party Browser, whose only surface the session layer touches is being passed to the isolated-tab helper; nodriver is AGPL-isolated to this sidecar and never installed in the workspace venv, so a parity-with marker cannot resolve there
class _FakeBrowser:
    """Enough of a browser to hand out distinguishable contexts."""

    def __init__(self) -> None:
        self.contexts: list[str] = []
        self.disposed: list[str] = []
        self.serial = 0


@pytest.fixture()
def manager(monkeypatch: pytest.MonkeyPatch) -> SessionManager:
    """A manager with the browser and display faked, and CDP calls intercepted."""
    browser = _FakeBrowser()

    async def _fake_open(_browser: Any, url: str, nav_steps: Any, session_state: Any = None) -> tuple[Any, Any]:
        # Yields, and that is load-bearing rather than incidental. The real helper does
        # several CDP round trips, so it suspends here; a fake that returns without ever
        # awaiting lets every task run to completion uninterrupted, and the read-then-write
        # race this fixture exists to expose becomes structurally unreachable. Verified by
        # removing the manager's lock and watching the concurrency test fail.
        await asyncio.sleep(0)
        browser.serial += 1
        ctx = f"ctx-{browser.serial}"
        browser.contexts.append(ctx)
        return object(), ctx

    async def _fake_dispose(_browser: Any, context_id: Any) -> None:
        browser.disposed.append(context_id)

    monkeypatch.setattr(hitl, "_open_isolated", _fake_open)
    monkeypatch.setattr(hitl, "_dispose_context", _fake_dispose)
    mgr = SessionManager(vnc=_FakeVnc(), browser_provider=lambda: browser, max_slots=2, ttl_seconds=100.0)
    mgr.browser = browser  # type: ignore[attr-defined]  -- test handle
    return mgr


async def test_opening_a_session_brings_up_the_display_and_mints_a_token(manager: SessionManager) -> None:
    """The token is returned exactly once, by this call, and is the bearer for everything after."""
    session = await manager.open(now=1000.0)

    assert manager.vnc.health()  # type: ignore[union-attr]
    assert session.expires_at == 1100.0
    assert session.max_slots == 2
    assert len(session.token) >= 32, "a short token is a guessable one"
    assert session.token != session.session_id, "the id is not a secret and must not double as one"


async def test_a_second_session_is_refused_rather_than_queued(manager: SessionManager) -> None:
    """One display means one operator, and queueing would hold an HTTP call open for hours."""
    await manager.open(now=1000.0)
    with pytest.raises(SessionUnavailable, match="already open"):
        await manager.open(now=1001.0)


async def test_an_expired_session_does_not_block_the_next_operator(manager: SessionManager) -> None:
    """Reaping is a background job, so `open` must not depend on it having run yet.

    Otherwise an operator arriving in the window between expiry and the next reaper tick is
    told the display is busy by a session nobody is using.
    """
    first = await manager.open(now=1000.0)
    second = await manager.open(now=2000.0)
    assert second.session_id != first.session_id


async def test_slots_are_a_ceiling_not_a_suggestion(manager: SessionManager) -> None:
    """A target holds its slot until a human is done, including while it sits being slow."""
    session = await manager.open(now=1000.0)
    await manager.open_tab(session, target_id="a", url="https://a.example")
    await manager.open_tab(session, target_id="b", url="https://b.example")
    assert session.free_slots() == 0

    with pytest.raises(SessionUnavailable, match="slots are occupied"):
        await manager.open_tab(session, target_id="c", url="https://c.example")
    assert len(session.tabs) == 2, "a refused target still took a tab"


async def test_concurrent_tab_opens_cannot_overspend_the_budget(manager: SessionManager) -> None:
    """The bug a serial test cannot see.

    "Read the free-slot count, then take one" is a read-then-write. Two callers that both read
    "one slot free" both take it, and a bounded working set quietly stops being bounded --
    which matters because the bound is what stops one operator's browser accumulating tabs
    until the container dies.
    """
    session = await manager.open(now=1000.0)
    results = await asyncio.gather(
        *(manager.open_tab(session, target_id=f"t{i}", url=f"https://{i}.example") for i in range(6)),
        return_exceptions=True,
    )

    opened = [r for r in results if not isinstance(r, BaseException)]
    refused = [r for r in results if isinstance(r, SessionUnavailable)]
    assert len(opened) == 2, f"the slot budget was overspent: {len(opened)} tabs opened into 2 slots"
    assert len(refused) == 4
    assert len(session.tabs) == 2


async def test_each_tab_gets_its_own_context(manager: SessionManager) -> None:
    """This chunk's acceptance criterion, at the unit level.

    Isolation is per TAB, not per session: a shared context would hand the second target the
    credentials a human just earned on the first, which is precisely the leak a walled site
    should never be handed.
    """
    session = await manager.open(now=1000.0)
    a = await manager.open_tab(session, target_id="a", url="https://a.example")
    b = await manager.open_tab(session, target_id="b", url="https://b.example")

    assert a.context_id != b.context_id, "two targets shared one browser context"


async def test_completing_a_tab_frees_its_slot_and_drops_its_context(manager: SessionManager) -> None:
    """The slot has to come back, or the budget is a one-way countdown to a dead session."""
    session = await manager.open(now=1000.0)
    a = await manager.open_tab(session, target_id="a", url="https://a.example")
    await manager.open_tab(session, target_id="b", url="https://b.example")

    await manager.complete_tab(session, a.tab_id)

    assert session.free_slots() == 1
    assert a.context_id in manager.browser.disposed  # type: ignore[attr-defined]
    # And the freed slot is genuinely reusable, not merely counted.
    await manager.open_tab(session, target_id="c", url="https://c.example")
    assert session.free_slots() == 0


async def test_completing_an_unknown_tab_is_refused(manager: SessionManager) -> None:
    session = await manager.open(now=1000.0)
    with pytest.raises(SessionNotFound):
        await manager.complete_tab(session, "not-a-tab")


async def test_the_reaper_closes_an_abandoned_session_and_drops_every_context(manager: SessionManager) -> None:
    """A hard TTL, because an operator who walks away leaves a live authenticated browser.

    An idle timeout cannot tell "walked away" from "reading carefully", so the only bound that
    does not depend on guessing intent is a ceiling.
    """
    session = await manager.open(now=1000.0)
    await manager.open_tab(session, target_id="a", url="https://a.example")
    await manager.open_tab(session, target_id="b", url="https://b.example")

    assert await manager.reap(now=1099.0) is False, "reaped a session that was still live"
    assert await manager.reap(now=1101.0) is True

    assert manager.current() is None
    assert len(manager.browser.disposed) == 2, "the reaped session leaked browser contexts"  # type: ignore[attr-defined]
    assert not manager.vnc.health()  # type: ignore[union-attr]


async def test_an_unknown_id_or_token_is_refused_and_says_nothing_useful(manager: SessionManager) -> None:
    """One error for both, so a wrong token cannot confirm a right id.

    Distinguishing them would turn the id into an oracle, and the id is the thing a guesser is
    working towards.
    """
    session = await manager.open(now=1000.0)

    with pytest.raises(SessionNotFound) as wrong_token:
        manager.authorize(session.session_id, "not-the-token", now=1001.0)
    with pytest.raises(SessionNotFound) as wrong_id:
        manager.authorize("not-the-session", session.token, now=1001.0)

    assert str(wrong_token.value) == str(wrong_id.value), (
        "the two failures are distinguishable, so a wrong token confirms a correct id"
    )
    assert manager.authorize(session.session_id, session.token, now=1001.0) is session


async def test_authorizing_against_no_session_is_refused(manager: SessionManager) -> None:
    with pytest.raises(SessionNotFound):
        manager.authorize("anything", "anything", now=1000.0)


async def test_an_expired_session_is_refused_without_waiting_for_the_reaper(
    manager: SessionManager,
) -> None:
    """A hard TTL that depends on a background task staying alive is not a hard TTL.

    The reaper polls, so between expiry and the next tick there is a window in which a passed
    ceiling is still honoured -- and the reaper is a task that can die, after which nothing
    else enforces it at all. Checking on the request path makes the reaper an optimisation
    that frees the display promptly, rather than the mechanism.
    """
    session = await manager.open(now=1000.0)
    assert manager.authorize(session.session_id, session.token, now=1099.0) is session

    with pytest.raises(SessionNotFound):
        manager.authorize(session.session_id, session.token, now=1100.0)
    with pytest.raises(SessionNotFound):
        manager.authorize(session.session_id, session.token, now=9999.0)


async def test_an_expired_session_is_refused_indistinguishably_from_an_unknown_one(
    manager: SessionManager,
) -> None:
    """An expired id is still an id that existed, so confirming it leaks the target of a guess."""
    session = await manager.open(now=1000.0)
    with pytest.raises(SessionNotFound) as expired:
        manager.authorize(session.session_id, session.token, now=2000.0)
    with pytest.raises(SessionNotFound) as unknown:
        manager.authorize("never-existed", session.token, now=1001.0)
    assert str(expired.value) == str(unknown.value)


async def test_a_non_ascii_id_or_token_is_refused_rather_than_crashing(manager: SessionManager) -> None:
    """Both operands come straight from a caller, and ``compare_digest`` rejects non-ASCII str.

    The id is a UTF-8 path segment and the token is a header, so one accented character would
    otherwise turn a refusal into an unhandled ``TypeError`` -- a 500 where a 404 belongs,
    which is both the wrong answer and a signal that the input reached further than it should.
    """
    session = await manager.open(now=1000.0)
    for bad in ("sessión", "toke\u00e9n", "\U0001f600"):
        with pytest.raises(SessionNotFound):
            manager.authorize(bad, session.token, now=1001.0)
        with pytest.raises(SessionNotFound):
            manager.authorize(session.session_id, bad, now=1001.0)


async def test_a_tab_that_fails_to_open_does_not_keep_its_slot(
    manager: SessionManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reservation must not outlive the attempt it was reserving for.

    Slots are reserved before the navigation so the budget stays honest while the lock is
    released. If a failed navigation left its reservation behind, a target that could not be
    opened would silently cost the operator a slot for the rest of the session.
    """
    session = await manager.open(now=1000.0)

    async def _explode(_browser: Any, _url: str, _nav: Any, _state: Any = None) -> tuple[Any, Any]:
        await asyncio.sleep(0)
        raise RuntimeError("navigation failed")

    monkeypatch.setattr(hitl, "_open_isolated", _explode)
    with pytest.raises(RuntimeError, match="navigation failed"):
        await manager.open_tab(session, target_id="doomed", url="https://x.example")

    assert session.free_slots() == session.max_slots, "a failed open kept its slot"
    assert session.tabs == {}


async def test_a_slow_tab_open_does_not_block_the_session(manager: SessionManager, monkeypatch) -> None:
    """The lock must not span the navigation, or the model this session documents is false.

    "Backgrounding a slow target still holds its slot" only works if a slow target does not
    also hold the manager. Holding the lock across the CDP work would serialise every open
    behind the slowest, defer the reaper, and hang a teardown.
    """
    session = await manager.open(now=1000.0)
    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow(_browser: Any, _url: str, _nav: Any, _state: Any = None) -> tuple[Any, Any]:
        started.set()
        await release.wait()
        return object(), "ctx-slow"

    monkeypatch.setattr(hitl, "_open_isolated", _slow)
    task = asyncio.create_task(manager.open_tab(session, target_id="slow", url="https://slow.example"))
    await asyncio.wait_for(started.wait(), timeout=1.0)

    # While that one is mid-navigation, the session must still answer.
    state = await asyncio.wait_for(asyncio.to_thread(lambda: session.free_slots()), timeout=1.0)
    assert state == session.max_slots - 1, "the in-flight tab did not reserve its slot"
    assert await asyncio.wait_for(manager.reap(now=1001.0), timeout=1.0) is False, (
        "the reaper could not run while a tab was opening, so the TTL depends on navigation speed"
    )

    release.set()
    await asyncio.wait_for(task, timeout=1.0)
    assert session.free_slots() == session.max_slots - 1


async def test_teardown_drops_contexts_stops_the_display_and_is_idempotent(manager: SessionManager) -> None:
    """Teardown is what an error handler and a reaper both call, so it must not raise twice."""
    session = await manager.open(now=1000.0)
    await manager.open_tab(session, target_id="a", url="https://a.example")

    await manager.close(session)
    await manager.close(session)
    await manager.close()

    assert manager.current() is None
    assert len(manager.browser.disposed) == 1  # type: ignore[attr-defined]
    assert not manager.vnc.health()  # type: ignore[union-attr]


async def test_a_context_that_will_not_dispose_does_not_block_the_rest_of_teardown(
    manager: SessionManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One stuck context must not strand the display and every other tab.

    The leak from a context that will not dispose is bounded by the browser's lifetime; a
    teardown that gives up halfway leaves an x11vnc running and a session that cannot be
    reopened.
    """
    session = await manager.open(now=1000.0)
    await manager.open_tab(session, target_id="a", url="https://a.example")
    await manager.open_tab(session, target_id="b", url="https://b.example")

    async def _explode(_browser: Any, _context_id: Any) -> None:
        raise RuntimeError("CDP is not answering")

    monkeypatch.setattr(hitl, "_dispose_context", _explode)
    await manager.close(session)

    assert manager.current() is None
    assert not manager.vnc.health(), "a stuck context left the display running"  # type: ignore[union-attr]


async def test_exported_local_storage_is_actually_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three docstrings promised a storage round-trip and only cookies made it back.

    The export captured ``origins`` faithfully and the apply path dropped them on the floor,
    so a solve that depended on a token in ``localStorage`` -- which plenty do -- silently
    restored half of itself and the target was challenged again for no visible reason.
    """
    written: list[tuple[str, str]] = []

    class _Tab:
        url = "https://example.gov/page"

        async def evaluate(self, expression: str, **_kw: Any) -> None:
            written.append(("evaluate", expression))

    state = {
        "cookies": [],
        "origins": [{"origin": "https://example.gov/page", "localStorage": '{"cf_token": "earned", "n": "2"}'}],
    }

    await hitl._apply_origin_storage(_Tab(), state)

    joined = " ".join(e for _, e in written)
    assert "cf_token" in joined
    assert "earned" in joined
    assert len(written) == 2, "only some of the exported keys were restored"


async def test_storage_is_not_written_into_the_wrong_origin() -> None:
    """localStorage belongs to an origin.

    Writing one origin's storage while the tab sits on another either fails or, worse, lands
    somewhere it was never exported from -- putting a human's token on a site that never saw it.
    """
    written: list[str] = []

    class _Tab:
        url = "https://other.example/page"

        async def evaluate(self, expression: str, **_kw: Any) -> None:
            written.append(expression)

    state = {"origins": [{"origin": "https://example.gov/page", "localStorage": '{"k": "v"}'}]}
    await hitl._apply_origin_storage(_Tab(), state)

    assert written == []


async def test_a_value_with_quotes_does_not_break_the_expression() -> None:
    """A stored value is arbitrary text a human's session put there.

    Building script by interpolating it turns a stray quote into a syntax error at best, and
    into evaluated source at worst. Both operands go through json.dumps for that reason.
    """
    written: list[str] = []

    class _Tab:
        url = "https://example.gov/"

        async def evaluate(self, expression: str, **_kw: Any) -> None:
            written.append(expression)

    nasty = '{"k": "va\\"lue\\u0027); alert(1); //"}'
    await hitl._apply_origin_storage(_Tab(), {"origins": [{"origin": "https://example.gov/", "localStorage": nasty}]})

    assert len(written) == 1
    # The payload is inside a JSON string literal rather than loose in the source.
    assert written[0].startswith("window.localStorage.setItem(")
    assert "alert(1)" in written[0]
    assert written[0].count("setItem") == 1, "the value escaped its literal and became more source"


async def test_unparseable_storage_is_skipped_without_losing_the_cookies() -> None:
    """Storage is a bonus; cookies carry the cleared challenge. Losing one must not lose both."""

    class _Tab:
        url = "https://example.gov/"

        async def evaluate(self, expression: str, **_kw: Any) -> None:
            raise AssertionError("should not have been called")

    await hitl._apply_origin_storage(
        _Tab(), {"origins": [{"origin": "https://example.gov/", "localStorage": "{oh no"}]}
    )


async def test_a_page_that_refuses_script_does_not_fail_the_restore() -> None:
    """One unwritable key must not cost the other keys or the fetch they were restored for."""

    class _Tab:
        url = "https://example.gov/"

        async def evaluate(self, expression: str, **_kw: Any) -> None:
            raise RuntimeError("script evaluation is blocked")

    await hitl._apply_origin_storage(
        _Tab(), {"origins": [{"origin": "https://example.gov/", "localStorage": '{"a":"1"}'}]}
    )


async def test_a_wedged_export_does_not_defer_the_reaper(manager: SessionManager, monkeypatch) -> None:
    """The completion path's CDP work must not span the lock either.

    Sibling of the slow-OPEN test above, and the same failure at the other end of a tab's
    life. A browser context that has stopped answering is exactly the state in which the hard
    TTL matters most -- an operator's authenticated session is still live in it -- so holding
    the manager across an untimed export makes the one guarantee that does not depend on
    guessing intent depend on a wedged browser instead.
    """
    session = await manager.open(now=1000.0)
    tab = await manager.open_tab(session, target_id="t", url="https://example.gov/a")

    started = asyncio.Event()
    release = asyncio.Event()

    async def _wedged(_tab: Any) -> dict[str, Any] | None:
        started.set()
        await release.wait()
        return {"cookies": []}

    monkeypatch.setattr(manager, "_export_state", _wedged)
    task = asyncio.create_task(manager.complete_tab(session, tab.tab_id))
    await asyncio.wait_for(started.wait(), timeout=1.0)

    assert await asyncio.wait_for(manager.reap(now=1001.0), timeout=1.0) is False, (
        "the reaper could not run while a tab was exporting, so the TTL depends on the browser answering"
    )
    assert await asyncio.wait_for(manager.reap(now=99_999.0), timeout=1.0) is True, (
        "an expired session could not be reaped while an export hung, which is the leak the TTL exists to stop"
    )

    release.set()
    await asyncio.wait_for(task, timeout=1.0)


async def test_a_tab_whose_export_hangs_still_gives_up_its_slot(manager: SessionManager, monkeypatch) -> None:
    """Popping is the claim, so a browser that stops answering costs the state and not the slot.

    A hang rather than a raise, because a hang is the only failure that can actually get here:
    `_export_state` catches its own errors and returns None by contract. An earlier version of
    this test monkeypatched it to raise, which asserted a premise production cannot produce
    while leaving the reachable path -- and the timeout that bounds it -- untested.

    The alternative to reclaiming the slot is stranding it behind a browser that is already
    broken, and a bounded working set that cannot be reclaimed is just a smaller ceiling on
    the same leak.
    """
    monkeypatch.setattr(hitl, "_COMPLETE_TAB_TIMEOUT_SECONDS", 0.05)
    session = await manager.open(now=1000.0)
    tab = await manager.open_tab(session, target_id="t", url="https://example.gov/a")
    before = session.free_slots()

    async def _hangs(_tab: Any) -> dict[str, Any] | None:
        await asyncio.sleep(3600)
        return {"cookies": []}

    monkeypatch.setattr(manager, "_export_state", _hangs)
    completed = await asyncio.wait_for(manager.complete_tab(session, tab.tab_id), timeout=2.0)

    assert completed.exported_state is None, "a hung export yields no state rather than hanging the caller"
    assert session.free_slots() == before + 1, "and the slot comes back regardless"


async def test_export_state_never_raises_into_a_completion(manager: SessionManager, monkeypatch) -> None:
    """The contract the narrowed catch in complete_tab depends on.

    `complete_tab` catches only TimeoutError. That is correct exactly as long as this holds,
    so it is asserted here rather than assumed -- if `_export_state` ever starts propagating,
    this fails and names the reason instead of a completion blowing up in production.
    """
    session = await manager.open(now=1000.0)
    tab = await manager.open_tab(session, target_id="t", url="https://example.gov/a")

    async def _boom(*_a: Any, **_k: Any) -> dict[str, Any]:
        raise RuntimeError("the browser stopped answering")

    monkeypatch.setattr(hitl, "_export_context_state", _boom)
    state = await manager._export_state(session.tabs[tab.tab_id])

    assert state is None, "a failing export returns None rather than raising into the completion"


class TestCredentialsAreNotRenderable:
    """A session token and a raw cookie jar must not walk out in a repr.

    Chunk 06's own criterion is "state is never logged and never appears in a repr", and until
    now that was enforced only on the MIT side, where `SealedSessionState` redacts the
    ENCRYPTED form and a test pins it. This container holds the plaintext -- `httpOnly` session
    cookies included, unencrypted, because it has no key -- so the same guarantee matters more
    here, not less.

    Latent rather than an active leak: nothing renders these objects today. That is precisely
    why it is worth pinning, because the next `log.debug("...%s", session)` or a structured
    handler that serialises `extra` does it silently and nothing fails.
    """

    def test_a_session_does_not_render_its_token(self) -> None:
        session = HitlSession(session_id="sess-1", token="super-secret-bearer", expires_at=1.0, max_slots=3)

        for rendered in (repr(session), str(session), f"{session}", "%s" % session):
            assert "super-secret-bearer" not in rendered, f"the bearer token was rendered: {rendered}"
            assert "redacted" in rendered

        # The id is deliberately NOT redacted: it is the handle the caller already sent, it is
        # what makes a log line traceable, and this module answers "no such session" and "wrong
        # token" with one exception precisely so the id is not the secret.
        assert "sess-1" in repr(session)

    def test_a_tab_does_not_render_its_exported_cookies(self) -> None:
        tab = HitlTab(
            tab_id="tab-1",
            target_id="t1",
            url="https://example.gov/x",
            tab=None,
            context_id=None,
            opened_at=1.0,
            exported_state={"cookies": [{"name": "session", "value": "live-httponly-value"}]},
        )

        for rendered in (repr(tab), str(tab), f"{tab}"):
            assert "live-httponly-value" not in rendered, f"an exported cookie was rendered: {rendered}"
            assert "redacted" in rendered

        assert "tab-1" in repr(tab)
        assert "https://example.gov/x" in repr(tab)

    def test_a_session_holding_tabs_does_not_leak_through_the_nesting(self) -> None:
        """The container's repr must not reintroduce what the element's repr removes.

        A dataclass repr recurses, so `HitlSession.tabs` would print each `HitlTab` in full if
        the session used the default. Redacting only the leaf leaves the parent as a second
        route to the same plaintext.
        """
        tab = HitlTab(
            tab_id="tab-1",
            target_id="t1",
            url="https://example.gov/x",
            tab=None,
            context_id=None,
            opened_at=1.0,
            exported_state={"cookies": [{"name": "session", "value": "live-httponly-value"}]},
        )
        session = HitlSession(session_id="sess-1", token="super-secret-bearer", expires_at=1.0, max_slots=3)
        session.tabs["tab-1"] = tab

        rendered = repr(session)
        assert "live-httponly-value" not in rendered
        assert "super-secret-bearer" not in rendered
