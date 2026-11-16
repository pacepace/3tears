"""``pwa.js`` run in headless Chromium: display mode, Media Session and the Screen Wake Lock.

Each test loads the shipped module into a page (served from memory, nothing on the network) and
drives it there, with the browser's own APIs where headless Chromium has them and small fakes
where it cannot (it is never installed, never locked, and grants no wake lock).

A machine without Chromium skips with the launch error; where ``THREETEARS_REQUIRE_CHROMIUM=1`` (the
testament pool) the launch failing fails the test instead, so a missing browser never reads as a pass.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, Route, sync_playwright

from threetears.pwa import export_static

pytestmark = pytest.mark.browser

_ORIGIN = "https://pwa.test"


def _serve(route: Route, pwa_js: bytes) -> None:
    path = route.request.url.removeprefix(_ORIGIN)
    if path == "/pwa.js":
        route.fulfill(body=pwa_js, content_type="text/javascript")
    else:
        route.fulfill(
            body="<!doctype html><html><head><title>t</title></head><body></body></html>", content_type="text/html"
        )


@pytest.fixture(scope="module")
def page(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Page]:
    exported = tmp_path_factory.mktemp("pwa")
    export_static(exported)
    pwa_js = (exported / "pwa.js").read_bytes()
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(args=["--disable-dev-shm-usage"])
        except PlaywrightError as exc:
            if os.environ.get("THREETEARS_REQUIRE_CHROMIUM") == "1":
                pytest.fail(f"headless Chromium cannot launch, and THREETEARS_REQUIRE_CHROMIUM=1 requires it: {exc}")
            pytest.skip(f"headless Chromium cannot launch: {exc}")
        page = browser.new_page()
        page.route(f"{_ORIGIN}/**", lambda route: _serve(route, pwa_js))
        page.goto(f"{_ORIGIN}/")
        yield page
        browser.close()


def _run(page: Page, body: str) -> Any:
    """``body`` as an async function of the imported module ``m``; its return value."""
    return page.evaluate(f"async () => {{ const m = await import('/pwa.js'); {body} }}")


def test_a_tab_is_browser_mode_and_an_installed_window_is_standalone(page: Page) -> None:
    assert _run(page, "return [m.displayMode(window), m.isInstalled(window)];") == ["browser", False]
    fake = """
      const win = { matchMedia: (q) => ({ matches: q === '(display-mode: standalone)',
                                          addEventListener() {}, removeEventListener() {} }) };
      return [m.displayMode(win), m.isInstalled(win), m.displayMode({})];
    """
    assert _run(page, fake) == ["standalone", True, "browser"]


def test_mark_display_mode_follows_a_change(page: Page) -> None:
    body = """
      let mode = 'browser';
      const lists = [];
      const win = { matchMedia: (q) => {
        const l = { q, get matches() { return q === `(display-mode: ${mode})`; }, fns: [],
                    addEventListener(_, f) { this.fns.push(f); }, removeEventListener(_, f) { this.fns = this.fns.filter((x) => x !== f); } };
        lists.push(l); return l; } };
      const stop = m.markDisplayMode(document, win);
      const before = document.documentElement.dataset.displayMode;
      mode = 'standalone';
      for (const l of lists) for (const f of l.fns) f();
      const after = document.documentElement.dataset.displayMode;
      stop();
      mode = 'fullscreen';
      for (const l of lists) for (const f of l.fns) f();
      return [before, after, document.documentElement.dataset.displayMode];
    """
    assert _run(page, body) == ["browser", "standalone", "standalone"]


def test_media_session_sets_metadata_and_state_on_the_real_api(page: Page) -> None:
    body = """
      const ms = new m.MediaSessionControl(navigator, { onPlay() {}, onPause() {} });
      ms.setNowPlaying({ title: 'Song', artist: 'Artist', album: 'Kairo',
                         artwork: [{ src: '/art.png', sizes: '512x512', type: 'image/png' }] });
      const md = navigator.mediaSession.metadata;
      ms.setPlaying(true);
      const playing = navigator.mediaSession.playbackState;
      ms.clear();
      return [ms.supported, md.title, md.artist, md.album, md.artwork[0].src, playing,
              navigator.mediaSession.metadata, navigator.mediaSession.playbackState];
    """
    assert _run(page, body) == [True, "Song", "Artist", "Kairo", f"{_ORIGIN}/art.png", "playing", None, "none"]


def test_media_session_routes_actions_and_clears_seeking_on_a_live_stream(page: Page) -> None:
    body = """
      const handlers = {};
      const calls = [];
      const nav = { mediaSession: { setActionHandler(a, f) {
        if (a === 'seekto') throw new TypeError('not supported here');
        handlers[a] = f; } } };
      const ms = new m.MediaSessionControl(nav, {
        onPlay: () => calls.push('play'), onPause: () => calls.push('pause'),
        onStop: () => { throw new Error('boom'); } });
      handlers.play(); handlers.pause(); handlers.stop();
      return [calls, 'seekforward' in handlers && handlers.seekforward === null,
              'nexttrack' in handlers && handlers.nexttrack === null, 'seekto' in handlers];
    """
    # a throwing handler is contained (logged), and an action the platform lacks is skipped
    assert _run(page, body) == [["play", "pause"], True, True, False]


def test_media_session_does_not_reset_unchanged_metadata(page: Page) -> None:
    body = """
      let sets = 0;
      const session = { setActionHandler() {}, set metadata(v) { sets += 1; } };
      class Meta { constructor(o) { Object.assign(this, o); } }
      const ms = new m.MediaSessionControl({ mediaSession: session }, { MediaMetadata: Meta });
      ms.setNowPlaying({ title: 'A' }); ms.setNowPlaying({ title: 'A' }); ms.setNowPlaying({ title: 'B' });
      const none = new m.MediaSessionControl({});
      none.setNowPlaying({ title: 'A' }); none.setPlaying(true); none.clear();
      return [sets, none.supported];
    """
    assert _run(page, body) == [2, False]


def test_wake_lock_is_held_while_wanted_and_visible(page: Page) -> None:
    body = """
      const doc = { visibilityState: 'visible' };
      const log = [];
      const nav = { wakeLock: { request: async (type) => {
        log.push('request ' + type);
        return { released: false, async release() { this.released = true; log.push('release'); } }; } } };
      const lock = new m.ScreenWakeLock(nav, doc);
      const held = await lock.set(true);
      const again = await lock.sync();
      doc.visibilityState = 'hidden';
      const hidden = await lock.sync();
      doc.visibilityState = 'visible';
      const back = await lock.sync();
      const off = await lock.set(false);
      const denied = await new m.ScreenWakeLock({ wakeLock: { request: async () => { throw new Error('no'); } } }, doc).set(true);
      const absent = await new m.ScreenWakeLock({}, doc).set(true);
      return [held, again, hidden, back, off, denied, absent, log];
    """
    assert _run(page, body) == [
        True,
        True,
        False,
        True,
        False,
        False,
        False,
        ["request screen", "release", "request screen", "release"],
    ]
