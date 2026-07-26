"""Keep the unit suite off the network.

``RobotsGate`` gained a real default fetcher so that "both behaviours on by default" is true
of a deployment nobody configured. The side effect is that every ``ScrapeTool`` built without
arguments -- which is most of this suite -- would reach for
``https://<origin>/robots.txt`` on its first fetch.

That is a live outbound request from a unit test: slow on a good day, flaky on a bad one, and
dependent on DNS for a resolution nothing in the test cares about. It also quietly makes the
suite's behaviour depend on what a real site happens to serve.

The default fetcher is replaced with one that fails immediately. That is not a weakening: an
unreachable ``robots.txt`` is already defined as "the site told us nothing", so every test
takes exactly the path it took before the default existed. Tests that are ABOUT robots inject
their own fetcher and are unaffected -- ``RobotsGate`` only builds a default when none is
given.
"""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _no_live_robots_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the default robots fetcher fail fast instead of reaching the network."""

    def _offline(_egress: Any = None) -> Any:
        async def _fetch(url: str) -> tuple[int, str]:
            raise RuntimeError(f"no network in unit tests (robots fetch of {url})")

        return _fetch

    monkeypatch.setattr("threetears.scrape.robots._default_fetch_via", _offline)
