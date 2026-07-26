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
def _no_live_robots_fetch(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the default robots fetcher fail fast instead of reaching the network.

    Opt out with ``@pytest.mark.real_robots_fetch`` when the REAL builder is the thing under
    test. That escape hatch is not a convenience: patching this suite-wide meant the actual
    ``_default_fetch_via`` was never executed by anything, so the branch's one security fix --
    binding the robots read to the configured exit -- had no test that could fail when it
    regressed. A blanket patch that hides the code it is protecting is worse than no patch.
    """
    if request.node.get_closest_marker("real_robots_fetch") is not None:
        return

    def _offline(_egress: Any = None) -> Any:
        async def _fetch(url: str) -> tuple[int, str]:
            raise RuntimeError(f"no network in unit tests (robots fetch of {url})")

        return _fetch

    monkeypatch.setattr("threetears.scrape.robots._default_fetch_via", _offline)
