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

import threetears.scrape.tool as scrape_tool_module
from threetears.scrape.robots import RobotsGate

# This suite's shared test infrastructure is imported by its repo-root name::
#
#     from packages.scrape.tests._driver_log_helpers import driver_warnings
#
# A sibling module rather than `from conftest import ...`: a root-level `conftest.py` exists and
# shadows this one. The directory is not put on `sys.path`, so none of its names can shadow, or
# be shadowed by, another suite's.


async def _offline_robots_fetch(url: str) -> tuple[int, str]:
    raise RuntimeError(f"no network in unit tests (robots fetch of {url})")


class _OfflineDefaultRobotsGate(RobotsGate):
    """The real gate, except that a gate built with no fetcher gets one that fails at once.

    An unreachable ``robots.txt`` is already defined as "the site told us nothing", so a tool
    built this way takes exactly the path it took before the default fetcher existed. A caller
    that passes its own ``fetch`` gets it unchanged.
    """

    def __init__(self, *args: Any, fetch: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, fetch=fetch if fetch is not None else _offline_robots_fetch, **kwargs)


@pytest.fixture(autouse=True)
def _no_live_robots_fetch(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the gate ``ScrapeTool`` builds by default read robots.txt offline.

    Replaces the ``RobotsGate`` name ``threetears.scrape.tool`` builds its default gate from,
    so a tool constructed without a ``robots`` argument -- most of this suite -- gets a gate
    whose fetch fails fast instead of reaching ``https://<origin>/robots.txt``. Everything else
    about that gate is the real one. A gate a test constructs itself is untouched: those pass
    their own ``fetch``, or route the default fetcher through a recording exit.

    Opt out with ``@pytest.mark.real_robots_fetch`` when the REAL builder is the thing under
    test. That escape hatch is not a convenience: patching this suite-wide meant the actual
    default fetcher was never executed by anything, so the branch's one security fix --
    binding the robots read to the configured exit -- had no test that could fail when it
    regressed. A blanket patch that hides the code it is protecting is worse than no patch.
    """
    if request.node.get_closest_marker("real_robots_fetch") is not None:
        return
    monkeypatch.setattr(scrape_tool_module, "RobotsGate", _OfflineDefaultRobotsGate)
