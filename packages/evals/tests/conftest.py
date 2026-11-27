"""evals' tests run under a raised event-loop stall limit.

The loop-stall guard (``threetears.core.testing.loop_stall``) holds every test to one loop turn of 0.25 s CPU,
because a serving process answers every call on one loop. evals serves nobody: it is an offline batch harness, so a
long turn only slows its own runs. Its runner and report writer do synchronous CPU work on the loop (gzip of case
files in ``WriterCaseFile.write``, report prose, JSON encoding), and on the testament pool 528 turns across its tests
ran past the limit (2026-11-26, 3tears 11d0494e5: median 0.36 s, 95th percentile 1.57 s, worst 5.46 s). The worst
turn depends on the pool machine: ``test_campaign_readings``' one turn in ``analysis.stats._betting_p_staked`` held
5.46 s of loop CPU on one runner and 14.80 s on another (3tears 0429ab38d), the same code.

Pace, 2026-11-26: evals takes a package-wide limit now, and moving its CPU work off the loop is a separate evals pass.
The limit is twice the worst turn measured on any pool machine, so a runaway still fails. Remove this file when that
pass lands.
"""

from __future__ import annotations

from pathlib import Path

import pytest

#: twice the worst turn measured on any pool machine (14.80 s), so only a runaway fails
_EVALS_STALL_LIMIT_SECONDS = 30.0

_WHY = (
    "evals is an offline batch harness that serves nobody; its runner and report writer still do CPU work on the "
    "loop (Pace 2026-11-26: moving it off is a separate evals pass)"
)

_HERE = Path(__file__).parent


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """give every evals test the raised limit, unless it states its own."""
    for item in items:
        if item.path.is_relative_to(_HERE) and item.get_closest_marker("loop_stall_limit") is None:
            item.add_marker(pytest.mark.loop_stall_limit(seconds=_EVALS_STALL_LIMIT_SECONDS, why=_WHY))
