"""evals' tests run under a raised event-loop stall limit.

The loop-stall guard (``threetears.core.testing.loop_stall``) holds every test to one loop turn of 0.25 s CPU,
because a serving process answers every call on one loop. evals serves nobody: it is an offline batch harness, so a
long turn only slows its own runs. Its runner and report writer do synchronous CPU work on the loop (gzip of case
files in ``WriterCaseFile.write``, report prose, JSON encoding), and on the testament pool 528 turns across its tests
ran past the limit (2026-11-26, 3tears 11d0494e5: median 0.36 s, 95th percentile 1.57 s, worst 5.46 s).

Pace, 2026-11-26: evals takes a package-wide limit now, and moving its CPU work off the loop is a separate evals pass.
The limit sits above the worst measured turn with room for slower pool machines, so a runaway still fails. Remove
this file when that pass lands.
"""

from __future__ import annotations

from pathlib import Path

import pytest

#: above the worst turn measured on the pool (5.46 s), so only a runaway fails
_EVALS_STALL_LIMIT_SECONDS = 10.0

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
