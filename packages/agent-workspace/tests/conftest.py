"""shared test setup for the agent-workspace package test suite.

exposes the core coordination fake NATS KV helpers (defined in the core
package's test tree) to workspace tests by adding the core tests root to
``sys.path``. the fake NATS KV is a test-only helper exercising the
semantics of ``nats-py`` :class:`KeyValue` that :class:`KVLease` depends
on; reusing it here rather than duplicating the implementation keeps the
workspace lease-wrapper tests in lockstep with core's lease tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

_CORE_COORDINATION_TESTS = (
    Path(__file__).resolve().parent.parent.parent
    / "core"
    / "tests"
    / "unit"
    / "coordination"
)
if str(_CORE_COORDINATION_TESTS) not in sys.path:
    sys.path.insert(0, str(_CORE_COORDINATION_TESTS))
