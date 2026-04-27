"""
shared pytest setup for the 3tears-conversations test suite.

exposes the core coordination fake-NATS KV helpers to tests that need
L2 parity with the rest of the 3tears packages. mirrors the hook used
by agent-workspace so the two suites track the same implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

_CORE_COORDINATION_TESTS = Path(__file__).resolve().parent.parent.parent / "core" / "tests" / "unit" / "coordination"
if str(_CORE_COORDINATION_TESTS) not in sys.path:
    sys.path.insert(0, str(_CORE_COORDINATION_TESTS))
