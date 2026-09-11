"""the two startup budgets a process waits on are one wall clock.

A pod waits for its PostgreSQL pool and its NATS connection under separate
budgets that are documented as mirrors of each other. The core config layer
cannot import the NATS constant, so the mirror is a claim; this test, in the
one package that can import both, makes it a fact.
"""

from __future__ import annotations

from datetime import timedelta

from threetears.core.config import DEFAULT_POOL_STARTUP_TIMEOUT_SECONDS
from threetears.nats import DEFAULT_STARTUP_TIMEOUT


def test_the_pool_and_nats_startup_budgets_are_equal() -> None:
    assert timedelta(seconds=DEFAULT_POOL_STARTUP_TIMEOUT_SECONDS) == DEFAULT_STARTUP_TIMEOUT
