"""Shared fixtures for core package tests."""

from __future__ import annotations

import pytest

from threetears.core._bridge import drain, shutdown


@pytest.fixture(autouse=True)
def _drain_bridge() -> None:  # type: ignore[misc]
    """Drain and shut down the async bridge after every test.

    fire_and_forget() submits coroutines to a background thread. If a test
    fixture tears down (e.g. SQLiteBackend.reset()) before those coroutines
    finish, the bridge thread accesses closed connections — segfault on Linux.

    drain() alone is insufficient: new coroutines can be submitted between
    drain and fixture teardown. shutdown() stops the loop entirely so no
    further work can execute against torn-down resources.
    """
    yield  # type: ignore[misc]
    drain()
    shutdown()
