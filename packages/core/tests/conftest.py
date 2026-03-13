"""Shared fixtures for core package tests."""

from __future__ import annotations

import pytest

from threetears.core._bridge import drain


@pytest.fixture(autouse=True)
def _drain_bridge() -> None:  # type: ignore[misc]
    """Drain the async bridge after every test.

    fire_and_forget() submits coroutines to a background thread. If a test
    fixture tears down (e.g. SQLiteBackend.reset()) before those coroutines
    finish, the bridge thread accesses closed connections — segfault on Linux.
    """
    yield  # type: ignore[misc]
    drain()
