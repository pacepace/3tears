"""Shared fixtures for core package tests."""

from __future__ import annotations

from threetears.core._bridge import drain, shutdown


def drain_and_shutdown_bridge() -> None:
    """Drain pending async bridge tasks then stop the loop.

    Must be called before closing any resources (SQLite connections, etc.)
    that fire_and_forget coroutines may still be using.
    """
    drain()
    shutdown()
