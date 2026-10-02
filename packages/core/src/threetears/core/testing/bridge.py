"""test teardown for the sync-to-async bridge's background loop.

The bridge (``threetears.core.fire_and_forget`` and the synchronous collection accessors) runs
coroutines scheduled from code with no running event loop on one background loop, owned by the
process. In production nothing stops that loop: it is a daemon thread that dies with the process.
A test suite is the one owner that must stop it, because a test closes resources -- a SQLite
connection, a pool -- that a still-queued coroutine would otherwise touch after they are gone, and
because a test that replaces the loop's factory needs the next caller to build a fresh one.

That is a test harness concern, so it lives here with the rest of the shared test infrastructure
rather than on the bridge's public surface, where a consumer calling it would be stopping a loop it
does not own.
"""

from __future__ import annotations

from threetears.core._bridge import drain, shutdown

__all__ = ["drain_and_shutdown_bridge"]


def drain_and_shutdown_bridge() -> None:
    """wait for every coroutine queued on the background loop, then stop the loop and its thread.

    Call it before closing any resource a ``fire_and_forget`` coroutine may still be using. A
    later bridge caller starts a new loop on first use, so it is safe between tests. A no-op when
    no loop is running.

    :return: nothing
    :rtype: None
    """
    drain()
    shutdown()
