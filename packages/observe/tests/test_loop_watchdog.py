"""the loop watchdog names a stall of a tool pod's event loop, with the code that held it, and is silent otherwise.

the watchdog runs on a clock the test moves (:mod:`.loop_watchdog_test_support`), so each stall is exactly as
long as the test holds the loop for, and a loop that keeps running is never mistaken for a held one by a slow
machine.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.observe import loop_watchdog as module

from packages.observe.tests.loop_watchdog_test_support import Log, Watched

#: pytest-timeout's default method is SIGALRM, the signal the held-loop dump owns: a test holding it would leave the
#: watchdog no dump to take (it leaves an alarm someone else handles alone), so these tests time out on a thread
pytestmark = pytest.mark.timeout(600, method="thread")


def _deliberately_held_loop(watched: Watched, *steps: float) -> None:
    """synchronous work on the loop: what freezes every call the pod would answer."""
    watched.hold(*steps)


async def test_a_held_loop_is_logged_with_the_code_that_held_it_and_its_end() -> None:
    logged = Log()
    watched = Watched(stall_seconds=0.1, beat_seconds=0.02, repeat_seconds=60.0, logger=logged)
    watched.watchdog.start()
    try:
        await watched.ran_again()
        _deliberately_held_loop(watched, 0.4)
        await watched.ran_again()
    finally:
        watched.watchdog.stop()
    stalls, ends = logged.stalls(), logged.ends()
    assert len(stalls) == 1, logged.warnings
    assert "_deliberately_held_loop" in stalls[0]["loop_stack"]
    assert stalls[0]["stalled_seconds"] >= 0.1, "logged once silent past the bound; the figure is rounded to the ms"
    assert len(ends) == 1 and ends[0]["stalled_seconds"] >= 0.3


async def test_a_loop_that_keeps_running_is_never_logged() -> None:
    logged = Log()
    watched = Watched(stall_seconds=0.1, beat_seconds=0.02, logger=logged)
    watched.watchdog.start()
    try:
        # 0.4 s in all, four times the bound: only the beats between the steps keep it from being a stall
        await watched.runs(20, 0.02)
    finally:
        watched.watchdog.stop()
    assert logged.warnings == []


async def test_a_long_stall_is_logged_again_while_it_lasts() -> None:
    logged = Log()
    watched = Watched(stall_seconds=0.05, beat_seconds=0.01, repeat_seconds=0.2, logger=logged)
    watched.watchdog.start()
    try:
        await watched.ran_again()
        # silent past the bound at 0.1 s, then a repeat at 0.35 s and at 0.6 s; 0.65 s is too soon for another
        _deliberately_held_loop(watched, 0.1, 0.25, 0.25, 0.05)
        await watched.ran_again()
    finally:
        watched.watchdog.stop()
    stalls = logged.stalls()
    assert len(stalls) == 3, logged.warnings
    assert [stall["stalled_seconds"] for stall in stalls] == [0.1, 0.35, 0.6]


def test_the_pods_loop_is_held_to_fifty_ms_and_seen_within_a_beat_of_it() -> None:
    assert module.STALL_SECONDS == 0.05
    assert module.BEAT_SECONDS <= module.STALL_SECONDS / 4, "a silence is seen within a beat of passing the bound"


async def test_a_pod_that_stalls_often_logs_only_its_budget_and_counts_the_rest(monkeypatch: Any) -> None:
    monkeypatch.setattr(module, "LOG_BUDGET", 2)
    logged = Log()
    watched = Watched(stall_seconds=0.05, beat_seconds=0.01, repeat_seconds=60.0, logger=logged)
    watched.watchdog.start()
    try:
        for _ in range(5):
            await watched.ran_again()
            _deliberately_held_loop(watched, 0.12)
        await watched.ran_again()
        health = watched.watchdog.health()
    finally:
        watched.watchdog.stop()
    assert len(logged.stalls()) == 2 and len(logged.ends()) == 2, logged.warnings
    assert health["stalls"] == 5 and health["unlogged"] == 3
    assert health["longest_ms"] >= 110


async def test_the_next_stall_logged_says_how_many_went_unlogged_before_it(monkeypatch: Any) -> None:
    monkeypatch.setattr(module, "LOG_BUDGET", 1)
    monkeypatch.setattr(module, "LOG_WINDOW_SECONDS", 0.5)
    logged = Log()
    watched = Watched(stall_seconds=0.05, beat_seconds=0.01, repeat_seconds=60.0, logger=logged)
    watched.watchdog.start()
    try:
        for _ in range(3):
            await watched.ran_again()
            _deliberately_held_loop(watched, 0.1)
        # the loop runs on past the log window, beating all the while
        await watched.runs(15, 0.04)
        _deliberately_held_loop(watched, 0.1)
        await watched.ran_again()
    finally:
        watched.watchdog.stop()
    stalls = logged.stalls()
    assert len(stalls) == 2, logged.warnings
    assert "unlogged_before" not in stalls[0] and stalls[1]["unlogged_before"] == 2


async def test_stop_ends_the_thread_and_the_heartbeat() -> None:
    watched = Watched(beat_seconds=0.01, logger=Log())
    watched.watchdog.start()
    watched.watchdog.stop()
    assert not watched.watchdog.running and not watched.watchdog.health()["alive"]


async def test_a_stall_the_budget_left_unlogged_is_logged_once_it_lasts_past_the_repeat(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(module, "LOG_BUDGET", 1)
    logged = Log()
    watched = Watched(stall_seconds=0.05, beat_seconds=0.01, repeat_seconds=0.25, logger=logged)
    watched.watchdog.start()
    try:
        await watched.ran_again()
        _deliberately_held_loop(watched, 0.1)
        await watched.ran_again()
        # past the bound at 0.1 s, the budget spent so only counted; logged at 0.4 s, past the repeat
        _deliberately_held_loop(watched, 0.1, 0.3)
        await watched.ran_again()
    finally:
        watched.watchdog.stop()
    stalls = logged.stalls()
    assert len(stalls) == 2, logged.warnings
    assert stalls[1]["stalled_seconds"] >= 0.25 and "_deliberately_held_loop" in stalls[1]["loop_stack"]
