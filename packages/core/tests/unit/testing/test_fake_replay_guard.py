"""the shipped guard double can be put inside its post-wipe refusal window.

A consumer that gates a surface on ``ReplayGuard.refusing_until`` -- answering every request in
the window with one retryable reply, before it reads a credential -- needs a double that is in the
window on demand. These pin that knob, and that the double's verdicts stay consistent with it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from threetears.core.testing.replay_guard import FakeReplayGuard


async def test_by_default_the_double_is_not_refusing() -> None:
    assert await FakeReplayGuard().refusing_until() is None


async def test_the_double_reports_the_window_it_was_put_in() -> None:
    until = datetime.now(UTC) + timedelta(seconds=30)
    guard = FakeReplayGuard(refusing_until=until)
    assert await guard.refusing_until() == until


async def test_a_window_that_has_passed_reads_as_not_refusing() -> None:
    # as the real guard: the question is whether it is refusing NOW.
    guard = FakeReplayGuard(refusing_until=datetime.now(UTC) - timedelta(seconds=1))
    assert await guard.refusing_until() is None


async def test_inside_the_window_a_remembering_double_refuses_what_the_real_guard_would() -> None:
    until = datetime.now(UTC) + timedelta(seconds=30)
    guard = FakeReplayGuard(refusing_until=until)
    assert await guard.record_unique("issued-inside", issued_at=until - timedelta(seconds=1)) is False
    assert await guard.record_unique("issued-after", issued_at=until) is True


async def test_a_fixed_verdict_still_wins() -> None:
    until = datetime.now(UTC) + timedelta(seconds=30)
    guard = FakeReplayGuard(fresh=True, refusing_until=until)
    assert await guard.record_unique("n", issued_at=until - timedelta(seconds=1)) is True


async def test_the_question_is_counted_so_a_test_can_assert_the_gate_asked() -> None:
    events: list[str] = []
    guard = FakeReplayGuard(events=events)
    await guard.refusing_until()
    assert guard.refusal_window_checks == 1
    assert events == ["refusing_until"]


def test_a_naive_window_is_refused() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        FakeReplayGuard(refusing_until=datetime.now())  # noqa: DTZ005 - the naive value is the case under test
