"""unit tests for :func:`threetears.epoch.tick.catchup_tick`.

Nothing sleeps. The pass is one call by construction -- cadence belongs to the
consumer's loop -- so a test drives it directly rather than waiting for a timer.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from threetears.core.testing.kv import FakeNatsClient
from threetears.epoch.client import EpochClient
from threetears.epoch.listener import EpochListener
from threetears.epoch.tick import catchup_tick
from threetears.nats.subjects import Subject


def _subject(path: str) -> Subject:
    return Subject(path=path, kind="point")


def _nats() -> Any:
    nats = MagicMock()
    nats.publish = AsyncMock()
    nats.subscribe_typed = AsyncMock()
    nats.kv_bucket = FakeNatsClient().kv_bucket
    return nats


class _ScriptedListener(EpochListener):
    """A real listener whose ``catch_up`` answers a script.

    Subclassed rather than faked. A stand-in would have to mirror every public
    method of ``EpochListener`` to satisfy fake-parity enforcement, and would
    silently rot as that class grows -- which is exactly what the gate exists
    to stop. Subclassing gives parity by construction and still lets the two
    members this function actually depends on be driven directly.
    """

    def __init__(self, nats: Any, results: dict[str, Any]) -> None:
        super().__init__(nats, EpochClient(MagicMock(), nats))
        self._results = results
        self.polled: list[str] = []

    async def catch_up(self, subject: Subject, on_bump: Any) -> int:
        """answer the script for ``subject``, recording that it was polled."""
        self.polled.append(subject.path)
        outcome = self._results[subject.path]
        if isinstance(outcome, Exception):
            raise outcome
        self._last_seen[subject.path] = outcome
        if outcome:
            await on_bump(outcome, None)
        return outcome


class TestOnePassOverEverySubject:
    @pytest.mark.asyncio
    async def test_every_subject_is_polled(self) -> None:
        listener = _ScriptedListener(_nats(), {"app.a.epoch": 0, "app.b.epoch": 0})

        await catchup_tick(
            listener,
            [(_subject("app.a.epoch"), AsyncMock()), (_subject("app.b.epoch"), AsyncMock())],
        )

        assert listener.polled == ["app.a.epoch", "app.b.epoch"]

    @pytest.mark.asyncio
    async def test_it_reports_how_many_advanced(self) -> None:
        listener = _ScriptedListener(_nats(), {"app.a.epoch": 7, "app.b.epoch": 0})

        advanced = await catchup_tick(
            listener,
            [(_subject("app.a.epoch"), AsyncMock()), (_subject("app.b.epoch"), AsyncMock())],
        )

        assert advanced == 1

    @pytest.mark.asyncio
    async def test_an_empty_pass_is_not_an_error(self) -> None:
        assert await catchup_tick(_ScriptedListener(_nats(), {}), []) == 0


class TestOneFailingSubjectDoesNotAbandonTheRest:
    """The failure mode this function exists to remove, one level up.

    A consumer polling its subjects inline abandons every subject after the
    first that raises -- so one broken domain silently stops catch-up for the
    others sharing the pass, and nothing says so.
    """

    @pytest.mark.asyncio
    async def test_subjects_after_a_failure_are_still_polled(self) -> None:
        listener = _ScriptedListener(_nats(), {"app.a.epoch": RuntimeError("broker"), "app.b.epoch": 3})

        with pytest.raises(RuntimeError, match="broker"):
            await catchup_tick(
                listener,
                [(_subject("app.a.epoch"), AsyncMock()), (_subject("app.b.epoch"), AsyncMock())],
            )

        assert "app.b.epoch" in listener.polled

    @pytest.mark.asyncio
    async def test_a_later_subject_still_fires_its_callback(self) -> None:
        listener = _ScriptedListener(_nats(), {"app.a.epoch": RuntimeError("broker"), "app.b.epoch": 3})
        on_bump_b = AsyncMock()

        with pytest.raises(RuntimeError):
            await catchup_tick(
                listener,
                [(_subject("app.a.epoch"), AsyncMock()), (_subject("app.b.epoch"), on_bump_b)],
            )

        on_bump_b.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_the_failure_still_surfaces(self) -> None:
        """Continuing is not swallowing: a consumer bug must still be visible."""
        listener = _ScriptedListener(_nats(), {"app.a.epoch": RuntimeError("broker")})

        with pytest.raises(RuntimeError, match="broker"):
            await catchup_tick(
                listener,
                [(_subject("app.a.epoch"), AsyncMock())],
            )

    @pytest.mark.asyncio
    async def test_the_first_failure_is_the_one_raised(self) -> None:
        listener = _ScriptedListener(
            _nats(), {"app.a.epoch": RuntimeError("first"), "app.b.epoch": RuntimeError("second")}
        )

        with pytest.raises(RuntimeError, match="first"):
            await catchup_tick(
                listener,
                [(_subject("app.a.epoch"), AsyncMock()), (_subject("app.b.epoch"), AsyncMock())],
            )


class TestAgainstARealListener:
    """One pass against the real thing, so the stand-in cannot drift unnoticed."""

    @pytest.mark.asyncio
    async def test_a_bumped_subject_advances_and_fires(self) -> None:
        nats = _nats()
        client = EpochClient(MagicMock(), nats)
        listener = EpochListener(nats, client)
        subject = _subject("app.real.epoch")
        on_bump = AsyncMock()
        await listener.subscribe(subject, on_bump)
        await client.bump(subject)

        advanced = await catchup_tick(listener, [(subject, on_bump)])

        assert advanced == 1
        on_bump.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_an_unchanged_subject_does_not_fire(self) -> None:
        nats = _nats()
        client = EpochClient(MagicMock(), nats)
        listener = EpochListener(nats, client)
        subject = _subject("app.quiet.epoch")
        on_bump = AsyncMock()
        await listener.subscribe(subject, on_bump)

        assert await catchup_tick(listener, [(subject, on_bump)]) == 0
        on_bump.assert_not_awaited()
