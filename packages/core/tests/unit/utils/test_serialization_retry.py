"""a unit YugabyteDB answers with a serialization conflict is run again (``threetears.core.utils.serialization_retry``)."""

from __future__ import annotations

import asyncpg
import pytest

from threetears.core.utils.serialization_retry import retry_serialization_conflicts


class _Unit:
    """conflicts ``conflicts`` times, then answers ``"done"`` (or raises ``then``)."""

    def __init__(
        self, conflicts: int, then: BaseException | None = None, conflict: type[Exception] = asyncpg.SerializationError
    ) -> None:
        self.conflicts = conflicts
        self.then = then
        self.conflict = conflict
        self.runs = 0

    async def __call__(self) -> str:
        self.runs += 1
        if self.runs <= self.conflicts:
            raise self.conflict("conflict with concurrently committed data")
        if self.then is not None:
            raise self.then
        return "done"


async def test_a_conflicting_unit_is_run_again_and_answers() -> None:
    """Flips if a unit YugabyteDB refused with a conflict is not run again, or its answer is lost."""
    unit = _Unit(conflicts=2)
    assert await retry_serialization_conflicts(unit, attempts=3, first_wait=0) == "done"
    assert unit.runs == 3


async def test_a_deadlocked_unit_is_run_again_and_answers() -> None:
    """Flips if a unit YugabyteDB aborted to break a deadlock cycle (40P01) is not run again: racing inserts of one
    row deadlock there (NamespaceCollection.ensure_namespace)."""
    unit = _Unit(conflicts=2, conflict=asyncpg.DeadlockDetectedError)
    assert await retry_serialization_conflicts(unit, attempts=3, first_wait=0) == "done"
    assert unit.runs == 3


async def test_a_unit_that_always_deadlocks_raises_the_deadlock_after_its_attempts() -> None:
    """Flips if a deadlock that never clears is swallowed, or retried past the attempts."""
    unit = _Unit(conflicts=99, conflict=asyncpg.DeadlockDetectedError)
    with pytest.raises(asyncpg.DeadlockDetectedError):
        await retry_serialization_conflicts(unit, attempts=4, first_wait=0)
    assert unit.runs == 4


async def test_a_unit_that_always_conflicts_raises_after_its_attempts() -> None:
    """Flips if a conflict that never clears is swallowed, or retried past the attempts."""
    unit = _Unit(conflicts=99)
    with pytest.raises(asyncpg.SerializationError):
        await retry_serialization_conflicts(unit, attempts=4, first_wait=0)
    assert unit.runs == 4


async def test_any_other_error_is_raised_at_once() -> None:
    """Flips if an error that is not the conflict (a unique violation) is retried or swallowed."""
    unit = _Unit(conflicts=0, then=asyncpg.UniqueViolationError("duplicate key"))
    with pytest.raises(asyncpg.UniqueViolationError):
        await retry_serialization_conflicts(unit, attempts=4, first_wait=0)
    assert unit.runs == 1


async def test_a_unit_is_tried_at_least_once() -> None:
    """Flips if zero attempts silently runs nothing and answers nothing."""
    with pytest.raises(ValueError, match="at least once"):
        await retry_serialization_conflicts(_Unit(conflicts=0), attempts=0, first_wait=0)
