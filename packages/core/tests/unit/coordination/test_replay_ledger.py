"""``ReplayLedger``: the guard and its test double are each a ledger, so a seam typed by it takes either."""

from __future__ import annotations

from datetime import timedelta

from threetears.core.coordination import ReplayGuard, ReplayLedger
from threetears.core.testing.replay_guard import FakeReplayGuard


# held by mypy (typecheck.sh checks tests): both are a ReplayLedger, signatures and all
def _guard(guard: ReplayGuard) -> ReplayLedger:
    return guard


def _fake(fake: FakeReplayGuard) -> ReplayLedger:
    return fake


def test_the_double_is_a_ledger() -> None:
    fake = FakeReplayGuard()
    assert isinstance(fake, ReplayLedger)
    ledger: ReplayLedger = fake
    ledger.require_covers(timedelta(seconds=0))


def test_the_guard_class_has_every_ledger_member() -> None:
    members = ("bucket_name", "verifier_future_tolerance", "require_covers", "refusing_until", "bind", "record_unique")
    assert all(hasattr(ReplayGuard, name) for name in members)
