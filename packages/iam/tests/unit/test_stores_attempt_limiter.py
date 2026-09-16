"""The attempt limiter, against the shared in-memory KV double.

This backs credential lockout and API-key throttling, so the properties asserted here are the
ones a security reviewer would ask about: does a lockout actually last a full window, is it
anchored at the first failure rather than a wall-clock boundary, and what happens when storage
is unreachable.

Separate from ``test_stores_nats_kv.py`` because the limiter stopped being a KV store in
0.43.0: its counts live in the coordination tables, so what a test wires is a registry.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import timedelta
from typing import Any

import pytest

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.testing.kv import FakeNatsClient
from threetears.iam.stores import AttemptLimiter
from threetears.iam.stores.attempt_limiter import CollectionAttemptLimiter

_WINDOW = timedelta(minutes=15)


@pytest.fixture
def nats() -> FakeNatsClient:
    return FakeNatsClient()


def _registry(nats: FakeNatsClient) -> CollectionRegistry:
    """a registry over the shared KV double: L1 plus L2, the shape an edge process has.

    The limiter's counts live in the coordination tables now, not in a bucket of its own, so
    the wiring a test supplies is a registry. No L3 here: these assertions are about the window
    and the verdict, which L2 alone answers.
    """
    l1 = SQLiteBackend(db_name=f"iam_limiter_{uuid.uuid4().hex[:8]}")
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l2_client=nats, kv_key_scope="iam-principal")
    return registry


def _limiter(nats: FakeNatsClient, **overrides: Any) -> CollectionAttemptLimiter:
    kwargs: dict[str, Any] = {"purpose": "lockout", "max_attempts": 3, "window": _WINDOW}
    kwargs.update(overrides)
    return CollectionAttemptLimiter(_registry(nats), **kwargs)


def test_the_limiter_satisfies_its_protocol(nats: FakeNatsClient) -> None:
    assert isinstance(_limiter(nats), AttemptLimiter)


# --- attempt limiter -------------------------------------------------------------------


async def test_limiter_counts_up_to_the_threshold(nats: FakeNatsClient) -> None:
    limiter = _limiter(nats)
    assert (await limiter.check("someone")).limited is False
    for expected in (1, 2):
        window = await limiter.record_failure("someone")
        assert (window.count, window.limited) == (expected, False)
    window = await limiter.record_failure("someone")
    assert (window.count, window.limited) == (3, True)


async def test_limiter_check_does_not_record(nats: FakeNatsClient) -> None:
    limiter = _limiter(nats)
    await limiter.record_failure("someone")
    for _ in range(5):
        assert (await limiter.check("someone")).count == 1


async def test_clear_resets_the_counter(nats: FakeNatsClient) -> None:
    limiter = _limiter(nats)
    for _ in range(3):
        await limiter.record_failure("someone")
    assert (await limiter.check("someone")).limited is True
    await limiter.clear("someone")
    assert (await limiter.check("someone")) == (await limiter.check("never-seen"))


async def test_keys_are_independent(nats: FakeNatsClient) -> None:
    limiter = _limiter(nats)
    for _ in range(3):
        await limiter.record_failure("victim")
    assert (await limiter.check("victim")).limited is True
    assert (await limiter.check("bystander")).limited is False


async def test_keys_are_case_sensitive(nats: FakeNatsClient) -> None:
    """The caller decides what a key means. Case-folding here would silently merge two
    distinct credentials -- and a base64 challenge or a mixed-case token is not the same
    value as its lowercase spelling."""
    limiter = _limiter(nats)
    for _ in range(3):
        await limiter.record_failure("Someone")
    assert (await limiter.check("Someone")).limited is True
    assert (await limiter.check("someone")).limited is False


async def test_the_raw_key_never_reaches_storage(nats: FakeNatsClient) -> None:
    """Keys are far likelier than values to end up in an operator's terminal."""
    limiter = _limiter(nats)
    await limiter.record_failure("user@example.com")
    bucket = await nats.kv_bucket(name="collections")
    assert await bucket.get(key="user@example.com") is None
    assert all("user@example.com" not in key for key in bucket.keys())


async def test_lockout_lasts_the_whole_window_from_the_first_failure(nats: FakeNatsClient) -> None:
    """The property an epoch-aligned window cannot provide.

    With a window keyed by ``floor(now / window)``, every key's window rolls at the same
    wall-clock instant, so a burst straddling a boundary gets ``2 x max_attempts`` back to
    back and a lockout can expire milliseconds after it began. Anchoring at the first
    failure means the retry_after reported is real.
    """
    limiter = _limiter(nats)
    for _ in range(3):
        await limiter.record_failure("someone")
    window = await limiter.check("someone")
    assert window.limited is True
    assert window.retry_after is not None
    # Nearly the full window is still to run -- not a stub value, and not near zero.
    assert timedelta(minutes=14) < window.retry_after <= _WINDOW


async def test_not_limited_reports_no_retry_after(nats: FakeNatsClient) -> None:
    limiter = _limiter(nats)
    await limiter.record_failure("someone")
    assert (await limiter.check("someone")).retry_after is None


async def test_concurrent_failures_all_count(nats: FakeNatsClient) -> None:
    """The CAS loop exists so a burst against one credential cannot lose increments -- which
    is precisely the burst a credential-stuffing run produces."""
    limiter = _limiter(nats, max_attempts=50)
    await asyncio.gather(*(limiter.record_failure("someone") for _ in range(20)))
    assert (await limiter.check("someone")).count == 20


async def test_defaults_to_fail_closed(nats: FakeNatsClient) -> None:
    """A limiter with nothing authoritative behind it must not silently admit on a KV
    outage. The default posture is what a caller gets when nobody thought about it."""
    limiter = _limiter(nats)
    assert limiter._counter.fail_open is False  # noqa: SLF001


async def test_fail_open_is_available_for_a_layered_throttle(nats: FakeNatsClient) -> None:
    assert _limiter(nats, fail_open=True)._counter.fail_open is True  # noqa: SLF001


# --- window semantics, driven through the clock seam ------------------------------------


class _Clock:
    """A movable time source, so the window boundary is a thing a test can stand on.

    Anchored near the wall clock by default: a counter row carries the expiry its window
    implies, and the tiers below read that against the wall clock, so a clock parked in 1970
    would make every row read as long expired -- a property of the tiers, not of the window.
    """

    def __init__(self, now: float | None = None) -> None:
        self.now = time.time() if now is None else now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _clocked_limiter(nats: FakeNatsClient, clock: _Clock, **overrides: object) -> CollectionAttemptLimiter:
    limiter = _limiter(nats, **overrides)
    limiter._counter._clock = clock  # noqa: SLF001 - the seam exists for exactly this
    return limiter


async def test_the_window_is_anchored_at_the_first_failure_not_the_wall_clock(
    nats: FakeNatsClient,
) -> None:
    """The property an epoch-aligned window cannot provide, asserted where it differs.

    With a window keyed by ``floor(now / window)``, the counter resets at absolute instants
    regardless of when the failures happened -- so failing 3 times just before a boundary and
    3 times just after yields six attempts with no lockout, and every key in the system rolls
    at the same moment. Anchored at the first failure, the second burst is still inside the
    first window and trips.

    This test FAILS against an epoch-aligned implementation, which is the whole point: the
    previous version of it passed against both.
    """
    boundary = (int(time.time()) // 900 + 1) * 900  # the next 900s epoch boundary, near now
    clock = _Clock(now=boundary - 1.0)  # 1s before it
    limiter = _clocked_limiter(nats, clock)
    for _ in range(3):
        await limiter.record_failure("someone")
    assert (await limiter.check("someone")).limited is True

    clock.advance(2.0)  # across the boundary an epoch-aligned window would reset on
    assert (await limiter.check("someone")).limited is True


async def test_the_window_does_reset_once_it_has_genuinely_elapsed(nats: FakeNatsClient) -> None:
    """The other half: anchored does not mean permanent."""
    clock = _Clock()
    limiter = _clocked_limiter(nats, clock)
    for _ in range(3):
        await limiter.record_failure("someone")
    assert (await limiter.check("someone")).limited is True

    clock.advance(_WINDOW.total_seconds() + 1)
    assert (await limiter.check("someone")).limited is False


async def test_retry_after_shrinks_as_the_window_elapses(nats: FakeNatsClient) -> None:
    """A stub that returned the whole window would satisfy "not near zero" but not this."""
    clock = _Clock()
    limiter = _clocked_limiter(nats, clock)
    for _ in range(3):
        await limiter.record_failure("someone")
    first = (await limiter.check("someone")).retry_after
    clock.advance(300)
    second = (await limiter.check("someone")).retry_after

    assert first is not None and second is not None
    assert second < first
    assert abs((first - second).total_seconds() - 300) < 1


# -- per-entry TTL -------------------------------------------------------------------------
#
# The bucket TTL is one number for every entry in it. The Protocol promises a per-call
# `ttl` -- "how long the ticket stays redeemable" -- and until this landed, that argument was
# recorded into the payload and read by nothing, so every entry stayed redeemable for the
# whole bucket lifetime. The in-memory double honoured it faithfully, so the double enforced
# an expiry production did not: precisely the drift a double is supposed to make impossible.
