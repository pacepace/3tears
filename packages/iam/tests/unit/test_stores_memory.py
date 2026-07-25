"""The in-memory store implementations, and the contract every store must honour."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from threetears.iam.stores import (
    AttemptLimiter,
    SingleUseTicketStore,
    StateStore,
    hash_ticket,
    new_ticket_secret,
)
from threetears.iam.stores.memory import MemoryAttemptLimiter, MemoryStateStore, MemoryTicketStore


class _Clock:
    """A manually advanced monotonic clock, so TTL tests do not sleep."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_the_memory_stores_satisfy_their_protocols() -> None:
    # The reason these are shipped rather than hand-rolled per consumer: a double that
    # drifts from the Protocol is how a store bug ships green.
    assert isinstance(MemoryTicketStore(), SingleUseTicketStore)
    assert isinstance(MemoryStateStore(), StateStore)
    assert isinstance(MemoryAttemptLimiter(), AttemptLimiter)


def test_ticket_secrets_are_unique_and_hashed_consistently() -> None:
    assert len({new_ticket_secret() for _ in range(100)}) == 100
    secret = new_ticket_secret()
    assert hash_ticket(secret) == hash_ticket(secret)
    assert len(hash_ticket(secret)) == 64


async def test_the_stored_form_is_the_hash_not_the_secret() -> None:
    # A store dump must not be a set of usable password-reset links.
    ticket = await MemoryTicketStore().issue({"user_id": "u1"}, ttl=timedelta(minutes=5))
    assert ticket.hashed != ticket.secret
    assert ticket.hashed == hash_ticket(ticket.secret)


async def test_ticket_round_trips_once() -> None:
    store = MemoryTicketStore()
    ticket = await store.issue({"user_id": "u1"}, ttl=timedelta(minutes=5))
    assert ticket.hashed == hash_ticket(ticket.secret)
    assert await store.redeem(ticket.secret) == {"user_id": "u1"}
    # Single use: the second redemption finds nothing.
    assert await store.redeem(ticket.secret) is None


async def test_unknown_ticket_redeems_to_none() -> None:
    assert await MemoryTicketStore().redeem(new_ticket_secret()) is None


async def test_expired_ticket_redeems_to_none() -> None:
    clock = _Clock()
    store = MemoryTicketStore(clock=clock)
    ticket = await store.issue({"user_id": "u1"}, ttl=timedelta(minutes=5))
    clock.advance(timedelta(minutes=5).total_seconds() + 1)
    assert await store.redeem(ticket.secret) is None


async def test_concurrent_redemptions_produce_exactly_one_winner() -> None:
    # The property that makes redemption safe: a check-then-act would let two parties both
    # complete a password reset from one ticket.
    store = MemoryTicketStore()
    ticket = await store.issue({"user_id": "u1"}, ttl=timedelta(minutes=5))
    results = await asyncio.gather(*(store.redeem(ticket.secret) for _ in range(10)))
    assert sum(1 for result in results if result is not None) == 1


async def test_state_store_round_trips_once() -> None:
    store = MemoryStateStore()
    await store.put("state-key", {"redirect": "/home"}, ttl=timedelta(minutes=5))
    assert await store.take("state-key") == {"redirect": "/home"}
    # Removing on read is what stops a captured callback URL from being replayed.
    assert await store.take("state-key") is None


async def test_state_store_expires() -> None:
    clock = _Clock()
    store = MemoryStateStore(clock=clock)
    await store.put("k", {"a": 1}, ttl=timedelta(minutes=5))
    clock.advance(301)
    assert await store.take("k") is None


async def test_limiter_counts_up_to_the_threshold() -> None:
    limiter = MemoryAttemptLimiter(max_attempts=3)
    assert not (await limiter.check("k")).limited
    for expected in (1, 2):
        window = await limiter.record_failure("k")
        assert window.count == expected
        assert not window.limited
    window = await limiter.record_failure("k")
    assert window.count == 3
    assert window.limited
    assert window.retry_after is not None


async def test_limiter_check_does_not_record() -> None:
    limiter = MemoryAttemptLimiter(max_attempts=3)
    await limiter.record_failure("k")
    for _ in range(5):
        assert (await limiter.check("k")).count == 1


async def test_limiter_clears_after_success() -> None:
    limiter = MemoryAttemptLimiter(max_attempts=2)
    await limiter.record_failure("k")
    await limiter.record_failure("k")
    assert (await limiter.check("k")).limited
    await limiter.clear("k")
    assert not (await limiter.check("k")).limited


async def test_limiter_window_rolls_over() -> None:
    clock = _Clock()
    limiter = MemoryAttemptLimiter(max_attempts=2, window=timedelta(minutes=15), clock=clock)
    await limiter.record_failure("k")
    await limiter.record_failure("k")
    assert (await limiter.check("k")).limited
    clock.advance(timedelta(minutes=15).total_seconds() + 1)
    assert not (await limiter.check("k")).limited


async def test_limiter_keys_are_independent() -> None:
    limiter = MemoryAttemptLimiter(max_attempts=1)
    await limiter.record_failure("alice")
    assert (await limiter.check("alice")).limited
    # One user's failures must never lock out another -- the whole point of keying at all.
    assert not (await limiter.check("bob")).limited


@pytest.mark.parametrize("payload", [{}, {"a": 1}, {"nested": {"b": [1, 2]}}])
async def test_ticket_payloads_round_trip_unchanged(payload: dict[str, object]) -> None:
    store = MemoryTicketStore()
    ticket = await store.issue(payload, ttl=timedelta(minutes=1))
    assert await store.redeem(ticket.secret) == payload
