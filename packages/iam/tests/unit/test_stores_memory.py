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
from threetears.iam.stores.memory import (
    MemoryAttemptLimiter,
    MemoryStateStore,
    MemoryStoreFullError,
    MemoryTicketStore,
)


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


async def test_state_store_get_does_not_consume() -> None:
    """The whole point of `get` over `take`: the value survives the read, because single-use
    is somebody else's job here."""
    store = MemoryStateStore()
    await store.put("state-1", {"nonce": "n"}, ttl=timedelta(minutes=5))
    assert await store.get("state-1") == {"nonce": "n"}
    assert await store.get("state-1") == {"nonce": "n"}
    assert await store.take("state-1") == {"nonce": "n"}
    assert await store.get("state-1") is None


async def test_state_store_get_honours_the_ttl() -> None:
    clock = _Clock()
    store = MemoryStateStore(clock=clock)
    await store.put("state-1", {"nonce": "n"}, ttl=timedelta(minutes=5))
    clock.advance(301)
    assert await store.get("state-1") is None


# --- bounded memory: expired entries are swept on write, and max_entries caps the live ones ---


async def test_the_state_store_cap_refuses_a_new_key_past_it() -> None:
    store = MemoryStateStore(max_entries=2)
    await store.put("a", {"n": 1}, ttl=timedelta(minutes=5))
    await store.put("b", {"n": 2}, ttl=timedelta(minutes=5))

    with pytest.raises(MemoryStoreFullError) as refused:
        await store.put("c", {"n": 3}, ttl=timedelta(minutes=5))

    assert refused.value.max_entries == 2
    # refused, not evicted: both in-flight entries survive
    assert await store.get("a") == {"n": 1}
    assert await store.get("b") == {"n": 2}
    assert await store.get("c") is None


async def test_overwriting_a_live_key_at_the_cap_is_not_refused() -> None:
    store = MemoryStateStore(max_entries=1)
    await store.put("a", {"n": 1}, ttl=timedelta(minutes=5))
    await store.put("a", {"n": 2}, ttl=timedelta(minutes=5))
    assert await store.take("a") == {"n": 2}


async def test_an_abandoned_state_entry_is_swept_by_a_later_write() -> None:
    """Issue, abandon, expire, write again: the abandoned entry no longer holds a slot.

    Nothing ever reads the abandoned key, so only a sweep on write can free it -- the cap of
    one is what makes the sweep visible from outside.
    """
    clock = _Clock()
    store = MemoryStateStore(clock=clock, max_entries=1)
    await store.put("abandoned", {"n": 1}, ttl=timedelta(seconds=10))

    clock.advance(11)
    await store.put("next", {"n": 2}, ttl=timedelta(seconds=10))

    assert await store.get("next") == {"n": 2}
    assert await store.get("abandoned") is None


async def test_a_ticket_cap_frees_as_tickets_expire_and_holds_while_they_are_live() -> None:
    clock = _Clock()
    store = MemoryTicketStore(clock=clock, max_entries=2)
    first = await store.issue({"n": 1}, ttl=timedelta(seconds=10))
    clock.advance(5)
    second = await store.issue({"n": 2}, ttl=timedelta(seconds=10))
    with pytest.raises(MemoryStoreFullError):
        await store.issue({"n": 3}, ttl=timedelta(seconds=10))

    clock.advance(6)  # the first has expired, the second has not
    third = await store.issue({"n": 3}, ttl=timedelta(seconds=10))
    with pytest.raises(MemoryStoreFullError):
        await store.issue({"n": 4}, ttl=timedelta(seconds=10))

    assert await store.redeem(first.secret) is None
    assert await store.redeem(second.secret) == {"n": 2}
    assert await store.redeem(third.secret) == {"n": 3}
    # redeeming freed both slots
    await store.issue({"n": 5}, ttl=timedelta(seconds=10))
    await store.issue({"n": 6}, ttl=timedelta(seconds=10))


async def test_a_rewritten_key_is_not_swept_at_its_first_expiry() -> None:
    """The sweep must not drop a key whose first write expired but whose rewrite is still live."""
    clock = _Clock()
    store = MemoryStateStore(clock=clock)
    await store.put("k", {"v": 1}, ttl=timedelta(seconds=10))
    clock.advance(5)
    await store.put("k", {"v": 2}, ttl=timedelta(seconds=60))
    clock.advance(10)  # past the first write's expiry
    await store.put("other", {"v": 3}, ttl=timedelta(seconds=60))  # a write, so a sweep
    assert await store.get("k") == {"v": 2}


async def test_many_short_lived_entries_cycle_through_a_small_cap() -> None:
    """Far more writes than the cap, each expiring before the next: never refused."""
    clock = _Clock()
    store = MemoryStateStore(clock=clock, max_entries=3)
    for n in range(500):
        await store.put(f"k{n}", {"n": n}, ttl=timedelta(seconds=1))
        clock.advance(1)
    assert await store.get("k499") is None
    await store.put("last", {"n": -1}, ttl=timedelta(seconds=1))
    assert await store.get("last") == {"n": -1}


def test_a_cap_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="max_entries"):
        MemoryStateStore(max_entries=0)


async def test_redemption_and_the_injected_clock_are_unchanged_by_the_sweep() -> None:
    clock = _Clock()
    store = MemoryTicketStore(clock=clock)
    ticket = await store.issue({"user_id": "u1"}, ttl=timedelta(seconds=10))
    clock.advance(9)
    await store.issue({"user_id": "u2"}, ttl=timedelta(seconds=10))  # a write, so a sweep
    assert await store.redeem(ticket.secret) == {"user_id": "u1"}
    assert await store.redeem(ticket.secret) is None


async def test_the_attempt_limiter_resets_a_lapsed_window_after_other_keys_write() -> None:
    clock = _Clock()
    limiter = MemoryAttemptLimiter(max_attempts=2, window=timedelta(seconds=10), clock=clock)
    await limiter.record_failure("victim")
    await limiter.record_failure("victim")
    assert (await limiter.check("victim")).limited

    clock.advance(11)
    await limiter.record_failure("someone-else")  # sweeps the lapsed window
    assert (await limiter.check("victim")).count == 0
    window = await limiter.record_failure("victim")
    assert window.count == 1
    assert not window.limited
