"""The JetStream KV stores, against the shared in-memory KV double.

These back password-reset tickets and every OAuth state in the platform, so the properties
asserted here are the ones a security reviewer would ask about: does a redemption happen
exactly once, does an entry expire at its own ttl rather than its bucket's, and what happens
when the broker is unreachable.

The attempt limiter used to live here too. It stopped being a KV store in 0.43.0 and its
tests moved with it, to ``test_stores_attempt_limiter.py``.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

import pytest

from threetears.core.testing.kv import FakeNatsClient
from threetears.iam.stores import SingleUseTicketStore, StateStore, hash_ticket
from threetears.iam.stores.nats_kv import (
    NatsKvStateStore,
    NatsKvTicketStore,
    state_store,
    ticket_store,
)


@pytest.fixture
def nats() -> FakeNatsClient:
    return FakeNatsClient()


async def _bucket(nats: FakeNatsClient, name: str = "state") -> object:
    return await nats.kv_bucket(name=name)


async def test_state_store_satisfies_its_protocol(nats: FakeNatsClient) -> None:
    store = NatsKvStateStore(await _bucket(nats))  # type: ignore[arg-type]
    assert isinstance(store, StateStore)


async def test_ticket_store_satisfies_its_protocol(nats: FakeNatsClient) -> None:
    store = NatsKvTicketStore(await _bucket(nats, "tickets"))  # type: ignore[arg-type]
    assert isinstance(store, SingleUseTicketStore)


# --- ticket store ----------------------------------------------------------------------


async def test_ticket_round_trips_and_redeems_once(nats: FakeNatsClient) -> None:
    store = NatsKvTicketStore(await _bucket(nats, "tickets"))  # type: ignore[arg-type]
    issued = await store.issue({"principal": "p-1"}, ttl=timedelta(minutes=30))
    assert await store.redeem(issued.secret) == {"principal": "p-1"}
    assert await store.redeem(issued.secret) is None


async def test_ticket_bookkeeping_never_reaches_the_caller(nats: FakeNatsClient) -> None:
    store = NatsKvTicketStore(await _bucket(nats, "tickets"))  # type: ignore[arg-type]
    issued = await store.issue({"principal": "p-1"}, ttl=timedelta(minutes=30))
    redeemed = await store.redeem(issued.secret)
    assert redeemed is not None
    assert "_ttl_seconds" not in redeemed


async def test_only_the_hash_is_stored(nats: FakeNatsClient) -> None:
    store = NatsKvTicketStore(await _bucket(nats, "tickets"))  # type: ignore[arg-type]
    issued = await store.issue({"principal": "p-1"}, ttl=timedelta(minutes=30))
    bucket = await nats.kv_bucket(name="tickets")
    assert issued.hashed == hash_ticket(issued.secret)
    assert await bucket.get(key=issued.secret) is None
    assert await bucket.get(key=issued.hashed) is not None


async def test_concurrent_redemption_produces_exactly_one_winner(nats: FakeNatsClient) -> None:
    """Two parties both setting a password off one reset ticket is the failure this
    revision-guarded claim exists to prevent."""
    store = NatsKvTicketStore(await _bucket(nats, "tickets"))  # type: ignore[arg-type]
    issued = await store.issue({"principal": "p-1"}, ttl=timedelta(minutes=30))
    results = await asyncio.gather(*(store.redeem(issued.secret) for _ in range(8)))
    assert sum(1 for result in results if result is not None) == 1


async def test_an_unknown_secret_redeems_to_nothing(nats: FakeNatsClient) -> None:
    store = NatsKvTicketStore(await _bucket(nats, "tickets"))  # type: ignore[arg-type]
    assert await store.redeem("never-issued") is None


# --- state store -----------------------------------------------------------------------


async def test_state_get_does_not_consume_but_take_does(nats: FakeNatsClient) -> None:
    store = NatsKvStateStore(await _bucket(nats))  # type: ignore[arg-type]
    await store.put("state-1", {"nonce": "n"}, ttl=timedelta(minutes=10))
    assert await store.get("state-1") == {"nonce": "n"}
    assert await store.get("state-1") == {"nonce": "n"}
    assert await store.take("state-1") == {"nonce": "n"}
    assert await store.take("state-1") is None
    assert await store.get("state-1") is None


async def test_state_bookkeeping_never_reaches_the_caller(nats: FakeNatsClient) -> None:
    store = NatsKvStateStore(await _bucket(nats))  # type: ignore[arg-type]
    await store.put("state-1", {"nonce": "n"}, ttl=timedelta(minutes=10))
    for payload in (await store.get("state-1"), await store.take("state-1")):
        assert payload is not None
        assert "_ttl_seconds" not in payload


async def test_concurrent_take_produces_exactly_one_winner(nats: FakeNatsClient) -> None:
    store = NatsKvStateStore(await _bucket(nats))  # type: ignore[arg-type]
    await store.put("state-1", {"nonce": "n"}, ttl=timedelta(minutes=10))
    results = await asyncio.gather(*(store.take("state-1") for _ in range(8)))
    assert sum(1 for result in results if result is not None) == 1


async def test_an_absent_key_reads_as_none(nats: FakeNatsClient) -> None:
    store = NatsKvStateStore(await _bucket(nats))  # type: ignore[arg-type]
    assert await store.get("never-put") is None
    assert await store.take("never-put") is None


class _WallClock:
    """A hand-wound wall clock, in unix seconds."""

    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


async def test_a_ticket_expires_at_its_own_ttl_not_the_buckets(nats: FakeNatsClient) -> None:
    clock = _WallClock()
    store = NatsKvTicketStore(await nats.kv_bucket(name="tickets"), clock=clock)
    issued = await store.issue({"user": "u1"}, ttl=timedelta(minutes=10))
    clock.advance(timedelta(minutes=11).total_seconds())
    assert await store.redeem(issued.secret) is None


async def test_a_ticket_within_its_ttl_still_redeems(nats: FakeNatsClient) -> None:
    clock = _WallClock()
    store = NatsKvTicketStore(await nats.kv_bucket(name="tickets"), clock=clock)
    issued = await store.issue({"user": "u1"}, ttl=timedelta(minutes=10))
    clock.advance(timedelta(minutes=9).total_seconds())
    assert await store.redeem(issued.secret) == {"user": "u1"}


async def test_an_expired_ticket_is_refused_without_being_consumed(nats: FakeNatsClient) -> None:
    # The expiry check runs BEFORE the delete, matching the Postgres store's predicate-inside-
    # the-DELETE ordering. Consuming first would let anyone holding an expired secret destroy
    # the record of it, and would make "expired" indistinguishable from "already redeemed".
    clock = _WallClock()
    bucket = await nats.kv_bucket(name="tickets")
    store = NatsKvTicketStore(bucket, clock=clock)
    issued = await store.issue({"user": "u1"}, ttl=timedelta(minutes=10))
    clock.advance(timedelta(minutes=11).total_seconds())
    assert await store.redeem(issued.secret) is None
    assert await bucket.get(key=issued.hashed) is not None


async def test_two_tickets_in_one_bucket_expire_independently(nats: FakeNatsClient) -> None:
    # THE property a bucket TTL cannot express: one bucket, two lifetimes.
    clock = _WallClock()
    store = NatsKvTicketStore(await nats.kv_bucket(name="tickets"), clock=clock)
    short = await store.issue({"kind": "short"}, ttl=timedelta(minutes=5))
    long_lived = await store.issue({"kind": "long"}, ttl=timedelta(hours=2))
    clock.advance(timedelta(minutes=30).total_seconds())
    assert await store.redeem(short.secret) is None
    assert await store.redeem(long_lived.secret) == {"kind": "long"}


async def test_state_take_honours_the_per_entry_ttl(nats: FakeNatsClient) -> None:
    clock = _WallClock()
    store = NatsKvStateStore(await nats.kv_bucket(name="state"), clock=clock)
    await store.put("s1", {"nonce": "n"}, ttl=timedelta(minutes=2))
    clock.advance(timedelta(minutes=3).total_seconds())
    assert await store.take("s1") is None


async def test_state_get_honours_the_per_entry_ttl(nats: FakeNatsClient) -> None:
    # `get` is the non-consuming read, and an expired value must not leak through it either.
    clock = _WallClock()
    store = NatsKvStateStore(await nats.kv_bucket(name="state"), clock=clock)
    await store.put("s1", {"nonce": "n"}, ttl=timedelta(minutes=2))
    assert await store.get("s1") == {"nonce": "n"}
    clock.advance(timedelta(minutes=3).total_seconds())
    assert await store.get("s1") is None


async def test_the_expiry_stamp_never_reaches_the_caller(nats: FakeNatsClient) -> None:
    clock = _WallClock()
    store = NatsKvStateStore(await nats.kv_bucket(name="state"), clock=clock)
    await store.put("s1", {"nonce": "n"}, ttl=timedelta(minutes=2))
    assert await store.take("s1") == {"nonce": "n"}


async def test_the_kv_store_and_the_memory_double_agree_on_expiry(nats: FakeNatsClient) -> None:
    """The double and production must answer the same question the same way.

    This is the test the whole change exists for: before it, `MemoryTicketStore` expired a
    ticket at its `ttl` and `NatsKvTicketStore` did not, so a service tested against the
    double shipped an expiry it did not have.
    """
    from threetears.iam.stores.memory import MemoryTicketStore

    clock = _WallClock()
    kv = NatsKvTicketStore(await nats.kv_bucket(name="tickets"), clock=clock)
    memory = MemoryTicketStore(clock=clock)

    kv_ticket = await kv.issue({"user": "u1"}, ttl=timedelta(minutes=10))
    memory_ticket = await memory.issue({"user": "u1"}, ttl=timedelta(minutes=10))
    clock.advance(timedelta(minutes=11).total_seconds())

    assert await kv.redeem(kv_ticket.secret) is None
    assert await memory.redeem(memory_ticket.secret) is None


# -- the paths a real broker reaches and a happy-path test does not ------------------------


async def test_a_corrupt_payload_reads_as_absent_not_as_an_error(nats: FakeNatsClient) -> None:
    """Documented behaviour, and security-relevant: a value that will not parse is unusable
    either way, and raising would turn it into a 500 on an authentication path where the
    correct answer is simply "this ticket is not valid"."""
    bucket = await nats.kv_bucket(name="tickets")
    store = NatsKvTicketStore(bucket)
    issued = await store.issue({"user": "u1"}, ttl=timedelta(minutes=10))
    await bucket.put(key=issued.hashed, value=b"\xff\xfe not json at all")
    assert await store.redeem(issued.secret) is None


async def test_a_non_object_payload_reads_as_absent_too(nats: FakeNatsClient) -> None:
    # Valid JSON, wrong shape: a bare list is not a payload mapping.
    bucket = await nats.kv_bucket(name="state")
    store = NatsKvStateStore(bucket)
    await store.put("s1", {"nonce": "n"}, ttl=timedelta(minutes=2))
    await bucket.put(key="s1", value=b'["not", "a", "mapping"]')
    assert await store.get("s1") is None


class _CollidingBucket:
    """A bucket whose ``create`` always reports the key as already present.

    # parity-with: threetears.nats.kv.KvBucketLike

    Delegates everything else, so the store under test is exercised normally: only the
    SET-NX outcome is forced, because a real 256-bit collision cannot be arranged.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def create(self, *, key: str, value: bytes) -> int | None:
        return None


async def test_a_hash_collision_refuses_to_overwrite_a_live_ticket(nats: FakeNatsClient) -> None:
    """`create`, not `put`: at 256 bits a collision is vanishingly unlikely, and the SET-NX
    form turns "impossible" into "detected" for free. Overwriting would silently destroy a
    live ticket -- for a password reset, someone else's."""
    bucket = _CollidingBucket(await nats.kv_bucket(name="tickets"))
    store = NatsKvTicketStore(bucket)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        await store.issue({"user": "u1"}, ttl=timedelta(minutes=10))


async def test_the_factories_open_memory_backed_buckets(nats: FakeNatsClient) -> None:
    """auth-flow state is memory-backed, and this test is the inverse of the one it replaces.

    0.40.0 made these factories request `storage="file"`, on the reasoning that a memory
    bucket "is not reliably readable by a second replica" and that this caused an
    intermittent double login on a two-replica deployment. The test that pinned it asserted
    `{"tickets": "file", "state": "file"}`.

    **That was deployed, the buckets were deleted so they recreated file-backed, and the
    double login continued.** Verified on cobalt-dev 2026-09-15: every auth-flow bucket read
    `file`, and a login still restarted once and then succeeded. A prediction was made and it
    failed, so the claim is withdrawn.

    Inverted rather than deleted. A deleted test leaves nothing to stop the next reader
    reaching the same wrong conclusion from the same symptom, and this one has now been
    reached twice.

    **This withdrawal covers these two factories only.** The login-lockout and API-key
    throttle counters also ask for file storage, for a different and still-standing reason --
    a lockout that resets to zero on every broker restart is not a lockout -- and they reach
    it through `WindowedCounter` from the identity service, not from here. Nothing in this
    change touches them.
    """
    await ticket_store(nats, name="tickets", ttl=timedelta(hours=1))
    await state_store(nats, name="state", ttl=timedelta(hours=1))

    opened = {name: bucket.storage for name, bucket in nats._buckets.items()}  # noqa: SLF001 -- the fake's recorded opens ARE the subject

    assert opened == {"tickets": "memory", "state": "memory"}, (
        f"an auth-flow store asked for file storage again: {opened}. The cross-replica "
        f"argument for it was tested against the double login and did not fix it."
    )


async def test_the_factories_open_a_bucket_and_wrap_it(nats: FakeNatsClient) -> None:
    # The factories resolve the bucket per call rather than holding one, so a broker
    # reconnect does not leave a stale handle behind.
    tickets = await ticket_store(nats, name="tickets", ttl=timedelta(hours=1))
    states = await state_store(nats, name="state", ttl=timedelta(hours=1))
    issued = await tickets.issue({"user": "u1"}, ttl=timedelta(minutes=5))
    assert await tickets.redeem(issued.secret) == {"user": "u1"}
    await states.put("k", {"v": 1}, ttl=timedelta(minutes=5))
    assert await states.take("k") == {"v": 1}
