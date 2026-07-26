"""The JetStream KV stores, against the shared in-memory KV double.

These back credential lockout, API-key throttling, password-reset tickets and every OAuth
state in the platform, so the properties asserted here are the ones a security reviewer
would ask about: does a lockout actually last, does a redemption happen exactly once, and
what happens when the broker is unreachable.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from threetears.core.testing.kv import FakeNatsClient
from threetears.iam.stores import AttemptLimiter, SingleUseTicketStore, StateStore, hash_ticket
from threetears.iam.stores.nats_kv import NatsKvAttemptLimiter, NatsKvStateStore, NatsKvTicketStore

_WINDOW = timedelta(minutes=15)


@pytest.fixture
def nats() -> FakeNatsClient:
    return FakeNatsClient()


def _limiter(nats: FakeNatsClient, **overrides: object) -> NatsKvAttemptLimiter:
    kwargs: dict[str, object] = {"bucket_name": "lockout", "max_attempts": 3, "window": _WINDOW}
    kwargs.update(overrides)
    return NatsKvAttemptLimiter(nats, **kwargs)  # type: ignore[arg-type]


async def _bucket(nats: FakeNatsClient, name: str = "state") -> object:
    return await nats.kv_bucket(name=name)


def test_the_stores_satisfy_their_protocols(nats: FakeNatsClient) -> None:
    assert isinstance(_limiter(nats), AttemptLimiter)


async def test_state_store_satisfies_its_protocol(nats: FakeNatsClient) -> None:
    store = NatsKvStateStore(await _bucket(nats))  # type: ignore[arg-type]
    assert isinstance(store, StateStore)


async def test_ticket_store_satisfies_its_protocol(nats: FakeNatsClient) -> None:
    store = NatsKvTicketStore(await _bucket(nats, "tickets"))  # type: ignore[arg-type]
    assert isinstance(store, SingleUseTicketStore)


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


async def test_the_raw_key_never_reaches_the_bucket(nats: FakeNatsClient) -> None:
    """Keys are far likelier than values to end up in an operator's terminal."""
    limiter = _limiter(nats)
    await limiter.record_failure("user@example.com")
    bucket = await nats.kv_bucket(name="lockout")
    assert await bucket.get(key="user@example.com") is None


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


# --- window semantics, driven through the clock seam ------------------------------------


class _Clock:
    """A movable time source, so the window boundary is a thing a test can stand on."""

    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _clocked_limiter(nats: FakeNatsClient, clock: _Clock, **overrides: object) -> NatsKvAttemptLimiter:
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
    clock = _Clock(now=899.0)  # 1s before a 900s epoch boundary
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
