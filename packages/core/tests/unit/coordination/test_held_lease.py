"""``KVLease.hold``: a lease kept alive in the background that says when it has been lost.

``acquire`` hands back a lease the caller must refresh by hand, so every consumer that holds one
across real work writes the same renewal loop -- scrape's session claim, and in scriob a git-writer
lease and a chat-turn guard, neither of which ever learned it had been lost. These tests pin the one
loop they all move onto, over a real ``KVLease`` and an in-memory bucket, so compare-and-swap,
holder identity and expiry are the genuine article.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

import pytest

from threetears.core.coordination.lease import HeldLease, KVLease, LeaseUnavailable
from threetears.core.testing.kv import FakeKvBucket, FakeNatsClient

_BUCKET = "held"
_TTL = timedelta(seconds=1)
_RENEW = timedelta(seconds=0.05)


def _lease(client: FakeNatsClient, pod: str) -> KVLease:
    return KVLease(nats_client=client, bucket_name=_BUCKET, pod_id=pod)  # type: ignore[arg-type]


async def _bucket(client: FakeNatsClient) -> FakeKvBucket:
    return await client.kv_bucket(name=_BUCKET)


async def _take_over(client: FakeNatsClient, key: str) -> None:
    """Rewrite the entry's holder by compare-and-swap, as a stale reclaim by another pod leaves it."""
    bucket = await _bucket(client)
    entry = await bucket.get_entry(key=key)
    assert entry is not None
    value, revision = entry
    assert await bucket.update(key=key, value=value.replace(b"pod-a", b"pod-b"), revision=revision) is not None


def _make_unreachable(bucket: FakeKvBucket, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Make every read of ``bucket`` raise while ``state["down"]`` is true."""
    state: dict[str, Any] = {"down": False}
    real = bucket.get_entry

    async def get_entry(*, key: str) -> tuple[bytes, int] | None:
        if state["down"]:
            raise ConnectionError("kv is unreachable")
        return await real(key=key)

    monkeypatch.setattr(bucket, "get_entry", get_entry)
    return state


async def test_a_held_lease_is_renewed_past_several_ttls() -> None:
    client = FakeNatsClient()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW)
    try:
        await asyncio.sleep(_TTL.total_seconds() * 2.5)
        assert held.held
        with pytest.raises(LeaseUnavailable):
            await _lease(client, "pod-b").hold("job", ttl=_TTL, renew_every=_RENEW)
    finally:
        await held.release()


async def test_a_takeover_is_noticed_on_the_next_renewal() -> None:
    client = FakeNatsClient()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW)
    try:
        await _take_over(client, "job")
        await asyncio.wait_for(held.until_lost(), timeout=5)
        assert not held.held
        assert held.lost.is_set()
    finally:
        await held.release()


async def test_releasing_a_lost_lease_leaves_the_new_owners_entry() -> None:
    client = FakeNatsClient()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW)
    await _take_over(client, "job")
    await asyncio.wait_for(held.until_lost(), timeout=5)
    await held.release()
    surviving = await (await _bucket(client)).get_entry(key="job")
    assert surviving is not None and b"pod-b" in surviving[0]


async def test_a_brief_transport_failure_is_not_a_lost_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeNatsClient()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW)
    try:
        state = _make_unreachable(await _bucket(client), monkeypatch)
        state["down"] = True
        await asyncio.sleep(_RENEW.total_seconds() * 4)
        state["down"] = False
        assert held.held, "a transient renewal failure was treated as losing the lease"
        await asyncio.sleep(_RENEW.total_seconds() * 3)
        assert held.held, "renewal did not resume once the bucket came back"
    finally:
        await held.release()


async def test_a_failure_lasting_past_the_ttl_gives_the_lease_up(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeNatsClient()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW)
    try:
        _make_unreachable(await _bucket(client), monkeypatch)["down"] = True
        await asyncio.wait_for(held.until_lost(), timeout=5)
        assert not held.held
    finally:
        await held.release()


async def test_release_frees_the_key_and_is_idempotent() -> None:
    client = FakeNatsClient()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW)
    await held.release()
    await held.release()
    assert await (await _bucket(client)).get_entry(key="job") is None
    other = await _lease(client, "pod-b").hold("job", ttl=_TTL, renew_every=_RENEW)
    await other.release()


async def test_release_stops_renewal() -> None:
    client = FakeNatsClient()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW)
    await held.release()
    await asyncio.sleep(_RENEW.total_seconds() * 3)
    assert await (await _bucket(client)).get_entry(key="job") is None, "a renewal after release put the entry back"


async def test_as_a_context_manager_it_releases_on_exit_even_when_the_body_fails() -> None:
    client = FakeNatsClient()
    with pytest.raises(RuntimeError, match="body"):
        async with await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW) as held:
            assert isinstance(held, HeldLease)
            raise RuntimeError("body")
    assert await (await _bucket(client)).get_entry(key="job") is None


async def test_a_release_that_cannot_reach_the_bucket_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeNatsClient()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW)
    _make_unreachable(await _bucket(client), monkeypatch)["down"] = True
    await held.release()  # the TTL frees the entry; a cleanup error must not replace the caller's outcome


async def test_wait_bound_is_honoured() -> None:
    client = FakeNatsClient()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW)
    try:
        with pytest.raises(LeaseUnavailable):
            await _lease(client, "pod-b").hold("job", ttl=_TTL, renew_every=_RENEW, max_wait_seconds=0)
    finally:
        await held.release()


@pytest.mark.parametrize(
    ("ttl", "renew_every", "match"),
    [
        (timedelta(seconds=0.5), timedelta(seconds=0.1), "whole number of seconds"),
        (timedelta(seconds=1), timedelta(seconds=1), "shorter than ttl"),
        (timedelta(seconds=1), timedelta(0), "renew_every must be positive"),
    ],
)
async def test_configuration_that_cannot_hold_is_refused(ttl: timedelta, renew_every: timedelta, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        await _lease(FakeNatsClient(), "pod-a").hold("job", ttl=ttl, renew_every=renew_every)
