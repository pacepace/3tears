"""tests for :class:`IdempotencyKeyStore`: claim-once-with-TTL over NATS JetStream KV.

The contract this pins:

- a fresh key claims and returns status="claimed" with a pending record;
- the SAME key claimed again returns status="exists" with the original
  claimer's record (not a second claim);
- claim is atomic under concurrency: N concurrent claimers on the same
  key, exactly one gets "claimed";
- complete()/fail() transition a claimed key to a terminal state and
  store the result/error, retrying under CAS contention;
- complete()/fail() raise IdempotencyKeyNotFound for a key that was
  never claimed;
- get() returns None for an unknown key, and the current record
  otherwise;
- the bucket is opened with the configured TTL.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

from threetears.core.coordination import (
    IdempotencyConflict,
    IdempotencyKeyNotFound,
    IdempotencyKeyStore,
)

from ._fake_kv import FakeNatsClient


@pytest.fixture
def client() -> FakeNatsClient:
    return FakeNatsClient()


class TestClaim:
    @pytest.mark.asyncio
    async def test_fresh_key_is_claimed(self, client: FakeNatsClient) -> None:
        store = IdempotencyKeyStore(client, bucket_name="jobs")  # type: ignore[arg-type]
        outcome = await store.claim("key-1")
        assert outcome.status == "claimed"
        assert outcome.record.key == "key-1"
        assert outcome.record.status == "pending"
        assert outcome.record.result is None
        assert outcome.record.error is None
        assert outcome.record.date_completed is None

    @pytest.mark.asyncio
    async def test_same_key_twice_second_call_sees_existing_claim(self, client: FakeNatsClient) -> None:
        store = IdempotencyKeyStore(client, bucket_name="jobs")  # type: ignore[arg-type]
        first = await store.claim("key-1")
        second = await store.claim("key-1")
        assert first.status == "claimed"
        assert second.status == "exists"
        assert second.record.status == "pending"

    @pytest.mark.asyncio
    async def test_distinct_keys_each_claimed(self, client: FakeNatsClient) -> None:
        store = IdempotencyKeyStore(client, bucket_name="jobs")  # type: ignore[arg-type]
        a = await store.claim("a")
        b = await store.claim("b")
        assert a.status == "claimed"
        assert b.status == "claimed"

    @pytest.mark.asyncio
    async def test_concurrent_claim_of_same_key_has_one_winner(self, client: FakeNatsClient) -> None:
        store = IdempotencyKeyStore(client, bucket_name="race")  # type: ignore[arg-type]
        outcomes = await asyncio.gather(*[store.claim("dup") for _ in range(8)])
        statuses = [o.status for o in outcomes]
        assert statuses.count("claimed") == 1
        assert statuses.count("exists") == 7

    @pytest.mark.asyncio
    async def test_exists_returns_existing_claimers_completed_result(self, client: FakeNatsClient) -> None:
        store = IdempotencyKeyStore(client, bucket_name="jobs")  # type: ignore[arg-type]
        await store.claim("key-1")
        await store.complete("key-1", result=b"done")

        outcome = await store.claim("key-1")

        assert outcome.status == "exists"
        assert outcome.record.status == "completed"
        assert outcome.record.result == b"done"

    @pytest.mark.asyncio
    async def test_claim_stores_metadata(self, client: FakeNatsClient) -> None:
        store = IdempotencyKeyStore(client, bucket_name="jobs")  # type: ignore[arg-type]
        outcome = await store.claim("key-1", metadata=b"request-hash-abc")

        assert outcome.record.metadata == b"request-hash-abc"

    @pytest.mark.asyncio
    async def test_claim_without_metadata_defaults_to_none(self, client: FakeNatsClient) -> None:
        store = IdempotencyKeyStore(client, bucket_name="jobs")  # type: ignore[arg-type]
        outcome = await store.claim("key-1")

        assert outcome.record.metadata is None

    @pytest.mark.asyncio
    async def test_exists_outcome_returns_original_claimers_metadata(self, client: FakeNatsClient) -> None:
        store = IdempotencyKeyStore(client, bucket_name="jobs")  # type: ignore[arg-type]
        await store.claim("key-1", metadata=b"request-hash-abc")

        outcome = await store.claim("key-1", metadata=b"different-hash")

        assert outcome.status == "exists"
        assert outcome.record.metadata == b"request-hash-abc"  # original claimer's, not the second caller's


class TestComplete:
    @pytest.mark.asyncio
    async def test_complete_stores_result_and_terminal_state(self, client: FakeNatsClient) -> None:
        store = IdempotencyKeyStore(client, bucket_name="jobs")  # type: ignore[arg-type]
        await store.claim("key-1")

        await store.complete("key-1", result=b"the-result")

        record = await store.get("key-1")
        assert record is not None
        assert record.status == "completed"
        assert record.result == b"the-result"
        assert record.error is None
        assert record.date_completed is not None

    @pytest.mark.asyncio
    async def test_complete_preserves_original_date_claimed(self, client: FakeNatsClient) -> None:
        store = IdempotencyKeyStore(client, bucket_name="jobs")  # type: ignore[arg-type]
        outcome = await store.claim("key-1")
        original_claimed = outcome.record.date_claimed

        await store.complete("key-1", result=b"x")

        record = await store.get("key-1")
        assert record is not None
        assert record.date_claimed == original_claimed

    @pytest.mark.asyncio
    async def test_complete_preserves_claim_time_metadata(self, client: FakeNatsClient) -> None:
        store = IdempotencyKeyStore(client, bucket_name="jobs")  # type: ignore[arg-type]
        await store.claim("key-1", metadata=b"request-hash-abc")

        await store.complete("key-1", result=b"x")

        record = await store.get("key-1")
        assert record is not None
        assert record.metadata == b"request-hash-abc"

    @pytest.mark.asyncio
    async def test_complete_unclaimed_key_raises(self, client: FakeNatsClient) -> None:
        store = IdempotencyKeyStore(client, bucket_name="jobs")  # type: ignore[arg-type]
        with pytest.raises(IdempotencyKeyNotFound):
            await store.complete("never-claimed", result=b"x")

    @pytest.mark.asyncio
    async def test_complete_retries_under_cas_contention(self) -> None:
        bucket = AsyncMock()
        bucket.get_entry = AsyncMock(
            side_effect=[
                (
                    b'{"key": "k", "status": "pending", "result": null, "error": null, '
                    b'"date_claimed": "2026-01-01T00:00:00+00:00", "date_completed": null}',
                    1,
                ),
                (
                    b'{"key": "k", "status": "pending", "result": null, "error": null, '
                    b'"date_claimed": "2026-01-01T00:00:00+00:00", "date_completed": null}',
                    2,
                ),
            ]
        )
        bucket.update = AsyncMock(side_effect=[None, 3])  # first CAS attempt loses, second wins
        client = AsyncMock()
        client.kv_bucket = AsyncMock(return_value=bucket)

        store = IdempotencyKeyStore(client, bucket_name="jobs")
        await store.complete("k", result=b"result")

        assert bucket.update.await_count == 2

    @pytest.mark.asyncio
    async def test_complete_raises_when_cas_retry_budget_exhausted(self) -> None:
        bucket = AsyncMock()
        bucket.get_entry = AsyncMock(
            return_value=(
                b'{"key": "k", "status": "pending", "result": null, "error": null, '
                b'"date_claimed": "2026-01-01T00:00:00+00:00", "date_completed": null}',
                1,
            )
        )
        bucket.update = AsyncMock(return_value=None)  # every CAS attempt loses
        client = AsyncMock()
        client.kv_bucket = AsyncMock(return_value=bucket)

        store = IdempotencyKeyStore(client, bucket_name="jobs")
        with pytest.raises(IdempotencyConflict):
            await store.complete("k", result=b"result")


class TestFail:
    @pytest.mark.asyncio
    async def test_fail_stores_error_and_terminal_state(self, client: FakeNatsClient) -> None:
        store = IdempotencyKeyStore(client, bucket_name="jobs")  # type: ignore[arg-type]
        await store.claim("key-1")

        await store.fail("key-1", error="boom")

        record = await store.get("key-1")
        assert record is not None
        assert record.status == "failed"
        assert record.error == "boom"
        assert record.result is None

    @pytest.mark.asyncio
    async def test_fail_unclaimed_key_raises(self, client: FakeNatsClient) -> None:
        store = IdempotencyKeyStore(client, bucket_name="jobs")  # type: ignore[arg-type]
        with pytest.raises(IdempotencyKeyNotFound):
            await store.fail("never-claimed", error="boom")


class TestGet:
    @pytest.mark.asyncio
    async def test_get_unknown_key_returns_none(self, client: FakeNatsClient) -> None:
        store = IdempotencyKeyStore(client, bucket_name="jobs")  # type: ignore[arg-type]
        assert await store.get("unknown") is None

    @pytest.mark.asyncio
    async def test_get_pending_key_returns_pending_record(self, client: FakeNatsClient) -> None:
        store = IdempotencyKeyStore(client, bucket_name="jobs")  # type: ignore[arg-type]
        await store.claim("key-1")
        record = await store.get("key-1")
        assert record is not None
        assert record.status == "pending"


class TestBucketConfiguration:
    @pytest.mark.asyncio
    async def test_bucket_opened_with_configured_ttl(self) -> None:
        bucket = AsyncMock()
        bucket.create = AsyncMock(return_value=1)
        spy_client = AsyncMock()
        spy_client.kv_bucket = AsyncMock(return_value=bucket)
        store = IdempotencyKeyStore(spy_client, bucket_name="b", ttl=timedelta(hours=6))
        await store.claim("x")
        kwargs = spy_client.kv_bucket.call_args.kwargs
        assert kwargs["ttl"] == timedelta(hours=6)

    @pytest.mark.asyncio
    async def test_default_ttl_is_24_hours(self) -> None:
        bucket = AsyncMock()
        bucket.create = AsyncMock(return_value=1)
        spy_client = AsyncMock()
        spy_client.kv_bucket = AsyncMock(return_value=bucket)
        store = IdempotencyKeyStore(spy_client, bucket_name="b")
        await store.claim("x")
        kwargs = spy_client.kv_bucket.call_args.kwargs
        assert kwargs["ttl"] == timedelta(hours=24)

    @pytest.mark.asyncio
    async def test_bucket_bound_once_across_calls(self) -> None:
        bucket = AsyncMock()
        bucket.create = AsyncMock(return_value=1)
        spy_client = AsyncMock()
        spy_client.kv_bucket = AsyncMock(return_value=bucket)
        store = IdempotencyKeyStore(spy_client, bucket_name="b")
        await store.claim("a")
        await store.claim("b")
        spy_client.kv_bucket.assert_awaited_once()
