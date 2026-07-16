"""tests for :class:`WindowedCounter`: generic windowed attempt counter/rate-limiter.

The contract this pins:

- a fresh key's first attempt records count=1; repeated attempts within the window increment;
- distinct keys count independently;
- the counter is SHARED, not per-process: two instances over the same bucket see each other's
  recorded attempts (mirrors `ReplayGuard`'s cross-replica contract);
- keys are hashed before storage (raw identifier never stored as a KV key);
- `count`/`is_over_threshold` are read-only -- they never themselves record an attempt;
- an expired window resets the count back to 1 on the next attempt, not to count+1;
- the bucket is opened with the window TTL and ``storage="file"``;
- a non-positive ``window_seconds`` is rejected at construction;
- fail-closed (default) propagates `KvError`; fail-open swallows it and returns 0 instead.
"""

from __future__ import annotations

import asyncio
import time
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.core.coordination import WindowedCounter
from threetears.nats import KvError

from ._fake_kv import FakeNatsClient


@pytest.fixture
def client() -> FakeNatsClient:
    return FakeNatsClient()


class TestWindowedCounter:
    @pytest.mark.asyncio
    async def test_first_attempt_records_count_one(self, client: FakeNatsClient) -> None:
        counter = WindowedCounter(client, bucket_name="throttle", window_seconds=60)  # type: ignore[arg-type]
        assert await counter.record_attempt("k") == 1

    @pytest.mark.asyncio
    async def test_repeated_attempts_increment_within_window(self, client: FakeNatsClient) -> None:
        counter = WindowedCounter(client, bucket_name="throttle", window_seconds=60)  # type: ignore[arg-type]
        assert await counter.record_attempt("k") == 1
        assert await counter.record_attempt("k") == 2
        assert await counter.record_attempt("k") == 3

    @pytest.mark.asyncio
    async def test_distinct_keys_count_independently(self, client: FakeNatsClient) -> None:
        counter = WindowedCounter(client, bucket_name="throttle", window_seconds=60)  # type: ignore[arg-type]
        assert await counter.record_attempt("a") == 1
        assert await counter.record_attempt("b") == 1
        assert await counter.record_attempt("a") == 2

    @pytest.mark.asyncio
    async def test_shared_across_instances_over_the_same_bucket(self, client: FakeNatsClient) -> None:
        c1 = WindowedCounter(client, bucket_name="shared", window_seconds=60)  # type: ignore[arg-type]
        c2 = WindowedCounter(client, bucket_name="shared", window_seconds=60)  # type: ignore[arg-type]
        assert await c1.record_attempt("k") == 1
        assert await c2.record_attempt("k") == 2

    @pytest.mark.asyncio
    async def test_read_only_methods_do_not_record(self, client: FakeNatsClient) -> None:
        counter = WindowedCounter(client, bucket_name="throttle", window_seconds=60)  # type: ignore[arg-type]
        assert await counter.count("k") == 0
        assert await counter.is_over_threshold("k", threshold=1) is False
        await counter.record_attempt("k")
        assert await counter.count("k") == 1
        assert await counter.count("k") == 1  # still 1 -- count() never increments
        assert await counter.is_over_threshold("k", threshold=1) is True
        assert await counter.is_over_threshold("k", threshold=2) is False

    @pytest.mark.asyncio
    async def test_expired_window_resets_to_one(self, client: FakeNatsClient) -> None:
        counter = WindowedCounter(client, bucket_name="throttle", window_seconds=60)  # type: ignore[arg-type]
        assert await counter.record_attempt("k") == 1
        assert await counter.record_attempt("k") == 2
        # simulate the window having elapsed by rewriting the stored window_start into the past.
        bucket = await counter._ensure_bucket()  # noqa: SLF001 - test reaches into internals deliberately
        kv_key = counter._key("k")  # noqa: SLF001
        entry = await bucket.get_entry(key=kv_key)
        assert entry is not None
        value, revision = entry
        import json

        payload = json.loads(value)
        payload["window_start"] = time.time() - 3600
        await bucket.update(key=kv_key, value=json.dumps(payload).encode("utf-8"), revision=revision)
        assert await counter.record_attempt("k") == 1

    @pytest.mark.asyncio
    async def test_hashed_keys_not_stored_raw(self, client: FakeNatsClient) -> None:
        counter = WindowedCounter(client, bucket_name="throttle", window_seconds=60)  # type: ignore[arg-type]
        await counter.record_attempt("super-secret-identifier")
        bucket = await counter._ensure_bucket()  # noqa: SLF001
        assert "super-secret-identifier" not in bucket._entries  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_transport_failure_propagates_fail_closed_by_default(self) -> None:
        bucket = AsyncMock()
        bucket.get_entry = AsyncMock(side_effect=KvError("kv down"))
        failing_client = AsyncMock()
        failing_client.kv_bucket = AsyncMock(return_value=bucket)
        counter = WindowedCounter(failing_client, bucket_name="b", window_seconds=60)
        with pytest.raises(KvError):
            await counter.record_attempt("x")

    @pytest.mark.asyncio
    async def test_transport_failure_swallowed_when_fail_open(self) -> None:
        bucket = AsyncMock()
        bucket.get_entry = AsyncMock(side_effect=KvError("kv down"))
        bucket.get = AsyncMock(side_effect=KvError("kv down"))
        failing_client = AsyncMock()
        failing_client.kv_bucket = AsyncMock(return_value=bucket)
        counter = WindowedCounter(failing_client, bucket_name="b", window_seconds=60, fail_open=True)
        assert await counter.record_attempt("x") == 0
        assert await counter.count("x") == 0
        assert await counter.is_over_threshold("x", threshold=1) is False

    @pytest.mark.asyncio
    async def test_bucket_opened_with_window_ttl_and_file_storage(self) -> None:
        bucket = AsyncMock()
        bucket.get_entry = AsyncMock(return_value=None)
        bucket.create = AsyncMock(return_value=1)
        spy_client = AsyncMock()
        spy_client.kv_bucket = AsyncMock(return_value=bucket)
        counter = WindowedCounter(spy_client, bucket_name="b", window_seconds=90)
        await counter.record_attempt("x")
        kwargs = spy_client.kv_bucket.call_args.kwargs
        assert kwargs["ttl"] == timedelta(seconds=90)
        assert kwargs["storage"] == "file"

    @pytest.mark.asyncio
    async def test_bucket_bound_once_across_calls(self) -> None:
        bucket = AsyncMock()
        bucket.get_entry = AsyncMock(return_value=None)
        bucket.create = AsyncMock(return_value=1)
        spy_client = AsyncMock()
        spy_client.kv_bucket = AsyncMock(return_value=bucket)
        counter = WindowedCounter(spy_client, bucket_name="b", window_seconds=60)
        await counter.record_attempt("a")
        await counter.record_attempt("b")
        spy_client.kv_bucket.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_concurrent_attempts_on_same_key_all_counted(self, client: FakeNatsClient) -> None:
        counter = WindowedCounter(client, bucket_name="race", window_seconds=60)  # type: ignore[arg-type]
        results = await asyncio.gather(*[counter.record_attempt("dup") for _ in range(8)])
        assert sorted(results) == list(range(1, 9))

    def test_non_positive_window_rejected(self) -> None:
        with pytest.raises(ValueError):
            WindowedCounter(MagicMock(), bucket_name="b", window_seconds=0)
        with pytest.raises(ValueError):
            WindowedCounter(MagicMock(), bucket_name="b", window_seconds=-5)

    def test_fail_open_property_reflects_constructor_arg(self) -> None:
        assert WindowedCounter(MagicMock(), bucket_name="b", window_seconds=60).fail_open is False
        assert WindowedCounter(MagicMock(), bucket_name="b", window_seconds=60, fail_open=True).fail_open is True
