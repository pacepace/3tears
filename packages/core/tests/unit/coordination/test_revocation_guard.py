"""tests for :class:`RevocationGuard`: shared, fail-closed timestamped revocation entries.

The contract this pins:

- recording a revocation and reading it back round-trips the exact ``revoked_at`` moment;
- ``is_revoked_before(key, moment=...)`` is ``True`` iff ``moment < revoked_at`` -- a session that
  started BEFORE the revocation is denylisted; one that starts AT or AFTER it is not
  (``data-model.md``'s "Revocation denylist entries");
- a key with no revocation entry is NOT revoked (``is_revoked_before`` returns ``False``, never
  raises) -- default-allow for the absent case;
- a second ``record_revocation`` call for the same key OVERWRITES the stored moment (unconditional
  write, not create-if-absent -- distinct from :class:`ReplayGuard`'s create-if-absent shape);
- it is FAIL-CLOSED on KV transport failure (propagates, never silently answers "not revoked");
- naive (timezone-unaware) timestamps are rejected at the API boundary -- comparing a naive moment
  against a stored aware one would silently misbehave;
- the bucket is opened with the configured TTL and ``file`` storage, same durability posture as
  :class:`ReplayGuard`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.core.coordination import RevocationGuard
from threetears.nats import KvError

from ._fake_kv import FakeNatsClient


@pytest.fixture
def client() -> FakeNatsClient:
    return FakeNatsClient()


_T0 = datetime(2026, 1, 1, tzinfo=UTC)


class TestRevocationGuard:
    @pytest.mark.asyncio
    async def test_record_and_read_back_round_trips(self, client: FakeNatsClient) -> None:
        guard = RevocationGuard(client, bucket_name="revocations", ttl_seconds=3600)  # type: ignore[arg-type]
        await guard.record_revocation("sub:1", revoked_at=_T0)
        assert await guard.revoked_at("sub:1") == _T0

    @pytest.mark.asyncio
    async def test_unrevoked_key_has_no_stored_timestamp(self, client: FakeNatsClient) -> None:
        guard = RevocationGuard(client, bucket_name="revocations", ttl_seconds=3600)  # type: ignore[arg-type]
        assert await guard.revoked_at("sub:never-revoked") is None

    @pytest.mark.asyncio
    async def test_unrevoked_key_is_not_revoked_before_anything(self, client: FakeNatsClient) -> None:
        guard = RevocationGuard(client, bucket_name="revocations", ttl_seconds=3600)  # type: ignore[arg-type]
        assert await guard.is_revoked_before("sub:never-revoked", moment=_T0) is False

    @pytest.mark.asyncio
    async def test_session_started_before_revocation_is_blocked(self, client: FakeNatsClient) -> None:
        guard = RevocationGuard(client, bucket_name="revocations", ttl_seconds=3600)  # type: ignore[arg-type]
        revoked_at = _T0
        session_started_at = _T0 - timedelta(minutes=5)  # started BEFORE the revocation
        await guard.record_revocation("sub:1", revoked_at=revoked_at)
        assert await guard.is_revoked_before("sub:1", moment=session_started_at) is True

    @pytest.mark.asyncio
    async def test_session_started_after_revocation_is_unaffected(self, client: FakeNatsClient) -> None:
        guard = RevocationGuard(client, bucket_name="revocations", ttl_seconds=3600)  # type: ignore[arg-type]
        revoked_at = _T0
        session_started_at = _T0 + timedelta(minutes=5)  # started AFTER the revocation
        await guard.record_revocation("sub:1", revoked_at=revoked_at)
        assert await guard.is_revoked_before("sub:1", moment=session_started_at) is False

    @pytest.mark.asyncio
    async def test_session_started_exactly_at_revocation_is_unaffected(self, client: FakeNatsClient) -> None:
        # the check is strict `<`, not `<=` -- a session starting AT the exact revocation instant
        # is not "before" it.
        guard = RevocationGuard(client, bucket_name="revocations", ttl_seconds=3600)  # type: ignore[arg-type]
        await guard.record_revocation("sub:1", revoked_at=_T0)
        assert await guard.is_revoked_before("sub:1", moment=_T0) is False

    @pytest.mark.asyncio
    async def test_distinct_keys_do_not_affect_each_other(self, client: FakeNatsClient) -> None:
        guard = RevocationGuard(client, bucket_name="revocations", ttl_seconds=3600)  # type: ignore[arg-type]
        await guard.record_revocation("sub:1", revoked_at=_T0)
        assert await guard.is_revoked_before("sub:2", moment=_T0 - timedelta(minutes=5)) is False

    @pytest.mark.asyncio
    async def test_second_record_call_overwrites_the_stored_moment(self, client: FakeNatsClient) -> None:
        # unconditional write, unlike ReplayGuard.record_unique's create-if-absent: a later
        # revocation call (re-run offboarding, narrowed cutoff) replaces the effective moment.
        guard = RevocationGuard(client, bucket_name="revocations", ttl_seconds=3600)  # type: ignore[arg-type]
        await guard.record_revocation("customer_id:1", revoked_at=_T0)
        later_cutoff = _T0 + timedelta(days=1)
        await guard.record_revocation("customer_id:1", revoked_at=later_cutoff)
        assert await guard.revoked_at("customer_id:1") == later_cutoff
        # a session that started between the two cutoffs is now unaffected, since the effective
        # cutoff moved later.
        assert await guard.is_revoked_before("customer_id:1", moment=_T0 + timedelta(hours=1)) is True

    @pytest.mark.asyncio
    async def test_hashed_keys_keep_similar_keys_distinct(self, client: FakeNatsClient) -> None:
        guard = RevocationGuard(client, bucket_name="revocations", ttl_seconds=3600)  # type: ignore[arg-type]
        await guard.record_revocation("sub:1", revoked_at=_T0)
        assert await guard.revoked_at("sub:11") is None

    @pytest.mark.asyncio
    async def test_transport_failure_propagates_fail_closed_on_record(self) -> None:
        bucket = AsyncMock()
        bucket.put = AsyncMock(side_effect=KvError("kv down"))
        failing_client = AsyncMock()
        failing_client.kv_bucket = AsyncMock(return_value=bucket)
        guard = RevocationGuard(failing_client, bucket_name="b", ttl_seconds=120)
        with pytest.raises(KvError):
            await guard.record_revocation("sub:1", revoked_at=_T0)

    @pytest.mark.asyncio
    async def test_transport_failure_propagates_fail_closed_on_check(self) -> None:
        bucket = AsyncMock()
        bucket.get = AsyncMock(side_effect=KvError("kv down"))
        failing_client = AsyncMock()
        failing_client.kv_bucket = AsyncMock(return_value=bucket)
        guard = RevocationGuard(failing_client, bucket_name="b", ttl_seconds=120)
        with pytest.raises(KvError):
            await guard.is_revoked_before("sub:1", moment=_T0)

    @pytest.mark.asyncio
    async def test_naive_revoked_at_is_rejected(self, client: FakeNatsClient) -> None:
        guard = RevocationGuard(client, bucket_name="revocations", ttl_seconds=3600)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="timezone-aware"):
            await guard.record_revocation("sub:1", revoked_at=datetime(2026, 1, 1))  # noqa: DTZ001

    @pytest.mark.asyncio
    async def test_naive_moment_is_rejected(self, client: FakeNatsClient) -> None:
        guard = RevocationGuard(client, bucket_name="revocations", ttl_seconds=3600)  # type: ignore[arg-type]
        await guard.record_revocation("sub:1", revoked_at=_T0)
        with pytest.raises(ValueError, match="timezone-aware"):
            await guard.is_revoked_before("sub:1", moment=datetime(2026, 1, 1))  # noqa: DTZ001

    @pytest.mark.asyncio
    async def test_bucket_opened_with_the_configured_ttl(self) -> None:
        bucket = AsyncMock()
        bucket.put = AsyncMock(return_value=1)
        spy_client = AsyncMock()
        spy_client.kv_bucket = AsyncMock(return_value=bucket)
        guard = RevocationGuard(spy_client, bucket_name="b", ttl_seconds=90)
        await guard.record_revocation("sub:1", revoked_at=_T0)
        kwargs = spy_client.kv_bucket.call_args.kwargs
        assert kwargs["ttl"] == timedelta(seconds=90)
        assert kwargs["storage"] == "file"

    @pytest.mark.asyncio
    async def test_bucket_bound_once_across_calls(self) -> None:
        bucket = AsyncMock()
        bucket.put = AsyncMock(return_value=1)
        spy_client = AsyncMock()
        spy_client.kv_bucket = AsyncMock(return_value=bucket)
        guard = RevocationGuard(spy_client, bucket_name="b", ttl_seconds=120)
        await guard.record_revocation("sub:1", revoked_at=_T0)
        await guard.record_revocation("sub:2", revoked_at=_T0)
        spy_client.kv_bucket.assert_awaited_once()

    def test_non_positive_ttl_rejected(self) -> None:
        with pytest.raises(ValueError):
            RevocationGuard(MagicMock(), bucket_name="b", ttl_seconds=0)
        with pytest.raises(ValueError):
            RevocationGuard(MagicMock(), bucket_name="b", ttl_seconds=-5)
