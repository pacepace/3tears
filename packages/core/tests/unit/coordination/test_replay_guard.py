"""tests for :class:`ReplayGuard`: shared, fail-closed single-use nonce protection.

The contract this pins:

- a fresh nonce records and returns True; the SAME nonce again returns False (replay);
- the cache is SHARED, not per-process: a nonce recorded by one guard instance is a replay for
  another instance over the same bucket (an in-process set would miss the cross-replica replay);
- it is FAIL-CLOSED: a KV transport failure propagates (never silently answers "fresh");
- a WIPED bucket fails closed too: an artifact issued before the bucket's current creation time,
  plus the verifier's future tolerance and the host drift allowance, is refused even though its
  nonce is not recorded;
- a guard refuses to serve a verifier whose future tolerance it was not sized for;
- the bucket is memory-backed and opened with the accept-window TTL so nonces self-expire;
- ``bind`` opens the bucket once, however many callers race it, and a service that binds at start
  moves the bucket's creation time -- and so the watermark -- to before anything it serves;
- construction rejects a non-positive TTL and a negative tolerance; a naive issue time raises.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.core.coordination import ReplayGuard
from threetears.core.coordination import replay_guard as replay_guard_module
from threetears.core.coordination.replay_guard import CLOCK_DRIFT_ALLOWANCE
from threetears.core.testing import kv as fake_kv_module
from threetears.nats import KvError

from threetears.core.testing.kv import FakeNatsClient

_SKEW = timedelta(seconds=60)
# how far past a bucket's creation time the guard refuses, for a guard built with _SKEW.
_REACH = _SKEW + CLOCK_DRIFT_ALLOWANCE


def _later() -> datetime:
    """an issue time comfortably after any bucket the fake created during this test.

    :return: now plus twice the refusal reach
    :rtype: datetime
    """
    return datetime.now(UTC) + 2 * _REACH


def _guard(client: object, *, bucket_name: str = "pop_nonces", ttl_seconds: int = 120) -> ReplayGuard:
    """build a guard sized for the test verifier's future tolerance.

    :param client: the KV-capable client double
    :ptype client: object
    :param bucket_name: bucket suffix
    :ptype bucket_name: str
    :param ttl_seconds: nonce TTL
    :ptype ttl_seconds: int
    :return: the guard
    :rtype: ReplayGuard
    """
    return ReplayGuard(client, bucket_name=bucket_name, ttl_seconds=ttl_seconds, verifier_future_tolerance=_SKEW)  # type: ignore[arg-type]


@pytest.fixture
def client() -> FakeNatsClient:
    return FakeNatsClient()


class TestReplayGuard:
    @pytest.mark.asyncio
    async def test_fresh_nonce_is_recorded(self, client: FakeNatsClient) -> None:
        assert await _guard(client).record_unique("nonce-1", issued_at=_later()) is True

    @pytest.mark.asyncio
    async def test_same_nonce_twice_is_replay(self, client: FakeNatsClient) -> None:
        guard = _guard(client)
        assert await guard.record_unique("n", issued_at=_later()) is True
        assert await guard.record_unique("n", issued_at=_later()) is False

    @pytest.mark.asyncio
    async def test_distinct_nonces_each_fresh(self, client: FakeNatsClient) -> None:
        guard = _guard(client)
        assert await guard.record_unique("a", issued_at=_later()) is True
        assert await guard.record_unique("b", issued_at=_later()) is True

    @pytest.mark.asyncio
    async def test_replay_detected_across_instances_sharing_a_bucket(self, client: FakeNatsClient) -> None:
        # two guards == two replica processes against the SAME shared bucket; a nonce recorded by
        # one must be a replay for the other. an in-process set could never catch this.
        g1 = _guard(client, bucket_name="shared")
        g2 = _guard(client, bucket_name="shared")
        assert await g1.record_unique("x", issued_at=_later()) is True
        assert await g2.record_unique("x", issued_at=_later()) is False

    @pytest.mark.asyncio
    async def test_hashed_keys_keep_similar_nonces_distinct(self, client: FakeNatsClient) -> None:
        guard = _guard(client, bucket_name="b")
        assert await guard.record_unique("nonce-1", issued_at=_later()) is True
        assert await guard.record_unique("nonce-2", issued_at=_later()) is True
        assert await guard.record_unique("nonce-1", issued_at=_later()) is False  # the first is now a replay

    @pytest.mark.asyncio
    async def test_transport_failure_propagates_fail_closed(self) -> None:
        bucket = AsyncMock()
        bucket.create = AsyncMock(side_effect=KvError("kv down"))
        failing_client = AsyncMock()
        failing_client.kv_bucket = AsyncMock(return_value=bucket)
        guard = _guard(failing_client, bucket_name="b")
        with pytest.raises(KvError):
            await guard.record_unique("x", issued_at=_later())

    @pytest.mark.asyncio
    async def test_creation_time_read_failure_propagates_fail_closed(self) -> None:
        # the nonce was recorded, but without the creation time the guard cannot rule out a wipe:
        # that must deny, never admit.
        bucket = AsyncMock()
        bucket.create = AsyncMock(return_value=1)
        bucket.date_created = AsyncMock(side_effect=KvError("no creation time"))
        failing_client = AsyncMock()
        failing_client.kv_bucket = AsyncMock(return_value=bucket)
        guard = _guard(failing_client, bucket_name="b")
        with pytest.raises(KvError):
            await guard.record_unique("x", issued_at=_later())

    @pytest.mark.asyncio
    async def test_bucket_opened_memory_backed_with_the_accept_window_ttl(self) -> None:
        bucket = AsyncMock()
        bucket.create = AsyncMock(return_value=1)
        bucket.date_created = AsyncMock(return_value=datetime.now(UTC))
        spy_client = AsyncMock()
        spy_client.kv_bucket = AsyncMock(return_value=bucket)
        guard = _guard(spy_client, bucket_name="b", ttl_seconds=90)
        await guard.record_unique("x", issued_at=_later())
        kwargs = spy_client.kv_bucket.call_args.kwargs
        assert kwargs["ttl"] == timedelta(seconds=90)
        # NATS is L2 and memory-only; a wipe is detected by the creation-time check, not survived.
        assert kwargs.get("storage", "memory") == "memory"

    @pytest.mark.asyncio
    async def test_bucket_bound_once_across_calls(self) -> None:
        bucket = AsyncMock()
        bucket.create = AsyncMock(return_value=1)
        bucket.date_created = AsyncMock(return_value=datetime.now(UTC))
        spy_client = AsyncMock()
        spy_client.kv_bucket = AsyncMock(return_value=bucket)
        guard = _guard(spy_client, bucket_name="b")
        await guard.record_unique("a", issued_at=_later())
        await guard.record_unique("b", issued_at=_later())
        spy_client.kv_bucket.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_replay_is_refused_without_reading_the_creation_time(self) -> None:
        # a present nonce is already a refusal; paying a second round trip for it is waste.
        bucket = AsyncMock()
        bucket.create = AsyncMock(return_value=None)
        bucket.date_created = AsyncMock(return_value=datetime.now(UTC))
        spy_client = AsyncMock()
        spy_client.kv_bucket = AsyncMock(return_value=bucket)
        guard = _guard(spy_client, bucket_name="b")
        assert await guard.record_unique("x", issued_at=_later()) is False
        bucket.date_created.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_concurrent_record_of_same_nonce_has_one_winner(self, client: FakeNatsClient) -> None:
        # CAS create-if-absent guarantees exactly one "fresh" even when the same nonce is recorded
        # concurrently -- e.g. a replay racing the original.
        guard = _guard(client, bucket_name="race")
        issued_at = _later()
        results = await asyncio.gather(*[guard.record_unique("dup", issued_at=issued_at) for _ in range(8)])
        assert results.count(True) == 1
        assert results.count(False) == 7

    def test_non_positive_ttl_rejected(self) -> None:
        with pytest.raises(ValueError):
            ReplayGuard(MagicMock(), bucket_name="b", ttl_seconds=0, verifier_future_tolerance=_SKEW)
        with pytest.raises(ValueError):
            ReplayGuard(MagicMock(), bucket_name="b", ttl_seconds=-5, verifier_future_tolerance=_SKEW)

    def test_negative_verifier_future_tolerance_rejected(self) -> None:
        with pytest.raises(ValueError):
            ReplayGuard(MagicMock(), bucket_name="b", ttl_seconds=60, verifier_future_tolerance=timedelta(seconds=-1))


class TestRequireCovers:
    """a guard refuses to serve a verifier that accepts issue times further ahead than it expects."""

    def test_a_verifier_within_the_sized_tolerance_is_served(self) -> None:
        guard = ReplayGuard(MagicMock(), bucket_name="b", ttl_seconds=60, verifier_future_tolerance=_SKEW)
        guard.require_covers(_SKEW)
        guard.require_covers(timedelta(0))
        assert guard.verifier_future_tolerance == _SKEW

    def test_a_verifier_with_a_wider_tolerance_is_refused(self) -> None:
        # the widened leeway would let a replay stamped at that edge past the wipe check; it must
        # fail where the verifier is wired, naming the fix.
        guard = ReplayGuard(MagicMock(), bucket_name="b", ttl_seconds=60, verifier_future_tolerance=_SKEW)
        with pytest.raises(ValueError, match="verifier_future_tolerance"):
            guard.require_covers(_SKEW + timedelta(seconds=1))

    @pytest.mark.asyncio
    async def test_naive_issued_at_rejected(self, client: FakeNatsClient) -> None:
        with pytest.raises(ValueError):
            await _guard(client).record_unique("x", issued_at=datetime.now())  # noqa: DTZ005 - the naive value is the case under test


class TestReplayGuardAfterAWipe:
    """a wiped bucket must not become a window in which a spent artifact is fresh again."""

    @pytest.mark.asyncio
    async def test_unseen_artifact_issued_before_the_wipe_is_refused(self, client: FakeNatsClient) -> None:
        # the guard cannot tell a never-seen artifact from one whose record the wipe erased, so
        # anything issued before the bucket's creation time is refused.
        bucket = await client.kv_bucket(name="pop_nonces")
        issued_at = datetime.now(UTC)
        bucket.wipe(date_created=issued_at + timedelta(seconds=1))
        assert await _guard(client).record_unique("unseen", issued_at=issued_at) is False

    @pytest.mark.asyncio
    async def test_a_replay_of_an_admitted_artifact_is_refused_across_a_wipe(self, client: FakeNatsClient) -> None:
        # the realistic sequence: the bucket exists, a proof is admitted, the broker restarts, and
        # the SAME proof is presented again through a handle that never saw an error.
        bucket = await client.kv_bucket(name="pop_nonces")
        created = datetime.now(UTC)
        bucket.wipe(date_created=created)
        guard = _guard(client)
        issued_at = created + 2 * _REACH
        assert await guard.record_unique("proof-1", issued_at=issued_at) is True
        bucket.wipe(date_created=issued_at + timedelta(seconds=30))
        assert await guard.record_unique("proof-1", issued_at=issued_at) is False

    @pytest.mark.asyncio
    async def test_artifact_within_the_tolerance_of_the_wipe_is_refused(self, client: FakeNatsClient) -> None:
        # an issue time just after the creation time could still be a replay stamped by a clock
        # running ahead; inside the verifier's tolerance it must be refused.
        bucket = await client.kv_bucket(name="pop_nonces")
        created = datetime.now(UTC)
        bucket.wipe(date_created=created)
        guard = _guard(client)
        assert await guard.record_unique("n", issued_at=created + _SKEW - timedelta(seconds=1)) is False

    @pytest.mark.asyncio
    async def test_artifact_within_the_drift_allowance_past_the_tolerance_is_refused(
        self, client: FakeNatsClient
    ) -> None:
        # the creation time is the broker's clock and the tolerance is the verifier's; the drift
        # between them is covered too, so an issue time at the bare tolerance is still refused.
        bucket = await client.kv_bucket(name="pop_nonces")
        created = datetime.now(UTC)
        bucket.wipe(date_created=created)
        guard = _guard(client)
        assert await guard.record_unique("n", issued_at=created + _SKEW) is False
        assert await guard.record_unique("m", issued_at=created + _REACH - timedelta(seconds=1)) is False

    @pytest.mark.asyncio
    async def test_artifact_issued_after_the_full_reach_is_fresh(self, client: FakeNatsClient) -> None:
        bucket = await client.kv_bucket(name="pop_nonces")
        created = datetime.now(UTC)
        bucket.wipe(date_created=created)
        guard = _guard(client)
        assert await guard.record_unique("n", issued_at=created + _REACH) is True

    @pytest.mark.asyncio
    async def test_a_refused_artifact_stays_refused(self, client: FakeNatsClient) -> None:
        # the refusal records the nonce, so presenting the same artifact later -- once the reach
        # has passed -- is a plain replay, not a second chance.
        bucket = await client.kv_bucket(name="pop_nonces")
        created = datetime.now(UTC)
        bucket.wipe(date_created=created)
        guard = _guard(client)
        stamped = created + timedelta(seconds=1)
        assert await guard.record_unique("n", issued_at=stamped) is False
        assert await guard.record_unique("n", issued_at=created + 2 * _REACH) is False


class _StubAnchor:
    """a `ReplayAnchor` whose recorded first-existence moment the test chooses.

    Deliberately not a fake of the coordination tables: these tests are about what the GUARD
    does with the anchor's answer, and a collection-backed double would make every one of them
    depend on the storage path too. `CollectionReplayAnchor`'s own behaviour is tested against
    the tables in `test_replay_anchor.py`.
    """

    def __init__(self, first_existed: datetime | None = None, *, fails: Exception | None = None) -> None:
        self._first_existed = first_existed
        self._fails = fails
        self.calls = 0

    async def first_existed(self, purpose: str, *, now: datetime) -> datetime:
        del purpose
        self.calls += 1
        if self._fails is not None:
            raise self._fails
        # `None` models the first caller: nothing was recorded, so this call's clock becomes the
        # ledger's birth time -- which is what the real claim-or-read returns.
        return self._first_existed if self._first_existed is not None else now


class TestReplayGuardWithAnAnchor:
    """the anchor tells a first run from a wipe, so only the wipe pays the refusal window."""

    @pytest.mark.asyncio
    async def test_a_first_run_admits_an_artifact_the_watermark_alone_would_refuse(
        self, client: FakeNatsClient
    ) -> None:
        # the case the anchor exists for: a bucket created moments ago, an artifact issued now,
        # and nothing that could have been recorded before either. Without an anchor this is
        # refused for the whole reach -- a fresh deployment refusing logins it has no reason to
        # doubt.
        bucket = await client.kv_bucket(name="pop_nonces")
        created = datetime.now(UTC)
        bucket.wipe(date_created=created)
        anchor = _StubAnchor()
        guard = ReplayGuard(
            client,  # type: ignore[arg-type]
            bucket_name="pop_nonces",
            ttl_seconds=120,
            verifier_future_tolerance=_SKEW,
            anchor=anchor,
        )
        assert await guard.record_unique("n", issued_at=created + timedelta(seconds=1)) is True

    @pytest.mark.asyncio
    async def test_a_wipe_still_refuses_inside_the_window(self, client: FakeNatsClient) -> None:
        # the anchor narrows the watermark to the case it was written for; it does not retire it.
        bucket = await client.kv_bucket(name="pop_nonces")
        created = datetime.now(UTC)
        bucket.wipe(date_created=created)
        anchor = _StubAnchor(created - timedelta(hours=3))
        guard = ReplayGuard(
            client,  # type: ignore[arg-type]
            bucket_name="pop_nonces",
            ttl_seconds=120,
            verifier_future_tolerance=_SKEW,
            anchor=anchor,
        )
        assert await guard.record_unique("n", issued_at=created + timedelta(seconds=1)) is False

    @pytest.mark.asyncio
    async def test_an_unreadable_anchor_keeps_the_watermark(self, client: FakeNatsClient) -> None:
        # the blind answer must be the conservative one: an anchor that cannot be read leaves the
        # guard exactly as unable to tell as having none, so it behaves as if it had none.
        bucket = await client.kv_bucket(name="pop_nonces")
        created = datetime.now(UTC)
        bucket.wipe(date_created=created)
        anchor = _StubAnchor(fails=KvError("anchor unreachable"))
        guard = ReplayGuard(
            client,  # type: ignore[arg-type]
            bucket_name="pop_nonces",
            ttl_seconds=120,
            verifier_future_tolerance=_SKEW,
            anchor=anchor,
        )
        assert await guard.record_unique("n", issued_at=created + timedelta(seconds=1)) is False

    @pytest.mark.asyncio
    async def test_a_replay_is_still_refused_on_a_first_run(self, client: FakeNatsClient) -> None:
        # skipping the watermark must not skip the guard: the second sighting of a nonce is a
        # replay whatever the anchor says.
        bucket = await client.kv_bucket(name="pop_nonces")
        created = datetime.now(UTC)
        bucket.wipe(date_created=created)
        guard = ReplayGuard(
            client,  # type: ignore[arg-type]
            bucket_name="pop_nonces",
            ttl_seconds=120,
            verifier_future_tolerance=_SKEW,
            anchor=_StubAnchor(),
        )
        issued_at = created + timedelta(seconds=1)
        assert await guard.record_unique("n", issued_at=issued_at) is True
        assert await guard.record_unique("n", issued_at=issued_at) is False

    @pytest.mark.asyncio
    async def test_the_anchor_is_read_once_not_per_artifact(self, client: FakeNatsClient) -> None:
        # the whole point of an anchor being affordable: it is a fact about the ledger's history,
        # which cannot change while the process runs, so it must not cost a durable round trip
        # per artifact.
        bucket = await client.kv_bucket(name="pop_nonces")
        created = datetime.now(UTC)
        bucket.wipe(date_created=created)
        anchor = _StubAnchor()
        guard = ReplayGuard(
            client,  # type: ignore[arg-type]
            bucket_name="pop_nonces",
            ttl_seconds=120,
            verifier_future_tolerance=_SKEW,
            anchor=anchor,
        )
        for nonce in ("a", "b", "c"):
            await guard.record_unique(nonce, issued_at=created + timedelta(seconds=1))
        assert anchor.calls == 1


class _BrokerClock:
    """one clock for the fake broker and the guard, under the test's control.

    The property under test is WHEN a bucket is created relative to the artifacts a service
    handles, and on a real clock a test cannot let minutes pass between a service starting and its
    first request. The fake stamps every bucket's creation time from it, and the guard passes it to
    its anchor as ``now``, so both hosts read the same moment -- NTP-synchronised, as the drift
    allowance assumes.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, start: datetime) -> None:
        self.moment = start
        clock = self

        class _Stamped(datetime):
            @classmethod
            def now(cls, tz: Any = None) -> datetime:  # type: ignore[override]
                del tz
                return clock.moment

        monkeypatch.setattr(fake_kv_module, "datetime", _Stamped)
        monkeypatch.setattr(replay_guard_module, "datetime", _Stamped)

    def advance(self, delta: timedelta) -> None:
        """let time pass on the broker.

        :param delta: how long
        :ptype delta: timedelta
        :return: None
        :rtype: None
        """
        self.moment += delta


def _anchored_after_a_wipe(client: FakeNatsClient, service_started: datetime) -> ReplayGuard:
    """a guard whose anchor says the ledger existed long before this bucket: a wipe, not a first run.

    :param client: the fake broker
    :ptype client: FakeNatsClient
    :param service_started: when the service came up; the ledger predates it by hours
    :ptype service_started: datetime
    :return: the guard
    :rtype: ReplayGuard
    """
    return ReplayGuard(
        client,  # type: ignore[arg-type]
        bucket_name="login_nonces",
        ttl_seconds=120,
        verifier_future_tolerance=_SKEW,
        anchor=_StubAnchor(service_started - timedelta(hours=3)),
    )


class TestBind:
    """a service binds its guard at start, so the bucket is never younger than what it serves."""

    @pytest.mark.asyncio
    async def test_bind_creates_the_bucket_before_any_record(self, client: FakeNatsClient) -> None:
        await _guard(client, bucket_name="pop_nonces").bind()
        # a bind-only open raises on an absent bucket, so this proves bind created it.
        bucket = await client.kv_bucket(name="pop_nonces", create_if_missing=False)
        assert bucket.keys() == ()

    @pytest.mark.asyncio
    async def test_bind_opens_the_bucket_once_however_often_it_is_called(self) -> None:
        bucket = AsyncMock()
        spy_client = AsyncMock()
        spy_client.kv_bucket = AsyncMock(return_value=bucket)
        guard = _guard(spy_client, bucket_name="b", ttl_seconds=90)
        await guard.bind()
        await guard.bind()
        spy_client.kv_bucket.assert_awaited_once()
        kwargs = spy_client.kv_bucket.call_args.kwargs
        assert kwargs["name"] == "b"
        assert kwargs["ttl"] == timedelta(seconds=90)
        assert kwargs["create_if_missing"] is True

    @pytest.mark.asyncio
    async def test_concurrent_binds_open_the_bucket_once(self) -> None:
        # startup wiring and a first request can race the bind; every caller must share ONE open.
        # The open suspends until every bind has started, which is the interleaving a lock-free
        # check-then-open would lose.
        bucket = AsyncMock()
        gate = asyncio.Event()

        async def _open(**_: object) -> object:
            await gate.wait()
            return bucket

        spy_client = AsyncMock()
        spy_client.kv_bucket = AsyncMock(side_effect=_open)
        guard = _guard(spy_client, bucket_name="b")
        binds = [asyncio.create_task(guard.bind()) for _ in range(8)]
        await asyncio.sleep(0)
        gate.set()
        handles = await asyncio.gather(*binds)
        spy_client.kv_bucket.assert_awaited_once()
        assert all(handle is bucket for handle in handles)

    @pytest.mark.asyncio
    async def test_record_unique_binds_an_unbound_guard(self) -> None:
        bucket = AsyncMock()
        bucket.create = AsyncMock(return_value=1)
        bucket.date_created = AsyncMock(return_value=datetime.now(UTC))
        spy_client = AsyncMock()
        spy_client.kv_bucket = AsyncMock(return_value=bucket)
        guard = _guard(spy_client, bucket_name="b")
        assert await guard.record_unique("x", issued_at=_later()) is True
        bucket.create.assert_awaited_once()
        # the record bound it, so a later bind is the same handle and no second open.
        assert await guard.bind() is bucket
        spy_client.kv_bucket.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_after_a_wipe_an_artifact_issued_after_a_bind_at_start_is_accepted(
        self, client: FakeNatsClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # the broker restarted and lost the bucket; the service came up and bound at once. Its
        # first login arrives ten minutes later, and the artifact was issued after the service
        # was up, so nothing about it can predate the wipe.
        started = datetime.now(UTC)
        broker = _BrokerClock(monkeypatch, started)
        guard = _anchored_after_a_wipe(client, started)
        await guard.bind()
        broker.advance(timedelta(minutes=10))
        assert await guard.record_unique("login", issued_at=started + timedelta(minutes=10)) is True

    @pytest.mark.asyncio
    async def test_after_a_wipe_an_artifact_issued_before_the_bind_is_still_refused(
        self, client: FakeNatsClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # binding early narrows the watermark; it does not retire it. An artifact issued before
        # the service started could have been accepted and recorded before the wipe.
        started = datetime.now(UTC)
        broker = _BrokerClock(monkeypatch, started)
        guard = _anchored_after_a_wipe(client, started)
        await guard.bind()
        broker.advance(timedelta(minutes=10))
        assert await guard.record_unique("stale", issued_at=started - timedelta(seconds=1)) is False

    @pytest.mark.asyncio
    async def test_after_a_wipe_an_unbound_guard_refuses_the_artifact_that_first_uses_it(
        self, client: FakeNatsClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # the defect bind exists to avoid, pinned so the cost stays visible: left to the first
        # record, the bucket is created at first USE, and the very artifact that triggered it --
        # issued long after the service came up -- lands inside the watermark.
        started = datetime.now(UTC)
        broker = _BrokerClock(monkeypatch, started)
        guard = _anchored_after_a_wipe(client, started)
        broker.advance(timedelta(minutes=10))
        issued_at = started + timedelta(minutes=10) - timedelta(seconds=1)
        assert await guard.record_unique("login", issued_at=issued_at) is False


class _ClaimingAnchor:
    """a `ReplayAnchor` shared by every replica: the first caller's ``now`` is recorded and returned to all.

    The claim-or-read ``CollectionReplayAnchor`` performs against its table, held in memory, so two
    guards standing in for two replicas see one ledger history.
    """

    def __init__(self, *, fails: Exception | None = None) -> None:
        self._claimed: datetime | None = None
        self._fails = fails
        self.calls = 0

    async def first_existed(self, purpose: str, *, now: datetime) -> datetime:
        del purpose
        self.calls += 1
        if self._fails is not None:
            raise self._fails
        if self._claimed is None:
            self._claimed = now
        return self._claimed


def _anchored(client: FakeNatsClient, anchor: object) -> ReplayGuard:
    """a guard over the shared ``login_nonces`` bucket with the given anchor.

    :param client: the fake broker
    :ptype client: FakeNatsClient
    :param anchor: the anchor double
    :ptype anchor: object
    :return: the guard
    :rtype: ReplayGuard
    """
    return ReplayGuard(
        client,  # type: ignore[arg-type]
        bucket_name="login_nonces",
        ttl_seconds=120,
        verifier_future_tolerance=_SKEW,
        anchor=anchor,  # type: ignore[arg-type]
    )


class TestBindStampsTheAnchor:
    """a guard bound at start has its ledger's first-existence moment before it records any nonce."""

    @pytest.mark.asyncio
    async def test_a_replay_of_the_first_nonce_is_refused_when_a_wipe_races_its_record(
        self, client: FakeNatsClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # the window this closes: the very first nonce is recorded, then the broker loses the
        # bucket before anything has stamped the anchor. Stamped only by that first record, the
        # anchor would take a moment AFTER the wipe, read as a first run, and admit the nonce's
        # replay. Stamped at bind, it predates the wipe, and the watermark applies.
        started = datetime.now(UTC)
        clock = _BrokerClock(monkeypatch, started)
        anchor = _ClaimingAnchor()
        replica_a = _anchored(client, anchor)
        replica_b = _anchored(client, anchor)
        await replica_a.bind()
        await replica_b.bind()

        clock.advance(timedelta(minutes=10))
        bucket = await client.kv_bucket(name="login_nonces")
        original_create = bucket.create

        async def _create_then_lose_the_bucket(**kwargs: Any) -> int | None:
            revision = await original_create(**kwargs)
            monkeypatch.setattr(bucket, "create", original_create)
            bucket.wipe(date_created=clock.moment)  # the broker restarted; another replica recreated it
            return revision

        monkeypatch.setattr(bucket, "create", _create_then_lose_the_bucket)
        first_issued = clock.moment
        # the wipe raced the nonce's own record, so the guard fails closed on it too.
        assert await replica_a.record_unique("first-nonce", issued_at=first_issued) is False
        assert await replica_b.record_unique("first-nonce", issued_at=first_issued) is False

    @pytest.mark.asyncio
    async def test_bind_reads_the_anchor_before_any_record_and_only_once(self, client: FakeNatsClient) -> None:
        anchor = _ClaimingAnchor()
        guard = _anchored(client, anchor)
        await guard.bind()
        assert anchor.calls == 1
        await guard.bind()
        await guard.record_unique("a", issued_at=_later())
        assert anchor.calls == 1

    @pytest.mark.asyncio
    async def test_an_anchor_failure_does_not_fail_the_bind(
        self, client: FakeNatsClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        # an unreachable anchor must not stop a service starting: the guard is exactly as blind
        # as one with no anchor, and keeps the conservative watermark.
        anchor = _ClaimingAnchor(fails=KvError("anchor unreachable"))
        guard = _anchored(client, anchor)
        with caplog.at_level("WARNING", logger="threetears.core.coordination.replay_guard"):
            handle = await guard.bind()
        assert handle is await client.kv_bucket(name="login_nonces")
        assert any("anchor unreadable" in r.getMessage() for r in caplog.records)
        # blind means conservative: an artifact inside the watermark is refused...
        assert await guard.record_unique("n", issued_at=datetime.now(UTC)) is False
        # ...and the read is tried again rather than remembered as a failure.
        assert anchor.calls == 2


class _RecordingHooks:
    """wraps a fake client's ``add_reconnect_callback`` to keep every hook registered."""

    def __init__(self, client: FakeNatsClient, monkeypatch: pytest.MonkeyPatch) -> None:
        self.hooks: list[Any] = []
        register = client.add_reconnect_callback

        def _register(callback: Any) -> None:
            self.hooks.append(callback)
            register(callback)

        monkeypatch.setattr(client, "add_reconnect_callback", _register)


class TestRebindOnReconnect:
    """a bucket lost under a running process is recreated when NATS comes back, not at the next artifact."""

    @pytest.mark.asyncio
    async def test_a_bucket_lost_under_a_running_process_is_recreated_at_reconnect(
        self, client: FakeNatsClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # the service has run for an hour when the broker restarts. The reconnect recreates the
        # bucket then, so a login ten minutes later -- issued well after the restart -- is admitted.
        # Left to the login itself, the bucket would be created by it, and it would be refused.
        started = datetime.now(UTC)
        clock = _BrokerClock(monkeypatch, started)
        guard = _anchored_after_a_wipe(client, started)
        await guard.bind()
        bucket = await client.kv_bucket(name="login_nonces")

        clock.advance(timedelta(hours=1))
        bucket.vanish()
        await client.reconnect()

        clock.advance(timedelta(minutes=10))
        assert await guard.record_unique("login", issued_at=clock.moment - timedelta(seconds=1)) is True

    @pytest.mark.asyncio
    async def test_without_a_reconnect_the_next_artifact_recreates_the_bucket_and_is_refused(
        self, client: FakeNatsClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # the same sequence with no reconnect signal: pinned so the cost the hook removes stays visible.
        started = datetime.now(UTC)
        clock = _BrokerClock(monkeypatch, started)
        guard = _anchored_after_a_wipe(client, started)
        await guard.bind()
        bucket = await client.kv_bucket(name="login_nonces")

        clock.advance(timedelta(hours=1))
        bucket.vanish()

        clock.advance(timedelta(minutes=10))
        assert await guard.record_unique("login", issued_at=clock.moment - timedelta(seconds=1)) is False

    @pytest.mark.asyncio
    async def test_bind_registers_one_hook_however_often_it_runs(
        self, client: FakeNatsClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded = _RecordingHooks(client, monkeypatch)
        guard = _guard(client, bucket_name="pop_nonces")
        await asyncio.gather(*[guard.bind() for _ in range(4)])
        await guard.bind()
        await guard.record_unique("x", issued_at=_later())
        assert len(recorded.hooks) == 1

    @pytest.mark.asyncio
    async def test_an_unbound_guard_registers_no_hook(
        self, client: FakeNatsClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded = _RecordingHooks(client, monkeypatch)
        _guard(client, bucket_name="pop_nonces")
        await client.reconnect()
        assert recorded.hooks == []
        with pytest.raises(KeyError):
            await client.kv_bucket(name="pop_nonces", create_if_missing=False)

    @pytest.mark.asyncio
    async def test_a_hook_that_cannot_reach_the_bucket_logs_and_does_not_raise(
        self, client: FakeNatsClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # the hook runs inside the client's reconnect path; it must not be the thing that fails it.
        recorded = _RecordingHooks(client, monkeypatch)
        guard = _guard(client, bucket_name="pop_nonces")
        await guard.bind()
        bucket = await client.kv_bucket(name="pop_nonces")
        monkeypatch.setattr(bucket, "date_created", AsyncMock(side_effect=KvError("broker still down")))

        with caplog.at_level("WARNING", logger="threetears.core.coordination.replay_guard"):
            await recorded.hooks[0]()

        assert any("pop_nonces" in r.getMessage() and "reconnect" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_a_client_without_reconnect_hooks_still_binds(self) -> None:
        # a KvCapable that is not a NatsClient -- the published double before it grew the hook, a
        # consumer's own -- binds exactly as before; only the availability improvement is absent.
        bucket = AsyncMock()
        spy_client = AsyncMock()
        spy_client.kv_bucket = AsyncMock(return_value=bucket)
        guard = _guard(spy_client, bucket_name="b")
        assert await guard.bind() is bucket
        spy_client.add_reconnect_callback.assert_not_called()
