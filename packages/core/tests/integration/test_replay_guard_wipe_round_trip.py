"""Integration test: a ReplayGuard refuses a replay across a wipe of its memory-backed bucket.

Against a real broker, because the property under test is one no double can witness: after one
connection's self-heal recreates a wiped stream, every OTHER connection's bucket handle keeps
working against the new, empty stream without raising. The nonce that would have refused the
replay is gone, and the guard never saw an error. Only the stream's server-side creation time,
read after the write, still tells it the truth.

Two ``NatsClient.connect()`` connections stand in for two replicas. Uses the session-scoped
``nats_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from threetears.core.coordination import ReplayGuard
from threetears.core.coordination.replay_guard import CLOCK_DRIFT_ALLOWANCE
from threetears.nats import NatsClient, set_default_namespace

pytestmark = pytest.mark.integration

_NAMESPACE = "replaywipe"
_BUCKET = "wipe_nonces"
# a zero-leeway verifier, like the tool pod's; the guard's refusal reach is then the drift allowance.
_TOLERANCE = timedelta(0)
_REACH = _TOLERANCE + CLOCK_DRIFT_ALLOWANCE


async def test_date_created_is_server_time_and_moves_forward_on_recreate(nats_container: str) -> None:
    set_default_namespace(_NAMESPACE)
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=_NAMESPACE, client_name="created-probe"
    ) as nc:
        bucket = await nc.kv_bucket(name="created_probe", ttl=timedelta(seconds=60))
        first = await bucket.date_created()
        assert first.tzinfo is not None
        await bucket.put(key="k", value=b"v")
        assert await bucket.date_created() == first  # a write does not move it

        await nc.jetstream_context().delete_stream(f"KV_{bucket.name}")
        # the read self-heals like every other operation, and describes the recreated stream.
        second = await bucket.date_created()
        assert second > first


async def test_a_replay_through_a_handle_that_never_saw_the_wipe_is_refused(nats_container: str) -> None:
    set_default_namespace(_NAMESPACE)
    async with (
        await NatsClient.connect(
            nats_url=nats_container, nats_subject_namespace=_NAMESPACE, client_name="replica-a"
        ) as a,
        await NatsClient.connect(
            nats_url=nats_container, nats_subject_namespace=_NAMESPACE, client_name="replica-b"
        ) as b,
    ):
        guard_a = ReplayGuard(a, bucket_name=_BUCKET, ttl_seconds=120, verifier_future_tolerance=_TOLERANCE)
        guard_b = ReplayGuard(b, bucket_name=_BUCKET, ttl_seconds=120, verifier_future_tolerance=_TOLERANCE)

        # replica A admits a proof on a bucket that already existed well before it was issued. the
        # proof is stamped "now": an issue time leading the clock by more than the verifier's
        # tolerance breaks the guard's contract, and would -- correctly -- be admitted after a wipe.
        bucket_a = await a.kv_bucket(name=_BUCKET, ttl=timedelta(seconds=120))
        await asyncio.sleep((_REACH + timedelta(milliseconds=500)).total_seconds())
        issued_at = datetime.now(UTC)
        assert issued_at >= (await bucket_a.date_created()) + _REACH
        assert await guard_a.record_unique("proof-1", issued_at=issued_at) is True
        assert await guard_a.record_unique("proof-1", issued_at=issued_at) is False  # plain replay

        # the broker loses the stream. replica B heals it first, recording something else.
        await a.jetstream_context().delete_stream(f"KV_{bucket_a.name}")
        bucket_b = await b.kv_bucket(name=_BUCKET, ttl=timedelta(seconds=120))
        recreated = datetime.now(UTC) + timedelta(seconds=10)
        assert await guard_b.record_unique("other", issued_at=recreated) is True

        # replica A's handle never raised: the nonce write lands in the new empty stream. the
        # replay is refused anyway, because the proof predates that stream.
        assert await guard_a.record_unique("proof-1", issued_at=issued_at) is False

        # a proof issued after the new stream exists, beyond the reach, is admitted on either replica.
        after = (await bucket_b.date_created()) + _REACH + timedelta(seconds=1)
        assert await guard_a.record_unique("proof-2", issued_at=after) is True
