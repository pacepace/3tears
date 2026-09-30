"""Integration test: a bind-only primitive recovers, without a restart, once its declarer re-declares.

A NATS restart wipes every memory-backed bucket. The hub re-declares all of them when its connection
comes back, but a pod can reach the bus first -- and a pod never creates a bucket, so a bind-only
primitive that failed on that first miss stayed broken until the pod restarted. Here the "hub" is a
second connection that declares through ``ensure_kv_bucket`` exactly as the hub's pod-bucket
declarer does, and the "pod" holds a guard and a lease that only bind.

Against a real broker, because the property is about what the server answers for a bucket that is
not there, and when. Uses the session-scoped ``nats_container`` fixture; a checkout without docker
skips cleanly.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from unittest.mock import patch
from uuid import uuid4

import pytest

from threetears.core.coordination import ReplayGuard
from threetears.core.coordination.lease import KVLease
from threetears.nats import KvError, NatsClient, set_default_namespace

pytestmark = pytest.mark.integration

_NAMESPACE = "rebindlive"
_SCOPE = "tool_pod-01947100000070008000000000000001"
#: how long after the wipe the "hub" re-declares -- longer than one bind attempt, well inside the wait.
_REDECLARE_AFTER = timedelta(seconds=1.5)


async def _declare(hub: NatsClient, bucket: str) -> None:
    """declare a pod bucket the way the hub's pod-bucket declarer does.

    :param hub: the declaring connection
    :ptype hub: NatsClient
    :param bucket: the bucket suffix
    :ptype bucket: str
    :return: nothing
    :rtype: None
    """
    await hub.ensure_kv_bucket(name=bucket, ttl=None, storage="memory", history=1, direct=True)


async def _redeclare_later(hub: NatsClient, bucket: str) -> None:
    """re-declare ``bucket`` after :data:`_REDECLARE_AFTER`, as the hub's reconnect hook does.

    :param hub: the declaring connection
    :ptype hub: NatsClient
    :param bucket: the bucket suffix
    :ptype bucket: str
    :return: nothing
    :rtype: None
    """
    await asyncio.sleep(_REDECLARE_AFTER.total_seconds())
    await _declare(hub, bucket)


async def test_a_bind_only_guard_records_again_once_the_hub_redeclares_its_wiped_bucket(nats_container: str) -> None:
    set_default_namespace(_NAMESPACE)
    bucket = f"nonces_{uuid4().hex}"
    async with (
        await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=_NAMESPACE, client_name="hub") as hub,
        await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=_NAMESPACE, client_name="pod") as pod,
    ):
        await _declare(hub, bucket)
        guard = ReplayGuard(
            pod,
            bucket_name=bucket,
            ttl_seconds=120,
            verifier_future_tolerance=timedelta(0),
            create_if_missing=False,
            key_scope=_SCOPE,
        )
        await guard.bind()
        later = datetime.now(UTC) + timedelta(minutes=5)
        assert await guard.record_unique("before-the-wipe", issued_at=later) is True

        # the broker restarts: the memory-backed bucket is gone, and the hub re-declares it only
        # once its own connection is back.
        await hub.jetstream_context().delete_stream(f"KV_{_NAMESPACE}-{bucket}")
        redeclare = asyncio.create_task(_redeclare_later(hub, bucket))

        # the pod's guard was bound before the wipe and is never rebuilt: its next record re-binds,
        # waits for the declarer, and records -- no restart, no manual step.
        assert await guard.record_unique("after-the-wipe", issued_at=later) is True
        await redeclare

        # recorded under the pod's own scope, in the bucket the hub declared.
        declared = await hub.kv_bucket(name=bucket, create_if_missing=False)
        digest = hashlib.sha256(b"after-the-wipe").hexdigest()
        assert await declared.get(key=f"{_SCOPE}.{digest}") is not None
        # and it is still a ledger: the same nonce is a replay.
        assert await guard.record_unique("after-the-wipe", issued_at=later) is False


async def test_a_pod_that_starts_before_the_hub_declares_binds_once_it_does(nats_container: str) -> None:
    set_default_namespace(_NAMESPACE)
    bucket = f"leases_{uuid4().hex}"
    async with (
        await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=_NAMESPACE, client_name="hub") as hub,
        await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=_NAMESPACE, client_name="pod") as pod,
    ):
        lease = KVLease(pod, bucket_name=bucket, pod_id="replica-1", create_if_missing=False, key_scope=_SCOPE)
        redeclare = asyncio.create_task(_redeclare_later(hub, bucket))
        handle = await lease.acquire("session", ttl_seconds=30, max_wait_seconds=0)
        await redeclare
        assert handle.key == f"{_SCOPE}.session"
        await handle.release()


async def test_a_bucket_nobody_declares_fails_once_the_wait_is_spent(nats_container: str) -> None:
    set_default_namespace(_NAMESPACE)
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=_NAMESPACE, client_name="pod"
    ) as pod:
        guard = ReplayGuard(
            pod,
            bucket_name=f"never_{uuid4().hex}",
            ttl_seconds=120,
            verifier_future_tolerance=timedelta(0),
            create_if_missing=False,
        )
        with (
            patch("threetears.nats.kv._BIND_WAIT_FOR_DECLARER_SECONDS", 1.0),
            pytest.raises(KvError, match="declar"),
        ):
            await guard.bind()
