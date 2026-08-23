"""Integration tests for the KV create-or-reconcile primitive, against a real broker.

Three things only a live NATS can answer, and each of them is load-bearing:

1. **The stream shape.** ``build_kv_stream_config`` mirrors nats-py's
   ``create_key_value``. A hand-kept field list goes stale silently and the
   symptom is not a config difference -- it is a stream that stops being a
   bucket, with ``key_value()`` raising ``BadBucketError`` on
   ``max_msgs_per_subject < 1``. So the two are BUILT side by side here and their
   server-side configs compared.
2. **The in-place flip.** ``allow_direct`` on a live bucket, with ``key_value()``
   still binding afterwards and the resulting handle actually using the direct
   read path.
3. **The refusal.** A reader that states a config the live bucket does not carry
   gets ``KvConfigMismatch`` rather than a silent bind.

Uses the session-scoped ``nats_container`` fixture; a checkout without docker
skips cleanly.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from nats.js.api import KeyValueConfig, StorageType

from threetears.nats import NatsClient, set_default_namespace
from threetears.nats.errors import KvConfigMismatch
from threetears.nats.kv import build_kv_stream_config

pytestmark = pytest.mark.integration

#: Stream-config fields that legitimately differ between the two construction
#: paths, so comparing them would fail for reasons that are not drift.
#:
#: ``name`` and ``subjects`` name the bucket. ``allow_direct`` is the whole point
#: of the primitive and is deliberately set on one side only. ``metadata`` and
#: ``created`` are server bookkeeping.
_NOT_COMPARABLE = frozenset({"name", "subjects", "allow_direct", "metadata", "created"})


def _server_shape(config: Any) -> dict[str, Any]:
    """every field the server reports for a stream, minus the incomparable ones.

    :param config: a ``StreamConfig`` as returned by ``stream_info``
    :ptype config: Any
    :return: field name -> value
    :rtype: dict[str, Any]
    """
    return {
        name: getattr(config, name)
        for name in sorted(vars(config))
        if not name.startswith("_") and name not in _NOT_COMPARABLE
    }


async def test_kv_stream_shape_matches_nats_py(nats_container: str) -> None:
    """Our builder and ``create_key_value`` must produce the same bucket.

    This is the anti-staleness guard. If nats-py adds a field to the KV shape and
    our mirror does not follow, this fails here rather than in a deployment where
    the symptom is a stream that has quietly stopped behaving like a bucket.
    """
    set_default_namespace("shapetest")
    async with await NatsClient.connect(
        nats_url=nats_container,
        nats_subject_namespace="shapetest",
        client_name="kv-shape",
    ) as nc:
        js = nc.jetstream_context()

        await js.create_key_value(KeyValueConfig(bucket="theirs", ttl=60, history=3, storage=StorageType.MEMORY))
        await js.add_stream(
            build_kv_stream_config(
                bucket="ours",
                ttl_seconds=60,
                history=3,
                storage_type=StorageType.MEMORY,
                direct=True,
            )
        )

        theirs = _server_shape((await js.stream_info("KV_theirs")).config)
        ours = _server_shape((await js.stream_info("KV_ours")).config)
        assert ours == theirs, "the KV stream shape drifted from nats-py's; a mirrored field is missing or wrong"

        # And it is genuinely a bucket: nats-py's own binder validates the shape.
        kv = await js.key_value("ours")
        await kv.put("k", b"v")
        assert (await kv.get("k")).value == b"v"


async def test_declaring_flips_allow_direct_on_a_live_bucket(nats_container: str) -> None:
    """The reconcile, end to end: an existing bucket ends the call with direct on.

    Also the KVC-06 mechanism. Delete-and-recreate was the other option; the
    in-place update is what a live cluster with 32k values in the bucket needs.
    """
    set_default_namespace("fliptest")
    async with await NatsClient.connect(
        nats_url=nats_container,
        nats_subject_namespace="fliptest",
        client_name="kv-flip",
    ) as nc:
        js = nc.jetstream_context()
        # Create it the way the platform used to: nats-py's own KV constructor,
        # which never sends allow_direct at all.
        await js.create_key_value(KeyValueConfig(bucket="fliptest-coll", storage=StorageType.MEMORY))
        kv = await js.key_value("fliptest-coll")
        await kv.put("before", b"1")
        assert (await js.stream_info("KV_fliptest-coll")).config.allow_direct is False

        bucket = await nc.ensure_kv_bucket(name="coll", direct=True)

        assert (await js.stream_info("KV_fliptest-coll")).config.allow_direct is True
        # js.key_value() still binds after the reconcile, and the data survived.
        rebound = await js.key_value("fliptest-coll")
        assert (await rebound.get("before")).value == b"1"
        # And the handle the wrapper hands back reads over the DIRECT path, which
        # is the only form a key-scoped $KV grant can constrain.
        assert rebound._direct is True  # noqa: SLF001 - nats-py's own read-path switch
        assert await bucket.get(key="before") == b"1"


async def test_ensure_kv_bucket_shares_the_client_bucket_cache(nats_container: str) -> None:
    """A declaration that did not share the cache would be invisible to consumers.

    ``kv_bucket`` caches by full name, so the first opener's handle wins for the
    life of the process. If the declaration wrote to a different place, hub
    bootstrap and hub collections would run against handles that disagree about
    ``direct`` -- the declaration would look done and change nothing that reads.
    """
    set_default_namespace("cachetest")
    async with await NatsClient.connect(
        nats_url=nats_container,
        nats_subject_namespace="cachetest",
        client_name="kv-cache",
    ) as nc:
        declared = await nc.ensure_kv_bucket(name="coll", direct=True)
        consumer = await nc.kv_bucket(name="coll")
        assert consumer is declared


async def test_a_reader_refuses_a_bucket_that_lost_its_direct(nats_container: str) -> None:
    """The failure mode the mismatch type exists for, live.

    A NATS restart wipes a memory bucket; whichever process opens first recreates
    it, and one that does not name ``direct`` recreates it without. A reader that
    requires ``direct`` must refuse rather than bind and silently put every read
    back on the body-carried form.
    """
    set_default_namespace("readtest")
    async with await NatsClient.connect(
        nats_url=nats_container,
        nats_subject_namespace="readtest",
        client_name="kv-read",
    ) as nc:
        js = nc.jetstream_context()
        await js.create_key_value(KeyValueConfig(bucket="readtest-coll", storage=StorageType.MEMORY))

        with pytest.raises(KvConfigMismatch):
            await nc.ensure_kv_bucket(name="coll", direct=True, create_if_missing=False)


async def test_deny_purge_is_not_expressible_on_a_kv_stream(nats_container: str) -> None:
    """Why the KV stream shape carries no ``deny_purge``, recorded as a live check.

    ``deny_purge: true`` was specified for the collections stream (KVC-08). The
    server refuses it: ``allow_rollup_hdrs`` and ``deny_purge`` are mutually
    exclusive -- ``roll-ups require the purge permission``, err_code 10052 -- and
    ``allow_rollup_hdrs`` is part of what MAKES a stream a KV bucket. Dropping it
    to get ``deny_purge`` produces a stream that accepts writes and then fails
    ``KeyValue.purge`` with ``rollup not permitted``.

    So the requirement is unreachable through stream config, and protecting the
    shared bucket from ``$JS.API.STREAM.PURGE`` belongs to the GRANT instead
    (``coll-task-05a``, which narrows ``$JS.API.STREAM.*``). This test exists so
    that verdict is re-checked against every nats-server the suite runs on rather
    than believed on the strength of one reading.
    """
    set_default_namespace("purgetest")
    async with await NatsClient.connect(
        nats_url=nats_container,
        nats_subject_namespace="purgetest",
        client_name="kv-purge",
    ) as nc:
        js = nc.jetstream_context()
        config = build_kv_stream_config(
            bucket="purgetest-coll",
            ttl_seconds=0,
            history=1,
            storage_type=StorageType.MEMORY,
            direct=True,
        )
        config.deny_purge = True
        with pytest.raises(Exception) as caught:  # noqa: PT011 - the type is nats-py's, the code is the assertion
            await js.add_stream(config)
        assert getattr(caught.value, "err_code", None) == 10052, (
            "nats-server no longer refuses deny_purge alongside allow_rollup_hdrs; "
            "KVC-08 may now be reachable and should be revisited"
        )


async def test_the_self_heal_recreates_with_direct(nats_container: str) -> None:
    """`_reopen` after a JetStream wipe must not silently drop ``allow_direct``.

    The cached handle survives the wipe and heals itself on the next operation.
    Before this landing that heal ran ``create_key_value``, which never sends the
    field -- so the bucket came back readable by anyone the bucket grant admits.
    """
    set_default_namespace("healdirect")
    async with await NatsClient.connect(
        nats_url=nats_container,
        nats_subject_namespace="healdirect",
        client_name="kv-heal-direct",
    ) as nc:
        bucket = await nc.ensure_kv_bucket(name="coll", ttl=timedelta(seconds=60), direct=True)
        await bucket.put(key="k", value=b"v1")

        js = nc.jetstream_context()
        await js.delete_stream("KV_healdirect-coll")

        # The next op re-opens and retries against the recreated bucket.
        assert await bucket.create(key="k2", value=b"v2") is not None
        assert (await js.stream_info("KV_healdirect-coll")).config.allow_direct is True
