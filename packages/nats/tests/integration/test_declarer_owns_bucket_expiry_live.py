"""a bucket's one declarer removes a stale bucket-wide expiry, against a real broker.

Found live on cobalt-dev: the shared rate-limit bucket was created long ago with a bucket-wide
``max_age`` of 300s. Its declarer (the hub) now declares every pod bucket with no bucket-wide
expiry, so each bind-only opener can give its own entries their own lifetime -- but ``max_age``
sat outside the reconciled set, so the stale 300s was logged as dropped and never removed, and
every opener asking for a 60s per-entry TTL was refused with ``KvConfigMismatch``. A service's rate
limiting then failed on every request.

Only a live server can answer the parts that matter: that JetStream accepts an in-place change of
``max_age`` on a live KV stream, that the bucket still binds as a bucket afterwards, and that an
opener's entries then really carry the per-entry lifetime in their ``Nats-TTL`` header.

Uses the session-scoped ``nats_container`` fixture.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from nats.js.api import Header, StorageType

from threetears.nats import NatsClient, set_default_namespace
from threetears.nats.errors import KvConfigMismatch
from threetears.nats.kv import build_kv_stream_config

pytestmark = pytest.mark.integration

#: the stale bucket-wide expiry the bucket was created with, as the old declarers created it
_STALE_MAX_AGE_SECONDS = 300

#: the per-entry lifetime a bind-only opener asks for, as the rate limiter does
_OPENER_TTL = timedelta(seconds=60)


async def _create_with_a_stale_expiry(nc: NatsClient, *, full_name: str) -> None:
    """create the bucket the way the old declarers did: a 300s bucket-wide ``max_age``.

    :param nc: connected client
    :ptype nc: NatsClient
    :param full_name: fully-qualified bucket name
    :ptype full_name: str
    :return: nothing
    :rtype: None
    """
    await nc.jetstream_context().add_stream(
        build_kv_stream_config(
            bucket=full_name,
            ttl_seconds=_STALE_MAX_AGE_SECONDS,
            history=1,
            storage_type=StorageType.MEMORY,
            direct=True,
        )
    )


async def test_a_declarer_owning_expiry_removes_the_stale_expiry_and_openers_bind(nats_container: str) -> None:
    """the fix, end to end: the declaration reconciles ``max_age`` away and the opener binds.

    (a) declaring with ``ttl=None, owns_expiry=True`` leaves the live stream with no bucket-wide
    expiry and per-entry TTLs allowed; (b) a bind-only opener asking for a 60s lifetime then binds,
    and every entry it writes carries that lifetime.
    """
    namespace = "ownsexpiry"
    full_name = f"{namespace}-ratelimits"
    set_default_namespace(namespace)
    async with (
        await NatsClient.connect(
            nats_url=nats_container, nats_subject_namespace=namespace, client_name="declarer"
        ) as declarer,
        await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="pod") as pod,
    ):
        js = declarer.jetstream_context()
        await _create_with_a_stale_expiry(declarer, full_name=full_name)
        assert (await js.stream_info(f"KV_{full_name}")).config.max_age == _STALE_MAX_AGE_SECONDS

        await declarer.ensure_kv_bucket(name="ratelimits", ttl=None, storage="memory", history=1, owns_expiry=True)

        live = (await js.stream_info(f"KV_{full_name}")).config
        assert not live.max_age, f"the stale bucket-wide expiry survived the declaration: max_age={live.max_age}"
        assert live.allow_msg_ttl is True

        opened = await pod.kv_bucket(name="ratelimits", ttl=_OPENER_TTL, create_if_missing=False)
        await opened.put(key="caller.window", value=b"1")

        stored = await js.get_last_msg(f"KV_{full_name}", f"$KV.{full_name}.caller.window")
        assert stored.headers is not None, "the entry carries no headers, so no per-entry lifetime"
        assert stored.headers.get(Header.MSG_TTL) == str(int(_OPENER_TTL.total_seconds()))
        assert await opened.get(key="caller.window") == b"1"


async def test_a_declarer_not_owning_expiry_leaves_it_and_openers_are_refused(nats_container: str) -> None:
    """(c) the negative control: today's behaviour, unchanged by default.

    Without ownership the declaration reports the stale ``max_age`` and leaves it, and a bind-only
    opener asking for a 60s lifetime is refused, because its entries would expire at 300s.
    """
    namespace = "keepsexpiry"
    full_name = f"{namespace}-ratelimits"
    set_default_namespace(namespace)
    async with (
        await NatsClient.connect(
            nats_url=nats_container, nats_subject_namespace=namespace, client_name="declarer"
        ) as declarer,
        await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="pod") as pod,
    ):
        js = declarer.jetstream_context()
        await _create_with_a_stale_expiry(declarer, full_name=full_name)

        await declarer.ensure_kv_bucket(name="ratelimits", ttl=None, storage="memory", history=1)

        assert (await js.stream_info(f"KV_{full_name}")).config.max_age == _STALE_MAX_AGE_SECONDS
        with pytest.raises(KvConfigMismatch, match="expires entries after 300s"):
            await pod.kv_bucket(name="ratelimits", ttl=_OPENER_TTL, create_if_missing=False)


async def test_a_declarer_owning_expiry_can_set_one_shorter_than_the_duplicate_window(nats_container: str) -> None:
    """JetStream refuses a duplicate window longer than ``max_age``; the reconcile brings both.

    A bucket created with no expiry carries the two-minute duplicate window, so moving it to a
    60s expiry in place is accepted only when the window comes down with it.
    """
    namespace = "shortexpiry"
    full_name = f"{namespace}-sessions"
    set_default_namespace(namespace)
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="declarer"
    ) as declarer:
        js = declarer.jetstream_context()
        await declarer.ensure_kv_bucket(name="sessions", ttl=None, history=1)
        assert not (await js.stream_info(f"KV_{full_name}")).config.max_age

        bucket = await declarer.ensure_kv_bucket(name="sessions", ttl=_OPENER_TTL, history=1, owns_expiry=True)

        live = (await js.stream_info(f"KV_{full_name}")).config
        assert live.max_age == _OPENER_TTL.total_seconds()
        assert live.duplicate_window == _OPENER_TTL.total_seconds()
        await bucket.put(key="k", value=b"v")
        assert await bucket.get(key="k") == b"v"
