"""a bucket's one declarer reconciles its whole shape, against a real broker.

Found live on cobalt-dev, in the hub's boot log: ``aibots-ratelimits`` was live with a bucket-wide
``max_age`` of 300s, and ``aibots-proxy_assertion_nonces`` on FILE storage with a ``max_age`` of 60s.
The hub declares every pod bucket on memory with no bucket-wide expiry, so each bind-only opener can
give its own entries their own lifetime -- but neither field was reconciled, so both were logged as
dropped and left, and every opener asking for a 60s per-entry TTL on the rate-limit bucket was
refused with ``KvConfigMismatch``. A service's rate limiting then failed on every request.

Only a live server can answer the parts that matter: that JetStream accepts an in-place change of
``max_age`` and history on a live KV stream, that a stream's storage can only be changed by deleting
and recreating it, that the bucket still binds as a bucket afterwards, that an opener's entries then
carry and honour their own lifetime, and that two owners recreating at once both succeed.

Uses the session-scoped ``nats_container`` fixture.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from nats.js.api import Header, StorageType

from threetears.nats import NatsClient, set_default_namespace
from threetears.nats.errors import KvConfigMismatch
from threetears.nats.kv import build_kv_stream_config
from threetears.nats.raw_errors import is_bucket_not_found

pytestmark = pytest.mark.integration

#: the stale bucket-wide expiry the rate-limit bucket was created with
_STALE_MAX_AGE_SECONDS = 300

#: the per-entry lifetime a bind-only opener asks for, as the rate limiter and the nonce guard do
_OPENER_TTL = timedelta(seconds=60)


async def _create_stale(nc: NatsClient, *, full_name: str, max_age_seconds: int, storage: StorageType) -> None:
    """create the bucket the way the old declarers did, with a bucket-wide expiry and the given storage.

    :param nc: connected client
    :ptype nc: NatsClient
    :param full_name: fully-qualified bucket name
    :ptype full_name: str
    :param max_age_seconds: the bucket-wide expiry
    :ptype max_age_seconds: int
    :param storage: the storage the stale bucket lives on
    :ptype storage: StorageType
    :return: nothing
    :rtype: None
    """
    await nc.jetstream_context().add_stream(
        build_kv_stream_config(
            bucket=full_name, ttl_seconds=max_age_seconds, history=1, storage_type=storage, direct=True
        )
    )


async def _entry_ttl_header(nc: NatsClient, *, full_name: str, key: str) -> str | None:
    """the ``Nats-TTL`` header the server stored on a key's latest entry.

    :param nc: connected client
    :ptype nc: NatsClient
    :param full_name: fully-qualified bucket name
    :ptype full_name: str
    :param key: the key
    :ptype key: str
    :return: the header's value, or ``None`` when the entry carries none
    :rtype: str | None
    """
    stored = await nc.jetstream_context().get_last_msg(f"KV_{full_name}", f"$KV.{full_name}.{key}")
    return None if stored.headers is None else stored.headers.get(Header.MSG_TTL)


async def test_an_owner_removes_a_stale_expiry_and_openers_bind(nats_container: str) -> None:
    """the rate-limit bucket: the declaration reconciles ``max_age`` away in place and the opener binds.

    (a) declaring with ``ttl=None, owns_bucket=True`` leaves the live stream with no bucket-wide
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
        await _create_stale(
            declarer, full_name=full_name, max_age_seconds=_STALE_MAX_AGE_SECONDS, storage=StorageType.MEMORY
        )

        await declarer.ensure_kv_bucket(name="ratelimits", ttl=None, storage="memory", history=1, owns_bucket=True)

        live = (await js.stream_info(f"KV_{full_name}")).config
        assert not live.max_age, f"the stale bucket-wide expiry survived the declaration: max_age={live.max_age}"
        assert live.allow_msg_ttl is True

        opened = await pod.kv_bucket(name="ratelimits", ttl=_OPENER_TTL, create_if_missing=False)
        await opened.put(key="caller.window", value=b"1")

        assert await _entry_ttl_header(declarer, full_name=full_name, key="caller.window") == str(
            int(_OPENER_TTL.total_seconds())
        )
        assert await opened.get(key="caller.window") == b"1"


async def test_without_ownership_the_stale_expiry_stays_and_openers_are_refused(nats_container: str) -> None:
    """(c) the negative control for the rate-limit bucket: today's behaviour, unchanged by default."""
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
        await _create_stale(
            declarer, full_name=full_name, max_age_seconds=_STALE_MAX_AGE_SECONDS, storage=StorageType.MEMORY
        )

        await declarer.ensure_kv_bucket(name="ratelimits", ttl=None, storage="memory", history=1)

        assert (await js.stream_info(f"KV_{full_name}")).config.max_age == _STALE_MAX_AGE_SECONDS
        with pytest.raises(KvConfigMismatch, match="expires entries after 300s"):
            await pod.kv_bucket(name="ratelimits", ttl=_OPENER_TTL, create_if_missing=False)


async def test_an_owner_recreates_a_file_bucket_on_memory_and_openers_bind(nats_container: str) -> None:
    """the nonce bucket: file storage cannot change in place, so the owner -- told it may drop the
    file bucket's entries -- recreates it empty.

    It comes back on memory, with no bucket-wide expiry and per-entry TTLs allowed; a bind-only
    opener asking for 60s binds, its entries carry that lifetime, and an entry given a short one
    really expires.
    """
    namespace = "ownsstorage"
    full_name = f"{namespace}-proxy_assertion_nonces"
    set_default_namespace(namespace)
    async with (
        await NatsClient.connect(
            nats_url=nats_container, nats_subject_namespace=namespace, client_name="declarer"
        ) as declarer,
        await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="pod") as pod,
    ):
        js = declarer.jetstream_context()
        await _create_stale(declarer, full_name=full_name, max_age_seconds=60, storage=StorageType.FILE)
        stale = await js.key_value(full_name)
        await stale.put("before", b"1")

        await declarer.ensure_kv_bucket(
            name="proxy_assertion_nonces",
            ttl=None,
            storage="memory",
            history=1,
            owns_bucket=True,
            drop_file_storage=True,
        )

        live = (await js.stream_info(f"KV_{full_name}")).config
        assert live.storage == StorageType.MEMORY
        assert not live.max_age
        assert live.allow_msg_ttl is True
        assert (await js.stream_info(f"KV_{full_name}")).state.messages == 0, "the recreate kept entries"

        opened = await pod.kv_bucket(name="proxy_assertion_nonces", ttl=_OPENER_TTL, create_if_missing=False)
        await opened.put(key="nonce.a", value=b"1")
        assert await _entry_ttl_header(declarer, full_name=full_name, key="nonce.a") == str(
            int(_OPENER_TTL.total_seconds())
        )

        await opened.put(key="nonce.short", value=b"1", ttl=timedelta(seconds=1))
        assert await opened.get(key="nonce.short") == b"1"
        deadline = asyncio.get_running_loop().time() + 10.0
        while await opened.get(key="nonce.short") is not None:
            assert asyncio.get_running_loop().time() < deadline, "the per-entry lifetime was not honoured"
            await asyncio.sleep(0.25)
        assert await opened.get(key="nonce.a") == b"1"


async def test_an_owner_without_drop_file_storage_refuses_a_file_bucket_and_leaves_it(nats_container: str) -> None:
    """a NATS restart keeps a file bucket's entries, so dropping them needs a second, explicit opt-in.

    Without ``drop_file_storage`` the owner refuses the bucket and touches nothing: it stays on
    file, keeps its expiry, and keeps the entry written before the declaration.
    """
    namespace = "refusesfile"
    full_name = f"{namespace}-proxy_assertion_nonces"
    set_default_namespace(namespace)
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="declarer"
    ) as declarer:
        js = declarer.jetstream_context()
        await _create_stale(declarer, full_name=full_name, max_age_seconds=60, storage=StorageType.FILE)
        stale = await js.key_value(full_name)
        await stale.put("before", b"1")

        with pytest.raises(KvConfigMismatch, match="drop_file_storage=True"):
            await declarer.ensure_kv_bucket(
                name="proxy_assertion_nonces", ttl=None, storage="memory", history=1, owns_bucket=True
            )

        live = (await js.stream_info(f"KV_{full_name}")).config
        assert live.storage == StorageType.FILE
        assert live.max_age == 60
        assert (await stale.get("before")).value == b"1", "a refused owner dropped the entries"


async def test_without_ownership_a_file_bucket_stays_on_file(nats_container: str) -> None:
    """the negative control for the nonce bucket: the declaration reports the drift and leaves it."""
    namespace = "keepsstorage"
    full_name = f"{namespace}-proxy_assertion_nonces"
    set_default_namespace(namespace)
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="declarer"
    ) as declarer:
        js = declarer.jetstream_context()
        await _create_stale(declarer, full_name=full_name, max_age_seconds=60, storage=StorageType.FILE)

        await declarer.ensure_kv_bucket(name="proxy_assertion_nonces", ttl=None, storage="memory", history=1)

        live = (await js.stream_info(f"KV_{full_name}")).config
        assert live.storage == StorageType.FILE
        assert live.max_age == 60


async def test_two_owners_recreating_at_once_both_succeed(nats_container: str) -> None:
    """two replicas of the declaring service may declare at the same moment, and neither may fail."""
    namespace = "twoowners"
    full_name = f"{namespace}-proxy_assertion_nonces"
    set_default_namespace(namespace)
    async with (
        await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="hub-a") as a,
        await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="hub-b") as b,
    ):
        await _create_stale(a, full_name=full_name, max_age_seconds=60, storage=StorageType.FILE)

        await asyncio.gather(
            a.ensure_kv_bucket(
                name="proxy_assertion_nonces", ttl=None, storage="memory", owns_bucket=True, drop_file_storage=True
            ),
            b.ensure_kv_bucket(
                name="proxy_assertion_nonces", ttl=None, storage="memory", owns_bucket=True, drop_file_storage=True
            ),
        )

        live = (await a.jetstream_context().stream_info(f"KV_{full_name}")).config
        assert live.storage == StorageType.MEMORY
        assert not live.max_age


async def test_a_delete_of_a_stream_already_deleted_is_answered_not_found(nats_container: str) -> None:
    """the race the recreate tolerates: a concurrent owner's delete landed first.

    The tolerance rests on the server ANSWERING the second delete as not-found, in the shape the
    classifier recognises; a deadline there would be a refusal and must not be tolerated.
    """
    namespace = "deletetwice"
    full_name = f"{namespace}-gone"
    set_default_namespace(namespace)
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="declarer"
    ) as declarer:
        js = declarer.jetstream_context()
        await _create_stale(declarer, full_name=full_name, max_age_seconds=0, storage=StorageType.MEMORY)
        await js.delete_stream(f"KV_{full_name}")

        with pytest.raises(Exception) as caught:  # noqa: PT011 -- the classifier is what is under test
            await js.delete_stream(f"KV_{full_name}")

        assert is_bucket_not_found(caught.value), repr(caught.value)


async def test_an_owner_can_set_an_expiry_shorter_than_the_duplicate_window(nats_container: str) -> None:
    """JetStream refuses a duplicate window longer than ``max_age``; the reconcile brings both.

    A bucket created with no expiry carries the two-minute duplicate window, so moving it to a
    60s expiry in place is accepted only when the window comes down with it. The declared history
    is the owner's too, and is reconciled in place alongside.
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

        bucket = await declarer.ensure_kv_bucket(name="sessions", ttl=_OPENER_TTL, history=3, owns_bucket=True)

        live = (await js.stream_info(f"KV_{full_name}")).config
        assert live.max_age == _OPENER_TTL.total_seconds()
        assert live.duplicate_window == _OPENER_TTL.total_seconds()
        assert live.max_msgs_per_subject == 3
        await bucket.put(key="k", value=b"v")
        assert await bucket.get(key="k") == b"v"
