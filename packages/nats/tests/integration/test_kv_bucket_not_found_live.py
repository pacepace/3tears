"""an absent KV bucket raises KvBucketNotFoundError against a real broker, and a raw handle's failures classify.

The unit suite drives nats-py over a scripted wire; this proves the scripted answers are the ones a
real JetStream server gives. A second connection plays the declarer: it declares the bucket, the
client under test binds it, and the declarer then deletes the backing stream -- what a NATS restart
does to memory storage before the declarer has come back.

Uses the session-scoped ``nats_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from threetears.nats import (
    KvBucketNotFoundError,
    KvError,
    NatsClient,
    is_bucket_not_found,
    is_key_not_found,
    is_nats_error,
    set_default_namespace,
)
from threetears.nats.kv import KvTimings

pytestmark = pytest.mark.integration

#: a declarer wait shrunk to keep each absent-bucket case well under a second
_FAST = KvTimings(
    bind_wait_for_declarer_seconds=0.3,
    bind_retry_first_delay_seconds=0.01,
    bind_retry_max_delay_seconds=0.05,
)


@asynccontextmanager
async def _clients(nats_url: str) -> AsyncIterator[tuple[NatsClient, NatsClient, str]]:
    """a declarer and a binder on one fresh namespace.

    :param nats_url: the broker
    :ptype nats_url: str
    :return: ``(declarer, binder, namespace)``
    :rtype: AsyncIterator[tuple[NatsClient, NatsClient, str]]
    """
    namespace = f"kvabsent{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    declarer = await NatsClient.connect(nats_url=nats_url, nats_subject_namespace=namespace, client_name="declarer")
    binder = await NatsClient.connect(
        nats_url=nats_url, nats_subject_namespace=namespace, client_name="binder", kv_timings=_FAST
    )
    try:
        yield declarer, binder, namespace
    finally:
        await binder.shutdown()
        await declarer.shutdown()


async def _raised_by(coro: Any) -> BaseException:
    """await ``coro`` and return what it raised.

    :param coro: the awaitable to run
    :ptype coro: Any
    :return: the exception
    :rtype: BaseException
    :raises AssertionError: when it raised nothing
    """
    try:
        await coro
    except BaseException as exc:  # noqa: BLE001 -- the exception IS the result under test
        return exc
    raise AssertionError("expected an exception, got none")


async def test_a_bind_only_open_of_a_bucket_nobody_declared_raises_the_typed_error(nats_container: str) -> None:
    async with _clients(nats_container) as (_declarer, binder, namespace):
        raised = await _raised_by(binder.kv_bucket(name="never", create_if_missing=False))

    assert isinstance(raised, KvBucketNotFoundError), repr(raised)
    assert raised.bucket == f"{namespace}-never"
    assert raised.__cause__ is not None and is_bucket_not_found(raised.__cause__)


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda b: b.get(key="k"), id="get"),
        pytest.param(lambda b: b.put(key="k", value=b"v"), id="put"),
        pytest.param(lambda b: b.create(key="fresh", value=b"v"), id="create"),
        pytest.param(lambda b: b.date_created(), id="date_created"),
    ],
)
async def test_an_operation_on_a_bucket_its_declarer_lost_raises_the_typed_error(
    nats_container: str, operation: Any
) -> None:
    async with _clients(nats_container) as (declarer, binder, namespace):
        await declarer.ensure_kv_bucket(name="lost")
        bound = await binder.kv_bucket(name="lost", create_if_missing=False)
        await bound.put(key="k", value=b"v")
        await declarer.jetstream_context().delete_stream(f"KV_{namespace}-lost")

        raised = await _raised_by(operation(bound))

    assert isinstance(raised, KvBucketNotFoundError), repr(raised)
    assert raised.bucket == f"{namespace}-lost"


async def test_a_bound_bucket_that_exists_still_answers(nats_container: str) -> None:
    # the positive control: the same harness, nothing deleted, every operation lands.
    async with _clients(nats_container) as (declarer, binder, _namespace):
        await declarer.ensure_kv_bucket(name="kept")
        bound = await binder.kv_bucket(name="kept", create_if_missing=False)
        await bound.put(key="k", value=b"v")
        assert await bound.get(key="k") == b"v"


async def test_a_raw_handles_failures_classify_as_the_server_answered_them(nats_container: str) -> None:
    async with _clients(nats_container) as (declarer, _binder, namespace):
        js = declarer.jetstream_context()
        absent_bind = await _raised_by(js.key_value(f"{namespace}-nope"))

        await declarer.ensure_kv_bucket(name="raw")
        raw = await js.key_value(f"{namespace}-raw")
        missing_key = await _raised_by(raw.get("missing"))

        await js.delete_stream(f"KV_{namespace}-raw")
        write_to_vanished = await _raised_by(raw.put("k", b"v"))
        info_of_vanished = await _raised_by(js.stream_info(f"KV_{namespace}-raw"))
        direct_get_of_vanished = await _raised_by(raw.get("k"))

    assert is_bucket_not_found(absent_bind), repr(absent_bind)
    assert is_bucket_not_found(write_to_vanished), repr(write_to_vanished)
    assert is_bucket_not_found(info_of_vanished), repr(info_of_vanished)

    assert is_key_not_found(missing_key), repr(missing_key)
    assert not is_bucket_not_found(missing_key), "a missing key says the bucket exists"

    # a direct get nobody serves is a no-responders, which any unserved subject answers too
    assert is_nats_error(direct_get_of_vanished), repr(direct_get_of_vanished)
    assert not is_bucket_not_found(direct_get_of_vanished), repr(direct_get_of_vanished)

    for raised in (absent_bind, missing_key, write_to_vanished, info_of_vanished):
        assert is_nats_error(raised)
        assert not isinstance(raised, KvError), "a raw handle's failures are nats-py's own"
