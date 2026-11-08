"""A KV write JetStream refuses for want of room, or for its size, raises a type that says so.

The L2 accessors in 3tears core count every failure they degrade on by reason, and an operator's
remedy differs by reason: a full bucket wants a larger ``max_bytes`` or fewer keys, an exhausted
server wants storage, an oversized value wants a smaller row, and a transport failure wants the
broker back. A plain ``KvError`` for all of them makes a full bucket read exactly like an outage.

Every type here is still a ``KvError``, so a handler that catches ``KvError`` keeps catching it.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from nats.errors import ConnectionClosedError as NatsConnectionClosedError
from nats.errors import MaxPayloadError as NatsMaxPayloadError
from nats.js.errors import APIError

from threetears.nats import (
    KvError,
    KvFullError,
    KvStoreExhaustedError,
    KvValueTooLargeError,
    NatsKvBucket,
)


# parity-exempt: a KeyValue handle whose every write raises one chosen error, counting the attempts it saw
class _RefusingKv:
    """a KeyValue handle that refuses every write with ``error``, counting the attempts."""

    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.attempts = 0

    async def put(self, key: str, value: bytes) -> int:
        self.attempts += 1
        raise self.error

    async def create(self, key: str, value: bytes) -> int:
        self.attempts += 1
        raise self.error

    async def update(self, key: str, value: bytes, last: int | None = None) -> int:
        self.attempts += 1
        raise self.error


# parity-exempt: a client whose JetStream re-open hands back a second refusing handle, so a retry is visible
class _Client:
    def __init__(self, reopened: _RefusingKv) -> None:
        self._reopened = reopened
        self.raw = object()

    def jetstream_context(self) -> _Client:
        return self

    async def add_stream(self, _config: object) -> None:
        # the re-open declares the stream before binding; here it is declared without error
        return None

    async def create_key_value(self, _config: object) -> _RefusingKv:
        return self._reopened

    async def key_value(self, _name: str) -> _RefusingKv:
        return self._reopened


def _bucket(error: BaseException) -> tuple[NatsKvBucket, _RefusingKv, _RefusingKv]:
    """a bucket whose handle, and whose re-opened handle, both refuse with ``error``."""
    first, reopened = _RefusingKv(error), _RefusingKv(error)
    bucket = NatsKvBucket(
        client=_Client(reopened),  # type: ignore[arg-type]
        full_name="KV_collections",
        kv=first,  # type: ignore[arg-type]
        ttl=timedelta(seconds=60),
    )
    return bucket, first, reopened


async def test_a_write_past_the_buckets_max_bytes_is_a_full_bucket_naming_it() -> None:
    bucket, _, _ = _bucket(APIError(code=503, err_code=10077, description="maximum bytes exceeded"))
    with pytest.raises(KvFullError) as raised:
        await bucket.put(key="k", value=b"v")
    assert raised.value.bucket == "KV_collections"
    assert "maximum bytes exceeded" in str(raised.value)
    assert isinstance(raised.value, KvError)
    assert raised.value.server_error_code == 10077


async def test_an_account_storage_limit_is_a_full_bucket() -> None:
    bucket, _, _ = _bucket(APIError(code=500, err_code=10047, description="insufficient storage resources available"))
    with pytest.raises(KvFullError):
        await bucket.put(key="k", value=b"v")


async def test_a_server_out_of_store_is_an_exhausted_store() -> None:
    bucket, _, _ = _bucket(APIError(code=503, err_code=10023, description="insufficient resources"))
    with pytest.raises(KvStoreExhaustedError) as raised:
        await bucket.create(key="k", value=b"v")
    assert raised.value.bucket == "KV_collections"
    assert not isinstance(raised.value, KvFullError)


async def test_a_value_over_the_streams_max_message_size_is_too_large() -> None:
    bucket, _, _ = _bucket(APIError(code=400, err_code=10054, description="message size exceeds maximum allowed"))
    with pytest.raises(KvValueTooLargeError):
        await bucket.update(key="k", value=b"v", revision=3)


async def test_a_value_over_the_brokers_max_payload_is_too_large() -> None:
    bucket, _, _ = _bucket(NatsMaxPayloadError())
    with pytest.raises(KvValueTooLargeError):
        await bucket.put(key="k", value=b"v")


async def test_a_refusal_for_room_is_not_retried_through_a_reopen() -> None:
    """a full bucket is just as full after a re-open; the retry would only double the round trips."""
    bucket, first, reopened = _bucket(APIError(code=503, err_code=10077, description="maximum bytes exceeded"))
    with pytest.raises(KvFullError):
        await bucket.put(key="k", value=b"v")
    assert (first.attempts, reopened.attempts) == (1, 0)


@pytest.mark.parametrize(
    ("error", "server_error_code"),
    [
        (APIError(code=503, err_code=10077, description="maximum messages exceeded"), 10077),
        (APIError(code=500, err_code=10118, description="stream is offline"), 10118),
        (NatsConnectionClosedError(), None),
    ],
)
async def test_any_other_failure_stays_a_plain_kv_error_saying_whether_the_server_answered(
    error: BaseException, server_error_code: int | None
) -> None:
    bucket, _, _ = _bucket(error)
    with pytest.raises(KvError) as raised:
        await bucket.put(key="k", value=b"v")
    assert not isinstance(raised.value, (KvFullError, KvStoreExhaustedError, KvValueTooLargeError))
    assert raised.value.server_error_code == server_error_code
