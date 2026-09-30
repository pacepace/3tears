"""tests: a KV coordination primitive can BIND a bucket somebody else declared, never creating it.

A tool pod granted an agent's coordination buckets holds key-addressed access to exactly
those buckets -- STREAM.INFO, the reads, and (for a write grant) the ``$KV.`` publishes --
and no stream-admin verb. An opener that tries STREAM.CREATE first is refused, and a JetStream
refusal is never answered: it blocks to its deadline before the bind that would have succeeded
is attempted. So a primitive over a bucket its process does not own must be able to skip the
create entirely.

The contract pinned, for :class:`DistributedCounter`, :class:`TokenBucket` and :class:`KVLease`:

- the default is unchanged: the primitive DECLARES its bucket (``create_if_missing=True``);
- ``create_if_missing=False`` reaches the client's opener as a bind;
- through the real :class:`~threetears.nats.NatsClient` opener, a bind-only primitive never
  issues STREAM.CREATE -- only the bind;
- a bind-only primitive over a bucket nobody created fails, rather than creating it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.core.coordination import DistributedCounter, KVLease, TokenBucket
from threetears.core.testing.kv import FakeNatsClient
from threetears.nats import NatsClient

#: builds one primitive over ``client`` with the given create flag (``None`` = the default),
#: and returns a coroutine function that makes the primitive open its bucket.
_Opener = Callable[[Any, "bool | None"], Callable[[], Awaitable[object]]]


def _counter(client: Any, create_if_missing: bool | None) -> Callable[[], Awaitable[object]]:
    """a counter whose first read opens its bucket.

    :param client: the KV-capable client
    :ptype client: Any
    :param create_if_missing: the flag to pass, or ``None`` to take the default
    :ptype create_if_missing: bool | None
    :return: a coroutine function that opens the bucket
    :rtype: Callable[[], Awaitable[object]]
    """
    extra = {} if create_if_missing is None else {"create_if_missing": create_if_missing}
    counter = DistributedCounter(client, bucket_name="owned", ttl=timedelta(minutes=2), **extra)
    return lambda: counter.get("k")


def _token_bucket(client: Any, create_if_missing: bool | None) -> Callable[[], Awaitable[object]]:
    """a token bucket whose first claim opens its bucket.

    :param client: the KV-capable client
    :ptype client: Any
    :param create_if_missing: the flag to pass, or ``None`` to take the default
    :ptype create_if_missing: bool | None
    :return: a coroutine function that opens the bucket
    :rtype: Callable[[], Awaitable[object]]
    """
    extra = {} if create_if_missing is None else {"create_if_missing": create_if_missing}
    bucket = TokenBucket(client, bucket_name="owned", refill_rate=1.0, capacity=5.0, **extra)
    return lambda: bucket.claim("k")


def _lease(client: Any, create_if_missing: bool | None) -> Callable[[], Awaitable[object]]:
    """a lease factory whose first fail-fast acquire opens its bucket.

    :param client: the KV-capable client
    :ptype client: Any
    :param create_if_missing: the flag to pass, or ``None`` to take the default
    :ptype create_if_missing: bool | None
    :return: a coroutine function that opens the bucket
    :rtype: Callable[[], Awaitable[object]]
    """
    extra = {} if create_if_missing is None else {"create_if_missing": create_if_missing}
    lease = KVLease(client, bucket_name="owned", pod_id="pod-a", **extra)
    return lambda: lease.acquire("k", ttl_seconds=30, max_wait_seconds=0)


_PRIMITIVES: list[Any] = [
    pytest.param(_counter, id="DistributedCounter"),
    pytest.param(_token_bucket, id="TokenBucket"),
    pytest.param(_lease, id="KVLease"),
]


class _RecordingClient(FakeNatsClient):
    """the shipped in-memory client, recording the create flag of every bucket open."""

    def __init__(self) -> None:
        super().__init__()
        self.create_flags: list[bool] = []

    async def kv_bucket(self, **kwargs: Any) -> Any:  # type: ignore[override]
        """record ``create_if_missing`` and delegate to the in-memory open.

        :param kwargs: the open's keyword arguments
        :ptype kwargs: Any
        :return: the in-memory bucket
        :rtype: Any
        """
        self.create_flags.append(kwargs.get("create_if_missing", True))
        return await super().kv_bucket(**kwargs)


def _client_over_recording_jetstream() -> tuple[NatsClient, MagicMock]:
    """a real :class:`NatsClient` whose JetStream context records every call.

    The bind returns a nats-py ``KeyValue`` stand-in whose reads miss and whose writes
    succeed, which is all a first counter read, token claim or lease acquire needs.

    :return: the client and the recording JetStream context
    :rtype: tuple[NatsClient, MagicMock]
    """
    kv = MagicMock()
    kv.get = AsyncMock(side_effect=_key_not_found)
    kv.create = AsyncMock(return_value=1)
    kv.update = AsyncMock(return_value=2)
    kv.put = AsyncMock(return_value=1)
    js = MagicMock()
    js.add_stream = AsyncMock()
    js.update_stream = AsyncMock()
    js.key_value = AsyncMock(return_value=kv)
    # a bind-only open that asks for an entry lifetime reads the live bucket's own expiry: the hub
    # declares pod buckets with none and per-entry TTLs allowed.
    js.stream_info = AsyncMock(return_value=MagicMock(config=MagicMock(max_age=0.0, allow_msg_ttl=True)))
    raw = MagicMock()
    raw.jetstream = MagicMock(return_value=js)
    return NatsClient(raw=raw, namespace="ns", client_name="bind-only-test"), js


async def _key_not_found(*_args: object, **_kwargs: object) -> None:
    """raise the nats-py miss a read of an absent key raises.

    :return: never returns
    :rtype: None
    :raises nats.js.errors.KeyNotFoundError: always
    """
    import nats.js.errors

    raise nats.js.errors.KeyNotFoundError


class TestTheDefaultStillDeclares:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("build", _PRIMITIVES)
    async def test_an_unflagged_primitive_asks_to_create_its_bucket(self, build: _Opener) -> None:
        client = _RecordingClient()
        await build(client, None)()
        assert client.create_flags == [True]


class TestBindOnly:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("build", _PRIMITIVES)
    async def test_the_flag_reaches_the_opener_as_a_bind(self, build: _Opener) -> None:
        client = _RecordingClient()
        await client.kv_bucket(name="owned")  # the owner declared it
        client.create_flags.clear()
        await build(client, False)()
        assert client.create_flags == [False]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("build", _PRIMITIVES)
    async def test_a_bind_only_open_never_issues_stream_create(self, build: _Opener) -> None:
        client, js = _client_over_recording_jetstream()
        await build(client, False)()
        js.add_stream.assert_not_awaited()
        js.update_stream.assert_not_awaited()
        js.key_value.assert_awaited_once_with("ns-owned")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("build", _PRIMITIVES)
    async def test_the_declaring_open_does_issue_stream_create(self, build: _Opener) -> None:
        # the positive control for the test above: the same harness sees a create when one is made.
        client, js = _client_over_recording_jetstream()
        await build(client, None)()
        js.add_stream.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("build", _PRIMITIVES)
    async def test_a_bind_only_primitive_does_not_create_an_absent_bucket(self, build: _Opener) -> None:
        client = FakeNatsClient()
        with pytest.raises(KeyError):
            await build(client, False)()
        with pytest.raises(KeyError):
            await client.kv_bucket(name="owned", create_if_missing=False)
