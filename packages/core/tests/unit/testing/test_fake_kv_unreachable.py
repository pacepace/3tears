"""the shipped KV double can be made unreachable, so a consumer tests its outage path on it.

A coordination primitive's hardest property is what it does while the bucket cannot be reached:
a lease must not give up a claim over one failed renewal, and must give it up once its TTL has
passed. Without a way to make the shared double fail, every consumer that tested that path wrote
its own KV copy, or patched the double's methods one at a time and missed the ones it did not
think of.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest

from threetears.core.testing.kv import FakeKvBucket, FakeNatsClient


def _every_operation(bucket: FakeKvBucket) -> dict[str, Callable[[], Awaitable[object]]]:
    """one call of each async bucket operation, by name."""
    return {
        "create": lambda: bucket.create(key="k", value=b"v"),
        "get": lambda: bucket.get(key="k"),
        "get_entry": lambda: bucket.get_entry(key="k"),
        "get_latest": lambda: bucket.get_latest(key="k"),
        "update": lambda: bucket.update(key="k", value=b"v", revision=1),
        "delete": lambda: bucket.delete(key="k"),
        "put": lambda: bucket.put(key="k", value=b"v"),
        "list_keys": lambda: bucket.list_keys(),
        "date_created": lambda: bucket.date_created(),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation", ["create", "get", "get_entry", "get_latest", "update", "delete", "put", "list_keys", "date_created"]
)
async def test_every_operation_raises_the_given_error_while_unreachable(operation: str) -> None:
    bucket = await FakeNatsClient().kv_bucket(name="claims")
    error = ConnectionError("kv is unreachable")
    bucket.become_unreachable(error)

    with pytest.raises(ConnectionError) as raised:
        await _every_operation(bucket)[operation]()

    assert raised.value is error


@pytest.mark.asyncio
async def test_watching_a_key_raises_while_unreachable() -> None:
    bucket = await FakeNatsClient().kv_bucket(name="claims")
    bucket.become_unreachable(TimeoutError("kv deadline"))

    with pytest.raises(TimeoutError):
        await anext(bucket.watch_key(key="k"))


@pytest.mark.asyncio
async def test_nothing_changes_while_unreachable_and_the_data_is_there_after() -> None:
    """an unreachable bucket is a transport failure, not a lost one: no write lands, none is lost."""
    bucket = await FakeNatsClient().kv_bucket(name="claims")
    revision = await bucket.create(key="k", value=b"before")
    bucket.become_unreachable(ConnectionError("kv is unreachable"))

    with pytest.raises(ConnectionError):
        await bucket.put(key="k", value=b"during")
    bucket.become_reachable()

    assert await bucket.get_entry(key="k") == (b"before", revision)


@pytest.mark.asyncio
async def test_the_client_hands_back_the_same_unreachable_bucket() -> None:
    """a consumer reaches its bucket through the client, so the failure must be on that instance."""
    client = FakeNatsClient()
    (await client.kv_bucket(name="claims")).become_unreachable(ConnectionError("kv is unreachable"))

    with pytest.raises(ConnectionError):
        await (await client.kv_bucket(name="claims", create_if_missing=False)).get(key="k")


def test_a_bucket_is_reachable_until_told_otherwise() -> None:
    bucket = FakeKvBucket(bucket_name="claims")
    assert bucket.unreachable_error is None
    bucket.become_unreachable(ConnectionError("down"))
    assert isinstance(bucket.unreachable_error, ConnectionError)
    bucket.become_reachable()
    assert bucket.unreachable_error is None
