"""the shipped KV double's ``watch_key``, which consumers write their watcher tests against.

It has to yield what the real watch yields -- the key's latest message, a deletion marker included,
then every later write or delete of that key and no other -- or a watcher tested against it passes
on a stream the broker never produces.
"""

from __future__ import annotations

import asyncio
import inspect
from contextlib import aclosing

import pytest

from threetears.core.testing.kv import FakeKvBucket
from threetears.nats.kv import NatsKvBucket
from threetears.nats.kv_watch import KvKeyUpdate, KvKeyWatching


async def _next(watch: object) -> KvKeyUpdate:
    """the watch's next update, bounded so a missing one fails rather than hangs.

    :param watch: the watch iterator
    :ptype watch: object
    :return: the update
    :rtype: KvKeyUpdate
    """
    return await asyncio.wait_for(anext(watch), timeout=1)  # type: ignore[call-overload]


async def test_the_current_value_then_each_write_and_delete_of_the_key() -> None:
    bucket = FakeKvBucket("t-dv")
    first = await bucket.put(key="k", value=b"3")

    async with aclosing(bucket.watch_key(key="k")) as watch:
        assert await _next(watch) == KvKeyUpdate(key="k", value=b"3", revision=first)
        second = await bucket.put(key="k", value=b"4")
        assert await _next(watch) == KvKeyUpdate(key="k", value=b"4", revision=second)
        await bucket.delete(key="k")
        removal = await _next(watch)
        assert removal.deleted
        assert removal.revision == second + 1


async def test_a_deleted_key_is_first_delivered_as_its_marker() -> None:
    bucket = FakeKvBucket("t-dv")
    await bucket.put(key="k", value=b"3")
    await bucket.delete(key="k")

    async with aclosing(bucket.watch_key(key="k")) as watch:
        latest = await _next(watch)

    assert latest.deleted
    assert latest.revision == 2


async def test_a_key_with_no_message_yields_nothing_until_one_is_written() -> None:
    bucket = FakeKvBucket("t-dv")

    async with aclosing(bucket.watch_key(key="k")) as watch:
        pending = asyncio.ensure_future(anext(watch))
        await asyncio.sleep(0.01)
        assert not pending.done()
        await bucket.create(key="k", value=b"1")
        assert (await asyncio.wait_for(pending, timeout=1)).value == b"1"


async def test_another_keys_writes_are_not_delivered() -> None:
    bucket = FakeKvBucket("t-dv")

    async with aclosing(bucket.watch_key(key="k")) as watch:
        pending = asyncio.ensure_future(anext(watch))
        await bucket.put(key="other", value=b"9")
        await asyncio.sleep(0.01)
        assert not pending.done()
        await bucket.put(key="k", value=b"1")
        assert (await asyncio.wait_for(pending, timeout=1)).value == b"1"


async def test_closing_a_watch_stops_its_deliveries() -> None:
    bucket = FakeKvBucket("t-dv")
    await bucket.put(key="k", value=b"1")

    async with aclosing(bucket.watch_key(key="k")) as watch:
        await _next(watch)

    # nothing is left listening, so a later write reaches no closed watch
    await bucket.put(key="k", value=b"2")
    async with aclosing(bucket.watch_key(key="k")) as fresh:
        assert (await _next(fresh)).value == b"2"


@pytest.mark.parametrize("key", ["", "a.*", "a.>", "a b"])
async def test_a_key_that_is_not_literal_is_refused_as_the_real_watch_refuses_it(key: str) -> None:
    with pytest.raises(ValueError, match="literal key"):
        await anext(FakeKvBucket("t-dv").watch_key(key=key))


def test_the_double_satisfies_the_watching_protocol() -> None:
    assert isinstance(FakeKvBucket("t-dv"), KvKeyWatching)


def test_the_double_takes_the_real_watchs_parameters() -> None:
    # the parity marker on FakeKvBucket names KvBucketLike, which does not carry watch_key, so the
    # signature is held here instead.
    assert inspect.signature(FakeKvBucket.watch_key) == inspect.signature(NatsKvBucket.watch_key)
