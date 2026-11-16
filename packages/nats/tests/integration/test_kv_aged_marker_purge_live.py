"""Integration: the owner's paged purge removes the markers stored before a cutoff and nothing else.

What a double cannot prove: that a ``DIRECT.GET`` of the next message at or after a sequence walks a real KV stream
in order with each message's server timestamp, that the walk stops at the first message stored at or after the
cutoff, that a filtered purge below a marker's sequence plus one removes just the marker, and that a follower
watching the bucket through it ends up where a fresh reader does.

Uses the session-scoped ``nats_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator
from contextlib import aclosing
from datetime import datetime

import pytest

from threetears.nats import NatsClient, NatsKvBucket, set_default_namespace
from threetears.nats.kv_watch import KvKeyUpdate
from threetears.nats.subject_permissions import kv_stream_name

pytestmark = pytest.mark.integration

_WAIT = 10.0


async def _next(watch: AsyncGenerator[KvKeyUpdate | None]) -> KvKeyUpdate | None:
    return await asyncio.wait_for(anext(watch), _WAIT)


async def _state(watch: AsyncGenerator[KvKeyUpdate | None]) -> dict[str, bytes]:
    """the live keys a fresh prefix watch delivers before it is caught up."""
    state: dict[str, bytes] = {}
    while (item := await _next(watch)) is not None:
        if item.deleted or item.value is None:
            state.pop(item.key, None)
        else:
            state[item.key] = item.value
    return state


async def _stored_at(nc: NatsClient, bucket: NatsKvBucket, key: str) -> tuple[int, datetime]:
    """the sequence and server timestamp of ``key``'s latest message."""
    message = await nc.jetstream_context().get_last_msg(
        kv_stream_name(bucket.name), f"$KV.{bucket.name}.{key}", direct=True
    )
    assert message.time is not None and message.seq is not None
    return int(message.seq), message.time


async def test_markers_before_the_cutoff_go_and_everything_else_stays(nats_container: str) -> None:
    namespace = f"amp{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="owner"
    ) as nc:
        bucket = await nc.kv_bucket(name="collections", history=1, direct=True)
        async with aclosing(bucket.watch_prefix(prefix="")) as follower:
            followed = await _state(follower)
            assert followed == {}

            # old: values, removed keys, and a key removed then written again (its latest message is a value)
            for n in range(5):
                await bucket.put(key=f"t.rows.kept-{n}", value=b"v")
            for n in range(20):
                await bucket.put(key=f"t.rows.gone-{n}", value=b"v")
                assert await bucket.delete(key=f"t.rows.gone-{n}")
            await bucket.put(key="t.rows.back", value=b"first")
            assert await bucket.delete(key="t.rows.back")
            await bucket.put(key="t.rows.back", value=b"again")

            # the cutoff is the server's own time for this message, so no clock but the server's is compared
            await bucket.put(key="t.fence", value=b"f")
            fence_sequence, cutoff = await _stored_at(nc, bucket, "t.fence")

            # young: removed after the cutoff, and a value written after it
            for n in range(4):
                await bucket.put(key=f"t.rows.young-{n}", value=b"v")
                assert await bucket.delete(key=f"t.rows.young-{n}")
            await bucket.put(key="t.rows.new", value=b"n")

            # the follower sees every change as it lands, removals included
            expected = {f"t.rows.kept-{n}": b"v" for n in range(5)} | {
                "t.rows.back": b"again",
                "t.fence": b"f",
                "t.rows.new": b"n",
            }
            while followed != expected:
                update = await _next(follower)
                assert update is not None
                if update.deleted or update.value is None:
                    followed.pop(update.key, None)
                else:
                    followed[update.key] = update.value

            stream = kv_stream_name(bucket.name)
            js = nc.jetstream_context()
            before = (await js.stream_info(stream)).state.messages
            assert before == 5 + 20 + 1 + 1 + 4 + 1

            # a dry run counts and removes nothing
            dry = await bucket.purge_aged_removal_markers(before=cutoff, max_messages=1_000, apply=False)
            assert (dry.purged, dry.done) == (20, True)
            assert (await js.stream_info(stream)).state.messages == before

            # paged: each page keeps to its budget, and the next resumes where it stopped
            purged = read = 0
            sequence = 1
            while True:
                page = await bucket.purge_aged_removal_markers(before=cutoff, from_sequence=sequence, max_messages=7)
                assert page.read <= 7
                purged += page.purged
                read += page.read
                sequence = page.next_sequence
                if page.done:
                    break
            assert purged == 20
            # it stopped at the first message stored at the cutoff, and the next page starts there
            assert sequence == fence_sequence

            assert (await js.stream_info(stream)).state.messages == before - 20
            for n in range(20):
                assert await bucket.get_latest(key=f"t.rows.gone-{n}") == (None, 0)
            for n in range(4):
                # a young marker is kept: its revision still fences a read that took it
                value, revision = await bucket.get_latest(key=f"t.rows.young-{n}")
                assert value is None and revision > fence_sequence
            for n in range(5):
                assert await bucket.get(key=f"t.rows.kept-{n}") == b"v"
            assert await bucket.get(key="t.rows.back") == b"again"
            assert sorted(await bucket.list_keys()) == sorted(expected)

            # a follower that missed nothing is where it was, and a purge sends it nothing to act on
            assert followed == expected
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(anext(follower), 0.5)

        # a reader arriving after the purge sees the same live keys the follower holds
        async with aclosing(bucket.watch_prefix(prefix="")) as fresh:
            assert await _state(fresh) == expected

        # the next pass from where this one stopped finds nothing old left
        again = await bucket.purge_aged_removal_markers(before=cutoff, from_sequence=sequence, max_messages=1_000)
        assert (again.read, again.purged, again.done) == (1, 0, True)

        # a key whose marker was purged is written again as any absent key is
        assert await bucket.create(key="t.rows.gone-0", value=b"reborn") is not None
        assert await bucket.get(key="t.rows.gone-0") == b"reborn"
