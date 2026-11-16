"""The collections bucket's marker purge: a pass resumes where the last stopped, keeps to its budget and its lease.

The walk itself (a page of the stream by sequence, each marker purged below its own sequence plus one) is the
bucket's, proven against a real server in ``packages/nats/tests/integration/test_kv_aged_marker_purge_live.py``;
here the bucket is a recorded stand-in, and what is proven is the purger around it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from threetears.core.collections.marker_purge import L2_MARKER_PURGE_AGE, RemovalMarkerPurger
from threetears.core.coordination.lease import KVLease
from threetears.core.testing.kv import FakeKvBucket
from threetears.nats import AgedMarkerPurge

_NOW = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)


@dataclass
class _Message:
    sequence: int
    stored: datetime
    marker: bool


# parity-exempt: the one bucket verb a purger calls, over a list of messages; the server's walk is proven live
@dataclass
class _Stream:
    messages: list[_Message]
    created: datetime = _NOW - timedelta(days=14)
    pages: list[tuple[int, int, bool]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "ns-collections"

    async def date_created(self) -> datetime:
        return self.created

    async def purge_aged_removal_markers(
        self, *, before: datetime, from_sequence: int = 1, max_messages: int, apply: bool = True
    ) -> AgedMarkerPurge:
        self.pages.append((from_sequence, max_messages, apply))
        read = purged = 0
        oldest: datetime | None = None
        sequence = from_sequence
        for message in [m for m in self.messages if m.sequence >= from_sequence]:
            if read == max_messages:
                return AgedMarkerPurge(read, purged, oldest, sequence, done=False)
            read += 1
            if message.stored >= before:
                return AgedMarkerPurge(read, purged, oldest, message.sequence, done=True)
            sequence = message.sequence + 1
            if message.marker:
                oldest = oldest or message.stored
                purged += 1
                if apply:
                    self.messages.remove(message)
        return AgedMarkerPurge(read, purged, oldest, sequence, done=True)


def _stream(old_markers: int, *, young: int = 2, values: int = 3) -> _Stream:
    """``old_markers`` markers two days old, ``values`` old values among them, then ``young`` markers an hour old."""
    messages: list[_Message] = []
    sequence = 1
    for n in range(old_markers + values):
        messages.append(_Message(sequence, _NOW - timedelta(days=2) + timedelta(seconds=n), marker=n >= values))
        sequence += 1
    for n in range(young):
        messages.append(_Message(sequence, _NOW - timedelta(hours=1) + timedelta(seconds=n), marker=True))
        sequence += 1
    return _Stream(messages)


def _purger(stream: _Stream, **kwargs: object) -> RemovalMarkerPurger:
    return RemovalMarkerPurger(stream, lease=None, clock=lambda: _NOW, **kwargs)  # type: ignore[arg-type]


class TestAPass:
    async def test_it_purges_the_old_markers_and_keeps_the_young_and_the_values(self) -> None:
        stream = _stream(10)

        result = await _purger(stream).run_pass()

        assert (result.purged, result.behind, result.skipped) == (10, False, False)
        assert [m.marker for m in stream.messages] == [False, False, False, True, True]
        assert result.oldest_marker_age == timedelta(days=2) - timedelta(seconds=3)

    async def test_the_next_pass_starts_where_this_one_stopped(self) -> None:
        stream = _stream(10)
        purger = _purger(stream)
        await purger.run_pass()
        stopped_at = stream.pages[-1]

        again = await purger.run_pass()

        assert again.purged == 0
        # the first message read is the young one the last pass stopped at, not the stream's first
        assert stream.pages[-1][0] == 14
        assert stopped_at[0] == 1

    async def test_a_pass_stops_at_its_budget_and_says_it_is_behind(self) -> None:
        stream = _stream(30)
        purger = _purger(stream, max_messages_per_pass=12, page_messages=5)

        first = await purger.run_pass()

        assert (first.read, first.behind) == (12, True)
        assert [page[1] for page in stream.pages] == [5, 5, 2]
        second = await purger.run_pass()
        third = await purger.run_pass()
        assert first.purged + second.purged + third.purged == 30
        assert third.behind is False

    async def test_a_dry_run_purges_nothing_and_does_not_move_the_position(self) -> None:
        stream = _stream(10)
        purger = _purger(stream)

        dry = await purger.run_pass(apply=False)
        real = await purger.run_pass()

        assert dry.purged == real.purged == 10
        assert stream.pages[-1][0] == 1

    async def test_a_stream_created_again_is_walked_from_its_first_message(self) -> None:
        stream = _stream(5)
        purger = _purger(stream)
        await purger.run_pass()
        stream.messages = _stream(4).messages
        stream.created = _NOW - timedelta(minutes=5)

        result = await purger.run_pass()

        assert result.purged == 4
        assert stream.pages[-1][0] == 1

    async def test_the_default_age_is_a_day(self) -> None:
        stream = _Stream([_Message(1, _NOW - timedelta(hours=23), marker=True)])

        assert (await _purger(stream).run_pass()).purged == 0
        assert L2_MARKER_PURGE_AGE == timedelta(hours=24)

    def test_an_age_that_is_not_positive_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            RemovalMarkerPurger(_Stream([]), lease=None, age=timedelta(0))


class TestTheLease:
    async def test_a_pass_skips_while_another_replica_holds_it(self) -> None:
        leases = FakeKvBucket("ns-leases")
        mine = KVLease(None, bucket=leases, pod_id="hub-a")
        theirs = KVLease(None, bucket=leases, pod_id="hub-b")
        stream = _stream(3)
        held = await theirs.hold("kv-marker-purge.ns-collections", ttl=60, renew_every=20)
        try:
            result = await RemovalMarkerPurger(stream, lease=mine, clock=lambda: _NOW).run_pass()  # type: ignore[arg-type]
        finally:
            await held.release()

        assert result.skipped is True
        assert stream.pages == []

    async def test_a_pass_under_the_lease_releases_it(self) -> None:
        leases = FakeKvBucket("ns-leases")
        stream = _stream(3)

        result = await RemovalMarkerPurger(
            stream,  # type: ignore[arg-type]
            lease=KVLease(None, bucket=leases, pod_id="hub-a"),
            clock=lambda: _NOW,
        ).run_pass()

        assert result.purged == 3
        assert await KVLease(None, bucket=leases, pod_id="hub-b").holder("kv-marker-purge.ns-collections") is None

    async def test_a_pass_stops_between_pages_once_the_lease_is_lost(self) -> None:
        # parity-exempt: a held lease reduced to its loss signal, set by the first page
        class _Held:
            def __init__(self) -> None:
                self.lost = asyncio.Event()

            async def __aenter__(self) -> _Held:
                return self

            async def __aexit__(self, *_: object) -> None:
                return None

        held = _Held()

        class _Lease:
            async def hold(self, key: str, **_: object) -> _Held:
                return held

        stream = _stream(30)
        purge = stream.purge_aged_removal_markers

        async def losing(**kwargs: Any) -> AgedMarkerPurge:
            held.lost.set()
            return await purge(**kwargs)

        stream.purge_aged_removal_markers = losing  # type: ignore[method-assign]

        result = await RemovalMarkerPurger(
            stream,  # type: ignore[arg-type]
            lease=_Lease(),  # type: ignore[arg-type]
            page_messages=5,
            clock=lambda: _NOW,
        ).run_pass()

        assert (result.lost, result.behind, result.read) == (True, True, 5)
