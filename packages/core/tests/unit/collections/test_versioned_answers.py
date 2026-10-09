"""answers computed once per version, read by every replica, and retired when the version moves.

Two replicas are two collections on two registries sharing one L2 bucket under one key scope, as the
replicas of one tool pod do. The interleavings a retirement can meet are pinned by holding one
writer's KV writes to one key at a gate, so each test drives exactly one ordering.
"""

from __future__ import annotations

import asyncio
import gzip
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.versioned_answers import AnswerNotComputable, VersionedAnswers
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.kv import FakeKvBucket, FakeNatsClient
from threetears.nats.collection_key_requests import CollectionKeysRequestUnavailableError
from threetears.nats.errors import KvError

_SCOPE = "tool_pod-answersunit"
_TABLE = "report_answers"


def _replica(nats: FakeNatsClient, l1_backend: Any = None) -> VersionedAnswers:
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1_backend, l2_client=nats, l3_pool=None, kv_key_scope=_SCOPE)  # type: ignore[arg-type]
    return VersionedAnswers(registry, DefaultCoreConfig(), nats, table_name=_TABLE)


class _Computer:
    """counts its computations and answers ``text``, optionally holding until released."""

    def __init__(self, text: str, *, hold: asyncio.Event | None = None) -> None:
        self.text = text
        self.calls = 0
        self.started = asyncio.Event()
        self._hold = hold

    async def __call__(self) -> str:
        self.calls += 1
        self.started.set()
        if self._hold is not None:
            await self._hold.wait()
        return self.text


def _text(compressed: bytes) -> str:
    return gzip.decompress(compressed).decode("utf-8")


async def _bucket(nats: FakeNatsClient) -> FakeKvBucket:
    bucket: FakeKvBucket = await nats.kv_bucket(name="collections")
    return bucket


def _entry_keys(bucket: FakeKvBucket, version: str) -> list[str]:
    """the answer entries of ``version`` live in the bucket."""
    return [key for key in bucket.keys() if key.startswith(f"{_SCOPE}.{_TABLE}.{version}_")]


def _shard_keys(bucket: FakeKvBucket, version: str) -> list[str]:
    """the index shards of ``version`` live in the bucket."""
    return [key for key in bucket.keys() if key.startswith(f"{_SCOPE}.{_TABLE}_index.{version}.")]


def _same_shard_request(answers: VersionedAnswers, version: str, request: str) -> str:
    """another request whose digest lands in the same index shard as ``request``'s."""
    first = answers.key_of(version, request)[1][0]
    return next(
        candidate
        for candidate in (f"{request}-{n}" for n in range(1000))
        if answers.key_of(version, candidate)[1][0] == first
    )


class _WriteGate:
    """holds writes to one key: the Nth write waits for ``open[N]``; each write done sets ``done[N]``."""

    def __init__(self, bucket: FakeKvBucket, key: str, writes: int) -> None:
        self.opened = [asyncio.Event() for _ in range(writes)]
        self.done = [asyncio.Event() for _ in range(writes)]
        self._seen = 0
        self._key = key
        for name in ("create", "update", "delete"):
            setattr(bucket, name, self._gated(getattr(bucket, name)))

    def _gated(self, original: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        async def gated(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("key") != self._key or self._seen >= len(self.opened):
                return await original(*args, **kwargs)
            index = self._seen
            self._seen += 1
            await self.opened[index].wait()
            try:
                return await original(*args, **kwargs)
            finally:
                self.done[index].set()

        return gated


class TestComputedOnce:
    async def test_an_answer_is_computed_once_and_served_again_compressed(self) -> None:
        answers = _replica(FakeNatsClient())
        compute = _Computer('{"rows": [1, 2, 3]}')
        first = await answers.answer("v1", "rows|contest|state=VA", compute, order=1)
        second = await answers.answer("v1", "rows|contest|state=VA", compute, order=1)
        assert _text(first) == _text(second) == '{"rows": [1, 2, 3]}'
        assert compute.calls == 1

    async def test_a_second_replica_serves_an_answer_it_did_not_compute(self) -> None:
        nats = FakeNatsClient()
        computed_here = _Computer("the answer")
        await _replica(nats).answer("v1", "rows|contest|", computed_here, order=1)
        never = _Computer("must not run")
        served = await _replica(nats).answer("v1", "rows|contest|", never, order=1)
        assert _text(served) == "the answer"
        assert (computed_here.calls, never.calls) == (1, 0)

    async def test_another_version_or_request_is_another_answer(self) -> None:
        answers = _replica(FakeNatsClient())
        compute = _Computer("x")
        await answers.answer("v1", "a", compute, order=1)
        await answers.answer("v2", "a", compute, order=2)
        await answers.answer("v1", "b", compute, order=1)
        assert compute.calls == 3

    async def test_a_failure_reaches_the_caller_and_is_not_cached(self) -> None:
        answers = _replica(FakeNatsClient())

        async def refuses() -> str:
            raise LookupError("v1 is not the current version")

        with pytest.raises(LookupError, match="not the current version"):
            await answers.answer("v1", "a", refuses, order=1)
        compute = _Computer("now it answers")
        assert _text(await answers.answer("v1", "a", compute, order=1)) == "now it answers"
        assert compute.calls == 1

    async def test_concurrent_callers_on_one_replica_compute_once(self) -> None:
        answers = _replica(FakeNatsClient())
        compute = _Computer("slow")
        got = await asyncio.gather(*(answers.answer("v1", "a", compute, order=1) for _ in range(5)))
        assert {_text(g) for g in got} == {"slow"}
        assert compute.calls == 1

    async def test_a_cancelled_caller_does_not_fail_the_one_computing(self) -> None:
        answers = _replica(FakeNatsClient())
        hold = asyncio.Event()
        first = _Computer("first", hold=hold)
        computing = asyncio.create_task(answers.answer("v1", "a", first, order=1))
        await first.started.wait()
        waiting = asyncio.create_task(answers.answer("v1", "a", _Computer("second"), order=1))
        await asyncio.sleep(0)
        waiting.cancel()
        hold.set()
        assert _text(await computing) == "first"
        with pytest.raises(asyncio.CancelledError):
            await waiting

    async def test_a_read_without_a_compute_is_refused_by_name(self) -> None:
        answers = _replica(FakeNatsClient())
        with pytest.raises(AnswerNotComputable, match="only through answer"):
            await answers.get_for(("v1", "a"))

    async def test_no_l1_is_taken_whatever_the_registry_offers(self) -> None:
        class _RecordingL1:
            """an L1 backend that records every call naming a table."""

            def __init__(self) -> None:
                self.tables: list[str] = []

            def __getattr__(self, name: str) -> Any:
                def call(*args: Any, **kwargs: Any) -> None:
                    self.tables.extend(str(a) for a in (*args, *kwargs.values()) if isinstance(a, str))

                return call

        l1 = _RecordingL1()
        answers = _replica(FakeNatsClient(), l1_backend=l1)
        assert _text(await answers.answer("v1", "a", _Computer("from L2 alone"), order=1)) == "from L2 alone"
        assert [table for table in l1.tables if table.startswith(_TABLE)] == []


class TestRetirement:
    async def test_retiring_deletes_older_versions_answers_and_keeps_the_current(self) -> None:
        nats = FakeNatsClient()
        answers = _replica(nats)
        for order, version in enumerate(("v1", "v2", "v3"), start=1):
            for request in ("a", "b"):
                await answers.answer(version, request, _Computer(f"{version}{request}"), order=order)
        assert await answers.retire_older_than(3) == 4
        bucket = await _bucket(nats)
        assert (_entry_keys(bucket, "v1"), _entry_keys(bucket, "v2"), len(_entry_keys(bucket, "v3"))) == ([], [], 2)
        assert (_shard_keys(bucket, "v1"), _shard_keys(bucket, "v2")) == ([], [])

    async def test_a_lagging_replica_never_retires_a_newer_version(self) -> None:
        nats = FakeNatsClient()
        leading, lagging = _replica(nats), _replica(nats)
        await leading.answer("v5", "a", _Computer("newest"), order=5)
        assert await lagging.retire_older_than(3) == 0
        lagging.current_version("v3", 3)
        await asyncio.sleep(0.01)
        assert len(_entry_keys(await _bucket(nats), "v5")) == 1

    async def test_a_version_older_than_a_retirement_is_answered_but_never_stored(self) -> None:
        nats = FakeNatsClient()
        answers = _replica(nats)
        await answers.retire_older_than(5)
        compute = _Computer("lagging copy")
        assert _text(await answers.answer("v2", "a", compute, order=2)) == "lagging copy"
        bucket = await _bucket(nats)
        assert (_entry_keys(bucket, "v2"), _shard_keys(bucket, "v2")) == ([], [])

    async def test_a_new_current_version_retires_the_old_once_and_retries_after_a_failure(self) -> None:
        nats = FakeNatsClient()
        answers = _replica(nats)
        await answers.answer("v1", "a", _Computer("old"), order=1)
        bucket = await _bucket(nats)
        original = bucket.update
        failures = [1]

        async def fails_once(*args: Any, **kwargs: Any) -> Any:
            if failures and "_index.versions" in str(kwargs.get("key")):
                failures.pop()
                raise KvError("broker unreachable")
            return await original(*args, **kwargs)

        bucket.update = fails_once  # type: ignore[method-assign]
        answers.current_version("v2", 2)
        await asyncio.sleep(0.01)
        assert len(_entry_keys(bucket, "v1")) == 1  # the first retirement failed
        answers.current_version("v2", 2)  # the next call tries again
        await asyncio.sleep(0.01)
        assert _entry_keys(bucket, "v1") == []


class TestNoAnswerOutlivesItsIndex:
    """every interleaving of a retirement with a replica computing the retired version leaves nothing."""

    async def test_retired_while_computing(self) -> None:
        nats = FakeNatsClient()
        writer, retirer = _replica(nats), _replica(nats)
        hold = asyncio.Event()
        compute = _Computer("late", hold=hold)
        computing = asyncio.create_task(writer.answer("v1", "a", compute, order=1))
        await compute.started.wait()  # recorded, computing, nothing stored yet
        await retirer.retire_older_than(2)
        hold.set()
        assert _text(await computing) == "late"  # the caller still gets its answer
        bucket = await _bucket(nats)
        assert (_entry_keys(bucket, "v1"), _shard_keys(bucket, "v1")) == ([], [])

    async def test_a_digest_recorded_between_the_retirements_read_and_its_delete(self) -> None:
        nats = FakeNatsClient()
        writer, retirer = _replica(nats), _replica(nats)
        await writer.answer("v1", "a", _Computer("first"), order=1)
        later = _same_shard_request(writer, "v1", "a")
        shard = f"{_SCOPE}.{_TABLE}_index.v1.{writer.key_of('v1', 'a')[1][0]}"
        bucket = await _bucket(nats)
        gate = _WriteGate(bucket, shard, writes=2)
        recording = asyncio.create_task(writer.answer("v1", later, _Computer("second"), order=1))
        await asyncio.sleep(0.01)  # the writer is held at its shard write
        retiring = asyncio.create_task(retirer.retire_older_than(2))
        await asyncio.sleep(0.01)  # the retirement has read the shard and is held at its delete
        gate.opened[0].set()  # the writer's digest lands now, between the read and the delete
        await gate.done[0].wait()
        gate.opened[1].set()
        await retiring
        assert _text(await recording) == "second"
        assert (_entry_keys(bucket, "v1"), _shard_keys(bucket, "v1")) == ([], [])

    async def test_a_digest_recorded_after_the_retirement_finished(self) -> None:
        nats = FakeNatsClient()
        writer, retirer = _replica(nats), _replica(nats)
        await writer.answer("v1", "a", _Computer("first"), order=1)
        later = _same_shard_request(writer, "v1", "a")
        shard = f"{_SCOPE}.{_TABLE}_index.v1.{writer.key_of('v1', 'a')[1][0]}"
        bucket = await _bucket(nats)
        gate = _WriteGate(bucket, shard, writes=1)
        recording = asyncio.create_task(writer.answer("v1", later, _Computer("second"), order=1))
        await asyncio.sleep(0.01)  # recorded its version before the floor rose; held at its shard write
        gate_retirement = asyncio.create_task(retirer.retire_older_than(2))
        await asyncio.sleep(0.01)
        gate.opened[0].set()
        await gate_retirement
        assert _text(await recording) == "second"
        assert (_entry_keys(bucket, "v1"), _shard_keys(bucket, "v1")) == ([], [])


class TestBookkeepingNeverFailsARead:
    async def test_an_unreachable_index_answers_uncached_and_stores_nothing(self) -> None:
        nats = FakeNatsClient()
        bucket = await _bucket(nats)
        original = bucket.get_latest

        async def unreachable(*args: Any, **kwargs: Any) -> Any:
            if "_index." in str(kwargs.get("key")):
                raise KvError("broker unreachable")
            return await original(*args, **kwargs)

        bucket.get_latest = unreachable  # type: ignore[method-assign]
        bucket.get_entry = unreachable  # type: ignore[method-assign]
        answers = _replica(nats)
        compute = _Computer("answered anyway")
        assert _text(await answers.answer("v1", "a", compute, order=1)) == "answered anyway"
        assert _entry_keys(bucket, "v1") == []

    async def test_an_index_that_keeps_changing_answers_uncached_and_stores_nothing(self) -> None:
        nats = FakeNatsClient()
        bucket = await _bucket(nats)

        async def always_loses(*args: Any, **kwargs: Any) -> None:
            del args, kwargs

        bucket.update = always_loses  # type: ignore[method-assign]
        bucket.create = always_loses  # type: ignore[method-assign]
        answers = _replica(nats)
        assert _text(await answers.answer("v1", "a", _Computer("answered anyway"), order=1)) == "answered anyway"
        assert _entry_keys(bucket, "v1") == []


def test_a_version_that_cannot_be_a_key_segment_is_refused() -> None:
    answers = _replica(FakeNatsClient())
    for version in ("v1_2", "v 1", "v1.2"):
        with pytest.raises(ValueError, match="version key segment"):
            answers.key_of(version, "a")


def test_the_request_is_digested_into_the_key() -> None:
    version, digest = _replica(FakeNatsClient()).key_of("v1", "rows|contest|state=VA & county=Loudoun")
    assert version == "v1"
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


class TestRetiredKeysArePurged:
    async def test_every_key_a_retired_version_touched_is_purged_relative_to_the_scope(self) -> None:
        purged: list[str] = []

        async def purger(keys: list[str]) -> int:
            purged.extend(keys)
            return len(keys)

        nats = FakeNatsClient()
        registry = CollectionRegistry()
        registry.configure(l1_backend=None, l2_client=nats, l3_pool=None, kv_key_scope=_SCOPE)  # type: ignore[arg-type]
        answers = VersionedAnswers(registry, DefaultCoreConfig(), nats, table_name=_TABLE, purger=purger)
        await answers.answer("v1", "a", _Computer("old"), order=1)
        await answers.answer("v2", "a", _Computer("new"), order=2)
        await answers.retire_older_than(2)
        digest = answers.key_of("v1", "a")[1]
        assert f"{_TABLE}.v1_{digest}" in purged
        assert f"{_TABLE}_index.v1.{digest[0]}" in purged  # the one shard v1 held, deleted then purged
        assert not any(key.startswith(_SCOPE) or "v2" in key for key in purged)

    async def test_a_purger_that_cannot_be_reached_keeps_the_markers_and_says_so_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def unreachable(keys: list[str]) -> int:
            raise CollectionKeysRequestUnavailableError("no hub answers this request")

        nats = FakeNatsClient()
        registry = CollectionRegistry()
        registry.configure(l1_backend=None, l2_client=nats, l3_pool=None, kv_key_scope=_SCOPE)  # type: ignore[arg-type]
        answers = VersionedAnswers(registry, DefaultCoreConfig(), nats, table_name=_TABLE, purger=unreachable)
        for order in (1, 2, 3):
            await answers.answer(f"v{order}", "a", _Computer("x"), order=order)
        await answers.retire_older_than(2)
        await answers.retire_older_than(3)
        bucket = await _bucket(nats)
        assert (_entry_keys(bucket, "v1"), _entry_keys(bucket, "v2"), len(_entry_keys(bucket, "v3"))) == ([], [], 1)
        said = [r for r in caplog.records if "could not be purged" in r.getMessage()]
        assert len(said) == 1


def _with_purger(nats: FakeNatsClient, purger: Callable[[list[str]], Awaitable[int]]) -> VersionedAnswers:
    registry = CollectionRegistry()
    registry.configure(l1_backend=None, l2_client=nats, l3_pool=None, kv_key_scope=_SCOPE)  # type: ignore[arg-type]
    return VersionedAnswers(registry, DefaultCoreConfig(), nats, table_name=_TABLE, purger=purger)


async def _eventually(condition: Callable[[], bool], seconds: float = 2.0) -> bool:
    deadline = asyncio.get_running_loop().time() + seconds
    while not condition() and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)
    return condition()


class TestAWriterRetiredWhileComputing:
    async def test_its_taken_back_answer_and_emptied_shard_are_purged(self) -> None:
        purged: list[str] = []

        async def recording(keys: list[str]) -> int:
            purged.extend(keys)
            return len(keys)

        nats = FakeNatsClient()
        writer, retirer = _with_purger(nats, recording), _replica(nats)
        hold = asyncio.Event()
        compute = _Computer("late", hold=hold)
        computing = asyncio.create_task(writer.answer("v1", "a", compute, order=1))
        await compute.started.wait()
        await retirer.retire_older_than(2)  # the writer holds the key's lock: its digest is kept
        hold.set()
        assert _text(await computing) == "late"
        digest = writer.key_of("v1", "a")[1]
        taken_back = {f"{_TABLE}.v1_{digest}", f"{_TABLE}_index.v1.{digest[0]}"}
        assert await _eventually(lambda: taken_back <= set(purged))
        bucket = await _bucket(nats)
        assert (_entry_keys(bucket, "v1"), _shard_keys(bucket, "v1")) == ([], [])

    async def test_a_writer_cancelled_after_its_answer_landed_leaves_it_for_the_next_retirement(self) -> None:
        nats = FakeNatsClient()
        writer, retirer = _replica(nats), _replica(nats)
        hold = asyncio.Event()
        compute = _Computer("written, then the writer went", hold=hold)
        computing = asyncio.create_task(writer.answer("v1", "a", compute, order=1))
        await compute.started.wait()
        await retirer.retire_older_than(2)  # the digest is kept: its writer holds the lock
        bucket = await _bucket(nats)
        entry = f"{_SCOPE}.{_TABLE}.v1_{writer.key_of('v1', 'a')[1]}"
        landed = asyncio.Event()
        original = bucket.create

        async def lands_then_signals(*args: Any, **kwargs: Any) -> Any:
            written = await original(*args, **kwargs)
            if kwargs.get("key") == entry:
                landed.set()
                await asyncio.sleep(3600)  # the writer dies here, after its answer landed, before its check
            return written

        bucket.create = lands_then_signals  # type: ignore[method-assign]
        hold.set()
        await landed.wait()
        computing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await computing
        bucket.create = original  # type: ignore[method-assign]
        assert _entry_keys(bucket, "v1") != []  # written, and its writer gone without taking it back
        await retirer.retire_older_than(2)  # the next retirement: no writer holds the lock now
        assert (_entry_keys(bucket, "v1"), _shard_keys(bucket, "v1")) == ([], [])

    async def test_an_unreadable_floor_takes_the_answer_back_and_still_answers(self) -> None:
        nats = FakeNatsClient()
        bucket = await _bucket(nats)
        hold = asyncio.Event()
        compute = _Computer("answered", hold=hold)
        answers = _replica(nats)
        computing = asyncio.create_task(answers.answer("v1", "a", compute, order=1))
        await compute.started.wait()

        def unreadable(original: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            async def read(*args: Any, **kwargs: Any) -> Any:
                if str(kwargs.get("key")).endswith("_index.versions"):
                    raise KvError("broker unreachable")
                return await original(*args, **kwargs)

            return read

        bucket.get_latest = unreadable(bucket.get_latest)  # type: ignore[method-assign]
        bucket.get_entry = unreadable(bucket.get_entry)  # type: ignore[method-assign]
        hold.set()
        assert _text(await computing) == "answered"
        assert _entry_keys(bucket, "v1") == []  # unknown counts as retired: never kept


class TestAPurgeNeverFailsAReadNorWedgesARetirement:
    async def test_a_purger_raising_anything_leaves_retirement_working(self) -> None:
        async def broken(keys: list[str]) -> int:
            raise RuntimeError("anything at all")

        nats = FakeNatsClient()
        answers = _with_purger(nats, broken)
        for order in (1, 2, 3):
            await answers.answer(f"v{order}", "a", _Computer("x"), order=order)
        answers.current_version("v2", 2)
        answers.current_version("v3", 3)
        bucket = await _bucket(nats)
        assert await _eventually(lambda: _entry_keys(bucket, "v1") == [] and _entry_keys(bucket, "v2") == [])
        assert len(_entry_keys(bucket, "v3")) == 1

    async def test_the_take_back_purge_does_not_hold_the_read(self) -> None:
        never = asyncio.Event()

        async def hangs(keys: list[str]) -> int:
            await never.wait()
            return 0

        nats = FakeNatsClient()
        writer, retirer = _with_purger(nats, hangs), _replica(nats)
        hold = asyncio.Event()
        compute = _Computer("late", hold=hold)
        computing = asyncio.create_task(writer.answer("v1", "a", compute, order=1))
        await compute.started.wait()
        await retirer.retire_older_than(2)
        hold.set()
        assert _text(await asyncio.wait_for(computing, timeout=2.0)) == "late"
