"""the catch-up pass judges a table whose write generation is kept in L2 from L2, never from the epoch bucket.

A versioned-answer cache keeps its tables in L2 alone, so no broker advances their generation and the epoch
bucket holds none: a pass that read them there would find no generation and drop them on every pass. The pass
reads them from L2 instead (``CollectionRegistry.catch_up_l2_generations``), and still catches a broadcast
the follower missed.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import timedelta
from typing import Any

import pytest
from sqlalchemy import MetaData

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.versioned_answers import AnswerOutcome, VersionedAnswers
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.kv import FakeNatsClient
from threetears.epoch import EpochGenerationReader, follow_l2_kept_tables, generation_catchup_tick
from threetears.nats import set_default_namespace

_SCOPE = "tool_pod-followl2"
_TABLE = "followed_answers"


@pytest.fixture(autouse=True)
def _namespace() -> None:
    set_default_namespace("followl2probe")


class _ReadsRecorded(EpochGenerationReader):
    """the platform reader, recording which tables the pass asked it for."""

    def __init__(self, bus: FakeNatsClient) -> None:
        super().__init__(bus, create_if_missing=True)  # type: ignore[arg-type]
        self.asked: list[str] = []

    async def read(self, table_name: str) -> str | None:
        self.asked.append(table_name)
        return await super().read(table_name)


async def _replica(bus: FakeNatsClient) -> VersionedAnswers:
    l1 = SQLiteBackend(db_name=f"followl2-{uuid.uuid4().hex}")
    l1.initialize(MetaData())
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l2_client=bus, l3_pool=None, kv_key_scope=_SCOPE)  # type: ignore[arg-type]
    answers = VersionedAnswers(registry, DefaultCoreConfig(), bus, table_name=_TABLE)
    await registry.start_invalidation_listener(bus)  # type: ignore[arg-type]
    return answers


async def _text() -> str:
    return "rows"


class TestTheCatchUpPassReadsL2KeptTablesFromL2:
    async def test_the_epoch_bucket_is_not_asked_and_a_heard_table_is_not_dropped(self) -> None:
        bus = FakeNatsClient()
        writer, follower = await _replica(bus), await _replica(bus)
        registry = follower.registry
        assert registry is not None
        reader = _ReadsRecorded(bus)
        await generation_catchup_tick(registry, reader)  # first sight
        await writer.answer("v1", "page", _text, order=1)
        assert await generation_catchup_tick(registry, reader) == 0
        assert reader.asked == [], "the pass read an L2-kept table's generation from the epoch bucket"

    async def test_a_missed_broadcast_is_caught_by_the_pass(self) -> None:
        bus = FakeNatsClient()
        writer, follower = await _replica(bus), await _replica(bus)
        registry = follower.registry
        assert registry is not None
        reader = _ReadsRecorded(bus)
        await writer.answer("v1", "page", _text, order=1)
        await generation_catchup_tick(registry, reader)
        await follower.answer("v1", "page", _text, order=1)
        await registry.stop_invalidation_listener()
        await writer.answer("v2", "page", _text, order=2)
        await writer.retire_older_than(2)
        await registry.start_invalidation_listener(bus)  # type: ignore[arg-type]
        assert await generation_catchup_tick(registry, reader) == 2
        given = await follower.answer_with_outcome("v1", "page", _text, order=1)
        assert given.outcome is AnswerOutcome.RETIRED, "the follower answered a retired version from its L1"


class TestAnUnexpectedErrorInTheL2Pass:
    async def test_it_reaches_the_caller_after_the_pass(self) -> None:
        bus = FakeNatsClient()
        answers = await _replica(bus)
        registry = answers.registry
        assert registry is not None
        bucket = await bus.kv_bucket(name="collections")
        original = bucket.get

        async def broken(*args: Any, **kwargs: Any) -> Any:
            if str(kwargs.get("key")).endswith("-generation"):
                raise RuntimeError("a bug in the read")
            return await original(*args, **kwargs)

        bucket.get = broken  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="a bug in the read"):
            await generation_catchup_tick(registry, _ReadsRecorded(bus))


class TestTheTablesAreFollowedByTheirKeysWithNoPass:
    async def test_a_missed_retirement_is_applied_when_its_advance_is_pushed(self) -> None:
        bus = FakeNatsClient()
        writer, follower = await _replica(bus), await _replica(bus)
        registry = follower.registry
        assert registry is not None
        await writer.answer("v1", "page", _text, order=1)
        following = follow_l2_kept_tables(registry, grace=timedelta(0))
        try:
            await _until(lambda: following.healthy)
            await follower.answer("v1", "page", _text, order=1)
            await registry.stop_invalidation_listener()  # the follower misses every broadcast after this
            await writer.answer("v2", "page", _text, order=2)
            await writer.retire_older_than(2)
            await registry.start_invalidation_listener(bus)  # type: ignore[arg-type]

            async def retired() -> bool:
                given = await follower.answer_with_outcome("v1", "page", _text, order=1)
                return given.outcome is AnswerOutcome.RETIRED

            for _ in range(200):
                if await retired():
                    break
                await asyncio.sleep(0.005)
            assert await retired(), "the pushed advance was not applied: the follower answered a retired version"
            assert not registry.behind_on(_TABLE)
        finally:
            await following.stop()


async def _until(condition: Any) -> None:
    for _ in range(400):
        if condition():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("the follower never became healthy")
