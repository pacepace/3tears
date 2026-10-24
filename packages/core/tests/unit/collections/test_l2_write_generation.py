"""a write generation kept in L2: how each way it can fail is handled.

``L2_WRITE_GENERATION`` keeps a table's generation in one L2 key beside its rows (``{scope}.{table}-generation``).
Its advance can lose every compare-and-swap round or meet a broken broker; its read can meet a value that is no
token, or no value at all; and the pass that judges it can meet an error it does not expect. Exercised here
through ``VersionedAnswers``, whose answers and index both declare it, over 3tears' fake bus.
"""

from __future__ import annotations

import asyncio
import gzip
import uuid
from datetime import timedelta
from typing import Any

import pytest
from sqlalchemy import MetaData

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.generation import L2_WRITE_GENERATION, split_generation_token
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.versioned_answers import AnswerOutcome, VersionedAnswers
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import GenerationUnavailableError
from threetears.core.testing.kv import FakeKvBucket, FakeNatsClient
from threetears.nats import set_default_namespace
from threetears.nats.errors import KvError

_SCOPE = "tool_pod-l2generation"
_TABLE = "generation_answers"
_ANSWERS_KEY = f"{_SCOPE}.{_TABLE}-generation"
_INDEX_KEY = f"{_SCOPE}.{_TABLE}_index-generation"


@pytest.fixture(autouse=True)
def _namespace() -> None:
    set_default_namespace("l2generation")


async def _replica(bus: FakeNatsClient) -> VersionedAnswers:
    l1 = SQLiteBackend(db_name=f"l2generation-{uuid.uuid4().hex}")
    l1.initialize(MetaData())
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l2_client=bus, l3_pool=None, kv_key_scope=_SCOPE)  # type: ignore[arg-type]
    answers = VersionedAnswers(registry, DefaultCoreConfig(), bus, table_name=_TABLE)
    await registry.start_invalidation_listener(bus)  # type: ignore[arg-type]
    return answers


def _registry(answers: VersionedAnswers) -> CollectionRegistry:
    registry = answers.registry
    assert registry is not None
    return registry


async def _bucket(bus: FakeNatsClient) -> FakeKvBucket:
    bucket: FakeKvBucket = await bus.kv_bucket(name="collections")
    return bucket


def _break_writes_to(bucket: FakeKvBucket, key: str, *, how: str) -> None:
    """every write to ``key`` loses its swap (``"lose"``) or meets a broken broker (``"raise"``)."""
    for name in ("create", "update"):
        original = getattr(bucket, name)

        async def broken(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
            if kwargs.get("key") == key:
                if how == "raise":
                    raise KvError("broker unreachable")
                return None
            return await _original(*args, **kwargs)

        setattr(bucket, name, broken)


async def _text() -> str:
    return "rows"


class TestTheDeclaration:
    def test_it_refuses_to_cache_absences_too(self) -> None:
        with pytest.raises(TypeError, match="keeps its write generation in L2"):

            class _Refused(BaseCollection[BaseEntity]):  # pyright: ignore[reportUnusedClass]
                write_generation = L2_WRITE_GENERATION
                negative_cache_max_age = timedelta(seconds=5)


class TestAnAdvanceThatFails:
    @pytest.mark.parametrize("how", ["lose", "raise"])
    async def test_an_answer_whose_generation_cannot_advance_is_still_stored_and_given(self, how: str) -> None:
        bus = FakeNatsClient()
        answers = await _replica(bus)
        await answers.answer("v1", "warm", _text, order=1)  # the index's rows exist; only the answers' advance fails
        _break_writes_to(await _bucket(bus), _ANSWERS_KEY, how=how)
        given = await answers.answer_with_outcome("v1", "page", _text, order=1)
        assert given.outcome is AnswerOutcome.COMPUTED
        assert gzip.decompress(given.body) == b"rows"
        stored = [key for key in (await _bucket(bus)).keys() if key.startswith(f"{_SCOPE}.{_TABLE}.v1_")]
        assert len(stored) == 2, f"the answer was not stored: {stored}"

    @pytest.mark.parametrize("how", ["lose", "raise"])
    async def test_an_index_whose_generation_cannot_advance_answers_uncached(self, how: str) -> None:
        bus = FakeNatsClient()
        answers = await _replica(bus)
        _break_writes_to(await _bucket(bus), _INDEX_KEY, how=how)
        given = await answers.answer_with_outcome("v1", "page", _text, order=1)
        assert given.outcome is AnswerOutcome.INDEX_UNWRITABLE
        assert gzip.decompress(given.body) == b"rows"


class TestReadingTheGeneration:
    async def test_a_table_with_none_is_given_its_first_incarnation(self) -> None:
        bus = FakeNatsClient()
        answers = await _replica(bus)
        token = await answers.l2_write_generation()
        assert token is not None
        parts = split_generation_token(token)
        assert parts is not None and parts[1] == 0
        assert await (await _bucket(bus)).get(key=_ANSWERS_KEY) == token.encode()
        assert await answers.l2_write_generation() == token, "a second read minted another incarnation"

    async def test_a_value_that_is_no_token_is_unavailable_and_the_pass_leaves_the_table(self) -> None:
        bus = FakeNatsClient()
        answers = await _replica(bus)
        registry = _registry(answers)
        await registry.catch_up_l2_generations()  # first sight: the marks are taken
        await answers.answer("v1", "page", _text, order=1)
        await (await _bucket(bus)).put(key=_ANSWERS_KEY, value=b"not a generation")
        with pytest.raises(GenerationUnavailableError, match="malformed"):
            await answers.l2_write_generation()
        recorded = registry.generation_marks.recorded(_TABLE)
        assert await registry.catch_up_l2_generations() == 0
        assert registry.generation_marks.recorded(_TABLE) == recorded, "an unreadable generation moved the mark"
        assert answers.get_row_sync(answers.key_of("v1", "page")) is not None, "an unreadable generation dropped L1"


class TestAHostThatNeverSchedulesThePass:
    async def test_the_registry_warns_while_no_pass_judges_the_tables(self, caplog: pytest.LogCaptureFixture) -> None:
        registry = CollectionRegistry()
        registry.l2_catch_up_overdue = timedelta(seconds=0.02)
        bus = FakeNatsClient()
        registry.configure(l1_backend=None, l2_client=bus, l3_pool=None, kv_key_scope=_SCOPE)  # type: ignore[arg-type]
        VersionedAnswers(registry, DefaultCoreConfig(), bus, table_name=_TABLE)
        with caplog.at_level("WARNING", logger="threetears.core.collections.registry"):
            await asyncio.sleep(0.05)
        assert any("no catch-up pass has judged them" in r.getMessage() for r in caplog.records)
        assert registry.l2_catch_up_age is None
        await registry.stop_invalidation_listener()

    async def test_a_host_that_runs_the_pass_hears_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        registry = CollectionRegistry()
        registry.l2_catch_up_overdue = timedelta(seconds=0.05)
        bus = FakeNatsClient()
        registry.configure(l1_backend=None, l2_client=bus, l3_pool=None, kv_key_scope=_SCOPE)  # type: ignore[arg-type]
        VersionedAnswers(registry, DefaultCoreConfig(), bus, table_name=_TABLE)
        with caplog.at_level("WARNING", logger="threetears.core.collections.registry"):
            for _ in range(6):
                await registry.catch_up_l2_generations()
                await asyncio.sleep(0.02)
        assert not any("no catch-up pass" in r.getMessage() for r in caplog.records)
        age = registry.l2_catch_up_age
        assert age is not None and age < timedelta(seconds=1)
        await registry.stop_invalidation_listener()
