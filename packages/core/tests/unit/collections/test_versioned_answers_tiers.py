"""an answer cache's reads stay at L1 between version moves, say how each answer was given, and announce
versions that never step back.

Two replicas are two collections on two registries, each with its own L1, sharing one L2 bucket under one
key scope and hearing each other's broadcasts through the registry's invalidation listener, as the
replicas of one tool pod do. Reads are counted by 3tears' own tier counters (``tier_reads_total``).
"""

from __future__ import annotations

import asyncio
import gzip
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from sqlalchemy import Column as SaColumn
from sqlalchemy import MetaData, Table, Text

from threetears.core.cache.duckdb import DuckDBBackend
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import tier_reads_total
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.versioned_answers import AnswerOutcome, VersionedAnswers
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.kv import FakeKvBucket, FakeNatsClient
from threetears.nats.errors import KvError

_SCOPE = "tool_pod-answerstiers"
_TABLE = "tier_answers"
_INDEX = f"{_TABLE}_index"

Reads = dict[str, dict[str, int]]


def _l1() -> SQLiteBackend:
    l1 = SQLiteBackend(db_name=f"answers-l1-{uuid.uuid4().hex}")
    l1.initialize(MetaData())
    return l1


async def _replica(bus: FakeNatsClient, *, l1: Any = None) -> VersionedAnswers:
    """one replica: its own L1, the shared L2, and the invalidation listener a pod starts."""
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1 or _l1(), l2_client=bus, l3_pool=None, kv_key_scope=_SCOPE)  # type: ignore[arg-type]
    answers = VersionedAnswers(registry, DefaultCoreConfig(), bus, table_name=_TABLE)
    await registry.start_invalidation_listener(bus)  # type: ignore[arg-type]
    return answers


def _answer_reads() -> Reads:
    return {table: tiers for table, tiers in tier_reads_total().items() if table in (_TABLE, _INDEX)}


async def _moved(step: Callable[[], Awaitable[Any]]) -> Reads:
    """the answer cache's reads ``step`` moved, by table then tier; a tier it did not move is absent."""
    before = _answer_reads()
    await step()
    moved: Reads = {}
    for table, tiers in _answer_reads().items():
        changed = {tier: reads - before.get(table, {}).get(tier, 0) for tier, reads in tiers.items()}
        changed = {tier: reads for tier, reads in changed.items() if reads}
        if changed:
            moved[table] = changed
    return moved


def _tiers(moved: Reads) -> set[str]:
    return {tier for tiers in moved.values() for tier in tiers}


class _Computer:
    """counts its computations and answers ``text``, optionally holding until released or failing."""

    def __init__(self, text: str = '{"rows": []}', *, hold: asyncio.Event | None = None, fail: bool = False) -> None:
        self.text = text
        self.calls = 0
        self.started = asyncio.Event()
        self._hold = hold
        self._fail = fail

    async def __call__(self) -> str:
        self.calls += 1
        self.started.set()
        if self._hold is not None:
            await self._hold.wait()
        if self._fail:
            raise LookupError("the copy moved under this compute")
        return self.text


async def _bucket(bus: FakeNatsClient) -> FakeKvBucket:
    bucket: FakeKvBucket = await bus.kv_bucket(name="collections")
    return bucket


def _entry_keys(bucket: FakeKvBucket, version: str) -> list[str]:
    return [key for key in bucket.keys() if key.startswith(f"{_SCOPE}.{_TABLE}.{version}_")]


def _count_l2_calls(bucket: FakeKvBucket) -> list[str]:
    """every L2 operation on ``bucket`` from here on, by name."""
    calls: list[str] = []
    for name in ("get", "get_entry", "get_latest", "create", "update", "put", "delete"):
        original = getattr(bucket, name)

        def counted(*args: Any, _name: str = name, _original: Any = original, **kwargs: Any) -> Any:
            calls.append(_name)
            return _original(*args, **kwargs)

        setattr(bucket, name, counted)
    return calls


class TestARepeatedReadIsReadFromL1:
    async def test_a_repeated_read_moves_only_l1_and_makes_no_l2_round_trip(self) -> None:
        bus = FakeNatsClient()
        answers = await _replica(bus)
        compute = _Computer()
        await answers.answer("v1", "page", compute, order=1)
        calls = _count_l2_calls(await _bucket(bus))
        moved = await _moved(lambda: answers.answer("v1", "page", compute, order=1))
        assert moved == {_TABLE: {"l1": 1}}, f"a repeated read left L1: {moved}"
        assert calls == [], f"a repeated read went to L2 (the index's compare-and-set among them): {calls}"
        assert compute.calls == 1

    async def test_the_answers_and_the_index_are_kept_in_the_registrys_l1(self) -> None:
        l1 = _l1()
        answers = await _replica(FakeNatsClient(), l1=l1)
        await answers.answer("v1", "page", _Computer(), order=1)
        assert l1.has_table(_TABLE) and l1.has_table(_INDEX)
        assert answers.get_row_sync(answers.key_of("v1", "page")) is not None


class TestAPeersAnswerComesThroughL2:
    async def test_a_peers_stored_answer_moves_l2_never_l3_and_then_only_l1(self) -> None:
        bus = FakeNatsClient()
        a, b = await _replica(bus), await _replica(bus)
        await a.answer("v1", "page", _Computer("from A"), order=1)
        never = _Computer("B must not compute")

        first = await _moved(lambda: b.answer("v1", "page", never, order=1))
        assert first == {_TABLE: {"l2": 1}}, f"B read A's answer past L2: {first}"
        again = await _moved(lambda: b.answer("v1", "page", never, order=1))
        assert again == {_TABLE: {"l1": 1}}, again
        assert never.calls == 0

    async def test_an_answer_waited_for_while_a_peer_computes_it_moves_l2_never_l3(self) -> None:
        bus = FakeNatsClient()
        a, b = await _replica(bus), await _replica(bus)
        release = asyncio.Event()
        slow = _Computer("from A", hold=release)
        computing = asyncio.create_task(a.answer("v1", "page", slow, order=1))
        await slow.started.wait()  # A holds the key's build lock and computes
        never = _Computer("B must not compute")
        waiting = asyncio.create_task(b.answer_with_outcome("v1", "page", never, order=1))
        await asyncio.sleep(0.05)  # B missed L2, found A's lock held, and waits on it
        release.set()
        await computing  # A's answer lands; B's next look at L2 (its first is at 0.1 s) finds it

        async def b_finds_it() -> None:
            assert (await waiting).outcome is AnswerOutcome.READ

        moved = await _moved(b_finds_it)
        assert never.calls == 0
        assert moved == {_TABLE: {"l2": 1}}, f"B counted A's answer as its own computation: {moved}"


class TestTheIndexIsReadFromL1BetweenMoves:
    async def test_an_index_read_between_moves_is_l1_and_a_peers_change_is_read_once_from_l2(self) -> None:
        bus = FakeNatsClient()
        a, b = await _replica(bus), await _replica(bus)
        await a.announce("v1", 1, scope="all")

        between = await _moved(lambda: a.announced("all"))
        assert between == {_INDEX: {"l1": 1}}, f"the index was read past L1 between moves: {between}"

        # a peer announces a newer version: the index row changes, and its broadcast evicts A's copy
        await b.announce("v2", 2, scope="all")
        after_peer = await _moved(lambda: a.announced("all"))
        assert after_peer == {_INDEX: {"l2": 1}}, f"A did not read the index the peer changed: {after_peer}"

        settled = await _moved(lambda: a.announced("all"))
        assert settled == {_INDEX: {"l1": 1}}, settled

    async def test_an_answer_read_reads_no_index_past_l1(self) -> None:
        bus = FakeNatsClient()
        a = await _replica(bus)
        await a.answer("v1", "first", _Computer(), order=1)
        between = await _moved(lambda: a.answer("v1", "second", _Computer(), order=1))
        assert _INDEX not in between, f"an answer between moves counted an index read: {between}"

    async def test_a_retired_answer_leaves_every_replicas_l1(self) -> None:
        bus = FakeNatsClient()
        a, b = await _replica(bus), await _replica(bus)
        await a.answer("v1", "page", _Computer(), order=1)
        await b.answer("v1", "page", _Computer(), order=1)  # B holds A's answer in its own L1 now
        await a.answer("v2", "page", _Computer(), order=2)
        await a.retire_older_than(2)
        compute = _Computer()
        given = await b.answer_with_outcome("v1", "page", compute, order=1)
        assert compute.calls == 1, "B answered a retired version from its L1"
        assert given.outcome is AnswerOutcome.RETIRED


class TestAMissedBroadcastIsCaughtUp:
    async def test_a_replica_that_missed_a_retirement_drops_its_l1_at_the_next_catch_up(self) -> None:
        bus = FakeNatsClient()
        a, b = await _replica(bus), await _replica(bus)
        b_registry = b.registry
        assert b_registry is not None
        await a.answer("v1", "page", _Computer(), order=1)
        await b_registry.catch_up_l2_generations()  # B's first sight of the tables: its marks are taken
        await b.answer("v1", "page", _Computer(), order=1)  # B holds the answer in its L1
        assert await b_registry.catch_up_l2_generations() == 0, "B dropped tables it had heard every write of"

        await b_registry.stop_invalidation_listener()  # B's listener is down: it misses what follows
        await a.answer("v2", "page", _Computer(), order=2)
        await a.retire_older_than(2)
        await b_registry.start_invalidation_listener(bus)  # type: ignore[arg-type]
        stale = _Computer()
        assert (await b.answer_with_outcome("v1", "page", stale, order=1)).outcome is AnswerOutcome.READ
        assert stale.calls == 0  # the missed retirement left the answer in B's L1

        assert await b_registry.catch_up_l2_generations() == 2, "the catch-up did not see the missed writes"
        caught_up = _Computer()
        given = await b.answer_with_outcome("v1", "page", caught_up, order=1)
        assert caught_up.calls == 1, "B still answered a retired version from its L1"
        assert given.outcome is AnswerOutcome.RETIRED

    async def test_a_replica_that_heard_every_write_drops_nothing(self) -> None:
        bus = FakeNatsClient()
        a, b = await _replica(bus), await _replica(bus)
        b_registry = b.registry
        assert b_registry is not None
        await b_registry.catch_up_l2_generations()
        for order in (1, 2, 3):
            await a.answer(f"v{order}", "page", _Computer(), order=order)
            await a.retire_older_than(order)
        assert await b_registry.catch_up_l2_generations() == 0


class TestTheOutcomeIsReported:
    async def test_computed_then_read(self) -> None:
        answers = await _replica(FakeNatsClient())
        compute = _Computer("x")
        first = await answers.answer_with_outcome("v1", "page", compute, order=1)
        second = await answers.answer_with_outcome("v1", "page", compute, order=1)
        assert (first.outcome, second.outcome) == (AnswerOutcome.COMPUTED, AnswerOutcome.READ)
        assert gzip.decompress(first.body) == gzip.decompress(second.body) == b"x"
        assert first.outcome.stored and second.outcome.stored

    async def test_a_peers_answer_is_read(self) -> None:
        bus = FakeNatsClient()
        a, b = await _replica(bus), await _replica(bus)
        await a.answer("v1", "page", _Computer(), order=1)
        assert (await b.answer_with_outcome("v1", "page", _Computer(), order=1)).outcome is AnswerOutcome.READ

    async def test_computed_after_a_peer_let_go_of_the_lock_without_its_value_is_a_fallback(self) -> None:
        bus = FakeNatsClient()
        a, b = await _replica(bus), await _replica(bus)
        b.peer_retake_seconds = 0.05  # type: ignore[misc]
        release = asyncio.Event()
        failing = _Computer(hold=release, fail=True)
        computing = asyncio.create_task(a.answer("v1", "page", failing, order=1))
        await failing.started.wait()  # A holds the build lock
        fallback = _Computer("from B")
        waiting = asyncio.create_task(b.answer_with_outcome("v1", "page", fallback, order=1))
        await asyncio.sleep(0.05)  # B found the lock held and waits on it
        release.set()  # A's compute fails: it lets go of the lock and no value lands
        given = await waiting
        assert fallback.calls == 1
        assert given.outcome is AnswerOutcome.PEER_FALLBACK
        assert given.outcome.stored
        computing.cancel()
        await asyncio.gather(computing, return_exceptions=True)

    async def test_a_retired_version_is_answered_and_not_stored(self) -> None:
        answers = await _replica(FakeNatsClient())
        await answers.retire_older_than(5)
        given = await answers.answer_with_outcome("v2", "page", _Computer("lagging"), order=2)
        assert given.outcome is AnswerOutcome.RETIRED and not given.outcome.stored
        assert gzip.decompress(given.body) == b"lagging"

    async def test_an_unwritable_index_is_answered_and_not_stored(self) -> None:
        bus = FakeNatsClient()
        answers = await _replica(bus)
        bucket = await _bucket(bus)
        original = bucket.create

        async def refuses(*args: Any, **kwargs: Any) -> Any:
            if "_index." in str(kwargs.get("key")):
                raise KvError("broker unreachable")
            return await original(*args, **kwargs)

        bucket.create = refuses  # type: ignore[method-assign]
        given = await answers.answer_with_outcome("v1", "page", _Computer(), order=1)
        assert given.outcome is AnswerOutcome.INDEX_UNWRITABLE and not given.outcome.stored
        assert _entry_keys(bucket, "v1") == []


class TestTheAnnouncedVersionNeverStepsBack:
    async def test_a_replica_behind_names_the_newer_version_another_announced(self) -> None:
        bus = FakeNatsClient()
        a, b = await _replica(bus), await _replica(bus)
        announced, record = await a.announce("v8", 8, scope="all", detail={"behind": []})
        assert announced.version == "v8" and record is not None and record.previous is None
        announced, record = await b.announce("v7", 7, scope="all")
        assert announced.version == "v8", "B's index stepped back behind A's"
        assert announced.detail == {"behind": []}
        newest, record = await a.announce("v9", 9, scope="all")
        assert newest.version == "v9"
        assert record is not None and [v.version for v in record.versions] == ["v9", "v8"]
        announced, _ = await b.announce("v8", 8, scope="all")
        assert announced.version == "v9", "B's index stepped back behind A's"
        recorded = await b.announced("all")
        assert recorded is not None and [v.version for v in recorded.versions] == ["v9", "v8"]

    async def test_each_scope_keeps_its_own_record(self) -> None:
        answers = await _replica(FakeNatsClient())
        await answers.announce("v9", 9, scope="all")
        announced, _ = await answers.announce("v3", 3, scope="VA")
        assert announced.version == "v3"

    async def test_retirement_keeps_the_version_replaced_and_retires_below_it(self) -> None:
        bus = FakeNatsClient()
        answers = await _replica(bus)
        bucket = await _bucket(bus)
        for order in (7, 8):
            await answers.answer(f"v{order}", "page", _Computer(), order=order)
            await answers.announce(f"v{order}", order)
        await asyncio.sleep(0.05)
        assert _entry_keys(bucket, "v7") != [], "the version an index named a moment ago lost its answers"
        await answers.answer("v9", "page", _Computer(), order=9)
        await answers.announce("v9", 9)
        for _ in range(100):
            if not _entry_keys(bucket, "v7"):
                break
            await asyncio.sleep(0.01)
        assert _entry_keys(bucket, "v7") == [], "a version two moves behind was kept"
        assert _entry_keys(bucket, "v8") != [], "the version replaced was retired"

    async def test_a_record_that_cannot_be_written_announces_the_replicas_own(self) -> None:
        bus = FakeNatsClient()
        answers = await _replica(bus)
        bucket = await _bucket(bus)

        async def unreachable(*args: Any, **kwargs: Any) -> Any:
            raise KvError("broker unreachable")

        bucket.get_entry = unreachable  # type: ignore[method-assign]
        announced, record = await answers.announce("v7", 7)
        assert (announced.version, record) == ("v7", None)


class TestScopesRetireOnTheirOwn:
    async def test_one_scopes_move_retires_none_of_anothers_answers(self) -> None:
        bus = FakeNatsClient()
        answers = await _replica(bus)
        bucket = await _bucket(bus)
        await answers.answer("va3", "page", _Computer(), order=3, scope="VA")
        await answers.announce("va3", 3, scope="VA")
        for order in (7, 8, 9):
            await answers.answer(f"all{order}", "page", _Computer(), order=order, scope="all")
            await answers.announce(f"all{order}", order, scope="all")
        for _ in range(100):
            if not _entry_keys(bucket, "all7"):
                break
            await asyncio.sleep(0.01)
        assert _entry_keys(bucket, "all7") == [], "the scope's own older version was kept"
        assert _entry_keys(bucket, "va3") != [], "another scope's move retired VA's current answers"
        fresh = await answers.answer_with_outcome("va3", "another page", _Computer(), order=3, scope="VA")
        assert fresh.outcome is AnswerOutcome.COMPUTED, "VA's version was refused below another scope's floor"
        assert fresh.outcome.stored

    async def test_retiring_one_scope_leaves_the_other(self) -> None:
        bus = FakeNatsClient()
        answers = await _replica(bus)
        bucket = await _bucket(bus)
        await answers.answer("a1", "page", _Computer(), order=1, scope="a")
        await answers.answer("b1", "page", _Computer(), order=1, scope="b")
        assert await answers.retire_older_than(5, scope="a") == 1
        assert (_entry_keys(bucket, "a1"), len(_entry_keys(bucket, "b1"))) == ([], 1)


class TestAWriterReadsTheFloorPastItsL1:
    async def test_an_answer_landing_on_a_version_retired_while_its_l1_missed_the_floor_is_taken_back(self) -> None:
        bus = FakeNatsClient()
        writer, retirer = await _replica(bus), await _replica(bus)
        await writer.answer("v1", "first", _Computer(), order=1)  # the writer's L1 holds the versions row
        writer_registry = writer.registry
        assert writer_registry is not None
        await writer_registry.stop_invalidation_listener()  # the floor's broadcast will not reach its L1
        release = asyncio.Event()
        slow = _Computer(hold=release)
        computing = asyncio.create_task(writer.answer_with_outcome("v1", "second", slow, order=1))
        await slow.started.wait()
        await retirer.retire_older_than(2)
        release.set()
        given = await computing
        assert given.outcome is AnswerOutcome.RETIRED, "the writer kept an answer on a retired version"
        assert _entry_keys(await _bucket(bus), "v1") == []


def _preinitialized(kind: str) -> Any:
    """an L1 backend a host has already initialized with tables of its own, as a pod's registry is."""
    held = MetaData()
    Table("host_rows", held, SaColumn("id", Text, primary_key=True))
    backend: Any = SQLiteBackend(db_name=f"answers-l1-{uuid.uuid4().hex}") if kind == "sqlite" else DuckDBBackend()
    backend.initialize(held)
    return backend


class TestTheTablesAreMadeOnAnyL1:
    @pytest.mark.parametrize("kind", ["sqlite", "duckdb"])
    async def test_an_answer_is_given_and_read_from_an_l1_initialized_before_the_cache(self, kind: str) -> None:
        l1 = _preinitialized(kind)
        answers = await _replica(FakeNatsClient(), l1=l1)
        first = await answers.answer_with_outcome("v1", "page", _Computer("x"), order=1)
        again = await answers.answer_with_outcome("v1", "page", _Computer("x"), order=1)
        assert (first.outcome, again.outcome) == (AnswerOutcome.COMPUTED, AnswerOutcome.READ)
        assert gzip.decompress(again.body) == b"x"
        assert l1.has_table(_TABLE) and l1.has_table(_INDEX) and l1.has_table("host_rows")


class TestASwapNeverCachesARowAPeerReplacedWhileItAdvanced:
    async def test_a_peers_newer_row_landing_during_the_generation_advance_is_not_overwritten_in_l1(self) -> None:
        bus = FakeNatsClient()
        a, b = await _replica(bus), await _replica(bus)
        bucket = await _bucket(bus)
        held = asyncio.Event()
        release = asyncio.Event()
        gated: list[bool] = []
        for name in ("create", "update"):
            original = getattr(bucket, name)

            async def gate(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
                if str(kwargs.get("key")).endswith(f"{_INDEX}-generation") and not gated:
                    gated.append(True)
                    held.set()
                    await release.wait()
                return await _original(*args, **kwargs)

            setattr(bucket, name, gate)
        announcing = asyncio.create_task(a.announce("v1", 1, scope="all"))
        await held.wait()  # A's swap won; it is advancing the table's generation
        await b.announce("v2", 2, scope="all")  # B replaces the row, and its broadcast reaches A
        release.set()
        await announcing
        recorded = await a.announced("all")
        assert recorded is not None and recorded.latest.version == "v2", "A cached its older row over B's newer one"
