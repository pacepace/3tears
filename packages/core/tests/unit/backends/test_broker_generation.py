"""a pod's write generation, advanced by the L3 broker and handed to the collection that wrote.

The contract this pins (epoch-task-06, stage 2):

- a reply that ends a commit names the token the broker's advance wrote for each switched-on table,
  and the one advance of that table for the commit is handed exactly that token, and stamps every
  row broadcast of the commit with their total, however many collection instances wrote them;
- a token belongs to its commit: a later write's reply, naming generations or not, ends it;
- every door that commits carries it: ``l3.query`` (an execute, or a fetch with ``RETURNING``),
  ``l3.batch`` and ``l3.tx.commit``; a rolled-back transaction, or a refused commit, hands out none;
- a reply carrying no generations field at all says the broker advanced nothing (a broker built
  before generations, or a hub not yet holding the table switched on): the pod still writes, its
  advance returns no token and its rows name no generation, with one warning per table; a reply
  that names generations and not the table, or lists it failed, still raises;
- a reply whose write committed and whose advance failed is still a success, so nothing retries
  the write, and the collection's advance raises :class:`GenerationUnavailableError`;
- each task is handed its own commit's token, never another's;
- ``current`` reads the epoch bucket and never mints a generation.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import timedelta
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import Column, DateTime, MetaData, String, Table
from threetears.nats import Subject

from threetears.core.backends import BrokerGenerationSource
from threetears.core.backends.broker_generation import (
    GENERATION_UNAVAILABLE_ERROR_CODE,
    GENERATIONS_FAILED_REPLY_FIELD,
    GENERATIONS_REPLY_FIELD,
)
from threetears.core.backends.nats_proxy import NatsProxyL3Backend
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import (
    WRITE_GENERATION,
    BaseCollection,
    CacheInvalidationMessage,
    CallerTransaction,
    CollectionRegistry,
    tables_with_write_generation,
)
from threetears.core.collections.generation import GenerationMarks
from threetears.core.collections.schema_backed import STRING_TYPE, SchemaBackedCollection, TableSchema
from threetears.core.collections.schema_backed import Column as SchemaColumn
from threetears.core.config import DefaultCoreConfig
from threetears.core.coordination.revocation import RevocationGuard
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import (
    DataLayerUnavailableError,
    GenerationNotCommittedError,
    GenerationUnavailableError,
)
from threetears.core.testing.kv import FakeNatsClient

_TABLE = "group_members"
_TX_ID = "019d9a00-0000-7000-8000-000000000000"


class _Broker:
    """answers each request with the next scripted reply, and records what was asked."""

    def __init__(self, *replies: dict[str, Any]) -> None:
        self.replies = list(replies)
        self.subjects: list[str] = []

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta | None = None) -> bytes:
        del payload, timeout
        self.subjects.append(str(subject))
        return json.dumps(self.replies.pop(0)).encode("utf-8")

    def proxy(self) -> NatsProxyL3Backend:
        nats = MagicMock()
        nats.request_raw = AsyncMock(side_effect=self.request_raw)
        return NatsProxyL3Backend(
            nats_client=nats, namespace_prefix="test", agent_id="agent-123", identity_token=lambda: "token"
        )


def _wrote(token: str, table: str = _TABLE, **extra: Any) -> dict[str, Any]:
    """a successful write reply naming the generation the broker advanced."""
    return {"success": True, "row_count": 1, GENERATIONS_REPLY_FIELD: {table: token}, **extra}


def _advance_failed(table: str = _TABLE) -> dict[str, Any]:
    """a reply whose write committed and whose advance failed: still a success."""
    return {
        "success": True,
        "row_count": 1,
        GENERATIONS_REPLY_FIELD: {},
        GENERATIONS_FAILED_REPLY_FIELD: [table],
        "error_code": GENERATION_UNAVAILABLE_ERROR_CODE,
        "error_message": "epoch bucket unreachable",
    }


class TestEveryDoorThatCommitsHandsOverItsToken:
    async def test_an_execute(self) -> None:
        proxy = _Broker(_wrote("inc:4")).proxy()
        assert await proxy.execute("UPDATE group_members SET member_id = $1", "p") == "UPDATE 1"
        assert await BrokerGenerationSource().advance(_TABLE) == "inc:4"

    async def test_a_fetch_that_writes(self) -> None:
        proxy = _Broker({**_wrote("inc:5"), "rows": [{"id": "m1"}]}).proxy()
        assert await proxy.fetch("INSERT INTO group_members (id) VALUES ($1) RETURNING id", "m1") == [{"id": "m1"}]
        assert await BrokerGenerationSource().advance(_TABLE) == "inc:5"

    async def test_a_batch(self) -> None:
        reply = {"success": True, "results": [{"success": True}], GENERATIONS_REPLY_FIELD: {_TABLE: "inc:6"}}
        proxy = _Broker(reply).proxy()
        await proxy.execute_batch([{"query": "DELETE FROM group_members", "params": []}])
        assert await BrokerGenerationSource().advance(_TABLE) == "inc:6"

    async def test_a_batch_run_statement_by_statement_names_what_committed_even_when_one_failed(self) -> None:
        reply = {
            "success": False,
            "error_code": "QUERY_EXECUTION_ERROR",
            "results": [{"success": True}, {"success": False}],
            GENERATIONS_REPLY_FIELD: {_TABLE: "inc:7"},
        }
        proxy = _Broker(reply).proxy()
        with pytest.raises(DataLayerUnavailableError):
            await proxy.execute_batch([{"query": "DELETE FROM group_members", "params": []}], transaction=False)
        assert await BrokerGenerationSource().advance(_TABLE) == "inc:7"

    async def test_a_transaction_commit(self) -> None:
        broker = _Broker({"success": True, "tx_id": _TX_ID}, {"success": True, "row_count": 1}, _wrote("inc:8"))
        async with broker.proxy().transaction() as conn:
            await conn.execute("UPDATE group_members SET member_id = $1", "p")
            # inside the transaction nothing has committed, so nothing is handed out
            with pytest.raises(GenerationUnavailableError):
                await BrokerGenerationSource().advance(_TABLE)
        assert broker.subjects[-1] == "test.l3.tx.commit"
        assert await BrokerGenerationSource().advance(_TABLE) == "inc:8"


class TestNothingIsHandedOutForWhatDidNotCommit:
    async def test_a_rolled_back_transaction_hands_out_no_earlier_token(self) -> None:
        broker = _Broker(_wrote("inc:1"), {"success": True, "tx_id": _TX_ID}, {"success": True})
        proxy = broker.proxy()
        # a write whose token nobody took, then a transaction that rolls back
        await proxy.execute("UPDATE group_members SET member_id = $1", "p")
        with pytest.raises(RuntimeError):
            async with proxy.transaction():
                raise RuntimeError("the body failed")
        assert broker.subjects[-1] == "test.l3.tx.rollback"
        with pytest.raises(GenerationNotCommittedError, match="rolled back"):
            await BrokerGenerationSource().advance(_TABLE)

    async def test_a_transaction_left_open_in_acquire_hands_out_no_earlier_token(self) -> None:
        # the acquire exit's safety net rolls back a transaction its body never ended
        broker = _Broker(_wrote("inc:1"), {"success": True, "tx_id": _TX_ID}, {"success": True})
        proxy = broker.proxy()
        await proxy.execute("UPDATE group_members SET member_id = $1", "p")
        async with proxy.acquire() as conn:
            await conn.transaction().__aenter__()
        assert broker.subjects[-1] == "test.l3.tx.rollback"
        with pytest.raises(GenerationNotCommittedError):
            await BrokerGenerationSource().advance(_TABLE)

    async def test_a_commit_that_got_no_reply_hands_out_no_earlier_token(self) -> None:
        broker = _Broker(_wrote("inc:1"), {"success": True, "tx_id": _TX_ID})
        proxy = broker.proxy()
        await proxy.execute("UPDATE group_members SET member_id = $1", "p")
        # the third request, the commit, finds no scripted reply: the transport fails
        with pytest.raises(DataLayerUnavailableError):
            async with proxy.transaction():
                pass
        assert broker.subjects[-1] == "test.l3.tx.commit"
        with pytest.raises(GenerationNotCommittedError, match="no reply"):
            await BrokerGenerationSource().advance(_TABLE)

    async def test_a_refused_commit_hands_out_no_earlier_token(self) -> None:
        broker = _Broker(
            _wrote("inc:1"),
            {"success": True, "tx_id": _TX_ID},
            {"success": False, "error_code": "TX_COMMIT_FAILED", "error_message": "refused"},
        )
        proxy = broker.proxy()
        await proxy.execute("UPDATE group_members SET member_id = $1", "p")
        with pytest.raises(DataLayerUnavailableError):
            async with proxy.transaction():
                pass
        with pytest.raises(GenerationNotCommittedError, match="refused"):
            await BrokerGenerationSource().advance(_TABLE)

    async def test_one_commits_token_is_handed_to_one_advance_of_its_table(self) -> None:
        await _Broker(_wrote("inc:2")).proxy().execute("DELETE FROM group_members")
        source = BrokerGenerationSource()
        assert await source.advance(_TABLE) == "inc:2"
        with pytest.raises(GenerationUnavailableError, match="already advanced"):
            await source.advance(_TABLE)

    async def test_a_later_write_whose_reply_names_nothing_ends_the_earlier_commits_token(self) -> None:
        # the first commit's token is never taken; the second commit wrote, and its reply names nothing
        proxy = _Broker(_wrote("inc:2"), {"success": True, "row_count": 1}).proxy()
        await proxy.execute("DELETE FROM group_members")
        await proxy.execute("DELETE FROM group_members")
        assert await BrokerGenerationSource().advance(_TABLE) is None

    async def test_a_later_transaction_commit_naming_nothing_ends_it_too(self) -> None:
        broker = _Broker(_wrote("inc:2"), {"success": True, "tx_id": _TX_ID}, {"success": True})
        proxy = broker.proxy()
        await proxy.execute("DELETE FROM group_members")
        async with proxy.transaction():
            pass
        assert await BrokerGenerationSource().advance(_TABLE) is None

    async def test_a_read_in_between_leaves_the_commits_token(self) -> None:
        proxy = _Broker(_wrote("inc:2"), {"success": True, "rows": []}).proxy()
        await proxy.execute("DELETE FROM group_members")
        await proxy.fetch("SELECT id FROM group_members")
        assert await BrokerGenerationSource().advance(_TABLE) == "inc:2"

    async def test_a_later_commit_of_the_table_replaces_its_token(self) -> None:
        broker = _Broker(_wrote("inc:2"), _wrote("inc:3"))
        proxy = broker.proxy()
        await proxy.execute("DELETE FROM group_members")
        await proxy.execute("DELETE FROM group_members")
        assert await BrokerGenerationSource().advance(_TABLE) == "inc:3"

    async def test_a_table_the_commit_did_not_advance_has_no_token(self) -> None:
        await _Broker(_wrote("inc:2", table="roles")).proxy().execute("DELETE FROM roles")
        with pytest.raises(GenerationUnavailableError, match="group_members") as raised:
            await BrokerGenerationSource().advance(_TABLE)
        # named nothing is not the same as rolled back: the cause it gives is the broker's
        assert not isinstance(raised.value, GenerationNotCommittedError)
        assert await BrokerGenerationSource().advance("roles") == "inc:2"


class TestABrokerBuiltBeforeGenerations:
    async def test_its_reply_still_answers_the_pod(self) -> None:
        proxy = _Broker({"success": True, "rows": [{"id": "m1"}]}, {"success": True, "row_count": 3}).proxy()
        assert await proxy.fetch("SELECT id FROM group_members") == [{"id": "m1"}]
        assert await proxy.execute("DELETE FROM group_members") == "DELETE 3"

    async def test_a_switched_on_advance_after_it_is_told_nothing_was_advanced(self) -> None:
        # owner, 2026-10-08: a pod switched on ahead of its hub still writes; raising would make
        # every hub release before any pod, which is lockstep
        await _Broker({"success": True, "row_count": 1}).proxy().execute("DELETE FROM group_members")
        assert await BrokerGenerationSource().advance(_TABLE) is None

    async def test_it_is_warned_about_once_per_table(self, caplog: pytest.LogCaptureFixture) -> None:
        from threetears.core.backends import broker_generation  # noqa: PLC0415 -- the once-per-table record

        broker_generation._WARNED_UNADVANCED.discard(_TABLE)
        proxy = _Broker({"success": True, "row_count": 1}, {"success": True, "row_count": 1}).proxy()
        with caplog.at_level("WARNING"):
            await proxy.execute("DELETE FROM group_members")
            assert await BrokerGenerationSource().advance(_TABLE) is None
            await proxy.execute("DELETE FROM group_members")
            assert await BrokerGenerationSource().advance(_TABLE) is None
        warned = [r for r in caplog.records if "advanced no write generation" in r.getMessage()]
        assert len(warned) == 1

    async def test_a_broker_that_names_generations_and_not_the_table_still_raises(self) -> None:
        await _Broker({"success": True, "row_count": 1, GENERATIONS_REPLY_FIELD: {}}).proxy().execute(
            "DELETE FROM group_members"
        )
        with pytest.raises(GenerationUnavailableError, match="named no write generation"):
            await BrokerGenerationSource().advance(_TABLE)

    async def test_a_task_that_committed_nothing_still_raises(self) -> None:
        with pytest.raises(GenerationUnavailableError, match="named no write generation"):
            await BrokerGenerationSource().advance(_TABLE)

    async def test_a_switched_on_collections_row_names_no_generation(self) -> None:
        bus = FakeNatsClient()
        members = _pod(_SwitchedOnBrokeredMembers, _Broker({"success": True, "row_count": 1}), bus)
        await members.save_entity(members.create({"id": "m1"}))
        (message,) = _messages(bus)
        assert (message.ids, message.generation, message.bump_rows) == (["m1"], None, None)


class TestAFailedAdvanceAfterACommittedWrite:
    async def test_the_write_is_answered_as_landed_so_nothing_retries_it(self) -> None:
        proxy = _Broker(_advance_failed()).proxy()
        assert await proxy.execute("UPDATE group_members SET member_id = $1", "p") == "UPDATE 1"

    async def test_the_advance_raises(self) -> None:
        await _Broker(_advance_failed()).proxy().execute("UPDATE group_members SET member_id = $1", "p")
        with pytest.raises(GenerationUnavailableError, match="epoch bucket unreachable"):
            await BrokerGenerationSource().advance(_TABLE)

    async def test_a_transaction_commit_and_a_batch_carry_it_too(self) -> None:
        broker = _Broker({"success": True, "tx_id": _TX_ID}, _advance_failed())
        async with broker.proxy().transaction():
            pass
        with pytest.raises(GenerationUnavailableError):
            await BrokerGenerationSource().advance(_TABLE)
        reply = {**_advance_failed(), "results": [{"success": True}]}
        await _Broker(reply).proxy().execute_batch([{"query": "DELETE FROM group_members", "params": []}])
        with pytest.raises(GenerationUnavailableError):
            await BrokerGenerationSource().advance(_TABLE)


class TestEachTaskIsHandedItsOwnCommit:
    async def test_concurrent_writers_of_one_table(self) -> None:
        gate = asyncio.Event()

        async def write(token: str) -> str:
            await _Broker(_wrote(token)).proxy().execute("DELETE FROM group_members")
            await gate.wait()
            return await BrokerGenerationSource().advance(_TABLE)

        first = asyncio.create_task(write("inc:1"))
        second = asyncio.create_task(write("inc:2"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        gate.set()
        assert (await first, await second) == ("inc:1", "inc:2")

    async def test_a_task_started_to_settle_a_commit_takes_its_token_for_the_writer(self) -> None:
        # CallerTransaction settles under asyncio.shield, which runs in a task of its own
        await _Broker(_wrote("inc:3")).proxy().execute("DELETE FROM group_members")
        assert await asyncio.shield(BrokerGenerationSource().advance(_TABLE)) == "inc:3"
        # taken for the writer: the commit's one advance has been made
        with pytest.raises(GenerationUnavailableError, match="already advanced"):
            await BrokerGenerationSource().advance(_TABLE)

    async def test_a_task_that_writes_does_not_add_to_the_record_it_inherited(self) -> None:
        await _Broker(_wrote("inc:3")).proxy().execute("DELETE FROM group_members")

        async def child() -> None:
            # the same table, and a token the child never takes
            await _Broker(_wrote("inc:9")).proxy().execute("DELETE FROM group_members")

        await asyncio.create_task(child())
        # the writer's own token is still its own
        assert await BrokerGenerationSource().advance(_TABLE) == "inc:3"


class _Reader:
    def __init__(self, token: str | None) -> None:
        self.token = token

    async def read(self, table_name: str) -> str | None:
        return self.token


class TestCurrentReadsAndNeverMints:
    async def test_it_reads_the_bucket(self) -> None:
        assert await BrokerGenerationSource(_Reader("inc:9")).current(_TABLE) == "inc:9"

    async def test_a_table_with_no_generation_yet_raises(self) -> None:
        with pytest.raises(GenerationUnavailableError, match="cannot mint"):
            await BrokerGenerationSource(_Reader(None)).current(_TABLE)

    async def test_no_reader_raises(self) -> None:
        with pytest.raises(GenerationUnavailableError):
            await BrokerGenerationSource().current(_TABLE)


# --------------------------------------------------------------------------------------------
# a switched-on collection on a pod, writing through the broker
# --------------------------------------------------------------------------------------------


class _Member(BaseEntity):
    primary_key_field = "id"


class _BrokeredMembers(BaseCollection[_Member]):
    """writes its rows through the L3 proxy, as a pod's collection does."""

    datetime_columns: ClassVar[frozenset[str]] = frozenset({"date_created", "date_updated"})

    def __init__(self, registry: CollectionRegistry, proxy: NatsProxyL3Backend) -> None:
        self._proxy = proxy
        super().__init__(registry, DefaultCoreConfig())

    @property
    def table_name(self) -> str:
        return _TABLE

    @property
    def entity_class(self) -> type[_Member]:
        return _Member

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        return None

    async def save_to_store(self, data: dict[str, Any], original_timestamp: Any = None, *, conn: Any = None) -> int:
        tag = await self._proxy.execute("INSERT INTO group_members (id) VALUES ($1)", data["id"])
        return int(tag.split()[-1])

    async def delete_from_store(self, entity_id: Any) -> None:
        await self._proxy.execute("DELETE FROM group_members WHERE id = $1", entity_id)

    def serialize(self, data: dict[str, Any]) -> bytes:
        return json.dumps(data, default=str).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        row: dict[str, Any] = json.loads(data)
        return row


class _SwitchedOnBrokeredMembers(_BrokeredMembers):
    write_generation = WRITE_GENERATION


def _pod(cls: type[_BrokeredMembers], broker: _Broker, bus: FakeNatsClient) -> _BrokeredMembers:
    metadata = MetaData()
    Table(
        _TABLE,
        metadata,
        Column("id", String(255), primary_key=True),
        Column("date_created", DateTime(timezone=True)),
        Column("date_updated", DateTime(timezone=True)),
    )
    l1 = SQLiteBackend(db_name=f"broker_generation_{uuid.uuid4().hex[:8]}")
    l1.initialize(metadata)
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l2_client=bus, l3_pool=object(), kv_key_scope="pod")  # type: ignore[arg-type]
    registry.set_generation_source(BrokerGenerationSource())
    return cls(registry, broker.proxy())


def _messages(bus: FakeNatsClient) -> list[CacheInvalidationMessage]:
    return [message for message in bus.published if isinstance(message, CacheInvalidationMessage)]


class TestASwitchedOnCollectionOnAPod:
    async def test_its_row_message_names_the_generation_the_broker_wrote(self) -> None:
        bus = FakeNatsClient()
        members = _pod(_SwitchedOnBrokeredMembers, _Broker(_wrote("inc:11")), bus)
        await members.save_entity(members.create({"id": "m1"}))
        (message,) = _messages(bus)
        assert (message.generation, message.bump_rows) == ("inc:11", 1)
        assert members.registry is not None
        assert members.registry.generation_marks.recorded(_TABLE) is None  # followed by nobody here

    async def test_a_failed_advance_raises_after_the_row_was_announced(self) -> None:
        bus = FakeNatsClient()
        members = _pod(_SwitchedOnBrokeredMembers, _Broker(_advance_failed()), bus)
        with pytest.raises(GenerationUnavailableError):
            await members.save_entity(members.create({"id": "m1"}))
        (message,) = _messages(bus)
        assert message.ids == ["m1"]
        assert message.generation is None

    async def test_a_collection_not_switched_on_takes_no_token(self) -> None:
        bus = FakeNatsClient()
        members = _pod(_BrokeredMembers, _Broker(_wrote("inc:12")), bus)
        await members.save_entity(members.create({"id": "m1"}))
        (message,) = _messages(bus)
        assert message.generation is None


class TestTheBrokerKnowsASwitchedOnTableFromItsClass:
    def test_switched_on_and_absence_caching_classes_name_their_tables_and_others_do_not(self) -> None:
        columns = [SchemaColumn("id", STRING_TYPE)]

        class _On(SchemaBackedCollection[_Member]):
            write_generation = WRITE_GENERATION
            schema = TableSchema(name="census_on_table", primary_key=("id",), columns=columns)

        class _Off(SchemaBackedCollection[_Member]):
            schema = TableSchema(name="census_off_table", primary_key=("id",), columns=columns)

        class _Absences(SchemaBackedCollection[_Member]):
            negative_cache_max_age = timedelta(seconds=30)
            schema = TableSchema(name="census_absences_table", primary_key=("id",), columns=columns)

        class _PerInstance(_SwitchedOnBrokeredMembers):
            @property
            def table_name(self) -> str:
                return self._proxy.default_namespace

        tables = tables_with_write_generation()
        assert {"census_on_table", "census_absences_table", _TABLE} <= tables
        assert "census_off_table" not in tables
        del _On, _Off, _Absences, _PerInstance


def _follower_at(token: str) -> GenerationMarks:
    """a follower of the table whose mark stands at ``token``."""
    marks = GenerationMarks()
    marks.follow(_TABLE)
    marks.settle(_TABLE, token)
    return marks


class TestOneCommitIsOneAdvanceWithOneCount:
    """a follower counts an advance complete only once it has heard every row broadcast of it."""

    async def test_two_collections_of_one_table_settling_one_transaction(self) -> None:
        bus = FakeNatsClient()
        first = _pod(_SwitchedOnBrokeredMembers, _Broker(), bus)
        second = _pod(_SwitchedOnBrokeredMembers, _Broker(), bus)
        broker = _Broker({"success": True, "tx_id": _TX_ID}, _wrote("inc:6"))
        async with broker.proxy().acquire() as conn, CallerTransaction(conn) as transaction:
            transaction.enroll(first, "m1")
            transaction.enroll(second, "m2")
        assert broker.subjects[-1] == "test.l3.tx.commit"
        messages = sorted(_messages(bus), key=lambda m: m.ids[0])
        assert [(m.ids[0], m.generation, m.bump_rows) for m in messages] == [("m1", "inc:6", 2), ("m2", "inc:6", 2)]
        follower = _follower_at("inc:5")
        follower.hear(_TABLE, "inc:6", messages[0].bump_rows or 0)
        assert follower.recorded(_TABLE) == "inc:5", "moved on before the other instance's row was heard"
        follower.hear(_TABLE, "inc:6", messages[1].bump_rows or 0)
        assert follower.recorded(_TABLE) == "inc:6"

    async def test_a_bulk_update_settles_its_rows_in_one_advance(self) -> None:
        bus = FakeNatsClient()
        members = _pod(_SwitchedOnBrokeredMembers, _Broker(), bus)
        await _Broker(_wrote("inc:5")).proxy().execute("UPDATE group_members SET n = n + 1 WHERE id = ANY($1)", ["a"])
        await members.invalidate_cache_many(["m1", "m2"])
        assert sorted((m.ids[0], m.generation, m.bump_rows) for m in _messages(bus)) == [
            ("m1", "inc:5", 2),
            ("m2", "inc:5", 2),
        ]

    async def test_a_second_advance_of_one_commit_raises_rather_than_split_its_count(self) -> None:
        bus = FakeNatsClient()
        members = _pod(_SwitchedOnBrokeredMembers, _Broker(), bus)
        await _Broker(_wrote("inc:5")).proxy().execute("UPDATE group_members SET n = n + 1 WHERE id = ANY($1)", ["a"])
        await members.invalidate_cache("m1")
        with pytest.raises(GenerationUnavailableError, match="already advanced"):
            await members.invalidate_cache("m2")
        assert [(m.ids, m.generation) for m in _messages(bus)] == [(["m1"], "inc:5"), (["m2"], None)]


class TestASourceThatCannotReadLeavesAbsenceCachingOff:
    """a pod's reader-less source advances through the broker and caches no absence."""

    @staticmethod
    def _guard(source: BrokerGenerationSource | None) -> Exception | RevocationGuard:
        registry = CollectionRegistry()
        registry.configure(l2_client=FakeNatsClient(), l3_pool=object(), kv_key_scope="pod")  # type: ignore[arg-type]
        if source is not None:
            registry.set_generation_source(source)
        assert (registry.readable_generation_source is not None) == (source is not None and source.reads_generations)
        try:
            return RevocationGuard(registry, purpose="test", ttl_seconds=60)
        except ValueError as exc:
            return exc

    def test_revocations_are_wired_exactly_as_with_no_source(self) -> None:
        without = self._guard(None)
        reader_less = self._guard(BrokerGenerationSource())
        assert isinstance(without, ValueError) and isinstance(reader_less, ValueError)
        assert type(without) is type(reader_less)

    def test_a_source_that_reads_caches_absences(self) -> None:
        assert isinstance(self._guard(BrokerGenerationSource(_Reader("inc:1"))), RevocationGuard)
