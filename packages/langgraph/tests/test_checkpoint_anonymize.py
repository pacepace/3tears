"""anonymizing a thread's stored checkpoints in place, for person erasure.

erasure keeps the conversation: every checkpoint, every pending write, every id and the
text of every message stay. what it changes is each stored field that identifies the
person -- a human message's ``name`` (the chat display name) and the turn metadata keys
in :data:`IDENTIFYING_METADATA_KEYS` -- which take :data:`ANONYMIZED_MARKER`.

the storage here is a real SQL engine (stdlib SQLite, in memory) running the saver's own
statements, and the blobs are written and read by the saver's own serializer, so a graph
built on the saver genuinely stores, reloads and resumes across the rewrite.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Annotated, Any, TypedDict
from uuid import UUID

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from threetears.observe.erasure import ANONYMIZED_MARKER
from threetears.langgraph import (
    IDENTIFYING_METADATA_KEYS,
    KEPT_METADATA_KEYS,
    AsyncQueryExecutor,
    CheckpointAnonymization,
    CheckpointL1Cache,
    CheckpointL2Cache,
    CheckpointL2PrefixCache,
    CheckpointScope,
    ThreeTierCheckpointSaver,
    anonymize_checkpoint_value,
    merge_metadata,
    unclassified_metadata_keys,
)

_UNSCOPED = CheckpointScope.unscoped(reason="tests drive a single-tenant saver")
_THREAD = "thread-erase"
_OTHER_THREAD = "thread-keep"
_NAME = "Alice Liddell"
_EXTERNAL_ID = "U0ALICE"
_CONTENT = "please summarise the Q3 numbers for the board"

#: the checkpoint tables as the saver's v001 migration declares them, in SQLite's dialect.
_DDL = (
    "CREATE TABLE checkpoints (thread_id VARCHAR(255) NOT NULL, checkpoint_ns VARCHAR(255) NOT NULL DEFAULT '', "
    "checkpoint_id VARCHAR(255) NOT NULL, parent_checkpoint_id VARCHAR(255), type VARCHAR(255), "
    "checkpoint BLOB NOT NULL, metadata_ BLOB, PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id))",
    "CREATE TABLE checkpoint_writes (thread_id VARCHAR(255) NOT NULL, checkpoint_ns VARCHAR(255) NOT NULL "
    "DEFAULT '', checkpoint_id VARCHAR(255) NOT NULL, task_id VARCHAR(255) NOT NULL, task_path VARCHAR(255) "
    "NOT NULL DEFAULT '', idx INTEGER NOT NULL, channel VARCHAR(255) NOT NULL, type VARCHAR(255), blob BLOB "
    "NOT NULL, PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx))",
)

_PLACEHOLDER = re.compile(r"\$(\d+)")


class SqliteQueryExecutor(AsyncQueryExecutor):
    """the saver's executor protocol over an in-memory SQLite database.

    ``$n`` placeholders become SQLite's ``?n``; everything else is the saver's own SQL,
    run by a real engine, so a statement the saver gets wrong fails here as it would on
    a server.
    """

    def __init__(self) -> None:
        """create the database and the two checkpoint tables.

        :return: nothing
        :rtype: None
        """
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        for statement in _DDL:
            self.db.execute(statement)
        self.statements: list[str] = []

    def _run(self, query: str, args: tuple[object, ...]) -> sqlite3.Cursor:
        """run one statement.

        :param query: the saver's SQL
        :ptype query: str
        :param args: its parameters
        :ptype args: tuple[object, ...]
        :return: the cursor
        :rtype: sqlite3.Cursor
        """
        self.statements.append(query)
        return self.db.execute(_PLACEHOLDER.sub(r"?\1", query), args)

    async def fetch(self, query: str, *args: object) -> list[dict[str, Any]]:
        """
        :param query: SQL
        :ptype query: str
        :param args: parameters
        :ptype args: object
        :return: every row, as dicts
        :rtype: list[dict[str, Any]]
        """
        return [dict(row) for row in self._run(query, args).fetchall()]

    async def fetchrow(self, query: str, *args: object) -> dict[str, Any] | None:
        """
        :param query: SQL
        :ptype query: str
        :param args: parameters
        :ptype args: object
        :return: the first row, or None
        :rtype: dict[str, Any] | None
        """
        row = self._run(query, args).fetchone()
        return dict(row) if row is not None else None

    async def execute(self, query: str, *args: object) -> str:
        """
        :param query: SQL
        :ptype query: str
        :param args: parameters
        :ptype args: object
        :return: an asyncpg-shaped status tag
        :rtype: str
        """
        cursor = self._run(query, args)
        self.db.commit()
        return f"UPDATE {cursor.rowcount}"

    def blobs(self) -> list[bytes]:
        """every stored checkpoint, metadata and pending-write blob, for a leak scan.

        :return: the raw blobs
        :rtype: list[bytes]
        """
        found: list[bytes] = []
        for row in self.db.execute("SELECT checkpoint, metadata_ FROM checkpoints"):
            found.extend(bytes(value) for value in row if value is not None)
        found.extend(bytes(row[0]) for row in self.db.execute("SELECT blob FROM checkpoint_writes"))
        return found


class DictL1Cache(CheckpointL1Cache):
    """a pod-local L1 over a dict, keyed as the protocol keys it."""

    def __init__(self) -> None:
        """start empty.

        :return: nothing
        :rtype: None
        """
        self.store: dict[tuple[str, str], bytes] = {}

    async def get(self, thread_id: str, checkpoint_ns: str) -> bytes | None:
        """
        :param thread_id: thread
        :ptype thread_id: str
        :param checkpoint_ns: namespace
        :ptype checkpoint_ns: str
        :return: the cached bundle, or None
        :rtype: bytes | None
        """
        return self.store.get((thread_id, checkpoint_ns))

    async def put(self, thread_id: str, checkpoint_ns: str, data: bytes) -> None:
        """
        :param thread_id: thread
        :ptype thread_id: str
        :param checkpoint_ns: namespace
        :ptype checkpoint_ns: str
        :param data: bundle
        :ptype data: bytes
        """
        self.store[(thread_id, checkpoint_ns)] = data

    async def delete(self, thread_id: str) -> None:
        """
        :param thread_id: thread whose every namespace is dropped
        :ptype thread_id: str
        """
        for key in [key for key in self.store if key[0] == thread_id]:
            del self.store[key]


class SweepingL2Cache(CheckpointL2Cache, CheckpointL2PrefixCache):
    """a shared L2 over a dict, with the optional prefix sweep."""

    def __init__(self) -> None:
        """start empty.

        :return: nothing
        :rtype: None
        """
        self.store: dict[tuple[str, str], bytes] = {}

    async def get(self, bucket: str, key: str) -> bytes | None:
        """
        :param bucket: bucket
        :ptype bucket: str
        :param key: key
        :ptype key: str
        :return: value, or None
        :rtype: bytes | None
        """
        return self.store.get((bucket, key))

    async def put(self, bucket: str, key: str, value: bytes) -> None:
        """
        :param bucket: bucket
        :ptype bucket: str
        :param key: key
        :ptype key: str
        :param value: value
        :ptype value: bytes
        """
        self.store[(bucket, key)] = value

    async def delete(self, bucket: str, key: str) -> None:
        """
        :param bucket: bucket
        :ptype bucket: str
        :param key: key
        :ptype key: str
        """
        self.store.pop((bucket, key), None)

    async def delete_prefix(self, bucket: str, prefix: str) -> None:
        """
        :param bucket: bucket
        :ptype bucket: str
        :param prefix: key prefix
        :ptype prefix: str
        """
        for key in [key for key in self.store if key[0] == bucket and key[1].startswith(prefix)]:
            del self.store[key]


class ExactKeyL2Cache(CheckpointL2Cache):
    """a shared L2 with only exact-key deletes, the shape most adapters have."""

    def __init__(self) -> None:
        """start empty.

        :return: nothing
        :rtype: None
        """
        self.store: dict[tuple[str, str], bytes] = {}

    async def get(self, bucket: str, key: str) -> bytes | None:
        """
        :param bucket: bucket
        :ptype bucket: str
        :param key: key
        :ptype key: str
        :return: value, or None
        :rtype: bytes | None
        """
        return self.store.get((bucket, key))

    async def put(self, bucket: str, key: str, value: bytes) -> None:
        """
        :param bucket: bucket
        :ptype bucket: str
        :param key: key
        :ptype key: str
        :param value: value
        :ptype value: bytes
        """
        self.store[(bucket, key)] = value

    async def delete(self, bucket: str, key: str) -> None:
        """
        :param bucket: bucket
        :ptype bucket: str
        :param key: key
        :ptype key: str
        """
        self.store.pop((bucket, key), None)


class ChatState(TypedDict):
    """the shape an agent graph's state has: messages plus the merged metadata channel."""

    messages: Annotated[list[AnyMessage], add_messages]
    metadata: Annotated[dict[str, Any], merge_metadata]


def _reply(state: ChatState) -> dict[str, Any]:
    """
    answer the last message, reading its text.

    :param state: graph state
    :ptype state: ChatState
    :return: the reply update
    :rtype: dict[str, Any]
    """
    return {"messages": [AIMessage(content=f"noted: {state['messages'][-1].content}")]}


def _approve(state: ChatState) -> dict[str, Any]:
    """
    pause for a human decision, then record it.

    :param state: graph state
    :ptype state: ChatState
    :return: the decision update
    :rtype: dict[str, Any]
    """
    decision = interrupt({"question": "send it?", "channel": state["metadata"].get("channel_ref")})
    return {"messages": [AIMessage(content=f"decision: {decision}")]}


def _graph(saver: ThreeTierCheckpointSaver, *, pause: bool = False) -> Any:
    """
    compile a two-node chat graph on the saver.

    :param saver: the checkpoint saver
    :ptype saver: ThreeTierCheckpointSaver
    :param pause: whether the second node interrupts for approval
    :ptype pause: bool
    :return: the compiled graph
    :rtype: Any
    """
    builder = StateGraph(ChatState)
    builder.add_node("reply", _reply)
    builder.add_edge(START, "reply")
    if pause:
        builder.add_node("approve", _approve)
        builder.add_edge("reply", "approve")
        builder.add_edge("approve", END)
    else:
        builder.add_edge("reply", END)
    return builder.compile(checkpointer=saver)


def _turn(content: str = _CONTENT) -> dict[str, Any]:
    """
    one channel turn as the SDK hands it to the graph.

    :param content: the message text
    :ptype content: str
    :return: the graph input
    :rtype: dict[str, Any]
    """
    return {
        "messages": [HumanMessage(content=content, name=_NAME)],
        "metadata": {
            "external_user_name": _NAME,
            "external_user_id": _EXTERNAL_ID,
            "channel_ref": "C0CHANNEL",
            "workspace_ref": "T0WORKSPACE",
            "surfaced_memory_ids": ["m-1", "m-2"],
        },
    }


def _config(thread: str = _THREAD) -> dict[str, Any]:
    """
    :param thread: thread id
    :ptype thread: str
    :return: the run config for it
    :rtype: dict[str, Any]
    """
    return {"configurable": {"thread_id": thread}}


def _leaks(executor: SqliteQueryExecutor) -> bool:
    """
    whether any stored blob still carries the person's name or external id.

    :param executor: the database
    :ptype executor: SqliteQueryExecutor
    :return: ``True`` when an identifying value survives anywhere in storage
    :rtype: bool
    """
    return any(_NAME.encode() in blob or _EXTERNAL_ID.encode() in blob for blob in executor.blobs())


@pytest.fixture()
def executor() -> SqliteQueryExecutor:
    """
    :return: a fresh database
    :rtype: SqliteQueryExecutor
    """
    return SqliteQueryExecutor()


class TestTheRule:
    """which fields identify the person: named in one place, applied everywhere a value can sit."""

    def test_the_identifying_metadata_keys_are_the_sender_identity_keys(self) -> None:
        """the channel router's two sender keys; channel and workspace references are not a person."""
        assert IDENTIFYING_METADATA_KEYS == frozenset({"external_user_name", "external_user_id"})

    def test_a_human_messages_name_is_anonymized_and_nothing_else_about_it(self) -> None:
        """content, id and type survive; only the display name changes."""
        message = HumanMessage(content=_CONTENT, name=_NAME, id="msg-1")

        (rewritten,) = anonymize_checkpoint_value([message])

        assert rewritten.name == ANONYMIZED_MARKER
        assert rewritten.content == _CONTENT
        assert rewritten.id == "msg-1"
        assert type(rewritten) is HumanMessage

    def test_an_ai_messages_name_is_the_agent_not_a_person(self) -> None:
        """only a human message carries the sender's name."""
        message = AIMessage(content="hi", name="support-agent", id="msg-2")

        assert anonymize_checkpoint_value(message) is message

    def test_identifying_keys_are_found_at_any_depth(self) -> None:
        """the metadata channel, the ``__start__`` input and a pending write all nest it differently."""
        value = {"__start__": {"metadata": {"external_user_id": _EXTERNAL_ID, "channel_ref": "C1"}}}

        assert anonymize_checkpoint_value(value) == {
            "__start__": {"metadata": {"external_user_id": ANONYMIZED_MARKER, "channel_ref": "C1"}}
        }

    def test_an_unknown_metadata_key_is_kept(self) -> None:
        """the metadata channel is working state the graph reads on resume; only named identity is changed."""
        value = {"surfaced_memory_ids": ["m-1"], "governed_knowledge_block": "text"}

        assert anonymize_checkpoint_value(value) == value

    def test_every_key_the_turn_fixture_carries_is_classified(self) -> None:
        """the fixture's turn metadata is fully classified, so a report from it means a new key."""
        assert set(_turn()["metadata"]) <= IDENTIFYING_METADATA_KEYS | KEPT_METADATA_KEYS
        assert not IDENTIFYING_METADATA_KEYS & KEPT_METADATA_KEYS, "a key cannot be both identifying and kept"

    def test_an_unclassified_turn_metadata_key_is_named_wherever_the_metadata_sits(self) -> None:
        """the metadata channel, the ``__start__`` input and a checkpoint's writes all nest it differently."""
        metadata = {"external_user_id": _EXTERNAL_ID, "channel_ref": "C1", "sender_email": "a@example.com"}
        checkpoint = {"channel_values": {"metadata": metadata, "messages": []}}
        start_input = {"__start__": {"metadata": metadata}}

        assert unclassified_metadata_keys(checkpoint) == {"sender_email"}
        assert unclassified_metadata_keys(start_input) == {"sender_email"}
        assert unclassified_metadata_keys(metadata, is_metadata=True) == {"sender_email"}

    def test_keys_outside_turn_metadata_are_not_reported(self) -> None:
        """a graph's own state keys, and keys nested under a classified metadata key, are not turn metadata."""
        value = {
            "channel_values": {
                "messages": [HumanMessage(content="x", name=_NAME)],
                "summary": {"anything": 1},
                "metadata": {"knowledge_injected_entries": [{"scope": "customer"}], "channel_ref": "C1"},
            }
        }

        assert unclassified_metadata_keys(value) == frozenset()
        assert unclassified_metadata_keys({"sender_email": "a@example.com"}) == frozenset()

    def test_none_stays_none(self) -> None:
        """a message with no name, and a metadata key holding None, carry nothing to anonymize."""
        message = HumanMessage(content="x")

        assert anonymize_checkpoint_value(message) is message
        assert anonymize_checkpoint_value({"external_user_name": None}) == {"external_user_name": None}

    def test_the_rule_is_idempotent(self) -> None:
        """a second pass changes nothing and returns the very objects it was given."""
        once = anonymize_checkpoint_value(
            {"messages": [HumanMessage(content="x", name=_NAME)], "external_user_id": "U"}
        )

        twice = anonymize_checkpoint_value(once)

        assert twice == once


class TestAnonymizingAThread:
    """the saver rewrites every stored blob of the named threads, and only theirs."""

    async def test_no_identifying_value_survives_in_storage(self, executor: SqliteQueryExecutor) -> None:
        """a byte scan of every checkpoint, metadata and write blob finds neither name nor external id."""
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED)
        await _graph(saver).ainvoke(_turn(), _config())
        assert _leaks(executor)

        result = await saver.aanonymize_threads([_THREAD])

        assert not _leaks(executor)
        assert isinstance(result, CheckpointAnonymization)
        assert result.threads == 1
        assert result.checkpoints_rewritten > 0
        assert result.writes_rewritten > 0
        assert result.l2_prefix_swept is None, "no L2, so there was nothing to sweep"

    async def test_a_fully_classified_thread_reports_no_unclassified_key(self, executor: SqliteQueryExecutor) -> None:
        """a clean result must be earned: every key the turn carried is classified."""
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED)
        await _graph(saver).ainvoke(_turn(), _config())

        result = await saver.aanonymize_threads([_THREAD])

        assert result.unclassified_metadata_keys == ()

    async def test_a_key_no_rule_classifies_is_reported_and_its_value_kept(self, executor: SqliteQueryExecutor) -> None:
        """a producer that starts sending a new sender field is told at erasure time, not never.

        the value is kept -- the rule changes only what it names -- so the report is the only
        thing standing between the operator and a false "complete".
        """
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED)
        turn = _turn()
        turn["metadata"] = {**turn["metadata"], "sender_email": "alice@example.com"}
        await _graph(saver).ainvoke(turn, _config())

        result = await saver.aanonymize_threads([_THREAD])

        assert result.unclassified_metadata_keys == ("sender_email",)
        assert not result.unreadable
        assert any(b"alice@example.com" in blob for blob in executor.blobs()), "an unclassified value is kept"
        assert not _leaks(executor), "the classified identifying keys are still anonymized"

    async def test_the_graph_loads_with_content_and_ids_intact(self, executor: SqliteQueryExecutor) -> None:
        """the state reloads through the real serializer: text and ids unchanged, identity masked."""
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED)
        graph = _graph(saver)
        await graph.ainvoke(_turn(), _config())
        before = await graph.aget_state(_config())
        checkpoint_ids_before = [row[0] for row in executor.db.execute("SELECT checkpoint_id FROM checkpoints")]

        await saver.aanonymize_threads([_THREAD])
        after = await graph.aget_state(_config())

        human_before, ai_before = before.values["messages"]
        human_after, ai_after = after.values["messages"]
        assert human_after.name == ANONYMIZED_MARKER
        assert human_after.content == human_before.content == _CONTENT
        assert human_after.id == human_before.id
        assert ai_after == ai_before
        assert after.values["metadata"] == {
            **before.values["metadata"],
            "external_user_name": ANONYMIZED_MARKER,
            "external_user_id": ANONYMIZED_MARKER,
        }
        assert after.config["configurable"]["checkpoint_id"] == before.config["configurable"]["checkpoint_id"]
        assert [row[0] for row in executor.db.execute("SELECT checkpoint_id FROM checkpoints")] == checkpoint_ids_before

    async def test_the_graph_continues_after_anonymization(self, executor: SqliteQueryExecutor) -> None:
        """a later turn appends to the anonymized history and checkpoints normally."""
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED)
        graph = _graph(saver)
        await graph.ainvoke(_turn(), _config())
        await saver.aanonymize_threads([_THREAD])

        state = await graph.ainvoke({"messages": [HumanMessage(content="and Q4?")]}, _config())

        assert [message.content for message in state["messages"]] == [
            _CONTENT,
            f"noted: {_CONTENT}",
            "and Q4?",
            "noted: and Q4?",
        ]
        assert state["messages"][0].name == ANONYMIZED_MARKER

    async def test_an_interrupted_graph_resumes_after_anonymization(self, executor: SqliteQueryExecutor) -> None:
        """pending writes (the interrupt among them) are rewritten and still drive the resume."""
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED)
        graph = _graph(saver, pause=True)
        await graph.ainvoke(_turn(), _config())
        assert (await graph.aget_state(_config())).interrupts

        await saver.aanonymize_threads([_THREAD])
        assert not _leaks(executor)
        state = await graph.ainvoke(Command(resume="approve"), _config())

        assert state["messages"][-1].content == "decision: approve"

    async def test_a_second_run_rewrites_nothing(self, executor: SqliteQueryExecutor) -> None:
        """idempotent: the stored bytes after two runs equal those after one, and the second reports no rewrites."""
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED)
        await _graph(saver).ainvoke(_turn(), _config())
        await saver.aanonymize_threads([_THREAD])
        after_one = sorted(executor.blobs())

        second = await saver.aanonymize_threads([_THREAD])

        assert sorted(executor.blobs()) == after_one
        assert (second.checkpoints_rewritten, second.writes_rewritten) == (0, 0)

    async def test_other_threads_are_untouched(self, executor: SqliteQueryExecutor) -> None:
        """only the named threads are rewritten."""
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED)
        graph = _graph(saver)
        await graph.ainvoke(_turn(), _config())
        await graph.ainvoke(_turn(), _config(_OTHER_THREAD))

        await saver.aanonymize_threads([_THREAD])

        kept = await graph.aget_state(_config(_OTHER_THREAD))
        assert kept.values["messages"][0].name == _NAME
        assert kept.values["metadata"]["external_user_id"] == _EXTERNAL_ID

    async def test_small_batches_reach_every_row(self, executor: SqliteQueryExecutor) -> None:
        """paging by a batch smaller than the thread still rewrites every checkpoint and write."""
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED)
        graph = _graph(saver)
        for content in ("one", "two", "three"):
            await graph.ainvoke(_turn(content), _config())

        await saver.aanonymize_threads([_THREAD], batch_size=1)

        assert not _leaks(executor)

    async def test_a_customer_scoped_saver_rewrites_its_own_keyspace(self, executor: SqliteQueryExecutor) -> None:
        """the customer prefix is folded into the key exactly as every other path folds it."""
        customer = UUID("11111111-1111-1111-1111-111111111111")
        saver = ThreeTierCheckpointSaver(executor, scope=CheckpointScope.for_customer(customer))
        await _graph(saver).ainvoke(_turn(), _config())

        await saver.aanonymize_threads([_THREAD])

        assert not _leaks(executor)

    async def test_a_bare_string_is_refused_before_any_statement(self, executor: SqliteQueryExecutor) -> None:
        """``aanonymize_threads("t-1")`` would iterate characters and rewrite nothing the caller meant."""
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED)

        with pytest.raises(TypeError, match="bare string"):
            await saver.aanonymize_threads(_THREAD)

        assert executor.statements == []

    @pytest.mark.parametrize("batch_size", [0, -1])
    async def test_a_nonsensical_batch_is_refused(self, executor: SqliteQueryExecutor, batch_size: int) -> None:
        """a batch of nothing would page forever.

        :param batch_size: the refused size
        :ptype batch_size: int
        """
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED)

        with pytest.raises(ValueError, match="batch_size"):
            await saver.aanonymize_threads([_THREAD], batch_size=batch_size)


class TestTheCachesAreEvicted:
    """a rewritten thread must not keep answering from a cached bundle that still names the person."""

    async def test_l1_and_l2_no_longer_serve_the_old_bundle(self, executor: SqliteQueryExecutor) -> None:
        """after the rewrite, the thread's cached bundles are gone and a read serves the anonymized state."""
        l1 = DictL1Cache()
        l2 = SweepingL2Cache()
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED, l1_cache=l1, l2_cache=l2)
        graph = _graph(saver)
        await graph.ainvoke(_turn(), _config())
        await graph.aget_state(_config())
        assert any(key[0] == _THREAD for key in l1.store)
        assert any(key[1] == _THREAD for key in l2.store)

        result = await saver.aanonymize_threads([_THREAD])

        assert not any(key[0] == _THREAD for key in l1.store)
        assert not any(key[1].startswith(_THREAD) for key in l2.store)
        assert result.l2_prefix_swept is True
        reread = await graph.aget_state(_config())
        assert reread.values["messages"][0].name == ANONYMIZED_MARKER

    async def test_an_l2_that_cannot_sweep_evicts_the_root_key_and_says_so(self, executor: SqliteQueryExecutor) -> None:
        """exact-key only: the root bundle goes, and the result reports the sweep did not happen."""
        l2 = ExactKeyL2Cache()
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED, l2_cache=l2)
        graph = _graph(saver)
        await graph.ainvoke(_turn(), _config())
        await graph.aget_state(_config())
        assert any(key[1] == _THREAD for key in l2.store)

        result = await saver.aanonymize_threads([_THREAD])

        assert not any(key[1] == _THREAD for key in l2.store)
        assert result.l2_prefix_swept is False

    async def test_a_failed_l2_eviction_fails_the_erasure(self, executor: SqliteQueryExecutor) -> None:
        """a shared cache still serving the name is the failure erasure exists to prevent, so it raises."""

        class BrokenL2(SweepingL2Cache):
            """an L2 whose delete fails the way a NATS timeout does."""

            async def delete(self, bucket: str, key: str) -> None:
                """
                :param bucket: bucket
                :ptype bucket: str
                :param key: key
                :ptype key: str
                :raises RuntimeError: always
                """
                raise RuntimeError("nats: timeout")

        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED, l2_cache=BrokenL2())
        await _graph(saver).ainvoke(_turn(), _config())

        with pytest.raises(RuntimeError, match="nats: timeout"):
            await saver.aanonymize_threads([_THREAD])


class TestAFailedErasureCanBeFound:
    """an erasure over many threads that fails must say where, and a row that can never be
    rewritten must not stop every row after it."""

    async def test_an_undecodable_blob_is_reported_and_every_other_row_still_rewritten(
        self, executor: SqliteQueryExecutor, caplog: pytest.LogCaptureFixture
    ) -> None:
        """a blob that fails to decode fails on every run: failing the run on it stopped the erasure at
        that row forever, and nothing named which thread held it."""
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED)
        await _graph(saver).ainvoke(_turn(), _config())
        await _graph(saver).ainvoke(_turn(), _config(_OTHER_THREAD))
        bad = executor.db.execute(
            "SELECT checkpoint_ns, checkpoint_id, task_id, idx FROM checkpoint_writes WHERE thread_id = ? "
            "ORDER BY checkpoint_ns, checkpoint_id, task_id, idx LIMIT 1",
            (_THREAD,),
        ).fetchone()
        # 0xc1 is never valid msgpack; the person's name still sits in the bytes after it
        executor.db.execute(
            "UPDATE checkpoint_writes SET blob = ? WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? "
            "AND task_id = ? AND idx = ?",
            (b"\xc1" + _NAME.encode(), _THREAD, *bad),
        )

        with caplog.at_level("ERROR", logger="threetears.langgraph.checkpoint"):
            result = await saver.aanonymize_threads([_THREAD, _OTHER_THREAD])

        [reported] = result.unreadable
        assert (reported.thread_id, reported.table, reported.column) == (_THREAD, "checkpoint_writes", "blob")
        assert (reported.checkpoint_ns, reported.checkpoint_id, reported.task_id, reported.idx) == tuple(bad)
        assert result.threads == 2 and result.checkpoints_rewritten > 0, "every other row is still rewritten"
        survivors = [blob for blob in executor.blobs() if _NAME.encode() in blob or _EXTERNAL_ID.encode() in blob]
        assert survivors == [b"\xc1" + _NAME.encode()], "only the unreadable blob still holds the name"
        [logged] = [r for r in caplog.records if "cannot be anonymized" in r.getMessage()]
        assert logged.__dict__["thread_id"] == _THREAD
        assert logged.__dict__["task_id"] == bad["task_id"] and logged.__dict__["idx"] == bad["idx"]

        again = await saver.aanonymize_threads([_THREAD])
        assert again.unreadable == (reported,), "a rerun names the same row, and only it"
        assert (again.checkpoints_rewritten, again.writes_rewritten) == (0, 0)

    async def test_a_failed_write_names_its_thread_and_stage_and_raises(
        self, executor: SqliteQueryExecutor, caplog: pytest.LogCaptureFixture
    ) -> None:
        """a failure a rerun can cure still raises -- with the thread and stage an operator needs."""
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED)
        await _graph(saver).ainvoke(_turn(), _config())

        async def refuse(query: str, *args: object) -> str:
            raise sqlite3.OperationalError("database is locked")

        executor.execute = refuse  # type: ignore[method-assign]

        with caplog.at_level("ERROR", logger="threetears.langgraph.checkpoint"):
            with pytest.raises(sqlite3.OperationalError) as raised:
                await saver.aanonymize_threads([_THREAD])

        notes = "\n".join(getattr(raised.value, "__notes__", []))
        assert repr(_THREAD) in notes and "(checkpoints)" in notes
        assert "writing checkpoint ns=" in notes
        [logged] = [r for r in caplog.records if "anonymization failed" in r.getMessage()]
        assert (logged.__dict__["thread_id"], logged.__dict__["stage"]) == (_THREAD, "checkpoints")
