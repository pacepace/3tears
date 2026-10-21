"""unit tests for intention tool input schemas + the tools' soft-fail edges.

Pins the Pydantic tool schemas (what the LLM may pass), and drives the
``intention_log`` / ``intention_mark_surfaced`` tools over an in-memory
collection stub to pin the error-string format, the embedding soft-fail and
the best-effort event dispatch -- without standing up a database.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from prometheus_client import REGISTRY
from pydantic import ValidationError

from threetears.agent.acl import AclCache
from threetears.core.backends import RowcountUnreadableError
from threetears.core.exceptions import (
    ConcurrentModificationError,
    DataLayerUnavailableError,
    GenerationUnavailableError,
)
from threetears.agent.intention.authorize import IntentionAuthorizerDependencies
from threetears.agent.intention.entities import IntentionEntity
from threetears.agent.intention.events import IntentionSurfacedEvent
from threetears.agent.intention.tools import (
    IntentionListInput,
    IntentionLogInput,
    IntentionMarkSurfacedInput,
    load_intention_list_tool,
    load_intention_log_tool,
    load_intention_mark_surfaced_tool,
)


class TestIntentionLogInput:
    def test_content_only(self) -> None:
        inp = IntentionLogInput(content="learn more about the user's work")
        assert inp.content == "learn more about the user's work"
        assert inp.source_memory_id is None

    def test_content_required(self) -> None:
        with pytest.raises(ValidationError):
            IntentionLogInput()  # type: ignore[call-arg]

    def test_optional_source_memory(self) -> None:
        inp = IntentionLogInput(content="x", source_memory_id="abc-123")
        assert inp.source_memory_id == "abc-123"


class TestIntentionListInput:
    def test_default_limit(self) -> None:
        assert IntentionListInput().limit == 20

    def test_limit_bounds(self) -> None:
        with pytest.raises(ValidationError):
            IntentionListInput(limit=0)
        with pytest.raises(ValidationError):
            IntentionListInput(limit=101)


class TestIntentionMarkSurfacedInput:
    def test_fields_required(self) -> None:
        with pytest.raises(ValidationError):
            IntentionMarkSurfacedInput(intention_id="i-1")  # type: ignore[call-arg]

    def test_valid(self) -> None:
        inp = IntentionMarkSurfacedInput(intention_id="i-1", new_status="asked")
        assert inp.intention_id == "i-1"
        assert inp.new_status == "asked"


class _EmptyMembershipLoader:
    """membership loader returning no memberships; the owner path never needs one."""

    async def load_for_user(self, user_id: UUID) -> tuple[Any, ...]:
        _ = user_id
        return ()

    async def load_for_agent(self, agent_id: UUID) -> tuple[Any, ...]:
        _ = agent_id
        return ()

    async def load_for_group(self, group_id: UUID) -> tuple[Any, ...]:
        _ = group_id
        return ()


class _EmptyGrantLoader:
    """grant loader returning no assignments / roles / groups."""

    async def load_assignments_for_groups(self, group_ids: tuple[UUID, ...], namespace: Any) -> tuple[Any, ...]:
        _ = group_ids, namespace
        return ()

    async def load_roles(self, role_ids: tuple[UUID, ...]) -> dict[UUID, Any]:
        _ = role_ids
        return {}

    async def load_groups(self, group_ids: tuple[UUID, ...]) -> dict[UUID, Any]:
        _ = group_ids
        return {}


def _authorizer() -> IntentionAuthorizerDependencies:
    return IntentionAuthorizerDependencies(
        acl_cache=AclCache(membership_loader=_EmptyMembershipLoader(), grant_loader=_EmptyGrantLoader()),
    )


# parity-exempt: answers only the four calls the intention tools make (get, save_entity, create, find_similar_for_dedup); the full IntentionsCollection surface is L3-backed and exercised by tests/integration/test_intention_tools.py
class _FakeIntentions:
    """in-memory stand-in for the slice of :class:`IntentionsCollection` the tools call."""

    def __init__(self, entity: IntentionEntity | None = None) -> None:
        self.entity = entity
        self.dedup_embeddings: list[list[float]] = []
        self.saved: list[IntentionEntity] = []

    async def get(self, key: tuple[UUID, UUID]) -> IntentionEntity | None:
        _ = key
        return self.entity

    async def save_entity(self, entity: IntentionEntity) -> None:
        self.saved.append(entity)

    def create(self, data: dict[str, Any]) -> IntentionEntity:
        return IntentionEntity(data, is_new=True)

    async def find_similar_for_dedup(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.dedup_embeddings.append(kwargs["embedding"])
        return []


class _FakeIntentionsDedupDown(_FakeIntentions):
    """the dedup lookup fails, as an L3 timeout does."""

    async def find_similar_for_dedup(self, **kwargs: Any) -> list[dict[str, Any]]:
        _ = kwargs
        raise ConnectionError("l3 timeout")


class _FakeIntentionsStoreDown(_FakeIntentions):
    """the save fails with a driver message that quotes the SQL, as asyncpg's do."""

    async def save_entity(self, entity: IntentionEntity) -> None:
        _ = entity
        raise ConnectionError('relation "intentions" ... INSERT INTO intentions VALUES ($1, $2) at postgres://u:pw@db')


class _RaisingEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    async def aembed_query(self, text: str) -> list[float]:
        self.calls += 1
        raise RuntimeError("upstream embedding outage")


class _OkEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    async def aembed_query(self, text: str) -> list[float]:
        _ = text
        self.calls += 1
        return [0.1, 0.2, 0.3]


async def _log(embedder: Any, collection: _FakeIntentions, content: str) -> str:
    agent_id = uuid4()
    (tool,) = await load_intention_log_tool(
        user_id=uuid4(),
        embedding_provider=embedder,
        agent_id=agent_id,
        customer_id=uuid4(),
        authorizer=_authorizer(),
        intentions_collection=collection,  # type: ignore[arg-type]
    )
    result = await tool.ainvoke({"content": content})
    assert isinstance(result, str)
    return result


class TestToolError:
    async def test_format(self) -> None:
        """a tool failure reads ``[TOOL ERROR] <tool>: <action> failed — <reason>``."""
        msg = await _log(_RaisingEmbedder(), _FakeIntentions(), "want")
        assert msg.startswith("[TOOL ERROR] intention_log: embed failed")
        assert "the embedder failed (RuntimeError)" in msg
        assert "upstream embedding outage" not in msg

    async def test_an_embedding_outage_is_counted_and_its_provider_text_never_reaches_the_reply(self) -> None:
        class _KeyLeaking(_RaisingEmbedder):
            async def aembed_query(self, text: str) -> list[float]:
                raise ValueError("Incorrect API key provided: sk-proj-FAKE1234567890")

        site = "threetears.agent.intention.tools:load_intention_log_tool.<locals>.intention_log"
        labels = {"site": site, "exc_type": "EmbeddingUnavailableError"}
        before = REGISTRY.get_sample_value("contained_errors_total", labels) or 0.0
        msg = await _log(_KeyLeaking(), _FakeIntentions(), "want")
        assert "sk-proj" not in msg
        assert (REGISTRY.get_sample_value("contained_errors_total", labels) or 0.0) == before + 1


class TestEmbeddingSoftFail:
    async def test_empty_text_never_reaches_the_embedder(self) -> None:
        embedder = _OkEmbedder()
        msg = await _log(embedder, _FakeIntentions(), "   ")
        assert msg.startswith("[TOOL ERROR] intention_log: input failed")
        assert embedder.calls == 0

    async def test_failure_soft_fails_to_a_tool_error(self) -> None:
        """an embedding outage is a tool error string, not a raised exception, and stores nothing."""
        embedder = _RaisingEmbedder()
        collection = _FakeIntentions()
        msg = await _log(embedder, collection, "want")
        assert embedder.calls == 1
        assert msg.startswith("[TOOL ERROR] intention_log: embed failed")
        assert collection.dedup_embeddings == []
        assert collection.saved == []

    async def test_a_want_logged_without_its_dedup_check_says_so(self) -> None:
        """a duplicate can be cleaned up; a refused want may never be retried, so it is stored."""
        collection = _FakeIntentionsDedupDown()
        site = "threetears.agent.intention.tools:load_intention_log_tool.<locals>.intention_log"
        labels = {"site": site, "exc_type": "ConnectionError"}
        before = REGISTRY.get_sample_value("contained_errors_total", labels) or 0.0
        msg = await _log(_OkEmbedder(), collection, "want")
        assert len(collection.saved) == 1
        assert msg == (
            f"Logged as [intention:{collection.saved[0].intention_id}]: want "
            "(not checked against your open wants, which could not be read: ConnectionError)"
        )
        assert "l3 timeout" not in msg, "a store's own message stays in the log, out of the reply"
        assert (REGISTRY.get_sample_value("contained_errors_total", labels) or 0.0) == before + 1

    async def test_a_checked_want_claims_nothing_extra(self) -> None:
        msg = await _log(_OkEmbedder(), _FakeIntentions(), "want")
        assert "not checked" not in msg

    async def test_a_failed_store_names_the_step_and_class_only(self) -> None:
        msg = await _log(_OkEmbedder(), _FakeIntentionsStoreDown(), "want")
        assert msg == "[TOOL ERROR] intention_log: store failed — ConnectionError"

    async def test_success_carries_the_vector_into_dedup_and_the_row(self) -> None:
        collection = _FakeIntentions()
        msg = await _log(_OkEmbedder(), collection, "want")
        assert msg.startswith("Logged as [intention:")
        assert collection.dedup_embeddings == [[0.1, 0.2, 0.3]]
        assert collection.saved[0].embedding == [0.1, 0.2, 0.3]


def _owned_want(user_id: UUID, agent_id: UUID) -> IntentionEntity:
    """one stored ``open`` want owned by ``user_id``."""
    now = datetime.now(UTC)
    return IntentionEntity(
        {
            "intention_id": uuid4(),
            "agent_id": agent_id,
            "customer_id": uuid4(),
            "user_id": user_id,
            "status": "open",
            "content": "ask about the wake threads",
            "embedding": None,
            "salience": Decimal("0.5000"),
            "last_decayed_at": None,
            "last_surfaced_at": None,
            "source_memory_id": None,
            "source_conversation_id": None,
            "date_created": now,
            "date_updated": now,
        },
        is_new=False,
    )


async def _mark_asked(collection_cls: type[_FakeIntentions] = _FakeIntentions) -> str:
    """move one owned want to ``asked`` through the tool, which emits the surfaced event."""
    user_id = uuid4()
    agent_id = uuid4()
    entity = _owned_want(user_id, agent_id)
    (tool,) = await load_intention_mark_surfaced_tool(
        user_id=user_id,
        agent_id=agent_id,
        customer_id=uuid4(),
        authorizer=_authorizer(),
        intentions_collection=collection_cls(entity),  # type: ignore[arg-type]
    )
    result = await tool.ainvoke({"intention_id": str(entity.intention_id), "new_status": "asked"})
    assert isinstance(result, str)
    return result


class TestEmitIntentionEvent:
    async def test_dispatches_event(self) -> None:
        captured: list[Any] = []

        async def _capture(event: Any, *, config: Any = None) -> None:
            captured.append(event)

        with patch("threetears.agent.intention.tools.dispatch_event", new=_capture):
            result = await _mark_asked()
        assert result.startswith("Marked [intention:")
        assert len(captured) == 1
        assert isinstance(captured[0], IntentionSurfacedEvent)

    async def test_no_run_manager_runtime_error_is_swallowed(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """outside a langgraph run, the no-manager ``RuntimeError`` is logged + swallowed.

        the row is already committed by the time the event fires, so a best-effort
        stream surface must not crash the tool.
        """

        async def _raise_no_manager(event: Any, *, config: Any = None) -> None:
            raise RuntimeError("Unable to find run manager")

        with (
            patch("threetears.agent.intention.tools.dispatch_event", new=_raise_no_manager),
            caplog.at_level(logging.DEBUG, logger="threetears.agent.intention.tools"),
        ):
            # must NOT raise
            result = await _mark_asked()
        assert result.startswith("Marked [intention:")
        assert any("intention event dropped" in r.message for r in caplog.records)

    async def test_non_runtime_error_propagates(self) -> None:
        """a non-``RuntimeError`` (e.g. a schema regression) still surfaces."""

        async def _raise_value_error(event: Any, *, config: Any = None) -> None:
            raise ValueError("schema regression")

        with patch("threetears.agent.intention.tools.dispatch_event", new=_raise_value_error):
            with pytest.raises(ValueError, match="schema regression"):
                await _mark_asked()


def _with_failing_reads(failure: Exception) -> type[_FakeIntentions]:
    """a collection whose dedup lookup and listing scans raise ``failure``."""

    class _ReadsFail(_FakeIntentions):
        async def find_similar_for_dedup(self, **kwargs: Any) -> list[dict[str, Any]]:
            _ = kwargs
            raise failure

        async def find_by_user(self, user_id: UUID, *, agent_id: UUID) -> list[IntentionEntity]:
            _ = user_id, agent_id
            raise failure

    return _ReadsFail


def _with_failing_saves(failure: Exception) -> type[_FakeIntentions]:
    """a collection holding one near-duplicate of every want logged, whose saves raise ``failure``."""

    class _SavesFail(_FakeIntentions):
        async def find_similar_for_dedup(self, **kwargs: Any) -> list[dict[str, Any]]:
            _ = kwargs
            assert self.entity is not None
            return [{"intention_id": self.entity.intention_id, "content": "", "salience": 0.5, "similarity": 0.99}]

        async def save_entity(self, entity: IntentionEntity) -> None:
            _ = entity
            raise failure

        async def bump_salience(self, intention_ids: list[UUID], *, agent_id: UUID, access_bump: float) -> None:
            _ = intention_ids, agent_id, access_bump

    return _SavesFail


async def _list_all(collection: _FakeIntentions) -> str:
    (tool,) = await load_intention_list_tool(
        uuid4(),
        uuid4(),
        uuid4(),
        _authorizer(),
        collection,  # type: ignore[arg-type]
    )
    result = await tool.ainvoke({"status": "all"})
    assert isinstance(result, str)
    return result


async def _relog(failure: Exception) -> str:
    """log a want that refreshes its stored near-duplicate, whose save raises ``failure``."""
    user_id, agent_id = uuid4(), uuid4()
    collection = _with_failing_saves(failure)(_owned_want(user_id, agent_id))
    (tool,) = await load_intention_log_tool(
        user_id=user_id,
        embedding_provider=_OkEmbedder(),  # type: ignore[arg-type]
        agent_id=agent_id,
        customer_id=uuid4(),
        authorizer=_authorizer(),
        intentions_collection=collection,  # type: ignore[arg-type]
    )
    result = await tool.ainvoke({"content": "ask about the wake threads"})
    assert isinstance(result, str)
    return result


def _with_failing_creates(failure: Exception) -> type[_FakeIntentions]:
    """a collection holding no want like the one logged, whose saves raise ``failure``."""

    class _CreatesFail(_FakeIntentions):
        async def save_entity(self, entity: IntentionEntity) -> None:
            _ = entity
            raise failure

    return _CreatesFail


#: what a new want's insert can meet that is not a bug: an outage, a generation that could not
#: advance, a row count the store did not report
_CREATE_REFUSALS: list[Exception] = [
    DataLayerUnavailableError("broker down"),
    GenerationUnavailableError("generation not advanced"),
    RowcountUnreadableError("a successful insert reply must carry row_count"),
]


#: what a fenced save of a stored want can meet that is not a bug: an outage, a lost CAS race, a
#: generation that could not advance, a row count the store did not report
_SAVE_REFUSALS: list[Exception] = [
    DataLayerUnavailableError("broker down"),
    ConcurrentModificationError("intentions", uuid4(), datetime.now(UTC)),
    GenerationUnavailableError("generation not advanced"),
    RowcountUnreadableError("a successful update reply must carry row_count"),
]


class TestOnlyAStoreFailureReadsAsOne:
    """each handler answers the failures its store raises; a bug in our code propagates as itself."""

    async def test_a_dedup_lookup_bug_propagates(self) -> None:
        with pytest.raises(TypeError, match="bug"):
            await _log(_OkEmbedder(), _with_failing_reads(TypeError("bug"))(), "want")

    async def test_a_store_outage_listing_is_a_tool_error(self) -> None:
        out = await _list_all(_with_failing_reads(DataLayerUnavailableError("broker down"))())
        assert out == "[TOOL ERROR] intention_list: query failed — DataLayerUnavailableError"

    async def test_a_listing_bug_propagates(self) -> None:
        with pytest.raises(AttributeError, match="bug"):
            await _list_all(_with_failing_reads(AttributeError("bug"))())

    @pytest.mark.parametrize("failure", _SAVE_REFUSALS, ids=lambda e: type(e).__name__)
    async def test_a_refused_refresh_is_a_tool_error(self, failure: Exception) -> None:
        assert await _relog(failure) == f"[TOOL ERROR] intention_log: refresh failed — {type(failure).__name__}"

    @pytest.mark.parametrize("failure", _CREATE_REFUSALS, ids=lambda e: type(e).__name__)
    async def test_a_refused_create_is_a_tool_error(self, failure: Exception) -> None:
        collection = _with_failing_creates(failure)()
        assert await _log(_OkEmbedder(), collection, "want") == (
            f"[TOOL ERROR] intention_log: store failed — {type(failure).__name__}"
        )

    async def test_a_create_bug_propagates(self) -> None:
        with pytest.raises(TypeError, match="bug"):
            await _log(_OkEmbedder(), _with_failing_creates(TypeError("bug"))(), "want")

    async def test_an_insert_that_lands_no_row_propagates(self) -> None:
        """the key is minted per call, so no race lands no row; a collection with no L3 store does."""
        failure = RuntimeError("L3 insert failed for intentions entity x: 0 rows affected")
        with pytest.raises(RuntimeError, match="0 rows affected"):
            await _log(_OkEmbedder(), _with_failing_creates(failure)(), "want")

    async def test_a_refresh_bug_propagates(self) -> None:
        with pytest.raises(TypeError, match="bug"):
            await _relog(TypeError("bug"))

    @pytest.mark.parametrize("failure", _SAVE_REFUSALS, ids=lambda e: type(e).__name__)
    async def test_a_refused_mark_is_a_tool_error(self, failure: Exception) -> None:
        out = await _mark_asked(_with_failing_saves(failure))
        assert out == f"[TOOL ERROR] intention_mark_surfaced: store failed — {type(failure).__name__}"

    async def test_a_mark_bug_propagates(self) -> None:
        with pytest.raises(KeyError, match="bug"):
            await _mark_asked(_with_failing_saves(KeyError("bug")))
