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
from pydantic import ValidationError

from threetears.agent.acl import AclCache
from threetears.agent.intention.authorize import IntentionAuthorizerDependencies
from threetears.agent.intention.entities import IntentionEntity
from threetears.agent.intention.events import IntentionSurfacedEvent
from threetears.agent.intention.tools import (
    IntentionListInput,
    IntentionLogInput,
    IntentionMarkSurfacedInput,
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
        assert "embedding provider returned None" in msg


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

    async def test_success_carries_the_vector_into_dedup_and_the_row(self) -> None:
        collection = _FakeIntentions()
        msg = await _log(_OkEmbedder(), collection, "want")
        assert msg.startswith("Logged as [intention:")
        assert collection.dedup_embeddings == [[0.1, 0.2, 0.3]]
        assert collection.saved[0].embedding == [0.1, 0.2, 0.3]


async def _mark_asked() -> str:
    """move one owned want to ``asked`` through the tool, which emits the surfaced event."""
    user_id = uuid4()
    agent_id = uuid4()
    now = datetime.now(UTC)
    entity = IntentionEntity(
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
    (tool,) = await load_intention_mark_surfaced_tool(
        user_id=user_id,
        agent_id=agent_id,
        customer_id=uuid4(),
        authorizer=_authorizer(),
        intentions_collection=_FakeIntentions(entity),  # type: ignore[arg-type]
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
