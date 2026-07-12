"""unit tests for agent-intention event classes + shared-registry registration.

The shared default registry lives in :mod:`threetears.langgraph.events`;
intention events register themselves into it on import. These tests pin
that registration so a future refactor that quietly drops the import side
effect (and breaks every product that parses intention events through the
shared registry) fails the build instead of silently regressing.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from threetears.agent.intention.events import (
    IntentionResolvedEvent,
    IntentionSurfacedEvent,
)
from threetears.langgraph.events import default_registry


class TestSharedRegistryHasIntentionEvents:
    def test_intention_surfaced_registered(self) -> None:
        assert "intention_surfaced" in default_registry.names()
        parsed = default_registry.parse(
            "intention_surfaced",
            {
                "agent_id": "a-1",
                "intention_id": "i-1",
                "new_status": "asked",
                "user_id": "u-1",
                "content_preview": "learn more Irish",
            },
        )
        assert isinstance(parsed, IntentionSurfacedEvent)
        assert parsed.intention_id == "i-1"
        assert parsed.new_status == "asked"

    def test_intention_resolved_registered(self) -> None:
        assert "intention_resolved" in default_registry.names()
        parsed = default_registry.parse(
            "intention_resolved",
            {
                "agent_id": "a-1",
                "intention_id": "i-1",
                "new_status": "granted",
                "user_id": "u-1",
                "content_preview": "done",
            },
        )
        assert isinstance(parsed, IntentionResolvedEvent)
        assert parsed.new_status == "granted"


class TestIntentionEventShape:
    def test_surfaced_requires_identifying_fields(self) -> None:
        # agent_id + intention_id are required.
        with pytest.raises(ValidationError):
            IntentionSurfacedEvent()  # type: ignore[call-arg]

    def test_surfaced_defaults(self) -> None:
        ev = IntentionSurfacedEvent(agent_id="a", intention_id="i")
        assert ev.new_status == "asked"
        assert ev.user_id is None  # null-scope tolerant (agent-internal want)
        assert ev.content_preview == ""

    def test_resolved_user_id_nullable(self) -> None:
        # a null user_id is legal for an agent-internal want -- never str(None).
        ev = IntentionResolvedEvent(agent_id="a", intention_id="i", new_status="dropped")
        assert ev.user_id is None
