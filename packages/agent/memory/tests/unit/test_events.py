"""Tests for memory-package event classes + shared-registry registration.

The shared default registry lives in :mod:`threetears.langgraph.events`;
memory events register themselves into it on import. These tests pin
that registration so a future refactor that quietly drops the import
side effect (and breaks every product that parses memory events through
the shared registry) fails the build instead of silently regressing.
"""

from __future__ import annotations

import pytest
from threetears.agent.memory.events import (
    MemoryCreatedEvent,
    MemoryRetrievedEvent,
)
from threetears.langgraph.events import default_registry


class TestSharedRegistryHasMemoryEvents:
    def test_memory_retrieved_registered(self) -> None:
        assert "memory_retrieved" in default_registry.names()
        parsed = default_registry.parse(
            "memory_retrieved",
            {"memories_count": 2, "memories": [{"memory_id": "abc"}]},
        )
        assert isinstance(parsed, MemoryRetrievedEvent)
        assert parsed.memories_count == 2

    def test_memory_created_registered(self) -> None:
        assert "memory_created" in default_registry.names()
        parsed = default_registry.parse(
            "memory_created",
            {
                "user_id": "u-1",
                "memory_id": "m-1",
                "conversation_id": "c-1",
                "type_memory": "topical_context",
                "content_preview": "hello",
            },
        )
        assert isinstance(parsed, MemoryCreatedEvent)
        assert parsed.user_id == "u-1"
        assert parsed.content_preview == "hello"


class TestMemoryEventDefaults:
    def test_memory_retrieved_defaults_empty(self) -> None:
        ev = MemoryRetrievedEvent()
        assert ev.memories_count == 0
        assert ev.memories == []

    def test_memory_created_requires_identifying_fields(self) -> None:
        # user_id, memory_id, conversation_id, type_memory are required.
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MemoryCreatedEvent()  # type: ignore[call-arg]
