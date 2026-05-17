"""Tests for tool-surface event classes + shared-registry registration.

Pins the registration into :data:`threetears.langgraph.events.default_registry`
so a future refactor that quietly drops the import side effect fails
the build.
"""

from __future__ import annotations

from threetears.agent.tools.events import TodosChangedEvent
from threetears.langgraph.events import default_registry


def test_todos_changed_registered_in_shared_default_registry() -> None:
    assert "todos_changed" in default_registry.names()
    parsed = default_registry.parse(
        "todos_changed",
        {
            "todos": [{"id": "t-1", "text": "ship the lift"}],
            "message_id_source": "m-1",
        },
    )
    assert isinstance(parsed, TodosChangedEvent)
    assert parsed.todos == [{"id": "t-1", "text": "ship the lift"}]
    assert parsed.message_id_source == "m-1"


def test_todos_changed_defaults() -> None:
    ev = TodosChangedEvent()
    assert ev.todos == []
    assert ev.message_id_source is None
