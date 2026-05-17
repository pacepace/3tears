"""Tests for conversation-lifecycle event classes + shared-registry registration.

Pins the registration into :data:`threetears.langgraph.events.default_registry`
so a future refactor that quietly drops the import side effect fails
the build.
"""

from __future__ import annotations

from threetears.conversations.events import ConversationSummarizedEvent
from threetears.langgraph.events import default_registry


def test_conversation_summarized_registered_in_shared_default_registry() -> None:
    assert "conversation_summarized" in default_registry.names()
    parsed = default_registry.parse(
        "conversation_summarized",
        {"messages_summarized": 12, "summary_text": "they argued about kerning"},
    )
    assert isinstance(parsed, ConversationSummarizedEvent)
    assert parsed.messages_summarized == 12
    assert parsed.summary_text == "they argued about kerning"


def test_conversation_summarized_defaults() -> None:
    ev = ConversationSummarizedEvent()
    assert ev.messages_summarized == 0
    assert ev.summary_text is None
