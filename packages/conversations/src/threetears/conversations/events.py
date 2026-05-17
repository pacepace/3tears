"""typed conversation-lifecycle events dispatched through langchain custom events.

these events ride the same in-process transport as the langgraph-layer
events in :mod:`threetears.langgraph.events` -- ``adispatch_custom_event``
inside a graph node, ``on_custom_event`` on the consumer side. consumers
parse them via the shared
:data:`threetears.langgraph.events.default_registry`, which this module
mutates on import.

the only event defined here today is :class:`ConversationSummarizedEvent`,
fired by the auto-summarizer when older messages have been compressed
into a single in-context summary so the conversation can continue under
its token budget. additional conversation-lifecycle events (renamed,
archived, etc.) can register here as the surface grows.
"""

from __future__ import annotations

from typing import Literal

from threetears.langgraph.events import FrameworkEvent, default_registry

__all__ = ["ConversationSummarizedEvent"]


class ConversationSummarizedEvent(FrameworkEvent):
    """fired when an auto-summarizer has compressed older messages.

    the in-context summary becomes the only artifact of the older
    messages for subsequent turns -- the originals stay in storage but
    are no longer in the prompt. consumers typically surface a
    "earlier messages summarized" affordance in the ui so the user
    knows what happened.

    note: this event reports the auto-summarizer's in-context summary.
    the v0.14.0 transcript-chunks ``conversation_summarize`` tool that
    PERSISTS summaries as durable memory chunks is a separate concern
    and dispatches its own event surface when it lands.

    :ivar messages_summarized: count of messages folded into the summary
    :ivar summary_text: the assembled summary text, or ``None`` when the
        producer did not surface the text (some auto-summarizer impls
        embed the summary directly into the next system message and do
        not surface it as a separate field)
    """

    type: Literal["conversation_summarized"] = "conversation_summarized"
    messages_summarized: int = 0
    summary_text: str | None = None


def _register_conversations_events(registry: object) -> None:
    """register conversation-lifecycle events into ``registry``.

    invoked at import time against
    :data:`threetears.langgraph.events.default_registry` via
    :meth:`FrameworkEventRegistry.add_framework_defaults_provider`.
    accepts the registry as an argument so it can also be reused by
    :meth:`FrameworkEventRegistry.reset_to_framework_defaults` after a
    test clears the registry.

    :param registry: registry to populate
    :ptype registry: FrameworkEventRegistry
    :return: nothing
    :rtype: None
    """
    if ConversationSummarizedEvent.model_fields["type"].default in registry.names():
        return
    registry.register(ConversationSummarizedEvent)


default_registry.add_framework_defaults_provider(_register_conversations_events)
