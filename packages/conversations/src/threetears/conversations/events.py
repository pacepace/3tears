"""typed conversation-lifecycle events dispatched through langchain custom events.

these events ride the same in-process transport as the langgraph-layer
events in :mod:`threetears.langgraph.events` -- ``adispatch_custom_event``
inside a graph node, ``on_custom_event`` on the consumer side. consumers
parse them via the shared
:data:`threetears.langgraph.events.default_registry`, which this module
mutates on import.

the only event defined here today is :class:`ConversationSummarizedEvent`,
fired both by the auto-summarizer (compressing older messages into a
single in-context summary so the conversation can continue under its
token budget) and by an explicit, agent-invoked summarize-and-persist
action. additional conversation-lifecycle events (renamed, archived,
etc.) can register here as the surface grows.
"""

from __future__ import annotations

from typing import Literal

from threetears.langgraph.events import FrameworkEvent, default_registry

__all__ = ["ConversationSummarizedEvent"]


class ConversationSummarizedEvent(FrameworkEvent):
    """fired whenever a range of conversation messages has been summarized,
    whether ephemerally (auto-compaction) or durably (an explicit persist
    action).

    consumers typically surface a "messages summarized" affordance in the
    ui so the user knows what happened, regardless of which producer fired
    it -- the two producers differ only in ``persisted``.

    correction: an earlier revision of this docstring said the durable,
    persisting producer "dispatches its own event surface when it lands" --
    that never happened, and the two producers were never unified until now.
    they share this one event type, distinguished by ``persisted``, rather
    than the ui needing to know about two separate event shapes for what is,
    from the user's perspective, the same kind of thing happening.

    :ivar messages_summarized: count of messages folded into the summary
    :ivar summary_text: the assembled summary text, or ``None`` when the
        producer did not surface the text (some auto-summarizer impls
        embed the summary directly into the next system message and do
        not surface it as a separate field)
    :ivar persisted: ``True`` when the summary was written as a durable,
        embedded, searchable memory (an explicit agent action); ``False``
        for the auto-summarizer's ephemeral in-context-only compaction.
        The auto-summarizer must NEVER set this ``True`` -- durable
        persistence is exclusively an explicit, agent-chosen action.
    """

    type: Literal["conversation_summarized"] = "conversation_summarized"
    messages_summarized: int = 0
    summary_text: str | None = None
    persisted: bool = False


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
