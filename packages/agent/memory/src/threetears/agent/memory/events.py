"""typed memory events dispatched through langchain custom events.

these events sit on the same in-process transport as the langgraph-layer
events in :mod:`threetears.langgraph.events` -- they ride
``adispatch_custom_event`` inside a langgraph run and surface as
``on_custom_event`` ticks in ``astream_events('v2')``. consumers parse
them via the shared :data:`threetears.langgraph.events.default_registry`,
which this module mutates on import.

memory-specific events live here rather than in the langgraph package so
that adding a new memory event does not require touching langgraph's
public surface. the wire-name discriminator (``type`` literal) is the
contract; producers and consumers agree on the name, the registry maps
it to the concrete class.

events defined here:

- :class:`MemoryRetrievedEvent` -- fired by a memory-retrieval node when
  grounding hits have been resolved for the current turn. consumers
  typically surface the hits in a "memories used" panel.
- :class:`MemoryCreatedEvent` -- fired by the memory extraction pipeline
  when a new memory row commits. server-push refresh signal for any ui
  panel rendering the user's memories (the memory-page-doesn't-refresh
  fix v0.14.0).

naming follows the framework convention: ``noun_verb`` with past-tense
verb. see :mod:`threetears.langgraph.events` for the registry semantics.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field
from threetears.langgraph.events import FrameworkEvent, default_registry

__all__ = [
    "MemoryCreatedEvent",
    "MemoryRetrievedEvent",
]


class MemoryRetrievedEvent(FrameworkEvent):
    """fired when a memory-retrieval node has resolved grounding hits.

    consumers typically render the hits in a "memories used this turn"
    panel so the user can see what the agent recalled. the ``memories``
    field carries truncated previews -- the full memory content lives
    in the database; this event is a notification, not a transfer.

    :ivar memories_count: number of memories surfaced this turn
    :ivar memories: list of memory preview dicts. each dict contains:
        ``memory_id`` (str), ``type`` (str), ``content_preview`` (str,
        first ~80 chars), ``similarity`` (float, 0-1), and an optional
        ``hybrid_score`` (float | None) when hybrid search was used
    """

    type: Literal["memory_retrieved"] = "memory_retrieved"
    memories_count: int = 0
    memories: list[dict[str, Any]] = Field(default_factory=list)


class MemoryCreatedEvent(FrameworkEvent):
    """fired when a new memory row has committed to storage.

    the dispatch happens AFTER the database write completes successfully
    so consumers can safely re-fetch from the store on receipt. the
    payload carries the user binding + identifying fields needed for a
    consumer to scope the refresh (one user's ui must not refresh on
    another user's memory create).

    fired by :func:`threetears.agent.memory.extraction.run_extraction`
    (and by the equivalent product-side memory-creation path when a
    product writes memories outside the standard extraction pipeline).

    :ivar user_id: string-form uuid of the user the memory belongs to
        (cast to str rather than uuid so the json round-trip stays
        symmetric -- pydantic serializes uuids as str and consumers
        typically need the str form for client-side scoping anyway)
    :ivar memory_id: string-form uuid of the new memory row
    :ivar conversation_id: string-form uuid of the conversation the
        memory anchors to (memories.conversation_id is NOT NULL in 3tears
        v0.6.0+)
    :ivar type_memory: discriminator for the memory's type
        (``'topical_context'``, ``'fact'``, ``'preference'``, etc.)
    :ivar content_preview: first ~120 chars of memory.content for ui
        rendering without re-fetching. the full content stays in the
        database; this is a preview only
    """

    type: Literal["memory_created"] = "memory_created"
    user_id: str
    memory_id: str
    conversation_id: str
    type_memory: str
    content_preview: str = ""


def _register() -> None:
    """register memory events into the shared default registry.

    runs at import time so a consumer that imports this module sees
    every memory event class in
    :data:`threetears.langgraph.events.default_registry`.

    :return: nothing
    :rtype: None
    """
    for cls in (
        MemoryRetrievedEvent,
        MemoryCreatedEvent,
    ):
        default_registry.register(cls)


_register()
