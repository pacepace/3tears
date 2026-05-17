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

helpers defined here:

- :func:`default_memory_created_dispatcher` -- opt-in default for
  :attr:`MemoryExtractor.on_memory_created`. builds and dispatches a
  :class:`MemoryCreatedEvent` from the just-committed memory entity,
  with a graceful no-op when called outside a langgraph run.
  consumers using the canonical extraction path wire this as their
  default callback; product-specific push (websocket, slack) wraps
  this helper rather than re-implementing the build+dispatch shape.

naming follows the framework convention: ``noun_verb`` with past-tense
verb. see :mod:`threetears.langgraph.events` for the registry semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field
from threetears.langgraph.events import (
    FrameworkEvent,
    default_registry,
    dispatch_event,
)
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.agent.memory.entities import MemoryEntity

__all__ = [
    "MemoryCreatedEvent",
    "MemoryRetrievedEvent",
    "default_memory_created_dispatcher",
]

log = get_logger(__name__)


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


_CONTENT_PREVIEW_LEN: int = 120
"""maximum chars carried in the event payload's content preview.

trims the ``content_preview`` field on :class:`MemoryCreatedEvent` to
match the field's docstring (~120 chars). callers that need the full
content re-fetch from storage by the ``memory_id`` on the event.
"""


async def default_memory_created_dispatcher(entity: MemoryEntity) -> None:
    """opt-in default callback for :attr:`MemoryExtractor.on_memory_created`.

    builds a :class:`MemoryCreatedEvent` from the just-committed memory
    entity and dispatches it through the langchain custom-event channel
    so consumers reading ``astream_events('v2')`` see the
    ``memory_created`` event on their stream. the wire payload carries
    the user binding + identifying fields needed for client-side
    scoping; the full content stays in storage and the
    ``content_preview`` field is truncated to
    :data:`_CONTENT_PREVIEW_LEN` chars.

    graceful no-op when called outside a langgraph run (no run manager
    in scope, e.g. cli / background-job harnesses that call
    :meth:`MemoryExtractor.run_extraction` directly without compiling
    a graph). the row is already committed; the missing stream event
    is logged at debug for ops visibility but the dispatcher does not
    raise.

    consumers that want product-specific push (websocket frame,
    slack message, sse delta) on top of the v2 stream event wrap
    this helper in their own callback and call it first so the framework
    dispatch always happens before the product push::

        async def my_callback(entity: MemoryEntity) -> None:
            await default_memory_created_dispatcher(entity)
            await my_product_push(entity)

        extractor = MemoryExtractor(
            on_memory_created=my_callback,
            ...,
        )

    consumers that want ONLY the v2 stream event (no product-specific
    push) pass the helper directly as the callback.

    :param entity: the :class:`MemoryEntity` that was just committed
        by :meth:`MemoryExtractor.run_extraction`. all required fields
        (``user_id``, ``memory_id``, ``conversation_id``, ``type_memory``)
        must be populated -- the extractor pipeline guarantees this
        post-``save_entity``, so callers wiring the helper outside that
        path are responsible for the same guarantee
    :ptype entity: MemoryEntity
    :return: nothing
    :rtype: None
    """
    content = getattr(entity, "content", None) or ""
    preview = content[:_CONTENT_PREVIEW_LEN]
    event = MemoryCreatedEvent(
        user_id=str(entity.user_id),
        memory_id=str(entity.memory_id),
        conversation_id=str(entity.conversation_id),
        type_memory=str(entity.type_memory),
        content_preview=preview,
    )

    try:
        await dispatch_event(event)
    except RuntimeError as exc:
        # ``adispatch_custom_event`` raises ``RuntimeError`` when no
        # run manager is in scope -- i.e. the extractor was called
        # outside a langgraph node (cli / background job / test
        # harness). the row is committed; the stream event is a
        # best-effort surface. log at debug so ops can see it if
        # they care; warning would be too loud for the normal-case
        # cli path. narrowing the catch to ``RuntimeError``
        # preserves the bug-surfacing property for other failures
        # inside ``dispatch_event`` (e.g. a pydantic schema regression).
        log.debug(
            "default_memory_created_dispatcher: no run manager in scope; "
            "memory_created event dropped (memory_id=%s, reason=%s)",
            event.memory_id,
            exc,
        )


def _register_memory_events(registry: Any) -> None:
    """register memory events into ``registry``.

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
    for cls in (
        MemoryRetrievedEvent,
        MemoryCreatedEvent,
    ):
        if cls.model_fields["type"].default in registry.names():
            continue
        registry.register(cls)


default_registry.add_framework_defaults_provider(_register_memory_events)
