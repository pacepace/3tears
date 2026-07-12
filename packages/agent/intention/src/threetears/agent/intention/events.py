"""typed intention events dispatched through langchain custom events.

These events ride the same in-process transport as the langgraph-layer
events in :mod:`threetears.langgraph.events`: they dispatch via
``adispatch_custom_event`` inside a langgraph run and surface as
``on_custom_event`` ticks in ``astream_events('v2')``. Consumers parse
them through the shared
:data:`threetears.langgraph.events.default_registry`, which this module
mutates on import (the memory ``events.py`` template, NOT wake's
plain-string-constant one).

Intention-specific events live here rather than in the langgraph package
so that adding a new intention event does not touch langgraph's public
surface. The wire-name discriminator (the ``type`` literal) is the
contract; producers and consumers agree on the name and the registry
maps it to the concrete class.

events defined here:

- :class:`IntentionSurfacedEvent` -- fired when the agent raises a
  standing want (``open -> asked``) via ``intention_mark_surfaced``.
  Feeds the presence timeline + the agenda greeting.
- :class:`IntentionResolvedEvent` -- fired when a want is resolved
  (``asked -> granted`` / ``asked -> dropped``), typically from the
  presence-API grant/drop path.

naming follows the framework convention: ``noun_verb`` with a past-tense
verb. See :mod:`threetears.langgraph.events` for registry semantics.
"""

from __future__ import annotations

from typing import Any, Literal

from threetears.langgraph.events import FrameworkEvent, default_registry
from threetears.observe import get_logger

__all__ = [
    "IntentionResolvedEvent",
    "IntentionSurfacedEvent",
]

log = get_logger(__name__)


class IntentionSurfacedEvent(FrameworkEvent):
    """fired when the agent raises a standing want to the user.

    Emitted by ``intention_mark_surfaced`` on the ``open -> asked``
    transition -- the deliberation wake decided this want was worth
    surfacing. Feeds the presence timeline ("Saoirse wants to ...") and
    the agenda greeting. One event per surfaced want.

    ``user_id`` is a string-form uuid when the want is user-scoped and
    ``None`` for an agent-internal / global want (the 3tears primitive
    allows a null user grain; metallm always sets it). A consumer scoping
    a per-user refresh checks for the non-null value.

    :ivar agent_id: string-form uuid of the agent that raised the want
    :ivar intention_id: string-form uuid of the surfaced want
    :ivar new_status: the status the want moved to (``'asked'``)
    :ivar user_id: string-form uuid of the owning user, or ``None`` on an
        agent-internal want
    :ivar content_preview: first ~120 chars of the want text for ui
        rendering without a follow-up read
    """

    type: Literal["intention_surfaced"] = "intention_surfaced"
    agent_id: str
    intention_id: str
    new_status: str = "asked"
    user_id: str | None = None
    content_preview: str = ""


class IntentionResolvedEvent(FrameworkEvent):
    """fired when a standing want is resolved (granted or dropped).

    Emitted by ``intention_mark_surfaced`` on the ``asked -> granted`` /
    ``asked -> dropped`` transition -- typically from the presence-API
    grant/drop path where the user responds to a surfaced want. Lets the
    presence timeline retire the open item. One event per resolution.

    ``user_id`` follows the same null-scope contract as
    :class:`IntentionSurfacedEvent`.

    :ivar agent_id: string-form uuid of the agent that owns the want
    :ivar intention_id: string-form uuid of the resolved want
    :ivar new_status: the terminal status (``'granted'`` or ``'dropped'``)
    :ivar user_id: string-form uuid of the owning user, or ``None`` on an
        agent-internal want
    :ivar content_preview: first ~120 chars of the want text for ui
        rendering without a follow-up read
    """

    type: Literal["intention_resolved"] = "intention_resolved"
    agent_id: str
    intention_id: str
    new_status: str
    user_id: str | None = None
    content_preview: str = ""


def _register_intention_events(registry: Any) -> None:
    """register intention events into ``registry``.

    Invoked at import time against
    :data:`threetears.langgraph.events.default_registry` via
    :meth:`FrameworkEventRegistry.add_framework_defaults_provider`, and
    reused by :meth:`FrameworkEventRegistry.reset_to_framework_defaults`
    after a test clears the registry. Idempotent: a wire name already
    present is skipped.

    :param registry: registry to populate
    :ptype registry: FrameworkEventRegistry
    :return: nothing
    :rtype: None
    """
    for cls in (
        IntentionSurfacedEvent,
        IntentionResolvedEvent,
    ):
        if cls.model_fields["type"].default in registry.names():
            continue
        registry.register(cls)


default_registry.add_framework_defaults_provider(_register_intention_events)
