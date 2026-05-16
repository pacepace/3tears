"""typed tool-surface events dispatched through langchain custom events.

these events ride the same in-process transport as the langgraph-layer
events in :mod:`threetears.langgraph.events` -- ``adispatch_custom_event``
inside a graph node, ``on_custom_event`` on the consumer side. consumers
parse them via the shared
:data:`threetears.langgraph.events.default_registry`, which this module
mutates on import.

the only event defined here today is :class:`TodosChangedEvent`, fired by
the todo-tool factory in :mod:`threetears.agent.tools.todo` whenever the
underlying todo list mutates. additional tool-surface events (e.g. for
the file/workspace tooling) can register here as the surface grows.

naming follows the framework convention: ``noun_verb`` with past-tense
verb. see :mod:`threetears.langgraph.events` for the registry semantics.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field
from threetears.langgraph.events import FrameworkEvent, default_registry

__all__ = ["TodosChangedEvent"]


class TodosChangedEvent(FrameworkEvent):
    """fired when an agent's todo list has changed.

    the ``todos`` payload is the full current list -- consumers render
    by replacement, not by patch -- so a missed event eventually self-
    corrects on the next mutation. ``message_id_source`` identifies the
    assistant message that drove the mutation, when known, so a ui that
    threads todos under the message that produced them can link them up.

    :ivar todos: full current todo list. each entry is a dict with the
        product-defined todo shape; the framework intentionally does not
        pin a schema here because the todo storage layer
        (:class:`threetears.agent.tools.todo.TodoStorage`) is an
        injection point and downstream products may layer their own
        columns (priority, due, etc.). consumers that need typed access
        own the typed projection
    :ivar message_id_source: string-form uuid of the assistant message
        that produced the mutation, or ``None`` when the mutation
        happened outside a message context
    """

    type: Literal["todos_changed"] = "todos_changed"
    todos: list[dict[str, Any]] = Field(default_factory=list)
    message_id_source: str | None = None


def _register() -> None:
    """register tool-surface events into the shared default registry.

    runs at import time so a consumer that imports this module sees
    :class:`TodosChangedEvent` in
    :data:`threetears.langgraph.events.default_registry`.

    :return: nothing
    :rtype: None
    """
    default_registry.register(TodosChangedEvent)


_register()
