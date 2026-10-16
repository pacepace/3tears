"""Todo tools -- LangChain tools for managing conversation-scoped todo lists.

Provides ``load_todo_tools`` as a factory. Callers supply a ``TodoStorage``
implementation (e.g., PostgreSQL-backed) and an optional ``snapshot_callback``
for real-time updates (e.g., LangGraph custom events).

A todo is named by its id (``[todo:<id>]``, which every result and listing
shows) or by its title. A title names a todo only when it matches exactly one:
the whole title in any case first, then the todos whose title holds it. When
it matches more than one, nothing changes and the candidates are listed with
their ids, so the next call names the one meant.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, tzinfo
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from threetears.observe.text_block import one_line
from threetears.core.utils.local_time import render_local_time, resolve_zone
from threetears.observe import get_logger

from threetears.agent.tools.utils import TOOL_ERROR_PREFIX, tool_no_change

__all__ = [
    "AddTodoInput",
    "CompleteTodoInput",
    "ListTodosInput",
    "RemoveTodoInput",
    "SnapshotCallback",
    "TODO_HANDLE",
    "TodoStorage",
    "UpdateTodoInput",
    "load_todo_tools",
    "todo_handle",
]

log = get_logger(__name__)

#: How many titles a "not found" answer lists before it counts the rest.
_LISTED_ON_A_MISS = 20

#: The list a todo goes on when the caller names none.
DEFAULT_LIST = "Todo List"

#: The handle kind a todo is shown and named by: ``[todo:<id>]``.
TODO_HANDLE = "todo"

#: A todo id as a caller may write it: bare, ``todo:<id>``, or ``[todo:<id>]``.
_HANDLE = re.compile(r"^\s*\[?\s*(?:todo:)?\s*([0-9a-fA-F-]{32,36})\s*\]?\s*$")


def todo_handle(todo_id: object) -> str:
    """The handle a todo is shown by: ``[todo:<id>]``.

    :param todo_id: the todo's id
    :ptype todo_id: object
    :return: the handle
    :rtype: str
    """
    return f"[{TODO_HANDLE}:{todo_id}]"


# -- Storage protocol ---------------------------------------------------------


@runtime_checkable
class TodoStorage(Protocol):
    """Protocol for todo persistence backends.

    Every row a method returns is a dict with at least ``todo_id``, ``title``,
    ``is_completed`` and ``list_name``. ``date_created`` and ``date_updated``
    (a ``datetime`` or an ISO string) are shown when present.
    """

    async def add(
        self,
        conversation_id: UUID,
        user_id: UUID,
        title: str,
        list_name: str,
        message_id: UUID | None,
    ) -> dict[str, Any]:
        """Add a todo item. Returns the row.

        Raises ValueError if the title is a duplicate (any case) in the same list.
        """
        ...

    async def get(self, conversation_id: UUID, todo_id: UUID) -> dict[str, Any] | None:
        """The todo with this id in this conversation, or ``None``.

        A todo of another conversation is ``None``: the id is checked against the
        conversation, never trusted on its own.
        """
        ...

    async def match_title(
        self,
        conversation_id: UUID,
        title: str,
        list_name: str | None,
    ) -> list[dict[str, Any]]:
        """The todos a title names, ordered by list then position.

        Those whose whole title is ``title`` in any case; when none is, those
        whose title holds ``title`` in any case. ``title`` is taken literally:
        ``%`` and ``_`` match only themselves. ``list_name`` ``None`` searches
        every list; otherwise only that list (any case).
        """
        ...

    async def complete(
        self,
        todo_id: UUID,
        message_id: UUID | None,
    ) -> None:
        """Mark a todo as completed."""
        ...

    async def update(
        self,
        todo_id: UUID,
        message_id: UUID | None,
        new_title: str | None = None,
        is_completed: bool | None = None,
        new_list_name: str | None = None,
    ) -> None:
        """Update a todo's title, completion state and/or list (a move goes to the end of the new list)."""
        ...

    async def remove(self, todo_id: UUID) -> None:
        """Delete a todo item."""
        ...

    async def list_all(self, conversation_id: UUID) -> list[dict[str, Any]]:
        """Return all todos for a conversation, ordered by list then sort."""
        ...

    async def resolve_list_name(
        self,
        conversation_id: UUID,
        list_name: str,
    ) -> str:
        """Resolve list_name to canonical casing from an existing list."""
        ...


# -- Type aliases -------------------------------------------------------------

SnapshotCallback = Callable[[list[dict[str, Any]], str | None], Awaitable[None]]


# -- Input schemas ------------------------------------------------------------

_ID_FIELD = "The todo's id, from a [todo:<id>] mark. Give this or title."
_TITLE_FIELD = (
    "The todo's whole title, any case, or a part of it that only one todo has. Give this or todo_id. "
    "When it fits more than one todo, nothing changes and their ids are listed."
)
_LIST_FIELD = "The list the todo is on. Leave it out to look in every list."


class AddTodoInput(BaseModel):
    """Input for add_todo tool."""

    title: str = Field(description="Title of the todo item to add")
    list_name: str = Field(
        default=DEFAULT_LIST,
        description="Name of the todo list to add to (e.g. 'Deployment Checklist')",
    )


class CompleteTodoInput(BaseModel):
    """Input for complete_todo tool."""

    todo_id: str | None = Field(default=None, description=_ID_FIELD)
    title: str | None = Field(default=None, description=_TITLE_FIELD)
    list_name: str | None = Field(default=None, description=_LIST_FIELD)


class UpdateTodoInput(BaseModel):
    """Input for update_todo tool."""

    todo_id: str | None = Field(default=None, description=_ID_FIELD)
    title: str | None = Field(default=None, description=_TITLE_FIELD)
    new_title: str | None = Field(default=None, description="New title (if renaming)")
    is_completed: bool | None = Field(default=None, description="New completion state")
    new_list_name: str | None = Field(default=None, description="The list to move it to (if moving)")
    list_name: str | None = Field(default=None, description=_LIST_FIELD)


class RemoveTodoInput(BaseModel):
    """Input for remove_todo tool."""

    todo_id: str | None = Field(default=None, description=_ID_FIELD)
    title: str | None = Field(default=None, description=_TITLE_FIELD)
    list_name: str | None = Field(default=None, description=_LIST_FIELD)


class ListTodosInput(BaseModel):
    """Input for list_todos tool."""

    list_name: str | None = Field(default=None, description="One list to show. Leave it out to show every list.")


# -- Rendering ----------------------------------------------------------------


def _uuid(value: object) -> UUID:
    """A row's id as a ``UUID``: a storage may hand it back as one or as its text (``list_all`` rows travel as JSON)."""
    if isinstance(value, UUID):
        return value
    return UUID(str(value))  # convert at border: a storage row's id read back from its text form


def _state(row: dict[str, Any]) -> str:
    return "done" if row.get("is_completed") else "open"


def _when(value: object, zone: tzinfo | None) -> str | None:
    """A stored time as the person reads it, or ``None`` when the row has none."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:  # NOSILENT: a stored time that does not parse is left out of the line, not guessed
            return None
    if not isinstance(value, datetime):
        return None
    return render_local_time(value, zone)


def _line(row: dict[str, Any]) -> str:
    """One todo as a candidate or a result names it: handle, title, list, state."""
    return f'{todo_handle(row["todo_id"])} "{row["title"]}" in "{row["list_name"]}" ({_state(row)})'


def _count(rows: Sequence[dict[str, Any]]) -> str:
    done = sum(1 for r in rows if r.get("is_completed"))
    return f"{len(rows)} todo{'' if len(rows) == 1 else 's'}, {len(rows) - done} open, {done} done"


# -- Tool factory -------------------------------------------------------------


def load_todo_tools(
    storage: TodoStorage,
    conversation_id: UUID,
    user_id: UUID,
    message_id: UUID | None = None,
    snapshot_callback: SnapshotCallback | None = None,
    *,
    markdown: bool = True,
    tz: tzinfo | str | None = None,
) -> list[BaseTool]:
    """Create persistent todo list tools.

    The ``snapshot_callback`` receives ``(todos, message_id_str)`` after
    each mutation so the host can emit real-time events (e.g., LangGraph's
    ``adispatch_custom_event``).

    ``markdown`` shapes what ``list_todos`` returns: a heading per list and
    checkboxes (the default), or plain lines -- ``Deployment:`` then
    ``- open: Deploy`` -- for a host whose model should not read markdown.

    :param tz: the zone of the person the listing is read to; when a todo was
        added and last changed is said in it. ``None`` says it in UTC.
    :ptype tz: tzinfo | str | None
    :raises ValueError: when ``tz`` is a name that is not a known zone
    """
    zone = resolve_zone(tz)
    tools: list[BaseTool] = []

    async def _snapshot(conv_id: UUID) -> list[dict[str, Any]]:
        todos = await storage.list_all(conv_id)
        if snapshot_callback:
            # convert at border: snapshot_callback Callable[..., str | None] contract
            msg_str = str(message_id) if message_id else None
            await snapshot_callback(todos, msg_str)
        return todos

    async def _not_found(tool_name: str, title: str, list_name: str | None) -> str:
        """Nothing matched: its own status, and what the conversation does hold, so the next call can name it."""
        todos = await storage.list_all(conversation_id)
        in_list = [t for t in todos if list_name is None or t["list_name"] == list_name]
        where = f' in "{list_name}"' if list_name is not None else ""
        if in_list:
            shown = ", ".join(
                f'"{t["title"]}" {todo_handle(t["todo_id"])}' + (" (done)" if t["is_completed"] else "")
                for t in in_list[:_LISTED_ON_A_MISS]
            )
            more = f"{len(in_list)} todos, the first {_LISTED_ON_A_MISS}: " if len(in_list) > _LISTED_ON_A_MISS else ""
            there = f"{'That list' if list_name is not None else 'This conversation'} has: {more}{shown}."
        elif todos:
            lists = ", ".join(f'"{name}"' for name in sorted({t["list_name"] for t in todos}))
            there = f"There is no such list. The lists here: {lists}."
        else:
            there = "This conversation has no todos."
        return tool_no_change(tool_name, f'no todo matching "{title}"{where}. {there}')

    async def _resolve(
        tool_name: str, todo_id: str | None, title: str | None, list_name: str | None
    ) -> tuple[dict[str, Any] | None, str | None]:
        """The one todo a call names, or the answer that says why there is not exactly one.

        By id when one is given; otherwise by title, which must match exactly one todo.
        """
        if todo_id is not None and todo_id.strip():
            found = _HANDLE.match(todo_id)
            if found is None:
                return (
                    None,
                    f"{TOOL_ERROR_PREFIX} {tool_name}: '{todo_id}' is not a todo id. Give the id in a [todo:<id>] mark.",
                )
            try:
                wanted = UUID(found.group(1))
            except ValueError:
                return (
                    None,
                    f"{TOOL_ERROR_PREFIX} {tool_name}: '{todo_id}' is not a todo id. Give the id in a [todo:<id>] mark.",
                )
            row = await storage.get(conversation_id, wanted)
            if row is None:
                return None, tool_no_change(
                    tool_name, f"no todo {todo_handle(wanted)} in this conversation. list_todos shows every todo's id."
                )
            return row, None
        text = (title or "").strip()
        if not text:
            return None, f"{TOOL_ERROR_PREFIX} {tool_name}: say which todo: give todo_id or title."
        named_list = await storage.resolve_list_name(conversation_id, list_name) if list_name else None
        rows = await storage.match_title(conversation_id, text, named_list)
        if len(rows) == 1:
            return rows[0], None
        if len(rows) > 1:
            lines = "\n".join(f"- {_line(r)}" for r in rows[:_LISTED_ON_A_MISS])
            more = (
                f"\n({len(rows)} match; the first {_LISTED_ON_A_MISS} are shown.)"
                if len(rows) > _LISTED_ON_A_MISS
                else ""
            )
            return None, tool_no_change(
                tool_name,
                f'"{text}" matches {len(rows)} todos, so nothing was changed. '
                f"Call again with the todo_id of the one you mean:\n{lines}{more}",
            )
        if named_list is not None:
            elsewhere = await storage.match_title(conversation_id, text, None)
            if elsewhere:
                lines = "\n".join(f"- {_line(r)}" for r in elsewhere[:_LISTED_ON_A_MISS])
                return None, tool_no_change(
                    tool_name,
                    f'no todo matching "{text}" in "{named_list}", so nothing was changed. '
                    f"It is on another list; call again with its todo_id:\n{lines}",
                )
        return None, await _not_found(tool_name, text, named_list)

    @tool("add_todo", args_schema=AddTodoInput)
    async def add_todo(title: str, list_name: str = DEFAULT_LIST) -> str:
        """Add a new todo item to a named todo list."""
        title = title.strip()
        if not title:
            return "[TOOL ERROR] add_todo: Title cannot be empty."
        try:
            list_name = await storage.resolve_list_name(conversation_id, list_name)
            row = await storage.add(conversation_id, user_id, title, list_name, message_id)
            todos = await _snapshot(conversation_id)
            in_list = [t for t in todos if t["list_name"] == list_name]
            return (
                f'Todo added: "{title}" {todo_handle(row["todo_id"])} to "{list_name}". '
                f"That list now has {_count(in_list)}."
            )
        except ValueError as ve:
            # The storage's word for "already on the list" (see ``TodoStorage.add``). Nothing
            # changed and nothing broke: its own status, so a caller that refuses a failed
            # tool for the rest of a turn does not refuse every later add_todo.
            return tool_no_change("add_todo", str(ve))
        except Exception as exc:
            log.error(
                "add_todo failed",
                extra={"extra_data": {"title": title, "error": str(exc)}},
            )
            return f"[TOOL ERROR] add_todo: {exc}"

    add_todo.description = (
        "Add an item to a todo list in this conversation. Use list_name to keep separate "
        "lists, e.g. 'Deployment Checklist' and 'Bug Fixes'. The person sees the list in the chat. "
        "Returns the new todo's [todo:<id>]."
    )
    tools.append(add_todo)

    @tool("complete_todo", args_schema=CompleteTodoInput)
    async def complete_todo(todo_id: str | None = None, title: str | None = None, list_name: str | None = None) -> str:
        """Mark a todo item as completed."""
        try:
            row, answer = await _resolve("complete_todo", todo_id, title, list_name)
            if row is None:
                return answer or ""
            if row["is_completed"]:
                return tool_no_change("complete_todo", f"{_line(row)} was already done.")
            await storage.complete(_uuid(row["todo_id"]), message_id)
            await _snapshot(conversation_id)
            return f'Completed {todo_handle(row["todo_id"])} "{row["title"]}" in "{row["list_name"]}"; it was open.'
        except Exception as exc:
            log.error(
                "complete_todo failed",
                extra={"extra_data": {"title": title, "todo_id": todo_id, "error": str(exc)}},
            )
            return f"[TOOL ERROR] complete_todo: {exc}"

    complete_todo.description = (
        "Mark a todo done, by its todo_id or its title. A title must fit exactly one todo; "
        "when it fits several, nothing changes and their ids are listed."
    )
    tools.append(complete_todo)

    @tool("update_todo", args_schema=UpdateTodoInput)
    async def update_todo(
        todo_id: str | None = None,
        title: str | None = None,
        new_title: str | None = None,
        is_completed: bool | None = None,
        new_list_name: str | None = None,
        list_name: str | None = None,
    ) -> str:
        """Rename a todo, mark it done or not done, or move it to another list."""
        if new_title is None and is_completed is None and new_list_name is None:
            return (
                "[TOOL ERROR] update_todo: nothing to change was given. "
                "Pass new_title, is_completed, new_list_name, or more than one."
            )
        if new_title is not None and not new_title.strip():
            return "[TOOL ERROR] update_todo: new_title is blank. Give the new title, or leave new_title out."
        if new_list_name is not None and not new_list_name.strip():
            return "[TOOL ERROR] update_todo: new_list_name is blank. Give the list, or leave new_list_name out."
        try:
            row, answer = await _resolve("update_todo", todo_id, title, list_name)
            if row is None:
                return answer or ""
            old_title, old_list, was_done = row["title"], row["list_name"], bool(row["is_completed"])
            target_list = (
                await storage.resolve_list_name(conversation_id, new_list_name.strip())
                if new_list_name is not None
                else old_list
            )
            renamed = new_title.strip() if new_title is not None else old_title
            changes: list[str] = []
            if renamed != old_title:
                changes.append(f'title "{old_title}" -> "{renamed}"')
            if is_completed is not None and is_completed != was_done:
                changes.append(f"{_state(row)} -> {'done' if is_completed else 'open'}")
            if target_list != old_list:
                changes.append(f'moved from "{old_list}" to "{target_list}"')
            if not changes:
                return tool_no_change("update_todo", f"{_line(row)} is already that.")
            # The todo itself cannot clash: it is only looked for under a title or a list it does not have.
            if renamed.lower() != old_title.lower() or target_list != old_list:
                clash = [
                    t
                    for t in await storage.list_all(conversation_id)
                    if t["list_name"] == target_list and t["title"].lower() == renamed.lower()
                ]
                if clash:
                    return tool_no_change(
                        "update_todo",
                        f'"{target_list}" already has {_line(clash[0])}, so nothing was changed. '
                        "Give another title, or remove one of them.",
                    )
            await storage.update(
                _uuid(row["todo_id"]),
                message_id,
                new_title=renamed if renamed != old_title else None,
                is_completed=is_completed if is_completed is not None and is_completed != was_done else None,
                new_list_name=target_list if target_list != old_list else None,
            )
            await _snapshot(conversation_id)
            return f'Updated {todo_handle(row["todo_id"])} in "{target_list}": {"; ".join(changes)}.'
        except Exception as exc:
            log.error(
                "update_todo failed",
                extra={"extra_data": {"title": title, "todo_id": todo_id, "error": str(exc)}},
            )
            return f"[TOOL ERROR] update_todo: {exc}"

    update_todo.description = (
        "Rename a todo, mark it done or not done, or move it to another list, by its todo_id or its title. "
        "A title must fit exactly one todo. Returns what changed, old -> new."
    )
    tools.append(update_todo)

    @tool("remove_todo", args_schema=RemoveTodoInput)
    async def remove_todo(todo_id: str | None = None, title: str | None = None, list_name: str | None = None) -> str:
        """Remove a todo item from the list."""
        try:
            row, answer = await _resolve("remove_todo", todo_id, title, list_name)
            if row is None:
                return answer or ""
            await storage.remove(_uuid(row["todo_id"]))
            todos = await _snapshot(conversation_id)
            left = [t for t in todos if t["list_name"] == row["list_name"]]
            return (
                f'Removed {todo_handle(row["todo_id"])} "{row["title"]}" from "{row["list_name"]}"; '
                f"it was {_state(row)}. It cannot be brought back. That list now has {_count(left)}."
            )
        except Exception as exc:
            log.error(
                "remove_todo failed",
                extra={"extra_data": {"title": title, "todo_id": todo_id, "error": str(exc)}},
            )
            return f"[TOOL ERROR] remove_todo: {exc}"

    remove_todo.description = (
        "Delete a todo, by its todo_id or its title. It cannot be undone. A title must fit exactly one todo; "
        "when it fits several, nothing is deleted and their ids are listed. To tick one off, use complete_todo."
    )
    tools.append(remove_todo)

    @tool("list_todos", args_schema=ListTodosInput)
    async def list_todos(list_name: str | None = None) -> str:
        """List the todo items in the conversation."""
        try:
            todos = await storage.list_all(conversation_id)
            if not todos:
                return "No todos in this conversation."
            if list_name:
                named = await storage.resolve_list_name(conversation_id, list_name)
                shown = [t for t in todos if t["list_name"] == named]
                if not shown:
                    lists = ", ".join(f'"{name}"' for name in sorted({t["list_name"] for t in todos}))
                    return tool_no_change("list_todos", f'there is no list "{list_name}". The lists here: {lists}.')
            else:
                shown = todos

            # Group by list_name
            groups: dict[str, list[dict[str, Any]]] = {}
            for t in shown:
                groups.setdefault(t["list_name"], []).append(t)

            lines: list[str] = [f"{_count(shown)}, in {len(groups)} list{'' if len(groups) == 1 else 's'}:"]
            for ln in sorted(groups.keys()):
                head = f"{ln} ({_count(groups[ln])})"
                lines.append(f"### {head}" if markdown else f"{head}:")
                for t in groups[ln]:
                    mark = ("[x]" if t["is_completed"] else "[ ]") if markdown else f"{_state(t)}:"
                    dates = []
                    added = _when(t.get("date_created"), zone)
                    if added:
                        dates.append(f"added {added}")
                    changed = _when(t.get("date_updated"), zone)
                    if changed and changed != added:
                        dates.append(f"changed {changed}")
                    when = f" ({'; '.join(dates)})" if dates else ""
                    lines.append(f"- {mark} {one_line(str(t['title']))} {todo_handle(t['todo_id'])}{when}")
                lines.append("")

            return "\n".join(lines).rstrip("\n")

        except Exception as exc:
            log.error(
                "list_todos failed",
                extra={"extra_data": {"error": str(exc)}},
            )
            return f"[TOOL ERROR] list_todos: {exc}"

    list_todos.description = (
        "List the todos in this conversation, or in one list: each with its [todo:<id>], whether it is done, "
        "and when it was added."
    )
    tools.append(list_todos)

    return tools
