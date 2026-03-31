"""Todo tools -- LangChain tools for managing conversation-scoped todo lists.

Provides ``load_todo_tools`` as a factory. Callers supply a ``TodoStorage``
implementation (e.g., PostgreSQL-backed) and an optional ``snapshot_callback``
for real-time updates (e.g., LangGraph custom events).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from threetears.observe import get_logger

log = get_logger(__name__)


# -- Storage protocol ---------------------------------------------------------


@runtime_checkable
class TodoStorage(Protocol):
    """Protocol for todo persistence backends."""

    async def add(
        self,
        conversation_id: UUID,
        user_id: UUID,
        title: str,
        list_name: str,
        message_id: UUID | None,
    ) -> dict[str, Any]:
        """Add a todo item. Returns dict with at least ``todo_id`` and ``title``.

        Raises ValueError if the title is a duplicate in the same list.
        """
        ...

    async def find_by_title(
        self,
        conversation_id: UUID,
        title: str,
        list_name: str,
    ) -> dict[str, Any] | None:
        """Find a todo by title (exact then fuzzy). Returns dict or None."""
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
    ) -> None:
        """Update a todo's title and/or completion state."""
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


class AddTodoInput(BaseModel):
    """Input for add_todo tool."""

    title: str = Field(description="Title of the todo item to add")
    list_name: str = Field(
        default="Todo List",
        description="Name of the todo list to add to (e.g. 'Deployment Checklist')",
    )


class CompleteTodoInput(BaseModel):
    """Input for complete_todo tool."""

    title: str = Field(description="Title (or partial title) of the todo to mark complete")
    list_name: str = Field(default="Todo List", description="Name of the todo list")


class UpdateTodoInput(BaseModel):
    """Input for update_todo tool."""

    title: str = Field(description="Current title of the todo to update")
    new_title: str | None = Field(default=None, description="New title (if renaming)")
    is_completed: bool | None = Field(default=None, description="New completion state")
    list_name: str = Field(default="Todo List", description="Name of the todo list")


class RemoveTodoInput(BaseModel):
    """Input for remove_todo tool."""

    title: str = Field(description="Title (or partial title) of the todo to remove")
    list_name: str = Field(default="Todo List", description="Name of the todo list")


# -- Tool factory -------------------------------------------------------------


def load_todo_tools(
    storage: TodoStorage,
    conversation_id: UUID,
    user_id: UUID,
    message_id: UUID | None = None,
    snapshot_callback: SnapshotCallback | None = None,
) -> list[BaseTool]:
    """Create persistent todo list tools.

    The ``snapshot_callback`` receives ``(todos, message_id_str)`` after
    each mutation so the host can emit real-time events (e.g., LangGraph's
    ``adispatch_custom_event``).
    """
    tools: list[BaseTool] = []

    async def _snapshot(conv_id: UUID) -> list[dict[str, Any]]:
        todos = await storage.list_all(conv_id)
        if snapshot_callback:
            msg_str = str(message_id) if message_id else None
            await snapshot_callback(todos, msg_str)
        return todos

    @tool("add_todo", args_schema=AddTodoInput)
    async def add_todo(title: str, list_name: str = "Todo List") -> str:
        """Add a new todo item to a named todo list."""
        title = title.strip()
        if not title:
            return "[TOOL ERROR] add_todo: Title cannot be empty."
        try:
            list_name = await storage.resolve_list_name(conversation_id, list_name)
            await storage.add(conversation_id, user_id, title, list_name, message_id)
            todos = await _snapshot(conversation_id)
            return f'Todo added: "{title}" to "{list_name}" ({len(todos)} total)'
        except ValueError as ve:
            return str(ve)
        except Exception as exc:
            log.error(
                "add_todo failed",
                extra={"extra_data": {"title": title, "error": str(exc)}},
            )
            return f"[TOOL ERROR] add_todo: {exc}"

    add_todo.description = (
        "Add a new item to a named todo list in the conversation. "
        "Use list_name to organize items into separate lists (e.g. 'Deployment Checklist', 'Bug Fixes'). "
        "The item will be visible as an interactive checklist in the chat."
    )
    tools.append(add_todo)

    @tool("complete_todo", args_schema=CompleteTodoInput)
    async def complete_todo(title: str, list_name: str = "Todo List") -> str:
        """Mark a todo item as completed by title."""
        try:
            list_name = await storage.resolve_list_name(conversation_id, list_name)
            row = await storage.find_by_title(conversation_id, title, list_name)
            if not row:
                return f'[TOOL ERROR] complete_todo: No todo matching "{title}" in "{list_name}" found.'
            await storage.complete(row["todo_id"], message_id)
            await _snapshot(conversation_id)
            return f'Completed: "{row["title"]}"'
        except Exception as exc:
            log.error(
                "complete_todo failed",
                extra={"extra_data": {"title": title, "error": str(exc)}},
            )
            return f"[TOOL ERROR] complete_todo: {exc}"

    complete_todo.description = (
        "Mark a todo item as completed. Matches by title (exact or partial) within the specified list. "
        "The checklist in the chat will update to show the item checked off."
    )
    tools.append(complete_todo)

    @tool("update_todo", args_schema=UpdateTodoInput)
    async def update_todo(
        title: str,
        new_title: str | None = None,
        is_completed: bool | None = None,
        list_name: str = "Todo List",
    ) -> str:
        """Update a todo item's title or completion status."""
        try:
            list_name = await storage.resolve_list_name(conversation_id, list_name)
            row = await storage.find_by_title(conversation_id, title, list_name)
            if not row:
                return f'[TOOL ERROR] update_todo: No todo matching "{title}" in "{list_name}" found.'
            await storage.update(row["todo_id"], message_id, new_title=new_title, is_completed=is_completed)
            await _snapshot(conversation_id)
            display_title = new_title.strip() if new_title else row["title"]
            return f'Updated: "{display_title}"'
        except Exception as exc:
            log.error(
                "update_todo failed",
                extra={"extra_data": {"title": title, "error": str(exc)}},
            )
            return f"[TOOL ERROR] update_todo: {exc}"

    update_todo.description = (
        "Update an existing todo item — change its title and/or completion status. "
        "Matches by current title (exact or partial)."
    )
    tools.append(update_todo)

    @tool("remove_todo", args_schema=RemoveTodoInput)
    async def remove_todo(title: str, list_name: str = "Todo List") -> str:
        """Remove a todo item from the list."""
        try:
            list_name = await storage.resolve_list_name(conversation_id, list_name)
            row = await storage.find_by_title(conversation_id, title, list_name)
            if not row:
                return f'[TOOL ERROR] remove_todo: No todo matching "{title}" in "{list_name}" found.'
            await storage.remove(row["todo_id"])
            todos = await _snapshot(conversation_id)
            return f'Removed: "{row["title"]}" ({len(todos)} remaining)'
        except Exception as exc:
            log.error(
                "remove_todo failed",
                extra={"extra_data": {"title": title, "error": str(exc)}},
            )
            return f"[TOOL ERROR] remove_todo: {exc}"

    remove_todo.description = (
        "Remove a todo item from the conversation's persistent checklist. Matches by title (exact or partial)."
    )
    tools.append(remove_todo)

    @tool("list_todos")
    async def list_todos() -> str:
        """List all todo items in the conversation."""
        try:
            todos = await storage.list_all(conversation_id)
            if not todos:
                return "No todos in this conversation."

            # Group by list_name
            groups: dict[str, list[dict[str, Any]]] = {}
            for t in todos:
                groups.setdefault(t["list_name"], []).append(t)

            lines: list[str] = []
            for ln in sorted(groups.keys()):
                lines.append(f"### {ln}")
                for t in groups[ln]:
                    check = "[x]" if t["is_completed"] else "[ ]"
                    lines.append(f"- {check} {t['title']}")
                lines.append("")

            return "\n".join(lines)

        except Exception as exc:
            log.error(
                "list_todos failed",
                extra={"extra_data": {"error": str(exc)}},
            )
            return f"[TOOL ERROR] list_todos: {exc}"

    list_todos.description = (
        "List all todo items in the conversation's persistent checklist, showing their completion status."
    )
    tools.append(list_todos)

    return tools
