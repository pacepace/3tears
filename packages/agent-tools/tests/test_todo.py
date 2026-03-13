"""Tests for todo tools module."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.agent.tools.todo import (
    AddTodoInput,
    CompleteTodoInput,
    RemoveTodoInput,
    SnapshotCallback,
    TodoStorage,
    UpdateTodoInput,
    load_todo_tools,
)


# -- In-memory storage for testing --------------------------------------------


class InMemoryTodoStorage:
    """Simple in-memory TodoStorage for unit tests."""

    def __init__(self) -> None:
        self._todos: list[dict[str, Any]] = []
        self._counter = 0

    async def resolve_list_name(self, conversation_id: UUID, list_name: str) -> str:
        """Resolve to canonical casing."""
        for t in self._todos:
            if t["list_name"].lower() == list_name.lower():
                return t["list_name"]
        return list_name

    async def find_by_title(
        self, conversation_id: UUID, title: str, list_name: str,
    ) -> dict[str, Any] | None:
        """Find by exact title first, then partial."""
        for t in self._todos:
            if t["title"] == title and t["list_name"].lower() == list_name.lower():
                return t
        for t in self._todos:
            if title.lower() in t["title"].lower() and t["list_name"].lower() == list_name.lower():
                return t
        return None

    async def add(
        self, conversation_id: UUID, user_id: UUID, title: str,
        list_name: str, message_id: UUID | None,
    ) -> dict[str, Any]:
        """Add a todo, raise on duplicate."""
        for t in self._todos:
            if t["title"].lower() == title.lower() and t["list_name"].lower() == list_name.lower():
                raise ValueError(f'Todo "{title}" already exists in "{list_name}".')
        self._counter += 1
        todo = {
            "todo_id": uuid4(),
            "title": title,
            "is_completed": False,
            "sort_order": self._counter,
            "list_name": list_name,
        }
        self._todos.append(todo)
        return todo

    async def complete(self, todo_id: UUID, message_id: UUID | None) -> None:
        """Mark completed."""
        for t in self._todos:
            if t["todo_id"] == todo_id:
                t["is_completed"] = True

    async def update(
        self, todo_id: UUID, message_id: UUID | None,
        new_title: str | None = None, is_completed: bool | None = None,
    ) -> None:
        """Update title/completion."""
        for t in self._todos:
            if t["todo_id"] == todo_id:
                if new_title is not None:
                    t["title"] = new_title.strip()
                if is_completed is not None:
                    t["is_completed"] = is_completed

    async def remove(self, todo_id: UUID) -> None:
        """Delete a todo."""
        self._todos = [t for t in self._todos if t["todo_id"] != todo_id]

    async def list_all(self, conversation_id: UUID) -> list[dict[str, Any]]:
        """Return all todos."""
        return [
            {
                "todo_id": str(t["todo_id"]),
                "title": t["title"],
                "is_completed": t["is_completed"],
                "sort_order": t["sort_order"],
                "list_name": t["list_name"],
            }
            for t in sorted(self._todos, key=lambda x: (x["list_name"], x["sort_order"]))
        ]


# -- Protocol compliance -----------------------------------------------------


class TestTodoStorageProtocol:
    def test_in_memory_is_protocol_compliant(self):
        assert isinstance(InMemoryTodoStorage(), TodoStorage)


# -- Input schemas ------------------------------------------------------------


class TestInputSchemas:
    def test_add_todo_defaults(self):
        inp = AddTodoInput(title="Buy milk")
        assert inp.list_name == "Todo List"

    def test_complete_todo_defaults(self):
        inp = CompleteTodoInput(title="Buy milk")
        assert inp.list_name == "Todo List"

    def test_update_todo_optional_fields(self):
        inp = UpdateTodoInput(title="Buy milk")
        assert inp.new_title is None
        assert inp.is_completed is None

    def test_remove_todo_defaults(self):
        inp = RemoveTodoInput(title="Buy milk")
        assert inp.list_name == "Todo List"


# -- Tool factory -------------------------------------------------------------


@pytest.fixture
def storage() -> InMemoryTodoStorage:
    return InMemoryTodoStorage()


@pytest.fixture
def conv_id() -> UUID:
    return uuid4()


@pytest.fixture
def user_id() -> UUID:
    return uuid4()


@pytest.fixture
def tools(storage: InMemoryTodoStorage, conv_id: UUID, user_id: UUID) -> list:
    return load_todo_tools(
        storage=storage,
        conversation_id=conv_id,
        user_id=user_id,
    )


def _find_tool(tools: list, name: str):
    for t in tools:
        if t.name == name:
            return t
    raise KeyError(f"Tool {name} not found")


class TestLoadTodoTools:
    def test_creates_five_tools(self, tools: list):
        assert len(tools) == 5
        names = {t.name for t in tools}
        assert names == {"add_todo", "complete_todo", "update_todo", "remove_todo", "list_todos"}

    async def test_add_todo(self, tools: list):
        add = _find_tool(tools, "add_todo")
        result = await add.ainvoke({"title": "Buy groceries"})
        assert "Buy groceries" in result
        assert "Todo List" in result

    async def test_add_todo_duplicate(self, tools: list):
        add = _find_tool(tools, "add_todo")
        await add.ainvoke({"title": "Buy milk"})
        result = await add.ainvoke({"title": "Buy milk"})
        assert "already exists" in result

    async def test_add_todo_empty_title(self, tools: list):
        add = _find_tool(tools, "add_todo")
        result = await add.ainvoke({"title": "   "})
        assert "TOOL ERROR" in result

    async def test_complete_todo(self, tools: list):
        add = _find_tool(tools, "add_todo")
        complete = _find_tool(tools, "complete_todo")
        await add.ainvoke({"title": "Task one"})
        result = await complete.ainvoke({"title": "Task one"})
        assert "Completed" in result

    async def test_complete_nonexistent(self, tools: list):
        complete = _find_tool(tools, "complete_todo")
        result = await complete.ainvoke({"title": "Does not exist"})
        assert "TOOL ERROR" in result

    async def test_update_todo_rename(self, tools: list):
        add = _find_tool(tools, "add_todo")
        update = _find_tool(tools, "update_todo")
        await add.ainvoke({"title": "Old name"})
        result = await update.ainvoke({"title": "Old name", "new_title": "New name"})
        assert "Updated" in result
        assert "New name" in result

    async def test_update_nonexistent(self, tools: list):
        update = _find_tool(tools, "update_todo")
        result = await update.ainvoke({"title": "Nope"})
        assert "TOOL ERROR" in result

    async def test_remove_todo(self, tools: list):
        add = _find_tool(tools, "add_todo")
        remove = _find_tool(tools, "remove_todo")
        await add.ainvoke({"title": "Temp item"})
        result = await remove.ainvoke({"title": "Temp item"})
        assert "Removed" in result

    async def test_remove_nonexistent(self, tools: list):
        remove = _find_tool(tools, "remove_todo")
        result = await remove.ainvoke({"title": "Ghost"})
        assert "TOOL ERROR" in result

    async def test_list_todos_empty(self, tools: list):
        lst = _find_tool(tools, "list_todos")
        result = await lst.ainvoke({})
        assert "No todos" in result

    async def test_list_todos_grouped(self, tools: list):
        add = _find_tool(tools, "add_todo")
        lst = _find_tool(tools, "list_todos")
        await add.ainvoke({"title": "A", "list_name": "Work"})
        await add.ainvoke({"title": "B", "list_name": "Home"})
        result = await lst.ainvoke({})
        assert "### Work" in result
        assert "### Home" in result
        assert "[ ] A" in result
        assert "[ ] B" in result

    async def test_list_shows_completion(self, tools: list):
        add = _find_tool(tools, "add_todo")
        complete = _find_tool(tools, "complete_todo")
        lst = _find_tool(tools, "list_todos")
        await add.ainvoke({"title": "Done item"})
        await complete.ainvoke({"title": "Done item"})
        result = await lst.ainvoke({})
        assert "[x] Done item" in result

    async def test_custom_list_name(self, tools: list):
        add = _find_tool(tools, "add_todo")
        await add.ainvoke({"title": "Deploy v2", "list_name": "Release Checklist"})
        lst = _find_tool(tools, "list_todos")
        result = await lst.ainvoke({})
        assert "Release Checklist" in result


# -- Snapshot callback --------------------------------------------------------


class TestSnapshotCallback:
    async def test_callback_fires_on_add(self):
        storage = InMemoryTodoStorage()
        snapshots: list[tuple] = []

        async def on_snapshot(todos: list[dict[str, Any]], msg_id: str | None) -> None:
            snapshots.append((todos, msg_id))

        msg_id = uuid4()
        tools = load_todo_tools(
            storage=storage,
            conversation_id=uuid4(),
            user_id=uuid4(),
            message_id=msg_id,
            snapshot_callback=on_snapshot,
        )
        add = _find_tool(tools, "add_todo")
        await add.ainvoke({"title": "Tracked item"})
        assert len(snapshots) == 1
        assert len(snapshots[0][0]) == 1
        assert snapshots[0][1] == str(msg_id)

    async def test_no_callback_when_none(self):
        storage = InMemoryTodoStorage()
        tools = load_todo_tools(
            storage=storage,
            conversation_id=uuid4(),
            user_id=uuid4(),
            snapshot_callback=None,
        )
        add = _find_tool(tools, "add_todo")
        # Should not raise
        result = await add.ainvoke({"title": "No callback item"})
        assert "added" in result.lower()
