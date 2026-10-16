"""Tests for todo tools module."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.agent.tools.todo import (
    AddTodoInput,
    CompleteTodoInput,
    RemoveTodoInput,
    TodoStorage,
    UpdateTodoInput,
    load_todo_tools,
)
from threetears.agent.tools.utils import tool_result_status


_ADDED = datetime(2026, 10, 9, 22, 0, tzinfo=UTC)
_CHANGED = datetime(2026, 10, 10, 15, 30, tzinfo=UTC)


# -- In-memory storage for testing --------------------------------------------


class InMemoryTodoStorage:
    """Simple in-memory TodoStorage for unit tests."""

    def __init__(self) -> None:
        self.todos: list[dict[str, Any]] = []
        self._counter = 0

    async def resolve_list_name(self, conversation_id: UUID, list_name: str) -> str:
        """Resolve to canonical casing."""
        for t in self.todos:
            if t["list_name"].lower() == list_name.lower():
                return t["list_name"]
        return list_name

    async def get(self, conversation_id: UUID, todo_id: UUID) -> dict[str, Any] | None:
        """The todo with this id in this conversation."""
        return next(
            (dict(t) for t in self.todos if t["todo_id"] == todo_id and t["conversation_id"] == conversation_id),
            None,
        )

    async def match_title(self, conversation_id: UUID, title: str, list_name: str | None) -> list[dict[str, Any]]:
        """Whole title in any case first, then the titles that hold it; the text taken literally."""
        mine = [
            t
            for t in sorted(self.todos, key=lambda x: (x["list_name"], x["sort_order"]))
            if t["conversation_id"] == conversation_id
            and (list_name is None or t["list_name"].lower() == list_name.lower())
        ]
        exact = [dict(t) for t in mine if t["title"].lower() == title.lower()]
        return exact or [dict(t) for t in mine if title.lower() in t["title"].lower()]

    async def add(
        self,
        conversation_id: UUID,
        user_id: UUID,
        title: str,
        list_name: str,
        message_id: UUID | None,
    ) -> dict[str, Any]:
        """Add a todo, raise on duplicate."""
        for t in self.todos:
            if t["title"].lower() == title.lower() and t["list_name"].lower() == list_name.lower():
                raise ValueError(f'Todo "{title}" already exists in "{list_name}".')
        self._counter += 1
        todo = {
            "todo_id": uuid4(),
            "conversation_id": conversation_id,
            "title": title,
            "is_completed": False,
            "sort_order": self._counter,
            "list_name": list_name,
            "date_created": _ADDED,
            "date_updated": _ADDED,
        }
        self.todos.append(todo)
        return dict(todo)

    async def complete(self, todo_id: UUID, message_id: UUID | None) -> None:
        """Mark completed."""
        for t in self.todos:
            if t["todo_id"] == todo_id:
                t["is_completed"] = True
                t["date_updated"] = _CHANGED

    async def update(
        self,
        todo_id: UUID,
        message_id: UUID | None,
        new_title: str | None = None,
        is_completed: bool | None = None,
        new_list_name: str | None = None,
    ) -> None:
        """Update title/completion/list."""
        for t in self.todos:
            if t["todo_id"] == todo_id:
                if new_title is not None:
                    t["title"] = new_title.strip()
                if is_completed is not None:
                    t["is_completed"] = is_completed
                if new_list_name is not None:
                    self._counter += 1
                    t["list_name"] = new_list_name
                    t["sort_order"] = self._counter
                t["date_updated"] = _CHANGED

    async def remove(self, todo_id: UUID) -> None:
        """Delete a todo."""
        self.todos = [t for t in self.todos if t["todo_id"] != todo_id]

    async def list_all(self, conversation_id: UUID) -> list[dict[str, Any]]:
        """Return all todos."""
        return [
            {
                "todo_id": str(t["todo_id"]),
                "title": t["title"],
                "is_completed": t["is_completed"],
                "sort_order": t["sort_order"],
                "list_name": t["list_name"],
                "date_created": t["date_created"].isoformat(),
                "date_updated": t["date_updated"].isoformat(),
            }
            for t in sorted(self.todos, key=lambda x: (x["list_name"], x["sort_order"]))
            if t["conversation_id"] == conversation_id
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

    def test_a_todo_is_named_in_every_list_unless_one_is_given(self):
        for schema in (CompleteTodoInput, UpdateTodoInput, RemoveTodoInput):
            inp = schema(title="Buy milk")
            assert inp.list_name is None and inp.todo_id is None

    def test_update_todo_optional_fields(self):
        inp = UpdateTodoInput(title="Buy milk")
        assert inp.new_title is None
        assert inp.is_completed is None
        assert inp.new_list_name is None


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


async def _add(tools: list, title: str, list_name: str = "Todo List") -> str:
    """Add a todo and hand back its id, read from the result's own mark."""
    result = await _find_tool(tools, "add_todo").ainvoke({"title": title, "list_name": list_name})
    found = re.search(r"\[todo:([0-9a-f-]{36})\]", result)
    assert found, result
    return found.group(1)


def _titles(storage: InMemoryTodoStorage) -> list[str]:
    return sorted(t["title"] for t in storage.todos)


class TestLoadTodoTools:
    def test_creates_five_tools(self, tools: list):
        assert len(tools) == 5
        names = {t.name for t in tools}
        assert names == {"add_todo", "complete_todo", "update_todo", "remove_todo", "list_todos"}

    async def test_add_todo_names_the_new_todo_and_the_list(self, tools: list, storage: InMemoryTodoStorage):
        result = await _find_tool(tools, "add_todo").ainvoke({"title": "Buy groceries"})
        todo_id = storage.todos[0]["todo_id"]
        assert result == (
            f'Todo added: "Buy groceries" [todo:{todo_id}] to "Todo List". That list now has 1 todo, 1 open, 0 done.'
        )

    async def test_add_todo_duplicate(self, tools: list):
        add = _find_tool(tools, "add_todo")
        await add.ainvoke({"title": "Buy milk"})
        result = await add.ainvoke({"title": "Buy milk"})
        assert result == '[NO CHANGE] add_todo: Todo "Buy milk" already exists in "Todo List".'
        assert tool_result_status(result) == "no_change", "a duplicate changed nothing and is not a failure"

    async def test_an_add_after_a_duplicate_still_adds(self, tools: list):
        add = _find_tool(tools, "add_todo")
        await add.ainvoke({"title": "Buy milk"})
        await add.ainvoke({"title": "Buy milk"})
        assert tool_result_status(await add.ainvoke({"title": "Buy eggs"})) == "ok"

    async def test_a_storage_failure_is_an_error_not_a_no_change(self, tools: list, storage: InMemoryTodoStorage):
        async def _down(*args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("database is down")

        storage.add = _down  # type: ignore[method-assign]
        result = await _find_tool(tools, "add_todo").ainvoke({"title": "Buy milk"})
        assert result == "[TOOL ERROR] add_todo: database is down" and tool_result_status(result) == "error"

    async def test_add_todo_empty_title(self, tools: list):
        add = _find_tool(tools, "add_todo")
        result = await add.ainvoke({"title": "   "})
        assert result.startswith("[TOOL ERROR] add_todo:") and tool_result_status(result) == "error"

    async def test_complete_todo_says_which_and_what_it_was(self, tools: list):
        todo_id = await _add(tools, "Task one")
        result = await _find_tool(tools, "complete_todo").ainvoke({"title": "Task one"})
        assert result == f'Completed [todo:{todo_id}] "Task one" in "Todo List"; it was open.'

    async def test_completing_a_done_todo_changes_nothing(self, tools: list):
        todo_id = await _add(tools, "Task one")
        complete = _find_tool(tools, "complete_todo")
        await complete.ainvoke({"todo_id": todo_id})
        result = await complete.ainvoke({"todo_id": todo_id})
        assert (
            result == f'[NO CHANGE] complete_todo: [todo:{todo_id}] "Task one" in "Todo List" (done) was already done.'
        )

    async def test_complete_nonexistent(self, tools: list):
        complete = _find_tool(tools, "complete_todo")
        result = await complete.ainvoke({"title": "Does not exist"})
        assert result == '[NO CHANGE] complete_todo: no todo matching "Does not exist". This conversation has no todos.'
        assert tool_result_status(result) == "no_change"

    @pytest.mark.parametrize("tool_name", ["complete_todo", "update_todo", "remove_todo"])
    async def test_a_miss_lists_what_the_list_does_hold(self, tools: list, tool_name: str):
        milk = await _add(tools, "Buy milk")
        bank = await _add(tools, "Call the bank")
        await _find_tool(tools, "complete_todo").ainvoke({"title": "Buy milk"})
        result = await _find_tool(tools, tool_name).ainvoke(
            {"title": "Ghost", "list_name": "Todo List", "is_completed": True}
        )
        assert result == (
            f'[NO CHANGE] {tool_name}: no todo matching "Ghost" in "Todo List". '
            f'That list has: "Buy milk" [todo:{milk}] (done), "Call the bank" [todo:{bank}].'
        )

    async def test_a_miss_in_a_list_that_is_not_there_names_the_lists_that_are(self, tools: list):
        await _add(tools, "Deploy", "Release")
        await _add(tools, "Buy milk")
        result = await _find_tool(tools, "remove_todo").ainvoke({"title": "Ghost", "list_name": "Shipping"})
        assert result == (
            '[NO CHANGE] remove_todo: no todo matching "Ghost" in "Shipping". '
            'There is no such list. The lists here: "Release", "Todo List".'
        )

    async def test_a_todo_on_another_list_is_named_where_it_is_and_nothing_changes(
        self, tools: list, storage: InMemoryTodoStorage
    ):
        deploy = await _add(tools, "Deploy", "Release")
        result = await _find_tool(tools, "remove_todo").ainvoke({"title": "Deploy", "list_name": "Todo List"})
        assert result == (
            '[NO CHANGE] remove_todo: no todo matching "Deploy" in "Todo List", so nothing was changed. '
            f'It is on another list; call again with its todo_id:\n- [todo:{deploy}] "Deploy" in "Release" (open)'
        )
        assert _titles(storage) == ["Deploy"]

    async def test_a_long_list_is_counted_and_its_first_titles_named(self, tools: list):
        for n in range(23):
            await _add(tools, f"Task {n:02d}")
        result = await _find_tool(tools, "complete_todo").ainvoke({"title": "Ghost"})
        assert 'This conversation has: 23 todos, the first 20: "Task 00" [todo:' in result
        assert '"Task 19" [todo:' in result and "Task 20" not in result

    async def test_update_todo_says_old_and_new(self, tools: list):
        todo_id = await _add(tools, "Old name")
        result = await _find_tool(tools, "update_todo").ainvoke(
            {"title": "Old name", "new_title": "New name", "is_completed": True}
        )
        assert result == f'Updated [todo:{todo_id}] in "Todo List": title "Old name" -> "New name"; open -> done.'

    async def test_update_todo_moves_a_todo_to_another_list(self, tools: list, storage: InMemoryTodoStorage):
        todo_id = await _add(tools, "Deploy")
        await _add(tools, "Tag", "Release")
        result = await _find_tool(tools, "update_todo").ainvoke({"todo_id": todo_id, "new_list_name": "release"})
        assert result == f'Updated [todo:{todo_id}] in "Release": moved from "Todo List" to "Release".'
        assert next(t for t in storage.todos if t["title"] == "Deploy")["list_name"] == "Release"

    async def test_update_nonexistent(self, tools: list):
        update = _find_tool(tools, "update_todo")
        result = await update.ainvoke({"title": "Nope", "new_title": "Yes"})
        assert result.startswith('[NO CHANGE] update_todo: no todo matching "Nope"')

    async def test_an_update_that_names_no_change_is_an_error(self, tools: list):
        await _add(tools, "Old name")
        result = await _find_tool(tools, "update_todo").ainvoke({"title": "Old name"})
        assert result.startswith("[TOOL ERROR] update_todo: nothing to change was given")

    async def test_a_blank_new_title_is_refused_and_the_todo_keeps_its_title(
        self, tools: list, storage: InMemoryTodoStorage
    ):
        await _add(tools, "Old name")
        result = await _find_tool(tools, "update_todo").ainvoke({"title": "Old name", "new_title": "   "})
        assert result.startswith("[TOOL ERROR] update_todo: new_title is blank")
        assert _titles(storage) == ["Old name"]

    async def test_a_rename_onto_another_todos_title_is_refused(self, tools: list, storage: InMemoryTodoStorage):
        await _add(tools, "Buy milk")
        eggs = await _add(tools, "Buy eggs")
        result = await _find_tool(tools, "update_todo").ainvoke({"title": "Buy milk", "new_title": "buy EGGS"})
        assert result == (
            f'[NO CHANGE] update_todo: "Todo List" already has [todo:{eggs}] "Buy eggs" in "Todo List" (open), '
            "so nothing was changed. Give another title, or remove one of them."
        )
        assert _titles(storage) == ["Buy eggs", "Buy milk"]

    async def test_a_rename_that_only_changes_case_goes_through(self, tools: list, storage: InMemoryTodoStorage):
        """Good traffic past the clash guard: a todo is not a clash with itself."""
        todo_id = await _add(tools, "buy milk")
        result = await _find_tool(tools, "update_todo").ainvoke({"todo_id": todo_id, "new_title": "Buy milk"})
        assert result == f'Updated [todo:{todo_id}] in "Todo List": title "buy milk" -> "Buy milk".'
        assert _titles(storage) == ["Buy milk"]

    async def test_an_update_to_what_it_already_is_changes_nothing(self, tools: list):
        todo_id = await _add(tools, "Same")
        result = await _find_tool(tools, "update_todo").ainvoke(
            {"todo_id": todo_id, "new_title": "Same", "is_completed": False}
        )
        assert result == f'[NO CHANGE] update_todo: [todo:{todo_id}] "Same" in "Todo List" (open) is already that.'

    async def test_remove_todo_says_what_it_removed_and_that_it_is_gone(self, tools: list):
        todo_id = await _add(tools, "Temp item")
        await _add(tools, "Stays")
        result = await _find_tool(tools, "remove_todo").ainvoke({"title": "Temp item"})
        assert result == (
            f'Removed [todo:{todo_id}] "Temp item" from "Todo List"; it was open. It cannot be brought back. '
            "That list now has 1 todo, 1 open, 0 done."
        )

    async def test_remove_nonexistent(self, tools: list):
        remove = _find_tool(tools, "remove_todo")
        result = await remove.ainvoke({"title": "Ghost"})
        assert tool_result_status(result) == "no_change" and "This conversation has no todos." in result

    async def test_list_todos_empty(self, tools: list):
        lst = _find_tool(tools, "list_todos")
        result = await lst.ainvoke({})
        assert "No todos" in result

    async def test_list_todos_grouped_with_ids_counts_and_dates(self, tools: list):
        a = await _add(tools, "A", "Work")
        b = await _add(tools, "B", "Home")
        result = await _find_tool(tools, "list_todos").ainvoke({})
        assert result == (
            "2 todos, 2 open, 0 done, in 2 lists:\n"
            "### Home (1 todo, 1 open, 0 done)\n"
            f"- [ ] B [todo:{b}] (added Oct 9, 2026 at 10:00 PM UTC)\n"
            "\n"
            "### Work (1 todo, 1 open, 0 done)\n"
            f"- [ ] A [todo:{a}] (added Oct 9, 2026 at 10:00 PM UTC)"
        )

    async def test_list_shows_completion_and_when_it_changed(self, tools: list):
        await _add(tools, "Done item")
        await _find_tool(tools, "complete_todo").ainvoke({"title": "Done item"})
        result = await _find_tool(tools, "list_todos").ainvoke({})
        assert "[x] Done item [todo:" in result
        assert "(added Oct 9, 2026 at 10:00 PM UTC; changed Oct 10, 2026 at 3:30 PM UTC)" in result

    async def test_a_listing_says_its_times_in_the_persons_zone(
        self, storage: InMemoryTodoStorage, conv_id: UUID, user_id: UUID
    ):
        tools = load_todo_tools(storage=storage, conversation_id=conv_id, user_id=user_id, tz="America/Los_Angeles")
        await _add(tools, "A")
        assert "(added Oct 9, 2026 at 3:00 PM PDT)" in await _find_tool(tools, "list_todos").ainvoke({})

    async def test_list_todos_shows_one_list_when_asked(self, tools: list):
        await _add(tools, "A", "Work")
        await _add(tools, "B", "Home")
        result = await _find_tool(tools, "list_todos").ainvoke({"list_name": "work"})
        assert result.startswith("1 todo, 1 open, 0 done, in 1 list:\n### Work") and "B" not in result

    async def test_list_todos_for_a_list_that_is_not_there_names_the_lists(self, tools: list):
        await _add(tools, "A", "Work")
        result = await _find_tool(tools, "list_todos").ainvoke({"list_name": "Garden"})
        assert result == '[NO CHANGE] list_todos: there is no list "Garden". The lists here: "Work".'

    async def test_a_host_can_turn_markdown_off(self, storage: InMemoryTodoStorage, conv_id: UUID, user_id: UUID):
        """Plain lines for a host whose model should not read markdown; the default stays markdown."""
        plain = load_todo_tools(storage=storage, conversation_id=conv_id, user_id=user_id, markdown=False)
        await _add(plain, "A", "Work")
        await _add(plain, "B", "Work")
        await _find_tool(plain, "complete_todo").ainvoke({"title": "B", "list_name": "Work"})
        result = await _find_tool(plain, "list_todos").ainvoke({})
        assert (
            "Work (2 todos, 1 open, 1 done):" in result
            and "- open: A [todo:" in result
            and "- done: B [todo:" in result
        )
        assert "#" not in result and "[ ]" not in result and "[x]" not in result

    async def test_custom_list_name(self, tools: list):
        await _add(tools, "Deploy v2", "Release Checklist")
        result = await _find_tool(tools, "list_todos").ainvoke({})
        assert "Release Checklist" in result


# -- Naming one todo ----------------------------------------------------------


class TestATodoIsNamedOnce:
    """A call changes the one todo it names, or nothing.

    Both kinds of traffic: a title or id that names one todo goes through; a title
    that fits several, or an id of another conversation, changes nothing.
    """

    async def test_an_exact_title_wins_over_a_longer_one_in_any_case(self, tools: list, storage: InMemoryTodoStorage):
        await _add(tools, "Call mom")
        await _add(tools, "Call mom's doctor")
        result = await _find_tool(tools, "remove_todo").ainvoke({"title": "call MOM"})
        assert result.startswith("Removed [todo:") and '"Call mom" from' in result
        assert _titles(storage) == ["Call mom's doctor"]

    async def test_a_part_that_fits_one_todo_names_it(self, tools: list, storage: InMemoryTodoStorage):
        await _add(tools, "Call mom's doctor")
        await _add(tools, "Buy milk")
        result = await _find_tool(tools, "complete_todo").ainvoke({"title": "doctor"})
        assert result.startswith("Completed [todo:") and '"Call mom\'s doctor"' in result

    @pytest.mark.parametrize("tool_name", ["complete_todo", "update_todo", "remove_todo"])
    async def test_a_part_that_fits_several_changes_nothing_and_lists_them(
        self, tools: list, storage: InMemoryTodoStorage, tool_name: str
    ):
        first = await _add(tools, "Call mom")
        second = await _add(tools, "Call the bank")
        before = [dict(t) for t in storage.todos]
        result = await _find_tool(tools, tool_name).ainvoke({"title": "call", "new_title": "X"})
        assert result == (
            f'[NO CHANGE] {tool_name}: "call" matches 2 todos, so nothing was changed. '
            "Call again with the todo_id of the one you mean:\n"
            f'- [todo:{first}] "Call mom" in "Todo List" (open)\n'
            f'- [todo:{second}] "Call the bank" in "Todo List" (open)'
        )
        assert storage.todos == before

    async def test_the_same_title_on_two_lists_is_two_candidates(self, tools: list, storage: InMemoryTodoStorage):
        await _add(tools, "Deploy", "Release")
        await _add(tools, "Deploy", "Staging")
        result = await _find_tool(tools, "remove_todo").ainvoke({"title": "Deploy"})
        assert result.startswith('[NO CHANGE] remove_todo: "Deploy" matches 2 todos')
        assert len(storage.todos) == 2

    @pytest.mark.parametrize("written", ["{id}", "todo:{id}", "[todo:{id}]", " [todo:{id}] "])
    async def test_an_id_names_its_todo_however_the_mark_is_written(
        self, tools: list, storage: InMemoryTodoStorage, written: str
    ):
        await _add(tools, "Call mom")
        target = await _add(tools, "Call mom's doctor")
        result = await _find_tool(tools, "remove_todo").ainvoke({"todo_id": written.format(id=target)})
        assert result.startswith(f"Removed [todo:{target}]")
        assert _titles(storage) == ["Call mom"]

    async def test_an_id_from_another_conversation_changes_nothing(
        self, tools: list, storage: InMemoryTodoStorage, user_id: UUID
    ):
        elsewhere = load_todo_tools(storage=storage, conversation_id=uuid4(), user_id=user_id)
        theirs = await _add(elsewhere, "Not yours")
        result = await _find_tool(tools, "remove_todo").ainvoke({"todo_id": theirs})
        assert result == (
            f"[NO CHANGE] remove_todo: no todo [todo:{theirs}] in this conversation. list_todos shows every todo's id."
        )
        assert _titles(storage) == ["Not yours"]

    async def test_a_text_that_is_not_an_id_is_refused(self, tools: list):
        result = await _find_tool(tools, "remove_todo").ainvoke({"todo_id": "the first one"})
        assert result.startswith("[TOOL ERROR] remove_todo: 'the first one' is not a todo id")

    async def test_a_call_that_names_nothing_is_refused(self, tools: list):
        result = await _find_tool(tools, "complete_todo").ainvoke({})
        assert result == "[TOOL ERROR] complete_todo: say which todo: give todo_id or title."


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
