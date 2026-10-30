"""The SQLite store passes the readings conformance kit, on a file and in memory.

The kit's backend hook counts statements: the store's connection is opened through a ``sqlite3.Connection`` subclass
whose ``execute`` calls the installed hook first, so a read that ran two statements, or hid a failing one, is seen.
Beside the kit: a narrowed read abandoned part way leaves the store usable, and one longer than a fetched page comes
back whole and in order.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from threetears.evals.schema import ResultReadings
from threetears.evals.storage import SqliteDocumentStore
from threetears.evals.testing import RESULT_READINGS_CONFORMANCE_CASES, ResultReadingsConformanceCase


class _HookedConnection(sqlite3.Connection):
    """A connection that calls the installed hook before each statement it executes, and records the statement.

    The hook is shared by every connection the test opens, the store's own and a read's own alike, since a statement
    is a query issued to the database whichever connection carries it.
    """

    hook: Callable[[], None] | None = None
    statements: list[str] = []

    def execute(self, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
        _HookedConnection.statements.append(sql)
        if _HookedConnection.hook is not None:
            _HookedConnection.hook()
        return super().execute(sql, parameters)


@pytest.fixture
def connections(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[_HookedConnection]]:
    """Every connection a store opens in this test, each a :class:`_HookedConnection`."""
    opened: list[_HookedConnection] = []
    connect = partial(sqlite3.connect, factory=_HookedConnection)

    def hooked_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection = connect(*args, **kwargs)
        assert isinstance(connection, _HookedConnection)
        opened.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", hooked_connect)
    monkeypatch.setattr(_HookedConnection, "hook", None)
    monkeypatch.setattr(_HookedConnection, "statements", [])
    yield opened


def _installer(connections: list[_HookedConnection]) -> Callable[[Callable[[], None]], None]:
    def install(hook: Callable[[], None]) -> None:
        _HookedConnection.hook = hook

    return install


@pytest.mark.parametrize("case", RESULT_READINGS_CONFORMANCE_CASES, ids=lambda case: case.name)
def test_the_sqlite_store_offers_readings(
    case: ResultReadingsConformanceCase, tmp_path: Path, connections: list[_HookedConnection]
) -> None:
    with SqliteDocumentStore(tmp_path / "evals.sqlite") as store:
        case.run(store, _installer(connections))


@pytest.mark.parametrize("case", RESULT_READINGS_CONFORMANCE_CASES, ids=lambda case: case.name)
def test_the_sqlite_store_offers_readings_in_memory(
    case: ResultReadingsConformanceCase, connections: list[_HookedConnection]
) -> None:
    with SqliteDocumentStore(":memory:") as store:
        case.run(store, _installer(connections))


def test_it_offers_the_capability() -> None:
    with SqliteDocumentStore(":memory:") as store:
        assert isinstance(store, ResultReadings)


def _doc(index: int) -> dict[str, Any]:
    return {"id": f"d{index:04d}", "scope_id": "s", "doc_type": "t", "n": index, "scores": [{"reasoning": "x"}]}


def test_a_narrowed_read_longer_than_a_page_comes_back_whole_and_in_order(tmp_path: Path) -> None:
    with SqliteDocumentStore(tmp_path / "evals.sqlite") as store:
        for index in reversed(range(1000)):
            store.upsert(_doc(index))
        rows = list(store.iter_narrowed("t", "s", exclude_in_items={"scores": ["reasoning"]}))
        assert [row["n"] for row in rows] == list(reversed(range(1000)))
        assert all(row["scores"] == [{}] for row in rows)


def test_a_narrowed_read_abandoned_part_way_leaves_the_store_usable(tmp_path: Path) -> None:
    with SqliteDocumentStore(tmp_path / "evals.sqlite") as store:
        for index in range(600):
            store.upsert(_doc(index))
        rows = store.iter_narrowed("t", "s")
        assert next(rows)["n"] == 0
        store.upsert(_doc(600))
        assert store.merge_fields("d0001", "s", {"n": -1})
        rows.close()
        assert store.get("d0001", "s") == {**_doc(1), "n": -1}
        assert len(list(store.iter_narrowed("t", "s"))) == 601


def test_a_paused_narrowed_read_never_blocks_a_write_after_another_process_wrote(tmp_path: Path) -> None:
    """A read left part way holds a snapshot; the store's writes must not be the connection holding it."""
    path = tmp_path / "evals.sqlite"
    with SqliteDocumentStore(path) as store, SqliteDocumentStore(path) as other_process:
        for index in range(600):
            store.upsert(_doc(index))
        rows = store.iter_narrowed("t", "s")
        assert next(rows)["n"] == 0
        other_process.upsert(_doc(600))
        store.upsert(_doc(601))
        assert store.merge_fields("d0001", "s", {"n": -1})
        assert store.delete("d0002", "s")
        remaining = list(rows)
        assert [row["n"] for row in remaining] == list(range(1, 600))
        assert remaining[0]["n"] == 1, "the read sees the rows as they were at the call"
        assert len(list(store.iter_narrowed("t", "s"))) == 601


def test_a_narrowed_read_closes_its_own_connection(tmp_path: Path, connections: list[_HookedConnection]) -> None:
    with SqliteDocumentStore(tmp_path / "evals.sqlite") as store:
        for index in range(3):
            store.upsert(_doc(index))
        assert len(list(store.iter_narrowed("t", "s"))) == 3
        (_own, reader) = connections
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            reader.execute("SELECT 1")
        rows = store.iter_narrowed("t", "s")
        next(rows)
        rows.close()  # type: ignore[attr-defined]
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connections[2].execute("SELECT 1")


def test_grouped_values_need_nothing_newer_than_json1_and_filter(
    tmp_path: Path, connections: list[_HookedConnection]
) -> None:
    """No ``->`` (SQLite 3.38) and no ORDER BY inside an aggregate (3.44): a system SQLite older than either runs it."""
    with SqliteDocumentStore(tmp_path / "evals.sqlite") as store:
        store.upsert({"id": "b", "scope_id": "s", "doc_type": "t", "run": "a", "v": True})
        store.upsert({"id": "a", "scope_id": "s", "doc_type": "t", "run": "a", "v": {"k": [1]}})
        _HookedConnection.statements.clear()
        groups = store.grouped("t", "s", by=["run"], measures={"v": ("values", "v"), "top": ("max", "run")})
        assert groups == [{"run": "a", "v": [["b", True], ["a", {"k": [1]}]], "top": "a"}]
        (statement,) = _HookedConnection.statements
        assert "->" not in statement
        assert "ORDER BY" not in statement.upper()
