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
    """A connection that calls the installed hook before each statement it executes."""

    hook: Callable[[], None] | None = None

    def execute(self, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
        if self.hook is not None:
            self.hook()
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
    yield opened


def _installer(connections: list[_HookedConnection]) -> Callable[[Callable[[], None]], None]:
    (connection,) = connections

    def install(hook: Callable[[], None]) -> None:
        connection.hook = hook

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
