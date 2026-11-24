"""fixtures shared across core's tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.keyed import binding


@pytest.fixture
def l1_statements() -> Callable[[SQLiteBackend], list[str]]:
    """trace a SQLite L1: every statement its connection on this thread runs from then on.

    For the tests that pin how many write transactions a change takes (count ``BEGIN IMMEDIATE``).

    :return: a call taking the backend and answering the list its statements are appended to
    :rtype: Callable[[SQLiteBackend], list[str]]
    """

    def trace(backend: SQLiteBackend) -> list[str]:
        statements: list[str] = []
        backend.get_connection().set_trace_callback(statements.append)
        return statements

    return trace


@pytest.fixture
def plain_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """the keyed helpers in the no-Arrow mode (a process with no pyarrow: each key column bound as one plain list),
    for a test pinning that binding: its reader is DuckDB behind a fake standing for asyncpg or the rail, which parses
    no PostgreSQL array literal. Arrow mode's binding is test_keyed_binding's."""
    monkeypatch.setattr(binding, "_ARROW", False)
