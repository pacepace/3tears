"""fixtures shared across core's tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from threetears.core.cache.sqlite import SQLiteBackend


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
