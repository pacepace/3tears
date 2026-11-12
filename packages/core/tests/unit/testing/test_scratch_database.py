"""a test run's own database is made fresh, dropped after, and earlier runs' leftovers are swept
(``threetears.core.testing.postgres.scratch_database``; reports #39: killed runs left their databases behind)."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

import asyncpg
import pytest

from threetears.core.testing.postgres import scratch_database

_ADMIN = "postgresql://test:test@127.0.0.1:5432/postgres?sslmode=disable"


class _Server:
    """a server's databases and how many sessions each has, as the helper's statements see them."""

    def __init__(self, **databases: int) -> None:
        self.sessions: dict[str, int] = {"postgres": 0, **databases}
        self.locks: list[str] = []

    async def connect(self, dsn: str) -> _Conn:
        name = urlsplit(dsn).path.lstrip("/")
        assert name in self.sessions, f"no database {name}"
        assert urlsplit(dsn).query == "sslmode=disable", "the DSN's options are kept"
        self.sessions[name] += 1
        return _Conn(self, name)


class _Conn:
    def __init__(self, server: _Server, database: str) -> None:
        self.server = server
        self.database = database

    async def fetch(self, sql: str, pattern: str) -> list[dict[str, str]]:
        assert "pg_stat_activity" in sql
        # the prefix matched literally: every ``_`` (a LIKE wildcard) escaped, then ``%``
        assert pattern.endswith("%") and re.search(r"(?<!\\)_", pattern) is None, pattern
        literal = pattern[:-1].replace("\\_", "_")
        return [
            {"datname": name} for name, count in self.server.sessions.items() if count == 0 and name.startswith(literal)
        ]

    async def execute(self, sql: str, *params: Any) -> str:
        if "advisory_lock" in sql:
            self.server.locks.append(params[0])
            return "SELECT 1"
        if "advisory_unlock" in sql:
            assert self.server.locks.pop() == params[0]
            return "SELECT 1"
        found = re.fullmatch(r'(CREATE|DROP) DATABASE (?:IF EXISTS )?"([a-z0-9_]+)"( WITH \(FORCE\))?', sql)
        assert found is not None, sql
        verb, name, force = found.groups()
        if verb == "CREATE":
            self.server.sessions[name] = 0
        elif name in self.server.sessions:
            if self.server.sessions[name] and not force:
                raise asyncpg.ObjectInUseError(f'database "{name}" is being accessed by other users')
            del self.server.sessions[name]
        return f"{verb} DATABASE"

    async def close(self) -> None:
        if self.database in self.server.sessions:
            self.server.sessions[self.database] -= 1


async def test_a_killed_runs_leftover_is_swept_and_the_runs_own_is_dropped_after() -> None:
    """Flips if a database of the prefix that nothing uses is left behind by the next run, or if the run's own
    database outlives it."""
    server = _Server(reports_it_deadbeef=0)
    async with scratch_database(_ADMIN, "reports_it_", connect=server.connect) as dsn:
        made = urlsplit(dsn).path.lstrip("/")
        assert re.fullmatch(r"reports_it_[0-9a-f]{8}", made)
        assert "reports_it_deadbeef" not in server.sessions
        assert made in server.sessions
    assert set(server.sessions) == {"postgres"}
    assert server.locks == []


async def test_a_database_in_use_and_any_other_name_are_kept() -> None:
    """Flips if the sweep drops a database a run in progress holds, one of another prefix, or one of the prefix that
    is no run's (its suffix is not eight hex digits)."""
    server = _Server(reports_it_cafef00d=1, other_it_deadbeef=0, reports_it_keepme=0, reportsxit_deadbeef=0)
    async with scratch_database(_ADMIN, "reports_it_", connect=server.connect):
        pass
    assert set(server.sessions) == {
        "postgres",
        "reports_it_cafef00d",
        "other_it_deadbeef",
        "reports_it_keepme",
        "reportsxit_deadbeef",
    }


async def test_the_database_is_dropped_after_a_run_that_raised_with_its_connections_open() -> None:
    """Flips if a run that raised leaves its database, or a connection the run left open stops the drop."""
    server = _Server()
    with pytest.raises(RuntimeError, match="the test failed"):
        async with scratch_database(_ADMIN, "reports_it_", connect=server.connect) as dsn:
            await server.connect(dsn)  # a pool the test never closed
            raise RuntimeError("the test failed")
    assert set(server.sessions) == {"postgres"}


async def test_a_run_holds_its_database_until_it_is_done() -> None:
    """Flips if a run's database has no session of its own between its create and the run's first connection, so
    a concurrent run's sweep could drop it."""
    server = _Server()
    async with scratch_database(_ADMIN, "reports_it_", connect=server.connect) as dsn:
        assert server.sessions[urlsplit(dsn).path.lstrip("/")] == 1
        async with scratch_database(_ADMIN, "reports_it_", connect=server.connect) as second:
            assert urlsplit(dsn).path.lstrip("/") in server.sessions
            assert second != dsn
    assert set(server.sessions) == {"postgres"}


@pytest.mark.parametrize("prefix", ["", "reports_it", "Reports_it_", "reports-it_", "reports_it_1", "x;drop_"])
async def test_a_prefix_that_is_not_lower_case_letters_and_underscores_ending_in_one_is_refused(prefix: str) -> None:
    """Flips if a prefix a sweep could match other databases by (or one that is not a plain identifier) is taken."""
    server = _Server()
    with pytest.raises(ValueError, match="scratch database prefix"):
        async with scratch_database(_ADMIN, prefix, connect=server.connect):
            pass
    assert set(server.sessions) == {"postgres"}
