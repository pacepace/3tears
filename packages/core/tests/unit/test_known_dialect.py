"""which database a reader answers from, known with no round trip where the reader can say
(:func:`~threetears.core.backends.dialect.known_on_yugabyte`)."""

from __future__ import annotations

import logging
from typing import Any

import asyncpg
import pytest
from prometheus_client import REGISTRY

from threetears.core.backends.dialect import check_index_hints, known_on_yugabyte
from threetears.core.backends.sql import SqlL3Backend
from threetears.core.keyed import with_options


class _Asked:
    def __init__(self, version: str) -> None:
        self.version = version
        self.asked = 0

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        self.asked += 1
        return [{"v": self.version}]


async def test_a_reader_that_cannot_say_is_asked_once_and_its_answer_kept() -> None:
    """Flips if a pool or the rail's proxy is asked its database on every page (a round trip a page), or the
    answer is another reader's."""
    yugabyte, postgres = _Asked("PostgreSQL 15.12-YB-2026.1.0.0-b0"), _Asked("PostgreSQL 16.15")
    assert [await known_on_yugabyte(yugabyte) for _ in range(3)] == [True] * 3
    assert [await known_on_yugabyte(postgres) for _ in range(3)] == [False] * 3
    assert (yugabyte.asked, postgres.asked) == (1, 1)


class _Settings:
    def __init__(self, version: str) -> None:
        self.server_version = version


class _NewConnection:
    """a new connection as the pool's ``init=`` hook sees it: its server's version known, one setting asked."""

    def __init__(self, version: str, hint_setting: str | None) -> None:
        self._version = version
        self._hint_setting = hint_setting
        self.asked: list[str] = []

    def get_settings(self) -> _Settings:
        return _Settings(self._version)

    async def fetchval(self, query: str, *params: Any) -> str | None:
        self.asked.append(query)
        return self._hint_setting


async def test_a_yugabyte_connection_says_whether_its_index_hints_are_honoured(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Flips if a YugabyteDB connection without pg_hint_plan is not said by the gauge (1) and one error log, one with
    it does not set the gauge back to 0, or a PostgreSQL connection is asked anything."""
    postgres = _NewConnection("16.15", None)
    assert await check_index_hints(postgres) is None and postgres.asked == []
    missing = _NewConnection("15.12-YB-2026.1.0.0-b0", None)
    with caplog.at_level(logging.ERROR):
        assert await check_index_hints(missing) is False
        assert await check_index_hints(_NewConnection("15.12-YB-2026.1.0.0-b0", None)) is False
    assert len(missing.asked) == 1 and "pg_hint_plan" in missing.asked[0]
    assert REGISTRY.get_sample_value("threetears_yb_index_hint_unavailable") == 1.0
    assert sum("pg_hint_plan is not loaded" in record.getMessage() for record in caplog.records) <= 1
    assert await check_index_hints(_NewConnection("15.12-YB-2026.1.0.0-b0", "on")) is True
    assert REGISTRY.get_sample_value("threetears_yb_index_hint_unavailable") == 0.0


class _PoolConnection:
    def __init__(self, version: str) -> None:
        self._version = version

    def get_settings(self) -> _Settings:
        return _Settings(self._version)


async def test_an_asyncpg_pool_is_asked_once_though_it_cannot_be_held_weakly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flips if a real asyncpg pool (slotted: no weak reference) is asked ``SELECT version()`` on every read -- the
    hub's and every service's L3 is one, so each keyed read paid a round trip -- or one pool's answer is another's."""
    asked: list[str] = []

    async def fetch(pool: asyncpg.Pool, query: str, *params: Any) -> list[dict[str, Any]]:
        asked.append(query)
        return [{"v": versions[id(pool)]}]

    monkeypatch.setattr(asyncpg.Pool, "fetch", fetch)
    yugabyte = asyncpg.create_pool("postgresql://u@127.0.0.1:1/d", min_size=0, max_size=1)
    postgres = asyncpg.create_pool("postgresql://u@127.0.0.1:1/d", min_size=0, max_size=1)
    versions = {id(yugabyte): "PostgreSQL 15.12-YB-2026.1.0.0-b0", id(postgres): "PostgreSQL 16.15"}
    assert [await known_on_yugabyte(yugabyte) for _ in range(3)] == [True] * 3
    assert [await known_on_yugabyte(postgres) for _ in range(3)] == [False] * 3
    assert len(asked) == 2


class _Rail:
    """the rail's proxy as a pod holds it: it says its broker's dialect, and its broker refuses ``version()``."""

    async def hub_capabilities(self) -> frozenset[str]:
        return frozenset({"dialect:yugabyte"})

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        raise AssertionError(f"the broker refuses a function it does not model: {query}")


async def test_a_collection_s_backend_answers_for_the_reader_behind_it() -> None:
    """Flips if a collection's SqlL3Backend (the registry wraps the rail's proxy and a bare pool in one) sends
    ``SELECT version()`` itself: over the rail the broker refuses it, so a pod's keyed read with an index hint fails;
    over a pool it is a second question beside the pool's own."""
    rail = SqlL3Backend(_Rail())
    assert await known_on_yugabyte(rail) and await known_on_yugabyte(rail)
    behind = _Asked("PostgreSQL 16.15")
    assert not await known_on_yugabyte(SqlL3Backend(behind))
    assert not await known_on_yugabyte(SqlL3Backend(behind))
    assert behind.asked == 1


async def test_a_reader_with_options_answers_for_the_connection_behind_it() -> None:
    """Flips if ``with_options``' wrapper (one a call) sends ``SELECT version()`` itself: a round trip every wrapped
    read over a pool, and a refused statement over the rail."""
    assert await known_on_yugabyte(with_options(_Rail(), namespace="n"))
    behind = _Asked("PostgreSQL 16.15")
    assert [await known_on_yugabyte(with_options(behind, namespace="n")) for _ in range(3)] == [False] * 3
    assert behind.asked == 1
