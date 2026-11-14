"""which database a reader answers from, known with no round trip where the reader can say
(:func:`~threetears.core.backends.dialect.known_on_yugabyte`)."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from prometheus_client import REGISTRY

from threetears.core.backends.dialect import check_index_hints, known_on_yugabyte


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
