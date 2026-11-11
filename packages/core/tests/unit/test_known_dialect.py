"""which database a reader answers from, known with no round trip where the reader can say
(:func:`~threetears.core.backends.dialect.known_on_yugabyte`)."""

from __future__ import annotations

from typing import Any

from threetears.core.backends.dialect import known_on_yugabyte


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
