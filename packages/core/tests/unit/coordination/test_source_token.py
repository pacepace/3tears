"""the source-token fence reads its row back whatever form the store returns the timestamp in."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import pytest

from threetears.core.coordination.source_token import SourceToken, SourceTokenFence, source_tokens_schema


class _Conn:
    """a connection whose one row carries ``as_of`` in the form a given store returns it."""

    def __init__(self, as_of: Any) -> None:
        self._as_of = as_of

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any]:
        return {"run_id": 9001, "as_of": self._as_of}


def _fence(as_of: Any) -> SourceTokenFence:
    @asynccontextmanager
    async def transaction() -> AsyncIterator[_Conn]:
        yield _Conn(as_of)

    return SourceTokenFence(transaction, table="source_tokens", signal="dbt")


@pytest.mark.parametrize(
    "stored",
    [
        datetime(2026, 11, 3, 23, 40, tzinfo=UTC),
        "2026-11-03T23:40:00+00:00",
        "2026-11-03 23:40:00+00",
    ],
    ids=["datetime (asyncpg)", "ISO string (the hub's L3 broker)", "Postgres text form"],
)
async def test_the_last_token_reads_back_as_an_aware_datetime(stored: Any) -> None:
    last = await _fence(stored).last()
    assert last == SourceToken(run_id=9001, as_of=datetime(2026, 11, 3, 23, 40, tzinfo=UTC))


async def test_a_repeated_token_is_not_newer_than_the_one_read_back_as_a_string() -> None:
    last = await _fence("2026-11-03T23:40:00+00:00").last()
    repeat = SourceToken(run_id=9001, as_of=datetime(2026, 11, 3, 23, 40, tzinfo=UTC))
    assert not repeat.newer_than(last)


class TestTheToken:
    """a token is newer only when its run is later and its data is not older."""

    _AT = datetime(2026, 11, 3, 23, 40, tzinfo=UTC)

    def test_a_first_token_is_newer_than_none(self) -> None:
        assert SourceToken(run_id=1, as_of=self._AT).newer_than(None)

    @pytest.mark.parametrize(
        ("run_id", "minutes", "newer"),
        [(9002, 0, True), (9002, 5, True), (9001, 0, False), (9000, 5, False), (9002, -1, False)],
        ids=["later run same data", "later run newer data", "a repeat", "an earlier run", "later run older data"],
    )
    def test_newer_means_a_later_run_with_data_not_older(self, run_id: int, minutes: int, newer: bool) -> None:
        from datetime import timedelta  # noqa: PLC0415

        last = SourceToken(run_id=9001, as_of=self._AT)
        assert SourceToken(run_id=run_id, as_of=self._AT + timedelta(minutes=minutes)).newer_than(last) is newer

    @pytest.mark.parametrize("run_id", [0, -1, True])
    def test_a_run_that_is_not_a_positive_whole_number_is_refused(self, run_id: Any) -> None:
        with pytest.raises(ValueError, match="positive whole number"):
            SourceToken(run_id=run_id, as_of=self._AT)

    def test_a_timestamp_without_a_zone_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no time zone"):
            SourceToken(run_id=1, as_of=datetime(2026, 11, 3, 23, 40))  # noqa: DTZ001


def test_the_fence_table_is_keyed_on_the_signal() -> None:
    schema = source_tokens_schema("my_tokens")
    assert (schema.name, schema.primary_key) == ("my_tokens", "signal")
    assert {c.name for c in schema.columns} == {
        "signal",
        "run_id",
        "as_of",
        "accepted_at",
        "date_created",
        "date_updated",
    }
