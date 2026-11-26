"""the source-token fence reads its row back whatever form the store returns the timestamp in."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import pytest

from threetears.core.backends.broker_generation import STAMP_S_MIN_FIELD, STAMPS_EPOCH_FIELD, stamp_fields
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.coordination.source_token import (
    SourceToken,
    SourceTokenFence,
    clear_signals,
    source_tokens_schema,
)
from threetears.core.ddl.row_epoch import next_epoch
from threetears.core.testing.stamped_tables import RecordingPool, assert_declared_stamped, stamps_in


class _Conn:
    """a connection whose one row carries ``as_of`` in the form a given store returns it."""

    def __init__(self, as_of: Any) -> None:
        self._as_of = as_of

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any]:
        return {"run_id": 9001, "as_of": self._as_of}

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        """the fence's read is a keyed point read (``threetears.core.keyed.read_row``), sent through ``fetch``."""
        return [await self.fetchrow(sql, *args)]


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
        "epoch",
        "gone",
    }


def test_the_fence_table_lives_in_its_owners_schema() -> None:
    from threetears.core.tenancy import TENANT_BY_SCHEMA

    assert source_tokens_schema("my_tokens").tenancy == TENANT_BY_SCHEMA


class _StampedPool(RecordingPool):
    """a recording pool that also keeps what each statement said on the wire of its stamp (``stamp_fields``)."""

    def __init__(self, answer: Any = None) -> None:
        super().__init__(answer)
        self.fields: list[dict[str, Any]] = []

    async def execute(self, sql: str, *params: Any) -> str:
        self.fields.append(stamp_fields())
        return await super().execute(sql, *params)

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        self.fields.append(stamp_fields())
        return await super().fetch(sql, *params)

    async def fetchrow(self, sql: str, *params: Any) -> dict[str, Any] | None:
        self.fields.append(stamp_fields())
        return await super().fetchrow(sql, *params)


def _accepting(sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    """the store's answer: the advance's upsert lands; no row is stored before it."""
    return [{"run_id": params[1]}] if sql.startswith("INSERT") else []


def _clearing(sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    """the store's answer: each gone mark names one row it marked."""
    return [{"signal": "marked"}] if sql.startswith("UPDATE") else []


def test_the_fence_table_is_stamped() -> None:
    """Flips if the fence table stops carrying its epoch (ROW_STAMPED: epoch, gone flag, restore bucket and index)."""
    assert_declared_stamped(source_tokens_schema())


async def test_an_advance_stamps_its_row_from_the_transaction_and_its_epoch_rises() -> None:
    """Flips if an advance writes no epoch, lets it go back (no GREATEST(epoch + 1, ...)), leaves a cleared row gone,
    draws its stamp from anywhere but the caller's transaction, or does not say on the wire that it stamps."""
    pool = _StampedPool(_accepting)
    held: list[CallerTransaction] = []

    @asynccontextmanager
    async def transaction() -> AsyncIterator[_StampedPool]:
        async with CallerTransaction(pool) as opened:
            held.append(opened)
            yield pool

    before = next_epoch()
    fence = SourceTokenFence(transaction, table="source_tokens", signal="refresh_ready")
    async with fence.advancing(SourceToken(run_id=9002, as_of=datetime(2026, 11, 3, tzinfo=UTC))) as advance:
        assert advance.accepted
    [(upsert, params)] = [(sql, params) for sql, params in pool.sent if sql.startswith("INSERT")]
    assert 'epoch = GREATEST("source_tokens".epoch + 1, EXCLUDED.epoch)' in upsert, upsert
    assert "gone = false" in upsert, upsert
    assert 'WHERE "source_tokens".gone OR (' in upsert, "a cleared row stands for no token"
    [stamp] = stamps_in(params, before)
    assert stamp == held[0].s_min, "the stamp is the caller's transaction's, named on its commit"
    upsert_fields = pool.fields[[sql for sql, _ in pool.sent].index(upsert)]
    assert upsert_fields == {STAMPS_EPOCH_FIELD: True, STAMP_S_MIN_FIELD: stamp}, upsert_fields


async def test_a_clear_writes_gone_marks_and_never_a_delete() -> None:
    """Flips if a clear deletes the fence's rows (a restore would find nothing), or marks them without rising their
    epoch, or marks a row already gone again, or sends its marks unstamped."""
    pool = _StampedPool(_clearing)
    before = next_epoch()
    cleared = await clear_signals(pool, "source_tokens", signal="refresh_ready", prefix="refresh_ready/")
    assert cleared == ["marked", "marked"]
    assert pool.sent and not any(sql.upper().startswith("DELETE") for sql, _ in pool.sent), pool.sent
    for (sql, params), fields in zip(pool.sent, pool.fields, strict=True):
        assert sql.startswith('UPDATE "source_tokens" SET "gone" = $1, "epoch" = GREATEST("epoch" + 1, $2::bigint)'), (
            sql
        )
        assert '(NOT "gone")' in sql, sql
        [stamp] = stamps_in(params, before)
        assert params[0] is True and fields == {STAMPS_EPOCH_FIELD: True, STAMP_S_MIN_FIELD: stamp}, (params, fields)


async def test_a_prefix_clear_marks_only_that_prefix_as_a_code_point_range() -> None:
    """Flips if a prefix clear names rows by anything but the C-ordered range of names starting with the prefix (a
    LIKE, a scan, or a range that also takes the job-less signal)."""
    pool = _StampedPool(_clearing)
    await clear_signals(pool, "source_tokens", prefix="refresh_ready/")
    [(sql, params)] = pool.sent
    assert sql.endswith(
        'WHERE "signal" COLLATE "C" >= $3 AND "signal" COLLATE "C" < $4 AND (NOT "gone") RETURNING "signal"'
    ), sql
    assert params[2:] == ("refresh_ready/", "refresh_ready0"), params


async def test_the_fence_reads_its_live_row_alone() -> None:
    """Flips if the fence's read would take a cleared (gone) row as the last token accepted."""
    pool = _StampedPool()

    @asynccontextmanager
    async def transaction() -> AsyncIterator[_StampedPool]:
        yield pool

    assert await SourceTokenFence(transaction, table="source_tokens", signal="refresh_ready").last() is None
    [(read, _)] = pool.sent
    assert read.startswith("SELECT") and '(NOT t."gone")' in read, read
