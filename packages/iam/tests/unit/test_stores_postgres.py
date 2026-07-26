"""The Postgres stores, against an in-memory pool double.

The properties here are the ones a table-backed store gets wrong: a claim that is not atomic,
and an expiry that depends on something having swept.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from threetears.iam.stores import SingleUseTicketStore, StateStore
from threetears.iam.stores.postgres import PostgresStateStore, PostgresTicketStore

_TTL = timedelta(minutes=30)


# parity-with: threetears.iam.stores.postgres.PoolLike
class _FakePool:
    """A pool double modelling the two statements these stores issue.

    Deliberately implements DELETE-RETURNING as a single indivisible step, because that is the
    guarantee under test: if the store ever split it into a read then a write, this double
    would still serialise it and the concurrency test below would pass on a broken store. The
    double therefore does NOT interleave -- the test relies on the store issuing exactly one
    statement, which is asserted separately.
    """

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.statements: list[str] = []

    def _key_column(self, query: str) -> str:
        return "hashed" if "hashed = $1" in query else "key"

    async def execute(self, query: str, *args: object) -> None:
        self.statements.append(query)
        collapsed = " ".join(query.split())
        if collapsed.startswith("INSERT"):
            key, payload, expires = str(args[0]), str(args[1]), args[2]
            self.rows[key] = {"payload": payload, "expires_at": expires}
        elif collapsed.startswith("DELETE") and "expires_at <= $1" in collapsed:
            cutoff = args[0]
            self.rows = {k: v for k, v in self.rows.items() if v["expires_at"] > cutoff}  # type: ignore[operator]
        else:  # pragma: no cover - the stores issue nothing else
            raise AssertionError(f"unhandled execute: {collapsed}")

    async def fetchrow(self, query: str, *args: object) -> dict[str, Any] | None:
        self.statements.append(query)
        collapsed = " ".join(query.split())
        key, now = str(args[0]), args[1]
        row = self.rows.get(key)
        if row is None or row["expires_at"] <= now:  # type: ignore[operator]
            return None
        if collapsed.startswith("DELETE"):
            del self.rows[key]
        return {"payload": row["payload"]}


@pytest.fixture
def pool() -> _FakePool:
    return _FakePool()


async def test_ticket_round_trips_and_redeems_once(pool: _FakePool) -> None:
    store = PostgresTicketStore(pool, table="tickets")
    issued = await store.issue({"principal": "p-1"}, ttl=_TTL)
    assert await store.redeem(issued.secret) == {"principal": "p-1"}
    assert await store.redeem(issued.secret) is None


async def test_only_the_hash_is_stored(pool: _FakePool) -> None:
    """A dump of this table must not be a set of usable links."""
    store = PostgresTicketStore(pool, table="tickets")
    issued = await store.issue({"principal": "p-1"}, ttl=_TTL)
    assert issued.hashed in pool.rows
    assert issued.secret not in pool.rows
    assert all(issued.secret not in str(row["payload"]) for row in pool.rows.values())


async def test_redemption_is_a_single_statement(pool: _FakePool) -> None:
    """The atomicity guarantee is structural: a read-then-delete would let two concurrent
    redemptions of one ticket both win, which for a reset ticket is two parties setting a
    password. Asserted by statement count, because a double cannot prove atomicity."""
    store = PostgresTicketStore(pool, table="tickets")
    issued = await store.issue({"principal": "p-1"}, ttl=_TTL)
    pool.statements.clear()
    await store.redeem(issued.secret)
    assert len(pool.statements) == 1
    assert "DELETE" in pool.statements[0] and "RETURNING" in pool.statements[0]


async def test_concurrent_redemption_produces_one_winner(pool: _FakePool) -> None:
    store = PostgresTicketStore(pool, table="tickets")
    issued = await store.issue({"principal": "p-1"}, ttl=_TTL)
    results = await asyncio.gather(*(store.redeem(issued.secret) for _ in range(8)))
    assert sum(1 for result in results if result is not None) == 1


async def test_an_expired_ticket_is_unredeemable_without_any_sweep(pool: _FakePool) -> None:
    """The property that makes a table tolerable here. Nothing has purged; the row is still
    physically present; it must still be refused."""
    store = PostgresTicketStore(pool, table="tickets")
    issued = await store.issue({"principal": "p-1"}, ttl=timedelta(seconds=-1))
    assert issued.hashed in pool.rows
    assert await store.redeem(issued.secret) is None


async def test_purge_removes_only_expired_rows(pool: _FakePool) -> None:
    store = PostgresTicketStore(pool, table="tickets")
    live = await store.issue({"n": 1}, ttl=_TTL)
    await store.issue({"n": 2}, ttl=timedelta(seconds=-1))
    await store.purge_expired()
    assert list(pool.rows) == [live.hashed]


async def test_an_unknown_secret_redeems_to_nothing(pool: _FakePool) -> None:
    assert await PostgresTicketStore(pool, table="tickets").redeem("never-issued") is None


@pytest.mark.parametrize("table", ["", "tickets; DROP TABLE users", 'tickets"', "tickets--x"])
def test_an_unsafe_table_name_is_refused(pool: _FakePool, table: str) -> None:
    """SQL has no placeholder for an identifier, so the name is interpolated -- bounded here
    so a configuration accident cannot become an injection."""
    with pytest.raises(ValueError, match="unsafe table name"):
        PostgresTicketStore(pool, table=table)


# --- state store -----------------------------------------------------------------------


async def test_state_get_does_not_consume_but_take_does(pool: _FakePool) -> None:
    store = PostgresStateStore(pool, table="state")
    await store.put("state-1", {"nonce": "n"}, ttl=_TTL)
    assert await store.get("state-1") == {"nonce": "n"}
    assert await store.get("state-1") == {"nonce": "n"}
    assert await store.take("state-1") == {"nonce": "n"}
    assert await store.take("state-1") is None


async def test_state_put_replaces_an_existing_key(pool: _FakePool) -> None:
    store = PostgresStateStore(pool, table="state")
    await store.put("state-1", {"nonce": "first"}, ttl=_TTL)
    await store.put("state-1", {"nonce": "second"}, ttl=_TTL)
    assert await store.get("state-1") == {"nonce": "second"}


async def test_expired_state_reads_as_absent_without_a_sweep(pool: _FakePool) -> None:
    store = PostgresStateStore(pool, table="state")
    await store.put("state-1", {"nonce": "n"}, ttl=timedelta(seconds=-1))
    assert await store.get("state-1") is None
    assert await store.take("state-1") is None


async def test_the_stores_satisfy_their_protocols(pool: _FakePool) -> None:
    assert isinstance(PostgresTicketStore(pool, table="tickets"), SingleUseTicketStore)
    assert isinstance(PostgresStateStore(pool, table="state"), StateStore)


async def test_a_corrupt_payload_reads_as_absent(pool: _FakePool) -> None:
    """Unusable either way; raising would turn it into a 500 on an auth path."""
    store = PostgresStateStore(pool, table="state")
    await store.put("state-1", {"nonce": "n"}, ttl=_TTL)
    pool.rows["state-1"]["payload"] = "{not json"
    assert await store.get("state-1") is None
