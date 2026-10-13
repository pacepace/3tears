"""``ScopeEpochs.begin`` refuses at once on a refused ``NOWAIT`` lock, told apart by its type alone.

A direct pool raises ``asyncpg.LockNotAvailableError`` for SQLSTATE 55P03, and the L3 broker's proxy
rebuilds the same type (``LOCK_NOT_AVAILABLE``). Only that type becomes
:class:`EpochRecordBusyError`; an outage whose message happens to say "could not obtain lock" stays
the outage it is.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import asyncpg
import pytest

from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.scope_epochs import EpochRecordBusyError, ScopeEpochs
from threetears.core.exceptions import DataLayerUnavailableError


class _Conn:
    """a connection whose first statement raises ``error``."""

    def __init__(self, error: BaseException) -> None:
        self._error = error

    @asynccontextmanager
    async def _transaction(self) -> Any:
        yield self

    def transaction(self, **_: Any) -> Any:
        return self._transaction()

    async def fetchrow(self, sql: str, *params: Any) -> Any:
        raise self._error


def _epochs() -> ScopeEpochs:
    return ScopeEpochs(SimpleNamespace(table_name="scope_epochs"))  # type: ignore[arg-type]


async def _begin(error: BaseException) -> int:
    conn = _Conn(error)
    async with CallerTransaction(conn):
        return await _epochs().begin(conn=conn)


async def test_a_refused_nowait_is_the_record_busy() -> None:
    refused = asyncpg.PostgresError.new({"C": "55P03", "M": 'could not obtain lock on row in relation "scope_epochs"'})
    assert isinstance(refused, asyncpg.LockNotAvailableError)

    with pytest.raises(EpochRecordBusyError) as raised:
        await _begin(refused)
    assert raised.value.__cause__ is refused


async def test_an_outage_whose_message_mentions_a_lock_stays_an_outage() -> None:
    outage = DataLayerUnavailableError("tx.fetchrow failed: QUERY_EXECUTION_ERROR: could not obtain lock (55P03)")

    with pytest.raises(DataLayerUnavailableError):
        await _begin(outage)
