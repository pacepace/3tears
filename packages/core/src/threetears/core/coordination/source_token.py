"""A fence on a producer's "my data is ready" signal: a token not newer than the last one accepted is refused.

For any pod woken by another system's pipeline (a dbt run, an export, an upload): the pod keeps one
fence row per signal in its own L3 table (:func:`source_tokens_schema`, declared among its tables)
and starts its work only on a token newer than the last it accepted.

**The token** names the data a signal is about: the producer's run (a number that grows with every
run, such as a dbt Cloud run id) and the newest data that run saw (``as_of``, such as the newest
``source_timestamp`` in the tables it built). A token is newer than the last accepted when its run
is later and its data is not older. So a repeat of a signal (Airflow retrying a task whose call
landed) is refused, a late signal from an earlier run is refused, and a later run whose data is
older than what was already signalled is refused too: it says nothing the pod has not already been
told. A later run with the same newest timestamp is accepted, because a run can change rows (a race
called, a correction) without moving any timestamp.

**One statement decides.** The fence is one row per signal in the pod's L3, advanced by an
``INSERT ... ON CONFLICT DO UPDATE ... WHERE``: Postgres takes the row's lock and evaluates the
condition against its latest version, so two replicas given the same token at the same moment
accept it exactly once. The work the token starts runs inside the same transaction
(:meth:`SourceTokenFence.advancing`): if starting it fails, the transaction rolls back and the token
is not spent, so the producer's retry is accepted rather than refused as a repeat.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

from threetears.observe import get_logger

from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    DATETIMETZ_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
)

__all__ = [
    "SOURCE_TOKENS_TABLE",
    "Advance",
    "SourceToken",
    "SourceTokenFence",
    "Transaction",
    "source_tokens_schema",
]

log = get_logger(__name__)

#: the fence table's default name
SOURCE_TOKENS_TABLE: Final = "source_tokens"

#: a transaction on the pod's L3, its connection yielded (an SDK pod's ``provider_transaction``)
Transaction = Callable[[], AbstractAsyncContextManager[Any]]

#: the fence row's columns, as the pod's table list declares them
_SIGNAL: Final = "signal"
_RUN_ID: Final = "run_id"
_AS_OF: Final = "as_of"


def source_tokens_schema(name: str = SOURCE_TOKENS_TABLE) -> TableSchema:
    """the fence table, for the owner to declare among its tables: one row per signal.

    :param name: the table's name
    :ptype name: str
    :return: the schema, keyed on ``signal``
    :rtype: TableSchema
    """
    return TableSchema(
        name=name,
        primary_key=_SIGNAL,
        columns=[
            Column(_SIGNAL, STRING_TYPE),
            Column(_RUN_ID, BIGINT_TYPE),
            Column(_AS_OF, DATETIMETZ_TYPE),
            Column("accepted_at", DATETIMETZ_TYPE),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
        on_conflict="update",
    )


@dataclass(frozen=True)
class SourceToken:
    """what a ready signal says about the data: the producer's run, and the newest data it saw.

    :ivar run_id: the producer's run, growing with every run (a dbt Cloud run id)
    :ivar as_of: the newest data the run saw, timezone-aware (the newest ``source_timestamp``)
    """

    run_id: int
    as_of: datetime

    def __post_init__(self) -> None:
        """refuse a token the fence cannot order.

        :return: nothing
        :rtype: None
        :raises ValueError: for a run that is not a positive number, or a timestamp without a zone
        """
        if isinstance(self.run_id, bool) or not isinstance(self.run_id, int) or self.run_id < 1:
            raise ValueError(f"a run id is a positive whole number, not {self.run_id!r}")
        if self.as_of.tzinfo is None or self.as_of.utcoffset() is None:
            raise ValueError(f"the timestamp {self.as_of.isoformat()} names no time zone; add one (Z for UTC)")

    def newer_than(self, last: SourceToken | None) -> bool:
        """whether this token says something ``last`` did not: a later run, with data not older.

        :param last: the last token accepted, None when none was
        :ptype last: SourceToken | None
        :return: True when it is newer
        :rtype: bool
        """
        return last is None or (self.run_id > last.run_id and self.as_of >= last.as_of)

    def render(self) -> dict[str, Any]:
        """the token as JSON values.

        :return: ``run_id`` and ``as_of`` (ISO 8601)
        :rtype: dict[str, Any]
        """
        return {_RUN_ID: self.run_id, _AS_OF: self.as_of.isoformat()}


@dataclass(frozen=True)
class Advance:
    """what the fence made of a token.

    :ivar accepted: whether the token was newer, and is now the last accepted
    :ivar last: the last token accepted before this one (None when there was none)
    """

    accepted: bool
    last: SourceToken | None


class SourceTokenFence:
    """one signal's fence, a row in the pod's L3: advances only to a newer token.

    :param transaction: opens one transaction on the pod's L3
    :ptype transaction: Transaction
    :param table: the fence's table (:func:`source_tokens_schema`), one of the pod's own
    :ptype table: str
    :param signal: which signal this fence keeps, the row's key
    :ptype signal: str
    """

    def __init__(self, transaction: Transaction, *, table: str, signal: str) -> None:
        self._transaction = transaction
        self._table = f'"{table}"'
        self._signal = signal

    @asynccontextmanager
    async def advancing(self, token: SourceToken) -> AsyncIterator[Advance]:
        """advance to ``token`` if it is newer; the block runs in the same transaction.

        Accepted, the token is the last accepted once the block ends without raising; a block that
        raises rolls the advance back, so the token can be signalled again. Refused, nothing changes
        and the block is told which token stood.

        :param token: the token signalled
        :ptype token: SourceToken
        :return: an async context manager yielding the outcome
        :rtype: AsyncIterator[Advance]
        """
        async with self._transaction() as conn:
            previous = await self._read(conn)
            row = await conn.fetchrow(
                f"INSERT INTO {self._table} ({_SIGNAL}, {_RUN_ID}, {_AS_OF}, accepted_at, date_created, date_updated) "  # noqa: S608 - the owner's own table
                "VALUES ($1, $2, $3, now(), now(), now()) "
                f"ON CONFLICT ({_SIGNAL}) DO UPDATE SET {_RUN_ID} = EXCLUDED.{_RUN_ID}, {_AS_OF} = EXCLUDED.{_AS_OF}, "
                "accepted_at = now(), date_updated = now() "
                f"WHERE {self._table}.{_RUN_ID} < EXCLUDED.{_RUN_ID} AND {self._table}.{_AS_OF} <= EXCLUDED.{_AS_OF} "
                f"RETURNING {_RUN_ID}",
                self._signal,
                token.run_id,
                token.as_of,
            )
            # refused: the token that stands, which a racing advance may have written since the read above
            last = previous if row is not None else await self._read(conn)
            advance = Advance(accepted=row is not None, last=last)
            log.info(
                "source token accepted" if advance.accepted else "source token refused: not newer than the last",
                extra={
                    "extra_data": {
                        "signal": self._signal,
                        "token": token.render(),
                        "last": None if last is None else last.render(),
                    }
                },
            )
            yield advance

    async def last(self) -> SourceToken | None:
        """the last token accepted.

        :return: the token, None when none was
        :rtype: SourceToken | None
        """
        async with self._transaction() as conn:
            token = await self._read(conn)
        return token

    async def _read(self, conn: Any) -> SourceToken | None:
        """the fence's row as a token.

        :param conn: the transaction's connection
        :ptype conn: Any
        :return: the token, None when no token was accepted
        :rtype: SourceToken | None
        """
        row = await conn.fetchrow(
            f"SELECT {_RUN_ID}, {_AS_OF} FROM {self._table} WHERE {_SIGNAL} = $1",  # noqa: S608 - the owner's own table
            self._signal,
        )
        return None if row is None else SourceToken(run_id=int(row[_RUN_ID]), as_of=_as_datetime(row[_AS_OF]))


def _as_datetime(stored: Any) -> datetime:
    """the stored ``as_of`` as a timezone-aware datetime, whatever form the store returned it in.

    asyncpg returns a ``timestamptz`` as a datetime; the hub's L3 broker returns it as text (ISO, or
    Postgres's ``2026-11-03 23:40:00+00``). The column is ``timestamptz``, so text without an offset is UTC.

    :param stored: the column's value
    :ptype stored: Any
    :return: the moment, timezone-aware
    :rtype: datetime
    """
    moment = stored if isinstance(stored, datetime) else datetime.fromisoformat(str(stored))
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
