"""Postgres implementations of the storage Protocols.

:mod:`threetears.iam.stores` argues that this state does not belong in a table, and for a
service with a broker that is still true -- use :mod:`threetears.iam.stores.nats_kv`, where
expiry is the bucket's job and nothing has to remember to sweep. These exist for the case
that argument does not cover: a service whose broker is OPTIONAL, where an OAuth handoff or a
reset ticket must survive the broker being unreachable, and whose Postgres is therefore the
only store always present.

**Expiry is enforced in the query, never by the sweep.** Every claim carries
``AND expires_at > now``, so an expired row is unredeemable the instant it expires whether or
not anything has deleted it. :meth:`PostgresTicketStore.purge_expired` bounds the table's
SIZE; it is hygiene, not correctness. That distinction is the whole reason a table is
tolerable here -- a store whose security depends on a cron job is a store whose security
depends on a cron job still running.

**Payloads are handed to asyncpg as dicts, never pre-serialized.** These stores assume the
caller registered `threetears.core.collections.asyncpg_init.register_jsonb_text_codec` on the
pool, which every 3tears consumer that owns an asyncpg pool is required to do. That codec is
the ONE ``json.dumps`` step; serializing here as well would silently double-encode, leaving
the column holding a JSON *string* rather than an object -- so ``payload->>'user_id'``, any
functional index, and any operator query would return nothing useful, while a round trip
through this module still looked fine.

**This package still owns no schema.** The caller creates the table in its own migration and
names it here; the DDL is published as :data:`TICKET_TABLE_DDL` / :data:`STATE_TABLE_DDL` so
the column set the queries assume is stated once rather than reconstructed per consumer. What
the package owns is the atomic claim -- a single ``DELETE ... RETURNING``, so two concurrent
redemptions of one ticket produce exactly one winner. A read-then-delete would let both
through, and that is a password set twice by two parties.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Protocol

from threetears.observe import get_logger

from threetears.iam.stores.base import TicketIssue, hash_ticket, new_ticket_secret

__all__ = [
    "EXPIRES_INDEX_DDL",
    "STATE_TABLE_DDL",
    "TICKET_TABLE_DDL",
    "PoolLike",
    "PostgresStateStore",
    "PostgresTicketStore",
]

log = get_logger(__name__)

#: The column set :class:`PostgresTicketStore` assumes. ``{table}`` is substituted with the
#: caller's table name. Apply this from your OWN migration -- this package runs none.
TICKET_TABLE_DDL: Final[str] = """
CREATE TABLE IF NOT EXISTS {table} (
    hashed TEXT PRIMARY KEY,
    payload JSONB NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL
)
"""

#: The column set :class:`PostgresStateStore` assumes. Same substitution and same rule.
STATE_TABLE_DDL: Final[str] = """
CREATE TABLE IF NOT EXISTS {table} (
    key TEXT PRIMARY KEY,
    payload JSONB NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL
)
"""

#: Index backing :meth:`PostgresTicketStore.purge_expired`, for either table.
EXPIRES_INDEX_DDL: Final[str] = "CREATE INDEX IF NOT EXISTS {index} ON {table} (expires_at)"

#: Table names are interpolated, not parameterized -- SQL has no placeholder for an
#: identifier. They come from application config rather than user input, and this bound keeps
#: an accident from becoming an injection.
_SAFE_TABLE_CHARS: Final[frozenset[str]] = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_.")


class PoolLike(Protocol):
    """The minimal pool surface these stores need.

    Matches the top-level ``fetchrow``/``execute`` methods :class:`asyncpg.Pool` exposes (the
    pool acquires and releases internally). Typed as a Protocol so this package takes no
    runtime dependency on asyncpg, and so a test can pass a double.
    """

    async def fetchrow(self, query: str, *args: object) -> Any:
        """Run ``query`` and return the first row, or ``None``."""
        ...

    async def execute(self, query: str, *args: object) -> Any:
        """Run ``query``, discarding any result."""
        ...


def _validate_table(table: str) -> str:
    """Reject a table name that cannot be safely interpolated."""
    if not table or not set(table.lower()) <= _SAFE_TABLE_CHARS:
        raise ValueError(f"unsafe table name {table!r}; expected letters, digits, underscore or dot")
    return table


def _decode(raw: Any) -> Mapping[str, Any] | None:
    """Decode a stored payload, treating corruption as absence.

    A registered JSONB codec hands back a ``dict`` already; the string branch covers a pool
    without one, and a row written before the codec was in place. A value that will not parse
    is unusable either way, and raising would turn it into a 500 on an authentication path
    where the correct answer is "this ticket is not valid".
    """
    if isinstance(raw, Mapping):
        return dict(raw)
    try:
        decoded = json.loads(raw)
    except TypeError, ValueError:
        log.warning("discarding an unparseable stored payload")
        return None
    return decoded if isinstance(decoded, dict) else None


class PostgresTicketStore:
    """Postgres-backed :class:`~threetears.iam.stores.base.SingleUseTicketStore`."""

    def __init__(self, pool: PoolLike, *, table: str) -> None:
        """
        :param pool: the connection pool.
        :ptype pool: PoolLike
        :param table: the caller's table, created by the caller's own migration from
            :data:`TICKET_TABLE_DDL`.
        :ptype table: str
        :raises ValueError: the table name is not safely interpolable.
        """
        self._pool = pool
        self._table = _validate_table(table)

    async def issue(self, payload: Mapping[str, Any], *, ttl: timedelta) -> TicketIssue:
        """Store ``payload`` against a fresh secret and return it.

        Only the hash is written, so a dump of this table is not a set of usable links.
        """
        secret = new_ticket_secret()
        hashed = hash_ticket(secret)
        await self._pool.execute(
            f"INSERT INTO {self._table} (hashed, payload, expires_at) VALUES ($1, $2, $3)",  # noqa: S608
            hashed,
            dict(payload),  # the pool's JSONB codec encodes; see the module docstring
            datetime.now(UTC) + ttl,
        )
        return TicketIssue(secret=secret, hashed=hashed)

    async def redeem(self, secret: str) -> Mapping[str, Any] | None:
        """Atomically consume ``secret`` and return its payload, or ``None``.

        One statement, so two concurrent redemptions cannot both win. The expiry predicate is
        in the same statement: an expired row is unredeemable whether or not it has been
        swept. ``None`` covers unknown, expired and already-redeemed alike -- a caller must
        not distinguish them, or it becomes an oracle.
        """
        row = await self._pool.fetchrow(
            f"DELETE FROM {self._table} WHERE hashed = $1 AND expires_at > $2 RETURNING payload",  # noqa: S608
            hash_ticket(secret),
            datetime.now(UTC),
        )
        return None if row is None else _decode(row["payload"])

    async def purge_expired(self) -> None:
        """Delete rows past their expiry.

        Bounds the table's size and nothing else -- :meth:`redeem` already refuses an expired
        row. Safe to call on a schedule, or on a rare write; correctness never depends on it
        having run.
        """
        await self._pool.execute(
            f"DELETE FROM {self._table} WHERE expires_at <= $1",  # noqa: S608
            datetime.now(UTC),
        )


class PostgresStateStore:
    """Postgres-backed :class:`~threetears.iam.stores.base.StateStore`."""

    def __init__(self, pool: PoolLike, *, table: str) -> None:
        """
        :param pool: the connection pool.
        :ptype pool: PoolLike
        :param table: the caller's table, created from :data:`STATE_TABLE_DDL`.
        :ptype table: str
        :raises ValueError: the table name is not safely interpolable.
        """
        self._pool = pool
        self._table = _validate_table(table)

    async def put(self, key: str, payload: Mapping[str, Any], *, ttl: timedelta) -> None:
        """Store ``payload`` under ``key`` for ``ttl``, replacing any existing value."""
        await self._pool.execute(
            f"INSERT INTO {self._table} (key, payload, expires_at) VALUES ($1, $2, $3) "  # noqa: S608
            "ON CONFLICT (key) DO UPDATE SET payload = EXCLUDED.payload, expires_at = EXCLUDED.expires_at",
            key,
            dict(payload),  # the pool's JSONB codec encodes; see the module docstring
            datetime.now(UTC) + ttl,
        )

    async def take(self, key: str) -> Mapping[str, Any] | None:
        """Atomically remove and return the value under ``key``, or ``None``."""
        row = await self._pool.fetchrow(
            f"DELETE FROM {self._table} WHERE key = $1 AND expires_at > $2 RETURNING payload",  # noqa: S608
            key,
            datetime.now(UTC),
        )
        return None if row is None else _decode(row["payload"])

    async def get(self, key: str) -> Mapping[str, Any] | None:
        """Return the value under ``key`` WITHOUT consuming it, or ``None``.

        Only correct where a separate mechanism enforces single use -- see the Protocol.
        """
        row = await self._pool.fetchrow(
            f"SELECT payload FROM {self._table} WHERE key = $1 AND expires_at > $2",  # noqa: S608
            key,
            datetime.now(UTC),
        )
        return None if row is None else _decode(row["payload"])

    async def purge_expired(self) -> None:
        """Delete rows past their expiry. Hygiene only -- see
        :meth:`PostgresTicketStore.purge_expired`."""
        await self._pool.execute(
            f"DELETE FROM {self._table} WHERE expires_at <= $1",  # noqa: S608
            datetime.now(UTC),
        )
