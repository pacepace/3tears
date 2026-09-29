"""
one DDL job per database at a time: the database-wide DDL lock.

**why per database.** measured on YugabyteDB 2026.1: two ``CREATE INDEX``
statements in ONE database -- on the same table or on different tables, in the
same schema or in different ones -- both hang until the catalog-version wait
times out (``yb_wait_for_backends_catalog_version_timeout``, 900s by default),
and both indexes are left invalid. a build waits for every running statement
and open transaction in its database, and two builds each wait for the other.
builds in different databases do not interfere. a lock keyed per schema lets
two agent schemas of one database migrate at once -- two hub replicas starting
together is enough -- which is exactly the failing case. so every DDL job in a
database takes this one lock, keyed on ``current_database()``.

**why polled.** the lock is taken with ``pg_try_advisory_lock`` at a bounded
interval, never with the blocking ``pg_advisory_lock``. a session blocked in
``pg_advisory_lock`` is a running statement holding a snapshot, and an online
index build waits for exactly that: PostgreSQL's ``CREATE INDEX CONCURRENTLY``
held by the lock holder waits on the blocked waiter, which waits on the holder
-- a deadlock. between polls a waiter holds no statement open, and an idle
connection delays no build. (YugabyteDB's blocking form also does not wait:
it answers ``Timed out waiting for Acquire Advisory Lock`` as soon as the lock
is contended.)

**why session-level, on one connection.** the lock has to outlive
transactions: YugabyteDB DDL auto-commits, so the lock stands in for the
transaction that cannot wrap a migration and its bookkeeping row. a session
lock is released only on its own connection, so the caller passes a
:class:`~threetears.core.data.migrations.session.MigrationSession` bound to one
connection, and the release says so loudly when it did not find the lock.

**what ends a build.** a build whose client times out and disconnects keeps
running server-side, keeps its session -- and so keeps this lock -- and blocks
every later build in the database until it finishes. ``pg_cancel_backend``
does not stop such a build; ``pg_terminate_backend`` does, and ending the
session releases the lock with it. code that abandons a long DDL statement
must terminate its backend, not cancel it.

every consumer that runs DDL in a database this platform migrates takes this
same lock, or it reintroduces the concurrency the lock exists to remove. in
3tears that is the migration runner (every writing entry point; the read
paths issue no DDL), ``DataStore.create_table``,
anything run inside ``DataStore.ddl_session``, and
``threetears.agent.tools.migrate_context_items_schema``. DDL 3tears issues
elsewhere is outside this database by construction: L1 SQLite / DuckDB caches
and ``threetears.geo``'s R-Tree are process-local; ``threetears.backup``'s
restore check creates and drops a scratch DATABASE of its own; the
``datasources`` drivers issue no DDL themselves, and their callers' ``CREATE
TABLE AS`` targets the customer warehouse a datasource names; and
``threetears.iam`` only publishes DDL text for a consumer's own migration.
consumers running DDL of their own call :func:`database_ddl_lock` or
``DataStore.ddl_session``.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Final

from threetears.core.data.migrations.errors import DdlLockError, DdlLockReleaseError, DdlLockTimeoutError
from threetears.core.data.migrations.session import MigrationSession
from threetears.observe import get_logger

__all__ = [
    "DDL_LOCK_NAMESPACE",
    "DdlLockLease",
    "DdlLockPolicy",
    "database_ddl_lock",
    "ddl_lock_key",
]

log = get_logger(__name__)

#: first key of the two-int4 advisory lock every DDL job in a database takes.
#: partitions this lock from every other advisory lock the platform uses; the
#: mnemonic is "DD1 0C4" (DDL lock), and it fits a signed int4 so asyncpg binds
#: it to the ``(int4, int4)`` overload. in ``pg_locks`` the lock shows as
#: ``locktype = 'advisory'``, ``classid = DDL_LOCK_NAMESPACE``,
#: ``objid = ddl_lock_key(database)``, ``objsubid = 2``.
DDL_LOCK_NAMESPACE: Final[int] = 0x0DD1_0C4

_CURRENT_DATABASE_SQL = "SELECT current_database() AS database_name"

_TRY_LOCK_SQL = "SELECT pg_try_advisory_lock($1, $2) AS acquired"

_UNLOCK_SQL = "SELECT pg_advisory_unlock($1, $2) AS released"


def ddl_lock_key(database: str) -> int:
    """
    derive the second key of the DDL lock from a database name.

    hashes the name with SHA-256 -- stable across processes and hosts, unlike
    Python's salted ``hash()`` -- and folds the first four bytes into the
    non-negative int4 range, so every pod computing the key for one database
    agrees on the lock. a collision between two databases costs a spurious
    serialisation between them, never a correctness break.

    :param database: database name, as ``current_database()`` returns it
    :ptype database: str
    :return: non-negative int4 lock key
    :rtype: int
    """
    digest = hashlib.sha256(database.encode("utf-8")).digest()
    key = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
    return key


@dataclass(frozen=True, slots=True)
class DdlLockPolicy:
    """
    how a caller waits for the DDL lock.

    :ivar poll_interval: seconds between ``pg_try_advisory_lock`` attempts;
        each attempt is one short statement, and the waiter holds nothing
        open between them
    :ivar max_wait: seconds after which the caller gives up with
        :class:`~threetears.core.data.migrations.errors.DdlLockTimeoutError`;
        ``None`` waits for as long as the holder holds, ``0`` tries once
    :ivar log_interval: seconds between the INFO lines that report a caller
        still waiting, naming the database and the time waited so far
    """

    poll_interval: float = 1.0
    max_wait: float | None = None
    log_interval: float = 30.0

    def __post_init__(self) -> None:
        """
        refuse a policy that would spin, never report, or wait a negative time.

        :raises ValueError: when ``poll_interval`` or ``log_interval`` is not
            positive, or ``max_wait`` is negative
        """
        if self.poll_interval <= 0:
            msg = f"poll_interval must be positive, got {self.poll_interval!r}"
            raise ValueError(msg)
        if self.log_interval <= 0:
            msg = f"log_interval must be positive, got {self.log_interval!r}"
            raise ValueError(msg)
        if self.max_wait is not None and self.max_wait < 0:
            msg = f"max_wait must be None or non-negative, got {self.max_wait!r}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class DdlLockLease:
    """
    the DDL lock, as held by the body of :func:`database_ddl_lock`.

    :ivar database: the database the lock covers
    :ivar key: the second advisory-lock key, ``ddl_lock_key(database)``
    :ivar waited_seconds: how long the caller polled before it got the lock
    """

    database: str
    key: int
    waited_seconds: float


@asynccontextmanager
async def database_ddl_lock(
    session: MigrationSession,
    policy: DdlLockPolicy | None = None,
) -> AsyncIterator[DdlLockLease]:
    """
    hold the DDL lock of the session's database for the body, released on the way out.

    reads ``current_database()`` on the session, polls
    ``pg_try_advisory_lock(DDL_LOCK_NAMESPACE, ddl_lock_key(database))`` until
    it is granted, runs the body, and unlocks on the same session -- after a
    body that returns, raises, or is cancelled alike.

    session-level locks are re-entrant: a session that already holds the lock
    gets it again at once, and each exit gives back one hold.

    :param session: one database session -- every statement of the body must
        run on it, since the lock is released only there
    :ptype session: MigrationSession
    :param policy: how to wait; defaults to a 1s poll with no deadline
    :ptype policy: DdlLockPolicy | None
    :return: async context manager yielding the held lease
    :rtype: AsyncIterator[DdlLockLease]
    :raises DdlLockTimeoutError: when the lock stayed held past
        ``policy.max_wait``; the body never ran
    :raises DdlLockReleaseError: when the body completed but the unlock
        failed or found the lock not held by this session; the lock may still
        be held, so the caller must close or terminate the connection rather
        than reuse it. when the BODY raised (a cancellation included), its own
        exception propagates instead, with the release failure logged at ERROR
        and attached as a note: the body's error is what the caller acts on,
        and a timeout wrapper must still see its cancellation
    :raises DdlLockError: when the session answers the lock statements with no
        row, which no PostgreSQL session does
    """
    effective = policy if policy is not None else DdlLockPolicy()
    database = await _current_database(session)
    key = ddl_lock_key(database)
    lease = await _acquire(session, database, key, effective)
    try:
        yield lease
    except BaseException as body_error:
        await _release_after_failed_body(session, database, key, body_error)
        raise
    await _release(session, database, key)


async def _current_database(session: MigrationSession) -> str:
    """
    read the name of the database the session is connected to.

    :param session: one database session
    :ptype session: MigrationSession
    :return: database name
    :rtype: str
    :raises DdlLockError: when the session returns no row
    """
    rows = await session.query(_CURRENT_DATABASE_SQL)
    if not rows:
        msg = "current_database() returned no row; the DDL lock needs a PostgreSQL-compatible session"
        raise DdlLockError(msg)
    database = str(rows[0]["database_name"])
    return database


async def _acquire(session: MigrationSession, database: str, key: int, policy: DdlLockPolicy) -> DdlLockLease:
    """
    poll for the lock until it is granted or ``policy.max_wait`` runs out.

    :param session: one database session
    :ptype session: MigrationSession
    :param database: the session's database, for logs and errors
    :ptype database: str
    :param key: the lock's second key
    :ptype key: int
    :param policy: how to wait
    :ptype policy: DdlLockPolicy
    :return: the held lease
    :rtype: DdlLockLease
    :raises DdlLockTimeoutError: when ``policy.max_wait`` runs out
    """
    loop = asyncio.get_running_loop()
    started = loop.time()
    next_report = started
    acquired = await _try_lock(session, database, key)
    while not acquired:
        waited = loop.time() - started
        if policy.max_wait is not None and waited >= policy.max_wait:
            log.warning(
                "gave up waiting for the DDL lock database=%s waited=%.1fs max_wait=%.1fs",
                database,
                waited,
                policy.max_wait,
            )
            raise DdlLockTimeoutError(database, waited)
        if loop.time() >= next_report:
            log.info("waiting for the DDL lock database=%s waited=%.1fs", database, waited)
            next_report = loop.time() + policy.log_interval
        delay = policy.poll_interval
        if policy.max_wait is not None:
            delay = min(delay, max(policy.max_wait - waited, 0.0))
        await asyncio.sleep(delay)
        acquired = await _try_lock(session, database, key)
    waited = loop.time() - started
    log.info("DDL lock acquired database=%s waited=%.1fs", database, waited)
    lease = DdlLockLease(database=database, key=key, waited_seconds=waited)
    return lease


async def _try_lock(session: MigrationSession, database: str, key: int) -> bool:
    """
    make one non-blocking attempt at the lock.

    a cancellation can arrive after the server granted the lock and before the
    answer did. the lock would then stay with the session, unowned by any
    caller, so an interrupted attempt gives back whatever it may have taken --
    an unlock of a lock the session does not hold only answers false.

    :param session: one database session
    :ptype session: MigrationSession
    :param database: the session's database, for logs
    :ptype database: str
    :param key: the lock's second key
    :ptype key: int
    :return: True when this attempt was granted the lock
    :rtype: bool
    :raises DdlLockError: when the attempt returns no row
    """
    try:
        rows = await session.query(_TRY_LOCK_SQL, DDL_LOCK_NAMESPACE, key)
    except asyncio.CancelledError:
        await _undo_interrupted_attempt(session, database, key)
        raise
    if not rows:
        msg = f"pg_try_advisory_lock returned no row on database {database!r}; the DDL lock needs a PostgreSQL session"
        raise DdlLockError(msg)
    acquired = bool(rows[0]["acquired"])
    return acquired


async def _undo_interrupted_attempt(session: MigrationSession, database: str, key: int) -> None:
    """
    give back a lock an interrupted attempt may have been granted.

    :param session: one database session
    :ptype session: MigrationSession
    :param database: the session's database, for logs
    :ptype database: str
    :param key: the lock's second key
    :ptype key: int
    """
    try:
        await session.query(_UNLOCK_SQL, DDL_LOCK_NAMESPACE, key)
    except Exception as exc:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- the cancellation propagating past this is the outcome the caller sees; this only reports, loudly, that the session may keep the lock
        log.error(
            "a cancelled attempt at the DDL lock could not be undone database=%s error=%s: %s; the session "
            "may hold the lock until it ends, so close or terminate the connection",
            database,
            type(exc).__name__,
            exc,
        )


async def _release_after_failed_body(
    session: MigrationSession,
    database: str,
    key: int,
    body_error: BaseException,
) -> None:
    """
    give the lock back after a body that raised, never replacing the body's exception.

    the unlock is most likely to fail exactly when the body failed because
    its connection died -- and a dead session has released the lock anyway.
    either way the body's exception is what the caller must see: a caller
    handling a failed migration, or a timeout wrapper waiting for its
    cancellation, would otherwise get a lock error in its place. so a failed
    release is logged at ERROR and attached to the body's exception as a note.

    :param session: the session the lock was taken on
    :ptype session: MigrationSession
    :param database: the session's database, for logs
    :ptype database: str
    :param key: the lock's second key
    :ptype key: int
    :param body_error: the exception the body raised, which the caller re-raises
    :ptype body_error: BaseException
    :raises asyncio.CancelledError: when the release itself is cancelled
    """
    try:
        await _release(session, database, key)
    except DdlLockReleaseError as release_error:
        log.error(
            "the DDL lock was not released after a run that failed with %s database=%s; close or terminate "
            "this connection, and see the run's own error for what went wrong: %s",
            type(body_error).__name__,
            database,
            release_error,
        )
        body_error.add_note(f"while unwinding: {release_error}")


async def _release(session: MigrationSession, database: str, key: int) -> None:
    """
    give the lock back on the session that holds it, loudly when that fails.

    :param session: the session the lock was taken on
    :ptype session: MigrationSession
    :param database: the session's database, for logs and errors
    :ptype database: str
    :param key: the lock's second key
    :ptype key: int
    :raises DdlLockReleaseError: when the unlock fails or reports the session
        did not hold the lock
    :raises asyncio.CancelledError: when the release itself is cancelled; the
        lock is then held until the session ends
    """
    try:
        rows = await session.query(_UNLOCK_SQL, DDL_LOCK_NAMESPACE, key)
    except asyncio.CancelledError:
        log.error(
            "releasing the DDL lock was cancelled database=%s; the session holds it until it ends, so close "
            "or terminate the connection",
            database,
        )
        raise
    except Exception as exc:  # prawduct:allow prawduct/broad-except -- any failure of the unlock leaves the lock possibly held; it is re-raised as DdlLockReleaseError, chained, so the caller cannot mistake it for a released lock
        log.error(
            "the DDL lock could not be released database=%s error=%s: %s",
            database,
            type(exc).__name__,
            exc,
        )
        raise DdlLockReleaseError(database, f"the unlock failed with {type(exc).__name__}: {exc}") from exc
    released = bool(rows and rows[0]["released"])
    if not released:
        log.error(
            "the DDL lock was not held by the session releasing it database=%s; the statements under it did "
            "not all run on one session",
            database,
        )
        raise DdlLockReleaseError(
            database,
            "the session releasing it did not hold it, so the statements under the lock did not all run on "
            "one session and another session may still hold it",
        )
    log.info("DDL lock released database=%s", database)
