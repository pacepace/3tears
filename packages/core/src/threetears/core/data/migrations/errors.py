"""
migration runner error types.

the concrete error classes cover the distinct failure modes the runner
surfaces to callers: duplicate version registration inside one package,
unresolved or cyclic package dependencies at apply time, failure of an
individual migration body, a bookkeeping ledger that disagrees with the
code about which migration a version is, a pool offered where one
database session is needed, and the database-wide DDL lock not being
taken or not being given back.
"""

from __future__ import annotations

__all__ = [
    "DdlLockError",
    "DdlLockReleaseError",
    "DdlLockTimeoutError",
    "DuplicateVersionError",
    "LedgerMismatchError",
    "MigrationError",
    "MigrationFailedError",
    "MissingDependencyError",
    "SessionRequiredError",
]


class MigrationError(Exception):
    """
    base class for every migration runner error.

    subclasses exist so callers can distinguish structural registration
    errors (caught at test time) from apply-time failures (caught at
    provisioning time). code outside this module catches the base class
    when it needs uniform handling.
    """


class DuplicateVersionError(MigrationError):
    """
    raised when two migration callables are registered at the same
    version within a single package.

    this is a pure structural error: the package's migration authors
    chose the same version twice. the correct remediation is to pick a
    fresh version number, never to let the second registration silently
    replace the first.
    """


class MissingDependencyError(MigrationError):
    """
    raised when the runner cannot topologically order registered packages.

    two distinct conditions produce this error:

    - a package declares ``depends_on`` a name no registered package
      provides.
    - two packages declare a cycle via mutual or transitive depends_on.

    both are authoring bugs. catching the error during apply exposes the
    gap at provision time rather than letting migrations run in a
    non-deterministic order.
    """


class LedgerMismatchError(MigrationError):
    """
    raised when ``_schema_migrations`` records a different migration at a
    version than the code registers there.

    the condition that produces it is renumbering: a branch shifts its
    migrations to make room for one that landed on the mainline, and is
    then pointed at a database that applied the OLD numbering. the runner
    decides what is pending from ``(version, package)`` alone, so every
    shifted version reads as already applied and its body never runs —
    including the mainline migration that took the vacated number.

    this fails closed rather than warning. the alternative is a service
    that starts healthy, reports nothing pending, and raises an undefined-
    column error hours later on an endpoint that has nothing to do with
    the migration that was skipped.

    the remedy is never to hand-edit the ledger to match: the schema and
    the ledger have genuinely diverged, and only one of them says so. on a
    disposable database, recreate it. on one that is not, apply the
    skipped migrations deliberately and stamp them.
    """


class MigrationFailedError(MigrationError):
    """
    raised when an individual migration body raises during apply.

    the runner wraps the original exception so callers see a uniform
    surface while retaining the underlying cause via ``__cause__``. the
    runner halts the apply sequence on the first failure and reverts
    bookkeeping for the failing migration only — previously-applied
    migrations keep their recorded version.
    """


class SessionRequiredError(MigrationError):
    """
    raised when something shaped like a pool is offered as one database session.

    :class:`~threetears.core.data.migrations.session.ConnectionSession` raises
    it for anything with ``acquire()``. the DDL lock is a session-level
    advisory lock, and a session lock belongs to one connection: a pool takes
    it on one connection, runs the DDL on others and releases on whichever it
    gets last -- and asyncpg's pool drops every advisory lock a connection
    holds when the connection is returned, so the lock would guard nothing.
    the message names what was passed and how to acquire one connection.
    """


class DdlLockError(MigrationError):
    """
    base class for failures of the database-wide DDL lock.

    see :mod:`threetears.core.data.migrations.ddl_lock` for why the lock is
    per database rather than per schema.
    """


class DdlLockTimeoutError(DdlLockError):
    """
    raised when the database-wide DDL lock stayed held past the caller's ``max_wait``.

    nothing has run under the lock when this is raised: the caller never held
    it. the holder is another migration or DDL job in the same database, and on
    YugabyteDB possibly an index build whose client gave up while the build
    kept running server-side -- such a build holds its session, and so this
    lock, until it finishes or its backend is ended with
    ``pg_terminate_backend``.

    :ivar database: name of the database whose lock was busy
    :ivar waited_seconds: how long the caller polled before giving up
    """

    def __init__(self, database: str, waited_seconds: float) -> None:
        """
        record which database's lock was busy and for how long.

        :param database: name of the database whose lock was busy
        :ptype database: str
        :param waited_seconds: how long the caller polled before giving up
        :ptype waited_seconds: float
        """
        self.database = database
        self.waited_seconds = waited_seconds
        msg = (
            f"the DDL lock for database {database!r} was still held by another session after "
            f"{waited_seconds:.1f}s; nothing was run. find the holder in pg_locks (locktype advisory, "
            f"objsubid 2) and pg_stat_activity; a build left running by a client that gave up is ended "
            f"only by pg_terminate_backend"
        )
        super().__init__(msg)


class DdlLockReleaseError(DdlLockError):
    """
    raised when the database-wide DDL lock could not be given back.

    either the unlock statement failed, or it ran and reported the session did
    not hold the lock -- which means the statements under it did not all run on
    one session. either way the lock may still be held by some session, and
    every later DDL job in the database will wait on it. the caller must not
    return the connection to a pool: close or terminate it, which ends the
    session and every lock it holds.

    :ivar database: name of the database whose lock was not released
    """

    def __init__(self, database: str, detail: str) -> None:
        """
        record which database's lock was not released and why.

        :param database: name of the database whose lock was not released
        :ptype database: str
        :param detail: what the unlock did instead of releasing the lock
        :ptype detail: str
        """
        self.database = database
        msg = (
            f"the DDL lock for database {database!r} was not released: {detail}. every later DDL job in "
            f"this database waits on it until the session holding it ends; close or terminate this "
            f"connection rather than returning it to a pool"
        )
        super().__init__(msg)
