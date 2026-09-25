"""
canonical migration runner.

composes per-package :class:`~threetears.core.data.migrations.registry.
PackageMigrations` into a single apply sequence. the runner knows the
scope (platform vs agent) of each package, performs topological ordering
across packages using declared ``depends_on`` edges, applies every
pending migration against a session bound to the target schema, and
records applied (version, package) tuples in a ``_schema_migrations``
table.

one runner instance owns registrations for the platform schema and for
every agent schema. :meth:`apply_for_platform_schema` and
:meth:`apply_for_agent_schema` take a
:class:`~threetears.core.data.migrations.session.MigrationSession` -- one
database connection -- already bound to the intended schema via its
search_path. the runner never hard-codes schema names; that stays the
caller's responsibility.

every run that writes holds the database-wide DDL lock
(:func:`~threetears.core.data.migrations.ddl_lock.database_ddl_lock`) from
before it reads ``_schema_migrations`` until after it writes the last row:
one migration per DATABASE at a time, whichever schema it targets, because
two concurrent index builds in one YugabyteDB database hang each other
until the catalog-version wait times out. the lock is a session lock, so
the store must be one connection; a pool-backed
:class:`~threetears.core.data.store.DataStore` is refused before any
statement runs.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, AsyncIterator, Mapping, cast

from threetears.core.data.migrations.ddl_lock import DdlLockPolicy, database_ddl_lock
from threetears.core.data.migrations.errors import (
    LedgerMismatchError,
    MigrationError,
    MigrationFailedError,
    MissingDependencyError,
    SessionRequiredError,
)
from threetears.core.data.migrations.preview import PreviewStore
from threetears.core.data.migrations.registry import MigrationFunc, PackageMigrations
from threetears.core.data.migrations.scope import MigrationScope
from threetears.core.data.migrations.session import MigrationSession
from threetears.observe import get_logger, traced

__all__ = [
    "MigrationRunner",
]

if TYPE_CHECKING:
    from threetears.core.data.store import DataStore

log = get_logger(__name__)


_CREATE_MIGRATIONS_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS _schema_migrations ("
    "version INTEGER NOT NULL, "
    "package VARCHAR(255) NOT NULL, "
    "description TEXT, "
    "date_applied TIMESTAMP NOT NULL DEFAULT now(), "
    "PRIMARY KEY (version, package)"
    ")"
)

#: ``description`` rides along because the runner verifies it, not merely
#: records it: it is the only evidence in the database of WHICH migration a
#: version number was, and reading it back is what catches a renumbering.
_SELECT_APPLIED_VERSIONS_SQL = "SELECT version, package, description FROM _schema_migrations ORDER BY version"

_SELECT_APPLIED_HISTORY_SQL = (
    "SELECT version, package, description, date_applied FROM _schema_migrations ORDER BY date_applied, version, package"
)

_SELECT_MAX_VERSION_SQL = "SELECT COALESCE(MAX(version), 0) AS max_version FROM _schema_migrations"

_INSERT_VERSION_SQL = "INSERT INTO _schema_migrations (version, package, description) VALUES ($1, $2, $3)"

_DELETE_VERSION_SQL = "DELETE FROM _schema_migrations WHERE version = $1 AND package = $2"


class MigrationRunner:
    """
    canonical migration runner composing registered packages.

    the runner is stateful during registration and stateless at apply
    time: every apply method receives a session bound to the target
    schema and uses it for the database-wide DDL lock, the migration
    bodies and the ``_schema_migrations`` bookkeeping alike.

    :param lock_policy: how a run waits for the database-wide DDL lock;
        defaults to a 1s poll with no deadline, reporting every 30s
    :ptype lock_policy: DdlLockPolicy | None
    """

    def __init__(self, *, lock_policy: DdlLockPolicy | None = None) -> None:
        """
        initialize an empty runner with no registered packages.

        :param lock_policy: how a run waits for the database-wide DDL lock;
            defaults to a 1s poll with no deadline, reporting every 30s
        :ptype lock_policy: DdlLockPolicy | None
        """
        self._packages: dict[str, PackageMigrations] = {}
        self._lock_policy = lock_policy if lock_policy is not None else DdlLockPolicy()

    @property
    def packages(self) -> Mapping[str, PackageMigrations]:
        """return an immutable view of registered migration packages.

        the hub CLI walks this mapping to render ``migrations status``,
        ``migrations history``, and ``migrations current`` for every
        registered package without reaching into the runner's private
        storage. the returned object is a :class:`types.MappingProxyType`
        wrapper over the internal dict: it supports ``in``, ``len``,
        iteration, and ``.keys()``/``.values()``/``.items()`` /
        ``.get()`` with the same semantics as the underlying dict, but
        mutation attempts (``packages["x"] = y``, ``packages.pop(...)``,
        ``packages.clear()``) raise :class:`TypeError`. this protects
        the runner's registration invariants (one :class:`
        PackageMigrations` per name; topological ordering relies on
        consistent membership) from accidental caller-side edits while
        still giving read-only consumers a direct, zero-copy view. the
        view is live: later :meth:`register` calls are visible to
        existing references, so callers holding a ``packages`` handle
        across a registration do not need to re-fetch. call
        :meth:`register` to add a package; there is no public remove
        operation by design (migration registrations are append-only
        within a process lifetime).

        :return: read-only mapping from package name to its
            :class:`PackageMigrations`
        :rtype: Mapping[str, PackageMigrations]
        """
        return MappingProxyType(self._packages)

    def register(self, package: PackageMigrations) -> None:
        """
        register a PackageMigrations instance with the runner.

        the runner holds one registration per package name; registering
        the same name twice replaces the prior entry. callers typically
        register once at import time in a central composition root.

        :param package: package-scoped migration registrations
        :ptype package: PackageMigrations
        """
        self._packages[package.name] = package

    @traced
    async def apply_for_platform_schema(self, store: MigrationSession, target: int | None = None) -> int:
        """
        apply all pending PLATFORM-scope migrations against store's schema.

        the caller must have bound the session to the target platform
        schema via search_path before calling. the runner does not
        qualify statements with a schema name. the whole run holds the
        database-wide DDL lock.

        :param store: one database session bound to the platform schema via
            search_path
        :ptype store: MigrationSession
        :param target: optional cap (inclusive) on applied version per
            package; if ``None`` apply everything. ignored when the
            package has no versions at or below the target.
        :ptype target: int | None
        :return: number of migrations applied across all platform packages
        :rtype: int
        :raises SessionRequiredError: when ``store`` is a pool-backed DataStore
        :raises DdlLockTimeoutError: when the lock stayed held past the
            runner's ``lock_policy.max_wait``
        :raises DdlLockReleaseError: when the lock could not be given back
        :raises MissingDependencyError: on unresolved/cyclic depends_on
        :raises MigrationFailedError: wrapping original migration exception
        """
        async with self._ddl_lock(store):
            result = await self._apply_scope(store, MigrationScope.PLATFORM, target)
        return result

    @traced
    async def apply_for_agent_schema(self, store: MigrationSession, target: int | None = None) -> int:
        """
        apply all pending AGENT-scope migrations against store's schema.

        callers use this after creating an agent schema and setting
        search_path. composes every registered agent-scoped package in
        topological order so every agent schema looks identical after
        provisioning. the whole run holds the database-wide DDL lock, so
        runs against two agent schemas of one database take turns.

        :param store: one database session bound to the agent schema via
            search_path
        :ptype store: MigrationSession
        :param target: optional cap (inclusive) on applied version per
            package; if ``None`` apply everything.
        :ptype target: int | None
        :return: number of migrations applied across all agent packages
        :rtype: int
        :raises SessionRequiredError: when ``store`` is a pool-backed DataStore
        :raises DdlLockTimeoutError: when the lock stayed held past the
            runner's ``lock_policy.max_wait``
        :raises DdlLockReleaseError: when the lock could not be given back
        :raises MissingDependencyError: on unresolved/cyclic depends_on
        :raises MigrationFailedError: wrapping original migration exception
        """
        async with self._ddl_lock(store):
            result = await self._apply_scope(store, MigrationScope.AGENT, target)
        return result

    @traced
    async def preview_for_scope(
        self,
        store: MigrationSession,
        scope: MigrationScope,
        target: int | None = None,
    ) -> PreviewStore:
        """
        simulate an apply and return the PreviewStore holding captured DDL.

        wraps the caller's store in a :class:`PreviewStore` and runs
        the normal apply sequence against the wrapper. the underlying
        store is only read from (for ``_schema_migrations`` bookkeeping
        SELECTs) and is never mutated by the preview sequence itself.

        the one side effect the preview MUST have on the underlying
        store is ``CREATE TABLE IF NOT EXISTS _schema_migrations``: the
        runner queries that table to decide which versions are pending,
        and the query cannot succeed against a fresh schema without it.
        the bookkeeping table is harmless (empty on fresh schemas) and
        its creation is idempotent. every other ``execute`` call the
        runner would issue is captured, not run.

        the returned PreviewStore exposes :meth:`captured_ddl` for a
        plain list of DDL strings and :meth:`captured_statements` for
        the full sequence including bookkeeping entries.

        :param store: store bound to target schema (read-only use except
            for the one-time ``_schema_migrations`` create); a preview takes
            no lock, so a pool-backed store is fine here
        :ptype store: MigrationSession
        :param scope: platform or agent scope
        :ptype scope: MigrationScope
        :param target: optional cap (inclusive) on applied version per
            package; if ``None`` preview everything pending.
        :ptype target: int | None
        :return: wrapper holding the captured statements
        :rtype: PreviewStore
        :raises MissingDependencyError: on unresolved/cyclic depends_on
        """
        # ensure the bookkeeping table exists on the underlying store so
        # the preview's read-through query against it succeeds. this is
        # a one-time idempotent CREATE; it neither writes any rows nor
        # hides pending migrations from the capture.
        await self._ensure_migrations_table(store)
        preview = PreviewStore(underlying=store)
        await self._apply_scope(preview, scope, target)
        return preview

    @traced
    async def downgrade_for_scope(
        self,
        store: MigrationSession,
        scope: MigrationScope,
        steps: int = 1,
    ) -> int:
        """
        roll back the last N applied migrations for the given scope.

        resolves the most-recently-applied migrations within the scope
        (ordered by date_applied descending) and executes each package's
        downgrade callable for them, most-recent first. removes the
        corresponding ``_schema_migrations`` rows.

        refuses to run if any targeted migration has no registered
        downgrade callable — raises :class:`MigrationError` naming the
        package and version, so the operator knows exactly which
        migration blocks the rollback. the rollback holds the
        database-wide DDL lock throughout.

        :param store: one database session bound to the target schema
        :ptype store: MigrationSession
        :param scope: platform or agent scope
        :ptype scope: MigrationScope
        :param steps: number of most-recent migrations to roll back
        :ptype steps: int
        :return: number of migrations rolled back
        :rtype: int
        :raises SessionRequiredError: when ``store`` is a pool-backed DataStore
        :raises DdlLockTimeoutError: when the lock stayed held past the
            runner's ``lock_policy.max_wait``
        :raises DdlLockReleaseError: when the lock could not be given back
        :raises MigrationError: if any targeted migration has no downgrade
        :raises MigrationFailedError: if a downgrade body raises
        """
        if steps <= 0:
            msg = f"downgrade steps must be >= 1, got {steps}"
            raise MigrationError(msg)
        async with self._ddl_lock(store):
            await self._ensure_migrations_table(store)
            history = await self._get_history(store)
            in_scope = {p.name for p in self._packages.values() if p.scope == scope}
            scope_history = [row for row in history if row["package"] in in_scope]
            if not scope_history:
                count = 0
            else:
                # roll back from most-recent backwards
                targets = list(reversed(scope_history))[:steps]
                # validate every target has a registered downgrade before
                # running any. partial rollbacks produce ambiguous DB state.
                for row in targets:
                    pkg_name = row["package"]
                    version_num = row["version"]
                    pkg = self._packages[pkg_name]
                    if version_num not in pkg.downgrades:
                        msg = (
                            f"no downgrade registered for {pkg_name}:{version_num}; "
                            "cannot roll back. add a @pkg.downgrade(N) callable "
                            "or use 'stamp --force' to reset bookkeeping manually."
                        )
                        raise MigrationError(msg)
                count = 0
                for row in targets:
                    pkg_name = row["package"]
                    version_num = row["version"]
                    pkg = self._packages[pkg_name]
                    down = pkg.downgrades[version_num]
                    count += await self._run_downgrade(store, pkg_name, version_num, down)
        return count

    @traced
    async def get_applied_history(self, store: MigrationSession) -> list[dict[str, Any]]:
        """
        return the applied-migration history as ordered dict rows.

        rows are ordered by date_applied ascending; each row carries
        keys ``version``, ``package``, ``description``, ``date_applied``.
        a read: it takes no lock.

        :param store: store bound to target schema
        :ptype store: MigrationSession
        :return: chronological list of applied migrations
        :rtype: list[dict[str, Any]]
        """
        await self._ensure_migrations_table(store)
        result = await self._get_history(store)
        return result

    @traced
    async def current_versions(
        self,
        store: MigrationSession,
        scope: MigrationScope,
    ) -> dict[str, int]:
        """
        return the current max-applied version per package for a scope.

        packages in the requested scope that have no rows applied yet
        return ``0``. packages outside the scope are omitted. a read: it
        takes no lock.

        :param store: store bound to target schema
        :ptype store: MigrationSession
        :param scope: platform or agent scope
        :ptype scope: MigrationScope
        :return: mapping of package name to current version
        :rtype: dict[str, int]
        """
        await self._ensure_migrations_table(store)
        history = await self._get_history(store)
        in_scope = {p.name for p in self._packages.values() if p.scope == scope}
        per_package: dict[str, int] = dict.fromkeys(in_scope, 0)
        for row in history:
            pkg = row["package"]
            ver = int(row["version"])
            if pkg in per_package and ver > per_package[pkg]:
                per_package[pkg] = ver
        return per_package

    @traced
    async def stamp_version(
        self,
        store: MigrationSession,
        package_name: str,
        version_num: int,
        description: str = "stamped",
    ) -> None:
        """
        insert a ``_schema_migrations`` row without running any migration body.

        for disaster-recovery use when bookkeeping drifts from reality.
        callers MUST verify the schema matches the claimed state before
        stamping; a stamp with no matching schema is a lie the runner
        will trust forever. the stamp holds the database-wide DDL lock,
        so it cannot land in the middle of a run reading the same ledger.

        :param store: one database session bound to the target schema
        :ptype store: MigrationSession
        :param package_name: package to stamp
        :ptype package_name: str
        :param version_num: version number to record
        :ptype version_num: int
        :param description: description text to record; defaults to
            ``"stamped"`` so the history reads as operator-intervened
        :ptype description: str
        :raises SessionRequiredError: when ``store`` is a pool-backed DataStore
        :raises DdlLockTimeoutError: when the lock stayed held past the
            runner's ``lock_policy.max_wait``
        :raises DdlLockReleaseError: when the lock could not be given back
        """
        async with self._ddl_lock(store):
            await self._ensure_migrations_table(store)
            await store.execute(_INSERT_VERSION_SQL, version_num, package_name, description)

    @traced
    async def apply_package(self, store: MigrationSession, package_name: str) -> int:
        """
        apply one named package's pending migrations against store's schema.

        used by per-package test harnesses that want to exercise a
        single package in isolation (MIG-07). does not resolve
        dependencies; callers must apply any depended-on packages first.
        holds the database-wide DDL lock like every other apply.

        :param store: one database session bound to the target schema via
            search_path
        :ptype store: MigrationSession
        :param package_name: name of the registered package to apply
        :ptype package_name: str
        :return: number of migrations applied for the named package
        :rtype: int
        :raises KeyError: if package_name is not registered
        :raises SessionRequiredError: when ``store`` is a pool-backed DataStore
        :raises DdlLockTimeoutError: when the lock stayed held past the
            runner's ``lock_policy.max_wait``
        :raises DdlLockReleaseError: when the lock could not be given back
        :raises MigrationFailedError: wrapping original migration exception
        """
        if package_name not in self._packages:
            msg = f"package {package_name!r} not registered"
            raise KeyError(msg)
        package = self._packages[package_name]
        async with self._ddl_lock(store):
            await self._ensure_migrations_table(store)
            applied = await self._get_applied_versions(store)
            count = await self._apply_package_pending(store, package, applied)
        return count

    def pending_sequence(self, scope: MigrationScope) -> list[tuple[str, int]]:
        """
        return ordered list of (package_name, version) tuples for a scope.

        returned list reflects topological ordering at the time of call;
        it ignores whether specific versions have already been applied,
        so it is useful for test introspection and migration authoring
        docs. does not touch any database.

        :param scope: platform or agent scope
        :ptype scope: MigrationScope
        :return: ordered (package_name, version) tuples
        :rtype: list[tuple[str, int]]
        :raises MissingDependencyError: on unresolved/cyclic depends_on
        """
        ordered_packages = self._topological_sort(scope)
        sequence: list[tuple[str, int]] = []
        for package in ordered_packages:
            for version_num in sorted(package.versions.keys()):
                sequence.append((package.name, version_num))
        return sequence

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def _ddl_lock(self, store: MigrationSession) -> AsyncIterator[None]:
        """
        hold the database-wide DDL lock across a migration critical section.

        refuses a :class:`~threetears.core.data.store.DataStore` first:
        it sends every statement through its registry's L3 pool, so the
        lock would be taken on one pooled connection -- and, on asyncpg,
        dropped by the pool's reset the moment that connection is handed
        back -- while the DDL ran on others. then takes
        :func:`~threetears.core.data.migrations.ddl_lock.database_ddl_lock`
        on the store with this runner's policy, and releases it in the
        lock's own ``finally``, so a raising or cancelled run still frees
        it.

        :param store: one database session bound to the target schema
        :ptype store: MigrationSession
        :return: async context manager holding the lock
        :rtype: AsyncIterator[None]
        :raises SessionRequiredError: when ``store`` is a DataStore
        """
        _refuse_pool_backed_store(store)
        async with database_ddl_lock(store, self._lock_policy):
            yield

    async def _apply_scope(
        self,
        store: MigrationSession,
        scope: MigrationScope,
        target: int | None = None,
    ) -> int:
        """
        apply every pending migration for the given scope.

        topologically orders packages, reads the applied-version set
        once, then walks packages applying any version not already
        recorded. halts on first failure after reverting bookkeeping for
        the failing migration.

        :param store: session bound to target schema
        :ptype store: MigrationSession
        :param scope: scope filter for registered packages
        :ptype scope: MigrationScope
        :param target: optional cap (inclusive) on version per package
        :ptype target: int | None
        :return: count of successful migrations applied
        :rtype: int
        :raises MissingDependencyError: on unresolved/cyclic depends_on
        :raises MigrationFailedError: wrapping original migration exception
        """
        ordered = self._topological_sort(scope)
        await self._ensure_migrations_table(store)
        applied = await self._get_applied_versions(store)
        # before any body runs: a ledger that disagrees with this build is not
        # something to apply the pending tail on top of. doing so would write
        # new rows into bookkeeping already known to be describing a different
        # sequence of migrations than the one that actually ran.
        self._verify_ledger_identity(ordered, applied)
        count = 0
        for package in ordered:
            applied_count = await self._apply_package_pending(store, package, applied, target)
            count += applied_count
        return count

    async def _apply_package_pending(
        self,
        store: MigrationSession,
        package: PackageMigrations,
        applied: dict[tuple[int, str], str | None],
        target: int | None = None,
    ) -> int:
        """
        apply pending migrations for one package in ascending version order.

        :param store: session bound to target schema
        :ptype store: MigrationSession
        :param package: package whose pending migrations run
        :ptype package: PackageMigrations
        :param applied: mapping of already-recorded (version, package_name) to
            description; verified against this build by
            :meth:`_verify_ledger_identity` before this runs
        :ptype applied: dict[tuple[int, str], str | None]
        :param target: optional cap (inclusive) on version within the package
        :ptype target: int | None
        :return: count of migrations this call applied
        :rtype: int
        :raises MigrationFailedError: wrapping original migration exception
        """
        count = 0
        for version_num in sorted(package.versions.keys()):
            if target is not None and version_num > target:
                break
            key = (version_num, package.name)
            if key in applied:
                continue
            func = package.versions[version_num]
            count += await self._run_one(store, package.name, version_num, func)
            applied[key] = func.__name__
        return count

    async def _run_one(
        self,
        store: MigrationSession,
        package_name: str,
        version_num: int,
        func: MigrationFunc,
    ) -> int:
        """
        execute one migration callable and record it in ``_schema_migrations``.

        on exception, reverts the bookkeeping row for this migration and
        raises :class:`MigrationFailedError` so the caller can drop the
        schema cleanly. previously-applied migrations keep their
        recorded version — only the failing migration is reverted.

        :param store: session bound to target schema
        :ptype store: MigrationSession
        :param package_name: name of package owning this migration
        :ptype package_name: str
        :param version_num: version number of this migration
        :ptype version_num: int
        :param func: async migration body taking the run's session
        :ptype func: MigrationFunc
        :return: 1 on success (return type matches caller's counter)
        :rtype: int
        :raises MigrationFailedError: wrapping original migration exception
        """
        description = func.__name__
        log.info(
            "applying migration package=%s version=%d description=%s",
            package_name,
            version_num,
            description,
        )
        try:
            await _invoke(func, store)
            await store.execute(_INSERT_VERSION_SQL, version_num, package_name, description)
        except Exception as exc:
            # best-effort revert of the version row in case the migration
            # body partially inserted the bookkeeping (it should not, but
            # belt-and-suspenders is the right default in a runner).
            try:
                await store.execute(_DELETE_VERSION_SQL, version_num, package_name)
            except Exception as revert_exc:
                log.warning(
                    "revert of _schema_migrations row failed: package=%s version=%d error=%s",
                    package_name,
                    version_num,
                    revert_exc,
                )
            msg = f"migration {package_name}:{version_num} ({description}) failed: {exc}"
            raise MigrationFailedError(msg) from exc
        log.info(
            "migration applied package=%s version=%d description=%s",
            package_name,
            version_num,
            description,
        )
        return 1

    async def _run_downgrade(
        self,
        store: MigrationSession,
        package_name: str,
        version_num: int,
        func: MigrationFunc,
    ) -> int:
        """
        execute one downgrade callable and delete its ``_schema_migrations`` row.

        on exception re-raises as :class:`MigrationFailedError` naming
        the package and version so the operator sees which downgrade
        failed. does not attempt to re-apply the upgrade — a failed
        downgrade leaves the bookkeeping row intact so the schema is
        still provably at version N; the operator decides how to
        proceed.

        :param store: session bound to target schema
        :ptype store: MigrationSession
        :param package_name: name of package owning the downgrade
        :ptype package_name: str
        :param version_num: version number being rolled back
        :ptype version_num: int
        :param func: async downgrade body taking the run's session
        :ptype func: MigrationFunc
        :return: 1 on success
        :rtype: int
        :raises MigrationFailedError: wrapping original downgrade exception
        """
        description = func.__name__
        log.info(
            "rolling back migration package=%s version=%d description=%s",
            package_name,
            version_num,
            description,
        )
        try:
            await _invoke(func, store)
            await store.execute(_DELETE_VERSION_SQL, version_num, package_name)
        except Exception as exc:
            msg = (
                f"downgrade {package_name}:{version_num} ({description}) failed: {exc}. "
                f"bookkeeping row left intact so apparent version is unchanged."
            )
            raise MigrationFailedError(msg) from exc
        log.info(
            "migration rolled back package=%s version=%d description=%s",
            package_name,
            version_num,
            description,
        )
        return 1

    async def _get_history(self, store: MigrationSession) -> list[dict[str, Any]]:
        """
        read the chronological apply-history from ``_schema_migrations``.

        :param store: session bound to target schema
        :ptype store: MigrationSession
        :return: list of rows ordered by date_applied
        :rtype: list[dict[str, Any]]
        """
        rows = await store.query(_SELECT_APPLIED_HISTORY_SQL)
        result = [dict(r) for r in rows]
        return result

    async def _ensure_migrations_table(self, store: MigrationSession) -> None:
        """
        create ``_schema_migrations`` if it does not already exist.

        :param store: session bound to target schema
        :ptype store: MigrationSession
        """
        await store.execute(_CREATE_MIGRATIONS_TABLE_SQL)

    async def _get_applied_versions(self, store: MigrationSession) -> dict[tuple[int, str], str | None]:
        """
        query ``_schema_migrations`` for applied versions and their descriptions.

        a mapping rather than a set because the description is what
        :meth:`_verify_ledger_identity` compares; membership alone cannot
        distinguish a version that ran from a version whose NUMBER ran
        carrying somebody else's migration.

        :param store: session bound to target schema
        :ptype store: MigrationSession
        :return: mapping of (version, package_name) to recorded description
        :rtype: dict[tuple[int, str], str | None]
        """
        rows = await store.query(_SELECT_APPLIED_VERSIONS_SQL)
        result = {(row["version"], row["package"]): row.get("description") for row in rows}
        return result

    def _verify_ledger_identity(
        self,
        ordered: list[PackageMigrations],
        applied: dict[tuple[int, str], str | None],
    ) -> None:
        """
        refuse to apply when the ledger names a different migration than the code.

        compares the ``description`` recorded at apply time
        (``func.__name__``) against the name of the callable the code now
        registers at that version. they diverge when migrations are
        renumbered and the resulting build meets a database that applied
        the old numbering: every shifted version reads as already applied,
        so its body never runs, and the version vacated at the bottom of
        the shift is skipped entirely.

        only versions this build still defines are compared. a ledger row
        AHEAD of the code is an older deployment meeting a database a
        newer one has already migrated -- an ordinary staged rollout, and
        failing it here would make one an outage.

        a row with no recorded description is left alone: it predates the
        runner recording one, and inventing a verdict from its absence
        would fail every database old enough to have one.

        :param ordered: packages in the scope being applied
        :ptype ordered: list[PackageMigrations]
        :param applied: mapping of (version, package_name) to recorded description
        :ptype applied: dict[tuple[int, str], str | None]
        :return: nothing when every comparable row agrees
        :rtype: None
        :raises LedgerMismatchError: naming the version and both migrations
        """
        mismatches: list[str] = []
        for package in ordered:
            for version_num, func in sorted(package.versions.items()):
                recorded = applied.get((version_num, package.name))
                if recorded is None:
                    continue
                if recorded != func.__name__:
                    mismatches.append(
                        f"{package.name}:{version_num} recorded as '{recorded}' but this build has '{func.__name__}'",
                    )
        if mismatches:
            joined = "; ".join(mismatches)
            msg = (
                f"_schema_migrations disagrees with this build about {len(mismatches)} "
                f"migration(s): {joined}. this database applied a different numbering, so "
                f"every listed version reads as already applied and its body never ran. "
                f"recreate the database, or apply the skipped migrations deliberately and "
                f"stamp them -- do not edit the ledger to match"
            )
            raise LedgerMismatchError(msg)

    def _topological_sort(self, scope: MigrationScope) -> list[PackageMigrations]:
        """
        topologically order packages in the given scope by depends_on.

        implements Kahn's algorithm over the scope's registered
        packages. packages outside the requested scope are filtered out
        before sorting so platform/agent scopes order independently.
        depends_on edges that point to packages in a different scope
        still have to resolve — they are treated as missing — because
        platform and agent schemas cannot depend on each other at apply
        time anyway.

        :param scope: scope to filter on
        :ptype scope: MigrationScope
        :return: packages ordered so dependencies precede dependents
        :rtype: list[PackageMigrations]
        :raises MissingDependencyError: on unresolved/cyclic depends_on
        """
        # Filter to the requested scope.
        in_scope: dict[str, PackageMigrations] = {
            name: pkg for name, pkg in self._packages.items() if pkg.scope == scope
        }

        # Build indegree map and adjacency list restricted to in-scope packages.
        indegree: dict[str, int] = dict.fromkeys(in_scope, 0)
        dependents: dict[str, list[str]] = {name: [] for name in in_scope}
        for pkg in in_scope.values():
            for dep_name in pkg.depends_on:
                if dep_name not in in_scope:
                    msg = (
                        f"package {pkg.name!r} (scope={scope.value}) depends_on "
                        f"{dep_name!r} which is not registered in the same scope"
                    )
                    raise MissingDependencyError(msg)
                indegree[pkg.name] += 1
                dependents[dep_name].append(pkg.name)

        # Kahn's algorithm. To make ordering deterministic regardless of
        # Python dict insertion order we pick the alphabetically-smallest
        # zero-indegree node at each step.
        queue: list[str] = sorted(name for name, deg in indegree.items() if deg == 0)
        ordered: list[PackageMigrations] = []
        while queue:
            name = queue.pop(0)
            ordered.append(in_scope[name])
            for dependent in dependents[name]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    # keep queue sorted so ordering is deterministic
                    queue.append(dependent)
                    queue.sort()

        if len(ordered) != len(in_scope):
            remaining = sorted(name for name, deg in indegree.items() if deg > 0)
            msg = f"cycle detected or unresolved dependency among packages: {remaining!r}"
            raise MissingDependencyError(msg)

        return ordered


def _refuse_pool_backed_store(store: MigrationSession) -> None:
    """
    refuse a DataStore as the store of a locked migration run.

    :param store: the store a run was handed
    :ptype store: MigrationSession
    :raises SessionRequiredError: when ``store`` is a DataStore
    """
    # imported here: store.py imports this package for its own annotations,
    # so a module-level import would be circular
    from threetears.core.data.store import DataStore  # noqa: PLC0415

    if isinstance(store, DataStore):
        msg = (
            "MigrationRunner needs one database session and was given a DataStore, which sends each "
            "statement to whichever pooled connection is free; the database-wide DDL lock taken that way "
            "is not held across the run. acquire one connection and pass ConnectionSession(conn): "
            "async with pool.acquire() as conn: await runner.apply_for_agent_schema(ConnectionSession(conn))"
        )
        raise SessionRequiredError(msg)


async def _invoke(func: MigrationFunc, store: MigrationSession) -> None:
    """
    run one migration or downgrade body against the run's session.

    bodies are annotated ``store: DataStore`` by convention (that is what
    :data:`~threetears.core.data.migrations.registry.MigrationFunc` names),
    yet every production run hands them a one-connection session -- the
    runner refuses a DataStore. what a body may use is the session's
    ``execute`` / ``query`` surface, which DataStore shares, so the cast
    names the convention rather than changing what runs.

    :param func: migration or downgrade body
    :ptype func: MigrationFunc
    :param store: the run's session
    :ptype store: MigrationSession
    """
    await func(cast("DataStore", store))
