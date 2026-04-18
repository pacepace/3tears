"""
canonical migration runner.

composes per-package :class:`~threetears.core.data.migrations.registry.
PackageMigrations` into a single apply sequence. the runner knows the
scope (platform vs agent) of each package, performs topological ordering
across packages using declared ``depends_on`` edges, applies every
pending migration against a DataStore bound to the target schema, and
records applied (version, package) tuples in a ``_schema_migrations``
table.

one runner instance owns registrations for the platform schema and for
every agent schema. :meth:`apply_for_platform_schema` and
:meth:`apply_for_agent_schema` take a DataStore already bound to the
intended schema (via search_path set by the L3 layer). the runner never
hard-codes schema names; that stays the caller's responsibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from threetears.core.data.migrations.errors import (
    MigrationError,
    MigrationFailedError,
    MissingDependencyError,
)
from threetears.core.data.migrations.preview import PreviewStore
from threetears.core.data.migrations.registry import MigrationFunc, PackageMigrations
from threetears.core.data.migrations.scope import MigrationScope
from threetears.observe import get_logger, traced

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

_SELECT_APPLIED_VERSIONS_SQL = (
    "SELECT version, package FROM _schema_migrations ORDER BY version"
)

_SELECT_APPLIED_HISTORY_SQL = (
    "SELECT version, package, description, date_applied "
    "FROM _schema_migrations ORDER BY date_applied, version, package"
)

_SELECT_MAX_VERSION_SQL = (
    "SELECT COALESCE(MAX(version), 0) AS max_version FROM _schema_migrations"
)

_INSERT_VERSION_SQL = (
    "INSERT INTO _schema_migrations (version, package, description) "
    "VALUES ($1, $2, $3)"
)

_DELETE_VERSION_SQL = (
    "DELETE FROM _schema_migrations WHERE version = $1 AND package = $2"
)


class MigrationRunner:
    """
    canonical migration runner composing registered packages.

    the runner is stateful during registration and stateless at apply
    time: every apply method receives a DataStore bound to the target
    schema and uses it both for the migration bodies and for the
    ``_schema_migrations`` bookkeeping.
    """

    def __init__(self) -> None:
        """
        initialize an empty runner with no registered packages.
        """
        self._packages: dict[str, PackageMigrations] = {}

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
    async def apply_for_platform_schema(
        self, store: DataStore, target: int | None = None
    ) -> int:
        """
        apply all pending PLATFORM-scope migrations against store's schema.

        the caller must have bound the DataStore to the target platform
        schema via search_path before calling. the runner does not
        qualify statements with a schema name.

        :param store: DataStore bound to platform schema via search_path
        :ptype store: DataStore
        :param target: optional cap (inclusive) on applied version per
            package; if ``None`` apply everything. ignored when the
            package has no versions at or below the target.
        :ptype target: int | None
        :return: number of migrations applied across all platform packages
        :rtype: int
        :raises MissingDependencyError: on unresolved/cyclic depends_on
        :raises MigrationFailedError: wrapping original migration exception
        """
        result = await self._apply_scope(store, MigrationScope.PLATFORM, target)
        return result

    @traced
    async def apply_for_agent_schema(
        self, store: DataStore, target: int | None = None
    ) -> int:
        """
        apply all pending AGENT-scope migrations against store's schema.

        callers use this after creating an agent schema and setting
        search_path. composes every registered agent-scoped package in
        topological order so every agent schema looks identical after
        provisioning.

        :param store: DataStore bound to agent schema via search_path
        :ptype store: DataStore
        :param target: optional cap (inclusive) on applied version per
            package; if ``None`` apply everything.
        :ptype target: int | None
        :return: number of migrations applied across all agent packages
        :rtype: int
        :raises MissingDependencyError: on unresolved/cyclic depends_on
        :raises MigrationFailedError: wrapping original migration exception
        """
        result = await self._apply_scope(store, MigrationScope.AGENT, target)
        return result

    @traced
    async def preview_for_scope(
        self,
        store: DataStore,
        scope: MigrationScope,
        target: int | None = None,
    ) -> PreviewStore:
        """
        simulate an apply and return the PreviewStore holding captured DDL.

        wraps the caller's DataStore in a :class:`PreviewStore` and runs
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

        :param store: DataStore bound to target schema (read-only use
            except for the one-time ``_schema_migrations`` create)
        :ptype store: DataStore
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
        await self._apply_scope(preview, scope, target)  # type: ignore[arg-type]
        return preview

    @traced
    async def downgrade_for_scope(
        self,
        store: DataStore,
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
        migration blocks the rollback.

        :param store: DataStore bound to target schema
        :ptype store: DataStore
        :param scope: platform or agent scope
        :ptype scope: MigrationScope
        :param steps: number of most-recent migrations to roll back
        :ptype steps: int
        :return: number of migrations rolled back
        :rtype: int
        :raises MigrationError: if any targeted migration has no downgrade
        :raises MigrationFailedError: if a downgrade body raises
        """
        if steps <= 0:
            msg = f"downgrade steps must be >= 1, got {steps}"
            raise MigrationError(msg)
        await self._ensure_migrations_table(store)
        history = await self._get_history(store)
        in_scope = {p.name for p in self._packages.values() if p.scope == scope}
        scope_history = [row for row in history if row["package"] in in_scope]
        if not scope_history:
            return 0
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
    async def get_applied_history(
        self, store: DataStore
    ) -> list[dict[str, Any]]:
        """
        return the applied-migration history as ordered dict rows.

        rows are ordered by date_applied ascending; each row carries
        keys ``version``, ``package``, ``description``, ``date_applied``.

        :param store: DataStore bound to target schema
        :ptype store: DataStore
        :return: chronological list of applied migrations
        :rtype: list[dict[str, Any]]
        """
        await self._ensure_migrations_table(store)
        result = await self._get_history(store)
        return result

    @traced
    async def current_versions(
        self,
        store: DataStore,
        scope: MigrationScope,
    ) -> dict[str, int]:
        """
        return the current max-applied version per package for a scope.

        packages in the requested scope that have no rows applied yet
        return ``0``. packages outside the scope are omitted.

        :param store: DataStore bound to target schema
        :ptype store: DataStore
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
        store: DataStore,
        package_name: str,
        version_num: int,
        description: str = "stamped",
    ) -> None:
        """
        insert a ``_schema_migrations`` row without running any DDL.

        for disaster-recovery use when bookkeeping drifts from reality.
        callers MUST verify the schema matches the claimed state before
        stamping; a stamp with no matching schema is a lie the runner
        will trust forever.

        :param store: DataStore bound to target schema
        :ptype store: DataStore
        :param package_name: package to stamp
        :ptype package_name: str
        :param version_num: version number to record
        :ptype version_num: int
        :param description: description text to record; defaults to
            ``"stamped"`` so the history reads as operator-intervened
        :ptype description: str
        """
        await self._ensure_migrations_table(store)
        await store.execute(
            _INSERT_VERSION_SQL, version_num, package_name, description
        )

    @traced
    async def apply_package(self, store: DataStore, package_name: str) -> int:
        """
        apply one named package's pending migrations against store's schema.

        used by per-package test harnesses that want to exercise a
        single package in isolation (MIG-07). does not resolve
        dependencies; callers must apply any depended-on packages first.

        :param store: DataStore bound to target schema via search_path
        :ptype store: DataStore
        :param package_name: name of the registered package to apply
        :ptype package_name: str
        :return: number of migrations applied for the named package
        :rtype: int
        :raises KeyError: if package_name is not registered
        :raises MigrationFailedError: wrapping original migration exception
        """
        if package_name not in self._packages:
            msg = f"package {package_name!r} not registered"
            raise KeyError(msg)
        package = self._packages[package_name]
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
        docs. does not touch any DataStore.

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

    async def _apply_scope(
        self,
        store: DataStore,
        scope: MigrationScope,
        target: int | None = None,
    ) -> int:
        """
        apply every pending migration for the given scope.

        topologically orders packages, reads the applied-version set
        once, then walks packages applying any version not already
        recorded. halts on first failure after reverting bookkeeping for
        the failing migration.

        :param store: DataStore bound to target schema
        :ptype store: DataStore
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
        count = 0
        for package in ordered:
            applied_count = await self._apply_package_pending(
                store, package, applied, target
            )
            count += applied_count
        return count

    async def _apply_package_pending(
        self,
        store: DataStore,
        package: PackageMigrations,
        applied: set[tuple[int, str]],
        target: int | None = None,
    ) -> int:
        """
        apply pending migrations for one package in ascending version order.

        :param store: DataStore bound to target schema
        :ptype store: DataStore
        :param package: package whose pending migrations run
        :ptype package: PackageMigrations
        :param applied: set of (version, package_name) tuples already recorded
        :ptype applied: set[tuple[int, str]]
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
            applied.add(key)
        return count

    async def _run_one(
        self,
        store: DataStore,
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

        :param store: DataStore bound to target schema
        :ptype store: DataStore
        :param package_name: name of package owning this migration
        :ptype package_name: str
        :param version_num: version number of this migration
        :ptype version_num: int
        :param func: async migration body taking a DataStore
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
            await func(store)
            await store.execute(
                _INSERT_VERSION_SQL, version_num, package_name, description
            )
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
            msg = (
                f"migration {package_name}:{version_num} ({description}) failed: {exc}"
            )
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
        store: DataStore,
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

        :param store: DataStore bound to target schema
        :ptype store: DataStore
        :param package_name: name of package owning the downgrade
        :ptype package_name: str
        :param version_num: version number being rolled back
        :ptype version_num: int
        :param func: async downgrade body taking a DataStore
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
            await func(store)
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

    async def _get_history(
        self, store: DataStore
    ) -> list[dict[str, Any]]:
        """
        read the chronological apply-history from ``_schema_migrations``.

        :param store: DataStore bound to target schema
        :ptype store: DataStore
        :return: list of rows ordered by date_applied
        :rtype: list[dict[str, Any]]
        """
        rows = await store.query(_SELECT_APPLIED_HISTORY_SQL)
        result = [dict(r) for r in rows]
        return result

    async def _ensure_migrations_table(self, store: DataStore) -> None:
        """
        create ``_schema_migrations`` if it does not already exist.

        :param store: DataStore bound to target schema
        :ptype store: DataStore
        """
        await store.execute(_CREATE_MIGRATIONS_TABLE_SQL)

    async def _get_applied_versions(
        self, store: DataStore
    ) -> set[tuple[int, str]]:
        """
        query ``_schema_migrations`` for (version, package) tuples.

        :param store: DataStore bound to target schema
        :ptype store: DataStore
        :return: set of applied (version, package_name) tuples
        :rtype: set[tuple[int, str]]
        """
        rows = await store.query(_SELECT_APPLIED_VERSIONS_SQL)
        result = {(row["version"], row["package"]) for row in rows}
        return result

    def _topological_sort(
        self, scope: MigrationScope
    ) -> list[PackageMigrations]:
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
            name: pkg
            for name, pkg in self._packages.items()
            if pkg.scope == scope
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
            remaining = sorted(
                name for name, deg in indegree.items() if deg > 0
            )
            msg = (
                f"cycle detected or unresolved dependency among packages: "
                f"{remaining!r}"
            )
            raise MissingDependencyError(msg)

        return ordered
