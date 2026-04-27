"""
package-scoped migration registration.

a :class:`PackageMigrations` holds the ordered set of version-tagged
async callables contributed by a single package (e.g. agent-workspace,
agent-memory). the canonical :class:`~threetears.core.data.migrations.
runner.MigrationRunner` composes multiple PackageMigrations into a
single topologically-ordered apply sequence.

package registration replaces the old pattern of each 3tears package
owning its own standalone MigrationRunner. the runner now composes
instead of duplicating, which is what lets the hub produce agent
schemas that include every package's tables in one coherent step.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from threetears.core.data.migrations.errors import DuplicateVersionError
from threetears.core.data.migrations.scope import MigrationScope

__all__ = [
    "MigrationFunc",
    "PackageMigrations",
]

if TYPE_CHECKING:
    from threetears.core.data.store import DataStore


MigrationFunc = Callable[["DataStore"], Awaitable[None]]


class PackageMigrations:
    """
    ordered set of versioned migrations contributed by one package.

    each registered package declares:

    - ``name`` — unique identifier used for dependency resolution and
      for the ``package`` column in ``_schema_migrations``.
    - ``scope`` — whether the migrations target the platform or agent
      schema.
    - ``depends_on`` — tuple of other package names whose migrations
      must complete before this package's migrations run.

    callers add migrations via the ``version`` decorator, which wraps
    an async callable receiving a DataStore and keys it by integer
    version. versions are unique within one package.

    :param name: unique package identifier
    :ptype name: str
    :param scope: platform-scope or agent-scope migrations
    :ptype scope: MigrationScope
    :param depends_on: other package names that must complete first
    :ptype depends_on: tuple[str, ...]
    """

    def __init__(
        self,
        name: str,
        scope: MigrationScope,
        depends_on: tuple[str, ...] = (),
    ) -> None:
        """
        initialize an empty registration bound to a package identifier.

        :param name: unique package identifier
        :ptype name: str
        :param scope: migration scope (platform or agent)
        :ptype scope: MigrationScope
        :param depends_on: other package names that must run first
        :ptype depends_on: tuple[str, ...]
        """
        self._name = name
        self._scope = scope
        self._depends_on = tuple(depends_on)
        self._versions: dict[int, MigrationFunc] = {}
        self._downgrades: dict[int, MigrationFunc] = {}

    @property
    def name(self) -> str:
        """
        return the unique package identifier.

        :return: package name as registered at construction
        :rtype: str
        """
        return self._name

    @property
    def scope(self) -> MigrationScope:
        """
        return the migration scope.

        :return: platform-scope or agent-scope enum value
        :rtype: MigrationScope
        """
        return self._scope

    @property
    def depends_on(self) -> tuple[str, ...]:
        """
        return the tuple of package names this package depends on.

        :return: immutable tuple of dependency package names
        :rtype: tuple[str, ...]
        """
        return self._depends_on

    @property
    def versions(self) -> dict[int, MigrationFunc]:
        """
        return the dict of registered migrations keyed by version.

        :return: copy of the version-to-callable mapping
        :rtype: dict[int, MigrationFunc]
        """
        return dict(self._versions)

    @property
    def downgrades(self) -> dict[int, MigrationFunc]:
        """
        return the dict of registered downgrade callables keyed by version.

        a downgrade callable is the inverse of the upgrade at the same
        version. packages declare downgrades only when they want to be
        rollbackable — packages without downgrades raise at rollback
        time so the operator knows they have to intervene manually.

        :return: copy of the version-to-downgrade-callable mapping
        :rtype: dict[int, MigrationFunc]
        """
        return dict(self._downgrades)

    def version(self, n: int) -> Callable[[MigrationFunc], MigrationFunc]:
        """
        decorator registering a migration callable at an integer version.

        :param n: unique version number within this package
        :ptype n: int
        :return: decorator that records the callable and returns it
        :rtype: Callable[[MigrationFunc], MigrationFunc]
        :raises DuplicateVersionError: if version already registered
        """

        def decorator(func: MigrationFunc) -> MigrationFunc:
            """
            register callable at version n within the enclosing package.

            :param func: async callable taking a DataStore
            :ptype func: MigrationFunc
            :return: input callable unchanged
            :rtype: MigrationFunc
            :raises DuplicateVersionError: if version already registered
            """
            if n in self._versions:
                msg = f"package {self._name!r}: migration version {n} already registered"
                raise DuplicateVersionError(msg)
            self._versions[n] = func
            return func

        return decorator

    def downgrade(self, n: int) -> Callable[[MigrationFunc], MigrationFunc]:
        """
        decorator registering a downgrade callable paired with version n.

        the downgrade callable must be the inverse of the upgrade at
        the same version: if the upgrade created a table, the downgrade
        drops it; if the upgrade added a column, the downgrade removes
        it. downgrades are optional — packages without downgrades are
        one-way, and the runner refuses to roll them back.

        :param n: version number whose upgrade this downgrade inverts
        :ptype n: int
        :return: decorator that records the callable and returns it
        :rtype: Callable[[MigrationFunc], MigrationFunc]
        :raises DuplicateVersionError: if downgrade already registered at n
        """

        def decorator(func: MigrationFunc) -> MigrationFunc:
            """
            register downgrade callable at version n.

            :param func: async callable taking a DataStore
            :ptype func: MigrationFunc
            :return: input callable unchanged
            :rtype: MigrationFunc
            :raises DuplicateVersionError: if downgrade already registered
            """
            if n in self._downgrades:
                msg = f"package {self._name!r}: downgrade for version {n} already registered"
                raise DuplicateVersionError(msg)
            self._downgrades[n] = func
            return func

        return decorator
