"""
migration runner error types.

three concrete error classes cover the distinct failure modes the runner
surfaces to callers: duplicate version registration inside one package,
unresolved or cyclic package dependencies at apply time, and failure of
an individual migration body.
"""

from __future__ import annotations


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


class MigrationFailedError(MigrationError):
    """
    raised when an individual migration body raises during apply.

    the runner wraps the original exception so callers see a uniform
    surface while retaining the underlying cause via ``__cause__``. the
    runner halts the apply sequence on the first failure and reverts
    bookkeeping for the failing migration only — previously-applied
    migrations keep their recorded version.
    """
