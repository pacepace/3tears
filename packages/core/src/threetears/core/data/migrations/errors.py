"""
migration runner error types.

four concrete error classes cover the distinct failure modes the runner
surfaces to callers: duplicate version registration inside one package,
unresolved or cyclic package dependencies at apply time, failure of an
individual migration body, and a bookkeeping ledger that disagrees with
the code about which migration a version is.
"""

from __future__ import annotations

__all__ = [
    "DuplicateVersionError",
    "LedgerMismatchError",
    "MigrationError",
    "MigrationFailedError",
    "MissingDependencyError",
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
