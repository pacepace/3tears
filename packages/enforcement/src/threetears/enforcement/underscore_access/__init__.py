"""underscore-access enforcement domain — six shape walkers.

the ``_name`` prefix in python is a stability contract, not merely a
module-private scope marker. it declares: "this is implementation
detail; i reserve the right to change it; do not bind to it." the
walkers exposed here detect six distinct violation shapes against
that contract:

- shape A: cross-module private import
- shape B: cross-class protected access (delegated to ruff SLF001)
- shape C: modules with public names but no ``__all__``
- shape D: subclass shadows a base-class private name
- shape E: ``__all__`` lists a private name
- shape F: a private name reached through ``setattr`` / ``getattr`` /
  ``delattr`` / ``hasattr`` with the name as a string -- the spelling
  SLF001 cannot see. it scans the ``tests/`` trees as well as ``src``

per-repo configuration goes through :class:`UnderscoreAccessConfig`;
:func:`run_underscore_enforcement` is the pytest-friendly entry point
that orchestrates one or all walkers, applies exemptions, emits the
report, and fails in strict mode.

shape B's exemptions carry their own reconciliation problem, and it
lives here for the same reason the walkers do. the ledger recording
WHY each exempted private access was acceptable is not read back by
any walker -- they scan ``src`` roots and never enter a ``tests/``
tree -- so it rots in both directions unless something checks it.
:mod:`~threetears.enforcement.underscore_access.ledger` answers both
directions, and
:mod:`~threetears.enforcement.underscore_access.ruff_config` is the
single definition of which paths ARE exempted, so two consumers
cannot answer that differently.

:mod:`~threetears.enforcement.underscore_access.pragma_policy` holds
the SLF001 suppression policy itself: no inline ``noqa`` for SLF001, and
a per-file ignore only on a ``src`` module the ledger records. Its
module docstring shows the thin shell a consumer repo adds to enable it.
"""

from threetears.enforcement.underscore_access.config import (
    UnderscoreAccessConfig,
)
from threetears.enforcement.underscore_access.ledger import (
    MODULE_SCOPE,
    ledger_paths,
    ledger_scope_entries,
    scoped_accesses,
    blanket_noqa_offenders,
    carry_forward_rationales,
    enclosing_scopes,
    ledger_entries,
    missing_files,
    orphan_rationales,
    private_accesses,
    unlisted_accesses,
    unresolved_entries,
)
from threetears.enforcement.underscore_access.pragma_policy import (
    is_src_module,
    ledger_entries_outside_src,
    scanned_python_files,
    slf001_ignored_files,
    slf001_ignores_outside_src,
    slf001_ignores_without_a_ledger_entry,
    slf001_policy_findings,
    slf001_pragma_offenders,
)
from threetears.enforcement.underscore_access.ruff_config import (
    all_exempted_files,
    exempted_files,
    is_vendored,
    ruff_configs,
    slf001_globs,
)
from threetears.enforcement.underscore_access.runner import (
    run_underscore_enforcement,
)
from threetears.enforcement.underscore_access.walkers import (
    package_id,
    same_package,
    shape_a_violations,
    shape_b_violations,
    shape_c_violations,
    shape_d_violations,
    shape_e_violations,
    shape_f_violations,
)

__all__ = [
    "MODULE_SCOPE",
    "ledger_paths",
    "ledger_scope_entries",
    "scoped_accesses",
    "UnderscoreAccessConfig",
    "all_exempted_files",
    "blanket_noqa_offenders",
    "carry_forward_rationales",
    "enclosing_scopes",
    "exempted_files",
    "is_src_module",
    "is_vendored",
    "ledger_entries",
    "ledger_entries_outside_src",
    "missing_files",
    "orphan_rationales",
    "package_id",
    "private_accesses",
    "ruff_configs",
    "run_underscore_enforcement",
    "same_package",
    "scanned_python_files",
    "slf001_globs",
    "slf001_ignored_files",
    "slf001_ignores_outside_src",
    "slf001_ignores_without_a_ledger_entry",
    "slf001_policy_findings",
    "slf001_pragma_offenders",
    "unlisted_accesses",
    "unresolved_entries",
    "shape_a_violations",
    "shape_b_violations",
    "shape_c_violations",
    "shape_d_violations",
    "shape_e_violations",
    "shape_f_violations",
]
