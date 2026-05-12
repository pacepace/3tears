"""underscore-access enforcement domain — five shape walkers.

the ``_name`` prefix in python is a stability contract, not merely a
module-private scope marker. it declares: "this is implementation
detail; i reserve the right to change it; do not bind to it." the
walkers exposed here detect five distinct violation shapes against
that contract:

- shape A: cross-module private import
- shape B: cross-class protected access (delegated to ruff SLF001)
- shape C: modules with public names but no ``__all__``
- shape D: subclass shadows a base-class private name
- shape E: ``__all__`` lists a private name

per-repo configuration goes through :class:`UnderscoreAccessConfig`;
:func:`run_underscore_enforcement` is the pytest-friendly entry point
that orchestrates one or all walkers, applies exemptions, emits the
report, and fails in strict mode.
"""

from threetears.enforcement.underscore_access.config import (
    UnderscoreAccessConfig,
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
)

__all__ = [
    "UnderscoreAccessConfig",
    "package_id",
    "run_underscore_enforcement",
    "same_package",
    "shape_a_violations",
    "shape_b_violations",
    "shape_c_violations",
    "shape_d_violations",
    "shape_e_violations",
]
