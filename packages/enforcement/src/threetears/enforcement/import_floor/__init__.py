"""import-floor enforcement domain.

**A package's base install is a promise about weight.** A consumer outside the platform (testament's
coordinator takes ``threetears.iam``'s sign-in modules and runs neither NATS nor Postgres) installs
the base and expects importing a base module to cost what the base declares and no more.

The gate is an **allowlist, not a denylist**: each base module is imported in a fresh interpreter
with the extras' roots refused, and everything it loaded must be the standard library, a declared
third-party root, or a declared family package. A denylist only catches the weight somebody
predicted; the next dependency to arrive will be one nobody wrote down.

The extras' roots are refused rather than left installed because an optional extra's module
imports at its top, under a guard naming the extra (lazy imports are banned in serving code, owner
ruling 2026-10-24), so a development environment with every extra installed would measure the
extras too. Refusing them measures the install the promise is about; a base module that imports
one without the guard fails the probe outright.

Per-package configuration goes through :class:`ImportFloorConfig`;
:func:`run_import_floor_enforcement` is the pytest-friendly entry point.
"""

from threetears.enforcement.import_floor.config import ImportFloorConfig
from threetears.enforcement.import_floor.runner import run_import_floor_enforcement
from threetears.enforcement.import_floor.walkers import ProbeError, modules_loaded_by_importing, outside_floor

__all__ = [
    "ImportFloorConfig",
    "ProbeError",
    "modules_loaded_by_importing",
    "outside_floor",
    "run_import_floor_enforcement",
]
