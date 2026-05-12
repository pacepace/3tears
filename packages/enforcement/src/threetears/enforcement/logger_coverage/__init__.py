"""logger-coverage enforcement domain — module-level logger contract.

every production module must declare a module-level
``log = get_logger(__name__)`` (or the legacy ``_logger`` alias). a
silent module is the most expensive class of operability defect: there
is nothing to grep, nothing to correlate, nothing to alert on. losing
the logger is never an accident — either the module exists for a
reason that genuinely produces no observable behaviour (a re-export
shim, a pure pydantic model, a constants table) and is therefore
exempt, or it must declare a logger.

per-repo configuration goes through :class:`LoggerCoverageConfig`;
:func:`run_logger_coverage_enforcement` is the pytest-friendly entry
point that orchestrates the walker, applies exemptions, emits the
report, and fails in strict mode.
"""

from threetears.enforcement.logger_coverage.config import (
    LoggerCoverageConfig,
)
from threetears.enforcement.logger_coverage.runner import (
    run_logger_coverage_enforcement,
)
from threetears.enforcement.logger_coverage.walkers import (
    find_modules_without_logger,
)

__all__ = [
    "LoggerCoverageConfig",
    "find_modules_without_logger",
    "run_logger_coverage_enforcement",
]
