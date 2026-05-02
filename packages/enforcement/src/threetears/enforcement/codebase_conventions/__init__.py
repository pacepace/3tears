"""codebase-conventions enforcement domain — four AST walkers.

every production source tree must obey four universal AST-level
conventions:

- no bare ``print(...)`` calls (use the project logger).
- no stdlib ``logging.getLogger(...)`` calls (use
  ``threetears.observe.get_logger``); per-line marker discipline and
  file-level allowlists provide narrow escape hatches.
- ``from __future__ import annotations`` at top of every module.
- every non-dunder, non-test function definition declares a return
  type.

per-repo configuration goes through :class:`CodebaseConventionsConfig`;
:func:`run_codebase_conventions_enforcement` is the pytest-friendly
entry point that orchestrates the walkers, applies exemptions, emits
the report, and fails in strict mode.
"""

from threetears.enforcement.codebase_conventions.config import (
    CodebaseConventionsConfig,
)
from threetears.enforcement.codebase_conventions.runner import (
    run_codebase_conventions_enforcement,
)
from threetears.enforcement.codebase_conventions.walkers import (
    find_missing_future_annotations,
    find_missing_return_types,
    find_print_calls,
    find_stdlib_getlogger_calls,
)

__all__ = [
    "CodebaseConventionsConfig",
    "find_missing_future_annotations",
    "find_missing_return_types",
    "find_print_calls",
    "find_stdlib_getlogger_calls",
    "run_codebase_conventions_enforcement",
]
