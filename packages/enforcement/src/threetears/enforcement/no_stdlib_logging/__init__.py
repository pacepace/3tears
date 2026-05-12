"""no-stdlib-logging enforcement domain — direct ``import logging`` walker.

every production module must use ``threetears.observe.get_logger``
instead of stdlib ``logging``. a stray ``import logging`` followed by
``logging.getLogger(...)`` silently drops correlation tags, call-site
info, and ``extra_data`` rendering — bypassing the
``ContextFormatter`` that ``threetears.observe`` installs.

per-repo configuration goes through :class:`NoStdlibLoggingConfig`;
:func:`run_no_stdlib_logging_enforcement` is the pytest-friendly
entry point that orchestrates the walker, applies exemptions, emits
the report, and fails in strict mode.
"""

from threetears.enforcement.no_stdlib_logging.config import (
    NoStdlibLoggingConfig,
)
from threetears.enforcement.no_stdlib_logging.runner import (
    run_no_stdlib_logging_enforcement,
)
from threetears.enforcement.no_stdlib_logging.walkers import (
    find_stdlib_logging_imports,
    is_stdlib_logging_module,
)

__all__ = [
    "NoStdlibLoggingConfig",
    "find_stdlib_logging_imports",
    "is_stdlib_logging_module",
    "run_no_stdlib_logging_enforcement",
]
