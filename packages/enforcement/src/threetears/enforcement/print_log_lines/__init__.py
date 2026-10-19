"""print-log-lines enforcement domain -- log lines that bypass ``threetears.observe``.

in every module that imports 3tears: no ``print(..., file=sys.stderr)`` and no direct stdlib
``logging`` call (``getLogger``, ``basicConfig``, a root-logger call). log through
``threetears.observe.get_logger``. ``print`` to stdout, modules that do not import 3tears, and
command-line modules listed in ``exempt_files`` are left alone, which suits an application whose
command line prints its answers.

configure with :class:`PrintLogLinesConfig`; :func:`run_print_log_lines_enforcement` is the
pytest-friendly entry point.
"""

from threetears.enforcement.print_log_lines.config import PrintLogLinesConfig
from threetears.enforcement.print_log_lines.runner import run_print_log_lines_enforcement
from threetears.enforcement.print_log_lines.walkers import (
    find_print_log_lines,
    find_scoped_modules,
    imports_any_root,
)

__all__ = [
    "PrintLogLinesConfig",
    "find_print_log_lines",
    "find_scoped_modules",
    "imports_any_root",
    "run_print_log_lines_enforcement",
]
