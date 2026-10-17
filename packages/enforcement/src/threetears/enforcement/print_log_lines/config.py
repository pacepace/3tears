"""configuration dataclass for print-log-lines enforcement.

the print-log-lines domain holds one contract: a module that runs on 3tears logs through
``threetears.observe.get_logger``. two spellings of a log line bypass it:

- ``print(..., file=sys.stderr)``: a line with no level, no fields, no correlation tags, and in a
  worker or a container no place in the log pipeline.
- stdlib ``logging`` used directly: ``logging.getLogger(...)``, ``logging.basicConfig(...)``, or a
  root-logger call (``logging.warning(...)``), which skip the ``ContextFormatter`` observe installs.

it is narrower than its siblings on purpose. ``codebase_conventions`` forbids every ``print`` and
``no_stdlib_logging`` every ``import logging``, which suits a library; an APPLICATION built on 3tears
also has a command line whose output IS printed, and helper scripts that never touch 3tears. so this
domain looks only at modules that import 3tears (:attr:`PrintLogLinesConfig.scope_import_roots`),
leaves ``print`` to stdout alone, and lets a command-line module that writes its answer to stderr be
listed in :attr:`PrintLogLinesConfig.exempt_files`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["PrintLogLinesConfig"]


@dataclass(frozen=True)
class PrintLogLinesConfig:
    """per-repo config for the print-log-lines enforcement domain.

    :ivar repo_root: absolute path to the consumer repo's root (the directory containing its
        top-level ``pyproject.toml``)
    :ivar src_roots: explicit trees to scan. ``None`` discovers them with
        :func:`threetears.enforcement.common.pyproject_discovery.discover_src_roots`
    :ivar exemptions_path: optional ``path:line:symbol`` exemptions file with a ``# rationale:``
        line per entry, as the sibling domains read; ``None`` reads none
    :ivar mode_env_var: environment variable choosing strict or report mode
    :ivar exempt_files: relative-posix-path -> rationale. a listed module is not scanned: a
        command line whose stderr lines are its answer to the person who ran it, not log lines
    :ivar line_marker: comment that excuses one call, on any line the call spans
    :ivar scope_import_roots: a module is in scope when it imports a module under one of these
        dotted roots (``threetears`` matches ``threetears.observe``, not ``threetearsx``). empty
        puts every module in scope
    :ivar minimum_scoped_modules: the non-vacuity floor: fewer in-scope modules than this fails
        the run, because a scan that reaches nothing reports what a clean repo reports. ``0``
        turns it off
    """

    repo_root: Path
    src_roots: tuple[Path, ...] | None = None
    exemptions_path: Path | None = None
    mode_env_var: str = "PRINT_LOG_LINES_ENFORCEMENT_MODE"
    exempt_files: dict[str, str] = field(default_factory=dict)
    line_marker: str = "# print-log: ok"
    scope_import_roots: tuple[str, ...] = ("threetears",)
    minimum_scoped_modules: int = 1
