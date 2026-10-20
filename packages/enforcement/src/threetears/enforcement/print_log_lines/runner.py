"""pytest-friendly orchestration for print-log-lines enforcement.

:func:`run_print_log_lines_enforcement` is the entry point a consumer's thin shell calls. it
resolves the src roots, checks the non-vacuity floor, runs the walker, applies the exemptions file,
emits the standard report and fails in strict mode.

single-walker; ``walker`` exists for symmetry with multi-walker domains and accepts only ``"all"``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from threetears.enforcement.common import (
    Exemption,
    MODE_REPORT,
    MODE_STRICT,
    apply_exemptions,
    discover_src_roots,
    emit_report,
    parse_exemptions_with_rationale,
    resolve_mode,
)

from threetears.enforcement.print_log_lines.config import PrintLogLinesConfig
from threetears.enforcement.print_log_lines.walkers import find_print_log_lines, find_scoped_modules

__all__ = ["run_print_log_lines_enforcement"]


_VALID_WALKERS: frozenset[str] = frozenset({"all"})


def run_print_log_lines_enforcement(config: PrintLogLinesConfig, walker: str = "all") -> None:
    """run the walker, apply exemptions, emit the report, fail if strict.

    **The non-vacuity floor runs first, in every mode.** A scan that reaches no in-scope module
    reports exactly what a clean repo reports, so fewer in-scope modules than
    :attr:`PrintLogLinesConfig.minimum_scoped_modules` fails the run: wrong src roots or a wrong
    scope root, not a clean tree.

    :param config: per-repo enforcement config
    :ptype config: PrintLogLinesConfig
    :param walker: which walker to invoke (``"all"``)
    :ptype walker: str
    :return: None
    :rtype: None
    :raises ValueError: ``walker`` is not ``"all"``
    :raises pytest.fail.Exception: below the floor, or in strict mode with violations
    """
    if walker not in _VALID_WALKERS:
        raise ValueError(f"walker must be one of {sorted(_VALID_WALKERS)}, got {walker!r}")

    src_roots = config.src_roots if config.src_roots is not None else discover_src_roots(config.repo_root)

    if config.minimum_scoped_modules > 0:
        scoped = find_scoped_modules(src_roots, config.repo_root, config.exempt_files, config.scope_import_roots)
        if len(scoped) < config.minimum_scoped_modules:
            pytest.fail(
                f"print-log-lines scan found {len(scoped)} module(s) importing {list(config.scope_import_roots)}, "
                f"below the configured floor of {config.minimum_scoped_modules}. that is a BROKEN SCAN, not a "
                f"clean repo: check the src roots ({[str(r) for r in src_roots]}) and scope_import_roots"
            )

    violations = find_print_log_lines(
        src_roots,
        config.repo_root,
        config.exempt_files,
        config.line_marker,
        config.scope_import_roots,
    )
    exemptions = _load_exemptions(config.exemptions_path)
    filtered = apply_exemptions(violations, exemptions, config.repo_root)

    mode = resolve_mode(config.mode_env_var, default=MODE_STRICT)
    report = emit_report(filtered, src_roots, exemptions, mode, config.repo_root, domain=f"print_log_lines.{walker}")
    print(report, file=sys.stderr)

    if mode == MODE_REPORT:
        return
    if filtered:
        pytest.fail(f"print-log-lines enforcement found {len(filtered)} violation(s):\n{report}")


def _load_exemptions(path: Path | None) -> list[Exemption]:
    """load exemptions from ``path``, or none when it is ``None``.

    :param path: exemptions file, or ``None``
    :ptype path: Path | None
    :return: parsed entries
    :rtype: list[Exemption]
    :raises FileNotFoundError: ``path`` is set but does not exist
    """
    if path is None:
        return []
    return parse_exemptions_with_rationale(path)
