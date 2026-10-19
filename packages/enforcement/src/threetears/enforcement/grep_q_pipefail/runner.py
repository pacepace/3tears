"""pytest-friendly entry point for the quiet-grep-under-pipefail domain. No exemptions file: an exemption would read
"this check may fail when it matched"."""

from __future__ import annotations

import sys

import pytest

from threetears.enforcement.common import MODE_REPORT, MODE_STRICT, emit_report, resolve_mode

from threetears.enforcement.grep_q_pipefail.config import GrepQPipefailConfig
from threetears.enforcement.grep_q_pipefail.walkers import find_grep_q_pipefail_violations, scanned_files

__all__ = ["run_grep_q_pipefail_enforcement"]


def run_grep_q_pipefail_enforcement(config: GrepQPipefailConfig) -> None:
    """scan, report, and fail in strict mode.

    :param config: per-repo configuration
    :ptype config: GrepQPipefailConfig
    :raises pytest.fail.Exception: in strict mode with violations
    :raises ValueError: when the scan finds no files
    """
    violations = find_grep_q_pipefail_violations(config)
    mode = resolve_mode(config.mode_env_var, default=MODE_STRICT)
    report = emit_report(violations, tuple(scanned_files(config)), [], mode, config.repo_root, domain="grep_q_pipefail")
    print(report, file=sys.stderr)
    if mode == MODE_REPORT:
        return
    if violations:
        pytest.fail(f"quiet-grep-under-pipefail enforcement found {len(violations)} violation(s):\n{report}")
