"""pytest-friendly orchestration for audit-details classification enforcement.

A single :func:`run_audit_details_enforcement` entry point lets each consumer's thin shell
invoke the walker over its own source roots. The runner is the policy point: it runs the
walker, emits the standardised report, and either fails or returns according to the
configured mode.

This domain takes no exemptions file. An exemption would read "this key is published
without anyone deciding whether it is personal", which is the defect itself; the remedy
for a key is to classify it, and for a wrapper helper to declare it a forwarder.
"""

from __future__ import annotations

import sys

import pytest

from threetears.enforcement.common import MODE_REPORT, MODE_STRICT, emit_report, resolve_mode

from threetears.enforcement.audit_details.config import AuditDetailsConfig
from threetears.enforcement.audit_details.walkers import find_audit_details_violations

__all__ = ["run_audit_details_enforcement"]

_VALID_WALKERS: frozenset[str] = frozenset({"all"})


def run_audit_details_enforcement(config: AuditDetailsConfig, walker: str = "all") -> None:
    """run the walker, emit the report, fail in strict mode.

    :param config: per-repo enforcement config
    :ptype config: AuditDetailsConfig
    :param walker: which walker to invoke (``"all"``)
    :ptype walker: str
    :raises ValueError: ``walker`` is not in the accepted set
    :raises pytest.fail.Exception: in strict mode with violations
    """
    if walker not in _VALID_WALKERS:
        raise ValueError(f"walker must be one of {sorted(_VALID_WALKERS)}, got {walker!r}")

    violations = find_audit_details_violations(config)
    mode = resolve_mode(config.mode_env_var, default=MODE_STRICT)
    report = emit_report(violations, config.src_roots, [], mode, config.repo_root, domain=f"audit_details.{walker}")
    print(report, file=sys.stderr)

    if mode == MODE_REPORT:
        return
    if violations:
        pytest.fail(f"audit-details enforcement found {len(violations)} violation(s):\n{report}")
