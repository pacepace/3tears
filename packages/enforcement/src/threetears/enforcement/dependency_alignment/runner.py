"""pytest-friendly orchestration for dependency-alignment enforcement."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from threetears.enforcement.common import (
    MODE_REPORT,
    MODE_STRICT,
    Exemption,
    apply_exemptions,
    emit_report,
    parse_exemptions_with_rationale,
    resolve_mode,
)
from threetears.enforcement.dependency_alignment.config import DependencyAlignmentConfig
from threetears.enforcement.dependency_alignment.walkers import (
    contract_purity_violations,
    dependency_alignment_violations,
)

__all__ = ["run_dependency_alignment_enforcement"]

_WALKERS = {
    "alignment": dependency_alignment_violations,
    "contract_purity": contract_purity_violations,
}


def run_dependency_alignment_enforcement(
    config: DependencyAlignmentConfig,
    walker: str = "alignment",
) -> None:
    """run the named walker, apply exemptions, emit report, fail if strict.

    :param config: per-repo enforcement config
    :ptype config: DependencyAlignmentConfig
    :param walker: which walker to run (``alignment`` / ``contract_purity``)
    :ptype walker: str
    :raises pytest.fail.Exception: in strict mode with violations
    """
    violations = _WALKERS[walker](config)

    exemptions = _load_exemptions(config.exemptions_path)
    filtered = apply_exemptions(violations, exemptions, config.repo_root)

    mode = resolve_mode(config.mode_env_var, default=MODE_STRICT)

    scan_roots = tuple(sorted({config.repo_root / g.split("/")[0] for g in config.package_globs}))
    report = emit_report(
        filtered,
        scan_roots,
        exemptions,
        mode,
        config.repo_root,
        domain="dependency_alignment",
    )
    print(report, file=sys.stderr)

    if mode == MODE_REPORT:
        return
    if filtered:
        pytest.fail(
            f"dependency-alignment enforcement found {len(filtered)} violation(s):\n{report}",
        )


def _load_exemptions(path: Path | None) -> list[Exemption]:
    """load exemptions from ``path``, or return ``[]`` when absent.

    :param path: exemption file path, or ``None`` to skip loading
    :ptype path: Path | None
    :return: parsed exemption entries
    :rtype: list[Exemption]
    """
    if path is None or not path.exists():
        return []
    return parse_exemptions_with_rationale(path)
