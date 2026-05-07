"""pytest-friendly orchestration for fake-protocol-parity enforcement.

a single :func:`run_fake_parity_enforcement` entry point lets each
consumer's thin shell run the walker once. the runner resolves the
scan roots (defaulting to every ``tests`` directory under repo root
or under ``packages/**``), calls the walker, applies exemptions,
emits the standardised report, and either raises ``pytest.fail`` or
returns silently per the configured mode.
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
    emit_report,
    parse_exemptions_with_rationale,
    resolve_mode,
)
from threetears.enforcement.fake_parity.config import FakeParityConfig
from threetears.enforcement.fake_parity.walkers import fake_parity_violations

__all__ = ["run_fake_parity_enforcement"]


def run_fake_parity_enforcement(config: FakeParityConfig) -> None:
    """run the walker, apply exemptions, emit report, fail if strict.

    :param config: per-repo enforcement config
    :ptype config: FakeParityConfig
    :raises pytest.fail.Exception: in strict mode with violations
    """
    scan_roots = _resolve_scan_roots(config)
    violations = fake_parity_violations(scan_roots, config.repo_root)

    exemptions = _load_exemptions(config.exemptions_path)
    filtered = apply_exemptions(violations, exemptions, config.repo_root)

    mode = resolve_mode(config.mode_env_var, default=MODE_STRICT)

    report = emit_report(
        filtered,
        scan_roots,
        exemptions,
        mode,
        config.repo_root,
        domain="fake_parity",
    )
    print(report, file=sys.stderr)

    if mode == MODE_REPORT:
        return
    if filtered:
        pytest.fail(
            f"fake-protocol-parity enforcement found {len(filtered)} "
            f"violation(s):\n{report}",
        )


def _resolve_scan_roots(config: FakeParityConfig) -> tuple[Path, ...]:
    """pick the directories to recurse into for fake declarations.

    explicit :attr:`FakeParityConfig.scan_roots` wins; otherwise we
    auto-discover every ``tests`` directory directly under
    ``repo_root`` and under any ``packages/*/`` subtree (covers both
    single-package layouts and the 3tears workspace shape).

    :param config: per-repo enforcement config
    :ptype config: FakeParityConfig
    :return: roots to scan
    :rtype: tuple[Path, ...]
    """
    if config.scan_roots is not None:
        return config.scan_roots
    discovered: list[Path] = []
    repo_tests = config.repo_root / "tests"
    if repo_tests.is_dir():
        discovered.append(repo_tests)
    packages_dir = config.repo_root / "packages"
    if packages_dir.is_dir():
        for child in sorted(packages_dir.iterdir()):
            if not child.is_dir():
                continue
            tests_dir = child / "tests"
            if tests_dir.is_dir():
                discovered.append(tests_dir)
            for grandchild in sorted(child.iterdir()):
                if not grandchild.is_dir():
                    continue
                nested_tests = grandchild / "tests"
                if nested_tests.is_dir():
                    discovered.append(nested_tests)
    return tuple(discovered)


def _load_exemptions(path: Path | None) -> list[Exemption]:
    """load exemptions from ``path``, or return ``[]`` when path is None.

    :param path: exemption file path, or ``None`` to skip loading
    :ptype path: Path | None
    :return: parsed exemption entries (empty when ``path`` is None)
    :rtype: list[Exemption]
    :raises FileNotFoundError: ``path`` is set but does not exist
    """
    if path is None:
        return []
    if not path.exists():
        return []
    return parse_exemptions_with_rationale(path)
