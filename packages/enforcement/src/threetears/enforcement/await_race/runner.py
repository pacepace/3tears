"""pytest-friendly orchestration for await-race enforcement: walk, report, fail in strict mode. No exemptions."""

from __future__ import annotations

import sys

import pytest

from threetears.enforcement.await_race.config import AwaitRaceConfig
from threetears.enforcement.await_race.walkers import find_await_race_violations
from threetears.enforcement.common import MODE_REPORT, MODE_STRICT, discover_src_roots, emit_report, resolve_mode

__all__ = ["run_await_race_enforcement"]


def run_await_race_enforcement(config: AwaitRaceConfig) -> None:
    """walk the repo's serving code for stale write-backs across an ``await``; fail on any in strict mode.

    :param config: per-repo enforcement config
    :ptype config: AwaitRaceConfig
    :raises pytest.fail.Exception: in strict mode with violations
    """
    src_roots = config.src_roots if config.src_roots is not None else discover_src_roots(config.repo_root)
    violations = find_await_race_violations(src_roots, non_serving=config.non_serving_roots)
    mode = resolve_mode(config.mode_env_var, default=MODE_STRICT)
    report = emit_report(violations, src_roots, [], mode, config.repo_root, domain="await_race")
    print(report, file=sys.stderr)
    if mode != MODE_REPORT and violations:
        pytest.fail(f"await-race enforcement found {len(violations)} violation(s):\n{report}")
