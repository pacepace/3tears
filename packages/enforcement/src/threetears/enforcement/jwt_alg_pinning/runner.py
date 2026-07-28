"""pytest-friendly orchestration for JWT algorithm-pinning enforcement.

A single :func:`run_jwt_alg_pinning_enforcement` entry point lets each consumer's thin shell
invoke the walkers. The runner is the policy point: it runs the checks, emits the
standardised report, and either fails or returns according to the configured mode.

This domain takes no exemptions file. An exemption here would read "this JWS verifier is
allowed to choose its algorithm at runtime", which is the vulnerability itself.
"""

from __future__ import annotations

import sys

import pytest

from threetears.enforcement.common import MODE_REPORT, MODE_STRICT, emit_report, resolve_mode

from threetears.enforcement.jwt_alg_pinning.config import JwtAlgPinningConfig
from threetears.enforcement.jwt_alg_pinning.walkers import find_alg_pinning_violations

__all__ = ["run_jwt_alg_pinning_enforcement"]

_VALID_WALKERS: frozenset[str] = frozenset({"all"})


def run_jwt_alg_pinning_enforcement(config: JwtAlgPinningConfig, walker: str = "all") -> None:
    """run the walkers, emit the report, fail in strict mode.

    :param config: per-repo enforcement config
    :ptype config: JwtAlgPinningConfig
    :param walker: which walker to invoke (``"all"``)
    :ptype walker: str
    :raises ValueError: ``walker`` is not in the accepted set
    :raises pytest.fail.Exception: in strict mode with violations
    """
    if walker not in _VALID_WALKERS:
        raise ValueError(f"walker must be one of {sorted(_VALID_WALKERS)}, got {walker!r}")

    violations = find_alg_pinning_violations(config)
    mode = resolve_mode(config.mode_env_var, default=MODE_STRICT)
    report = emit_report(
        violations,
        tuple(module.path for module in config.modules),
        [],
        mode,
        config.repo_root,
        domain=f"jwt_alg_pinning.{walker}",
    )
    print(report, file=sys.stderr)

    if mode == MODE_REPORT:
        return
    if violations:
        pytest.fail(f"JWT alg-pinning enforcement found {len(violations)} violation(s):\n{report}")
