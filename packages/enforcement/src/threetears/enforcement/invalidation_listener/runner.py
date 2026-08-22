"""pytest-friendly orchestration for invalidation-listener enforcement.

A single :func:`run_invalidation_listener_enforcement` entry point lets each consumer's
thin shell invoke the walkers. The runner is the policy point: it resolves src roots,
applies the exemption file, enforces the non-vacuity floor, emits the standardised report,
and either fails or returns according to the configured mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from threetears.enforcement.common import (
    Exemption,
    MODE_REPORT,
    MODE_STRICT,
    Violation,
    apply_exemptions,
    emit_report,
    find_local_src_roots,
    parse_exemptions_with_rationale,
    resolve_mode,
)
from threetears.enforcement.invalidation_listener.config import InvalidationListenerConfig
from threetears.enforcement.invalidation_listener.walkers import (
    count_l2_live_registries,
    find_starts_without_stops,
    find_unlistened_registries,
)

__all__ = ["run_invalidation_listener_enforcement"]

_VALID_WALKERS: frozenset[str] = frozenset({"all", "unlistened", "unpaired"})


def run_invalidation_listener_enforcement(
    config: InvalidationListenerConfig,
    walker: str = "all",
) -> None:
    """run the walkers, apply exemptions, emit report, fail in strict mode.

    Accepted ``walker`` values: ``"all"`` (both, the default), ``"unlistened"`` (L2-live
    registries with no listener) and ``"unpaired"`` (a start with no stop). Any other value
    raises :class:`ValueError` so a typo cannot silently no-op a walker.

    **The non-vacuity floor runs first and is not exemptable.** A scan that reaches nothing
    reports exactly what a clean repo reports, so
    :attr:`InvalidationListenerConfig.minimum_live_registries` is checked before the
    findings are: a reader that stopped matching fails loudly instead of passing quietly.

    :param config: per-repo enforcement config
    :ptype config: InvalidationListenerConfig
    :param walker: which walker to invoke
    :ptype walker: str
    :return: nothing
    :rtype: None
    :raises ValueError: ``walker`` is not in the accepted set
    :raises pytest.fail.Exception: in strict mode with violations, or when the scan sees
        fewer L2-live registries than the configured floor
    """
    if walker not in _VALID_WALKERS:
        raise ValueError(f"walker must be one of {sorted(_VALID_WALKERS)}, got {walker!r}")

    src_roots = config.src_roots if config.src_roots is not None else find_local_src_roots(config.repo_root)

    if config.minimum_live_registries > 0:
        seen = count_l2_live_registries(src_roots, config.skip_basenames)
        if seen < config.minimum_live_registries:
            pytest.fail(
                f"invalidation-listener reader saw {seen} L2-live registries, below the configured "
                f"floor of {config.minimum_live_registries}. that is a BROKEN READER, not a clean "
                f"repo: a scan matching nothing demands nothing of anybody while reporting green. "
                f"check the src roots ({[str(r) for r in src_roots]}) and the registry constructor "
                f"name before lowering this number"
            )

    violations: list[Violation] = []
    if walker in {"all", "unlistened"}:
        violations.extend(find_unlistened_registries(src_roots, config.skip_basenames))
    if walker in {"all", "unpaired"}:
        violations.extend(find_starts_without_stops(src_roots, config.skip_basenames))

    exemptions = _load_exemptions(config.exemptions_path)
    filtered = apply_exemptions(violations, exemptions, config.repo_root)

    mode = resolve_mode(config.mode_env_var, default=MODE_STRICT)
    report = emit_report(
        filtered,
        src_roots,
        exemptions,
        mode,
        config.repo_root,
        domain=f"invalidation_listener.{walker}",
    )
    print(report, file=sys.stderr)

    if mode == MODE_REPORT:
        return
    if filtered:
        pytest.fail(f"invalidation-listener enforcement found {len(filtered)} violation(s):\n{report}")


def _load_exemptions(path: Path | None) -> list[Exemption]:
    """load exemptions from ``path``, or return ``[]`` when ``path`` is None.

    :param path: exemption file path, or ``None`` to skip loading
    :ptype path: Path | None
    :return: parsed exemption entries (empty when ``path`` is None or absent)
    :rtype: list[Exemption]
    """
    if path is None or not path.exists():
        return []
    return parse_exemptions_with_rationale(path)
