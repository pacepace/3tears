"""pytest-friendly orchestration for dict-state-detection enforcement.

a single :func:`run_dict_state_enforcement` entry point lets each
consumer's thin shell invoke either of the two walkers (detection /
allowlist-integrity) or both. the runner is the policy point: it
resolves src roots, applies the in-code allowlist + known-violations
filtering, runs the stale-entry meta-walker, optionally applies a
file-based exemption list, emits the standardised report, and either
raises :func:`pytest.fail` or returns silently according to the
configured mode.

accepted ``walker`` values:

- ``"detect"``: only the dict-state detector runs; stale-entry check
  is skipped.
- ``"allowlist_integrity"``: only the stale-entry meta-walker runs;
  detection violations are not surfaced (used by repos that want a
  separate hygiene test, but in practice the same single entry point
  with ``"all"`` is enough).
- ``"all"`` (default): both walkers run; their violations share the
  same report.
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
    discover_src_roots,
    emit_report,
    parse_exemptions_with_rationale,
    resolve_mode,
)

from threetears.enforcement.dict_state_detection.config import (
    DictStateConfig,
)
from threetears.enforcement.dict_state_detection.walkers import (
    filter_against_allowlist,
    find_dict_state_violations,
    find_stale_allowlist_entries,
)

__all__ = ["run_dict_state_enforcement"]


_VALID_WALKERS: frozenset[str] = frozenset(
    {
        "detect",
        "allowlist_integrity",
        "all",
    }
)


def run_dict_state_enforcement(
    config: DictStateConfig,
    walker: str = "all",
) -> None:
    """run the configured walker(s), apply allowlist/exemptions, fail if strict.

    :param config: per-repo enforcement config
    :ptype config: DictStateConfig
    :param walker: which walker(s) to invoke; one of ``"detect"``,
        ``"allowlist_integrity"``, ``"all"``
    :ptype walker: str
    :raises ValueError: ``walker`` is not in the accepted set
    :raises pytest.fail.Exception: in strict mode with violations
    """
    if walker not in _VALID_WALKERS:
        raise ValueError(f"walker must be one of {sorted(_VALID_WALKERS)}, got {walker!r}")

    src_roots = _resolve_src_roots(config)
    aggregated: list[Violation] = []

    if walker in {"detect", "all"}:
        detection = find_dict_state_violations(src_roots, config.repo_root)
        true_violations, _allowed = filter_against_allowlist(
            detection,
            config.allowlist,
            config.known_violations,
            config.repo_root,
        )
        aggregated.extend(true_violations)

    if walker in {"allowlist_integrity", "all"}:
        aggregated.extend(
            find_stale_allowlist_entries(
                src_roots,
                config.allowlist,
                config.known_violations,
                config.repo_root,
            )
        )

    exemptions = _load_exemptions(config.exemptions_path)
    filtered = apply_exemptions(aggregated, exemptions, config.repo_root)

    mode = resolve_mode(config.mode_env_var, default=MODE_STRICT)

    report = emit_report(
        filtered,
        src_roots,
        exemptions,
        mode,
        config.repo_root,
        domain=f"dict_state_detection.{walker}",
    )
    print(report, file=sys.stderr)

    if mode == MODE_REPORT:
        return
    if filtered:
        pytest.fail(f"dict-state-detection enforcement found {len(filtered)} violation(s):\n{report}")


def _resolve_src_roots(config: DictStateConfig) -> tuple[Path, ...]:
    """pick the src roots to scan.

    explicit ``config.src_roots`` wins; otherwise we walk the path-dep
    graph so the walker sees siblings declared as workspace members
    or path-deps. this is the option-B behaviour described in the
    task plan.

    :param config: per-repo enforcement config
    :ptype config: DictStateConfig
    :return: src roots to scan
    :rtype: tuple[Path, ...]
    """
    if config.src_roots is not None:
        return config.src_roots
    return discover_src_roots(config.repo_root)


def _load_exemptions(path: Path | None) -> list[Exemption]:
    """load exemptions from ``path``, or return ``[]`` when ``path`` is None.

    :param path: exemption file path, or ``None`` to skip loading
    :ptype path: Path | None
    :return: parsed exemption entries (empty when ``path`` is None)
    :rtype: list[Exemption]
    :raises FileNotFoundError: ``path`` is set but does not exist
    """
    if path is None:
        return []
    return parse_exemptions_with_rationale(path)
