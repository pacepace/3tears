"""pytest-friendly orchestration for coercion-coverage enforcement.

a single :func:`run_coercion_enforcement` entry point lets each
consumer's thin shell invoke the walker. the runner is the policy
point: it resolves src roots, applies exemptions, emits the
standardised report, and either raises ``pytest.fail`` or returns
silently according to the configured mode.

this domain is single-walker today; the ``walker`` parameter exists
for symmetry with multi-walker domains and to leave the door open for
future expansion. accepted values are ``"all"`` (the only walker, the
default) — anything else raises :class:`ValueError`.
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

from threetears.enforcement.coercion_coverage.config import (
    CoerceCoverageConfig,
)
from threetears.enforcement.coercion_coverage.walkers import (
    find_run_overrides,
)

__all__ = ["run_coercion_enforcement"]


_VALID_WALKERS: frozenset[str] = frozenset({"all"})


def run_coercion_enforcement(
    config: CoerceCoverageConfig,
    walker: str = "all",
) -> None:
    """run the named walker, apply exemptions, emit report, fail if strict.

    accepted ``walker`` values: ``"all"`` (the only walker today).
    raises :class:`ValueError` for any other value so a typo doesn't
    silently no-op a future multi-walker expansion.

    src roots come from :attr:`CoerceCoverageConfig.src_roots` when
    set, else from :func:`discover_src_roots
    <threetears.enforcement.common.pyproject_discovery.discover_src_roots>`
    (which walks path-deps so a Tool base in a sibling package is
    visible to the AST scanner).

    exemptions are parsed when :attr:`CoerceCoverageConfig.exemptions_path`
    is set and exists; a missing file raises :class:`FileNotFoundError`
    via the parser. when ``exemptions_path`` is ``None`` the runner
    skips exemption loading entirely.

    in :data:`~threetears.enforcement.common.modes.MODE_REPORT` the
    runner returns normally regardless of violations; in
    :data:`~threetears.enforcement.common.modes.MODE_STRICT` (the
    default) it calls :func:`pytest.fail` with the rendered report.

    :param config: per-repo enforcement config
    :ptype config: CoerceCoverageConfig
    :param walker: which walker to invoke (``"all"``)
    :ptype walker: str
    :raises ValueError: ``walker`` is not in the accepted set
    :raises pytest.fail.Exception: in strict mode with violations
    """
    if walker not in _VALID_WALKERS:
        raise ValueError(f"walker must be one of {sorted(_VALID_WALKERS)}, got {walker!r}")

    src_roots = _resolve_src_roots(config)
    violations = find_run_overrides(
        src_roots,
        config.repo_root,
        config.base_class_suffixes,
    )

    exemptions = _load_exemptions(config.exemptions_path)
    filtered = apply_exemptions(violations, exemptions, config.repo_root)

    mode = resolve_mode(config.mode_env_var, default=MODE_STRICT)

    report = emit_report(
        filtered,
        src_roots,
        exemptions,
        mode,
        config.repo_root,
        domain=f"coercion_coverage.{walker}",
    )
    print(report, file=sys.stderr)

    if mode == MODE_REPORT:
        return
    if filtered:
        pytest.fail(f"coercion-coverage enforcement found {len(filtered)} violation(s):\n{report}")


def _resolve_src_roots(config: CoerceCoverageConfig) -> tuple[Path, ...]:
    """pick the src roots to scan.

    explicit ``config.src_roots`` wins; otherwise we walk the path-dep
    graph so the inheritance walker sees siblings declared as
    workspace members or path-deps. this is the option-B behaviour
    described in the task plan.

    :param config: per-repo enforcement config
    :ptype config: CoerceCoverageConfig
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
