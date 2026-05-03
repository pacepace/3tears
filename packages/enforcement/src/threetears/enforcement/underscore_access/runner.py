"""pytest-friendly orchestration for underscore-access enforcement.

a single :func:`run_underscore_enforcement` entry point lets each
consumer's thin shell invoke any one of the five shapes (or all of
them as a single combined pass). the runner is the policy point: it
resolves both root sets (``scan_roots`` for *where to look for
violations* and ``inheritance_roots`` for *where to build the
inheritance graph* used by shape A's ``same_package`` and shape D's
cross-package shadow detection), applies exemptions, emits the
standardised report, and either raises ``pytest.fail`` or returns
silently according to the configured mode.

the function does not return a structured result object on purpose —
its callers are pytest tests that only care about pass/fail. the
report goes to stderr so ``pytest -s`` surfaces the inventory
regardless of mode.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
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
    find_local_src_roots,
    parse_exemptions_with_rationale,
    resolve_mode,
)

from threetears.enforcement.underscore_access.config import (
    UnderscoreAccessConfig,
)
from threetears.enforcement.underscore_access.walkers import (
    shape_a_violations,
    shape_b_violations,
    shape_c_violations,
    shape_d_violations,
    shape_e_violations,
)

__all__ = ["run_underscore_enforcement"]


_VALID_WALKERS: frozenset[str] = frozenset(
    {
        "shape_a",
        "shape_b",
        "shape_c",
        "shape_d",
        "shape_e",
        "all",
    }
)


def run_underscore_enforcement(
    config: UnderscoreAccessConfig,
    walker: str,
) -> None:
    """run the named walker(s), apply exemptions, emit report, fail if strict.

    accepted ``walker`` values: ``"shape_a"`` through ``"shape_e"``
    or ``"all"`` to run every shape with a single combined report.
    raises :class:`ValueError` for any other value.

    src-root responsibilities are split:

    - ``scan_roots`` — where to LOOK for violations. comes from
      :attr:`UnderscoreAccessConfig.scan_roots` when set, else from
      :func:`find_local_src_roots
      <threetears.enforcement.common.repo_layout.find_local_src_roots>`
      so the walkers stay scoped to the consumer's own code.
    - ``inheritance_roots`` — where to BUILD the inheritance graph
      and resolve module-to-file lookups (shape A) and cross-package
      private-name shadow detection (shape D). comes from
      :attr:`UnderscoreAccessConfig.inheritance_roots` when set, else
      from :func:`discover_src_roots
      <threetears.enforcement.common.pyproject_discovery.discover_src_roots>`
      (which walks path-deps).

    exemptions are parsed when :attr:`UnderscoreAccessConfig.exemptions_path`
    is set and exists. a missing file raises :class:`FileNotFoundError`
    via the parser only if ``exemptions_path`` itself is configured —
    when it's ``None`` the runner skips exemption loading entirely.

    in :data:`~threetears.enforcement.common.modes.MODE_REPORT` the
    runner returns normally regardless of violations; in
    :data:`~threetears.enforcement.common.modes.MODE_STRICT` (the
    default) it calls :func:`pytest.fail` with the rendered report.

    :param config: per-repo enforcement config
    :ptype config: UnderscoreAccessConfig
    :param walker: which walker to invoke (``shape_a``..``shape_e``
        or ``all``)
    :ptype walker: str
    :raises ValueError: ``walker`` is not in the accepted set
    :raises pytest.fail.Exception: in strict mode with violations
    """
    if walker not in _VALID_WALKERS:
        raise ValueError(f"walker must be one of {sorted(_VALID_WALKERS)}, got {walker!r}")

    scan_roots = _resolve_scan_roots(config)
    inheritance_roots = _resolve_inheritance_roots(config)
    violations = _run_walker(walker, config, scan_roots, inheritance_roots)

    exemptions = _load_exemptions(config.exemptions_path)
    filtered = apply_exemptions(violations, exemptions, config.repo_root)

    mode = resolve_mode(config.mode_env_var, default=MODE_STRICT)

    report = emit_report(
        filtered,
        scan_roots,
        exemptions,
        mode,
        config.repo_root,
        domain=f"underscore_access.{walker}",
    )
    inheritance_line = f"inheritance_roots: {[str(r) for r in inheritance_roots]}"
    print(report, file=sys.stderr)
    print(inheritance_line, file=sys.stderr)

    if mode == MODE_REPORT:
        return
    if filtered:
        pytest.fail(f"underscore-access enforcement found {len(filtered)} violation(s):\n{report}\n{inheritance_line}")


def _resolve_scan_roots(config: UnderscoreAccessConfig) -> tuple[Path, ...]:
    """pick the roots to scan for violations.

    explicit :attr:`UnderscoreAccessConfig.scan_roots` wins; otherwise
    we use :func:`find_local_src_roots
    <threetears.enforcement.common.repo_layout.find_local_src_roots>`
    so violations stay scoped to the consumer repo's own code (never
    walks path-deps).

    :param config: per-repo enforcement config
    :ptype config: UnderscoreAccessConfig
    :return: src roots to scan for violations
    :rtype: tuple[Path, ...]
    """
    if config.scan_roots is not None:
        return config.scan_roots
    return find_local_src_roots(config.repo_root)


def _resolve_inheritance_roots(
    config: UnderscoreAccessConfig,
) -> tuple[Path, ...]:
    """pick the roots from which to build the inheritance graph.

    explicit :attr:`UnderscoreAccessConfig.inheritance_roots` wins;
    otherwise we use :func:`discover_src_roots
    <threetears.enforcement.common.pyproject_discovery.discover_src_roots>`
    so module-to-file resolution and cross-package shadow detection
    work across siblings declared as workspace members or path-deps.

    :param config: per-repo enforcement config
    :ptype config: UnderscoreAccessConfig
    :return: src roots from which to build the inheritance graph
    :rtype: tuple[Path, ...]
    """
    if config.inheritance_roots is not None:
        return config.inheritance_roots
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


def _run_walker(
    walker: str,
    config: UnderscoreAccessConfig,
    scan_roots: tuple[Path, ...],
    inheritance_roots: tuple[Path, ...],
) -> list[Violation]:
    """dispatch to the named walker(s) and return raw violations.

    :param walker: walker identifier
    :ptype walker: str
    :param config: per-repo enforcement config
    :ptype config: UnderscoreAccessConfig
    :param scan_roots: resolved roots to scan for violations
    :ptype scan_roots: tuple[Path, ...]
    :param inheritance_roots: resolved roots from which to build the
        inheritance graph
    :ptype inheritance_roots: tuple[Path, ...]
    :return: raw (un-filtered) violations
    :rtype: list[Violation]
    """
    runners: dict[str, Callable[[], list[Violation]]] = {
        "shape_a": lambda: shape_a_violations(
            scan_roots,
            config.repo_root,
            inheritance_roots,
        ),
        "shape_b": lambda: shape_b_violations(config.repo_root, scan_roots) if config.enable_shape_b_ruff else [],
        "shape_c": lambda: shape_c_violations(
            scan_roots,
            config.repo_root,
            config.skip_basenames,
        ),
        "shape_d": lambda: shape_d_violations(
            scan_roots,
            config.repo_root,
            inheritance_roots,
        ),
        "shape_e": lambda: shape_e_violations(scan_roots, config.repo_root),
    }
    if walker == "all":
        out: list[Violation] = []
        for name in ("shape_a", "shape_b", "shape_c", "shape_d", "shape_e"):
            out.extend(runners[name]())
        return out
    return runners[walker]()
