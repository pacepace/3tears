"""pytest-friendly orchestration for cache-primitive-usage enforcement.

a single :func:`run_cache_enforcement` entry point lets each consumer's
thin shell invoke any one of the four walkers or all of them. the
runner is the policy point: it resolves both root sets (``scan_roots``
for *where to look for violations* and ``inheritance_roots`` for *where
to build the cross-package class-base graph*), runs the selected
walker(s), applies the rationale-required exemption file, emits the
standardised report, and either raises :func:`pytest.fail` or returns
silently according to the configured mode.

accepted ``walker`` values:

- ``"sqlite_construction"``: only :func:`find_sqlite_constructions`.
- ``"wrapper_class"``: only :func:`find_wrapper_classes`.
- ``"pool_access"``: only :func:`find_direct_pool_access`.
- ``"missing_collection"``: only :func:`find_missing_collections`.
- ``"all"`` (default): run every walker; aggregated violations share
  the same report.
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
    find_local_src_roots,
    parse_exemptions_with_rationale,
    resolve_mode,
)

from threetears.enforcement.cache.config import (
    CacheEnforcementConfig,
)
from threetears.enforcement.cache.walkers import (
    find_direct_pool_access,
    find_missing_collections,
    find_sqlite_constructions,
    find_wrapper_classes,
)

__all__ = ["run_cache_enforcement"]


_VALID_WALKERS: frozenset[str] = frozenset(
    {
        "sqlite_construction",
        "wrapper_class",
        "pool_access",
        "missing_collection",
        "all",
    }
)


def run_cache_enforcement(
    config: CacheEnforcementConfig,
    walker: str = "all",
) -> None:
    """run the configured walker(s), apply exemptions, fail if strict.

    src-root responsibilities are split:

    - ``scan_roots`` — where to LOOK for violations. comes from
      :attr:`CacheEnforcementConfig.scan_roots` when set, else from
      :func:`find_local_src_roots
      <threetears.enforcement.common.repo_layout.find_local_src_roots>`
      so the walkers stay scoped to the consumer's own code (path-deps
      are NOT scanned for violations).
    - ``inheritance_roots`` — where to BUILD the cross-package
      class-base graph used by transitive subclass detection in
      :func:`find_wrapper_classes` and :func:`find_missing_collections`.
      comes from :attr:`CacheEnforcementConfig.inheritance_roots` when
      set, else from :func:`discover_src_roots
      <threetears.enforcement.common.pyproject_discovery.discover_src_roots>`
      so chains like ``MemoriesCollection → SchemaBackedCollection →
      BaseCollection`` resolve across sibling packages.

    in :data:`~threetears.enforcement.common.modes.MODE_REPORT` the
    runner returns normally regardless of violations; in
    :data:`~threetears.enforcement.common.modes.MODE_STRICT` (the
    default) it calls :func:`pytest.fail` with the rendered report.

    the report's header surfaces both root sets explicitly so the
    operator can see what was scanned vs. what graph was built.

    :param config: per-repo enforcement config
    :ptype config: CacheEnforcementConfig
    :param walker: which walker(s) to invoke; one of
        ``"sqlite_construction"``, ``"wrapper_class"``,
        ``"pool_access"``, ``"missing_collection"``, ``"all"``
    :ptype walker: str
    :raises ValueError: ``walker`` is not in the accepted set
    :raises ExemptionError: the exemption file violates the
        rationale-discipline contract
    :raises pytest.fail.Exception: in strict mode with violations
    """
    if walker not in _VALID_WALKERS:
        raise ValueError(f"walker must be one of {sorted(_VALID_WALKERS)}, got {walker!r}")

    scan_roots = _resolve_scan_roots(config)
    inheritance_roots = _resolve_inheritance_roots(config)
    violations = _run_walkers(config, walker, scan_roots, inheritance_roots)
    exemptions = _load_exemptions(config.exemptions_path)
    filtered = apply_exemptions(violations, exemptions, config.repo_root)

    mode = resolve_mode(config.mode_env_var, default=MODE_STRICT)

    report = emit_report(
        filtered,
        scan_roots,
        exemptions,
        mode,
        config.repo_root,
        domain=f"cache.{walker}",
    )
    # the report header carries scan_roots; the inheritance roots are
    # printed alongside so the operator can see both scopes at a glance.
    inheritance_line = f"inheritance_roots: {[str(r) for r in inheritance_roots]}"
    print(report, file=sys.stderr)
    print(inheritance_line, file=sys.stderr)

    if mode == MODE_REPORT:
        return
    if filtered:
        pytest.fail(f"cache enforcement found {len(filtered)} violation(s):\n{report}\n{inheritance_line}")


def _resolve_scan_roots(config: CacheEnforcementConfig) -> tuple[Path, ...]:
    """pick the roots to scan for violations.

    explicit :attr:`CacheEnforcementConfig.scan_roots` wins; otherwise
    we use :func:`find_local_src_roots
    <threetears.enforcement.common.repo_layout.find_local_src_roots>`
    so violations stay scoped to the consumer repo's own code (never
    walks path-deps).

    :param config: per-repo enforcement config
    :ptype config: CacheEnforcementConfig
    :return: src roots to scan for violations
    :rtype: tuple[Path, ...]
    """
    if config.scan_roots is not None:
        return config.scan_roots
    return find_local_src_roots(config.repo_root)


def _resolve_inheritance_roots(
    config: CacheEnforcementConfig,
) -> tuple[Path, ...]:
    """pick the roots from which to build the cross-package inheritance graph.

    explicit :attr:`CacheEnforcementConfig.inheritance_roots` wins;
    otherwise we walk the path-dep graph via :func:`discover_src_roots
    <threetears.enforcement.common.pyproject_discovery.discover_src_roots>`
    so transitive Collection-base chains resolve across siblings
    declared as workspace members or path-deps.

    :param config: per-repo enforcement config
    :ptype config: CacheEnforcementConfig
    :return: src roots from which to build the inheritance graph
    :rtype: tuple[Path, ...]
    """
    if config.inheritance_roots is not None:
        return config.inheritance_roots
    return discover_src_roots(config.repo_root)


def _run_walkers(
    config: CacheEnforcementConfig,
    walker: str,
    scan_roots: tuple[Path, ...],
    inheritance_roots: tuple[Path, ...],
) -> list[Violation]:
    """invoke the requested walker(s) and return aggregated violations.

    :param config: per-repo enforcement config
    :ptype config: CacheEnforcementConfig
    :param walker: walker selector
    :ptype walker: str
    :param scan_roots: src roots to scan for violations
    :ptype scan_roots: tuple[Path, ...]
    :param inheritance_roots: src roots from which to build the
        cross-package class-base graph
    :ptype inheritance_roots: tuple[Path, ...]
    :return: combined violation list, in walker order
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    if walker in {"sqlite_construction", "all"}:
        violations.extend(
            find_sqlite_constructions(
                scan_roots,
                config.repo_root,
                config.allowed_sqlite_construction_sites,
            )
        )
    if walker in {"wrapper_class", "all"}:
        violations.extend(
            find_wrapper_classes(
                scan_roots,
                config.repo_root,
                inheritance_roots,
                config.base_collection_names,
                config.cache_method_names,
            )
        )
    if walker in {"pool_access", "all"}:
        violations.extend(
            find_direct_pool_access(
                scan_roots,
                config.repo_root,
                config.collection_table_allowlist,
            )
        )
    if walker in {"missing_collection", "all"}:
        violations.extend(
            find_missing_collections(
                scan_roots,
                config.repo_root,
                inheritance_roots,
                config.collection_table_allowlist,
                config.migration_table_allowlist,
                config.base_collection_names,
            )
        )
    return violations


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
