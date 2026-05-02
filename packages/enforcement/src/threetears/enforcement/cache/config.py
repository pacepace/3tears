"""configuration dataclass for cache-primitive-usage enforcement.

the cache domain enforces a single contract: every stateful data
surface in a consuming repo is a :class:`BaseCollection` (transitively)
rather than a bespoke wrapper around :class:`SQLiteBackend` or raw
``pool`` access. four walkers cover the four ways the contract can be
broken (bespoke construction, wrapper classes, direct pool access,
migration-defined tables that lack a Collection); one config dataclass
captures every per-repo knob.

the configurable knobs let consumers manage repo-specific allowlists
without forking the walkers:

- :attr:`allowed_sqlite_construction_sites` — relative-posix paths
  where bare :class:`SQLiteBackend(...)` construction is sanctioned
  (named factories like ``hub/common/l1_cache.py``). ``tests/`` trees
  are always permitted regardless of this set; the canonical
  hardcodes that.
- :attr:`collection_table_allowlist` — table-to-Collection-class
  mapping. drives both the direct-pool-access walker (which knows
  which tables have a Collection) and the missing-collection walker
  (which verifies the named class actually exists and transitively
  reaches a Collection base).
- :attr:`migration_table_allowlist` — bookkeeping tables defined in
  migrations that legitimately lack Collections
  (``_schema_migrations``, LangGraph ``checkpoints``).
- :attr:`base_collection_names` — terminal class names that satisfy
  "this is a Collection". default is ``{"BaseCollection"}``;
  intermediate bases (``SchemaBackedCollection``, etc.) are walked
  transitively via the common inheritance graph and do NOT need to
  be listed here.
- :attr:`scan_roots` / :attr:`inheritance_roots` — split src-root
  responsibilities. ``scan_roots`` is *where to look for violations*
  (defaults to :func:`find_local_src_roots
  <threetears.enforcement.common.repo_layout.find_local_src_roots>`,
  the consumer's own code only — never path-deps). ``inheritance_roots``
  is *where to build the cross-package class-base graph* used by
  :func:`find_wrapper_classes` and :func:`find_missing_collections`'s
  transitive subclass walk (defaults to :func:`discover_src_roots
  <threetears.enforcement.common.pyproject_discovery.discover_src_roots>`,
  which is path-dep aware so chains like ``MemoriesCollection →
  SchemaBackedCollection → BaseCollection`` resolve across package
  boundaries). splitting them prevents Option B path-dep walking from
  dragging upstream code into the violation scan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["CacheEnforcementConfig"]


_DEFAULT_CACHE_METHOD_NAMES: frozenset[str] = frozenset({
    "get", "put", "set", "delete", "upsert",
})


_DEFAULT_BASE_COLLECTION_NAMES: frozenset[str] = frozenset({"BaseCollection"})


@dataclass(frozen=True)
class CacheEnforcementConfig:
    """per-repo config for the cache-primitive-usage enforcement domain.

    :ivar repo_root: absolute path to the consumer repo's root (the
        directory containing its top-level ``pyproject.toml``).
    :ivar scan_roots: where to LOOK for violations. when ``None``, the
        runner uses :func:`find_local_src_roots
        <threetears.enforcement.common.repo_layout.find_local_src_roots>`
        — always scoped to the consumer's own code, never walks
        path-deps. set explicitly only to override discovery in tests
        or specialised harnesses.
    :ivar inheritance_roots: where to BUILD the inheritance graph
        (used by :func:`~threetears.enforcement.cache.walkers.find_wrapper_classes`
        and :func:`~threetears.enforcement.cache.walkers.find_missing_collections`
        for :func:`transitively_subclasses_any
        <threetears.enforcement.common.inheritance.transitively_subclasses_any>`).
        when ``None``, the runner uses :func:`discover_src_roots
        <threetears.enforcement.common.pyproject_discovery.discover_src_roots>`
        — path-dep aware so chains like ``MemoriesCollection →
        SchemaBackedCollection → BaseCollection`` resolve correctly
        across package boundaries.
    :ivar exemptions_path: path to ``_cache_exemptions.txt``;
        ``None`` means "no exemptions file". the file uses the shared
        rationale-required ``file:line:symbol`` format parsed by
        :func:`~threetears.enforcement.common.exemptions.parse_exemptions_with_rationale`.
    :ivar mode_env_var: environment variable controlling strict vs
        report mode. defaults to ``CACHE_ENFORCEMENT_MODE``.
    :ivar allowed_sqlite_construction_sites: forward-slash relative
        paths (relative to ``repo_root``) where bare
        :class:`SQLiteBackend(...)` construction is sanctioned. files
        under any ``tests/`` directory are always permitted regardless
        of this set; the canonical hardcodes that.
    :ivar collection_table_allowlist: ``{table_name:
        expected_collection_class_name}`` mapping. drives both
        :func:`~threetears.enforcement.cache.walkers.find_direct_pool_access`
        (which tables have a Collection) and
        :func:`~threetears.enforcement.cache.walkers.find_missing_collections`
        (which verifies the named class transitively reaches a
        Collection base).
    :ivar migration_table_allowlist: tables created in migrations that
        legitimately lack Collections. bookkeeping tables (e.g.
        ``_schema_migrations``), LangGraph state (``checkpoints``,
        ``checkpoint_writes``), and tables owned by sibling repos.
    :ivar cache_method_names: public method names that, combined with
        a :class:`SQLiteBackend` field, indicate a wrapper class.
        kept liberal so any new bespoke wrapper hits the walker;
        defaults to ``{"get", "put", "set", "delete", "upsert"}``.
    :ivar base_collection_names: terminal class names that satisfy
        "this is a Collection." default is just ``{"BaseCollection"}``
        — transitive walking through ``SchemaBackedCollection`` and
        other intermediate bases is automatic via
        :func:`~threetears.enforcement.common.inheritance.transitively_subclasses_any`.
        consumers do NOT need to enumerate intermediate bases here.
    """

    repo_root: Path
    scan_roots: tuple[Path, ...] | None = None
    inheritance_roots: tuple[Path, ...] | None = None
    exemptions_path: Path | None = None
    mode_env_var: str = "CACHE_ENFORCEMENT_MODE"
    allowed_sqlite_construction_sites: frozenset[str] = frozenset()
    collection_table_allowlist: dict[str, str] = field(default_factory=dict)
    migration_table_allowlist: frozenset[str] = frozenset()
    cache_method_names: frozenset[str] = field(
        default_factory=lambda: _DEFAULT_CACHE_METHOD_NAMES,
    )
    base_collection_names: frozenset[str] = field(
        default_factory=lambda: _DEFAULT_BASE_COLLECTION_NAMES,
    )
