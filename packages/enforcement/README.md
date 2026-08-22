# 3tears-enforcement

Shared static-analysis enforcement scanners for the 3tears ecosystem.

## What this package does

Each module under `threetears.enforcement.<domain>` ships an AST-based scanner that enforces a single architectural invariant across a Python source tree. Consumer repos import the scanner, inject their per-repo configuration (allowlists, exemption files, src roots), and run it from a thin pytest test class.

This replaces a previous pattern in which the same enforcement test files were vendored verbatim across multiple repos with manual sync requirements. The shared package eliminates duplication and drift while keeping per-repo configuration where it belongs.

## Domains

> **Adopting `invalidation_listener` in a repo that already runs a local copy.**
> The hub's pre-shared walker keyed its exemption file by BARE MODULE PATH
> (`src/aibots/gateway/acl.py`). The shared parser wants a `path:line:symbol`
> triple and rejects anything else with `ExemptionError`, so an unmigrated file
> does not degrade -- it aborts the domain. Rewrite each entry as
> `<path>:*:<registry-spelling>`, keeping the `# rationale:` line above it: `*`
> means "any line in that file", which is what you want, because keying an
> exemption to a line number silently stops matching when the line moves.

| Module | Invariant enforced |
|---|---|
| `cache` | Every stateful data surface routes through `BaseCollection`; no bespoke SQLiteBackend wrappers; no direct pool access to Collection-backed tables; every migration-defined table has a Collection class. |
| `underscore_access` | Underscore prefix is a stability contract: no cross-module private import, no cross-class protected access, modules with public names have `__all__`, no subclass shadowing of base private attributes, no `__all__` listing private names. |
| `codebase_conventions` | No bare `print()`, no stdlib `logging.getLogger` (use `threetears.observe`), `from __future__ import annotations` required, return type annotations required. |
| `coercion_coverage` | Tool subclasses override `execute`, never `run`, preserving the `normalize_kwargs → execute` input-coercion path. |
| `dependency_alignment` | Declared dependencies match actual imports (no undeclared module-top sibling import, no declaration nothing imports); designated contracts packages import only stdlib, their own namespace, and configured extras; a package with a pinned `DependencyFloor` declares exactly the ruled hard-dependency list, no more and no fewer. |
| `dict_state_detection` | No raw `dict`/`OrderedDict` persistent state in `__init__`; use `SQLiteBackend` (L1) or NATS KV for shared state. |
| `invalidation_listener` | Every L2-live `CollectionRegistry` starts the cross-pod invalidation listener, and every start is paired with a stop. Carries a REQUIRED non-vacuity floor: a scan reaching nothing reports what a clean repo reports. |
| `logger_coverage` | Every production module declares a module-level `log = get_logger(__name__)` unless explicitly exempt. |
| `migration_yugabyte_safety` | Migration shapes are yugabyte-safe per `threetears.core.data.migrations.enforcement`. |
| `nats_wrapper_usage` | All `nats-py` imports route through `threetears.nats.NatsClient`; no direct `import nats`. |
| `no_silent_swallow` | Exception handlers must log, re-raise, or carry `# NOSILENT: <reason>`. |
| `no_stdlib_logging` | No production module imports stdlib `logging` directly; use `threetears.observe`. |

## How to use it

Each domain exposes a configuration dataclass and a high-level runner:

```python
# tests/enforcement/test_cache_primitive_usage.py
from pathlib import Path
from threetears.enforcement.cache import CacheEnforcementConfig, run_cache_enforcement

_CONFIG = CacheEnforcementConfig(
    repo_root=Path(__file__).parents[2],
    allowed_sqlite_construction_sites=frozenset({
        "packages/registry/src/threetears/registry/l1_cache.py",
    }),
    collection_table_allowlist={
        "memories": "MemoriesCollection",
        # ... per-repo
    },
    migration_table_allowlist=frozenset({"_schema_migrations"}),
    exemptions_path=Path(__file__).parent / "_cache_exemptions.txt",
    enforcement_mode_env_var="CACHE_ENFORCEMENT_MODE",
)

class TestCachePrimitiveUsage:
    def test_no_bespoke_sqlite_backend_construction(self) -> None:
        run_cache_enforcement(_CONFIG, walker="sqlite_construction")

    def test_no_bespoke_cache_wrapper_classes(self) -> None:
        run_cache_enforcement(_CONFIG, walker="wrapper_class")

    def test_no_direct_pool_access_to_collection_tables(self) -> None:
        run_cache_enforcement(_CONFIG, walker="pool_access")

    def test_all_tables_have_collections(self) -> None:
        run_cache_enforcement(_CONFIG, walker="missing_collection")
```

Per-repo exemption files (e.g., `_cache_exemptions.txt`) stay in the consumer repo's `tests/enforcement/` directory. The package's `parse_exemptions_with_rationale` reads them at test time.

## How to onboard a new repo

1. Add `3tears-enforcement` as a dev dependency.
2. For each domain you want to enforce: create a thin shell test file at `tests/enforcement/test_<domain>.py` following the pattern above. Inject your repo's allowlists/exemptions.
3. Create per-domain exemption files at `tests/enforcement/_<domain>_exemptions.txt` if needed. Every exemption requires a preceding `# rationale: <specific reason>` line.
4. Run `pytest tests/enforcement/` to verify the scanners work against your tree.

## How to add a new enforcement domain

1. Create `src/threetears/enforcement/<domain>/` with `walkers.py`, `config.py`, `runner.py`, and `__init__.py`.
2. Use `common/` helpers (`ast_helpers`, `collection_registry`, `repo_layout`, `pyproject_discovery`, `inheritance`, `exemptions`, `modes`, `violations`, `reports`). Do not duplicate scaffolding. `collection_registry` in particular: if your domain asks which `CollectionRegistry` names are L2-live, take the answer from there rather than re-deriving it -- two copies of that question is what this module exists to end.
3. Walkers return `list[Violation]`. Configs are frozen dataclasses. Runners orchestrate walker → exemption-application → mode-resolution → report.
4. Write unit tests in `tests/<domain>/`.
5. Document the domain in this README.
