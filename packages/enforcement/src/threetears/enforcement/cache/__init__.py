"""cache enforcement domain — the 3tears Collection-as-the-primitive contract.

four walkers in this domain enforce that every stateful data surface
in a 3tears-consuming repo is a :class:`BaseCollection` (transitively)
rather than a bespoke wrapper around :class:`SQLiteBackend` or a raw
``pool.fetch`` call:

- :func:`find_sqlite_constructions` — flag every
  ``SQLiteBackend(...)`` call outside the sanctioned factories.
- :func:`find_wrapper_classes` — flag classes that hold a
  ``SQLiteBackend`` field AND expose cache api (``get`` / ``put`` /
  ``set`` / ``delete`` / ``upsert``) AND do NOT transitively subclass
  any ``base_collection_names``.
- :func:`find_direct_pool_access` — flag ``pool.fetch`` /
  ``fetchrow`` / ``fetchval`` / ``execute`` calls whose SQL text
  references a table mapped in ``collection_table_allowlist``.
- :func:`find_missing_collections` — flag migration-defined tables
  whose mapped Collection class does not transitively subclass any
  ``base_collection_names``. **uses** :func:`transitively_subclasses_any
  <threetears.enforcement.common.inheritance.transitively_subclasses_any>`
  so chains like ``MemoriesCollection → SchemaBackedCollection →
  BaseCollection`` resolve correctly across path-dep package
  boundaries (this is the originally-failing bug fix the cache lift
  was created for).

per-repo configuration goes through :class:`CacheEnforcementConfig`;
:func:`run_cache_enforcement` is the pytest-friendly entry point that
orchestrates the walkers, applies the rationale-required exemption
list, emits the report, and fails in strict mode.
"""

from threetears.enforcement.cache.config import (
    CacheEnforcementConfig,
)
from threetears.enforcement.cache.runner import (
    run_cache_enforcement,
)
from threetears.enforcement.cache.walkers import (
    find_direct_pool_access,
    find_missing_collections,
    find_sqlite_constructions,
    find_wrapper_classes,
)

__all__ = [
    "CacheEnforcementConfig",
    "find_direct_pool_access",
    "find_missing_collections",
    "find_sqlite_constructions",
    "find_wrapper_classes",
    "run_cache_enforcement",
]
