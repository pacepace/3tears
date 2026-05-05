"""thin shell — actual walker logic in :mod:`threetears.enforcement.cache`.

the 3tears self-test consumes the shared ``3tears-enforcement``
workspace package and injects only the per-repo configuration
(allowed-construction sites, table-to-collection mapping, migration
allowlist). every walker, exemption parser, mode resolver, report
emitter, and inheritance walker lives in the package — this file
exists solely to declare the per-repo knobs and call the runner.

the four test methods preserve the canonical class / method names
so any external CI looking for them continues to find them.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.cache import (
    CacheEnforcementConfig,
    run_cache_enforcement,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


_CONFIG = CacheEnforcementConfig(
    repo_root=_REPO_ROOT,
    allowed_sqlite_construction_sites=frozenset(
        {
            # namespace-task-01 phase 8.5l-3: the registry process gained
            # its first L1 tier for :class:`HeartbeatCollection`. the
            # sanctioned construction site is the per-process factory.
            # every other site is a bespoke wrapper in disguise.
            "packages/registry/src/threetears/registry/l1_cache.py",
        }
    ),
    collection_table_allowlist={
        # 3tears package-owned tables (every table here is created by a
        # migration under ``packages/*/migrations/`` and has a
        # Collection somewhere under ``packages/*/src/``).
        "conversations": "ConversationsCollection",
        "context_items": "ContextItemCollection",
        "memories": "MemoriesCollection",
        "media": "MediaCollection",
        "media_content": "MediaContentCollection",
        "memory_chunks": "MemoryChunkCollection",
        # conversation_memory_refs adopted under namespace-task-01
        # phase 8.5l-2 on top of 8.5l-1's composite-pk BaseCollection
        # support.
        "conversation_memory_refs": "MemoryRefsCollection",
        "workspaces": "WorkspaceCollection",
        "workspace_files": "WorkspaceFileCollection",
        "workspace_file_versions": "WorkspaceFileVersionCollection",
        # pod_heartbeats adopted under namespace-task-01 phase 8.5l-3.
        # no migration creates this table (the walker only reports
        # missing Collections for migration-defined tables); listing
        # it here excludes it from the pool-access allowlist path
        # should a future caller reach for raw SQL.
        "pod_heartbeats": "HeartbeatCollection",
    },
    migration_table_allowlist=frozenset(
        {
            # internal migration runner state — not a business entity.
            "_schema_migrations",
            # checkpoints / checkpoint_writes are accessed via
            # ``threetears.langgraph.ThreeTierCheckpointSaver`` (the
            # LangGraph ``BaseCheckpointSaver`` contract). LangGraph
            # mandates shape details that cannot be expressed through
            # ``BaseCollection``; 3tears wraps the three-tier backend
            # under the LangGraph contract so the platform gets
            # L1/L2/L3 while satisfying LangGraph.
            "checkpoints",
            "checkpoint_writes",
            # config_epochs is platform-internal coordination state
            # accessed by :class:`threetears.epoch.client.EpochClient`
            # via raw atomic INSERT ... ON CONFLICT. wrapping in a
            # Collection would be circular: the table IS the
            # cache-coordination signal that BaseCollection's L1/L2
            # eviction broadcasts cross-product, so layering its own
            # caches on top of itself buys nothing and risks staleness
            # of the staleness signal. one row per subject, atomic
            # row-lock serialization, no derived view to cache.
            "config_epochs",
        }
    ),
    exemptions_path=_REPO_ROOT / "tests" / "enforcement" / "_cache_exemptions.txt",
)


class TestCachePrimitiveUsage:
    """four enforcement tests across the 3tears Collection primitive."""

    def test_no_bespoke_sqlite_backend_construction(self) -> None:
        """SQLiteBackend is constructed only inside sanctioned factories."""
        run_cache_enforcement(_CONFIG, walker="sqlite_construction")

    def test_no_bespoke_cache_wrapper_classes(self) -> None:
        """no class stores SQLiteBackend + exposes cache api without BaseCollection."""
        run_cache_enforcement(_CONFIG, walker="wrapper_class")

    def test_no_direct_pool_access_to_collection_tables(self) -> None:
        """pool.fetch/execute never targets a Collection-backed table directly."""
        run_cache_enforcement(_CONFIG, walker="pool_access")

    def test_all_tables_have_collections(self) -> None:
        """every migration-defined table has a matching Collection class."""
        run_cache_enforcement(_CONFIG, walker="missing_collection")
