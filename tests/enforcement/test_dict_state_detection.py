"""Thin shell over the canonical dict-state walker.

Persistent state in an ``__init__`` goes in an L1 backend or NATS KV, not a raw dict: a
dict is per-process, so two pods disagree and a restart forgets. The walker, the allowlist
machinery and the stale-entry meta-check all live in
:mod:`threetears.enforcement.dict_state_detection`.

This replaces a hand-rolled copy that lived in ``packages/core/tests/``. The copy was not
wrong -- it was the only one of the two that worked, because the shared domain keyed its
allowlist on ``(file, exact line, attr)`` and had never had a caller to prove that unusable.
A line number is invalidated by any edit ABOVE the assignment, so every entry below would
have gone stale the first time someone added an import. The shared domain keys on
``(file, class_name, attr)`` now, which is what the copy always did and what survives a
refactor that does not move the attribute.

**ALLOWLIST is forever; KNOWN_VIOLATIONS is a debt list.** The first is state that
genuinely cannot live in a backend -- live ``asyncio`` handles, the collection
infrastructure itself. The second is state that should migrate and has not. Both are
audited: an entry that stops matching a real violation is reported as stale, so the lists
cannot quietly outlive the code they describe.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.dict_state_detection import (
    DictStateAllowlistEntry,
    DictStateConfig,
    run_dict_state_enforcement,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The packages whose state this domain watches. Explicit rather than discovered: these are
#: the trees the allowlist below describes, and a wider scan would report packages nobody has
#: triaged yet as unexplained failures.
#:
#: ``search`` is here with nothing to declare, which is the point of adding it (search-spec.md
#: §4.10 b). Its in-process limiter is the one piece of state it is *expected* to grow (§3.9,
#: SR-O2) and that entry is to be argued when the limiter lands -- so the root goes in first,
#: while the answer is still "none", rather than arriving alongside the state it should have
#: been questioning.
_SRC_ROOTS = (
    _REPO_ROOT / "packages/core/src/threetears/core",
    _REPO_ROOT / "packages/registry/src/threetears/registry",
    _REPO_ROOT / "packages/search/src/threetears/search",
    _REPO_ROOT / "packages/agent/memory/src/threetears/agent/memory",
    _REPO_ROOT / "packages/agent/tools/src/threetears/agent/tools",
    _REPO_ROOT / "packages/langgraph/src/threetears/langgraph",
)

#: Genuinely ephemeral, non-serializable, or the infrastructure itself -- never migrating.
_ALLOWLIST = (
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/collections/registry.py",
        class_name="CollectionRegistry",
        attr_name="_collections",
        rationale=("registry managing L1/L2/L3 backends, IS the infrastructure"),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/collections/registry.py",
        class_name="CollectionRegistry",
        attr_name="_overrides",
        rationale=("per-collection backend overrides, IS the infrastructure"),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/collections/derived.py",
        class_name="DerivedCollection",
        attr_name="_inflight",
        rationale=(
            "live asyncio.Lock handles, one per key being derived — bound to a single event loop "
            "and non-serializable, so they cannot live in an L1/L2/L3 backend by construction. "
            "cross-POD single-flight is a separate mechanism and does use NATS "
            "(nats_distributed_lock); this map is only the in-process half, and entries are "
            "dropped as soon as nobody holds or awaits a key"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/task_registry.py",
        class_name="KeyedTaskRegistry",
        attr_name="_tasks",
        rationale=(
            "live asyncio.Task handles — per-worker + non-serializable; the cancel registry IS "
            "the ephemeral infrastructure, cannot live in an L1/L2/L3 backend"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/data/store.py",
        class_name="DataStore",
        attr_name="_collections",
        rationale=("registry of collection instances, IS the infrastructure"),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/data/migrations/runner.py",
        class_name="MigrationRunner",
        attr_name="_packages",
        rationale=("static config, PackageMigrations registered once at startup"),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/data/migrations/registry.py",
        class_name="PackageMigrations",
        attr_name="_versions",
        rationale=("static config, migration callables registered once at startup"),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/data/migrations/registry.py",
        class_name="PackageMigrations",
        attr_name="_downgrades",
        rationale=("static config, downgrade callables registered once at startup"),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/cache/kv.py",
        class_name="NatsKvClient",
        attr_name="_buckets",
        rationale=("live NATS KV connection references, non-serializable"),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/cache/sqlite.py",
        class_name="SQLiteBackend",
        attr_name="_schema_info",
        rationale=("schema metadata for type-aware serialization, infrastructure"),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/cache/duckdb.py",
        class_name="DuckDBBackend",
        attr_name="_schema_info",
        rationale=("schema metadata for type-aware serialization, infrastructure"),
    ),
    DictStateAllowlistEntry(
        file="packages/agent/tools/src/threetears/agent/tools/server.py",
        class_name="ToolServer",
        attr_name="_tools",
        rationale=("live TearsTool instances, non-serializable"),
    ),
    DictStateAllowlistEntry(
        file="packages/agent/tools/src/threetears/agent/tools/registry.py",
        class_name="ToolRegistry",
        attr_name="_factories",
        rationale=("static config, tool factories registered once at startup"),
    ),
    DictStateAllowlistEntry(
        file="packages/agent/tools/src/threetears/agent/tools/builtin/analyze_media.py",
        class_name="AnalyzeMediaTool",
        attr_name="_analyzers",
        rationale=("constructor-injected config mapping, not cached state"),
    ),
    DictStateAllowlistEntry(
        file="packages/agent/tools/src/threetears/agent/tools/object_resolver.py",
        class_name="HubObjectResolver",
        attr_name="_cache",
        rationale=(
            "per-instance memo of IMMUTABLE (customer_id, object_id)->ObjectHandle mappings (an "
            "object id->key binding never changes once committed), FIFO-capped, tenant-keyed on "
            "the VERIFIED customer so no cross-tenant reuse; genuinely ephemeral per-pod (no "
            "cross-instance coherence needed precisely because the mappings are immutable). Its "
            "sibling engagement resolver deliberately has NO cache because scope IS mutable"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/agent/tools/src/threetears/agent/tools/relevance.py",
        class_name="ToolRelevanceIndex",
        attr_name="_cache",
        rationale=(
            "per-instance memo of tool-set-content-hash -> {tool_name: embedding vector}, LRU- "
            "bounded (cache_size). Purely a performance optimization over re-embedding an "
            "unchanged tool set — a cache miss (cold start, eviction, or content-hash change from "
            "a live-discovered/admin-edited tool) just re-embeds via the batch embed call, so "
            "correctness never depends on a hit. No cross-instance coherence needed for the same "
            "reason: a pod that misses just pays one extra embedding call, same shape as the "
            "already-allowlisted HubObjectResolver._cache above"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/testing/kv.py",
        class_name="FakeKvBucket",
        attr_name="_entries",
        rationale=(
            "an in-memory test double's entire storage, standing in for a JetStream KV bucket. It "
            "exists precisely so a test needs no backend at all, so routing it through an "
            "L1/L2/L3 backend would defeat its purpose. Published rather than kept per-repo "
            "because every consumer was hand-rolling this same double and drifting from the "
            "wrapper it stands in for"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/testing/kv.py",
        class_name="FakeNatsClient",
        attr_name="_buckets",
        rationale=(
            "the double's bucket registry, mirroring NatsClient's own internal bucket cache so "
            "repeat kv_bucket calls return the same instance -- same test-double rationale as "
            "FakeKvBucket._entries above"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/agent/tools/src/threetears/agent/tools/dynamic_pod.py",
        class_name="DynamicToolPod",
        attr_name="_resources",
        rationale=(
            "live per-spec resource handles (drivers/clients) built by build_tools; non- "
            "serializable and rebuilt from load_specs on restart — the same infrastructure shape "
            "as the already-allowlisted ToolServer._tools, cannot live in an L1/L2/L3 backend"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/agent/tools/src/threetears/agent/tools/dynamic_pod.py",
        class_name="DynamicToolPod",
        attr_name="_tool_names",
        rationale=(
            "per-run bookkeeping of spec_key -> registered tool mcp_names, held in lockstep with "
            "the live ToolServer registrations so deregister_spec can unregister exactly what it "
            "registered; ephemeral pod-local, no cross-instance coherence, rebuilt from "
            "load_specs on restart"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/backends/sql.py",
        class_name="SqlL3Backend",
        attr_name="_schemas",
        rationale=(
            "static config, table→TableSchema registered once at collection construction "
            "(startup); non-serializable (stores TableSchema dataclass refs, not data), read-only "
            "on the CRUD hot path"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/search/src/threetears/search/limiter.py",
        class_name="InProcessRateLimiter",
        attr_name="_buckets",
        rationale=(
            "the argued SR-O2 entry search-spec.md §3.9 said would arrive with the in-process "
            "limiter (D8's second mechanism, for the deployment mode that has no bus to share a "
            "bucket through). Four reasons it is an allowlist entry and not a migration: (a) the "
            "state is a monotonic-clock reading plus a token count, which is not comparable "
            "outside this process, so there is no serialisation another reader could use; (b) the "
            "cross-instance version already exists as a DIFFERENT object -- core's NATS "
            "TokenBucket, host-injected -- so this is not a backend-shaped thing built badly, it "
            "is the half of the ruling that must hold where there is no backend to reach, and the "
            "leaf may not import core anyway (SR-L7); (c) a restart forgetting the buckets is "
            "correct, not a defect: every key resets to full, which is what an upstream would "
            "infer from a process that had been making no calls, so the loss is bounded by one "
            "burst; (d) it is bounded by construction -- keys are the host's configured "
            "(instance, egress) pairs at two floats each, soft-capped at max_tracked_keys with "
            "full buckets evicted first, and evicting a full bucket is unobservable because a "
            "fresh key starts full"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/registry/src/threetears/registry/catalog.py",
        class_name="ToolCatalog",
        attr_name="_entries",
        rationale=("synced from NATS KV on load, not a local-only cache"),
    ),
    DictStateAllowlistEntry(
        file="packages/langgraph/src/threetears/langgraph/events.py",
        class_name="FrameworkEventRegistry",
        attr_name="_by_name",
        rationale=(
            "static config, FrameworkEvent class objects registered once at import time; non- "
            "serializable (stores Python class refs, not data)"
        ),
    ),
)

#: Tracked for migration: real violations that are acknowledged, not accepted.
_KNOWN_VIOLATIONS = (
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/collections/flush.py",
        class_name="WriteBuffer",
        attr_name="_buf",
        rationale=("pending write buffer in raw dict, migrate to SQLiteBackend L1"),
    ),
)

_CONFIG = DictStateConfig(
    repo_root=_REPO_ROOT,
    src_roots=_SRC_ROOTS,
    allowlist=_ALLOWLIST,
    known_violations=_KNOWN_VIOLATIONS,
)


class TestDictStateDetection:
    def test_no_unexplained_raw_dict_state(self) -> None:
        """Every raw-dict assignment in an ``__init__`` is either allowlisted or a known
        violation. A new one fails here rather than becoming a pod-local cache nobody meant
        to add."""
        run_dict_state_enforcement(_CONFIG, walker="detect")

    def test_no_stale_allowlist_entries(self) -> None:
        """Guard the guard: an entry that no longer matches a real violation is reported, so
        the lists cannot outlive the code they describe. This is what stops KNOWN_VIOLATIONS
        from silently becoming a list of things that were fixed years ago."""
        run_dict_state_enforcement(_CONFIG, walker="allowlist_integrity")
