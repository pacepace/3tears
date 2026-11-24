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
        file="packages/core/src/threetears/core/shared_arrow.py",
        class_name="SharedArrowStore",
        attr_name="_revokers",
        rationale=(
            "open file descriptors holding each live version's revoke flock in this process: an OS handle per version, never data; the lock is the cross-process state and dies with the process, which is what lets kept mappings go"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/shared_arrow.py",
        class_name="SharedArrowStore",
        attr_name="_evictable",
        rationale=(
            "the versions the last sweep found held only by idle workers' flocks, with their bytes: re-derived from the locks at every sweep, so a restart reads it again from the directory"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/shared_arrow.py",
        class_name="MappedTables",
        attr_name="_by_inode",
        rationale=(
            "each memory-mapped file's Arrow table by device and inode, lent to the next version's mapping so it "
            "maps only changed files; tables over this process's own address space, which no backend can hold, "
            "and gone with the mapping"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/shared_arrow.py",
        class_name="SharedArrowReader",
        attr_name="_lend",
        rationale=(
            "the mapped tables of a let-go version that the current version shares, kept for its first read to "
            "take instead of mapping again; memory maps of this process, which no backend can hold, and a "
            "restart maps afresh"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/shared_arrow.py",
        class_name="_KeptVersions",
        attr_name="_kept",
        rationale=(
            "the memory map a worker process holds of the version it read last, per store directory; a map is "
            "an open handle on this process's own address space, which no backend can hold, and a restart "
            "maps again on its next read"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/shared_arrow.py",
        class_name="SharedArrowStore",
        attr_name="_slots",
        rationale=(
            "the result slots this process reserved for its own worker processes and has not yet claimed or "
            "discarded, each with the bytes it may take; a slot names files in this pod's own memory-backed "
            "directory that a worker of this process is writing, so another pod has nothing to do with it, and a "
            "restart takes the directory over and removes every result in it"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/shared_arrow.py",
        class_name="SharedArrowStore",
        attr_name="_pieces",
        rationale=(
            "the pieces of each result slot this process reserved in pieces for its own worker processes and has "
            "not yet claimed or discarded (each piece also in _slots); they name files in this pod's own "
            "memory-backed directory, so another pod has nothing to do with them, and a restart takes the "
            "directory over and removes every result in it, as _slots"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/shared_arrow.py",
        class_name="SharedArrowStore",
        attr_name="_claimed_bytes",
        rationale=(
            "the bytes each result this process claimed still holds in its own memory-backed directory, until "
            "the sweep removes its files; it counts this pod's own files against this process's byte cap, so "
            "another pod has nothing to do with it, and a restart takes the directory over and removes every "
            "result in it, as _slots"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/shared_arrow.py",
        class_name="SharedArrowStore",
        attr_name="_handing",
        rationale=(
            "the hand-offs this process is writing now into its own memory-backed directory for its own worker "
            "processes, each with the bytes it may take; another pod has nothing to do with them, and a restart "
            "takes the directory over and removes every hand-off in it, as _slots"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/shared_arrow.py",
        class_name="SharedArrowStore",
        attr_name="_handed_bytes",
        rationale=(
            "the bytes each hand-off this process wrote still holds in its own memory-backed directory, until the "
            "sweep removes its files; it counts this pod's own files against this process's byte cap, as "
            "_claimed_bytes"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/shared_arrow.py",
        class_name="SharedArrowStore",
        attr_name="_manifests",
        rationale=(
            "the manifest of each version directory in this process's own memory-backed store, so a publish starts "
            "from its base's without reading it and a sweep finds the part files a retired version alone named; "
            "another pod has nothing to do with them, and a restart reads afresh the manifests the directory holds"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/shared_arrow.py",
        class_name="SharedArrowStore",
        attr_name="_part_keys",
        rationale=(
            "each part file of this process's own memory-backed store by its (device, inode), so the byte tally "
            "drops it as a sweep removes it; derived from the files the store wrote, as local, and recorded afresh "
            "at a restart"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/shared_arrow.py",
        class_name="_HeldFiles",
        attr_name="_dirs",
        rationale=(
            "each directory of this process's own memory-backed store with the files (device, inode, bytes) it "
            "held as it took its name, so a status read counts them without walking the directory; another pod "
            "has nothing to do with them, and a restart records afresh what the directory holds"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/shared_arrow.py",
        class_name="_HeldFiles",
        attr_name="_links",
        rationale=(
            "how many of this store's directories hold each file (by device and inode), so a hard-linked file is "
            "counted once and its bytes go with its last directory; derived from _dirs, as local and as "
            "short-lived"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/search/src/threetears/search/standalone.py",
        class_name="_Breakers",
        attr_name="_circuits",
        rationale=(
            "one live circuit-breaker state machine per failing upstream of this transport, guarded "
            "by a lock and read off a monotonic clock; a breaker protects this process's own calls, "
            "so state shared across pods or kept across a restart would trip one replica for another"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/security/door_proof.py",
        class_name="DoorProofModes",
        attr_name="_switches",
        rationale=(
            "one live DoorProofSwitch object per door in this process, built on first use; the "
            "switches hold a lock and a clock and are not data, so there is nothing to serialize"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/security/door_proof.py",
        class_name="DoorProofSwitch",
        attr_name="_absent",
        rationale=(
            "this replica's tally of unverified sender claims, for an operator reading the "
            "process; nothing decides on it (a window ends by its date), and the record that "
            "outlives the process is the door_proof_absent prometheus counter and the decision log"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/collections/base.py",
        class_name="_L1Fence",
        attr_name="_keys",
        rationale=(
            "the reads and writes of each key in flight in THIS process, ordering this process's "
            "own L1 writes against each other: L1 is per-process, so the state ordering writes to "
            "it is too, and a shared or durable copy would order writes to a cache no other pod "
            "has. the L2 revision is the cross-pod half of the same fence. an entry lives only "
            "while an operation on its key is in flight and is dropped when the last one ends"
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
        file="packages/langgraph/src/threetears/langgraph/checkpoint.py",
        class_name="ThreeTierCheckpointSaver",
        attr_name="_l2_write_bypassed",
        rationale=(
            "this process's record of the L2 keys whose own write and then delete both failed, "
            "so the key may still hold an older checkpoint. It is per process on purpose: it "
            "describes this worker's failed writes, and the worker reads those keys from L3 until "
            "one of its writes or deletes of the key lands. Other workers are protected by the "
            "delete, not by this record, so sharing it would add nothing. Bounded by "
            "4096 keys (oldest dropped, with a WARNING); a restart forgets it and costs "
            "at most one read of a stale L2 entry per key, the state before the bypass existed. "
            "Kept apart from the purge bypass (a set), which only a successful purge lifts"
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
        file="packages/agent/tools/src/threetears/agent/tools/relevance.py",
        class_name="ToolRelevanceIndex",
        attr_name="_inflight",
        rationale=(
            "the asyncio tasks embedding a tool set right now, keyed by content hash and removed "
            "when each finishes. Live task handles, non-serializable, meaningful only on the event "
            "loop that owns them; they exist so the latency ceiling cancels a caller's wait rather "
            "than the embedding, and so concurrent turns share one call. Nothing survives a task"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/search/src/threetears/search/standalone.py",
        class_name="_GuardedRoutes",
        attr_name="_proxied",
        rationale=(
            "live httpx transports -- open connection pools -- one per proxy URL a request's route "
            "named, owned by one client and closed with it. They are objects holding sockets, not "
            "data: an L1/L2/L3 backend stores values, cannot hold a connection, and a transport "
            "rebuilt per request would defeat the pooling it exists for. Nothing outlives the client"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/security/service_envelope.py",
        class_name="InProcessEnvelopeLedger",
        attr_name="_spent",
        rationale=(
            "spent envelope ids for a door ONE process consumes (a runner's own inbox): only this "
            "process can be replayed to, a restart brings a new identity generation a replayed refusal "
            "does not name, and each id is forgotten once its envelope expires (65s), so a shared or "
            "durable store would add a round trip and protect nothing"
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
        file="packages/agent/tools/src/threetears/agent/tools/dynamic_pod.py",
        class_name="DynamicToolPod",
        attr_name="_spec_locks",
        rationale=(
            "spec_key -> asyncio.Lock serializing register_spec / deregister_spec for one spec in "
            "this process, so two overlapping rebuilds cannot overwrite each other's bookkeeping; "
            "a lock is an in-process synchronisation primitive, not state -- it cannot be serialized "
            "into an L1/L2/L3 backend, and it guards the pod-local _resources / _tool_names above"
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
        file="packages/registry/src/threetears/registry/catalog.py",
        class_name="ToolCatalog",
        attr_name="_written",
        rationale=(
            "what THIS process last wrote to each catalog key and saw land, key -> content digest, so a "
            "re-registration that changes nothing is not written again. it describes this process's own "
            "writes, so no other replica may share it; it is cleared whenever a bucket is bound and dropped "
            "per key on any failed write or delete, and a restart correctly starts it empty and writes once"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/registry/src/threetears/registry/health.py",
        class_name="HeartbeatSubscriber",
        attr_name="_heard_at",
        rationale=(
            "THIS registry's own liveness record of the pods it has heard, pod id -> when it last heard it. "
            "it is the judgement that must survive an L2 outage, so a shared store is exactly what it must "
            "not depend on, and an L1 backend adds nothing to a per-process view; a restart correctly "
            "rebuilds it from the next heartbeats. keyed only by pods in the known-pods set and dropped with "
            "a pod when it is deregistered"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/registry/src/threetears/registry/health.py",
        class_name="HeartbeatSubscriber",
        attr_name="_local_misses",
        rationale=(
            "THIS registry's own liveness record of the pods it has heard, pod id -> consecutive sweeps "
            "missed. it is the judgement that must survive an L2 outage, so a shared store is exactly what it"
            " must not depend on, and an L1 backend adds nothing to a per-process view; a restart correctly "
            "rebuilds it from the next heartbeats. keyed only by pods in the known-pods set and dropped with "
            "a pod when it is deregistered"
        ),
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
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/backends/git/layout.py",
        class_name="TemplateLayout",
        attr_name="_tables",
        rationale=(
            "static config: the table layouts a product passes to the constructor, by table name, "
            "built once and never changed after"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/backends/git/layout.py",
        class_name="TemplateLayout",
        attr_name="_by_table",
        rationale=(
            "static config: the compiled path pattern of each table in _tables, built once in the "
            "constructor from the same layouts"
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

#: Where an object's dict or set state is never changed in place after ``__init__``: its owner assigns the next
#: immutable value whole (threetears.core.collections.frozen_state). Nothing in _ALLOWLIST excuses a change here (Pace,
#: 2026-11-23: "small, owner-only, under the pass lock" is no exception).
_IMMUTABLE_ROOTS = (_REPO_ROOT / "packages/core/src/threetears/core/collections",)

#: Changes in place under _IMMUTABLE_ROOTS that are per key, where rebuilding the whole value at each change costs the
#: keys in flight: each needs a redesign agreed first, named here. A debt list, audited for staleness.
_IMMUTABLE_KNOWN_VIOLATIONS = (
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/collections/base.py",
        class_name="_L1Fence",
        attr_name="_keys",
        rationale=(
            "one mutable activity record a key in flight, shared by identity with every ticket on the key; immutable it "
            "needs a fence-wide sequence the tickets compare against, a redesign of the L1 fence to agree first"
        ),
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/collections/flush.py",
        class_name="WriteBuffer",
        attr_name="_buf",
        rationale="one pending write a key: rebuilt whole a write it is quadratic; migrates to SQLiteBackend L1",
    ),
    DictStateAllowlistEntry(
        file="packages/core/src/threetears/core/collections/flush.py",
        class_name="WriteBuffer",
        attr_name="_in_flight",
        rationale="the keys a drain claimed, one a pending write: migrates with _buf to SQLiteBackend L1",
    ),
)

_CONFIG = DictStateConfig(
    repo_root=_REPO_ROOT,
    src_roots=_SRC_ROOTS,
    allowlist=_ALLOWLIST,
    known_violations=_KNOWN_VIOLATIONS,
    immutable_roots=_IMMUTABLE_ROOTS,
    immutable_known_violations=_IMMUTABLE_KNOWN_VIOLATIONS,
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
