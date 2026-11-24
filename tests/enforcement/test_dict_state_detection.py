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
        file="packages/agent/tools/src/threetears/agent/tools/builtin/analyze_media.py",
        class_name="AnalyzeMediaTool",
        attr_name="_analyzers",
        rationale=("constructor-injected config mapping, not cached state"),
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
#: immutable value whole (threetears.core.frozen_state). Nothing in _ALLOWLIST excuses a change here (Pace,
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
