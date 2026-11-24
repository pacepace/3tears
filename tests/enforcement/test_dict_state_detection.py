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

**ALLOWLIST is forever, and there is no debt list.** The allowlist is state that genuinely
cannot live in a backend -- live ``asyncio`` handles, the collection infrastructure itself --
audited: an entry that stops matching a real violation is reported as stale. Nothing is
tracked as an acknowledged violation (Pace, 2026-11-23: redesign it, never waive it); the
last debts, the L1 fence's per-key records and the write buffer's dict and claim set, were
redesigned as immutable packed batches and the lists deleted.
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
        file="packages/agent/tools/src/threetears/agent/tools/builtin/analyze_media.py",
        class_name="AnalyzeMediaTool",
        attr_name="_analyzers",
        rationale=("constructor-injected config mapping, not cached state"),
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

#: Where an object's dict or set state is never changed in place after ``__init__``: its owner assigns the next
#: immutable value whole (threetears.core.frozen_state). Nothing in _ALLOWLIST excuses a change here (Pace,
#: 2026-11-23: "small, owner-only, under the pass lock" is no exception).
_IMMUTABLE_ROOTS = (_REPO_ROOT / "packages/core/src/threetears/core/collections",)

_CONFIG = DictStateConfig(
    repo_root=_REPO_ROOT,
    src_roots=_SRC_ROOTS,
    allowlist=_ALLOWLIST,
    immutable_roots=_IMMUTABLE_ROOTS,
)


class TestDictStateDetection:
    def test_no_unexplained_raw_dict_state(self) -> None:
        """Every raw-dict assignment in an ``__init__`` is allowlisted. A new one fails here rather than becoming a pod-local cache nobody meant
        to add."""
        run_dict_state_enforcement(_CONFIG, walker="detect")

    def test_no_stale_allowlist_entries(self) -> None:
        """Guard the guard: an entry that no longer matches a real violation is reported, so
        the allowlist cannot outlive the code it describes."""
        run_dict_state_enforcement(_CONFIG, walker="allowlist_integrity")
