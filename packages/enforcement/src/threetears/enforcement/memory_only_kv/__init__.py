"""memory-only KV enforcement domain.

**NATS is L2.** A KV bucket asking for ``storage="file"`` is the cache tier quietly taking the
source-of-truth role while keeping none of the properties that role needs: one replica, no
backups, no schema, no migrations. It reads as prudence and it is a promotion nobody reviewed.

The rule is one line: **memory, always.** Anything that genuinely cannot be lost belongs in a
``BaseCollection``, which composes L1, L2 (this same NATS, memory-backed) and L3 rather than
making anyone choose between them; ``threetears.core.coordination.tables`` is the worked example,
and the primitives over it are what this rule's last four exemptions became.

It has history. File storage was reached for twice in one week as the answer to an intermittent
login failure, once in this repo and once in a consumer, and neither time was it the cause. The
reflex is real and the gate is aimed at the reflex.

**There is no exemption mechanism, and that is the point.** Each of the four file-backed buckets
this rule removed carried a specific, honest rationale naming the work that would remove it, and
that is how they stayed for months. The work is done, so the escape hatch went with it.

Per-repo configuration goes through :class:`MemoryOnlyKvConfig`;
:func:`run_memory_only_kv_enforcement` is the pytest-friendly entry point.
"""

from threetears.enforcement.memory_only_kv.config import DEFAULT_SOURCE_GLOBS, MemoryOnlyKvConfig
from threetears.enforcement.memory_only_kv.runner import (
    run_memory_only_kv_enforcement,
    source_files,
)
from threetears.enforcement.memory_only_kv.walkers import (
    DURABLE_STORAGE,
    KV_OPENING_CALLS,
    STORAGE_KEYWORD,
    FileBackedKvCall,
    file_backed_kv_calls,
    scan_for_file_backed_kv,
)

__all__ = [
    "DEFAULT_SOURCE_GLOBS",
    "DURABLE_STORAGE",
    "KV_OPENING_CALLS",
    "STORAGE_KEYWORD",
    "FileBackedKvCall",
    "MemoryOnlyKvConfig",
    "file_backed_kv_calls",
    "run_memory_only_kv_enforcement",
    "scan_for_file_backed_kv",
    "source_files",
]
