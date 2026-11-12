"""Arrow's memory pool in a serving process: the system allocator, and what the pool holds unused handed back.

pyarrow's default pool in our images is mimalloc. Its per-thread heaps keep the pages they freed for the next
allocation on that thread, and a serving process allocates from many threads (the event loop's executor, Arrow's own
pools): reports' pod held 946 MiB in it after a refresh against a fresh replica's 456 (``/proc/1/smaps``, 2026-10-26),
and ``release_unused`` did not reach the other threads' heaps. The system allocator returns freed pages to one place,
and under glibc ``release_unused`` also runs ``malloc_trim(0)``.

Only the serving process switches (:func:`use_system_pool_in_serving_process`): worker processes started by spawn
import pyarrow afresh and keep mimalloc, which is fastest for their short whole-batch work and is given back when the
worker ends.
"""

from __future__ import annotations

import time

from threetears.observe import get_logger

try:
    import pyarrow as pa

    _PYARROW = True
except ImportError:  # NOSILENT: the snapshot extra is optional; both calls refuse without it, saying so
    _PYARROW = False

__all__ = ["release_unused", "use_system_pool_in_serving_process"]

log = get_logger(__name__)


def _need_pyarrow() -> None:
    if not _PYARROW:
        raise ImportError("Arrow's memory pool needs pyarrow: install 3tears[snapshot]")


def use_system_pool_in_serving_process() -> str:
    """make the system allocator Arrow's default memory pool in the calling process.

    Call it once at the serving process's start, before any Arrow allocation: buffers already allocated stay in the
    pool that made them. It sets this process only: a worker started by spawn imports pyarrow afresh and keeps
    mimalloc; a child forked after the call inherits the system pool. A second call changes nothing.

    :return: the backend the default pool was before the call (``"system"`` on a second call)
    :rtype: str
    :raises ImportError: without pyarrow
    """
    _need_pyarrow()
    before = str(pa.default_memory_pool().backend_name)
    if before != "system":
        pa.set_memory_pool(pa.system_memory_pool())
    log.info(
        "Arrow's default memory pool in the serving process",
        extra={"extra_data": {"before": before, "after": str(pa.default_memory_pool().backend_name)}},
    )
    return before


def release_unused() -> int:
    """hand back to the system what Arrow's default pool holds and no longer uses (on the system pool under glibc,
    ``malloc_trim(0)``); for a serving process once a large piece of work is over.

    :return: the bytes the pool still has allocated after it
    :rtype: int
    :raises ImportError: without pyarrow
    """
    _need_pyarrow()
    pool = pa.default_memory_pool()
    started = time.monotonic()
    pool.release_unused()
    allocated = int(pool.bytes_allocated())
    log.info(
        "Arrow's memory pool released what it held unused",
        extra={
            "extra_data": {
                "backend": pool.backend_name,
                "allocated_mib": allocated >> 20,
                "seconds": round(time.monotonic() - started, 3),
            }
        },
    )
    return allocated
