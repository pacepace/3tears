"""How many CPUs this process may use: its container's CPU quota, not the machine's cores.

In a pod, ``os.cpu_count()`` answers the node's cores (32 on a large node) while the pod's CPU limit
is a cgroup quota (4, say). A pool sized from the cores runs 32 threads on a 4-CPU quota: the cgroup
spends its quota early in each 100 ms period and the kernel then freezes every thread in it, the
event loop included, for the rest of the period. That was the reports tool pod's 50-100 ms loop
stalls during a full LSF pull (401 throttled periods, 53 s throttled, on a 4-CPU limit over 32
cores).

:func:`effective_cpu_count` reads the quota -- cgroup v2 ``cpu.max`` (``"<quota> <period>"``, or
``"max <period>"`` for none), else cgroup v1 ``cpu.cfs_quota_us`` / ``cpu.cfs_period_us`` (``-1``
for none) -- rounded up to a whole CPU and never more than the machine's cores; with no quota, the
machine's cores. :func:`bound_native_threads` sizes pyarrow's thread pools to it, and
:func:`default_executor_workers` the asyncio default executor, as a tool pod's boot does.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Final

from threetears.observe import get_logger

try:
    import pyarrow

    _HAS_PYARROW = True
except ImportError:  # NOSILENT: pyarrow is optional (the snapshot extra); without it there is no Arrow pool to size
    _HAS_PYARROW = False

__all__ = [
    "bound_native_threads",
    "default_executor_workers",
    "effective_cpu_count",
]

log = get_logger(__name__)

#: where the container's cgroup filesystem is mounted
_CGROUP_ROOT: Final = Path("/sys/fs/cgroup")

#: the most threads pyarrow's I/O pool runs: its own default, kept whatever the quota
_MAX_ARROW_IO_THREADS: Final = 8

#: the cgroup v1 directories the CPU controller is mounted under, in the order they are tried
_V1_CPU_DIRS: Final = ("cpu", "cpu,cpuacct", "cpuacct,cpu")


def _v2_quota(cpu_max: Path) -> float | None:
    """the CPUs a cgroup v2 ``cpu.max`` allows, or ``None`` when it sets no quota.

    :param cpu_max: the ``cpu.max`` file
    :ptype cpu_max: Path
    :return: ``1.5`` for ``"150000 100000"``; ``None`` for ``"max 100000"``
    :rtype: float | None
    :raises OSError: when the file cannot be read
    :raises ValueError: when it does not hold a quota and a period
    """
    fields = cpu_max.read_text().split()
    if len(fields) != 2:
        raise ValueError(f"cpu.max holds {len(fields)} fields, not a quota and a period")
    limit, period = fields
    return None if limit == "max" else _ratio(int(limit), int(period))


def _v1_quota(cpu_dir: Path) -> float | None:
    """the CPUs a cgroup v1 CPU controller allows, or ``None`` when it sets no quota.

    :param cpu_dir: the controller's directory, holding ``cpu.cfs_quota_us`` and ``cpu.cfs_period_us``
    :ptype cpu_dir: Path
    :return: ``2.0`` for a quota of 200000 over a period of 100000; ``None`` for a quota of ``-1``
    :rtype: float | None
    :raises OSError: when a file cannot be read
    :raises ValueError: when a file does not hold an integer
    """
    limit_us = int((cpu_dir / "cpu.cfs_quota_us").read_text().strip())
    period_us = int((cpu_dir / "cpu.cfs_period_us").read_text().strip())
    return None if limit_us <= 0 else _ratio(limit_us, period_us)


def _ratio(limit: int, period: int) -> float:
    """a quota over its period, in CPUs.

    :param limit: the CPU time allowed per period
    :ptype limit: int
    :param period: the period
    :ptype period: int
    :return: the quota in CPUs
    :rtype: float
    :raises ValueError: when either is not positive
    """
    if limit <= 0 or period <= 0:
        raise ValueError(f"a CPU quota of {limit} over a period of {period} is not a quota")
    return limit / period


def _quota(root: Path) -> float | None:
    """the cgroup's CPU quota in CPUs, or ``None`` when it sets none or none can be read.

    :param root: the cgroup filesystem's root
    :ptype root: Path
    :return: the quota, or ``None``
    :rtype: float | None
    """
    quota: float | None = None
    try:
        cpu_max = root / "cpu.max"
        if cpu_max.is_file():
            quota = _v2_quota(cpu_max)
        else:
            v1 = next((root / d for d in _V1_CPU_DIRS if (root / d / "cpu.cfs_quota_us").is_file()), None)
            quota = _v1_quota(v1) if v1 is not None else None
    except (OSError, ValueError) as exc:
        # NOSILENT: an unreadable quota is no quota; the machine's cores bound the pools instead
        log.warning(
            "the CPU quota could not be read; sizing pools by the machine's cores",
            extra={"extra_data": {"root": str(root), "error": f"{type(exc).__name__}: {exc}"}},
        )
        quota = None
    return quota


def effective_cpu_count(root: Path = _CGROUP_ROOT, *, cpu_count: Callable[[], int | None] = os.cpu_count) -> int:
    """the CPUs this process may use: its cgroup's quota rounded up, else the machine's cores.

    :param root: the cgroup filesystem's root (a test's stand-in)
    :ptype root: Path
    :param cpu_count: the machine's cores (a test's stand-in)
    :ptype cpu_count: Callable[[], int | None]
    :return: at least 1, and never more than the machine's cores
    :rtype: int
    """
    cores = max(1, cpu_count() or 1)
    quota = _quota(root)
    return cores if quota is None else max(1, min(cores, math.ceil(quota)))


def default_executor_workers(cpus: int) -> int:
    """the asyncio default executor's size for ``cpus``: the standard library's own rule, on the quota.

    ``ThreadPoolExecutor`` defaults to ``min(32, os.cpu_count() + 4)``, which is 32 on a large node
    whatever the pod's limit; this is the same rule over the CPUs the process may use.

    :param cpus: the CPUs the process may use (:func:`effective_cpu_count`)
    :ptype cpus: int
    :return: the executor's ``max_workers``
    :rtype: int
    """
    return min(32, max(1, cpus) + 4)


def _arrow_io_threads(cpus: int) -> int:
    """the size of pyarrow's I/O pool for ``cpus``: twice the CPUs, at least 2, at most :data:`_MAX_ARROW_IO_THREADS`.

    I/O threads mostly wait on the network, so they may outnumber the CPUs; pyarrow's own default
    (8) is the ceiling.

    :param cpus: the CPUs the process may use (:func:`effective_cpu_count`)
    :ptype cpus: int
    :return: the I/O pool's size
    :rtype: int
    """
    return min(_MAX_ARROW_IO_THREADS, max(2, 2 * cpus))


def bound_native_threads(cpus: int) -> bool:
    """size pyarrow's CPU pool to ``cpus`` and its I/O pool to :func:`_arrow_io_threads`, when pyarrow is installed.

    pyarrow sizes its CPU pool from the machine's cores, so on a 32-core node a 4-CPU pod runs 32
    compute threads and is throttled for most of every period.

    :param cpus: the CPUs the process may use (:func:`effective_cpu_count`)
    :ptype cpus: int
    :return: whether pyarrow was there to size
    :rtype: bool
    """
    if _HAS_PYARROW:
        pyarrow.set_cpu_count(max(1, cpus))
        pyarrow.set_io_thread_count(_arrow_io_threads(cpus))
    return _HAS_PYARROW
