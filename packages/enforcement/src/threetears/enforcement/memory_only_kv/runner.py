"""pytest-friendly orchestration for the memory-only KV gate.

One :func:`run_memory_only_kv_enforcement` entry point so each consumer's shell is a few lines.
No mode env var and no exemptions file, unlike the other domains here: this gate is strict by
construction, because an exemption is exactly how the buckets it removed stayed for months.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.memory_only_kv.config import MemoryOnlyKvConfig
from threetears.enforcement.memory_only_kv.walkers import scan_for_file_backed_kv

__all__ = ["run_memory_only_kv_enforcement", "source_files"]


def source_files(config: MemoryOnlyKvConfig) -> list[Path]:
    """every module this repo's globs resolve to.

    :param config: per-repo config
    :ptype config: MemoryOnlyKvConfig
    :return: the modules, de-duplicated
    :rtype: list[Path]
    """
    found: set[Path] = set()
    for pattern in config.source_globs:
        found.update(path for path in config.repo_root.glob(pattern) if path.is_file())
    return sorted(found)


def run_memory_only_kv_enforcement(config: MemoryOnlyKvConfig) -> None:
    """fail when any module opens a KV bucket with file storage.

    Also fails when the globs resolve to nothing: a gate that scanned no files passes, and a
    consumer whose layout this config does not describe would read green forever. That is not a
    hypothetical -- the version of this gate that lived in one repo's tests missed every nested
    agent package for exactly that reason.

    :param config: per-repo config
    :ptype config: MemoryOnlyKvConfig
    :return: nothing
    :rtype: None
    :raises pytest.fail.Exception: when a file-backed bucket is opened, or nothing was scanned
    """
    paths = source_files(config)
    if not paths:
        pytest.fail(
            f"memory-only KV: no source files matched {list(config.source_globs)} under "
            f"{config.repo_root} -- the gate would pass by scanning nothing. Point "
            f"MemoryOnlyKvConfig.source_globs at this repo's source."
        )

    violations = [
        f"{call.path.relative_to(config.repo_root).as_posix()}:{call.lineno} opens a KV bucket "
        f"with storage='file' via {call.callee}()"
        for call in scan_for_file_backed_kv(paths)
    ]
    if violations:
        listed = "\n  ".join(violations)
        pytest.fail(
            f"NATS is L2 and these ask it to be L3:\n  {listed}\n\n"
            f"Put the state in a BaseCollection, which composes L1, L2 and L3 rather than making "
            f"anyone choose -- threetears.core.coordination.tables is the worked example. There is "
            f"no exemption file: a file-backed bucket cannot be exempted, only designed out."
        )
