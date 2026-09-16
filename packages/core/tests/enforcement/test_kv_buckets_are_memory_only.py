"""every NATS KV bucket in this repo is memory-backed; durability belongs to L3.

A thin shell over :mod:`threetears.enforcement.memory_only_kv`, which is where the walker lives
now that a second repo needs it: every consumer that opens a KV bucket adopts the same gate
rather than copying the scanner. The rule, its history and why it has no exemption mechanism are
in that package's docstring.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.memory_only_kv import MemoryOnlyKvConfig, run_memory_only_kv_enforcement

__all__: list[str] = []

_REPO_ROOT = Path(__file__).resolve().parents[4]


def test_no_kv_bucket_asks_for_file_storage() -> None:
    """the gate, over every package's source.

    :return: nothing
    :rtype: None
    """
    run_memory_only_kv_enforcement(MemoryOnlyKvConfig(repo_root=_REPO_ROOT))
