"""NATS KV buckets are memory-backed; durability belongs to L3."""

from __future__ import annotations

from threetears.enforcement.kv_memory_only.walkers import (
    DURABLE_STORAGE,
    KV_OPENING_CALLS,
    STORAGE_KEYWORD,
    file_backed_kv_calls,
)

__all__ = [
    "DURABLE_STORAGE",
    "KV_OPENING_CALLS",
    "STORAGE_KEYWORD",
    "file_backed_kv_calls",
]
