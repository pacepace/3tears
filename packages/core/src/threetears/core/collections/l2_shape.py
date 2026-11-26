"""the shape tag every collection L2 (NATS KV) entry carries, so a release never needs a cache clear.

**Why.** An L2 entry is a row as one release encoded it. A release that changes a table's shape -- a column added,
renamed or retyped, the key changed, the row epoch declared, the codec changed -- used to need the shared collections
bucket cleared at its first start (the hub's one-off v200 step), or a new pod would decode an older pod's entry as if
it were its own. A rolling deploy made that worse: old and new pods write the same keys for a while.

**What.** Every entry a collection writes opens with :func:`shape_header`: a NUL byte (no JSON row starts with one),
the word ``shape:``, and the 8-byte digest of the collection's shape, as 16 hex characters. A read compares that
prefix and nothing else. An entry whose prefix is not this collection's -- an older release's entry with no tag, or
another shape's -- is not this collection's row: the read treats it as a miss (:class:`L2ShapeMismatch`, a
:class:`~threetears.core.exceptions.CorruptCacheEntry`, so every read path that already falls through to L3 on a
corrupt entry does so here, keeping the key's revision so the refill from L3 replaces the entry at it), counts it on
``threetears_collection_l2_shape_mismatches_total{table}``, and never raises to its caller.

**What the tag covers** (:func:`shape_digest`'s inputs) is exactly what changes how an entry decodes or what it means:

- the declared columns' names, types and order (a vector column's width with its type);
- the row epoch kind (``plain``, ``stamped`` or ``versioned``) and the columns a row epoch adds (``epoch``,
  ``gone``) and the move columns (``moveable``);
- the key columns (and the stored key, the key and then the epoch for a versioned table);
- the codec version (:data:`L2_CODEC_VERSION`, and a hand-written collection's own
  :attr:`~threetears.core.collections.base.BaseCollection.l2_shape_version`).

Nothing else: indexes, comments, purposes, TTL policies, tenancy, eviction, foreign keys, nullability, server
defaults. None of them changes the entry, so a change to one must not empty every cache for nothing.

**Cost.** The digest is computed once: a :class:`~threetears.core.collections.schema_backed.TableSchema` computes its
own when it is built (at import), a collection its header once when it is first asked. A read compares one prefix
(``bytes.startswith``) and hands the codec the bytes after it; a write prepends it.

A leaf module: the collections import it, and it imports neither.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Final

from threetears.core.exceptions import CorruptCacheEntry
from threetears.observe.metrics import counter

__all__ = [
    "L2_CODEC_VERSION",
    "L2ShapeMismatch",
    "count_shape_mismatch",
    "shape_digest",
    "shape_header",
    "without_shape_header",
]

#: the version of the L2 envelope and codec every collection shares. Raise it when the framework changes how every
#: collection's entries are encoded (the envelope, the JSON codec's rendering of a type): every entry then reads as
#: another shape's, once, and is refilled from L3.
L2_CODEC_VERSION: Final = 1

#: how many bytes of the digest the tag keeps: 64 bits, so two shapes that differ collide once in 2**64
_SHAPE_DIGEST_BYTES: Final = 8

#: what every tagged entry opens with, before the digest. A NUL first, as the absent-marker's prefix, so no
#: JSON-serialised row can start with it and an untagged entry can never read as tagged
_HEADER_MAGIC: Final = b"\x00shape:"


class L2ShapeMismatch(CorruptCacheEntry):
    """an L2 entry that is not this collection's shape: an older release's (no tag), or another shape's.

    A :class:`CorruptCacheEntry`, so every read path treats it as one: a miss that keeps the key's revision, the
    row read from L3 and the entry replaced at that revision.
    """

    def __init__(self, table_name: str, value: bytes) -> None:
        super().__init__(table_name, "l2 shape tag", value[: len(_HEADER_MAGIC) + 2 * _SHAPE_DIGEST_BYTES])


def shape_digest(inputs: Mapping[str, Any]) -> bytes:
    """the 8-byte digest of a shape's declared inputs.

    :param inputs: what the shape is (JSON-able: names, type tags, tuples); key order is irrelevant
    :ptype inputs: Mapping[str, Any]
    :return: the digest
    :rtype: bytes
    """
    text = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(text.encode("utf-8"), digest_size=_SHAPE_DIGEST_BYTES).digest()


def shape_header(digest: bytes) -> bytes:
    """the prefix every entry of a shape opens with: the marker, then the digest as hex.

    Hex rather than the raw 8 bytes so the entry stays text after its NUL, as every other L2 value of a collection
    is: a tool that prints a value prints this one too.

    :param digest: the shape's digest (:func:`shape_digest`)
    :ptype digest: bytes
    :return: the header
    :rtype: bytes
    """
    return _HEADER_MAGIC + digest.hex().encode("ascii")


def without_shape_header(raw: bytes) -> bytes:
    """an entry's codec bytes, whatever shape's header it carries: the bytes after one, or all of an untagged entry.

    For a collection whose L2 is its record, where an entry of another shape is still a committed row.

    :param raw: the stored bytes
    :ptype raw: bytes
    :return: the bytes the codec reads
    :rtype: bytes
    """
    if raw.startswith(_HEADER_MAGIC):
        return raw[len(_HEADER_MAGIC) + 2 * _SHAPE_DIGEST_BYTES :]
    return raw


def count_shape_mismatch(table: str) -> None:
    """count one L2 entry a read of ``table`` ignored because its shape tag was not the table's.

    :param table: the table
    :ptype table: str
    """
    # fetched where it is moved, never held at import: a fresh import of the package would register it twice
    counter(
        "threetears_collection_l2_shape_mismatches_total",
        description=(
            "L2 entries a collection read ignored because their shape tag was not its own (an older release's entry, "
            "or another shape's): each read as a miss, the row read from L3 and the entry replaced"
        ),
        label_names=("table",),
    ).labels(table=table).inc()
