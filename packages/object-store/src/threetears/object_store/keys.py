"""Tenant-scoped object-key builder (the platform's scope-first scheme).

The locked layout (scope-and-objects-design.md section 8) is, under one
bucket per environment::

    <customer_id>/<scope>/<category>/<YYYY>/<MM>/<DD>/<object_id>/<filename>

``customer_id`` is the tenant-isolation prefix; ``scope`` is a
framework-general owning-context label the producer supplies
(``engagement-<id>`` / ``conversation-<id>`` / ``agent-<slug>``);
``object_id`` is a UUIDv7 (unique + time-ordered) folder so derivatives
co-locate; ``filename`` keeps the original name + extension for human
readability and correct download naming.
"""

from __future__ import annotations

import re
from datetime import datetime
from uuid import UUID

__all__ = ["build_object_key", "sanitize_segment"]

#: anything outside the safe key-segment alphabet collapses to a hyphen.
_UNSAFE = re.compile(r"[^a-z0-9-]+")

#: fallback leaf when no usable filename is supplied.
_DEFAULT_FILENAME = "object"


def sanitize_segment(value: str) -> str:
    """Lower-case and collapse ``value`` to the ``[a-z0-9-]`` key alphabet.

    :param value: raw segment (scope label, category, filename stem)
    :ptype value: str
    :return: sanitized segment safe as one S3 key path component; falls
        back to ``object`` when nothing usable remains
    :rtype: str
    """
    cleaned = _UNSAFE.sub("-", value.strip().lower()).strip("-")
    return cleaned or _DEFAULT_FILENAME


def _sanitize_filename(filename: str | None) -> str:
    """Sanitize a filename's stem while preserving its extension.

    :param filename: original filename (may carry an extension); ``None``
        or empty yields the ``object`` fallback
    :ptype filename: str | None
    :return: readable, key-safe ``<stem>.<ext>`` (or ``<stem>``)
    :rtype: str
    """
    stem, dot, ext = (filename or "").rpartition(".")
    if not dot:
        result = sanitize_segment(filename or "")
    else:
        safe_stem = sanitize_segment(stem)
        safe_ext = _UNSAFE.sub("", ext.lower())
        result = f"{safe_stem}.{safe_ext}" if safe_ext else safe_stem
    return result


def build_object_key(
    *,
    customer_id: UUID,
    scope: str,
    category: str,
    object_id: UUID,
    created: datetime,
    filename: str | None = None,
) -> str:
    """Build the scope-first object key (locked design section 8).

    :param customer_id: verified tenant UUID; the isolation prefix
    :ptype customer_id: UUID
    :param scope: owning-context label, e.g. ``engagement-<id>`` (sanitized)
    :ptype scope: str
    :param category: object kind, e.g. ``reports`` / ``evidence`` (sanitized)
    :ptype category: str
    :param object_id: unique object UUID (UUIDv7, time-ordered)
    :ptype object_id: UUID
    :param created: UTC creation timestamp; drives the ``YYYY/MM/DD`` partition
    :ptype created: datetime
    :param filename: original filename + extension; ``None`` -> ``object``
    :ptype filename: str | None
    :return: the full tenant-scoped object key
    :rtype: str
    """
    leaf = _sanitize_filename(filename)
    return f"{customer_id}/{sanitize_segment(scope)}/{sanitize_segment(category)}/{created:%Y/%m/%d}/{object_id}/{leaf}"
