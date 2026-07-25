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

This builder lives in the dependency-free contract (beside
:class:`~threetears.media.contracts.protocols.ObjectHandle`) rather than the
S3 impl package: the layout is a *contract* every producer encodes, and a
producing tool must be able to build a key without inheriting the aioboto3
client tree. The impl package re-exports it for back-compat.
"""

from __future__ import annotations

import re
from datetime import datetime
from uuid import UUID

__all__ = ["SHARED_PREFIX", "build_object_key", "sanitize_segment"]

#: anything outside the safe key-segment alphabet collapses to a hyphen.
_UNSAFE = re.compile(r"[^a-z0-9-]+")

#: fallback leaf when no usable filename is supplied.
_DEFAULT_FILENAME = "object"

#: isolation prefix for objects owned by the platform rather than by one
#: tenant. the scheme leads with ``customer_id`` because that segment IS the
#: isolation boundary bucket policy grants on, so a platform-shared object
#: still needs *a* leading segment to grant against -- it just cannot be a
#: customer. see :func:`build_object_key`.
SHARED_PREFIX = "shared"


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
    customer_id: UUID | None,
    scope: str,
    category: str,
    object_id: UUID | None = None,
    created: datetime | None = None,
    filename: str | None = None,
    path: str | None = None,
) -> str:
    """Build the scope-first object key (locked design section 8).

    Two shapes, distinguished by whether the object is tenant-owned and
    whether its address is opaque or meaningful:

    - **tenant-owned, opaque address** (the original and still the common
      case): ``customer_id`` plus ``object_id`` and ``created`` produce
      ``<customer_id>/<scope>/<category>/<YYYY>/<MM>/<DD>/<object_id>/<filename>``.
    - **deterministic address** (``path`` supplied): the caller's own path
      replaces the date partition, object-id folder and filename. Required
      when a reader must derive the key from a request without a lookup --
      a CDN fetching a map tile cannot consult a database to translate
      ``z/x/y`` into a UUID.

    ``customer_id=None`` addresses a platform-shared object under
    :data:`SHARED_PREFIX` instead of a tenant. That is not a loophole in the
    tenant isolation: ``platform.datasources.customer_id`` is itself nullable
    with NULL meaning platform-shared, so objects derived from such a row
    have no tenant to be scoped to, and forcing one would fork a single
    shared artifact into one copy per customer.

    :param customer_id: verified tenant UUID, or ``None`` for a
        platform-shared object
    :ptype customer_id: UUID | None
    :param scope: owning-context label, e.g. ``engagement-<id>`` (sanitized)
    :ptype scope: str
    :param category: object kind, e.g. ``reports`` / ``evidence`` (sanitized)
    :ptype category: str
    :param object_id: unique object UUID (UUIDv7, time-ordered); required
        unless ``path`` is supplied
    :ptype object_id: UUID | None
    :param created: UTC creation timestamp; drives the ``YYYY/MM/DD``
        partition; required unless ``path`` is supplied
    :ptype created: datetime | None
    :param filename: original filename + extension; ``None`` -> ``object``
    :ptype filename: str | None
    :param path: caller-supplied deterministic tail, replacing the date /
        object-id / filename segments. Each component is sanitized
        individually so separators survive, and the final component keeps
        its extension
    :ptype path: str | None
    :return: the full object key
    :rtype: str
    :raises ValueError: when neither ``path`` nor both of
        ``object_id``/``created`` are supplied
    """
    owner = str(customer_id) if customer_id is not None else SHARED_PREFIX
    head = f"{owner}/{sanitize_segment(scope)}/{sanitize_segment(category)}"
    if path is not None:
        parts = [part for part in path.split("/") if part]
        if not parts:
            raise ValueError("build_object_key: path= must contain at least one component")
        # the last component is a filename: sanitized as a stem plus extension
        # so ``98.mvt`` does not collapse to ``98-mvt`` and lose the type a
        # reader (and a CDN's content negotiation) depends on.
        tail = "/".join([*(sanitize_segment(part) for part in parts[:-1]), _sanitize_filename(parts[-1])])
        return f"{head}/{tail}"
    if object_id is None or created is None:
        raise ValueError("build_object_key requires either path=, or both object_id= and created=")
    leaf = _sanitize_filename(filename)
    return f"{head}/{created:%Y/%m/%d}/{object_id}/{leaf}"
